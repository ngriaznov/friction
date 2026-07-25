"""Build a sentence-level feature-parity fixture.

Each fixture entry carries the sentence text, the reference dependency parse,
and the exact per-feature COUNTS the reference extractor produced. Shipping
the parse alongside the counts is deliberate: a Rust port can then be checked
against the reference features while being fed the reference parse, which
separates "the extractor is wrong" from "the parser is wrong". Without that
separation a parity failure says only that something broke.

Selection is stratified to cover every feature, including the rare ones a
random sample would miss.
"""
import json
import pathlib
import sys

sys.path.insert(0, "/Users/nikitagriaznov/Documents/Work/POC/friction/docs/research/regvec")

import biber  # noqa: E402
import spacy  # noqa: E402

ROOT = pathlib.Path("/Users/nikitagriaznov/Documents/Work/POC/friction")
NLP = biber.NLP

# Relations the register module reads, plus the ones the transducers license on.
KEEP_DEPS = {
    "ROOT", "acl", "advcl", "agent", "amod", "aux", "auxpass", "cc", "ccomp",
    "conj", "csubj", "det", "dobj", "mark", "nsubj", "nsubjpass", "pobj",
    "prep", "xcomp", "punct",
}


def resolve(rec):
    for base in ("human", "llm", "quarantine"):
        p = ROOT / "corpus" / base / rec["genre"] / f"{rec['id']}.md"
        if p.exists():
            return p
    return None


def counts_for(doc):
    """Per-feature COUNTS (not rates) for one spaCy doc."""
    return {
        "present_participial": len(biber.present_participials(doc)),
        "nominalization": sum(1 for t in doc if biber.is_nominalization(t)),
        "phrasal_coord": len(biber.phrasal_coords(doc)),
        "that_subj": len(biber.that_subject_clauses(doc)),
        "agentless_passive": len(biber.agentless_passives(doc)),
        "clausal_coord": len(biber.clausal_coords(doc)),
        "attributive_adj": sum(1 for t in doc if t.dep_ == "amod"),
        "past_part_postnom": len(biber.past_part_postnominal(doc)),
        "adverbs": sum(1 for t in doc if t.pos_ == "ADV"),
        "downtoners": sum(1 for t in doc if t.lower_ in biber.DOWNTONERS),
        "demonstratives": sum(
            1 for t in doc
            if t.lower_ in ("this", "that", "these", "those")
            and t.dep_ not in ("mark",) and t.pos_ in ("DET", "PRON")
        ),
        "prepositions": sum(1 for t in doc if t.pos_ == "ADP"),
        "infinitives": len(biber.infinitives(doc)),
        "nouns": sum(1 for t in doc if t.pos_ in ("NOUN", "PROPN")),
        "hedges": sum(1 for t in doc if t.lower_ in biber.HEDGES),
        "first_person": sum(
            1 for t in doc
            if t.lower_ in ("i", "we", "my", "our", "us", "me")
        ),
        "demon_sent_initial": sum(
            1 for t in doc
            if t.is_sent_start and t.lower_ in ("this", "these", "that")
        ),
    }


def sent_entry(sent, src_id, idx):
    text = sent.text.strip()
    sub = NLP(text)          # re-parse in isolation so offsets are local
    if len(sub) < 4 or len(sub) > 60:
        return None
    if not text.isascii():
        return None
    # Re-parsing in isolation can re-segment: spaCy sometimes splits what the
    # document-level parse called one sentence into two, which yields two
    # roots and inflates the sentence-initial feature. Such an entry is not a
    # sentence-level fixture, so drop it rather than record it.
    if sum(1 for _ in sub.sents) != 1:
        return None
    toks = [
        [
            t.text,
            t.tag_,
            -1 if t.dep_ == "ROOT" else t.head.i,
            t.dep_ if t.dep_ in KEEP_DEPS else "other",
        ]
        for t in sub
    ]
    return {
        "id": f"{src_id}:{idx}",
        "text": text,
        "tokens": toks,
        "counts": counts_for(sub),
        "words": sum(1 for t in sub if not t.is_punct and not t.is_space),
    }


def main():
    manifest = [
        json.loads(l) for l in
        (ROOT / "corpus" / "manifest.jsonl").read_text().splitlines() if l.strip()
    ]
    # Train split only. Never dev, never holdout.
    recs = [
        r for r in manifest
        if r.get("split") == "train" and r.get("genre") in ("docs", "readme")
    ]
    recs.sort(key=lambda r: r["id"])
    print(f"{len(recs)} docs+readme train documents", file=sys.stderr)

    pool = []
    for r in recs:
        p = resolve(r)
        if p is None:
            continue
        doc = NLP(p.read_text(errors="replace")[:20000])
        for i, s in enumerate(doc.sents):
            e = sent_entry(s, r["id"], i)
            if e is not None:
                e["_class"] = r["class"]
                pool.append(e)
    print(f"{len(pool)} candidate sentences", file=sys.stderr)

    feats = list(next(iter(pool))["counts"].keys())
    chosen, seen = [], set()

    # Stratify: for each feature, and for each dependency relation, take the
    # first few sentences that exercise it. Rare items would otherwise be
    # absent entirely, and they are exactly the ones worth pinning.
    def take(pred, want, tag):
        got = 0
        for e in pool:
            if got >= want:
                break
            if e["id"] in seen or not pred(e):
                continue
            seen.add(e["id"])
            chosen.append(e)
            got += 1
        print(f"  {tag}: {got}", file=sys.stderr)

    for f in feats:
        take(lambda e, f=f: e["counts"][f] > 0, 6, f"feature {f}")
    for d in sorted(KEEP_DEPS):
        take(lambda e, d=d: any(t[3] == d for t in e["tokens"]), 4, f"dep {d}")
    # A plain baseline: sentences where nothing marked fires at all.
    take(lambda e: all(e["counts"][f] == 0 for f in
                       ("present_participial", "nominalization",
                        "agentless_passive")), 10, "baseline")

    chosen.sort(key=lambda e: e["id"])
    for e in chosen:
        e.pop("_class", None)

    cov = {f: sum(1 for e in chosen if e["counts"][f] > 0) for f in feats}
    depcov = {}
    for e in chosen:
        for t in e["tokens"]:
            depcov[t[3]] = depcov.get(t[3], 0) + 1

    out = {
        "comment": (
            "Sentence-level feature-parity fixture. Each entry carries the "
            "reference dependency parse and the exact per-feature counts the "
            "reference extractor produced from it. A port is checked by "
            "feeding it the parse in this file and comparing counts, which "
            "isolates extractor errors from parser errors. Counts, not rates: "
            "rates depend on a word total and are derived. Drawn from the "
            "train split only, docs and readme genres. Selection is "
            "stratified over features and relations so rare items appear. "
            "Each token is [surface, penn_pos, head_index, relation]; "
            "head_index -1 marks the root."
        ),
        "sentences": chosen,
    }
    dest = ROOT / "docs" / "research" / "regvec" / "feature_parity.json"
    dest.write_text(
        json.dumps(out, sort_keys=True, ensure_ascii=True,
                   separators=(",", ":")) + "\n"
    )
    print(f"\n{len(chosen)} sentences -> {dest}", file=sys.stderr)
    print("feature coverage:", json.dumps(cov, indent=1), file=sys.stderr)
    print("relation coverage:", json.dumps(depcov, sort_keys=True, indent=1), file=sys.stderr)


if __name__ == "__main__":
    main()
