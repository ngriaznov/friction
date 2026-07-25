#!/usr/bin/env python3
"""Drafts and corrects the dependency-parse gold file from friction's own
sentence stream.

Reads the newline-delimited JSON `crates/friction-nlp/examples/dump_sentences.rs`
produces on stdin (one object per candidate sentence: friction's own
tokenization and part-of-speech tags, byte offsets relative to the
sentence's own text) and writes a CoNLL-U-shaped gold file.

Pipeline, in order:

1. Parse every sentence's `text` with spaCy `en_core_web_sm`, offline,
   as a one-time drafting aid. spaCy is MIT-licensed; nothing from its
   model weights or output format is redistributed here beyond the
   relation labels and head indices this script derives from its parse.
2. Align spaCy's tokenization to friction's own, in both directions:
   merge spaCy's `X`,`-`,`Y` hyphen splits and `can`,`not` `cannot` split
   back into a single token when the merge exactly reconstructs one
   friction token, and project one spaCy token's head/relation onto a
   run of several friction tokens when friction's own tokenizer is the
   one that over-split (mixed letter/digit runs, a number fused with
   trailing punctuation). Sentences whose tokenization still disagrees
   after both are dropped.
3. Drop non-projective trees: arc-eager (the transition system this gold
   file trains) can only derive projective ones.
4. Apply closed-class relation corrections: `det` on determiners, `aux`
   on modals, `mark` on subordinating conjunctions introducing a clause.
   These are facts about the token's own part of speech, not the
   parser's discretion.
5. Map spaCy's relation labels to friction's target set, collapsing
   everything else to `other`.
6. Take a fixed stride over the surviving pool (sorted by document id,
   then sentence index) to land near a target sentence count.
7. Split train/test by document (not by sentence), by a deterministic
   hash of the document id.
8. Write the CoNLL-U-shaped gold file, sentences in the same stable
   sorted order, so the file is byte-reproducible from the input stream.

Every step reports its own numbers to stderr; nothing here claims to be
hand-annotated gold — see `crates/friction-nlp/weights/NOTICE.md` for the
honest framing this drafting process gets before it ships.

```text
cargo run -p friction-nlp --example dump_sentences -- corpus \
    | tools/gold_dep/venv/bin/python tools/gold_dep/draft_gold.py \
        --out crates/friction-nlp/weights/gold_dep_en.conllu
```
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field

import spacy

# The exact ClearNLP-scheme relation names the register module's
# transducers and detectors match on (see
# `docs/superpowers/specs/2026-07-25-register-parser-design.md`, "Label
# set"). Everything spaCy emits outside this set collapses to `other`.
TARGET_RELATIONS = frozenset(
    {
        "root",
        "acl",
        "advcl",
        "agent",
        "amod",
        "aux",
        "auxpass",
        "cc",
        "ccomp",
        "conj",
        "csubj",
        "det",
        "dobj",
        "mark",
        "nsubj",
        "nsubjpass",
        "pobj",
        "prep",
        "xcomp",
        "punct",
    }
)

# Friction's own Penn tags that unambiguously mark a determiner, wherever
# they occur — see `crates/friction-nlp/weights/NOTICE.md` for the same
# discipline applied to the POS gold file.
DETERMINER_TAGS = frozenset({"DT", "PDT"})
MODAL_TAG = "MD"

# spaCy's own universal-POS category for a subordinating conjunction.
# Friction's own Penn tagset does not distinguish this from a preposition
# (both are `IN`), so this one correction reads spaCy's finer-grained tag
# rather than friction's — the only one of the three that does, and the
# reason it is documented separately in `weights/NOTICE.md`.
SUBORDINATOR_POS = "SCONJ"

DEFAULT_TARGET_SENTENCES = 4000
TRAIN_SPLIT_PERCENT = 80


@dataclass
class FrictionToken:
    surface: str
    pos: str
    start: int
    end: int


@dataclass
class InputSentence:
    doc_id: str
    genre: str
    sent_index: int
    text: str
    tokens: list[FrictionToken]


@dataclass
class GoldToken:
    surface: str
    pos: str
    head: int  # 1-based index of head token, or 0 for root
    relation: str


@dataclass
class GoldSentence:
    doc_id: str
    genre: str
    sent_index: int
    text: str
    tokens: list[GoldToken] = field(default_factory=list)


def read_input_sentences(stream) -> list[InputSentence]:
    """Parses the dumper's newline-delimited JSON stream."""
    sentences: list[InputSentence] = []
    for lineno, line in enumerate(stream, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as err:
            raise SystemExit(f"stdin:{lineno}: invalid JSON: {err}") from err
        tokens = [
            FrictionToken(t["surface"], t["pos"], t["start"], t["end"])
            for t in record["tokens"]
        ]
        sentences.append(
            InputSentence(
                record["doc_id"],
                record["genre"],
                record["sent_index"],
                record["text"],
                tokens,
            )
        )
    return sentences


def naive_alignment_matches(friction_tokens: list[FrictionToken], doc) -> bool:
    """True if friction's tokens and spaCy's raw tokens already agree
    span-for-span, with no merge normalization applied — the "before
    normalization" measurement.
    """
    if len(friction_tokens) != len(doc):
        return False
    return all(
        ft.start == tok.idx and ft.end == tok.idx + len(tok.text)
        for ft, tok in zip(friction_tokens, doc)
    )


class AlignmentFailure(Exception):
    """A sentence's tokenization could not be reconciled with spaCy's,
    even after merge and split-projection normalization."""


@dataclass
class AlignGroup:
    """One unit of the alignment between friction's tokens and spaCy's:
    the friction token index(es) and spaCy token index(es) that jointly
    cover one contiguous span of sentence text. Exactly one side has more
    than one index — `kind` says which, `"single"` meaning neither does.
    """

    friction: list[int]
    spacy: list[int]
    kind: str  # "single" | "merge" (spaCy over-split) | "split" (friction over-split)


def align_tokens(
    friction_tokens: list[FrictionToken], doc
) -> tuple[list[AlignGroup], dict[str, int], dict[str, int]]:
    """Aligns friction tokens to spaCy tokens in both directions.

    spaCy and friction tokenize differently in two distinct, opposite
    ways: spaCy sometimes splits what friction keeps as one token
    (`high-quality`, `cannot`) — handled by merging spaCy tokens back
    together, restricted to the two normalization patterns this file
    implements — and friction sometimes splits what spaCy keeps as one
    token (friction's tokenizer breaks at every letter/digit boundary
    and folds trailing `.`/`,` into a number, so `S2O` becomes three
    friction tokens and `2017,` becomes one where spaCy keeps `2017`
    and `,` separate) — handled by *projecting* one spaCy token's
    head/relation onto a run of friction tokens, unconditionally
    (mechanical, not a judgement call: see `process_group` below).

    Returns `(groups, merge_counts, split_counts)`. Raises
    [`AlignmentFailure`] if any span cannot be reconciled by either
    direction, or tokens are left over on either side.

    spaCy, unlike friction's own tokenizer (`tokenize()` in
    `tag_perceptron.rs` never emits a `Whitespace`-kind token), sometimes
    emits a standalone whitespace token — reliably for a literal embedded
    newline in a soft-wrapped source paragraph. Such a token carries no
    friction counterpart to align against and is never any other token's
    head (verified: no non-space spaCy token in this corpus has a `head`
    that is itself a space token), so it is filtered out up front rather
    than walked over — this is restoring an apples-to-apples token list,
    not a normalization choice.
    """
    spacy_indices = [i for i, t in enumerate(doc) if not t.is_space]

    groups: list[AlignGroup] = []
    merge_counts = {"hyphen": 0, "cannot": 0}
    split_counts = {"groups": 0, "fragments": 0}
    fi = 0
    si = 0
    n_friction = len(friction_tokens)
    n_spacy = len(spacy_indices)

    while fi < n_friction and si < n_spacy:
        ft = friction_tokens[fi]
        st = doc[spacy_indices[si]]
        s_start = st.idx
        s_end = st.idx + len(st.text)

        if ft.start != s_start:
            raise AlignmentFailure(
                f"span start mismatch: friction {ft.start} vs spaCy {s_start}"
            )

        if ft.end == s_end:
            groups.append(AlignGroup([fi], [spacy_indices[si]], "single"))
            fi += 1
            si += 1
            continue

        if ft.end < s_end:
            # friction's token is shorter: this one spaCy token covers a
            # run of several friction tokens — project it.
            fj = fi
            end = friction_tokens[fj].end
            while end < s_end:
                fj += 1
                if fj >= n_friction:
                    raise AlignmentFailure("ran out of friction tokens mid-spaCy-token")
                end = friction_tokens[fj].end
            if end != s_end:
                raise AlignmentFailure(
                    f"span end mismatch (split): spaCy {s_end} vs accumulated friction {end}"
                )
            friction_group = list(range(fi, fj + 1))
            groups.append(AlignGroup(friction_group, [spacy_indices[si]], "split"))
            split_counts["groups"] += 1
            split_counts["fragments"] += len(friction_group) - 1
            fi = fj + 1
            si += 1
            continue

        # ft.end > s_end: friction's token is longer — a run of several
        # spaCy tokens make up this one friction token. Restricted to the
        # two normalization patterns this file implements (unlike the
        # split direction above, which is unconditional).
        sj = si
        end = doc[spacy_indices[sj]].idx + len(doc[spacy_indices[sj]].text)
        while end < ft.end:
            sj += 1
            if sj >= n_spacy:
                raise AlignmentFailure("ran out of spaCy tokens mid-friction-token")
            end = doc[spacy_indices[sj]].idx + len(doc[spacy_indices[sj]].text)
        if end != ft.end:
            raise AlignmentFailure(
                f"span end mismatch (merge): friction {ft.end} vs accumulated spaCy {end}"
            )

        spacy_group = [spacy_indices[k] for k in range(si, sj + 1)]
        if len(spacy_group) == 3 and doc[spacy_group[1]].text == "-":
            merge_counts["hyphen"] += 1
        elif (
            len(spacy_group) == 2
            and doc[spacy_group[0]].text.lower() == "can"
            and doc[spacy_group[1]].text.lower() == "not"
            and ft.surface.lower() == "cannot"
        ):
            merge_counts["cannot"] += 1
        else:
            texts = [doc[k].text for k in spacy_group]
            raise AlignmentFailure(
                f"unsupported {len(spacy_group)}-token merge for friction token "
                f"{ft.surface!r}: spaCy tokens {texts!r}"
            )

        groups.append(AlignGroup([fi], spacy_group, "merge"))
        fi += 1
        si = sj + 1

    if fi != n_friction or si != n_spacy:
        raise AlignmentFailure("tokens remain unconsumed on one side after alignment")

    return groups, merge_counts, split_counts


def build_spacy_index_to_friction_index(groups: list[AlignGroup]) -> dict[int, int]:
    """Maps every spaCy token index to the friction token index any
    *external* arc pointing at it should be redirected to: the merged
    token for a merge group, and the head fragment (first friction
    token) for a split group.
    """
    mapping: dict[int, int] = {}
    for group in groups:
        target = group.friction[0]
        for spacy_idx in group.spacy:
            mapping[spacy_idx] = target
    return mapping


@dataclass
class CorrectionCounts:
    det: int = 0
    aux: int = 0
    mark: int = 0


def relation_for_token(
    friction_pos: str, spacy_dep: str, spacy_pos_universal: str, corrections: CorrectionCounts
) -> str:
    """Maps `spacy_dep` to the target relation set and applies the three
    closed-class overrides, counting every actual change (a change from
    what plain label-mapping alone would have produced) into `corrections`.
    """
    raw = "root" if spacy_dep.upper() == "ROOT" else spacy_dep.lower()
    mapped = raw if raw in TARGET_RELATIONS else "other"

    if friction_pos in DETERMINER_TAGS:
        final = "det"
    elif friction_pos == MODAL_TAG:
        final = "aux"
    elif spacy_pos_universal == SUBORDINATOR_POS:
        final = "mark"
    else:
        final = mapped

    if final != mapped:
        if final == "det":
            corrections.det += 1
        elif final == "aux":
            corrections.aux += 1
        elif final == "mark":
            corrections.mark += 1

    return final


def is_projective(heads: list[int]) -> bool:
    """`heads[i]` is the 1-based head of token `i+1`, or 0 for the root.
    Projective iff, for every real (non-root) arc head->dep, every token
    strictly between them (by linear position) is a descendant of head.
    """
    n = len(heads)

    def is_descendant(k: int, ancestor: int) -> bool:
        seen = 0
        cur = k
        while cur != 0:
            if cur == ancestor:
                return True
            cur = heads[cur - 1]
            seen += 1
            if seen > n:  # cycle guard; should never trigger on a valid tree
                return False
        return False

    for dep in range(1, n + 1):
        head = heads[dep - 1]
        if head == 0:
            continue
        lo, hi = min(head, dep), max(head, dep)
        for between in range(lo + 1, hi):
            if not is_descendant(between, head):
                return False
    return True


def merge_representative(group: AlignGroup, doc) -> int:
    """For a `"merge"` group, the spaCy token whose head lies outside the
    group (leftmost, if more than one does — this happens for `cannot`,
    where both `can` and `not` attach directly to the same external
    verb); if none does, the rightmost token in the group.
    """
    external = [i for i in group.spacy if doc[i].head.i not in group.spacy]
    return external[0] if external else group.spacy[-1]


def outgoing_head_and_relation(
    representative_spacy_idx: int,
    friction_pos: str,
    doc,
    spacy_to_friction: dict[int, int],
    corrections: CorrectionCounts,
) -> tuple[int, str]:
    """The `(head_1based, relation)` a token derives from standing in for
    `representative_spacy_idx` in spaCy's own parse: `0`/`"root"` if that
    spaCy token is the sentence root, its mapped head/relation otherwise.
    """
    head_tok = doc[representative_spacy_idx]
    is_root = head_tok.head.i == head_tok.i
    relation = relation_for_token(friction_pos, head_tok.dep_, head_tok.pos_, corrections)
    if is_root:
        return 0, relation
    if head_tok.head.i not in spacy_to_friction:
        # Not observed in this corpus (verified: no non-space spaCy token
        # ever has a whitespace token as its head), but if it ever
        # happens, drop the sentence rather than crash on a missing key.
        raise AlignmentFailure("token's head maps to no aligned friction token")
    return spacy_to_friction[head_tok.head.i] + 1, relation


def process_group(
    group: AlignGroup,
    friction_tokens: list[FrictionToken],
    doc,
    spacy_to_friction: dict[int, int],
    corrections: CorrectionCounts,
) -> list[GoldToken]:
    """Builds the gold token(s) for one alignment group.

    `"single"` and `"merge"` groups produce exactly one gold token, whose
    head/relation come from spaCy's parse (through the merge group's
    chosen representative). A `"split"` group produces one gold token per
    friction fragment: the first fragment is the group head and inherits
    spaCy's head/relation exactly like a `"single"` group would; every
    other fragment is a synthetic attachment — friction's tokenizer, not
    spaCy's parser, drew that boundary, so there is no dependency
    judgement to inherit for it — pinned to the first fragment with a
    fixed `other` relation.
    """
    if group.kind == "split":
        head_idx = group.friction[0]
        head_1based, relation = outgoing_head_and_relation(
            group.spacy[0], friction_tokens[head_idx].pos, doc, spacy_to_friction, corrections
        )
        tokens = [
            GoldToken(friction_tokens[head_idx].surface, friction_tokens[head_idx].pos, head_1based, relation)
        ]
        for frag_idx in group.friction[1:]:
            tokens.append(
                GoldToken(
                    friction_tokens[frag_idx].surface,
                    friction_tokens[frag_idx].pos,
                    head_idx + 1,
                    "other",
                )
            )
        return tokens

    representative = merge_representative(group, doc) if group.kind == "merge" else group.spacy[0]
    friction_idx = group.friction[0]
    head_1based, relation = outgoing_head_and_relation(
        representative, friction_tokens[friction_idx].pos, doc, spacy_to_friction, corrections
    )
    return [GoldToken(friction_tokens[friction_idx].surface, friction_tokens[friction_idx].pos, head_1based, relation)]


def build_gold_sentence(
    sentence: InputSentence, doc, corrections: CorrectionCounts
) -> tuple[GoldSentence, dict[str, int], dict[str, int]]:
    """Aligns, then builds the head/relation-labeled gold sentence for one
    input sentence, raising [`AlignmentFailure`] if it cannot. Multi-root
    spaCy parses (spaCy decided the text is more than one sentence) are
    also treated as an alignment failure here, since friction already
    segmented this as one.
    """
    root_count = sum(1 for t in doc if t.head.i == t.i)
    if root_count != 1:
        raise AlignmentFailure(f"spaCy found {root_count} roots, expected exactly 1")

    groups, merge_counts, split_counts = align_tokens(sentence.tokens, doc)
    spacy_to_friction = build_spacy_index_to_friction_index(groups)

    gold_tokens: list[GoldToken] = []
    for group in groups:
        gold_tokens.extend(
            process_group(group, sentence.tokens, doc, spacy_to_friction, corrections)
        )

    root_tokens = [t for t in gold_tokens if t.head == 0]
    if len(root_tokens) != 1:
        raise AlignmentFailure(
            f"post-alignment tree has {len(root_tokens)} root tokens, expected exactly 1"
        )

    return (
        GoldSentence(sentence.doc_id, sentence.genre, sentence.sent_index, sentence.text, gold_tokens),
        merge_counts,
        split_counts,
    )


def doc_split(doc_id: str) -> str:
    """Deterministic train/test split by document id: a document's own
    sha256 puts it in test with ~20% probability, train otherwise. Every
    sentence from the same document always lands in the same split.
    """
    digest = hashlib.sha256(doc_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    return "train" if bucket < TRAIN_SPLIT_PERCENT else "test"


def write_conllu(sentences: list[GoldSentence], out_path: str) -> None:
    # A friction sentence's `text` is a byte-honest slice of its source
    # document, which for a soft-wrapped markdown paragraph can carry a
    # literal embedded newline. That is real and correctly preserved
    # everywhere token offsets matter; only this single-line `# text`
    # comment collapses interior whitespace runs (including such
    # newlines) to one space, purely so the comment stays one physical
    # line — token surfaces themselves never contain whitespace, so
    # nothing about the gold annotation itself is touched by this.
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for sent in sentences:
            split = doc_split(sent.doc_id)
            display_text = " ".join(sent.text.split())
            f.write(f"# sent_id = {sent.doc_id}:{sent.sent_index}\n")
            f.write(f"# split = {split}\n")
            f.write(f"# text = {display_text}\n")
            for i, tok in enumerate(sent.tokens, start=1):
                f.write(f"{i}\t{tok.surface}\t{tok.pos}\t{tok.head}\t{tok.relation}\n")
            f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="crates/friction-nlp/weights/gold_dep_en.conllu",
        help="output CoNLL-U-shaped gold file path",
    )
    parser.add_argument(
        "--target-sentences",
        type=int,
        default=DEFAULT_TARGET_SENTENCES,
        help="approximate sentence count to stride-sample down to",
    )
    args = parser.parse_args()

    input_sentences = read_input_sentences(sys.stdin)
    total_candidates = len(input_sentences)
    print(f"read {total_candidates} candidate sentences from stdin", file=sys.stderr)

    nlp = spacy.load("en_core_web_sm", exclude=["ner", "lemmatizer"])

    aligned_naive = 0
    aligned_normalized = 0
    misaligned = 0
    hyphen_merges = 0
    cannot_merges = 0
    split_groups_total = 0
    split_fragments_total = 0

    pool: list[GoldSentence] = []
    corrections = CorrectionCounts()

    texts = [s.text for s in input_sentences]
    docs = nlp.pipe(texts, batch_size=128)

    non_projective_count = 0

    for sentence, doc in zip(input_sentences, docs):
        if naive_alignment_matches(sentence.tokens, doc):
            aligned_naive += 1

        try:
            gold_sentence, merge_counts, split_counts = build_gold_sentence(sentence, doc, corrections)
        except AlignmentFailure:
            misaligned += 1
            continue

        aligned_normalized += 1
        hyphen_merges += merge_counts["hyphen"]
        cannot_merges += merge_counts["cannot"]
        split_groups_total += split_counts["groups"]
        split_fragments_total += split_counts["fragments"]

        heads = [t.head for t in gold_sentence.tokens]
        if not is_projective(heads):
            non_projective_count += 1
            continue

        pool.append(gold_sentence)

    print(
        f"alignment before normalization: {aligned_naive}/{total_candidates} "
        f"({100 * aligned_naive / max(total_candidates, 1):.2f}%)",
        file=sys.stderr,
    )
    print(
        f"alignment after normalization: {aligned_normalized}/{total_candidates} "
        f"({100 * aligned_normalized / max(total_candidates, 1):.2f}%); "
        f"merges: {hyphen_merges} hyphen-run, {cannot_merges} cannot; "
        f"projections (friction over-split, spaCy token kept whole): "
        f"{split_groups_total} spaCy tokens projected onto "
        f"{split_groups_total + split_fragments_total} friction fragments "
        f"({split_fragments_total} synthetic `other`-relation attachments); "
        f"{misaligned} sentences dropped as still misaligned",
        file=sys.stderr,
    )

    non_proj_rate = 100 * non_projective_count / max(aligned_normalized, 1)
    print(
        f"non-projective trees: {non_projective_count}/{aligned_normalized} "
        f"({non_proj_rate:.2f}% of aligned sentences)",
        file=sys.stderr,
    )
    if non_proj_rate > 5.0:
        print(
            "STOPPING: non-projective rate exceeds 5% of the aligned pool; "
            "this would invalidate the arc-eager transition system's design "
            "choice. Not writing a gold file.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    print(
        f"closed-class corrections over the full pool ({len(pool)} sentences): "
        f"det={corrections.det}, aux={corrections.aux}, mark={corrections.mark}",
        file=sys.stderr,
    )

    relation_freq_pool: dict[str, int] = {}
    for sent in pool:
        for tok in sent.tokens:
            relation_freq_pool[tok.relation] = relation_freq_pool.get(tok.relation, 0) + 1
    total_pool_tokens = sum(relation_freq_pool.values())
    print(f"relation frequency over the full {len(pool)}-sentence pool:", file=sys.stderr)
    for rel in sorted(relation_freq_pool):
        print(f"  {rel}: {relation_freq_pool[rel]}", file=sys.stderr)
    other_count = relation_freq_pool.get("other", 0)
    print(
        f"  ({other_count}/{total_pool_tokens} tokens, "
        f"{100 * other_count / max(total_pool_tokens, 1):.2f}%, collapsed to other)",
        file=sys.stderr,
    )

    pool.sort(key=lambda s: (s.doc_id, s.sent_index))
    pool_n = len(pool)
    target = args.target_sentences
    if pool_n <= target:
        stride = 1
        sampled = pool
        print(
            f"pool ({pool_n}) is at or below the {target}-sentence target; using all of it",
            file=sys.stderr,
        )
    else:
        stride = max(1, round(pool_n / target))
        sampled = pool[::stride]
        print(
            f"stride {stride} over {pool_n}-sentence pool yields {len(sampled)} sentences "
            f"(target {target})",
            file=sys.stderr,
        )

    train_sentences = [s for s in sampled if doc_split(s.doc_id) == "train"]
    test_sentences = [s for s in sampled if doc_split(s.doc_id) == "test"]
    train_docs = {s.doc_id for s in train_sentences}
    test_docs = {s.doc_id for s in test_sentences}
    overlap = train_docs & test_docs
    print(
        f"train/test split by document: {len(train_sentences)} train sentences "
        f"({len(train_docs)} docs), {len(test_sentences)} test sentences "
        f"({len(test_docs)} docs), document overlap: {len(overlap)}",
        file=sys.stderr,
    )
    assert not overlap, "a document ended up in both splits — split-by-document is broken"

    write_conllu(sampled, args.out)

    sample_relation_freq: dict[str, int] = {}
    sample_tokens = 0
    for sent in sampled:
        for tok in sent.tokens:
            sample_relation_freq[tok.relation] = sample_relation_freq.get(tok.relation, 0) + 1
            sample_tokens += 1
    print(f"relation frequency over the final {len(sampled)}-sentence gold file:", file=sys.stderr)
    for rel in sorted(sample_relation_freq):
        print(f"  {rel}: {sample_relation_freq[rel]}", file=sys.stderr)
    sample_other = sample_relation_freq.get("other", 0)
    print(
        f"  ({sample_other}/{sample_tokens} tokens, "
        f"{100 * sample_other / max(sample_tokens, 1):.2f}%, collapsed to other)",
        file=sys.stderr,
    )

    print(f"wrote {len(sampled)} sentences to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
