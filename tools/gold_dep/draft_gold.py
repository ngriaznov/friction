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
2. Align spaCy's tokenization to friction's own, in both directions,
   unconditionally in each: *merging*, whenever a run of consecutive
   spaCy tokens spans exactly one friction token (spaCy splits what
   friction keeps as one — hyphen compounds, `cannot`, possessives,
   contractions, punctuation runs, all of it), and *projecting*,
   whenever one spaCy token's span covers a run of several friction
   tokens (friction's own tokenizer is the one that over-split — mixed
   letter/digit runs, a number fused with trailing punctuation).
   Sentences whose tokenization still disagrees after both are dropped.
3. Drop non-projective trees: arc-eager (the transition system this gold
   file trains) can only derive projective ones.
4. Apply closed-class relation corrections: `det` on determiners, `aux`
   on modals, `mark` on subordinating conjunctions introducing a clause.
   These are facts about the token's own part of speech, not the
   parser's discretion — except for the sentence root, which has no head
   and so cannot bear a head-relative relation at all; the root always
   keeps relation `root`, correction or not.
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
import sys
from dataclasses import dataclass, field
from pathlib import Path

import spacy

# `tools/gold_common` is a sibling directory, not an installed package --
# add `tools/` (this file's grandparent) to the path once, up front, so the
# import below resolves the same way whether this script is invoked
# directly or via `python tools/gold_dep/draft_gold.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gold_common.align import (  # noqa: E402
    AlignGroup,
    AlignmentFailure,
    FrictionToken,
    InputSentence,
    align_tokens,
    build_spacy_index_to_friction_index,
    doc_split,
    naive_alignment_matches,
    read_input_sentences,
    sentence_aligns,
    spacy_root_count,
)

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

# No cap by default: use every aligned sentence.
#
# An earlier version of this script capped the output near 4,000 sentences,
# carried over from how the part-of-speech gold set was sized. That cap made
# sense there and does not make sense here. The POS set was reviewed by hand,
# so its size was rationed by review capacity — a real, scarce budget. This
# file is drafted and corrected entirely mechanically; nothing in it is
# reviewed sentence by sentence, so there is no budget to ration and a stride
# is pure data loss.
#
# It measurably cost accuracy: training on the strided 2,744 sentences instead
# of the full pool gave a parser several points worse on every relation, and
# the three rarest relations were the worst hit, since a stride thins exactly
# the tail that was already thinnest.
DEFAULT_TARGET_SENTENCES = 0  # 0 means "no cap"
TRAIN_SPLIT_PERCENT = 80


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


def sentence_aligns_single_root(friction_tokens: list[FrictionToken], doc, **flags: bool) -> bool:
    """Like [`sentence_aligns`] from `gold_common.align`, but also requires
    spaCy to have parsed the text as exactly one sentence (a single root) --
    the dependency file's own extra precondition, since a multi-root parse
    means spaCy disagreed with friction about the sentence boundary, which
    the POS file does not care about but this one does (see
    `build_gold_sentence` below).
    """
    if spacy_root_count(doc) != 1:
        return False
    return sentence_aligns(friction_tokens, doc, **flags)


@dataclass
class CorrectionCounts:
    det: int = 0
    aux: int = 0
    mark: int = 0
    # A closed-class correction would have fired on the sentence root,
    # where it is incoherent (a root has no head, so it cannot bear a
    # head-relative relation) — see `outgoing_head_and_relation`, which
    # suppresses the correction and counts it here instead of applying it.
    suppressed_root_det: int = 0
    suppressed_root_aux: int = 0
    suppressed_root_mark: int = 0


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

    A token that is itself the sentence root (its own head) always wins
    that choice outright, ahead of ordinary external-head tokens in the
    same group. Two real cases the generalized merge rule surfaced
    needed this, not just "root counts as external":

    - A sentence-initial contraction like `Let's` merges `Let`+`'s` into
      one friction token; `Let` is spaCy's root and `'s` (dep `dobj`,
      head `Let`) points *within* the group, so with a plain "not in
      group" check neither token looks external and the rightmost
      fallback picks `'s`, losing the root (measured: 151 sentences).
    - A merge group can contain the root *and* a token with a genuine
      external head to something else nearby (`pre-installed`, where
      `installed` is the root but `pre` independently points to `are`
      outside the group) — plain left-to-right tie-breaking among
      "external-or-root" candidates would then pick `pre`, still losing
      the root, since it comes first (measured: 7 further sentences).

    Root detection therefore runs first and short-circuits: nothing is
    more "outside the group" than being the top of the whole tree, so it
    is never merely one candidate among several.
    """
    for i in group.spacy:
        if doc[i].head.i == i:
            return i
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
    `representative_spacy_idx` in spaCy's own parse: `(0, "root")` if
    that spaCy token is the sentence root, its mapped head/relation
    otherwise.

    A root is never eligible for the closed-class corrections
    `relation_for_token` applies: `det`/`aux`/`mark` are all
    head-relative facts ("this token is a determiner *of its head*"),
    and a root has no head. Applying one anyway is exactly the bug this
    guard exists to prevent — a root token bearing e.g. `aux` produces a
    gold tree with `head == 0` and `relation != "root"`, which is
    internally inconsistent and unrecoverable downstream. Whichever
    correction *would* have applied is still tallied, into
    `corrections.suppressed_root_*`, so the gap between "what the
    closed-class rule would say" and "what a root can coherently say" is
    visible rather than silently absorbed.
    """
    head_tok = doc[representative_spacy_idx]
    is_root = head_tok.head.i == head_tok.i

    if is_root:
        scratch = CorrectionCounts()
        would_be = relation_for_token(friction_pos, head_tok.dep_, head_tok.pos_, scratch)
        if would_be != "root":
            if would_be == "det":
                corrections.suppressed_root_det += 1
            elif would_be == "aux":
                corrections.suppressed_root_aux += 1
            elif would_be == "mark":
                corrections.suppressed_root_mark += 1
        return 0, "root"

    relation = relation_for_token(friction_pos, head_tok.dep_, head_tok.pos_, corrections)
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
    root_count = spacy_root_count(doc)
    if root_count != 1:
        raise AlignmentFailure(f"spaCy found {root_count} roots, expected exactly 1")

    groups, merge_shape_counts, split_counts = align_tokens(sentence.tokens, doc)
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

    # Regression guard for the root/closed-class-correction bug: a token
    # has head 0 exactly when its relation is "root", in both directions.
    # `outgoing_head_and_relation` should make this hold by construction
    # now — this is a loud, unconditional assertion, not a drop, because
    # a violation here means that guard itself broke.
    for tok in gold_tokens:
        assert (tok.head == 0) == (tok.relation == "root"), (
            f"root/relation invariant broken for {tok.surface!r}: "
            f"head={tok.head}, relation={tok.relation!r}"
        )

    return (
        GoldSentence(sentence.doc_id, sentence.genre, sentence.sent_index, sentence.text, gold_tokens),
        merge_shape_counts,
        split_counts,
    )


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
            split = doc_split(sent.doc_id, TRAIN_SPLIT_PERCENT)
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
        help="approximate sentence count to stride-sample down to; "
             "0 (the default) means no cap, use every aligned sentence",
    )
    args = parser.parse_args()

    input_sentences = read_input_sentences(sys.stdin)
    total_candidates = len(input_sentences)
    print(f"read {total_candidates} candidate sentences from stdin", file=sys.stderr)

    nlp = spacy.load("en_core_web_sm", exclude=["ner", "lemmatizer"])

    texts = [s.text for s in input_sentences]
    # Materialized once (not consumed as a one-shot generator) so the
    # staged alignment-rate measurements below and the real build pass
    # can each iterate it independently without re-running spaCy.
    docs = list(nlp.pipe(texts, batch_size=128))

    aligned_naive = sum(
        1 for s, d in zip(input_sentences, docs) if naive_alignment_matches(s.tokens, d)
    )
    aligned_merge_only = sum(
        1
        for s, d in zip(input_sentences, docs)
        if sentence_aligns_single_root(s.tokens, d, allow_merge=True, allow_split=False, skip_whitespace=False)
    )
    aligned_merge_projection = sum(
        1
        for s, d in zip(input_sentences, docs)
        if sentence_aligns_single_root(s.tokens, d, allow_merge=True, allow_split=True, skip_whitespace=False)
    )

    def pct(n: int) -> str:
        return f"{n}/{total_candidates} ({100 * n / max(total_candidates, 1):.2f}%)"

    print(f"alignment, raw (no normalization): {pct(aligned_naive)}", file=sys.stderr)
    print(f"alignment, +merge only: {pct(aligned_merge_only)}", file=sys.stderr)
    print(
        f"alignment, +merge +projection (whitespace tokens not yet filtered): "
        f"{pct(aligned_merge_projection)}",
        file=sys.stderr,
    )

    aligned_normalized = 0
    misaligned = 0
    merge_shape_totals: dict[str, int] = {}
    split_groups_total = 0
    split_fragments_total = 0

    pool: list[GoldSentence] = []
    corrections = CorrectionCounts()

    non_projective_count = 0

    for sentence, doc in zip(input_sentences, docs):
        try:
            gold_sentence, merge_shape_counts, split_counts = build_gold_sentence(
                sentence, doc, corrections
            )
        except AlignmentFailure:
            misaligned += 1
            continue

        aligned_normalized += 1
        for shape, count in merge_shape_counts.items():
            merge_shape_totals[shape] = merge_shape_totals.get(shape, 0) + count
        split_groups_total += split_counts["groups"]
        split_fragments_total += split_counts["fragments"]

        heads = [t.head for t in gold_sentence.tokens]
        if not is_projective(heads):
            non_projective_count += 1
            continue

        pool.append(gold_sentence)

    print(
        f"alignment, +merge +projection +whitespace-token fix (final): "
        f"{pct(aligned_normalized)}; {misaligned} sentences dropped as still misaligned",
        file=sys.stderr,
    )
    total_merges = sum(merge_shape_totals.values())
    print(f"merges by shape ({total_merges} total, all accepted unconditionally):", file=sys.stderr)
    for shape in sorted(merge_shape_totals):
        print(f"  {shape}: {merge_shape_totals[shape]}", file=sys.stderr)
    print(
        f"projections (friction over-split, spaCy token kept whole): "
        f"{split_groups_total} spaCy tokens projected onto "
        f"{split_groups_total + split_fragments_total} friction fragments "
        f"({split_fragments_total} synthetic `other`-relation attachments)",
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
    suppressed_total = (
        corrections.suppressed_root_det
        + corrections.suppressed_root_aux
        + corrections.suppressed_root_mark
    )
    print(
        f"closed-class corrections suppressed at the sentence root (would have "
        f"produced an incoherent head==0/relation!=root token): "
        f"det={corrections.suppressed_root_det}, aux={corrections.suppressed_root_aux}, "
        f"mark={corrections.suppressed_root_mark}, total={suppressed_total}",
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
    if target <= 0:
        stride = 1
        sampled = pool
        print(
            f"no sentence cap; using the whole {pool_n}-sentence pool",
            file=sys.stderr,
        )
    elif pool_n <= target:
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

    train_sentences = [s for s in sampled if doc_split(s.doc_id, TRAIN_SPLIT_PERCENT) == "train"]
    test_sentences = [s for s in sampled if doc_split(s.doc_id, TRAIN_SPLIT_PERCENT) == "test"]
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
