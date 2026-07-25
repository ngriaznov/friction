#!/usr/bin/env python3
"""Drafts the part-of-speech gold file from friction's own sentence stream.

Reads the newline-delimited JSON `crates/friction-nlp/examples/dump_sentences.rs`
produces on stdin (one object per candidate sentence: friction's own
tokenization, its own current tagger's output, and byte offsets relative to
the sentence's own text) and writes a tab-separated gold-tag file in the
same `word<TAB>tag` format `crates/friction-nlp/weights/gold_pos_en.tsv`
already used, plus a `# split=train`/`# split=test` marker line ahead of
every sentence's tokens (see `write_gold_file` below for why this is
backward-compatible with the existing format rather than a breaking change).

Pipeline, in order:

1. Parse every sentence's `text` with spaCy `en_core_web_sm`, offline, as a
   one-time drafting aid -- the same role `tools/gold_dep/draft_gold.py`
   uses it for. spaCy is MIT-licensed; nothing from its model weights is
   redistributed here beyond the tags this script derives from its output.
2. Align spaCy's tokenization to friction's own, in both directions, using
   the exact same bidirectional merge/projection logic
   `tools/gold_common/align.py` implements (shared with the dependency-gold
   pipeline, not reimplemented here). Sentences whose tokenization still
   disagrees after both are dropped.
3. Every friction token's gold tag is built from its own [`TokenKind`],
   computed independently of spaCy (see `classify_token_kind` below, a
   direct port of the same-named Rust function in `src/tag.rs`):
   - `Number` and `Symbol` tokens always get their deterministic
     passthrough tag (`CD`/`SYM`), regardless of anything spaCy said --
     the same invariant `PerceptronTagger::tag` and
     `examples/train_perceptron.rs`'s consistency check enforce at
     inference and training time.
   - `Punctuation` tokens always get [`punctuation_tag`]'s classification
     of their own surface text -- a direct Python port of
     `tag_perceptron.rs`'s Rust function of the same name (see
     `crates/friction-nlp/weights/NOTICE.md`'s tagset decision for why
     punctuation is never tagged by its own literal surface text anymore).
   - `Word` tokens take the aligned spaCy token's own fine-grained tag
     (`token.tag_`), which is already Penn-Treebank-shaped -- including
     spaCy's own use of the same closed punctuation tags this script
     independently assigns to non-word tokens, which is exactly why this
     tagset was chosen (see NOTICE.md).
4. For a `Word`-kind friction token produced by *projection* (one spaCy
   token's span covers several friction tokens -- friction's own tokenizer
   over-split, e.g. `S2O` -> `S`/`2`/`O`), the first friction fragment
   inherits that spaCy token's tag directly; every other fragment that is
   *also* `Word`-kind (a `Number`/`Punctuation`/`Symbol` fragment always
   takes its own deterministic tag regardless, per point 3) inherits the
   exact same tag -- there is no independent spaCy judgement about a
   sub-fragment friction's own tokenizer invented, so propagating the
   parent token's tag preserves the most information available rather
   than substituting an arbitrary placeholder.
5. For a `Word`-kind friction token produced by a *merge* (several spaCy
   tokens collapse onto one friction token -- `cannot`, a hyphen compound,
   a contraction, a possessive), the *first* spaCy fragment's tag is taken
   verbatim. This mirrors real Penn-Treebank convention for the two most
   common shapes (`cannot` as one token is tagged `MD` in genuine PTB, and
   the first fragment is exactly `can`'s own tag; a possessive like
   `Python's` inherits the head noun's tag) at some cost to hyphen
   compounds acting adjectivally before a noun (`state-of-the-art
   solution`), where the first fragment (`state`, typically `NN`) is not
   always the compound's true category (`JJ`) -- a known, documented
   limitation, not an oversight; see "merges by shape" in this script's
   own stderr report for how often this shape actually occurs.
6. Apply a small closed-class correction pass over `Word`-kind tags: a
   short, exact-case (never case-folded), unambiguous-by-lexeme dictionary
   (articles, `not`, three unambiguous coordinators, unambiguous
   personal/possessive pronouns) plus one structural correction (a token
   immediately after a modal is a bare infinitive, so `NN`/`NNS` there
   becomes `VB`) -- see `CLOSED_CLASS_TAGS` and `correct_modal_followups`
   below for the exact membership and the reasoning for what was
   deliberately left out: English modals (real noun/verb readings, left to
   spaCy's own context-sensitive tagging) and, less obviously, `to` itself
   -- genuine Penn Treebank convention tags every "to" as `TO`, but spaCy
   usefully distinguishes infinitival `TO` from prepositional `IN`, and a
   direct measurement showed forcing blanket `TO` would have overridden
   1,955 of spaCy's own (more informative) distinctions.
7. Use the entire aligned pool -- no stride-sampling. The previous gold
   file was stride-sampled because every sentence was hand-reviewed, which
   this pool never is; sampling it down would be pure data loss (the exact
   mistake the dependency-gold pipeline made and then reversed -- see
   `tools/gold_dep/draft_gold.py`'s own `DEFAULT_TARGET_SENTENCES` comment).
8. Split train/test by document (not by sentence), by the same
   deterministic document-id hash `tools/gold_common/align.py::doc_split`
   uses for the dependency gold file, at the same 80/20 ratio.
9. Write the gold file, sentences in a stable sorted order (by document id,
   then sentence index), so it is byte-reproducible from the input stream.

Every step reports its own numbers to stderr; see
`crates/friction-nlp/weights/NOTICE.md` for the honest framing this
drafting process gets before it ships.

```text
cargo run --release -p friction-nlp --example dump_sentences -- corpus \
    | tools/gold_dep/venv/bin/python tools/gold_pos/draft_pos.py \
        --out crates/friction-nlp/weights/gold_pos_en.tsv
```
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import spacy

# `tools/gold_common` is a sibling directory, not an installed package --
# add `tools/` (this file's grandparent) to the path once, up front.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gold_common.align import (  # noqa: E402
    AlignGroup,
    AlignmentFailure,
    FrictionToken,
    InputSentence,
    align_tokens,
    doc_split,
    read_input_sentences,
)

TRAIN_SPLIT_PERCENT = 80

# --- classify_token_kind: a direct port of `src/tag.rs::classify_token_kind`. ---
#
# `dump_sentences.rs` does not emit a token's `TokenKind` directly (its
# JSONL schema carries only `surface`/`pos`/`start`/`end` -- see its own
# `TokenOut` struct), and this script cannot trust the `pos` field for that
# purpose either: it is friction's own *current* tagger's output, computed
# under whatever tagset happens to be shipped when `dump_sentences` last
# ran. Recomputing `TokenKind` directly from `surface` -- a pure function
# of the token's characters, exactly like the Rust original -- is
# independent of either tagger and always correct.
#
# This is safe to port faithfully in plain Python (rather than needing
# Unicode-table parity with Rust's `char::is_alphabetic`/`is_numeric`)
# because `dump_sentences.rs`'s own gold-shape filter already restricts
# every candidate sentence to pure ASCII text (see `passes_gold_filter`),
# so Python's `str.isalpha`/`str.isnumeric` agree with Rust's char
# classification on every character this script ever sees.
PROSE_PUNCTUATION = set(
    ".,;:!?'\"()[]{}-"
    + "–—…/‘’“”"
)


def classify_token_kind(text: str) -> str:
    """Direct port of `crate::tag::classify_token_kind`. Returns one of
    `"Word"`, `"Number"`, `"Punctuation"`, `"Symbol"` (the `"Whitespace"`
    case never occurs: no token this script sees is pure whitespace)."""
    if text == "":
        return "Symbol"
    first = text[0]
    if first.isnumeric() and all(c.isnumeric() or c in ".," for c in text):
        return "Number"
    if all(c.isalpha() or c in "'-" for c in text) and any(c.isalpha() for c in text):
        return "Word"
    if all(c in PROSE_PUNCTUATION for c in text):
        return "Punctuation"
    return "Symbol"


def punctuation_tag(surface: str) -> str:
    """Direct port of `crate::tag_perceptron::punctuation_tag` -- see that
    function's own doc comment for the full reasoning. Order matters: a
    run satisfying an earlier rule never reaches a later one.
    """
    if surface == "":
        return "NFP"
    if all(c in ".!?" for c in surface):
        return "."
    if all(c == "," for c in surface):
        return ","
    if all(c in ";:" for c in surface):
        return ":"
    if all(c in "-–—" for c in surface):
        return "HYPH"
    if all(c in "([{" for c in surface):
        return "-LRB-"
    if all(c in ")]}" for c in surface):
        return "-RRB-"
    if all(c in "‘“" for c in surface):
        return "``"
    if all(c in "'\"’”" for c in surface):
        return "''"
    return "NFP"


# --- closed-class word-level corrections (Task 2, point 6 above) ---
#
# Restricted to lexemes whose Penn tag is a fact independent of context, not
# a judgement call. Two things deliberately narrow this list from a first
# draft that looked plausible on paper but was empirically wrong:
#
# - **Matching is exact-case, never case-folded.** A case-folded version of
#   this dictionary (checking `surface.lower()`) was tried first and
#   measured directly against this corpus: it fired 2,109 "corrections",
#   the overwhelming majority of them wrong -- `US` (the country, all
#   caps) matched lowercase `"us"` and was forced from spaCy's correct
#   `NNP` to `PRP` 47 times; capital standalone `A` (a list marker, as in
#   "Q & A") matched `"a"` and was forced from `NNP`/`NN`/`RB` to `DT` 14
#   times; `I` (matched via `.lower()` against `"i"`) and `IT`/`OR`-shaped
#   acronyms had the same problem. Exact-case matching (this dictionary,
#   lowercase keys only, checked against the token's own unmodified
#   surface) eliminates the whole class: re-measured this way, the same
#   corpus produces only 5 corrections, every one a genuine spaCy
#   tokenization-context mistake (see `draft_pos_corrections_check` notes
#   in this repo's own development history if reproducing this
#   measurement). The cost is that a sentence-initial capitalized instance
#   ("The", "Not", "It") is never corrected by this dictionary -- accepted,
#   since spaCy is already reliable on these extremely common closed-class
#   words in sentence-initial position specifically.
# - **`to` is not in this list**, despite genuine Penn Treebank convention
#   tagging every occurrence of the word "to" as `TO` regardless of
#   function. spaCy's `en_core_web_sm` does not follow that convention: it
#   tags infinitival "to" (`I want to go`) as `TO` but prepositional "to"
#   (`I went to Paris`) as `IN`, a real and useful distinction (this
#   crate's own `gates.rs`/`chunk.rs` look for `TO` specifically to find an
#   infinitive-marker construction, which is exactly spaCy's `TO` sense,
#   not the prepositional one) -- measured directly, forcing every literal
#   "to" to `TO` regardless of spaCy's own distinction "corrected" 1,955
#   tokens, effectively erasing that distinction across the whole corpus.
#   That is a judgement call this drafting process should not be making
#   unilaterally, so it is left to spaCy.
#
# Also excludes English modals (`will`, `would`, `may`, `might`, `must` all
# have real noun/main-verb readings -- "a must-have", "his last will", "the
# might of the empire" -- so a blind lexeme override risks introducing
# exactly the kind of error this pass exists to prevent) and excludes "her"
# (genuinely ambiguous between object pronoun `PRP` and possessive
# determiner `PRP$` by context: "I see her" vs "her book"). "his" is kept
# despite superficially the same ambiguity, because Penn Treebank
# convention tags every use of "his" `PRP$` regardless of syntactic
# position (unlike "her", English does not use a morphologically distinct
# absolute form for "his").
CLOSED_CLASS_TAGS: dict[str, str] = {
    "a": "DT",
    "an": "DT",
    "the": "DT",
    "not": "RB",
    "and": "CC",
    "or": "CC",
    "nor": "CC",
    "i": "PRP",
    "you": "PRP",
    "he": "PRP",
    "she": "PRP",
    "it": "PRP",
    "we": "PRP",
    "they": "PRP",
    "me": "PRP",
    "him": "PRP",
    "them": "PRP",
    "us": "PRP",
    "my": "PRP$",
    "your": "PRP$",
    "our": "PRP$",
    "their": "PRP$",
    "its": "PRP$",
    "his": "PRP$",
}


class CorrectionCounts:
    def __init__(self) -> None:
        self.by_tag: dict[str, int] = {}
        self.modal_followup = 0

    def record(self, forced_tag: str) -> None:
        self.by_tag[forced_tag] = self.by_tag.get(forced_tag, 0) + 1

    def total(self) -> int:
        return sum(self.by_tag.values()) + self.modal_followup


def apply_closed_class_dictionary(
    surfaces: list[str], tags: list[str], corrections: CorrectionCounts
) -> None:
    """Overrides `tags[i]` in place wherever `surfaces[i]` (its exact,
    unmodified case -- see [`CLOSED_CLASS_TAGS`]'s own docs for why this
    must not be case-folded) is a member of [`CLOSED_CLASS_TAGS`], counting
    every actual change (spaCy's drafted tag disagreed with the
    closed-class fact) into `corrections`.
    """
    for i, surface in enumerate(surfaces):
        forced = CLOSED_CLASS_TAGS.get(surface)
        if forced is not None and tags[i] != forced:
            corrections.record(forced)
            tags[i] = forced


def correct_modal_followups(tags: list[str], corrections: CorrectionCounts) -> None:
    """A token immediately after a modal (`MD`) is a bare infinitive in
    English -- a syntactic fact about modals, not a judgement about the
    specific token -- so `NN`/`NNS` there is always a mistag, corrected to
    `VB`. Mirrors the equivalent hand-verified correction the previous gold
    file's `NOTICE.md` documents, generalized from a one-time observation
    into a structural rule applied uniformly.
    """
    for i in range(len(tags) - 1):
        if tags[i] == "MD" and tags[i + 1] in ("NN", "NNS"):
            corrections.modal_followup += 1
            tags[i + 1] = "VB"


def draft_sentence_tags(
    sentence: InputSentence, doc
) -> tuple[list[str], list[str], dict[str, int], dict[str, int]]:
    """Builds `(surfaces, tags)` for every friction token in `sentence`,
    raising [`AlignmentFailure`] if spaCy's tokenization cannot be
    reconciled with friction's. Also returns this sentence's own merge-shape
    and split-fragment counts (for `main`'s pooled reporting), unmodified by
    the closed-class correction pass -- callers apply that separately so its
    counts stay attributable to the correction step, not the alignment step.
    """
    groups, merge_shape_counts, split_counts = align_tokens(sentence.tokens, doc)

    n = len(sentence.tokens)
    surfaces = [t.surface for t in sentence.tokens]
    kinds = [classify_token_kind(s) for s in surfaces]
    raw_tags: list[str | None] = [None] * n

    for group in groups:
        if group.kind == "split":
            source_tag = doc[group.spacy[0]].tag_
            for friction_idx in group.friction:
                raw_tags[friction_idx] = source_tag
            continue

        # "single" and "merge" groups both cover exactly one friction
        # token; a merge group's tag comes from its *first* spaCy
        # fragment (see this module's own docstring, point 5, for why).
        friction_idx = group.friction[0]
        source_spacy_idx = group.spacy[0]
        raw_tags[friction_idx] = doc[source_spacy_idx].tag_

    assert all(t is not None for t in raw_tags), (
        "every friction token must be covered by exactly one alignment group"
    )

    tags: list[str] = []
    for surface, kind, raw_tag in zip(surfaces, kinds, raw_tags):
        if kind == "Number":
            tags.append("CD")
        elif kind == "Symbol":
            tags.append("SYM")
        elif kind == "Punctuation":
            tags.append(punctuation_tag(surface))
        else:
            assert kind == "Word"
            tags.append(raw_tag)  # type: ignore[arg-type]

    return surfaces, tags, merge_shape_counts, split_counts


def write_gold_file(
    sentences: list[tuple[str, list[str], list[str]]], out_path: str
) -> None:
    """Writes `(split, surfaces, tags)` triples in the existing
    `word<TAB>tag` gold-file format, one sentence per blank-line-delimited
    block, with a `# split=train`/`# split=test` marker line immediately
    before each block's tokens.

    This is additive, not a breaking format change: `parse_gold_file` in
    `crates/friction-nlp/src/tag_perceptron.rs` already skips any line with
    no tab character without disturbing sentence boundaries (a comment line
    matches that exactly), so an unmodified reader of this file still
    recovers the same sentences it always did -- it simply never sees the
    marker. `parse_gold_file_split` is the split-aware reader that does.
    """
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for split, surfaces, tags in sentences:
            f.write(f"# split={split}\n")
            for surface, tag in zip(surfaces, tags):
                f.write(f"{surface}\t{tag}\n")
            f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="crates/friction-nlp/weights/gold_pos_en.tsv",
        help="output gold-tag file path",
    )
    args = parser.parse_args()

    input_sentences = read_input_sentences(sys.stdin)
    total_candidates = len(input_sentences)
    print(f"read {total_candidates} candidate sentences from stdin", file=sys.stderr)

    nlp = spacy.load("en_core_web_sm", exclude=["ner", "lemmatizer", "parser"])

    texts = [s.text for s in input_sentences]
    docs = list(nlp.pipe(texts, batch_size=128))

    aligned_sentences: list[tuple[str, list[str], list[str]]] = []
    aligned_doc_sent: list[tuple[str, int]] = []
    misaligned = 0
    merge_shape_totals: dict[str, int] = {}
    split_groups_total = 0
    split_fragments_total = 0
    corrections = CorrectionCounts()

    for sentence, doc in zip(input_sentences, docs):
        try:
            surfaces, tags, merge_shape_counts, split_counts = draft_sentence_tags(
                sentence, doc
            )
        except AlignmentFailure:
            misaligned += 1
            continue

        for shape, count in merge_shape_counts.items():
            merge_shape_totals[shape] = merge_shape_totals.get(shape, 0) + count
        split_groups_total += split_counts["groups"]
        split_fragments_total += split_counts["fragments"]

        apply_closed_class_dictionary(surfaces, tags, corrections)
        correct_modal_followups(tags, corrections)

        split = doc_split(sentence.doc_id, TRAIN_SPLIT_PERCENT)
        aligned_sentences.append((split, surfaces, tags))
        aligned_doc_sent.append((sentence.doc_id, sentence.sent_index))

    aligned_count = len(aligned_sentences)
    print(
        f"alignment: {aligned_count}/{total_candidates} "
        f"({100 * aligned_count / max(total_candidates, 1):.2f}%) aligned; "
        f"{misaligned} sentences dropped as misaligned",
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
        f"({split_fragments_total} propagated-tag fragments beyond the first)",
        file=sys.stderr,
    )

    print(
        f"closed-class corrections over the full {aligned_count}-sentence aligned pool "
        f"(spaCy's drafted tag disagreed with a closed-class fact): "
        f"{dict(sorted(corrections.by_tag.items()))}, "
        f"modal-followup (NN/NNS after MD -> VB): {corrections.modal_followup}, "
        f"total: {corrections.total()}",
        file=sys.stderr,
    )

    # Sort by (doc_id, sent_index) for a stable, byte-reproducible file --
    # same convention `tools/gold_dep/draft_gold.py` uses.
    order = sorted(range(aligned_count), key=lambda i: aligned_doc_sent[i])
    sorted_sentences = [aligned_sentences[i] for i in order]

    train_sentences = [s for s in sorted_sentences if s[0] == "train"]
    test_sentences = [s for s in sorted_sentences if s[0] == "test"]
    train_docs = {
        aligned_doc_sent[i][0] for i in order if aligned_sentences[i][0] == "train"
    }
    test_docs = {
        aligned_doc_sent[i][0] for i in order if aligned_sentences[i][0] == "test"
    }
    overlap = train_docs & test_docs
    print(
        f"train/test split by document: {len(train_sentences)} train sentences "
        f"({len(train_docs)} docs), {len(test_sentences)} test sentences "
        f"({len(test_docs)} docs), document overlap: {len(overlap)}",
        file=sys.stderr,
    )
    assert not overlap, "a document ended up in both splits -- split-by-document is broken"

    write_gold_file(sorted_sentences, args.out)

    total_tokens = sum(len(tags) for _, _, tags in sorted_sentences)
    train_tokens = sum(len(tags) for _, _, tags in train_sentences)
    test_tokens = sum(len(tags) for _, _, tags in test_sentences)
    print(
        f"wrote {len(sorted_sentences)} sentences ({total_tokens} tokens) to {args.out}: "
        f"{len(train_sentences)} train sentences ({train_tokens} tokens), "
        f"{len(test_sentences)} test sentences ({test_tokens} tokens)",
        file=sys.stderr,
    )

    word_tokens = sum(
        1
        for _, surfaces, tags in sorted_sentences
        for s in surfaces
        if classify_token_kind(s) == "Word"
    )
    print(f"of which {word_tokens} are Word-kind tokens the model actually scores", file=sys.stderr)


if __name__ == "__main__":
    main()
