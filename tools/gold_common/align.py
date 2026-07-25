"""Shared spaCy<->friction token-alignment machinery, factored out of
`tools/gold_dep/draft_gold.py` so `tools/gold_pos/draft_pos.py` can reuse the
exact same bidirectional merge/projection logic rather than reimplementing
it. Neither drafting script owns this file; both import it.

Reads the newline-delimited JSON `crates/friction-nlp/examples/dump_sentences.rs`
produces on stdin (one object per candidate sentence: friction's own
tokenization and part-of-speech tags, byte offsets relative to the
sentence's own text) and aligns each sentence's friction tokens to spaCy's
own tokenization of the same text, in both directions, unconditionally in
each: *merging*, whenever a run of consecutive spaCy tokens spans exactly
one friction token (spaCy splits what friction keeps as one -- hyphen
compounds, `cannot`, possessives, contractions, punctuation runs, all of
it), and *projecting*, whenever one spaCy token's span covers a run of
several friction tokens (friction's own tokenizer is the one that
over-split -- mixed letter/digit runs, a number fused with trailing
punctuation). Sentences whose tokenization still disagrees after both are
dropped by the caller (this module raises [`AlignmentFailure`]; it does not
decide what a caller does with that).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass


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


def read_input_sentences(stream) -> list[InputSentence]:
    """Parses `dump_sentences`'s newline-delimited JSON stream."""
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
    span-for-span, with no merge normalization applied -- the "before
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
    than one index -- `kind` says which, `"single"` meaning neither does.
    """

    friction: list[int]
    spacy: list[int]
    kind: str  # "single" | "merge" (spaCy over-split) | "split" (friction over-split)


def classify_merge_shape(friction_surface: str, spacy_texts: list[str]) -> str:
    """Categorizes a merge group for reporting only -- the merge rule
    itself (below) accepts every shape identically; this just names which
    one a given group was, so a caller's `main` can print a breakdown.
    """
    n = len(spacy_texts)
    if n == 3 and spacy_texts[1] == "-":
        return "hyphen_single"
    if n >= 5 and n % 2 == 1 and all(spacy_texts[i] == "-" for i in range(1, n, 2)):
        return "hyphen_chain"
    if n == 2 and spacy_texts[0].lower() == "can" and spacy_texts[1].lower() == "not":
        return "cannot"
    lowered = friction_surface.lower()
    if lowered.endswith("n't"):
        return "contraction_nt"
    if re.search(r"'(ll|re|ve|d|m)$", lowered):
        return "contraction_other"
    if friction_surface.endswith("'s") or friction_surface.endswith("'s"):
        return "possessive_s"
    if re.search(r"[0-9][.,)]$", friction_surface):
        return "number_or_word_plus_trailing_punct"
    if all(re.fullmatch(r"[^\w\s]+", t) for t in spacy_texts):
        return "punct_run"
    return "other"


def align_tokens(
    friction_tokens: list[FrictionToken],
    doc,
    *,
    allow_merge: bool = True,
    allow_split: bool = True,
    skip_whitespace: bool = True,
) -> tuple[list[AlignGroup], dict[str, int], dict[str, int]]:
    """Aligns friction tokens to spaCy tokens in both directions,
    unconditionally in each: *merging*, whenever a run of consecutive
    spaCy tokens spans exactly one friction token (spaCy splits what
    friction keeps as one -- hyphen compounds, `cannot`, possessives,
    contractions, punctuation runs), and *projecting*, whenever one spaCy
    token's span covers a run of several friction tokens (friction's own
    tokenizer breaks at every letter/digit boundary and folds trailing
    `.`/`,` into a number, so `S2O` becomes three friction tokens and
    `2017,` becomes one where spaCy keeps `2017` and `,` separate) --
    handled by *projecting* one spaCy token's tag onto the run (mechanical,
    not a judgement call -- each caller documents its own per-token
    consequence of this). Both directions are unconditional and symmetric:
    there is no character-shape restriction on either one, since both are
    the same operation (several tokens on one side correspond to one on
    the other) just facing opposite ways.

    `allow_merge`/`allow_split`/`skip_whitespace` default to the
    production behavior; setting any to `False` is only for measuring
    the alignment rate *without* that piece of normalization, i.e. the
    staged rates a caller's `main` reports (raw / +merge / +merge+projection
    / +merge+projection+whitespace-fix).

    Returns `(groups, merge_shape_counts, split_counts)`. Raises
    [`AlignmentFailure`] if any span cannot be reconciled, tokens are
    left over on either side, or a disabled direction would have been
    needed.

    spaCy, unlike friction's own tokenizer (`tokenize()` in
    `tag_perceptron.rs` never emits a `Whitespace`-kind token), sometimes
    emits a standalone whitespace token -- reliably for a literal embedded
    newline in a soft-wrapped source paragraph. Such a token carries no
    friction counterpart to align against and is never any other token's
    head, so by default it is filtered out up front rather than walked
    over -- this is restoring an apples-to-apples token list, not a
    normalization choice.
    """
    spacy_indices = (
        [i for i, t in enumerate(doc) if not t.is_space]
        if skip_whitespace
        else list(range(len(doc)))
    )

    groups: list[AlignGroup] = []
    merge_shape_counts: dict[str, int] = {}
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
            # run of several friction tokens -- project it.
            if not allow_split:
                raise AlignmentFailure("split direction disabled for this measurement")
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

        # ft.end > s_end: friction's token is longer -- a run of several
        # spaCy tokens make up this one friction token. Accepted
        # unconditionally, symmetric with the split direction above.
        if not allow_merge:
            raise AlignmentFailure("merge direction disabled for this measurement")
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
        shape = classify_merge_shape(ft.surface, [doc[k].text for k in spacy_group])
        merge_shape_counts[shape] = merge_shape_counts.get(shape, 0) + 1

        groups.append(AlignGroup([fi], spacy_group, "merge"))
        fi += 1
        si = sj + 1

    if fi != n_friction or si != n_spacy:
        raise AlignmentFailure("tokens remain unconsumed on one side after alignment")

    return groups, merge_shape_counts, split_counts


def spacy_root_count(doc) -> int:
    """How many tokens in `doc` are their own head -- spaCy's convention
    for the root of each sentence it thinks the text contains. Exactly 1
    is the well-formed case; a caller that already segmented this text as
    one sentence treats anything else as spaCy disagreeing about where the
    sentence boundary is.
    """
    return sum(1 for t in doc if t.head.i == t.i)


def sentence_aligns(friction_tokens: list[FrictionToken], doc, **flags: bool) -> bool:
    """True if `align_tokens(friction_tokens, doc, **flags)` would
    succeed, without building anything from the groups -- used only for
    staged alignment-rate measurements.
    """
    try:
        align_tokens(friction_tokens, doc, **flags)
        return True
    except AlignmentFailure:
        return False


def build_spacy_index_to_friction_index(groups: list[AlignGroup]) -> dict[int, int]:
    """Maps every spaCy token index to the friction token index any
    *external* reference to it should be redirected to: the merged
    token for a merge group, and the head fragment (first friction
    token) for a split group.
    """
    mapping: dict[int, int] = {}
    for group in groups:
        target = group.friction[0]
        for spacy_idx in group.spacy:
            mapping[spacy_idx] = target
    return mapping


def doc_split(doc_id: str, train_percent: int = 80) -> str:
    """Deterministic train/test split by document id: a document's own
    sha256 puts it in test with `(100 - train_percent)`% probability, train
    otherwise. Every sentence from the same document always lands in the
    same split, and the split is a pure function of `doc_id` alone -- no
    seed, no ambient state -- so it is reproducible across separate runs
    and across the dependency-gold and POS-gold pipelines alike (which use
    the same `train_percent = 80` default).
    """
    digest = hashlib.sha256(doc_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    return "train" if bucket < train_percent else "test"
