# Dependency parser for the register module — design

Status: proposed
Date: 2026-07-25

## Why this exists

The `register` module rephrases LLM prose by moving its Biber-feature vector
toward a measured human target. Its feature detectors and its rewrite
transducers are both written against dependency structure: a present
participial is `VBG` with `dep ∈ {acl, advcl, xcomp}` and no `aux` child; a
nominalization is unpackable only when it has both a `det` child spelled
`the` and a `prep` child spelled `of` carrying a `pobj`. Without labelled
dependencies there is no detector and no transducer.

`friction-nlp` today has no parser that can supply this. `DepRelation` has
five variants — `Subject`, `Object`, `Coordination`, `ParticipialModifier`,
`Other` — and cannot distinguish `advcl` from `acl` (which selects between
two different transducers), phrasal from clausal coordination (two of the
five homing features), or mark a passive auxiliary at all (the one feature
LLMs *under*-produce, and the reason this module is interesting). The
`OnnxParser` behind the `onnx` feature is scaffolding: `ort` is an optional
dependency, no model artifact exists, and its own module docs record that no
supported export path was found.

This document specifies that parser. It is the first of four planned
documents and blocks the other three.

## Context: decisions this sits inside

Recorded here because they constrain the parser, and specified in full
elsewhere.

**The closure rule changes.** friction's four-operation engine guarantees
that every emitted word is already in the input or derived from an input word
by static table. The register transducers break that: they emit `.`, `This`,
`and`, `that`, `was`, `were`, `is`, `are`. Those eight closed-class tokens are
the complete set of non-derived output across all five transducers — every
content word remains input-derived, via `third_sg`/`past`/`past_participle`
or the 23-entry noun→verb table. Critically, none is chosen by search: each
is a fixed constant of the rule that emits it.

The replacement invariant is therefore narrower than "no synthesis":

> **Constant-function-word closure.** Content words are input-derived only —
> already present, or produced by a static derivational table. Function words
> may be emitted only from a per-transducer fixed constant set, declared in
> data. No word is ever chosen by search.

This retires exactly one pinned red fixture (`chatspeak_vs_human`, metric-
centroid optimization) and keeps five (`word_salad_synthesis` and the four
`bridge_*_insertion` cases), all of which test search-chosen insertion and
remain correctly red. Enforcement belongs in `friction-harness`'s closure
checker and is specified with the transducers, not here.

**Four documents, not one.** (1) this parser; (2) feature extraction and
target estimation; (3) transducers, objective, and selection; (4) idiolect
targeting. Each gates the next.

**Naming.** The workspace forbids referring to planning documents from code,
comments, doc comments, test names, or authored string literals. That ban
extends to this document's own phase labels: no `P0`…`P6` in Rust source.
`docs/` is exempt.

## Scope

In: an arc-eager transition parser producing one labelled edge per token,
its gold data pipeline, its training binary, its shipped weight artifact,
and the extension of `DepRelation` to the labels the register module reads.

Out: the feature extractor, the target vector, the transducers, and any
change to the existing four-operation engine. The parser ships inert — it
adds an API and an artifact, and nothing in the current fix path calls it.

## Label set

The register prototype (`docs/research/regvec/`) reads spaCy's
ClearNLP/OntoNotes scheme, **not** Universal Dependencies v2. This
distinction is load-bearing: UD calls `dobj`→`obj`, `pobj`→`obl`,
`prep`→`case`, `auxpass`→`aux:pass`, `nsubjpass`→`nsubj:pass`. Training
against UD labels while the transducers match on ClearNLP names would leave
every licensing condition matching nothing, silently — the exact failure the
prototype's own delta-validation check was built to catch.

`DepRelation` therefore takes the ClearNLP names the prototype reads:

```
Root  Acl  Advcl  Agent  Amod  Aux  AuxPass  Cc  Ccomp  Conj
Csubj  Det  Dobj  Mark  Nsubj  NsubjPass  Pobj  Prep  Xcomp  Punct  Other
```

Twenty-one variants. `Root` is not an arc label — a root token carries
`head: None` — so the arc-labelled action space is the other twenty.
`Other` is the collapse target for any drafted relation outside this set,
and is retained from the current enum; the four variants that are not
(`Subject`, `Object`, `Coordination`, `ParticipialModifier`) are removed.
The enum is already `#[non_exhaustive]`, so widening it is not a breaking
change. `HeuristicParser` is the removed variants' only consumer and is
retired with them (its sole reference elsewhere is a doc comment in
`friction-metrics/src/symmetry.rs`).

`DepEdge` and `SentenceParse` are unchanged. `SentenceParse::new` already
enforces one edge per token at matching index, in-bounds heads, and no
self-head — the invariants a transition parser must satisfy anyway, and the
tree-completeness the prototype's `_subtree_span` walk depends on.

## Architecture

Arc-eager, following the shape of the existing tagger so the two artifacts
are maintained the same way.

**Transition system.** Configuration is (stack, buffer, arc set). Four
transition types: `Shift`, `Reduce`, `LeftArc(label)`, `RightArc(label)`.
With 20 arc labels the action space is `2 + 2×20 = 42`. Preconditions are the
standard ones — `LeftArc` requires a stack top with no head assigned,
`Reduce` requires a stack top that has one — and an action whose
precondition fails is masked out of the argmax rather than trusted to score
low. A terminal configuration with a non-empty stack attaches every
remaining token to the root, so a parse is always produced.

**Classifier.** Averaged perceptron over sparse string features, reusing the
tagger's existing averaging and weight-table code (`tag_perceptron.rs`'s
`WeightTable` generalizes; the tagger's own use becomes one instantiation).
Structured perceptron training with early update: decode greedily, and on
the first action that diverges from the gold derivation, update and move to
the next sentence.

**Feature templates.** Standard Zhang & Nivre configuration features —
surface form and POS of the top two stack items and first three buffer
items, the leftmost/rightmost dependents of the stack top and their labels,
plus distance and valency. Exact templates are an implementation detail
fixed at training time and recorded in `NOTICE.md`, since changing them
invalidates the artifact.

**Determinism.** No RNG at runtime. Training shuffles with a fixed seed
recorded in `NOTICE.md`. Ties in the argmax break toward the lowest action
index, as the tagger already does. Same input, same weights, same tree.

**Projectivity.** Arc-eager derives only projective trees. English technical
prose is overwhelmingly projective, but not entirely. Non-projective gold
sentences are *dropped* at training time and their count reported, rather
than pseudo-projectivized: the lifting/lowering machinery is a meaningful
amount of code to serve a small tail, and a dropped sentence is honest where
a mis-restored one is not. If the drop rate exceeds 5% this decision gets
revisited before training proceeds.

## Gold data

Sourced the way `gold_pos_en.tsv` was, for the same reasons — no external
treebank licence to reason about, and training data in the genre the parser
actually runs on.

1. **Extraction.** `corpus/human/{docs,readme,blog,email,forum}` restricted
   to `class: human, split: train` in `corpus/manifest.jsonl`. Prose is
   extracted with the real `friction-parse` pipeline and segmented with
   `SrxSegmenter`, matching the tagger's gold pipeline exactly. Sentences of
   4–40 words, ASCII, no tagger `UNKNOWN` sentinel.
   **No dev-split or holdout-split document is read at any point.**
2. **Drafting.** spaCy `en_core_web_sm` parses every candidate once,
   offline. spaCy is MIT-licensed, so unlike the tagger's nlprule drafting
   step this raises no licensing question that needs arguing.
3. **Sampling.** A fixed stride over the candidate pool in corpus order —
   not a random sample — so the set is reproducible from the corpus alone.
   The stride is chosen once, to land the gold set near a target sentence
   count, and recorded in `NOTICE.md` alongside the resulting counts. Unlike
   the tagger, hand correction here is per-sentence expensive, so the target
   is set by review capacity rather than by how much data exists.
4. **Correction.** Draft output is not gold. Three passes, mirroring the
   tagger's: token alignment against friction's own tokenizer (spaCy's
   segmentation differs and mismatched sentences are dropped, not patched);
   a closed-class override for relations that are facts rather than
   judgements (`det` on determiners, `aux` on modals, `mark` on
   subordinators); and hand review of a sample, with every correction
   recorded.
5. **Split.** The gold set is itself split train/test by document, not by
   sentence, so no test sentence shares a document with a training sentence.

Artifacts: `weights/gold_dep_en.conllu`, `weights/parser_en.json.gz`, and a
`NOTICE.md` section covering provenance, feature templates, seed, and
reproduction — extending the existing file rather than adding a second one.

## Acceptance

The plan this derives from proposes ≥88% UAS on UD English EWT dev. That
number is not directly meaningful here: the gold data is silver (spaCy-
drafted), the label scheme is ClearNLP, and the domain is friction's own
corpus rather than EWT's web text. Reporting it would invite a comparison
the setup does not support. The gates are therefore:

1. **UAS ≥ 88% and LAS ≥ 85%** on the held-out own-corpus gold test split.
2. **Per-relation agreement with spaCy ≥ 90%** on the relations the register
   module actually reads — `acl`, `advcl`, `auxpass`, `conj`, `det`, `dobj`,
   `nsubj`, `pobj`, `prep`. Aggregate accuracy can hide a relation that is
   rare in the corpus and load-bearing in a transducer; this gate is
   per-relation precisely so it cannot.
3. **Determinism.** 100 documents × 3 runs, byte-identical parses.
4. **No regression** in the existing workspace gates: `cargo fmt --check`,
   `cargo clippy --workspace --all-targets -- -D warnings`, and
   `cargo test --workspace` all clean.
5. **Artifact reproducibility.** Retraining from the vendored gold file
   reproduces the shipped weights byte-for-byte.

Gate 2 is the one that matters. Gate 1 is the conventional number; gate 2 is
whether the transducers will actually fire.

For reference only, and reported without being gated on: UAS against UD
English EWT dev, label-mapped. Evaluating against EWT does not redistribute
it. If that number is very low while gate 1 passes, the parser has learned
friction's corpus rather than English, and that is worth knowing.

## Risks

**Silver-data ceiling.** Training on spaCy output caps accuracy near
spaCy's. This is accepted: the register transducers were written and
validated against spaCy's decisions, so reproducing them faithfully is the
actual requirement, and a parser that disagreed with spaCy in a "better" way
would still break the licensing conditions.

**Error inheritance.** spaCy's systematic errors become the parser's. The
hand-review pass over a sample is the only defence, and it is a sample. Any
transducer later found to misfire on a structure should send us back to the
gold data before the transducer.

**Corpus size.** 264 human train documents yield on the order of a few
thousand usable sentences after filtering — small for a parser. If gate 1 or
gate 2 fails after training, the first response is more gold data (the
corpus supports drafting far more than the tagger's 1,609-sentence sample),
not a weaker gate.

**Retiring `HeuristicParser`.** It is effectively unused, so this is low
risk, but it is a public API removal and belongs in the same change as the
replacement rather than trailing it.

## What this does not settle

Whether register runs as a fifth operation inside `friction fix` or as a
separate pass afterward. It matters — the four operations delete text, which
changes word counts, which changes every per-1000-word rate and therefore the
objective — but it does not affect the parser, and it is specified with the
transducers.
