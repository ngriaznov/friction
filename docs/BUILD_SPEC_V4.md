# friction — build spec v4.1 (source rewrite plan)

friction is a deterministic engine that removes the machine layer from LLM-generated
technical documentation by subtraction, paired substitution, and derivational
pivoting. It never synthesizes text: every word it emits is either present in the
input or derived from an input word through a static morphology table. Where no safe edit exists, it reports instead of rewriting. No model runs at
fix time. This spec is written against the existing workspace and is the build
contract for a coding agent; every design rule below is backed by an experiment run
in July 2026, recorded in the evidence ledger.

## Problem statement

Take LLM-generated technical docs that contain real information and remove the
layer that makes them read as machine-written, at the sentence level and the
document-skeleton level, without touching the information.

## Evidence ledger (what earned its place, what is banned)

Validated by experiment:
- Paired substitution ("will walk you through" to "covers", "in order to" to "to"):
  zero bad edits across all runs; prevented the fragment failure.
- Gated span deletion (seam attested or sentence-initial, clause check): produced
  clean, publishable output on the heaviest slop paragraph in 19 ms.
- Ritual-sentence deletion: ritual closers are the highest-scoring detected spans;
  whole-sentence removal is safe by construction.
- Template mining over paired corpora: auto-discovered the true doc-slop frames;
  topic confound demonstrated and patched via shared-vocabulary constraint.
- DMS detection (differential matching statistics, below): 16/16 held-out
  document classification, token-level span boundaries, zero false positives on
  human text, ~0.5M tokens/s in a Python prototype.
- Derivational pivot (light-verb construction collapse, "performs validation of"
  to "validates"): 5/5 constructed cases correct with tense inheritance, 6/6 trap
  cases rejected by the intended gates, clean rewrites on real corpus sentences
  ("made the decision to switch" to "decided to switch"). Corpus rate evidence:
  licensed LVC instances run ~1.8x more frequent in the machine corpus than the
  human corpus, independently reproducing the published nominalization finding.
- Full-engine paragraph runs: all four operations composed in one pass (9 ms),
  correct residue behavior (gate-held spans left verbatim), one-edit surgical
  precision on a real corpus paragraph (5 ms).
- Near-no-op behavior: clean text passed through untouched in every run.

Killed by experiment, banned from the build, each preserved as a red-test fixture:
- Free corpus-path synthesis (bigram pathfinding): produced word salad.
- Bridge insertion in every vocabulary regime: open vocabulary inserted content
  words ("documentation"), whitelisted function words flipped meaning ("can",
  "because"), inert whitelist plus sparse attestation broke grammar ("with",
  "to"). Search-chosen insertion of any kind is banned.
- Metric-centroid optimization: produced chat-speak that scored perfectly.
- Aggressive collapse to imperatives: produced telegraphic "dumb human" output.
- DMS as a per-edit improvement judge: scored an obviously better fix as worse.
- Document-level embedding checks: a small embedder's whole-document vector is a
  topic centroid and misses meaning changes.

Scoped, not killed: DMS detection is generator-family-specific (a Claude-register
text evaded a Qwen/Gemma/Llama index). Detection packs are therefore per-family.

## Architecture

Pipeline: parse -> discourse-context annotation -> detection (DMS + literal
automaton) -> closed-set repair -> diagnostics for everything else -> atomic apply.
One pass, plus one bounded cleanup pass (see idempotence).

### Detection layer

Two mechanisms, both deterministic, both offline-built:

1. DMS. Two suffix automata, one over a machine corpus for the model family being
   cleaned, one over the curated human corpus. Stream the input once through each,
   computing per-token longest-match lengths. The differential profile localizes
   machine-register spans with boundaries, including phrasings never mined.
   Grounding: match lengths estimate cross-entropy against a reference
   (Kontoyiannis); the differential is a model-free likelihood-ratio profile.
   Linear time, integer-only. Role: span finder and document-level reporter.
   Explicitly not a per-edit judge.
2. Literal automaton. Aho-Corasick over the mined inventory for short tells that
   carry no match-length signal (single words, short phrases).

### Repair layer: the closed operation set

Exactly four operations. Nothing else edits text.

1. Ritual deletion. Sentences or blocks matching ritual frames (closers like
   "If you have any questions...", congratulations blocks, preview paragraphs
   whose content lemmas are covered by adjacent headers) are removed whole.
2. Gated span deletion. A matched slop span is removed only if the resulting seam
   is bigram-attested in the human corpus or falls at a sentence start, and the
   sentence passes the clause-completeness gate afterward.
3. Paired substitution. Slop frame to human frame from the versioned inventory,
   with morphology from static tables ("will walk you through" -> "covers").
   Pairs come from mining plus curation; every pair ships with property tests.
4. Derivational pivot. A licensed light-verb construction (light verb, optional
   article, nominalization from the derivational lexicon, optional object-reading
   "of" phrase) collapses to its root verb: delete the light verb, derive the
   verb from the nominalization via the static table, inherit tense and
   agreement from the deleted light verb, promote the "of" complement to direct
   object. "The agent performs validation of the config file" becomes "The agent
   validates the config file." Closure holds because the emitted verb is a
   morphological form of an input word; nothing is chosen by search.
   Pivot-specific gates: (light verb, nominalization) pairs must be individually
   licensed in the pack (mined evidence, machine corpus using the construction
   where the human corpus uses the verb); the nominalization must be bare (no
   adjectives, no quantifiers, no plural); active voice only; subject-reading
   "of" phrases (the approval of the committee) are unlicensed by construction.

When no operation passes its gates, the span is reported as a Suggest-tier
diagnostic and the text is left byte-identical. Doing nothing is the designed
fallback, not a failure mode.

## Hard gates on every edit

1. Closure by construction: the operation set can only delete or substitute from
   the inventory, so no content word can enter the text. There is no bridge or
   synthesis path to gate.
2. Clause completeness: after the edit, the sentence retains a finite verb or is
   imperative-initial. Tag n-grams cannot see this (measured), so the check runs
   on the chunker's clause analysis.
3. Guard tokens: no edit may touch code spans, identifiers, links, numbers, named
   entities, negation, quantifiers, modals, or logical connectives. Modals and
   connectives were added after measured meaning flips.
4. Discourse binding: skip the edit if a neighboring sentence opens with a
   pronoun, demonstrative, or connective that could bind into the region; never
   delete a block later text references; do not break counted enumerations.
5. Seam attestation for deletions, as defined in operation 2.
6. Near-no-op: on curated human text, edits per document must stay under the
   corpus-calibrated threshold (friction's NF-3, revalidated this session).
   The pivot is a rate tell, not a binary one (humans use LVCs too, just ~1.8x
   less), so this gate is what stops it over-firing on human prose.
7. Prose blocks only. Edits fire exclusively inside prose blocks of the parse
   tree, never in headings, code, tables, or link text. Guaranteed by
   friction-parse's prose extraction; the measured failure ("Making the
   Decision" heading pivoted to "deciding") is the regression fixture.

## Crate-by-crate changes

- friction-parse: keep (block tree, byte ranges, round-trip guarantee). Add the
  discourse-context annotator (DocLead, SectionLead, ProcedureStep, InlineBody).
- friction-nlp: nlprule demoted to optional backend behind the existing trait.
  Add a fast averaged-perceptron tagger as default (also removes the build-time
  model download). Add the clause and coordination chunker (needed by gates 2 and
  4). Keep the inflection module; add static morph tables for the pair inventory
  and a derivational lexicon (noun-to-verb mappings seeded from NOMLEX/CatVar,
  entries activated only by pack-level pair licensing, never wholesale).
- corpus-tool: add `mine` (paired, position-conditioned template mining with the
  shared-vocabulary constraint; block-level mining for ritual and preview frames)
  and `index` (builds the DMS suffix automata per model family). Topic-matched
  generation recipe: same prompts through a stock model and an antislop-tuned
  model so ratio mining isolates style; published slop lists as seed and check.
  ternlight may be used here, offline only, to cluster near-duplicate templates.
- friction-packs: the pack becomes inventory plus indexes: slop frames, pairs
  with morph data, ritual and preview frames, guard-token lists, per-family DMS
  automata, all versioned and sha256-pinned. Compile-time checks: no pair's
  output matches any slop frame (disjointness, which yields idempotence), and no
  pair introduces content words absent from its input side plus schema function
  words (closure).
- friction-rules -> friction-match: DMS streaming plus the literal automaton,
  plus a tag-pattern channel that recognizes licensed light-verb constructions
  over the tagged token stream (no dependency parser required; the LVC pattern
  is shallow). Emits spans with boundaries and matched-frame ids. No edit logic.
- friction-synth -> friction-edit: applies the closed four-operation set behind
  the gates. No search, no bridges, no schema selection machinery.
- friction-plan: retired to a fixed pipeline order.
- friction-apply: unchanged (atomic span patches, re-parse between passes).
- friction-metrics: out of the fix path. Retained in `check` and the harness as
  guardrails; envelopes re-derived from curated well-written docs; the em-dash
  zero-band and centroid targets removed.
- friction-cli: `fix` (one command, stdout or in-place), `check` (spans, tell
  counts, DMS document report), `explain` (which spans fired, which operation or
  why none). SARIF stays.

## Invariants

- NF-1 determinism: unchanged. Same input and pack, same bytes.
- NF-2 idempotence: by construction from pack disjointness plus the bounded
  second pass; soft CI canary asserts a third pass changes nothing.
- NF-3 near-no-op on human text: promoted to a hard gate (gate 6).
- NF-4 offline: unchanged; mining, indexing, and ternlight are offline only.
- NF-5 span honesty: unchanged.
- NF-6 tiers: the four operations are Fix-tier by construction; everything else
  is Suggest.
- NF-7 no global tic: with synthesis gone, this reduces to inventory hygiene:
  a pair's output frame must not exceed human-corpus frequency bands; checked at
  pack build.
- NF-8 toolchain gates: unchanged.

## Scoring harness (M0, built first)

Objective: tell-span count reduction, subject to all gates, with an independent
human-vs-LLM classifier as a capped secondary signal. DMS is reported at document
level but never ranks individual edits. Distribution stats are plateau guardrails
only.

Fixtures, all from this session's runs, all must stay red-proof:
- chat-speak sample scores worse than good human prose;
- dumb-human collapse scores worse than a conservative de-slop of the same text;
- the three bridge failures ("can", "because", "with"/"to") each rejected;
- word-salad synthesis output rejected;
- the clean Regression 1 output accepted verbatim;
- the six pivot traps each rejected: "the approval of the committee" (subject
  genitive), "a full initialization" (modified nominal), "several
  initializations" (quantified), "create an index" (unlicensed pair),
  "facilitates the integration" (non-light verb), passive "is performed";
- the heading pivot ("Making the Decision" -> "deciding") rejected via gate 7;
- the composed four-operation paragraph output accepted verbatim, including its
  gate-held residue ("simply", "By following these steps,") left in place.

Auto-generated property tests: instantiate each pair with corpus-drawn fillers,
assert output, closure, clause completeness, and a clean parse.

Data protocol: tune on dev, report on the sealed holdout, never tune on holdout.

## Data work (the growth axis)

Coverage grows by growing the inventory and the indexes, not by cleverer search.
The real-corpus paragraph run showed the frontier concretely: "delve into,"
"development journey," and "fortunate enough to have had the opportunity to"
survived a pass not because the machinery fails on them but because the demo
inventory lacked their entries. Each is one mined pair or span away. LVC pair
licensing follows the same recipe: mine (light verb, nominalization) pairs where
the machine corpus prefers the construction and the human corpus the verb.
Critical path: curate well-written human technical docs (the current human corpus
is terse; that register produced the chat-speak envelope), generate per-family
machine corpora with the topic-matched recipe, re-mine and re-index on model
drift. The human side of every pair and every target frequency band comes
exclusively from real human docs; model output is confined to the slop side.

## Migration order

M0 scoring harness plus fixtures. M1 corpus curation, paired generation, `mine`
and `index`, inventory pack v1. M2 pack format with compile-time disjointness and
closure checks. M3 friction-match (DMS plus automaton). M4 chunker, perceptron
tagger, morph tables. M5 friction-edit plus CLI simplification, metric demotion,
envelope re-derivation. Each milestone lands green on fmt, clippy -D warnings,
tests, and the fixtures.

## Research appendix (out of scope for the build)

Recorded, not built: the discourse-conditioned schema transducer (frame-to-frame
synthesis with distribution-matched selection) remains attractive but untested
beyond its paired-substitution core, which is what shipped. The POS-skeleton DMS
channel for cross-family detection is a designed experiment awaiting a run. Both
re-enter the spec only with experimental results behind them.

## Honest limits (README)

Removes machine framing surgically. Does not add ideas, voice, or fluency. Where
it cannot edit safely it tells you instead. Ceiling: a careful copy editor with a
narrow brief. An empty source stays empty. Not for fiction, not a detector-beater,
and detection packs are specific to the model family they were built from.
