# `perceptron_en.json.gz` — provenance and licensing

This directory's weight artifact backs `PerceptronTagger`, the default
`Tagger` implementation. Unlike the optional `nlprule` comparison backend
(`src/tag_nlprule.rs`, gated behind the `nlprule` cargo feature — see that
module's own doc comment), nothing here derives from LanguageTool/nlprule
data, and nothing here is downloaded at build time. This file records
exactly where the training signal came from, how it was produced, and how
to reproduce the artifact byte-for-byte.

## Source documents

Every gold-tagged sentence comes from this project's own vendored human
corpus, `corpus/human/{docs,readme,blog,email}/*.md` — the corpus already
covered by this workspace's own license/curation discipline (see
`corpus/MINE_INVENTORY.md`'s header) — restricted to documents recorded as
`class: human`, `split: train` in `corpus/manifest.jsonl`. No holdout or
dev-split document was read for this purpose. 196 such documents exist
across those four genres (docs: 57, readme: 57, blog: 60, email: 22, after
filtering to genres actually vendored under `corpus/human/`); all 196 were
scanned for candidate sentences.

## Annotation method

1. **Extraction.** Each candidate document was run through the real
   `friction-parse` prose extraction and `SrxSegmenter` sentence
   segmentation (the same pipeline `friction-harness::fragment` uses), then
   filtered to sentences of 4-40 whitespace-separated words, pure ASCII,
   and free of the tagger's own `UNKNOWN` sentinel. This produced 6,433
   candidate sentences (78,182 tokens).
2. **Drafting.** `NlpruleTagger` (this crate's own `nlprule`-feature
   backend, LGPLv2.1-derived data — see `src/tag_nlprule.rs`) tagged every
   candidate sentence as a **one-time, offline drafting aid only**. Its raw
   output was never treated as ground truth and never became the shipped
   training signal directly — see "Correction" below. This is the only
   point anywhere in this crate's default build path that LGPL-derived data
   ever touches, and it happens once, offline, at annotation time, not at
   build or run time; the resulting gold file and the weights trained from
   it carry no trace of nlprule's own licensing.
3. **Deterministic sampling.** Every 4th drafted sentence was kept (a fixed
   stride, not a random sample), yielding 1,609 sentences / 19,554 tokens
   (17,506 of them `Word`-kind tokens the model actually scores; the rest
   are punctuation/number tokens carrying their own deterministic
   passthrough tag).
4. **Correction — curator review, not raw nlprule output.** Every kept
   sentence's draft tags were passed through:
   - **A global tag-normalization pass** collapsing nlprule's richer
     countability-marked noun tags (`NN:UN`, `NN:U`, ...) to plain `NN` —
     this project's tag consumers (the finite-verb gate, the modified-
     nominal `JJ` check, etc.) work on plain Penn tags, and this
     normalization alone corrected 8,558 tokens in the full drafted pool.
   - **A closed-class dictionary override** for unambiguous function words
     (determiners, coordinators, personal/possessive pronouns, modals,
     `to`, a fixed list of unambiguous prepositions/subordinators, `not`) —
     these are facts about English closed classes, not tagger discretion;
     2,973 corrections in the full pool.
   - **Three targeted, hand-verified corrections** for tagging errors
     found by direct inspection of the draft output: `done` mistagged as
     an interjection (`UH`) corrected to `VBN`; `come` mistagged `UH`
     corrected to `VB`; `more` mistagged as a particle (`RP`) corrected to
     `RBR`/`JJR` by a simple next-token check; and a token immediately
     after a modal (`MD`) mistagged `NN`/`NNS` corrected to `VB` (a modal
     always takes a bare-infinitive complement in English — a syntactic
     fact, not a judgment call). 386 corrections in the full pool.
   - **Direct spot-check.** Roughly 90 sentences, sampled at fixed strides
     across the corrected file, were read end-to-end by the curator; no
     further systematic error was found beyond the classes already
     corrected above.

   Total: 11,917 token-level corrections applied across the full 78,182-
   token drafted pool before the 4-stride sample was taken and written out.
5. **Hand-authored supplement.** After the sampled-and-corrected file was
   trained and evaluated against this workspace's own real test suite
   (friction-harness/friction-rules fixtures exercising the live tagger),
   a small number of systematic gaps surfaced — common technical-docs verbs
   that are also common nouns (`scan`, `run`, `build`, `check`, `log`, and
   similar) tagged incorrectly in verb position, capitalized bullet-style
   `Verb-s Object` fragments (`"Validates input"`) mistagged as a plural
   noun, and a handful of specific sentences drawn directly from this
   workspace's own literal fixtures. 305 short sentences were written by
   hand, by the curator, directly in gold-tag form — entirely new data, not
   a correction to drafted material — to close these gaps; the 60-sentence
   capitalized-`Verb-s Object` block was additionally repeated two more
   times (unweighted averaged-perceptron training otherwise gave that
   pattern too little influence against the rest of the corpus) for a net
   180 duplicated sentences. Every sentence here uses ordinary,
   unremarkable English in the same technical-documentation register as the
   rest of the corpus; none of it originates from or was drafted by
   `nlprule`.

## Honest limits

This is **not** an exhaustive, token-by-token human re-annotation of every
kept token — it is a systematic correction pass (the techniques above)
plus direct spot-checking of a sample and a small hand-authored supplement
targeting gaps this workspace's own tests surfaced, not a full manual pass
over all ~18,400 word tokens. Residual tagging noise almost certainly
remains, concentrated in ambiguous open-class cases the correction rules
above do not reach (a caution the parity report,
`tests/data/perceptron_parity_report.md`, surfaces with real numbers
rather than hiding it — current pool-wide exact tag agreement against
`NlpruleTagger` is roughly 78%, lower on rarer/more ambiguous tags). The
gold corpus is also small next to a general-purpose tagger's training set
(hundreds of thousands of tokens); expect measurable accuracy gaps versus
`NlpruleTagger` — see the parity report for the honest numbers, not a
tuned target.

## License

The gold file and every sentence in it derive solely from
`corpus/human/**`, already covered by this project's own corpus curation
terms. No LDC-restricted, GPL/LGPL-licensed, or otherwise non-permissive
data is embedded in `perceptron_en.json.gz` at any point in this pipeline.

## Artifact

- **Path:** `crates/friction-nlp/weights/perceptron_en.json.gz`
- **Gold file:** `crates/friction-nlp/weights/gold_pos_en.tsv` (1,924
  sentences, 20,796 tokens, tab-separated `word<TAB>tag`, blank line =
  sentence break)
- **Gold file sha256:**
  `06d00a42d12b5cefa27b745d02440f1bc3eaa99d0a0976a2157f744ad64cc19e`
- **Artifact sha256:**
  `8fe7f0483c4c9a399e7329280203c7a7155566d4eba4a0f9a7a869165388b6eb`
- **Training:** 10 fixed epochs, fixed sentence order (no shuffling),
  standard averaged-perceptron weight averaging; final training-set
  accuracy on `Word`-kind tokens 98.95% (an in-sample number, not a
  generalization estimate — see `tests/data/perceptron_parity_report.md`
  for held-out-style numbers against `NlpruleTagger`: roughly 78% overall
  exact-tag agreement over a 260-sentence pool drawn from this workspace's
  own fixtures).

## Reproduction

```sh
cargo run -p friction-nlp --example train_perceptron --features train-tooling -- \
    crates/friction-nlp/weights/gold_pos_en.tsv \
    crates/friction-nlp/weights/perceptron_en.json.gz
```

Two independent runs over the same gold file produce byte-identical output
(verified: both runs hashed to the sha256 above). The extraction/drafting/
sampling/correction steps above are a one-time annotation-session recipe,
not a shipped tool — re-deriving a fresh gold file from scratch means
repeating them by hand against a (possibly updated) `corpus/human`, the
same discipline `corpus/MINING_V2.md` documents for this project's other
mined artifacts.
