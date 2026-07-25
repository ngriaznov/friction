# `perceptron_en.json.gz` — provenance and licensing

This directory's weight artifact backs `PerceptronTagger`, the default
(and only) `Tagger` implementation. Nothing here derives from
LanguageTool/nlprule data, and nothing here is downloaded at build time.
(An `nlprule`-backed comparison backend existed in this crate's history
and was removed once the perceptron was validated against it; the
historical references below describe that one-time offline process.) This file records
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
2. **Drafting.** `NlpruleTagger` (this crate's since-removed
   `nlprule`-feature backend, LGPLv2.1-derived data) tagged every
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

# `gold_dep_en.conllu` — provenance and licensing

This section covers the dependency-parse gold file backing the arc-eager
parser's own weight artifact. Unlike `gold_pos_en.tsv` above, this file is
**silver data**: it is spaCy's own parse of friction's own sentences, run
once offline and mechanically corrected on a small, well-defined set of
closed-class facts. No sentence in it was hand-annotated token by token,
and no claim to the contrary should be read into it — see "Honest limits"
below.

## Source documents

Every sentence comes from this project's own vendored human corpus,
`corpus/human/{docs,readme,blog,email,forum}/*.md` (falling back to
`corpus/quarantine/<genre>/*.md` for the small number of human documents
held there instead), restricted to documents recorded as `class: human`,
`split: train` in `corpus/manifest.jsonl`. 264 such documents exist across
all five genres (blog: 68, docs: 58, readme: 57, email: 30, forum: 51); all
264 resolved to a file on disk and were scanned for candidate sentences.
No `dev`-split or `holdout`-split document was read for this purpose, at
any point. Unlike the POS gold file, sentences are drawn from every genre
the corpus vendors, not the four `gold_pos_en.tsv` used — the parser this
trains is not genre-specific, so more data outweighs matching the tagger's
genre mix exactly.

## Annotation method

1. **Extraction.** `crates/friction-nlp/examples/dump_sentences.rs` ran
   every candidate document through the real `friction-parse` prose
   extraction and `SrxSegmenter` sentence segmentation — the same pipeline
   `gold_pos_en.tsv` and `friction_harness::fragment` both use — tagged
   each sentence with `PerceptronTagger`, and filtered to the tagger's own
   gold-sentence shape (4-40 whitespace-separated words, pure ASCII, no
   `UNKNOWN` sentinel). This produced 27,754 candidate sentences, of which
   15,186 survived the filter (216,185 tokens).
2. **Drafting.** `tools/gold_dep/draft_gold.py` parsed every one of those
   15,186 sentences with spaCy `en_core_web_sm` (MIT-licensed), once,
   offline, as a drafting aid only — the same role `NlpruleTagger` played
   for the POS gold file, without that file's LGPL question, since spaCy
   and its default English model are both MIT.
3. **Alignment.** spaCy tokenizes differently from friction in both
   directions, and a sentence whose token boundaries disagree cannot
   borrow spaCy's head indices (they address the wrong tokens). Two
   things are done about it:

   - **Merging**, when spaCy splits what friction keeps as one token: a
     run of exactly three spaCy tokens `X`, `-`, `Y` spanning one
     friction token merges into one (friction keeps `high-quality` as a
     single token; spaCy splits it), and spaCy's `can`/`not` split of
     `cannot` merges the same way. A merged token's outgoing
     head/relation is taken from whichever original spaCy token has a
     head outside the merged group (for `cannot`, both `can` and `not`
     attach directly to the same external verb in every case observed
     here, so this is not actually ambiguous in practice); if none does,
     the rightmost token in the group stands in. This direction is
     restricted to exactly these two patterns, deliberately: it is a
     judgement call about which token should carry the merged relation,
     and only these two are made unconditionally in this file.
   - **Projecting**, when friction splits what spaCy keeps as one token —
     the more common direction here, since friction's own tokenizer
     (`tokenize()` in `tag_perceptron.rs`) breaks at every letter/digit
     boundary (`S2O` becomes three friction tokens, `S`/`2`/`O`) and
     friction's punctuation scan glues a whole run of punctuation
     characters into one token (`).`  is one friction token where spaCy
     keeps `)` and `.` separate). Unlike merging, this direction is
     applied unconditionally, whenever a single spaCy token's span
     covers a run of two or more consecutive friction tokens: the first
     friction fragment inherits that spaCy token's head and relation
     (through the same label mapping and closed-class corrections every
     other token gets); every other fragment attaches to that first
     fragment with a fixed `other` relation, since friction's tokenizer
     drew that internal boundary, not spaCy's parser, and there is no
     dependency judgement to inherit for it. This is mechanical, not a
     judgement call, so it carries no pattern restriction the way
     merging does.

   Sentences that still disagree token-for-token after both are dropped,
   not patched further; a sentence with more than one spaCy-parsed root
   (spaCy decided the text was more than one sentence, where friction
   already segmented it as one) is treated the same way.

   Measured on the full 15,186-sentence candidate pool: alignment before
   any normalization was 8,544 sentences (56.26%); after merging and
   projection, 11,807 (77.75%) — 948 hyphen-run merges, 39 `cannot`
   merges, and 1,229 spaCy tokens projected onto 4,309 friction fragments
   (3,080 of them synthetic `other`-relation attachments), with 3,379
   sentences dropped as still misaligned. This remains well below the
   97.4% figure originally quoted for this step; that figure turned out
   to be an artifact of how it was measured, not a property of the real
   data — see "On the original 97.4% figure" below. The 3,379 remaining
   drops break down as: punctuation-run fusion (`).`, `>=`, and similar)
   1,233; a possessive `'s` friction keeps glued to its noun (`Python's`,
   `there's`) 699; other contractions (`'re`, `'ll`, `'ve`, `'d`, `'m`)
   345; `n't` (`don't`, `can't`) 314; a number or word fused with
   trailing punctuation beyond a bare `.`/`,` 215; hyphen compounds with
   more than one hyphen (`out-of-the-box`, `end-to-end`) 64; and 149
   miscellaneous cases (leading punctuation, quoted single words). Every
   one of these is structurally the same "spaCy over-split, friction
   didn't" shape the two merge patterns above already handle — just
   outside the two specific patterns this file implements. Generalizing
   the merge direction to be unconditional, the same way projection is,
   would recover essentially all of them (measured: 14,677/15,186,
   96.65%) — quoted here as a measured input to that decision, not
   implemented, since it was intentionally left out of scope for this
   revision.
4. **Bias check.** Before projection existed, sentences dropped for
   misalignment were checked against sentences kept, on the *earlier*,
   merge-only pipeline (59.94% alignment): mean sentence length in
   words, the rate of whitespace-delimited words containing both a
   letter and a digit, and the rate of words containing `_`, `::`, `/`,
   or a non-trailing `.` (identifier- and path-shaped words). Kept
   sentences averaged 10.4 words with a 0.017% letter+digit rate and a
   0.339% identifier-shaped rate; dropped sentences averaged 14.9 words
   (43% longer) with a 1.210% letter+digit rate (71x higher) and a
   1.856% identifier-shaped rate (5.5x higher). The merge-only gold set
   would have been measurably biased against exactly the sentences
   technical documentation is made of — the ones with product names,
   version strings, and code-adjacent tokens. Projection (step 3) was
   added specifically because of this finding, and directly targets its
   largest cause (the letter/digit-boundary splits driving the
   letter+digit gap); the punctuation-run and contraction gaps identified
   in step 3's remaining-drops breakdown are the same kind of bias,
   unaddressed by this revision.
5. **Projectivity.** Of the 11,807 aligned sentences, 85 (0.72%) produced
   a non-projective tree and were dropped — arc-eager, the transition
   system this gold file trains, can only derive projective trees. This
   is below the 5%-of-pool threshold that would have stopped this run
   for reconsideration, and close to (a little above) the 0.94% measured
   on the smaller proxy sample. 11,722 sentences (152,667 tokens)
   survived to the correction stage.
6. **Closed-class correction.** Three overrides replace spaCy's drafted
   relation with one implied unambiguously by the token's own part of
   speech, mirroring the POS gold file's closed-class dictionary pass:
   `det` on any token friction's own tagger calls a determiner (`DT` or
   `PDT`) — 3,417 corrections; `aux` on any token it calls a modal
   (`MD`) — 130 corrections; `mark` on a subordinating conjunction
   introducing a clause — 986 corrections. The third reads spaCy's own
   universal-POS tag (`SCONJ`) rather than friction's, because friction's
   Penn tagset does not distinguish a subordinator from a preposition
   (both are `IN`) and so cannot make that call unambiguously on its
   own; spaCy's finer category can. A synthetic `other`-relation
   attachment created by projection (step 3) is not eligible for these
   overrides — its relation is fixed by construction, not drafted by
   spaCy, so there is nothing to correct. No correction beyond these
   three closed classes was applied — everything else stands exactly as
   spaCy drafted it.
7. **Label mapping.** spaCy's relation labels were mapped to the fixed
   target set `root acl advcl agent amod aux auxpass cc ccomp conj csubj
   det dobj mark nsubj nsubjpass pobj prep xcomp punct` — the ClearNLP/
   OntoNotes relation names `DepRelation` itself uses, not Universal
   Dependencies v2's renamed equivalents; every other spaCy relation
   collapses to `other`, and so does every synthetic projection
   attachment. Over the 11,722-sentence corrected pool (152,667 tokens),
   28,717 tokens (18.81%) collapsed to `other`; the per-relation
   breakdown (over the final sampled file, 3,908 sentences, 51,001
   tokens) is: `acl` 308, `advcl` 763, `agent` 100, `amod` 2,779, `aux`
   2,053, `auxpass` 650, `cc` 1,509, `ccomp` 472, `conj` 1,594, `csubj`
   49, `det` 5,547, `dobj` 2,646, `mark` 779, `nsubj` 2,776, `nsubjpass`
   479, `other` 9,613 (18.85%), `pobj` 4,194, `prep` 4,822, `punct`
   5,396, `root` 3,885, `xcomp` 587.
8. **Sampling.** A fixed stride over the 11,722-sentence corrected pool,
   sorted by document id then sentence index (not a random sample), lands
   the gold set near a 4,000-sentence target: stride 3 yields 3,908
   sentences. The stride is rounded to the nearest whole number rather
   than truncated, landing closer to 4,000 than the next stride down or
   up would have (stride 2 would have landed at 5,861, further away).
9. **Split.** Train/test is assigned by document id, not by sentence: a
   document goes to `test` if the first four bytes of its id's sha256
   digest, taken as an integer, fall in the lowest fifth of the digest
   space, and to `train` otherwise — a fixed, seedless function of the
   id alone, so every sentence from one document always lands in the
   same split. Of the 3,908 sampled sentences, 2,975 (198 documents) are
   `train` and 933 (65 documents) are `test`; the two document sets are
   disjoint by construction, verified empty-intersection at write time.

## On the original 97.4% figure

The alignment rate this file measures (77.75%, up from 59.94% before
projection existed) is well below the 97.4% originally quoted as a
baseline for the two merge normalizations. That figure was not wrong
about the corpus so much as circular about how it was produced: it
reconstructed sentence text from friction's own token list through a
detokenizer, rather than measuring against the sentence's real text. So
when friction's tokenizer split `S2O` into `S`/`2`/`O`, the detokenizer
re-emitted `"S 2 O"`, spaCy tokenized *that* into three tokens too, and
the disagreement disappeared by construction — it was never given the
chance to show up. The 77.75% here is measured against real sentence
text and is the honest number for this corpus; treat the two normalized
figures below alongside it as this drafting step's real ceiling absent
further normalization work, not as a regression from something once
achieved.

## Honest limits

This is drafted, corrected silver data, not hand-annotated gold. Every
relation not covered by the three closed-class overrides in step 6 is
exactly what spaCy's `en_core_web_sm` parser predicted, unreviewed at the
individual-sentence level; the only broad-strokes check this file's own
build applies is the alignment and projectivity filtering above, which
catches disagreement with friction's own tokenizer and with the parser's
own transition system, not disagreement with the true parse. Training
against this file caps parser accuracy near spaCy's own, and any
systematic error spaCy's parser makes is inherited rather than caught.
Treat a transducer that misfires on a specific dependency structure as a
reason to revisit this file before revisiting the transducer.

Separately: the alignment step is a source of measured, directional
sampling bias (step 4, "Bias check"), not just an efficiency loss. It
under-represents exactly the sentence shapes technical documentation is
made of — ones naming products, versions, and code-adjacent identifiers.
Projection closes the largest share of that gap; step 3's
remaining-drops breakdown names the shapes that are still under-sampled
in this revision (mid-token punctuation runs and contractions, chiefly).
A user of this file who cares about parser behavior on identifier-heavy
prose specifically should weight that limitation accordingly, not treat
the 77.75% headline figure as evenly distributed across sentence types.

## License

`gold_dep_en.conllu` derives solely from `corpus/human/**` /
`corpus/quarantine/**`, already covered by this project's own corpus
curation terms, plus relation labels and head indices computed offline
from spaCy `en_core_web_sm` (MIT license) — no spaCy model weights, code,
or training data are embedded in the gold file or in any artifact trained
from it.

## Artifact

- **Path:** `crates/friction-nlp/weights/gold_dep_en.conllu`
- **Format:** one token per line, tab-separated `index surface
  penn_pos head_index relation` (1-based token index; `head_index` is
  the 1-based index of the head token, or `0` for the sentence root),
  preceded by three comment lines (`# sent_id = <doc_id>:<sent_index>`,
  `# split = train|test`, `# text = <sentence text>`) and followed by a
  blank line. Sentences appear sorted by document id then sentence
  index, so the file is byte-reproducible from the corpus alone.
- **Size:** 1,437,474 bytes; 3,908 sentences, 51,001 tokens.
- **sha256:**
  `d3e6238f177ceeaa85e4445580adf1273478a95ea842c9f6b63814f7ce5267be`

## Reproduction

```sh
cargo run -p friction-nlp --example dump_sentences -- corpus \
    | tools/gold_dep/venv/bin/python tools/gold_dep/draft_gold.py \
        --out crates/friction-nlp/weights/gold_dep_en.conllu
```

(`tools/gold_dep/venv` is a local virtualenv with `spacy` and
`en_core_web_sm` installed; any Python environment with the same two
installed works identically, since neither the extraction nor the
drafting step reads anything else from the environment.) Run twice from
the same corpus checkout, this reproduces the sha256 above byte for
byte — verified directly, not merely asserted, before this file was
written up.
