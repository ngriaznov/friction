# Weight artifacts — provenance and licensing

> **The gold files this document describes are not tracked in git.**
> `gold_pos_en.tsv` and `gold_dep_en.conllu` are derived data: no build
> reads them (only the `train-tooling` examples do), and each regeneration
> would add several megabytes to history permanently. The artifacts trained
> *from* them — `perceptron_en.json.gz` and `parser_en.json.gz` — are
> committed, because they are compiled into the binary.
>
> Rebuild them from `corpus/` with the pipeline described below, using the
> pinned environment in `tools/requirements.txt`. The pin matters: a
> different model version parses differently and would silently produce
> different gold, so "byte-reproducible" holds only against spaCy 3.8.14
> with `en_core_web_sm` 3.8.0.

This directory's weight artifact backs `PerceptronTagger`, the default
(and only) `Tagger` implementation. Nothing here derives from
LanguageTool/nlprule data, and nothing here is downloaded at build time.
This file records exactly where the training signal came from, how it was
produced, and how to reproduce the artifact byte-for-byte.

## Derived binary artifacts (`.bin`)

`parser_en.json.gz` and `perceptron_en.json.gz` remain the audited
interchange artifacts this file's provenance sections describe — nothing
below changes what they are or how they're reproduced. `parser_en.bin`
and `perceptron_en.bin` are **derived** artifacts, committed alongside
them, built by `corpus-tool weights-pack` from those exact files.

**Format version 2** (current): a serialized *hash table*, not a
serialized `HashMap`. Version 1 (see git history) `postcard`-encoded the
in-memory `WeightFile` and decoded it into a real `std::collections::HashMap`
at every process start — hashing and heap-allocating every one of tens to
hundreds of thousands of feature-string keys dominated `friction fix`'s
startup cost far more than the postcard decode itself did. Version 2
instead packs the hash table's own on-disk layout — a fixed-seed
(`xxh64`), open-addressed, linearly-probed table over a contiguous string
pool, plus a sparse per-feature `(class/action index, weight)` row region
(most features carry only a handful of nonzero weights out of the full
class/action space, so a dense row per feature was rejected as an
artifact-size multiplier for no benefit) — embedded **raw, uncompressed**.
`PerceptronTagger::new`/`PerceptronParser::new` load it as a zero-copy view
straight over the `include_bytes!`'d static: no `HashMap` is built, no
feature string is re-hashed, no per-feature array is allocated, and (being
uncompressed) nothing is decompressed either. This mirrors
`friction-packs/packs/jargon-attest-v1.bin`'s own embedded `BinaryFuse8`
filter — bytes in, a view out, zero construction — which is itself the
precedent this format follows. See `friction_nlp::weights_bin`'s module
docs for the shared header/hash-table primitives and each of
`tag_perceptron`/`dep_perceptron`'s own module docs for the exact per-
artifact payload layout.

Measured on this workspace's own machine (see this crate's `README`/task
history for the full numbers): switching from version 1 to version 2 cut
the combined tagger+parser *load* time from roughly 67ms to a small
number of milliseconds — construction, not decoding, was the entire
version-1 cost, and version 2 has none. The trade is committed artifact
size: raw (uncompressed) storage of an open-addressed table at a ~0.7
load factor is bigger than a gzip-compressed dense postcard dump, so the
`.bin` pair grew from 7.06MB (version 1: `parser_en.bin` 5.81MB +
`perceptron_en.bin` 1.24MB) to substantially more (version 2: see the
sizes `corpus-tool weights-pack`'s own summary line prints after a
regeneration) — raw was chosen over gzip because per-request decompression
time for an artifact this size risked eating back the load-time win this
format exists for; this trade-off is revisitable if the size delta proves
too costly in practice.

The header's sha256 is checked against a compile-time constant of the
currently embedded `json.gz` at every `new()` call — this is what makes a
`.bin` regenerated from a stale or hand-edited `json.gz` (or simply
forgotten after a retrain) fail loudly at process init, rather than
silently drifting from the audited source. A `from_json_gz` constructor
on both `PerceptronTagger` and `PerceptronParser` keeps the JSON-loading
path available (used by the converter below and by each module's own
parity tests: a fixed-sentence-battery behavioral parity check, plus a
full-vocabulary equivalence check iterating every feature/tagdict/class
entry the source `json.gz` describes against what the packed view
returns, including a handful of keys known to be absent).

**Regenerate both `.bin` files** after retraining either artifact (or
whenever `cargo test` reports a stale-artifact or sha256-mismatch
failure):

```sh
cargo run -p corpus-tool -- weights-pack
```

This reads both `json.gz` files, computes each one's sha256, and writes
`parser_en.bin`/`perceptron_en.bin` next to them (see `corpus-tool
weights-pack --help` for the input/output path flags, defaulted to this
directory). Deterministic: running it twice over the same `json.gz` bytes
produces bit-identical `.bin` bytes (pinned by `friction-nlp`'s own
`pack_perceptron_*_bin_is_deterministic` tests and `corpus-tool`'s
`packing_the_same_json_gz_twice_is_byte_identical`).

**This section describes the second generation of this gold file and
artifact** (spaCy-drafted, 11x the token volume, a closed punctuation
tagset). The first generation (nlprule-drafted, hand-curated, ~20K tokens)
is described in this repository's git history rather than kept here
alongside a scheme it no longer matches. The "Honest limits" section below
still applies the same discipline of surfacing real numbers rather than a
tuned target.

## Tagset decision: closed Penn/spaCy punctuation tags, not surface text

The tagger used to give every punctuation token its own literal surface
text as its tag (`"!!,"`, `"..."`, `"({},"`, and so on — genuinely
one-of-a-kind strings, since friction's tokenizer groups a whole maximal
run of punctuation characters into one token, unlike real Penn-Treebank/
CoNLL-U tokenization, which always isolates a single character). Measured
before this change: 178 distinct tags in total, of which roughly 140 were
punctuation passthrough strings — meaning any downstream consumer that
treats a tag as a syntactic class (a dependency parser's own features, a
skeleton/n-gram pack keyed on coarse tags) instead saw a near-unique
fingerprint per punctuation token.

**Decision: adopt Penn/spaCy's closed punctuation tagset** — `.` `,` `:`
`-LRB-` `-RRB-` `` ` `` `''` `HYPH` `NFP` — computed from a punctuation
token's own character *set*, never its literal text (see
`punctuation_tag` in `src/tag_perceptron.rs` for the exact classification
rule, and its direct Python port in `tools/gold_pos/draft_pos.py` used to
draft gold labels the same way). The alternative (collapsing every
punctuation token to one single tag) was rejected: it would have thrown
away the real distinction between, say, a closing paren and a colon, which
several existing consumers care about structurally. Adopting spaCy's own
scheme instead of inventing a new closed set means the POS gold file's
`Word`-token tags (drafted directly from spaCy's `tag_`) and its
non-`Word` passthrough tags are already drawn from the same vocabulary —
no separate reconciliation needed between "what spaCy calls a preposition"
and "what this scheme calls a preposition."

**Before changing this, every consumer of tag values across the workspace
was checked** (`grep` across `crates/` for `pos.as_str()`, `PosTag::new`,
`coarse_tag`, and specifically `friction-edit/src/gates.rs`,
`friction-nlp/src/chunk.rs`, `friction-nlp/src/lvc.rs`,
`friction-metrics/`): every consumer matches on a small fixed set of
alphabetic tags (`VB*`, `NN*`, `JJ*`, `RB*`, `CC`, `DT`, `PDT`, `WDT`,
`PRP$`, `CD`, `TO`, `MD`, ...) and none depends on punctuation carrying its
own surface text. `coarse_tag` in `src/tag.rs` (which returns a punctuation
tag verbatim, since its first character is not alphabetic) needed no
change either: the new punctuation tags are already members of a small
closed set themselves, so "return it verbatim" is now exactly the right
coarse behavior, not a loophole. The one real consumer of the *old*
scheme's surface-text tags — `friction-packs/packs/attestation-v1.toml`'s
skeleton pack, built by `corpus-tool attest` calling `coarse_tag` over
every token including punctuation — goes stale from this change and is
regenerated separately, outside this task's scope (see that pack's own
provenance header).

**The tagset is small and closed: 49 distinct tags** in the current gold
file (verified by direct count, not estimated):

- **9 punctuation tags:** `.` `,` `:` `-LRB-` `-RRB-` `` ` `` `''` `HYPH`
  `NFP`
- **2 deterministic passthrough tags:** `CD` (any `Number`-kind token),
  `SYM` (any `Symbol`-kind token — never a punctuation token; disjoint from
  the 9 above)
- **38 word tags**, all standard Penn-Treebank/OntoNotes tags as spaCy's
  `en_core_web_sm` emits them: `CC CD DT EX FW IN JJ JJR JJS MD NN NNP
  NNPS NNS PDT POS PRP PRP$ RB RBR RBS RP TO UH VB VBD VBG VBN VBP VBZ WDT
  WP WP$ WRB ADD AFX LS XX $` (the last five — `ADD` a URL/email-shaped
  token, `AFX` an affix, `LS` a list-item marker, `XX` unclassifiable, `$`
  a currency symbol — are OntoNotes extensions to the original 1990 Penn
  tagset that spaCy also emits; kept rather than collapsed, since the goal
  is "adopt spaCy's own scheme," not "adopt a subset of it").

No punctuation token's tag is ever its own surface text under the new
scheme, by construction (`punctuation_tag` returns one of the 9 fixed
strings above, never `surface` itself, no matter what characters the
token's maximal punctuation run contains).

## Source documents

Every gold-tagged sentence comes from this project's own vendored human
corpus, `corpus/human/{docs,readme,blog,email,forum}/*.md` (falling back to
`corpus/quarantine/<genre>/*.md` for the small number of human documents
held there instead), restricted to documents recorded as `class: human`,
`split: train` in `corpus/manifest.jsonl`. No holdout or dev-split document
was read for this purpose, at any point. 264 such documents exist across
all five genres (blog: 68, docs: 58, readme: 57, email: 30, forum: 51) —
the same 264-document pool `gold_dep_en.conllu` draws from, not the
narrower four-genre, 196-document pool the first-generation POS gold file
used; nothing about POS tagging is genre-specific, so there is no reason to
leave a fifth of the available signal on the table.

## Annotation method

1. **Extraction.** `crates/friction-nlp/examples/dump_sentences.rs` ran
   every candidate document through the real `friction-parse` prose
   extraction and `SrxSegmenter` sentence segmentation, tagged each
   sentence with the (previous-generation) `PerceptronTagger`, and filtered
   to sentences of 4-40 whitespace-separated words, pure ASCII, and free of
   the tagger's own `UNKNOWN` sentinel. This produced 27,754 candidate
   sentences, of which 15,186 survived the filter (216,185 tokens) — 11x
   the first-generation gold file's 19,554-token drafted pool.
2. **Drafting.** `tools/gold_pos/draft_pos.py` parsed every one of those
   15,186 sentences with spaCy `en_core_web_sm` (MIT-licensed), once,
   offline, as a drafting aid only — reusing
   `tools/gold_common/align.py`'s bidirectional merge/projection alignment
   machinery (factored out of, and shared with, `tools/gold_dep/draft_gold.py`
   rather than reimplemented).
3. **Alignment.** Measured on the full 15,186-sentence candidate pool:
   14,904 sentences (98.14%) aligned after merging and projection; 282
   dropped as still misaligned after both. This is markedly higher than
   `gold_dep_en.conllu`'s 77.75% on the same underlying corpus, because POS
   drafting needs no single-root/projectivity precondition (a dependency
   fact, irrelevant to tagging a token in isolation) — `sentence_aligns`
   from `gold_common.align` is used directly, without the dependency
   file's extra single-root filter.

   Merges by shape (5,682 total, every shape accepted unconditionally):
   `punct_run` 2,122; `hyphen_single` 1,305; `possessive_s` 796;
   `contraction_other` 423; `contraction_nt` 377;
   `number_or_word_plus_trailing_punct` 295; `other` 238; `hyphen_chain`
   80; `cannot` 46. Projections: 1,970 spaCy tokens projected onto 7,112
   friction fragments (5,142 fragments beyond each group's first).
4. **Per-token-kind tagging rule**, applied uniformly regardless of
   alignment-group shape:
   - A `Number`-kind or `Symbol`-kind friction token always gets its
     deterministic passthrough tag (`CD`/`SYM`) — spaCy's opinion is never
     even consulted for these, matching the same invariant
     `PerceptronTagger::tag` and `examples/train_perceptron.rs`'s
     consistency check enforce at inference and training time.
   - A `Punctuation`-kind friction token always gets `punctuation_tag`'s
     classification of its own surface text (see the tagset decision
     above) — likewise independent of spaCy's own tokenization of the same
     span.
   - A `Word`-kind friction token in a `"single"` alignment group takes its
     aligned spaCy token's `tag_` directly.
   - A `Word`-kind friction token produced by **projection** (one spaCy
     token's span covers several friction tokens — friction's own
     tokenizer over-split, e.g. `S2O` -> `S`/`2`/`O`): the first friction
     fragment inherits that spaCy token's tag; every other fragment that is
     *also* `Word`-kind inherits the exact same tag (there is no
     independent spaCy judgement about a sub-fragment friction's own
     tokenizer invented, so propagating preserves the most information
     available, rather than substituting an arbitrary placeholder). A
     `Number`/`Punctuation`/`Symbol`-kind fragment still takes its own
     deterministic tag regardless, per the point above.
   - A `Word`-kind friction token produced by a **merge** (several spaCy
     tokens collapse onto one friction token — `cannot`, a hyphen compound,
     a contraction, a possessive): the tag is the *first* spaCy fragment's
     tag, verbatim. This matches genuine Penn-Treebank convention exactly
     for the two most common shapes (`cannot` kept as one token is tagged
     `MD` in real PTB, and the first fragment is exactly `can`'s own tag; a
     possessive like `Python's` inherits the head noun's tag), at a known
     cost for hyphen compounds acting adjectivally before a noun
     (`state-of-the-art solution`), where the first fragment (typically
     `NN`) is not always the compound's true category (`JJ`). Hyphen
     compounds are 1,385 of the 5,682 merges (~24%); this is a documented
     limitation, not an oversight.
5. **Closed-class correction pass**, over `Word`-kind tags only, applied
   after the above: a short, **exact-case** (never case-folded) dictionary
   — `a`/`an`/`the` -> `DT`, `not` -> `RB`, `and`/`or`/`nor` -> `CC`,
   unambiguous personal pronouns (`i you he she it we they me him them
   us`) -> `PRP`, unambiguous possessive determiners (`my your our their
   its his`) -> `PRP$` — plus one structural rule (a token immediately
   after a modal is a bare infinitive in English, so `NN`/`NNS` there
   becomes `VB`). Two things this deliberately does *not* do, both found by
   direct measurement rather than assumed:
   - **Matching must be exact-case.** A first draft case-folded the
     dictionary (matching `surface.lower()`); measured on this corpus, that
     fired 2,109 "corrections", almost all wrong — capitalized abbreviations
     and acronyms that happen to share letters with a closed-class word
     (`US`, matched against lowercase `"us"`, forced from spaCy's correct
     `NNP` to `PRP` 47 times; a standalone capital `A` as a list marker,
     matched against `"a"`, forced to `DT` 11 times; similarly for `I`/`OR`-
     shaped tokens). Exact-case matching eliminates this class entirely:
     re-measured, the same corpus produces only 61 corrections total (`CC`
     10, `DT` 12, `PRP` 29, `PRP$` 1, modal-followup 9), every one checked
     and a genuine spaCy tagging slip, mostly in markdown-emphasis-adjacent
     contexts (`**a`) that visibly confuse spaCy's own tokenization context.
   - **`to` is deliberately excluded**, despite genuine Penn Treebank
     convention tagging every occurrence of the word "to" as `TO`
     regardless of function. spaCy's `en_core_web_sm` does not follow that
     convention — it tags infinitival "to" (`I want to go`) `TO` but
     prepositional "to" (`I went to Paris`) `IN`, a real, useful
     distinction this workspace's own `gates.rs`/`chunk.rs` rely on (they
     look for `TO` specifically to find an infinitive-marker construction,
     exactly spaCy's `TO` sense). Measured directly: forcing every literal
     "to" to `TO` would have overridden 1,955 of spaCy's own correct,
     more-informative distinctions — a judgement call this drafting
     process should not make unilaterally, so it does not.
   - English modals (`will would may might must`) are excluded from the
     dictionary entirely: each has a real noun/main-verb reading ("a
     must-have", "his last will", "the might of the empire"), so a blind
     lexeme override risks introducing exactly the class of error this pass
     exists to prevent; spaCy's own context-sensitive tagging is trusted
     instead. `her` is excluded too (ambiguous `PRP`/`PRP$` by
     context: "I see her" vs "her book"); `his` is kept despite the same
     surface shape, since English has no morphologically distinct absolute
     form for it the way it does for "hers".
6. **No stride-sampling: the entire aligned pool is used.** The
   first-generation gold file was stride-sampled to 1,609 sentences because
   every sentence was hand-reviewed — a real, scarce review budget. This
   file is drafted and corrected entirely mechanically; nothing in it is
   reviewed sentence by sentence, so there is no budget to ration, and
   subsampling would be pure data loss (exactly the mistake
   `gold_dep_en.conllu`'s first revision made and then reversed, at a
   measured cost of several points of parser accuracy).
7. **Split.** Train/test is assigned by document id, not by sentence,
   using the same deterministic hash `tools/gold_common/align.py::doc_split`
   also gives `gold_dep_en.conllu` (80/20, seedless, a pure function of the
   document id — so every sentence from one document always lands in the
   same split). Of the 14,904 aligned sentences: 11,251 train (159,695
   tokens, 198 documents), 3,653 test (51,292 tokens, 65 documents),
   document overlap verified empty at write time. The first-generation file
   had **no held-out split at all** — this is the first time this tagger's
   accuracy has an honest, unseen-data measurement (see "Artifact" below).

## Gold-file format

Preserves the existing `word<TAB>tag` format exactly (one token per line,
blank line = sentence break) and adds exactly one new line per sentence: a
`# split=train` or `# split=test` comment immediately before that
sentence's tokens. This is additive, not a breaking format change:
`parse_gold_file` already skips any line with no tab character without
disturbing sentence boundaries (a comment line matches that exactly), so
an unmodified reader recovers the identical sentences it always did, never seeing the marker. `parse_gold_file_split` is the new,
split-aware reader `examples/train_perceptron.rs` (train-split only) and
`examples/eval_perceptron.rs` (test-split only) actually use — see
`src/tag_perceptron.rs`'s `train_support` module.

## Honest limits

This is drafted, corrected silver data, not hand-annotated gold — the same
honest framing `gold_dep_en.conllu` gives itself. Every `Word`-kind tag not
covered by the closed-class correction pass above is exactly what spaCy's
`en_core_web_sm` tagger predicted, unreviewed at the individual-sentence
level. A few tags show real disagreement even under the new, shared closed
scheme rather than a scheme artifact: `-RRB-` (spaCy sometimes treats a
trailing close-paren as part of a larger punctuation-run token differently
than friction's own tokenizer groups it), `''` (quote-direction judgement
calls), and `NFP` (spaCy's own bucket for irregular punctuation runs
doesn't always land on the same run friction's tokenizer isolated) — see
this crate's own before/after per-tag agreement table (below) for the
honest numbers on these, not a tuned target. Separately, at least one
outright spaCy tagging slip survived uncorrected into the gold file (a rare
identifier token mistagged `$`, 2 occurrences out of 210,987 tokens) —
below the threshold this drafting process treats as worth a targeted
correction rule, but worth naming rather than silently absorbing.

## License

The gold file and every sentence in it derive solely from
`corpus/human/**` / `corpus/quarantine/**`, already covered by this
project's own corpus curation terms, plus part-of-speech tags computed
offline from spaCy `en_core_web_sm` (MIT license) — no spaCy model
weights, code, or training data are embedded in the gold file or in any
artifact trained from it.

## Artifact

- **Path:** `crates/friction-nlp/weights/perceptron_en.json.gz`
- **Gold file:** `crates/friction-nlp/weights/gold_pos_en.tsv` (14,904
  sentences, 210,987 tokens — 180,360 of them `Word`-kind tokens the model
  actually scores; 11,251/159,695 train, 3,653/51,292 test)
- **Training:** 10 fixed epochs, fixed sentence order (no shuffling),
  standard averaged-perceptron weight averaging, train-split sentences
  only (`parse_gold_file_split(text, "train")` — test-split sentences are
  never seen during training); final training-set accuracy on `Word`-kind
  tokens 98.23%.
- **Held-out test-split accuracy** (`examples/eval_perceptron.rs`, full
  per-tag precision/recall/F1 table in this crate's own development
  history / task report — reproduce with the command below): **94.50%**
  overall tag accuracy (48,472/51,292), against gold labels the model
  never trained on. Per-tag F1 for the tags this retraining specifically
  targeted: `VBG` 0.9286 (support 1,051), `VBN` 0.9032 (1,128), `NNP`
  0.8766 (3,543), `RB` 0.9263 (1,799), `VBP` 0.9353 (1,062), `VB` 0.9402
  (2,673), `IN` 0.9738 (5,549), `WDT` 0.9503 (303), `TO` 0.9514 (947), `CC`
  0.9941 (1,447), `JJ` 0.8714 (2,798), `NN` 0.9081 (7,156). Tags with fewer
  than 50 held-out occurrences (`ADD`, `FW`, `LS`, `PDT`, `POS`, `RBS`,
  `WP$`, `XX`, `` ` ``) are measured but too thin to trust individually.
- **spaCy-agreement, before vs. after retraining** (raw per-token tag
  agreement between friction's own tagger output and spaCy `en_core_web_sm`,
  aligned the same way as gold-drafting, over the full 210,987-token
  aligned candidate pool — the same methodology applied identically before
  and after, for a direct comparison; not a byte-identical reproduction of
  whatever exact sample produced this task's original 76.19% figure, but a
  closely matching re-measurement, 75.85%, on the same corpus): **75.85% ->
  94.14%** overall. The tags this retraining specifically targeted:
  `VBG` 52.10% -> 96.43%, `VBN` 54.60% -> 97.72%, `NNP` 47.53% -> 84.06%,
  `RB` 48.71% -> 96.24%, `VBP` 54.60% -> 97.15%, `VB` 66.01% -> 97.84%,
  `IN` 75.89% -> 99.06%, `WDT` 0.67% -> 98.82%. `VBG` — the tag downstream
  present-participle detection tests for exactly — moved the most of any
  tag in absolute terms, from worse than a coin flip to effectively solved
  at this corpus's scale.

## Reproduction

```sh
# 1. Dump friction's own tokenization/tags for every candidate sentence.
cargo run --release -p friction-nlp --example dump_sentences -- corpus \
    > sentences.jsonl

# 2. Draft gold tags with spaCy, align, correct, split, and write the file.
tools/gold_dep/venv/bin/python tools/gold_pos/draft_pos.py \
    --out crates/friction-nlp/weights/gold_pos_en.tsv \
    < sentences.jsonl

# 3. Retrain (train-split sentences only).
cargo run --release -p friction-nlp --example train_perceptron --features train-tooling -- \
    crates/friction-nlp/weights/gold_pos_en.tsv \
    crates/friction-nlp/weights/perceptron_en.json.gz

# 4. Measure held-out accuracy (test-split sentences only).
cargo run --release -p friction-nlp --example eval_perceptron --features train-tooling -- \
    crates/friction-nlp/weights/gold_pos_en.tsv
```

(`tools/gold_dep/venv` is a local virtualenv with `spacy` and
`en_core_web_sm` installed; any Python environment with the same two
installed works identically.) Step 1 depends on whichever `PerceptronTagger`
weights are currently embedded only for its own gold-shape sentence filter
(word count, ASCII, no `UNKNOWN`), not for anything drafted into the gold
file itself — every gold tag comes from spaCy or a deterministic
passthrough rule, never from friction's own (previous-generation) tagger.

# `gold_dep_en.conllu` — provenance and licensing

This section covers the dependency-parse gold file backing the arc-eager
parser's own weight artifact. Unlike `gold_pos_en.tsv` above, this file is
**silver data**: it is spaCy's own parse of friction's own sentences, run
once offline and mechanically corrected on a small, well-defined set of
closed-class facts. No sentence in it was hand-annotated token by token,
and no claim to the contrary should be read into it. See "Honest limits"
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
   borrow spaCy's head indices (they address the wrong tokens). Both
   directions are handled, unconditionally, symmetrically:

   - **Merging**, when a run of consecutive spaCy tokens spans exactly
     one friction token — spaCy splits what friction keeps as one:
     hyphen compounds (`high-quality`), `cannot`, possessives
     (`Python's`), contractions (`don't`, `we're`, `I'll`), punctuation
     runs (`).`, `>=`), a number or word fused with trailing punctuation,
     all of it. Accepted regardless of which characters are involved —
     an earlier revision of this step restricted merging to just the
     hyphen and `cannot` patterns, treating everything else as a drop;
     that restriction was this file's own conservatism, not a linguistic
     one, and is gone. A merged token's outgoing head/relation is taken
     from whichever original spaCy token has a head outside the merged
     group; if none does, the rightmost token in the group stands in —
     except that a token which is itself the sentence's root always wins
     that choice outright, even over an ordinary external-head token
     elsewhere in the same group (see `merge_representative` in
     `draft_gold.py` for the two concrete cases this mattered for: a
     sentence-initial contraction like `Let's`, where `Let` is the root
     and `'s` points within the group so neither looked "external" under
     a naive check; and a group like `pre-installed` that contains both
     the root, `installed`, and a token with a genuine external head,
     `pre`, where naive left-to-right tie-breaking picked the wrong one).
   - **Projecting**, when one spaCy token's span covers a run of two or
     more consecutive friction tokens — friction's own tokenizer is the
     one that over-split: it breaks at every letter/digit boundary
     (`S2O` becomes three friction tokens, `S`/`2`/`O`) and folds a
     trailing `.`/`,` into an adjacent number. The first friction
     fragment inherits that spaCy token's head and relation (through the
     same label mapping and closed-class corrections every other token
     gets); every other fragment attaches to the first fragment with a
     fixed `other` relation, since friction's tokenizer drew that
     internal boundary, not spaCy's parser, and there is no dependency
     judgement to inherit for it.

   Both directions are the same operation facing opposite ways, so both
   are unconditional now; only irreconcilable spans are
   dropped, and a sentence where spaCy parsed more than one root (it
   decided the text was more than one sentence, where friction already
   segmented it as one) is dropped the same way.

   Measured on the full 15,186-sentence candidate pool, at each stage of
   normalization: raw (no normalization) 8,544 (56.26%); +merge only
   11,002 (72.45%); +merge+projection 12,147 (79.99%, spaCy's own
   standalone-whitespace tokens — see below — not yet filtered here);
   +merge+projection+whitespace-token fix, the final figure, 14,621
   (**96.28%**), with 565 sentences dropped as still misaligned. 5,365
   merges were made, by shape: `punct_run` 1,902, `hyphen_single` 1,275,
   `possessive_s` 780, `contraction_other` (`'re`/`'ll`/`'ve`/`'d`/`'m`)
   410, `contraction_nt` 362, `number_or_word_plus_trailing_punct` 286,
   `hyphen_chain` (more than one hyphen: `out-of-the-box`) 79, `cannot`
   44, and 227 miscellaneous. Separately, 1,831 spaCy tokens were
   projected onto 6,414 friction fragments (4,583 synthetic
   `other`-relation attachments).

   spaCy, unlike friction's own tokenizer (which never emits a
   whitespace-kind token), sometimes emits a standalone whitespace token
   — reliably for a literal embedded newline in a soft-wrapped source
   paragraph. Such a token has no friction counterpart and is never any
   other token's head (verified: zero exceptions across the corpus), so
   it is filtered out before alignment rather than walked over; this
   alone raised the aligned count by about 2,100 sentences and is a
   correctness fix, not a normalization choice.

   The remaining 565 drops: 335 are the multi-root case above (spaCy
   itself disagreed with friction about the sentence boundary — often on
   fragmentary extraction artifacts, e.g. `*No RFCs were approved this
   week.*` from surrounding markdown emphasis markup); 201 are friction
   tokens that don't reduce to a single spaCy token even after
   projection (multi-part identifiers spaCy itself further subdivides
   unpredictably — IP-and-port-shaped strings, version numbers with
   several dots, `$PWD/../openssl/build/lib/pkgconfig`-style paths); 29
   are the mirror case for merging. None of these were special-cased
   further; they are the irreconcilable remainder, not a gap
   left on purpose.
4. **Bias check, twice.** Before projection or the merge generalization
   existed, sentences dropped for misalignment were checked against
   sentences kept — on the *original*, hyphen/`cannot`-only pipeline
   (59.94% alignment) — on three measures: mean sentence length in
   words, the rate of whitespace-delimited words containing both a
   letter and a digit, and the rate of words containing `_`, `::`, `/`,
   or a non-trailing `.` (identifier- and path-shaped words). Kept
   sentences averaged 10.4 words with a 0.017% letter+digit rate and a
   0.339% identifier-shaped rate; dropped sentences averaged 14.9 words
   (43% longer), 1.210% letter+digit (71x higher), 1.856%
   identifier-shaped (5.5x higher) — a gold set built that way would
   have been measurably biased against exactly the sentences technical
   documentation is made of.

   The same check was repeated after projection and the generalized
   merge (96.28% alignment): kept sentences now average 12.1 words with
   a 0.500% letter+digit rate and a 0.886% identifier-shaped rate;
   dropped sentences average 13.6 words (12% longer, down from 43%),
   2.703% letter+digit (5.4x higher, down from 71x), 5.303%
   identifier-shaped (6.0x higher — essentially unchanged as a *ratio*,
   though both absolute rates roughly doubled or more). Read together:
   generalizing the merge direction closed most of the letter+digit gap,
   because that gap was mostly the same letter/digit-boundary splits
   projection and the general merge rule both now handle. It did **not**
   close the identifier-shaped gap by the same proportion, because the
   565 sentences still dropped (step 3's breakdown) concentrate in
   exactly the multi-part, punctuation-and-slash-heavy constructs —
   paths, version strings, command invocations — that no 1:N or N:1
   token-count reconciliation can resolve; those need actual parsing of
   the internal structure, not alignment. The honest summary: the bias
   this drafting step introduces is substantially smaller than it was,
   not eliminated, and what remains is concentrated in exactly the
   sentence shapes named here.
5. **Root/closed-class-correction guard.** The three closed-class
   corrections in step 6 are all head-relative facts ("this token is a
   determiner *of its head*"); the sentence root has no head, so none of
   them can coherently apply to it. An earlier revision applied them
   unconditionally, which produced 23 gold tokens with `head == 0` (a
   root) and a relation of `aux`, `det`, or `mark` instead of `root` —
   internally inconsistent, and invisible to any consumer that doesn't
   specifically check for it (a training loop comparing `(head,
   relation)` pairs would see `(None, aux)` where it expected `(None,
   root)`, fail to reproduce that sentence's gold, and silently drop it
   with no error pointing back to the cause). Fixed: a token that is
   itself the sentence root always gets relation `root`, unconditionally,
   with no closed-class correction applied. Every correction the guard
   suppressed is still tallied (into `suppressed_root_det`/`_aux`/`_mark`
   in `draft_gold.py`), so the gap between "what the closed-class rule
   would say" and "what a root can coherently say" is visible rather
   than silently absorbed: 45 suppressed `det`, 55 suppressed `aux`, 7
   suppressed `mark`, 107 total, over the full corrected pool. A
   regression guard now asserts, for every gold token written, that
   `head == 0` if and only if `relation == "root"`.

   One of the 55 suppressed `aux` cases is a genuine friction tagging
   error the correction was faithfully propagating: in "This endpoints
   sends Alt-Svc header field to clients if it is", friction's own
   tagger assigns `sends` the tag `MD` (modal) instead of `VBZ` — `sends`
   is not a modal by any reading, this is a tagger mistake on an
   unusual sentence. The guard fixes the *symptom* (a root cannot bear
   `aux` regardless of why the correction fired), not the underlying
   tag; the same tagger error would still silently corrupt this token's
   relation anywhere it occurs as a *non*-root dependent, since only the
   root position makes the inconsistency visible at all. This is exactly
   the honest-limits point already made about spaCy's own errors, just
   one level down: the closed-class correction is only as good as the
   POS tag it keys on, and a wrong tag produces a wrong correction with
   no local signal that anything went wrong.
6. **Projectivity.** Of the 14,621 aligned sentences, 132 (0.90%)
   produced a non-projective tree and were dropped — arc-eager, the
   transition system this gold file trains, can only derive projective
   trees. This is below the 5%-of-pool threshold that would have
   stopped this run for reconsideration, and close to the 0.94% measured
   on the smaller proxy sample. 14,489 sentences (202,391 tokens)
   survived to the correction stage.
7. **Closed-class correction.** Three overrides replace spaCy's drafted
   relation with one implied unambiguously by the token's own part of
   speech (subject to the root guard in step 5), mirroring the POS gold
   file's closed-class dictionary pass: `det` on any token friction's
   own tagger calls a determiner (`DT` or `PDT`) — 4,469 corrections;
   `aux` on any token it calls a modal (`MD`) — 132 corrections; `mark`
   on a subordinating conjunction introducing a clause — 1,315
   corrections. The third reads spaCy's own universal-POS tag (`SCONJ`)
   rather than friction's, because friction's Penn tagset does not
   distinguish a subordinator from a preposition (both are `IN`) and so
   cannot make that call unambiguously on its own; spaCy's finer
   category can. A synthetic `other`-relation attachment created by
   projection (step 3) is not eligible for these overrides — its
   relation is fixed by construction, not drafted by spaCy, so there is
   nothing to correct. No correction beyond these three closed classes
   was applied — everything else stands exactly as spaCy drafted it.
8. **Label mapping.** spaCy's relation labels were mapped to the fixed
   target set `root acl advcl agent amod aux auxpass cc ccomp conj csubj
   det dobj mark nsubj nsubjpass pobj prep xcomp punct` — the ClearNLP/
   OntoNotes relation names `DepRelation` itself uses, not Universal
   Dependencies v2's renamed equivalents; every other spaCy relation
   collapses to `other`, and so does every synthetic projection
   attachment. Over the 14,489-sentence corrected pool (202,391 tokens),
   39,702 tokens (19.62%) collapsed to `other`; the per-relation
   breakdown (over the final sampled file, 3,623 sentences, 51,007
   tokens) is: `acl` 315, `advcl` 789, `agent` 88, `amod` 2,635, `aux`
   2,090, `auxpass` 584, `cc` 1,420, `ccomp` 547, `conj` 1,530, `csubj`
   51, `det` 5,364, `dobj` 2,640, `mark` 809, `nsubj` 2,773, `nsubjpass`
   426, `other` 10,091 (19.78%), `pobj` 4,157, `prep` 4,669, `punct`
   5,749, `root` 3,623, `xcomp` 657. `agent`, `csubj`, and `acl` remain
   the three thinnest relations in the final file, as they were before
   this revision (88, 51, and 315 respectively) — a larger pre-sample
   pool raised their pool-level counts, but the fixed ~4,000-sentence
   sampling target caps how much of that reaches the shipped file; a
   consumer that needs more of these three specifically should widen the
   sampling target before looking elsewhere.
9. **Sampling.** A fixed stride over the 14,489-sentence corrected pool,
   sorted by document id then sentence index (not a random sample), lands
   the gold set near a 4,000-sentence target: stride 4 yields 3,623
   sentences. The stride is rounded to the nearest whole number rather
   than truncated, landing closer to 4,000 than the next stride down or
   up would have (stride 3 would have landed at 4,830, further away).
10. **Split.** Train/test is assigned by document id, not by sentence: a
    document goes to `test` if the first four bytes of its id's sha256
    digest, taken as an integer, fall in the lowest fifth of the digest
    space, and to `train` otherwise — a fixed, seedless function of the
    id alone, so every sentence from one document always lands in the
    same split. Of the 3,623 sampled sentences, 2,744 (198 documents) are
    `train` and 879 (65 documents) are `test`; the two document sets are
    disjoint by construction, verified empty-intersection at write time.

## On the original 97.4% figure

The alignment rate this file measures (96.28%, after both normalization
directions were generalized and a real whitespace-token bug was fixed)
is close to, but arrived at differently from, the 97.4% originally
quoted as a baseline before either normalization existed. That original
figure was not wrong about the corpus so much as circular about how it
was produced: it reconstructed sentence text from friction's own token
list through a detokenizer, rather than measuring against the sentence's
real text. So when friction's tokenizer split `S2O` into `S`/`2`/`O`,
the detokenizer re-emitted `"S 2 O"`, spaCy tokenized *that* into three
tokens too, and the disagreement disappeared by construction — it was
never given the chance to show up. An intermediate revision of this file
measured 59.94%, then 77.75%, against the sentence's *real* text with
progressively more normalization; 96.28% is the current, most-normalized
figure, also against real text. Treat the sequence of figures across
this document's revisions as a genuine, real improvement in how much of
the corpus this pipeline can use, not evidence that any one of them was
wrong.

## Honest limits

This is drafted, corrected silver data, not hand-annotated gold. Every
relation not covered by the three closed-class overrides in step 7 (and
their step-5 root exception) is exactly what spaCy's `en_core_web_sm`
parser predicted, unreviewed at the individual-sentence level; the only
broad-strokes checks this file's own build applies are the alignment and
projectivity filtering above, which catch disagreement with friction's
own tokenizer and with the parser's own transition system, not
disagreement with the true parse. Training against this file caps
parser accuracy near spaCy's own, and any systematic error spaCy's
parser makes is inherited rather than caught. A closed-class correction
is also only as good as the friction POS tag it keys on — step 5's
`sends`/`MD` case is a concrete instance of a wrong tag being faithfully
propagated by an otherwise-correct correction rule, and nothing in this
file's own build would catch the same error at a non-root position, since
only the root case produces the kind of internal inconsistency this file
can detect at all. Treat a transducer that misfires on a specific
dependency structure, or a training run that reproduces a friction
tagging error, as a reason to revisit this file (or `gold_pos_en.tsv`)
before revisiting the consumer.

Separately: the alignment step is a source of measured, directional
sampling bias (step 4, "Bias check, twice"), not just an efficiency
loss. It under-represents exactly the sentence shapes technical
documentation is made of: ones naming products, versions, and
code-adjacent identifiers. Generalizing the merge direction closed most
of the letter+digit-boundary share of that gap; the identifier-shaped
share (paths, version strings, command invocations — mid-token `/`,
`::`, `.`, `_`) is essentially unchanged as a ratio, because what remains
undropped needs actual internal parsing, not token-count reconciliation.
A user of this file who cares about parser behavior on identifier-heavy
prose specifically should weight that limitation, not treat
the 96.28% headline figure as evenly distributed across sentence types.

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
  the 1-based index of the head token, or `0` for the sentence root, and
  `head_index == 0` if and only if `relation == "root"`), preceded by
  three comment lines (`# sent_id = <doc_id>:<sent_index>`, `# split =
  train|test`, `# text = <sentence text>`) and followed by a blank line.
  Sentences appear sorted by document id then sentence index, so the
  file is byte-reproducible from the corpus alone.
- **Size:** 1,419,847 bytes; 3,623 sentences, 51,007 tokens.
- **sha256:**
  `5ae512bfb4cdb90db76cbb01488f7d5986e4bcbd8a7b3844d6320045d542ce36`

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
