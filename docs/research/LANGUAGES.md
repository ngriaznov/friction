# The language framework

What it takes to scale rule coverage within English, and to bring up a
second language. Written from a three-lens audit of the codebase at
0.5.4 + the review-sweep branch; every coupling claim below was checked
against the code, not the file names.

## Part 1 — extending LLM-speak coverage (English)

The pipeline (mine → adjudicate → frame-pack gauntlet → pack, per
[EXTENDING.md](../EXTENDING.md)) is structurally sound. Its throughput
ceiling is human labor, in this order:

1. **Human-corpus volume.** Candidates die at the attestation floors
   (the insight_adj rejection: "insights" at 10.2/M against a 20/M
   floor), not in mining. Every 100k words added to `corpus/human`
   unlocks rules that are already minable today. This is the highest-
   use ongoing investment, and it also sharpens every existing
   band.
2. **Report → TOML transcription.** Every mining command deliberately
   emits markdown reports, never pack rows; a human transcribes chosen
   candidates by hand. Keep the human decision, remove the toil: an
   `adjudicate --draft-toml` mode that emits gauntlet-ready candidate
   rows (pattern, evidence numbers, provenance stub) for human review
   would batch the mechanical half. The gauntlet still re-validates
   everything from scratch, so this adds no risk surface.
3. **The snapshot ritual** stays human by design. It scales with the
   number of behavior-changing releases, not with rule count, so it is
   not the binding constraint.

Planned corpus arcs that unlock parked rules: a test-register pair
(generated test suites vs pre-2022 OSS test prose) to graduate the
curated meaningless/positive-contrast/this-ensures family into measured
rules, and a chat-register corpus to separate the see-saw maxim from
informative contrast (T10's only honest path back).

## Part 2 — a second language: the three layers

### Layer A — already portable (no work)

friction-core (structural types), friction-parse (byte-range
extraction), both perceptrons' algorithms and their training tooling
(TSV gold tags; CoNLL-U treebanks — Universal Dependencies ships
German/French/Russian in exactly this format), weights-pack
serialization, the pack parsers and the frame-rule DSL, the gauntlet
math, the gate architecture (attest-before-edit, verb-survival), DMS
suffix automata, the attestation mechanism, and the trait seams:
`Tagger`/`DepParser`/`Segmenter`, and `edit_document()` already takes
every pack by reference. The engine core does not need touching.

### Layer B — real seams, need a second dataset plus small plumbing

- SRX segmentation: the format and crate are multi-language; only the
  wrapper hardcodes `language_rules("en")`.
- Tagger/parser weights: retrain on a UD treebank; the artifact
  pipeline is generic. (Caveat: the dependency-relation taxonomy has
  analytic-language bias — AuxPass models the English periphrastic
  passive — revisit per language.)
- Envelope + register bands: re-measure against the target language's
  human corpus; genre keys and band mechanics unchanged.
- Inventory / frame-rules / jargon packs: same schemas, new content,
  authored through the same mining pipeline over a per-language corpus.
- DMS index, attestation pack, evidence packs: re-run the mining
  pipeline; the artifacts are language-blind token streams.
- `ManifestRecord.lang` (BCP-47) already exists on every corpus record
  and is read by nothing — the corpus partition key is staged, dormant.

### Layer C — English in code (the actual work)

1. **No selection mechanism exists**, the master bottleneck. Every
   pack is a bare `include_str!`/`include_bytes!` static; every
   consumer reaches for the global directly; `Engine::load()` takes no
   arguments. Nothing else can land until a `Lang`-keyed pack bundle
   and `Engine::load(lang)` exist. Pure plumbing; do it first.
2. **The ASCII tokenizer wall.** Three-plus independently maintained
   copies of an `[a-z']`-class word regex (friction-match `token.rs`,
   friction-harness `clean.rs`, friction-packs `validate.rs`, plus
   mining-side variants) feed every detection channel and every mining
   command. Cyrillic text produces zero word tokens; umlauts and
   diacritics split words silently. Fix by unifying on ONE shared
   Unicode-letter tokenizer — byte-fenced, since `\p{L}` and `[A-Za-z]`
   agree on ASCII English text.
3. **Morphology and grammar as code.** `inflect.rs` (English suffix
   orthography), `chunk.rs` (finiteness = Penn-tag literals; imperative
   = bare-infinitive-initial, a theory that does not transfer), the
   T4–T9 transducers (generative English syntax — the single largest
   per-language lift), the lexicon schema (English categories: no
   case/gender/declension fields), the a/an repair, the five register
   detectors' closed word lists, quote/punctuation classes (no «» „“).
   The framework answer is a per-language profile behind a trait:
   finiteness test, closure word sets, morphology generator, transducer
   set — with English as the first implementation, extracted rather
   than rewritten.

### The posture that makes this tractable: detection before repair

The engine already has a first-class detect-only mode
(`diagnostic_only`, report-only rules, detection channels that never
edit). A new language can ship as **check-only** (DMS differential,
inventory detection, register bands, envelope metrics) with zero
transducers and zero morphology. That is Layer B work almost entirely.
Repair (Layer C item 3) follows per construction, each rewrite earning
its way through the same gauntlet-and-refutation discipline English
went through.

## Phasing

- **Phase 0 — groundwork inside English, all byte-fenced:** unify the
  tokenizer on Unicode classes; introduce `Lang` + the pack bundle +
  `Engine::load(lang)` with `en` as the only member; start reading
  `manifest.lang` in corpus-tool; parameterize the SRX language tag and
  the `LEXICON_EN` call sites. English output must not move a byte.
- **Phase 1 — first target language, check-only:** stage its human +
  machine corpus per genre (the same discipline the English corpus
  went through — this, not code, is the expensive part); retrain
  tagger/parser on its UD treebank; author its SRX block; measure
  envelope + register bands; run index/attest/dms-pack; mine detection
  entries. Ship `friction check --lang xx`. German or French first —
  closest morphological fit; Russian benefits most from the Phase 0
  tokenizer fix but needs the most Layer C work for repair.
- **Phase 2 — repair for that language:** lexicon schema v2 for its
  language family, morphology module, finiteness/closure profile, then
  transducers one at a time, each with corpus evidence and its
  refutations recorded in the pack, exactly as for English.

The discipline (gauntlet as authority, byte-fence, staged-inert
activation, refutation records) is the part of friction that is
already language-agnostic. The framework's job is to let a second
language buy into that discipline without rewriting it.
