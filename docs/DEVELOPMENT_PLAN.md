# friction — Development Plan

Deterministic Rust engine that measurably reduces LLM-speak in prose by injecting "friction": sentence-length variance, asymmetry, and register texture, until text re-enters a human statistical envelope. Not a detector, not an LLM paraphraser: a metrics-driven, idempotent, span-aware rewriter.

Requirement IDs (`FR-*`, `NF-*`, milestone acceptance criteria) are stable and intended for traceability verification against the implementation.

---

## Non-negotiable invariants (apply to every phase)

- **NF-1 Determinism.** Identical input bytes + identical pack version → identical output bytes. No wall-clock, no global RNG, no HashMap iteration order leaking into output (use `BTreeMap`/sorted collections anywhere order affects output).
- **NF-2 Idempotence.** `fix(fix(x)) == fix(x)` for every corpus document. Enforced in CI.
- **NF-3 Human-text near-no-op.** On the human validation corpus, ≤ 2% of sentences receive any machine-applied patch.
- **NF-4 Offline at lint time.** No network access during `check`/`fix`. Models and packs may be downloaded at install or first run, pinned by exact version + sha256 (fail hard on mismatch); cached locally thereafter. Lint with a missing model must degrade gracefully: model-dependent rules drop to Suggest or Off with a clear diagnostic, never a crash.
- **NF-4b Model determinism discipline.** Downloaded models are classifiers/parsers only (tagger, dependency parser, optional embedder) — never generative in the Fix tier. Inference: fixed intra-op thread count, pinned ort/ONNX Runtime version. Rules consuming model output apply a margin gate: if top-2 decision margin < ε, the finding is demoted to Suggest.
- **NF-5 Span honesty.** Every patch carries byte ranges into the *original* source of its round; patches applied in one atomic pass per round, re-parse between rounds.
- **NF-6 Meaning preservation tiers.** Every fix is tiered `Fix` (machine-applicable, meaning-preserving by construction) or `Suggest` (diagnostic only). Deletions and contractions may be `Fix`; any transform that reorders propositional content is at most `Suggest` until promoted by eval evidence.
- **NF-7 No global tic.** Fix-strategy selection uses a PRNG seeded with xxhash64 of (sentence bytes, rule id). Same input → same choice; no constant strategy across documents.
- **NF-8 Rust 2024, workspace lints, `cargo clippy -- -D warnings`, rustfmt clean.**

---

## Phase 0 — Test corpus (FIRST; everything downstream depends on it)

**Goal:** A paired, genre-labeled corpus of human and small-model prose, with a frozen train/dev/holdout split, sufficient to (a) estimate human envelopes, (b) validate that the metric vector separates the classes, (c) serve as golden inputs for all later tests.

### 0.1 Corpus schema

- **FR-0.1** Define `corpus/` layout: one document per file, sidecar JSONL manifest with fields: `id`, `class` (`human` | `llm`), `genre` (`docs` | `blog` | `readme` | `email` | `forum`), `source`, `model` (for llm class: model name + quantization), `prompt_id` (llm class), `license`, `lang` (BCP-47, `en` only in v1), `split` (`train` | `dev` | `holdout`), `sha256`.
- **FR-0.2** Target volume: ≥ 400 human docs and ≥ 400 LLM docs total, ≥ 60 per (class, genre) cell, 300–2000 words each. Split 70/15/15 by document, stratified by (class, genre). Holdout is sealed: no code or threshold may be tuned against it.

### 0.2 Human corpus (permissively licensed only)

- **FR-0.3** Collect from license-safe sources, recording license per doc: pre-2022 project READMEs and documentation from permissively licensed GitHub repos (MIT/BSD/Apache); pre-2022 blog posts explicitly under CC-BY/CC0; Project Gutenberg essays (register mismatch — cap at 10% of human corpus); StackExchange answers (CC-BY-SA — quarantine in a separate directory, use for measurement only, never redistribute in the shipped pack); own writing and any privately contributed docs.
- **FR-0.4** Cutoff rule: human docs must have provenance predating 2022-01-01 OR be personally attested, to minimize AI contamination. Record the evidence (archive.org timestamp, git commit date, publication date) in the manifest.
- **FR-0.5** Cleaning pipeline (a small Rust or script tool, itself deterministic): strip boilerplate (nav, footers, badge walls), normalize to UTF-8 + LF, keep markdown structure, drop docs < 300 words after cleaning.

### 0.3 LLM corpus (generated, fully controlled)

- **FR-0.6** Generate with local small models via Ollama or llama.cpp; minimum matrix: Qwen2.5 7B-instruct, Gemma-class 4B/9B, Llama-class 8B, one 3B-tier model. Temperature 0.7 default plus a temperature-0.2 slice (≥ 20% of docs) — low-temp output is *more* uniform and is the hard case.
- **FR-0.7** Prompt set: ≥ 40 prompts per genre, written to elicit the same topics/genres as the human corpus (e.g., "write a README for a CLI tool that does X", "write a blog post explaining Y"). Prompts stored in-repo (`corpus/prompts/*.toml`) with `prompt_id`; generation script re-runnable end-to-end.
- **FR-0.8** No system prompt asking for "human-like" style — capture the models' *default* register; a separate small slice (≤ 10%) generated WITH a "sound human" instruction, labeled `style_prompted: true`, to test robustness against prompt-level evasion.
- **FR-0.9** Generation script records full config (model digest, sampler params, seed) in the manifest; regeneration with same config must reproduce byte-identical output where the runtime supports seeding, else record non-reproducibility explicitly.

### 0.4 Acceptance criteria (Phase 0)

- **AC-0.a** Manifest validates against a schema check (`cargo run -p corpus-tool -- validate`), all licenses recorded, all sha256 verified.
- **AC-0.b** Per-cell minimums met (FR-0.2); a `corpus-tool stats` report is committed.
- **AC-0.c** Holdout split written once and committed; CI fails if holdout file hashes ever change.

---

## Phase 1 — Workspace scaffold + `friction-core` + `friction-parse`

- **FR-1.1** Cargo workspace with crates: `friction-core`, `friction-parse`, `friction-nlp`, `friction-metrics`, `friction-rules`, `friction-plan`, `friction-apply`, `friction-packs`, `friction-cli`, `corpus-tool` (dev-only).
- **FR-1.2** `friction-core`: `Document { source: Arc<str>, blocks, prose }`, `Block` (markdown AST node + byte range), `ProseUnit` (block-scoped sentences → tokens, all with byte spans), `Patch { range, replacement, rule, tier }`, `MetricVector`, `Envelope`, `RuleId`, `Finding`, error types (`thiserror`).
- **FR-1.3** `friction-parse`: pulldown-cmark → block tree with exact byte ranges; prose extraction excludes code blocks, inline code, link URLs, tables' structure (cell text is prose); round-trip property test: re-serializing untouched document reproduces input bytes.
- **AC-1.a** `friction-parse` round-trip passes on 100% of corpus docs.
- **AC-1.b** Workspace builds with NF-8 gates in CI (GitHub Actions: fmt, clippy, test, corpus-hash check).

## Phase 2 — `friction-nlp`

- **FR-2.1** Sentence segmentation via SRX rules (`srx` crate) wrapped in `trait Segmenter`; abbreviation handling verified against a golden set of ≥ 100 tricky sentences (decimals, "e.g.", initials, code-ish tokens).
- **FR-2.2** POS tagging + morphology via `nlprule` behind `trait Tagger` (own trait from day one; nlprule is an implementation detail). Tokenizer binary embedded or downloaded at build time into `OUT_DIR` — never at runtime (NF-4).
- **FR-2.3** Inflection service: given (surface form, target lemma) produce agreeing form ("leverages"→"uses", "Leveraging"→"Using" with capitalization transfer). Golden tests ≥ 50 pairs.
- **FR-2.4** `trait DepParser` with two implementations: `Heuristic` (POS-pattern approximations, always available) and `Onnx` (small pinned UD dependency parser via `ort`, downloaded per NF-4, sha256-pinned in `friction-packs`). Consumers (fusion subject check, participial attachment, triad structure) written against the trait; per-decision margin per NF-4b. Model choice documented (candidate: UDPipe 2 English EWT-class model, target < 100 MB).
- **FR-2.5** Model bootstrap: `friction setup` subcommand downloads + verifies models/packs into a cache dir (XDG-compliant), prints resolved versions; `friction check --require-models` fails if absent (for CI parity).
- **AC-2.a** Segmenter + tagger deterministic across runs (hash of full corpus token stream stable in CI).
- **AC-2.b** With models present, full-corpus parse decisions hash-stable across ≥ 3 repeated runs on the same machine; cross-implementation (Heuristic vs Onnx) disagreement rate reported, not gated.

## Phase 3 — `friction-metrics` + envelope estimation (the go/no-go gate)

- **FR-3.1** Metric vector v1, computed per document and per paragraph: sentence length mean/stddev/CV (burstiness); sentence-initial discourse-marker density per 1000 tokens; triad rate ("X, Y, and Z" coordination patterns per sentence); contraction ratio (contracted / contractible); bullet-stem parallelism score; paragraph-shape stats (sentences per paragraph mean/CV); em-dash and semicolon density; participial-closer rate; "not just/only X but (also) Y" rate; ritual open/close markers (In conclusion, Overall, In today's...).
- **FR-3.2** Every metric: pure function, unit-tested against hand-computed fixtures, deterministic.
- **FR-3.3** Envelope estimation tool (`corpus-tool envelope`): per (genre, metric) compute human percentile bands (p10–p90 default) from train split; serialize to versioned TOML in `friction-packs`.
- **FR-3.4** Separation report (`corpus-tool separate`): on dev split, per metric and for a simple combined score, report AUC human-vs-llm per genre.
- **AC-3.a (GO/NO-GO)** Combined score AUC ≥ 0.85 on dev split for ≥ 3 of 4 primary metrics families; if not met, iterate metrics before any transform work.
- **AC-3.b** Envelope pack committed with version + generation config; regeneration reproducible.

## Phase 4 — `friction-rules` v1 + `friction-apply`

Rule trait per architecture discussion (`id`, `family`, `gate(metrics, envelope) -> Off|Detect|Fix(Budget)`, `scan`, `fix(finding, ctx, strategy_rng)`).

**Rule families and initial rules (each: detector + fixer + golden before/after files + idempotence test):**

- **FR-4.1 Lexical (Fix tier):** phrase deletions (discourse filler: "It's worth noting that", "It is important to note"), inflection-aware substitutions (leverage→use, utilize→use, individuals→people, numerous→many, commence→start; initial table ≥ 60 entries, sourced from Phase-0 log-odds mining, FR-4.8). Density-gated per NF-3.
- **FR-4.2 Connective surgery (Fix tier):** sentence-initial Moreover/Additionally/Furthermore/However-overuse — strategies: delete+recapitalize | swap to short connective ("But", "And") | none; strategy chosen via hash-seeded RNG (NF-7); budgeted to bring density into envelope, not to zero.
- **FR-4.3 Contraction insertion (Fix tier):** do not→don't etc., exception list (emphasis "do NOT", sentence-final auxiliaries, legal-ish contexts by genre flag).
- **FR-4.4 Rhythm (Fix for splits, Suggest for fusion initially):** split over-long uniform sentences at coordinators/semicolons; fuse adjacent short same-subject sentences (same head noun, both < N tokens). Gated on paragraph-level burstiness deficit only.
- **FR-4.5 Symmetry (mixed):** triad→dyad reduction (Suggest in v1), participial-closer deletion or promotion-to-sentence (Fix), "not just X but also Y" reframe (Suggest), ritual-conclusion paragraph deletion (Fix when paragraph adds no new content nouns vs. rest of doc — conservative heuristic, else Suggest).
- **FR-4.6 Structural (Fix tier):** un-bullet lists of ≤ 3 short items into a prose sentence; strip bolded lead-in labels in bullets; merge header-per-short-paragraph sections (Suggest in v1).
- **FR-4.7 `friction-apply`:** per-round patch collection → conflict resolution (leftmost-longest, then rule priority) → atomic apply → re-parse; fixpoint driver bounded at 4 rounds; CI idempotence sweep over full corpus (NF-2).
- **FR-4.8 Mining tool (`corpus-tool mine`):** log-odds ratio with Dirichlet prior over 1–3-grams, train split, human vs llm; output ranked candidates for the lexical tables; shipped tables are hand-curated from this output.
- **AC-4.a** Every rule has ≥ 3 golden before/after fixtures and passes idempotence.
- **AC-4.b** NF-3 holds on human dev split; report committed.
- **AC-4.c** On llm dev split, `friction fix` moves ≥ 70% of out-of-envelope (doc, metric) pairs into envelope within round budget.

## Phase 5 — `friction-plan` + `friction-cli`

- **FR-5.1** Planner: envelope deltas → ordered schedule with per-family budgets; ordering fixed and documented (structural → symmetry → connective → lexical → rhythm → contraction), rationale: outer transforms invalidate fewer inner spans.
- **FR-5.2** CLI: `friction check` (diagnostics + metric report + exit code), `friction fix` (Fix tier), `friction fix --suggest`, `friction explain` (before/after MetricVector table), `--format json|sarif`, `--genre`, `--pack <path>`, stdin/stdout mode for skill piping.
- **FR-5.3** miette diagnostics with labeled spans for every Finding; SARIF output validates against schema.
- **AC-5.a** End-to-end snapshot tests (insta) on 20 representative docs: full CLI output byte-stable.

## Phase 6 — Evaluation + hardening

- **FR-6.1** Holdout run (first and only tuning-free evaluation): report AUC shift human-vs-fixed-llm (goal: fixed LLM text's combined score distribution overlaps human envelope; report honestly, no target gaming), NF-3 on human holdout, idempotence, throughput (target ≥ 1 MB/s single-thread on prose).
- **FR-6.2** Meaning-preservation audit: sample 50 fixed docs, human review checklist (no negation flips, no dropped propositions from Fix-tier patches); any violation demotes the rule to Suggest.
- **FR-6.3** Self-fingerprint check: run `corpus-tool separate` with classes {fixed-llm vs human} using *new* candidate metrics (top mined n-grams of the fixed corpus) to confirm the tool hasn't introduced its own tic; document findings.
- **FR-6.4** Fuzzing (`cargo-fuzz`) on parse+apply: no panics, no span violations, output always valid UTF-8.

## Phase 7 — Skill packaging

- **FR-7.1** Release automation: static binaries (x86_64/aarch64, Linux + macOS), checksummed GitHub release.
- **FR-7.2** Skill folder: SKILL.md workflow (when to run check vs fix vs suggest; how to present `explain` deltas), bootstrap script fetching the pinned binary release by checksum and running `friction setup` to fetch models (both offline-cached after first run). LLM's role limited to Suggest-tier decisions — Fix tier is always the binary (determinism claim preserved).
- **FR-7.3** README with honest scope statement: surface/rhythm/register transforms; content-level tells (genericity, hedging-both-sides, absence of concrete detail) out of scope for the deterministic engine.

---

## Sequencing and effort (rough)

| Phase | Depends on | Est. size |
|---|---|---|
| 0 Corpus | — | 3–5 days (mostly scripting + curation) |
| 1 Scaffold/parse | 0 (for round-trip tests) | 2–3 days |
| 2 NLP | 1 | 3–4 days |
| 3 Metrics/envelopes | 0,1,2 | 4–5 days ← go/no-go |
| 4 Rules/apply | 3 | 2–3 weeks (the bulk) |
| 5 Plan/CLI | 4 | 3–4 days |
| 6 Eval/hardening | 5 | 1 week |
| 7 Skill | 6 | 2 days |

## Open decisions (resolve before Phase 4, defaults given)

- **OD-1** Fusion rule promotion to Fix tier: default keep Suggest in v1.
- **OD-2** Genre set frozen at {docs, blog, readme, email, forum} for v1: default yes.
- **OD-3** nlprule vs custom tagger: default nlprule behind trait; revisit only if quality blocks FR-2.3.
- **OD-4** Multilingual: out of scope v1; architecture must not hard-code English outside packs + nlp crate.
- **OD-5** Ship v1 rules against `Heuristic` DepParser only vs. require `Onnx` from the start: default — build both per FR-2.4, gate rule-by-rule on measured precision delta in Phase 4 goldens; a rule may *require* the Onnx parser (declared in its metadata) and self-demote to Suggest when only Heuristic is available.
- **OD-6** Optional tiny embedder for ritual-conclusion semantic-novelty check: default no in v1 (content-noun overlap heuristic first); revisit if FR-4.5 precision is poor.
