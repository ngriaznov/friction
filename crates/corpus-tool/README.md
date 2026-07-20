# corpus-tool

Development-only CLI for managing the `friction` human/LLM validation
corpus. Not shipped as part of the `friction` release; used only while
building and maintaining the corpus.

## Corpus directory layout

```text
corpus/
  manifest.jsonl                     one JSON object per document
  genconfig.toml                     generate config: models, temps, targets
  prompts/<genre>.toml                generation prompt catalog
  human/<genre>/<id>.md               human-authored docs
  llm/<genre>/<id>.md                 LLM-generated docs
  quarantine/<genre>/<id>.md          CC-BY-SA material, no class subdir
  holdout.lock                        sealed holdout manifest
```

`<genre>` is one of `docs`, `blog`, `readme`, `email`, `forum` (a fixed
set).

A document's path is derived entirely from its manifest record — there
is no separate "quarantined" field. A record is quarantined — CC-BY-SA
StackExchange material, measured but never redistributed in the shipped
pack — exactly when its `license` field names CC-BY-SA
(case-insensitive, e.g. `"CC-BY-SA"` or `"CC-BY-SA-4.0"`); quarantined
docs live under `quarantine/<genre>/<id>.md` regardless of `class`.
Everything else lives under `<class>/<genre>/<id>.md`. See
`src/corpus_layout.rs`.

All subcommands default `--corpus-dir` to `corpus` (relative to the
current directory), and are meant to be run from the repository root —
e.g. `cargo run -p corpus-tool -- validate`.

## Manifest schema

`corpus/manifest.jsonl`: one JSON object per line, `deny_unknown_fields`
(an unrecognized field is a hard parse error, not a silently dropped
typo). See `src/manifest.rs` for the canonical `serde` definitions.

| Field                 | Type                        | Notes |
|------------------------|-----------------------------|-------|
| `id`                   | string                      | unique within the manifest |
| `class`                | `"human"` \| `"llm"`        | |
| `genre`                | `"docs"` \| `"blog"` \| `"readme"` \| `"email"` \| `"forum"` | |
| `source`               | string                      | where the doc came from |
| `model`                | `{name, quantization}` or `null` | `llm`-only; required for `llm` docs |
| `prompt_id`            | string or `null`            | `llm`-only; required for `llm` docs |
| `license`              | string                      | must be non-empty |
| `lang`                 | string                      | BCP-47; `"en"` only in v1 |
| `split`                | `"train"` \| `"dev"` \| `"holdout"` \| `null` | assigned by `split` |
| `sha256`               | string                      | lowercase hex digest of the doc file |
| `provenance_evidence`  | string or `null`            | archive.org timestamp, git commit date, or publication date |
| `style_prompted`       | bool (default `false`)      | generated with a "sound human" instruction |
| `gen_config`           | JSON object or `null`       | model digest, sampler params, seed, reproducible flag; required for `llm` docs |

Provenance rule: a `human` doc needs either `provenance_evidence` set,
or `license` exactly equal to `"personal-attestation"`.

## Subcommands

- **`validate`** — manifest parses strictly; every referenced
  file exists and its sha256 matches; license is non-empty; `human` docs
  carry provenance; `llm` docs carry
  `model`/`prompt_id`/`gen_config`; ids are unique. Word counts outside
  `[300, 2000]` are warned to stderr, not failed. Non-zero exit
  on any hard violation. An absent or empty corpus prints `"empty
  corpus"` and exits 0.
- **`stats`** — per-`(class, genre)` doc counts, word-count
  summary stats (min/mean/max), and split counts, in deterministic
  order (`(class, genre)` sorted by declaration order in the enums, not
  alphabetically — still fully deterministic). `--report <path>` writes
  a markdown report; default is stdout.
- **`split`** — deterministic stratified 70/15/15 split by
  `(class, genre)`. Within each cell, candidates are ordered by
  `sha256(id)` hex (ascending, ordinary lexicographic order over the hex
  string) and sliced at the 70%/85% boundaries — no RNG anywhere. Docs
  already holding `split: "holdout"` are sealed and never reassigned;
  the command errors out with a clear message if the freshly computed
  holdout slice for a cell would move a sealed doc out (or a
  non-sealed doc in). `--dry-run` prints the computed split without
  writing the manifest.
- **`seal`** — writes `<corpus_dir>/holdout.lock`: one
  `id<TAB>sha256<TAB>relpath` line per holdout doc, sorted by id.
  Refuses to overwrite an existing lock whose content would differ
  unless `--init` is passed.
- **`holdout-check`** — verifies `<corpus_dir>/holdout.lock` against the
  manifest and on-disk files; exits non-zero on drift. Semantics match
  `scripts/check-holdout.sh` (used by CI), plus a manifest cross-check
  (the manifest record for a locked id must still say
  `split: "holdout"` with a matching sha256, and every manifest
  `holdout` record must appear in the lock). `relpath` entries in the
  lock are relative to the current working directory — run this (and
  `seal`) from the repository root. An absent lock file is a no-op
  success, so CI stays green before the holdout is sealed.
- **`clean`** — `--incoming <dir> --out <dir>`: reads every
  `.md` file under `--incoming` in sorted order, normalizes it to
  UTF-8 + LF, strips common README boilerplate (badge-image walls;
  standalone HTML nav/footer/layout wrapper tag lines such as
  `<div align="center">`, `<p align="center">`, `<hr>`, bare `<img>`),
  and writes survivors under `--out`, mirroring the incoming directory's
  relative layout. Markdown structure (headings, lists, code fences) is
  left untouched. Docs under 300 words after cleaning are dropped (not
  written) and reported. This command does not touch the manifest — it
  only produces cleaned `.md` files; adding manifest entries for them is
  a separate, manual curation step.
- **`ingest`** — `--incoming <dir> --corpus-dir <dir>`: folds
  collector-supplied raw human-corpus candidates into the real corpus.
  Reads every `<dir>/<genre>/*.md` file plus its `<dir>/meta-*.jsonl`
  metadata fragments (`file`, `genre`, `source`, `license`,
  `license_evidence`, `provenance_evidence`, `title` — one JSON object per
  doc), applies the identical cleaning transform as `clean`, drops docs
  that fall under 300 words after cleaning (reported, not written),
  normalizes the license to a small canonical set (`MIT`, `Apache-2.0`,
  `BSD-2-Clause`, `BSD-3-Clause`, `CC-BY-4.0`, `CC-BY-3.0`, `CC0-1.0`,
  `PD`, `CC-BY-SA-3.0`, `CC-BY-SA-4.0`), and writes each survivor under
  its layout-correct path (`class: human`; quarantined automatically when
  the license is CC-BY-SA, per `corpus_layout`) with a full manifest
  record (`lang: "en"`, `split: null`, `style_prompted: false`,
  `provenance_evidence` from the fragment). A fragment is refused (listed
  in the run summary, not ingested) if its license, `license_evidence`, or
  `provenance_evidence` is missing/empty, its license doesn't normalize,
  its genre isn't one of the fixed five, its source `.md` file is
  missing, or another fragment already claims the same `file`. Each doc's
  id is the first 16 hex characters of `sha256(source)` — stable across
  reruns — with a deterministic fallback (mixing in the fragment's own
  file path) on the rare case where two fragments share one `source`
  (e.g. several essays pulled from one anthology page). Reruns are
  incremental: a fragment whose id is already in the manifest is skipped
  without touching the filesystem again. Does not delete or move
  `--incoming` — it's a read-only input.
- **`generate`** — generates the `llm` corpus
  against a local [Ollama](https://ollama.com) server. See "Generating
  the LLM corpus" below.
- **`envelope`** — for every `human`-class, `train`-split document,
  computes its metric vector, groups by genre, and for each
  `(genre, metric)` pair estimates a `[lo, hi]` percentile band
  (nearest-rank method; `--lo-percentile`/`--hi-percentile`, default
  10/90). Writes the result as a versioned TOML pack to `--out`
  (default `crates/friction-packs/packs/envelope-v1.toml`). Quarantined
  (CC-BY-SA) human docs are included in the estimate — quarantine
  restricts redistributing document *text*, not aggregate statistics —
  and a genre with zero train-split human docs is omitted from the pack
  (warning to stderr) rather than emitting a degenerate band.
- **`separate`** — on the dev split, measures how well the metric
  vector separates `llm` docs from `human` docs, per genre and per
  metric (AUC via the Mann-Whitney U statistic, oriented so `AUC > 0.5`
  always means "separates llm from human"), plus a combined per-document
  score (fraction of metrics falling outside that document's genre's
  envelope band, loaded from `--envelope`) and that score's own AUC.
  Writes a markdown report to `--report`. Like `envelope`, quarantined
  human docs are not excluded from the dev-split measurement. A genre
  missing data for one class (or missing from the envelope pack) is
  reported as `n/a` rather than a fabricated AUC.
- **`remove`** — `--id <id>` (repeatable): validates every requested id
  is present in the manifest before touching anything, then for each
  deletes its corpus file (`<class>/<genre>/<id>.md`, or
  `quarantine/<genre>/<id>.md` when quarantined) and drops its manifest
  line. The raw doc under `corpus/incoming/` is never touched.

## Generating the LLM corpus (`generate`)

`corpus-tool generate` reads `corpus/genconfig.toml` (model matrix,
temperature/style-prompted slicing, per-genre targets — see the file
itself for the annotated schema) and `corpus/prompts/<genre>.toml`
(`[[prompts]]` tables with `id`, `text`, `topic`), builds a
deterministic job plan, and — unless `--dry-run` — executes it against
Ollama's `/api/generate`, `/api/show`, and `/api/tags` endpoints.

**Planning (deterministic, no RNG).** For each genre, prompts are
consumed in `id` order and assigned to models round-robin (model-minor,
prompt-major: every model gets a turn at a prompt before any model moves
to the next one), up to `targets.docs_per_genre` jobs. Within that
sequence, every `low_every`-th job (`floor(1 / temperature.low_fraction)`)
uses the low temperature instead of the default (this guarantees
`>= low_fraction` of docs use it, for any job count), and the last slot
of every `style_every`-sized block (`ceil(1 / style_prompted.fraction)`)
is style-prompted (this guarantees `<= fraction`). A style-prompted
job has `style_prompted.instruction` appended to its prompt text; every
other job sends its prompt text verbatim, with no system or style prompt
at all.

**IDs and seeds.** A job's seed is `base_seed` plus a stable
hash of `(model, prompt_id, slice)`; its doc id is the first 16 hex
characters of `sha256(model + prompt_id + seed + temperature)`. Both are
pure functions of the job's own fields — no ambient RNG, no clock. This
also makes reruns incremental: a job whose doc id is already in
the manifest is skipped without another Ollama call, so `generate` can be
re-run end-to-end (e.g. after pulling another model) and only fills in
what's missing.

**Missing models.** A model in `genconfig.toml`'s matrix that isn't
currently pulled in the local Ollama is not an error: `generate` warns
once per missing model, skips its jobs, and continues with the rest of
the plan. If any model was skipped this way, the process exits with code
`3` (not `0`) after printing its summary line, so automation can tell
"ran clean" apart from "ran but under-generated"; any other failure
(bad config, a live Ollama request erroring outright) is a normal
non-zero-exit error.

**Flags:** `--dry-run` (print the job plan, tab-separated, in
deterministic order, touching neither the network nor the corpus),
`--limit N` (cap the plan to its first N jobs), `--model <name>`
(restrict to one matrix model), `--genre <g>` (restrict to one genre).

**Manifest record.** Each generated doc gets `source: "ollama"`,
`model: {name, quantization}` (quantization from `/api/show`),
`prompt_id`, `license: "generated"`, `style_prompted`, and `gen_config`
(endpoint, model digest, temperature, seed, `num_predict`, a
`reproducible` flag). The model digest is `/api/show`'s
top-level `digest` field when present; some Ollama versions omit it, in
which case a `sha256:`-prefixed hash computed over the response's
`details` + `model_info` is recorded instead (still stable and
content-derived, just not Ollama's own blob digest — see
`src/ollama.rs`).

## Adding a subcommand

Each subcommand is a module under `src/commands/` exposing a
clap-derived `Args` struct and a `pub fn run(&Args) -> anyhow::Result<()>`
(`generate` is the one exception — see `src/commands/mod.rs` for why it
returns `anyhow::Result<generate::GenerateOutcome>` instead). To add a
new subcommand:

1. Add `src/commands/<name>.rs` with `pub struct Args` and `pub fn run`.
2. Add `pub mod <name>;` to `src/commands/mod.rs`.
3. Add a `<Name>(commands::<name>::Args)` variant to the `Command` enum
   in `src/cli.rs`, and a matching arm in the `match` in `run()`.

No other file needs to change.

## Tests

- Unit tests live next to their modules (`#[cfg(test)] mod tests`) and
  cover pure logic: hashing, manifest (de)serialization, path resolution,
  the `clean` text-transform functions, `genconfig`/`prompts` parsing,
  `ollama`'s digest-fallback logic, and `generate`'s seed/doc-id
  derivation and job planning (including a golden dry-run plan test).
- Integration tests live under `tests/`, one file per subcommand, and
  build corpora in a `tempfile::tempdir()` per test (via helpers in
  `tests/common/mod.rs`) so every test is isolated and cleans up after
  itself.
- `tests/fixtures/valid_corpus/` is a small, static, checked-in golden
  corpus (one `human` doc, one `llm` doc, precomputed sha256 hashes)
  used as a regression fixture distinct from the dynamically-built
  corpora used in most tests.
- `tests/generate.rs`'s live-Ollama smoke test only runs with
  `FRICTION_OLLAMA_TEST=1` set (and a local Ollama server reachable, with
  `granite4.1:3b` pulled) — every other test, including the rest of
  `generate.rs`, is fully offline. Run it with:
  `FRICTION_OLLAMA_TEST=1 cargo test -p corpus-tool --test generate`.
- Test names describe the behavior they exercise, e.g.
  `validate_human_without_provenance_rejected`.
