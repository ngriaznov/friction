# friction

A deterministic engine that removes the machine layer from LLM-generated technical
documentation — the ritual closers, filler spans, hedge phrases, and light-verb
constructions that make text read as machine-written — without touching the
information.

friction never synthesizes text. Every word it emits is either already present in
the input or derived from an input word through a static morphology table
("performs validation of" → "validates"). Where no safe edit exists, it reports
instead of rewriting. No model runs at fix time; everything is offline, integer-
and-regex-level, and byte-deterministic.

## What it does

Exactly four operations edit text. Nothing else does.

1. **Ritual deletion** — sentences matching ritual frames are removed whole:
   *"If you have any questions or require further assistance, please reach out
   to our support team."* → gone.
2. **Gated span deletion** — a detected filler span is removed only if the
   resulting seam is attested in a human-written corpus or falls at a sentence
   start, and the sentence still has a finite verb afterward:
   *"It is important to note that the agent…"* → *"The agent…"*.
3. **Paired substitution** — a slop frame becomes its plain-register counterpart
   from a versioned, audited inventory: *"will walk you through"* → *"covers"*,
   *"in order to"* → *"to"*, *"leverages"* → *"uses"*, *"is crucial for"* →
   *"is important for"*.
4. **Derivational pivot** — a licensed light-verb construction collapses to its
   root verb, inheriting tense and agreement: *"the agent performs validation of
   the config file"* → *"the agent validates the config file"*, *"we made the
   decision to switch"* → *"we decided to switch"*.

Every edit passes a stack of hard gates: no content word may enter the text
(closure), the sentence must stay clause-complete, edits never touch code spans,
links, numbers, identifiers, negation, quantifiers, modals, or logical
connectives, and edits fire only inside prose blocks — never in headings, code,
tables, or link text. On human-written text the whole engine is calibrated to be
a near-no-op. When a gate says no, the candidate is kept verbatim and reported
as a suggestion — doing nothing is the designed fallback, not a failure mode.

Detection (what finds the candidates) runs three channels: a mined literal
inventory, a shallow tag-pattern scan for light-verb constructions, and a
differential matching-statistics profile computed against per-generator-family
suffix-automaton indexes — which also powers the document-level report in
`friction check`.

## Install

```
cargo build --release -p friction-cli
```

The binary lands at `target/release/friction`. The build is fully
self-contained — the bundled part-of-speech model is a 452 KB vendored artifact,
and nothing is downloaded at build or run time.

## Usage

### `friction fix` — repair a document

```
friction fix README.md            # fixed text to stdout, summary to stderr
friction fix README.md --in-place # rewrite the file atomically
cat draft.md | friction fix -     # stdin
friction fix draft.md --suggest   # also list what was detected but held
```

Input:

> This guide will walk you through configuring the backup agent for your staging
> environment. It is important to note that the agent performs validation of the
> configuration file before each run. Once validation succeeds, the agent
> conducts an analysis of the snapshot catalog and simply uploads any missing
> segments. By following these steps, you can quickly and easily verify that
> your backups are consistent. If you have any questions or require further
> assistance, please reach out to our support team.

Output:

> This guide covers configuring the backup agent for your staging environment.
> The agent validates the configuration file before each run. Once validation
> succeeds, the agent conducts an analysis of the snapshot catalog and simply
> uploads any missing segments. You can verify that your backups are consistent.

```
friction fix: 2 pass(es), 8 patch(es) applied
  edit.recapitalize: 2
  pivot.lvc: 1
  ritual.delete: 1
  span.delete: 3
  sub.apply: 1
  suggest: 2 finding(s) remain
```

Note what it kept: "conducts an analysis of" was a valid second pivot but the
per-document budget held it, and "simply" stayed because deleting it would
create a word seam unattested in human writing. Both are reported, not forced.

Clean text passes through byte-identical:

```
$ friction fix clean.md
friction fix: 1 pass(es), 0 patch(es) applied
```

### `friction check` — detect and measure, change nothing

```
friction check draft.md --family qwen --genre blog
friction check draft.md --family gemma --format sarif > report.sarif
```

Reports detected spans with byte-exact locations, tell counts per family,
distribution metrics against the genre envelope, and the document-level
matching-statistics differential. Exit code `1` when findings exist, `0` when
clean, `2` on errors — CI-friendly, and the SARIF output validates against the
SARIF 2.1.0 schema.

`--family` is required and matters: detection indexes are specific to the
generator family they were mined from (`qwen`, `gemma`, `llama`, `granite`).
Text from a model family you have no index for will mostly evade the
statistical channel — see limits below.

### `friction explain` — why did (or didn't) it edit?

```
$ friction explain draft.md
pass 1: 8 operation(s) fired
  sub.apply                0..32   -> "This guide covers"
  span.delete              92..121   (deleted)
  pivot.lvc                131..153  -> "validates"
  ritual.delete            409..504  (deleted)
  2 held:
    pivot.lvc              194..315  KEPT (pivot held: near-no-op budget exhausted)
    span.delete            194..315  KEPT (deletion span.simply held: SeamNotAttested)
pass 2: 0 operation(s) fired — converged
```

Every range is a byte offset into your original file. Every hold names the gate
that declined.

## Guarantees

- **Deterministic** — same input, same pack, same bytes. Always.
- **Idempotent** — a second pass may clean up after first-pass deletions; a
  third pass changes nothing (enforced by a CI canary over fixtures and real
  corpus documents).
- **Near-no-op on human text** — edits per document on curated human writing
  stay under a corpus-calibrated threshold (~1.8 edits per 1000 words, ceiling).
- **Span-honest** — every reported range slices your original bytes to exactly
  the text the finding is about.
- **Closed** — there is no code path that can insert a searched-for word. The
  approaches that would (bridge insertion, corpus-path synthesis,
  metric-centroid rewriting) were tried during research, failed, and are pinned
  as red tests so they can't come back.

## What it deliberately won't do

friction removes machine framing surgically. It does not add ideas, voice, or
fluency — its ceiling is a careful copy editor with a narrow brief, and an empty
source stays empty. It is not for fiction. It is not a detector-beater. And
detection packs are specific to the model family they were built from: the
shipped indexes cover qwen/gemma/llama/granite-style output, so prose from an
unindexed family will largely pass the statistical channel untouched (the
literal inventory still applies). Growing coverage means growing the data — see
below — not cleverer search.

## The data behind it

The packs under `crates/friction-packs/packs/` are versioned, sha256-recorded,
and audited at load time (a substitution whose replacement re-triggers any
detection frame, or introduces an unattested content word, fails the build):

- `inventory-v1.toml` — the curated tell inventory: deletion spans,
  substitution pairs, ritual frames, licensed light-verb pairs, guard-token
  classes. Hand-reviewed; every mined entry carries its corpus counts.
- `dms-index-v1.toml` — per-family machine/human token streams for the
  matching-statistics channel.
- `attestation-v1.toml` — human-corpus bigram seams, tag-skeleton sets, and the
  near-no-op calibration.

They are produced by `corpus-tool` (`mine-inventory`, `mine-paired`, `index`,
`attest`) from the repository's document corpus — machine text from six local
models plus a topic-matched stock-vs-antislop paired corpus that cancels topic
so mining isolates pure style; the human side of every pair comes exclusively
from license-vetted, pre-2022 human documents. All tuning uses the train split;
a sealed holdout is guarded by CI and never touched.

For the algorithms themselves — the matching-statistics construction, mining
thresholds, gate definitions, and the validated reference prototypes — see
[docs/research/ALGORITHMS.md](docs/research/ALGORITHMS.md) and
[docs/research/ref/](docs/research/ref/).

## Development

Rust 2024 workspace. `cargo fmt --check`, `cargo clippy --workspace
--all-targets -- -D warnings`, and `cargo test --workspace` must all pass; the
test suite includes the literal accept/reject/rank fixtures from the validation
research (`docs/research/fixtures.json`) — if a change makes a reject fixture
pass or an accept fixture fail, the change is wrong, not the fixture.
