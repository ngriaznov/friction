# Extending friction

Where each kind of expansion goes, what it must update, and how to prove
it safe before it ships. [OPERATIONS.md](OPERATIONS.md) explains what the
engine does; this file explains how to add to it without breaking the
discipline that keeps it deterministic.

The one rule over all of them: **every change is either byte-fenced or
intentional.** A refactor or optimization must produce byte-identical
`fix` and `check` output on the corpus (see [Verification](#verification)
below). A behavior change must name every output it changes and carry the
corpus evidence for why.

## Adding a curated inventory entry

Substitutions, gated span deletions, and ritual frames live in
`crates/friction-packs/packs/inventory-en-v1.toml`.

- `substitution_pairs` need `id`, `pattern`, `replacement`, `repair`,
  `source`, `attested_replacement_tokens`, and `notes` recording the
  rejected alternatives. Structural rule: a substitution always has a
  real replacement.
- `deletion_spans` and `ritual_frames` may set
  `repair = "diagnostic_only"` for detect-only entries — the engine emits
  a finding and never edits (enforced in `friction-edit/src/sentence.rs`
  at both deletion sites; a test suite pins it). `deletion_spans` require
  an `anchor` (`sentence_initial` | `mid_sentence` | `trailing`).
- Gates (seam attestation, finite-verb survival, closure allowance) apply
  automatically; do not try to bypass them per entry.

Add an engine test in `crates/friction-edit/tests/` (the `sub_vacuous.rs`
suites are the template), then run the snapshot ritual.

## Adding a frame rule (collocation rewrite)

Rules live in `crates/friction-packs/packs/frame-rules-en-v1.toml` and
compile into `frame-pack-en-v1.bin` via `corpus-tool frame-pack`, which
runs the rejection gauntlet: anchor evidence, human-rate ceilings,
attestation floors, machine-tilt fences.

**The gauntlet is the authority.** If it rejects a rule, the rule is
wrong, not the gate. Record the rejection in the TOML notes and stop. On
record: an adjective-deletion class for "valuable/actionable insights"
measured clean at the phrase level but was rejected because the bare
anchor "insight" is human-tilted (0 machine vs 11 human) and "insights"
sits under the 20/M attestation floor. The rejection stands.

Mechanics: measure evidence with `corpus-tool adjudicate --regenerate`,
recompile with `corpus-tool frame-pack`, update the rule/class count pins
in `crates/friction-packs/src/frame_rules.rs`'s test, add an engine test
in `crates/friction-edit/tests/frame_op.rs` style, snapshot ritual.

## Adding a register feature (cadence detector)

Detectors live in `crates/friction-register/src/features.rs`. Two count
structs sit side by side, on purpose:

- `RegisterCounts` — all detectors, pinned against the reference
  implementation by `tests/feature_parity.rs`. Every new detector gets a
  field here.
- `CoreCounts` — only the features the engine reads per sentence at
  runtime (`friction-edit/src/register.rs`, `count_features`). Add a
  field here **only** when the engine actually consumes it; the parity
  suite pins the two structs against each other, so they cannot silently
  disagree.

Activation is staged-inert: land the counter and plumbing with no band in
`register-en-v1.toml` (byte-identical output — the fence proves it), then
measure bands with `corpus-tool register-bands` over the human corpus and
activate via the measured TOML in a separate commit. Nonzero-high bands
arm only at two or more instances (the Wilson-bound floor in
`confidently_above`); zero-high bands arm on one.

## Adding a transducer (rewrite rule with grammar)

Candidates are built in `crates/friction-register/src/transduce.rs` (a
`CandidateKind` variant plus a builder that declines fail-closed), then
plumbed through `collect_candidates` in `friction-edit/src/register.rs`.
Selection is purely rate-improvement scoring — there is no per-candidate
confidence knob.

Before building, read the refutation records in `register-en-v1.toml`'s
band notes. Two shipped-then-reverted transducers are documented there:
comma-and sentence splitting (rhythm: robotic) and contrast-tail deletion
(the same byte shape carries meaning-bearing contrasts in docs). A new
transducer over a similar shape needs new evidence, not a retry.

## Adding a restructuring rule (parse-gated, non-cadence)

Clause restructuring lives beside the transducers but answers a
different selection question: no band, no rate: a correct rewrite
applies per instance, like a frame rule. Candidates build in
`friction-register/src/transduce.rs` (T10/T11 are the models: walk the
`SentenceParse`, require the construction's relations plus independent
POS corroboration, decline with a named reason on any anomaly), and the
pass in `friction-edit/src/restructure.rs` gates every candidate through
the standard rewrite gates and applies every survivor. Two disciplines
are specific to this class:

- **The trigger prefilter is part of the contract.** The pass skips
  tagging and parsing for sentences without a literal trigger word, so
  every new rule must have a closed lexical trigger and register its
  stem where `run_restructure` builds the list. A rule without one
  belongs elsewhere.
- **A dobj-gated substitution is one table line** —
  `[transitive_verbs]` in `lexicon-en.toml` maps the machine-tilted
  lemma to its attested replacement — but the entry owes the same
  evidence a `substitution_pairs` replacement does, plus a written
  sense check: see the `surface` entry's recorded caveat for what
  happens when one lemma carries two senses and only one is safe.

The compile fence also rejects any frame rule whose template cannot
realize against its own pattern ("unrealizable template"); if a
construction needs a data-dependent derivation the frame grammar cannot
express, that rejection is the signal it belongs here instead.

## Adding a metric

Metrics live in `crates/friction-metrics/`, assembled by
`compute_segmented` in `src/compute.rs`, which runs the expensive
document passes **once** and threads the shared results in:

- needs POS tags → take `&[Vec<TaggedToken>]` (the `*_from_tagged`
  pattern in `symmetry.rs`); never call the tagger inside a metric on the
  `compute_segmented` path.
- needs a word-token denominator → take the precomputed total (the
  `*_with_total` pattern in `signals.rs`).

Standalone `pub` wrappers that do their own pass exist for external
callers and tests; keep new metrics to the same two-layer shape.

## Hot-path invariants (enforced, not just documented)

These hold today by construction and are guarded by `debug_assert`s and
pinning tests, so `cargo test` fails loudly if a change violates them:

- **Arc-eager writes each token's arc exactly once** — the parser's
  incremental `children_bounds` index depends on it
  (`friction-nlp/src/dep_arceager.rs`).
- **Tokens and sentences ascend in source order** — the frame scan's
  monotone cursor depends on it (`friction-match/src/frame.rs`).
- **The hoisted working text always equals the splicer chain** when a
  pattern is tested (`friction-edit/src/sentence.rs`).
- **`CoreCounts` agrees with `RegisterCounts`** on shared fields
  (`friction-register/tests/feature_parity.rs`).

If a legitimate future change needs to break one of these (e.g. an arc
re-linking parser transition), the assert tells you every place that must
be redesigned with it. Delete the optimization, not just the assert.

## Verification

- **Tests**: `cargo test --workspace` — every suite green, no exceptions.
- **Snapshot ritual**: `INSTA_UPDATE=always cargo test -p friction-cli
  --test snapshot`, then `git diff` the `.snap` files and hand-review
  every changed line. A diff you cannot explain is a bug, not noise.
  Never commit a snapshot change from a change that claimed to be
  byte-safe.
- **Byte-fence** (for refactors/optimizations): build the baseline binary
  from `main`, build the candidate, run both over the corpus
  (`corpus/llm`, `corpus/review`, `corpus/human`) hashing `fix` stdout
  and `check --format json` stdout per file, and diff the hash lists.
  Zero unexplained differences or the change does not land.
- **Docs**: run `friction fix` on any prose you touched — the tool's own
  output on its own documentation is the cheapest review it has.
