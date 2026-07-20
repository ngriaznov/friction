# Holdout report

This is the sealed-holdout evaluation: the first and only time the holdout
split (see `corpus-tool holdout-check`, `corpus/holdout.lock`) has been
measured. Everything below is report-only — no code, threshold, envelope
pack, or rule was changed in response to any number on this page, and none
will be after it either.

## Methodology

One run, no tuning. In order:

1. Verified the working tree was gate-clean before measuring anything:
   `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets
   --all-features -- -D warnings`, `cargo test --workspace`, and
   `corpus-tool holdout-check` all passed first (110 holdout lines
   verified against the manifest and on-disk files).
2. Built the release CLI once: `cargo build --release -p friction-cli`.
3. Ran `corpus-tool separate-holdout --report corpus/HOLDOUT_REPORT.md`
   (a new sibling subcommand to `corpus-tool separate`, added for this
   evaluation — see `crates/corpus-tool/src/commands/separate_holdout.rs`).
   For every `split: holdout` manifest record it computed:
   - **human-holdout**: each human-holdout document's metric vector,
     untouched.
   - **llm-holdout**: each llm-holdout document's metric vector,
     untouched — the baseline.
   - **fixed-llm-holdout**: the same llm-holdout documents after running
     `target/release/friction fix <path> --genre <genre>` (the release
     binary, invoked as a subprocess, once per document, output captured
     into a fresh temp directory) — the tool's measured effect.

   All three groups were scored against the same envelope pack
   `corpus-tool envelope` already froze from the train split
   (`crates/friction-packs/packs/envelope-v2.toml` — the exact pack
   `corpus-tool separate`'s dev-split report also uses), via the same
   combined score and Mann-Whitney AUC code `corpus-tool separate`
   already defines (`crate::commands::separate::{combined_score,
   mann_whitney_auc}`, reused, not reimplemented, by the new command).
4. Ran the near-no-op report
   (`cargo run -p friction-apply --release --example near_noop_report`,
   extended to add a HOLDOUT section alongside its existing TRAIN/DEV
   ones) over every human-holdout document.
5. Ran the corpus-wide idempotence sweep, holdout-scoped
   (`cargo test -p friction-apply --release --test idempotence_sweep --
   --ignored idempotence_sweep_holdout --nocapture`), over every one of
   the 110 holdout documents (60 human, 50 llm).

Quarantined (CC-BY-SA) human docs are not excluded from any of the above —
quarantine only restricts redistributing document *text* in a shipped
pack, not measuring it, matching `corpus-tool separate`'s own convention
on the dev split.

## AUC table (baseline vs after-fix, per genre)

AUC is the Mann-Whitney U statistic, tie-corrected via midranks, oriented
so AUC > 0.5 always means the two groups compared separate; both columns
below score `llm higher` (the llm/fixed-llm group's combined score is the
larger one in every genre, both before and after fixing).

| genre | human n | llm n | baseline AUC (human vs llm) | after-fix AUC (human vs fixed-llm) |
|---|---|---|---|---|
| docs | 13 | 10 | 0.9462 (llm higher) | 0.9308 (llm higher) |
| blog | 15 | 10 | 0.9867 (llm higher) | 0.9800 (llm higher) |
| readme | 13 | 10 | 0.9000 (llm higher) | 0.8923 (llm higher) |
| email | 7 | 10 | 1.0000 (llm higher) | 1.0000 (llm higher) |
| forum | 12 | 10 | 0.9083 (llm higher) | 0.8833 (llm higher) |

Combined-score metrics included per genre (of the 21 `MetricVector`
fields, per the train-derived `include` rule already baked into the
envelope pack — see `corpus-tool separate`'s own report for the same
notes on the dev split):

- **docs**: 15 of 21 included; excluded: `bullet_parallelism` (train AUC
  0.5051, llm higher), `em_dash_density` (train AUC 0.5202, llm lower),
  `heading_density` (train AUC 0.5463, llm lower), `not_just_but_rate`
  (train AUC 0.5024, llm higher), `paragraph_shape_mean` (train AUC
  0.5191, llm lower), `ritual_marker_rate` (train AUC 0.5000, llm higher).
- **blog**: 19 of 21 included; excluded: `paragraph_shape_mean` (train AUC
  0.5147, llm lower), `sentence_length_mean` (train AUC 0.5166, llm
  higher).
- **readme**: 14 of 21 included; excluded: `bullet_parallelism` (train AUC
  0.5336, llm lower), `contraction_ratio` (train AUC 0.5147, llm higher),
  `discourse_marker_density` (train AUC 0.5252, llm higher),
  `em_dash_density` (train AUC 0.5130, llm lower), `not_just_but_rate`
  (train AUC 0.5000, llm higher), `ritual_marker_rate` (train AUC 0.5109,
  llm higher), `semicolon_density` (train AUC 0.5454, llm lower).
- **email**: 18 of 21 included; excluded: `contraction_ratio` (train AUC
  0.5072, llm lower), `not_just_but_rate` (train AUC 0.5036, llm lower),
  `semicolon_density` (train AUC 0.5355, llm lower).
- **forum**: 17 of 21 included; excluded: `bullet_parallelism` (train AUC
  0.5134, llm higher), `em_dash_density` (train AUC 0.5047, llm lower),
  `top_opener_concentration` (train AUC 0.5126, llm higher), `triad_rate`
  (train AUC 0.5104, llm higher).

## Combined-score distributions

The combined score is the mean, over a document's genre's *included*
metrics, of a per-metric normalized directional exceedance beyond that
metric's train-human envelope band (`0.0` inside the band). Lower is more
human-like.

| genre | group | n | mean | median |
|---|---|---|---|---|
| docs | human-holdout | 13 | 0.0367 | 0.0307 |
| docs | llm-holdout (raw) | 10 | 0.1680 | 0.1712 |
| docs | llm-holdout (fixed) | 10 | 0.1596 | 0.1712 |
| blog | human-holdout | 15 | 0.0299 | 0.0257 |
| blog | llm-holdout (raw) | 10 | 0.1527 | 0.1531 |
| blog | llm-holdout (fixed) | 10 | 0.1392 | 0.1336 |
| readme | human-holdout | 13 | 0.0508 | 0.0182 |
| readme | llm-holdout (raw) | 10 | 0.1693 | 0.1561 |
| readme | llm-holdout (fixed) | 10 | 0.1600 | 0.1297 |
| email | human-holdout | 7 | 0.0461 | 0.0555 |
| email | llm-holdout (raw) | 10 | 0.4249 | 0.4113 |
| email | llm-holdout (fixed) | 10 | 0.4174 | 0.4186 |
| forum | human-holdout | 12 | 0.0377 | 0.0171 |
| forum | llm-holdout (raw) | 10 | 0.1289 | 0.1349 |
| forum | llm-holdout (fixed) | 10 | 0.1184 | 0.1342 |

Fixing moves the llm-holdout mean toward the human-holdout mean in every
genre, by a modest amount: `docs` −5.0% relative (closing 6.4% of the
raw-to-human gap), `blog` −8.8% (11.0% of the gap), `readme` −5.5% (7.8%
of the gap), `email` −1.8% (2.0% of the gap), `forum` −8.1% (11.5% of the
gap). No genre's fixed-llm mean gets within even a third of the way to its
human mean.

## Near-no-op on human holdout

Ran the near-no-op machinery (`friction_apply::touched_original_ranges`
over `FixEngine::fix_document`) across every human-holdout document,
reporting the percentage of sentences that received at least one applied
patch. Ceiling: 2.0% overall.

| genre | docs | sentences | touched | % |
|---|---:|---:|---:|---:|
| docs | 13 | 1573 | 0 | 0.000 |
| blog | 15 | 1543 | 9 | 0.583 |
| readme | 13 | 1275 | 2 | 0.157 |
| email | 7 | 1303 | 0 | 0.000 |
| forum | 12 | 1389 | 5 | 0.360 |
| **overall** | 60 | 7083 | 16 | 0.226 |

**Result: 0.226% overall — well within the 2.0% ceiling.** (Full table,
alongside the TRAIN and DEV splits it was already reporting, in
`corpus/NEARNOOP.md`.)

## Idempotence

`fix(fix(x)) == fix(x)`, byte-for-byte, checked over all 110 holdout
documents (60 human, 50 llm) via `FixEngine::fix_document` run twice per
document (`cargo test -p friction-apply --test idempotence_sweep --
--ignored idempotence_sweep_holdout`).

**Result: 110/110 idempotent. No divergence found.**

## Conclusion

The metrics layer separates human from LLM holdout prose about as well as
it did on the dev split (dev combined-score AUCs ran 0.8636–1.0000 across
the five genres in `corpus/SEPARATION.md`; holdout baseline AUCs run
0.9000–1.0000) — the stratified split held, and separation generalizes to
documents the metrics and envelope pack never saw. Running the release
`fix` CLI over every llm-holdout document moves the combined-score
distribution in the right direction in all five genres — mean and AUC
both shift, if only slightly, toward the human envelope, never away from
it — but the movement is small: a 2–11% (genre-dependent) reduction of the
raw-to-human combined-score gap, and an AUC change of at most 0.025 in any
genre (`email`, already at a ceiling of 1.0000 before fixing, cannot move
at all). No genre's fixed-llm distribution comes anywhere close to
overlapping the human envelope the way the gate's 0.85 dev-split threshold
implies it eventually should.

This is the expected shape of result for a deterministic, rule-based
fixer scored against a metric vector where most of the discriminating
signal is structural and lexical rather than content-level. Of the 21
tracked metrics, several with real separating power in a given genre
(`sentence_length_mean`, `paragraph_shape_mean`/`cv`,
`sentence_opener_repeat_rate`, `top_opener_concentration`, `em_dash_density`,
`semicolon_density`) have no corresponding Fix-tier rule at all in the
current six-family rule set — nothing in `friction-rules` ever proposes a
patch that would move them. A few more (`not_just_but_rate`, `triad_rate`,
and `heading_density` via `structural.header_merge`) are scanned and
reported but, by design, never auto-applied (`Tier::Suggest` only). The
metrics a Fix-tier rule *can* move — `contraction_ratio`
(`contraction.insert`), `discourse_marker_density`
(`connective.surgery`), `llm_favored_phrase_rate`/`human_favored_phrase_rate`
(`lexical.substitution`, `lexical.filler_phrase`), `participial_closer_rate`
(`symmetry.participial_closer`), `bullet_parallelism`/`list_item_density`
(`structural.unbullet`), `bold_span_density`
(`structural.bold_label_strip`), `ritual_marker_rate`
(`symmetry.ritual_conclusion`), and sentence length via `rhythm.split` —
are real, and the near-no-op result above confirms the tool applies them
conservatively (well under 2% of human-holdout sentences touched at all)
rather than over-firing on llm text to chase the number. But averaged
across a combined score built from twenty-one metrics, moving a handful of
them by a bounded, conservative amount was never going to close a gap this
size on its own.

Put plainly: fixing reduces surface/lexical/rhythm LLM tics — the specific
things the six rule families target — and the holdout numbers show that
reduction is real, not noise, in every genre. It does not, and by design
cannot, touch the content-level tells the metric vector doesn't measure at
all: genericity, hedging both sides of a claim, and absence of concrete,
checkable detail. Those remain in the fixed output exactly as they were in
the raw LLM output, because nothing in this engine looks for them. Closing
the rest of the gap would require either metrics that capture those
content-level properties, or accepting that a purely surface-level,
deterministic tool has a ceiling on how close it can bring LLM prose to
the human envelope — this holdout run is evidence for where that ceiling
currently sits, not a claim that it has been reached optimally.
