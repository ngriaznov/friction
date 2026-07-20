//! Golden before/after fixtures, budget/ordering, and multi-round
//! convergence checks for [`friction_rules::SentenceSplitRule`].
//!
//! Every fixture pair lives under `tests/golden/rhythm/<name>.before.md` /
//! `<name>.after.md`. This file drives them through a small, local
//! single-round harness ([`run_round`]) that mirrors `friction-apply`'s
//! driver semantics for exactly one rule (gate once, scan, walk findings in
//! `SentenceSplitRule::scan`'s own order, check budget *before* each `fix`
//! call, apply the resulting non-overlapping patches) — the same
//! documented stand-in `crate::families::connective`'s own golden-fixture
//! harness uses, and for the same reason: this crate does not depend on
//! `friction-apply` (that dependency runs the other way).
//!
//! [`run_until_fixed_point`] goes one step further for the one fixture
//! that genuinely needs it (`two_boundaries_needs_two_rounds`): it re-runs
//! [`run_round`] against the *real*, freshly re-measured
//! `sentence_length_cv` of each round's own output (via
//! `friction_metrics::compute`, exactly as `friction-apply`'s real driver
//! would), stopping the first time a round applies zero patches — proof
//! that a sentence needing two splits actually converges to a stable fixed
//! point within a small, bounded number of rounds rather than oscillating.

use std::fs;
use std::path::Path;

use friction_core::{Document, Envelope, Finding, MetricVector, Patch, Tier, span};
use friction_nlp::{NlpruleTagger, SrxSegmenter};
use friction_rules::{Gate, MapEnvelope, Rule, RuleContext, SentenceSplitRule, StrategyRng};

/// The `MetricVector` field `SentenceSplitRule` gates on.
const GATED_METRIC: &str = "sentence_length_cv";

/// A `MetricVector` with only `sentence_length_cv` set — the only field
/// this rule's `gate` reads.
fn metrics(cv: f64) -> MetricVector {
    MetricVector {
        sentence_length_cv: cv,
        ..MetricVector::default()
    }
}

/// The source bytes of the sentence containing `finding`, for seeding a
/// [`StrategyRng`] — a local copy of `friction-apply`'s own
/// `sentence_bytes_for` helper (documented there; not importable, since
/// this crate cannot depend on `friction-apply`). `SentenceSplitRule::fix`
/// never actually consults its `StrategyRng` argument (a split boundary has
/// exactly one meaning-preserving fix), but this harness still seeds and
/// threads one through, mirroring the real driver's own shape exactly.
fn sentence_bytes_for<'a>(document: &'a Document, finding: &Finding) -> &'a [u8] {
    for unit in document.prose() {
        for sentence in &unit.sentences {
            if span::contains_range(&sentence.range, &finding.range) {
                return document
                    .text(&sentence.range)
                    .expect("sentence ranges are already validated against the document")
                    .as_bytes();
            }
        }
    }
    document.text(&finding.range).map_or(&[], str::as_bytes)
}

/// Applies non-overlapping `patches` to `source` in one right-to-left pass
/// — a local copy of `friction-apply`'s own `apply_patches` mechanics.
fn apply_patches(source: &str, patches: &[Patch]) -> String {
    let mut ordered: Vec<&Patch> = patches.iter().collect();
    ordered.sort_by_key(|patch| std::cmp::Reverse(patch.range.start));
    let mut result = source.to_string();
    for patch in ordered {
        result.replace_range(patch.range.clone(), patch.replacement.as_str());
    }
    result
}

/// One local, single-round pass of gate -> scan -> (budgeted) fix -> apply,
/// scoped to [`SentenceSplitRule`] alone, for a given `genre` and
/// already-known `cv`. Returns the resulting text and how many patches were
/// applied.
fn run_round(
    source: &str,
    cv: f64,
    band: Envelope,
    genre: &str,
    tagger: &NlpruleTagger,
) -> (String, usize) {
    let parsed = friction_parse::parse(source).expect("fixture source is valid markdown");
    let document = friction_nlp::segment_document(&parsed, &SrxSegmenter::new())
        .expect("fixture source segments cleanly");
    let envelope = MapEnvelope::new().with(GATED_METRIC, band);
    let ctx = RuleContext::new(&document, tagger, genre, &envelope);
    let rule = SentenceSplitRule::new();

    let Gate::Fix { mut budget } = rule.gate(&metrics(cv), &envelope) else {
        return (source.to_string(), 0);
    };

    let mut patches: Vec<Patch> = Vec::new();
    for finding in rule.scan(&ctx) {
        if finding.tier != Tier::Fix || budget.is_exhausted() {
            continue;
        }
        let sentence_bytes = sentence_bytes_for(&document, &finding);
        let mut rng = StrategyRng::seeded(sentence_bytes, rule.id());
        if let Some(patch) = rule.fix(&finding, &ctx, &mut rng)
            && patch.tier == Tier::Fix
        {
            budget = budget
                .take_one()
                .expect("budget was checked non-exhausted above");
            patches.push(patch);
        }
    }

    let applied = patches.len();
    (apply_patches(source, &patches), applied)
}

/// Reads `tests/golden/rhythm/<name>.before.md` / `<name>.after.md`,
/// stripping no bytes — the fixture files' own trailing newline is part of
/// the compared content.
fn read_fixture(name: &str) -> (String, String) {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden/rhythm");
    let before = fs::read_to_string(dir.join(format!("{name}.before.md")))
        .unwrap_or_else(|e| panic!("reading {name}.before.md: {e}"));
    let after = fs::read_to_string(dir.join(format!("{name}.after.md")))
        .unwrap_or_else(|e| panic!("reading {name}.after.md: {e}"));
    (before, after)
}

fn tagger() -> NlpruleTagger {
    NlpruleTagger::new().expect("embedded tagger model loads")
}

/// Every single-round fixture with the `(cv, envelope band, genre)` that
/// reproduces each one's `after.md` exactly in one [`run_round`] pass — see
/// this file's module docs for the harness these are run through.
///
/// `two_boundaries_needs_two_rounds` is deliberately not listed here: it
/// needs a *second* round (with the real, re-measured `sentence_length_cv`
/// of round one's own output) to reach its `after.md`, so it is exercised
/// separately by [`two_boundaries_needs_two_rounds_converges_without_oscillating`]
/// via [`run_until_fixed_point`] instead.
const SINGLE_ROUND_FIXTURES: &[(&str, f64, Envelope, &str)] = &[
    // A single over-long sentence with one semicolon boundary; a generous
    // budget (10) lets `scan`'s single finding through untouched by the
    // budget check.
    ("semicolon_split", 0.2, Envelope::new(0.6, 1.0), "docs"),
    // A single over-long sentence whose only boundary is a `", and "`
    // coordinator followed by the pronoun "it".
    (
        "coordinator_and_split",
        0.2,
        Envelope::new(0.6, 1.0),
        "docs",
    ),
    // Same shape, `", but "` coordinator followed by the determiner "the".
    (
        "coordinator_but_split",
        0.2,
        Envelope::new(0.6, 1.0),
        "docs",
    ),
    // `sentence_length_cv` already inside [0, 100]: gate is Off, so the
    // rule never scans at all, even though the sentence is well over the
    // genre's over-long threshold and has a perfectly good semicolon
    // boundary.
    (
        "no_op_cv_in_envelope",
        5.0,
        Envelope::new(0.0, 100.0),
        "docs",
    ),
    // Two over-long sentences; band lo 0.66, cv 0.60 -> deficit ~0.06 ->
    // budget floor(0.06 / 0.05) = 1 (a deficit picked comfortably clear of
    // the exact `0.05` grid, like `split.rs`'s own hand-computed budget
    // test, so the assertion is not sensitive to `f64` subtraction's own
    // rounding). The *second* (textually) sentence is longer (25 tokens
    // vs. 21), so `SentenceSplitRule::scan`'s length-descending order
    // processes it first and the budget runs out before the shorter,
    // textually-first sentence is ever offered a fix — proof that
    // processing order is length-first, not source-first.
    (
        "budget_stops_length_desc",
        0.60,
        Envelope::new(0.66, 1.0),
        "docs",
    ),
];

/// Every single-round fixture's `before.md`, run through [`run_round`] with
/// its documented `(cv, band, genre)`, reproduces `after.md` exactly.
#[test]
fn golden_fixtures_match_expected_output() {
    let tagger = tagger();
    for &(name, cv, band, genre) in SINGLE_ROUND_FIXTURES {
        let (before, after) = read_fixture(name);
        let (actual, _applied) = run_round(&before, cv, band, genre, &tagger);
        assert_eq!(
            actual, after,
            "fixture {name:?} did not match its golden output"
        );
    }
}

/// The `budget_stops_length_desc` fixture specifically applies exactly 1
/// patch (its documented budget) even though the text contains 2
/// over-long, splittable sentences.
#[test]
fn budget_stops_length_desc_applies_exactly_its_budget() {
    let (before, _after) = read_fixture("budget_stops_length_desc");
    let tagger = tagger();
    let (_actual, applied) = run_round(&before, 0.60, Envelope::new(0.66, 1.0), "docs", &tagger);
    assert_eq!(applied, 1);
}

/// The `no_op_cv_in_envelope` fixture applies zero patches — the gate
/// itself is `Off`.
#[test]
fn no_op_fixture_applies_zero_patches() {
    let (before, _after) = read_fixture("no_op_cv_in_envelope");
    let tagger = tagger();
    let (_actual, applied) = run_round(&before, 5.0, Envelope::new(0.0, 100.0), "docs", &tagger);
    assert_eq!(applied, 0);
}

/// Idempotence, for every single-round fixture's `before.md`: running
/// [`run_round`] once with a deliberately generous budget (so nothing is
/// left unprocessed purely for lack of budget), then running it again on
/// that output with the same generous budget, changes nothing further and
/// applies zero additional patches.
///
/// Deliberately *not* the same `(cv, band)` each fixture's golden-output
/// test uses — `budget_stops_length_desc`'s golden output is an
/// intentionally partial single round, not a fixed point, so checking
/// idempotence against it under that same tight budget would just be
/// re-observing the budget stopping again. A generous budget isolates the
/// real property: does `scan` ever refind something `fix` already handled?
#[test]
fn idempotent_across_single_round_fixtures_given_a_generous_budget() {
    let generous_band = Envelope::new(0.99, 1.0);
    let generous_cv = 0.0;
    let tagger = tagger();

    for &(name, _cv, _band, genre) in SINGLE_ROUND_FIXTURES {
        let (before, _after) = read_fixture(name);
        let (once, _first_applied) = run_round(&before, generous_cv, generous_band, genre, &tagger);
        let (twice, second_applied) = run_round(&once, generous_cv, generous_band, genre, &tagger);
        assert_eq!(
            once, twice,
            "fixture {name:?}: fixing its own already-fixed output changed it"
        );
        assert_eq!(
            second_applied, 0,
            "fixture {name:?}: re-fixing already-fixed output applied more patches"
        );
    }
}

/// Runs [`run_round`] repeatedly against a fixed `(band, genre)`, but with
/// each round's `sentence_length_cv` *re-measured for real* from the
/// previous round's own resulting text (via `friction_metrics::compute`,
/// exactly as `friction-apply`'s real fixpoint driver does — see
/// `friction_apply::run_fixpoint`'s own docs) — stopping the first time a
/// round applies zero patches, or after `max_rounds`, whichever comes
/// first.
///
/// Returns the final text and how many patches each round applied, in
/// round order — a fixture that genuinely needs more than one round shows
/// up as more than one non-zero entry before the trailing zero; a fixture
/// whose logic oscillated (kept "fixing" the same already-fixed text
/// forever, e.g. a hypothetical bug where a boundary marker was not fully
/// consumed by its own patch) would instead never reach a zero-patch round
/// at all within `max_rounds`, which this function's own
/// `max_rounds`-exhaustion path makes an observable, assertable failure
/// rather than a silent infinite loop.
fn run_until_fixed_point(
    source: &str,
    band: Envelope,
    genre: &str,
    tagger: &NlpruleTagger,
    max_rounds: usize,
) -> (String, Vec<usize>) {
    let segmenter = SrxSegmenter::new();
    let mut current = source.to_string();
    let mut applied_per_round = Vec::new();

    for _ in 0..max_rounds {
        let parsed =
            friction_parse::parse(current.as_str()).expect("round source is valid markdown");
        let metrics = friction_metrics::compute(&parsed, &segmenter, tagger);
        let (next, applied) = run_round(&current, metrics.sentence_length_cv, band, genre, tagger);
        applied_per_round.push(applied);
        current = next;
        if applied == 0 {
            break;
        }
    }

    (current, applied_per_round)
}

/// The idempotence subtlety this rule's module docs call out: a sentence
/// with two semicolon boundaries, far enough over the genre's threshold
/// that splitting it once still leaves one half over-long (with the
/// *other* semicolon still in it), converges to a stable three-sentence
/// fixed point within a small, bounded number of rounds — genuine
/// multi-round progress, not an oscillation where the same "fix" keeps
/// reapplying to unchanged text forever.
#[test]
fn two_boundaries_needs_two_rounds_converges_without_oscillating() {
    let (before, after) = read_fixture("two_boundaries_needs_two_rounds");
    let tagger = tagger();
    let band = Envelope::new(0.6, 1.0);

    let (final_text, applied_per_round) =
        run_until_fixed_point(&before, band, "readme", &tagger, 6);

    assert_eq!(
        final_text, after,
        "expected the fixed point to match the hand-derived after.md"
    );
    // Genuine two-round progress (one split per round on the still-too-long
    // remainder), then a third, empty round confirms the fixed point —
    // not one giant round, and not stuck oscillating for all 6 allotted
    // rounds.
    assert_eq!(
        applied_per_round,
        vec![1, 1, 0],
        "expected exactly two splitting rounds followed by one confirming empty round"
    );

    // Idempotence: fixing the already-converged output again changes
    // nothing further.
    let (after_again, applied_again) = run_until_fixed_point(&after, band, "readme", &tagger, 2);
    assert_eq!(after_again, after);
    assert_eq!(applied_again, vec![0]);
}
