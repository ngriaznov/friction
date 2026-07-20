//! Golden before/after fixtures and idempotence checks for
//! [`friction_rules::ConnectiveSurgery`].
//!
//! Every fixture pair lives under `tests/golden/connective/<name>.before.md`
//! / `<name>.after.md`. This file drives them through a small, local
//! single-round harness ([`run_round`]) that mirrors `friction-apply`'s
//! driver semantics for exactly one rule (gate once, scan, walk findings in
//! source order, check budget *before* each `fix` call, apply the
//! resulting non-overlapping patches) — this crate does not depend on
//! `friction-apply` (that dependency runs the other way), so the harness
//! is a deliberate, documented stand-in rather than a call into the real
//! driver.
//!
//! Each fixture's expected `after.md` content was derived by hand from
//! this rule's documented span rules (delete: connective + comma +
//! whitespace, recapitalizing what follows if it starts lowercase; swap:
//! connective + comma -> the class's coordinator) together with the exact
//! strategy each specific sentence's hash-seeded draw resolves to —
//! verified independently of this crate's own Rust implementation via a
//! standalone script that reimplements `xxh64` seeding (via the `xxhash`
//! `PyPI` package, a separate implementation of the same standard
//! algorithm) and splitmix64 (transcribed by hand from the published
//! algorithm, the same one `friction_rules::StrategyRng`'s own doc
//! comment cites) and applies `ConnectiveSurgery`'s documented 45/45/10
//! strategy bucketing.

use std::fs;
use std::path::Path;

use friction_core::{Document, Envelope, Finding, MetricVector, Patch, Tier, span};
use friction_nlp::{SrxSegmenter, Tagger};
use friction_rules::{ConnectiveSurgery, Gate, MapEnvelope, Rule, RuleContext, StrategyRng};

/// A stub tagger; `ConnectiveSurgery` never consults part-of-speech tags.
struct NoopTagger;
impl Tagger for NoopTagger {
    fn tag(&self, _text: &str, _base_offset: usize) -> Vec<friction_nlp::TaggedToken> {
        Vec::new()
    }
}

/// The `MetricVector` field `ConnectiveSurgery` gates on.
const GATED_METRIC: &str = "discourse_marker_density";

/// A `MetricVector` with only `discourse_marker_density` set — the only
/// field this rule's `gate` reads.
fn metrics(density: f64) -> MetricVector {
    MetricVector {
        discourse_marker_density: density,
        ..MetricVector::default()
    }
}

/// The source bytes of the sentence containing `finding`, for seeding a
/// [`StrategyRng`] — a local copy of `friction-apply`'s own
/// `sentence_bytes_for` helper (documented there; not importable, since
/// this crate cannot depend on `friction-apply`).
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

/// Applies non-overlapping `patches` to `source` in one right-to-left
/// pass — a local copy of `friction-apply`'s own `apply_patches`
/// mechanics (see that crate for the full doc), safe here because every
/// patch `run_round` collects belongs to a distinct sentence and sentences
/// never overlap.
fn apply_patches(source: &str, patches: &[Patch]) -> String {
    let mut ordered: Vec<&Patch> = patches.iter().collect();
    ordered.sort_by_key(|patch| std::cmp::Reverse(patch.range.start));
    let mut result = source.to_string();
    for patch in ordered {
        result.replace_range(patch.range.clone(), patch.replacement.as_str());
    }
    result
}

/// One local, single-round pass of gate -> scan -> (budgeted) fix ->
/// apply, scoped to [`ConnectiveSurgery`] alone. Returns the resulting
/// text and how many patches were applied. See this file's module docs
/// for why this exists instead of calling `friction-apply` directly.
fn run_round(source: &str, density: f64, band: Envelope) -> (String, usize) {
    let parsed = friction_parse::parse(source).expect("fixture source is valid markdown");
    let document = friction_nlp::segment_document(&parsed, &SrxSegmenter::new())
        .expect("fixture source segments cleanly");
    let envelope = MapEnvelope::new().with(GATED_METRIC, band);
    let ctx = RuleContext::new(&document, &NoopTagger, "blog", &envelope);
    let rule = ConnectiveSurgery::new();

    let Gate::Fix { mut budget } = rule.gate(&metrics(density), &envelope) else {
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

/// Reads `tests/golden/connective/<name>.before.md` /
/// `<name>.after.md`, stripping no bytes — the fixture files' own
/// trailing newline is part of the compared content.
fn read_fixture(name: &str) -> (String, String) {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden/connective");
    let before = fs::read_to_string(dir.join(format!("{name}.before.md")))
        .unwrap_or_else(|e| panic!("reading {name}.before.md: {e}"));
    let after = fs::read_to_string(dir.join(format!("{name}.after.md")))
        .unwrap_or_else(|e| panic!("reading {name}.after.md: {e}"));
    (before, after)
}

/// All four connective fixtures with the `(density, envelope band)` that
/// reproduces each one's `after.md` exactly — see this file's module docs
/// for how each was derived.
const FIXTURES: &[(&str, f64, Envelope)] = &[
    // Four connectives, generous budget (10): every occurrence gets a
    // strategy draw, and (verified independently) all four land on
    // Delete or Swap, none on "leave unchanged" — a document-wide
    // demonstration of both active fix strategies.
    ("mixed_delete_and_swap", 10.0, Envelope::new(0.0, 0.0)),
    // Four connectives again, but budget 2: only the first two findings
    // in source order (Moreover -> Delete, However -> Swap) consume it;
    // the third and fourth are never even offered a strategy draw.
    ("budget_stops_mid_document", 2.0, Envelope::new(0.0, 0.0)),
    // Density already inside [0, 100]: gate is Off, so the rule never
    // scans at all, regardless of how many connectives the text has.
    ("no_op_density_in_envelope", 5.0, Envelope::new(0.0, 100.0)),
    // "In addition to ..." (no comma right after "In addition") must not
    // match; "Nevertheless," genuinely does and (verified independently)
    // draws Swap.
    ("false_positive_guard", 1.0, Envelope::new(0.0, 0.0)),
];

/// Every fixture's `before.md`, run through [`run_round`] with its
/// documented `(density, band)`, reproduces `after.md` exactly.
#[test]
fn golden_fixtures_match_expected_output() {
    for &(name, density, band) in FIXTURES {
        let (before, after) = read_fixture(name);
        let (actual, _applied) = run_round(&before, density, band);
        assert_eq!(
            actual, after,
            "fixture {name:?} did not match its golden output"
        );
    }
}

/// The `budget_stops_mid_document` fixture specifically applies exactly 2
/// patches (its documented budget) even though the text contains 4
/// connective occurrences.
#[test]
fn budget_stops_mid_document_applies_exactly_its_budget() {
    let (before, _after) = read_fixture("budget_stops_mid_document");
    let (_actual, applied) = run_round(&before, 2.0, Envelope::new(0.0, 0.0));
    assert_eq!(applied, 2);
}

/// The `no_op_density_in_envelope` fixture applies zero patches — the
/// gate itself is `Off`.
#[test]
fn no_op_fixture_applies_zero_patches() {
    let (before, _after) = read_fixture("no_op_density_in_envelope");
    let (_actual, applied) = run_round(&before, 5.0, Envelope::new(0.0, 100.0));
    assert_eq!(applied, 0);
}

/// Idempotence, for every fixture's `before.md`: running [`run_round`]
/// once with a deliberately generous budget (so nothing is left
/// unprocessed purely for lack of budget), then running it again on that
/// output with the same generous budget, changes nothing further and
/// applies zero additional patches.
///
/// This is deliberately *not* the same `(density, band)` each fixture's
/// golden-output test uses — `budget_stops_mid_document`'s golden output
/// is an intentionally partial single round (see [`FIXTURES`]), not a
/// fixed point, so checking idempotence against it under that same tight
/// budget would just be re-observing the budget stopping again, not
/// proving the rule itself is idempotent. A generous budget isolates that
/// property: does `scan` ever refind something `fix` already handled?
#[test]
fn idempotent_across_all_fixtures_given_a_generous_budget() {
    let generous_band = Envelope::new(0.0, 0.0);
    let generous_density = 1_000_000.0;

    for &(name, _density, _band) in FIXTURES {
        let (before, _after) = read_fixture(name);
        let (once, _first_applied) = run_round(&before, generous_density, generous_band);
        let (twice, second_applied) = run_round(&once, generous_density, generous_band);
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
