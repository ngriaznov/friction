//! Golden before/after fixtures, idempotence checks, and a human-register
//! no-op check for [`friction_rules::ParticipialCloserRule`] and
//! [`friction_rules::RitualConclusionRule`] — the two Fix-tier (or
//! mixed-tier) rules in the symmetry family. [`friction_rules::
//! TriadReductionRule`] and [`friction_rules::NotJustButRule`] are Suggest
//! tier only (never produce a patch — see each rule's own module docs), so
//! they have no before/after golden fixtures of their own; their detection
//! logic is instead covered by the hand-built and cross-crate consistency
//! tests in `friction-rules::families::symmetry::{triad_reduction,
//! not_just_but}`'s own `#[cfg(test)]` modules.
//!
//! Every fixture pair lives under `tests/golden/symmetry/<name>.before.md`
//! / `<name>.after.md`. This file drives them through a small, local
//! single-round harness ([`run_round`]) that mirrors `friction-apply`'s
//! driver semantics for exactly one rule (gate once, scan, walk findings in
//! source order, check budget *before* each `fix` call, apply the
//! resulting non-overlapping patches) — this crate does not depend on
//! `friction-apply` (that dependency runs the other way), so the harness is
//! a deliberate, documented stand-in rather than a call into the real
//! driver. It is a close copy of `tests/connective_golden.rs`'s own
//! `run_round`, generalized over which rule and gated metric is under test,
//! the same way `tests/lexical_golden.rs` generalizes it — except this
//! harness uses the real `NlpruleTagger`, since both rules under test read
//! part-of-speech tags (`ConnectiveSurgery`/the lexical rules never
//! needed one).
//!
//! # How each fixture's expected output was derived
//!
//! [`ParticipialCloserRule`] only ever applies its content-preserving
//! **promote** strategy automatically (see that rule's own module docs'
//! "Mixed tier, per finding" section): whether a fixture's closer has a
//! clear object continuation to promote — and is therefore Fix tier at
//! all — was read directly off the fixture text itself (a determiner,
//! pronoun, or noun immediately after the participle), the same
//! `has_object_like_continuation` check the rule's own unit tests exercise
//! directly. A closer without one is Suggest tier and never produces a
//! patch, however content-bearing or filler it looks — this rule cannot
//! tell the two apart safely, so it never guesses.
//!
//! [`RitualConclusionRule`] fixture text was chosen so every noun the
//! flagged final paragraph mentions is a word that also appears, unchanged,
//! in an earlier sentence of the same document (the Fix-tier fixtures), or
//! so it mentions at least one noun that does not (the Suggest-tier
//! fixture) — read directly off the fixture text itself, not derived from
//! running the tagger blind.

use std::fs;
use std::path::Path;

use friction_core::{Document, Envelope, Finding, MetricVector, Patch, Tier, span};
use friction_nlp::{NlpruleTagger, SrxSegmenter};
use friction_rules::{
    Gate, MapEnvelope, ParticipialCloserRule, RitualConclusionRule, Rule, RuleContext, StrategyRng,
};

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

/// Applies non-overlapping `patches` to `source` in one right-to-left pass
/// — a local copy of `friction-apply`'s own `apply_patches` mechanics, safe
/// here because every patch [`run_round`] collects comes from one rule's
/// own left-to-right, non-overlapping `scan`.
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
/// scoped to whichever `rule` is passed in, gated against `metrics` and a
/// [`MapEnvelope`] built from `(gated_metric, band)`, using the real
/// [`NlpruleTagger`]. Returns the resulting text and how many patches were
/// applied.
fn run_round(
    source: &str,
    rule: &dyn Rule,
    tagger: &NlpruleTagger,
    gated_metric: &'static str,
    metrics: MetricVector,
    band: Envelope,
) -> (String, usize) {
    let parsed = friction_parse::parse(source).expect("fixture source is valid markdown");
    let document = friction_nlp::segment_document(&parsed, &SrxSegmenter::new())
        .expect("fixture source segments cleanly");
    let envelope = MapEnvelope::new().with(gated_metric, band);
    let ctx = RuleContext::new(&document, tagger, "blog", &envelope);

    let Gate::Fix { mut budget } = rule.gate(&metrics, &envelope) else {
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

/// Reads `tests/golden/symmetry/<name>.before.md` / `<name>.after.md`,
/// stripping no bytes — the fixture files' own trailing newline is part of
/// the compared content.
fn read_fixture(name: &str) -> (String, String) {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden/symmetry");
    let before = fs::read_to_string(dir.join(format!("{name}.before.md")))
        .unwrap_or_else(|e| panic!("reading {name}.before.md: {e}"));
    let after = fs::read_to_string(dir.join(format!("{name}.after.md")))
        .unwrap_or_else(|e| panic!("reading {name}.after.md: {e}"));
    (before, after)
}

/// A generous "always fire, essentially unlimited budget" gating setup: a
/// zero-width envelope band paired with a large metric value.
const GENEROUS_BAND: Envelope = Envelope::new(0.0, 0.0);
const GENEROUS_VALUE: f64 = 1_000_000.0;

fn metrics_with_participial_rate(value: f64) -> MetricVector {
    MetricVector {
        participial_closer_rate: value,
        ..MetricVector::default()
    }
}

fn metrics_with_ritual_rate(value: f64) -> MetricVector {
    MetricVector {
        ritual_marker_rate: value,
        ..MetricVector::default()
    }
}

// =====================================================================
// ParticipialCloserRule
// =====================================================================

const PARTICIPIAL_METRIC: &str = "participial_closer_rate";

/// Every fixture whose golden output was produced under a generous
/// (essentially unbudgeted) round, and does end up applying a patch (i.e.
/// its closer has a clear object continuation to promote).
const PARTICIPIAL_GENEROUS_FIXTURES: &[&str] =
    &["participial_promote", "participial_promote_content_bearing"];

#[test]
fn participial_golden_fixtures_match_expected_output() {
    let tagger = NlpruleTagger::new().expect("embedded model loads");
    let rule = ParticipialCloserRule::new();
    for &name in PARTICIPIAL_GENEROUS_FIXTURES {
        let (before, after) = read_fixture(name);
        let (actual, _applied) = run_round(
            &before,
            &rule,
            &tagger,
            PARTICIPIAL_METRIC,
            metrics_with_participial_rate(GENEROUS_VALUE),
            GENEROUS_BAND,
        );
        assert_eq!(
            actual, after,
            "fixture {name:?} did not match its golden output"
        );
    }
}

/// The budget-stops-mid-document fixture: two closer sentences in one
/// document, a band whose ceiling only licenses fixing one of them (a
/// hand-computed target count of 1 — see `families::symmetry::
/// participial_closer::ParticipialCloserRule::fix`'s own "Exact,
/// per-round budgeting" docs), so only the first (leftmost) finding is
/// fixed and the second is left untouched by this round.
#[test]
fn participial_budget_stops_mid_document_fixes_only_the_first_finding() {
    let tagger = NlpruleTagger::new().expect("embedded model loads");
    let rule = ParticipialCloserRule::new();
    let (before, after) = read_fixture("participial_budget_stops_mid_document");
    let band = Envelope::new(0.0, 0.5);
    // Hand-computed: 2 sentences, both closers -> current = 1.0; surplus =
    // 1.0 - 0.5 = 0.5; target_count = floor(0.5 * 2) = 1.
    let (actual, applied) = run_round(
        &before,
        &rule,
        &tagger,
        PARTICIPIAL_METRIC,
        metrics_with_participial_rate(1.0),
        band,
    );
    assert_eq!(actual, after);
    assert_eq!(applied, 1);
}

/// The no-object-continuation fixture: the closer clause ("removing manual
/// cleanup steps") has nothing determiner/pronoun/noun-shaped right after
/// its participle, so this rule cannot verify deleting it is safe (see the
/// rule's own module docs) — the finding is Suggest tier and, even under a
/// generous budget, `fix` never proposes a patch for it. Regression
/// fixture for the finding that this rule used to fall back to an
/// unconditional, silent **delete** here.
#[test]
fn participial_suggest_fixture_applies_zero_patches_even_with_a_generous_budget() {
    let tagger = NlpruleTagger::new().expect("embedded model loads");
    let rule = ParticipialCloserRule::new();
    let (before, after) = read_fixture("participial_suggest_no_object");
    assert_eq!(
        before, after,
        "Suggest-tier fixture's before/after must be identical"
    );
    let (actual, applied) = run_round(
        &before,
        &rule,
        &tagger,
        PARTICIPIAL_METRIC,
        metrics_with_participial_rate(GENEROUS_VALUE),
        GENEROUS_BAND,
    );
    assert_eq!(applied, 0);
    assert_eq!(actual, before);

    // Extra assertion, direct against `scan`: the finding really is
    // Suggest tier, not merely "budget happened to be zero" — the same
    // distinction `ritual_suggest_fixture_applies_zero_patches_even_with_a_
    // generous_budget` makes for `RitualConclusionRule`.
    let parsed = friction_parse::parse(before.as_str()).expect("fixture source is valid markdown");
    let document = friction_nlp::segment_document(&parsed, &SrxSegmenter::new())
        .expect("fixture source segments cleanly");
    let envelope = MapEnvelope::new().with(PARTICIPIAL_METRIC, GENEROUS_BAND);
    let ctx = RuleContext::new(&document, &tagger, "blog", &envelope);
    let findings = rule.scan(&ctx);
    assert_eq!(findings.len(), 1);
    assert_eq!(findings[0].tier, Tier::Suggest);
}

/// The human-register fixture applies zero patches under a *generous*
/// budget: this is a specificity check (the rule genuinely finds nothing
/// to flag), not merely a gate-off check.
#[test]
fn participial_no_op_fixture_applies_zero_patches_even_with_a_generous_budget() {
    let tagger = NlpruleTagger::new().expect("embedded model loads");
    let rule = ParticipialCloserRule::new();
    let (before, after) = read_fixture("participial_no_op_human_register");
    assert_eq!(
        before, after,
        "no-op fixture's before/after must be identical"
    );
    let (actual, applied) = run_round(
        &before,
        &rule,
        &tagger,
        PARTICIPIAL_METRIC,
        metrics_with_participial_rate(GENEROUS_VALUE),
        GENEROUS_BAND,
    );
    assert_eq!(applied, 0);
    assert_eq!(actual, before);
}

/// The human-register fixture also applies zero patches through the
/// ordinary gate-off path (rate inside the envelope).
#[test]
fn participial_no_op_fixture_gates_off_inside_the_envelope() {
    let tagger = NlpruleTagger::new().expect("embedded model loads");
    let rule = ParticipialCloserRule::new();
    let (before, _after) = read_fixture("participial_no_op_human_register");
    let (actual, applied) = run_round(
        &before,
        &rule,
        &tagger,
        PARTICIPIAL_METRIC,
        metrics_with_participial_rate(0.05),
        Envelope::new(0.0, 1.0),
    );
    assert_eq!(applied, 0);
    assert_eq!(actual, before);
}

/// Idempotence, for every generous-budget fixture's `before.md`: running
/// [`run_round`] once with a generous budget, then again on that output
/// with the same generous budget, changes nothing further and applies zero
/// additional patches.
#[test]
fn participial_idempotent_across_all_fixtures_given_a_generous_budget() {
    let tagger = NlpruleTagger::new().expect("embedded model loads");
    let rule = ParticipialCloserRule::new();
    for &name in PARTICIPIAL_GENEROUS_FIXTURES {
        let (before, _after) = read_fixture(name);
        let (once, _first_applied) = run_round(
            &before,
            &rule,
            &tagger,
            PARTICIPIAL_METRIC,
            metrics_with_participial_rate(GENEROUS_VALUE),
            GENEROUS_BAND,
        );
        let (twice, second_applied) = run_round(
            &once,
            &rule,
            &tagger,
            PARTICIPIAL_METRIC,
            metrics_with_participial_rate(GENEROUS_VALUE),
            GENEROUS_BAND,
        );
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

// =====================================================================
// RitualConclusionRule
// =====================================================================

const RITUAL_METRIC: &str = "ritual_marker_rate";

const RITUAL_FIX_FIXTURES: &[&str] = &[
    "ritual_fix_overall",
    "ritual_fix_to_summarize",
    "ritual_fix_ultimately",
];

/// Every Fix-tier fixture's `before.md`, run through a generous
/// [`run_round`], reproduces `after.md` exactly: the final paragraph,
/// including its own trailing blank line, is deleted outright.
#[test]
fn ritual_fix_golden_fixtures_match_expected_output() {
    let tagger = NlpruleTagger::new().expect("embedded model loads");
    let rule = RitualConclusionRule::new();
    for &name in RITUAL_FIX_FIXTURES {
        let (before, after) = read_fixture(name);
        let (actual, applied) = run_round(
            &before,
            &rule,
            &tagger,
            RITUAL_METRIC,
            metrics_with_ritual_rate(GENEROUS_VALUE),
            GENEROUS_BAND,
        );
        assert_eq!(
            actual, after,
            "fixture {name:?} did not match its golden output"
        );
        assert_eq!(applied, 1, "fixture {name:?} expected exactly one patch");
    }
}

/// The Suggest-tier fixture: the final paragraph opens with a ritual
/// marker, but mentions a noun ("roadmap") no earlier sentence does, so
/// the finding is Suggest tier and — even under a generous budget — no
/// patch is ever proposed for it.
#[test]
fn ritual_suggest_fixture_applies_zero_patches_even_with_a_generous_budget() {
    let tagger = NlpruleTagger::new().expect("embedded model loads");
    let rule = RitualConclusionRule::new();
    let (before, after) = read_fixture("ritual_suggest_new_noun");
    assert_eq!(
        before, after,
        "Suggest-tier fixture's before/after must be identical"
    );
    let (actual, applied) = run_round(
        &before,
        &rule,
        &tagger,
        RITUAL_METRIC,
        metrics_with_ritual_rate(GENEROUS_VALUE),
        GENEROUS_BAND,
    );
    assert_eq!(applied, 0);
    assert_eq!(actual, before);
}

/// The no-marker fixture: the final paragraph has no ritual marker at all,
/// so `scan` reports nothing — a specificity check distinct from the
/// Suggest-tier fixture above (which does report a finding, just not one
/// eligible for a patch).
#[test]
fn ritual_no_op_fixture_applies_zero_patches_even_with_a_generous_budget() {
    let tagger = NlpruleTagger::new().expect("embedded model loads");
    let rule = RitualConclusionRule::new();
    let (before, after) = read_fixture("ritual_no_op_no_marker");
    assert_eq!(
        before, after,
        "no-op fixture's before/after must be identical"
    );
    let (actual, applied) = run_round(
        &before,
        &rule,
        &tagger,
        RITUAL_METRIC,
        metrics_with_ritual_rate(GENEROUS_VALUE),
        GENEROUS_BAND,
    );
    assert_eq!(applied, 0);
    assert_eq!(actual, before);
}

/// The no-marker fixture also applies zero patches through the ordinary
/// gate-off path (rate inside the envelope).
#[test]
fn ritual_no_op_fixture_gates_off_inside_the_envelope() {
    let tagger = NlpruleTagger::new().expect("embedded model loads");
    let rule = RitualConclusionRule::new();
    let (before, _after) = read_fixture("ritual_no_op_no_marker");
    let (actual, applied) = run_round(
        &before,
        &rule,
        &tagger,
        RITUAL_METRIC,
        metrics_with_ritual_rate(0.05),
        Envelope::new(0.0, 1.0),
    );
    assert_eq!(applied, 0);
    assert_eq!(actual, before);
}

/// Idempotence, for every Fix-tier fixture's `before.md`.
#[test]
fn ritual_fix_idempotent_across_all_fixtures_given_a_generous_budget() {
    let tagger = NlpruleTagger::new().expect("embedded model loads");
    let rule = RitualConclusionRule::new();
    for &name in RITUAL_FIX_FIXTURES {
        let (before, _after) = read_fixture(name);
        let (once, _first_applied) = run_round(
            &before,
            &rule,
            &tagger,
            RITUAL_METRIC,
            metrics_with_ritual_rate(GENEROUS_VALUE),
            GENEROUS_BAND,
        );
        let (twice, second_applied) = run_round(
            &once,
            &rule,
            &tagger,
            RITUAL_METRIC,
            metrics_with_ritual_rate(GENEROUS_VALUE),
            GENEROUS_BAND,
        );
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
