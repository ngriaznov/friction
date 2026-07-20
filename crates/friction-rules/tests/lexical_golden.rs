//! Golden before/after fixtures, idempotence checks, and a human-register
//! no-op check for [`friction_rules::FillerPhraseRule`] and
//! [`friction_rules::SubstitutionRule`].
//!
//! Every fixture pair lives under `tests/golden/lexical/<name>.before.md` /
//! `<name>.after.md`. This file drives them through a small, local
//! single-round harness ([`run_round`]) that mirrors `friction-apply`'s
//! driver semantics for exactly one rule (gate once, scan, walk findings in
//! source order, check budget *before* each `fix` call, apply the
//! resulting non-overlapping patches) — this crate does not depend on
//! `friction-apply` (that dependency runs the other way), so the harness is
//! a deliberate, documented stand-in rather than a call into the real
//! driver. It is a close copy of `tests/connective_golden.rs`'s own
//! `run_round`, generalized over which `MetricVector` field the rule under
//! test gates on.
//!
//! Every fixture's expected `after.md` content was derived by hand from the
//! rule's own documented span rules: [`FillerPhraseRule`] deletes a matched
//! phrase plus a redundant trailing comma and separating whitespace,
//! recapitalizing the next word only when the match was sentence-initial;
//! [`SubstitutionRule`] replaces a matched lemma occurrence with
//! `friction_nlp::inflect(surface, replacement_lemma)`, traced by hand
//! against that function's own documented suffix rules (silent-e drop,
//! irregular-verb/irregular-noun table lookups for the target lemma).
//! Neither rule consults its `StrategyRng` argument (each finding has
//! exactly one meaning-preserving fix), so — unlike
//! `connective_golden.rs` — no independent seeded-RNG cross-check was
//! needed to derive these fixtures.

use std::fs;
use std::path::Path;

use friction_core::{Document, Envelope, Finding, MetricVector, Patch, Tier, span};
use friction_nlp::{SrxSegmenter, Tagger};
use friction_rules::{FillerPhraseRule, Gate, MapEnvelope, Rule, RuleContext, StrategyRng};

/// A stub tagger; neither rule under test consults part-of-speech tags.
struct NoopTagger;
impl Tagger for NoopTagger {
    fn tag(&self, _text: &str, _base_offset: usize) -> Vec<friction_nlp::TaggedToken> {
        Vec::new()
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
/// [`MapEnvelope`] built from `(gated_metric, band)`. Returns the resulting
/// text and how many patches were applied.
fn run_round(
    source: &str,
    rule: &dyn Rule,
    gated_metric: &'static str,
    metrics: MetricVector,
    band: Envelope,
) -> (String, usize) {
    let parsed = friction_parse::parse(source).expect("fixture source is valid markdown");
    let document = friction_nlp::segment_document(&parsed, &SrxSegmenter::new())
        .expect("fixture source segments cleanly");
    let envelope = MapEnvelope::new().with(gated_metric, band);
    let ctx = RuleContext::new(&document, &NoopTagger, "blog", &envelope);

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

/// Reads `tests/golden/lexical/<name>.before.md` / `<name>.after.md`,
/// stripping no bytes — the fixture files' own trailing newline is part of
/// the compared content.
fn read_fixture(name: &str) -> (String, String) {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden/lexical");
    let before = fs::read_to_string(dir.join(format!("{name}.before.md")))
        .unwrap_or_else(|e| panic!("reading {name}.before.md: {e}"));
    let after = fs::read_to_string(dir.join(format!("{name}.after.md")))
        .unwrap_or_else(|e| panic!("reading {name}.after.md: {e}"));
    (before, after)
}

/// A generous "always fire, essentially unlimited budget" gating setup: a
/// zero-width envelope band (so any positive metric value is outside it)
/// paired with a very large metric value, giving `Budget::from_envelope_
/// excess` a budget far larger than any fixture could ever exhaust.
const GENEROUS_BAND: Envelope = Envelope::new(0.0, 0.0);
const GENEROUS_VALUE: f64 = 1_000_000.0;

const DISCOURSE_MARKER_DENSITY: &str = "discourse_marker_density";
const LLM_FAVORED_PHRASE_RATE: &str = "llm_favored_phrase_rate";

fn metrics_with_discourse_density(value: f64) -> MetricVector {
    MetricVector {
        discourse_marker_density: value,
        ..MetricVector::default()
    }
}

fn metrics_with_llm_phrase_rate(value: f64) -> MetricVector {
    MetricVector {
        llm_favored_phrase_rate: value,
        ..MetricVector::default()
    }
}

// =====================================================================
// FillerPhraseRule
// =====================================================================

const FILLER_FIXTURES: &[&str] = &[
    "filler_sentence_initial_and_mid_sentence",
    "filler_multiple_and_comma_cleanup",
    "filler_bookend_phrases",
];

/// Every filler-phrase fixture's `before.md`, run through a generous
/// [`run_round`], reproduces `after.md` exactly.
#[test]
fn filler_golden_fixtures_match_expected_output() {
    let rule = FillerPhraseRule::new();
    for &name in FILLER_FIXTURES {
        let (before, after) = read_fixture(name);
        let (actual, _applied) = run_round(
            &before,
            &rule,
            DISCOURSE_MARKER_DENSITY,
            metrics_with_discourse_density(GENEROUS_VALUE),
            GENEROUS_BAND,
        );
        assert_eq!(
            actual, after,
            "fixture {name:?} did not match its golden output"
        );
    }
}

/// The human-register fixture applies zero patches under a *generous*
/// budget: this is a specificity check (the rule genuinely finds nothing
/// to flag in clean human prose), not merely a gate-off check.
#[test]
fn filler_no_op_fixture_applies_zero_patches_even_with_a_generous_budget() {
    let rule = FillerPhraseRule::new();
    let (before, after) = read_fixture("filler_no_op_human_register");
    assert_eq!(
        before, after,
        "no-op fixture's before/after must be identical"
    );
    let (actual, applied) = run_round(
        &before,
        &rule,
        DISCOURSE_MARKER_DENSITY,
        metrics_with_discourse_density(GENEROUS_VALUE),
        GENEROUS_BAND,
    );
    assert_eq!(applied, 0);
    assert_eq!(actual, before);
}

/// The human-register fixture also applies zero patches through the
/// ordinary gate-off path (density inside the envelope), the other way a
/// real document would see zero patches.
#[test]
fn filler_no_op_fixture_gates_off_inside_the_envelope() {
    let rule = FillerPhraseRule::new();
    let (before, _after) = read_fixture("filler_no_op_human_register");
    let (actual, applied) = run_round(
        &before,
        &rule,
        DISCOURSE_MARKER_DENSITY,
        metrics_with_discourse_density(5.0),
        Envelope::new(0.0, 100.0),
    );
    assert_eq!(applied, 0);
    assert_eq!(actual, before);
}

/// Idempotence, for every filler fixture's `before.md`: running
/// [`run_round`] once with a generous budget, then again on that output
/// with the same generous budget, changes nothing further and applies zero
/// additional patches.
#[test]
fn filler_idempotent_across_all_fixtures_given_a_generous_budget() {
    let rule = FillerPhraseRule::new();
    for &name in FILLER_FIXTURES {
        let (before, _after) = read_fixture(name);
        let (once, _first_applied) = run_round(
            &before,
            &rule,
            DISCOURSE_MARKER_DENSITY,
            metrics_with_discourse_density(GENEROUS_VALUE),
            GENEROUS_BAND,
        );
        let (twice, second_applied) = run_round(
            &once,
            &rule,
            DISCOURSE_MARKER_DENSITY,
            metrics_with_discourse_density(GENEROUS_VALUE),
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
// SubstitutionRule
// =====================================================================

const SUBSTITUTION_FIXTURES: &[&str] = &[
    "substitution_verb_inflections",
    "substitution_adjectives_and_regular_plural",
    "substitution_irregular_past_targets",
];

/// Every substitution fixture's `before.md`, run through a generous
/// [`run_round`], reproduces `after.md` exactly.
#[test]
fn substitution_golden_fixtures_match_expected_output() {
    let rule = friction_rules::SubstitutionRule::new();
    for &name in SUBSTITUTION_FIXTURES {
        let (before, after) = read_fixture(name);
        let (actual, _applied) = run_round(
            &before,
            &rule,
            LLM_FAVORED_PHRASE_RATE,
            metrics_with_llm_phrase_rate(GENEROUS_VALUE),
            GENEROUS_BAND,
        );
        assert_eq!(
            actual, after,
            "fixture {name:?} did not match its golden output"
        );
    }
}

/// The human-register fixture applies zero patches under a *generous*
/// budget: this is a specificity check, not merely a gate-off check.
#[test]
fn substitution_no_op_fixture_applies_zero_patches_even_with_a_generous_budget() {
    let rule = friction_rules::SubstitutionRule::new();
    let (before, after) = read_fixture("substitution_no_op_human_register");
    assert_eq!(
        before, after,
        "no-op fixture's before/after must be identical"
    );
    let (actual, applied) = run_round(
        &before,
        &rule,
        LLM_FAVORED_PHRASE_RATE,
        metrics_with_llm_phrase_rate(GENEROUS_VALUE),
        GENEROUS_BAND,
    );
    assert_eq!(applied, 0);
    assert_eq!(actual, before);
}

/// The human-register fixture also applies zero patches through the
/// ordinary gate-off path (rate inside the envelope).
#[test]
fn substitution_no_op_fixture_gates_off_inside_the_envelope() {
    let rule = friction_rules::SubstitutionRule::new();
    let (before, _after) = read_fixture("substitution_no_op_human_register");
    let (actual, applied) = run_round(
        &before,
        &rule,
        LLM_FAVORED_PHRASE_RATE,
        metrics_with_llm_phrase_rate(5.0),
        Envelope::new(0.0, 100.0),
    );
    assert_eq!(applied, 0);
    assert_eq!(actual, before);
}

/// Idempotence, for every substitution fixture's `before.md`.
#[test]
fn substitution_idempotent_across_all_fixtures_given_a_generous_budget() {
    let rule = friction_rules::SubstitutionRule::new();
    for &name in SUBSTITUTION_FIXTURES {
        let (before, _after) = read_fixture(name);
        let (once, _first_applied) = run_round(
            &before,
            &rule,
            LLM_FAVORED_PHRASE_RATE,
            metrics_with_llm_phrase_rate(GENEROUS_VALUE),
            GENEROUS_BAND,
        );
        let (twice, second_applied) = run_round(
            &once,
            &rule,
            LLM_FAVORED_PHRASE_RATE,
            metrics_with_llm_phrase_rate(GENEROUS_VALUE),
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
