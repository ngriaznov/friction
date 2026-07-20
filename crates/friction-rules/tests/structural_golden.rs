//! Golden before/after fixtures and idempotence checks for the structural
//! rule family ([`UnbulletRule`], [`BoldLabelStripRule`],
//! [`HeaderMergeRule`]).
//!
//! Every fixture pair lives under
//! `tests/golden/structural/<name>.before.md` / `<name>.after.md`. This
//! file drives them through a small, local single-round harness that
//! mirrors `friction-apply`'s driver semantics for exactly one rule (gate
//! once, scan, walk findings in source order, check budget *before* each
//! `fix` call, apply the resulting non-overlapping patches) — this crate
//! does not depend on `friction-apply` (that dependency runs the other
//! way), so the harness is a deliberate, documented stand-in, the same
//! shape `tests/connective_golden.rs` already established for this crate.
//!
//! [`HeaderMergeRule`] never proposes a patch (Suggest tier only — see
//! that rule's own module docs), so its fixtures' `before.md` and
//! `after.md` are byte-identical; its harness only checks `scan`'s
//! findings, not any applied text.
//!
//! [`UnbulletRule`]'s own fixtures are driven through the real
//! [`friction_nlp::NlpruleTagger`], not a stub — this rule's "machine-
//! flavored stem parallelism" qualifying check (see its own module docs)
//! is exactly a real part-of-speech-tag comparison, so a fixture whose
//! items only *look* parallel to a human eye can still fail to qualify
//! under the tagger this rule actually ships with (a stub tagger that tags
//! nothing would hide that gap entirely, since every item would then land
//! in the same "no detectable stem" bucket regardless of its real
//! wording). [`BoldLabelStripRule`] and [`HeaderMergeRule`] need no
//! part-of-speech information at all (checked directly: both rules' own
//! `#[cfg(test)]` modules use a stub tagger too), so their fixtures keep
//! using [`NoopTagger`].

use std::fs;
use std::path::Path;

use friction_core::{Envelope, Finding, MetricVector, Patch, Tier, span};
use friction_nlp::{NlpruleTagger, SrxSegmenter, Tagger};
use friction_rules::{
    BoldLabelStripRule, Gate, HeaderMergeRule, MapEnvelope, Rule, RuleContext, StrategyRng,
    UnbulletRule,
};

/// A stub tagger, used only for the two structural rules
/// ([`BoldLabelStripRule`], [`HeaderMergeRule`]) that need no
/// part-of-speech information at all — see the module docs.
struct NoopTagger;
impl Tagger for NoopTagger {
    fn tag(&self, _text: &str, _base_offset: usize) -> Vec<friction_nlp::TaggedToken> {
        Vec::new()
    }
}

/// The real, embedded tagger — [`UnbulletRule`]'s own stem-parallelism
/// check needs genuine part-of-speech tags (see the module docs).
fn real_tagger() -> NlpruleTagger {
    NlpruleTagger::new().expect("embedded tagger model loads")
}

fn read_fixture(name: &str) -> (String, String) {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden/structural");
    let before = fs::read_to_string(dir.join(format!("{name}.before.md")))
        .unwrap_or_else(|e| panic!("reading {name}.before.md: {e}"));
    let after = fs::read_to_string(dir.join(format!("{name}.after.md")))
        .unwrap_or_else(|e| panic!("reading {name}.after.md: {e}"));
    (before, after)
}

/// A local copy of `friction-apply`'s own `apply_patches` mechanics (see
/// that crate for the full doc) — safe here because every patch these
/// harnesses collect belongs to a distinct, disjoint span.
fn apply_patches(source: &str, patches: &[Patch]) -> String {
    let mut ordered: Vec<&Patch> = patches.iter().collect();
    ordered.sort_by_key(|patch| std::cmp::Reverse(patch.range.start));
    let mut result = source.to_string();
    for patch in ordered {
        result.replace_range(patch.range.clone(), patch.replacement.as_str());
    }
    result
}

/// A local copy of `friction-apply`'s own `sentence_bytes_for` (see that
/// crate for the full doc): the source bytes of the sentence containing
/// `finding`, for seeding a [`StrategyRng`] — falls back to `finding`'s own
/// range for a structural finding that spans a whole block, matching
/// `sentence_bytes_for`'s own documented fallback.
fn sentence_bytes_for<'a>(document: &'a friction_core::Document, finding: &Finding) -> &'a [u8] {
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

/// One local, single-round pass of gate -> scan -> (budgeted) fix -> apply
/// for `rule`, generic over which [`friction_core::MetricVector`] field
/// carries `density` (every structural rule here gates on exactly one
/// field, so the caller supplies a closure that builds the right vector)
/// and over which `tagger` drives `scan`/`fix` — see the module docs for
/// why [`UnbulletRule`]'s own callers pass the real tagger while the other
/// two rules' callers keep using [`NoopTagger`].
fn run_round(
    source: &str,
    rule: &dyn Rule,
    metrics: MetricVector,
    tagger: &dyn Tagger,
) -> (String, usize) {
    let parsed = friction_parse::parse(source).expect("fixture source is valid markdown");
    let document = friction_nlp::segment_document(&parsed, &SrxSegmenter::new())
        .expect("fixture source segments cleanly");
    let envelope_band = Envelope::new(0.0, 0.0);
    let envelope = MapEnvelope::new()
        .with("list_item_density", envelope_band)
        .with("bold_span_density", envelope_band)
        .with("heading_density", envelope_band);
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

/// A generous, always-out-of-band metric vector for `field`, high enough
/// that [`Budget::from_envelope_excess`](friction_rules::Budget) never
/// under-budgets any fixture in this file.
fn generous_metrics(set: impl FnOnce(&mut MetricVector)) -> MetricVector {
    let mut metrics = MetricVector::default();
    set(&mut metrics);
    metrics
}

fn unbullet_metrics() -> MetricVector {
    generous_metrics(|m| m.list_item_density = 1_000_000.0)
}

fn bold_label_metrics() -> MetricVector {
    generous_metrics(|m| m.bold_span_density = 1_000_000.0)
}

fn header_merge_metrics() -> MetricVector {
    generous_metrics(|m| m.heading_density = 1_000_000.0)
}

// ---------------------------------------------------------------------
// UnbulletRule
// ---------------------------------------------------------------------

const UNBULLET_FIXTURES: &[&str] = &[
    "unbullet_colon_lead_two_items",
    "unbullet_standalone_three_items",
    "unbullet_multiple_lists_in_one_document",
];

#[test]
fn unbullet_golden_fixtures_match_expected_output() {
    let rule = UnbulletRule::new();
    let tagger = real_tagger();
    for &name in UNBULLET_FIXTURES {
        let (before, after) = read_fixture(name);
        let (actual, applied) = run_round(&before, &rule, unbullet_metrics(), &tagger);
        assert!(
            applied > 0,
            "fixture {name:?} expected at least one patch applied"
        );
        assert_eq!(
            actual, after,
            "fixture {name:?} did not match its golden output"
        );
    }
}

#[test]
fn unbullet_no_op_nested_list_untouched_matches_and_applies_nothing() {
    let rule = UnbulletRule::new();
    let tagger = real_tagger();
    let (before, after) = read_fixture("unbullet_no_op_nested_list_untouched");
    assert_eq!(
        before, after,
        "no-op fixture's before/after must be byte-identical"
    );
    let (actual, applied) = run_round(&before, &rule, unbullet_metrics(), &tagger);
    assert_eq!(applied, 0);
    assert_eq!(actual, before);
}

#[test]
fn unbullet_gate_off_inside_envelope_applies_nothing() {
    let rule = UnbulletRule::new();
    let in_band = MetricVector {
        list_item_density: 5.0,
        ..MetricVector::default()
    };
    let envelope = MapEnvelope::new().with("list_item_density", Envelope::new(0.0, 100.0));
    assert_eq!(rule.gate(&in_band, &envelope), Gate::Off);
}

#[test]
fn unbullet_idempotent_across_all_fixtures() {
    let rule = UnbulletRule::new();
    let tagger = real_tagger();
    let mut names: Vec<&str> = UNBULLET_FIXTURES.to_vec();
    names.push("unbullet_no_op_nested_list_untouched");
    for name in names {
        let (before, _after) = read_fixture(name);
        let (once, _) = run_round(&before, &rule, unbullet_metrics(), &tagger);
        let (twice, second_applied) = run_round(&once, &rule, unbullet_metrics(), &tagger);
        assert_eq!(
            once, twice,
            "fixture {name:?}: fixing its own output changed it"
        );
        assert_eq!(
            second_applied, 0,
            "fixture {name:?}: re-fixing applied more patches"
        );
    }
}

// ---------------------------------------------------------------------
// BoldLabelStripRule
// ---------------------------------------------------------------------

const BOLD_LABEL_FIXTURES: &[&str] = &[
    "bold_label_flat_list",
    "bold_label_nested_items",
    "bold_label_mixed_with_non_lead_bold",
];

#[test]
fn bold_label_golden_fixtures_match_expected_output() {
    let rule = BoldLabelStripRule::new();
    for &name in BOLD_LABEL_FIXTURES {
        let (before, after) = read_fixture(name);
        let (actual, applied) = run_round(&before, &rule, bold_label_metrics(), &NoopTagger);
        assert!(
            applied > 0,
            "fixture {name:?} expected at least one patch applied"
        );
        assert_eq!(
            actual, after,
            "fixture {name:?} did not match its golden output"
        );
    }
}

/// This fixture is a *gate-driven* no-op, not a pattern mismatch: its
/// bullets genuinely match the bold-label shape (confirmed below by
/// fixing them anyway under a generous, out-of-band budget), but a real
/// document whose `bold_span_density` already sits inside the genre's
/// band gates `Off` outright before `scan` ever runs.
#[test]
fn bold_label_no_op_in_envelope_is_a_gate_off_no_op_not_a_pattern_mismatch() {
    let rule = BoldLabelStripRule::new();
    let (before, after) = read_fixture("bold_label_no_op_in_envelope");
    assert_eq!(
        before, after,
        "no-op fixture's before/after must be byte-identical"
    );

    let in_band_density = MetricVector {
        bold_span_density: 5.0,
        ..MetricVector::default()
    };
    let in_band_envelope = MapEnvelope::new().with("bold_span_density", Envelope::new(0.0, 100.0));
    assert_eq!(rule.gate(&in_band_density, &in_band_envelope), Gate::Off);

    let (actual, applied) = run_round(&before, &rule, bold_label_metrics(), &NoopTagger);
    assert!(applied > 0);
    assert_ne!(actual, before);
}

#[test]
fn bold_label_idempotent_across_all_fixtures() {
    let rule = BoldLabelStripRule::new();
    let mut names: Vec<&str> = BOLD_LABEL_FIXTURES.to_vec();
    names.push("bold_label_no_op_in_envelope");
    for name in names {
        let (before, _after) = read_fixture(name);
        let (once, _) = run_round(&before, &rule, bold_label_metrics(), &NoopTagger);
        let (twice, second_applied) = run_round(&once, &rule, bold_label_metrics(), &NoopTagger);
        assert_eq!(
            once, twice,
            "fixture {name:?}: fixing its own output changed it"
        );
        assert_eq!(
            second_applied, 0,
            "fixture {name:?}: re-fixing applied more patches"
        );
    }
}

// ---------------------------------------------------------------------
// HeaderMergeRule (Suggest tier: findings only, never a patch)
// ---------------------------------------------------------------------

fn scan_only(source: &str, rule: &dyn Rule, metrics: MetricVector) -> Vec<Finding> {
    let parsed = friction_parse::parse(source).expect("fixture source is valid markdown");
    let document = friction_nlp::segment_document(&parsed, &SrxSegmenter::new())
        .expect("fixture source segments cleanly");
    let envelope = MapEnvelope::new().with("heading_density", Envelope::new(0.0, 0.0));
    let ctx = RuleContext::new(&document, &NoopTagger, "blog", &envelope);
    match rule.gate(&metrics, &envelope) {
        Gate::Detect => rule.scan(&ctx),
        Gate::Off => Vec::new(),
        Gate::Fix { .. } => panic!("HeaderMergeRule must never gate Fix"),
    }
}

#[test]
fn header_merge_repeated_sections_produce_one_suggest_finding_per_section() {
    let rule = HeaderMergeRule::new();
    let (before, after) = read_fixture("header_merge_repeated_short_sections");
    assert_eq!(
        before, after,
        "Suggest-tier fixture's before/after must be byte-identical"
    );
    let findings = scan_only(&before, &rule, header_merge_metrics());
    assert_eq!(findings.len(), 3);
    for finding in &findings {
        assert_eq!(finding.tier, Tier::Suggest);
    }
}

#[test]
fn header_merge_no_op_single_section_produces_no_findings() {
    let rule = HeaderMergeRule::new();
    let (before, after) = read_fixture("header_merge_no_op_single_section");
    assert_eq!(before, after);
    let findings = scan_only(&before, &rule, header_merge_metrics());
    assert!(findings.is_empty());
}

#[test]
fn header_merge_never_proposes_a_patch() {
    let rule = HeaderMergeRule::new();
    let (before, _after) = read_fixture("header_merge_repeated_short_sections");
    let parsed = friction_parse::parse(before.as_str()).expect("valid markdown");
    let document =
        friction_nlp::segment_document(&parsed, &SrxSegmenter::new()).expect("segments cleanly");
    let envelope = MapEnvelope::new().with("heading_density", Envelope::new(0.0, 0.0));
    let ctx = RuleContext::new(&document, &NoopTagger, "blog", &envelope);
    let findings = rule.scan(&ctx);
    assert!(!findings.is_empty());
    let mut rng = StrategyRng::from_seed(0);
    for finding in &findings {
        assert!(rule.fix(finding, &ctx, &mut rng).is_none());
    }
}
