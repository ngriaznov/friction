//! `"not just/only X but (also) Y"` detection: flags a sentence built
//! around this coordination shape (`"This is not just fast but also
//! reliable."`), an LLM tic that overuses the pattern relative to human
//! writing.
//!
//! # Suggest tier, not Fix
//!
//! Reframing a `"not just X but Y"` sentence necessarily rewrites which
//! half of the coordination the sentence emphasizes, and a purely
//! syntactic scan has no way to know which half — if either — is safe to
//! drop or demote without changing what the sentence asserts. Every
//! finding here is therefore [`friction_core::Tier::Suggest`] and
//! [`NotJustButRule::fix`] always declines; the engine surfaces these as
//! diagnostics only.
//!
//! # Mirrored pattern
//!
//! [`NOT_JUST_BUT_RE`] is a byte-identical copy of `friction-metrics::
//! lexical::NOT_JUST_BUT_RE`'s pattern string (that constant is private to
//! its crate — see `families::symmetry`'s own module docs for why every
//! submodule here mirrors rather than imports). This module's
//! `not_just_but_finding_count_matches_the_public_rate_metric` test
//! cross-checks this rule's `scan` against `friction_metrics::
//! not_just_but_rate`'s public, rate-returning function on a shared
//! document, so the two patterns cannot silently drift apart without a
//! test failing.

use std::sync::LazyLock;

use friction_core::{Finding, MetricVector, Patch, RuleId, Tier};
use regex::Regex;

use crate::context::{GenreEnvelope, RuleContext};
use crate::rule::{Gate, Rule, RuleFamily};
use crate::strategy::StrategyRng;

/// This rule's stable identifier.
const RULE_ID: RuleId = RuleId::new("symmetry.not_just_but");

/// The [`MetricVector`] field this rule gates on.
const GATED_METRIC: &str = "not_just_but_rate";

/// Mirrors `friction_metrics::lexical::NOT_JUST_BUT_RE`'s pattern exactly
/// — see that constant's own docs for what each part of the pattern does.
static NOT_JUST_BUT_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?is)\bnot\s+(?:just|only)\b.*?\bbut\b(?:\s+also\b)?")
        .expect("not-just-but pattern is a fixed, valid regex")
});

/// Flags `"not just/only X but (also) Y"` constructions. Suggest tier
/// only — see the module docs.
#[derive(Debug, Clone, Copy, Default)]
pub struct NotJustButRule;

impl NotJustButRule {
    /// Creates the rule.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Rule for NotJustButRule {
    fn id(&self) -> RuleId {
        RULE_ID
    }

    fn family(&self) -> RuleFamily {
        RuleFamily::Symmetry
    }

    fn gate(&self, metrics: &MetricVector, envelope: &dyn GenreEnvelope) -> Gate {
        let Some(band) = envelope.band(GATED_METRIC) else {
            return Gate::Off;
        };
        if metrics.not_just_but_rate <= band.hi {
            Gate::Off
        } else {
            // Suggest tier only: see the module docs. `Detect` surfaces
            // every finding as a diagnostic without ever calling `fix`.
            Gate::Detect
        }
    }

    fn scan(&self, ctx: &RuleContext<'_>) -> Vec<Finding> {
        let document = ctx.document();
        let mut findings = Vec::new();
        for (_, sentence) in ctx.sentences() {
            let Ok(text) = document.text(&sentence.range) else {
                continue;
            };
            if let Some(m) = NOT_JUST_BUT_RE.find(text) {
                let start = sentence.range.start + m.start();
                let end = sentence.range.start + m.end();
                findings.push(Finding::new(
                    RULE_ID,
                    start..end,
                    "\"not just/only X but (also) Y\" reads as an LLM tic; consider reframing which half the sentence emphasizes",
                    Tier::Suggest,
                ));
            }
        }
        findings
    }

    fn fix(
        &self,
        _finding: &Finding,
        _ctx: &RuleContext<'_>,
        _strategy_rng: &mut StrategyRng,
    ) -> Option<Patch> {
        // Suggest tier only; see the module docs. Every finding this rule
        // reports has Tier::Suggest, so `friction-apply`'s driver never
        // calls `fix` for it in the first place — this always declining is
        // defense in depth, not a live code path.
        None
    }
}

#[cfg(test)]
mod tests {
    use friction_core::Envelope;

    use super::*;
    use crate::context::MapEnvelope;

    fn document(source: &str) -> friction_core::Document {
        let parsed = friction_parse::parse(source).expect("valid markdown parses");
        friction_nlp::segment_document(&parsed, &friction_nlp::SrxSegmenter::new())
            .expect("segmentation succeeds")
    }

    struct NoopTagger;
    impl friction_nlp::Tagger for NoopTagger {
        fn tag(&self, _text: &str, _base_offset: usize) -> Vec<friction_nlp::TaggedToken> {
            Vec::new()
        }
    }

    fn metrics_with_rate(rate: f64) -> MetricVector {
        MetricVector {
            not_just_but_rate: rate,
            ..MetricVector::default()
        }
    }

    // -----------------------------------------------------------------
    // scan()
    // -----------------------------------------------------------------

    #[test]
    fn scan_finds_not_just_but_also() {
        let source = "This solution is not just fast but also reliable.\n";
        let doc = document(source);
        let envelope = MapEnvelope::new();
        let ctx = RuleContext::new(&doc, &NoopTagger, "blog", &envelope);
        let findings = NotJustButRule::new().scan(&ctx);
        assert_eq!(findings.len(), 1);
        assert_eq!(&source[findings[0].range.clone()], "not just fast but also");
        assert_eq!(findings[0].tier, Tier::Suggest);
    }

    #[test]
    fn scan_finds_not_only_but_without_also() {
        let source = "The plan covers not only budget but timeline too.\n";
        let doc = document(source);
        let envelope = MapEnvelope::new();
        let ctx = RuleContext::new(&doc, &NoopTagger, "blog", &envelope);
        let findings = NotJustButRule::new().scan(&ctx);
        assert_eq!(findings.len(), 1);
        assert_eq!(&source[findings[0].range.clone()], "not only budget but");
    }

    #[test]
    fn scan_finds_nothing_in_plain_prose() {
        let source = "It works well. Nothing else to add.\n";
        let doc = document(source);
        let envelope = MapEnvelope::new();
        let ctx = RuleContext::new(&doc, &NoopTagger, "blog", &envelope);
        assert!(NotJustButRule::new().scan(&ctx).is_empty());
    }

    // -----------------------------------------------------------------
    // Cross-crate consistency against friction_metrics::not_just_but_rate
    // -----------------------------------------------------------------

    #[test]
    fn not_just_but_finding_count_matches_the_public_rate_metric() {
        let source = "This solution is not just fast but also reliable. It works well. \
            The plan covers not only budget but timeline too. Nothing else to add.\n";
        let doc = document(source);
        let sentence_count = doc
            .prose()
            .iter()
            .map(|unit| unit.sentences.len())
            .sum::<usize>();
        let metrics_rate = friction_metrics::not_just_but_rate(&doc);
        #[allow(clippy::cast_precision_loss)]
        let metrics_matches = metrics_rate * sentence_count as f64;

        let envelope = MapEnvelope::new();
        let ctx = RuleContext::new(&doc, &NoopTagger, "blog", &envelope);
        let rule_matches = NotJustButRule::new().scan(&ctx).len();

        #[allow(clippy::cast_precision_loss)]
        let rule_matches_f64 = rule_matches as f64;
        assert!(
            (metrics_matches - rule_matches_f64).abs() < 1e-9,
            "friction_metrics::not_just_but_rate implies {metrics_matches} matches, \
             NotJustButRule::scan found {rule_matches}"
        );
        assert_eq!(rule_matches, 2);
    }

    // -----------------------------------------------------------------
    // gate()
    // -----------------------------------------------------------------

    #[test]
    fn gate_is_off_without_a_band() {
        let rule = NotJustButRule::new();
        let envelope = MapEnvelope::new();
        assert_eq!(rule.gate(&metrics_with_rate(0.5), &envelope), Gate::Off);
    }

    #[test]
    fn gate_is_off_inside_band() {
        let rule = NotJustButRule::new();
        let envelope = MapEnvelope::new().with(GATED_METRIC, Envelope::new(0.0, 0.5));
        assert_eq!(rule.gate(&metrics_with_rate(0.2), &envelope), Gate::Off);
    }

    #[test]
    fn gate_is_detect_above_band() {
        let rule = NotJustButRule::new();
        let envelope = MapEnvelope::new().with(GATED_METRIC, Envelope::new(0.0, 0.1));
        assert_eq!(rule.gate(&metrics_with_rate(0.5), &envelope), Gate::Detect);
    }

    // -----------------------------------------------------------------
    // fix() always declines
    // -----------------------------------------------------------------

    #[test]
    fn fix_always_declines() {
        let rule = NotJustButRule::new();
        let source = "This solution is not just fast but also reliable.\n";
        let doc = document(source);
        let envelope = MapEnvelope::new();
        let ctx = RuleContext::new(&doc, &NoopTagger, "blog", &envelope);
        let finding = &rule.scan(&ctx)[0];
        let mut rng = StrategyRng::from_seed(0);
        assert!(rule.fix(finding, &ctx, &mut rng).is_none());
    }
}
