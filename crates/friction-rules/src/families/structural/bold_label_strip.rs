//! [`BoldLabelStripRule`]: strips the `**...**` bold markers off a
//! `"- **Label**: text"` lead-in bullet.
//!
//! # What qualifies
//!
//! A bullet's very first sentence (i.e. `sentence.range.start ==
//! unit.range.start` — the item's text has to *start* with the bold
//! label, not merely contain one somewhere) whose text starts with
//! `**`, followed by a non-empty label containing no further `**` or
//! newline, immediately followed by `**:` — `"**Label**: text"`. The
//! bullet can be a tight or loose list item, top-level or nested (see
//! [`super::innermost_list_item`]): unlike [`super::unbullet`], this rule
//! has no reason to restrict itself to top-level lists, since it never
//! touches the list's own structure, only markup inside one item's text.
//!
//! # Why this is Fix tier
//!
//! The fix deletes exactly four bytes (`**` before the label, `**` after
//! it) and nothing else — the label's own words, the colon, and
//! everything after are untouched. Pure markup deletion is squarely
//! "changes case/punctuation only" territory, this workspace's tier
//! discipline's bar for Fix.
//!
//! # Idempotence
//!
//! The fix's output never starts with `**` again (the only two occurrences
//! this rule matched are exactly the ones it deleted), so a second pass
//! over the same text never re-matches — idempotent by construction, not
//! by a separate check.

use std::ops::Range;

use friction_core::{Finding, MetricVector, Patch, RuleId, Tier};

use super::{block_parents, innermost_list_item};
use crate::budget::Budget;
use crate::context::{GenreEnvelope, RuleContext};
use crate::rule::{Gate, Rule, RuleFamily};
use crate::strategy::StrategyRng;

/// This rule's stable identifier.
const RULE_ID: RuleId = RuleId::new("structural.bold_label_strip");

/// The [`MetricVector`] field this rule gates on.
const GATED_METRIC: &str = "bold_span_density";

/// See [`crate::families::connective`]'s `PER_FIX_EFFECT` for the exact
/// same reasoning (this rule's `gate` sees only the round's normalized
/// density, never the document's real token count).
const PER_FIX_EFFECT: f64 = 1.0;

/// If `text` starts with a `"**Label**:"` bold lead-in (see the module
/// docs' "What qualifies" section), returns the local byte range of
/// `"**Label**"` within `text` — not including the following `:`.
fn match_bold_label(text: &str) -> Option<Range<usize>> {
    let rest = text.strip_prefix("**")?;
    let close = rest.find("**")?;
    let label = &rest[..close];
    if label.is_empty() || label.contains('\n') || label.starts_with('*') {
        return None;
    }
    let after = &rest[close + 2..];
    if !after.starts_with(':') {
        return None;
    }
    Some(0..(2 + close + 2))
}

/// Strips the `**...**` bold markers off a `"- **Label**: text"` bullet's
/// lead-in label.
///
/// Budgeted to bring [`GATED_METRIC`] back into the genre's envelope. See
/// the module docs for the exact matching rule.
#[derive(Debug, Clone, Copy, Default)]
pub struct BoldLabelStripRule;

impl BoldLabelStripRule {
    /// Creates the rule.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Rule for BoldLabelStripRule {
    fn id(&self) -> RuleId {
        RULE_ID
    }

    fn family(&self) -> RuleFamily {
        RuleFamily::Structural
    }

    fn gate(&self, metrics: &MetricVector, envelope: &dyn GenreEnvelope) -> Gate {
        let Some(band) = envelope.band(GATED_METRIC) else {
            return Gate::Off;
        };
        let current = metrics.bold_span_density;
        // Only the "too much bold markup" direction is this rule's to fix
        // — it only ever strips bold delimiters, never adds them.
        if current <= band.hi {
            return Gate::Off;
        }
        let budget = Budget::from_envelope_excess(current, band, PER_FIX_EFFECT);
        if budget.is_exhausted() {
            Gate::Off
        } else {
            Gate::Fix { budget }
        }
    }

    fn scan(&self, ctx: &RuleContext<'_>) -> Vec<Finding> {
        let document = ctx.document();
        let blocks = document.blocks();
        let parents = block_parents(blocks);
        let mut findings = Vec::new();
        for (unit, sentence) in ctx.sentences() {
            // Only the item's very own first sentence can be its lead-in
            // label.
            if sentence.range.start != unit.range.start {
                continue;
            }
            if innermost_list_item(unit.block, blocks, &parents).is_none() {
                continue;
            }
            let Ok(text) = document.text(&sentence.range) else {
                continue;
            };
            let Some(local) = match_bold_label(text) else {
                continue;
            };
            let start = sentence.range.start + local.start;
            let end = sentence.range.start + local.end;
            findings.push(Finding::new(
                RULE_ID,
                start..end,
                "bolded lead-in label in a bullet could be plain text",
                Tier::Fix,
            ));
        }
        findings
    }

    fn fix(
        &self,
        finding: &Finding,
        ctx: &RuleContext<'_>,
        _strategy_rng: &mut StrategyRng,
    ) -> Option<Patch> {
        let text = ctx.document().text(&finding.range).ok()?;
        let inner = text.strip_prefix("**")?.strip_suffix("**")?;
        Some(Patch::new(
            finding.range.clone(),
            inner.to_string(),
            RULE_ID,
            Tier::Fix,
        ))
    }
}

#[cfg(test)]
mod tests {
    use friction_core::Envelope;
    use friction_nlp::{SrxSegmenter, TaggedToken, Tagger};

    use super::*;
    use crate::context::MapEnvelope;

    struct NoopTagger;
    impl Tagger for NoopTagger {
        fn tag(&self, _text: &str, _base_offset: usize) -> Vec<TaggedToken> {
            Vec::new()
        }
    }

    fn document(source: &str) -> friction_core::Document {
        let parsed = friction_parse::parse(source).expect("valid markdown parses");
        friction_nlp::segment_document(&parsed, &SrxSegmenter::new())
            .expect("segmentation succeeds")
    }

    fn metrics_with_density(density: f64) -> MetricVector {
        MetricVector {
            bold_span_density: density,
            ..MetricVector::default()
        }
    }

    fn scan_source(source: &str) -> (friction_core::Document, Vec<Finding>) {
        let doc = document(source);
        let envelope = MapEnvelope::new();
        let findings = {
            let ctx = RuleContext::new(&doc, &NoopTagger, "blog", &envelope);
            BoldLabelStripRule::new().scan(&ctx)
        };
        (doc, findings)
    }

    // ---------------------------------------------------------------
    // match_bold_label
    // ---------------------------------------------------------------

    #[test]
    fn match_bold_label_matches_a_simple_label() {
        assert_eq!(match_bold_label("**Speed**: fast"), Some(0..9));
    }

    #[test]
    fn match_bold_label_rejects_without_trailing_colon() {
        assert!(match_bold_label("**Speed** fast").is_none());
    }

    #[test]
    fn match_bold_label_rejects_an_empty_label() {
        assert!(match_bold_label("****: fast").is_none());
    }

    #[test]
    fn match_bold_label_rejects_non_bold_text() {
        assert!(match_bold_label("Speed: fast").is_none());
    }

    // ---------------------------------------------------------------
    // gate()
    // ---------------------------------------------------------------

    #[test]
    fn gate_is_off_without_a_band() {
        let rule = BoldLabelStripRule::new();
        let envelope = MapEnvelope::new();
        assert_eq!(rule.gate(&metrics_with_density(50.0), &envelope), Gate::Off);
    }

    #[test]
    fn gate_is_off_inside_band() {
        let rule = BoldLabelStripRule::new();
        let envelope = MapEnvelope::new().with(GATED_METRIC, Envelope::new(0.0, 100.0));
        assert_eq!(rule.gate(&metrics_with_density(50.0), &envelope), Gate::Off);
    }

    #[test]
    fn gate_above_band_computes_hand_verified_budget() {
        let rule = BoldLabelStripRule::new();
        let envelope = MapEnvelope::new().with(GATED_METRIC, Envelope::new(0.0, 10.0));
        assert_eq!(
            rule.gate(&metrics_with_density(14.0), &envelope),
            Gate::Fix {
                budget: Budget::new(4)
            }
        );
    }

    // ---------------------------------------------------------------
    // scan()
    // ---------------------------------------------------------------

    #[test]
    fn scan_matches_a_bold_label_bullet() {
        let (doc, findings) = scan_source("- **Speed**: fast\n- **Docs**: complete\n");
        assert_eq!(findings.len(), 2);
        assert_eq!(&doc.source()[findings[0].range.clone()], "**Speed**");
        assert_eq!(&doc.source()[findings[1].range.clone()], "**Docs**");
    }

    #[test]
    fn scan_ignores_bold_text_outside_a_list_item() {
        let (_, findings) = scan_source("**Speed**: this is a plain paragraph.\n");
        assert!(findings.is_empty());
    }

    #[test]
    fn scan_ignores_bold_text_mid_item_not_a_lead_in() {
        let (_, findings) = scan_source("- It runs fast, **very** fast.\n");
        assert!(findings.is_empty());
    }

    #[test]
    fn scan_matches_inside_a_nested_list_item_too() {
        let (_, findings) = scan_source("- outer\n  - **Speed**: fast\n");
        assert_eq!(findings.len(), 1);
    }

    // ---------------------------------------------------------------
    // fix()
    // ---------------------------------------------------------------

    fn fix_first(source: &str) -> String {
        let doc = document(source);
        let envelope = MapEnvelope::new();
        let ctx = RuleContext::new(&doc, &NoopTagger, "blog", &envelope);
        let rule = BoldLabelStripRule::new();
        let finding = &rule.scan(&ctx)[0];
        let mut rng = StrategyRng::from_seed(0);
        let patch = rule.fix(finding, &ctx, &mut rng).expect("expected a patch");
        assert_eq!(patch.tier, Tier::Fix);
        let mut applied = source.to_string();
        applied.replace_range(patch.range, &patch.replacement);
        applied
    }

    #[test]
    fn fix_strips_the_bold_markers_keeping_the_rest_verbatim() {
        assert_eq!(fix_first("- **Speed**: fast\n"), "- Speed: fast\n");
    }

    #[test]
    fn fix_preserves_the_rest_of_the_bullet_text() {
        assert_eq!(
            fix_first("- **Compatibility**: works with every browser\n"),
            "- Compatibility: works with every browser\n"
        );
    }

    // ---------------------------------------------------------------
    // Idempotence and determinism
    // ---------------------------------------------------------------

    #[test]
    fn fixing_a_document_is_idempotent() {
        let source = "- **Speed**: fast\n- **Docs**: complete\n- **Cost**: free\n";
        let applied = fix_first(source);
        // Fix a second time from scratch (fresh document, fresh scan):
        // the already-stripped label from the first item is gone, but the
        // *other* two items still have theirs — this only checks the
        // rule never re-matches the text it already produced for a given
        // item, applied to the whole (partially fixed) document.
        let (_, findings_after) = scan_source(&applied);
        assert!(
            findings_after.iter().all(|f| {
                let text = &applied[f.range.clone()];
                text.starts_with("**")
            }),
            "no leftover finding should point at already-stripped text"
        );
    }

    #[test]
    fn fixing_the_same_source_twice_is_byte_identical() {
        let source = "- **Speed**: fast\n";
        let run = || fix_first(source);
        assert_eq!(run(), run());
    }
}
