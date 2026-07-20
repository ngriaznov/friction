//! [`HeaderMergeRule`]: flags a repeated
//! heading-immediately-followed-by-one-short-paragraph pattern as a
//! candidate for merging into flowing prose.
//!
//! # What qualifies as one "section"
//!
//! A heading block, immediately followed — with nothing at all in
//! between, at the same nesting level (see [`super::block_parents`]) —
//! by exactly one [`friction_core::BlockKind::Paragraph`] block, whose own
//! prose text is at most [`SHORT_PARAGRAPH_MAX_TOKENS`] word tokens (see
//! [`super::count_word_tokens`]), before the next heading (any level) or
//! the end of the document.
//!
//! # Why "repeated" matters
//!
//! One heading followed by one short paragraph is completely ordinary
//! document structure — a short section is not a tell on its own. The
//! *pattern repeating* across a document (several sections in a row, each
//! one heading plus one short paragraph and nothing else) is the
//! machine-flavored tell this rule looks for: a uniform, templated
//! heading-per-thought outline a human writer's own sections rarely fall
//! into quite so mechanically. [`scan`](HeaderMergeRule::scan) only
//! surfaces findings at all once [`MIN_REPEATED_SECTIONS`] qualifying
//! sections exist in the same document.
//!
//! # Why this is Suggest tier, never Fix
//!
//! Merging several one-paragraph sections into flowing prose is a genuine
//! editorial decision: which heading's wording survives as a topic
//! sentence, whether the sections even belong in the same paragraph, and
//! whether the document's readers rely on those headings to navigate or
//! skim are all judgment calls this rule has no safe, meaning-preserving
//! default for. So `fix` always declines (`None`) and no patch is ever
//! proposed — this rule only ever gates `Detect`, never `Fix`, so `fix` is
//! never actually invoked in ordinary operation; it exists only to satisfy
//! [`crate::Rule`]'s object-safe shape.

use friction_core::{Block, BlockKind, Document, Finding, MetricVector, Patch, RuleId, Tier};

use super::{block_parents, count_word_tokens};
use crate::context::{GenreEnvelope, RuleContext};
use crate::rule::{Gate, Rule, RuleFamily};
use crate::strategy::StrategyRng;

/// This rule's stable identifier.
const RULE_ID: RuleId = RuleId::new("structural.header_merge");

/// The [`MetricVector`] field this rule gates on.
const GATED_METRIC: &str = "heading_density";

/// The maximum word-token count (see [`super::count_word_tokens`]) a
/// section's paragraph may have and still count as "short".
const SHORT_PARAGRAPH_MAX_TOKENS: usize = 40;

/// The minimum number of qualifying sections a document must have before
/// this rule surfaces anything at all — see the module docs' "Why
/// 'repeated' matters" section.
const MIN_REPEATED_SECTIONS: usize = 2;

/// `document`'s directly-owned prose text for `block_index`, joining more
/// than one [`friction_core::ProseUnit`] run (if any) with a single space.
/// `None` if the block owns no prose at all.
fn block_prose_text(block_index: usize, document: &Document) -> Option<String> {
    let mut combined = String::new();
    let mut any = false;
    for unit in document
        .prose()
        .iter()
        .filter(|unit| unit.block == block_index)
    {
        if let Ok(text) = document.text(&unit.range) {
            if any {
                combined.push(' ');
            }
            combined.push_str(text);
            any = true;
        }
    }
    any.then_some(combined)
}

/// The index, in `blocks`, of the next [`BlockKind::Heading`] strictly
/// after `from`, or `blocks.len()` if none remains.
fn next_heading_index(from: usize, blocks: &[Block]) -> usize {
    blocks[from..]
        .iter()
        .position(|block| matches!(block.kind, BlockKind::Heading { .. }))
        .map_or(blocks.len(), |offset| from + offset)
}

/// Every qualifying `(heading_index, paragraph_index)` section in
/// `document`, in source order — see the module docs' "What qualifies"
/// section.
fn qualifying_sections(
    document: &Document,
    blocks: &[Block],
    parents: &[Option<usize>],
) -> Vec<(usize, usize)> {
    let mut sections = Vec::new();
    for (index, block) in blocks.iter().enumerate() {
        if !matches!(block.kind, BlockKind::Heading { .. }) {
            continue;
        }
        let next_heading = next_heading_index(index + 1, blocks);
        let between = &blocks[(index + 1)..next_heading];
        if between.len() != 1 {
            continue;
        }
        let paragraph_index = index + 1;
        if between[0].kind != BlockKind::Paragraph {
            continue;
        }
        if parents[paragraph_index] != parents[index] {
            continue;
        }
        let Some(text) = block_prose_text(paragraph_index, document) else {
            continue;
        };
        if count_word_tokens(&text) > SHORT_PARAGRAPH_MAX_TOKENS {
            continue;
        }
        sections.push((index, paragraph_index));
    }
    sections
}

/// Flags a repeated heading-plus-one-short-paragraph pattern as a
/// candidate for merging into flowing prose. Diagnostic only — see the
/// module docs for why this never proposes a patch.
#[derive(Debug, Clone, Copy, Default)]
pub struct HeaderMergeRule;

impl HeaderMergeRule {
    /// Creates the rule.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Rule for HeaderMergeRule {
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
        // This rule never fixes anything (see the module docs), so there
        // is no budget to compute — only whether to surface diagnostics
        // at all this round.
        if metrics.heading_density <= band.hi {
            Gate::Off
        } else {
            Gate::Detect
        }
    }

    fn scan(&self, ctx: &RuleContext<'_>) -> Vec<Finding> {
        let document = ctx.document();
        let blocks = document.blocks();
        let parents = block_parents(blocks);
        let sections = qualifying_sections(document, blocks, &parents);
        if sections.len() < MIN_REPEATED_SECTIONS {
            return Vec::new();
        }
        sections
            .into_iter()
            .map(|(heading_index, paragraph_index)| {
                let range = blocks[heading_index].range.start..blocks[paragraph_index].range.end;
                Finding::new(
                    RULE_ID,
                    range,
                    "heading immediately followed by one short paragraph repeats across this \
                     document; consider merging these sections into flowing prose",
                    Tier::Suggest,
                )
            })
            .collect()
    }

    fn fix(
        &self,
        _finding: &Finding,
        _ctx: &RuleContext<'_>,
        _strategy_rng: &mut StrategyRng,
    ) -> Option<Patch> {
        // See the module docs' "Why this is Suggest tier, never Fix"
        // section: this rule gates `Detect` exclusively, so `friction-
        // apply`'s driver never actually calls this. Always declining
        // keeps that contract true even if a caller invoked it directly.
        None
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

    fn document(source: &str) -> Document {
        let parsed = friction_parse::parse(source).expect("valid markdown parses");
        friction_nlp::segment_document(&parsed, &SrxSegmenter::new())
            .expect("segmentation succeeds")
    }

    fn metrics_with_density(density: f64) -> MetricVector {
        MetricVector {
            heading_density: density,
            ..MetricVector::default()
        }
    }

    fn scan_source(source: &str) -> Vec<Finding> {
        let doc = document(source);
        let envelope = MapEnvelope::new();
        let ctx = RuleContext::new(&doc, &NoopTagger, "blog", &envelope);
        HeaderMergeRule::new().scan(&ctx)
    }

    // ---------------------------------------------------------------
    // gate()
    // ---------------------------------------------------------------

    #[test]
    fn gate_is_off_without_a_band() {
        let rule = HeaderMergeRule::new();
        let envelope = MapEnvelope::new();
        assert_eq!(rule.gate(&metrics_with_density(50.0), &envelope), Gate::Off);
    }

    #[test]
    fn gate_is_off_inside_band() {
        let rule = HeaderMergeRule::new();
        let envelope = MapEnvelope::new().with(GATED_METRIC, Envelope::new(0.0, 100.0));
        assert_eq!(rule.gate(&metrics_with_density(50.0), &envelope), Gate::Off);
    }

    #[test]
    fn gate_is_detect_above_band() {
        let rule = HeaderMergeRule::new();
        let envelope = MapEnvelope::new().with(GATED_METRIC, Envelope::new(0.0, 10.0));
        assert_eq!(
            rule.gate(&metrics_with_density(20.0), &envelope),
            Gate::Detect
        );
    }

    // ---------------------------------------------------------------
    // scan()
    // ---------------------------------------------------------------

    const REPEATED_SECTIONS: &str = "## First\n\nA short paragraph here.\n\n## Second\n\nAnother short one.\n\n## Third\n\nAnd a third short paragraph.\n";

    #[test]
    fn scan_finds_every_repeated_section() {
        let findings = scan_source(REPEATED_SECTIONS);
        assert_eq!(findings.len(), 3);
        for finding in &findings {
            assert_eq!(finding.tier, Tier::Suggest);
        }
    }

    #[test]
    fn scan_reports_nothing_below_the_repetition_threshold() {
        // Only one qualifying section in the whole document: not a
        // repeated pattern, so nothing is surfaced even though the lone
        // section, taken alone, would otherwise qualify.
        let source = "## Only section\n\nA short paragraph here.\n";
        assert!(scan_source(source).is_empty());
    }

    #[test]
    fn scan_ignores_a_heading_followed_by_a_long_paragraph() {
        let long_paragraph = "word ".repeat(50);
        let source = format!(
            "## First\n\n{long_paragraph}\n\n## Second\n\nShort one.\n\n## Third\n\nShort two.\n"
        );
        let findings = scan_source(&source);
        // Only "Second" and "Third" qualify; "First" (long paragraph)
        // does not, but two qualifying sections still clears the
        // repetition threshold.
        assert_eq!(findings.len(), 2);
    }

    #[test]
    fn scan_ignores_a_heading_followed_by_more_than_one_block() {
        let source = "## First\n\nShort one.\n\nAnother paragraph.\n\n## Second\n\nShort two.\n\n## Third\n\nShort three.\n";
        let findings = scan_source(source);
        // "First" has two paragraphs after it, not exactly one -> doesn't
        // qualify; "Second" and "Third" still clear the threshold.
        assert_eq!(findings.len(), 2);
    }

    #[test]
    fn scan_ignores_a_heading_followed_by_a_list() {
        let source =
            "## First\n\n- one\n- two\n\n## Second\n\nShort two.\n\n## Third\n\nShort three.\n";
        let findings = scan_source(source);
        assert_eq!(findings.len(), 2);
    }

    #[test]
    fn no_op_on_plain_prose_without_headings() {
        assert!(
            scan_source("Just a paragraph of ordinary prose. Nothing structural here.\n")
                .is_empty()
        );
    }

    // ---------------------------------------------------------------
    // fix() never proposes a patch
    // ---------------------------------------------------------------

    #[test]
    fn fix_always_declines() {
        let doc = document(REPEATED_SECTIONS);
        let envelope = MapEnvelope::new();
        let ctx = RuleContext::new(&doc, &NoopTagger, "blog", &envelope);
        let rule = HeaderMergeRule::new();
        let findings = rule.scan(&ctx);
        assert!(!findings.is_empty());
        let mut rng = StrategyRng::from_seed(0);
        for finding in &findings {
            assert!(rule.fix(finding, &ctx, &mut rng).is_none());
        }
    }

    // ---------------------------------------------------------------
    // Determinism
    // ---------------------------------------------------------------

    #[test]
    fn scanning_the_same_source_twice_is_byte_identical() {
        let run = || scan_source(REPEATED_SECTIONS);
        let a = run();
        let b = run();
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.range, y.range);
            assert_eq!(x.message, y.message);
        }
    }
}
