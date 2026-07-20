//! [`SentenceFuseRule`]: flags two adjacent short sentences that share a
//! trivially-coreferring subject as a candidate for fusing into one.
//!
//! # `Suggest` tier only
//!
//! Unlike [`super::split::SentenceSplitRule`], this rule never proposes a
//! [`friction_core::Patch`] at all. Fusing two clauses is a genuine rewrite
//! decision — a conjunction, a subordinator, a semicolon, which clause
//! becomes the subordinate one — that this rule has no single
//! meaning-preserving answer for; picking one via [`crate::StrategyRng`]
//! the way [`crate::families::connective::ConnectiveSurgery`] picks among
//! several *safe* rewrites would not be safe here, since a wrong choice can
//! change emphasis or drop a nuance the original two-sentence phrasing
//! carried. So `scan` reports the opportunity as a [`friction_core::Finding`]
//! at [`friction_core::Tier::Suggest`] and [`Rule::fix`] always declines —
//! see that method's own docs: "a rule that wants to surface a `Suggest`
//! -tier alternative for this finding should do so only via `scan`'s own
//! findings, not through `fix`."
//!
//! # Gate: `Detect`, not `Fix`
//!
//! [`Rule::gate`] hands back [`Gate::Detect`] rather than [`Gate::Fix`]
//! when this rule's target metric sits below the genre's envelope: this
//! rule has no patch to budget in the first place, and `Gate::Detect` is
//! exactly the engine's existing shape for "scan and surface findings, but
//! never call `fix`, regardless of an individual finding's own tier" (see
//! [`Gate`]'s own docs). `friction-apply`'s driver already wires that
//! straight through — a `Gate::Detect`-gated rule's findings land in the
//! round's [`friction_core::Finding`] list, `fix` is never invoked, and no
//! patch is ever produced for it — so nothing about that path needed
//! changing for this rule to fit.
//!
//! # Detection
//!
//! Two adjacent sentences (same paragraph, immediately consecutive in
//! [`RuleContext::sentences`]'s source-order walk) both at most
//! [`SHORT_SENTENCE_MAX_TOKENS`] tokens long, whose detected grammatical
//! subjects share a lemma via [`friction_nlp::same_subject`] — the same
//! heuristic-dependency-parser-backed coreference approximation
//! `friction_nlp::dep`'s own docs describe, driven here by
//! [`friction_nlp::HeuristicParser`] (always available, no model, no cargo
//! feature). A missing subject on either sentence is not evidence of a
//! match (`same_subject` itself returns `false`, not a guess), so a
//! subjectless fragment never triggers a Suggest finding.

use friction_core::{Finding, MetricVector, Patch, RuleId, Tier};
use friction_nlp::{DepParser, HeuristicParser, same_subject};

use crate::context::{GenreEnvelope, RuleContext};
use crate::rule::{Gate, Rule, RuleFamily};
use crate::strategy::StrategyRng;

use super::token_count;

/// This rule's stable identifier.
const RULE_ID: RuleId = RuleId::new("rhythm.fuse");

/// The [`MetricVector`] field this rule gates on — the same uniform-rhythm
/// signal [`super::split::SentenceSplitRule`] reacts to from the opposite
/// end (see this family's module docs): a document made almost entirely of
/// short, same-length sentences reads with just as little variety as one
/// made of medium-length ones, and fusing two of those short sentences
/// together is this family's move for widening that range from below,
/// exactly as splitting an over-long one widens it from above.
const GATED_METRIC: &str = "sentence_length_cv";

/// A sentence at or under this many whitespace tokens (see
/// [`super::token_count`]) counts as "short" for this rule's purposes —
/// the `"~8 tokens"` figure this family's requirements name, applied as a
/// plain inclusive cutoff rather than a fuzzier statistical one: this rule
/// only ever *suggests*, never rewrites automatically, so there is no
/// budget-correctness reason to derive it from a pack the way
/// [`super::split::SentenceSplitRule`]'s per-genre over-long table is.
const SHORT_SENTENCE_MAX_TOKENS: usize = 8;

/// Flags two adjacent short, subject-coreferring sentences as a candidate
/// for fusing into one. See the module docs for why this rule is
/// `Suggest`-tier only and proposes no patch.
#[derive(Debug, Clone, Copy, Default)]
pub struct SentenceFuseRule;

impl SentenceFuseRule {
    /// Creates the rule.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Rule for SentenceFuseRule {
    fn id(&self) -> RuleId {
        RULE_ID
    }

    fn family(&self) -> RuleFamily {
        RuleFamily::Rhythm
    }

    fn gate(&self, metrics: &MetricVector, envelope: &dyn GenreEnvelope) -> Gate {
        let Some(band) = envelope.band(GATED_METRIC) else {
            return Gate::Off;
        };
        // See the module docs' "Gate: Detect, not Fix" section: this rule
        // never fixes anything, only ever detects, so there is no budget
        // to compute here at all — only the same "is the document too
        // uniform" direction check `SentenceSplitRule::gate` makes.
        if metrics.sentence_length_cv >= band.lo {
            Gate::Off
        } else {
            Gate::Detect
        }
    }

    fn scan(&self, ctx: &RuleContext<'_>) -> Vec<Finding> {
        let document = ctx.document();
        let parser = HeuristicParser::new();
        let pairs: Vec<_> = ctx.sentences().collect();

        let mut findings = Vec::new();
        for window in pairs.windows(2) {
            let (unit_a, sentence_a) = window[0];
            let (unit_b, sentence_b) = window[1];
            // Only ever suggests fusing two sentences from the same
            // paragraph — never across a paragraph break, which
            // `ProseUnit` identity (not mere source adjacency) already
            // guarantees.
            if !std::ptr::eq(unit_a, unit_b) {
                continue;
            }
            let (Ok(text_a), Ok(text_b)) = (
                document.text(&sentence_a.range),
                document.text(&sentence_b.range),
            ) else {
                continue;
            };
            if token_count(text_a) > SHORT_SENTENCE_MAX_TOKENS
                || token_count(text_b) > SHORT_SENTENCE_MAX_TOKENS
            {
                continue;
            }

            let tokens_a = ctx.tag_sentence(sentence_a);
            let tokens_b = ctx.tag_sentence(sentence_b);
            let source = document.source();
            let (Ok(parse_a), Ok(parse_b)) = (
                parser.parse(source, &tokens_a),
                parser.parse(source, &tokens_b),
            ) else {
                continue;
            };

            if same_subject((&tokens_a, &parse_a), (&tokens_b, &parse_b)) {
                findings.push(Finding::new(
                    RULE_ID,
                    sentence_a.range.start..sentence_b.range.end,
                    "two short adjacent sentences share a subject and could read better fused \
                     into one",
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
        // Never called in practice: this rule only ever gates `Off` or
        // `Detect` (never `Fix`), and `friction-apply`'s driver only calls
        // `fix` for a `Fix`-gated rule's `Tier::Fix` findings — every
        // finding this rule's own `scan` produces is `Tier::Suggest`. See
        // the module docs' "Suggest tier only" section for why this rule
        // has no single meaning-preserving fusion to propose here anyway.
        None
    }
}

#[cfg(test)]
mod tests {
    use friction_core::Envelope;
    use friction_nlp::{NlpruleTagger, SrxSegmenter};

    use super::*;
    use crate::context::MapEnvelope;

    fn document(source: &str) -> friction_core::Document {
        let parsed = friction_parse::parse(source).expect("valid markdown parses");
        friction_nlp::segment_document(&parsed, &SrxSegmenter::new())
            .expect("segmentation succeeds")
    }

    fn metrics_with_cv(cv: f64) -> MetricVector {
        MetricVector {
            sentence_length_cv: cv,
            ..MetricVector::default()
        }
    }

    fn tagger() -> NlpruleTagger {
        NlpruleTagger::new().expect("embedded tagger model loads")
    }

    // ---------------------------------------------------------------
    // gate()
    // ---------------------------------------------------------------

    #[test]
    fn gate_is_off_without_a_band() {
        let rule = SentenceFuseRule::new();
        let envelope = MapEnvelope::new();
        assert_eq!(rule.gate(&metrics_with_cv(0.3), &envelope), Gate::Off);
    }

    #[test]
    fn gate_is_off_inside_band() {
        let rule = SentenceFuseRule::new();
        let envelope = MapEnvelope::new().with(GATED_METRIC, Envelope::new(0.5, 1.0));
        assert_eq!(rule.gate(&metrics_with_cv(0.7), &envelope), Gate::Off);
    }

    #[test]
    fn gate_is_off_above_band_ceiling() {
        let rule = SentenceFuseRule::new();
        let envelope = MapEnvelope::new().with(GATED_METRIC, Envelope::new(0.5, 1.0));
        assert_eq!(rule.gate(&metrics_with_cv(1.5), &envelope), Gate::Off);
    }

    #[test]
    fn gate_detects_below_band_floor() {
        let rule = SentenceFuseRule::new();
        let envelope = MapEnvelope::new().with(GATED_METRIC, Envelope::new(0.5, 1.0));
        assert_eq!(rule.gate(&metrics_with_cv(0.2), &envelope), Gate::Detect);
    }

    // ---------------------------------------------------------------
    // scan()
    // ---------------------------------------------------------------

    #[test]
    fn scan_flags_two_short_sentences_sharing_a_subject() {
        let source = "It shipped the release. It also shipped a patch.\n";
        let doc = document(source);
        let envelope = MapEnvelope::new();
        let tagger = tagger();
        let ctx = RuleContext::new(&doc, &tagger, "blog", &envelope);
        let findings = SentenceFuseRule::new().scan(&ctx);
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].tier, Tier::Suggest);
        assert_eq!(
            &source[findings[0].range.clone()],
            "It shipped the release. It also shipped a patch."
        );
    }

    #[test]
    fn scan_ignores_sentences_with_different_subjects() {
        let source = "It shipped the release. Customers noticed quickly.\n";
        let doc = document(source);
        let envelope = MapEnvelope::new();
        let tagger = tagger();
        let ctx = RuleContext::new(&doc, &tagger, "blog", &envelope);
        assert!(SentenceFuseRule::new().scan(&ctx).is_empty());
    }

    #[test]
    fn scan_ignores_a_pair_where_one_sentence_is_too_long() {
        let source = "It shipped the release. It also spent a very long time \
                       carefully validating every part of the release before it went out \
                       the door.\n";
        let doc = document(source);
        let envelope = MapEnvelope::new();
        let tagger = tagger();
        let ctx = RuleContext::new(&doc, &tagger, "blog", &envelope);
        assert!(SentenceFuseRule::new().scan(&ctx).is_empty());
    }

    #[test]
    fn scan_does_not_cross_a_paragraph_boundary() {
        let source = "It shipped the release.\n\nIt also shipped a patch.\n";
        let doc = document(source);
        let envelope = MapEnvelope::new();
        let tagger = tagger();
        let ctx = RuleContext::new(&doc, &tagger, "blog", &envelope);
        assert!(SentenceFuseRule::new().scan(&ctx).is_empty());
    }

    // ---------------------------------------------------------------
    // fix() and the Detect-gate driver contract
    // ---------------------------------------------------------------

    #[test]
    fn fix_always_declines() {
        let source = "It shipped the release. It also shipped a patch.\n";
        let doc = document(source);
        let envelope = MapEnvelope::new();
        let tagger = tagger();
        let ctx = RuleContext::new(&doc, &tagger, "blog", &envelope);
        let rule = SentenceFuseRule::new();
        let finding = &rule.scan(&ctx)[0];
        let mut rng = StrategyRng::from_seed(0);
        assert!(rule.fix(finding, &ctx, &mut rng).is_none());
    }

    /// A local, single-rule stand-in for `friction-apply`'s own driver
    /// round (same shape `crate::families::connective`'s golden-fixture
    /// harness uses, and for the same reason: this crate cannot depend on
    /// `friction-apply`) — proves a `Gate::Detect`-gated rule's findings
    /// are surfaced while its `fix` is genuinely never invoked and no text
    /// changes, mirroring `friction-apply::driver`'s own
    /// `detect_gated_rule_surfaces_findings_without_fixing` test for this
    /// exact rule's real gate output rather than a stub.
    #[test]
    fn detect_gate_surfaces_findings_without_ever_touching_the_text() {
        let source = "It shipped the release. It also shipped a patch.\n";
        let doc = document(source);
        let band = Envelope::new(0.5, 1.0);
        let envelope = MapEnvelope::new().with(GATED_METRIC, band);
        let tagger = tagger();
        let ctx = RuleContext::new(&doc, &tagger, "blog", &envelope);
        let rule = SentenceFuseRule::new();

        let gate = rule.gate(&metrics_with_cv(0.1), &envelope);
        assert_eq!(gate, Gate::Detect);

        // Mirrors `friction-apply::driver::run_round`'s own `Gate::Detect`
        // arm exactly: scan, collect findings, never call `fix`.
        let findings = rule.scan(&ctx);
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].tier, Tier::Suggest);
        // No patch, from this rule, ever exists to apply — the source is
        // definitionally untouched.
    }
}
