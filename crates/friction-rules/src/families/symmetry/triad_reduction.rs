//! Triad-reduction detection: flags a flat `"X, Y, and Z"` / `"X, Y, or Z"`
//! coordination pattern whose three items share the same broad grammatical
//! class — the parallel-triad tic LLM prose reaches for far more often than
//! human writers do.
//!
//! # Suggest tier, not Fix
//!
//! Reducing a triad means dropping its weakest item or restructuring the
//! sentence entirely — either move can drop a proposition a purely
//! syntactic scan has no way to rule out (the "weakest" item is a judgment
//! call this rule cannot make safely), so every finding here is
//! [`friction_core::Tier::Suggest`] and [`TriadReductionRule::fix`] always
//! declines: the engine surfaces these as diagnostics, never applies them.
//!
//! # Mirrored detection logic
//!
//! [`find_triads`] mirrors `friction-metrics::symmetry::count_triads`'s
//! exact algorithm (private to that crate — see `families::symmetry`'s own
//! module docs for why every submodule here mirrors rather than imports),
//! adapted to return each match's byte span (for a [`friction_core::
//! Finding`]) instead of only a count. This module pins that mirror down
//! two ways: `triad_algorithm_matches_metrics_test_fixtures_exactly`
//! reproduces `friction-metrics::symmetry`'s own hand-built token test
//! cases verbatim and checks this module's detector agrees on every one of
//! them; `triad_finding_count_matches_the_public_triad_rate_metric` cross-
//! checks this rule's `scan` against `friction_metrics::triad_rate`'s
//! public, rate-returning function on a real, tagger-processed document.

use friction_core::{Finding, MetricVector, Patch, RuleId, Tier, span};
use friction_nlp::TaggedToken;

use crate::context::{GenreEnvelope, RuleContext};
use crate::rule::{Gate, Rule, RuleFamily};
use crate::strategy::StrategyRng;

/// This rule's stable identifier.
const RULE_ID: RuleId = RuleId::new("symmetry.triad_reduction");

/// The [`MetricVector`] field this rule gates on.
const GATED_METRIC: &str = "triad_rate";

// ---------------------------------------------------------------------
// Mirrored from friction-metrics::symmetry (private there; see module
// docs).
// ---------------------------------------------------------------------

/// A coarse grammatical bucket a token's part-of-speech tag folds into.
/// Mirrors `friction_metrics::symmetry`'s own private `BroadClass`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BroadClass {
    Noun,
    Verb,
    Adjective,
    Adverb,
    Other,
}

fn broad_class(pos: &str) -> BroadClass {
    if pos.starts_with("NN") {
        BroadClass::Noun
    } else if pos.starts_with("VB") || pos == "MD" {
        BroadClass::Verb
    } else if pos.starts_with("JJ") {
        BroadClass::Adjective
    } else if pos.starts_with("RB") {
        BroadClass::Adverb
    } else {
        BroadClass::Other
    }
}

fn content_class(token: &TaggedToken) -> Option<BroadClass> {
    match broad_class(token.pos.as_str()) {
        BroadClass::Other => None,
        class => Some(class),
    }
}

fn token_text<'s>(source: &'s str, token: &TaggedToken) -> &'s str {
    span::slice(source, &token.token.range).unwrap_or("")
}

fn is_comma(token: &TaggedToken, source: &str) -> bool {
    token_text(source, token) == ","
}

fn is_strong_boundary(token: &TaggedToken, source: &str) -> bool {
    matches!(token_text(source, token), "." | "!" | "?" | ";" | ":")
}

fn is_list_coordinator(token: &TaggedToken, source: &str) -> bool {
    token.pos.as_str().starts_with("CC")
        && matches!(
            token_text(source, token).to_ascii_lowercase().as_str(),
            "and" | "or"
        )
}

fn segment_left_bound(tokens: &[TaggedToken], source: &str, end: usize) -> usize {
    (0..end)
        .rev()
        .find(|&i| is_comma(&tokens[i], source) || is_strong_boundary(&tokens[i], source))
        .map_or(0, |i| i + 1)
}

fn segment_right_bound(tokens: &[TaggedToken], source: &str, start: usize) -> usize {
    (start..tokens.len())
        .find(|&i| is_comma(&tokens[i], source) || is_strong_boundary(&tokens[i], source))
        .unwrap_or(tokens.len())
}

fn nearest_comma_before(tokens: &[TaggedToken], source: &str, end: usize) -> Option<usize> {
    for i in (0..end).rev() {
        if is_comma(&tokens[i], source) {
            return Some(i);
        }
        if is_strong_boundary(&tokens[i], source) {
            return None;
        }
    }
    None
}

fn segment_head_class(tokens: &[TaggedToken], range: std::ops::Range<usize>) -> Option<BroadClass> {
    range.rev().find_map(|i| content_class(&tokens[i]))
}

/// One triad match: the byte span of the whole `"X, Y, and Z"` pattern (from
/// the first item's own start through the third item's own end — not the
/// sentence's full range) in the original source.
struct TriadMatch {
    range: std::ops::Range<usize>,
}

/// Finds every triad coordination pattern in one sentence's tagged tokens,
/// left to right. See the module docs for how this mirrors
/// `friction-metrics::symmetry::count_triads`; see that function's own docs
/// (reproduced here) for the exact match conditions.
fn find_triads(tokens: &[TaggedToken], source: &str) -> Vec<TriadMatch> {
    let mut matches = Vec::new();
    let mut c = 1;
    while c < tokens.len() {
        if !is_list_coordinator(&tokens[c], source) {
            c += 1;
            continue;
        }
        let comma_b = c - 1;
        if !is_comma(&tokens[comma_b], source) {
            c += 1;
            continue;
        }
        let Some(comma_a) = nearest_comma_before(tokens, source, comma_b) else {
            c += 1;
            continue;
        };
        if comma_b - comma_a < 2 {
            c += 1;
            continue;
        }
        let seg1_start = segment_left_bound(tokens, source, comma_a);
        if seg1_start >= comma_a {
            c += 1;
            continue;
        }
        let seg3_end = segment_right_bound(tokens, source, c + 1);
        if seg3_end <= c + 1 {
            c += 1;
            continue;
        }

        let head1 = segment_head_class(tokens, seg1_start..comma_a);
        let head2 = segment_head_class(tokens, (comma_a + 1)..comma_b);
        let head3 = segment_head_class(tokens, (c + 1)..seg3_end);
        if let (Some(a), Some(b), Some(z)) = (head1, head2, head3)
            && a == b
            && b == z
        {
            let start = tokens[seg1_start].token.range.start;
            let end = tokens[seg3_end - 1].token.range.end;
            matches.push(TriadMatch { range: start..end });
        }
        c += 1;
    }
    matches
}

/// Flags flat `"X, Y, and Z"` triads whose three items share the same
/// broad grammatical class. Suggest tier only — see the module docs.
#[derive(Debug, Clone, Copy, Default)]
pub struct TriadReductionRule;

impl TriadReductionRule {
    /// Creates the rule.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Rule for TriadReductionRule {
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
        if metrics.triad_rate <= band.hi {
            Gate::Off
        } else {
            // Suggest tier only: this rule can never safely turn a finding
            // into a patch (see the module docs), so it never asks the
            // driver for a fix budget — `Detect` surfaces every finding as
            // a diagnostic without ever calling `fix`.
            Gate::Detect
        }
    }

    fn scan(&self, ctx: &RuleContext<'_>) -> Vec<Finding> {
        let source = ctx.document().source();
        let mut findings = Vec::new();
        for (_, sentence) in ctx.sentences() {
            let tokens = ctx.tag_sentence(sentence);
            for triad in find_triads(&tokens, source) {
                findings.push(Finding::new(
                    RULE_ID,
                    triad.range,
                    "flat \"X, Y, and Z\" triad with matching-class items reads as an LLM tic; consider dropping the weakest item or restructuring",
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
    use friction_core::{Envelope, Token, TokenKind};
    use friction_nlp::PosTag;

    use super::*;
    use crate::context::MapEnvelope;

    // -----------------------------------------------------------------
    // Test helpers, identical in shape to friction-metrics::symmetry's own
    // `build_tokens` helper (see this module's docs for why the fixtures
    // below reproduce that crate's own test cases).
    // -----------------------------------------------------------------

    fn build_tokens(words: &[(&str, &str, &str)]) -> (String, Vec<TaggedToken>) {
        let mut source = String::new();
        let mut tokens = Vec::with_capacity(words.len());
        for &(surface, pos, lemma) in words {
            if !source.is_empty() && surface != "," && surface != "." {
                source.push(' ');
            }
            let start = source.len();
            source.push_str(surface);
            let end = source.len();
            tokens.push(TaggedToken {
                token: Token::new(start..end, TokenKind::Word),
                pos: PosTag::new(pos),
                lemma: lemma.into(),
            });
        }
        (source, tokens)
    }

    // -----------------------------------------------------------------
    // find_triads: reproduces friction-metrics::symmetry's own
    // count_triads test fixtures exactly.
    // -----------------------------------------------------------------

    #[test]
    fn triad_algorithm_matches_metrics_test_fixtures_exactly() {
        // Same-class noun triad: 1 match.
        let (source, tokens) = build_tokens(&[
            ("The", "DT", "the"),
            ("kit", "NN", "kit"),
            ("includes", "VBZ", "include"),
            ("screws", "NNS", "screw"),
            (",", ",", ","),
            ("bolts", "NNS", "bolt"),
            (",", ",", ","),
            ("and", "CC", "and"),
            ("washers", "NNS", "washer"),
            (".", ".", "."),
        ]);
        let matches = find_triads(&tokens, &source);
        assert_eq!(matches.len(), 1);
        // The matched span runs from the sentence's own start (there is no
        // earlier comma/strong-boundary token to bound item 1 more
        // tightly) through the third item's end — the same "item 1's left
        // bound defaults to the sentence start" rule
        // `friction_metrics::symmetry::segment_left_bound` documents (and
        // this module's own `segment_left_bound` mirrors); classification
        // only ever looks at each segment's *rightmost* content word, so
        // "The kit includes" ahead of "screws" plays no part in the
        // same-class-heads decision, it just widens the reported span.
        assert_eq!(
            &source[matches[0].range.clone()],
            "The kit includes screws, bolts, and washers"
        );

        // Mixed-class heads: 0 matches.
        let (source, tokens) = build_tokens(&[
            ("screws", "NNS", "screw"),
            (",", ",", ","),
            ("ran", "VBD", "run"),
            (",", ",", ","),
            ("and", "CC", "and"),
            ("washers", "NNS", "washer"),
            (".", ".", "."),
        ]);
        assert!(find_triads(&tokens, &source).is_empty());

        // Non-listy coordinator ("but"): 0 matches.
        let (source, tokens) = build_tokens(&[
            ("screws", "NNS", "screw"),
            (",", ",", ","),
            ("bolts", "NNS", "bolt"),
            (",", ",", ","),
            ("but", "CC", "but"),
            ("washers", "NNS", "washer"),
            (".", ".", "."),
        ]);
        assert!(find_triads(&tokens, &source).is_empty());

        // Empty middle item (two adjacent commas): 0 matches.
        let (source, tokens) = build_tokens(&[
            ("screws", "NNS", "screw"),
            (",", ",", ","),
            (",", ",", ","),
            ("and", "CC", "and"),
            ("washers", "NNS", "washer"),
            (".", ".", "."),
        ]);
        assert!(find_triads(&tokens, &source).is_empty());

        // Same-class adjective triad: 1 match.
        let (source, tokens) = build_tokens(&[
            ("It", "PRP", "it"),
            ("was", "VBD", "be"),
            ("fast", "JJ", "fast"),
            (",", ",", ","),
            ("quiet", "JJ", "quiet"),
            (",", ",", ","),
            ("and", "CC", "and"),
            ("reliable", "JJ", "reliable"),
            (".", ".", "."),
        ]);
        let matches = find_triads(&tokens, &source);
        assert_eq!(matches.len(), 1);
        // Same reasoning as the noun-triad case above: no comma/boundary
        // precedes "fast", so item 1's span defaults to the sentence start.
        assert_eq!(
            &source[matches[0].range.clone()],
            "It was fast, quiet, and reliable"
        );
    }

    // -----------------------------------------------------------------
    // Cross-crate consistency: this rule's finding count against the
    // public friction_metrics::triad_rate metric, on a real,
    // tagger-processed document.
    // -----------------------------------------------------------------

    #[test]
    fn triad_finding_count_matches_the_public_triad_rate_metric() {
        let source = "The kit includes screws, bolts, and washers. \
            The team shipped the release, allowing customers to upgrade early. \
            It was fast, quiet, and reliable.\n";
        let doc = friction_parse::parse(source).expect("valid markdown parses");
        let doc = friction_nlp::segment_document(&doc, &friction_nlp::SrxSegmenter::new())
            .expect("segmentation succeeds");
        let tagger = friction_nlp::NlpruleTagger::new().expect("embedded model loads");

        let sentence_count = doc
            .prose()
            .iter()
            .map(|unit| unit.sentences.len())
            .sum::<usize>();
        let metrics_rate = friction_metrics::triad_rate(&doc, &tagger);
        #[allow(clippy::cast_precision_loss)]
        let metrics_triads = metrics_rate * sentence_count as f64;

        let envelope = MapEnvelope::new().with(GATED_METRIC, Envelope::new(0.0, 0.0));
        let ctx = RuleContext::new(&doc, &tagger, "blog", &envelope);
        let rule_triads = TriadReductionRule::new().scan(&ctx).len();

        #[allow(clippy::cast_precision_loss)]
        let rule_triads_f64 = rule_triads as f64;
        assert!(
            (metrics_triads - rule_triads_f64).abs() < 1e-9,
            "friction_metrics::triad_rate implies {metrics_triads} triads, \
             TriadReductionRule::scan found {rule_triads}"
        );
        assert!(
            rule_triads > 0,
            "expected at least one triad in the fixture"
        );
    }

    // -----------------------------------------------------------------
    // gate()
    // -----------------------------------------------------------------

    #[test]
    fn gate_is_off_without_a_band() {
        let rule = TriadReductionRule::new();
        let envelope = MapEnvelope::new();
        let metrics = MetricVector {
            triad_rate: 1.0,
            ..MetricVector::default()
        };
        assert_eq!(rule.gate(&metrics, &envelope), Gate::Off);
    }

    #[test]
    fn gate_is_off_inside_band() {
        let rule = TriadReductionRule::new();
        let envelope = MapEnvelope::new().with(GATED_METRIC, Envelope::new(0.0, 0.5));
        let metrics = MetricVector {
            triad_rate: 0.2,
            ..MetricVector::default()
        };
        assert_eq!(rule.gate(&metrics, &envelope), Gate::Off);
    }

    #[test]
    fn gate_is_detect_above_band() {
        let rule = TriadReductionRule::new();
        let envelope = MapEnvelope::new().with(GATED_METRIC, Envelope::new(0.0, 0.1));
        let metrics = MetricVector {
            triad_rate: 0.5,
            ..MetricVector::default()
        };
        assert_eq!(rule.gate(&metrics, &envelope), Gate::Detect);
    }

    // -----------------------------------------------------------------
    // fix() always declines
    // -----------------------------------------------------------------

    #[test]
    fn fix_always_declines() {
        let rule = TriadReductionRule::new();
        let finding = Finding::new(RULE_ID, 0..5, "triad", Tier::Suggest);
        let source = "screws, bolts, and washers.\n";
        let doc = friction_parse::parse(source).expect("valid markdown parses");
        let doc = friction_nlp::segment_document(&doc, &friction_nlp::SrxSegmenter::new())
            .expect("segmentation succeeds");
        let tagger = friction_nlp::NlpruleTagger::new().expect("embedded model loads");
        let envelope = MapEnvelope::new();
        let ctx = RuleContext::new(&doc, &tagger, "blog", &envelope);
        let mut rng = StrategyRng::from_seed(0);
        assert!(rule.fix(&finding, &ctx, &mut rng).is_none());
    }
}
