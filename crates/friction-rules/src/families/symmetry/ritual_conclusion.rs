//! Ritual-conclusion deletion: a document's *final* paragraph opening with
//! a stock transition phrase (`"Overall, ..."`, `"To summarize, ..."`,
//! `"In today's fast-paced world, ..."`) — the ritual bookend LLM prose
//! reaches for far more often than human writers do.
//!
//! # Mixed tier, per finding
//!
//! At most one finding is ever possible per document (the document's final
//! paragraph, if it opens with a marker at all — see [`RitualConclusionRule::scan`]).
//! That finding is [`friction_core::Tier::Fix`] — and
//! [`RitualConclusionRule::fix`] deletes the whole paragraph, including its
//! own trailing blank line — only when [`paragraph_is_safe_to_delete`]'s
//! conservative heuristic confirms doing so drops no proposition the rest
//! of the document did not already state. That heuristic has two parts,
//! both required:
//!
//! - No new noun *or adjective* lemma: every content lemma the flagged
//!   paragraph mentions (outside the marker phrase itself) must already
//!   appear somewhere earlier in the document. Nouns alone are not enough
//!   — this tagger's own disambiguator sometimes tags a plain past-tense
//!   verb as an adjective when it sees a short, marker-led sentence in
//!   isolation (`"the update broke the server"` tags `"broke"` as `JJ`,
//!   not `VBD`; `"the vendor never honored the refunds"` tags `"honored"`
//!   the same way — see [`paragraph_adds_no_new_content`]'s own docs), so
//!   a noun-only check would miss exactly the case where a final paragraph
//!   swaps in a *different* verb to state something the rest of the
//!   document never claimed. Restricting the broadened check to noun and
//!   adjective tags (not the full open class) keeps ordinary stylistic
//!   restatement — a paraphrase that reuses the same nouns but a
//!   *different, harmless* verb, e.g. `"the kit and its screws work well
//!   together"` restating `"the kit includes screws"` — from being
//!   penalized just for using new phrasing; see
//!   [`paragraph_adds_no_new_content`]'s own docs for why verbs are
//!   deliberately excluded from this comparison.
//! - No negation cue ([`paragraph_has_negation`]): a paragraph containing
//!   `"not"`, `"never"`, `"no"`, `"n't"`, or similar unconditionally blocks
//!   Fix tier, regardless of noun/adjective overlap — negation reverses a
//!   claim's truth value outright, which is never a stylistic restatement.
//!
//! Neither check is a claim of semantic completeness — a syntactic scan
//! fundamentally cannot verify a paragraph's claim is truly redundant, only
//! rule out the cheap, checkable signals of it *not* being. Whenever either
//! check fails, the finding is [`friction_core::Tier::Suggest`] and carries
//! no patch — the paragraph might be doing real summarizing (or reversing)
//! work, which this rule cannot safely judge on its own, so it is surfaced
//! as a diagnostic instead of auto-deleted.
//!
//! # Mirrored marker list
//!
//! [`RITUAL_MARKERS`] is a byte-identical copy of `friction-metrics::
//! lexical::RITUAL_MARKERS` (private to that crate — see `families::
//! symmetry`'s own module docs for why every submodule here mirrors rather
//! than imports), along with the same leading-markup-stripping and
//! word-boundary matching rules that crate's `ritual_marker_rate` uses (so
//! a bold- or italic-wrapped marker like `"**Overall:**"` is still
//! recognized — see [`strip_leading_markup`]). This module's
//! `ritual_markers_are_all_recognized_by_the_public_rate_metric` test
//! cross-checks every entry against `friction_metrics::ritual_marker_rate`'s
//! public, rate-returning function, so the two lists cannot silently drift
//! apart without a test failing.
//!
//! Unlike `friction_metrics::ritual_marker_rate` itself (which flags a
//! paragraph whose *first or last* sentence opens with a marker), this
//! rule only ever looks at whether the final paragraph's *first* sentence
//! opens with one — "final paragraph opening with a ritual conclusion
//! marker" is a narrower, more specific pattern than the broader density
//! metric it gates on, the same relationship `families::connective::
//! ConnectiveSurgery` has with its own (broader) gated metric.

use std::collections::HashSet;

use friction_core::{
    Block, BlockKind, Document, Finding, MetricVector, Patch, ProseUnit, RuleId, Tier,
};

use crate::budget::Budget;
use crate::context::{GenreEnvelope, RuleContext};
use crate::rule::{Gate, Rule, RuleFamily};
use crate::strategy::StrategyRng;

/// This rule's stable identifier.
const RULE_ID: RuleId = RuleId::new("symmetry.ritual_conclusion");

/// The [`MetricVector`] field this rule gates on.
const GATED_METRIC: &str = "ritual_marker_rate";

/// Mirrors `friction_metrics::lexical::RITUAL_MARKERS` exactly (sorted,
/// ASCII byte order — checked by this module's own
/// `ritual_markers_sorted_and_unique` test, the same invariant that
/// crate's own copy checks of itself).
const RITUAL_MARKERS: [&str; 15] = [
    "As we can see",
    "At the end of the day",
    "In closing",
    "In conclusion",
    "In summary",
    "In today's digital age",
    "In today's fast-paced world",
    "In today's world",
    "Looking ahead",
    "Overall",
    "To conclude",
    "To sum up",
    "To summarize",
    "To wrap up",
    "Ultimately",
];

/// Strips leading whitespace and leading markdown emphasis/strong
/// delimiters (`*`/`_`) from `text`, repeating both strips until a pass
/// removes nothing further. Mirrors `friction_metrics::lexical::
/// strip_leading_markup` exactly — see that function's own docs for why
/// this is needed at all (a sentence's source-sliced text keeps a bold- or
/// italic-wrapped marker's literal delimiter prefix).
fn strip_leading_markup(text: &str) -> &str {
    let mut current = text;
    loop {
        let next = current.trim_start().trim_start_matches(['*', '_']);
        if next == current {
            return next;
        }
        current = next;
    }
}

/// If `text` begins with `marker`, case-insensitively (a `'` in `marker`
/// also matching the Unicode right single quotation mark `’`), immediately
/// followed by a non-alphanumeric character or the end of `text`, returns
/// the exact byte length of `text` consumed by the match — which is not
/// always `marker.len()`, since a smart-quote apostrophe (`’`, 3 bytes) can
/// stand in for `marker`'s plain ASCII `'` (1 byte). Mirrors
/// `friction_metrics::lexical::starts_with_marker`'s matching rule, adapted
/// to report the consumed length a [`friction_core::Finding`]'s span needs
/// instead of only a `bool`.
fn matched_marker_len(text: &str, marker: &str) -> Option<usize> {
    let mut chars = text.char_indices();
    let mut consumed = 0usize;
    for expected in marker.chars() {
        let (idx, actual) = chars.next()?;
        let matches = if expected == '\'' {
            actual == '\'' || actual == '\u{2019}'
        } else {
            actual.eq_ignore_ascii_case(&expected)
        };
        if !matches {
            return None;
        }
        consumed = idx + actual.len_utf8();
    }
    let after = &text[consumed..];
    if after.chars().next().is_some_and(char::is_alphanumeric) {
        return None;
    }
    Some(consumed)
}

/// If `text` (after stripping leading markup) starts with one of
/// [`RITUAL_MARKERS`], returns `(start, end, marker)`: the byte range in
/// `text` (not the stripped copy) the match occupies, and which marker
/// matched.
fn find_ritual_marker(text: &str) -> Option<(usize, usize, &'static str)> {
    let stripped = strip_leading_markup(text);
    let prefix_len = text.len() - stripped.len();
    RITUAL_MARKERS.iter().find_map(|&marker| {
        matched_marker_len(stripped, marker).map(|len| (prefix_len, prefix_len + len, marker))
    })
}

/// `true` if `pos` (a tagger's Penn-Treebank-style tag) is a noun tag
/// (`NN`, `NNS`, `NNP`, `NNPS`) or an adjective tag (`JJ`, `JJR`, `JJS`).
///
/// Nouns alone are not a reliable enough signal — see
/// [`paragraph_adds_no_new_content`]'s own docs for why adjective tags are
/// deliberately included here (this tagger, run on a single short
/// marker-led sentence in isolation, sometimes tags a plain past-tense verb
/// as an adjective instead), while verb tags are deliberately still
/// excluded (to avoid penalizing ordinary paraphrase, which routinely
/// reaches for a different, harmless verb).
fn is_noun_or_adjective(pos: &str) -> bool {
    pos.starts_with("NN") || pos.starts_with("JJ")
}

/// Lemmas ([`friction_nlp::TaggedToken::lemma`]) of every noun- or
/// adjective-tagged token found across `unit`'s sentences, tagged via
/// `ctx`. Tokens whose start offset falls inside `exclude` (the flagged
/// paragraph's own ritual-marker span, when comparing that paragraph — see
/// [`paragraph_adds_no_new_content`]) are skipped: the marker phrase itself
/// (`"Overall"`, `"In conclusion"`, ...) is not part of the paragraph's own
/// claim, and several markers tag as an adjective or a noun in their own
/// right (`"Overall"` as `JJ`, `"conclusion"` in `"In conclusion"` as
/// `NN`), so counting it would flag every marker-led paragraph as "new
/// content" regardless of what it actually says.
fn content_lemmas(
    ctx: &RuleContext<'_>,
    unit: &ProseUnit,
    exclude: Option<&std::ops::Range<usize>>,
) -> HashSet<Box<str>> {
    let mut lemmas = HashSet::new();
    for sentence in &unit.sentences {
        for token in ctx.tag_sentence(sentence) {
            if exclude.is_some_and(|marker| marker.contains(&token.token.range.start)) {
                continue;
            }
            if is_noun_or_adjective(token.pos.as_str()) {
                lemmas.insert(token.lemma);
            }
        }
    }
    lemmas
}

/// `true` if every noun/adjective lemma [`content_lemmas`] finds in `ctx`'s
/// document's prose unit at `flagged_index` (outside `marker_range`)
/// already appears in some *other* prose unit of the same document — the
/// conservative "adds no new content" heuristic this rule's Fix/Suggest
/// split is built on. A paragraph with no content lemmas of its own
/// trivially passes (there is nothing it could be the sole mention of).
///
/// This is necessary but not sufficient for Fix tier — see
/// [`paragraph_has_negation`] for the other half of the check, and the
/// module docs' "Mixed tier, per finding" section for why neither claims to
/// be a complete semantic check.
fn paragraph_adds_no_new_content(
    ctx: &RuleContext<'_>,
    flagged_index: usize,
    marker_range: &std::ops::Range<usize>,
) -> bool {
    let document = ctx.document();
    let Some(flagged_unit) = document.prose().get(flagged_index) else {
        return false;
    };
    let flagged_lemmas = content_lemmas(ctx, flagged_unit, Some(marker_range));
    if flagged_lemmas.is_empty() {
        return true;
    }
    let mut earlier_lemmas: HashSet<Box<str>> = HashSet::new();
    for (index, unit) in document.prose().iter().enumerate() {
        if index == flagged_index {
            continue;
        }
        earlier_lemmas.extend(content_lemmas(ctx, unit, None));
    }
    flagged_lemmas.is_subset(&earlier_lemmas)
}

/// Lemmas this rule treats as an explicit negation cue — matched
/// case-sensitively against a token's own tagger-assigned lemma (already
/// lowercase for these words in practice; `"n't"` and `"not"` both
/// lemmatize to `"not"`, and a bare `"No"` lemmatizes to `"no"`).
const NEGATION_LEMMAS: [&str; 10] = [
    "cannot", "neither", "never", "no", "nobody", "none", "nor", "not", "nothing", "nowhere",
];

/// `true` if any sentence in `unit` contains a token whose lemma is one of
/// [`NEGATION_LEMMAS`]. Unlike [`paragraph_adds_no_new_content`], this is
/// not a subset comparison against the rest of the document — a negation
/// cue unconditionally blocks Fix tier, because it can reverse the truth
/// value of a claim built from words the document already used elsewhere
/// (`"the vendor guaranteed refunds"` -> `"the vendor never honored the
/// refunds"` reuses no new noun, yet asserts the opposite of what a reader
/// would otherwise conclude), which [`paragraph_adds_no_new_content`]'s
/// lemma-overlap check cannot by itself detect.
fn paragraph_has_negation(ctx: &RuleContext<'_>, unit: &ProseUnit) -> bool {
    unit.sentences.iter().any(|sentence| {
        ctx.tag_sentence(sentence)
            .iter()
            .any(|token| NEGATION_LEMMAS.contains(&&*token.lemma))
    })
}

/// `true` if [`RitualConclusionRule`] can verify that deleting the flagged
/// final paragraph (`document.prose()[flagged_index]`) drops no proposition
/// the rest of the document did not already state — both
/// [`paragraph_adds_no_new_content`] and the absence of
/// [`paragraph_has_negation`] must hold. See the module docs' "Mixed tier,
/// per finding" section for the rationale behind each half.
fn paragraph_is_safe_to_delete(
    ctx: &RuleContext<'_>,
    flagged_index: usize,
    flagged_unit: &ProseUnit,
    marker_range: &std::ops::Range<usize>,
) -> bool {
    paragraph_adds_no_new_content(ctx, flagged_index, marker_range)
        && !paragraph_has_negation(ctx, flagged_unit)
}

/// The document's final paragraph, if — and only if — it is a genuine
/// trailing [`BlockKind::Paragraph`] (not a list item, table cell, or
/// heading) with nothing but whitespace after it in the source: the
/// conservative shape [`RitualConclusionRule`] knows how to delete safely.
/// Returns `(prose_index, unit, block)`.
fn final_paragraph(document: &Document) -> Option<(usize, &ProseUnit, &Block)> {
    let prose = document.prose();
    let index = prose.len().checked_sub(1)?;
    let unit = &prose[index];
    let block = document.blocks().get(unit.block)?;
    if !matches!(block.kind, BlockKind::Paragraph) {
        return None;
    }
    let source = document.source();
    if !source[block.range.end..].chars().all(char::is_whitespace) {
        return None;
    }
    Some((index, unit, block))
}

/// Deletes (Fix tier) or suggests deleting (Suggest tier) a document's
/// final paragraph when it opens with a ritual conclusion marker. See the
/// module docs for the exact Fix/Suggest split.
#[derive(Debug, Clone, Copy, Default)]
pub struct RitualConclusionRule;

impl RitualConclusionRule {
    /// Creates the rule.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Rule for RitualConclusionRule {
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
        if metrics.ritual_marker_rate <= band.hi {
            return Gate::Off;
        }
        // At most one finding is ever possible per document (see `scan`),
        // so a budget of 1 is always sufficient — no envelope-excess
        // scaling needed the way `families::connective::ConnectiveSurgery`
        // or `families::contraction::ContractionRule` (which can each
        // produce many findings per document) require.
        Gate::Fix {
            budget: Budget::new(1),
        }
    }

    fn scan(&self, ctx: &RuleContext<'_>) -> Vec<Finding> {
        let document = ctx.document();
        let Some((index, unit, _block)) = final_paragraph(document) else {
            return Vec::new();
        };
        let Some(first_sentence) = unit.sentences.first() else {
            return Vec::new();
        };
        let Ok(text) = document.text(&first_sentence.range) else {
            return Vec::new();
        };
        let Some((local_start, local_end, marker)) = find_ritual_marker(text) else {
            return Vec::new();
        };
        let start = first_sentence.range.start + local_start;
        let end = first_sentence.range.start + local_end;
        let marker_range = start..end;

        let tier = if paragraph_is_safe_to_delete(ctx, index, unit, &marker_range) {
            Tier::Fix
        } else {
            Tier::Suggest
        };
        vec![Finding::new(
            RULE_ID,
            start..end,
            format!(
                "the document's final paragraph opens with the ritual conclusion marker {marker:?}"
            ),
            tier,
        )]
    }

    fn fix(
        &self,
        finding: &Finding,
        ctx: &RuleContext<'_>,
        _strategy_rng: &mut StrategyRng,
    ) -> Option<Patch> {
        let document = ctx.document();
        let (_index, _unit, block) = final_paragraph(document)?;

        // Recompute rather than trust a possibly-stale `finding` (the same
        // defense-in-depth other rules in this workspace apply — e.g.
        // `families::contraction::ContractionRule::fix` re-derives its
        // exact target count instead of trusting `gate`'s own estimate):
        // confirm this exact finding is still the current, Fix-tier
        // result of a fresh `scan` before proposing a patch for it.
        let current = self.scan(ctx);
        let still_fix_tier = current
            .iter()
            .any(|f| f.range == finding.range && f.tier == Tier::Fix);
        if !still_fix_tier {
            return None;
        }

        Some(Patch::new(
            block.range.start..document.source().len(),
            "",
            RULE_ID,
            Tier::Fix,
        ))
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

    fn metrics_with_rate(rate: f64) -> MetricVector {
        MetricVector {
            ritual_marker_rate: rate,
            ..MetricVector::default()
        }
    }

    fn permissive_envelope() -> MapEnvelope {
        MapEnvelope::new().with(GATED_METRIC, Envelope::new(0.0, 0.0))
    }

    // -----------------------------------------------------------------
    // Mirrored marker list / matching parity with friction_metrics
    // -----------------------------------------------------------------

    #[test]
    fn ritual_markers_sorted_and_unique() {
        assert!(RITUAL_MARKERS.windows(2).all(|w| w[0] < w[1]));
    }

    /// Every entry in this module's mirrored [`RITUAL_MARKERS`] is also
    /// recognized by `friction_metrics::ritual_marker_rate`'s public
    /// function — the cross-crate consistency check that keeps the two
    /// lists from silently drifting apart.
    #[test]
    fn ritual_markers_are_all_recognized_by_the_public_rate_metric() {
        for marker in RITUAL_MARKERS {
            let source = format!("{marker}, it works out fine.\n");
            let doc = document(&source);
            let rate = friction_metrics::ritual_marker_rate(&doc);
            assert!(
                (rate - 1.0).abs() < f64::EPSILON,
                "marker {marker:?} was not recognized by friction_metrics::ritual_marker_rate (rate {rate})"
            );
        }
    }

    #[test]
    fn find_ritual_marker_matches_bold_wrapped_marker() {
        let (start, end, marker) = find_ritual_marker("**Overall:** it works.").unwrap();
        assert_eq!(marker, "Overall");
        assert_eq!(&"**Overall:** it works."[start..end], "Overall");
    }

    #[test]
    fn find_ritual_marker_matches_smart_quote_apostrophe() {
        let text = "In today\u{2019}s digital age, everything changes.";
        let (start, end, marker) = find_ritual_marker(text).unwrap();
        assert_eq!(marker, "In today's digital age");
        assert_eq!(&text[start..end], "In today\u{2019}s digital age");
    }

    #[test]
    fn find_ritual_marker_none_for_plain_prose() {
        assert!(find_ritual_marker("It works out fine.").is_none());
    }

    // -----------------------------------------------------------------
    // final_paragraph
    // -----------------------------------------------------------------

    #[test]
    fn final_paragraph_accepts_a_trailing_plain_paragraph() {
        let source = "First paragraph text.\n\nOverall, it works.\n";
        let doc = document(source);
        assert!(final_paragraph(&doc).is_some());
    }

    #[test]
    fn final_paragraph_declines_when_the_document_ends_in_a_list() {
        let source = "Overall, it works.\n\n- one\n- two\n";
        let doc = document(source);
        assert!(final_paragraph(&doc).is_none());
    }

    // -----------------------------------------------------------------
    // gate()
    // -----------------------------------------------------------------

    #[test]
    fn gate_is_off_without_a_band() {
        let rule = RitualConclusionRule::new();
        let envelope = MapEnvelope::new();
        assert_eq!(rule.gate(&metrics_with_rate(1.0), &envelope), Gate::Off);
    }

    #[test]
    fn gate_is_off_inside_band() {
        let rule = RitualConclusionRule::new();
        let envelope = MapEnvelope::new().with(GATED_METRIC, Envelope::new(0.0, 0.5));
        assert_eq!(rule.gate(&metrics_with_rate(0.2), &envelope), Gate::Off);
    }

    #[test]
    fn gate_above_band_returns_budget_of_one() {
        let rule = RitualConclusionRule::new();
        let envelope = MapEnvelope::new().with(GATED_METRIC, Envelope::new(0.0, 0.1));
        assert_eq!(
            rule.gate(&metrics_with_rate(0.5), &envelope),
            Gate::Fix {
                budget: Budget::new(1)
            }
        );
    }

    // -----------------------------------------------------------------
    // scan(): Fix vs. Suggest branch
    // -----------------------------------------------------------------

    #[test]
    fn scan_is_fix_tier_when_the_paragraph_adds_no_new_content() {
        // Every noun/adjective the final paragraph mentions ("kit",
        // "screws") already appeared earlier, and it contains no negation.
        let source = "The kit includes screws.\n\n\
            Overall, the kit still includes the screws.\n";
        let doc = document(source);
        let envelope = permissive_envelope();
        let tagger = friction_nlp::NlpruleTagger::new().expect("embedded model loads");
        let ctx = RuleContext::new(&doc, &tagger, "blog", &envelope);
        let findings = RitualConclusionRule::new().scan(&ctx);
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].tier, Tier::Fix);
    }

    /// Regression test for the finding that this rule's noun-only heuristic
    /// missed a final paragraph that *reverses* an earlier claim while
    /// reusing only its nouns: `paragraph_has_negation` blocks Fix tier
    /// unconditionally whenever the flagged paragraph contains a negation
    /// cue, regardless of how much noun/adjective overlap it has.
    #[test]
    fn scan_is_suggest_tier_when_the_paragraph_negates_an_earlier_claim() {
        let source = "The vendor guaranteed full refunds.\n\n\
            Ultimately, the vendor never honored the refunds.\n";
        let doc = document(source);
        let envelope = permissive_envelope();
        let tagger = friction_nlp::NlpruleTagger::new().expect("embedded model loads");
        let ctx = RuleContext::new(&doc, &tagger, "blog", &envelope);
        let findings = RitualConclusionRule::new().scan(&ctx);
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].tier, Tier::Suggest);
    }

    /// Regression test for the finding's other example: a final paragraph
    /// that reverses an earlier claim through a *different verb*, with no
    /// negation word at all and no noun this rule's tagger has not already
    /// seen. This rule's tagger tags the new verb ("broke") as an adjective
    /// when it sees this short, marker-led sentence in isolation, which is
    /// exactly why the noun/adjective check (not noun-only) is needed for
    /// this case to come out Suggest tier rather than Fix.
    #[test]
    fn scan_is_suggest_tier_when_a_mistagged_new_verb_reverses_an_earlier_claim() {
        let source = "The team shipped the update to the server.\n\n\
            Overall, the update broke the server.\n";
        let doc = document(source);
        let envelope = permissive_envelope();
        let tagger = friction_nlp::NlpruleTagger::new().expect("embedded model loads");
        let ctx = RuleContext::new(&doc, &tagger, "blog", &envelope);
        let findings = RitualConclusionRule::new().scan(&ctx);
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].tier, Tier::Suggest);
    }

    #[test]
    fn scan_is_suggest_tier_when_the_paragraph_adds_a_new_noun() {
        // "roadmap" is a noun that appears only in the final paragraph.
        let source = "The kit includes screws.\n\n\
            Overall, check the roadmap for what comes next.\n";
        let doc = document(source);
        let envelope = permissive_envelope();
        let tagger = friction_nlp::NlpruleTagger::new().expect("embedded model loads");
        let ctx = RuleContext::new(&doc, &tagger, "blog", &envelope);
        let findings = RitualConclusionRule::new().scan(&ctx);
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].tier, Tier::Suggest);
    }

    #[test]
    fn scan_finds_nothing_when_the_final_paragraph_has_no_marker() {
        let source = "The kit includes screws.\n\nIt works well.\n";
        let doc = document(source);
        let envelope = permissive_envelope();
        let tagger = friction_nlp::NlpruleTagger::new().expect("embedded model loads");
        let ctx = RuleContext::new(&doc, &tagger, "blog", &envelope);
        assert!(RitualConclusionRule::new().scan(&ctx).is_empty());
    }

    #[test]
    fn scan_finds_nothing_when_only_a_non_final_paragraph_has_a_marker() {
        let source = "Overall, it works.\n\nA plain closing paragraph.\n";
        let doc = document(source);
        let envelope = permissive_envelope();
        let tagger = friction_nlp::NlpruleTagger::new().expect("embedded model loads");
        let ctx = RuleContext::new(&doc, &tagger, "blog", &envelope);
        assert!(RitualConclusionRule::new().scan(&ctx).is_empty());
    }

    // -----------------------------------------------------------------
    // fix()
    // -----------------------------------------------------------------

    #[test]
    fn fix_deletes_the_whole_paragraph_including_its_trailing_blank_line() {
        let source = "The kit includes screws.\n\n\
            Overall, the kit still includes the screws.\n";
        let doc = document(source);
        let envelope = permissive_envelope();
        let tagger = friction_nlp::NlpruleTagger::new().expect("embedded model loads");
        let ctx = RuleContext::new(&doc, &tagger, "blog", &envelope);
        let rule = RitualConclusionRule::new();
        let finding = &rule.scan(&ctx)[0];
        let mut rng = StrategyRng::from_seed(0);
        let patch = rule.fix(finding, &ctx, &mut rng).expect("expected a patch");

        let mut applied = source.to_string();
        applied.replace_range(patch.range, &patch.replacement);
        assert_eq!(applied, "The kit includes screws.\n\n");
    }

    #[test]
    fn fix_declines_the_suggest_tier_branch() {
        let source = "The kit includes screws.\n\n\
            Overall, check the roadmap for what comes next.\n";
        let doc = document(source);
        let envelope = permissive_envelope();
        let tagger = friction_nlp::NlpruleTagger::new().expect("embedded model loads");
        let ctx = RuleContext::new(&doc, &tagger, "blog", &envelope);
        let rule = RitualConclusionRule::new();
        let finding = &rule.scan(&ctx)[0];
        assert_eq!(finding.tier, Tier::Suggest);
        let mut rng = StrategyRng::from_seed(0);
        assert!(rule.fix(finding, &ctx, &mut rng).is_none());
    }

    // -----------------------------------------------------------------
    // Idempotence
    // -----------------------------------------------------------------

    #[test]
    fn fixing_a_document_is_idempotent() {
        let source = "The kit includes screws.\n\n\
            Overall, the kit still includes the screws.\n";
        let doc = document(source);
        let envelope = permissive_envelope();
        let tagger = friction_nlp::NlpruleTagger::new().expect("embedded model loads");
        let rule = RitualConclusionRule::new();

        let patch = {
            let ctx = RuleContext::new(&doc, &tagger, "blog", &envelope);
            let finding = &rule.scan(&ctx)[0];
            let mut rng = StrategyRng::from_seed(0);
            rule.fix(finding, &ctx, &mut rng).expect("expected a patch")
        };
        let mut fixed = source.to_string();
        fixed.replace_range(patch.range, &patch.replacement);

        let fixed_doc = document(&fixed);
        let ctx_after = RuleContext::new(&fixed_doc, &tagger, "blog", &envelope);
        assert!(rule.scan(&ctx_after).is_empty());
    }
}
