//! Discourse-filler phrase deletion: removes a closed table of
//! sentence-initial and mid-sentence hedge/transition phrases
//! ("It's worth noting that", "Needless to say,", "At the end of the day,")
//! that carry no propositional content of their own — a sentence keeps
//! exactly the same meaning with or without one, so deleting it is Fix
//! tier by construction.
//!
//! # Matching
//!
//! [`FILLER_PHRASES`] is matched case-insensitively (ASCII case folding —
//! every entry is plain ASCII except the apostrophe in the "It's..."
//! entries, which also matches the Unicode right single quotation mark
//! `’`, the same convention `friction-metrics::lexical` uses for its own
//! marker tables) against a sentence's own text, left to right, requiring
//! a non-alphanumeric boundary on both sides of a match so a phrase never
//! matches as a fragment of a longer word or a longer phrase. At a given
//! start position the *longest* matching entry wins (relevant only if a
//! future entry is added as a substring of another; today's table has no
//! such overlaps, see this module's `no_entry_is_a_prefix_of_another`
//! test).
//!
//! This rule does not attempt `friction-parse`'s markdown-emphasis
//! bridging trick (`friction-metrics::lexical::strip_leading_markup`
//! strips a leading `**`/`__` before checking a sentence-initial marker):
//! like `crate::families::connective::ConnectiveSurgery`, it treats a
//! sentence's own text (which a well-behaved `Segmenter` never prefixes
//! with whitespace) as the whole of what "sentence-initial" means. A
//! bold-wrapped filler phrase (`"**It's worth noting that** ..."`) is
//! left for a future revision.
//!
//! # Fix: deletion, recapitalization, and cleanup
//!
//! Deleting a filler phrase also consumes, in the same patch:
//!
//! - one redundant comma directly after the phrase, if present (a common
//!   LLM tic: `"...worth noting that, the system..."`);
//! - every space between the phrase and whatever follows it (collapses an
//!   accidental double space rather than leaving one behind); and
//! - when the phrase was sentence-initial (its match started exactly at
//!   the sentence's own start) and the next surviving character is
//!   lowercase, that one character, replaced with its uppercase form, so
//!   the sentence still opens with a capital letter.
//!
//! # Idempotence
//!
//! Deletion is idempotent by construction: the phrase's literal text is
//! gone from the output, so a second pass over the same result has
//! nothing left to find (checked directly for every golden fixture by
//! `crate::families::lexical`'s integration tests, and by this module's
//! `fix_is_idempotent_on_synthetic_sentences` test).

use std::ops::Range;

use friction_core::{Finding, MetricVector, Patch, RuleId, Sentence, Tier, span};

use crate::budget::Budget;
use crate::context::{GenreEnvelope, RuleContext};
use crate::rule::{Gate, Rule, RuleFamily};
use crate::strategy::StrategyRng;

/// This rule's stable identifier.
const RULE_ID: RuleId = RuleId::new("lexical.filler_phrase");

/// The [`MetricVector`] field this rule gates on: sentence-initial
/// discourse-marker density, per 1000 word tokens. Most of
/// [`FILLER_PHRASES`]' sentence-initial occurrences already count toward
/// this metric — `friction-metrics::lexical::DISCOURSE_MARKERS` includes
/// `"It's worth noting"` and `"It is important to note"` as
/// word-boundary-terminated prefixes, which several entries here extend
/// with a trailing `"that"` — so this rule and the metric it gates on
/// measure (mostly) the same phenomenon.
const GATED_METRIC: &str = "discourse_marker_density";

/// How much fixing one occurrence is projected to move [`GATED_METRIC`],
/// for [`Budget::from_envelope_excess`].
///
/// The metric is already normalized to a rate per 1000 word tokens, and a
/// [`MetricVector`] carries no raw token count `gate` could use to compute
/// this document's exact per-occurrence effect (by design — see that
/// type's own docs). `1.0`, one point of the metric's own per-1000-token
/// scale, is the natural dimensionless unit given that constraint: close
/// to the real effect for a document near 1000 tokens, an understatement
/// for a longer one — which only ever makes the resulting budget
/// conservative, never excessive, and a conservative budget is exactly
/// what a later round's fresh `gate` call (re-measured against the actual
/// result) tops up.
const PER_FIX_EFFECT: f64 = 1.0;

/// Discourse-filler phrases this rule deletes: sentence-initial or
/// mid-sentence hedge/transition phrases that add no propositional
/// content. Sorted alphabetically (ASCII byte order, checked by this
/// module's `filler_phrases_sorted_and_unique` test); matched
/// case-insensitively (see the module docs).
///
/// Curated from `corpus/MINING.md`'s llm-favored n-grams where a direct
/// filler phrase was present there (`"here's a"`, `"as we"` inform the
/// `"As we can see,"` and `"here's"`-adjacent entries below) plus
/// canonical filler phrases documented across style guides as
/// disproportionately common in LLM output — the same two-source curation
/// `crate::families::connective::CONNECTIVES` and
/// `friction-metrics::lexical::RITUAL_MARKERS` both already use for their
/// own tables. Every entry here is a phrase whose removal changes no
/// proposition the sentence asserts, which is what keeps this a Fix-tier
/// (not Suggest-tier) rule.
const FILLER_PHRASES: &[&str] = &[
    "As we can see,",
    "At the end of the day,",
    "For what it's worth,",
    "Generally speaking,",
    "I should also mention that",
    "I should note that",
    "In fact,",
    "It goes without saying that",
    "It is also worth noting that",
    "It is important to note that",
    "It is important to understand that",
    "It is worth mentioning that",
    "It is worth noting that",
    "It should be noted that",
    "It's also worth noting that",
    "It's important to note that",
    "It's worth mentioning that",
    "It's worth noting that",
    "Needless to say,",
    "Simply put,",
    "That being said,",
    "To put it another way,",
    "To put it simply,",
    "When all is said and done,",
    "With that said,",
];

/// `true` for a character this rule treats as part of a word for its
/// boundary check — matching [`char::is_alphanumeric`] (Unicode-aware, so
/// an accented letter still counts).
fn is_word_char(c: char) -> bool {
    c.is_alphanumeric()
}

/// `true` if `expected` (a [`FILLER_PHRASES`] character) matches `actual`
/// (a sentence-text character): ASCII case-insensitive, with `expected ==
/// '\''` additionally matching the Unicode right single quotation mark
/// `’` in `actual`.
const fn char_matches(expected: char, actual: char) -> bool {
    if expected == '\'' {
        actual == '\'' || actual == '\u{2019}'
    } else {
        expected.eq_ignore_ascii_case(&actual)
    }
}

/// If `phrase` matches `chars` starting at char index `start`, returns the
/// char index just past the match.
fn match_phrase_at(chars: &[(usize, char)], start: usize, phrase: &str) -> Option<usize> {
    let mut i = start;
    for expected in phrase.chars() {
        let &(_, actual) = chars.get(i)?;
        if !char_matches(expected, actual) {
            return None;
        }
        i += 1;
    }
    Some(i)
}

/// Finds every non-overlapping [`FILLER_PHRASES`] occurrence in `text`,
/// left to right: byte ranges relative to `text`'s own start. At a given
/// start position, the longest matching entry wins; a match is only
/// accepted with a non-word character (or start/end of `text`) on both
/// sides.
fn find_filler_matches(text: &str) -> Vec<Range<usize>> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let mut matches = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let before_ok = i == 0 || !is_word_char(chars[i - 1].1);
        let mut best_end: Option<usize> = None;
        if before_ok {
            for phrase in FILLER_PHRASES {
                let Some(end_idx) = match_phrase_at(&chars, i, phrase) else {
                    continue;
                };
                let after_ok = chars.get(end_idx).is_none_or(|&(_, c)| !is_word_char(c));
                if after_ok && best_end.is_none_or(|cur| end_idx > cur) {
                    best_end = Some(end_idx);
                }
            }
        }
        if let Some(end_idx) = best_end {
            let start_byte = chars[i].0;
            let end_byte = chars.get(end_idx).map_or(text.len(), |&(b, _)| b);
            matches.push(start_byte..end_byte);
            i = end_idx;
        } else {
            i += 1;
        }
    }
    matches
}

/// The sentence in `ctx`'s document that fully contains `range`, if any.
fn containing_sentence<'a>(ctx: &RuleContext<'a>, range: &Range<usize>) -> Option<&'a Sentence> {
    ctx.sentences()
        .map(|(_, sentence)| sentence)
        .find(|sentence| span::contains_range(&sentence.range, range))
}

/// Discourse-filler phrase deletion: deletes each [`FILLER_PHRASES`]
/// occurrence, budgeted to bring [`GATED_METRIC`] back into the genre's
/// envelope. See the module docs for matching and fix-up details.
#[derive(Debug, Clone, Copy, Default)]
pub struct FillerPhraseRule;

impl FillerPhraseRule {
    /// Creates the rule.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Rule for FillerPhraseRule {
    fn id(&self) -> RuleId {
        RULE_ID
    }

    fn family(&self) -> RuleFamily {
        RuleFamily::Lexical
    }

    fn gate(&self, metrics: &MetricVector, envelope: &dyn GenreEnvelope) -> Gate {
        let Some(band) = envelope.band(GATED_METRIC) else {
            return Gate::Off;
        };
        let current = metrics.discourse_marker_density;
        // This rule only ever deletes filler phrases, never inserts one,
        // so it has no safe move for a document already inside the band
        // or (unusually) below its floor.
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
        let mut findings = Vec::new();
        for (_, sentence) in ctx.sentences() {
            let Ok(text) = document.text(&sentence.range) else {
                continue;
            };
            for relative in find_filler_matches(text) {
                let start = sentence.range.start + relative.start;
                let end = sentence.range.start + relative.end;
                findings.push(Finding::new(
                    RULE_ID,
                    start..end,
                    "discourse-filler phrase carries no meaning of its own for this genre's \
                     envelope",
                    Tier::Fix,
                ));
            }
        }
        findings
    }

    fn fix(
        &self,
        finding: &Finding,
        ctx: &RuleContext<'_>,
        _strategy_rng: &mut StrategyRng,
    ) -> Option<Patch> {
        let source = ctx.document().source();
        let sentence = containing_sentence(ctx, &finding.range)?;
        let sentence_initial = finding.range.start == sentence.range.start;

        let mut end = finding.range.end;
        // One redundant leftover comma directly after the phrase (see the
        // module docs' "Fix" section).
        if source[end..].starts_with(',') {
            end += 1;
        }
        // Every space separating the phrase from what follows -- collapses
        // any accidental double space rather than leaving one behind.
        while source[end..].starts_with(' ') {
            end += 1;
        }

        let (patch_end, replacement) = if sentence_initial {
            match source[end..].chars().next() {
                Some(c) if c.is_lowercase() => {
                    let upper: String = c.to_uppercase().collect();
                    (end + c.len_utf8(), upper)
                }
                _ => (end, String::new()),
            }
        } else {
            (end, String::new())
        };

        Some(Patch::new(
            finding.range.start..patch_end,
            replacement,
            RULE_ID,
            Tier::Fix,
        ))
    }
}

#[cfg(test)]
mod tests {
    use friction_core::Envelope;
    use friction_nlp::{SrxSegmenter, Tagger};

    use super::*;
    use crate::context::MapEnvelope;

    struct NoopTagger;
    impl Tagger for NoopTagger {
        fn tag(&self, _text: &str, _base_offset: usize) -> Vec<friction_nlp::TaggedToken> {
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
            discourse_marker_density: density,
            ..MetricVector::default()
        }
    }

    fn apply(source: &str, patch: &Patch) -> String {
        let mut applied = source.to_string();
        applied.replace_range(patch.range.clone(), &patch.replacement);
        applied
    }

    // ---------------------------------------------------------------
    // Table hygiene
    // ---------------------------------------------------------------

    #[test]
    fn filler_phrases_sorted_and_unique() {
        assert!(FILLER_PHRASES.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn filler_phrases_has_at_least_twenty_five_entries() {
        assert!(
            FILLER_PHRASES.len() >= 25,
            "expected at least 25 curated filler phrases, has {}",
            FILLER_PHRASES.len()
        );
    }

    /// No entry is a prefix of another — if it were, `find_filler_matches`'
    /// longest-match rule would be load-bearing rather than defensive, and
    /// worth a dedicated regression test; today's table needs no such
    /// case.
    #[test]
    fn no_entry_is_a_prefix_of_another() {
        for (i, a) in FILLER_PHRASES.iter().enumerate() {
            for b in &FILLER_PHRASES[i + 1..] {
                assert!(
                    !b.to_ascii_lowercase().starts_with(&a.to_ascii_lowercase())
                        && !a.to_ascii_lowercase().starts_with(&b.to_ascii_lowercase()),
                    "{a:?} and {b:?} must not prefix one another"
                );
            }
        }
    }

    // ---------------------------------------------------------------
    // gate()
    // ---------------------------------------------------------------

    #[test]
    fn gate_is_off_without_a_band() {
        let rule = FillerPhraseRule::new();
        let envelope = MapEnvelope::new();
        assert_eq!(
            rule.gate(&metrics_with_density(500.0), &envelope),
            Gate::Off
        );
    }

    #[test]
    fn gate_is_off_inside_band() {
        let rule = FillerPhraseRule::new();
        let envelope = MapEnvelope::new().with(GATED_METRIC, Envelope::new(0.0, 10.0));
        assert_eq!(rule.gate(&metrics_with_density(5.0), &envelope), Gate::Off);
    }

    #[test]
    fn gate_is_off_below_band_floor() {
        let rule = FillerPhraseRule::new();
        let envelope = MapEnvelope::new().with(GATED_METRIC, Envelope::new(5.0, 10.0));
        assert_eq!(rule.gate(&metrics_with_density(1.0), &envelope), Gate::Off);
    }

    /// Hand-computed: current 13.0, band hi 10.0, `PER_FIX_EFFECT` 1.0 ->
    /// excess 3.0 -> budget 3.
    #[test]
    fn gate_above_band_computes_hand_verified_budget() {
        let rule = FillerPhraseRule::new();
        let envelope = MapEnvelope::new().with(GATED_METRIC, Envelope::new(0.0, 10.0));
        assert_eq!(
            rule.gate(&metrics_with_density(13.0), &envelope),
            Gate::Fix {
                budget: Budget::new(3)
            }
        );
    }

    // ---------------------------------------------------------------
    // scan()
    // ---------------------------------------------------------------

    #[test]
    fn scan_finds_sentence_initial_and_mid_sentence_occurrences() {
        let source = "It is worth noting that the plan changed. \
                       The team shipped it, needless to say, on time.";
        let doc = document(source);
        let envelope = MapEnvelope::new();
        let ctx = RuleContext::new(&doc, &NoopTagger, "blog", &envelope);
        let rule = FillerPhraseRule::new();

        let findings = rule.scan(&ctx);
        assert_eq!(findings.len(), 2);
        assert_eq!(
            &source[findings[0].range.clone()],
            "It is worth noting that"
        );
        assert_eq!(&source[findings[1].range.clone()], "needless to say,");
    }

    #[test]
    fn scan_requires_a_word_boundary() {
        let source = "Itisworthnotingthat is one long token here.";
        let doc = document(source);
        let envelope = MapEnvelope::new();
        let ctx = RuleContext::new(&doc, &NoopTagger, "blog", &envelope);
        assert!(FillerPhraseRule::new().scan(&ctx).is_empty());
    }

    // ---------------------------------------------------------------
    // fix()
    // ---------------------------------------------------------------

    #[test]
    fn fix_deletes_sentence_initial_phrase_and_recapitalizes() {
        let source = "It is worth noting that performance improved.";
        let doc = document(source);
        let envelope = MapEnvelope::new();
        let ctx = RuleContext::new(&doc, &NoopTagger, "blog", &envelope);
        let rule = FillerPhraseRule::new();
        let finding = &rule.scan(&ctx)[0];

        let mut rng = StrategyRng::from_seed(0);
        let patch = rule
            .fix(finding, &ctx, &mut rng)
            .expect("finding has a fix");
        assert_eq!(apply(source, &patch), "Performance improved.");
    }

    #[test]
    fn fix_deletes_mid_sentence_phrase_without_leaving_double_space() {
        let source = "The team shipped it, needless to say, on time.";
        let doc = document(source);
        let envelope = MapEnvelope::new();
        let ctx = RuleContext::new(&doc, &NoopTagger, "blog", &envelope);
        let rule = FillerPhraseRule::new();
        let finding = &rule.scan(&ctx)[0];

        let mut rng = StrategyRng::from_seed(0);
        let patch = rule
            .fix(finding, &ctx, &mut rng)
            .expect("finding has a fix");
        assert_eq!(apply(source, &patch), "The team shipped it, on time.");
    }

    #[test]
    fn fix_strips_a_redundant_leftover_comma() {
        let source = "It's worth noting that, performance improved.";
        let doc = document(source);
        let envelope = MapEnvelope::new();
        let ctx = RuleContext::new(&doc, &NoopTagger, "blog", &envelope);
        let rule = FillerPhraseRule::new();
        let finding = &rule.scan(&ctx)[0];

        let mut rng = StrategyRng::from_seed(0);
        let patch = rule
            .fix(finding, &ctx, &mut rng)
            .expect("finding has a fix");
        assert_eq!(apply(source, &patch), "Performance improved.");
    }

    #[test]
    fn fix_matches_curly_apostrophe_variant() {
        let source = "It\u{2019}s worth noting that performance improved.";
        let doc = document(source);
        let envelope = MapEnvelope::new();
        let ctx = RuleContext::new(&doc, &NoopTagger, "blog", &envelope);
        let rule = FillerPhraseRule::new();
        let finding = &rule.scan(&ctx)[0];

        let mut rng = StrategyRng::from_seed(0);
        let patch = rule
            .fix(finding, &ctx, &mut rng)
            .expect("finding has a fix");
        assert_eq!(apply(source, &patch), "Performance improved.");
    }

    #[test]
    fn fix_does_not_recapitalize_mid_sentence_deletion() {
        let source = "The plan works, that being said, risks remain.";
        let doc = document(source);
        let envelope = MapEnvelope::new();
        let ctx = RuleContext::new(&doc, &NoopTagger, "blog", &envelope);
        let rule = FillerPhraseRule::new();
        let finding = &rule.scan(&ctx)[0];

        let mut rng = StrategyRng::from_seed(0);
        let patch = rule
            .fix(finding, &ctx, &mut rng)
            .expect("finding has a fix");
        assert_eq!(apply(source, &patch), "The plan works, risks remain.");
    }

    // ---------------------------------------------------------------
    // Idempotence and determinism
    // ---------------------------------------------------------------

    /// Fixing every finding once, applying the patches, then scanning the
    /// result again finds nothing left: the fixed output is a fixed point.
    #[test]
    fn fix_is_idempotent_on_synthetic_sentences() {
        let sources = [
            "It is worth noting that performance improved.",
            "The team shipped it, needless to say, on time.",
            "At the end of the day, this is what matters most.",
            "Simply put, the migration succeeded, and generally speaking, users are happy.",
        ];
        for source in sources {
            let doc = document(source);
            let envelope = MapEnvelope::new();
            let ctx = RuleContext::new(&doc, &NoopTagger, "blog", &envelope);
            let rule = FillerPhraseRule::new();

            let mut patches: Vec<Patch> = rule
                .scan(&ctx)
                .iter()
                .filter_map(|finding| {
                    let mut rng = StrategyRng::seeded(source.as_bytes(), rule.id());
                    rule.fix(finding, &ctx, &mut rng)
                })
                .collect();
            patches.sort_by_key(|p| p.range.start);

            let mut fixed = source.to_string();
            for patch in patches.iter().rev() {
                fixed.replace_range(patch.range.clone(), &patch.replacement);
            }

            let fixed_doc = document(&fixed);
            let fixed_ctx = RuleContext::new(&fixed_doc, &NoopTagger, "blog", &envelope);
            assert!(
                rule.scan(&fixed_ctx).is_empty(),
                "expected no findings left after fixing {source:?}, got fixed text {fixed:?}"
            );
        }
    }

    #[test]
    fn fixing_the_same_source_twice_is_byte_identical() {
        let source = "It is worth noting that performance improved, and needless to say, \
                       costs dropped.";
        let run = || {
            let doc = document(source);
            let envelope = MapEnvelope::new();
            let ctx = RuleContext::new(&doc, &NoopTagger, "blog", &envelope);
            let rule = FillerPhraseRule::new();
            rule.scan(&ctx)
                .iter()
                .filter_map(|finding| {
                    let mut rng = StrategyRng::seeded(source.as_bytes(), rule.id());
                    rule.fix(finding, &ctx, &mut rng)
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(run(), run());
    }
}
