//! The seed tell-span inventory (RITUAL/SUBS/SPANS, transcribed verbatim
//! from `ref_engine.py`) plus the derivational-pivot family from
//! [`crate::pivot`], unified into one scan.
//!
//! [`tell_span_hits`] is the single entry point every other module in this
//! crate that needs "how machine-register is this text" goes through:
//! [`crate::score`]'s primary tier counts its length, and
//! [`crate::closure`] reads each hit's `licensed_tokens` to build its
//! per-input pack vocabulary.

use std::sync::LazyLock;

use friction_nlp::Tagger;
use regex::Regex;

use crate::clean::{clean, split_sentences, tokenize};
use crate::pivot::{self, PivotOutcome};

/// Transcribed verbatim from `ref_engine.py::RITUAL`, case-insensitive.
static RITUAL: LazyLock<Vec<(&'static str, Regex)>> = LazyLock::new(|| {
    build(&[
        (
            "ritual.if_you_have_questions",
            r"(?i)if you have any questions.{0,60}(reach out|contact|let us know)",
        ),
        ("ritual.congratulations_open", r"(?i)^congratulations"),
        ("ritual.happy_verbing_open", r"(?i)^happy \w+ing"),
        ("ritual.we_hope", r"(?i)we hope (this|you)"),
    ])
});

/// Transcribed verbatim from `ref_engine.py::SUBS`, case-insensitive.
static SUBS: LazyLock<Vec<(&'static str, Regex, &'static str)>> = LazyLock::new(|| {
    [
        (
            "sub.this_guide_will_walk_you_through",
            r"(?i)\bthis guide will walk you through\b",
            "this guide covers",
        ),
        (
            "sub.will_walk_you_through",
            r"(?i)\bwill walk you through\b",
            "covers",
        ),
        ("sub.in_order_to", r"(?i)\bin order to\b", "to"),
        ("sub.prior_to", r"(?i)\bprior to\b", "before"),
        ("sub.utilize", r"(?i)\butilizes?\b", "uses"),
        ("sub.leverage", r"(?i)\bleverages?\b", "uses"),
    ]
    .into_iter()
    .map(|(id, pattern, replacement)| {
        (
            id,
            Regex::new(pattern).unwrap_or_else(|e| panic!("SUBS pattern {id} must compile: {e}")),
            replacement,
        )
    })
    .collect()
});

/// Transcribed verbatim from `ref_engine.py::SPANS`, case-insensitive.
static SPANS: LazyLock<Vec<(&'static str, Regex)>> = LazyLock::new(|| {
    build(&[
        (
            "span.it_is_important_to_note_that_leading",
            r"(?i)^it is important to note that\s+",
        ),
        (
            "span.it_is_important_to_note_that_mid",
            r"(?i)\bit is important to note that\s+",
        ),
        (
            "span.its_worth_noting_that_leading",
            r"(?i)^it'?s worth noting that\s+",
        ),
        (
            "span.by_following_these_steps_leading",
            r"(?i)^by following these steps,?\s+",
        ),
        ("span.quickly_and_easily", r"(?i)\bquickly and easily\s+"),
        ("span.simply", r"(?i)\bsimply\s+"),
        (
            "span.to_suit_your_needs_trailing",
            r"(?i)\s+to suit your needs\b",
        ),
        ("span.please_note_that_leading", r"(?i)^please note that\s+"),
    ])
});

/// Compiles a `(id, pattern)` list into `(id, Regex)`, panicking on a
/// malformed pattern — every pattern here is a fixed, hand-transcribed
/// literal, so a compile failure is this module's own bug, not a runtime
/// condition.
fn build(patterns: &[(&'static str, &'static str)]) -> Vec<(&'static str, Regex)> {
    patterns
        .iter()
        .map(|(id, pattern)| {
            (
                *id,
                Regex::new(pattern).unwrap_or_else(|e| panic!("pattern {id} must compile: {e}")),
            )
        })
        .collect()
}

/// Which of the four tell-span families a [`TellSpanHit`] belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TellSpanFamily {
    /// A whole-sentence ritual frame (mined seed inventory).
    Ritual,
    /// A paired substitution frame.
    Substitution,
    /// A gated-deletion span.
    Deletion,
    /// A licensed derivational-pivot (light-verb) construction.
    Pivot,
}

/// One detected tell span.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TellSpanHit {
    /// Which family detected this hit.
    pub family: TellSpanFamily,
    /// Stable identifier of the specific pattern that matched.
    pub pattern_id: &'static str,
    /// Index of the sentence (within `clean(text)`'s sentence split) this
    /// hit was found in.
    pub sentence_index: usize,
    /// The matched surface text.
    pub matched_text: String,
    /// Tokens this hit's own pack-provided replacement legitimately
    /// introduces: empty for Ritual/Deletion (delete-only), the
    /// replacement's tokens for a Substitution hit, the single derived
    /// verb's token for a licensed Pivot hit. This is
    /// [`crate::closure`]'s per-input "pack-derivable" allowance.
    pub licensed_tokens: Vec<String>,
}

/// True if two half-open byte ranges share at least one byte.
const fn ranges_overlap(a: &std::ops::Range<usize>, b: &std::ops::Range<usize>) -> bool {
    a.start < b.end && b.start < a.end
}

/// Splits `clean(text)` into sentences and scans each one against all four
/// families.
///
/// Scan order: RITUAL (whole-sentence, matched once per pattern), SUBS
/// (every non-overlapping occurrence counts), SPANS (every non-overlapping
/// occurrence counts), PIVOT ([`pivot::match_pivot`], counted only when
/// `Licensed`).
///
/// Within SUBS and, separately, within SPANS, a pattern's match is skipped
/// when its byte range overlaps a match an earlier pattern (in the list
/// order above) already claimed in the same sentence. Both lists
/// deliberately pair a specific pattern with a more general fallback (e.g.
/// `sub.this_guide_will_walk_you_through` before the mid-sentence
/// `sub.will_walk_you_through`, which it contains) so that one conceptual
/// edit is never counted twice just because two patterns both happen to
/// match the same span of text.
#[must_use]
pub fn tell_span_hits(text: &str, tagger: &dyn Tagger) -> Vec<TellSpanHit> {
    let cleaned = clean(text);
    let mut hits = Vec::new();

    for (sentence_index, sentence) in split_sentences(&cleaned).into_iter().enumerate() {
        for (pattern_id, pattern) in RITUAL.iter() {
            if let Some(m) = pattern.find(sentence) {
                hits.push(TellSpanHit {
                    family: TellSpanFamily::Ritual,
                    pattern_id,
                    sentence_index,
                    matched_text: m.as_str().to_string(),
                    licensed_tokens: Vec::new(),
                });
            }
        }

        let mut subs_claimed: Vec<std::ops::Range<usize>> = Vec::new();
        for (pattern_id, pattern, replacement) in SUBS.iter() {
            for m in pattern.find_iter(sentence) {
                let range = m.range();
                if subs_claimed.iter().any(|r| ranges_overlap(r, &range)) {
                    continue;
                }
                subs_claimed.push(range);
                hits.push(TellSpanHit {
                    family: TellSpanFamily::Substitution,
                    pattern_id,
                    sentence_index,
                    matched_text: m.as_str().to_string(),
                    licensed_tokens: tokenize(replacement),
                });
            }
        }

        let mut spans_claimed: Vec<std::ops::Range<usize>> = Vec::new();
        for (pattern_id, pattern) in SPANS.iter() {
            for m in pattern.find_iter(sentence) {
                let range = m.range();
                if spans_claimed.iter().any(|r| ranges_overlap(r, &range)) {
                    continue;
                }
                spans_claimed.push(range);
                hits.push(TellSpanHit {
                    family: TellSpanFamily::Deletion,
                    pattern_id,
                    sentence_index,
                    matched_text: m.as_str().to_string(),
                    licensed_tokens: Vec::new(),
                });
            }
        }

        if let PivotOutcome::Licensed(licensed) = pivot::match_pivot(sentence, tagger) {
            hits.push(TellSpanHit {
                family: TellSpanFamily::Pivot,
                pattern_id: "pivot.licensed_lvc",
                sentence_index,
                matched_text: licensed.matched_text,
                licensed_tokens: tokenize(&licensed.derived_verb),
            });
        }
    }

    hits
}

/// The number of tell spans [`tell_span_hits`] finds in `text`.
#[must_use]
pub fn count_tell_spans(text: &str, tagger: &dyn Tagger) -> usize {
    tell_span_hits(text, tagger).len()
}

#[cfg(test)]
mod tests {
    use std::sync::OnceLock;

    use friction_nlp::NlpruleTagger;

    use super::*;

    fn tagger() -> &'static NlpruleTagger {
        static TAGGER: OnceLock<NlpruleTagger> = OnceLock::new();
        TAGGER.get_or_init(|| NlpruleTagger::new().expect("embedded model must load"))
    }

    #[test]
    fn detects_ritual_sentence() {
        let text = "If you have any questions, please reach out to us.";
        let hits = tell_span_hits(text, tagger());
        assert!(hits.iter().any(|h| h.family == TellSpanFamily::Ritual));
    }

    #[test]
    fn detects_substitution_with_licensed_replacement_tokens() {
        let text = "This guide will walk you through the setup.";
        let hits = tell_span_hits(text, tagger());
        let hit = hits
            .iter()
            .find(|h| h.family == TellSpanFamily::Substitution)
            .expect("substitution hit expected");
        assert_eq!(hit.licensed_tokens, vec!["this", "guide", "covers"]);
    }

    #[test]
    fn detects_deletion_span() {
        let text = "It is important to note that the tool is fast.";
        let hits = tell_span_hits(text, tagger());
        assert!(hits.iter().any(|h| h.family == TellSpanFamily::Deletion));
    }

    #[test]
    fn detects_licensed_pivot() {
        let text = "The system performs an initialization of the database.";
        let hits = tell_span_hits(text, tagger());
        let hit = hits
            .iter()
            .find(|h| h.family == TellSpanFamily::Pivot)
            .expect("pivot hit expected");
        assert_eq!(hit.licensed_tokens, vec!["initializes"]);
    }

    #[test]
    fn seed_inventory_matches_neither_rank_fixture_sample_by_ritual_subs_spans_alone() {
        // Empirically established finding from the investigation summary:
        // the primary seed inventory is not, by itself, load-bearing for
        // either rank fixture. This is a guardrail against silently
        // "fixing" that in a way that would mask the fragment-rate
        // guardrail's real role.
        let chatspeak = crate::fixtures::CHATSPEAK_MD;
        let good_docs = crate::fixtures::GOOD_DOCS_MD;
        assert_eq!(count_tell_spans(chatspeak, tagger()), 0);
        assert_eq!(count_tell_spans(good_docs, tagger()), 0);
    }

    #[test]
    fn overlapping_subs_pattern_pair_counts_once() {
        // "this guide will walk you through" is matched by both the
        // specific `sub.this_guide_will_walk_you_through` pattern and the
        // more general `sub.will_walk_you_through` fallback it contains;
        // only the specific (earlier-listed) one should count.
        let text = "This guide will walk you through the setup.";
        let hits: Vec<_> = tell_span_hits(text, tagger())
            .into_iter()
            .filter(|h| h.family == TellSpanFamily::Substitution)
            .collect();
        assert_eq!(
            hits.len(),
            1,
            "expected exactly one substitution hit: {hits:?}"
        );
        assert_eq!(hits[0].pattern_id, "sub.this_guide_will_walk_you_through");
    }

    #[test]
    fn overlapping_spans_pattern_pair_counts_once() {
        // A sentence-initial "It is important to note that " is matched by
        // both the anchored `span.it_is_important_to_note_that_leading`
        // pattern and the unanchored `_mid` fallback; only the
        // earlier-listed one should count.
        let text = "It is important to note that the tool is fast.";
        let hits: Vec<_> = tell_span_hits(text, tagger())
            .into_iter()
            .filter(|h| h.family == TellSpanFamily::Deletion)
            .collect();
        assert_eq!(hits.len(), 1, "expected exactly one deletion hit: {hits:?}");
        assert_eq!(
            hits[0].pattern_id,
            "span.it_is_important_to_note_that_leading"
        );
    }

    #[test]
    fn tell_span_hits_are_deterministic() {
        let text = "This guide will walk you through configuring the backup agent.";
        assert_eq!(
            tell_span_hits(text, tagger()),
            tell_span_hits(text, tagger())
        );
    }
}
