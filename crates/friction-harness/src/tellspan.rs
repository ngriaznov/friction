//! The curated tell-span inventory, unified into one scan.
//!
//! Ritual frames, substitution pairs, and deletion spans are sourced from
//! `friction_packs::INVENTORY`; the fourth family, a derivational-pivot
//! scan, comes from [`crate::pivot`].
//!
//! [`tell_span_hits`] is the single entry point every other module in this
//! crate that needs "how machine-register is this text" goes through:
//! [`crate::score`]'s primary tier counts its length, and
//! [`crate::closure`] reads each hit's `licensed_tokens` to build its
//! per-input pack vocabulary.

use std::ops::Range;

use friction_nlp::Tagger;

use crate::clean::{clean, split_sentences, tokenize};
use crate::pivot::{self, PivotOutcome};

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
const fn ranges_overlap(a: &Range<usize>, b: &Range<usize>) -> bool {
    a.start < b.end && b.start < a.end
}

/// Greedy longest-match-first interval selection: sorts candidate matches
/// by descending byte length (ties broken by the payload's own `Ord`,
/// which every caller here uses as a `pattern_id`-first tuple or plain
/// `pattern_id`, for determinism), then accepts each in turn if it
/// doesn't overlap an already-accepted range.
///
/// Equivalent to "prefer the more specific pattern" for every case this
/// module pairs a specific pattern with a general fallback (a specific
/// pattern's match is always a superset — hence longer — of the general
/// fallback it's paired with), but correct regardless of the patterns'
/// declaration or sort order — unlike the pack's own alphabetical-by-id
/// ordering, which is not "specific before general" (e.g. `sub.delve_into`
/// sorts before `sub.we_will_delve_into` alphabetically, the wrong
/// precedence under a naive "first in list order wins" rule).
fn resolve_overlaps<T: Ord>(mut candidates: Vec<(Range<usize>, T)>) -> Vec<(Range<usize>, T)> {
    candidates.sort_by(|a, b| {
        let len_a = a.0.end.saturating_sub(a.0.start);
        let len_b = b.0.end.saturating_sub(b.0.start);
        len_b.cmp(&len_a).then_with(|| a.1.cmp(&b.1))
    });

    let mut accepted: Vec<(Range<usize>, T)> = Vec::new();
    for (range, payload) in candidates {
        if accepted.iter().any(|(r, _)| ranges_overlap(r, &range)) {
            continue;
        }
        accepted.push((range, payload));
    }
    accepted
}

/// Splits `clean(text)` into sentences and scans each against all four families.
///
/// Ritual frames, substitution pairs, and deletion spans are sourced from
/// `friction_packs::INVENTORY`; the fourth comes from
/// [`pivot::match_pivot`].
///
/// Scan order: ritual frames (whole-sentence, matched once per pattern),
/// substitution pairs (every non-overlapping occurrence counts, resolved
/// via [`resolve_overlaps`]), deletion spans (same), pivot (counted only
/// when `Licensed`).
#[must_use]
pub fn tell_span_hits(text: &str, tagger: &dyn Tagger) -> Vec<TellSpanHit> {
    let cleaned = clean(text);
    let mut hits = Vec::new();
    let inventory = &friction_packs::INVENTORY.pack;

    for (sentence_index, sentence) in split_sentences(&cleaned).into_iter().enumerate() {
        for ritual in inventory.ritual_frames() {
            if let Some(m) = ritual.pattern.find(sentence) {
                hits.push(TellSpanHit {
                    family: TellSpanFamily::Ritual,
                    pattern_id: &ritual.id,
                    sentence_index,
                    matched_text: m.as_str().to_string(),
                    licensed_tokens: Vec::new(),
                });
            }
        }

        let mut sub_candidates: Vec<(Range<usize>, (&'static str, &'static str))> = Vec::new();
        for sub in inventory.substitution_pairs() {
            for m in sub.pattern.find_iter(sentence) {
                sub_candidates.push((m.range(), (&sub.id, &sub.replacement)));
            }
        }
        for (range, (pattern_id, replacement)) in resolve_overlaps(sub_candidates) {
            hits.push(TellSpanHit {
                family: TellSpanFamily::Substitution,
                pattern_id,
                sentence_index,
                matched_text: sentence[range].to_string(),
                licensed_tokens: tokenize(replacement),
            });
        }

        let mut span_candidates: Vec<(Range<usize>, &'static str)> = Vec::new();
        for span in inventory.deletion_spans() {
            for m in span.pattern.find_iter(sentence) {
                span_candidates.push((m.range(), &span.id));
            }
        }
        for (range, pattern_id) in resolve_overlaps(span_candidates) {
            hits.push(TellSpanHit {
                family: TellSpanFamily::Deletion,
                pattern_id,
                sentence_index,
                matched_text: sentence[range].to_string(),
                licensed_tokens: Vec::new(),
            });
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

    use friction_nlp::PerceptronTagger;

    use super::*;

    fn tagger() -> &'static PerceptronTagger {
        static TAGGER: OnceLock<PerceptronTagger> = OnceLock::new();
        TAGGER.get_or_init(|| PerceptronTagger::new().expect("embedded model must load"))
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
        // the full pack's Ritual/Substitution/Deletion families are not,
        // by themselves, load-bearing for either rank fixture. This is a
        // guardrail against silently "fixing" that in a way that would
        // mask the fragment-rate guardrail's real role. If a newly-mined
        // pack entry ever spuriously fires on either sample, the fix is
        // tightening that pattern in `inventory-en-v1.toml`, never loosening
        // this test.
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
        // only the longer (specific) match should count.
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
        // pattern and the unanchored `_mid` fallback; only the longer
        // (anchored) match should count.
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

    #[test]
    fn tell_span_hits_fires_on_a_pack_only_ritual_entry_absent_from_the_old_hardcoded_static() {
        // `ritual.would_you_like_me_to_elaborate` is a new, mined
        // pack entry (never part of the prior hardcoded RITUAL
        // static) — proves the swap to `friction_packs::INVENTORY` is
        // real, not a no-op refactor.
        let text = "Would you like me to elaborate on any specific aspect of this setup?";
        let hits = tell_span_hits(text, tagger());
        assert!(
            hits.iter()
                .any(|h| h.pattern_id == "ritual.would_you_like_me_to_elaborate"),
            "expected the pack-only ritual entry to fire: {hits:?}"
        );
    }

    #[test]
    fn resolve_overlaps_prefers_the_longer_match_regardless_of_id_order() {
        let candidates = vec![(0..3, "z_short"), (0..10, "a_long"), (20..25, "m_disjoint")];
        let accepted = resolve_overlaps(candidates);
        let mut ids: Vec<&str> = accepted.iter().map(|(_, id)| *id).collect();
        ids.sort_unstable();
        assert_eq!(ids, vec!["a_long", "m_disjoint"]);
    }

    #[test]
    fn resolve_overlaps_breaks_length_ties_by_payload_order() {
        let candidates = vec![(0..5, "b"), (0..5, "a")];
        let accepted = resolve_overlaps(candidates);
        assert_eq!(accepted.len(), 1);
        assert_eq!(accepted[0].1, "a");
    }
}
