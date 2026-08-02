//! The literal automaton and its regex fallback (this crate's algorithm
//! reference, §2): short, no-length-signal tells covered by a single
//! Aho-Corasick automaton, everything else covered by a compiled-once
//! regex scan.

use std::collections::BTreeSet;

use aho_corasick::{AhoCorasick, MatchKind};
use friction_packs::{Anchor, InventoryPack};
use regex::Regex;

use crate::span::{Channel, MatchScore, MatchSpan};
use crate::token::ScopedUnit;

/// Recognizes `pattern_source` as a plain, case-insensitive,
/// boundary-delimited literal phrase: optional leading `(?i)`, optional
/// leading `\b`, one or more ASCII-letter/apostrophe words separated by
/// single literal spaces, then EITHER a trailing `\b` OR a trailing
/// `\s+` (the `deletion_spans` "consume trailing whitespace" shape):
/// nothing else. Returns the lowercased phrase, or `None` for any
/// pattern using a regex feature this heuristic doesn't recognize as
/// trivially literal (alternation, character classes, `^`/`$`,
/// quantifiers) — those fall back to [`regex_fallback_spans`], which is
/// an intended outcome of this channel's design, not a gap.
pub fn literal_phrase(pattern_source: &str) -> Option<Box<str>> {
    let stripped = pattern_source
        .strip_prefix("(?i)")
        .unwrap_or(pattern_source);
    let stripped = stripped.strip_prefix(r"\b").unwrap_or(stripped);

    let body = stripped
        .strip_suffix(r"\b")
        .or_else(|| stripped.strip_suffix(r"\s+"))?;

    if body.is_empty() {
        return None;
    }

    let words: Vec<&str> = body.split(' ').collect();
    let all_literal_words = words
        .iter()
        .all(|w| !w.is_empty() && w.chars().all(|c| c.is_ascii_alphabetic() || c == '\''));
    if !all_literal_words {
        return None;
    }

    Some(body.to_ascii_lowercase().into_boxed_str())
}

/// A once-built Aho-Corasick automaton over the inventory's literal-
/// eligible tells, plus the bookkeeping needed to translate a match back
/// to its frame id and to know which entries the regex fallback still
/// owns.
pub struct LiteralAutomaton {
    automaton: AhoCorasick,
    /// Parallel to the automaton's own pattern ids.
    frame_ids: Vec<Box<str>>,
    /// The inventory entry ids this automaton covers, the regex
    /// fallback's own exclusion set.
    ac_covered_ids: BTreeSet<Box<str>>,
}

impl LiteralAutomaton {
    /// Built once from `inventory`'s `deletion_spans` (mid-sentence
    /// anchor only) and `substitution_pairs` whose pattern
    /// [`literal_phrase`]-extracts cleanly. `MatchKind::LeftmostLongest`,
    /// ASCII case-insensitive.
    ///
    /// # Errors
    /// [`crate::MatchError::Automaton`] if the collected literal patterns
    /// fail to build into a working automaton.
    pub fn build(inventory: &InventoryPack) -> Result<Self, crate::MatchError> {
        let mut patterns: Vec<String> = Vec::new();
        let mut frame_ids: Vec<Box<str>> = Vec::new();
        let mut ac_covered_ids: BTreeSet<Box<str>> = BTreeSet::new();

        for span in inventory.deletion_spans() {
            if span.anchor != Anchor::MidSentence {
                continue;
            }
            if let Some(phrase) = literal_phrase(&span.pattern_source) {
                patterns.push(phrase.into());
                frame_ids.push(span.id.clone());
                ac_covered_ids.insert(span.id.clone());
            }
        }
        for pair in inventory.substitution_pairs() {
            if let Some(phrase) = literal_phrase(&pair.pattern_source) {
                patterns.push(phrase.into());
                frame_ids.push(pair.id.clone());
                ac_covered_ids.insert(pair.id.clone());
            }
        }

        let automaton = AhoCorasick::builder()
            .match_kind(MatchKind::LeftmostLongest)
            .ascii_case_insensitive(true)
            .build(&patterns)?;

        Ok(Self {
            automaton,
            frame_ids,
            ac_covered_ids,
        })
    }

    /// Scans every unit in `units`, in order.
    pub(crate) fn scan_units(&self, units: &[ScopedUnit<'_>]) -> Vec<MatchSpan> {
        units.iter().flat_map(|unit| self.scan_unit(unit)).collect()
    }

    /// Scans one prose unit's full text. A match is kept only if the
    /// byte immediately before/after it (if any) is not a word character
    /// (`is_alphanumeric` or `'`) — the automaton's own word-boundary
    /// discipline, since `aho-corasick` has none built in.
    fn scan_unit(&self, unit: &ScopedUnit<'_>) -> Vec<MatchSpan> {
        let mut out = Vec::new();
        for mat in self.automaton.find_iter(unit.text) {
            let start = mat.start();
            let end = mat.end();

            let before_is_word = unit.text[..start]
                .chars()
                .next_back()
                .is_some_and(is_word_char);
            let after_is_word = unit.text[end..].chars().next().is_some_and(is_word_char);
            if before_is_word || after_is_word {
                continue;
            }

            let frame_id = self.frame_ids[mat.pattern().as_usize()].clone();
            out.push(MatchSpan {
                range: (unit.unit.range.start + start)..(unit.unit.range.start + end),
                channel: Channel::Literal,
                frame_id,
                score: MatchScore::Present,
                message: None,
            });
        }
        out
    }
}

/// `true` for a word-continuing character: alphanumeric or apostrophe.
fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '\''
}

/// Compiled-once-at-pack-load regex fallback for every `deletion_spans`
/// (`SentenceInitial`/`Trailing` anchor) and `substitution_pairs` entry
/// [`literal_phrase`] rejected. Runs each pattern per SENTENCE (not per
/// unit) so `^`/`$` anchor correctly against sentence, not paragraph,
/// boundaries.
pub fn regex_fallback_spans(
    inventory: &InventoryPack,
    covered: &LiteralAutomaton,
    units: &[ScopedUnit<'_>],
) -> Vec<MatchSpan> {
    let mut entries: Vec<(&str, &Regex)> = Vec::new();
    for span in inventory.deletion_spans() {
        if !covered.ac_covered_ids.contains(&span.id) {
            entries.push((span.id.as_ref(), &span.pattern));
        }
    }
    for pair in inventory.substitution_pairs() {
        if !covered.ac_covered_ids.contains(&pair.id) {
            entries.push((pair.id.as_ref(), &pair.pattern));
        }
    }

    let mut out = Vec::new();
    for unit in units {
        for sentence in &unit.sentences {
            let local_start = sentence.start - unit.unit.range.start;
            let local_end = sentence.end - unit.unit.range.start;
            let sentence_text = &unit.text[local_start..local_end];

            for (id, pattern) in &entries {
                for mat in pattern.find_iter(sentence_text) {
                    out.push(MatchSpan {
                        range: (sentence.start + mat.start())..(sentence.start + mat.end()),
                        channel: Channel::Literal,
                        frame_id: (*id).into(),
                        score: MatchScore::Present,
                        message: None,
                    });
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn literal_phrase_extracts_a_plain_bounded_phrase() {
        assert_eq!(
            literal_phrase(r"(?i)\bprior to\b").as_deref(),
            Some("prior to")
        );
    }

    #[test]
    fn literal_phrase_extracts_the_trailing_whitespace_shape() {
        assert_eq!(
            literal_phrase(r"(?i)\bsimply\s+").as_deref(),
            Some("simply")
        );
        assert_eq!(
            literal_phrase(r"(?i)\bit is important to note that\s+").as_deref(),
            Some("it is important to note that")
        );
    }

    #[test]
    fn literal_phrase_rejects_alternation_and_optional_groups() {
        assert_eq!(literal_phrase(r"(?i)\bwe'?ll delve into\b"), None);
        assert_eq!(literal_phrase("(?i)we hope (this|you)"), None);
        assert_eq!(literal_phrase(r"(?i)^happy \w+ing"), None);
    }

    #[test]
    fn literal_phrase_rejects_sentence_anchored_patterns() {
        // No trailing `\b`/`\s+` at all (a `^`-anchored pattern with a
        // trailing `\s+` still qualifies structurally, but a pattern
        // whose body isn't a plain word/space run does not).
        assert_eq!(
            literal_phrase(r"(?i)(please )?replace the bracketed placeholders[^.\n]*"),
            None
        );
    }

    #[test]
    fn build_covers_mid_sentence_deletion_spans_and_literal_substitution_pairs() {
        let automaton =
            LiteralAutomaton::build(&friction_packs::INVENTORY.pack).expect("automaton builds");
        assert!(automaton.ac_covered_ids.contains("span.simply"));
        assert!(automaton.ac_covered_ids.contains("sub.prior_to"));
        // Sentence-initial anchor: never automaton-eligible.
        assert!(
            !automaton
                .ac_covered_ids
                .contains("span.it_is_important_to_note_that_leading")
        );
    }
}
