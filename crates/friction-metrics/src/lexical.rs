//! Lexical marker metrics: sentence-initial discourse markers, contraction
//! ratio, ritual open/close phrases, and the `"not just/only X but (also)
//! Y"` construction.
//!
//! Every function here is a pure function of a [`friction_core::Document`]:
//! same input, same output, on any machine or run — one left-to-right,
//! source-order pass over paragraphs and sentences, no `HashMap`, no
//! parallel reduction, and integer counters summed before dividing once,
//! so order never affects the result.
//!
//! # Matching rules
//!
//! - **Case**: phrase lists are ASCII; case-folds ASCII-only, not
//!   locale-aware, since there's no ambient locale to vary by.
//! - **Apostrophes**: a `'` also matches `’` (`U+2019`) — markdown and
//!   LLM output commonly use "smart" quotes.
//! - **Word boundaries**: a marker must be followed by a non-alphanumeric
//!   character or nothing: `"However"` matches `"However, it..."`, not
//!   `"Howeverish..."`.
//! - **Leading markdown emphasis**: `friction-parse` bridges emphasis
//!   delimiters into sentence text instead of stripping them, so
//!   `"**Overall:** it works."` keeps its `**` prefix; leading whitespace
//!   and `*`/`_` are stripped to a fixed point before matching, or the
//!   match silently fails.
//! - **Tokenization** (contraction ratio, discourse-marker density):
//!   [`word_tokens`] splits text into maximal alphabetic runs; an
//!   interior apostrophe (ASCII or `’`, flanked by letters) is part of
//!   the word, so `"don't"`/`"it's"` are single tokens, `"well-known"` is
//!   two. Simpler than `friction-nlp`'s tokenizer: only word count and
//!   phrase matching are needed, not POS-taggable spans.

use std::sync::LazyLock;

use friction_core::Document;
use regex::Regex;

/// Sentence-initial discourse markers: transition/hedge phrases
/// disproportionately common in LLM output vs. human writing
/// (`"Moreover, ..."`, `"It's worth noting that ..."`). Sorted
/// alphabetically (ASCII byte order); matched case-insensitively,
/// sentence-initial only.
const DISCOURSE_MARKERS: [&str; 32] = [
    "Additionally",
    "Consequently",
    "Conversely",
    "Finally",
    "Firstly",
    "Furthermore",
    "However",
    "Importantly",
    "In conclusion",
    "In fact",
    "In other words",
    "In short",
    "In summary",
    "Indeed",
    "It is important to note",
    "It's worth noting",
    "Lastly",
    "Likewise",
    "Meanwhile",
    "Moreover",
    "Nevertheless",
    "Nonetheless",
    "Notably",
    "Overall",
    "Similarly",
    "Specifically",
    "Subsequently",
    "That said",
    "Therefore",
    "Thus",
    "To summarize",
    "Ultimately",
];

/// Ritual open/close phrases: stock transitions LLM prose
/// disproportionately uses to open/close a document or paragraph
/// (`"In today's fast-paced world, ..."`, `"...To summarize, ..."`).
/// Sorted alphabetically; matched case-insensitively against a
/// paragraph's first or last sentence.
///
/// Overlaps `DISCOURSE_MARKERS` on a few entries (`"Overall"`, `"To
/// summarize"`) by design — both hedges and bookends, tracked
/// independently.
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

/// A contractible pair: `expanded` (one or more whitespace-separated
/// words, e.g. `"do not"` or `"cannot"`) and its `contracted` single-token
/// form (e.g. `"don't"`, `"can't"`).
struct ContractionPair {
    expanded: &'static str,
    contracted: &'static str,
}

/// Standard English contractions and their expanded forms. Sorted
/// alphabetically by `expanded`. Some contracted forms are ambiguous
/// (`"it's"` -> `"it is"` or `"it has"`); this table picks the more
/// common expansion, no context disambiguation.
const CONTRACTION_PAIRS: [ContractionPair; 28] = [
    ContractionPair {
        expanded: "cannot",
        contracted: "can't",
    },
    ContractionPair {
        expanded: "could not",
        contracted: "couldn't",
    },
    ContractionPair {
        expanded: "did not",
        contracted: "didn't",
    },
    ContractionPair {
        expanded: "do not",
        contracted: "don't",
    },
    ContractionPair {
        expanded: "does not",
        contracted: "doesn't",
    },
    ContractionPair {
        expanded: "had not",
        contracted: "hadn't",
    },
    ContractionPair {
        expanded: "has not",
        contracted: "hasn't",
    },
    ContractionPair {
        expanded: "have not",
        contracted: "haven't",
    },
    ContractionPair {
        expanded: "he is",
        contracted: "he's",
    },
    ContractionPair {
        expanded: "i am",
        contracted: "i'm",
    },
    ContractionPair {
        expanded: "i have",
        contracted: "i've",
    },
    ContractionPair {
        expanded: "i will",
        contracted: "i'll",
    },
    ContractionPair {
        expanded: "is not",
        contracted: "isn't",
    },
    ContractionPair {
        expanded: "it is",
        contracted: "it's",
    },
    ContractionPair {
        expanded: "must not",
        contracted: "mustn't",
    },
    ContractionPair {
        expanded: "she is",
        contracted: "she's",
    },
    ContractionPair {
        expanded: "should not",
        contracted: "shouldn't",
    },
    ContractionPair {
        expanded: "that is",
        contracted: "that's",
    },
    ContractionPair {
        expanded: "they are",
        contracted: "they're",
    },
    ContractionPair {
        expanded: "they will",
        contracted: "they'll",
    },
    ContractionPair {
        expanded: "was not",
        contracted: "wasn't",
    },
    ContractionPair {
        expanded: "we are",
        contracted: "we're",
    },
    ContractionPair {
        expanded: "we will",
        contracted: "we'll",
    },
    ContractionPair {
        expanded: "were not",
        contracted: "weren't",
    },
    ContractionPair {
        expanded: "will not",
        contracted: "won't",
    },
    ContractionPair {
        expanded: "would not",
        contracted: "wouldn't",
    },
    ContractionPair {
        expanded: "you are",
        contracted: "you're",
    },
    ContractionPair {
        expanded: "you will",
        contracted: "you'll",
    },
];

/// Matches `"not just/only X but (also) Y"`, case- and
/// newline-insensitive (`(?i)` folds case, `(?s)` lets `.` cross soft
/// line breaks in a markdown-source sentence). Built once, lazily —
/// compiling a regex isn't free and this runs repeatedly.
static NOT_JUST_BUT_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?is)\bnot\s+(?:just|only)\b.*?\bbut\b(?:\s+also\b)?")
        .expect("not-just-but pattern is a fixed, valid regex")
});

/// The canonical `(expanded, contracted)` pairs this workspace treats as
/// standard English contractions.
///
/// Same data [`contraction_ratio`] counts over, as plain string pairs
/// instead of the private [`ContractionPair`] shape. Exposed for crates
/// needing this *exact* table — notably `friction-rules`'
/// contraction-insertion rule (turns `expanded` into `contracted` when a
/// document reads too formally for its genre) — so it reads the same
/// data instead of risking drift. Same order as [`CONTRACTION_PAIRS`].
#[must_use]
pub fn contraction_pairs() -> Vec<(&'static str, &'static str)> {
    CONTRACTION_PAIRS
        .iter()
        .map(|pair| (pair.expanded, pair.contracted))
        .collect()
}

/// Sentence-initial discourse-marker density, per 1000 word tokens.
///
/// Counts sentences starting with a [`DISCOURSE_MARKERS`] entry (after
/// [`strip_leading_markup`]), divides by total word tokens, scales to
/// per-1000. Returns `0.0` for a document with no word tokens.
#[must_use]
pub fn discourse_marker_density(document: &Document) -> f64 {
    let mut marker_sentences = 0u64;
    let mut total_tokens = 0u64;
    for unit in document.prose() {
        for sentence in &unit.sentences {
            let Ok(text) = document.text(&sentence.range) else {
                continue;
            };
            total_tokens += word_tokens(text).len() as u64;
            if starts_with_any(text, &DISCOURSE_MARKERS) {
                marker_sentences += 1;
            }
        }
    }
    if total_tokens == 0 {
        return 0.0;
    }
    #[allow(clippy::cast_precision_loss)]
    let density = marker_sentences as f64 * 1000.0 / total_tokens as f64;
    density
}

/// The raw `(contracted, contractible)` occurrence counts
/// [`contraction_ratio`] divides, over every [`CONTRACTION_PAIRS`] entry.
///
/// Tokenizes per sentence, not per paragraph — an `expanded` phrase must
/// never match across a sentence boundary, e.g. "is" ending one sentence
/// and "not" opening the next isn't `"is not"` -> `"isn't"`. Counts exact
/// single-token matches of `contracted` and consecutive-token matches of
/// `expanded`, one-word and multi-word forms alike.
///
/// Exposed separately from [`contraction_ratio`] because callers need raw
/// counts for the *exact* per-occurrence effect (`1 / (contracted +
/// contractible)`) — notably `friction-rules`' contraction-insertion
/// rule, which reads its own document rather than assuming a size.
#[must_use]
pub fn contraction_counts(document: &Document) -> (u64, u64) {
    let mut contracted = 0u64;
    let mut contractible = 0u64;
    for unit in document.prose() {
        for sentence in &unit.sentences {
            let Ok(text) = document.text(&sentence.range) else {
                continue;
            };
            let tokens = word_tokens(text);
            for pair in &CONTRACTION_PAIRS {
                contracted += tokens
                    .iter()
                    .filter(|t| t.as_str() == pair.contracted)
                    .count() as u64;
                let expanded_words: Vec<&str> = pair.expanded.split_whitespace().collect();
                contractible += count_phrase_occurrences(&tokens, &expanded_words) as u64;
            }
        }
    }
    (contracted, contractible)
}

/// Ratio of contracted to contractible forms: `contracted / (contracted +
/// contractible)`, from [`contraction_counts`].
///
/// Returns `0.0`, not `NaN`, when neither form appears in the document.
#[must_use]
pub fn contraction_ratio(document: &Document) -> f64 {
    let (contracted, contractible) = contraction_counts(document);
    let denominator = contracted + contractible;
    if denominator == 0 {
        return 0.0;
    }
    #[allow(clippy::cast_precision_loss)]
    let ratio = contracted as f64 / denominator as f64;
    ratio
}

/// Rate of paragraphs that open or close with a ritual transition
/// phrase.
///
/// A paragraph is a [`friction_core::ProseUnit`] with a sentence;
/// "flagged" means its first or last sentence (same one, for
/// one-sentence paragraphs) starts with a [`RITUAL_MARKERS`] entry after
/// [`strip_leading_markup`] (lets `"**Overall:** ..."` match). Returns
/// `0.0` for a document with no paragraphs that have sentences.
#[must_use]
pub fn ritual_marker_rate(document: &Document) -> f64 {
    let mut paragraphs = 0u64;
    let mut flagged = 0u64;
    for unit in document.prose() {
        let Some(first) = unit.sentences.first() else {
            continue;
        };
        let last = unit.sentences.last().unwrap_or(first);
        paragraphs += 1;

        let opens = document
            .text(&first.range)
            .is_ok_and(|text| starts_with_any(text, &RITUAL_MARKERS));
        let closes = document
            .text(&last.range)
            .is_ok_and(|text| starts_with_any(text, &RITUAL_MARKERS));
        if opens || closes {
            flagged += 1;
        }
    }
    if paragraphs == 0 {
        return 0.0;
    }
    #[allow(clippy::cast_precision_loss)]
    let rate = flagged as f64 / paragraphs as f64;
    rate
}

/// Rate of `"not just/only X but (also) Y"` constructions, per sentence.
///
/// Counts sentences matching [`NOT_JUST_BUT_RE`], divides by total
/// sentence count. Returns `0.0` for a document with no sentences.
#[must_use]
pub fn not_just_but_rate(document: &Document) -> f64 {
    let mut sentences = 0u64;
    let mut matches = 0u64;
    for unit in document.prose() {
        for sentence in &unit.sentences {
            let Ok(text) = document.text(&sentence.range) else {
                continue;
            };
            sentences += 1;
            if NOT_JUST_BUT_RE.is_match(text) {
                matches += 1;
            }
        }
    }
    if sentences == 0 {
        return 0.0;
    }
    #[allow(clippy::cast_precision_loss)]
    let rate = matches as f64 / sentences as f64;
    rate
}

/// Returns `true` if `text`, after [`strip_leading_markup`], starts with
/// any phrase in `markers` (case/apostrophe/word-boundary rules: module
/// docs).
fn starts_with_any(text: &str, markers: &[&str]) -> bool {
    let stripped = strip_leading_markup(text);
    markers
        .iter()
        .any(|marker| starts_with_marker(stripped, marker))
}

/// Strips leading whitespace and markdown emphasis/strong delimiters
/// (`*`/`_`) from `text`, repeating until a pass removes nothing further.
///
/// `friction-parse` bridges emphasis into sentence text rather than
/// stripping it, so `"**Overall:**"` keeps its prefix; repeating (not
/// once) handles a delimiter run followed by more whitespace, e.g. `"**
/// Overall"`.
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

/// Returns `true` if `text` begins with `marker`, case-insensitively
/// (ASCII case-folding — every marker table here is ASCII), followed by
/// a non-alphanumeric character or end of `text`. A `'` in `marker` also
/// matches `’`.
fn starts_with_marker(text: &str, marker: &str) -> bool {
    let mut text_chars = text.chars();
    for expected in marker.chars() {
        let Some(actual) = text_chars.next() else {
            return false;
        };
        let matches = if expected == '\'' {
            actual == '\'' || actual == '\u{2019}'
        } else {
            actual.eq_ignore_ascii_case(&expected)
        };
        if !matches {
            return false;
        }
    }
    text_chars.next().is_none_or(|c| !c.is_alphanumeric())
}

/// Splits `text` into lowercase word tokens; see the module docs'
/// "Tokenization" section for the exact rule.
fn word_tokens(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut tokens = Vec::new();
    let mut current = String::new();
    for (i, &c) in chars.iter().enumerate() {
        let is_apostrophe = c == '\'' || c == '\u{2019}';
        let is_interior_apostrophe = is_apostrophe
            && i > 0
            && chars[i - 1].is_alphabetic()
            && chars.get(i + 1).is_some_and(|next| next.is_alphabetic());
        if c.is_alphabetic() || is_interior_apostrophe {
            current.push(c.to_ascii_lowercase());
        } else if !current.is_empty() {
            tokens.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

/// Counts how many `tokens` windows of `phrase.len()` match `phrase`
/// element-wise (both sides already lowercase).
fn count_phrase_occurrences(tokens: &[String], phrase: &[&str]) -> usize {
    if phrase.is_empty() || tokens.len() < phrase.len() {
        return 0;
    }
    tokens
        .windows(phrase.len())
        .filter(|window| window.iter().zip(phrase).all(|(t, p)| t == p))
        .count()
}

#[cfg(test)]
// Every fixture is a hand-computed exact value (see each test's doc
// comment), so `==` instead of an epsilon is intentional, not a
// float-precision bug. Several also use a one-sentence paragraph
// (`&[a..b]`), which clippy misreads as a `Range` literal rather than a
// one-element array.
#[allow(clippy::float_cmp, clippy::single_range_in_vec_init)]
mod tests {
    use std::ops::Range;

    use friction_core::{Block, BlockKind, ProseUnit, Sentence};

    use super::*;

    /// `DISCOURSE_MARKERS` is sorted (ASCII byte order), no duplicates —
    /// documents its own invariant rather than relying on the author's
    /// eye.
    #[test]
    fn discourse_markers_sorted_and_unique() {
        assert!(DISCOURSE_MARKERS.windows(2).all(|w| w[0] < w[1]));
    }

    /// `RITUAL_MARKERS` is sorted (ASCII byte order) with no duplicates.
    #[test]
    fn ritual_markers_sorted_and_unique() {
        assert!(RITUAL_MARKERS.windows(2).all(|w| w[0] < w[1]));
    }

    /// `CONTRACTION_PAIRS` is sorted by `expanded` (ASCII byte order) with
    /// no duplicate `expanded` or `contracted` entries.
    #[test]
    fn contraction_pairs_sorted_and_unique() {
        assert!(
            CONTRACTION_PAIRS
                .windows(2)
                .all(|w| w[0].expanded < w[1].expanded)
        );
        for (i, a) in CONTRACTION_PAIRS.iter().enumerate() {
            for b in &CONTRACTION_PAIRS[i + 1..] {
                assert_ne!(a.contracted, b.contracted);
            }
        }
    }

    /// `contraction_pairs` hands back [`CONTRACTION_PAIRS`], same order,
    /// unwrapped into plain tuples — the accessor other crates (e.g.
    /// `friction-rules`) reuse instead of keeping their own copy.
    #[test]
    fn contraction_pairs_matches_internal_table() {
        let pairs = contraction_pairs();
        assert_eq!(pairs.len(), CONTRACTION_PAIRS.len());
        for (public, internal) in pairs.iter().zip(CONTRACTION_PAIRS.iter()) {
            assert_eq!(*public, (internal.expanded, internal.contracted));
        }
    }

    /// Builds a single-block, single-paragraph document from pre-cut
    /// sentence ranges, bypassing `friction-parse`/`friction-nlp` —
    /// hand-computed offsets keep expected values exact and independent
    /// of other crates.
    fn doc_from_sentences(source: &'static str, sentence_ranges: &[Range<usize>]) -> Document {
        let sentences = sentence_ranges
            .iter()
            .cloned()
            .map(|range| Sentence::new(range, Vec::new()))
            .collect();
        let blocks = vec![Block::new(BlockKind::Paragraph, 0..source.len())];
        let prose = vec![ProseUnit::new(0, 0..source.len(), sentences)];
        Document::new(source, blocks, prose).expect("hand-built fixture must be well-formed")
    }

    /// Builds a multi-paragraph document: one block/prose-unit/sentence
    /// group per entry in `paragraphs`.
    fn doc_from_paragraphs(
        source: &'static str,
        paragraphs: &[(Range<usize>, &[Range<usize>])],
    ) -> Document {
        let mut blocks = Vec::new();
        let mut prose = Vec::new();
        for (block_index, (para_range, sentence_ranges)) in paragraphs.iter().enumerate() {
            blocks.push(Block::new(BlockKind::Paragraph, para_range.clone()));
            let sentences = sentence_ranges
                .iter()
                .cloned()
                .map(|range| Sentence::new(range, Vec::new()))
                .collect();
            prose.push(ProseUnit::new(block_index, para_range.clone(), sentences));
        }
        Document::new(source, blocks, prose).expect("hand-built fixture must be well-formed")
    }

    // --- strip_leading_markup ------------------------------------------

    /// Plain text with only leading whitespace trims like
    /// `str::trim_start`, with no markup to strip.
    #[test]
    fn strip_leading_markup_trims_whitespace_only() {
        assert_eq!(
            strip_leading_markup("  Overall it works."),
            "Overall it works."
        );
    }

    /// A leading `**` run (bold) is stripped down to the word underneath.
    #[test]
    fn strip_leading_markup_strips_bold_delimiters() {
        assert_eq!(
            strip_leading_markup("**Overall:** it works."),
            "Overall:** it works."
        );
    }

    /// A leading `__` run (underscore strong emphasis) is stripped the
    /// same way `**` is.
    #[test]
    fn strip_leading_markup_strips_underscore_delimiters() {
        assert_eq!(
            strip_leading_markup("__However__, it works."),
            "However__, it works."
        );
    }

    /// Stripping repeats until neither removes anything further: `"**
    /// Overall"` has a space *inside* the leading `**`, so a single pass
    /// would stop at `" Overall"` — this checks the second pass strips
    /// that remaining space.
    #[test]
    fn strip_leading_markup_repeats_until_fixed_point() {
        assert_eq!(
            strip_leading_markup("** Overall it works."),
            "Overall it works."
        );
    }

    /// Text with no leading whitespace or markup is returned unchanged.
    #[test]
    fn strip_leading_markup_no_op_on_plain_text() {
        assert_eq!(
            strip_leading_markup("Overall it works."),
            "Overall it works."
        );
    }

    // --- discourse_marker_density -----------------------------------

    /// "However, it works." (3 tokens) and "Fine." (1 token, no marker):
    /// 1 marked sentence over 4 tokens, scaled to per-1000 = `1 * 1000 /
    /// 4 = 250.0` exactly.
    #[test]
    fn discourse_marker_density_hand_computed() {
        let source = "However, it works. Fine.";
        let doc = doc_from_sentences(source, &[0..18, 19..24]);
        assert_eq!(discourse_marker_density(&doc), 250.0);
    }

    /// Regression test for the bold-wrapped-marker detector bug: source
    /// text `"**However**, it works."` must still be recognized as
    /// opening with "However" — `word_tokens` gives 3 tokens, so 1 marked
    /// sentence over 3 = `1 * 1000 / 3`.
    #[test]
    fn discourse_marker_density_matches_bold_wrapped_marker() {
        let source = "**However**, it works.";
        let doc = doc_from_sentences(source, &[0..source.len()]);
        assert!((discourse_marker_density(&doc) - 1000.0 / 3.0).abs() < 1e-9);
    }

    /// Same recognition applies to the underscore delimiter, not just
    /// `**`.
    #[test]
    fn discourse_marker_density_matches_underscore_wrapped_marker() {
        let source = "__However__, it works.";
        let doc = doc_from_sentences(source, &[0..source.len()]);
        assert!((discourse_marker_density(&doc) - 1000.0 / 3.0).abs() < 1e-9);
    }

    /// A bold-wrapped word that isn't in `DISCOURSE_MARKERS` must not
    /// spuriously match just because its emphasis delimiters got
    /// stripped.
    #[test]
    fn discourse_marker_density_bold_non_marker_does_not_match() {
        let source = "**Something** happened.";
        let doc = doc_from_sentences(source, &[0..source.len()]);
        assert_eq!(discourse_marker_density(&doc), 0.0);
    }

    /// A marker word must be followed by a non-alphanumeric character:
    /// "Howeverish" is not "However" plus a word boundary, so density is
    /// `0.0`, not from 1 marked sentence.
    #[test]
    fn discourse_marker_density_requires_word_boundary() {
        let source = "Howeverish results came in.";
        let doc = doc_from_sentences(source, &[0..source.len()]);
        assert_eq!(discourse_marker_density(&doc), 0.0);
    }

    /// A document with no word tokens has density `0.0`, not `NaN` from a
    /// zero-over-zero division.
    #[test]
    fn discourse_marker_density_zero_tokens_is_zero() {
        let source = "";
        let doc = doc_from_sentences(source, &[0..0]);
        assert_eq!(discourse_marker_density(&doc), 0.0);
    }

    // --- contraction_ratio -------------------------------------------

    /// "Do not stop. Don't stop. It is fine. It's fine.": one `"do not"`
    /// bigram and `"don't"` token (pair 1), one `"it is"` bigram and
    /// `"it's"` token (pair 2). contracted = 2, contractible = 2, ratio =
    /// `2 / (2 + 2) = 0.5` exactly.
    #[test]
    fn contraction_ratio_hand_computed() {
        let source = "Do not stop. Don't stop. It is fine. It's fine.";
        let doc = doc_from_sentences(source, &[0..source.len()]);
        assert_eq!(contraction_ratio(&doc), 0.5);
    }

    /// A document with neither form present has ratio `0.0`, not `NaN`.
    #[test]
    fn contraction_ratio_no_forms_is_zero() {
        let source = "Cats sit on mats quietly.";
        let doc = doc_from_sentences(source, &[0..source.len()]);
        assert_eq!(contraction_ratio(&doc), 0.0);
    }

    /// Regression test: an `expanded` pair must never match across a
    /// sentence boundary.
    ///
    /// "She does. Not enough is done. It isn't clear." — no sentence
    /// alone has a contractible `expanded` phrase; the only real
    /// contraction is `"isn't"`: `contracted = 1`, `contractible = 0`,
    /// ratio = `1 / (1 + 0) = 1.0`.
    ///
    /// Tokenizing the whole paragraph as one stream (the bug) would match
    /// sentence 1's `"does"` against sentence 2's `"not"` as `"does not"`
    /// -> `"doesn't"`, inflating `contractible` to 1 and the ratio to `1
    /// / (1 + 1) = 0.5`.
    #[test]
    fn contraction_ratio_does_not_match_across_sentence_boundary() {
        let source = "She does. Not enough is done. It isn't clear.";
        let doc = doc_from_sentences(source, &[0..9, 10..29, 30..45]);
        assert_eq!(contraction_ratio(&doc), 1.0);
    }

    // --- ritual_marker_rate --------------------------------------------

    /// Four paragraphs: "Fine here." (not flagged), "Overall it works."
    /// (opens with "Overall", flagged), "Good stuff." (not flagged),
    /// "Fine so far. To summarize it works." (closes with "To
    /// summarize", flagged). 2 flagged / 4 paragraphs = `0.5` exactly.
    #[test]
    fn ritual_marker_rate_hand_computed() {
        let source = "Fine here.\n\nOverall it works.\n\nGood stuff.\n\nFine so far. To summarize it works.\n";
        let doc = doc_from_paragraphs(
            source,
            &[
                (0..10, &[0..10]),
                (12..29, &[12..29]),
                (31..42, &[31..42]),
                (44..79, &[44..56, 57..79]),
            ],
        );
        assert_eq!(ritual_marker_rate(&doc), 0.5);
    }

    /// A document with no paragraphs that have sentences has rate `0.0`,
    /// not `NaN`.
    #[test]
    fn ritual_marker_rate_no_paragraphs_is_zero() {
        let doc = doc_from_paragraphs("", &[]);
        assert_eq!(ritual_marker_rate(&doc), 0.0);
    }

    /// Regression test for the bug reproduced against
    /// `corpus/llm/blog/cde08357ef2e91de.md`: a one-sentence paragraph
    /// whose entire text is `"**Overall:**"` must be flagged — before
    /// the leading-markup strip, `starts_with_marker` saw `*` first and
    /// never matched "Overall", silently undercounting.
    #[test]
    fn ritual_marker_rate_matches_bold_wrapped_marker() {
        let source = "**Overall:**";
        let doc = doc_from_paragraphs(source, &[(0..source.len(), &[0..source.len()])]);
        assert_eq!(ritual_marker_rate(&doc), 1.0);
    }

    /// Same recognition applies when the marker opens a longer sentence,
    /// and to the underscore delimiter.
    #[test]
    fn ritual_marker_rate_matches_underscore_wrapped_marker() {
        let source = "__Overall__, it works out fine.";
        let doc = doc_from_paragraphs(source, &[(0..source.len(), &[0..source.len()])]);
        assert_eq!(ritual_marker_rate(&doc), 1.0);
    }

    // --- not_just_but_rate ----------------------------------------------

    /// Four sentences: "...not just fast but also reliable." (matches),
    /// "It works well." (no match), "...not only budget but timeline
    /// too." (matches, no "also" needed), "Nothing else to add." (no
    /// match). 2 matches / 4 sentences = `0.5` exactly.
    #[test]
    fn not_just_but_rate_hand_computed() {
        let source = "This solution is not just fast but also reliable. It works well. \
            The plan covers not only budget but timeline too. Nothing else to add.";
        let doc = doc_from_sentences(source, &[0..49, 50..64, 65..114, 115..135]);
        assert_eq!(not_just_but_rate(&doc), 0.5);
    }

    /// A document with no sentences has rate `0.0`, not `NaN`.
    #[test]
    fn not_just_but_rate_no_sentences_is_zero() {
        let doc = doc_from_sentences("", &[]);
        assert_eq!(not_just_but_rate(&doc), 0.0);
    }
}
