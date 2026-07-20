//! Lexical marker metrics: sentence-initial discourse markers, contraction
//! ratio, ritual open/close phrases, and the `"not just/only X but (also)
//! Y"` construction.
//!
//! Every function in this module is a pure function of a
//! [`friction_core::Document`]: given the same document it returns the
//! same numbers, on any machine, on any run. Each walks the document's
//! paragraphs ([`Document::prose`]) and their sentences in a single
//! left-to-right pass, in source order — no `HashMap`/`HashSet`, no
//! parallel reduction, and float accumulation (the density and rate
//! metrics) always sums small integer counters first and divides once at
//! the end, so accumulation order never affects the result.
//!
//! # Matching rules
//!
//! - **Case**: every phrase list below is plain ASCII, and matching folds
//!   case with ASCII case-folding (`char::eq_ignore_ascii_case`,
//!   `str::to_ascii_lowercase`) rather than a locale-aware transform —
//!   there is no ambient locale to make that non-deterministic.
//! - **Apostrophes**: a `'` in a marker or contraction table also matches
//!   the Unicode right single quotation mark `’` (`U+2019`), since
//!   markdown sources and LLM output commonly use "smart" quotes for
//!   contractions and possessives.
//! - **Word boundaries**: a sentence-initial marker match additionally
//!   requires that the character right after the marker (if any) is not
//!   alphanumeric, so `"However"` matches `"However, it..."` but not
//!   `"Howeverish results..."`.
//! - **Leading markdown emphasis**: before a sentence-initial marker match
//!   is attempted, leading whitespace and leading `*`/`_` characters
//!   (markdown emphasis/strong delimiters) are stripped, repeating both
//!   strips until neither removes anything further. `friction-parse`
//!   deliberately *bridges* emphasis/strong delimiters into a sentence's
//!   source text rather than stripping them (see `friction-parse`'s
//!   `extract` module docs), so a bold- or italic-wrapped marker like
//!   `"**Overall:** it works."` or `"__However__, it works."` keeps its
//!   literal `**`/`__` prefix in the sentence text; without this stripping
//!   step the marker match would silently fail on every such sentence.
//! - **Tokenization** (for the contraction ratio and the per-1000-token
//!   discourse-marker density): [`word_tokens`] splits text into maximal
//!   runs of alphabetic characters, treating an interior apostrophe
//!   (ASCII or the Unicode right single quotation mark, surrounded by
//!   alphabetic characters on both sides) as part of the word — so
//!   `"don't"` and `"it's"` are single tokens — while hyphens and
//!   leading/trailing quotation marks are word separators, so
//!   `"well-known"` tokenizes as two words. This is intentionally simpler
//!   than `friction-nlp`'s tokenizer: these metrics only need a word
//!   count and word-boundary phrase matching, not POS-taggable token
//!   spans.

use std::sync::LazyLock;

use friction_core::Document;
use regex::Regex;

/// Sentence-initial discourse markers: transition/hedge phrases that, when
/// they open a sentence, are disproportionately common in LLM output
/// relative to human writing (`"Moreover, ..."`, `"It's worth noting that
/// ..."`). Sorted alphabetically (ASCII byte order); matched
/// case-insensitively only at the very start of a sentence, not mid-
/// sentence.
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

/// Ritual open/close phrases: stock transition phrases that LLM prose
/// disproportionately uses to open or close a document or paragraph
/// (`"In today's fast-paced world, ..."`, `"...To summarize, ..."`).
/// Sorted alphabetically (ASCII byte order); matched case-insensitively
/// against the start of a paragraph's first or last sentence.
///
/// This list overlaps `DISCOURSE_MARKERS` in a few entries (`"Overall"`,
/// `"To summarize"`) by design: those phrases are both general-purpose
/// sentence-initial hedges *and* characteristic paragraph/document
/// bookends, so both metrics track them independently.
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
/// words, e.g. `"do not"` or the already-single-word `"cannot"`) and its
/// `contracted` single-token form (e.g. `"don't"`, `"can't"`).
struct ContractionPair {
    expanded: &'static str,
    contracted: &'static str,
}

/// Contractible pairs: standard English contractions and their expanded
/// forms. Sorted alphabetically by `expanded` (ASCII byte order). A few
/// contracted forms are genuinely ambiguous in isolation (`"it's"` can
/// expand to either `"it is"` or `"it has"`); this table picks the
/// overwhelmingly more common expansion for each and does not attempt
/// disambiguation from context.
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

/// Matches a `"not just/only X but (also) Y"` coordination, case- and
/// newline-insensitively (`(?i)` folds case, `(?s)` lets `.` cross the
/// soft line breaks that can appear inside a single markdown-source
/// sentence): `"not"`, then `"just"` or `"only"`, then eventually `"but"`,
/// optionally followed by `"also"`. Built once, lazily, since compiling a
/// regex is not free and every one of these functions is called
/// repeatedly.
static NOT_JUST_BUT_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?is)\bnot\s+(?:just|only)\b.*?\bbut\b(?:\s+also\b)?")
        .expect("not-just-but pattern is a fixed, valid regex")
});

/// The canonical `(expanded, contracted)` pairs this workspace treats as
/// standard English contractions.
///
/// The same data [`contraction_ratio`] counts over, just handed back as
/// plain string pairs instead of the private [`ContractionPair`] shape it
/// is stored in internally. Exposed so other crates that need this *exact*
/// table — most notably
/// `friction-rules`' contraction-insertion rule, which turns one of these
/// `expanded` phrases back into its `contracted` form when a document
/// reads more formally than its genre's own writers typically do — read
/// the same data this metric does, rather than maintaining an independent
/// copy that could quietly drift out of sync with it. Returned in the
/// same order as the table's own declaration (sorted by `expanded`, ASCII
/// byte order; see [`CONTRACTION_PAIRS`]'s own doc comment).
#[must_use]
pub fn contraction_pairs() -> Vec<(&'static str, &'static str)> {
    CONTRACTION_PAIRS
        .iter()
        .map(|pair| (pair.expanded, pair.contracted))
        .collect()
}

/// Sentence-initial discourse-marker density, per 1000 word tokens.
///
/// Counts sentences whose text (after stripping leading whitespace and
/// markdown emphasis delimiters, see [`strip_leading_markup`]) starts with
/// one of [`DISCOURSE_MARKERS`], divides by the document's total word
/// token count (see the module docs for the tokenization rule), and scales
/// to a per-1000-token rate. Returns `0.0` for a document with no word
/// tokens.
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
/// [`contraction_ratio`] divides to produce its ratio, counted over every
/// [`CONTRACTION_PAIRS`] entry across the whole document.
///
/// For each sentence (not the whole paragraph — an `expanded` phrase must
/// not be allowed to match across a sentence boundary, e.g. the "is" that
/// ends one sentence and the "not" that opens the next must never count
/// as the `"is not"` -> `"isn't"` pair), [`word_tokens`] the sentence's
/// text once, then for every pair counts exact single-token matches of
/// `contracted` and exact consecutive-token matches of `expanded`'s words
/// (so a one-word expanded form like `"cannot"` and a multi-word one like
/// `"do not"` are counted the same way).
///
/// Exposed as its own function (rather than folded straight into
/// [`contraction_ratio`]) because the raw counts, not just their ratio,
/// are what a caller needs to work out the metric's *exact* per-occurrence
/// effect (`1 / (contracted + contractible)`) for a specific document —
/// most notably `friction-rules`' contraction-insertion rule, which reads
/// its own real document this same way rather than guessing that effect
/// from a fixed assumed document size (see that rule's module docs).
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
/// Returns `0.0` when neither a contracted nor a contractible form appears
/// anywhere in the document, rather than `NaN`.
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

/// Rate of paragraphs that open or close with a ritual transition phrase,
/// per paragraph.
///
/// A paragraph is a [`friction_core::ProseUnit`] with at least one
/// sentence; it counts as "flagged" if its first sentence or its last
/// sentence (which may be the same sentence, for a one-sentence
/// paragraph) starts with one of [`RITUAL_MARKERS`] (after stripping
/// leading whitespace and markdown emphasis delimiters, see
/// [`strip_leading_markup`] — this is what lets a bold-wrapped marker like
/// `"**Overall:** ..."` match). Returns `0.0` for a document with no
/// paragraphs that have any sentences.
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
/// Counts sentences whose text matches [`NOT_JUST_BUT_RE`] and divides by
/// the document's total sentence count. Returns `0.0` for a document with
/// no sentences.
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

/// Returns `true` if `text`, after stripping leading whitespace and
/// leading markdown emphasis delimiters (see [`strip_leading_markup`]),
/// starts with any phrase in `markers` (see the module docs for the exact
/// case/apostrophe/word-boundary rules).
fn starts_with_any(text: &str, markers: &[&str]) -> bool {
    let stripped = strip_leading_markup(text);
    markers
        .iter()
        .any(|marker| starts_with_marker(stripped, marker))
}

/// Strips leading whitespace and leading markdown emphasis/strong
/// delimiters (`*`/`_`) from `text`, repeating both strips in sequence
/// until a pass removes nothing further.
///
/// A sentence's source-sliced text keeps a bold- or italic-wrapped
/// marker's literal delimiter prefix — `friction-parse`'s prose
/// extraction bridges emphasis/strong runs into the surrounding sentence
/// rather than stripping them — so `"**Overall:**"` and `"__However__,"`
/// both need their `**`/`__` skipped before [`starts_with_marker`] can see
/// the word underneath. Repeating the whitespace-then-delimiter strip
/// (rather than doing each once) handles a delimiter run followed by more
/// whitespace, e.g. `"** Overall"` (a space inside the emphasis markers).
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
/// (ASCII case-folding; every marker table in this module is plain
/// ASCII), immediately followed by a non-alphanumeric character or the
/// end of `text`. A `'` in `marker` also matches the Unicode right single
/// quotation mark `’` in `text`.
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

/// Counts how many times the consecutive-token sequence `phrase` occurs in
/// `tokens` (every `tokens` window of `phrase.len()` that matches
/// `phrase`, element-wise, exactly — both sides are already lowercase).
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
// Every fixture below is a hand-computed exact value (see each test's doc
// comment for the arithmetic), so comparing with `==` rather than an
// epsilon is intentional, not a float-precision bug. And several
// fixtures deliberately use a one-sentence paragraph (`&[a..b]`), which
// clippy's heuristic misreads as `Range` literal syntax rather than a
// one-element array of ranges.
#[allow(clippy::float_cmp, clippy::single_range_in_vec_init)]
mod tests {
    use std::ops::Range;

    use friction_core::{Block, BlockKind, ProseUnit, Sentence};

    use super::*;

    /// `DISCOURSE_MARKERS` is sorted (ASCII byte order) with no
    /// duplicates, so the list documents its own invariant instead of
    /// relying on the author having gotten it right by eye.
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

    /// `contraction_pairs` hands back exactly [`CONTRACTION_PAIRS`], in
    /// the same order, just unwrapped from the private [`ContractionPair`]
    /// struct into plain tuples — the public accessor other crates (e.g.
    /// `friction-rules`) reuse instead of maintaining their own copy.
    #[test]
    fn contraction_pairs_matches_internal_table() {
        let pairs = contraction_pairs();
        assert_eq!(pairs.len(), CONTRACTION_PAIRS.len());
        for (public, internal) in pairs.iter().zip(CONTRACTION_PAIRS.iter()) {
            assert_eq!(*public, (internal.expanded, internal.contracted));
        }
    }

    /// Builds a single-block, single-paragraph document out of pre-cut
    /// sentence ranges, without going through `friction-parse` or
    /// `friction-nlp` — these fixtures are hand-computed byte offsets, so
    /// bypassing real segmentation keeps the expected values exact and
    /// independent of any other crate's behavior.
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

    /// Plain text with only leading whitespace is trimmed the same as
    /// `str::trim_start` would, with no markup to strip.
    #[test]
    fn strip_leading_markup_trims_whitespace_only() {
        assert_eq!(
            strip_leading_markup("  Overall it works."),
            "Overall it works."
        );
    }

    /// A single run of leading `**` (bold) is stripped down to the word
    /// underneath.
    #[test]
    fn strip_leading_markup_strips_bold_delimiters() {
        assert_eq!(
            strip_leading_markup("**Overall:** it works."),
            "Overall:** it works."
        );
    }

    /// A single run of leading `__` (the underscore strong-emphasis
    /// delimiter) is stripped the same way `**` is.
    #[test]
    fn strip_leading_markup_strips_underscore_delimiters() {
        assert_eq!(
            strip_leading_markup("__However__, it works."),
            "However__, it works."
        );
    }

    /// Whitespace and delimiter stripping repeat until neither removes
    /// anything further: `"** Overall"` has a space *inside* the leading
    /// `**`, so a single whitespace-then-delimiter pass would stop after
    /// stripping `**` and land on `" Overall"` — this checks the second
    /// pass strips that remaining leading space too.
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

    /// "However, it works." (3 tokens, sentence-initial "However") and
    /// "Fine." (1 token, no marker): 1 marked sentence over 4 total
    /// tokens, scaled to per-1000 = `1 * 1000 / 4 = 250.0` exactly.
    #[test]
    fn discourse_marker_density_hand_computed() {
        let source = "However, it works. Fine.";
        let doc = doc_from_sentences(source, &[0..18, 19..24]);
        assert_eq!(discourse_marker_density(&doc), 250.0);
    }

    /// Regression test for the bold-wrapped-marker detector bug: a
    /// sentence whose source text is `"**However**, it works."` (the
    /// literal form a markdown-source ritual/discourse marker takes once
    /// `friction-parse` bridges the emphasis delimiters into the sentence,
    /// see the module docs) must still be recognized as opening with
    /// "However" — 3 tokens ("However", "it", "works"; the bold markers
    /// are not `str::split_whitespace` tokens on their own, they cling to
    /// "However" and "works" respectively... actually the source is
    /// `"**However**, it works."`, whose whitespace-split tokens are
    /// `["**However**,", "it", "works."]`, i.e. 3 tokens), 1 marked
    /// sentence over 3 tokens = `1 * 1000 / 3`.
    #[test]
    fn discourse_marker_density_matches_bold_wrapped_marker() {
        let source = "**However**, it works.";
        let doc = doc_from_sentences(source, &[0..source.len()]);
        assert!((discourse_marker_density(&doc) - 1000.0 / 3.0).abs() < 1e-9);
    }

    /// The same bold-wrapped-marker recognition applies to the underscore
    /// strong-emphasis delimiter, not just `**`.
    #[test]
    fn discourse_marker_density_matches_underscore_wrapped_marker() {
        let source = "__However__, it works.";
        let doc = doc_from_sentences(source, &[0..source.len()]);
        assert!((discourse_marker_density(&doc) - 1000.0 / 3.0).abs() < 1e-9);
    }

    /// A bold-wrapped word that is *not* one of `DISCOURSE_MARKERS` must
    /// not spuriously match just because its emphasis delimiters got
    /// stripped — stripping markup does not loosen the marker-phrase
    /// comparison itself.
    #[test]
    fn discourse_marker_density_bold_non_marker_does_not_match() {
        let source = "**Something** happened.";
        let doc = doc_from_sentences(source, &[0..source.len()]);
        assert_eq!(discourse_marker_density(&doc), 0.0);
    }

    /// A marker word must be followed by a non-alphanumeric character to
    /// count: "Howeverish" is not "However" plus a word boundary, so this
    /// sentence contributes 0 marked sentences (density `0.0`), not 1.
    #[test]
    fn discourse_marker_density_requires_word_boundary() {
        let source = "Howeverish results came in.";
        let doc = doc_from_sentences(source, &[0..source.len()]);
        assert_eq!(discourse_marker_density(&doc), 0.0);
    }

    /// A document with no word tokens (empty sentence text) has density
    /// `0.0`, not `NaN` from a zero-over-zero division.
    #[test]
    fn discourse_marker_density_zero_tokens_is_zero() {
        let source = "";
        let doc = doc_from_sentences(source, &[0..0]);
        assert_eq!(discourse_marker_density(&doc), 0.0);
    }

    // --- contraction_ratio -------------------------------------------

    /// "Do not stop. Don't stop. It is fine. It's fine." tokenizes to
    /// `[do, not, stop, don't, stop, it, is, fine, it's, fine]`: one
    /// `"do not"` bigram and one `"don't"` token (pair 1), one `"it is"`
    /// bigram and one `"it's"` token (pair 2). contracted = 2,
    /// contractible = 2, ratio = `2 / (2 + 2) = 0.5` exactly.
    #[test]
    fn contraction_ratio_hand_computed() {
        let source = "Do not stop. Don't stop. It is fine. It's fine.";
        let doc = doc_from_sentences(source, &[0..source.len()]);
        assert_eq!(contraction_ratio(&doc), 0.5);
    }

    /// A document with neither a contracted nor a contractible form
    /// present has ratio `0.0`, not `NaN`.
    #[test]
    fn contraction_ratio_no_forms_is_zero() {
        let source = "Cats sit on mats quietly.";
        let doc = doc_from_sentences(source, &[0..source.len()]);
        assert_eq!(contraction_ratio(&doc), 0.0);
    }

    /// Regression test: an `expanded` pair must never match across a
    /// sentence boundary.
    ///
    /// "She does. Not enough is done. It isn't clear." is one paragraph
    /// with three sentences. Tokenized per sentence: `["she", "does"]`,
    /// `["not", "enough", "is", "done"]`, `["it", "isn't", "clear"]`.
    /// None of the three sentences, taken alone, contains a contractible
    /// `expanded` phrase — the only real contraction present is `"isn't"`
    /// in the third sentence, giving `contracted = 1`, `contractible = 0`,
    /// ratio = `1 / (1 + 0) = 1.0` exactly.
    ///
    /// Tokenizing the whole paragraph as one stream instead (the bug)
    /// would concatenate sentence 1's last token `"does"` directly
    /// against sentence 2's first token `"not"`, spuriously matching the
    /// `"does not"` -> `"doesn't"` pair's `expanded` bigram even though
    /// those words never formed a contractible construction — inflating
    /// `contractible` to 1 and deflating the ratio to `1 / (1 + 1) =
    /// 0.5`.
    #[test]
    fn contraction_ratio_does_not_match_across_sentence_boundary() {
        let source = "She does. Not enough is done. It isn't clear.";
        let doc = doc_from_sentences(source, &[0..9, 10..29, 30..45]);
        assert_eq!(contraction_ratio(&doc), 1.0);
    }

    // --- ritual_marker_rate --------------------------------------------

    /// Four paragraphs: "Fine here." (not flagged), "Overall it works."
    /// (opens with "Overall", flagged), "Good stuff." (not flagged), and
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

    /// A document with no paragraphs that have any sentences has rate
    /// `0.0`, not `NaN`.
    #[test]
    fn ritual_marker_rate_no_paragraphs_is_zero() {
        let doc = doc_from_paragraphs("", &[]);
        assert_eq!(ritual_marker_rate(&doc), 0.0);
    }

    /// Regression test for the exact bug reproduced against train doc
    /// `corpus/llm/blog/cde08357ef2e91de.md`: a one-sentence paragraph
    /// whose entire text is a bold-wrapped ritual marker,
    /// `"**Overall:**"` (the literal form `friction-parse` bridges into a
    /// sentence's source text, see the module docs), must be flagged —
    /// before the leading-markup strip, `starts_with_marker` saw a `*` as
    /// the first character and never matched "Overall" at all, silently
    /// undercounting this pattern.
    #[test]
    fn ritual_marker_rate_matches_bold_wrapped_marker() {
        let source = "**Overall:**";
        let doc = doc_from_paragraphs(source, &[(0..source.len(), &[0..source.len()])]);
        assert_eq!(ritual_marker_rate(&doc), 1.0);
    }

    /// The same bold-wrapped-marker recognition applies when the marker
    /// opens a longer sentence, and to the underscore delimiter too.
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
