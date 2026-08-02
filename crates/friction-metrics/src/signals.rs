//! Structural, mined-phrase, and sentence-opener "signal" metrics.
//!
//! New-metric candidates the train-split error-analysis brief argued for:
//! the other families ([`crate::rhythm`], [`crate::lexical`],
//! [`crate::symmetry`]) don't look at document *structure* or
//! discriminative *n-gram* register mined from the train corpus, a
//! qualitative read found those the most visually obvious llm/human tell.
//!
//! Every public function is a pure function of a
//! [`friction_core::Document`]; none need a `Tagger`.
//!
//! # Tokenization
//!
//! [`word_tokens`] duplicates [`crate::lexical`]'s private tokenizer: the
//! mined-phrase metrics need apostrophe-aware tokens so `"here's"`
//! tokenizes to one word, matching the pack's entries — the same rule
//! [`crate::lexical`] uses for contraction matching.
//!
//! # Degenerate cases
//!
//! No function here returns `NaN` or `inf`: densities are `0.0` for zero
//! word tokens; sentence-opener metrics are `0.0` for too few
//! observations (see each function's docs).

use std::collections::BTreeMap;
use std::sync::LazyLock;

use friction_core::{Block, BlockKind, Document};
use serde::Deserialize;

// ---------------------------------------------------------------------
// Shared tokenization
// ---------------------------------------------------------------------

/// Splits `text` into lowercase word tokens: maximal alphabetic runs,
/// treating an interior apostrophe (ASCII `'` or `’`, between alphabetic
/// characters) as part of the word. See the module docs' "Tokenization"
/// section for why this duplicates [`crate::lexical`]'s tokenizer.
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

/// Counts consecutive-token window matches of `phrase` in `tokens` (exact
/// match, both sides already lowercase). Mirrors [`crate::lexical`]'s
/// `count_phrase_occurrences`, but over an owned `String` — mined phrases
/// parse from TOML at run time, not `&'static str` literals.
fn count_phrase_occurrences(tokens: &[String], phrase: &[String]) -> usize {
    if phrase.is_empty() || tokens.len() < phrase.len() {
        return 0;
    }
    tokens
        .windows(phrase.len())
        .filter(|window| window.iter().zip(phrase).all(|(t, p)| t == p))
        .count()
}

/// Total word-token count across every sentence in `document` — the
/// shared denominator for every "per 1000 tokens" density in this module.
fn total_word_tokens(document: &Document) -> u64 {
    document
        .prose()
        .iter()
        .flat_map(|unit| &unit.sentences)
        .filter_map(|sentence| document.text(&sentence.range).ok())
        .map(|text| word_tokens(text).len() as u64)
        .sum()
}

/// `occurrences` scaled to a rate per 1000 word tokens in `document` (see
/// [`total_word_tokens`]). `0.0` for a document with no word tokens, never
/// `NaN`.
fn density_per_1000_tokens(document: &Document, occurrences: u64) -> f64 {
    let total = total_word_tokens(document);
    if total == 0 {
        return 0.0;
    }
    #[allow(clippy::cast_precision_loss)]
    let density = occurrences as f64 * 1000.0 / total as f64;
    density
}

// ---------------------------------------------------------------------
// Mined-phrase densities
// ---------------------------------------------------------------------

/// Embedded `mined-ngrams-v1` pack source, pulled in at compile time —
/// see `friction-packs/packs/mined-ngrams-v1.toml` for curation rationale.
/// Embedding raw TOML (vs. depending on `friction-packs`) keeps this
/// family's only coupling to that pack a build-time file read, not a
/// crate dependency.
const MINED_PACK_SOURCE: &str = include_str!("../../friction-packs/packs/mined-ngrams-v1.toml");

/// One curated n-gram entry from the mined pack. Only `ngram` is used;
/// `z` and `category` are left unparsed rather than declared unused.
#[derive(Debug, Deserialize)]
struct MinedEntry {
    ngram: String,
}

/// The mined pack's shape, as relevant here: two ordered lists of
/// n-grams. The `[pack]` metadata table is in the TOML but not modeled —
/// `toml`'s default (non-`deny_unknown_fields`) deserialization ignores
/// undeclared keys.
#[derive(Debug, Deserialize)]
struct MinedPack {
    llm_favored: Vec<MinedEntry>,
    human_favored: Vec<MinedEntry>,
}

/// The embedded pack, parsed once and reused for the life of the process.
///
/// # Panics
/// Panics if the embedded `mined-ngrams-v1.toml` fails to parse — a
/// malformed pack shipped with this crate (see `mined_pack_parses`), not
/// something a caller can fix by retrying.
static MINED_PACK: LazyLock<MinedPack> = LazyLock::new(|| {
    toml::from_str(MINED_PACK_SOURCE)
        .expect("embedded mined-ngrams-v1.toml must parse: see this module's tests")
});

/// [`MINED_PACK`]'s llm-favored entries, pre-tokenized once via
/// [`word_tokens`] so matching only tokenizes each document's sentence
/// text, not the phrase list, per call.
static LLM_FAVORED_PHRASES: LazyLock<Vec<Vec<String>>> = LazyLock::new(|| {
    MINED_PACK
        .llm_favored
        .iter()
        .map(|entry| word_tokens(&entry.ngram))
        .collect()
});

/// [`MINED_PACK`]'s human-favored entries, pre-tokenized; see
/// [`LLM_FAVORED_PHRASES`].
static HUMAN_FAVORED_PHRASES: LazyLock<Vec<Vec<String>>> = LazyLock::new(|| {
    MINED_PACK
        .human_favored
        .iter()
        .map(|entry| word_tokens(&entry.ngram))
        .collect()
});

/// Total occurrences of every phrase in `phrases` in `document`, per
/// sentence (matches never cross a sentence boundary, per
/// [`crate::lexical::contraction_ratio`]), scaled per 1000 word tokens.
fn phrase_rate(document: &Document, phrases: &[Vec<String>]) -> f64 {
    let mut matches = 0u64;
    for unit in document.prose() {
        for sentence in &unit.sentences {
            let Ok(text) = document.text(&sentence.range) else {
                continue;
            };
            let tokens = word_tokens(text);
            for phrase in phrases {
                matches += count_phrase_occurrences(&tokens, phrase) as u64;
            }
        }
    }
    density_per_1000_tokens(document, matches)
}

/// Rate of llm-favored n-grams, per 1000 word tokens.
///
/// From `crates/friction-packs/packs/mined-ngrams-v1.toml`'s
/// `llm_favored` list, matched case-insensitively at word boundaries: a
/// phrase never matches inside a longer word.
#[must_use]
pub fn llm_favored_phrase_rate(document: &Document) -> f64 {
    phrase_rate(document, &LLM_FAVORED_PHRASES)
}

/// Rate of human-favored n-grams (the pack's `human_favored` list), per
/// 1000 word tokens. See [`llm_favored_phrase_rate`] for the matching
/// rule.
#[must_use]
pub fn human_favored_phrase_rate(document: &Document) -> f64 {
    phrase_rate(document, &HUMAN_FAVORED_PHRASES)
}

// ---------------------------------------------------------------------
// Structural densities
// ---------------------------------------------------------------------

/// `true` if `block` is an ATX or setext heading, any level.
const fn is_heading(block: &Block) -> bool {
    matches!(block.kind, BlockKind::Heading { .. })
}

/// `true` if `block` is a single list item.
const fn is_list_item(block: &Block) -> bool {
    matches!(block.kind, BlockKind::ListItem)
}

/// The count of `document.blocks()` entries matching `predicate`, scaled
/// to a rate per 1000 word tokens (see [`density_per_1000_tokens`]).
fn block_kind_density(document: &Document, predicate: fn(&Block) -> bool) -> f64 {
    let count = document
        .blocks()
        .iter()
        .filter(|block| predicate(block))
        .count() as u64;
    density_per_1000_tokens(document, count)
}

/// Heading-block density: [`friction_core::BlockKind::Heading`] blocks
/// (any level) in `document`, per 1000 word tokens.
#[must_use]
pub fn heading_density(document: &Document) -> f64 {
    block_kind_density(document, is_heading)
}

/// List-item-block density: [`friction_core::BlockKind::ListItem`] blocks
/// in `document` — every list item, top-level or nested — per 1000 word
/// tokens.
#[must_use]
pub fn list_item_density(document: &Document) -> f64 {
    block_kind_density(document, is_list_item)
}

/// Counts bold/strong-emphasis spans in `text` by counting delimiters
/// and halving: each span contributes two delimiter occurrences.
///
/// An approximation, not a structural parse: `friction-parse` deliberately
/// *bridges* emphasis delimiter bytes into sentence text rather than
/// stripping them, so this can't distinguish `**bold**` from a stray
/// unpaired `**` (e.g. split across sentences by segmentation): an odd
/// count drops its last occurrence via integer division rather than
/// mis-counting a whole span.
fn count_strong_delimiter_spans(text: &str) -> usize {
    let double_star = text.matches("**").count();
    let double_underscore = text.matches("__").count();
    usize::midpoint(double_star, double_underscore)
}

/// Bold/strong-emphasis span density: [`count_strong_delimiter_spans`]
/// summed over every sentence in `document`, per 1000 word tokens.
#[must_use]
pub fn bold_span_density(document: &Document) -> f64 {
    let mut spans = 0u64;
    for unit in document.prose() {
        for sentence in &unit.sentences {
            let Ok(text) = document.text(&sentence.range) else {
                continue;
            };
            spans += count_strong_delimiter_spans(text) as u64;
        }
    }
    density_per_1000_tokens(document, spans)
}

// ---------------------------------------------------------------------
// Sentence-opener uniformity
// ---------------------------------------------------------------------

/// The leading unigram of `text`: its first [`word_tokens`] entry
/// (lowercase), or `None` if `text` has no alphabetic word, e.g. pure
/// punctuation.
fn leading_unigram(text: &str) -> Option<String> {
    word_tokens(text).into_iter().next()
}

/// The leading unigram of every sentence in `document`, in source order —
/// one entry per sentence, `None` where [`leading_unigram`] found no word.
fn sentence_leading_unigrams(document: &Document) -> Vec<Option<String>> {
    document
        .prose()
        .iter()
        .flat_map(|unit| &unit.sentences)
        .map(|sentence| {
            document
                .text(&sentence.range)
                .ok()
                .and_then(leading_unigram)
        })
        .collect()
}

/// Fraction of `document`'s sentences (all but the first) whose leading
/// unigram (see [`leading_unigram`]) equals the preceding one's.
///
/// Denominator is always `sentence_count - 1` regardless of whether either
/// side of a pair has a detectable opener — a pair missing one can't
/// match, so it counts against the rate rather than being excluded.
/// `0.0` for fewer than two sentences.
#[must_use]
pub fn sentence_opener_repeat_rate(document: &Document) -> f64 {
    let openers = sentence_leading_unigrams(document);
    if openers.len() < 2 {
        return 0.0;
    }
    let matches = openers
        .windows(2)
        .filter(|pair| matches!((&pair[0], &pair[1]), (Some(a), Some(b)) if a == b))
        .count() as u64;
    #[allow(clippy::cast_precision_loss)]
    let total = (openers.len() - 1) as f64;
    #[allow(clippy::cast_precision_loss)]
    let rate = matches as f64 / total;
    rate
}

/// The most common sentence-leading unigram's share of all detected
/// openers in `document`.
///
/// Among sentences with a detectable leading unigram (see
/// [`leading_unigram`]), this is the largest per-word count over that
/// total. Sentences with no detectable opener are excluded from both
/// tally and denominator. `0.0` if none has a detectable opener.
#[must_use]
pub fn top_opener_concentration(document: &Document) -> f64 {
    let openers: Vec<String> = sentence_leading_unigrams(document)
        .into_iter()
        .flatten()
        .collect();
    if openers.is_empty() {
        return 0.0;
    }
    let mut counts: BTreeMap<String, u64> = BTreeMap::new();
    for opener in &openers {
        *counts.entry(opener.clone()).or_insert(0) += 1;
    }
    let max = counts.values().copied().max().unwrap_or(0);
    #[allow(clippy::cast_precision_loss)]
    let concentration = max as f64 / openers.len() as f64;
    concentration
}

#[cfg(test)]
// Every fixture below is a hand-computed exact value, so `==` is correct
// where the arithmetic terminates exactly; a few densities repeat and use
// an epsilon instead, noted at the call site.
#[allow(clippy::float_cmp, clippy::single_range_in_vec_init)]
mod tests {
    use std::ops::Range;

    use friction_core::{Block, BlockKind, ProseUnit, Sentence};

    use super::*;

    const EPSILON: f64 = 1e-9;

    /// Builds a single-block, single-paragraph, single-sentence document
    /// spanning `source` — through `friction-core`'s own constructors,
    /// independent of `friction-parse`/`friction-nlp`.
    fn doc_single_sentence(source: &str) -> Document {
        let block = Block::new(BlockKind::Paragraph, 0..source.len());
        let sentence = Sentence::new(0..source.len(), Vec::new());
        let prose = ProseUnit::new(0, 0..source.len(), vec![sentence]);
        Document::new(source, vec![block], vec![prose]).expect("hand-built fixture is well-formed")
    }

    /// Builds a single-block, single-paragraph document out of pre-cut
    /// sentence ranges in `source` — mirrors `crate::lexical`'s own test
    /// helper.
    fn doc_from_sentences(source: &str, sentence_ranges: &[Range<usize>]) -> Document {
        let sentences = sentence_ranges
            .iter()
            .cloned()
            .map(|range| Sentence::new(range, Vec::new()))
            .collect();
        let block = Block::new(BlockKind::Paragraph, 0..source.len());
        let prose = ProseUnit::new(0, 0..source.len(), sentences);
        Document::new(source, vec![block], vec![prose]).expect("hand-built fixture is well-formed")
    }

    // -------------------------------------------------------------
    // Mined pack itself
    // -------------------------------------------------------------

    /// The embedded pack parses, and both lists are non-empty — the same
    /// guarantee `MINED_PACK`'s `LazyLock::expect` makes at first use,
    /// exercised eagerly so a malformed pack fails a test, not a later
    /// unrelated call.
    #[test]
    fn mined_pack_parses() {
        assert!(!MINED_PACK.llm_favored.is_empty());
        assert!(!MINED_PACK.human_favored.is_empty());
    }

    // -------------------------------------------------------------
    // llm_favored_phrase_rate / human_favored_phrase_rate
    // -------------------------------------------------------------

    /// "Your plan is good." tokenizes to `[your, plan, is, good]` (4
    /// tokens): `"your"` is the top llm-favored unigram, `"good"` a
    /// human-favored one, and no bigram/trigram matches. One match each:
    /// `1 * 1000 / 4 = 250.0`, both metrics.
    #[test]
    fn phrase_rates_hand_computed() {
        let doc = doc_single_sentence("Your plan is good.");
        assert_eq!(llm_favored_phrase_rate(&doc), 250.0);
        assert_eq!(human_favored_phrase_rate(&doc), 250.0);
    }

    /// A document with no word tokens has both rates `0.0`, not `NaN`
    /// from a zero-over-zero division.
    #[test]
    fn phrase_rates_zero_tokens_is_zero() {
        let doc = Document::new("", Vec::new(), Vec::new()).expect("empty document is valid");
        assert_eq!(llm_favored_phrase_rate(&doc), 0.0);
        assert_eq!(human_favored_phrase_rate(&doc), 0.0);
    }

    // -------------------------------------------------------------
    // heading_density / list_item_density
    // -------------------------------------------------------------

    /// One heading block ("Overview") plus one paragraph ("Body text here
    /// now.", 4 tokens: body, text, here, now). `heading_density = 1 *
    /// 1000 / 4 = 250.0`; `list_item_density`/`bold_span_density` are
    /// `0.0`.
    #[test]
    fn heading_density_hand_computed() {
        let heading = "Overview";
        let body = "Body text here now.";
        let source = format!("{heading}\n\n{body}");
        let heading_range = 0..heading.len();
        let body_start = heading.len() + 2;
        let body_range = body_start..(body_start + body.len());

        let blocks = vec![
            Block::new(BlockKind::Heading { level: 1 }, heading_range.clone()),
            Block::new(BlockKind::Paragraph, body_range.clone()),
        ];
        let prose = vec![
            ProseUnit::new(0, heading_range, Vec::new()),
            ProseUnit::new(
                1,
                body_range.clone(),
                vec![Sentence::new(body_range, Vec::new())],
            ),
        ];
        let doc = Document::new(source.as_str(), blocks, prose)
            .expect("hand-built fixture is well-formed");

        assert_eq!(heading_density(&doc), 250.0);
        assert_eq!(list_item_density(&doc), 0.0);
        assert_eq!(bold_span_density(&doc), 0.0);
    }

    /// Two list items ("Configure the server", "Restart the service"; 3
    /// tokens each, 6 total). `list_item_density = 2 * 1000 / 6 =
    /// 333.333...` (repeating, epsilon); `heading_density` is `0.0`.
    #[test]
    fn list_item_density_hand_computed() {
        let item1 = "Configure the server";
        let item2 = "Restart the service";
        let source = format!("{item1}\n{item2}");
        let item1_range = 0..item1.len();
        let item2_start = item1.len() + 1;
        let item2_range = item2_start..(item2_start + item2.len());
        let list_range = 0..item2_range.end;

        let blocks = vec![
            Block::new(
                BlockKind::List {
                    ordered: false,
                    start: None,
                },
                list_range,
            ),
            Block::new(BlockKind::ListItem, item1_range.clone()),
            Block::new(BlockKind::ListItem, item2_range.clone()),
        ];
        let prose = vec![
            ProseUnit::new(
                1,
                item1_range.clone(),
                vec![Sentence::new(item1_range, Vec::new())],
            ),
            ProseUnit::new(
                2,
                item2_range.clone(),
                vec![Sentence::new(item2_range, Vec::new())],
            ),
        ];
        let doc = Document::new(source.as_str(), blocks, prose)
            .expect("hand-built fixture is well-formed");

        let expected = 2.0 * 1000.0 / 6.0;
        assert!((list_item_density(&doc) - expected).abs() < EPSILON);
        assert_eq!(heading_density(&doc), 0.0);
    }

    // -------------------------------------------------------------
    // bold_span_density
    // -------------------------------------------------------------

    /// "The **bold** word appears here." has one `**...**` span (two `**`
    /// occurrences, halved) over 5 tokens (the, bold, word, appears,
    /// here). `1 * 1000 / 5 = 200.0`.
    #[test]
    fn bold_span_density_hand_computed() {
        let doc = doc_single_sentence("The **bold** word appears here.");
        assert_eq!(bold_span_density(&doc), 200.0);
    }

    /// A document with no bold markup has density `0.0`.
    #[test]
    fn bold_span_density_zero_for_no_bold() {
        let doc = doc_single_sentence("Nothing bold in here at all.");
        assert_eq!(bold_span_density(&doc), 0.0);
    }

    // -------------------------------------------------------------
    // sentence_opener_repeat_rate / top_opener_concentration
    // -------------------------------------------------------------

    /// Three sentences: "Overall it works." (opener "overall"), "Overall
    /// it scales." (opener "overall", matches previous), "Fine so far."
    /// (opener "fine", no match). 1 matching pair of 2:
    /// `sentence_opener_repeat_rate = 0.5`. Opener counts: overall = 2,
    /// fine = 1, of 3 total: `top_opener_concentration = 2/3`.
    #[test]
    fn opener_metrics_hand_computed() {
        let s1 = "Overall it works.";
        let s2 = "Overall it scales.";
        let s3 = "Fine so far.";
        let source = format!("{s1} {s2} {s3}");
        let s1_range = 0..s1.len();
        let s2_start = s1_range.end + 1;
        let s2_range = s2_start..(s2_start + s2.len());
        let s3_start = s2_range.end + 1;
        let s3_range = s3_start..(s3_start + s3.len());

        let doc = doc_from_sentences(&source, &[s1_range, s2_range, s3_range]);

        assert_eq!(sentence_opener_repeat_rate(&doc), 0.5);
        assert!((top_opener_concentration(&doc) - (2.0 / 3.0)).abs() < EPSILON);
    }

    /// A sentence with no detectable leading unigram (pure punctuation)
    /// neither matches nor counts as an opener: "..." (no opener), "Yes it
    /// works." (opener "yes"). The pair has `None` on one side, so it
    /// can't match: `sentence_opener_repeat_rate = 0.0`. Only one sentence
    /// has a detectable opener: `top_opener_concentration = 1 / 1 = 1.0`.
    #[test]
    fn opener_metrics_handle_sentence_with_no_leading_word() {
        let s1 = "...";
        let s2 = "Yes it works.";
        let source = format!("{s1} {s2}");
        let s1_range = 0..s1.len();
        let s2_start = s1_range.end + 1;
        let s2_range = s2_start..(s2_start + s2.len());

        let doc = doc_from_sentences(&source, &[s1_range, s2_range]);

        assert_eq!(sentence_opener_repeat_rate(&doc), 0.0);
        assert_eq!(top_opener_concentration(&doc), 1.0);
    }

    /// A document with fewer than two sentences has
    /// `sentence_opener_repeat_rate` `0.0` (no pair exists), and
    /// `top_opener_concentration` `0.0` for zero sentences or `1.0` for
    /// one sentence with a detectable opener — never `NaN`.
    #[test]
    fn opener_metrics_degenerate_cases() {
        let empty = Document::new("", Vec::new(), Vec::new()).expect("empty document is valid");
        assert_eq!(sentence_opener_repeat_rate(&empty), 0.0);
        assert_eq!(top_opener_concentration(&empty), 0.0);

        let single = doc_single_sentence("Only one sentence lives here.");
        assert_eq!(sentence_opener_repeat_rate(&single), 0.0);
        assert_eq!(top_opener_concentration(&single), 1.0);
    }
}
