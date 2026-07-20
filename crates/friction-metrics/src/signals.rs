//! Structural, mined-phrase, and sentence-opener "signal" metrics.
//!
//! These are the new-metric candidates the train-split error-analysis
//! brief argued for: none of the other three metric families
//! ([`crate::rhythm`], [`crate::lexical`], [`crate::symmetry`]) look at
//! document *structure* (headings, lists, bold spans) or at discriminative
//! *n-gram* register mined directly from the train corpus, and the brief's
//! qualitative read of train documents found those to be this corpus's
//! most visually obvious llm/human tell.
//!
//! Every public function here is a pure function of a
//! [`friction_core::Document`]: given the same document it returns the
//! same numbers, on any machine, on any run. None of them need a `Tagger`
//! — every signal in this module is either block-structural (headings,
//! list items) or shallow lexical (n-gram/unigram matching over sentence
//! text), so, like [`crate::lexical`], these walk `document.prose()` and
//! `document.blocks()` directly rather than tagging anything.
//!
//! # Tokenization
//!
//! [`word_tokens`] is a direct duplicate of [`crate::lexical`]'s own
//! private tokenizer (maximal runs of alphabetic characters, folding an
//! interior apostrophe — ASCII or the Unicode right single quotation mark
//! `’` — into the word, lowercased). It is duplicated rather than shared
//! deliberately: each metric-family module in this crate is a
//! self-contained set of pure functions over `Document` (see this crate's
//! module layout), and the mined-phrase metrics below specifically need
//! apostrophe-aware word tokens (`"here's"` must tokenize to one word to
//! match the mined pack's `here's` entry) — the same rule
//! [`crate::lexical`] already uses for its own contraction matching, for
//! the same reason.
//!
//! # Degenerate cases
//!
//! No function in this module ever returns `NaN` or `inf`: every "per
//! 1000 tokens" density is `0.0` for a document with zero word tokens, and
//! both sentence-opener metrics are `0.0` for a document with fewer than
//! the observations they need (see each function's own docs).

use std::collections::BTreeMap;
use std::sync::LazyLock;

use friction_core::{Block, BlockKind, Document};
use serde::Deserialize;

// ---------------------------------------------------------------------
// Shared tokenization
// ---------------------------------------------------------------------

/// Splits `text` into lowercase word tokens: maximal runs of alphabetic
/// characters, treating an interior apostrophe (ASCII `'` or the Unicode
/// right single quotation mark `’`, surrounded by alphabetic characters on
/// both sides) as part of the word. See the module docs' "Tokenization"
/// section for why this duplicates [`crate::lexical`]'s own tokenizer
/// rather than importing it.
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
/// `tokens` (every window of `phrase.len()` tokens that matches `phrase`
/// element-wise, exactly — both sides already lowercase). Mirrors
/// [`crate::lexical`]'s own `count_phrase_occurrences`, over an owned
/// `String` phrase rather than a `&str` slice, since the mined phrases are
/// parsed from a TOML pack at run time rather than written as `&'static
/// str` literals.
fn count_phrase_occurrences(tokens: &[String], phrase: &[String]) -> usize {
    if phrase.is_empty() || tokens.len() < phrase.len() {
        return 0;
    }
    tokens
        .windows(phrase.len())
        .filter(|window| window.iter().zip(phrase).all(|(t, p)| t == p))
        .count()
}

/// Total word-token count across every sentence in `document`, in source
/// order (walking `document.prose()` then each unit's `sentences`) — the
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

/// The embedded `mined-ngrams-v1` pack source, pulled in at compile time
/// from `friction-packs`' own pack directory — see that crate's
/// `packs/mined-ngrams-v1.toml` for the full curation rationale and
/// provenance. Embedding the raw TOML text (rather than depending on the
/// `friction-packs` crate) keeps this metric family's only coupling to
/// that pack a build-time file read, not a crate dependency edge.
const MINED_PACK_SOURCE: &str = include_str!("../../friction-packs/packs/mined-ngrams-v1.toml");

/// One curated n-gram entry from the mined pack. Only `ngram` is used
/// here; `z` and `category` (also present in the pack's TOML) are left
/// unparsed rather than declared as unused struct fields.
#[derive(Debug, Deserialize)]
struct MinedEntry {
    ngram: String,
}

/// The mined pack's shape, as relevant to this module: two ordered lists
/// of curated n-grams. The pack's `[pack]` metadata table is present in
/// the TOML but not modeled here — `toml`'s default (non-`deny_unknown_fields`)
/// deserialization simply ignores table keys a struct doesn't declare.
#[derive(Debug, Deserialize)]
struct MinedPack {
    llm_favored: Vec<MinedEntry>,
    human_favored: Vec<MinedEntry>,
}

/// The embedded pack, parsed once and reused for the life of the process.
///
/// # Panics
/// Panics if the embedded `mined-ngrams-v1.toml` fails to parse — that
/// would mean this crate shipped with a malformed pack file, a bug in this
/// crate's own embedded data (covered by this module's
/// `mined_pack_parses` test), not a condition any caller can recover from
/// by retrying.
static MINED_PACK: LazyLock<MinedPack> = LazyLock::new(|| {
    toml::from_str(MINED_PACK_SOURCE)
        .expect("embedded mined-ngrams-v1.toml must parse: see this module's tests")
});

/// Each of [`MINED_PACK`]'s llm-favored entries, pre-tokenized once via
/// [`word_tokens`] so every document's phrase matching only tokenizes its
/// own sentence text, not the pack's phrase list, on every call.
static LLM_FAVORED_PHRASES: LazyLock<Vec<Vec<String>>> = LazyLock::new(|| {
    MINED_PACK
        .llm_favored
        .iter()
        .map(|entry| word_tokens(&entry.ngram))
        .collect()
});

/// Each of [`MINED_PACK`]'s human-favored entries, pre-tokenized; see
/// [`LLM_FAVORED_PHRASES`].
static HUMAN_FAVORED_PHRASES: LazyLock<Vec<Vec<String>>> = LazyLock::new(|| {
    MINED_PACK
        .human_favored
        .iter()
        .map(|entry| word_tokens(&entry.ngram))
        .collect()
});

/// Total occurrences of every phrase in `phrases` across `document`, one
/// sentence at a time (a phrase match never crosses a sentence boundary,
/// the same rule [`crate::lexical::contraction_ratio`] uses for its own
/// expanded-form matching), scaled to a rate per 1000 word tokens.
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

/// Rate of curated llm-favored mined n-grams, per 1000 word tokens.
///
/// The n-grams come from `crates/friction-packs/packs/mined-ngrams-v1.
/// toml`'s `llm_favored` list, matched case-insensitively (word tokens are
/// lowercased) at word boundaries — phrase matching is over already-split
/// word tokens, so a phrase never matches a substring of a longer word.
#[must_use]
pub fn llm_favored_phrase_rate(document: &Document) -> f64 {
    phrase_rate(document, &LLM_FAVORED_PHRASES)
}

/// Rate of curated human-favored mined n-grams (the same pack's
/// `human_favored` list), per 1000 word tokens. See
/// [`llm_favored_phrase_rate`] for the matching rule.
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

/// `true` if `block` is a single list item within a markdown list.
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

/// Heading-block density: the count of [`friction_core::BlockKind::Heading`]
/// blocks (any level) in `document`, per 1000 word tokens.
#[must_use]
pub fn heading_density(document: &Document) -> f64 {
    block_kind_density(document, is_heading)
}

/// List-item-block density: the count of
/// [`friction_core::BlockKind::ListItem`] blocks in `document` — every
/// item of every list, top-level or nested — per 1000 word tokens.
#[must_use]
pub fn list_item_density(document: &Document) -> f64 {
    block_kind_density(document, is_list_item)
}

/// Counts bold/strong-emphasis spans in `text` by counting delimiter
/// occurrences and halving: every `**...**` or `__...__` span contributes
/// exactly two delimiter occurrences (one opening, one closing).
///
/// This is a documented approximation, not a structural parse: `friction-
/// parse`'s prose extraction deliberately *bridges* emphasis/strong
/// delimiter bytes into a sentence's literal text rather than stripping
/// them (see `friction-parse::extract`'s module docs), so a bold span's
/// `**`/`__` markup survives in the text this function scans — but this
/// function does not distinguish `**bold**` from a stray, unpaired `**`
/// (e.g. one delimiter split across two sentences by segmentation), so an
/// odd total delimiter count silently drops its last (unpaired) occurrence
/// via integer division rather than over- or under-counting by a whole
/// span.
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
/// (lowercased), or `None` if `text` has no alphabetic word at all (e.g. a
/// sentence that is pure punctuation).
fn leading_unigram(text: &str) -> Option<String> {
    word_tokens(text).into_iter().next()
}

/// The leading unigram of every sentence in `document`, in source order
/// (walking `document.prose()` then each unit's `sentences`) — one entry
/// per sentence, `None` where [`leading_unigram`] found no word.
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

/// Fraction of `document`'s sentences (every sentence but the first) whose
/// leading unigram (see [`leading_unigram`]) equals the immediately
/// preceding sentence's.
///
/// The denominator is always `sentence_count - 1` regardless of whether
/// either sentence in a pair has a detectable opener; a pair where either
/// side lacks one simply cannot match (there is nothing to repeat), so it
/// counts against the rate rather than being excluded from it. Returns
/// `0.0` for a document with fewer than two sentences.
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
/// sentence openers in `document`.
///
/// Among every sentence that has a detectable leading unigram (see
/// [`leading_unigram`]), this is the largest per-word count divided by
/// that total. Sentences with no detectable opener are excluded from both
/// the tally and its denominator — they contribute no "which word opens
/// most often" signal either way. Returns `0.0` if no sentence in
/// `document` has a detectable opener.
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
// Every fixture below is a hand-computed exact value (see each test's doc
// comment), so `==` rather than an epsilon is correct wherever the
// arithmetic terminates exactly; a few densities divide into a repeating
// fraction and use an epsilon instead, noted at the call site.
#[allow(clippy::float_cmp, clippy::single_range_in_vec_init)]
mod tests {
    use std::ops::Range;

    use friction_core::{Block, BlockKind, ProseUnit, Sentence};

    use super::*;

    const EPSILON: f64 = 1e-9;

    /// Builds a single-block, single-paragraph, single-sentence document
    /// spanning the whole of `source` — entirely through `friction-core`'s
    /// own constructors, independent of `friction-parse`/`friction-nlp`.
    fn doc_single_sentence(source: &str) -> Document {
        let block = Block::new(BlockKind::Paragraph, 0..source.len());
        let sentence = Sentence::new(0..source.len(), Vec::new());
        let prose = ProseUnit::new(0, 0..source.len(), vec![sentence]);
        Document::new(source, vec![block], vec![prose]).expect("hand-built fixture is well-formed")
    }

    /// Builds a single-block, single-paragraph document out of pre-cut
    /// sentence ranges within `source` — mirrors `crate::lexical`'s own
    /// test helper of the same shape.
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

    /// The embedded pack parses, and both curated lists are non-empty —
    /// this is the same guarantee `MINED_PACK`'s own `LazyLock::expect`
    /// makes at first use, exercised eagerly here so a malformed pack
    /// fails a test rather than only a later, unrelated call.
    #[test]
    fn mined_pack_parses() {
        assert!(!MINED_PACK.llm_favored.is_empty());
        assert!(!MINED_PACK.human_favored.is_empty());
    }

    // -------------------------------------------------------------
    // llm_favored_phrase_rate / human_favored_phrase_rate
    // -------------------------------------------------------------

    /// "Your plan is good." tokenizes to `[your, plan, is, good]` (4
    /// tokens). `"your"` is the pack's top llm-favored unigram; `"good"`
    /// is a human-favored unigram; neither list contains any bigram/
    /// trigram formed by this sentence's other adjacent tokens. One match
    /// each, over 4 tokens: rate `= 1 * 1000 / 4 = 250.0` exactly, for
    /// both metrics.
    #[test]
    fn phrase_rates_hand_computed() {
        let doc = doc_single_sentence("Your plan is good.");
        assert_eq!(llm_favored_phrase_rate(&doc), 250.0);
        assert_eq!(human_favored_phrase_rate(&doc), 250.0);
    }

    /// A document with no word tokens at all has both rates `0.0`, not
    /// `NaN` from a zero-over-zero division.
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
    /// now.", 4 word tokens: body, text, here, now); no list items, no
    /// bold spans. `heading_density = 1 * 1000 / 4 = 250.0` exactly;
    /// `list_item_density` and `bold_span_density` are both `0.0`.
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
    /// word tokens each, 6 total) under one list block. `list_item_density
    /// = 2 * 1000 / 6 = 333.333...` (repeating, epsilon comparison);
    /// `heading_density` is `0.0`.
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
    /// delimiter occurrences, halved) over 5 word tokens (the, bold, word,
    /// appears, here). `1 * 1000 / 5 = 200.0` exactly.
    #[test]
    fn bold_span_density_hand_computed() {
        let doc = doc_single_sentence("The **bold** word appears here.");
        assert_eq!(bold_span_density(&doc), 200.0);
    }

    /// A document with no bold markup at all has density `0.0`.
    #[test]
    fn bold_span_density_zero_for_no_bold() {
        let doc = doc_single_sentence("Nothing bold in here at all.");
        assert_eq!(bold_span_density(&doc), 0.0);
    }

    // -------------------------------------------------------------
    // sentence_opener_repeat_rate / top_opener_concentration
    // -------------------------------------------------------------

    /// Three sentences: "Overall it works." (opener "overall"), "Overall
    /// it scales." (opener "overall", matches the previous sentence),
    /// "Fine so far." (opener "fine", does not match). 1 matching
    /// consecutive pair out of 2: `sentence_opener_repeat_rate = 0.5`
    /// exactly. Opener counts: overall = 2, fine = 1, over 3 total
    /// openers: `top_opener_concentration = 2/3`.
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
    /// neither matches nor is counted as an opener: "..." (no opener),
    /// "Yes it works." (opener "yes"). The one consecutive pair has a
    /// `None` on one side, so it cannot match: `sentence_opener_repeat_rate
    /// = 0.0`. Only one sentence has a detectable opener at all, so
    /// `top_opener_concentration = 1 / 1 = 1.0`.
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
    /// `sentence_opener_repeat_rate` `0.0` (no consecutive pair exists),
    /// and `top_opener_concentration` `0.0` for zero sentences or `1.0`
    /// for exactly one sentence with a detectable opener — never `NaN`.
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
