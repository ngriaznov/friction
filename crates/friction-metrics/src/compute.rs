//! The document → [`MetricVector`] compute boundary: wires the rhythm,
//! lexical, symmetry, and signals metric families into one vector, for a
//! whole document or per paragraph.
//!
//! [`compute`] and [`compute_by_paragraph`] take a [`Document`] from
//! `friction-parse::parse` (structure extracted, not yet
//! sentence-segmented) plus a [`Segmenter`] and a [`Tagger`], and return a
//! complete vector: callers never segment manually or track which field
//! belongs to which family.

use friction_core::{Block, Document, MetricVector, ProseUnit};
use friction_nlp::{Segmenter, Tagger, segment_document};

use crate::lexical::{
    contraction_ratio, discourse_marker_density, not_just_but_rate, ritual_marker_rate,
};
use crate::rhythm::{
    em_dash_density, paragraph_shape, semicolon_density, sentence_length_by_document,
};
use crate::signals::{
    bold_span_density_with_total, heading_density_with_total, human_favored_phrase_rate_with_total,
    list_item_density_with_total, llm_favored_phrase_rate_with_total, sentence_opener_repeat_rate,
    top_opener_concentration, total_word_tokens,
};
use crate::symmetry::{
    bullet_parallelism, participial_closer_rate_from_tagged, tagged_sentences,
    triad_rate_from_tagged,
};

/// Computes the full 21-field [`MetricVector`] for `document`, document-wide.
///
/// Segments `document` with `segmenter` first (`document` itself is left
/// untouched), then runs every metric family's document-level function over
/// the result — see [`compute_segmented`] for which field comes from which
/// family.
///
/// # Panics
/// Panics if `segmenter` produces a sentence range that escapes its prose
/// unit or the document's bounds — a bug in the `Segmenter` implementation
/// itself; every implementation `friction-nlp` ships is well-behaved.
#[must_use]
pub fn compute(
    document: &Document,
    segmenter: &dyn Segmenter,
    tagger: &dyn Tagger,
) -> MetricVector {
    let with_sentences = segment_document(document, segmenter).expect(
        "a well-behaved Segmenter never produces a sentence range escaping its prose unit or \
         the document's bounds",
    );
    compute_segmented(&with_sentences, tagger)
}

/// Computes one [`MetricVector`] per paragraph of `document`, in source
/// order.
///
/// A paragraph is a [`friction_core::ProseUnit`] that segments to at least
/// one sentence; one with none (e.g. a heading) contributes no entry — same
/// convention as [`crate::rhythm::paragraph_shape`].
///
/// Each vector is [`compute_segmented`] run on a single-paragraph
/// sub-document, so every family's document-wide definition also serves,
/// unmodified, as its paragraph-scoped one, no separate per-paragraph
/// implementation to keep in sync. `paragraph_shape_mean`/`_cv` degenerate
/// accordingly: one paragraph-shape observation, so mean is that
/// paragraph's sentence count and cv is `0.0` (the single-observation
/// convention — see [`crate::rhythm::RhythmStats`]).
///
/// # Panics
/// Panics under the same well-behaved-`Segmenter` condition as [`compute`].
#[must_use]
pub fn compute_by_paragraph(
    document: &Document,
    segmenter: &dyn Segmenter,
    tagger: &dyn Tagger,
) -> Vec<MetricVector> {
    let with_sentences = segment_document(document, segmenter).expect(
        "a well-behaved Segmenter never produces a sentence range escaping its prose unit or \
         the document's bounds",
    );
    with_sentences
        .prose()
        .iter()
        .filter(|unit| !unit.sentences.is_empty())
        .map(|unit| compute_segmented(&single_paragraph_document(&with_sentences, unit), tagger))
        .collect()
}

/// The shared core of [`compute`] and [`compute_by_paragraph`]: runs every
/// metric family's document-level function over an already-segmented
/// `document` and assembles the result into one [`MetricVector`]. Each
/// field below is written by exactly one family's call.
///
/// `triad_rate`/`participial_closer_rate` and the five `_with_total`
/// density calls each depend on a full pass over `document` (a
/// perceptron-tagging pass, a word-tokenization pass) that's identical
/// across every metric that needs it. Rather than let each of those seven
/// `pub fn`s in [`crate::symmetry`]/[`crate::signals`] repeat its own pass,
/// this function runs it once — [`tagged_sentences`] and
/// [`total_word_tokens`] — and threads the shared result into the
/// `_from_tagged`/`_with_total` variant of each.
fn compute_segmented(document: &Document, tagger: &dyn Tagger) -> MetricVector {
    let sentence_stats = sentence_length_by_document(document);
    let shape_stats = paragraph_shape(document);
    let source = document.source();
    let pos_tagged_sentences = tagged_sentences(document, tagger);
    let total_tokens = total_word_tokens(document);
    MetricVector {
        sentence_length_mean: sentence_stats.mean,
        sentence_length_stddev: sentence_stats.stddev,
        sentence_length_cv: sentence_stats.cv,
        discourse_marker_density: discourse_marker_density(document),
        triad_rate: triad_rate_from_tagged(&pos_tagged_sentences, source),
        contraction_ratio: contraction_ratio(document),
        bullet_parallelism: bullet_parallelism(document, tagger),
        paragraph_shape_mean: shape_stats.mean,
        paragraph_shape_cv: shape_stats.cv,
        em_dash_density: em_dash_density(document),
        semicolon_density: semicolon_density(document),
        participial_closer_rate: participial_closer_rate_from_tagged(&pos_tagged_sentences, source),
        not_just_but_rate: not_just_but_rate(document),
        ritual_marker_rate: ritual_marker_rate(document),
        llm_favored_phrase_rate: llm_favored_phrase_rate_with_total(document, total_tokens),
        human_favored_phrase_rate: human_favored_phrase_rate_with_total(document, total_tokens),
        heading_density: heading_density_with_total(document, total_tokens),
        list_item_density: list_item_density_with_total(document, total_tokens),
        bold_span_density: bold_span_density_with_total(document, total_tokens),
        sentence_opener_repeat_rate: sentence_opener_repeat_rate(document),
        top_opener_concentration: top_opener_concentration(document),
    }
}

/// Builds a single-paragraph sub-document from `unit`, one of `document`'s
/// own prose units: shared source text ([`Document::source_arc`] clone, so
/// cheap), one block (`unit`'s owning block, re-indexed to `0`), and one
/// prose unit (`unit` re-parented to that block).
///
/// # Panics
/// Never, for a `unit` from `document.prose()`: its block index and range
/// were already validated against `document`, and copying that same pair
/// unchanged into a fresh single-block document can't fail the same check.
fn single_paragraph_document(document: &Document, unit: &ProseUnit) -> Document {
    let block: Block = document.blocks()[unit.block].clone();
    let sub_unit = ProseUnit::new(0, unit.range.clone(), unit.sentences.clone());
    Document::new(document.source_arc(), vec![block], vec![sub_unit])
        .expect("a paragraph's own block/prose slice, taken from a valid Document, is itself valid")
}

#[cfg(test)]
mod tests {
    use friction_core::{Sentence, Token, TokenKind as CoreTokenKind};
    use friction_nlp::PosTag;

    use super::*;

    /// Stub [`Segmenter`]: one sentence spanning the whole text — enough
    /// structure for `compute` to walk without the real SRX ruleset.
    struct WholeUnitSegmenter;

    impl Segmenter for WholeUnitSegmenter {
        // Deliberately a one-element `Vec<Range<usize>>`, not a range
        // clippy could mistake for "whole range as index sequence".
        #[allow(clippy::single_range_in_vec_init)]
        fn segment(&self, text: &str, base_offset: usize) -> Vec<std::ops::Range<usize>> {
            if text.trim().is_empty() {
                Vec::new()
            } else {
                vec![base_offset..base_offset + text.len()]
            }
        }
    }

    /// Stub [`Tagger`]: tags every token `"NN"`, the least eventful tag —
    /// these fixtures check that `compute` reaches every field, not any
    /// tagger-quality-dependent symmetry value.
    struct FlatNounTagger;

    impl Tagger for FlatNounTagger {
        fn tag(&self, text: &str, base_offset: usize) -> Vec<friction_nlp::TaggedToken> {
            let mut tokens = Vec::new();
            let mut start = None;
            for (i, c) in text.char_indices() {
                if c.is_whitespace() {
                    if let Some(s) = start.take() {
                        tokens.push(word(text, base_offset, s, i));
                    }
                } else if start.is_none() {
                    start = Some(i);
                }
            }
            if let Some(s) = start {
                tokens.push(word(text, base_offset, s, text.len()));
            }
            tokens
        }
    }

    fn word(text: &str, base_offset: usize, start: usize, end: usize) -> friction_nlp::TaggedToken {
        friction_nlp::TaggedToken {
            token: Token::new(
                (base_offset + start)..(base_offset + end),
                CoreTokenKind::Word,
            ),
            pos: PosTag::new("NN"),
            lemma: text[start..end].to_ascii_lowercase().into(),
        }
    }

    /// `compute` reaches all twenty-one fields, wired to the right family:
    /// hand-computed against one single-sentence, single-paragraph document
    /// under the stub segmenter/tagger above.
    ///
    /// Source: `"However, it can't stop — it just keeps going; still
    /// going."`, segmented (by [`WholeUnitSegmenter`]) to one sentence
    /// spanning the whole paragraph.
    ///
    /// - `sentence_length_mean/stddev/cv`: 11 whitespace tokens; single
    ///   observation, so `stddev = cv = 0.0`.
    /// - `discourse_marker_density`: opens with `"However"` (1 marked
    ///   sentence) over 10 word tokens — `1 * 1000 / 10 = 100.0`.
    /// - `contraction_ratio`: `"can't"` is the only contracted/contractible
    ///   form — `1 / (1 + 0) = 1.0`.
    /// - `em_dash_density`/`semicolon_density`: one dash, one semicolon,
    ///   over 11 tokens — `1000 / 11` each.
    /// - `paragraph_shape_mean/cv`: one paragraph, one sentence — mean
    ///   `1.0`, `cv = 0.0`.
    /// - `triad_rate`/`participial_closer_rate`/`bullet_parallelism`: no
    ///   coordinator, trailing-comma-`VBG`, or list — all `0.0` regardless
    ///   of the stub tagger's uniform `"NN"` tag.
    /// - `not_just_but_rate`/`ritual_marker_rate`: no matching pattern —
    ///   both `0.0`.
    /// - `llm_favored_phrase_rate`/`human_favored_phrase_rate`: no word or
    ///   bigram matches a curated mined-pack entry — both `0.0`.
    /// - `heading_density`/`list_item_density`/`bold_span_density`: no
    ///   markup — all `0.0`.
    /// - `sentence_opener_repeat_rate`: one sentence, no consecutive pair —
    ///   `0.0`.
    /// - `top_opener_concentration`: one opener observed ("however") —
    ///   `1 / 1 = 1.0`.
    #[test]
    fn compute_reaches_every_metric_family() {
        const EPSILON: f64 = 1e-9;
        let source = "However, it can't stop — it just keeps going; still going.";
        let document = friction_parse::parse(source).expect("valid markdown parses");
        let metrics = compute(&document, &WholeUnitSegmenter, &FlatNounTagger);

        let expected_dash_semicolon = 1000.0 / 11.0;
        let expected = MetricVector {
            sentence_length_mean: 11.0,
            sentence_length_stddev: 0.0,
            sentence_length_cv: 0.0,
            discourse_marker_density: 100.0,
            triad_rate: 0.0,
            contraction_ratio: 1.0,
            bullet_parallelism: 0.0,
            paragraph_shape_mean: 1.0,
            paragraph_shape_cv: 0.0,
            em_dash_density: expected_dash_semicolon,
            semicolon_density: expected_dash_semicolon,
            participial_closer_rate: 0.0,
            not_just_but_rate: 0.0,
            ritual_marker_rate: 0.0,
            llm_favored_phrase_rate: 0.0,
            human_favored_phrase_rate: 0.0,
            heading_density: 0.0,
            list_item_density: 0.0,
            bold_span_density: 0.0,
            sentence_opener_repeat_rate: 0.0,
            top_opener_concentration: 1.0,
        };

        for (name, value) in metrics.named_values() {
            let expected_value = expected.get(name).expect("named field exists");
            assert!(
                (value - expected_value).abs() < EPSILON,
                "{name}: expected {expected_value}, got {value}"
            );
        }
    }

    /// One vector per non-empty paragraph, in source order: skips a
    /// heading-only unit that segments to zero sentences.
    #[test]
    fn compute_by_paragraph_returns_one_vector_per_sentence_bearing_paragraph() {
        let source = "First paragraph here.\n\nSecond paragraph, quite different.\n";
        let document = friction_parse::parse(source).expect("valid markdown parses");
        let vectors = compute_by_paragraph(&document, &WholeUnitSegmenter, &FlatNounTagger);
        assert_eq!(vectors.len(), 2);
    }

    /// Empty document: empty per-paragraph list, all-zero document vector
    /// (the degenerate-input convention every family documents itself).
    #[test]
    fn compute_handles_empty_document() {
        let document = friction_parse::parse("").expect("empty source parses");
        let metrics = compute(&document, &WholeUnitSegmenter, &FlatNounTagger);
        assert_eq!(metrics, MetricVector::default());
        assert!(compute_by_paragraph(&document, &WholeUnitSegmenter, &FlatNounTagger).is_empty());
    }

    /// A single-paragraph document has no cross-paragraph signal to lose,
    /// so `compute_by_paragraph` must agree field-for-field with `compute`.
    #[test]
    fn compute_by_paragraph_matches_compute_for_a_single_paragraph_document() {
        let source = "The kit includes screws, bolts, and washers.";
        let document = friction_parse::parse(source).expect("valid markdown parses");
        let whole = compute(&document, &WholeUnitSegmenter, &FlatNounTagger);
        let by_paragraph = compute_by_paragraph(&document, &WholeUnitSegmenter, &FlatNounTagger);
        assert_eq!(by_paragraph, vec![whole]);
    }

    /// Exercises `single_paragraph_document`'s block re-indexing directly,
    /// bypassing segmentation.
    #[test]
    fn single_paragraph_document_reindexes_block_to_zero() {
        let source = "Paragraph one.\n\nParagraph two.\n";
        let block = Block::new(friction_core::BlockKind::Paragraph, 16..30);
        let sentence = Sentence::new(16..30, Vec::new());
        let unit = ProseUnit::new(1, 16..30, vec![sentence]);
        let document = Document::new(
            source,
            vec![
                Block::new(friction_core::BlockKind::Paragraph, 0..14),
                block,
            ],
            vec![unit.clone()],
        )
        .expect("hand-built fixture is well-formed");

        let sub = single_paragraph_document(&document, &unit);
        assert_eq!(sub.blocks().len(), 1);
        assert_eq!(sub.prose().len(), 1);
        assert_eq!(sub.prose()[0].block, 0);
        assert_eq!(sub.text(&sub.prose()[0].range).unwrap(), "Paragraph two.");
    }
}
