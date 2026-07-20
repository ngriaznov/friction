//! The document → [`MetricVector`] compute boundary: wires the rhythm,
//! lexical, and symmetry metric families together into one fully-populated
//! vector, both for a whole document and for each of its paragraphs
//! individually.
//!
//! [`compute`] and [`compute_by_paragraph`] are the only two entry points
//! other crates need: each takes a [`Document`] as produced by
//! `friction-parse::parse` — block/prose structure extracted, but not yet
//! sentence-segmented — plus the two `friction-nlp` components every
//! metric family here needs (a [`Segmenter`] to fill in sentence
//! boundaries, a [`Tagger`] for the symmetry family's part-of-speech
//! lookups), and returns a complete vector. Callers never need to remember
//! to segment a document themselves before computing its metrics, or to
//! know which of the fourteen fields comes from which family.

use friction_core::{Block, Document, MetricVector, ProseUnit};
use friction_nlp::{Segmenter, Tagger, segment_document};

use crate::lexical::{
    contraction_ratio, discourse_marker_density, not_just_but_rate, ritual_marker_rate,
};
use crate::rhythm::{
    em_dash_density, paragraph_shape, semicolon_density, sentence_length_by_document,
};
use crate::symmetry::{bullet_parallelism, participial_closer_rate, triad_rate};

/// Computes the full 14-field [`MetricVector`] for `document`, document-wide.
///
/// Segments `document` with `segmenter` first (`document` itself is left
/// untouched — this operates on the segmented copy), then computes every
/// metric family's document-level function over the result: the three
/// [`sentence_length_by_document`] fields, [`paragraph_shape`]'s mean/cv,
/// the two punctuation densities, and [`crate::rhythm`]'s own contribution
/// from the rhythm family; the four lexical-marker rates/ratios/densities
/// from [`crate::lexical`]; and the three tagger-dependent structural rates
/// from [`crate::symmetry`]. Every one of [`MetricVector`]'s fourteen
/// fields is written by exactly one of those calls.
///
/// # Panics
/// Panics if `segmenter` produces a sentence range that escapes its prose
/// unit or the document's bounds — a bug in the `Segmenter` implementation
/// itself (every implementation `friction-nlp` ships is well-behaved and
/// cannot trigger this; see [`friction_nlp::segment_document`]'s own docs).
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
/// one sentence; a paragraph with none (e.g. a heading that segmented to
/// zero complete sentences) contributes no entry — the same convention
/// [`crate::rhythm::sentence_length_by_paragraph`] and
/// [`crate::rhythm::paragraph_shape`] already use.
///
/// Each paragraph's vector is [`compute_segmented`] run on a
/// single-paragraph sub-document built from that paragraph's own block and
/// (already-segmented) prose unit, so every metric family's document-wide
/// definition also serves, unmodified, as its paragraph-scoped definition —
/// there is no separate per-paragraph implementation of any of the
/// fourteen metrics to keep in sync with the document-wide one.
/// `paragraph_shape_mean`/`paragraph_shape_cv` degenerate accordingly: a
/// one-paragraph sub-document has exactly one paragraph-shape observation,
/// so `paragraph_shape_mean` is that paragraph's own sentence count and
/// `paragraph_shape_cv` is `0.0` (the single-observation convention, not a
/// special case — see [`crate::rhythm::RhythmStats`]).
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

/// The shared core of [`compute`] and [`compute_by_paragraph`]: computes
/// every metric family's document-level function over an already-segmented
/// `document`, and assembles the result into one [`MetricVector`]. See
/// [`compute`] for which field comes from which family.
fn compute_segmented(document: &Document, tagger: &dyn Tagger) -> MetricVector {
    let sentence_stats = sentence_length_by_document(document);
    let shape_stats = paragraph_shape(document);
    MetricVector {
        sentence_length_mean: sentence_stats.mean,
        sentence_length_stddev: sentence_stats.stddev,
        sentence_length_cv: sentence_stats.cv,
        discourse_marker_density: discourse_marker_density(document),
        triad_rate: triad_rate(document, tagger),
        contraction_ratio: contraction_ratio(document),
        bullet_parallelism: bullet_parallelism(document, tagger),
        paragraph_shape_mean: shape_stats.mean,
        paragraph_shape_cv: shape_stats.cv,
        em_dash_density: em_dash_density(document),
        semicolon_density: semicolon_density(document),
        participial_closer_rate: participial_closer_rate(document, tagger),
        not_just_but_rate: not_just_but_rate(document),
        ritual_marker_rate: ritual_marker_rate(document),
    }
}

/// Builds a single-paragraph sub-document from `unit`, one of `document`'s
/// own prose units: the same source text (an `Arc<str>` clone via
/// [`Document::source_arc`], so this is cheap), one block (`unit`'s own
/// owning block, re-indexed to `0`), and one prose unit (`unit` itself,
/// re-parented to that block, sentences and all).
///
/// # Panics
/// Never, for a `unit` that actually came from `document.prose()`: its
/// block index and range were already validated against `document` when
/// `document` itself was constructed, and copying that same block/range
/// pair unchanged into a fresh single-block document cannot fail that same
/// validation.
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

    /// A stub [`Segmenter`] that always segments an entire prose unit's
    /// text into one sentence spanning it in full — enough structure for
    /// `compute` to walk without depending on the real SRX ruleset.
    struct WholeUnitSegmenter;

    impl Segmenter for WholeUnitSegmenter {
        // A one-sentence-per-unit stub deliberately returns a one-element
        // `Vec<Range<usize>>`, not a `Vec` clippy could mistake for meaning
        // "the whole range as a sequence of indices".
        #[allow(clippy::single_range_in_vec_init)]
        fn segment(&self, text: &str, base_offset: usize) -> Vec<std::ops::Range<usize>> {
            if text.trim().is_empty() {
                Vec::new()
            } else {
                vec![base_offset..base_offset + text.len()]
            }
        }
    }

    /// A stub [`Tagger`] that tags every non-empty token span as a plain
    /// noun (`"NN"`), the least eventful tag for the symmetry family's
    /// purposes here — these fixtures only check that `compute` reaches
    /// every field, not any tagger-quality-dependent symmetry value.
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

    /// `compute` reaches every one of the fourteen fields, wired to the
    /// right family: hand-computed against one single-sentence, single-
    /// paragraph document under the stub segmenter/tagger above.
    ///
    /// Source: `"However, it can't stop — it just keeps going; still
    /// going."`, one paragraph, segmented (by [`WholeUnitSegmenter`]) to
    /// exactly one sentence spanning the whole paragraph.
    ///
    /// - `sentence_length_mean/stddev/cv`: whitespace-token count of the
    ///   one sentence is 11 (`However,`, `it`, `can't`, `stop`, `—`, `it`,
    ///   `just`, `keeps`, `going;`, `still`, `going.`); a single
    ///   observation, so `stddev = cv = 0.0`.
    /// - `discourse_marker_density`: the sentence starts with `"However"`
    ///   (1 marked sentence); [`crate::lexical`]'s own word-token count
    ///   (alphabetic runs, dropping punctuation and the em dash) is 10
    ///   (`however, it, can't, stop, it, just, keeps, going, still,
    ///   going`) — density `= 1 * 1000 / 10 = 100.0`.
    /// - `contraction_ratio`: `"can't"` is the only contracted or
    ///   contractible form present (`1 / (1 + 0) = 1.0`).
    /// - `em_dash_density`/`semicolon_density`: one literal em dash and one
    ///   semicolon, over the same 11 whitespace tokens: `1000 / 11` each.
    /// - `paragraph_shape_mean/cv`: one paragraph with one sentence: mean
    ///   `1.0`, `cv = 0.0` (single-observation convention).
    /// - `triad_rate`/`participial_closer_rate`/`bullet_parallelism`: no
    ///   coordinator, no trailing-comma-`VBG`, no list at all — all `0.0`
    ///   regardless of the stub tagger's uniform `"NN"` tagging.
    /// - `not_just_but_rate`/`ritual_marker_rate`: no matching pattern,
    ///   no ritual open/close phrase — both `0.0`.
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
        };

        for (name, value) in metrics.named_values() {
            let expected_value = expected.get(name).expect("named field exists");
            assert!(
                (value - expected_value).abs() < EPSILON,
                "{name}: expected {expected_value}, got {value}"
            );
        }
    }

    /// `compute_by_paragraph` returns one vector per non-empty paragraph,
    /// in source order, skipping a heading-only prose unit that segments
    /// to zero sentences under a segmenter that refuses blank text.
    #[test]
    fn compute_by_paragraph_returns_one_vector_per_sentence_bearing_paragraph() {
        let source = "First paragraph here.\n\nSecond paragraph, quite different.\n";
        let document = friction_parse::parse(source).expect("valid markdown parses");
        let vectors = compute_by_paragraph(&document, &WholeUnitSegmenter, &FlatNounTagger);
        assert_eq!(vectors.len(), 2);
    }

    /// An empty document yields an empty per-paragraph vector list, and a
    /// document-level vector that is all-zero (the degenerate-input
    /// convention every metric family already documents on its own).
    #[test]
    fn compute_handles_empty_document() {
        let document = friction_parse::parse("").expect("empty source parses");
        let metrics = compute(&document, &WholeUnitSegmenter, &FlatNounTagger);
        assert_eq!(metrics, MetricVector::default());
        assert!(compute_by_paragraph(&document, &WholeUnitSegmenter, &FlatNounTagger).is_empty());
    }

    /// Every metric a single-sentence document produces document-wide is
    /// reproduced exactly by `compute_by_paragraph` for that document's
    /// one paragraph — a single-paragraph document has no cross-paragraph
    /// signal to lose, so the two must agree field-for-field.
    #[test]
    fn compute_by_paragraph_matches_compute_for_a_single_paragraph_document() {
        let source = "The kit includes screws, bolts, and washers.";
        let document = friction_parse::parse(source).expect("valid markdown parses");
        let whole = compute(&document, &WholeUnitSegmenter, &FlatNounTagger);
        let by_paragraph = compute_by_paragraph(&document, &WholeUnitSegmenter, &FlatNounTagger);
        assert_eq!(by_paragraph, vec![whole]);
    }

    /// A dummy sentence/token pair used only to exercise
    /// `single_paragraph_document`'s block re-indexing directly, bypassing
    /// segmentation entirely.
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
