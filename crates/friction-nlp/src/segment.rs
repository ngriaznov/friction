//! Sentence segmentation: splitting prose text into sentence byte ranges.
//!
//! [`Segmenter`] is the boundary between this crate and its
//! implementation(s) — currently a single SRX-rule-based one, in
//! [`crate::segment_srx`] — so other crates and future implementations
//! only ever need to depend on the trait. [`segment_document`] wires a
//! [`Segmenter`] up to [`friction_core::Document`], filling in each
//! [`ProseUnit`]'s sentences with absolute byte ranges into the document's
//! original source.

use std::ops::Range;

use friction_core::{CoreError, Document, ProseUnit, Sentence};

/// Splits prose text into sentence byte ranges.
///
/// Implementations receive the text to segment together with
/// `base_offset`, the byte position at which `text` begins in some larger
/// source; every range in the returned [`Vec`] is already shifted by
/// `base_offset`, so callers slice the *original* source with it directly
/// rather than `text`. Ranges are returned in source order, never straddle
/// a UTF-8 character boundary, and exclude each sentence's surrounding
/// whitespace.
///
/// Implementations must be deterministic: identical `text` and
/// `base_offset` always produce identical output, on any machine, on any
/// run.
pub trait Segmenter {
    /// Segments `text` into sentence byte ranges, each offset by
    /// `base_offset`.
    fn segment(&self, text: &str, base_offset: usize) -> Vec<Range<usize>>;
}

/// Errors produced while segmenting a [`Document`]'s prose into sentences.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum SegmentError {
    /// A [`Segmenter`] produced sentence ranges that failed the
    /// document's span-honesty validation (out of bounds, not contained
    /// in their prose unit, or splitting a UTF-8 character) — a
    /// [`Segmenter`] bug, since every range it returns must already be an
    /// offset into the original source.
    #[error("segmented document failed span validation: {0}")]
    InvalidStructure(#[from] CoreError),
}

/// Segments every [`ProseUnit`] in `document` with `segmenter`.
///
/// Returns a new [`Document`] whose prose units have their `sentences`
/// filled in with absolute byte ranges into the original source. Sentence
/// `tokens` are left empty; tokenization is a later, separate pass. This
/// is a pure function of `document` and `segmenter`'s behavior: identical
/// inputs always produce an identical result.
///
/// # Errors
/// Returns [`SegmentError`] if `segmenter` produces a range that escapes
/// its prose unit or the document bounds. Well-behaved implementations of
/// [`Segmenter`] cannot trigger this.
pub fn segment_document(
    document: &Document,
    segmenter: &dyn Segmenter,
) -> Result<Document, SegmentError> {
    let prose = document
        .prose()
        .iter()
        .map(|unit| segment_prose_unit(document, unit, segmenter))
        .collect();

    Document::new(document.source_arc(), document.blocks().to_vec(), prose)
        .map_err(SegmentError::from)
}

/// Segments a single [`ProseUnit`], reusing its `block` index and `range`
/// unchanged and replacing `sentences`.
fn segment_prose_unit(
    document: &Document,
    unit: &ProseUnit,
    segmenter: &dyn Segmenter,
) -> ProseUnit {
    let text = document
        .text(&unit.range)
        .expect("ProseUnit ranges are already validated by Document::new");

    let sentences = segmenter
        .segment(text, unit.range.start)
        .into_iter()
        .map(|range| Sentence::new(range, Vec::new()))
        .collect();

    ProseUnit::new(unit.block, unit.range.clone(), sentences)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stub `Segmenter` that always returns the whole input as one
    /// sentence, to exercise the `Document` wiring independent of any
    /// real segmentation rules.
    struct WholeTextSegmenter;

    impl Segmenter for WholeTextSegmenter {
        // A genuine `Vec` holding one sentence range, not a range of
        // values to collect into a `Vec` — the false positive this lint
        // warns about does not apply here.
        #[allow(clippy::single_range_in_vec_init)]
        fn segment(&self, text: &str, base_offset: usize) -> Vec<Range<usize>> {
            if text.is_empty() {
                Vec::new()
            } else {
                vec![base_offset..base_offset + text.len()]
            }
        }
    }

    /// `segment_document` fills in absolute sentence ranges for every
    /// prose unit, leaving tokens empty, without disturbing block/prose
    /// structure.
    #[test]
    fn segment_document_fills_sentences_with_absolute_ranges() {
        let source = "Hello world.\n\nSecond paragraph.\n";
        let document = friction_parse::parse(source).expect("valid markdown parses");
        assert_eq!(document.prose().len(), 2, "two paragraphs expected");

        let segmented =
            segment_document(&document, &WholeTextSegmenter).expect("segmentation must succeed");

        assert_eq!(segmented.source(), source);
        assert_eq!(segmented.blocks(), document.blocks());
        assert_eq!(segmented.prose().len(), 2);

        for (unit, original) in segmented.prose().iter().zip(document.prose()) {
            assert_eq!(unit.block, original.block);
            assert_eq!(unit.range, original.range);
            assert_eq!(unit.sentences.len(), 1);
            let sentence = &unit.sentences[0];
            assert_eq!(sentence.range, unit.range);
            assert!(sentence.tokens.is_empty());
            assert_eq!(
                segmented.text(&sentence.range).unwrap(),
                document.text(&unit.range).unwrap()
            );
        }
    }

    /// An empty document segments to an empty, valid document.
    #[test]
    fn segment_document_accepts_empty_document() {
        let document = friction_parse::parse("").expect("empty source parses");
        let segmented =
            segment_document(&document, &WholeTextSegmenter).expect("segmentation must succeed");
        assert!(segmented.prose().is_empty());
    }
}
