//! Clause-completeness guardrail: the fraction of a text's sentences that
//! lack a finite verb and are not imperative-initial.
//!
//! Unlike [`crate::clean`] and the modules built on it,
//! this module deliberately runs the **real** `friction-parse` +
//! `friction-nlp` pipeline (real markdown block extraction, the real SRX
//! sentence segmenter, the real tagger) rather than the harness's own
//! naive string-level splitter — this is the component validated
//! empirically against actual [`friction_nlp::NlpruleTagger`] output on the
//! full markdown sample files, and real prose extraction correctly handles
//! the block-level structure (code fences, table syntax, link markup) a
//! regex-based splitter would mis-segment.

use friction_nlp::{SrxSegmenter, Tagger, segment_document};

use crate::error::HarnessError;

/// Full Penn tags that count as a finite verb for clause completeness.
const FINITE_VERB_TAGS: [&str; 4] = ["VBZ", "VBP", "VBD", "MD"];

/// The clause-completeness rule, transcribed as a guardrail: a sentence is
/// complete if it contains a token tagged `VBZ`/`VBP`/`VBD`/`MD`, or its
/// first token is tagged `VB` (imperative).
///
/// Segments `text` with `segmenter` (the real prose-extraction +
/// sentence-segmentation pipeline), tags each sentence with `tagger`, and
/// returns the fraction of non-empty sentences that are incomplete (`0.0`
/// if there are none).
///
/// # Errors
/// Propagates [`friction_parse::ParseError`] / [`friction_nlp::SegmentError`].
pub fn fragment_rate(
    text: &str,
    segmenter: &SrxSegmenter,
    tagger: &dyn Tagger,
) -> Result<f64, HarnessError> {
    let document = friction_parse::parse(text)?;
    let with_sentences = segment_document(&document, segmenter)?;

    let mut total = 0usize;
    let mut incomplete = 0usize;

    for unit in with_sentences.prose() {
        for sentence in &unit.sentences {
            let Ok(sentence_text) = with_sentences.text(&sentence.range) else {
                continue;
            };
            let tokens = tagger.tag(sentence_text, 0);
            if tokens.is_empty() {
                continue;
            }
            total += 1;
            let has_finite_verb = tokens
                .iter()
                .any(|t| FINITE_VERB_TAGS.contains(&t.pos.as_str()));
            let imperative_initial = tokens.first().is_some_and(|t| t.pos.as_str() == "VB");
            if !has_finite_verb && !imperative_initial {
                incomplete += 1;
            }
        }
    }

    if total == 0 {
        return Ok(0.0);
    }
    #[allow(clippy::cast_precision_loss)]
    Ok(incomplete as f64 / total as f64)
}

#[cfg(test)]
// `fragment_rate` returns an exact `0.0` for the degenerate/all-complete
// cases below; comparing against that literal is the correct check, not a
// precision bug `float_cmp` should flag.
#[allow(clippy::float_cmp)]
mod tests {
    use std::sync::OnceLock;

    use friction_nlp::{NlpruleTagger, SrxSegmenter};

    use super::*;

    fn tagger() -> &'static NlpruleTagger {
        static TAGGER: OnceLock<NlpruleTagger> = OnceLock::new();
        TAGGER.get_or_init(|| NlpruleTagger::new().expect("embedded model must load"))
    }

    fn segmenter() -> SrxSegmenter {
        SrxSegmenter::new()
    }

    #[test]
    fn complete_sentences_have_zero_fragment_rate() {
        let text = "The tool scans the project directory. Results stream in as they are found.";
        let seg = segmenter();
        let rate = fragment_rate(text, &seg, tagger()).expect("scoring succeeds");
        assert_eq!(rate, 0.0);
    }

    #[test]
    fn verbless_fragments_are_counted() {
        let text = "You get an AU. Then a VST3. Sometimes a VST2 shows up.";
        let seg = segmenter();
        let rate = fragment_rate(text, &seg, tagger()).expect("scoring succeeds");
        assert!(
            rate > 0.0,
            "expected at least one fragment, got rate {rate}"
        );
    }

    #[test]
    fn empty_text_has_zero_fragment_rate() {
        let seg = segmenter();
        let rate = fragment_rate("", &seg, tagger()).expect("scoring succeeds");
        assert_eq!(rate, 0.0);
    }

    #[test]
    fn chatspeak_sample_has_a_higher_fragment_rate_than_good_docs_sample() {
        let seg = segmenter();
        let chatspeak =
            fragment_rate(crate::fixtures::CHATSPEAK_MD, &seg, tagger()).expect("scoring succeeds");
        let good_docs =
            fragment_rate(crate::fixtures::GOOD_DOCS_MD, &seg, tagger()).expect("scoring succeeds");
        assert!(
            chatspeak > good_docs,
            "expected chatspeak ({chatspeak}) > good_docs ({good_docs})"
        );
    }
}
