//! Light-verb-construction detection: every licensed match, not just the
//! first.

use std::collections::BTreeMap;

use friction_nlp::Tagger;
use friction_nlp::lvc::{CandidateOutcome, classify_candidate};

use crate::span::{Channel, MatchScore, MatchSpan};
use crate::token::ScopedUnit;

/// Tags every sentence in `units` (per-sentence, real absolute byte
/// offsets via `tagger.tag(text, sentence.start)`) and reports every
/// licensed light-verb construction found — unlike
/// `friction_harness::pivot::match_pivot`'s single-match-per-sentence
/// contract, this reports every non-overlapping licensed match, since a
/// detection channel has no reason to stop at the first one.
#[must_use]
pub fn scan_units(
    units: &[ScopedUnit<'_>],
    document_text: &str,
    tagger: &dyn Tagger,
    lexicon: &BTreeMap<Box<str>, Box<str>>,
) -> Vec<MatchSpan> {
    let mut spans = Vec::new();

    for unit in units {
        for sentence in &unit.sentences {
            let sentence_text = &document_text[sentence.clone()];
            let tokens = tagger.tag(sentence_text, sentence.start);

            let mut i = 0usize;
            while i < tokens.len() {
                match classify_candidate(&tokens, document_text, i, lexicon) {
                    CandidateOutcome::Licensed(licensed) => {
                        let frame_id: Box<str> =
                            format!("lvc.{}", licensed.nominalization).into_boxed_str();
                        spans.push(MatchSpan {
                            range: licensed.range.clone(),
                            channel: Channel::Lvc,
                            frame_id,
                            score: MatchScore::Present,
                        });
                        i = licensed.end_token_index;
                    }
                    _ => i += 1,
                }
            }
        }
    }

    spans
}

#[cfg(test)]
mod tests {
    use super::*;

    use friction_core::{Block, BlockKind, Document, ProseUnit};
    use friction_nlp::{PerceptronTagger, SrxSegmenter};
    use std::sync::OnceLock;

    fn tagger() -> &'static PerceptronTagger {
        static TAGGER: OnceLock<PerceptronTagger> = OnceLock::new();
        TAGGER.get_or_init(|| PerceptronTagger::new().expect("embedded model must load"))
    }

    fn lexicon() -> BTreeMap<Box<str>, Box<str>> {
        BTreeMap::from([
            (
                Box::from("initialization") as Box<str>,
                Box::from("initialize") as Box<str>,
            ),
            (
                Box::from("decision") as Box<str>,
                Box::from("decide") as Box<str>,
            ),
            (
                Box::from("comparison") as Box<str>,
                Box::from("compare") as Box<str>,
            ),
        ])
    }

    fn scoped_document(source: &str) -> Document {
        let blocks = vec![Block::new(BlockKind::Paragraph, 0..source.len())];
        let prose = vec![ProseUnit::new(0, 0..source.len(), Vec::new())];
        Document::new(source, blocks, prose).expect("document is well-formed")
    }

    #[test]
    fn scan_units_reports_every_non_overlapping_licensed_match_in_a_sentence() {
        let source = "The team made a decision and performed a comparison.";
        let document = scoped_document(source);
        let segmenter = SrxSegmenter::new();
        let units = crate::token::prose_scope(&document, &segmenter);

        let spans = scan_units(&units, document.source(), tagger(), &lexicon());
        assert_eq!(spans.len(), 2);
        assert_eq!(&*spans[0].frame_id, "lvc.decision");
        assert_eq!(&*spans[1].frame_id, "lvc.comparison");
        assert_eq!(&source[spans[0].range.clone()], "made a decision");
        assert_eq!(&source[spans[1].range.clone()], "performed a comparison");
    }

    #[test]
    fn scan_units_finds_nothing_when_no_construction_is_licensed() {
        let source = "The committee reviewed the plan carefully.";
        let document = scoped_document(source);
        let segmenter = SrxSegmenter::new();
        let units = crate::token::prose_scope(&document, &segmenter);

        let spans = scan_units(&units, document.source(), tagger(), &lexicon());
        assert!(spans.is_empty());
    }
}
