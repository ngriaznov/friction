//! Metaphor-compound jargon detection (SYNTHESIS.md §4).
//!
//! A curated metaphor lexeme is flagged only when it sits in the HEAD
//! position of a noun compound — "semantic wells", "cross-domain
//! resonance" — never when the same word is a bare noun ("the well is
//! deep") or a modifier ("soup kitchen").
//!
//! # Why this channel needs a tagger and the others (mostly) don't
//!
//! [`crate::frame`]'s two templates are closed-vocabulary lexical frames
//! matched directly over the word stream. This channel can't get away
//! with that: `well` is usually an adverb or a bare discourse marker
//! ("works well", "well, actually"), and only its NN/NNS noun sense is
//! ever the metaphor. Distinguishing that sense needs part-of-speech
//! tags, so [`scan_units`] runs [`Tagger::tag`] per sentence, the same
//! shape [`crate::lvc`] already uses.
//!
//! # Detection-only
//!
//! There is no deterministic true replacement for an invented term
//! (SYNTHESIS.md §4) — this channel has no corresponding entry in
//! `friction-edit`'s repair engine and never will on its own; a flagged
//! span is reported (`friction check`'s spans, `friction fix`'s
//! paraphrase report), never rewritten.

use friction_nlp::{TaggedToken, Tagger};
use friction_packs::JargonPack;

use crate::span::{Channel, MatchScore, MatchSpan};
use crate::token::ScopedUnit;

/// This channel's one `frame_id`.
///
/// Unlike [`crate::frame`]'s two templates or [`crate::lvc`]'s per-
/// nominalization ids, every `jargon.metaphor` span carries the same id:
/// the pack (which specific lexeme, which compound) is the detail, not a
/// second axis of classification this crate's span shape needs to carry.
pub const JARGON_METAPHOR_ID: &str = "jargon.metaphor";

/// `true` for the part-of-speech tags this channel accepts as a compound
/// MODIFIER: common noun (singular or plural) or adjective. A hyphenated
/// modifier like "cross-domain" tokenizes as one [`Token`](friction_core::Token)
/// already (see `friction_nlp::tag_perceptron`'s own tokenizer docs), so
/// no separate hyphen-joining logic is needed here.
fn is_modifier_tag(tag: &str) -> bool {
    matches!(tag, "NN" | "NNS" | "JJ")
}

/// `true` for the part-of-speech tags this channel accepts as a compound
/// HEAD: common noun, singular or plural, only — never `NNP`/`NNPS`
/// (proper noun), which this channel's independent capitalization guard
/// ([`compound_is_clean`]) also declines regardless of what the tagger
/// assigns.
fn is_head_tag(tag: &str) -> bool {
    matches!(tag, "NN" | "NNS")
}

/// `true` if `text` starts with an uppercase letter.
fn starts_uppercase(text: &str) -> bool {
    text.chars().next().is_some_and(char::is_uppercase)
}

/// Walks backward from `head_idx` over a MAXIMAL contiguous run of
/// [`is_modifier_tag`] tokens, returning the run's start index (so the
/// compound is `tokens[start..=head_idx]`).
///
/// Returns `None` if `head_idx` is the sentence's first token (nothing to
/// look backward at) or if `tokens[head_idx - 1]` isn't itself a modifier
/// — head-position-only detection requires AT LEAST ONE modifier; a bare
/// lexeme ("the well is deep") never matches.
fn modifier_run_start(tokens: &[TaggedToken], head_idx: usize) -> Option<usize> {
    if head_idx == 0 {
        return None;
    }
    let mut start = head_idx;
    while start > 0 && is_modifier_tag(tokens[start - 1].pos.as_str()) {
        start -= 1;
    }
    (start < head_idx).then_some(start)
}

/// `true` if no token in `tokens[start..=head_idx]` is capitalized mid-
/// sentence — the proper-noun/product-name guard. Index 0 (a genuinely
/// sentence-initial token) is exempt: capitalization there is a sentence-
/// case artifact, not a signal about the word itself.
fn compound_is_clean(
    tokens: &[TaggedToken],
    document_text: &str,
    start: usize,
    head_idx: usize,
) -> bool {
    (start..=head_idx)
        .all(|i| i == 0 || !starts_uppercase(&document_text[tokens[i].token.range.clone()]))
}

/// The compound's lookup key for [`JargonPack::is_attested_exception`]:
/// every token's text lowercased, hyphens folded to spaces, joined with a
/// single space each — "cross-domain resonance" -> "cross domain
/// resonance".
fn normalized_compound(
    tokens: &[TaggedToken],
    document_text: &str,
    start: usize,
    head_idx: usize,
) -> String {
    tokens[start..=head_idx]
        .iter()
        .map(|t| {
            document_text[t.token.range.clone()]
                .to_lowercase()
                .replace('-', " ")
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Attempts to find a `jargon.metaphor` compound headed at `tokens[head_idx]`.
fn jargon_span_at(
    tokens: &[TaggedToken],
    document_text: &str,
    head_idx: usize,
    pack: &JargonPack,
) -> Option<MatchSpan> {
    let head = &tokens[head_idx];
    if !is_head_tag(head.pos.as_str()) {
        return None;
    }
    let head_text = &document_text[head.token.range.clone()];
    if !pack.is_head_word(&head_text.to_lowercase()) {
        return None;
    }

    let start = modifier_run_start(tokens, head_idx)?;
    if !compound_is_clean(tokens, document_text, start, head_idx) {
        return None;
    }

    let normalized = normalized_compound(tokens, document_text, start, head_idx);
    if pack.is_attested_exception(&normalized) {
        return None;
    }

    let range = tokens[start].token.range.start..head.token.range.end;
    Some(MatchSpan {
        range,
        channel: Channel::Jargon,
        frame_id: JARGON_METAPHOR_ID.into(),
        score: MatchScore::Present,
    })
}

/// Tags every sentence in `units` (per-sentence, real absolute byte
/// offsets via `tagger.tag(text, sentence.start)`) and reports every
/// `jargon.metaphor` compound found, in source order.
#[must_use]
pub fn scan_units(
    units: &[ScopedUnit<'_>],
    document_text: &str,
    tagger: &dyn Tagger,
    pack: &JargonPack,
) -> Vec<MatchSpan> {
    let mut spans = Vec::new();
    for unit in units {
        for sentence in &unit.sentences {
            let sentence_text = &document_text[sentence.clone()];
            let tokens = tagger.tag(sentence_text, sentence.start);
            for head_idx in 0..tokens.len() {
                if let Some(span) = jargon_span_at(&tokens, document_text, head_idx, pack) {
                    spans.push(span);
                }
            }
        }
    }
    spans
}

#[cfg(test)]
mod tests {
    use std::sync::OnceLock;

    use friction_core::{Block, BlockKind, Document, ProseUnit};
    use friction_nlp::{PerceptronTagger, SrxSegmenter};

    use super::*;

    fn tagger() -> &'static PerceptronTagger {
        static TAGGER: OnceLock<PerceptronTagger> = OnceLock::new();
        TAGGER.get_or_init(|| PerceptronTagger::new().expect("embedded model must load"))
    }

    const TEST_PACK: &str = r#"
        [[lexemes]]
        lexeme = "well"
        plural = "wells"
        source = "external"
        notes = "test"

        [[lexemes]]
        lexeme = "resonance"
        source = "external"
        notes = "test"

        [[lexemes]]
        lexeme = "soup"
        source = "mined"
        notes = "test"

        [[lexemes]]
        lexeme = "tapestry"
        source = "external"
        notes = "test"

        [[lexemes]]
        lexeme = "fabric"
        source = "mined"
        notes = "test"

        [[attested_exceptions]]
        compound = "data fabric"
        notes = "established industry term"
    "#;

    fn pack() -> &'static JargonPack {
        static PACK: OnceLock<JargonPack> = OnceLock::new();
        PACK.get_or_init(|| JargonPack::parse(TEST_PACK).expect("test pack parses"))
    }

    fn scoped_document(source: &str) -> Document {
        let blocks = vec![Block::new(BlockKind::Paragraph, 0..source.len())];
        let prose = vec![ProseUnit::new(0, 0..source.len(), Vec::new())];
        Document::new(source, blocks, prose).expect("document is well-formed")
    }

    fn spans_for(source: &str) -> Vec<MatchSpan> {
        let document = scoped_document(source);
        let segmenter = SrxSegmenter::new();
        let units = crate::token::prose_scope(&document, &segmenter);
        scan_units(&units, document.source(), tagger(), pack())
    }

    /// End-to-end scan through real markdown parsing (so an inline-code
    /// span is genuinely excluded from prose, not just excluded by a
    /// hand-built `ScopedUnit`).
    fn spans_for_markdown(source: &str) -> Vec<MatchSpan> {
        let document = friction_parse::parse(source).expect("valid markdown parses");
        let segmenter = SrxSegmenter::new();
        let units = crate::token::prose_scope(&document, &segmenter);
        scan_units(&units, document.source(), tagger(), pack())
    }

    #[test]
    fn positive_semantic_wells() {
        let source = "The pipeline partitions memory into semantic wells before indexing.";
        let spans = spans_for(source);
        assert_eq!(spans.len(), 1, "{spans:?}");
        assert_eq!(&*spans[0].frame_id, JARGON_METAPHOR_ID);
        assert_eq!(&source[spans[0].range.clone()], "semantic wells");
    }

    #[test]
    fn positive_cross_domain_resonance() {
        let source = "The report claims cross-domain resonance across every subsystem.";
        let spans = spans_for(source);
        assert_eq!(spans.len(), 1, "{spans:?}");
        assert_eq!(&source[spans[0].range.clone()], "cross-domain resonance");
    }

    #[test]
    fn positive_one_topical_soup_excludes_the_cardinal_modifier() {
        let source = "Every comment thread collapses into one topical soup by noon.";
        let spans = spans_for(source);
        assert_eq!(spans.len(), 1, "{spans:?}");
        // "one" is CD, not a modifier tag, so the span starts at "topical".
        assert_eq!(&source[spans[0].range.clone()], "topical soup");
    }

    #[test]
    fn positive_a_rich_tapestry_of_services_pins_the_exact_span() {
        let source = "The platform is a rich tapestry of services underneath.";
        let spans = spans_for(source);
        assert_eq!(spans.len(), 1, "{spans:?}");
        // "a" is DT, not a modifier tag, so the span starts at "rich".
        assert_eq!(&source[spans[0].range.clone()], "rich tapestry");
    }

    #[test]
    fn negative_echo_is_never_a_listed_lexeme() {
        assert!(!pack().is_head_word("echo"));
        for lexeme in pack().lexemes() {
            assert_ne!(&*lexeme.lexeme, "echo");
        }
    }

    #[test]
    fn negative_the_well_is_deep_has_no_modifier() {
        let source = "In the desert, the well is deep and cold.";
        assert!(spans_for(source).is_empty());
    }

    #[test]
    fn negative_works_well_is_an_adverb_not_a_noun() {
        let source = "The current pipeline works well under load.";
        assert!(spans_for(source).is_empty());
    }

    #[test]
    fn negative_data_fabric_is_an_attested_exception() {
        let source = "The team runs everything on a shared data fabric.";
        assert!(spans_for(source).is_empty());
    }

    #[test]
    fn negative_microsoft_fabric_is_capitalized() {
        let source = "The team migrated to Microsoft Fabric last quarter.";
        assert!(spans_for(source).is_empty());
    }

    #[test]
    fn negative_soup_kitchen_lexeme_is_a_modifier_not_a_head() {
        let source = "The volunteers staffed the soup kitchen downtown.";
        assert!(spans_for(source).is_empty());
    }

    #[test]
    fn negative_lexeme_inside_backticks_is_excluded_from_prose() {
        let source = "Configure the `semantic wells` option in the config file.";
        assert!(spans_for_markdown(source).is_empty());
    }

    #[test]
    fn negative_standalone_resonance_has_no_modifier() {
        let source = "Resonance is measurable across the whole network.";
        assert!(spans_for(source).is_empty());
    }
}
