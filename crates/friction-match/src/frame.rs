//! Contrast-frame template detection (SYNTHESIS.md §1, §3).
//!
//! Two deterministic templates — the dismissive-foil interrogative ("is
//! X *just* A, or B?") and declarative epanorthosis ("not just X — it's
//! Y") — kept in exactly one place so `friction-edit`'s frame-gated
//! `just`-deletion (`frame.dejust`) consumes the identical detection this
//! crate's own [`Channel::Frame`] spans report, never a second, drifting
//! copy of the template.
//!
//! # Precision-first: the dismissive marker carries the whole gate
//!
//! A plain either-or question ("is it A or B?") is ordinary legitimate
//! writing; only the dismissive-foil asymmetry — a marker word planted in
//! the first disjunct to make it sound unworthy of consideration — is the
//! tell. [`marker_is_guarded`] is the one guard-token check both
//! templates and `friction-edit`'s own dejust gate share: a marker
//! immediately preceded by `not` or followed by
//! `as`/`about`/`now`/`yet`/`enough` is a genuine multi-word sense ("just
//! as fast", "just about done", "not just a wrapper"), not the dismissive
//! particle, and never licenses a match.
//!
//! # No tagger
//!
//! Both templates are closed-vocabulary lexical frames (a fixed
//! auxiliary set, a fixed four-word marker set, `or`, `not`, the
//! connector set) — the same shape [`crate::literal`]'s automaton covers,
//! not a syntactic pattern that needs part-of-speech disambiguation. This
//! module runs directly over [`crate::token::AnalysisToken`]'s
//! word/punctuation stream, so `friction-edit` can call
//! [`find_contrast_question`] against its own per-sentence working text
//! (`friction_match::token::tokenize_str`) without threading a tagger
//! into an operation that never needed one.

use std::ops::Range;
use std::sync::LazyLock;

use regex::Regex;

use crate::span::{Channel, MatchScore, MatchSpan};
use crate::token::{AnalysisToken, AnalysisTokenKind, ScopedUnit};

/// The four marker words either template's foil disjunct may carry.
///
/// `"only"` licenses detection but is never dejusted — `friction-edit`'s
/// own gate declines it, since it often carries real quantity meaning
/// ("is it only one replica, or...").
pub const DISMISSIVE_MARKERS: [&str; 4] = ["just", "merely", "simply", "only"];

/// Auxiliary/inversion openers that license an interrogative contrast
/// frame's embedded question.
const QUESTION_AUXILIARIES: [&str; 8] = ["is", "are", "was", "were", "does", "do", "can", "could"];

/// Words that turn a following marker into a multi-word sense rather than
/// the dismissive particle: "just as fast", "just about done", "just
/// now", "just yet", "just enough".
const MARKER_FOLLOWER_GUARD: [&str; 5] = ["as", "about", "now", "yet", "enough"];

/// This crate's `frame_id` for template 1.
pub const CONTRAST_QUESTION_ID: &str = "frame.contrast.question";
/// This crate's `frame_id` for template 2.
pub const CONTRAST_CORRECTION_ID: &str = "frame.contrast.correction";

/// One detected `frame.contrast.question` template match.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContrastQuestion {
    /// From the auxiliary through the sentence-final `?`.
    pub range: Range<usize>,
    /// The licensing marker's own token range.
    pub marker_range: Range<usize>,
    /// The marker's case-folded text (`"just"`, `"merely"`, `"simply"`,
    /// or `"only"`).
    pub marker: Box<str>,
}

/// `true` if the token at `idx` is a guarded (non-dismissive) use of a
/// marker word — see this module's own docs.
fn marker_is_guarded(tokens: &[AnalysisToken], idx: usize) -> bool {
    let prev_not = idx > 0
        && tokens[idx - 1].kind == AnalysisTokenKind::Word
        && tokens[idx - 1].text.as_ref() == "not";
    let next_guard = tokens.get(idx + 1).is_some_and(|t| {
        t.kind == AnalysisTokenKind::Word && MARKER_FOLLOWER_GUARD.contains(&t.text.as_ref())
    });
    prev_not || next_guard
}

/// Finds template 1 (`frame.contrast.question`) in one sentence's own
/// tokens, in source order.
///
/// "Auxiliary-initial" describes the embedded question's own shape, not
/// the enclosing sentence's literal first token: a colon-introduced
/// lead-in ("A few architecture questions: is X…") or a connective ("And
/// is X…") routinely sits in front of the real interrogative core in
/// technical prose. The first auxiliary occurrence anywhere in the
/// sentence is this template's anchor; the first `or` after it closes the
/// first disjunct; a licensed (unguarded) marker must sit strictly
/// between the two; the sentence must end in `?` with a non-empty second
/// disjunct.
#[must_use]
pub fn find_contrast_question(tokens: &[AnalysisToken]) -> Option<ContrastQuestion> {
    let last = tokens.last()?;
    if !(last.kind == AnalysisTokenKind::Punctuation && last.text.as_ref() == "?") {
        return None;
    }
    let aux_idx = tokens.iter().position(|t| {
        t.kind == AnalysisTokenKind::Word && QUESTION_AUXILIARIES.contains(&t.text.as_ref())
    })?;
    let or_idx = tokens[aux_idx + 1..]
        .iter()
        .position(|t| t.kind == AnalysisTokenKind::Word && t.text.as_ref() == "or")
        .map(|rel| aux_idx + 1 + rel)?;
    // The second disjunct must be non-empty: at least one token between
    // "or" and the sentence-final "?".
    if or_idx + 1 >= tokens.len() - 1 {
        return None;
    }
    let marker_idx = (aux_idx + 1..or_idx).find(|&i| {
        tokens[i].kind == AnalysisTokenKind::Word
            && DISMISSIVE_MARKERS.contains(&tokens[i].text.as_ref())
            && !marker_is_guarded(tokens, i)
    })?;
    Some(ContrastQuestion {
        range: tokens[aux_idx].range.start..last.range.end,
        marker_range: tokens[marker_idx].range.clone(),
        marker: tokens[marker_idx].text.clone(),
    })
}

/// Template 2a: `not just|only|merely X` followed later in the SAME
/// sentence by a copula-anchored second limb (an em dash or comma before
/// `it's`/`it is`). A plain `but`/`but also` connector is deliberately
/// NOT sufficient: `not only X but also Y` is the standard additive
/// correlative of ordinary human prose (it fired on a human corpus doc
/// when it was accepted here), not the corrective restatement this
/// template targets — the anchor is what separates epanorthosis from
/// coordination.
///
/// The alternation is lazy (`.*?`) so a long sentence's closest anchor
/// after the opener is matched, not its last one.
static MARKED_CORRECTION: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\bnot\s+(?:just|only|merely)\b.*?(?:—|,)\s*(?:it'?s|it\s+is)\b")
        .expect("MARKED_CORRECTION pattern is valid")
});

/// Template 2b, single-sentence form: `It's not X — it's Y.` — the
/// marker-free "bare" shape, licensed only by the repeated `it's`/`it is`
/// anchor (see this module's own docs: a lone "not X but Y" is too common
/// in legitimate prose to flag on its own).
static BARE_CORRECTION_SINGLE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^\s*(?:it'?s|it\s+is)\s+not\b.*—\s*(?:it'?s|it\s+is)\b")
        .expect("BARE_CORRECTION_SINGLE pattern is valid")
});

/// Template 2b, two-sentence form's first sentence: `It's not X.`
static BARE_CORRECTION_FIRST: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^\s*(?:it'?s|it\s+is)\s+not\b")
        .expect("BARE_CORRECTION_FIRST pattern is valid")
});

/// Template 2b, two-sentence form's second sentence: `It's Y.`
static BARE_CORRECTION_SECOND: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^\s*(?:it'?s|it\s+is)\b").expect("BARE_CORRECTION_SECOND pattern is valid")
});

/// Local (relative-to-`unit.text`) byte range of `unit.sentences[index]`.
const fn local_sentence_range(unit: &ScopedUnit<'_>, range: &Range<usize>) -> Range<usize> {
    let base = unit.unit.range.start;
    (range.start - base)..(range.end - base)
}

/// Scans one unit's sentences for template 2 (both the marked and bare
/// sub-forms), checked in that order so a marked instance ("It's not
/// just X — it's Y") is never double-reported under the bare form too.
fn scan_correction_spans(unit: &ScopedUnit<'_>) -> Vec<MatchSpan> {
    let mut spans = Vec::new();
    let sentences = &unit.sentences;
    let mut i = 0usize;
    while i < sentences.len() {
        let range = &sentences[i];
        let local = local_sentence_range(unit, range);
        let text = &unit.text[local];

        if let Some(m) = MARKED_CORRECTION.find(text) {
            spans.push(correction_span((range.start + m.start())..range.end));
            i += 1;
            continue;
        }
        if BARE_CORRECTION_SINGLE.is_match(text) {
            spans.push(correction_span(range.clone()));
            i += 1;
            continue;
        }
        if BARE_CORRECTION_FIRST.is_match(text)
            && text.trim_end().ends_with('.')
            && let Some(next_range) = sentences.get(i + 1)
        {
            let next_local = local_sentence_range(unit, next_range);
            let next_text = &unit.text[next_local];
            if BARE_CORRECTION_SECOND.is_match(next_text) {
                spans.push(correction_span(range.start..next_range.end));
                i += 2;
                continue;
            }
        }
        i += 1;
    }
    spans
}

fn correction_span(range: Range<usize>) -> MatchSpan {
    MatchSpan {
        range,
        channel: Channel::Frame,
        frame_id: CONTRAST_CORRECTION_ID.into(),
        score: MatchScore::Present,
    }
}

/// Every [`AnalysisToken`] in `unit.tokens` whose range falls entirely
/// within `sentence_range`, in source order.
fn sentence_tokens(unit: &ScopedUnit<'_>, sentence_range: &Range<usize>) -> Vec<AnalysisToken> {
    unit.tokens
        .iter()
        .filter(|t| t.range.start >= sentence_range.start && t.range.end <= sentence_range.end)
        .cloned()
        .collect()
}

/// Scans every unit for both templates, returning [`Channel::Frame`]
/// spans in unit/sentence order.
///
/// The one entry point both [`crate::MatchEngine::scan`] (real document
/// spans) and `friction fix`'s paraphrase report (spans over the FIXED
/// output) use.
#[must_use]
pub fn scan_units(units: &[ScopedUnit<'_>]) -> Vec<MatchSpan> {
    let mut spans = Vec::new();
    for unit in units {
        for sentence_range in &unit.sentences {
            let tokens = sentence_tokens(unit, sentence_range);
            if let Some(question) = find_contrast_question(&tokens) {
                spans.push(MatchSpan {
                    range: question.range,
                    channel: Channel::Frame,
                    frame_id: CONTRAST_QUESTION_ID.into(),
                    score: MatchScore::Present,
                });
            }
        }
        spans.extend(scan_correction_spans(unit));
    }
    spans
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::token::tokenize_str;

    fn sentence_tokens_from(text: &str) -> Vec<AnalysisToken> {
        tokenize_str(text, 0)
    }

    fn scoped_document(source: &str) -> friction_core::Document {
        use friction_core::{Block, BlockKind, Document, ProseUnit};
        let blocks = vec![Block::new(BlockKind::Paragraph, 0..source.len())];
        let prose = vec![ProseUnit::new(0, 0..source.len(), Vec::new())];
        Document::new(source, blocks, prose).expect("document is well-formed")
    }

    /// Scans `source` for [`Channel::Frame`] spans end to end (parse,
    /// scope, scan) — used by the template 2 tests, which need real
    /// sentence segmentation across a whole unit rather than one bare
    /// tokenized sentence.
    fn frame_spans(source: &str) -> Vec<MatchSpan> {
        let document = scoped_document(source);
        let segmenter = friction_nlp::SrxSegmenter::new();
        let units = crate::token::prose_scope(&document, &segmenter);
        scan_units(&units)
    }

    // --- Template 1: frame.contrast.question -----------------------------

    #[test]
    fn contrast_question_positive_canonical() {
        let text = "is provenance just frontmatter metadata, or does every derived graph edge \
                     retain immutable lineage?";
        let tokens = sentence_tokens_from(text);
        let m = find_contrast_question(&tokens).expect("template must match");
        assert_eq!(&*m.marker, "just");
        assert_eq!(&text[m.marker_range.clone()], "just");
        assert_eq!(&text[m.range], text);
    }

    #[test]
    fn contrast_question_negative_plain_either_or_has_no_marker() {
        let text = "is it A or B?";
        let tokens = sentence_tokens_from(text);
        assert!(find_contrast_question(&tokens).is_none());
    }

    #[test]
    fn contrast_question_only_is_detected_as_a_frame() {
        let text = "is it only one replica, or does it shard across the cluster?";
        let tokens = sentence_tokens_from(text);
        let m = find_contrast_question(&tokens).expect("only licenses detection");
        assert_eq!(&*m.marker, "only");
    }

    #[test]
    fn contrast_question_guard_just_as_declines() {
        let text = "is it just as fast, or slower?";
        let tokens = sentence_tokens_from(text);
        assert!(find_contrast_question(&tokens).is_none());
    }

    #[test]
    fn contrast_question_guard_just_about_declines() {
        let text = "is it just about done, or does it need another pass?";
        let tokens = sentence_tokens_from(text);
        assert!(find_contrast_question(&tokens).is_none());
    }

    #[test]
    fn contrast_question_guard_not_just_declines() {
        let text = "is it not just fast, or is it also correct?";
        let tokens = sentence_tokens_from(text);
        assert!(find_contrast_question(&tokens).is_none());
    }

    #[test]
    fn contrast_question_requires_a_question_mark() {
        let text = "is provenance just frontmatter metadata, or does it retain full lineage.";
        let tokens = sentence_tokens_from(text);
        assert!(find_contrast_question(&tokens).is_none());
    }

    #[test]
    fn contrast_question_anchors_past_a_colon_lead_in() {
        let text = "A few architecture questions: is provenance just frontmatter metadata, or \
                     does every derived graph edge retain immutable lineage?";
        let tokens = sentence_tokens_from(text);
        let m = find_contrast_question(&tokens).expect("template must match past the lead-in");
        assert_eq!(
            &text[m.range.start..],
            "is provenance just frontmatter metadata, or does \
                     every derived graph edge retain immutable lineage?"
        );
    }

    #[test]
    fn contrast_question_declines_without_a_marker_even_after_a_connective_lead_in() {
        let text = "And is retrieval one flat embedding manifold, or do you partition memory \
                     into semantic wells?";
        let tokens = sentence_tokens_from(text);
        assert!(find_contrast_question(&tokens).is_none());
    }

    // --- Template 2: frame.contrast.correction ----------------------------

    #[test]
    fn contrast_correction_additive_correlative_is_not_matched() {
        // The standard additive correlative of ordinary human prose —
        // this exact shape appears in a human corpus document and must
        // never be flagged: coordination, not corrective restatement.
        let source = "NASA invests not only in space exploration but also in developing \
                      products derived from the scientific endeavors necessary to support \
                      its missions.";
        assert!(frame_spans(source).is_empty());
    }

    #[test]
    fn contrast_correction_positive_marked_em_dash() {
        let source = "The tool is not just a formatter — it's a full static analysis pipeline.";
        let spans = frame_spans(source);
        assert_eq!(spans.len(), 1);
        assert_eq!(&*spans[0].frame_id, CONTRAST_CORRECTION_ID);
    }

    #[test]
    fn contrast_correction_positive_bare_anchor_single_sentence() {
        let source = "It's not a wrapper — it's a rewrite of the parser.";
        let spans = frame_spans(source);
        assert_eq!(spans.len(), 1);
        assert_eq!(&*spans[0].frame_id, CONTRAST_CORRECTION_ID);
    }

    #[test]
    fn contrast_correction_positive_bare_anchor_two_sentences() {
        let source = "It's not a wrapper. It's a rewrite of the parser.";
        let spans = frame_spans(source);
        assert_eq!(spans.len(), 1);
        let span = &spans[0];
        assert_eq!(&source[span.range.clone()], source);
    }

    #[test]
    fn contrast_correction_declarative_without_it_s_anchor_is_not_matched() {
        // A lone "not X but Y" with no marker and no repeated "it's"
        // anchor is too common in legitimate prose to flag.
        let source = "It's not a wrapper but a rewrite of the parser.";
        let spans = frame_spans(source);
        assert!(spans.is_empty(), "expected no span: {spans:?}");
    }

    #[test]
    fn marker_is_guarded_flags_not_prefix_and_follower_words() {
        let tokens = sentence_tokens_from("it is not just fast");
        // tokens: it, is, not, just, fast
        let just_idx = tokens.iter().position(|t| &*t.text == "just").unwrap();
        assert!(marker_is_guarded(&tokens, just_idx));

        let tokens2 = sentence_tokens_from("it is just about done");
        let just_idx2 = tokens2.iter().position(|t| &*t.text == "just").unwrap();
        assert!(marker_is_guarded(&tokens2, just_idx2));

        let tokens3 = sentence_tokens_from("it is just fast");
        let just_idx3 = tokens3.iter().position(|t| &*t.text == "just").unwrap();
        assert!(!marker_is_guarded(&tokens3, just_idx3));
    }
}
