//! List-free `jargon.compound` detection (SYNTHESIS.md §4).
//!
//! [`crate::jargon`]'s `jargon.metaphor` match needs a curated lexeme in
//! HEAD position — it can only ever catch a modifier-of-a-listed-word
//! compound. §4's own design goes further: "every head word frequent, the
//! compound unattested" needs no list of metaphor-borrowed words at all,
//! only a general-vocabulary frequency floor on each word plus a
//! web-scale zero-check on the pairing. This module is that generalized
//! version — hence "list-free" — built on
//! [`friction_packs::general_evidence`]'s mined unigram/bigram filters
//! instead of a curated TOML.
//!
//! # Optional by construction
//!
//! Unlike every other channel in this crate, [`scan_units`] has no pack
//! parameter to thread through [`crate::MatchEngine`]: it reads
//! [`friction_packs::general_evidence`] itself and returns no findings at
//! all when that resolves to [`None`] (see that function's own docs on
//! why the artifact is optional and never embedded). This is the arming
//! fence — with no `general-evidence-v1` `.bin` installed or cached
//! anywhere, this module contributes nothing and every other channel's
//! output is unaffected, so a document scanned without the pack produces
//! byte-identical spans to a build of this crate that never had this
//! module at all.
//!
//! # Filter cascade
//!
//! A candidate pair only reaches a flag after clearing three independent
//! checks, each of which can suppress it (see [`compound_span_at`]):
//! either word is unknown to the general-vocabulary filter (a name, a
//! typo, an identifier — absence of evidence, not evidence of jargon, so
//! this skips rather than flags); the pairing itself is an attested
//! bigram; or [`friction_packs::normalize_compound`] of the pair is
//! attested by the SAME corpus-attested filter [`crate::jargon`] already
//! consults ([`friction_packs::JargonAttestPack`]) — established
//! terminology the mined bigram table happened to miss. Only a pair that
//! survives all three is unattested pseudo-terminology.

use friction_core::TokenKind;
use friction_nlp::TaggedToken;
use friction_packs::{GeneralEvidencePack, JargonAttestPack};
// Only the native (non-wasm32) path in `scan_units` below uses
// `.par_iter()` — see that function's own wasm32 fallback comment.
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use crate::span::{Channel, MatchScore, MatchSpan};
use crate::tagging::TaggedSentence;

/// This channel's one `frame_id`.
///
/// Every `jargon.compound` span carries it, matching
/// [`crate::jargon::JARGON_METAPHOR_ID`]'s own reasoning: the specific
/// compound is per-occurrence detail, carried in [`MatchSpan::message`]
/// instead, not a second classification axis.
pub const JARGON_COMPOUND_ID: &str = "jargon.compound";

/// A candidate pair's word must be at least this many ASCII letters long
/// — the same floor [`friction_packs::general_evidence::normalize_word`]
/// applies to every table key, checked here too so a hyphenated or
/// apostrophed token never even reaches a candidate pair.
const MIN_WORD_LEN: usize = 3;

/// `true` for the part-of-speech tags this module accepts as a
/// compound's FIRST (modifier) position: common noun (singular or
/// plural) or adjective — the same set [`crate::jargon::is_modifier_tag`]
/// uses, duplicated locally since that function is private to its own
/// module.
fn is_modifier_tag(tag: &str) -> bool {
    matches!(tag, "NN" | "NNS" | "JJ")
}

/// `true` for the part-of-speech tags this module accepts as a
/// compound's SECOND (head) position: common noun, singular or plural,
/// only — `NNP`/`NNPS` (proper noun) is never in this set, so a
/// candidate pair can never have a proper-noun head, and neither set
/// admits `NNP`/`NNPS` in either position.
fn is_head_tag(tag: &str) -> bool {
    matches!(tag, "NN" | "NNS")
}

/// `true` for a candidate token's TEXT shape: at least [`MIN_WORD_LEN`]
/// ASCII letters and nothing else. Case-insensitive (both
/// [`GeneralEvidencePack::has_word`] and [`GeneralEvidencePack::has_bigram`]
/// case-fold internally), so this only rejects shape, never case.
fn is_candidate_word(text: &str) -> bool {
    text.len() >= MIN_WORD_LEN && text.bytes().all(|b| b.is_ascii_alphabetic())
}

/// The flagged-occurrence message every `jargon.compound` span carries:
/// e.g. `"'semantic ladder' is unattested in reference corpora
/// (report-only)"` — matching [`crate::overuse::scan_units`]'s own
/// measured-finding wording (name the thing, state the fact, mark it
/// report-only).
fn compound_message(normalized: &str) -> String {
    format!("'{normalized}' is unattested in reference corpora (report-only)")
}

/// Attempts to find a `jargon.compound` span for the adjacent pair
/// `tokens[idx]`, `tokens[idx + 1]`.
///
/// Every early return here is a SKIP, not a suppression the way
/// [`crate::jargon::jargon_span_at`]'s attested-exception check is: an
/// unknown word, a non-`Word` token, or a wrong tag just means this pair
/// was never a candidate at all, and the difference matters for the same
/// reason [`friction_packs::general_evidence`]'s own docs give — absence
/// from a general-vocabulary filter is "no evidence either way", never
/// "attested".
fn compound_span_at(
    tokens: &[TaggedToken],
    document_text: &str,
    idx: usize,
    evidence: &GeneralEvidencePack,
    attest: &JargonAttestPack,
) -> Option<MatchSpan> {
    let (first, second) = (&tokens[idx], &tokens[idx + 1]);
    if first.token.kind != TokenKind::Word || second.token.kind != TokenKind::Word {
        return None;
    }
    if !is_modifier_tag(first.pos.as_str()) || !is_head_tag(second.pos.as_str()) {
        return None;
    }

    let w1 = &document_text[first.token.range.clone()];
    let w2 = &document_text[second.token.range.clone()];
    if !is_candidate_word(w1) || !is_candidate_word(w2) {
        return None;
    }

    // (a) both words independently attested — an unknown word is a name,
    // typo, or code identifier, so this pair is skipped, not flagged.
    if !evidence.has_word(w1) || !evidence.has_word(w2) {
        return None;
    }
    // (b) the pairing itself is an attested bigram.
    if evidence.has_bigram(w1, w2) {
        return None;
    }

    let range = first.token.range.start..second.token.range.end;
    let normalized = friction_packs::normalize_compound(&document_text[range.clone()]);
    // (c) the corpus-attested compound filter (jargon.metaphor's own
    // oracle, SYNTHESIS.md §4) recognizes this as established
    // terminology the mined bigram table missed.
    if attest.is_attested(&normalized) {
        return None;
    }

    Some(MatchSpan {
        range,
        channel: Channel::Jargon,
        frame_id: JARGON_COMPOUND_ID.into(),
        score: MatchScore::Present,
        message: Some(compound_message(&normalized).into()),
    })
}

/// One sentence's `jargon.compound` spans, in pair-start order — the
/// per-sentence unit of work [`scan_units`]'s `par_iter` runs.
fn sentence_compound_spans(
    tokens: &[TaggedToken],
    document_text: &str,
    evidence: &GeneralEvidencePack,
    attest: &JargonAttestPack,
) -> Vec<MatchSpan> {
    if tokens.len() < 2 {
        return Vec::new();
    }
    (0..tokens.len() - 1)
        .filter_map(|idx| compound_span_at(tokens, document_text, idx, evidence, attest))
        .collect()
}

/// Reports every `jargon.compound` pair found across `tagged`, in source
/// order.
///
/// Returns an empty [`Vec`] with no other work at all when
/// [`friction_packs::general_evidence`] resolves to [`None`] — this
/// module's whole arming fence (see this module's own docs). `attest` is
/// [`crate::jargon`]'s own corpus-attested filter, reused here for the
/// filter cascade's third check. Link-label sentences
/// ([`TaggedSentence::in_link_label`]) are skipped whole, the same
/// exclusion [`crate::jargon::scan_units`] applies and for the same
/// reason (see [`crate::jargon::is_link_label`]'s own docs); a code span
/// never reaches `tagged` at all, since [`crate::token::prose_scope`]
/// already excludes it upstream of tagging.
///
/// Mirrors [`crate::jargon::scan_units`]'s own parallelism: per-sentence
/// work is independent of every other sentence's, so this runs over
/// `tagged` with a rayon `par_iter` on native targets (an indexed
/// `Vec<Vec<MatchSpan>>` collected before flattening, to keep the result
/// in document order regardless of thread count) and a plain
/// order-preserving `.iter()` on wasm32, where rayon cannot spawn OS
/// threads.
#[must_use]
pub fn scan_units(
    tagged: &[TaggedSentence],
    document_text: &str,
    attest: &JargonAttestPack,
) -> Vec<MatchSpan> {
    let Some(evidence) = friction_packs::general_evidence() else {
        return Vec::new();
    };

    #[cfg(not(target_arch = "wasm32"))]
    {
        let per_sentence: Vec<Vec<MatchSpan>> = tagged
            .par_iter()
            .map(|sentence| {
                if sentence.in_link_label {
                    return Vec::new();
                }
                sentence_compound_spans(&sentence.tokens, document_text, evidence, attest)
            })
            .collect();
        per_sentence.into_iter().flatten().collect()
    }
    #[cfg(target_arch = "wasm32")]
    {
        tagged
            .iter()
            .flat_map(|sentence| {
                if sentence.in_link_label {
                    return Vec::new();
                }
                sentence_compound_spans(&sentence.tokens, document_text, evidence, attest)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::sync::OnceLock;

    use friction_core::{Block, BlockKind, Document, ProseUnit};
    use friction_nlp::{PerceptronTagger, SrxSegmenter};

    use super::*;

    fn tagger() -> &'static PerceptronTagger {
        static TAGGER: OnceLock<PerceptronTagger> = OnceLock::new();
        TAGGER.get_or_init(|| PerceptronTagger::new().expect("embedded model must load"))
    }

    /// A small synthetic `jargon-attest-v1` filter attesting nothing —
    /// every case in [`pack_present_scan_covers_every_filter_stage`]
    /// must be decided by [`friction_packs::general_evidence`]'s own
    /// tables, never by this filter, so it's deliberately near-empty
    /// (one filler key so the filter isn't a size-0 edge case).
    fn attest() -> &'static JargonAttestPack {
        static ATTEST: OnceLock<JargonAttestPack> = OnceLock::new();
        ATTEST.get_or_init(|| {
            let keys: BTreeSet<String> =
                std::iter::once("unattested filler key".to_string()).collect();
            let built =
                friction_packs::build_pack_bytes(&keys).expect("small synthetic filter builds");
            let bin: &'static [u8] = Box::leak(built.bytes.into_boxed_slice());
            let sidecar: &'static str = Box::leak(
                format!(
                    "[pack]\nversion = \"jargon-attest-v1\"\nkey_count = {}\n",
                    built.key_count
                )
                .into_boxed_str(),
            );
            JargonAttestPack::load(bin, sidecar).expect("synthetic filter pack loads")
        })
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
        let tagged = crate::tagging::tag_units(&units, document.source(), tagger());
        scan_units(&tagged, document.source(), attest())
    }

    /// The one test in this file (this crate's own test binary) allowed
    /// to touch the real global [`friction_packs::general_evidence`]/
    /// install-seam statics — installing is a process-wide, at-most-once
    /// operation (see [`friction_packs::install_general_evidence_bin`]'s
    /// own docs), so every case that needs the pack PRESENT has to live
    /// in this one test, the same discipline
    /// `friction_packs::general_evidence`'s own test module uses for its
    /// one real-global-touching test. The absence case
    /// ([`crate::MatchEngine`]'s own no-pack byte-identical guarantee)
    /// is exercised separately, in `tests/jargon_compound_no_pack.rs`,
    /// a genuinely separate process this install can never reach.
    #[test]
    fn pack_present_scan_covers_every_filter_stage() {
        let unigrams: BTreeSet<String> = ["classroom", "teacher", "semantic", "ladder"]
            .into_iter()
            .map(String::from)
            .collect();
        let bigrams: BTreeSet<String> = std::iter::once("classroom teacher".to_string()).collect();
        let built = friction_packs::build_general_evidence_bin(&unigrams, &bigrams)
            .expect("small synthetic evidence pack builds");
        friction_packs::install_general_evidence_bin(built.bytes)
            .expect("first (and only) install in this test binary succeeds");
        assert!(
            friction_packs::general_evidence().is_some(),
            "installed pack resolves"
        );

        // An attested bigram never flags, even though both words are
        // independently attested unigrams too.
        let source = "Every classroom teacher graded the assignment overnight.";
        assert!(spans_for(source).is_empty(), "{:?}", spans_for(source));

        // Both words attested, but the pairing is absent from the
        // bigram table and from the (near-empty) attest filter: flags.
        let source = "The team built a semantic ladder to rank every result.";
        let spans = spans_for(source);
        assert_eq!(spans.len(), 1, "{spans:?}");
        assert_eq!(&spans[0].channel, &Channel::Jargon);
        assert_eq!(&*spans[0].frame_id, JARGON_COMPOUND_ID);
        assert_eq!(&source[spans[0].range.clone()], "semantic ladder");
        let message = spans[0]
            .message
            .as_deref()
            .expect("jargon.compound span carries a message");
        assert!(message.contains("semantic ladder"), "{message}");
        assert!(message.contains("unattested"), "{message}");
        assert!(message.contains("report-only"), "{message}");

        // "bridge" is absent from the evidence unigram table entirely:
        // an unknown word skips the pair rather than flagging it.
        let source = "The team built a semantic bridge to rank every result.";
        assert!(spans_for(source).is_empty(), "{:?}", spans_for(source));

        // Same two attested unigrams as the "semantic ladder" positive
        // case above, but capitalized as a product name (the same
        // mid-sentence NNP-NNP shape `jargon.rs`'s own
        // `negative_microsoft_fabric_is_capitalized` test uses): tags
        // as a proper-noun pair, excluded on tag membership alone,
        // never reaching the evidence filter regardless of what it
        // attests.
        let source = "The team migrated to Semantic Ladder last quarter.";
        assert!(spans_for(source).is_empty(), "{:?}", spans_for(source));
    }
}
