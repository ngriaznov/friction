//! Shared per-sentence part-of-speech tagging.
//!
//! [`crate::lvc`] and [`crate::jargon`] are the only two channels needing
//! part-of-speech tags (see [`crate::jargon`]'s own module docs), and
//! until now each tagged every in-scope sentence on its own — two full
//! tag passes over the same text per [`crate::MatchEngine::scan`] call.
//! [`tag_units`] tags each sentence exactly once; both channels then take
//! borrowed [`TaggedSentence`] slices instead of a [`Tagger`] of their
//! own. `friction fix`'s standalone paraphrase scan (`friction-cli`'s
//! `fix` module), which calls [`crate::jargon::scan_units`] outside a
//! [`crate::MatchEngine`], goes through the same [`tag_units`] entry
//! point.

use std::ops::Range;

use friction_core::Token;
use friction_nlp::{TaggedToken, Tagger};
// Only the native (non-wasm32) path below uses `.par_iter()` — see
// `tag_units`'s own wasm32 fallback comment.
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use crate::jargon::is_link_label;
use crate::token::ScopedUnit;

/// One sentence's tagged tokens, computed once and shared by every
/// channel that needs part-of-speech tags.
///
/// Absolute byte offsets (`tagger.tag(text, range.start)`) — the
/// convention [`crate::lvc`] and [`crate::jargon`] both already used
/// before this module existed, so neither channel's own span-building
/// arithmetic changes.
pub struct TaggedSentence {
    /// This sentence's byte range, absolute into the document source.
    pub range: Range<usize>,
    /// This sentence's tagged tokens, absolute into the document source.
    pub tokens: Vec<TaggedToken>,
    /// Whether the prose unit this sentence belongs to is a markdown
    /// link/image label ([`crate::jargon::is_link_label`]) — a unit-level
    /// property that has nothing to do with tagging, precomputed here
    /// once so [`crate::jargon::scan_units`] doesn't need its own second
    /// walk over `units` just to re-derive it per sentence.
    pub in_link_label: bool,
}

/// Every sentence's byte range across `units`, paired with whether its
/// own prose unit is a link label ([`is_link_label`]), in document order.
///
/// The bookkeeping [`tag_units`] itself needs before it can tag anything,
/// exposed separately for a caller that has tags from elsewhere
/// (`friction fix`'s paraphrase scan, reusing
/// `friction_edit::RegisterSentenceTags` — see [`from_local_tags`]'s own
/// docs) and only needs this module's link-label bookkeeping to wrap
/// them into [`TaggedSentence`]s, never a fresh [`Tagger::tag`] call.
#[must_use]
pub fn sentence_scopes(units: &[ScopedUnit<'_>], document_text: &str) -> Vec<(Range<usize>, bool)> {
    units
        .iter()
        .flat_map(|unit| {
            let in_link_label = is_link_label(unit, document_text);
            unit.sentences
                .iter()
                .cloned()
                .map(move |range| (range, in_link_label))
        })
        .collect()
}

/// Tags every sentence across `units`, in a flattened, order-preserving
/// rayon `par_iter`.
///
/// The same discipline `friction_edit::register`'s own sentence-context
/// build uses (see that module's `build_sentence_contexts` docs) and
/// [`crate::jargon`] itself used before this module existed. Every
/// sentence's tag is a pure function of its own text, so thread count
/// never affects the result, only how it's computed.
#[must_use]
pub fn tag_units(
    units: &[ScopedUnit<'_>],
    document_text: &str,
    tagger: &dyn Tagger,
) -> Vec<TaggedSentence> {
    let flat = sentence_scopes(units, document_text);

    // rayon cannot spawn OS threads on wasm32-unknown-unknown; sequential
    // `.iter()` there is byte-identical (see `friction-edit::parse_ctx`'s
    // own comment on the same substitution for the determinism proof).
    #[cfg(not(target_arch = "wasm32"))]
    {
        flat.par_iter()
            .map(|(range, in_link_label)| {
                let text = &document_text[range.clone()];
                let tokens = tagger.tag(text, range.start);
                TaggedSentence {
                    range: range.clone(),
                    tokens,
                    in_link_label: *in_link_label,
                }
            })
            .collect()
    }
    #[cfg(target_arch = "wasm32")]
    {
        flat.iter()
            .map(|(range, in_link_label)| {
                let text = &document_text[range.clone()];
                let tokens = tagger.tag(text, range.start);
                TaggedSentence {
                    range: range.clone(),
                    tokens,
                    in_link_label: *in_link_label,
                }
            })
            .collect()
    }
}

/// Builds a [`TaggedSentence`] from already-tagged LOCAL-offset tokens.
///
/// `range` is absolute into the document; `local_tokens` uses offset `0`
/// (relative to `range` — the convention `friction_edit::RegisterSentenceTags`
/// documents); `in_link_label` is whatever the caller already determined
/// for `range`'s own unit. Everything [`tag_units`] would have produced
/// for this sentence, without calling [`Tagger::tag`] again — shifting
/// each token's range by `range.start` is `O(tokens)`, far cheaper than
/// re-tagging.
#[must_use]
pub fn from_local_tags(
    range: Range<usize>,
    local_tokens: &[TaggedToken],
    in_link_label: bool,
) -> TaggedSentence {
    let offset = range.start;
    let tokens = local_tokens
        .iter()
        .map(|token| TaggedToken {
            token: Token::new(
                (token.token.range.start + offset)..(token.token.range.end + offset),
                token.token.kind,
            ),
            pos: token.pos.clone(),
            lemma: token.lemma.clone(),
        })
        .collect();
    TaggedSentence {
        range,
        tokens,
        in_link_label,
    }
}
