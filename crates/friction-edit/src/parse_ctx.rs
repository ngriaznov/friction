//! The per-sentence tag+parse context shared by every parse-aware pass.
//!
//! [`register`](crate::register) and [`restructure`](crate::restructure) both need the same
//! thing per sentence: its byte range, its tagged tokens, and its dependency parse, built once
//! and reused across candidate generation, gating, and conflict checking. This module holds that
//! shared shape and the `rayon`-parallel builder that produces it, extracted out of `register.rs`
//! (its original, sole owner) once `restructure.rs` needed the same pair without duplicating
//! roughly eighty lines of tagging/parsing/filtering logic.

use std::ops::Range;

use friction_nlp::{DepParser, SentenceParse, TaggedToken, Tagger};
use rayon::prelude::*;

/// One in-scope sentence's text range, tags, and dependency parse:
/// computed once and reused by every stage a caller runs over it
/// (counting, candidate generation, closure/gate checking, conflict
/// checking) so none re-tag or re-parse.
pub struct SentenceCtx {
    /// This sentence's byte range in the document's original source.
    pub range: Range<usize>,
    /// Tagged tokens, offset 0 (local to this sentence's own text).
    pub tokens: Vec<TaggedToken>,
    /// This sentence's dependency parse, indexed the same way as
    /// `tokens`.
    pub parse: SentenceParse,
    /// This "sentence" opens a prose run split off mid-sentence by an
    /// excluded construct (same-block predecessor run with no
    /// sentence-terminal punctuation — the `crate::document` rule the
    /// four operations use for recapitalization). Its text is the tail
    /// of a clause, not a clause: a paired em-dash aside cut at an
    /// inline-code span leaves exactly one dash visible here, and
    /// rewriting it splits the pair ("run — [`code`]: rather than", a
    /// real defect measured on this repository's own comments).
    pub continues_previous: bool,
}

/// Tags and parses every non-empty sentence in `units`, silently
/// dropping any sentence a per-sentence parse failure or empty trimmed
/// text excludes.
///
/// Every sentence's tag+parse is independent of every other's — neither
/// [`Tagger::tag`] nor [`DepParser::parse`] reads or writes anything but
/// its own `text` argument (see both traits' `Send + Sync` docs) — so
/// this runs over a flattened, order-preserving list of sentence ranges
/// with a rayon `par_iter`. `filter_map` over an indexed parallel
/// iterator, collected straight into a `Vec`, keeps the surviving
/// sentences in exactly the document order the sequential version
/// produced: rayon's collect for an `IndexedParallelIterator` reassembles
/// results by source position regardless of which thread computed which
/// element or how many threads ran, which is the byte-determinism this
/// module's callers (and `tests::rayon_thread_count_determinism`) depend
/// on.
pub fn build_sentence_contexts(
    source: &str,
    units: &[friction_match::token::ScopedUnit<'_>],
    tagger: &dyn Tagger,
    parser: &dyn DepParser,
) -> Vec<SentenceCtx> {
    let ranges: Vec<(Range<usize>, bool)> = units
        .iter()
        .enumerate()
        .flat_map(|(unit_index, unit)| {
            // A unit sharing its predecessor's block index is a later
            // run of the same prose session, split off by an excluded
            // construct (`friction_parse::extract`'s guarantee). Its
            // first "sentence" is a genuine sentence only if that
            // predecessor ended with sentence-terminal punctuation —
            // the same rule `crate::document` applies for
            // recapitalization. Same-block runs share a block kind, so
            // scope filtering never separates them and the scoped
            // predecessor is the true one.
            let continues_previous =
                unit_index
                    .checked_sub(1)
                    .map(|i| &units[i])
                    .is_some_and(|prev| {
                        prev.unit.block == unit.unit.block
                            && !crate::document::ends_with_sentence_terminal_punctuation(prev.text)
                    });
            unit.sentences
                .iter()
                .enumerate()
                .map(move |(sentence_index, range)| {
                    (range.clone(), sentence_index == 0 && continues_previous)
                })
        })
        .collect();
    ranges
        .par_iter()
        .filter_map(|(range, continues_previous)| {
            build_sentence_ctx(source, range, tagger, parser, *continues_previous)
        })
        .collect()
}

/// Tags and parses one sentence, `None` if `range` fails to slice `source`
/// (never expected — `range` always comes from this same `source`'s own
/// segmentation), its trimmed text is empty, or the parse fails (allowed
/// by [`DepParser::parse`]'s contract, never expected from the shipped
/// perceptron parser): a dropped sentence is silently excluded from a
/// parse-aware pass rather than failing the whole document.
fn build_sentence_ctx(
    source: &str,
    range: &Range<usize>,
    tagger: &dyn Tagger,
    parser: &dyn DepParser,
    continues_previous: bool,
) -> Option<SentenceCtx> {
    let text = source.get(range.clone())?;
    if text.trim().is_empty() {
        return None;
    }
    let tokens = tagger.tag(text, 0);
    let parse = parser.parse(text, &tokens).ok()?;
    Some(SentenceCtx {
        range: range.clone(),
        tokens,
        parse,
        continues_previous,
    })
}
