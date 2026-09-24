//! Document orchestrator: parses, segments, and runs
//! [`crate::sentence::edit_sentence`] over every prose sentence,
//! bounded to two engine passes.

use std::ops::Range;

use friction_core::{Finding, Patch, find_overlaps};
use friction_match::token::tokenize_str;
use friction_nlp::{DepParser, Segmenter, Tagger, segment_document};
use friction_packs::{AttestationPack, InventoryPack, RegisterPack};

use crate::error::EditError;
use crate::nearnoop::PivotBudget;
use crate::register;
use crate::restructure;
use crate::sentence::{DocumentCasing, EditContext, SentencePosition, edit_sentence};

/// Maximum number of internal engine passes per [`edit_document`] call.
pub const MAX_PASSES: usize = 2;

/// What happened during one internal engine pass.
#[derive(Debug, Default)]
pub struct PassReport {
    /// How many patches were applied this pass.
    pub patches_applied: usize,
    /// How many candidate patches were dropped for overlapping an
    /// already-accepted one this pass.
    pub patches_dropped: usize,
    /// Every patch applied this pass, in original-document byte order —
    /// lets a caller (e.g. near-no-op calibration) filter by `rule`
    /// without the engine needing to know what it's counting.
    pub applied_patches: Vec<Patch>,
    /// Gate-held candidates surfaced this pass (Suggest-tier).
    pub held: Vec<Finding>,
}

/// The full report for one [`edit_document`] call.
#[derive(Debug, Default)]
pub struct EditReport {
    /// One entry per internal pass actually run: zero or more bounded
    /// five/six-op passes, then one restructure pass, then one register
    /// pass.
    pub passes: Vec<PassReport>,
    /// Reusable artifacts for the FINAL text this call returned — see
    /// [`crate::register::ReusableScan`]'s own docs. `None` whenever the
    /// register pass applied at least one patch (it returns `Some` only
    /// when it applied zero), since a patch shifts every later byte
    /// offset out from under them.
    pub reusable_scan: Option<crate::register::ReusableScan>,
    /// Index into `passes` of the last bounded five/six-op pass (the
    /// zero-patch convergence pass, or the bounded-out final round).
    ///
    /// Threaded explicitly rather than derived from `passes.len()`:
    /// the held-candidate selection used to assume `passes.len() - 2`
    /// was always this pass (true only when register was the sole pass
    /// following the bounded loop). Inserting the restructure pass
    /// between them shifted that arithmetic silently — see
    /// [`EditReport::remaining_held`], the selection this field enables.
    pub final_bounded_pass_index: usize,
}

impl EditReport {
    /// The engine's current held candidates, positioned against the
    /// fixed output.
    ///
    /// Selection: the last bounded pass's holds (its re-scan of every
    /// sentence carries the gate-held diagnostics against the converged
    /// text), merged with every pass after it (restructure, register),
    /// each of which reports only its own holds. Keys on
    /// [`Self::final_bounded_pass_index`] rather than `passes.len()`, so
    /// inserting another trailing pass cannot silently select the wrong
    /// pass's holds.
    ///
    /// Positioning: a pass reports its holds against the text IT
    /// received, so each finding is shifted through that pass's own
    /// applied patches and every later pass's. Without the shift, an
    /// edit a trailing pass makes earlier in the document (a
    /// participial-closer split, a register rewrite) leaves every hold
    /// after it pointing a few bytes off.
    #[must_use]
    pub fn remaining_held(&self) -> Vec<Finding> {
        let first = self.final_bounded_pass_index;
        let mut held = Vec::new();
        for (index, pass) in self.passes.iter().enumerate().skip(first) {
            for finding in &pass.held {
                let mut finding = finding.clone();
                for later in &self.passes[index..] {
                    finding.range = rebase_range(&finding.range, &later.applied_patches);
                }
                held.push(finding);
            }
        }
        held
    }
}

/// Maps `range`, a byte range in the text a pass received, into the text
/// that pass produced by applying `patches` (non-overlapping, in that
/// input's coordinates). A bound inside a replaced span snaps outward to
/// the replacement's edge, so the rebased range still covers the edit.
fn rebase_range(range: &Range<usize>, patches: &[Patch]) -> Range<usize> {
    let rebase = |pos: usize, is_end: bool| -> usize {
        let mut shifted = pos;
        for patch in patches {
            let (start, end) = (patch.range.start, patch.range.end);
            let inserted = patch.replacement.len();
            if end <= pos && !(is_end && start == pos && start == end) {
                shifted = shifted + inserted - (end - start);
            } else if start < pos && pos < end {
                let new_start = shifted - (pos - start);
                return if is_end {
                    new_start + inserted
                } else {
                    new_start
                };
            }
        }
        shifted
    };
    let start = rebase(range.start, false);
    let end = rebase(range.end, true).max(start);
    start..end
}

/// Total prose word-token count across `source`'s prose blocks, used to
/// scale the per-document pivot budget.
///
/// `pub(crate)`: `Engine::word_count` exposes this convention to
/// external callers (e.g. `corpus-tool attest --calibrate-near-noop`)
/// so a calibration threshold and its budget count words identically.
pub(crate) fn prose_word_count(
    source: &str,
    segmenter: &dyn Segmenter,
) -> Result<usize, EditError> {
    prose_word_count_with(source, friction_parse::Syntax::Markdown, segmenter)
}

/// [`prose_word_count`] with the surface syntax chosen by the caller.
pub(crate) fn prose_word_count_with(
    source: &str,
    syntax: friction_parse::Syntax,
    segmenter: &dyn Segmenter,
) -> Result<usize, EditError> {
    let document = friction_parse::parse_with(source, syntax)?;
    let with_sentences = segment_document(&document, segmenter)?;
    let mut count = 0usize;
    for unit in with_sentences.prose() {
        let text = with_sentences.text(&unit.range)?;
        count += tokenize_str(text, 0)
            .iter()
            .filter(|t| t.kind == friction_match::token::AnalysisTokenKind::Word)
            .count();
    }
    Ok(count)
}

/// Runs the five-operation pipeline over every prose sentence in
/// `source`, bounded to [`MAX_PASSES`] passes.
///
/// Then runs the restructure pass ([`crate::restructure::run_restructure`])
/// and the register pass ([`crate::register::run_register`]) — each once,
/// over the previous stage's converged/rewritten text.
///
/// Restructure and register both run only after the bounded loop
/// converges: the five operations are subtractive and shift the prose
/// word count that every per-1000-word register rate depends on, so
/// homing against a still-moving denominator would chase a moving
/// target. Restructure sits before register, not after: its own
/// rewrites remove exactly the spans register's features count, so
/// register must see the restructured text or its Wilson-bound arming is
/// computed against a vector the reader never sees (see
/// `crate::restructure`'s own module docs for the full ordering
/// argument).
///
/// # Errors
/// Returns [`EditError`] if `source` fails to parse or segment.
#[allow(clippy::too_many_arguments)] // the engine's one assembly point: each argument is a distinct pack or model
pub fn edit_document(
    source: &str,
    syntax: friction_parse::Syntax,
    inventory: &InventoryPack,
    attestation: &AttestationPack,
    register_pack: &RegisterPack,
    tagger: &dyn Tagger,
    parser: &dyn DepParser,
    segmenter: &dyn Segmenter,
) -> Result<(String, EditReport), EditError> {
    let word_count = prose_word_count_with(source, syntax, segmenter)?;
    let mut pivot_budget = attestation
        .near_noop()
        .map_or_else(PivotBudget::unlimited, |calibration| {
            PivotBudget::for_document(word_count, calibration)
        });

    let mut current = source.to_string();
    let mut report = EditReport::default();
    // Reused across every pass in this call (never across documents or
    // invocations — see `sentence::OriginalStateCache`'s own docs): most
    // sentences are byte-identical between pass 1 and pass 2, so this
    // spares a second tag call for each one pass 2 would otherwise redo.
    let mut original_cache = crate::sentence::OriginalStateCache::default();
    // Same reuse story, one layer up: caches each sentence's WHOLE
    // five-operation outcome (patches, held findings, budget consumption)
    // rather than just the original-text tag/clause facts above — see
    // `sentence::GenerationCache`'s own docs.
    let mut generation_cache = crate::sentence::GenerationCache::default();

    for _ in 0..MAX_PASSES {
        let document = friction_parse::parse_with(current.as_str(), syntax)?;
        let with_sentences = segment_document(&document, segmenter)?;

        let mut candidates: Vec<Patch> = Vec::new();
        let mut held: Vec<Finding> = Vec::new();

        let (positions, casing) = collect_positions_and_casing(&with_sentences, &current);

        let ctx = EditContext {
            inventory,
            attestation,
            tagger,
            casing: &casing,
        };
        for position in &positions {
            let outcome = edit_sentence(
                current.as_str(),
                position,
                &ctx,
                &mut pivot_budget,
                &mut original_cache,
                &mut generation_cache,
            );
            candidates.extend(outcome.patches);
            held.extend(outcome.held);
        }

        let (accepted, dropped) = resolve(current.as_str(), candidates);
        let patches_applied = accepted.len();
        let next = apply(current.as_str(), &accepted);
        let converged = accepted.is_empty();

        report.passes.push(PassReport {
            patches_applied,
            patches_dropped: dropped,
            applied_patches: accepted,
            held,
        });

        current = next;
        if converged {
            break;
        }
    }
    report.final_bounded_pass_index = report.passes.len().saturating_sub(1);

    let bounded_held = &mut report
        .passes
        .last_mut()
        .expect("the bounded loop always pushes at least one pass")
        .held;
    let (restructured, restructure_pass) = restructure::run_restructure(
        &current,
        syntax,
        attestation,
        tagger,
        parser,
        segmenter,
        bounded_held,
    )?;
    report.passes.push(restructure_pass);
    current = restructured;

    let (registered, register_pass, reusable_scan) =
        register::run_register(&current, syntax, register_pack, tagger, parser, segmenter)?;
    report.passes.push(register_pass);
    report.reusable_scan = reusable_scan;
    current = registered;

    Ok((current, report))
}

/// First walk of one bounded pass: collects every in-scope sentence's
/// position, plus casing evidence the recapitalization guard needs from
/// every untouched sentence before any edit (a later sentence marks an
/// earlier opener deliberate) — split out of [`edit_document`] itself
/// only to keep that function's own line count down.
fn collect_positions_and_casing(
    with_sentences: &friction_core::Document,
    current: &str,
) -> (Vec<SentencePosition>, DocumentCasing) {
    let mut positions: Vec<SentencePosition> = Vec::new();
    let mut casing = DocumentCasing::default();
    let prose_units = with_sentences.prose();
    // Precomputed once per pass, each in one O(units)/O(blocks) walk,
    // so the per-unit loop below never rescans either: `sole_item_content`
    // used to re-`filter`+`count` the WHOLE `prose_units` slice for
    // every unit just to ask "does my block own exactly one prose
    // unit", and `in_ordered_list_item` used to re-scan the WHOLE
    // `blocks` slice for every candidate list item just to find its
    // tightest enclosing list — both `O(units)`-per-unit, i.e.
    // `O(units^2)` (`O(blocks^2)` for the latter) over the pass as a
    // whole. See `units_per_block`/`ordered_list_items`'s own docs.
    let units_per_block = units_per_block(with_sentences.blocks().len(), prose_units);
    let ordered_list_item = ordered_list_items(with_sentences.blocks());
    for (unit_index, unit) in prose_units.iter().enumerate() {
        // Prose-blocks-only (gate 7): `friction_parse::parse` also
        // extracts prose from headings/table cells, so this engine
        // filters, reusing the detection layer's
        // `friction_match::token::is_in_scope` allowlist.
        let block_kind = &with_sentences.blocks()[unit.block].kind;
        if !friction_match::token::is_in_scope(block_kind) {
            continue;
        }
        // A prose unit sharing its predecessor's block index is a
        // later run of the same prose session, split off by an
        // excluded construct or gap (`friction_parse::extract`'s
        // guarantee): not a real end-of-block boundary. Its first
        // sentence is new only if the predecessor ended with
        // sentence-terminal punctuation; otherwise the segmenter
        // manufactured a "sentence" from a fragment.
        let unit_starts_new_sentence = match unit_index.checked_sub(1).map(|i| &prose_units[i]) {
            Some(prev) if prev.block == unit.block => with_sentences
                .text(&prev.range)
                .is_ok_and(ends_with_sentence_terminal_punctuation),
            _ => true,
        };
        let sentences = &unit.sentences;
        // Entire prose content of a numbered list item: its unit is
        // the item's only prose unit and only sentence, and its
        // tightest enclosing list block is ordered.
        let sole_item_content = sentences.len() == 1
            && units_per_block[unit.block] == 1
            && ordered_list_item[unit.block];
        for (i, sentence) in sentences.iter().enumerate() {
            let position = SentencePosition {
                range: sentence.range.clone(),
                prev_end: (i > 0).then(|| sentences[i - 1].range.end),
                next_range: sentences.get(i + 1).map(|s| s.range.clone()),
                is_sentence_start: i > 0 || unit_starts_new_sentence,
                sole_content_of_ordered_list_item: sole_item_content,
            };
            casing.record_sentence(&current[sentence.range.clone()], position.is_sentence_start);
            positions.push(position);
        }
    }
    (positions, casing)
}

/// How many `prose_units` belong to each block index, `0..blocks_len`.
///
/// One `O(units)` pass, computed once per pass rather than once per unit:
/// the per-unit loop in [`edit_document`] used to ask "does my own block
/// own exactly one prose unit" by re-`filter`+`count`-ing the WHOLE
/// `prose_units` slice for every single unit (`O(units)` per unit,
/// `O(units^2)` over the pass) — a real cost on large documents, where
/// `positions`-building dominated `friction fix`'s run time. This answers
/// the same question by index lookup instead.
fn units_per_block(blocks_len: usize, prose_units: &[friction_core::ProseUnit]) -> Vec<usize> {
    let mut counts = vec![0usize; blocks_len];
    for unit in prose_units {
        counts[unit.block] += 1;
    }
    counts
}

/// For every block index, `true` if that block is a list item whose
/// tightest enclosing list block is ordered (numbered).
///
/// One `O(blocks)` pre-order walk (a stack of currently-open `List`
/// ancestors — blocks are pre-order, non-overlapping siblings by
/// `friction-parse`'s own documented invariant, so the top of the stack
/// after popping every block that no longer contains the current one is
/// exactly its tightest enclosing list), computed once per pass instead
/// of a fresh `O(blocks)` range-containment scan per candidate list item
/// (`O(blocks)` per item, `O(blocks^2)` over the pass on a
/// list-item-dense document).
fn ordered_list_items(blocks: &[friction_core::Block]) -> Vec<bool> {
    let mut ordered = vec![false; blocks.len()];
    let mut open_lists: Vec<usize> = Vec::new();
    for (index, block) in blocks.iter().enumerate() {
        while let Some(&top) = open_lists.last() {
            if friction_core::span::contains_range(&blocks[top].range, &block.range) {
                break;
            }
            open_lists.pop();
        }
        if matches!(block.kind, friction_core::BlockKind::ListItem)
            && let Some(&top) = open_lists.last()
            && let friction_core::BlockKind::List {
                ordered: list_ordered,
                ..
            } = blocks[top].kind
        {
            ordered[index] = list_ordered;
        }
        if matches!(block.kind, friction_core::BlockKind::List { .. }) {
            open_lists.push(index);
        }
    }
    ordered
}

/// `true` if `text`, trimmed of trailing whitespace and closing
/// quote/bracket characters, ends in a sentence-terminal `.`, `!`, or
/// `?` — tells a genuine new sentence from a prose unit that only looks
/// like one because the segmenter saw nothing but truncated text.
///
/// `pub(crate)`: `crate::register` reuses this — a range cut short by
/// an excluded construct (inline code, a link) is a fragment the
/// segmenter couldn't see past, not a clause worth a dependency parse.
pub(crate) fn ends_with_sentence_terminal_punctuation(text: &str) -> bool {
    text.trim_end()
        .trim_end_matches(['"', '\'', '\u{2019}', '\u{201d}', ')', ']'])
        .ends_with(['.', '!', '?'])
}

/// Validates and resolves candidate patches into the disjoint,
/// applicable subset: leftmost-first, dropping anything overlapping an
/// already-accepted patch. Sentence-level patches should never overlap
/// in practice (ranges are disjoint by construction; splices stay
/// within their own sentence except a ritual deletion's separator
/// extension into already-consumed whitespace): a safety net, not the
/// primary correctness mechanism.
///
/// `pub(crate)`: `crate::register` reuses this same safety net rather
/// than duplicating the leftmost-first tie-break.
pub(crate) fn resolve(source: &str, mut candidates: Vec<Patch>) -> (Vec<Patch>, usize) {
    let before = candidates.len();
    candidates.retain(|p| p.validate(source).is_ok());
    let mut dropped = before - candidates.len();
    candidates.sort_by(|a, b| {
        a.range
            .start
            .cmp(&b.range.start)
            .then_with(|| b.range.end.cmp(&a.range.end))
            .then_with(|| a.rule.as_str().cmp(b.rule.as_str()))
    });

    let mut accepted: Vec<Patch> = Vec::with_capacity(candidates.len());
    for patch in candidates {
        let overlaps = accepted
            .iter()
            .any(|kept: &Patch| friction_core::span::ranges_overlap(&kept.range, &patch.range));
        if overlaps {
            dropped += 1;
        } else {
            accepted.push(patch);
        }
    }
    debug_assert!(
        find_overlaps(&accepted).is_empty(),
        "resolve must never accept overlapping patches"
    );
    (accepted, dropped)
}

/// Applies non-overlapping `patches` to `source`, right-to-left so no
/// earlier patch's range is invalidated by a later replacement.
///
/// `pub(crate)`: shared with `crate::register` for the same reason as
/// [`resolve`].
pub(crate) fn apply(source: &str, patches: &[Patch]) -> String {
    let mut ordered: Vec<&Patch> = patches.iter().collect();
    ordered.sort_by_key(|p| std::cmp::Reverse(p.range.start));
    let mut result = source.to_string();
    for patch in ordered {
        result.replace_range(patch.range.clone(), patch.replacement.as_str());
    }
    result
}

#[cfg(test)]
mod tests {
    use friction_core::{RuleId, Tier};

    use super::*;

    fn finding(rule: &'static str, range: Range<usize>) -> Finding {
        Finding::new(RuleId::new(rule), range, "held", Tier::Suggest)
    }

    fn pass(held: Vec<Finding>, patches: Vec<Patch>) -> PassReport {
        PassReport {
            patches_applied: patches.len(),
            patches_dropped: 0,
            applied_patches: patches,
            held,
        }
    }

    fn patch(range: Range<usize>, replacement: &str) -> Patch {
        Patch::new(range, replacement, RuleId::new("x"), Tier::Fix)
    }

    /// Unions exactly the last bounded pass's holds and every later
    /// pass's, never an earlier round's: with restructure between the
    /// bounded loop and register, a `passes.len()`-derived index would
    /// select restructure's holds in place of the bounded loop's.
    #[test]
    fn remaining_held_unions_bounded_restructure_and_register_passes() {
        let ritual = finding("ritual.delete", 1..2);
        let restructure = finding("restructure.ensures_that", 2..3);
        let register = finding("pivot.lvc", 3..4);
        let report = EditReport {
            passes: vec![
                pass(vec![finding("span.delete", 0..1)], Vec::new()),
                pass(vec![ritual.clone()], Vec::new()),
                pass(vec![restructure.clone()], Vec::new()),
                pass(vec![register.clone()], Vec::new()),
            ],
            reusable_scan: None,
            final_bounded_pass_index: 1,
        };
        assert_eq!(report.remaining_held(), vec![ritual, restructure, register]);
    }

    #[test]
    fn remaining_held_handles_no_passes() {
        assert!(EditReport::default().remaining_held().is_empty());
    }

    /// A later pass that lengthens text before a hold shifts the hold by
    /// the same amount; one that edits after it leaves it alone. The
    /// demo-paragraph case: "loop, allowing" -> "loop. That allowed"
    /// (+4 bytes) used to leave every later hold 4 bytes early.
    #[test]
    fn remaining_held_shifts_through_later_passes() {
        let report = EditReport {
            passes: vec![
                pass(vec![finding("span.delete", 20..30)], Vec::new()),
                pass(
                    Vec::new(),
                    vec![patch(5..10, "123456789"), patch(40..45, "")],
                ),
            ],
            reusable_scan: None,
            final_bounded_pass_index: 0,
        };
        assert_eq!(report.remaining_held()[0].range, 24..34);
    }

    /// A hold is shifted through its OWN pass's patches too: a pass
    /// reports holds against the text it received.
    #[test]
    fn remaining_held_shifts_through_its_own_pass() {
        let report = EditReport {
            passes: vec![pass(
                vec![finding("register.em_dash", 10..20)],
                vec![patch(0..4, "")],
            )],
            reusable_scan: None,
            final_bounded_pass_index: 0,
        };
        assert_eq!(report.remaining_held()[0].range, 6..16);
    }

    /// Bounds inside a replaced span snap outward, so the rebased hold
    /// still covers the replacement.
    #[test]
    fn rebase_snaps_bounds_inside_a_patch_outward() {
        let patches = [patch(10..20, "abc")];
        assert_eq!(rebase_range(&(5..15), &patches), 5..13);
        assert_eq!(rebase_range(&(15..30), &patches), 10..23);
        assert_eq!(rebase_range(&(12..18), &patches), 10..13);
    }
}
