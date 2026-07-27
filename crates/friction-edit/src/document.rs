//! Document orchestrator: parses, segments, and runs
//! [`crate::sentence::edit_sentence`] over every prose sentence,
//! bounded to two engine passes.

use friction_core::{Finding, Patch, find_overlaps};
use friction_match::token::tokenize_str;
use friction_nlp::{DepParser, Segmenter, Tagger, segment_document};
use friction_packs::{AttestationPack, InventoryPack, RegisterPack};

use crate::error::EditError;
use crate::nearnoop::PivotBudget;
use crate::register;
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
    /// One entry per internal pass actually run.
    pub passes: Vec<PassReport>,
    /// Reusable artifacts for the FINAL text this call returned — see
    /// [`crate::register::ReusableScan`]'s own docs. `None` whenever the
    /// register pass applied at least one patch (it returns `Some` only
    /// when it applied zero), since a patch shifts every later byte
    /// offset out from under them.
    pub reusable_scan: Option<crate::register::ReusableScan>,
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
    let document = friction_parse::parse(source)?;
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
/// `source`, bounded to [`MAX_PASSES`] passes, then runs the register
/// pass ([`crate::register::run_register`]) once over the converged text.
///
/// Register runs only after convergence: the five operations are
/// subtractive and shift the prose word count that every per-1000-word
/// register rate depends on, so homing against a still-moving
/// denominator would chase a moving target. Running last homes the
/// vector of the text that actually ships.
///
/// # Errors
/// Returns [`EditError`] if `source` fails to parse or segment.
pub fn edit_document(
    source: &str,
    inventory: &InventoryPack,
    attestation: &AttestationPack,
    register_pack: &RegisterPack,
    tagger: &dyn Tagger,
    parser: &dyn DepParser,
    segmenter: &dyn Segmenter,
) -> Result<(String, EditReport), EditError> {
    let word_count = prose_word_count(source, segmenter)?;
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
        let document = friction_parse::parse(current.as_str())?;
        let with_sentences = segment_document(&document, segmenter)?;

        let mut candidates: Vec<Patch> = Vec::new();
        let mut held: Vec<Finding> = Vec::new();

        // First walk: collect every in-scope sentence's position, plus
        // casing evidence the recapitalization guard needs from every
        // untouched sentence before any edit (a later `mimalloc` marks
        // an earlier opener deliberate).
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
            // guarantee) — not a real end-of-block boundary. Its first
            // sentence is genuinely new only if the predecessor ended
            // with sentence-terminal punctuation; otherwise the
            // segmenter manufactured a "sentence" from a fragment.
            let unit_starts_new_sentence = match unit_index.checked_sub(1).map(|i| &prose_units[i])
            {
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
                casing
                    .record_sentence(&current[sentence.range.clone()], position.is_sentence_start);
                positions.push(position);
            }
        }

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

    let (registered, register_pass, reusable_scan) =
        register::run_register(&current, register_pack, tagger, parser, segmenter)?;
    report.passes.push(register_pass);
    report.reusable_scan = reusable_scan;
    current = registered;

    Ok((current, report))
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
/// applicable subset — leftmost-first, dropping anything overlapping an
/// already-accepted patch. Sentence-level patches should never overlap
/// in practice (ranges are disjoint by construction; splices stay
/// within their own sentence except a ritual deletion's separator
/// extension into already-consumed whitespace) — a safety net, not the
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
