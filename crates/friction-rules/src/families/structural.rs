//! The structural rule family: document/list-shape transforms, as opposed
//! to word choice, connective surgery, or sentence rhythm.
//!
//! Three rules, all reacting to a markdown document's block structure
//! rather than its sentence-level wording:
//!
//! - [`UnbulletRule`] (Fix tier): collapses a short, 2-3-item bulleted list
//!   into one flowing prose sentence, when the document leans on bulleted
//!   lists more than its genre's human envelope does.
//! - [`BoldLabelStripRule`] (Fix tier): strips the `**...**` bold markers
//!   off a `"- **Label**: text"` lead-in bullet, when the document is
//!   denser with bold markup than its genre's envelope.
//! - [`HeaderMergeRule`] (Suggest tier): flags a repeated
//!   heading-immediately-followed-by-one-short-paragraph pattern as a
//!   candidate for merging into flowing prose. It never proposes a patch —
//!   deciding how several one-paragraph sections should read as continuous
//!   prose can drop or reorder structure a reader relied on to navigate the
//!   document, so this is diagnostic-only by design, not a gap to close
//!   later.
//!
//! # Why `UnbulletRule` and `BoldLabelStripRule` are Fix tier
//!
//! Both transforms are safe to apply automatically because of what they
//! *cannot* do to a document's propositional content:
//!
//! - `UnbulletRule` only ever *joins* a list's items into one sentence, in
//!   their original source order, verbatim — it never reorders them,
//!   drops one, or changes a word inside one. Joining prose that already
//!   exists, in the order it already existed in, is exactly the kind of
//!   transform this workspace's tier discipline calls "merge adjacent text
//!   without reordering": safe to apply without a human in the loop.
//! - `BoldLabelStripRule` only ever deletes four bytes (`**` twice) — pure
//!   markup deletion, changing how a label *renders*, never what it reads
//!   as. Same category as the punctuation/case-only transforms this
//!   workspace's tier discipline reserves for Fix.
//!
//! Neither rule needs more than one meaning-preserving strategy per
//! finding (a qualifying list has exactly one join order — its own; a
//! bold-label bullet has exactly one way to strip its markers), so neither
//! rule's `fix` consults its `StrategyRng` argument — see
//! [`crate::Rule::fix`]'s own docs for why that is a legitimate use of the
//! API, not an oversight.
//!
//! # Shared block-tree helpers
//!
//! All three rules need to reason about a document's block *nesting*, not
//! just its flat block list — "is this list itself inside another list's
//! item", "what block, if any, immediately precedes this one at the same
//! nesting level", "is this list item's text held directly by it, or by a
//! nested container". [`block_parents`], [`previous_sibling`], and
//! [`innermost_list_item`] below are the small, private, block-tree
//! utilities every submodule shares for that reasoning; each is a pure
//! function of the block slice (and, for the last one, the parent index it
//! computed), with no rule-specific knowledge baked in.

mod bold_label_strip;
mod header_merge;
mod unbullet;

pub use bold_label_strip::BoldLabelStripRule;
pub use header_merge::HeaderMergeRule;
pub use unbullet::UnbulletRule;

use friction_core::span::contains_range;
use friction_core::{Block, BlockKind};

/// For every block in `blocks` (assumed pre-order / source-order, as
/// `friction-parse` always produces — see that crate's own module docs),
/// the index of its nearest *containing* block, or `None` for a top-level
/// block.
///
/// A single left-to-right pass with a stack of "currently open ancestor"
/// indices: a block's parent is whichever open ancestor's range still
/// contains it (ranges only ever shrink as blocks get more deeply nested,
/// so popping stops the first time containment fails).
fn block_parents(blocks: &[Block]) -> Vec<Option<usize>> {
    let mut parents = vec![None; blocks.len()];
    let mut stack: Vec<usize> = Vec::new();
    for (index, block) in blocks.iter().enumerate() {
        while let Some(&top) = stack.last() {
            if contains_range(&blocks[top].range, &block.range) {
                break;
            }
            stack.pop();
        }
        parents[index] = stack.last().copied();
        stack.push(index);
    }
    parents
}

/// The index of the block immediately before `index` that shares `index`'s
/// own parent (per `parents`, from [`block_parents`]) — i.e. `index`'s
/// previous sibling in source order, skipping over any block nested inside
/// that sibling. `None` if `index` is its parent's first child (or a
/// top-level block with nothing top-level before it).
///
/// This is deliberately not just `index - 1`: `blocks` is pre-order, so the
/// immediately-preceding *entry* in the slice is often a block that just
/// opened around `index` (its parent), not a true sibling — e.g. the first
/// item of a list is preceded in the flat block list by the list itself.
fn previous_sibling(index: usize, parents: &[Option<usize>]) -> Option<usize> {
    let target = parents[index];
    (0..index).rev().find(|&i| parents[i] == target)
}

/// The index of the [`BlockKind::ListItem`] block that directly or
/// transitively owns `block_index`'s prose: `block_index` itself if it is
/// already a list item (the tight-item case), otherwise the nearest
/// list-item ancestor (the loose-item case, where the item's text lives in
/// a nested paragraph). `None` if no ancestor (or `block_index` itself) is
/// a list item at all.
fn innermost_list_item(
    block_index: usize,
    blocks: &[Block],
    parents: &[Option<usize>],
) -> Option<usize> {
    let mut current = Some(block_index);
    while let Some(index) = current {
        if blocks[index].kind == BlockKind::ListItem {
            return Some(index);
        }
        current = parents[index];
    }
    None
}

/// A rough word-token count for `text`: the number of whitespace-delimited
/// chunks that contain at least one alphanumeric character (so a stray
/// punctuation-only chunk, e.g. a lone `"-"`, doesn't inflate the count).
///
/// Deliberately not a real tokenizer: every caller here only needs an
/// approximate "is this fragment short" gate, not an exact token count for
/// a metric, and paying for a `Tagger` call just to count words would tie
/// this cheap check to tagger availability and quality for no benefit.
fn count_word_tokens(text: &str) -> usize {
    text.split_whitespace()
        .filter(|chunk| chunk.chars().any(char::is_alphanumeric))
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(kind: BlockKind, range: std::ops::Range<usize>) -> Block {
        Block::new(kind, range)
    }

    fn list_item() -> BlockKind {
        BlockKind::ListItem
    }

    /// A flat sequence of sibling blocks (no nesting) all report `None` as
    /// their parent.
    #[test]
    fn block_parents_flat_siblings_have_no_parent() {
        let blocks = vec![
            block(BlockKind::Paragraph, 0..5),
            block(BlockKind::Paragraph, 5..10),
        ];
        assert_eq!(block_parents(&blocks), vec![None, None]);
    }

    /// A list's items report the list itself as their parent; the list
    /// reports no parent.
    #[test]
    fn block_parents_list_items_point_to_their_list() {
        let blocks = vec![
            block(
                BlockKind::List {
                    ordered: false,
                    start: None,
                },
                0..20,
            ),
            block(list_item(), 0..10),
            block(list_item(), 10..20),
        ];
        assert_eq!(block_parents(&blocks), vec![None, Some(0), Some(0)]);
    }

    /// A sublist nested inside an outer item's range reports that item as
    /// its parent, two levels deep.
    #[test]
    fn block_parents_handles_nested_lists() {
        let blocks = vec![
            block(
                BlockKind::List {
                    ordered: false,
                    start: None,
                },
                0..30,
            ), // outer list
            block(list_item(), 0..20), // outer item
            block(
                BlockKind::List {
                    ordered: false,
                    start: None,
                },
                8..20,
            ), // inner list
            block(list_item(), 8..20), // inner item
            block(list_item(), 20..30), // outer item 2
        ];
        let parents = block_parents(&blocks);
        assert_eq!(parents, vec![None, Some(0), Some(1), Some(2), Some(0)]);
    }

    /// `previous_sibling` finds the nearest same-parent block before
    /// `index`, skipping anything nested inside it, and returns `None` for
    /// a first child / first top-level block.
    #[test]
    fn previous_sibling_skips_nested_blocks_and_finds_true_sibling() {
        let blocks = vec![
            block(BlockKind::Paragraph, 0..10), // 0: top-level
            block(
                BlockKind::List {
                    ordered: false,
                    start: None,
                },
                11..40,
            ), // 1: top-level
            block(list_item(), 11..25),         // 2: child of 1
            block(
                BlockKind::List {
                    ordered: false,
                    start: None,
                },
                15..25,
            ), // 3: child of 2
            block(list_item(), 15..25),         // 4: child of 3
            block(list_item(), 25..40),         // 5: child of 1
        ];
        let parents = block_parents(&blocks);
        assert_eq!(previous_sibling(1, &parents), Some(0));
        assert_eq!(previous_sibling(0, &parents), None);
        assert_eq!(previous_sibling(5, &parents), Some(2));
        assert_eq!(previous_sibling(4, &parents), None);
    }

    /// `innermost_list_item` returns the block itself for a tight item,
    /// climbs to the nearest list-item ancestor for a loose item's nested
    /// paragraph, and `None` for a block with no list-item ancestor at all.
    #[test]
    fn innermost_list_item_resolves_tight_loose_and_absent_cases() {
        let blocks = vec![
            block(BlockKind::Paragraph, 0..5), // 0: unrelated top-level paragraph
            block(list_item(), 6..20),         // 1: tight item
            block(list_item(), 20..40),        // 2: loose item
            block(BlockKind::Paragraph, 24..38), // 3: nested paragraph inside item 2
        ];
        let parents = block_parents(&blocks);
        assert_eq!(innermost_list_item(0, &blocks, &parents), None);
        assert_eq!(innermost_list_item(1, &blocks, &parents), Some(1));
        assert_eq!(innermost_list_item(3, &blocks, &parents), Some(2));
    }

    /// `count_word_tokens` counts whitespace-delimited chunks with at
    /// least one alphanumeric character, ignoring punctuation-only chunks.
    #[test]
    fn count_word_tokens_ignores_punctuation_only_chunks() {
        assert_eq!(count_word_tokens("Fast, reliable, and documented"), 4);
        assert_eq!(count_word_tokens("- -- -"), 0);
        assert_eq!(count_word_tokens(""), 0);
        assert_eq!(count_word_tokens("one"), 1);
    }
}
