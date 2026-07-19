//! [`Patch`]: a machine-applicable edit, plus overlap detection.

use std::ops::Range;

use crate::error::CoreError;
use crate::rule::RuleId;
use crate::span::{self, Spanned};

/// A single proposed edit: replace the bytes at `range` in the original
/// source with `replacement`.
///
/// `range` is always a byte range into the source of the round the patch
/// was produced against; patches are collected per round and applied
/// atomically before the document is re-parsed by `friction-apply`.
/// Conflict *resolution* (leftmost-longest, then rule priority) is
/// `friction-apply`'s responsibility — this crate only provides overlap
/// *detection* via [`find_overlaps`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Patch {
    /// Byte range in the original source this patch replaces.
    pub range: Range<usize>,
    /// Text to substitute for `range`.
    pub replacement: String,
    /// The rule that produced this patch.
    pub rule: RuleId,
    /// Meaning-preservation tier.
    pub tier: Tier,
}

impl Patch {
    /// Creates a new patch.
    #[must_use]
    pub fn new(
        range: Range<usize>,
        replacement: impl Into<String>,
        rule: RuleId,
        tier: Tier,
    ) -> Self {
        Self {
            range,
            replacement: replacement.into(),
            rule,
            tier,
        }
    }

    /// Validates this patch's `range` against `source`.
    ///
    /// # Errors
    /// Returns [`CoreError`] if `range` is out of bounds for `source` or
    /// splits a UTF-8 character.
    pub fn validate(&self, source: &str) -> Result<(), CoreError> {
        span::validate_range(source, &self.range)
    }

    /// `true` if this patch deletes its range without inserting any
    /// replacement text.
    #[must_use]
    pub const fn is_deletion(&self) -> bool {
        self.replacement.is_empty()
    }

    /// `true` if this patch inserts text without consuming any source
    /// (`range` is empty).
    #[must_use]
    pub fn is_insertion(&self) -> bool {
        self.range.is_empty()
    }
}

impl Spanned for Patch {
    fn range(&self) -> Range<usize> {
        self.range.clone()
    }
}

/// Meaning-preservation tier of a [`Patch`] or [`crate::Finding`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Tier {
    /// Machine-applicable: meaning-preserving by construction. Applied
    /// automatically by `friction fix`.
    ///
    /// Ordered before [`Tier::Suggest`] so a `Vec<Tier>` (or anything keyed
    /// by it) sorts Fix-tier items first.
    Fix,
    /// Diagnostic only: may reorder or drop propositional content, so it is
    /// never applied automatically. Surfaced by `friction fix --suggest` /
    /// `friction check`.
    Suggest,
}

/// Finds every pair of overlapping patches in `patches`.
///
/// Returns index pairs `(i, j)` with `i < j` into `patches`, in ascending
/// order of `i` then `j` — deterministic for a given input slice regardless
/// of the patches' own contents. Two ranges overlap per
/// [`span::ranges_overlap`]: zero-length (insertion) ranges overlap any
/// range containing that offset, including another insertion at the exact
/// same offset.
///
/// This only *detects* conflicts; *resolving* them (leftmost-longest, then
/// rule priority) is `friction-apply`'s responsibility.
#[must_use]
pub fn find_overlaps(patches: &[Patch]) -> Vec<(usize, usize)> {
    let mut overlaps = Vec::new();
    for i in 0..patches.len() {
        for j in (i + 1)..patches.len() {
            if span::ranges_overlap(&patches[i].range, &patches[j].range) {
                overlaps.push((i, j));
            }
        }
    }
    overlaps
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rule(id: &'static str) -> RuleId {
        RuleId::new(id)
    }

    /// A patch reports its own range via `Spanned` and validates against a
    /// source that contains it.
    #[test]
    fn patch_validate_accepts_in_bounds_range() {
        let patch = Patch::new(0..5, "Hi", rule("lexical.hello"), Tier::Fix);
        assert!(patch.validate("hello world").is_ok());
        assert_eq!(patch.range(), 0..5);
    }

    /// An out-of-bounds patch range is rejected.
    #[test]
    fn patch_validate_rejects_out_of_bounds_range() {
        let patch = Patch::new(0..50, "Hi", rule("lexical.hello"), Tier::Fix);
        assert!(patch.validate("hello").is_err());
    }

    /// `is_deletion` / `is_insertion` classify a patch by its shape.
    #[test]
    fn patch_classifies_deletion_and_insertion() {
        let deletion = Patch::new(0..5, "", rule("lexical.filler"), Tier::Fix);
        assert!(deletion.is_deletion());
        assert!(!deletion.is_insertion());

        let insertion = Patch::new(5..5, "!", rule("contraction.insert"), Tier::Fix);
        assert!(insertion.is_insertion());
        assert!(!insertion.is_deletion());

        let substitution = Patch::new(0..5, "howdy", rule("lexical.hello"), Tier::Fix);
        assert!(!substitution.is_deletion());
        assert!(!substitution.is_insertion());
    }

    /// `Tier` orders `Fix` before `Suggest`, so tiering is meaningful for
    /// deterministic sorting.
    #[test]
    fn tier_orders_fix_before_suggest() {
        assert!(Tier::Fix < Tier::Suggest);
    }

    /// `find_overlaps` returns no pairs for disjoint patches.
    #[test]
    fn find_overlaps_reports_none_for_disjoint_patches() {
        let patches = vec![
            Patch::new(0..5, "a", rule("r1"), Tier::Fix),
            Patch::new(5..10, "b", rule("r2"), Tier::Fix),
            Patch::new(20..25, "c", rule("r3"), Tier::Fix),
        ];
        assert!(find_overlaps(&patches).is_empty());
    }

    /// `find_overlaps` reports every overlapping pair, indices ascending.
    #[test]
    fn find_overlaps_reports_all_overlapping_pairs() {
        let patches = vec![
            Patch::new(0..10, "a", rule("r1"), Tier::Fix),
            Patch::new(5..15, "b", rule("r2"), Tier::Fix),
            Patch::new(8..12, "c", rule("r3"), Tier::Suggest),
            Patch::new(100..110, "d", rule("r4"), Tier::Fix),
        ];
        assert_eq!(find_overlaps(&patches), vec![(0, 1), (0, 2), (1, 2)]);
    }

    /// `find_overlaps` on an empty or single-patch slice returns no pairs.
    #[test]
    fn find_overlaps_handles_small_inputs() {
        assert!(find_overlaps(&[]).is_empty());
        let one = vec![Patch::new(0..5, "a", rule("r1"), Tier::Fix)];
        assert!(find_overlaps(&one).is_empty());
    }
}
