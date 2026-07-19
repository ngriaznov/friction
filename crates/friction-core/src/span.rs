//! Byte-span validation helpers.
//!
//! Every span in this crate is a half-open [`Range<usize>`] of byte offsets
//! into a [`crate::Document`]'s original source text. These helpers are the
//! single place that checks the two ways a span can be dishonest: pointing
//! outside the source, or splitting a UTF-8 code point in half.

use std::ops::Range;

use crate::error::CoreError;

/// Validates that `range` is a legal byte span into `source`: `start <=
/// end`, `end <= source.len()`, and both endpoints fall on UTF-8 character
/// boundaries.
///
/// # Errors
/// Returns [`CoreError::InvalidRange`] if `range.start > range.end`,
/// [`CoreError::RangeOutOfBounds`] if `range.end > source.len()`, or
/// [`CoreError::NotCharBoundary`] if either endpoint splits a UTF-8
/// character.
pub fn validate_range(source: &str, range: &Range<usize>) -> Result<(), CoreError> {
    if range.start > range.end {
        return Err(CoreError::InvalidRange {
            start: range.start,
            end: range.end,
        });
    }
    if range.end > source.len() {
        return Err(CoreError::RangeOutOfBounds {
            range: range.clone(),
            source_len: source.len(),
        });
    }
    if !source.is_char_boundary(range.start) || !source.is_char_boundary(range.end) {
        return Err(CoreError::NotCharBoundary {
            range: range.clone(),
        });
    }
    Ok(())
}

/// Validates `range` against `source` (see [`validate_range`]) and returns
/// the slice of `source` it addresses.
///
/// # Errors
/// See [`validate_range`].
pub fn slice<'src>(source: &'src str, range: &Range<usize>) -> Result<&'src str, CoreError> {
    validate_range(source, range)?;
    Ok(&source[range.start..range.end])
}

/// Returns `true` if `inner` is fully contained within `outer`: `outer.start
/// <= inner.start` and `inner.end <= outer.end`.
#[must_use]
pub const fn contains_range(outer: &Range<usize>, inner: &Range<usize>) -> bool {
    outer.start <= inner.start && inner.end <= outer.end
}

/// Returns `true` if `a` and `b` overlap.
///
/// Uses half-open interval overlap for non-empty ranges. A zero-length
/// range (an insertion point) is treated as overlapping any range that
/// contains that offset, and as overlapping another zero-length range only
/// when they sit at the exact same offset — two insertions at the same
/// point conflict (a rule must pick which comes first), but an insertion at
/// the boundary immediately after a non-empty range does not overlap it.
// Each arm intentionally checks whether a single point falls in a
// half-open range (`x.start >= other.start && x.start < other.end`), not a
// symmetric pairwise comparison of both ranges' starts and ends — clippy's
// heuristic misreads that as a copy-paste bug.
#[must_use]
#[allow(clippy::suspicious_operation_groupings)]
pub fn ranges_overlap(a: &Range<usize>, b: &Range<usize>) -> bool {
    match (a.is_empty(), b.is_empty()) {
        (true, true) => a.start == b.start,
        (true, false) => a.start >= b.start && a.start < b.end,
        (false, true) => b.start >= a.start && b.start < a.end,
        (false, false) => a.start < b.end && b.start < a.end,
    }
}

/// A value that carries a byte span into a [`crate::Document`]'s original
/// source text.
///
/// Implemented by every level of the [`crate::Document`] structure —
/// [`crate::Block`], [`crate::ProseUnit`], [`crate::Sentence`],
/// [`crate::Token`] — as well as [`crate::Patch`] and [`crate::Finding`],
/// so span-generic code (conflict detection, diagnostics rendering) can be
/// written once against this trait.
pub trait Spanned {
    /// The byte range this value occupies in the original source.
    fn range(&self) -> Range<usize>;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Valid ranges (including empty and full-source) pass.
    #[test]
    fn validate_range_accepts_valid_spans() {
        let source = "hello world";
        assert!(validate_range(source, &(0..5)).is_ok());
        assert!(validate_range(source, &(0..0)).is_ok());
        assert!(validate_range(source, &(0..source.len())).is_ok());
        assert!(validate_range(source, &(11..11)).is_ok());
    }

    /// `start > end` is rejected as `InvalidRange`.
    ///
    /// The inverted range below is the point of the test, not a mistake.
    #[allow(clippy::reversed_empty_ranges)]
    #[test]
    fn validate_range_rejects_inverted_range() {
        let err = validate_range("hello", &(3..1)).unwrap_err();
        assert!(matches!(err, CoreError::InvalidRange { start: 3, end: 1 }));
    }

    /// A range past the end of the source is rejected as
    /// `RangeOutOfBounds`.
    #[test]
    fn validate_range_rejects_out_of_bounds() {
        let err = validate_range("hello", &(0..10)).unwrap_err();
        assert!(matches!(
            err,
            CoreError::RangeOutOfBounds { source_len: 5, .. }
        ));
    }

    /// A range splitting a multi-byte UTF-8 character is rejected as
    /// `NotCharBoundary`.
    #[test]
    fn validate_range_rejects_non_char_boundary() {
        let source = "café"; // 'é' is a 2-byte UTF-8 character.
        let boundary = source.len() - 1;
        let err = validate_range(source, &(0..boundary)).unwrap_err();
        assert!(matches!(err, CoreError::NotCharBoundary { .. }));
    }

    /// `slice` returns the addressed substring for a valid range.
    #[test]
    fn slice_returns_addressed_text() {
        let source = "hello world";
        assert_eq!(slice(source, &(0..5)).unwrap(), "hello");
        assert_eq!(slice(source, &(6..11)).unwrap(), "world");
    }

    /// `slice` propagates validation errors instead of panicking.
    #[test]
    fn slice_propagates_validation_errors() {
        assert!(slice("hello", &(0..99)).is_err());
    }

    /// `contains_range`: containment, exact equality, and non-containment.
    #[test]
    fn contains_range_checks_containment() {
        assert!(contains_range(&(0..10), &(2..8)));
        assert!(contains_range(&(0..10), &(0..10)));
        assert!(!contains_range(&(0..10), &(5..15)));
        assert!(!contains_range(&(2..8), &(0..10)));
    }

    /// `ranges_overlap`: overlapping, adjacent (non-overlapping), and
    /// disjoint non-empty ranges.
    #[test]
    fn ranges_overlap_non_empty() {
        assert!(ranges_overlap(&(0..5), &(4..10)));
        assert!(!ranges_overlap(&(0..5), &(5..10)));
        assert!(!ranges_overlap(&(0..5), &(10..15)));
    }

    /// `ranges_overlap`: an insertion point overlaps a range that contains
    /// it (including sitting exactly at the range's start), but not one it
    /// merely abuts at the range's end.
    #[test]
    fn ranges_overlap_insertion_vs_range() {
        assert!(ranges_overlap(&(3..3), &(0..5)));
        assert!(ranges_overlap(&(0..5), &(3..3)));
        assert!(ranges_overlap(&(0..5), &(0..0)));
        assert!(!ranges_overlap(&(5..5), &(0..5)));
        assert!(!ranges_overlap(&(0..5), &(5..5)));
    }

    /// `ranges_overlap`: two insertions overlap only at the exact same
    /// offset.
    #[test]
    fn ranges_overlap_insertion_vs_insertion() {
        assert!(ranges_overlap(&(3..3), &(3..3)));
        assert!(!ranges_overlap(&(3..3), &(4..4)));
    }
}
