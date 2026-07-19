//! Shared error type for `friction-core`.

use std::ops::Range;

/// Errors produced while constructing or validating core domain types.
///
/// Every variant describes a violation of the invariant that every span is
/// a byte range into the original source text, or another structural
/// invariant of a [`crate::Document`]; none are recoverable by retrying the
/// same input unchanged.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum CoreError {
    /// A range's `start` is greater than its `end`.
    #[error("range start {start} is greater than range end {end}")]
    InvalidRange {
        /// Offending start offset.
        start: usize,
        /// Offending end offset.
        end: usize,
    },

    /// A range extends past the end of the source it was checked against.
    #[error("range {range:?} is out of bounds for a source of length {source_len}")]
    RangeOutOfBounds {
        /// The out-of-bounds range.
        range: Range<usize>,
        /// Length in bytes of the source the range was checked against.
        source_len: usize,
    },

    /// A range's start or end does not fall on a UTF-8 character boundary.
    #[error("range {range:?} does not fall on a UTF-8 character boundary")]
    NotCharBoundary {
        /// The offending range.
        range: Range<usize>,
    },

    /// A range is not fully contained within the range of its declared
    /// parent (a block, prose unit, or sentence).
    #[error("range {inner:?} is not contained within parent range {outer:?}")]
    RangeNotContained {
        /// The child range.
        inner: Range<usize>,
        /// The parent range it should be contained in.
        outer: Range<usize>,
    },

    /// A [`crate::ProseUnit`] refers to a block index that does not exist
    /// in the owning document.
    #[error("block index {index} is out of bounds ({len} blocks)")]
    BlockIndexOutOfBounds {
        /// The offending index.
        index: usize,
        /// Number of blocks available.
        len: usize,
    },

    /// An [`crate::Envelope`]'s lower bound exceeds its upper bound, or
    /// either bound is non-finite.
    #[error("invalid envelope bounds: lo={lo}, hi={hi}")]
    InvalidEnvelope {
        /// Lower bound.
        lo: f64,
        /// Upper bound.
        hi: f64,
    },
}
