//! [`ParseError`]: `friction-parse`'s error type.

use friction_core::CoreError;

/// Errors produced while parsing markdown source into a
/// [`friction_core::Document`].
///
/// Two failure modes: the extracted block/prose structure failing
/// [`friction_core::Document::new`]'s span-honesty validation (which a
/// correct extraction should never trigger, since `pulldown-cmark` is
/// documented as total over UTF-8 text), and `pulldown-cmark` itself
/// panicking on some input despite that documented totality — see
/// [`ParseError::UnderlyingParserPanicked`]. `#[non_exhaustive]` leaves
/// room for future parse-time diagnostics without a breaking change.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ParseError {
    /// The extracted document structure violated a `friction-core`
    /// span-honesty invariant.
    #[error(transparent)]
    Core(#[from] CoreError),
    /// `pulldown-cmark`'s own event-stream construction panicked while
    /// parsing `source` — an upstream parser-internal invariant
    /// violation on adversarial input (`friction-parse`'s own fuzz suite,
    /// `fuzz/fuzz_targets/fuzz_parse.rs`, found and minimized a 19-byte
    /// repro of one such case: a heading-attribute-style `{...}` span
    /// nested inside a loose list item, which trips a `tree.rs` internal
    /// assertion in `pulldown-cmark` 0.13.4). [`crate::parse`] catches
    /// this panic at the `pulldown-cmark` boundary via
    /// `std::panic::catch_unwind` and surfaces it here instead, so a
    /// single pathological document can never abort a caller's process —
    /// see that function's own doc comment.
    #[error("the underlying markdown parser panicked while parsing this input: {0}")]
    UnderlyingParserPanicked(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `ParseError::Core` wraps and displays the underlying `CoreError`.
    #[test]
    fn parse_error_wraps_core_error() {
        let core = CoreError::InvalidRange { start: 5, end: 1 };
        let err = ParseError::from(core);
        assert!(
            err.to_string()
                .contains("range start 5 is greater than range end 1")
        );
    }
}
