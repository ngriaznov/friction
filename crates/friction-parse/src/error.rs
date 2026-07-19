//! [`ParseError`]: `friction-parse`'s error type.

use friction_core::CoreError;

/// Errors produced while parsing markdown source into a
/// [`friction_core::Document`].
///
/// The only failure mode today is the extracted block/prose structure
/// failing [`friction_core::Document::new`]'s span-honesty validation
/// — which a correct extraction should never trigger, since
/// `pulldown-cmark` is total over UTF-8 text. `#[non_exhaustive]` leaves
/// room for future parse-time diagnostics without a breaking change.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ParseError {
    /// The extracted document structure violated a `friction-core`
    /// span-honesty invariant.
    #[error(transparent)]
    Core(#[from] CoreError),
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
