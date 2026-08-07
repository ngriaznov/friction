//! The shared error type for this crate.

/// Errors produced while building a [`crate::MatchEngine`] or scanning a
/// document with one.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum MatchError {
    /// The literal automaton failed to build from the inventory pack's
    /// literal-eligible patterns.
    #[error("failed to build the literal automaton: {0}")]
    Automaton(#[from] aho_corasick::BuildError),
    /// A prose range failed to slice against the document's source.
    #[error(transparent)]
    Core(#[from] friction_core::CoreError),
}
