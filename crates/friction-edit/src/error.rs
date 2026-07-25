//! [`EditError`]: everything that can go wrong loading the engine or
//! running it over a document.

/// An error from constructing an [`crate::Engine`] or running it.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum EditError {
    /// `source` failed to parse as markdown.
    #[error("failed to parse document: {0}")]
    Parse(#[from] friction_parse::ParseError),
    /// `source` failed to segment into sentences.
    #[error("failed to segment document: {0}")]
    Segment(#[from] friction_nlp::SegmentError),
    /// The embedded part-of-speech tagger failed to load.
    #[error("failed to load tagger: {0}")]
    Tagger(#[from] friction_nlp::PerceptronTagError),
    /// The embedded dependency parser failed to load.
    #[error("failed to load dependency parser: {0}")]
    Parser(#[from] friction_nlp::PerceptronParseError),
    /// A document byte range failed validation.
    #[error("document span error: {0}")]
    Core(#[from] friction_core::CoreError),
}
