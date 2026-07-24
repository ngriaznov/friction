//! [`HarnessError`]: this crate's error type.

/// Errors produced while scoring text or loading the harness's tagger
/// model.
#[derive(Debug, thiserror::Error)]
pub enum HarnessError {
    /// The embedded tagger model failed to load.
    #[error("failed to load the embedded tagger model: {0}")]
    TaggerLoad(#[source] friction_nlp::PerceptronTagError),
    /// Parsing the input into a document failed.
    #[error(transparent)]
    Parse(#[from] friction_parse::ParseError),
    /// Segmenting the parsed document into sentences failed.
    #[error(transparent)]
    Segment(#[from] friction_nlp::SegmentError),
}
