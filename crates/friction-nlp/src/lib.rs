//! Sentence segmentation, POS tagging, inflection, and dependency parsing.
//!
//! Provides [`Segmenter`] (implemented by [`SrxSegmenter`]), `trait
//! Tagger`, the inflection service, and [`DepParser`] with
//! [`HeuristicParser`] (always available) and, behind the `onnx` cargo
//! feature, `OnnxParser`.
//!
//! Segmentation, tagging, inflection, and dependency parsing are all
//! implemented in this crate.

mod segment;
mod segment_srx;

pub use segment::{SegmentError, Segmenter, segment_document};
pub use segment_srx::SrxSegmenter;

// --- POS tagging, morphology, and inflection (owned by the tagging agent;
// see src/tag.rs, src/tag_nlprule.rs, src/inflect.rs) ---
mod inflect;
mod tag;
mod tag_nlprule;

pub use inflect::inflect;
pub use tag::{PosTag, TaggedToken, Tagger};
pub use tag_nlprule::{NlpruleTagger, TagError};
// --- end tagging block ---

// --- dependency parsing (owned by the dep-parser agent; see src/dep.rs,
// src/dep_heuristic.rs, src/dep_onnx.rs) ---
mod dep;
mod dep_heuristic;
#[cfg(feature = "onnx")]
mod dep_onnx;

pub use dep::{
    Confidence, DepEdge, DepParseError, DepParser, DepRelation, SentenceParse, same_subject,
    subject_text,
};
pub use dep_heuristic::HeuristicParser;
#[cfg(feature = "onnx")]
pub use dep_onnx::{OnnxParser, softmax_top2_margin};
// --- end dependency-parsing block ---
