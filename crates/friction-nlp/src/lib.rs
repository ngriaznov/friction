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

/// Light-verb-construction tables and matching.
///
/// Shared between runtime detection (`friction-harness::pivot` and a
/// detection crate's own LVC channel, both built on
/// [`lvc::classify_candidate`]) and offline mining.
pub mod lvc;

// --- POS tagging, morphology, and inflection (owned by the tagging agent;
// see src/tag.rs, src/tag_perceptron.rs, src/tag_nlprule.rs,
// src/inflect.rs, src/chunk.rs) ---
mod chunk;
mod inflect;
mod tag;
#[cfg(feature = "nlprule")]
mod tag_nlprule;
mod tag_perceptron;

pub use chunk::{
    Clause, ClauseChunks, CoordinationGroup, FINITE_VERB_TAGS, chunk_clauses, coordination_groups,
    has_finite_verb, is_complete_after_deletion, is_imperative_initial, opens_with_binding_cue,
    overlaps_counted_enumeration,
};
pub use inflect::{WordClass, agreeing_forms, inflect};
pub use tag::{PosTag, TaggedToken, Tagger, classify_token_kind, coarse_tag};
#[cfg(feature = "nlprule")]
pub use tag_nlprule::{NlpruleTagger, TagError};
#[cfg(feature = "train-tooling")]
pub use tag_perceptron::train_support;
pub use tag_perceptron::{PerceptronTagError, PerceptronTagger};
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
