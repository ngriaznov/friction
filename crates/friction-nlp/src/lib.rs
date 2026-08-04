//! Sentence segmentation, POS tagging, inflection, and dependency parsing.
//!
//! Provides [`Segmenter`] (implemented by [`SrxSegmenter`]), `trait
//! Tagger`, the inflection service, the dependency-parsing types
//! ([`DepParser`]), and [`dep_arceager`] — the arc-eager transition system
//! a trainer drives to turn a gold [`SentenceParse`] into training data
//! (or, later, an actual [`DepParser`] implementation), not a ready-to-use
//! parser itself.

mod segment;
mod segment_srx;
mod weights_bin;

pub use segment::{SegmentError, Segmenter, segment_document};
pub use segment_srx::SrxSegmenter;

mod lexicon;

pub use lexicon::{LEXICON_EN, Lexicon, LexiconError, PastForms, VerbForms, WordSet};

/// Light-verb-construction tables and matching.
///
/// Shared between runtime detection (`friction-harness::pivot` and a
/// detection crate's own LVC channel, both built on
/// [`lvc::classify_candidate`]) and offline mining.
pub mod lvc;

// --- POS tagging, morphology, and inflection; see src/tag.rs,
// src/tag_perceptron.rs, src/inflect.rs, src/chunk.rs ---
mod chunk;
mod inflect;
mod tag;
mod tag_perceptron;

pub use chunk::{
    Clause, ClauseChunks, CoordinationGroup, FINITE_VERB_TAGS, chunk_clauses, coordination_groups,
    has_finite_verb, is_complete_after_deletion, is_imperative_initial, opens_with_binding_cue,
    overlaps_counted_enumeration,
};
pub use inflect::{WordClass, agreeing_forms, inflect, irregular_verb_base, lemmatize};
pub use tag::{PosTag, TaggedToken, Tagger, classify_token_kind, coarse_tag};
#[cfg(feature = "train-tooling")]
pub use tag_perceptron::train_support;
pub use tag_perceptron::{PerceptronTagError, PerceptronTagger, pack_perceptron_tagger_bin};
// --- end tagging block ---

// --- dependency parsing (src/dep.rs, src/dep_arceager.rs,
// src/dep_perceptron.rs) ---
mod dep;
mod dep_perceptron;

pub use dep::{Confidence, DepEdge, DepParseError, DepParser, DepRelation, SentenceParse};
#[cfg(feature = "train-tooling")]
pub use dep_perceptron::train_support as dep_train_support;
pub use dep_perceptron::{PerceptronParseError, PerceptronParser, pack_perceptron_parser_bin};

/// The arc-eager transition system a trainer drives.
///
/// [`Configuration`](dep_arceager::Configuration) and
/// [`Transition`](dep_arceager::Transition), plus the static
/// [`oracle`](dep_arceager::oracle) that turns a gold [`SentenceParse`]
/// into the transition sequence a trainer learns from. Kept in its own
/// namespace rather than flattened into this crate's root re-exports (the
/// convention elsewhere on this page): `oracle`/`derive` are generic
/// enough names that flattening would risk shadowing a future export, and
/// every consumer is expected to spell out `dep_arceager::` anyway to
/// keep "the transition system" and "the parser" visually distinct.
pub mod dep_arceager;
// --- end dependency-parsing block ---
