//! Metric vector computation and human-envelope estimation.
//!
//! Provides the metric vector v1, pure deterministic metric functions,
//! envelope estimation, and separation reporting.

// Structural/symmetry metrics.
mod symmetry;

pub use symmetry::{bullet_parallelism, participial_closer_rate, triad_rate};

// Lexical marker metrics.
mod lexical;

pub use lexical::{
    contraction_counts, contraction_pairs, contraction_ratio, discourse_marker_density,
    not_just_but_rate, ritual_marker_rate,
};

// Rhythm/shape metrics.
mod rhythm;

pub use rhythm::{
    RhythmStats, em_dash_density, paragraph_shape, semicolon_density, sentence_length_by_document,
};

// Structural/mined-phrase/opener signal metrics.
mod signals;

pub use signals::{
    bold_span_density, heading_density, human_favored_phrase_rate, list_item_density,
    llm_favored_phrase_rate, sentence_opener_repeat_rate, top_opener_concentration,
};

// Wires the four families above into one MetricVector, document-wide and per paragraph.
mod compute;

pub use compute::{compute, compute_by_paragraph};
