//! Metric vector computation and human-envelope estimation.
//!
//! Provides the metric vector v1, pure deterministic metric functions,
//! envelope estimation, and separation reporting.

// --- structural/symmetry metrics (owned by the symmetry agent; see
// src/symmetry.rs) ---
mod symmetry;

pub use symmetry::{bullet_parallelism, participial_closer_rate, triad_rate};
// --- end symmetry block ---

// --- lexical marker metrics (owned by the lexical agent; see
// src/lexical.rs) ---
mod lexical;

pub use lexical::{
    contraction_counts, contraction_pairs, contraction_ratio, discourse_marker_density,
    not_just_but_rate, ritual_marker_rate,
};
// --- end lexical block ---

// --- rhythm/shape metrics (owned by the rhythm agent; see src/rhythm.rs)
// ---
mod rhythm;

pub use rhythm::{
    RhythmStats, em_dash_density, paragraph_shape, semicolon_density, sentence_length_by_document,
    sentence_length_by_paragraph,
};
// --- end rhythm block ---

// --- structural/mined-phrase/opener "signal" metrics (owned by the
// signals agent; see src/signals.rs) ---
mod signals;

pub use signals::{
    bold_span_density, heading_density, human_favored_phrase_rate, list_item_density,
    llm_favored_phrase_rate, sentence_opener_repeat_rate, top_opener_concentration,
};
// --- end signals block ---

// --- integration: wires the four families above into one MetricVector,
// document-wide and per paragraph (owned by the integrator) ---
mod compute;

pub use compute::{compute, compute_by_paragraph};
// --- end integration block ---
