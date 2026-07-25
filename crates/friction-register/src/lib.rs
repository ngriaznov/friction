//! Register measurement and rephrasing.
//!
//! Two halves, deliberately separable. [`features`] counts the
//! register-marking constructions in a sentence; [`transduce`] proposes
//! rewrites that move those counts. Measurement never depends on rewriting,
//! so a counting bug and a rewriting bug can be told apart — which matters
//! because a miscounted feature is invisible in the output text and only
//! shows up as a rewrite that optimizes toward the wrong place.

pub mod features;
pub mod transduce;
