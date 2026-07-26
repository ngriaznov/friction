//! Register measurement and rephrasing.
//!
//! Two halves, deliberately separable: [`features`] counts, [`transduce`]
//! proposes rewrites that move those counts. Kept apart because a
//! miscounted feature is invisible in the output -- it would surface only
//! as a wrongly-optimized rewrite.

pub mod features;
pub mod transduce;
