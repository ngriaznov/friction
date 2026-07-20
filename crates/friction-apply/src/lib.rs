//! Patch application: conflict resolution, atomic apply, and the fixpoint
//! driver.
//!
//! This crate owns the mechanics of turning a set of rules' proposed
//! patches into applied text — it defines no rules itself (see
//! `friction-rules` for the [`friction_rules::Rule`] trait every family
//! implements) and no metrics (see `friction-metrics`).
//!
//! # The round pipeline
//!
//! [`run_fixpoint`] is the entry point: it runs [`MAX_ROUNDS`] rounds of
//! parse -> compute the document's [`friction_core::MetricVector`] -> gate
//! every rule -> scan the ones that aren't `Off` -> fix the `Fix`-tier
//! findings a `Fix`-gated rule's budget allows -> resolve conflicts among
//! every rule's proposed patches -> apply the survivors in one atomic pass
//! -> re-parse for the next round, stopping early the first time a round
//! applies zero patches.
//!
//! # Conflict resolution
//!
//! [`resolve_round`] sorts a round's candidate patches by `(byte position
//! ascending, patch length descending, rule-family priority ascending)` —
//! "leftmost-longest, then rule priority" — and greedily accepts each one
//! that doesn't overlap an already-accepted patch, dropping the rest.
//! [`apply_patches`] then splices the accepted, non-overlapping patches
//! into the original source in one pass, right-to-left, so no patch's byte
//! range ever needs adjusting for an earlier patch's effect.
//!
//! # Span honesty and safety
//!
//! Every patch is validated against its round's source (in bounds, on a
//! UTF-8 character boundary) before it is ever considered for conflict
//! resolution; an invalid patch is dropped, never applied and never
//! panicked on. Because every accepted patch's range is already
//! byte-boundary-valid and `replacement` is a Rust `String` (already
//! guaranteed valid UTF-8 by the type system), splicing it into the source
//! can never produce invalid UTF-8 output.
//!
//! # The public fix entry point
//!
//! [`fix_document`] is [`run_fixpoint`] wired to the fixed tranche-1 rule
//! set ([`registered_rules`]: lexical, connective, contraction) in a
//! deterministic registration order; [`FixEngine`] additionally loads the
//! model-backed segmenter/tagger and the shipped envelope pack once and
//! reuses them across calls, the shape a real caller (CLI, corpus
//! tooling) wants. [`touched_original_ranges`] maps a (possibly
//! multi-round) run's applied patches back to spans of the *original*
//! input, for reporting tools that need to know what was actually
//! touched.

mod conflict;
mod coverage;
mod driver;
mod fix;

pub use conflict::{Candidate, apply_patches, resolve_round};
pub use coverage::touched_original_ranges;
pub use driver::{ApplyError, FixpointReport, MAX_ROUNDS, RoundReport, run_fixpoint};
pub use fix::{EngineError, FixEngine, fix_document, registered_rules};
