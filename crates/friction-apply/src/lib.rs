//! Patch application: conflict resolution, atomic apply, and plain
//! round/fixpoint report shapes.
//!
//! This crate used to also own the rule engine's fixpoint driver (parse ->
//! compute metrics -> gate every rule -> scan -> fix -> resolve conflicts
//! -> apply -> re-parse, bounded and reported). That driver, and the
//! `registered_rules`/`FixEngine`/`fix_document` entry points built on it,
//! were coupled to the now-retired `friction-rules`/`friction-plan`
//! abstraction (`Rule`/`Gate`/`RuleContext`/`Plan`) and are gone along
//! with it — the real repair engine lives in `friction-edit`, which runs
//! its own bounded, fixed-order pass loop directly rather than through
//! this crate.
//!
//! What remains is the reusable mechanics any round-based patch pipeline
//! needs, kept dependency-free (only [`friction_core`] types):
//!
//! # Conflict resolution
//!
//! [`resolve_round`] sorts a round's candidate [`Candidate`]s by `(byte
//! position ascending, patch length descending, priority ascending)` —
//! "leftmost-longest, then priority" — and greedily accepts each one that
//! doesn't overlap an already-accepted patch, dropping the rest.
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
//! # Report shapes
//!
//! [`RoundReport`]/[`FixpointReport`]/[`MAX_ROUNDS`] are plain,
//! dependency-free report shapes a caller running its own round pipeline
//! can populate and return. [`touched_original_ranges`] maps a (possibly
//! multi-round) run's applied patches back to spans of the *original*
//! input, for reporting tools that need to know what was actually
//! touched.

mod conflict;
mod coverage;
mod driver;

pub use conflict::{Candidate, apply_patches, resolve_round};
pub use coverage::touched_original_ranges;
pub use driver::{FixpointReport, MAX_ROUNDS, RoundReport};
