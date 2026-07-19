//! Patch application: conflict resolution, atomic apply, and the fixpoint
//! driver.
//!
//! Provides per-round patch collection, conflict resolution
//! (leftmost-longest, then rule priority), atomic apply, re-parse between
//! rounds, and a fixpoint driver bounded at 4 rounds (idempotence is swept
//! in CI against this crate's output).
//!
//! This crate is currently a scaffold stub.
