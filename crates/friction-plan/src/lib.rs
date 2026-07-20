//! Fix planning: envelope deltas to an ordered, budgeted rule schedule.
//!
//! [`Plan`] turns a document's [`friction_core::MetricVector`] and a
//! genre's envelope bands into an ordered schedule of `(family,
//! per-family advisory budget)` entries — which of the six rule families
//! ([`friction_rules::RuleFamily`]) should act this run, in what fixed
//! order, and how hard. See [`Plan`]'s own docs for the exact order, its
//! rationale, and what "advisory" means (a `Plan` never replaces a rule's
//! own gating — every rule still gates and budgets itself independently,
//! every round, from the real document).
//!
//! `crate::mapping` centralizes the family -> driving-metric table this
//! crate uses to compute those advisory budgets, transcribed from each
//! family's real rule-level `gate` implementations in `friction-rules`
//! (see that module's own docs for exactly how, and where a real rule has
//! no excess-scaled budget to transcribe from).
//!
//! A `Plan` serializes two ways: [`std::fmt::Display`] for a
//! human-readable table (e.g. a CLI `explain` report) and
//! [`serde::Serialize`] for machine-readable JSON — both deterministic,
//! pure functions of the `Plan`'s own fields (see [`Plan`]'s docs).

mod mapping;
mod plan;

pub use plan::{MetricDelta, Plan, PlanEntry};
