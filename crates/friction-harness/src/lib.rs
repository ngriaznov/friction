//! Scoring harness: the fixed regression fixtures and the total-ordered
//! score they must satisfy, built ahead of a real fix-producing engine.
//!
//! No component in this crate rewrites text. It exists to make the
//! regression fixtures assertable *now*, against a real tagger and a real
//! parse/segment pipeline, with three independent, priority-ordered
//! signals: tell-span count (the named objective), a clause-completeness
//! guardrail, and a capped secondary human-vs-machine classifier — see
//! [`score`] for how those three combine, and [`fixtures`] for the
//! fixtures themselves.
//!
//! # Modules
//!
//! - [`fixtures`]: the embedded fixture manifest and sample documents.
//! - [`clean`]: shared string-level cleaning/tokenization conventions.
//! - [`pivot`]: derivational-pivot (light-verb construction) detection.
//! - [`tellspan`]: the unified four-family tell-span scan.
//! - [`closure`]: the per-input closure-by-construction check.
//! - [`fragment`]: the clause-completeness guardrail, over the real
//!   parse/segment/tag pipeline.
//! - [`score`]: [`score::Score`], [`score::Scorer`], and
//!   [`score::shared_scorer`].
//! - [`gates`]: fixture-id-to-gate traceability.
//! - [`verify`]: reusable per-fixture assertion logic.
//! - [`error`][error]: [`error::HarnessError`].
//!
//! Byte-exact engine assertions (once deferred here as
//! `pending::pending_engine_assertions`, now that a real engine exists)
//! live in `tests/engine_fixtures.rs`, a dev-dependency on `friction-edit`.

pub mod clean;
pub mod closure;
pub mod error;
pub mod fixtures;
pub mod fragment;
pub mod gates;
pub mod pivot;
pub mod score;
pub mod tellspan;
pub mod verify;
