//! The symmetry rule family.
//!
//! Targets the coordination-pattern and closer-clause "too neatly
//! balanced" tics that recur unusually often in LLM-authored prose: a
//! dangling present-participle closer clause tacked onto a sentence
//! ([`ParticipialCloserRule`]), a flat `"X, Y, and Z"` triad
//! ([`TriadReductionRule`]), a `"not just X but (also) Y"` coordination
//! ([`NotJustButRule`]), and a document's final paragraph opening with a
//! ritual conclusion marker ([`RitualConclusionRule`]).
//!
//! # Tier discipline
//!
//! Only [`ParticipialCloserRule`] is unconditionally Fix tier: both its
//! strategies (delete the closer clause; promote it to its own sentence)
//! only ever delete or split existing text without reordering or dropping
//! the proposition the closer clause itself asserts. [`TriadReductionRule`]
//! and [`NotJustButRule`] are Suggest tier only — dropping the weakest item
//! of a triad, or reframing a `"not just X but Y"` coordination, both risk
//! dropping propositional content a purely syntactic scan cannot rule out,
//! so neither rule ever proposes a patch; `fix` always declines.
//! [`RitualConclusionRule`] is mixed *per finding*: deleting a final
//! paragraph is Fix tier only when a conservative, tagger-driven heuristic
//! confirms the paragraph introduces no content noun the rest of the
//! document did not already mention (so nothing propositional is actually
//! lost by removing it) — otherwise the same finding is Suggest tier and
//! carries no patch.
//!
//! # Reused detection logic
//!
//! `friction-metrics::symmetry`'s triad and participial-closer detectors,
//! and `friction-metrics::lexical`'s ritual-marker list and
//! `"not just...but"` pattern, are the canonical source these rules gate
//! on ([`friction_core::MetricVector::triad_rate`],
//! [`friction_core::MetricVector::participial_closer_rate`],
//! [`friction_core::MetricVector::ritual_marker_rate`],
//! [`friction_core::MetricVector::not_just_but_rate`]) — but the exact
//! matching logic those metrics use internally is private to that crate
//! (it needs to return only a rate, never a byte span), while a `Rule`'s
//! `scan` needs the exact span a `Finding`/`Patch` addresses. Each
//! submodule below therefore mirrors the relevant detector locally rather
//! than importing it, and pins that mirror down with its own consistency
//! test against `friction-metrics`' public, rate-returning function (see
//! each submodule's own docs for the specific test) — the same
//! "reimplement, then cross-check against the public metric" approach this
//! workspace already uses elsewhere for closed tables (e.g.
//! `families::contraction`'s reuse of
//! `friction_metrics::contraction_pairs`, where importing the *data* was
//! possible because it needed no span information).

mod not_just_but;
mod participial_closer;
mod ritual_conclusion;
mod triad_reduction;

pub use not_just_but::NotJustButRule;
pub use participial_closer::ParticipialCloserRule;
pub use ritual_conclusion::RitualConclusionRule;
pub use triad_reduction::TriadReductionRule;
