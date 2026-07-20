//! Detection and fix rules: the [`Rule`] trait, its supporting engine
//! types, and nothing else.
//!
//! Family implementations (lexical, connective surgery, contraction
//! insertion, rhythm, symmetry, structural) build on this crate but live
//! elsewhere; this crate defines only the shape they share.
//!
//! # What a rule is allowed to do
//!
//! Every rule scans a document for [`friction_core::Finding`]s and, for
//! the ones it chooses to fix, proposes a [`friction_core::Patch`] against
//! the *original* bytes of the round it ran in (see [`RuleContext`]).
//! `friction-apply` collects those patches across every active rule in
//! every round and is solely responsible for turning them into applied
//! text — nothing in this crate ever mutates a document itself.
//!
//! # Density gating
//!
//! [`Rule::gate`] is the gate every rule runs through before it is allowed
//! to do anything at all: `Off` when its target metric(s) already sit
//! inside the genre's human envelope (the common case, on human-typical
//! text — the rule costs nothing), `Detect` when outside the envelope but
//! this round should only surface diagnostics, or `Fix` with a [`Budget`]
//! sized to close the gap between the metric's current value and the edge
//! of the envelope. A rule never tries to drive its metric to zero, only
//! back into the band it came from.
//!
//! # Determinism and idempotence
//!
//! A rule must be a pure function of its inputs: `gate` of `(metrics,
//! envelope)`, `scan` of `ctx`, `fix` of `(finding, ctx, strategy_rng)`.
//! Any choice among multiple meaning-preserving fix strategies for the
//! same finding must go through [`StrategyRng`], seeded from the sentence
//! being fixed and the rule's own id — never a fixed constant (which would
//! itself become a detectable, uniform tic) and never ambient randomness
//! (which would break reproducibility: the same input bytes and the same
//! pack must always produce the same output bytes). A rule must also be
//! idempotent by construction: run again on its own previous output, it
//! must find nothing left to fix — a filler-phrase deletion rule's pattern
//! must not match its own already-deleted result, and a substitution
//! table's replacement values must never themselves appear as a lookup key
//! elsewhere in the same table.

mod budget;
mod context;
pub mod families;
mod rule;
mod strategy;

pub use budget::Budget;
pub use context::{GenreEnvelope, MapEnvelope, RuleContext};
pub use families::connective::ConnectiveSurgery;
pub use families::contraction::ContractionRule;
pub use families::lexical::{FillerPhraseRule, SubstitutionRule};
pub use families::rhythm::{SentenceFuseRule, SentenceSplitRule};
pub use families::structural::{BoldLabelStripRule, HeaderMergeRule, UnbulletRule};
pub use families::symmetry::{
    NotJustButRule, ParticipialCloserRule, RitualConclusionRule, TriadReductionRule,
};
pub use rule::{Gate, Rule, RuleFamily};
pub use strategy::StrategyRng;
