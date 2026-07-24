//! Plain, dependency-free round/fixpoint report shapes.
//!
//! This module used to also own the fixpoint driver itself (parse ->
//! metrics -> gate -> scan -> fix -> apply, run over a registered rule
//! set). That driver was coupled to the rule engine's own `Rule`/`Gate`/
//! `RuleContext` abstraction, which is retired; the real repair engine now
//! lives in `friction-edit` and runs its own bounded pass loop directly
//! (`friction_edit::document::edit_document`), not through this crate.
//!
//! [`RoundReport`]/[`FixpointReport`] survive as plain report shapes —
//! they reference only [`friction_core`] types, so any caller running its
//! own round-based patch pipeline can still populate and return them for a
//! stable, familiar report type. [`crate::coverage::touched_original_ranges`]
//! is built directly on top of [`RoundReport`].

use friction_core::{Finding, Patch, RuleId};

/// Up to this many rounds a caller running its own round-based pipeline is
/// expected to bound itself to. Kept here as a documented, shared
/// constant rather than a per-caller magic number.
pub const MAX_ROUNDS: usize = 16;

/// One round's outcome: which rules fired, what they found, and how many
/// of their proposed patches were applied versus dropped.
#[derive(Debug, Clone)]
pub struct RoundReport {
    /// This round's 1-indexed number.
    pub round: usize,
    /// Ids of every rule that produced at least one *accepted* patch this
    /// round, sorted and deduplicated.
    pub rules_fired: Vec<RuleId>,
    /// Every finding surfaced this round, in scan order — diagnostic
    /// information independent of which findings ended up fixed.
    pub findings: Vec<Finding>,
    /// How many patches were applied this round (after conflict
    /// resolution).
    pub patches_applied: usize,
    /// How many candidate patches were dropped this round, either for
    /// failing span validation against the round's source or for losing
    /// conflict resolution to a higher-priority overlapping patch.
    pub patches_dropped: usize,
    /// The accepted patches actually applied this round (after conflict
    /// resolution), sorted by `range.start` ascending. Every range indexes
    /// *this round's own* source text — never the original document's
    /// bytes past round 1.
    pub applied_patches: Vec<Patch>,
}

/// The full fixpoint driver's outcome: one [`RoundReport`] per round
/// actually run.
#[derive(Debug, Clone)]
pub struct FixpointReport {
    /// Per-round reports, in round order.
    pub rounds: Vec<RoundReport>,
}

impl FixpointReport {
    /// Total patches applied across every round.
    #[must_use]
    pub fn total_patches_applied(&self) -> usize {
        self.rounds.iter().map(|r| r.patches_applied).sum()
    }
}
