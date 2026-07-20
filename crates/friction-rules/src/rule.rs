//! The [`Rule`] trait and its supporting [`RuleFamily`]/[`Gate`] types.

use friction_core::{Finding, MetricVector, Patch, RuleId};

use crate::budget::Budget;
use crate::context::{GenreEnvelope, RuleContext};
use crate::strategy::StrategyRng;

/// Which family a [`Rule`] belongs to.
///
/// This also fixes the conflict-resolution priority order
/// `friction-apply` uses when two patches from different families overlap
/// in the same round: outer, larger-span transforms are preferred over
/// smaller ones nested inside them, because an outer transform invalidates
/// fewer of the inner ones than the reverse would. Spelled out, the fixed
/// order (highest priority first) is:
///
/// `Structural, Symmetry, Connective, Lexical, Rhythm, Contraction`
///
/// [`RuleFamily::priority`] exposes that order as a small integer (lower
/// sorts first / wins a tie).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum RuleFamily {
    /// Discourse-filler phrase deletion and inflection-aware lexical
    /// substitution ("leverage" -> "use").
    Lexical,
    /// Sentence-initial connective surgery (delete, recapitalize, or swap
    /// to a shorter connective).
    Connective,
    /// Contraction insertion ("do not" -> "don't").
    Contraction,
    /// Sentence-length and paragraph-shape rhythm (splitting, fusing).
    Rhythm,
    /// Coordination-pattern and closer-clause symmetry (triads,
    /// participial closers, "not just X but also Y").
    Symmetry,
    /// Document/list structure (un-bulleting, bold lead-in stripping,
    /// header/section merging).
    Structural,
}

impl RuleFamily {
    /// This family's conflict-resolution priority: lower sorts first
    /// (wins a tie against a higher value). See the type's own docs for
    /// the full spelled-out order and its rationale.
    #[must_use]
    pub const fn priority(self) -> u8 {
        match self {
            Self::Structural => 0,
            Self::Symmetry => 1,
            Self::Connective => 2,
            Self::Lexical => 3,
            Self::Rhythm => 4,
            Self::Contraction => 5,
        }
    }
}

/// The density-gating decision a [`Rule`] makes for one round, from the
/// round's [`MetricVector`] and the genre's human envelope bands.
///
/// A rule that finds its target metric(s) already inside the envelope
/// gates `Off` — it does not scan at all, so a human-typical document pays
/// no cost and receives no patches from it. A rule outside the envelope
/// gates `Detect` (surface findings as diagnostics only — e.g. a rule
/// whose fix would normally require a model-backed dependency parser that
/// is not available this run) or `Fix` with a [`Budget`] sized to close
/// the gap between the metric's current value and the edge of the
/// envelope, never to drive it to zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gate {
    /// Inside the envelope (or a required precondition is unmet): do not
    /// scan.
    Off,
    /// Outside the envelope, but findings are diagnostic-only this round —
    /// never auto-applied, regardless of an individual finding's own
    /// [`friction_core::Tier`].
    Detect,
    /// Outside the envelope: scan, and fix up to `budget` findings.
    Fix {
        /// How many findings this rule may fix this round.
        budget: Budget,
    },
}

/// A detection-and-fix rule.
///
/// Implementations live in the family crates (lexical, connective surgery,
/// contraction insertion, rhythm, symmetry, structural) that build on this
/// trait; `friction-rules` only defines the shape every family implements
/// and the engine types ([`RuleContext`], [`Budget`], [`StrategyRng`])
/// they receive from `friction-apply`'s driver.
///
/// Object-safe by design (`friction-apply` holds its active rule set as
/// `&[&dyn Rule]`), so every method takes owned or borrowed data only —
/// none are generic.
pub trait Rule {
    /// This rule's stable identifier (e.g. `"lexical.leverage"`).
    fn id(&self) -> RuleId;

    /// Which [`RuleFamily`] this rule belongs to.
    fn family(&self) -> RuleFamily;

    /// The density-gating decision for this round: given the round's
    /// document-level [`MetricVector`] and the genre's envelope bands,
    /// decide whether — and how much — to fix.
    ///
    /// Must be a pure function of its two arguments: the same metrics and
    /// envelope must always gate the same way.
    fn gate(&self, metrics: &MetricVector, envelope: &dyn GenreEnvelope) -> Gate;

    /// Scans `ctx`'s document for this rule's pattern, returning every
    /// match as a [`Finding`], in source order.
    ///
    /// Called whenever [`Rule::gate`] returned anything other than
    /// [`Gate::Off`]. Must be a pure function of `ctx`.
    fn scan(&self, ctx: &RuleContext<'_>) -> Vec<Finding>;

    /// Proposes a fix for one `Fix`-tier finding from this rule's own
    /// `scan`, or declines (`None`) if this particular finding has no safe
    /// fix (e.g. an exception case the rule recognizes only once it looks
    /// closer).
    ///
    /// `strategy_rng` is seeded by the caller (via
    /// [`StrategyRng::seeded`]) from the finding's sentence and this
    /// rule's id, for a rule with more than one meaning-preserving fix
    /// strategy to choose between deterministically. A rule with only one
    /// strategy is free to ignore it.
    ///
    /// The returned [`Patch`]'s `range` must be a byte range into `ctx`'s
    /// document's *original* source (not a re-derived offset), and its
    /// `tier` must be [`friction_core::Tier::Fix`] — meaning-preserving by
    /// construction — never [`friction_core::Tier::Suggest`]; a rule that
    /// wants to surface a `Suggest`-tier alternative for this finding
    /// should do so only via `scan`'s own findings, not through `fix`.
    fn fix(
        &self,
        finding: &Finding,
        ctx: &RuleContext<'_>,
        strategy_rng: &mut StrategyRng,
    ) -> Option<Patch>;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The fixed conflict-resolution priority order, spelled out in the
    /// type's docs, matches `priority`'s numeric values exactly:
    /// `Structural, Symmetry, Connective, Lexical, Rhythm, Contraction`.
    #[test]
    fn priority_matches_documented_order() {
        let ordered = [
            RuleFamily::Structural,
            RuleFamily::Symmetry,
            RuleFamily::Connective,
            RuleFamily::Lexical,
            RuleFamily::Rhythm,
            RuleFamily::Contraction,
        ];
        for pair in ordered.windows(2) {
            assert!(
                pair[0].priority() < pair[1].priority(),
                "{:?} (priority {}) must sort before {:?} (priority {})",
                pair[0],
                pair[0].priority(),
                pair[1],
                pair[1].priority()
            );
        }
    }
}
