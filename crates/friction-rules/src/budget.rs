//! [`Budget`]: how many fixes a [`crate::Rule`] may apply in one round.

use friction_core::Envelope;

/// A bounded fix allowance for one rule, for one round.
///
/// A rule is never allowed to chase its target metric to zero — only back
/// into the genre's human envelope. `Budget` is the number that enforces
/// that: computed once, in [`Rule::gate`](crate::Rule::gate), from how far
/// the metric's current value sits outside the envelope, so that applying
/// up to [`Budget::remaining`] fixes is projected to land the metric back
/// at the edge of the band, never past it. See
/// [`Budget::from_envelope_excess`] for the exact formula.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Budget(usize);

impl Budget {
    /// No fixes allowed this round.
    pub const ZERO: Self = Self(0);

    /// Creates a budget for exactly `n` fixes.
    #[must_use]
    pub const fn new(n: usize) -> Self {
        Self(n)
    }

    /// How many fixes remain.
    #[must_use]
    pub const fn remaining(self) -> usize {
        self.0
    }

    /// `true` if no fixes remain.
    #[must_use]
    pub const fn is_exhausted(self) -> bool {
        self.0 == 0
    }

    /// Consumes one unit of budget, returning the updated budget — or
    /// `None` if none remained to consume.
    #[must_use]
    pub const fn take_one(self) -> Option<Self> {
        match self.0.checked_sub(1) {
            Some(n) => Some(Self(n)),
            None => None,
        }
    }

    /// Computes a budget from how far `current` sits outside `envelope`,
    /// for a metric where a single fix is projected to change the metric's
    /// value by a magnitude of `per_fix_effect` (always positive — the
    /// caller already knows *which direction* from which side of the band
    /// `current` fell on; this is only the size of one fix's effect).
    ///
    /// Three cases:
    /// - `current` is above `envelope.hi`: the excess is `current - hi`;
    ///   the budget is `floor(excess / per_fix_effect)`, enough fixes to
    ///   close exactly that excess.
    /// - `current` is below `envelope.lo`: symmetric, using the deficit
    ///   `lo - current`.
    /// - `current` already sits inside `[lo, hi]`: [`Budget::ZERO`] —
    ///   there is nothing to fix.
    ///
    /// Rounding is always down (`floor`), never up: an under-shoot leaves
    /// the metric still a little outside the band, which the *next*
    /// round's fresh `gate` call re-measures and, if still warranted, tops
    /// up — but an over-shoot risks pushing a document that only barely
    /// needed a nudge past the *opposite* edge of its own envelope, which
    /// is exactly the near-no-op behavior this formula exists to protect
    /// on human-typical text.
    ///
    /// `per_fix_effect` that is non-finite or not strictly positive (a
    /// caller bug — a fix must have *some* positive effect on the metric
    /// it targets, or gating it on that metric makes no sense), or a
    /// non-finite `current`, yields [`Budget::ZERO`] rather than a
    /// nonsensical or unbounded budget.
    #[must_use]
    pub fn from_envelope_excess(current: f64, envelope: Envelope, per_fix_effect: f64) -> Self {
        if !current.is_finite() || !per_fix_effect.is_finite() || per_fix_effect <= 0.0 {
            return Self::ZERO;
        }
        let outside = if current > envelope.hi {
            current - envelope.hi
        } else if current < envelope.lo {
            envelope.lo - current
        } else {
            return Self::ZERO;
        };
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        let n = (outside / per_fix_effect).floor() as usize;
        Self(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn envelope(lo: f64, hi: f64) -> Envelope {
        Envelope::new(lo, hi)
    }

    /// `remaining`/`is_exhausted` reflect the constructed count.
    #[test]
    fn new_reports_remaining_and_exhaustion() {
        assert_eq!(Budget::new(3).remaining(), 3);
        assert!(!Budget::new(3).is_exhausted());
        assert!(Budget::ZERO.is_exhausted());
        assert_eq!(Budget::ZERO.remaining(), 0);
    }

    /// `take_one` decrements until exhausted, then returns `None`.
    #[test]
    fn take_one_decrements_to_none() {
        let b = Budget::new(2);
        let b = b.take_one().expect("2 -> 1");
        assert_eq!(b.remaining(), 1);
        let b = b.take_one().expect("1 -> 0");
        assert_eq!(b.remaining(), 0);
        assert!(b.take_one().is_none());
    }

    /// Hand-computed: current 15.0 above hi 10.0, per-fix effect 2.0 ->
    /// excess 5.0 -> floor(5.0 / 2.0) = 2.
    #[test]
    fn from_envelope_excess_above_hi_hand_computed() {
        let budget = Budget::from_envelope_excess(15.0, envelope(0.0, 10.0), 2.0);
        assert_eq!(budget.remaining(), 2);
    }

    /// Hand-computed: current 15.0 above hi 10.0, per-fix effect 3.0 ->
    /// excess 5.0 -> floor(5.0 / 3.0) = 1 (conservative rounding down, not
    /// 2).
    #[test]
    fn from_envelope_excess_rounds_down_conservatively() {
        let budget = Budget::from_envelope_excess(15.0, envelope(0.0, 10.0), 3.0);
        assert_eq!(budget.remaining(), 1);
    }

    /// Hand-computed: current 1.0 below lo 4.0, per-fix effect 0.5 ->
    /// deficit 3.0 -> floor(3.0 / 0.5) = 6.
    #[test]
    fn from_envelope_excess_below_lo_hand_computed() {
        let budget = Budget::from_envelope_excess(1.0, envelope(4.0, 10.0), 0.5);
        assert_eq!(budget.remaining(), 6);
    }

    /// A value already inside the band needs no fixes, regardless of
    /// per-fix effect.
    #[test]
    fn from_envelope_excess_inside_band_is_zero() {
        let budget = Budget::from_envelope_excess(5.0, envelope(0.0, 10.0), 0.1);
        assert_eq!(budget.remaining(), 0);

        // Both boundary values count as "inside" (Envelope::contains is
        // inclusive on both ends).
        assert_eq!(
            Budget::from_envelope_excess(0.0, envelope(0.0, 10.0), 0.1).remaining(),
            0
        );
        assert_eq!(
            Budget::from_envelope_excess(10.0, envelope(0.0, 10.0), 0.1).remaining(),
            0
        );
    }

    /// A non-positive or non-finite `per_fix_effect` never produces a
    /// non-zero budget, however far outside the band `current` is.
    #[test]
    fn from_envelope_excess_rejects_bad_per_fix_effect() {
        assert_eq!(
            Budget::from_envelope_excess(100.0, envelope(0.0, 1.0), 0.0).remaining(),
            0
        );
        assert_eq!(
            Budget::from_envelope_excess(100.0, envelope(0.0, 1.0), -1.0).remaining(),
            0
        );
        assert_eq!(
            Budget::from_envelope_excess(100.0, envelope(0.0, 1.0), f64::NAN).remaining(),
            0
        );
        assert_eq!(
            Budget::from_envelope_excess(100.0, envelope(0.0, 1.0), f64::INFINITY).remaining(),
            0
        );
    }

    /// A non-finite `current` never produces a non-zero budget.
    #[test]
    fn from_envelope_excess_rejects_non_finite_current() {
        assert_eq!(
            Budget::from_envelope_excess(f64::NAN, envelope(0.0, 1.0), 1.0).remaining(),
            0
        );
        assert_eq!(
            Budget::from_envelope_excess(f64::INFINITY, envelope(0.0, 1.0), 1.0).remaining(),
            0
        );
    }
}
