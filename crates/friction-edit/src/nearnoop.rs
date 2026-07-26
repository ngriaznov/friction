//! [`PivotBudget`]: the derivational-pivot overfire brake.
//!
//! Humans use light-verb constructions at a measurably lower rate than
//! machine-register text, so a document whose pivot rate would exceed
//! the calibrated human ceiling must have its excess pivots held, even
//! when every other gate passed. One budget is built per document and
//! shared, left to right, across every sentence's pivot stage.

use friction_packs::NearNoOpCalibration;

/// A per-document pivot-application budget.
#[derive(Debug, Clone, Copy)]
pub struct PivotBudget {
    remaining: u32,
}

impl PivotBudget {
    /// Builds a budget for a document of `word_count` prose words, scaled
    /// from `calibration`'s per-1000-word threshold.
    #[must_use]
    pub fn for_document(word_count: usize, calibration: &NearNoOpCalibration) -> Self {
        #[allow(clippy::cast_precision_loss)]
        let words = word_count as f64;
        let max = (calibration.threshold_per_1000_words * words / 1000.0).ceil();
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let remaining = if max.is_finite() && max > 0.0 {
            max as u32
        } else {
            0
        };
        Self { remaining }
    }

    /// An effectively unlimited budget — used when no calibration has been
    /// recorded yet (stage 1 of the attestation pack, before
    /// `corpus-tool attest --calibrate-near-noop` has run).
    #[must_use]
    pub const fn unlimited() -> Self {
        Self {
            remaining: u32::MAX,
        }
    }

    /// Takes one unit of budget if any remains, returning whether it was
    /// available.
    pub const fn try_take(&mut self) -> bool {
        if self.remaining > 0 {
            self.remaining -= 1;
            true
        } else {
            false
        }
    }

    /// The budget remaining, for reporting/tests.
    #[must_use]
    pub const fn remaining(&self) -> u32 {
        self.remaining
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn calibration(threshold: f64) -> NearNoOpCalibration {
        NearNoOpCalibration {
            threshold_per_1000_words: threshold,
            sample_doc_count: 264,
            dev_check_max_per_1000_words: None,
            method: "test".into(),
        }
    }

    #[test]
    fn budget_scales_with_word_count() {
        let budget = PivotBudget::for_document(2000, &calibration(1.5));
        // 1.5 * 2000 / 1000 = 3.0 -> ceil = 3
        assert_eq!(budget.remaining(), 3);
    }

    #[test]
    fn budget_exhausts_after_its_allotment() {
        let mut budget = PivotBudget::for_document(1000, &calibration(2.0));
        assert_eq!(budget.remaining(), 2);
        assert!(budget.try_take());
        assert!(budget.try_take());
        assert!(!budget.try_take());
        assert_eq!(budget.remaining(), 0);
    }

    #[test]
    fn zero_word_document_has_zero_budget() {
        let budget = PivotBudget::for_document(0, &calibration(2.0));
        assert_eq!(budget.remaining(), 0);
    }

    #[test]
    fn unlimited_budget_never_exhausts_across_many_takes() {
        let mut budget = PivotBudget::unlimited();
        for _ in 0..10_000 {
            assert!(budget.try_take());
        }
    }
}
