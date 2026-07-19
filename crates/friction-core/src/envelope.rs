//! [`Envelope`]: a per-metric human percentile band.

use crate::error::CoreError;

/// A percentile band `[lo, hi]` for a single metric, estimated from the
/// human corpus per (genre, metric) by `corpus-tool envelope` and shipped
/// as a versioned pack (`friction-packs`).
///
/// With the `serde` feature enabled, `Envelope` derives
/// `Serialize`/`Deserialize` so `friction-packs` can (de)serialize it
/// to/from the versioned envelope TOML.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Envelope {
    /// Lower bound (e.g. p10) of the band.
    pub lo: f64,
    /// Upper bound (e.g. p90) of the band.
    pub hi: f64,
}

impl Envelope {
    /// Creates a new envelope band.
    ///
    /// Does not itself check `lo <= hi` or finiteness — call
    /// [`Envelope::validate`] to check those invariants explicitly, e.g.
    /// after deserializing a pack from disk.
    #[must_use]
    pub const fn new(lo: f64, hi: f64) -> Self {
        Self { lo, hi }
    }

    /// `true` if `value` falls within `[lo, hi]`, inclusive.
    #[must_use]
    pub const fn contains(&self, value: f64) -> bool {
        value >= self.lo && value <= self.hi
    }

    /// Validates that `lo <= hi` and both bounds are finite.
    ///
    /// # Errors
    /// Returns [`CoreError::InvalidEnvelope`] if either bound is
    /// non-finite (`NaN` or infinite) or `lo > hi`.
    pub const fn validate(&self) -> Result<(), CoreError> {
        if !self.lo.is_finite() || !self.hi.is_finite() || self.lo > self.hi {
            return Err(CoreError::InvalidEnvelope {
                lo: self.lo,
                hi: self.hi,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `contains` treats the band as inclusive on both ends.
    #[test]
    fn contains_is_inclusive_on_both_bounds() {
        let envelope = Envelope::new(10.0, 20.0);
        assert!(envelope.contains(10.0));
        assert!(envelope.contains(15.0));
        assert!(envelope.contains(20.0));
        assert!(!envelope.contains(9.999));
        assert!(!envelope.contains(20.001));
    }

    /// `validate` accepts a well-formed, finite band.
    #[test]
    fn validate_accepts_well_formed_band() {
        assert!(Envelope::new(1.0, 2.0).validate().is_ok());
        assert!(Envelope::new(5.0, 5.0).validate().is_ok());
    }

    /// `validate` rejects an inverted band (`lo > hi`).
    #[test]
    fn validate_rejects_inverted_band() {
        let err = Envelope::new(5.0, 1.0).validate().unwrap_err();
        let CoreError::InvalidEnvelope { lo, hi } = err else {
            panic!("expected InvalidEnvelope, got {err:?}");
        };
        assert_eq!((lo, hi), (5.0, 1.0));
    }

    /// `validate` rejects non-finite bounds (`NaN`, `+inf`, `-inf`).
    #[test]
    fn validate_rejects_non_finite_bounds() {
        assert!(Envelope::new(f64::NAN, 1.0).validate().is_err());
        assert!(Envelope::new(0.0, f64::INFINITY).validate().is_err());
        assert!(Envelope::new(f64::NEG_INFINITY, 0.0).validate().is_err());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serde_round_trips_through_json() {
        let envelope = Envelope::new(4.5, 9.75);
        let json = serde_json::to_string(&envelope).unwrap();
        let round_tripped: Envelope = serde_json::from_str(&json).unwrap();
        assert_eq!(envelope, round_tripped);
    }
}
