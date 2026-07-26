//! [`RegisterPack`]: a parsed `register-v1` pack — per-feature human rate
//! bands, measured per 1000 prose words, that `friction-edit`'s register
//! module treats as a termination condition rather than a target to
//! optimize toward.
//!
//! Structurally this is [`crate::EnvelopePack`]'s shape narrowed by one
//! axis (`[<feature>]` instead of `[<genre>.<metric>]`, since this pack
//! covers exactly one genre — `docs` — by construction) and widened by
//! one field (`median`, carried for provenance even though the consumer
//! never optimizes toward it). It is not built on `EnvelopePack` directly:
//! that type's `include`/`direction` fields and multi-genre keying exist
//! for `friction check`'s combined-score envelope, which this pack has no
//! use for, and reusing it would mean carrying fields this reader always
//! ignores.

use std::collections::BTreeMap;

use serde::Deserialize;

use crate::PackError;

/// One feature's human rate band, in occurrences per 1000 prose words.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RegisterBand {
    pub low: f64,
    pub high: f64,
    /// The per-document median rate. Provenance only: a register
    /// consumer's termination condition is band membership
    /// (`low..=high`), never distance from `median` — see
    /// `friction-edit`'s register module docs for why a document that
    /// lands exactly on the centroid is itself a tell.
    pub median: f64,
}

impl RegisterBand {
    /// `true` if `rate` falls inside `[low, high]`, inclusive on both ends.
    #[must_use]
    pub fn contains(self, rate: f64) -> bool {
        rate >= self.low && rate <= self.high
    }
}

/// A parsed `register-v1` pack: one [`RegisterBand`] per named feature.
#[derive(Debug, Clone, Default)]
pub struct RegisterPack {
    features: BTreeMap<Box<str>, RegisterBand>,
}

impl RegisterPack {
    /// Parses a pack from its TOML text: a `[pack]` metadata table
    /// (provenance only, not read back into any typed field — see the
    /// module docs and the pack file's own header comment) plus one
    /// `[features.<name>]` table per band.
    ///
    /// # Errors
    /// Returns [`PackError::Toml`] if `toml` is not valid TOML in the
    /// expected shape, or [`PackError::InvalidRegisterBand`] if a parsed
    /// band does not satisfy `low <= median <= high` with every bound
    /// finite.
    ///
    /// Non-strict on purpose: `[features.em_dash]`'s measured population
    /// has a zero 10th percentile, zero median, *and* zero 90th
    /// percentile (56 of 58 sampled documents contain no em dash at
    /// all), a degenerate `low == median == high == 0.0` band that a
    /// strict `<` would reject outright despite being the correct
    /// reading of the data.
    pub fn parse(toml: &str) -> Result<Self, PackError> {
        let raw: RawPack = toml::from_str(toml).map_err(PackError::from)?;
        // Collecting into `Result<BTreeMap<_, _>, _>` short-circuits on the
        // first malformed band, so validation and construction stay one
        // expression instead of a loop that accumulates into a `mut` map and
        // returns early out of the middle of it.
        let features = raw
            .features
            .into_iter()
            .map(|(name, band)| {
                let well_formed = band.low.is_finite()
                    && band.median.is_finite()
                    && band.high.is_finite()
                    && band.low <= band.median
                    && band.median <= band.high;
                if !well_formed {
                    return Err(PackError::InvalidRegisterBand {
                        feature: name,
                        low: band.low,
                        median: band.median,
                        high: band.high,
                    });
                }
                Ok((
                    name.into_boxed_str(),
                    RegisterBand {
                        low: band.low,
                        high: band.high,
                        median: band.median,
                    },
                ))
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        Ok(Self { features })
    }

    /// The band for `feature`, or `None` if this pack does not define one.
    #[must_use]
    pub fn band(&self, feature: &str) -> Option<RegisterBand> {
        self.features.get(feature).copied()
    }

    /// Every feature name this pack defines a band for, in sorted order.
    #[must_use]
    pub fn features(&self) -> Vec<&str> {
        self.features.keys().map(AsRef::as_ref).collect()
    }
}

#[derive(Debug, Deserialize)]
struct RawBand {
    low: f64,
    high: f64,
    median: f64,
}

/// The TOML shape of the pack as a whole. Deliberately not
/// `#[serde(deny_unknown_fields)]` on the outer struct: `[pack]` is free-
/// form provenance metadata this reader never round-trips (matching
/// `dms.rs`/`inventory.rs`'s own precedent), so an unrecognized `[pack]`
/// key must not fail the parse.
#[derive(Debug, Deserialize)]
struct RawPack {
    #[serde(default)]
    features: BTreeMap<String, RawBand>,
}

#[cfg(test)]
// Every comparison below is against an exact hand-written literal (a
// value parsed straight out of `SAMPLE`'s own TOML text), matching
// `envelope.rs`'s own precedent for this exact situation — exact
// equality is the correct check here, not an epsilon comparison.
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"
        [pack]
        version = "test"
        measured_from = "class:human, split:train, genre:docs"

        [features.nominalization]
        low = 15.0
        high = 50.0
        median = 28.0

        [features.agentless_passive]
        low = 5.0
        high = 21.0
        median = 11.0
    "#;

    #[test]
    fn parse_looks_up_known_bands() {
        let pack = RegisterPack::parse(SAMPLE).expect("sample pack parses");
        let nom = pack.band("nominalization").expect("nominalization band");
        assert_eq!(nom.low, 15.0);
        assert_eq!(nom.high, 50.0);
        assert_eq!(nom.median, 28.0);
        assert!(pack.band("not_a_feature").is_none());
    }

    #[test]
    fn parse_skips_the_pack_metadata_table() {
        let pack = RegisterPack::parse(SAMPLE).expect("sample pack parses");
        assert_eq!(pack.features(), vec!["agentless_passive", "nominalization"]);
    }

    #[test]
    fn parse_rejects_malformed_toml() {
        assert!(RegisterPack::parse("not [ valid toml").is_err());
    }

    #[test]
    fn parse_rejects_band_where_median_is_outside_low_and_high() {
        let bad = r"
            [features.nominalization]
            low = 15.0
            high = 50.0
            median = 60.0
        ";
        let err = RegisterPack::parse(bad).expect_err("out-of-range median must be rejected");
        assert!(matches!(err, PackError::InvalidRegisterBand { .. }));
    }

    #[test]
    fn parse_rejects_inverted_band() {
        let bad = r"
            [features.nominalization]
            low = 50.0
            high = 15.0
            median = 28.0
        ";
        let err = RegisterPack::parse(bad).expect_err("inverted band must be rejected");
        assert!(matches!(err, PackError::InvalidRegisterBand { .. }));
    }

    /// A degenerate `low == median == high` band -- exactly
    /// `[features.em_dash]`'s measured shape -- parses and is accepted,
    /// even though it would fail a strict `low < median < high` check.
    /// See [`RegisterPack::parse`]'s own docs for why.
    #[test]
    fn parse_accepts_a_degenerate_zero_width_band() {
        let toml = r"
            [features.em_dash]
            low = 0.0
            high = 0.0
            median = 0.0
        ";
        let pack = RegisterPack::parse(toml).expect("zero-width band must parse");
        let band = pack.band("em_dash").expect("em_dash band");
        assert!(band.contains(0.0));
        assert!(!band.contains(0.0001));
    }

    #[test]
    fn contains_is_inclusive_on_both_ends() {
        let band = RegisterBand {
            low: 10.0,
            high: 20.0,
            median: 15.0,
        };
        assert!(band.contains(10.0));
        assert!(band.contains(20.0));
        assert!(band.contains(15.0));
        assert!(!band.contains(9.999));
        assert!(!band.contains(20.001));
    }

    #[test]
    fn embedded_register_v1_parses_and_covers_all_three_features() {
        let pack = &crate::registry::REGISTER.pack;
        for feature in ["nominalization", "agentless_passive", "em_dash"] {
            assert!(
                pack.band(feature).is_some(),
                "expected a band for feature {feature:?}"
            );
        }
    }
}
