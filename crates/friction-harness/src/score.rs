//! [`Score`]: the harness's total-ordered, three-tier friction score, and
//! [`Scorer`], which computes one from raw text.

use std::sync::OnceLock;

use friction_core::MetricVector;
use friction_nlp::{PerceptronTagger, SrxSegmenter, Tagger};

use crate::clean::{is_punctuation_token, tokenize};
use crate::error::HarnessError;
use crate::{fragment, tellspan};

/// A total-ordered, three-tier friction score. Lower always means better.
///
/// Fields compare lexicographically in this declared order — the priority
/// order — via `#[derive(PartialOrd, Ord)]`:
///
/// 1. `tell_span_density_ppm`: seed tell-span hits per 1000 word tokens,
///    scaled to parts-per-million and rounded to a `u32` so the whole type
///    can derive `Ord` without float-comparison pitfalls: the named
///    objective (tell-span count reduction). For every accept fixture this
///    alone already decides the comparison, so tagger noise in the lower
///    tiers can never flip an accept-fixture result.
/// 2. `fragment_rate_ppm`: the clause-completeness guardrail from
///    [`crate::fragment`], scaled to parts-per-million: stands in for
///    "subject to all gates" until a real gate-enforcing engine exists.
///    Only ever consulted on a tier-1 tie.
/// 3. `secondary_ppm`: the reused human-vs-LLM combined-score classifier
///    ([`friction_packs::EnvelopePack::combined_score`]), averaged over
///    every genre `ENVELOPE_V2` defines and scaled to parts-per-million —
///    the "capped secondary signal": capped in *magnitude* (already
///    bounded to `[0, 1]` by [`friction_packs::exceedance`]'s own cap) and
///    capped in *influence* (lowest tier: it can only break a tie both
///    tier 1 and tier 2 leave standing, never overturn either). Known
///    limitation, verified empirically and left as-is by this crate's
///    scope: this signal alone gets the chat-speak-vs-human-prose and
///    dumb-human-collapse-vs-conservative-rewrite orderings backwards on
///    several individual genres (rewarding band conformance a deliberately
///    gamed text was optimized for). It is never the sole decider for
///    either ordering in this harness. The shared `_ppm` postfix is
///    intentional and load-bearing here: it names the common unit
///    (parts-per-million) every tier is scaled to so the type can derive
///    `Ord` without float-comparison pitfalls, not an accidental naming
///    collision `struct_field_names` should flag.
#[allow(clippy::struct_field_names)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Score {
    tell_span_density_ppm: u32,
    fragment_rate_ppm: u32,
    secondary_ppm: u32,
}

/// The result of comparing one [`Score`] against a baseline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// Strictly lower (better) than the baseline.
    Better,
    /// Equal to the baseline.
    Tied,
    /// Strictly higher (worse) than the baseline.
    Worse,
}

impl Score {
    /// Compares `self` against `baseline`: lower is better, so
    /// `self < baseline` yields [`Verdict::Better`].
    #[must_use]
    pub fn compare_to(&self, baseline: &Self) -> Verdict {
        match self.cmp(baseline) {
            std::cmp::Ordering::Less => Verdict::Better,
            std::cmp::Ordering::Equal => Verdict::Tied,
            std::cmp::Ordering::Greater => Verdict::Worse,
        }
    }
}

/// Computes [`Score`]s from raw text: owns the tagger model and sentence
/// segmenter every tier needs.
pub struct Scorer {
    segmenter: SrxSegmenter,
    tagger: PerceptronTagger,
}

impl Scorer {
    /// Builds a new scorer, loading the embedded tagger model.
    ///
    /// # Errors
    /// Returns [`HarnessError::TaggerLoad`] if the embedded tagger model
    /// fails to load.
    pub fn new() -> Result<Self, HarnessError> {
        let tagger = PerceptronTagger::new().map_err(HarnessError::TaggerLoad)?;
        Ok(Self {
            segmenter: SrxSegmenter::new(),
            tagger,
        })
    }

    /// This scorer's tagger.
    #[must_use]
    pub fn tagger(&self) -> &dyn Tagger {
        &self.tagger
    }

    /// This scorer's sentence segmenter.
    #[must_use]
    pub const fn segmenter(&self) -> &SrxSegmenter {
        &self.segmenter
    }

    /// Scores `text`.
    ///
    /// # Errors
    /// Propagates parse/segment errors from the underlying pipeline.
    pub fn score(&self, text: &str) -> Result<Score, HarnessError> {
        let tell_span_count = tellspan::count_tell_spans(text, &self.tagger);
        let word_token_count = tokenize(text)
            .into_iter()
            .filter(|t| !is_punctuation_token(t))
            .count();
        #[allow(clippy::cast_precision_loss)]
        let tell_span_density_ppm = ppm_ratio(tell_span_count as f64, word_token_count as f64);

        let fragment_rate = fragment::fragment_rate(text, &self.segmenter, &self.tagger)?;
        let fragment_rate_ppm = ppm_unit(fragment_rate);

        let document = friction_parse::parse(text)?;
        let metrics = friction_metrics::compute(&document, &self.segmenter, &self.tagger);
        let secondary_ppm = ppm_unit(secondary_signal(&metrics));

        Ok(Score {
            tell_span_density_ppm,
            fragment_rate_ppm,
            secondary_ppm,
        })
    }
}

/// Process-wide shared [`Scorer`] (the tagger model load is paid once per
/// test binary, mirroring [`friction_packs::ENVELOPE_V2`]'s own
/// `LazyLock` pattern).
#[must_use]
pub fn shared_scorer() -> &'static Scorer {
    static SCORER: OnceLock<Scorer> = OnceLock::new();
    SCORER.get_or_init(|| Scorer::new().expect("embedded tagger model must load"))
}

/// Mean of `ENVELOPE_V2.combined_score(genre, metrics)` over
/// `ENVELOPE_V2.genres()`, skipping `None`s, `0.0` if every genre is
/// `None`.
fn secondary_signal(metrics: &MetricVector) -> f64 {
    let scores: Vec<f64> = friction_packs::ENVELOPE_V2
        .genres()
        .into_iter()
        .filter_map(|genre| friction_packs::ENVELOPE_V2.combined_score(genre, metrics))
        .collect();
    if scores.is_empty() {
        0.0
    } else {
        #[allow(clippy::cast_precision_loss)]
        let count = scores.len() as f64;
        scores.iter().sum::<f64>() / count
    }
}

/// Scales `count / total` to parts-per-million, rounded to a `u32`. `0` if
/// `total` is not positive (no basis for a rate).
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn ppm_ratio(count: f64, total: f64) -> u32 {
    if total <= 0.0 {
        return 0;
    }
    ((count / total) * 1_000_000.0)
        .round()
        .clamp(0.0, f64::from(u32::MAX)) as u32
}

/// Scales a value already in `[0, 1]` to parts-per-million, rounded to a
/// `u32`, clamping out-of-range input rather than panicking.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn ppm_unit(value: f64) -> u32 {
    (value.clamp(0.0, 1.0) * 1_000_000.0).round() as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compare_to_orders_lower_score_as_better() {
        let low = Score {
            tell_span_density_ppm: 10,
            fragment_rate_ppm: 0,
            secondary_ppm: 0,
        };
        let high = Score {
            tell_span_density_ppm: 20,
            fragment_rate_ppm: 0,
            secondary_ppm: 0,
        };
        assert_eq!(low.compare_to(&high), Verdict::Better);
        assert_eq!(high.compare_to(&low), Verdict::Worse);
        assert_eq!(low.compare_to(&low), Verdict::Tied);
    }

    #[test]
    fn score_priority_is_lexicographic() {
        // A worse tier-1 value always outranks a better tier-2/3 value.
        let worse_tier1 = Score {
            tell_span_density_ppm: 5,
            fragment_rate_ppm: 0,
            secondary_ppm: 0,
        };
        let better_tier1 = Score {
            tell_span_density_ppm: 4,
            fragment_rate_ppm: 1_000_000,
            secondary_ppm: 1_000_000,
        };
        assert_eq!(better_tier1.compare_to(&worse_tier1), Verdict::Better);
    }

    #[test]
    fn ppm_ratio_handles_zero_total() {
        assert_eq!(ppm_ratio(3.0, 0.0), 0);
    }

    #[test]
    fn ppm_unit_scales_and_rounds() {
        assert_eq!(ppm_unit(0.0), 0);
        assert_eq!(ppm_unit(1.0), 1_000_000);
        assert_eq!(ppm_unit(0.5), 500_000);
    }

    #[test]
    fn scoring_is_deterministic() {
        let a = Scorer::new().expect("tagger loads");
        let b = Scorer::new().expect("tagger loads");
        let text = "This guide will walk you through configuring the backup agent.";
        assert_eq!(
            a.score(text).expect("scoring succeeds"),
            b.score(text).expect("scoring succeeds")
        );
    }
}
