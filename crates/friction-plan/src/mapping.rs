//! The fixed family -> driving-metric mapping [`crate::Plan`] uses to turn
//! envelope deltas into per-family budget estimates.
//!
//! Every entry here is read straight off the matching rule's own
//! `gate` in `friction-rules` (metric name, which side of the band is
//! actionable, and — where the real rule scales its budget by envelope
//! excess at all — the exact `PER_FIX_EFFECT` it uses), so this table is a
//! *transcription*, not a reinterpretation. Where a family's real rule
//! never scales a budget by excess at all (a `Detect`-only rule, or a
//! `Fix`-tier rule that instead uses a flat safety-cap budget — see each
//! table's own doc comment below for which), this crate still needs *some*
//! per-fix effect size to produce an advisory number, since a `Plan`'s
//! whole purpose is to size every family the same way. It uses this
//! crate's common default of `1.0`, which is also the *exact* value most
//! of the real excess-scaled rules already use
//! (`structural.bold_label_strip`, `structural.unbullet`,
//! `connective.surgery`, `lexical.filler_phrase`, `lexical.substitution`)
//! — so `1.0` is the representative choice for this codebase's rules in
//! general, not an arbitrary one. `rhythm.split` is the one exception
//! (`0.05`, see below), and this table uses that real value rather than
//! the default, since `rhythm` is the one family where a plausible, much
//! more accurate number is directly on hand.

use friction_rules::RuleFamily;

/// Which side of an envelope band a metric's excess is measured from —
/// mirrors the single direction check every real gating rule for that
/// metric already makes before calling
/// [`friction_rules::Budget::from_envelope_excess`] (or, for a
/// `Detect`-only/flat-budget rule, the equivalent "is this even the
/// direction I could act on" check its own `gate` still makes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// Only `current > band.hi` is actionable (the metric reads too
    /// high).
    AboveHi,
    /// Only `current < band.lo` is actionable (the metric reads too low).
    BelowLo,
}

/// One metric that drives a family's advisory budget: its name (matching
/// [`friction_core::MetricVector::FIELD_NAMES`]), which side of its
/// envelope band is actionable, and the per-fix effect size used to
/// convert its envelope excess into a fix count via
/// [`friction_rules::Budget::from_envelope_excess`].
#[derive(Debug, Clone, Copy)]
pub struct DrivingMetric {
    pub(crate) name: &'static str,
    pub(crate) direction: Direction,
    pub(crate) per_fix_effect: f64,
}

const fn metric(name: &'static str, direction: Direction, per_fix_effect: f64) -> DrivingMetric {
    DrivingMetric {
        name,
        direction,
        per_fix_effect,
    }
}

/// This crate's common default per-fix effect size (see the module docs'
/// rationale): the exact value `structural.bold_label_strip`,
/// `structural.unbullet`, `connective.surgery`, `lexical.filler_phrase`,
/// and `lexical.substitution` already use for real, and the
/// representative stand-in this table uses for every metric whose real
/// rule has no `PER_FIX_EFFECT` of its own to transcribe.
const DEFAULT_PER_FIX_EFFECT: f64 = 1.0;

/// Structural: `structural.bold_label_strip` (`bold_span_density`, above
/// `hi`, `PER_FIX_EFFECT = 1.0`) and `structural.unbullet`
/// (`list_item_density`, above `hi`, `PER_FIX_EFFECT = 1.0`) both scale a
/// real `Fix` budget by excess exactly this way.
/// `structural.header_merge` (`heading_density`, above `hi`) is
/// `Detect`-only in the real rule (it never produces a patch at all — see
/// that rule's own module docs), so it has no `PER_FIX_EFFECT` to
/// transcribe; this table uses the crate-wide default for it.
const STRUCTURAL: &[DrivingMetric] = &[
    metric("bold_span_density", Direction::AboveHi, 1.0),
    metric("list_item_density", Direction::AboveHi, 1.0),
    metric(
        "heading_density",
        Direction::AboveHi,
        DEFAULT_PER_FIX_EFFECT,
    ),
];

/// Symmetry: none of this family's four metrics has a real
/// excess-scaled `Fix` budget to transcribe. `symmetry.not_just_but`
/// (`not_just_but_rate`) and `symmetry.triad_reduction` (`triad_rate`)
/// are both `Detect`-only. `symmetry.ritual_conclusion`
/// (`ritual_marker_rate`) and `symmetry.participial_closer`
/// (`participial_closer_rate`) do reach `Gate::Fix`, but with a flat
/// budget (`1` and a `1000`-entry safety cap respectively — see each
/// rule's own `gate`) rather than one scaled from envelope excess. Every
/// entry here therefore uses the crate-wide default; all four still keep
/// their real rule's own actionable direction (all four only ever fix an
/// excess *above* `hi`).
const SYMMETRY: &[DrivingMetric] = &[
    metric(
        "not_just_but_rate",
        Direction::AboveHi,
        DEFAULT_PER_FIX_EFFECT,
    ),
    metric(
        "participial_closer_rate",
        Direction::AboveHi,
        DEFAULT_PER_FIX_EFFECT,
    ),
    metric(
        "ritual_marker_rate",
        Direction::AboveHi,
        DEFAULT_PER_FIX_EFFECT,
    ),
    metric("triad_rate", Direction::AboveHi, DEFAULT_PER_FIX_EFFECT),
];

/// Connective: `connective.surgery` (`discourse_marker_density`, above
/// `hi`, `PER_FIX_EFFECT = 1.0`).
const CONNECTIVE: &[DrivingMetric] = &[metric("discourse_marker_density", Direction::AboveHi, 1.0)];

/// Lexical: `lexical.filler_phrase` (`discourse_marker_density`, above
/// `hi`, `PER_FIX_EFFECT = 1.0` — the same metric `connective.surgery`
/// targets, from a different family, with the same real effect size) and
/// `lexical.substitution` (`llm_favored_phrase_rate`, above `hi`,
/// `PER_FIX_EFFECT = 1.0`).
const LEXICAL: &[DrivingMetric] = &[
    metric("discourse_marker_density", Direction::AboveHi, 1.0),
    metric("llm_favored_phrase_rate", Direction::AboveHi, 1.0),
];

/// Rhythm: `rhythm.split` (`sentence_length_cv`, below `lo`,
/// `PER_FIX_EFFECT = 0.05`) is this family's only real excess-scaled
/// `Fix` rule, so this table uses its exact value rather than the
/// crate-wide default. `rhythm.fuse` targets the same metric in the same
/// direction (too uniform, below `lo`) but is `Detect`-only, so it
/// contributes no separate entry.
const RHYTHM: &[DrivingMetric] = &[metric("sentence_length_cv", Direction::BelowLo, 0.05)];

/// Contraction: `contraction.insert` (`contraction_ratio`, below `lo`)
/// reaches `Gate::Fix`, but — like `symmetry.ritual_conclusion` and
/// `symmetry.participial_closer` — with a flat `1000`-entry safety-cap
/// budget rather than one scaled from envelope excess (the real per-round
/// limit is instead computed exactly, from the real document, inside
/// `fix` itself — see that rule's own module docs). This table uses the
/// crate-wide default, keeping the real rule's own actionable direction
/// (an excess *below* `lo`, i.e. a contraction deficit).
const CONTRACTION: &[DrivingMetric] = &[metric(
    "contraction_ratio",
    Direction::BelowLo,
    DEFAULT_PER_FIX_EFFECT,
)];

/// The driving metrics for `family`, in a fixed order (matching the
/// per-family doc comments above).
pub const fn metrics_for(family: RuleFamily) -> &'static [DrivingMetric] {
    match family {
        RuleFamily::Structural => STRUCTURAL,
        RuleFamily::Symmetry => SYMMETRY,
        RuleFamily::Connective => CONNECTIVE,
        RuleFamily::Lexical => LEXICAL,
        RuleFamily::Rhythm => RHYTHM,
        RuleFamily::Contraction => CONTRACTION,
        // `RuleFamily` is `#[non_exhaustive]`: a future variant this
        // table has no mapping for yet drives no metrics at all, rather
        // than panicking a downstream `Plan::build` caller.
        _ => &[],
    }
}

/// The fixed six families in [`crate::Plan`]'s schedule order (see that
/// type's own docs for the rationale): `Structural, Symmetry, Connective,
/// Lexical, Rhythm, Contraction`. Identical to
/// [`friction_rules::RuleFamily::priority`]'s own ascending order.
pub const ALL_FAMILIES: [RuleFamily; 6] = [
    RuleFamily::Structural,
    RuleFamily::Symmetry,
    RuleFamily::Connective,
    RuleFamily::Lexical,
    RuleFamily::Rhythm,
    RuleFamily::Contraction,
];

/// This family's canonical lowercase name, used for JSON/table output.
///
/// `friction_rules::RuleFamily` has no `Display`/`Serialize` of its own
/// and is `#[non_exhaustive]`, so this crate owns a small, explicit
/// mapping rather than deriving one.
pub const fn family_name(family: RuleFamily) -> &'static str {
    match family {
        RuleFamily::Structural => "structural",
        RuleFamily::Symmetry => "symmetry",
        RuleFamily::Connective => "connective",
        RuleFamily::Lexical => "lexical",
        RuleFamily::Rhythm => "rhythm",
        RuleFamily::Contraction => "contraction",
        _ => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use friction_core::MetricVector;

    use super::*;

    /// Every driving metric name in every family's table is a real
    /// [`MetricVector`] field — catches a typo in this file at test time
    /// rather than silently sizing a family's budget from a metric that
    /// is always absent.
    #[test]
    fn every_driving_metric_name_is_a_real_metric_vector_field() {
        for &family in &ALL_FAMILIES {
            for dm in metrics_for(family) {
                assert!(
                    MetricVector::FIELD_NAMES.contains(&dm.name),
                    "family {family:?}'s driving metric {:?} is not a MetricVector field",
                    dm.name
                );
            }
        }
    }

    /// `ALL_FAMILIES` is exactly the fixed six, in the documented order.
    #[test]
    fn all_families_is_the_fixed_six_in_order() {
        assert_eq!(
            ALL_FAMILIES,
            [
                RuleFamily::Structural,
                RuleFamily::Symmetry,
                RuleFamily::Connective,
                RuleFamily::Lexical,
                RuleFamily::Rhythm,
                RuleFamily::Contraction,
            ]
        );
    }

    /// `family_name` gives every one of the fixed six a distinct,
    /// non-empty name.
    #[test]
    fn family_name_is_distinct_and_non_empty_for_every_family() {
        let names: Vec<&str> = ALL_FAMILIES.iter().map(|&f| family_name(f)).collect();
        for name in &names {
            assert!(!name.is_empty());
            assert_ne!(*name, "unknown");
        }
        let mut sorted = names.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), names.len(), "family names must be distinct");
    }
}
