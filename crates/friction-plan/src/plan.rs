//! [`Plan`]: an ordered, budgeted rule schedule built from a document's
//! [`MetricVector`] and a genre's envelope bands.

use std::fmt;

use friction_core::{Envelope, MetricVector};
use friction_rules::{Budget, GenreEnvelope, RuleFamily};
use serde::Serialize;

use crate::mapping::{ALL_FAMILIES, Direction, family_name, metrics_for};

/// One driving metric's contribution to a [`PlanEntry`]'s budget.
///
/// `excess` and `budget` are always `0.0`/`0` when `band` is `None` (the
/// envelope pack has no band for this metric in this genre) or when
/// `current` sits on the non-actionable side of the band (see
/// [`crate::mapping::Direction`]) — a `Plan` never invents work a rule
/// couldn't safely do.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MetricDelta {
    /// The metric's name (matches [`MetricVector::FIELD_NAMES`]).
    pub metric: &'static str,
    /// The document's current value for this metric.
    pub current: f64,
    /// This genre's envelope band for this metric, or `None` if the pack
    /// has no band for it.
    pub band: Option<Envelope>,
    /// How far `current` sits outside `band` in the actionable direction;
    /// `0.0` if inside the band, on the non-actionable side, or bandless.
    pub excess: f64,
    /// The per-fix effect size used to convert `excess` into `budget` via
    /// [`Budget::from_envelope_excess`] (see `crate::mapping` for where
    /// each value comes from).
    pub per_fix_effect: f64,
    /// This metric's own contribution to the owning [`PlanEntry`]'s
    /// summed `budget`.
    pub budget: usize,
}

/// One family's scheduled entry: its family, the summed advisory budget
/// across every metric that drives it (see `crate::mapping`), and the
/// per-metric deltas that summed budget came from.
///
/// A family's position in the fixed schedule is its index in
/// [`Plan::entries`], not a field here — see [`Plan`]'s own docs for the
/// order and its rationale.
#[derive(Debug, Clone, Serialize)]
pub struct PlanEntry {
    /// This entry's family.
    #[serde(serialize_with = "serialize_family")]
    pub family: RuleFamily,
    /// The summed advisory budget across every driving metric in
    /// `driving_metrics`: how many patches this family is estimated to
    /// need this round.
    pub budget: usize,
    /// Every metric that drives this family's budget, and each one's own
    /// contribution, in `crate::mapping`'s fixed per-family order.
    pub driving_metrics: Vec<MetricDelta>,
}

// `serde(serialize_with = ...)` requires this exact `fn(&T, S) -> ...`
// shape, so the by-reference `family` parameter clippy would otherwise
// flag (`RuleFamily` is `Copy` and one byte) is not this function's
// choice to make.
#[allow(clippy::trivially_copy_pass_by_ref)]
fn serialize_family<S>(family: &RuleFamily, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    serializer.serialize_str(family_name(*family))
}

/// An ordered, budgeted rule schedule: which of the six rule families
/// should act this run, in what fixed order, and how hard.
///
/// # Family order
///
/// [`Plan::entries`] always lists every one of the six families, in this
/// fixed order, regardless of the metrics or envelope a `Plan` was built
/// from:
///
/// `Structural, Symmetry, Connective, Lexical, Rhythm, Contraction`
///
/// This is [`RuleFamily::priority`]'s own documented order — outer,
/// larger-span transforms first (structural document/list shape, then
/// coordination/closer symmetry, then sentence-initial connectives) and
/// smaller, more localized ones last (lexical substitution, sentence
/// rhythm, contraction insertion). The rationale is the same one that
/// order already serves for conflict-resolution priority: an outer
/// transform invalidates fewer of the spans nested inside it than the
/// reverse would, so doing outer work first leaves the most later work
/// intact.
///
/// # Budgets are advisory, not authoritative
///
/// Every rule still gates and budgets itself independently, from the
/// *real* document each round (see `friction-rules`' own `Rule::gate`
/// docs) — a `Plan` never substitutes for that. It exists as a
/// document-level, pre-round estimate for two consumers: a CLI wanting to
/// explain *why* a run is about to touch a document before it does, and
/// (via `friction-apply`'s driver, see its own docs) an optional
/// additional cap on how many patches a family may contribute to one
/// round, layered on top of — never in place of — each rule's own
/// gating.
///
/// # Determinism
///
/// [`Plan::build`] is a pure function of `(metrics, envelope)`: the same
/// two inputs always produce the same `Plan`, byte-for-byte through
/// either [`fmt::Display`] or [`serde::Serialize`]. Every float is a
/// direct, unrounded copy of its input; both serializations only ever
/// round for display/output formatting (see each impl's own docs).
#[derive(Debug, Clone, Serialize)]
pub struct Plan {
    entries: Vec<PlanEntry>,
}

impl Plan {
    /// Builds a `Plan` from a document's metrics and its genre's envelope
    /// bands.
    ///
    /// For each of the fixed six families, in order, sums
    /// [`Budget::from_envelope_excess`] (via `crate::mapping`'s per-family
    /// metric table) across every metric that family targets, giving that
    /// sum as the family's advisory budget.
    #[must_use]
    pub fn build(metrics: &MetricVector, envelope: &dyn GenreEnvelope) -> Self {
        let entries = ALL_FAMILIES
            .iter()
            .map(|&family| Self::entry_for(family, metrics, envelope))
            .collect();
        Self { entries }
    }

    fn entry_for(
        family: RuleFamily,
        metrics: &MetricVector,
        envelope: &dyn GenreEnvelope,
    ) -> PlanEntry {
        let mut driving_metrics = Vec::new();
        let mut budget = 0usize;
        for dm in metrics_for(family) {
            let current = metrics
                .get(dm.name)
                .expect("crate::mapping's driving metric names are all real MetricVector fields");
            let band = envelope.band(dm.name);
            let (excess, contribution) = band.map_or((0.0, 0), |b| {
                let actionable = match dm.direction {
                    Direction::AboveHi => current > b.hi,
                    Direction::BelowLo => current < b.lo,
                };
                if actionable {
                    let excess = match dm.direction {
                        Direction::AboveHi => current - b.hi,
                        Direction::BelowLo => b.lo - current,
                    };
                    let contribution =
                        Budget::from_envelope_excess(current, b, dm.per_fix_effect).remaining();
                    (excess, contribution)
                } else {
                    (0.0, 0)
                }
            });
            budget += contribution;
            driving_metrics.push(MetricDelta {
                metric: dm.name,
                current,
                band,
                excess,
                per_fix_effect: dm.per_fix_effect,
                budget: contribution,
            });
        }
        PlanEntry {
            family,
            budget,
            driving_metrics,
        }
    }

    /// Every family's entry, in the fixed schedule order (see this
    /// type's own docs).
    #[must_use]
    pub fn entries(&self) -> &[PlanEntry] {
        &self.entries
    }

    /// `family`'s advisory budget: how many patches this family is
    /// estimated to need this round, summed across its driving metrics.
    ///
    /// `0` for any family not present in [`Plan::entries`] — unreachable
    /// for a `Plan` built by [`Plan::build`], which always populates the
    /// fixed six, but a safe default rather than a panic for a caller
    /// holding a `RuleFamily` from a future, `#[non_exhaustive]` variant.
    #[must_use]
    pub fn budget_for(&self, family: RuleFamily) -> usize {
        self.entries
            .iter()
            .find(|entry| entry.family == family)
            .map_or(0, |entry| entry.budget)
    }
}

/// Renders `plan` as a deterministic, human-readable Markdown table: one
/// row per driving metric (a family with no driving metrics — unreachable
/// for the fixed six today, but possible for a future family — gets one
/// row of its own with `-` placeholders), grouped under each family in
/// schedule order, with the family name and its summed budget shown once,
/// on that family's first row.
///
/// Every `f64` is formatted to a fixed four decimal places
/// (`{:.4}`) — deterministic regardless of the underlying value's exact
/// binary representation, and matching this codebase's existing
/// float-report convention (e.g. `corpus-tool mine`'s Markdown tables).
impl fmt::Display for Plan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "| family | budget | metric | current | lo | hi | excess |"
        )?;
        writeln!(f, "|---|---|---|---|---|---|---|")?;
        for entry in &self.entries {
            if entry.driving_metrics.is_empty() {
                writeln!(
                    f,
                    "| {} | {} | - | - | - | - | - |",
                    family_name(entry.family),
                    entry.budget
                )?;
                continue;
            }
            for (i, delta) in entry.driving_metrics.iter().enumerate() {
                let family_cell = if i == 0 {
                    family_name(entry.family)
                } else {
                    ""
                };
                let (lo, hi) = delta.band.map_or_else(
                    || ("-".to_string(), "-".to_string()),
                    |b| (format!("{:.4}", b.lo), format!("{:.4}", b.hi)),
                );
                write!(f, "| {family_cell} | ")?;
                if i == 0 {
                    write!(f, "{}", entry.budget)?;
                }
                writeln!(
                    f,
                    " | {} | {:.4} | {lo} | {hi} | {:.4} |",
                    delta.metric, delta.current, delta.excess
                )?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use friction_rules::MapEnvelope;

    use super::*;

    /// Hand-computed: `bold_span_density` at `15.0` against a `[0.0,
    /// 10.0]` band, `PER_FIX_EFFECT = 1.0` (see `crate::mapping::
    /// STRUCTURAL`) -> excess `5.0` -> `floor(5.0 / 1.0) = 5`.
    /// `list_item_density` sits inside its own band (budget `0`), and
    /// `heading_density` has no band at all (budget `0`), so the
    /// structural family's summed budget is exactly `5`.
    // Exact: every value under test is either an input literal copied
    // straight through or the result of exact-in-f64 arithmetic
    // (differences and floors of small decimal literals).
    #[allow(clippy::float_cmp)]
    #[test]
    fn structural_budget_is_hand_computed_from_bold_span_density_alone() {
        let metrics = MetricVector {
            bold_span_density: 15.0,
            list_item_density: 3.0,
            ..MetricVector::default()
        };
        let envelope = MapEnvelope::new()
            .with("bold_span_density", Envelope::new(0.0, 10.0))
            .with("list_item_density", Envelope::new(0.0, 10.0));

        let plan = Plan::build(&metrics, &envelope);
        assert_eq!(plan.budget_for(RuleFamily::Structural), 5);

        let entry = &plan.entries()[0];
        assert_eq!(entry.family, RuleFamily::Structural);
        let bold = entry
            .driving_metrics
            .iter()
            .find(|d| d.metric == "bold_span_density")
            .expect("bold_span_density must be a structural driving metric");
        assert_eq!(bold.excess, 5.0);
        assert_eq!(bold.budget, 5);
        let list_item = entry
            .driving_metrics
            .iter()
            .find(|d| d.metric == "list_item_density")
            .expect("list_item_density must be a structural driving metric");
        assert_eq!(list_item.excess, 0.0);
        assert_eq!(list_item.budget, 0);
        let heading = entry
            .driving_metrics
            .iter()
            .find(|d| d.metric == "heading_density")
            .expect("heading_density must be a structural driving metric");
        assert_eq!(heading.band, None);
        assert_eq!(heading.budget, 0);
    }

    /// Hand-computed: `sentence_length_cv` at `0.30` against a `[0.50,
    /// 1.00]` band, `PER_FIX_EFFECT = 0.05` (see `crate::mapping::
    /// RHYTHM`, the one family whose table uses its real rule's exact
    /// value rather than the crate-wide default) -> deficit `0.20` ->
    /// `floor(0.20 / 0.05) = 4`.
    #[test]
    fn rhythm_budget_is_hand_computed_below_the_band_floor() {
        let metrics = MetricVector {
            sentence_length_cv: 0.30,
            ..MetricVector::default()
        };
        let envelope = MapEnvelope::new().with("sentence_length_cv", Envelope::new(0.50, 1.00));

        let plan = Plan::build(&metrics, &envelope);
        assert_eq!(plan.budget_for(RuleFamily::Rhythm), 4);
    }

    /// Hand-computed: `discourse_marker_density` at `7.0` against a
    /// `[0.0, 2.0]` band, `PER_FIX_EFFECT = 1.0` -> excess `5.0` ->
    /// budget `5`. Both `connective` and `lexical` target this same
    /// metric (see `crate::mapping::CONNECTIVE`/`LEXICAL`), so both
    /// families see that same `5` contribution — `lexical`'s own budget
    /// is higher only because `llm_favored_phrase_rate` adds more on top.
    #[test]
    fn discourse_marker_density_drives_both_connective_and_lexical() {
        let metrics = MetricVector {
            discourse_marker_density: 7.0,
            llm_favored_phrase_rate: 4.0,
            ..MetricVector::default()
        };
        let envelope = MapEnvelope::new()
            .with("discourse_marker_density", Envelope::new(0.0, 2.0))
            .with("llm_favored_phrase_rate", Envelope::new(0.0, 1.0));

        let plan = Plan::build(&metrics, &envelope);
        assert_eq!(plan.budget_for(RuleFamily::Connective), 5);
        // lexical: discourse_marker_density's 5 + llm_favored_phrase_rate's
        // floor(3.0 / 1.0) = 3 -> 8.
        assert_eq!(plan.budget_for(RuleFamily::Lexical), 8);
    }

    /// A metric on the non-actionable side of its band (e.g. a
    /// `contraction_ratio` *above* `hi`, when the real
    /// `contraction.insert` rule only ever acts on a deficit *below*
    /// `lo`) contributes no budget at all, exactly like a value already
    /// inside the band would.
    #[test]
    fn non_actionable_direction_contributes_no_budget() {
        let metrics = MetricVector {
            contraction_ratio: 0.95,
            ..MetricVector::default()
        };
        let envelope = MapEnvelope::new().with("contraction_ratio", Envelope::new(0.2, 0.8));

        let plan = Plan::build(&metrics, &envelope);
        assert_eq!(plan.budget_for(RuleFamily::Contraction), 0);
    }

    /// `Plan::entries` always lists the fixed six families, in the fixed
    /// schedule order, regardless of which metrics or bands a `Plan` was
    /// built from — an empty envelope (every family gates to a `0`
    /// budget) and a maximally "everything is on fire" one produce the
    /// exact same family ordering.
    #[test]
    fn family_order_is_stable_regardless_of_metrics_or_envelope() {
        let expected_order = [
            RuleFamily::Structural,
            RuleFamily::Symmetry,
            RuleFamily::Connective,
            RuleFamily::Lexical,
            RuleFamily::Rhythm,
            RuleFamily::Contraction,
        ];

        let empty_plan = Plan::build(&MetricVector::default(), &MapEnvelope::new());
        let order: Vec<RuleFamily> = empty_plan.entries().iter().map(|e| e.family).collect();
        assert_eq!(order, expected_order);

        let hot_metrics = MetricVector {
            bold_span_density: 1000.0,
            list_item_density: 1000.0,
            heading_density: 1000.0,
            not_just_but_rate: 1000.0,
            participial_closer_rate: 1000.0,
            ritual_marker_rate: 1000.0,
            triad_rate: 1000.0,
            discourse_marker_density: 1000.0,
            llm_favored_phrase_rate: 1000.0,
            sentence_length_cv: -1000.0,
            contraction_ratio: -1000.0,
            ..MetricVector::default()
        };
        let mut hot_envelope = MapEnvelope::new();
        for &family in &ALL_FAMILIES {
            for dm in metrics_for(family) {
                hot_envelope = hot_envelope.with(dm.name, Envelope::new(0.4, 0.6));
            }
        }
        let hot_plan = Plan::build(&hot_metrics, &hot_envelope);
        let hot_order: Vec<RuleFamily> = hot_plan.entries().iter().map(|e| e.family).collect();
        assert_eq!(hot_order, expected_order);
        // Every family actually did gate a non-zero budget this time,
        // confirming the ordering check above wasn't vacuously true over
        // an all-zero plan.
        for &family in &expected_order {
            assert!(
                hot_plan.budget_for(family) > 0,
                "{family:?} should have a non-zero budget against a maximally-outside-band input"
            );
        }
    }

    /// `budget_for` returns `0`, not a panic, for a family this `Plan`
    /// somehow has no entry for — unreachable via `Plan::build` today,
    /// but exercised directly to document the contract `friction-apply`'s
    /// driver relies on (a missing plan entry never blocks a family
    /// outright).
    #[test]
    fn budget_for_defaults_to_zero_without_an_entry() {
        let plan = Plan { entries: vec![] };
        assert_eq!(plan.budget_for(RuleFamily::Structural), 0);
    }

    /// `Plan::build` is a pure function of its two inputs: calling it
    /// twice with equal `(metrics, envelope)` produces entries that
    /// compare equal field-by-field (there is no hidden nondeterminism
    /// — e.g. iteration-order-dependent map lookups — anywhere in
    /// `entry_for`).
    #[test]
    fn build_is_deterministic_across_repeated_calls() {
        let metrics = MetricVector {
            discourse_marker_density: 9.0,
            sentence_length_cv: 0.1,
            ..MetricVector::default()
        };
        let envelope = MapEnvelope::new()
            .with("discourse_marker_density", Envelope::new(0.0, 3.0))
            .with("sentence_length_cv", Envelope::new(0.4, 0.9));

        let first = Plan::build(&metrics, &envelope);
        let second = Plan::build(&metrics, &envelope);

        for (a, b) in first.entries().iter().zip(second.entries()) {
            assert_eq!(a.family, b.family);
            assert_eq!(a.budget, b.budget);
            assert_eq!(a.driving_metrics, b.driving_metrics);
        }
    }
}
