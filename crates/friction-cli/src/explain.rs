//! `friction explain`: runs the fixpoint engine internally (exactly like
//! `friction fix`) but never prints the fixed text. Instead it prints a
//! before/after [`friction_core::MetricVector`] comparison table (value,
//! envelope band, in/out movement) plus the plan schedule
//! [`friction_plan::Plan`] built from the document's *original* metrics —
//! which of the six rule families are estimated to need work, in what
//! fixed order, and how much — alongside a short summary of what the
//! fixpoint driver actually ran to produce `after`.
//!
//! # "Plan schedule"
//!
//! [`friction_plan::Plan::build`] is a pure, pre-execution estimate: it
//! never runs a rule, only reads `metrics_before` against the genre's
//! envelope bands (see that type's own docs for the exact family order
//! and budget formula). It is deliberately *not* the same thing as "what
//! actually got fixed" — every rule still gates and budgets itself
//! independently from the real document, every round — so this command
//! also reports the fixpoint driver's own realized round count and patch
//! total for comparison.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Args;
use friction_apply::fix_document;
use friction_core::MetricVector;
use friction_plan::Plan;
use serde::Serialize;

use crate::common::{
    CliError, Engine, Format, Genre, Pack, PackEnvelope, read_input, resolve_genre,
};
use crate::table::{self, ComparisonRow, Movement};

/// Arguments for `friction explain`.
#[derive(Debug, Args)]
pub struct ExplainArgs {
    /// File to explain, or `-` to read from stdin.
    input: String,

    /// Genre to fix against (defaults to `docs` with a printed note if
    /// omitted).
    #[arg(long, value_enum)]
    genre: Option<Genre>,

    /// Override the embedded envelope pack with one loaded from `PATH`.
    #[arg(long, value_name = "PATH")]
    pack: Option<PathBuf>,

    /// Output format. `sarif` is not supported here (see `friction
    /// check`).
    #[arg(long, value_enum, default_value_t = Format::Text)]
    format: Format,
}

#[derive(Debug, Serialize)]
struct MetricComparison {
    name: &'static str,
    before: f64,
    after: f64,
    lo: Option<f64>,
    hi: Option<f64>,
    movement: &'static str,
}

/// A short summary of what the fixpoint driver actually ran, for
/// comparison against the advisory [`Plan`] — see the module docs.
#[derive(Debug, Serialize)]
struct FixpointSummary {
    rounds: usize,
    patches_applied: usize,
}

#[derive(Debug, Serialize)]
struct ExplainReport {
    genre: &'static str,
    metrics: Vec<MetricComparison>,
    plan: Plan,
    fixpoint: FixpointSummary,
}

/// One metric's before/after values, envelope band, and classified
/// movement — [`compare_metrics`]'s per-field output, ahead of formatting
/// into either [`MetricComparison`] (JSON) or [`ComparisonRow`] (text
/// table).
struct RawComparison {
    name: &'static str,
    before: f64,
    after: f64,
    lo: Option<f64>,
    hi: Option<f64>,
    movement: Movement,
}

/// Runs `friction explain`.
pub fn run(args: &ExplainArgs) -> ExitCode {
    match run_inner(args) {
        Ok(exit) => exit,
        Err(err) => err.report(),
    }
}

fn run_inner(args: &ExplainArgs) -> Result<ExitCode, CliError> {
    if args.format == Format::Sarif {
        return Err(CliError::SarifUnsupported);
    }

    let source = read_input(&args.input)?;
    let genre = resolve_genre(args.genre);
    let pack = Pack::load(args.pack.as_deref())?;
    let envelope = PackEnvelope::new(pack.as_pack(), genre.as_str());
    let engine = Engine::load()?;

    let metrics_before = compute_metrics(&source, &engine)?;
    let plan = Plan::build(&metrics_before, &envelope);

    let (output, report) = fix_document(
        &source,
        genre.as_str(),
        &envelope,
        &engine.segmenter,
        &engine.tagger,
    )?;
    let metrics_after = compute_metrics(&output, &engine)?;

    let comparisons = compare_metrics(
        &metrics_before,
        &metrics_after,
        pack.as_pack(),
        genre.as_str(),
    );
    let fixpoint_summary = FixpointSummary {
        rounds: report.rounds.len(),
        patches_applied: report.total_patches_applied(),
    };

    match args.format {
        Format::Json | Format::Sarif => {
            let explain_report = ExplainReport {
                genre: genre.as_str(),
                metrics: comparisons
                    .iter()
                    .map(|c| MetricComparison {
                        name: c.name,
                        before: c.before,
                        after: c.after,
                        lo: c.lo,
                        hi: c.hi,
                        movement: c.movement.label(),
                    })
                    .collect(),
                plan,
                fixpoint: fixpoint_summary,
            };
            let json = serde_json::to_string_pretty(&explain_report)
                .expect("ExplainReport serializes: every field is plain data");
            println!("{json}");
        }
        Format::Text => {
            let rows: Vec<ComparisonRow<'_>> = comparisons
                .iter()
                .map(|c| ComparisonRow {
                    name: c.name,
                    before: c.before,
                    after: c.after,
                    lo: c.lo,
                    hi: c.hi,
                })
                .collect();
            print!("{}", table::render_comparison_table(&rows));
            println!();
            println!("plan schedule (friction-plan, built from the document's original metrics):");
            print!("{plan}");
            println!();
            println!(
                "fixpoint: {} round(s), {} patch(es) applied",
                fixpoint_summary.rounds, fixpoint_summary.patches_applied
            );
        }
    }

    Ok(ExitCode::SUCCESS)
}

fn compute_metrics(source: &str, engine: &Engine) -> Result<MetricVector, CliError> {
    let document = friction_parse::parse(source)?;
    Ok(friction_metrics::compute(
        &document,
        &engine.segmenter,
        &engine.tagger,
    ))
}

/// Pairs every metric's before/after value with its envelope band (if
/// any) and classified movement, in [`MetricVector::FIELD_NAMES`] order.
fn compare_metrics(
    before: &MetricVector,
    after: &MetricVector,
    pack: &friction_packs::EnvelopePack,
    genre: &str,
) -> Vec<RawComparison> {
    MetricVector::FIELD_NAMES
        .iter()
        .map(|&name| {
            let before_value = before.get(name).unwrap_or(0.0);
            let after_value = after.get(name).unwrap_or(0.0);
            let band = pack.band(genre, name);
            let movement =
                Movement::classify(before_value, after_value, band.map(|b| (b.lo, b.hi)));
            RawComparison {
                name,
                before: before_value,
                after: after_value,
                lo: band.map(|b| b.lo),
                hi: band.map(|b| b.hi),
                movement,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `compare_metrics` covers every field of `MetricVector`, in the
    /// vector's own declared order.
    #[test]
    fn compare_metrics_covers_every_field_in_order() {
        let before = MetricVector::default();
        let after = MetricVector::default();
        let pack = friction_packs::EnvelopePack::parse("").expect("empty pack parses");
        let rows = compare_metrics(&before, &after, &pack, "blog");
        let names: Vec<&str> = rows.iter().map(|r| r.name).collect();
        assert_eq!(names, MetricVector::FIELD_NAMES.to_vec());
    }

    /// A metric with a band both before and after inside it classifies as
    /// `StayedIn`.
    #[test]
    fn compare_metrics_classifies_movement_against_the_genre_band() {
        let before = MetricVector {
            triad_rate: 0.2,
            ..MetricVector::default()
        };
        let after = MetricVector {
            triad_rate: 0.2,
            ..MetricVector::default()
        };
        let pack = friction_packs::EnvelopePack::parse("[blog.triad_rate]\nlo = 0.0\nhi = 0.5\n")
            .expect("sample pack parses");
        let rows = compare_metrics(&before, &after, &pack, "blog");
        let triad = rows.iter().find(|r| r.name == "triad_rate").unwrap();
        assert_eq!(triad.movement, Movement::StayedIn);
    }
}
