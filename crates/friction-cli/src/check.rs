//! `friction check`: parse + metrics + gate + scan, with no fixes applied.
//!
//! Prints a per-metric table (value, envelope band, in/out) and every
//! surfaced finding, in `--format text` (a plain table plus `miette`
//! labeled-span diagnostics — see [`crate::diagnostics`]), `--format
//! json` (stable `serde` structs), or `--format sarif` ([`crate::sarif`]).
//!
//! Exit code: `0` if every banded metric sits inside its envelope and no
//! rule surfaced a finding; `1` if either is false; `2` on error (see
//! [`CliError::report`]).

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Args;
use friction_core::MetricVector;
use serde::Serialize;

use crate::common::{
    CliError, Engine, Format, Genre, Pack, PackEnvelope, display_path, offset_to_line_col,
    read_input, resolve_genre,
};
use crate::diagnostics::{color_enabled, render_findings};
use crate::scan::scan;
use crate::{sarif, table};

/// Arguments for `friction check`.
#[derive(Debug, Args)]
pub struct CheckArgs {
    /// File to check, or `-` to read from stdin.
    input: String,

    /// Genre to check against (defaults to `docs` with a printed note if
    /// omitted).
    #[arg(long, value_enum)]
    genre: Option<Genre>,

    /// Override the embedded envelope pack with one loaded from `PATH`.
    #[arg(long, value_name = "PATH")]
    pack: Option<PathBuf>,

    /// Output format.
    #[arg(long, value_enum, default_value_t = Format::Text)]
    format: Format,

    /// Disable `--format text`'s ANSI color, regardless of whether stdout
    /// is a terminal. Implied by the `NO_COLOR` environment variable; see
    /// `crate::diagnostics` for the full auto-detection policy.
    #[arg(long)]
    no_color: bool,
}

/// One metric's value, this genre's envelope band for it (if the pack has
/// one), and whether the value falls inside that band.
#[derive(Debug, Serialize)]
struct MetricRow {
    name: &'static str,
    value: f64,
    lo: Option<f64>,
    hi: Option<f64>,
    in_envelope: Option<bool>,
}

/// One finding, flattened to a stable, serializable shape (1-based
/// line/column alongside the raw byte range, for a JSON consumer that
/// wants either).
#[derive(Debug, Serialize)]
struct FindingRow {
    rule: String,
    tier: &'static str,
    start: usize,
    end: usize,
    line: usize,
    column: usize,
    message: String,
}

/// The full `--format json` shape: every field a `MetricVector` has, plus
/// every surfaced finding.
#[derive(Debug, Serialize)]
struct CheckReport {
    genre: &'static str,
    metrics: Vec<MetricRow>,
    findings: Vec<FindingRow>,
}

/// Runs `friction check`.
pub fn run(args: &CheckArgs) -> ExitCode {
    match run_inner(args) {
        Ok(exit) => exit,
        Err(err) => err.report(),
    }
}

fn run_inner(args: &CheckArgs) -> Result<ExitCode, CliError> {
    let source = read_input(&args.input)?;
    let genre = resolve_genre(args.genre);
    let pack = Pack::load(args.pack.as_deref())?;
    let envelope = PackEnvelope::new(pack.as_pack(), genre.as_str());
    let engine = Engine::load()?;

    let outcome = scan(&source, genre.as_str(), &envelope, &engine)?;
    let rows = metric_rows(&outcome.metrics, pack.as_pack(), genre.as_str());
    let all_in_envelope = rows.iter().all(|row| row.in_envelope.unwrap_or(true));
    let path_label = display_path(&args.input);

    match args.format {
        Format::Text => {
            print!("{}", table::render_metric_table(&rows_for_table(&rows)));
            let color = color_enabled(args.no_color);
            let rendered = render_findings(&source, path_label, &outcome.findings, color);
            print!("{rendered}");
        }
        Format::Json => {
            let report = CheckReport {
                genre: genre.as_str(),
                metrics: rows,
                findings: outcome
                    .findings
                    .iter()
                    .map(|f| {
                        let (line, column) = offset_to_line_col(&source, f.range.start);
                        FindingRow {
                            rule: f.rule.as_str().to_string(),
                            tier: tier_str(f.tier),
                            start: f.range.start,
                            end: f.range.end,
                            line,
                            column,
                            message: f.message.clone(),
                        }
                    })
                    .collect(),
            };
            let json = serde_json::to_string_pretty(&report)
                .expect("CheckReport serializes: every field is plain data");
            println!("{json}");
        }
        Format::Sarif => {
            let json = sarif::render(&outcome.findings, &source, path_label);
            println!("{json}");
        }
    }

    let exit_ok = all_in_envelope && outcome.findings.is_empty();
    Ok(if exit_ok {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    })
}

const fn tier_str(tier: friction_core::Tier) -> &'static str {
    match tier {
        friction_core::Tier::Fix => "fix",
        friction_core::Tier::Suggest => "suggest",
    }
}

fn metric_rows(
    metrics: &MetricVector,
    pack: &friction_packs::EnvelopePack,
    genre: &str,
) -> Vec<MetricRow> {
    metrics
        .named_values()
        .into_iter()
        .map(|(name, value)| {
            let band = pack.band(genre, name);
            MetricRow {
                name,
                value,
                lo: band.map(|b| b.lo),
                hi: band.map(|b| b.hi),
                in_envelope: band.map(|b| b.contains(value)),
            }
        })
        .collect()
}

fn rows_for_table(rows: &[MetricRow]) -> Vec<table::MetricTableRow<'_>> {
    rows.iter()
        .map(|row| table::MetricTableRow {
            name: row.name,
            value: row.value,
            lo: row.lo,
            hi: row.hi,
            in_envelope: row.in_envelope,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `metric_rows` looks up a band for every metric the pack has one
    /// for, and reports `None` for the rest — hand-checked against a tiny
    /// pack with exactly one banded metric.
    #[test]
    fn metric_rows_looks_up_bands_and_reports_containment() {
        let pack = friction_packs::EnvelopePack::parse("[blog.triad_rate]\nlo = 0.0\nhi = 0.5\n")
            .expect("sample pack parses");
        let metrics = MetricVector {
            triad_rate: 0.3,
            em_dash_density: 9.0,
            ..MetricVector::default()
        };
        let rows = metric_rows(&metrics, &pack, "blog");

        let triad = rows
            .iter()
            .find(|r| r.name == "triad_rate")
            .expect("triad_rate row exists");
        assert_eq!(triad.lo, Some(0.0));
        assert_eq!(triad.hi, Some(0.5));
        assert_eq!(triad.in_envelope, Some(true));

        let em_dash = rows
            .iter()
            .find(|r| r.name == "em_dash_density")
            .expect("em_dash_density row exists");
        assert_eq!(em_dash.lo, None);
        assert_eq!(em_dash.in_envelope, None);
    }
}
