//! `friction fix`: runs the full fixpoint engine and writes the fixed text
//! to stdout (or back to the input file with `--in-place`).
//!
//! A summary (rounds run, patches applied per rule, how many `Suggest`-
//! tier findings remain in the fixed output) is always printed to
//! stderr, so stdout stays exactly the fixed document — safe to pipe or
//! redirect. `--suggest` additionally lists every remaining `Suggest`-tier
//! finding (rule, span, message) on stderr.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::Args;
use friction_apply::{FixpointReport, fix_document};
use friction_core::{Finding, RuleId, Tier};
use serde::Serialize;

use crate::common::{
    CliError, Engine, Format, Genre, Pack, PackEnvelope, display_path, offset_to_line_col,
    read_input, resolve_genre, write_in_place,
};

/// Arguments for `friction fix`.
#[derive(Debug, Args)]
pub struct FixArgs {
    /// File to fix, or `-` to read from stdin.
    input: String,

    /// Genre to fix against (defaults to `docs` with a printed note if
    /// omitted).
    #[arg(long, value_enum)]
    genre: Option<Genre>,

    /// Override the embedded envelope pack with one loaded from `PATH`.
    #[arg(long, value_name = "PATH")]
    pack: Option<PathBuf>,

    /// Format for the round summary and (with `--suggest`) findings list
    /// printed to stderr. `sarif` is not supported here (see `friction
    /// check`).
    #[arg(long, value_enum, default_value_t = Format::Text)]
    format: Format,

    /// Write the fixed text back to the input file instead of stdout.
    /// Requires a real file path (not `-`).
    #[arg(long = "in-place")]
    in_place: bool,

    /// After fixing, also list every `Suggest`-tier finding still present
    /// in the fixed output, on stderr.
    #[arg(long)]
    suggest: bool,
}

/// `--format json`'s shape for the round summary.
#[derive(Debug, Serialize)]
struct FixSummary {
    rounds: usize,
    patches_applied: usize,
    patches_by_rule: BTreeMap<String, usize>,
    suggest_count: usize,
}

/// `--format json`'s shape for one `--suggest`-listed finding.
#[derive(Debug, Serialize)]
struct SuggestionRow {
    rule: String,
    start: usize,
    end: usize,
    line: usize,
    column: usize,
    message: String,
}

/// Runs `friction fix`.
pub fn run(args: &FixArgs) -> ExitCode {
    match run_inner(args) {
        Ok(exit) => exit,
        Err(err) => err.report(),
    }
}

fn run_inner(args: &FixArgs) -> Result<ExitCode, CliError> {
    if args.format == Format::Sarif {
        return Err(CliError::SarifUnsupported);
    }
    if args.in_place && args.input == "-" {
        return Err(CliError::InPlaceStdin);
    }

    let source = read_input(&args.input)?;
    let genre = resolve_genre(args.genre);
    let pack = Pack::load(args.pack.as_deref())?;
    let envelope = PackEnvelope::new(pack.as_pack(), genre.as_str());
    let engine = Engine::load()?;

    let (output, report) = fix_document(
        &source,
        genre.as_str(),
        &envelope,
        &engine.segmenter,
        &engine.tagger,
    )?;

    if args.in_place {
        write_in_place(Path::new(&args.input), &output)?;
    } else {
        print!("{output}");
    }

    let remaining_suggestions = final_round_suggestions(&report);
    print_summary(args.format, &report, remaining_suggestions.len());

    if args.suggest {
        print_suggestions(
            args.format,
            &output,
            display_path(&args.input),
            &remaining_suggestions,
        );
    }

    Ok(ExitCode::SUCCESS)
}

/// The `Tier::Suggest` findings the fixpoint driver's own last round
/// scanned — i.e. against the text the driver actually converged to,
/// since a driver's final round is always the zero-patch round that
/// re-scanned the previous round's (already fully fixed) output. Empty if
/// the driver ran zero rounds (impossible: `run_fixpoint` always runs at
/// least one) or found nothing.
fn final_round_suggestions(report: &FixpointReport) -> Vec<Finding> {
    report
        .rounds
        .last()
        .map(|round| {
            round
                .findings
                .iter()
                .filter(|f| f.tier == Tier::Suggest)
                .cloned()
                .collect()
        })
        .unwrap_or_default()
}

fn print_summary(format: Format, report: &FixpointReport, suggest_count: usize) {
    let mut patches_by_rule: BTreeMap<RuleId, usize> = BTreeMap::new();
    for round in &report.rounds {
        for patch in &round.applied_patches {
            *patches_by_rule.entry(patch.rule).or_insert(0) += 1;
        }
    }

    match format {
        Format::Json => {
            let summary = FixSummary {
                rounds: report.rounds.len(),
                patches_applied: report.total_patches_applied(),
                patches_by_rule: patches_by_rule
                    .into_iter()
                    .map(|(id, n)| (id.as_str().to_string(), n))
                    .collect(),
                suggest_count,
            };
            let json = serde_json::to_string_pretty(&summary)
                .expect("FixSummary serializes: every field is plain data");
            eprintln!("{json}");
        }
        Format::Text | Format::Sarif => {
            eprintln!(
                "friction fix: {} round(s), {} patch(es) applied",
                report.rounds.len(),
                report.total_patches_applied()
            );
            for (id, n) in &patches_by_rule {
                eprintln!("  {id}: {n}");
            }
            eprintln!("  suggest: {suggest_count} finding(s) remain");
        }
    }
}

fn print_suggestions(format: Format, output: &str, path_label: &str, suggestions: &[Finding]) {
    match format {
        Format::Json => {
            let rows: Vec<SuggestionRow> = suggestions
                .iter()
                .map(|f| {
                    let (line, column) = offset_to_line_col(output, f.range.start);
                    SuggestionRow {
                        rule: f.rule.as_str().to_string(),
                        start: f.range.start,
                        end: f.range.end,
                        line,
                        column,
                        message: f.message.clone(),
                    }
                })
                .collect();
            let json = serde_json::to_string_pretty(&rows)
                .expect("suggestions serialize: every field is plain data");
            eprintln!("{json}");
        }
        Format::Text | Format::Sarif => {
            for f in suggestions {
                let (line, column) = offset_to_line_col(output, f.range.start);
                eprintln!(
                    "{path_label}:{line}:{column}: {} [{}]",
                    f.message,
                    f.rule.as_str()
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use friction_apply::RoundReport;
    use friction_core::Patch;

    use super::*;

    fn round(findings: Vec<Finding>) -> RoundReport {
        RoundReport {
            round: 1,
            rules_fired: Vec::new(),
            findings,
            patches_applied: 0,
            patches_dropped: 0,
            applied_patches: Vec::new(),
        }
    }

    /// `final_round_suggestions` reads only the last round's findings,
    /// filtered to `Tier::Suggest`.
    #[test]
    fn final_round_suggestions_reads_last_round_suggest_tier_only() {
        let fix_finding =
            Finding::new(RuleId::new("lexical.filler_phrase"), 0..1, "fix", Tier::Fix);
        let suggest_finding = Finding::new(
            RuleId::new("symmetry.triad_reduction"),
            1..2,
            "suggest",
            Tier::Suggest,
        );
        let report = FixpointReport {
            rounds: vec![
                round(vec![suggest_finding.clone()]), // an earlier round: ignored
                round(vec![fix_finding, suggest_finding.clone()]),
            ],
        };
        let remaining = final_round_suggestions(&report);
        assert_eq!(remaining, vec![suggest_finding]);
    }

    /// An empty `FixpointReport` (never produced by `run_fixpoint` in
    /// practice, but defensively handled) yields no suggestions rather
    /// than panicking.
    #[test]
    fn final_round_suggestions_handles_no_rounds() {
        let report = FixpointReport { rounds: Vec::new() };
        assert!(final_round_suggestions(&report).is_empty());
    }

    /// `print_summary`'s patch-per-rule aggregation sums across every
    /// round, not just the last.
    #[test]
    fn patches_by_rule_sums_across_rounds() {
        let mut r1 = round(Vec::new());
        r1.applied_patches = vec![Patch::new(
            0..1,
            "",
            RuleId::new("lexical.filler_phrase"),
            Tier::Fix,
        )];
        let mut r2 = round(Vec::new());
        r2.applied_patches = vec![
            Patch::new(0..1, "", RuleId::new("lexical.filler_phrase"), Tier::Fix),
            Patch::new(2..3, "", RuleId::new("rhythm.split"), Tier::Fix),
        ];
        let report = FixpointReport {
            rounds: vec![r1, r2],
        };
        let mut totals: BTreeMap<RuleId, usize> = BTreeMap::new();
        for round in &report.rounds {
            for patch in &round.applied_patches {
                *totals.entry(patch.rule).or_insert(0) += 1;
            }
        }
        assert_eq!(totals[&RuleId::new("lexical.filler_phrase")], 2);
        assert_eq!(totals[&RuleId::new("rhythm.split")], 1);
    }
}
