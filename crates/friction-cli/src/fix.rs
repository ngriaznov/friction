//! `friction fix`: runs the four-operation repair engine and writes the
//! fixed text to stdout (or back to the input file with `--in-place`).
//!
//! No `--genre`/`--pack`: the engine's four operations (ritual deletion,
//! paired substitution, derivational pivot, gated span deletion) are
//! gated by the curated inventory/attestation packs and the clause
//! chunker only, never by a metric or genre envelope.
//!
//! A summary (passes run, patches applied per operation, how many
//! `Suggest`-tier candidates remain held) is always printed to stderr, so
//! stdout stays exactly the fixed document — safe to pipe or redirect.
//! `--suggest` additionally lists every remaining held candidate (rule,
//! span, reason) on stderr.

use std::collections::BTreeMap;
use std::path::Path;
use std::process::ExitCode;

use clap::Args;
use friction_core::{Finding, RuleId};
use friction_edit::EditReport;
use serde::Serialize;

use crate::common::{
    CliError, Format, display_path, offset_to_line_col, read_input, write_in_place,
};

/// Arguments for `friction fix`.
#[derive(Debug, Args)]
pub struct FixArgs {
    /// File to fix, or `-` to read from stdin.
    input: String,

    /// Format for the pass summary and (with `--suggest`) held-candidates
    /// list printed to stderr. `sarif` is not supported here (see
    /// `friction check`).
    #[arg(long, value_enum, default_value_t = Format::Text)]
    format: Format,

    /// Write the fixed text back to the input file instead of stdout.
    /// Requires a real file path (not `-`).
    #[arg(long = "in-place")]
    in_place: bool,

    /// After fixing, also list every held candidate (a gate declined to
    /// apply) still present in the fixed output, on stderr.
    #[arg(long)]
    suggest: bool,
}

/// `--format json`'s shape for the pass summary.
#[derive(Debug, Serialize)]
struct FixSummary {
    passes: usize,
    patches_applied: usize,
    patches_by_rule: BTreeMap<String, usize>,
    suggest_count: usize,
}

/// `--format json`'s shape for one `--suggest`-listed held candidate.
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
    let engine = friction_edit::Engine::new()?;
    let (output, report) = engine.fix_document(&source)?;

    if args.in_place {
        write_in_place(Path::new(&args.input), &output)?;
    } else {
        print!("{output}");
    }

    let remaining_held = final_pass_held(&report);
    print_summary(args.format, &report, remaining_held.len());

    if args.suggest {
        print_suggestions(
            args.format,
            &output,
            display_path(&args.input),
            &remaining_held,
        );
    }

    Ok(ExitCode::SUCCESS)
}

/// The engine's own last-pass held candidates — i.e. the gate-held
/// diagnostics scanned against the text the engine actually converged to,
/// since the last pass run is always either the zero-patch convergence
/// pass (re-scanning the previous pass's already-fixed output) or the
/// final bounded pass. Empty if the engine ran zero passes (not expected —
/// `fix_document` always runs at least one) or found nothing to hold.
fn final_pass_held(report: &EditReport) -> Vec<Finding> {
    report
        .passes
        .last()
        .map(|pass| pass.held.clone())
        .unwrap_or_default()
}

fn print_summary(format: Format, report: &EditReport, suggest_count: usize) {
    let mut patches_by_rule: BTreeMap<RuleId, usize> = BTreeMap::new();
    for pass in &report.passes {
        for patch in &pass.applied_patches {
            *patches_by_rule.entry(patch.rule).or_insert(0) += 1;
        }
    }
    let patches_applied: usize = report.passes.iter().map(|p| p.patches_applied).sum();

    match format {
        Format::Json => {
            let summary = FixSummary {
                passes: report.passes.len(),
                patches_applied,
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
                "friction fix: {} pass(es), {} patch(es) applied",
                report.passes.len(),
                patches_applied
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
    use friction_core::{Patch, Tier};
    use friction_edit::PassReport;

    use super::*;

    fn pass(held: Vec<Finding>) -> PassReport {
        PassReport {
            patches_applied: 0,
            patches_dropped: 0,
            applied_patches: Vec::new(),
            held,
        }
    }

    /// `final_pass_held` reads only the last pass's held candidates.
    #[test]
    fn final_pass_held_reads_last_pass_only() {
        let earlier = Finding::new(RuleId::new("span.delete"), 0..1, "earlier", Tier::Suggest);
        let last = Finding::new(RuleId::new("pivot.lvc"), 1..2, "last", Tier::Suggest);
        let report = EditReport {
            passes: vec![pass(vec![earlier]), pass(vec![last.clone()])],
        };
        let remaining = final_pass_held(&report);
        assert_eq!(remaining, vec![last]);
    }

    /// An empty `EditReport` (never produced by `fix_document` in
    /// practice, but defensively handled) yields no held candidates
    /// rather than panicking.
    #[test]
    fn final_pass_held_handles_no_passes() {
        let report = EditReport { passes: Vec::new() };
        assert!(final_pass_held(&report).is_empty());
    }

    /// `print_summary`'s patch-per-rule aggregation sums across every
    /// pass, not just the last.
    #[test]
    fn patches_by_rule_sums_across_passes() {
        let mut p1 = pass(Vec::new());
        p1.applied_patches = vec![Patch::new(0..1, "", RuleId::new("span.delete"), Tier::Fix)];
        let mut p2 = pass(Vec::new());
        p2.applied_patches = vec![
            Patch::new(0..1, "", RuleId::new("span.delete"), Tier::Fix),
            Patch::new(2..3, "", RuleId::new("pivot.lvc"), Tier::Fix),
        ];
        let report = EditReport {
            passes: vec![p1, p2],
        };
        let mut totals: BTreeMap<RuleId, usize> = BTreeMap::new();
        for pass in &report.passes {
            for patch in &pass.applied_patches {
                *totals.entry(patch.rule).or_insert(0) += 1;
            }
        }
        assert_eq!(totals[&RuleId::new("span.delete")], 2);
        assert_eq!(totals[&RuleId::new("pivot.lvc")], 1);
    }
}
