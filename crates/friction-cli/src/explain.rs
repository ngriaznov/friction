//! `friction explain`: runs the four-operation repair engine internally
//! (exactly like `friction fix`) but never prints the fixed text. Instead
//! it prints, per pass, every operation that actually fired (rule,
//! range, and — for a substitution/pivot — what it changed to) and every
//! candidate a gate held back, followed by a short convergence summary.
//!
//! No `--genre`/`--pack`: same reasoning as `friction fix` — the engine's
//! four operations are never gated by a metric or genre envelope.

use std::process::ExitCode;

use clap::Args;
use friction_core::{Finding, Patch};
use friction_edit::EditReport;
use serde::Serialize;

use crate::common::{CliError, Format, read_input};

/// Arguments for `friction explain`.
#[derive(Debug, Args)]
pub struct ExplainArgs {
    /// File to explain, or `-` to read from stdin.
    input: String,

    /// Output format. `sarif` is not supported here (see `friction
    /// check`).
    #[arg(long, value_enum, default_value_t = Format::Text)]
    format: Format,
}

#[derive(Debug, Serialize)]
struct OperationRow {
    rule: String,
    start: usize,
    end: usize,
    replacement: String,
}

#[derive(Debug, Serialize)]
struct HeldRow {
    rule: String,
    start: usize,
    end: usize,
    reason: String,
}

#[derive(Debug, Serialize)]
struct PassRow {
    pass: usize,
    fired: Vec<OperationRow>,
    held: Vec<HeldRow>,
}

#[derive(Debug, Serialize)]
struct ExplainReport {
    passes: Vec<PassRow>,
    patches_applied: usize,
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
    let engine = friction_edit::Engine::new()?;
    let (_output, report) = engine.fix_document(&source)?;

    let passes = pass_rows(&report);
    let patches_applied = report
        .passes
        .iter()
        .map(|p| p.patches_applied)
        .sum::<usize>();

    match args.format {
        Format::Json | Format::Sarif => {
            let explain_report = ExplainReport {
                passes,
                patches_applied,
            };
            let json = serde_json::to_string_pretty(&explain_report)
                .expect("ExplainReport serializes: every field is plain data");
            println!("{json}");
        }
        Format::Text => {
            print_text(&report);
        }
    }

    Ok(ExitCode::SUCCESS)
}

fn pass_rows(report: &EditReport) -> Vec<PassRow> {
    report
        .passes
        .iter()
        .enumerate()
        .map(|(i, pass)| PassRow {
            pass: i + 1,
            fired: pass.applied_patches.iter().map(operation_row).collect(),
            held: pass.held.iter().map(held_row).collect(),
        })
        .collect()
}

fn operation_row(patch: &Patch) -> OperationRow {
    OperationRow {
        rule: patch.rule.as_str().to_string(),
        start: patch.range.start,
        end: patch.range.end,
        replacement: patch.replacement.clone(),
    }
}

fn held_row(finding: &Finding) -> HeldRow {
    HeldRow {
        rule: finding.rule.as_str().to_string(),
        start: finding.range.start,
        end: finding.range.end,
        reason: finding.message.clone(),
    }
}

fn print_text(report: &EditReport) {
    for (i, pass) in report.passes.iter().enumerate() {
        let pass_num = i + 1;
        if pass.patches_applied == 0 {
            println!("pass {pass_num}: 0 operation(s) fired — converged");
        } else {
            println!(
                "pass {pass_num}: {} operation(s) fired",
                pass.patches_applied
            );
            for patch in &pass.applied_patches {
                if patch.replacement.is_empty() {
                    println!(
                        "  {:<24} {}..{}  (deleted)",
                        patch.rule.as_str(),
                        patch.range.start,
                        patch.range.end
                    );
                } else {
                    println!(
                        "  {:<24} {}..{}  -> {:?}",
                        patch.rule.as_str(),
                        patch.range.start,
                        patch.range.end,
                        patch.replacement
                    );
                }
            }
        }
        if !pass.held.is_empty() {
            println!("  {} held:", pass.held.len());
            for finding in &pass.held {
                println!(
                    "    {:<22} {}..{}  KEPT ({})",
                    finding.rule.as_str(),
                    finding.range.start,
                    finding.range.end,
                    finding.message
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use friction_core::{RuleId, Tier};
    use friction_edit::PassReport;

    use super::*;

    fn pass(applied: Vec<Patch>, held: Vec<Finding>) -> PassReport {
        let patches_applied = applied.len();
        PassReport {
            patches_applied,
            patches_dropped: 0,
            applied_patches: applied,
            held,
        }
    }

    /// `pass_rows` numbers passes 1-indexed and carries every fired
    /// operation and held candidate through unchanged.
    #[test]
    fn pass_rows_numbers_passes_and_preserves_fired_and_held() {
        let fired = Patch::new(0..5, "uses", RuleId::new("sub.apply"), Tier::Fix);
        let held = Finding::new(
            RuleId::new("span.delete"),
            6..9,
            "held: reason",
            Tier::Suggest,
        );
        let report = EditReport {
            passes: vec![pass(vec![fired], vec![held])],
        };
        let rows = pass_rows(&report);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].pass, 1);
        assert_eq!(rows[0].fired.len(), 1);
        assert_eq!(rows[0].fired[0].rule, "sub.apply");
        assert_eq!(rows[0].fired[0].replacement, "uses");
        assert_eq!(rows[0].held.len(), 1);
        assert_eq!(rows[0].held[0].reason, "held: reason");
    }

    /// A zero-patch pass produces an empty `fired` list.
    #[test]
    fn pass_rows_handles_a_converged_zero_patch_pass() {
        let report = EditReport {
            passes: vec![pass(Vec::new(), Vec::new())],
        };
        let rows = pass_rows(&report);
        assert!(rows[0].fired.is_empty());
        assert!(rows[0].held.is_empty());
    }
}
