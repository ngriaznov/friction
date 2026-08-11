//! `friction explain`: runs the five-operation repair engine internally
//! (exactly like `friction fix`) but never prints the fixed text. Instead
//! it prints, per pass, every operation that actually fired (rule, range,
//! and, for a substitution/pivot, what it changed to) and every candidate
//! a gate held back, followed by a short convergence summary.
//!
//! No `--genre`/`--pack`: same reasoning as `friction fix` — the engine's
//! five operations are never gated by a metric or genre envelope.
//!
//! # Report-only bookkeeping, summarized by default (`--verbose`)
//!
//! Most held candidates on a real document are frame-pack `Report`-kind
//! rows (`vguard.`/`adjg.`/`intg.`/`conng.`/`phg.`/`pilot.` and friends):
//! bookkeeping that documents a measured unigram rate, never a candidate
//! edit a gate actually held back (see
//! `friction_edit::sentence::generate_sentence`'s own docs for where
//! these findings come from). On a real document there can be dozens of
//! them per pass, per sentence — thousands total — drowning the gate
//! holds and quotation holds a human actually needs to see in `--format
//! text`'s default output. [`is_report_only`] identifies a report-only
//! finding by its own message's `"report-only"` marker (both of
//! `generate_sentence`'s report-rule message formats carry it) rather
//! than a hardcoded rule-id prefix list, so a future report-kind rule
//! family needs no update here. By default, `--format text` collapses
//! them to one summary line per pass; `--verbose` lists every one,
//! preserving the exact per-line format `--format text` has always used.
//! Real holds (gate holds, quotation holds — anything whose message
//! doesn't carry the marker) are unaffected either way: always listed,
//! in the same format as before. `--format json`'s `held` array is
//! unaffected by `--verbose` — a JSON consumer already gets a structured
//! `reason` per row and can filter for the marker itself; flooding a
//! terminal and flooding a parser are not the same problem.

use std::process::ExitCode;

use clap::Args;
use friction_core::{Finding, Lang, Patch};
use friction_edit::EditReport;
use serde::Serialize;

use crate::common::{CliError, Format, parse_lang, read_input};

/// Arguments for `friction explain`.
#[derive(Debug, Args)]
pub struct ExplainArgs {
    /// File to explain, or `-` to read from stdin.
    input: String,

    /// Language to analyze the document as (BCP-47 primary tag). `en` is
    /// the only supported language in v1.
    #[arg(long, default_value = "en", value_parser = parse_lang)]
    lang: Lang,

    /// Output format. `sarif` is not supported here (see `friction
    /// check`).
    #[arg(long, value_enum, default_value_t = Format::Text)]
    format: Format,

    /// List every report-only frame-guard observation individually
    /// (`--format text` only), instead of collapsing them to one summary
    /// line per pass. See this module's own docs for what "report-only"
    /// means here and why it floods the default output.
    #[arg(long)]
    verbose: bool,
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
    let syntax = crate::common::syntax_of(&args.input, &source);
    let engine = friction_edit::Engine::for_lang(args.lang)?;
    let (_output, report) = engine.fix_document_with(&source, syntax)?;

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
            print!("{}", render_text(&report, args.verbose));
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

/// `true` when `finding.message` carries the report-only bookkeeping
/// marker every frame-pack `Report`-kind guard finding shares (see this
/// module's own docs) — identified by the message's own `"report-only"`
/// substring, not a hardcoded rule-id prefix list, so a future
/// report-kind rule family needs no update here.
fn is_report_only(finding: &Finding) -> bool {
    finding.message.contains("report-only")
}

/// Renders `report` as `--format text`'s output: per pass, every fired
/// operation, then every held candidate — real holds always listed,
/// report-only observations summarized to one line unless `verbose`
/// (see this module's own docs).
fn render_text(report: &EditReport, verbose: bool) -> String {
    use std::fmt::Write as _;

    let mut out = String::new();
    for (i, pass) in report.passes.iter().enumerate() {
        let pass_num = i + 1;
        if pass.patches_applied == 0 {
            let _ = writeln!(out, "pass {pass_num}: 0 operation(s) fired — converged");
        } else {
            let _ = writeln!(
                out,
                "pass {pass_num}: {} operation(s) fired",
                pass.patches_applied
            );
            for patch in &pass.applied_patches {
                if patch.replacement.is_empty() {
                    let _ = writeln!(
                        out,
                        "  {:<24} {}..{}  (deleted)",
                        patch.rule.as_str(),
                        patch.range.start,
                        patch.range.end
                    );
                } else {
                    let _ = writeln!(
                        out,
                        "  {:<24} {}..{}  -> {:?}",
                        patch.rule.as_str(),
                        patch.range.start,
                        patch.range.end,
                        patch.replacement
                    );
                }
            }
        }

        let (report_only, real_held): (Vec<&Finding>, Vec<&Finding>) =
            pass.held.iter().partition(|f| is_report_only(f));

        if !real_held.is_empty() {
            let _ = writeln!(out, "  {} held:", real_held.len());
            for finding in &real_held {
                write_held_line(&mut out, finding);
            }
        }

        if !report_only.is_empty() {
            if verbose {
                let _ = writeln!(out, "  {} report-only observation(s):", report_only.len());
                for finding in &report_only {
                    write_held_line(&mut out, finding);
                }
            } else {
                let _ = writeln!(
                    out,
                    "  {} report-only observation(s) (--verbose to list)",
                    report_only.len()
                );
            }
        }
    }
    out
}

/// One held-candidate line, in the exact format `--format text` has
/// always used — shared by the real-held and (`--verbose` only)
/// report-only listings so a listed row looks identical either way.
fn write_held_line(out: &mut String, finding: &Finding) {
    use std::fmt::Write as _;
    let _ = writeln!(
        out,
        "    {:<22} {}..{}  KEPT ({})",
        finding.rule.as_str(),
        finding.range.start,
        finding.range.end,
        finding.message
    );
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
            reusable_scan: None,
            final_bounded_pass_index: 0,
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
            reusable_scan: None,
            final_bounded_pass_index: 0,
        };
        let rows = pass_rows(&report);
        assert!(rows[0].fired.is_empty());
        assert!(rows[0].held.is_empty());
    }

    fn report_only_finding(rule: &'static str, range: std::ops::Range<usize>) -> Finding {
        Finding::new(
            RuleId::new(rule),
            range,
            format!("frame {rule}: measured 10.0/M machine vs 1.0/M human; report-only"),
            Tier::Suggest,
        )
    }

    fn real_finding(rule: &'static str, range: std::ops::Range<usize>) -> Finding {
        Finding::new(
            RuleId::new(rule),
            range,
            "held: a real reason",
            Tier::Suggest,
        )
    }

    /// `is_report_only` matches both of `generate_sentence`'s own
    /// report-rule message shapes (the rate-carrying one and the
    /// pilot-bucket one) and rejects an ordinary held-candidate message.
    #[test]
    fn is_report_only_matches_the_message_marker_not_the_rule_id() {
        let rate = Finding::new(
            RuleId::new("vguard.require"),
            0..5,
            "frame vguard.require: measured 100.0/M machine vs 10.0/M human; report-only",
            Tier::Suggest,
        );
        let pilot = Finding::new(
            RuleId::new("pilot.such-as"),
            0..5,
            "frame pilot.such-as: report-only (see the frame-pack compile report)",
            Tier::Suggest,
        );
        let real = Finding::new(
            RuleId::new("register.unpack"),
            0..5,
            "nominalization: \"venture\" has no licensed rewrite — needs a human hand",
            Tier::Suggest,
        );
        assert!(is_report_only(&rate));
        assert!(is_report_only(&pilot));
        assert!(!is_report_only(&real));
    }

    /// By default (`verbose: false`), a pass with only report-only holds
    /// prints one summary line instead of a per-finding `KEPT` listing,
    /// and never prints a `N held:` header at all (there are zero real
    /// holds to introduce).
    #[test]
    fn render_text_default_summarizes_report_only_holds_to_one_line() {
        let held = vec![
            report_only_finding("vguard.require", 0..5),
            report_only_finding("adjg.robust", 6..12),
        ];
        let report = EditReport {
            passes: vec![pass(Vec::new(), held)],
            reusable_scan: None,
            final_bounded_pass_index: 0,
        };
        let text = render_text(&report, false);
        assert!(
            text.contains("2 report-only observation(s) (--verbose to list)"),
            "got: {text}"
        );
        assert!(!text.contains("KEPT"), "got: {text}");
        assert!(!text.contains("held:"), "got: {text}");
    }

    /// `--verbose` lists every report-only finding individually, in the
    /// exact `KEPT (...)` per-line format `--format text` has always
    /// used for held candidates.
    #[test]
    fn render_text_verbose_lists_every_report_only_finding() {
        let held = vec![
            report_only_finding("vguard.require", 0..5),
            report_only_finding("adjg.robust", 6..12),
        ];
        let report = EditReport {
            passes: vec![pass(Vec::new(), held)],
            reusable_scan: None,
            final_bounded_pass_index: 0,
        };
        let text = render_text(&report, true);
        assert!(
            text.contains("2 report-only observation(s):"),
            "got: {text}"
        );
        assert!(text.contains("vguard.require"));
        assert!(text.contains("adjg.robust"));
        assert!(text.contains("KEPT ("));
        assert!(!text.contains("--verbose to list"), "got: {text}");
    }

    /// A real hold (a gate hold, a quotation hold — anything whose
    /// message carries no report-only marker) is always listed in full,
    /// under its own `N held:` header, regardless of `verbose` — the
    /// summarization applies only to report-only bookkeeping.
    #[test]
    fn render_text_always_lists_real_holds_regardless_of_verbose() {
        let held = vec![real_finding("span.delete", 0..5)];
        let report = EditReport {
            passes: vec![pass(Vec::new(), held)],
            reusable_scan: None,
            final_bounded_pass_index: 0,
        };
        for verbose in [false, true] {
            let text = render_text(&report, verbose);
            assert!(text.contains("1 held:"), "got: {text}");
            assert!(text.contains("span.delete"), "got: {text}");
            assert!(text.contains("KEPT (held: a real reason)"), "got: {text}");
            assert!(!text.contains("report-only"), "got: {text}");
        }
    }

    /// A pass mixing real holds and report-only holds separates the two:
    /// the real hold is listed under `N held:`, the report-only holds
    /// are summarized (or, with `--verbose`, listed under their own
    /// header) — never merged into one count.
    #[test]
    fn render_text_separates_real_holds_from_report_only_holds() {
        let held = vec![
            real_finding("span.delete", 0..5),
            report_only_finding("vguard.require", 6..12),
            report_only_finding("adjg.robust", 13..20),
        ];
        let report = EditReport {
            passes: vec![pass(Vec::new(), held)],
            reusable_scan: None,
            final_bounded_pass_index: 0,
        };

        let default_text = render_text(&report, false);
        assert!(default_text.contains("1 held:"), "got: {default_text}");
        assert!(
            default_text.contains("2 report-only observation(s) (--verbose to list)"),
            "got: {default_text}"
        );

        let verbose_text = render_text(&report, true);
        assert!(verbose_text.contains("1 held:"), "got: {verbose_text}");
        assert!(
            verbose_text.contains("2 report-only observation(s):"),
            "got: {verbose_text}"
        );
    }

    /// A pass with zero held candidates at all prints neither a `held:`
    /// header nor a report-only summary line.
    #[test]
    fn render_text_prints_nothing_extra_for_a_pass_with_no_holds() {
        let report = EditReport {
            passes: vec![pass(Vec::new(), Vec::new())],
            reusable_scan: None,
            final_bounded_pass_index: 0,
        };
        let text = render_text(&report, false);
        assert!(!text.contains("held"), "got: {text}");
        assert!(!text.contains("report-only"), "got: {text}");
    }
}
