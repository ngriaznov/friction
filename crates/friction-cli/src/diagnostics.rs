//! Renders [`Finding`]s as `miette` labeled-span diagnostics for `friction
//! check`'s `--format text` output.
//!
//! # Color and determinism
//!
//! `miette`'s fancy renderer can colorize its output with ANSI escapes,
//! which would make two byte-identical runs differ whenever one happens to
//! run under a different `is_terminal`/`NO_COLOR` combination. The policy
//! here is the "auto-detect, but let the caller force it off" the crate
//! brief asks for, implemented as the simplest robust rule: color is
//! enabled only when stdout is an actual terminal *and* neither `NO_COLOR`
//! nor `--no-color` is set (see [`color_enabled`]). Piped output — every
//! test harness, every CI job, every `friction check foo.md | less` — is
//! therefore always byte-stable without needing `--no-color` explicitly;
//! the flag and the environment variable exist for a human at an actual
//! terminal who wants to force plain text anyway (e.g. to paste into
//! something that mangles ANSI codes).

use std::fmt;
use std::io::IsTerminal as _;

use friction_core::{Finding, Tier};
use miette::{
    Diagnostic, GraphicalReportHandler, GraphicalTheme, LabeledSpan, NamedSource, SourceCode,
};

/// Whether `check`'s text-mode diagnostics should be colorized: only when
/// stdout is an actual terminal and neither `--no-color` nor `NO_COLOR`
/// asked for plain text. See the module docs for the full rationale.
#[must_use]
pub fn color_enabled(no_color_flag: bool) -> bool {
    if no_color_flag || std::env::var_os("NO_COLOR").is_some() {
        return false;
    }
    std::io::stdout().is_terminal()
}

/// One [`Finding`], adapted to `miette`'s [`Diagnostic`] trait: a labeled
/// span over the finding's rule id (as the diagnostic's `code`), its
/// message (both as the top-level error text and as the span's own
/// label), and a severity derived from its [`Tier`] via [`severity_for`]
/// (`Fix` findings have an automatic remedy that `friction fix` can apply
/// on its own, so they are downgraded to `Warning`; `Suggest` findings
/// need a human's judgment and are the ones a reader most needs to see, so
/// they keep the higher `Error` severity — `Advice` is deliberately *not*
/// used here despite the name collision with `Tier`). `sarif::level_for`
/// mirrors this exact ordering (`Suggest` outranks `Fix`) in SARIF's
/// `level` vocabulary, so the two output formats never disagree about
/// which findings are more urgent.
struct FindingDiagnostic {
    message: String,
    rule: String,
    tier: Tier,
    start: usize,
    len: usize,
    src: NamedSource<String>,
}

impl fmt::Debug for FindingDiagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FindingDiagnostic")
            .field("rule", &self.rule)
            .field("message", &self.message)
            .finish_non_exhaustive()
    }
}

impl fmt::Display for FindingDiagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for FindingDiagnostic {}

/// `miette`'s [`Severity`](miette::Severity) for one [`Tier`]: `Fix`
/// findings have an automatic remedy, so they are downgraded to
/// `Warning`; `Suggest` findings need a human's judgment and are the ones
/// a reader most needs to see, so they keep the higher `Error` severity.
///
/// This ordering (`Suggest` outranks `Fix`) must agree with
/// [`crate::sarif::level_for`]'s ordering in SARIF's `level` vocabulary —
/// see [`FindingDiagnostic`]'s doc comment, and
/// `severity_ordering_agrees_with_sarif_level_ordering` below, which
/// pins the two orderings together so they cannot drift apart again.
const fn severity_for(tier: Tier) -> miette::Severity {
    match tier {
        Tier::Fix => miette::Severity::Warning,
        Tier::Suggest => miette::Severity::Error,
    }
}

impl Diagnostic for FindingDiagnostic {
    fn code<'a>(&'a self) -> Option<Box<dyn fmt::Display + 'a>> {
        Some(Box::new(self.rule.clone()))
    }

    fn severity(&self) -> Option<miette::Severity> {
        Some(severity_for(self.tier))
    }

    fn source_code(&self) -> Option<&dyn SourceCode> {
        Some(&self.src)
    }

    fn labels(&self) -> Option<Box<dyn Iterator<Item = LabeledSpan> + '_>> {
        Some(Box::new(std::iter::once(LabeledSpan::new(
            Some(self.message.clone()),
            self.start,
            self.len,
        ))))
    }
}

/// Renders every finding in `findings` as a `miette` labeled-span
/// diagnostic against `source`, one after another, separated by a blank
/// line.
///
/// `path_label` names the source in each diagnostic's header (a real
/// path, or `"<stdin>"` — see [`crate::common::display_path`]). `color`
/// should come from [`color_enabled`].
#[must_use]
pub fn render_findings(
    source: &str,
    path_label: &str,
    findings: &[Finding],
    color: bool,
) -> String {
    // A fixed 200-column render width (`GraphicalReportHandler::new`'s
    // own default, kept explicit here) rather than one probed from the
    // real terminal: probing would make the same input produce different
    // byte output depending on the invoking terminal's width, which the
    // determinism requirement (`check --format text`'s output must be
    // byte-identical across runs) rules out.
    let theme = if color {
        GraphicalTheme::unicode()
    } else {
        GraphicalTheme::unicode_nocolor()
    };
    let handler = GraphicalReportHandler::new_themed(theme).with_width(200);
    let mut out = String::new();
    for finding in findings {
        let diagnostic = FindingDiagnostic {
            message: finding.message.clone(),
            rule: finding.rule.as_str().to_string(),
            tier: finding.tier,
            start: finding.range.start,
            len: finding.range.len(),
            src: NamedSource::new(path_label, source.to_string()),
        };
        handler
            .render_report(&mut out, &diagnostic)
            .expect("writing to a String cannot fail");
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use friction_core::RuleId;

    use super::*;

    /// Rendering with color disabled never emits an ANSI escape byte —
    /// the byte-stability guarantee `check --format text` (as piped by
    /// every test in this crate) depends on.
    #[test]
    fn render_findings_without_color_emits_no_ansi_escapes() {
        let source = "Moreover, it works.";
        let findings = vec![Finding::new(
            RuleId::new("connective.surgery"),
            0..9,
            "overused sentence-initial connective",
            Tier::Fix,
        )];
        let rendered = render_findings(source, "doc.md", &findings, false);
        assert!(!rendered.contains('\u{1b}'), "got: {rendered:?}");
        assert!(rendered.contains("connective.surgery"));
        assert!(rendered.contains("overused sentence-initial connective"));
    }

    /// An empty finding list renders to an empty string.
    #[test]
    fn render_findings_empty_list_is_empty_string() {
        assert_eq!(render_findings("text", "doc.md", &[], false), "");
    }

    /// Regression test for a real disagreement between this module's
    /// text/miette severity and `sarif::level_for`'s SARIF `level` for the
    /// identical `Tier`: text output used to rank `Tier::Suggest` as
    /// *more* severe than `Tier::Fix` (`Error` > `Warning`) while SARIF
    /// ranked it as *less* severe (`"note"` < `"warning"`) — a CI gate
    /// keyed on SARIF `level` would then treat exactly the findings the
    /// text renderer flags as most urgent as safely ignorable. Both
    /// formats must rank `Suggest` above `Fix` (or both must rank it
    /// below); this test fails if they ever diverge again.
    #[test]
    fn severity_ordering_agrees_with_sarif_level_ordering() {
        assert!(
            severity_for(Tier::Suggest) > severity_for(Tier::Fix),
            "text/miette severity must rank Suggest above Fix"
        );

        let sarif_rank = |level: &str| -> u8 {
            match level {
                "none" => 0,
                "note" => 1,
                "warning" => 2,
                "error" => 3,
                other => panic!("unknown SARIF level {other:?}"),
            }
        };
        let fix_rank = sarif_rank(crate::sarif::level_for(Tier::Fix));
        let suggest_rank = sarif_rank(crate::sarif::level_for(Tier::Suggest));
        assert!(
            suggest_rank > fix_rank,
            "SARIF level must also rank Suggest above Fix, to agree with text/miette output"
        );
    }
}
