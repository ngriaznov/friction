//! Renders [`friction_match::MatchSpan`]s as `miette` labeled-span
//! diagnostics for `friction check`'s `--format text` output.
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
use std::sync::Arc;

use friction_core::Tier;
use friction_match::MatchSpan;
use miette::{
    Diagnostic, GraphicalReportHandler, GraphicalTheme, LabeledSpan, NamedSource, SourceCode,
};

use crate::fast_source::LineIndexedSource;

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

/// One detection span (or, historically, `Finding`), adapted to `miette`'s
/// [`Diagnostic`] trait: a labeled span over its rule/frame id (as the
/// diagnostic's `code`), its message (both as the top-level error text and
/// as the span's own label), and a severity derived from a [`Tier`] via
/// [`severity_for`] — `Advice` is deliberately *not* used here despite the
/// name collision with `Tier`. `crate::sarif::LEVEL` uses the same
/// severity for every span, so the two output formats never disagree.
struct FindingDiagnostic {
    message: String,
    rule: String,
    tier: Tier,
    start: usize,
    len: usize,
    /// Shared, not owned: `render_spans` builds one of these for the
    /// whole document and hands every diagnostic a clone of it (an `Arc`
    /// bump, not a copy — [`LineIndexedSource`] sits behind the `Arc` so
    /// cloning it per span never re-copies the source or its line-start
    /// index). [`LineIndexedSource`] (not a plain `Arc<str>`) is what
    /// keeps each `render_report` call cheap regardless of document size
    /// — see its own module docs.
    src: NamedSource<Arc<LineIndexedSource>>,
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

/// Renders every detection span in `spans` as a `miette` labeled-span
/// diagnostic against `source`, the same rendering [`render_findings`]
/// gives a [`Finding`] — [`friction_match::MatchSpan`]'s `frame_id` is a
/// runtime string rather than a `friction_core::RuleId` (which requires a
/// `'static` string), so this is a small, parallel entry point rather than
/// a conversion into [`Finding`].
///
/// Every span is rendered at [`miette::Severity::Warning`] (the same
/// severity [`severity_for`] gives a `Tier::Fix` finding): a span is a
/// candidate `friction fix` can act on, not necessarily something that
/// needs a human's judgment on its own — the gates inside the repair
/// engine are what decide, per candidate, whether it actually applies or
/// gets held.
#[must_use]
pub fn render_spans(source: &str, path_label: &str, spans: &[MatchSpan], color: bool) -> String {
    let theme = if color {
        GraphicalTheme::unicode()
    } else {
        GraphicalTheme::unicode_nocolor()
    };
    let handler = GraphicalReportHandler::new_themed(theme).with_width(200);
    let named = NamedSource::new(
        path_label,
        Arc::new(LineIndexedSource::new(Arc::<str>::from(source))),
    );
    let mut out = String::new();
    for span in spans {
        let message = format!("{:?} tell: {}", span.channel, span.frame_id);
        let diagnostic = FindingDiagnostic {
            message: message.clone(),
            rule: span.frame_id.to_string(),
            tier: Tier::Fix,
            start: span.range.start,
            len: span.range.len(),
            src: named.clone(),
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
    use friction_match::{Channel, MatchScore};

    use super::*;

    fn span(range: std::ops::Range<usize>, frame_id: &str) -> MatchSpan {
        MatchSpan {
            range,
            channel: Channel::Literal,
            frame_id: frame_id.into(),
            score: MatchScore::Present,
        }
    }

    /// Rendering with color disabled never emits an ANSI escape byte —
    /// the byte-stability guarantee `check --format text` (as piped by
    /// every test in this crate) depends on.
    #[test]
    fn render_spans_without_color_emits_no_ansi_escapes() {
        let source = "Moreover, it works.";
        let spans = vec![span(0..9, "ritual.moreover")];
        let rendered = render_spans(source, "doc.md", &spans, false);
        assert!(!rendered.contains('\u{1b}'), "got: {rendered:?}");
        assert!(rendered.contains("ritual.moreover"));
        assert!(rendered.contains("tell:"));
    }

    /// An empty span list renders to an empty string.
    #[test]
    fn render_spans_empty_list_is_empty_string() {
        assert_eq!(render_spans("text", "doc.md", &[], false), "");
    }

    /// `render_spans` always rates a span at [`miette::Severity::Warning`]
    /// — the same severity `crate::sarif::LEVEL` ("warning") assigns every
    /// span, so the two output formats never disagree.
    #[test]
    fn render_spans_uses_the_same_severity_as_sarif_level() {
        assert_eq!(severity_for(Tier::Fix), miette::Severity::Warning);
        assert_eq!(crate::sarif::LEVEL, "warning");
    }
}
