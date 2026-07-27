//! `friction check`: parse + metrics + fix-time detection (DMS, literal
//! tell inventory, licensed light-verb constructions, contrast-frame
//! templates, metaphor-compound jargon), with no fixes applied.
//!
//! Prints a per-metric table (value, envelope band, in/out), a per-family
//! DMS summary, and every detected span, in `--format text` (a plain
//! table plus `miette` labeled-span diagnostics — see
//! [`crate::diagnostics`]), `--format json` (stable `serde` structs), or
//! `--format sarif` ([`crate::sarif`]).
//!
//! Exit code: `0` if every banded metric sits inside its envelope and no
//! span was detected; `1` if either is false; `2` on error (see
//! [`CliError::report`]).

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::Args;
use friction_core::MetricVector;
use friction_match::{Channel, DocumentReport, MatchEngine, MatchScore, MatchSpan};
use friction_packs::ModelFamily;
use serde::Serialize;

use crate::common::{
    CliError, Engine, Family, Format, Genre, LineIndex, Pack, display_path, read_input,
    resolve_genre,
};
use crate::diagnostics::{color_enabled, render_spans};
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

    /// Which generator family the DMS channel compares this document
    /// against. Required: DMS is generator-family-specific, and silently
    /// picking one would misreport.
    #[arg(long, value_enum)]
    family: Family,

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

/// One detected span, flattened to a stable, serializable shape (1-based
/// line/column alongside the raw byte range).
#[derive(Debug, Serialize)]
struct SpanRow {
    channel: &'static str,
    frame_id: String,
    start: usize,
    end: usize,
    line: usize,
    column: usize,
    /// The DMS differential score (`sum(d)` over the run), when this
    /// span's channel is [`Channel::Dms`] — `None` for `Literal`/`Lvc`/
    /// `Frame`/`Jargon` spans, whose channels report presence only.
    score: Option<i64>,
}

/// One generator family's document-level DMS statistics.
#[derive(Debug, Serialize)]
struct DmsFamilyRow {
    family: &'static str,
    mean_machine: f64,
    mean_human: f64,
    differential: f64,
    token_count: usize,
}

/// The full `--format json` shape.
#[derive(Debug, Serialize)]
struct CheckReport {
    genre: &'static str,
    family: &'static str,
    metrics: Vec<MetricRow>,
    spans: Vec<SpanRow>,
    tell_counts: BTreeMap<String, usize>,
    dms: Vec<DmsFamilyRow>,
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
    let family: ModelFamily = args.family.into();
    let pack = Pack::load(args.pack.as_deref())?;
    let engine = Engine::load()?;

    let document = friction_parse::parse(source.clone())?;
    let metrics = friction_metrics::compute(&document, &engine.segmenter, &engine.tagger);

    let match_engine = MatchEngine::new(
        &friction_packs::INVENTORY.pack,
        &friction_packs::DMS.pack,
        &friction_packs::JARGON.pack,
        &friction_packs::JARGON_ATTEST,
        family,
        &engine.tagger,
        &engine.segmenter,
    )?;
    let report = match_engine.scan(&document)?;

    let rows = metric_rows(&metrics, pack.as_pack(), genre.as_str());
    let all_in_envelope = rows.iter().all(|row| row.in_envelope.unwrap_or(true));
    let path_label = display_path(&args.input);

    match args.format {
        Format::Text => {
            print!("{}", table::render_metric_table(&rows_for_table(&rows)));
            println!();
            print!("{}", render_dms_summary(&report));
            println!();
            print!("{}", render_tell_counts(&report.spans));
            let color = color_enabled(args.no_color);
            let rendered = render_spans(&source, path_label, &report.spans, color);
            print!("{rendered}");
        }
        Format::Json => {
            let check_report = CheckReport {
                genre: genre.as_str(),
                family: family.as_str(),
                metrics: rows,
                spans: span_rows(&source, &report.spans),
                tell_counts: tell_counts(&report.spans),
                dms: dms_rows(&report),
            };
            let json = serde_json::to_string_pretty(&check_report)
                .expect("CheckReport serializes: every field is plain data");
            println!("{json}");
        }
        Format::Sarif => {
            let json = sarif::render(&report.spans, &source, path_label);
            println!("{json}");
        }
    }

    let exit_ok = all_in_envelope && report.spans.is_empty();
    Ok(if exit_ok {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    })
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

const fn channel_str(channel: Channel) -> &'static str {
    match channel {
        Channel::Dms => "dms",
        Channel::Literal => "literal",
        Channel::Lvc => "lvc",
        Channel::Frame => "frame",
        Channel::Jargon => "jargon",
    }
}

const fn span_score(span: &MatchSpan) -> Option<i64> {
    match span.score {
        MatchScore::Differential(d) => Some(d),
        MatchScore::Present => None,
    }
}

fn span_rows(source: &str, spans: &[MatchSpan]) -> Vec<SpanRow> {
    let lines = LineIndex::new(source);
    spans
        .iter()
        .map(|span| {
            let (line, column) = lines.line_col(source, span.range.start);
            SpanRow {
                channel: channel_str(span.channel),
                frame_id: span.frame_id.to_string(),
                start: span.range.start,
                end: span.range.end,
                line,
                column,
                score: span_score(span),
            }
        })
        .collect()
}

/// This span's `frame_id`, up to (not including) its first `.` — the
/// grouping key [`tell_counts`] and [`render_tell_counts`] use. Every
/// channel's frame id is namespaced this way (`"dms.<family>"`,
/// `"lvc.<nominalization>"`, and the inventory pack's own dotted entry
/// ids for `Literal`), so this collapses each span to the pack family/
/// entry-kind that produced it.
fn frame_prefix(frame_id: &str) -> &str {
    frame_id.split('.').next().unwrap_or(frame_id)
}

fn tell_counts(spans: &[MatchSpan]) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for span in spans {
        *counts
            .entry(frame_prefix(&span.frame_id).to_string())
            .or_insert(0) += 1;
    }
    counts
}

fn dms_rows(report: &DocumentReport) -> Vec<DmsFamilyRow> {
    report
        .dms
        .families
        .iter()
        .map(|family| DmsFamilyRow {
            family: family.family.as_str(),
            mean_machine: family.mean_machine,
            mean_human: family.mean_human,
            differential: family.differential,
            token_count: family.token_count,
        })
        .collect()
}

fn render_dms_summary(report: &DocumentReport) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    let _ = writeln!(out, "dms (target family: {}):", report.dms.target_family);
    for family in &report.dms.families {
        let _ = writeln!(
            out,
            "  {:<8} mean_machine={:.4}  mean_human={:.4}  differential={:.4}  tokens={}",
            family.family.as_str(),
            family.mean_machine,
            family.mean_human,
            family.differential,
            family.token_count,
        );
    }
    out
}

fn render_tell_counts(spans: &[MatchSpan]) -> String {
    use std::fmt::Write as _;
    let counts = tell_counts(spans);
    let mut out = String::new();
    let _ = writeln!(out, "tell counts ({} span(s) total):", spans.len());
    for (prefix, count) in &counts {
        let _ = writeln!(out, "  {prefix}: {count}");
    }
    out
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

    /// `frame_prefix` collapses a dotted frame id to its first segment,
    /// and leaves an unprefixed one alone.
    #[test]
    fn frame_prefix_splits_on_first_dot() {
        assert_eq!(frame_prefix("dms.qwen"), "dms");
        assert_eq!(frame_prefix("lvc.decision"), "lvc");
        assert_eq!(frame_prefix("span.simply"), "span");
        assert_eq!(frame_prefix("noprefix"), "noprefix");
    }

    /// `tell_counts` groups spans by [`frame_prefix`] and counts them.
    #[test]
    fn tell_counts_groups_by_frame_prefix() {
        let spans = vec![
            MatchSpan {
                range: 0..5,
                channel: Channel::Dms,
                frame_id: "dms.qwen".into(),
                score: MatchScore::Differential(3),
            },
            MatchSpan {
                range: 6..10,
                channel: Channel::Dms,
                frame_id: "dms.gemma".into(),
                score: MatchScore::Differential(1),
            },
            MatchSpan {
                range: 11..15,
                channel: Channel::Lvc,
                frame_id: "lvc.decision".into(),
                score: MatchScore::Present,
            },
        ];
        let counts = tell_counts(&spans);
        assert_eq!(counts["dms"], 2);
        assert_eq!(counts["lvc"], 1);
    }
}
