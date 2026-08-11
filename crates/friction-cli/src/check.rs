//! `friction check`: parse + metrics + fix-time detection (DMS, literal
//! tell inventory, licensed light-verb constructions, contrast-frame
//! templates, metaphor-compound jargon, per-document word overuse), with
//! no fixes applied.
//!
//! Prints a per-metric table (value, envelope band, in/out), the pooled
//! DMS machine-vs-human summary, and every detected span, in
//! `--format text` (a plain table plus `miette` labeled-span diagnostics
//! — see [`crate::diagnostics`]), `--format json` (stable `serde`
//! structs), or `--format sarif` ([`crate::sarif`]).
//!
//! Exit code: `0` if every banded metric sits inside its envelope and no
//! span was detected; `1` if either is false; `2` on error (see
//! [`CliError::report`]).

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::Args;
use friction_core::{Lang, MetricVector};
use friction_match::{Channel, DocumentReport, MatchEngine, MatchScore, MatchSpan};
use serde::Serialize;

use crate::common::{
    CliError, Engine, Format, LineIndex, Pack, display_path, parse_lang, read_input,
};
use crate::diagnostics::{color_enabled, render_spans};
use crate::{sarif, table};

/// Arguments for `friction check`.
#[derive(Debug, Args)]
pub struct CheckArgs {
    /// File to check, or `-` to read from stdin.
    input: String,

    /// Language to analyze the document as (BCP-47 primary tag). `en` is
    /// the only supported language in v1.
    #[arg(long, default_value = "en", value_parser = parse_lang)]
    lang: Lang,

    /// Override the embedded envelope pack with one loaded from `PATH`.
    #[arg(long, value_name = "PATH")]
    pack: Option<PathBuf>,

    /// Output format.
    #[arg(long, value_enum, default_value_t = Format::Text)]
    format: Format,

    /// After the report, list every DMS-flagged span that no compiled
    /// frame rule covers: the work queue for the next rule-generation
    /// batch (a span the statistical channel flags but no adjudicated
    /// rule can explain or rewrite).
    #[arg(long)]
    residual: bool,

    /// Disable `--format text`'s ANSI color, regardless of whether stdout
    /// is a terminal. Implied by the `NO_COLOR` environment variable; see
    /// `crate::diagnostics` for the full auto-detection policy.
    #[arg(long)]
    no_color: bool,
}

/// One metric's value, its cross-genre union band (if the pack has
/// one), whether the value falls inside that band, and whether this
/// pack's own holdout measurement found the metric discriminative
/// anywhere at all ([`weak_signal`]).
#[derive(Debug, Serialize)]
struct MetricRow {
    name: &'static str,
    value: f64,
    lo: Option<f64>,
    hi: Option<f64>,
    in_envelope: Option<bool>,
    /// `true` when no genre in the envelope pack counts this metric
    /// toward its own combined out-of-envelope score
    /// ([`friction_packs::EnvelopePack::included_anywhere`]): this pack's
    /// own train-split holdout measured the metric at AUC ~0.5 (no better
    /// than chance at separating human from machine text) in every genre
    /// it was checked in. `value`/`lo`/`hi`/`in_envelope` are still
    /// reported (there may still be a band to show), but rendering an
    /// authoritative-looking in/out verdict for a non-signal would imply
    /// evidence the holdout itself refutes — see this row's own `STATUS`
    /// cell in `--format text` (rendered `weak` instead of `in`/`OUT`;
    /// see `table::render_metric_table`'s own docs).
    weak_signal: bool,
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
    /// `Frame`/`Jargon`/`Overuse` spans, whose channels report presence
    /// only.
    score: Option<i64>,
    /// This span's own [`MatchSpan::message`], when its channel supplies
    /// one (`dms.machine` and `overuse.word` today) — `None` for a
    /// channel whose spans all share a generic, frame-id-derived message
    /// instead.
    #[serde(skip_serializing_if = "Option::is_none")]
    message: Option<String>,
}

/// The pooled DMS machine-vs-human document-level statistics — no family
/// breakdown (see `friction_match::dms`'s own module docs for why).
#[derive(Debug, Serialize)]
struct DmsMachineRow {
    mean_machine: f64,
    mean_human: f64,
    differential: f64,
    token_count: usize,
}

/// The full `--format json` shape.
#[derive(Debug, Serialize)]
struct CheckReport {
    metrics: Vec<MetricRow>,
    spans: Vec<SpanRow>,
    tell_counts: BTreeMap<String, usize>,
    dms: DmsMachineRow,
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
    let pack = Pack::load(args.pack.as_deref())?;
    let engine = Engine::load(args.lang)?;
    let packs = friction_packs::PackSet::for_lang(args.lang);

    let syntax = crate::common::syntax_of(&args.input, &source);
    let document = friction_parse::parse_with(source.clone(), syntax)?;
    let metrics = friction_metrics::compute(&document, &engine.segmenter, &engine.tagger);

    let match_engine = MatchEngine::new(
        &packs.inventory.pack,
        &packs.dms.pack,
        &packs.jargon.pack,
        packs.jargon_attest,
        packs.human_evidence,
        &engine.tagger,
        &engine.segmenter,
    )?;
    let report = match_engine.scan(&document)?;

    let rows = metric_rows(&metrics, pack.as_pack());
    let all_in_envelope = all_in_envelope(&rows);
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
            if args.residual {
                print!(
                    "{}",
                    render_residual(&residual_spans(
                        &source,
                        &document,
                        &report.spans,
                        &engine,
                        packs
                    ))
                );
            }
        }
        Format::Json => {
            let check_report = CheckReport {
                metrics: rows,
                spans: span_rows(&source, &report.spans),
                tell_counts: tell_counts(&report.spans),
                dms: dms_row(&report),
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

fn metric_rows(metrics: &MetricVector, pack: &friction_packs::EnvelopePack) -> Vec<MetricRow> {
    metrics
        .named_values()
        .into_iter()
        .map(|(name, value)| {
            let band = pack.band_union(name);
            MetricRow {
                name,
                value,
                lo: band.map(|b| b.lo),
                hi: band.map(|b| b.hi),
                in_envelope: band.map(|b| b.contains(value)),
                weak_signal: !pack.included_anywhere(name),
            }
        })
        .collect()
}

/// Whether every row's value sits inside its band — the metric half of
/// `check`'s exit-code decision (see `run_inner`'s own `exit_ok`).
///
/// A `weak_signal` row is treated as passing regardless of its own
/// `in_envelope` value: this pack's own holdout found the metric no
/// better than chance at separating human from machine text in every
/// genre it was checked in, so a value outside its band is not evidence
/// of anything a nonzero exit code should report. A row with no band at
/// all (`in_envelope: None`) already passes via `unwrap_or(true)`,
/// unchanged from before `weak_signal` existed.
fn all_in_envelope(rows: &[MetricRow]) -> bool {
    rows.iter()
        .all(|row| row.weak_signal || row.in_envelope.unwrap_or(true))
}

fn rows_for_table(rows: &[MetricRow]) -> Vec<table::MetricTableRow<'_>> {
    rows.iter()
        .map(|row| table::MetricTableRow {
            name: row.name,
            value: row.value,
            lo: row.lo,
            hi: row.hi,
            in_envelope: row.in_envelope,
            weak_signal: row.weak_signal,
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
        Channel::Overuse => "overuse",
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
                message: span.message.as_deref().map(str::to_string),
            }
        })
        .collect()
}

/// This span's `frame_id`, up to (not including) its first `.` — the
/// grouping key [`tell_counts`] and [`render_tell_counts`] use. Every
/// channel's frame id is namespaced this way (the constant `"dms.machine"`,
/// `"lvc.<nominalization>"`, and the inventory pack's own dotted entry
/// ids for `Literal`), so this collapses each span to the pack channel/
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

const fn dms_row(report: &DocumentReport) -> DmsMachineRow {
    let machine = &report.dms.machine;
    DmsMachineRow {
        mean_machine: machine.mean_machine,
        mean_human: machine.mean_human,
        differential: machine.differential,
        token_count: machine.token_count,
    }
}

fn render_dms_summary(report: &DocumentReport) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    let machine = &report.dms.machine;
    let _ = writeln!(
        out,
        "dms: mean_machine={:.4}  mean_human={:.4}  differential={:.4}  tokens={}",
        machine.mean_machine, machine.mean_human, machine.differential, machine.token_count,
    );
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

/// One DMS-flagged span no compiled frame rule covers.
struct ResidualSpan<'a> {
    range: std::ops::Range<usize>,
    frame_id: &'a str,
    text: &'a str,
}

/// The DMS spans of `spans` that overlap no frame-rule match anywhere
/// in the document: the statistical channel sees a machine tell
/// there, but no adjudicated rule can explain or rewrite it, so it is
/// exactly the evidence queue the next rule batch should be generated
/// from.
fn residual_spans<'a>(
    source: &'a str,
    document: &friction_core::Document,
    spans: &'a [MatchSpan],
    engine: &Engine,
    packs: &friction_packs::PackSet,
) -> Vec<ResidualSpan<'a>> {
    let units = friction_match::token::prose_scope(document, &engine.segmenter);
    let tagged = friction_match::tagging::tag_units(&units, source, &engine.tagger);
    let view = &packs.frame.pack;
    let index = friction_match::frame_rewrite::FrameIndex::build(view);
    let mut frame_ranges: Vec<std::ops::Range<usize>> = Vec::new();
    for sentence in &tagged {
        for m in
            friction_match::frame_rewrite::scan_sentence(view, &index, &sentence.tokens, source)
        {
            frame_ranges.push(m.bytes);
        }
    }
    spans
        .iter()
        .filter(|span| span.channel == Channel::Dms)
        .filter(|span| {
            !frame_ranges
                .iter()
                .any(|f| span.range.start < f.end && f.start < span.range.end)
        })
        .map(|span| ResidualSpan {
            range: span.range.clone(),
            frame_id: &span.frame_id,
            text: source.get(span.range.clone()).unwrap_or_default(),
        })
        .collect()
}

/// Renders the `--residual` section.
fn render_residual(residual: &[ResidualSpan<'_>]) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    let _ = writeln!(
        out,
        "\nresidual: {} DMS span(s) covered by no frame rule",
        residual.len()
    );
    for span in residual {
        let _ = writeln!(
            out,
            "  {}..{} [{}] {:?}",
            span.range.start, span.range.end, span.frame_id, span.text
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `metric_rows` looks up a band for every metric the pack has one
    /// for, and reports `None` for the rest: hand-checked against a tiny
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
        let rows = metric_rows(&metrics, &pack);

        let triad = rows
            .iter()
            .find(|r| r.name == "triad_rate")
            .expect("triad_rate row exists");
        assert_eq!(triad.lo, Some(0.0));
        assert_eq!(triad.hi, Some(0.5));
        assert_eq!(triad.in_envelope, Some(true));
        assert!(
            !triad.weak_signal,
            "triad_rate has a band with the include default (true), so it is not weak"
        );

        let em_dash = rows
            .iter()
            .find(|r| r.name == "em_dash_density")
            .expect("em_dash_density row exists");
        assert_eq!(em_dash.lo, None);
        assert_eq!(em_dash.in_envelope, None);
        assert!(
            em_dash.weak_signal,
            "em_dash_density has no band anywhere in this pack, so it is weak too"
        );
    }

    /// A metric every genre with a band marks `include: false` is
    /// `weak_signal` even though it still has a band (still shown, just
    /// not judged) — the AUC-~0.5-everywhere case.
    #[test]
    fn metric_rows_marks_a_metric_excluded_by_every_genre_as_weak_signal() {
        let pack = friction_packs::EnvelopePack::parse(
            "[blog.bullet_parallelism]\nlo = 0.0\nhi = 1.0\ninclude = false\n\
             [docs.bullet_parallelism]\nlo = 0.0\nhi = 1.0\ninclude = false\n",
        )
        .expect("pack parses");
        let metrics = MetricVector::default();
        let rows = metric_rows(&metrics, &pack);

        let row = rows
            .iter()
            .find(|r| r.name == "bullet_parallelism")
            .expect("bullet_parallelism row exists");
        assert!(row.weak_signal);
        assert!(row.lo.is_some(), "the band itself is still reported");
    }

    fn row(name: &'static str, in_envelope: Option<bool>, weak_signal: bool) -> MetricRow {
        MetricRow {
            name,
            value: 0.0,
            lo: Some(0.0),
            hi: Some(1.0),
            in_envelope,
            weak_signal,
        }
    }

    /// A row that's out of its band but `weak_signal` never fails
    /// `all_in_envelope` — the exit-code exclusion `MetricRow::weak_signal`
    /// exists for.
    #[test]
    fn all_in_envelope_ignores_out_of_band_weak_signal_rows() {
        let rows = vec![
            row("triad_rate", Some(true), false),
            row("bullet_parallelism", Some(false), true),
        ];
        assert!(
            all_in_envelope(&rows),
            "an out-of-band weak-signal row must not fail the check"
        );
    }

    /// A genuinely out-of-band, non-weak row still fails
    /// `all_in_envelope` — the exclusion is scoped to `weak_signal` rows
    /// only, not a blanket relaxation.
    #[test]
    fn all_in_envelope_still_fails_on_a_real_out_of_band_row() {
        let rows = vec![
            row("triad_rate", Some(true), false),
            row("em_dash_density", Some(false), false),
        ];
        assert!(!all_in_envelope(&rows));
    }

    /// An unbanded row (`in_envelope: None`) still passes regardless of
    /// `weak_signal` — unchanged from before `weak_signal` existed.
    #[test]
    fn all_in_envelope_passes_unbanded_rows() {
        let rows = vec![row("unbanded_metric", None, false)];
        assert!(all_in_envelope(&rows));
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
                message: None,
            },
            MatchSpan {
                range: 6..10,
                channel: Channel::Dms,
                frame_id: "dms.gemma".into(),
                score: MatchScore::Differential(1),
                message: None,
            },
            MatchSpan {
                range: 11..15,
                channel: Channel::Lvc,
                frame_id: "lvc.decision".into(),
                score: MatchScore::Present,
                message: None,
            },
        ];
        let counts = tell_counts(&spans);
        assert_eq!(counts["dms"], 2);
        assert_eq!(counts["lvc"], 1);
    }
}
