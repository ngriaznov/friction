//! wasm-bindgen bindings: `init_engine`, `fix_text`, `check_text`,
//! `explain_text`, `engine_version` — the fixed export contract a
//! companion web page is written against.
//!
//! # Why this crate re-derives `check`/`explain`'s JSON shapes instead of
//! importing them
//!
//! `friction-cli`'s `check`/`fix`/`explain` command cores
//! (`crates/friction-cli/src/{check,fix,explain}.rs`) are private modules
//! of that crate's `[[bin]]` target only — its `[lib]` target exists
//! solely to expose `offset_to_line_col` to the SARIF fuzz target (see
//! that crate's own `src/lib.rs` doc comment) and does not re-export
//! them. Restructuring `friction-cli` to share those modules was
//! deliberately not done (the byte-fence discipline — the native
//! binary's behavior must stay untouched — argues for not touching it at
//! all), so [`check_text`]/[`explain_text`] below call the same
//! `friction-parse`/`friction-metrics`/`friction-match`/`friction-edit`
//! entry points those modules call, and their small serializable row
//! types are hand-kept field-for-field mirrors of
//! `friction-cli`'s own `CheckReport`/`ExplainReport` (see each type's
//! doc comment for which CLI struct it mirrors). `fix_text` needs no such
//! mirror: it returns exactly the fixed document text
//! `Engine::fix_document_with` already produces, the same bytes `friction
//! fix` prints to stdout without `--suggest`/`--in-place`.
//!
//! # Runtime asset installation
//!
//! This crate's own `Cargo.toml` disables every heavy embedded asset
//! (`friction-nlp`'s `embedded-weights`, `friction-packs`'s
//! `embedded-dms`) across its whole dependency closure (see that file's
//! comments). [`init_engine`] is the seam a host page calls once, with
//! the tagger/parser/DMS-index bytes fetched separately, before any
//! other export here — see its own doc comment.

use std::collections::BTreeMap;
use std::sync::OnceLock;

use friction_core::{Finding, Lang, MetricVector, Patch};
use friction_edit::{EditReport, Engine};
use friction_match::{Channel, DocumentReport, MatchEngine, MatchScore, MatchSpan};
use friction_nlp::SrxSegmenter;
use friction_packs::EnvelopePack;
use serde::Serialize;
use wasm_bindgen::prelude::*;

/// `en` is the only language this workspace supports in v1 — see
/// `friction_core::Lang`'s own docs. Every export below is implicitly
/// scoped to it, the same way `friction-cli`'s `--lang` flag defaults to
/// `en`.
const LANG: Lang = Lang::En;

/// Formats any `Display`-able error as a [`JsError`], the shape every
/// export below reports a failure as.
fn to_js_error(err: impl std::fmt::Display) -> JsError {
    JsError::new(&err.to_string())
}

/// The engine [`init_engine`] builds once and every later
/// [`fix_text`]/[`check_text`]/[`explain_text`] call reuses.
///
/// `PerceptronTagger`/`PerceptronParser::for_lang` re-parse whatever
/// bytes are installed on every call they're not otherwise cached
/// across (see their own docs: "every call re-reads whatever is
/// currently installed") — for the audited `json.gz` interchange format
/// a host page may install, that is a real, repeated cost (full JSON
/// decompression and parsing), not just a formality. Building the engine
/// once here and reusing it is what makes that a one-time cost per page
/// load instead of a per-call one; it changes nothing about which bytes
/// end up feeding any given call, since the install seams are one-shot
/// by contract (a second `install_*` call always errors — see
/// `init_engine`'s own doc comment).
static ENGINE: OnceLock<Engine> = OnceLock::new();

/// Borrows the [`ENGINE`] [`init_engine`] built, or a [`JsError`] naming
/// the missing call if it was never (successfully) called.
fn engine() -> Result<&'static Engine, JsError> {
    ENGINE.get().ok_or_else(|| {
        JsError::new(
            "init_engine must be called (and must succeed) before fix_text/check_text/explain_text",
        )
    })
}

/// The surface syntax to parse `source` as: HTML if it's unambiguously an
/// HTML page opener ([`friction_parse::looks_like_html`]), markdown
/// otherwise. Mirrors `friction-cli`'s own `syntax_of` on its stdin (`-`)
/// path — the one path that, like this crate's callers, has no filename
/// extension to sniff instead.
fn syntax_of(source: &str) -> friction_parse::Syntax {
    if friction_parse::looks_like_html(source) {
        friction_parse::Syntax::Html
    } else {
        friction_parse::Syntax::Markdown
    }
}

/// Line-start offsets for one document, so resolving many byte offsets
/// into 1-based `(line, column)` pairs costs one scan rather than one per
/// offset. A byte-identical copy of `friction-cli`'s own
/// `common::LineIndex` (private to that crate) — see this struct's own
/// use in [`span_rows`] for why `check_text`'s span rows need it.
struct LineIndex {
    starts: Vec<usize>,
}

impl LineIndex {
    fn new(source: &str) -> Self {
        let mut starts = vec![0];
        starts.extend(source.match_indices('\n').map(|(index, _)| index + 1));
        Self { starts }
    }

    fn line_col(&self, source: &str, offset: usize) -> (usize, usize) {
        let mut offset = offset.min(source.len());
        while !source.is_char_boundary(offset) {
            offset -= 1;
        }
        let line = self.starts.partition_point(|&start| start <= offset);
        let line_start = self.starts[line - 1];
        let column = 1 + source[line_start..offset].chars().count();
        (line, column)
    }
}

/// Installs the tagger, parser, and DMS-index bytes a native build embeds
/// instead.
///
/// Then builds the engine once (tagger, parser, and the packs
/// `Engine::for_lang` bundles) so a construction error in any of them
/// surfaces here rather than lazily on the first
/// [`fix_text`]/[`check_text`]/[`explain_text`] call. The packs outside
/// that bundle (DMS view, jargon, evidence) still load on first use:
/// the DMS bytes' expensive parse already ran in this function's own
/// validation pass, and the rest are embedded artifacts whose parse the
/// native test suite pins, so nothing that can fail on real input is
/// deferred.
///
/// `tagger_bytes`/`parser_bytes` are either the audited `json.gz`
/// interchange format or the derived `.bin` view format — both
/// auto-detected by the install seams
/// ([`friction_nlp::install_tagger_weights`]/[`friction_nlp::install_parser_weights`]).
/// `dms_bytes` is `dms-index-en-v1.toml`'s UTF-8 source
/// ([`friction_packs::pack_dms_index_bin`] rebuilds the automaton from it
/// as part of this function's own validation pass — see below).
///
/// Every input is validated (parsed in full, for `dms_bytes` also packed)
/// *before* any `install_*` call runs. Every one of those install seams
/// has a one-shot slot that can be set at most once for the life of the
/// module instance, with no reset/uninstall path — see
/// [`friction_nlp::install_tagger_weights`]'s and
/// [`friction_packs::install_dms_index_bin`]'s own docs. Validating first
/// means a caller-correctable bad-bytes error (a truncated/corrupted
/// fetch) is this function's own `Err` arm, never a slot permanently
/// consumed by a doomed install a retry can't undo. Since each validation
/// pass runs the identical parse its install path would run, bytes that
/// pass validation are guaranteed to install (and, downstream, to build)
/// successfully.
///
/// # Errors
/// Returns a [`JsError`] if any input fails to validate or any install
/// call fails — in particular, a second `init_engine` call always errors
/// (the install seams reject a redundant install; see their own
/// `InstallError` docs), never silently no-ops and never panics.
#[wasm_bindgen]
pub fn init_engine(
    tagger_bytes: Vec<u8>,
    parser_bytes: Vec<u8>,
    dms_bytes: Vec<u8>,
) -> Result<(), JsError> {
    friction_nlp::validate_tagger_weights(&tagger_bytes).map_err(to_js_error)?;
    friction_nlp::validate_parser_weights(&parser_bytes).map_err(to_js_error)?;
    let dms_text = String::from_utf8(dms_bytes).map_err(|err| {
        JsError::new(&format!(
            "dms-index-en-v1.toml bytes were not valid UTF-8: {err}"
        ))
    })?;
    let dms_packed = friction_packs::pack_dms_index_bin(&dms_text).map_err(to_js_error)?;

    friction_nlp::install_tagger_weights(tagger_bytes).map_err(to_js_error)?;
    friction_nlp::install_parser_weights(parser_bytes).map_err(to_js_error)?;
    friction_packs::install_dms_index_bin(dms_packed).map_err(to_js_error)?;
    // Builds the tagger, parser, and every pack `PackSet::for_lang`
    // bundles now (surfacing a malformed-artifact error here instead of
    // on the first real call) and caches the result in `ENGINE` — see
    // that static's own doc comment for why every later call reuses it
    // rather than rebuilding from the installed bytes each time.
    let built = Engine::for_lang(LANG).map_err(to_js_error)?;
    ENGINE
        .set(built)
        .map_err(|_| JsError::new("the engine is already initialized"))?;
    Ok(())
}

/// This crate's own `Cargo.toml` version.
#[wasm_bindgen]
#[must_use]
pub fn engine_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Runs `friction fix`'s repair engine and returns the fixed document
/// text — the same bytes `friction fix` prints to stdout without
/// `--suggest`/`--in-place`.
///
/// # Errors
/// Returns a [`JsError`] if `input` fails to parse or segment, or if
/// [`init_engine`] was never called (or failed) first — this crate
/// checks [`ENGINE`] itself rather than calling
/// `Engine::for_lang`/`PerceptronTagger::for_lang` fresh each time, so an
/// uninstalled engine is this function's own `Err` arm, never the wasm
/// trap an uninstalled tagger's panic contract would otherwise produce.
#[wasm_bindgen]
pub fn fix_text(input: &str) -> Result<String, JsError> {
    let syntax = syntax_of(input);
    let (output, _report) = engine()?
        .fix_document_with(input, syntax)
        .map_err(to_js_error)?;
    Ok(output)
}

/// One metric's value, `docs`' envelope band for it (if the embedded
/// pack has one), whether the value falls inside that band, and whether
/// any genre's own combined score counts this metric at all. Mirrors
/// `friction-cli::check::MetricRow` field-for-field.
#[derive(Debug, Serialize)]
struct MetricRow {
    name: &'static str,
    value: f64,
    lo: Option<f64>,
    hi: Option<f64>,
    in_envelope: Option<bool>,
    weak_signal: bool,
}

/// One detected span, flattened to a stable, serializable shape. Mirrors
/// `friction-cli::check::SpanRow` field-for-field.
#[derive(Debug, Serialize)]
struct SpanRow {
    channel: &'static str,
    frame_id: String,
    start: usize,
    end: usize,
    line: usize,
    column: usize,
    score: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    message: Option<String>,
}

/// The pooled DMS machine-vs-human document-level statistics. Mirrors
/// `friction-cli::check::DmsMachineRow` field-for-field.
#[derive(Debug, Serialize)]
struct DmsMachineRow {
    mean_machine: f64,
    mean_human: f64,
    differential: f64,
    token_count: usize,
}

/// The full `check_text` JSON shape — the same shape as `friction check
/// --format json`. Mirrors `friction-cli::check::CheckReport`
/// field-for-field.
#[derive(Debug, Serialize)]
struct CheckReport {
    metrics: Vec<MetricRow>,
    spans: Vec<SpanRow>,
    tell_counts: BTreeMap<String, usize>,
    dms: DmsMachineRow,
}

/// Runs `friction check`'s metrics and fix-time detection, with no fixes
/// applied, and returns the same JSON shape as `friction check --format
/// json`.
///
/// Genre-free, like the CLI: every metric is judged against the
/// cross-genre union band — see `EnvelopePack::band_union`.
///
/// # Errors
/// Returns a [`JsError`] if `input` fails to parse, if the detection
/// layer fails to build or run, or if `init_engine` was never called
/// first — see [`fix_text`]'s own doc comment for the same
/// uninstalled-engine behavior.
#[wasm_bindgen]
pub fn check_text(input: &str) -> Result<String, JsError> {
    // `SrxSegmenter::for_lang` is a `const fn` (no parsing to redo); the
    // tagger is `ENGINE`'s own, reused for the same reason `fix_text`/
    // `explain_text` reuse the whole engine — see that static's doc
    // comment.
    let segmenter = SrxSegmenter::for_lang(LANG);
    let tagger = engine()?.tagger();
    let packs = friction_packs::PackSet::for_lang(LANG);
    let pack: &EnvelopePack = &friction_packs::ENVELOPE_V2;

    let syntax = syntax_of(input);
    let document = friction_parse::parse_with(input, syntax).map_err(to_js_error)?;
    let metrics = friction_metrics::compute(&document, &segmenter, tagger);

    let match_engine = MatchEngine::new(
        &packs.inventory.pack,
        &packs.dms.pack,
        &packs.jargon.pack,
        packs.jargon_attest,
        packs.human_evidence,
        tagger,
        &segmenter,
    )
    .map_err(to_js_error)?;
    let report = match_engine.scan(&document).map_err(to_js_error)?;

    let rows = metric_rows(&metrics, pack);
    let check_report = CheckReport {
        metrics: rows,
        spans: span_rows(input, &report.spans),
        tell_counts: tell_counts(&report.spans),
        dms: dms_row(&report),
    };
    serde_json::to_string_pretty(&check_report).map_err(to_js_error)
}

fn metric_rows(metrics: &MetricVector, pack: &EnvelopePack) -> Vec<MetricRow> {
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

/// This span's `frame_id`, up to (not including) its first `.` — see
/// `friction-cli::check::frame_prefix`'s own docs for why this is the
/// grouping key.
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

/// One operation the repair engine actually fired. Mirrors
/// `friction-cli::explain::OperationRow` field-for-field.
#[derive(Debug, Serialize)]
struct OperationRow {
    rule: String,
    start: usize,
    end: usize,
    replacement: String,
}

/// One candidate a gate held back. Mirrors
/// `friction-cli::explain::HeldRow` field-for-field.
#[derive(Debug, Serialize)]
struct HeldRow {
    rule: String,
    start: usize,
    end: usize,
    reason: String,
}

/// One pass's fired operations and held candidates. Mirrors
/// `friction-cli::explain::PassRow` field-for-field.
#[derive(Debug, Serialize)]
struct PassRow {
    pass: usize,
    fired: Vec<OperationRow>,
    held: Vec<HeldRow>,
}

/// The full `explain_text` JSON shape — the same shape as `friction
/// explain --format json`. Mirrors `friction-cli::explain::ExplainReport`
/// field-for-field.
#[derive(Debug, Serialize)]
struct ExplainReport {
    passes: Vec<PassRow>,
    patches_applied: usize,
}

/// Runs the repair engine (exactly like [`fix_text`]) but never returns
/// the fixed text.
///
/// Instead, per pass, every operation that fired and every candidate a
/// gate held back, followed by a convergence summary — the same JSON
/// shape as `friction explain --format json`.
///
/// # Errors
/// See [`fix_text`]'s own doc comment.
#[wasm_bindgen]
pub fn explain_text(input: &str) -> Result<String, JsError> {
    let syntax = syntax_of(input);
    let (_output, report) = engine()?
        .fix_document_with(input, syntax)
        .map_err(to_js_error)?;

    let passes = pass_rows(&report);
    let patches_applied = report
        .passes
        .iter()
        .map(|p| p.patches_applied)
        .sum::<usize>();

    let explain_report = ExplainReport {
        passes,
        patches_applied,
    };
    serde_json::to_string_pretty(&explain_report).map_err(to_js_error)
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
