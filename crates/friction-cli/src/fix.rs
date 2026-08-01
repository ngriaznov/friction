//! `friction fix`: runs the five-operation repair engine and writes the
//! fixed text to stdout (or back to the input file with `--in-place`).
//! `jargon.metaphor` and `overuse.word` (both unioned into the paraphrase
//! report below) are detection-only and add no sixth operation.
//!
//! No `--genre`/`--pack`: the engine's five operations (ritual deletion,
//! paired substitution, derivational pivot, gated span deletion, frame-
//! gated `just`-deletion) are gated by the curated inventory/attestation
//! packs and the clause chunker only, never by a metric or genre
//! envelope.
//!
//! A summary (passes run, patches applied per operation, how many
//! `Suggest`-tier candidates remain held, how many DMS paraphrase spans
//! were flagged) is always printed to stderr, so stdout stays exactly the
//! fixed document — safe to pipe or redirect. `--suggest` additionally
//! lists every remaining held candidate (rule, span, reason) and every
//! flagged paraphrase span (family, score, snippet) on stderr.
//!
//! No `--family` either: the paraphrase report (below) scans every
//! generator family the embedded DMS pack defines, since fix runs without
//! a declared target family and reporting only one would be an arbitrary
//! choice.
//!
//! # DMS, contrast frames, jargon, and overuse surface paraphrase candidates, never edits
//!
//! After fixing, the FIXED output is scanned with the DMS channel
//! (`friction_match::dms_spans_for_family`) against every family
//! [`friction_packs::ModelFamily::ALL`] defines, with the contrast-frame
//! template channel (`friction_match::frame::scan_units`) — the same
//! detector [`friction_match::Channel::Frame`] spans in `friction check`
//! come from — with the metaphor-compound jargon channel
//! (`friction_match::jargon::scan_units`, the same detector
//! [`friction_match::Channel::Jargon`] spans come from) — and with the
//! per-document word-overuse channel (`friction_match::overuse::scan_units`,
//! the same detector [`friction_match::Channel::Overuse`] spans come
//! from). Every span from all four is unioned into one report (see
//! [`union_paraphrase_spans`]). These are detection-only: the closed-
//! operation engine above has no licensed rewrite for a DMS tell (no
//! fixed substitution string, no attested derivation), no licensed
//! rewrite for an invented term or an overused word (no deterministic
//! true replacement — SYNTHESIS.md §4 for jargon; which synonym to reach
//! for is exactly the same kind of judgment call for overuse), and only
//! ever acts on a `frame.contrast.question` span whose marker is
//! `just`/`merely`/`simply` (`frame.dejust`) — a `frame.contrast.question`
//! span whose marker is `only`, every `frame.contrast.correction` span,
//! every `jargon.metaphor` span, and every `overuse.word` span all have no
//! licensed edit either, so all four are left for a human to paraphrase
//! alongside DMS's. DMS stays banned as an edit judge — `friction-edit`'s
//! own crate docs say every gate is the curated inventory pack and the
//! corpus-attested seam-bigram/skeleton tables, never a metric or genre
//! envelope, and this module does not change that: `friction_edit::Engine`
//! is never given the DMS pack, only `friction_match` is, purely for
//! reporting.

use std::collections::BTreeMap;
use std::ops::Range;
use std::path::Path;
use std::process::ExitCode;

use clap::Args;
use friction_core::{Finding, RuleId};
use friction_edit::EditReport;
use friction_match::{MatchScore, MatchSpan};
use friction_packs::ModelFamily;
use rayon::prelude::*;
use serde::Serialize;

use crate::common::{CliError, Format, LineIndex, display_path, read_input, write_in_place};

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

/// `--format json`'s shape for the pass summary — always a single JSON
/// document on stderr, so a consumer can parse the whole stream with one
/// `serde_json::from_str` call. Without `--suggest` the two detail arrays
/// are omitted entirely (not `null`), keeping the plain summary shape;
/// with it, they carry the same rows the text format lists line by line.
#[derive(Debug, Serialize)]
struct FixSummary {
    passes: usize,
    patches_applied: usize,
    patches_by_rule: BTreeMap<String, usize>,
    suggest_count: usize,
    paraphrase_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    suggestions: Option<Vec<SuggestionRow>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    paraphrase: Option<Vec<ParaphraseRow>>,
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

/// `--format json`'s shape for one `--suggest`-listed paraphrase span (see
/// [`union_paraphrase_spans`]).
#[derive(Debug, Serialize)]
struct ParaphraseRow {
    frame_id: String,
    start: usize,
    end: usize,
    line: usize,
    column: usize,
    score: i64,
    snippet: String,
}

/// One DMS paraphrase candidate after unioning per-family spans down to
/// one entry per flagged byte region (see [`union_paraphrase_spans`]).
#[derive(Debug, Clone, PartialEq, Eq)]
struct ParaphraseSpan {
    /// The winning family's frame id, e.g. `"dms.claude"`.
    frame_id: Box<str>,
    /// Byte range into the FIXED output.
    range: Range<usize>,
    /// The winning family's differential score.
    score: i64,
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
    let syntax = crate::common::syntax_of(&args.input, &source);
    let engine = friction_edit::Engine::new()?;
    let (output, report) = engine.fix_document_with(&source, syntax)?;

    if args.in_place {
        write_in_place(Path::new(&args.input), &output)?;
    } else {
        print!("{output}");
    }

    let remaining_held = final_pass_held(&report);
    let paraphrase_spans = scan_paraphrase_spans(
        &output,
        syntax,
        engine.tagger(),
        report.reusable_scan.as_ref(),
    )?;
    print_summary(
        args.format,
        &report,
        &output,
        &remaining_held,
        &paraphrase_spans,
        args.suggest,
    );

    if args.suggest && args.format != Format::Json {
        print_suggestions(&output, display_path(&args.input), &remaining_held);
        print_paraphrase_spans(&output, &paraphrase_spans);
    }

    Ok(ExitCode::SUCCESS)
}

/// The engine's current held candidates: the register pass's holds
/// (always the report's final entry) merged with the last bounded
/// pass's — the bounded passes re-scan every sentence each round, so
/// the final bounded pass (the zero-patch convergence pass, or the
/// bounded-out last round) carries the gate-held diagnostics against
/// the text the engine actually converged to, and the register pass
/// carries only its own remainder findings, never those.
fn final_pass_held(report: &EditReport) -> Vec<Finding> {
    let passes = &report.passes;
    let mut held: Vec<Finding> = passes
        .len()
        .checked_sub(2)
        .map(|i| passes[i].held.clone())
        .unwrap_or_default();
    held.extend(
        passes
            .last()
            .map(|pass| pass.held.clone())
            .unwrap_or_default(),
    );
    held
}

fn print_summary(
    format: Format,
    report: &EditReport,
    output: &str,
    suggestions: &[Finding],
    paraphrase_spans: &[ParaphraseSpan],
    suggest: bool,
) {
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
                suggest_count: suggestions.len(),
                paraphrase_count: paraphrase_spans.len(),
                suggestions: suggest.then(|| suggestion_rows(output, suggestions)),
                paraphrase: suggest.then(|| paraphrase_rows(output, paraphrase_spans)),
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
            eprintln!("  suggest: {} finding(s) remain", suggestions.len());
            eprintln!(
                "  paraphrase: {} span(s) flagged for manual rewrite",
                paraphrase_spans.len()
            );
            // A count with no way to see what it counts is a dead end:
            // point at the flag that prints the actual findings.
            if !suggest && (!suggestions.is_empty() || !paraphrase_spans.is_empty()) {
                eprintln!("  (run with --suggest to see them)");
            }
        }
    }
}

/// The `--suggest` held-candidate rows, positioned against the fixed
/// output's line/column grid.
fn suggestion_rows(output: &str, suggestions: &[Finding]) -> Vec<SuggestionRow> {
    let lines = LineIndex::new(output);
    suggestions
        .iter()
        .map(|f| {
            let (line, column) = lines.line_col(output, f.range.start);
            SuggestionRow {
                rule: f.rule.as_str().to_string(),
                start: f.range.start,
                end: f.range.end,
                line,
                column,
                message: f.message.clone(),
            }
        })
        .collect()
}

/// The `--suggest` paraphrase rows, positioned against the fixed output's
/// line/column grid.
fn paraphrase_rows(output: &str, spans: &[ParaphraseSpan]) -> Vec<ParaphraseRow> {
    let lines = LineIndex::new(output);
    spans
        .iter()
        .map(|s| {
            let (line, column) = lines.line_col(output, s.range.start);
            ParaphraseRow {
                frame_id: s.frame_id.to_string(),
                start: s.range.start,
                end: s.range.end,
                line,
                column,
                score: s.score,
                snippet: output[s.range.clone()].to_string(),
            }
        })
        .collect()
}

/// Text-format `--suggest` held-candidates list; `--format json` instead
/// embeds the same rows in [`FixSummary`]'s single document.
fn print_suggestions(output: &str, path_label: &str, suggestions: &[Finding]) {
    let lines = LineIndex::new(output);
    for f in suggestions {
        let (line, column) = lines.line_col(output, f.range.start);
        eprintln!(
            "{path_label}:{line}:{column}: {} [{}]",
            f.message,
            f.rule.as_str()
        );
    }
}

/// Scans `output` — the FIXED text `fix` is about to print — for DMS
/// paraphrase candidates against every family the embedded DMS pack
/// defines, plus contrast-frame template spans
/// (`friction_match::frame::scan_units`), metaphor-compound jargon spans
/// (`friction_match::jargon::scan_units`), and per-document word-overuse
/// spans (`friction_match::overuse::scan_units`), unioning everything into
/// one report (see [`union_paraphrase_spans`]). Only prose blocks are
/// scanned, the same scoping every detection channel already uses
/// (`friction_match::token::prose_scope`).
///
/// `reusable` is `run_inner`'s own `report.reusable_scan`
/// ([`friction_edit::EditReport::reusable_scan`]): when `Some`, the
/// engine's register pass already parsed `output` and tagged its
/// sentences (register applied zero patches this call, so its own input
/// and `output` are the exact same bytes — see
/// `friction_edit::ReusableScan`'s own docs), and this function reuses
/// that parse and those tags instead of redoing either. When `None`
/// (register applied at least one patch, so its own artifacts describe a
/// different text than `output`), this falls back to parsing `output` and
/// tagging its sentences fresh, exactly as before this reuse path
/// existed.
///
/// Reuses `run_inner`'s already-built `friction_edit::Engine`'s tagger
/// (jargon detection needs part-of-speech tags — see
/// `friction_match::jargon`'s own module docs) via
/// [`friction_edit::Engine::tagger`] rather than constructing a second
/// `friction_nlp::PerceptronTagger` — one embedded-weight load per `fix`
/// invocation instead of two. Only used on the fresh-tag fallback path;
/// ignored (but still accepted, keeping this function's signature stable
/// either way) on a reuse hit.
///
/// The four sources — every family's DMS walk, the contrast-frame scan,
/// the jargon scan, and the overuse scan (the jargon scan is itself
/// internally parallel — see `friction_match::jargon::scan_units`) — read
/// only `units`/`document` and never each other's output, so they run as
/// four concurrent `rayon::join` tasks (the DMS walk is itself a
/// `par_iter` over [`ModelFamily::ALL`], five more independent tasks).
/// Deterministic regardless of run order or thread count: [`union_paraphrase_spans`]
/// sorts by `(start, end, frame_id)` before merging, so which task
/// finishes first never affects the returned bytes.
///
/// [`friction_match::tagging::TaggedSentence`]s built from `reusable`'s
/// already-tagged sentences, `None` if there's nothing to reuse (no
/// `reusable`) or reuse would be incomplete.
///
/// `register::SentenceCtx` (the source `reusable.sentences` came from)
/// silently drops a sentence whose trimmed text is empty or whose
/// dependency parse fails (`build_sentence_ctx`'s own docs — the latter
/// "never expected" from the shipped parser, but not contractually
/// impossible), so `reusable.sentences` can in principle be a strict
/// SUBSET of `units`'s full flattened sentence list. Reusing that subset
/// directly would silently under-scan the missing sentences relative to
/// the fresh-tag path (a real behavior change, not just a slower one), so
/// this checks the count first and returns `None` — falling back to a
/// full, correct retag — on any mismatch, rather than ever reusing a
/// partial set. An empty-trimmed sentence costs nothing either way (no
/// tokens to scan), so in practice this only ever falls back on the
/// "never expected" parse-failure case.
///
/// Every kept sentence's link-label status is looked up by exact range,
/// not position, since `reusable.sentences`' order need not otherwise be
/// assumed to align with `units`'s.
fn reused_tags(
    reusable: Option<&friction_edit::ReusableScan>,
    units: &[friction_match::token::ScopedUnit<'_>],
    document_text: &str,
) -> Option<Vec<friction_match::tagging::TaggedSentence>> {
    let scan = reusable?;
    let scopes = friction_match::tagging::sentence_scopes(units, document_text);
    if scan.sentences.len() != scopes.len() {
        return None;
    }
    let link_labels: std::collections::BTreeMap<(usize, usize), bool> = scopes
        .into_iter()
        .map(|(range, in_link_label)| ((range.start, range.end), in_link_label))
        .collect();
    Some(
        scan.sentences
            .iter()
            .map(|sentence| {
                let in_link_label = link_labels
                    .get(&(sentence.range.start, sentence.range.end))
                    .copied()
                    .unwrap_or(false);
                friction_match::tagging::from_local_tags(
                    sentence.range.clone(),
                    &sentence.tokens,
                    in_link_label,
                )
            })
            .collect(),
    )
}

/// # Errors
/// Returns [`CliError::Parse`] if `output` fails to parse as `syntax` —
/// not expected, since `output` is text `fix_document` has already parsed
/// successfully once. Never returns this error on the reuse path, since
/// no parsing happens there.
fn scan_paraphrase_spans(
    output: &str,
    syntax: friction_parse::Syntax,
    tagger: &dyn friction_nlp::Tagger,
    reusable: Option<&friction_edit::ReusableScan>,
) -> Result<Vec<ParaphraseSpan>, CliError> {
    let segmenter = friction_nlp::SrxSegmenter::new();

    // Deferred-init: exactly one of these two arms runs, and `document`
    // borrows whichever one supplied the parse — the reused
    // `scan.document` or a freshly parsed `owned_document`.
    let owned_document;
    let document: &friction_core::Document = if let Some(scan) = reusable {
        &scan.document
    } else {
        owned_document = friction_parse::parse_with(output, syntax)?;
        &owned_document
    };
    let units = friction_match::token::prose_scope(document, &segmenter);

    let (dms_spans, (frame_spans, (jargon_spans, overuse_spans))) = rayon::join(
        || dms_spans_for_every_family(&units),
        || {
            rayon::join(
                || friction_match::frame::scan_units(&units),
                || {
                    // Jargon and overuse both need the tagged sentences,
                    // so tagging happens once, before their parallel
                    // scans. No `reusable`, or (never expected —
                    // `build_sentence_ctx`'s own docs — but checked,
                    // not assumed) register dropped a sentence: tag
                    // every sentence once, through the same shared-tag
                    // entry point `friction_match::MatchEngine::scan`
                    // uses for its own LVC/jargon channels — see
                    // `friction_match::tagging`'s own module docs.
                    let tagged_sentences = reused_tags(reusable, &units, document.source())
                        .unwrap_or_else(|| {
                            friction_match::tagging::tag_units(&units, document.source(), tagger)
                        });
                    rayon::join(
                        || {
                            friction_match::jargon::scan_units(
                                &tagged_sentences,
                                document.source(),
                                &friction_packs::JARGON.pack,
                                &friction_packs::JARGON_ATTEST,
                            )
                        },
                        || {
                            friction_match::overuse::scan_units(
                                &tagged_sentences,
                                document.source(),
                                &friction_packs::HUMAN_EVIDENCE,
                            )
                        },
                    )
                },
            )
        },
    );

    let mut spans = dms_spans;
    spans.extend(frame_spans);
    spans.extend(jargon_spans);
    spans.extend(overuse_spans);

    Ok(union_paraphrase_spans(spans))
}

/// Runs [`friction_match::dms_spans_for_family`] for every family
/// [`ModelFamily::ALL`] defines, concurrently (each family's walk touches
/// only its own automaton, never another's), collecting an
/// [`ModelFamily::ALL`]-ordered `Vec<Vec<MatchSpan>>` before flattening —
/// keeps assembly order fixed and thread-count-independent, matching
/// [`register`](friction_edit::register)'s own ordered-collect discipline,
/// even though [`union_paraphrase_spans`]'s later sort would tolerate
/// either order.
fn dms_spans_for_every_family(units: &[friction_match::token::ScopedUnit<'_>]) -> Vec<MatchSpan> {
    let per_family: Vec<Vec<MatchSpan>> = ModelFamily::ALL
        .par_iter()
        .filter_map(|&family| {
            friction_match::dms_spans_for_family(units, &friction_packs::DMS.pack, family)
        })
        .collect();
    per_family.into_iter().flatten().collect()
}

/// A [`MatchSpan`]'s DMS differential score. Always
/// [`MatchScore::Differential`] for the [`Channel::Dms`](friction_match::Channel::Dms)
/// spans [`scan_paraphrase_spans`] collects; `0` for the `Present` variant
/// is a defensive fallback that never fires in practice.
const fn dms_score(span: &MatchSpan) -> i64 {
    match span.score {
        MatchScore::Differential(d) => d,
        MatchScore::Present => 0,
    }
}

/// Unions DMS spans flagged across every generator family, plus the
/// contrast-frame template spans, down to one entry per flagged byte
/// region: fix's paraphrase report is per region, not per source, so a
/// DMS family and a frame template flagging overlapping words report
/// once. Sorts by `(start, end, frame_id)` first — mirroring
/// `friction_match::span::merge_spans`'s own tie-break discipline — then
/// folds each span into the running region whenever its start falls at or
/// before that region's end (overlapping or exactly adjacent), keeping
/// whichever span scored higher ([`dms_score`]'s `0` fallback for a
/// `MatchScore::Present` frame span means an overlapping DMS span always
/// wins the label on a genuine tie, never the reverse — an arbitrary but
/// deterministic tie-break, not a claim that DMS is the "better" find).
/// An exact tie keeps the alphabetically-earliest `frame_id` (the one
/// already carried by the running region, since ties are resolved by not
/// overwriting), so the result never depends on scan order. Because
/// starts only grow across the sorted input, the merged spans come out
/// already sorted the same way — no second sort is needed.
fn union_paraphrase_spans(mut spans: Vec<MatchSpan>) -> Vec<ParaphraseSpan> {
    spans.sort_by(|a, b| {
        a.range
            .start
            .cmp(&b.range.start)
            .then(a.range.end.cmp(&b.range.end))
            .then(a.frame_id.cmp(&b.frame_id))
    });

    let mut merged: Vec<ParaphraseSpan> = Vec::new();
    for span in spans {
        let score = dms_score(&span);
        if let Some(last) = merged.last_mut()
            && span.range.start <= last.range.end
        {
            last.range.end = last.range.end.max(span.range.end);
            if score > last.score {
                last.score = score;
                last.frame_id = span.frame_id;
            }
            continue;
        }
        merged.push(ParaphraseSpan {
            frame_id: span.frame_id,
            range: span.range,
            score,
        });
    }
    merged
}

/// Text-format `--suggest` paraphrase list; `--format json` instead
/// embeds the same rows in [`FixSummary`]'s single document.
fn print_paraphrase_spans(output: &str, spans: &[ParaphraseSpan]) {
    let lines = LineIndex::new(output);
    for s in spans {
        let (line, column) = lines.line_col(output, s.range.start);
        let snippet = &output[s.range.clone()];
        eprintln!(
            "{line}:{column}: paraphrase candidate (score {}): {snippet:?} [{}]",
            s.score, s.frame_id
        );
    }
}

#[cfg(test)]
mod tests {
    use friction_core::{Patch, Tier};
    use friction_edit::PassReport;
    use friction_match::Channel;

    use super::*;

    fn dms_span(range: Range<usize>, frame_id: &str, score: i64) -> MatchSpan {
        MatchSpan {
            range,
            channel: Channel::Dms,
            frame_id: frame_id.into(),
            score: MatchScore::Differential(score),
            message: None,
        }
    }

    /// Overlapping spans from different families merge into one region,
    /// keeping the higher-scored family's label.
    #[test]
    fn union_paraphrase_spans_merges_overlapping_spans_keeping_the_higher_score() {
        let a = dms_span(0..10, "dms.qwen", 5);
        let b = dms_span(5..15, "dms.claude", 9);
        let merged = union_paraphrase_spans(vec![a, b]);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].range, 0..15);
        assert_eq!(&*merged[0].frame_id, "dms.claude");
        assert_eq!(merged[0].score, 9);
    }

    /// Exactly-adjacent spans (one's end equals the other's start) merge
    /// too, per this module's documented "overlapping or adjacent" policy.
    #[test]
    fn union_paraphrase_spans_merges_adjacent_spans() {
        let a = dms_span(0..5, "dms.qwen", 4);
        let b = dms_span(5..10, "dms.claude", 4);
        let merged = union_paraphrase_spans(vec![a, b]);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].range, 0..10);
    }

    /// Non-overlapping, non-adjacent spans stay separate entries.
    #[test]
    fn union_paraphrase_spans_keeps_disjoint_spans_separate() {
        let a = dms_span(0..5, "dms.qwen", 4);
        let b = dms_span(10..15, "dms.claude", 4);
        let merged = union_paraphrase_spans(vec![a, b]);
        assert_eq!(merged.len(), 2);
    }

    /// An exact score tie keeps the alphabetically-earliest family,
    /// regardless of the input `Vec`'s order — the initial
    /// `(start, end, frame_id)` sort makes the outcome order-independent.
    #[test]
    fn union_paraphrase_spans_breaks_score_ties_by_frame_id_order() {
        let a = dms_span(0..10, "dms.claude", 5);
        let b = dms_span(0..10, "dms.qwen", 5);
        let merged = union_paraphrase_spans(vec![b, a]);
        assert_eq!(merged.len(), 1);
        assert_eq!(&*merged[0].frame_id, "dms.claude");
    }

    /// The result is sorted by `(start, end, frame_id)`, independent of
    /// input order.
    #[test]
    fn union_paraphrase_spans_sorts_the_result() {
        let a = dms_span(20..25, "dms.qwen", 4);
        let b = dms_span(0..5, "dms.claude", 4);
        let merged = union_paraphrase_spans(vec![a, b]);
        assert_eq!(merged[0].range, 0..5);
        assert_eq!(merged[1].range, 20..25);
    }

    fn pass(held: Vec<Finding>) -> PassReport {
        PassReport {
            patches_applied: 0,
            patches_dropped: 0,
            applied_patches: Vec::new(),
            held,
        }
    }

    /// `final_pass_held` merges the last bounded pass's holds (the
    /// gate-held candidates against the converged text — frame and
    /// deletion holds live there) with the register pass's own (always
    /// the final entry). A pass before the converged one contributes
    /// nothing.
    #[test]
    fn final_pass_held_merges_converged_and_register_passes() {
        let stale = Finding::new(RuleId::new("span.delete"), 0..1, "stale", Tier::Suggest);
        let converged = Finding::new(RuleId::new("vsub.notify"), 0..1, "gate", Tier::Suggest);
        let register = Finding::new(RuleId::new("pivot.lvc"), 1..2, "register", Tier::Suggest);
        let report = EditReport {
            passes: vec![
                pass(vec![stale]),
                pass(vec![converged.clone()]),
                pass(vec![register.clone()]),
            ],
            reusable_scan: None,
        };
        assert_eq!(final_pass_held(&report), vec![converged, register]);
    }

    /// An empty `EditReport` (never produced by `fix_document` in
    /// practice, but defensively handled) yields no held candidates
    /// rather than panicking.
    #[test]
    fn final_pass_held_handles_no_passes() {
        let report = EditReport {
            passes: Vec::new(),
            reusable_scan: None,
        };
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
            reusable_scan: None,
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
