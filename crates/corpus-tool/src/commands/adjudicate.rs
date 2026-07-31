//! `corpus-tool adjudicate` — the evidence referee for the frame-rewrite
//! rule set.
//!
//! Re-derives every rule's verdict straight from the shipped corpora and
//! reports where the re-measurement agrees or disagrees with the verdict
//! already recorded in `frame-rules-v1.toml`.
//!
//! For every rule in all seven buckets (`rules_ship`, `rules_flip`,
//! `rules_surface`, `rules_pilot`, `rules_quarantine`,
//! `rules_no_evidence`, `rules_staged_surface`), this command extracts the
//! rule's literal probes (`friction_packs::frame_rules::literal_probes`
//! for the six knowledge buckets; the pattern's own whitespace-split
//! surface tokens for `rules_pilot`), counts each probe's non-overlapping,
//! sentence-bounded occurrences over the train+dev splits of both
//! corpus sides, normalizes to occurrences per million tokens, and classifies
//! the result with `friction_packs::frame_rules::classify` — the exact
//! function the shipped verdicts were computed with. A rule whose pattern
//! fails to parse, or which extracts no literal content at all, is
//! reported "unprobeable" rather than causing a crash: several
//! `rules_staged_surface` rows are deliberately malformed (the pack
//! compiler must reject them; this tool must not choke on them — see
//! `friction_packs::frame_rules`'s own module docs on the compile fence).
//!
//! Report-only, mirroring `output-bands`/`register-bands`'s "never writes
//! pack TOML directly" discipline: folding a promoted or demoted rule
//! into a different bucket is still a human, hand-editing step.
//!
//! # Tokenization
//!
//! Prose is extracted per document via `friction-parse` (`Document::prose`),
//! split into sentences with `friction_harness::clean::split_sentences`,
//! and each sentence is tagged with the embedded `PerceptronTagger` — the
//! same three-stage pipeline `corpus-tool mine-inventory` already uses
//! (see that module's own doc comment for why prose extraction beats raw
//! whole-file tokenization here: probes are natural-language phrases, and
//! a code block is not one). Every tagged token contributes two parallel
//! per-sentence streams: its lowercased dictionary lemma
//! (`TaggedToken::lemma`) and its lowercased surface text.
//! Knowledge-bucket probes match the lemma stream (pattern literals are
//! lemma-matched — see `friction_packs::frame_rules::PatElem::Lit`);
//! `rules_pilot` probes are already-inflected surface phrases (e.g.
//! "utilizing") and match the surface stream instead. Both streams are
//! the same length, one entry per tagged token (word or punctuation), and
//! that per-token count is what "per million tokens" normalizes against.
//! A probe word matches a stream position when it equals **either** the
//! token's lemma or its lowercased surface: knowledge-bucket literals
//! are lemma-matched, but the surface bucket materializes inflected
//! forms ("utilizes") that only the surface stream can attest, and the
//! recorded verdicts were measured over both shapes.
//!
//! # Probe matching
//!
//! A probe is a flattened word sequence: one `literal_probes` element can
//! itself be a single multi-word quoted literal (e.g. `"is a testament
//! to"`, kept as one string by the pattern lexer) or a run of
//! single class-expanded words — flattening every probe on whitespace
//! turns both shapes into one word per stream position before matching.
//! Every probe across every rule is indexed once by its first word into a
//! shared lookup table, so one left-to-right scan per sentence tests only
//! the probes that could plausibly start at each position, never all of
//! them. A probe's occurrences within one sentence are counted greedily,
//! left to right, non-overlapping — the convention `str::matches` uses
//! for a literal substring, generalized to a token sequence; matches
//! never span two sentences because each sentence is scanned in
//! isolation and per-probe match state is not carried between calls.
//!
//! # Split discipline
//!
//! TRAIN and DEV splits, both classes: the recorded verdicts were
//! measured over roughly this token volume, and adjudication is an
//! evidence census, not an evaluation — only the sealed HOLDOUT split
//! stays untouched (it backs the near-no-op and holdout fixtures, which
//! must never see tuning pressure from rule selection).

use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use anyhow::Context as _;
use clap::Args as ClapArgs;
use friction_core::Document;
use friction_nlp::{PerceptronTagger, Tagger};
use friction_packs::frame_rules::{
    FrameRule, FrameRuleSet, PilotRule, RuleKind, Verdict, classify, literal_probes, parse_pattern,
};
use serde::Serialize;

use crate::corpus_layout::relpath;
use crate::manifest::{self, Class, ManifestRecord, Split};
use crate::metric_source::load_document;

/// Output format for `adjudicate`'s report.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum OutputFormat {
    /// Human-readable text (the default).
    Text,
    /// One JSON document with the same content.
    Json,
}

/// Arguments for `corpus-tool adjudicate`.
#[derive(Debug, ClapArgs)]
pub struct Args {
    /// Corpus root directory.
    #[arg(long, default_value = "corpus")]
    pub corpus_dir: PathBuf,
    /// Path to the frame-rewrite rule set to adjudicate.
    #[arg(
        long,
        default_value = "crates/friction-packs/packs/frame-rules-v1.toml"
    )]
    pub rules: PathBuf,
    /// Report format.
    #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
    pub format: OutputFormat,
}

/// The seven buckets `frame-rules-v1.toml` carries, in the fixed order
/// this command reports them (the same order the rule-set file itself
/// declares them in).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Bucket {
    Ship,
    Flip,
    Surface,
    Pilot,
    Quarantine,
    NoEvidence,
    StagedSurface,
}

impl Bucket {
    const ALL: [Self; 7] = [
        Self::Ship,
        Self::Flip,
        Self::Surface,
        Self::Pilot,
        Self::Quarantine,
        Self::NoEvidence,
        Self::StagedSurface,
    ];

    const fn name(self) -> &'static str {
        match self {
            Self::Ship => "ship",
            Self::Flip => "flip",
            Self::Surface => "surface",
            Self::Pilot => "pilot",
            Self::Quarantine => "quarantine",
            Self::NoEvidence => "no_evidence",
            Self::StagedSurface => "staged_surface",
        }
    }
}

/// One rule row, in final report order, before corpus measurement.
struct RuleRow {
    bucket: Bucket,
    id: String,
    /// The recorded verdict, normalized to [`verdict_label`]'s
    /// vocabulary. `rules_pilot` rows carry no `v` field at all (see
    /// [`PilotRule`]); a pilot rule sits in a compile-eligible bucket
    /// exactly like `ship`/`surface`, so `"confirmed"` is its implicit
    /// recorded status for this comparison.
    recorded_label: String,
    kind: RuleKind,
    /// `false` when the pattern failed to parse or extracted no literal
    /// content at all — this rule's [`Measured`] is
    /// [`Measured::Unprobeable`] regardless of what the corpus scan
    /// finds, because no probe was ever registered for it.
    probeable: bool,
}

/// One probe: a flattened, lowercased word sequence to search for, plus
/// which rule it counts toward. A probe word matches a position when it
/// equals either the token's lemma or its lowercased surface.
struct ProbeEntry {
    rule_idx: usize,
    words: Vec<String>,
}

/// Flattens one `literal_probes` element into a lowercase word sequence.
///
/// A `literal_probes` probe element can be a single multi-word quoted
/// literal (the pattern lexer keeps `"is a testament to"` as one string)
/// or a run of already-single-word entries (class expansion splits a
/// hyphenated member into one entry per word); splitting every entry on
/// whitespace normalizes both shapes into one word per output position,
/// matching one-for-one against a tagged sentence's per-token stream.
fn flatten_probe(probe: &[String]) -> Vec<String> {
    probe
        .iter()
        .flat_map(|part| part.split_whitespace())
        .map(str::to_lowercase)
        .collect()
}

/// Normalizes a recorded `v` field into the vocabulary [`verdict_label`]
/// produces: strip a leading `frame:` namespace prefix, strip a trailing
/// `(...)` annotation (e.g. `(human-tilted)`), lowercase, then fold the
/// one label that renames outright (`weak-differential` -> `weak`) —
/// every other observed label already matches its [`Verdict`] counterpart
/// once case-folded.
fn normalize_recorded_verdict(raw: &str) -> String {
    let unprefixed = raw.strip_prefix("frame:").unwrap_or(raw);
    let unannotated = match unprefixed.find('(') {
        Some(idx) if unprefixed.ends_with(')') => &unprefixed[..idx],
        _ => unprefixed,
    };
    let lower = unannotated.to_lowercase();
    if lower == "weak-differential" {
        "weak".to_string()
    } else {
        lower
    }
}

/// The label [`classify`]'s verdict renders as, in the same vocabulary
/// [`normalize_recorded_verdict`] folds recorded labels into.
const fn verdict_label(verdict: Verdict) -> &'static str {
    match verdict {
        Verdict::Confirmed => "confirmed",
        Verdict::GuardConfirmed => "guard-confirmed",
        Verdict::GuardWrong => "guard-wrong",
        Verdict::DirectionError => "direction-error",
        Verdict::Weak => "weak",
        Verdict::NoEvidence => "no-evidence",
    }
}

/// One rule's outcome after re-measurement.
enum Measured {
    /// The pattern failed to parse, or extracted no literal content at
    /// all — never scanned, never classified.
    Unprobeable,
    Evidence {
        verdict: Verdict,
        m_count: u64,
        h_count: u64,
        m_per_million: f64,
        h_per_million: f64,
    },
}

impl Measured {
    const fn label(&self) -> &'static str {
        match self {
            Self::Unprobeable => "unprobeable",
            Self::Evidence { verdict, .. } => verdict_label(*verdict),
        }
    }
}

/// Registers every rule of one knowledge bucket (all buckets but
/// `rules_pilot`) into `rows`/`probes`, in TOML array order.
fn push_knowledge_bucket(
    bucket: Bucket,
    rules: &[FrameRule],
    classes: &BTreeMap<String, Vec<String>>,
    rows: &mut Vec<RuleRow>,
    probes: &mut Vec<ProbeEntry>,
) {
    for rule in rules {
        let rule_idx = rows.len();
        let kind =
            RuleKind::parse(&rule.k).expect("kind letter already validated by FrameRuleSet::parse");
        let mut probeable = false;
        if let Ok(pattern) = parse_pattern(&rule.p) {
            for probe in literal_probes(&pattern, classes, 64) {
                let words = flatten_probe(&probe);
                if words.is_empty() {
                    continue;
                }
                probeable = true;
                probes.push(ProbeEntry { rule_idx, words });
            }
        }
        rows.push(RuleRow {
            bucket,
            id: rule.id.clone(),
            recorded_label: normalize_recorded_verdict(&rule.v),
            kind,
            probeable,
        });
    }
}

/// Registers every `rules_pilot` rule into `rows`/`probes`, in TOML array
/// order. A pilot rule's id is synthesized as `pilot:<p>` — [`PilotRule`]
/// carries no stable id field, and this is the same convention
/// `FrameRuleSet::parse` already uses to name a pilot rule in its own
/// `UnknownKind` error.
fn push_pilot_bucket(rules: &[PilotRule], rows: &mut Vec<RuleRow>, probes: &mut Vec<ProbeEntry>) {
    for rule in rules {
        let rule_idx = rows.len();
        let kind =
            RuleKind::parse(&rule.k).expect("kind letter already validated by FrameRuleSet::parse");
        let words = flatten_probe(std::slice::from_ref(&rule.p));
        let probeable = !words.is_empty();
        if probeable {
            probes.push(ProbeEntry { rule_idx, words });
        }
        rows.push(RuleRow {
            bucket: Bucket::Pilot,
            id: format!("pilot:{}", rule.p),
            recorded_label: "confirmed".to_string(),
            kind,
            probeable,
        });
    }
}

/// Builds every rule row and every probe, in final report order.
fn build_rows_and_probes(set: &FrameRuleSet) -> (Vec<RuleRow>, Vec<ProbeEntry>) {
    let mut rows = Vec::new();
    let mut probes = Vec::new();
    push_knowledge_bucket(
        Bucket::Ship,
        &set.rules_ship,
        &set.classes,
        &mut rows,
        &mut probes,
    );
    push_knowledge_bucket(
        Bucket::Flip,
        &set.rules_flip,
        &set.classes,
        &mut rows,
        &mut probes,
    );
    push_knowledge_bucket(
        Bucket::Surface,
        &set.rules_surface,
        &set.classes,
        &mut rows,
        &mut probes,
    );
    push_pilot_bucket(&set.rules_pilot, &mut rows, &mut probes);
    push_knowledge_bucket(
        Bucket::Quarantine,
        &set.rules_quarantine,
        &set.classes,
        &mut rows,
        &mut probes,
    );
    push_knowledge_bucket(
        Bucket::NoEvidence,
        &set.rules_no_evidence,
        &set.classes,
        &mut rows,
        &mut probes,
    );
    push_knowledge_bucket(
        Bucket::StagedSurface,
        &set.rules_staged_surface,
        &set.classes,
        &mut rows,
        &mut probes,
    );
    (rows, probes)
}

/// Indexes every probe by its first word, so a sentence scan only tests
/// probes that could plausibly start at the current position.
fn build_index(probes: &[ProbeEntry]) -> HashMap<String, Vec<u32>> {
    let mut index: HashMap<String, Vec<u32>> = HashMap::new();
    for (i, probe) in probes.iter().enumerate() {
        let Some(first) = probe.words.first() else {
            continue;
        };
        #[allow(
            clippy::cast_possible_truncation,
            reason = "probe count is far below u32::MAX"
        )]
        index.entry(first.clone()).or_default().push(i as u32);
    }
    index
}

/// Scans one sentence's token stream, counting every probe's
/// non-overlapping, leftmost occurrences into `counts[probe_idx]`
/// (`.0` for the machine side, `.1` for the human side). Per-probe match
/// state (`next_allowed`) is local to this call, so nothing here can ever
/// bridge two sentences.
fn scan_sentence(
    lemma: &[String],
    surface: &[String],
    index: &HashMap<String, Vec<u32>>,
    probes: &[ProbeEntry],
    is_llm: bool,
    counts: &mut [(u64, u64)],
) {
    let word_matches = |word: &str, at: usize| word == lemma[at] || word == surface[at];
    let mut next_allowed: HashMap<u32, usize> = HashMap::new();
    for i in 0..lemma.len() {
        // Candidate probes are those whose first word matches either
        // stream here; when lemma and surface agree, the two lookups
        // return the same list and the duplicate-skip below drops the
        // repeat.
        let by_lemma = index.get(&lemma[i]).map_or(&[][..], Vec::as_slice);
        let by_surface = if surface[i] == lemma[i] {
            &[]
        } else {
            index.get(&surface[i]).map_or(&[][..], Vec::as_slice)
        };
        for &probe_idx in by_lemma.iter().chain(by_surface) {
            let probe = &probes[probe_idx as usize];
            let plen = probe.words.len();
            if i + plen > lemma.len() {
                continue;
            }
            if let Some(&allowed_from) = next_allowed.get(&probe_idx)
                && i < allowed_from
            {
                continue;
            }
            if probe
                .words
                .iter()
                .enumerate()
                .all(|(j, word)| word_matches(word, i + j))
            {
                let entry = &mut counts[probe_idx as usize];
                if is_llm {
                    entry.0 += 1;
                } else {
                    entry.1 += 1;
                }
                next_allowed.insert(probe_idx, i + plen);
            }
        }
    }
}

/// One sentence's parallel lemma/surface token streams, same length, one
/// entry per tagged token.
struct SentenceTokens {
    lemma: Vec<String>,
    surface: Vec<String>,
}

/// Extracts `document`'s prose, splits it into sentences, and tags each
/// one, producing one [`SentenceTokens`] per non-empty sentence.
fn tokenize_document(document: &Document, tagger: &dyn Tagger) -> Vec<SentenceTokens> {
    let mut out = Vec::new();
    for unit in document.prose() {
        let Ok(text) = document.text(&unit.range) else {
            continue;
        };
        for sentence in friction_harness::clean::split_sentences(text) {
            let sentence_tags = tagger.tag(sentence, 0);
            if sentence_tags.is_empty() {
                continue;
            }
            let mut lemma = Vec::with_capacity(sentence_tags.len());
            let mut surface = Vec::with_capacity(sentence_tags.len());
            for token in &sentence_tags {
                lemma.push(token.lemma.to_lowercase());
                surface.push(sentence[token.token.range.clone()].to_lowercase());
            }
            out.push(SentenceTokens { lemma, surface });
        }
    }
    out
}

/// The full corpus pass's result: token totals and doc counts per side,
/// plus every probe's raw occurrence counts.
struct CorpusScan {
    m_total: u64,
    h_total: u64,
    doc_count_llm: usize,
    doc_count_human: usize,
    probe_counts: Vec<(u64, u64)>,
}

/// Reads and tokenizes every selected doc, scanning each sentence's
/// paired lemma/surface streams against the shared probe index.
fn scan_corpus(
    corpus_dir: &Path,
    docs: &[&ManifestRecord],
    tagger: &dyn Tagger,
    probes: &[ProbeEntry],
    index: &HashMap<String, Vec<u32>>,
) -> anyhow::Result<CorpusScan> {
    let mut scan = CorpusScan {
        m_total: 0,
        h_total: 0,
        doc_count_llm: 0,
        doc_count_human: 0,
        probe_counts: vec![(0, 0); probes.len()],
    };
    for record in docs {
        let path = corpus_dir.join(relpath(record));
        let document = load_document(&path, &record.id)?;
        let is_llm = record.class == Class::Llm;
        match record.class {
            Class::Llm => scan.doc_count_llm += 1,
            Class::Human => scan.doc_count_human += 1,
        }
        for sentence in tokenize_document(&document, tagger) {
            let len = sentence.lemma.len() as u64;
            if is_llm {
                scan.m_total += len;
            } else {
                scan.h_total += len;
            }
            scan_sentence(
                &sentence.lemma,
                &sentence.surface,
                index,
                probes,
                is_llm,
                &mut scan.probe_counts,
            );
        }
    }
    Ok(scan)
}

fn per_million(count: u64, total: u64) -> f64 {
    if total == 0 {
        0.0
    } else {
        #[allow(
            clippy::cast_precision_loss,
            reason = "corpus counts are far below 2^52"
        )]
        {
            count as f64 / total as f64 * 1_000_000.0
        }
    }
}

/// Sums every rule's probe counts and classifies the result — `NoEvidence`
/// or a real [`Verdict`] for a probeable rule, [`Measured::Unprobeable`]
/// otherwise.
fn aggregate_measurements(
    rows: &[RuleRow],
    probes: &[ProbeEntry],
    probe_counts: &[(u64, u64)],
    m_total: u64,
    h_total: u64,
) -> Vec<Measured> {
    let mut rule_m = vec![0u64; rows.len()];
    let mut rule_h = vec![0u64; rows.len()];
    for (probe, &(m, h)) in probes.iter().zip(probe_counts) {
        rule_m[probe.rule_idx] += m;
        rule_h[probe.rule_idx] += h;
    }

    rows.iter()
        .enumerate()
        .map(|(idx, row)| {
            if !row.probeable {
                return Measured::Unprobeable;
            }
            let m_count = rule_m[idx];
            let h_count = rule_h[idx];
            let verdict = classify(row.kind, m_count, h_count, m_total, h_total);
            Measured::Evidence {
                verdict,
                m_count,
                h_count,
                m_per_million: per_million(m_count, m_total),
                h_per_million: per_million(h_count, h_total),
            }
        })
        .collect()
}

// --- report model ---

#[derive(Debug, Serialize)]
struct TokenTotals {
    llm: u64,
    human: u64,
}

#[derive(Debug, Serialize)]
struct DocCounts {
    llm: usize,
    human: usize,
}

#[derive(Debug, Serialize)]
struct MatrixCell {
    recorded: String,
    measured: String,
    count: u64,
}

#[derive(Debug, Serialize)]
struct BucketSummary {
    bucket: &'static str,
    rule_count: usize,
    matrix: Vec<MatrixCell>,
}

#[derive(Debug, Serialize)]
struct MovementRow {
    id: String,
    bucket: &'static str,
    recorded: String,
    measured: String,
    m_count: Option<u64>,
    h_count: Option<u64>,
    m_per_million: Option<f64>,
    h_per_million: Option<f64>,
}

#[derive(Debug, Serialize)]
struct AdjudicateReport {
    rules_path: String,
    corpus_dir: String,
    split: &'static str,
    token_totals: TokenTotals,
    doc_counts: DocCounts,
    buckets: Vec<BucketSummary>,
    movements: Vec<MovementRow>,
    agreement_numerator: u64,
    agreement_denominator: u64,
}

/// Builds the per-bucket recorded-vs-measured matrix and the movements
/// list, in deterministic (bucket-then-rule) order.
fn build_report(
    args: &Args,
    rows: &[RuleRow],
    measured: &[Measured],
    scan: &CorpusScan,
) -> AdjudicateReport {
    let mut buckets = Vec::with_capacity(Bucket::ALL.len());
    for bucket in Bucket::ALL {
        let mut matrix: BTreeMap<(String, String), u64> = BTreeMap::new();
        let mut rule_count = 0usize;
        for (row, outcome) in rows.iter().zip(measured) {
            if row.bucket != bucket {
                continue;
            }
            rule_count += 1;
            *matrix
                .entry((row.recorded_label.clone(), outcome.label().to_string()))
                .or_insert(0) += 1;
        }
        buckets.push(BucketSummary {
            bucket: bucket.name(),
            rule_count,
            matrix: matrix
                .into_iter()
                .map(|((recorded, measured), count)| MatrixCell {
                    recorded,
                    measured,
                    count,
                })
                .collect(),
        });
    }

    let mut movements = Vec::new();
    let mut agreement_numerator = 0u64;
    for (row, outcome) in rows.iter().zip(measured) {
        let measured_label = outcome.label();
        if measured_label == row.recorded_label {
            agreement_numerator += 1;
            continue;
        }
        let (m_count, h_count, m_per_million, h_per_million) = match outcome {
            Measured::Unprobeable => (None, None, None, None),
            Measured::Evidence {
                m_count,
                h_count,
                m_per_million,
                h_per_million,
                ..
            } => (
                Some(*m_count),
                Some(*h_count),
                Some(*m_per_million),
                Some(*h_per_million),
            ),
        };
        movements.push(MovementRow {
            id: row.id.clone(),
            bucket: row.bucket.name(),
            recorded: row.recorded_label.clone(),
            measured: measured_label.to_string(),
            m_count,
            h_count,
            m_per_million,
            h_per_million,
        });
    }

    AdjudicateReport {
        rules_path: args.rules.display().to_string(),
        corpus_dir: args.corpus_dir.display().to_string(),
        split: "train+dev",
        token_totals: TokenTotals {
            llm: scan.m_total,
            human: scan.h_total,
        },
        doc_counts: DocCounts {
            llm: scan.doc_count_llm,
            human: scan.doc_count_human,
        },
        buckets,
        movements,
        agreement_numerator,
        agreement_denominator: u64::try_from(rows.len()).unwrap_or(u64::MAX),
    }
}

fn render_text(report: &AdjudicateReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "adjudicate: rules={} corpus={} split={}",
        report.rules_path, report.corpus_dir, report.split
    )
    .expect("write to String is infallible");
    writeln!(
        out,
        "tokens: llm={} human={}",
        report.token_totals.llm, report.token_totals.human
    )
    .expect("write to String is infallible");
    writeln!(
        out,
        "docs: llm={} human={}",
        report.doc_counts.llm, report.doc_counts.human
    )
    .expect("write to String is infallible");

    for bucket in &report.buckets {
        writeln!(out).expect("write to String is infallible");
        writeln!(
            out,
            "--- {} ({} rule(s)) ---",
            bucket.bucket, bucket.rule_count
        )
        .expect("write to String is infallible");
        writeln!(out, "recorded -> measured : count").expect("write to String is infallible");
        for cell in &bucket.matrix {
            writeln!(
                out,
                "  {} -> {} : {}",
                cell.recorded, cell.measured, cell.count
            )
            .expect("write to String is infallible");
        }
    }

    writeln!(out).expect("write to String is infallible");
    writeln!(out, "--- movements ({}) ---", report.movements.len())
        .expect("write to String is infallible");
    for row in &report.movements {
        let counts = match (
            row.m_count,
            row.h_count,
            row.m_per_million,
            row.h_per_million,
        ) {
            (Some(m), Some(h), Some(m_pm), Some(h_pm)) => {
                format!("m={m} h={h} m_pm={m_pm:.2} h_pm={h_pm:.2}")
            }
            _ => "m=n/a h=n/a".to_string(),
        };
        writeln!(
            out,
            "  {} [{}] recorded={} measured={} {}",
            row.id, row.bucket, row.recorded, row.measured, counts
        )
        .expect("write to String is infallible");
    }

    writeln!(out).expect("write to String is infallible");
    #[allow(clippy::cast_precision_loss, reason = "rule counts are far below 2^52")]
    let percent = if report.agreement_denominator == 0 {
        0.0
    } else {
        report.agreement_numerator as f64 / report.agreement_denominator as f64 * 100.0
    };
    writeln!(
        out,
        "agreement: {}/{} ({percent:.1}%)",
        report.agreement_numerator, report.agreement_denominator
    )
    .expect("write to String is infallible");

    out
}

/// Runs `adjudicate`.
///
/// # Errors
///
/// Returns an error if `--rules` can't be read or doesn't parse as a
/// [`FrameRuleSet`], if the manifest or a referenced document can't be
/// read or parsed, or if the embedded tagger model fails to load.
pub fn run(args: &Args) -> anyhow::Result<()> {
    let rules_text = std::fs::read_to_string(&args.rules)
        .with_context(|| format!("adjudicate: failed to read {}", args.rules.display()))?;
    let rule_set = FrameRuleSet::parse(&rules_text)
        .map_err(|e| anyhow::anyhow!("adjudicate: {} did not parse: {e}", args.rules.display()))?;

    let manifest_path = args.corpus_dir.join("manifest.jsonl");
    let records = manifest::read_manifest(&manifest_path)?.unwrap_or_default();
    let mut docs: Vec<&ManifestRecord> = records
        .iter()
        .filter(|r| matches!(r.split, Some(Split::Train | Split::Dev)))
        .collect();
    docs.sort_by(|a, b| a.id.cmp(&b.id));

    let tagger = PerceptronTagger::new()
        .context("adjudicate: failed to load the embedded English tagger model")?;

    let (rows, probes) = build_rows_and_probes(&rule_set);
    let index = build_index(&probes);

    let scan = scan_corpus(&args.corpus_dir, &docs, &tagger, &probes, &index)?;
    let measured = aggregate_measurements(
        &rows,
        &probes,
        &scan.probe_counts,
        scan.m_total,
        scan.h_total,
    );
    let report = build_report(args, &rows, &measured, &scan);

    match args.format {
        OutputFormat::Text => print!("{}", render_text(&report)),
        OutputFormat::Json => println!(
            "{}",
            serde_json::to_string_pretty(&report)
                .expect("AdjudicateReport serialization is infallible")
        ),
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn probe(rule_idx: usize, words: &[&str]) -> ProbeEntry {
        ProbeEntry {
            rule_idx,
            words: words.iter().map(|s| (*s).to_string()).collect(),
        }
    }

    fn tokens(words: &[&str]) -> Vec<String> {
        words.iter().map(|s| (*s).to_string()).collect()
    }

    // --- scan_sentence: probe counting ---

    #[test]
    fn scan_sentence_counts_non_overlapping_leftmost_matches() {
        // "use use use use" against probe ["use", "use"]: greedy
        // non-overlapping consumption finds matches at 0..2 and 2..4,
        // never a third at 1..3.
        let probes = vec![probe(0, &["use", "use"])];
        let index = build_index(&probes);
        let sentence = tokens(&["use", "use", "use", "use"]);
        let mut counts = vec![(0u64, 0u64)];
        scan_sentence(&sentence, &sentence, &index, &probes, true, &mut counts);
        assert_eq!(counts[0], (2, 0));
    }

    #[test]
    fn scan_sentence_never_bridges_two_separate_calls() {
        // Two calls stand in for two different sentences: "please" ends
        // one, "use" starts the next. A probe spanning both words must
        // never match, because per-probe match state never survives a
        // call boundary.
        let probes = vec![probe(0, &["please", "use"])];
        let index = build_index(&probes);
        let mut counts = vec![(0u64, 0u64)];
        let first = tokens(&["please"]);
        let second = tokens(&["use"]);
        scan_sentence(&first, &first, &index, &probes, true, &mut counts);
        scan_sentence(&second, &second, &index, &probes, true, &mut counts);
        assert_eq!(counts[0], (0, 0));
    }

    #[test]
    fn scan_sentence_attributes_counts_to_the_matching_side() {
        let probes = vec![probe(0, &["leverage"])];
        let index = build_index(&probes);
        let mut counts = vec![(0u64, 0u64)];
        let sentence = tokens(&["leverage"]);
        scan_sentence(&sentence, &sentence, &index, &probes, false, &mut counts);
        assert_eq!(counts[0], (0, 1));
    }

    #[test]
    fn scan_sentence_matches_on_either_lemma_or_surface() {
        // Probe "utilizing" is an inflected surface: the lemma stream
        // holds "utilize" and cannot match it, the surface stream can —
        // one occurrence, counted once, not twice.
        let probes = vec![probe(0, &["utilizing"]), probe(1, &["utilize"])];
        let index = build_index(&probes);
        let lemma = tokens(&["utilize"]);
        let surface = tokens(&["utilizing"]);
        let mut counts = vec![(0u64, 0u64); 2];
        scan_sentence(&lemma, &surface, &index, &probes, true, &mut counts);
        assert_eq!(counts[0], (1, 0), "surface-form probe matches via surface");
        assert_eq!(counts[1], (1, 0), "lemma probe matches via lemma");
    }

    // --- flatten_probe ---

    #[test]
    fn flatten_probe_splits_a_multi_word_literal_and_lowercases() {
        assert_eq!(
            flatten_probe(&["Is A Testament To".to_string()]),
            vec!["is", "a", "testament", "to"]
        );
        assert_eq!(
            flatten_probe(&["important".to_string(), "to".to_string()]),
            vec!["important", "to"]
        );
    }

    // --- normalize_recorded_verdict ---

    #[test]
    fn normalize_recorded_verdict_handles_every_shape_observed_in_the_pack() {
        assert_eq!(normalize_recorded_verdict("CONFIRMED"), "confirmed");
        assert_eq!(normalize_recorded_verdict("frame:CONFIRMED"), "confirmed");
        assert_eq!(
            normalize_recorded_verdict("DIRECTION-ERROR"),
            "direction-error"
        );
        assert_eq!(
            normalize_recorded_verdict("frame:DIRECTION-ERROR"),
            "direction-error"
        );
        assert_eq!(
            normalize_recorded_verdict("frame:DIRECTION-ERROR(human-tilted)"),
            "direction-error"
        );
        assert_eq!(
            normalize_recorded_verdict("frame:guard-WRONG(machine-tilted)"),
            "guard-wrong"
        );
        assert_eq!(normalize_recorded_verdict("guard-WRONG"), "guard-wrong");
        assert_eq!(
            normalize_recorded_verdict("guard-confirmed"),
            "guard-confirmed"
        );
        assert_eq!(
            normalize_recorded_verdict("frame:guard-confirmed"),
            "guard-confirmed"
        );
        assert_eq!(normalize_recorded_verdict("no-evidence"), "no-evidence");
        assert_eq!(
            normalize_recorded_verdict("frame:no-evidence"),
            "no-evidence"
        );
        assert_eq!(normalize_recorded_verdict("weak"), "weak");
        assert_eq!(normalize_recorded_verdict("frame:weak"), "weak");
        assert_eq!(
            normalize_recorded_verdict("frame:weak-differential"),
            "weak"
        );
    }

    // --- classify() end-to-end sanity, through this module's own label
    // mapping (not just friction_packs's own tests) ---

    #[test]
    fn classify_confirms_a_rewrite_at_the_documented_ratio() {
        let verdict = classify(RuleKind::Rewrite, 10, 2, 1_000_000, 1_000_000);
        assert_eq!(verdict, Verdict::Confirmed);
        assert_eq!(verdict_label(verdict), "confirmed");
    }

    #[test]
    fn classify_reports_no_evidence_when_both_sides_are_silent() {
        let verdict = classify(RuleKind::Rewrite, 0, 0, 1_000_000, 1_000_000);
        assert_eq!(verdict, Verdict::NoEvidence);
        assert_eq!(verdict_label(verdict), "no-evidence");
    }

    #[test]
    fn classify_confirms_a_guard_whose_human_side_is_at_least_as_dense() {
        let verdict = classify(RuleKind::Guard, 5, 5, 1_000_000, 1_000_000);
        assert_eq!(verdict, Verdict::GuardConfirmed);
        assert_eq!(verdict_label(verdict), "guard-confirmed");
    }
}
