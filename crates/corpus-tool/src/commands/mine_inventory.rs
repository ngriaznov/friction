//! `corpus-tool mine-inventory` — ratio-threshold literal n-gram mining,
//! POS-skeleton mining, block-position-conditioned mining, and
//! light-verb-construction pair-rate measurement.
//!
//! All on the TRAIN split only, written as one markdown report for a
//! human curator to hand-transcribe into
//! `crates/friction-packs/packs/inventory-v1.toml`. This tool never
//! writes pack TOML directly — mirroring the already
//! established `mine` -> `mined-ngrams-v1.toml` relationship: a human
//! reads the report and copies selected entries into the pack by hand.
//!
//! # Tokenization convention — a deliberate split from `index`
//!
//! This command extracts prose via `friction-parse` (`document.prose()`,
//! like the existing `mine` command) rather than running
//! `clean()`/`tokenize()` over raw file bytes, for two reasons:
//! block-position-conditioned mining (see
//! [`preview_candidate_block_indices`]) structurally needs the real block
//! tree — there is no way to recover "first block after H1" from a flat
//! regex-cleaned stream — and staying consistent with the convention
//! `mine.rs` already uses is one less inconsistency for a reader to
//! reconcile. Within each prose unit, sentences come from
//! `friction_harness::clean::split_sentences`, and each sentence's tokens
//! from `friction_harness::clean::tokenize` — the canonical shared
//! primitive, used here in place of `mine.rs`'s own hand-rolled
//! `word_segments` (that module is left untouched). `corpus-tool index`, by
//! contrast, tokenizes raw whole-file text with no block-tree extraction at
//! all. See that module's own doc comment for why the two commands
//! deliberately do not share one convention.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::PathBuf;

use anyhow::Context as _;
use clap::Args as ClapArgs;
use friction_core::{Block, BlockKind, Document, TokenKind};
use friction_nlp::lvc::{DERIVATIONAL_LEXICON, LightVerbForm, conjugate, scan_construction_shape};
use friction_nlp::{PerceptronTagger, Tagger};

use crate::commands::mine::{ClassCounts, accumulate_ngrams};
use crate::commands::ngram_mining;
use crate::manifest::{self, Class, ManifestRecord, Split};
use crate::metric_source::load_document;

/// Arguments for `corpus-tool mine-inventory`.
#[derive(Debug, ClapArgs)]
pub struct Args {
    /// Corpus root directory.
    #[arg(long, default_value = "corpus")]
    pub corpus_dir: PathBuf,
    /// Path to write the markdown mining report to.
    #[arg(long)]
    pub report: PathBuf,
    /// The `eps` smoothing constant in the ratio-mining score. Documented
    /// and overridable for calibration only: the shipped default is the
    /// one the algorithm reference validated.
    #[arg(long, default_value_t = 0.4)]
    pub eps: f64,
    /// Minimum human-corpus per-token frequency every token of a
    /// candidate n-gram must reach to be scored at all (drops topic
    /// vocabulary).
    #[arg(long, default_value_t = 25)]
    pub min_human_token_freq: u64,
    /// Minimum machine-side occurrence count for a 2-gram to be scored.
    #[arg(long, default_value_t = 8)]
    pub min_machine_count_n2: u64,
    /// Minimum machine-side occurrence count for a 3-gram to be scored.
    #[arg(long, default_value_t = 5)]
    pub min_machine_count_n3: u64,
    /// Minimum machine-side occurrence count for a 4-gram to be scored.
    #[arg(long, default_value_t = 4)]
    pub min_machine_count_n4: u64,
    /// How many entries to report per family/order/direction.
    #[arg(long, default_value_t = 50)]
    pub top: usize,
    /// Minimum `constr_M` count for a seed LVC pair's per-pair rate to be
    /// reported as reliable, rather than flagged as too sparse to trust.
    #[arg(long, default_value_t = 2)]
    pub min_lvc_count: u64,
    /// Runs the POS-skeleton mining pass (needs the tagger, slower).
    /// Pass `--skeleton false` to skip it for quick iteration.
    ///
    /// Explicitly takes a `true`/`false` value (`action =
    /// clap::ArgAction::Set`) rather than being a bare presence flag —
    /// a plain `bool` field would otherwise auto-infer a "sets true when
    /// present" flag, which can never express turning the pass *off*
    /// from the command line even though it defaults on.
    #[arg(long, action = clap::ArgAction::Set, default_value_t = true)]
    pub skeleton: bool,
}

/// Minimum machine-side occurrence count for n-gram order `order` (2, 3,
/// or 4); `0` for any other order (only used internally for the order-1
/// frequency/verb-count tables, which are never gated by a machine-count
/// threshold).
const fn min_machine_count(order: usize, args: &Args) -> u64 {
    match order {
        2 => args.min_machine_count_n2,
        3 => args.min_machine_count_n3,
        4 => args.min_machine_count_n4,
        _ => 0,
    }
}

/// Runs `mine-inventory`.
///
/// # Errors
///
/// Returns an error if the manifest or a referenced document can't be
/// read or parsed, if the embedded tagger model fails to load, or if
/// `--report` can't be written.
pub fn run(args: &Args) -> anyhow::Result<()> {
    let tagger =
        PerceptronTagger::new().context("failed to load the embedded English tagger model")?;

    let manifest_path = args.corpus_dir.join("manifest.jsonl");
    let records = manifest::read_manifest(&manifest_path)?.unwrap_or_default();

    let mut train: Vec<&ManifestRecord> = records
        .iter()
        .filter(|r| r.split == Some(Split::Train))
        .collect();
    train.sort_by(|a, b| a.id.cmp(&b.id));

    let data = collect_corpus_data(args, &train, &tagger)?;

    // Per-order (1..=4), per-class literal n-gram counts. Order 1 is the
    // human/machine per-token frequency table used to gate every other
    // order, and (for the LVC section) the bare-verb-form occurrence
    // table, never itself reported as its own mined section.
    let counts = accumulate_by_order(&data.sentences, word_runs);
    let literal_orders: Vec<OrderReport> = [2, 3, 4]
        .into_iter()
        .map(|order| build_literal_order_report(order, &counts, args))
        .collect();

    let skeleton_orders: Option<Vec<OrderReport>> = if args.skeleton {
        let skeleton_counts =
            accumulate_by_order(&data.sentences, |s| skeleton_word_runs(s, &tagger));
        Some(
            [2, 3, 4]
                .into_iter()
                .map(|order| build_literal_order_report(order, &skeleton_counts, args))
                .collect(),
        )
    } else {
        None
    };

    let block_position = BlockPositionReport {
        preview: build_block_family_report(&data.preview_candidates, args),
        closing: build_block_family_report(&data.closing_candidates, args),
        preview_overlap: data.preview_overlap,
    };

    let lvc = build_lvc_report(&data.sentences, &counts, &tagger, args);

    let header = ReportHeader {
        train_doc_count: train.len(),
        human_docs: data.human_docs,
        llm_docs: data.llm_docs,
        args: ArgsSnapshot::from(args),
    };

    let report = render_report(
        &header,
        &literal_orders,
        skeleton_orders.as_deref(),
        &block_position,
        &lvc,
    );

    if let Some(parent) = args.report.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&args.report, &report)?;
    println!("mine-inventory: wrote report to {}", args.report.display());
    Ok(())
}

/// Everything a single pass over the TRAIN split collects: pooled
/// sentences (for literal/skeleton/LVC mining) and the block-position
/// candidates (for §3.5), plus their per-doc header lemma overlap.
struct CorpusData {
    human_docs: usize,
    llm_docs: usize,
    sentences: Vec<SentenceRecord>,
    preview_candidates: Vec<BlockCandidate>,
    closing_candidates: Vec<BlockCandidate>,
    preview_overlap: Vec<PreviewOverlapEvidence>,
}

/// Reads and parses every doc in `train`, pooling its sentences and
/// block-position candidates into one [`CorpusData`].
fn collect_corpus_data(
    args: &Args,
    train: &[&ManifestRecord],
    tagger: &dyn Tagger,
) -> anyhow::Result<CorpusData> {
    let mut data = CorpusData {
        human_docs: 0,
        llm_docs: 0,
        sentences: Vec::new(),
        preview_candidates: Vec::new(),
        closing_candidates: Vec::new(),
        preview_overlap: Vec::new(),
    };

    for &record in train {
        let path = args.corpus_dir.join(crate::corpus_layout::relpath(record));
        let document = load_document(&path, &record.id)?;
        match record.class {
            Class::Human => data.human_docs += 1,
            Class::Llm => data.llm_docs += 1,
        }

        for unit in document.prose() {
            let Ok(text) = document.text(&unit.range) else {
                continue;
            };
            for sentence in friction_harness::clean::split_sentences(text) {
                data.sentences.push(SentenceRecord {
                    class: record.class,
                    text: sentence.to_string(),
                });
            }
        }

        collect_block_candidates(
            &document,
            record,
            tagger,
            &mut data.preview_candidates,
            &mut data.closing_candidates,
            &mut data.preview_overlap,
        );
    }
    Ok(data)
}

/// Accumulates per-order (1..=4), per-class n-gram counts over `sentences`,
/// using `runs_for` to turn each sentence's text into word-runs: shared by
/// the literal (§3.3) and POS-skeleton (§3.4) mining passes, which differ
/// only in that one function.
fn accumulate_by_order(
    sentences: &[SentenceRecord],
    mut runs_for: impl FnMut(&str) -> Vec<Vec<String>>,
) -> BTreeMap<usize, ClassCounts> {
    let mut counts: BTreeMap<usize, ClassCounts> = BTreeMap::new();
    for order in 1..=4 {
        counts.insert(order, ClassCounts::default());
    }
    for record in sentences {
        let runs = runs_for(&record.text);
        for order in 1..=4usize {
            let class_counts = counts.get_mut(&order).expect("order was pre-inserted");
            let target = match record.class {
                Class::Human => &mut class_counts.human,
                Class::Llm => &mut class_counts.llm,
            };
            accumulate_ngrams(&runs, order, target);
        }
    }
    counts
}

// --- shared record shapes ---

/// One pooled sentence, tagged with which class produced it.
struct SentenceRecord {
    class: Class,
    text: String,
}

/// One block-position candidate's prose text, tagged with which class
/// produced it.
struct BlockCandidate {
    class: Class,
    text: String,
}

/// One preview-frame candidate's content-lemma overlap with its own
/// document's header text: supporting evidence only, never auto-gated.
struct PreviewOverlapEvidence {
    doc_id: String,
    overlap_lemma_count: usize,
    candidate_lemma_count: usize,
}

// --- tokenization ---

/// Splits `sentence`'s canonical tokens into word-runs: maximal runs of
/// non-punctuation tokens, broken at any [`friction_harness::clean::is_punctuation_token`]
/// boundary. Used by every n-gram-window family in this module.
///
/// `pub(crate)` so `commands::mine_paired` reuses the exact same
/// tokenization convention rather than re-implementing it.
pub(crate) fn word_runs(sentence: &str) -> Vec<Vec<String>> {
    let tokens = friction_harness::clean::tokenize(sentence);
    let mut runs = Vec::new();
    let mut current: Vec<String> = Vec::new();
    for token in tokens {
        if friction_harness::clean::is_punctuation_token(&token) {
            if !current.is_empty() {
                runs.push(std::mem::take(&mut current));
            }
        } else {
            current.push(token);
        }
    }
    if !current.is_empty() {
        runs.push(current);
    }
    runs
}

/// Same word-run shape as [`word_runs`], but every `NN`/`NNS`/`NNP`/`NNPS`-
/// tagged token is replaced by the fixed literal `<NOUN>` (lowercased
/// surface form otherwise passed through verbatim); a run breaks at any
/// tagger-classified [`TokenKind::Punctuation`] token and simply skips
/// (without breaking) any `Number`/`Symbol`/`Whitespace` token.
fn skeleton_word_runs(sentence: &str, tagger: &dyn Tagger) -> Vec<Vec<String>> {
    let tagged_tokens = tagger.tag(sentence, 0);
    let mut runs = Vec::new();
    let mut current: Vec<String> = Vec::new();
    for t in &tagged_tokens {
        match t.token.kind {
            TokenKind::Word => {
                let surface = &sentence[t.token.range.clone()];
                let is_noun = matches!(t.pos.as_str(), "NN" | "NNS" | "NNP" | "NNPS");
                current.push(if is_noun {
                    "<NOUN>".to_string()
                } else {
                    surface.to_lowercase()
                });
            }
            TokenKind::Punctuation => {
                if !current.is_empty() {
                    runs.push(std::mem::take(&mut current));
                }
            }
            TokenKind::Number | TokenKind::Symbol | TokenKind::Whitespace | _ => {}
        }
    }
    if !current.is_empty() {
        runs.push(current);
    }
    runs
}

// --- ratio-threshold scoring (shared by literal and skeleton mining) ---
//
// The scoring core itself (`ratio_score`, the frequency gate, the
// per-order build/sort/truncate logic) lives in `commands::ngram_mining`
// now, shared with `mine-paired` — see that module's doc comment. This
// module keeps only its own report types and the thin adapter that maps
// onto them, so its rendered output (and existing tests) are unaffected.

/// One scored n-gram entry (literal or skeleton mining).
#[derive(Debug, Clone)]
struct LiteralEntry {
    ngram: String,
    count_machine: u64,
    count_human: u64,
    score: f64,
}

/// One n-gram order's machine-favored and human-favored top lists.
struct OrderReport {
    order: usize,
    total_machine: u64,
    total_human: u64,
    machine_favored: Vec<LiteralEntry>,
    human_favored: Vec<LiteralEntry>,
}

/// Builds one order's [`OrderReport`] from pooled per-order `counts`,
/// gating on `counts[&1]`'s per-token human frequency and this order's
/// own machine-count threshold (mirrored onto the human side for the
/// human-favored direction, per the module docs).
///
/// A thin wrapper over `ngram_mining::build_ratio_order_report`
/// (`counts.llm` = "a" = machine, `counts.human` = "b" = human, per
/// that module's own documented generic-reuse contract): pure
/// extraction, so this produces byte-identical output to the original
/// inline implementation.
fn build_literal_order_report(
    order: usize,
    counts: &BTreeMap<usize, ClassCounts>,
    args: &Args,
) -> OrderReport {
    let class_counts = &counts[&order];
    let human_unigram = &counts[&1].human;
    let threshold = min_machine_count(order, args);

    let report = ngram_mining::build_ratio_order_report(
        order,
        class_counts,
        human_unigram,
        args.min_human_token_freq,
        threshold,
        threshold,
        args.top,
        args.eps,
    );

    OrderReport {
        order: report.order,
        total_machine: report.total_a,
        total_human: report.total_b,
        machine_favored: report
            .a_favored
            .into_iter()
            .map(|e| LiteralEntry {
                ngram: e.ngram,
                count_machine: e.count_a,
                count_human: e.count_b,
                score: e.score,
            })
            .collect(),
        human_favored: report
            .b_favored
            .into_iter()
            .map(|e| LiteralEntry {
                ngram: e.ngram,
                count_machine: e.count_a,
                count_human: e.count_b,
                score: e.score,
            })
            .collect(),
    }
}

// --- block-position-conditioned mining ---

/// For each `Heading { level: 1 }` block, the index of the nearest
/// following `Paragraph` block in source order: skipped if no paragraph
/// appears before the next heading (of any level) or before the document
/// ends.
fn preview_candidate_block_indices(blocks: &[Block]) -> Vec<usize> {
    let mut out = Vec::new();
    for (i, block) in blocks.iter().enumerate() {
        if !matches!(block.kind, BlockKind::Heading { level: 1 }) {
            continue;
        }
        for (j, candidate) in blocks.iter().enumerate().skip(i + 1) {
            match candidate.kind {
                BlockKind::Paragraph => {
                    out.push(j);
                    break;
                }
                BlockKind::Heading { .. } => break,
                _ => {}
            }
        }
    }
    out
}

/// The index of the document's last `Paragraph` block by `range.start`,
/// independent of nesting, or `None` if the document has no paragraph at
/// all.
fn closing_candidate_block_index(blocks: &[Block]) -> Option<usize> {
    blocks
        .iter()
        .enumerate()
        .filter(|(_, b)| matches!(b.kind, BlockKind::Paragraph))
        .max_by_key(|(_, b)| b.range.start)
        .map(|(i, _)| i)
}

/// Every `Heading` block's index, any level, in source order.
fn heading_block_indices(blocks: &[Block]) -> Vec<usize> {
    blocks
        .iter()
        .enumerate()
        .filter(|(_, b)| matches!(b.kind, BlockKind::Heading { .. }))
        .map(|(i, _)| i)
        .collect()
}

/// The concatenated prose text of every [`friction_core::ProseUnit`]
/// belonging to block `block_index` (a block can yield more than one
/// prose run if excluded content interrupts it — see `friction-parse`'s
/// own docs).
fn prose_text_for_block(document: &Document, block_index: usize) -> String {
    document
        .prose()
        .iter()
        .filter(|unit| unit.block == block_index)
        .filter_map(|unit| document.text(&unit.range).ok())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Content lemmas of `text`: the lemma of every tagged token whose coarse
/// tag starts with `N`, `V`, `J`, or `R` (noun/verb/adjective/adverb):
/// this module's operationalization of "content word" for the header
/// lemma-overlap supporting evidence.
fn content_lemmas(text: &str, tagger: &dyn Tagger) -> BTreeSet<String> {
    tagger
        .tag(text, 0)
        .into_iter()
        .filter(|t| matches!(t.pos.as_str().chars().next(), Some('N' | 'V' | 'J' | 'R')))
        .map(|t| t.lemma.to_string())
        .collect()
}

/// Walks one document's block tree, collecting its preview-frame and
/// closing-frame candidate texts (see the module docs) and, for a
/// preview-frame candidate, its content-lemma overlap with the
/// document's own header text.
fn collect_block_candidates(
    document: &Document,
    record: &ManifestRecord,
    tagger: &dyn Tagger,
    preview_candidates: &mut Vec<BlockCandidate>,
    closing_candidates: &mut Vec<BlockCandidate>,
    preview_overlap: &mut Vec<PreviewOverlapEvidence>,
) {
    let blocks = document.blocks();

    let header_text = heading_block_indices(blocks)
        .into_iter()
        .map(|i| prose_text_for_block(document, i))
        .collect::<Vec<_>>()
        .join(" ");
    let header_lemmas = if header_text.trim().is_empty() {
        BTreeSet::new()
    } else {
        content_lemmas(&header_text, tagger)
    };

    for block_index in preview_candidate_block_indices(blocks) {
        let text = prose_text_for_block(document, block_index);
        if text.trim().is_empty() {
            continue;
        }
        let candidate_lemmas = content_lemmas(&text, tagger);
        let overlap = candidate_lemmas.intersection(&header_lemmas).count();
        preview_overlap.push(PreviewOverlapEvidence {
            doc_id: record.id.clone(),
            overlap_lemma_count: overlap,
            candidate_lemma_count: candidate_lemmas.len(),
        });
        preview_candidates.push(BlockCandidate {
            class: record.class,
            text,
        });
    }

    if let Some(block_index) = closing_candidate_block_index(blocks) {
        let text = prose_text_for_block(document, block_index);
        if !text.trim().is_empty() {
            closing_candidates.push(BlockCandidate {
                class: record.class,
                text,
            });
        }
    }
}

/// One raw (un-gated) n-gram count entry for a block-position family.
#[derive(Clone)]
struct RawCountEntry {
    ngram: String,
    count_machine: u64,
    count_human: u64,
}

/// One block-position family's (preview or closing) raw top-by-count
/// n-gram lists, per order: no frequency or machine-count gate: the
/// candidate pool is small enough that the report states raw counts and
/// leaves gating to the human curator (see the module docs).
struct BlockFamilyReport {
    candidate_count: usize,
    orders: Vec<(usize, Vec<RawCountEntry>, Vec<RawCountEntry>)>,
}

fn build_block_family_report(candidates: &[BlockCandidate], args: &Args) -> BlockFamilyReport {
    let mut counts: BTreeMap<usize, ClassCounts> = BTreeMap::new();
    for order in 2..=4 {
        counts.insert(order, ClassCounts::default());
    }
    for candidate in candidates {
        for sentence in friction_harness::clean::split_sentences(&candidate.text) {
            let runs = word_runs(sentence);
            for order in 2..=4usize {
                let class_counts = counts.get_mut(&order).expect("order was pre-inserted");
                let target = match candidate.class {
                    Class::Human => &mut class_counts.human,
                    Class::Llm => &mut class_counts.llm,
                };
                accumulate_ngrams(&runs, order, target);
            }
        }
    }

    let mut orders = Vec::new();
    for order in [2, 3, 4] {
        let class_counts = &counts[&order];
        let mut vocab: BTreeSet<&str> = BTreeSet::new();
        vocab.extend(class_counts.llm.keys().map(String::as_str));
        vocab.extend(class_counts.human.keys().map(String::as_str));

        let mut entries: Vec<RawCountEntry> = vocab
            .into_iter()
            .map(|ngram| RawCountEntry {
                ngram: ngram.to_string(),
                count_machine: class_counts.llm.get(ngram).copied().unwrap_or(0),
                count_human: class_counts.human.get(ngram).copied().unwrap_or(0),
            })
            .collect();

        let mut by_machine = entries.clone();
        by_machine.sort_by(|a, b| {
            b.count_machine
                .cmp(&a.count_machine)
                .then_with(|| a.ngram.cmp(&b.ngram))
        });
        by_machine.truncate(args.top);

        entries.sort_by(|a, b| {
            b.count_human
                .cmp(&a.count_human)
                .then_with(|| a.ngram.cmp(&b.ngram))
        });
        entries.truncate(args.top);

        orders.push((order, by_machine, entries));
    }

    BlockFamilyReport {
        candidate_count: candidates.len(),
        orders,
    }
}

struct BlockPositionReport {
    preview: BlockFamilyReport,
    closing: BlockFamilyReport,
    preview_overlap: Vec<PreviewOverlapEvidence>,
}

// --- LVC pair licensing ---

/// One seed `(nominalization, derived verb)` pair's measured rate: raw
/// construction and bare-verb-form counts per class, and whether it
/// licenses under the within-class criterion (`constr_M > verb_M` and
/// `verb_H > constr_H`).
struct SeedLvcRow {
    nominalization: String,
    derived_verb: String,
    constr_machine: u64,
    constr_human: u64,
    verb_machine: u64,
    verb_human: u64,
    licenses: bool,
    reliable: bool,
}

/// A new-candidate nominalization discovered by the second-pass suffix
/// scan (not a `DERIVATIONAL_LEXICON` key), with its raw construction
/// counts.
struct CandidateRow {
    candidate: String,
    count_machine: u64,
    count_human: u64,
}

struct LvcReport {
    seed_rows: Vec<SeedLvcRow>,
    aggregate_rate_machine_per_1k: f64,
    aggregate_rate_human_per_1k: f64,
    candidate_rows: Vec<CandidateRow>,
}

/// Nominalizing suffixes the new-candidate scan looks for — `-ing` is
/// deliberately excluded (too noisy: it also marks ordinary gerunds/
/// present participles, not just nominalizations).
const CANDIDATE_SUFFIXES: [&str; 6] = ["tion", "sion", "ment", "ance", "ence", "al"];

fn build_lvc_report(
    sentences: &[SentenceRecord],
    counts: &BTreeMap<usize, ClassCounts>,
    tagger: &dyn Tagger,
    args: &Args,
) -> LvcReport {
    let (seed_rows, aggregate_rate_machine_per_1k, aggregate_rate_human_per_1k) =
        build_seed_lvc_rows(sentences, counts, tagger, args);
    let candidate_rows = discover_new_candidates(sentences, tagger, args);

    LvcReport {
        seed_rows,
        aggregate_rate_machine_per_1k,
        aggregate_rate_human_per_1k,
        candidate_rows,
    }
}

/// Builds the seed-pair rate table (§3.6.1): per-pair construction and
/// bare-verb-form counts, the within-class licensing verdict, and the
/// aggregate machine/human construction rate per 1k tokens.
fn build_seed_lvc_rows(
    sentences: &[SentenceRecord],
    counts: &BTreeMap<usize, ClassCounts>,
    tagger: &dyn Tagger,
    args: &Args,
) -> (Vec<SeedLvcRow>, f64, f64) {
    let seed_noms: BTreeSet<&str> = DERIVATIONAL_LEXICON.keys().copied().collect();
    let mut constr_counts: BTreeMap<String, (u64, u64)> = BTreeMap::new();
    for &nom in &seed_noms {
        constr_counts.insert(nom.to_string(), (0, 0));
    }
    for record in sentences {
        for m in scan_construction_shape(&record.text, tagger, &seed_noms) {
            let entry = constr_counts.entry(m.nominalization).or_insert((0, 0));
            match record.class {
                Class::Llm => entry.0 += 1,
                Class::Human => entry.1 += 1,
            }
        }
    }

    let unigram_machine = &counts[&1].llm;
    let unigram_human = &counts[&1].human;

    let mut seed_rows = Vec::new();
    let mut total_constr_machine = 0u64;
    let mut total_constr_human = 0u64;
    for (&nom, &verb) in DERIVATIONAL_LEXICON.iter() {
        let (constr_machine, constr_human) = constr_counts.get(nom).copied().unwrap_or((0, 0));
        total_constr_machine += constr_machine;
        total_constr_human += constr_human;

        let mut verb_machine = 0u64;
        let mut verb_human = 0u64;
        for form in [
            LightVerbForm::Base,
            LightVerbForm::ThirdSg,
            LightVerbForm::Past,
            LightVerbForm::Gerund,
        ] {
            let inflected = conjugate(verb, form);
            verb_machine += unigram_machine.get(&inflected).copied().unwrap_or(0);
            verb_human += unigram_human.get(&inflected).copied().unwrap_or(0);
        }

        seed_rows.push(SeedLvcRow {
            nominalization: nom.to_string(),
            derived_verb: verb.to_string(),
            constr_machine,
            constr_human,
            verb_machine,
            verb_human,
            licenses: constr_machine > verb_machine && verb_human > constr_human,
            reliable: constr_machine >= args.min_lvc_count,
        });
    }

    let total_machine_tokens: u64 = counts[&1].llm.values().sum();
    let total_human_tokens: u64 = counts[&1].human.values().sum();
    #[allow(clippy::cast_precision_loss)]
    let aggregate_rate_machine_per_1k = if total_machine_tokens > 0 {
        total_constr_machine as f64 / total_machine_tokens as f64 * 1000.0
    } else {
        0.0
    };
    #[allow(clippy::cast_precision_loss)]
    let aggregate_rate_human_per_1k = if total_human_tokens > 0 {
        total_constr_human as f64 / total_human_tokens as f64 * 1000.0
    } else {
        0.0
    };

    (
        seed_rows,
        aggregate_rate_machine_per_1k,
        aggregate_rate_human_per_1k,
    )
}

/// Builds the new-candidate discovery table (§3.6.2): every `NN*`-tagged
/// token not already a `DERIVATIONAL_LEXICON` key and ending in one of
/// [`CANDIDATE_SUFFIXES`], with its raw construction-shape counts.
fn discover_new_candidates(
    sentences: &[SentenceRecord],
    tagger: &dyn Tagger,
    args: &Args,
) -> Vec<CandidateRow> {
    let mut discovered: BTreeSet<String> = BTreeSet::new();
    for record in sentences {
        for tagged in tagger.tag(&record.text, 0) {
            if !matches!(tagged.pos.as_str(), "NN" | "NNS" | "NNP" | "NNPS") {
                continue;
            }
            let surface = &record.text[tagged.token.range.clone()];
            let lower = surface.to_lowercase();
            if DERIVATIONAL_LEXICON.contains_key(lower.as_str()) {
                continue;
            }
            if CANDIDATE_SUFFIXES.iter().any(|suf| lower.ends_with(suf)) {
                discovered.insert(lower);
            }
        }
    }

    let candidate_refs: BTreeSet<&str> = discovered.iter().map(String::as_str).collect();
    let mut candidate_counts: BTreeMap<String, (u64, u64)> = BTreeMap::new();
    for record in sentences {
        for m in scan_construction_shape(&record.text, tagger, &candidate_refs) {
            let entry = candidate_counts.entry(m.nominalization).or_insert((0, 0));
            match record.class {
                Class::Llm => entry.0 += 1,
                Class::Human => entry.1 += 1,
            }
        }
    }

    let mut candidate_rows: Vec<CandidateRow> = candidate_counts
        .into_iter()
        .filter(|&(_, (m, h))| m + h > 0)
        .map(|(candidate, (count_machine, count_human))| CandidateRow {
            candidate,
            count_machine,
            count_human,
        })
        .collect();
    candidate_rows.sort_by(|a, b| {
        (b.count_machine + b.count_human)
            .cmp(&(a.count_machine + a.count_human))
            .then_with(|| a.candidate.cmp(&b.candidate))
    });
    candidate_rows.truncate(args.top);
    candidate_rows
}

// --- report rendering ---

struct ArgsSnapshot {
    eps: f64,
    min_human_token_freq: u64,
    min_machine_count_n2: u64,
    min_machine_count_n3: u64,
    min_machine_count_n4: u64,
    top: usize,
    min_lvc_count: u64,
    skeleton: bool,
}

impl From<&Args> for ArgsSnapshot {
    fn from(args: &Args) -> Self {
        Self {
            eps: args.eps,
            min_human_token_freq: args.min_human_token_freq,
            min_machine_count_n2: args.min_machine_count_n2,
            min_machine_count_n3: args.min_machine_count_n3,
            min_machine_count_n4: args.min_machine_count_n4,
            top: args.top,
            min_lvc_count: args.min_lvc_count,
            skeleton: args.skeleton,
        }
    }
}

struct ReportHeader {
    train_doc_count: usize,
    human_docs: usize,
    llm_docs: usize,
    args: ArgsSnapshot,
}

fn render_report(
    header: &ReportHeader,
    literal_orders: &[OrderReport],
    skeleton_orders: Option<&[OrderReport]>,
    block_position: &BlockPositionReport,
    lvc: &LvcReport,
) -> String {
    let mut out = String::new();
    writeln!(out, "# Inventory mining report").expect("write to String is infallible");
    writeln!(out).expect("write to String is infallible");
    writeln!(
        out,
        "Train split only, pooled across genres. {} document(s) ({} human, {} llm). \
         eps={}, min_human_token_freq={}, min_machine_count(n2/n3/n4)={}/{}/{}, top={}, \
         min_lvc_count={}, skeleton_pass={}.",
        header.train_doc_count,
        header.human_docs,
        header.llm_docs,
        header.args.eps,
        header.args.min_human_token_freq,
        header.args.min_machine_count_n2,
        header.args.min_machine_count_n3,
        header.args.min_machine_count_n4,
        header.args.top,
        header.args.min_lvc_count,
        header.args.skeleton,
    )
    .expect("write to String is infallible");

    writeln!(out).expect("write to String is infallible");
    writeln!(out, "## Literal n-gram mining").expect("write to String is infallible");
    for order_report in literal_orders {
        render_literal_order(&mut out, order_report);
    }

    if let Some(skeleton_orders) = skeleton_orders {
        writeln!(out).expect("write to String is infallible");
        writeln!(out, "## POS-skeleton mining").expect("write to String is infallible");
        writeln!(
            out,
            "Content nouns (`NN`/`NNS`/`NNP`/`NNPS`) are abstracted to the literal `<NOUN>` \
             before n-gram keys are built; every other category passes through verbatim."
        )
        .expect("write to String is infallible");
        for order_report in skeleton_orders {
            render_literal_order(&mut out, order_report);
        }
    }

    writeln!(out).expect("write to String is infallible");
    writeln!(out, "## Block-position-conditioned mining").expect("write to String is infallible");
    render_block_family(
        &mut out,
        &block_position.preview,
        "Preview frame (first paragraph after H1)",
    );
    if !block_position.preview_overlap.is_empty() {
        writeln!(out).expect("write to String is infallible");
        writeln!(
            out,
            "#### Header lemma overlap (supporting evidence, not auto-gated)"
        )
        .expect("write to String is infallible");
        writeln!(
            out,
            "| doc id | overlap lemma count | candidate content-lemma count |"
        )
        .expect("write to String is infallible");
        writeln!(out, "|---|---|---|").expect("write to String is infallible");
        for evidence in &block_position.preview_overlap {
            writeln!(
                out,
                "| {} | {} | {} |",
                evidence.doc_id, evidence.overlap_lemma_count, evidence.candidate_lemma_count
            )
            .expect("write to String is infallible");
        }
    }
    render_block_family(
        &mut out,
        &block_position.closing,
        "Ritual/closing frame (last paragraph)",
    );

    render_lvc_section(&mut out, lvc);

    out
}

fn render_lvc_section(out: &mut String, lvc: &LvcReport) {
    writeln!(out).expect("write to String is infallible");
    writeln!(out, "## Light-verb-construction pairs").expect("write to String is infallible");
    writeln!(
        out,
        "These per-pair and aggregate rates are measured fresh on every run so drift is \
         visible as the corpus grows. They do not currently have to reproduce any historical \
         figure, and a seed pair's presence in the shipped pack does not depend on what this \
         section measures — the seed pairs were validated at the construction level, not by \
         corpus rate."
    )
    .expect("write to String is infallible");
    writeln!(
        out,
        "Aggregate LVC-shape rate: machine {:.4}/1k tokens, human {:.4}/1k tokens.",
        lvc.aggregate_rate_machine_per_1k, lvc.aggregate_rate_human_per_1k
    )
    .expect("write to String is infallible");
    writeln!(out).expect("write to String is infallible");
    writeln!(out, "### Seed pairs").expect("write to String is infallible");
    writeln!(
        out,
        "| nominalization | derived verb | constr_M | constr_H | verb_M | verb_H | licenses | reliable |"
    )
    .expect("write to String is infallible");
    writeln!(out, "|---|---|---|---|---|---|---|---|").expect("write to String is infallible");
    for row in &lvc.seed_rows {
        writeln!(
            out,
            "| {} | {} | {} | {} | {} | {} | {} | {} |",
            row.nominalization,
            row.derived_verb,
            row.constr_machine,
            row.constr_human,
            row.verb_machine,
            row.verb_human,
            row.licenses,
            row.reliable
        )
        .expect("write to String is infallible");
    }

    writeln!(out).expect("write to String is infallible");
    writeln!(out, "### New-candidate discovery").expect("write to String is infallible");
    writeln!(
        out,
        "Candidates for a human to check against a derivational dictionary — never auto-added."
    )
    .expect("write to String is infallible");
    writeln!(out, "| candidate | machine count | human count |")
        .expect("write to String is infallible");
    writeln!(out, "|---|---|---|").expect("write to String is infallible");
    for row in &lvc.candidate_rows {
        writeln!(
            out,
            "| {} | {} | {} |",
            row.candidate, row.count_machine, row.count_human
        )
        .expect("write to String is infallible");
    }
}

fn render_literal_order(out: &mut String, order_report: &OrderReport) {
    writeln!(out).expect("write to String is infallible");
    writeln!(out, "### {}-gram", order_report.order).expect("write to String is infallible");
    writeln!(
        out,
        "Total n-gram tokens: machine={}, human={}.",
        order_report.total_machine, order_report.total_human
    )
    .expect("write to String is infallible");
    writeln!(out).expect("write to String is infallible");
    writeln!(out, "#### machine-favored").expect("write to String is infallible");
    render_literal_table(out, &order_report.machine_favored);
    writeln!(out).expect("write to String is infallible");
    writeln!(out, "#### human-favored").expect("write to String is infallible");
    render_literal_table(out, &order_report.human_favored);
}

fn render_literal_table(out: &mut String, entries: &[LiteralEntry]) {
    writeln!(out, "| n-gram | machine count | human count | score |")
        .expect("write to String is infallible");
    writeln!(out, "|---|---|---|---|").expect("write to String is infallible");
    for entry in entries {
        writeln!(
            out,
            "| {} | {} | {} | {:.4} |",
            entry.ngram, entry.count_machine, entry.count_human, entry.score
        )
        .expect("write to String is infallible");
    }
}

fn render_block_family(out: &mut String, family: &BlockFamilyReport, title: &str) {
    writeln!(out).expect("write to String is infallible");
    writeln!(out, "### {title}").expect("write to String is infallible");
    writeln!(out, "Candidates found: {}.", family.candidate_count)
        .expect("write to String is infallible");
    for (order, by_machine, by_human) in &family.orders {
        writeln!(out).expect("write to String is infallible");
        writeln!(out, "#### {order}-gram (raw counts, not gated)")
            .expect("write to String is infallible");
        writeln!(out, "machine-favored:").expect("write to String is infallible");
        render_raw_count_table(out, by_machine);
        writeln!(out, "human-favored:").expect("write to String is infallible");
        render_raw_count_table(out, by_human);
    }
}

fn render_raw_count_table(out: &mut String, entries: &[RawCountEntry]) {
    writeln!(out, "| n-gram | machine count | human count |")
        .expect("write to String is infallible");
    writeln!(out, "|---|---|---|").expect("write to String is infallible");
    for entry in entries {
        writeln!(
            out,
            "| {} | {} | {} |",
            entry.ngram, entry.count_machine, entry.count_human
        )
        .expect("write to String is infallible");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- word_runs ---

    #[test]
    fn word_runs_splits_on_punctuation_boundaries() {
        let runs = word_runs("It works. Moreover, it helps.");
        assert_eq!(
            runs,
            vec![
                vec!["it".to_string(), "works".to_string()],
                vec!["moreover".to_string()],
                vec!["it".to_string(), "helps".to_string()],
            ]
        );
    }

    #[test]
    fn word_runs_handles_empty_text() {
        assert!(word_runs("").is_empty());
    }

    // --- ratio_score / passes_token_freq_gate (now in `ngram_mining`) ---

    #[test]
    fn ratio_score_matches_hand_computed_example() {
        // count_num=8,total_num=100,count_den=0,total_den=100,eps=0.4:
        // numerator = 8/100 + 0.4/100 = 0.084; denominator = 0/100 +
        // 0.4/100 = 0.004; score = 0.084/0.004 = 21.0.
        let score = ngram_mining::ratio_score(8, 100, 0, 100, 0.4).unwrap();
        assert!((score - 21.0).abs() < 1e-9, "score={score}");
    }

    #[test]
    fn ratio_score_none_when_a_total_is_zero() {
        assert_eq!(ngram_mining::ratio_score(1, 0, 1, 10, 0.4), None);
        assert_eq!(ngram_mining::ratio_score(1, 10, 1, 0, 0.4), None);
    }

    #[test]
    fn passes_human_freq_gate_requires_every_token_above_threshold() {
        let mut freq = BTreeMap::new();
        freq.insert("delve".to_string(), 0u64);
        freq.insert("into".to_string(), 378u64);
        assert!(!ngram_mining::passes_token_freq_gate(
            "delve into",
            &freq,
            25
        ));
        assert!(ngram_mining::passes_token_freq_gate("into", &freq, 25));
    }

    // --- min_machine_count ---

    #[test]
    fn min_machine_count_dispatches_by_order() {
        let args = Args {
            corpus_dir: PathBuf::from("corpus"),
            report: PathBuf::from("report.md"),
            eps: 0.4,
            min_human_token_freq: 25,
            min_machine_count_n2: 8,
            min_machine_count_n3: 5,
            min_machine_count_n4: 4,
            top: 50,
            min_lvc_count: 2,
            skeleton: true,
        };
        assert_eq!(min_machine_count(2, &args), 8);
        assert_eq!(min_machine_count(3, &args), 5);
        assert_eq!(min_machine_count(4, &args), 4);
        assert_eq!(min_machine_count(1, &args), 0);
    }

    // --- block-position helpers ---

    #[test]
    fn preview_candidate_block_indices_finds_paragraph_after_h1() {
        let blocks = vec![
            Block::new(BlockKind::Heading { level: 1 }, 0..5),
            Block::new(BlockKind::Paragraph, 6..20),
            Block::new(BlockKind::Heading { level: 2 }, 21..30),
        ];
        assert_eq!(preview_candidate_block_indices(&blocks), vec![1]);
    }

    #[test]
    fn preview_candidate_block_indices_skips_when_next_block_is_a_heading() {
        let blocks = vec![
            Block::new(BlockKind::Heading { level: 1 }, 0..5),
            Block::new(BlockKind::Heading { level: 2 }, 6..15),
            Block::new(BlockKind::Paragraph, 16..30),
        ];
        assert!(preview_candidate_block_indices(&blocks).is_empty());
    }

    #[test]
    fn preview_candidate_block_indices_empty_when_no_h1() {
        let blocks = vec![
            Block::new(BlockKind::Heading { level: 2 }, 0..5),
            Block::new(BlockKind::Paragraph, 6..20),
        ];
        assert!(preview_candidate_block_indices(&blocks).is_empty());
    }

    #[test]
    fn closing_candidate_block_index_picks_last_paragraph_by_start() {
        let blocks = vec![
            Block::new(BlockKind::Paragraph, 0..5),
            Block::new(BlockKind::Heading { level: 2 }, 6..10),
            Block::new(BlockKind::Paragraph, 11..20),
        ];
        assert_eq!(closing_candidate_block_index(&blocks), Some(2));
    }

    #[test]
    fn closing_candidate_block_index_none_without_paragraphs() {
        let blocks = vec![Block::new(BlockKind::Heading { level: 1 }, 0..5)];
        assert_eq!(closing_candidate_block_index(&blocks), None);
    }
}
