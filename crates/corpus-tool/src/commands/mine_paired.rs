//! `corpus-tool mine-paired` — ratio-threshold literal n-gram mining over
//! the stock/antislop paired corpus.
//!
//! Reads `corpus/paired/manifest.jsonl` (see `commands::generate_paired`),
//! with a read-only human-train cross-check column.
//!
//! # Scope
//!
//! Orders 2..4 only — no skeleton, block-position, or light-verb-
//! construction pass, unlike `mine-inventory`. A separate subcommand
//! (rather than a flag on `mine-inventory`) because the two commands
//! read structurally different corpora — `corpus/paired/manifest.jsonl`
//! keyed by [`Side`], not `corpus/manifest.jsonl` keyed by `Class` — and
//! this command's job is deliberately narrower.
//!
//! # Why the antislop-frequency gate, and why it's secondary
//!
//! Unlike `mine-inventory` (where the machine and human corpora are
//! topic-independent, so a human-corpus frequency gate is the primary
//! defense against a rare topic noun dominating a score), stock and
//! antislop here answer the *identical* prompt under the *identical*
//! derived seed — topic vocabulary already cancels at the per-prompt
//! level, and pooling across every pair plus the existing stock-count
//! floors (which require a phrase to recur across multiple prompts)
//! further blocks a one-prompt topic noun from surfacing. The
//! per-token antislop-frequency gate (`--min-antislop-token-freq`,
//! default 8, provisional — scaled down from `mine-inventory`'s
//! `min_human_token_freq=25` for the much smaller paired corpus,
//! pending recalibration once the real paired corpus exists) is
//! therefore a secondary robustness net, not the primary
//! topic-cancellation mechanism. Antislop is the reference here (not
//! the human corpus) because it's the best available "how this exact
//! prompt set reads when competently written, minus the tells we're
//! trying to catch": the human corpus covers different topics and
//! genres entirely, and requiring literal attestation in the small
//! human train set would filter out real style tells that happen not to
//! occur there.
//!
//! # The human-train cross-check
//!
//! A read-only pass over `corpus/manifest.jsonl`, restricted to
//! `Class::Human && Split::Train` docs, never touches `corpus/llm`, and
//! never contributes to scoring — it exists purely so every surfaced
//! n-gram carries a `human_train_count` column, the report's only bridge
//! back to real human text. Per the project's own rule that the human
//! side of any substitution pair or frequency-band judgment must come
//! exclusively from the real human corpus, antislop output is never
//! treated as human text or as a source of replacement text. See the
//! banner on the antislop-favored table.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use clap::Args as ClapArgs;

use crate::commands::mine::{ClassCounts, accumulate_ngrams};
use crate::commands::mine_inventory::word_runs;
use crate::commands::ngram_mining;
use crate::manifest::{self, Class, ManifestRecord, Split};
use crate::metric_source::load_document;
use crate::paired_manifest::{self, PairedManifestRecord, Side};

/// Arguments for `corpus-tool mine-paired`.
#[derive(Debug, ClapArgs)]
pub struct Args {
    /// Corpus root directory: read-only, for the human-train
    /// cross-check pass only (see the module docs).
    #[arg(long, default_value = "corpus")]
    pub corpus_dir: PathBuf,
    /// Paired corpus root directory.
    #[arg(long, default_value = "corpus/paired")]
    pub paired_dir: PathBuf,
    /// Path to write the markdown mining report to.
    #[arg(long)]
    pub report: PathBuf,
    /// The `eps` smoothing constant in the ratio-mining score. Shares
    /// its default with `mine-inventory`'s own `--eps`.
    #[arg(long, default_value_t = 0.4)]
    pub eps: f64,
    /// Minimum antislop-side per-token frequency every token of a
    /// candidate n-gram must reach to be scored at all. Provisional:
    /// see the module docs.
    #[arg(long, default_value_t = 8)]
    pub min_antislop_token_freq: u64,
    /// Minimum stock-side occurrence count for a 2-gram to be scored.
    #[arg(long, default_value_t = 8)]
    pub min_stock_count_n2: u64,
    /// Minimum stock-side occurrence count for a 3-gram to be scored.
    #[arg(long, default_value_t = 5)]
    pub min_stock_count_n3: u64,
    /// Minimum stock-side occurrence count for a 4-gram to be scored.
    #[arg(long, default_value_t = 4)]
    pub min_stock_count_n4: u64,
    /// How many entries to report per order/direction.
    #[arg(long, default_value_t = 50)]
    pub top: usize,
}

/// Minimum stock-side occurrence count for n-gram order `order` (2, 3,
/// or 4); `0` for any other order.
const fn min_stock_count(order: usize, args: &Args) -> u64 {
    match order {
        2 => args.min_stock_count_n2,
        3 => args.min_stock_count_n3,
        4 => args.min_stock_count_n4,
        _ => 0,
    }
}

/// Runs `mine-paired`.
///
/// # Errors
///
/// Returns an error if the paired manifest, a referenced paired doc, the
/// (optional) `corpus/manifest.jsonl`, or a referenced human-train doc
/// can't be read or parsed, or if `--report` can't be written.
pub fn run(args: &Args) -> anyhow::Result<()> {
    let paired_manifest_path = args.paired_dir.join("manifest.jsonl");
    let records = paired_manifest::read_paired_manifest(&paired_manifest_path)?.unwrap_or_default();

    let (stock_docs, antislop_docs, sentences) = collect_paired_sentences(args, &records)?;
    let counts = accumulate_by_order(&sentences);
    let human_train_counts = human_train_ngram_counts(&args.corpus_dir)?;

    let orders: Vec<PairedOrderReport> = [2, 3, 4]
        .into_iter()
        .map(|order| build_paired_order_report(order, &counts, &human_train_counts, args))
        .collect();

    let header = ReportHeader {
        stock_docs,
        antislop_docs,
        args: ArgsSnapshot::from(args),
    };
    let report = render_report(&header, &orders);

    if let Some(parent) = args.report.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&args.report, &report)?;
    println!("mine-paired: wrote report to {}", args.report.display());
    Ok(())
}

/// One pooled sentence, tagged with which side produced it.
struct SentenceRecord {
    side: Side,
    text: String,
}

/// Reads and parses every doc in `records` (stock side fully, in
/// `(genre, pair_id)` order, then antislop side fully, in the same
/// order: deterministic traversal), pooling their sentences.
///
/// Returns `(stock_doc_count, antislop_doc_count, sentences)`.
fn collect_paired_sentences(
    args: &Args,
    records: &[PairedManifestRecord],
) -> anyhow::Result<(usize, usize, Vec<SentenceRecord>)> {
    let mut sorted: Vec<&PairedManifestRecord> = records.iter().collect();
    sorted.sort_by(|a, b| {
        (a.side, a.genre, a.pair_id.as_str()).cmp(&(b.side, b.genre, b.pair_id.as_str()))
    });

    let mut stock_docs = 0usize;
    let mut antislop_docs = 0usize;
    let mut sentences = Vec::new();

    for record in sorted {
        let path = args
            .paired_dir
            .join(paired_manifest::paired_relpath(record));
        let document = load_document(&path, &record.pair_id)?;
        match record.side {
            Side::Stock => stock_docs += 1,
            Side::Antislop => antislop_docs += 1,
        }

        for unit in document.prose() {
            let Ok(text) = document.text(&unit.range) else {
                continue;
            };
            for sentence in friction_harness::clean::split_sentences(text) {
                sentences.push(SentenceRecord {
                    side: record.side,
                    text: sentence.to_string(),
                });
            }
        }
    }
    Ok((stock_docs, antislop_docs, sentences))
}

/// Accumulates per-order (1..=4) n-gram counts over `sentences` into one
/// [`ClassCounts`] per order — `counts.llm` = stock, `counts.human` =
/// antislop.
///
/// This is a deliberate reuse of `ClassCounts`'s documented generic
/// "for any two classes" contract (see `commands::mine::ClassCounts`),
/// not a naming mistake: the paired corpus has no `llm`/`human` classes
/// of its own at all (see `commands::generate_paired`'s module docs on
/// why this corpus is never manifest-tracked).
fn accumulate_by_order(sentences: &[SentenceRecord]) -> BTreeMap<usize, ClassCounts> {
    let mut counts: BTreeMap<usize, ClassCounts> = BTreeMap::new();
    for order in 1..=4 {
        counts.insert(order, ClassCounts::default());
    }
    for record in sentences {
        let runs = word_runs(&record.text);
        for order in 1..=4usize {
            let class_counts = counts.get_mut(&order).expect("order was pre-inserted");
            let target = match record.side {
                Side::Stock => &mut class_counts.llm,
                Side::Antislop => &mut class_counts.human,
            };
            accumulate_ngrams(&runs, order, target);
        }
    }
    counts
}

/// Read-only pass over `corpus/manifest.jsonl`, restricted to
/// `Class::Human && Split::Train` docs, producing per-order (2..=4)
/// n-gram counts: the human-train cross-check reference (see the
/// module docs). Never touches `corpus/llm`.
///
/// Returns per-order empty maps (not an error) if `corpus_dir` has no
/// `manifest.jsonl` at all.
fn human_train_ngram_counts(
    corpus_dir: &Path,
) -> anyhow::Result<BTreeMap<usize, BTreeMap<String, u64>>> {
    let manifest_path = corpus_dir.join("manifest.jsonl");
    let records = manifest::read_manifest(&manifest_path)?.unwrap_or_default();

    let mut train: Vec<&ManifestRecord> = records
        .iter()
        .filter(|r| r.class == Class::Human && r.split == Some(Split::Train))
        .collect();
    train.sort_by(|a, b| a.id.cmp(&b.id));

    let mut counts: BTreeMap<usize, BTreeMap<String, u64>> = BTreeMap::new();
    for order in 2..=4 {
        counts.insert(order, BTreeMap::new());
    }

    for record in train {
        let path = corpus_dir.join(crate::corpus_layout::relpath(record));
        let document = load_document(&path, &record.id)?;
        for unit in document.prose() {
            let Ok(text) = document.text(&unit.range) else {
                continue;
            };
            for sentence in friction_harness::clean::split_sentences(text) {
                let runs = word_runs(sentence);
                for order in 2..=4usize {
                    let target = counts.get_mut(&order).expect("order was pre-inserted");
                    accumulate_ngrams(&runs, order, target);
                }
            }
        }
    }
    Ok(counts)
}

/// One scored n-gram entry, with its read-only human-train cross-check
/// count.
#[derive(Debug, Clone)]
struct PairedEntry {
    ngram: String,
    stock_count: u64,
    antislop_count: u64,
    score: f64,
    human_train_count: u64,
}

/// One n-gram order's stock-favored and antislop-favored top lists.
struct PairedOrderReport {
    order: usize,
    total_stock: u64,
    total_antislop: u64,
    stock_favored: Vec<PairedEntry>,
    antislop_favored: Vec<PairedEntry>,
}

/// Builds one order's [`PairedOrderReport`] from pooled `counts` (via
/// `ngram_mining::build_ratio_order_report`, gated on `counts[&1].human`
/// — the antislop unigram table — and this order's own stock-count
/// threshold), attaching each surfaced n-gram's raw `human_train_count`
/// from the read-only cross-check pass.
fn build_paired_order_report(
    order: usize,
    counts: &BTreeMap<usize, ClassCounts>,
    human_train_counts: &BTreeMap<usize, BTreeMap<String, u64>>,
    args: &Args,
) -> PairedOrderReport {
    let class_counts = &counts[&order];
    let antislop_unigram = &counts[&1].human;
    let threshold = min_stock_count(order, args);

    let report = ngram_mining::build_ratio_order_report(
        order,
        class_counts,
        antislop_unigram,
        args.min_antislop_token_freq,
        threshold,
        threshold,
        args.top,
        args.eps,
    );

    let human_counts_for_order = human_train_counts.get(&order);
    let lookup = |ngram: &str| -> u64 {
        human_counts_for_order
            .and_then(|m| m.get(ngram))
            .copied()
            .unwrap_or(0)
    };

    PairedOrderReport {
        order: report.order,
        total_stock: report.total_a,
        total_antislop: report.total_b,
        stock_favored: report
            .a_favored
            .into_iter()
            .map(|e| PairedEntry {
                human_train_count: lookup(&e.ngram),
                ngram: e.ngram,
                stock_count: e.count_a,
                antislop_count: e.count_b,
                score: e.score,
            })
            .collect(),
        antislop_favored: report
            .b_favored
            .into_iter()
            .map(|e| PairedEntry {
                human_train_count: lookup(&e.ngram),
                ngram: e.ngram,
                stock_count: e.count_a,
                antislop_count: e.count_b,
                score: e.score,
            })
            .collect(),
    }
}

// --- report rendering ---

struct ArgsSnapshot {
    eps: f64,
    min_antislop_token_freq: u64,
    min_stock_count_n2: u64,
    min_stock_count_n3: u64,
    min_stock_count_n4: u64,
    top: usize,
}

impl From<&Args> for ArgsSnapshot {
    fn from(args: &Args) -> Self {
        Self {
            eps: args.eps,
            min_antislop_token_freq: args.min_antislop_token_freq,
            min_stock_count_n2: args.min_stock_count_n2,
            min_stock_count_n3: args.min_stock_count_n3,
            min_stock_count_n4: args.min_stock_count_n4,
            top: args.top,
        }
    }
}

struct ReportHeader {
    stock_docs: usize,
    antislop_docs: usize,
    args: ArgsSnapshot,
}

/// The banner over every antislop-favored table: never a source of
/// substitution-pair replacement text, only diagnostic (see the module
/// docs).
const ANTISLOP_FAVORED_BANNER: &str = "antislop-favored entries are diagnostic only — they show \
what the antislop model avoided, never a source of substitution-pair replacement text; any \
replacement text must be hand-picked from corpus/human train docs per inventory-v1.toml's \
curation convention.";

fn render_report(header: &ReportHeader, orders: &[PairedOrderReport]) -> String {
    let mut out = String::new();
    writeln!(out, "# Paired stock/antislop mining report").expect("write to String is infallible");
    writeln!(out).expect("write to String is infallible");
    writeln!(
        out,
        "Stock/antislop paired mining corpus only (`corpus/paired/`, never manifest-tracked, \
         never split). {} stock doc(s), {} antislop doc(s). eps={}, \
         min_antislop_token_freq={} (provisional — scaled down from mine-inventory's \
         min_human_token_freq=25 for the much smaller paired corpus, pending recalibration once \
         the real paired corpus exists), min_stock_count(n2/n3/n4)={}/{}/{}, top={}.",
        header.stock_docs,
        header.antislop_docs,
        header.args.eps,
        header.args.min_antislop_token_freq,
        header.args.min_stock_count_n2,
        header.args.min_stock_count_n3,
        header.args.min_stock_count_n4,
        header.args.top,
    )
    .expect("write to String is infallible");
    writeln!(
        out,
        "The `human_train_count` column is a read-only cross-check against `corpus/human` \
         train-split docs only — it never touches `corpus/llm`, never contributes to scoring, \
         and is this report's only bridge back to real human text."
    )
    .expect("write to String is infallible");

    for order_report in orders {
        render_paired_order(&mut out, order_report);
    }

    out
}

fn render_paired_order(out: &mut String, order_report: &PairedOrderReport) {
    writeln!(out).expect("write to String is infallible");
    writeln!(out, "## {}-gram", order_report.order).expect("write to String is infallible");
    writeln!(
        out,
        "Total n-gram tokens: stock={}, antislop={}.",
        order_report.total_stock, order_report.total_antislop
    )
    .expect("write to String is infallible");

    writeln!(out).expect("write to String is infallible");
    writeln!(out, "### stock-favored").expect("write to String is infallible");
    render_paired_table(out, &order_report.stock_favored);

    writeln!(out).expect("write to String is infallible");
    writeln!(out, "### antislop-favored").expect("write to String is infallible");
    writeln!(out, "{ANTISLOP_FAVORED_BANNER}").expect("write to String is infallible");
    render_paired_table(out, &order_report.antislop_favored);
}

fn render_paired_table(out: &mut String, entries: &[PairedEntry]) {
    writeln!(
        out,
        "| n-gram | stock count | antislop count | score | human_train_count |"
    )
    .expect("write to String is infallible");
    writeln!(out, "|---|---|---|---|---|").expect("write to String is infallible");
    for entry in entries {
        writeln!(
            out,
            "| {} | {} | {} | {:.4} | {} |",
            entry.ngram,
            entry.stock_count,
            entry.antislop_count,
            entry.score,
            entry.human_train_count
        )
        .expect("write to String is infallible");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `accumulate_by_order` labels `Side::Stock` sentences into
    /// `counts.llm` and `Side::Antislop` sentences into `counts.human` —
    /// not swapped.
    #[test]
    fn build_stock_antislop_counts_labels_buckets_correctly() {
        let sentences = vec![
            SentenceRecord {
                side: Side::Stock,
                text: "alpha beta gamma.".to_string(),
            },
            SentenceRecord {
                side: Side::Antislop,
                text: "delta epsilon zeta.".to_string(),
            },
        ];
        let counts = accumulate_by_order(&sentences);
        let order2 = &counts[&2];
        assert_eq!(order2.llm.get("alpha beta").copied(), Some(1));
        assert!(!order2.llm.contains_key("delta epsilon"));
        assert_eq!(order2.human.get("delta epsilon").copied(), Some(1));
        assert!(!order2.human.contains_key("alpha beta"));
    }

    #[test]
    fn min_stock_count_dispatches_by_order() {
        let args = Args {
            corpus_dir: PathBuf::from("corpus"),
            paired_dir: PathBuf::from("corpus/paired"),
            report: PathBuf::from("report.md"),
            eps: 0.4,
            min_antislop_token_freq: 8,
            min_stock_count_n2: 8,
            min_stock_count_n3: 5,
            min_stock_count_n4: 4,
            top: 50,
        };
        assert_eq!(min_stock_count(2, &args), 8);
        assert_eq!(min_stock_count(3, &args), 5);
        assert_eq!(min_stock_count(4, &args), 4);
        assert_eq!(min_stock_count(1, &args), 0);
    }

    /// `render_report` includes the never-a-replacement-source banner on
    /// every antislop-favored section.
    #[test]
    fn render_report_carries_antislop_favored_banner() {
        let header = ReportHeader {
            stock_docs: 0,
            antislop_docs: 0,
            args: ArgsSnapshot {
                eps: 0.4,
                min_antislop_token_freq: 8,
                min_stock_count_n2: 8,
                min_stock_count_n3: 5,
                min_stock_count_n4: 4,
                top: 50,
            },
        };
        let orders = vec![PairedOrderReport {
            order: 2,
            total_stock: 0,
            total_antislop: 0,
            stock_favored: Vec::new(),
            antislop_favored: Vec::new(),
        }];
        let report = render_report(&header, &orders);
        assert!(report.contains("never a source of substitution-pair replacement text"));
    }
}
