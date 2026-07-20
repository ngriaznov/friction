//! `corpus-tool mine` — mines 1-, 2-, and 3-gram phrases from the
//! TRAIN-split corpus and ranks them by how discriminative they are
//! between `llm` and `human` prose.
//!
//! # Method
//!
//! For every train-split document, `friction-parse` extracts the prose
//! text (markdown syntax stripped, byte-exact — see
//! `friction_parse::parse`). Each prose unit's text is lowercased and
//! split into alphabetic word tokens; any punctuation character (not just
//! whitespace) ends the current n-gram window, so an n-gram never bridges
//! a sentence boundary, a table-cell/list-item split, or a heading — this
//! is a deliberately coarser rule than `friction-metrics`'s own sentence
//! segmentation (which needs the tagger model this tool does not load),
//! adequate for a phrase-mining pass whose output is hand-curated
//! afterward, not scored directly. See [`word_segments`].
//!
//! For each n-gram order (1, 2, or 3) counts are pooled across all genres
//! into two class totals — `llm` and `human` — and every n-gram is scored
//! by the *log-odds ratio with an informative Dirichlet prior*, following
//! Monroe, Colaresi & Quinn (2008) ("Fightin' Words"), eq. 16, with the
//! prior drawn from the two classes' own combined ("background") counts:
//!
//! For n-gram `w`, let `y_llm`/`y_human` be its count in each class and
//! `n_llm`/`n_human` the class's total n-gram-token count (for that
//! order). The prior pseudo-count for `w` is its combined count, `alpha_w
//! = y_llm + y_human`, and the prior's total mass is the combined token
//! count, `alpha_0 = n_llm + n_human` (this is what "informative" means
//! here: the prior is not flat across the vocabulary, it favors words
//! already common in the pooled data). Then
//!
//! ```text
//! delta_w = ln((y_llm + alpha_w) / (n_llm + alpha_0 - y_llm - alpha_w))
//!         - ln((y_human + alpha_w) / (n_human + alpha_0 - y_human - alpha_w))
//!
//! variance_w = 1 / (y_llm + alpha_w) + 1 / (y_human + alpha_w)
//!
//! z_w = delta_w / sqrt(variance_w)
//! ```
//!
//! `z_w > 0` means `w` is llm-favored (its estimated log-odds of
//! appearing in `llm` text exceeds `human` text once smoothed by the
//! prior); `z_w < 0` means human-favored. `z_w`'s magnitude accounts for
//! how much evidence backs the estimate — a rare n-gram seen only a
//! handful of times gets pulled toward zero and given a wide (uncertain)
//! z-score, while a frequent one gets a sharp, confident one — which is
//! the entire point of scoring by z rather than by raw log-odds or raw
//! frequency ratio. See [`log_odds_z`].
//!
//! # Ordering
//!
//! llm-favored output is sorted by `z` descending (ties broken
//! lexicographically ascending on the n-gram text); human-favored output
//! is sorted by `z` ascending — equivalently, by how negative (strongly
//! human-favored) it is, descending — with the same lexicographic
//! tie-break. Both are pure functions of the input counts, so re-running
//! `mine` against the same corpus produces byte-identical report output.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::PathBuf;

use clap::Args as ClapArgs;

use crate::manifest::{self, Class, ManifestRecord, Split};
use crate::metric_source::load_document;

/// Which n-gram order(s) `mine` mines. `All` mines 1, 2, and 3 in one run
/// (the default) so a single invocation produces the full report.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum NgramOrderArg {
    #[value(name = "1")]
    One,
    #[value(name = "2")]
    Two,
    #[value(name = "3")]
    Three,
    All,
}

impl NgramOrderArg {
    /// The concrete n-gram orders this selects, ascending.
    const fn orders(self) -> &'static [usize] {
        match self {
            Self::One => &[1],
            Self::Two => &[2],
            Self::Three => &[3],
            Self::All => &[1, 2, 3],
        }
    }
}

/// Arguments for `corpus-tool mine`.
#[derive(Debug, ClapArgs)]
pub struct Args {
    /// Corpus root directory.
    #[arg(long, default_value = "corpus")]
    pub corpus_dir: PathBuf,
    /// N-gram order(s) to mine.
    #[arg(long, value_enum, default_value_t = NgramOrderArg::All)]
    pub n: NgramOrderArg,
    /// How many n-grams to report per direction (llm-favored,
    /// human-favored), per order.
    #[arg(long, default_value_t = 50)]
    pub top: usize,
    /// Minimum combined (llm + human) occurrence count an n-gram needs to
    /// be scored at all — filters out the long tail of once-seen n-grams
    /// whose z-score is mostly prior-driven noise.
    #[arg(long, default_value_t = 5)]
    pub min_count: u64,
    /// Path to write the markdown mining report to.
    #[arg(long)]
    pub report: PathBuf,
}

/// Runs `mine`.
///
/// Reads the manifest, restricts to `train`-split documents of either
/// class, tokenizes each document's prose (see the module docs), and
/// accumulates per-order, per-class n-gram counts. For each requested
/// order, scores every n-gram meeting `--min-count` via [`log_odds_z`],
/// and writes the top `--top` llm-favored and human-favored n-grams
/// (with counts and z-scores) to `--report` as a markdown report.
///
/// # Errors
///
/// Returns an error if the manifest or any referenced document can't be
/// read or parsed, or if `--report` can't be written.
pub fn run(args: &Args) -> anyhow::Result<()> {
    let manifest_path = args.corpus_dir.join("manifest.jsonl");
    let records = manifest::read_manifest(&manifest_path)?.unwrap_or_default();

    let mut train: Vec<&ManifestRecord> = records
        .iter()
        .filter(|r| r.split == Some(Split::Train))
        .collect();
    train.sort_by(|a, b| a.id.cmp(&b.id));

    let mut human_docs = 0usize;
    let mut llm_docs = 0usize;

    // order -> class -> ngram -> count.
    let mut counts: BTreeMap<usize, ClassCounts> = BTreeMap::new();
    for &order in args.n.orders() {
        counts.insert(order, ClassCounts::default());
    }

    for record in &train {
        let path = args.corpus_dir.join(crate::corpus_layout::relpath(record));
        let document = load_document(&path, &record.id)?;
        match record.class {
            Class::Human => human_docs += 1,
            Class::Llm => llm_docs += 1,
        }

        let mut segments: Vec<Vec<String>> = Vec::new();
        for unit in document.prose() {
            let Ok(text) = document.text(&unit.range) else {
                continue;
            };
            segments.extend(word_segments(text));
        }

        for &order in args.n.orders() {
            let class_counts = counts.get_mut(&order).expect("order was pre-inserted");
            let target = match record.class {
                Class::Human => &mut class_counts.human,
                Class::Llm => &mut class_counts.llm,
            };
            accumulate_ngrams(&segments, order, target);
        }
    }

    let mut orders_report = Vec::new();
    for &order in args.n.orders() {
        let class_counts = &counts[&order];
        let entries = score_entries(class_counts, args.min_count);
        let llm_favored = top_llm_favored(&entries, args.top);
        let human_favored = top_human_favored(&entries, args.top);
        orders_report.push(OrderReport {
            order,
            n_llm: class_counts.llm.values().sum(),
            n_human: class_counts.human.values().sum(),
            vocab_size: entries.len(),
            llm_favored,
            human_favored,
        });
    }

    let header = ReportHeader {
        train_doc_count: train.len(),
        human_docs,
        llm_docs,
        top: args.top,
        min_count: args.min_count,
    };
    let report = render_report(&header, &orders_report);

    if let Some(parent) = args.report.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&args.report, &report)?;
    println!(
        "mine: wrote {} order(s) to {}",
        orders_report.len(),
        args.report.display()
    );
    Ok(())
}

// --- tokenization ---

/// Splits `text` into word segments.
///
/// Each segment is a maximal run of lowercase word tokens with no
/// intervening punctuation (only whitespace between tokens). Punctuation
/// of any kind — sentence-enders, commas, dashes, parens, quotes — ends
/// the current segment, so n-grams built from a segment never cross it; a
/// digit or symbol does the same (this tool mines word phrases, not
/// numbers or code-like tokens).
///
/// Tokenization within a segment mirrors `friction-metrics`'s own word
/// tokenizer: a run of alphabetic characters, with an interior apostrophe
/// (ASCII `'` or the Unicode right single quotation mark `’`, alphabetic
/// on both sides) folded into the word so `"don't"` is one token, not
/// split at the apostrophe. Unlike `friction-metrics`, the apostrophe
/// itself is normalized to ASCII `'` in the emitted token — so `"don't"`
/// and `"don’t"` count as the same n-gram regardless of which quotation
/// style a given document happens to use, which matters here (unlike a
/// per-document metric) since counts are pooled across the whole corpus.
///
/// `pub` so other corpus-scale tooling (e.g. the `self_fingerprint`
/// example, which needs the exact same tokenization for a *different*
/// pair of classes) can reuse this instead of re-implementing it.
#[must_use]
pub fn word_segments(text: &str) -> Vec<Vec<String>> {
    let chars: Vec<char> = text.chars().collect();
    let mut segments = Vec::new();
    let mut current_segment: Vec<String> = Vec::new();
    let mut current_word = String::new();

    for (i, &c) in chars.iter().enumerate() {
        let is_apostrophe = c == '\'' || c == '\u{2019}';
        let is_interior_apostrophe = is_apostrophe
            && i > 0
            && chars[i - 1].is_alphabetic()
            && chars.get(i + 1).is_some_and(|next| next.is_alphabetic());

        if c.is_alphabetic() {
            current_word.push(c.to_ascii_lowercase());
            continue;
        }
        if is_interior_apostrophe {
            current_word.push('\'');
            continue;
        }
        if !current_word.is_empty() {
            current_segment.push(std::mem::take(&mut current_word));
        }
        if !c.is_whitespace() && !current_segment.is_empty() {
            segments.push(std::mem::take(&mut current_segment));
        }
    }
    if !current_word.is_empty() {
        current_segment.push(current_word);
    }
    if !current_segment.is_empty() {
        segments.push(current_segment);
    }
    segments
}

/// Adds every `order`-length window of every segment in `segments` to
/// `counts` (n-gram text -> occurrence count), space-joined.
///
/// `pub` — see [`word_segments`]'s doc comment for why.
pub fn accumulate_ngrams(
    segments: &[Vec<String>],
    order: usize,
    counts: &mut BTreeMap<String, u64>,
) {
    for segment in segments {
        if segment.len() < order {
            continue;
        }
        for window in segment.windows(order) {
            *counts.entry(window.join(" ")).or_insert(0) += 1;
        }
    }
}

// --- log-odds scoring ---

/// Per-order n-gram counts for two classes.
///
/// Labeled `human`/`llm` here to match this module's own comparison — but
/// [`score_entries`] treats them as opaque bags of counts, so other
/// callers (e.g. `self_fingerprint`, which compares `human` against a
/// *fixed* llm class rather than the raw one) can reuse this struct and
/// its scoring for any two classes, not just the two this module mines by
/// default.
#[derive(Debug, Default)]
pub struct ClassCounts {
    /// Human-favored side of the comparison (this module's own reading;
    /// reused generically by other callers).
    pub human: BTreeMap<String, u64>,
    /// Llm-favored side of the comparison (ditto).
    pub llm: BTreeMap<String, u64>,
}

/// One scored n-gram: its class counts, the raw log-odds `delta`, and the
/// prior-variance-normalized `z`.
#[derive(Debug, Clone, PartialEq)]
pub struct Entry {
    pub ngram: String,
    pub count_llm: u64,
    pub count_human: u64,
    pub delta: f64,
    pub z: f64,
}

/// Computes the informative-Dirichlet-prior log-odds `(delta, z)` for one
/// n-gram, given its llm/human counts and each class's total n-gram-token
/// count. See the module docs for the formula.
///
/// Returns `None` if either class's smoothed denominator would be
/// non-positive (only possible in the degenerate case where a class's
/// entire token count is this one repeated n-gram) — undefined, rather
/// than a `NaN`/`inf` result.
#[must_use]
pub fn log_odds_z(y_llm: u64, n_llm: u64, y_human: u64, n_human: u64) -> Option<(f64, f64)> {
    #[allow(clippy::cast_precision_loss)]
    let (y_llm, n_llm, y_human, n_human) =
        (y_llm as f64, n_llm as f64, y_human as f64, n_human as f64);

    let alpha_w = y_llm + y_human;
    let alpha_0 = n_llm + n_human;

    let denom_llm = n_llm + alpha_0 - y_llm - alpha_w;
    let denom_human = n_human + alpha_0 - y_human - alpha_w;
    if denom_llm <= 0.0 || denom_human <= 0.0 {
        return None;
    }

    let term_llm = (y_llm + alpha_w) / denom_llm;
    let term_human = (y_human + alpha_w) / denom_human;
    let delta = term_llm.ln() - term_human.ln();

    let variance = 1.0 / (y_llm + alpha_w) + 1.0 / (y_human + alpha_w);
    let z = delta / variance.sqrt();
    Some((delta, z))
}

/// Scores every n-gram meeting `min_count`.
///
/// Considers every n-gram appearing in `counts.llm` or `counts.human`
/// whose combined count reaches `min_count`, skipping any [`log_odds_z`]
/// gives `None` for. Result order is unspecified (`BTreeSet` iteration
/// order, i.e. n-gram-text ascending) — callers sort for their own
/// purposes via [`top_llm_favored`]/[`top_human_favored`].
///
/// `pub` — see [`ClassCounts`]'s doc comment for why.
#[must_use]
pub fn score_entries(counts: &ClassCounts, min_count: u64) -> Vec<Entry> {
    let n_llm: u64 = counts.llm.values().sum();
    let n_human: u64 = counts.human.values().sum();

    let mut vocab: BTreeSet<&str> = BTreeSet::new();
    vocab.extend(counts.llm.keys().map(String::as_str));
    vocab.extend(counts.human.keys().map(String::as_str));

    let mut entries = Vec::new();
    for ngram in vocab {
        let count_llm = counts.llm.get(ngram).copied().unwrap_or(0);
        let count_human = counts.human.get(ngram).copied().unwrap_or(0);
        if count_llm + count_human < min_count {
            continue;
        }
        if let Some((delta, z)) = log_odds_z(count_llm, n_llm, count_human, n_human) {
            entries.push(Entry {
                ngram: ngram.to_string(),
                count_llm,
                count_human,
                delta,
                z,
            });
        }
    }
    entries
}

/// The top `top_n` llm-favored entries: sorted by `z` descending, ties
/// broken by n-gram text ascending.
///
/// `pub` — see [`ClassCounts`]'s doc comment for why.
#[must_use]
pub fn top_llm_favored(entries: &[Entry], top_n: usize) -> Vec<Entry> {
    let mut sorted = entries.to_vec();
    sorted.sort_by(|a, b| b.z.total_cmp(&a.z).then_with(|| a.ngram.cmp(&b.ngram)));
    sorted.truncate(top_n);
    sorted
}

/// The top `top_n` human-favored entries: sorted by `z` ascending (most
/// negative — most strongly human-favored — first), ties broken by
/// n-gram text ascending.
///
/// `pub` — see [`ClassCounts`]'s doc comment for why.
#[must_use]
pub fn top_human_favored(entries: &[Entry], top_n: usize) -> Vec<Entry> {
    let mut sorted = entries.to_vec();
    sorted.sort_by(|a, b| a.z.total_cmp(&b.z).then_with(|| a.ngram.cmp(&b.ngram)));
    sorted.truncate(top_n);
    sorted
}

// --- report rendering ---

struct ReportHeader {
    train_doc_count: usize,
    human_docs: usize,
    llm_docs: usize,
    top: usize,
    min_count: u64,
}

struct OrderReport {
    order: usize,
    n_llm: u64,
    n_human: u64,
    vocab_size: usize,
    llm_favored: Vec<Entry>,
    human_favored: Vec<Entry>,
}

fn render_report(header: &ReportHeader, orders: &[OrderReport]) -> String {
    let mut out = String::new();
    writeln!(out, "# N-gram mining report").expect("write to String is infallible");
    writeln!(out).expect("write to String is infallible");
    writeln!(
        out,
        "Train split only, pooled across genres. {} document(s) ({} human, {} llm). \
         Scored via the log-odds ratio with an informative Dirichlet prior (Monroe, Colaresi \
         & Quinn 2008, eq. 16; prior drawn from the two classes' own combined counts — see \
         this module's doc comment for the exact formula). z > 0 is llm-favored, z < 0 is \
         human-favored. Entries below --min-count ({}) are omitted; top {} per direction per \
         order.",
        header.train_doc_count, header.human_docs, header.llm_docs, header.min_count, header.top
    )
    .expect("write to String is infallible");

    for order_report in orders {
        writeln!(out).expect("write to String is infallible");
        writeln!(out, "## {}-gram", order_report.order).expect("write to String is infallible");
        writeln!(out).expect("write to String is infallible");
        writeln!(
            out,
            "Total n-gram tokens: llm={}, human={}. Scored vocabulary (>= min-count): {}.",
            order_report.n_llm, order_report.n_human, order_report.vocab_size
        )
        .expect("write to String is infallible");

        writeln!(out).expect("write to String is infallible");
        writeln!(out, "### llm-favored").expect("write to String is infallible");
        writeln!(out).expect("write to String is infallible");
        render_entry_table(&mut out, &order_report.llm_favored);

        writeln!(out).expect("write to String is infallible");
        writeln!(out, "### human-favored").expect("write to String is infallible");
        writeln!(out).expect("write to String is infallible");
        render_entry_table(&mut out, &order_report.human_favored);
    }

    out
}

fn render_entry_table(out: &mut String, entries: &[Entry]) {
    writeln!(out, "| n-gram | llm count | human count | z | delta |")
        .expect("write to String is infallible");
    writeln!(out, "|---|---|---|---|---|").expect("write to String is infallible");
    for entry in entries {
        writeln!(
            out,
            "| {} | {} | {} | {:.4} | {:.4} |",
            entry.ngram, entry.count_llm, entry.count_human, entry.z, entry.delta
        )
        .expect("write to String is infallible");
    }
}

#[cfg(test)]
// `log_odds_z_equal_counts_is_zero` compares against an exact `0.0`
// literal: with identical counts and totals on both sides, `delta` and
// `z` are algebraically exactly zero (ln(1) and 0/sqrt(..)), not an
// approximation, so exact equality is the correct check there rather
// than a float-precision bug.
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    // --- word_segments ---

    /// Plain prose with no punctuation is one segment.
    #[test]
    fn word_segments_no_punctuation_is_one_segment() {
        let segments = word_segments("as an ai language model");
        assert_eq!(segments, vec![vec!["as", "an", "ai", "language", "model"]]);
    }

    /// A period splits the text into two segments, so a 2-gram can never
    /// span the sentence boundary.
    #[test]
    fn word_segments_period_splits_segments() {
        let segments = word_segments("It works. Moreover it helps.");
        assert_eq!(
            segments,
            vec![vec!["it", "works"], vec!["moreover", "it", "helps"],]
        );
    }

    /// A comma also splits segments (not just sentence-enders).
    #[test]
    fn word_segments_comma_splits_segments() {
        let segments = word_segments("fast, reliable, and cheap");
        assert_eq!(
            segments,
            vec![vec!["fast"], vec!["reliable"], vec!["and", "cheap"]]
        );
    }

    /// An interior apostrophe (ASCII or right single quotation mark)
    /// stays part of the word rather than splitting it, and is normalized
    /// to ASCII `'` either way, so both spellings of `"it's"` count as
    /// the same token.
    #[test]
    fn word_segments_keeps_interior_apostrophe() {
        let segments = word_segments("don't stop it\u{2019}s fine");
        assert_eq!(segments, vec![vec!["don't", "stop", "it's", "fine"]]);
    }

    /// Empty text produces no segments.
    #[test]
    fn word_segments_empty_text_is_empty() {
        assert!(word_segments("").is_empty());
    }

    // --- accumulate_ngrams ---

    /// 2-grams over a 4-word single segment: three overlapping windows,
    /// each counted once.
    #[test]
    fn accumulate_ngrams_counts_overlapping_windows() {
        let segments = vec![vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "b".to_string(),
        ]];
        let mut counts = BTreeMap::new();
        accumulate_ngrams(&segments, 2, &mut counts);
        assert_eq!(counts["a b"], 1);
        assert_eq!(counts["b c"], 1);
        assert_eq!(counts["c b"], 1);
        assert_eq!(counts.len(), 3);
    }

    /// A segment shorter than the requested order contributes nothing.
    #[test]
    fn accumulate_ngrams_short_segment_contributes_nothing() {
        let segments = vec![vec!["only".to_string()]];
        let mut counts = BTreeMap::new();
        accumulate_ngrams(&segments, 2, &mut counts);
        assert!(counts.is_empty());
    }

    // --- log_odds_z: hand-computed fixtures ---

    /// `y_llm=10`, `n_llm=100`, `y_human=2`, `n_human=100`. `alpha_w` =
    /// 12, `alpha_0` = 200. `term_llm` = (10+12)/(100+200-10-12) =
    /// 22/278; `term_human` = (2+12)/(100+200-2-12) = 14/286. `delta` =
    /// ln(22/278) - ln(14/286) ≈ 0.480356 (positive: llm-favored,
    /// matching that llm's raw rate 10/100 exceeds human's 2/100).
    /// `variance` = 1/22 + 1/14 ≈ 0.116883; `sqrt(variance)` ≈ 0.341881;
    /// `z` = `delta` / `sqrt(variance)` ≈ 1.405035.
    #[test]
    fn log_odds_z_matches_hand_computed_llm_favored_example() {
        let (delta, z) = log_odds_z(10, 100, 2, 100).unwrap();
        assert!(
            (delta - 0.480_355_820_872_272_7).abs() < 1e-9,
            "delta={delta}"
        );
        assert!((z - 1.405_035_073_810_234).abs() < 1e-9, "z={z}");
    }

    /// Swapping which class is `llm` and which is `human` (with the same
    /// counts) negates both `delta` and `z` exactly — the scoring is
    /// antisymmetric in the two classes, as the formula's structure
    /// requires.
    #[test]
    fn log_odds_z_swapping_classes_negates_result() {
        let (delta_a, z_a) = log_odds_z(10, 100, 2, 100).unwrap();
        let (delta_b, z_b) = log_odds_z(2, 100, 10, 100).unwrap();
        assert!((delta_a + delta_b).abs() < 1e-12);
        assert!((z_a + z_b).abs() < 1e-12);
    }

    /// Identical counts and totals in both classes give exactly zero
    /// log-odds and zero z (no evidence of a difference either way).
    #[test]
    fn log_odds_z_equal_counts_is_zero() {
        let (delta, z) = log_odds_z(5, 50, 5, 50).unwrap();
        assert_eq!(delta, 0.0);
        assert_eq!(z, 0.0);
    }

    // --- score_entries / top_llm_favored / top_human_favored ---

    fn counts_from(human: &[(&str, u64)], llm: &[(&str, u64)]) -> ClassCounts {
        ClassCounts {
            human: human.iter().map(|&(k, v)| (k.to_string(), v)).collect(),
            llm: llm.iter().map(|&(k, v)| (k.to_string(), v)).collect(),
        }
    }

    /// An n-gram below `min_count` (combined llm + human) is excluded
    /// from scoring entirely.
    #[test]
    fn score_entries_filters_below_min_count() {
        let counts = counts_from(&[("rare phrase", 1)], &[("rare phrase", 1)]);
        let entries = score_entries(&counts, 5);
        assert!(entries.is_empty());
    }

    /// A tiny hand-built corpus: "as an ai" appears often in llm, rarely
    /// in human; "in my experience" is the reverse. Both clear
    /// min-count=2, and each ranks top of its own favored direction.
    #[test]
    fn top_favored_ranks_the_strongly_skewed_ngram_first() {
        let counts = counts_from(
            &[
                ("in my experience", 8),
                ("as an ai", 1),
                ("neutral phrase", 4),
            ],
            &[
                ("as an ai", 9),
                ("in my experience", 1),
                ("neutral phrase", 4),
            ],
        );
        let entries = score_entries(&counts, 2);

        let llm_top = top_llm_favored(&entries, 1);
        assert_eq!(llm_top[0].ngram, "as an ai");
        assert!(llm_top[0].z > 0.0);

        let human_top = top_human_favored(&entries, 1);
        assert_eq!(human_top[0].ngram, "in my experience");
        assert!(human_top[0].z < 0.0);
    }

    /// Two n-grams tied on `z` (both absent from one class, present once
    /// in the other, with identical totals) break the tie
    /// lexicographically ascending, for both directions.
    #[test]
    fn tied_z_breaks_lexicographically() {
        let counts = counts_from(&[], &[("zed phrase", 3), ("alpha phrase", 3)]);
        let entries = score_entries(&counts, 2);
        let top = top_llm_favored(&entries, 2);
        assert_eq!(top[0].ngram, "alpha phrase");
        assert_eq!(top[1].ngram, "zed phrase");
    }

    /// Re-scoring and re-sorting the same input twice is byte-identical
    /// (deterministic ordering) — checked by comparing the rendered
    /// ngram-order sequence, not just set membership.
    #[test]
    fn top_llm_favored_ordering_is_deterministic() {
        let counts = counts_from(
            &[("b phrase", 2), ("a phrase", 2), ("c phrase", 2)],
            &[("b phrase", 6), ("a phrase", 6), ("c phrase", 6)],
        );
        let entries = score_entries(&counts, 2);
        let first = top_llm_favored(&entries, 3);
        let second = top_llm_favored(&entries, 3);
        let names_a: Vec<&str> = first.iter().map(|e| e.ngram.as_str()).collect();
        let names_b: Vec<&str> = second.iter().map(|e| e.ngram.as_str()).collect();
        assert_eq!(names_a, names_b);
        // All three are tied on z (identical llm/human ratio), so the
        // deterministic tie-break is purely lexicographic.
        assert_eq!(names_a, vec!["a phrase", "b phrase", "c phrase"]);
    }

    // --- render_report ---

    /// The rendered report contains the header summary and both
    /// direction sections for every requested order.
    #[test]
    fn render_report_contains_expected_sections() {
        let header = ReportHeader {
            train_doc_count: 10,
            human_docs: 5,
            llm_docs: 5,
            top: 5,
            min_count: 2,
        };
        let orders = vec![OrderReport {
            order: 1,
            n_llm: 100,
            n_human: 100,
            vocab_size: 3,
            llm_favored: vec![Entry {
                ngram: "moreover".to_string(),
                count_llm: 9,
                count_human: 1,
                delta: 1.2,
                z: 3.4,
            }],
            human_favored: vec![Entry {
                ngram: "yeah".to_string(),
                count_llm: 1,
                count_human: 9,
                delta: -1.2,
                z: -3.4,
            }],
        }];
        let report = render_report(&header, &orders);
        assert!(report.contains("# N-gram mining report"));
        assert!(report.contains("## 1-gram"));
        assert!(report.contains("### llm-favored"));
        assert!(report.contains("### human-favored"));
        assert!(report.contains("moreover"));
        assert!(report.contains("yeah"));
    }

    /// Rendering the same input twice produces byte-identical output.
    #[test]
    fn render_report_is_deterministic() {
        let header = ReportHeader {
            train_doc_count: 1,
            human_docs: 1,
            llm_docs: 0,
            top: 1,
            min_count: 1,
        };
        let orders = vec![OrderReport {
            order: 1,
            n_llm: 1,
            n_human: 1,
            vocab_size: 0,
            llm_favored: Vec::new(),
            human_favored: Vec::new(),
        }];
        let a = render_report(&header, &orders);
        let b = render_report(&header, &orders);
        assert_eq!(a, b);
    }
}
