//! `corpus-tool attest` — builds the seam-bigram membership table and
//! POS-skeleton n-gram sets from TRAIN-split human docs.
//!
//! `friction_packs::attestation::AttestationPack` reconstructs those two
//! tables from the versioned TOML pack this command writes
//! (`attestation-v1.toml`) at load time.
//!
//! # Tokenization convention — string-level, whole-document, like `index`
//!
//! Like `corpus-tool index` (and unlike `mine-inventory`), this command
//! reads each doc's raw whole-file text directly — no `friction-parse`
//! block-tree prose extraction. `friction_harness::clean::clean` is run
//! once over the whole document, `friction_harness::clean::split_sentences`
//! splits the cleaned text into sentences, and each sentence's
//! word/punctuation tokens come from `friction_harness::clean::tokenize`
//! (which re-applies `clean` to its input: a no-op on already-cleaned text,
//! so this costs nothing and keeps every call site using the same one
//! public function rather than a hand-rolled variant).
//!
//! # Two independent per-sentence passes, not one paired stream
//!
//! Each sentence contributes to the bigram table via
//! `friction_harness::clean::tokenize` and to the skeleton set via the
//! shipped [`friction_nlp::PerceptronTagger`]'s own tokenization:
//! **independently**, not positionally paired token-for-token. See
//! `friction_packs::attestation`'s module doc for why forcing positional
//! correspondence between the two tokenizations is a correctness trap (a
//! run of `"..."` is three tokens on the bigram side and one on the
//! skeleton side; digits are silently dropped on the bigram side but
//! tagged `CD` on the skeleton side) that this design avoids by
//! construction: a sentence with an empty word-token list is simply
//! skipped for the bigram pass, and independently, a sentence with no
//! taggable tokens is skipped for the skeleton pass.
//!
//! # Vocabulary id assignment
//!
//! The word vocabulary reserves id `0` for the literal `"<s>"`
//! sentence-start sentinel (which the tokenizer's own alphabet can never
//! produce, since `"<"` is not part of any token shape it emits);
//! every other distinct token observed is assigned an id in sorted
//! lexicographic order — deterministic, independent of which document
//! happens to be processed first. The coarse-tag vocabulary works the
//! same way, reserving id `0`/`1` for the `"<S>"`/`"<E>"` sentence
//! sentinels and sorting the remainder.
//!
//! # Two-stage build
//!
//! This command builds stage 1 only (the bigram/skeleton tables
//! themselves, from TRAIN human docs alone, no engine dependency). Stage
//! 2 (near-no-op pivot-rate calibration, which needs a real repair engine
//! to run against) is a distinct, later tool; a stage-1-only pack simply
//! has no `[near_noop]` table, which
//! `friction_packs::AttestationPack::parse` accepts.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::PathBuf;

use anyhow::Context as _;
use clap::Args as ClapArgs;
use friction_nlp::{PerceptronTagger, Tagger, coarse_tag};

use crate::corpus_layout::relpath;
use crate::hashing::sha256_hex;
use crate::manifest::{self, Class, ManifestRecord, Split};

/// The reserved sentence-start token prepended to every sentence's word
/// sequence before windowing, so a sentence's first real word always has
/// a left neighbor to form a bigram with.
const SENTENCE_START: &str = "<s>";

/// The reserved sentence-start/end sentinels prepended/appended to every
/// sentence's coarse-tag sequence before windowing.
const SKELETON_START: &str = "<S>";
const SKELETON_END: &str = "<E>";

/// Arguments for `corpus-tool attest`.
#[derive(Debug, ClapArgs)]
pub struct Args {
    /// Corpus root directory.
    #[arg(long, default_value = "corpus")]
    pub corpus_dir: PathBuf,
    /// Path to write the versioned attestation pack to.
    #[arg(
        long,
        default_value = "crates/friction-packs/packs/attestation-v1.toml"
    )]
    pub pack: PathBuf,
    /// Also runs stage 2: measures the natural derivational-pivot rate a
    /// real `friction-edit` engine produces over every TRAIN human doc
    /// (using the CURRENTLY EMBEDDED, stage-1 attestation pack — i.e. the
    /// on-disk `--pack` file as of this binary's own build, unconstrained
    /// by any prior `[near_noop]` table), sets
    /// `threshold_per_1000_words` from the observed TRAIN maximum with a
    /// documented 15% headroom, reads DEV once as a labeled cross-check
    /// that is never fed back into the threshold, and rewrites `--pack`
    /// with the result. Requires a prior, separate `corpus-tool attest`
    /// run (without this flag) and a rebuild, so this binary embeds an
    /// up-to-date stage-1 pack before measuring against it — bootstrap
    /// order, not a bug: stage 2 needs a real engine, which needs a
    /// stage-1 pack to exist first.
    #[arg(long)]
    pub calibrate_near_noop: bool,
}

/// One sentence's independently-computed bigram tokens and skeleton tags.
struct SentenceData {
    /// Word/punctuation tokens from `friction_harness::clean::tokenize`,
    /// empty if the sentence had none.
    bigram_tokens: Vec<String>,
    /// Coarse part-of-speech tags from the tagger's own tokenization,
    /// empty if the tagger found no tokens.
    skeleton_tags: Vec<Box<str>>,
}

/// One document's measured derivational-pivot rate: patches per 1000
/// prose words, from a real, uncalibrated (`PivotBudget::unlimited`)
/// engine run.
fn measure_pivot_rates(
    engine: &friction_edit::Engine,
    corpus_dir: &std::path::Path,
    records: &[&ManifestRecord],
) -> anyhow::Result<Vec<f64>> {
    let mut rates = Vec::with_capacity(records.len());
    for &record in records {
        let path = corpus_dir.join(relpath(record));
        let text = std::fs::read_to_string(&path)
            .map_err(|e| anyhow::anyhow!("attest: failed to read {}: {e}", path.display()))?;
        let word_count = engine.word_count(&text).map_err(|e| {
            anyhow::anyhow!("attest: failed to count words in {}: {e}", path.display())
        })?;
        if word_count == 0 {
            continue;
        }
        let (_out, report) = engine
            .fix_document(&text)
            .map_err(|e| anyhow::anyhow!("attest: engine failed on {}: {e}", path.display()))?;
        let pivot_count = report
            .passes
            .iter()
            .flat_map(|pass| pass.applied_patches.iter())
            .filter(|patch| patch.rule.as_str() == "pivot.lvc")
            .count();
        #[allow(clippy::cast_precision_loss)]
        let rate = pivot_count as f64 / (word_count as f64 / 1000.0);
        rates.push(rate);
    }
    Ok(rates)
}

/// The near-no-op calibration result: a `threshold_per_1000_words` set
/// from the observed TRAIN maximum with a documented 15% headroom, and a
/// DEV-split cross-check that is read once, labeled, and never fed back
/// into the threshold.
struct NearNoOpResult {
    threshold_per_1000_words: f64,
    train_max_rate: f64,
    dev_max_rate: Option<f64>,
    sample_doc_count: u32,
}

fn calibrate_near_noop(
    corpus_dir: &std::path::Path,
    train_human: &[&ManifestRecord],
    dev_human: &[&ManifestRecord],
) -> anyhow::Result<NearNoOpResult> {
    let engine = friction_edit::Engine::new()
        .context("attest --calibrate-near-noop: failed to load the friction-edit engine")?;

    let train_rates = measure_pivot_rates(&engine, corpus_dir, train_human)?;
    let train_max_rate = train_rates.iter().copied().fold(0.0_f64, f64::max);
    // 15% headroom over the observed TRAIN maximum: a hard ceiling gate
    // should not sit exactly at the noisiest single document's own rate.
    let threshold_per_1000_words = (train_max_rate * 1.15 * 100.0).ceil() / 100.0;

    // DEV-SPLIT CALIBRATION CHECK: READ ONCE, NEVER TUNED AGAINST. This
    // value is recorded for the printed report and the pack's own
    // `dev_check_max_per_1000_words` field only; nothing above this line
    // is allowed to read it, and nothing below adjusts
    // `threshold_per_1000_words` in response to it.
    let dev_rates = measure_pivot_rates(&engine, corpus_dir, dev_human)?;
    let dev_max_rate = if dev_rates.is_empty() {
        None
    } else {
        Some(dev_rates.iter().copied().fold(0.0_f64, f64::max))
    };

    Ok(NearNoOpResult {
        threshold_per_1000_words,
        train_max_rate,
        dev_max_rate,
        sample_doc_count: u32::try_from(train_human.len()).unwrap_or(u32::MAX),
    })
}

fn render_near_noop_section(result: &NearNoOpResult) -> String {
    let mut out = String::new();
    writeln!(out, "\n[near_noop]").expect("write to String is infallible");
    writeln!(
        out,
        "threshold_per_1000_words = {}",
        result.threshold_per_1000_words
    )
    .expect("write to String is infallible");
    writeln!(out, "sample_doc_count = {}", result.sample_doc_count)
        .expect("write to String is infallible");
    if let Some(dev_max) = result.dev_max_rate {
        writeln!(out, "dev_check_max_per_1000_words = {dev_max}")
            .expect("write to String is infallible");
    }
    writeln!(out, "method = \"train-max-times-1.15\"").expect("write to String is infallible");
    out
}

/// Runs `attest`.
///
/// # Errors
///
/// Returns an error if the manifest or a referenced document can't be
/// read, if the embedded tagger model fails to load, or if `--pack`
/// can't be written.
pub fn run(args: &Args) -> anyhow::Result<()> {
    let tagger =
        PerceptronTagger::new().context("failed to load the embedded English tagger model")?;

    let manifest_path = args.corpus_dir.join("manifest.jsonl");
    let manifest_bytes = std::fs::read(&manifest_path).unwrap_or_default();
    let records = manifest::read_manifest(&manifest_path)?.unwrap_or_default();

    let mut train_human: Vec<&ManifestRecord> = records
        .iter()
        .filter(|r| r.split == Some(Split::Train) && r.class == Class::Human)
        .collect();
    train_human.sort_by(|a, b| a.id.cmp(&b.id));

    let mut sentences: Vec<SentenceData> = Vec::new();
    for &record in &train_human {
        let path = args.corpus_dir.join(relpath(record));
        let bytes = std::fs::read(&path)
            .map_err(|e| anyhow::anyhow!("attest: failed to read {}: {e}", path.display()))?;
        let text = String::from_utf8(bytes)
            .map_err(|e| anyhow::anyhow!("attest: {} is not valid UTF-8: {e}", path.display()))?;

        let cleaned = friction_harness::clean::clean(&text);
        for sentence in friction_harness::clean::split_sentences(&cleaned) {
            let bigram_tokens = friction_harness::clean::tokenize(sentence);
            let skeleton_tags: Vec<Box<str>> = tagger
                .tag(sentence, 0)
                .iter()
                .map(|t| coarse_tag(&t.pos))
                .collect();
            if bigram_tokens.is_empty() && skeleton_tags.is_empty() {
                continue;
            }
            sentences.push(SentenceData {
                bigram_tokens,
                skeleton_tags,
            });
        }
    }

    let (word_vocab, bigram) = build_bigram(&sentences);
    let (tag_vocab, tag5, tag4) = build_skeleton(&sentences);

    let header = PackHeader {
        corpus_manifest_sha256: sha256_hex(&manifest_bytes),
        train_human_doc_count: train_human.len(),
    };
    let mut pack_text = render_pack(&header, &word_vocab, &bigram, &tag_vocab, &tag5, &tag4);

    if let Some(parent) = args.pack.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&args.pack, &pack_text)?;
    println!(
        "attest: wrote {} word vocab token(s), {} bigram edge(s), {} tag vocab token(s), \
         {} skeleton 5-gram(s), {} skeleton 4-gram(s) from {} train human doc(s) to {}",
        word_vocab.len(),
        bigram.values().map(BTreeSet::len).sum::<usize>(),
        tag_vocab.len(),
        tag5.len(),
        tag4.len(),
        train_human.len(),
        args.pack.display()
    );

    if args.calibrate_near_noop {
        let mut dev_human: Vec<&ManifestRecord> = records
            .iter()
            .filter(|r| r.split == Some(Split::Dev) && r.class == Class::Human)
            .collect();
        dev_human.sort_by(|a, b| a.id.cmp(&b.id));

        let result = calibrate_near_noop(&args.corpus_dir, &train_human, &dev_human)?;
        pack_text.push_str(&render_near_noop_section(&result));

        std::fs::write(&args.pack, &pack_text)?;
        println!(
            "attest --calibrate-near-noop: threshold_per_1000_words={:.2} (train max \
             {:.4} x1.15), dev_check_max_per_1000_words={} ({} dev human doc(s)), rewrote {}",
            result.threshold_per_1000_words,
            result.train_max_rate,
            result
                .dev_max_rate
                .map_or_else(|| "n/a".to_string(), |v| format!("{v:.4}")),
            dev_human.len(),
            args.pack.display()
        );
    }

    Ok(())
}

/// Builds the sorted word vocabulary (id `0` reserved for `"<s>"`) and the
/// bigram membership map (`left id -> attested right ids`) from every
/// sentence's `bigram_tokens`.
fn build_bigram(sentences: &[SentenceData]) -> (Vec<String>, BTreeMap<u32, BTreeSet<u32>>) {
    let mut distinct: BTreeSet<String> = BTreeSet::new();
    for sentence in sentences {
        distinct.extend(sentence.bigram_tokens.iter().cloned());
    }
    let mut vocab: Vec<String> = Vec::with_capacity(distinct.len() + 1);
    vocab.push(SENTENCE_START.to_string());
    vocab.extend(distinct);

    let mut id_of: BTreeMap<&str, u32> = BTreeMap::new();
    for (index, token) in vocab.iter().enumerate() {
        #[allow(clippy::cast_possible_truncation)]
        let id = index as u32;
        id_of.insert(token.as_str(), id);
    }

    let mut bigram: BTreeMap<u32, BTreeSet<u32>> = BTreeMap::new();
    for sentence in sentences {
        if sentence.bigram_tokens.is_empty() {
            continue;
        }
        let mut seq: Vec<&str> = Vec::with_capacity(sentence.bigram_tokens.len() + 1);
        seq.push(SENTENCE_START);
        seq.extend(sentence.bigram_tokens.iter().map(String::as_str));
        for window in seq.windows(2) {
            let left = id_of[window[0]];
            let right = id_of[window[1]];
            bigram.entry(left).or_default().insert(right);
        }
    }

    (vocab, bigram)
}

/// Builds the sorted coarse-tag vocabulary (ids `0`/`1` reserved for
/// `"<S>"`/`"<E>"`) and the packed 5-gram/4-gram tag-window sets from
/// every sentence's `skeleton_tags`.
fn build_skeleton(sentences: &[SentenceData]) -> (Vec<String>, BTreeSet<u64>, BTreeSet<u64>) {
    let mut distinct: BTreeSet<String> = BTreeSet::new();
    for sentence in sentences {
        distinct.extend(sentence.skeleton_tags.iter().map(ToString::to_string));
    }
    distinct.remove(SKELETON_START);
    distinct.remove(SKELETON_END);

    let mut vocab: Vec<String> = Vec::with_capacity(distinct.len() + 2);
    vocab.push(SKELETON_START.to_string());
    vocab.push(SKELETON_END.to_string());
    vocab.extend(distinct);

    let mut id_of: BTreeMap<&str, u32> = BTreeMap::new();
    for (index, tag) in vocab.iter().enumerate() {
        #[allow(clippy::cast_possible_truncation)]
        let id = index as u32;
        id_of.insert(tag.as_str(), id);
    }
    #[allow(clippy::cast_possible_truncation)]
    let base = vocab.len() as u64;

    let mut tag5: BTreeSet<u64> = BTreeSet::new();
    let mut tag4: BTreeSet<u64> = BTreeSet::new();
    for sentence in sentences {
        if sentence.skeleton_tags.is_empty() {
            continue;
        }
        let mut seq: Vec<&str> = Vec::with_capacity(sentence.skeleton_tags.len() + 2);
        seq.push(SKELETON_START);
        seq.extend(sentence.skeleton_tags.iter().map(AsRef::as_ref));
        seq.push(SKELETON_END);

        for window in seq.windows(5) {
            tag5.insert(pack_window(window, &id_of, base));
        }
        for window in seq.windows(4) {
            tag4.insert(pack_window(window, &id_of, base));
        }
    }

    (vocab, tag5, tag4)
}

/// Packs a fixed-length window of tag text into a base-`base` `u64`
/// encoding, one "digit" per tag's vocabulary id.
fn pack_window(window: &[&str], id_of: &BTreeMap<&str, u32>, base: u64) -> u64 {
    window
        .iter()
        .fold(0u64, |acc, tag| acc * base + u64::from(id_of[tag]))
}

struct PackHeader {
    corpus_manifest_sha256: String,
    train_human_doc_count: usize,
}

/// Escapes `text` into a valid double-quoted TOML basic-string literal —
/// identical discipline to `corpus-tool index`'s own `toml_quote` (word
/// tokens are plain ASCII in the overwhelming common case, but this is
/// applied uniformly rather than special-cased, for the same reasons that
/// module documents).
fn toml_quote(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for ch in text.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\u{8}' => out.push_str("\\b"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\u{c}' => out.push_str("\\f"),
            '\r' => out.push_str("\\r"),
            c if (c as u32) < 0x20 || c as u32 == 0x7F => {
                write!(out, "\\u{:04X}", c as u32).expect("write to String is infallible");
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Renders a comma-joined `u32` id list.
fn render_ids_csv(ids: &[u32]) -> String {
    ids.iter().map(u32::to_string).collect::<Vec<_>>().join(",")
}

/// Renders a comma-joined `u64` list.
fn render_u64_csv(values: &BTreeSet<u64>) -> String {
    values
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn render_pack(
    header: &PackHeader,
    word_vocab: &[String],
    bigram: &BTreeMap<u32, BTreeSet<u32>>,
    tag_vocab: &[String],
    tag5: &BTreeSet<u64>,
    tag4: &BTreeSet<u64>,
) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# attestation-v1 pack: seam-bigram membership table and POS-skeleton n-gram sets \
         mined from TRAIN-split human docs (see `friction_packs::attestation`)."
    )
    .expect("write to String is infallible");
    writeln!(
        out,
        "# Generated by `corpus-tool attest` — do not hand-edit."
    )
    .expect("write to String is infallible");
    writeln!(out).expect("write to String is infallible");

    writeln!(out, "[pack]").expect("write to String is infallible");
    writeln!(out, "version = \"attestation-v1\"").expect("write to String is infallible");
    writeln!(
        out,
        "corpus_manifest_sha256 = {}",
        toml_quote(&header.corpus_manifest_sha256)
    )
    .expect("write to String is infallible");
    writeln!(
        out,
        "train_human_doc_count = {}",
        header.train_human_doc_count
    )
    .expect("write to String is infallible");
    writeln!(out).expect("write to String is infallible");

    writeln!(out, "[vocab]").expect("write to String is infallible");
    writeln!(out, "tokens = [").expect("write to String is infallible");
    for token in word_vocab {
        writeln!(out, "    {},", toml_quote(token)).expect("write to String is infallible");
    }
    writeln!(out, "]").expect("write to String is infallible");
    writeln!(out).expect("write to String is infallible");

    let lefts: Vec<u32> = bigram.keys().copied().collect();
    let rights: Vec<String> = bigram
        .values()
        .map(|set| render_ids_csv(&set.iter().copied().collect::<Vec<_>>()))
        .collect();
    writeln!(out, "[bigram]").expect("write to String is infallible");
    writeln!(out, "lefts = {}", toml_quote(&render_ids_csv(&lefts)))
        .expect("write to String is infallible");
    writeln!(out, "rights = {}", toml_quote(&rights.join(";")))
        .expect("write to String is infallible");
    writeln!(out).expect("write to String is infallible");

    writeln!(out, "[skeleton]").expect("write to String is infallible");
    writeln!(out, "tags = [").expect("write to String is infallible");
    for tag in tag_vocab {
        writeln!(out, "    {},", toml_quote(tag)).expect("write to String is infallible");
    }
    writeln!(out, "]").expect("write to String is infallible");
    writeln!(out, "tag5 = {}", toml_quote(&render_u64_csv(tag5)))
        .expect("write to String is infallible");
    writeln!(out, "tag4 = {}", toml_quote(&render_u64_csv(tag4)))
        .expect("write to String is infallible");

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sentence(bigram_tokens: &[&str], skeleton_tags: &[&str]) -> SentenceData {
        SentenceData {
            bigram_tokens: bigram_tokens.iter().map(|s| (*s).to_string()).collect(),
            skeleton_tags: skeleton_tags.iter().map(|s| Box::from(*s)).collect(),
        }
    }

    // --- build_bigram ---

    #[test]
    fn build_bigram_reserves_sentence_start_at_id_zero() {
        let (vocab, _) = build_bigram(&[sentence(&["the", "fox"], &[])]);
        assert_eq!(vocab[0], "<s>");
    }

    #[test]
    fn build_bigram_sorts_the_remaining_vocab_lexicographically() {
        let (vocab, _) = build_bigram(&[sentence(&["fox", "the", "ant"], &[])]);
        assert_eq!(vocab, vec!["<s>", "ant", "fox", "the"]);
    }

    #[test]
    fn build_bigram_records_the_sentence_start_edge() {
        let (vocab, bigram) = build_bigram(&[sentence(&["the", "fox"], &[])]);
        let id_of =
            |text: &str| u32::try_from(vocab.iter().position(|t| t == text).unwrap()).unwrap();
        let s_id = id_of("<s>");
        let the_id = id_of("the");
        let fox_id = id_of("fox");
        assert!(bigram[&s_id].contains(&the_id));
        assert!(bigram[&the_id].contains(&fox_id));
    }

    #[test]
    fn build_bigram_is_membership_only_duplicate_edges_collapse() {
        let (_, bigram) = build_bigram(&[
            sentence(&["the", "fox"], &[]),
            sentence(&["the", "fox"], &[]),
        ]);
        let total_edges: usize = bigram.values().map(BTreeSet::len).sum();
        // "<s>"->"the" and "the"->"fox": exactly two distinct edges,
        // regardless of how many times each sentence repeats.
        assert_eq!(total_edges, 2);
    }

    #[test]
    fn build_bigram_skips_a_sentence_with_no_bigram_tokens() {
        let (vocab, bigram) = build_bigram(&[sentence(&[], &["DT"])]);
        assert_eq!(vocab, vec!["<s>"]);
        assert!(bigram.is_empty());
    }

    // --- build_skeleton ---

    #[test]
    fn build_skeleton_reserves_sentinels_at_ids_zero_and_one() {
        let (vocab, _, _) = build_skeleton(&[sentence(&[], &["DT", "NN"])]);
        assert_eq!(vocab[0], "<S>");
        assert_eq!(vocab[1], "<E>");
    }

    #[test]
    fn build_skeleton_produces_one_5gram_for_a_3tag_sentence() {
        // <S> DT NN <E> is only 4 long, so windows(5) produces nothing.
        // Windows(4) produces exactly one window.
        let (_, tag5, tag4) = build_skeleton(&[sentence(&[], &["DT", "NN"])]);
        assert!(tag5.is_empty());
        assert_eq!(tag4.len(), 1);
    }

    #[test]
    fn build_skeleton_produces_windows_for_a_longer_sentence() {
        let (_, tag5, tag4) = build_skeleton(&[sentence(&[], &["DT", "JJ", "NN", "VB"])]);
        // Wrapped: <S> DT JJ NN VB <E> = 6 tags -> windows(5): 2, windows(4): 3.
        assert_eq!(tag5.len(), 2);
        assert_eq!(tag4.len(), 3);
    }

    #[test]
    fn build_skeleton_skips_a_sentence_with_no_tags() {
        let (vocab, tag5, tag4) = build_skeleton(&[sentence(&["word"], &[])]);
        assert_eq!(vocab, vec!["<S>", "<E>"]);
        assert!(tag5.is_empty());
        assert!(tag4.is_empty());
    }

    #[test]
    fn build_skeleton_keeps_punctuation_tags_verbatim() {
        let (vocab, _, _) = build_skeleton(&[sentence(&[], &["DT", "NN", "."])]);
        assert!(vocab.contains(&".".to_string()));
    }

    // --- toml_quote / render_ids_csv / render_u64_csv ---

    #[test]
    fn toml_quote_leaves_plain_ascii_tokens_alone() {
        assert_eq!(toml_quote("hello"), "\"hello\"");
        assert_eq!(toml_quote("don't"), "\"don't\"");
    }

    #[test]
    fn render_ids_csv_comma_joins() {
        assert_eq!(render_ids_csv(&[1, 2, 3]), "1,2,3");
        assert_eq!(render_ids_csv(&[]), "");
    }

    #[test]
    fn render_u64_csv_comma_joins_in_sorted_order() {
        let mut set = BTreeSet::new();
        set.insert(300u64);
        set.insert(5u64);
        set.insert(42u64);
        assert_eq!(render_u64_csv(&set), "5,42,300");
    }

    // --- full render_pack round trip ---

    #[test]
    fn render_pack_round_trips_through_attestation_pack_parse() {
        let sentences = vec![sentence(&["the", "fox"], &["DT", "NN"])];
        let (word_vocab, bigram) = build_bigram(&sentences);
        let (tag_vocab, tag5, tag4) = build_skeleton(&sentences);
        let header = PackHeader {
            corpus_manifest_sha256: "deadbeef".to_string(),
            train_human_doc_count: 1,
        };
        let text = render_pack(&header, &word_vocab, &bigram, &tag_vocab, &tag5, &tag4);

        let parsed = friction_packs::AttestationPack::parse(&text).expect("generated pack parses");
        assert!(parsed.bigram().attests("<s>", "the"));
        assert!(parsed.bigram().attests("the", "fox"));
        assert!(!parsed.bigram().attests("<s>", "fox"));
    }
}
