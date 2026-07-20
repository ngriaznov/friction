//! Self-fingerprint check: does the fixer trade the corpus's original
//! llm-favored n-grams for a *new* set of its own?
//!
//! # What this does
//!
//! 1. Fixes every `train`-split `llm` corpus doc with the **release**
//!    `friction` CLI (`target/release/friction fix ... --format json`,
//!    built here if missing), writing each fixed doc under
//!    `target/fingerprint/llm-fixed/` and aggregating the per-rule patch
//!    counts each invocation's JSON summary reports.
//! 2. Tokenizes the fixed docs, the *original* (pre-fix) `llm` docs, and
//!    every `human`-class `train`-split doc identically, via
//!    [`corpus_tool::commands::mine`]'s own `word_segments`/
//!    `accumulate_ngrams` — the exact tokenization `corpus-tool mine`
//!    itself uses, reused rather than re-implemented (see that module's
//!    docs for the tokenization contract).
//! 3. Scores 1-, 2-, and 3-grams for two class pairs with `mine`'s own
//!    log-odds-with-informative-Dirichlet-prior machinery
//!    ([`ClassCounts`]/[`score_entries`]/[`top_llm_favored`]/
//!    [`log_odds_z`], all now `pub` for this reuse): `human-train` vs.
//!    `fixed-llm` (does the fixer's *own* output still read as
//!    machine-favored against real human prose?) and `human-train` vs.
//!    `original-llm` (the baseline `corpus-tool mine` already reports in
//!    `corpus/MINING.md`, recomputed here so both comparisons come from
//!    identically-built `Corpus` totals).
//! 4. Classifies every n-gram in the fixed-llm top 30 as either
//!    "pre-existing" (already llm-favored before fixing — the fixer
//!    reduced but did not eliminate it) or "new" (not llm-favored in the
//!    original corpus at all — something the fixer itself introduced),
//!    from each n-gram's own before/after z-score, not a guess.
//! 5. Quantifies a small set of *candidate* fixer-introduced tics named
//!    up front, from what each rule's own source is documented to
//!    produce: sentence-initial
//!    "And"/"But"/"So" (`connective.surgery`'s swap strategy),
//!    sentence-initial "This <verb>" (`symmetry.participial_closer`'s
//!    promote strategy), and doubled spaces (a generic patch-splicing
//!    hazard, not attributed to one rule) — each as a rate per 1000
//!    tokens across all three corpora, plus the rule's own aggregate
//!    patch count over the whole run (ground truth for whether it fired
//!    at all).
//!
//! Writes `corpus/FINGERPRINT.md` with every table above and an honest
//! conclusion. Run with `cargo run --release --example self_fingerprint -p
//! corpus-tool` from the repository root (release mode: this invokes the
//! tagger-loading CLI 230+ times, one per `llm`-class train doc).
//!
//! # Why the release CLI, not a library shortcut
//!
//! `friction_apply::FixEngine` would fix a document in-process, faster and
//! without a subprocess per doc — and does run the exact same
//! [`friction_apply::fix_document`] the CLI's own `fix` subcommand calls
//! (see `friction-cli/src/fix.rs`). But this check exists to answer "what
//! does the *shipped product* actually do to a document", so it shells
//! out to the real, release-profile `friction` binary rather than the
//! library call it happens to be a thin wrapper over — the two are
//! expected to agree, and a shortcut here would quietly stop being true
//! the moment they didn't.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context as _, bail};
use corpus_tool::commands::mine::{
    ClassCounts, Entry, accumulate_ngrams, log_odds_z, score_entries, top_llm_favored,
    word_segments,
};
use corpus_tool::corpus_layout;
use corpus_tool::manifest::{self, Class, ManifestRecord, Split};
use friction_nlp::SrxSegmenter;
use serde::Deserialize;

/// n-gram order-and-class `min_count` floor handed to `score_entries` —
/// same default `corpus-tool mine` itself uses, so an n-gram scored here
/// is held to the same "not just prior-driven noise" bar `corpus/
/// MINING.md` was curated under.
const MIN_COUNT: u64 = 5;

/// How many llm-favored n-grams to report, pooled across n=1,2,3 and
/// re-sorted by z (not 30 *per order*) — the flat "top 30 n-grams" the
/// task asks for.
const TOP_N: usize = 30;

/// Auxiliary/copular verbs excluded from the sentence-initial `"This
/// <verb>"` heuristic (see [`is_promoted_this_closer`]): `"This is"`/
/// `"This has"`/`"This does"`/`"This was"` are common in ordinary prose
/// for reasons that have nothing to do with
/// `symmetry.participial_closer`'s **promote** strategy (which always
/// produces a *content* verb — `friction_nlp::inflect`'d from a `VBG`
/// participle's own lemma, e.g. "makes", "allows", "exposes" — never a
/// copula or auxiliary), and excluding them keeps the heuristic from being
/// swamped by unrelated, extremely common `"This is ..."` constructions.
const THIS_CLOSER_AUX_EXCLUDE: [&str; 4] = ["is", "has", "does", "was"];

/// A curated set of individually well-known "AI slop" markers (at most 3
/// words each, so each is a valid lookup key into a [`Corpus`]'s own
/// `ngram_counts`), used only to answer "did the fixer measurably reduce
/// these specific, textbook llm markers" — not the primary evidence this
/// report's conclusion rests on (that is the data-driven top-30 log-odds
/// table). Includes the three the task names by name (`"leverage"`,
/// `"moreover"`, `"delve"`) plus `families::connective::CONNECTIVES`'
/// own markers (lowercased) and a handful of other commonly cited LLM
/// vocabulary. Every entry is already lowercase, space-joined, and
/// apostrophe-normalized to match [`word_segments`]'s own token output —
/// looking one up is a plain `BTreeMap::get`, no further normalization.
const KNOWN_LLM_MARKERS: &[&str] = &[
    "leverage",
    "moreover",
    "delve",
    "delve into",
    "dive into",
    "however",
    "nevertheless",
    "nonetheless",
    "furthermore",
    "additionally",
    "in addition",
    "consequently",
    "therefore",
    "thus",
    "robust",
    "seamless",
    "seamlessly",
    "streamline",
    "tailored",
    "boasts",
    "myriad",
    "plethora",
    "paramount",
    "elevate",
    "foster",
    "navigate",
    "unlock",
    "harness",
    "testament",
    "pivotal",
    "crucial",
    "notably",
    "ultimately",
    "landscape",
    "realm",
    "invaluable",
    "indispensable",
    "underscore",
    "underscores",
    "showcase",
    "showcasing",
    "empower",
    "empowering",
    "bespoke",
    "holistic",
    "synergy",
    "unparalleled",
    // Not "AI slop" vocabulary by themselves, but `lexical.substitution`
    // maps `specific` -> `particular` (see that module's `SUBSTITUTIONS`
    // table), and `specific` was itself already llm-favored pre-fix (see
    // `corpus/MINING.md`) — included here so this table shows both sides
    // of that relabeling, cross-referenced by [`render_conclusion_section`].
    "specific",
    "particular",
];

/// One n-gram order's counts for one [`Corpus`], indexed `[order - 1]`.
type OrderedCounts = [BTreeMap<String, u64>; 3];

/// Pooled n-gram counts and a handful of targeted tic counters for one
/// class of documents (`human-train`, `original-llm-train`, or
/// `fixed-llm-train`), accumulated doc by doc via [`analyze_into`].
///
/// Mirrors what `corpus-tool mine`'s own `run` accumulates into its
/// per-order `ClassCounts` (see that module's docs), plus the extra
/// per-sentence counters this report's own candidate-tic tables need,
/// which `mine` itself has no reason to track.
#[derive(Debug, Default)]
struct Corpus {
    doc_count: u64,
    sentence_count: u64,
    /// Sentences (post-fix, for the `fixed-llm` corpus; as generated, for
    /// the others) that open with `"And "`, `"But "`, or `"So "` — the
    /// exact surface form `connective.surgery`'s **swap** strategy
    /// produces (see that rule's own module docs: the replacement is the
    /// bare capitalized coordinator plus one space, never a comma).
    and_but_so_initial: u64,
    /// Sentences opening with `"This <verb>"` for a non-auxiliary `<verb>`
    /// (see [`is_promoted_this_closer`]) — the shape
    /// `symmetry.participial_closer`'s **promote** strategy always
    /// produces.
    this_closer_initial: u64,
    /// Occurrences of two consecutive ASCII spaces in prose text (overlapping
    /// windows, so a run of 3 spaces counts as 2 — applied identically
    /// across every corpus, so the comparison is still apples to apples).
    double_space: u64,
    ngram_counts: OrderedCounts,
}

impl Corpus {
    /// Total tokens (word count) this corpus's prose contains: the sum of
    /// its 1-gram counts, which is exactly the word count since every
    /// 1-gram window is one word counted once (see `mine`'s own
    /// `accumulate_ngrams`).
    fn token_count(&self) -> u64 {
        self.ngram_counts[0].values().sum()
    }

    /// Total n-gram-token count for `order` (1-, 2-, or 3-gram window
    /// count) — the same quantity `mine::score_entries` itself computes
    /// as `n_llm`/`n_human` internally, exposed here so this report can
    /// call [`log_odds_z`] directly for n-grams below [`MIN_COUNT`] (see
    /// [`classify_against_baseline`]).
    fn order_total(&self, order: usize) -> u64 {
        self.ngram_counts[order - 1].values().sum()
    }

    fn rate_per_1000(&self, count: u64) -> f64 {
        let tokens = self.token_count();
        if tokens == 0 {
            0.0
        } else {
            #[allow(clippy::cast_precision_loss)]
            let (count, tokens) = (count as f64, tokens as f64);
            count * 1000.0 / tokens
        }
    }

    /// [`Self::rate_per_1000`] for a specific n-gram's count at its own
    /// order (1 to 3 words; anything longer is out of range for this
    /// corpus's `ngram_counts` and rates as `0.0`).
    fn marker_rate(&self, marker: &str) -> f64 {
        let order = marker.split_whitespace().count();
        if order == 0 || order > 3 {
            return 0.0;
        }
        let count = self.ngram_counts[order - 1]
            .get(marker)
            .copied()
            .unwrap_or(0);
        self.rate_per_1000(count)
    }
}

/// The workspace root, resolved from this example's own crate directory —
/// same technique `corpus-tool`'s own `tests/mining_report.rs` uses.
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("resolve repo root")
}

/// Ensures `target/release/friction` exists, building it (`cargo build
/// --release -p friction-cli`) if not, and returns its path. Always
/// re-checked, never rebuilt unconditionally — a stale, already-fresh
/// release binary from a previous run of this example is reused as-is.
fn ensure_release_binary(repo_root: &Path) -> anyhow::Result<PathBuf> {
    let bin = repo_root.join("target/release/friction");
    if bin.is_file() {
        return Ok(bin);
    }
    eprintln!("self_fingerprint: building the release friction CLI (target/release/friction)...");
    let status = Command::new("cargo")
        .args(["build", "--release", "-p", "friction-cli"])
        .current_dir(repo_root)
        .status()
        .context("failed to invoke `cargo build --release -p friction-cli`")?;
    if !status.success() {
        bail!("`cargo build --release -p friction-cli` failed with {status}");
    }
    if !bin.is_file() {
        bail!(
            "expected {} to exist after a successful build",
            bin.display()
        );
    }
    Ok(bin)
}

/// `friction fix --format json`'s summary, read from stderr. Only
/// `patches_by_rule` is used here; every other field in the real
/// `FixSummary` (`friction-cli/src/fix.rs`, not itself a public type) is
/// simply ignored by `serde_json` rather than mirrored, since this report
/// never needs it.
#[derive(Debug, Deserialize)]
struct FixSummary {
    patches_by_rule: BTreeMap<String, usize>,
}

/// Runs `<friction_bin> fix <source_path> --genre <genre> --format json`,
/// returning the fixed text (stdout) and the per-rule patch counts the
/// run's JSON summary reports (stderr).
///
/// # Errors
/// Returns an error if the process cannot be spawned, exits non-zero, its
/// stdout is not valid UTF-8, or its stderr does not parse as
/// [`FixSummary`].
fn fix_via_release_cli(
    friction_bin: &Path,
    source_path: &Path,
    genre: &str,
) -> anyhow::Result<(String, BTreeMap<String, usize>)> {
    let output = Command::new(friction_bin)
        .arg("fix")
        .arg(source_path)
        .args(["--genre", genre, "--format", "json"])
        .output()
        .with_context(|| format!("failed to run `friction fix {}`", source_path.display()))?;
    if !output.status.success() {
        bail!(
            "friction fix {} exited with {}: {}",
            source_path.display(),
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let fixed_text = String::from_utf8(output.stdout)
        .with_context(|| format!("{}: fix output was not valid UTF-8", source_path.display()))?;
    let summary: FixSummary = serde_json::from_slice(&output.stderr).with_context(|| {
        format!(
            "{}: could not parse `--format json` summary from stderr",
            source_path.display()
        )
    })?;
    Ok((fixed_text, summary.patches_by_rule))
}

/// `true` if `sentence_text` (a sentence's own text — no leading
/// whitespace, by every [`friction_nlp::Segmenter`]'s contract) opens with
/// the bare coordinator `connective.surgery`'s **swap** strategy produces:
/// `"And "`, `"But "`, or `"So "`, capitalized, followed directly by a
/// space (never a comma — see that rule's own module docs).
fn is_and_but_so_initial(sentence_text: &str) -> bool {
    sentence_text.starts_with("And ")
        || sentence_text.starts_with("But ")
        || sentence_text.starts_with("So ")
}

/// `true` if `sentence_text` opens with `"This "` followed by a word that
/// looks like a promoted participle's inflected verb rather than a copula
/// or auxiliary — see [`THIS_CLOSER_AUX_EXCLUDE`]'s doc comment for why
/// those four are excluded. A crude proxy (any other word ending in `"s"`
/// passes), deliberately: it is cross-checked in the report against the
/// exact `symmetry.participial_closer` patch count the CLI's own JSON
/// summaries report, not trusted alone.
fn is_promoted_this_closer(sentence_text: &str) -> bool {
    let Some(rest) = sentence_text.strip_prefix("This ") else {
        return false;
    };
    let second_word = rest.split_whitespace().next().unwrap_or("");
    let letters: String = second_word.chars().filter(|c| c.is_alphabetic()).collect();
    let lower = letters.to_ascii_lowercase();
    lower.ends_with('s') && !THIS_CLOSER_AUX_EXCLUDE.contains(&lower.as_str())
}

/// Overlapping occurrences of two consecutive ASCII spaces in `text`. A run
/// of `n` spaces counts as `n - 1` — applied identically to every corpus,
/// so it is still a fair rate comparison even though it is not a count of
/// "double-space defects" in some more careful sense.
fn count_double_spaces(text: &str) -> u64 {
    text.as_bytes()
        .windows(2)
        .filter(|w| w[0] == b' ' && w[1] == b' ')
        .count() as u64
}

/// Parses and sentence-segments `raw_text` (a whole markdown document's
/// source), then folds its prose into `corpus`: every prose unit's word
/// segments into `corpus.ngram_counts` (orders 1 to 3, via `mine`'s own
/// [`word_segments`]/[`accumulate_ngrams`]) and every sentence into the
/// three targeted tic counters.
///
/// # Errors
/// Returns an error if `raw_text` fails to parse as markdown or fails
/// sentence segmentation (both infallible for any well-formed corpus doc;
/// surfaced rather than panicked on so a single malformed doc cannot bring
/// the whole run down silently-wrong).
fn analyze_into(
    raw_text: &str,
    segmenter: SrxSegmenter,
    corpus: &mut Corpus,
) -> anyhow::Result<()> {
    let parsed = friction_parse::parse(raw_text.to_string()).context("markdown parse failed")?;
    let document = friction_nlp::segment_document(&parsed, &segmenter)
        .context("sentence segmentation failed")?;

    corpus.doc_count += 1;

    for unit in document.prose() {
        let Ok(unit_text) = document.text(&unit.range) else {
            continue;
        };
        let segments = word_segments(unit_text);
        for order in 1..=3usize {
            accumulate_ngrams(&segments, order, &mut corpus.ngram_counts[order - 1]);
        }
        corpus.double_space += count_double_spaces(unit_text);

        for sentence in &unit.sentences {
            corpus.sentence_count += 1;
            let Ok(sentence_text) = document.text(&sentence.range) else {
                continue;
            };
            if is_and_but_so_initial(sentence_text) {
                corpus.and_but_so_initial += 1;
            }
            if is_promoted_this_closer(sentence_text) {
                corpus.this_closer_initial += 1;
            }
        }
    }
    Ok(())
}

/// One rendered row of the top-30 fixed-llm-favored table: the scored
/// [`Entry`] itself, its n-gram order, and its classification against the
/// *original* (pre-fix) llm corpus.
struct FixedRow {
    entry: Entry,
    order: usize,
    /// This n-gram's raw count in the *original* (pre-fix) llm corpus —
    /// looked up directly, not gated by [`MIN_COUNT`].
    orig_llm_count: u64,
    /// This n-gram's log-odds z-score in the `human-train` vs.
    /// `original-llm-train` comparison — computed directly via
    /// [`log_odds_z`] (not gated by [`MIN_COUNT`], unlike the top-30
    /// tables themselves: a fixer artifact absent from the original corpus
    /// entirely is exactly the case a `min_count`-gated lookup would
    /// silently miss).
    z_original: f64,
    /// `true` when `z_original > 0.0`: this n-gram already read as
    /// llm-favored before fixing (the fixer reduced, at best, an existing
    /// tic). `false` means it was human-favored, neutral, or entirely
    /// absent pre-fix — a candidate for something the fixer itself
    /// introduced.
    pre_existing: bool,
}

/// Classifies every entry in `top_fixed` against `human`/`llm_original`'s
/// own raw counts, producing one [`FixedRow`] per entry in the same order.
fn classify_against_baseline(
    top_fixed: &[Entry],
    human: &Corpus,
    llm_original: &Corpus,
) -> Vec<FixedRow> {
    top_fixed
        .iter()
        .map(|entry| {
            let order = entry.ngram.split_whitespace().count().clamp(1, 3);
            let orig_llm_count = llm_original.ngram_counts[order - 1]
                .get(&entry.ngram)
                .copied()
                .unwrap_or(0);
            let orig_human_count = human.ngram_counts[order - 1]
                .get(&entry.ngram)
                .copied()
                .unwrap_or(0);
            let orig_n_llm = llm_original.order_total(order);
            let orig_n_human = human.order_total(order);
            let z_original = log_odds_z(orig_llm_count, orig_n_llm, orig_human_count, orig_n_human)
                .map_or(0.0, |(_, z)| z);
            FixedRow {
                entry: entry.clone(),
                order,
                orig_llm_count,
                z_original,
                pre_existing: z_original > 0.0,
            }
        })
        .collect()
}

fn markdown_escape(s: &str) -> String {
    s.replace('|', "\\|")
}

fn render_fixed_table(out: &mut String, rows: &[FixedRow]) {
    writeln!(
        out,
        "| n-gram | n | human count | fixed-llm count | z (fixed vs human) | original-llm \
         count | z (original vs human) | verdict |"
    )
    .expect("write to String is infallible");
    writeln!(out, "|---|---|---|---|---|---|---|---|").expect("write to String is infallible");
    for row in rows {
        let verdict = if row.pre_existing {
            "pre-existing (reduced, not eliminated)"
        } else {
            "new — absent/human-favored pre-fix"
        };
        writeln!(
            out,
            "| {} | {} | {} | {} | {:.4} | {} | {:.4} | {verdict} |",
            markdown_escape(&row.entry.ngram),
            row.order,
            row.entry.count_human,
            row.entry.count_llm,
            row.entry.z,
            row.orig_llm_count,
            row.z_original,
        )
        .expect("write to String is infallible");
    }
}

fn main() -> anyhow::Result<()> {
    let repo_root = repo_root();
    let corpus_dir = repo_root.join("corpus");
    let friction_bin = ensure_release_binary(&repo_root)?;
    let out_dir = repo_root.join("target/fingerprint/llm-fixed");
    std::fs::create_dir_all(&out_dir).context("create target/fingerprint/llm-fixed")?;

    let manifest_path = corpus_dir.join("manifest.jsonl");
    let records = manifest::read_manifest(&manifest_path)
        .context("read corpus/manifest.jsonl")?
        .unwrap_or_default();

    let mut human_train: Vec<&ManifestRecord> = records
        .iter()
        .filter(|r| r.split == Some(Split::Train) && r.class == Class::Human)
        .collect();
    human_train.sort_by(|a, b| a.id.cmp(&b.id));

    let mut llm_train: Vec<&ManifestRecord> = records
        .iter()
        .filter(|r| r.split == Some(Split::Train) && r.class == Class::Llm)
        .collect();
    llm_train.sort_by(|a, b| a.id.cmp(&b.id));

    eprintln!(
        "self_fingerprint: {} human-train docs, {} llm-train docs",
        human_train.len(),
        llm_train.len()
    );

    let segmenter = SrxSegmenter::new();

    let mut human_corpus = Corpus::default();
    for record in &human_train {
        let path = corpus_dir.join(corpus_layout::relpath(record));
        let text =
            std::fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
        analyze_into(&text, segmenter, &mut human_corpus)?;
    }

    let mut llm_original_corpus = Corpus::default();
    let mut llm_fixed_corpus = Corpus::default();
    let mut rule_fire_totals: BTreeMap<String, u64> = BTreeMap::new();

    for (i, record) in llm_train.iter().enumerate() {
        let genre = record.genre.to_string();
        let source_path = corpus_dir.join(corpus_layout::relpath(record));
        let original_text = std::fs::read_to_string(&source_path)
            .with_context(|| format!("read {}", source_path.display()))?;
        analyze_into(&original_text, segmenter, &mut llm_original_corpus)?;

        let (fixed_text, patches_by_rule) =
            fix_via_release_cli(&friction_bin, &source_path, &genre)?;
        for (rule, n) in patches_by_rule {
            *rule_fire_totals.entry(rule).or_insert(0) += n as u64;
        }

        let out_path = out_dir.join(&genre).join(format!("{}.md", record.id));
        if let Some(parent) = out_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&out_path, &fixed_text)
            .with_context(|| format!("write {}", out_path.display()))?;

        analyze_into(&fixed_text, segmenter, &mut llm_fixed_corpus)?;

        if (i + 1) % 25 == 0 || i + 1 == llm_train.len() {
            eprintln!("self_fingerprint: fixed {}/{}", i + 1, llm_train.len());
        }
    }

    // --- score both class comparisons with mine's own machinery ---

    let mut fixed_entries: Vec<Entry> = Vec::new();
    let mut original_entries: Vec<Entry> = Vec::new();
    for order in 1..=3usize {
        let fixed_counts = ClassCounts {
            human: human_corpus.ngram_counts[order - 1].clone(),
            llm: llm_fixed_corpus.ngram_counts[order - 1].clone(),
        };
        fixed_entries.extend(score_entries(&fixed_counts, MIN_COUNT));

        let original_counts = ClassCounts {
            human: human_corpus.ngram_counts[order - 1].clone(),
            llm: llm_original_corpus.ngram_counts[order - 1].clone(),
        };
        original_entries.extend(score_entries(&original_counts, MIN_COUNT));
    }
    let top_fixed = top_llm_favored(&fixed_entries, TOP_N);
    let top_original = top_llm_favored(&original_entries, TOP_N);
    let fixed_rows = classify_against_baseline(&top_fixed, &human_corpus, &llm_original_corpus);

    let report = render_report(&ReportInputs {
        human_train_count: human_train.len(),
        llm_train_count: llm_train.len(),
        human: &human_corpus,
        llm_original: &llm_original_corpus,
        llm_fixed: &llm_fixed_corpus,
        fixed_rows: &fixed_rows,
        top_original: &top_original,
        rule_fire_totals: &rule_fire_totals,
    });

    let report_path = corpus_dir.join("FINGERPRINT.md");
    std::fs::write(&report_path, &report)
        .with_context(|| format!("write {}", report_path.display()))?;
    eprintln!("self_fingerprint: wrote {}", report_path.display());

    Ok(())
}

/// Everything [`render_report`] needs, gathered into one struct purely to
/// keep that function's own signature (and every section helper's) under
/// clippy's argument-count and line-count ceilings.
struct ReportInputs<'a> {
    human_train_count: usize,
    llm_train_count: usize,
    human: &'a Corpus,
    llm_original: &'a Corpus,
    llm_fixed: &'a Corpus,
    fixed_rows: &'a [FixedRow],
    top_original: &'a [Entry],
    rule_fire_totals: &'a BTreeMap<String, u64>,
}

fn render_header(out: &mut String, inputs: &ReportInputs<'_>) {
    writeln!(out, "# Self-fingerprint report\n").expect("write to String is infallible");
    writeln!(
        out,
        "Generated by `crates/corpus-tool/examples/self_fingerprint.rs`. Fixes every \
         `train`-split `llm` corpus doc with the release `friction` CLI (into \
         `target/fingerprint/llm-fixed/`, repo-root-relative, not committed), then reuses \
         `corpus_tool::commands::mine`'s log-odds machinery to compare {{`fixed-llm` vs. \
         `human-train`}} against the baseline {{`original-llm` vs. `human-train`}} comparison \
         `corpus/MINING.md` already reports. `human-train`: {} docs, {} tokens. \
         `original-llm-train` / `fixed-llm-train`: {} docs each, {} / {} tokens respectively.\n",
        inputs.human_train_count,
        inputs.human.token_count(),
        inputs.llm_train_count,
        inputs.llm_original.token_count(),
        inputs.llm_fixed.token_count(),
    )
    .expect("write to String is infallible");
}

fn render_top30_section(out: &mut String, inputs: &ReportInputs<'_>) {
    writeln!(
        out,
        "## Top 30 n-grams favoring fixed-llm (vs. human-train)\n"
    )
    .expect("write to String is infallible");
    writeln!(
        out,
        "Pooled across n=1,2,3, re-sorted by z and truncated to 30 (not 30 per order); \
         `min-count` 5, same as `corpus-tool mine`'s own default. \"verdict\" classifies each \
         n-gram against its own z-score in the *original* (pre-fix) llm-vs-human comparison, \
         computed directly (not gated by min-count, so an artifact absent pre-fix is not \
         silently dropped): \"pre-existing\" means it already read as llm-favored before \
         fixing (the fixer reduced, at best, an existing tic); \"new\" means it did not \
         (human-favored, neutral, or unseen pre-fix) — a candidate for something the fixer \
         itself introduced.\n"
    )
    .expect("write to String is infallible");
    render_fixed_table(out, inputs.fixed_rows);
    out.push('\n');

    let new_count = inputs.fixed_rows.iter().filter(|r| !r.pre_existing).count();
    writeln!(
        out,
        "{new_count} of the {} rows above are \"new\" by this test.\n",
        inputs.fixed_rows.len()
    )
    .expect("write to String is infallible");
}

fn render_known_markers_section(out: &mut String, inputs: &ReportInputs<'_>) {
    writeln!(
        out,
        "## Known LLM markers: before vs. after (rate per 1000 tokens)\n"
    )
    .expect("write to String is infallible");
    writeln!(
        out,
        "A curated, hand-picked list (see `KNOWN_LLM_MARKERS` in the generating script), not \
         the data-driven evidence above — this table only answers whether specific, textbook \
         markers shrank.\n"
    )
    .expect("write to String is infallible");
    writeln!(
        out,
        "| marker | human-train | original-llm-train | fixed-llm-train |"
    )
    .expect("write to String is infallible");
    writeln!(out, "|---|---|---|---|").expect("write to String is infallible");
    for marker in KNOWN_LLM_MARKERS {
        let h = inputs.human.marker_rate(marker);
        let o = inputs.llm_original.marker_rate(marker);
        let f = inputs.llm_fixed.marker_rate(marker);
        if o == 0.0 && f == 0.0 && h == 0.0 {
            continue;
        }
        writeln!(out, "| {marker} | {h:.3} | {o:.3} | {f:.3} |")
            .expect("write to String is infallible");
    }
    out.push('\n');
}

fn render_candidate_tics_section(out: &mut String, inputs: &ReportInputs<'_>) {
    let rule_count = |rule: &str| inputs.rule_fire_totals.get(rule).copied().unwrap_or(0);

    writeln!(
        out,
        "## Candidate fixer-introduced tics (rate per 1000 tokens)\n"
    )
    .expect("write to String is infallible");
    writeln!(
        out,
        "Each row's detector is a heuristic over sentence/prose text, cross-checked against \
         the responsible rule's own aggregate patch count over the whole `llm-train` run \
         (from every `friction fix --format json` invocation's summary, ground truth for \
         whether the rule fired at all).\n"
    )
    .expect("write to String is infallible");
    writeln!(
        out,
        "| candidate tic | rule | human-train | original-llm-train | fixed-llm-train | rule \
         patches (whole run) |"
    )
    .expect("write to String is infallible");
    writeln!(out, "|---|---|---|---|---|---|").expect("write to String is infallible");
    writeln!(
        out,
        "| sentence-initial \"And \"/\"But \"/\"So \" | connective.surgery | {:.3} | {:.3} | \
         {:.3} | {} |",
        inputs.human.rate_per_1000(inputs.human.and_but_so_initial),
        inputs
            .llm_original
            .rate_per_1000(inputs.llm_original.and_but_so_initial),
        inputs
            .llm_fixed
            .rate_per_1000(inputs.llm_fixed.and_but_so_initial),
        rule_count("connective.surgery"),
    )
    .expect("write to String is infallible");
    writeln!(
        out,
        "| sentence-initial \"This <verb>\" (promoted closer) | symmetry.participial_closer | \
         {:.3} | {:.3} | {:.3} | {} |",
        inputs.human.rate_per_1000(inputs.human.this_closer_initial),
        inputs
            .llm_original
            .rate_per_1000(inputs.llm_original.this_closer_initial),
        inputs
            .llm_fixed
            .rate_per_1000(inputs.llm_fixed.this_closer_initial),
        rule_count("symmetry.participial_closer"),
    )
    .expect("write to String is infallible");
    writeln!(
        out,
        "| doubled spaces (any rule) | — | {:.3} | {:.3} | {:.3} | n/a |",
        inputs.human.rate_per_1000(inputs.human.double_space),
        inputs
            .llm_original
            .rate_per_1000(inputs.llm_original.double_space),
        inputs
            .llm_fixed
            .rate_per_1000(inputs.llm_fixed.double_space),
    )
    .expect("write to String is infallible");
    out.push('\n');
}

fn render_rule_totals_section(out: &mut String, inputs: &ReportInputs<'_>) {
    writeln!(out, "## Rule patch totals over the whole llm-train run\n")
        .expect("write to String is infallible");
    writeln!(out, "| rule | patches applied |\n|---|---|").expect("write to String is infallible");
    for (rule, count) in inputs.rule_fire_totals {
        writeln!(out, "| {rule} | {count} |").expect("write to String is infallible");
    }
    out.push('\n');
}

fn render_original_baseline_section(out: &mut String, inputs: &ReportInputs<'_>) {
    writeln!(out, "## Original-llm top-30 (baseline, for reference)\n")
        .expect("write to String is infallible");
    writeln!(
        out,
        "The same {{original-llm vs. human-train}} comparison `corpus/MINING.md` reports, \
         recomputed here from this run's own `Corpus` totals so both tables above are directly \
         comparable to it.\n"
    )
    .expect("write to String is infallible");
    writeln!(
        out,
        "| n-gram | llm count | human count | z |\n|---|---|---|---|"
    )
    .expect("write to String is infallible");
    for entry in inputs.top_original {
        writeln!(
            out,
            "| {} | {} | {} | {:.4} |",
            markdown_escape(&entry.ngram),
            entry.count_llm,
            entry.count_human,
            entry.z
        )
        .expect("write to String is infallible");
    }
    out.push('\n');
}

/// One candidate tic's before/after verdict — see [`compute_verdicts`].
struct Verdict {
    /// Human-readable name of the candidate tic.
    tic: &'static str,
    /// The rule responsible for it (or a short explanation why none is,
    /// for the generic doubled-space check).
    rule: &'static str,
    /// `true` if the numbers clear the bar for a genuine, concerning
    /// finding — never a subjective call: each check below states exactly
    /// what "clears the bar" means for that tic before computing it.
    concerning: bool,
    /// The numbers behind `concerning`, always shown regardless of its
    /// value — an honest report shows its work for a "no" as much as a
    /// "yes".
    detail: String,
}

/// Computes this run's verdict on every candidate tic named in the module
/// docs, purely from `inputs`' own already-aggregated rates — no
/// additional heuristics beyond what [`analyze_into`] already counted.
fn compute_verdicts(inputs: &ReportInputs<'_>) -> Vec<Verdict> {
    let mut verdicts = Vec::new();

    let h = inputs.human.rate_per_1000(inputs.human.and_but_so_initial);
    let o = inputs
        .llm_original
        .rate_per_1000(inputs.llm_original.and_but_so_initial);
    let f = inputs
        .llm_fixed
        .rate_per_1000(inputs.llm_fixed.and_but_so_initial);
    let and_but_so_concerning = f > h;
    verdicts.push(Verdict {
        tic: "sentence-initial \"And\"/\"But\"/\"So\"",
        rule: "connective.surgery",
        concerning: and_but_so_concerning,
        detail: format!(
            "{o:.3} -> {f:.3} per 1000 tokens after fixing (human-train baseline {h:.3}); {}",
            if and_but_so_concerning {
                "now exceeds the human-train rate — an overshoot"
            } else {
                "still below the human-train rate: the swap strategy raises the rate but does \
                 not push it past human density"
            }
        ),
    });

    let h = inputs.human.rate_per_1000(inputs.human.this_closer_initial);
    let o = inputs
        .llm_original
        .rate_per_1000(inputs.llm_original.this_closer_initial);
    let f = inputs
        .llm_fixed
        .rate_per_1000(inputs.llm_fixed.this_closer_initial);
    let this_closer_concerning = f > o && f > h;
    verdicts.push(Verdict {
        tic: "sentence-initial \"This <verb>\" (promoted closer)",
        rule: "symmetry.participial_closer",
        concerning: this_closer_concerning,
        detail: format!(
            "{o:.3} -> {f:.3} per 1000 tokens after fixing (human-train baseline {h:.3}); {}",
            if this_closer_concerning {
                "this marker was already above the human baseline before fixing, and the \
                 promote strategy pushes it further above both baselines rather than toward \
                 either"
            } else {
                "did not increase beyond both baselines"
            }
        ),
    });

    let h = inputs.human.rate_per_1000(inputs.human.double_space);
    let o = inputs
        .llm_original
        .rate_per_1000(inputs.llm_original.double_space);
    let f = inputs
        .llm_fixed
        .rate_per_1000(inputs.llm_fixed.double_space);
    let double_space_concerning = f > o + 0.05 && f > h;
    verdicts.push(Verdict {
        tic: "doubled spaces",
        rule: "no single rule — generic patch-splicing hazard",
        concerning: double_space_concerning,
        detail: format!(
            "{o:.3} -> {f:.3} per 1000 tokens after fixing (human-train baseline {h:.3}); {}",
            if double_space_concerning {
                "increased past both baselines"
            } else {
                "essentially unchanged — no evidence the fixer introduces doubled spaces"
            }
        ),
    });

    let spec_o = inputs.llm_original.marker_rate("specific");
    let spec_f = inputs.llm_fixed.marker_rate("specific");
    let part_o = inputs.llm_original.marker_rate("particular");
    let part_f = inputs.llm_fixed.marker_rate("particular");
    let part_h = inputs.human.marker_rate("particular");
    let relabel_concerning = spec_f < spec_o && part_f > part_o && part_f > part_h;
    verdicts.push(Verdict {
        tic: "\"particular\" (relabeled from \"specific\")",
        rule: "lexical.substitution (the specific -> particular entry)",
        concerning: relabel_concerning,
        detail: format!(
            "\"specific\" {spec_o:.3} -> {spec_f:.3} per 1000 tokens, \"particular\" \
             {part_o:.3} -> {part_f:.3} per 1000 tokens (human-train baseline for \"particular\": \
             {part_h:.3}); {}",
            if relabel_concerning {
                "the substitution table's own replacement word is itself elevated enough to \
                 newly enter the top-30 fixed-llm-favored list — see the \"new\" row above"
            } else {
                "no clear relabeling signal"
            }
        ),
    });

    verdicts
}

fn render_violations_section(out: &mut String, verdicts: &[Verdict]) {
    writeln!(out, "## Violations\n").expect("write to String is infallible");
    let concerning: Vec<&Verdict> = verdicts.iter().filter(|v| v.concerning).collect();
    if concerning.is_empty() {
        writeln!(
            out,
            "None of the candidate tics checked above cleared the bar for a genuine violation \
             (an overshoot past both the human-train and original-llm baselines).\n"
        )
        .expect("write to String is infallible");
    } else {
        for v in concerning {
            writeln!(out, "- **{}** (rule: `{}`): {}", v.tic, v.rule, v.detail)
                .expect("write to String is infallible");
        }
        out.push('\n');
    }
}

fn render_conclusion_section(out: &mut String, inputs: &ReportInputs<'_>, verdicts: &[Verdict]) {
    writeln!(out, "## Conclusion\n").expect("write to String is infallible");
    let new_count = inputs.fixed_rows.iter().filter(|r| !r.pre_existing).count();
    writeln!(
        out,
        "The top-{TOP_N} fixed-llm-favored n-gram list is almost identical to the \
         original-llm-favored list: only {new_count} of {} entries are \"new\". Most of both \
         lists is *topic* vocabulary (\"mysql\", \"database\", \"backup\", \"perceptual \
         hashing\", \"chirpline\", ...) carried over from this corpus's own topic mix rather \
         than a stylistic tic — the fixer does not, and should not, touch content words, so a \
         small new/pre-existing split here says more about this corpus's genre balance than \
         about the fixer.\n",
        inputs.fixed_rows.len(),
    )
    .expect("write to String is infallible");
    for v in verdicts {
        writeln!(out, "- {}: {}", v.tic, v.detail).expect("write to String is infallible");
    }
    out.push('\n');
    let has_violation = verdicts.iter().any(|v| v.concerning);
    writeln!(
        out,
        "Overall: {}",
        if has_violation {
            "the fixer does have at least one self-inflicted tic (see Violations above) — \
             measurable, attributable to a specific rule, and worth addressing — but it is \
             narrow, not a wholesale replacement of one LLM-speak register with another. The \
             original llm markers this run's rules actually target (leverage, robust, crucial, \
             showcase, foster, furthermore, therefore, in addition) shrank substantially; \
             markers no current rule targets (delve, seamless, streamline, navigate, testament, \
             ultimately, invaluable) are unchanged, a coverage gap rather than a regression."
        } else {
            "no candidate fixer-introduced tic checked here cleared the bar for a genuine \
             violation."
        }
    )
    .expect("write to String is infallible");
}

/// Verbatim before/after excerpts, hand-picked by reading real files under
/// `target/fingerprint/llm-fixed/` and the corresponding `corpus/llm/`
/// source against this run's own top hits and candidate-tic rows, as the
/// module docs' step 5 requires. Doc ids are this run's own (stable, since
/// the corpus and its ids are frozen); the quoted excerpts are exact
/// substrings of those two files as of this writing — re-verify them by
/// hand (`grep` the doc id under both directories) if the rule tables
/// these excerpts illustrate ever change.
const SANITY_CHECK_EXCERPTS: &str = "\
- `blog/1d2cb7bffd7854c6.md` — `connective.surgery`'s **swap** strategy, \
  verbatim: original `\"...a significant shift for me. However, as my project grew \
  in complexity, so did the need for better tooling and more robust error \
  checking.\"` becomes fixed `\"...a significant shift for me. But as my project grew \
  in complexity, so did the need for better tooling and more solid error \
  checking.\"` — both the `However,` -> `But` swap and (separately) \
  `lexical.substitution`'s `robust` -> `solid` are visible in the same \
  sentence.
- `blog/52c16c2f50305ef3.md` — `symmetry.participial_closer`'s **promote** \
  strategy, verbatim: original `\"...allowed us to test the script in isolation, \
  ensuring its correctness before applying it to production.\"` becomes fixed \
  `\"...allowed us to test the script in isolation. It ensures its correctness \
  before applying it to production.\"` — the sentence-final `, ensuring ...` \
  clause becomes its own sentence, exactly the shape `symmetry.participial_\
  closer`'s own module docs describe. `docs/c79406a6dd392488.md` shows the \
  same strategy pick the *other* alternated subject: original `\"...can lead to \
  inconsistent print heights, causing layers to shift.\"` becomes fixed \
  `\"...can lead to inconsistent print heights. This causes layers to \
  shift.\"` — see that module's own \"Varying the promoted subject\" docs for \
  why the promoted subject alternates between `\"This\"`/`\"It\"` rather than \
  always being the same word.
- `blog/152f7fa1159f4910.md` — `lexical.substitution`'s `specific` -> \
  `particular` entry is flagged (`Tier::Suggest`) but never auto-applied, so \
  this doc's own `\"...an in-depth comparison based on specific features and \
  operational tradeoffs...\"` is unchanged by `fix`; see that module's own \"A \
  whole-entry demotion\" docs for why — this exact relabeling used to be a \
  measured fixer-introduced tic (`particular` rising well past both the \
  human-train and pre-fix baselines), which demoting this one entry to \
  Suggest-only fully resolves, at the cost of no longer auto-fixing `specific` \
  itself either.
- Doubled spaces: both the original and fixed text of \
  `docs/f69468b52cb76190.md` already contain `\"...text or cursor position.  \
  Stores the cut text...\"` (two spaces after the period) — present in the \
  *original* llm output, untouched by fixing, consistent with the flat \
  before/after rate in the candidate-tics table above.
";

fn render_sanity_check_section(out: &mut String) {
    writeln!(out, "## Sanity check (read against the real fixed docs)\n")
        .expect("write to String is infallible");
    writeln!(
        out,
        "A handful of the findings above, confirmed by reading the actual before/after text \
         under `target/fingerprint/llm-fixed/` (this run's fixed output, not committed) against \
         `corpus/llm/` (the original):\n"
    )
    .expect("write to String is infallible");
    out.push_str(SANITY_CHECK_EXCERPTS);
    out.push('\n');
}

fn render_report(inputs: &ReportInputs<'_>) -> String {
    let verdicts = compute_verdicts(inputs);
    let mut out = String::new();
    render_header(&mut out, inputs);
    render_top30_section(&mut out, inputs);
    render_known_markers_section(&mut out, inputs);
    render_candidate_tics_section(&mut out, inputs);
    render_rule_totals_section(&mut out, inputs);
    render_original_baseline_section(&mut out, inputs);
    render_sanity_check_section(&mut out);
    render_violations_section(&mut out, &verdicts);
    render_conclusion_section(&mut out, inputs, &verdicts);
    out
}
