//! Human near-no-op report: runs `fix_document` over human-class corpus
//! docs, per genre, and reports what fraction of sentences received at
//! least one machine patch.
//!
//! Run with:
//!
//! ```text
//! cargo run -p friction-apply --release --example near_noop_report
//! ```
//!
//! Measures the TRAIN split first (the split rule gating was tuned
//! against), the DEV split second (held out, evaluation only — no tuning
//! happens after seeing this number), and the HOLDOUT split third (the
//! sealed, one-shot evaluation split — see `corpus-tool holdout-check`;
//! by the time this number is seen, no code/threshold/pack/rule change is
//! permitted in response to it either). Prints all three breakdowns to
//! stdout and writes them, in the same deterministic format, to
//! `corpus/NEARNOOP.md`.
//!
//! # What "a sentence received a patch" means
//!
//! A sentence counts as touched if any applied patch (from any round of
//! that document's `fix_document` run) replaced a span overlapping the
//! sentence's own byte range in the *original* document — computed via
//! [`friction_apply::touched_original_ranges`], which maps a possibly
//! multi-round fix back to original-document coordinates using the exact
//! patches applied, not an approximate diff.

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use corpus_tool::corpus_layout::relpath;
use corpus_tool::manifest::{Class, Genre, ManifestRecord, Split, read_manifest};
use friction_apply::{FixEngine, touched_original_ranges};
use friction_core::span::ranges_overlap;
use friction_nlp::SrxSegmenter;

/// The fixed set of five genres, in report order — matches their
/// declaration order in `corpus_tool::manifest::Genre`.
const GENRES: [Genre; 5] = [
    Genre::Docs,
    Genre::Blog,
    Genre::Readme,
    Genre::Email,
    Genre::Forum,
];

/// Per-genre sentence counts for one split.
#[derive(Debug, Clone, Copy, Default)]
struct GenreStats {
    docs: usize,
    total_sentences: usize,
    touched_sentences: usize,
}

impl GenreStats {
    fn percent(self) -> f64 {
        if self.total_sentences == 0 {
            0.0
        } else {
            #[allow(clippy::cast_precision_loss)]
            let pct = 100.0 * self.touched_sentences as f64 / self.total_sentences as f64;
            pct
        }
    }
}

fn corpus_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus")
}

/// Total sentences and touched-sentence count for one document.
///
/// # Panics
/// Panics if `source` fails to parse or segment — a corpus fixture bug,
/// not a condition this report should silently paper over.
fn doc_sentence_stats(engine: &FixEngine, source: &str, genre: &str) -> (usize, usize) {
    let segmenter = SrxSegmenter::new();
    let parsed = friction_parse::parse(source).expect("corpus doc must parse as markdown");
    let with_sentences = friction_nlp::segment_document(&parsed, &segmenter)
        .expect("corpus doc must sentence-segment");
    let sentence_ranges: Vec<_> = with_sentences
        .prose()
        .iter()
        .flat_map(|unit| unit.sentences.iter().map(|s| s.range.clone()))
        .collect();

    let (_output, report) = engine
        .fix_document(source, genre)
        .expect("corpus doc must run through fix_document");
    let touched = touched_original_ranges(source.len(), &report.rounds);

    let touched_sentences = sentence_ranges
        .iter()
        .filter(|s| touched.iter().any(|t| ranges_overlap(s, t)))
        .count();

    (sentence_ranges.len(), touched_sentences)
}

/// Loads the manifest and returns every human-class record for `split`,
/// sorted by id for a deterministic processing order.
fn human_records(corpus_dir: &Path, split: Split) -> Vec<ManifestRecord> {
    let manifest_path = corpus_dir.join("manifest.jsonl");
    let mut records: Vec<ManifestRecord> = read_manifest(&manifest_path)
        .expect("manifest.jsonl must read")
        .expect("manifest.jsonl must exist")
        .into_iter()
        .filter(|r| r.class == Class::Human && r.split == Some(split))
        .collect();
    records.sort_by(|a, b| a.id.cmp(&b.id));
    records
}

/// Runs the report over every human record in `split`, returning
/// per-genre stats (in [`GENRES`] order) plus the overall total.
fn run_split(engine: &FixEngine, corpus_dir: &Path, split: Split) -> ([GenreStats; 5], GenreStats) {
    let mut per_genre: [GenreStats; 5] = [GenreStats::default(); 5];
    let records = human_records(corpus_dir, split);

    for record in &records {
        let idx = GENRES
            .iter()
            .position(|g| *g == record.genre)
            .expect("GENRES lists every corpus_tool::manifest::Genre variant");
        let path = corpus_dir.join(relpath(record));
        let source = fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("{}: failed to read: {e}", path.display()));
        let genre_str = record.genre.to_string();
        let (total, touched) = doc_sentence_stats(engine, &source, &genre_str);

        per_genre[idx].docs += 1;
        per_genre[idx].total_sentences += total;
        per_genre[idx].touched_sentences += touched;
    }

    let mut overall = GenreStats::default();
    for stats in &per_genre {
        overall.docs += stats.docs;
        overall.total_sentences += stats.total_sentences;
        overall.touched_sentences += stats.touched_sentences;
    }

    (per_genre, overall)
}

fn render_table(out: &mut String, per_genre: &[GenreStats; 5], overall: GenreStats) {
    writeln!(out, "| genre | docs | sentences | touched | % |")
        .expect("write to String is infallible");
    writeln!(out, "|---|---:|---:|---:|---:|").expect("write to String is infallible");
    for (genre, stats) in GENRES.iter().zip(per_genre) {
        writeln!(
            out,
            "| {genre} | {} | {} | {} | {:.3} |",
            stats.docs,
            stats.total_sentences,
            stats.touched_sentences,
            stats.percent()
        )
        .expect("write to String is infallible");
    }
    writeln!(
        out,
        "| **overall** | {} | {} | {} | {:.3} |",
        overall.docs,
        overall.total_sentences,
        overall.touched_sentences,
        overall.percent()
    )
    .expect("write to String is infallible");
}

fn main() {
    let corpus_dir = corpus_dir();
    let engine = FixEngine::new().expect("embedded tagger model must load");

    let (train_per_genre, train_overall) = run_split(&engine, &corpus_dir, Split::Train);
    let (dev_per_genre, dev_overall) = run_split(&engine, &corpus_dir, Split::Dev);
    let (holdout_per_genre, holdout_overall) = run_split(&engine, &corpus_dir, Split::Holdout);

    let mut report = String::new();
    writeln!(report, "# Human near-no-op report").expect("write to String is infallible");
    writeln!(report).expect("write to String is infallible");
    writeln!(
        report,
        "Generated by `cargo run -p friction-apply --release --example near_noop_report`."
    )
    .expect("write to String is infallible");
    writeln!(
        report,
        "Fraction of human-class corpus sentences that received at least one \
         `fix_document` patch, per genre and overall — the gate is `<= 2.0%` overall."
    )
    .expect("write to String is infallible");
    writeln!(report).expect("write to String is infallible");
    writeln!(report, "## TRAIN split (tuning target)").expect("write to String is infallible");
    writeln!(report).expect("write to String is infallible");
    render_table(&mut report, &train_per_genre, train_overall);
    writeln!(report).expect("write to String is infallible");
    writeln!(report, "## DEV split (evaluation only, no further tuning)")
        .expect("write to String is infallible");
    writeln!(report).expect("write to String is infallible");
    render_table(&mut report, &dev_per_genre, dev_overall);
    writeln!(report).expect("write to String is infallible");
    writeln!(
        report,
        "## HOLDOUT split (sealed, one-shot evaluation — see `corpus-tool holdout-check`)"
    )
    .expect("write to String is infallible");
    writeln!(report).expect("write to String is infallible");
    render_table(&mut report, &holdout_per_genre, holdout_overall);
    writeln!(report).expect("write to String is infallible");

    let out_path = corpus_dir.join("NEARNOOP.md");
    fs::write(&out_path, &report).expect("must write corpus/NEARNOOP.md");

    println!("{report}");
    println!("wrote {}", out_path.display());

    assert!(
        train_overall.percent() <= 2.0,
        "TRAIN overall near-no-op percentage {:.3}% exceeds the 2.0% cap",
        train_overall.percent()
    );
    assert!(
        holdout_overall.percent() <= 2.0,
        "HOLDOUT overall near-no-op percentage {:.3}% exceeds the 2.0% cap",
        holdout_overall.percent()
    );
}
