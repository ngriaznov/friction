//! `corpus-tool stats` — prints per-`(class, genre)` corpus statistics.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::PathBuf;

use clap::Args as ClapArgs;

use crate::corpus_layout::relpath;
use crate::hashing::word_count;
use crate::manifest::{self, Class, Genre, ManifestRecord, Split};

/// Arguments for `corpus-tool stats`.
#[derive(Debug, ClapArgs)]
pub struct Args {
    /// Corpus root directory.
    #[arg(long, default_value = "corpus")]
    pub corpus_dir: PathBuf,
    /// Write the markdown report to this path instead of stdout.
    #[arg(long)]
    pub report: Option<PathBuf>,
}

#[derive(Debug, Default, Clone, Copy)]
struct WordStats {
    count: usize,
    min: usize,
    max: usize,
    sum: usize,
}

impl WordStats {
    fn push(&mut self, words: usize) {
        if self.count == 0 {
            self.min = words;
            self.max = words;
        } else {
            self.min = self.min.min(words);
            self.max = self.max.max(words);
        }
        self.sum += words;
        self.count += 1;
    }

    fn mean(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            #[allow(clippy::cast_precision_loss)]
            let mean = self.sum as f64 / self.count as f64;
            mean
        }
    }
}

#[derive(Debug, Default)]
struct CellStats {
    docs: usize,
    words: WordStats,
    train: usize,
    dev: usize,
    holdout: usize,
    unsplit: usize,
}

/// Runs `stats`.
///
/// Prints per-`(class, genre)` doc counts, word-count summary stats
/// (min/mean/max), and split counts, in deterministic
/// (`(class, genre)`-sorted) order. Writes a markdown report to
/// `--report <path>` if given, else prints it to stdout.
///
/// # Errors
///
/// Returns an error if the manifest can't be read or the report can't be
/// written.
pub fn run(args: &Args) -> anyhow::Result<()> {
    let manifest_path = args.corpus_dir.join("manifest.jsonl");
    let records = manifest::read_manifest(&manifest_path)?.unwrap_or_default();

    let mut cells: BTreeMap<(Class, Genre), CellStats> = BTreeMap::new();
    for record in &records {
        let cell = cells.entry((record.class, record.genre)).or_default();
        cell.docs += 1;
        match record.split {
            Some(Split::Train) => cell.train += 1,
            Some(Split::Dev) => cell.dev += 1,
            Some(Split::Holdout) => cell.holdout += 1,
            None => cell.unsplit += 1,
        }
        if let Ok(bytes) = std::fs::read(args.corpus_dir.join(relpath(record)))
            && let Ok(text) = std::str::from_utf8(&bytes)
        {
            cell.words.push(word_count(text));
        }
    }

    let report = render_markdown(&records, &cells);

    match &args.report {
        Some(path) => std::fs::write(path, &report)?,
        None => print!("{report}"),
    }

    Ok(())
}

fn render_markdown(
    records: &[ManifestRecord],
    cells: &BTreeMap<(Class, Genre), CellStats>,
) -> String {
    let mut out = String::new();
    writeln!(out, "# Corpus statistics").expect("write to String is infallible");
    writeln!(out).expect("write to String is infallible");
    writeln!(out, "Total docs: {}", records.len()).expect("write to String is infallible");
    writeln!(out).expect("write to String is infallible");
    writeln!(
        out,
        "| class | genre | docs | words min | words mean | words max | train | dev | holdout | unsplit |"
    )
    .expect("write to String is infallible");
    writeln!(out, "|---|---|---|---|---|---|---|---|---|---|")
        .expect("write to String is infallible");
    for (&(class, genre), cell) in cells {
        writeln!(
            out,
            "| {class} | {genre} | {} | {} | {:.1} | {} | {} | {} | {} | {} |",
            cell.docs,
            cell.words.min,
            cell.words.mean(),
            cell.words.max,
            cell.train,
            cell.dev,
            cell.holdout,
            cell.unsplit,
        )
        .expect("write to String is infallible");
    }
    out
}
