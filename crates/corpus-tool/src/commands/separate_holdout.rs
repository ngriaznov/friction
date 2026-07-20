//! `corpus-tool separate-holdout` — the sealed-holdout evaluation.
//!
//! Runs once, over the frozen holdout split (see `corpus-tool
//! holdout-check`), comparing three groups of documents per genre:
//!
//! - **human-holdout**: untouched human prose.
//! - **llm-holdout**: untouched LLM prose — the baseline.
//! - **fixed-llm-holdout**: the same LLM-holdout documents after being run
//!   through `friction fix --genre <genre>`, using the release binary at
//!   `--friction-bin` as a subprocess (never this crate's own copy of the
//!   fix engine — the holdout run measures the exact shipped artifact,
//!   not a library call that happens to share its logic) — the tool's
//!   measured effect.
//!
//! For each genre this reports two AUCs (human vs llm-raw, human vs
//! llm-fixed, both via [`crate::commands::separate::mann_whitney_auc`] on
//! the combined score) and the combined score's mean/median for all three
//! groups — reusing every scoring primitive `crate::commands::separate`
//! already defines rather than redefining any of them, so a holdout
//! number and its dev-split counterpart are never computed two different
//! ways by accident.
//!
//! This command changes no metric, envelope, or rule: it is measurement
//! plumbing only, scored against the envelope pack `corpus-tool envelope`
//! already froze from the train split (`--envelope`, defaulted to the
//! same shipped `envelope-v2.toml` `separate` uses). As with `separate`,
//! quarantined (CC-BY-SA) human docs are not excluded — quarantine only
//! restricts redistributing document *text*, not measuring it.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context as _, bail};
use clap::Args as ClapArgs;

use crate::commands::separate::{
    ALL_GENRES, LoadedPack, MetricBand, combined_score, inclusion_note, load_envelope_pack,
    mann_whitney_auc,
};
use crate::corpus_layout::relpath;
use crate::manifest::{self, Class, Genre, ManifestRecord, Split};
use crate::metric_source::{FrictionMetricsSource, MetricSource, load_document};
use friction_core::MetricVector;

/// Arguments for `corpus-tool separate-holdout`.
#[derive(Debug, ClapArgs)]
pub struct Args {
    /// Corpus root directory.
    #[arg(long, default_value = "corpus")]
    pub corpus_dir: PathBuf,
    /// Path to the envelope pack (`corpus-tool envelope`'s output, the
    /// same one `separate` scores the dev split against) used for the
    /// combined score. Already frozen before this run; never refit
    /// against holdout data.
    #[arg(long, default_value = "crates/friction-packs/packs/envelope-v2.toml")]
    pub envelope: PathBuf,
    /// Path to the release `friction` binary. Invoked as `<bin> fix
    /// <path> --genre <genre>`, one subprocess per llm-holdout doc, to
    /// produce that doc's fixed text — the exact shipped artifact, never
    /// this tool's own in-process copy of the fix engine.
    #[arg(long, default_value = "target/release/friction")]
    pub friction_bin: PathBuf,
    /// Path to write the markdown holdout report to.
    #[arg(long)]
    pub report: PathBuf,
}

/// Runs `separate-holdout`.
///
/// Loads every `split: holdout` manifest record, computes each
/// human-holdout and llm-holdout document's [`MetricVector`] directly,
/// and — for every llm-holdout document — additionally shells out to
/// `--friction-bin fix --genre <that doc's genre>`, writes its stdout
/// into a fresh temp directory, and computes *that* fixed document's
/// [`MetricVector`] too. Then, per genre, reports the combined-score AUC
/// of human vs llm-raw (baseline) and human vs llm-fixed (the tool's
/// effect), plus all three groups' combined-score mean/median.
///
/// # Errors
///
/// Returns an error if `--friction-bin` does not exist as a file; if the
/// manifest, any referenced document, or the envelope pack can't be
/// read/parsed; if the release binary exits non-zero or emits non-UTF-8
/// output fixing any llm-holdout doc; or if `--report` can't be written.
pub fn run(args: &Args) -> anyhow::Result<()> {
    let source = FrictionMetricsSource::new()?;
    run_with_source(args, &source)
}

fn run_with_source(args: &Args, source: &dyn MetricSource) -> anyhow::Result<()> {
    if !args.friction_bin.is_file() {
        bail!(
            "release binary not found at {} — build it first with `cargo build --release -p \
             friction-cli`",
            args.friction_bin.display()
        );
    }

    let manifest_path = args.corpus_dir.join("manifest.jsonl");
    let records = manifest::read_manifest(&manifest_path)?.unwrap_or_default();

    let mut holdout: Vec<&ManifestRecord> = records
        .iter()
        .filter(|r| r.split == Some(Split::Holdout))
        .collect();
    holdout.sort_by(|a, b| a.id.cmp(&b.id));

    let tmp = tempfile::tempdir()
        .context("failed to create a temp dir for fixed llm-holdout documents")?;

    let mut by_genre: BTreeMap<Genre, GenreGroups> = BTreeMap::new();
    for record in &holdout {
        let path = args.corpus_dir.join(relpath(record));
        let document = load_document(&path, &record.id)?;
        let metrics = source.compute(&document);
        let entry = by_genre.entry(record.genre).or_default();

        match record.class {
            Class::Human => entry.human.push(metrics),
            Class::Llm => {
                entry.llm_raw.push(metrics);
                let genre_str = record.genre.to_string();
                let fixed_text = run_friction_fix(&args.friction_bin, &path, &genre_str)
                    .with_context(|| {
                        format!("{}: fixing with the release binary failed", record.id)
                    })?;
                let fixed_path = tmp.path().join(format!("{}.md", record.id));
                std::fs::write(&fixed_path, &fixed_text).with_context(|| {
                    format!("failed to write fixed doc to {}", fixed_path.display())
                })?;
                let fixed_document = load_document(&fixed_path, &record.id)
                    .with_context(|| format!("{}: fixed output failed to parse", record.id))?;
                entry.llm_fixed.push(source.compute(&fixed_document));
            }
        }
    }

    let envelope_pack = load_envelope_pack(&args.envelope)
        .with_context(|| format!("failed to read envelope pack {}", args.envelope.display()))?;

    let report = render_report(&by_genre, &envelope_pack);

    if let Some(parent) = args.report.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&args.report, &report)?;
    println!(
        "separate-holdout: wrote report to {}",
        args.report.display()
    );
    Ok(())
}

#[derive(Debug, Default)]
struct GenreGroups {
    human: Vec<MetricVector>,
    llm_raw: Vec<MetricVector>,
    llm_fixed: Vec<MetricVector>,
}

/// Runs `<bin> fix <path> --genre <genre>` and returns its stdout (the
/// fixed document text, exactly as `friction fix` without `--in-place`
/// emits it) as a `String`.
///
/// # Errors
/// Returns an error if the process can't be spawned, exits non-zero, or
/// its stdout isn't valid UTF-8.
fn run_friction_fix(bin: &Path, path: &Path, genre: &str) -> anyhow::Result<String> {
    let output = Command::new(bin)
        .arg("fix")
        .arg(path)
        .arg("--genre")
        .arg(genre)
        .output()
        .with_context(|| format!("failed to spawn {} fix {}", bin.display(), path.display()))?;
    if !output.status.success() {
        bail!(
            "{} fix {} exited with {}: {}",
            bin.display(),
            path.display(),
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
    }
    String::from_utf8(output.stdout).with_context(|| {
        format!(
            "{} fix {}: stdout was not valid UTF-8",
            bin.display(),
            path.display()
        )
    })
}

// --- combined-score helpers ---

/// The combined score of every document in `values` that has one (see
/// [`combined_score`]), against `bands` — `None` bands (no envelope entry
/// for this genre) yields an empty list, matching
/// `crate::commands::separate`'s own "no basis for a score" convention.
fn scores(values: &[MetricVector], bands: Option<&BTreeMap<String, MetricBand>>) -> Vec<f64> {
    let Some(bands) = bands else {
        return Vec::new();
    };
    values
        .iter()
        .filter_map(|v| combined_score(v, bands))
        .collect()
}

fn mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        None
    } else {
        #[allow(clippy::cast_precision_loss)]
        Some(values.iter().sum::<f64>() / values.len() as f64)
    }
}

fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    Some(if n % 2 == 1 {
        sorted[n / 2]
    } else {
        f64::midpoint(sorted[n / 2 - 1], sorted[n / 2])
    })
}

fn fmt_opt(value: Option<f64>) -> String {
    value.map_or_else(|| "n/a".to_string(), |v| format!("{v:.4}"))
}

fn fmt_auc(value: Option<(f64, crate::commands::separate::Direction)>) -> String {
    value.map_or_else(
        || "n/a".to_string(),
        |(auc, direction)| format!("{auc:.4} ({})", direction.as_str()),
    )
}

// --- report rendering ---

/// One genre's combined-score vectors for all three groups (human,
/// llm-raw, llm-fixed), retained across the AUC-summary loop in
/// [`render_report`] so the distribution section below it doesn't
/// recompute them.
type GenreScores = (Genre, Vec<f64>, Vec<f64>, Vec<f64>);

fn render_report(by_genre: &BTreeMap<Genre, GenreGroups>, envelope_pack: &LoadedPack) -> String {
    let mut out = String::new();
    writeln!(out, "# Holdout separation report").expect("write to String is infallible");
    writeln!(out).expect("write to String is infallible");
    writeln!(
        out,
        "Sealed holdout split (see `corpus-tool holdout-check`). One run, no tuning: this \
         command's output, and the corpus/pack/rule state it was run against, are not to be \
         adjusted based on these numbers. Three groups per genre — human-holdout (untouched), \
         llm-holdout (untouched, the baseline), and fixed-llm-holdout (the same llm-holdout \
         documents after `friction fix --genre <genre>`, run via the release binary as a \
         subprocess, into a fresh temp directory) — scored by the same combined score \
         `corpus-tool separate` uses on the dev split: the mean, over a document's genre's \
         *included* envelope-pack metrics, of a per-metric normalized directional exceedance \
         beyond that metric's train-human envelope band. AUC is the Mann-Whitney U statistic, \
         tie-corrected via midranks, oriented so AUC > 0.5 always means the two groups compared \
         separate (see `direction` for which one scores higher). All figures to 4 decimal places."
    )
    .expect("write to String is infallible");

    writeln!(out).expect("write to String is infallible");
    writeln!(out, "## AUC summary").expect("write to String is infallible");
    writeln!(out).expect("write to String is infallible");
    writeln!(
        out,
        "| genre | human n | llm n | baseline AUC (human vs llm) | after-fix AUC (human vs \
         fixed-llm) |"
    )
    .expect("write to String is infallible");
    writeln!(out, "|---|---|---|---|---|").expect("write to String is infallible");

    // Retained per genre for the distribution section below, so the
    // combined-score vectors are computed exactly once per genre.
    let mut per_genre_scores: Vec<GenreScores> = Vec::new();

    for genre in ALL_GENRES {
        let empty = GenreGroups::default();
        let groups = by_genre.get(&genre).unwrap_or(&empty);
        let bands = envelope_pack.bands.get(&genre.to_string());

        let human_scores = scores(&groups.human, bands);
        let llm_scores = scores(&groups.llm_raw, bands);
        let fixed_scores = scores(&groups.llm_fixed, bands);

        let baseline = mann_whitney_auc(&human_scores, &llm_scores);
        let after_fix = mann_whitney_auc(&human_scores, &fixed_scores);

        writeln!(
            out,
            "| {genre} | {} | {} | {} | {} |",
            groups.human.len(),
            groups.llm_raw.len(),
            fmt_auc(baseline),
            fmt_auc(after_fix),
        )
        .expect("write to String is infallible");

        per_genre_scores.push((genre, human_scores, llm_scores, fixed_scores));
    }

    writeln!(out).expect("write to String is infallible");
    for genre in ALL_GENRES {
        writeln!(
            out,
            "{}",
            inclusion_note(envelope_pack.bands.get(&genre.to_string()))
        )
        .expect("write to String is infallible");
    }

    writeln!(out).expect("write to String is infallible");
    writeln!(out, "## Combined-score distributions").expect("write to String is infallible");
    writeln!(out).expect("write to String is infallible");
    writeln!(out, "| genre | group | n | mean | median |").expect("write to String is infallible");
    writeln!(out, "|---|---|---|---|---|").expect("write to String is infallible");
    for (genre, human_scores, llm_scores, fixed_scores) in &per_genre_scores {
        for (label, group_scores) in [
            ("human-holdout", human_scores),
            ("llm-holdout (raw)", llm_scores),
            ("llm-holdout (fixed)", fixed_scores),
        ] {
            writeln!(
                out,
                "| {genre} | {label} | {} | {} | {} |",
                group_scores.len(),
                fmt_opt(mean(group_scores)),
                fmt_opt(median(group_scores)),
            )
            .expect("write to String is infallible");
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_of_empty_is_none() {
        assert_eq!(mean(&[]), None);
    }

    #[test]
    fn mean_matches_hand_computed_value() {
        assert_eq!(mean(&[1.0, 2.0, 3.0]), Some(2.0));
    }

    #[test]
    fn median_of_empty_is_none() {
        assert_eq!(median(&[]), None);
    }

    /// Odd-length input: the median is the middle element after sorting,
    /// independent of input order.
    #[test]
    fn median_odd_length_is_middle_element() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), Some(2.0));
    }

    /// Even-length input: the median is the mean of the two middle
    /// elements after sorting.
    #[test]
    fn median_even_length_is_mean_of_two_middle_elements() {
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), Some(2.5));
    }

    #[test]
    fn fmt_opt_none_is_na() {
        assert_eq!(fmt_opt(None), "n/a");
    }

    #[test]
    fn fmt_opt_some_is_four_decimal_places() {
        assert_eq!(fmt_opt(Some(0.5)), "0.5000");
    }

    /// `scores` returns an empty list (not a filtered-down partial list)
    /// when there is no envelope entry for the genre at all — mirrors
    /// `combined_scores_for_genre`'s own `None` convention in
    /// `crate::commands::separate`.
    #[test]
    fn scores_with_no_bands_is_empty() {
        let values = vec![MetricVector::default(), MetricVector::default()];
        assert!(scores(&values, None).is_empty());
    }

    /// The rendered report contains every genre's AUC-summary row and the
    /// distribution section's three group labels.
    ///
    /// Loads an empty envelope pack through the real [`load_envelope_pack`]
    /// (rather than hand-building a [`LoadedPack`]: its fields are
    /// crate-private by design, constructible only by parsing an actual
    /// pack file) — every genre then has no bands at all, so every AUC
    /// and every combined-score cell renders `n/a` rather than a
    /// fabricated number.
    #[test]
    fn render_report_contains_all_genres_and_all_three_groups() {
        let dir = tempfile::tempdir().expect("tempdir");
        let pack_path = dir.path().join("envelope.toml");
        std::fs::write(&pack_path, "").expect("write empty pack file");
        let pack = load_envelope_pack(&pack_path).expect("empty TOML parses to an empty pack");

        let by_genre: BTreeMap<Genre, GenreGroups> = BTreeMap::new();
        let report = render_report(&by_genre, &pack);
        assert!(report.contains("## AUC summary"));
        assert!(report.contains("## Combined-score distributions"));
        assert!(report.contains("human-holdout"));
        assert!(report.contains("llm-holdout (raw)"));
        assert!(report.contains("llm-holdout (fixed)"));
        for genre in ALL_GENRES {
            assert!(report.contains(&format!("| {genre} |")));
        }
    }
}
