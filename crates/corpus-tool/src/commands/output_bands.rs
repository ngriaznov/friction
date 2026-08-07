//! `corpus-tool output-bands` — measures per-frame natural occurrence rates.
//!
//! Measures, per frame, how often a fixed literal replacement phrase
//! already occurs naturally in the train-split human corpus, and writes a
//! markdown report a curator hand-transcribes into
//! `crates/friction-packs/packs/inventory-en-v1.toml`'s
//! `[[output_frequency_bands]]` family (rule 3, output-frequency hygiene
//! — see `friction_packs::validate::check_frequency_hygiene`).
//!
//! Report-only, mirroring `mine-inventory`'s "human hand-transcribes into
//! the pack" discipline: this tool never writes pack TOML directly.
//! Tokenization goes through `friction_harness::clean::tokenize`, the
//! same primitive `corpus-tool index` already reuses.

use std::fmt::Write as _;
use std::path::PathBuf;

use clap::Args as ClapArgs;
use friction_core::Lang;

use crate::corpus_layout::relpath;
use crate::manifest::{self, Class, ManifestRecord, Split};

/// The 9 literal replacement strings the substitution pairs
/// introduce (`use`/`uses` are two separate entries — see
/// `inventory-en-v1.toml`'s own header comment for the 8 named replacement
/// pairs this covers).
const DEFAULT_FRAMES: [&str; 9] = [
    "covers",
    "to",
    "before",
    "use",
    "uses",
    "important for",
    "matters",
    "we cover",
    "development",
];

/// A documented floor for a frame whose measured rate is exactly zero, so
/// a first future occurrence in a fixed document isn't an automatic
/// hygiene violation.
const ZERO_BASELINE_FLOOR: f64 = 0.01;

/// Arguments for `corpus-tool output-bands`.
#[derive(Debug, ClapArgs)]
pub struct Args {
    /// Corpus root directory.
    #[arg(long, default_value = "corpus")]
    pub corpus_dir: PathBuf,
    /// Path to write the markdown report to.
    #[arg(long, default_value = "corpus/OUTPUT_BANDS.md")]
    pub report: PathBuf,
    /// Which frames to measure. Defaults to all 9 named frames
    /// when empty.
    #[arg(long = "frame")]
    pub frames: Vec<String>,
    /// `max = observed_rate * ceiling_multiplier` (or
    /// [`ZERO_BASELINE_FLOOR`] when `observed_rate` is exactly `0.0`).
    #[arg(long, default_value_t = 3.0)]
    pub ceiling_multiplier: f64,
    /// BCP-47 language to measure: corpus records are filtered to
    /// `record.lang == lang.as_str()` (raw string comparison — see
    /// `manifest::filter_lang`) before anything else runs. Defaults to
    /// `en`; every record in the corpus is tagged `en` today, so this
    /// selects the whole corpus and changes no output.
    #[arg(long, default_value = "en")]
    pub lang: Lang,
}

/// One frame's measured rate.
struct FrameMeasurement {
    frame: String,
    occurrences: u64,
    total_words: u64,
    observed_rate: f64,
    max: f64,
    sample_doc_count: usize,
    /// Auditable method label: `"human_train_rate_x<multiplier>"` for a
    /// frame with a nonzero observed rate, `"zero_baseline_floor"` for one
    /// that never occurred (see [`ZERO_BASELINE_FLOOR`]).
    method: String,
}

/// Runs `output-bands`.
///
/// # Errors
///
/// Returns an error if the manifest or a referenced document can't be
/// read, or if `--report` can't be written.
pub fn run(args: &Args) -> anyhow::Result<()> {
    let manifest_path = args.corpus_dir.join("manifest.jsonl");
    let records = manifest::read_manifest(&manifest_path)?.unwrap_or_default();
    let records = manifest::filter_lang(records, args.lang);

    let mut human_train: Vec<&ManifestRecord> = records
        .iter()
        .filter(|r| r.split == Some(Split::Train) && r.class == Class::Human)
        .collect();
    human_train.sort_by(|a, b| a.id.cmp(&b.id));

    let frames: Vec<String> = if args.frames.is_empty() {
        DEFAULT_FRAMES.iter().map(|s| (*s).to_string()).collect()
    } else {
        args.frames.clone()
    };

    let mut doc_token_lists: Vec<Vec<String>> = Vec::with_capacity(human_train.len());
    let mut total_words: u64 = 0;
    for record in &human_train {
        let path = args.corpus_dir.join(relpath(record));
        let bytes = std::fs::read(&path)
            .map_err(|e| anyhow::anyhow!("output-bands: failed to read {}: {e}", path.display()))?;
        let text = String::from_utf8(bytes).map_err(|e| {
            anyhow::anyhow!("output-bands: {} is not valid UTF-8: {e}", path.display())
        })?;
        let tokens = friction_harness::clean::tokenize(&text);
        total_words += tokens
            .iter()
            .filter(|t| !friction_harness::clean::is_punctuation_token(t))
            .count() as u64;
        doc_token_lists.push(tokens);
    }

    let measurements: Vec<FrameMeasurement> = frames
        .into_iter()
        .map(|frame| {
            measure_frame(
                &frame,
                &doc_token_lists,
                total_words,
                args.ceiling_multiplier,
                human_train.len(),
            )
        })
        .collect();

    let report_text = render_report(&measurements, human_train.len(), total_words);
    if let Some(parent) = args.report.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&args.report, &report_text)?;
    println!(
        "output-bands: measured {} frame(s) over {} train human doc(s) ({} word(s)); wrote {}",
        measurements.len(),
        human_train.len(),
        total_words,
        args.report.display()
    );

    Ok(())
}

/// Lowercases and tokenizes `frame` into the word-token sequence a match
/// must reproduce contiguously.
fn frame_tokens(frame: &str) -> Vec<String> {
    friction_harness::clean::tokenize(frame)
        .into_iter()
        .filter(|t| !friction_harness::clean::is_punctuation_token(t))
        .collect()
}

/// Counts every contiguous, case-insensitive occurrence of `needle`
/// within `haystack` (both already lowercased word-token sequences).
fn count_contiguous_occurrences(haystack: &[String], needle: &[String]) -> u64 {
    if needle.is_empty() || haystack.len() < needle.len() {
        return 0;
    }
    let mut count = 0u64;
    for window in haystack.windows(needle.len()) {
        if window == needle {
            count += 1;
        }
    }
    count
}

fn measure_frame(
    frame: &str,
    doc_token_lists: &[Vec<String>],
    total_words: u64,
    ceiling_multiplier: f64,
    sample_doc_count: usize,
) -> FrameMeasurement {
    let needle = frame_tokens(frame);
    let occurrences: u64 = doc_token_lists
        .iter()
        .map(|tokens| count_contiguous_occurrences(tokens, &needle))
        .sum();

    #[allow(clippy::cast_precision_loss)]
    let observed_rate = if total_words == 0 {
        0.0
    } else {
        (occurrences as f64 / total_words as f64) * 1000.0
    };
    let (max, method) = if observed_rate > 0.0 {
        (
            observed_rate * ceiling_multiplier,
            format!("human_train_rate_x{ceiling_multiplier}"),
        )
    } else {
        (ZERO_BASELINE_FLOOR, "zero_baseline_floor".to_string())
    };

    FrameMeasurement {
        frame: frame.to_string(),
        occurrences,
        total_words,
        observed_rate,
        max,
        sample_doc_count,
        method,
    }
}

fn render_report(measurements: &[FrameMeasurement], doc_count: usize, total_words: u64) -> String {
    let mut out = String::new();
    writeln!(out, "# Output-frequency bands").expect("write to String is infallible");
    writeln!(out).expect("write to String is infallible");
    writeln!(
        out,
        "Generated by `corpus-tool output-bands` over {doc_count} train-split human doc(s), \
         {total_words} word token(s) total. Report-only — hand-transcribe selected entries \
         into `crates/friction-packs/packs/inventory-en-v1.toml`'s `[[output_frequency_bands]]` \
         family."
    )
    .expect("write to String is infallible");
    writeln!(out).expect("write to String is infallible");

    writeln!(
        out,
        "| frame | occurrences | total words | observed_rate (per 1000 words) | max | sample_doc_count |"
    )
    .expect("write to String is infallible");
    writeln!(out, "|---|---|---|---|---|---|").expect("write to String is infallible");
    for m in measurements {
        writeln!(
            out,
            "| {:?} | {} | {} | {:.6} | {:.6} | {} |",
            m.frame, m.occurrences, m.total_words, m.observed_rate, m.max, m.sample_doc_count
        )
        .expect("write to String is infallible");
    }
    writeln!(out).expect("write to String is infallible");

    writeln!(out, "## Ready-to-paste TOML").expect("write to String is infallible");
    writeln!(out).expect("write to String is infallible");
    for m in measurements {
        writeln!(out, "```toml").expect("write to String is infallible");
        writeln!(out, "[[output_frequency_bands]]").expect("write to String is infallible");
        writeln!(out, "frame = {:?}", m.frame).expect("write to String is infallible");
        writeln!(out, "unit = \"per_thousand_words\"").expect("write to String is infallible");
        writeln!(out, "observed_rate = {:.6}", m.observed_rate)
            .expect("write to String is infallible");
        writeln!(out, "max = {:.6}", m.max).expect("write to String is infallible");
        writeln!(out, "sample_doc_count = {}", m.sample_doc_count)
            .expect("write to String is infallible");
        writeln!(out, "method = {:?}", m.method).expect("write to String is infallible");
        writeln!(out, "```").expect("write to String is infallible");
        writeln!(out).expect("write to String is infallible");
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_tokens_splits_multi_word_frames() {
        assert_eq!(frame_tokens("we cover"), vec!["we", "cover"]);
        assert_eq!(frame_tokens("important for"), vec!["important", "for"]);
        assert_eq!(frame_tokens("covers"), vec!["covers"]);
    }

    #[test]
    fn count_contiguous_occurrences_counts_overlapping_free_windows() {
        let haystack: Vec<String> = ["we", "cover", "the", "we", "cover", "it"]
            .into_iter()
            .map(String::from)
            .collect();
        let needle = vec!["we".to_string(), "cover".to_string()];
        assert_eq!(count_contiguous_occurrences(&haystack, &needle), 2);
    }

    #[test]
    fn count_contiguous_occurrences_returns_zero_for_absent_frame() {
        let haystack: Vec<String> = ["the", "quick", "fox"]
            .into_iter()
            .map(String::from)
            .collect();
        let needle = vec!["slow".to_string(), "fox".to_string()];
        assert_eq!(count_contiguous_occurrences(&haystack, &needle), 0);
    }

    #[test]
    fn measure_frame_uses_the_zero_baseline_floor_when_never_observed() {
        let docs = vec![vec![
            "the".to_string(),
            "quick".to_string(),
            "fox".to_string(),
        ]];
        let m = measure_frame("development", &docs, 3, 3.0, 1);
        assert_eq!(m.occurrences, 0);
        assert!(m.observed_rate.abs() < f64::EPSILON);
        assert!((m.max - ZERO_BASELINE_FLOOR).abs() < f64::EPSILON);
    }

    #[test]
    fn measure_frame_scales_max_by_the_ceiling_multiplier() {
        let docs = vec![vec!["development".to_string(); 10]];
        let m = measure_frame("development", &docs, 1000, 3.0, 1);
        assert!(m.observed_rate > 0.0);
        assert!((m.observed_rate.mul_add(-3.0, m.max)).abs() < 1e-9);
    }
}
