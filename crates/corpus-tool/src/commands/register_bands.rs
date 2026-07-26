//! `corpus-tool register-bands` — measures the per-document em-dash rate
//! over the train-split human docs-genre population.
//!
//! For hand-transcribing into
//! `crates/friction-packs/packs/register-v1.toml`'s `[features.em_dash]`.
//! Runs through [`friction_edit::register::measure_em_dash_rate`] — the
//! same sentence-context build, tagging, parsing, and counting path
//! `friction-edit`'s register pass itself uses at runtime — so a
//! remeasurement can never silently diverge from what the shipped engine
//! actually counts. Report-only, mirroring `output-bands`/`mine-inventory`'s
//! "human hand-transcribes into the pack" discipline: this tool never
//! writes pack TOML directly.

use std::path::PathBuf;

use clap::Args as ClapArgs;
use friction_nlp::{PerceptronParser, PerceptronTagger, SrxSegmenter};

use crate::corpus_layout::relpath;
use crate::manifest::{self, Class, Genre, Split};

/// Arguments for `corpus-tool register-bands`.
#[derive(Debug, ClapArgs)]
pub struct Args {
    /// Corpus root directory.
    #[arg(long, default_value = "corpus")]
    pub corpus_dir: PathBuf,
}

/// Runs `register-bands`.
///
/// Loads every `class:human, split:train, genre:docs` document (resolving
/// each record's path via [`relpath`] — some live under
/// `corpus/quarantine/docs/` rather than `corpus/human/docs/`), measures
/// its per-1000-prose-word em-dash rate, and prints each document's rate
/// plus the population's 10th/50th/90th percentile (nearest-rank, the
/// same method `register-v1.toml`'s existing two bands were measured
/// with).
///
/// # Errors
/// Returns an error if the manifest or a referenced document can't be
/// read, isn't valid UTF-8, or fails to parse/segment; or if the embedded
/// tagger or dependency-parser model fails to load.
pub fn run(args: &Args) -> anyhow::Result<()> {
    let manifest_path = args.corpus_dir.join("manifest.jsonl");
    let records = manifest::read_manifest(&manifest_path)?.unwrap_or_default();

    let mut docs: Vec<_> = records
        .iter()
        .filter(|r| {
            r.split == Some(Split::Train) && r.class == Class::Human && r.genre == Genre::Docs
        })
        .collect();
    docs.sort_by(|a, b| a.id.cmp(&b.id));

    let tagger = PerceptronTagger::new()?;
    let parser = PerceptronParser::new()?;
    let segmenter = SrxSegmenter::new();

    let mut rates = Vec::with_capacity(docs.len());
    for record in &docs {
        let path = args.corpus_dir.join(relpath(record));
        let text = std::fs::read_to_string(&path).map_err(|e| {
            anyhow::anyhow!("register-bands: failed to read {}: {e}", path.display())
        })?;
        let rate = friction_edit::register::measure_em_dash_rate(
            &text, &tagger, &parser, &segmenter,
        )
        .map_err(|e| anyhow::anyhow!("register-bands: {}: {e}", record.id))?;
        println!("{rate:>10.4}  {}", record.id);
        rates.push(rate);
    }

    if rates.is_empty() {
        println!("register-bands: no class:human, split:train, genre:docs documents found");
        return Ok(());
    }

    let mut sorted = rates;
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("em-dash rates are never NaN"));
    let p10 = nearest_rank_percentile(&sorted, 10.0);
    let median = nearest_rank_percentile(&sorted, 50.0);
    let p90 = nearest_rank_percentile(&sorted, 90.0);
    let max = sorted.last().copied().unwrap_or(0.0);

    println!("---");
    println!("sample_doc_count = {}", sorted.len());
    println!("low (p10)  = {p10:.4}");
    println!("median     = {median:.4}");
    println!("high (p90) = {p90:.4}");
    println!("max        = {max:.4}");

    Ok(())
}

/// Nearest-rank percentile: the value at 1-indexed rank
/// `ceil(percentile / 100 * n)`, clamped to `[1, n]`. Identical method to
/// `corpus-tool envelope`'s own `nearest_rank_percentile` (kept as a
/// separate copy rather than shared, since `envelope`'s is private to
/// that module and this tool's whole per-subcommand convention is one
/// module, no cross-imports between sibling `commands::*` files).
///
/// # Panics
/// Panics (via `debug_assert`) if `sorted_values` is empty; the one
/// caller here only invokes this after checking `rates` is non-empty.
fn nearest_rank_percentile(sorted_values: &[f64], percentile: f64) -> f64 {
    debug_assert!(
        !sorted_values.is_empty(),
        "nearest_rank_percentile: empty input"
    );
    let n = sorted_values.len();
    #[allow(clippy::cast_precision_loss)]
    let n_f64 = n as f64;
    let rank = (percentile / 100.0 * n_f64).ceil().max(1.0);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let idx = ((rank as usize).min(n)) - 1;
    sorted_values[idx]
}

#[cfg(test)]
// Every comparison below is between an exact array element the function
// under test returned and the same literal computed by hand — there is
// no floating-point arithmetic drift to guard against, so exact equality
// is the correct check, not an approximation bug.
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn nearest_rank_percentile_matches_hand_computed_deciles() {
        let values: Vec<f64> = (1..=10).map(f64::from).collect();
        assert_eq!(nearest_rank_percentile(&values, 10.0), 1.0);
        assert_eq!(nearest_rank_percentile(&values, 90.0), 9.0);
    }

    /// The population documented in this feature's spec: 56 zero-rate
    /// documents plus 2 nonzero ones -- p10/median/p90 must all land on
    /// a zero entry.
    #[test]
    fn nearest_rank_percentile_is_zero_when_the_vast_majority_of_values_are_zero() {
        let mut values = vec![0.0; 56];
        values.push(0.8);
        values.push(1.49);
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(nearest_rank_percentile(&values, 10.0), 0.0);
        assert_eq!(nearest_rank_percentile(&values, 50.0), 0.0);
        assert_eq!(nearest_rank_percentile(&values, 90.0), 0.0);
    }
}
