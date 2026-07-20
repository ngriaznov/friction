//! `corpus-tool envelope` — estimates per-`(genre, metric)` train-derived
//! stats and writes them as a versioned TOML pack.
//!
//! Three things, all from the train split: a human percentile band, an
//! LLM direction, and a combined-score inclusion flag.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::PathBuf;

use clap::Args as ClapArgs;

use crate::commands::separate::{Direction, mann_whitney_auc};
use crate::corpus_layout::relpath;
use crate::hashing::sha256_hex;
use crate::manifest::{self, Class, Genre, ManifestRecord, Split};
use crate::metric_source::{FrictionMetricsSource, MetricSource, load_document};
use friction_core::MetricVector;

/// The fixed set of five genres, for warning about a genre with no
/// train-split human docs at all (there's no `Genre::VARIANTS` in
/// `crate::manifest`, so this names them explicitly).
const ALL_GENRES: [Genre; 5] = [
    Genre::Docs,
    Genre::Blog,
    Genre::Readme,
    Genre::Email,
    Genre::Forum,
];

/// Arguments for `corpus-tool envelope`.
#[derive(Debug, ClapArgs)]
pub struct Args {
    /// Corpus root directory.
    #[arg(long, default_value = "corpus")]
    pub corpus_dir: PathBuf,
    /// Path to write the versioned envelope pack to.
    #[arg(long, default_value = "crates/friction-packs/packs/envelope-v2.toml")]
    pub out: PathBuf,
    /// Lower percentile of the band (nearest-rank method; see module
    /// docs).
    #[arg(long, default_value_t = 10.0)]
    pub lo_percentile: f64,
    /// Upper percentile of the band (nearest-rank method; see module
    /// docs).
    #[arg(long, default_value_t = 90.0)]
    pub hi_percentile: f64,
    /// A `(genre, metric)`'s train-internal AUC (human vs llm, both
    /// train-split, via [`mann_whitney_auc`]) must reach this to be
    /// marked `include = true` — i.e. to count toward that genre's
    /// combined score at all. Below it, the metric is judged
    /// non-discriminative for that genre *on train-split evidence alone*
    /// and excluded, never by hand-picking against dev results. Must be
    /// in `[0.5, 1.0]` (an oriented AUC is never below 0.5).
    #[arg(long, default_value_t = 0.55)]
    pub auc_include_threshold: f64,
}

/// Runs `envelope`.
///
/// For every `train`-split document (both classes), parses it and
/// computes its [`MetricVector`] (via [`FrictionMetricsSource`] — see
/// `crate::metric_source` for why that indirection exists), groups the
/// results by genre and class. For each `(genre, metric)` pair this
/// estimates two independent things, both from the train split only:
///
/// - a `[lo, hi]` human percentile band (nearest-rank method; see
///   [`nearest_rank_percentile`]), from `human`-class train docs only —
///   unchanged from `envelope-v1`;
/// - a `direction` (which class's values run higher) and an `include`
///   flag, from the train-internal Mann-Whitney AUC of `human` vs `llm`
///   train docs (see [`mann_whitney_auc`], the same statistic
///   `corpus-tool separate` reports on the dev split) — `include` is
///   `true` iff that AUC reaches `--auc-include-threshold`. This is the
///   *only* mechanism that ever drops a metric from a genre's combined
///   score: a train-derived rule, applied uniformly, never a per-genre
///   hand override tuned against dev results.
///
/// If a genre has train-human docs but no train-llm docs at all, the
/// train-internal AUC is undefined for every metric in that genre: the
/// pack still gets its percentile bands, but every metric defaults to
/// `include = true` (there's no train-split evidence to justify
/// excluding it) with a placeholder `direction` and no `train_auc`
/// recorded — a warning is printed to stderr.
///
/// Writes the result as a versioned TOML pack (`envelope-v2`) to `--out`.
///
/// Quarantined (CC-BY-SA) docs are included in both estimates — the
/// quarantine restriction is about never redistributing the *document
/// text* itself in a shipped pack, not about excluding its aggregate
/// statistics from one.
///
/// A genre with zero train-split human docs is omitted from the pack
/// entirely (with a warning to stderr) rather than emitting a degenerate
/// band.
///
/// # Errors
///
/// Returns an error if the manifest, or any referenced document, can't be
/// read or parsed, if `--lo-percentile`/`--hi-percentile` are out of
/// `[0, 100]` or not `lo < hi`, if `--auc-include-threshold` is out of
/// `[0.5, 1.0]`, or if `--out` can't be written.
pub fn run(args: &Args) -> anyhow::Result<()> {
    let source = FrictionMetricsSource::new()?;
    run_with_source(args, &source)
}

/// The actual implementation, generic over [`MetricSource`] so tests can
/// substitute a fixture source instead of parsing real prose through the
/// (currently placeholder) default.
fn run_with_source(args: &Args, source: &dyn MetricSource) -> anyhow::Result<()> {
    anyhow::ensure!(
        (0.0..=100.0).contains(&args.lo_percentile) && (0.0..=100.0).contains(&args.hi_percentile),
        "--lo-percentile and --hi-percentile must both be in [0, 100] \
         (got lo={}, hi={})",
        args.lo_percentile,
        args.hi_percentile
    );
    anyhow::ensure!(
        args.lo_percentile < args.hi_percentile,
        "--lo-percentile ({}) must be less than --hi-percentile ({})",
        args.lo_percentile,
        args.hi_percentile
    );
    anyhow::ensure!(
        (0.5..=1.0).contains(&args.auc_include_threshold),
        "--auc-include-threshold must be in [0.5, 1.0] (got {})",
        args.auc_include_threshold
    );

    let manifest_path = args.corpus_dir.join("manifest.jsonl");
    let manifest_bytes = std::fs::read(&manifest_path).unwrap_or_default();
    let records = manifest::read_manifest(&manifest_path)?.unwrap_or_default();

    let mut train: Vec<&ManifestRecord> = records
        .iter()
        .filter(|r| r.split == Some(Split::Train))
        .collect();
    train.sort_by(|a, b| a.id.cmp(&b.id));

    // genre name -> that genre's train-split MetricVectors, by class. A
    // `BTreeMap<String, _>` rather than `BTreeMap<Genre, _>` so the
    // pack's genre sections come out alphabetically sorted, matching its
    // metric sub-sections (see `render_pack`).
    let mut by_genre: BTreeMap<String, GenreVectors> = BTreeMap::new();
    for record in &train {
        let path = args.corpus_dir.join(relpath(record));
        let document = load_document(&path, &record.id)?;
        let metrics = source.compute(&document);
        let entry = by_genre.entry(record.genre.to_string()).or_default();
        match record.class {
            Class::Human => entry.human.push(metrics),
            Class::Llm => entry.llm.push(metrics),
        }
    }

    for genre in ALL_GENRES {
        let key = genre.to_string();
        let human_empty = by_genre.get(&key).is_none_or(|g| g.human.is_empty());
        if human_empty {
            eprintln!(
                "warning: envelope: no train-split human docs for genre \"{genre}\"; omitted \
                 from pack"
            );
        } else if by_genre[&key].llm.is_empty() {
            eprintln!(
                "warning: envelope: no train-split llm docs for genre \"{genre}\"; \
                 direction/inclusion defaulted (include=true, no train_auc) for all its metrics"
            );
        }
    }

    let bands = estimate_bands(
        &by_genre,
        args.lo_percentile,
        args.hi_percentile,
        args.auc_include_threshold,
    );

    let mut human_docs_per_genre: BTreeMap<String, usize> = BTreeMap::new();
    let mut llm_docs_per_genre: BTreeMap<String, usize> = BTreeMap::new();
    let mut train_human_doc_count = 0usize;
    let mut train_llm_doc_count = 0usize;
    for (genre, vectors) in &by_genre {
        if !vectors.human.is_empty() {
            human_docs_per_genre.insert(genre.clone(), vectors.human.len());
        }
        if !vectors.llm.is_empty() {
            llm_docs_per_genre.insert(genre.clone(), vectors.llm.len());
        }
        train_human_doc_count += vectors.human.len();
        train_llm_doc_count += vectors.llm.len();
    }

    let header = PackHeader {
        version: "envelope-v2",
        lo_percentile: args.lo_percentile,
        hi_percentile: args.hi_percentile,
        auc_include_threshold: args.auc_include_threshold,
        corpus_manifest_sha256: sha256_hex(&manifest_bytes),
        train_human_doc_count,
        train_llm_doc_count,
        human_docs_per_genre,
        llm_docs_per_genre,
    };

    let pack = render_pack(&header, &bands);

    if let Some(parent) = args.out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&args.out, &pack)?;
    println!(
        "envelope: wrote {} genre(s), {} band(s) to {}",
        bands.len(),
        bands.values().map(BTreeMap::len).sum::<usize>(),
        args.out.display()
    );
    Ok(())
}

/// A genre's train-split [`MetricVector`]s, by class.
#[derive(Debug, Default)]
struct GenreVectors {
    human: Vec<MetricVector>,
    llm: Vec<MetricVector>,
}

/// Nearest-rank percentile of `sorted_values` (ascending, non-empty).
///
/// For `n` values and percentile `p` (`0.0..=100.0`), the result is the
/// value at 1-indexed rank `ceil(p / 100 * n)`, clamped to `[1, n]` —
/// always one of the actual data points, never an interpolation between
/// two of them, so the same input multiset always yields the same output
/// bit-for-bit on every platform.
///
/// # Panics
/// Panics (via `debug_assert`) if `sorted_values` is empty; every caller
/// here only invokes this on a non-empty per-genre vector.
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

/// One `(genre, metric)` pack entry: the train-human percentile band
/// plus the train-internal direction/inclusion verdict. See
/// [`estimate_bands`] for how each field is derived.
#[derive(Debug, Clone, Copy)]
struct MetricEntry {
    lo: f64,
    hi: f64,
    direction: Direction,
    include: bool,
    /// The train-internal Mann-Whitney AUC (human vs llm, train split)
    /// that `include` was decided from. `None` when that comparison was
    /// undefined (no train-split llm docs for this genre) — in that case
    /// `direction` is a meaningless placeholder, not a real verdict.
    train_auc: Option<f64>,
}

/// genre name -> metric name -> entry.
type Bands = BTreeMap<String, BTreeMap<&'static str, MetricEntry>>;

/// Builds [`Bands`] from `by_genre`: a `[lo, hi]` band from each genre's
/// `human` vectors (nearest-rank percentiles at `lo_p`/`hi_p`, unchanged
/// from `envelope-v1`), and a `direction`/`include`/`train_auc` from a
/// train-internal Mann-Whitney AUC of that genre's `human` vs `llm`
/// vectors, oriented so `include = true` iff the AUC reaches
/// `auc_include_threshold`.
///
/// A genre with no `human` vectors at all is skipped entirely (its
/// warning is the caller's responsibility, in `run_with_source`, since
/// this function has no stderr access and no reason to).
fn estimate_bands(
    by_genre: &BTreeMap<String, GenreVectors>,
    lo_p: f64,
    hi_p: f64,
    auc_include_threshold: f64,
) -> Bands {
    let mut bands: Bands = BTreeMap::new();
    for (genre, vectors) in by_genre {
        if vectors.human.is_empty() {
            continue;
        }
        let mut metrics: BTreeMap<&'static str, MetricEntry> = BTreeMap::new();
        for name in MetricVector::FIELD_NAMES {
            let human_values: Vec<f64> = vectors
                .human
                .iter()
                .map(|v| v.get(name).expect("FIELD_NAMES names a real field"))
                .collect();
            let llm_values: Vec<f64> = vectors
                .llm
                .iter()
                .map(|v| v.get(name).expect("FIELD_NAMES names a real field"))
                .collect();

            let mut sorted_human = human_values.clone();
            sorted_human.sort_by(f64::total_cmp);
            let lo = nearest_rank_percentile(&sorted_human, lo_p);
            let hi = nearest_rank_percentile(&sorted_human, hi_p);

            let (direction, include, train_auc) = match mann_whitney_auc(&human_values, &llm_values)
            {
                Some((auc, direction)) => (direction, auc >= auc_include_threshold, Some(auc)),
                // No train-split llm docs for this genre: the
                // train-internal comparison is undefined, so there is
                // no train evidence to exclude this metric on —
                // default to keeping it. `direction` here is an
                // arbitrary placeholder; `train_auc: None` is the
                // signal callers must check before trusting it.
                None => (Direction::LlmHigher, true, None),
            };

            metrics.insert(
                name,
                MetricEntry {
                    lo,
                    hi,
                    direction,
                    include,
                    train_auc,
                },
            );
        }
        bands.insert(genre.clone(), metrics);
    }
    bands
}

struct PackHeader {
    version: &'static str,
    lo_percentile: f64,
    hi_percentile: f64,
    auc_include_threshold: f64,
    corpus_manifest_sha256: String,
    train_human_doc_count: usize,
    train_llm_doc_count: usize,
    human_docs_per_genre: BTreeMap<String, usize>,
    llm_docs_per_genre: BTreeMap<String, usize>,
}

/// Renders the envelope pack as TOML text.
///
/// Format: a `[pack]` header table (version, generation parameters, the
/// manifest's sha256, and doc-count summaries), then one
/// `[<genre>.<metric>]` table per band with `lo`/`hi`/`direction`/
/// `include` keys plus `train_auc` when it was defined (see
/// [`MetricEntry`]). Genres and metrics both come out in ascending
/// alphabetical order (via the `BTreeMap` keys feeding this function), so
/// the same corpus and arguments always produce byte-identical output.
fn render_pack(header: &PackHeader, bands: &Bands) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# envelope-v2 pack: per-(genre, metric) human percentile bands, train-derived llm \
         direction, and train-AUC combined-score inclusion flag."
    )
    .expect("write to String is infallible");
    writeln!(
        out,
        "# Generated by `corpus-tool envelope` — do not hand-edit."
    )
    .expect("write to String is infallible");
    writeln!(out).expect("write to String is infallible");
    writeln!(out, "[pack]").expect("write to String is infallible");
    writeln!(out, "version = {:?}", header.version).expect("write to String is infallible");
    writeln!(out, "percentile_method = \"nearest-rank\"").expect("write to String is infallible");
    writeln!(
        out,
        "lo_percentile = {}",
        fmt_toml_float(header.lo_percentile)
    )
    .expect("write to String is infallible");
    writeln!(
        out,
        "hi_percentile = {}",
        fmt_toml_float(header.hi_percentile)
    )
    .expect("write to String is infallible");
    writeln!(
        out,
        "auc_include_threshold = {}",
        fmt_toml_float(header.auc_include_threshold)
    )
    .expect("write to String is infallible");
    writeln!(out, "auc_method = \"mann-whitney-tie-corrected\"")
        .expect("write to String is infallible");
    writeln!(
        out,
        "corpus_manifest_sha256 = {:?}",
        header.corpus_manifest_sha256
    )
    .expect("write to String is infallible");
    writeln!(
        out,
        "train_human_doc_count = {}",
        header.train_human_doc_count
    )
    .expect("write to String is infallible");
    writeln!(out, "train_llm_doc_count = {}", header.train_llm_doc_count)
        .expect("write to String is infallible");
    writeln!(out).expect("write to String is infallible");

    writeln!(out, "[pack.human_docs_per_genre]").expect("write to String is infallible");
    for (genre, count) in &header.human_docs_per_genre {
        writeln!(out, "{genre} = {count}").expect("write to String is infallible");
    }
    writeln!(out).expect("write to String is infallible");

    writeln!(out, "[pack.llm_docs_per_genre]").expect("write to String is infallible");
    for (genre, count) in &header.llm_docs_per_genre {
        writeln!(out, "{genre} = {count}").expect("write to String is infallible");
    }

    for (genre, metrics) in bands {
        for (metric, entry) in metrics {
            writeln!(out).expect("write to String is infallible");
            writeln!(out, "[{genre}.{metric}]").expect("write to String is infallible");
            writeln!(out, "lo = {}", fmt_toml_float(entry.lo))
                .expect("write to String is infallible");
            writeln!(out, "hi = {}", fmt_toml_float(entry.hi))
                .expect("write to String is infallible");
            writeln!(out, "direction = {:?}", entry.direction.as_str())
                .expect("write to String is infallible");
            writeln!(out, "include = {}", entry.include).expect("write to String is infallible");
            if let Some(auc) = entry.train_auc {
                writeln!(out, "train_auc = {}", fmt_toml_float(auc))
                    .expect("write to String is infallible");
            }
        }
    }
    out
}

/// Formats `v` as a TOML float literal: TOML requires a decimal point (or
/// exponent) on a float or it parses as an integer, so this appends `.0`
/// to Rust's shortest round-trippable `Display` output when it would
/// otherwise look like a bare integer.
fn fmt_toml_float(v: f64) -> String {
    let s = format!("{v}");
    if s.contains(['.', 'e', 'E']) {
        s
    } else {
        format!("{s}.0")
    }
}

#[cfg(test)]
// Every comparison below is between an exact array element the function
// under test returned and the same literal computed by hand — there is
// no floating-point arithmetic drift to guard against, so exact equality
// is the correct check, not an approximation bug.
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    // --- nearest_rank_percentile: hand-computed fixtures ---

    /// n = 10, values 1..=10: p10 -> rank ceil(0.10*10)=1 -> value 1;
    /// p90 -> rank ceil(0.90*10)=9 -> value 9 (1-indexed rank 9 is
    /// `values[8]`).
    #[test]
    fn nearest_rank_percentile_matches_hand_computed_deciles() {
        let values: Vec<f64> = (1..=10).map(f64::from).collect();
        assert_eq!(nearest_rank_percentile(&values, 10.0), 1.0);
        assert_eq!(nearest_rank_percentile(&values, 90.0), 9.0);
    }

    /// n = 7, values [10, 20, .., 70]: p25 -> rank ceil(0.25*7) =
    /// ceil(1.75) = 2 -> `values[1]` = 20; p75 -> rank ceil(0.75*7) =
    /// ceil(5.25) = 6 -> `values[5]` = 60.
    #[test]
    fn nearest_rank_percentile_matches_hand_computed_non_multiple_of_ten() {
        let values: Vec<f64> = (1..=7).map(|i| f64::from(i) * 10.0).collect();
        assert_eq!(nearest_rank_percentile(&values, 25.0), 20.0);
        assert_eq!(nearest_rank_percentile(&values, 75.0), 60.0);
    }

    /// A single-value genre: every percentile picks that one value (rank
    /// clamps to 1).
    #[test]
    fn nearest_rank_percentile_single_value_clamps_to_that_value() {
        let values = [42.0];
        assert_eq!(nearest_rank_percentile(&values, 0.0), 42.0);
        assert_eq!(nearest_rank_percentile(&values, 10.0), 42.0);
        assert_eq!(nearest_rank_percentile(&values, 100.0), 42.0);
    }

    /// p100 always lands on the last (largest) value.
    #[test]
    fn nearest_rank_percentile_p100_is_max() {
        let values = [1.0, 5.0, 9.0];
        assert_eq!(nearest_rank_percentile(&values, 100.0), 9.0);
    }

    // --- fmt_toml_float ---

    #[test]
    fn fmt_toml_float_appends_decimal_to_whole_numbers() {
        assert_eq!(fmt_toml_float(0.0), "0.0");
        assert_eq!(fmt_toml_float(3.0), "3.0");
    }

    #[test]
    fn fmt_toml_float_leaves_fractional_values_alone() {
        assert_eq!(fmt_toml_float(3.5), "3.5");
        assert_eq!(fmt_toml_float(0.125), "0.125");
    }

    // --- estimate_bands / render_pack: small, hand-checkable example ---

    fn vector_with(name: &str, value: f64) -> MetricVector {
        let mut v = MetricVector::default();
        match name {
            "triad_rate" => v.triad_rate = value,
            "em_dash_density" => v.em_dash_density = value,
            _ => unreachable!("test helper only covers the fields used below"),
        }
        v
    }

    /// A two-genre example exercising both branches of the
    /// direction/inclusion verdict:
    ///
    /// - `docs`: `triad_rate` is `[1,2,3,4,5]` for human (p10 -> 1, p90
    ///   -> 5, hand-computed the same way as `envelope-v1`) and
    ///   `[101,...,105]` for llm — llm always higher, so the raw
    ///   Mann-Whitney AUC is exactly `1.0`, oriented `LlmHigher`, and
    ///   `1.0 >= 0.55` so `include` is `true`. Every *other* field is
    ///   `0.0` for all 5 human and 5 llm docs (identical, tied), so its
    ///   AUC is the oriented tie value `0.5`, which is `< 0.55` ->
    ///   `include` is `false` — this is the exact mechanism that drops a
    ///   non-discriminative metric (e.g. the diagnosis brief's
    ///   `ritual_marker_rate` at train AUC ~0.50) from a genre's combined
    ///   score.
    /// - `blog`: `em_dash_density` is `[10, 20]` for human (p10 -> rank
    ///   ceil(0.1*2)=1 -> 10, p90 -> rank ceil(0.9*2)=2 -> 20) with *no*
    ///   llm vectors at all — the train-internal AUC is undefined, so
    ///   `train_auc` is `None` and `include` defaults to `true`.
    #[test]
    fn estimate_bands_matches_hand_computed_example() {
        let mut by_genre: BTreeMap<String, GenreVectors> = BTreeMap::new();
        by_genre.insert(
            "docs".to_string(),
            GenreVectors {
                human: (1..=5)
                    .map(|i| vector_with("triad_rate", f64::from(i)))
                    .collect(),
                llm: (1..=5)
                    .map(|i| vector_with("triad_rate", f64::from(i) + 100.0))
                    .collect(),
            },
        );
        by_genre.insert(
            "blog".to_string(),
            GenreVectors {
                human: vec![
                    vector_with("em_dash_density", 10.0),
                    vector_with("em_dash_density", 20.0),
                ],
                llm: vec![],
            },
        );

        let bands = estimate_bands(&by_genre, 10.0, 90.0, 0.55);

        let docs_triad = bands["docs"]["triad_rate"];
        assert_eq!((docs_triad.lo, docs_triad.hi), (1.0, 5.0));
        assert_eq!(docs_triad.direction, Direction::LlmHigher);
        assert_eq!(docs_triad.train_auc, Some(1.0));
        assert!(docs_triad.include);

        let docs_em_dash = bands["docs"]["em_dash_density"];
        assert_eq!((docs_em_dash.lo, docs_em_dash.hi), (0.0, 0.0));
        assert_eq!(docs_em_dash.train_auc, Some(0.5));
        assert!(!docs_em_dash.include);

        let blog_em_dash = bands["blog"]["em_dash_density"];
        assert_eq!((blog_em_dash.lo, blog_em_dash.hi), (10.0, 20.0));
        assert_eq!(blog_em_dash.train_auc, None);
        assert!(blog_em_dash.include);
    }

    /// A genre with no `human` vectors at all is omitted from the
    /// output, regardless of `llm` data.
    #[test]
    fn estimate_bands_skips_genre_with_no_human_vectors() {
        let mut by_genre: BTreeMap<String, GenreVectors> = BTreeMap::new();
        by_genre.insert(
            "forum".to_string(),
            GenreVectors {
                human: vec![],
                llm: vec![vector_with("triad_rate", 1.0)],
            },
        );
        let bands = estimate_bands(&by_genre, 10.0, 90.0, 0.55);
        assert!(!bands.contains_key("forum"));
    }

    /// The rendered pack contains a `[pack]` header with the given
    /// version/percentiles/threshold/hash/counts, one
    /// `[<genre>.<metric>]` table per band with `lo`/`hi`/`direction`/
    /// `include`/`train_auc`, and both doc-count sub-tables, with genres
    /// and metrics in alphabetical order.
    #[test]
    fn render_pack_produces_expected_shape() {
        let mut by_genre: BTreeMap<String, GenreVectors> = BTreeMap::new();
        by_genre.insert(
            "docs".to_string(),
            GenreVectors {
                human: vec![vector_with("triad_rate", 1.0)],
                llm: vec![vector_with("triad_rate", 9.0)],
            },
        );
        let bands = estimate_bands(&by_genre, 10.0, 90.0, 0.55);

        let mut human_docs_per_genre = BTreeMap::new();
        human_docs_per_genre.insert("docs".to_string(), 1usize);
        let mut llm_docs_per_genre = BTreeMap::new();
        llm_docs_per_genre.insert("docs".to_string(), 1usize);

        let header = PackHeader {
            version: "envelope-v2",
            lo_percentile: 10.0,
            hi_percentile: 90.0,
            auc_include_threshold: 0.55,
            corpus_manifest_sha256: "deadbeef".to_string(),
            train_human_doc_count: 1,
            train_llm_doc_count: 1,
            human_docs_per_genre,
            llm_docs_per_genre,
        };

        let text = render_pack(&header, &bands);
        assert!(text.contains("[pack]"));
        assert!(text.contains("version = \"envelope-v2\""));
        assert!(text.contains("lo_percentile = 10.0"));
        assert!(text.contains("hi_percentile = 90.0"));
        assert!(text.contains("auc_include_threshold = 0.55"));
        assert!(text.contains("corpus_manifest_sha256 = \"deadbeef\""));
        assert!(text.contains("train_human_doc_count = 1"));
        assert!(text.contains("train_llm_doc_count = 1"));
        assert!(text.contains("[pack.human_docs_per_genre]"));
        assert!(text.contains("[pack.llm_docs_per_genre]"));
        assert!(text.contains("docs = 1"));
        assert!(text.contains("[docs.triad_rate]"));
        assert!(text.contains("lo = 1.0"));
        assert!(text.contains("hi = 1.0"));
        assert!(text.contains("direction = \"llm higher\""));
        assert!(text.contains("include = true"));
        assert!(text.contains("train_auc = 1.0"));
    }

    /// A metric with no train-split llm data for its genre gets no
    /// `train_auc` line at all (rather than a fabricated value).
    #[test]
    fn render_pack_omits_train_auc_when_undefined() {
        let mut by_genre: BTreeMap<String, GenreVectors> = BTreeMap::new();
        by_genre.insert(
            "blog".to_string(),
            GenreVectors {
                human: vec![vector_with("triad_rate", 1.0)],
                llm: vec![],
            },
        );
        let bands = estimate_bands(&by_genre, 10.0, 90.0, 0.55);
        let header = PackHeader {
            version: "envelope-v2",
            lo_percentile: 10.0,
            hi_percentile: 90.0,
            auc_include_threshold: 0.55,
            corpus_manifest_sha256: "deadbeef".to_string(),
            train_human_doc_count: 1,
            train_llm_doc_count: 0,
            human_docs_per_genre: BTreeMap::new(),
            llm_docs_per_genre: BTreeMap::new(),
        };
        let text = render_pack(&header, &bands);
        assert!(text.contains("[blog.triad_rate]"));
        assert!(text.contains("include = true"));
        assert!(!text.contains("train_auc ="));
    }

    /// Rendering the same input twice produces byte-identical output.
    #[test]
    fn render_pack_is_deterministic() {
        let mut by_genre: BTreeMap<String, GenreVectors> = BTreeMap::new();
        by_genre.insert(
            "docs".to_string(),
            GenreVectors {
                human: vec![
                    vector_with("triad_rate", 1.0),
                    vector_with("em_dash_density", 2.0),
                ],
                llm: vec![vector_with("triad_rate", 3.0)],
            },
        );
        by_genre.insert(
            "blog".to_string(),
            GenreVectors {
                human: vec![vector_with("triad_rate", 3.0)],
                llm: vec![],
            },
        );
        let bands = estimate_bands(&by_genre, 10.0, 90.0, 0.55);
        let header = PackHeader {
            version: "envelope-v2",
            lo_percentile: 10.0,
            hi_percentile: 90.0,
            auc_include_threshold: 0.55,
            corpus_manifest_sha256: "abc123".to_string(),
            train_human_doc_count: 3,
            train_llm_doc_count: 1,
            human_docs_per_genre: BTreeMap::new(),
            llm_docs_per_genre: BTreeMap::new(),
        };
        let a = render_pack(&header, &bands);
        let b = render_pack(&header, &bands);
        assert_eq!(a, b);
    }
}
