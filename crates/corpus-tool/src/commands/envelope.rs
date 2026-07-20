//! `corpus-tool envelope` — estimates per-`(genre, metric)` human
//! percentile bands from the train-split human corpus and writes them as
//! a versioned TOML pack.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::PathBuf;

use clap::Args as ClapArgs;

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
    #[arg(long, default_value = "crates/friction-packs/packs/envelope-v1.toml")]
    pub out: PathBuf,
    /// Lower percentile of the band (nearest-rank method; see module
    /// docs).
    #[arg(long, default_value_t = 10.0)]
    pub lo_percentile: f64,
    /// Upper percentile of the band (nearest-rank method; see module
    /// docs).
    #[arg(long, default_value_t = 90.0)]
    pub hi_percentile: f64,
}

/// Runs `envelope`.
///
/// For every `human`-class, `train`-split document, parses it and
/// computes its [`MetricVector`] (via [`FrictionMetricsSource`] — see
/// `crate::metric_source` for why that indirection exists), groups the
/// results by genre, and for each `(genre, metric)` pair estimates a
/// `[lo, hi]` band with the nearest-rank percentile method (see
/// [`nearest_rank_percentile`]). Writes the result as a versioned TOML
/// pack to `--out`.
///
/// Quarantined (CC-BY-SA) human docs are included in the estimate — the
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
/// `[0, 100]` or not `lo < hi`, or if `--out` can't be written.
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

    let manifest_path = args.corpus_dir.join("manifest.jsonl");
    let manifest_bytes = std::fs::read(&manifest_path).unwrap_or_default();
    let records = manifest::read_manifest(&manifest_path)?.unwrap_or_default();

    let mut train_human: Vec<&ManifestRecord> = records
        .iter()
        .filter(|r| r.class == Class::Human && r.split == Some(Split::Train))
        .collect();
    train_human.sort_by(|a, b| a.id.cmp(&b.id));

    // genre name -> that genre's train-human MetricVectors, in doc-id
    // order. A `BTreeMap<String, _>` rather than `BTreeMap<Genre, _>` so
    // the pack's genre sections come out alphabetically sorted, matching
    // its metric sub-sections (see `render_pack`).
    let mut by_genre: BTreeMap<String, Vec<MetricVector>> = BTreeMap::new();
    for record in &train_human {
        let path = args.corpus_dir.join(relpath(record));
        let document = load_document(&path, &record.id)?;
        let metrics = source.compute(&document);
        by_genre
            .entry(record.genre.to_string())
            .or_default()
            .push(metrics);
    }

    for genre in ALL_GENRES {
        if !by_genre.contains_key(&genre.to_string()) {
            eprintln!(
                "warning: envelope: no train-split human docs for genre \"{genre}\"; omitted from pack"
            );
        }
    }

    let bands = estimate_bands(&by_genre, args.lo_percentile, args.hi_percentile);

    let mut docs_per_genre: BTreeMap<String, usize> = BTreeMap::new();
    for (genre, vectors) in &by_genre {
        docs_per_genre.insert(genre.clone(), vectors.len());
    }

    let header = PackHeader {
        version: "envelope-v1",
        lo_percentile: args.lo_percentile,
        hi_percentile: args.hi_percentile,
        corpus_manifest_sha256: sha256_hex(&manifest_bytes),
        train_human_doc_count: train_human.len(),
        docs_per_genre,
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

/// genre name -> metric name -> `(lo, hi)` band.
type Bands = BTreeMap<String, BTreeMap<&'static str, (f64, f64)>>;

fn estimate_bands(by_genre: &BTreeMap<String, Vec<MetricVector>>, lo_p: f64, hi_p: f64) -> Bands {
    let mut bands: Bands = BTreeMap::new();
    for (genre, vectors) in by_genre {
        let mut metrics: BTreeMap<&'static str, (f64, f64)> = BTreeMap::new();
        for name in MetricVector::FIELD_NAMES {
            let mut values: Vec<f64> = vectors
                .iter()
                .map(|v| v.get(name).expect("FIELD_NAMES names a real field"))
                .collect();
            values.sort_by(f64::total_cmp);
            let lo = nearest_rank_percentile(&values, lo_p);
            let hi = nearest_rank_percentile(&values, hi_p);
            metrics.insert(name, (lo, hi));
        }
        bands.insert(genre.clone(), metrics);
    }
    bands
}

struct PackHeader {
    version: &'static str,
    lo_percentile: f64,
    hi_percentile: f64,
    corpus_manifest_sha256: String,
    train_human_doc_count: usize,
    docs_per_genre: BTreeMap<String, usize>,
}

/// Renders the envelope pack as TOML text.
///
/// Format: a `[pack]` header table (version, generation parameters, the
/// manifest's sha256, and doc-count summary), then one `[<genre>.<metric>]`
/// table per band with `lo`/`hi` keys. Genres and metrics both come out in
/// ascending alphabetical order (via the `BTreeMap` keys feeding this
/// function), so the same corpus and arguments always produce
/// byte-identical output.
fn render_pack(header: &PackHeader, bands: &Bands) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# envelope-v1 pack: per-(genre, metric) human percentile bands."
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
    writeln!(out).expect("write to String is infallible");

    writeln!(out, "[pack.docs_per_genre]").expect("write to String is infallible");
    for (genre, count) in &header.docs_per_genre {
        writeln!(out, "{genre} = {count}").expect("write to String is infallible");
    }

    for (genre, metrics) in bands {
        for (metric, &(lo, hi)) in metrics {
            writeln!(out).expect("write to String is infallible");
            writeln!(out, "[{genre}.{metric}]").expect("write to String is infallible");
            writeln!(out, "lo = {}", fmt_toml_float(lo)).expect("write to String is infallible");
            writeln!(out, "hi = {}", fmt_toml_float(hi)).expect("write to String is infallible");
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

    /// A two-genre, one-metric-varying example where the percentile
    /// result can be checked by hand: `docs` gets `triad_rate` values
    /// [1,2,3,4,5] (p10 -> value 1, p90 -> value 5); `blog` gets
    /// `em_dash_density` values [10,20] (p10 -> rank ceil(0.1*2)=1 ->
    /// value 10, p90 -> rank ceil(0.9*2)=2 -> value 20). Every other
    /// metric is 0.0 for every doc, so its band is `[0.0, 0.0]`.
    #[test]
    fn estimate_bands_matches_hand_computed_example() {
        let mut by_genre: BTreeMap<String, Vec<MetricVector>> = BTreeMap::new();
        by_genre.insert(
            "docs".to_string(),
            (1..=5)
                .map(|i| vector_with("triad_rate", f64::from(i)))
                .collect(),
        );
        by_genre.insert(
            "blog".to_string(),
            vec![
                vector_with("em_dash_density", 10.0),
                vector_with("em_dash_density", 20.0),
            ],
        );

        let bands = estimate_bands(&by_genre, 10.0, 90.0);

        assert_eq!(bands["docs"]["triad_rate"], (1.0, 5.0));
        assert_eq!(bands["docs"]["em_dash_density"], (0.0, 0.0));
        assert_eq!(bands["blog"]["em_dash_density"], (10.0, 20.0));
    }

    /// The rendered pack contains a `[pack]` header with the given
    /// version/percentiles/hash/count, and one `[<genre>.<metric>]`
    /// table per band, with genres and metrics in alphabetical order.
    #[test]
    fn render_pack_produces_expected_shape() {
        let mut by_genre: BTreeMap<String, Vec<MetricVector>> = BTreeMap::new();
        by_genre.insert("docs".to_string(), vec![vector_with("triad_rate", 1.0)]);
        let bands = estimate_bands(&by_genre, 10.0, 90.0);
        let mut docs_per_genre = BTreeMap::new();
        docs_per_genre.insert("docs".to_string(), 1usize);

        let header = PackHeader {
            version: "envelope-v1",
            lo_percentile: 10.0,
            hi_percentile: 90.0,
            corpus_manifest_sha256: "deadbeef".to_string(),
            train_human_doc_count: 1,
            docs_per_genre,
        };

        let text = render_pack(&header, &bands);
        assert!(text.contains("[pack]"));
        assert!(text.contains("version = \"envelope-v1\""));
        assert!(text.contains("lo_percentile = 10.0"));
        assert!(text.contains("hi_percentile = 90.0"));
        assert!(text.contains("corpus_manifest_sha256 = \"deadbeef\""));
        assert!(text.contains("train_human_doc_count = 1"));
        assert!(text.contains("[pack.docs_per_genre]"));
        assert!(text.contains("docs = 1"));
        assert!(text.contains("[docs.triad_rate]"));
        assert!(text.contains("lo = 1.0"));
        assert!(text.contains("hi = 1.0"));
    }

    /// Rendering the same input twice produces byte-identical output.
    #[test]
    fn render_pack_is_deterministic() {
        let mut by_genre: BTreeMap<String, Vec<MetricVector>> = BTreeMap::new();
        by_genre.insert(
            "docs".to_string(),
            vec![
                vector_with("triad_rate", 1.0),
                vector_with("em_dash_density", 2.0),
            ],
        );
        by_genre.insert("blog".to_string(), vec![vector_with("triad_rate", 3.0)]);
        let bands = estimate_bands(&by_genre, 10.0, 90.0);
        let header = PackHeader {
            version: "envelope-v1",
            lo_percentile: 10.0,
            hi_percentile: 90.0,
            corpus_manifest_sha256: "abc123".to_string(),
            train_human_doc_count: 3,
            docs_per_genre: BTreeMap::new(),
        };
        let a = render_pack(&header, &bands);
        let b = render_pack(&header, &bands);
        assert_eq!(a, b);
    }
}
