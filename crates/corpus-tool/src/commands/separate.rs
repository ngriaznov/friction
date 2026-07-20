//! `corpus-tool separate`.
//!
//! On the dev split, measures how well the metric vector separates `llm`
//! docs from `human` docs, per genre and per metric, plus a combined
//! per-document score against the human envelope.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::PathBuf;

use anyhow::Context as _;
use clap::Args as ClapArgs;
use serde::Deserialize;

use crate::corpus_layout::relpath;
use crate::manifest::{self, Class, Genre, ManifestRecord, Split};
use crate::metric_source::{FrictionMetricsSource, MetricSource, load_document};
use friction_core::{Envelope, MetricVector};

/// The fixed set of five genres, in the report's section order (matches
/// their declaration order in `crate::manifest::Genre`, not alphabetical
/// — there's no `Genre::VARIANTS` to iterate instead).
const ALL_GENRES: [Genre; 5] = [
    Genre::Docs,
    Genre::Blog,
    Genre::Readme,
    Genre::Email,
    Genre::Forum,
];

/// Arguments for `corpus-tool separate`.
#[derive(Debug, ClapArgs)]
pub struct Args {
    /// Corpus root directory.
    #[arg(long, default_value = "corpus")]
    pub corpus_dir: PathBuf,
    /// Path to the envelope pack (`corpus-tool envelope`'s output) used
    /// for the combined score.
    #[arg(long, default_value = "crates/friction-packs/packs/envelope-v2.toml")]
    pub envelope: PathBuf,
    /// Path to write the markdown separation report to.
    #[arg(long)]
    pub report: PathBuf,
}

/// Runs `separate`.
///
/// For every dev-split document, computes its [`MetricVector`] (via
/// [`FrictionMetricsSource`] — see `crate::metric_source`), then for
/// each `(genre, metric)` pair computes the AUC of `human` vs `llm` via
/// the Mann-Whitney U statistic ([`mann_whitney_auc`]), oriented so
/// `AUC > 0.5` always means "this metric separates llm from human",
/// regardless of which class actually scores higher — [`Direction`]
/// records which way it points. Also computes, per document, a combined
/// score (the mean, over that document's genre's *included* metrics —
/// see [`MetricBand`] — of a per-metric normalized directional
/// exceedance beyond that document's genre's train-human envelope band,
/// loaded from `--envelope`; see [`combined_score`]) and that score's
/// own AUC. Writes a markdown report to `--report`.
///
/// "Included" is entirely a property of the envelope pack, not this
/// command: `corpus-tool envelope` decides, per `(genre, metric)`, from a
/// train-internal AUC comparison, whether that metric counts toward its
/// genre's combined score at all (see the pack's `include` field) — a
/// metric the diagnosis judged non-discriminative for a genre (e.g.
/// `ritual_marker_rate` before its detector fix) is dropped from that
/// genre's combined score this way, never by a hand-picked override here.
///
/// A genre with no dev-split docs of one class (or missing from the
/// envelope pack, for the combined score) is reported with `n/a` rather
/// than a fabricated AUC.
///
/// The report ends with a "Combined-score gate" section: how many of the
/// five genres reach a combined-score AUC of 0.85 or higher, against a
/// target of at least three, with an explicit `MET`/`NOT MET` verdict —
/// so the report always states plainly whether the metrics layer is
/// separating llm from human well enough yet, instead of leaving a
/// reader to eyeball five AUC numbers and guess.
///
/// Quarantined (CC-BY-SA) human docs are not excluded: the dev-split
/// filter above is purely `record.split == Some(Split::Dev)`, with no
/// check of quarantine status. As in `corpus-tool envelope`, the
/// quarantine restriction is about never redistributing the *document
/// text* itself in a shipped pack, not about excluding it from
/// measurement — so a quarantined doc's dev-split metrics flow into the
/// human-vs-llm AUC and combined-score numbers the same as any other
/// dev-split doc.
///
/// # Errors
///
/// Returns an error if the manifest, any referenced document, or the
/// envelope pack can't be read/parsed, or if `--report` can't be written.
pub fn run(args: &Args) -> anyhow::Result<()> {
    let source = FrictionMetricsSource::new()?;
    run_with_source(args, &source)
}

fn run_with_source(args: &Args, source: &dyn MetricSource) -> anyhow::Result<()> {
    let manifest_path = args.corpus_dir.join("manifest.jsonl");
    let records = manifest::read_manifest(&manifest_path)?.unwrap_or_default();

    let mut dev: Vec<&ManifestRecord> = records
        .iter()
        .filter(|r| r.split == Some(Split::Dev))
        .collect();
    dev.sort_by(|a, b| a.id.cmp(&b.id));

    // genre -> class -> that class's dev-split MetricVectors, in doc-id
    // order (fixed by the `dev` sort above).
    let mut by_genre: BTreeMap<Genre, GenreVectors> = BTreeMap::new();
    for record in &dev {
        let path = args.corpus_dir.join(relpath(record));
        let document = load_document(&path, &record.id)?;
        let metrics = source.compute(&document);
        let entry = by_genre.entry(record.genre).or_default();
        match record.class {
            Class::Human => entry.human.push(metrics),
            Class::Llm => entry.llm.push(metrics),
        }
    }

    let envelope_pack = load_envelope_pack(&args.envelope)
        .with_context(|| format!("failed to read envelope pack {}", args.envelope.display()))?;

    let report = render_report(&by_genre, &envelope_pack);

    if let Some(parent) = args.report.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&args.report, &report)?;
    println!("separate: wrote report to {}", args.report.display());
    Ok(())
}

#[derive(Debug, Default)]
struct GenreVectors {
    human: Vec<MetricVector>,
    llm: Vec<MetricVector>,
}

// --- Mann-Whitney U / AUC ---

/// Which class the *raw* (unoriented) Mann-Whitney statistic favored.
///
/// The class whose values tend to be larger. [`mann_whitney_auc`] always
/// returns an AUC `>= 0.5`; `Direction` is what tells the two possible
/// underlying situations apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// `llm` values tend to be larger than `human` values.
    LlmHigher,
    /// `llm` values tend to be smaller than `human` values.
    LlmLower,
}

impl Direction {
    /// `pub(crate)`, not private: `corpus-tool envelope` writes this
    /// exact string into the envelope-v2 pack's `direction` field (see
    /// `crate::commands::envelope::render_pack`), and [`Direction::parse`]
    /// below is its inverse.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::LlmHigher => "llm higher",
            Self::LlmLower => "llm lower",
        }
    }

    /// Parses the exact string [`Direction::as_str`] produces. Returns
    /// `None` for anything else (an unrecognized or hand-edited envelope
    /// pack).
    fn parse(s: &str) -> Option<Self> {
        match s {
            "llm higher" => Some(Self::LlmHigher),
            "llm lower" => Some(Self::LlmLower),
            _ => None,
        }
    }
}

/// Computes the AUC of `human` vs `llm` via the Mann-Whitney U statistic,
/// with tie correction, oriented so the result is always `>= 0.5`.
///
/// # Method
///
/// Pool `human` (size `n1`) and `llm` (size `n2`) into one list of `n1 +
/// n2` values and sort it ascending. Assign 1-indexed ranks; every run of
/// equal values is given the *average* ("midrank") of the ranks it spans,
/// the standard tie correction for this test. Let `r2` be the sum of
/// midranks landing on `llm` values. Then
///
/// ```text
/// U2  = r2 - n2 * (n2 + 1) / 2
/// auc = U2 / (n1 * n2)
/// ```
///
/// `auc` is exactly the probability that a randomly chosen `llm` value
/// exceeds a randomly chosen `human` value, counting a tie as half a win
/// — equivalently, the mean, over every `(human, llm)` pair, of `1` if
/// `llm > human`, `0.5` if equal, `0` otherwise. This function returns
/// `(oriented_auc, direction)` where `oriented_auc = max(auc, 1 - auc)`
/// and `direction` records which side the raw `auc` favored — so
/// `oriented_auc > 0.5` always means "this metric separates the two
/// classes", regardless of which one happens to score higher.
///
/// Returns `None` if either slice is empty (undefined).
pub fn mann_whitney_auc(human: &[f64], llm: &[f64]) -> Option<(f64, Direction)> {
    let n1 = human.len();
    let n2 = llm.len();
    if n1 == 0 || n2 == 0 {
        return None;
    }

    let mut combined: Vec<(f64, bool)> = Vec::with_capacity(n1 + n2);
    combined.extend(human.iter().map(|&v| (v, false)));
    combined.extend(llm.iter().map(|&v| (v, true)));
    combined.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut rank_sum_llm = 0.0f64;
    let mut i = 0usize;
    while i < combined.len() {
        let mut j = i + 1;
        // Exact equality is intentional here: this is a tie-run scan over
        // already-sorted values (not an approximate-equality check), and
        // two `MetricVector` values computed identically must compare
        // exactly equal for tie handling to be well-defined at all.
        #[allow(clippy::float_cmp)]
        while j < combined.len() && combined[j].0 == combined[i].0 {
            j += 1;
        }
        // Ranks (i+1)..=j (1-indexed); midrank is their average.
        #[allow(clippy::cast_precision_loss)]
        let midrank = ((i + 1) + j) as f64 / 2.0;
        for &(_, is_llm) in &combined[i..j] {
            if is_llm {
                rank_sum_llm += midrank;
            }
        }
        i = j;
    }

    #[allow(clippy::cast_precision_loss)]
    let (n1_f64, n2_f64) = (n1 as f64, n2 as f64);
    let u2 = rank_sum_llm - n2_f64 * (n2_f64 + 1.0) / 2.0;
    let auc_llm_over_human = u2 / (n1_f64 * n2_f64);

    Some(if auc_llm_over_human >= 0.5 {
        (auc_llm_over_human, Direction::LlmHigher)
    } else {
        (1.0 - auc_llm_over_human, Direction::LlmLower)
    })
}

// --- envelope pack loading ---

/// One `(genre, metric)` entry from an envelope-v2 pack: the train-human
/// percentile band, which side of it is llm-like (from a train-internal
/// AUC comparison — see `crate::commands::envelope`), and whether this
/// metric counts toward its genre's combined score at all.
#[derive(Debug, Clone, Copy)]
struct MetricBand {
    envelope: Envelope,
    direction: Direction,
    include: bool,
    /// The train-internal AUC `include` was decided from; `None` when
    /// that comparison was undefined for this `(genre, metric)` (no
    /// train-split llm docs for the genre) — `include` then defaults to
    /// `true` and `direction` is a meaningless placeholder.
    train_auc: Option<f64>,
}

/// genre name -> metric name -> band.
type EnvelopePack = BTreeMap<String, BTreeMap<String, MetricBand>>;

#[derive(Debug, Deserialize)]
struct RawEnvelopeEntry {
    lo: f64,
    hi: f64,
    direction: String,
    include: bool,
    #[serde(default)]
    train_auc: Option<f64>,
}

/// The `[pack]` header fields `separate` itself reads; every other header
/// field (version, percentiles, doc counts, ...) is generation metadata
/// `envelope` records for its own reproducibility and isn't needed here,
/// so it's left for serde to silently ignore rather than declared.
#[derive(Debug, Default, Deserialize)]
struct RawPackHeader {
    #[serde(default)]
    auc_include_threshold: Option<f64>,
}

/// The TOML shape `corpus-tool envelope` writes: a `[pack]` header table
/// plus one top-level table per genre, each mapping metric name to a
/// `{lo, hi, direction, include, train_auc?}` sub-table. `#[serde(flatten)]`
/// captures every top-level key except `pack` into `genres`.
#[derive(Debug, Deserialize)]
struct RawPack {
    #[serde(default)]
    pack: RawPackHeader,
    #[serde(flatten)]
    genres: BTreeMap<String, BTreeMap<String, RawEnvelopeEntry>>,
}

/// An envelope pack plus the one header field the report displays for
/// transparency: the train-AUC threshold `envelope` used to decide each
/// metric's `include` flag (the flags themselves are already baked into
/// `bands`; this is purely for the report's inclusion note to name the
/// rule instead of just its effect).
struct LoadedPack {
    bands: EnvelopePack,
    auc_include_threshold: Option<f64>,
}

fn load_envelope_pack(path: &std::path::Path) -> anyhow::Result<LoadedPack> {
    let text = std::fs::read_to_string(path)?;
    parse_envelope_pack(&text)
}

fn parse_envelope_pack(text: &str) -> anyhow::Result<LoadedPack> {
    let raw: RawPack = toml::from_str(text)?;
    let bands = raw
        .genres
        .into_iter()
        .map(|(genre, metrics)| -> anyhow::Result<_> {
            let metrics = metrics
                .into_iter()
                .map(|(name, entry)| -> anyhow::Result<_> {
                    let envelope = Envelope::new(entry.lo, entry.hi);
                    envelope.validate().with_context(|| {
                        format!(
                            "envelope pack: invalid band for genre {genre:?}, metric {name:?}: \
                             lo={}, hi={}",
                            entry.lo, entry.hi
                        )
                    })?;
                    let direction = Direction::parse(&entry.direction).with_context(|| {
                        format!(
                            "envelope pack: unrecognized direction {:?} for genre {genre:?}, \
                             metric {name:?}",
                            entry.direction
                        )
                    })?;
                    Ok((
                        name,
                        MetricBand {
                            envelope,
                            direction,
                            include: entry.include,
                            train_auc: entry.train_auc,
                        },
                    ))
                })
                .collect::<anyhow::Result<_>>()?;
            Ok((genre, metrics))
        })
        .collect::<anyhow::Result<EnvelopePack>>()?;
    Ok(LoadedPack {
        bands,
        auc_include_threshold: raw.pack.auc_include_threshold,
    })
}

// --- combined score ---

/// `0.0` if `value` falls inside `envelope`'s `[lo, hi]` band (inclusive
/// on both ends); otherwise the distance from `value` to the *nearer*
/// band edge, divided by the band width, capped at `1.0` — a normalized
/// directional exceedance, replacing `envelope-v1`'s binary in/out
/// verdict with how far out. A zero-width band (`lo == hi`) treats any
/// miss as full (`1.0`) exceedance rather than dividing by zero.
fn exceedance(value: f64, envelope: Envelope) -> f64 {
    let width = envelope.hi - envelope.lo;
    if value < envelope.lo {
        if width <= 0.0 {
            1.0
        } else {
            ((envelope.lo - value) / width).min(1.0)
        }
    } else if value > envelope.hi {
        if width <= 0.0 {
            1.0
        } else {
            ((value - envelope.hi) / width).min(1.0)
        }
    } else {
        0.0
    }
}

/// The mean, over `metrics`'s fields that `bands` marks `include: true`
/// (see [`MetricBand`] — a train-derived rule, decided entirely by
/// `corpus-tool envelope`, never hand-picked here), of that field's
/// [`exceedance`] beyond its genre's train-human envelope band.
///
/// Generic over [`MetricVector::FIELD_NAMES`]'s current length — adding a
/// metric to `MetricVector` (and its envelope pack entries) changes what
/// this averages over without any code change here.
///
/// `None` if `bands` has no entry at all for one of `metrics`'s fields
/// (an incomplete/absent per-genre envelope — distinct from a field
/// present but excluded), or if every field was excluded (no basis for a
/// score).
fn combined_score(metrics: &MetricVector, bands: &BTreeMap<String, MetricBand>) -> Option<f64> {
    let mut included_total = 0.0;
    let mut included_count = 0usize;
    for (name, value) in metrics.named_values() {
        let band = bands.get(name)?;
        if !band.include {
            continue;
        }
        included_count += 1;
        included_total += exceedance(value, band.envelope);
    }
    if included_count == 0 {
        return None;
    }
    #[allow(clippy::cast_precision_loss)]
    Some(included_total / included_count as f64)
}

// --- report rendering ---

/// The combined-score AUC a genre must clear, and how many of the five
/// genres need to clear it, for the metric layer to be judged separated
/// enough to be worth building deterministic fix-rules on top of rather
/// than iterating on the metrics themselves first. This is a per-genre
/// approximation of that judgment call — this tool computes AUC per
/// genre, not per metric family — so treat the gate section's verdict as
/// a legible summary of the same numbers already in the per-genre
/// tables above it, not an independent, more authoritative signal.
const GATE_AUC_THRESHOLD: f64 = 0.85;
const GATE_MIN_GENRES: usize = 3;

fn render_report(by_genre: &BTreeMap<Genre, GenreVectors>, envelope_pack: &LoadedPack) -> String {
    let mut out = String::new();
    writeln!(out, "# Separation report").expect("write to String is infallible");
    writeln!(out).expect("write to String is infallible");
    let threshold_note = envelope_pack.auc_include_threshold.map_or_else(
        || "n/a (envelope pack header has no `auc_include_threshold`)".to_string(),
        |t| format!("{t:.4}"),
    );
    writeln!(
        out,
        "Dev split, human vs llm. AUC is the Mann-Whitney U statistic, tie-corrected via \
         midranks, oriented so AUC > 0.5 always means the metric separates the two classes \
         (see `direction` for which one scores higher). Combined score: the mean, over a \
         document's genre's *included* metrics (envelope pack `include` flag, a train-internal \
         AUC >= {threshold_note} rule decided entirely by `corpus-tool envelope` — see each \
         genre's \"Combined-score metrics\" line below for which metrics that excluded and \
         why), of a per-metric normalized directional exceedance beyond that metric's \
         train-human envelope band (0.0 inside the band, else the distance to the nearer edge \
         over the band width, capped at 1.0); its own AUC uses the same Mann-Whitney method. \
         All figures to 4 decimal places."
    )
    .expect("write to String is infallible");

    let mut gate_aucs: Vec<(Genre, Option<f64>)> = Vec::new();

    for genre in ALL_GENRES {
        let empty = GenreVectors::default();
        let vectors = by_genre.get(&genre).unwrap_or(&empty);
        writeln!(out).expect("write to String is infallible");
        writeln!(out, "## {genre}").expect("write to String is infallible");
        writeln!(out).expect("write to String is infallible");
        writeln!(out, "| metric | human n | llm n | AUC | direction |")
            .expect("write to String is infallible");
        writeln!(out, "|---|---|---|---|---|").expect("write to String is infallible");

        for name in MetricVector::FIELD_NAMES {
            let human: Vec<f64> = vectors.human.iter().filter_map(|v| v.get(name)).collect();
            let llm: Vec<f64> = vectors.llm.iter().filter_map(|v| v.get(name)).collect();
            match mann_whitney_auc(&human, &llm) {
                Some((auc, direction)) => writeln!(
                    out,
                    "| {name} | {} | {} | {auc:.4} | {} |",
                    human.len(),
                    llm.len(),
                    direction.as_str()
                )
                .expect("write to String is infallible"),
                None => writeln!(
                    out,
                    "| {name} | {} | {} | n/a | n/a |",
                    human.len(),
                    llm.len()
                )
                .expect("write to String is infallible"),
            }
        }

        writeln!(out).expect("write to String is infallible");
        let bands = envelope_pack.bands.get(&genre.to_string());
        let combined = combined_scores_for_genre(vectors, bands);
        let summary = combined_score_summary(genre, vectors, combined.as_ref());
        writeln!(out, "{summary}").expect("write to String is infallible");
        writeln!(out, "{}", inclusion_note(bands)).expect("write to String is infallible");

        let auc = combined
            .as_ref()
            .and_then(|(human, llm)| mann_whitney_auc(human, llm))
            .map(|(auc, _)| auc);
        gate_aucs.push((genre, auc));
    }

    writeln!(out).expect("write to String is infallible");
    out.push_str(&gate_section(
        &gate_aucs,
        GATE_AUC_THRESHOLD,
        GATE_MIN_GENRES,
    ));

    out
}

/// The per-document combined scores (human, then llm), given `bands` (a
/// genre's envelope-pack entry) — `None` if there is no entry at all for
/// this genre.
fn combined_scores_for_genre(
    vectors: &GenreVectors,
    bands: Option<&BTreeMap<String, MetricBand>>,
) -> Option<(Vec<f64>, Vec<f64>)> {
    let bands = bands?;
    let human_scores = vectors
        .human
        .iter()
        .filter_map(|v| combined_score(v, bands))
        .collect();
    let llm_scores = vectors
        .llm
        .iter()
        .filter_map(|v| combined_score(v, bands))
        .collect();
    Some((human_scores, llm_scores))
}

/// A one-line note, per genre, on which of that genre's envelope-pack
/// metrics were dropped from the combined score by the train-derived
/// inclusion rule (`include: false`), and why (its train-split AUC, when
/// the pack recorded one) — so the report always states the rule's
/// concrete effect, not just its existence.
fn inclusion_note(bands: Option<&BTreeMap<String, MetricBand>>) -> String {
    let Some(bands) = bands else {
        return "Combined-score metrics: n/a (no envelope for this genre).".to_string();
    };
    let total = bands.len();
    let mut excluded: Vec<(&String, &MetricBand)> =
        bands.iter().filter(|(_, band)| !band.include).collect();
    excluded.sort_by(|a, b| a.0.cmp(b.0));
    let included = total - excluded.len();
    if excluded.is_empty() {
        return format!("Combined-score metrics: {included} of {total} included (none excluded).");
    }
    let details: Vec<String> = excluded
        .iter()
        .map(|(name, band)| {
            band.train_auc.map_or_else(
                || format!("{name} (no train-split llm docs for this genre)"),
                |auc| format!("{name} (train AUC {auc:.4}, {})", band.direction.as_str()),
            )
        })
        .collect();
    format!(
        "Combined-score metrics: {included} of {total} included; excluded: {}.",
        details.join(", ")
    )
}

fn combined_score_summary(
    genre: Genre,
    vectors: &GenreVectors,
    combined: Option<&(Vec<f64>, Vec<f64>)>,
) -> String {
    let human_n = vectors.human.len();
    let llm_n = vectors.llm.len();

    let Some((human_scores, llm_scores)) = combined else {
        return format!(
            "Summary: {genre} — human n={human_n}, llm n={llm_n}, combined-score AUC = n/a \
             (no envelope for this genre)"
        );
    };

    match mann_whitney_auc(human_scores, llm_scores) {
        Some((auc, direction)) => format!(
            "Summary: {genre} — human n={human_n}, llm n={llm_n}, combined-score AUC = \
             {auc:.4} ({})",
            direction.as_str()
        ),
        None => {
            format!("Summary: {genre} — human n={human_n}, llm n={llm_n}, combined-score AUC = n/a")
        }
    }
}

/// Renders the "does the metrics layer separate llm from human well
/// enough yet" verdict: how many of `entries` (one `(genre,
/// Option<combined-score AUC>)` pair per genre, `None` standing in for
/// `n/a`) reach `threshold`, against the `min_genres` needed to call it
/// met, plus a per-genre breakdown so the verdict is checkable at a
/// glance against the tables above it rather than taken on faith.
fn gate_section(entries: &[(Genre, Option<f64>)], threshold: f64, min_genres: usize) -> String {
    let met_count = entries
        .iter()
        .filter(|(_, auc)| auc.is_some_and(|a| a >= threshold))
        .count();
    let status = if met_count >= min_genres {
        "MET"
    } else {
        "NOT MET"
    };

    let mut out = String::new();
    writeln!(out, "## Combined-score gate").expect("write to String is infallible");
    writeln!(out).expect("write to String is infallible");
    writeln!(
        out,
        "Genres whose combined-score AUC reaches {threshold:.4}: {met_count} of {} (target: \
         at least {min_genres}). Status: {status}.",
        entries.len()
    )
    .expect("write to String is infallible");
    writeln!(out).expect("write to String is infallible");
    writeln!(
        out,
        "| genre | combined-score AUC | reaches {threshold:.4} |"
    )
    .expect("write to String is infallible");
    writeln!(out, "|---|---|---|").expect("write to String is infallible");
    for (genre, auc) in entries {
        match auc {
            Some(a) => writeln!(
                out,
                "| {genre} | {a:.4} | {} |",
                if *a >= threshold { "yes" } else { "no" }
            )
            .expect("write to String is infallible"),
            None => writeln!(out, "| {genre} | n/a | no |").expect("write to String is infallible"),
        }
    }
    out
}

#[cfg(test)]
// Comparisons below are against exact hand-computed literals (e.g. a
// perfect-separation AUC of exactly 1.0), not approximated sums — exact
// equality is the correct check there. Where the expected value involves
// real division (the tie-handling case), an explicit epsilon check is
// used instead of `assert_eq!`.
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    // --- mann_whitney_auc: hand-computed fixtures ---

    /// No ties, llm strictly greater throughout: every one of the 3*3
    /// pairs is an llm win, so the raw AUC is 1.0 and stays 1.0 oriented.
    #[test]
    fn mann_whitney_auc_perfect_separation_llm_higher() {
        let (auc, direction) = mann_whitney_auc(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]).unwrap();
        assert_eq!(auc, 1.0);
        assert_eq!(direction, Direction::LlmHigher);
    }

    /// Same data, classes swapped: the raw statistic now favors human
    /// (llm always smaller), but the *oriented* AUC is still 1.0 —
    /// direction flips to `LlmLower`.
    #[test]
    fn mann_whitney_auc_perfect_separation_llm_lower_orients_to_same_auc() {
        let (auc, direction) = mann_whitney_auc(&[4.0, 5.0, 6.0], &[1.0, 2.0, 3.0]).unwrap();
        assert_eq!(auc, 1.0);
        assert_eq!(direction, Direction::LlmLower);
    }

    /// human = [1,2,3], llm = [2,3,4] (two ties). Hand count: comparing
    /// each of the 9 (human, llm) pairs, llm wins outright in 6, ties in
    /// 2 (worth 0.5 each), loses in 1: (6 + 2*0.5) / 9 = 7/9 ≈ 0.7778.
    #[test]
    fn mann_whitney_auc_matches_hand_computed_value_with_ties() {
        let (auc, direction) = mann_whitney_auc(&[1.0, 2.0, 3.0], &[2.0, 3.0, 4.0]).unwrap();
        assert!((auc - 7.0 / 9.0).abs() < 1e-12, "auc={auc}");
        assert_eq!(direction, Direction::LlmHigher);
    }

    /// Fully overlapping identical distributions: every pair ties, so the
    /// raw AUC is exactly 0.5 (already oriented — no class dominates).
    #[test]
    fn mann_whitney_auc_identical_distributions_is_one_half() {
        let (auc, _direction) = mann_whitney_auc(&[5.0, 5.0, 5.0], &[5.0, 5.0, 5.0]).unwrap();
        assert_eq!(auc, 0.5);
    }

    /// An empty group makes the AUC undefined.
    #[test]
    fn mann_whitney_auc_empty_group_is_none() {
        assert!(mann_whitney_auc(&[], &[1.0]).is_none());
        assert!(mann_whitney_auc(&[1.0], &[]).is_none());
    }

    // --- exceedance: hand-computed fixtures ---

    /// A value inside `[0, 10]`, including both boundaries, exceeds by
    /// exactly `0.0`.
    #[test]
    fn exceedance_inside_band_including_boundaries_is_zero() {
        let band = Envelope::new(0.0, 10.0);
        assert_eq!(exceedance(5.0, band), 0.0);
        assert_eq!(exceedance(0.0, band), 0.0);
        assert_eq!(exceedance(10.0, band), 0.0);
    }

    /// Band `[10, 20]` (width 10): a value 5 below `lo` exceeds by
    /// `5 / 10 = 0.5`; a value 5 above `hi` also exceeds by `0.5` — same
    /// magnitude on either side, since the formula only cares about
    /// distance to the *nearer* edge.
    #[test]
    fn exceedance_matches_hand_computed_value_on_either_side() {
        let band = Envelope::new(10.0, 20.0);
        assert!((exceedance(5.0, band) - 0.5).abs() < 1e-12);
        assert!((exceedance(25.0, band) - 0.5).abs() < 1e-12);
    }

    /// A miss more than a full band-width beyond the edge is capped at
    /// `1.0`, not left to grow unbounded.
    #[test]
    fn exceedance_caps_at_one() {
        let band = Envelope::new(10.0, 20.0);
        assert_eq!(exceedance(1000.0, band), 1.0);
        assert_eq!(exceedance(-1000.0, band), 1.0);
    }

    /// A zero-width band (`lo == hi`, e.g. a genre where every train-human
    /// doc had the exact same value) treats any miss as full exceedance
    /// rather than dividing by zero.
    #[test]
    fn exceedance_zero_width_band_treats_any_miss_as_full() {
        let band = Envelope::new(5.0, 5.0);
        assert_eq!(exceedance(5.0, band), 0.0);
        assert_eq!(exceedance(5.001, band), 1.0);
        assert_eq!(exceedance(4.999, band), 1.0);
    }

    // --- combined_score ---

    fn band(lo: f64, hi: f64, direction: Direction, include: bool) -> MetricBand {
        MetricBand {
            envelope: Envelope::new(lo, hi),
            direction,
            include,
            train_auc: None,
        }
    }

    fn bands_all(lo: f64, hi: f64) -> BTreeMap<String, MetricBand> {
        MetricVector::FIELD_NAMES
            .iter()
            .map(|&name| (name.to_string(), band(lo, hi, Direction::LlmHigher, true)))
            .collect()
    }

    /// A vector entirely within `[0, 10]` on every included field scores
    /// exactly `0.0` (mean of all-zero exceedances).
    #[test]
    fn combined_score_all_inside_is_zero() {
        let bands = bands_all(0.0, 10.0);
        let metrics = MetricVector {
            triad_rate: 5.0,
            ..MetricVector::default()
        };
        assert_eq!(combined_score(&metrics, &bands), Some(0.0));
    }

    /// Exactly one field (`triad_rate`) outside its `[0, 1]` band, by 4x
    /// the band width (capped at `1.0`): mean over `N` included fields is
    /// `1.0 / N`.
    #[test]
    fn combined_score_matches_hand_computed_mean_exceedance() {
        let bands = bands_all(0.0, 1.0);
        let metrics = MetricVector {
            triad_rate: 5.0,
            ..MetricVector::default()
        };
        let score = combined_score(&metrics, &bands).unwrap();
        let n = MetricVector::FIELD_NAMES.len();
        #[allow(clippy::cast_precision_loss)]
        let expected = 1.0 / n as f64;
        assert!((score - expected).abs() < 1e-12, "score={score}");
    }

    /// A band boundary is inclusive: a value exactly at `hi` counts as
    /// inside.
    #[test]
    fn combined_score_boundary_value_counts_as_inside() {
        let bands = bands_all(0.0, 5.0);
        let metrics = MetricVector {
            triad_rate: 5.0,
            ..MetricVector::default()
        };
        assert_eq!(combined_score(&metrics, &bands), Some(0.0));
    }

    /// A missing metric in `bands` makes the score undefined.
    #[test]
    fn combined_score_missing_metric_is_none() {
        let mut bands = bands_all(0.0, 10.0);
        bands.remove("triad_rate");
        assert_eq!(combined_score(&MetricVector::default(), &bands), None);
    }

    /// The inclusion-rule example: `triad_rate` is excluded
    /// (`include: false`) even though its value is wildly out of band —
    /// an excluded field contributes nothing to the mean, so a document
    /// whose only miss is on an excluded field scores a perfect `0.0`,
    /// not some fraction reflecting that miss.
    #[test]
    fn combined_score_skips_excluded_fields() {
        let mut bands = bands_all(0.0, 1.0);
        bands.insert(
            "triad_rate".to_string(),
            band(0.0, 1.0, Direction::LlmHigher, false),
        );
        let metrics = MetricVector {
            triad_rate: 100.0, // wildly out of [0, 1], but excluded
            ..MetricVector::default()
        };
        assert_eq!(combined_score(&metrics, &bands), Some(0.0));
    }

    /// When *every* field is excluded, there is no basis for a score at
    /// all — `None`, the same "undefined" signal as a missing band,
    /// rather than a fabricated `0.0`.
    #[test]
    fn combined_score_all_excluded_is_none() {
        let bands: BTreeMap<String, MetricBand> = MetricVector::FIELD_NAMES
            .iter()
            .map(|&name| {
                (
                    name.to_string(),
                    band(0.0, 1.0, Direction::LlmHigher, false),
                )
            })
            .collect();
        assert_eq!(combined_score(&MetricVector::default(), &bands), None);
    }

    // --- envelope pack parsing ---

    /// A pack in exactly the shape `corpus-tool envelope` writes (v2:
    /// `lo`/`hi`/`direction`/`include`/`train_auc`, plus the `[pack]`
    /// header's `auc_include_threshold`) parses into the expected
    /// genre -> metric -> `MetricBand` map.
    #[test]
    fn parse_envelope_pack_reads_genre_metric_bands() {
        let text = r#"
[pack]
version = "envelope-v2"
percentile_method = "nearest-rank"
lo_percentile = 10.0
hi_percentile = 90.0
auc_include_threshold = 0.55
corpus_manifest_sha256 = "deadbeef"
train_human_doc_count = 2
train_llm_doc_count = 2

[pack.human_docs_per_genre]
docs = 2

[pack.llm_docs_per_genre]
docs = 2

[docs.triad_rate]
lo = 0.1
hi = 0.9
direction = "llm higher"
include = true
train_auc = 0.7
"#;
        let pack = parse_envelope_pack(text).unwrap();
        assert_eq!(pack.auc_include_threshold, Some(0.55));
        let band = pack.bands["docs"]["triad_rate"];
        assert_eq!(band.envelope.lo, 0.1);
        assert_eq!(band.envelope.hi, 0.9);
        assert_eq!(band.direction, Direction::LlmHigher);
        assert!(band.include);
        assert_eq!(band.train_auc, Some(0.7));
    }

    /// A metric entry with no `train_auc` line (the "no train-split llm
    /// docs for this genre" case) parses to `train_auc: None`.
    #[test]
    fn parse_envelope_pack_missing_train_auc_is_none() {
        let text = r#"
[docs.triad_rate]
lo = 0.1
hi = 0.9
direction = "llm higher"
include = true
"#;
        let pack = parse_envelope_pack(text).unwrap();
        assert_eq!(pack.bands["docs"]["triad_rate"].train_auc, None);
    }

    /// An unrecognized `direction` string is a hard parse error, not a
    /// silently defaulted direction.
    #[test]
    fn parse_envelope_pack_rejects_unrecognized_direction() {
        let text = r#"
[docs.triad_rate]
lo = 0.1
hi = 0.9
direction = "sideways"
include = true
"#;
        assert!(parse_envelope_pack(text).is_err());
    }

    // --- report rendering ---

    fn empty_pack() -> LoadedPack {
        LoadedPack {
            bands: BTreeMap::new(),
            auc_include_threshold: None,
        }
    }

    /// The rendered report contains every genre's section, in
    /// declaration order, plus the summary line for a genre with data.
    #[test]
    fn render_report_contains_all_genre_sections_and_summary() {
        let mut by_genre: BTreeMap<Genre, GenreVectors> = BTreeMap::new();
        by_genre.insert(
            Genre::Docs,
            GenreVectors {
                human: vec![MetricVector {
                    triad_rate: 1.0,
                    ..MetricVector::default()
                }],
                llm: vec![MetricVector {
                    triad_rate: 9.0,
                    ..MetricVector::default()
                }],
            },
        );

        let report = render_report(&by_genre, &empty_pack());
        assert!(report.contains("## docs"));
        assert!(report.contains("## blog"));
        assert!(report.contains("## readme"));
        assert!(report.contains("## email"));
        assert!(report.contains("## forum"));
        assert!(report.contains("| triad_rate | 1 | 1 | 1.0000 | llm higher |"));
        assert!(report.contains("Summary: docs — human n=1, llm n=1"));
        assert!(report.contains("no envelope for this genre"));
        assert!(report.contains("Combined-score metrics: n/a (no envelope for this genre)."));
    }

    /// A genre whose envelope entry excludes one metric shows that
    /// metric's name and train AUC in the "Combined-score metrics" line.
    #[test]
    fn render_report_states_excluded_metrics_and_why() {
        let mut by_genre: BTreeMap<Genre, GenreVectors> = BTreeMap::new();
        by_genre.insert(
            Genre::Docs,
            GenreVectors {
                human: vec![MetricVector::default()],
                llm: vec![MetricVector::default()],
            },
        );
        let mut docs_bands = bands_all(0.0, 10.0);
        docs_bands.insert(
            "triad_rate".to_string(),
            MetricBand {
                envelope: Envelope::new(0.0, 10.0),
                direction: Direction::LlmHigher,
                include: false,
                train_auc: Some(0.5),
            },
        );
        let mut bands = BTreeMap::new();
        bands.insert("docs".to_string(), docs_bands);
        let pack = LoadedPack {
            bands,
            auc_include_threshold: Some(0.55),
        };

        let report = render_report(&by_genre, &pack);
        let names_len = MetricVector::FIELD_NAMES.len();
        assert!(report.contains(&format!(
            "Combined-score metrics: {} of {names_len} included; excluded: triad_rate (train \
             AUC 0.5000, llm higher).",
            names_len - 1
        )));
    }

    /// Rendering twice from the same input is byte-identical.
    #[test]
    fn render_report_is_deterministic() {
        let mut by_genre: BTreeMap<Genre, GenreVectors> = BTreeMap::new();
        by_genre.insert(
            Genre::Blog,
            GenreVectors {
                human: vec![MetricVector::default()],
                llm: vec![MetricVector::default()],
            },
        );
        let a = render_report(&by_genre, &empty_pack());
        let b = render_report(&by_genre, &empty_pack());
        assert_eq!(a, b);
    }
}
