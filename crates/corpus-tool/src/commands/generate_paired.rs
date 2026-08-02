//! `corpus-tool generate-paired` — plans and generates the stock/antislop
//! paired mining corpus under `corpus/paired/`.
//!
//! Reads `corpus/genconfig-paired.toml` (see `crate::paired_genconfig`)
//! and `corpus/prompts/<genre>.toml` (see `crate::prompts`, the same
//! catalogs `generate` reads), builds a deterministic job plan, and —
//! unless `--dry-run` — executes it against a local Ollama server
//! (`crate::ollama`), writing `corpus/paired/<side>/<genre>/<pair_id>.md`
//! plus a manifest record per generated doc.
//!
//! # Why "paired"
//!
//! Every `(genre, prompt)` combination is answered twice — once by the
//! `stock` model, once by the `antislop`-tuned counterpart of the *same*
//! base model — under the identical derived seed (see
//! [`derive_paired_seed`], which deliberately omits `model` from its
//! hash input, unlike `commands::generate::derive_seed`). That shared
//! seed and shared `pair_id` is the whole point: `mine-paired` mines the
//! stock/antislop contrast per prompt, so topic vocabulary cancels at
//! the pairing level rather than needing a topic-independent gate the
//! way `mine-inventory` needs one against the human corpus.
//!
//! This corpus is a MINING RESOURCE, not evaluation data: it is never
//! added to `corpus/manifest.jsonl`, never split train/dev/holdout, and
//! this module never reads or writes `corpus/holdout.lock`.
//!
//! # Scope: `prompts_per_genre`
//!
//! `corpus/prompts/*.toml` catalogs have grown well past the
//! (documentation-only) "11 prompts x 6 models = 66" figure in
//! `corpus/genconfig.toml`'s comments. Rather than silently using every
//! available prompt (which would balloon the paired corpus every time a
//! prompt is added, breaking any fixed-size contract downstream), the
//! scope is pinned explicitly by `PairedGenConfig::prompts_per_genre`:
//! always the first N prompts per genre in ascending `id` order — the
//! same sort `crate::prompts::load` already applies. A genre with fewer
//! prompts than `prompts_per_genre` is a hard error (see
//! [`build_paired_plan`]), never a silent truncation.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::time::Duration;

use anyhow::Context;
use clap::Args as ClapArgs;

use crate::commands::generate;
use crate::hashing::{sha256_hex, stable_hash_u64};
use crate::manifest::{Genre, ModelInfo};
use crate::ollama::{DEFAULT_TIMEOUT_SECS, GenerateParams, ModelDigest, OllamaClient};
use crate::paired_genconfig::{self, PairedGenConfig};
use crate::paired_manifest::{self, PairedManifestRecord, Side};
use crate::prompts::{self, Prompt};

/// Arguments for `corpus-tool generate-paired`.
#[derive(Debug, ClapArgs)]
pub struct Args {
    /// Paired corpus root directory.
    #[arg(long, default_value = "corpus/paired")]
    pub paired_dir: PathBuf,
    /// Path to the paired generation config.
    #[arg(long, default_value = "corpus/genconfig-paired.toml")]
    pub genconfig: PathBuf,
    /// Directory of `<genre>.toml` prompt files — the same catalogs
    /// `generate` reads.
    #[arg(long, default_value = "corpus/prompts")]
    pub prompts_dir: PathBuf,
    /// Print the planned jobs, in deterministic order, without calling
    /// Ollama or touching the paired corpus.
    #[arg(long)]
    pub dry_run: bool,
    /// Cap the total number of planned side-jobs (applied after building
    /// the full deterministic plan, so it's a prefix of it).
    #[arg(long)]
    pub limit: Option<usize>,
    /// Restrict generation to a single genre.
    #[arg(long)]
    pub genre: Option<String>,
    /// Restrict generation to a single side (`stock` or `antislop`):
    /// lets one side top up alone without regenerating the other.
    #[arg(long)]
    pub side: Option<String>,
}

/// One planned `(genre, prompt, side)` job, fully resolved (including
/// its deterministic pair id and seed, shared with the other side of the
/// same pair) before any network call is made.
#[derive(Debug, Clone, PartialEq)]
pub struct PlannedPairedJob {
    pub genre: Genre,
    pub side: Side,
    pub model: String,
    pub prompt_id: String,
    pub prompt_text: String,
    pub temperature: f64,
    pub seed: u64,
    pub pair_id: String,
}

/// Summary of a `generate-paired` run.
///
/// Returned so callers (the CLI dispatcher, tests) can decide what to
/// report/exit with — this module never calls `std::process::exit`
/// itself, keeping [`run`] safe to call from tests.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GeneratePairedOutcome {
    pub planned: usize,
    pub generated: usize,
    pub skipped_existing: usize,
    pub skipped_models: BTreeSet<String>,
}

impl GeneratePairedOutcome {
    /// True if any job was skipped because its model wasn't available in
    /// Ollama — the caller should exit with
    /// [`generate::EXIT_CODE_MODELS_SKIPPED`] (reused as-is: same
    /// semantics as a `generate` run skipping a matrix model).
    pub fn any_models_skipped(&self) -> bool {
        !self.skipped_models.is_empty()
    }
}

/// Runs `generate-paired`.
///
/// # Errors
///
/// Returns an error if the paired genconfig or a required prompt file
/// can't be read/parsed, an unknown `--genre`/`--side` filter is given,
/// a genre has fewer prompts than `prompts_per_genre` requires, or (in
/// live mode) a manifest/file I/O or Ollama request fails outright. A
/// model missing from Ollama is *not* an error — see
/// [`GeneratePairedOutcome::skipped_models`].
pub fn run(args: &Args) -> anyhow::Result<GeneratePairedOutcome> {
    let config = paired_genconfig::load(&args.genconfig)?;

    let genre_filter = args
        .genre
        .as_deref()
        .map(generate::parse_genre)
        .transpose()?;
    let side_filter = args.side.as_deref().map(parse_side).transpose()?;
    let genres_needed: Vec<Genre> =
        genre_filter.map_or_else(|| generate::ALL_GENRES.to_vec(), |g| vec![g]);

    let mut prompts_by_genre: BTreeMap<Genre, Vec<Prompt>> = BTreeMap::new();
    for genre in &genres_needed {
        prompts_by_genre.insert(*genre, prompts::load(&args.prompts_dir, *genre)?);
    }

    let plan = build_paired_plan(
        &config,
        &prompts_by_genre,
        genre_filter,
        side_filter,
        args.limit,
    )?;

    if args.dry_run {
        print!("{}", render_plan(&plan));
        println!("generate-paired --dry-run: {} planned job(s)", plan.len());
        return Ok(GeneratePairedOutcome {
            planned: plan.len(),
            ..GeneratePairedOutcome::default()
        });
    }

    execute_paired_plan(args, &config, plan)
}

/// Builds the deterministic paired job plan.
///
/// Genres are walked in the frozen [`generate::ALL_GENRES`] order (or
/// just the one requested by `genre_filter`). For each genre, the first
/// `config.prompts_per_genre` prompts (already sorted ascending by id —
/// see `crate::prompts::load`) are used. A genre with fewer prompts than
/// that is a hard error, never a silent truncation, since the paired
/// corpus's size guarantee depends on every genre contributing exactly N
/// prompts. For each of those N prompts, one seed and one `pair_id` are
/// derived (shared by both sides), and a stock job is emitted before an
/// antislop job — a fixed order, so `--dry-run` output and
/// `outcome.planned` count side-jobs 1:1. `side_filter` restricts which
/// of the two is emitted; `limit` truncates the flat side-job list
/// afterward.
///
/// # Errors
///
/// Returns an error if any requested genre's prompt list is shorter than
/// `config.prompts_per_genre`.
pub fn build_paired_plan(
    config: &PairedGenConfig,
    prompts_by_genre: &BTreeMap<Genre, Vec<Prompt>>,
    genre_filter: Option<Genre>,
    side_filter: Option<Side>,
    limit: Option<usize>,
) -> anyhow::Result<Vec<PlannedPairedJob>> {
    let genres: Vec<Genre> =
        genre_filter.map_or_else(|| generate::ALL_GENRES.to_vec(), |g| vec![g]);

    let mut jobs = Vec::new();
    for genre in genres {
        let empty = Vec::new();
        let prompts = prompts_by_genre.get(&genre).unwrap_or(&empty);
        if prompts.len() < config.prompts_per_genre {
            anyhow::bail!(
                "generate-paired: genre \"{genre}\" has {} prompts but \
                 paired.prompts_per_genre={} requires at least that many",
                prompts.len(),
                config.prompts_per_genre
            );
        }

        for prompt in &prompts[..config.prompts_per_genre] {
            let seed = derive_paired_seed(config.base_seed, genre, &prompt.id);
            let pair_id = derive_pair_id(genre, &prompt.id, seed, config.temperature);

            for side in [Side::Stock, Side::Antislop] {
                if side_filter.is_some_and(|filter| filter != side) {
                    continue;
                }
                let model = match side {
                    Side::Stock => &config.models.stock,
                    Side::Antislop => &config.models.antislop,
                };
                jobs.push(PlannedPairedJob {
                    genre,
                    side,
                    model: model.clone(),
                    prompt_id: prompt.id.clone(),
                    prompt_text: prompt.text.clone(),
                    temperature: config.temperature,
                    seed,
                    pair_id: pair_id.clone(),
                });
            }
        }
    }

    if let Some(limit) = limit {
        jobs.truncate(limit);
    }
    Ok(jobs)
}

/// Derives the seed shared by both sides of one `(genre, prompt)` pair:
/// `base_seed` plus a stable hash of `(genre, prompt_id)`, reduced
/// modulo a fixed prime so the result stays a modest, predictable
/// magnitude. No ambient RNG.
///
/// Deliberately omits `model` entirely, unlike
/// `commands::generate::derive_seed` — that's the whole point: "same
/// prompt, same seed, only the model differs" is what makes the
/// stock/antislop pair a controlled contrast.
fn derive_paired_seed(base_seed: u64, genre: Genre, prompt_id: &str) -> u64 {
    let input = format!("{genre}|{prompt_id}");
    let hash = stable_hash_u64(input.as_bytes());
    base_seed.wrapping_add(hash % 1_000_000_007)
}

/// Derives the pair id shared by both sides of one pair: the first 16
/// hex chars of `sha256(genre + prompt_id + seed + temperature)`,
/// temperature formatted to a fixed 2 decimal places so the hash input
/// never varies with float-formatting quirks.
///
/// Mirrors `commands::generate::derive_doc_id` exactly in shape (sha256
/// prefix, fixed 2-decimal temperature format) but likewise omits
/// `model`, so `corpus/paired/stock/<genre>/<id>.md` and
/// `corpus/paired/antislop/<genre>/<id>.md` name the SAME id for one
/// matched pair.
fn derive_pair_id(genre: Genre, prompt_id: &str, seed: u64, temperature: f64) -> String {
    let input = format!("{genre}{prompt_id}{seed}{temperature:.2}");
    sha256_hex(input.as_bytes())[..16].to_string()
}

/// Parses a side value (`"stock"`/`"antislop"`, case-insensitive) —
/// mirrors `commands::generate::parse_genre`'s shape.
fn parse_side(raw: &str) -> anyhow::Result<Side> {
    match raw.to_ascii_lowercase().as_str() {
        "stock" => Ok(Side::Stock),
        "antislop" => Ok(Side::Antislop),
        other => anyhow::bail!(
            "generate-paired: unknown side \"{other}\" (expected one of stock, antislop)"
        ),
    }
}

/// Renders the job plan as tab-separated lines (genre, side, model,
/// prompt id, temperature, seed, pair id), in plan order — the
/// `--dry-run` output.
fn render_plan(plan: &[PlannedPairedJob]) -> String {
    use std::fmt::Write as _;

    let mut out = String::new();
    for job in plan {
        writeln!(
            out,
            "{}\t{}\t{}\t{}\t{:.2}\t{}\t{}",
            job.genre, job.side, job.model, job.prompt_id, job.temperature, job.seed, job.pair_id
        )
        .expect("write to String is infallible");
    }
    out
}

/// Executes `plan` against Ollama: skips side-jobs already in the paired
/// manifest and jobs whose model isn't currently pulled (warn + skip,
/// continue), fetches each needed model's digest once, generates, writes
/// `corpus/paired/<side>/<genre>/<pair_id>.md`, and appends a manifest
/// record per doc: incrementally, so a crash mid-run loses at most the
/// in-flight job.
fn execute_paired_plan(
    args: &Args,
    config: &PairedGenConfig,
    plan: Vec<PlannedPairedJob>,
) -> anyhow::Result<GeneratePairedOutcome> {
    let manifest_path = args.paired_dir.join("manifest.jsonl");
    let mut records = paired_manifest::read_paired_manifest(&manifest_path)?.unwrap_or_default();
    let mut known: BTreeSet<(String, Side)> = records
        .iter()
        .map(|r| (r.pair_id.clone(), r.side))
        .collect();

    let timeout = Duration::from_secs(
        config
            .ollama
            .request_timeout_secs
            .unwrap_or(DEFAULT_TIMEOUT_SECS),
    );
    let client = OllamaClient::new(config.ollama.endpoint.clone(), timeout);
    let available = client
        .available_models()
        .context("generate-paired: querying ollama /api/tags")?;

    let mut digest_cache: BTreeMap<String, ModelDigest> = BTreeMap::new();
    let mut outcome = GeneratePairedOutcome {
        planned: plan.len(),
        ..GeneratePairedOutcome::default()
    };

    for job in plan {
        let key = (job.pair_id.clone(), job.side);
        if known.contains(&key) {
            outcome.skipped_existing += 1;
            continue;
        }

        if !available.contains(&job.model) {
            if outcome.skipped_models.insert(job.model.clone()) {
                eprintln!(
                    "generate-paired: warning: model \"{}\" not available in ollama, \
                     skipping its jobs (run `ollama pull {}` to enable them)",
                    job.model, job.model
                );
            }
            continue;
        }

        let digest = fetch_digest_cached(&client, &mut digest_cache, &job.model)?;
        let record = run_one_paired_job(args, config, &client, &job, &digest)?;

        known.insert((record.pair_id.clone(), record.side));
        records.push(record);
        paired_manifest::write_paired_manifest(&manifest_path, &records)?;
        outcome.generated += 1;
    }

    print_summary(&outcome);
    Ok(outcome)
}

/// Runs a single job that's already known to be new (not in the paired
/// manifest) and available (its model is pulled): generates the
/// completion, regenerates once more with the identical config to verify
/// reproducibility, writes
/// `corpus/paired/<side>/<genre>/<pair_id>.md` (from the first
/// generation), and builds the matching manifest record.
fn run_one_paired_job(
    args: &Args,
    config: &PairedGenConfig,
    client: &OllamaClient,
    job: &PlannedPairedJob,
    digest: &ModelDigest,
) -> anyhow::Result<PairedManifestRecord> {
    let params = GenerateParams {
        temperature: job.temperature,
        seed: job.seed,
        num_predict: config.ollama.num_predict,
    };
    let response = client
        .generate(&job.model, &job.prompt_text, params)
        .with_context(|| {
            format!(
                "generate-paired: {} {} ({}/{})",
                job.pair_id, job.side, job.model, job.prompt_id
            )
        })?;

    // Same honest, actually-checked reproducibility discipline as
    // `commands::generate::run_one_job`: regenerate with the identical
    // config and compare raw output bytes, never assume Ollama's seed
    // guarantees determinism.
    let retry = client
        .generate(&job.model, &job.prompt_text, params)
        .with_context(|| {
            format!(
                "generate-paired: reproducibility check for {} {} ({}/{})",
                job.pair_id, job.side, job.model, job.prompt_id
            )
        })?;

    let mut content = response.trim().to_string();
    content.push('\n');

    let relpath = format!("{}/{}/{}.md", job.side, job.genre, job.pair_id);
    let doc_path = args.paired_dir.join(&relpath);
    if let Some(parent) = doc_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    std::fs::write(&doc_path, &content)
        .with_context(|| format!("writing {}", doc_path.display()))?;

    Ok(build_paired_record(
        job,
        digest,
        config.ollama.num_predict,
        &response,
        &retry,
        sha256_hex(content.as_bytes()),
    ))
}

/// Builds one job's [`PairedManifestRecord`], honestly computing
/// `reproducible` from whether `first` and `retry` (two same-config
/// generations) produced byte-identical output — this is the only place
/// `reproducible` is decided; it must never be a literal.
fn build_paired_record(
    job: &PlannedPairedJob,
    digest: &ModelDigest,
    num_predict: u32,
    first: &str,
    retry: &str,
    sha256: String,
) -> PairedManifestRecord {
    PairedManifestRecord {
        pair_id: job.pair_id.clone(),
        side: job.side,
        genre: job.genre,
        model: ModelInfo {
            name: job.model.clone(),
            quantization: digest.quantization.clone(),
        },
        model_digest: digest.digest.clone(),
        prompt_id: job.prompt_id.clone(),
        seed: job.seed,
        temperature: job.temperature,
        num_predict,
        reproducible: first == retry,
        sha256,
    }
}

fn print_summary(outcome: &GeneratePairedOutcome) {
    let unavailable = if outcome.skipped_models.is_empty() {
        String::new()
    } else {
        let names: Vec<&str> = outcome.skipped_models.iter().map(String::as_str).collect();
        format!(" ({})", names.join(", "))
    };
    println!(
        "generate-paired: {} generated, {} already in manifest, {} model(s) unavailable{unavailable}",
        outcome.generated,
        outcome.skipped_existing,
        outcome.skipped_models.len(),
    );
}

fn fetch_digest_cached(
    client: &OllamaClient,
    cache: &mut BTreeMap<String, ModelDigest>,
    model: &str,
) -> anyhow::Result<ModelDigest> {
    if let Some(digest) = cache.get(model) {
        return Ok(digest.clone());
    }
    let digest = client
        .show(model)
        .with_context(|| format!("generate-paired: fetching digest for model \"{model}\""))?;
    cache.insert(model.to_string(), digest.clone());
    Ok(digest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genconfig::OllamaConfig;
    use crate::paired_genconfig::PairedModels;

    fn config(prompts_per_genre: usize) -> PairedGenConfig {
        PairedGenConfig {
            ollama: OllamaConfig {
                endpoint: "http://localhost:11434".to_string(),
                num_predict: 1536,
                request_timeout_secs: None,
            },
            models: PairedModels {
                stock: "stock-model".to_string(),
                antislop: "antislop-model".to_string(),
            },
            base_seed: 0,
            temperature: 0.7,
            prompts_per_genre,
        }
    }

    fn prompt(id: &str) -> Prompt {
        Prompt {
            id: id.to_string(),
            text: format!("prompt text for {id}"),
            topic: "t".to_string(),
        }
    }

    /// `derive_paired_seed` matches a hand-computed golden vector:
    /// `base_seed=0`, hash of `"docs|p001"` mod `1_000_000_007`.
    #[test]
    fn derive_paired_seed_matches_known_vector() {
        assert_eq!(derive_paired_seed(0, Genre::Docs, "p001"), 164_439_214);
    }

    /// `derive_pair_id` matches a hand-computed golden vector —
    /// `sha256("docsp001164439214" + "0.70")[..16]` — pinning the exact
    /// concatenation order (`genre` + `prompt_id` + `seed` +
    /// `temperature`) and float formatting.
    #[test]
    fn derive_pair_id_matches_known_vector() {
        let seed = derive_paired_seed(0, Genre::Docs, "p001");
        assert_eq!(
            derive_pair_id(Genre::Docs, "p001", seed, 0.7),
            "e4d74df7a1b23e1c"
        );
    }

    /// Seed and pair-id derivation are pure functions of their inputs:
    /// same inputs, same outputs, every time.
    #[test]
    fn derive_paired_seed_is_deterministic() {
        assert_eq!(
            derive_paired_seed(42, Genre::Blog, "b001"),
            derive_paired_seed(42, Genre::Blog, "b001")
        );
    }

    /// Changing any one of `genre`/`prompt_id`/`seed`/`temperature`
    /// changes the pair id (no accidental collisions from a degenerate
    /// concatenation).
    #[test]
    fn derive_pair_id_differs_when_any_component_differs() {
        let base = derive_pair_id(Genre::Docs, "p1", 1, 0.7);
        assert_ne!(base, derive_pair_id(Genre::Blog, "p1", 1, 0.7));
        assert_ne!(base, derive_pair_id(Genre::Docs, "p2", 1, 0.7));
        assert_ne!(base, derive_pair_id(Genre::Docs, "p1", 2, 0.7));
        assert_ne!(base, derive_pair_id(Genre::Docs, "p1", 1, 0.2));
    }

    /// `build_paired_plan` walks genres in the frozen `ALL_GENRES` order
    /// when no `--genre` filter is given.
    #[test]
    fn build_paired_plan_visits_genres_in_frozen_order() {
        let cfg = config(1);
        let mut by_genre = BTreeMap::new();
        for genre in generate::ALL_GENRES {
            by_genre.insert(genre, vec![prompt("p001")]);
        }
        let plan = build_paired_plan(&cfg, &by_genre, None, None, None).unwrap();
        let genres: Vec<Genre> = plan.iter().map(|j| j.genre).collect();
        let expected: Vec<Genre> = generate::ALL_GENRES.iter().flat_map(|&g| [g, g]).collect();
        assert_eq!(genres, expected);
    }

    /// Only the first `prompts_per_genre` prompts (ascending id order)
    /// are used, even when more are available.
    #[test]
    fn build_paired_plan_takes_first_n_prompts_in_ascending_id_order() {
        let cfg = config(2);
        let mut by_genre = BTreeMap::new();
        by_genre.insert(
            Genre::Docs,
            vec![prompt("p001"), prompt("p002"), prompt("p003")],
        );
        let plan = build_paired_plan(&cfg, &by_genre, Some(Genre::Docs), None, None).unwrap();
        let prompt_ids: BTreeSet<&str> = plan.iter().map(|j| j.prompt_id.as_str()).collect();
        assert_eq!(prompt_ids, BTreeSet::from(["p001", "p002"]));
        assert_eq!(plan.len(), 4); // 2 prompts x 2 sides
    }

    /// A genre with fewer prompts than `prompts_per_genre` requires is a
    /// hard error, never a silent truncation.
    #[test]
    fn build_paired_plan_errors_when_genre_has_fewer_prompts_than_required() {
        let cfg = config(5);
        let mut by_genre = BTreeMap::new();
        by_genre.insert(Genre::Docs, vec![prompt("p001"), prompt("p002")]);
        let err = build_paired_plan(&cfg, &by_genre, Some(Genre::Docs), None, None).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("docs"), "message was: {msg}");
        assert!(msg.contains('5'), "message was: {msg}");
    }

    /// `--side` restricts the plan to a single side, in order, without
    /// disturbing the deterministic seed/pair-id derivation.
    #[test]
    fn build_paired_plan_side_filter_restricts_to_one_side() {
        let cfg = config(2);
        let mut by_genre = BTreeMap::new();
        by_genre.insert(Genre::Docs, vec![prompt("p001"), prompt("p002")]);

        let plan = build_paired_plan(
            &cfg,
            &by_genre,
            Some(Genre::Docs),
            Some(Side::Antislop),
            None,
        )
        .unwrap();
        assert_eq!(plan.len(), 2);
        assert!(plan.iter().all(|j| j.side == Side::Antislop));
        assert!(plan.iter().all(|j| j.model == "antislop-model"));
    }

    /// `--limit` truncates the deterministic flat plan to a prefix.
    #[test]
    fn build_paired_plan_limit_truncates_to_prefix() {
        let cfg = config(3);
        let mut by_genre = BTreeMap::new();
        by_genre.insert(
            Genre::Docs,
            (1..=3).map(|i| prompt(&format!("p{i:03}"))).collect(),
        );

        let full = build_paired_plan(&cfg, &by_genre, Some(Genre::Docs), None, None).unwrap();
        let limited = build_paired_plan(&cfg, &by_genre, Some(Genre::Docs), None, Some(3)).unwrap();
        assert_eq!(limited.len(), 3);
        assert_eq!(limited.as_slice(), &full[..3]);
    }

    /// Planning the same scenario twice produces field-for-field
    /// identical jobs.
    #[test]
    fn build_paired_plan_is_deterministic() {
        let cfg = config(3);
        let mut by_genre = BTreeMap::new();
        by_genre.insert(
            Genre::Blog,
            (1..=3).map(|i| prompt(&format!("p{i:03}"))).collect(),
        );
        let first = build_paired_plan(&cfg, &by_genre, Some(Genre::Blog), None, None).unwrap();
        let second = build_paired_plan(&cfg, &by_genre, Some(Genre::Blog), None, None).unwrap();
        assert_eq!(first, second);
    }

    /// `parse_side` accepts the frozen set case-insensitively and
    /// rejects anything else with a clear error.
    #[test]
    fn parse_side_accepts_known_rejects_unknown() {
        assert_eq!(parse_side("Stock").unwrap(), Side::Stock);
        assert_eq!(parse_side("ANTISLOP").unwrap(), Side::Antislop);
        assert!(parse_side("chaos").is_err());
    }

    /// Golden test: the exact rendered plan text for a small, fully
    /// fixed scenario (1 genre x 2 prompts, `prompts_per_genre=2`,
    /// `base_seed=0`). Any change to the planning/derivation algorithm
    /// should be a deliberate, reviewed change to this literal string.
    #[test]
    fn render_paired_plan_golden_output_for_fixed_scenario() {
        let cfg = config(2);
        let mut by_genre = BTreeMap::new();
        by_genre.insert(Genre::Blog, vec![prompt("p001"), prompt("p002")]);
        let plan = build_paired_plan(&cfg, &by_genre, Some(Genre::Blog), None, None).unwrap();
        let rendered = render_plan(&plan);

        let expected = "\
blog\tstock\tstock-model\tp001\t0.70\t173164955\t9db2e88fd07d0d19
blog\tantislop\tantislop-model\tp001\t0.70\t173164955\t9db2e88fd07d0d19
blog\tstock\tstock-model\tp002\t0.70\t874557933\t906c337b853c3018
blog\tantislop\tantislop-model\tp002\t0.70\t874557933\t906c337b853c3018
";
        assert_eq!(rendered, expected);
    }

    /// Rendering the same plan twice is byte-identical.
    #[test]
    fn render_plan_is_deterministic() {
        let cfg = config(2);
        let mut by_genre = BTreeMap::new();
        by_genre.insert(Genre::Blog, vec![prompt("p001"), prompt("p002")]);
        let plan = build_paired_plan(&cfg, &by_genre, Some(Genre::Blog), None, None).unwrap();
        assert_eq!(render_plan(&plan), render_plan(&plan));
    }

    fn digest() -> ModelDigest {
        ModelDigest {
            digest: "sha256:deadbeef".to_string(),
            quantization: "Q4_K_M".to_string(),
        }
    }

    fn job() -> PlannedPairedJob {
        PlannedPairedJob {
            genre: Genre::Docs,
            side: Side::Stock,
            model: "stock-model".to_string(),
            prompt_id: "p001".to_string(),
            prompt_text: "write something".to_string(),
            temperature: 0.7,
            seed: 42,
            pair_id: "abc123".to_string(),
        }
    }

    /// `build_paired_record` records `reproducible: true` when a
    /// same-config regeneration matches the first generation
    /// byte-for-byte. The property the flag is supposed to certify,
    /// actually checked rather than assumed.
    #[test]
    fn paired_record_marks_reproducible_true_when_outputs_match() {
        let record = build_paired_record(
            &job(),
            &digest(),
            1536,
            "identical output",
            "identical output",
            "shasum".to_string(),
        );
        assert!(record.reproducible);
    }

    /// This is the regression the original hardcoded-`true` bug
    /// would have masked — when a same-seed regeneration produces
    /// different bytes, the flag must honestly record `false`, never a
    /// blanket `true`.
    #[test]
    fn paired_record_marks_reproducible_false_when_outputs_differ() {
        let record = build_paired_record(
            &job(),
            &digest(),
            1536,
            "first run output",
            "second run output, slightly different",
            "shasum".to_string(),
        );
        assert!(!record.reproducible);
    }

    /// `build_paired_record` also records the rest of the required
    /// fields verbatim (pair id, side, genre, model, seed, temperature).
    #[test]
    fn build_paired_record_records_full_fields() {
        let record = build_paired_record(&job(), &digest(), 1536, "a", "a", "shasum".to_string());
        assert_eq!(record.pair_id, "abc123");
        assert_eq!(record.side, Side::Stock);
        assert_eq!(record.genre, Genre::Docs);
        assert_eq!(record.model.name, "stock-model");
        assert_eq!(record.model.quantization, "Q4_K_M");
        assert_eq!(record.model_digest, "sha256:deadbeef");
        assert_eq!(record.seed, 42);
        #[allow(clippy::float_cmp)] // exact: assigned verbatim from `job.temperature`.
        {
            assert_eq!(record.temperature, 0.7);
        }
        assert_eq!(record.num_predict, 1536);
        assert_eq!(record.sha256, "shasum");
    }
}
