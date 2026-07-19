//! `corpus-tool generate` — plans jobs from the model matrix and
//! temperature/style slices, generates via Ollama, and records the full
//! generation config per doc.
//!
//! Reads `corpus/genconfig.toml` (see `crate::genconfig`) and
//! `corpus/prompts/<genre>.toml` (see `crate::prompts`), builds a
//! deterministic job plan, and — unless `--dry-run` — executes it against
//! a local Ollama server (`crate::ollama`), writing `corpus/llm/<genre>/
//! <id>.md` plus a manifest record per generated doc. Reruns are
//! incremental: a job whose deterministic doc id is already in
//! the manifest is skipped without calling Ollama again.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::time::Duration;

use anyhow::Context;
use clap::Args as ClapArgs;

use crate::genconfig::{self, GenConfig};
use crate::hashing::{sha256_hex, stable_hash_u64};
use crate::manifest::{self, Class, Genre, ManifestRecord, ModelInfo};
use crate::ollama::{DEFAULT_TIMEOUT_SECS, GenerateParams, OllamaClient};
use crate::prompts::{self, Prompt};

/// Process exit code for "completed but skipped some models".
///
/// Used by `cli::run`, not by this module — see [`GenerateOutcome`] —
/// when `generate` completed but skipped jobs because one or more matrix
/// models weren't available in Ollama. `0` is full success; a plain
/// `anyhow` error (mapped to `1` by the harness) covers everything else
/// (bad config, network failure mid-run, ...).
pub const EXIT_CODE_MODELS_SKIPPED: i32 = 3;

/// The frozen genre set, in the fixed order genres are planned and
/// printed.
const ALL_GENRES: [Genre; 5] = [
    Genre::Docs,
    Genre::Blog,
    Genre::Readme,
    Genre::Email,
    Genre::Forum,
];

/// Arguments for `corpus-tool generate`.
#[derive(Debug, ClapArgs)]
pub struct Args {
    /// Corpus root directory.
    #[arg(long, default_value = "corpus")]
    pub corpus_dir: PathBuf,
    /// Path to the generation config.
    #[arg(long, default_value = "corpus/genconfig.toml")]
    pub genconfig: PathBuf,
    /// Directory of `<genre>.toml` prompt files.
    #[arg(long, default_value = "corpus/prompts")]
    pub prompts_dir: PathBuf,
    /// Print the planned jobs, in deterministic order, without calling
    /// Ollama or touching the corpus.
    #[arg(long)]
    pub dry_run: bool,
    /// Cap the total number of planned jobs (applied after building the
    /// full deterministic plan, so it's a prefix of it).
    #[arg(long)]
    pub limit: Option<usize>,
    /// Restrict generation to a single model from the matrix.
    #[arg(long)]
    pub model: Option<String>,
    /// Restrict generation to a single genre.
    #[arg(long)]
    pub genre: Option<String>,
}

/// One planned `(model, prompt, temperature-slice, style-slice)` job,
/// fully resolved (including its deterministic doc id and seed) before
/// any network call is made.
#[derive(Debug, Clone, PartialEq)]
pub struct PlannedJob {
    pub genre: Genre,
    pub model: String,
    pub prompt_id: String,
    pub prompt_text: String,
    pub temperature: f64,
    pub style_prompted: bool,
    pub seed: u64,
    pub doc_id: String,
}

/// Summary of a `generate` run.
///
/// Returned so callers (the CLI dispatcher, tests) can decide what to
/// report/exit with — this module never calls `std::process::exit`
/// itself, keeping [`run`] safe to call from tests.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GenerateOutcome {
    pub planned: usize,
    pub generated: usize,
    pub skipped_existing: usize,
    pub skipped_models: BTreeSet<String>,
}

impl GenerateOutcome {
    /// True if any job was skipped because its model wasn't available in
    /// Ollama — the caller should exit with [`EXIT_CODE_MODELS_SKIPPED`].
    pub fn any_models_skipped(&self) -> bool {
        !self.skipped_models.is_empty()
    }
}

/// Runs `generate`.
///
/// # Errors
///
/// Returns an error if the genconfig or a required prompt file can't be
/// read/parsed, an unknown `--model`/`--genre` filter is given, or (in
/// live mode) a manifest/file I/O or Ollama request fails outright. A
/// model missing from Ollama is *not* an error — see
/// [`GenerateOutcome::skipped_models`].
pub fn run(args: &Args) -> anyhow::Result<GenerateOutcome> {
    let config = genconfig::load(&args.genconfig)?;

    let genre_filter = args.genre.as_deref().map(parse_genre).transpose()?;
    let genres_needed: Vec<Genre> = genre_filter.map_or_else(|| ALL_GENRES.to_vec(), |g| vec![g]);

    let mut prompts_by_genre: BTreeMap<Genre, Vec<Prompt>> = BTreeMap::new();
    for genre in &genres_needed {
        prompts_by_genre.insert(*genre, prompts::load(&args.prompts_dir, *genre)?);
    }

    let plan = build_plan(
        &config,
        &prompts_by_genre,
        genre_filter,
        args.model.as_deref(),
        args.limit,
    )?;

    if args.dry_run {
        print!("{}", render_plan(&plan));
        println!("generate --dry-run: {} planned job(s)", plan.len());
        return Ok(GenerateOutcome {
            planned: plan.len(),
            ..GenerateOutcome::default()
        });
    }

    execute_plan(args, &config, plan)
}

/// Builds the deterministic job plan.
///
/// For each requested genre, prompts are consumed in id order and
/// assigned to models round-robin (model-minor, prompt-major — see
/// [`plan_genre_jobs`]) up to `targets.docs_per_genre`, then the whole
/// cross-genre plan is truncated to `limit` if given.
///
/// # Errors
///
/// Returns an error if `model_filter` names a model not present in the
/// genconfig matrix.
pub fn build_plan(
    config: &GenConfig,
    prompts_by_genre: &BTreeMap<Genre, Vec<Prompt>>,
    genre_filter: Option<Genre>,
    model_filter: Option<&str>,
    limit: Option<usize>,
) -> anyhow::Result<Vec<PlannedJob>> {
    let models: Vec<String> = match model_filter {
        None => config.models.iter().map(|m| m.name.clone()).collect(),
        Some(name) => {
            if !config.models.iter().any(|m| m.name == name) {
                anyhow::bail!("generate: --model \"{name}\" is not in the genconfig model matrix");
            }
            vec![name.to_string()]
        }
    };

    let genres: Vec<Genre> = genre_filter.map_or_else(|| ALL_GENRES.to_vec(), |g| vec![g]);

    let mut jobs = Vec::new();
    for genre in genres {
        let empty = Vec::new();
        let prompts = prompts_by_genre.get(&genre).unwrap_or(&empty);
        jobs.extend(plan_genre_jobs(genre, prompts, &models, config));
    }

    if let Some(limit) = limit {
        jobs.truncate(limit);
    }
    Ok(jobs)
}

/// Plans one genre's jobs: `target = min(targets.docs_per_genre, P * M)`
/// jobs, `k = 0..target`, model `= models[k % M]`, prompt
/// `= prompts[(k / M) % P]` — every model gets an even share, and (for
/// `target <= P * M`, the common case) prompts are only reused once every
/// model has had a turn at the previous ones.
fn plan_genre_jobs(
    genre: Genre,
    prompts: &[Prompt],
    models: &[String],
    config: &GenConfig,
) -> Vec<PlannedJob> {
    let p = prompts.len();
    let m = models.len();
    if p == 0 || m == 0 {
        return Vec::new();
    }

    let target = config.targets.docs_per_genre.min(p * m);
    let low_every = config.temperature.low_every();
    let style_every = config.style_prompted.style_every();

    (0..target)
        .map(|k| {
            let model = &models[k % m];
            let prompt = &prompts[(k / m) % p];

            let low_temp = k % low_every == 0;
            let style_prompted = k % style_every == style_every - 1;
            let temperature = if low_temp {
                config.temperature.low
            } else {
                config.temperature.default
            };

            let slice = format!(
                "temp={}|style={style_prompted}",
                if low_temp { "low" } else { "default" }
            );
            let seed = derive_seed(config.base_seed, model, &prompt.id, &slice);
            let doc_id = derive_doc_id(model, &prompt.id, seed, temperature);

            PlannedJob {
                genre,
                model: model.clone(),
                prompt_id: prompt.id.clone(),
                prompt_text: prompt.text.clone(),
                temperature,
                style_prompted,
                seed,
                doc_id,
            }
        })
        .collect()
}

/// Derives a job's seed: `base_seed` plus a stable hash of
/// `(model, prompt_id, slice)`, reduced modulo a fixed prime so the
/// result stays a modest, predictable magnitude. No ambient RNG.
fn derive_seed(base_seed: u64, model: &str, prompt_id: &str, slice: &str) -> u64 {
    let input = format!("{model}|{prompt_id}|{slice}");
    let hash = stable_hash_u64(input.as_bytes());
    base_seed.wrapping_add(hash % 1_000_000_007)
}

/// Derives a job's deterministic doc id: the first 16 hex chars
/// of `sha256(model + prompt_id + seed + temperature)`, temperature
/// formatted to a fixed 2 decimal places so the hash input never varies
/// with float-formatting quirks.
fn derive_doc_id(model: &str, prompt_id: &str, seed: u64, temperature: f64) -> String {
    let input = format!("{model}{prompt_id}{seed}{temperature:.2}");
    sha256_hex(input.as_bytes())[..16].to_string()
}

/// Parses a genre value against the frozen genre set. Shared with
/// `ingest`, which parses the `genre` field of incoming metadata
/// fragments the same way.
pub(crate) fn parse_genre(raw: &str) -> anyhow::Result<Genre> {
    match raw.to_ascii_lowercase().as_str() {
        "docs" => Ok(Genre::Docs),
        "blog" => Ok(Genre::Blog),
        "readme" => Ok(Genre::Readme),
        "email" => Ok(Genre::Email),
        "forum" => Ok(Genre::Forum),
        other => anyhow::bail!(
            "generate: unknown genre \"{other}\" (expected one of docs, blog, readme, email, forum)"
        ),
    }
}

/// Renders the job plan as tab-separated lines (genre, model, prompt id,
/// temperature, `style_prompted`, seed, doc id), in plan order — the
/// `--dry-run` output.
fn render_plan(plan: &[PlannedJob]) -> String {
    use std::fmt::Write as _;

    let mut out = String::new();
    for job in plan {
        writeln!(
            out,
            "{}\t{}\t{}\t{:.2}\t{}\t{}\t{}",
            job.genre,
            job.model,
            job.prompt_id,
            job.temperature,
            job.style_prompted,
            job.seed,
            job.doc_id
        )
        .expect("write to String is infallible");
    }
    out
}

/// Wraps a prompt with the "sound human" style instruction. Only used
/// for the style-prompted slice; every other job passes `prompt_text`
/// verbatim with no system/style prompt at all.
fn wrap_style_prompt(prompt_text: &str, instruction: &str) -> String {
    format!("{prompt_text}\n\n{instruction}")
}

/// Executes `plan` against Ollama: skips jobs already in the manifest
/// and jobs whose model isn't currently pulled (warn + skip,
/// continue), fetches each needed model's digest once, generates, writes
/// `corpus/llm/<genre>/<id>.md`, and appends a manifest record per doc —
/// incrementally, so a crash mid-run loses at most the in-flight job.
fn execute_plan(
    args: &Args,
    config: &GenConfig,
    plan: Vec<PlannedJob>,
) -> anyhow::Result<GenerateOutcome> {
    let manifest_path = args.corpus_dir.join("manifest.jsonl");
    let mut records = manifest::read_manifest(&manifest_path)?.unwrap_or_default();
    let mut known_ids: BTreeSet<String> = records.iter().map(|r| r.id.clone()).collect();

    let timeout = Duration::from_secs(
        config
            .ollama
            .request_timeout_secs
            .unwrap_or(DEFAULT_TIMEOUT_SECS),
    );
    let client = OllamaClient::new(config.ollama.endpoint.clone(), timeout);
    let available = client
        .available_models()
        .context("generate: querying ollama /api/tags")?;

    let mut digest_cache: BTreeMap<String, crate::ollama::ModelDigest> = BTreeMap::new();
    let mut outcome = GenerateOutcome {
        planned: plan.len(),
        ..GenerateOutcome::default()
    };

    for job in plan {
        if known_ids.contains(&job.doc_id) {
            outcome.skipped_existing += 1;
            continue;
        }

        if !available.contains(&job.model) {
            if outcome.skipped_models.insert(job.model.clone()) {
                eprintln!(
                    "generate: warning: model \"{}\" not available in ollama, \
                     skipping its jobs (run `ollama pull {}` to enable them)",
                    job.model, job.model
                );
            }
            continue;
        }

        let digest = fetch_digest_cached(&client, &mut digest_cache, &job.model)?;
        let record = run_one_job(args, config, &client, &job, &digest)?;

        known_ids.insert(record.id.clone());
        records.push(record);
        manifest::write_manifest(&manifest_path, &records)?;
        outcome.generated += 1;
    }

    print_summary(&outcome);
    Ok(outcome)
}

/// Runs a single job that's already known to be new (not in the
/// manifest) and available (its model is pulled): generates the
/// completion, regenerates once more with the identical config to verify
/// reproducibility, writes `corpus/llm/<genre>/<id>.md` (from the
/// first generation), and builds the matching manifest record.
fn run_one_job(
    args: &Args,
    config: &GenConfig,
    client: &OllamaClient,
    job: &PlannedJob,
    digest: &crate::ollama::ModelDigest,
) -> anyhow::Result<ManifestRecord> {
    let prompt_text = if job.style_prompted {
        wrap_style_prompt(&job.prompt_text, &config.style_prompted.instruction)
    } else {
        job.prompt_text.clone()
    };

    let params = GenerateParams {
        temperature: job.temperature,
        seed: job.seed,
        num_predict: config.ollama.num_predict,
    };
    let response = client
        .generate(&job.model, &prompt_text, params)
        .with_context(|| format!("generate: {} ({}/{})", job.doc_id, job.model, job.prompt_id))?;

    // Regeneration with the same config must reproduce byte-identical
    // output where the runtime supports seeding, else record
    // non-reproducibility explicitly — so actually regenerate with the
    // identical config and compare raw output bytes rather than assuming
    // Ollama's seed guarantees determinism (it does not, in general,
    // across backends/batching).
    let retry = client
        .generate(&job.model, &prompt_text, params)
        .with_context(|| {
            format!(
                "generate: reproducibility check for {} ({}/{})",
                job.doc_id, job.model, job.prompt_id
            )
        })?;

    let mut content = response.trim().to_string();
    content.push('\n');

    let relpath = format!("llm/{}/{}.md", job.genre, job.doc_id);
    let doc_path = args.corpus_dir.join(&relpath);
    if let Some(parent) = doc_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    std::fs::write(&doc_path, &content)
        .with_context(|| format!("writing {}", doc_path.display()))?;

    Ok(ManifestRecord {
        id: job.doc_id.clone(),
        class: Class::Llm,
        genre: job.genre,
        source: "ollama".to_string(),
        model: Some(ModelInfo {
            name: job.model.clone(),
            quantization: digest.quantization.clone(),
        }),
        prompt_id: Some(job.prompt_id.clone()),
        license: "generated".to_string(),
        lang: "en".to_string(),
        split: None,
        sha256: sha256_hex(content.as_bytes()),
        provenance_evidence: None,
        style_prompted: job.style_prompted,
        gen_config: Some(build_gen_config(
            &config.ollama.endpoint,
            &digest.digest,
            job.temperature,
            job.seed,
            config.ollama.num_predict,
            &response,
            &retry,
        )),
    })
}

/// Builds the `gen_config` JSON recorded per generated doc: full
/// generation config plus an honestly-computed `reproducible` flag —
/// `true` only if `retry` (a same-config regeneration) produced
/// byte-identical output to `first`, never assumed. This is the only place
/// `reproducible` is decided; it must never be a literal.
fn build_gen_config(
    endpoint: &str,
    model_digest: &str,
    temperature: f64,
    seed: u64,
    num_predict: u32,
    first: &str,
    retry: &str,
) -> serde_json::Value {
    serde_json::json!({
        "endpoint": endpoint,
        "model_digest": model_digest,
        "temperature": temperature,
        "seed": seed,
        "num_predict": num_predict,
        "reproducible": first == retry,
    })
}

fn print_summary(outcome: &GenerateOutcome) {
    let unavailable = if outcome.skipped_models.is_empty() {
        String::new()
    } else {
        let names: Vec<&str> = outcome.skipped_models.iter().map(String::as_str).collect();
        format!(" ({})", names.join(", "))
    };
    println!(
        "generate: {} generated, {} already in manifest, {} model(s) unavailable{unavailable}",
        outcome.generated,
        outcome.skipped_existing,
        outcome.skipped_models.len(),
    );
}

fn fetch_digest_cached(
    client: &OllamaClient,
    cache: &mut BTreeMap<String, crate::ollama::ModelDigest>,
    model: &str,
) -> anyhow::Result<crate::ollama::ModelDigest> {
    if let Some(digest) = cache.get(model) {
        return Ok(digest.clone());
    }
    let digest = client
        .show(model)
        .with_context(|| format!("generate: fetching digest for model \"{model}\""))?;
    cache.insert(model.to_string(), digest.clone());
    Ok(digest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genconfig::{
        ModelSpec, OllamaConfig, StyleConfig, TargetsConfig, TemperatureConfig,
    };

    fn config(
        models: &[&str],
        docs_per_genre: usize,
        low_fraction: f64,
        style_fraction: f64,
    ) -> GenConfig {
        GenConfig {
            ollama: OllamaConfig {
                endpoint: "http://localhost:11434".to_string(),
                num_predict: 512,
                request_timeout_secs: None,
            },
            base_seed: 0,
            models: models
                .iter()
                .map(|&name| ModelSpec {
                    name: name.to_string(),
                })
                .collect(),
            temperature: TemperatureConfig {
                default: 0.7,
                low: 0.2,
                low_fraction,
            },
            style_prompted: StyleConfig {
                fraction: style_fraction,
                instruction: "sound human".to_string(),
            },
            targets: TargetsConfig { docs_per_genre },
        }
    }

    fn prompt(id: &str) -> Prompt {
        Prompt {
            id: id.to_string(),
            text: format!("prompt text for {id}"),
            topic: "t".to_string(),
        }
    }

    /// `derive_doc_id` matches a hand-computed golden vector —
    /// `sha256("mp10.70")[..16]` — pinning the exact concatenation order
    /// (`model` + `prompt_id` + `seed` + `temperature`) and float
    /// formatting.
    #[test]
    fn derive_doc_id_matches_known_vector() {
        assert_eq!(derive_doc_id("m", "p", 1, 0.7), "c0ec7e0ee15b215f");
    }

    /// `derive_seed` matches a hand-computed golden vector
    /// (`base_seed=0`, hash of `"m|p|s"` mod `1_000_000_007`).
    #[test]
    fn derive_seed_matches_known_vector() {
        assert_eq!(derive_seed(0, "m", "p", "s"), 186_061_091);
    }

    /// Seed and doc id derivation are pure functions of their
    /// inputs — same inputs, same outputs, every time.
    #[test]
    fn derive_functions_are_deterministic() {
        assert_eq!(
            derive_seed(42, "a", "b", "c"),
            derive_seed(42, "a", "b", "c")
        );
        assert_eq!(
            derive_doc_id("a", "b", 42, 0.7),
            derive_doc_id("a", "b", 42, 0.7)
        );
    }

    /// Changing any one of `model`/`prompt_id`/`seed`/`temperature`
    /// changes the doc id (no accidental collisions from a degenerate
    /// concatenation).
    #[test]
    fn derive_doc_id_differs_when_any_component_differs() {
        let base = derive_doc_id("m1", "p1", 1, 0.7);
        assert_ne!(base, derive_doc_id("m2", "p1", 1, 0.7));
        assert_ne!(base, derive_doc_id("m1", "p2", 1, 0.7));
        assert_ne!(base, derive_doc_id("m1", "p1", 2, 0.7));
        assert_ne!(base, derive_doc_id("m1", "p1", 1, 0.2));
    }

    /// `plan_genre_jobs` assigns models round-robin
    /// (model-minor): with 2 models and enough prompts, job k uses
    /// `models[k % 2]`.
    #[test]
    fn plan_genre_jobs_assigns_models_round_robin() {
        let cfg = config(&["alpha", "beta"], 4, 0.5, 0.5);
        let prompts: Vec<Prompt> = (1..=3).map(|i| prompt(&format!("p{i:03}"))).collect();
        let models: Vec<String> = cfg.models.iter().map(|m| m.name.clone()).collect();
        let jobs = plan_genre_jobs(Genre::Blog, &prompts, &models, &cfg);

        assert_eq!(jobs.len(), 4);
        assert_eq!(jobs[0].model, "alpha");
        assert_eq!(jobs[1].model, "beta");
        assert_eq!(jobs[2].model, "alpha");
        assert_eq!(jobs[3].model, "beta");
        // prompt-major: model[0] and model[1] both see p001 before p002.
        assert_eq!(jobs[0].prompt_id, "p001");
        assert_eq!(jobs[1].prompt_id, "p001");
        assert_eq!(jobs[2].prompt_id, "p002");
    }

    /// `target = min(docs_per_genre, prompts * models)` — a
    /// target larger than the available (prompt, model) combinations is
    /// capped, not padded with repeats.
    #[test]
    fn plan_genre_jobs_caps_target_at_available_combinations() {
        let cfg = config(&["alpha"], 100, 0.5, 0.5);
        let prompts: Vec<Prompt> = (1..=3).map(|i| prompt(&format!("p{i:03}"))).collect();
        let models: Vec<String> = cfg.models.iter().map(|m| m.name.clone()).collect();
        let jobs = plan_genre_jobs(Genre::Docs, &prompts, &models, &cfg);
        assert_eq!(jobs.len(), 3);
    }

    /// Empty prompts or empty models yields an empty plan rather than
    /// panicking (division by zero guarded).
    #[test]
    fn plan_genre_jobs_handles_empty_prompts_or_models() {
        let cfg = config(&["alpha"], 10, 0.5, 0.5);
        let models: Vec<String> = cfg.models.iter().map(|m| m.name.clone()).collect();
        assert!(plan_genre_jobs(Genre::Docs, &[], &models, &cfg).is_empty());
        assert!(plan_genre_jobs(Genre::Docs, &[prompt("p1")], &[], &cfg).is_empty());
    }

    /// Temperature and style slicing realize the configured
    /// fractions over a full-size (66-job) genre plan.
    #[test]
    #[allow(clippy::float_cmp)] // exact: `temperature` is assigned verbatim from
    // `config.temperature.low`/`.default` in `plan_genre_jobs`, never computed.
    fn plan_genre_jobs_realizes_temperature_and_style_fractions() {
        let cfg = config(&["m1", "m2", "m3", "m4", "m5", "m6"], 66, 0.2, 0.10);
        let prompts: Vec<Prompt> = (1..=11).map(|i| prompt(&format!("p{i:03}"))).collect();
        let models: Vec<String> = cfg.models.iter().map(|m| m.name.clone()).collect();
        let jobs = plan_genre_jobs(Genre::Readme, &prompts, &models, &cfg);

        assert_eq!(jobs.len(), 66);
        let low_temp_count = jobs
            .iter()
            .filter(|j| j.temperature == cfg.temperature.low)
            .count();
        let style_count = jobs.iter().filter(|j| j.style_prompted).count();
        assert_eq!(low_temp_count, 14); // ceil(66/5)
        assert_eq!(style_count, 6); // floor(66/10)
        #[allow(clippy::cast_precision_loss)]
        let low_fraction = low_temp_count as f64 / jobs.len() as f64;
        #[allow(clippy::cast_precision_loss)]
        let style_fraction = style_count as f64 / jobs.len() as f64;
        assert!(low_fraction >= 0.20);
        assert!(style_fraction <= 0.10);
    }

    /// Planning the same genre twice from the same inputs produces
    /// byte-for-byte (field-for-field) identical jobs.
    #[test]
    fn plan_genre_jobs_is_deterministic() {
        let cfg = config(&["a", "b", "c"], 20, 0.2, 0.1);
        let prompts: Vec<Prompt> = (1..=10).map(|i| prompt(&format!("p{i:03}"))).collect();
        let models: Vec<String> = cfg.models.iter().map(|m| m.name.clone()).collect();
        let first = plan_genre_jobs(Genre::Email, &prompts, &models, &cfg);
        let second = plan_genre_jobs(Genre::Email, &prompts, &models, &cfg);
        assert_eq!(first, second);
    }

    /// `build_plan` walks genres in the frozen `ALL_GENRES` order when no
    /// `--genre` filter is given.
    #[test]
    fn build_plan_visits_genres_in_frozen_order() {
        let cfg = config(&["alpha"], 1, 0.5, 0.5);
        let mut by_genre = BTreeMap::new();
        for genre in ALL_GENRES {
            by_genre.insert(genre, vec![prompt("p001")]);
        }
        let plan = build_plan(&cfg, &by_genre, None, None, None).unwrap();
        let genres: Vec<Genre> = plan.iter().map(|j| j.genre).collect();
        assert_eq!(genres, ALL_GENRES.to_vec());
    }

    /// `--model` restricts the plan to a single model and errors clearly
    /// if that model isn't in the matrix.
    #[test]
    fn build_plan_model_filter_restricts_and_validates() {
        let cfg = config(&["alpha", "beta"], 10, 0.5, 0.5);
        let mut by_genre = BTreeMap::new();
        by_genre.insert(Genre::Blog, vec![prompt("p001"), prompt("p002")]);

        let plan = build_plan(&cfg, &by_genre, Some(Genre::Blog), Some("beta"), None).unwrap();
        assert!(plan.iter().all(|j| j.model == "beta"));

        let err = build_plan(&cfg, &by_genre, Some(Genre::Blog), Some("gamma"), None).unwrap_err();
        assert!(err.to_string().contains("gamma"));
    }

    /// `--limit` truncates the deterministic plan to a prefix.
    #[test]
    fn build_plan_limit_truncates_to_prefix() {
        let cfg = config(&["alpha"], 10, 0.5, 0.5);
        let mut by_genre = BTreeMap::new();
        by_genre.insert(
            Genre::Docs,
            (1..=10).map(|i| prompt(&format!("p{i:03}"))).collect(),
        );

        let full = build_plan(&cfg, &by_genre, Some(Genre::Docs), None, None).unwrap();
        let limited = build_plan(&cfg, &by_genre, Some(Genre::Docs), None, Some(3)).unwrap();
        assert_eq!(limited.len(), 3);
        assert_eq!(limited.as_slice(), &full[..3]);
    }

    /// `parse_genre` accepts the frozen set case-insensitively and
    /// rejects anything else with a clear error.
    #[test]
    fn parse_genre_accepts_known_rejects_unknown() {
        assert_eq!(parse_genre("Blog").unwrap(), Genre::Blog);
        assert_eq!(parse_genre("FORUM").unwrap(), Genre::Forum);
        assert!(parse_genre("essay").is_err());
    }

    /// `build_gen_config` records `reproducible: true` when a
    /// same-config regeneration matches the first generation byte-for-byte
    /// — the property the flag is supposed to certify, actually checked
    /// rather than assumed.
    #[test]
    fn build_gen_config_marks_reproducible_true_when_outputs_match() {
        let cfg = build_gen_config(
            "http://localhost:11434",
            "sha256:deadbeef",
            0.7,
            42,
            512,
            "identical output",
            "identical output",
        );
        assert_eq!(cfg["reproducible"], serde_json::json!(true));
    }

    /// This is the regression the original hardcoded-`true` bug
    /// would have masked — when a same-seed regeneration produces
    /// different bytes (as Ollama/llama.cpp backends may under
    /// multi-threaded inference), the flag must honestly record `false`,
    /// never a blanket `true`.
    #[test]
    fn build_gen_config_marks_reproducible_false_when_outputs_differ() {
        let cfg = build_gen_config(
            "http://localhost:11434",
            "sha256:deadbeef",
            0.7,
            42,
            512,
            "first run output",
            "second run output, slightly different",
        );
        assert_eq!(cfg["reproducible"], serde_json::json!(false));
    }

    /// `build_gen_config` also records the rest of the required
    /// fields (model digest, sampler params, seed) verbatim.
    #[test]
    fn build_gen_config_records_full_sampler_config() {
        let cfg = build_gen_config("http://x:11434", "sha256:abc", 0.2, 7, 256, "a", "a");
        assert_eq!(cfg["endpoint"], serde_json::json!("http://x:11434"));
        assert_eq!(cfg["model_digest"], serde_json::json!("sha256:abc"));
        assert_eq!(cfg["temperature"], serde_json::json!(0.2));
        assert_eq!(cfg["seed"], serde_json::json!(7));
        assert_eq!(cfg["num_predict"], serde_json::json!(256));
    }

    /// `wrap_style_prompt` appends the instruction; it never mutates the
    /// base prompt text used for non-style-prompted jobs.
    #[test]
    fn wrap_style_prompt_appends_instruction() {
        let wrapped = wrap_style_prompt("write a thing", "sound human");
        assert!(wrapped.starts_with("write a thing"));
        assert!(wrapped.contains("sound human"));
    }

    /// The `--dry-run` plan rendering is deterministic across repeated
    /// calls on the same plan (golden-test precondition).
    #[test]
    fn render_plan_is_deterministic() {
        let cfg = config(&["alpha", "beta"], 4, 0.5, 0.5);
        let prompts: Vec<Prompt> = (1..=3).map(|i| prompt(&format!("p{i:03}"))).collect();
        let models: Vec<String> = cfg.models.iter().map(|m| m.name.clone()).collect();
        let jobs = plan_genre_jobs(Genre::Blog, &prompts, &models, &cfg);
        assert_eq!(render_plan(&jobs), render_plan(&jobs));
    }

    /// Golden test: the exact rendered plan text for a small, fully
    /// fixed scenario (2 models x 3 prompts, target 4, `base_seed` 0).
    /// Any change to the planning/derivation algorithm should be a
    /// deliberate, reviewed change to this literal string.
    #[test]
    fn render_plan_golden_output_for_fixed_scenario() {
        let cfg = config(&["alpha", "beta"], 4, 0.5, 0.5);
        let prompts: Vec<Prompt> = (1..=3).map(|i| prompt(&format!("p{i:03}"))).collect();
        let models: Vec<String> = cfg.models.iter().map(|m| m.name.clone()).collect();
        let jobs = plan_genre_jobs(Genre::Blog, &prompts, &models, &cfg);
        let rendered = render_plan(&jobs);

        let expected = "\
blog\talpha\tp001\t0.20\tfalse\t688454372\t2f0c9ab3792d6250
blog\tbeta\tp001\t0.70\ttrue\t670050945\te3241b3f81e10bec
blog\talpha\tp002\t0.20\tfalse\t229848140\tccc4e6fb1b2c1f36
blog\tbeta\tp002\t0.70\ttrue\t257725084\t1626baa5faac6793
";
        assert_eq!(rendered, expected);
    }
}
