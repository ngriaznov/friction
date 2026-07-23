//! Integration tests for `corpus-tool generate-paired`.
//!
//! Everything here except the last, env-var-gated test runs fully
//! offline: `--dry-run` never touches the network, and the error-path
//! tests fail before any Ollama call would happen.

mod common;

use corpus_tool::commands::generate_paired::{self, Args};
use corpus_tool::paired_manifest;

fn paired_genconfig_toml(stock: &str, antislop: &str, prompts_per_genre: usize) -> String {
    format!(
        r#"
base_seed = 7
temperature = 0.7
prompts_per_genre = {prompts_per_genre}

[ollama]
endpoint = "http://localhost:11434"
num_predict = 64

[models]
stock = "{stock}"
antislop = "{antislop}"
"#
    )
}

fn write_paired_genconfig(
    dir: &std::path::Path,
    stock: &str,
    antislop: &str,
    prompts_per_genre: usize,
) -> std::path::PathBuf {
    let path = dir.join("genconfig-paired.toml");
    std::fs::write(
        &path,
        paired_genconfig_toml(stock, antislop, prompts_per_genre),
    )
    .unwrap();
    path
}

fn write_prompts(prompts_dir: &std::path::Path, genre: &str, ids: &[&str]) {
    use std::fmt::Write as _;

    std::fs::create_dir_all(prompts_dir).unwrap();
    let mut body = String::new();
    for id in ids {
        writeln!(body, "[[prompts]]\nid = \"{id}\"").unwrap();
        writeln!(body, "text = \"Write two sentences about {id}.\"").unwrap();
        writeln!(body, "topic = \"t\"\n").unwrap();
    }
    std::fs::write(prompts_dir.join(format!("{genre}.toml")), body).unwrap();
}

fn args(dir: &std::path::Path, genconfig: &std::path::Path, dry_run: bool) -> Args {
    Args {
        paired_dir: dir.join("paired"),
        genconfig: genconfig.to_path_buf(),
        prompts_dir: dir.join("prompts"),
        dry_run,
        limit: None,
        genre: None,
        side: None,
    }
}

/// A missing prompt file for a needed genre is a clear error (not a
/// panic), and mentions the genre.
#[test]
fn generate_paired_missing_prompt_file_is_clear_error() {
    let dir = tempfile::tempdir().unwrap();
    let genconfig = write_paired_genconfig(dir.path(), "stock-m", "antislop-m", 2);
    // No corpus/prompts/*.toml written at all.

    let err = generate_paired::run(&args(dir.path(), &genconfig, true)).unwrap_err();
    let msg = format!("{err:#}");
    assert!(msg.contains("docs"), "message was: {msg}");
}

/// `--dry-run` never touches the paired corpus: no manifest, no
/// `stock`/`antislop` dirs, and it reports exactly `5 genres *
/// prompts_per_genre * 2 sides` planned jobs without generating
/// anything.
#[test]
fn generate_paired_dry_run_does_not_touch_paired_dir_or_call_ollama() {
    let dir = tempfile::tempdir().unwrap();
    let genconfig = write_paired_genconfig(dir.path(), "stock-m", "antislop-m", 3);
    let prompts_dir = dir.path().join("prompts");
    for genre in ["docs", "blog", "readme", "email", "forum"] {
        write_prompts(&prompts_dir, genre, &["p001", "p002", "p003"]);
    }

    let outcome = generate_paired::run(&args(dir.path(), &genconfig, true)).unwrap();

    assert_eq!(outcome.planned, 5 * 3 * 2);
    assert_eq!(outcome.generated, 0);
    assert_eq!(outcome.skipped_existing, 0);
    assert!(outcome.skipped_models.is_empty());
    assert!(!dir.path().join("paired/manifest.jsonl").exists());
    assert!(!dir.path().join("paired/stock").exists());
    assert!(!dir.path().join("paired/antislop").exists());
}

/// `--dry-run` plans are deterministic: running it twice on identical
/// inputs produces field-for-field identical outcomes.
#[test]
fn generate_paired_dry_run_is_deterministic() {
    let dir = tempfile::tempdir().unwrap();
    let genconfig = write_paired_genconfig(dir.path(), "stock-m", "antislop-m", 2);
    let prompts_dir = dir.path().join("prompts");
    for genre in ["docs", "blog", "readme", "email", "forum"] {
        write_prompts(&prompts_dir, genre, &["p001", "p002"]);
    }

    let first = generate_paired::run(&args(dir.path(), &genconfig, true)).unwrap();
    let second = generate_paired::run(&args(dir.path(), &genconfig, true)).unwrap();
    assert_eq!(first, second);
}

/// `--genre` and `--side` narrow the plan as expected, and unknown
/// values for either are clear errors.
#[test]
fn generate_paired_genre_and_side_filters_bound_the_plan() {
    let dir = tempfile::tempdir().unwrap();
    let genconfig = write_paired_genconfig(dir.path(), "stock-m", "antislop-m", 2);
    let prompts_dir = dir.path().join("prompts");
    write_prompts(&prompts_dir, "blog", &["p001", "p002", "p003"]);

    let mut a = args(dir.path(), &genconfig, true);
    a.genre = Some("blog".to_string());
    a.side = Some("stock".to_string());
    let outcome = generate_paired::run(&a).unwrap();
    assert_eq!(outcome.planned, 2); // 2 prompts x 1 side

    let mut bad_genre = args(dir.path(), &genconfig, true);
    bad_genre.genre = Some("essay".to_string());
    let err = generate_paired::run(&bad_genre).unwrap_err();
    assert!(err.to_string().contains("essay"));

    let mut bad_side = args(dir.path(), &genconfig, true);
    bad_side.genre = Some("blog".to_string());
    bad_side.side = Some("nope".to_string());
    let err = generate_paired::run(&bad_side).unwrap_err();
    assert!(err.to_string().contains("nope"));
}

/// A genre whose prompt catalog is shorter than `prompts_per_genre`
/// requires is a hard, clear error — never a silent truncation, since
/// the paired corpus's size guarantee depends on every genre
/// contributing exactly N prompts.
#[test]
fn generate_paired_errors_when_prompt_catalog_shorter_than_prompts_per_genre() {
    let dir = tempfile::tempdir().unwrap();
    let genconfig = write_paired_genconfig(dir.path(), "stock-m", "antislop-m", 5);
    let prompts_dir = dir.path().join("prompts");
    write_prompts(&prompts_dir, "blog", &["p001", "p002"]);

    let mut a = args(dir.path(), &genconfig, true);
    a.genre = Some("blog".to_string());
    let err = generate_paired::run(&a).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("blog"), "message was: {msg}");
    assert!(msg.contains("prompts_per_genre"), "message was: {msg}");
}

/// Live-Ollama smoke test: gated behind `FRICTION_OLLAMA_TEST=1` so it
/// never runs in an environment without a local Ollama server (CI, other
/// agents' sandboxes). Requires `gemma3:12b` and the antislop-tuned
/// counterpart pulled locally.
///
/// Exercises the full path: HTTP calls to `/api/tags`, `/api/show`,
/// `/api/generate` for both sides of one pair; both docs written under
/// `corpus/paired/`, sharing one `pair_id`; manifest rows for both
/// sides; and rerun-incrementality (a second run generates nothing new).
#[test]
fn generate_paired_live_ollama_smoke_test() {
    if std::env::var("FRICTION_OLLAMA_TEST").as_deref() != Ok("1") {
        eprintln!("skipping: set FRICTION_OLLAMA_TEST=1 to run against a local Ollama server");
        return;
    }

    let dir = tempfile::tempdir().unwrap();
    let genconfig = write_paired_genconfig(
        dir.path(),
        "gemma3:12b",
        "hf.co/mradermacher/gemma-3-12b-it-antislop-GGUF:Q4_K_M",
        1,
    );
    let prompts_dir = dir.path().join("prompts");
    write_prompts(&prompts_dir, "docs", &["p001"]);

    let mut a = args(dir.path(), &genconfig, false);
    a.genre = Some("docs".to_string());

    let first = generate_paired::run(&a).unwrap();
    assert_eq!(first.generated, 2);
    assert!(first.skipped_models.is_empty());

    let records = paired_manifest::read_paired_manifest(&dir.path().join("paired/manifest.jsonl"))
        .unwrap()
        .unwrap();
    assert_eq!(records.len(), 2);
    let pair_id = records[0].pair_id.clone();
    assert!(records.iter().all(|r| r.pair_id == pair_id));

    let stock_path = dir.path().join(format!("paired/stock/docs/{pair_id}.md"));
    let antislop_path = dir
        .path()
        .join(format!("paired/antislop/docs/{pair_id}.md"));
    assert!(stock_path.exists());
    assert!(antislop_path.exists());

    // Rerunning with the same config generates nothing new.
    let second = generate_paired::run(&a).unwrap();
    assert_eq!(second.generated, 0);
    assert_eq!(second.skipped_existing, 2);
}
