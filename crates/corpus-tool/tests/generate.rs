//! Integration tests for `corpus-tool generate`.
//!
//! Everything here except the last, env-var-gated test runs fully
//! offline: `--dry-run` never touches the network, and the error-path
//! tests fail before any Ollama call would happen.

mod common;

use corpus_tool::commands::generate::{self, Args};
use corpus_tool::manifest::{self, Class};

fn genconfig_toml(models: &[&str]) -> String {
    use std::fmt::Write as _;

    let mut model_tables = String::new();
    for name in models {
        writeln!(model_tables, "[[models]]\nname = \"{name}\"\n").unwrap();
    }
    format!(
        r#"
base_seed = 7

[ollama]
endpoint = "http://localhost:11434"
num_predict = 64

{model_tables}
[temperature]
default = 0.7
low = 0.2
low_fraction = 0.2

[style_prompted]
fraction = 0.1
instruction = "sound human, not AI-generated"

[targets]
docs_per_genre = 6
"#
    )
}

fn write_genconfig(dir: &std::path::Path, models: &[&str]) -> std::path::PathBuf {
    let path = dir.join("genconfig.toml");
    std::fs::write(&path, genconfig_toml(models)).unwrap();
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
        corpus_dir: dir.join("corpus"),
        genconfig: genconfig.to_path_buf(),
        prompts_dir: dir.join("prompts"),
        dry_run,
        limit: None,
        model: None,
        genre: None,
    }
}

/// A missing prompt file for a needed genre is a clear error
/// (not a panic), and mentions the genre.
#[test]
fn generate_missing_prompt_file_is_clear_error() {
    let dir = tempfile::tempdir().unwrap();
    let genconfig = write_genconfig(dir.path(), &["granite4.1:3b"]);
    // No corpus/prompts/*.toml written at all.

    let err = generate::run(&args(dir.path(), &genconfig, true)).unwrap_err();
    let msg = format!("{err:#}");
    assert!(msg.contains("docs"), "message was: {msg}");
}

/// `--dry-run` never touches the corpus: no manifest, no `llm/` docs, and
/// it reports the planned count without generating anything.
#[test]
fn generate_dry_run_does_not_touch_corpus_or_call_ollama() {
    let dir = tempfile::tempdir().unwrap();
    let genconfig = write_genconfig(dir.path(), &["m1", "m2"]);
    let prompts_dir = dir.path().join("prompts");
    for genre in ["docs", "blog", "readme", "email", "forum"] {
        write_prompts(&prompts_dir, genre, &["p001", "p002", "p003"]);
    }

    let outcome = generate::run(&args(dir.path(), &genconfig, true)).unwrap();

    assert!(outcome.planned > 0);
    assert_eq!(outcome.generated, 0);
    assert_eq!(outcome.skipped_existing, 0);
    assert!(outcome.skipped_models.is_empty());
    assert!(!dir.path().join("corpus/manifest.jsonl").exists());
    assert!(!dir.path().join("corpus/llm").exists());
}

/// `--dry-run` plans are deterministic: running it twice on identical
/// inputs plans identically.
#[test]
fn generate_dry_run_plan_count_is_deterministic_across_runs() {
    let dir = tempfile::tempdir().unwrap();
    let genconfig = write_genconfig(dir.path(), &["m1", "m2", "m3"]);
    let prompts_dir = dir.path().join("prompts");
    for genre in ["docs", "blog", "readme", "email", "forum"] {
        write_prompts(&prompts_dir, genre, &["p001", "p002"]);
    }

    let first = generate::run(&args(dir.path(), &genconfig, true)).unwrap();
    let second = generate::run(&args(dir.path(), &genconfig, true)).unwrap();
    assert_eq!(first, second);
}

/// `--model` + `--limit` narrow the plan as expected, and an unknown
/// `--genre` value is a clear error.
#[test]
fn generate_model_and_limit_filters_bound_the_plan() {
    let dir = tempfile::tempdir().unwrap();
    let genconfig = write_genconfig(dir.path(), &["m1", "m2"]);
    let prompts_dir = dir.path().join("prompts");
    write_prompts(&prompts_dir, "blog", &["p001", "p002", "p003", "p004"]);

    let mut a = args(dir.path(), &genconfig, true);
    a.genre = Some("blog".to_string());
    a.model = Some("m2".to_string());
    a.limit = Some(2);
    let outcome = generate::run(&a).unwrap();
    assert_eq!(outcome.planned, 2);

    let mut bad_genre = args(dir.path(), &genconfig, true);
    bad_genre.genre = Some("essay".to_string());
    let err = generate::run(&bad_genre).unwrap_err();
    assert!(err.to_string().contains("essay"));

    let mut bad_model = args(dir.path(), &genconfig, true);
    bad_model.genre = Some("blog".to_string());
    bad_model.model = Some("nope".to_string());
    let err = generate::run(&bad_model).unwrap_err();
    assert!(err.to_string().contains("nope"));
}

/// Live-Ollama smoke test: gated behind `FRICTION_OLLAMA_TEST=1` so it
/// never runs in an environment without a local Ollama server (CI, other
/// agents' sandboxes). Requires `granite4.1:3b` pulled locally.
///
/// Exercises the full path: HTTP calls to `/api/tags`, `/api/show`,
/// `/api/generate`; a doc written under `corpus/llm/`; a manifest record
/// with `gen_config`; and rerun-incrementality (a second
/// run with the same config generates nothing new).
#[test]
fn generate_live_ollama_smoke_test() {
    if std::env::var("FRICTION_OLLAMA_TEST").as_deref() != Ok("1") {
        eprintln!("skipping: set FRICTION_OLLAMA_TEST=1 to run against a local Ollama server");
        return;
    }

    let dir = tempfile::tempdir().unwrap();
    let genconfig = write_genconfig(dir.path(), &["granite4.1:3b"]);
    let prompts_dir = dir.path().join("prompts");
    write_prompts(&prompts_dir, "docs", &["p001"]);

    let mut a = args(dir.path(), &genconfig, false);
    a.genre = Some("docs".to_string());
    a.limit = Some(1);

    let first = generate::run(&a).unwrap();
    assert_eq!(first.generated, 1);
    assert!(first.skipped_models.is_empty());

    let records = manifest::read_manifest(&dir.path().join("corpus/manifest.jsonl"))
        .unwrap()
        .unwrap();
    assert_eq!(records.len(), 1);
    let record = &records[0];
    assert_eq!(record.class, Class::Llm);
    assert!(record.gen_config.is_some());
    let doc_path = dir.path().join(format!("corpus/llm/docs/{}.md", record.id));
    assert!(doc_path.exists());
    assert!(
        !std::fs::read_to_string(&doc_path)
            .unwrap()
            .trim()
            .is_empty()
    );

    // Rerunning with the same config generates nothing new.
    let second = generate::run(&a).unwrap();
    assert_eq!(second.generated, 0);
    assert_eq!(second.skipped_existing, 1);
}
