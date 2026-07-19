//! Integration tests for `corpus-tool seal`.

mod common;

use corpus_tool::commands::seal::{self, Args};
use corpus_tool::manifest::{Genre, Split};

fn args(corpus_dir: &std::path::Path, init: bool) -> Args {
    Args {
        corpus_dir: corpus_dir.to_path_buf(),
        init,
    }
}

fn holdout_record(dir: &std::path::Path, id: &str) -> corpus_tool::manifest::ManifestRecord {
    let words = common::filler_words(320);
    let sha = common::write_doc(dir, &format!("human/docs/{id}.md"), &words);
    let mut record = common::human_record(id, Genre::Docs, sha);
    record.split = Some(Split::Holdout);
    record
}

/// An empty corpus doesn't error.
#[test]
fn seal_empty_corpus_is_ok() {
    let dir = tempfile::tempdir().unwrap();
    assert!(seal::run(&args(dir.path(), false)).is_ok());
}

/// `seal` writes `holdout.lock` with one tab-separated
/// `id\tsha256\trelpath` line per holdout doc, sorted by id.
#[test]
fn seal_writes_lock_sorted_by_id() {
    let dir = tempfile::tempdir().unwrap();
    let records = vec![
        holdout_record(dir.path(), "zzz"),
        holdout_record(dir.path(), "aaa"),
    ];
    common::write_manifest_raw(&dir.path().join("manifest.jsonl"), &records);

    seal::run(&args(dir.path(), false)).unwrap();

    let lock = std::fs::read_to_string(dir.path().join("holdout.lock")).unwrap();
    let lines: Vec<&str> = lock.lines().collect();
    assert_eq!(lines.len(), 2);
    assert!(lines[0].starts_with("aaa\t"));
    assert!(lines[1].starts_with("zzz\t"));
    for line in &lines {
        assert_eq!(line.split('\t').count(), 3);
    }
}

/// Sealing twice with no changes is a no-op success.
#[test]
fn seal_rerun_with_no_changes_is_noop() {
    let dir = tempfile::tempdir().unwrap();
    let records = vec![holdout_record(dir.path(), "h001")];
    common::write_manifest_raw(&dir.path().join("manifest.jsonl"), &records);

    seal::run(&args(dir.path(), false)).unwrap();
    let first = std::fs::read_to_string(dir.path().join("holdout.lock")).unwrap();

    assert!(seal::run(&args(dir.path(), false)).is_ok());
    let second = std::fs::read_to_string(dir.path().join("holdout.lock")).unwrap();
    assert_eq!(first, second);
}

/// If the lock exists and its content would change, `seal`
/// refuses without `--init`.
#[test]
fn seal_refuses_content_change_without_init() {
    let dir = tempfile::tempdir().unwrap();
    let manifest_path = dir.path().join("manifest.jsonl");
    let records = vec![holdout_record(dir.path(), "h001")];
    common::write_manifest_raw(&manifest_path, &records);
    seal::run(&args(dir.path(), false)).unwrap();

    // Add a second holdout doc: the lock content would now differ.
    let mut records = records;
    records.push(holdout_record(dir.path(), "h002"));
    common::write_manifest_raw(&manifest_path, &records);

    let err = seal::run(&args(dir.path(), false)).unwrap_err();
    assert!(err.to_string().contains("--init"));
}

/// `seal` recomputes each holdout doc's sha256 from its actual
/// on-disk bytes rather than trusting the manifest's recorded sha256 — if
/// a doc was hand-edited after its manifest entry was last written (a
/// stale manifest), `seal` refuses rather than silently freezing the
/// wrong hash into `holdout.lock`.
#[test]
fn seal_rejects_stale_manifest_sha256() {
    let dir = tempfile::tempdir().unwrap();
    let mut record = holdout_record(dir.path(), "h001");
    // Corrupt the manifest's recorded sha256 so it no longer matches the
    // file actually on disk, simulating a doc edited without re-running
    // `validate`.
    record.sha256 = "0".repeat(64);
    common::write_manifest_raw(&dir.path().join("manifest.jsonl"), &[record]);

    let err = seal::run(&args(dir.path(), false)).unwrap_err();
    let message = err.to_string();
    assert!(message.contains("h001"), "message: {message}");
    assert!(
        message.contains("stale") || message.contains("validate"),
        "message: {message}"
    );
    assert!(
        !dir.path().join("holdout.lock").exists(),
        "seal must not write a lock file when a hash mismatch is detected"
    );
}

/// With `--init`, `seal` overwrites a lock whose content would
/// change.
#[test]
fn seal_init_overwrites_changed_lock() {
    let dir = tempfile::tempdir().unwrap();
    let manifest_path = dir.path().join("manifest.jsonl");
    let records = vec![holdout_record(dir.path(), "h001")];
    common::write_manifest_raw(&manifest_path, &records);
    seal::run(&args(dir.path(), false)).unwrap();

    let mut records = records;
    records.push(holdout_record(dir.path(), "h002"));
    common::write_manifest_raw(&manifest_path, &records);

    assert!(seal::run(&args(dir.path(), true)).is_ok());
    let lock = std::fs::read_to_string(dir.path().join("holdout.lock")).unwrap();
    assert_eq!(lock.lines().count(), 2);
}
