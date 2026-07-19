//! Integration tests for `corpus-tool holdout-check`.
//!
//! `holdout-check` resolves lock `relpath` entries relative to the
//! current working directory (matching `scripts/check-holdout.sh`), so
//! these tests `chdir` into the temp corpus's parent for the duration of
//! each check. Tests run single-threaded within this file to avoid CWD
//! races between tests (`cargo test` still runs other test *binaries* in
//! parallel, which is fine since CWD is process-global, not
//! binary-shared).

mod common;

use std::sync::Mutex;

use corpus_tool::commands::holdout_check::{self, Args as CheckArgs};
use corpus_tool::commands::seal::{self, Args as SealArgs};
use corpus_tool::manifest::{self, Genre, Split};

// Serializes tests within this binary since they all mutate the
// process-global current directory.
static CWD_LOCK: Mutex<()> = Mutex::new(());

fn with_cwd<R>(dir: &std::path::Path, f: impl FnOnce() -> R) -> R {
    let _guard = CWD_LOCK.lock().unwrap();
    let original = std::env::current_dir().unwrap();
    std::env::set_current_dir(dir).unwrap();
    let result = f();
    std::env::set_current_dir(original).unwrap();
    result
}

fn seal_args(init: bool) -> SealArgs {
    SealArgs {
        corpus_dir: "corpus".into(),
        init,
    }
}

fn check_args() -> CheckArgs {
    CheckArgs {
        corpus_dir: "corpus".into(),
    }
}

fn build_holdout_corpus(root: &std::path::Path) {
    let corpus_dir = root.join("corpus");
    let words = common::filler_words(320);
    let sha = common::write_doc(&corpus_dir, "human/docs/h001.md", &words);
    let mut record = common::human_record("h001", Genre::Docs, sha);
    record.split = Some(Split::Holdout);
    common::write_manifest_raw(
        &corpus_dir.join("manifest.jsonl"),
        std::slice::from_ref(&record),
    );
}

/// An absent lock file is a no-op success (matches
/// `scripts/check-holdout.sh`).
#[test]
fn holdout_check_absent_lock_is_noop_success() {
    let dir = tempfile::tempdir().unwrap();
    build_holdout_corpus(dir.path());
    // Note: no seal() call, so no lock file exists yet.
    let result = with_cwd(dir.path(), || holdout_check::run(&check_args()));
    assert!(result.is_ok());
}

/// A freshly sealed corpus passes `holdout-check`.
#[test]
fn holdout_check_passes_on_freshly_sealed_corpus() {
    let dir = tempfile::tempdir().unwrap();
    build_holdout_corpus(dir.path());

    let result = with_cwd(dir.path(), || {
        seal::run(&seal_args(false)).unwrap();
        holdout_check::run(&check_args())
    });
    assert!(result.is_ok());
}

/// Mutating a sealed doc's file content after sealing (without
/// touching the lock or manifest) is detected as drift.
#[test]
fn holdout_check_fails_on_file_content_drift() {
    let dir = tempfile::tempdir().unwrap();
    build_holdout_corpus(dir.path());

    let result = with_cwd(dir.path(), || {
        seal::run(&seal_args(false)).unwrap();
        std::fs::write(
            dir.path().join("corpus/human/docs/h001.md"),
            "tampered content",
        )
        .unwrap();
        holdout_check::run(&check_args())
    });
    assert!(result.is_err());
}

/// Unsealing a doc in the manifest (flipping `split` away from
/// `holdout`) after sealing is detected as drift.
#[test]
fn holdout_check_fails_on_manifest_unseal_drift() {
    let dir = tempfile::tempdir().unwrap();
    build_holdout_corpus(dir.path());

    let result = with_cwd(dir.path(), || {
        seal::run(&seal_args(false)).unwrap();

        let manifest_path = dir.path().join("corpus/manifest.jsonl");
        let mut records = manifest::read_manifest(&manifest_path).unwrap().unwrap();
        records[0].split = Some(Split::Train);
        manifest::write_manifest(&manifest_path, &records).unwrap();

        holdout_check::run(&check_args())
    });
    assert!(result.is_err());
}

/// `holdout-check` rejects a malformed lock line (wrong field count).
#[test]
fn holdout_check_fails_on_malformed_lock_line() {
    let dir = tempfile::tempdir().unwrap();
    build_holdout_corpus(dir.path());

    let result = with_cwd(dir.path(), || {
        std::fs::write(dir.path().join("corpus/holdout.lock"), "only-one-field\n").unwrap();
        holdout_check::run(&check_args())
    });
    assert!(result.is_err());
}
