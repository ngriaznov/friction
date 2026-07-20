//! Corpus-wide idempotence sweep: `fix_document(fix_document(x)) ==
//! fix_document(x)`, byte-for-byte, over every corpus document.
//!
//! Two variants, mirroring `friction-nlp`'s own `corpus_determinism`
//! test:
//!
//! - A fast, always-on smoke check over the first 30 corpus documents
//!   (`idempotence_sweep_smoke`), run as part of the normal test suite.
//! - The full corpus check (`idempotence_sweep_full`), `#[ignore]`d by
//!   default since it runs the fixpoint driver (segmenter + tagger +
//!   every tranche-1 rule) twice over every document in the corpus
//!   fixture. Run it explicitly with:
//!
//!   ```text
//!   cargo test -p friction-apply --test idempotence_sweep -- --ignored --nocapture
//!   ```
//!
//!   which prints a running count and, on any failure, the offending
//!   doc's path plus the byte offset of the first output divergence.
//!
//! # Genre
//!
//! `friction_apply::FixEngine::fix_document` needs a genre to look up
//! envelope bands. Every corpus doc's directory layout already encodes it
//! (`<class-or-quarantine>/<genre>/<id>.md` — see `corpus-tool`'s own
//! `corpus_layout` module), so this test reads it straight from the
//! path's parent directory name rather than depending on the manifest at
//! all.

use std::fs;
use std::path::{Path, PathBuf};

use friction_apply::FixEngine;

/// Root of the corpus fixture directory, resolved relative to this
/// crate's manifest directory so the test works regardless of the
/// process's current working directory.
fn corpus_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus")
}

/// Appends every `.md` file found (recursively) under `dir` to `out`, one
/// directory level at a time, sorting each level's entries by name before
/// descending — never relies on `std::fs::read_dir`'s unspecified order.
fn collect_md_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    let mut children: Vec<PathBuf> = entries
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .collect();
    children.sort();
    for path in children {
        if path.is_dir() {
            collect_md_files(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "md") {
            out.push(path);
        }
    }
}

/// Every corpus document this sweep runs over: every `.md` file under
/// `corpus/human`, `corpus/llm`, and `corpus/quarantine` (712 files as of
/// this writing — same set `friction-nlp`'s `corpus_determinism_full`
/// covers), sorted by path for a deterministic, machine-independent
/// order. `corpus/incoming` is excluded (a staging area, see
/// `friction-nlp`'s own test for the full rationale).
fn corpus_docs() -> Vec<PathBuf> {
    let root = corpus_root();
    let mut docs = Vec::new();
    for subdir in ["human", "llm", "quarantine"] {
        collect_md_files(&root.join(subdir), &mut docs);
    }
    docs.sort();
    docs
}

/// This document's genre, read from its parent directory name
/// (`<class-or-quarantine>/<genre>/<id>.md`).
///
/// # Panics
/// Panics if `path` doesn't have a parent directory with a valid UTF-8
/// name — every path `corpus_docs` returns does, by construction.
fn genre_of(path: &Path) -> &str {
    path.parent()
        .and_then(|p| p.file_name())
        .and_then(|n| n.to_str())
        .expect("every corpus doc path has a UTF-8 genre directory component")
}

/// Runs the fixpoint driver twice over `source` and returns `(once,
/// twice)`.
///
/// # Panics
/// Panics if `source` fails to parse/segment either time — a malformed
/// corpus fixture is a fixture bug this sweep should fail loudly on, not
/// silently skip.
fn fix_twice(engine: &FixEngine, source: &str, genre: &str) -> (String, String) {
    let (once, _) = engine
        .fix_document(source, genre)
        .expect("corpus fixture must parse and segment");
    let (twice, _) = engine
        .fix_document(&once, genre)
        .expect("fix_document's own output must parse and segment");
    (once, twice)
}

/// The byte offset of the first point where `a` and `b` diverge, or
/// `None` if they're identical — used only to make an idempotence
/// failure's assertion message point at *where*, not just *that*, output
/// changed on a second pass.
fn first_divergence(a: &str, b: &str) -> Option<usize> {
    a.bytes()
        .zip(b.bytes())
        .position(|(x, y)| x != y)
        .or_else(|| (a.len() != b.len()).then_some(a.len().min(b.len())))
}

/// Asserts `fix(fix(source)) == fix(source)`, byte-for-byte, with a
/// message naming `path` and the first divergent byte offset on failure.
fn assert_idempotent(engine: &FixEngine, path: &Path, source: &str, genre: &str) {
    let (once, twice) = fix_twice(engine, source, genre);
    if once != twice {
        let at = first_divergence(&once, &twice).unwrap_or(0);
        panic!(
            "{}: fix(fix(x)) != fix(x) — first divergence at byte {at}\n  fix(x)      = {:?}\n  fix(fix(x)) = {:?}",
            path.display(),
            &once[at.min(once.len())..(at + 60).min(once.len())],
            &twice[at.min(twice.len())..(at + 60).min(twice.len())],
        );
    }
}

/// Fast smoke check: the first 30 corpus documents (by sorted path), run
/// as part of the normal `cargo test --workspace`.
#[test]
fn idempotence_sweep_smoke() {
    let engine = FixEngine::new().expect("embedded tagger model must load");
    let docs = corpus_docs();
    assert!(
        docs.len() >= 30,
        "expected at least 30 corpus docs, found {}",
        docs.len()
    );
    for path in docs.into_iter().take(30) {
        let source = fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("{}: failed to read: {e}", path.display()));
        let genre = genre_of(&path);
        assert_idempotent(&engine, &path, &source, genre);
    }
}

/// Full corpus check: every document under `corpus/human`, `corpus/llm`,
/// and `corpus/quarantine`. `#[ignore]`d by default — see this module's
/// docs for the explicit run command.
#[test]
#[ignore = "runs the fixpoint driver twice over the full corpus; see module docs for the explicit invocation"]
fn idempotence_sweep_full() {
    let engine = FixEngine::new().expect("embedded tagger model must load");
    let docs = corpus_docs();
    println!("idempotence sweep: {} documents", docs.len());
    for (i, path) in docs.iter().enumerate() {
        let source = fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("{}: failed to read: {e}", path.display()));
        let genre = genre_of(path);
        assert_idempotent(&engine, path, &source, genre);
        if (i + 1) % 100 == 0 || i + 1 == docs.len() {
            println!("  ...{}/{} idempotent", i + 1, docs.len());
        }
    }
}
