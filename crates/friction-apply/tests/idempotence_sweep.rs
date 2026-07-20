//! Corpus-wide idempotence sweep: `fix_document(fix_document(x)) ==
//! fix_document(x)`, byte-for-byte, over every corpus document.
//!
//! Three variants, mirroring `friction-nlp`'s own `corpus_determinism`
//! test:
//!
//! - A fast, always-on smoke check over the first 30 corpus documents
//!   (`idempotence_sweep_smoke`), run as part of the normal test suite.
//! - The full corpus check (`idempotence_sweep_full`), `#[ignore]`d by
//!   default since it runs the fixpoint driver (segmenter + tagger +
//!   every registered rule, across all six families) twice over every
//!   document in the corpus fixture. Run it explicitly with:
//!
//!   ```text
//!   cargo test -p friction-apply --test idempotence_sweep -- --ignored --nocapture
//!   ```
//!
//!   which prints a running count and, on any failure, the offending
//!   doc's path plus the byte offset of the first output divergence.
//! - The sealed-holdout-only check (`idempotence_sweep_holdout`), also
//!   `#[ignore]`d by default, filtering to exactly the manifest's
//!   `split: holdout` documents (a subset `idempotence_sweep_full`
//!   already covers as part of the whole corpus — this variant exists so
//!   the holdout evaluation has its own explicit, holdout-scoped
//!   idempotence result to report, rather than inferring it from the
//!   full sweep's superset). Run it explicitly with:
//!
//!   ```text
//!   cargo test -p friction-apply --test idempotence_sweep -- --ignored idempotence_sweep_holdout --nocapture
//!   ```
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

/// Every `split: holdout` manifest document's path and genre (110 as of
/// this writing: 60 human, 50 llm), sorted by manifest id for a
/// deterministic order. Reads genre from the manifest record directly
/// (not the parent directory name, unlike [`genre_of`]) since this
/// variant needs the manifest anyway to find the holdout split at all.
///
/// # Panics
/// Panics if the manifest doesn't exist or fails to parse — a missing or
/// malformed manifest is a fixture bug this sweep should fail loudly on.
fn holdout_docs() -> Vec<(PathBuf, String)> {
    use corpus_tool::corpus_layout::relpath;
    use corpus_tool::manifest::{Split, read_manifest};

    let root = corpus_root();
    let manifest_path = root.join("manifest.jsonl");
    let mut records = read_manifest(&manifest_path)
        .expect("manifest.jsonl must read")
        .expect("manifest.jsonl must exist")
        .into_iter()
        .filter(|r| r.split == Some(Split::Holdout))
        .collect::<Vec<_>>();
    records.sort_by(|a, b| a.id.cmp(&b.id));
    records
        .iter()
        .map(|r| (root.join(relpath(r)), r.genre.to_string()))
        .collect()
}

/// Sealed-holdout-only check: every `split: holdout` manifest document —
/// a subset [`idempotence_sweep_full`] already covers as part of the
/// whole corpus, kept as its own variant so the holdout evaluation has an
/// explicit, holdout-scoped idempotence result to report rather than one
/// inferred from the full sweep's superset. `#[ignore]`d by default — see
/// module docs for the explicit run command.
#[test]
#[ignore = "runs the fixpoint driver twice over every holdout document; see module docs for the explicit invocation"]
fn idempotence_sweep_holdout() {
    let engine = FixEngine::new().expect("embedded tagger model must load");
    let docs = holdout_docs();
    println!("idempotence sweep (holdout): {} documents", docs.len());
    for (i, (path, genre)) in docs.iter().enumerate() {
        let source = fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("{}: failed to read: {e}", path.display()));
        assert_idempotent(&engine, path, &source, genre);
        if (i + 1) % 25 == 0 || i + 1 == docs.len() {
            println!("  ...{}/{} idempotent", i + 1, docs.len());
        }
    }
}
