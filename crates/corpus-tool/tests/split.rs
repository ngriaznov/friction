//! Integration tests for `corpus-tool split`.

mod common;

use corpus_tool::commands::split::{self, Args};
use corpus_tool::hashing::sha256_hex;
use corpus_tool::manifest::{self, Genre, Split};

fn args(corpus_dir: &std::path::Path, dry_run: bool) -> Args {
    Args {
        corpus_dir: corpus_dir.to_path_buf(),
        dry_run,
    }
}

fn build_cell(dir: &std::path::Path, n: usize) -> Vec<corpus_tool::manifest::ManifestRecord> {
    let words = common::filler_words(320);
    (0..n)
        .map(|i| {
            let id = format!("h{i:03}");
            let sha = common::write_doc(dir, &format!("human/docs/{id}.md"), &words);
            common::human_record(&id, Genre::Docs, sha)
        })
        .collect()
}

/// An empty corpus doesn't error.
#[test]
fn split_empty_corpus_is_ok() {
    let dir = tempfile::tempdir().unwrap();
    assert!(split::run(&args(dir.path(), false)).is_ok());
}

/// A cell of 20 docs splits 70/15/15 -> 14/3/3.
#[test]
fn split_is_70_15_15_stratified_by_cell() {
    let dir = tempfile::tempdir().unwrap();
    let records = build_cell(dir.path(), 20);
    let manifest_path = dir.path().join("manifest.jsonl");
    common::write_manifest_raw(&manifest_path, &records);

    split::run(&args(dir.path(), false)).unwrap();

    let after = manifest::read_manifest(&manifest_path).unwrap().unwrap();
    let train = after
        .iter()
        .filter(|r| r.split == Some(Split::Train))
        .count();
    let dev = after.iter().filter(|r| r.split == Some(Split::Dev)).count();
    let holdout = after
        .iter()
        .filter(|r| r.split == Some(Split::Holdout))
        .count();

    assert_eq!((train, dev, holdout), (14, 3, 3));
    assert_eq!(train + dev + holdout, 20);
}

/// Candidates are ordered by `sha256(id)` hex, not by manifest
/// order or id lexicographic order — the holdout slice is exactly the
/// docs with the lexicographically greatest `sha256(id)` hex values.
#[test]
fn split_orders_candidates_by_sha256_of_id() {
    let dir = tempfile::tempdir().unwrap();
    let records = build_cell(dir.path(), 20);
    let manifest_path = dir.path().join("manifest.jsonl");
    common::write_manifest_raw(&manifest_path, &records);

    split::run(&args(dir.path(), false)).unwrap();

    let after = manifest::read_manifest(&manifest_path).unwrap().unwrap();

    let mut by_hash: Vec<(String, String)> = after
        .iter()
        .map(|r| (sha256_hex(r.id.as_bytes()), r.id.clone()))
        .collect();
    by_hash.sort();
    let expected_holdout_ids: std::collections::BTreeSet<&str> =
        by_hash[17..].iter().map(|(_, id)| id.as_str()).collect();

    let actual_holdout_ids: std::collections::BTreeSet<&str> = after
        .iter()
        .filter(|r| r.split == Some(Split::Holdout))
        .map(|r| r.id.as_str())
        .collect();

    assert_eq!(actual_holdout_ids, expected_holdout_ids);
}

/// Running `split` twice on the same manifest produces the same
/// assignment both times (deterministic, idempotent).
#[test]
fn split_is_deterministic_across_runs() {
    let dir = tempfile::tempdir().unwrap();
    let records = build_cell(dir.path(), 20);
    let manifest_path = dir.path().join("manifest.jsonl");
    common::write_manifest_raw(&manifest_path, &records);

    split::run(&args(dir.path(), false)).unwrap();
    let first = manifest::read_manifest(&manifest_path).unwrap().unwrap();

    split::run(&args(dir.path(), false)).unwrap();
    let second = manifest::read_manifest(&manifest_path).unwrap().unwrap();

    assert_eq!(first, second);
}

/// A doc already sealed as `holdout` and consistent with the
/// freshly computed holdout slice is left alone, and the rest of the
/// cell still gets split.
#[test]
fn split_leaves_correctly_sealed_holdout_alone() {
    let dir = tempfile::tempdir().unwrap();
    let mut records = build_cell(dir.path(), 4);

    let mut by_hash: Vec<(String, usize)> = records
        .iter()
        .enumerate()
        .map(|(idx, r)| (sha256_hex(r.id.as_bytes()), idx))
        .collect();
    by_hash.sort();
    // With n=4: train_end=2, dev_end=3, so the last slot (by hash order)
    // is the sole holdout candidate.
    let holdout_idx = by_hash[3].1;
    records[holdout_idx].split = Some(Split::Holdout);

    let manifest_path = dir.path().join("manifest.jsonl");
    common::write_manifest_raw(&manifest_path, &records);

    assert!(split::run(&args(dir.path(), false)).is_ok());

    let after = manifest::read_manifest(&manifest_path).unwrap().unwrap();
    let holdout_id = &records[holdout_idx].id;
    let after_record = after.iter().find(|r| &r.id == holdout_id).unwrap();
    assert_eq!(after_record.split, Some(Split::Holdout));
}

/// A doc sealed as `holdout` that does NOT match the freshly
/// computed holdout slice causes `split` to fail rather than silently
/// reassigning it.
#[test]
fn split_rejects_reassigning_sealed_holdout() {
    let dir = tempfile::tempdir().unwrap();
    let mut records = build_cell(dir.path(), 4);

    let mut by_hash: Vec<(String, usize)> = records
        .iter()
        .enumerate()
        .map(|(idx, r)| (sha256_hex(r.id.as_bytes()), idx))
        .collect();
    by_hash.sort();
    // The *first*-by-hash doc belongs in train, not holdout: sealing it
    // as holdout contradicts the freshly computed slice.
    let wrong_idx = by_hash[0].1;
    records[wrong_idx].split = Some(Split::Holdout);

    let manifest_path = dir.path().join("manifest.jsonl");
    common::write_manifest_raw(&manifest_path, &records);

    let err = split::run(&args(dir.path(), false)).unwrap_err();
    assert!(err.to_string().contains("holdout membership"));
}
