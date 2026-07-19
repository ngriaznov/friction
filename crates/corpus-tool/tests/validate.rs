//! Integration tests for `corpus-tool validate`.

mod common;

use corpus_tool::commands::validate::{self, Args};
use corpus_tool::manifest::Genre;

fn args(corpus_dir: &std::path::Path) -> Args {
    Args {
        corpus_dir: corpus_dir.to_path_buf(),
    }
}

/// A missing corpus directory (no manifest at all) is success,
/// not an error.
#[test]
fn validate_missing_corpus_dir_is_ok() {
    let dir = tempfile::tempdir().unwrap();
    let corpus_dir = dir.path().join("corpus");
    assert!(validate::run(&args(&corpus_dir)).is_ok());
}

/// An empty manifest file is success.
#[test]
fn validate_empty_manifest_is_ok() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("manifest.jsonl"), "").unwrap();
    assert!(validate::run(&args(dir.path())).is_ok());
}

/// The static golden fixture corpus under
/// `tests/fixtures/valid_corpus/` (checked into the repo, hashes
/// precomputed) validates cleanly. This is a regression fixture distinct
/// from the dynamically-built corpora used elsewhere in this file.
#[test]
fn validate_static_golden_fixture_corpus_passes() {
    let corpus_dir =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/valid_corpus");
    let result = validate::run(&args(&corpus_dir));
    assert!(
        result.is_ok(),
        "expected static fixture corpus to validate: {result:?}"
    );
}

/// A well-formed corpus (human + llm + quarantined docs, correct
/// hashes, all required fields) validates cleanly.
#[test]
fn validate_well_formed_corpus_passes() {
    let dir = tempfile::tempdir().unwrap();
    let words = common::filler_words(320);

    let human_sha = common::write_doc(dir.path(), "human/docs/h001.md", &words);
    let llm_sha = common::write_doc(dir.path(), "llm/blog/l001.md", &words);

    let mut records = vec![
        common::human_record("h001", Genre::Docs, human_sha),
        common::llm_record("l001", Genre::Blog, llm_sha),
    ];
    // A quarantined doc: license names CC-BY-SA, so it resolves
    // under quarantine/<genre>/ regardless of class.
    let se_sha = common::write_doc(dir.path(), "quarantine/forum/se001.md", &words);
    let mut se = common::human_record("se001", Genre::Forum, se_sha);
    se.license = "CC-BY-SA-4.0".to_string();
    se.provenance_evidence = Some("stackexchange export 2020-01-01".to_string());
    records.push(se);

    common::write_manifest_raw(&dir.path().join("manifest.jsonl"), &records);

    let result = validate::run(&args(dir.path()));
    assert!(result.is_ok(), "expected validate to pass: {result:?}");
}

/// Duplicate ids are rejected.
#[test]
fn validate_duplicate_id_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let words = common::filler_words(320);
    let sha = common::write_doc(dir.path(), "human/docs/h001.md", &words);

    let records = vec![
        common::human_record("h001", Genre::Docs, sha.clone()),
        common::human_record("h001", Genre::Docs, sha),
    ];
    common::write_manifest_raw(&dir.path().join("manifest.jsonl"), &records);

    let err = validate::run(&args(dir.path())).unwrap_err();
    assert!(err.to_string().contains("error(s)"));
}

/// A manifest entry whose file is missing on disk is rejected.
#[test]
fn validate_missing_file_rejected() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(dir.path()).unwrap();
    let record = common::human_record("h001", Genre::Docs, "deadbeef".to_string());
    common::write_manifest_raw(
        &dir.path().join("manifest.jsonl"),
        std::slice::from_ref(&record),
    );

    assert!(validate::run(&args(dir.path())).is_err());
}

/// A sha256 that doesn't match the file's actual content is
/// rejected.
#[test]
fn validate_sha256_mismatch_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let words = common::filler_words(320);
    common::write_doc(dir.path(), "human/docs/h001.md", &words);

    let record = common::human_record("h001", Genre::Docs, "0000000000".to_string());
    common::write_manifest_raw(
        &dir.path().join("manifest.jsonl"),
        std::slice::from_ref(&record),
    );

    assert!(validate::run(&args(dir.path())).is_err());
}

/// An empty license string is rejected.
#[test]
fn validate_empty_license_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let words = common::filler_words(320);
    let sha = common::write_doc(dir.path(), "human/docs/h001.md", &words);

    let mut record = common::human_record("h001", Genre::Docs, sha);
    record.license = String::new();
    record.provenance_evidence = Some("git commit 2020-01-01".to_string());
    common::write_manifest_raw(
        &dir.path().join("manifest.jsonl"),
        std::slice::from_ref(&record),
    );

    assert!(validate::run(&args(dir.path())).is_err());
}

/// A human doc with neither `provenance_evidence` nor
/// `license: "personal-attestation"` is rejected.
#[test]
fn validate_human_without_provenance_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let words = common::filler_words(320);
    let sha = common::write_doc(dir.path(), "human/readme/h001.md", &words);

    let mut record = common::human_record("h001", Genre::Readme, sha);
    record.license = "MIT".to_string();
    record.provenance_evidence = None;
    common::write_manifest_raw(
        &dir.path().join("manifest.jsonl"),
        std::slice::from_ref(&record),
    );

    let err = validate::run(&args(dir.path())).unwrap_err();
    assert!(err.to_string().contains("error(s)"));
}

/// A human doc with `provenance_evidence` set (and a non-attested
/// license) passes.
#[test]
fn validate_human_with_provenance_evidence_passes() {
    let dir = tempfile::tempdir().unwrap();
    let words = common::filler_words(320);
    let sha = common::write_doc(dir.path(), "human/readme/h001.md", &words);

    let mut record = common::human_record("h001", Genre::Readme, sha);
    record.license = "MIT".to_string();
    record.provenance_evidence = Some("archive.org 2019-06-01".to_string());
    common::write_manifest_raw(
        &dir.path().join("manifest.jsonl"),
        std::slice::from_ref(&record),
    );

    assert!(validate::run(&args(dir.path())).is_ok());
}

/// An llm doc missing `model` is rejected.
#[test]
fn validate_llm_missing_model_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let words = common::filler_words(320);
    let sha = common::write_doc(dir.path(), "llm/blog/l001.md", &words);

    let mut record = common::llm_record("l001", Genre::Blog, sha);
    record.model = None;
    common::write_manifest_raw(
        &dir.path().join("manifest.jsonl"),
        std::slice::from_ref(&record),
    );

    assert!(validate::run(&args(dir.path())).is_err());
}

/// An llm doc missing `gen_config` is rejected.
#[test]
fn validate_llm_missing_gen_config_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let words = common::filler_words(320);
    let sha = common::write_doc(dir.path(), "llm/blog/l001.md", &words);

    let mut record = common::llm_record("l001", Genre::Blog, sha);
    record.gen_config = None;
    common::write_manifest_raw(
        &dir.path().join("manifest.jsonl"),
        std::slice::from_ref(&record),
    );

    assert!(validate::run(&args(dir.path())).is_err());
}

/// A word count outside [300, 2000] is a warning, not a hard
/// validation failure.
#[test]
fn validate_word_count_out_of_range_is_warn_only() {
    let dir = tempfile::tempdir().unwrap();
    let words = common::filler_words(30); // well under 300
    let sha = common::write_doc(dir.path(), "human/docs/h001.md", &words);
    let record = common::human_record("h001", Genre::Docs, sha);
    common::write_manifest_raw(
        &dir.path().join("manifest.jsonl"),
        std::slice::from_ref(&record),
    );

    assert!(validate::run(&args(dir.path())).is_ok());
}

/// The manifest itself must parse strictly — an unknown field
/// anywhere in the JSONL file is a hard error via `manifest::read_manifest`.
#[test]
fn validate_malformed_manifest_json_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("manifest.jsonl"),
        r#"{"id":"h1","class":"human","genre":"docs","source":"x","license":"MIT","lang":"en","sha256":"abc","unexpected_field":true}"#,
    )
    .unwrap();

    assert!(validate::run(&args(dir.path())).is_err());
}
