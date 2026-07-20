//! Integration tests for `corpus-tool envelope`.

mod common;

use corpus_tool::commands::envelope::{self, Args};
use corpus_tool::manifest::{Genre, Split};

fn args(corpus_dir: &std::path::Path, out: std::path::PathBuf) -> Args {
    Args {
        corpus_dir: corpus_dir.to_path_buf(),
        out,
        lo_percentile: 10.0,
        hi_percentile: 90.0,
    }
}

/// An empty/missing corpus still writes a well-formed pack: a `[pack]`
/// header with a zero doc count, and no genre sections.
#[test]
fn envelope_empty_corpus_writes_header_only_pack() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("envelope-v1.toml");
    envelope::run(&args(dir.path(), out.clone())).unwrap();

    let text = std::fs::read_to_string(&out).unwrap();
    assert!(text.contains("[pack]"));
    assert!(text.contains("train_human_doc_count = 0"));
    assert!(!text.contains("[pack.docs_per_genre]\ndocs"));
}

/// Only `human`/`train` docs feed the pack: an `llm` doc and a `dev`-split
/// human doc in the same genre are excluded from `docs_per_genre`.
#[test]
fn envelope_only_counts_train_split_human_docs() {
    let dir = tempfile::tempdir().unwrap();
    let words = common::filler_words(320);

    let sha_train = common::write_doc(dir.path(), "human/docs/h001.md", &words);
    let sha_dev = common::write_doc(dir.path(), "human/docs/h002.md", &words);
    let sha_llm = common::write_doc(dir.path(), "llm/docs/l001.md", &words);

    let mut h_train = common::human_record("h001", Genre::Docs, sha_train);
    h_train.split = Some(Split::Train);
    let mut h_dev = common::human_record("h002", Genre::Docs, sha_dev);
    h_dev.split = Some(Split::Dev);
    let mut llm = common::llm_record("l001", Genre::Docs, sha_llm);
    llm.split = Some(Split::Train);

    common::write_manifest_raw(&dir.path().join("manifest.jsonl"), &[h_train, h_dev, llm]);

    let out = dir.path().join("envelope-v1.toml");
    envelope::run(&args(dir.path(), out.clone())).unwrap();

    let text = std::fs::read_to_string(&out).unwrap();
    assert!(text.contains("train_human_doc_count = 1"));
    assert!(text.contains("[pack.docs_per_genre]"));
    assert!(text.contains("docs = 1"));
    assert!(text.contains("[docs."));
}

/// Running twice against the same corpus produces byte-identical pack
/// files.
#[test]
fn envelope_output_is_deterministic_across_runs() {
    let dir = tempfile::tempdir().unwrap();
    let words = common::filler_words(320);
    let sha = common::write_doc(dir.path(), "human/blog/h001.md", &words);
    let mut record = common::human_record("h001", Genre::Blog, sha);
    record.split = Some(Split::Train);
    common::write_manifest_raw(&dir.path().join("manifest.jsonl"), &[record]);

    let out_a = dir.path().join("a.toml");
    let out_b = dir.path().join("b.toml");
    envelope::run(&args(dir.path(), out_a.clone())).unwrap();
    envelope::run(&args(dir.path(), out_b.clone())).unwrap();

    let a = std::fs::read_to_string(&out_a).unwrap();
    let b = std::fs::read_to_string(&out_b).unwrap();
    assert_eq!(a, b);
}

/// `--lo-percentile >= --hi-percentile` is a hard error, not a silently
/// degenerate band.
#[test]
fn envelope_rejects_lo_percentile_not_below_hi_percentile() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("envelope-v1.toml");
    let mut bad_args = args(dir.path(), out);
    bad_args.lo_percentile = 90.0;
    bad_args.hi_percentile = 10.0;
    assert!(envelope::run(&bad_args).is_err());
}

/// A percentile argument outside `[0, 100]` is a hard error.
#[test]
fn envelope_rejects_percentile_out_of_range() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("envelope-v1.toml");
    let mut bad_args = args(dir.path(), out);
    bad_args.hi_percentile = 150.0;
    assert!(envelope::run(&bad_args).is_err());
}

/// `--out` writes to a not-yet-existing directory, creating parents as
/// needed.
#[test]
fn envelope_creates_missing_out_directory() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("nested/packs/envelope-v1.toml");
    envelope::run(&args(dir.path(), out.clone())).unwrap();
    assert!(out.exists());
}
