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
        auc_include_threshold: 0.55,
    }
}

/// An empty/missing corpus still writes a well-formed pack: a `[pack]`
/// header with zero doc counts, and no genre sections.
#[test]
fn envelope_empty_corpus_writes_header_only_pack() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("envelope-v2.toml");
    envelope::run(&args(dir.path(), out.clone())).unwrap();

    let text = std::fs::read_to_string(&out).unwrap();
    assert!(text.contains("[pack]"));
    assert!(text.contains("train_human_doc_count = 0"));
    assert!(text.contains("train_llm_doc_count = 0"));
    assert!(!text.contains("[pack.human_docs_per_genre]\ndocs"));
    assert!(!text.contains("[pack.llm_docs_per_genre]\ndocs"));
}

/// Percentile bands still come from `human`/`train` docs only (a
/// `dev`-split human doc is excluded from them), but a `train`-split
/// `llm` doc in the same genre now *does* feed the pack — its own doc
/// count, and the train-internal AUC/direction/include verdict for every
/// metric in that genre.
#[test]
fn envelope_bands_are_human_train_only_but_llm_train_feeds_direction() {
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

    let out = dir.path().join("envelope-v2.toml");
    envelope::run(&args(dir.path(), out.clone())).unwrap();

    let text = std::fs::read_to_string(&out).unwrap();
    assert!(text.contains("train_human_doc_count = 1"));
    assert!(text.contains("train_llm_doc_count = 1"));
    assert!(text.contains("[pack.human_docs_per_genre]"));
    assert!(text.contains("[pack.llm_docs_per_genre]"));
    assert!(text.contains("docs = 1"));
    assert!(text.contains("[docs."));
    assert!(text.contains("direction ="));
    assert!(text.contains("include ="));
}

/// The human/llm train doc in the fixture above have byte-identical
/// content, so every `MetricVector` field ties between the two classes:
/// the oriented Mann-Whitney AUC is exactly the tied value `0.5` for
/// every metric, which is below the default `0.55` inclusion threshold —
/// so every metric in that genre is written `include = false`, and none
/// is `include = true`.
#[test]
fn envelope_ties_between_identical_human_and_llm_train_content_exclude_every_metric() {
    let dir = tempfile::tempdir().unwrap();
    let words = common::filler_words(320);

    let sha_human_a = common::write_doc(dir.path(), "human/docs/h1.md", &words);
    let sha_human_b = common::write_doc(dir.path(), "human/docs/h2.md", &words);
    let sha_llm_a = common::write_doc(dir.path(), "llm/docs/l1.md", &words);
    let sha_llm_b = common::write_doc(dir.path(), "llm/docs/l2.md", &words);

    let mut h1 = common::human_record("h1", Genre::Docs, sha_human_a);
    h1.split = Some(Split::Train);
    let mut h2 = common::human_record("h2", Genre::Docs, sha_human_b);
    h2.split = Some(Split::Train);
    let mut l1 = common::llm_record("l1", Genre::Docs, sha_llm_a);
    l1.split = Some(Split::Train);
    let mut l2 = common::llm_record("l2", Genre::Docs, sha_llm_b);
    l2.split = Some(Split::Train);

    common::write_manifest_raw(&dir.path().join("manifest.jsonl"), &[h1, h2, l1, l2]);

    let out = dir.path().join("envelope-v2.toml");
    envelope::run(&args(dir.path(), out.clone())).unwrap();

    let text = std::fs::read_to_string(&out).unwrap();
    assert!(text.contains("train_auc = 0.5"));
    assert!(text.contains("include = false"));
    assert!(!text.contains("include = true"));
}

/// A genre with train-human docs but no train-llm docs at all still gets
/// its percentile bands, but every metric defaults to `include = true`
/// with no `train_auc` recorded (the train-internal comparison is
/// undefined).
#[test]
fn envelope_defaults_to_include_true_when_genre_has_no_train_llm_docs() {
    let dir = tempfile::tempdir().unwrap();
    let words = common::filler_words(320);
    let sha = common::write_doc(dir.path(), "human/readme/h001.md", &words);
    let mut record = common::human_record("h001", Genre::Readme, sha);
    record.split = Some(Split::Train);
    common::write_manifest_raw(&dir.path().join("manifest.jsonl"), &[record]);

    let out = dir.path().join("envelope-v2.toml");
    envelope::run(&args(dir.path(), out.clone())).unwrap();

    let text = std::fs::read_to_string(&out).unwrap();
    assert!(text.contains("[readme."));
    assert!(text.contains("include = true"));
    assert!(!text.contains("include = false"));
    assert!(!text.contains("train_auc ="));
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
    let out = dir.path().join("envelope-v2.toml");
    let mut bad_args = args(dir.path(), out);
    bad_args.lo_percentile = 90.0;
    bad_args.hi_percentile = 10.0;
    assert!(envelope::run(&bad_args).is_err());
}

/// A percentile argument outside `[0, 100]` is a hard error.
#[test]
fn envelope_rejects_percentile_out_of_range() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("envelope-v2.toml");
    let mut bad_args = args(dir.path(), out);
    bad_args.hi_percentile = 150.0;
    assert!(envelope::run(&bad_args).is_err());
}

/// An `--auc-include-threshold` outside `[0.5, 1.0]` is a hard error: an
/// oriented AUC is never below 0.5, so a lower threshold would make
/// `include` trivially always true.
#[test]
fn envelope_rejects_auc_include_threshold_out_of_range() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("envelope-v2.toml");
    let mut bad_args = args(dir.path(), out);
    bad_args.auc_include_threshold = 0.2;
    assert!(envelope::run(&bad_args).is_err());
}

/// `--out` writes to a not-yet-existing directory, creating parents as
/// needed.
#[test]
fn envelope_creates_missing_out_directory() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("nested/packs/envelope-v2.toml");
    envelope::run(&args(dir.path(), out.clone())).unwrap();
    assert!(out.exists());
}
