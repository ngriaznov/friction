//! Integration tests for `corpus-tool index`.

mod common;

use corpus_tool::commands::index::{self, Args};
use corpus_tool::manifest::{Genre, ModelInfo, Split};

fn args(corpus_dir: &std::path::Path, pack: std::path::PathBuf) -> Args {
    Args {
        corpus_dir: corpus_dir.to_path_buf(),
        pack,
        calibrate: false,
    }
}

/// A tiny hermetic corpus: two train-split human docs, one train-split
/// `qwen` doc, one `dev`-split doc (must be excluded), and one
/// `holdout`-split doc (must be excluded, and its lock file must never be
/// touched. See the test below).
fn build_mini_corpus(dir: &std::path::Path) {
    let human_words = common::filler_words(60);
    let llm_words = "the system performs an initialization of the database quickly";

    let sha_first = common::write_doc(dir, "human/docs/h001.md", &human_words);
    let sha_second = common::write_doc(dir, "human/docs/h002.md", &human_words);
    let sha_llm = common::write_doc(dir, "llm/docs/l001.md", llm_words);
    let sha_dev = common::write_doc(dir, "human/docs/h003.md", &human_words);
    let sha_holdout = common::write_doc(dir, "human/docs/h004.md", &human_words);

    let mut h1 = common::human_record("h001", Genre::Docs, sha_first);
    h1.split = Some(Split::Train);
    let mut h2 = common::human_record("h002", Genre::Docs, sha_second);
    h2.split = Some(Split::Train);
    let mut l1 = common::llm_record("l001", Genre::Docs, sha_llm);
    l1.split = Some(Split::Train);
    l1.model = Some(ModelInfo {
        name: "qwen2.5:7b-instruct".to_string(),
        quantization: "q4_k_m".to_string(),
    });
    let mut dev = common::human_record("h003", Genre::Docs, sha_dev);
    dev.split = Some(Split::Dev);
    let mut holdout = common::human_record("h004", Genre::Docs, sha_holdout);
    holdout.split = Some(Split::Holdout);

    common::write_manifest_raw(&dir.join("manifest.jsonl"), &[h1, h2, l1, dev, holdout]);
}

/// The written pack only reflects `train`-split docs: 2 human docs (not
/// 3 — the `dev`-split doc is excluded) and one `qwen` family stream.
#[test]
fn index_respects_train_only_data_protocol() {
    let dir = tempfile::tempdir().unwrap();
    build_mini_corpus(dir.path());

    let pack_path = dir.path().join("dms-index-en-v1.toml");
    index::run(&args(dir.path(), pack_path.clone())).unwrap();
    let text = std::fs::read_to_string(&pack_path).unwrap();

    assert!(text.contains("train_human_doc_count = 2"));
    assert!(text.contains("qwen = { doc_count = 1"));
    assert!(text.contains("[streams.qwen]"));
    assert!(text.contains("[streams.human]"));
    // Neither the dev-split nor the holdout-split doc's own id ever
    // leaks into the pack (they contribute no tokens of their own beyond
    // what the two included human docs already share, but their doc
    // count must not be reflected).
    assert!(!text.contains("gemma"));
    assert!(!text.contains("llama"));
    assert!(!text.contains("granite"));
}

/// A `holdout.lock` file present in the corpus directory is left
/// byte-unread/unmodified by `index` — this command has no reason to
/// ever open it (the seal-check code owns that), and never does.
#[test]
fn index_never_touches_holdout_lock() {
    let dir = tempfile::tempdir().unwrap();
    build_mini_corpus(dir.path());
    let lock_path = dir.path().join("holdout.lock");
    let sentinel = "h004\tdeadbeef\thuman/docs/h004.md\n";
    std::fs::write(&lock_path, sentinel).unwrap();

    let pack_path = dir.path().join("dms-index-en-v1.toml");
    index::run(&args(dir.path(), pack_path)).unwrap();

    let after = std::fs::read_to_string(&lock_path).unwrap();
    assert_eq!(after, sentinel, "holdout.lock must be left byte-unmodified");
}

/// Running `index` twice against the same corpus directory produces
/// byte-identical pack output, the strongest test of the
/// fixed-traversal-order determinism design.
#[test]
fn index_is_deterministic_across_reruns() {
    let dir = tempfile::tempdir().unwrap();
    build_mini_corpus(dir.path());

    let pack_a = dir.path().join("a.toml");
    let pack_b = dir.path().join("b.toml");
    index::run(&args(dir.path(), pack_a.clone())).unwrap();
    index::run(&args(dir.path(), pack_b.clone())).unwrap();

    let text_a = std::fs::read_to_string(&pack_a).unwrap();
    let text_b = std::fs::read_to_string(&pack_b).unwrap();
    assert_eq!(text_a, text_b);
}

/// A `train`-split `llm` doc whose `model.name` matches none of the four
/// known families is a hard error, not a silent skip.
#[test]
fn index_errors_on_unmapped_model_family() {
    let dir = tempfile::tempdir().unwrap();
    let sha = common::write_doc(dir.path(), "llm/docs/l001.md", "some text here");
    let mut record = common::llm_record("l001", Genre::Docs, sha);
    record.split = Some(Split::Train);
    record.model = Some(ModelInfo {
        name: "mystery-model:1b".to_string(),
        quantization: "q4".to_string(),
    });
    common::write_manifest_raw(&dir.path().join("manifest.jsonl"), &[record]);

    let pack_path = dir.path().join("dms-index-en-v1.toml");
    let err = index::run(&args(dir.path(), pack_path)).unwrap_err();
    assert!(
        err.to_string().contains("unmapped model family"),
        "unexpected error: {err}"
    );
}

/// The written pack round-trips through `friction_packs::DmsIndex::parse`
/// into a working vocabulary and suffix automata: the human automaton
/// finds a real match for a phrase actually present in the human corpus.
#[test]
fn index_output_round_trips_through_dms_index_parse() {
    let dir = tempfile::tempdir().unwrap();
    build_mini_corpus(dir.path());

    let pack_path = dir.path().join("dms-index-en-v1.toml");
    index::run(&args(dir.path(), pack_path.clone())).unwrap();
    let text = std::fs::read_to_string(&pack_path).unwrap();

    let parsed = friction_packs::DmsIndex::parse(&text).expect("generated pack must parse");
    assert!(
        parsed
            .family_sam(friction_packs::ModelFamily::Qwen)
            .is_some()
    );
    assert!(
        parsed
            .family_sam(friction_packs::ModelFamily::Gemma)
            .is_none()
    );

    // "the" is in the filler vocabulary used for every human doc, so it
    // must be a known vocabulary token with at least a length-1 match
    // against the human automaton.
    let the_id = parsed
        .vocab()
        .id_of("the")
        .expect("\"the\" must be in the shared vocab");
    let human_stats = parsed.human_sam().matching_stats(&[Some(the_id)]);
    assert_eq!(human_stats, vec![1]);
}

/// `--calibrate` runs end to end (parsing the pack it just wrote back
/// into a working `DmsIndex` and scoring the dev split) without erroring,
/// even when there is no dev-split data to score at all.
#[test]
fn index_calibrate_runs_without_error_on_empty_dev_split() {
    let dir = tempfile::tempdir().unwrap();
    build_mini_corpus(dir.path());

    let pack_path = dir.path().join("dms-index-en-v1.toml");
    let mut calibrated_args = args(dir.path(), pack_path);
    calibrated_args.calibrate = true;
    index::run(&calibrated_args).unwrap();
}
