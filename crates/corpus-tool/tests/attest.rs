//! Integration tests for `corpus-tool attest`.

mod common;

use corpus_tool::commands::attest::{self, Args};
use corpus_tool::manifest::{Genre, Split};

fn args(corpus_dir: &std::path::Path, pack: std::path::PathBuf) -> Args {
    Args {
        corpus_dir: corpus_dir.to_path_buf(),
        pack,
        calibrate_near_noop: false,
    }
}

/// A tiny hermetic corpus: two train-split human docs (whose text is
/// chosen so the expected bigram/skeleton output is hand-verifiable), one
/// train-split `llm` doc (must be excluded), one `dev`-split human doc
/// (must be excluded), and one `holdout`-split human doc (must be
/// excluded).
fn build_mini_corpus(dir: &std::path::Path) {
    let doc_text = "The quick fox runs. The quick fox jumps.";

    let sha_first = common::write_doc(dir, "human/docs/h001.md", doc_text);
    let sha_second = common::write_doc(dir, "human/docs/h002.md", "A slow dog sleeps.");
    let sha_llm = common::write_doc(dir, "llm/docs/l001.md", "The quick fox performs a jump.");
    let sha_dev = common::write_doc(dir, "human/docs/h003.md", doc_text);
    let sha_holdout = common::write_doc(dir, "human/docs/h004.md", doc_text);

    let mut h1 = common::human_record("h001", Genre::Docs, sha_first);
    h1.split = Some(Split::Train);
    let mut h2 = common::human_record("h002", Genre::Docs, sha_second);
    h2.split = Some(Split::Train);
    let mut l1 = common::llm_record("l001", Genre::Docs, sha_llm);
    l1.split = Some(Split::Train);
    let mut dev = common::human_record("h003", Genre::Docs, sha_dev);
    dev.split = Some(Split::Dev);
    let mut holdout = common::human_record("h004", Genre::Docs, sha_holdout);
    holdout.split = Some(Split::Holdout);

    common::write_manifest_raw(&dir.join("manifest.jsonl"), &[h1, h2, l1, dev, holdout]);
}

/// The written pack only reflects TRAIN-split human docs: 2 docs (not the
/// `llm`, `dev`, or `holdout` doc's text at all).
#[test]
fn attest_respects_train_human_only_data_protocol() {
    let dir = tempfile::tempdir().unwrap();
    build_mini_corpus(dir.path());

    let pack_path = dir.path().join("attestation-v1.toml");
    attest::run(&args(dir.path(), pack_path.clone())).unwrap();
    let text = std::fs::read_to_string(&pack_path).unwrap();

    assert!(text.contains("train_human_doc_count = 2"));
    // The llm doc's own distinctive word ("performs") never leaks in. The
    // dev/holdout docs share h001's text, so they contribute no
    // distinctive vocabulary of their own to check against. The doc-count
    // assertion above is what actually proves they were excluded.
    assert!(!text.contains("\"performs\""));
    // h002 (train-split human) is legitimately included: its distinctive
    // vocabulary does appear.
    assert!(text.contains("\"slow\""));
    assert!(text.contains("\"dog\""));
}

/// The generated pack round-trips through
/// `friction_packs::AttestationPack::parse` and attests the exact bigrams
/// and skeleton windows the two train-human sentences contain.
#[test]
fn attest_output_round_trips_and_attests_known_bigrams() {
    let dir = tempfile::tempdir().unwrap();
    build_mini_corpus(dir.path());

    let pack_path = dir.path().join("attestation-v1.toml");
    attest::run(&args(dir.path(), pack_path.clone())).unwrap();
    let text = std::fs::read_to_string(&pack_path).unwrap();

    let parsed = friction_packs::AttestationPack::parse(&text).expect("generated pack must parse");

    // "The quick fox runs." / "The quick fox jumps." (both train-human,
    // lowercased by tokenize()): every adjacent word pair is attested.
    assert!(parsed.bigram().attests("<s>", "the"));
    assert!(parsed.bigram().attests("the", "quick"));
    assert!(parsed.bigram().attests("quick", "fox"));
    assert!(parsed.bigram().attests("fox", "runs"));
    assert!(parsed.bigram().attests("fox", "jumps"));
    assert!(parsed.bigram().attests("runs", "."));
    assert!(parsed.bigram().attests("jumps", "."));

    // A pair that never occurs adjacently in the train-human corpus.
    assert!(!parsed.bigram().attests("<s>", "fox"));
    // Out-of-vocabulary words (only in the excluded llm/dev/holdout docs).
    assert!(!parsed.bigram().attests("the", "performs"));
}

/// Running `attest` twice against the same corpus directory produces
/// byte-identical pack output.
#[test]
fn attest_is_deterministic_across_reruns() {
    let dir = tempfile::tempdir().unwrap();
    build_mini_corpus(dir.path());

    let pack_a = dir.path().join("a.toml");
    let pack_b = dir.path().join("b.toml");
    attest::run(&args(dir.path(), pack_a.clone())).unwrap();
    attest::run(&args(dir.path(), pack_b.clone())).unwrap();

    let text_a = std::fs::read_to_string(&pack_a).unwrap();
    let text_b = std::fs::read_to_string(&pack_b).unwrap();
    assert_eq!(text_a, text_b);
}

/// A `holdout.lock` file present in the corpus directory is left
/// byte-unread/unmodified by `attest`.
#[test]
fn attest_never_touches_holdout_lock() {
    let dir = tempfile::tempdir().unwrap();
    build_mini_corpus(dir.path());
    let lock_path = dir.path().join("holdout.lock");
    let sentinel = "h004\tdeadbeef\thuman/docs/h004.md\n";
    std::fs::write(&lock_path, sentinel).unwrap();

    let pack_path = dir.path().join("attestation-v1.toml");
    attest::run(&args(dir.path(), pack_path)).unwrap();

    let after = std::fs::read_to_string(&lock_path).unwrap();
    assert_eq!(after, sentinel, "holdout.lock must be left byte-unmodified");
}

/// A pack built from an empty (no train-human docs) corpus still parses
/// cleanly, with an empty bigram/skeleton table rather than an error.
#[test]
fn attest_handles_a_corpus_with_no_train_human_docs() {
    let dir = tempfile::tempdir().unwrap();
    let sha = common::write_doc(dir.path(), "llm/docs/l001.md", "some text here");
    let mut record = common::llm_record("l001", Genre::Docs, sha);
    record.split = Some(Split::Train);
    common::write_manifest_raw(&dir.path().join("manifest.jsonl"), &[record]);

    let pack_path = dir.path().join("attestation-v1.toml");
    attest::run(&args(dir.path(), pack_path.clone())).unwrap();
    let text = std::fs::read_to_string(&pack_path).unwrap();

    assert!(text.contains("train_human_doc_count = 0"));
    let parsed = friction_packs::AttestationPack::parse(&text).expect("empty pack must parse");
    assert_eq!(parsed.bigram().edge_count(), 0);
    assert_eq!(parsed.skeleton().tag5_len(), 0);
}
