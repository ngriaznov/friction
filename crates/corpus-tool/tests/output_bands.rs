//! Integration tests for `corpus-tool output-bands`.

mod common;

use corpus_tool::commands::output_bands::{self, Args};
use corpus_tool::manifest::{Genre, Split};
use friction_core::Lang;

fn build_mini_corpus(dir: &std::path::Path) {
    let train_text = "Our development plan covers the rollout. Development matters to us, \
                       and we cover the timeline before the launch.";
    let dev_text = "development development development";

    let sha_train = common::write_doc(dir, "human/docs/h001.md", train_text);
    let sha_dev = common::write_doc(dir, "human/docs/h002.md", dev_text);

    let mut h1 = common::human_record("h001", Genre::Docs, sha_train);
    h1.split = Some(Split::Train);
    let mut h2 = common::human_record("h002", Genre::Docs, sha_dev);
    h2.split = Some(Split::Dev);

    common::write_manifest_raw(&dir.join("manifest.jsonl"), &[h1, h2]);
}

fn args(corpus_dir: &std::path::Path, report: std::path::PathBuf) -> Args {
    Args {
        corpus_dir: corpus_dir.to_path_buf(),
        report,
        frames: Vec::new(),
        ceiling_multiplier: 3.0,
        lang: Lang::En,
    }
}

#[test]
fn output_bands_writes_a_report_with_measured_rates_for_default_frames() {
    let dir = tempfile::tempdir().unwrap();
    build_mini_corpus(dir.path());

    let report_path = dir.path().join("OUTPUT_BANDS.md");
    output_bands::run(&args(dir.path(), report_path.clone())).unwrap();

    let text = std::fs::read_to_string(&report_path).unwrap();
    assert!(text.contains("\"development\""));
    assert!(text.contains("\"we cover\""));
    assert!(text.contains("[[output_frequency_bands]]"));
    // Only the train-split doc's text (not the dev-split doc's) feeds the
    // measurement.
    assert!(text.contains("1 train-split human doc"));
}

#[test]
fn output_bands_respects_a_custom_frame_list() {
    let dir = tempfile::tempdir().unwrap();
    build_mini_corpus(dir.path());

    let report_path = dir.path().join("OUTPUT_BANDS.md");
    let mut custom_args = args(dir.path(), report_path.clone());
    custom_args.frames = vec!["matters".to_string()];
    output_bands::run(&custom_args).unwrap();

    let text = std::fs::read_to_string(&report_path).unwrap();
    assert!(text.contains("\"matters\""));
    assert!(!text.contains("\"development\""));
    assert!(!text.contains("\"we cover\""));
}

/// A `holdout.lock` file present in the corpus directory is left
/// byte-unread/unmodified by `output-bands` — this command has no reason
/// to ever open it, and never does.
#[test]
fn output_bands_never_touches_holdout_lock() {
    let dir = tempfile::tempdir().unwrap();
    build_mini_corpus(dir.path());
    let lock_path = dir.path().join("holdout.lock");
    let sentinel = "h999\tdeadbeef\thuman/docs/h999.md\n";
    std::fs::write(&lock_path, sentinel).unwrap();

    let report_path = dir.path().join("OUTPUT_BANDS.md");
    output_bands::run(&args(dir.path(), report_path)).unwrap();

    let after = std::fs::read_to_string(&lock_path).unwrap();
    assert_eq!(after, sentinel, "holdout.lock must be left byte-unmodified");
}
