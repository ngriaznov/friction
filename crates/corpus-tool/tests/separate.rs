//! Integration tests for `corpus-tool separate`.

mod common;

use corpus_tool::commands::{envelope, separate};
use corpus_tool::manifest::{Genre, Split};

fn envelope_args(corpus_dir: &std::path::Path, out: std::path::PathBuf) -> envelope::Args {
    envelope::Args {
        corpus_dir: corpus_dir.to_path_buf(),
        out,
        lo_percentile: 10.0,
        hi_percentile: 90.0,
    }
}

fn separate_args(
    corpus_dir: &std::path::Path,
    envelope_pack: std::path::PathBuf,
    report: std::path::PathBuf,
) -> separate::Args {
    separate::Args {
        corpus_dir: corpus_dir.to_path_buf(),
        envelope: envelope_pack,
        report,
    }
}

/// Builds a small corpus: one train-split human `docs` doc (feeds the
/// envelope pack), plus dev-split human and llm `docs` docs (feed the
/// separation report). Every doc has identical filler content, so with
/// the placeholder all-zero `MetricSource`, every metric is tied between
/// classes — the report's AUCs should all land exactly at the oriented
/// tie value, 0.5.
fn build_corpus(dir: &std::path::Path) {
    let words = common::filler_words(320);

    let sha_train = common::write_doc(dir, "human/docs/h_train.md", &words);
    let sha_human_a = common::write_doc(dir, "human/docs/h_dev1.md", &words);
    let sha_human_b = common::write_doc(dir, "human/docs/h_dev2.md", &words);
    let sha_llm_a = common::write_doc(dir, "llm/docs/l_dev1.md", &words);
    let sha_llm_b = common::write_doc(dir, "llm/docs/l_dev2.md", &words);

    let mut h_train = common::human_record("h_train", Genre::Docs, sha_train);
    h_train.split = Some(Split::Train);
    let mut h_dev1 = common::human_record("h_dev1", Genre::Docs, sha_human_a);
    h_dev1.split = Some(Split::Dev);
    let mut h_dev2 = common::human_record("h_dev2", Genre::Docs, sha_human_b);
    h_dev2.split = Some(Split::Dev);
    let mut l_dev1 = common::llm_record("l_dev1", Genre::Docs, sha_llm_a);
    l_dev1.split = Some(Split::Dev);
    let mut l_dev2 = common::llm_record("l_dev2", Genre::Docs, sha_llm_b);
    l_dev2.split = Some(Split::Dev);

    common::write_manifest_raw(
        &dir.join("manifest.jsonl"),
        &[h_train, h_dev1, h_dev2, l_dev1, l_dev2],
    );
}

/// End to end: `envelope` produces a pack for `docs`, then `separate`
/// reads it. Every metric row for `docs` shows the tied-value AUC (0.5,
/// oriented to "llm higher") with the right per-class doc counts; genres
/// with no dev docs at all report `n/a` and "no envelope for this genre".
#[test]
fn separate_report_reflects_tied_metrics_and_missing_genres() {
    let dir = tempfile::tempdir().unwrap();
    build_corpus(dir.path());

    let pack_path = dir.path().join("envelope-v1.toml");
    envelope::run(&envelope_args(dir.path(), pack_path.clone())).unwrap();

    let report_path = dir.path().join("report.md");
    separate::run(&separate_args(dir.path(), pack_path, report_path.clone())).unwrap();

    let report = std::fs::read_to_string(&report_path).unwrap();

    // All five genre sections appear, in the fixed declaration order.
    let docs_pos = report.find("## docs").unwrap();
    let blog_pos = report.find("## blog").unwrap();
    let readme_pos = report.find("## readme").unwrap();
    let email_pos = report.find("## email").unwrap();
    let forum_pos = report.find("## forum").unwrap();
    assert!(docs_pos < blog_pos);
    assert!(blog_pos < readme_pos);
    assert!(readme_pos < email_pos);
    assert!(email_pos < forum_pos);

    // `docs` has 2 human / 2 llm dev docs, all metrics tied -> AUC 0.5000
    // oriented to "llm higher" (see `build_corpus`'s doc comment).
    assert!(report.contains("| triad_rate | 2 | 2 | 0.5000 | llm higher |"));
    assert!(report.contains("Summary: docs — human n=2, llm n=2, combined-score AUC = 0.5000"));

    // `blog` has no dev docs at all: every metric row is n/a, and the
    // combined-score summary reports the missing envelope explicitly.
    assert!(report.contains("| triad_rate | 0 | 0 | n/a | n/a |"));
    assert!(report.contains(
        "Summary: blog — human n=0, llm n=0, combined-score AUC = n/a (no envelope for this genre)"
    ));
}

/// Running `separate` twice against the same corpus and pack produces
/// byte-identical reports.
#[test]
fn separate_report_is_deterministic_across_runs() {
    let dir = tempfile::tempdir().unwrap();
    build_corpus(dir.path());

    let pack_path = dir.path().join("envelope-v1.toml");
    envelope::run(&envelope_args(dir.path(), pack_path.clone())).unwrap();

    let report_a = dir.path().join("a.md");
    let report_b = dir.path().join("b.md");
    separate::run(&separate_args(
        dir.path(),
        pack_path.clone(),
        report_a.clone(),
    ))
    .unwrap();
    separate::run(&separate_args(dir.path(), pack_path, report_b.clone())).unwrap();

    let a = std::fs::read_to_string(&report_a).unwrap();
    let b = std::fs::read_to_string(&report_b).unwrap();
    assert_eq!(a, b);
}

/// A genre with human dev docs but no llm dev docs (or vice versa)
/// reports `n/a` for that metric's AUC rather than a fabricated value.
#[test]
fn separate_report_shows_na_when_one_class_is_absent_in_a_genre() {
    let dir = tempfile::tempdir().unwrap();
    let words = common::filler_words(320);

    let sha_train = common::write_doc(dir.path(), "human/readme/h_train.md", &words);
    let sha_dev = common::write_doc(dir.path(), "human/readme/h_dev.md", &words);

    let mut h_train = common::human_record("h_train", Genre::Readme, sha_train);
    h_train.split = Some(Split::Train);
    let mut h_dev = common::human_record("h_dev", Genre::Readme, sha_dev);
    h_dev.split = Some(Split::Dev);

    common::write_manifest_raw(&dir.path().join("manifest.jsonl"), &[h_train, h_dev]);

    let pack_path = dir.path().join("envelope-v1.toml");
    envelope::run(&envelope_args(dir.path(), pack_path.clone())).unwrap();

    let report_path = dir.path().join("report.md");
    separate::run(&separate_args(dir.path(), pack_path, report_path.clone())).unwrap();
    let report = std::fs::read_to_string(&report_path).unwrap();

    assert!(report.contains("| triad_rate | 1 | 0 | n/a | n/a |"));
    assert!(report.contains("Summary: readme — human n=1, llm n=0, combined-score AUC = n/a"));
}

/// The report ends with an explicit go/no-go verdict, not just five raw
/// AUC numbers a reader has to eyeball. With `build_corpus`'s fixture,
/// `docs` gets a defined but low combined-score AUC (0.5, tied metrics)
/// and the other four genres have no envelope entry at all (`n/a`), so
/// zero genres clear the 0.85 bar and the section must say so plainly.
#[test]
fn separate_report_states_gate_verdict_from_combined_score_aucs() {
    let dir = tempfile::tempdir().unwrap();
    build_corpus(dir.path());

    let pack_path = dir.path().join("envelope-v1.toml");
    envelope::run(&envelope_args(dir.path(), pack_path.clone())).unwrap();

    let report_path = dir.path().join("report.md");
    separate::run(&separate_args(dir.path(), pack_path, report_path.clone())).unwrap();
    let report = std::fs::read_to_string(&report_path).unwrap();

    assert!(report.contains("## Combined-score gate"));
    assert!(report.contains(
        "Genres whose combined-score AUC reaches 0.8500: 0 of 5 (target: at least 3). \
         Status: NOT MET."
    ));
    assert!(report.contains("| docs | 0.5000 | no |"));
    assert!(report.contains("| blog | n/a | no |"));
}
