//! Integration tests for `corpus-tool stats`.

mod common;

use corpus_tool::commands::stats::{self, Args};
use corpus_tool::manifest::Genre;

fn args(corpus_dir: &std::path::Path, report: Option<std::path::PathBuf>) -> Args {
    Args {
        corpus_dir: corpus_dir.to_path_buf(),
        report,
    }
}

/// An empty/missing corpus doesn't error.
#[test]
fn stats_empty_corpus_does_not_error() {
    let dir = tempfile::tempdir().unwrap();
    assert!(stats::run(&args(dir.path(), None)).is_ok());
}

/// `--report <path>` writes a markdown report containing the
/// per-cell doc counts and split counts.
#[test]
fn stats_report_written_to_file_with_counts() {
    let dir = tempfile::tempdir().unwrap();
    let words = common::filler_words(320);

    let sha1 = common::write_doc(dir.path(), "human/docs/h001.md", &words);
    let sha2 = common::write_doc(dir.path(), "human/docs/h002.md", &words);
    let sha3 = common::write_doc(dir.path(), "llm/blog/l001.md", &words);

    let mut h1 = common::human_record("h001", Genre::Docs, sha1);
    h1.split = Some(corpus_tool::manifest::Split::Train);
    let mut h2 = common::human_record("h002", Genre::Docs, sha2);
    h2.split = Some(corpus_tool::manifest::Split::Dev);
    let l1 = common::llm_record("l001", Genre::Blog, sha3);

    common::write_manifest_raw(&dir.path().join("manifest.jsonl"), &[h1, h2, l1]);

    let report_path = dir.path().join("report.md");
    stats::run(&args(dir.path(), Some(report_path.clone()))).unwrap();

    let report = std::fs::read_to_string(&report_path).unwrap();
    assert!(report.contains("# Corpus statistics"));
    assert!(report.contains("Total docs: 3"));
    assert!(report.contains("| human | docs | 2 |"));
    assert!(report.contains("| llm | blog | 1 |"));
}

/// Running `stats` twice on the same corpus produces
/// byte-identical reports (deterministic ordering, no ambient state).
#[test]
fn stats_output_is_deterministic_across_runs() {
    let dir = tempfile::tempdir().unwrap();
    let words = common::filler_words(320);

    let sha1 = common::write_doc(dir.path(), "human/docs/h001.md", &words);
    let sha2 = common::write_doc(dir.path(), "llm/email/l001.md", &words);
    let records = vec![
        common::human_record("h001", Genre::Docs, sha1),
        common::llm_record("l001", Genre::Email, sha2),
    ];
    common::write_manifest_raw(&dir.path().join("manifest.jsonl"), &records);

    let report_a = dir.path().join("a.md");
    let report_b = dir.path().join("b.md");
    stats::run(&args(dir.path(), Some(report_a.clone()))).unwrap();
    stats::run(&args(dir.path(), Some(report_b.clone()))).unwrap();

    let a = std::fs::read_to_string(&report_a).unwrap();
    let b = std::fs::read_to_string(&report_b).unwrap();
    assert_eq!(a, b);
}
