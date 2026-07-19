//! Integration tests for `corpus-tool clean`.

mod common;

use corpus_tool::commands::clean::{self, Args};

fn args(incoming: &std::path::Path, out: &std::path::Path) -> Args {
    Args {
        incoming: incoming.to_path_buf(),
        out: out.to_path_buf(),
    }
}

/// CRLF line endings normalize to LF and the surviving doc keeps
/// its markdown structure.
#[test]
fn clean_normalizes_line_endings_and_keeps_markdown() {
    let dir = tempfile::tempdir().unwrap();
    let incoming = dir.path().join("incoming");
    let out = dir.path().join("out");

    let body = common::filler_words(320);
    let content = format!("# Title\r\n\r\n{body}\r\n");
    std::fs::create_dir_all(&incoming).unwrap();
    std::fs::write(incoming.join("doc.md"), content).unwrap();

    clean::run(&args(&incoming, &out)).unwrap();

    let cleaned = std::fs::read_to_string(out.join("doc.md")).unwrap();
    assert!(!cleaned.contains('\r'));
    assert!(cleaned.contains("# Title"));
}

/// A badge wall at the top of a README-style doc is stripped
/// while the following prose survives.
#[test]
fn clean_strips_badge_wall_from_readme() {
    let dir = tempfile::tempdir().unwrap();
    let incoming = dir.path().join("incoming");
    let out = dir.path().join("out");
    std::fs::create_dir_all(&incoming).unwrap();

    let body = common::filler_words(320);
    let content = format!(
        "![Build](https://img.shields.io/badge/build-passing.svg)\n\
         [![Coverage](cov.svg)](https://example.com/coverage)\n\n\
         # my-project\n\n{body}\n"
    );
    std::fs::write(incoming.join("readme.md"), content).unwrap();

    clean::run(&args(&incoming, &out)).unwrap();

    let cleaned = std::fs::read_to_string(out.join("readme.md")).unwrap();
    assert!(!cleaned.contains("shields.io"));
    assert!(!cleaned.contains("cov.svg"));
    assert!(cleaned.contains("# my-project"));
}

/// An HTML centering `<div>`/`<p align>` nav block is stripped.
#[test]
fn clean_strips_html_wrapper_block() {
    let dir = tempfile::tempdir().unwrap();
    let incoming = dir.path().join("incoming");
    let out = dir.path().join("out");
    std::fs::create_dir_all(&incoming).unwrap();

    let body = common::filler_words(320);
    let content =
        format!("<div align=\"center\">\n<img src=\"logo.png\">\n</div>\n\n# Title\n\n{body}\n");
    std::fs::write(incoming.join("doc.md"), content).unwrap();

    clean::run(&args(&incoming, &out)).unwrap();

    let cleaned = std::fs::read_to_string(out.join("doc.md")).unwrap();
    assert!(!cleaned.contains("<div"));
    assert!(cleaned.contains("# Title"));
}

/// A doc under 300 words after cleaning is dropped, not written
/// to `--out`.
#[test]
fn clean_drops_docs_under_300_words() {
    let dir = tempfile::tempdir().unwrap();
    let incoming = dir.path().join("incoming");
    let out = dir.path().join("out");
    std::fs::create_dir_all(&incoming).unwrap();

    let short_body = common::filler_words(50);
    std::fs::write(
        incoming.join("short.md"),
        format!("# Short\n\n{short_body}\n"),
    )
    .unwrap();

    clean::run(&args(&incoming, &out)).unwrap();

    assert!(!out.join("short.md").exists());
}

/// A doc that clears 300 words after cleaning is kept.
#[test]
fn clean_keeps_docs_at_or_above_300_words() {
    let dir = tempfile::tempdir().unwrap();
    let incoming = dir.path().join("incoming");
    let out = dir.path().join("out");
    std::fs::create_dir_all(&incoming).unwrap();

    let body = common::filler_words(300);
    std::fs::write(incoming.join("long.md"), format!("# Long\n\n{body}\n")).unwrap();

    clean::run(&args(&incoming, &out)).unwrap();

    assert!(out.join("long.md").exists());
}

/// Running `clean` twice on the same incoming directory produces
/// byte-identical output.
#[test]
fn clean_is_deterministic_across_runs() {
    let dir = tempfile::tempdir().unwrap();
    let incoming = dir.path().join("incoming");
    let out_a = dir.path().join("out_a");
    let out_b = dir.path().join("out_b");
    std::fs::create_dir_all(&incoming).unwrap();

    let body = common::filler_words(320);
    std::fs::write(
        incoming.join("doc.md"),
        format!("![Badge](b.svg)\n\n# Title\n\n{body}\n"),
    )
    .unwrap();

    clean::run(&args(&incoming, &out_a)).unwrap();
    clean::run(&args(&incoming, &out_b)).unwrap();

    let a = std::fs::read(out_a.join("doc.md")).unwrap();
    let b = std::fs::read(out_b.join("doc.md")).unwrap();
    assert_eq!(a, b);
}

/// Nested incoming directories mirror into `--out`.
#[test]
fn clean_mirrors_nested_directory_layout() {
    let dir = tempfile::tempdir().unwrap();
    let incoming = dir.path().join("incoming");
    let out = dir.path().join("out");
    let body = common::filler_words(320);
    std::fs::create_dir_all(incoming.join("sub/dir")).unwrap();
    std::fs::write(
        incoming.join("sub/dir/nested.md"),
        format!("# Nested\n\n{body}\n"),
    )
    .unwrap();

    clean::run(&args(&incoming, &out)).unwrap();

    assert!(out.join("sub/dir/nested.md").exists());
}
