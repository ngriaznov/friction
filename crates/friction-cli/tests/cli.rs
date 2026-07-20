//! End-to-end integration tests for the built `friction` binary: exit
//! codes, stdin/stdout piping, `--in-place`, and output-shape stability
//! across repeated runs.
//!
//! SARIF-schema validity has its own test file (`tests/sarif_schema.rs`);
//! this file covers `check`/`fix`/`explain` behavior in general.

use std::fs;
use std::path::{Path, PathBuf};

use assert_cmd::Command;
use predicates::prelude::*;

/// A small fixture that reliably triggers findings (`connective.surgery`,
/// `lexical.filler_phrase`, `symmetry.not_just_but`,
/// `symmetry.triad_reduction`) and several out-of-envelope `blog`
/// metrics — see this crate's own exploration for why: two
/// sentence-initial connectives, two `"it is worth noting that"`
/// filler phrases, two `"not just X but also Y"` constructions, and one
/// flat three-item triad.
const MESSY_BLOG: &str = "Moreover, it is worth noting that this release is not just fast but \
                           also reliable. Furthermore, it is worth noting that the results are \
                           not just accurate but also consistent. In conclusion, this release \
                           delivers speed, reliability, and consistency for every team.\n";

/// A real human-written `docs`-genre corpus document that sits inside
/// every `docs` envelope band and triggers no findings — `check` against
/// it exits `0`. Referencing the existing corpus fixture rather than
/// vendoring a copy: the exact bytes matter here (they were empirically
/// confirmed clean against the shipped `envelope-v2` pack), so drift
/// between a copy and the original corpus file would silently break this
/// test's premise.
fn clean_docs_fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus/human/docs/1fedcd88e3cf6845.md")
}

fn friction() -> Command {
    Command::cargo_bin("friction").expect("the friction binary builds")
}

fn write_fixture(dir: &Path, name: &str, contents: &str) -> PathBuf {
    let path = dir.join(name);
    fs::write(&path, contents).expect("fixture writes");
    path
}

// ---------------------------------------------------------------------
// check
// ---------------------------------------------------------------------

/// `check` on a document that sits fully inside its genre's envelope,
/// with no rule findings, exits `0`.
#[test]
fn check_exits_zero_for_a_clean_in_envelope_document() {
    friction()
        .arg("check")
        .arg(clean_docs_fixture())
        .args(["--genre", "docs"])
        .assert()
        .success();
}

/// `check` on a document with real findings exits `1` and prints every
/// firing rule's id to stdout.
#[test]
fn check_exits_one_and_lists_findings_for_a_messy_document() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = write_fixture(dir.path(), "messy.md", MESSY_BLOG);

    friction()
        .arg("check")
        .arg(&path)
        .args(["--genre", "blog"])
        .assert()
        .code(1)
        .stdout(predicate::str::contains("connective.surgery"))
        .stdout(predicate::str::contains("symmetry.not_just_but"));
}

/// `check -` reads from stdin, and an omitted `--genre` defaults to
/// `docs` with a note on stderr rather than failing.
#[test]
fn check_reads_stdin_and_defaults_genre_with_a_note() {
    friction()
        .arg("check")
        .arg("-")
        .write_stdin(MESSY_BLOG)
        .assert()
        .code(1)
        .stderr(predicate::str::contains("defaulting to"));
}

/// `check --format json` never fails to parse as JSON, and two runs over
/// the same input produce byte-identical stdout — the JSON
/// shape-stability guarantee.
#[test]
fn check_json_output_is_byte_identical_across_runs() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = write_fixture(dir.path(), "messy.md", MESSY_BLOG);

    let run = || {
        friction()
            .arg("check")
            .arg(&path)
            .args(["--genre", "blog", "--format", "json"])
            .output()
            .expect("friction runs")
    };
    let first = run();
    let second = run();
    assert_eq!(first.stdout, second.stdout);

    let value: serde_json::Value =
        serde_json::from_slice(&first.stdout).expect("check --format json prints valid JSON");
    assert_eq!(value["genre"], "blog");
    assert!(value["findings"].as_array().unwrap().len() >= 4);
}

/// `check --format text` never emits an ANSI escape byte when stdout is
/// piped (the default in every test harness) — the byte-stability
/// guarantee `crate::diagnostics` documents.
#[test]
fn check_text_output_is_never_colorized_when_piped() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = write_fixture(dir.path(), "messy.md", MESSY_BLOG);

    let output = friction()
        .arg("check")
        .arg(&path)
        .args(["--genre", "blog"])
        .output()
        .expect("friction runs");
    assert!(
        !output.stdout.contains(&0x1b),
        "text output must contain no ESC bytes when piped"
    );
}

/// A missing input file is a hard error: exit code `2`, a message on
/// stderr, nothing on stdout.
#[test]
fn check_exits_two_for_a_missing_file() {
    friction()
        .arg("check")
        .arg("/no/such/file/anywhere.md")
        .args(["--genre", "docs"])
        .assert()
        .code(2)
        .stdout(predicate::str::is_empty())
        .stderr(predicate::str::contains("error"));
}

// ---------------------------------------------------------------------
// fix
// ---------------------------------------------------------------------

/// `fix` writes the fixed text to stdout and a round summary to stderr,
/// leaving stdout containing only fixed prose (never the summary).
#[test]
fn fix_writes_fixed_text_to_stdout_and_summary_to_stderr() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = write_fixture(dir.path(), "messy.md", MESSY_BLOG);

    let output = friction()
        .arg("fix")
        .arg(&path)
        .args(["--genre", "blog"])
        .output()
        .expect("friction runs");
    assert!(output.status.success());

    let stdout = String::from_utf8(output.stdout).expect("valid UTF-8");
    assert_ne!(
        stdout, MESSY_BLOG,
        "fix must actually change the messy input"
    );
    assert!(
        !stdout.contains("it is worth noting that"),
        "the filler phrase must be gone from the fixed output"
    );

    let stderr = String::from_utf8(output.stderr).expect("valid UTF-8");
    assert!(stderr.contains("round(s)"));
    assert!(stderr.contains("patch(es) applied"));
}

/// `fix --in-place` rewrites the input file itself and prints nothing to
/// stdout.
#[test]
fn fix_in_place_rewrites_the_input_file() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = write_fixture(dir.path(), "messy.md", MESSY_BLOG);

    friction()
        .arg("fix")
        .arg(&path)
        .args(["--genre", "blog", "--in-place"])
        .assert()
        .success()
        .stdout(predicate::str::is_empty());

    let rewritten = fs::read_to_string(&path).expect("file was rewritten");
    assert_ne!(rewritten, MESSY_BLOG);
}

/// `fix --in-place -` (stdin has no file to write back to) is rejected
/// with exit code `2`.
#[test]
fn fix_in_place_rejects_stdin_input() {
    friction()
        .arg("fix")
        .arg("-")
        .args(["--genre", "blog", "--in-place"])
        .write_stdin(MESSY_BLOG)
        .assert()
        .code(2)
        .stderr(predicate::str::contains("stdin"));
}

/// `fix --suggest` lists remaining `Suggest`-tier findings on stderr,
/// alongside (not replacing) the round summary.
#[test]
fn fix_suggest_lists_remaining_suggestions_on_stderr() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = write_fixture(dir.path(), "messy.md", MESSY_BLOG);

    let output = friction()
        .arg("fix")
        .arg(&path)
        .args(["--genre", "blog", "--suggest"])
        .output()
        .expect("friction runs");
    assert!(output.status.success());

    let stderr = String::from_utf8(output.stderr).expect("valid UTF-8");
    assert!(
        stderr.contains("round(s)"),
        "the summary must still be there"
    );
    assert!(
        stderr.contains("symmetry.not_just_but") || stderr.contains("symmetry.triad_reduction"),
        "a Suggest-tier finding (neither rule ever auto-fixes) must be listed, got: {stderr}"
    );
}

/// `fix -` reads from stdin and writes the fixed text to stdout.
#[test]
fn fix_reads_stdin() {
    let output = friction()
        .arg("fix")
        .arg("-")
        .args(["--genre", "blog"])
        .write_stdin(MESSY_BLOG)
        .output()
        .expect("friction runs");
    assert!(output.status.success());
    assert!(!output.stdout.is_empty());
}

/// `fix --format sarif` is rejected (SARIF is `check`-only) with exit
/// code `2`.
#[test]
fn fix_rejects_sarif_format() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = write_fixture(dir.path(), "messy.md", MESSY_BLOG);

    friction()
        .arg("fix")
        .arg(&path)
        .args(["--genre", "blog", "--format", "sarif"])
        .assert()
        .code(2)
        .stderr(predicate::str::contains("sarif"));
}

// ---------------------------------------------------------------------
// explain
// ---------------------------------------------------------------------

/// `explain` prints a before/after metric table and a plan schedule,
/// never the fixed text itself.
#[test]
fn explain_prints_comparison_table_and_schedule_without_fixed_text() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = write_fixture(dir.path(), "messy.md", MESSY_BLOG);

    let output = friction()
        .arg("explain")
        .arg(&path)
        .args(["--genre", "blog"])
        .output()
        .expect("friction runs");
    assert!(output.status.success());

    let stdout = String::from_utf8(output.stdout).expect("valid UTF-8");
    assert!(stdout.contains("METRIC"));
    assert!(stdout.contains("triad_rate"));
    assert!(stdout.contains("plan schedule"));
    assert!(
        !stdout.contains("it is worth noting that"),
        "explain must never print the fixed (or original) document text"
    );
}

/// `explain --format json` prints a `metrics`/`plan`/`fixpoint` shaped
/// report, stable across two runs.
#[test]
fn explain_json_output_is_byte_identical_across_runs() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = write_fixture(dir.path(), "messy.md", MESSY_BLOG);

    let run = || {
        friction()
            .arg("explain")
            .arg(&path)
            .args(["--genre", "blog", "--format", "json"])
            .output()
            .expect("friction runs")
    };
    let first = run();
    let second = run();
    assert_eq!(first.stdout, second.stdout);

    let value: serde_json::Value =
        serde_json::from_slice(&first.stdout).expect("explain --format json prints valid JSON");
    assert!(value["metrics"].as_array().unwrap().len() == 21);
    assert_eq!(value["plan"]["entries"].as_array().unwrap().len(), 6);
    assert!(value["fixpoint"]["rounds"].as_u64().unwrap() >= 1);
}

/// `explain --format sarif` is rejected (SARIF is `check`-only) with exit
/// code `2`.
#[test]
fn explain_rejects_sarif_format() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = write_fixture(dir.path(), "messy.md", MESSY_BLOG);

    friction()
        .arg("explain")
        .arg(&path)
        .args(["--genre", "blog", "--format", "sarif"])
        .assert()
        .code(2);
}

// ---------------------------------------------------------------------
// setup (regression: still wired up after this crate's other subcommands
// were added)
// ---------------------------------------------------------------------

/// `friction setup` on the (currently empty) registry still succeeds —
/// this crate's other subcommands must not have disturbed it.
#[test]
fn setup_still_works_on_an_empty_registry() {
    friction().arg("setup").assert().success();
}
