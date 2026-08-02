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

/// A fixture that reliably drives every one of the four repair
/// operations (ritual deletion, paired substitution, derivational pivot,
/// gated span deletion) in one document, and leaves at least one gate-held
/// candidate behind (a pivot held on the near-no-op budget, a deletion
/// held on an unattested seam) — the `composed_four_operation_paragraph`
/// fixture from `docs/research/fixtures.json`, copied here verbatim.
const MESSY_BLOG: &str = "This guide will walk you through configuring the backup agent for \
                           your staging environment. It is important to note that the agent \
                           performs validation of the configuration file before each run. Once \
                           validation succeeds, the agent conducts an analysis of the snapshot \
                           catalog and simply uploads any missing segments. By following these \
                           steps, you can quickly and easily verify that your backups are \
                           consistent. If you have any questions or require further assistance, \
                           please reach out to our support team.";

/// A short document with zero detected spans against every generator
/// family — the `near_noop_clean_text` fixture from
/// `docs/research/fixtures.json`, copied here verbatim. Its metrics are
/// not necessarily inside any real genre's envelope (it's short), so
/// `check`'s exit-code tests that need spans *and* metrics both clean use
/// this together with an empty `--pack` override (see [`empty_pack`]).
const CLEAN: &str = "Run the scanner from the project root. Results stream in as they are \
                      found, and nothing is deleted without confirmation.";

/// Two paragraphs, copied from `corpus/llm/docs/57397cc503b594ba.md`, that
/// the DMS channel reliably flags one span in each of (originally verified
/// against the `claude` family stream with `friction check --family claude
/// --genre docs`, back when the channel was still per-family; the pooled
/// machine automaton the channel now scans against subsumes that stream,
/// so the same two spans still fire — see this module's own `dms.machine`
/// assertions): confirms `fix`'s paraphrase report actually fires on real
/// corpus text, not only on synthetic unit-test spans. Zero-patch fixture
/// on purpose (`fix` applies no repair-engine edit to either paragraph) so
/// the fixed output is byte-identical to the input, isolating the
/// paraphrase report from the repair engine's own output.
///
/// One byte differs from the corpus source: the original's em dash before
/// "including" is a plain comma here. Register's em-dash operation
/// (`register.em_dash`) now licenses a rewrite there — a real fix, not a
/// bug, but one this fixture must not exercise, since it exists to isolate
/// the paraphrase *report* from the repair engine's own edits, not to
/// re-litigate whether that edit is correct (`friction-register`'s own
/// tests cover that). The DMS channel scores token n-grams, not
/// punctuation, so this substitution changes neither flagged span.
const DMS_MACHINE_FLAGGED_DOC: &str = "Ledgerline includes a built-in caching layer designed to \
    reduce redundant database round-trips without requiring you to manage a separate cache \
    server for common cases. This page explains the model behind it: how query results get \
    cached, how the cache learns about writes, and the difference between the two cache scopes \
    Ledgerline supports.\n\n\
    **Process-level scope** keeps the cache alive for the lifetime of the worker process, \
    shared across every request that process handles. This catches far more redundant queries, \
    since popular lookups get reused across users and requests, but it requires the \
    invalidation logic described above to actually be correct for your access patterns, \
    including writes made by *other* processes, which process-level caching does not see \
    unless you also configure a shared invalidation channel (Ledgerline supports Redis pub/sub \
    for this; see the Distributed Cache Invalidation reference page).";

fn friction() -> Command {
    Command::cargo_bin("friction").expect("the friction binary builds")
}

fn write_fixture(dir: &Path, name: &str, contents: &str) -> PathBuf {
    let path = dir.join(name);
    fs::write(&path, contents).expect("fixture writes");
    path
}

/// An envelope pack with no bands at all: every metric row reports
/// `in_envelope: None`, which `check`'s exit-code rule treats as passing
/// (`in_envelope.unwrap_or(true)`) — isolates the span-detection half of
/// `check`'s exit code from the metric-envelope half.
fn empty_pack(dir: &Path) -> PathBuf {
    write_fixture(dir, "empty.toml", "")
}

// ---------------------------------------------------------------------
// check
// ---------------------------------------------------------------------

/// `check` on a document with zero detected spans and an envelope pack
/// with no bands (so no metric can ever be "out") exits `0`.
#[test]
fn check_exits_zero_for_a_clean_document() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = write_fixture(dir.path(), "clean.md", CLEAN);
    let pack = empty_pack(dir.path());

    friction()
        .arg("check")
        .arg(&path)
        .args(["--genre", "docs", "--family", "qwen", "--pack"])
        .arg(&pack)
        .assert()
        .success();
}

/// `check` on a document with real detected spans exits `1` and prints
/// every detected frame id to stdout.
#[test]
fn check_exits_one_and_lists_spans_for_a_messy_document() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = write_fixture(dir.path(), "messy.md", MESSY_BLOG);

    friction()
        .arg("check")
        .arg(&path)
        .args(["--genre", "blog", "--family", "qwen"])
        .assert()
        .code(1)
        .stdout(predicate::str::contains("tell:"));
}

/// `check -` reads from stdin, and an omitted `--genre` defaults to
/// `docs` with a note on stderr rather than failing.
#[test]
fn check_reads_stdin_and_defaults_genre_with_a_note() {
    friction()
        .arg("check")
        .arg("-")
        .args(["--family", "qwen"])
        .write_stdin(MESSY_BLOG)
        .assert()
        .code(1)
        .stderr(predicate::str::contains("defaulting to"));
}

/// `check --family` is required: omitting it is a clap usage error (exit
/// `2`), not a silently-defaulted family.
#[test]
fn check_requires_family() {
    friction()
        .arg("check")
        .arg("-")
        .args(["--genre", "docs"])
        .write_stdin(CLEAN)
        .assert()
        .code(2);
}

/// `check --format json` never fails to parse as JSON, and two runs over
/// the same input produce byte-identical stdout, the JSON
/// shape-stability guarantee.
#[test]
fn check_json_output_is_byte_identical_across_runs() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = write_fixture(dir.path(), "messy.md", MESSY_BLOG);

    let run = || {
        friction()
            .arg("check")
            .arg(&path)
            .args(["--genre", "blog", "--family", "qwen", "--format", "json"])
            .output()
            .expect("friction runs")
    };
    let first = run();
    let second = run();
    assert_eq!(first.stdout, second.stdout);

    let value: serde_json::Value =
        serde_json::from_slice(&first.stdout).expect("check --format json prints valid JSON");
    assert_eq!(value["genre"], "blog");
    assert_eq!(value["family"], "qwen");
    assert!(!value["spans"].as_array().unwrap().is_empty());
    // `dms` is the single pooled machine-vs-human report object now, not
    // one entry per family — see `friction_match::dms`'s own module docs
    // for why family attribution was retired.
    assert!(value["dms"]["token_count"].as_u64().unwrap() > 0);
    assert!(value["dms"]["mean_machine"].as_f64().is_some());
    assert!(value["dms"]["mean_human"].as_f64().is_some());
    assert!(value["dms"]["differential"].as_f64().is_some());
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
        .args(["--genre", "blog", "--family", "qwen"])
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
        .args(["--genre", "docs", "--family", "qwen"])
        .assert()
        .code(2)
        .stdout(predicate::str::is_empty())
        .stderr(predicate::str::contains("error"));
}

// ---------------------------------------------------------------------
// fix
// ---------------------------------------------------------------------

/// `fix` writes the fixed text to stdout and a pass summary to stderr,
/// leaving stdout containing only fixed prose (never the summary).
#[test]
fn fix_writes_fixed_text_to_stdout_and_summary_to_stderr() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = write_fixture(dir.path(), "messy.md", MESSY_BLOG);

    let output = friction()
        .arg("fix")
        .arg(&path)
        .output()
        .expect("friction runs");
    assert!(output.status.success());

    let stdout = String::from_utf8(output.stdout).expect("valid UTF-8");
    assert_ne!(
        stdout, MESSY_BLOG,
        "fix must actually change the messy input"
    );
    assert!(
        !stdout.contains("It is important to note that"),
        "the ritual/filler phrasing must be gone from the fixed output"
    );

    let stderr = String::from_utf8(output.stderr).expect("valid UTF-8");
    assert!(stderr.contains("pass(es)"));
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
        .arg("--in-place")
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
        .arg("--in-place")
        .write_stdin(MESSY_BLOG)
        .assert()
        .code(2)
        .stderr(predicate::str::contains("stdin"));
}

/// `fix --suggest` lists remaining held candidates on stderr, alongside
/// (not replacing) the pass summary.
#[test]
fn fix_suggest_lists_remaining_suggestions_on_stderr() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = write_fixture(dir.path(), "messy.md", MESSY_BLOG);

    let output = friction()
        .arg("fix")
        .arg(&path)
        .arg("--suggest")
        .output()
        .expect("friction runs");
    assert!(output.status.success());

    let stderr = String::from_utf8(output.stderr).expect("valid UTF-8");
    assert!(
        stderr.contains("pass(es)"),
        "the summary must still be there"
    );
    assert!(
        stderr.contains("pivot.lvc") || stderr.contains("span.delete"),
        "a held candidate must be listed, got: {stderr}"
    );
}

/// `fix` reports DMS paraphrase spans on stderr without touching stdout:
/// the fixed text is unchanged (this fixture's own repair-engine patch
/// count is zero) and the paraphrase count line is present alongside the
/// pass summary.
#[test]
fn fix_reports_paraphrase_count_without_changing_stdout() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = write_fixture(dir.path(), "flagged.md", DMS_MACHINE_FLAGGED_DOC);

    let output = friction()
        .arg("fix")
        .arg(&path)
        .arg("--suggest")
        .output()
        .expect("friction runs");
    assert!(output.status.success());

    let stdout = String::from_utf8(output.stdout).expect("valid UTF-8");
    assert_eq!(
        stdout, DMS_MACHINE_FLAGGED_DOC,
        "stdout must stay exactly the fixed text, unperturbed by the paraphrase report"
    );

    let stderr = String::from_utf8(output.stderr).expect("valid UTF-8");
    assert!(
        stderr.contains("paraphrase: 2 span(s) flagged for manual rewrite"),
        "the paraphrase count line must be present, got: {stderr}"
    );
    assert!(
        stderr.contains("dms.machine"),
        "--suggest must list the flagged spans' frame id, got: {stderr}"
    );
}

/// `fix --format json --suggest`'s paraphrase array parses as JSON and is
/// byte-identical across two runs over the same input.
#[test]
fn fix_json_paraphrase_array_is_byte_identical_across_runs() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = write_fixture(dir.path(), "flagged.md", DMS_MACHINE_FLAGGED_DOC);

    let run = || {
        friction()
            .arg("fix")
            .arg(&path)
            .args(["--format", "json", "--suggest"])
            .output()
            .expect("friction runs")
    };
    let first = run();
    let second = run();
    assert_eq!(first.stdout, second.stdout);
    assert_eq!(first.stderr, second.stderr);

    // stderr in `--format json` mode is ONE JSON document (the summary
    // with the `suggestions` and `paraphrase` arrays embedded), so an
    // agent consumes the whole stream with a single parse.
    let stderr = String::from_utf8(first.stderr).expect("valid UTF-8");
    let value: serde_json::Value = serde_json::from_str(&stderr)
        .expect("fix --format json stderr parses as one JSON document");
    assert_eq!(value["paraphrase_count"], 2);
    let rows = value["paraphrase"]
        .as_array()
        .expect("paraphrase array is embedded in the summary document");
    assert_eq!(rows.len(), 2);
    for row in rows {
        assert_eq!(row["frame_id"], "dms.machine");
        assert!(row["score"].as_i64().unwrap() > 0);
        assert!(!row["snippet"].as_str().unwrap().is_empty());
    }
}

/// `fix -` reads from stdin and writes the fixed text to stdout.
#[test]
fn fix_reads_stdin() {
    let output = friction()
        .arg("fix")
        .arg("-")
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
        .args(["--format", "sarif"])
        .assert()
        .code(2)
        .stderr(predicate::str::contains("sarif"));
}

// ---------------------------------------------------------------------
// explain
// ---------------------------------------------------------------------

/// `explain` prints a per-pass fired/held report, never the fixed text
/// itself.
#[test]
fn explain_prints_pass_report_without_fixed_text() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = write_fixture(dir.path(), "messy.md", MESSY_BLOG);

    let output = friction()
        .arg("explain")
        .arg(&path)
        .output()
        .expect("friction runs");
    assert!(output.status.success());

    let stdout = String::from_utf8(output.stdout).expect("valid UTF-8");
    assert!(stdout.contains("pass 1:"));
    assert!(stdout.contains("operation(s) fired"));
    assert!(
        !stdout.contains("It is important to note that"),
        "explain must never print the fixed (or original) document text"
    );
}

/// `explain --format json` prints a `passes`/`patches_applied` shaped
/// report, stable across two runs.
#[test]
fn explain_json_output_is_byte_identical_across_runs() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = write_fixture(dir.path(), "messy.md", MESSY_BLOG);

    let run = || {
        friction()
            .arg("explain")
            .arg(&path)
            .args(["--format", "json"])
            .output()
            .expect("friction runs")
    };
    let first = run();
    let second = run();
    assert_eq!(first.stdout, second.stdout);

    let value: serde_json::Value =
        serde_json::from_slice(&first.stdout).expect("explain --format json prints valid JSON");
    assert!(!value["passes"].as_array().unwrap().is_empty());
    assert!(value["patches_applied"].as_u64().unwrap() >= 1);
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
        .args(["--format", "sarif"])
        .assert()
        .code(2);
}
