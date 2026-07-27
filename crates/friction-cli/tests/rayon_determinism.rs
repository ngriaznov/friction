//! Byte-determinism under rayon: `friction fix`/`friction check` over a
//! real corpus document must produce identical output regardless of how
//! many worker threads rayon's per-sentence parallelism (the register
//! pass's sentence-context build, `fix`'s DMS-family/frame/jargon
//! paraphrase scan, `check`'s six detection channels plus the DMS
//! document-report's per-family walk — see `friction-edit::register`,
//! `friction-match`, and `friction-cli::fix`'s own docs) actually runs
//! on.
//!
//! Each combination below is a genuinely fresh child process (via
//! `assert_cmd`), not an in-process thread-pool rebuild: rayon's global
//! pool reads `RAYON_NUM_THREADS` once, lazily, on its first use, and a
//! single `cargo test` binary runs every `#[test]` in one process — a
//! pool some earlier test already initialized would make a later
//! `std::env::set_var` a no-op. A fresh process per combination sidesteps
//! that entirely and is also the most literal reading of "run with
//! `RAYON_NUM_THREADS=1,2,8`": the real thing a user sets before invoking
//! the built binary.

use std::path::{Path, PathBuf};
use std::process::Output;

use assert_cmd::Command;

/// Thread counts this suite pins rayon's global pool to, one child
/// process per count — 1 (no parallelism at all, the sequential-code
/// baseline every other count must match), 2 (the smallest genuine
/// parallel split), and 8 (comfortably above this corpus document's own
/// sentence/family/channel counts, so every parallel site's tasks are
/// oversubscribed across threads at least once).
const THREAD_COUNTS: [&str; 3] = ["1", "2", "8"];

/// How many times each thread count reruns the same command — repetition
/// alone (same thread count, different scheduling luck each run) is part
/// of the determinism claim, not just variation across counts.
const RUNS_PER_THREAD_COUNT: usize = 3;

/// A real, multi-sentence corpus document — the same file
/// (`tests/snapshot.rs`'s `SELECTED[0]`) the end-to-end snapshot suite
/// already pins, so this determinism check exercises the same real prose
/// that suite's byte-identical-output guarantee already trusts, not a
/// synthetic fixture too small to actually split across threads.
const DOC_RELPATH: &str = "corpus/human/blog/016b54b46d29feb8.md";

/// The workspace root, resolved the same way `tests/snapshot.rs` does —
/// used only as the spawned binary's working directory so `DOC_RELPATH`
/// resolves regardless of where this repository happens to be checked
/// out.
fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("the workspace root exists two levels up from CARGO_MANIFEST_DIR")
}

/// The built `friction` binary, working directory set to the workspace
/// root, `RAYON_NUM_THREADS` set to `threads` for this one child process.
fn friction(threads: &str) -> Command {
    let mut cmd = Command::cargo_bin("friction").expect("the friction binary builds");
    cmd.current_dir(workspace_root());
    cmd.env("RAYON_NUM_THREADS", threads);
    cmd
}

/// Runs `friction <args...> DOC_RELPATH` under `threads` worker threads,
/// returning raw stdout+stderr bytes concatenated (stdout carries the
/// fixed/reported text, stderr carries `fix`'s pass/paraphrase summary
/// and `--suggest` detail — both are downstream of the parallelized scans
/// this test guards, so both must hold byte-identical too).
fn run(threads: &str, args: &[&str]) -> Vec<u8> {
    let output: Output = friction(threads)
        .args(args)
        .arg(DOC_RELPATH)
        .output()
        .expect("friction runs");
    let mut bytes = output.stdout;
    bytes.extend_from_slice(&output.stderr);
    bytes
}

/// Runs `args` over every thread count in [`THREAD_COUNTS`],
/// [`RUNS_PER_THREAD_COUNT`] times each, and asserts every single run
/// produced byte-identical output to the very first one — the hard gate
/// this workspace's rayon rollout is pinned on: identical output bytes
/// for identical input on every run and every thread count.
fn assert_identical_across_thread_counts_and_reruns(command_label: &str, args: &[&str]) {
    let mut baseline: Option<Vec<u8>> = None;
    for threads in THREAD_COUNTS {
        for run_index in 0..RUNS_PER_THREAD_COUNT {
            let bytes = run(threads, args);
            match &baseline {
                None => baseline = Some(bytes),
                Some(expected) => {
                    assert_eq!(
                        &bytes, expected,
                        "`friction {command_label}` on {DOC_RELPATH} diverged with \
                         RAYON_NUM_THREADS={threads} (run {run_index}) from the RAYON_NUM_THREADS=\
                         {}, run 0 baseline",
                        THREAD_COUNTS[0]
                    );
                }
            }
        }
    }
}

/// `friction fix`: exercises the register pass's rayon-parallel
/// sentence-context build (`friction_edit::register::build_sentence_contexts`)
/// and the paraphrase scan's concurrent DMS-family/frame/jargon tasks
/// (`friction_cli::fix::scan_paraphrase_spans`, itself calling
/// `friction_match::jargon::scan_units`'s own per-sentence `par_iter`).
#[test]
fn fix_is_byte_identical_across_thread_counts() {
    assert_identical_across_thread_counts_and_reruns("fix", &["fix", "--suggest"]);
}

/// `friction check`: exercises `friction_match::MatchEngine::scan`'s
/// seven concurrent `rayon::scope` tasks (six detection channels plus the
/// DMS document-report, itself a `par_iter` over every family).
#[test]
fn check_is_byte_identical_across_thread_counts() {
    assert_identical_across_thread_counts_and_reruns(
        "check",
        &["check", "--format", "json", "--genre", "blog", "--family", "qwen"],
    );
}
