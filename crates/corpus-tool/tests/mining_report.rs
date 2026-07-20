//! Guards against `corpus/MINING.md` drifting from the exact `mine`
//! invocation documented as its own generation command, and from the
//! provenance command `crates/friction-packs/packs/mined-ngrams-v1.toml`
//! claims it was curated from.
//!
//! This exercises the real, committed `corpus/` directory (not a tempdir
//! fixture) via `corpus_tool::commands::mine`, so a checked-in report
//! that was regenerated with the wrong flags (or not regenerated after a
//! corpus/scoring change) fails `cargo test` directly instead of silently
//! drifting out of sync with what the pack's curation rationale claims a
//! reviewer can see in it.

use std::path::Path;

use corpus_tool::commands::mine::{self, Args, NgramOrderArg};

/// The exact flags `crates/friction-packs/packs/mined-ngrams-v1.toml`'s
/// provenance comment claims `corpus/MINING.md` was produced with:
/// `corpus-tool mine --n all --top 120 --min-count 5 --report <path>`.
/// If that provenance comment ever changes, this constant (and the
/// regenerated `corpus/MINING.md`) must change with it.
const DOCUMENTED_TOP: usize = 120;
const DOCUMENTED_MIN_COUNT: u64 = 5;

fn repo_root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("resolve repo root")
}

/// Re-running `mine` against the real train-split corpus with the flags
/// the pack claims as provenance reproduces the committed
/// `corpus/MINING.md` byte-for-byte, twice in a row (the module doc
/// comment's own determinism claim). A mismatch means either the
/// checked-in report was generated with different flags than documented
/// (e.g. left at `--top`'s default instead of the documented `--top
/// 120`), or the corpus/scoring changed underneath it without a
/// regeneration.
#[test]
fn mining_report_matches_documented_provenance_command() {
    let root = repo_root();
    let corpus_dir = root.join("corpus");
    let committed_report = root.join("corpus/MINING.md");

    assert!(
        committed_report.is_file(),
        "expected corpus/MINING.md to exist at {}",
        committed_report.display()
    );
    let committed = std::fs::read_to_string(&committed_report).expect("read corpus/MINING.md");

    let tmp = tempfile::tempdir().expect("tempdir");
    let regenerated_path = tmp.path().join("MINING.md");
    let args = Args {
        corpus_dir,
        n: NgramOrderArg::All,
        top: DOCUMENTED_TOP,
        min_count: DOCUMENTED_MIN_COUNT,
        report: regenerated_path.clone(),
    };
    mine::run(&args).expect("mine against the real corpus");
    let regenerated = std::fs::read_to_string(&regenerated_path).expect("read regenerated report");

    assert_eq!(
        regenerated, committed,
        "corpus/MINING.md does not match `corpus-tool mine --n all --top {DOCUMENTED_TOP} \
         --min-count {DOCUMENTED_MIN_COUNT} --report <path>` (the provenance command documented \
         in crates/friction-packs/packs/mined-ngrams-v1.toml) — regenerate it with that exact \
         command"
    );

    // Determinism: a second run against the same corpus is byte-identical
    // to the first, not just to the committed file.
    let rerun_path = tmp.path().join("MINING-rerun.md");
    mine::run(&Args {
        report: rerun_path.clone(),
        ..args
    })
    .expect("re-run mine against the real corpus");
    let rerun = std::fs::read_to_string(&rerun_path).expect("read rerun report");
    assert_eq!(
        rerun, regenerated,
        "mine produced different output on a second run against the same unchanged corpus"
    );
}

/// The two entries the pack's curation rationale names as visible only in
/// the raw top-120 list (and deliberately dropped from the curated pack)
/// are actually present in `corpus/MINING.md` at the documented `--top
/// 120`. This is the concrete, human-checkable half of the provenance
/// claim the test above enforces structurally.
#[test]
fn mining_report_contains_entries_named_in_pack_curation_rationale() {
    let root = repo_root();
    let committed =
        std::fs::read_to_string(root.join("corpus/MINING.md")).expect("read corpus/MINING.md");

    for needle in ["| wharfgate |", "| route flapping |"] {
        assert!(
            committed.contains(needle),
            "corpus/MINING.md is missing {needle:?}, which \
             crates/friction-packs/packs/mined-ngrams-v1.toml's curation rationale names as a \
             raw top-120 entry a reviewer should be able to see and verify was dropped"
        );
    }
}
