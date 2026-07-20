//! End-to-end snapshot suite: pins the full byte output of `friction
//! check --format json`, `friction check --format text --no-color`,
//! `friction explain`, and `friction fix` (stdout only) over a fixed,
//! deterministically-selected 20-document corpus sample.
//!
//! # Selection rule
//!
//! `corpus/manifest.jsonl` has five genres (`blog`, `docs`, `email`,
//! `forum`, `readme`) and two classes (`human`, `llm`). For each of the
//! five genres, this suite picks the two lexicographically-first
//! (by the manifest's `id` field — a random 16-hex-digit string, so
//! "lexicographically-first" is an arbitrary but fully reproducible
//! tiebreak, not a meaningful ordering) document ids with `split =
//! "train"` for `class = "human"`, and the same two for `class = "llm"`:
//! 5 genres x (2 human + 2 llm) = 20 documents. [`SELECTED`] is that list,
//! resolved to each document's on-disk path per corpus-tool's own
//! `corpus_layout::relpath` rule (a document is filed under
//! `quarantine/<genre>/` instead of `<class>/<genre>/` when its manifest
//! `license` is CC-BY-SA — three of the twenty selected ids land there:
//! one `email` human doc, both `forum` human docs).
//!
//! [`SELECTED`] is a hand-derived, pinned snapshot of applying that rule
//! to the manifest once — not re-derived at test time — since the
//! manifest is a stable, committed corpus artifact and re-deriving the
//! selection on every run would add a `serde_json`/file-IO dependency to
//! this test for no additional coverage (a manifest edit that changed the
//! selection would need this list updated by hand regardless, the same
//! way any other golden-fixture input change does).
//!
//! # Running
//!
//! `INSTA_UPDATE=always cargo test -p friction-cli --test snapshot`
//! writes/refreshes every `.snap` file under `tests/snapshots/`. A plain
//! `cargo test -p friction-cli --test snapshot` afterwards must then pass
//! with no changes reported — insta's default comparison is byte-exact —
//! which is this suite's determinism guarantee: two consecutive full runs
//! produce byte-identical output for every one of the 80 pinned snapshots
//! (20 documents x 4 output kinds), or the suite fails.
//!
//! # Why every command runs from the workspace root
//!
//! `check --format text`'s diagnostics header and `check --format
//! sarif`'s `artifactLocation.uri` both embed the input path verbatim
//! (see `crate::common::display_path`) — never resolved to an absolute
//! path. Running the binary with its working directory set to the
//! workspace root and passing each document's already-relative
//! `corpus/...` path keeps that embedded path identical regardless of
//! where this repository happens to be checked out, which byte-identical
//! snapshot output across machines requires.

use std::path::{Path, PathBuf};

use assert_cmd::Command;

/// One corpus document selected into the suite: its manifest id (used
/// only to build a readable snapshot name), the `--genre` value its
/// folder implies, and its path relative to the workspace root.
struct SelectedDoc {
    id: &'static str,
    genre: &'static str,
    relpath: &'static str,
}

/// See the module docs for the selection rule this list is a pinned
/// result of, in `(genre, class)` order matching `corpus/manifest.jsonl`'s
/// own genre set, human ids before llm ids within each genre, and each
/// pair in ascending id order.
const SELECTED: &[SelectedDoc] = &[
    SelectedDoc {
        id: "016b54b46d29feb8",
        genre: "blog",
        relpath: "corpus/human/blog/016b54b46d29feb8.md",
    },
    SelectedDoc {
        id: "0589bf2932eba95a",
        genre: "blog",
        relpath: "corpus/human/blog/0589bf2932eba95a.md",
    },
    SelectedDoc {
        id: "152f7fa1159f4910",
        genre: "blog",
        relpath: "corpus/llm/blog/152f7fa1159f4910.md",
    },
    SelectedDoc {
        id: "19f12335d308d0e0",
        genre: "blog",
        relpath: "corpus/llm/blog/19f12335d308d0e0.md",
    },
    SelectedDoc {
        id: "01ec8967989205a2",
        genre: "docs",
        relpath: "corpus/human/docs/01ec8967989205a2.md",
    },
    SelectedDoc {
        id: "08d07d7b04ccd440",
        genre: "docs",
        relpath: "corpus/human/docs/08d07d7b04ccd440.md",
    },
    SelectedDoc {
        id: "00533d2e3a398154",
        genre: "docs",
        relpath: "corpus/llm/docs/00533d2e3a398154.md",
    },
    SelectedDoc {
        id: "0a0197006e9ca159",
        genre: "docs",
        relpath: "corpus/llm/docs/0a0197006e9ca159.md",
    },
    // CC-BY-SA licensed: filed under quarantine/, not human/ (see module
    // docs).
    SelectedDoc {
        id: "026f8f57c3920652",
        genre: "email",
        relpath: "corpus/quarantine/email/026f8f57c3920652.md",
    },
    SelectedDoc {
        id: "0ac47db2525fd485",
        genre: "email",
        relpath: "corpus/human/email/0ac47db2525fd485.md",
    },
    SelectedDoc {
        id: "062a9d6f268e8994",
        genre: "email",
        relpath: "corpus/llm/email/062a9d6f268e8994.md",
    },
    SelectedDoc {
        id: "29ff22ba7de18cf6",
        genre: "email",
        relpath: "corpus/llm/email/29ff22ba7de18cf6.md",
    },
    // Both selected forum/human ids are CC-BY-SA (StackExchange
    // provenance) and so are filed under quarantine/, not human/.
    SelectedDoc {
        id: "0710f627ec229d97",
        genre: "forum",
        relpath: "corpus/quarantine/forum/0710f627ec229d97.md",
    },
    SelectedDoc {
        id: "0a3d4030ac673c91",
        genre: "forum",
        relpath: "corpus/quarantine/forum/0a3d4030ac673c91.md",
    },
    SelectedDoc {
        id: "0720879ca70251ed",
        genre: "forum",
        relpath: "corpus/llm/forum/0720879ca70251ed.md",
    },
    SelectedDoc {
        id: "137c3759df60fa49",
        genre: "forum",
        relpath: "corpus/llm/forum/137c3759df60fa49.md",
    },
    SelectedDoc {
        id: "0217ca71eb7abfce",
        genre: "readme",
        relpath: "corpus/human/readme/0217ca71eb7abfce.md",
    },
    SelectedDoc {
        id: "05f0fbd8371252a5",
        genre: "readme",
        relpath: "corpus/human/readme/05f0fbd8371252a5.md",
    },
    SelectedDoc {
        id: "04c4c78e378a7c8f",
        genre: "readme",
        relpath: "corpus/llm/readme/04c4c78e378a7c8f.md",
    },
    SelectedDoc {
        id: "0859e6e20fb1cf84",
        genre: "readme",
        relpath: "corpus/llm/readme/0859e6e20fb1cf84.md",
    },
];

/// The workspace root, resolved once from this crate's own manifest
/// directory (`crates/friction-cli`) two levels up — used only as the
/// spawned binary's working directory (see the module docs on why every
/// command runs from here), never embedded in any snapshot itself.
fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("the workspace root exists two levels up from CARGO_MANIFEST_DIR")
}

/// The built `friction` binary, with its working directory set to the
/// workspace root so every document's already-relative `corpus/...` path
/// resolves, and appears verbatim in any output that embeds it.
fn friction() -> Command {
    let mut cmd = Command::cargo_bin("friction").expect("the friction binary builds");
    cmd.current_dir(workspace_root());
    cmd
}

/// Runs `friction <args...>` against `doc.relpath`, returning stdout as
/// UTF-8. Exit status is deliberately not asserted here: `check`'s exit
/// code is a function of the document's own content (findings and
/// envelope membership), not a success/failure signal for this suite.
fn run_stdout(doc: &SelectedDoc, args: &[&str]) -> String {
    let output = friction()
        .args(args)
        .arg(doc.relpath)
        .args(["--genre", doc.genre])
        .output()
        .expect("friction runs");
    String::from_utf8(output.stdout).expect("stdout is valid UTF-8")
}

/// A snapshot name unique per document and output kind, stable regardless
/// of iteration order.
fn snapshot_name(kind: &str, doc: &SelectedDoc) -> String {
    format!(
        "{kind}__{}_{}_{}",
        doc.genre,
        class_from_relpath(doc),
        doc.id
    )
}

/// The corpus class folder a document's `relpath` was resolved under
/// (`human`, `llm`, or `quarantine`) — folded into the snapshot name so a
/// quarantined human doc's snapshot is still clearly distinguishable from
/// an `llm` one at the same genre.
fn class_from_relpath(doc: &SelectedDoc) -> &'static str {
    if doc.relpath.starts_with("corpus/human/") {
        "human"
    } else if doc.relpath.starts_with("corpus/llm/") {
        "llm"
    } else {
        "quarantine"
    }
}

/// `check --format json`, snapshotted for every selected document: the
/// `CheckReport` shape (genre, metric rows, findings) is stable JSON with
/// no embedded path, so this is directly cross-machine portable.
#[test]
fn check_json_snapshots() {
    for doc in SELECTED {
        let stdout = run_stdout(doc, &["check", "--format", "json"]);
        insta::assert_snapshot!(snapshot_name("check_json", doc), stdout);
    }
}

/// `check --format text --no-color`, snapshotted for every selected
/// document: `--no-color` makes the miette diagnostic rendering
/// deterministic regardless of whether the test harness's stdout happens
/// to be a terminal (see `crate::diagnostics`'s own module docs).
#[test]
fn check_text_no_color_snapshots() {
    for doc in SELECTED {
        let stdout = run_stdout(doc, &["check", "--format", "text", "--no-color"]);
        insta::assert_snapshot!(snapshot_name("check_text", doc), stdout);
    }
}

/// `explain` (default `--format text`), snapshotted for every selected
/// document: the before/after metric table, plan schedule, and fixpoint
/// summary — never the document text itself.
#[test]
fn explain_snapshots() {
    for doc in SELECTED {
        let stdout = run_stdout(doc, &["explain"]);
        insta::assert_snapshot!(snapshot_name("explain", doc), stdout);
    }
}

/// `fix` (default `--format text`), snapshotted for every selected
/// document: stdout only (the fixed document text) — the round summary
/// and any `--suggest` output go to stderr and are not part of this
/// snapshot, matching `fix`'s own documented stdout/stderr split.
#[test]
fn fix_stdout_snapshots() {
    for doc in SELECTED {
        let stdout = run_stdout(doc, &["fix"]);
        insta::assert_snapshot!(snapshot_name("fix", doc), stdout);
    }
}

/// Every selected document actually resolves to a real file on disk — a
/// broken entry in [`SELECTED`] would otherwise silently make the other
/// four tests fail with an unhelpful "friction runs" panic instead of
/// pointing at the missing path.
#[test]
fn selected_docs_exist_on_disk() {
    let root = workspace_root();
    for doc in SELECTED {
        let path = root.join(doc.relpath);
        assert!(
            path.is_file(),
            "selected doc {} ({}) does not exist at {}",
            doc.id,
            doc.genre,
            path.display()
        );
    }
}
