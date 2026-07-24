//! Idempotence canary: `fix_document(fix_document(fix_document(x))) ==
//! fix_document(fix_document(x))`, byte-for-byte, over the fixture inputs
//! plus a deterministic sample of the TRAIN-split corpus.
//!
//! This deliberately re-invokes `Engine::fix_document` (which already
//! runs its own internal, bounded two-pass loop) three separate times
//! over its own prior output — the stronger interpretation of "a third
//! pass must be a no-op": it must hold not just within one
//! `edit_document` call's own bounded loop, but treating the engine as a
//! black box a caller might re-invoke, exactly `friction fix`'s real
//! usage pattern (a user re-running `friction fix` on already-fixed
//! output).

use std::path::{Path, PathBuf};

use friction_edit::Engine;

fn corpus_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus")
}

/// Fixture inputs this crate's own accept/reject fixtures exercise,
/// embedded directly (no dependency on `friction-harness` — see this
/// crate's own `B.0` dependency graph, which does not include it).
fn fixture_inputs() -> Vec<&'static str> {
    vec![
        "This guide will walk you through installing the tool, performing a basic scan, and \
         previewing matches before making any deletions. It is important to note that the \
         configuration file simply needs to be placed in the root directory of your project. \
         By following these steps, you can quickly and easily configure the application to \
         suit your needs.",
        "This guide will walk you through configuring the backup agent for your staging \
         environment. It is important to note that the agent performs validation of the \
         configuration file before each run. Once validation succeeds, the agent conducts an \
         analysis of the snapshot catalog and simply uploads any missing segments. By \
         following these steps, you can quickly and easily verify that your backups are \
         consistent. If you have any questions or require further assistance, please reach \
         out to our support team.",
        "The system performs an initialization of the database.",
        "The tool performs validation of the configuration file before applying changes.",
        "Before deployment, the script conducts an analysis of the dependency graph.",
        "You should make a decision soon.",
        "The installer performed a comparison of the two versions.",
        "However, after careful consideration and experimentation, we made the decision to \
         switch from Gerrit to GitHub pull requests.",
        "Run the scanner from the project root. Results stream in as they are found, and \
         nothing is deleted without confirmation.",
        "Obtain the approval of the committee before merging.",
        "The migration performs a full initialization of the database.",
        "The tool performs several initializations of the database during testing.",
        "Create an index of the most common queries.",
        "The wizard facilitates the integration of the plugin with your DAW.",
        "An initialization of the database is performed by the system.",
        "When you install audio plugins on a Mac, vendors typically install multiple versions \
         of the same plugin.",
        "It's worth noting that most DAWs scan all these plugins on startup.",
        "Worse, plugin uninstallers are often unreliable or missing entirely, leaving orphaned \
         files scattered across your Library.",
        "Initially, the biggest hurdle was simply getting used to it.",
        "This bloat matters because most DAWs scan all these plugins on startup, leading to \
         slower load times.",
    ]
}

/// A fixed, deterministic sample of up to 50 TRAIN-split human+llm corpus
/// documents' text: manifest records sorted by id, then every Nth taken
/// so the sample spans the whole split rather than clustering at the
/// front — the same id-sorted-then-take-every-Nth selection
/// `friction-apply`'s own idempotence sweep precedent uses, not random
/// sampling, so this test's own sample is itself deterministic.
fn sample_of_train_corpus_docs(limit: usize) -> Vec<String> {
    let root = corpus_root();
    let manifest_path = root.join("manifest.jsonl");
    let Ok(Some(mut records)) = corpus_tool::manifest::read_manifest(&manifest_path) else {
        return Vec::new();
    };
    records.sort_by(|a, b| a.id.cmp(&b.id));
    let train: Vec<_> = records
        .into_iter()
        .filter(|r| r.split == Some(corpus_tool::manifest::Split::Train))
        .collect();

    if train.is_empty() {
        return Vec::new();
    }
    let stride = (train.len() / limit.max(1)).max(1);

    let mut out = Vec::with_capacity(limit);
    for record in train.iter().step_by(stride) {
        if out.len() >= limit {
            break;
        }
        let path = root.join(corpus_tool::corpus_layout::relpath(record));
        if let Ok(text) = std::fs::read_to_string(&path) {
            out.push(text);
        }
    }
    out
}

#[test]
fn third_pass_is_byte_identical_to_second_pass() {
    let engine = Engine::new().expect("engine must load");

    let mut inputs: Vec<String> = fixture_inputs().into_iter().map(String::from).collect();
    inputs.extend(sample_of_train_corpus_docs(50));

    assert!(
        inputs.len() > 15,
        "expected at least the fixture inputs to be present, got {}",
        inputs.len()
    );

    for input in &inputs {
        let (pass1, _) = engine
            .fix_document(input)
            .unwrap_or_else(|e| panic!("pass 1 failed: {e}\ninput: {input:?}"));
        let (pass2, _) = engine
            .fix_document(&pass1)
            .unwrap_or_else(|e| panic!("pass 2 failed: {e}\ninput: {pass1:?}"));
        let (pass3, _) = engine
            .fix_document(&pass2)
            .unwrap_or_else(|e| panic!("pass 3 failed: {e}\ninput: {pass2:?}"));

        assert_eq!(
            pass2, pass3,
            "third full engine run must be a no-op for input: {input:?}"
        );
    }
}
