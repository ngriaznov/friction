//! The anaphoric-shell-demonstrative diagnostic-only entries: a bare
//! "this" standing in for an unnamed referent, decorated with an
//! unearned verb ("This ensures ..."). Naming what "this" refers to is
//! human work no mechanical rewrite can do, so these entries are
//! `repair = "diagnostic_only"` (`span.this_ensures` in
//! `inventory-en-v1.toml`) — friction marks the span and never edits.
//! Mirrors `diagnostic_only.rs`'s span idiom for `span.positive_contrast`.

use friction_edit::Engine;

fn engine() -> Engine {
    Engine::new().expect("embedded packs and models load")
}

/// `span.this_ensures` marks the anaphoric shell without rewriting it —
/// the sentence must survive byte-identical, and the finding must name
/// the span id.
#[test]
fn diagnostic_only_this_ensures_marks_without_rewriting() {
    let engine = engine();
    for source in [
        "The cache store is updated on every write. This ensures that cached \
         results are not returned for queries that rely on outdated data.\n",
        "Select the model from the dropdown menu under \"Device.\" This ensures \
         compatibility with the correct firmware settings.\n",
    ] {
        let (fixed, report) = engine.fix_document(source).expect("engine runs");
        assert_eq!(fixed, source, "diagnostic spans must never edit");
        assert!(
            report
                .passes
                .iter()
                .flat_map(|p| &p.held)
                .any(|f| f.message.contains("span.this_ensures")),
            "the span must surface as a finding: {source:?}"
        );
    }
}
