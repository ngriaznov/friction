//! End-to-end integration against the real embedded packs and real NLP
//! backends: constructs a `MatchEngine`, scans a small real document, and
//! asserts the result is well-formed.

mod support;

use friction_core::span::Spanned;
use friction_packs::ModelFamily;
use support::engine;

const SOURCE: &str = "\
# Backup agent

This guide will walk you through configuring the backup agent for your \
staging environment.

It is important to note that the agent performs validation of the \
configuration file before each run.

- Prior to release, please review the changelog.
- The team conducts an analysis of the snapshot catalog regularly.
";

#[test]
fn scanning_a_real_document_against_the_embedded_packs_runs_to_completion() {
    assert!(
        !ModelFamily::ALL.is_empty(),
        "at least one model family must be defined"
    );
    assert!(
        friction_packs::DMS
            .pack
            .family_sam(ModelFamily::Qwen)
            .is_some(),
        "the embedded DMS pack must define a Qwen stream"
    );

    let document = friction_parse::parse(SOURCE).expect("valid markdown parses");
    let report = engine().scan(&document).expect("scan runs to completion");

    // The document-level DMS report covers every family the pack defines.
    assert_eq!(report.dms.families.len(), ModelFamily::ALL.len());
    assert_eq!(report.dms.target_family, ModelFamily::Qwen);

    // Every span's range round-trips through the source: it slices
    // cleanly and addresses non-empty text.
    for span in &report.spans {
        let text = document
            .text(&span.range())
            .expect("every emitted span must slice the original source cleanly");
        assert!(!text.is_empty(), "span {span:?} sliced to empty text");
    }

    // The heading is excluded from prose scope, so no span may reach into
    // it.
    let heading_end = SOURCE.find("This guide").expect("heading precedes prose");
    for span in &report.spans {
        assert!(
            span.range.start >= heading_end,
            "span {span:?} reaches into the heading"
        );
    }
}
