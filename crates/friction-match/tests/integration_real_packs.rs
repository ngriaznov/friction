//! End-to-end integration against the real embedded packs and real NLP
//! backends: constructs a `MatchEngine`, scans a small real document, and
//! asserts the result is well-formed.

mod support;

use friction_core::span::Spanned;
use friction_match::Channel;
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

/// `"color harmony"` is a real Wikipedia article title (verified against
/// the raw `jargon-attest-v1` input data), and — since jargon-v1.toml's
/// `attested_exceptions` pruning — is no longer in the hand-curated TOML
/// list either. Suppression here can only be the embedded
/// `jargon-attest-v1` `BinaryFuse8` filter (SYNTHESIS.md §4), not the
/// TOML override layer.
#[test]
fn color_harmony_is_suppressed_by_the_embedded_attestation_filter() {
    assert!(
        !friction_packs::JARGON
            .pack
            .is_attested_exception("color harmony"),
        "color harmony was pruned from jargon-v1.toml's attested_exceptions \
         precisely because the filter already covers it"
    );
    assert!(
        friction_packs::JARGON_ATTEST.is_attested("color harmony"),
        "color harmony is a real Wikipedia title and must be in the embedded filter"
    );

    let source = "Good UI design depends on color harmony across every surface.";
    let document = friction_parse::parse(source).expect("valid markdown parses");
    let report = engine().scan(&document).expect("scan runs to completion");
    assert!(
        report
            .spans
            .iter()
            .all(|span| span.channel != Channel::Jargon),
        "color harmony must not surface a jargon.metaphor span: {:?}",
        report.spans
    );
}

/// A genuinely invented pseudo-jargon compound — never a Wikipedia title,
/// never an `OpenAlex` topic, never a TOML exception — still fires. The
/// filter only ever suppresses; it never invents a suppression for a
/// compound that isn't actually attested anywhere.
#[test]
fn celestial_tapestry_is_not_a_title_so_it_still_fires() {
    assert!(!friction_packs::JARGON_ATTEST.is_attested("celestial tapestry"));

    let source = "The report imagines a celestial tapestry of interconnected services.";
    let document = friction_parse::parse(source).expect("valid markdown parses");
    let report = engine().scan(&document).expect("scan runs to completion");
    assert!(
        report
            .spans
            .iter()
            .any(|span| span.channel == Channel::Jargon),
        "celestial tapestry must still surface a jargon.metaphor span: {:?}",
        report.spans
    );
}
