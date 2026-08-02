//! Determinism: identical input plus identical pack always yields
//! byte-identical spans.

mod support;

use friction_match::MatchEngine;
use friction_nlp::{PerceptronTagger, SrxSegmenter};
use friction_packs::ModelFamily;
use support::engine;

const SOURCE: &str = "\
This guide will walk you through configuring the backup agent for your \
staging environment. It is important to note that the agent performs \
validation of the configuration file before each run. Once validation \
succeeds, the agent conducts an analysis of the snapshot catalog and \
simply uploads any missing segments.
";

#[test]
fn scanning_the_same_document_twice_produces_byte_identical_reports() {
    let document = friction_parse::parse(SOURCE).expect("valid markdown parses");
    let engine = engine();

    let first = engine.scan(&document).expect("first scan succeeds");
    let second = engine.scan(&document).expect("second scan succeeds");

    assert_eq!(first.spans, second.spans);
    assert_eq!(
        first.dms.machine.token_count,
        second.dms.machine.token_count
    );
    assert!(
        (first.dms.machine.differential - second.dms.machine.differential).abs() < f64::EPSILON
    );
}

#[test]
fn two_independently_constructed_engines_from_the_same_pack_bytes_agree() {
    let document = friction_parse::parse(SOURCE).expect("valid markdown parses");

    let tagger_a = PerceptronTagger::new().expect("tagger loads");
    let segmenter_a = SrxSegmenter::new();
    let engine_a = MatchEngine::new(
        &friction_packs::INVENTORY.pack,
        &friction_packs::DMS.pack,
        &friction_packs::JARGON.pack,
        &friction_packs::JARGON_ATTEST,
        &friction_packs::HUMAN_EVIDENCE,
        ModelFamily::Qwen,
        &tagger_a,
        &segmenter_a,
    )
    .expect("engine builds");

    let tagger_b = PerceptronTagger::new().expect("tagger loads");
    let segmenter_b = SrxSegmenter::new();
    let engine_b = MatchEngine::new(
        &friction_packs::INVENTORY.pack,
        &friction_packs::DMS.pack,
        &friction_packs::JARGON.pack,
        &friction_packs::JARGON_ATTEST,
        &friction_packs::HUMAN_EVIDENCE,
        ModelFamily::Qwen,
        &tagger_b,
        &segmenter_b,
    )
    .expect("engine builds");

    let report_a = engine_a.scan(&document).expect("scan succeeds");
    let report_b = engine_b.scan(&document).expect("scan succeeds");

    assert_eq!(report_a.spans, report_b.spans);
}
