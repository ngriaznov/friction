//! Exercises every channel against the shared literal research fixtures
//! (`docs/research/fixtures.json`, via `friction-harness::fixtures`).

mod support;

use friction_harness::fixtures::FIXTURES;
use friction_match::Channel;
use support::engine;

fn fixture_input(id: &str) -> &'static str {
    FIXTURES
        .accept
        .iter()
        .find(|f| f.id == id)
        .and_then(|f| f.input.as_deref())
        .unwrap_or_else(|| panic!("fixture {id:?} has an input"))
}

#[test]
fn composed_four_operation_paragraph_input_detects_all_four_tell_families() {
    let input = fixture_input("composed_four_operation_paragraph");
    let document = friction_parse::parse(input).expect("valid markdown parses");
    let report = engine().scan(&document).expect("scan succeeds");

    assert!(
        report.spans.iter().any(|s| s.channel == Channel::Literal),
        "expected at least one Literal span, got {:?}",
        report.spans
    );

    assert!(
        report
            .spans
            .iter()
            .any(|s| s.channel == Channel::Lvc && &*s.frame_id == "lvc.validation"),
        "expected an Lvc span for \"performs validation of\" -> validates, got {:?}",
        report.spans
    );
    assert!(
        report
            .spans
            .iter()
            .any(|s| s.channel == Channel::Lvc && &*s.frame_id == "lvc.analysis"),
        "expected an Lvc span for \"conducts an analysis of\" -> analyzes, got {:?}",
        report.spans
    );

    assert!(
        report.dms.machine.differential > 0.0,
        "expected a positive pooled DMS document-level differential, got {:?}",
        report.dms.machine
    );
}

#[test]
fn near_noop_clean_text_input_produces_zero_spans() {
    let input = fixture_input("near_noop_clean_text");
    let document = friction_parse::parse(input).expect("valid markdown parses");
    let report = engine().scan(&document).expect("scan succeeds");
    assert!(
        report.spans.is_empty(),
        "expected zero spans for the near-no-op fixture, got {:?}",
        report.spans
    );
}
