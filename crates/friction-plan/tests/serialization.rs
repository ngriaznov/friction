//! `Plan`'s two serialization surfaces — [`std::fmt::Display`] (a
//! human-readable table) and `serde` JSON — each produce exactly the same
//! bytes for the same `(metrics, envelope)` input, run after run, and
//! neither depends on iteration order anywhere.

use friction_core::{Envelope, MetricVector};
use friction_plan::Plan;
use friction_rules::MapEnvelope;

fn sample_plan() -> Plan {
    let metrics = MetricVector {
        bold_span_density: 12.0,
        discourse_marker_density: 6.0,
        sentence_length_cv: 0.2,
        ..MetricVector::default()
    };
    let envelope = MapEnvelope::new()
        .with("bold_span_density", Envelope::new(0.0, 8.0))
        .with("discourse_marker_density", Envelope::new(0.0, 2.0))
        .with("sentence_length_cv", Envelope::new(0.5, 1.0));
    Plan::build(&metrics, &envelope)
}

/// `Display` renders a Markdown table whose header, family names, and
/// budget totals are present and byte-identical across repeated calls.
#[test]
fn display_renders_a_stable_markdown_table() {
    let plan = sample_plan();
    let once = plan.to_string();
    let twice = plan.to_string();
    assert_eq!(once, twice, "Display output must be deterministic");

    assert!(once.starts_with("| family | budget | metric | current | lo | hi | excess |\n"));
    assert!(once.contains("structural"));
    assert!(once.contains("symmetry"));
    assert!(once.contains("connective"));
    assert!(once.contains("lexical"));
    assert!(once.contains("rhythm"));
    assert!(once.contains("contraction"));
    // bold_span_density: current 12.0000, band lo 0.0000, hi 8.0000.
    assert!(once.contains("bold_span_density | 12.0000 | 0.0000 | 8.0000 | 4.0000"));
}

/// A metric with no envelope band at all gets `-` placeholders for its
/// `lo`/`hi` columns rather than a panic or a missing row.
#[test]
fn display_shows_placeholders_for_bandless_metrics() {
    let plan = Plan::build(&MetricVector::default(), &MapEnvelope::new());
    let table = plan.to_string();
    assert!(table.contains("bold_span_density | 0.0000 | - | - | 0.0000 |"));
}

/// `serde_json::to_string` is stable across repeated calls, and every
/// family name a real `Plan` produces round-trips as a plain lowercase
/// JSON string.
#[test]
fn json_serialization_is_stable_and_contains_every_family() {
    let plan = sample_plan();
    let once = serde_json::to_string(&plan).expect("Plan must serialize");
    let twice = serde_json::to_string(&plan).expect("Plan must serialize");
    assert_eq!(once, twice, "JSON output must be deterministic");

    for family in [
        "structural",
        "symmetry",
        "connective",
        "lexical",
        "rhythm",
        "contraction",
    ] {
        assert!(
            once.contains(&format!("\"family\":\"{family}\"")),
            "expected {family:?} in JSON output: {once}"
        );
    }

    // Family order in the JSON array matches the fixed schedule order,
    // not any alphabetical or hash-derived one.
    let structural_at = once.find("\"structural\"").unwrap();
    let contraction_at = once.find("\"contraction\"").unwrap();
    assert!(
        structural_at < contraction_at,
        "structural must serialize before contraction"
    );
}
