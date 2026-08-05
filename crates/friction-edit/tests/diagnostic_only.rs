//! The diagnostic-only contract, enforced end to end: an inventory
//! entry whose `repair = "diagnostic_only"` is recorded as a finding
//! and NEVER executed as an edit — for both families that carry the
//! field. Before this enforcement existed the shipped diagnostic-only
//! ritual frame deleted its whole sentence (verified against the
//! released 0.5.2 binary), destroying real content its own pack notes
//! forbid touching.

use friction_edit::Engine;

fn engine() -> Engine {
    Engine::new().expect("embedded packs and models load")
}

/// The shipped diagnostic-only ritual frame: the sentence carries real
/// information ("test the deploy path early") and must survive intact,
/// surfacing as a held finding instead.
#[test]
fn diagnostic_only_ritual_frame_never_deletes_its_sentence() {
    let source = "The setup steps are below. I was fortunate enough to have had the \
                  opportunity to test the deploy path early. Run the installer first.\n";
    let (fixed, report) = engine().fix_document(source).expect("engine runs");
    assert_eq!(fixed, source, "the sentence must survive intact");
    assert!(
        report
            .passes
            .iter()
            .flat_map(|p| &p.held)
            .any(|f| f.message.contains("diagnostic-only")),
        "the frame must surface as a diagnostic finding"
    );
}

/// The diagnostic-only deletion span (`span.positive_contrast`): marks
/// the generated-test-register tell, rewrites nothing — including in
/// the phrase's legitimate domain uses (radiology, display
/// engineering), which are exactly why no mechanical rewrite is safe.
#[test]
fn diagnostic_only_span_marks_without_rewriting() {
    let engine = engine();
    for source in [
        "A positive contrast would make the empty shared set meaningful.\n",
        "The radiology suite uses a positive contrast agent for the study.\n",
    ] {
        let (fixed, report) = engine.fix_document(source).expect("engine runs");
        assert_eq!(fixed, source, "diagnostic spans must never edit");
        assert!(
            report
                .passes
                .iter()
                .flat_map(|p| &p.held)
                .any(|f| f.message.contains("span.positive_contrast")),
            "the span must surface as a finding: {source:?}"
        );
    }
}
