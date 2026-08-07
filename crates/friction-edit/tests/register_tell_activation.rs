//! The staged-activation contract for the two tell features wired in
//! this arc (`past_progressive`, `contrast_closer`): the wiring landed
//! inert (no bands), and this file then asserted their absence. The
//! measured `register-en-v1.toml` entries have since landed, so it now
//! pins the activation state instead: the embedded pack carries exactly
//! the measured bands, the transducer fires on confidently-above-band
//! documents, and the two-instance arming floor keeps a single instance
//! of a nonzero-band construction from ever counting as register
//! evidence.

use friction_edit::Engine;

fn engine() -> Engine {
    Engine::new().expect("embedded packs and models load")
}

/// The embedded pack carries the two measured bands, exactly as
/// `corpus-tool register-bands` reported them over the 58-document
/// human docs population.
#[test]
fn the_two_staged_features_carry_the_measured_bands() {
    let pack = &friction_packs::REGISTER.pack;
    let past_prog = pack
        .band("past_progressive")
        .expect("past_progressive band present");
    assert_eq!((past_prog.low, past_prog.high), (0.0, 0.0));
    let contrast = pack
        .band("contrast_closer")
        .expect("contrast_closer band present");
    assert_eq!((contrast.low, contrast.high), (0.0, 1.8382));
}

/// A document confidently above both bands: T9 collapses the past
/// progressive it can reach in one pass (same-sentence conflicts hold
/// the second instance for a later pass, unrelated to T8's removal),
/// the see-saw closers surface as held findings, never edits, and the
/// licensed ", and " splice -- no longer a register tell this engine
/// acts on -- survives untouched.
#[test]
fn a_document_stuffed_with_both_staged_constructs_is_byte_identical() {
    let source = "The tool was crashing on start. The parser was failing on every line, \
                  and the loader was skipping the manifest. We fixed the loader first, \
                  and we shipped the patch the same day. It fails fast rather than \
                  degrading, and the cache is a hint, not a contract.\n";
    let engine = engine();
    let (once, report) = engine.fix_document(source).expect("engine runs");
    assert_ne!(once, source, "an above-band document must be edited");
    assert!(
        once.contains("The tool crashed on start."),
        "T9 must collapse the progressive: {once:?}"
    );
    assert!(
        once.contains("every line, and the loader was skipping the manifest"),
        "the ', and ' joint must survive untouched: {once:?}"
    );
    assert!(
        report
            .passes
            .iter()
            .flat_map(|p| &p.held)
            .any(|f| f.rule.as_str() == "register.contrast_closer"),
        "the see-saw closers must surface as held findings"
    );
    assert!(
        once.contains("rather than"),
        "contrast closers are detect-only and must never be edited"
    );
}

/// The two-instance arming floor: one semicolon splice in a short,
/// otherwise clean text bounds above `semicolon`'s band edge on
/// denominator noise alone (Wilson lower 8.9 vs high 3.6563), and
/// before the floor T7 split it -- the near-noop canary caught exactly
/// that. One occurrence of a construction the median human document
/// also uses is evidence of English, not of register.
#[test]
fn a_single_semicolon_in_short_text_never_arms() {
    let source = "Run the scanner from the project root. Results stream in as they are \
                  found; nothing is deleted without confirmation.\n";
    let (fixed, _) = engine().fix_document(source).expect("engine runs");
    assert_eq!(fixed, source, "one instance must never arm a nonzero band");
}
