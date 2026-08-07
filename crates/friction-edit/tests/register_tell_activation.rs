//! The staged-activation contract for the two tell features wired in
//! this arc (`past_progressive`, `contrast_closer`): the wiring landed
//! inert (no bands), and this file then asserted their absence. The
//! measured `register-en-v1.toml` entries have since landed, so it now
//! pins the activation state instead: the embedded pack carries exactly
//! the measured bands, the transducer fires on confidently-above-band
//! documents, and the two-instance arming floor keeps a single instance
//! of a nonzero-band construction from ever counting as register
//! evidence.
//!
//! The arc has since shipped a second repair on top: `contrast_closer`
//! was detect-only when this file was first written, but two narrow
//! sentence-final tail shapes (`", not <tail>."`, `"rather than
//! <tail>."`) now have their own transducer (T10). The stuffed-document
//! test below pins both halves of that split: a licensed sentence-final
//! tail gets deleted, while a mid-sentence "rather than" -- unlicensed,
//! since more clause follows it -- still surfaces as a held finding.

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
/// the licensed ", and " splice -- no longer a register tell this engine
/// acts on -- survives untouched, and the two `contrast_closer`
/// instances split exactly along T10's own licensing line: the
/// sentence-final ", not a contract." tail is deleted, while the
/// mid-sentence "rather than degrading" -- more clause follows it, so it
/// never reaches the sentence's own end -- stays a held finding.
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
        once.contains("rather than degrading"),
        "a mid-sentence \"rather than\" is never licensed and must survive untouched: {once:?}"
    );
    assert!(
        once.contains("the cache is a hint.") && !once.contains("not a contract"),
        "a licensed sentence-final \", not <tail>.\" tail must be deleted: {once:?}"
    );
    assert!(
        report
            .passes
            .iter()
            .flat_map(|p| &p.held)
            .any(|f| f.rule.as_str() == "register.contrast_closer"),
        "the still-unlicensed mid-sentence \"rather than\" must surface as a held finding"
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
