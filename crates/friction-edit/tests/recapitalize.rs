//! Sentence-initial recapitalization guards.
//!
//! The recapitalization step exists to repair an opener the engine's own
//! leading deletion exposed — it must never "fix" an author's deliberate
//! lowercase opener (a product name like `mimalloc`), and must never
//! emit a recapitalization-only patch for a sentence it did not
//! otherwise edit.

use friction_edit::Engine;

fn engine() -> Engine {
    Engine::new().expect("engine must load: embedded tagger/packs are always present")
}

/// A document whose only lowercase sentence openers are deliberate
/// (`mimalloc`, shown lowercase mid-sentence too) and which contains
/// nothing else the engine would edit must round-trip byte-identically —
/// no recapitalization-only patches.
#[test]
fn untouched_lowercase_opener_passes_through_byte_identical() {
    let input = "mimalloc's advantages come from its page-level design. Allocation stays fast \
                 because mimalloc drops free pages eagerly. mimalloc keeps thread caches small.";
    let (out, report) = engine().fix_document(input).unwrap();
    assert_eq!(
        out, input,
        "a document with deliberate lowercase openers and no other applicable edit must pass \
         through byte-identical"
    );
    let patches: usize = report.passes.iter().map(|p| p.patches_applied).sum();
    assert_eq!(patches, 0, "no patches may be applied to this document");
}

/// When a sentence *is* edited elsewhere but its own original lowercase
/// opener is untouched, the document-local casing evidence (`mimalloc`
/// appears lowercase mid-sentence in the next sentence) must hold the
/// recapitalization: the opener stays lowercase.
#[test]
fn deliberate_lowercase_opener_survives_an_edit_elsewhere_in_its_sentence() {
    let input = "mimalloc leverages the operating system's page cache. Fragmentation stays low \
                 because mimalloc drops free pages eagerly.";
    let expected = "mimalloc uses the operating system's page cache. Fragmentation stays low \
                    because mimalloc drops free pages eagerly.";
    let (out, _report) = engine().fix_document(input).unwrap();
    assert_eq!(
        out, expected,
        "the substitution must fire while the deliberate lowercase opener survives"
    );
}

/// The guard must not reach the case it was never about: an opener newly
/// exposed by the engine's own leading deletion carries no authorial
/// casing intent and is still capitalized (the behavior the byte-exact
/// fixture suite depends on).
#[test]
fn leading_deletion_still_recapitalizes_the_exposed_opener() {
    let input = "It is important to note that the configuration file simply needs to be placed \
                 in the root directory of your project.";
    let expected =
        "The configuration file needs to be placed in the root directory of your project.";
    let (out, _report) = engine().fix_document(input).unwrap();
    assert_eq!(
        out, expected,
        "a leading deletion's exposed opener must still be recapitalized"
    );
}
