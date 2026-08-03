//! End-to-end coverage for the curated generated-test-register
//! substitutions (`sub.non_vacuous`, `sub.vacuous`, `sub.vacuously`):
//! each rewrites to its plain human counterpart, and the overlapping
//! patterns resolve by keeping the longer, earlier-starting match, so
//! "non-vacuous" is never mangled by the bare-"vacuous" rule.

use friction_edit::Engine;

fn engine() -> Engine {
    Engine::new().expect("embedded packs and models load")
}

#[test]
fn vacuous_family_rewrites_to_plain_human_words() {
    let engine = engine();
    for (source, expected) in [
        (
            "We added a non-vacuous test for the parser.\n",
            "We added a real test for the parser.\n",
        ),
        (
            "This check is vacuous and passes trivially.\n",
            "This check is meaningless and passes trivially.\n",
        ),
        (
            "The assertion passes vacuously when the list is empty.\n",
            "The assertion passes trivially when the list is empty.\n",
        ),
        // Sentence-initial: the substitution recapitalizes.
        (
            "A vacuous assertion slipped into the suite.\n",
            "A meaningless assertion slipped into the suite.\n",
        ),
    ] {
        let (fixed, _) = engine.fix_document(source).expect("engine runs");
        assert_eq!(fixed, expected);
    }
}

/// The bare-"vacuous" pattern also matches inside "non-vacuous" (the
/// hyphen is a word boundary); overlap resolution must keep the
/// longer, earlier-starting `sub.non_vacuous` match, never producing
/// "non-meaningless" or a half-replaced hybrid.
#[test]
fn non_vacuous_is_never_split_by_the_bare_rule() {
    let (fixed, _) = engine()
        .fix_document("The suite keeps every non-vacuous assertion and drops the vacuous one.\n")
        .expect("engine runs");
    assert_eq!(
        fixed,
        "The suite keeps every real assertion and drops the meaningless one.\n"
    );
}
