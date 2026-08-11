//! End-to-end coverage for the derivational-pivot operation
//! (`RULE_PIVOT`, id `"pivot.lvc"`): collapsing a licensed light-verb
//! construction to its derived verb.

use friction_edit::Engine;

fn engine() -> Engine {
    Engine::new().expect("embedded packs and models load")
}

/// A licensed construction with no complement after it still
/// collapses to the derived verb, inflected to match the light verb's
/// own form.
#[test]
fn plain_construction_still_pivots() {
    let (fixed, _) = engine()
        .fix_document("You should make a decision soon.\n")
        .expect("engine runs");
    assert_eq!(fixed, "You should decide soon.\n");
}

/// A licensed construction immediately followed by a preposition (or
/// infinitival "to") holds instead of collapsing: the derived verb's
/// complement shape isn't the construction's, and the corpus never
/// adjudicated the construction-plus-preposition shape for any
/// derived verb — "decided to switch" might read fine on its own, but
/// the same collapse over "makes a request to Ledgerline" produces
/// "asks to Ledgerline". A true preposition (`IN`) always holds; an
/// infinitival "to" (`TO` followed by `VB`) passes — "made the
/// decision to switch" collapsing to "decided to switch" reads fine
/// for every derived verb, and the real-corpus pivot fixture pins
/// exactly that case.
#[test]
fn construction_followed_by_preposition_holds() {
    let source = "The team made a decision on the vendor list.\n";
    let (fixed, report) = engine().fix_document(source).expect("engine runs");
    assert_eq!(
        fixed, source,
        "the pivot must not collapse the construction"
    );
    assert!(
        report.passes.iter().flat_map(|p| &p.held).any(|f| {
            f.rule.as_str() == "pivot.lvc"
                && f.message.contains("construction followed by a preposition")
        }),
        "the preposition guard must surface as a held finding"
    );
}
