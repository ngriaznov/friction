//! One `#[test]` function per `reject`-fixture id (12), each asserting the
//! documented [`Gate`](friction_harness::gates::Gate) and the specific
//! mechanism ([`RejectFlag::ClosureViolated`] vs
//! [`RejectFlag::ScoreNotBetter`]) that flags its bad output — pinning the
//! *specific* mechanism per fixture so a regression in the wrong check is
//! caught even if the other still happens to save it.

use friction_harness::closure::{ClosureVerdict, check_closure};
use friction_harness::fixtures::{FIXTURES, RejectFixture};
use friction_harness::gates::{Gate, expected_pivot_rejection, gate_for_fixture_id};
use friction_harness::pivot::{PivotOutcome, match_pivot};
use friction_harness::score::{Verdict, shared_scorer};
use friction_harness::verify::{RejectFlag, check_accept_pair, check_reject_output};

/// Looks up a `reject` fixture by id.
///
/// # Panics
/// Panics if `id` is not present in `FIXTURES.reject` — the coverage
/// signal for an id this file hasn't been extended for.
fn find_reject(id: &str) -> &'static RejectFixture {
    FIXTURES
        .reject
        .iter()
        .find(|f| f.id == id)
        .unwrap_or_else(|| panic!("unhandled fixture id: {id}"))
}

/// Asserts every one of `fixture`'s documented bad outputs is flagged by
/// `check_reject_output`, and that each is flagged by exactly the
/// mechanism named in `expected` (one entry per bad output, in the same
/// order `RejectFixture::bad_outputs` returns them).
fn assert_bad_outputs_flagged(fixture: &RejectFixture, expected: &[fn(&RejectFlag) -> bool]) {
    let input = fixture.input.as_deref().expect("fixture has a plain input");
    let bad_outputs = fixture.bad_outputs();
    assert_eq!(
        bad_outputs.len(),
        expected.len(),
        "fixture {} bad-output count mismatch",
        fixture.id
    );
    for (bad_output, matches_expected) in bad_outputs.iter().zip(expected) {
        let flag =
            check_reject_output(shared_scorer(), input, bad_output).expect("scoring must succeed");
        assert!(
            !matches!(flag, RejectFlag::Unflagged),
            "fixture {} bad output {bad_output:?} was not flagged",
            fixture.id
        );
        assert!(
            matches_expected(&flag),
            "fixture {} bad output {bad_output:?} flagged by an unexpected mechanism: {flag:?}",
            fixture.id
        );
    }
}

const fn is_closure_violated(flag: &RejectFlag) -> bool {
    matches!(flag, RejectFlag::ClosureViolated(_))
}

const fn is_score_not_better(flag: &RejectFlag) -> bool {
    matches!(flag, RejectFlag::ScoreNotBetter(_))
}

#[test]
fn reject_fixture_pivot_trap_subject_genitive() {
    let fixture = find_reject("pivot_trap_subject_genitive");
    assert_eq!(gate_for_fixture_id(&fixture.id), Gate::PairLicensing);
    // "Obtain" is not a light verb at all, so the pivot machinery never
    // fires. The bad output's inserted "approve" is caught by closure.
    assert_bad_outputs_flagged(fixture, &[is_closure_violated]);
}

#[test]
fn reject_fixture_pivot_trap_modified_nominal() {
    let fixture = find_reject("pivot_trap_modified_nominal");
    assert_eq!(
        gate_for_fixture_id(&fixture.id),
        Gate::ModifiedOrPluralNominal
    );
    let input = fixture.input.as_deref().expect("fixture has an input");
    let outcome = match_pivot(input, shared_scorer().tagger());
    assert_eq!(
        outcome,
        PivotOutcome::Rejected(expected_pivot_rejection(&fixture.id).unwrap())
    );
}

#[test]
fn reject_fixture_pivot_trap_quantified() {
    let fixture = find_reject("pivot_trap_quantified");
    assert_eq!(
        gate_for_fixture_id(&fixture.id),
        Gate::ModifiedOrPluralNominal
    );
    let input = fixture.input.as_deref().expect("fixture has an input");
    let outcome = match_pivot(input, shared_scorer().tagger());
    assert_eq!(
        outcome,
        PivotOutcome::Rejected(expected_pivot_rejection(&fixture.id).unwrap())
    );
}

#[test]
fn reject_fixture_pivot_trap_artifact_noun() {
    let fixture = find_reject("pivot_trap_artifact_noun");
    assert_eq!(gate_for_fixture_id(&fixture.id), Gate::PairLicensing);
    // "Create" is not a light verb, so no pivot fires and closure alone
    // cannot catch the bad output either ("index" already appears in the
    // input as a noun). This one is caught by the fragment-rate tier:
    // "Index the most common queries." loses its imperative reading once
    // "Create" is gone, so it scores strictly worse, not better.
    assert_bad_outputs_flagged(fixture, &[is_score_not_better]);
}

#[test]
fn reject_fixture_pivot_trap_non_light_verb() {
    let fixture = find_reject("pivot_trap_non_light_verb");
    assert_eq!(gate_for_fixture_id(&fixture.id), Gate::NonLightVerb);
    // "facilitates" is not in the light-verb table, so "integrates" (a
    // real DERIV value, but never actually licensed here) is caught by
    // closure's per-input vocabulary check.
    assert_bad_outputs_flagged(fixture, &[is_closure_violated]);
}

#[test]
fn reject_fixture_pivot_trap_passive() {
    let fixture = find_reject("pivot_trap_passive");
    assert_eq!(gate_for_fixture_id(&fixture.id), Gate::ActiveVoiceOnly);
    let input = fixture.input.as_deref().expect("fixture has an input");
    let outcome = match_pivot(input, shared_scorer().tagger());
    assert_eq!(
        outcome,
        PivotOutcome::Rejected(expected_pivot_rejection(&fixture.id).unwrap())
    );
}

/// "Making the Decision" structurally *is* a licensed light-verb
/// construction — `match_pivot` correctly returns `Licensed` for it, and
/// closure correctly holds for "deciding" against it, because the edit is
/// textually valid. The only thing wrong with it is that it is a heading,
/// not prose (gate 7), and nothing in this crate is block-type-aware.
/// Rather than force a false "rejected" claim, this test demonstrates the
/// trap's actual mechanism instead of asserting a rejection the engine
/// cannot produce: a real rejection needs a block-tree-aware engine, out
/// of scope here.
#[test]
fn reject_fixture_heading_pivot() {
    let fixture = find_reject("heading_pivot");
    assert_eq!(gate_for_fixture_id(&fixture.id), Gate::ProseBlocksOnly);
    let block = fixture
        .input_block
        .as_ref()
        .expect("heading_pivot carries an input_block");
    let bad_output = fixture
        .bad_output
        .as_deref()
        .expect("heading_pivot carries a bad_output");

    assert_eq!(block.kind, "heading", "this fixture's input is a heading");

    let outcome = match_pivot(&block.text, shared_scorer().tagger());
    match outcome {
        PivotOutcome::Licensed(pivot) => {
            assert_eq!(pivot.derived_verb, bad_output);
        }
        other => panic!("expected a Licensed pivot for the heading text, got {other:?}"),
    }

    let closure = check_closure(&block.text, bad_output, shared_scorer().tagger());
    assert_eq!(closure, ClosureVerdict::Holds);
}

#[test]
fn reject_fixture_bridge_modal_insertion() {
    let fixture = find_reject("bridge_modal_insertion");
    assert_eq!(
        gate_for_fixture_id(&fixture.id),
        Gate::NoInsertionGuardToken
    );
    // "can" never appears in the input at all, so closure catches it.
    assert_bad_outputs_flagged(fixture, &[is_closure_violated]);
}

#[test]
fn reject_fixture_bridge_connective_insertion() {
    let fixture = find_reject("bridge_connective_insertion");
    assert_eq!(
        gate_for_fixture_id(&fixture.id),
        Gate::NoInsertionGuardToken
    );
    // Neither "because" nor "with" appears in the input, so both bad
    // outputs are closure violations.
    assert_bad_outputs_flagged(fixture, &[is_closure_violated, is_closure_violated]);

    let input = fixture.input.as_deref().expect("fixture has an input");
    let good_output = fixture
        .good_output
        .as_deref()
        .expect("bridge_connective_insertion carries a good_output");
    assert_eq!(
        check_closure(input, good_output, shared_scorer().tagger()),
        ClosureVerdict::Holds
    );
    let check =
        check_accept_pair(shared_scorer(), input, good_output).expect("scoring must succeed");
    assert_eq!(check.verdict, Verdict::Better);
}

#[test]
fn reject_fixture_bridge_content_insertion() {
    let fixture = find_reject("bridge_content_insertion");
    assert_eq!(gate_for_fixture_id(&fixture.id), Gate::ClosureContentWord);
    // "documentation" never appears in the input at all.
    assert_bad_outputs_flagged(fixture, &[is_closure_violated]);
}

#[test]
fn reject_fixture_bridge_inert_insertion() {
    let fixture = find_reject("bridge_inert_insertion");
    assert_eq!(
        gate_for_fixture_id(&fixture.id),
        Gate::ClosureInertVocabulary
    );
    // "to" already appears once in the input ("used to it"); the bad
    // output uses it twice — this is the multiset-count catch, not a
    // plain set-membership check.
    assert_bad_outputs_flagged(fixture, &[is_closure_violated]);

    let input = fixture.input.as_deref().expect("fixture has an input");
    let good_output = fixture
        .good_output
        .as_deref()
        .expect("bridge_inert_insertion carries a good_output");
    assert_eq!(
        check_closure(input, good_output, shared_scorer().tagger()),
        ClosureVerdict::Holds
    );
    let check =
        check_accept_pair(shared_scorer(), input, good_output).expect("scoring must succeed");
    assert_eq!(check.verdict, Verdict::Better);
}

#[test]
fn reject_fixture_word_salad_synthesis() {
    let fixture = find_reject("word_salad_synthesis");
    assert_eq!(gate_for_fixture_id(&fixture.id), Gate::OpenSynthesisBanned);
    // Open synthesis inserts a flood of tokens absent from the input.
    // Closure catches it immediately.
    assert_bad_outputs_flagged(fixture, &[is_closure_violated]);
}

/// `dejust_guard_just_as` is not in [`gate_for_fixture_id`]'s table — none
/// of that table's [`Gate`] variants describe a marker guard, and this
/// fixture's bad output ("Is it as fast, or slower?") is not caught by
/// closure OR score either: deleting a word never violates closure, and
/// this crate's generic tell-span counter has no signal for the
/// contrast-frame channel at all (see the accept-fixture sibling's own
/// comment on `frame_dejust_provenance_question`). That is exactly why
/// this fixture exists — "just as fast" is the comparison sense, not the
/// dismissive particle, and only a real semantic guard inside
/// `friction-edit`'s own `frame.dejust` gate can tell the two apart. The
/// real engine's behavior (no edit at all) is pinned directly in
/// `tests/engine_fixtures.rs`; this test only pins the fixture's own
/// documented shape.
#[test]
fn reject_fixture_dejust_guard_just_as() {
    let fixture = find_reject("dejust_guard_just_as");
    assert_eq!(fixture.bad_outputs(), vec!["Is it as fast, or slower?"]);
    assert!(
        fixture.gate.contains("marker guard"),
        "fixture gate description: {}",
        fixture.gate
    );
}
