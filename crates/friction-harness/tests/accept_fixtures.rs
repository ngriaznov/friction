//! One `#[test]` function per `accept`-fixture id, so a failure names the
//! fixture in the test-runner output without inspecting panic text.
//!
//! Every check goes through [`check_accept_pair`], which asserts closure
//! holds, the output has strictly fewer tell spans than the input, and the
//! output's [`Score`](friction_harness::score::Score) is
//! [`Verdict::Better`](friction_harness::score::Verdict::Better) than the
//! input's — except `near_noop_clean_text`, whose byte-identical
//! input/output pair makes a strict tell-span decrease and a strictly
//! better score arithmetically impossible; see that test's own comment.

use friction_harness::closure::{ClosureVerdict, check_closure};
use friction_harness::fixtures::{AcceptFixture, FIXTURES};
use friction_harness::score::{Verdict, shared_scorer};
use friction_harness::verify::check_accept_pair;

/// Looks up an `accept` fixture by id.
///
/// # Panics
/// Panics if `id` is not present in `FIXTURES.accept` — the coverage
/// signal for an id this file hasn't been extended for, mirroring
/// `tests/coverage.rs`'s own check from the opposite direction.
fn find_accept(id: &str) -> &'static AcceptFixture {
    FIXTURES
        .accept
        .iter()
        .find(|f| f.id == id)
        .unwrap_or_else(|| panic!("unhandled fixture id: {id}"))
}

/// Runs the three standard accept-fixture checks (closure holds, output
/// has strictly fewer tell spans, output's score is strictly better) for
/// one `(input, output)` pair.
fn assert_accept_pair_improves(input: &str, output: &str) {
    let check = check_accept_pair(shared_scorer(), input, output).expect("scoring must succeed");
    assert_eq!(check.closure, ClosureVerdict::Holds, "closure must hold");
    assert!(
        check.output_tell_count < check.input_tell_count,
        "expected output_tell_count ({}) < input_tell_count ({})",
        check.output_tell_count,
        check.input_tell_count
    );
    assert_eq!(check.verdict, Verdict::Better, "output must score better");
}

#[test]
fn accept_fixture_regression1_getting_started() {
    let fixture = find_accept("regression1_getting_started");
    let input = fixture.input.as_deref().expect("fixture has an input");
    let output = fixture.output.as_deref().expect("fixture has an output");
    assert_accept_pair_improves(input, output);
}

#[test]
fn accept_fixture_composed_four_operation_paragraph() {
    let fixture = find_accept("composed_four_operation_paragraph");
    let input = fixture.input.as_deref().expect("fixture has an input");
    let output = fixture.output.as_deref().expect("fixture has an output");
    assert_accept_pair_improves(input, output);
}

#[test]
fn accept_fixture_pivot_constructed_cases() {
    let fixture = find_accept("pivot_constructed_cases");
    let pairs = fixture.pairs.as_ref().expect("fixture has pairs");
    assert_eq!(pairs.len(), 5, "expected all 5 constructed pivot pairs");
    for (input, output) in pairs {
        assert_accept_pair_improves(input, output);
    }
}

#[test]
fn accept_fixture_pivot_real_corpus() {
    let fixture = find_accept("pivot_real_corpus");
    let input = fixture.input.as_deref().expect("fixture has an input");
    let output = fixture.output.as_deref().expect("fixture has an output");
    assert_accept_pair_improves(input, output);
}

/// The near-no-op fixture's input and output are byte-identical, so two of
/// the three standard checks are arithmetically impossible to apply
/// literally: an output cannot have *strictly fewer* tell spans than an
/// input it is byte-identical to, and an unchanged text cannot score
/// *strictly better* than itself. This test asserts the fixture's own
/// stronger, degenerate-case requirement instead: both tell-span counts
/// are exactly zero (not merely equal to each other), closure holds
/// trivially, and the score comparison is a tie — the correct outcome for
/// doing nothing to text that was already clean.
#[test]
fn accept_fixture_near_noop_clean_text_is_zero_tell_identity() {
    let fixture = find_accept("near_noop_clean_text");
    let input = fixture.input.as_deref().expect("fixture has an input");
    let output = fixture.output.as_deref().expect("fixture has an output");
    assert_eq!(input, output, "near-no-op fixture must be byte-identical");

    let check = check_accept_pair(shared_scorer(), input, output).expect("scoring must succeed");
    assert_eq!(check.closure, ClosureVerdict::Holds);
    assert_eq!(
        check.input_tell_count, 0,
        "clean text must have zero tell spans"
    );
    assert_eq!(
        check.output_tell_count, 0,
        "clean text must have zero tell spans"
    );
    assert_eq!(
        check.verdict,
        Verdict::Tied,
        "identical text must score identically to itself"
    );
}

/// `frame_dejust_provenance_question` exercises the new frame-gated
/// `just`-deletion operation (`frame.dejust`), which this crate's generic
/// `count_tell_spans`/`score` machinery has no signal for: `tellspan`'s
/// own module docs name its four families (ritual/substitution/deletion/
/// pivot), sourced from `friction_packs::INVENTORY` and
/// `crate::pivot::match_pivot` — the new contrast-frame template channel
/// (`friction_match::frame`) lives entirely in `friction-match`, not in
/// that inventory, so this fixture's own tell count would read 0 on both
/// sides and the standard [`assert_accept_pair_improves`] check would be
/// vacuous (mirroring `near_noop_clean_text`'s own bespoke test above,
/// for the same reason: the generic checks don't fit every fixture).
/// Closure is still the right general invariant to check here — a marker
/// deletion introduces no new token either way — and the real engine's
/// own byte-exact output is asserted directly in
/// `tests/engine_fixtures.rs`, this workspace's dedicated real-engine
/// layer.
#[test]
fn accept_fixture_frame_dejust_provenance_question() {
    let fixture = find_accept("frame_dejust_provenance_question");
    let input = fixture.input.as_deref().expect("fixture has an input");
    let output = fixture.output.as_deref().expect("fixture has an output");
    let closure = check_closure(input, output, shared_scorer().tagger());
    assert_eq!(closure, ClosureVerdict::Holds, "closure must hold");
}
