//! Determinism checks: identical input must produce identical output,
//! across independently-constructed [`Scorer`]s (not the shared static, so
//! this also catches any hidden nondeterminism a shared cache might mask).

use friction_harness::closure::check_closure;
use friction_harness::fixtures::{CHATSPEAK_MD, FIXTURES};
use friction_harness::score::Scorer;
use friction_harness::tellspan::tell_span_hits;

fn accept_fixture_pair() -> (String, String) {
    let fixture = FIXTURES
        .accept
        .iter()
        .find(|f| f.id == "regression1_getting_started")
        .expect("regression1_getting_started fixture must exist");
    (
        fixture.input.clone().expect("fixture has an input"),
        fixture.output.clone().expect("fixture has an output"),
    )
}

#[test]
fn scoring_is_deterministic_across_repeated_calls() {
    let a = Scorer::new().expect("tagger model loads");
    let b = Scorer::new().expect("tagger model loads");

    let (input, output) = accept_fixture_pair();
    assert_eq!(
        a.score(&input).expect("scoring succeeds"),
        b.score(&input).expect("scoring succeeds")
    );
    assert_eq!(
        a.score(&output).expect("scoring succeeds"),
        b.score(&output).expect("scoring succeeds")
    );
    assert_eq!(
        a.score(CHATSPEAK_MD).expect("scoring succeeds"),
        b.score(CHATSPEAK_MD).expect("scoring succeeds")
    );
}

#[test]
fn tell_span_hits_are_deterministic_across_independent_taggers() {
    let a = Scorer::new().expect("tagger model loads");
    let b = Scorer::new().expect("tagger model loads");

    let (input, output) = accept_fixture_pair();
    assert_eq!(
        tell_span_hits(&input, a.tagger()),
        tell_span_hits(&input, b.tagger())
    );
    assert_eq!(
        tell_span_hits(&output, a.tagger()),
        tell_span_hits(&output, b.tagger())
    );
    assert_eq!(
        tell_span_hits(CHATSPEAK_MD, a.tagger()),
        tell_span_hits(CHATSPEAK_MD, b.tagger())
    );
}

#[test]
fn closure_check_is_deterministic_across_independent_taggers() {
    let a = Scorer::new().expect("tagger model loads");
    let b = Scorer::new().expect("tagger model loads");

    let (input, output) = accept_fixture_pair();
    assert_eq!(
        check_closure(&input, &output, a.tagger()),
        check_closure(&input, &output, b.tagger())
    );
}
