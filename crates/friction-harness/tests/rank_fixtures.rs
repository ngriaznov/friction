//! One `#[test]` function per `rank`-fixture id: the scorer must order the
//! documented `better` text ahead of the documented `worse` text.

use friction_harness::fixtures::{FIXTURES, sample_by_path};
use friction_harness::score::{Verdict, shared_scorer};

#[test]
fn rank_fixture_chatspeak_vs_human() {
    let fixture = FIXTURES
        .rank
        .iter()
        .find(|f| f.id == "chatspeak_vs_human")
        .expect("chatspeak_vs_human fixture must exist");

    let worse_path = fixture
        .worse_file
        .as_deref()
        .expect("chatspeak_vs_human uses worse_file");
    let better_path = fixture
        .better_file
        .as_deref()
        .expect("chatspeak_vs_human uses better_file");
    let worse = sample_by_path(worse_path).expect("worse_file resolves to an embedded sample");
    let better = sample_by_path(better_path).expect("better_file resolves to an embedded sample");

    let scorer = shared_scorer();
    let worse_score = scorer.score(worse).expect("scoring must succeed");
    let better_score = scorer.score(better).expect("scoring must succeed");
    assert_eq!(
        better_score.compare_to(&worse_score),
        Verdict::Better,
        "the well-written-docs sample must score better than the chat-speak sample"
    );
}

#[test]
fn rank_fixture_dumb_human_vs_conservative() {
    let fixture = FIXTURES
        .rank
        .iter()
        .find(|f| f.id == "dumb_human_vs_conservative")
        .expect("dumb_human_vs_conservative fixture must exist");

    let worse = fixture
        .worse
        .as_deref()
        .expect("dumb_human_vs_conservative uses inline worse");
    let better = fixture
        .better
        .as_deref()
        .expect("dumb_human_vs_conservative uses inline better");

    let scorer = shared_scorer();
    let worse_score = scorer.score(worse).expect("scoring must succeed");
    let better_score = scorer.score(better).expect("scoring must succeed");
    assert_eq!(
        better_score.compare_to(&worse_score),
        Verdict::Better,
        "the conservative rewrite must score better than the dumb-human collapse"
    );
}
