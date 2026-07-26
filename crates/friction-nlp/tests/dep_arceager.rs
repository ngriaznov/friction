//! Oracle correctness for the arc-eager transition system.
//!
//! The trainer this system feeds is only as correct as the oracle: every
//! gold tree it's handed gets turned into transitions by [`derive`], and
//! if that mapping is wrong the trainer learns the wrong lesson from
//! every sentence, silently. These tests hand-build small, concrete gold
//! trees (not fixtures loaded from a corpus), derive their transition
//! sequence, replay it from scratch, and check the exact resulting arcs
//! against the exact expected sequence: real values, not just "it didn't
//! panic".

use friction_nlp::dep_arceager::{Configuration, DeriveError, Transition, derive, oracle};
use friction_nlp::{Confidence, DepEdge, DepRelation, SentenceParse};

/// Builds a gold [`SentenceParse`] from `(head, relation)` pairs, one per
/// token in order.
fn gold(edges: &[(Option<usize>, DepRelation)]) -> SentenceParse {
    let edges = edges
        .iter()
        .enumerate()
        .map(|(token, &(head, relation))| DepEdge {
            token,
            head,
            relation,
            confidence: Confidence::CERTAIN,
        })
        .collect();
    SentenceParse::new(edges).expect("test fixture is a well-formed tree")
}

/// Replays `transitions` from a fresh [`Configuration`] over `token_count`
/// tokens and materializes the result as a [`SentenceParse`].
fn replay(token_count: usize, transitions: &[Transition]) -> SentenceParse {
    let mut config = Configuration::new(token_count);
    for &transition in transitions {
        assert!(
            config.is_allowed(transition),
            "replayed transition {transition:?} was not legal for {config:?}"
        );
        config.apply(transition);
    }
    SentenceParse::new(config.finish()).expect("a finished derivation is always well-formed")
}

/// Asserts `derive(&gold)` succeeds, produces exactly `expected`, and that
/// replaying `expected` reproduces `gold`'s arcs exactly (head and
/// relation, token by token).
fn assert_derives_to(gold: &SentenceParse, expected: &[Transition]) {
    let transitions = derive(gold).expect("fixture is projective and should derive cleanly");
    assert_eq!(transitions, expected, "unexpected transition sequence");

    let produced = replay(gold.edges().len(), &transitions);
    for (produced_edge, gold_edge) in produced.edges().iter().zip(gold.edges()) {
        assert_eq!(
            produced_edge.head, gold_edge.head,
            "token {} head mismatch",
            produced_edge.token
        );
        assert_eq!(
            produced_edge.relation, gold_edge.relation,
            "token {} relation mismatch",
            produced_edge.token
        );
    }
}

// ---------------------------------------------------------------------
// 1. Oracle round-trip
// ---------------------------------------------------------------------

/// A minimal three-token sentence: a left (subject) dependent and a right
/// (punctuation) dependent of a single root.
#[test]
fn derive_round_trips_a_simple_three_token_sentence() {
    // "Birds fly ." — fly(1) is root; Birds(0) is its left dependent,
    // "."(2) its right dependent.
    let gold = gold(&[
        (Some(1), DepRelation::Nsubj),
        (None, DepRelation::Root),
        (Some(1), DepRelation::Punct),
    ]);
    assert_derives_to(
        &gold,
        &[
            Transition::Shift,
            Transition::LeftArc(DepRelation::Nsubj),
            Transition::Shift,
            Transition::RightArc(DepRelation::Punct),
            Transition::Reduce,
        ],
    );
}

/// A right-branching chain: each token's head is its immediate left
/// neighbor, so every attachment is a `RightArc` fired the instant the
/// dependent is shifted, with no token ever needing to wait on the stack.
#[test]
fn derive_round_trips_a_right_branching_chain() {
    let gold = gold(&[
        (None, DepRelation::Root),
        (Some(0), DepRelation::Advcl),
        (Some(1), DepRelation::Ccomp),
        (Some(2), DepRelation::Xcomp),
    ]);
    assert_derives_to(
        &gold,
        &[
            Transition::Shift,
            Transition::RightArc(DepRelation::Advcl),
            Transition::RightArc(DepRelation::Ccomp),
            Transition::RightArc(DepRelation::Xcomp),
            Transition::Reduce,
            Transition::Reduce,
            Transition::Reduce,
        ],
    );
}

/// A root with two of its own left dependents (a determiner and an
/// adjective modifying the subject), which must each wait on the stack
/// until the subject's own attachment to the root exposes them in turn.
#[test]
fn derive_round_trips_left_dependents_of_the_root() {
    // "The old team shipped ." (without the trailing period, to keep this
    // fixture to exactly the dependents under test) — shipped(3) is root;
    // team(2) is its left (subject) dependent; The(0)/old(1) are in turn
    // team's own left dependents.
    let gold = gold(&[
        (Some(2), DepRelation::Det),
        (Some(2), DepRelation::Amod),
        (Some(3), DepRelation::Nsubj),
        (None, DepRelation::Root),
    ]);
    assert_derives_to(
        &gold,
        &[
            Transition::Shift,
            Transition::Shift,
            Transition::LeftArc(DepRelation::Amod),
            Transition::LeftArc(DepRelation::Det),
            Transition::Shift,
            Transition::LeftArc(DepRelation::Nsubj),
            Transition::Shift,
        ],
    );
}

/// Four tokens all governed by a root that sits at the far right of the
/// sentence: none of them can attach until the buffer front reaches the
/// root's own index, forcing four consecutive `Shift`s (one per
/// not-yet-attachable token) before any arc fires at all.
#[test]
fn derive_round_trips_a_head_several_tokens_to_the_right() {
    let gold = gold(&[
        (Some(4), DepRelation::Nsubj),
        (Some(4), DepRelation::Dobj),
        (Some(4), DepRelation::Amod),
        (Some(4), DepRelation::Det),
        (None, DepRelation::Root),
    ]);
    assert_derives_to(
        &gold,
        &[
            Transition::Shift,
            Transition::Shift,
            Transition::Shift,
            Transition::Shift,
            // Popped stack order is last-pushed-first: token 3, then 2,
            // then 1, then 0.
            Transition::LeftArc(DepRelation::Det),
            Transition::LeftArc(DepRelation::Amod),
            Transition::LeftArc(DepRelation::Dobj),
            Transition::LeftArc(DepRelation::Nsubj),
            Transition::Shift,
        ],
    );
}

/// A single-token sentence: nothing to attach, so the only legal step is
/// shifting it onto the stack before [`Configuration::finish`] marks it
/// root.
#[test]
fn derive_round_trips_a_single_token_sentence() {
    let gold = gold(&[(None, DepRelation::Root)]);
    assert_derives_to(&gold, &[Transition::Shift]);
}

// ---------------------------------------------------------------------
// 2. Projective-only
// ---------------------------------------------------------------------

/// A gold tree with a genuine crossing dependency — arcs `(0, 2)` and
/// `(1, 3)` interleave (`0 < 1 < 2 < 3`) — cannot be produced by this
/// transition system: it has no way to defer attaching token 0 to token 2
/// past token 1's own later attachment to token 3. `derive` must reject
/// it rather than hand back a transition sequence that silently drops the
/// crossing arc.
#[test]
fn derive_rejects_a_non_projective_gold_tree() {
    let gold = gold(&[
        (Some(2), DepRelation::Amod),
        (Some(3), DepRelation::Nsubj),
        (None, DepRelation::Root),
        (Some(2), DepRelation::Dobj),
    ]);
    assert!(
        matches!(derive(&gold), Err(DeriveError)),
        "a non-projective gold tree must not derive cleanly"
    );
}

// ---------------------------------------------------------------------
// 3. Precondition masking
// ---------------------------------------------------------------------

/// `LeftArc` is rejected once the stack top already has a head assigned.
#[test]
fn is_allowed_rejects_left_arc_once_stack_top_has_a_head() {
    let mut config = Configuration::new(3);
    config.apply(Transition::Shift); // stack = [0], buffer = 1
    config.apply(Transition::RightArc(DepRelation::Dobj)); // arcs[1] = (0, Dobj), stack = [0, 1]
    assert!(config.arc(1).is_some());
    assert!(!config.is_allowed(Transition::LeftArc(DepRelation::Other)));
}

/// `Reduce` is rejected while the stack top still has no head.
#[test]
fn is_allowed_rejects_reduce_while_stack_top_is_headless() {
    let mut config = Configuration::new(2);
    config.apply(Transition::Shift); // stack = [0], arcs[0] still None
    assert!(config.arc(0).is_none());
    assert!(!config.is_allowed(Transition::Reduce));
}

/// `Shift`, `LeftArc`, and `RightArc` are all rejected once the buffer is
/// exhausted.
#[test]
fn is_allowed_rejects_shift_and_arcs_on_an_empty_buffer() {
    let mut config = Configuration::new(1);
    config.apply(Transition::Shift); // buffer now empty
    assert!(config.buffer_front().is_none());
    assert!(!config.is_allowed(Transition::Shift));
    assert!(!config.is_allowed(Transition::LeftArc(DepRelation::Other)));
    assert!(!config.is_allowed(Transition::RightArc(DepRelation::Other)));
}

/// `apply` panics rather than silently corrupting state when handed a
/// disallowed transition.
#[test]
#[should_panic(expected = "disallowed transition")]
fn apply_panics_on_a_disallowed_transition() {
    let mut config = Configuration::new(1);
    // The buffer is non-empty but the stack is empty, so Reduce has
    // nothing to pop.
    config.apply(Transition::Reduce);
}

/// [`oracle`] returns `None` exactly at a terminal configuration.
#[test]
fn oracle_returns_none_at_a_terminal_configuration() {
    let gold = gold(&[(None, DepRelation::Root)]);
    let mut config = Configuration::new(1);
    assert!(!config.is_terminal());
    config.apply(Transition::Shift);
    assert!(config.is_terminal());
    assert_eq!(oracle(&gold, &config), None);
}

// ---------------------------------------------------------------------
// 4. Completeness
// ---------------------------------------------------------------------

/// After any successful derivation, every token has either a real head or
/// is the root, and the result is accepted by [`SentenceParse::new`].
#[test]
fn derived_parses_are_always_complete_and_well_formed() {
    let gold = gold(&[
        (Some(2), DepRelation::Det),
        (Some(2), DepRelation::Amod),
        (Some(3), DepRelation::Nsubj),
        (None, DepRelation::Root),
    ]);
    let transitions = derive(&gold).unwrap();
    let parse = replay(gold.edges().len(), &transitions);

    assert_eq!(parse.edges().len(), gold.edges().len());
    let root_count = parse.edges().iter().filter(|e| e.head.is_none()).count();
    assert_eq!(root_count, 1, "exactly one token should end up rootless");
    for edge in parse.edges() {
        assert!(
            edge.head.is_some() || edge.relation == DepRelation::Root,
            "a headless token must carry DepRelation::Root: {edge:?}"
        );
    }
}

// ---------------------------------------------------------------------
// 5. Determinism
// ---------------------------------------------------------------------

/// Deriving the same gold tree twice produces byte-for-byte identical
/// transition sequences.
#[test]
fn derive_is_deterministic_across_repeated_calls() {
    let gold = gold(&[
        (Some(4), DepRelation::Nsubj),
        (Some(4), DepRelation::Dobj),
        (Some(4), DepRelation::Amod),
        (Some(4), DepRelation::Det),
        (None, DepRelation::Root),
    ]);
    let first = derive(&gold).unwrap();
    let second = derive(&gold).unwrap();
    assert_eq!(first, second);
}
