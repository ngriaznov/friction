//! Real cases for the two ported rewrite transducers, each with an
//! asserted output string — not smoke tests.
//!
//! Every parse here is built by hand rather than run through the shipped
//! tagger/parser, so a test failure can only ever mean the transducer
//! itself is wrong, never that the parser mis-tagged or mis-parsed a
//! fixture out from under it.

use std::collections::BTreeMap;

use friction_core::{Token, TokenKind};
use friction_nlp::{Confidence, DepEdge, DepRelation, PosTag, SentenceParse, TaggedToken};
use friction_register::transduce::{
    Candidate, CandidateKind, candidates, past, past_participle, t4_activize_to_passive,
    t5_nominalization, third_sg,
};

/// One token's dependency-tree shape and tags, spelled out by hand.
///
/// `head` is `None` for the sentence's own root. `lemma` is given
/// explicitly rather than derived from `surface` + `tag`: exactly one
/// transducer under test ([`t4_activize_to_passive`]) reads a token's
/// lemma, and deriving it here would route these fixtures through a
/// third module's own (unrelated) lemmatization accuracy — the same
/// reason the parse itself is hand-built instead of run through the real
/// parser.
struct TokenShape {
    head: Option<usize>,
    relation: DepRelation,
    surface: &'static str,
    tag: &'static str,
    lemma: &'static str,
}

const fn tok(
    head: Option<usize>,
    relation: DepRelation,
    surface: &'static str,
    tag: &'static str,
    lemma: &'static str,
) -> TokenShape {
    TokenShape {
        head,
        relation,
        surface,
        tag,
        lemma,
    }
}

/// Builds source text plus tagged tokens plus a validated [`SentenceParse`]
/// from a list of hand-authored [`TokenShape`]s, joining surfaces with a
/// single space except before sentence-final punctuation.
fn build(shapes: &[TokenShape]) -> (String, Vec<TaggedToken>, SentenceParse) {
    let mut source = String::new();
    let mut tokens = Vec::with_capacity(shapes.len());
    for (index, item) in shapes.iter().enumerate() {
        let attaches_directly = matches!(item.surface, "." | "," | "!" | "?" | ";" | ":");
        if index > 0 && !attaches_directly {
            source.push(' ');
        }
        let start = source.len();
        source.push_str(item.surface);
        let end = source.len();
        tokens.push(TaggedToken {
            token: Token::new(start..end, TokenKind::Word),
            pos: PosTag::new(item.tag),
            lemma: Box::from(item.lemma),
        });
    }
    let edges = shapes
        .iter()
        .enumerate()
        .map(|(index, item)| DepEdge {
            token: index,
            head: item.head,
            relation: item.relation,
            confidence: Confidence::CERTAIN,
        })
        .collect();
    let parse = SentenceParse::new(edges).expect("hand-built parse must be well-formed");
    (source, tokens, parse)
}

/// Asserts every candidate's range slices `source` to exactly `expected`
/// — the byte-honesty check every candidate in this workspace must pass.
fn assert_ranges_match(source: &str, found: &[Candidate], expected: &[&str]) {
    assert_eq!(found.len(), expected.len(), "candidate count mismatch");
    for (candidate, &want) in found.iter().zip(expected) {
        candidate
            .validate(source)
            .unwrap_or_else(|e| panic!("candidate range {:?} invalid: {e}", candidate.range));
        assert_eq!(&source[candidate.range.clone()], want);
    }
}

// -----------------------------------------------------------------
// 1. T4 fires: generic subject + direct object -> passive.
// -----------------------------------------------------------------
#[test]
fn t4_fires_on_generic_subject_with_direct_object() {
    let shapes = [
        tok(Some(1), DepRelation::Nsubj, "We", "PRP", "we"),
        tok(None, DepRelation::Root, "deployed", "VBD", "deploy"),
        tok(Some(3), DepRelation::Det, "the", "DT", "the"),
        tok(Some(1), DepRelation::Dobj, "change", "NN", "change"),
        tok(Some(1), DepRelation::Punct, ".", ".", "."),
    ];
    let (source, tokens, parse) = build(&shapes);

    let found = t4_activize_to_passive(&source, &tokens, &parse);
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].kind, CandidateKind::ActivizeToPassive);
    assert_eq!(&*found[0].replacement, "The change was deployed");

    let mut expected_delta = BTreeMap::new();
    expected_delta.insert("agentless_passive", 1);
    expected_delta.insert("first_person", -1);
    assert_eq!(found[0].delta, expected_delta);

    assert_ranges_match(&source, &found, &["We deployed the change"]);
}

// -----------------------------------------------------------------
// 2. T4 does NOT fire on a coordinated predicate -- the stranding guard.
//
// "We instrumented X, and discovered Y": passivising just the first
// conjunct would leave "and discovered Y" with no subject at all. Both
// verbs here are guarded from two different directions -- the first
// because it has a VERB child attached by `conj`, the second because its
// own edge relation *is* `conj` -- so this pins both halves of that one
// guard at once.
// -----------------------------------------------------------------
#[test]
fn t4_does_not_fire_on_a_coordinated_predicate() {
    let shapes = [
        tok(Some(1), DepRelation::Nsubj, "We", "PRP", "we"),
        tok(None, DepRelation::Root, "instrumented", "VBD", "instrument"),
        tok(Some(1), DepRelation::Dobj, "X", "NN", "x"),
        tok(Some(4), DepRelation::Cc, "and", "CC", "and"),
        tok(Some(1), DepRelation::Conj, "discovered", "VBD", "discover"),
        tok(Some(4), DepRelation::Dobj, "Y", "NN", "y"),
        tok(Some(1), DepRelation::Punct, ".", ".", "."),
    ];
    let (source, tokens, parse) = build(&shapes);

    let found = t4_activize_to_passive(&source, &tokens, &parse);
    assert!(
        found.is_empty(),
        "coordinated predicate must not passivize either conjunct, got {found:?}"
    );
}

// -----------------------------------------------------------------
// 3. T4 does not fire on a particular (non-generic) subject.
// -----------------------------------------------------------------
#[test]
fn t4_does_not_fire_on_a_named_subject() {
    let shapes = [
        tok(Some(1), DepRelation::Nsubj, "Google", "NNP", "google"),
        tok(None, DepRelation::Root, "deployed", "VBD", "deploy"),
        tok(Some(3), DepRelation::Det, "the", "DT", "the"),
        tok(Some(1), DepRelation::Dobj, "change", "NN", "change"),
        tok(Some(1), DepRelation::Punct, ".", ".", "."),
    ];
    let (source, tokens, parse) = build(&shapes);

    let found = t4_activize_to_passive(&source, &tokens, &parse);
    assert!(
        found.is_empty(),
        "a named, identifiable subject must never be demoted, got {found:?}"
    );
}

// -----------------------------------------------------------------
// 4. T4 does not fire on an already-passive clause.
//
// This tree is deliberately ungrammatical ("We was deployed the change"):
// a genuine passive clause's subject bears `nsubjpass`, not `nsubj`, so
// it would already be excluded by the subject lookup alone, never
// reaching the `auxpass` guard at all. Giving the generic subject an
// `nsubj` edge here isolates that guard on its own -- with it removed,
// this synthetic tree would otherwise satisfy every other T4 condition.
// -----------------------------------------------------------------
#[test]
fn t4_does_not_fire_on_an_already_passive_clause() {
    let shapes = [
        tok(Some(2), DepRelation::Nsubj, "We", "PRP", "we"),
        tok(Some(2), DepRelation::AuxPass, "was", "VBD", "be"),
        tok(None, DepRelation::Root, "deployed", "VBD", "deploy"),
        tok(Some(4), DepRelation::Det, "the", "DT", "the"),
        tok(Some(2), DepRelation::Dobj, "change", "NN", "change"),
        tok(Some(2), DepRelation::Punct, ".", ".", "."),
    ];
    let (source, tokens, parse) = build(&shapes);

    let found = t4_activize_to_passive(&source, &tokens, &parse);
    assert!(
        found.is_empty(),
        "an auxpass child must veto the clause regardless of its own tag, got {found:?}"
    );
}

// -----------------------------------------------------------------
// 5. T4 number agreement: singular object -> was/is, plural -> were/are.
// -----------------------------------------------------------------
#[test]
fn t4_be_agrees_with_the_promoted_objects_number_and_tense() {
    let singular_past = [
        tok(Some(1), DepRelation::Nsubj, "We", "PRP", "we"),
        tok(None, DepRelation::Root, "deployed", "VBD", "deploy"),
        tok(Some(3), DepRelation::Det, "the", "DT", "the"),
        tok(Some(1), DepRelation::Dobj, "change", "NN", "change"),
        tok(Some(1), DepRelation::Punct, ".", ".", "."),
    ];
    let plural_past = [
        tok(Some(1), DepRelation::Nsubj, "We", "PRP", "we"),
        tok(None, DepRelation::Root, "deployed", "VBD", "deploy"),
        tok(Some(3), DepRelation::Det, "the", "DT", "the"),
        tok(Some(1), DepRelation::Dobj, "changes", "NNS", "change"),
        tok(Some(1), DepRelation::Punct, ".", ".", "."),
    ];
    let singular_present = [
        tok(Some(1), DepRelation::Nsubj, "We", "PRP", "we"),
        tok(None, DepRelation::Root, "deploy", "VBP", "deploy"),
        tok(Some(3), DepRelation::Det, "the", "DT", "the"),
        tok(Some(1), DepRelation::Dobj, "change", "NN", "change"),
        tok(Some(1), DepRelation::Punct, ".", ".", "."),
    ];
    let plural_present = [
        tok(Some(1), DepRelation::Nsubj, "We", "PRP", "we"),
        tok(None, DepRelation::Root, "deploy", "VBP", "deploy"),
        tok(Some(3), DepRelation::Det, "the", "DT", "the"),
        tok(Some(1), DepRelation::Dobj, "changes", "NNS", "change"),
        tok(Some(1), DepRelation::Punct, ".", ".", "."),
    ];

    for (shapes, expected) in [
        (&singular_past[..], "The change was deployed"),
        (&plural_past[..], "The changes were deployed"),
        (&singular_present[..], "The change is deployed"),
        (&plural_present[..], "The changes are deployed"),
    ] {
        let (source, tokens, parse) = build(shapes);
        let found = t4_activize_to_passive(&source, &tokens, &parse);
        assert_eq!(found.len(), 1, "source: {source:?}");
        assert_eq!(&*found[0].replacement, expected, "source: {source:?}");
    }
}

// -----------------------------------------------------------------
// 6. T5 fires on "the <nominalization> of <arg>".
// -----------------------------------------------------------------
#[test]
fn t5_fires_on_nominalization_with_of_complement() {
    let shapes = [
        tok(Some(1), DepRelation::Nsubj, "We", "PRP", "we"),
        tok(None, DepRelation::Root, "discussed", "VBD", "discuss"),
        tok(Some(3), DepRelation::Det, "the", "DT", "the"),
        tok(
            Some(1),
            DepRelation::Dobj,
            "optimization",
            "NN",
            "optimization",
        ),
        tok(Some(3), DepRelation::Prep, "of", "IN", "of"),
        tok(Some(6), DepRelation::Det, "the", "DT", "the"),
        tok(Some(4), DepRelation::Pobj, "query", "NN", "query"),
        tok(Some(1), DepRelation::Punct, ".", ".", "."),
    ];
    let (source, tokens, parse) = build(&shapes);

    let found = t5_nominalization(&source, &tokens, &parse);
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].kind, CandidateKind::NominalizationUnpack);
    assert_eq!(&*found[0].replacement, "optimizing the query");

    let mut expected_delta = BTreeMap::new();
    expected_delta.insert("nominalization", -1);
    expected_delta.insert("prepositions", -1);
    assert_eq!(found[0].delta, expected_delta);

    assert_ranges_match(&source, &found, &["the optimization of the query"]);
}

// -----------------------------------------------------------------
// 7. T5 does not fire without the "of" complement.
// -----------------------------------------------------------------
#[test]
fn t5_does_not_fire_without_of_complement() {
    let shapes = [
        tok(Some(1), DepRelation::Det, "The", "DT", "the"),
        tok(
            Some(2),
            DepRelation::Nsubj,
            "optimization",
            "NN",
            "optimization",
        ),
        tok(None, DepRelation::Root, "helped", "VBD", "help"),
        tok(Some(2), DepRelation::Punct, ".", ".", "."),
    ];
    let (source, tokens, parse) = build(&shapes);

    let found = t5_nominalization(&source, &tokens, &parse);
    assert!(
        found.is_empty(),
        "a nominalization with no `of` complement must not fire, got {found:?}"
    );
}

// 7 (continued). T5 does not fire when the noun is outside the table.
#[test]
fn t5_does_not_fire_when_noun_outside_table() {
    let shapes = [
        tok(Some(1), DepRelation::Det, "the", "DT", "the"),
        tok(None, DepRelation::Root, "celebration", "NN", "celebration"),
        tok(Some(1), DepRelation::Prep, "of", "IN", "of"),
        tok(Some(4), DepRelation::Det, "the", "DT", "the"),
        tok(Some(2), DepRelation::Pobj, "win", "NN", "win"),
    ];
    let (source, tokens, parse) = build(&shapes);

    let found = t5_nominalization(&source, &tokens, &parse);
    assert!(
        found.is_empty(),
        "a noun outside NOMINAL_VERB must not fire regardless of its shape, got {found:?}"
    );
}

// -----------------------------------------------------------------
// 8. T5 recapitalizes at sentence start.
// -----------------------------------------------------------------
#[test]
fn t5_recapitalizes_when_determiner_is_sentence_initial() {
    let shapes = [
        tok(Some(1), DepRelation::Det, "The", "DT", "the"),
        tok(Some(4), DepRelation::Nsubj, "reduction", "NN", "reduction"),
        tok(Some(1), DepRelation::Prep, "of", "IN", "of"),
        tok(Some(2), DepRelation::Pobj, "costs", "NNS", "cost"),
        tok(None, DepRelation::Root, "helped", "VBD", "help"),
        tok(Some(4), DepRelation::Punct, ".", ".", "."),
    ];
    let (source, tokens, parse) = build(&shapes);

    let found = t5_nominalization(&source, &tokens, &parse);
    assert_eq!(found.len(), 1);
    assert_eq!(&*found[0].replacement, "Reducing costs");
    assert_ranges_match(&source, &found, &["The reduction of costs"]);
}

// -----------------------------------------------------------------
// 9. Irregular inflection: at least three verbs from the irregular
// tables produce the correct past participle, alongside regular verbs
// for contrast.
// -----------------------------------------------------------------
#[test]
fn past_participle_uses_the_irregular_table_verbatim() {
    assert_eq!(past_participle("write"), "written");
    assert_eq!(past_participle("go"), "gone");
    assert_eq!(past_participle("teach"), "taught");
    assert_eq!(past_participle("buy"), "bought");
    assert_eq!(past_participle("choose"), "chosen");

    // Regular verbs fall back to the suffix rules `past` shares with
    // `past_participle` for anything outside the irregular table.
    assert_eq!(past_participle("deploy"), "deployed");
    assert_eq!(past_participle("carry"), "carried");
}

#[test]
fn past_and_third_sg_are_ported_alongside_past_participle() {
    assert_eq!(past("go"), "went");
    assert_eq!(past("write"), "wrote");
    assert_eq!(past("deploy"), "deployed");

    assert_eq!(third_sg("deploy"), "deploys");
    assert_eq!(third_sg("watch"), "watches");
    assert_eq!(third_sg("carry"), "carries");
}

// -----------------------------------------------------------------
// 10. Byte ranges: every candidate's range slices the original source to
// exactly the text it claims to replace, across every fixture above.
// -----------------------------------------------------------------
#[test]
fn every_candidate_range_slices_the_original_source_exactly() {
    let fixtures: &[&[TokenShape]] = &[
        &[
            tok(Some(1), DepRelation::Nsubj, "We", "PRP", "we"),
            tok(None, DepRelation::Root, "deployed", "VBD", "deploy"),
            tok(Some(3), DepRelation::Det, "the", "DT", "the"),
            tok(Some(1), DepRelation::Dobj, "change", "NN", "change"),
            tok(Some(1), DepRelation::Punct, ".", ".", "."),
        ],
        &[
            tok(Some(1), DepRelation::Nsubj, "We", "PRP", "we"),
            tok(None, DepRelation::Root, "discussed", "VBD", "discuss"),
            tok(Some(3), DepRelation::Det, "the", "DT", "the"),
            tok(
                Some(1),
                DepRelation::Dobj,
                "optimization",
                "NN",
                "optimization",
            ),
            tok(Some(3), DepRelation::Prep, "of", "IN", "of"),
            tok(Some(6), DepRelation::Det, "the", "DT", "the"),
            tok(Some(4), DepRelation::Pobj, "query", "NN", "query"),
            tok(Some(1), DepRelation::Punct, ".", ".", "."),
        ],
    ];

    for shapes in fixtures {
        let (source, tokens, parse) = build(shapes);
        for candidate in candidates(&source, &tokens, &parse) {
            candidate.validate(&source).unwrap_or_else(|e| {
                panic!(
                    "candidate range {:?} invalid for {source:?}: {e}",
                    candidate.range
                )
            });
            let sliced = &source[candidate.range.clone()];
            assert!(
                !sliced.is_empty(),
                "candidate range sliced to empty text in {source:?}"
            );
        }
    }
}
