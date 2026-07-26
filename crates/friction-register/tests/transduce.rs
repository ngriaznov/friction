//! Real cases for the two ported rewrite transducers, each with an
//! asserted output string — not smoke tests.
//!
//! Every parse is built by hand, not run through the shipped tagger/parser,
//! so a failure can only mean the transducer is wrong, never that the
//! parser mis-tagged or mis-parsed a fixture underneath it.

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
/// explicitly rather than derived from `surface` + `tag`: only
/// [`t4_activize_to_passive`] reads it, and deriving it here would route
/// fixtures through a third module's own lemmatization accuracy — the
/// same reason the parse is hand-built, not run through the real parser.
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

/// Builds source text, tagged tokens, and a validated [`SentenceParse`]
/// from hand-authored [`TokenShape`]s, joining surfaces with a single
/// space except before sentence-final punctuation.
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

/// Asserts every candidate's range slices `source` to exactly `expected` — the byte-honesty check every candidate here must pass.
fn assert_ranges_match(source: &str, found: &[Candidate], expected: &[&str]) {
    assert_eq!(found.len(), expected.len(), "candidate count mismatch");
    for (candidate, &want) in found.iter().zip(expected) {
        candidate
            .validate(source)
            .unwrap_or_else(|e| panic!("candidate range {:?} invalid: {e}", candidate.range));
        assert_eq!(&source[candidate.range.clone()], want);
    }
}

// 1. T4 fires: generic subject + direct object -> passive.
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

// 2. T4 does NOT fire on a coordinated predicate -- the stranding guard.
//
// "We instrumented X, and discovered Y": passivising just the first
// conjunct would leave "and discovered Y" with no subject. Both verbs
// are guarded from two directions -- one has a VERB child attached by
// `conj`, the other's own edge relation *is* `conj` -- pinning both
// halves of that one guard at once.
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

// 3. T4 does not fire on a particular (non-generic) subject.
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

// 4. T4 does not fire on an already-passive clause.
//
// Deliberately ungrammatical ("We was deployed the change"): a genuine
// passive subject bears `nsubjpass`, not `nsubj`, so it would already be
// excluded before reaching the `auxpass` guard. Giving the subject
// `nsubj` here isolates that guard alone -- remove it and this synthetic
// tree satisfies every other T4 condition.
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

// 5. T4 number agreement: singular object -> was/is, plural -> were/are.
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

// 5 (continued). T4 does not fire on a reflexive-pronoun object --
// pinned against a real sentence ("I ... tore myself away from ..."):
// "myself" has no independent referent to promote, so "Myself was torn
// away" is not a licensed rewrite.
#[test]
fn t4_does_not_fire_on_a_reflexive_pronoun_object() {
    let shapes = [
        tok(Some(2), DepRelation::Nsubj, "I", "PRP", "i"),
        tok(
            Some(2),
            DepRelation::Other,
            "reluctantly",
            "RB",
            "reluctantly",
        ),
        tok(None, DepRelation::Root, "tore", "VBD", "tear"),
        tok(Some(2), DepRelation::Dobj, "myself", "PRP", "myself"),
        tok(Some(2), DepRelation::Punct, ".", ".", "."),
    ];
    let (source, tokens, parse) = build(&shapes);

    let found = t4_activize_to_passive(&source, &tokens, &parse);
    assert!(
        found.is_empty(),
        "a reflexive-pronoun object must never be promoted to subject position, got {found:?}"
    );
}

// 5 (continued). T4 does not fire on stative "have" -- pinned against
// "we had about a day of downtime": "about a day of downtime was had" is
// not idiomatic English.
#[test]
fn t4_does_not_fire_on_stative_have() {
    let shapes = [
        tok(Some(1), DepRelation::Nsubj, "We", "PRP", "we"),
        tok(None, DepRelation::Root, "had", "VBD", "have"),
        tok(Some(3), DepRelation::Det, "a", "DT", "a"),
        tok(Some(1), DepRelation::Dobj, "day", "NN", "day"),
        tok(Some(1), DepRelation::Punct, ".", ".", "."),
    ];
    let (source, tokens, parse) = build(&shapes);

    let found = t4_activize_to_passive(&source, &tokens, &parse);
    assert!(
        found.is_empty(),
        "stative \"have\" must never be promoted to a passive, got {found:?}"
    );
}

// 5 (continued). T4 does not fire on the copula "become" -- pinned
// against "before they become problems": "problems are become" is not
// English.
#[test]
fn t4_does_not_fire_on_the_copula_become() {
    let shapes = [
        tok(Some(1), DepRelation::Nsubj, "They", "PRP", "they"),
        tok(None, DepRelation::Root, "become", "VBP", "become"),
        tok(Some(1), DepRelation::Dobj, "problems", "NNS", "problem"),
    ];
    let (source, tokens, parse) = build(&shapes);

    let found = t4_activize_to_passive(&source, &tokens, &parse);
    assert!(
        found.is_empty(),
        "the copula \"become\" must never be promoted to a passive, got {found:?}"
    );
}

// 5 (continued). T4 does not fire when the verb's lemma wasn't reduced
// to its base form -- pinned against "rewrote the migration script",
// where an unreduced lemma "rewrote" would produce "the migration script
// was rewroted" once regularly suffixed.
#[test]
fn t4_does_not_fire_when_the_verbs_lemma_was_not_reduced_from_its_surface() {
    let shapes = [
        tok(Some(1), DepRelation::Nsubj, "We", "PRP", "we"),
        tok(None, DepRelation::Root, "rewrote", "VBD", "rewrote"),
        tok(Some(3), DepRelation::Det, "the", "DT", "the"),
        tok(Some(1), DepRelation::Dobj, "script", "NN", "script"),
    ];
    let (source, tokens, parse) = build(&shapes);

    let found = t4_activize_to_passive(&source, &tokens, &parse);
    assert!(
        found.is_empty(),
        "an unreduced verb lemma must hold the candidate rather than double-inflect it, got {found:?}"
    );
}

// 5 (continued). T4 does not fire when the object isn't a plausible
// nominal -- pinned against "felt most productive", where an adjectival
// complement was mislabeled `dobj`.
#[test]
fn t4_does_not_fire_on_a_non_nominal_object() {
    let shapes = [
        tok(Some(1), DepRelation::Nsubj, "They", "PRP", "they"),
        tok(None, DepRelation::Root, "felt", "VBD", "feel"),
        tok(Some(1), DepRelation::Dobj, "productive", "JJ", "productive"),
    ];
    let (source, tokens, parse) = build(&shapes);

    let found = t4_activize_to_passive(&source, &tokens, &parse);
    assert!(
        found.is_empty(),
        "a non-nominal object must never be promoted, got {found:?}"
    );
}

// 5 (continued). T4 does not fire when the object's subtree contains a
// finite verb, or is directly followed by one -- pinned against "we
// anticipate this reporting workload will continue to grow", where the
// parser flattened an embedded clause into a `dobj`.
#[test]
fn t4_does_not_fire_when_a_finite_verb_directly_follows_the_object() {
    let shapes = [
        tok(Some(1), DepRelation::Nsubj, "We", "PRP", "we"),
        tok(None, DepRelation::Root, "anticipate", "VBP", "anticipate"),
        tok(Some(3), DepRelation::Det, "this", "DT", "this"),
        tok(Some(1), DepRelation::Dobj, "workload", "NN", "workload"),
        tok(Some(1), DepRelation::Other, "will", "MD", "will"),
    ];
    let (source, tokens, parse) = build(&shapes);

    let found = t4_activize_to_passive(&source, &tokens, &parse);
    assert!(
        found.is_empty(),
        "a dangling finite verb right after the object must hold the candidate, got {found:?}"
    );
}

// 5 (continued). T4 does not fire when the object's subtree ends on a
// bare preposition -- pinned against "established a rough budget range
// of $60,000 - $85,000", where the range's argument attached elsewhere,
// stranding "of" and producing "a rough budget range of was
// established".
#[test]
fn t4_does_not_fire_when_the_object_ends_on_a_bare_preposition() {
    let shapes = [
        tok(Some(1), DepRelation::Nsubj, "We", "PRP", "we"),
        tok(None, DepRelation::Root, "established", "VBD", "establish"),
        tok(Some(4), DepRelation::Det, "a", "DT", "a"),
        tok(Some(4), DepRelation::Amod, "rough", "JJ", "rough"),
        tok(Some(1), DepRelation::Dobj, "range", "NN", "range"),
        // Attached under the object's subtree (as the mis-parse does),
        // but with no `pobj` argument -- its true argument attached
        // elsewhere in the real sentence.
        tok(Some(4), DepRelation::Prep, "of", "IN", "of"),
    ];
    let (source, tokens, parse) = build(&shapes);

    let found = t4_activize_to_passive(&source, &tokens, &parse);
    assert!(
        found.is_empty(),
        "an object ending on a bare preposition must hold the candidate, got {found:?}"
    );
}

#[test]
fn t4_does_not_fire_when_the_objects_own_subtree_contains_a_finite_verb() {
    let shapes = [
        tok(Some(1), DepRelation::Nsubj, "They", "PRP", "they"),
        tok(None, DepRelation::Root, "found", "VBD", "find"),
        tok(Some(1), DepRelation::Dobj, "reasons", "NNS", "reason"),
        // A relative clause wrongly folded into the object's own subtree
        // instead of attaching above it.
        tok(Some(2), DepRelation::Acl, "that", "IN", "that"),
        tok(Some(3), DepRelation::Nsubj, "it", "PRP", "it"),
        tok(Some(3), DepRelation::Root, "worked", "VBD", "work"),
    ];
    let (source, tokens, parse) = build(&shapes);

    let found = t4_activize_to_passive(&source, &tokens, &parse);
    assert!(
        found.is_empty(),
        "an object subtree containing its own finite verb must hold the candidate, got {found:?}"
    );
}

// 5 (continued). T4 never promotes a span containing Markdown structural
// syntax -- pinned against a bridged bold placeholder
// ("**[Proposed Cutover Date ...]**") that reached the parser as literal
// prose and produced a garbage candidate.
#[test]
fn t4_does_not_fire_across_markdown_structural_syntax() {
    let shapes = [
        tok(Some(1), DepRelation::Nsubj, "We", "PRP", "we"),
        tok(None, DepRelation::Root, "scheduled", "VBD", "schedule"),
        tok(Some(1), DepRelation::Dobj, "date", "NN", "date"),
        // Falls inside the span this transducer would otherwise promote
        // -- mirroring how a bridged bold placeholder's asterisk lands
        // adjacent to real prose tokens in the actual parse.
        tok(Some(2), DepRelation::Other, "*", "SYM", "*"),
    ];
    let (source, tokens, parse) = build(&shapes);

    let found = t4_activize_to_passive(&source, &tokens, &parse);
    assert!(
        found.is_empty(),
        "a span containing Markdown structural syntax must never be promoted, got {found:?}"
    );
}

// 5 (continued). A personal pronoun is never promoted into subject
// position; no candidate is produced.
//
// Replaces an earlier test asserting "You are encouraged" and "Them are
// plucked" -- grammatically right but "Them are plucked" is not a
// sentence anyone would write.
//
// Two independent reasons, either sufficient. A ditransitive verb's
// indirect object is often mislabeled `dobj`, so promoting it names the
// wrong participant -- measured on "see how much time and effort they
// save you", which yielded "...you are saved", stranding the real
// object. And a pronoun carries almost no information, so promoting it
// buys nothing even when it's the right object.
//
// Number agreement is covered separately, on real objects, by
// `t4_be_agrees_with_the_promoted_objects_number_and_tense`.
#[test]
fn t4_does_not_promote_a_personal_pronoun_object() {
    let cases = [
        ("encourage", "VBP", "you", "PRP"),
        ("pluck", "VBP", "them", "PRP"),
        ("notified", "VBD", "her", "PRP"),
    ];

    for (verb, verb_tag, object, object_tag) in cases {
        let shapes = [
            tok(Some(1), DepRelation::Nsubj, "We", "PRP", "we"),
            tok(None, DepRelation::Root, verb, verb_tag, verb),
            tok(Some(1), DepRelation::Dobj, object, object_tag, object),
        ];
        let (source, tokens, parse) = build(&shapes);
        let found = t4_activize_to_passive(&source, &tokens, &parse);
        assert!(
            found.is_empty(),
            "a personal-pronoun object must never be promoted, got {found:?} for {source:?}"
        );
    }
}

// 5 (continued). T4 recapitalizes the promoted object only when its
// candidate range opens the sentence -- pinned against "... when I found
// an exception", which must become "... when an exception was found",
// not "... when An exception was found".
#[test]
fn t4_does_not_recapitalize_a_mid_sentence_promoted_object() {
    let shapes = [
        tok(Some(2), DepRelation::Other, "when", "IN", "when"),
        tok(Some(2), DepRelation::Nsubj, "I", "PRP", "i"),
        tok(None, DepRelation::Root, "found", "VBD", "find"),
        tok(Some(4), DepRelation::Det, "an", "DT", "a"),
        tok(Some(2), DepRelation::Dobj, "exception", "NN", "exception"),
    ];
    let (source, tokens, parse) = build(&shapes);

    let found = t4_activize_to_passive(&source, &tokens, &parse);
    assert_eq!(found.len(), 1);
    assert_eq!(&*found[0].replacement, "an exception was found");
}

// 6. T5 fires on "the <nominalization> of <arg>".
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

// 7. T5 does not fire without the "of" complement.
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

// 7 (continued). T5 does not fire when the pobj's subtree is immediately
// followed by a bare noun -- the stranded compound-noun-tail guard,
// pinned against "the integration of the third-party analytics SDK",
// where a parser mis-attachment left "SDK" outside the `pobj` subtree.
#[test]
fn t5_does_not_fire_when_a_bare_noun_immediately_follows_the_pobj_subtree() {
    let shapes = [
        tok(Some(1), DepRelation::Nsubj, "We", "PRP", "we"),
        tok(None, DepRelation::Root, "finalized", "VBD", "finalize"),
        tok(Some(3), DepRelation::Det, "the", "DT", "the"),
        tok(
            Some(1),
            DepRelation::Dobj,
            "integration",
            "NN",
            "integration",
        ),
        tok(Some(3), DepRelation::Prep, "of", "IN", "of"),
        tok(Some(8), DepRelation::Det, "the", "DT", "the"),
        // "third-party" attaches under the pobj's subtree (as a real
        // parse would), but "SDK" -- the phrase's true final noun -- is
        // mis-attached to the root verb, stranding it just past the
        // subtree this transducer would otherwise use as its argument.
        tok(
            Some(8),
            DepRelation::Other,
            "third-party",
            "NN",
            "third-party",
        ),
        tok(Some(4), DepRelation::Pobj, "analytics", "NNS", "analytics"),
        tok(Some(1), DepRelation::Other, "SDK", "NNP", "sdk"),
        tok(Some(1), DepRelation::Punct, ".", ".", "."),
    ];
    let (source, tokens, parse) = build(&shapes);

    let found = t5_nominalization(&source, &tokens, &parse);
    assert!(
        found.is_empty(),
        "a stranded compound-noun tail must hold the candidate rather than drop content, got {found:?}"
    );
}

// 8. T5 recapitalizes at sentence start.
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

// 9. Irregular inflection: verbs from the irregular tables produce the
// correct past participle, alongside regular verbs for contrast.
#[test]
fn past_participle_uses_the_irregular_table_verbatim() {
    assert_eq!(past_participle("write"), "written");
    assert_eq!(past_participle("go"), "gone");
    assert_eq!(past_participle("teach"), "taught");
    assert_eq!(past_participle("buy"), "bought");
    assert_eq!(past_participle("choose"), "chosen");

    // Invariant verbs (base == past == past participle) -- added after a
    // real corpus document exposed "hit" falling through to the regular
    // "-ed" suffix rule and producing "hited".
    assert_eq!(past_participle("hit"), "hit");
    assert_eq!(past_participle("cost"), "cost");
    assert_eq!(past_participle("shut"), "shut");
    assert_eq!(past_participle("spread"), "spread");

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

// 10. Byte ranges: every candidate's range slices the source to exactly
// the text it claims to replace, across every fixture above.
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
