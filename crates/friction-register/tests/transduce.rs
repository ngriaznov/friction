//! Real cases for the five rewrite transducers, each with an asserted
//! output string: not smoke tests.
//!
//! Every parse is built by hand, not run through the shipped tagger/parser,
//! so a failure can only mean the transducer is wrong, never that the
//! parser mis-tagged or mis-parsed a fixture underneath it.

use std::collections::BTreeMap;

use friction_core::{Token, TokenKind};
use friction_nlp::{Confidence, DepEdge, DepRelation, PosTag, SentenceParse, TaggedToken};
use friction_register::transduce::{
    Candidate, CandidateKind, candidates, past, past_participle, t4_activize_to_passive,
    t5_nominalization, t6_em_dash, t7_semicolon, t9_past_progressive, third_sg,
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

// 5 (continued). T4 number agreement over a coordinated object: the
// whole promoted phrase is plural even when its head conjunct is
// singular. Pinned against real output where head-only agreement
// produced "a thumbnail, title, price, and two badges is rendered".
#[test]
fn t4_be_agrees_with_a_coordinated_object_not_just_its_head() {
    let shapes = [
        tok(Some(1), DepRelation::Nsubj, "We", "PRP", "we"),
        tok(None, DepRelation::Root, "render", "VBP", "render"),
        tok(Some(3), DepRelation::Det, "a", "DT", "a"),
        tok(Some(1), DepRelation::Dobj, "thumbnail", "NN", "thumbnail"),
        tok(Some(3), DepRelation::Punct, ",", ",", ","),
        tok(Some(3), DepRelation::Conj, "title", "NN", "title"),
        tok(Some(3), DepRelation::Punct, ",", ",", ","),
        tok(Some(3), DepRelation::Conj, "price", "NN", "price"),
        tok(Some(3), DepRelation::Punct, ",", ",", ","),
        tok(Some(3), DepRelation::Cc, "and", "CC", "and"),
        tok(Some(11), DepRelation::Other, "two", "CD", "two"),
        tok(Some(3), DepRelation::Conj, "badges", "NNS", "badge"),
        tok(Some(1), DepRelation::Punct, ".", ".", "."),
    ];
    let (source, tokens, parse) = build(&shapes);

    let found = t4_activize_to_passive(&source, &tokens, &parse);
    assert_eq!(found.len(), 1);
    assert_eq!(
        &*found[0].replacement,
        "A thumbnail, title, price, and two badges are rendered"
    );
    assert_ranges_match(
        &source,
        &found,
        &["We render a thumbnail, title, price, and two badges"],
    );
}

// 5 (continued). "or"/"nor" coordination instead agrees with the
// conjunct nearest the verb slot: blanket-plural would trade one
// agreement error for another ("thumbnails or a badge are rendered").
#[test]
fn t4_or_coordination_agrees_with_the_nearest_conjunct() {
    let shapes = [
        tok(Some(1), DepRelation::Nsubj, "We", "PRP", "we"),
        tok(None, DepRelation::Root, "render", "VBP", "render"),
        tok(Some(1), DepRelation::Dobj, "thumbnails", "NNS", "thumbnail"),
        tok(Some(2), DepRelation::Cc, "or", "CC", "or"),
        tok(Some(5), DepRelation::Det, "a", "DT", "a"),
        tok(Some(2), DepRelation::Conj, "badge", "NN", "badge"),
        tok(Some(1), DepRelation::Punct, ".", ".", "."),
    ];
    let (source, tokens, parse) = build(&shapes);

    let found = t4_activize_to_passive(&source, &tokens, &parse);
    assert_eq!(found.len(), 1);
    assert_eq!(&*found[0].replacement, "Thumbnails or a badge is rendered");
    assert_ranges_match(&source, &found, &["We render thumbnails or a badge"]);
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
// position. No candidate is produced.
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

// 9 (continued). Consonant doubling follows stress, not just letter
// shape: an unstressed final syllable never doubles ("render" surfaced
// as "renderred" in real T4 output), while a stressed one does.
#[test]
fn past_participle_doubles_only_stressed_final_syllables() {
    assert_eq!(past_participle("render"), "rendered");
    assert_eq!(past_participle("offer"), "offered");
    assert_eq!(past_participle("order"), "ordered");
    assert_eq!(past_participle("prefer"), "preferred");

    assert_eq!(past("render"), "rendered");
    assert_eq!(past("prefer"), "preferred");
}

#[test]
fn past_and_third_sg_ship_alongside_past_participle() {
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

// ---------------------------------------------------------------------
// T6: em-dash reduction.
// ---------------------------------------------------------------------

/// [`TokenShape`] plus whether this token glues to the previous one with
/// no space at all -- `build`'s spacing rule (a space before every token
/// except a fixed sentence-final-punctuation set) can't spell an
/// unspaced em dash ("word—word"), which T6's cases (b)/(c) must also
/// cover.
struct GluedTokenShape {
    head: Option<usize>,
    relation: DepRelation,
    surface: &'static str,
    tag: &'static str,
    lemma: &'static str,
    glued: bool,
}

const fn g(
    head: Option<usize>,
    relation: DepRelation,
    surface: &'static str,
    tag: &'static str,
    lemma: &'static str,
) -> GluedTokenShape {
    GluedTokenShape {
        head,
        relation,
        surface,
        tag,
        lemma,
        glued: false,
    }
}

const fn glued(
    head: Option<usize>,
    relation: DepRelation,
    surface: &'static str,
    tag: &'static str,
    lemma: &'static str,
) -> GluedTokenShape {
    GluedTokenShape {
        head,
        relation,
        surface,
        tag,
        lemma,
        glued: true,
    }
}

fn build_glued(shapes: &[GluedTokenShape]) -> (String, Vec<TaggedToken>, SentenceParse) {
    let mut source = String::new();
    let mut tokens = Vec::with_capacity(shapes.len());
    for (index, item) in shapes.iter().enumerate() {
        let attaches_directly =
            item.glued || matches!(item.surface, "." | "," | "!" | "?" | ";" | ":");
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

// 11. Case (a): a paired " — X — " interpolation with no finite verb in
// X collapses to ", X, ".
#[test]
fn t6_fires_on_a_paired_parenthetical_with_no_finite_verb_inside() {
    let shapes = [
        g(Some(1), DepRelation::Det, "The", "DT", "the"),
        g(Some(2), DepRelation::Nsubj, "service", "NN", "service"),
        g(None, DepRelation::Root, "reads", "VBZ", "read"),
        g(Some(2), DepRelation::Dobj, "config", "NN", "config"),
        g(Some(2), DepRelation::Punct, "\u{2014}", "HYPH", "\u{2014}"),
        g(Some(2), DepRelation::Other, "not", "RB", "not"),
        g(Some(2), DepRelation::Other, "secrets", "NNS", "secret"),
        g(Some(2), DepRelation::Punct, "\u{2014}", "HYPH", "\u{2014}"),
        g(Some(2), DepRelation::Prep, "from", "IN", "from"),
        g(Some(8), DepRelation::Pobj, "disk", "NN", "disk"),
        g(Some(2), DepRelation::Punct, ".", ".", "."),
    ];
    let (source, tokens, parse) = build_glued(&shapes);

    let found = t6_em_dash(&source, &tokens, &parse);
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].kind, CandidateKind::EmDash);
    assert_eq!(&*found[0].replacement, ", not secrets, ");
    assert!(!found[0].replacement.contains('\u{2014}'));

    let mut expected_delta = BTreeMap::new();
    expected_delta.insert("em_dash", -2);
    assert_eq!(found[0].delta, expected_delta);

    let mut fixed = source;
    fixed.replace_range(found[0].range.clone(), &found[0].replacement);
    assert_eq!(fixed, "The service reads config, not secrets, from disk.");
}

// 12. Case (b): a single em dash with no finite verb after it (a
// fragment/elaboration) collapses to ", ".
#[test]
fn t6_fires_on_a_fragment_with_no_finite_verb_after_the_dash() {
    let shapes = [
        g(Some(1), DepRelation::Nsubj, "It", "PRP", "it"),
        g(None, DepRelation::Root, "is", "VBZ", "be"),
        g(Some(4), DepRelation::Det, "a", "DT", "a"),
        g(Some(4), DepRelation::Other, "registry", "NN", "registry"),
        g(Some(1), DepRelation::Other, "entry", "NN", "entry"),
        g(Some(1), DepRelation::Punct, "\u{2014}", "HYPH", "\u{2014}"),
        g(Some(8), DepRelation::Det, "no", "DT", "no"),
        g(Some(8), DepRelation::Other, "lockstep", "NN", "lockstep"),
        g(Some(1), DepRelation::Other, "deploy", "NN", "deploy"),
        g(Some(1), DepRelation::Punct, ".", ".", "."),
    ];
    let (source, tokens, parse) = build_glued(&shapes);

    let found = t6_em_dash(&source, &tokens, &parse);
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].kind, CandidateKind::EmDash);
    assert_eq!(&*found[0].replacement, ", ");

    let mut expected_delta = BTreeMap::new();
    expected_delta.insert("em_dash", -1);
    assert_eq!(found[0].delta, expected_delta);

    let mut fixed = source;
    fixed.replace_range(found[0].range.clone(), &found[0].replacement);
    assert_eq!(fixed, "It is a registry entry, no lockstep deploy.");
}

// 13. Case (b), unspaced form: "word—word" with no surrounding spaces is
// handled the same as the spaced form.
#[test]
fn t6_fires_on_an_unspaced_fragment_dash() {
    let shapes = [
        g(Some(1), DepRelation::Nsubj, "Config", "NN", "config"),
        g(None, DepRelation::Root, "comes", "VBZ", "come"),
        g(Some(1), DepRelation::Prep, "from", "IN", "from"),
        g(Some(2), DepRelation::Pobj, "env", "NN", "env"),
        glued(Some(1), DepRelation::Punct, "\u{2014}", "HYPH", "\u{2014}"),
        glued(Some(6), DepRelation::Det, "no", "DT", "no"),
        g(Some(1), DepRelation::Other, "flags", "NNS", "flag"),
        g(Some(6), DepRelation::Other, "required", "VBN", "require"),
        g(Some(1), DepRelation::Punct, ".", ".", "."),
    ];
    let (source, tokens, parse) = build_glued(&shapes);
    assert_eq!(source, "Config comes from env\u{2014}no flags required.");

    let found = t6_em_dash(&source, &tokens, &parse);
    assert_eq!(found.len(), 1);
    assert_eq!(&source[found[0].range.clone()], "\u{2014}");
    assert_eq!(&*found[0].replacement, ", ");

    let mut fixed = source;
    fixed.replace_range(found[0].range.clone(), &found[0].replacement);
    assert_eq!(fixed, "Config comes from env, no flags required.");
}

// 14. Case (c): a single em dash followed by an independent clause (its
// own finite verb and subject) splits into two sentences, recapitalizing
// the new sentence's first word.
#[test]
fn t6_fires_on_an_independent_clause_and_recapitalizes() {
    let shapes = [
        g(Some(1), DepRelation::Nsubj, "It", "PRP", "it"),
        g(None, DepRelation::Root, "runs", "VBZ", "run"),
        g(Some(1), DepRelation::Prep, "on", "IN", "on"),
        g(Some(5), DepRelation::Det, "the", "DT", "the"),
        g(Some(5), DepRelation::Amod, "internal", "JJ", "internal"),
        g(Some(2), DepRelation::Pobj, "network", "NN", "network"),
        g(Some(1), DepRelation::Punct, "\u{2014}", "HYPH", "\u{2014}"),
        g(Some(8), DepRelation::Nsubj, "it", "PRP", "it"),
        g(Some(1), DepRelation::Other, "is", "VBZ", "be"),
        g(Some(8), DepRelation::Other, "never", "RB", "never"),
        g(Some(8), DepRelation::Other, "reached", "VBN", "reach"),
        g(Some(10), DepRelation::Prep, "by", "IN", "by"),
        g(Some(13), DepRelation::Other, "end", "NN", "end"),
        g(Some(11), DepRelation::Pobj, "users", "NNS", "user"),
        g(Some(1), DepRelation::Punct, ".", ".", "."),
    ];
    let (source, tokens, parse) = build_glued(&shapes);

    let found = t6_em_dash(&source, &tokens, &parse);
    assert_eq!(found.len(), 1);
    assert_eq!(&*found[0].replacement, ". It");
    assert!(!found[0].replacement.contains('\u{2014}'));

    let mut expected_delta = BTreeMap::new();
    expected_delta.insert("em_dash", -1);
    assert_eq!(found[0].delta, expected_delta);

    let mut fixed = source;
    fixed.replace_range(found[0].range.clone(), &found[0].replacement);
    assert_eq!(
        fixed,
        "It runs on the internal network. It is never reached by end users."
    );
}

// 15. Case (c), trivial: the word right after the dash is already
// capitalized (a proper noun), so recapitalizing is a no-op.
#[test]
fn t6_case_c_is_trivial_when_the_following_word_is_already_capitalized() {
    let shapes = [
        g(Some(1), DepRelation::Nsubj, "It", "PRP", "it"),
        g(None, DepRelation::Root, "talks", "VBZ", "talk"),
        g(Some(1), DepRelation::Prep, "to", "IN", "to"),
        g(Some(4), DepRelation::Det, "the", "DT", "the"),
        g(Some(2), DepRelation::Pobj, "cache", "NN", "cache"),
        g(Some(1), DepRelation::Punct, "\u{2014}", "HYPH", "\u{2014}"),
        g(Some(7), DepRelation::Nsubj, "Redis", "NNP", "redis"),
        g(Some(1), DepRelation::Other, "handles", "VBZ", "handle"),
        g(Some(7), DepRelation::Other, "eviction", "NN", "eviction"),
        g(Some(1), DepRelation::Punct, ".", ".", "."),
    ];
    let (source, tokens, parse) = build_glued(&shapes);

    let found = t6_em_dash(&source, &tokens, &parse);
    assert_eq!(found.len(), 1);
    assert_eq!(&*found[0].replacement, ". Redis");

    let mut fixed = source;
    fixed.replace_range(found[0].range.clone(), &found[0].replacement);
    assert_eq!(fixed, "It talks to the cache. Redis handles eviction.");
}

// 16. Case (c), semicolon fallback: the word right after the dash starts
// with a digit, which can't be recapitalized, so the replacement is a
// semicolon instead of a period.
#[test]
fn t6_case_c_falls_back_to_a_semicolon_when_the_following_word_cannot_be_capitalized() {
    let shapes = [
        g(Some(1), DepRelation::Det, "The", "DT", "the"),
        g(Some(2), DepRelation::Nsubj, "queue", "NN", "queue"),
        g(None, DepRelation::Root, "holds", "VBZ", "hold"),
        g(Some(2), DepRelation::Dobj, "items", "NNS", "item"),
        g(Some(2), DepRelation::Punct, "\u{2014}", "HYPH", "\u{2014}"),
        g(Some(6), DepRelation::Other, "3", "CD", "3"),
        g(Some(7), DepRelation::Nsubj, "workers", "NNS", "worker"),
        g(Some(2), DepRelation::Other, "drain", "VBP", "drain"),
        g(Some(7), DepRelation::Dobj, "it", "PRP", "it"),
        g(Some(7), DepRelation::Other, "fast", "RB", "fast"),
        g(Some(2), DepRelation::Punct, ".", ".", "."),
    ];
    let (source, tokens, parse) = build_glued(&shapes);

    let found = t6_em_dash(&source, &tokens, &parse);
    assert_eq!(found.len(), 1);
    assert_eq!(&*found[0].replacement, "; 3");
    assert!(!found[0].replacement.contains('\u{2014}'));

    let mut fixed = source;
    fixed.replace_range(found[0].range.clone(), &found[0].replacement);
    assert_eq!(fixed, "The queue holds items; 3 workers drain it fast.");
}

// 17. An en dash (U+2013) -- as in a numeric range -- never produces a
// candidate, spaced or not.
#[test]
fn t6_never_fires_on_an_en_dash() {
    let shapes = [
        g(None, DepRelation::Root, "Supported", "VBN", "support"),
        g(Some(0), DepRelation::Other, "on", "IN", "on"),
        g(Some(0), DepRelation::Other, "pages", "NNS", "page"),
        g(Some(0), DepRelation::Other, "1", "CD", "1"),
        glued(Some(0), DepRelation::Other, "\u{2013}", "HYPH", "\u{2013}"),
        glued(Some(0), DepRelation::Other, "64", "CD", "64"),
        g(Some(0), DepRelation::Punct, ".", ".", "."),
    ];
    let (source, tokens, parse) = build_glued(&shapes);
    assert_eq!(source, "Supported on pages 1\u{2013}64.");
    assert!(t6_em_dash(&source, &tokens, &parse).is_empty());
}

// 18. A sentence with three em dashes is too ambiguous to decompose into
// a single ordered pair of edits, so it produces no candidate at all.
#[test]
fn t6_declines_a_sentence_with_three_em_dashes() {
    let shapes = [
        g(Some(1), DepRelation::Nsubj, "It", "PRP", "it"),
        g(None, DepRelation::Root, "works", "VBZ", "work"),
        g(Some(1), DepRelation::Punct, "\u{2014}", "HYPH", "\u{2014}"),
        g(Some(1), DepRelation::Other, "sometimes", "RB", "sometimes"),
        g(Some(1), DepRelation::Punct, "\u{2014}", "HYPH", "\u{2014}"),
        g(Some(1), DepRelation::Cc, "but", "CC", "but"),
        g(Some(1), DepRelation::Conj, "fails", "VBZ", "fail"),
        g(Some(6), DepRelation::Punct, "\u{2014}", "HYPH", "\u{2014}"),
        g(Some(6), DepRelation::Other, "often", "RB", "often"),
        g(Some(1), DepRelation::Punct, ".", ".", "."),
    ];
    let (source, tokens, parse) = build_glued(&shapes);
    assert!(t6_em_dash(&source, &tokens, &parse).is_empty());
}

// 19. A dash immediately followed by a backtick-flanked span never
// produces a candidate, even when every other case-(c) condition holds --
// the inline-code guard takes priority over the independent-clause
// rewrite.
#[test]
fn t6_declines_when_the_span_would_cross_an_inline_code_boundary() {
    let shapes = [
        g(Some(1), DepRelation::Nsubj, "It", "PRP", "it"),
        g(None, DepRelation::Root, "reads", "VBZ", "read"),
        g(Some(3), DepRelation::Det, "the", "DT", "the"),
        g(Some(1), DepRelation::Dobj, "flag", "NN", "flag"),
        g(Some(1), DepRelation::Punct, "\u{2014}", "HYPH", "\u{2014}"),
        g(Some(6), DepRelation::Nsubj, "`debug`", "NN", "debug"),
        g(Some(1), DepRelation::Other, "enables", "VBZ", "enable"),
        g(Some(6), DepRelation::Amod, "verbose", "JJ", "verbose"),
        g(Some(6), DepRelation::Dobj, "output", "NN", "output"),
        g(Some(1), DepRelation::Punct, ".", ".", "."),
    ];
    let (source, tokens, parse) = build_glued(&shapes);
    assert!(source.contains('`'));
    assert!(t6_em_dash(&source, &tokens, &parse).is_empty());
}

// 20. `t6_em_dash` is folded into `candidates()` alongside T4/T5.
#[test]
fn candidates_includes_t6_em_dash_output() {
    let shapes = [
        g(Some(1), DepRelation::Nsubj, "It", "PRP", "it"),
        g(None, DepRelation::Root, "is", "VBZ", "be"),
        g(Some(4), DepRelation::Det, "a", "DT", "a"),
        g(Some(4), DepRelation::Other, "registry", "NN", "registry"),
        g(Some(1), DepRelation::Other, "entry", "NN", "entry"),
        g(Some(1), DepRelation::Punct, "\u{2014}", "HYPH", "\u{2014}"),
        g(Some(8), DepRelation::Det, "no", "DT", "no"),
        g(Some(8), DepRelation::Other, "lockstep", "NN", "lockstep"),
        g(Some(1), DepRelation::Other, "deploy", "NN", "deploy"),
        g(Some(1), DepRelation::Punct, ".", ".", "."),
    ];
    let (source, tokens, parse) = build_glued(&shapes);
    let found = candidates(&source, &tokens, &parse);
    assert!(found.iter().any(|c| c.kind == CandidateKind::EmDash));
}

// 21. A verbless left side is a definition lead-in: the dash becomes a
// colon, regardless of what the right side parses as — even a full
// independent clause, which after a comma would be a splice.
#[test]
fn t6_uses_a_colon_after_a_verbless_lead_in() {
    let shapes = [
        g(Some(1), DepRelation::Other, "Rate", "NN", "rate"),
        g(Some(5), DepRelation::Nsubj, "limiting", "NN", "limiting"),
        g(Some(5), DepRelation::Punct, "\u{2014}", "HYPH", "\u{2014}"),
        g(Some(4), DepRelation::Det, "the", "DT", "the"),
        g(Some(5), DepRelation::Nsubj, "design", "NN", "design"),
        g(None, DepRelation::Root, "calls", "VBZ", "call"),
        g(Some(5), DepRelation::Prep, "for", "IN", "for"),
        g(Some(6), DepRelation::Pobj, "limits", "NNS", "limit"),
        g(Some(5), DepRelation::Punct, ".", ".", "."),
    ];
    let (source, tokens, parse) = build_glued(&shapes);

    let found = t6_em_dash(&source, &tokens, &parse);
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].kind, CandidateKind::EmDash);
    assert_eq!(&*found[0].replacement, ": ");

    let mut expected_delta = BTreeMap::new();
    expected_delta.insert("em_dash", -1);
    assert_eq!(found[0].delta, expected_delta);
}

// 22. A paired interpolation that carries its own commas is set off with
// parentheses, not a comma pair — comma delimiters around a comma-bearing
// list would flatten the aside into one long list.
#[test]
fn t6_paired_parenthetical_with_internal_commas_uses_parentheses() {
    let shapes = [
        g(Some(1), DepRelation::Nsubj, "It", "PRP", "it"),
        g(None, DepRelation::Root, "returns", "VBZ", "return"),
        g(Some(1), DepRelation::Dobj, "everything", "NN", "everything"),
        g(Some(1), DepRelation::Punct, "\u{2014}", "HYPH", "\u{2014}"),
        g(Some(1), DepRelation::Other, "metrics", "NNS", "metric"),
        g(Some(4), DepRelation::Punct, ",", ",", ","),
        g(
            Some(4),
            DepRelation::Other,
            "dimensions",
            "NNS",
            "dimension",
        ),
        g(Some(4), DepRelation::Punct, ",", ",", ","),
        g(Some(4), DepRelation::Other, "and", "CC", "and"),
        g(Some(4), DepRelation::Other, "buckets", "NNS", "bucket"),
        g(Some(1), DepRelation::Punct, "\u{2014}", "HYPH", "\u{2014}"),
        g(Some(1), DepRelation::Prep, "in", "IN", "in"),
        g(Some(11), DepRelation::Pobj, "one", "CD", "one"),
        g(Some(11), DepRelation::Other, "call", "NN", "call"),
        g(Some(1), DepRelation::Punct, ".", ".", "."),
    ];
    let (source, tokens, parse) = build_glued(&shapes);

    let found = t6_em_dash(&source, &tokens, &parse);
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].kind, CandidateKind::EmDash);
    assert_eq!(
        &*found[0].replacement,
        " (metrics, dimensions, and buckets) "
    );
    assert!(!found[0].replacement.contains('\u{2014}'));

    let mut expected_delta = BTreeMap::new();
    expected_delta.insert("em_dash", -2);
    assert_eq!(found[0].delta, expected_delta);
}

// ---------------------------------------------------------------------
// T7: semicolon-splice reduction.
// ---------------------------------------------------------------------

// 23. A semicolon joining two independent clauses (both sides carry a
// finite verb and their own subject) splits into two sentences,
// recapitalizing the new sentence's first word — the one rewrite shape
// this module has.
#[test]
fn t7_fires_when_both_sides_are_independent_clauses() {
    let shapes = [
        g(Some(1), DepRelation::Nsubj, "It", "PRP", "it"),
        g(None, DepRelation::Root, "runs", "VBZ", "run"),
        g(Some(1), DepRelation::Prep, "on", "IN", "on"),
        g(Some(5), DepRelation::Det, "the", "DT", "the"),
        g(Some(5), DepRelation::Amod, "internal", "JJ", "internal"),
        g(Some(2), DepRelation::Pobj, "network", "NN", "network"),
        g(Some(1), DepRelation::Punct, ";", ":", ";"),
        g(Some(8), DepRelation::Nsubj, "it", "PRP", "it"),
        g(Some(1), DepRelation::Other, "is", "VBZ", "be"),
        g(Some(8), DepRelation::Other, "never", "RB", "never"),
        g(Some(8), DepRelation::Other, "reached", "VBN", "reach"),
        g(Some(10), DepRelation::Prep, "by", "IN", "by"),
        g(Some(13), DepRelation::Other, "end", "NN", "end"),
        g(Some(11), DepRelation::Pobj, "users", "NNS", "user"),
        g(Some(1), DepRelation::Punct, ".", ".", "."),
    ];
    let (source, tokens, parse) = build_glued(&shapes);

    let found = t7_semicolon(&source, &tokens, &parse);
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].kind, CandidateKind::Semicolon);
    assert_eq!(&*found[0].replacement, ". It");
    assert!(!found[0].replacement.contains(';'));

    let mut expected_delta = BTreeMap::new();
    expected_delta.insert("semicolon", -1);
    assert_eq!(found[0].delta, expected_delta);

    let mut fixed = source;
    fixed.replace_range(found[0].range.clone(), &found[0].replacement);
    assert_eq!(
        fixed,
        "It runs on the internal network. It is never reached by end users."
    );
}

// 24. An elliptical right side (no finite clause of its own) declines:
//     a decline, not a comma or any other substitute.
#[test]
fn t7_declines_when_the_right_side_is_elliptical() {
    let shapes = [
        g(Some(1), DepRelation::Nsubj, "It", "PRP", "it"),
        g(None, DepRelation::Root, "works", "VBZ", "work"),
        g(Some(1), DepRelation::Prep, "at", "IN", "at"),
        g(Some(4), DepRelation::Det, "the", "DT", "the"),
        g(Some(2), DepRelation::Pobj, "edge", "NN", "edge"),
        g(Some(1), DepRelation::Punct, ";", ":", ";"),
        g(Some(7), DepRelation::Other, "not", "RB", "not"),
        g(
            Some(1),
            DepRelation::Other,
            "implemented",
            "VBN",
            "implement",
        ),
        g(Some(7), DepRelation::Prep, "in", "IN", "in"),
        g(Some(10), DepRelation::Det, "this", "DT", "this"),
        g(Some(8), DepRelation::Pobj, "iteration", "NN", "iteration"),
        g(Some(1), DepRelation::Punct, ".", ".", "."),
    ];
    let (source, tokens, parse) = build_glued(&shapes);
    assert!(t7_semicolon(&source, &tokens, &parse).is_empty());
}

// 25. A serial "super-comma" list (two semicolons, one segment (the
//     last) verbless) produces no candidate for either semicolon:
//     with a verbless segment present, there is no principled way to
//     tell which semicolon (if any) is a genuine clause boundary
//     rather than a list separator, so this declines the whole
//     sentence.
#[test]
fn t7_declines_every_semicolon_in_a_serial_list_with_a_verbless_segment() {
    let shapes = [
        g(Some(1), DepRelation::Det, "The", "DT", "the"),
        g(Some(2), DepRelation::Nsubj, "office", "NN", "office"),
        g(None, DepRelation::Root, "handles", "VBZ", "handle"),
        g(Some(2), DepRelation::Dobj, "onboarding", "NN", "onboarding"),
        g(Some(2), DepRelation::Punct, ";", ":", ";"),
        g(Some(6), DepRelation::Det, "the", "DT", "the"),
        g(Some(7), DepRelation::Nsubj, "site", "NN", "site"),
        g(Some(2), DepRelation::Other, "handles", "VBZ", "handle"),
        g(Some(7), DepRelation::Dobj, "billing", "NN", "billing"),
        g(Some(2), DepRelation::Punct, ";", ":", ";"),
        g(Some(2), DepRelation::Cc, "and", "CC", "and"),
        g(Some(13), DepRelation::Det, "the", "DT", "the"),
        g(Some(13), DepRelation::Amod, "remote", "JJ", "remote"),
        g(Some(2), DepRelation::Other, "team", "NN", "team"),
        g(Some(2), DepRelation::Punct, ".", ".", "."),
    ];
    let (source, tokens, parse) = build_glued(&shapes);
    assert!(t7_semicolon(&source, &tokens, &parse).is_empty());
}

// 26. Once every segment carries its own finite clause, a two-semicolon
// sentence yields two independent candidates — the counterpart to test
// 25, which shares its two-semicolon shape but declines because one
// segment lacks a verb.
#[test]
fn t7_fires_independently_on_each_semicolon_when_every_segment_is_finite() {
    let shapes = [
        g(Some(1), DepRelation::Nsubj, "It", "PRP", "it"),
        g(None, DepRelation::Root, "reads", "VBZ", "read"),
        g(Some(3), DepRelation::Det, "the", "DT", "the"),
        g(Some(1), DepRelation::Dobj, "queue", "NN", "queue"),
        g(Some(1), DepRelation::Punct, ";", ":", ";"),
        g(Some(6), DepRelation::Nsubj, "it", "PRP", "it"),
        g(Some(1), DepRelation::Other, "commits", "VBZ", "commit"),
        g(Some(6), DepRelation::Dobj, "offsets", "NNS", "offset"),
        g(Some(1), DepRelation::Punct, ";", ":", ";"),
        g(Some(10), DepRelation::Nsubj, "it", "PRP", "it"),
        g(Some(1), DepRelation::Other, "acks", "VBZ", "ack"),
        g(Some(13), DepRelation::Det, "the", "DT", "the"),
        g(Some(10), DepRelation::Dobj, "batch", "NN", "batch"),
        g(Some(1), DepRelation::Punct, ".", ".", "."),
    ];
    let (source, tokens, parse) = build_glued(&shapes);

    let found = t7_semicolon(&source, &tokens, &parse);
    assert_eq!(found.len(), 2);
    assert!(found.iter().all(|c| c.kind == CandidateKind::Semicolon));
    assert!(found.iter().all(|c| &*c.replacement == ". It"));

    let mut fixed = source;
    // Apply both, later range first, so the earlier byte range stays valid.
    let mut sorted = found;
    sorted.sort_by_key(|c| std::cmp::Reverse(c.range.start));
    for candidate in &sorted {
        fixed.replace_range(candidate.range.clone(), &candidate.replacement);
    }
    assert_eq!(
        fixed,
        "It reads the queue. It commits offsets. It acks the batch."
    );
}

// 27. The word right after the semicolon starts with a digit, which
// can't be recapitalized — declines outright, unlike T6's case (c),
// which has a semicolon fallback available. There is no fallback for a
// semicolon: the source already has one.
#[test]
fn t7_declines_when_the_following_word_cannot_be_capitalized() {
    let shapes = [
        g(Some(1), DepRelation::Det, "The", "DT", "the"),
        g(Some(2), DepRelation::Nsubj, "queue", "NN", "queue"),
        g(None, DepRelation::Root, "holds", "VBZ", "hold"),
        g(Some(2), DepRelation::Dobj, "items", "NNS", "item"),
        g(Some(2), DepRelation::Punct, ";", ":", ";"),
        g(Some(6), DepRelation::Other, "3", "CD", "3"),
        g(Some(7), DepRelation::Nsubj, "workers", "NNS", "worker"),
        g(Some(2), DepRelation::Other, "drain", "VBP", "drain"),
        g(Some(7), DepRelation::Dobj, "it", "PRP", "it"),
        g(Some(7), DepRelation::Other, "fast", "RB", "fast"),
        g(Some(2), DepRelation::Punct, ".", ".", "."),
    ];
    let (source, tokens, parse) = build_glued(&shapes);
    assert!(t7_semicolon(&source, &tokens, &parse).is_empty());
}

// 28. A semicolon immediately followed by a backtick-flanked span never
//     produces a candidate, even when every other condition holds. The
//     inline-code guard takes priority.
#[test]
fn t7_declines_when_the_span_would_cross_an_inline_code_boundary() {
    let shapes = [
        g(Some(1), DepRelation::Nsubj, "It", "PRP", "it"),
        g(None, DepRelation::Root, "reads", "VBZ", "read"),
        g(Some(3), DepRelation::Det, "the", "DT", "the"),
        g(Some(1), DepRelation::Dobj, "flag", "NN", "flag"),
        g(Some(1), DepRelation::Punct, ";", ":", ";"),
        g(Some(6), DepRelation::Nsubj, "`debug`", "NN", "debug"),
        g(Some(1), DepRelation::Other, "enables", "VBZ", "enable"),
        g(Some(6), DepRelation::Amod, "verbose", "JJ", "verbose"),
        g(Some(6), DepRelation::Dobj, "output", "NN", "output"),
        g(Some(1), DepRelation::Punct, ".", ".", "."),
    ];
    let (source, tokens, parse) = build_glued(&shapes);
    assert!(source.contains('`'));
    assert!(t7_semicolon(&source, &tokens, &parse).is_empty());
}

// 29. Only the ASCII semicolon (U+003B) counts: a Greek question mark
// (U+037E), a visual lookalike in most fonts, never fires.
#[test]
fn t7_never_fires_on_a_greek_question_mark() {
    let shapes = [
        g(Some(1), DepRelation::Nsubj, "It", "PRP", "it"),
        g(None, DepRelation::Root, "works", "VBZ", "work"),
        g(Some(1), DepRelation::Punct, "\u{37e}", ".", "\u{37e}"),
    ];
    let (source, tokens, parse) = build_glued(&shapes);
    assert!(source.contains('\u{37e}'));
    assert!(t7_semicolon(&source, &tokens, &parse).is_empty());
}

// 30. `t7_semicolon` is folded into `candidates()` alongside T4/T5/T6.
#[test]
fn candidates_includes_t7_semicolon_output() {
    let shapes = [
        g(Some(1), DepRelation::Nsubj, "It", "PRP", "it"),
        g(None, DepRelation::Root, "runs", "VBZ", "run"),
        g(Some(1), DepRelation::Prep, "on", "IN", "on"),
        g(Some(5), DepRelation::Det, "the", "DT", "the"),
        g(Some(5), DepRelation::Amod, "internal", "JJ", "internal"),
        g(Some(2), DepRelation::Pobj, "network", "NN", "network"),
        g(Some(1), DepRelation::Punct, ";", ":", ";"),
        g(Some(8), DepRelation::Nsubj, "it", "PRP", "it"),
        g(Some(1), DepRelation::Other, "is", "VBZ", "be"),
        g(Some(8), DepRelation::Other, "never", "RB", "never"),
        g(Some(8), DepRelation::Other, "reached", "VBN", "reach"),
        g(Some(10), DepRelation::Prep, "by", "IN", "by"),
        g(Some(13), DepRelation::Other, "end", "NN", "end"),
        g(Some(11), DepRelation::Pobj, "users", "NNS", "user"),
        g(Some(1), DepRelation::Punct, ".", ".", "."),
    ];
    let (source, tokens, parse) = build_glued(&shapes);
    let found = candidates(&source, &tokens, &parse);
    assert!(found.iter().any(|c| c.kind == CandidateKind::Semicolon));
}

// 23. A fused subject+finite contraction right after the dash is an
// independent clause, not a fragment — the tagger tags "it's" PRP, and a
// comma there is the measured splice from real prose.
#[test]
fn t6_treats_a_subject_contraction_after_the_dash_as_a_clause() {
    let shapes = [
        g(Some(3), DepRelation::Nsubj, "None", "NN", "none"),
        g(Some(0), DepRelation::Prep, "of", "IN", "of"),
        g(Some(1), DepRelation::Pobj, "it", "PRP", "it"),
        g(None, DepRelation::Root, "is", "VBZ", "be"),
        g(Some(3), DepRelation::Other, "wrong", "JJ", "wrong"),
        g(Some(3), DepRelation::Punct, "\u{2014}", "HYPH", "\u{2014}"),
        g(Some(8), DepRelation::Nsubj, "it's", "PRP", "it's"),
        g(Some(8), DepRelation::Other, "machine", "NN", "machine"),
        g(Some(3), DepRelation::Other, "register", "NN", "register"),
        g(Some(3), DepRelation::Punct, ".", ".", "."),
    ];
    let (source, tokens, parse) = build_glued(&shapes);

    let found = t6_em_dash(&source, &tokens, &parse);
    assert_eq!(found.len(), 1);
    assert!(
        found[0].replacement.starts_with(". "),
        "expected a sentence break, got {:?}",
        found[0].replacement
    );
    assert!(found[0].replacement.contains("It's"));
}

// 24. An imperative right side ("— pull in the published module") is a
// complete command: sentence break, never a comma splice.
#[test]
fn t6_treats_an_imperative_after_the_dash_as_a_clause() {
    let shapes = [
        g(Some(1), DepRelation::Nsubj, "packages", "NNS", "package"),
        g(None, DepRelation::Root, "aren't", "VBP", "be"),
        g(
            Some(1),
            DepRelation::Other,
            "importable",
            "JJ",
            "importable",
        ),
        g(Some(1), DepRelation::Punct, "\u{2014}", "HYPH", "\u{2014}"),
        g(Some(1), DepRelation::Other, "pull", "VB", "pull"),
        g(Some(4), DepRelation::Prep, "in", "IN", "in"),
        g(Some(7), DepRelation::Det, "the", "DT", "the"),
        g(Some(5), DepRelation::Pobj, "module", "NN", "module"),
        g(Some(1), DepRelation::Punct, ".", ".", "."),
    ];
    let (source, tokens, parse) = build_glued(&shapes);

    let found = t6_em_dash(&source, &tokens, &parse);
    assert_eq!(found.len(), 1);
    assert!(
        found[0].replacement.starts_with(". "),
        "expected a sentence break, got {:?}",
        found[0].replacement
    );
    assert!(found[0].replacement.contains("Pull"));
}

// 25. A comma-bearing fragment takes a colon, not a bare comma — a comma
// delimiter would flatten it into one long false list (the measured
// "forward, faster, simpler, and easier" mush).
#[test]
fn t6_uses_a_colon_before_a_comma_bearing_fragment() {
    let shapes = [
        g(Some(1), DepRelation::Nsubj, "This", "DT", "this"),
        g(None, DepRelation::Root, "pushes", "VBZ", "push"),
        g(Some(1), DepRelation::Dobj, "React", "NNP", "react"),
        g(Some(1), DepRelation::Other, "forward", "RB", "forward"),
        g(Some(1), DepRelation::Punct, "\u{2014}", "HYPH", "\u{2014}"),
        g(Some(1), DepRelation::Other, "faster", "JJR", "fast"),
        g(Some(5), DepRelation::Punct, ",", ",", ","),
        g(Some(5), DepRelation::Other, "simpler", "JJR", "simple"),
        g(Some(5), DepRelation::Punct, ",", ",", ","),
        g(Some(5), DepRelation::Other, "and", "CC", "and"),
        g(Some(5), DepRelation::Other, "easier", "JJR", "easy"),
        g(Some(1), DepRelation::Punct, ".", ".", "."),
    ];
    let (source, tokens, parse) = build_glued(&shapes);

    let found = t6_em_dash(&source, &tokens, &parse);
    assert_eq!(found.len(), 1);
    assert_eq!(&*found[0].replacement, ": ");
}

// 26. A colon-introduced enumeration keeps its semicolons even when every
//     item carries its own finite clause. The semicolons are the
//     construction's correct separators, and splitting any of them breaks
//     the enumeration's symmetry.
#[test]
fn t7_declines_every_semicolon_in_a_colon_introduced_enumeration() {
    let shapes = [
        g(Some(1), DepRelation::Nsubj, "It", "PRP", "it"),
        g(None, DepRelation::Root, "runs", "VBZ", "run"),
        g(Some(1), DepRelation::Prep, "in", "IN", "in"),
        g(Some(2), DepRelation::Pobj, "stages", "NNS", "stage"),
        g(Some(1), DepRelation::Punct, ":", ":", ":"),
        g(Some(6), DepRelation::Nsubj, "Intro", "NNP", "intro"),
        g(Some(1), DepRelation::Other, "teaches", "VBZ", "teach"),
        g(Some(6), DepRelation::Dobj, "basics", "NNS", "basic"),
        g(Some(1), DepRelation::Punct, ";", ";", ";"),
        g(Some(10), DepRelation::Nsubj, "Core", "NNP", "core"),
        g(Some(1), DepRelation::Other, "covers", "VBZ", "cover"),
        g(Some(10), DepRelation::Dobj, "years", "NNS", "year"),
        g(Some(1), DepRelation::Punct, ".", ".", "."),
    ];
    let (source, tokens, parse) = build_glued(&shapes);
    assert!(t7_semicolon(&source, &tokens, &parse).is_empty());
}

// ---------------------------------------------------------------------
// T9: past-progressive simplification.
// ---------------------------------------------------------------------

// 38. A regular verb's past progressive collapses to its simple past,
// via friction_nlp::past.
#[test]
fn t9_fires_on_a_regular_past_progressive() {
    let shapes = [
        g(Some(2), DepRelation::Nsubj, "She", "PRP", "she"),
        g(Some(2), DepRelation::Aux, "was", "VBD", "be"),
        g(None, DepRelation::Root, "marking", "VBG", "mark"),
        g(Some(4), DepRelation::Det, "the", "DT", "the"),
        g(Some(2), DepRelation::Dobj, "papers", "NNS", "paper"),
        g(Some(2), DepRelation::Punct, ".", ".", "."),
    ];
    let (source, tokens, parse) = build_glued(&shapes);

    let found = t9_past_progressive(&source, &tokens, &parse);
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].kind, CandidateKind::PastProgressive);
    assert_eq!(&*found[0].replacement, "marked");

    let mut expected_delta = BTreeMap::new();
    expected_delta.insert("past_progressive", -1);
    assert_eq!(found[0].delta, expected_delta);

    let mut fixed = source;
    fixed.replace_range(found[0].range.clone(), &found[0].replacement);
    assert_eq!(fixed, "She marked the papers.");
}

// 39. An irregular verb's past progressive uses the irregular table, the
// same lookup `past` shares with `past_participle`.
#[test]
fn t9_fires_on_an_irregular_past_progressive() {
    let shapes = [
        g(Some(2), DepRelation::Nsubj, "He", "PRP", "he"),
        g(Some(2), DepRelation::Aux, "was", "VBD", "be"),
        g(None, DepRelation::Root, "writing", "VBG", "write"),
        g(Some(4), DepRelation::Det, "the", "DT", "the"),
        g(Some(2), DepRelation::Dobj, "docs", "NNS", "doc"),
        g(Some(2), DepRelation::Punct, ".", ".", "."),
    ];
    let (source, tokens, parse) = build_glued(&shapes);

    let found = t9_past_progressive(&source, &tokens, &parse);
    assert_eq!(found.len(), 1);
    assert_eq!(&*found[0].replacement, "wrote");

    let mut fixed = source;
    fixed.replace_range(found[0].range.clone(), &found[0].replacement);
    assert_eq!(fixed, "He wrote the docs.");
}

// A participle directly before a bare noun reads as an attributive
// adjective whatever the parse says: "support were compelling factors"
// carried a mis-attached aux edge from the real parser and produced
// "compelled factors" on a corpus fixture during snapshot review. The
// bare-noun decline is fail-closed: a genuine progressive with a
// bare-plural object ("was writing docs") declines too, by design.
#[test]
fn t9_declines_a_participle_directly_before_a_bare_noun() {
    let shapes = [
        g(
            Some(2),
            DepRelation::Nsubj,
            "Guarantees",
            "NNS",
            "guarantee",
        ),
        g(Some(2), DepRelation::Aux, "were", "VBD", "be"),
        g(None, DepRelation::Root, "compelling", "VBG", "compel"),
        g(Some(2), DepRelation::Dobj, "factors", "NNS", "factor"),
        g(Some(2), DepRelation::Punct, ".", ".", "."),
    ];
    let (source, tokens, parse) = build_glued(&shapes);
    assert!(t9_past_progressive(&source, &tokens, &parse).is_empty());
}

// 40. Capitalization transfers from the auxiliary: a sentence-initial
// "Was" produces a capitalized replacement.
#[test]
fn t9_transfers_capitalization_from_the_auxiliary() {
    let shapes = [
        g(Some(1), DepRelation::Aux, "Was", "VBD", "be"),
        g(None, DepRelation::Root, "marking", "VBG", "mark"),
        g(Some(3), DepRelation::Det, "every", "DT", "every"),
        g(Some(1), DepRelation::Dobj, "commit", "NN", "commit"),
        g(Some(1), DepRelation::Punct, ".", ".", "."),
    ];
    let (source, tokens, parse) = build_glued(&shapes);
    assert_eq!(source, "Was marking every commit.");

    let found = t9_past_progressive(&source, &tokens, &parse);
    assert_eq!(found.len(), 1);
    assert_eq!(&*found[0].replacement, "Marked");

    let mut fixed = source;
    fixed.replace_range(found[0].range.clone(), &found[0].replacement);
    assert_eq!(fixed, "Marked every commit.");
}

// 41. "when"/"while" anywhere in the sentence declines the whole
// sentence: the progressive may be doing real aspectual work against a
// temporal frame, and homing a register band never licenses a meaning
// change.
#[test]
fn t9_declines_when_a_temporal_frame_word_is_anywhere_in_the_sentence() {
    for frame_word in ["when", "while"] {
        let shapes = [
            g(Some(2), DepRelation::Nsubj, "It", "PRP", "it"),
            g(Some(2), DepRelation::Aux, "was", "VBD", "be"),
            g(None, DepRelation::Root, "marking", "VBG", "mark"),
            g(Some(4), DepRelation::Det, "the", "DT", "the"),
            g(Some(2), DepRelation::Dobj, "change", "NN", "change"),
            g(Some(2), DepRelation::Other, frame_word, "WRB", frame_word),
            g(Some(2), DepRelation::Other, "review", "NN", "review"),
            g(Some(2), DepRelation::Other, "started", "VBD", "start"),
            g(Some(2), DepRelation::Punct, ".", ".", "."),
        ];
        let (source, tokens, parse) = build_glued(&shapes);
        assert!(
            t9_past_progressive(&source, &tokens, &parse).is_empty(),
            "a {frame_word:?} anywhere in the sentence must hold every candidate"
        );
    }
}

// 42. An adverb between the auxiliary and the participle declines:
// collapsing "was quietly marking" forces an adverb-placement decision no
// table can make.
#[test]
fn t9_declines_when_an_adverb_sits_between_the_auxiliary_and_the_participle() {
    let shapes = [
        g(Some(3), DepRelation::Nsubj, "She", "PRP", "she"),
        g(Some(3), DepRelation::Aux, "was", "VBD", "be"),
        g(Some(3), DepRelation::Other, "quietly", "RB", "quietly"),
        g(None, DepRelation::Root, "marking", "VBG", "mark"),
        g(Some(3), DepRelation::Dobj, "review", "NN", "review"),
        g(Some(3), DepRelation::Punct, ".", ".", "."),
    ];
    let (source, tokens, parse) = build_glued(&shapes);
    assert!(t9_past_progressive(&source, &tokens, &parse).is_empty());
}

// 43. A "being"/"going" participle declines: neither is the plain
// progressive this transducer targets.
#[test]
fn t9_declines_on_being_and_going_participles() {
    let being_shapes = [
        g(Some(2), DepRelation::Nsubj, "She", "PRP", "she"),
        g(Some(2), DepRelation::Aux, "was", "VBD", "be"),
        g(None, DepRelation::Root, "being", "VBG", "be"),
        g(Some(2), DepRelation::Other, "helpful", "JJ", "helpful"),
        g(Some(2), DepRelation::Punct, ".", ".", "."),
    ];
    let going_shapes = [
        g(Some(2), DepRelation::Nsubj, "She", "PRP", "she"),
        g(Some(2), DepRelation::Aux, "was", "VBD", "be"),
        g(None, DepRelation::Root, "going", "VBG", "go"),
        g(Some(2), DepRelation::Other, "home", "NN", "home"),
        g(Some(2), DepRelation::Punct, ".", ".", "."),
    ];

    for shapes in [&being_shapes[..], &going_shapes[..]] {
        let (source, tokens, parse) = build_glued(shapes);
        assert!(
            t9_past_progressive(&source, &tokens, &parse).is_empty(),
            "source: {source:?}"
        );
    }
}

// 44. An auxiliary not attached to the participle by its own `aux` edge
// declines: a copula taking an adjectival predicate that merely happens
// to carry a `VBG` tag ("The plan was promising") is not a genuine
// progressive, and rewriting it to "The plan promised" changes the
// claim's meaning outright.
#[test]
fn t9_declines_when_the_auxiliary_is_not_the_participles_own_aux_child() {
    let shapes = [
        g(Some(1), DepRelation::Det, "The", "DT", "the"),
        g(Some(2), DepRelation::Nsubj, "plan", "NN", "plan"),
        g(None, DepRelation::Root, "was", "VBD", "be"),
        g(Some(2), DepRelation::Other, "promising", "VBG", "promise"),
        g(Some(2), DepRelation::Punct, ".", ".", "."),
    ];
    let (source, tokens, parse) = build_glued(&shapes);
    assert!(t9_past_progressive(&source, &tokens, &parse).is_empty());
}

// 45. `t9_past_progressive` is folded into `candidates()` alongside
// T4-T8.
#[test]
fn candidates_includes_t9_past_progressive_output() {
    let shapes = [
        g(Some(2), DepRelation::Nsubj, "She", "PRP", "she"),
        g(Some(2), DepRelation::Aux, "was", "VBD", "be"),
        g(None, DepRelation::Root, "marking", "VBG", "mark"),
        g(Some(4), DepRelation::Det, "the", "DT", "the"),
        g(Some(2), DepRelation::Dobj, "papers", "NNS", "paper"),
        g(Some(2), DepRelation::Punct, ".", ".", "."),
    ];
    let (source, tokens, parse) = build_glued(&shapes);
    let found = candidates(&source, &tokens, &parse);
    assert!(
        found
            .iter()
            .any(|c| c.kind == CandidateKind::PastProgressive)
    );
}
