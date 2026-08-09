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
    Candidate, CandidateKind, RestructureOutcome, SubstitutionCandidate, candidates, past,
    past_participle, t4_activize_to_passive, t5_nominalization, t6_em_dash, t7_semicolon,
    t9_past_progressive, t10_ensures_that_restructure, t11_transitive_substitution, third_sg,
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

// ---------------------------------------------------------------------
// T10: "ensure that NP is/are VBN" -> imperative/infinitival "VB NP".
//
// Every fixture's `lemma` is given explicitly, same convention as every
// other transducer's tests in this file (see this file's own module
// docs) -- except where a fixture deliberately exercises the shipped
// tagger's own silent-"e" lemmatization gap, which lives in
// `friction-register/src/transduce.rs`'s own `t10_private_tests` module
// instead (`t10_derive_base_verb` is private, unreachable from here).
// ---------------------------------------------------------------------

/// Asserts `outcomes` is exactly one [`RestructureOutcome::Candidate`]
/// whose range slices `source` to `expected_span` and whose replacement
/// is `expected_replacement`.
fn assert_single_candidate(
    source: &str,
    outcomes: &[RestructureOutcome],
    expected_span: &str,
    expected_replacement: &str,
) {
    assert_eq!(outcomes.len(), 1, "outcomes: {outcomes:?}");
    match &outcomes[0] {
        RestructureOutcome::Candidate { range, replacement } => {
            assert_eq!(&source[range.clone()], expected_span);
            assert_eq!(&**replacement, expected_replacement);
        }
        other @ RestructureOutcome::Declined { .. } => {
            panic!("expected a Candidate, got {other:?}")
        }
    }
}

/// Asserts `outcomes` is exactly one [`RestructureOutcome::Declined`]
/// whose `reason` is `expected_reason`.
fn assert_single_decline(outcomes: &[RestructureOutcome], expected_reason: &str) {
    assert_eq!(outcomes.len(), 1, "outcomes: {outcomes:?}");
    match &outcomes[0] {
        RestructureOutcome::Declined { reason, .. } => assert_eq!(*reason, expected_reason),
        other @ RestructureOutcome::Candidate { .. } => {
            panic!("expected a Declined, got {other:?}")
        }
    }
}

// T10-1. Accept: sentence-initial imperative, the design's own flagship
// example.
#[test]
fn t10_accepts_sentence_initial_imperative() {
    let shapes = [
        tok(None, DepRelation::Root, "Ensure", "VB", "ensure"),
        tok(Some(5), DepRelation::Mark, "that", "IN", "that"),
        tok(Some(3), DepRelation::Det, "the", "DT", "the"),
        tok(Some(5), DepRelation::NsubjPass, "setting", "NN", "setting"),
        tok(Some(5), DepRelation::AuxPass, "is", "VBZ", "be"),
        tok(Some(0), DepRelation::Ccomp, "enabled", "VBN", "enable"),
        tok(Some(0), DepRelation::Punct, ".", ".", "."),
    ];
    let (source, tokens, parse) = build(&shapes);
    let found = t10_ensures_that_restructure(&source, &tokens, &parse);
    assert_single_candidate(
        &source,
        &found,
        "Ensure that the setting is enabled",
        "Enable the setting",
    );
}

// T10-2. Accept: infinitival "to ensure that" -- "to" stays untouched.
#[test]
fn t10_accepts_infinitival() {
    let shapes = [
        tok(Some(1), DepRelation::Aux, "to", "TO", "to"),
        tok(None, DepRelation::Root, "ensure", "VB", "ensure"),
        tok(Some(6), DepRelation::Mark, "that", "IN", "that"),
        tok(Some(4), DepRelation::Det, "the", "DT", "the"),
        tok(Some(6), DepRelation::NsubjPass, "setting", "NN", "setting"),
        tok(Some(6), DepRelation::AuxPass, "is", "VBZ", "be"),
        tok(Some(1), DepRelation::Ccomp, "enabled", "VBN", "enable"),
    ];
    let (source, tokens, parse) = build(&shapes);
    let found = t10_ensures_that_restructure(&source, &tokens, &parse);
    assert_single_candidate(
        &source,
        &found,
        "ensure that the setting is enabled",
        "enable the setting",
    );
}

// T10-3. Accept: NP with a coordinated subject -- the whole coordinated
// phrase is promoted, not just the head conjunct.
#[test]
fn t10_accepts_a_coordinated_subject() {
    let shapes = [
        tok(None, DepRelation::Root, "Ensure", "VB", "ensure"),
        tok(Some(7), DepRelation::Mark, "that", "IN", "that"),
        tok(Some(3), DepRelation::Det, "the", "DT", "the"),
        tok(
            Some(7),
            DepRelation::NsubjPass,
            "settings",
            "NNS",
            "setting",
        ),
        tok(Some(3), DepRelation::Cc, "and", "CC", "and"),
        tok(Some(3), DepRelation::Conj, "flags", "NNS", "flag"),
        tok(Some(7), DepRelation::AuxPass, "are", "VBP", "be"),
        tok(Some(0), DepRelation::Ccomp, "enabled", "VBN", "enable"),
        tok(Some(0), DepRelation::Punct, ".", ".", "."),
    ];
    let (source, tokens, parse) = build(&shapes);
    let found = t10_ensures_that_restructure(&source, &tokens, &parse);
    assert_single_candidate(
        &source,
        &found,
        "Ensure that the settings and flags are enabled",
        "Enable the settings and flags",
    );
}

// T10-4. Accept: NP with a determiner other than "the".
#[test]
fn t10_accepts_a_determiner_other_than_the() {
    let shapes = [
        tok(None, DepRelation::Root, "Ensure", "VB", "ensure"),
        tok(Some(5), DepRelation::Mark, "that", "IN", "that"),
        tok(Some(3), DepRelation::Det, "this", "DT", "this"),
        tok(
            Some(5),
            DepRelation::NsubjPass,
            "configuration",
            "NN",
            "configuration",
        ),
        tok(Some(5), DepRelation::AuxPass, "is", "VBZ", "be"),
        tok(Some(0), DepRelation::Ccomp, "enabled", "VBN", "enable"),
        tok(Some(0), DepRelation::Punct, ".", ".", "."),
    ];
    let (source, tokens, parse) = build(&shapes);
    let found = t10_ensures_that_restructure(&source, &tokens, &parse);
    assert_single_candidate(
        &source,
        &found,
        "Ensure that this configuration is enabled",
        "Enable this configuration",
    );
}

// T10-5. Decline: an adverb between the auxiliary and the participle
// forces an adverb-placement decision no table can make (T9's own
// precedent).
#[test]
fn t10_declines_on_an_adverb_gap() {
    let shapes = [
        tok(None, DepRelation::Root, "Ensure", "VB", "ensure"),
        tok(Some(6), DepRelation::Mark, "that", "IN", "that"),
        tok(Some(3), DepRelation::Det, "the", "DT", "the"),
        tok(Some(6), DepRelation::NsubjPass, "setting", "NN", "setting"),
        tok(Some(6), DepRelation::AuxPass, "is", "VBZ", "be"),
        tok(Some(6), DepRelation::Other, "properly", "RB", "properly"),
        tok(Some(0), DepRelation::Ccomp, "enabled", "VBN", "enable"),
        tok(Some(0), DepRelation::Punct, ".", ".", "."),
    ];
    let (source, tokens, parse) = build(&shapes);
    let found = t10_ensures_that_restructure(&source, &tokens, &parse);
    assert_single_decline(&found, "adverb between the auxiliary and the participle");
}

// T10-6. Decline: negation via a fused "isn't" auxiliary -- the single
// highest-stakes guard in this transducer, its own dedicated test.
// Missing this turns "ensure telemetry isn't sent" into "Send
// telemetry", the opposite of what the author wrote.
#[test]
fn t10_declines_on_negation_via_fused_auxiliary() {
    let shapes = [
        tok(None, DepRelation::Root, "Ensure", "VB", "ensure"),
        tok(Some(4), DepRelation::Mark, "that", "IN", "that"),
        tok(
            Some(4),
            DepRelation::NsubjPass,
            "telemetry",
            "NN",
            "telemetry",
        ),
        tok(Some(4), DepRelation::AuxPass, "isn't", "VBZ", "isn't"),
        tok(Some(0), DepRelation::Ccomp, "sent", "VBN", "send"),
        tok(Some(0), DepRelation::Punct, ".", ".", "."),
    ];
    let (source, tokens, parse) = build(&shapes);
    let found = t10_ensures_that_restructure(&source, &tokens, &parse);
    assert_single_decline(&found, "negation would invert the sentence's meaning");
}

// T10-7. Decline: negation via "never" reachable through the
// participle's own subtree (not merely the auxiliary-adjacent slot) --
// isolates the subtree-scan half of the negation guard from the
// adverb-gap check, which a "never" sitting directly between the
// auxiliary and the participle would otherwise trip first.
#[test]
fn t10_declines_on_negation_reachable_through_the_participles_subtree() {
    let shapes = [
        tok(None, DepRelation::Root, "Ensure", "VB", "ensure"),
        tok(Some(5), DepRelation::Mark, "that", "IN", "that"),
        tok(Some(3), DepRelation::Det, "the", "DT", "the"),
        tok(Some(5), DepRelation::NsubjPass, "setting", "NN", "setting"),
        tok(Some(5), DepRelation::AuxPass, "is", "VBZ", "be"),
        tok(Some(0), DepRelation::Ccomp, "enabled", "VBN", "enable"),
        tok(Some(5), DepRelation::Other, "never", "RB", "never"),
        tok(Some(0), DepRelation::Punct, ".", ".", "."),
    ];
    let (source, tokens, parse) = build(&shapes);
    let found = t10_ensures_that_restructure(&source, &tokens, &parse);
    assert_single_decline(&found, "negation would invert the sentence's meaning");
}

// T10-8. Safety net: "is not enabled", the exact shape the design's own
// risk register names as the worst possible failure mode -- must never
// produce a Candidate, whichever named guard catches it.
#[test]
fn t10_never_produces_a_candidate_for_is_not_enabled() {
    let shapes = [
        tok(None, DepRelation::Root, "Ensure", "VB", "ensure"),
        tok(Some(6), DepRelation::Mark, "that", "IN", "that"),
        tok(Some(3), DepRelation::Det, "the", "DT", "the"),
        tok(Some(6), DepRelation::NsubjPass, "setting", "NN", "setting"),
        tok(Some(6), DepRelation::AuxPass, "is", "VBZ", "be"),
        tok(Some(6), DepRelation::Other, "not", "RB", "not"),
        tok(Some(0), DepRelation::Ccomp, "enabled", "VBN", "enable"),
        tok(Some(0), DepRelation::Punct, ".", ".", "."),
    ];
    let (source, tokens, parse) = build(&shapes);
    let found = t10_ensures_that_restructure(&source, &tokens, &parse);
    assert!(
        !found
            .iter()
            .any(|outcome| matches!(outcome, RestructureOutcome::Candidate { .. })),
        "found: {found:?}"
    );
}

// T10-9a. Accept (§A1): a finite subject-agreement shape with a bare
// demonstrative pronoun subject ("This ensures that ..."). The shipped
// parser attaches a subject-standing "This"/"That"/"These"/"Those" via a
// `Det` edge landing directly on the verb, not `Nsubj` -- see
// `t10_has_finite_subject`'s own docs; this is also the design brief's
// own first-named example and the dominant real-corpus shape. `ensures`
// (`VBZ`) re-inflects the derived verb via `third_sg`.
#[test]
fn t10_accepts_a_bare_demonstrative_finite_subject() {
    let shapes = [
        tok(Some(1), DepRelation::Det, "This", "DT", "this"),
        tok(None, DepRelation::Root, "ensures", "VBZ", "ensure"),
        tok(Some(6), DepRelation::Mark, "that", "IN", "that"),
        tok(Some(4), DepRelation::Det, "the", "DT", "the"),
        tok(Some(6), DepRelation::NsubjPass, "setting", "NN", "setting"),
        tok(Some(6), DepRelation::AuxPass, "is", "VBZ", "be"),
        tok(Some(1), DepRelation::Ccomp, "enabled", "VBN", "enable"),
        tok(Some(1), DepRelation::Punct, ".", ".", "."),
    ];
    let (source, tokens, parse) = build(&shapes);
    let found = t10_ensures_that_restructure(&source, &tokens, &parse);
    // The outer subject ("This") is never touched -- the range starts at
    // "ensures", not at "This".
    assert_single_candidate(
        &source,
        &found,
        "ensures that the setting is enabled",
        "enables the setting",
    );
}

// T10-9b. Accept (§A1): a genuine `Nsubj`-headed finite subject, plural,
// with the past-tense form ("These flags ensure that X is enabled") --
// the design brief's own second-named example. `ensure` (`VBP`)
// re-inflects to the base form, agreeing the same way for any
// non-3rd-person-singular subject regardless of its own number.
#[test]
fn t10_accepts_a_plural_nsubj_finite_subject() {
    let shapes = [
        tok(Some(2), DepRelation::Det, "These", "DT", "these"),
        tok(Some(2), DepRelation::Nsubj, "flags", "NNS", "flag"),
        tok(None, DepRelation::Root, "ensure", "VBP", "ensure"),
        tok(Some(7), DepRelation::Mark, "that", "IN", "that"),
        tok(Some(5), DepRelation::Det, "the", "DT", "the"),
        tok(Some(7), DepRelation::NsubjPass, "setting", "NN", "setting"),
        tok(Some(7), DepRelation::AuxPass, "is", "VBZ", "be"),
        tok(Some(2), DepRelation::Ccomp, "enabled", "VBN", "enable"),
        tok(Some(2), DepRelation::Punct, ".", ".", "."),
    ];
    let (source, tokens, parse) = build(&shapes);
    let found = t10_ensures_that_restructure(&source, &tokens, &parse);
    assert_single_candidate(
        &source,
        &found,
        "ensure that the setting is enabled",
        "enable the setting",
    );
}

// T10-9c. Accept (§A1): past tense ("ensured", `VBD`) re-inflects the
// derived verb to `past`, invariant across subject number -- exercised
// with a plural `Nsubj` subject ("they") to isolate the past-tense path
// from T10-9b's base-form one.
#[test]
fn t10_accepts_a_past_tense_finite_subject() {
    let shapes = [
        tok(Some(1), DepRelation::Nsubj, "they", "PRP", "they"),
        tok(None, DepRelation::Root, "ensured", "VBD", "ensure"),
        tok(Some(6), DepRelation::Mark, "that", "IN", "that"),
        tok(Some(4), DepRelation::Det, "the", "DT", "the"),
        tok(Some(6), DepRelation::NsubjPass, "data", "NNS", "datum"),
        tok(Some(6), DepRelation::AuxPass, "was", "VBD", "be"),
        tok(Some(1), DepRelation::Ccomp, "stored", "VBN", "store"),
        tok(Some(1), DepRelation::Punct, ".", ".", "."),
    ];
    let (source, tokens, parse) = build(&shapes);
    let found = t10_ensures_that_restructure(&source, &tokens, &parse);
    assert_single_candidate(
        &source,
        &found,
        "ensured that the data was stored",
        "stored the data",
    );
}

// T10-9d. Decline: a genuinely unsupported outer shape -- a modal-headed
// `ensure` ("You should ensure that ..."), neither imperative,
// infinitival, nor finite (no `Nsubj`/qualifying `Det` child of its own;
// the subject "You" belongs to "should", not to "ensure"). Replaces the
// old finite-subject decline case now that §A1 licenses it.
#[test]
fn t10_declines_on_a_modal_headed_ensure() {
    let shapes = [
        tok(Some(2), DepRelation::Nsubj, "You", "PRP", "you"),
        tok(Some(2), DepRelation::Aux, "should", "MD", "should"),
        tok(None, DepRelation::Root, "ensure", "VB", "ensure"),
        tok(Some(7), DepRelation::Mark, "that", "IN", "that"),
        tok(Some(5), DepRelation::Det, "the", "DT", "the"),
        tok(Some(7), DepRelation::NsubjPass, "setting", "NN", "setting"),
        tok(Some(7), DepRelation::AuxPass, "is", "VBZ", "be"),
        tok(Some(2), DepRelation::Ccomp, "enabled", "VBN", "enable"),
        tok(Some(2), DepRelation::Punct, ".", ".", "."),
    ];
    let (source, tokens, parse) = build(&shapes);
    let found = t10_ensures_that_restructure(&source, &tokens, &parse);
    assert_single_decline(
        &found,
        "no licensed outer mood shape (unsupported construction)",
    );
}

// T10-10. Silent: a modal complement never matches the construction at
// all -- no `Ccomp` child tagged `VBN`, so no `RestructureOutcome` of
// any kind, the same silent-decline convention `t4_activize_to_passive`
// uses for a verb that was never a licensable candidate.
#[test]
fn t10_is_silent_on_a_modal_complement() {
    let shapes = [
        tok(None, DepRelation::Root, "ensure", "VB", "ensure"),
        tok(Some(4), DepRelation::Mark, "that", "IN", "that"),
        tok(Some(4), DepRelation::Nsubj, "X", "NN", "x"),
        tok(Some(4), DepRelation::Aux, "can", "MD", "can"),
        tok(
            Some(0),
            DepRelation::Ccomp,
            "communicate",
            "VB",
            "communicate",
        ),
        tok(Some(4), DepRelation::Prep, "with", "IN", "with"),
        tok(Some(5), DepRelation::Pobj, "Y", "NNP", "y"),
    ];
    let (source, tokens, parse) = build(&shapes);
    let found = t10_ensures_that_restructure(&source, &tokens, &parse);
    assert!(found.is_empty(), "found: {found:?}");
}

// T10-11. Silent: an adjectival complement (no `AuxPass` child, `comp`
// tagged `VBZ` not `VBN`) never matches either.
#[test]
fn t10_is_silent_on_an_adjectival_complement() {
    let shapes = [
        tok(None, DepRelation::Root, "ensure", "VB", "ensure"),
        tok(Some(3), DepRelation::Mark, "that", "IN", "that"),
        tok(
            Some(3),
            DepRelation::Nsubj,
            "pair-programming",
            "NN",
            "pair-programming",
        ),
        tok(Some(0), DepRelation::Ccomp, "remains", "VBZ", "remain"),
        tok(
            Some(3),
            DepRelation::Other,
            "sustainable",
            "JJ",
            "sustainable",
        ),
    ];
    let (source, tokens, parse) = build(&shapes);
    let found = t10_ensures_that_restructure(&source, &tokens, &parse);
    assert!(found.is_empty(), "found: {found:?}");
}

// T10-12. Silent: an intransitive complement (`comp` tagged `VBZ`, no
// `AuxPass`/`NsubjPass` shape) never matches.
#[test]
fn t10_is_silent_on_an_intransitive_complement() {
    let shapes = [
        tok(None, DepRelation::Root, "ensure", "VB", "ensure"),
        tok(Some(4), DepRelation::Mark, "that", "IN", "that"),
        tok(Some(3), DepRelation::Det, "the", "DT", "the"),
        tok(Some(4), DepRelation::Nsubj, "process", "NN", "process"),
        tok(Some(0), DepRelation::Ccomp, "completes", "VBZ", "complete"),
        tok(
            Some(4),
            DepRelation::Other,
            "successfully",
            "RB",
            "successfully",
        ),
    ];
    let (source, tokens, parse) = build(&shapes);
    let found = t10_ensures_that_restructure(&source, &tokens, &parse);
    assert!(found.is_empty(), "found: {found:?}");
}

// T10-13. Decline: the promoted subject sits inside a bracketed aside --
// displayed material, not asserted, the same reasoning
// `object_is_licensable` applies on T4's side.
#[test]
fn t10_declines_on_a_bracketed_subject() {
    let shapes = [
        tok(None, DepRelation::Root, "Ensure", "VB", "ensure"),
        tok(Some(8), DepRelation::Mark, "that", "IN", "that"),
        tok(Some(8), DepRelation::Punct, "(", "-LRB-", "("),
        tok(Some(5), DepRelation::Det, "the", "DT", "the"),
        tok(Some(5), DepRelation::Other, "legacy", "JJ", "legacy"),
        tok(Some(8), DepRelation::NsubjPass, "flag", "NN", "flag"),
        tok(Some(8), DepRelation::Punct, ")", "-RRB-", ")"),
        tok(Some(8), DepRelation::AuxPass, "is", "VBZ", "be"),
        tok(Some(0), DepRelation::Ccomp, "enabled", "VBN", "enable"),
        tok(Some(0), DepRelation::Punct, ".", ".", "."),
    ];
    let (source, tokens, parse) = build(&shapes);
    let found = t10_ensures_that_restructure(&source, &tokens, &parse);
    assert_single_decline(
        &found,
        "the promoted noun phrase is not a licensable subject",
    );
}

// T10-14. Decline: the promoted subject has a post-modifying `prep` --
// moving the whole subtree would carry a modifier that may attach
// elsewhere, exactly `object_is_licensable`'s own reasoning on T4's
// side.
#[test]
fn t10_declines_on_a_subject_with_a_post_modifying_prep() {
    let shapes = [
        tok(None, DepRelation::Root, "Ensure", "VB", "ensure"),
        tok(Some(8), DepRelation::Mark, "that", "IN", "that"),
        tok(Some(3), DepRelation::Det, "the", "DT", "the"),
        tok(Some(8), DepRelation::NsubjPass, "setting", "NN", "setting"),
        tok(Some(3), DepRelation::Prep, "for", "IN", "for"),
        tok(Some(6), DepRelation::Other, "legacy", "JJ", "legacy"),
        tok(Some(4), DepRelation::Pobj, "devices", "NNS", "device"),
        tok(Some(8), DepRelation::AuxPass, "is", "VBZ", "be"),
        tok(Some(0), DepRelation::Ccomp, "enabled", "VBN", "enable"),
        tok(Some(0), DepRelation::Punct, ".", ".", "."),
    ];
    let (source, tokens, parse) = build(&shapes);
    let found = t10_ensures_that_restructure(&source, &tokens, &parse);
    assert_single_decline(
        &found,
        "the promoted noun phrase is not a licensable subject",
    );
}

// ---------------------------------------------------------------------
// §A2: the aux-headed past-passive complement shape, where the shipped
// parser's `Ccomp` lands on the auxiliary ("were") rather than the
// participle. Every fixture below was cross-checked against the real
// shipped tagger/parser's own output on the analogous real sentence
// (see `crates/friction-edit/tests/restructure_op.rs`'s own A2 fixtures
// for the end-to-end confirmation); these hand-built trees isolate the
// transducer logic the way this file's own module docs describe.
// ---------------------------------------------------------------------

// T10-16. Accept (§A2): the aux-headed shape with no adverb gap --
// `Ccomp` lands on "were" (`VBD`), the subject attaches to it as a plain
// `Nsubj`, and its one `VBN` child ("cleaned") is the true participle,
// immediately adjacent to "were". Infinitival outer shape.
#[test]
fn t10_accepts_the_aux_headed_shape_with_no_adverb_gap() {
    let shapes = [
        tok(Some(1), DepRelation::Aux, "to", "TO", "to"),
        tok(None, DepRelation::Root, "ensure", "VB", "ensure"),
        tok(Some(5), DepRelation::Mark, "that", "IN", "that"),
        tok(Some(4), DepRelation::Det, "the", "DT", "the"),
        tok(Some(5), DepRelation::Nsubj, "listeners", "NNS", "listener"),
        tok(Some(1), DepRelation::Ccomp, "were", "VBD", "were"),
        tok(Some(5), DepRelation::Other, "cleaned", "VBN", "clean"),
        tok(Some(1), DepRelation::Punct, ".", ".", "."),
    ];
    let (source, tokens, parse) = build(&shapes);
    let found = t10_ensures_that_restructure(&source, &tokens, &parse);
    assert_single_candidate(
        &source,
        &found,
        "ensure that the listeners were cleaned",
        "clean the listeners",
    );
}

// T10-17. Decline (§A2): the adverb gap between "were" and its `VBN`
// child -- verified against the shipped parser as the design brief's own
// worked example ("to ensure that event listeners were properly cleaned
// up"): the very adverb that causes the parser to head the clause at the
// auxiliary is also what this guard declines on, so this shape is
// expected to decline here far more often than it accepts.
#[test]
fn t10_declines_on_the_aux_headed_shapes_own_adverb_gap() {
    let shapes = [
        tok(Some(1), DepRelation::Aux, "to", "TO", "to"),
        tok(None, DepRelation::Root, "ensure", "VB", "ensure"),
        tok(Some(5), DepRelation::Mark, "that", "IN", "that"),
        tok(Some(4), DepRelation::Det, "the", "DT", "the"),
        tok(Some(5), DepRelation::Nsubj, "listeners", "NNS", "listener"),
        tok(Some(1), DepRelation::Ccomp, "were", "VBD", "were"),
        tok(Some(7), DepRelation::Other, "properly", "RB", "properly"),
        tok(Some(5), DepRelation::Other, "cleaned", "VBN", "clean"),
        tok(Some(1), DepRelation::Punct, ".", ".", "."),
    ];
    let (source, tokens, parse) = build(&shapes);
    let found = t10_ensures_that_restructure(&source, &tokens, &parse);
    assert_single_decline(&found, "adverb between the auxiliary and the participle");
}

// T10-18. Decline (§A2): negation reachable through the participle's own
// subtree ("never"), with the auxiliary and participle adjacent -- so
// this isolates the negation guard from the adverb-gap guard for the
// aux-headed shape, mirroring T10-7's isolation on the standard shape.
#[test]
fn t10_declines_on_negation_in_the_aux_headed_shapes_participle_subtree() {
    let shapes = [
        tok(Some(1), DepRelation::Aux, "to", "TO", "to"),
        tok(None, DepRelation::Root, "ensure", "VB", "ensure"),
        tok(Some(5), DepRelation::Mark, "that", "IN", "that"),
        tok(Some(4), DepRelation::Det, "the", "DT", "the"),
        tok(Some(5), DepRelation::Nsubj, "listeners", "NNS", "listener"),
        tok(Some(1), DepRelation::Ccomp, "were", "VBD", "were"),
        tok(Some(5), DepRelation::Other, "cleaned", "VBN", "clean"),
        tok(Some(6), DepRelation::Other, "never", "RB", "never"),
        tok(Some(1), DepRelation::Punct, ".", ".", "."),
    ];
    let (source, tokens, parse) = build(&shapes);
    let found = t10_ensures_that_restructure(&source, &tokens, &parse);
    assert_single_decline(&found, "negation would invert the sentence's meaning");
}

// T10-19. Decline (§A2): a fused negated aux surface ("weren't") playing
// the auxiliary role -- the single highest-stakes guard, exercised for
// this shape the same way T10-6 exercises it for the standard one.
#[test]
fn t10_declines_on_a_fused_negated_aux_in_the_aux_headed_shape() {
    let shapes = [
        tok(Some(1), DepRelation::Aux, "to", "TO", "to"),
        tok(None, DepRelation::Root, "ensure", "VB", "ensure"),
        tok(Some(4), DepRelation::Mark, "that", "IN", "that"),
        tok(Some(4), DepRelation::Nsubj, "listeners", "NNS", "listener"),
        tok(Some(1), DepRelation::Ccomp, "weren't", "VBP", "weren't"),
        tok(Some(4), DepRelation::Other, "cleaned", "VBN", "clean"),
        tok(Some(1), DepRelation::Punct, ".", ".", "."),
    ];
    let (source, tokens, parse) = build(&shapes);
    let found = t10_ensures_that_restructure(&source, &tokens, &parse);
    assert_single_decline(&found, "negation would invert the sentence's meaning");
}

// T10-20. Silent (§A2): an aux-headed candidate where the subject
// attaches as `NsubjPass` rather than the consistently-observed `Nsubj`
// -- the inconsistent sub-shape the design explicitly calls out declining
// rather than guessing at. `t10_match_passive_complement` only recognizes
// `Nsubj` for this shape, so this never matches at all: silent, the same
// convention every other never-matched shape in this module uses.
#[test]
fn t10_is_silent_on_an_aux_headed_candidate_with_an_inconsistent_subject_relation() {
    let shapes = [
        tok(Some(1), DepRelation::Aux, "to", "TO", "to"),
        tok(None, DepRelation::Root, "ensure", "VB", "ensure"),
        tok(Some(5), DepRelation::Mark, "that", "IN", "that"),
        tok(Some(4), DepRelation::Det, "the", "DT", "the"),
        tok(
            Some(5),
            DepRelation::NsubjPass,
            "listeners",
            "NNS",
            "listener",
        ),
        tok(Some(1), DepRelation::Ccomp, "were", "VBD", "were"),
        tok(Some(5), DepRelation::Other, "cleaned", "VBN", "clean"),
        tok(Some(1), DepRelation::Punct, ".", ".", "."),
    ];
    let (source, tokens, parse) = build(&shapes);
    let found = t10_ensures_that_restructure(&source, &tokens, &parse);
    assert!(found.is_empty(), "found: {found:?}");
}

// T10-15. `t10_ensures_that_restructure` is not folded into `candidates()`
// -- it is selected and applied by `crate::restructure`, not the
// register band machinery `candidates()` feeds (see this crate's own
// design notes on why R1 has no band).
#[test]
fn t10_is_not_folded_into_candidates() {
    let shapes = [
        tok(None, DepRelation::Root, "Ensure", "VB", "ensure"),
        tok(Some(5), DepRelation::Mark, "that", "IN", "that"),
        tok(Some(3), DepRelation::Det, "the", "DT", "the"),
        tok(Some(5), DepRelation::NsubjPass, "setting", "NN", "setting"),
        tok(Some(5), DepRelation::AuxPass, "is", "VBZ", "be"),
        tok(Some(0), DepRelation::Ccomp, "enabled", "VBN", "enable"),
        tok(Some(0), DepRelation::Punct, ".", ".", "."),
    ];
    let (source, tokens, parse) = build(&shapes);
    assert!(candidates(&source, &tokens, &parse).is_empty());
}

// ---------------------------------------------------------------------
// T11: dobj-gated transitive substitution (R2: `surface` -> `find`).
// ---------------------------------------------------------------------

/// Asserts `found` is exactly one [`SubstitutionCandidate`] whose range
/// slices `source` to `expected_span` and whose replacement is
/// `expected_replacement`.
fn assert_single_substitution(
    source: &str,
    found: &[SubstitutionCandidate],
    expected_span: &str,
    expected_replacement: &str,
) {
    assert_eq!(found.len(), 1, "found: {found:?}");
    assert_eq!(&source[found[0].range.clone()], expected_span);
    assert_eq!(&*found[0].replacement, expected_replacement);
}

// T11-1. Accept: `VBD` tag with a plain nominal `dobj`.
#[test]
fn t11_accepts_vbd_with_a_nominal_object() {
    let shapes = [
        tok(Some(1), DepRelation::Det, "This", "DT", "this"),
        tok(None, DepRelation::Root, "surfaced", "VBD", "surface"),
        tok(Some(3), DepRelation::Other, "two", "CD", "two"),
        tok(Some(1), DepRelation::Dobj, "tests", "NNS", "test"),
        tok(Some(1), DepRelation::Punct, ".", ".", "."),
    ];
    let (source, tokens, parse) = build(&shapes);
    let found = t11_transitive_substitution(&source, &tokens, &parse);
    assert_single_substitution(&source, &found, "surfaced", "found");
}

// T11-2. Accept: `VBZ` tag.
#[test]
fn t11_accepts_vbz_with_a_nominal_object() {
    let shapes = [
        tok(Some(2), DepRelation::Det, "This", "DT", "this"),
        tok(Some(2), DepRelation::Nsubj, "tool", "NN", "tool"),
        tok(None, DepRelation::Root, "surfaces", "VBZ", "surface"),
        tok(Some(4), DepRelation::Amod, "hidden", "JJ", "hidden"),
        tok(
            Some(2),
            DepRelation::Dobj,
            "dependencies",
            "NNS",
            "dependency",
        ),
        tok(Some(2), DepRelation::Punct, ".", ".", "."),
    ];
    let (source, tokens, parse) = build(&shapes);
    let found = t11_transitive_substitution(&source, &tokens, &parse);
    assert_single_substitution(&source, &found, "surfaces", "finds");
}

// T11-3. Accept: `VBG` tag.
#[test]
fn t11_accepts_vbg_with_a_nominal_object() {
    let shapes = [
        tok(None, DepRelation::Root, "surfacing", "VBG", "surface"),
        tok(Some(2), DepRelation::Amod, "hidden", "JJ", "hidden"),
        tok(
            Some(0),
            DepRelation::Dobj,
            "dependencies",
            "NNS",
            "dependency",
        ),
    ];
    let (source, tokens, parse) = build(&shapes);
    let found = t11_transitive_substitution(&source, &tokens, &parse);
    assert_single_substitution(&source, &found, "surfacing", "finding");
}

// T11-4. Accept: `VBN` tag -- rare (a participle rarely takes a direct
// object), but one of the five corpus-attested tags `vsub.surface::*`
// already establishes as worth covering.
#[test]
fn t11_accepts_vbn_with_a_nominal_object() {
    let shapes = [
        tok(None, DepRelation::Root, "surfaced", "VBN", "surface"),
        tok(Some(0), DepRelation::Dobj, "concerns", "NNS", "concern"),
    ];
    let (source, tokens, parse) = build(&shapes);
    let found = t11_transitive_substitution(&source, &tokens, &parse);
    assert_single_substitution(&source, &found, "surfaced", "found");
}

// T11-5. Accept: bare `VB` tag.
#[test]
fn t11_accepts_vb_with_a_nominal_object() {
    let shapes = [
        tok(None, DepRelation::Root, "surface", "VB", "surface"),
        tok(Some(0), DepRelation::Dobj, "concerns", "NNS", "concern"),
    ];
    let (source, tokens, parse) = build(&shapes);
    let found = t11_transitive_substitution(&source, &tokens, &parse);
    assert_single_substitution(&source, &found, "surface", "find");
}

// T11-6. Decline: no `Dobj` child at all ("surfaced as a significant
// contributor") -- the unmodified `vsub.surface::*` report-only frame
// rows already account for this decline; this module adds nothing.
#[test]
fn t11_declines_with_no_object_at_all() {
    let shapes = [
        tok(None, DepRelation::Root, "surfaced", "VBD", "surface"),
        tok(Some(0), DepRelation::Prep, "as", "IN", "as"),
        tok(Some(3), DepRelation::Det, "a", "DT", "a"),
        tok(
            Some(4),
            DepRelation::Amod,
            "significant",
            "JJ",
            "significant",
        ),
        tok(
            Some(1),
            DepRelation::Pobj,
            "contributor",
            "NN",
            "contributor",
        ),
    ];
    let (source, tokens, parse) = build(&shapes);
    let found = t11_transitive_substitution(&source, &tokens, &parse);
    assert!(found.is_empty(), "found: {found:?}");
}

// T11-7. Decline: an `IN` token between the verb and a `Dobj`-labeled
// object -- the counter-hardening guard against exactly the parser
// mis-attachment that could turn "surfaced **as** a significant
// contributor" into a false `dobj` on "contributor"; this fixture forces
// that mislabeling by hand to prove the guard catches it even when the
// parser's own relation label is wrong.
#[test]
fn t11_declines_when_a_preposition_sits_between_the_verb_and_a_mislabeled_object() {
    let shapes = [
        tok(None, DepRelation::Root, "surfaced", "VBD", "surface"),
        tok(Some(0), DepRelation::Other, "as", "IN", "as"),
        tok(Some(3), DepRelation::Det, "a", "DT", "a"),
        tok(
            Some(0),
            DepRelation::Dobj,
            "contributor",
            "NN",
            "contributor",
        ),
    ];
    let (source, tokens, parse) = build(&shapes);
    let found = t11_transitive_substitution(&source, &tokens, &parse);
    assert!(found.is_empty(), "found: {found:?}");
}

// T11-8. Decline: an adverbial object ("never surfaced anywhere") -- no
// `Dobj` child (the adverb attaches by a non-`Dobj` relation), so this is
// the same no-object silent decline as T11-6, exercised on the brief's
// own worked non-example.
#[test]
fn t11_declines_on_an_adverbial_object() {
    let shapes = [
        tok(Some(1), DepRelation::Other, "never", "RB", "never"),
        tok(None, DepRelation::Root, "surfaced", "VBD", "surface"),
        tok(Some(1), DepRelation::Other, "anywhere", "RB", "anywhere"),
    ];
    let (source, tokens, parse) = build(&shapes);
    let found = t11_transitive_substitution(&source, &tokens, &parse);
    assert!(found.is_empty(), "found: {found:?}");
}

// T11-9. Decline: an implausible object part of speech (`JJ`) standing
// in as a `Dobj` -- `is_plausible_object_pos`'s own guard.
#[test]
fn t11_declines_on_an_implausible_object_pos() {
    let shapes = [
        tok(None, DepRelation::Root, "surfaced", "VBD", "surface"),
        tok(Some(0), DepRelation::Dobj, "obvious", "JJ", "obvious"),
    ];
    let (source, tokens, parse) = build(&shapes);
    let found = t11_transitive_substitution(&source, &tokens, &parse);
    assert!(found.is_empty(), "found: {found:?}");
}

// T11-10. Decline: a bare pronoun object ("We surfaced it."). Recorded
// decision: T11 does not import T4's separate reflexive/personal-pronoun
// guards -- `is_plausible_object_pos` already permits `PRP`, but
// `has_noun` (borrowed unchanged, the same corroboration guard for both
// transducers) never does, since a bare pronoun carries no `NN`/`NNS`/
// `NNP`/`NNPS` tag in its own one-token subtree. Every pronoun object,
// reflexive or not, therefore declines by construction; no pronoun check
// was needed or added.
#[test]
fn t11_declines_on_a_bare_pronoun_object() {
    let shapes = [
        tok(Some(1), DepRelation::Nsubj, "We", "PRP", "we"),
        tok(None, DepRelation::Root, "surfaced", "VBD", "surface"),
        tok(Some(1), DepRelation::Dobj, "it", "PRP", "it"),
        tok(Some(1), DepRelation::Punct, ".", ".", "."),
    ];
    let (source, tokens, parse) = build(&shapes);
    let found = t11_transitive_substitution(&source, &tokens, &parse);
    assert!(found.is_empty(), "found: {found:?}");
}

// T11-11. Decline: the object's subtree attaches leftward of the verb --
// a parse this transducer doesn't trust, the same defensive ordering
// convention `t10_match_passive_complement` applies to its own
// attachments.
#[test]
fn t11_declines_when_the_object_attaches_leftward_of_the_verb() {
    let shapes = [
        tok(Some(1), DepRelation::Dobj, "concerns", "NNS", "concern"),
        tok(None, DepRelation::Root, "surfaced", "VBD", "surface"),
    ];
    let (source, tokens, parse) = build(&shapes);
    let found = t11_transitive_substitution(&source, &tokens, &parse);
    assert!(found.is_empty(), "found: {found:?}");
}

// T11-12. A lemma outside `LEXICON_EN.transitive_verbs` never matches,
// whatever its tag or object shape -- confirms the loop is gated on the
// table, not a hardcoded `"surface"` string comparison.
#[test]
fn t11_declines_on_a_lemma_outside_the_transitive_verbs_table() {
    let shapes = [
        tok(None, DepRelation::Root, "deployed", "VBD", "deploy"),
        tok(Some(0), DepRelation::Dobj, "changes", "NNS", "change"),
    ];
    let (source, tokens, parse) = build(&shapes);
    let found = t11_transitive_substitution(&source, &tokens, &parse);
    assert!(found.is_empty(), "found: {found:?}");
}

// T11-13. `t11_transitive_substitution` is not folded into `candidates()`
// either -- same reasoning as T10-15: selected and applied by
// `crate::restructure`, not the register band machinery.
#[test]
fn t11_is_not_folded_into_candidates() {
    let shapes = [
        tok(None, DepRelation::Root, "surfaced", "VBD", "surface"),
        tok(Some(0), DepRelation::Dobj, "concerns", "NNS", "concern"),
    ];
    let (source, tokens, parse) = build(&shapes);
    assert!(candidates(&source, &tokens, &parse).is_empty());
}
