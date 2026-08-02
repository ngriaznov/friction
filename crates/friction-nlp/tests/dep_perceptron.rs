//! Integration tests for [`PerceptronParser`], the trained arc-eager
//! [`DepParser`] implementation.
//!
//! These tests exercise the shipped `weights/parser_en.json.gz` artifact
//! through the real [`Tagger`] + [`DepParser`] pipeline (never a synthetic
//! shortcut): the same path any real caller in this workspace uses.

use friction_nlp::{DepParser, PerceptronParser, PerceptronTagger, Tagger};

fn parser() -> &'static PerceptronParser {
    use std::sync::OnceLock;
    static PARSER: OnceLock<PerceptronParser> = OnceLock::new();
    PARSER.get_or_init(|| PerceptronParser::new().expect("embedded weights must load"))
}

fn tagger() -> &'static PerceptronTagger {
    use std::sync::OnceLock;
    static TAGGER: OnceLock<PerceptronTagger> = OnceLock::new();
    TAGGER.get_or_init(|| PerceptronTagger::new().expect("embedded weights must load"))
}

/// The shipped, embedded artifact decompresses and parses as valid JSON:
/// covers the gunzip/deserialize round trip independent of how accurate
/// the resulting model is.
#[test]
fn embedded_artifact_loads() {
    PerceptronParser::new().expect("embedded weights must load");
}

/// Every parse produces exactly one [`friction_nlp::DepEdge`] per input
/// token, and exactly one of those tokens is the sentence root (`head:
/// None`) — the structural invariant [`friction_nlp::dep_arceager::Configuration::finish`]
/// guarantees by construction, checked here end to end through the real
/// trained model.
#[test]
fn parse_produces_one_edge_per_token_and_exactly_one_root() {
    let text = "The team shipped the update to every customer last week.";
    let tokens = tagger().tag(text, 0);
    let parse = parser().parse(text, &tokens).expect("parse must succeed");

    assert_eq!(parse.edges().len(), tokens.len());
    let root_count = parse
        .edges()
        .iter()
        .filter(|edge| edge.head.is_none())
        .count();
    assert_eq!(
        root_count, 1,
        "exactly one token must be the sentence root, got {root_count}"
    );
}

/// Parsing the same sentence repeatedly, including across fresh
/// [`PerceptronParser`] instances, produces byte-for-byte (structurally)
/// identical output every time, no RNG anywhere in the inference path.
#[test]
fn parsing_is_deterministic_across_repeated_calls_and_fresh_instances() {
    let text = "Engineers reviewed the code carefully before the release shipped.";
    let tokens = tagger().tag(text, 0);

    let first = parser().parse(text, &tokens).expect("parse must succeed");
    let second = parser().parse(text, &tokens).expect("parse must succeed");
    assert_eq!(first, second);

    let fresh = PerceptronParser::new().expect("embedded weights must load");
    let third = fresh.parse(text, &tokens).expect("parse must succeed");
    assert_eq!(first, third);
}

/// An empty token slice never panics and produces an empty (trivially
/// valid) parse — `DepParser::parse`'s contract requires this even though
/// no real document ever tags to zero tokens.
#[test]
fn parse_accepts_empty_input() {
    let parse = parser().parse("", &[]).expect("empty input must parse");
    assert!(parse.edges().is_empty());
}

/// A short, hand-picked sentence whose exact head/relation structure was
/// verified by hand against the shipped model's actual output — a smoke
/// test of the `Tagger` -> `PerceptronParser` wiring (tokenization, tag
/// lookup, greedy decode, edge construction), *not* a claim that this one
/// sentence is representative of overall accuracy. See `weights/NOTICE.md`
/// for the honest, corpus-wide UAS/LAS numbers.
#[test]
fn parses_a_verified_simple_sentence_correctly() {
    let text = "The team shipped the update.";
    let tokens = tagger().tag(text, 0);
    let parse = parser().parse(text, &tokens).expect("parse must succeed");
    let edges = parse.edges();
    assert_eq!(
        edges.len(),
        6,
        "\"The team shipped the update.\" is 6 tokens"
    );

    let word = |i: usize| &text[tokens[i].token.range.clone()];
    assert_eq!(word(0), "The");
    assert_eq!(word(1), "team");
    assert_eq!(word(2), "shipped");
    assert_eq!(word(3), "the");
    assert_eq!(word(4), "update");
    assert_eq!(word(5), ".");

    // shipped(2) is the sentence root.
    assert_eq!(edges[2].head, None);
    assert_eq!(edges[2].relation.as_str(), "root");
    // The(0) --det--> team(1)
    assert_eq!(edges[0].head, Some(1));
    assert_eq!(edges[0].relation.as_str(), "det");
    // team(1) --nsubj--> shipped(2)
    assert_eq!(edges[1].head, Some(2));
    assert_eq!(edges[1].relation.as_str(), "nsubj");
    // the(3) --det--> update(4)
    assert_eq!(edges[3].head, Some(4));
    assert_eq!(edges[3].relation.as_str(), "det");
    // update(4) --dobj--> shipped(2)
    assert_eq!(edges[4].head, Some(2));
    assert_eq!(edges[4].relation.as_str(), "dobj");
    // .(5) --punct--> shipped(2)
    assert_eq!(edges[5].head, Some(2));
    assert_eq!(edges[5].relation.as_str(), "punct");
}
