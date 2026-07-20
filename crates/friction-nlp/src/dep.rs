//! Dependency parsing: the [`DepParser`] trait and the relation vocabulary
//! its implementations produce.
//!
//! # Input contract
//!
//! [`DepParser::parse`] consumes exactly what [`crate::Tagger::tag`]
//! produces for one sentence: the sentence's original source text (so an
//! implementation can slice out exact surface characters when a POS tag
//! alone is ambiguous — e.g. telling a comma from a semicolon, or "and"
//! from "but") plus that sentence's slice of [`crate::TaggedToken`]s.
//! Nothing here re-tags or re-tokenizes; a caller is responsible for
//! partitioning a document's tagged tokens into per-sentence slices before
//! calling [`DepParser::parse`].
//!
//! # Confidence contract
//!
//! Every [`DepEdge`] carries a [`Confidence`]: an estimate, in `[0.0,
//! 1.0]`, of how much the parser's chosen head/relation for that token beat
//! its next-best alternative. `1.0` means the parser considers the
//! decision essentially unambiguous; values near `0.0` mean it found
//! multiple similarly plausible analyses (for a pattern-based
//! implementation, a weak or partial pattern match; for a model-backed
//! implementation, a narrow top-2 softmax margin — see
//! [`crate::dep_onnx::softmax_top2_margin`]). Callers that turn a
//! `DepParser` finding into a `Patch` or `Finding` should treat a
//! low-confidence edge as grounds to demote that finding from `Fix` to
//! `Suggest` tier rather than trusting it outright — this crate does not
//! hardcode the threshold, since how conservative to be is a rule's
//! decision, not a parser's.

use friction_core::span;

use crate::tag::TaggedToken;

/// The dependency relation a [`DepEdge`] assigns its token, relative to its
/// head.
///
/// This is deliberately the minimal vocabulary the rules in this
/// workspace's plan need: nominal subjects, direct objects, list/triad
/// coordination, and sentence-final participial modifiers. Every other
/// relation — including the sentence's syntactic root, which by
/// construction has no head to relate to — is [`Self::Other`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DepRelation {
    /// A nominal subject of its head.
    Subject,
    /// A direct object of its head.
    Object,
    /// A conjunct coordinated with its head in a list (`X, Y, and Z`): the
    /// head is the list's first conjunct.
    Coordination,
    /// A participial phrase modifying its head, typically attached at the
    /// end of a sentence (`..., raising concerns about scalability.`).
    ParticipialModifier,
    /// Any relation not covered by another variant, including the
    /// sentence's root token (which carries `head: None`).
    Other,
}

/// A confidence margin in `[0.0, 1.0]` for a single dependency decision.
/// See the module docs for the contract this value carries.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Confidence(f32);

impl Confidence {
    /// The parser considers this decision unambiguous.
    pub const CERTAIN: Self = Self(1.0);
    /// The parser found no meaningful signal for this decision; treat it as
    /// a placeholder, not a claim.
    pub const LOW: Self = Self(0.0);

    /// Creates a confidence value, clamping `value` into `[0.0, 1.0]`.
    #[must_use]
    pub const fn new(value: f32) -> Self {
        Self(value.clamp(0.0, 1.0))
    }

    /// The underlying margin value, in `[0.0, 1.0]`.
    #[must_use]
    pub const fn value(self) -> f32 {
        self.0
    }
}

/// One token's dependency edge: its relation to a governing head token
/// within the same sentence, or to no head at all if it is the sentence's
/// root.
#[derive(Debug, Clone, PartialEq)]
pub struct DepEdge {
    /// Index of the dependent token within its sentence's token slice.
    pub token: usize,
    /// Index of the governing head token, or `None` if `token` is the
    /// sentence's syntactic root.
    pub head: Option<usize>,
    /// The dependency relation `token` bears to `head`.
    pub relation: DepRelation,
    /// This decision's confidence margin.
    pub confidence: Confidence,
}

/// The dependency structure [`DepParser::parse`] produces for one sentence:
/// exactly one [`DepEdge`] per input token, at the same index.
#[derive(Debug, Clone, PartialEq)]
pub struct SentenceParse {
    edges: Vec<DepEdge>,
}

impl SentenceParse {
    /// Builds a parse from its edges, validating internal consistency:
    /// `edges[i].token == i` for every `i`, and every `head` is a valid,
    /// non-self index into `edges`.
    ///
    /// # Errors
    /// Returns the first [`DepParseError`] found.
    pub fn new(edges: Vec<DepEdge>) -> Result<Self, DepParseError> {
        for (index, edge) in edges.iter().enumerate() {
            if edge.token != index {
                return Err(DepParseError::EdgeOutOfOrder {
                    index,
                    token: edge.token,
                });
            }
            if let Some(head) = edge.head {
                if head >= edges.len() {
                    return Err(DepParseError::HeadOutOfBounds {
                        head,
                        len: edges.len(),
                    });
                }
                if head == edge.token {
                    return Err(DepParseError::SelfHead { token: edge.token });
                }
            }
        }
        Ok(Self { edges })
    }

    /// This parse's edges, one per sentence token, in token order.
    #[must_use]
    pub fn edges(&self) -> &[DepEdge] {
        &self.edges
    }

    /// The edge for the token at `index`, if any.
    #[must_use]
    pub fn edge(&self, index: usize) -> Option<&DepEdge> {
        self.edges.get(index)
    }

    /// The sentence's root edge (the token with no head), if the parse is
    /// non-empty.
    #[must_use]
    pub fn root(&self) -> Option<&DepEdge> {
        self.edges.iter().find(|edge| edge.head.is_none())
    }

    /// Edges bearing exactly `relation`, in token order.
    pub fn by_relation(&self, relation: DepRelation) -> impl Iterator<Item = &DepEdge> {
        self.edges
            .iter()
            .filter(move |edge| edge.relation == relation)
    }
}

/// Errors produced while building or using a dependency parse.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum DepParseError {
    /// [`SentenceParse::new`] was given edges out of token order.
    #[error("edge at position {index} names token {token}, expected {index}")]
    EdgeOutOfOrder {
        /// Position of the offending edge in the input.
        index: usize,
        /// The token index it actually named.
        token: usize,
    },

    /// An edge's head index does not address a token in the same sentence.
    #[error("dependency head index {head} is out of bounds for a sentence of {len} tokens")]
    HeadOutOfBounds {
        /// The offending head index.
        head: usize,
        /// Number of tokens in the sentence.
        len: usize,
    },

    /// An edge names itself as its own head.
    #[error("token {token} cannot be its own dependency head")]
    SelfHead {
        /// The offending token index.
        token: usize,
    },

    /// A model-backed [`DepParser`] has no usable model loaded.
    #[error("dependency-parser model is not available: {reason}")]
    ModelNotAvailable {
        /// Plain-language explanation of why no model is available.
        reason: Box<str>,
    },
}

/// Produces dependency structure — head indices and dependency relations —
/// for a part-of-speech-tagged sentence.
///
/// See the module docs for the input contract and the confidence contract
/// every [`DepEdge`] carries.
pub trait DepParser {
    /// Parses one sentence's dependency structure.
    ///
    /// `source` is the original document text `tokens`' spans address;
    /// `tokens` is that sentence's tagged tokens, in source order.
    ///
    /// # Errors
    /// Returns [`DepParseError`] if the parse cannot be produced (a
    /// model-backed implementation with no usable model, or an internal
    /// consistency failure) — implementations must not panic on any input,
    /// including a malformed or empty token slice.
    fn parse(&self, source: &str, tokens: &[TaggedToken]) -> Result<SentenceParse, DepParseError>;
}

/// The comparison key [`same_subject`] uses for a sentence's subject: its
/// tagger-assigned lemma (already lowercase by [`crate::TaggedToken`]'s own
/// contract), if [`SentenceParse`] found a subject at all.
fn subject_lemma<'t>(tokens: &'t [TaggedToken], parse: &SentenceParse) -> Option<&'t str> {
    let edge = parse.by_relation(DepRelation::Subject).next()?;
    tokens.get(edge.token).map(|token| &*token.lemma)
}

/// The exact surface text of `tokens`' syntactic subject in `source`, if
/// `parse` found one.
#[must_use]
pub fn subject_text<'s>(
    source: &'s str,
    tokens: &[TaggedToken],
    parse: &SentenceParse,
) -> Option<&'s str> {
    let edge = parse.by_relation(DepRelation::Subject).next()?;
    let token = tokens.get(edge.token)?;
    span::slice(source, &token.token.range).ok()
}

/// Approximates whether two adjacent sentences share the same grammatical
/// subject, by comparing each sentence's detected subject token's lemma.
///
/// Returns `false` — not an error — if either sentence has no detected
/// subject; a missing subject is not evidence of a match.
#[must_use]
pub fn same_subject(
    a: (&[TaggedToken], &SentenceParse),
    b: (&[TaggedToken], &SentenceParse),
) -> bool {
    match (subject_lemma(a.0, a.1), subject_lemma(b.0, b.1)) {
        (Some(x), Some(y)) => x == y,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use friction_core::{Token, TokenKind};

    use super::*;
    use crate::tag::PosTag;

    fn tok(range: std::ops::Range<usize>, pos: &str, lemma: &str) -> TaggedToken {
        TaggedToken {
            token: Token::new(range, TokenKind::Word),
            pos: PosTag::new(pos),
            lemma: lemma.into(),
        }
    }

    /// A well-formed, densely-indexed edge list is accepted.
    #[test]
    fn sentence_parse_new_accepts_well_formed_edges() {
        let edges = vec![
            DepEdge {
                token: 0,
                head: Some(1),
                relation: DepRelation::Subject,
                confidence: Confidence::CERTAIN,
            },
            DepEdge {
                token: 1,
                head: None,
                relation: DepRelation::Other,
                confidence: Confidence::CERTAIN,
            },
        ];
        let parse = SentenceParse::new(edges).unwrap();
        assert_eq!(parse.edges().len(), 2);
        assert_eq!(parse.root().unwrap().token, 1);
    }

    /// An edge whose `token` does not match its position is rejected.
    #[test]
    fn sentence_parse_new_rejects_out_of_order_edges() {
        let edges = vec![DepEdge {
            token: 3,
            head: None,
            relation: DepRelation::Other,
            confidence: Confidence::CERTAIN,
        }];
        let err = SentenceParse::new(edges).unwrap_err();
        assert!(matches!(err, DepParseError::EdgeOutOfOrder { .. }));
    }

    /// A head index past the end of the sentence is rejected.
    #[test]
    fn sentence_parse_new_rejects_out_of_bounds_head() {
        let edges = vec![DepEdge {
            token: 0,
            head: Some(5),
            relation: DepRelation::Other,
            confidence: Confidence::CERTAIN,
        }];
        let err = SentenceParse::new(edges).unwrap_err();
        assert!(matches!(err, DepParseError::HeadOutOfBounds { .. }));
    }

    /// A token that names itself as its own head is rejected.
    #[test]
    fn sentence_parse_new_rejects_self_head() {
        let edges = vec![DepEdge {
            token: 0,
            head: Some(0),
            relation: DepRelation::Other,
            confidence: Confidence::CERTAIN,
        }];
        let err = SentenceParse::new(edges).unwrap_err();
        assert!(matches!(err, DepParseError::SelfHead { .. }));
    }

    /// `Confidence::new` clamps out-of-range input into `[0.0, 1.0]`.
    #[test]
    fn confidence_clamps_to_unit_range() {
        assert!((Confidence::new(-1.0).value() - 0.0).abs() < f32::EPSILON);
        assert!((Confidence::new(2.0).value() - 1.0).abs() < f32::EPSILON);
        assert!((Confidence::new(0.5).value() - 0.5).abs() < f32::EPSILON);
    }

    /// `same_subject` matches two sentences whose detected subjects share a
    /// lemma, even with different surface forms.
    #[test]
    fn same_subject_matches_via_lemma() {
        let tokens_a = vec![tok(0..4, "PRP", "they")];
        let parse_a = SentenceParse::new(vec![DepEdge {
            token: 0,
            head: None,
            relation: DepRelation::Subject,
            confidence: Confidence::CERTAIN,
        }])
        .unwrap();

        let tokens_b = vec![tok(0..4, "NN", "they")];
        let parse_b = SentenceParse::new(vec![DepEdge {
            token: 0,
            head: None,
            relation: DepRelation::Subject,
            confidence: Confidence::CERTAIN,
        }])
        .unwrap();

        assert!(same_subject((&tokens_a, &parse_a), (&tokens_b, &parse_b)));
    }

    /// `same_subject` returns `false`, not a panic, when one sentence has
    /// no detected subject.
    #[test]
    fn same_subject_false_when_subject_missing() {
        let tokens_a = vec![tok(0..4, "PRP", "they")];
        let parse_a = SentenceParse::new(vec![DepEdge {
            token: 0,
            head: None,
            relation: DepRelation::Subject,
            confidence: Confidence::CERTAIN,
        }])
        .unwrap();

        let tokens_b = vec![tok(0..3, "VBD", "run")];
        let parse_b = SentenceParse::new(vec![DepEdge {
            token: 0,
            head: None,
            relation: DepRelation::Other,
            confidence: Confidence::CERTAIN,
        }])
        .unwrap();

        assert!(!same_subject((&tokens_a, &parse_a), (&tokens_b, &parse_b)));
    }

    /// `subject_text` returns the exact source slice of the detected
    /// subject token.
    #[test]
    fn subject_text_returns_surface_form() {
        let source = "We shipped";
        let tokens = vec![tok(0..2, "PRP", "we"), tok(3..10, "VBD", "ship")];
        let parse = SentenceParse::new(vec![
            DepEdge {
                token: 0,
                head: Some(1),
                relation: DepRelation::Subject,
                confidence: Confidence::CERTAIN,
            },
            DepEdge {
                token: 1,
                head: None,
                relation: DepRelation::Other,
                confidence: Confidence::CERTAIN,
            },
        ])
        .unwrap();
        assert_eq!(subject_text(source, &tokens, &parse), Some("We"));
    }
}
