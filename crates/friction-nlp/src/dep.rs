//! Dependency parsing: the [`DepParser`] trait and the relation vocabulary
//! its implementations produce.
//!
//! # Input contract
//!
//! [`DepParser::parse`] consumes exactly what [`crate::Tagger::tag`]
//! produces for one sentence: the source text (so an implementation can
//! slice exact characters when a POS tag alone is ambiguous, e.g. a comma
//! vs. a semicolon) plus that sentence's slice of [`crate::TaggedToken`]s.
//! Nothing here re-tags or re-tokenizes; a caller partitions a document's
//! tagged tokens into per-sentence slices first.
//!
//! # Confidence contract
//!
//! Every [`DepEdge`] carries a [`Confidence`] in `[0.0, 1.0]`: how much the
//! chosen head/relation beat its next-best alternative. `1.0` means
//! essentially unambiguous; values near `0.0` mean multiple similarly
//! plausible analyses (a weak pattern match, or a narrow top-2 margin for
//! a model-backed implementation). Callers turning a `DepParser` finding
//! into a `Patch` or `Finding` should treat low confidence as grounds to
//! demote from `Fix` to `Suggest` rather than trust it outright — this
//! crate doesn't hardcode the threshold, since how conservative to be is
//! a rule's decision, not a parser's.

use std::fmt;
use std::str::FromStr;

use crate::tag::TaggedToken;

/// The dependency relation a [`DepEdge`] assigns its token, relative to its
/// head.
///
/// These are ClearNLP/OntoNotes relation names, not Universal Dependencies
/// v2, deliberate. UD spells several differently (`obj` for
/// [`Self::Dobj`], `obl` for what this folds into
/// [`Self::Prep`]/[`Self::Pobj`], `case` for a plain adposition,
/// `aux:pass` for [`Self::AuxPass`], `nsubj:pass` for
/// [`Self::NsubjPass`]), but every [`SentenceParse`] consumer here matches
/// `ClearNLP` spellings: swapping in UD names wouldn't fail loudly, it
/// would silently match nothing.
///
/// [`Self::Root`] isn't itself an arc label: a root's [`DepEdge`] carries
/// `head: None`, so there's no head to describe. It exists purely so a
/// relation is always nameable — but the arc-eager action space
/// ([`crate::dep_arceager::Transition::LeftArc`] /
/// [`crate::dep_arceager::Transition::RightArc`]) never labels one
/// `Root`.
///
/// [`Self::Other`] is the collapse target for any relation this crate
/// doesn't model as its own variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DepRelation {
    /// The sentence's syntactic root. Not an arc label: see the enum
    /// docs.
    Root,
    /// A clausal modifier of a noun (a relative or reduced-relative
    /// clause): "the file **that changed**".
    Acl,
    /// An adverbial clause modifier: "shipped **because tests passed**".
    Advcl,
    /// The demoted logical subject of a passive verb, introduced by "by":
    /// "reviewed **by the team**".
    Agent,
    /// An adjectival modifier of a noun: "the **quick** fox".
    Amod,
    /// A non-passive auxiliary or modal verb: "**will** ship", "**has**
    /// shipped".
    Aux,
    /// The passive auxiliary "be": "**was** shipped".
    AuxPass,
    /// A coordinating conjunction: "screws, bolts, **and** washers".
    Cc,
    /// A clausal complement with its own subject: "said **[the build
    /// failed]**".
    Ccomp,
    /// A conjunct coordinated with its head in a flat list (`X, Y, and Z`):
    /// the head is the list's first conjunct.
    Conj,
    /// A clausal subject: "**[what she said]** surprised everyone".
    Csubj,
    /// A determiner: "**the** kit".
    Det,
    /// A direct object of its head verb.
    Dobj,
    /// A subordinating marker introducing a finite clause: "**because**
    /// tests passed", "**if** it fails".
    Mark,
    /// A nominal subject of its (active-voice) head verb.
    Nsubj,
    /// A nominal subject of its passive-voice head verb.
    NsubjPass,
    /// The object of a preposition: "about **scalability**".
    Pobj,
    /// A prepositional modifier: "raising concerns **about** scalability".
    Prep,
    /// An open clausal complement with no subject of its own, controlled
    /// by its head: "wants **[to leave]**".
    Xcomp,
    /// A punctuation mark attached as a leaf.
    Punct,
    /// Any relation not covered by another variant.
    Other,
}

impl DepRelation {
    /// This relation's lowercase canonical name — the same spelling
    /// [`DepRelation::from_str`] parses back, so a gold file's text labels
    /// round-trip through this type exactly.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Root => "root",
            Self::Acl => "acl",
            Self::Advcl => "advcl",
            Self::Agent => "agent",
            Self::Amod => "amod",
            Self::Aux => "aux",
            Self::AuxPass => "auxpass",
            Self::Cc => "cc",
            Self::Ccomp => "ccomp",
            Self::Conj => "conj",
            Self::Csubj => "csubj",
            Self::Det => "det",
            Self::Dobj => "dobj",
            Self::Mark => "mark",
            Self::Nsubj => "nsubj",
            Self::NsubjPass => "nsubjpass",
            Self::Pobj => "pobj",
            Self::Prep => "prep",
            Self::Xcomp => "xcomp",
            Self::Punct => "punct",
            Self::Other => "other",
        }
    }
}

impl fmt::Display for DepRelation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A relation name outside [`DepRelation`]'s vocabulary was given to
/// [`DepRelation::from_str`].
#[derive(Debug, thiserror::Error)]
#[error("unrecognized dependency relation name: {0:?}")]
pub struct ParseDepRelationError(Box<str>);

impl FromStr for DepRelation {
    type Err = ParseDepRelationError;

    /// Parses a relation's lowercase canonical name, the exact inverse of
    /// [`DepRelation::as_str`]. Case- and space-sensitive by design: gold
    /// file labels are expected already normalized, and silently accepting
    /// near misses (`"Nsubj"`, `"nsubj_pass"`) would hide a gold-file bug
    /// this type exists to catch at load time.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "root" => Self::Root,
            "acl" => Self::Acl,
            "advcl" => Self::Advcl,
            "agent" => Self::Agent,
            "amod" => Self::Amod,
            "aux" => Self::Aux,
            "auxpass" => Self::AuxPass,
            "cc" => Self::Cc,
            "ccomp" => Self::Ccomp,
            "conj" => Self::Conj,
            "csubj" => Self::Csubj,
            "det" => Self::Det,
            "dobj" => Self::Dobj,
            "mark" => Self::Mark,
            "nsubj" => Self::Nsubj,
            "nsubjpass" => Self::NsubjPass,
            "pobj" => Self::Pobj,
            "prep" => Self::Prep,
            "xcomp" => Self::Xcomp,
            "punct" => Self::Punct,
            "other" => Self::Other,
            other => return Err(ParseDepRelationError(other.into())),
        })
    }
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
    /// `children[head]` is every token index whose edge's `head` is
    /// `Some(head)`, in token order — built once in [`Self::new`] so a
    /// caller walking subtrees (`friction_register::transduce`'s
    /// `subtree_indices`/`children_with_relation`) looks up a token's
    /// children in `O(1)` instead of re-scanning every edge in the
    /// sentence per node visited. A tree over `edges.len()` tokens has
    /// exactly `edges.len()` parent-child links total, so this costs one
    /// `O(tokens)` pass to build and `O(tokens)` total space.
    children: Vec<Vec<usize>>,
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
        let mut children = vec![Vec::new(); edges.len()];
        for edge in &edges {
            if let Some(head) = edge.head {
                children[head].push(edge.token);
            }
        }
        Ok(Self { edges, children })
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

    /// Every token index whose edge's `head` is exactly `index`, in token
    /// order — `index`'s direct children, none of its deeper descendants.
    /// Empty (not a panic) if `index` is out of bounds, matching
    /// [`Self::edge`]'s own out-of-bounds-is-empty convention.
    #[must_use]
    pub fn children_of(&self, index: usize) -> &[usize] {
        self.children.get(index).map_or(&[], Vec::as_slice)
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

/// Produces dependency structure, head indices and dependency relations,
/// for a part-of-speech-tagged sentence.
///
/// See the module docs for the input contract and the confidence contract
/// every [`DepEdge`] carries.
///
/// `Send + Sync`, for the same reason as [`crate::Tagger`]: callers parse
/// independent sentences concurrently through a single shared `&dyn
/// DepParser` (`friction-edit`'s register pass builds every sentence's
/// context with a rayon `par_iter`). [`crate::PerceptronParser`]'s own
/// scratch state is always call-local, never `self`.
pub trait DepParser: Send + Sync {
    /// Parses one sentence's dependency structure.
    ///
    /// `source` is the original document text `tokens`' spans address;
    /// `tokens` is that sentence's tagged tokens, in source order.
    ///
    /// # Errors
    /// Returns [`DepParseError`] if the parse can't be produced (no usable
    /// model, or an internal consistency failure) — implementations must
    /// not panic on any input, including a malformed or empty slice.
    fn parse(&self, source: &str, tokens: &[TaggedToken]) -> Result<SentenceParse, DepParseError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A well-formed, densely-indexed edge list is accepted.
    #[test]
    fn sentence_parse_new_accepts_well_formed_edges() {
        let edges = vec![
            DepEdge {
                token: 0,
                head: Some(1),
                relation: DepRelation::Nsubj,
                confidence: Confidence::CERTAIN,
            },
            DepEdge {
                token: 1,
                head: None,
                relation: DepRelation::Root,
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

    /// Every [`DepRelation`] variant's canonical name round-trips through
    /// [`DepRelation::as_str`] then [`DepRelation::from_str`] back to the
    /// same variant: the property a gold file's text labels depend on.
    #[test]
    fn dep_relation_round_trips_every_variant() {
        const ALL: [DepRelation; 21] = [
            DepRelation::Root,
            DepRelation::Acl,
            DepRelation::Advcl,
            DepRelation::Agent,
            DepRelation::Amod,
            DepRelation::Aux,
            DepRelation::AuxPass,
            DepRelation::Cc,
            DepRelation::Ccomp,
            DepRelation::Conj,
            DepRelation::Csubj,
            DepRelation::Det,
            DepRelation::Dobj,
            DepRelation::Mark,
            DepRelation::Nsubj,
            DepRelation::NsubjPass,
            DepRelation::Pobj,
            DepRelation::Prep,
            DepRelation::Xcomp,
            DepRelation::Punct,
            DepRelation::Other,
        ];
        for relation in ALL {
            let name = relation.as_str();
            assert_eq!(
                name.parse::<DepRelation>().unwrap(),
                relation,
                "round-trip failed for {name:?}"
            );
        }
    }

    /// An unrecognized name is rejected rather than silently coerced to
    /// [`DepRelation::Other`]: a gold file's typo should surface as a
    /// load-time error, not disappear into the catch-all bucket.
    #[test]
    fn dep_relation_from_str_rejects_unknown_names() {
        assert!("Nsubj".parse::<DepRelation>().is_err());
        assert!("obj".parse::<DepRelation>().is_err());
        assert!("".parse::<DepRelation>().is_err());
    }
}
