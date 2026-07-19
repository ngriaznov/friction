//! Core domain types shared across the `friction` workspace.
//!
//! Defines [`Document`], [`Block`]/[`BlockKind`], [`ProseUnit`]/[`Sentence`]/
//! [`Token`], [`Patch`]/[`Tier`], [`MetricVector`], [`Envelope`], [`RuleId`],
//! [`Finding`], and the shared [`CoreError`] error type. Every other crate
//! in the workspace builds on these shapes; `friction-core` itself performs
//! no parsing, NLP, or metric computation — see `friction-parse`,
//! `friction-nlp`, and `friction-metrics` for that.
//!
//! # Span honesty
//!
//! Every byte range in this crate — on a [`Block`], [`ProseUnit`],
//! [`Sentence`], [`Token`], [`Patch`], or [`Finding`] — is an offset into
//! the *original* source text of the [`Document`] it was produced from, not
//! into some intermediate representation. The [`span`] module centralizes
//! the two checks that keep that invariant checkable rather than merely
//! assumed: [`span::validate_range`] (in-bounds + UTF-8 char-boundary) and
//! [`span::contains_range`] (parent/child containment); [`Document::new`]
//! runs both recursively over the whole block/prose/sentence/token tree.
//! [`span::Spanned`] gives span-generic code a single trait to program
//! against.
//!
//! # Determinism
//!
//! No type in this crate uses `HashMap`/`HashSet`, and none of its methods
//! consult wall-clock time or ambient randomness. Where a fixed, meaningful
//! order matters (e.g. addressing a [`MetricVector`] generically), it is
//! declaration order, exposed via [`MetricVector::FIELD_NAMES`] /
//! [`MetricVector::named_values`], not iteration order over a hash-based
//! collection.
//!
//! # Meaning-preservation tiers
//!
//! [`Tier`] marks every [`Patch`] and [`Finding`] as either `Fix`
//! (machine-applicable, meaning-preserving by construction) or `Suggest`
//! (diagnostic only, never auto-applied).
//!
//! # Conflict detection
//!
//! [`find_overlaps`] detects overlapping patch ranges within a round.
//! Resolving conflicts (leftmost-longest, then rule priority) is
//! `friction-apply`'s job; this crate only detects them.

mod block;
mod document;
mod envelope;
mod error;
mod finding;
mod metrics;
mod patch;
mod rule;
pub mod span;

pub use block::{Block, BlockKind};
pub use document::{Document, ProseUnit, Sentence, Token, TokenKind};
pub use envelope::Envelope;
pub use error::CoreError;
pub use finding::Finding;
pub use metrics::MetricVector;
pub use patch::{Patch, Tier, find_overlaps};
pub use rule::RuleId;
pub use span::Spanned;
