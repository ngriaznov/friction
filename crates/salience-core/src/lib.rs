//! Deterministic, inference-free per-line salience over function bodies.
//!
//! Given a function lowered into [`ir::FunctionIr`], this crate classifies every
//! statement into one of four tiers and projects the result onto source lines:
//!
//! | tier | meaning |
//! |------|---------|
//! | `core` | behavior-carrying: branch predicates weighted by what they control-dominate, loop-carried dataflow, statements on def-use chains reaching an effect |
//! | `boundary` | the frontier where behavior leaves the body: returns, throws, state writes, calls into opaque dependencies |
//! | `plumbing` | present but not behavior-carrying: local shuffling, results that reach no effect |
//! | `inert` | denylisted calls (logging, metrics, tracing) and the computation that exists only to feed them |
//!
//! Alongside the tier, every span carries a continuous score in `0.0..=1.0`.
//! The tier is for policy — an edit-gating hook wants a predicate. The score is
//! for ranking — a weighted call graph, a profiler choosing where to start, or a
//! vulnerability triage queue wants an ordering, and four buckets throw away the
//! gradient between a predicate guarding two lines and one guarding forty.
//!
//! # What this is not
//!
//! - **Not compression.** Nothing is removed. The artifact is metadata about
//!   source, and a consumer that ignores it sees the file unchanged.
//! - **Not learned.** No model runs, at build time or query time. Every tier is
//!   the output of a dominance, loop or reachability query, and every span
//!   carries the reason that produced it.
//! - **Not criterion-anchored.** Classic slicing answers "what affects *this*".
//!   This answers "what carries behavior at all", unconditionally, so the result
//!   is computed once and cached rather than recomputed per question.
//! - **Not repo-level ranking.** The unit is the statement inside one body.
//!
//! # Analysis scope
//!
//! Intraprocedural. Calls are atoms at the abstraction frontier: a call into a
//! dependency is an effect, and its body is never analyzed. That is a
//! deliberate ceiling, not a missing feature — it keeps the cost per function
//! bounded and independent of how large the dependency graph is.
//!
//! # Multi-language
//!
//! Nothing above mentions a language. Frontends lower a substrate into
//! [`ir::FunctionIr`] and get everything else; see `salience-jvm` (JVM bytecode
//! via mokapot) and `salience-py` (CPython bytecode). Because
//! [reaching definitions](graph) are computed here, a frontend over mutable
//! locals is exactly as sound as one over an SSA IR.
//!
//! # Example
//!
//! ```
//! use salience_core::ir::{FunctionId, FunctionIr, Node, NodeKind};
//! use salience_core::{Denylist, ScoreWeights, Tier, analyze};
//!
//! // if (x > 0) return x; else return 0;
//! let ir = FunctionIr {
//!     id: FunctionId {
//!         file: "Demo.java".into(),
//!         name: "clamp".into(),
//!         signature: String::new(),
//!         decl_line: Some(1),
//!     },
//!     nodes: vec![
//!         Node::pure(2).with_dataflow([], [1]).with_kind(NodeKind::Branch).with_succs([1, 2]),
//!         Node::pure(3).with_dataflow([], [1]).with_kind(NodeKind::Return),
//!         Node::pure(5).with_kind(NodeKind::Return),
//!     ],
//!     entry: 0,
//! };
//!
//! let sal = analyze(&ir, &Denylist::new(), &ScoreWeights::default());
//! assert_eq!(sal.nodes[0].tier, Tier::Core); // the predicate governs both arms
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs, clippy::pedantic)]
// `CPython`, `MokaIR`, `LLVM` and friends are proper nouns in prose here, not
// identifiers; backticking them would make the docs read like code.
#![allow(
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::module_name_repetitions
)]

pub mod artifact;
pub mod graph;
pub mod ir;
pub mod salience;

pub use artifact::{Sidecar, SpanRecord};
pub use graph::Graph;
pub use ir::{FunctionId, FunctionIr, IrError, Node, NodeKind};
pub use salience::{
    Denylist, FunctionSalience, LineSpan, NodeSalience, Reason, ScoreWeights, Tier, analyze,
    project_to_lines,
};
