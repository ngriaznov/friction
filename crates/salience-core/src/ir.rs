//! The language-neutral IR contract.
//!
//! This is the seam that makes the analysis multi-language. A frontend lowers
//! whatever substrate it has — JVM bytecode, CPython bytecode, LLVM IR, MIR —
//! into [`FunctionIr`] and gets tiering, scoring and line projection for free.
//!
//! The contract is deliberately small. A frontend must answer four questions
//! per instruction: what line did it come from, what does it define, what does
//! it use, and where can control go next. Everything else the core derives.
//!
//! Two properties are load-bearing:
//!
//! - **The graph is instruction-level, not block-level.** Frontends do not have
//!   to discover basic blocks; every substrate we care about already exposes a
//!   per-instruction successor relation.
//! - **Definitions need not be in SSA form.** The core computes reaching
//!   definitions itself, so a frontend over non-SSA locals (CPython's
//!   `STORE_FAST`) is exactly as correct as one over an SSA IR (mokapot's
//!   MokaIR). Frontends report facts; they do not do dataflow.

use std::collections::BTreeSet;

/// Index of a node in [`FunctionIr::nodes`].
pub type NodeId = usize;

/// A frontend-assigned variable identifier.
///
/// The core attaches no meaning to the numbering beyond equality: two nodes
/// referring to `VarId(7)` are talking about the same storage location. A
/// frontend over an SSA IR may hand out a fresh id per definition; a frontend
/// over mutable locals reuses one id per slot. Both are correct because
/// reaching definitions is computed here, not there.
pub type VarId = u32;

/// How a call site relates to the analysis frontier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CallOpacity {
    /// An atom at the abstraction frontier: the callee's body is never
    /// analyzed, so the call is treated as an effect.
    Opaque,
    /// A call known to carry no behavior — logging, metrics, tracing. Matched
    /// against the [`Denylist`](crate::Denylist), never inferred.
    Inert,
}

/// What a node does, as far as salience is concerned.
///
/// This is not a disassembly. It is the minimum classification the tiering
/// rules dispatch on, and every substrate can produce it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NodeKind {
    /// Computation with no effect outside the function: arithmetic, moves,
    /// constants, conversions, local reads.
    Pure,
    /// A conditional transfer of control. Must have more than one successor
    /// for its control-dominance weight to be non-zero.
    Branch,
    /// Returns from the function. An effect: values reaching it are observable.
    Return,
    /// Throws. An effect, and an abnormal exit for post-dominance purposes.
    Throw,
    /// A call. Effect-ness depends on [`CallOpacity`].
    Call {
        /// Fully-qualified callee name, used for denylist matching and for the
        /// `reason` text in the artifact.
        callee: String,
        /// Whether the callee is an opaque atom or a known-inert call.
        opacity: CallOpacity,
    },
    /// A write to storage that outlives the function: instance or static
    /// field, array element, global, captured cell.
    StateWrite {
        /// Human-readable target, e.g. `com.example.Cache#entries`.
        target: String,
    },
}

impl NodeKind {
    /// Whether this node is an effect — a point where behavior becomes
    /// observable outside the function body.
    ///
    /// Effects are the anchors of the backward slice: a statement is
    /// behavior-carrying precisely when a definition it makes can reach one.
    #[must_use]
    pub fn is_effect(&self) -> bool {
        match self {
            Self::Return | Self::Throw | Self::StateWrite { .. } => true,
            Self::Call { opacity, .. } => *opacity == CallOpacity::Opaque,
            Self::Pure | Self::Branch => false,
        }
    }

    /// Whether this node is denylisted as carrying no behavior.
    #[must_use]
    pub fn is_inert(&self) -> bool {
        matches!(
            self,
            Self::Call {
                opacity: CallOpacity::Inert,
                ..
            }
        )
    }

    /// A short stable tag used in artifact reasons.
    #[must_use]
    pub fn tag(&self) -> &'static str {
        match self {
            Self::Pure => "pure",
            Self::Branch => "branch",
            Self::Return => "return",
            Self::Throw => "throw",
            Self::Call { .. } => "call",
            Self::StateWrite { .. } => "state-write",
        }
    }
}

/// One instruction in the language-neutral graph.
#[derive(Debug, Clone)]
pub struct Node {
    /// Source line this instruction was lowered from, if the substrate carries
    /// one. `None` for compiler-synthesized instructions with no source
    /// position — these still participate in the analysis but project onto no
    /// line, which is exactly the behavior we want for synthetic constructs.
    pub line: Option<u32>,
    /// What this instruction does.
    pub kind: NodeKind,
    /// Variables this instruction defines.
    pub defs: Vec<VarId>,
    /// Variables this instruction reads.
    pub uses: Vec<VarId>,
    /// Control-flow successors, as node indices.
    pub succs: Vec<NodeId>,
    /// Frontend-supplied text for debugging and for the artifact's `detail`
    /// field. Never parsed.
    pub label: String,
}

impl Node {
    /// Whether this node carries no salience signal of its own.
    ///
    /// An unconditional jump, a `nop`, or a stack-housekeeping instruction has
    /// no definitions, reads nothing, and is not an effect. It is real in the
    /// graph — control flows through it, and dominance depends on it — but it
    /// says nothing about whether the *source line* it came from matters. A
    /// closing brace compiled to a `goto` should not make its line read as
    /// plumbing, so these are excluded from line projection while remaining
    /// full participants in the analysis.
    #[must_use]
    pub fn is_structural(&self) -> bool {
        self.defs.is_empty() && self.uses.is_empty() && matches!(self.kind, NodeKind::Pure)
    }

    /// A pure node with no dataflow, used mostly in tests.
    #[must_use]
    pub fn pure(line: u32) -> Self {
        Self {
            line: Some(line),
            kind: NodeKind::Pure,
            defs: Vec::new(),
            uses: Vec::new(),
            succs: Vec::new(),
            label: String::new(),
        }
    }

    /// Sets this node's successors, returning it, for terse construction.
    #[must_use]
    pub fn with_succs(mut self, succs: impl IntoIterator<Item = NodeId>) -> Self {
        self.succs = succs.into_iter().collect();
        self
    }

    /// Sets this node's defs and uses, returning it, for terse construction.
    #[must_use]
    pub fn with_dataflow(
        mut self,
        defs: impl IntoIterator<Item = VarId>,
        uses: impl IntoIterator<Item = VarId>,
    ) -> Self {
        self.defs = defs.into_iter().collect();
        self.uses = uses.into_iter().collect();
        self
    }

    /// Sets this node's kind, returning it, for terse construction.
    #[must_use]
    pub fn with_kind(mut self, kind: NodeKind) -> Self {
        self.kind = kind;
        self
    }
}

/// Identity of the analyzed function, carried through to the artifact.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct FunctionId {
    /// Source file path as recorded by the substrate.
    pub file: String,
    /// Fully-qualified function name including any owner type.
    pub name: String,
    /// Signature or descriptor, when the substrate has one. Empty otherwise.
    pub signature: String,
    /// First line of the function body, when known.
    pub decl_line: Option<u32>,
}

/// A single function, lowered into the neutral contract.
#[derive(Debug, Clone)]
pub struct FunctionIr {
    /// Who this is.
    pub id: FunctionId,
    /// Instructions, indexed by [`NodeId`].
    pub nodes: Vec<Node>,
    /// Entry node. Conventionally `0`, but stated explicitly because some
    /// substrates emit prologue nodes out of order.
    pub entry: NodeId,
}

impl FunctionIr {
    /// Number of nodes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the function has no instructions — an abstract or native method,
    /// or a body the substrate declined to lower.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Every distinct source line this function's instructions map to.
    ///
    /// Used to bound line-range projection, and to report how much of the body
    /// carried position information at all.
    #[must_use]
    pub fn lines(&self) -> BTreeSet<u32> {
        self.nodes.iter().filter_map(|n| n.line).collect()
    }

    /// Nodes that are effects — the anchors of the backward slice.
    #[must_use]
    pub fn effects(&self) -> Vec<NodeId> {
        (0..self.nodes.len())
            .filter(|&n| self.nodes[n].kind.is_effect())
            .collect()
    }

    /// Validates the graph's internal consistency.
    ///
    /// A frontend bug that emits an out-of-range successor would otherwise
    /// surface as a panic deep in the dominator computation, so we check once
    /// at the boundary and report which node is wrong.
    ///
    /// # Errors
    ///
    /// Returns the offending node and successor when a successor index is out
    /// of range, or when the entry node is.
    pub fn validate(&self) -> Result<(), IrError> {
        if !self.nodes.is_empty() && self.entry >= self.nodes.len() {
            return Err(IrError::EntryOutOfRange {
                entry: self.entry,
                len: self.nodes.len(),
            });
        }
        for (id, node) in self.nodes.iter().enumerate() {
            for &s in &node.succs {
                if s >= self.nodes.len() {
                    return Err(IrError::SuccessorOutOfRange {
                        node: id,
                        succ: s,
                        len: self.nodes.len(),
                    });
                }
            }
        }
        Ok(())
    }
}

/// A malformed [`FunctionIr`] handed in by a frontend.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum IrError {
    /// The entry index is past the end of the node list.
    #[error("entry node {entry} is out of range (function has {len} nodes)")]
    EntryOutOfRange {
        /// The bad entry index.
        entry: NodeId,
        /// Number of nodes present.
        len: usize,
    },
    /// A successor index is past the end of the node list.
    #[error("node {node} has successor {succ}, which is out of range (function has {len} nodes)")]
    SuccessorOutOfRange {
        /// The node carrying the bad edge.
        node: NodeId,
        /// The out-of-range successor.
        succ: NodeId,
        /// Number of nodes present.
        len: usize,
    },
}
