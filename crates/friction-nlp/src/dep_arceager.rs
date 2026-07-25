//! The arc-eager transition system: [`Configuration`], the shift/reduce/
//! attach state machine a dependency parser steps through left to right,
//! and the static [`oracle`] that recovers the unique transition sequence
//! a known-correct [`SentenceParse`] reduces to.
//!
//! This module is the transition system itself, not a parser: nothing
//! here reads a gold file, holds a weight vector, or makes a probabilistic
//! choice. [`oracle`]/[`derive`] exist so a trainer can turn gold trees
//! into the `(configuration, correct transition)` pairs its classifier
//! learns from; [`Configuration::is_allowed`]/[`Configuration::apply`]
//! exist so that same trainer (or, later, a trained model) can drive the
//! system at inference time. Every function here is a pure function of
//! its arguments — no RNG, no hashing over an order that could vary
//! between runs or platforms — so the same gold tree always derives the
//! identical transition sequence.
//!
//! # Why arc-eager
//!
//! Arc-standard (the other classic choice) only attaches a token to its
//! head once every one of that token's own dependents has already been
//! collected, which forces a long right-branching chain — a verb governing
//! a stack of trailing prepositional/participial modifiers, exactly the
//! shape this workspace's target sentences skew toward — to sit unattached
//! on the stack until the whole chain has been shifted. Arc-eager attaches
//! a token to its head the moment the two are adjacent at the stack/buffer
//! boundary, which keeps the stack shallow for that shape and lets
//! [`Transition::RightArc`] fire immediately instead of accumulating
//! depth.
//!
//! # Projectivity is a hard limit, not a bug
//!
//! This system can only produce projective trees: dependency arcs that
//! never cross when drawn as arcs above the sentence. [`derive`] verifies
//! its own output by replaying it and comparing the result against gold;
//! when a gold tree has a crossing arc (or is otherwise unreachable by
//! this transition system) it returns [`DeriveError`] instead of quietly
//! handing back a transition sequence that reproduces the wrong tree. A
//! caller building training data from gold trees is expected to drop, not
//! repair, any sentence that fails here.

use crate::dep::{Confidence, DepEdge, DepRelation, SentenceParse};

/// One step of the arc-eager transition system.
///
/// The two arc-creating variants carry the [`DepRelation`] the new arc is
/// labelled with; [`DepRelation::Root`] is never a valid label here (see
/// that variant's own docs) — nothing in this module enforces that by
/// construction, since the type would have to either duplicate
/// [`DepRelation`] minus one variant or accept a label it then has to
/// reject at run time, and run-time rejection already happens naturally:
/// a static oracle-derived transition never proposes `Root` as an arc
/// label in the first place (see [`oracle`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transition {
    /// Moves the buffer's front token onto the stack.
    Shift,
    /// Pops the stack's top token. Only legal once that token already has
    /// a head — see [`Configuration::is_allowed`].
    Reduce,
    /// Attaches the stack's top token as a dependent of the buffer's front
    /// token, labelled `relation`, then pops the stack top.
    LeftArc(DepRelation),
    /// Attaches the buffer's front token as a dependent of the stack's top
    /// token, labelled `relation`, then shifts that (now-attached) token
    /// onto the stack.
    RightArc(DepRelation),
}

/// The arc-eager parser state: which tokens are still to be shifted, which
/// are held on the stack awaiting attachment, and which arcs have been
/// assigned so far.
///
/// Deliberately carries no notion of an artificial root node distinct from
/// the real tokens — [`crate::dep::SentenceParse`]'s own data model already
/// represents "this token is the root" as `head: None`, so a token that
/// never receives an arc during the derivation simply stays that way; see
/// [`Configuration::finish`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Configuration {
    /// Token indices held for possible attachment, top (most recent) at
    /// the end.
    stack: Vec<usize>,
    /// Index of the buffer's front token; every index `< buffer` has
    /// already been shifted, every index `>= buffer` is still pending.
    buffer: usize,
    /// This sentence's token count, fixed at construction.
    len: usize,
    /// Per-token `(head, relation)` once assigned; `None` until then (and,
    /// for a genuine root token, forever — see [`Configuration::finish`]).
    arcs: Vec<Option<(usize, DepRelation)>>,
}

impl Configuration {
    /// The initial configuration for a sentence of `token_count` tokens:
    /// an empty stack, the whole sentence still in the buffer, and no arcs
    /// assigned.
    #[must_use]
    pub fn new(token_count: usize) -> Self {
        Self {
            stack: Vec::new(),
            buffer: 0,
            len: token_count,
            arcs: vec![None; token_count],
        }
    }

    /// The stack, bottom to top; the last element (if any) is the current
    /// top.
    #[must_use]
    pub fn stack(&self) -> &[usize] {
        &self.stack
    }

    /// The buffer's front token index, or `None` once every token has been
    /// shifted.
    #[must_use]
    pub fn buffer_front(&self) -> Option<usize> {
        (self.buffer < self.len).then_some(self.buffer)
    }

    /// This configuration's sentence's total token count.
    #[must_use]
    pub const fn token_count(&self) -> usize {
        self.len
    }

    /// `token`'s assigned `(head, relation)`, if any arc has attached it
    /// yet.
    #[must_use]
    pub fn arc(&self, token: usize) -> Option<(usize, DepRelation)> {
        self.arcs[token]
    }

    /// Whether `transition` may legally be applied to this configuration —
    /// the standard arc-eager preconditions:
    ///
    /// - [`Transition::Shift`]: the buffer is non-empty.
    /// - [`Transition::LeftArc`]: the stack and buffer are both
    ///   non-empty, and the stack's top token has no head yet (an
    ///   arc-eager token gets at most one head, ever, so re-attaching an
    ///   already-headed token would silently overwrite its real one).
    /// - [`Transition::RightArc`]: the stack and buffer are both
    ///   non-empty. No head-check on either side: the buffer's front
    ///   token is fresh out of the buffer and cannot already have a head.
    /// - [`Transition::Reduce`]: the stack is non-empty and its top token
    ///   already has a head (popping a still-headless token would strand
    ///   it: nothing can attach to a token buried below the stack top
    ///   ever again).
    #[must_use]
    pub fn is_allowed(&self, transition: Transition) -> bool {
        let buffer_nonempty = self.buffer < self.len;
        match transition {
            Transition::Shift => buffer_nonempty,
            Transition::LeftArc(_) => {
                buffer_nonempty
                    && self
                        .stack
                        .last()
                        .is_some_and(|&top| self.arcs[top].is_none())
            }
            Transition::RightArc(_) => buffer_nonempty && !self.stack.is_empty(),
            Transition::Reduce => self
                .stack
                .last()
                .is_some_and(|&top| self.arcs[top].is_some()),
        }
    }

    /// Applies `transition`, mutating this configuration in place.
    ///
    /// # Panics
    /// Panics if [`Configuration::is_allowed`] would return `false` for
    /// `transition`, rather than applying a partial or corrupted effect.
    /// This is a deliberate choice over returning `Result`: a disallowed
    /// transition reaching this method is a bug in the caller (the
    /// [`oracle`] never proposes one, and a beam/greedy driver above this
    /// type is expected to have already consulted
    /// [`Configuration::is_allowed`] before ranking candidates) — not a
    /// data-dependent failure a caller should be routing through `?`
    /// error-propagation on every single step. Forcing a `Result` return
    /// here would push every call site into either `.unwrap()`ing (no
    /// safer than a panic, just spelled differently) or writing dead
    /// recovery code for a state this system's own invariants already
    /// make unreachable.
    pub fn apply(&mut self, transition: Transition) {
        assert!(
            self.is_allowed(transition),
            "disallowed transition {transition:?} for configuration {self:?}"
        );
        match transition {
            Transition::Shift => {
                self.stack.push(self.buffer);
                self.buffer += 1;
            }
            Transition::LeftArc(relation) => {
                let top = self.stack.pop().expect("is_allowed checked stack.last()");
                self.arcs[top] = Some((self.buffer, relation));
            }
            Transition::RightArc(relation) => {
                let top = *self.stack.last().expect("is_allowed checked !is_empty()");
                self.arcs[self.buffer] = Some((top, relation));
                self.stack.push(self.buffer);
                self.buffer += 1;
            }
            Transition::Reduce => {
                self.stack.pop();
            }
        }
    }

    /// Whether no further transition is legal: the buffer is exhausted
    /// and the stack is either empty or stuck on a headless top (which
    /// [`Transition::Reduce`] refuses, and every other transition
    /// requires a non-empty buffer).
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        self.buffer >= self.len
            && self
                .stack
                .last()
                .is_none_or(|&top| self.arcs[top].is_none())
    }

    /// Consumes this configuration and materializes it as one [`DepEdge`]
    /// per token, ready for [`SentenceParse::new`].
    ///
    /// Every token still lacking a head — whether the sentence's genuine
    /// root or a token this derivation simply never got around to
    /// attaching — becomes a root edge (`head: None`,
    /// [`DepRelation::Root`]). This finishing step runs unconditionally,
    /// not only when [`Configuration::is_terminal`] holds: a caller
    /// walking the resulting tree (a subtree walk, a `by_relation` scan)
    /// has no way to represent "and also this one token has no opinion
    /// yet", so a complete-but-possibly-imperfect tree is always
    /// preferable to a structurally partial one. Every edge carries
    /// [`Confidence::CERTAIN`]: this transition system has no per-decision
    /// score of its own to report — an oracle-driven derivation is
    /// replaying a known-correct tree, not weighing alternatives, and a
    /// trained model driving the same [`Configuration`] at inference time
    /// is expected to attach its own real margin before a caller ever
    /// sees these edges.
    #[must_use]
    pub fn finish(self) -> Vec<DepEdge> {
        self.arcs
            .into_iter()
            .enumerate()
            .map(|(token, arc)| {
                let (head, relation) = match arc {
                    Some((head, relation)) => (Some(head), relation),
                    None => (None, DepRelation::Root),
                };
                DepEdge {
                    token,
                    head,
                    relation,
                    confidence: Confidence::CERTAIN,
                }
            })
            .collect()
    }
}

/// `gold`'s recorded head for `token`, or `None` if `token` is gold's root.
fn gold_head(gold: &SentenceParse, token: usize) -> Option<usize> {
    gold.edge(token).and_then(|edge| edge.head)
}

/// `gold`'s recorded relation for `token`.
///
/// # Panics
/// Panics if `token` is out of bounds for `gold` — every caller in this
/// module only ever passes a token index drawn from a [`Configuration`]
/// constructed over `gold.edges().len()` tokens, so this is an internal
/// invariant, not a data-dependent condition.
fn gold_relation(gold: &SentenceParse, token: usize) -> DepRelation {
    gold.edge(token)
        .unwrap_or_else(|| panic!("token {token} out of bounds for this gold parse"))
        .relation
}

/// Whether any token still waiting in the buffer (from `config`'s current
/// front onward, inclusive) has `head` as its gold head — i.e. whether
/// `head` still has an unattached gold dependent ahead of it.
fn has_remaining_gold_child(gold: &SentenceParse, config: &Configuration, head: usize) -> bool {
    (config.buffer..config.len).any(|token| gold_head(gold, token) == Some(head))
}

/// The static arc-eager oracle: the unique transition [`derive`] applies
/// next to reproduce `gold` from `config`, or `None` once `config` is
/// terminal.
///
/// When more than one rule's precondition would independently hold —
/// which never actually happens for a well-formed projective gold tree,
/// since a 2-cycle would be required for both the `LeftArc` and
/// `RightArc` conditions to hold at once — priority is fixed in the order
/// the rules are listed below (`LeftArc` first, `Shift` last: the lowest
/// [`Transition`] variant that applies wins), so this function never has
/// to consult anything beyond that fixed order to stay deterministic:
///
/// 1. [`Transition::LeftArc`] if the stack top's gold head is the buffer
///    front.
/// 2. [`Transition::RightArc`] if the buffer front's gold head is the
///    stack top.
/// 3. [`Transition::Reduce`] if the stack top already has a head assigned
///    and no gold dependent of it remains in the buffer (see
///    [`has_remaining_gold_child`]).
/// 4. [`Transition::Shift`] otherwise.
#[must_use]
pub fn oracle(gold: &SentenceParse, config: &Configuration) -> Option<Transition> {
    if config.is_terminal() {
        return None;
    }

    let stack_top = config.stack.last().copied();
    let buffer_front = config.buffer_front();

    if let (Some(top), Some(front)) = (stack_top, buffer_front) {
        if gold_head(gold, top) == Some(front) {
            return Some(Transition::LeftArc(gold_relation(gold, top)));
        }
        if gold_head(gold, front) == Some(top) {
            return Some(Transition::RightArc(gold_relation(gold, front)));
        }
    }

    if let Some(top) = stack_top
        && config.arcs[top].is_some()
        && !has_remaining_gold_child(gold, config, top)
    {
        return Some(Transition::Reduce);
    }

    Some(Transition::Shift)
}

/// A gold tree could not be fully reproduced by the arc-eager transition
/// system.
///
/// In practice this means `gold` has at least one non-projective
/// (crossing) arc: this transition system can only ever build a
/// projective tree, since the stack/buffer discipline has no way to defer
/// an attachment past an intervening token and revisit it later. It can
/// also mean the token this arc-eager system attaches a dependent to at
/// the moment they become adjacent disagrees with `gold`'s recorded head
/// for some other, structural reason (e.g. a head assigned to a token
/// already past the point where it could still receive one). Either way,
/// the fix is to drop the sentence from training data, not to patch this
/// system into accepting a tree shape it fundamentally cannot represent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("gold parse is not reachable by the arc-eager transition system (non-projective?)")]
pub struct DeriveError;

/// Runs the [`oracle`] to completion over a fresh [`Configuration`] for
/// `gold`, returning the transition sequence that reproduces it.
///
/// # Errors
/// Returns [`DeriveError`] if replaying the derived transitions does not
/// reproduce `gold`'s arcs exactly — see [`DeriveError`]'s own docs for
/// what that means and why it happens instead of a silent partial result.
pub fn derive(gold: &SentenceParse) -> Result<Vec<Transition>, DeriveError> {
    let mut config = Configuration::new(gold.edges().len());
    let mut transitions = Vec::new();
    while let Some(transition) = oracle(gold, &config) {
        config.apply(transition);
        transitions.push(transition);
    }

    let produced = config.finish();
    let reproduces_gold = produced
        .iter()
        .zip(gold.edges())
        .all(|(produced, gold)| produced.head == gold.head && produced.relation == gold.relation);

    if reproduces_gold {
        Ok(transitions)
    } else {
        Err(DeriveError)
    }
}
