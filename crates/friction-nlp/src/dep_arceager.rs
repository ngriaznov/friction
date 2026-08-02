//! The arc-eager transition system: [`Configuration`], the shift/reduce/
//! attach state machine a dependency parser steps through left to right,
//! and the static [`oracle`] that recovers the transition sequence a
//! known-correct [`SentenceParse`] reduces to.
//!
//! Not a parser itself: nothing here reads a gold file, holds weights, or
//! makes a probabilistic choice. [`oracle`]/[`derive`] turn gold trees
//! into `(configuration, correct transition)` training pairs;
//! [`Configuration::is_allowed`]/[`Configuration::apply`] let a trainer
//! (or, later, a trained model) drive the system at inference. Every
//! function is pure (no RNG, no order-dependent hashing) so a gold tree
//! always derives the identical sequence.
//!
//! # Why arc-eager
//!
//! Arc-standard only attaches a token once all its own dependents are
//! collected, forcing long right-branching chains — this workspace's
//! sentences skew toward a verb governing trailing modifiers — to sit
//! unattached until the whole chain is shifted. Arc-eager attaches a
//! token the moment it's adjacent to its head, keeping the stack shallow.
//!
//! # Projectivity is a hard limit, not a bug
//!
//! Only projective trees are producible: arcs that never cross when drawn
//! above the sentence. [`derive`] verifies its output by replaying it
//! against gold; a crossing arc returns [`DeriveError`] rather than
//! quietly reproducing the wrong tree. Callers are expected to drop, not
//! repair, any sentence that fails here.

use crate::dep::{Confidence, DepEdge, DepRelation, SentenceParse};

/// One step of the arc-eager transition system.
///
/// The two arc-creating variants carry the [`DepRelation`] label;
/// [`DepRelation::Root`] is never valid here — unenforced by construction
/// (not worth a near-duplicate type), but the oracle never proposes
/// `Root` as a label anyway (see [`oracle`]).
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

/// The arc-eager parser state: tokens still to be shifted, tokens held on
/// the stack awaiting attachment, and arcs assigned so far.
///
/// No artificial root node — [`crate::dep::SentenceParse`] already
/// represents "this token is the root" as `head: None`, so an unattached
/// token simply stays that way; see [`Configuration::finish`].
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
    /// empty stack, whole sentence in the buffer, no arcs assigned.
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

    /// This configuration's total token count.
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

    /// Whether `transition` may legally be applied:
    ///
    /// - [`Transition::Shift`]: buffer non-empty.
    /// - [`Transition::LeftArc`]: stack and buffer non-empty, stack top
    ///   headless (a token gets one head ever; re-attaching would
    ///   silently overwrite it).
    /// - [`Transition::RightArc`]: stack and buffer non-empty. No
    ///   head-check: a token fresh out of the buffer can't have a head.
    /// - [`Transition::Reduce`]: stack non-empty, top already headed
    ///   (popping a headless token strands it: nothing can attach below
    ///   the stack top again).
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
    /// Panics if [`Configuration::is_allowed`] would return `false`. A
    /// disallowed transition reaching here is a caller bug ([`oracle`]
    /// never proposes one; a driver is expected to check
    /// [`is_allowed`](Configuration::is_allowed) first) — not worth
    /// `Result`-and-`?` on every step when every call site would just
    /// `.unwrap()` anyway.
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

    /// Whether no further transition is legal: buffer exhausted and the
    /// stack empty or stuck on a headless top ([`Transition::Reduce`]
    /// refuses that. Everything else needs a non-empty buffer).
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        self.buffer >= self.len
            && self
                .stack
                .last()
                .is_none_or(|&top| self.arcs[top].is_none())
    }

    /// Consumes this configuration into one [`DepEdge`] per token, ready
    /// for [`SentenceParse::new`].
    ///
    /// Any still-headless token — genuine root or one this derivation
    /// never reached — becomes a root edge (`head: None`,
    /// [`DepRelation::Root`]). Runs even when not terminal: a
    /// tree-walking caller has no way to represent "no opinion yet", so a
    /// complete but imperfect tree beats a partial one. Every edge
    /// carries [`Confidence::CERTAIN`] — an oracle replay isn't weighing
    /// alternatives; a trained model driving the same [`Configuration`]
    /// is expected to attach its own real margin before a caller sees
    /// these edges.
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
/// Panics if `token` is out of bounds — every caller passes an index
/// drawn from a [`Configuration`] built over `gold.edges().len()` tokens,
/// an internal invariant, not a data-dependent case.
fn gold_relation(gold: &SentenceParse, token: usize) -> DepRelation {
    gold.edge(token)
        .unwrap_or_else(|| panic!("token {token} out of bounds for this gold parse"))
        .relation
}

/// Whether `head` has an unattached gold dependent still in the buffer
/// (from `config`'s current front onward).
fn has_remaining_gold_child(gold: &SentenceParse, config: &Configuration, head: usize) -> bool {
    (config.buffer..config.len).any(|token| gold_head(gold, token) == Some(head))
}

/// The static arc-eager oracle: the transition [`derive`] applies next to
/// reproduce `gold` from `config`, or `None` once terminal.
///
/// Two rules can't hold at once for a well-formed projective tree, but
/// priority is still fixed by list order (`LeftArc` first, `Shift` last)
/// so this stays deterministic without extra bookkeeping:
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
/// Usually a non-projective (crossing) arc — the stack/buffer discipline
/// can't defer an attachment past an intervening token. Can also be a
/// structural disagreement with gold's recorded head. Either way, drop
/// the sentence rather than patch this system to accept a shape it can't
/// represent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("gold parse is not reachable by the arc-eager transition system (non-projective?)")]
pub struct DeriveError;

/// Runs the [`oracle`] to completion over a fresh [`Configuration`] for
/// `gold`, returning the transition sequence that reproduces it.
///
/// # Errors
/// Returns [`DeriveError`] if replaying the transitions doesn't reproduce
/// `gold`'s arcs exactly: see that type's docs for why.
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
