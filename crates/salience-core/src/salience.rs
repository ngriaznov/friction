//! Tier assignment and scoring.
//!
//! Two outputs, for two different consumers:
//!
//! - A **tier** — `core` / `boundary` / `plumbing` / `inert` — which is a
//!   classification, meant for policy. An agent harness gating edits wants a
//!   predicate, not a number.
//! - A **score** in `0.0..=1.0`, which is a ranking. A weighted call graph, a
//!   profiler picking where to start, or a vulnerability triage queue wants an
//!   ordering, and collapsing that to four buckets throws away the gradient
//!   between a predicate guarding two lines and one guarding forty.
//!
//! Both are derived from the same graph facts, and neither involves a model.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::graph::Graph;
use crate::ir::{FunctionIr, NodeId, NodeKind};

/// How much behavior a statement carries.
///
/// Ordered by salience so that projecting several instructions onto one source
/// line is a `max`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Tier {
    /// Explicitly known to carry no behavior: a denylisted call, or computation
    /// that exists only to feed one.
    Inert,
    /// Present, but not behavior-carrying: local shuffling, temporaries, and
    /// computation whose results never reach an effect.
    Plumbing,
    /// The frontier where behavior becomes observable outside the body —
    /// returns, throws, state writes, and calls into opaque dependencies.
    Boundary,
    /// Behavior-carrying: branch predicates, loop-carried dataflow, and
    /// anything on a def-use chain that reaches an effect.
    Core,
}

impl Tier {
    /// Stable lowercase name used in the artifact.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Core => "core",
            Self::Boundary => "boundary",
            Self::Plumbing => "plumbing",
            Self::Inert => "inert",
        }
    }
}

/// Why a node landed in its tier.
///
/// Reasons are accumulated rather than replaced: a statement can be both a
/// predicate and loop-carried, and a consumer deciding whether to demand
/// verification benefits from seeing both.
#[derive(Debug, Clone, PartialEq)]
pub enum Reason {
    /// A conditional whose outcome governs `weight` of the function body.
    BranchPredicate {
        /// Fraction of the body control-dependent on this branch, `0.0..=1.0`.
        weight: f64,
        /// Number of instructions governed.
        governed: usize,
    },
    /// Defines a value that survives a loop iteration boundary.
    LoopCarried {
        /// Nesting depth of the innermost containing loop.
        depth: u32,
    },
    /// On a def-use or control chain reaching an effect.
    ReachesEffect {
        /// Number of dependence steps to the nearest effect.
        distance: u32,
        /// What that effect is.
        effect: String,
    },
    /// Is itself an effect site.
    EffectSite {
        /// The kind of effect: `return`, `throw`, `state-write`, `call`.
        kind: &'static str,
        /// Target or callee, when the frontend supplied one.
        target: String,
    },
    /// Matched the denylist.
    Denylisted {
        /// The callee that matched.
        callee: String,
    },
    /// Exists only to construct arguments for denylisted calls.
    InertSupport,
    /// Computation whose results reach no effect.
    Unreaching,
}

impl Reason {
    /// The tier this reason on its own implies.
    #[must_use]
    pub fn tier(&self) -> Tier {
        match self {
            Self::BranchPredicate { .. }
            | Self::LoopCarried { .. }
            | Self::ReachesEffect { .. } => Tier::Core,
            Self::EffectSite { .. } => Tier::Boundary,
            Self::Denylisted { .. } | Self::InertSupport => Tier::Inert,
            Self::Unreaching => Tier::Plumbing,
        }
    }

    /// One-line human phrasing, used verbatim in the artifact.
    #[must_use]
    pub fn describe(&self) -> String {
        match self {
            Self::BranchPredicate { weight, governed } => format!(
                "branch predicate governing {governed} instructions ({:.0}% of body)",
                weight * 100.0
            ),
            Self::LoopCarried { depth } => {
                format!("loop-carried definition at nesting depth {depth}")
            }
            Self::ReachesEffect { distance, effect } => {
                format!("reaches {effect} in {distance} dependence step(s)")
            }
            Self::EffectSite { kind, target } if target.is_empty() => (*kind).to_owned(),
            Self::EffectSite { kind, target } => format!("{kind} -> {target}"),
            Self::Denylisted { callee } => format!("denylisted call {callee}"),
            Self::InertSupport => "builds arguments for a denylisted call only".to_owned(),
            Self::Unreaching => "result reaches no effect".to_owned(),
        }
    }
}

/// Calls known to carry no behavior.
///
/// Matching is literal substring containment, deliberately. A regex engine
/// would invite patterns whose cost is input-dependent, and the whole point of
/// the artifact is a bounded, predictable analysis budget.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Denylist {
    patterns: Vec<String>,
}

impl Denylist {
    /// Logging, metrics and tracing across the substrates the prototype
    /// supports. Intentionally short: every entry is a place where a false
    /// positive silently demotes real behavior, so the list earns its way in.
    pub const DEFAULT: &'static [&'static str] = &[
        // JVM logging
        "org.slf4j.Logger",
        "ch.qos.logback",
        "org.apache.logging.log4j",
        "java.util.logging.Logger",
        "java.io.PrintStream::println",
        "java.io.PrintStream::print",
        // JVM metrics and tracing
        "io.micrometer",
        "io.opentelemetry",
        "io.opentracing",
        "com.codahale.metrics",
        // Python
        "logging.debug",
        "logging.info",
        "logging.warning",
        "logging.error",
        "logging.exception",
        "logging.critical",
        "logging.log",
        "logger.debug",
        "logger.info",
        "logger.warning",
        "logger.error",
        "logger.exception",
        "logger.critical",
        "log.debug",
        "log.info",
        "log.warning",
        "log.error",
        "print",
    ];

    /// The default denylist.
    #[must_use]
    pub fn new() -> Self {
        Self {
            patterns: Self::DEFAULT.iter().map(|s| (*s).to_owned()).collect(),
        }
    }

    /// An empty denylist — nothing is treated as inert.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            patterns: Vec::new(),
        }
    }

    /// Adds a pattern.
    #[must_use]
    pub fn with(mut self, pattern: impl Into<String>) -> Self {
        self.patterns.push(pattern.into());
        self
    }

    /// Whether `callee` matches any pattern.
    ///
    /// Matching is case-insensitive and aligned to name-segment boundaries.
    /// Both details are load-bearing rather than conveniences:
    ///
    /// - Logger handles are conventionally spelled `LOG`, `Log` or `logger`
    ///   depending on the codebase, and a case-sensitive list would silently
    ///   match one project's logging and miss the next one's.
    /// - A bare pattern with no separator, such as `print`, matches only a
    ///   callee that *is* that name. Plain substring containment would quietly
    ///   classify `pprint`, `footprint` and `sprint` as inert.
    /// - A qualified pattern must begin at a segment boundary, so `log.info`
    ///   matches `LOG.info` but not `mylog.info`.
    #[must_use]
    pub fn matches(&self, callee: &str) -> bool {
        let callee = callee.to_ascii_lowercase();
        self.patterns.iter().any(|pattern| {
            let pattern = pattern.to_ascii_lowercase();
            let qualified = pattern.contains('.') || pattern.contains(':');
            if !qualified {
                return callee == pattern;
            }
            match callee.find(&pattern) {
                Some(0) => true,
                Some(i) => matches!(callee.as_bytes()[i - 1], b'.' | b':'),
                None => false,
            }
        })
    }
}

impl Default for Denylist {
    fn default() -> Self {
        Self::new()
    }
}

/// Weights composing the continuous score.
///
/// These are the only tunable numbers in the crate. They affect ranking only —
/// tier assignment is categorical and ignores them entirely — so a consumer
/// that dislikes the defaults changes an ordering, never a policy decision.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoreWeights {
    /// Floor contributed by the `core` tier.
    pub base_core: f64,
    /// Floor contributed by the `boundary` tier.
    pub base_boundary: f64,
    /// Floor contributed by the `plumbing` tier.
    pub base_plumbing: f64,
    /// Multiplier on a branch's control-dominance fraction.
    pub control: f64,
    /// Multiplier on normalized loop nesting depth.
    pub loop_depth: f64,
    /// Multiplier on proximity to the nearest effect.
    pub proximity: f64,
}

impl Default for ScoreWeights {
    fn default() -> Self {
        Self {
            base_core: 0.55,
            base_boundary: 0.45,
            base_plumbing: 0.15,
            control: 0.35,
            loop_depth: 0.15,
            proximity: 0.20,
        }
    }
}

/// Per-node analysis result.
#[derive(Debug, Clone)]
pub struct NodeSalience {
    /// The node this describes.
    pub node: NodeId,
    /// Assigned tier.
    pub tier: Tier,
    /// Continuous salience, `0.0..=1.0`.
    pub score: f64,
    /// Every reason that applied, ordered by descending tier.
    pub reasons: Vec<Reason>,
}

/// The analysis of one function, before line projection.
#[derive(Debug, Clone)]
pub struct FunctionSalience {
    /// Per-node results, indexed by [`NodeId`].
    pub nodes: Vec<NodeSalience>,
    /// The derived graph, retained so callers can inspect loops and dominance
    /// without recomputing.
    pub graph: Graph,
}

/// Runs tiering and scoring over a lowered function.
///
/// # Panics
///
/// Does not panic on any graph shape. Callers should still run
/// [`FunctionIr::validate`] first, which reports frontend bugs with a useful
/// message instead of letting them surface as silently wrong tiers.
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn analyze(ir: &FunctionIr, denylist: &Denylist, weights: &ScoreWeights) -> FunctionSalience {
    let graph = Graph::build(ir);
    let n = ir.nodes.len();

    // --- denylist seeds --------------------------------------------------
    // A frontend reports `log.info(...)` as an opaque call, because a frontend
    // cannot know which callees are inert. Matching happens here so one policy
    // governs every language.
    let denylisted: Vec<bool> = (0..n)
        .map(|i| match &ir.nodes[i].kind {
            NodeKind::Call { callee, .. } => denylist.matches(callee),
            _ => ir.nodes[i].kind.is_inert(),
        })
        .collect();

    // --- inert set -------------------------------------------------------
    // A statement is inert when its value feeds a denylisted call and reaches
    // no hard effect. Both halves are needed: the first finds the logging
    // machinery, the second protects any value the logging merely happens to
    // share with real work.
    //
    // Both halves are backward reachability over def-use, deliberately, rather
    // than an "every consumer is inert" fixpoint. That formulation cannot
    // absorb a dependency cycle, and the single most valuable case here is a
    // cycle: a counter incremented only to be logged reads its own previous
    // value, so it and its increment form a loop a least-fixpoint never enters.
    //
    // "Hard effect" rather than "effect" is the other load-bearing choice. An
    // opaque call is an effect only because we declined to look inside it, so
    // `prices.size()` inside a log argument must not count as the value having
    // escaped — otherwise the denylist stops at the logging call itself and
    // leaves every argument expression tiered as behavior, which is not the job
    // it exists to do. Returns, throws and state writes are observable without
    // seeing inside any callee, so they always count. Absorbing opaque calls
    // this way is a real soundness trade, bounded to the frontier we already
    // chose not to cross.
    let feeds_denylist = backward_closure(&graph, (0..n).filter(|&i| denylisted[i]));
    let reaches_hard = backward_closure(
        &graph,
        (0..n).filter(|&i| is_hard_effect(&ir.nodes[i].kind)),
    );
    let mut inert: Vec<bool> = (0..n)
        .map(|i| denylisted[i] || (feeds_denylist[i] && !reaches_hard[i]))
        .collect();

    // Instructions that only *consume* an inert value sit downstream of that
    // closure rather than inside it, so backward reachability never reaches
    // them. CPython's `POP_TOP` discarding a log call's result is the common
    // case; without this pass a logging line reads as plumbing rather than
    // inert, because the discard is the one instruction on it that the closure
    // missed. The set only grows, so this terminates.
    let mut changed = true;
    while changed {
        changed = false;
        for node in 0..n {
            if inert[node]
                || is_hard_effect(&ir.nodes[node].kind)
                || !ir.nodes[node].defs.is_empty()
                || ir.nodes[node].uses.is_empty()
            {
                continue;
            }
            let sources: BTreeSet<NodeId> = graph.uses_defs[node]
                .iter()
                .map(|&d| graph.defs[d].node)
                .collect();
            if !sources.is_empty() && sources.iter().all(|&s| inert[s]) {
                inert[node] = true;
                changed = true;
            }
        }
    }

    // --- effects and the backward slice ---------------------------------
    // Distance from the nearest effect, over data and control dependence
    // combined. Effects seed at zero; everything reachable backward from them
    // is behavior-carrying, and how far it sits scales the score.
    //
    // Inert nodes are excluded as seeds. If `log.info(...)` — or the
    // `prices.size()` call built for it — anchored the slice, every argument
    // would come back core.
    let mut dist: Vec<Option<u32>> = vec![None; n];
    let mut nearest: Vec<String> = vec![String::new(); n];
    let mut q: VecDeque<NodeId> = VecDeque::new();
    for e in ir.effects() {
        if inert[e] {
            continue;
        }
        dist[e] = Some(0);
        nearest[e] = effect_label(&ir.nodes[e].kind);
        q.push_back(e);
    }
    while let Some(node) = q.pop_front() {
        let d = dist[node].expect("queued nodes always carry a distance");
        let label = nearest[node].clone();
        // Data dependence: whoever defined what this node reads.
        let mut back: BTreeSet<NodeId> = graph.uses_defs[node]
            .iter()
            .map(|&d| graph.defs[d].node)
            .collect();
        // Control dependence: whichever branches decide whether this runs.
        back.extend(graph.ctrl_deps[node].iter().copied());
        for p in back {
            if dist[p].is_none() && !inert[p] {
                dist[p] = Some(d + 1);
                nearest[p].clone_from(&label);
                q.push_back(p);
            }
        }
    }

    // --- per-node tier and score -----------------------------------------
    let mut nodes = Vec::with_capacity(n);
    for node in 0..n {
        let kind = &ir.nodes[node].kind;
        let mut reasons: Vec<Reason> = Vec::new();

        if inert[node] {
            reasons.push(match kind {
                NodeKind::Call { callee, .. } => Reason::Denylisted {
                    callee: callee.clone(),
                },
                _ => Reason::InertSupport,
            });
        } else {
            let weight = graph.control_weight(node);
            if matches!(kind, NodeKind::Branch) && weight > 0.0 {
                reasons.push(Reason::BranchPredicate {
                    weight,
                    governed: graph.controls[node].iter().filter(|&&m| m != node).count(),
                });
            }
            let carried = graph
                .defs
                .iter()
                .enumerate()
                .any(|(d, site)| site.node == node && graph.is_loop_carried(d));
            if carried {
                reasons.push(Reason::LoopCarried {
                    depth: graph.loop_depth[node],
                });
            }
            if kind.is_effect() {
                reasons.push(Reason::EffectSite {
                    kind: kind.tag(),
                    target: effect_target(kind),
                });
            } else if let Some(d) = dist[node]
                && d > 0
            {
                reasons.push(Reason::ReachesEffect {
                    distance: d,
                    effect: nearest[node].clone(),
                });
            }
            if reasons.is_empty() {
                reasons.push(Reason::Unreaching);
            }
        }

        reasons.sort_by_key(|r| std::cmp::Reverse(r.tier()));
        let tier = reasons
            .iter()
            .map(Reason::tier)
            .max()
            .unwrap_or(Tier::Plumbing);
        let score = score_of(tier, node, &graph, dist[node], weights);
        nodes.push(NodeSalience {
            node,
            tier,
            score,
            reasons,
        });
    }

    FunctionSalience { nodes, graph }
}

/// Composes the continuous score from the same facts that drove the tier.
fn score_of(tier: Tier, node: NodeId, graph: &Graph, dist: Option<u32>, w: &ScoreWeights) -> f64 {
    if tier == Tier::Inert {
        return 0.0;
    }
    let base = match tier {
        Tier::Core => w.base_core,
        Tier::Boundary => w.base_boundary,
        Tier::Plumbing => w.base_plumbing,
        Tier::Inert => 0.0,
    };
    let control = w.control * graph.control_weight(node);
    let depth = w.loop_depth * f64::from(graph.loop_depth[node].min(3)) / 3.0;
    let proximity = match dist {
        Some(d) => w.proximity / f64::from(d + 1),
        None => 0.0,
    };
    (base + control + depth + proximity).clamp(0.0, 1.0)
}

/// Every node that can reach one of `seeds` by following def-use edges
/// backwards — that is, every node whose value flows into a seed.
///
/// Data dependence only. The question these closures answer is what a value
/// *contributes to*, and control dependence answers a different one — whether a
/// statement runs at all. Mixing them in would make almost every statement in a
/// guarded block look like it feeds whatever the block eventually does.
fn backward_closure(graph: &Graph, seeds: impl Iterator<Item = NodeId>) -> Vec<bool> {
    let mut seen = vec![false; graph.n];
    let mut q: VecDeque<NodeId> = VecDeque::new();
    for s in seeds {
        if !seen[s] {
            seen[s] = true;
            q.push_back(s);
        }
    }
    while let Some(node) = q.pop_front() {
        for &d in &graph.uses_defs[node] {
            let src = graph.defs[d].node;
            if !seen[src] {
                seen[src] = true;
                q.push_back(src);
            }
        }
    }
    seen
}

/// Effects that stay effects no matter who consumes them.
///
/// The distinction from [`NodeKind::is_effect`] is the whole inert trade: an
/// opaque call is an effect only because we chose not to look inside it, so it
/// may be absorbed when its result feeds nothing but logging. A return, a throw
/// or a state write is observable without looking inside anything, so it never
/// is.
fn is_hard_effect(kind: &NodeKind) -> bool {
    matches!(
        kind,
        NodeKind::Return | NodeKind::Throw | NodeKind::StateWrite { .. }
    )
}

/// Short label for the effect a slice terminates at.
fn effect_label(kind: &NodeKind) -> String {
    match kind {
        NodeKind::Return => "return value".to_owned(),
        NodeKind::Throw => "throw".to_owned(),
        NodeKind::StateWrite { target } => format!("state write {target}"),
        NodeKind::Call { callee, .. } => format!("opaque call {callee}"),
        NodeKind::Pure | NodeKind::Branch => "effect".to_owned(),
    }
}

/// The target string recorded on an effect-site reason.
fn effect_target(kind: &NodeKind) -> String {
    match kind {
        NodeKind::StateWrite { target } => target.clone(),
        NodeKind::Call { callee, .. } => callee.clone(),
        _ => String::new(),
    }
}

/// Accumulator while several instructions are folded onto one source line:
/// the winning tier, the highest score seen, and every distinct reason with the
/// tier it implies.
type LineAccum = (Tier, f64, Vec<(Tier, String)>);

/// One contiguous run of source lines sharing a tier.
#[derive(Debug, Clone, PartialEq)]
pub struct LineSpan {
    /// First line, inclusive.
    pub start: u32,
    /// Last line, inclusive.
    pub end: u32,
    /// Tier for every line in the run.
    pub tier: Tier,
    /// Highest score of any instruction in the run.
    pub score: f64,
    /// Distinct reasons observed across the run, most salient first.
    pub reasons: Vec<String>,
}

/// Projects per-node results onto source lines, then merges runs.
///
/// Aggregation across the instructions on one line is a `max` over tiers, with
/// one deliberate asymmetry: `inert` is the *weakest* tier here, even though a
/// denylist match is the *strongest* signal at the node level. A line holding
/// both a log call and real work is not inert — the log call merely happens to
/// share it. Ordering [`Tier`] by salience makes that fall out of `max`.
#[must_use]
pub fn project_to_lines(ir: &FunctionIr, sal: &FunctionSalience) -> Vec<LineSpan> {
    // Reasons are carried with the tier they imply so the aggregate can be
    // ordered by salience. A line holding both a core statement and a plumbing
    // one must lead with the core reason: a consumer reading only the first
    // entry — a hook message, a gutter tooltip — would otherwise be told the
    // line is inconsequential precisely when it is not.
    let mut per_line: BTreeMap<u32, LineAccum> = BTreeMap::new();
    for ns in &sal.nodes {
        let Some(line) = ir.nodes[ns.node].line else {
            continue; // synthetic instruction with no source position
        };
        if ir.nodes[ns.node].is_structural() {
            continue; // jumps and nops carry no signal about their line
        }
        let entry = per_line
            .entry(line)
            .or_insert((Tier::Inert, 0.0, Vec::new()));
        if ns.tier > entry.0 {
            entry.0 = ns.tier;
        }
        if ns.score > entry.1 {
            entry.1 = ns.score;
        }
        for r in &ns.reasons {
            let text = r.describe();
            if !entry.2.iter().any(|(_, t)| *t == text) {
                entry.2.push((r.tier(), text));
            }
        }
    }

    // Merge consecutive lines that agree on tier. Gaps in line numbering break
    // a run: a blank line between two core statements is not itself core.
    let mut spans: Vec<LineSpan> = Vec::new();
    for (line, (tier, score, mut reasons)) in per_line {
        // Stable sort by descending tier keeps ties in discovery order, so the
        // output stays reproducible.
        reasons.sort_by(|a, b| b.0.cmp(&a.0));
        let reasons: Vec<String> = reasons.into_iter().map(|(_, t)| t).collect();
        match spans.last_mut() {
            Some(last) if last.tier == tier && last.end + 1 == line => {
                last.end = line;
                last.score = last.score.max(score);
                for r in reasons {
                    if !last.reasons.contains(&r) {
                        last.reasons.push(r);
                    }
                }
            }
            _ => spans.push(LineSpan {
                start: line,
                end: line,
                tier,
                score,
                reasons,
            }),
        }
    }
    spans
}
