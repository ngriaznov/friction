//! Behavioral tests for the tiering engine, written against hand-built IR so
//! they pin the algorithm rather than any one frontend's lowering.

use salience_core::graph::Graph;
use salience_core::ir::{CallOpacity, FunctionId, FunctionIr, Node, NodeKind};
use salience_core::{Denylist, ScoreWeights, Sidecar, Tier, analyze, project_to_lines};

fn func(name: &str, nodes: Vec<Node>) -> FunctionIr {
    FunctionIr {
        id: FunctionId {
            file: "Demo.java".into(),
            name: name.into(),
            signature: String::new(),
            decl_line: Some(1),
        },
        nodes,
        entry: 0,
    }
}

fn run(ir: &FunctionIr) -> salience_core::FunctionSalience {
    ir.validate().expect("test IR must be well-formed");
    analyze(ir, &Denylist::new(), &ScoreWeights::default())
}

/// `if (x > 0) return x; else return 0;`
///
/// The predicate governs both arms, so it is core; both returns are the
/// frontier where the value becomes observable.
#[test]
fn branch_predicate_is_core_and_returns_are_boundary() {
    let ir = func(
        "clamp",
        vec![
            Node::pure(2)
                .with_dataflow([], [1])
                .with_kind(NodeKind::Branch)
                .with_succs([1, 2]),
            Node::pure(3)
                .with_dataflow([], [1])
                .with_kind(NodeKind::Return),
            Node::pure(5).with_kind(NodeKind::Return),
        ],
    );
    let sal = run(&ir);
    assert_eq!(sal.nodes[0].tier, Tier::Core);
    assert_eq!(sal.nodes[1].tier, Tier::Boundary);
    assert_eq!(sal.nodes[2].tier, Tier::Boundary);

    // The predicate governs two of three instructions.
    let w = sal.graph.control_weight(0);
    assert!((w - 2.0 / 3.0).abs() < 1e-9, "control weight was {w}");
}

/// A predicate guarding a large region must outrank one guarding a small
/// region. This is the "weighted by how much of the body it control-dominates"
/// property, and it is what a ranking consumer actually buys.
#[test]
fn control_weight_scales_with_governed_region() {
    // n0 branches over a long guarded run; n1 branches over a single node.
    let mut nodes = vec![
        Node::pure(1).with_kind(NodeKind::Branch).with_succs([1, 8]),
        Node::pure(2).with_kind(NodeKind::Branch).with_succs([2, 3]),
        Node::pure(3).with_succs([3]),
        Node::pure(4).with_succs([4]),
        Node::pure(5).with_succs([5]),
        Node::pure(6).with_succs([6]),
        Node::pure(7).with_succs([7]),
        Node::pure(8).with_succs([8]),
    ];
    nodes.push(Node::pure(9).with_kind(NodeKind::Return));
    let ir = func("weights", nodes);
    let sal = run(&ir);

    let wide = sal.graph.control_weight(0);
    let narrow = sal.graph.control_weight(1);
    assert!(
        wide > narrow,
        "outer branch {wide} should outrank inner {narrow}"
    );
    assert!(sal.nodes[0].score > sal.nodes[1].score);
}

/// An accumulator whose value survives the back edge is core; a temporary
/// recomputed from scratch each iteration is not.
#[test]
fn loop_carried_definition_is_core() {
    // sum=0; i=0; while (i<n) { sum=sum+x; i=i+1; } return sum;
    const SUM: u32 = 1;
    const I: u32 = 2;
    const X: u32 = 3;
    let ir = func(
        "accumulate",
        vec![
            Node::pure(2).with_dataflow([SUM], []).with_succs([1]),
            Node::pure(3).with_dataflow([I], []).with_succs([2]),
            Node::pure(4)
                .with_dataflow([], [I])
                .with_kind(NodeKind::Branch)
                .with_succs([3, 5]),
            Node::pure(5).with_dataflow([SUM], [SUM, X]).with_succs([4]),
            Node::pure(6).with_dataflow([I], [I]).with_succs([2]),
            Node::pure(8)
                .with_dataflow([], [SUM])
                .with_kind(NodeKind::Return),
        ],
    );
    let sal = run(&ir);
    let g = &sal.graph;

    assert_eq!(g.loops.len(), 1, "one natural loop expected");
    assert_eq!(g.loops[0].header, 2);
    assert_eq!(
        g.loops[0].body.iter().copied().collect::<Vec<_>>(),
        vec![2, 3, 4]
    );

    assert_eq!(sal.nodes[3].tier, Tier::Core, "sum accumulator");
    assert_eq!(sal.nodes[4].tier, Tier::Core, "induction variable");
    assert!(
        sal.nodes[3]
            .reasons
            .iter()
            .any(|r| matches!(r, salience_core::Reason::LoopCarried { .. })),
        "expected a loop-carried reason, got {:?}",
        sal.nodes[3].reasons
    );
}

/// A logging call is inert, and so is the string building that exists only to
/// feed it — but a value the log shares with real work is not demoted.
#[test]
fn denylisted_calls_and_their_argument_support_are_inert() {
    const MSG: u32 = 1;
    const VALUE: u32 = 2;
    let ir = func(
        "logs",
        vec![
            // value = compute()
            Node::pure(2)
                .with_dataflow([VALUE], [])
                .with_kind(NodeKind::Call {
                    callee: "com.example.Service::compute".into(),
                    opacity: CallOpacity::Opaque,
                })
                .with_succs([1]),
            // msg = "result: " + value      <- exists only for the log call
            Node::pure(3).with_dataflow([MSG], [VALUE]).with_succs([2]),
            // logger.info(msg)
            Node::pure(4)
                .with_dataflow([], [MSG])
                .with_kind(NodeKind::Call {
                    callee: "org.slf4j.Logger::info".into(),
                    opacity: CallOpacity::Opaque,
                })
                .with_succs([3]),
            // return value                  <- keeps `value` behavior-carrying
            Node::pure(5)
                .with_dataflow([], [VALUE])
                .with_kind(NodeKind::Return),
        ],
    );
    let sal = run(&ir);

    assert_eq!(sal.nodes[2].tier, Tier::Inert, "the log call itself");
    assert_eq!(sal.nodes[1].tier, Tier::Inert, "message building for it");
    assert_eq!(sal.nodes[2].score, 0.0);
    assert_ne!(
        sal.nodes[0].tier,
        Tier::Inert,
        "compute() also feeds the return, so it must survive"
    );
}

/// Computation whose result reaches no effect is plumbing, not core. This is
/// the case that separates a dependence-grounded analysis from a syntactic one:
/// the statement looks identical to a core assignment.
#[test]
fn computation_reaching_no_effect_is_plumbing() {
    const DEAD: u32 = 1;
    const LIVE: u32 = 2;
    let ir = func(
        "dead",
        vec![
            Node::pure(2).with_dataflow([DEAD], []).with_succs([1]),
            Node::pure(3).with_dataflow([LIVE], []).with_succs([2]),
            Node::pure(4)
                .with_dataflow([], [LIVE])
                .with_kind(NodeKind::Return),
        ],
    );
    let sal = run(&ir);
    assert_eq!(sal.nodes[0].tier, Tier::Plumbing, "unused definition");
    assert_eq!(sal.nodes[1].tier, Tier::Core, "feeds the return");
}

/// A state write is the frontier, and everything computing the written value is
/// core behind it.
#[test]
fn state_write_is_boundary_with_core_behind_it() {
    const V: u32 = 1;
    let ir = func(
        "store",
        vec![
            Node::pure(2).with_dataflow([V], []).with_succs([1]),
            Node::pure(3)
                .with_dataflow([], [V])
                .with_kind(NodeKind::StateWrite {
                    target: "com.example.Cache#entries".into(),
                })
                .with_succs([2]),
            Node::pure(4).with_kind(NodeKind::Return),
        ],
    );
    let sal = run(&ir);
    assert_eq!(sal.nodes[1].tier, Tier::Boundary);
    assert_eq!(sal.nodes[0].tier, Tier::Core);
}

/// Post-dominance must stay total even when a function never returns normally,
/// which is exactly the shape a `while (true)` service loop compiles to.
#[test]
fn infinite_loop_without_exit_still_analyzes() {
    let ir = func(
        "spin",
        vec![
            Node::pure(2).with_succs([1]),
            Node::pure(3).with_kind(NodeKind::Branch).with_succs([1, 0]),
        ],
    );
    let sal = run(&ir);
    assert_eq!(sal.nodes.len(), 2);
    assert!(!sal.graph.loops.is_empty());
}

/// Several instructions on one line collapse to the most salient tier — and
/// `inert` is the weakest, so a line holding both a log call and real work
/// reads as real work.
#[test]
fn line_projection_takes_the_most_salient_tier() {
    const V: u32 = 1;
    let ir = func(
        "shared_line",
        vec![
            // Both of these lowered from line 7.
            Node::pure(7)
                .with_dataflow([], [])
                .with_kind(NodeKind::Call {
                    callee: "org.slf4j.Logger::debug".into(),
                    opacity: CallOpacity::Opaque,
                })
                .with_succs([1]),
            Node::pure(7).with_dataflow([V], []).with_succs([2]),
            Node::pure(8)
                .with_dataflow([], [V])
                .with_kind(NodeKind::Return),
        ],
    );
    let sal = run(&ir);
    let spans = project_to_lines(&ir, &sal);
    let seven = spans
        .iter()
        .find(|s| s.start <= 7 && 7 <= s.end)
        .expect("line 7 must be mapped");
    assert_eq!(seven.tier, Tier::Core);
}

/// Instructions with no source position participate in the analysis but project
/// onto no line. Synthetic members must not invent spans.
#[test]
fn instructions_without_a_line_project_nowhere() {
    let ir = func(
        "synthetic",
        vec![
            Node {
                line: None,
                ..Node::pure(0)
            }
            .with_succs([1]),
            Node::pure(4).with_kind(NodeKind::Return),
        ],
    );
    let sal = run(&ir);
    let spans = project_to_lines(&ir, &sal);
    assert_eq!(spans.len(), 1);
    assert_eq!(spans[0].start, 4);
}

/// The whole artifact must be byte-identical across runs, or it cannot be
/// cached and cannot be diffed.
#[test]
fn artifact_is_deterministic() {
    const V: u32 = 1;
    let ir = func(
        "determinism",
        vec![
            Node::pure(2).with_dataflow([V], []).with_succs([1]),
            Node::pure(3)
                .with_dataflow([], [V])
                .with_kind(NodeKind::Branch)
                .with_succs([2, 3]),
            Node::pure(4)
                .with_dataflow([], [V])
                .with_kind(NodeKind::Return),
            Node::pure(5).with_kind(NodeKind::Return),
        ],
    );

    let render = || {
        let sal = run(&ir);
        let mut side = Sidecar::new("Demo.java", "test");
        side.push(&ir, &sal);
        side.finish();
        serde_json::to_string_pretty(&side).expect("sidecar serializes")
    };
    assert_eq!(render(), render());
}

/// The query an edit-policy hook makes.
#[test]
fn tier_at_answers_the_hook_question() {
    const V: u32 = 1;
    let ir = func(
        "hooked",
        vec![
            Node::pure(10).with_dataflow([V], []).with_succs([1]),
            Node::pure(11)
                .with_dataflow([], [V])
                .with_kind(NodeKind::Return),
        ],
    );
    let sal = run(&ir);
    let mut side = Sidecar::new("Demo.java", "test");
    side.push(&ir, &sal);
    side.finish();

    assert_eq!(side.tier_at(10), Some("core"));
    assert_eq!(side.tier_at(11), Some("boundary"));
    assert_eq!(side.tier_at(99), None, "unmapped lines are not claimed");
}

/// Malformed frontend output is reported, not tolerated.
#[test]
fn validate_rejects_dangling_edges() {
    let ir = func("bad", vec![Node::pure(1).with_succs([7])]);
    assert!(ir.validate().is_err());
}

/// Dominance and post-dominance on a diamond, checked directly.
#[test]
fn diamond_dominance() {
    let ir = func(
        "diamond",
        vec![
            Node::pure(1).with_kind(NodeKind::Branch).with_succs([1, 2]),
            Node::pure(2).with_succs([3]),
            Node::pure(3).with_succs([3]),
            Node::pure(4).with_kind(NodeKind::Return),
        ],
    );
    ir.validate().unwrap();
    let g = Graph::build(&ir);
    assert!(g.dominates(0, 3));
    assert!(!g.dominates(1, 3), "either arm may be skipped");
    // Both arms are control-dependent on the branch; the join is not.
    assert!(g.ctrl_deps[1].contains(&0));
    assert!(g.ctrl_deps[2].contains(&0));
    assert!(g.ctrl_deps[3].is_empty(), "the join always executes");
}

/// A value the logging merely *shares* with real work must not be demoted.
/// This is the guard against the inert rule being too eager.
#[test]
fn a_value_shared_with_real_work_is_not_inert() {
    const V: u32 = 1;
    let ir = func(
        "shared",
        vec![
            // v = compute()
            Node::pure(2)
                .with_dataflow([V], [])
                .with_kind(NodeKind::Call {
                    callee: "com.example.Service::compute".into(),
                    opacity: CallOpacity::Opaque,
                })
                .with_succs([1]),
            // logger.info(v)
            Node::pure(3)
                .with_dataflow([], [V])
                .with_kind(NodeKind::Call {
                    callee: "org.slf4j.Logger::info".into(),
                    opacity: CallOpacity::Opaque,
                })
                .with_succs([2]),
            // return v
            Node::pure(4)
                .with_dataflow([], [V])
                .with_kind(NodeKind::Return),
        ],
    );
    let sal = run(&ir);
    assert_eq!(sal.nodes[1].tier, Tier::Inert, "the log call itself");
    assert_ne!(
        sal.nodes[0].tier,
        Tier::Inert,
        "compute() also reaches the return and must survive"
    );
}

/// A counter incremented only to be logged is inert even though its definition
/// and its use form a cycle. An "all consumers are inert" fixpoint cannot reach
/// this; backward reachability can, and this is the case that motivates it.
#[test]
fn a_log_only_counter_in_a_dependency_cycle_is_inert() {
    const COUNT: u32 = 1;
    const SUM: u32 = 2;
    const X: u32 = 3;
    let ir = func(
        "counters",
        vec![
            // count = 0
            Node::pure(2).with_dataflow([COUNT], []).with_succs([1]),
            // sum = 0
            Node::pure(3).with_dataflow([SUM], []).with_succs([2]),
            // loop header
            Node::pure(4).with_kind(NodeKind::Branch).with_succs([3, 6]),
            // sum = sum + x      <- real accumulator
            Node::pure(5).with_dataflow([SUM], [SUM, X]).with_succs([4]),
            // count = count + 1  <- feeds only the log, and reads itself
            Node::pure(6)
                .with_dataflow([COUNT], [COUNT])
                .with_succs([5]),
            Node::pure(7).with_succs([2]),
            // logger.debug(count)
            Node::pure(9)
                .with_dataflow([], [COUNT])
                .with_kind(NodeKind::Call {
                    callee: "org.slf4j.Logger::debug".into(),
                    opacity: CallOpacity::Opaque,
                })
                .with_succs([7]),
            // return sum
            Node::pure(10)
                .with_dataflow([], [SUM])
                .with_kind(NodeKind::Return),
        ],
    );
    let sal = run(&ir);
    assert_eq!(
        sal.nodes[4].tier,
        Tier::Inert,
        "count += 1 feeds only the log"
    );
    assert_eq!(sal.nodes[0].tier, Tier::Inert, "count = 0 likewise");
    assert_eq!(
        sal.nodes[3].tier,
        Tier::Core,
        "sum += x reaches the return and stays core"
    );
}

/// An opaque call whose result nobody reads is still a real call. It must not
/// be absorbed just because it feeds nothing.
#[test]
fn an_unused_opaque_call_is_not_inert() {
    let ir = func(
        "fire_and_forget",
        vec![
            Node::pure(2)
                .with_kind(NodeKind::Call {
                    callee: "com.example.Service::sendEmail".into(),
                    opacity: CallOpacity::Opaque,
                })
                .with_succs([1]),
            Node::pure(3).with_kind(NodeKind::Return),
        ],
    );
    let sal = run(&ir);
    assert_eq!(sal.nodes[0].tier, Tier::Boundary);
}

/// The denylist must not fire on names that merely contain a pattern.
#[test]
fn denylist_respects_name_boundaries() {
    let d = Denylist::new();
    assert!(d.matches("print"), "exact bare name");
    assert!(!d.matches("pprint"), "must not match a longer name");
    assert!(!d.matches("footprint"), "must not match a suffix");
    assert!(d.matches("LOG.info"), "case-insensitive");
    assert!(d.matches("java.util.logging.Logger::info"));
    assert!(!d.matches("mylog.info"), "must align to a segment boundary");
    assert!(!d.matches("com.example.Blogger::info"));
}
