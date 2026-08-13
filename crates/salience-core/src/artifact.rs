//! The sidecar artifact: line ranges to tier, with a reason for each.
//!
//! This is the whole deliverable. Everything upstream exists to produce it, and
//! every consumer — an agent reading a file, a harness hook gating an edit, a
//! profiler picking a starting point, a dependency graph weighting its nodes —
//! reads this and nothing else.
//!
//! Two design commitments show up in the schema:
//!
//! - **Nothing is removed.** The artifact is metadata *about* source, never a
//!   rewritten copy of it. A consumer that ignores the sidecar sees exactly the
//!   file it would have seen anyway.
//! - **Every span carries its reason.** A tier a consumer cannot interrogate is
//!   a number to be trusted on faith, which is precisely what an inference-free
//!   analysis exists to avoid.

use serde::{Deserialize, Serialize};

use crate::ir::FunctionIr;
use crate::salience::{FunctionSalience, LineSpan, Tier, project_to_lines};

/// Schema identifier, bumped on any breaking change to the shape below.
pub const SCHEMA: &str = "salience-sidecar/v1";

/// One tiered run of source lines.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpanRecord {
    /// First line, inclusive, 1-based.
    pub start: u32,
    /// Last line, inclusive, 1-based.
    pub end: u32,
    /// `core` / `boundary` / `plumbing` / `inert`.
    pub tier: String,
    /// Continuous salience for ranking, `0.0..=1.0`.
    pub score: f64,
    /// Why this run got its tier.
    pub reasons: Vec<String>,
}

impl SpanRecord {
    fn from_span(s: &LineSpan) -> Self {
        Self {
            start: s.start,
            end: s.end,
            tier: s.tier.name().to_owned(),
            // Two decimals: the score is a ranking signal, and pretending to
            // more precision than the weights justify invites false diffs
            // between runs on trivially different inputs.
            score: (s.score * 100.0).round() / 100.0,
            reasons: s.reasons.clone(),
        }
    }
}

/// How much of the body carried source positions at all.
///
/// This is the honesty field. On substrates where line mapping is partial —
/// synthetic members, compiler-generated bridges, optimized-away constructs —
/// a consumer needs to know it is looking at an incomplete picture rather than
/// at a body that genuinely has no core.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Coverage {
    /// Instructions lowered from this function.
    pub instructions: usize,
    /// How many carried a source line.
    pub with_line: usize,
}

/// Counts per tier, for quick filtering without walking every span.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TierCounts {
    /// Lines tiered `core`.
    pub core: usize,
    /// Lines tiered `boundary`.
    pub boundary: usize,
    /// Lines tiered `plumbing`.
    pub plumbing: usize,
    /// Lines tiered `inert`.
    pub inert: usize,
}

/// The salience map for one function.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionRecord {
    /// Fully-qualified name.
    pub name: String,
    /// Descriptor or signature, when the substrate has one.
    #[serde(skip_serializing_if = "String::is_empty", default)]
    pub signature: String,
    /// First line of the body, when known.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub decl_line: Option<u32>,
    /// Position-mapping coverage.
    pub coverage: Coverage,
    /// Line counts per tier.
    pub summary: TierCounts,
    /// The map itself, ordered by start line.
    pub spans: Vec<SpanRecord>,
}

/// The sidecar for one source file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Sidecar {
    /// Schema identifier.
    pub schema: String,
    /// What produced this, including the frontend name.
    pub generator: String,
    /// Source file the spans refer to.
    pub file: String,
    /// One entry per analyzed function, ordered by declaration line then name.
    pub functions: Vec<FunctionRecord>,
}

impl Sidecar {
    /// Creates an empty sidecar for `file`.
    #[must_use]
    pub fn new(file: impl Into<String>, generator: impl Into<String>) -> Self {
        Self {
            schema: SCHEMA.to_owned(),
            generator: generator.into(),
            file: file.into(),
            functions: Vec::new(),
        }
    }

    /// Appends the analysis of one function.
    pub fn push(&mut self, ir: &FunctionIr, sal: &FunctionSalience) {
        let spans = project_to_lines(ir, sal);
        let mut summary = TierCounts::default();
        for s in &spans {
            let lines = (s.end - s.start + 1) as usize;
            match s.tier {
                Tier::Core => summary.core += lines,
                Tier::Boundary => summary.boundary += lines,
                Tier::Plumbing => summary.plumbing += lines,
                Tier::Inert => summary.inert += lines,
            }
        }
        self.functions.push(FunctionRecord {
            name: ir.id.name.clone(),
            signature: ir.id.signature.clone(),
            decl_line: ir.id.decl_line,
            coverage: Coverage {
                instructions: ir.nodes.len(),
                with_line: ir.nodes.iter().filter(|n| n.line.is_some()).count(),
            },
            summary,
            spans: spans.iter().map(SpanRecord::from_span).collect(),
        });
    }

    /// Sorts functions into a stable order, so that two runs over the same
    /// input produce byte-identical JSON regardless of the order a frontend
    /// happened to walk the class or module.
    pub fn finish(&mut self) {
        self.functions.sort_by(|a, b| {
            a.decl_line
                .cmp(&b.decl_line)
                .then_with(|| a.name.cmp(&b.name))
        });
    }

    /// The tier covering `line`, if any function's map claims it.
    ///
    /// This is the query an edit-policy hook makes: given a line the agent
    /// wants to touch, how sensitive is it? Returns the most salient tier when
    /// nested functions overlap.
    #[must_use]
    pub fn tier_at(&self, line: u32) -> Option<&str> {
        let mut best: Option<(Tier, &str)> = None;
        for f in &self.functions {
            for s in &f.spans {
                if line >= s.start && line <= s.end {
                    let t = match s.tier.as_str() {
                        "core" => Tier::Core,
                        "boundary" => Tier::Boundary,
                        "plumbing" => Tier::Plumbing,
                        _ => Tier::Inert,
                    };
                    if best.is_none_or(|(bt, _)| t > bt) {
                        best = Some((t, s.tier.as_str()));
                    }
                }
            }
        }
        best.map(|(_, name)| name)
    }
}
