//! CPython bytecode frontend: `.py` source to [`FunctionIr`].
//!
//! The lowering itself runs in Python — see `src/lower.py`, embedded at compile
//! time and executed through the interpreter the caller points at. That split
//! is deliberate rather than expedient: `dis` and `co_lines()` are the only
//! authoritative description of CPython bytecode, they change with every minor
//! release, and reimplementing them in Rust would mean shipping a decoder that
//! silently disagrees with the interpreter it claims to model.
//!
//! What crosses the boundary is JSON in the shape of
//! [`FunctionIr`](salience_core::ir::FunctionIr). That the neutral contract
//! serializes cleanly enough for a frontend in another language to speak it is
//! the point: a substrate whose tooling lives outside Rust — CPython here, but
//! equally a JVM agent, a `rustc` driver, or a compiler plugin — can participate
//! without the core knowing anything about it.
//!
//! # Line fidelity
//!
//! Since PEP 626, every instruction carries a source line and `co_lines()`
//! guarantees the ranges are non-decreasing, consecutive, and cover the whole
//! code object. Python 3.11 added `co_positions()` for column offsets, which
//! this frontend does not yet use — line granularity is what the artifact
//! records.

#![forbid(unsafe_code)]
#![warn(missing_docs, clippy::pedantic)]
// `CPython`, `MokaIR`, `JSR`, `SMAP` and friends are proper nouns in prose,
// not identifiers.
#![allow(clippy::doc_markdown)]

use std::io::Write as _;
use std::path::Path;
use std::process::Command;

use salience_core::ir::{CallOpacity, FunctionId, FunctionIr, Node, NodeKind, VarId};
use serde::Deserialize;

/// The lowering script, embedded so the crate is self-contained.
const LOWER_PY: &str = include_str!("lower.py");

/// Something went wrong invoking or parsing the Python lowering.
#[derive(Debug, thiserror::Error)]
pub enum PyError {
    /// The interpreter could not be started, or the temp file could not be
    /// written.
    #[error("could not run the Python lowering: {0}")]
    Spawn(#[from] std::io::Error),
    /// The interpreter exited non-zero.
    #[error("python exited with status {status}: {stderr}")]
    Failed {
        /// Exit status, rendered.
        status: String,
        /// Whatever the interpreter wrote to stderr, usually a `SyntaxError`.
        stderr: String,
    },
    /// The interpreter produced output this crate could not read.
    #[error("could not parse the lowering output: {0}")]
    Decode(#[from] serde_json::Error),
}

// --- wire format -------------------------------------------------------
// Mirrors `FunctionIr` exactly. Kept separate from the core types rather than
// deriving `Deserialize` on them, so the core stays free of a serialization
// contract it does not otherwise need.

#[derive(Debug, Deserialize)]
struct WireModule {
    file: String,
    functions: Vec<WireFunction>,
}

#[derive(Debug, Deserialize)]
struct WireFunction {
    name: String,
    signature: String,
    decl_line: Option<u32>,
    entry: usize,
    nodes: Vec<WireNode>,
}

#[derive(Debug, Deserialize)]
struct WireNode {
    line: Option<u32>,
    kind: WireKind,
    defs: Vec<VarId>,
    uses: Vec<VarId>,
    succs: Vec<usize>,
    label: String,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "t", rename_all = "snake_case")]
enum WireKind {
    Pure,
    Branch,
    Return,
    Throw,
    Call {
        callee: String,
        #[serde(default)]
        opacity: String,
    },
    StateWrite {
        target: String,
    },
}

impl From<WireKind> for NodeKind {
    fn from(w: WireKind) -> Self {
        match w {
            WireKind::Pure => Self::Pure,
            WireKind::Branch => Self::Branch,
            WireKind::Return => Self::Return,
            WireKind::Throw => Self::Throw,
            WireKind::Call { callee, opacity } => Self::Call {
                callee,
                opacity: if opacity == "inert" {
                    CallOpacity::Inert
                } else {
                    CallOpacity::Opaque
                },
            },
            WireKind::StateWrite { target } => Self::StateWrite { target },
        }
    }
}

/// A lowered Python module.
#[derive(Debug, Clone)]
pub struct LoweredModule {
    /// The source file analyzed.
    pub file: String,
    /// One entry per code object: module level, each function, each
    /// comprehension and each nested closure.
    pub functions: Vec<FunctionIr>,
}

/// Lowers a Python source file using `python3` from `PATH`.
///
/// # Errors
///
/// See [`PyError`]. A syntax error in the target file surfaces as
/// [`PyError::Failed`] carrying the interpreter's own message.
pub fn lower_file(path: &Path) -> Result<LoweredModule, PyError> {
    lower_file_with(path, "python3")
}

/// Lowers a Python source file using a named interpreter.
///
/// Useful when the code under analysis targets a different minor version than
/// the ambient one — bytecode is version-specific, so lowering 3.12 source with
/// a 3.9 interpreter would describe a program that never runs.
///
/// # Errors
///
/// See [`PyError`].
pub fn lower_file_with(path: &Path, interpreter: &str) -> Result<LoweredModule, PyError> {
    // The script goes to a temp file rather than `python3 -c` so that tracebacks
    // carry real line numbers when it is being worked on.
    let mut script = std::env::temp_dir();
    script.push(format!("salience-lower-{}.py", std::process::id()));
    {
        let mut fh = std::fs::File::create(&script)?;
        fh.write_all(LOWER_PY.as_bytes())?;
    }

    let output = Command::new(interpreter).arg(&script).arg(path).output();
    let _ = std::fs::remove_file(&script);
    let output = output?;

    if !output.status.success() {
        return Err(PyError::Failed {
            status: output.status.to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        });
    }

    let module: WireModule = serde_json::from_slice(&output.stdout)?;
    let file = module.file;
    let functions = module
        .functions
        .into_iter()
        .map(|f| FunctionIr {
            id: FunctionId {
                file: file.clone(),
                name: f.name,
                signature: f.signature,
                decl_line: f.decl_line,
            },
            nodes: f
                .nodes
                .into_iter()
                .map(|n| Node {
                    line: n.line,
                    kind: n.kind.into(),
                    defs: n.defs,
                    uses: n.uses,
                    succs: n.succs,
                    label: n.label,
                })
                .collect(),
            entry: f.entry,
        })
        .collect();

    Ok(LoweredModule { file, functions })
}
