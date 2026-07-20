//! `friction` command-line interface.
//!
//! Provides `friction setup` (downloads and sha256-verifies the pinned NLP
//! artifact registry into a local cache), `friction check` (detect-only:
//! metrics, envelope bands, and rule findings), `friction fix` (runs the
//! full fixpoint engine and writes the fixed text out), and `friction
//! explain` (runs the same fixpoint engine but reports a before/after
//! metric comparison and the executed round schedule instead of the fixed
//! text).
//!
//! Every subcommand but `setup` is offline and deterministic: no network
//! access, no wall-clock timestamps, and no absolute filesystem paths in
//! any output — see each subcommand's own module docs for its exact
//! output contract.

mod check;
mod common;
mod diagnostics;
mod explain;
mod fix;
mod sarif;
mod scan;
mod setup;
mod table;

use clap::{Parser, Subcommand};

/// `friction`: a deterministic engine that reduces LLM-speak in prose.
#[derive(Debug, Parser)]
#[command(name = "friction", version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Downloads and sha256-verifies the pinned NLP artifact registry into
    /// a local cache directory.
    Setup(setup::SetupArgs),
    /// Parses, measures, and scans for rule findings, applying no fixes.
    Check(check::CheckArgs),
    /// Runs the full fixpoint engine and writes the fixed text out.
    Fix(fix::FixArgs),
    /// Runs the fixpoint engine and reports a before/after metric
    /// comparison and the executed round schedule, without emitting the
    /// fixed text.
    Explain(explain::ExplainArgs),
}

fn main() -> std::process::ExitCode {
    let cli = Cli::parse();
    match cli.command {
        Command::Setup(args) => setup::run(&args),
        Command::Check(args) => check::run(&args),
        Command::Fix(args) => fix::run(&args),
        Command::Explain(args) => explain::run(&args),
    }
}
