//! `friction` command-line interface.
//!
//! Provides `friction check` (detect-only: metrics, envelope bands, and
//! fix-time detection spans), `friction fix` (runs the four-operation
//! repair engine and writes the fixed text out), and `friction explain`
//! (runs the same repair engine but reports what fired and what was held,
//! per pass, instead of the fixed text).
//!
//! Every subcommand is offline and deterministic: no network access, no
//! wall-clock timestamps, and no absolute filesystem paths in any output —
//! see each subcommand's own module docs for its exact output contract.

mod check;
mod common;
mod diagnostics;
mod explain;
mod fix;
mod sarif;
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
    /// Parses, measures, and runs fix-time detection, applying no fixes.
    Check(check::CheckArgs),
    /// Runs the four-operation repair engine and writes the fixed text
    /// out.
    Fix(fix::FixArgs),
    /// Runs the repair engine and reports what fired and what was held,
    /// per pass, without emitting the fixed text.
    Explain(explain::ExplainArgs),
}

fn main() -> std::process::ExitCode {
    let cli = Cli::parse();
    match cli.command {
        Command::Check(args) => check::run(&args),
        Command::Fix(args) => fix::run(&args),
        Command::Explain(args) => explain::run(&args),
    }
}
