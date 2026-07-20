//! `friction` command-line interface.
//!
//! Provides `friction setup`, which downloads and sha256-verifies the
//! pinned NLP artifact registry (`friction-packs`) into a local cache.
//!
//! This crate is still a scaffold for its other planned subcommands
//! (`check`, `fix`, `fix --suggest`, `explain`) — only `setup` is wired up
//! today.

mod setup;

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
}

fn main() -> std::process::ExitCode {
    let cli = Cli::parse();
    match cli.command {
        Command::Setup(args) => setup::run(&args),
    }
}
