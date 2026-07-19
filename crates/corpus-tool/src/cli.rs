//! Top-level CLI definition and dispatch.

use clap::{Parser, Subcommand};

use crate::commands;

/// `corpus-tool`: manage the friction validation corpus.
#[derive(Debug, Parser)]
#[command(
    name = "corpus-tool",
    version,
    about = "Manage the friction validation corpus"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

/// Subcommands. Each variant delegates to a same-named module under
/// `crate::commands`, one module per subcommand, so a new subcommand slots
/// in without touching the others.
#[derive(Debug, Subcommand)]
enum Command {
    /// Validate the manifest and corpus files.
    Validate(commands::validate::Args),
    /// Print per-`(class, genre)` corpus statistics.
    Stats(commands::stats::Args),
    /// Compute the deterministic stratified train/dev/holdout split.
    Split(commands::split::Args),
    /// Freeze the holdout split into `<corpus_dir>/holdout.lock`.
    Seal(commands::seal::Args),
    /// Verify `<corpus_dir>/holdout.lock` against the manifest and files.
    HoldoutCheck(commands::holdout_check::Args),
    /// Clean an incoming raw-doc directory into the corpus layout.
    Clean(commands::clean::Args),
    /// Ingest incoming human-corpus docs + metadata into the manifest.
    Ingest(commands::ingest::Args),
    /// Remove one or more docs: drops the manifest record and corpus
    /// file, leaving the raw original under `corpus/incoming/` in place.
    Remove(commands::remove::Args),
    /// Generate the LLM corpus via Ollama.
    Generate(commands::generate::Args),
}

/// Parses process arguments and runs the selected subcommand.
///
/// # Errors
///
/// Returns an error if the selected subcommand fails; the caller (`main`)
/// should treat this as a non-zero exit. See each `commands::*::run` for
/// what specifically can fail.
///
/// `generate` is the one subcommand that can also make the *process*
/// exit non-zero on success: if any job was skipped because its model
/// wasn't available in Ollama, this calls
/// `std::process::exit(commands::generate::EXIT_CODE_MODELS_SKIPPED)`
/// after printing the summary, rather than returning `Ok(())` — every
/// other subcommand's success is a plain `Ok(())`.
pub fn run() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Validate(args) => commands::validate::run(&args),
        Command::Stats(args) => commands::stats::run(&args),
        Command::Split(args) => commands::split::run(&args),
        Command::Seal(args) => commands::seal::run(&args),
        Command::HoldoutCheck(args) => commands::holdout_check::run(&args),
        Command::Clean(args) => commands::clean::run(&args),
        Command::Ingest(args) => commands::ingest::run(&args),
        Command::Remove(args) => commands::remove::run(&args),
        Command::Generate(args) => {
            let outcome = commands::generate::run(&args)?;
            if outcome.any_models_skipped() {
                std::process::exit(commands::generate::EXIT_CODE_MODELS_SKIPPED);
            }
            Ok(())
        }
    }
}
