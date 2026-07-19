//! CLI subcommands.
//!
//! Each subcommand is its own module exposing an `Args` (clap derive) and
//! a `run`. Every subcommand but `generate` returns
//! `anyhow::Result<()>`; `generate` returns
//! `anyhow::Result<generate::GenerateOutcome>` so the CLI dispatcher can
//! choose a distinct exit code when jobs were skipped for a missing
//! model, without this module ever calling `std::process::exit` itself
//! (which would make it untestable in-process).

pub mod clean;
pub mod generate;
pub mod holdout_check;
pub mod ingest;
pub mod remove;
pub mod seal;
pub mod split;
pub mod stats;
pub mod validate;
