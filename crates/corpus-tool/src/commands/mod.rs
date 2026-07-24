//! CLI subcommands.
//!
//! Each subcommand is its own module exposing an `Args` (clap derive) and
//! a `run`. Every subcommand but `generate` returns
//! `anyhow::Result<()>`; `generate` returns
//! `anyhow::Result<generate::GenerateOutcome>` so the CLI dispatcher can
//! choose a distinct exit code when jobs were skipped for a missing
//! model, without this module ever calling `std::process::exit` itself
//! (which would make it untestable in-process).

pub mod attest;
pub mod clean;
pub mod envelope;
pub mod fix_entities;
pub mod generate;
pub mod generate_paired;
pub mod holdout_check;
pub mod index;
pub mod ingest;
pub mod mine;
pub mod mine_inventory;
pub mod mine_paired;
pub(crate) mod ngram_mining;
pub mod output_bands;
pub mod pack_check;
pub mod remove;
pub mod seal;
pub mod separate;
pub mod separate_holdout;
pub mod split;
pub mod stats;
pub mod validate;
