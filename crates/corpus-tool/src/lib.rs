//! `corpus-tool`: development-only corpus management CLI (dev tool, not
//! shipped). Provides the corpus schema, the deterministic stratified
//! split, the cleaning pipeline, LLM corpus generation, and the
//! `validate`/`stats`/`seal`/`holdout-check` acceptance checks.
//!
//! See `README.md` in this crate for the corpus directory layout and the
//! manifest schema.

pub mod cli;
pub mod commands;
pub mod corpus_layout;
pub mod genconfig;
pub mod hashing;
pub mod manifest;
pub mod ollama;
pub mod prompts;
