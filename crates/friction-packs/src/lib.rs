//! Versioned data packs: the downloadable-artifact registry, plus (once
//! other crates in this workspace populate them) envelope bands and
//! lexical substitution tables.
//!
//! # Artifact registry
//!
//! [`REGISTRY`] is the built-in list of downloadable, sha256-pinned NLP
//! artifacts that `friction setup` fetches into a local runtime cache
//! directory. Each entry names a stable version, a source URL, an expected
//! size, and an expected sha256 checksum; nothing is fetched by this
//! crate itself — `friction-cli`'s `setup` subcommand is responsible for
//! downloading and verifying entries into a local cache, and for failing
//! hard on a checksum mismatch rather than ever trusting unverified bytes.
//!
//! [`REGISTRY`] is currently empty, and correctly so: every NLP asset
//! `friction-nlp` uses today is either vendored directly into the repo
//! (its sentence-segmentation ruleset) or downloaded, verified, and
//! embedded at *build* time by that crate's own `build.rs` (its
//! part-of-speech tagger model) — neither needs a runtime cache directory
//! to exist for a compiled `friction` binary to work, so listing them
//! here would describe a fetch path nothing actually takes. No
//! dependency-parser model is registered for the same reason it isn't
//! used: no downloadable ONNX English dependency parser under roughly
//! 100 MB has been located as of this writing (see `friction-nlp`'s
//! crate docs for what that means for `DepParser`). An entry can be added
//! here the moment one is sourced and independently verified, without
//! changing how `friction setup` or any consumer of [`REGISTRY`] works —
//! see `registry.toml`'s own header comment for the full accounting.
//! `friction setup` treats an empty registry as a normal, successful
//! outcome ("nothing to download") rather than an error.
//!
//! # Determinism
//!
//! [`REGISTRY`] is a `Vec`, not a hash-based collection, and preserves the
//! declaration order of the embedded `registry.toml` exactly — iterating
//! it gives identical results on every run and every machine.

mod artifact;

pub use artifact::{Artifact, ArtifactKind, PackError, REGISTRY, Sha256, parse_registry};
