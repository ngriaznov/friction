//! Versioned data packs: the downloadable-artifact registry and the
//! per-`(genre, metric)` human envelope bands `friction-rules` gates on.
//! Lexical substitution/filler tables are not a pack — `friction-rules`
//! ships them hand-curated and compiled in; see that crate's docs for why.
//!
//! # Envelope bands
//!
//! [`EnvelopePack`] parses the TOML file `corpus-tool envelope` writes
//! (`packs/envelope-v2.toml`, embedded into this crate and exposed
//! pre-parsed as [`ENVELOPE_V2`]) into a `(genre, metric) -> [lo, hi]`
//! lookup. This is the "packs" a caller building `friction-rules`'
//! `GenreEnvelope` trait for a genre reads from — see
//! `friction-apply::FixEngine` for the adapter that wires the two
//! together.
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
//! # DMS index
//!
//! [`DmsIndex`] parses the token-id-stream pack `corpus-tool index`
//! writes (`packs/dms-index-v1.toml`) into a shared [`Vocab`] plus one
//! suffix automaton ([`Sam`]) per stream — the human corpus and whichever
//! of the four [`ModelFamily`] generator corpora the pack defines. This
//! milestone builds and unit-tests the reconstruction only; nothing wires
//! a [`DmsIndex`] into a running fix-time detection pass yet.
//!
//! # Inventory pack
//!
//! [`InventoryPack`] parses the curated tell-span inventory
//! (`packs/inventory-v1.toml`, embedded and exposed pre-parsed as
//! [`INVENTORY`]) into typed, deterministically-sorted tables:
//! deletion spans, substitution pairs, ritual frames, a (currently empty)
//! preview-frame family, licensed light-verb-construction pairs, guard
//! tokens, a closure function-word allowance, and output-frequency
//! hygiene bands. [`validate`] runs three build-time audits over a parsed
//! pack — disjointness, closure, and output-frequency hygiene — see that
//! function's own docs. [`load_dms_pack`] and [`INVENTORY`] both go
//! through [`LoadedPack`], this crate's one unified pack-registry shape.
//!
//! # Determinism
//!
//! [`REGISTRY`] is a `Vec`, not a hash-based collection, and preserves the
//! declaration order of the embedded `registry.toml` exactly — iterating
//! it gives identical results on every run and every machine. [`Sam`]'s
//! internal `next` table is a `HashMap` for point-lookup performance, not
//! iteration order — see that type's own doc comment for why this does
//! not compromise determinism.

mod artifact;
mod dms;
mod envelope;
mod inventory;
mod registry;
mod validate;

pub use artifact::{Artifact, ArtifactKind, PackError, REGISTRY, Sha256, parse_registry};
pub use dms::{DmsIndex, ModelFamily, Sam, Vocab};
pub use envelope::{ENVELOPE_V2, EnvelopePack, exceedance};
pub use inventory::{
    Anchor, DeletionSpan, FrequencyUnit, GuardTokens, InventoryPack, LvcPair, OutputFrequencyBand,
    PreviewFrame, RepairKind, RitualFrame, SubstitutionPair,
};
pub use registry::{INVENTORY, LoadedPack, load_dms_pack};
pub use validate::{
    ClosureViolation, DisjointnessViolation, FrequencyHygieneReason, FrequencyHygieneViolation,
    Violation, check_closure, check_disjointness, check_frequency_hygiene, validate,
};
