//! Versioned data packs: the curated tell inventory, the per-family DMS
//! index streams, the human-corpus attestation data, and the
//! per-`(genre, metric)` human envelope bands.
//!
//! # Envelope bands
//!
//! [`EnvelopePack`] parses the TOML file `corpus-tool envelope` writes
//! (`packs/envelope-v2.toml`, embedded into this crate and exposed
//! pre-parsed as [`ENVELOPE_V2`]) into a `(genre, metric) -> [lo, hi]`
//! lookup, used by `friction check`'s metric guardrails.
//!
//! # Offline by construction
//!
//! Every asset this workspace's NLP layer uses is vendored directly into
//! the repo (the sentence-segmentation ruleset, the part-of-speech
//! tagger's weight artifact) and every pack in this crate is embedded
//! (`include_str!` for the TOML packs, `include_bytes!` for the
//! derived binary artifacts) — nothing is fetched at build time or at
//! run time, and no runtime cache directory needs to exist for a
//! compiled `friction` binary to work.
//!
//! # DMS index
//!
//! [`DmsIndex`] parses the token-id-stream pack `corpus-tool index`
//! writes (`packs/dms-index-v1.toml`) into a shared [`Vocab`] plus one
//! suffix automaton ([`Sam`]) per stream — the human corpus, whichever of
//! the five [`ModelFamily`] generator corpora the pack defines, and one
//! more: [`DmsIndex::pooled_machine_sam`], every present family stream
//! concatenated in a fixed order. That last one is what a fix-time
//! detection pass actually scans against (`friction-match`'s `dms`
//! module) — family attribution is not a product surface, only "does
//! this read machine-generated" is, so the per-family automata exist for
//! artifact compatibility, not for the runtime walk. That TOML path is
//! the audited source, used offline; the runtime pass reads [`DMS`]
//! instead — a zero-copy [`DmsIndexView`] over the derived artifact
//! `corpus-tool dms-pack` serializes the finished automata into
//! (`packs/dms-index-v1.bin`, embedded), so process start pays no TOML
//! parse or automaton construction. See `dms_bin`'s module docs for the
//! layout and the drift guard keeping the two representations in
//! lockstep.
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
//! # Attestation pack
//!
//! [`AttestationPack`] parses the seam-bigram and POS-skeleton mining
//! output `corpus-tool attest` writes (`packs/attestation-v1.toml`,
//! embedded and exposed pre-parsed as [`ATTESTATION`]) into a
//! [`BigramTable`] (has this exact word pair been observed adjacent to
//! each other in the TRAIN human corpus) and a [`SkeletonSet`] (has this
//! exact run of coarse part-of-speech tags been observed as a TRAIN human
//! sentence's skeleton), plus an optional [`NearNoOpCalibration`] filled
//! in by a later calibration stage. See that module's own doc comment
//! for why the two tables are built from — and queried through — two
//! independent tokenizations rather than one shared, positionally-paired
//! stream.
//!
//! # Register pack
//!
//! [`RegisterPack`] parses the two-feature human rate-band pack
//! (`packs/register-v1.toml`, embedded and exposed pre-parsed as
//! [`REGISTER`]) `friction-edit`'s register module homes toward: a
//! `[low, high]` band per feature (nominalization, agentless passive),
//! measured per 1000 prose words from the train-split `docs` genre. See
//! that module's own doc comment for why membership in the band, not
//! distance from its `median`, is the termination condition.
//!
//! # Jargon pack
//!
//! [`JargonPack`] parses the curated metaphor-lexeme pack
//! (`packs/jargon-v1.toml`, embedded and exposed pre-parsed as
//! [`JARGON`]) `friction-match`'s `jargon.metaphor` channel flags against:
//! a lexeme list (each with `source`/`notes` provenance) and a small
//! attested-exceptions allowlist of established compounds never flagged.
//! Detection-only — see that module's own doc comment.
//!
//! # Jargon attestation pack
//!
//! [`jargon_attest`] parses the web-scale compound attestation pack
//! (`packs/jargon-attest-v1.bin` + its `.toml` sidecar, embedded and
//! exposed pre-parsed as [`JARGON_ATTEST`]) into a [`JargonAttestPack`]: a
//! `BinaryFuse8` filter over ~2M normalized Wikipedia-title/OpenAlex-topic
//! compound keys, replacing the bottleneck of a hand-curated exception
//! list with a real attestation oracle. See that module's own doc
//! comment for the normalization/hash sharing discipline and the
//! determinism guarantee `corpus-tool jargon-attest` relies on.
//!
//! # Human evidence pack
//!
//! [`human_evidence`] parses the external human-corpus evidence pack
//! (`packs/human-evidence-v1.bin`, embedded and exposed pre-parsed as
//! [`HUMAN_EVIDENCE`]) `corpus-tool human-evidence` mines from
//! locally-staged external human corpora: unigram and literal-probe
//! occurrence counts that pool into [`frame_compile::CorpusEvidence`],
//! strengthening (never replacing) the `dms-index-v1` human signal the
//! frame-rewrite compile fences already consult. See that module's own
//! doc comment for the two tables' differing absence semantics and why
//! the shipped pack starts empty.
//!
//! # Determinism
//!
//! Every pack parses into sorted or declaration-ordered collections, so
//! iterating any of them gives identical results on every run and every
//! machine. [`Sam`]'s internal `next` table is a `HashMap` for
//! point-lookup performance, not iteration order — see that type's own
//! doc comment for why this does not compromise determinism.

mod artifact;
mod attestation;
mod dms;
mod dms_bin;
mod envelope;
pub mod frame_bin;
pub mod frame_compile;
pub mod frame_rules;
pub mod human_evidence;
mod inventory;
mod jargon;
pub mod jargon_attest;
mod register;
mod registry;
mod validate;

pub use artifact::{PackError, Sha256};
pub use attestation::{AttestationPack, BigramTable, NearNoOpCalibration, SkeletonSet};
pub use dms::{DmsIndex, ModelFamily, Sam, Vocab};
pub use dms_bin::{DmsBinError, DmsIndexView, SamView, VocabView, pack_dms_index_bin};
pub use envelope::{ENVELOPE_V2, EnvelopePack, exceedance};
pub use human_evidence::{HumanEvidencePack, build_pack_bytes as build_human_evidence_bin};
pub use inventory::{
    Anchor, DeletionSpan, FrequencyUnit, GuardTokens, InventoryPack, LvcPair, OutputFrequencyBand,
    PreviewFrame, RepairKind, RitualFrame, SubstitutionPair,
};
pub use jargon::{AttestedException, JargonPack, Lexeme, LexemeSource};
pub use jargon_attest::{BuiltPack, JargonAttestPack, build_pack_bytes, normalize_compound};
pub use register::{RegisterBand, RegisterPack};
pub use registry::{
    ATTESTATION, DMS, FRAME, HUMAN_EVIDENCE, INVENTORY, JARGON, JARGON_ATTEST, LoadedPack,
    MACHINE_EVIDENCE, REGISTER, load_dms_pack,
};
pub use validate::{
    ClosureViolation, DisjointnessViolation, FrequencyHygieneReason, FrequencyHygieneViolation,
    Violation, check_closure, check_disjointness, check_frequency_hygiene, validate,
};
