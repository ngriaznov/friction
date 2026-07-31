//! [`PackError`] (the error every parser here reports) and [`Sha256`]
//! (the pinned checksum the pack registry verifies pack bytes against).

use std::fmt;

use sha2::{Digest, Sha256 as Sha2Hasher};

/// Errors produced while parsing or validating a pack.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum PackError {
    /// TOML failed to parse.
    #[error("pack is not valid TOML: {0}")]
    Toml(#[from] Box<toml::de::Error>),

    /// A `sha256` field isn't 64 well-formed hex digits (either case).
    #[error("{value:?} is not a valid sha256 checksum (expected 64 hex digits)")]
    InvalidChecksum {
        /// The offending value.
        value: String,
    },

    /// An envelope pack's `[<genre>.<metric>]` band has `lo > hi` or is
    /// non-finite (see [`crate::EnvelopePack::parse`]).
    #[error("{genre}.{metric}: invalid band [lo={lo}, hi={hi}] (must be finite with lo <= hi)")]
    InvalidBand {
        genre: String,
        metric: String,
        /// The offending lower bound.
        lo: f64,
        /// The offending upper bound.
        hi: f64,
    },

    /// A DMS index pack's `[streams.<name>].ids` isn't a comma-separated
    /// list of `u32` values (see [`crate::DmsIndex::parse`]).
    #[error("streams.{stream}.ids: {value:?} is not a valid u32 token id")]
    DmsMalformedIds {
        /// Which stream's `ids` was malformed.
        stream: String,
        /// The offending comma-separated field.
        value: String,
    },

    /// A DMS index stream references a token id past its own `[vocab]`
    /// table's end (see [`crate::DmsIndex::parse`]).
    #[error(
        "streams.{stream}.ids: token id {id} is out of range for a vocab of {vocab_len} token(s)"
    )]
    DmsIdOutOfRange {
        /// Which stream referenced the out-of-range id.
        stream: String,
        /// The offending token id.
        id: u32,
        vocab_len: usize,
    },

    /// An inventory entry's `pattern` field isn't a valid regex (see
    /// [`crate::InventoryPack::parse`]).
    #[error("{id}: pattern {pattern:?} does not compile: {message}")]
    InventoryInvalidPattern {
        id: String,
        /// The offending pattern source.
        pattern: String,
        /// The underlying regex compile error message.
        message: String,
    },

    /// A `deletion_spans` entry's `anchor` isn't one of `"sentence_initial"`,
    /// `"mid_sentence"`, or `"trailing"`.
    #[error("{id}: {value:?} is not a known anchor")]
    InventoryUnknownAnchor {
        id: String,
        /// The offending value.
        value: String,
    },

    /// A `deletion_spans`/`ritual_frames`/`preview_frames` entry's `repair`
    /// isn't `"delete"` or `"diagnostic_only"`.
    #[error("{id}: {value:?} is not a known repair kind for this family")]
    InventoryUnknownRepairKind {
        id: String,
        /// The offending value.
        value: String,
    },

    /// A `substitution_pairs` entry's `repair` isn't literally `"substitute"`
    /// — by construction it can never be `diagnostic_only`.
    #[error("{id}: {family}'s repair field must be \"substitute\", found {found:?}")]
    InventoryInvalidRepairForFamily {
        id: String,
        /// The family name (`"substitution_pairs"`).
        family: &'static str,
        /// The offending value.
        found: String,
    },

    /// An `output_frequency_bands` entry's `unit` field isn't
    /// `"per_thousand_words"`.
    #[error("{frame}: {value:?} is not a known frequency unit")]
    InventoryUnknownFrequencyUnit {
        /// The band's frame text.
        frame: String,
        /// The offending value.
        value: String,
    },

    /// The same `id` is reused across
    /// `deletion_spans`/`substitution_pairs`/`ritual_frames`/`preview_frames`.
    #[error("duplicate id across inventory families: {id:?}")]
    InventoryDuplicateId {
        /// The duplicated id.
        id: String,
    },

    /// The same `nominalization` is reused across `lvc_pairs` entries.
    #[error("duplicate lvc_pairs nominalization: {nominalization:?}")]
    InventoryDuplicateLvcNominalization {
        /// The duplicated nominalization.
        nominalization: String,
    },

    /// An `lvc_pairs` entry's `(nominalization, derived_verb)` doesn't
    /// resolve in `friction_nlp::lvc::DERIVATIONAL_LEXICON`.
    #[error(
        "lvc_pairs: ({nominalization:?}, {derived_verb:?}) does not resolve in \
         friction_nlp::lvc::DERIVATIONAL_LEXICON"
    )]
    InventoryUnresolvableLvcPair {
        nominalization: String,
        derived_verb: String,
    },

    /// An `lvc_pairs` entry's `light_verb_base` isn't a base-form key in
    /// `friction_nlp::lvc::LIGHT_VERBS`.
    #[error("lvc_pairs.{nominalization}: {light_verb_base:?} is not a known base-form light verb")]
    InventoryUnknownLightVerbBase {
        nominalization: String,
        /// The offending light-verb-base value.
        light_verb_base: String,
    },

    /// An attestation pack's `[bigram].lefts`/`[bigram].rights` has a
    /// non-`u32` value (see [`crate::AttestationPack::parse`]).
    #[error("attestation {field}: {value:?} is not a valid u32 token id")]
    AttestationMalformedId {
        /// Which field was malformed (`"bigram.lefts"` or
        /// `"bigram.rights"`).
        field: &'static str,
        /// The offending value.
        value: String,
    },

    /// An attestation pack's `[bigram]` table references a token id past
    /// its own `[vocab]` table's end.
    #[error(
        "attestation {field}: token id {id} is out of range for a vocab of {vocab_len} token(s)"
    )]
    AttestationIdOutOfRange {
        /// Which field referenced the out-of-range id.
        field: &'static str,
        /// The offending token id.
        id: u32,
        vocab_len: usize,
    },

    /// An attestation pack's `[bigram].lefts` and `[bigram].rights` don't
    /// name the same number of groups.
    #[error("attestation bigram: {lefts} left id(s) but {rights} right-group(s) (must match 1:1)")]
    AttestationBigramShapeMismatch {
        /// The number of ids in `lefts`.
        lefts: usize,
        /// The number of `;`-delimited groups in `rights`.
        rights: usize,
    },

    /// An attestation pack's `[skeleton].tag5`/`tag4` field has a
    /// non-`u64` value.
    #[error("attestation skeleton: {value:?} is not a valid u64 packed tag code")]
    AttestationMalformedSkeletonCode {
        /// The offending value.
        value: String,
    },

    /// A register pack's `[features.<name>]` band doesn't satisfy `low <
    /// median < high` with every bound finite (see
    /// [`crate::RegisterPack::parse`]).
    #[error(
        "features.{feature}: invalid band [low={low}, median={median}, high={high}] \
         (must be finite with low < median < high)"
    )]
    InvalidRegisterBand {
        /// The band's feature name.
        feature: String,
        /// The offending lower bound.
        low: f64,
        /// The offending median.
        median: f64,
        /// The offending upper bound.
        high: f64,
    },

    /// A `jargon-v1` `lexemes` entry's `source` isn't `"mined"` or
    /// `"external"` (see [`crate::JargonPack::parse`]).
    #[error("lexemes.{lexeme}: {value:?} is not a known jargon lexeme source")]
    JargonUnknownSource {
        /// The offending entry's `lexeme`.
        lexeme: String,
        /// The offending value.
        value: String,
    },

    /// The same `lexeme` (or the same `plural`) appears twice in
    /// `jargon-v1`'s `lexemes`.
    #[error("duplicate jargon-v1 lexeme: {lexeme:?}")]
    JargonDuplicateLexeme {
        /// The duplicated lexeme text.
        lexeme: String,
    },

    /// The same `compound` appears twice in `jargon-v1`'s
    /// `attested_exceptions`.
    #[error("duplicate jargon-v1 attested exception: {compound:?}")]
    JargonDuplicateException {
        /// The duplicated compound text.
        compound: String,
    },

    /// A `jargon-v1` `lexemes`/`attested_exceptions` entry's `notes` is
    /// empty or whitespace-only — provenance is mandatory for every entry
    /// in this pack (see [`crate::JargonPack::parse`]).
    #[error("{entry:?}: notes must be non-empty (provenance is mandatory for jargon-v1)")]
    JargonMissingNotes {
        /// The entry's own key (`lexeme` or `compound`).
        entry: String,
    },

    /// `xorf::BinaryFuse8::try_from` exhausted its bounded retry loop
    /// (see [`crate::jargon_attest::build_pack_bytes`]) — in measured
    /// practice this needs a pathological key set (near-total hash
    /// collision), not simply a large one.
    #[error("jargon-attest-v1: BinaryFuse8 construction failed: {message}")]
    JargonAttestConstructionFailed {
        /// `xorf`'s own error message.
        message: String,
    },

    /// A `jargon-attest-v1` `.bin` is shorter than its fixed header (magic
    /// + key count + filter descriptor) — see [`crate::jargon_attest::JargonAttestPack::load`].
    #[error("jargon-attest-v1: pack is {len} byte(s), shorter than the {min}-byte header")]
    JargonAttestTruncated {
        /// The pack's actual byte length.
        len: usize,
        /// The minimum required byte length.
        min: usize,
    },

    /// A `jargon-attest-v1` `.bin` doesn't start with the expected magic
    /// bytes — see [`crate::jargon_attest::JargonAttestPack::load`].
    #[error("jargon-attest-v1: pack does not start with the expected magic bytes")]
    JargonAttestBadMagic,

    /// A `jargon-attest-v1` sidecar's `[pack].version` isn't exactly
    /// `"jargon-attest-v1"`.
    #[error("jargon-attest-v1: sidecar [pack].version is {found:?}, expected \"jargon-attest-v1\"")]
    JargonAttestVersionMismatch {
        /// The sidecar's actual `version` value.
        found: String,
    },

    /// A `jargon-attest-v1` sidecar's `[pack].key_count` doesn't match the
    /// key count embedded in the `.bin`'s own header — the two files
    /// describe different builds.
    #[error(
        "jargon-attest-v1: sidecar [pack].key_count={sidecar} does not match the {embedded} \
         key(s) embedded in the .bin header"
    )]
    JargonAttestKeyCountMismatch {
        /// The sidecar's recorded key count.
        sidecar: u64,
        /// The `.bin` header's own recorded key count.
        embedded: u64,
    },

    /// A `human-evidence-v1` `.bin` ran out of bytes while a fixed-shape
    /// section was being read (see
    /// [`crate::human_evidence::HumanEvidencePack::load`]).
    #[error("human-evidence-v1: pack truncated at {section}")]
    HumanEvidenceTruncated {
        /// Which section ran out of bytes.
        section: &'static str,
    },

    /// A `human-evidence-v1` `.bin` doesn't start with the expected magic
    /// bytes.
    #[error("human-evidence-v1: pack does not start with the expected magic bytes")]
    HumanEvidenceBadMagic,

    /// A `human-evidence-v1` `.bin`'s format version isn't one this build
    /// of `friction-packs` knows how to read.
    #[error("human-evidence-v1: unsupported pack version {0}")]
    HumanEvidenceUnsupportedVersion(u16),
}

impl From<toml::de::Error> for PackError {
    fn from(err: toml::de::Error) -> Self {
        Self::Toml(Box::new(err))
    }
}

/// The value of one ASCII hex digit, `0..=15`.
///
/// # Panics
/// Panics if `byte` isn't an ASCII hex digit — every caller checks with
/// `is_ascii_hexdigit` first.
const fn hex_nibble(byte: u8) -> u8 {
    match byte {
        b'0'..=b'9' => byte - b'0',
        b'a'..=b'f' => byte - b'a' + 10,
        b'A'..=b'F' => byte - b'A' + 10,
        _ => panic!("byte was not validated as an ASCII hex digit"),
    }
}

/// A pinned SHA-256 checksum, stored as 32 raw bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Sha256([u8; 32]);

impl Sha256 {
    /// Parses a 64-character hex digest (either case) into a checksum.
    ///
    /// # Errors
    /// Returns [`PackError::InvalidChecksum`] if `hex` isn't exactly 64
    /// ASCII hex digits.
    pub fn parse_hex(hex: &str) -> Result<Self, PackError> {
        let trimmed = hex.trim();
        if trimmed.len() != 64 || !trimmed.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(PackError::InvalidChecksum {
                value: hex.to_string(),
            });
        }
        let mut bytes = [0u8; 32];
        for (chunk, out) in trimmed.as_bytes().chunks_exact(2).zip(bytes.iter_mut()) {
            *out = (hex_nibble(chunk[0]) << 4) | hex_nibble(chunk[1]);
        }
        Ok(Self(bytes))
    }

    /// The checksum's raw bytes.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Returns `true` if `bytes` hashes to this checksum.
    #[must_use]
    pub fn verify(&self, bytes: &[u8]) -> bool {
        let mut hasher = Sha2Hasher::new();
        hasher.update(bytes);
        hasher.finalize().as_slice() == self.0
    }

    /// Computes the sha256 checksum of `bytes` directly (no hex-digest
    /// parsing).
    #[must_use]
    pub fn of_bytes(bytes: &[u8]) -> Self {
        let mut hasher = Sha2Hasher::new();
        hasher.update(bytes);
        let digest = hasher.finalize();
        let mut out = [0u8; 32];
        out.copy_from_slice(digest.as_slice());
        Self(out)
    }
}

impl fmt::Display for Sha256 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in &self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `Sha256::parse_hex` round-trips a well-formed digest through
    /// `Display` as lowercase hex.
    #[test]
    fn sha256_parses_and_displays_lowercase() {
        let hex = "c4368b1c4c73825e3ebdf8d5750ce87bb035a69fd4f99db7930f8828a96347aa";
        let checksum = Sha256::parse_hex(hex).unwrap();
        assert_eq!(checksum.to_string(), hex);

        let upper = hex.to_uppercase();
        let from_upper = Sha256::parse_hex(&upper).unwrap();
        assert_eq!(from_upper.to_string(), hex);
    }

    /// A wrong-length checksum is rejected.
    #[test]
    fn sha256_rejects_wrong_length() {
        let err = Sha256::parse_hex("deadbeef").unwrap_err();
        assert!(matches!(err, PackError::InvalidChecksum { .. }));
    }

    /// A checksum containing non-hex characters is rejected.
    #[test]
    fn sha256_rejects_non_hex_characters() {
        let bad = "z".repeat(64);
        let err = Sha256::parse_hex(&bad).unwrap_err();
        assert!(matches!(err, PackError::InvalidChecksum { .. }));
    }

    /// `Sha256::of_bytes` agrees with `Sha256::parse_hex` on a known digest.
    #[test]
    fn sha256_of_bytes_matches_parse_hex() {
        let hex = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
        assert_eq!(Sha256::of_bytes(b"").to_string(), hex);
        assert_eq!(Sha256::of_bytes(b""), Sha256::parse_hex(hex).unwrap());
    }

    /// `Sha256::verify` accepts the exact bytes it was computed from and
    /// rejects any others.
    #[test]
    fn sha256_verify_checks_bytes() {
        let checksum =
            Sha256::parse_hex("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
                .unwrap();
        assert!(checksum.verify(b""));
        assert!(!checksum.verify(b"not empty"));
    }
}
