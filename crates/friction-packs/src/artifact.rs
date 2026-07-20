//! The built-in registry of downloadable, sha256-pinned NLP artifacts.
//!
//! The registry's data lives in `registry.toml`, embedded into the binary
//! via [`include_str!`] and parsed once into [`REGISTRY`]. Nothing in this
//! module performs I/O: fetching the artifacts an entry describes and
//! verifying them against [`Artifact::sha256`] is `friction-cli`'s job
//! (the `friction setup` subcommand).

use std::fmt;
use std::str::FromStr;
use std::sync::LazyLock;

use serde::Deserialize;
use sha2::{Digest, Sha256 as Sha2Hasher};

/// The embedded registry source. See `registry.toml` in this crate for the
/// field documentation and the current artifact list.
const REGISTRY_TOML: &str = include_str!("registry.toml");

/// The built-in artifact registry, parsed once from the embedded
/// `registry.toml` and reused for the life of the process, in the fixed
/// order the artifacts appear in that file.
///
/// # Panics
/// Panics if the embedded `registry.toml` fails to parse. That would mean
/// this crate shipped with a malformed registry file — a bug in this
/// crate's own data, covered by its tests — not a condition any caller can
/// recover from by retrying.
pub static REGISTRY: LazyLock<Vec<Artifact>> = LazyLock::new(|| {
    parse_registry(REGISTRY_TOML)
        .expect("embedded registry.toml must parse: see this crate's registry_toml tests")
});

/// Errors produced while parsing or validating the artifact registry.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum PackError {
    /// The registry TOML failed to parse.
    #[error("registry.toml is not valid TOML: {0}")]
    Toml(#[from] Box<toml::de::Error>),

    /// An artifact's `sha256` field is not a well-formed 64-character
    /// lowercase or uppercase hex digest.
    #[error("{value:?} is not a valid sha256 checksum (expected 64 hex digits)")]
    InvalidChecksum {
        /// The offending value, as written in the registry.
        value: String,
    },

    /// An artifact's `kind` field does not name a known [`ArtifactKind`].
    #[error("{kind:?} is not a known artifact kind")]
    UnknownArtifactKind {
        /// The offending value, as written in the registry.
        kind: String,
    },
}

impl From<toml::de::Error> for PackError {
    fn from(err: toml::de::Error) -> Self {
        Self::Toml(Box::new(err))
    }
}

/// The value of one ASCII hex digit, `0..=15`.
///
/// # Panics
/// Panics if `byte` is not an ASCII hex digit — every caller here checks
/// that with `is_ascii_hexdigit` first.
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
    /// Returns [`PackError::InvalidChecksum`] if `hex` is not exactly 64
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
}

impl fmt::Display for Sha256 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in &self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

/// The kind of NLP artifact an [`Artifact`] entry describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ArtifactKind {
    /// An SRX (or equivalent) sentence-segmentation ruleset.
    SegmentationRuleset,
    /// A part-of-speech tagger's tokenizer binary.
    TaggerTokenizer,
    /// A part-of-speech tagger's compiled rule set.
    TaggerRules,
    /// A dependency-parser model.
    DependencyParserModel,
}

impl ArtifactKind {
    const fn as_str(self) -> &'static str {
        match self {
            Self::SegmentationRuleset => "segmentation-ruleset",
            Self::TaggerTokenizer => "tagger-tokenizer",
            Self::TaggerRules => "tagger-rules",
            Self::DependencyParserModel => "dependency-parser-model",
        }
    }
}

impl fmt::Display for ArtifactKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for ArtifactKind {
    type Err = PackError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "segmentation-ruleset" => Ok(Self::SegmentationRuleset),
            "tagger-tokenizer" => Ok(Self::TaggerTokenizer),
            "tagger-rules" => Ok(Self::TaggerRules),
            "dependency-parser-model" => Ok(Self::DependencyParserModel),
            other => Err(PackError::UnknownArtifactKind {
                kind: other.to_string(),
            }),
        }
    }
}

/// One versioned, sha256-pinned downloadable artifact from the registry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Artifact {
    /// Stable identifier; used as the artifact's cache subdirectory name.
    pub name: Box<str>,
    /// What kind of artifact this is.
    pub kind: ArtifactKind,
    /// The version this pin corresponds to: an upstream release tag where
    /// one exists, otherwise a snapshot date for artifacts fetched from an
    /// unversioned branch.
    pub pack_version: Box<str>,
    /// Source URL. Downloaded byte-for-byte; never mutated at fetch time.
    pub url: Box<str>,
    /// Expected checksum of the artifact's exact bytes.
    pub sha256: Sha256,
    /// Expected size in bytes, used as a download sanity bound.
    pub size_bytes: u64,
    /// The artifact's own license (not `friction`'s).
    pub license: Box<str>,
    /// A caveat on the artifact's license or provenance that a maintainer
    /// should read before this artifact is bundled into a distributed
    /// pack, if any.
    pub license_note: Option<Box<str>>,
}

/// The TOML shape of a single `[[artifacts]]` entry, before checksum and
/// kind validation.
#[derive(Debug, Deserialize)]
struct RawArtifact {
    name: String,
    kind: String,
    pack_version: String,
    url: String,
    sha256: String,
    size_bytes: u64,
    license: String,
    #[serde(default)]
    license_note: Option<String>,
}

/// The TOML shape of the registry file as a whole.
#[derive(Debug, Deserialize)]
struct RawRegistry {
    /// Reserved for future format changes; unused today but present so a
    /// breaking registry-format change can be detected explicitly rather
    /// than silently misparsed.
    #[allow(dead_code)]
    schema_version: u32,
    #[serde(default)]
    artifacts: Vec<RawArtifact>,
}

impl TryFrom<RawArtifact> for Artifact {
    type Error = PackError;

    fn try_from(raw: RawArtifact) -> Result<Self, PackError> {
        Ok(Self {
            name: raw.name.into_boxed_str(),
            kind: raw.kind.parse()?,
            pack_version: raw.pack_version.into_boxed_str(),
            url: raw.url.into_boxed_str(),
            sha256: Sha256::parse_hex(&raw.sha256)?,
            size_bytes: raw.size_bytes,
            license: raw.license.into_boxed_str(),
            license_note: raw.license_note.map(String::into_boxed_str),
        })
    }
}

/// Parses `toml_src` (the shape of `registry.toml`) into the artifact list,
/// in file order.
///
/// # Errors
/// Returns [`PackError`] if `toml_src` is not valid TOML in the expected
/// shape, or if any entry's `sha256` or `kind` field is malformed.
pub fn parse_registry(toml_src: &str) -> Result<Vec<Artifact>, PackError> {
    let raw: RawRegistry = toml::from_str(toml_src)?;
    raw.artifacts.into_iter().map(Artifact::try_from).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The embedded `registry.toml` parses without error. It currently
    /// declares no artifacts (see the file's own header comment for why:
    /// every NLP asset this workspace uses today is either vendored or
    /// fetched at build time, not through this runtime-cache mechanism) —
    /// this pins that as a deliberate, tested state rather than an
    /// accident.
    #[test]
    fn embedded_registry_parses() {
        assert!(REGISTRY.is_empty());
    }

    /// Every entry in the embedded registry has a non-empty URL, a
    /// plausible size, and a checksum that round-trips through display.
    #[test]
    fn embedded_registry_entries_are_well_formed() {
        for artifact in &*REGISTRY {
            assert!(!artifact.url.is_empty(), "{} has no url", artifact.name);
            assert!(artifact.size_bytes > 0, "{} has a zero size", artifact.name);
            assert_eq!(artifact.sha256.to_string().len(), 64);
        }
    }

    /// `Sha256::parse_hex` accepts a well-formed digest and round-trips it
    /// through `Display` as lowercase hex.
    #[test]
    fn sha256_parses_and_displays_lowercase() {
        let hex = "c4368b1c4c73825e3ebdf8d5750ce87bb035a69fd4f99db7930f8828a96347aa";
        let checksum = Sha256::parse_hex(hex).unwrap();
        assert_eq!(checksum.to_string(), hex);

        let upper = hex.to_uppercase();
        let from_upper = Sha256::parse_hex(&upper).unwrap();
        assert_eq!(from_upper.to_string(), hex);
    }

    /// A checksum of the wrong length is rejected.
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

    /// `Sha256::verify` matches the checksum of the exact bytes it was
    /// computed from, and rejects any other bytes.
    #[test]
    fn sha256_verify_checks_bytes() {
        let checksum =
            Sha256::parse_hex("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
                .unwrap();
        assert!(checksum.verify(b""));
        assert!(!checksum.verify(b"not empty"));
    }

    /// An unknown `kind` string is rejected with a clear error rather than
    /// silently defaulting.
    #[test]
    fn unknown_artifact_kind_is_rejected() {
        let toml_src = r#"
            schema_version = 1

            [[artifacts]]
            name = "mystery"
            kind = "not-a-real-kind"
            pack_version = "1"
            url = "https://example.invalid/x"
            sha256 = "0000000000000000000000000000000000000000000000000000000000000000000000000000"
            size_bytes = 1
            license = "MIT"
        "#;
        let err = parse_registry(toml_src).unwrap_err();
        assert!(matches!(err, PackError::UnknownArtifactKind { .. }));
    }

    /// A malformed checksum in the registry is rejected with a clear
    /// error rather than a panic.
    #[test]
    fn malformed_registry_checksum_is_rejected() {
        let toml_src = r#"
            schema_version = 1

            [[artifacts]]
            name = "mystery"
            kind = "segmentation-ruleset"
            pack_version = "1"
            url = "https://example.invalid/x"
            sha256 = "not-hex"
            size_bytes = 1
            license = "MIT"
        "#;
        let err = parse_registry(toml_src).unwrap_err();
        assert!(matches!(err, PackError::InvalidChecksum { .. }));
    }

    /// A registry with no `[[artifacts]]` entries parses to an empty list
    /// rather than erroring.
    #[test]
    fn empty_registry_parses_to_empty_list() {
        let artifacts = parse_registry("schema_version = 1\n").unwrap();
        assert!(artifacts.is_empty());
    }

    /// `ArtifactKind` round-trips through its string form.
    #[test]
    fn artifact_kind_round_trips() {
        for kind in [
            ArtifactKind::SegmentationRuleset,
            ArtifactKind::TaggerTokenizer,
            ArtifactKind::TaggerRules,
            ArtifactKind::DependencyParserModel,
        ] {
            let parsed: ArtifactKind = kind.to_string().parse().unwrap();
            assert_eq!(parsed, kind);
        }
    }
}
