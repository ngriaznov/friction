//! Shared on-disk header for the derived, `postcard`-encoded weight
//! artifacts (`weights/parser_en.bin`, `weights/perceptron_en.bin`).
//!
//! The `json.gz` files in this crate's `weights/` directory stay the
//! audited interchange artifact from the training pipeline (see
//! `weights/NOTICE.md`); the `.bin` files are a derived, faster-to-load
//! encoding of the exact same data, produced by `corpus-tool
//! weights-pack` (see [`crate::tag_perceptron::pack_perceptron_tagger_bin`]
//! / [`crate::dep_perceptron::pack_perceptron_parser_bin`]).
//!
//! Both files share one fixed-length header, built by [`write_header`] and
//! validated by [`split_header`]: an 8-byte ASCII magic (distinct per
//! artifact — [`crate::tag_perceptron`] and [`crate::dep_perceptron`] each
//! define their own, so one can never be silently loaded in place of the
//! other), a little-endian `u16` format version, and the 64-byte
//! lowercase-hex sha256 of the `json.gz` this `.bin` was derived from —
//! immediately followed by the postcard-encoded payload.
//!
//! Recording the source sha256 lets a stale `.bin` (source `json.gz`
//! regenerated, `corpus-tool weights-pack` not re-run) fail loudly at
//! process init rather than silently drifting from the audited source:
//! each artifact's own `new()` compares [`split_header`]'s returned
//! sha256 against a compile-time constant of its own currently embedded
//! `json.gz`, not against a value computed at load time (hashing a
//! multi-megabyte file on every startup would undo the whole point of
//! shipping a faster-to-load artifact).

/// The on-disk format every header currently writes. Bumped only if this
/// module's *layout* changes (magic + version + sha256 + payload); a
/// payload-shape change is a per-artifact magic change instead, since the
/// two artifact types never share a magic.
const FORMAT_VERSION: u16 = 1;

/// A sha256 digest is recorded as lowercase hex (matching
/// `corpus_tool::hashing::sha256_hex`'s output shape), not raw digest
/// bytes, so a hex dump of the header stays human-legible.
const SHA256_HEX_LEN: usize = 64;

/// The fixed-length prefix every `.bin` starts with: 8-byte magic + 2-byte
/// little-endian version + 64-byte hex sha256.
///
/// `pub` (rather than `pub(crate)`, likewise for [`write_header`] and
/// [`split_header`] below): `weights_bin` is a private module, so nothing
/// here is reachable from outside the crate by path regardless — `pub`
/// just avoids clippy's `redundant_pub_crate` (an item can't be "more
/// public than necessary" when its only path to the outside is already
/// sealed one level up).
pub const HEADER_LEN: usize = 8 + 2 + SHA256_HEX_LEN;

/// Errors reading a derived binary weight artifact's header.
///
/// `pub` purely so [`PerceptronTagError`](crate::PerceptronTagError)/
/// [`PerceptronParseError`](crate::PerceptronParseError) — both `pub`
/// enums — can wrap it in a `#[from]` variant without tripping the
/// `private_interfaces` lint; see [`HEADER_LEN`]'s docs for why this
/// doesn't actually widen this module's external surface.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum WeightsBinError {
    /// Shorter than [`HEADER_LEN`] — not a header this module wrote.
    #[error("weight artifact truncated: {len} byte(s), need at least {min}")]
    Truncated { len: usize, min: usize },
    /// The first 8 bytes didn't match the caller's expected magic.
    #[error("weight artifact has an unrecognized magic")]
    BadMagic,
    /// The gzip payload after the header failed to decompress.
    #[error("weight artifact payload is not valid gzip data")]
    PayloadCorrupt,
    /// The version field isn't [`FORMAT_VERSION`].
    #[error("weight artifact format version {found} is unsupported (expected {expected})")]
    VersionMismatch { found: u16, expected: u16 },
    /// The sha256 field isn't valid UTF-8 — never expected for a header
    /// this module itself wrote, only for corrupted or hand-edited input.
    #[error("weight artifact's recorded sha256 is not valid UTF-8")]
    BadSha256,
}

/// Builds the fixed-length header for `magic` (8 ASCII bytes, distinct per
/// artifact) and `source_sha256_hex` (the source `json.gz`'s sha256, as
/// lowercase hex — exactly [`SHA256_HEX_LEN`] characters).
///
/// # Panics
/// Panics if `source_sha256_hex` isn't exactly [`SHA256_HEX_LEN`] bytes —
/// a caller bug (every sha256 hex digest is exactly 64 characters), never
/// a runtime condition.
pub fn write_header(magic: [u8; 8], source_sha256_hex: &str) -> Vec<u8> {
    assert_eq!(
        source_sha256_hex.len(),
        SHA256_HEX_LEN,
        "a sha256 hex digest is always exactly {SHA256_HEX_LEN} characters"
    );
    let mut out = Vec::with_capacity(HEADER_LEN);
    out.extend_from_slice(&magic);
    out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(source_sha256_hex.as_bytes());
    out
}

/// Validates `bytes`'s header against `magic` and splits it into
/// `(source_sha256_hex, payload)`. Does not check `source_sha256_hex`
/// against any expected value — the caller compares it against its own
/// compile-time constant (see this module's docs for why).
///
/// # Errors
/// [`WeightsBinError::Truncated`], [`WeightsBinError::BadMagic`],
/// [`WeightsBinError::VersionMismatch`], or [`WeightsBinError::BadSha256`]
/// — see each variant's own docs.
pub fn split_header(bytes: &[u8], magic: [u8; 8]) -> Result<(&str, &[u8]), WeightsBinError> {
    if bytes.len() < HEADER_LEN {
        return Err(WeightsBinError::Truncated {
            len: bytes.len(),
            min: HEADER_LEN,
        });
    }
    if bytes[..8] != magic {
        return Err(WeightsBinError::BadMagic);
    }
    let version = u16::from_le_bytes(
        bytes[8..10]
            .try_into()
            .expect("bytes.len() >= HEADER_LEN guarantees 2 bytes at [8..10]"),
    );
    if version != FORMAT_VERSION {
        return Err(WeightsBinError::VersionMismatch {
            found: version,
            expected: FORMAT_VERSION,
        });
    }
    let sha_hex =
        std::str::from_utf8(&bytes[10..HEADER_LEN]).map_err(|_| WeightsBinError::BadSha256)?;
    Ok((sha_hex, &bytes[HEADER_LEN..]))
}

/// Gzip-compresses a packed artifact's payload (everything after the
/// header). The header itself stays uncompressed so staleness detection
/// (`split_header` + sha comparison) never pays a decompression.
///
/// `flate2` writes mtime 0 and a fixed OS byte by default, so identical
/// payloads compress to identical bytes — the artifact stays a pure
/// function of its source, which the determinism tests pin.
#[must_use]
pub fn gzip_payload(payload: &[u8]) -> Vec<u8> {
    use std::io::Write as _;
    let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::best());
    encoder
        .write_all(payload)
        .expect("writing to an in-memory gzip encoder cannot fail");
    encoder
        .finish()
        .expect("finishing an in-memory gzip encoder cannot fail")
}

/// Inverse of [`gzip_payload`].
///
/// # Errors
/// Returns [`WeightsBinError::PayloadCorrupt`] if `payload` is not valid
/// gzip data.
pub fn gunzip_payload(payload: &[u8]) -> Result<Vec<u8>, WeightsBinError> {
    use std::io::Read as _;
    let mut decoder = flate2::read::GzDecoder::new(payload);
    let mut out = Vec::new();
    decoder
        .read_to_end(&mut out)
        .map_err(|_| WeightsBinError::PayloadCorrupt)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MAGIC: [u8; 8] = *b"TESTMAG1";
    const SHA: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    #[test]
    fn write_then_split_round_trips() {
        let mut bytes = write_header(MAGIC, SHA);
        bytes.extend_from_slice(b"payload");
        let (sha, payload) = split_header(&bytes, MAGIC).expect("valid header");
        assert_eq!(sha, SHA);
        assert_eq!(payload, b"payload");
    }

    #[test]
    fn write_header_length_is_exactly_header_len() {
        let bytes = write_header(MAGIC, SHA);
        assert_eq!(bytes.len(), HEADER_LEN);
    }

    #[test]
    fn split_header_rejects_truncated_input() {
        let err = split_header(b"short", MAGIC).unwrap_err();
        assert!(matches!(err, WeightsBinError::Truncated { .. }));
    }

    #[test]
    fn split_header_rejects_wrong_magic() {
        let bytes = write_header(*b"OTHRMAG1", SHA);
        let err = split_header(&bytes, MAGIC).unwrap_err();
        assert!(matches!(err, WeightsBinError::BadMagic));
    }

    #[test]
    fn split_header_rejects_wrong_version() {
        let mut bytes = write_header(MAGIC, SHA);
        bytes[8..10].copy_from_slice(&99u16.to_le_bytes());
        let err = split_header(&bytes, MAGIC).unwrap_err();
        assert!(matches!(
            err,
            WeightsBinError::VersionMismatch { found: 99, .. }
        ));
    }

    #[test]
    #[should_panic(expected = "always exactly 64 characters")]
    fn write_header_panics_on_wrong_length_sha() {
        let _ = write_header(MAGIC, "short");
    }
}
