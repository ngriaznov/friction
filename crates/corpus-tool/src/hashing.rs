//! Deterministic hashing and word-count helpers.

use std::fmt::Write as _;

use sha2::{Digest, Sha256};

/// Lowercase hex SHA-256 digest of `bytes`.
///
/// Used both for manifest `sha256` verification and, applied to
/// document ids, as the deterministic ordering key for the stratified
/// split — never as an ambient RNG seed.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest
        .iter()
        .fold(String::with_capacity(64), |mut acc, byte| {
            write!(acc, "{byte:02x}").expect("writing to a String never fails");
            acc
        })
}

/// Whitespace-token word count. This is the single definition of "word
/// count" used consistently by `validate`'s range check and `clean`'s
/// drop threshold.
pub fn word_count(text: &str) -> usize {
    text.split_whitespace().count()
}

/// A stable (non-cryptographic-use, but deterministic) `u64` derived from
/// `bytes`: the first 8 bytes of `sha256(bytes)`, big-endian.
///
/// Used by `generate` to derive per-job seeds from
/// `(model, prompt_id, slice)` without any ambient RNG: same input
/// bytes always produce the same `u64`, on any machine, any run.
pub fn stable_hash_u64(bytes: &[u8]) -> u64 {
    let digest = Sha256::digest(bytes);
    let mut first8 = [0u8; 8];
    first8.copy_from_slice(&digest[..8]);
    u64::from_be_bytes(first8)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// sha256 hex digest matches a known test vector (empty
    /// string), confirming encoding (lowercase, no separators).
    #[test]
    fn sha256_hex_matches_known_vector_for_empty_input() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    /// sha256 hex digest of `"abc"` matches the standard NIST test
    /// vector.
    #[test]
    fn sha256_hex_matches_known_vector_for_abc() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    /// Hashing the same bytes twice is byte-identical (no ambient
    /// state).
    #[test]
    fn sha256_hex_is_deterministic_across_calls() {
        assert_eq!(sha256_hex(b"friction"), sha256_hex(b"friction"));
    }

    /// Word count is a plain whitespace-token count.
    #[test]
    fn word_count_counts_whitespace_tokens() {
        assert_eq!(word_count("one two  three\nfour\ttab"), 5);
        assert_eq!(word_count(""), 0);
        assert_eq!(word_count("   "), 0);
    }

    /// `stable_hash_u64` matches a known test vector (first
    /// 8 bytes of `sha256("m|p|s")`, big-endian), pinning the exact
    /// derivation so seed generation never silently drifts.
    #[test]
    fn stable_hash_u64_matches_known_vector() {
        assert_eq!(stable_hash_u64(b"m|p|s"), 5_472_942_670_496_659_515);
    }

    /// `stable_hash_u64` is deterministic across calls, no ambient
    /// state.
    #[test]
    fn stable_hash_u64_is_deterministic_across_calls() {
        assert_eq!(stable_hash_u64(b"friction"), stable_hash_u64(b"friction"));
    }

    /// Different inputs (almost certainly) hash differently —
    /// guards against an accidental constant-function regression.
    #[test]
    fn stable_hash_u64_differs_for_different_inputs() {
        assert_ne!(stable_hash_u64(b"a"), stable_hash_u64(b"b"));
    }
}
