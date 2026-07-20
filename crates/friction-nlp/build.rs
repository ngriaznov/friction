//! Downloads and verifies the `nlprule` English tagger model at build
//! time, decompresses it, and writes it into `OUT_DIR` so `src/tag_nlprule.rs`
//! can embed it with `include_bytes!`.
//!
//! This is the *only* network access anywhere in the tagging/inflection
//! code this crate owns: `check`/`fix` never download anything at
//! runtime (see the crate-level determinism and offline discipline this
//! workspace requires). A hash mismatch fails the build hard rather than
//! silently trusting an unpinned or tampered download.
//!
//! On an incremental build with an unchanged pin, the cached, already-
//! verified file in `OUT_DIR` is reused and no network round-trip happens
//! at all (see [`is_cached`]).

use std::env;
use std::fs;
use std::io::Read as _;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

/// Pinned source of the nlprule English tokenizer/tagger binary: tags,
/// lemmas, and disambiguation rules for the tagset [`crate::tag_nlprule`]
/// consumes. Captured from the `0.6.4` GitHub release, matching the
/// `nlprule` crate version this crate depends on (see `Cargo.toml`).
const TOKENIZER_URL: &str =
    "https://github.com/bminixhofer/nlprule/releases/download/0.6.4/en_tokenizer.bin.gz";

/// sha256 of the *compressed* download at [`TOKENIZER_URL`], checked
/// before any of its bytes are trusted or decompressed.
const TOKENIZER_SHA256: &str = "b500dd208ace9ba218f6b52f8cdab63d4c09d6f2967e9bd8f917bf5984d4468a";

/// Name of the decompressed model file written into `OUT_DIR`, and the
/// name `src/tag_nlprule.rs`'s `include_bytes!` expects.
const TOKENIZER_OUT_NAME: &str = "en_tokenizer.bin";

/// Stamp file recording which pinned hash produced the cached `OUT_DIR`
/// artifact, so a rebuild with an unchanged pin can skip the network
/// round-trip; a changed pin (this file edited) invalidates it.
const STAMP_NAME: &str = "en_tokenizer.sha256";

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let out_dir =
        PathBuf::from(env::var("OUT_DIR").expect("cargo always sets OUT_DIR for build scripts"));
    let bin_path = out_dir.join(TOKENIZER_OUT_NAME);
    let stamp_path = out_dir.join(STAMP_NAME);

    if is_cached(&bin_path, &stamp_path) {
        return;
    }

    let compressed = download(TOKENIZER_URL);
    verify_sha256(&compressed, TOKENIZER_SHA256, TOKENIZER_URL);
    let decompressed = decompress_gzip(&compressed);

    fs::write(&bin_path, &decompressed)
        .unwrap_or_else(|err| panic!("failed to write {}: {err}", bin_path.display()));
    fs::write(&stamp_path, TOKENIZER_SHA256)
        .unwrap_or_else(|err| panic!("failed to write {}: {err}", stamp_path.display()));
}

/// A previously-verified model is already sitting in `OUT_DIR` under the
/// pin currently compiled into this file.
fn is_cached(bin_path: &Path, stamp_path: &Path) -> bool {
    let Ok(stamped_hash) = fs::read_to_string(stamp_path) else {
        return false;
    };
    stamped_hash.trim() == TOKENIZER_SHA256 && bin_path.is_file()
}

/// Downloads `url`'s full response body.
///
/// # Panics
/// Panics (failing the build) if the request or the read fails. Model
/// downloads only ever happen at build time, never at `check`/`fix`
/// runtime, so a hard failure here cannot surface as a runtime crash.
fn download(url: &str) -> Vec<u8> {
    let mut response = ureq::get(url)
        .call()
        .unwrap_or_else(|err| panic!("failed to download {url}: {err}"));
    let mut buf = Vec::new();
    response
        .body_mut()
        .as_reader()
        .read_to_end(&mut buf)
        .unwrap_or_else(|err| panic!("failed to read response body from {url}: {err}"));
    buf
}

/// Verifies `bytes` hashes to `expected_hex`, panicking (failing the
/// build) on mismatch rather than trusting an unpinned or tampered
/// download.
fn verify_sha256(bytes: &[u8], expected_hex: &str, source: &str) {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let actual_hex = hex_encode(&hasher.finalize());
    assert!(
        actual_hex == expected_hex,
        "sha256 mismatch for {source}: expected {expected_hex}, got {actual_hex} \
         (refusing to trust an unpinned or tampered download)"
    );
}

fn hex_encode(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    bytes
        .iter()
        .fold(String::with_capacity(bytes.len() * 2), |mut acc, byte| {
            let _ = write!(acc, "{byte:02x}");
            acc
        })
}

/// Gunzips `bytes`.
///
/// # Panics
/// Panics (failing the build) if `bytes` is not valid gzip. Reached only
/// after [`verify_sha256`] has already confirmed `bytes` matches the
/// pinned hash, so this indicates a corrupt pin (the pinned hash points at
/// a file that no longer decompresses cleanly), not an untrusted input.
fn decompress_gzip(bytes: &[u8]) -> Vec<u8> {
    let mut decoder = flate2::read::GzDecoder::new(bytes);
    let mut out = Vec::new();
    decoder
        .read_to_end(&mut out)
        .unwrap_or_else(|err| panic!("failed to gunzip tokenizer model: {err}"));
    out
}
