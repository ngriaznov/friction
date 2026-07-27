//! `corpus-tool weights-pack` — builds derived weight artifacts.
//!
//! Produces the `postcard`-encoded `.bin` weight artifacts
//! `friction_nlp::PerceptronTagger::new` and
//! `friction_nlp::PerceptronParser::new` load at runtime (see
//! `crates/friction-nlp/weights/NOTICE.md`'s "derived artifacts"
//! section).
//!
//! Deterministic and fully offline: both inputs are the vendored
//! `json.gz` weight artifacts already checked into `crates/friction-nlp/
//! weights/`; this command never makes a network request. Reads each
//! `json.gz`, computes its own sha256, and hands both to the matching
//! `friction_nlp::pack_perceptron_tagger_bin` /
//! `pack_perceptron_parser_bin` — the same functions
//! `friction-nlp`'s own `pack_perceptron_*_bin_is_deterministic` tests
//! pin — which parse it, `postcard`-encode the result, and prepend the
//! shared header (`friction_nlp`'s private `weights_bin` module) carrying
//! that sha256, so a stale `.bin` (source `json.gz` regenerated, this
//! command not re-run) fails loudly at `friction_nlp` init instead of
//! silently drifting.

use std::path::PathBuf;

use anyhow::Context as _;
use clap::Args as ClapArgs;

use crate::hashing::sha256_hex;

/// Arguments for `corpus-tool weights-pack`.
#[derive(Debug, ClapArgs)]
pub struct Args {
    /// Path to the vendored dependency-parser `json.gz` weight artifact.
    #[arg(long, default_value = "crates/friction-nlp/weights/parser_en.json.gz")]
    pub parser_json_gz: PathBuf,
    /// Path to the vendored POS-tagger `json.gz` weight artifact.
    #[arg(
        long,
        default_value = "crates/friction-nlp/weights/perceptron_en.json.gz"
    )]
    pub tagger_json_gz: PathBuf,
    /// Path to write the derived dependency-parser binary artifact to.
    #[arg(long, default_value = "crates/friction-nlp/weights/parser_en.bin")]
    pub out_parser_bin: PathBuf,
    /// Path to write the derived POS-tagger binary artifact to.
    #[arg(long, default_value = "crates/friction-nlp/weights/perceptron_en.bin")]
    pub out_tagger_bin: PathBuf,
}

/// Runs `weights-pack`: packs the dependency parser, then the tagger.
///
/// # Errors
/// Returns an error if either `json.gz` can't be read, if packing fails
/// (malformed `json.gz` — see
/// [`friction_nlp::PerceptronParseError`]/[`friction_nlp::PerceptronTagError`]),
/// or if either `.bin` can't be written.
pub fn run(args: &Args) -> anyhow::Result<()> {
    pack_one(
        "parser",
        &args.parser_json_gz,
        &args.out_parser_bin,
        friction_nlp::pack_perceptron_parser_bin,
    )?;
    pack_one(
        "tagger",
        &args.tagger_json_gz,
        &args.out_tagger_bin,
        friction_nlp::pack_perceptron_tagger_bin,
    )?;
    Ok(())
}

/// Reads `json_gz_path`, hashes it, packs it via `pack` (either
/// `friction_nlp::pack_perceptron_parser_bin` or
/// `pack_perceptron_tagger_bin`), and writes the result to `out_bin_path`
/// — the one shared shape both artifacts pack through, so the two
/// [`run`] call sites differ only in which function and paths they pass.
fn pack_one<E: std::fmt::Display>(
    label: &str,
    json_gz_path: &std::path::Path,
    out_bin_path: &std::path::Path,
    pack: impl FnOnce(&[u8], &str) -> Result<Vec<u8>, E>,
) -> anyhow::Result<()> {
    let json_gz_bytes = std::fs::read(json_gz_path)
        .with_context(|| format!("reading {}", json_gz_path.display()))?;
    let source_sha256 = sha256_hex(&json_gz_bytes);
    let packed = pack(&json_gz_bytes, &source_sha256)
        .map_err(|err| anyhow::anyhow!("packing {label} weights: {err}"))?;
    if let Some(parent) = out_bin_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(out_bin_path, &packed)
        .with_context(|| format!("writing {}", out_bin_path.display()))?;
    println!(
        "weights-pack: {label}: {} ({} bytes) -> {} ({} bytes), source sha256 {source_sha256}",
        json_gz_path.display(),
        json_gz_bytes.len(),
        out_bin_path.display(),
        packed.len(),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Running `weights-pack` twice over the same `json.gz` inputs
    /// produces bit-identical `.bin` bytes — `postcard` is deterministic
    /// and so is the shared header (see `friction_nlp`'s own
    /// `pack_perceptron_*_bin_is_deterministic` unit tests for the
    /// underlying guarantee); this test pins it at this command's own
    /// read-hash-pack-write level, over the real vendored artifacts.
    #[test]
    fn packing_the_same_json_gz_twice_is_byte_identical() {
        let repo_root = repo_root();
        let tagger_json_gz = repo_root.join("crates/friction-nlp/weights/perceptron_en.json.gz");
        let bytes = std::fs::read(&tagger_json_gz).expect("vendored artifact exists");
        let sha = sha256_hex(&bytes);
        let first = friction_nlp::pack_perceptron_tagger_bin(&bytes, &sha).expect("packs");
        let second = friction_nlp::pack_perceptron_tagger_bin(&bytes, &sha).expect("packs");
        assert_eq!(first, second);
    }

    /// `pack_one` writes a file whose bytes match calling the packer
    /// directly, and prints a summary line naming both paths.
    #[test]
    fn pack_one_writes_the_packed_bytes_to_out_bin_path() {
        let repo_root = repo_root();
        let parser_json_gz = repo_root.join("crates/friction-nlp/weights/parser_en.json.gz");
        let dir = tempfile::tempdir().expect("tempdir");
        let out_bin = dir.path().join("parser_en.bin");

        pack_one(
            "parser",
            &parser_json_gz,
            &out_bin,
            friction_nlp::pack_perceptron_parser_bin,
        )
        .expect("packs");

        let written = std::fs::read(&out_bin).expect("out file written");
        let bytes = std::fs::read(&parser_json_gz).expect("vendored artifact exists");
        let sha = sha256_hex(&bytes);
        let expected = friction_nlp::pack_perceptron_parser_bin(&bytes, &sha).expect("packs");
        assert_eq!(written, expected);
    }

    /// This crate's own working directory convention (tests run from the
    /// crate root; the vendored weights live at the workspace root) —
    /// mirrors the relative-path assumption `corpus-tool`'s default CLI
    /// arguments already make (`crates/friction-nlp/weights/...`).
    fn repo_root() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()
            .expect("workspace root exists")
    }
}
