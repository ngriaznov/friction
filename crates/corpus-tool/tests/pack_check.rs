//! Integration tests for `corpus-tool pack-check`.

use std::path::PathBuf;

use corpus_tool::commands::pack_check::{self, Args};

fn write_pack(dir: &std::path::Path, contents: &str) -> PathBuf {
    let path = dir.join("pack.toml");
    std::fs::write(&path, contents).unwrap();
    path
}

const CLEAN_PACK: &str = r#"
    [pack]
    version = "inventory-v1"
    corpus_manifest_sha256 = "x"
    paired_manifest_sha256 = "x"

    [[deletion_spans]]
    id = "span.simply"
    pattern = '(?i)\bsimply\s+'
    anchor = "mid_sentence"
    repair = "delete"
    source = "seed"
"#;

#[test]
fn pack_check_succeeds_on_a_clean_hermetic_pack() {
    let dir = tempfile::tempdir().unwrap();
    let pack = write_pack(dir.path(), CLEAN_PACK);
    pack_check::run(&Args { pack }).unwrap();
}

#[test]
fn pack_check_fails_on_a_structurally_invalid_pack() {
    let dir = tempfile::tempdir().unwrap();
    let pack = write_pack(dir.path(), "not [ valid toml");
    let err = pack_check::run(&Args { pack }).unwrap_err();
    assert!(err.to_string().contains("structurally invalid"));
}

#[test]
fn pack_check_fails_on_a_disjointness_violation() {
    let toml = r#"
        [pack]
        version = "inventory-v1"
        corpus_manifest_sha256 = "x"
        paired_manifest_sha256 = "x"

        [[substitution_pairs]]
        id = "sub.only"
        pattern = "(?i)\\btarget\\b"
        replacement = "target"
        repair = "substitute"
        source = "seed"
    "#;
    let dir = tempfile::tempdir().unwrap();
    let pack = write_pack(dir.path(), toml);
    let err = pack_check::run(&Args { pack }).unwrap_err();
    assert!(err.to_string().contains("violation"));
}

#[test]
fn pack_check_fails_on_a_closure_violation() {
    let toml = r#"
        [pack]
        version = "inventory-v1"
        corpus_manifest_sha256 = "x"
        paired_manifest_sha256 = "x"

        [[substitution_pairs]]
        id = "sub.only"
        pattern = "(?i)\\bfoo\\b"
        replacement = "surprise"
        repair = "substitute"
        source = "seed"
    "#;
    let dir = tempfile::tempdir().unwrap();
    let pack = write_pack(dir.path(), toml);
    let err = pack_check::run(&Args { pack }).unwrap_err();
    assert!(err.to_string().contains("violation"));
}

#[test]
fn pack_check_fails_on_a_frequency_hygiene_violation() {
    let toml = r#"
        [pack]
        version = "inventory-v1"
        corpus_manifest_sha256 = "x"
        paired_manifest_sha256 = "x"

        [[substitution_pairs]]
        id = "sub.only"
        pattern = "(?i)\\bfoo\\b"
        replacement = "bar"
        repair = "substitute"
        source = "seed"
        declared_literal_tokens = ["bar"]

        [[output_frequency_bands]]
        frame = "bar"
        unit = "per_thousand_words"
        observed_rate = 0.5
        max = 0.1
        sample_doc_count = 10
        method = "test"
    "#;
    let dir = tempfile::tempdir().unwrap();
    let pack = write_pack(dir.path(), toml);
    let err = pack_check::run(&Args { pack }).unwrap_err();
    assert!(err.to_string().contains("violation"));
}

/// The CLI-path belt to `validate_on_the_embedded_inventory_v1_pack_is_clean`'s
/// suspenders: points at the real shipped pack via a
/// `CARGO_MANIFEST_DIR`-relative path.
#[test]
fn pack_check_succeeds_on_the_real_shipped_inventory_v1_pack() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let pack = PathBuf::from(manifest_dir).join("../friction-packs/packs/inventory-en-v1.toml");
    pack_check::run(&Args { pack }).unwrap();
}
