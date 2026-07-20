//! Validates `friction check --format sarif`'s actual output against the
//! vendored SARIF 2.1.0 JSON schema (`tests/data/sarif-schema-2.1.0.json`,
//! downloaded once from the upstream `oasis-tcs/sarif-spec` repository and
//! committed here) using the `jsonschema` crate.

use std::fs;
use std::path::{Path, PathBuf};

use assert_cmd::Command;

fn schema_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/sarif-schema-2.1.0.json")
}

fn write_fixture(dir: &Path, name: &str, contents: &str) -> PathBuf {
    let path = dir.join(name);
    fs::write(&path, contents).expect("fixture writes");
    path
}

const MESSY_BLOG: &str = "Moreover, it is worth noting that this release is not just fast but \
                           also reliable. Furthermore, it is worth noting that the results are \
                           not just accurate but also consistent. In conclusion, this release \
                           delivers speed, reliability, and consistency for every team.\n";

/// `check --format sarif` output, with at least one finding, validates
/// cleanly against the vendored SARIF 2.1.0 schema.
#[test]
fn sarif_output_with_findings_validates_against_the_schema() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = write_fixture(dir.path(), "messy.md", MESSY_BLOG);

    let output = Command::cargo_bin("friction")
        .expect("the friction binary builds")
        .arg("check")
        .arg(&path)
        .args(["--genre", "blog", "--format", "sarif"])
        .output()
        .expect("friction runs");

    let instance: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("check --format sarif prints valid JSON");
    validate_against_schema(&instance);

    // Sanity: this fixture is guaranteed to produce findings (see
    // tests/cli.rs's own doc comment on `MESSY_BLOG`), so the schema
    // check above is exercising a non-empty `results` array, not just
    // the empty-log shape.
    assert!(
        !instance["runs"][0]["results"]
            .as_array()
            .expect("runs[0].results is an array")
            .is_empty()
    );
}

/// SARIF output for a document with zero findings — the empty-`results`
/// shape — also validates against the schema.
#[test]
fn sarif_output_with_no_findings_validates_against_the_schema() {
    let clean =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus/human/docs/1fedcd88e3cf6845.md");

    let output = Command::cargo_bin("friction")
        .expect("the friction binary builds")
        .arg("check")
        .arg(&clean)
        .args(["--genre", "docs", "--format", "sarif"])
        .output()
        .expect("friction runs");

    let instance: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("check --format sarif prints valid JSON");
    validate_against_schema(&instance);
    assert!(
        instance["runs"][0]["results"]
            .as_array()
            .expect("runs[0].results is an array")
            .is_empty()
    );
}

fn validate_against_schema(instance: &serde_json::Value) {
    let schema_text = fs::read_to_string(schema_path()).expect("vendored SARIF schema reads");
    let schema: serde_json::Value =
        serde_json::from_str(&schema_text).expect("vendored SARIF schema is valid JSON");
    let validator =
        jsonschema::validator_for(&schema).expect("vendored SARIF schema itself compiles");

    let errors: Vec<String> = validator
        .iter_errors(instance)
        .map(|e| format!("{e} (at {})", e.instance_path))
        .collect();
    assert!(
        errors.is_empty(),
        "SARIF output failed schema validation:\n{}",
        errors.join("\n")
    );
}
