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

/// The `composed_four_operation_paragraph` fixture from
/// `docs/research/fixtures.json`, copied here verbatim: reliably
/// produces detection spans across multiple channels.
const MESSY_BLOG: &str = "This guide will walk you through configuring the backup agent for \
                           your staging environment. It is important to note that the agent \
                           performs validation of the configuration file before each run. Once \
                           validation succeeds, the agent conducts an analysis of the snapshot \
                           catalog and simply uploads any missing segments. By following these \
                           steps, you can quickly and easily verify that your backups are \
                           consistent. If you have any questions or require further assistance, \
                           please reach out to our support team.";

/// The `near_noop_clean_text` fixture from `docs/research/fixtures.json`,
/// copied here verbatim: produces zero detection spans against every
/// generator family.
const CLEAN: &str = "Run the scanner from the project root. Results stream in as they are \
                      found, and nothing is deleted without confirmation.";

/// `check --format sarif` output, with at least one detected span,
/// validates cleanly against the vendored SARIF 2.1.0 schema.
#[test]
fn sarif_output_with_spans_validates_against_the_schema() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = write_fixture(dir.path(), "messy.md", MESSY_BLOG);

    let output = Command::cargo_bin("friction")
        .expect("the friction binary builds")
        .arg("check")
        .arg(&path)
        .args(["--genre", "blog", "--family", "qwen", "--format", "sarif"])
        .output()
        .expect("friction runs");

    let instance: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("check --format sarif prints valid JSON");
    validate_against_schema(&instance);

    // Sanity: this fixture is guaranteed to produce spans (see this
    // file's own doc comment on `MESSY_BLOG`), so the schema check above
    // is exercising a non-empty `results` array, not just the empty-log
    // shape.
    assert!(
        !instance["runs"][0]["results"]
            .as_array()
            .expect("runs[0].results is an array")
            .is_empty()
    );
}

/// SARIF output for a document with zero detected spans (the
/// empty-`results` shape) also validates against the schema.
#[test]
fn sarif_output_with_no_spans_validates_against_the_schema() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = write_fixture(dir.path(), "clean.md", CLEAN);

    let output = Command::cargo_bin("friction")
        .expect("the friction binary builds")
        .arg("check")
        .arg(&path)
        .args(["--genre", "docs", "--family", "qwen", "--format", "sarif"])
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
