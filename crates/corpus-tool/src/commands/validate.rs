//! `corpus-tool validate` — checks the manifest and corpus files are
//! internally consistent.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use clap::Args as ClapArgs;

use crate::corpus_layout::relpath;
use crate::hashing::{sha256_hex, word_count};
use crate::manifest::{self, Class, ManifestRecord, PERSONAL_ATTESTATION_LICENSE};

const MIN_WORDS: usize = 300;
const MAX_WORDS: usize = 2000;

/// Arguments for `corpus-tool validate`.
#[derive(Debug, ClapArgs)]
pub struct Args {
    /// Corpus root directory.
    #[arg(long, default_value = "corpus")]
    pub corpus_dir: PathBuf,
}

/// Runs `validate`.
///
/// Hard failures (non-zero exit): the manifest fails to parse strictly;
/// a referenced file is missing; a file's sha256 doesn't match; a
/// license is empty; a `human` doc has neither `provenance_evidence` nor
/// `license: "personal-attestation"`; an `llm` doc is missing
/// `model`, `prompt_id`, or `gen_config`; or an `id` is duplicated.
///
/// Word counts outside `[300, 2000]` are warned to stderr, not failed.
/// An absent or empty manifest prints `"empty corpus"` and
/// succeeds.
///
/// # Errors
///
/// Returns an error summarizing the violation count if any hard rule is
/// violated; every individual violation is printed to stderr first.
pub fn run(args: &Args) -> anyhow::Result<()> {
    let manifest_path = args.corpus_dir.join("manifest.jsonl");
    let records = match manifest::read_manifest(&manifest_path)? {
        Some(records) if !records.is_empty() => records,
        _ => {
            println!("empty corpus");
            return Ok(());
        }
    };

    let mut errors = Vec::new();

    let mut seen_ids = BTreeSet::new();
    for record in &records {
        if !seen_ids.insert(record.id.as_str()) {
            errors.push(format!("duplicate id: {}", record.id));
        }
    }

    for record in &records {
        validate_record(record, &args.corpus_dir, &mut errors);
    }

    if !errors.is_empty() {
        for err in &errors {
            eprintln!("error: {err}");
        }
        anyhow::bail!(
            "validate failed: {} error(s) across {} doc(s)",
            errors.len(),
            records.len()
        );
    }

    println!("validate passed: {} doc(s)", records.len());
    Ok(())
}

fn validate_record(record: &ManifestRecord, corpus_dir: &Path, errors: &mut Vec<String>) {
    let rel = relpath(record);
    let path = corpus_dir.join(&rel);

    let Ok(bytes) = std::fs::read(&path) else {
        errors.push(format!("{}: missing file {rel}", record.id));
        return;
    };

    let actual_sha = sha256_hex(&bytes);
    if actual_sha != record.sha256 {
        errors.push(format!(
            "{}: sha256 mismatch (manifest {}, file {actual_sha})",
            record.id, record.sha256
        ));
    }

    if record.license.trim().is_empty() {
        errors.push(format!("{}: empty license", record.id));
    }

    match record.class {
        Class::Human => {
            let attested = record.license.trim() == PERSONAL_ATTESTATION_LICENSE;
            if record.provenance_evidence.is_none() && !attested {
                errors.push(format!(
                    "{}: human doc needs provenance_evidence or license \
                     \"{PERSONAL_ATTESTATION_LICENSE}\"",
                    record.id
                ));
            }
        }
        Class::Llm => {
            if record.model.is_none() {
                errors.push(format!("{}: llm doc missing model", record.id));
            }
            if record.prompt_id.is_none() {
                errors.push(format!("{}: llm doc missing prompt_id", record.id));
            }
            if record.gen_config.is_none() {
                errors.push(format!("{}: llm doc missing gen_config", record.id));
            }
        }
    }

    match std::str::from_utf8(&bytes) {
        Ok(text) => {
            let words = word_count(text);
            if !(MIN_WORDS..=MAX_WORDS).contains(&words) {
                eprintln!(
                    "warning: {} word count {words} outside [{MIN_WORDS}, {MAX_WORDS}]",
                    record.id
                );
            }
        }
        Err(_) => errors.push(format!("{}: file is not valid UTF-8", record.id)),
    }
}
