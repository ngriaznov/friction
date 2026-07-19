//! `corpus-tool holdout-check` — verifies `<corpus_dir>/holdout.lock`
//! against the manifest and on-disk files. Mirrors
//! `scripts/check-holdout.sh`, plus a manifest cross-check.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use clap::Args as ClapArgs;

use crate::hashing::sha256_hex;
use crate::manifest::{self, Split};

/// Arguments for `corpus-tool holdout-check`.
#[derive(Debug, ClapArgs)]
pub struct Args {
    /// Corpus root directory.
    #[arg(long, default_value = "corpus")]
    pub corpus_dir: PathBuf,
}

struct LockLine {
    id: String,
    sha256: String,
    relpath: String,
}

/// Runs `holdout-check`.
///
/// Parses `<corpus_dir>/holdout.lock` (tab-separated `id`, `sha256`,
/// `relpath` — `relpath` is relative to the current working directory, so
/// run this from the repository root, matching
/// `scripts/check-holdout.sh`). Then checks: every line's file exists
/// with a matching hash; the manifest record for that id is
/// `split: holdout` with the same sha256; and every manifest record with
/// `split: holdout` is present in the lock. An absent lock file is a
/// no-op success, so CI stays green before the holdout is sealed.
///
/// # Errors
///
/// Returns an error if the lock file is malformed, the manifest can't be
/// read, or any drift is detected.
pub fn run(args: &Args) -> anyhow::Result<()> {
    let lock_path = args.corpus_dir.join("holdout.lock");
    let Ok(lock_text) = std::fs::read_to_string(&lock_path) else {
        println!("holdout lock absent - skipping");
        return Ok(());
    };

    let mut lines = Vec::new();
    for (idx, raw) in lock_text.lines().enumerate() {
        let trimmed = raw.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let mut parts = raw.split('\t');
        let (Some(id), Some(sha256), Some(path_field), None) =
            (parts.next(), parts.next(), parts.next(), parts.next())
        else {
            anyhow::bail!(
                "{}:{}: malformed line (expected id<TAB>sha256<TAB>path)",
                lock_path.display(),
                idx + 1
            );
        };
        lines.push(LockLine {
            id: id.to_string(),
            sha256: sha256.to_string(),
            relpath: path_field.to_string(),
        });
    }

    let manifest_path = args.corpus_dir.join("manifest.jsonl");
    let records = manifest::read_manifest(&manifest_path)?.unwrap_or_default();
    let by_id: BTreeMap<&str, _> = records.iter().map(|r| (r.id.as_str(), r)).collect();

    let mut failures = Vec::new();
    let mut seen_ids = BTreeSet::new();

    for line in &lines {
        seen_ids.insert(line.id.clone());

        match std::fs::read(&line.relpath) {
            Ok(bytes) => {
                let actual = sha256_hex(&bytes);
                if actual != line.sha256 {
                    failures.push(format!(
                        "{}: file hash mismatch (lock {}, file {actual})",
                        line.id, line.sha256
                    ));
                }
            }
            Err(_) => failures.push(format!("{}: missing file {}", line.id, line.relpath)),
        }

        match by_id.get(line.id.as_str()) {
            None => failures.push(format!("{}: not present in manifest", line.id)),
            Some(record) => {
                if record.split != Some(Split::Holdout) {
                    failures.push(format!("{}: manifest split is not holdout", line.id));
                }
                if record.sha256 != line.sha256 {
                    failures.push(format!(
                        "{}: manifest sha256 ({}) != lock sha256 ({})",
                        line.id, record.sha256, line.sha256
                    ));
                }
            }
        }
    }

    for record in records
        .iter()
        .filter(|record| record.split == Some(Split::Holdout))
    {
        if !seen_ids.contains(&record.id) {
            failures.push(format!(
                "{}: holdout doc missing from holdout.lock",
                record.id
            ));
        }
    }

    if !failures.is_empty() {
        for failure in &failures {
            eprintln!("error: {failure}");
        }
        anyhow::bail!("holdout check failed: {} mismatch(es)", failures.len());
    }

    println!("holdout check passed: {} line(s) verified", lines.len());
    Ok(())
}
