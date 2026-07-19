//! `corpus-tool seal` — freeze the holdout split.

use std::fmt::Write as _;
use std::io::ErrorKind;
use std::path::PathBuf;

use anyhow::Context;
use clap::Args as ClapArgs;

use crate::corpus_layout::relpath;
use crate::hashing::sha256_hex;
use crate::manifest::{self, Split};

/// Arguments for `corpus-tool seal`.
#[derive(Debug, ClapArgs)]
pub struct Args {
    /// Corpus root directory.
    #[arg(long, default_value = "corpus")]
    pub corpus_dir: PathBuf,
    /// Overwrite an existing lock file even if its content would change.
    /// Only needed the first time the holdout split is frozen.
    #[arg(long)]
    pub init: bool,
}

/// Runs `seal`.
///
/// Writes `<corpus_dir>/holdout.lock`, one `id<TAB>sha256<TAB>relpath`
/// line per holdout doc (`split: holdout`), sorted by id. The sha256
/// written is always recomputed from the file's actual on-disk bytes (as
/// `validate` does) rather than trusted from the manifest field: sealing
/// is precisely the operation meant to freeze byte-exact holdout state, so
/// it must not risk freezing a stale hash if the manifest wasn't
/// re-validated since a doc was last touched. Refuses to overwrite an
/// existing lock whose content would differ unless `--init` is passed, so
/// an accidental re-seal after the holdout has drifted requires an
/// explicit, visible opt-in.
///
/// # Errors
///
/// Returns an error if the manifest can't be read, a holdout doc's file
/// can't be read (missing or unreadable), its on-disk sha256 doesn't match
/// the manifest's recorded sha256 (the manifest is stale — rerun
/// `validate` first), or the lock file can't be read or written, or an
/// existing lock would change without `--init`.
pub fn run(args: &Args) -> anyhow::Result<()> {
    let manifest_path = args.corpus_dir.join("manifest.jsonl");
    let Some(records) = manifest::read_manifest(&manifest_path)? else {
        println!("empty corpus");
        return Ok(());
    };

    let mut holdout: Vec<_> = records
        .iter()
        .filter(|record| record.split == Some(Split::Holdout))
        .collect();
    holdout.sort_by(|a, b| a.id.cmp(&b.id));

    let mut content = String::new();
    for record in &holdout {
        let rel = args.corpus_dir.join(relpath(record));
        let bytes = std::fs::read(&rel)
            .with_context(|| format!("seal: reading holdout doc {}", rel.display()))?;
        let actual_sha = sha256_hex(&bytes);
        if actual_sha != record.sha256 {
            anyhow::bail!(
                "seal: {}: on-disk sha256 ({actual_sha}) != manifest sha256 ({}); \
                 the manifest is stale — rerun `corpus-tool validate` before sealing",
                record.id,
                record.sha256,
            );
        }
        writeln!(content, "{}\t{}\t{}", record.id, actual_sha, rel.display())
            .expect("write to String is infallible");
    }

    let lock_path = args.corpus_dir.join("holdout.lock");
    let existing = match std::fs::read_to_string(&lock_path) {
        Ok(text) => Some(text),
        Err(err) if err.kind() == ErrorKind::NotFound => None,
        Err(err) => return Err(err.into()),
    };

    match existing {
        Some(text) if text == content => {
            println!("holdout.lock up to date ({} doc(s))", holdout.len());
        }
        Some(_) if !args.init => {
            anyhow::bail!(
                "holdout.lock exists and its content would change; rerun with --init to \
                 confirm (holdout is sealed and should not normally change)"
            );
        }
        _ => {
            std::fs::write(&lock_path, &content)?;
            println!("holdout.lock written ({} doc(s))", holdout.len());
        }
    }

    Ok(())
}
