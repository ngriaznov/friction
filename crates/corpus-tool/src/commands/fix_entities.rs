//! `corpus-tool fix-entities` — maintenance pass over already-ingested docs.
//!
//! Reprocesses every corpus doc still carrying raw (un-decoded) HTML
//! entities left behind by ingestion before entity decoding was added to
//! `clean`'s cleaning pipeline (see `crate::commands::clean::decode_entities`),
//! and refreshes the manifest's `sha256` field to match.

use std::path::PathBuf;

use anyhow::Context;
use clap::Args as ClapArgs;

use crate::commands::clean::decode_entities;
use crate::corpus_layout::relpath;
use crate::hashing::sha256_hex;
use crate::manifest;

/// Arguments for `corpus-tool fix-entities`.
#[derive(Debug, ClapArgs)]
pub struct Args {
    /// Corpus root directory.
    #[arg(long, default_value = "corpus")]
    pub corpus_dir: PathBuf,
    /// Report which docs would change without writing anything (files or
    /// the manifest).
    #[arg(long)]
    pub dry_run: bool,
}

/// Runs `fix-entities`.
///
/// For every manifest record (spanning `human`, `llm`, and `quarantine`,
/// since a record's own class/license decides which of those its file
/// lives under — see `crate::corpus_layout::relpath`), reads its on-disk
/// file and decodes raw HTML entities with the exact transform `clean`
/// applies to newly-ingested docs. Only when decoding actually changes the
/// bytes does this rewrite the file (in place, same path) and update that
/// record's `sha256` field. Every other manifest field — id, class, genre,
/// split, license, path — is left untouched, and records whose file
/// already has no entities to decode are not written at all.
///
/// Deterministic and idempotent: decoding is a pure function of the
/// file's bytes, so running this twice in a row leaves the second run
/// with nothing to change.
///
/// # Errors
///
/// Returns an error if the manifest can't be read, a record's file can't
/// be read or (outside `--dry-run`) written, or the manifest can't be
/// written back.
pub fn run(args: &Args) -> anyhow::Result<()> {
    let manifest_path = args.corpus_dir.join("manifest.jsonl");
    let Some(mut records) = manifest::read_manifest(&manifest_path)? else {
        println!("empty corpus");
        return Ok(());
    };

    let mut changed: Vec<(String, PathBuf)> = Vec::new();

    for record in &mut records {
        let path = args.corpus_dir.join(relpath(record));
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("fix-entities: reading {}", path.display()))?;
        let decoded = decode_entities(&text);
        if decoded == text {
            continue;
        }

        changed.push((record.id.clone(), path.clone()));
        if !args.dry_run {
            std::fs::write(&path, &decoded)
                .with_context(|| format!("fix-entities: writing {}", path.display()))?;
            record.sha256 = sha256_hex(decoded.as_bytes());
        }
    }

    changed.sort();

    if !args.dry_run && !changed.is_empty() {
        manifest::write_manifest(&manifest_path, &records)?;
    }

    let verb = if args.dry_run {
        "would change"
    } else {
        "changed"
    };
    println!("fix-entities: {} doc(s) {verb}", changed.len());
    for (id, path) in &changed {
        println!("  {id} ({})", path.display());
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::{Class, Genre, ManifestRecord};

    fn record(id: &str, sha256: &str) -> ManifestRecord {
        ManifestRecord {
            id: id.to_string(),
            class: Class::Human,
            genre: Genre::Forum,
            source: "https://example.com".to_string(),
            model: None,
            prompt_id: None,
            license: "CC-BY-SA-4.0".to_string(),
            lang: "en".to_string(),
            split: None,
            sha256: sha256.to_string(),
            provenance_evidence: Some("test".to_string()),
            style_prompted: false,
            gen_config: None,
        }
    }

    fn setup(dir: &std::path::Path, id: &str, body: &str) -> ManifestRecord {
        let body_bytes = body.as_bytes();
        let record = record(id, &sha256_hex(body_bytes));
        manifest::write_manifest(&dir.join("manifest.jsonl"), std::slice::from_ref(&record))
            .unwrap();
        let path = dir.join(relpath(&record));
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, body).unwrap();
        record
    }

    /// A doc containing entities is rewritten in place with them decoded,
    /// and its manifest `sha256` is updated to match the new bytes; every
    /// other field is untouched.
    #[test]
    fn fix_entities_decodes_file_and_updates_sha256() {
        let dir = tempfile::tempdir().unwrap();
        let original = setup(
            dir.path(),
            "affected00000001",
            "doesn&#39;t &amp; won&apos;t\n",
        );

        let args = Args {
            corpus_dir: dir.path().to_path_buf(),
            dry_run: false,
        };
        run(&args).unwrap();

        let path = dir.path().join(relpath(&original));
        let body = std::fs::read_to_string(&path).unwrap();
        assert_eq!(body, "doesn't & won't\n");

        let records = manifest::read_manifest(&dir.path().join("manifest.jsonl"))
            .unwrap()
            .unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].sha256, sha256_hex(body.as_bytes()));
        assert_ne!(records[0].sha256, original.sha256);
        assert_eq!(records[0].id, original.id);
        assert_eq!(records[0].class, original.class);
        assert_eq!(records[0].genre, original.genre);
        assert_eq!(records[0].license, original.license);
    }

    /// A doc with no entities is left byte-identical and its manifest
    /// line (including `sha256`) is untouched.
    #[test]
    fn fix_entities_leaves_unaffected_doc_untouched() {
        let dir = tempfile::tempdir().unwrap();
        let original = setup(dir.path(), "clean0000000001", "Nothing to decode here.\n");

        let args = Args {
            corpus_dir: dir.path().to_path_buf(),
            dry_run: false,
        };
        run(&args).unwrap();

        let records = manifest::read_manifest(&dir.path().join("manifest.jsonl"))
            .unwrap()
            .unwrap();
        assert_eq!(records[0].sha256, original.sha256);
    }

    /// `--dry-run` reports what would change but writes neither the file
    /// nor the manifest.
    #[test]
    fn fix_entities_dry_run_writes_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let original = setup(dir.path(), "affected00000002", "a &amp; b\n");

        let args = Args {
            corpus_dir: dir.path().to_path_buf(),
            dry_run: true,
        };
        run(&args).unwrap();

        let path = dir.path().join(relpath(&original));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "a &amp; b\n");
        let records = manifest::read_manifest(&dir.path().join("manifest.jsonl"))
            .unwrap()
            .unwrap();
        assert_eq!(records[0].sha256, original.sha256);
    }

    /// Running twice in a row is idempotent: the second run finds nothing
    /// left to change, and doesn't rewrite the manifest.
    #[test]
    fn fix_entities_is_idempotent_across_reruns() {
        let dir = tempfile::tempdir().unwrap();
        let original = setup(dir.path(), "affected00000003", "doesn&#39;t stop&amp;go\n");

        let args = Args {
            corpus_dir: dir.path().to_path_buf(),
            dry_run: false,
        };
        run(&args).unwrap();
        let after_first = manifest::read_manifest(&dir.path().join("manifest.jsonl"))
            .unwrap()
            .unwrap();

        run(&args).unwrap();
        let after_second = manifest::read_manifest(&dir.path().join("manifest.jsonl"))
            .unwrap()
            .unwrap();

        assert_eq!(after_first, after_second);
        assert_ne!(after_first[0].sha256, original.sha256);
    }
}
