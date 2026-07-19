//! `corpus-tool remove` — drops one or more docs from the corpus.
//!
//! Deletes the manifest record and its corpus file, leaving the raw
//! original under `corpus/incoming/` untouched (removal is a corpus
//! decision, not a retraction of what was collected).

use std::collections::BTreeSet;
use std::path::PathBuf;

use anyhow::Context;
use clap::Args as ClapArgs;

use crate::corpus_layout::relpath;
use crate::manifest;

/// Arguments for `corpus-tool remove`.
#[derive(Debug, ClapArgs)]
pub struct Args {
    /// Corpus root directory.
    #[arg(long, default_value = "corpus")]
    pub corpus_dir: PathBuf,
    /// Id of a manifest record to remove. Repeatable.
    #[arg(long = "id", required = true)]
    pub ids: Vec<String>,
}

/// Runs `remove`.
///
/// Validates that every requested id is present in the manifest before
/// touching anything, so a typo'd id fails loudly with no partial
/// effect. For each removed record: deletes its corpus file
/// (`<class>/<genre>/<id>.md`, or `quarantine/<genre>/<id>.md` when
/// quarantined) and drops its manifest line; the corresponding raw doc
/// under `corpus/incoming/` is never touched, matching `ingest`'s
/// one-way flow from `incoming` into the corpus.
///
/// # Errors
///
/// Returns an error if the manifest can't be read or written, any
/// requested id is not in the manifest, or a record's file can't be
/// deleted for a reason other than already being absent.
pub fn run(args: &Args) -> anyhow::Result<()> {
    let manifest_path = args.corpus_dir.join("manifest.jsonl");
    let Some(mut records) = manifest::read_manifest(&manifest_path)? else {
        anyhow::bail!("no manifest at {}", manifest_path.display());
    };

    let requested: BTreeSet<&str> = args.ids.iter().map(String::as_str).collect();
    let missing: Vec<&str> = requested
        .iter()
        .copied()
        .filter(|id| !records.iter().any(|r| r.id == *id))
        .collect();
    if !missing.is_empty() {
        anyhow::bail!(
            "remove: id(s) not in manifest, nothing removed: {}",
            missing.join(", ")
        );
    }

    let mut removed = Vec::new();
    records.retain(|record| {
        if requested.contains(record.id.as_str()) {
            removed.push(record.clone());
            false
        } else {
            true
        }
    });
    removed.sort_by(|a, b| a.id.cmp(&b.id));

    for record in &removed {
        let path = args.corpus_dir.join(relpath(record));
        match std::fs::remove_file(&path) {
            Ok(()) => {}
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
            Err(err) => {
                return Err(err).with_context(|| format!("removing {}", path.display()));
            }
        }
        println!("removed {} ({})", record.id, path.display());
    }

    manifest::write_manifest(&manifest_path, &records)?;
    println!("remove: {} doc(s) removed", removed.len());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::{Class, Genre, ManifestRecord};

    fn record(id: &str, genre: Genre) -> ManifestRecord {
        ManifestRecord {
            id: id.to_string(),
            class: Class::Human,
            genre,
            source: "https://example.com".to_string(),
            model: None,
            prompt_id: None,
            license: "MIT".to_string(),
            lang: "en".to_string(),
            split: None,
            sha256: "deadbeef".to_string(),
            provenance_evidence: Some("test".to_string()),
            style_prompted: false,
            gen_config: None,
        }
    }

    fn setup(dir: &std::path::Path, records: &[ManifestRecord]) {
        manifest::write_manifest(&dir.join("manifest.jsonl"), records).unwrap();
        for record in records {
            let path = dir.join(relpath(record));
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, "body").unwrap();
        }
    }

    /// Removing one id deletes both its manifest line and its corpus
    /// file, leaving the other record untouched.
    #[test]
    fn remove_deletes_file_and_manifest_line() {
        let dir = tempfile::tempdir().unwrap();
        let keep = record("keep0000000000ab", Genre::Blog);
        let gone = record("gone0000000000cd", Genre::Blog);
        setup(dir.path(), &[keep.clone(), gone.clone()]);

        let args = Args {
            corpus_dir: dir.path().to_path_buf(),
            ids: vec![gone.id.clone()],
        };
        run(&args).unwrap();

        let records = manifest::read_manifest(&dir.path().join("manifest.jsonl"))
            .unwrap()
            .unwrap();
        assert_eq!(records, vec![keep.clone()]);
        assert!(dir.path().join(relpath(&keep)).exists());
        assert!(!dir.path().join(relpath(&gone)).exists());
    }

    /// An id absent from the manifest fails the whole call with no
    /// partial effect: the file that *was* going to be removed alongside
    /// it stays put.
    #[test]
    fn remove_refuses_unknown_id_without_partial_effect() {
        let dir = tempfile::tempdir().unwrap();
        let present = record("present0000000ab", Genre::Docs);
        setup(dir.path(), std::slice::from_ref(&present));

        let args = Args {
            corpus_dir: dir.path().to_path_buf(),
            ids: vec![present.id.clone(), "doesnotexist0000".to_string()],
        };
        let err = run(&args).unwrap_err();
        assert!(err.to_string().contains("doesnotexist0000"));

        let records = manifest::read_manifest(&dir.path().join("manifest.jsonl"))
            .unwrap()
            .unwrap();
        assert_eq!(records, vec![present.clone()]);
        assert!(dir.path().join(relpath(&present)).exists());
    }

    /// Removing a quarantined (CC-BY-SA) record deletes the file from
    /// `quarantine/<genre>/`, not `human/<genre>/`.
    #[test]
    fn remove_deletes_quarantined_file_from_quarantine_dir() {
        let dir = tempfile::tempdir().unwrap();
        let mut forum = record("forum000000000ab", Genre::Forum);
        forum.license = "CC-BY-SA-4.0".to_string();
        setup(dir.path(), &[forum.clone()]);

        let args = Args {
            corpus_dir: dir.path().to_path_buf(),
            ids: vec![forum.id.clone()],
        };
        run(&args).unwrap();

        assert!(
            !dir.path()
                .join("quarantine/forum")
                .join(format!("{}.md", forum.id))
                .exists()
        );
    }
}
