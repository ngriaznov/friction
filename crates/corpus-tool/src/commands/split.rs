//! `corpus-tool split` — deterministic stratified 70/15/15 split.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use clap::Args as ClapArgs;

use crate::hashing::sha256_hex;
use crate::manifest::{self, Class, Genre, Split};

/// Arguments for `corpus-tool split`.
#[derive(Debug, ClapArgs)]
pub struct Args {
    /// Corpus root directory.
    #[arg(long, default_value = "corpus")]
    pub corpus_dir: PathBuf,
    /// Compute and print the split without writing the manifest.
    #[arg(long)]
    pub dry_run: bool,
}

/// Runs `split`.
///
/// Assigns `train`/`dev`/`holdout` per `(class, genre)` cell. Within each
/// cell, candidates are ordered by `sha256(id)` hex (ascending) and
/// sliced at the 70%/85% boundaries — fully deterministic, no RNG.
///
/// Docs already sealed as `holdout` are never reassigned: if the freshly
/// computed holdout slice for a cell would move any doc into or out of an
/// already-sealed holdout set, the run fails instead of silently
/// reassigning.
///
/// # Errors
///
/// Returns an error if the manifest can't be read or written, or if
/// sealed holdout membership would have to change.
pub fn run(args: &Args) -> anyhow::Result<()> {
    let manifest_path = args.corpus_dir.join("manifest.jsonl");
    let Some(mut records) = manifest::read_manifest(&manifest_path)? else {
        println!("empty corpus");
        return Ok(());
    };

    let mut cells: BTreeMap<(Class, Genre), Vec<usize>> = BTreeMap::new();
    for (idx, record) in records.iter().enumerate() {
        cells
            .entry((record.class, record.genre))
            .or_default()
            .push(idx);
    }

    let mut assignments: BTreeMap<String, Split> = BTreeMap::new();

    for (&(class, genre), indices) in &cells {
        let mut ordered = indices.clone();
        ordered.sort_by_key(|&idx| sha256_hex(records[idx].id.as_bytes()));

        let n = ordered.len();
        let train_end = n * 70 / 100;
        let dev_end = n * 85 / 100;

        let sealed: BTreeSet<&str> = indices
            .iter()
            .filter_map(|&idx| {
                (records[idx].split == Some(Split::Holdout)).then_some(records[idx].id.as_str())
            })
            .collect();

        let computed_holdout: BTreeSet<&str> = ordered[dev_end..]
            .iter()
            .map(|&idx| records[idx].id.as_str())
            .collect();

        if !sealed.is_empty() && sealed != computed_holdout {
            anyhow::bail!(
                "split: holdout membership for ({class}, {genre}) would change; sealed docs \
                 are never reassigned (sealed={sealed:?}, freshly computed={computed_holdout:?})"
            );
        }

        for (slot, &idx) in ordered.iter().enumerate() {
            let split = if slot < train_end {
                Split::Train
            } else if slot < dev_end {
                Split::Dev
            } else {
                Split::Holdout
            };
            assignments.insert(records[idx].id.clone(), split);
        }
    }

    for record in &mut records {
        if let Some(&split) = assignments.get(&record.id) {
            record.split = Some(split);
        }
    }

    if args.dry_run {
        for record in &records {
            println!(
                "{}\t{}\t{}\t{}",
                record.id,
                record.class,
                record.genre,
                record
                    .split
                    .map_or_else(|| "none".to_string(), |s| s.to_string())
            );
        }
        return Ok(());
    }

    manifest::write_manifest(&manifest_path, &records)?;
    println!(
        "split: wrote {} record(s) to {}",
        records.len(),
        manifest_path.display()
    );
    Ok(())
}
