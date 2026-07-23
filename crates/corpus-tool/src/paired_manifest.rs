//! `corpus/paired/manifest.jsonl` schema and JSONL (de)serialization —
//! the manifest for the stock/antislop paired mining corpus (see
//! `commands::generate_paired`).
//!
//! Structurally unrelated to `crate::manifest::ManifestRecord`: no
//! `split`/`license`/`provenance_evidence`/`style_prompted` field, none
//! of which apply to a corpus that is never split train/dev/holdout and
//! never carries provenance claims of its own. `deny_unknown_fields`,
//! matching the rest of this crate's manifest-strictness convention.

use std::fmt;
use std::io;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::manifest::{Genre, ManifestError, ModelInfo};

/// Which side of a stock/antislop pair one record is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Side {
    Stock,
    Antislop,
}

impl fmt::Display for Side {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Stock => "stock",
            Self::Antislop => "antislop",
        })
    }
}

/// One record of `corpus/paired/manifest.jsonl`: one generated side of
/// one `(genre, prompt)` pair.
///
/// `deny_unknown_fields` makes an unrecognized field a hard parse error
/// rather than a silently ignored typo.
///
/// `PartialEq` only (not `Eq`): `temperature` is an `f64`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PairedManifestRecord {
    /// Shared across both sides of one `(genre, prompt)` pair — see
    /// `commands::generate_paired::derive_pair_id`.
    pub pair_id: String,
    pub side: Side,
    pub genre: Genre,
    pub model: ModelInfo,
    pub model_digest: String,
    pub prompt_id: String,
    pub seed: u64,
    pub temperature: f64,
    pub num_predict: u32,
    /// Honestly computed by regenerating once more with the identical
    /// config and comparing output bytes — see
    /// `commands::generate_paired::run_one_paired_job` — never a
    /// hardcoded `true`.
    pub reproducible: bool,
    pub sha256: String,
}

/// Reads and strictly parses `corpus/paired/manifest.jsonl`: one JSON
/// object per non-blank line, `deny_unknown_fields`.
///
/// Returns `Ok(None)` if `path` does not exist at all — an absent
/// manifest means an empty paired corpus, valid rather than an error.
///
/// # Errors
///
/// Returns [`ManifestError::Io`] on any I/O failure other than "not
/// found", or [`ManifestError::Parse`] on the first line that fails to
/// parse as a [`PairedManifestRecord`].
pub fn read_paired_manifest(
    path: &Path,
) -> Result<Option<Vec<PairedManifestRecord>>, ManifestError> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(source) => {
            return Err(ManifestError::Io {
                path: path.to_path_buf(),
                source,
            });
        }
    };

    let mut records = Vec::new();
    for (idx, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let record: PairedManifestRecord =
            serde_json::from_str(line).map_err(|source| ManifestError::Parse {
                path: path.to_path_buf(),
                line: idx + 1,
                source,
            })?;
        records.push(record);
    }
    Ok(Some(records))
}

/// Writes `records` as JSONL, one compact JSON object per line.
///
/// Sorted by `(pair_id, side)` first for deterministic byte output
/// regardless of input order — mirrors `crate::manifest::write_manifest`'s
/// own determinism convention.
///
/// # Errors
///
/// Returns [`ManifestError::Write`] if `path` cannot be written.
pub fn write_paired_manifest(
    path: &Path,
    records: &[PairedManifestRecord],
) -> Result<(), ManifestError> {
    let mut sorted: Vec<&PairedManifestRecord> = records.iter().collect();
    sorted.sort_by(|a, b| (a.pair_id.as_str(), a.side).cmp(&(b.pair_id.as_str(), b.side)));

    let mut out = String::new();
    for record in sorted {
        let line = serde_json::to_string(record)
            .expect("PairedManifestRecord serialization is infallible");
        out.push_str(&line);
        out.push('\n');
    }

    std::fs::write(path, out).map_err(|source| ManifestError::Write {
        path: path.to_path_buf(),
        source,
    })
}

/// Path to a paired record's document, relative to `corpus/paired/`:
/// `<side>/<genre>/<pair_id>.md`.
pub fn paired_relpath(record: &PairedManifestRecord) -> String {
    format!("{}/{}/{}.md", record.side, record.genre, record.pair_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::Genre;

    fn sample_record() -> PairedManifestRecord {
        PairedManifestRecord {
            pair_id: "abc123".to_string(),
            side: Side::Stock,
            genre: Genre::Docs,
            model: ModelInfo {
                name: "gemma3:12b".to_string(),
                quantization: "Q4_K_M".to_string(),
            },
            model_digest: "sha256:deadbeef".to_string(),
            prompt_id: "docs-001".to_string(),
            seed: 42,
            temperature: 0.7,
            num_predict: 1536,
            reproducible: true,
            sha256: "cafef00d".to_string(),
        }
    }

    /// A minimal well-formed record round-trips through serde.
    #[test]
    fn paired_manifest_record_round_trips() {
        let record = sample_record();
        let json = serde_json::to_string(&record).unwrap();
        let parsed: PairedManifestRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, record);
    }

    /// An unrecognized field is a hard parse error, not a silently
    /// dropped typo.
    #[test]
    fn paired_manifest_record_rejects_unknown_field() {
        let json = r#"{"pair_id":"a","side":"stock","genre":"docs","model":{"name":"m","quantization":"q"},"model_digest":"d","prompt_id":"p","seed":1,"temperature":0.7,"num_predict":10,"reproducible":true,"sha256":"s","oops":"typo"}"#;
        let result: Result<PairedManifestRecord, _> = serde_json::from_str(json);
        assert!(result.is_err());
    }

    /// `write_paired_manifest` sorts by `(pair_id, side)` regardless of
    /// input order, so the same record set always serializes to
    /// identical bytes.
    #[test]
    fn write_paired_manifest_sorts_by_pair_id_and_side_for_determinism() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("manifest.jsonl");

        let mut antislop = sample_record();
        antislop.pair_id = "zzz".to_string();
        antislop.side = Side::Antislop;
        let mut stock_z = sample_record();
        stock_z.pair_id = "zzz".to_string();
        stock_z.side = Side::Stock;
        let mut stock_a = sample_record();
        stock_a.pair_id = "aaa".to_string();
        stock_a.side = Side::Stock;

        write_paired_manifest(&path, &[antislop, stock_z, stock_a]).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 3);
        assert!(lines[0].contains("\"aaa\""));
        assert!(lines[1].contains("\"zzz\"") && lines[1].contains("\"stock\""));
        assert!(lines[2].contains("\"zzz\"") && lines[2].contains("\"antislop\""));
    }

    /// A missing manifest file is `Ok(None)`, not an error.
    #[test]
    fn read_paired_manifest_missing_file_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("does-not-exist.jsonl");
        assert!(read_paired_manifest(&path).unwrap().is_none());
    }

    /// Blank lines between records are skipped.
    #[test]
    fn read_paired_manifest_skips_blank_lines() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("manifest.jsonl");
        let json = serde_json::to_string(&sample_record()).unwrap();
        std::fs::write(&path, format!("{json}\n\n{json}\n")).unwrap();
        let records = read_paired_manifest(&path).unwrap().unwrap();
        assert_eq!(records.len(), 2);
    }

    /// `paired_relpath` joins `side`/`genre`/`pair_id` in the expected shape.
    #[test]
    fn paired_relpath_joins_side_genre_and_pair_id() {
        assert_eq!(paired_relpath(&sample_record()), "stock/docs/abc123.md");
    }
}
