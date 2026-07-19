//! Corpus manifest schema and JSONL (de)serialization.
//!
//! One JSON object per line in `<corpus_dir>/manifest.jsonl`, strictly
//! parsed (`deny_unknown_fields`) so a typo'd field fails loudly instead
//! of silently vanishing.

use std::fmt;
use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// License string that satisfies the human-provenance rule for
/// docs with no external provenance evidence: the contributor personally
/// attests to authorship and date.
pub const PERSONAL_ATTESTATION_LICENSE: &str = "personal-attestation";

/// Document class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Class {
    Human,
    Llm,
}

impl fmt::Display for Class {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Human => "human",
            Self::Llm => "llm",
        })
    }
}

/// Document genre; the fixed set of five genres used by the corpus.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Genre {
    Docs,
    Blog,
    Readme,
    Email,
    Forum,
}

impl fmt::Display for Genre {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Docs => "docs",
            Self::Blog => "blog",
            Self::Readme => "readme",
            Self::Email => "email",
            Self::Forum => "forum",
        })
    }
}

/// Frozen train/dev/holdout split label.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Split {
    Train,
    Dev,
    Holdout,
}

impl fmt::Display for Split {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Train => "train",
            Self::Dev => "dev",
            Self::Holdout => "holdout",
        })
    }
}

/// LLM generation model identity: "model name + quantization".
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelInfo {
    pub name: String,
    pub quantization: String,
}

/// One manifest record: one row of `<corpus_dir>/manifest.jsonl`.
///
/// `deny_unknown_fields` makes an unrecognized field a hard parse error
/// rather than a silently ignored typo.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ManifestRecord {
    pub id: String,
    pub class: Class,
    pub genre: Genre,
    pub source: String,
    /// `llm`-only: model name + quantization. Nullable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<ModelInfo>,
    /// `llm`-only: which prompt (from `corpus/prompts/*.toml`)
    /// generated this doc. Nullable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_id: Option<String>,
    pub license: String,
    /// BCP-47 language tag; `"en"` only in v1.
    pub lang: String,
    /// Nullable until assigned by `split`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub split: Option<Split>,
    pub sha256: String,
    /// Human-corpus cutoff evidence: archive.org timestamp, git commit
    /// date, publication date. Nullable — human docs may instead
    /// carry `license: "personal-attestation"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance_evidence: Option<String>,
    /// True if this doc was generated with a "sound human" style
    /// instruction; defaults to false.
    #[serde(default)]
    pub style_prompted: bool,
    /// `llm`-only: full generation config (model digest, sampler params,
    /// seed, reproducible flag). Nullable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gen_config: Option<serde_json::Value>,
}

/// Errors reading or parsing `<corpus_dir>/manifest.jsonl`.
#[derive(Debug, thiserror::Error)]
pub enum ManifestError {
    #[error("failed to read {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("failed to write {path}: {source}")]
    Write {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("{path}:{line}: {source}")]
    Parse {
        path: PathBuf,
        line: usize,
        #[source]
        source: serde_json::Error,
    },
}

/// Reads and strictly parses `<corpus_dir>/manifest.jsonl`: one JSON
/// object per non-blank line, `deny_unknown_fields`.
///
/// Returns `Ok(None)` if `path` does not exist at all — an absent manifest
/// means an empty corpus, which every subcommand treats as trivially
/// valid rather than an error.
///
/// # Errors
///
/// Returns [`ManifestError::Io`] on any I/O failure other than "not
/// found", or [`ManifestError::Parse`] on the first line that fails to
/// parse as a [`ManifestRecord`].
pub fn read_manifest(path: &Path) -> Result<Option<Vec<ManifestRecord>>, ManifestError> {
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
        let record: ManifestRecord =
            serde_json::from_str(line).map_err(|source| ManifestError::Parse {
                path: path.to_path_buf(),
                line: idx + 1,
                source,
            })?;
        records.push(record);
    }
    Ok(Some(records))
}

/// Writes `records` as JSONL, one compact JSON object per line, sorted by
/// `id` first for deterministic byte output regardless of input
/// order.
///
/// # Errors
///
/// Returns [`ManifestError::Write`] if `path` cannot be written.
pub fn write_manifest(path: &Path, records: &[ManifestRecord]) -> Result<(), ManifestError> {
    let mut sorted: Vec<&ManifestRecord> = records.iter().collect();
    sorted.sort_by(|a, b| a.id.cmp(&b.id));

    let mut out = String::new();
    for record in sorted {
        let line =
            serde_json::to_string(record).expect("ManifestRecord serialization is infallible");
        out.push_str(&line);
        out.push('\n');
    }

    std::fs::write(path, out).map_err(|source| ManifestError::Write {
        path: path.to_path_buf(),
        source,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_human_json() -> &'static str {
        r#"{"id":"h1","class":"human","genre":"docs","source":"github","license":"MIT","lang":"en","sha256":"abc","provenance_evidence":"git commit 2019-01-01"}"#
    }

    /// A minimal well-formed human record round-trips through
    /// serde with the optional fields defaulting to `None`/`false`.
    #[test]
    fn manifest_record_parses_minimal_human_record() {
        let record: ManifestRecord = serde_json::from_str(sample_human_json()).unwrap();
        assert_eq!(record.id, "h1");
        assert_eq!(record.class, Class::Human);
        assert_eq!(record.genre, Genre::Docs);
        assert!(record.model.is_none());
        assert!(record.prompt_id.is_none());
        assert!(record.split.is_none());
        assert!(!record.style_prompted);
        assert!(record.gen_config.is_none());
    }

    /// An unrecognized field is a hard parse error
    /// (`deny_unknown_fields`), not a silently dropped typo.
    #[test]
    fn manifest_record_rejects_unknown_field() {
        let json = r#"{"id":"h1","class":"human","genre":"docs","source":"github","license":"MIT","lang":"en","sha256":"abc","oops":"typo"}"#;
        let result: Result<ManifestRecord, _> = serde_json::from_str(json);
        assert!(result.is_err());
    }

    /// An `llm` record carries `model` + `prompt_id` + `gen_config`.
    #[test]
    fn manifest_record_parses_llm_record_with_model_and_gen_config() {
        let json = r#"{"id":"l1","class":"llm","genre":"blog","source":"ollama","model":{"name":"qwen2.5-7b-instruct","quantization":"q4_k_m"},"prompt_id":"p1","license":"generated","lang":"en","sha256":"def","gen_config":{"seed":1,"temperature":0.7}}"#;
        let record: ManifestRecord = serde_json::from_str(json).unwrap();
        assert_eq!(record.class, Class::Llm);
        let model = record.model.expect("llm record has a model");
        assert_eq!(model.name, "qwen2.5-7b-instruct");
        assert_eq!(model.quantization, "q4_k_m");
        assert_eq!(record.prompt_id.as_deref(), Some("p1"));
        assert!(record.gen_config.is_some());
    }

    /// `write_manifest` output is sorted by id regardless of input
    /// order, so the same record set always serializes to identical bytes.
    #[test]
    fn write_manifest_sorts_by_id_for_determinism() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("manifest.jsonl");

        let mut a: ManifestRecord = serde_json::from_str(sample_human_json()).unwrap();
        a.id = "zzz".to_string();
        let mut b: ManifestRecord = serde_json::from_str(sample_human_json()).unwrap();
        b.id = "aaa".to_string();

        write_manifest(&path, &[a, b]).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("\"aaa\""));
        assert!(lines[1].contains("\"zzz\""));
    }

    /// A missing manifest file is `Ok(None)`, not an error — an
    /// absent corpus is a valid (empty) state.
    #[test]
    fn read_manifest_missing_file_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("does-not-exist.jsonl");
        assert!(read_manifest(&path).unwrap().is_none());
    }

    /// Blank lines between records are skipped rather than
    /// producing a parse error.
    #[test]
    fn read_manifest_skips_blank_lines() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("manifest.jsonl");
        std::fs::write(
            &path,
            format!("{}\n\n{}\n", sample_human_json(), sample_human_json()),
        )
        .unwrap();
        let records = read_manifest(&path).unwrap().unwrap();
        assert_eq!(records.len(), 2);
    }
}
