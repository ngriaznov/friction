//! Shared test helpers for `corpus-tool` integration tests.
//!
//! Not every helper is used by every test binary that includes this
//! module (each `tests/*.rs` file compiles as its own crate), so
//! unused-item warnings are expected and suppressed here rather than
//! per test file.
#![allow(dead_code)]

use std::path::Path;

use corpus_tool::hashing::sha256_hex;
use corpus_tool::manifest::{Class, Genre, ManifestRecord, ModelInfo};

/// Repeats a small filler vocabulary to produce exactly `n` whitespace
/// tokens, so tests can hit exact word-count boundaries deterministically
/// without depending on hand-typed prose.
pub fn filler_words(n: usize) -> String {
    const VOCAB: [&str; 8] = [
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
    ];
    (0..n)
        .map(|i| VOCAB[i % VOCAB.len()])
        .collect::<Vec<_>>()
        .join(" ")
}

/// Writes `content` to `dir/relpath`, creating parent directories as
/// needed, and returns the file's sha256 hex digest.
pub fn write_doc(dir: &Path, relpath: &str, content: &str) -> String {
    let path = dir.join(relpath);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("create fixture parent dir");
    }
    std::fs::write(&path, content).expect("write fixture doc");
    sha256_hex(content.as_bytes())
}

/// A minimal valid `human`-class record: `license: "personal-attestation"`
/// (satisfies the provenance rule without needing `provenance_evidence`).
pub fn human_record(id: &str, genre: Genre, sha256: String) -> ManifestRecord {
    ManifestRecord {
        id: id.to_string(),
        class: Class::Human,
        genre,
        source: "test-fixture".to_string(),
        model: None,
        prompt_id: None,
        license: "personal-attestation".to_string(),
        lang: "en".to_string(),
        split: None,
        sha256,
        provenance_evidence: None,
        style_prompted: false,
        gen_config: None,
    }
}

/// A minimal valid `llm`-class record: model + `prompt_id` + `gen_config`
/// all populated (satisfies the required fields for an `llm` record).
pub fn llm_record(id: &str, genre: Genre, sha256: String) -> ManifestRecord {
    ManifestRecord {
        id: id.to_string(),
        class: Class::Llm,
        genre,
        source: "test-fixture".to_string(),
        model: Some(ModelInfo {
            name: "qwen2.5-7b-instruct".to_string(),
            quantization: "q4_k_m".to_string(),
        }),
        prompt_id: Some("p001".to_string()),
        license: "generated".to_string(),
        lang: "en".to_string(),
        split: None,
        sha256,
        provenance_evidence: None,
        style_prompted: false,
        gen_config: Some(serde_json::json!({
            "model_digest": "sha256:deadbeef",
            "temperature": 0.7,
            "seed": 42,
            "reproducible": true
        })),
    }
}

/// Writes a JSONL manifest file from `records` (unsorted, as authored —
/// distinct from `corpus_tool::manifest::write_manifest`'s sorted output,
/// so tests can exercise input-order independence).
pub fn write_manifest_raw(path: &Path, records: &[ManifestRecord]) {
    let lines: Vec<String> = records
        .iter()
        .map(|r| serde_json::to_string(r).expect("serialize fixture record"))
        .collect();
    std::fs::write(path, lines.join("\n") + "\n").expect("write fixture manifest");
}
