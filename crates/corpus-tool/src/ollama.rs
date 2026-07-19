//! Minimal Ollama HTTP client used by `corpus-tool generate`:
//! `/api/tags` (available models), `/api/show` (model digest +
//! quantization), `/api/generate` (`stream: false`).
//!
//! Kept deliberately thin — request/response shapes are the small subset
//! `generate` actually needs, tolerant of the rest of Ollama's response
//! (no `deny_unknown_fields` here: this is an external API, not our
//! schema).

use std::collections::BTreeSet;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::hashing::sha256_hex;

/// Default per-request timeout when a genconfig doesn't set one.
pub const DEFAULT_TIMEOUT_SECS: u64 = 300;

/// An Ollama request failed (transport, HTTP status, or response
/// decoding).
#[derive(Debug, thiserror::Error)]
#[error("ollama request to {url} failed: {source}")]
pub struct OllamaError {
    pub url: String,
    #[source]
    pub source: ureq::Error,
}

/// Model identity + sampler parameters for one `/api/generate` call.
#[derive(Debug, Clone, Copy)]
pub struct GenerateParams {
    pub temperature: f64,
    pub seed: u64,
    pub num_predict: u32,
}

/// Digest + quantization for one model, as resolved from `/api/show`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelDigest {
    /// The model's content digest. Sourced from `/api/show`'s top-level
    /// `digest` field when present; if the running Ollama version omits
    /// it (observed on Ollama 0.32.x), falls back to a `sha256:`-prefixed
    /// hash computed over the response's `details` + `model_info`
    /// objects — still a stable, content-derived identifier for the
    /// exact model, just not Ollama's own blob digest.
    pub digest: String,
    /// `details.quantization_level` (e.g. `"Q4_K_M"`), or `"unknown"`.
    pub quantization: String,
}

/// A thin, blocking Ollama client bound to one `endpoint`.
pub struct OllamaClient {
    endpoint: String,
    agent: ureq::Agent,
}

impl OllamaClient {
    /// Builds a client for `endpoint` (e.g. `http://localhost:11434`)
    /// with a global per-request `timeout`.
    pub fn new(endpoint: impl Into<String>, timeout: Duration) -> Self {
        let config = ureq::Agent::config_builder()
            .timeout_global(Some(timeout))
            .build();
        Self {
            endpoint: endpoint.into(),
            agent: ureq::Agent::new_with_config(config),
        }
    }

    /// Names of models Ollama currently has pulled locally, via
    /// `GET /api/tags`.
    ///
    /// # Errors
    ///
    /// Returns [`OllamaError`] on any transport, HTTP-status, or decode
    /// failure.
    pub fn available_models(&self) -> Result<BTreeSet<String>, OllamaError> {
        let url = format!("{}/api/tags", self.endpoint);
        let mut resp = self.agent.get(&url).call().map_err(|source| OllamaError {
            url: url.clone(),
            source,
        })?;
        let body: TagsResponse = resp.body_mut().read_json().map_err(|source| OllamaError {
            url: url.clone(),
            source,
        })?;
        Ok(body.models.into_iter().map(|m| m.name).collect())
    }

    /// Fetches `model`'s digest + quantization via `POST /api/show`.
    /// Call once per model and cache the result — `generate`
    /// does not need to re-fetch it per job.
    ///
    /// # Errors
    ///
    /// Returns [`OllamaError`] on any transport, HTTP-status, or decode
    /// failure.
    pub fn show(&self, model: &str) -> Result<ModelDigest, OllamaError> {
        let url = format!("{}/api/show", self.endpoint);
        let mut resp = self
            .agent
            .post(&url)
            .send_json(serde_json::json!({ "model": model }))
            .map_err(|source| OllamaError {
                url: url.clone(),
                source,
            })?;
        let value: serde_json::Value =
            resp.body_mut().read_json().map_err(|source| OllamaError {
                url: url.clone(),
                source,
            })?;
        Ok(model_digest_from_show_response(&value))
    }

    /// Generates one completion via `POST /api/generate` with
    /// `stream: false`, returning the raw `response`
    /// text.
    ///
    /// # Errors
    ///
    /// Returns [`OllamaError`] on any transport, HTTP-status, or decode
    /// failure.
    pub fn generate(
        &self,
        model: &str,
        prompt: &str,
        params: GenerateParams,
    ) -> Result<String, OllamaError> {
        let url = format!("{}/api/generate", self.endpoint);
        let request = GenerateRequest {
            model,
            prompt,
            stream: false,
            options: GenerateOptions {
                temperature: params.temperature,
                seed: params.seed,
                num_predict: params.num_predict,
            },
        };
        let mut resp = self
            .agent
            .post(&url)
            .send_json(&request)
            .map_err(|source| OllamaError {
                url: url.clone(),
                source,
            })?;
        let body: GenerateResponse = resp.body_mut().read_json().map_err(|source| OllamaError {
            url: url.clone(),
            source,
        })?;
        Ok(body.response)
    }
}

/// Extracted so it's unit-testable without a live server.
fn model_digest_from_show_response(value: &serde_json::Value) -> ModelDigest {
    let digest = value
        .get("digest")
        .and_then(serde_json::Value::as_str)
        .map_or_else(
            || {
                let stable = serde_json::json!({
                    "details": value.get("details"),
                    "model_info": value.get("model_info"),
                });
                let bytes = serde_json::to_vec(&stable).unwrap_or_default();
                format!("sha256:{}", sha256_hex(&bytes))
            },
            str::to_string,
        );
    let quantization = value
        .get("details")
        .and_then(|d| d.get("quantization_level"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or("unknown")
        .to_string();
    ModelDigest {
        digest,
        quantization,
    }
}

#[derive(Debug, Deserialize)]
struct TagsResponse {
    #[serde(default)]
    models: Vec<TagsModel>,
}

#[derive(Debug, Deserialize)]
struct TagsModel {
    name: String,
}

#[derive(Debug, Serialize)]
struct GenerateRequest<'a> {
    model: &'a str,
    prompt: &'a str,
    stream: bool,
    options: GenerateOptions,
}

#[derive(Debug, Serialize)]
struct GenerateOptions {
    temperature: f64,
    seed: u64,
    num_predict: u32,
}

#[derive(Debug, Deserialize)]
struct GenerateResponse {
    #[serde(default)]
    response: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// When `/api/show` returns a top-level `digest`, it's used
    /// as-is.
    #[test]
    fn model_digest_prefers_top_level_digest_field() {
        let value = serde_json::json!({
            "digest": "sha256:abcdef",
            "details": { "quantization_level": "Q4_K_M" }
        });
        let digest = model_digest_from_show_response(&value);
        assert_eq!(digest.digest, "sha256:abcdef");
        assert_eq!(digest.quantization, "Q4_K_M");
    }

    /// When `/api/show` has no `digest` field (observed on Ollama
    /// 0.32.x), a stable content-derived fallback is computed instead of
    /// panicking or leaving it empty.
    #[test]
    fn model_digest_falls_back_to_computed_hash_when_absent() {
        let value = serde_json::json!({
            "details": { "quantization_level": "Q4_K_M", "family": "granite" }
        });
        let digest = model_digest_from_show_response(&value);
        assert!(digest.digest.starts_with("sha256:"));
        assert_eq!(digest.quantization, "Q4_K_M");
    }

    /// The computed fallback is deterministic: same response body, same
    /// digest.
    #[test]
    fn model_digest_fallback_is_deterministic() {
        let value = serde_json::json!({ "details": { "quantization_level": "Q8_0" } });
        assert_eq!(
            model_digest_from_show_response(&value).digest,
            model_digest_from_show_response(&value).digest
        );
    }

    /// Missing `details` entirely still yields `"unknown"` quantization
    /// rather than panicking.
    #[test]
    fn model_digest_missing_details_yields_unknown_quantization() {
        let value = serde_json::json!({});
        assert_eq!(
            model_digest_from_show_response(&value).quantization,
            "unknown"
        );
    }
}
