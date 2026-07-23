//! `corpus/genconfig-paired.toml` schema and loading — stock/antislop
//! paired-generation configuration consumed by `corpus-tool
//! generate-paired`.
//!
//! Kept as its own file/struct rather than folded into
//! `crate::genconfig::GenConfig`: the corpus this drives
//! (`corpus/paired/`, see `crate::paired_manifest`) is a mining
//! resource with a lifecycle deliberately disjoint from the
//! manifest-tracked, split eval corpus `GenConfig` drives — no split, no
//! `corpus/manifest.jsonl` entry, ever. Extending `GenConfig` instead
//! would force every existing `GenConfig` test fixture to grow a field
//! it doesn't need. `deny_unknown_fields` throughout, matching this
//! crate's config-loading convention (`src/genconfig.rs`).

use std::path::Path;

use anyhow::Context;
use serde::Deserialize;

use crate::genconfig::OllamaConfig;

/// Top-level `corpus/genconfig-paired.toml` document.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PairedGenConfig {
    pub ollama: OllamaConfig,
    pub models: PairedModels,
    /// Fixed additive seed base. Combined with a stable hash of
    /// `(genre, prompt_id)` — deliberately omitting `model`, unlike
    /// `crate::genconfig::GenConfig::base_seed` — so both sides of one
    /// pair are derived from the same seed; see
    /// `commands::generate_paired::derive_paired_seed`.
    pub base_seed: u64,
    /// Fixed for every paired job: no low-temperature or style-prompted
    /// slicing, so ratio mining over the paired corpus sees only the
    /// stock/antislop contrast.
    pub temperature: f64,
    /// First N prompts per genre, ascending id order (the same sort
    /// `crate::prompts::load` already applies), used for every paired
    /// generation run — pinned so the paired corpus's size stays fixed
    /// even as `corpus/prompts/*.toml` grows past N for the `generate`
    /// recipe.
    pub prompts_per_genre: usize,
}

/// The stock/antislop model pair. Both entries share one 12B base model
/// in the shipped config; only the antislop-tuning differs.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PairedModels {
    pub stock: String,
    pub antislop: String,
}

impl PairedGenConfig {
    /// Sanity-checks the loaded config beyond what serde's type-level
    /// schema already enforces.
    ///
    /// # Errors
    ///
    /// Returns an error if `prompts_per_genre` is zero, `models.stock`
    /// and `models.antislop` name the same model (the two sides would be
    /// meaningless — there'd be no contrast to mine), or `temperature`
    /// isn't positive.
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.prompts_per_genre == 0 {
            anyhow::bail!("genconfig-paired: prompts_per_genre must be > 0");
        }
        if self.models.stock == self.models.antislop {
            anyhow::bail!(
                "genconfig-paired: models.stock and models.antislop must differ (both are \
                 \"{}\") — identical models make the paired stock/antislop contrast meaningless",
                self.models.stock
            );
        }
        if self.temperature <= 0.0 {
            anyhow::bail!(
                "genconfig-paired: temperature must be > 0.0, got {}",
                self.temperature
            );
        }
        Ok(())
    }
}

/// Reads and parses `path` as a [`PairedGenConfig`], then validates it.
///
/// # Errors
///
/// Returns an error if the file can't be read, fails to parse, or fails
/// [`PairedGenConfig::validate`].
pub fn load(path: &Path) -> anyhow::Result<PairedGenConfig> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading paired genconfig at {}", path.display()))?;
    let config: PairedGenConfig = toml::from_str(&text)
        .with_context(|| format!("parsing paired genconfig at {}", path.display()))?;
    config.validate()?;
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL: &str = r#"
        base_seed = 1
        temperature = 0.7
        prompts_per_genre = 11

        [ollama]
        num_predict = 1536

        [models]
        stock = "gemma3:12b"
        antislop = "hf.co/mradermacher/gemma-3-12b-it-antislop-GGUF:Q4_K_M"
    "#;

    /// A well-formed paired genconfig parses, and the endpoint defaults
    /// when omitted.
    #[test]
    fn paired_genconfig_parses_minimal_document_with_default_endpoint() {
        let config: PairedGenConfig = toml::from_str(MINIMAL).unwrap();
        assert_eq!(config.ollama.endpoint, "http://localhost:11434");
        assert_eq!(config.prompts_per_genre, 11);
        assert!(config.validate().is_ok());
    }

    /// An unrecognized top-level key is a hard parse error.
    #[test]
    fn paired_genconfig_rejects_unknown_field() {
        let text = format!("{MINIMAL}\noops = 1\n");
        let result: Result<PairedGenConfig, _> = toml::from_str(&text);
        assert!(result.is_err());
    }

    /// `validate` rejects `prompts_per_genre = 0`.
    #[test]
    fn validate_rejects_zero_prompts_per_genre() {
        let mut config: PairedGenConfig = toml::from_str(MINIMAL).unwrap();
        config.prompts_per_genre = 0;
        assert!(config.validate().is_err());
    }

    /// `validate` rejects `models.stock == models.antislop`.
    #[test]
    fn validate_rejects_identical_stock_and_antislop_models() {
        let mut config: PairedGenConfig = toml::from_str(MINIMAL).unwrap();
        config.models.antislop = config.models.stock.clone();
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("must differ"));
    }

    /// `validate` rejects a non-positive `temperature`.
    #[test]
    fn validate_rejects_non_positive_temperature() {
        let mut config: PairedGenConfig = toml::from_str(MINIMAL).unwrap();
        config.temperature = 0.0;
        assert!(config.validate().is_err());
    }

    /// `load` surfaces a clear (non-panicking) error for a missing file.
    #[test]
    fn load_missing_file_returns_clear_error() {
        let dir = tempfile::tempdir().unwrap();
        let result = load(&dir.path().join("does-not-exist.toml"));
        assert!(result.is_err());
    }
}
