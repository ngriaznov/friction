//! `corpus/genconfig.toml` schema and loading — generation configuration
//! consumed by `corpus-tool generate`.
//!
//! `deny_unknown_fields` throughout, matching the manifest's strictness
//! convention (`src/manifest.rs`): a typo'd key is a hard parse error, not
//! a silently ignored one.

use std::path::Path;

use anyhow::Context;
use serde::Deserialize;

/// Top-level `corpus/genconfig.toml` document.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenConfig {
    pub ollama: OllamaConfig,
    /// Fixed additive seed base. Combined with a stable
    /// hash of `(model, prompt_id, slice)` to derive each job's seed —
    /// changing this after docs have been generated changes every
    /// derived doc id and seed.
    pub base_seed: u64,
    /// Model matrix. Order is significant: it is the deterministic
    /// round-robin order jobs are assigned to models within a genre (see
    /// `commands::generate::plan_genre_jobs`).
    pub models: Vec<ModelSpec>,
    pub temperature: TemperatureConfig,
    pub style_prompted: StyleConfig,
    pub targets: TargetsConfig,
}

/// Ollama endpoint + per-request generation limits.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OllamaConfig {
    #[serde(default = "default_endpoint")]
    pub endpoint: String,
    /// Cap on `options.num_predict` sent to `/api/generate`.
    pub num_predict: u32,
    /// Per-request timeout in seconds; defaults to 300 (small models on
    /// modest hardware can take a while for a ~1000-word response).
    #[serde(default)]
    pub request_timeout_secs: Option<u64>,
}

fn default_endpoint() -> String {
    "http://localhost:11434".to_string()
}

/// One entry in the model matrix.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelSpec {
    pub name: String,
}

/// Temperature slicing: `default` for most jobs, `low` for at
/// least `low_fraction` of them.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TemperatureConfig {
    pub default: f64,
    pub low: f64,
    /// Minimum fraction of docs (per genre) that must use `low` instead
    /// of `default`. Must be in `(0, 1]`.
    pub low_fraction: f64,
}

impl TemperatureConfig {
    /// Every `low_every`-th planned job (0-indexed within a genre) uses
    /// `low` instead of `default`.
    ///
    /// `floor(1 / low_fraction)`, clamped to at least 1, guarantees the
    /// realized fraction is `>= low_fraction` for *any* job count `N`:
    /// selecting jobs at index `0, low_every, 2*low_every, ...` yields
    /// `ceil(N / low_every)` of them, and `low_every <= 1 / low_fraction`
    /// implies `ceil(N / low_every) >= N * low_fraction`.
    pub fn low_every(&self) -> usize {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let every = (1.0 / self.low_fraction).floor().max(1.0) as usize;
        every
    }
}

/// Style-prompted slice.
///
/// At most `fraction` of docs are generated with `instruction` appended
/// to the prompt and `style_prompted: true` recorded in the manifest.
/// All other jobs pass the prompt verbatim, with no system or style
/// prompt at all.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StyleConfig {
    /// Maximum fraction of docs (per genre) that may be style-prompted.
    /// Must be in `(0, 1]`.
    pub fraction: f64,
    pub instruction: String,
}

impl StyleConfig {
    /// Every `style_every`-th planned job (0-indexed within a genre, at
    /// the *last* slot of each block: index `style_every - 1`,
    /// `2*style_every - 1`, ...) is style-prompted.
    ///
    /// `ceil(1 / fraction)`, clamped to at least 1, guarantees the
    /// realized fraction is `<= fraction` for *any* job count `N`:
    /// selecting the last slot of each `style_every`-sized block yields
    /// exactly `floor(N / style_every)` of them, and
    /// `style_every >= 1 / fraction` implies
    /// `floor(N / style_every) <= N * fraction`.
    pub fn style_every(&self) -> usize {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let every = (1.0 / self.fraction).ceil().max(1.0) as usize;
        every
    }
}

/// Per-genre generation targets.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetsConfig {
    /// Target doc count for the `llm` corpus, per genre, when run with
    /// the full model matrix — sized so each `(class=llm, genre)` cell
    /// clears the `>= 60` minimum.
    pub docs_per_genre: usize,
}

/// Reads and parses `path` as a [`GenConfig`], then validates it.
///
/// # Errors
///
/// Returns an error if the file can't be read, fails to parse, or fails
/// [`GenConfig::validate`].
pub fn load(path: &Path) -> anyhow::Result<GenConfig> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading genconfig at {}", path.display()))?;
    let config: GenConfig = toml::from_str(&text)
        .with_context(|| format!("parsing genconfig at {}", path.display()))?;
    config.validate()?;
    Ok(config)
}

impl GenConfig {
    /// Sanity-checks the loaded config beyond what serde's type-level
    /// schema already enforces.
    ///
    /// # Errors
    ///
    /// Returns an error if the model matrix is empty, any fraction is
    /// outside `(0, 1]`, `low` isn't below `default`, or
    /// `targets.docs_per_genre` is zero.
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.models.is_empty() {
            anyhow::bail!("genconfig: `models` must not be empty");
        }
        if !(0.0 < self.temperature.low_fraction && self.temperature.low_fraction <= 1.0) {
            anyhow::bail!(
                "genconfig: temperature.low_fraction must be in (0, 1], got {}",
                self.temperature.low_fraction
            );
        }
        if !(0.0 < self.style_prompted.fraction && self.style_prompted.fraction <= 1.0) {
            anyhow::bail!(
                "genconfig: style_prompted.fraction must be in (0, 1], got {}",
                self.style_prompted.fraction
            );
        }
        if self.temperature.low >= self.temperature.default {
            anyhow::bail!(
                "genconfig: temperature.low ({}) must be below temperature.default ({}) \
                 (low is the hard, more-uniform case)",
                self.temperature.low,
                self.temperature.default
            );
        }
        if self.targets.docs_per_genre == 0 {
            anyhow::bail!("genconfig: targets.docs_per_genre must be > 0");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL: &str = r#"
        base_seed = 1

        [ollama]
        num_predict = 512

        [[models]]
        name = "a"
        [[models]]
        name = "b"

        [temperature]
        default = 0.7
        low = 0.2
        low_fraction = 0.2

        [style_prompted]
        fraction = 0.1
        instruction = "sound human"

        [targets]
        docs_per_genre = 10
    "#;

    /// A well-formed genconfig parses, and the endpoint
    /// defaults when omitted.
    #[test]
    fn genconfig_parses_minimal_document_with_default_endpoint() {
        let config: GenConfig = toml::from_str(MINIMAL).unwrap();
        assert_eq!(config.ollama.endpoint, "http://localhost:11434");
        assert_eq!(config.models.len(), 2);
        assert!(config.validate().is_ok());
    }

    /// An unrecognized top-level key is a hard
    /// parse error.
    #[test]
    fn genconfig_rejects_unknown_field() {
        let text = format!("{MINIMAL}\noops = 1\n");
        let result: Result<GenConfig, _> = toml::from_str(&text);
        assert!(result.is_err());
    }

    /// `low_every() == floor(1 / low_fraction)`, e.g. 0.2 -> 5.
    #[test]
    fn temperature_low_every_matches_expected_ratio() {
        let config: GenConfig = toml::from_str(MINIMAL).unwrap();
        assert_eq!(config.temperature.low_every(), 5);
    }

    /// `style_every() == ceil(1 / fraction)`, e.g. 0.1 -> 10.
    #[test]
    fn style_every_matches_expected_ratio() {
        let config: GenConfig = toml::from_str(MINIMAL).unwrap();
        assert_eq!(config.style_prompted.style_every(), 10);
    }

    /// Selecting every `low_every`-th job out of N always realizes
    /// at least `low_fraction` of them, for a range of N (including N not
    /// a multiple of `low_every`).
    #[test]
    fn low_every_selection_always_meets_or_exceeds_fraction() {
        let config: GenConfig = toml::from_str(MINIMAL).unwrap();
        let every = config.temperature.low_every();
        for n in [1usize, 4, 5, 6, 11, 66, 100] {
            let selected = (0..n).filter(|k| k % every == 0).count();
            #[allow(clippy::cast_precision_loss)]
            let realized = selected as f64 / n as f64;
            assert!(
                realized >= config.temperature.low_fraction,
                "n={n}: realized {realized} < {}",
                config.temperature.low_fraction
            );
        }
    }

    /// Selecting the last slot of every `style_every`-sized block
    /// out of N always realizes <= `fraction` of them.
    #[test]
    fn style_every_selection_never_exceeds_fraction() {
        let config: GenConfig = toml::from_str(MINIMAL).unwrap();
        let every = config.style_prompted.style_every();
        for n in [1usize, 4, 9, 10, 11, 66, 100] {
            let selected = (0..n).filter(|k| k % every == every - 1).count();
            #[allow(clippy::cast_precision_loss)]
            let realized = selected as f64 / n as f64;
            assert!(
                realized <= config.style_prompted.fraction,
                "n={n}: realized {realized} > {}",
                config.style_prompted.fraction
            );
        }
    }

    /// A genconfig with an empty model matrix fails validation.
    #[test]
    fn validate_rejects_empty_model_matrix() {
        let mut config: GenConfig = toml::from_str(MINIMAL).unwrap();
        config.models.clear();
        assert!(config.validate().is_err());
    }

    /// A genconfig with `low >= default` fails validation.
    #[test]
    fn validate_rejects_low_temperature_not_below_default() {
        let mut config: GenConfig = toml::from_str(MINIMAL).unwrap();
        config.temperature.low = 0.9;
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
