//! [`MetricVector`]: the stylometric metric families.

/// A vector of stylometric metrics, computed per document or per paragraph
/// by `friction-metrics`.
///
/// `friction-core` only defines the shape; every metric is a pure function
/// over a [`crate::Document`] computed in `friction-metrics`, and
/// per-(genre, metric) human bands are estimated into an [`crate::Envelope`]
/// by `corpus-tool envelope`.
///
/// Fields are named rather than indexed so callers can address a specific
/// metric directly; [`MetricVector::named_values`] and [`MetricVector::get`]
/// additionally expose the vector in a fixed, deterministic `(name, value)`
/// order for generic processing — envelope estimation, TOML serialization,
/// tabular `friction explain` reports — without resorting to a `HashMap`.
///
/// With the `serde` feature enabled, `MetricVector` derives
/// `Serialize`/`Deserialize` (field order is fixed by the struct
/// definition, not iteration order, so this is deterministic).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub struct MetricVector {
    /// Mean sentence length, in tokens.
    pub sentence_length_mean: f64,
    /// Standard deviation of sentence length, in tokens.
    pub sentence_length_stddev: f64,
    /// Coefficient of variation (`stddev / mean`) of sentence length —
    /// burstiness.
    pub sentence_length_cv: f64,
    /// Sentence-initial discourse-marker density, per 1000 tokens.
    pub discourse_marker_density: f64,
    /// Rate of triad coordination patterns (`"X, Y, and Z"`) per sentence.
    pub triad_rate: f64,
    /// Ratio of contracted to contractible forms.
    pub contraction_ratio: f64,
    /// Bullet-stem parallelism score.
    pub bullet_parallelism: f64,
    /// Mean sentences per paragraph.
    pub paragraph_shape_mean: f64,
    /// Coefficient of variation of sentences per paragraph.
    pub paragraph_shape_cv: f64,
    /// Em-dash density, per 1000 tokens.
    pub em_dash_density: f64,
    /// Semicolon density, per 1000 tokens.
    pub semicolon_density: f64,
    /// Rate of participial closer clauses.
    pub participial_closer_rate: f64,
    /// Rate of `"not just/only X but (also) Y"` constructions.
    pub not_just_but_rate: f64,
    /// Rate of ritual open/close markers (`"In conclusion"`, `"Overall"`,
    /// `"In today's..."`).
    pub ritual_marker_rate: f64,
    /// Rate of curated llm-favored mined n-grams (`crates/friction-packs/
    /// packs/mined-ngrams-v1.toml`), per 1000 tokens.
    pub llm_favored_phrase_rate: f64,
    /// Rate of curated human-favored mined n-grams (same pack), per 1000
    /// tokens.
    pub human_favored_phrase_rate: f64,
    /// ATX/setext heading-block density, per 1000 tokens.
    pub heading_density: f64,
    /// Markdown list-item-block density, per 1000 tokens.
    pub list_item_density: f64,
    /// Bold/strong-emphasis (`**...**`/`__...__`) span density, per 1000
    /// tokens.
    pub bold_span_density: f64,
    /// Fraction of a document's sentences (excluding the first) whose
    /// leading unigram matches the immediately preceding sentence's.
    pub sentence_opener_repeat_rate: f64,
    /// The most common sentence-leading unigram's share of all detected
    /// sentence openers in the document.
    pub top_opener_concentration: f64,
}

/// Number of metrics in [`MetricVector`]; the length of
/// [`MetricVector::FIELD_NAMES`] and of [`MetricVector::named_values`]'s
/// output.
const FIELD_COUNT: usize = 21;

impl MetricVector {
    /// The metric names, in the same fixed order as
    /// [`MetricVector::named_values`].
    pub const FIELD_NAMES: [&'static str; FIELD_COUNT] = [
        "sentence_length_mean",
        "sentence_length_stddev",
        "sentence_length_cv",
        "discourse_marker_density",
        "triad_rate",
        "contraction_ratio",
        "bullet_parallelism",
        "paragraph_shape_mean",
        "paragraph_shape_cv",
        "em_dash_density",
        "semicolon_density",
        "participial_closer_rate",
        "not_just_but_rate",
        "ritual_marker_rate",
        "llm_favored_phrase_rate",
        "human_favored_phrase_rate",
        "heading_density",
        "list_item_density",
        "bold_span_density",
        "sentence_opener_repeat_rate",
        "top_opener_concentration",
    ];

    /// Returns `(name, value)` pairs in a fixed, deterministic order
    /// matching [`MetricVector::FIELD_NAMES`] and the struct's field
    /// declaration order.
    #[must_use]
    pub fn named_values(&self) -> [(&'static str, f64); FIELD_COUNT] {
        let values = self.values();
        std::array::from_fn(|i| (Self::FIELD_NAMES[i], values[i]))
    }

    /// Looks up a metric's value by name.
    ///
    /// Returns `None` if `name` is not one of [`MetricVector::FIELD_NAMES`].
    #[must_use]
    pub fn get(&self, name: &str) -> Option<f64> {
        self.named_values()
            .into_iter()
            .find(|(n, _)| *n == name)
            .map(|(_, v)| v)
    }

    /// Values in the same fixed order as [`MetricVector::FIELD_NAMES`].
    const fn values(&self) -> [f64; FIELD_COUNT] {
        [
            self.sentence_length_mean,
            self.sentence_length_stddev,
            self.sentence_length_cv,
            self.discourse_marker_density,
            self.triad_rate,
            self.contraction_ratio,
            self.bullet_parallelism,
            self.paragraph_shape_mean,
            self.paragraph_shape_cv,
            self.em_dash_density,
            self.semicolon_density,
            self.participial_closer_rate,
            self.not_just_but_rate,
            self.ritual_marker_rate,
            self.llm_favored_phrase_rate,
            self.human_favored_phrase_rate,
            self.heading_density,
            self.list_item_density,
            self.bold_span_density,
            self.sentence_opener_repeat_rate,
            self.top_opener_concentration,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `FIELD_NAMES` covers every metric family, in declaration order.
    #[test]
    fn field_names_cover_all_metric_families() {
        assert_eq!(
            MetricVector::FIELD_NAMES,
            [
                "sentence_length_mean",
                "sentence_length_stddev",
                "sentence_length_cv",
                "discourse_marker_density",
                "triad_rate",
                "contraction_ratio",
                "bullet_parallelism",
                "paragraph_shape_mean",
                "paragraph_shape_cv",
                "em_dash_density",
                "semicolon_density",
                "participial_closer_rate",
                "not_just_but_rate",
                "ritual_marker_rate",
                "llm_favored_phrase_rate",
                "human_favored_phrase_rate",
                "heading_density",
                "list_item_density",
                "bold_span_density",
                "sentence_opener_repeat_rate",
                "top_opener_concentration",
            ]
        );
    }

    /// `named_values` pairs each field name with its actual field value, in
    /// declaration order.
    #[test]
    fn named_values_matches_field_order() {
        let metrics = MetricVector {
            sentence_length_mean: 18.5,
            sentence_length_stddev: 6.2,
            ritual_marker_rate: 0.1,
            top_opener_concentration: 0.77,
            ..MetricVector::default()
        };
        let pairs = metrics.named_values();
        assert_eq!(pairs[0], ("sentence_length_mean", 18.5));
        assert_eq!(pairs[1], ("sentence_length_stddev", 6.2));
        assert_eq!(pairs[13], ("ritual_marker_rate", 0.1));
        assert_eq!(pairs[FIELD_COUNT - 1], ("top_opener_concentration", 0.77));
    }

    /// `get` looks up a known metric by name and returns `None` for an
    /// unknown one.
    #[test]
    fn get_looks_up_by_name() {
        let metrics = MetricVector {
            triad_rate: 0.42,
            ..MetricVector::default()
        };
        assert_eq!(metrics.get("triad_rate"), Some(0.42));
        assert_eq!(metrics.get("not_a_metric"), None);
    }

    /// `Default` yields an all-zero vector (the near-no-op baseline).
    #[test]
    fn default_is_all_zero() {
        let metrics = MetricVector::default();
        assert!(metrics.named_values().iter().all(|(_, v)| *v == 0.0));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serde_round_trips_through_json() {
        let metrics = MetricVector {
            triad_rate: 0.42,
            em_dash_density: 3.1,
            ..MetricVector::default()
        };
        let json = serde_json::to_string(&metrics).unwrap();
        let round_tripped: MetricVector = serde_json::from_str(&json).unwrap();
        assert_eq!(metrics, round_tripped);
    }
}
