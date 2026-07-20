//! The document → [`MetricVector`] computation boundary used by
//! `envelope` and `separate`.
//!
//! `envelope` and `separate` need exactly one thing from `friction-metrics`:
//! a function `&Document -> MetricVector`. [`MetricSource`] names that
//! single call so both subcommands (and their percentile/AUC math) can be
//! unit-tested — against hand-built [`MetricVector`]s, or a fixture
//! `MetricSource` — independent of the real NLP pipeline underneath
//! [`FrictionMetricsSource`].

use std::path::Path;

use anyhow::Context as _;
use friction_core::{Document, MetricVector};
use friction_nlp::{NlpruleTagger, SrxSegmenter};

/// Computes a document's [`MetricVector`]. See the module docs for why
/// this indirection exists instead of a direct call to `friction-metrics`.
pub trait MetricSource {
    /// Computes the metric vector for `document`, deterministically:
    /// identical documents must produce identical vectors, on any
    /// machine, on any run.
    fn compute(&self, document: &Document) -> MetricVector;
}

/// The [`MetricSource`] `envelope` and `separate` use by default.
///
/// Computes [`friction_metrics::compute`] over the real segmentation/
/// tagging pipeline (`friction-nlp`'s [`SrxSegmenter`] and
/// [`NlpruleTagger`]). Holds its [`NlpruleTagger`] (which loads the
/// embedded English model) so that cost is paid once per corpus-scale run
/// — [`FrictionMetricsSource::new`] — rather than once per document.
pub struct FrictionMetricsSource {
    segmenter: SrxSegmenter,
    tagger: NlpruleTagger,
}

impl FrictionMetricsSource {
    /// Loads the tagger model and builds a `FrictionMetricsSource`.
    ///
    /// # Errors
    /// Returns an error if the embedded English tagger model fails to
    /// load (see [`NlpruleTagger::new`]).
    pub fn new() -> anyhow::Result<Self> {
        let tagger =
            NlpruleTagger::new().context("failed to load the embedded English tagger model")?;
        Ok(Self {
            segmenter: SrxSegmenter::new(),
            tagger,
        })
    }
}

impl MetricSource for FrictionMetricsSource {
    fn compute(&self, document: &Document) -> MetricVector {
        friction_metrics::compute(document, &self.segmenter, &self.tagger)
    }
}

/// Reads `path` as UTF-8 and parses it into a [`Document`].
///
/// Shared by `envelope` and `separate`, the two subcommands that both
/// need to turn a corpus doc on disk into something a [`MetricSource`]
/// can consume.
///
/// # Errors
/// Returns an error (with `doc_id` in the message, for corpus-scale runs
/// where knowing *which* doc failed matters) if `path` can't be read, is
/// not valid UTF-8, or fails to parse as markdown.
pub fn load_document(path: &Path, doc_id: &str) -> anyhow::Result<Document> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("{doc_id}: failed to read {}", path.display()))?;
    let text = String::from_utf8(bytes)
        .with_context(|| format!("{doc_id}: {} is not valid UTF-8", path.display()))?;
    friction_parse::parse(text).with_context(|| format!("{doc_id}: failed to parse"))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct ConstantSource(MetricVector);

    impl MetricSource for ConstantSource {
        fn compute(&self, _document: &Document) -> MetricVector {
            self.0
        }
    }

    /// `MetricSource` is a plain trait object boundary: a fixture source
    /// can be substituted for `FrictionMetricsSource` and produces exactly
    /// the vector it was built with, regardless of the document passed in.
    #[test]
    fn metric_source_trait_object_dispatches_to_impl() {
        let doc = friction_parse::parse("Hello, world.\n").unwrap();
        let vector = MetricVector {
            triad_rate: 0.42,
            ..MetricVector::default()
        };
        let source: &dyn MetricSource = &ConstantSource(vector);
        assert_eq!(source.compute(&doc), vector);
    }

    /// `FrictionMetricsSource` runs the real segmentation/tagging pipeline
    /// end to end and produces a hand-computed vector.
    ///
    /// Source: `"However, it can't stop working."` — one paragraph, one
    /// sentence, no comma-plus-coordinator, comma-plus-`VBG`, or list at
    /// all, so every field below is fixed by the text alone, independent
    /// of the real tagger's specific tag choices:
    ///
    /// - `sentence_length_mean`: whitespace-token count is 5 (`However,`,
    ///   `it`, `can't`, `stop`, `working.`); a single sentence, so
    ///   `stddev = cv = 0.0`.
    /// - `discourse_marker_density`: the sentence starts with `"However"`
    ///   (1 marked sentence) over 5 word tokens (`however, it, can't,
    ///   stop, working`) — `1 * 1000 / 5 = 200.0`.
    /// - `contraction_ratio`: `"can't"` is the only contracted or
    ///   contractible form present — `1 / (1 + 0) = 1.0`.
    /// - `paragraph_shape_mean`/`cv`: one paragraph, one sentence — mean
    ///   `1.0`, `cv = 0.0` (single-observation convention).
    /// - Every other field (`triad_rate`, `bullet_parallelism`,
    ///   `em_dash_density`, `semicolon_density`, `participial_closer_rate`,
    ///   `not_just_but_rate`, `ritual_marker_rate`, `llm_favored_phrase_rate`,
    ///   `human_favored_phrase_rate`, `heading_density`, `list_item_density`,
    ///   `bold_span_density`, `sentence_opener_repeat_rate`) is `0.0`: none
    ///   of their trigger patterns (a coordinated list, a dash, a
    ///   semicolon, a trailing participial clause, `"not just/only ...
    ///   but"`, a ritual open/close phrase, a mined-pack phrase, a
    ///   heading/list/bold span, a second sentence to repeat an opener)
    ///   appears in the text at all — except `top_opener_concentration`,
    ///   which is `1.0`: the sole sentence's leading unigram ("however")
    ///   is the only opener observed, so it is `100%` of the (one-element)
    ///   opener distribution.
    #[test]
    fn friction_metrics_source_computes_real_pipeline_end_to_end() {
        const EPSILON: f64 = 1e-9;

        let doc = friction_parse::parse("However, it can't stop working.\n").unwrap();
        let source = FrictionMetricsSource::new().expect("embedded tagger model loads");
        let metrics = source.compute(&doc);

        let expected = MetricVector {
            sentence_length_mean: 5.0,
            sentence_length_stddev: 0.0,
            sentence_length_cv: 0.0,
            discourse_marker_density: 200.0,
            triad_rate: 0.0,
            contraction_ratio: 1.0,
            bullet_parallelism: 0.0,
            paragraph_shape_mean: 1.0,
            paragraph_shape_cv: 0.0,
            em_dash_density: 0.0,
            semicolon_density: 0.0,
            participial_closer_rate: 0.0,
            not_just_but_rate: 0.0,
            ritual_marker_rate: 0.0,
            llm_favored_phrase_rate: 0.0,
            human_favored_phrase_rate: 0.0,
            heading_density: 0.0,
            list_item_density: 0.0,
            bold_span_density: 0.0,
            sentence_opener_repeat_rate: 0.0,
            top_opener_concentration: 1.0,
        };

        for (name, value) in metrics.named_values() {
            let expected_value = expected.get(name).expect("named field exists");
            assert!(
                (value - expected_value).abs() < EPSILON,
                "{name}: expected {expected_value}, got {value}"
            );
        }
    }
}
