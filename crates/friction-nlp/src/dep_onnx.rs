//! [`OnnxParser`]: an `ort`-backed [`DepParser`], gated behind the `onnx`
//! cargo feature.
//!
//! # Status: no model is registered yet
//!
//! As of this writing, no downloadable ONNX English universal-dependencies
//! parser under roughly 100 MB has been located: spaCy's dependency parser
//! has no supported ONNX export path, and `UDPipe` 2's English models ship
//! as `PyTorch` weights, not ONNX. `friction-packs`' artifact registry
//! therefore has no `dependency-parser-model` entry, and
//! [`OnnxParser::load`] always returns [`DepParseError::ModelNotAvailable`]
//! in practice — never a panic, never a silent no-op.
//!
//! The `ort` integration below is real, not a placeholder: session
//! construction with intra-op parallelism pinned to one thread (this
//! workspace's determinism discipline for model-backed inference), and
//! [`softmax_top2_margin`], the confidence-margin computation a real
//! implementation of [`DepParser::parse`] would report per decision. What's
//! missing is a *model contract* — this module cannot honestly decode a
//! model's output tensors into [`SentenceParse`] edges without knowing
//! what those tensors mean, and inventing one for a model that does not
//! exist would be worse than the honest "not available" error this module
//! actually returns. The heuristic parser (`HeuristicParser`, always
//! available, no feature flag) is `friction-nlp`'s dependency parser for
//! the foreseeable future; this module exists so wiring in a real model
//! later is a matter of filling in decode logic, not designing a new
//! integration point.

use std::path::Path;

use crate::dep::{Confidence, DepParseError, DepParser, SentenceParse};
use crate::tag::TaggedToken;

/// A dependency parser backed by a local ONNX model.
///
/// # Determinism
/// Intra-op parallelism is fixed to a single thread at load time, so
/// inference does not depend on the host machine's core count.
#[derive(Debug)]
pub struct OnnxParser {
    #[expect(
        dead_code,
        reason = "populated once a model contract exists; see module docs"
    )]
    session: ort::session::Session,
}

impl OnnxParser {
    /// Loads a dependency-parser model from `model_path` (typically a path
    /// resolved from `friction-packs`' cache directory).
    ///
    /// # Errors
    /// Returns [`DepParseError::ModelNotAvailable`] if `model_path` does
    /// not exist or the session fails to load. As of this writing no
    /// pinned model is registered anywhere for a caller to pass in (see
    /// the module docs), so this returns that error for any realistic
    /// caller today.
    pub fn load(model_path: impl AsRef<Path>) -> Result<Self, DepParseError> {
        let path = model_path.as_ref();
        if !path.is_file() {
            return Err(DepParseError::ModelNotAvailable {
                reason: format!(
                    "no dependency-parser model file at {} (none is registered in \
                     friction-packs yet; see friction-nlp's dep_onnx module docs)",
                    path.display()
                )
                .into(),
            });
        }
        // Each `ort` builder step's `Result` is parameterized by a
        // different recoverable-builder type, so each is converted to
        // `DepParseError` immediately (a shared closure won't typecheck
        // across them) rather than chained with `?`/`and_then` under one
        // error type.
        let builder = ort::session::Session::builder().map_err(not_available)?;
        let mut builder = builder.with_intra_threads(1).map_err(not_available)?;
        let session = builder.commit_from_file(path).map_err(not_available)?;
        Ok(Self { session })
    }
}

/// Converts any `ort` builder-step error into [`DepParseError::ModelNotAvailable`].
///
/// Takes `err` by value because [`Result::map_err`] (every call site) calls
/// its callback with an owned error, not a borrowed one — a reference
/// parameter would not typecheck there despite the body only reading it.
#[allow(
    clippy::needless_pass_by_value,
    reason = "required by the `Result::map_err` call sites this feeds"
)]
fn not_available<T>(err: ort::Error<T>) -> DepParseError {
    DepParseError::ModelNotAvailable {
        reason: err.to_string().into(),
    }
}

impl DepParser for OnnxParser {
    fn parse(
        &self,
        _source: &str,
        _tokens: &[TaggedToken],
    ) -> Result<SentenceParse, DepParseError> {
        Err(DepParseError::ModelNotAvailable {
            reason: "OnnxParser has no output-decoding contract implemented for any model yet \
                      (none is registered in friction-packs); see friction-nlp's dep_onnx \
                      module docs"
                .into(),
        })
    }
}

/// The top-scoring label's index and its softmax margin over the runner-up.
///
/// Given raw model logits over a decision's label vocabulary, this is the
/// [`Confidence`] a [`DepParser::parse`] implementation should report for
/// that decision (the "margin gate from softmax top-2 delta" this
/// workspace's model-backed inference is expected to compute).
///
/// Returns `None` for an empty `logits` slice or a degenerate
/// all-negative-infinity input; a real model never produces either.
#[must_use]
pub fn softmax_top2_margin(logits: &[f32]) -> Option<(usize, Confidence)> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return None;
    }
    let exps: Vec<f32> = logits.iter().map(|&logit| (logit - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum <= 0.0 {
        return None;
    }

    let mut top_index = 0usize;
    let mut top_prob = f32::NEG_INFINITY;
    let mut second_prob = f32::NEG_INFINITY;
    for (index, &exp) in exps.iter().enumerate() {
        let prob = exp / sum;
        if prob > top_prob {
            second_prob = top_prob;
            top_prob = prob;
            top_index = index;
        } else if prob > second_prob {
            second_prob = prob;
        }
    }
    let margin = if second_prob.is_finite() {
        top_prob - second_prob
    } else {
        top_prob
    };
    Some((top_index, Confidence::new(margin)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A clear winner among the logits yields a large margin and the
    /// correct top index.
    #[test]
    fn softmax_top2_margin_finds_clear_winner() {
        let (index, confidence) = softmax_top2_margin(&[0.0, 5.0, 0.0]).unwrap();
        assert_eq!(index, 1);
        assert!(
            confidence.value() > 0.9,
            "confidence was {}",
            confidence.value()
        );
    }

    /// Two near-tied logits yield a small margin — the low-confidence case
    /// callers are expected to demote to `Suggest` tier.
    #[test]
    fn softmax_top2_margin_finds_narrow_margin_for_near_tie() {
        let (_, confidence) = softmax_top2_margin(&[1.0, 1.01]).unwrap();
        assert!(
            confidence.value() < 0.1,
            "confidence was {}",
            confidence.value()
        );
    }

    /// A single-label distribution is maximally confident.
    #[test]
    fn softmax_top2_margin_single_label_is_certain() {
        let (index, confidence) = softmax_top2_margin(&[3.0]).unwrap();
        assert_eq!(index, 0);
        assert!((confidence.value() - 1.0).abs() < 1e-6);
    }

    /// An empty logits slice returns `None` rather than panicking.
    #[test]
    fn softmax_top2_margin_empty_input_returns_none() {
        assert!(softmax_top2_margin(&[]).is_none());
    }

    /// Loading a model from a path that does not exist reports a clear,
    /// non-panicking "not available" error rather than propagating an I/O
    /// error or crashing — the only path exercisable today, since no
    /// pinned model exists to load successfully (see module docs).
    #[test]
    fn load_missing_model_reports_not_available() {
        let err = OnnxParser::load("/nonexistent/friction-onnx-test-model.onnx").unwrap_err();
        assert!(matches!(err, DepParseError::ModelNotAvailable { .. }));
    }
}
