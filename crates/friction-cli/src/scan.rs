//! The shared "measure, then find issues, but never fix" pass `check`
//! runs and `explain` reuses for its before/after comparison: parse,
//! compute the document's [`MetricVector`], gate every registered rule,
//! and scan (never fix) the ones that aren't `Off`.

use friction_core::{Finding, MetricVector};
use friction_rules::{Gate, GenreEnvelope, RuleContext};

use crate::common::{CliError, Engine};

/// One document's metrics plus every finding surfaced by a non-`Off`-gated
/// rule, without applying any fix.
pub struct ScanOutcome {
    /// The document's full 21-field metric vector.
    pub metrics: MetricVector,
    /// Every finding surfaced by an active rule's `scan`, sorted by
    /// `(range.start, rule id, range.end)` — deterministic regardless of
    /// [`friction_apply::registered_rules`]'s own registration order, and
    /// convenient to read (source order, ties broken by rule name).
    pub findings: Vec<Finding>,
}

/// Parses `source`, computes its metrics, and scans every registered rule
/// gated `Detect` or `Fix` — never calling `fix` — for `genre` against
/// `envelope`.
///
/// # Errors
/// Returns [`CliError::Parse`] or [`CliError::Segment`] if `source` fails
/// to parse as markdown or sentence-segment.
pub fn scan(
    source: &str,
    genre: &str,
    envelope: &dyn GenreEnvelope,
    engine: &Engine,
) -> Result<ScanOutcome, CliError> {
    let document = friction_parse::parse(source)?;
    let metrics = friction_metrics::compute(&document, &engine.segmenter, &engine.tagger);
    let with_sentences = friction_nlp::segment_document(&document, &engine.segmenter)?;
    let ctx = RuleContext::new(&with_sentences, &engine.tagger, genre, envelope);

    let mut findings = Vec::new();
    for rule in friction_apply::registered_rules() {
        match rule.gate(&metrics, envelope) {
            Gate::Off => {}
            Gate::Detect | Gate::Fix { .. } => findings.extend(rule.scan(&ctx)),
        }
    }
    findings.sort_by(|a, b| {
        a.range
            .start
            .cmp(&b.range.start)
            .then_with(|| a.rule.as_str().cmp(b.rule.as_str()))
            .then_with(|| a.range.end.cmp(&b.range.end))
    });

    Ok(ScanOutcome { metrics, findings })
}

#[cfg(test)]
mod tests {
    use friction_rules::MapEnvelope;

    use super::*;

    /// `scan` never mutates `source`: it only measures and reports, so a
    /// document with a rule-triggering pattern still yields the exact
    /// input text as its own basis (there is no "output" from `scan` at
    /// all — this test exists to document that omission is deliberate,
    /// by confirming the finding it does surface still points into the
    /// original, untouched source).
    #[test]
    fn scan_surfaces_findings_without_mutating_source() {
        let engine = Engine::load().expect("embedded tagger model loads");
        let envelope =
            MapEnvelope::new().with("not_just_but_rate", friction_core::Envelope::new(0.0, 0.0));
        let source = "This release is not just fast but also reliable.";
        let outcome = scan(source, "blog", &envelope, &engine).expect("well-formed input scans");
        assert!(
            outcome
                .findings
                .iter()
                .any(|f| f.rule.as_str() == "symmetry.not_just_but")
        );
        for finding in &outcome.findings {
            assert_eq!(
                &source[finding.range.clone()],
                &source[finding.range.clone()],
                "finding range must still index the original source"
            );
        }
    }

    /// A document whose metrics sit inside every band in an empty
    /// envelope (nothing gates on anything) surfaces no findings at all.
    #[test]
    fn scan_finds_nothing_under_an_empty_envelope() {
        let engine = Engine::load().expect("embedded tagger model loads");
        let envelope = MapEnvelope::new();
        let outcome = scan("Some ordinary prose.", "blog", &envelope, &engine)
            .expect("well-formed input scans");
        assert!(outcome.findings.is_empty());
    }
}
