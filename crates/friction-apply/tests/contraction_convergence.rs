//! Regression: `ContractionRule` must bring a below-envelope document's
//! `contraction_ratio` back into its genre's band within a handful of
//! rounds, even when the document carries far more contractible
//! occurrences than a small fixed per-fix-effect assumption would expect.
//!
//! Before the fix, `ContractionRule::gate` sized its per-round budget from
//! a fixed constant that assumed a denominator of `10`; a document whose
//! real denominator was much larger (a realistic case well inside this
//! workspace's own supported word-count range) needed far more rounds to
//! close the gap than `friction_apply::MAX_ROUNDS` allows, leaving the
//! document stuck far outside its envelope. This test reproduces that
//! shape directly and asserts real convergence.

use friction_apply::{MAX_ROUNDS, run_fixpoint};
use friction_core::Envelope;
use friction_nlp::{SrxSegmenter, Tagger};
use friction_rules::{ContractionRule, MapEnvelope, Rule};

/// A stub tagger; `ContractionRule` never consults part-of-speech tags.
struct NoopTagger;
impl Tagger for NoopTagger {
    fn tag(&self, _text: &str, _base_offset: usize) -> Vec<friction_nlp::TaggedToken> {
        Vec::new()
    }
}

/// `N` independent sentences, each with exactly one contractible `"is
/// not"`, e.g. `"Sentence 0 is not ready. Sentence 1 is not ready. ..."` —
/// the same shape the convergence finding used, well inside the 300-2000
/// word range this workspace's own corpus documents target.
fn many_contractible_sentences(n: usize) -> String {
    (0..n)
        .map(|i| format!("Sentence {i} is not ready."))
        .collect::<Vec<_>>()
        .join(" ")
}

/// The real, shipped `forum` genre's `contraction_ratio` band
/// (`crates/friction-packs/packs/envelope-v2.toml`'s `[forum.
/// contraction_ratio]` table: `lo = 0.2`, `hi = 0.9`).
fn forum_contraction_band() -> MapEnvelope {
    MapEnvelope::new().with("contraction_ratio", Envelope::new(0.2, 0.9))
}

/// `text`'s `contraction_ratio`, computed the same way `friction_metrics::
/// compute` does internally (parse, then sentence-segment — the metric
/// walks a document's *sentences*, so an unsegmented, freshly-parsed
/// document always reports a ratio of `0.0` regardless of its content).
fn contraction_ratio_of(text: &str, segmenter: &dyn friction_nlp::Segmenter) -> f64 {
    let parsed = friction_parse::parse(text).expect("fixed output re-parses");
    let with_sentences =
        friction_nlp::segment_document(&parsed, segmenter).expect("fixed output re-segments");
    friction_metrics::contraction_ratio(&with_sentences)
}

/// A 200-sentence document (~1400 words), the exact size the original
/// finding demonstrated non-convergence on, now reaches its envelope floor
/// well within `MAX_ROUNDS`.
#[test]
fn two_hundred_sentence_document_converges_within_max_rounds() {
    let source = many_contractible_sentences(200);
    let rule = ContractionRule::new();
    let rules: [&dyn Rule; 1] = [&rule];
    let segmenter = SrxSegmenter::new();
    let tagger = NoopTagger;
    let envelope = forum_contraction_band();

    let (output, report) = run_fixpoint(&source, &rules, &segmenter, &tagger, "forum", &envelope)
        .expect("well-formed synthetic document must not fail to parse/segment");

    assert!(
        report.rounds.len() < MAX_ROUNDS,
        "expected convergence well before MAX_ROUNDS, took {} rounds",
        report.rounds.len()
    );

    let final_ratio = contraction_ratio_of(&output, &segmenter);
    assert!(
        final_ratio >= 0.2,
        "expected the document to have re-entered its envelope (ratio >= 0.2), got {final_ratio}"
    );
}

/// The same shape at a smaller size (20 sentences) also converges — not
/// just the large end of the range.
#[test]
fn twenty_sentence_document_converges_within_max_rounds() {
    let source = many_contractible_sentences(20);
    let rule = ContractionRule::new();
    let rules: [&dyn Rule; 1] = [&rule];
    let segmenter = SrxSegmenter::new();
    let tagger = NoopTagger;
    let envelope = forum_contraction_band();

    let (output, report) = run_fixpoint(&source, &rules, &segmenter, &tagger, "forum", &envelope)
        .expect("well-formed synthetic document must not fail to parse/segment");

    assert!(
        report.rounds.len() < MAX_ROUNDS,
        "expected convergence well before MAX_ROUNDS, took {} rounds",
        report.rounds.len()
    );

    let final_ratio = contraction_ratio_of(&output, &segmenter);
    assert!(
        final_ratio >= 0.2,
        "expected the document to have re-entered its envelope (ratio >= 0.2), got {final_ratio}"
    );
}
