//! Golden before/after fixtures for the contraction-insertion rule
//! (`friction_rules::families::contraction::ContractionRule`).
//!
//! Each `<name>.before.md` / `<name>.after.md` pair under
//! `tests/golden/contraction/` is a hand-written, hand-verified fixture:
//! `before` is rewritten by directly driving `ContractionRule::scan`/`fix`
//! to a fixed point (bypassing the envelope-driven `gate`, since these
//! fixtures exercise the rule's matching and exception logic in
//! isolation, not the density-gating formula), and the result must equal
//! `after` byte-for-byte. `exceptions.before.md` / `exceptions.after.md`
//! are byte-identical: every expanded form in that fixture is
//! deliberately an exception case, so the rule must leave it completely
//! untouched.
//!
//! Every fixture is additionally checked for idempotence: running the
//! rule again on its own already-fixed output must change nothing.

use std::fs;
use std::path::Path;

use friction_core::{Envelope, Patch, Tier, span};
use friction_nlp::{SrxSegmenter, TaggedToken, Tagger};
use friction_rules::families::contraction::ContractionRule;
use friction_rules::{MapEnvelope, Rule, RuleContext, StrategyRng};

/// A stub tagger; the contraction rule's logic never consults POS tags.
struct NoopTagger;
impl Tagger for NoopTagger {
    fn tag(&self, _text: &str, _base_offset: usize) -> Vec<TaggedToken> {
        Vec::new()
    }
}

/// Runs `ContractionRule` to a fixed point over `source` (bypassing
/// `gate`): parse -> segment -> scan -> fix every finding -> apply ->
/// repeat, until a pass finds nothing left to fix.
///
/// `ContractionRule::scan` already guarantees its own findings are
/// pairwise non-overlapping (see that method's doc comment), so applying
/// them in one right-to-left pass needs no separate conflict-resolution
/// step beyond that ordering.
///
/// `ContractionRule::fix` itself works out exactly how many of a round's
/// findings to apply from the real, current document and the genre's
/// envelope band (see that rule's module docs' "Exact, per-round
/// budgeting" section) — `gate` is bypassed here, but `fix` still needs
/// *some* `contraction_ratio` band to compute against, so the envelope
/// carries one with `lo = hi = 1.0`, comfortably above any real document's
/// ratio, so every fixture's findings all get fixed rather than declined
/// for lack of a band.
fn run_to_fixpoint(source: &str) -> String {
    let tagger = NoopTagger;
    let segmenter = SrxSegmenter::new();
    let envelope = MapEnvelope::new().with("contraction_ratio", Envelope::new(1.0, 1.0));
    let rule = ContractionRule::new();

    let mut current = source.to_string();
    for _ in 0..8 {
        let document =
            friction_parse::parse(current.as_str()).expect("fixture must be valid markdown");
        let with_sentences = friction_nlp::segment_document(&document, &segmenter)
            .expect("fixture must sentence-segment");
        let ctx = RuleContext::new(&with_sentences, &tagger, "blog", &envelope);

        let findings = rule.scan(&ctx);
        if findings.is_empty() {
            return current;
        }

        let mut patches: Vec<Patch> = Vec::with_capacity(findings.len());
        for finding in &findings {
            let sentence_bytes = sentence_bytes_for(&with_sentences, &finding.range);
            let mut rng = StrategyRng::seeded(sentence_bytes, rule.id());
            if let Some(patch) = rule.fix(finding, &ctx, &mut rng) {
                assert_eq!(
                    patch.tier,
                    Tier::Fix,
                    "contraction patches are always Fix-tier"
                );
                patches.push(patch);
            }
        }

        patches.sort_by_key(|p| std::cmp::Reverse(p.range.start));
        for patch in &patches {
            current.replace_range(patch.range.clone(), &patch.replacement);
        }
    }
    panic!("ContractionRule did not converge within 8 passes over {source:?}");
}

/// The source bytes of the sentence containing `range`, mirroring
/// `friction-apply`'s own driver so these fixtures seed `StrategyRng`
/// exactly the way the real engine would (even though this rule ignores
/// its `strategy_rng` argument — see `ContractionRule::fix`'s own docs).
fn sentence_bytes_for<'a>(
    document: &'a friction_core::Document,
    range: &std::ops::Range<usize>,
) -> &'a [u8] {
    for unit in document.prose() {
        for sentence in &unit.sentences {
            if span::contains_range(&sentence.range, range) {
                return document
                    .text(&sentence.range)
                    .expect("sentence ranges are already validated against the document")
                    .as_bytes();
            }
        }
    }
    document.text(range).map_or(&[], str::as_bytes)
}

fn fixture(name: &str, ext: &str) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden/contraction")
        .join(format!("{name}.{ext}.md"));
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()))
}

fn check_golden(name: &str) {
    let before = fixture(name, "before");
    let after = fixture(name, "after");

    let fixed = run_to_fixpoint(&before);
    assert_eq!(
        fixed, after,
        "fixture {name}: fixed output did not match the golden `after` file"
    );

    let fixed_again = run_to_fixpoint(&fixed);
    assert_eq!(fixed_again, fixed, "fixture {name}: not idempotent");
}

#[test]
fn plain_prose_contracts() {
    check_golden("plain_prose");
}

#[test]
fn multi_sentence_contracts_across_paragraphs() {
    check_golden("multi_sentence");
}

#[test]
fn capitalization_is_preserved() {
    check_golden("capitalization");
}

#[test]
fn exception_rich_document_is_a_no_op() {
    check_golden("exceptions");
}
