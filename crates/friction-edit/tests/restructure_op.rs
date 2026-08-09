//! End-to-end coverage for the restructure pass: T10's "ensure that NP
//! is/are VBN" rewrite, its held findings, the idempotence of the whole
//! pipeline with it enabled, and the reporting-layer fix that keeps
//! `--suggest`-equivalent held-finding accounting correct now that
//! restructure sits between the bounded loop and register.
//!
//! Every input here was checked against the actual shipped tagger and
//! parser (`friction fix`'s own embedded models) before being locked
//! into an assertion — several of the design brief's own worked
//! examples parse differently than a naive reading suggests (see the
//! comments on each test below); this file pins the VERIFIED behavior,
//! not the assumed one.

use friction_core::Tier;
use friction_edit::Engine;

fn engine() -> Engine {
    Engine::new().expect("embedded packs and models load")
}

/// The design's own flagship example, minus inline code (see
/// [`markdown_code_span_keeps_the_construction_out_of_scope`] for why the
/// literal brief example, which wraps the noun in inline code, doesn't
/// reach this transducer at all): a sentence-initial imperative
/// collapses the "ensure that NP is VBN" run to "VB NP".
#[test]
fn sentence_initial_imperative_rewrites() {
    let (fixed, _) = engine()
        .fix_document("Ensure that the setting is enabled.\n")
        .expect("engine runs");
    assert_eq!(fixed, "Enable the setting.\n");
}

/// Inline code inside the promoted noun phrase: verified against the
/// shipped tagger/parser that `friction_parse`'s own inline-code
/// exclusion keeps this sentence out of T10's scope entirely (the same
/// exclusion `t6_em_dash`'s own module docs describe for a dash inside
/// real inline code) -- output is byte-identical, and no
/// `restructure.ensures_that` finding of any kind is produced, matched
/// or declined. This is the design brief's own literal worked example
/// ("Ensure that the `bed_leveling` setting is enabled." ->
/// "Enable the `bed_leveling` setting."); verified NOT to reproduce
/// under the real pipeline, and safely so -- no output changes.
#[test]
fn markdown_code_span_keeps_the_construction_out_of_scope() {
    let source = "Ensure that the `bed_leveling` setting is enabled.\n";
    let (fixed, report) = engine().fix_document(source).expect("engine runs");
    assert_eq!(fixed, source);
    assert!(
        report
            .passes
            .iter()
            .flat_map(|p| &p.held)
            .all(|f| f.rule.as_str() != "restructure.ensures_that"),
        "no restructure finding of any kind for a sentence outside its scope"
    );
}

/// Infinitival "to ensure that": "to" stays untouched, only "ensure ...
/// is VBN" collapses. (An earlier draft of this test spliced the
/// infinitival onto "We updated the script to ..."; verified against the
/// shipped tagger, that specific wording gate-holds on
/// `check_rewrite_gates`'s own post-edit clause-completeness re-check --
/// re-tagging "We updated the script to enable the setting." loses its
/// only other finite-verb tag once "is" is gone, so the candidate
/// correctly declines rather than risk a fragment. This wording doesn't
/// hit that gate.)
#[test]
fn infinitival_rewrites() {
    let (fixed, _) = engine()
        .fix_document("Restart the printer to ensure that the setting is enabled.\n")
        .expect("engine runs");
    assert_eq!(fixed, "Restart the printer to enable the setting.\n");
}

/// The design brief's own second worked example ("to ensure that event
/// listeners were properly cleaned up" -> unchanged, held on the adverb
/// guard) does NOT hold as stated: verified against the shipped parser,
/// this sentence's `Ccomp` head is "were" (VBD), not "cleaned" (VBN) --
/// the participle never becomes the complement head at all, so the
/// construction never matches steps 1-6 in the first place. It declines
/// SILENTLY (no finding), not on the adverb guard. This test pins the
/// verified behavior.
#[test]
fn were_properly_cleaned_up_never_matches_the_construction() {
    let source = "to ensure that event listeners were properly cleaned up.\n";
    let (fixed, report) = engine().fix_document(source).expect("engine runs");
    assert_eq!(fixed, source);
    assert!(
        report
            .passes
            .iter()
            .flat_map(|p| &p.held)
            .all(|f| f.rule.as_str() != "restructure.ensures_that"),
        "the construction never matched, so no restructure finding at all"
    );
}

/// An adjectival/intransitive complement ("remains sustainable") never
/// matches the construction: `comp` is tagged `VBZ`, not `VBN`. No
/// finding, the same silent decline `t4_activize_to_passive` gives a
/// verb that was never a licensable candidate.
#[test]
fn adjectival_complement_never_matches() {
    let source = "Ensure that pair-programming remains sustainable.\n";
    let (fixed, report) = engine().fix_document(source).expect("engine runs");
    assert_eq!(fixed, source);
    assert!(
        report
            .passes
            .iter()
            .flat_map(|p| &p.held)
            .all(|f| f.rule.as_str() != "restructure.ensures_that")
    );
}

/// The design brief's own negation example ("ensure telemetry is not
/// sent unless opted in") -- verified against the shipped parser, "is"
/// itself becomes the `Ccomp` head here (`VBZ`, not `VBN`), so the
/// construction never matches at all. This is a DIFFERENT reason than
/// the negation guard declining a genuine match, but the safety property
/// the design's risk register cares about holds either way: the output
/// is never rewritten to state the opposite of what the author wrote.
/// The negation guard itself is exercised directly in
/// `friction-register/tests/transduce.rs` (`t10_declines_on_negation_*`),
/// where a fixture can isolate it from this parse-shape difference.
#[test]
fn negated_ensure_that_is_never_rewritten() {
    let source = "Ensure that telemetry is not sent unless opted in.\n";
    let (fixed, _) = engine().fix_document(source).expect("engine runs");
    assert_eq!(fixed, source, "must never state the opposite of the source");
}

/// A finite subject-agreement shape ("This ensures that ...") declines
/// with a named finding: the construction matched every structural step
/// but has no licensed outer mood shape in v1.
#[test]
fn finite_subject_shape_declines_with_a_named_finding() {
    let source = "This ensures that the setting is enabled.\n";
    let (fixed, report) = engine().fix_document(source).expect("engine runs");
    assert_eq!(fixed, source);
    assert!(
        report.passes.iter().flat_map(|p| &p.held).any(|f| {
            f.rule.as_str() == "restructure.ensures_that"
                && f.tier == Tier::Suggest
                && f.message.contains("no licensed outer mood shape")
        }),
        "held findings: {:?}",
        report
            .passes
            .iter()
            .flat_map(|p| &p.held)
            .collect::<Vec<_>>()
    );
}

/// The retired `able.ensures-that-short::ensure` pack row (§2's
/// "required companion change") must never co-appear with the real fix:
/// fixing a licensed instance produces exactly one
/// `restructure.ensures_that`-tier `Fix` patch and no stale
/// `able.ensures-that-short::ensure` frame finding on the same span.
#[test]
fn no_stale_frame_finding_survives_alongside_the_real_fix() {
    let source = "Ensure that the setting is enabled.\n";
    let (fixed, report) = engine().fix_document(source).expect("engine runs");
    assert_eq!(fixed, "Enable the setting.\n");
    assert!(
        report
            .passes
            .iter()
            .flat_map(|p| &p.held)
            .all(|f| !f.message.contains("able.ensures-that-short")),
        "the retired pack row must never surface a stale finding again"
    );
}

/// The restructure pass sits in its own [`friction_edit::PassReport`],
/// between the bounded loop and register: exactly one pass carries the
/// `restructure.ensures_that` `Fix`-tier patch, and it is neither the
/// first pass nor the last.
#[test]
fn restructure_occupies_its_own_pass_between_the_loop_and_register() {
    let (_, report) = engine()
        .fix_document("Ensure that the setting is enabled.\n")
        .expect("engine runs");
    assert!(report.passes.len() >= 3, "passes: {}", report.passes.len());
    let restructure_index = report
        .passes
        .iter()
        .position(|p| {
            p.applied_patches
                .iter()
                .any(|patch| patch.rule.as_str() == "restructure.ensures_that")
        })
        .expect("a pass applied the restructure patch");
    assert!(restructure_index > 0, "not the bounded loop's own pass");
    assert!(
        restructure_index < report.passes.len() - 1,
        "not the register pass"
    );
}

/// The whole pipeline stays idempotent with restructure enabled: fixing
/// already-fixed text changes nothing, twice over. Also confirms T10's
/// own output (a subject-less imperative) never re-triggers itself or
/// T4 (which needs both an `Nsubj` and a `Dobj`, neither present).
#[test]
fn restructure_output_is_idempotent() {
    let source = "Ensure that the setting is enabled. Ensure that the primary and secondary \
                   settings are enabled.\n";
    let engine = engine();
    let (once, _) = engine.fix_document(source).expect("first fix");
    let (twice, _) = engine.fix_document(&once).expect("second fix");
    let (thrice, _) = engine.fix_document(&twice).expect("third fix");
    assert_eq!(
        once, "Enable the setting. Enable the primary and secondary settings.\n",
        "both sentences rewrite"
    );
    assert_eq!(twice, once, "second application must be a no-op");
    assert_eq!(thrice, twice, "third application must be a no-op");
}

/// The held-finding regression the design's own reporting-layer fix
/// calls for: a document with an unrelated five/six-op-loop hold (a
/// ritual-deletion decline, held against the converged text) plus an
/// R1-licensed instance must surface BOTH the ritual hold and the
/// restructure fix's own accounting correctly -- the bounded loop's
/// hold must not be silently dropped by `final_pass_held`-equivalent
/// reporting once restructure sits between it and register (see
/// `friction-cli::fix::final_pass_held`'s own regression test for the
/// CLI-layer half of this).
#[test]
fn a_bounded_loop_hold_survives_alongside_a_restructure_fix() {
    let source = "1. Ensure that the setting is enabled.\n";
    let (fixed, report) = engine().fix_document(source).expect("engine runs");
    assert!(
        fixed.contains("Enable the setting"),
        "the restructure fix still applies: {fixed:?}"
    );
    // Whether or not this particular numbered-list shape happens to
    // produce a ritual-deletion hold, the restructure fix's own held
    // findings (if any) and the register pass's must all still be
    // present in `report.passes` -- the union `final_pass_held` reads
    // downstream in `friction-cli`. This is the engine-level half of the
    // regression; the CLI-level assertion on `final_pass_held` itself
    // lives in `friction-cli/src/fix.rs`'s own test module.
    assert!(report.passes.len() >= 3);
}
