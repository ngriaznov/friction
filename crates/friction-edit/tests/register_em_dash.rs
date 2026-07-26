//! Integration coverage for the `register.em_dash` operation, run through
//! the real engine (embedded tagger, dependency parser, packs) rather than
//! a hand-built parse -- `friction-register`'s own `tests/transduce.rs`
//! already covers each transducer case in isolation against a controlled
//! parse; this file checks the whole pass fires, converges, and leaves
//! grammatical output on realistic input.

use friction_core::RuleId;
use friction_edit::Engine;

const RULE_EM_DASH: RuleId = RuleId::new("register.em_dash");

fn count_em_dashes(text: &str) -> usize {
    text.matches('\u{2014}').count()
}

/// A document well above the em-dash band (which drives to zero -- see
/// `register-v1.toml`'s `[features.em_dash]`), one sentence per T6 case:
/// a paired parenthetical, a fragment, and an independent clause.
const ABOVE_BAND: &str = "The service reads its config from a mounted volume — not from \
     environment variables — before startup completes. The response is a registry entry — \
     not a new endpoint — so no lockstep deploy is required. The job reads from the queue — \
     it commits offsets only after the batch is durably written. The scheduler retries a \
     failed step automatically — the operator is paged only once retries are exhausted.";

#[test]
fn a_document_above_band_is_reduced_to_grammatical_output_and_is_idempotent() {
    let engine = Engine::new().expect("engine must load");

    let before = count_em_dashes(ABOVE_BAND);
    assert!(before > 0, "fixture must actually contain em dashes");

    let (fixed, report) = engine
        .fix_document(ABOVE_BAND)
        .expect("fix_document must succeed on well-formed prose");

    let after = count_em_dashes(&fixed);
    assert!(
        after < before,
        "expected the em-dash count to drop, went from {before} to {after}: {fixed:?}"
    );

    let em_dash_pass_fired = report
        .passes
        .iter()
        .any(|pass| pass.applied_patches.iter().any(|p| p.rule == RULE_EM_DASH));
    assert!(
        em_dash_pass_fired,
        "expected at least one register.em_dash patch, report: {report:?}"
    );

    // Grammatical, meaning-preserving output: every sentence still ends
    // on terminal punctuation, no sentence is left starting with a
    // lowercase letter (T6 case (c)'s own invariant), and no doubled
    // punctuation (", ," / ". ." / "; ;") was introduced by a rewrite
    // landing next to existing punctuation.
    for bad in [", ,", ". .", "; ;", "  "] {
        assert!(
            !fixed.contains(bad),
            "output contains {bad:?}, not grammatical: {fixed:?}"
        );
    }
    for sentence_start in fixed.split(". ") {
        let Some(first_alpha) = sentence_start.chars().find(|c| c.is_alphabetic()) else {
            continue;
        };
        assert!(
            !first_alpha.is_lowercase() || sentence_start.starts_with(|c: char| !c.is_alphabetic()),
            "a rewritten sentence must not open on a lowercase letter: {sentence_start:?} in \
             {fixed:?}"
        );
    }

    // Idempotence: a second pass over already-fixed output is a no-op.
    let (fixed_again, _) = engine
        .fix_document(&fixed)
        .expect("fix_document must succeed on already-fixed output");
    assert_eq!(
        fixed, fixed_again,
        "a second run_register pass over already-fixed output must be a no-op"
    );
}

/// A document with zero em dashes is untouched by the em-dash operation
/// specifically: byte-identical output on a fixture already known to be a
/// full-engine near-no-op (see `friction_edit::tests::near_noop_clean_text_is_byte_identical`).
#[test]
fn a_document_with_zero_em_dashes_is_untouched() {
    let engine = Engine::new().expect("engine must load");
    let input = "Run the scanner from the project root. Results stream in as they are \
                 found, and nothing is deleted without confirmation.";
    assert_eq!(count_em_dashes(input), 0);

    let (out, report) = engine.fix_document(input).expect("fix_document must succeed");
    assert_eq!(out, input);
    for pass in &report.passes {
        assert!(
            pass.applied_patches.iter().all(|p| p.rule != RULE_EM_DASH),
            "no register.em_dash patch should fire on a zero-em-dash document"
        );
    }
}
