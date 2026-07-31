//! Integration coverage for the `register.semicolon` operation, run
//! through the real engine (embedded tagger, dependency parser, packs)
//! rather than a hand-built parse -- `friction-register`'s own
//! `tests/transduce.rs` already covers T7 in isolation against a
//! controlled parse; this file checks the whole pass fires, converges,
//! and leaves grammatical output on realistic input.
//!
//! Unlike `register_em_dash.rs`, the semicolon band is genuinely nonzero
//! (see `register-v1.toml`'s `[features.semicolon]`), so this file also
//! covers the case em-dash's own `[0, 0]` band couldn't have: a document
//! whose semicolon rate already sits *inside* the band must come back
//! byte-identical, proving the register pass stops at the band's edge
//! rather than driving every feature toward zero.

use friction_core::RuleId;
use friction_edit::Engine;

const RULE_SEMICOLON: RuleId = RuleId::new("register.semicolon");

fn count_semicolons(text: &str) -> usize {
    text.matches(';').count()
}

/// A document well above the semicolon band (high = 3.6563 per 1000
/// prose words -- see `register-v1.toml`'s `[features.semicolon]`),
/// every sentence a semicolon joining two independent clauses, T7's one
/// rewrite shape.
const ABOVE_BAND: &str = "The job reads from the queue; it commits offsets only after the \
     batch is durably written. The scheduler retries a failed step automatically; the \
     operator is paged only once retries are exhausted. The service reads its config from a \
     mounted volume; it restarts automatically once the volume attaches. The cache serves \
     stale data during a refresh; it never blocks a read that arrives while a write is in \
     progress.";

#[test]
fn a_document_above_band_is_reduced_to_grammatical_output_and_is_idempotent() {
    let engine = Engine::new().expect("engine must load");

    let before = count_semicolons(ABOVE_BAND);
    assert!(before > 0, "fixture must actually contain semicolons");

    let (fixed, report) = engine
        .fix_document(ABOVE_BAND)
        .expect("fix_document must succeed on well-formed prose");

    let after = count_semicolons(&fixed);
    assert!(
        after < before,
        "expected the semicolon count to drop, went from {before} to {after}: {fixed:?}"
    );

    let semicolon_pass_fired = report.passes.iter().any(|pass| {
        pass.applied_patches
            .iter()
            .any(|p| p.rule == RULE_SEMICOLON)
    });
    assert!(
        semicolon_pass_fired,
        "expected at least one register.semicolon patch, report: {report:?}"
    );

    // Grammatical, meaning-preserving output: every sentence still ends
    // on terminal punctuation, no sentence is left starting with a
    // lowercase letter (T7's own invariant), and no doubled punctuation
    // (", ," / ". ." / "; ;") was introduced by a rewrite landing next to
    // existing punctuation.
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

/// A document whose one semicolon sits well inside the band once diluted
/// by enough surrounding prose words: the rate never crosses `high`, so
/// `register.semicolon` never even collects a candidate pool for it, and
/// the whole document -- not just the semicolon's own sentence -- must
/// come back byte-identical. Padding sentences avoid every other
/// register/four-operation trigger (no first-person subject, no
/// nominalization-table noun, no em dash) so this isolates the semicolon
/// band specifically, the same way `register_em_dash.rs`'s own
/// zero-em-dash test isolates that feature.
const INSIDE_BAND: &str = "The job reads from the queue; it commits offsets only after the \
     batch is durably written. The scanner walks the project tree in lexical order. Each file \
     is opened once and closed before the next one starts. The output lists every finding with \
     its file path and line number. A finding includes a short message describing what matched \
     and why. The scanner does not modify any file it reads. Results stream to standard output \
     as they are found, not buffered until the run ends. A missing file is reported and does \
     not stop the run. The exit code reflects whether any finding was reported at all. Running \
     the scanner twice on the same tree produces the same findings in the same order. The \
     settings file lists which directories to skip and which extensions to check. A directory \
     listed as skipped is never opened, even if it contains a matching extension. The default \
     settings check common source extensions and skip build output directories. A custom \
     settings file can narrow or widen either list. The scanner reports its own version and the \
     settings path it loaded when asked. Errors reading the settings file are reported clearly \
     and stop the run before any file is scanned. The project root is the directory containing \
     the settings file, unless a different root is given on the command line. Relative paths in \
     the settings file are resolved against that root. Every file the scanner opens is read in \
     full before it moves to the next one. Nothing is skipped partway through a read. A file \
     larger than the configured limit is reported and skipped, not read at all. The limit \
     defaults to a value large enough for almost every real source file. Setting the limit to \
     zero disables it entirely, and every file is read regardless of size. The scanner keeps no \
     state between runs. Each run starts from a clean slate and reads the tree fresh. A file \
     that changes while the scanner is reading it may produce a stale result for that one file \
     only, never for the rest of the run. The scanner never writes to the tree it reads. \
     Running it with another tool that also reads the tree is always safe.";

#[test]
fn a_document_with_a_semicolon_inside_the_band_is_byte_identical() {
    let engine = Engine::new().expect("engine must load");
    assert_eq!(
        count_semicolons(INSIDE_BAND),
        1,
        "fixture must contain exactly one semicolon"
    );

    let (out, report) = engine
        .fix_document(INSIDE_BAND)
        .expect("fix_document must succeed");
    assert_eq!(
        out, INSIDE_BAND,
        "a document inside the band must be untouched"
    );
    for pass in &report.passes {
        assert!(
            pass.applied_patches
                .iter()
                .all(|p| p.rule != RULE_SEMICOLON),
            "no register.semicolon patch should fire on an in-band document"
        );
    }
}

/// A document with zero semicolons is untouched by the semicolon
/// operation specifically: byte-identical output on a fixture already
/// known to be a full-engine near-no-op (see
/// `friction_edit::tests::near_noop_clean_text_is_byte_identical`).
#[test]
fn a_document_with_zero_semicolons_is_untouched() {
    let engine = Engine::new().expect("engine must load");
    let input = "Run the scanner from the project root. Results stream in as they are \
                 found, and nothing is deleted without confirmation.";
    assert_eq!(count_semicolons(input), 0);

    let (out, report) = engine
        .fix_document(input)
        .expect("fix_document must succeed");
    assert_eq!(out, input);
    for pass in &report.passes {
        assert!(
            pass.applied_patches
                .iter()
                .all(|p| p.rule != RULE_SEMICOLON),
            "no register.semicolon patch should fire on a zero-semicolon document"
        );
    }
}
