//! End-to-end coverage for the frame-rewrite operation: agreement-
//! preserving rewrites, report-only findings, gate holds, and the
//! idempotence of the whole pipeline with the operation enabled.

use friction_edit::Engine;

fn engine() -> Engine {
    Engine::new().expect("embedded packs and models load")
}

/// A lemma-matched rewrite realizes its target in the matched token's
/// form: past tense stays past tense through the inflection tables.
#[test]
fn rewrite_agrees_with_a_past_tense_match() {
    let (fixed, _) = engine()
        .fix_document("We utilized the cache for speed.\n")
        .expect("engine runs");
    assert_eq!(fixed, "We used the cache for speed.\n");
}

/// Third-person-singular agreement holds too.
#[test]
fn rewrite_agrees_with_a_third_singular_match() {
    let (fixed, _) = engine()
        .fix_document("It showcases the results.\n")
        .expect("engine runs");
    assert_eq!(fixed, "It shows the results.\n");
}

/// A sentence-initial match rewrites with the capitalization
/// transferred, keeping a multi-word target's particle. (Mid-sentence
/// inflected variants of this rule mostly hold at the seam-bigram
/// gate. The corpus is too small to attest most (context, new-word)
/// pairs, and holding is the designed fail-closed behavior; the
/// irregular realization itself is pinned in friction-nlp's inflect
/// tests.)
#[test]
fn sentence_initial_rewrite_transfers_capitalization() {
    let (fixed, _) = engine()
        .fix_document("Expedite the rollout now.\n")
        .expect("engine runs");
    assert_eq!(fixed, "Speed up the rollout now.\n");
}

/// A report-only rule (unauthored target) never edits itself. It
/// surfaces a finding carrying its measured evidence. A separate
/// rewrite rule over the same span may still edit (here the
/// adjudicated adjective deletion), and the report documents the
/// original text regardless.
#[test]
fn report_only_rule_emits_a_finding() {
    let source = "The project has a vibrant community around the toolkit.\n";
    let (fixed, report) = engine().fix_document(source).expect("engine runs");
    assert_eq!(
        fixed, "The project has a community around the toolkit.\n",
        "the adjective-deletion rewrite applies"
    );
    assert!(
        report
            .passes
            .iter()
            .flat_map(|p| &p.held)
            .any(
                |f| f.rule.as_str() == "col.vibrant-community" && f.message.contains("report-only")
            ),
        "the collocation must surface as a report finding"
    );
}

/// A gate-held rewrite candidate becomes a Suggest finding naming the
/// gate, and the text stays unchanged.
#[test]
fn gate_held_rewrite_surfaces_as_a_finding() {
    let source = "The tool notifies the operator when the job fails.\n";
    let (fixed, report) = engine().fix_document(source).expect("engine runs");
    assert_eq!(fixed, source);
    assert!(
        report
            .passes
            .iter()
            .flat_map(|p| &p.held)
            .any(|f| f.rule.as_str() == "vsub.notify" && f.message.contains("held")),
        "the declined rewrite must surface as a held finding"
    );
}

/// The whole pipeline stays idempotent with the frame operation on:
/// fixing already-fixed text changes nothing, twice over.
#[test]
fn frame_rewrites_are_idempotent() {
    let source = "We utilized the cache to expedite the build. The team leveraged automation \
                  and commenced testing. It showcases the results and fosters collaboration \
                  alongside the dashboard.\n";
    let engine = engine();
    let (once, _) = engine.fix_document(source).expect("first fix");
    let (twice, _) = engine.fix_document(&once).expect("second fix");
    let (thrice, _) = engine.fix_document(&twice).expect("third fix");
    assert_eq!(once, twice, "second application must be a no-op");
    assert_eq!(twice, thrice, "third application must be a no-op");
}

/// The `vsub.harness` split gates the harness->use rewrite on a verb
/// tag: noun "harness" (a test harness, this crate's own module docs
/// describing "the harness") must come back byte-identical. Both
/// sentences here are real misfires the ungated rule produced on this
/// repository's own comments ("half of the use", "in this use").
#[test]
fn noun_harness_is_never_rewritten() {
    let engine = engine();
    for source in [
        "This is the string-level half of the harness: fixtures are contracts.\n",
        "It is never the sole decider for either ordering in this harness.\n",
        "The test harness runs nightly.\n",
    ] {
        let (fixed, _) = engine.fix_document(source).expect("engine runs");
        assert_eq!(fixed, source, "noun harness must stay untouched");
    }
}

/// The same split still rewrites genuine verb usages, in the matched
/// tag's own form, across the inflection paradigm.
#[test]
fn verb_harness_still_rewrites_per_form() {
    let engine = engine();
    for (source, expected) in [
        (
            "We harness existing tooling to cut costs.\n",
            "We use existing tooling to cut costs.\n",
        ),
        (
            "It harnesses the cache for speed.\n",
            "It uses the cache for speed.\n",
        ),
        (
            "Harnessing the scheduler proved difficult.\n",
            "Using the scheduler proved difficult.\n",
        ),
        (
            "The power is harnessed by the runtime.\n",
            "The power is used by the runtime.\n",
        ),
    ] {
        let (fixed, _) = engine.fix_document(source).expect("engine runs");
        assert_eq!(fixed, expected, "verb harness must still rewrite");
    }
}

/// A hard-wrapped sentence puts a mid-sentence match at a physical
/// line start; the rewrite must keep its lowercase form. The casing
/// decision keys on sentence position (the first real character
/// walking back, across the wrap), never on line position — a `\n`
/// inside a paragraph is formatting, not punctuation.
#[test]
fn hard_wrapped_mid_sentence_rewrite_stays_lowercase() {
    let (fixed, _) = engine()
        .fix_document("The script commits the result\nalongside the manifest.\n")
        .expect("engine runs");
    assert_eq!(fixed, "The script commits the result\nwith the manifest.\n");
}
