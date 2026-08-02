//! Integration coverage for the `frame.dejust` operation, run through the
//! real engine (embedded tagger, packs) — `friction-match`'s own
//! `src/frame.rs` tests already cover the template matcher in isolation;
//! this file checks the gated deletion itself: fires on the canonical
//! shape, is idempotent, covers the `merely` marker, and declines every
//! guard-token case.

use friction_core::RuleId;
use friction_edit::Engine;

const RULE_FRAME_DEJUST: RuleId = RuleId::new("frame.dejust");

fn frame_dejust_fired(report: &friction_edit::EditReport) -> bool {
    report.passes.iter().any(|pass| {
        pass.applied_patches
            .iter()
            .any(|p| p.rule == RULE_FRAME_DEJUST)
    })
}

/// The canonical sentence from `docs/research/SYNTHESIS.md` §1 and
/// `docs/research/fixtures.json`'s `frame_dejust_provenance_question`
/// fixture: `just` sits strictly between the auxiliary and the
/// coordinating `or`, so it is deleted along with its trailing space,
/// leaving the genuine disjunctive question intact. The "A few
/// architecture questions:" lead-in keeps the auxiliary mid-sentence, so
/// this fixture also exercises the colon-anchored aux search without
/// tripping the engine's own (unrelated) sentence-initial
/// recapitalization guard.
const PROVENANCE_QUESTION: &str = "A few architecture questions: is provenance just frontmatter \
     metadata, or does every derived graph edge, rewrite, and supersession retain immutable \
     lineage back to the exact source chunks and authoring client?";

#[test]
fn dejust_fires_on_the_canonical_sentence() {
    let engine = Engine::new().expect("engine must load");
    let (out, report) = engine
        .fix_document(PROVENANCE_QUESTION)
        .expect("fix_document must succeed on well-formed prose");

    assert!(
        frame_dejust_fired(&report),
        "expected a frame.dejust patch, report: {report:?}"
    );
    assert_eq!(
        out,
        "A few architecture questions: is provenance frontmatter metadata, or does every \
         derived graph edge, rewrite, and supersession retain immutable lineage back to the \
         exact source chunks and authoring client?"
    );
    assert!(!out.contains("just"), "the marker must be gone: {out:?}");
}

/// A second full engine run over the already-fixed output must be a
/// no-op: the marker is gone, so `find_contrast_question` no longer
/// matches, and `frame.dejust` never re-fires.
#[test]
fn dejust_is_idempotent() {
    let engine = Engine::new().expect("engine must load");
    let (once, _) = engine.fix_document(PROVENANCE_QUESTION).unwrap();
    let (twice, report2) = engine.fix_document(&once).unwrap();

    assert_eq!(once, twice, "a second fix pass must change nothing further");
    assert!(
        !frame_dejust_fired(&report2),
        "frame.dejust must not re-fire on already-fixed output: {report2:?}"
    );
}

/// The `merely` marker is deleted exactly like `just`.
#[test]
fn dejust_fires_on_a_merely_case() {
    let engine = Engine::new().expect("engine must load");
    let input = "And is retrieval merely one flat embedding manifold, or does it partition \
                 memory into semantic wells with centroid-drift detection?";
    let (out, report) = engine.fix_document(input).unwrap();

    assert!(
        frame_dejust_fired(&report),
        "expected a frame.dejust patch, report: {report:?}"
    );
    assert!(!out.contains("merely"), "the marker must be gone: {out:?}");
    assert!(
        out.contains("And is retrieval one flat embedding manifold"),
        "the disjunctive question must survive: {out:?}"
    );
}

/// `only` licenses detection (`Channel::Frame` fires) but is never
/// dejusted. It often carries real quantity meaning ("is it only one
/// replica, or...").
#[test]
fn dejust_declines_the_only_marker() {
    let engine = Engine::new().expect("engine must load");
    let input = "is it only one replica, or does it shard the workload across the cluster?";
    let (out, report) = engine.fix_document(input).unwrap();

    assert!(
        !frame_dejust_fired(&report),
        "frame.dejust must never fire on an only-marked question: {report:?}"
    );
    assert!(out.contains("only"), "the marker must survive: {out:?}");
}

/// "just as fast" is the multi-word "as"-guarded sense, not the
/// dismissive particle — `frame.dejust` must decline it, and the
/// question must come out byte-identical (the reject-fixture case from
/// `docs/research/fixtures.json`).
#[test]
fn dejust_declines_the_just_as_guard() {
    let engine = Engine::new().expect("engine must load");
    let input = "Is it just as fast, or slower?";
    let (out, report) = engine.fix_document(input).unwrap();

    assert_eq!(out, input, "a guarded marker must produce no edit at all");
    assert!(
        !frame_dejust_fired(&report),
        "frame.dejust must never fire on a just-as guard case: {report:?}"
    );
}

/// "just about done" is the multi-word "about"-guarded sense.
#[test]
fn dejust_declines_the_just_about_guard() {
    let engine = Engine::new().expect("engine must load");
    let input = "Is it just about done, or does it need another pass?";
    let (out, report) = engine.fix_document(input).unwrap();

    assert_eq!(out, input, "a guarded marker must produce no edit at all");
    assert!(!frame_dejust_fired(&report));
}

/// "not just" is a negated sense, not the bare dismissive particle.
#[test]
fn dejust_declines_the_not_just_guard() {
    let engine = Engine::new().expect("engine must load");
    let input = "Is it not just fast, or is it also correct?";
    let (out, report) = engine.fix_document(input).unwrap();

    assert_eq!(out, input, "a guarded marker must produce no edit at all");
    assert!(!frame_dejust_fired(&report));
}

/// A plain either-or question with no marker at all is untouched, the
/// precision-first design's own baseline case.
#[test]
fn dejust_never_fires_without_a_marker() {
    let engine = Engine::new().expect("engine must load");
    let input = "And is retrieval one flat embedding manifold, or do you partition memory into \
                 semantic wells with centroid-drift detection, split/merge thresholds, and \
                 cross-domain resonance so large projects do not collapse into one topical soup?";
    let (out, report) = engine.fix_document(input).unwrap();

    assert_eq!(out, input, "no dismissive marker means no edit at all");
    assert!(!frame_dejust_fired(&report));
}
