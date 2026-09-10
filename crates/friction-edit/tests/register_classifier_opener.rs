//! Activation coverage for the `classifier_opener` register tell: the
//! numbered-classifier sentence opener ("Two tempos: ...") announcing a
//! scheme before delivering it. Detect-only, like `contrast_closer`:
//! the shipped parser cannot attest that a sentence's delivery carries
//! as many top-level conjuncts as the opener announces (measured — see
//! `friction_register::features::classifier_openers`'s own docs,
//! including the mismatch case where a naive conjunct count fails
//! open), so a licensed deletion was declined and every instance is a
//! held Suggest finding instead.
//!
//! Corpus evidence (2026-09-10 sweep): corpus/human 0 instances in
//! 338,962 words; corpus/llm 10 in 375,526 (26.6/M);
//! corpus/review/machine 2 in 66,797. Perfect one-sided separation on a
//! thin count — the curated-seed reading, with the human-corpus
//! precision gauntlet (zero findings introduced on corpus/human at
//! admission) as the bar.
//!
//! Contrast with the shipped `con.one-caveat-colon` frame deletion:
//! "One caveat: X" deletes safely because a singular lead-in promises
//! no count for the delivery to honor; the plural classifier opener's
//! whole hazard is that promise, so its deletion needed the very count
//! attestation the parser cannot give, and detection is the ceiling.

use friction_core::Tier;
use friction_edit::Engine;

fn engine() -> Engine {
    Engine::new().expect("embedded packs and models load")
}

/// The embedded pack carries the measured band: 58 of 58 human
/// docs-genre train documents measure a zero rate, so every bound is
/// exactly 0.0 — the em-dash reading, arming on a single instance.
#[test]
fn the_pack_carries_the_measured_band() {
    let band = friction_packs::REGISTER
        .pack
        .band("classifier_opener")
        .expect("classifier_opener band present");
    assert_eq!((band.low, band.median, band.high), (0.0, 0.0, 0.0));
}

/// The acceptance example the declined deletion was gated on: a real
/// machine-shaped sentence whose opener announces "Two" and whose tail
/// genuinely carries two top-level conjuncts buried in parenthetical
/// comma-lists. Output is byte-identical (detect-only), and exactly one
/// `register.classifier_opener` Suggest finding lands on the opener
/// span itself.
#[test]
fn the_cloudwatch_example_is_detected_and_never_edited() {
    let source = "Two tempos: CloudWatch alarms on discrete events (Keycloak admin changes in \
                  master, failed-login bursts, non-service S3 reads, WAF spikes, GuardDuty) \
                  and a scheduled anomaly job over the audit collections (volume per actor, \
                  off-hours, bulk, cross-facility), both to on-call, plus the response \
                  runbook.\n";
    let (fixed, report) = engine().fix_document(source).expect("engine runs");
    assert_eq!(fixed, source, "detect-only: the text must never be edited");
    let findings: Vec<_> = report
        .passes
        .iter()
        .flat_map(|p| &p.held)
        .filter(|f| f.rule.as_str() == "register.classifier_opener")
        .collect();
    assert_eq!(findings.len(), 1, "held findings: {findings:?}");
    let finding = findings[0];
    assert_eq!(finding.tier, Tier::Suggest);
    assert_eq!(&source[finding.range.clone()], "Two tempos:");
    assert!(
        finding.message.contains("no licensed rewrite"),
        "the finding must say why no edit happened: {finding:?}"
    );
}

/// The count-mismatch shape ("Two tempos:" delivering three items) gets
/// the same finding: detection is independent of what follows the
/// colon, and the deletion this would have had to block is exactly why
/// no deletion ships (the parser reads this tail's "A" as a bare
/// determiner root, so a conjunct count would have come back 2 and
/// licensed the deletion the mismatch must forbid).
#[test]
fn a_count_mismatch_instance_still_gets_the_finding() {
    let source = "Two tempos: alarms, a scheduled job and the response runbook.\n";
    let (fixed, report) = engine().fix_document(source).expect("engine runs");
    assert_eq!(fixed, source);
    assert!(
        report
            .passes
            .iter()
            .flat_map(|p| &p.held)
            .any(|f| f.rule.as_str() == "register.classifier_opener"),
        "held findings: {:?}",
        report
            .passes
            .iter()
            .flat_map(|p| &p.held)
            .collect::<Vec<_>>()
    );
}

/// The anaphora shape that would have blocked a deletion ("the second
/// tempo" referring back to the classifier noun) still gets the
/// detection finding — the reader deletes nothing, so the reference
/// stays intact either way.
#[test]
fn a_recurring_classifier_noun_still_gets_the_finding() {
    let source = "Two tempos: alarms fire on discrete events, and a scheduled job scans the \
                  collections. The second tempo runs hourly.\n";
    let (fixed, report) = engine().fix_document(source).expect("engine runs");
    assert_eq!(fixed, source);
    assert!(
        report
            .passes
            .iter()
            .flat_map(|p| &p.held)
            .any(|f| f.rule.as_str() == "register.classifier_opener"),
        "held findings: {:?}",
        report
            .passes
            .iter()
            .flat_map(|p| &p.held)
            .collect::<Vec<_>>()
    );
}

/// Human-attested neighboring shapes never fire: a digit cardinal, a
/// singular classifier, a colon-less cardinal sentence, and a
/// mid-sentence occurrence.
#[test]
fn neighboring_human_shapes_never_fire() {
    let sources = [
        "12 rules: keep each one short and testable.\n",
        "One caveat: the cache is cold right after a deploy.\n",
        "Two tempos govern the pipeline, and both page on-call.\n",
        "The design settles on two tempos: polling and push.\n",
    ];
    let engine = engine();
    for source in sources {
        let (_, report) = engine.fix_document(source).expect("engine runs");
        assert!(
            report
                .passes
                .iter()
                .flat_map(|p| &p.held)
                .all(|f| f.rule.as_str() != "register.classifier_opener"),
            "no classifier_opener finding expected for {source:?}, got: {:?}",
            report
                .passes
                .iter()
                .flat_map(|p| &p.held)
                .collect::<Vec<_>>()
        );
    }
}

/// Determinism: two runs over the acceptance example produce identical
/// bytes and identical finding sets.
#[test]
fn detection_is_deterministic() {
    let source = "Two tempos: alarms on discrete events, and a scheduled job over the audit \
                  collections.\n";
    let engine = engine();
    let (once, report_a) = engine.fix_document(source).expect("first run");
    let (twice, report_b) = engine.fix_document(source).expect("second run");
    assert_eq!(once, twice);
    let ranges = |report: &friction_edit::EditReport| {
        report
            .passes
            .iter()
            .flat_map(|p| &p.held)
            .filter(|f| f.rule.as_str() == "register.classifier_opener")
            .map(|f| f.range.clone())
            .collect::<Vec<_>>()
    };
    assert_eq!(ranges(&report_a), ranges(&report_b));
    assert!(
        !ranges(&report_a).is_empty(),
        "the zero-high band arms on one instance"
    );
}
