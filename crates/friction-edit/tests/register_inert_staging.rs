//! The staged-inert contract for the three newest register features:
//! `comma_and`, `past_progressive`, and `contrast_closer` landed their
//! counters, transducers (the first two), and wiring in this same change,
//! but deliberately with NO band entries in
//! `crates/friction-packs/packs/register-en-v1.toml`. `RegisterPack::band`
//! returns `None` for all three, so every armed-feature check in
//! `friction-edit::register` that gates on `register_pack.band(name)`
//! short-circuits to "does not need fixing" for them, and the engine's
//! output must be byte-identical to what it produced before this change
//! landed. The measured band entries -- and the behavior change they
//! license -- are a separate, later step; this file is what proves today's
//! wiring is inert until then.

use friction_core::RuleId;
use friction_edit::Engine;

const RULE_COMMA_AND: RuleId = RuleId::new("register.comma_and");
const RULE_PAST_PROGRESSIVE: RuleId = RuleId::new("register.past_progressive");
const RULE_CONTRAST: RuleId = RuleId::new("register.contrast_closer");

/// The embedded pack has no band for any of the three staged features --
/// documenting the contract by construction, not just by absence of a
/// failing test. The bands arrive with a separate, later change that adds
/// `[features.comma_and]`, `[features.past_progressive]`, and
/// `[features.contrast_closer]` tables to `register-en-v1.toml`; until
/// then, this assertion is exactly what keeps the wiring in this change
/// disarmed.
#[test]
fn the_three_staged_features_have_no_band_in_the_embedded_pack() {
    let pack = &friction_packs::REGISTER.pack;
    for feature in ["comma_and", "past_progressive", "contrast_closer"] {
        assert!(
            pack.band(feature).is_none(),
            "expected NO band for staged feature {feature:?} -- if this fails, the measured \
             bands landed and this whole file's contract (and its name) is obsolete"
        );
    }
}

/// A small document stuffed with all three staged constructs -- a
/// `", and "` splice joining two independent clauses (the `comma_and`/T8
/// shape), a `"was"`/`"were"` + `VBG` past progressive (the
/// `past_progressive`/T9 shape), and both `contrast_closer` literals
/// (`"rather than"` and `", not "`) -- comes back byte-identical from
/// [`Engine::fix_document`] today. Deliberately free of every OTHER
/// armed feature's own trigger: no em dash, no semicolon, no
/// nominalizing-suffix noun, short enough that agentless-passive's
/// Wilson upper bound never clears the band regardless (see
/// `friction-edit::register`'s own Wilson-bound docs) -- so a failure
/// here is attributable to one of the three staged features, not a
/// pre-existing armed one firing on unrelated grounds.
const STUFFED: &str = "The scanner reads each file once, and it never touches what it reads. \
     It was checking every line before it moved to the next file. The team chose caching \
     rather than a cache-aside pattern for the lookup. The report is clear, not vague, about \
     what changed. The tool prints one line per file, and it flushes output before it exits. \
     The worker was retrying the failed step until the queue drained. The build finished \
     without incident, and no extra step was needed afterward.";

#[test]
fn a_document_stuffed_with_all_three_staged_constructs_is_byte_identical() {
    // Sanity-check the fixture actually exercises what it claims to,
    // before asserting the (much less interesting) no-op case.
    assert!(
        STUFFED.contains(", and "),
        "fixture must contain a comma_and splice"
    );
    assert!(
        STUFFED.contains("was checking") || STUFFED.contains("was retrying"),
        "fixture must contain a past-progressive pair"
    );
    assert!(
        STUFFED.to_lowercase().contains("rather than"),
        "fixture must contain a \"rather than\" contrast closer"
    );
    assert!(
        STUFFED.contains(", not "),
        "fixture must contain a \", not \" contrast closer"
    );

    let engine = Engine::new().expect("engine must load");
    let (fixed, report) = engine
        .fix_document(STUFFED)
        .expect("fix_document must succeed on well-formed prose");

    assert_eq!(
        fixed, STUFFED,
        "a document exercising only staged (bandless) features must come back byte-identical"
    );

    for pass in &report.passes {
        assert!(
            pass.applied_patches
                .iter()
                .all(|p| p.rule != RULE_COMMA_AND && p.rule != RULE_PAST_PROGRESSIVE),
            "no register.comma_and or register.past_progressive patch may fire while both \
             features are bandless: {report:?}"
        );
        assert!(
            pass.held.iter().all(|f| f.rule != RULE_CONTRAST),
            "register.contrast_closer must never surface a finding while bandless: {report:?}"
        );
    }
}
