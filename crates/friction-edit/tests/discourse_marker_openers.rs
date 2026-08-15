//! Sentence-initial discourse-marker coverage: what deletes, what stays,
//! and why.
//!
//! `pilot.additionally-` (the pair-mined `additionally ,` delete rule,
//! `frame-rules-en-v1.toml` `rules_pilot`) is a surface-matched literal,
//! not a `<S>`-anchored frame pattern, so it fires wherever "additionally,"
//! occurs, sentence-initial or not. Every sibling discourse marker in the
//! "furthermore/moreover/consequently/notably/in addition/what is
//! more/similarly/likewise/essentially" family was re-measured as a
//! sentence-initial "<marker> ," opener against the shipped corpora (see
//! `frame-rules-en-v1.toml`'s own "SENTENCE-INITIAL DISCOURSE-MARKER
//! AUDIT" preamble comment for the full evidence table) and every one
//! measures human-tilted or unmeasurable — none clear the confirm bar,
//! so none get a deletion rule. They stay untouched, on the same footing
//! as the already-guarded "however"/"for example" connectives.

use friction_edit::Engine;

fn engine() -> Engine {
    Engine::new().expect("engine must load: embedded tagger/packs are always present")
}

/// The shipped delete rule for "additionally," recapitalizes the exposed
/// opener when it fires sentence-initially.
#[test]
fn additionally_opener_deletes_and_recapitalizes() {
    let (fixed, _) = engine()
        .fix_document("Additionally, it reduces load.\n")
        .expect("engine runs");
    assert_eq!(fixed, "It reduces load.\n");
}

/// `pilot.additionally-` is a bare surface literal, not a `<S>`-anchored
/// frame pattern: it fires mid-sentence too, wherever "additionally,"
/// occurs — there is no positional gate on this particular rule.
#[test]
fn additionally_delete_also_fires_mid_sentence() {
    let (fixed, _) = engine()
        .fix_document("The tool works well; additionally, it reduces load.\n")
        .expect("engine runs");
    assert_eq!(fixed, "The tool works well; it reduces load.\n");
}

/// "Furthermore," measures human-tilted as a sentence-initial opener
/// (`m_pm=13.63`, `h_pm=21.28` — DIRECTION-ERROR) and must pass through
/// untouched, exactly like the already-guarded "however".
#[test]
fn furthermore_opener_is_untouched() {
    let input = "Furthermore, it reduces load.\n";
    let (fixed, _) = engine().fix_document(input).expect("engine runs");
    assert_eq!(
        fixed, input,
        "furthermore is human-tilted; it must not delete"
    );
}

/// "Moreover," measures weak (ratio 1.20, below the confirm bar) as a
/// sentence-initial opener and must pass through untouched.
#[test]
fn moreover_opener_is_untouched() {
    let input = "Moreover, it reduces load.\n";
    let (fixed, _) = engine().fix_document(input).expect("engine runs");
    assert_eq!(
        fixed, input,
        "moreover has no confirmed machine tilt; it must not delete"
    );
}

/// "Notably," measures human-tilted (`m_pm=2.73`, `h_pm=6.08` —
/// DIRECTION-ERROR) as a sentence-initial opener and must pass through
/// untouched.
#[test]
fn notably_opener_is_untouched() {
    let input = "Notably, it reduces load.\n";
    let (fixed, _) = engine().fix_document(input).expect("engine runs");
    assert_eq!(fixed, input, "notably is human-tilted; it must not delete");
}

/// "Similarly," measures human-tilted (`m_pm=5.45`, `h_pm=30.41` —
/// DIRECTION-ERROR) as a sentence-initial opener and must pass through
/// untouched.
#[test]
fn similarly_opener_is_untouched() {
    let input = "Similarly, it reduces load.\n";
    let (fixed, _) = engine().fix_document(input).expect("engine runs");
    assert_eq!(
        fixed, input,
        "similarly is human-tilted; it must not delete"
    );
}

/// "Likewise," measures human-tilted (`m_pm=0.0`, `h_pm=12.16` —
/// DIRECTION-ERROR) as a sentence-initial opener and must pass through
/// untouched.
#[test]
fn likewise_opener_is_untouched() {
    let input = "Likewise, it reduces load.\n";
    let (fixed, _) = engine().fix_document(input).expect("engine runs");
    assert_eq!(fixed, input, "likewise is human-tilted; it must not delete");
}

/// "In addition," measures human-tilted as a sentence-initial opener
/// (`m_pm=0.0`, `h_pm=18.24` — DIRECTION-ERROR), consistent with the
/// already-shipped bare guard `phg.in-addition` (`m_pm=8.8`, `h_pm=184.3`).
/// It must pass through untouched.
#[test]
fn in_addition_opener_is_untouched() {
    let input = "In addition, it reduces load.\n";
    let (fixed, _) = engine().fix_document(input).expect("engine runs");
    assert_eq!(
        fixed, input,
        "in addition is human-tilted; it must not delete"
    );
}

/// "Consequently," and "What is more," are unmeasurable at the current
/// corpus size (zero occurrences on both sides) and so stay excluded
/// too: no-evidence never promotes to a deletion rule.
#[test]
fn consequently_and_what_is_more_openers_are_untouched() {
    let (fixed, _) = engine()
        .fix_document("Consequently, it reduces load.\n")
        .expect("engine runs");
    assert_eq!(fixed, "Consequently, it reduces load.\n");
    let (fixed, _) = engine()
        .fix_document("What is more, it reduces load.\n")
        .expect("engine runs");
    assert_eq!(fixed, "What is more, it reduces load.\n");
}

/// "Essentially," measures weak (ratio 1.35, below the confirm bar) as
/// a sentence-initial opener and must pass through untouched.
#[test]
fn essentially_opener_is_untouched() {
    let input = "Essentially, it reduces load.\n";
    let (fixed, _) = engine().fix_document(input).expect("engine runs");
    assert_eq!(
        fixed, input,
        "essentially has no confirmed machine tilt; it must not delete"
    );
}

/// The already-shipped sentence-initial-opener deletion family
/// (importantly/crucially/ultimately/overall/in conclusion) is
/// unaffected by this audit and keeps firing, with recapitalization.
#[test]
fn already_confirmed_openers_still_delete() {
    let (fixed, _) = engine()
        .fix_document("Ultimately, it reduces load.\n")
        .expect("engine runs");
    assert_eq!(fixed, "It reduces load.\n");
    let (fixed, _) = engine()
        .fix_document("Crucially, it reduces load.\n")
        .expect("engine runs");
    assert_eq!(fixed, "It reduces load.\n");
    let (fixed, _) = engine()
        .fix_document("Overall, it reduces load.\n")
        .expect("engine runs");
    assert_eq!(fixed, "It reduces load.\n");
    let (fixed, _) = engine()
        .fix_document("In conclusion, it reduces load.\n")
        .expect("engine runs");
    assert_eq!(fixed, "It reduces load.\n");
}

/// "However," stays the established precedent this audit follows: a
/// human-normal connective must never delete even though it also
/// appears in machine text.
#[test]
fn however_opener_is_untouched() {
    let input = "However, it reduces load.\n";
    let (fixed, _) = engine().fix_document(input).expect("engine runs");
    assert_eq!(fixed, input);
}

/// The detached ", deliberately" flourish (27x machine-tilted, grep-
/// measured 2026-08-12) deletes in its comma-apposed and pre-colon
/// shapes; the bare modifier form is meaning-bearing ("deliberately
/// simple" — deleting it flips intent to accident) and only reports.
#[test]
fn detached_deliberately_deletes_and_bare_form_survives() {
    let (fixed, _) = engine()
        .fix_document("It reads credentials from the config, deliberately, so the file is the single source.\n")
        .expect("engine runs");
    assert_eq!(
        fixed,
        "It reads credentials from the config, so the file is the single source.\n"
    );

    let (fixed, _) = engine()
        .fix_document("The defaults are strict, deliberately: loose defaults hide bugs.\n")
        .expect("engine runs");
    assert_eq!(
        fixed,
        "The defaults are strict: loose defaults hide bugs.\n"
    );

    let source = "The scoring is deliberately simple enough to reason about.\n";
    let (fixed, _) = engine().fix_document(source).expect("engine runs");
    assert_eq!(fixed, source, "bare modifier form must never delete");
}

/// A parenthetical adverb between a finite verb and its complement
/// deletes with BOTH commas — keeping one would splice the verb from
/// what it links to ("The config is, honestly, the weakest part" must
/// not become "is, the"). At a clause boundary the single kept comma
/// remains correct, pinned by the deliberately test above.
#[test]
fn parenthetical_between_verb_and_complement_drops_both_commas() {
    let (fixed, _) = engine()
        .fix_document("The config is, honestly, the weakest part of the design.\n")
        .expect("engine runs");
    assert_eq!(fixed, "The config is the weakest part of the design.\n");

    let (fixed, _) = engine()
        .fix_document("The config is, deliberately, the single source of truth.\n")
        .expect("engine runs");
    assert_eq!(fixed, "The config is the single source of truth.\n");
}

/// "load-bearing" is collocation-scoped: the metaphor rewrites over a
/// curated abstract head when the seams attest, and the literal
/// architectural sense (walls) can never match — no wall collocation is
/// in the family. The hyphenated leading literal must never arm
/// agreeing inflection ("load-bearing" would classify by "bearing" as a
/// gerund and realize "importanting").
#[test]
fn load_bearing_rewrites_abstract_heads_and_never_walls() {
    let (fixed, _) = engine()
        .fix_document("The doc carries two load-bearing assumptions about ordering, and one load-bearing assumption held.\n")
        .expect("engine runs");
    assert!(
        fixed.contains("important assumption held"),
        "singular head with attested seams must rewrite: {fixed}"
    );

    let source = "We understand this wall may be load-bearing given the age of the house.\n";
    let (fixed, _) = engine().fix_document(source).expect("engine runs");
    assert_eq!(
        fixed, source,
        "the literal architectural sense must never match"
    );
}
