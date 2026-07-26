//! Light-verb-construction (LVC) tables and matching.
//!
//! Shared between detection (`friction-harness::pivot`'s runtime gate,
//! which licenses only the fixed 30-entry [`DERIVATIONAL_LEXICON`]) and
//! offline mining (a corpus tool's LVC-rate measurement and
//! new-candidate discovery, which needs the same surface tables but
//! must probe an arbitrary candidate nominalization, not just the
//! licensed set). This module owns the tables and the two matching
//! functions; deciding which pairs are *licensed* for a runtime rewrite
//! stays a pack concern.
//!
//! Everything here except [`scan_construction_shape`] is a straight
//! relocation of what used to be private tables in
//! `friction-harness::pivot`: [`LightVerbForm`], [`LIGHT_VERBS`]
//! (formerly `LV`), [`BE_FORMS`] (formerly `BE`),
//! [`DERIVATIONAL_LEXICON`] (formerly `DERIV`), and [`conjugate`]
//! (formerly a private `inflect`, renamed to avoid colliding with
//! [`crate::inflect`] — a different, already public,
//! surface-shape-based inflector this crate ships; the two must not
//! share a name).

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;
use std::sync::LazyLock;

use crate::{TaggedToken, Tagger};

/// Which inflectional form a light-verb surface token represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LightVerbForm {
    /// Bare infinitive/present plural form (`perform`, `conduct`,
    /// `make`, `do`).
    Base,
    /// Third-person singular present (`performs`, `conducts`, `makes`,
    /// `does`).
    ThirdSg,
    /// Past tense (`performed`, `conducted`, `made`, `did`).
    Past,
    /// Gerund/present participle (`performing`, `conducting`, `making`,
    /// `doing`).
    Gerund,
}

/// Every base/3sg/past/gerund surface form of the four licensed light
/// verbs, mapped to its inflectional feature.
pub static LIGHT_VERBS: LazyLock<BTreeMap<&'static str, LightVerbForm>> = LazyLock::new(|| {
    BTreeMap::from([
        ("perform", LightVerbForm::Base),
        ("performs", LightVerbForm::ThirdSg),
        ("performed", LightVerbForm::Past),
        ("performing", LightVerbForm::Gerund),
        ("conduct", LightVerbForm::Base),
        ("conducts", LightVerbForm::ThirdSg),
        ("conducted", LightVerbForm::Past),
        ("conducting", LightVerbForm::Gerund),
        ("make", LightVerbForm::Base),
        ("makes", LightVerbForm::ThirdSg),
        ("made", LightVerbForm::Past),
        ("making", LightVerbForm::Gerund),
        ("do", LightVerbForm::Base),
        ("does", LightVerbForm::ThirdSg),
        ("did", LightVerbForm::Past),
        ("doing", LightVerbForm::Gerund),
    ])
});

/// Forms of "be" that mark a preceding past-form light verb as passive.
pub static BE_FORMS: LazyLock<BTreeSet<&'static str>> =
    LazyLock::new(|| BTreeSet::from(["is", "are", "was", "were", "been", "being", "be"]));

/// The licensed (nominalization -> derived verb) candidate table: 31
/// entries.
///
/// A *candidate* table, not an auto-licensed one — a runtime rewrite
/// gate still consults a pack's own licensing decision (today,
/// `friction-harness::pivot` licenses every entry unconditionally, a
/// pack-format simplification this crate doesn't change). Entries are
/// seeded from NOMLEX/CatVar-style noun -> verb mappings, chosen so the
/// derived verb is unambiguous and the nominalization isn't a common
/// artifact noun (deliberately absent: `"approval"`,
/// subject-genitive-prone; `"index"`, an artifact noun rather than an
/// LVC nominalization).
pub static DERIVATIONAL_LEXICON: LazyLock<BTreeMap<&'static str, &'static str>> =
    LazyLock::new(|| {
        BTreeMap::from([
            ("initialization", "initialize"),
            ("configuration", "configure"),
            ("installation", "install"),
            ("validation", "validate"),
            ("optimization", "optimize"),
            ("migration", "migrate"),
            ("deletion", "delete"),
            ("creation", "create"),
            ("execution", "execute"),
            ("analysis", "analyze"),
            ("assessment", "assess"),
            ("evaluation", "evaluate"),
            ("comparison", "compare"),
            ("conversion", "convert"),
            ("integration", "integrate"),
            ("implementation", "implement"),
            ("modification", "modify"),
            ("verification", "verify"),
            ("authentication", "authenticate"),
            ("encryption", "encrypt"),
            ("compression", "compress"),
            ("reduction", "reduce"),
            ("adjustment", "adjust"),
            ("extraction", "extract"),
            ("decision", "decide"),
            ("investigation", "investigate"),
            ("inspection", "inspect"),
            ("calculation", "calculate"),
            ("transformation", "transform"),
            ("aggregation", "aggregate"),
            ("synchronization", "synchronize"),
        ])
    });

/// Produces the inflected form of `verb` matching `form`.
#[must_use]
pub fn conjugate(verb: &str, form: LightVerbForm) -> String {
    match form {
        LightVerbForm::Base => verb.to_string(),
        LightVerbForm::ThirdSg => verb
            .strip_suffix('y')
            .map_or_else(|| format!("{verb}s"), |stem| format!("{stem}ies")),
        LightVerbForm::Past => verb
            .strip_suffix('e')
            .map_or_else(|| format!("{verb}ed"), |stem| format!("{stem}ed")),
        LightVerbForm::Gerund => {
            let stem = verb.strip_suffix('e').unwrap_or(verb);
            format!("{stem}ing")
        }
    }
}

/// Why a candidate light-verb construction was rejected before
/// licensing was checked.
///
/// Relocated from a private enum in `friction-harness::pivot`, which
/// now re-exports this type so its public shape is unchanged; see
/// [`classify_candidate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PivotRejection {
    /// The light verb is a past form preceded by a form of "be".
    Passive,
    /// The token after the (optional) determiner is tagged `JJ` — an
    /// adjective or quantifier modifying the nominalization.
    ModifiedNominal,
    /// The nominalization is plural.
    PluralNominal,
}

/// A licensed `LV (DET)? NOM (of)?` construction found by
/// [`classify_candidate`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LicensedConstruction {
    /// Absolute byte range in whatever `text` [`classify_candidate`]
    /// was called with, from the light verb through the trailing `of`
    /// (if present) or the nominalization (if not).
    pub range: Range<usize>,
    /// Index one past the construction's last token (`of`'s index + 1,
    /// or the nominalization's index + 1) — lets a caller scanning for
    /// every match in a sentence resume immediately after this one
    /// without rescanning tokens already consumed by it.
    pub end_token_index: usize,
    /// The derived verb, inflected to match the light verb's own form,
    /// e.g. `"initializes"`.
    pub derived_verb: Box<str>,
    /// The matched nominalization's lowercased surface text.
    pub nominalization: Box<str>,
}

/// The outcome of [`classify_candidate`] for one candidate light-verb
/// position.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CandidateOutcome {
    /// `tokens[lv_index]` is not a member of [`LIGHT_VERBS`] at all.
    NotLightVerb,
    /// A light verb was found but a guard rejected it before licensing
    /// was even checked.
    Rejected(PivotRejection),
    /// A light verb (optionally followed by a determiner) was found,
    /// but no token follows to be a candidate nominalization.
    NoNominalFollows,
    /// A light-verb-construction shape was found, but the
    /// nominalization is not a key in the caller's `lexicon` — not a
    /// licensed pair.
    Unlicensed,
    /// A licensed light-verb construction, ready to collapse.
    Licensed(LicensedConstruction),
}

/// Classifies the light-verb-construction candidate rooted at
/// `tokens[lv_index]`.
///
/// `tokens` must already carry the caller's real absolute byte offsets
/// (`tagger.tag(text, base_offset)`) and `text` must be the exact
/// string those offsets index into — this function does no tagging
/// itself and is agnostic to whatever `base_offset` the caller used.
///
/// Factored out of `friction-harness::pivot`'s private loop body so
/// both that module (stops at the sentence's first licensed match,
/// matching `ref_pivot.py::pivot()`) and a detection channel (reports
/// every non-overlapping licensed match) share one gate/licensing
/// implementation, differing only in outer-loop policy.
///
/// Gate order mirrors `ref_pivot.py::pivot()`: passive, then
/// modified-nominal (`JJ`), then plural-nominal, then licensing.
#[must_use]
pub fn classify_candidate(
    tokens: &[TaggedToken],
    text: &str,
    lv_index: usize,
    lexicon: &BTreeMap<Box<str>, Box<str>>,
) -> CandidateOutcome {
    let surface = |t: &TaggedToken| -> &str { &text[t.token.range.clone()] };

    let lv_lower = surface(&tokens[lv_index]).to_lowercase();
    let Some(&feat) = LIGHT_VERBS.get(lv_lower.as_str()) else {
        return CandidateOutcome::NotLightVerb;
    };

    if feat == LightVerbForm::Past
        && lv_index > 0
        && BE_FORMS.contains(surface(&tokens[lv_index - 1]).to_lowercase().as_str())
    {
        return CandidateOutcome::Rejected(PivotRejection::Passive);
    }

    let mut j = lv_index + 1;
    if j < tokens.len()
        && matches!(
            surface(&tokens[j]).to_lowercase().as_str(),
            "a" | "an" | "the"
        )
    {
        j += 1;
    }

    if j >= tokens.len() {
        return CandidateOutcome::NoNominalFollows;
    }
    if tokens[j].pos.as_str() == "JJ" {
        return CandidateOutcome::Rejected(PivotRejection::ModifiedNominal);
    }

    let nom = surface(&tokens[j]).to_lowercase();
    if let Some(stem) = nom.strip_suffix('s')
        && lexicon.contains_key(stem)
    {
        return CandidateOutcome::Rejected(PivotRejection::PluralNominal);
    }
    let Some(base_verb) = lexicon.get(nom.as_str()) else {
        return CandidateOutcome::Unlicensed;
    };

    let derived_verb = conjugate(base_verb, feat);

    let k = j + 1;
    let has_of = k < tokens.len() && surface(&tokens[k]).to_lowercase() == "of";
    let end_idx = if has_of { k } else { j };

    let range = tokens[lv_index].token.range.start..tokens[end_idx].token.range.end;

    CandidateOutcome::Licensed(LicensedConstruction {
        range,
        end_token_index: end_idx + 1,
        derived_verb: derived_verb.into_boxed_str(),
        nominalization: nom.into_boxed_str(),
    })
}

/// One `LV (DET)? NOM (of)?` construction-shape match found by
/// [`scan_construction_shape`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConstructionMatch {
    /// The matched light verb's lowercased surface form.
    pub light_verb: String,
    /// Which inflectional form the light verb was in.
    pub light_verb_form: LightVerbForm,
    /// The lowercased candidate token found in nominalization position.
    pub nominalization: String,
    /// Whether the match was followed by a literal `"of"`.
    pub has_of: bool,
    /// The full matched surface text, e.g. `"performs an initialization
    /// of"`.
    pub matched_text: String,
}

/// Scans `sentence` for every `LV (DET)? candidate (of)?` construction
/// shape.
///
/// `candidate` is looked up in `candidate_noms` rather than
/// [`DERIVATIONAL_LEXICON`], and skips the passive/`JJ`/plural gates
/// `friction-harness::pivot`'s runtime `match_pivot` applies before
/// licensing — the same left-to-right walk, stripped down for offline
/// measurement so it can probe nominalizations that aren't (yet)
/// licensed at all.
///
/// Unlike a runtime matcher, this doesn't stop at the first match: it
/// walks every light-verb-table token left to right and reports every
/// position whose following token is a member of `candidate_noms`, so a
/// mining pass can count occurrences across a corpus rather than
/// produce a single per-sentence verdict.
#[must_use]
pub fn scan_construction_shape(
    sentence: &str,
    tagger: &dyn Tagger,
    candidate_noms: &BTreeSet<&str>,
) -> Vec<ConstructionMatch> {
    let tokens = tagger.tag(sentence, 0);
    let surface = |t: &TaggedToken| -> &str { &sentence[t.token.range.clone()] };
    let mut matches = Vec::new();

    for i in 0..tokens.len() {
        let lv_lower = surface(&tokens[i]).to_lowercase();
        let Some(&feat) = LIGHT_VERBS.get(lv_lower.as_str()) else {
            continue;
        };

        let mut j = i + 1;
        if j < tokens.len()
            && matches!(
                surface(&tokens[j]).to_lowercase().as_str(),
                "a" | "an" | "the"
            )
        {
            j += 1;
        }
        if j >= tokens.len() {
            continue;
        }

        let nom = surface(&tokens[j]).to_lowercase();
        if !candidate_noms.contains(nom.as_str()) {
            continue;
        }

        let k = j + 1;
        let has_of = k < tokens.len() && surface(&tokens[k]).to_lowercase() == "of";
        let end_idx = if has_of { k } else { j };
        let matched_text = tokens[i..=end_idx]
            .iter()
            .map(surface)
            .collect::<Vec<_>>()
            .join(" ");

        matches.push(ConstructionMatch {
            light_verb: lv_lower,
            light_verb_form: feat,
            nominalization: nom,
            has_of,
            matched_text,
        });
    }
    matches
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PerceptronTagger;
    use std::sync::OnceLock;

    fn tagger() -> &'static PerceptronTagger {
        static TAGGER: OnceLock<PerceptronTagger> = OnceLock::new();
        TAGGER.get_or_init(|| PerceptronTagger::new().expect("embedded weights must load"))
    }

    #[test]
    fn conjugate_matches_reference_form_map() {
        assert_eq!(conjugate("initialize", LightVerbForm::Base), "initialize");
        assert_eq!(
            conjugate("initialize", LightVerbForm::ThirdSg),
            "initializes"
        );
        assert_eq!(conjugate("decide", LightVerbForm::Past), "decided");
        assert_eq!(conjugate("analyze", LightVerbForm::Gerund), "analyzing");
        assert_eq!(conjugate("verify", LightVerbForm::ThirdSg), "verifies");
    }

    #[test]
    fn classify_candidate_byte_range_is_offset_agnostic() {
        // The real sentence sits at a non-zero offset, proving
        // `classify_candidate` is offset-aware, not hardcoded to
        // `base_offset == 0`.
        let text = "PREFIX. The system performs an initialization of the database.";
        let base_offset = "PREFIX. ".len();
        let sentence = &text[base_offset..];
        let tokens = tagger().tag(sentence, base_offset);
        let lv_index = tokens
            .iter()
            .position(|t| &text[t.token.range.clone()] == "performs")
            .expect("tagger must find \"performs\"");

        let lexicon = BTreeMap::from([(
            Box::from("initialization") as Box<str>,
            Box::from("initialize") as Box<str>,
        )]);

        match classify_candidate(&tokens, text, lv_index, &lexicon) {
            CandidateOutcome::Licensed(lc) => {
                assert_eq!(&text[lc.range.clone()], "performs an initialization of");
                assert_eq!(&*lc.derived_verb, "initializes");
                assert_eq!(&*lc.nominalization, "initialization");
            }
            other => panic!("expected Licensed, got {other:?}"),
        }
    }

    #[test]
    fn classify_candidate_rejects_passive() {
        let text = "An initialization of the database is performed by the system.";
        let tokens = tagger().tag(text, 0);
        let lv_index = tokens
            .iter()
            .position(|t| &text[t.token.range.clone()] == "performed")
            .expect("tagger must find \"performed\"");
        let lexicon = BTreeMap::from([(
            Box::from("initialization") as Box<str>,
            Box::from("initialize") as Box<str>,
        )]);
        assert_eq!(
            classify_candidate(&tokens, text, lv_index, &lexicon),
            CandidateOutcome::Rejected(PivotRejection::Passive)
        );
    }

    #[test]
    fn classify_candidate_reports_not_light_verb_for_a_non_light_verb_token() {
        let text = "The committee reviewed the plan.";
        let tokens = tagger().tag(text, 0);
        let lexicon = BTreeMap::new();
        assert_eq!(
            classify_candidate(&tokens, text, 0, &lexicon),
            CandidateOutcome::NotLightVerb
        );
    }

    #[test]
    fn classify_candidate_reports_unlicensed_for_an_unlicensed_nominalization() {
        let text = "The wizard performs setup.";
        let tokens = tagger().tag(text, 0);
        let lv_index = tokens
            .iter()
            .position(|t| &text[t.token.range.clone()] == "performs")
            .expect("tagger must find \"performs\"");
        let lexicon = BTreeMap::new();
        assert_eq!(
            classify_candidate(&tokens, text, lv_index, &lexicon),
            CandidateOutcome::Unlicensed
        );
    }

    #[test]
    fn scan_construction_shape_finds_licensed_construction() {
        let candidates = BTreeSet::from(["initialization"]);
        let matches = scan_construction_shape(
            "The system performs an initialization of the database.",
            tagger(),
            &candidates,
        );
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].nominalization, "initialization");
        assert!(matches[0].has_of);
        assert_eq!(matches[0].light_verb_form, LightVerbForm::ThirdSg);
    }

    #[test]
    fn scan_construction_shape_probes_beyond_the_licensed_table() {
        // "setup" isn't a `DERIVATIONAL_LEXICON` key, but
        // `scan_construction_shape` takes an arbitrary candidate set,
        // so it must still find this where `match_pivot` would not.
        let candidates = BTreeSet::from(["setup"]);
        let matches = scan_construction_shape("The wizard performs setup.", tagger(), &candidates);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].nominalization, "setup");
        assert!(!matches[0].has_of);
    }

    #[test]
    fn scan_construction_shape_finds_no_light_verb_returns_empty() {
        let candidates = BTreeSet::from(["initialization"]);
        let matches = scan_construction_shape(
            "The committee reviewed the initialization plan.",
            tagger(),
            &candidates,
        );
        assert!(matches.is_empty());
    }

    #[test]
    fn scan_construction_shape_finds_every_match_not_just_the_first() {
        let candidates = BTreeSet::from(["decision", "comparison"]);
        let sentence = "The team made a decision and performed a comparison.";
        let matches = scan_construction_shape(sentence, tagger(), &candidates);
        assert_eq!(matches.len(), 2);
        assert_eq!(matches[0].nominalization, "decision");
        assert_eq!(matches[1].nominalization, "comparison");
    }

    #[test]
    fn scan_construction_shape_is_deterministic() {
        let candidates = BTreeSet::from(["initialization"]);
        let sentence = "The system performs an initialization of the database.";
        assert_eq!(
            scan_construction_shape(sentence, tagger(), &candidates),
            scan_construction_shape(sentence, tagger(), &candidates)
        );
    }
}
