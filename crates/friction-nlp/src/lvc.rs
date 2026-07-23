//! Light-verb-construction (LVC) tables and matching.
//!
//! Shared between detection (`friction-harness::pivot`'s runtime gate,
//! which licenses only the fixed 30-entry [`DERIVATIONAL_LEXICON`]) and
//! offline mining (a corpus tool's LVC-rate measurement and new-candidate
//! discovery, which needs the same surface tables but must be able to
//! probe an arbitrary candidate nominalization, not just the licensed
//! set). This module owns the tables and the two matching functions;
//! deciding which pairs are *licensed* for a runtime rewrite stays a pack
//! concern, not this module's.
//!
//! Everything here except [`scan_construction_shape`] is a straight
//! relocation, unchanged in content, of what used to be private tables in
//! `friction-harness::pivot`: [`LightVerbForm`], [`LIGHT_VERBS`] (formerly
//! `LV`), [`BE_FORMS`] (formerly `BE`), [`DERIVATIONAL_LEXICON`] (formerly
//! `DERIV`), and [`conjugate`] (formerly a private `inflect`, renamed here
//! to avoid colliding with [`crate::inflect`] — a different, already
//! public, surface-shape-based inflector this crate ships; the two serve
//! different callers and must not share a name).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::LazyLock;

use crate::{TaggedToken, Tagger};

/// Which inflectional form a light-verb surface token represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LightVerbForm {
    /// Bare infinitive/present plural form (`perform`, `conduct`, `make`, `do`).
    Base,
    /// Third-person singular present (`performs`, `conducts`, `makes`, `does`).
    ThirdSg,
    /// Past tense (`performed`, `conducted`, `made`, `did`).
    Past,
    /// Gerund/present participle (`performing`, `conducting`, `making`, `doing`).
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

/// The licensed (nominalization -> derived verb) candidate table: 31 entries.
///
/// This is a *candidate* table, not an auto-licensed one — a runtime
/// rewrite gate still has to consult a pack's own licensing decision
/// (today, `friction-harness::pivot` licenses every entry here
/// unconditionally; that is a pack-format simplification this milestone
/// does not change). Entries here are seeded from NOMLEX/CatVar-style
/// noun -> verb derivational mappings, chosen so the derived verb is
/// unambiguous and the nominalization is not a common artifact noun
/// (deliberately absent: `"approval"`, subject-genitive-prone; `"index"`,
/// an artifact noun rather than an LVC nominalization).
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
    /// The full matched surface text, e.g. `"performs an initialization of"`.
    pub matched_text: String,
}

/// Scans `sentence` for every `LV (DET)? candidate (of)?` construction shape.
///
/// `candidate` is looked up in `candidate_noms` rather than
/// [`DERIVATIONAL_LEXICON`], and without the passive/`JJ`/plural gates
/// `friction-harness::pivot`'s runtime `match_pivot` applies before
/// licensing — this is the same left-to-right walk (LV-form -> optional
/// determiner -> candidate token -> optional literal `"of"`), stripped
/// down for offline measurement rather than a runtime accept/reject
/// decision, so it can probe candidate nominalizations that are not (yet)
/// licensed at all.
///
/// Unlike a runtime matcher, this does not stop at the first match: it
/// walks every light-verb-table token in the sentence, left to right, and
/// reports every position whose following (optional-determiner) token is
/// a member of `candidate_noms`, so a mining pass can count occurrences
/// across a whole corpus rather than produce a single per-sentence
/// verdict.
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
    use crate::NlpruleTagger;
    use std::sync::OnceLock;

    fn tagger() -> &'static NlpruleTagger {
        static TAGGER: OnceLock<NlpruleTagger> = OnceLock::new();
        TAGGER.get_or_init(|| NlpruleTagger::new().expect("embedded model must load"))
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
        // "setup" is not a `DERIVATIONAL_LEXICON` key at all, but
        // `scan_construction_shape` takes an arbitrary candidate set, so
        // it must still find this one where `match_pivot` would not.
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
