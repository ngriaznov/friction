//! Derivational pivot (light-verb construction collapse) detection.
//!
//! Transcribed from `ref_pivot.py`'s `DERIV`/`LV` tables and its `pivot()`
//! matching logic. `LIGHT_VERBS`/`BE_FORMS`/`conjugate`/`LightVerbForm`
//! still come from `friction_nlp::lvc` — fixed, closed grammatical
//! classes (the perform/conduct/make/do inflection tables), not
//! curated pack data — but the licensed (nominalization -> derived verb)
//! lookup itself is sourced from `friction_packs::INVENTORY.pack.
//! lvc_lexicon()` instead of `friction_nlp::lvc::DERIVATIONAL_LEXICON`
//! directly, so the pack's own `lvc_pairs` family (checked against that
//! same lexicon at load time — see `friction_packs::InventoryPack::parse`)
//! is the single source of truth for which pairs are licensed at runtime.
//! This module adds a fourth tell family on top of the ritual/
//! substitution/deletion families, because without it three of the five
//! accept fixtures (`pivot_constructed_cases`, `pivot_real_corpus`, and
//! half of `composed_four_operation_paragraph`) cannot satisfy "output
//! contains strictly fewer tell spans than input" — those three families
//! never match a light-verb-construction sentence at all, so raw
//! tell-span count is `0` on both sides of every pivot example. This is
//! the same table [`crate::closure`] needs for its "pack-derivable"
//! allowance, so it earns its place twice over.
//!
//! This milestone only detects and reports licensed pivots; it never
//! rewrites text (that is a later milestone's job — see
//! [`crate::pending`]). Per-(light-verb, nominalization) licensing (as
//! opposed to "any of the four light verbs plus a licensed
//! nominalization") stays out of scope this milestone; this module's
//! actual gating logic is unchanged — only its licensed-pair data source
//! moved to the pack.

use friction_nlp::lvc::{BE_FORMS, LIGHT_VERBS, LightVerbForm, conjugate};
use friction_nlp::{TaggedToken, Tagger};

/// The outcome of scanning one sentence for a derivational-pivot
/// construction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PivotOutcome {
    /// No token in the sentence is a member of the light-verb table.
    NoLightVerb,
    /// A light verb was found but a guard rejected it before licensing was
    /// even checked, in the same order `ref_pivot.py` checks them.
    Rejected(PivotRejection),
    /// A light verb plus determiner-optional-nominalization was found, but
    /// the nominalization is not a key in
    /// `friction_packs::INVENTORY.pack.lvc_lexicon()` — not a licensed
    /// pair.
    Unlicensed,
    /// A licensed light-verb construction, ready to collapse.
    Licensed(LicensedPivot),
}

/// Why a candidate light-verb construction was rejected before licensing
/// was checked.
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

/// A licensed derivational-pivot match: the light-verb-construction span
/// found, and the single derived verb it collapses to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LicensedPivot {
    /// The matched surface text, e.g. `"performs an initialization of"`.
    pub matched_text: String,
    /// The derived verb, inflected to match the light verb's own form, e.g.
    /// `"initializes"`.
    pub derived_verb: String,
}

/// Faithful transcription of `ref_pivot.py::pivot()`'s *matching* half only
/// (this milestone never rewrites text).
///
/// Walks `tagger.tag(sentence, 0)` directly — surface text via each
/// `TaggedToken`'s own byte range, lowercased for every table lookup —
/// rather than a second, separately tokenized word list:
/// `friction_nlp::Tagger` tokenizes internally and has no "tag this
/// pre-split word list" entry point the way NLTK's tagger does, so
/// aligning a second tokenizer's indices against the tagger's own tags
/// would be an added, unnecessary failure mode. The only place a POS tag
/// is consulted at all is the modified-nominal (`JJ`) guard; the LV
/// lookup, the passive-preceding-`BE` check, and the plural-suffix check
/// are pure surface-string comparisons, exactly as in `ref_pivot.py`.
///
/// Scans every light-verb-table token in the sentence, left to right, the
/// same way `ref_pivot.py`'s `pivot()` loop does: `Passive`, `ModifiedNominal`,
/// and `PluralNominal` stop the scan immediately and reject (the reference
/// returns on these), but a candidate that runs out of following tokens or
/// whose nominalization isn't a licensed pair is *not* a final answer for the
/// sentence — the reference's loop `continue`s past it to the next
/// light-verb-table token, and so does this one. Only once every light-verb
/// token in the sentence has been tried without a reject or a license does
/// this return `Unlicensed` (or `NoLightVerb` if the sentence never
/// contained a light-verb-table token at all).
#[must_use]
pub fn match_pivot(sentence: &str, tagger: &dyn Tagger) -> PivotOutcome {
    let tokens = tagger.tag(sentence, 0);
    let surface = |t: &TaggedToken| -> &str { &sentence[t.token.range.clone()] };

    let mut saw_light_verb = false;

    for i in 0..tokens.len() {
        let lv_lower = surface(&tokens[i]).to_lowercase();
        let Some(&feat) = LIGHT_VERBS.get(lv_lower.as_str()) else {
            continue;
        };
        saw_light_verb = true;

        if feat == LightVerbForm::Past
            && i > 0
            && BE_FORMS.contains(surface(&tokens[i - 1]).to_lowercase().as_str())
        {
            return PivotOutcome::Rejected(PivotRejection::Passive);
        }

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
        if tokens[j].pos.as_str() == "JJ" {
            return PivotOutcome::Rejected(PivotRejection::ModifiedNominal);
        }

        let lvc_lexicon = friction_packs::INVENTORY.pack.lvc_lexicon();
        let nom = surface(&tokens[j]).to_lowercase();
        if let Some(stem) = nom.strip_suffix('s')
            && lvc_lexicon.contains_key(stem)
        {
            return PivotOutcome::Rejected(PivotRejection::PluralNominal);
        }
        let Some(base_verb) = lvc_lexicon.get(nom.as_str()) else {
            continue;
        };

        let derived_verb = conjugate(base_verb, feat);

        let k = j + 1;
        let has_of = k < tokens.len() && surface(&tokens[k]).to_lowercase() == "of";
        let end_idx = if has_of { k } else { j };

        let matched_text = tokens[i..=end_idx]
            .iter()
            .map(surface)
            .collect::<Vec<_>>()
            .join(" ");

        return PivotOutcome::Licensed(LicensedPivot {
            matched_text,
            derived_verb,
        });
    }

    if saw_light_verb {
        PivotOutcome::Unlicensed
    } else {
        PivotOutcome::NoLightVerb
    }
}

#[cfg(test)]
mod tests {
    use std::sync::OnceLock;

    use friction_nlp::NlpruleTagger;

    use super::*;

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
    fn match_pivot_licenses_the_five_constructed_accept_cases() {
        let cases = [
            (
                "The system performs an initialization of the database.",
                "initializes",
            ),
            (
                "The tool performs validation of the configuration file before applying changes.",
                "validates",
            ),
            (
                "Before deployment, the script conducts an analysis of the dependency graph.",
                "analyzes",
            ),
            ("You should make a decision soon.", "decide"),
            (
                "The installer performed a comparison of the two versions.",
                "compared",
            ),
        ];
        for (sentence, expected_verb) in cases {
            match match_pivot(sentence, tagger()) {
                PivotOutcome::Licensed(pivot) => {
                    assert_eq!(pivot.derived_verb, expected_verb, "sentence: {sentence}");
                }
                other => panic!("expected Licensed for {sentence:?}, got {other:?}"),
            }
        }
    }

    #[test]
    fn match_pivot_rejects_passive() {
        let sentence = "An initialization of the database is performed by the system.";
        assert_eq!(
            match_pivot(sentence, tagger()),
            PivotOutcome::Rejected(PivotRejection::Passive)
        );
    }

    #[test]
    fn match_pivot_rejects_modified_nominal() {
        let sentence = "The migration performs a full initialization of the database.";
        assert_eq!(
            match_pivot(sentence, tagger()),
            PivotOutcome::Rejected(PivotRejection::ModifiedNominal)
        );
    }

    #[test]
    fn match_pivot_rejects_quantified_as_modified_nominal() {
        let sentence = "The tool performs several initializations of the database during testing.";
        assert_eq!(
            match_pivot(sentence, tagger()),
            PivotOutcome::Rejected(PivotRejection::ModifiedNominal)
        );
    }

    #[test]
    fn match_pivot_finds_no_light_verb_for_non_light_verb_governors() {
        assert_eq!(
            match_pivot(
                "Obtain the approval of the committee before merging.",
                tagger()
            ),
            PivotOutcome::NoLightVerb
        );
        assert_eq!(
            match_pivot("Create an index of the most common queries.", tagger()),
            PivotOutcome::NoLightVerb
        );
        assert_eq!(
            match_pivot(
                "The wizard facilitates the integration of the plugin with your DAW.",
                tagger()
            ),
            PivotOutcome::NoLightVerb
        );
    }

    #[test]
    fn match_pivot_continues_past_an_unlicensed_candidate_to_a_later_licensed_one() {
        // "performs setup" is a candidate light-verb construction whose
        // nominalization ("setup") is not a licensed pair; the scan must
        // not stop there and report `Unlicensed` for the whole sentence —
        // it has to keep going and find the later, genuinely licensed
        // "made a decision".
        let sentence = "The wizard performs setup and the team made a decision to ship.";
        match match_pivot(sentence, tagger()) {
            PivotOutcome::Licensed(pivot) => assert_eq!(pivot.derived_verb, "decided"),
            other => panic!("expected Licensed via the later light verb, got {other:?}"),
        }
    }

    #[test]
    fn match_pivot_rewrites_real_corpus_sentence() {
        let sentence = "However, after careful consideration and experimentation, we made the \
                         decision to switch from Gerrit to GitHub pull requests.";
        match match_pivot(sentence, tagger()) {
            PivotOutcome::Licensed(pivot) => assert_eq!(pivot.derived_verb, "decided"),
            other => panic!("expected Licensed, got {other:?}"),
        }
    }

    #[test]
    fn match_pivot_is_deterministic() {
        let sentence = "The system performs an initialization of the database.";
        assert_eq!(
            match_pivot(sentence, tagger()),
            match_pivot(sentence, tagger())
        );
    }
}
