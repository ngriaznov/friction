//! Static traceability from a `reject` fixture id to the gate it exercises,
//! and (for the three pivot-only fixtures) to the exact
//! [`crate::pivot::PivotRejection`] expected.

use crate::pivot::PivotRejection;

/// The gate a `reject` fixture is documented to exercise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gate {
    /// The (light verb, nominalization) pair is not individually licensed.
    PairLicensing,
    /// The nominalization is modified (an adjective or quantifier) or
    /// plural.
    ModifiedOrPluralNominal,
    /// The governing verb is not in the light-verb table at all.
    NonLightVerb,
    /// The construction is passive, not active, voice.
    ActiveVoiceOnly,
    /// The edit only ever fires inside prose blocks, never headings.
    ProseBlocksOnly,
    /// No token may be inserted, including from a guard-token category
    /// (modals, connectives) that looks superficially safe.
    NoInsertionGuardToken,
    /// No content word may be inserted, full stop.
    ClosureContentWord,
    /// No token may be inserted even from an inert (function-word)
    /// vocabulary; deletion fires only when the seam attests.
    ClosureInertVocabulary,
    /// Open, search-chosen synthesis is banned entirely.
    OpenSynthesisBanned,
}

/// Maps a reject-fixture id to its [`Gate`] by id, never by parsing the
/// fixture's free-text `gate` field.
///
/// # Panics
/// Panics on an id this table hasn't been extended for — the panic is the
/// coverage signal for a new reject fixture landing in `fixtures.json`.
#[must_use]
pub fn gate_for_fixture_id(id: &str) -> Gate {
    match id {
        "pivot_trap_subject_genitive" | "pivot_trap_artifact_noun" => Gate::PairLicensing,
        "pivot_trap_modified_nominal" | "pivot_trap_quantified" => Gate::ModifiedOrPluralNominal,
        "pivot_trap_non_light_verb" => Gate::NonLightVerb,
        "pivot_trap_passive" => Gate::ActiveVoiceOnly,
        "heading_pivot" => Gate::ProseBlocksOnly,
        "bridge_modal_insertion" | "bridge_connective_insertion" => Gate::NoInsertionGuardToken,
        "bridge_content_insertion" => Gate::ClosureContentWord,
        "bridge_inert_insertion" => Gate::ClosureInertVocabulary,
        "word_salad_synthesis" => Gate::OpenSynthesisBanned,
        other => panic!("unhandled fixture id: {other}"),
    }
}

/// Expected [`PivotRejection`] for the three reject fixtures with no
/// `bad_output`.
///
/// `pivot_trap_modified_nominal`, `pivot_trap_quantified`, and
/// `pivot_trap_passive` are asserted by checking
/// [`crate::pivot::match_pivot`] directly against this expected rejection,
/// rather than through closure/score.
///
/// `pivot_trap_quantified` maps to [`PivotRejection::ModifiedNominal`], not
/// a separate "plural" variant: verified empirically that
/// [`friction_nlp::PerceptronTagger`] tags "several" as `JJ`, so it is
/// rejected by the modified-nominal guard before the plural-suffix guard
/// is ever reached.
#[must_use]
pub fn expected_pivot_rejection(id: &str) -> Option<PivotRejection> {
    match id {
        "pivot_trap_modified_nominal" | "pivot_trap_quantified" => {
            Some(PivotRejection::ModifiedNominal)
        }
        "pivot_trap_passive" => Some(PivotRejection::Passive),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_for_fixture_id_covers_every_documented_id() {
        let ids = [
            "pivot_trap_subject_genitive",
            "pivot_trap_modified_nominal",
            "pivot_trap_quantified",
            "pivot_trap_artifact_noun",
            "pivot_trap_non_light_verb",
            "pivot_trap_passive",
            "heading_pivot",
            "bridge_modal_insertion",
            "bridge_connective_insertion",
            "bridge_content_insertion",
            "bridge_inert_insertion",
            "word_salad_synthesis",
        ];
        for id in ids {
            let _ = gate_for_fixture_id(id);
        }
    }

    #[test]
    #[should_panic(expected = "unhandled fixture id")]
    fn gate_for_fixture_id_panics_on_unknown_id() {
        let _ = gate_for_fixture_id("not_a_real_fixture");
    }

    #[test]
    fn expected_pivot_rejection_covers_the_three_gate_only_fixtures() {
        assert_eq!(
            expected_pivot_rejection("pivot_trap_modified_nominal"),
            Some(PivotRejection::ModifiedNominal)
        );
        assert_eq!(
            expected_pivot_rejection("pivot_trap_quantified"),
            Some(PivotRejection::ModifiedNominal)
        );
        assert_eq!(
            expected_pivot_rejection("pivot_trap_passive"),
            Some(PivotRejection::Passive)
        );
        assert_eq!(expected_pivot_rejection("bridge_modal_insertion"), None);
    }
}
