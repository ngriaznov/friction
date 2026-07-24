//! Closure-by-construction check: every word an edit emits must already be
//! traceable to the input, either verbatim or through a real, per-input
//! licensed derivation.
//!
//! This is the harness-side check for the invariant "the operation set can
//! only delete or substitute from the inventory, so no content word can
//! enter the text" — there is no bridge or synthesis path in the closed
//! operation set for this check to have to gate around; it only has to
//! catch a violation of that invariant.

use std::collections::BTreeMap;

use friction_nlp::Tagger;

use crate::clean::{is_punctuation_token, tokenize};
use crate::tellspan;

/// The result of a [`check_closure`] call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClosureVerdict {
    /// Every token in the output is accounted for.
    Holds,
    /// Every token whose output count exceeds its input count, and which
    /// is not licensed by a real tell-span hit found in `input` itself —
    /// this is what catches an inserted word that happens to already
    /// appear elsewhere in the input, e.g. `bridge_inert_insertion`'s
    /// extra `"to"`.
    Violated {
        /// The offending tokens, in sorted order.
        excess_tokens: Vec<String>,
    },
}

/// Closure by construction: every token in `output` must either already
/// appear in `input` at least as many times as it does in `output`, be a
/// `.,;:!?` punctuation token, or be pack-derivable.
///
/// "Pack-derivable" means one of the tokens a real
/// `tellspan::tell_span_hits` scan of `input` actually licenses *for this
/// input* — not the whole static SUBS/DERIV tables in the abstract. A
/// token derivable from some *other* light verb than the one actually
/// present is not pack-derivable here: this is what correctly rejects
/// `pivot_trap_non_light_verb`'s `"integrates"`, a real `DERIV` value but
/// never actually licensed by `"facilitates"` (which is not a light verb
/// at all, so it produces no pivot hit and hence licenses nothing).
#[must_use]
pub fn check_closure(input: &str, output: &str, tagger: &dyn Tagger) -> ClosureVerdict {
    let input_counts = word_token_counts(input);
    let output_counts = word_token_counts(output);

    let mut licensed_counts: BTreeMap<String, usize> = BTreeMap::new();
    for hit in tellspan::tell_span_hits(input, tagger) {
        for token in hit.licensed_tokens {
            *licensed_counts.entry(token).or_insert(0) += 1;
        }
    }

    let mut excess_tokens: Vec<String> = output_counts
        .iter()
        .filter_map(|(token, &needed)| {
            let available = input_counts.get(token).copied().unwrap_or(0)
                + licensed_counts.get(token).copied().unwrap_or(0);
            (needed > available).then(|| token.clone())
        })
        .collect();
    excess_tokens.sort();

    if excess_tokens.is_empty() {
        ClosureVerdict::Holds
    } else {
        ClosureVerdict::Violated { excess_tokens }
    }
}

/// Tokenizes `text` and counts occurrences of every non-punctuation word
/// token, in a `BTreeMap` for deterministic iteration.
fn word_token_counts(text: &str) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for token in tokenize(text) {
        if is_punctuation_token(&token) {
            continue;
        }
        *counts.entry(token).or_insert(0) += 1;
    }
    counts
}

#[cfg(test)]
mod tests {
    use std::sync::OnceLock;

    use friction_nlp::PerceptronTagger;

    use super::*;

    fn tagger() -> &'static PerceptronTagger {
        static TAGGER: OnceLock<PerceptronTagger> = OnceLock::new();
        TAGGER.get_or_init(|| PerceptronTagger::new().expect("embedded model must load"))
    }

    #[test]
    fn holds_for_pure_deletion() {
        let input = "It is important to note that the tool is fast.";
        let output = "The tool is fast.";
        assert_eq!(
            check_closure(input, output, tagger()),
            ClosureVerdict::Holds
        );
    }

    #[test]
    fn holds_for_a_licensed_substitution() {
        let input = "This guide will walk you through the setup.";
        let output = "This guide covers the setup.";
        assert_eq!(
            check_closure(input, output, tagger()),
            ClosureVerdict::Holds
        );
    }

    #[test]
    fn holds_for_a_licensed_pivot() {
        let input = "The system performs an initialization of the database.";
        let output = "The system initializes the database.";
        assert_eq!(
            check_closure(input, output, tagger()),
            ClosureVerdict::Holds
        );
    }

    #[test]
    fn violated_for_an_inserted_word_absent_from_input() {
        let input = "Worse, plugin uninstallers are often unreliable or missing entirely.";
        let output = "Worse, plugin uninstallers are often unreliable or missing documentation.";
        let verdict = check_closure(input, output, tagger());
        assert_eq!(
            verdict,
            ClosureVerdict::Violated {
                excess_tokens: vec!["documentation".to_string()]
            }
        );
    }

    #[test]
    fn violated_for_an_extra_copy_of_a_word_already_in_input() {
        let input = "Initially, the biggest hurdle was simply getting used to it.";
        let output = "Initially, the biggest hurdle was to getting used to it.";
        let verdict = check_closure(input, output, tagger());
        assert_eq!(
            verdict,
            ClosureVerdict::Violated {
                excess_tokens: vec!["to".to_string()]
            }
        );
    }

    #[test]
    fn violated_when_a_derivable_form_is_licensed_by_a_different_light_verb() {
        // "integrates" is a real DERIV value, but this input's light verb
        // is "facilitates" (not in the light-verb table at all), so no
        // pivot hit licenses "integrates" here.
        let input = "The wizard facilitates the integration of the plugin with your DAW.";
        let output = "The wizard integrates the plugin with your DAW.";
        let verdict = check_closure(input, output, tagger());
        assert_eq!(
            verdict,
            ClosureVerdict::Violated {
                excess_tokens: vec!["integrates".to_string()]
            }
        );
    }

    #[test]
    fn closure_check_is_deterministic() {
        let input = "This guide will walk you through the setup.";
        let output = "This guide covers the setup.";
        assert_eq!(
            check_closure(input, output, tagger()),
            check_closure(input, output, tagger())
        );
    }
}
