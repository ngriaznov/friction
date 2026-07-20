//! The rhythm rule family: sentence-length shape and burstiness.
//!
//! Both rules here react to the same signal — [`friction_core::MetricVector::
//! sentence_length_cv`] sitting *below* the genre's human envelope, i.e. a
//! document whose sentences read as suspiciously uniform in length, a
//! well-documented LLM tell (human writers naturally mix short, punchy
//! sentences with longer, more complex ones; a model trained to produce
//! "readable" prose tends to converge on a narrow medium-length band
//! instead) — but they respond to it from opposite ends and at different
//! tiers:
//!
//! - [`SentenceSplitRule`] (Fix tier): carves a genuinely over-long
//!   sentence into a shorter piece and a remainder at its strongest clause
//!   boundary (a semicolon, or a coordinating `", and"`/`", but"`/`", so"`
//!   with a real clause after it). Splitting only ever changes punctuation
//!   and case — never reorders or drops a proposition — so it is safe to
//!   apply automatically.
//! - [`SentenceFuseRule`] (Suggest tier only): flags two adjacent short
//!   sentences that share a trivially-coreferring subject as a candidate
//!   for fusing into one. Fusing decides how to *combine* two clauses
//!   (conjunction choice, subordination, ...), which is a genuine
//!   rewrite judgment call this rule does not make for the user — so it
//!   only ever surfaces a [`friction_core::Finding`] at
//!   [`friction_core::Tier::Suggest`] and proposes no patch at all. See
//!   [`SentenceFuseRule`]'s own docs for how a `Suggest`-only rule fits the
//!   engine's [`crate::Gate`]/driver contract.
//!
//! # Shared token definition
//!
//! Both submodules measure a sentence's length in the same unit:
//! whitespace-delimited tokens ([`token_count`]), deliberately matching
//! `friction-metrics::rhythm`'s own token definition (see that module's
//! docs) rather than the part-of-speech tagger's tokenization — the
//! [`friction_core::MetricVector::sentence_length_cv`] value both rules
//! gate on is computed with that same whitespace definition, so a
//! per-genre threshold expressed in "tokens" means the same thing on both
//! sides of the gate.

mod fuse;
mod split;

pub use fuse::SentenceFuseRule;
pub use split::SentenceSplitRule;

/// Counts the tokens in `text`: maximal runs of non-whitespace characters.
/// See the module docs' "Shared token definition" section for why this
/// (not a tagger's tokenization) is the definition both submodules use.
fn token_count(text: &str) -> usize {
    text.split_whitespace().count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_count_counts_whitespace_delimited_runs() {
        assert_eq!(token_count(""), 0);
        assert_eq!(token_count("one"), 1);
        assert_eq!(token_count("one two  three\tfour\nfive"), 5);
    }
}
