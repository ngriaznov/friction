//! The lexical rule family.
//!
//! Discourse-filler phrase deletion ([`FillerPhraseRule`]) and
//! inflection-aware near-synonym substitution ([`SubstitutionRule`]) — the
//! two Fix-tier rules that touch individual words and short fixed phrases,
//! as opposed to sentence rhythm, coordination symmetry, or document
//! structure.
//!
//! Both rules share the same shape: a hand-curated, closed literal table
//! (phrases to delete, or lemmas to substitute), a left-to-right,
//! non-overlapping, word-boundary-respecting scan over each sentence's
//! text, and a fix that only ever deletes or swaps in a same-meaning
//! replacement — never reorders or drops propositional content, so both
//! stay Fix tier. Neither rule needs more than one meaning-preserving fix
//! strategy per finding (a filler phrase is simply deleted; a lexical
//! substitution has exactly one designated replacement lemma), so neither
//! rule's `fix` consults its `StrategyRng` argument — see [`crate::Rule::
//! fix`]'s own docs for why that is a legitimate use of the API, not an
//! oversight.
//!
//! # Curation source
//!
//! Both tables were curated by hand from `corpus/MINING.md`'s train-split
//! log-odds mining report (llm-favored 1-/2-/3-grams) plus canonical,
//! widely-documented LLM-register tells not specific to this corpus's
//! topics. Entries mined from `corpus/MINING.md` that read as topic
//! artifacts of this corpus's specific prompt set (domain nouns like
//! `database`, `backup`, `cron`, `postgres`) rather than general
//! llm-vs-human register were left out, the same curation judgment
//! `friction-packs/packs/mined-ngrams-v1.toml`'s own header documents for
//! the metrics layer's mined-phrase pack.

mod filler;
mod substitution;

pub use filler::FillerPhraseRule;
pub use substitution::SubstitutionRule;
