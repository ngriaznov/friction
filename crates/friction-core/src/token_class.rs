//! Canonical word/punctuation character-class fragments shared by every
//! word-tokenizer regex in the workspace.
//!
//! Three crates each compile their own `regex::Regex` over prose text
//! (`friction-core` stays dependency-free, so it cannot own a `Regex`
//! itself — see the crate-level docs' "no parsing" note): `friction-match`
//! (`token::WORD_OR_PUNCTUATION`, the fix-time detection tokenizer, over
//! raw mixed-case source), `friction-harness` (`clean::TOKEN`, the
//! corpus-tool mining and runtime-artifact tokenizer, over pre-lowercased
//! text), and `friction-packs` (`validate::WORD_TOKEN`, pack-build-time
//! word auditing, also over pre-lowercased text). Before this module
//! existed, each site hand-wrote its own `[A-Za-z...]`/`[a-z...]` literal;
//! the three copies were required to agree byte-for-byte on what counts as
//! a "word" but had no mechanism enforcing that. This module is that
//! mechanism: every site now composes its regex from these constants, so
//! a future edit to word-token shape happens once and every consumer
//! moves in lockstep.
//!
//! # ASCII vs. Unicode
//!
//! Two families of fragments are exposed, and only the `_ASCII` family is
//! shipped. Switching all three consumers to the `_UNICODE` family
//! (`\p{L}`/`\p{Ll}` in place of `A-Za-z`/`a-z`) was tried first and
//! measured against the full corpus byte-fence (`friction fix` and
//! `friction check --genre docs --format json` over every corpus file,
//! base binary vs. Unicode-wired binary). The hypothesis that English
//! prose is ASCII-letter-only turned out false: the shipped human corpus
//! contains real, tokenization-visible non-ASCII letters — accented
//! proper nouns and loanwords such as "Nicolò", "Jùnliàng", and "Condé"
//! in `corpus/human/blog` and elsewhere. Under `\p{L}`, an accented name
//! tokenizes as one word run instead of splitting at the accent, which
//! shifts word/token counts and therefore every metric derived from them
//! — 51 corpus files (and the `bench/big10mb.md` synthetic file) produced
//! a different `friction check --format json` byte stream, though `friction
//! fix` stayed byte-identical on all 1007 fence files (Unicode-letter
//! matching never changed which spans a fix-time rule fires on in this
//! corpus, only the metrics report). So `_ASCII` is the exact legacy
//! behavior every shipped consumer composes, preserving byte-for-byte
//! output; `_UNICODE` is staged for Phase 1 of
//! `docs/research/LANGUAGES.md` — a genuinely non-English corpus, where
//! `\p{L}` is required rather than merely permissible. It also tokenizes
//! German umlauts, French diacritics, and Cyrillic — see this module's
//! test suite for worked examples in each language. Consult each
//! consumer's own module docs for which family it currently
//! composes from and why.

/// The "word with apostrophes" class: ASCII letters, upper and lower case.
///
/// Plus the curly single-quote pair (`'`, `'`) and a straight apostrophe
/// (`'`), for matching against raw, mixed-case, not-yet-normalized source
/// text.
///
/// Consumer: `friction_match::token::WORD_OR_PUNCTUATION` (deliberately
/// not `(?i)`-cased: both letter cases are named explicitly so no
/// case-folding machinery engages, matching `LOWERCASE_WORD_APOSTROPHE_ASCII`
/// acting on already-lowercased text exactly rather than merely
/// approximating it).
pub const WORD_APOSTROPHE_ASCII: &str = "A-Za-z\u{2018}\u{2019}'";

/// The "lowercase word" class: ASCII lowercase letters plus a straight
/// apostrophe (`'`).
///
/// For matching against text that has already been case-folded and had
/// curly single quotes normalized to `'`.
///
/// Consumers: `friction_harness::clean::TOKEN` (corpus-tool mining and
/// runtime artifact builds) and `friction_packs::validate::WORD_TOKEN`
/// (pack validation) — both operate on text `friction_harness::clean::clean`
/// has already lowercased and quote-normalized, so this class carries no
/// curly-quote or uppercase members.
pub const LOWERCASE_WORD_APOSTROPHE_ASCII: &str = "a-z'";

/// The prose punctuation class: `.,;:!?`.
///
/// Tokenized as single-character tokens alongside word runs.
///
/// Consumers: `friction_match::token::WORD_OR_PUNCTUATION` and
/// `friction_harness::clean::TOKEN`. `friction_packs::validate::WORD_TOKEN`
/// does not compose this fragment — pack validation only ever needs word
/// tokens, never punctuation tokens.
pub const PROSE_PUNCTUATION: &str = ".,;:!?";

/// The Unicode-ready alternate to [`WORD_APOSTROPHE_ASCII`].
///
/// Unicode letters (`\p{L}`, any case, any script) plus the curly
/// single-quote pair and a straight apostrophe, for matching against raw,
/// mixed-case, not-yet-normalized source text in a language whose
/// alphabet is not a subset of ASCII.
///
/// Not currently composed by any shipped consumer: measured against the
/// full corpus byte-fence and reverted (see this module's docs) because
/// the shipped English corpus contains tokenization-visible non-ASCII
/// letters (accented proper nouns/loanwords), so adopting `\p{L}` for `en`
/// would move `check`'s metrics on real corpus files. Staged for the
/// first `check`-only non-English language per
/// `docs/research/LANGUAGES.md`'s Phase 1. See this crate's test suite
/// for German/French/Russian worked examples.
pub const WORD_APOSTROPHE_UNICODE: &str = "\\p{L}\u{2018}\u{2019}'";

/// The Unicode-ready alternate to [`LOWERCASE_WORD_APOSTROPHE_ASCII`].
///
/// Unicode lowercase letters (`\p{Ll}`) plus a straight apostrophe, for
/// matching against text that has already been case-folded (Rust's
/// `str::to_lowercase`, which is Unicode-aware, produces `\p{Ll}`
/// characters for every script that has a lowercase form) and had curly
/// single quotes normalized to `'`.
///
/// Not currently composed by any shipped consumer, for the same reason as
/// [`WORD_APOSTROPHE_UNICODE`]. See this crate's test suite for
/// German/French/Russian worked examples.
pub const LOWERCASE_WORD_APOSTROPHE_UNICODE: &str = "\\p{Ll}'";

#[cfg(test)]
mod tests {
    //! These tests exercise the fragments the way a consumer would: built
    //! into a `[...]+` character-class regex, same as every real call
    //! site. `friction-core` takes `regex` only as a dev-dependency here —
    //! the shipped crate stays dependency-free (see the module docs).

    use regex::Regex;

    use super::{
        LOWERCASE_WORD_APOSTROPHE_ASCII, LOWERCASE_WORD_APOSTROPHE_UNICODE, PROSE_PUNCTUATION,
        WORD_APOSTROPHE_ASCII, WORD_APOSTROPHE_UNICODE,
    };

    fn word_tokens(class: &str, text: &str) -> Vec<String> {
        let pattern = format!("[{class}]+");
        let re = Regex::new(&pattern).expect("class fragment must compile as a regex");
        re.find_iter(text).map(|m| m.as_str().to_owned()).collect()
    }

    #[test]
    fn ascii_word_class_matches_english() {
        assert_eq!(
            word_tokens(WORD_APOSTROPHE_ASCII, "Don't stop"),
            vec!["Don't", "stop"]
        );
    }

    #[test]
    fn ascii_lowercase_word_class_matches_lowered_english() {
        assert_eq!(
            word_tokens(LOWERCASE_WORD_APOSTROPHE_ASCII, "don't stop"),
            vec!["don't", "stop"]
        );
    }

    #[test]
    fn ascii_word_class_agrees_with_unicode_class_on_ascii_text() {
        // On input with no non-ASCII letters, the two classes must find
        // identical token runs — the shipped English corpus does contain
        // tokenization-visible non-ASCII letters elsewhere (accented
        // proper nouns), which is exactly why they are NOT interchangeable
        // in general; see this module's docs for the measured evidence.
        let text = "Don't stop believing, friend! It's fine.";
        assert_eq!(
            word_tokens(WORD_APOSTROPHE_ASCII, text),
            word_tokens(WORD_APOSTROPHE_UNICODE, text)
        );
    }

    #[test]
    fn ascii_lowercase_word_class_agrees_with_unicode_class_on_ascii_text() {
        let text = "don't stop believing, friend! it's fine.";
        assert_eq!(
            word_tokens(LOWERCASE_WORD_APOSTROPHE_ASCII, text),
            word_tokens(LOWERCASE_WORD_APOSTROPHE_UNICODE, text)
        );
    }

    #[test]
    fn unicode_word_class_tokenizes_german_umlauts() {
        // "Größe" and "Straße" carry ö and ß — neither is in [A-Za-z].
        assert_eq!(
            word_tokens(WORD_APOSTROPHE_UNICODE, "Die Größe der Straße überrascht."),
            vec!["Die", "Größe", "der", "Straße", "überrascht"]
        );
    }

    #[test]
    fn unicode_word_class_tokenizes_french_diacritics_and_apostrophe() {
        // é/è/ç are outside [A-Za-z]; l'été must stay one token, matching
        // the same interior-apostrophe convention English "don't" uses.
        assert_eq!(
            word_tokens(WORD_APOSTROPHE_UNICODE, "L'été a été très agréable à Noël."),
            vec!["L'été", "a", "été", "très", "agréable", "à", "Noël"]
        );
    }

    #[test]
    fn unicode_word_class_tokenizes_cyrillic() {
        // Cyrillic letters are entirely outside [A-Za-z]; the ASCII class
        // produces zero word tokens on this sentence.
        assert!(word_tokens(WORD_APOSTROPHE_ASCII, "Привет, как дела?").is_empty());
        assert_eq!(
            word_tokens(WORD_APOSTROPHE_UNICODE, "Привет, как дела?"),
            vec!["Привет", "как", "дела"]
        );
    }

    #[test]
    fn lowercase_unicode_word_class_tokenizes_lowered_cyrillic() {
        let lowered = "привет, как дела?".to_owned();
        assert_eq!(
            word_tokens(LOWERCASE_WORD_APOSTROPHE_UNICODE, &lowered),
            vec!["привет", "как", "дела"]
        );
    }

    #[test]
    fn prose_punctuation_is_the_six_prose_marks() {
        assert_eq!(PROSE_PUNCTUATION, ".,;:!?");
    }
}
