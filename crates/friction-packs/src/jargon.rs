//! The curated metaphor-lexeme pack (`jargon-v1`): compound-head metaphor
//! nouns `friction-match`'s `jargon.metaphor` channel flags, plus a small
//! attested-exceptions allowlist of established compounds that channel
//! never flags regardless of shape.
//!
//! Detection-only (SYNTHESIS.md §4): there is no deterministic true
//! replacement for an invented term, so unlike [`crate::InventoryPack`]
//! this pack carries no repair machinery at all — no pattern, no
//! replacement text, just the lexeme list and its exceptions.
//!
//! This module is the read side only: [`JargonPack::parse`] turns the
//! pack's TOML shape into typed, validated, deterministically-sorted
//! in-memory tables. `crates/friction-packs/packs/jargon-v1.toml` itself
//! is hand-curated — see that file's own header comment.

use std::collections::BTreeSet;

use serde::Deserialize;

use crate::PackError;

/// Where a [`Lexeme`]'s inclusion evidence comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LexemeSource {
    /// Measured directly against this corpus (train split); `notes`
    /// records the per-100k-word llm-vs-human rate.
    Mined,
    /// Documented LLM metaphor vocabulary external to this corpus;
    /// `notes` records the source.
    External,
}

impl LexemeSource {
    fn parse(lexeme: &str, value: &str) -> Result<Self, PackError> {
        match value {
            "mined" => Ok(Self::Mined),
            "external" => Ok(Self::External),
            other => Err(PackError::JargonUnknownSource {
                lexeme: lexeme.to_string(),
                value: other.to_string(),
            }),
        }
    }
}

/// One curated metaphor lexeme: a lowercase singular noun, its plural (if
/// worth listing explicitly), and its provenance.
#[derive(Debug, Clone)]
pub struct Lexeme {
    /// Lowercase singular form.
    pub lexeme: Box<str>,
    /// Lowercase plural form, when listed.
    pub plural: Option<Box<str>>,
    pub source: LexemeSource,
    /// Provenance — mandatory, non-empty for every entry.
    pub notes: Box<str>,
}

impl Lexeme {
    fn try_from_raw(raw: RawLexeme) -> Result<Self, PackError> {
        let source = LexemeSource::parse(&raw.lexeme, &raw.source)?;
        if raw.notes.trim().is_empty() {
            return Err(PackError::JargonMissingNotes { entry: raw.lexeme });
        }
        Ok(Self {
            lexeme: raw.lexeme.into_boxed_str(),
            plural: raw.plural.map(String::into_boxed_str),
            source,
            notes: raw.notes.into_boxed_str(),
        })
    }
}

/// One compound exempted from `jargon.metaphor` regardless of its head or
/// modifiers.
#[derive(Debug, Clone)]
pub struct AttestedException {
    /// Lowercase, space-separated (hyphens already normalized).
    pub compound: Box<str>,
    /// Why this compound is exempt — mandatory, non-empty.
    pub notes: Box<str>,
}

impl AttestedException {
    fn try_from_raw(raw: RawAttestedException) -> Result<Self, PackError> {
        if raw.notes.trim().is_empty() {
            return Err(PackError::JargonMissingNotes {
                entry: raw.compound,
            });
        }
        Ok(Self {
            compound: raw.compound.into_boxed_str(),
            notes: raw.notes.into_boxed_str(),
        })
    }
}

/// A parsed, validated, deterministically-sorted `jargon-v1` pack.
#[derive(Debug, Clone, Default)]
pub struct JargonPack {
    lexemes: Vec<Lexeme>,
    attested_exceptions: Vec<AttestedException>,
    /// Every listed lexeme AND plural surface form, lowercase — the
    /// membership set [`JargonPack::is_head_word`] queries. Precomputed
    /// at parse time, the same "derived lookup lives on the pack"
    /// discipline [`crate::InventoryPack::lvc_lexicon`] uses.
    head_words: BTreeSet<Box<str>>,
    /// Every attested-exception compound, lowercase. Precomputed at parse
    /// time for the same reason as `head_words`.
    exception_compounds: BTreeSet<Box<str>>,
}

impl JargonPack {
    /// Parses `toml`'s `jargon-v1`-shaped contents into a validated pack.
    ///
    /// # Errors
    /// Returns [`PackError::Toml`] if `toml` is not valid TOML in the
    /// expected shape. Returns [`PackError::JargonUnknownSource`] if a
    /// `lexemes` entry's `source` is neither `"mined"` nor `"external"`,
    /// [`PackError::JargonMissingNotes`] if any entry's `notes` is empty
    /// or whitespace-only, or [`PackError::JargonDuplicateLexeme`] /
    /// [`PackError::JargonDuplicateException`] for a repeated key.
    pub fn parse(toml: &str) -> Result<Self, PackError> {
        let raw: RawPack = toml::from_str(toml).map_err(PackError::from)?;

        let mut lexemes: Vec<Lexeme> = raw
            .lexemes
            .into_iter()
            .map(Lexeme::try_from_raw)
            .collect::<Result<_, _>>()?;
        lexemes.sort_by(|a, b| a.lexeme.cmp(&b.lexeme));

        let mut seen_lexemes: BTreeSet<&str> = BTreeSet::new();
        let mut head_words: BTreeSet<Box<str>> = BTreeSet::new();
        for lexeme in &lexemes {
            if !seen_lexemes.insert(lexeme.lexeme.as_ref()) {
                return Err(PackError::JargonDuplicateLexeme {
                    lexeme: lexeme.lexeme.to_string(),
                });
            }
            head_words.insert(lexeme.lexeme.clone());
            if let Some(plural) = &lexeme.plural {
                if !seen_lexemes.insert(plural.as_ref()) {
                    return Err(PackError::JargonDuplicateLexeme {
                        lexeme: plural.to_string(),
                    });
                }
                head_words.insert(plural.clone());
            }
        }

        let mut attested_exceptions: Vec<AttestedException> = raw
            .attested_exceptions
            .into_iter()
            .map(AttestedException::try_from_raw)
            .collect::<Result<_, _>>()?;
        attested_exceptions.sort_by(|a, b| a.compound.cmp(&b.compound));

        let mut seen_exceptions: BTreeSet<&str> = BTreeSet::new();
        let mut exception_compounds: BTreeSet<Box<str>> = BTreeSet::new();
        for exception in &attested_exceptions {
            if !seen_exceptions.insert(exception.compound.as_ref()) {
                return Err(PackError::JargonDuplicateException {
                    compound: exception.compound.to_string(),
                });
            }
            exception_compounds.insert(exception.compound.clone());
        }

        Ok(Self {
            lexemes,
            attested_exceptions,
            head_words,
            exception_compounds,
        })
    }

    #[must_use]
    pub fn lexemes(&self) -> &[Lexeme] {
        &self.lexemes
    }

    #[must_use]
    pub fn attested_exceptions(&self) -> &[AttestedException] {
        &self.attested_exceptions
    }

    /// `true` if `word` (already lowercase) is a listed lexeme or one of
    /// its listed plurals.
    #[must_use]
    pub fn is_head_word(&self, word: &str) -> bool {
        self.head_words.contains(word)
    }

    /// `true` if `compound` (already lowercase, hyphens normalized to
    /// spaces) is an exact attested exception.
    #[must_use]
    pub fn is_attested_exception(&self, compound: &str) -> bool {
        self.exception_compounds.contains(compound)
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawLexeme {
    lexeme: String,
    #[serde(default)]
    plural: Option<String>,
    source: String,
    notes: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawAttestedException {
    compound: String,
    notes: String,
}

/// The TOML shape of the pack as a whole. Deliberately **not**
/// `#[serde(deny_unknown_fields)]` — matches `register.rs`'s own
/// precedent of tolerating a free-form `[pack]` metadata table this
/// reader never round-trips.
#[derive(Debug, Deserialize)]
struct RawPack {
    #[serde(default)]
    lexemes: Vec<RawLexeme>,
    #[serde(default)]
    attested_exceptions: Vec<RawAttestedException>,
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL_PACK: &str = r#"
        [pack]
        version = "jargon-v1"

        [[lexemes]]
        lexeme = "well"
        plural = "wells"
        source = "external"
        notes = "test provenance"

        [[lexemes]]
        lexeme = "soup"
        source = "mined"
        notes = "test provenance"

        [[attested_exceptions]]
        compound = "primordial soup"
        notes = "established idiom"
    "#;

    #[test]
    fn parse_sorts_lexemes_and_exceptions_regardless_of_file_order() {
        let toml = r#"
            [[lexemes]]
            lexeme = "well"
            source = "external"
            notes = "n"

            [[lexemes]]
            lexeme = "cornerstone"
            source = "mined"
            notes = "n"

            [[attested_exceptions]]
            compound = "water well"
            notes = "n"

            [[attested_exceptions]]
            compound = "data fabric"
            notes = "n"
        "#;
        let pack = JargonPack::parse(toml).unwrap();
        let lexemes: Vec<&str> = pack.lexemes().iter().map(|l| &*l.lexeme).collect();
        assert_eq!(lexemes, vec!["cornerstone", "well"]);
        let exceptions: Vec<&str> = pack
            .attested_exceptions()
            .iter()
            .map(|e| &*e.compound)
            .collect();
        assert_eq!(exceptions, vec!["data fabric", "water well"]);
    }

    #[test]
    fn parse_rejects_unknown_source() {
        let toml = r#"
            [[lexemes]]
            lexeme = "well"
            source = "invented"
            notes = "n"
        "#;
        let err = JargonPack::parse(toml).unwrap_err();
        assert!(matches!(err, PackError::JargonUnknownSource { .. }));
    }

    #[test]
    fn parse_rejects_empty_notes_on_a_lexeme() {
        let toml = r#"
            [[lexemes]]
            lexeme = "well"
            source = "external"
            notes = "   "
        "#;
        let err = JargonPack::parse(toml).unwrap_err();
        assert!(matches!(err, PackError::JargonMissingNotes { .. }));
    }

    #[test]
    fn parse_rejects_empty_notes_on_an_exception() {
        let toml = r#"
            [[attested_exceptions]]
            compound = "data fabric"
            notes = ""
        "#;
        let err = JargonPack::parse(toml).unwrap_err();
        assert!(matches!(err, PackError::JargonMissingNotes { .. }));
    }

    #[test]
    fn parse_rejects_duplicate_lexeme() {
        let toml = r#"
            [[lexemes]]
            lexeme = "well"
            source = "external"
            notes = "n"

            [[lexemes]]
            lexeme = "well"
            source = "external"
            notes = "n"
        "#;
        let err = JargonPack::parse(toml).unwrap_err();
        assert!(matches!(err, PackError::JargonDuplicateLexeme { .. }));
    }

    #[test]
    fn parse_rejects_a_plural_that_duplicates_another_lexeme() {
        let toml = r#"
            [[lexemes]]
            lexeme = "well"
            plural = "wells"
            source = "external"
            notes = "n"

            [[lexemes]]
            lexeme = "wells"
            source = "external"
            notes = "n"
        "#;
        let err = JargonPack::parse(toml).unwrap_err();
        assert!(matches!(err, PackError::JargonDuplicateLexeme { .. }));
    }

    #[test]
    fn parse_rejects_duplicate_exception() {
        let toml = r#"
            [[attested_exceptions]]
            compound = "data fabric"
            notes = "n"

            [[attested_exceptions]]
            compound = "data fabric"
            notes = "n"
        "#;
        let err = JargonPack::parse(toml).unwrap_err();
        assert!(matches!(err, PackError::JargonDuplicateException { .. }));
    }

    #[test]
    fn is_head_word_matches_singular_and_plural() {
        let pack = JargonPack::parse(MINIMAL_PACK).unwrap();
        assert!(pack.is_head_word("well"));
        assert!(pack.is_head_word("wells"));
        assert!(pack.is_head_word("soup"));
        assert!(!pack.is_head_word("soups"));
        assert!(!pack.is_head_word("echo"));
    }

    #[test]
    fn is_attested_exception_matches_exact_compound_only() {
        let pack = JargonPack::parse(MINIMAL_PACK).unwrap();
        assert!(pack.is_attested_exception("primordial soup"));
        assert!(!pack.is_attested_exception("primordial soups"));
        assert!(!pack.is_attested_exception("soup"));
    }

    #[test]
    fn empty_pack_parses_with_no_lexemes_or_exceptions() {
        let pack = JargonPack::parse("[pack]\nversion = \"jargon-v1\"\n").unwrap();
        assert!(pack.lexemes().is_empty());
        assert!(pack.attested_exceptions().is_empty());
    }

    #[test]
    fn parse_is_deterministic() {
        let a = JargonPack::parse(MINIMAL_PACK).unwrap();
        let b = JargonPack::parse(MINIMAL_PACK).unwrap();
        let ids = |p: &JargonPack| -> Vec<String> {
            p.lexemes().iter().map(|l| l.lexeme.to_string()).collect()
        };
        assert_eq!(ids(&a), ids(&b));
    }
}
