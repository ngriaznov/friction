//! [`Lexicon`]: a parsed English lexicon pack (`lexicon-en-v1`) — the
//! irregular-inflection tables, closed word lists, and small phrase sets
//! this crate's tagging, chunking, and transduction modules consult.
//!
//! The pack is authored by hand as `data/lexicon-en.toml`, embedded at
//! build time via `include_str!` (same mechanism as the SRX segmentation
//! ruleset in `src/segment_srx.rs`), and parsed once into [`LEXICON_EN`].

use std::collections::BTreeMap;
use std::sync::LazyLock;

use serde::Deserialize;

/// The embedded, current lexicon pack (`lexicon-en-v1`), vendored into
/// this crate the same way `data/friction-en.srx` is.
const LEXICON_EN_TOML: &str = include_str!("../data/lexicon-en.toml");

/// The parsed, embedded `lexicon-en-v1` pack, parsed once and reused for
/// the life of the process.
///
/// # Panics
/// Panics if the embedded `lexicon-en.toml` fails to parse — a bug in
/// this crate's own vendored data (covered by this module's tests), not a
/// condition any caller can recover from by retrying.
pub static LEXICON_EN: LazyLock<Lexicon> = LazyLock::new(|| {
    Lexicon::parse(LEXICON_EN_TOML)
        .expect("embedded lexicon-en.toml must parse: see this crate's lexicon tests")
});

/// A sorted, deduplicated, immutable set of lowercase words or phrases.
#[derive(Debug, Clone, Default)]
pub struct WordSet(Box<[Box<str>]>);

impl WordSet {
    /// Builds a [`WordSet`] from an unsorted, possibly-duplicated list.
    fn new(words: Vec<String>) -> Self {
        let mut words: Vec<Box<str>> = words.into_iter().map(String::into_boxed_str).collect();
        words.sort_unstable();
        words.dedup();
        Self(words.into_boxed_slice())
    }

    /// Whether `word` is a member of this set.
    #[must_use]
    pub fn contains(&self, word: &str) -> bool {
        self.0
            .binary_search_by(|entry| entry.as_ref().cmp(word))
            .is_ok()
    }

    /// Every member, in sorted order.
    pub fn iter(&self) -> impl Iterator<Item = &str> {
        self.0.iter().map(Box::as_ref)
    }

    /// The number of members.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether this set has no members.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// A verb's inflected surface forms, keyed by its base (lemma) form in
/// [`Lexicon::verbs`].
#[derive(Debug, Clone)]
pub struct VerbForms {
    /// Third-person singular present (`writes`).
    pub sg3: Box<str>,
    /// Present participle / gerund (`writing`).
    pub ing: Box<str>,
    /// Simple past (`wrote`).
    pub past: Box<str>,
    /// Past participle (`written`); equal to `past` when the source
    /// table gave only three forms.
    pub participle: Box<str>,
}

/// A verb's past and past-participle forms, keyed by its base form in
/// [`Lexicon::past_only`].
#[derive(Debug, Clone)]
pub struct PastForms {
    /// Simple past (`hit`).
    pub past: Box<str>,
    /// Past participle (`hit`).
    pub participle: Box<str>,
}

/// A parsed `lexicon-en-v1` pack: every irregular-inflection table, closed
/// word list, and phrase set it defines.
#[derive(Debug, Clone)]
pub struct Lexicon {
    /// Irregular verbs' full inflected forms, keyed by base form. This
    /// key set is the sole basis for surface classification (see
    /// [`Lexicon::past_only`]).
    pub verbs: BTreeMap<Box<str>, VerbForms>,
    /// Irregular verbs consulted for past/participle generation only,
    /// keyed by base form. Deliberately disjoint from
    /// [`Lexicon::verbs`]'s key set — see `data/lexicon-en.toml`.
    pub past_only: BTreeMap<Box<str>, PastForms>,
    /// Irregular nouns' plural forms, keyed by singular form.
    pub nouns: BTreeMap<Box<str>, Box<str>>,
    /// Multi-syllable words whose final consonant must not double under
    /// `-ing`/`-ed` suffixation.
    pub no_double: WordSet,
    /// Subordinating conjunctions.
    pub subordinators: WordSet,
    /// Nominalizations mapped to their gerund verb form (`optimization`
    /// -> `optimizing`).
    pub nominal_verbs: BTreeMap<Box<str>, Box<str>>,
    /// Subjects generic enough to demote under passivization without
    /// losing information.
    pub generic_subjects: WordSet,
    /// Reflexive pronouns.
    pub reflexive_pronouns: WordSet,
    /// Object pronouns.
    pub object_pronouns: WordSet,
    /// Subject-initial contractions (`it's`, `they're`).
    pub subject_contractions: WordSet,
    /// Negated-auxiliary contractions (`isn't`, `won't`).
    pub negated_aux_contractions: WordSet,
    /// Stative verb lemmas.
    pub stative_verbs: WordSet,
}

/// The `[pack]` table.
#[derive(Debug, Deserialize)]
struct RawPackMeta {
    version: String,
}

/// The `[no_double]` / `[subordinators]` table shape: a single `words`
/// array.
#[derive(Debug, Deserialize)]
struct RawWords {
    words: Vec<String>,
}

/// The `[generic_subjects]` table shape: a single `phrases` array.
#[derive(Debug, Deserialize)]
struct RawPhrases {
    phrases: Vec<String>,
}

/// The `[pronouns]` table shape.
#[derive(Debug, Deserialize)]
struct RawPronouns {
    reflexive: Vec<String>,
    object: Vec<String>,
}

/// The `[contractions]` table shape.
#[derive(Debug, Deserialize)]
struct RawContractions {
    subject: Vec<String>,
    negated_aux: Vec<String>,
}

/// The `[stative_verbs]` table shape: a single `lemmas` array.
#[derive(Debug, Deserialize)]
struct RawStativeVerbs {
    lemmas: Vec<String>,
}

/// The full TOML shape of a `lexicon-en-v1` pack, before validation.
#[derive(Debug, Deserialize)]
struct RawPack {
    pack: RawPackMeta,
    irregular_verbs: BTreeMap<String, Vec<String>>,
    irregular_past_only: BTreeMap<String, Vec<String>>,
    irregular_nouns: BTreeMap<String, String>,
    no_double: RawWords,
    subordinators: RawWords,
    nominal_verbs: BTreeMap<String, String>,
    generic_subjects: RawPhrases,
    pronouns: RawPronouns,
    contractions: RawContractions,
    stative_verbs: RawStativeVerbs,
}

/// The pack version this build of `friction-nlp` knows how to read.
const SUPPORTED_VERSION: &str = "lexicon-en-v1";

impl Lexicon {
    /// Selects the embedded lexicon pack for `lang` — `Lang::En` ->
    /// [`LEXICON_EN`].
    ///
    /// This is the seam a language-aware caller should reach for. It is
    /// deliberately not yet wired into this crate's own English-only
    /// morphology code (`inflect.rs`, `chunk.rs`), which still reaches
    /// for [`LEXICON_EN`] directly: those modules are generative-English
    /// grammar in code (suffix orthography, finiteness tests) that a
    /// second language cannot simply supply a new lexicon file for — a
    /// later phase moves them behind a per-language profile trait
    /// (`docs/research/LANGUAGES.md`, "Layer C"), and rewriting their
    /// call sites to go through `for_lang` now would only add a layer of
    /// indirection over a `lang` that, for them, can only ever be `En`
    /// until that trait exists.
    ///
    /// The match is exhaustive over [`friction_core::Lang`]: adding a
    /// language variant without a lexicon pack for it is a compile error
    /// here, not a runtime surprise.
    #[must_use]
    pub fn for_lang(lang: friction_core::Lang) -> &'static Self {
        match lang {
            friction_core::Lang::En => &LEXICON_EN,
        }
    }

    /// Parses a pack from its TOML text.
    ///
    /// # Errors
    /// Returns [`LexiconError::Toml`] if `toml` is not valid TOML in the
    /// expected shape, and any other [`LexiconError`] variant if it parses
    /// but fails a validation rule (unsupported `[pack].version`, a
    /// malformed verb-form array, an empty string or uppercase key
    /// anywhere, a `verbs`/`past_only` key overlap, a `nominal_verbs`
    /// value not ending in `"ing"`, or an empty section).
    pub fn parse(toml: &str) -> Result<Self, LexiconError> {
        let raw: RawPack = toml::from_str(toml).map_err(LexiconError::from)?;

        if raw.pack.version != SUPPORTED_VERSION {
            return Err(LexiconError::Version {
                found: raw.pack.version,
            });
        }

        let mut verbs = BTreeMap::new();
        for (base, forms) in raw.irregular_verbs {
            check_key("irregular_verbs", &base)?;
            let [sg3, ing, past, participle] = verb_forms("irregular_verbs", &base, forms)?;
            check_value("irregular_verbs", &sg3)?;
            check_value("irregular_verbs", &ing)?;
            check_value("irregular_verbs", &past)?;
            check_value("irregular_verbs", &participle)?;
            verbs.insert(
                base.into_boxed_str(),
                VerbForms {
                    sg3: sg3.into_boxed_str(),
                    ing: ing.into_boxed_str(),
                    past: past.into_boxed_str(),
                    participle: participle.into_boxed_str(),
                },
            );
        }
        if verbs.is_empty() {
            return Err(LexiconError::EmptySection {
                section: "irregular_verbs",
            });
        }

        let mut past_only = BTreeMap::new();
        for (base, forms) in raw.irregular_past_only {
            check_key("irregular_past_only", &base)?;
            let [past, participle] = past_only_forms(&base, forms)?;
            check_value("irregular_past_only", &past)?;
            check_value("irregular_past_only", &participle)?;
            if verbs.contains_key(base.as_str()) {
                return Err(LexiconError::VerbPastOnlyOverlap { word: base });
            }
            past_only.insert(
                base.into_boxed_str(),
                PastForms {
                    past: past.into_boxed_str(),
                    participle: participle.into_boxed_str(),
                },
            );
        }
        if past_only.is_empty() {
            return Err(LexiconError::EmptySection {
                section: "irregular_past_only",
            });
        }

        let mut nouns = BTreeMap::new();
        for (singular, plural) in raw.irregular_nouns {
            check_key("irregular_nouns", &singular)?;
            check_value("irregular_nouns", &plural)?;
            nouns.insert(singular.into_boxed_str(), plural.into_boxed_str());
        }
        if nouns.is_empty() {
            return Err(LexiconError::EmptySection {
                section: "irregular_nouns",
            });
        }

        let no_double = word_set("no_double", raw.no_double.words)?;
        let subordinators = word_set("subordinators", raw.subordinators.words)?;

        let mut nominal_verbs = BTreeMap::new();
        for (noun, form) in raw.nominal_verbs {
            check_key("nominal_verbs", &noun)?;
            check_value("nominal_verbs", &form)?;
            if !form.ends_with("ing") {
                return Err(LexiconError::NominalVerbNotIng { noun, form });
            }
            nominal_verbs.insert(noun.into_boxed_str(), form.into_boxed_str());
        }
        if nominal_verbs.is_empty() {
            return Err(LexiconError::EmptySection {
                section: "nominal_verbs",
            });
        }

        let generic_subjects = word_set("generic_subjects", raw.generic_subjects.phrases)?;
        let reflexive_pronouns = word_set("pronouns.reflexive", raw.pronouns.reflexive)?;
        let object_pronouns = word_set("pronouns.object", raw.pronouns.object)?;
        let subject_contractions = word_set("contractions.subject", raw.contractions.subject)?;
        let negated_aux_contractions =
            word_set("contractions.negated_aux", raw.contractions.negated_aux)?;
        let stative_verbs = word_set("stative_verbs", raw.stative_verbs.lemmas)?;

        Ok(Self {
            verbs,
            past_only,
            nouns,
            no_double,
            subordinators,
            nominal_verbs,
            generic_subjects,
            reflexive_pronouns,
            object_pronouns,
            subject_contractions,
            negated_aux_contractions,
            stative_verbs,
        })
    }

    /// Whether `lemma` is a known irregular verb's base form (either
    /// table).
    #[must_use]
    pub fn is_irregular_verb_base(&self, lemma: &str) -> bool {
        self.verbs.contains_key(lemma) || self.past_only.contains_key(lemma)
    }
}

/// Validates and builds a [`WordSet`] for `section`, rejecting an empty
/// entry, an uppercase entry, or an empty section.
fn word_set(section: &'static str, words: Vec<String>) -> Result<WordSet, LexiconError> {
    if words.is_empty() {
        return Err(LexiconError::EmptySection { section });
    }
    for word in &words {
        check_key(section, word)?;
    }
    Ok(WordSet::new(words))
}

/// Validates that `key` is non-empty and has no uppercase ASCII letters.
fn check_key(section: &'static str, key: &str) -> Result<(), LexiconError> {
    if key.is_empty() {
        return Err(LexiconError::EmptyString { section });
    }
    if key.bytes().any(|byte| byte.is_ascii_uppercase()) {
        return Err(LexiconError::UppercaseKey {
            section,
            key: key.to_owned(),
        });
    }
    Ok(())
}

/// Validates that a (non-key) value string is non-empty.
const fn check_value(section: &'static str, value: &str) -> Result<(), LexiconError> {
    if value.is_empty() {
        return Err(LexiconError::EmptyString { section });
    }
    Ok(())
}

/// Normalizes an `[irregular_verbs]` entry's 3- or 4-element form array
/// into `[sg3, ing, past, participle]`, defaulting `participle` to `past`
/// for a 3-element entry.
fn verb_forms(
    section: &'static str,
    base: &str,
    forms: Vec<String>,
) -> Result<[String; 4], LexiconError> {
    match <[String; 3]>::try_from(forms) {
        Ok([sg3, ing, past]) => {
            let participle = past.clone();
            Ok([sg3, ing, past, participle])
        }
        Err(forms) => match <[String; 4]>::try_from(forms) {
            Ok(quad) => Ok(quad),
            Err(forms) => Err(LexiconError::VerbFormCount {
                section,
                key: base.to_owned(),
                len: forms.len(),
            }),
        },
    }
}

/// Normalizes an `[irregular_past_only]` entry's 2-element form array
/// into `[past, participle]`.
fn past_only_forms(base: &str, forms: Vec<String>) -> Result<[String; 2], LexiconError> {
    <[String; 2]>::try_from(forms).map_err(|forms| LexiconError::PastOnlyFormCount {
        key: base.to_owned(),
        len: forms.len(),
    })
}

/// Errors produced while parsing or validating a [`Lexicon`] pack.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum LexiconError {
    /// TOML failed to parse, or didn't match the expected pack shape.
    #[error("lexicon pack is not valid TOML: {0}")]
    Toml(#[from] Box<toml::de::Error>),

    /// `[pack].version` isn't `"lexicon-en-v1"`.
    #[error("unsupported lexicon pack version {found:?} (expected {SUPPORTED_VERSION:?})")]
    Version {
        /// The offending version string.
        found: String,
    },

    /// An `[irregular_verbs]` entry's form array has neither 3 nor 4
    /// elements.
    #[error("[{section}] {key:?}: {len}-element form array (expected 3 or 4 elements)")]
    VerbFormCount {
        /// Always `"irregular_verbs"`.
        section: &'static str,
        /// The offending base form.
        key: String,
        /// The array's actual length.
        len: usize,
    },

    /// An `[irregular_past_only]` entry's form array doesn't have exactly
    /// 2 elements.
    #[error(
        "[irregular_past_only] {key:?}: {len}-element form array (expected exactly 2 elements)"
    )]
    PastOnlyFormCount {
        /// The offending base form.
        key: String,
        /// The array's actual length.
        len: usize,
    },

    /// A section has an empty string as a key or value.
    #[error("[{section}]: empty string is not a valid key or value")]
    EmptyString {
        /// Which section the empty string was found in.
        section: &'static str,
    },

    /// A section has a key containing an uppercase ASCII letter.
    #[error("[{section}] {key:?}: keys must not contain uppercase ASCII letters")]
    UppercaseKey {
        /// Which section the offending key was found in.
        section: &'static str,
        /// The offending key.
        key: String,
    },

    /// A base form appears in both `[irregular_verbs]` and
    /// `[irregular_past_only]`; their key sets must stay disjoint (see
    /// `data/lexicon-en.toml`).
    #[error("{word:?} appears in both [irregular_verbs] and [irregular_past_only]")]
    VerbPastOnlyOverlap {
        /// The overlapping base form.
        word: String,
    },

    /// A `[nominal_verbs]` value doesn't end with `"ing"`.
    #[error("[nominal_verbs] {noun:?} -> {form:?}: gerund form must end with \"ing\"")]
    NominalVerbNotIng {
        /// The nominalization key.
        noun: String,
        /// The offending gerund value.
        form: String,
    },

    /// A section has no entries at all.
    #[error("[{section}] must not be empty")]
    EmptySection {
        /// The empty section.
        section: &'static str,
    },
}

impl From<toml::de::Error> for LexiconError {
    fn from(err: toml::de::Error) -> Self {
        Self::Toml(Box::new(err))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The embedded `lexicon-en.toml` this crate ships parses cleanly.
    #[test]
    fn embedded_pack_parses() {
        Lexicon::parse(LEXICON_EN_TOML).expect("embedded pack must parse");
    }

    /// `for_lang(Lang::En)` selects the same static [`LEXICON_EN`] holds.
    #[test]
    fn for_lang_en_selects_lexicon_en() {
        let selected: &Lexicon = Lexicon::for_lang(friction_core::Lang::En);
        assert!(std::ptr::eq(selected, &raw const *LEXICON_EN));
    }

    /// Every section's entry count matches the pinned expectation, so a
    /// silent drop from `data/lexicon-en.toml` fails loudly instead of
    /// quietly shrinking coverage.
    #[test]
    fn embedded_pack_has_pinned_counts() {
        let lexicon = &*LEXICON_EN;
        assert_eq!(lexicon.verbs.len(), 47);
        assert_eq!(lexicon.past_only.len(), 20);
        assert_eq!(lexicon.nouns.len(), 13);
        assert_eq!(lexicon.no_double.len(), 51);
        assert_eq!(lexicon.subordinators.len(), 13);
        assert_eq!(lexicon.nominal_verbs.len(), 23);
        assert_eq!(lexicon.generic_subjects.len(), 6);
        assert_eq!(lexicon.reflexive_pronouns.len(), 9);
        assert_eq!(lexicon.object_pronouns.len(), 8);
        assert_eq!(lexicon.subject_contractions.len(), 30);
        assert_eq!(lexicon.negated_aux_contractions.len(), 16);
        assert_eq!(lexicon.stative_verbs.len(), 9);
    }

    /// Spot-checks a handful of entries against known-correct values.
    #[test]
    fn embedded_pack_spot_values() {
        let lexicon = &*LEXICON_EN;

        let write = &lexicon.verbs["write"];
        assert_eq!(&*write.sg3, "writes");
        assert_eq!(&*write.ing, "writing");
        assert_eq!(&*write.past, "wrote");
        assert_eq!(&*write.participle, "written");

        // "have" is a 3-element entry: participle defaults to past.
        assert_eq!(&*lexicon.verbs["have"].participle, "had");

        let hit = &lexicon.past_only["hit"];
        assert_eq!(&*hit.past, "hit");
        assert_eq!(&*hit.participle, "hit");

        assert_eq!(&*lexicon.nouns["person"], "people");

        assert!(lexicon.no_double.contains("render"));
        assert!(lexicon.no_double.contains("offer"));
        assert!(!lexicon.no_double.contains("prefer"));

        assert!(lexicon.generic_subjects.contains("the team"));
    }

    /// A bad `[pack].version` is rejected.
    #[test]
    fn parse_rejects_bad_version() {
        let bad = sample_toml().replace(
            r#"version = "lexicon-en-v1""#,
            r#"version = "lexicon-en-v2""#,
        );
        let err = Lexicon::parse(&bad).expect_err("bad version must be rejected");
        assert!(matches!(err, LexiconError::Version { .. }));
    }

    /// A 5-element `[irregular_verbs]` form array is rejected.
    #[test]
    fn parse_rejects_five_element_verb_array() {
        let bad = sample_toml().replace(
            r#"write = ["writes", "writing", "wrote", "written"]"#,
            r#"write = ["writes", "writing", "wrote", "written", "extra"]"#,
        );
        let err = Lexicon::parse(&bad).expect_err("5-element form array must be rejected");
        assert!(matches!(err, LexiconError::VerbFormCount { .. }));
    }

    /// An uppercase key anywhere is rejected.
    #[test]
    fn parse_rejects_uppercase_key() {
        let bad = sample_toml().replace("render", "Render");
        let err = Lexicon::parse(&bad).expect_err("uppercase key must be rejected");
        assert!(matches!(err, LexiconError::UppercaseKey { .. }));
    }

    /// A `[nominal_verbs]` value not ending in `"ing"` is rejected.
    #[test]
    fn parse_rejects_nominal_verb_not_ending_in_ing() {
        let bad = sample_toml().replace(
            r#"optimization = "optimizing""#,
            r#"optimization = "optimize""#,
        );
        let err = Lexicon::parse(&bad).expect_err("non-ing gerund must be rejected");
        assert!(matches!(err, LexiconError::NominalVerbNotIng { .. }));
    }

    /// An empty section is rejected.
    #[test]
    fn parse_rejects_empty_section() {
        let bad = sample_toml().replace(r#"words = ["render"]"#, "words = []");
        let err = Lexicon::parse(&bad).expect_err("empty section must be rejected");
        assert!(matches!(err, LexiconError::EmptySection { .. }));
    }

    /// [`WordSet::contains`] against a positive and a negative case.
    #[test]
    fn word_set_contains_positive_and_negative() {
        let set = WordSet::new(vec!["alpha".to_owned(), "beta".to_owned()]);
        assert!(set.contains("alpha"));
        assert!(!set.contains("gamma"));
    }

    /// A minimal but complete pack, one entry per section — small enough
    /// for the rejection tests above to mutate a single line and stay
    /// legible.
    fn sample_toml() -> String {
        r#"
            [pack]
            version = "lexicon-en-v1"

            [irregular_verbs]
            write = ["writes", "writing", "wrote", "written"]
            have = ["has", "having", "had"]

            [irregular_past_only]
            hit = ["hit", "hit"]

            [irregular_nouns]
            person = "people"

            [no_double]
            words = ["render"]

            [subordinators]
            words = ["because"]

            [nominal_verbs]
            optimization = "optimizing"

            [generic_subjects]
            phrases = ["the team"]

            [pronouns]
            reflexive = ["myself"]
            object = ["me"]

            [contractions]
            subject = ["it's"]
            negated_aux = ["isn't"]

            [stative_verbs]
            lemmas = ["have"]
        "#
        .to_owned()
    }
}
