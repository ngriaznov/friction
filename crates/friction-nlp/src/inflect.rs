//! Inflection: given a surface word and a replacement lemma, produce the
//! form of the replacement that agrees with the surface word's
//! morphology.
//!
//! A lexical-substitution helper, not a general morphological generator:
//! [`inflect`] looks only at the *shape* of `surface` (its ending, plus
//! small closed tables of common irregular verbs and nouns) to decide
//! which of four forms to produce — base, third-person-singular present /
//! plural ("-s"), gerund/present-participle ("-ing"), or past ("-ed") —
//! and applies that form to `target_lemma`. It needs no part-of-speech
//! tag: regular "-s" formation is identical for a plural noun and a
//! third-person-singular verb, so one code path covers both.

use crate::lexicon::LEXICON_EN;

/// A lemma's real part of speech, and therefore which inflected forms
/// [`agreeing_forms`] may legitimately derive for it.
///
/// Mechanically generating an "-s" form regardless of a lemma's real part
/// of speech is a genuine false-positive risk: `"valuable"` (adjective)
/// plus "-s" produces `"valuables"`, not "more than one valuable thing"
/// but an *established, differently-meaning* noun. [`WordClass`] fixes
/// this at the root — [`agreeing_forms`] only derives forms a lemma's
/// actual class can legitimately take.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WordClass {
    /// A regular verb: base, third-person-singular/plural (-s/-es), gerund
    /// (-ing), and past (-ed) forms.
    Verb,
    /// A noun: base and plural (-s/-es) forms only.
    Noun,
    /// An adjective or adverb: only its own unmodified base form.
    Adjective,
}

/// Fixed, unambiguously-regular templates (all forms of the plain regular
/// verb "use") used only to derive a lemma's own candidate surface forms
/// via [`inflect`] — see [`agreeing_forms`]'s docs.
///
/// Order matters for [`WordClass::Noun`], which uses only
/// `AGREEMENT_TEMPLATES[0]` (the plural/third-person-singular template).
const AGREEMENT_TEMPLATES: [&str; 3] = ["uses", "using", "used"];

/// The forms `lemma` can legitimately take as a member of `class`.
///
/// Always includes `lemma` itself; [`WordClass::Noun`] additionally
/// includes the plural form; [`WordClass::Verb`] additionally includes
/// plural/third-person-singular, gerund, and past. Deduplicated, in a
/// fixed order (base first, then each [`AGREEMENT_TEMPLATES`] template).
///
/// Reuses [`inflect`] rather than a separate surface-form generator:
/// applying a lemma to fixed, unambiguously-regular templates runs the
/// same code path that later generates an actual replacement, so the two
/// directions (which surface forms match a lemma, what a matched form's
/// replacement looks like) can never quietly disagree.
///
/// # Examples
/// ```
/// use friction_nlp::{WordClass, agreeing_forms};
///
/// let forms = agreeing_forms("leverage", WordClass::Verb);
/// assert!(forms.contains(&"leverages".to_string()));
/// assert!(forms.contains(&"leveraging".to_string()));
///
/// let forms = agreeing_forms("valuable", WordClass::Adjective);
/// assert_eq!(forms, vec!["valuable".to_string()]);
/// ```
#[must_use]
pub fn agreeing_forms(lemma: &str, class: WordClass) -> Vec<String> {
    let templates: &[&str] = match class {
        WordClass::Adjective => &[],
        WordClass::Noun => &AGREEMENT_TEMPLATES[..1],
        WordClass::Verb => &AGREEMENT_TEMPLATES,
    };
    let mut forms = vec![lemma.to_string()];
    for &template in templates {
        if let Some(form) = inflect(template, lemma)
            && !forms.contains(&form)
        {
            forms.push(form);
        }
    }
    forms
}

/// [`LEXICON_EN`]'s `verbs` table, indexed the other way round: every
/// inflected form (3sg, gerund, past) maps to its base, built once.
/// [`lemmatize`]'s own irregular-table check is a linear scan repeated by
/// every caller that tries multiple POS tags against the same surface
/// word (the scan itself doesn't depend on `pos`) — [`irregular_verb_base`]
/// gives those callers a single hash lookup instead.
///
/// Built from `verbs` only, never `past_only`: surface classification
/// (this map's sole purpose) is defined over `verbs`' key set alone, so a
/// `past_only` base like "hit" is deliberately absent here too — see
/// [`Lexicon::past_only`](crate::lexicon::Lexicon::past_only).
///
/// `FxHashMap` (rustc's own hasher): seedless and deterministic, unlike
/// `ahash`/`foldhash`'s random default seeding, which this workspace's
/// reproducibility doctrine rules out for anything on a hot path — and
/// std's `HashMap` default `SipHash`'s `DoS` resistance buys nothing here,
/// since these keys are compiled-in strings, never attacker input.
static IRREGULAR_VERB_BASES: std::sync::LazyLock<
    rustc_hash::FxHashMap<&'static str, &'static str>,
> = std::sync::LazyLock::new(|| {
    let verbs = &LEXICON_EN.verbs;
    let mut map =
        rustc_hash::FxHashMap::with_capacity_and_hasher(verbs.len() * 3, rustc_hash::FxBuildHasher);
    for (base, forms) in verbs {
        for form in [&forms.sg3, &forms.ing, &forms.past] {
            if form.as_ref() != base.as_ref() {
                let displaced = map.insert(form.as_ref(), base.as_ref());
                // [`lemmatize`]'s linear scan returns the FIRST matching
                // row. A map keeps the LAST insert. The two agree only
                // while every inflected form is a distinct key, so a new
                // table row that breaks that silently changes lemmas.
                // Fail loudly instead.
                assert!(
                    displaced.is_none_or(|prev| prev == base.as_ref()),
                    "LEXICON_EN.verbs: form {form:?} maps to two bases"
                );
            }
        }
    }
    map
});

/// [`LEXICON_EN`]'s `nouns` table, indexed plural -> singular —
/// [`lemmatize`]'s noun table lookup, hashed for the same hot-path reason
/// (and with the same seedless-`FxHashMap` determinism argument) as
/// [`IRREGULAR_VERB_BASES`] above.
static IRREGULAR_NOUN_SINGULARS: std::sync::LazyLock<
    rustc_hash::FxHashMap<&'static str, &'static str>,
> = std::sync::LazyLock::new(|| {
    let nouns = &LEXICON_EN.nouns;
    let mut map =
        rustc_hash::FxHashMap::with_capacity_and_hasher(nouns.len(), rustc_hash::FxBuildHasher);
    for (singular, plural) in nouns {
        let displaced = map.insert(plural.as_ref(), singular.as_ref());
        assert!(
            displaced.is_none_or(|prev| prev == singular.as_ref()),
            "LEXICON_EN.nouns: plural {plural:?} maps to two singulars"
        );
    }
    map
});

/// The base form of `lower`, if it's an irregular verb's inflected form.
///
/// A direct hash-map lookup over [`LEXICON_EN`]'s `verbs` table
/// (third-person-singular, gerund, or past form only — same table,
/// reverse direction), for callers on a hot path who'd otherwise pay for
/// `lemmatize`'s linear scan once per POS tag tried against the same
/// word. `lower` must already be lowercased; unlike [`lemmatize`] this
/// performs no case folding of its own.
///
/// # Examples
/// ```
/// use friction_nlp::irregular_verb_base;
///
/// assert_eq!(irregular_verb_base("went"), Some("go"));
/// assert_eq!(irregular_verb_base("walked"), None);
/// ```
#[must_use]
pub fn irregular_verb_base(lower: &str) -> Option<&'static str> {
    IRREGULAR_VERB_BASES.get(lower).copied()
}

/// Best-effort reverse lemmatization for a tagged surface word.
///
/// Given its surface text and a coarse Penn tag, guesses the base-form
/// lemma by generating candidate stems and keeping the one that
/// round-trips back to `surface` through [`inflect`]'s own forward
/// generation: so this never duplicates a suffix/irregular rule, only
/// reuses the forward direction's tables and logic in reverse. Falls
/// back to the lowercased `surface` when no candidate round-trips,
/// matching [`crate::TaggedToken::lemma`]'s documented fallback.
///
/// `pos` only picks which suffix family to try (`VBG` gerunds,
/// `VBD`/`VBN` past forms, `VBZ`/`NNS` "-s" forms); any other tag returns
/// the lowercased surface unchanged, since a base-form word's lemma is
/// already itself.
#[must_use]
pub fn lemmatize(surface: &str, pos: &str) -> Box<str> {
    let lower = surface.to_lowercase();
    if !lower.chars().any(char::is_alphabetic) {
        return Box::from(lower);
    }

    // Hash lookups over the same two tables the old linear scans walked
    // (verbs before nouns, preserving the scan order). The one shape
    // difference — [`IRREGULAR_VERB_BASES`] skips a form equal to its own
    // base ("put"/"read"/"set" past forms) — is unobservable: those words
    // carry no strippable suffix, so the candidate loop below is empty and
    // the fallback returns `lower`, which IS the base, byte for byte.
    if let Some(&base) = IRREGULAR_VERB_BASES.get(lower.as_str()) {
        return Box::from(base);
    }
    if let Some(&singular) = IRREGULAR_NOUN_SINGULARS.get(lower.as_str()) {
        return Box::from(singular);
    }

    // `probe` is always one of three fixed literals below, so its `Form`
    // is hardcoded rather than rediscovered by `inflect`'s own
    // `classify_surface_form` call on every candidate — pinned against
    // that classification by `lemmatize_probe_forms_match_classification`
    // in this module's tests.
    let (candidates, form, probe): (Vec<String>, Form, &str) = match pos {
        "VBG" => (ing_candidates(&lower), Form::Ing, "using"),
        "VBD" | "VBN" => (past_candidates(&lower), Form::Past, "used"),
        "VBZ" | "NNS" => (suffix_s_candidates(&lower), Form::SuffixS, "uses"),
        _ => (Vec::new(), Form::Base, ""),
    };
    for candidate in candidates {
        if inflect_as(form, &candidate, probe).as_deref() == Some(lower.as_str()) {
            return candidate.into_boxed_str();
        }
    }

    Box::from(lower)
}

/// Undoes a doubled final consonant before a vowel suffix was stripped
/// (`"stopp"` -> also try `"stop"`), the reverse of
/// [`should_double_final_consonant`]'s forward rule.
fn undoubled(stem: &str) -> Option<String> {
    let chars: Vec<char> = stem.chars().collect();
    let n = chars.len();
    if n >= 2 && chars[n - 1] == chars[n - 2] && !is_vowel(chars[n - 1]) {
        Some(chars[..n - 1].iter().collect())
    } else {
        None
    }
}

/// Candidate base-form stems for a word tagged `VBG` (gerund/present
/// participle, `"-ing"`), in priority order.
///
/// Order matters more than it looks: naively stripping `"-ing"` almost
/// always leaves a stem that trivially regenerates the original surface
/// through [`inflect`]'s default branch, whether or not it's the *real*
/// lemma — so a structurally-informed candidate (undoing a doubled final
/// consonant, restoring a `"-ie"` ending) must be tried *before* the
/// naive stem, or the naive-but-wrong one always wins by being checked
/// first. A short (<=2-character) stem is additionally tried with a
/// restored trailing `"e"` first, since a bare two-letter consonant-final
/// "word" is rarely a genuine English verb on its own (`"us"` vs. the
/// real `"use"`).
fn ing_candidates(word: &str) -> Vec<String> {
    let Some(stem) = word.strip_suffix("ing") else {
        return Vec::new();
    };
    let mut out = Vec::new();
    if let Some(un) = undoubled(stem) {
        out.push(un);
    }
    // Only the genuine `"ie"`-verb family (tie/die/lie/vie: stem length 2
    // after stripping `"-ing"`) restores an `"-ie"` ending — a longer
    // `"-ying"` stem (`"carrying"` -> `"carry"`, `"worrying"` -> `"worry"`)
    // is the far more common plain consonant-`"y"` verb and must not be
    // shadowed by a wrongly-restored `"-ie"` guess.
    if stem.chars().count() <= 2
        && let Some(base) = word.strip_suffix("ying")
    {
        out.push(format!("{base}ie"));
    }
    if stem.chars().count() <= 2 {
        out.push(format!("{stem}e"));
        out.push(stem.to_string());
    } else {
        out.push(stem.to_string());
        out.push(format!("{stem}e"));
    }
    out
}

/// Candidate base-form stems for a word tagged `VBD`/`VBN` (past tense /
/// past participle, `"-ed"`), in priority order — see [`ing_candidates`]'s
/// docs for why structurally-informed candidates (`"-ied"` -> `"-y"`,
/// undoubling) must come before the naive stripped stem.
fn past_candidates(word: &str) -> Vec<String> {
    let mut out = Vec::new();
    if let Some(stem) = word.strip_suffix("ied") {
        out.push(format!("{stem}y"));
    }
    let Some(stem) = word.strip_suffix("ed") else {
        return out;
    };
    if let Some(un) = undoubled(stem) {
        out.push(un);
    }
    if stem.chars().count() <= 2 {
        out.push(format!("{stem}e"));
        out.push(stem.to_string());
    } else {
        out.push(stem.to_string());
        out.push(format!("{stem}e"));
    }
    out
}

/// Candidate base-form stems for a word tagged `VBZ`/`NNS`
/// (third-person-singular present / plural, `"-s"`).
fn suffix_s_candidates(word: &str) -> Vec<String> {
    let mut out = Vec::new();
    if let Some(stem) = word.strip_suffix("ies") {
        out.push(format!("{stem}y"));
    }
    if let Some(stem) = word.strip_suffix("es") {
        out.push(stem.to_string());
    }
    if let Some(stem) = word.strip_suffix('s') {
        out.push(stem.to_string());
    }
    out
}

/// Produces the form of `target_lemma` that agrees with `surface`'s
/// morphology, with `surface`'s capitalization pattern (lowercase, Title
/// Case, or ALL CAPS) transferred onto the result.
///
/// Returns `None` if either `surface` or `target_lemma` has no
/// alphabetic character — inflecting a non-word token isn't meaningful.
///
/// # Examples
/// ```
/// use friction_nlp::inflect;
///
/// assert_eq!(inflect("leverages", "use").as_deref(), Some("uses"));
/// assert_eq!(inflect("Leveraging", "use").as_deref(), Some("Using"));
/// assert_eq!(inflect("utilized", "use").as_deref(), Some("used"));
/// ```
#[must_use]
pub fn inflect(surface: &str, target_lemma: &str) -> Option<String> {
    if !surface.chars().any(char::is_alphabetic) {
        return None;
    }
    let form = classify_surface_form(&surface.to_lowercase());
    inflect_as(form, target_lemma, surface)
}

/// [`inflect`]'s generation half, taking an already-known `Form` rather
/// than classifying `surface` itself — split out so [`lemmatize`]'s
/// round-trip loop can hardcode the `Form` for its fixed probe literals
/// and skip [`classify_surface_form`]'s table scan entirely, once per
/// candidate, on every call.
fn inflect_as(form: Form, target_lemma: &str, surface: &str) -> Option<String> {
    if !target_lemma.chars().any(char::is_alphabetic) {
        return None;
    }
    let generated = generate(&target_lemma.to_lowercase(), form);
    Some(apply_capitalization(
        &generated,
        detect_capitalization(surface),
    ))
}

/// The morphological form [`inflect`] detected on a surface word, and will
/// reproduce on the target lemma.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Form {
    /// Uninflected: singular noun, verb infinitive/plural-agreement form.
    Base,
    /// Third-person-singular present verb, or plural noun: both formed
    /// the same way in English ("-s"/"-es"/consonant-y -> "-ies").
    SuffixS,
    /// Gerund / present participle ("-ing").
    Ing,
    /// Past tense / past participle ("-ed", or irregular).
    Past,
}

fn classify_surface_form(surface_lower: &str) -> Form {
    classify_irregular(surface_lower).unwrap_or_else(|| classify_regular(surface_lower))
}

/// Classifies `word` against [`LEXICON_EN`]'s `verbs` (sg3/ing/past/base,
/// in that per-entry precedence) then `nouns` (plural/singular) tables.
/// Never consults `past_only`: surface classification is defined over
/// `verbs`' key set alone (see [`IRREGULAR_VERB_BASES`]'s docs).
///
/// `verbs`/`nouns` are `BTreeMap`s, so iteration is alphabetical rather
/// than the old curated array order — safe only because no surface form
/// is assigned conflicting outcomes by two different entries, which the
/// `verbs_table_*` tests below pin.
fn classify_irregular(word: &str) -> Option<Form> {
    for (base, forms) in &LEXICON_EN.verbs {
        if word == forms.sg3.as_ref() {
            return Some(Form::SuffixS);
        }
        if word == forms.ing.as_ref() {
            return Some(Form::Ing);
        }
        if word == forms.past.as_ref() {
            return Some(Form::Past);
        }
        if word == base.as_ref() {
            return Some(Form::Base);
        }
    }
    for (singular, plural) in &LEXICON_EN.nouns {
        if word == plural.as_ref() {
            return Some(Form::SuffixS);
        }
        if word == singular.as_ref() {
            return Some(Form::Base);
        }
    }
    None
}

fn classify_regular(word: &str) -> Form {
    if word.len() > 3 && word.ends_with("ing") {
        Form::Ing
    } else if word.len() > 2 && word.ends_with("ed") {
        Form::Past
    } else if word.len() > 1
        && word.ends_with('s')
        && !word.ends_with("ss")
        && !word.ends_with("us")
        && !word.ends_with("is")
    {
        Form::SuffixS
    } else {
        Form::Base
    }
}

fn generate(lemma: &str, form: Form) -> String {
    match form {
        Form::Base => lemma.to_string(),
        Form::SuffixS => generate_suffix_s(lemma),
        Form::Ing => generate_ing(lemma),
        Form::Past => generate_past(lemma),
    }
}

fn generate_suffix_s(lemma: &str) -> String {
    if LEXICON_EN
        .nouns
        .values()
        .any(|plural| plural.as_ref() == lemma)
    {
        // Already an irregular plural (or invariant form used directly as
        // a replacement lemma, e.g. "people"): nothing to add.
        return lemma.to_string();
    }
    if let Some(plural) = LEXICON_EN.nouns.get(lemma) {
        return plural.to_string();
    }
    if let Some(forms) = LEXICON_EN.verbs.get(lemma) {
        return forms.sg3.to_string();
    }
    regular_suffix_s(lemma)
}

fn regular_suffix_s(lemma: &str) -> String {
    if lemma.ends_with(['s', 'x', 'z']) || lemma.ends_with("ch") || lemma.ends_with("sh") {
        format!("{lemma}es")
    } else if ends_with_consonant_y(lemma) {
        format!("{}ies", &lemma[..lemma.len() - 1])
    } else {
        format!("{lemma}s")
    }
}

fn generate_ing(lemma: &str) -> String {
    if let Some(forms) = LEXICON_EN.verbs.get(lemma) {
        return forms.ing.to_string();
    }
    regular_ing(lemma)
}

// A `strip_suffix`-then-`map_or_else` reads as a single expression here
// only by threading `lemma` through three more closures; the plain
// if/else-if chain (matching `regular_past`'s shape below) stays readable.
#[allow(clippy::option_if_let_else)]
fn regular_ing(lemma: &str) -> String {
    if let Some(stem) = lemma.strip_suffix("ie") {
        format!("{stem}ying")
    } else if ends_with_silent_e(lemma) {
        format!("{}ing", &lemma[..lemma.len() - 1])
    } else if should_double_final_consonant(lemma) {
        format!(
            "{lemma}{}ing",
            lemma.chars().last().expect("checked non-empty")
        )
    } else {
        format!("{lemma}ing")
    }
}

fn generate_past(lemma: &str) -> String {
    if let Some(forms) = LEXICON_EN.verbs.get(lemma) {
        return forms.past.to_string();
    }
    regular_past(lemma)
}

fn regular_past(lemma: &str) -> String {
    if lemma.ends_with('e') {
        format!("{lemma}d")
    } else if ends_with_consonant_y(lemma) {
        format!("{}ied", &lemma[..lemma.len() - 1])
    } else if should_double_final_consonant(lemma) {
        format!(
            "{lemma}{}ed",
            lemma.chars().last().expect("checked non-empty")
        )
    } else {
        format!("{lemma}ed")
    }
}

/// The past-tense form of `lemma`: irregular tables first, else the
/// regular suffix rule.
///
/// Checks [`LEXICON_EN`]'s `verbs` table, then `past_only` — unlike
/// [`classify_irregular`]/[`generate_past`] above, which never consult
/// `past_only` (see [`IRREGULAR_VERB_BASES`]'s docs). This is the public,
/// general-purpose past-tense lookup; those are the internal
/// surface-classification path, pinned to its own narrower table on
/// purpose.
#[must_use]
pub fn past(lemma: &str) -> String {
    if let Some(forms) = LEXICON_EN.verbs.get(lemma) {
        return forms.past.to_string();
    }
    if let Some(forms) = LEXICON_EN.past_only.get(lemma) {
        return forms.past.to_string();
    }
    regular_past(lemma)
}

/// The past-participle form of `lemma`: irregular tables first, else the
/// regular suffix rule. See [`past`]'s docs for why this consults
/// `past_only` and the internal generators above do not.
#[must_use]
pub fn past_participle(lemma: &str) -> String {
    if let Some(forms) = LEXICON_EN.verbs.get(lemma) {
        return forms.participle.to_string();
    }
    if let Some(forms) = LEXICON_EN.past_only.get(lemma) {
        return forms.participle.to_string();
    }
    regular_past(lemma)
}

/// Endings after which the regular third-person-singular suffix is
/// `"-es"`, not `"-s"`. `"o"` joins the true sibilants (`"go"` ->
/// `"goes"`) because English spelling extends the same epenthetic vowel
/// to a trailing `"o"`, not because it's phonetically a sibilant.
const SIBILANT_ENDINGS: &[&str] = &["s", "x", "z", "ch", "sh", "o"];

/// The regular third-person-singular present form of `lemma` (`"deploy"`
/// -> `"deploys"`, `"carry"` -> `"carries"`).
///
/// Consults no irregular table: English third-singular formation is
/// regular outside `be`/`have`. Its `"o"` ending deliberately diverges
/// from [`regular_suffix_s`]'s ("go" -> "goes" here, but "photo" ->
/// "photos" there) — don't unify them, nouns don't take the epenthetic
/// vowel a verb's "-o" does.
#[must_use]
pub fn third_sg(lemma: &str) -> String {
    if SIBILANT_ENDINGS
        .iter()
        .any(|suffix| lemma.ends_with(suffix))
    {
        format!("{lemma}es")
    } else if ends_with_consonant_y(lemma) {
        format!("{}ies", &lemma[..lemma.len() - 1])
    } else {
        format!("{lemma}s")
    }
}

/// `lemma` ends in "y" preceded by a consonant (so `y` -> `i` before
/// "-es"/"-ed": "carry" -> "carries"/"carried"), as opposed to a vowel
/// (which keeps the `y`: "play" -> "plays"/"played").
fn ends_with_consonant_y(lemma: &str) -> bool {
    let chars: Vec<char> = lemma.chars().collect();
    let Some(&last) = chars.last() else {
        return false;
    };
    last == 'y' && chars.len() >= 2 && !is_vowel(chars[chars.len() - 2])
}

/// `lemma` ends in a silent "e" that a vowel suffix ("-ing") drops
/// ("use" -> "using"), excluding endings where the "e" is pronounced/part
/// of a double vowel and stays ("agree" -> "agreeing", "see" -> "seeing").
fn ends_with_silent_e(lemma: &str) -> bool {
    lemma.ends_with('e')
        && !lemma.ends_with("ee")
        && !lemma.ends_with("oe")
        && !lemma.ends_with("ye")
}

/// English's "1-1-1" doubling rule, approximated without syllable/stress
/// information: double the final consonant before a vowel suffix when the
/// word ends in a single vowel followed by a single non-w/x/y consonant
/// preceded by another consonant (or is only three letters), and isn't a
/// known exception (see [`LEXICON_EN`]'s `no_double`).
fn should_double_final_consonant(lemma: &str) -> bool {
    if LEXICON_EN.no_double.contains(lemma) {
        return false;
    }
    let chars: Vec<char> = lemma.chars().collect();
    if chars.len() < 3 {
        return false;
    }
    let last = chars[chars.len() - 1];
    let before_last = chars[chars.len() - 2];
    if is_vowel(last) || matches!(last, 'w' | 'x' | 'y') || !is_vowel(before_last) {
        return false;
    }
    if chars.len() == 3 {
        return true;
    }
    // Past this point chars.len() >= 4 (the == 3 case already returned).
    let two_back = chars[chars.len() - 3];
    let three_back = chars[chars.len() - 4];
    // "qu" is a single consonant unit (/kw/): the 'u' after 'q' isn't a
    // second vowel, so "quit"/"equip" are still single-vowel CVC words
    // for doubling ("quitting", "equipping"), not VVC words like "boat"
    // ("boating").
    let two_back_is_real_vowel = is_vowel(two_back) && !(two_back == 'u' && three_back == 'q');
    !two_back_is_real_vowel
}

const fn is_vowel(c: char) -> bool {
    matches!(c.to_ascii_lowercase(), 'a' | 'e' | 'i' | 'o' | 'u')
}

/// The capitalization pattern detected on a surface word, to be
/// transferred onto a generated replacement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Capitalization {
    Lower,
    Title,
    AllCaps,
}

fn detect_capitalization(surface: &str) -> Capitalization {
    let alphabetic: Vec<char> = surface.chars().filter(|c| c.is_alphabetic()).collect();
    match alphabetic.as_slice() {
        [only] => {
            if only.is_uppercase() {
                Capitalization::Title
            } else {
                Capitalization::Lower
            }
        }
        [first, rest @ ..] if rest.iter().all(|c| c.is_uppercase()) && first.is_uppercase() => {
            Capitalization::AllCaps
        }
        [first, ..] if first.is_uppercase() => Capitalization::Title,
        _ => Capitalization::Lower,
    }
}

fn apply_capitalization(word: &str, cap: Capitalization) -> String {
    match cap {
        Capitalization::Lower => word.to_string(),
        Capitalization::AllCaps => word.to_uppercase(),
        Capitalization::Title => {
            let mut chars = word.chars();
            chars.next().map_or_else(String::new, |first| {
                first.to_uppercase().collect::<String>() + chars.as_str()
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `(surface, target_lemma, expected)` triples covering: regular
    /// third-person-singular/plural "-s" (plain, sibilant "+es",
    /// consonant-y "-ies"), regular gerund "-ing" (silent-e drop, "ie" ->
    /// "ying", doubling, vowel-y kept), regular past "-ed" (silent-e,
    /// consonant-y, doubling), the irregular-verb table (both
    /// directions), irregular noun plurals (both directions),
    /// capitalization transfer independent of target-lemma casing,
    /// base-form passthrough, and `None` for non-word input.
    const GOLDEN: &[(&str, &str, Option<&str>)] = &[
        // --- regular third-person-singular / plural "-s" ---
        ("leverages", "use", Some("uses")),
        ("utilizes", "employ", Some("employs")),
        ("optimizes", "improve", Some("improves")),
        ("facilitates", "help", Some("helps")),
        ("showcases", "show", Some("shows")),
        ("watches", "see", Some("sees")),
        ("carries", "bring", Some("brings")),
        ("tries", "attempt", Some("attempts")),
        ("matches", "fit", Some("fits")),
        ("wants", "need", Some("needs")),
        // --- irregular surface -> regular target (via IRREGULAR_VERBS) ---
        ("goes", "come", Some("comes")),
        ("is", "become", Some("becomes")),
        ("has", "own", Some("owns")),
        ("does", "perform", Some("performs")),
        // --- regular surface -> irregular target (via IRREGULAR_VERBS) ---
        ("wants", "go", Some("goes")),
        ("needs", "have", Some("has")),
        // --- gerund "-ing" ---
        ("leveraging", "use", Some("using")),
        ("utilizing", "employ", Some("employing")),
        ("running", "execute", Some("executing")),
        ("stopping", "halt", Some("halting")),
        ("getting", "obtain", Some("obtaining")),
        ("beginning", "start", Some("starting")),
        ("planning", "design", Some("designing")),
        ("dying", "expire", Some("expiring")),
        ("carrying", "supply", Some("supplying")),
        ("tying", "lie", Some("lying")),
        ("dying", "die", Some("dying")),
        ("going", "come", Some("coming")),
        ("having", "be", Some("being")),
        // --- consonant doubling (CVC target lemma) ---
        ("planning", "ban", Some("banning")),
        ("running", "occur", Some("occurring")),
        ("stopping", "quit", Some("quitting")),
        ("planned", "equip", Some("equipped")),
        ("used", "prefer", Some("preferred")),
        // --- NO_DOUBLE_EXCEPTIONS: an unstressed final syllable that
        // looks CVC but must not double ---
        ("used", "encounter", Some("encountered")),
        ("used", "empower", Some("empowered")),
        ("used", "discover", Some("discovered")),
        ("used", "render", Some("rendered")),
        ("used", "surrender", Some("surrendered")),
        ("used", "offer", Some("offered")),
        ("used", "order", Some("ordered")),
        // --- past "-ed" ---
        ("utilized", "use", Some("used")),
        ("leveraged", "employ", Some("employed")),
        ("optimized", "improve", Some("improved")),
        ("facilitated", "help", Some("helped")),
        ("carried", "supply", Some("supplied")),
        ("stopped", "halt", Some("halted")),
        ("planned", "design", Some("designed")),
        ("tried", "attempt", Some("attempted")),
        ("showcased", "show", Some("showed")),
        ("went", "come", Some("came")),
        ("was", "have", Some("had")),
        ("built", "make", Some("made")),
        // --- irregular noun plurals (both directions) ---
        ("individuals", "people", Some("people")),
        ("children", "kid", Some("kids")),
        ("mice", "rat", Some("rats")),
        ("geese", "duck", Some("ducks")),
        ("feet", "foot", Some("feet")),
        ("women", "lady", Some("ladies")),
        ("analyses", "report", Some("reports")),
        // --- base form: no inflection ---
        ("use", "leverage", Some("leverage")),
        ("person", "individual", Some("individual")),
        // --- capitalization transfer ---
        ("Leveraging", "use", Some("Using")),
        ("UTILIZES", "use", Some("USES")),
        ("Individuals", "people", Some("People")),
        ("LEVERAGED", "use", Some("USED")),
        ("Optimizing", "improve", Some("Improving")),
        ("Runs", "go", Some("Goes")),
        ("WENT", "come", Some("CAME")),
        ("Use", "leverage", Some("Leverage")),
        // --- non-word input: no inflection to perform ---
        ("123", "use", None),
        ("", "use", None),
        ("use", "", None),
        ("...", "use", None),
    ];

    #[test]
    fn inflect_matches_golden_pairs() {
        assert!(
            GOLDEN.len() >= 50,
            "golden table should cover at least 50 pairs, has {}",
            GOLDEN.len()
        );
        for &(surface, target_lemma, expected) in GOLDEN {
            let actual = inflect(surface, target_lemma);
            assert_eq!(
                actual.as_deref(),
                expected,
                "inflect({surface:?}, {target_lemma:?})"
            );
        }
    }

    /// [`inflect`] is a pure function: repeated calls with the same
    /// arguments produce byte-identical output.
    #[test]
    fn inflect_is_deterministic() {
        for &(surface, target_lemma, _) in GOLDEN {
            assert_eq!(
                inflect(surface, target_lemma),
                inflect(surface, target_lemma)
            );
        }
    }

    #[test]
    fn agreeing_forms_verb_covers_all_four_inflections() {
        let forms = agreeing_forms("leverage", WordClass::Verb);
        for expected in ["leverage", "leverages", "leveraging", "leveraged"] {
            assert!(
                forms.contains(&expected.to_string()),
                "missing {expected:?} in {forms:?}"
            );
        }
    }

    #[test]
    fn agreeing_forms_noun_includes_only_base_and_plural() {
        let forms = agreeing_forms("individual", WordClass::Noun);
        assert_eq!(
            forms,
            vec!["individual".to_string(), "individuals".to_string()]
        );
    }

    /// An adjective only ever matches its own base form — never a
    /// mechanically-generated "-s" form, which is exactly what keeps
    /// `"valuable"`/`"vital"`/`"initial"` from spuriously matching the
    /// unrelated real words `"valuables"`/`"vitals"`/`"initials"`.
    #[test]
    fn agreeing_forms_adjective_is_base_form_only() {
        assert_eq!(
            agreeing_forms("valuable", WordClass::Adjective),
            vec!["valuable".to_string()]
        );
    }

    /// `(surface, pos, expected_lemma)` triples covering the regular
    /// gerund/past/"-s" reverse-derivation paths, the irregular-verb and
    /// irregular-noun tables consulted first, and the documented
    /// surface-text fallback for a tag this function does not attempt
    /// (or a candidate that fails to round-trip).
    const LEMMATIZE_GOLDEN: &[(&str, &str, &str)] = &[
        ("jumping", "VBG", "jump"),
        ("using", "VBG", "use"),
        ("carrying", "VBG", "carry"),
        ("stopping", "VBG", "stop"),
        ("tying", "VBG", "tie"),
        ("walked", "VBD", "walk"),
        ("used", "VBD", "use"),
        ("carried", "VBD", "carry"),
        ("stopped", "VBD", "stop"),
        ("scans", "VBZ", "scan"),
        ("foxes", "NNS", "fox"),
        ("carries", "VBZ", "carry"),
        ("went", "VBD", "go"),
        ("children", "NNS", "child"),
        ("was", "VBD", "be"),
        ("dog", "NN", "dog"),
        ("Running", "VBG", "run"),
    ];

    #[test]
    fn lemmatize_matches_golden_pairs() {
        for &(surface, pos, expected) in LEMMATIZE_GOLDEN {
            assert_eq!(
                &*lemmatize(surface, pos),
                expected,
                "lemmatize({surface:?}, {pos:?})"
            );
        }
    }

    /// [`lemmatize`] hardcodes `Form::Ing`/`Form::Past`/`Form::SuffixS` for
    /// its `"using"`/`"used"`/`"uses"` probes instead of calling
    /// [`classify_surface_form`] on them (see its own comment). Pins that
    /// the hardcoded value actually matches what classification would
    /// produce, so a future `LEXICON_EN` edit that changed it fails here
    /// instead of silently mis-lemmatizing.
    #[test]
    fn lemmatize_probe_forms_match_classification() {
        assert_eq!(classify_surface_form("using"), Form::Ing);
        assert_eq!(classify_surface_form("used"), Form::Past);
        assert_eq!(classify_surface_form("uses"), Form::SuffixS);
    }

    #[test]
    fn lemmatize_is_deterministic() {
        for &(surface, pos, _) in LEMMATIZE_GOLDEN {
            assert_eq!(lemmatize(surface, pos), lemmatize(surface, pos));
        }
    }

    #[test]
    fn agreeing_forms_is_deterministic() {
        for (lemma, class) in [
            ("leverage", WordClass::Verb),
            ("individual", WordClass::Noun),
            ("robust", WordClass::Adjective),
        ] {
            assert_eq!(agreeing_forms(lemma, class), agreeing_forms(lemma, class));
        }
    }

    /// [`past`]: `verbs` table, then `past_only`, then the regular rule.
    #[test]
    fn past_covers_all_three_sources() {
        assert_eq!(past("go"), "went");
        assert_eq!(past("write"), "wrote");
        assert_eq!(past("hit"), "hit");
        assert_eq!(past("deploy"), "deployed");
        assert_eq!(past("render"), "rendered");
    }

    /// [`past_participle`]: `verbs` table (explicit participle or,
    /// pinned, a 3-element entry's fallback to its own past form), then
    /// `past_only`, then the regular rule.
    #[test]
    fn past_participle_covers_all_three_sources() {
        assert_eq!(past_participle("write"), "written");
        assert_eq!(past_participle("go"), "gone");
        assert_eq!(past_participle("hit"), "hit");
        assert_eq!(past_participle("show"), "shown");
        // "know" is a 3-element [irregular_verbs] entry: no explicit
        // participle, so this falls back to "knew" (== past), not "known".
        assert_eq!(past_participle("know"), "knew");
        assert_eq!(past_participle("deploy"), "deployed");
        assert_eq!(past_participle("prefer"), "preferred");
    }

    /// [`third_sg`]: sibilant/"o" -> "-es", consonant-"y" -> "-ies", else
    /// "-s". No irregular table.
    #[test]
    fn third_sg_matches_expected_forms() {
        assert_eq!(third_sg("deploy"), "deploys");
        assert_eq!(third_sg("watch"), "watches");
        assert_eq!(third_sg("carry"), "carries");
        assert_eq!(third_sg("go"), "goes");
    }

    /// Pins [`classify_irregular`]'s safety property under `verbs`'
    /// `BTreeMap` (alphabetical) iteration: no surface form is assigned a
    /// different [`Form`] by two different entries. The old code walked a
    /// curated `IRREGULAR_VERBS` array in a fixed order, so "the first
    /// entry that recognizes a word wins" was well-defined only by that
    /// order; this proves cross-entry "first" never actually matters, so
    /// alphabetical iteration is safe too.
    ///
    /// Within one entry, a word can legitimately match more than one
    /// field (`"put"`'s `past` and its own base form are both `"put"`);
    /// that's resolved by the fixed sg3/ing/past/base check order, not by
    /// entry iteration order, so it's collapsed to that entry's single
    /// first-match answer before comparing across entries.
    #[test]
    fn verbs_table_has_no_cross_entry_form_conflicts() {
        let mut seen: std::collections::HashMap<&str, Form> = std::collections::HashMap::new();
        for (base, forms) in &LEXICON_EN.verbs {
            let mut local: std::collections::HashMap<&str, Form> = std::collections::HashMap::new();
            for (word, form) in [
                (forms.sg3.as_ref(), Form::SuffixS),
                (forms.ing.as_ref(), Form::Ing),
                (forms.past.as_ref(), Form::Past),
                (base.as_ref(), Form::Base),
            ] {
                local.entry(word).or_insert(form);
            }
            for (word, form) in local {
                match seen.insert(word, form) {
                    Some(prev) if prev != form => {
                        panic!("{word:?} classifies as both {prev:?} and {form:?} (entry {base:?})")
                    }
                    _ => {}
                }
            }
        }
    }

    /// Same property as `verbs_table_has_no_cross_entry_form_conflicts`,
    /// for `nouns` (plural/singular).
    #[test]
    fn nouns_table_has_no_cross_entry_form_conflicts() {
        let mut seen: std::collections::HashMap<&str, Form> = std::collections::HashMap::new();
        for (singular, plural) in &LEXICON_EN.nouns {
            for (word, form) in [
                (plural.as_ref(), Form::SuffixS),
                (singular.as_ref(), Form::Base),
            ] {
                match seen.insert(word, form) {
                    Some(prev) if prev != form => panic!(
                        "{word:?} classifies as both {prev:?} and {form:?} (entry {singular:?})"
                    ),
                    _ => {}
                }
            }
        }
    }
}
