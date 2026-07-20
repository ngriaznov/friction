//! Inflection: given a surface word and a replacement lemma, produce the
//! form of the replacement that agrees with the surface word's
//! morphology.
//!
//! This is a lexical-substitution helper, not a general morphological
//! generator: [`inflect`] looks only at the *shape* of `surface` (its
//! ending, plus small closed tables of common irregular verbs and nouns)
//! to decide which of four forms to produce — base, third-person-singular
//! present / plural ("-s"), gerund/present-participle ("-ing"), or past
//! ("-ed") — and applies that same form to `target_lemma`. It does not
//! receive or need a part-of-speech tag: regular "-s" formation is
//! identical for a plural noun and a third-person-singular verb, so one
//! code path covers both.

/// Produces the form of `target_lemma` that agrees with `surface`'s
/// morphology, with `surface`'s capitalization pattern (lowercase, Title
/// Case, or ALL CAPS) transferred onto the result.
///
/// Returns `None` if either `surface` or `target_lemma` contains no
/// alphabetic character — inflecting a token that is not a word is not
/// meaningful.
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
    if !surface.chars().any(char::is_alphabetic) || !target_lemma.chars().any(char::is_alphabetic) {
        return None;
    }

    let surface_lower = surface.to_lowercase();
    let lemma_lower = target_lemma.to_lowercase();
    let form = classify_surface_form(&surface_lower);
    let generated = generate(&lemma_lower, form);
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
    /// Third-person-singular present verb, or plural noun — both formed
    /// the same way in English ("-s"/"-es"/consonant-y -> "-ies").
    SuffixS,
    /// Gerund / present participle ("-ing").
    Ing,
    /// Past tense / past participle ("-ed", or irregular).
    Past,
}

/// Common irregular verbs whose third-person-singular, gerund, or past
/// form is not derivable by the regular suffix rules — `(base, 3sg, ing,
/// past)`. Regular verbs (even ones that look tricky, like "leave" ->
/// "leaves"/"leaving"/"left"... wait, "left" *is* irregular, so "leave" is
/// here too) are deliberately omitted: the regular rules already produce
/// their correct forms.
const IRREGULAR_VERBS: &[(&str, &str, &str, &str)] = &[
    ("be", "is", "being", "was"),
    ("have", "has", "having", "had"),
    ("do", "does", "doing", "did"),
    ("go", "goes", "going", "went"),
    ("make", "makes", "making", "made"),
    ("take", "takes", "taking", "took"),
    ("get", "gets", "getting", "got"),
    ("give", "gives", "giving", "gave"),
    ("know", "knows", "knowing", "knew"),
    ("think", "thinks", "thinking", "thought"),
    ("see", "sees", "seeing", "saw"),
    ("come", "comes", "coming", "came"),
    ("write", "writes", "writing", "wrote"),
    ("build", "builds", "building", "built"),
    ("find", "finds", "finding", "found"),
    ("leave", "leaves", "leaving", "left"),
    ("bring", "brings", "bringing", "brought"),
    ("keep", "keeps", "keeping", "kept"),
    ("begin", "begins", "beginning", "began"),
    ("run", "runs", "running", "ran"),
    ("buy", "buys", "buying", "bought"),
    ("catch", "catches", "catching", "caught"),
    ("choose", "chooses", "choosing", "chose"),
    ("drive", "drives", "driving", "drove"),
    ("feel", "feels", "feeling", "felt"),
    ("grow", "grows", "growing", "grew"),
    ("hold", "holds", "holding", "held"),
    ("lead", "leads", "leading", "led"),
    ("lose", "loses", "losing", "lost"),
    ("mean", "means", "meaning", "meant"),
    ("meet", "meets", "meeting", "met"),
    ("pay", "pays", "paying", "paid"),
    ("put", "puts", "putting", "put"),
    ("read", "reads", "reading", "read"),
    ("say", "says", "saying", "said"),
    ("sell", "sells", "selling", "sold"),
    ("send", "sends", "sending", "sent"),
    ("set", "sets", "setting", "set"),
    ("speak", "speaks", "speaking", "spoke"),
    ("spend", "spends", "spending", "spent"),
    ("stand", "stands", "standing", "stood"),
    ("teach", "teaches", "teaching", "taught"),
    ("tell", "tells", "telling", "told"),
    ("understand", "understands", "understanding", "understood"),
    ("win", "wins", "winning", "won"),
    ("become", "becomes", "becoming", "became"),
];

/// Common irregular noun plurals — `(singular, plural)`. Regular plurals
/// ("widget" -> "widgets") are handled by the suffix rules and do not need
/// an entry here.
const IRREGULAR_NOUNS: &[(&str, &str)] = &[
    ("person", "people"),
    ("child", "children"),
    ("man", "men"),
    ("woman", "women"),
    ("mouse", "mice"),
    ("goose", "geese"),
    ("foot", "feet"),
    ("tooth", "teeth"),
    ("criterion", "criteria"),
    ("phenomenon", "phenomena"),
    ("datum", "data"),
    ("analysis", "analyses"),
    ("index", "indices"),
];

/// Multi-syllable common words whose final consonant looks like it should
/// double under the naive consonant-vowel-consonant heuristic (see
/// [`should_double_final_consonant`]) but does not, because English
/// doubling depends on which syllable is stressed — information this
/// heuristic (deliberately, to stay dependency-free) does not have.
const NO_DOUBLE_EXCEPTIONS: &[&str] = &[
    "open", "happen", "offer", "listen", "enter", "visit", "focus", "profit", "target", "market",
    "differ", "cover", "answer", "gather", "wonder", "matter", "order", "honor", "favor", "labor",
    "color", "consider", "deliver", "suffer", "benefit", "exhibit", "edit", "credit", "limit",
    "orbit", "audit", "format", "budget", "signal", "panel", "cancel", "model", "label", "travel",
    "level", "total", "equal", "fuel",
];

fn classify_surface_form(surface_lower: &str) -> Form {
    classify_irregular(surface_lower).unwrap_or_else(|| classify_regular(surface_lower))
}

fn classify_irregular(word: &str) -> Option<Form> {
    for &(base, sg3, ing, past) in IRREGULAR_VERBS {
        if word == sg3 {
            return Some(Form::SuffixS);
        }
        if word == ing {
            return Some(Form::Ing);
        }
        if word == past {
            return Some(Form::Past);
        }
        if word == base {
            return Some(Form::Base);
        }
    }
    for &(singular, plural) in IRREGULAR_NOUNS {
        if word == plural {
            return Some(Form::SuffixS);
        }
        if word == singular {
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
    if IRREGULAR_NOUNS.iter().any(|&(_, plural)| plural == lemma) {
        // Already an irregular plural (or invariant form used directly as
        // a replacement lemma, e.g. "people"): nothing to add.
        return lemma.to_string();
    }
    if let Some(&(_, plural)) = IRREGULAR_NOUNS
        .iter()
        .find(|&&(singular, _)| singular == lemma)
    {
        return plural.to_string();
    }
    if let Some(&(_, sg3, _, _)) = IRREGULAR_VERBS.iter().find(|&&(base, ..)| base == lemma) {
        return sg3.to_string();
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
    if let Some(&(_, _, ing, _)) = IRREGULAR_VERBS.iter().find(|&&(base, ..)| base == lemma) {
        return ing.to_string();
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
    if let Some(&(_, _, _, past)) = IRREGULAR_VERBS.iter().find(|&&(base, ..)| base == lemma) {
        return past.to_string();
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
/// preceded by another consonant (or the word is only three letters), and
/// the word is not a known exception to the rule (see
/// [`NO_DOUBLE_EXCEPTIONS`]).
fn should_double_final_consonant(lemma: &str) -> bool {
    if NO_DOUBLE_EXCEPTIONS.contains(&lemma) {
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
    // "qu" is a single consonant unit (/kw/): the 'u' immediately after a
    // 'q' is not a second vowel of its own, so e.g. "quit" and "equip"
    // are still single-vowel CVC words for doubling purposes ("quitting",
    // "equipping"), not two-vowel VVC words like "boat" ("boating").
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
    /// "ying", consonant doubling, vowel-y kept), regular past "-ed"
    /// (silent-e, consonant-y, doubling), the common-verb irregulars
    /// table (surface irregular -> regular target, and regular surface ->
    /// irregular target), irregular noun plurals (both directions),
    /// capitalization transfer (Title Case, ALL CAPS) independent of
    /// target-lemma casing, uninflected base-form passthrough, and
    /// `None` for non-word input.
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
}
