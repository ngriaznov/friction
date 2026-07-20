//! Inflection-aware near-synonym substitution: swaps a closed table of
//! LLM-tell lemmas ("leverage", "utilize", "numerous", ...) for a plain
//! near-synonym, agreeing the replacement's surface form with the matched
//! word's own morphology (`"leverages"` -> `"uses"`, `"Leveraging"` ->
//! `"Using"`).
//!
//! # Curation and the meaning-preservation hard bar
//!
//! [`SUBSTITUTIONS`] starts from `corpus/MINING.md`'s train-split
//! llm-favored 1-grams (`crucial`, `robust`, `specific`, `necessary`,
//! `valuable`, `powerful`, `incredibly`, `approximately`, `initial`) plus
//! canonical LLM-register tells not specific to this corpus's topics
//! (`leverage`, `utilize`, `numerous`, `commence`, `facilitate`,
//! `subsequently`, and similar Latinate near-synonyms of a plainer word).
//! Matching here is pure whole-word lookup (see "Matching" below) — it
//! never looks at a matched word's actual grammatical role in its
//! sentence, only its spelling — so every entry must be a *true*
//! near-synonym *for every ordinary use of that spelling*, not just the
//! sense the curator had in mind: swapping it must change register, never
//! meaning, no matter which of a word's senses or parts of speech actually
//! matched. A candidate is left out — noted here rather than silently
//! dropped — whenever that bar was not clearly met:
//!
//! - **`ensure`** (the single highest-z llm-favored unigram in the mined
//!   report): no clean single-word synonym preserves its "make certain"
//!   sense without drifting toward "guarantee" (implies a promise) or
//!   "confirm" (implies checking something already true).
//! - **`potential`**, **`process`**: commonly used as a noun or verb in
//!   this corpus in ways a single adjective/verb target could not agree
//!   with grammatically for every occurrence.
//! - **`efficient`**, **`challenges`**, **`understanding`** (as a noun),
//!   **`overall`**, **`myriad`**, **`additional`**: a candidate target
//!   either shifted connotation (`efficient` != `effective`; `challenges`
//!   != `problems`), collided with `understanding`'s dual noun/gerund
//!   shape, or (`overall`) belongs to the *connective* family, not lexical
//!   substitution.
//! - **`individual`** (-> a candidate `person`): reads just as often as a
//!   plain adjective ("individual liability") as it does as the intended
//!   noun ("an individual"), and matching does not distinguish the two —
//!   an adjectival hit produces an ungrammatical sentence, and in
//!   legal/contract prose `"individual"` is frequently a defined term of
//!   art specifically narrower than `"person"` (which usually also covers
//!   corporations and other entities), so the swap can silently widen a
//!   defined term.
//! - **`benefit`** (-> a candidate `advantage`): at least as common as a
//!   verb ("this will benefit users") as it is as the noun the entry was
//!   written for, and `"advantage"` is not a natural verb in modern usage.
//! - **`prioritize`** (-> a candidate `rank`): ordinarily means "treat as
//!   most important," not "put in a sorted order" — `"rank"` requires and
//!   implies a comparison set the source sentence usually never had, so
//!   the swap changes what action is being asserted, not just its
//!   register.
//! - **`significant`**/**`significantly`** (-> candidates `big`/
//!   `greatly`): in technical/scientific prose these words usually carry
//!   the statistical-significance sense (unlikely to be due to chance),
//!   which is orthogonal to effect size — swapping in a magnitude word
//!   asserts something about size the source never claimed.
//! - **`perpetual`** (-> a candidate `constant`): a term of art in
//!   legal/licensing prose meaning unlimited in duration ("a perpetual
//!   license"), a durational claim `"constant"` (steady/unchanging) does
//!   not carry at all.
//! - **`underscore`** (-> a candidate `stress`, for its emphasize-verb
//!   sense): at least as common as the ordinary noun for the `_`
//!   punctuation character, especially in the docs/readme prose this rule
//!   targets ("separate words with an underscore") — the noun use has
//!   nothing to do with emphasis.
//! - Domain nouns that are topic artifacts of this corpus's specific
//!   prompt set rather than general register (`database`, `backup`,
//!   `cron`, `postgres`, `mysql`, `sqs`, `settings`, `instances`,
//!   `configuration`, `migration`, `environment`, `monitoring`, `tool`,
//!   `directory`, `script`, `query`, `table`, `index`, `commit`, `batch`,
//!   `queue`, `step`/`steps`, `job`, `file`, `date`): the same "topic
//!   artifact, not register" judgment
//!   `friction-packs/packs/mined-ngrams-v1.toml`'s own curation notes make
//!   for the metrics layer's mined-phrase pack.
//!
//! # Matching: reusing `inflect` for both directions
//!
//! [`friction_nlp::inflect`] takes a *surface word* and a *target lemma*
//! and produces the target's form that agrees with the surface word's own
//! morphology — the direction needed to build a replacement. Recognizing
//! whether a word found in the document is an inflected form of one of
//! [`SUBSTITUTIONS`]' lemmas needs the same suffix rules in reverse
//! (generate a lemma's own candidate surface forms, then compare). Rather
//! than duplicating `inflect`'s silent-e/consonant-y/consonant-doubling
//! logic — a real risk of the two directions quietly disagreeing —
//! [`surface_forms`] reuses `inflect` itself: it applies a lemma to fixed,
//! unambiguously-regular templates (`"uses"`, `"using"`, `"used"`, all
//! forms of the plain regular verb "use"), so the exact same code path
//! that will later generate the replacement also generates the candidate
//! match forms. This also means a replacement's own inflected forms
//! automatically pick up `inflect`'s irregular-verb and irregular-noun
//! tables where relevant (e.g. matched `"acquired"` correctly replaces
//! with `"got"`, the irregular past of the `"acquire"` entry's target
//! lemma `"get"`, not a fictional regular `"geted"`).
//!
//! ## Why each entry is tagged with a [`LemmaClass`]
//!
//! Mechanically generating an "-s" form for *every* lemma, regardless of
//! its real part of speech, is not just wasted work for an adjective —
//! it is a genuine false-positive risk. `"valuable"` (adjective) plus an
//! -s suffix produces `"valuables"`, which is not "more than one valuable
//! thing" but an *established, differently-meaning* English noun
//! ("please store your valuables in the safe"); `"vital"` + "-s" likewise
//! collides with the medical noun "vitals" ("check the patient's
//! vitals"), and `"initial"` + "-s" collides with "initials" (a name's
//! abbreviated letters). None of these are what this rule means to match.
//! [`LemmaClass`] fixes this at the root: [`surface_forms`] only derives
//! the forms a lemma's *actual* class can legitimately take —
//! [`LemmaClass::Verb`] gets all three templates (present-tense/plural,
//! gerund, past), [`LemmaClass::Noun`] gets only the plural template, and
//! [`LemmaClass::Adjective`] (covering adverbs too) gets none at all,
//! matching only its own unmodified base form — so a table entry can
//! never generate a candidate match form it was never meant to (checked
//! directly by this module's `adjective_forms_do_not_collide_with_
//! unrelated_words` regression test).
//!
//! # Table closure
//!
//! No [`SUBSTITUTIONS`] replacement is itself a table lemma, and — more
//! strongly than a plain lemma-string check — no replacement is even
//! equal to any lemma's *generated surface form* (any tense/number
//! variant), verified directly against the real matching machinery by
//! this module's `substitution_table_is_closed_across_all_generated_
//! surface_forms` test. That is what guarantees idempotence: applying a
//! fix can never hand a later scan a fresh match.
//!
//! # A known simplification: indefinite articles
//!
//! This rule replaces exactly the matched word, never the article in
//! front of it. A vowel-initial lemma preceded by `"an"` and replaced with
//! a consonant-initial target (or vice versa) can leave a grammatically
//! awkward `"an person"` — unfortunate but not a meaning change (the
//! sentence's asserted proposition is unaffected), and out of scope for a
//! single-token lexical substitution; fixing the surrounding article would
//! belong to a grammar-repair pass, not this rule.

use std::collections::BTreeMap;
use std::sync::LazyLock;

use friction_core::{Finding, MetricVector, Patch, RuleId, Tier};
use friction_nlp::inflect;

use crate::budget::Budget;
use crate::context::{GenreEnvelope, RuleContext};
use crate::rule::{Gate, Rule, RuleFamily};
use crate::strategy::StrategyRng;

/// This rule's stable identifier.
const RULE_ID: RuleId = RuleId::new("lexical.substitution");

/// The [`MetricVector`] field this rule gates on: the rate of curated
/// llm-favored mined n-grams (`friction-packs/packs/mined-ngrams-v1.
/// toml`), per 1000 word tokens — the metrics-layer counterpart of this
/// rule's own hand-curated lemma table.
const GATED_METRIC: &str = "llm_favored_phrase_rate";

/// How much fixing one occurrence is projected to move [`GATED_METRIC`].
/// See `crate::families::lexical::filler::PER_FIX_EFFECT`'s docs for why
/// `1.0` (one point of the metric's own per-1000-token scale) is the
/// right dimensionless unit here too: a [`MetricVector`] carries no raw
/// token count, so this is a deliberately conservative estimate that a
/// later round's fresh `gate` call tops up if still needed.
const PER_FIX_EFFECT: f64 = 1.0;

/// A [`SubstitutionEntry`]'s real part of speech, and therefore which
/// inflected forms [`surface_forms`] may legitimately derive for it — see
/// the module docs' "Why each entry is tagged" section for why this
/// matters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LemmaClass {
    /// A regular verb: matches its base, third-person-singular/plural
    /// (-s/-es), gerund (-ing), and past (-ed) forms.
    Verb,
    /// A noun: matches its base and plural (-s/-es) forms only.
    Noun,
    /// An adjective or adverb: matches only its own unmodified base form.
    Adjective,
}

/// One [`SUBSTITUTIONS`] entry: a base lemma to match, its real part of
/// speech, the plain near-synonym lemma to replace it with, and a short
/// plain-language rationale (also surfaced in this rule's [`Finding`]
/// messages).
struct SubstitutionEntry {
    lemma: &'static str,
    class: LemmaClass,
    replacement: &'static str,
    note: &'static str,
}

/// Closed lexical substitution table, sorted alphabetically by `lemma`
/// (ASCII byte order, checked by this module's
/// `substitutions_sorted_and_unique_lemmas` test). See the module docs for
/// curation source, the meaning-preservation bar, and what was
/// deliberately left out.
const SUBSTITUTIONS: &[SubstitutionEntry] = &[
    SubstitutionEntry {
        lemma: "accomplish",
        class: LemmaClass::Verb,
        replacement: "do",
        note: "a plain, shorter verb for the same act",
    },
    SubstitutionEntry {
        lemma: "acquire",
        class: LemmaClass::Verb,
        replacement: "get",
        note: "acquire is a formal register synonym of get",
    },
    SubstitutionEntry {
        lemma: "ameliorate",
        class: LemmaClass::Verb,
        replacement: "improve",
        note: "ameliorate is a rarer, more Latinate synonym of improve",
    },
    SubstitutionEntry {
        lemma: "approximately",
        class: LemmaClass::Adjective,
        replacement: "about",
        note: "mined llm-favored (MINING.md); about is the everyday equivalent",
    },
    SubstitutionEntry {
        lemma: "ascertain",
        class: LemmaClass::Verb,
        replacement: "learn",
        note: "ascertain (find out for certain) is a formal synonym of learn",
    },
    SubstitutionEntry {
        lemma: "assist",
        class: LemmaClass::Verb,
        replacement: "help",
        note: "assist is a formal register synonym of help",
    },
    SubstitutionEntry {
        lemma: "attain",
        class: LemmaClass::Verb,
        replacement: "reach",
        note: "attain is a formal register synonym of reach",
    },
    SubstitutionEntry {
        lemma: "augment",
        class: LemmaClass::Verb,
        replacement: "increase",
        note: "augment is a formal register synonym of increase",
    },
    SubstitutionEntry {
        lemma: "bolster",
        class: LemmaClass::Verb,
        replacement: "support",
        note: "bolster (strengthen existing support) is a synonym of support",
    },
    SubstitutionEntry {
        lemma: "commence",
        class: LemmaClass::Verb,
        replacement: "start",
        note: "canonical LLM-register tell for start",
    },
    SubstitutionEntry {
        lemma: "comprehend",
        class: LemmaClass::Verb,
        replacement: "understand",
        note: "comprehend is a more Latinate synonym of understand",
    },
    SubstitutionEntry {
        lemma: "crucial",
        class: LemmaClass::Adjective,
        replacement: "key",
        note: "mined llm-favored (MINING.md, z=6.47); key keeps the emphasis, drops the register",
    },
    SubstitutionEntry {
        lemma: "demonstrate",
        class: LemmaClass::Verb,
        replacement: "show",
        note: "demonstrate is a more formal synonym of show",
    },
    SubstitutionEntry {
        lemma: "eliminate",
        class: LemmaClass::Verb,
        replacement: "remove",
        note: "eliminate is a more formal synonym of remove",
    },
    SubstitutionEntry {
        lemma: "elucidate",
        class: LemmaClass::Verb,
        replacement: "explain",
        note: "elucidate is a rarer, more Latinate synonym of explain",
    },
    SubstitutionEntry {
        lemma: "embark",
        class: LemmaClass::Verb,
        replacement: "start",
        note: "embark (on/upon) is a more formal synonym of start",
    },
    SubstitutionEntry {
        lemma: "encompass",
        class: LemmaClass::Verb,
        replacement: "include",
        note: "encompass is a more formal synonym of include",
    },
    SubstitutionEntry {
        lemma: "endeavor",
        class: LemmaClass::Verb,
        replacement: "try",
        note: "endeavor is a more formal synonym of try",
    },
    SubstitutionEntry {
        lemma: "enhance",
        class: LemmaClass::Verb,
        replacement: "improve",
        note: "enhance is a more formal synonym of improve",
    },
    SubstitutionEntry {
        lemma: "exemplify",
        class: LemmaClass::Verb,
        replacement: "show",
        note: "exemplify is a more formal synonym of show (by example)",
    },
    SubstitutionEntry {
        lemma: "facilitate",
        class: LemmaClass::Verb,
        replacement: "help",
        note: "canonical LLM-register tell for help",
    },
    SubstitutionEntry {
        lemma: "finalize",
        class: LemmaClass::Verb,
        replacement: "finish",
        note: "finalize is a more formal synonym of finish",
    },
    SubstitutionEntry {
        lemma: "foster",
        class: LemmaClass::Verb,
        replacement: "encourage",
        note: "foster is a more formal synonym of encourage",
    },
    SubstitutionEntry {
        lemma: "fundamental",
        class: LemmaClass::Adjective,
        replacement: "basic",
        note: "fundamental is a more formal synonym of basic",
    },
    SubstitutionEntry {
        lemma: "garner",
        class: LemmaClass::Verb,
        replacement: "earn",
        note: "garner is a more formal synonym of earn",
    },
    SubstitutionEntry {
        lemma: "illustrate",
        class: LemmaClass::Verb,
        replacement: "show",
        note: "illustrate is a more formal synonym of show",
    },
    SubstitutionEntry {
        lemma: "incredibly",
        class: LemmaClass::Adjective,
        replacement: "very",
        note: "mined llm-favored (MINING.md); very is the plain intensifier",
    },
    SubstitutionEntry {
        lemma: "initial",
        class: LemmaClass::Adjective,
        replacement: "first",
        note: "mined llm-favored (MINING.md); first is the plain equivalent",
    },
    SubstitutionEntry {
        lemma: "initiate",
        class: LemmaClass::Verb,
        replacement: "start",
        note: "initiate is a more formal synonym of start",
    },
    SubstitutionEntry {
        lemma: "leverage",
        class: LemmaClass::Verb,
        replacement: "use",
        note: "canonical LLM-register tell for use",
    },
    SubstitutionEntry {
        lemma: "locate",
        class: LemmaClass::Verb,
        replacement: "find",
        note: "locate is a more formal synonym of find",
    },
    SubstitutionEntry {
        lemma: "mandatory",
        class: LemmaClass::Adjective,
        replacement: "required",
        note: "mandatory is a more formal synonym of required",
    },
    SubstitutionEntry {
        lemma: "modification",
        class: LemmaClass::Noun,
        replacement: "change",
        note: "modification is a more formal synonym of change",
    },
    SubstitutionEntry {
        lemma: "necessary",
        class: LemmaClass::Adjective,
        replacement: "needed",
        note: "mined llm-favored (MINING.md); needed is the plain equivalent",
    },
    SubstitutionEntry {
        lemma: "necessitate",
        class: LemmaClass::Verb,
        replacement: "require",
        note: "necessitate is a more formal synonym of require",
    },
    SubstitutionEntry {
        lemma: "numerous",
        class: LemmaClass::Adjective,
        replacement: "many",
        note: "canonical LLM-register tell for many",
    },
    SubstitutionEntry {
        lemma: "obtain",
        class: LemmaClass::Verb,
        replacement: "get",
        note: "obtain is a more formal synonym of get",
    },
    SubstitutionEntry {
        lemma: "optimal",
        class: LemmaClass::Adjective,
        replacement: "ideal",
        note: "optimal and ideal both take a/an; best would force a superlative-only register",
    },
    SubstitutionEntry {
        lemma: "optimize",
        class: LemmaClass::Verb,
        replacement: "improve",
        note: "optimize is a more formal synonym of improve",
    },
    SubstitutionEntry {
        lemma: "paramount",
        class: LemmaClass::Adjective,
        replacement: "key",
        note: "paramount is a more formal synonym of key",
    },
    SubstitutionEntry {
        lemma: "participate",
        class: LemmaClass::Verb,
        replacement: "join",
        note: "participate is a more formal synonym of join",
    },
    SubstitutionEntry {
        lemma: "pivotal",
        class: LemmaClass::Adjective,
        replacement: "key",
        note: "pivotal is a more formal synonym of key",
    },
    SubstitutionEntry {
        lemma: "plethora",
        class: LemmaClass::Noun,
        replacement: "range",
        note: "range keeps 'a ... of' grammatical, unlike a vowel-initial target",
    },
    SubstitutionEntry {
        lemma: "powerful",
        class: LemmaClass::Adjective,
        replacement: "strong",
        note: "mined llm-favored (MINING.md); strong is the plain equivalent",
    },
    SubstitutionEntry {
        lemma: "profound",
        class: LemmaClass::Adjective,
        replacement: "deep",
        note: "profound is a more formal synonym of deep",
    },
    SubstitutionEntry {
        lemma: "robust",
        class: LemmaClass::Adjective,
        replacement: "solid",
        note: "mined llm-favored (MINING.md, z=6.29); solid is the plain equivalent",
    },
    SubstitutionEntry {
        lemma: "showcase",
        class: LemmaClass::Verb,
        replacement: "show",
        note: "showcase is a more formal synonym of show",
    },
    SubstitutionEntry {
        lemma: "specific",
        class: LemmaClass::Adjective,
        replacement: "particular",
        note: "mined llm-favored (MINING.md); particular is a plainer near-synonym",
    },
    SubstitutionEntry {
        lemma: "strive",
        class: LemmaClass::Verb,
        replacement: "try",
        note: "strive is a more formal synonym of try",
    },
    SubstitutionEntry {
        lemma: "subsequently",
        class: LemmaClass::Adjective,
        replacement: "later",
        note: "canonical LLM-register tell for later",
    },
    SubstitutionEntry {
        lemma: "substantial",
        class: LemmaClass::Adjective,
        replacement: "large",
        note: "substantial is a more formal synonym of large",
    },
    SubstitutionEntry {
        lemma: "tremendous",
        class: LemmaClass::Adjective,
        replacement: "huge",
        note: "tremendous is a more formal synonym of huge",
    },
    SubstitutionEntry {
        lemma: "underpin",
        class: LemmaClass::Verb,
        replacement: "support",
        note: "underpin (be the foundation of) is a synonym of support",
    },
    SubstitutionEntry {
        lemma: "utilization",
        class: LemmaClass::Noun,
        replacement: "use",
        note: "canonical LLM-register tell for use (noun form)",
    },
    SubstitutionEntry {
        lemma: "utilize",
        class: LemmaClass::Verb,
        replacement: "use",
        note: "canonical LLM-register tell for use",
    },
    SubstitutionEntry {
        lemma: "valuable",
        class: LemmaClass::Adjective,
        replacement: "useful",
        note: "mined llm-favored (MINING.md); useful is the plain equivalent",
    },
    SubstitutionEntry {
        lemma: "vital",
        class: LemmaClass::Adjective,
        replacement: "key",
        note: "vital is a more formal synonym of key",
    },
];

/// Fixed, unambiguously-regular templates (all forms of the plain regular
/// verb "use") used only to derive a lemma's own candidate surface forms
/// via `inflect` — see the module docs' "Matching" section.
///
/// Order matters for [`LemmaClass::Noun`], which uses only
/// `FORM_TEMPLATES[0]` (the plural/third-person-singular template) — see
/// [`surface_forms`].
const FORM_TEMPLATES: [&str; 3] = ["uses", "using", "used"];

/// The candidate lowercase surface forms `lemma` can take, restricted to
/// what `class` can legitimately inflect to (see the module docs' "Why
/// each entry is tagged" section): always includes `lemma`'s own base
/// form; [`LemmaClass::Noun`] additionally includes the plural form;
/// [`LemmaClass::Verb`] additionally includes all of plural/
/// third-person-singular, gerund, and past. Deduplicated.
fn surface_forms(lemma: &str, class: LemmaClass) -> Vec<String> {
    let templates: &[&str] = match class {
        LemmaClass::Adjective => &[],
        LemmaClass::Noun => &FORM_TEMPLATES[..1],
        LemmaClass::Verb => &FORM_TEMPLATES,
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

/// Every [`SUBSTITUTIONS`] entry's [`surface_forms`], indexed by position
/// in [`SUBSTITUTIONS`] — computed once, lazily, since deriving them calls
/// `inflect` and this table is reused across every sentence a document has.
static SUBSTITUTION_FORMS: LazyLock<Vec<Vec<String>>> = LazyLock::new(|| {
    SUBSTITUTIONS
        .iter()
        .map(|entry| surface_forms(entry.lemma, entry.class))
        .collect()
});

/// Every generated surface form (see [`SUBSTITUTION_FORMS`]) mapped to the
/// [`SUBSTITUTIONS`] index it came from, for `O(log n)` lookup during
/// `scan`/`fix`. If two entries' forms ever collided, the lower index
/// (earlier, alphabetically-first lemma) wins deterministically — checked
/// not to matter in practice by this module's `substitution_table_is_
/// closed_across_all_generated_surface_forms` test, which asserts none of
/// these keys ever equals a table replacement.
static SUBSTITUTION_FORM_INDEX: LazyLock<BTreeMap<String, usize>> = LazyLock::new(|| {
    let mut map = BTreeMap::new();
    for (index, forms) in SUBSTITUTION_FORMS.iter().enumerate() {
        for form in forms {
            map.entry(form.clone()).or_insert(index);
        }
    }
    map
});

/// Splits `text` into `(byte range, lowercase text)` pairs for each
/// maximal run of alphabetic characters — the same word-span shape
/// `friction-metrics`' own tokenizers use, kept local here since this
/// rule additionally needs each word's byte range (not just its text) to
/// build a [`Finding`]/[`Patch`].
fn word_spans(text: &str) -> Vec<(std::ops::Range<usize>, String)> {
    let mut spans = Vec::new();
    let mut start: Option<usize> = None;
    for (i, c) in text.char_indices() {
        if c.is_alphabetic() {
            start.get_or_insert(i);
        } else if let Some(s) = start.take() {
            spans.push((s..i, text[s..i].to_ascii_lowercase()));
        }
    }
    if let Some(s) = start {
        spans.push((s..text.len(), text[s..].to_ascii_lowercase()));
    }
    spans
}

/// Inflection-aware near-synonym substitution.
///
/// Replaces each matched [`SUBSTITUTIONS`] lemma occurrence with its
/// designated near-synonym, budgeted to bring [`GATED_METRIC`] back into
/// the genre's envelope. See the module docs for the matching strategy and
/// the closure invariant.
#[derive(Debug, Clone, Copy, Default)]
pub struct SubstitutionRule;

impl SubstitutionRule {
    /// Creates the rule.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Rule for SubstitutionRule {
    fn id(&self) -> RuleId {
        RULE_ID
    }

    fn family(&self) -> RuleFamily {
        RuleFamily::Lexical
    }

    fn gate(&self, metrics: &MetricVector, envelope: &dyn GenreEnvelope) -> Gate {
        let Some(band) = envelope.band(GATED_METRIC) else {
            return Gate::Off;
        };
        let current = metrics.llm_favored_phrase_rate;
        // This rule only ever substitutes away from an llm-favored lemma,
        // never toward one, so it has no safe move for a document already
        // inside the band or (unusually) below its floor.
        if current <= band.hi {
            return Gate::Off;
        }
        let budget = Budget::from_envelope_excess(current, band, PER_FIX_EFFECT);
        if budget.is_exhausted() {
            Gate::Off
        } else {
            Gate::Fix { budget }
        }
    }

    fn scan(&self, ctx: &RuleContext<'_>) -> Vec<Finding> {
        let document = ctx.document();
        let mut findings = Vec::new();
        for (_, sentence) in ctx.sentences() {
            let Ok(text) = document.text(&sentence.range) else {
                continue;
            };
            for (relative, word_lower) in word_spans(text) {
                let Some(&entry_index) = SUBSTITUTION_FORM_INDEX.get(&word_lower) else {
                    continue;
                };
                let entry = &SUBSTITUTIONS[entry_index];
                let start = sentence.range.start + relative.start;
                let end = sentence.range.start + relative.end;
                findings.push(Finding::new(
                    RULE_ID,
                    start..end,
                    format!(
                        "\"{}\" reads as machine-favored register ({}); \"{}\" is a plain \
                         near-synonym",
                        &text[relative.clone()],
                        entry.note,
                        entry.replacement
                    ),
                    Tier::Fix,
                ));
            }
        }
        findings
    }

    fn fix(
        &self,
        finding: &Finding,
        ctx: &RuleContext<'_>,
        _strategy_rng: &mut StrategyRng,
    ) -> Option<Patch> {
        let source = ctx.document().source();
        let surface = &source[finding.range.clone()];
        let word_lower = surface.to_ascii_lowercase();
        let entry_index = *SUBSTITUTION_FORM_INDEX.get(&word_lower)?;
        let entry = &SUBSTITUTIONS[entry_index];
        let replacement = inflect(surface, entry.replacement)?;
        Some(Patch::new(
            finding.range.clone(),
            replacement,
            RULE_ID,
            Tier::Fix,
        ))
    }
}

#[cfg(test)]
mod tests {
    use friction_core::Envelope;
    use friction_nlp::{SrxSegmenter, Tagger};

    use super::*;
    use crate::context::MapEnvelope;

    struct NoopTagger;
    impl Tagger for NoopTagger {
        fn tag(&self, _text: &str, _base_offset: usize) -> Vec<friction_nlp::TaggedToken> {
            Vec::new()
        }
    }

    fn document(source: &str) -> friction_core::Document {
        let parsed = friction_parse::parse(source).expect("valid markdown parses");
        friction_nlp::segment_document(&parsed, &SrxSegmenter::new())
            .expect("segmentation succeeds")
    }

    fn metrics_with_rate(rate: f64) -> MetricVector {
        MetricVector {
            llm_favored_phrase_rate: rate,
            ..MetricVector::default()
        }
    }

    fn apply(source: &str, patch: &Patch) -> String {
        let mut applied = source.to_string();
        applied.replace_range(patch.range.clone(), &patch.replacement);
        applied
    }

    // ---------------------------------------------------------------
    // Table hygiene / closure
    // ---------------------------------------------------------------

    #[test]
    fn substitutions_has_at_least_fifty_entries() {
        assert!(
            SUBSTITUTIONS.len() >= 50,
            "expected at least 50 curated substitution entries, has {}",
            SUBSTITUTIONS.len()
        );
    }

    #[test]
    fn substitutions_sorted_and_unique_lemmas() {
        assert!(SUBSTITUTIONS.windows(2).all(|w| w[0].lemma < w[1].lemma));
    }

    /// The closed-table invariant this workspace requires: no replacement
    /// is itself a table lemma.
    #[test]
    fn substitution_table_is_closed_no_replacement_is_a_lemma() {
        let lemmas: std::collections::BTreeSet<&str> =
            SUBSTITUTIONS.iter().map(|e| e.lemma).collect();
        for entry in SUBSTITUTIONS {
            assert!(
                !lemmas.contains(entry.replacement),
                "{} -> {}: a replacement must never itself be a table lemma",
                entry.lemma,
                entry.replacement
            );
        }
    }

    /// The stronger version of the closure check, run against the real
    /// matching machinery: no *generated surface form* of any lemma (not
    /// just the bare lemma string) ever equals any table replacement.
    /// This is what actually guarantees idempotence, since `scan` matches
    /// against every generated form, not only the base lemma.
    #[test]
    fn substitution_table_is_closed_across_all_generated_surface_forms() {
        let replacements: std::collections::BTreeSet<&str> =
            SUBSTITUTIONS.iter().map(|e| e.replacement).collect();
        for form in SUBSTITUTION_FORM_INDEX.keys() {
            assert!(
                !replacements.contains(form.as_str()),
                "generated surface form {form:?} collides with a table replacement -- \
                 idempotence at risk"
            );
        }
    }

    // ---------------------------------------------------------------
    // surface_forms / inflect wiring
    // ---------------------------------------------------------------

    #[test]
    fn surface_forms_covers_leverage_verb_inflections() {
        let forms = surface_forms("leverage", LemmaClass::Verb);
        for expected in ["leverage", "leverages", "leveraging", "leveraged"] {
            assert!(
                forms.contains(&expected.to_string()),
                "missing {expected:?} in {forms:?}"
            );
        }
    }

    #[test]
    fn surface_forms_noun_class_includes_plural_only() {
        let forms = surface_forms("individual", LemmaClass::Noun);
        assert!(forms.contains(&"individual".to_string()));
        assert!(forms.contains(&"individuals".to_string()));
        assert_eq!(
            forms.len(),
            2,
            "a noun should have exactly base + plural: {forms:?}"
        );
    }

    /// The false-positive fix this module's docs describe: an
    /// [`LemmaClass::Adjective`] entry matches *only* its own base form,
    /// never a mechanically-generated "-s" form — which is exactly what
    /// prevents `"valuable"`/`"vital"`/`"initial"` from spuriously
    /// matching the unrelated real words `"valuables"`/`"vitals"`/
    /// `"initials"`.
    #[test]
    fn adjective_forms_do_not_collide_with_unrelated_words() {
        let forms = surface_forms("valuable", LemmaClass::Adjective);
        assert_eq!(forms, vec!["valuable".to_string()]);

        for (source, unrelated_word) in [
            (
                "Please store your valuables in the hotel safe.",
                "valuables",
            ),
            (
                "The nurse checked the patient's vitals every hour.",
                "vitals",
            ),
            (
                "Please write your initials at the bottom of the form.",
                "initials",
            ),
        ] {
            let doc = document(source);
            let envelope = MapEnvelope::new();
            let ctx = RuleContext::new(&doc, &NoopTagger, "blog", &envelope);
            let findings = SubstitutionRule::new().scan(&ctx);
            assert!(
                findings.is_empty(),
                "{unrelated_word:?} in {source:?} must not match any SUBSTITUTIONS entry, got \
                 {findings:?}"
            );
        }
    }

    // ---------------------------------------------------------------
    // gate()
    // ---------------------------------------------------------------

    #[test]
    fn gate_is_off_without_a_band() {
        let rule = SubstitutionRule::new();
        let envelope = MapEnvelope::new();
        assert_eq!(rule.gate(&metrics_with_rate(500.0), &envelope), Gate::Off);
    }

    #[test]
    fn gate_is_off_inside_band() {
        let rule = SubstitutionRule::new();
        let envelope = MapEnvelope::new().with(GATED_METRIC, Envelope::new(0.0, 10.0));
        assert_eq!(rule.gate(&metrics_with_rate(5.0), &envelope), Gate::Off);
    }

    /// Hand-computed: current 13.0, band hi 10.0, `PER_FIX_EFFECT` 1.0 ->
    /// excess 3.0 -> budget 3.
    #[test]
    fn gate_above_band_computes_hand_verified_budget() {
        let rule = SubstitutionRule::new();
        let envelope = MapEnvelope::new().with(GATED_METRIC, Envelope::new(0.0, 10.0));
        assert_eq!(
            rule.gate(&metrics_with_rate(13.0), &envelope),
            Gate::Fix {
                budget: Budget::new(3)
            }
        );
    }

    // ---------------------------------------------------------------
    // scan() / fix()
    // ---------------------------------------------------------------

    #[test]
    fn scan_finds_every_occurrence_case_insensitively() {
        let source = "The team will leverage the new API. LEVERAGING it further is planned.";
        let doc = document(source);
        let envelope = MapEnvelope::new();
        let ctx = RuleContext::new(&doc, &NoopTagger, "blog", &envelope);
        let findings = SubstitutionRule::new().scan(&ctx);
        assert_eq!(findings.len(), 2);
        assert_eq!(&source[findings[0].range.clone()], "leverage");
        assert_eq!(&source[findings[1].range.clone()], "LEVERAGING");
    }

    /// A lemma must match a *whole* word span, never a substring of a
    /// longer, unrelated word: `"commencement"` does not match even
    /// though it contains `"commence"`, but the standalone word
    /// `"commence"` right after it does.
    #[test]
    fn scan_does_not_match_a_lemma_as_a_substring_of_a_longer_word() {
        let source = "We announced the commencement date, not when to commence exactly.";
        let doc = document(source);
        let envelope = MapEnvelope::new();
        let ctx = RuleContext::new(&doc, &NoopTagger, "blog", &envelope);
        let findings = SubstitutionRule::new().scan(&ctx);
        assert_eq!(findings.len(), 1);
        assert_eq!(&source[findings[0].range.clone()], "commence");
    }

    /// A document containing none of [`SUBSTITUTIONS`]' words produces no
    /// findings at all.
    #[test]
    fn scan_finds_nothing_in_a_clean_document() {
        let source = "The cat sat on the mat and slept until noon.";
        let doc = document(source);
        let envelope = MapEnvelope::new();
        let ctx = RuleContext::new(&doc, &NoopTagger, "blog", &envelope);
        assert!(SubstitutionRule::new().scan(&ctx).is_empty());
    }

    #[test]
    fn fix_agrees_surface_form_leverages_to_uses() {
        let source = "The pipeline leverages a shared cache.";
        let doc = document(source);
        let envelope = MapEnvelope::new();
        let ctx = RuleContext::new(&doc, &NoopTagger, "blog", &envelope);
        let rule = SubstitutionRule::new();
        let finding = &rule.scan(&ctx)[0];
        let mut rng = StrategyRng::from_seed(0);
        let patch = rule
            .fix(finding, &ctx, &mut rng)
            .expect("finding has a fix");
        assert_eq!(apply(source, &patch), "The pipeline uses a shared cache.");
    }

    #[test]
    fn fix_agrees_surface_form_leveraging_to_using() {
        let source = "Leveraging the cache speeds things up.";
        let doc = document(source);
        let envelope = MapEnvelope::new();
        let ctx = RuleContext::new(&doc, &NoopTagger, "blog", &envelope);
        let rule = SubstitutionRule::new();
        let finding = &rule.scan(&ctx)[0];
        let mut rng = StrategyRng::from_seed(0);
        let patch = rule
            .fix(finding, &ctx, &mut rng)
            .expect("finding has a fix");
        assert_eq!(apply(source, &patch), "Using the cache speeds things up.");
    }

    /// The replacement's own morphology can land on an irregular form: the
    /// `"acquire"` -> `"get"` entry's past-tense replacement is `"got"`
    /// (irregular), never a fictional regular `"geted"`.
    #[test]
    fn fix_verb_replacement_uses_irregular_past_tense() {
        let source = "The team acquired new hardware last quarter.";
        let doc = document(source);
        let envelope = MapEnvelope::new();
        let ctx = RuleContext::new(&doc, &NoopTagger, "blog", &envelope);
        let rule = SubstitutionRule::new();
        let finding = &rule.scan(&ctx)[0];
        let mut rng = StrategyRng::from_seed(0);
        let patch = rule
            .fix(finding, &ctx, &mut rng)
            .expect("finding has a fix");
        assert_eq!(
            apply(source, &patch),
            "The team got new hardware last quarter."
        );
    }

    #[test]
    fn fix_pluralizes_via_regular_noun() {
        let source = "Numerous modifications were made to the proposal.";
        let doc = document(source);
        let envelope = MapEnvelope::new();
        let ctx = RuleContext::new(&doc, &NoopTagger, "blog", &envelope);
        let rule = SubstitutionRule::new();
        let findings = rule.scan(&ctx);
        assert_eq!(findings.len(), 2);

        let mut fixed = source.to_string();
        let mut patches: Vec<Patch> = findings
            .iter()
            .map(|f| {
                let mut rng = StrategyRng::from_seed(0);
                rule.fix(f, &ctx, &mut rng).expect("finding has a fix")
            })
            .collect();
        patches.sort_by_key(|p| p.range.start);
        for patch in patches.iter().rev() {
            fixed.replace_range(patch.range.clone(), &patch.replacement);
        }
        assert_eq!(fixed, "Many changes were made to the proposal.");
    }

    #[test]
    fn fix_utilization_maps_to_use_as_a_noun() {
        let source = "Utilization of the tool grew steadily.";
        let doc = document(source);
        let envelope = MapEnvelope::new();
        let ctx = RuleContext::new(&doc, &NoopTagger, "blog", &envelope);
        let rule = SubstitutionRule::new();
        let finding = &rule.scan(&ctx)[0];
        let mut rng = StrategyRng::from_seed(0);
        let patch = rule
            .fix(finding, &ctx, &mut rng)
            .expect("finding has a fix");
        assert_eq!(apply(source, &patch), "Use of the tool grew steadily.");
    }

    // ---------------------------------------------------------------
    // Idempotence and determinism
    // ---------------------------------------------------------------

    #[test]
    fn fix_is_idempotent_on_synthetic_sentences() {
        let sources = [
            "The pipeline leverages a shared cache.",
            "Leveraging the cache speeds things up.",
            "Numerous modifications were reviewed for the proposal.",
            "This is a crucial and robust solution for our specific needs.",
        ];
        for source in sources {
            let doc = document(source);
            let envelope = MapEnvelope::new();
            let ctx = RuleContext::new(&doc, &NoopTagger, "blog", &envelope);
            let rule = SubstitutionRule::new();

            let mut patches: Vec<Patch> = rule
                .scan(&ctx)
                .iter()
                .filter_map(|finding| {
                    let mut rng = StrategyRng::seeded(source.as_bytes(), rule.id());
                    rule.fix(finding, &ctx, &mut rng)
                })
                .collect();
            patches.sort_by_key(|p| p.range.start);

            let mut fixed = source.to_string();
            for patch in patches.iter().rev() {
                fixed.replace_range(patch.range.clone(), &patch.replacement);
            }

            let fixed_doc = document(&fixed);
            let fixed_ctx = RuleContext::new(&fixed_doc, &NoopTagger, "blog", &envelope);
            assert!(
                rule.scan(&fixed_ctx).is_empty(),
                "expected no findings left after fixing {source:?}, got fixed text {fixed:?}"
            );
        }
    }

    #[test]
    fn fixing_the_same_source_twice_is_byte_identical() {
        let source = "The pipeline leverages numerous stakeholders' feedback to facilitate \
                       specific, robust improvements.";
        let run = || {
            let doc = document(source);
            let envelope = MapEnvelope::new();
            let ctx = RuleContext::new(&doc, &NoopTagger, "blog", &envelope);
            let rule = SubstitutionRule::new();
            rule.scan(&ctx)
                .iter()
                .filter_map(|finding| {
                    let mut rng = StrategyRng::seeded(source.as_bytes(), rule.id());
                    rule.fix(finding, &ctx, &mut rng)
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(run(), run());
    }
}
