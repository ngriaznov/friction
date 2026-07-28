//! Register-marking construction counts over a dependency parse.
//!
//! The features are a reading of Biber's register categories over this
//! crate's own [`SentenceParse`]. How the human-band targets were
//! measured from these counts -- including what the corpus turned out
//! not to support -- is documented in
//! `docs/research/regvec/TARGET_ESTIMATION.md`; because the bands were
//! measured with exactly this counting, its semantics are pinned:
//! changing a detector means re-measuring every band.
//!
//! # Reconstructing a coarse part-of-speech
//!
//! The detectors match on a coarse lexical category (noun, adjective,
//! adposition, ...) that a Penn tag alone doesn't determine, so
//! [`coarse_pos`] rebuilds it from the Penn tag plus, where ambiguous,
//! relation and surface text. Storing a precomputed category in the
//! fixture was rejected: no [`SentenceParse`] this crate builds would
//! ever carry it.
//!
//! Three entries are lexical exceptions the fixture forced into the open
//! (negation, indefinite `-thing`/`-one`/`-body` pronouns, a closed
//! subordinator set) -- see each list's own doc for specifics. One case
//! doesn't close: `since` is dual-use, and the fixture sentence where it's
//! a subordinator attached with `prep` rather than `mark` can't be told
//! apart from a genuine prepositional `since` -- see [`prepositions`].

use friction_nlp::{DepEdge, DepRelation, SentenceParse, TaggedToken};

/// Per-feature counts of the register-marking constructions this module
/// detects in one sentence.
///
/// A struct with one field per feature, not a `HashMap<&str, usize>`:
/// the field set is fixed by the feature inventory the bands were
/// measured over, and a typo in a map key would fail to compile instead
/// of silently reading a missing feature. [`Self::em_dashes`] and [`Self::semicolons`] are the
/// exceptions -- not Biber categories at all, each added later for its
/// own band (see [`em_dashes`]'s and [`semicolons`]'s own docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RegisterCounts {
    /// A non-finite `VBG` clause, not a progressive. See [`present_participials`].
    pub present_participial: usize,
    /// A deverbal or deadjectival noun. See [`nominalizations`].
    pub nominalization: usize,
    /// A phrase-level coordination. See [`phrasal_coordinations`].
    pub phrasal_coord: usize,
    /// A `that`-clause as clausal subject. See [`that_subjects`].
    pub that_subj: usize,
    /// A passive verb with no agent. See [`agentless_passives`].
    pub agentless_passive: usize,
    /// A clause-level coordination. See [`clausal_coordinations`].
    pub clausal_coord: usize,
    /// An adjectival noun modifier. See [`attributive_adjectives`].
    pub attributive_adj: usize,
    /// A postnominal past-participial modifier. See [`postnominal_past_participles`].
    pub past_part_postnom: usize,
    /// An adverb. See [`adverbs`].
    pub adverbs: usize,
    /// A downtoner. See [`downtoners`].
    pub downtoners: usize,
    /// A demonstrative, excluding complementizer `that`. See [`demonstratives`].
    pub demonstratives: usize,
    /// A preposition. See [`prepositions`].
    pub prepositions: usize,
    /// A bare-infinitive `to`-clause. See [`infinitives`].
    pub infinitives: usize,
    /// A common or proper noun. See [`nouns`].
    pub nouns: usize,
    /// A hedge. See [`hedges`].
    pub hedges: usize,
    /// A first-person pronoun. See [`first_person_pronouns`].
    pub first_person: usize,
    /// A sentence-initial demonstrative. See [`sentence_initial_demonstratives`].
    pub demon_sent_initial: usize,
    /// An em dash (U+2014). See [`em_dashes`].
    pub em_dashes: usize,
    /// An ASCII semicolon (U+003B). See [`semicolons`].
    pub semicolons: usize,
}

impl RegisterCounts {
    /// Counts every feature in one sentence.
    ///
    /// Calls the same detector functions this module exposes individually
    /// and takes their lengths, so counting and "which token" can't
    /// quietly disagree. `tokens` and `parse` must describe the same
    /// sentence, one entry each, in the same order (as
    /// [`friction_nlp::DepParser::parse`]'s output already does).
    #[must_use]
    pub fn count(text: &str, tokens: &[TaggedToken], parse: &SentenceParse) -> Self {
        Self {
            present_participial: present_participials(text, tokens, parse).len(),
            nominalization: nominalizations(text, tokens, parse).len(),
            phrasal_coord: phrasal_coordinations(text, tokens, parse).len(),
            that_subj: that_subjects(text, tokens, parse).len(),
            agentless_passive: agentless_passives(text, tokens, parse).len(),
            clausal_coord: clausal_coordinations(text, tokens, parse).len(),
            attributive_adj: attributive_adjectives(parse).len(),
            past_part_postnom: postnominal_past_participles(text, tokens, parse).len(),
            adverbs: adverbs(text, tokens, parse).len(),
            downtoners: downtoners(text, tokens).len(),
            demonstratives: demonstratives(text, tokens, parse).len(),
            prepositions: prepositions(text, tokens, parse).len(),
            infinitives: infinitives(text, tokens, parse).len(),
            nouns: nouns(text, tokens, parse).len(),
            hedges: hedges(text, tokens).len(),
            first_person: first_person_pronouns(text, tokens).len(),
            demon_sent_initial: sentence_initial_demonstratives(text, tokens).len(),
            em_dashes: em_dashes(text).len(),
            semicolons: semicolons(text).len(),
        }
    }
}

// Coarse part-of-speech reconstruction; see the module doc for why.

/// A coarse part-of-speech, reconstructed from a Penn tag and, where
/// ambiguous, relation and surface text. See [`coarse_pos`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CoarsePos {
    Noun,
    ProperNoun,
    Adjective,
    Adverb,
    Adposition,
    /// A complementizer or subordinating conjunction: never a preposition
    /// for [`prepositions`], regardless of relation.
    Subordinator,
    Determiner,
    Pronoun,
    Number,
    /// A main or auxiliary verb: [`clausal_coordinations`], the only
    /// detector that cares, accepts both, so folding them together
    /// avoids deciding whether "be"/"have"/"do"/a modal is acting as an
    /// auxiliary, a call needing context beyond a tag and relation.
    VerbLike,
    /// Everything else: coordinators, punctuation, symbols, and the
    /// negation particle (`not`/`n't`, never an adverb -- see module doc).
    Other,
}

const NEGATION_PARTICLES: [&str; 2] = ["not", "n't"];

/// The `-thing`/`-one`/`-body` indefinite pronouns: tagged `NN` (Penn has
/// no indefinite-pronoun tag) but not nouns -- confirmed by two fixture
/// sentences excluding "something"/"everything" from [`nouns`].
const INDEFINITE_PRONOUNS: [&str; 10] = [
    "something",
    "anything",
    "nothing",
    "everything",
    "someone",
    "anyone",
    "nobody",
    "everybody",
    "everyone",
    "anybody",
];

/// Complementizers/subordinating conjunctions that are never prepositions
/// -- tagged `IN`, sometimes attached with a relation other than `mark`.
/// Narrower than the full subordinator set: dual-use words (`since`,
/// `while`, `once`, `before`, `after`, `as`) go to [`coarse_pos`]'s
/// relation-based check instead, since each is also a genuine preposition
/// somewhere -- hardcoding them would trade one miss for the other.
const PURE_SUBORDINATORS: [&str; 7] = [
    "that", "if", "because", "although", "though", "unless", "whether",
];

fn surface_of<'a>(text: &'a str, tokens: &[TaggedToken], index: usize) -> &'a str {
    &text[tokens[index].token.range.clone()]
}

fn is_one_of(word: &str, candidates: &[&str]) -> bool {
    candidates
        .iter()
        .any(|candidate| word.eq_ignore_ascii_case(candidate))
}

/// Reconstructs the coarse lexical category the detectors match on,
/// given `pos` (Penn tag), `relation` (to its head), and `surface`
/// (exact text). See the module doc for exceptions and the open case
/// (`since`).
fn coarse_pos(pos: &str, relation: DepRelation, surface: &str) -> CoarsePos {
    if matches!(pos, "RB" | "RBR" | "RBS") && is_one_of(surface, &NEGATION_PARTICLES) {
        return CoarsePos::Other;
    }
    match pos {
        "NN" | "NNS" => {
            if is_one_of(surface, &INDEFINITE_PRONOUNS) {
                CoarsePos::Pronoun
            } else {
                CoarsePos::Noun
            }
        }
        "NNP" | "NNPS" => CoarsePos::ProperNoun,
        "JJ" | "JJR" | "JJS" => CoarsePos::Adjective,
        "RB" | "RBR" | "RBS" => CoarsePos::Adverb,
        "IN" | "RP" => {
            if relation == DepRelation::Mark || is_one_of(surface, &PURE_SUBORDINATORS) {
                CoarsePos::Subordinator
            } else {
                CoarsePos::Adposition
            }
        }
        // `DT`/`PDT`/`WDT` cover both an adnominal determiner ("that
        // book") and a standalone demonstrative/relative pronoun ("that
        // is fine") -- Penn doesn't distinguish them, but `det` does. A
        // fixture sentence confirms it: coordinated "those" counts as
        // phrasal only once read as a pronoun.
        "DT" | "PDT" | "WDT" => {
            if relation == DepRelation::Det {
                CoarsePos::Determiner
            } else {
                CoarsePos::Pronoun
            }
        }
        "PRP" | "WP" => CoarsePos::Pronoun,
        "CD" => CoarsePos::Number,
        "VB" | "VBD" | "VBG" | "VBN" | "VBP" | "VBZ" | "MD" => CoarsePos::VerbLike,
        _ => CoarsePos::Other,
    }
}

/// [`coarse_pos`] for the token at `index`, using its own relation from
/// `parse` -- other call sites need a *different* token's relation (a
/// child's, checking whether it marks its head), so only this one earns
/// a wrapper.
fn coarse_pos_at(
    text: &str,
    tokens: &[TaggedToken],
    parse: &SentenceParse,
    index: usize,
) -> CoarsePos {
    let relation = parse
        .edge(index)
        .map_or(DepRelation::Other, |edge| edge.relation);
    coarse_pos(
        tokens[index].pos.as_str(),
        relation,
        surface_of(text, tokens, index),
    )
}

fn children_of(parse: &SentenceParse, head: usize) -> impl Iterator<Item = &DepEdge> {
    parse
        .edges()
        .iter()
        .filter(move |edge| edge.head == Some(head))
}

// Detectors. Each returns matched token indices, in source order, so a
// caller knows *which* token licensed a count without re-deriving it.

/// A `VBG` heading a non-finite clause (`acl`/`advcl`/`xcomp`), excluding progressives.
///
/// A progressive ("is running") has an `aux`/`auxpass` child and is
/// excluded regardless of relation. An `xcomp` case ("kept running")
/// only counts with a comma directly before it (a real adjunct: "...,
/// running low on memory, ...") -- otherwise it's progressive-shaped.
#[must_use]
pub fn present_participials(
    text: &str,
    tokens: &[TaggedToken],
    parse: &SentenceParse,
) -> Vec<usize> {
    let mut out = Vec::new();
    for edge in parse.edges() {
        let index = edge.token;
        if tokens[index].pos.as_str() != "VBG" {
            continue;
        }
        if !matches!(
            edge.relation,
            DepRelation::Acl | DepRelation::Advcl | DepRelation::Xcomp
        ) {
            continue;
        }
        let has_aux = children_of(parse, index)
            .any(|child| matches!(child.relation, DepRelation::Aux | DepRelation::AuxPass));
        if has_aux {
            continue;
        }
        if edge.relation == DepRelation::Xcomp {
            let comma_before = index > 0 && surface_of(text, tokens, index - 1) == ",";
            if !comma_before {
                continue;
            }
        }
        out.push(index);
    }
    out
}

/// A deverbal or deadjectival noun, identified by suffix.
///
/// A common noun (not proper, not one of [`coarse_pos`]'s indefinite
/// pronouns) at least six characters long, ending in a fixed
/// nominalizing suffix and absent from a fixed exception list
/// (`signal`, `nature`, `figure`...). Both lists are pinned by the band
/// measurement -- the human bands were measured with exactly these
/// lists, so extending either means re-measuring the bands.
#[must_use]
pub fn nominalizations(text: &str, tokens: &[TaggedToken], parse: &SentenceParse) -> Vec<usize> {
    const SUFFIXES: [&str; 14] = [
        "tion", "sion", "ment", "ness", "ity", "ance", "ence", "ism", "ship", "hood", "ency",
        "ancy", "ure", "al",
    ];
    const EXCEPTIONS: [&str; 12] = [
        "animal", "signal", "material", "capital", "total", "final", "several", "nature", "future",
        "picture", "measure", "figure",
    ];

    let mut out = Vec::new();
    for index in 0..tokens.len() {
        if coarse_pos_at(text, tokens, parse, index) != CoarsePos::Noun {
            continue;
        }
        let low = surface_of(text, tokens, index).to_lowercase();
        if EXCEPTIONS.contains(&low.as_str()) || low.chars().count() < 6 {
            continue;
        }
        if SUFFIXES.iter().any(|suffix| low.ends_with(suffix)) {
            out.push(index);
        }
    }
    out
}

/// A coordination (`conj`) whose conjunct and its head are both
/// phrase-level (noun, proper noun, pronoun, adjective, adverb,
/// number), not clausal.
#[must_use]
pub fn phrasal_coordinations(
    text: &str,
    tokens: &[TaggedToken],
    parse: &SentenceParse,
) -> Vec<usize> {
    const fn is_phrasal(pos: CoarsePos) -> bool {
        matches!(
            pos,
            CoarsePos::Noun
                | CoarsePos::ProperNoun
                | CoarsePos::Pronoun
                | CoarsePos::Adjective
                | CoarsePos::Adverb
                | CoarsePos::Number
        )
    }

    let mut out = Vec::new();
    for edge in parse.edges() {
        if edge.relation != DepRelation::Conj {
            continue;
        }
        let Some(head) = edge.head else { continue };
        let index = edge.token;
        if is_phrasal(coarse_pos_at(text, tokens, parse, index))
            && is_phrasal(coarse_pos_at(text, tokens, parse, head))
        {
            out.push(index);
        }
    }
    out
}

/// A `that`-clause functioning as a clausal subject: a `csubj` token with a `mark` child spelled "that".
#[must_use]
pub fn that_subjects(text: &str, tokens: &[TaggedToken], parse: &SentenceParse) -> Vec<usize> {
    let mut out = Vec::new();
    for edge in parse.edges() {
        if edge.relation != DepRelation::Csubj {
            continue;
        }
        let index = edge.token;
        let has_that_mark = children_of(parse, index).any(|child| {
            child.relation == DepRelation::Mark
                && surface_of(text, tokens, child.token).eq_ignore_ascii_case("that")
        });
        if has_that_mark {
            out.push(index);
        }
    }
    out
}

/// A coordination (`conj`) whose conjunct is verb-like: a clausal, not phrasal, coordination.
#[must_use]
pub fn clausal_coordinations(
    text: &str,
    tokens: &[TaggedToken],
    parse: &SentenceParse,
) -> Vec<usize> {
    parse
        .edges()
        .iter()
        .filter(|edge| {
            edge.relation == DepRelation::Conj
                && coarse_pos_at(text, tokens, parse, edge.token) == CoarsePos::VerbLike
        })
        .map(|edge| edge.token)
        .collect()
}

/// A passive verb (one with an `auxpass` child) that names no agent.
///
/// "Names no agent" means no `agent` child and no `by`-prep child (a
/// passive *with* an agent is deliberately excluded). Fixture-pinned:
/// `auxpass`+`agent` is 0, `auxpass`+`by`-prep is 0, bare `auxpass` is
/// 1.
#[must_use]
pub fn agentless_passives(text: &str, tokens: &[TaggedToken], parse: &SentenceParse) -> Vec<usize> {
    let mut out = Vec::new();
    for index in 0..tokens.len() {
        let children: Vec<&DepEdge> = children_of(parse, index).collect();
        if !children
            .iter()
            .any(|child| child.relation == DepRelation::AuxPass)
        {
            continue;
        }
        let has_agent = children.iter().any(|child| {
            child.relation == DepRelation::Agent
                || (child.relation == DepRelation::Prep
                    && surface_of(text, tokens, child.token).eq_ignore_ascii_case("by"))
        });
        if !has_agent {
            out.push(index);
        }
    }
    out
}

/// An adjectival modifier of a noun (`amod`).
#[must_use]
pub fn attributive_adjectives(parse: &SentenceParse) -> Vec<usize> {
    parse
        .edges()
        .iter()
        .filter(|edge| edge.relation == DepRelation::Amod)
        .map(|edge| edge.token)
        .collect()
}

/// A postnominal past-participial modifier: a `VBN` attached to its head noun with `acl` ("the result **set returned by the database**").
#[must_use]
pub fn postnominal_past_participles(
    text: &str,
    tokens: &[TaggedToken],
    parse: &SentenceParse,
) -> Vec<usize> {
    let mut out = Vec::new();
    for edge in parse.edges() {
        let index = edge.token;
        if tokens[index].pos.as_str() != "VBN" || edge.relation != DepRelation::Acl {
            continue;
        }
        let Some(head) = edge.head else { continue };
        if matches!(
            coarse_pos_at(text, tokens, parse, head),
            CoarsePos::Noun | CoarsePos::ProperNoun
        ) {
            out.push(index);
        }
    }
    out
}

/// An adverb.
#[must_use]
pub fn adverbs(text: &str, tokens: &[TaggedToken], parse: &SentenceParse) -> Vec<usize> {
    (0..tokens.len())
        .filter(|&index| coarse_pos_at(text, tokens, parse, index) == CoarsePos::Adverb)
        .collect()
}

const DOWNTONER_WORDS: [&str; 11] = [
    "barely",
    "hardly",
    "mildly",
    "nearly",
    "partially",
    "partly",
    "practically",
    "scarcely",
    "slightly",
    "somewhat",
    "almost",
];

/// A downtoner (a fixed lexical set: "barely", "nearly", "somewhat"...).
#[must_use]
pub fn downtoners(text: &str, tokens: &[TaggedToken]) -> Vec<usize> {
    (0..tokens.len())
        .filter(|&index| is_one_of(surface_of(text, tokens, index), &DOWNTONER_WORDS))
        .collect()
}

const DEMONSTRATIVE_WORDS: [&str; 4] = ["this", "that", "these", "those"];

/// A demonstrative determiner or pronoun ("this"/"that"/"these"/"those").
///
/// Excludes complementizer "that" (relation `mark`) -- the single most
/// load-bearing exclusion here: counting it silently as a demonstrative
/// is invisible in a rewrite's output, caught only by a downstream
/// delta-validation check.
#[must_use]
pub fn demonstratives(text: &str, tokens: &[TaggedToken], parse: &SentenceParse) -> Vec<usize> {
    let mut out = Vec::new();
    for edge in parse.edges() {
        let index = edge.token;
        if edge.relation == DepRelation::Mark {
            continue;
        }
        if !is_one_of(surface_of(text, tokens, index), &DEMONSTRATIVE_WORDS) {
            continue;
        }
        if matches!(
            coarse_pos_at(text, tokens, parse, index),
            CoarsePos::Determiner | CoarsePos::Pronoun
        ) {
            out.push(index);
        }
    }
    out
}

/// A preposition.
///
/// `since` attached with a relation other than `mark` is the one case
/// this module can't tell from a genuine prepositional `since` -- see
/// the module doc. Affects one sentence of the fixture's 188.
#[must_use]
pub fn prepositions(text: &str, tokens: &[TaggedToken], parse: &SentenceParse) -> Vec<usize> {
    (0..tokens.len())
        .filter(|&index| coarse_pos_at(text, tokens, parse, index) == CoarsePos::Adposition)
        .collect()
}

/// A bare-infinitive `to`-clause: a `VB` with an `aux` child spelled "to".
#[must_use]
pub fn infinitives(text: &str, tokens: &[TaggedToken], parse: &SentenceParse) -> Vec<usize> {
    let mut out = Vec::new();
    for edge in parse.edges() {
        let index = edge.token;
        if tokens[index].pos.as_str() != "VB" {
            continue;
        }
        let has_to_aux = children_of(parse, index).any(|child| {
            child.relation == DepRelation::Aux
                && surface_of(text, tokens, child.token).eq_ignore_ascii_case("to")
        });
        if has_to_aux {
            out.push(index);
        }
    }
    out
}

/// A common or proper noun.
#[must_use]
pub fn nouns(text: &str, tokens: &[TaggedToken], parse: &SentenceParse) -> Vec<usize> {
    (0..tokens.len())
        .filter(|&index| {
            matches!(
                coarse_pos_at(text, tokens, parse, index),
                CoarsePos::Noun | CoarsePos::ProperNoun
            )
        })
        .collect()
}

const HEDGE_WORDS: [&str; 22] = [
    "perhaps",
    "maybe",
    "possibly",
    "probably",
    "apparently",
    "seemingly",
    "arguably",
    "presumably",
    "roughly",
    "approximately",
    "likely",
    "might",
    "could",
    "may",
    "suggest",
    "suggests",
    "appear",
    "appears",
    "seem",
    "seems",
    "tend",
    "tends",
];

/// A hedge (a fixed lexical set: "perhaps", "might", "suggests"...).
#[must_use]
pub fn hedges(text: &str, tokens: &[TaggedToken]) -> Vec<usize> {
    (0..tokens.len())
        .filter(|&index| is_one_of(surface_of(text, tokens, index), &HEDGE_WORDS))
        .collect()
}

const FIRST_PERSON_WORDS: [&str; 6] = ["i", "we", "my", "our", "us", "me"];

/// A first-person pronoun or possessive ("I", "we", "my", "our", "us", "me").
#[must_use]
pub fn first_person_pronouns(text: &str, tokens: &[TaggedToken]) -> Vec<usize> {
    (0..tokens.len())
        .filter(|&index| is_one_of(surface_of(text, tokens, index), &FIRST_PERSON_WORDS))
        .collect()
}

const SENTENCE_INITIAL_DEMONSTRATIVE_WORDS: [&str; 3] = ["this", "these", "that"];

/// A sentence-initial demonstrative: "this"/"these"/"that" (not "those")
/// at token index 0 -- "index 0" since callers pass one sentence at a
/// time.
///
/// A positional signature, not one of Biber's categories: the repair for
/// a present participial ("... . **This** ensures X") concentrates
/// demonstratives sentence-initially, which plain [`demonstratives`]
/// can't express. No relation/part-of-speech filter and no "those" (the
/// band measurement counted exactly this set), so a sentence-initial
/// complementizer "that" -- impossible in well-formed English -- would
/// count here if it occurred.
#[must_use]
pub fn sentence_initial_demonstratives(text: &str, tokens: &[TaggedToken]) -> Vec<usize> {
    if tokens.is_empty()
        || !is_one_of(
            surface_of(text, tokens, 0),
            &SENTENCE_INITIAL_DEMONSTRATIVE_WORDS,
        )
    {
        Vec::new()
    } else {
        vec![0]
    }
}

/// An em dash (U+2014).
///
/// Strictly that character, never an en dash (U+2013, which numeric
/// ranges like "1-64" use and this feature must stay invisible to) and
/// never a hyphen-minus surrogate ("--").
/// `friction-metrics::rhythm::em_dash_density` treats those as
/// equivalent for its own purpose (a coarse rhythm signal); this feature
/// exists specifically because human technical writing almost never uses
/// the real character at all (see `register-v1.toml`'s
/// `[features.em_dash]`), a distinction a broader detector would blur.
///
/// Returns byte offsets, not token indices, unlike every other detector
/// in this module: the tokenizer merges a dash directly adjacent to
/// another punctuation mark (a quote, a comma) into one multi-character
/// token, so counting by character avoids ever under-counting an
/// occurrence that's present in the text but not its own token.
#[must_use]
pub fn em_dashes(text: &str) -> Vec<usize> {
    text.char_indices()
        .filter_map(|(index, c)| (c == '\u{2014}').then_some(index))
        .collect()
}

#[cfg(test)]
mod em_dash_tests {
    use super::em_dashes;

    /// Counts every real em dash, spaced or not, including a doubled run.
    #[test]
    fn em_dashes_counts_every_occurrence() {
        let text = "A value\u{2014}B, and C \u{2014} D\u{2014}\u{2014}E.";
        assert_eq!(em_dashes(text).len(), 4);
    }

    /// An en dash (U+2013), as in a numeric range, never counts.
    #[test]
    fn em_dashes_ignores_en_dash() {
        assert_eq!(em_dashes("Supported on pages 1\u{2013}64.").len(), 0);
    }

    /// A hyphen-minus surrogate ("--"), which
    /// `friction-metrics::rhythm::count_em_dashes` treats as an em dash
    /// for its own purpose, never counts here.
    #[test]
    fn em_dashes_ignores_hyphen_minus_surrogate() {
        assert_eq!(em_dashes("It works fine--somehow; trust me.").len(), 0);
    }

    /// No occurrence at all is the common case in human technical prose.
    #[test]
    fn em_dashes_is_zero_for_ordinary_text() {
        assert_eq!(em_dashes("The build finished without incident.").len(), 0);
    }
}

/// An ASCII semicolon (U+003B).
///
/// Strictly that character, never U+037E (the Greek question mark, a
/// visual lookalike in most fonts but a distinct code point) -- unlike
/// [`em_dashes`], there's no equivalent-surrogate character this feature
/// must also stay blind to, since nothing else renders as a semicolon.
///
/// A semicolon inside a decoded HTML character reference (`&amp;` ->
/// `&`) never reaches this function at all: `friction_parse::extract`
/// runs Markdown text through `pulldown-cmark`, which resolves every
/// *recognized* named/numeric reference to its literal character before
/// this crate ever sees the text, so the reference's own `;` byte is
/// gone along with the rest of the escape. An unrecognized reference
/// (`&notareal;`) is left as literal source text, semicolon included --
/// correctly counted here, since at that point it's just a semicolon a
/// human reader would also see.
///
/// Returns byte offsets, not token indices, for the same reason
/// [`em_dashes`] does: a semicolon glued to adjacent punctuation
/// tokenizes as one multi-character token, and counting by character
/// avoids under-counting an occurrence that's present in the text but
/// not its own token.
#[must_use]
pub fn semicolons(text: &str) -> Vec<usize> {
    text.char_indices()
        .filter_map(|(index, c)| (c == ';').then_some(index))
        .collect()
}

#[cfg(test)]
mod semicolon_tests {
    use super::semicolons;

    /// Counts every occurrence, including a run of two.
    #[test]
    fn semicolons_counts_every_occurrence() {
        let text = "A holds items; B drains them;; C waits.";
        assert_eq!(semicolons(text).len(), 3);
    }

    /// A Greek question mark (U+037E) -- a visual lookalike in most
    /// fonts -- never counts; only the ASCII code point does.
    #[test]
    fn semicolons_ignores_greek_question_mark() {
        assert_eq!(semicolons("Is this a question\u{37e}").len(), 0);
    }

    /// Nonzero is the common case in human technical prose, unlike
    /// `em_dashes` -- this feature's band is genuinely nonzero (see
    /// `register-v1.toml`'s `[features.semicolon]`).
    #[test]
    fn semicolons_counts_ordinary_prose_usage() {
        assert_eq!(
            semicolons("The service retries once; the caller sees no error.").len(),
            1
        );
    }

    #[test]
    fn semicolons_is_zero_for_text_with_none() {
        assert_eq!(semicolons("The build finished without incident.").len(), 0);
    }
}
