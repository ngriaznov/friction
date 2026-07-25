//! Register-marking construction counts over a dependency parse.
//!
//! Ported from `docs/research/regvec/biber.py`'s counting logic -- see that
//! file's own module doc for what it is and, more importantly, what it is
//! not: a reading of Biber's register categories through spaCy's parser
//! output, not a faithful `pseudobibeR` port, and the research phase's own
//! largest source of error. This module ports the *counting*, driven by
//! this crate's own [`SentenceParse`] instead of spaCy's `Doc`.
//!
//! # Reconstructing a coarse part-of-speech spaCy predicted directly
//!
//! `biber.py`'s detectors read spaCy's `Token.pos_` (a universal
//! part-of-speech spaCy's tagger predicts independently of the fine-grained
//! Penn tag and of the parser's dependency label) directly. Neither this
//! crate's tagger nor `docs/research/regvec/feature_parity.json` -- the
//! fixture this module is checked against, which carries only a token's
//! Penn tag and a closed dependency-relation vocabulary -- exposes that
//! second, independently-predicted layer. [`coarse_pos`] rebuilds it from
//! the Penn tag plus, where the tag alone is genuinely ambiguous, the
//! token's relation and surface text. The alternative -- shipping a
//! spaCy-only `pos_` field in the fixture and reading it directly -- was
//! rejected: it would make every consumer of a [`SentenceParse`] this crate
//! ever builds from its own parser permanently unable to satisfy this
//! module, since nothing outside the fixture will ever carry that field.
//!
//! Three of the table's entries are lexical exceptions the fixture itself
//! forced into the open by disagreeing with a pure tag-based mapping:
//! negation (`not`/`n't`, tag `RB`, never counts as an adverb), indefinite
//! `-thing`/`-one`/`-body` pronouns (tag `NN`, but a pronoun, not a noun),
//! and a closed set of English words that are complementizers or
//! subordinating conjunctions and never prepositions (`that`, `if`,
//! `because`, `although`, `though`, `unless`, `whether` -- every instance of
//! each across the whole fixture is a subordinator use, never a genuine
//! prepositional one). A fourth, narrower case does not fully close:
//! `since` is genuinely dual-use (a preposition in "since Monday", a
//! subordinator in "since it failed"), and the one fixture sentence where
//! it is a subordinator attached with a `prep` relation rather than `mark`
//! cannot be told apart, from tag and relation alone, from a genuine
//! prepositional `since` -- see [`prepositions`]'s own doc.

use friction_nlp::{DepEdge, DepRelation, SentenceParse, TaggedToken};

/// Per-feature counts of the register-marking constructions this module
/// detects in one sentence.
///
/// A struct with one named field per feature, not a `HashMap<&str,
/// usize>`: these seventeen features are a fixed, closed set (the counting
/// half of `docs/research/regvec/biber.py`'s own per-sentence dict, minus
/// the two outputs -- `mean_word_length` and the raw word total -- that are
/// rates or rate inputs rather than counts, and so are a caller's business,
/// not this module's), and a typo in a map key would silently produce a
/// missing feature rather than a compile error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RegisterCounts {
    /// A non-finite `VBG` clause (`acl`/`advcl`/`xcomp`), excluding
    /// progressives. See [`present_participials`].
    pub present_participial: usize,
    /// A deverbal or deadjectival noun by suffix. See [`nominalizations`].
    pub nominalization: usize,
    /// A coordination whose conjuncts are phrases, not clauses. See
    /// [`phrasal_coordinations`].
    pub phrasal_coord: usize,
    /// A `that`-clause functioning as a clausal subject. See
    /// [`that_subjects`].
    pub that_subj: usize,
    /// A passive verb with no agent. See [`agentless_passives`].
    pub agentless_passive: usize,
    /// A coordination whose conjuncts are clauses. See
    /// [`clausal_coordinations`].
    pub clausal_coord: usize,
    /// An adjectival modifier of a noun. See [`attributive_adjectives`].
    pub attributive_adj: usize,
    /// A postnominal past-participial modifier. See
    /// [`postnominal_past_participles`].
    pub past_part_postnom: usize,
    /// An adverb. See [`adverbs`].
    pub adverbs: usize,
    /// A downtoner. See [`downtoners`].
    pub downtoners: usize,
    /// A demonstrative determiner or pronoun, excluding complementizer
    /// `that`. See [`demonstratives`].
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
    /// A sentence-initial demonstrative. See
    /// [`sentence_initial_demonstratives`].
    pub demon_sent_initial: usize,
}

impl RegisterCounts {
    /// Counts every feature in one sentence.
    ///
    /// Calls the same per-feature detector functions this module exposes
    /// individually and takes their lengths -- counting and "which token"
    /// are two views onto one implementation, never two implementations
    /// that could quietly disagree about what matched. `tokens` and
    /// `parse` must describe the same sentence, one entry each, in the
    /// same order (the contract [`friction_nlp::DepParser::parse`]'s own
    /// output already satisfies).
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
        }
    }
}

// --------------------------------------------------------------------
// Coarse part-of-speech reconstruction -- see the module doc for why
// this exists instead of a direct field read.
// --------------------------------------------------------------------

/// A coarse part-of-speech category, reconstructed from a Penn tag (and,
/// for the handful of tags that are genuinely ambiguous on their own,
/// relation and surface text too). See [`coarse_pos`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CoarsePos {
    Noun,
    ProperNoun,
    Adjective,
    Adverb,
    Adposition,
    /// A complementizer or subordinating conjunction: never a preposition
    /// for the purposes of [`prepositions`], whatever dependency relation
    /// it happens to carry.
    Subordinator,
    Determiner,
    Pronoun,
    Number,
    /// A main or auxiliary verb. `biber.py`'s only detector that
    /// distinguishes spaCy's `VERB` from its `AUX` is
    /// [`clausal_coordinations`], which accepts both -- so nothing in this
    /// module ever needs to tell them apart, and folding them into one
    /// variant avoids reconstructing a distinction (which of "be", "have",
    /// "do", and the modals are functioning as an auxiliary in this
    /// particular clause) that genuinely depends on more context than a
    /// Penn tag and a relation carry.
    VerbLike,
    /// Everything else: coordinators, punctuation, symbols, and the
    /// negation particle (`not`/`n't`), which the fixture confirms is
    /// never counted as an adverb (see the module doc).
    Other,
}

const NEGATION_PARTICLES: [&str; 2] = ["not", "n't"];

/// The `-thing`/`-one`/`-body` indefinite pronouns: tagged `NN` in the
/// Penn tagset (Penn has no separate indefinite-pronoun tag), but not
/// nouns -- confirmed by two fixture sentences where a nominal-feature
/// count only matches once "something"/"everything" are excluded from
/// [`nouns`] (and, downstream, from [`postnominal_past_participles`]'s
/// head check).
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

/// English words that are complementizers or subordinating conjunctions
/// and never prepositions. Tagged `IN` like a preposition, and sometimes
/// attached with a relation other than `mark` in the reference parse (a
/// parser-labeling quirk, not a genuine prepositional use) -- every
/// occurrence of every word in this list, across the whole fixture, is a
/// subordinator. Deliberately narrower than the full set of English
/// subordinators: words that really can be either (`since`, `while`,
/// `once`, `before`, `after`, `as`) are left to the relation-based check
/// in [`coarse_pos`], not added here, because unlike this list every one of
/// them is attested as a genuine preposition somewhere in ordinary use --
/// hardcoding them would trade a real preposition miss for a subordinator
/// miss rather than fixing anything.
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

/// Reconstructs the coarse part-of-speech a `biber.py` detector would read
/// from spaCy's `Token.pos_`, from `pos` (a Penn tag), `relation` (this
/// token's own dependency relation to its head), and `surface` (this
/// token's exact surface text). See the module doc for the lexical
/// exceptions and the one case (dual-use `since`) that does not close.
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
        // book") and a standalone demonstrative or relative pronoun
        // ("that is fine", relative "that"/"which" heading its own
        // clause) -- Penn's tagset does not distinguish them, but the
        // `det` relation does: a token modifying a head noun gets `det`,
        // one standing in for a noun phrase on its own gets some other
        // relation (`nsubj`, `dobj`, `conj`...). Confirmed by a fixture
        // sentence where a coordinated "those" only counts as a phrasal
        // coordination once it is read as a pronoun rather than a
        // determiner.
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

/// [`coarse_pos`] for the token at `index`, reading its own relation from
/// `parse`. Every other call site needs a *different* token's relation
/// (a child's, when deciding whether it marks its head), so only this one
/// convenience wrapper is worth having.
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

// --------------------------------------------------------------------
// Detectors. Each returns the indices of the tokens it matched, in
// source order, so a rewriting caller knows *which* token licensed a
// count without re-deriving the same condition independently.
// --------------------------------------------------------------------

/// A `VBG` heading a non-finite clause (`acl`/`advcl`/`xcomp`), excluding
/// progressives.
///
/// A progressive ("is running") has an `aux`/`auxpass` child and is
/// excluded regardless of relation. An `xcomp` case ("kept running") is
/// progressive-shaped rather than a true participial adjunct unless a
/// comma directly precedes it (a real participial adjunct: "..., running
/// low on memory, ..."), so it counts only then.
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

/// A deverbal or deadjectival noun by suffix.
///
/// A common noun (not a proper noun, and not one of the indefinite
/// pronouns [`coarse_pos`] excludes from `NOUN`) at least six characters
/// long, ending in one of a fixed set of nominalizing suffixes, and not
/// one of a fixed exception list of words that end in those suffixes
/// without being nominalizations (`signal`, `nature`, `figure`...). Ported
/// verbatim from `biber.py`'s `NOMINAL_SUFFIXES`/`NOMINAL_EXCEPT`.
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

/// A coordination (`conj`) whose conjunct, and whose conjunct's own head,
/// are both phrase-level categories (noun, proper noun, pronoun,
/// adjective, adverb, or number) rather than clauses.
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

/// A `that`-clause functioning as a clausal subject: a `csubj` token with
/// a `mark` child spelled "that".
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

/// A coordination (`conj`) whose conjunct is verb-like: a clausal, not
/// phrasal, coordination.
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
/// "Names no agent" means no `agent` child, and no `prep` child spelled
/// "by". A passive *with* an agent is the complement class this
/// deliberately does not count -- the fixture pins both directly (one
/// sentence has `auxpass` and an `agent` child and a count of 0; another
/// has `auxpass` and a `by`-prep child and a count of 0; a third has bare
/// `auxpass` and a count of 1).
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

/// A postnominal past-participial modifier: a `VBN` attached to its head
/// noun with `acl` ("the result **set returned by the database**").
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

/// A demonstrative determiner or pronoun: "this"/"that"/"these"/"those",
/// excluding complementizer "that" (relation `mark`).
///
/// This exclusion is the single most load-bearing line in this module.
/// `biber.py`'s own comment on the equivalent line names the exact failure
/// mode: complementizer "that" silently counting as a demonstrative is
/// invisible in a rewrite's output text and was only caught by a
/// downstream delta-validation check, not by reading the result.
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
/// `since` used as a subordinator but attached with a relation other than
/// `mark` in the reference parse is the one case this module cannot
/// distinguish from a genuine prepositional `since` -- see the module doc.
/// It affects one sentence out of the reference fixture's 188.
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

/// A first-person pronoun or possessive ("I", "we", "my", "our", "us",
/// "me").
#[must_use]
pub fn first_person_pronouns(text: &str, tokens: &[TaggedToken]) -> Vec<usize> {
    (0..tokens.len())
        .filter(|&index| is_one_of(surface_of(text, tokens, index), &FIRST_PERSON_WORDS))
        .collect()
}

const SENTENCE_INITIAL_DEMONSTRATIVE_WORDS: [&str; 3] = ["this", "these", "that"];

/// A sentence-initial demonstrative: "this"/"these"/"that" (not "those") at
/// token index 0.
///
/// A positional signature, not one of Biber's own categories: the repair
/// for a present participial ("... . **This** ensures X") concentrates
/// demonstratives at sentence-initial position, which the plain
/// [`demonstratives`] count cannot express on its own. Unlike
/// [`demonstratives`], this has no dependency-relation or part-of-speech
/// filter and no "those" -- `biber.py`'s own detector doesn't have them
/// either, so a sentence-initial complementizer "that" (which cannot occur
/// in well-formed English, since a complementizer never opens a sentence)
/// would count here if it somehow did.
///
/// Callers pass one sentence's tokens at a time, so "sentence-initial"
/// here is simply "index 0" rather than a document-relative flag.
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
