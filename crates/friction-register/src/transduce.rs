//! Rewrite transducers ported from a Python register-rewriting prototype.
//!
//! Each one proposes a candidate edit that predicts an exact per-feature
//! count delta (Biber features are countable, never rates), together
//! with a confidence and the byte range it would replace. The prototype
//! defined five transducers; only two are ported here.
//!
//! [`t4_activize_to_passive`] and [`t5_nominalization`] depend on the
//! `nsubj`/`dobj`/`det`/`prep`/`pobj` relations, which the shipped parser
//! resolves at 82-96% accuracy. The other three (participial-clause
//! rewrites, out of scope here) depend on `acl`/`advcl`, resolved at only
//! 52-58%: porting them would mean firing wrongly about half the time,
//! which is worse than not firing.
//!
//! Every function here only proposes candidates. None of them mutate
//! `source`, pick between overlapping candidates, or apply anything —
//! that is a caller's decision, made with information (which candidates
//! were also produced elsewhere, which the caller has already committed
//! to) this module deliberately does not have.

use std::collections::BTreeMap;
use std::ops::Range;

use friction_core::CoreError;
use friction_core::span::{Spanned, validate_range};
use friction_nlp::{DepEdge, DepRelation, SentenceParse, TaggedToken, coarse_tag};

/// A proposed rewrite: replace the source bytes at `range` with
/// `replacement`, moving the Biber feature counts named in `delta` by the
/// listed amount.
///
/// Mirrors `friction_core::Patch`'s shape (byte range plus replacement
/// text) but is deliberately its own type rather than a reuse of `Patch`:
/// a `Patch` is a decision already made (which `friction-edit` applies
/// atomically), while a `Candidate` is a proposal that has not yet been
/// selected — folding the two together would make "has this been chosen
/// yet" an implicit, un-typed question a caller has to track by
/// convention instead of by the type system.
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    /// Which transducer produced this candidate.
    pub kind: CandidateKind,
    /// Byte range in the original source this candidate would replace.
    pub range: Range<usize>,
    /// The text to substitute for `range`.
    pub replacement: Box<str>,
    /// The Biber feature-count changes this candidate predicts, keyed by
    /// feature name (e.g. `"agentless_passive"`, `+1`). A [`BTreeMap`]
    /// rather than a `HashMap`: this workspace never uses hash-ordered
    /// collections where a fixed, meaningful order is available instead,
    /// and alphabetical-by-feature-name is exactly such an order here.
    pub delta: BTreeMap<&'static str, i32>,
    /// How much this transducer's own author trusts this class of
    /// rewrite, in `[0.0, 1.0]`. Deliberately a plain `f32`, not
    /// [`friction_nlp::Confidence`]: that type's documented contract is a
    /// single parse decision's margin over its next-best alternative,
    /// which is not what this number means — it is a fixed, hand-set
    /// trust level per transducer kind, unrelated to any one parse
    /// edge's ambiguity.
    pub confidence: f32,
}

impl Candidate {
    /// Validates this candidate's `range` against `source`: in bounds, and
    /// both endpoints on a UTF-8 character boundary.
    ///
    /// Mirrors `friction_core::Patch::validate`: a `Candidate` carries the
    /// same span-honesty obligation `Patch` does, so it is checked the
    /// same way.
    ///
    /// # Errors
    /// Returns [`CoreError`] if `range` is out of bounds for `source` or
    /// splits a UTF-8 character.
    pub fn validate(&self, source: &str) -> Result<(), CoreError> {
        validate_range(source, &self.range)
    }
}

impl Spanned for Candidate {
    fn range(&self) -> Range<usize> {
        self.range.clone()
    }
}

/// Which transducer produced a [`Candidate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CandidateKind {
    /// Produced by [`t4_activize_to_passive`].
    ActivizeToPassive,
    /// Produced by [`t5_nominalization`].
    NominalizationUnpack,
}

// ---------------------------------------------------------------------
// Subtree helpers. `SentenceParse` guarantees one edge per token and a
// well-formed tree (see its own docs), so a walk from any token down
// through its children always terminates and always includes at least
// the token itself.
// ---------------------------------------------------------------------

/// Every edge in `parse` whose head is `head` and whose relation is
/// `relation`, in token order.
fn children_with_relation(
    parse: &SentenceParse,
    head: usize,
    relation: DepRelation,
) -> impl Iterator<Item = &DepEdge> {
    parse
        .edges()
        .iter()
        .filter(move |edge| edge.head == Some(head) && edge.relation == relation)
}

/// Every token index in `root`'s own subtree: `root` itself, plus every
/// token reachable by repeatedly following `head` edges down from it.
fn subtree_indices(parse: &SentenceParse, root: usize) -> Vec<usize> {
    let mut collected = vec![root];
    let mut frontier = vec![root];
    while let Some(current) = frontier.pop() {
        for edge in parse.edges() {
            if edge.head == Some(current) {
                collected.push(edge.token);
                frontier.push(edge.token);
            }
        }
    }
    collected
}

/// The first and last token index in `root`'s subtree, by token order.
///
/// Not necessarily `root` at either end: a subtree can extend leftward of
/// its head (a determiner) or rightward of it (a prepositional phrase),
/// so the span is the min/max index reached, not `root`'s own index.
fn subtree_span(parse: &SentenceParse, root: usize) -> (usize, usize) {
    let indices = subtree_indices(parse, root);
    let first = indices
        .iter()
        .copied()
        .min()
        .expect("subtree_indices always includes root");
    let last = indices
        .iter()
        .copied()
        .max()
        .expect("subtree_indices always includes root");
    (first, last)
}

/// The exact source text token `index` occupies.
fn token_text<'src>(source: &'src str, tokens: &[TaggedToken], index: usize) -> &'src str {
    &source[tokens[index].token.range.clone()]
}

/// The exact source text spanning from the start of token `first` to the
/// end of token `last`, inclusive. Built from the same per-token ranges
/// every other span in this workspace is validated against, so it is
/// byte-honest by construction rather than by a separate check.
fn span_text<'src>(
    source: &'src str,
    tokens: &[TaggedToken],
    first: usize,
    last: usize,
) -> &'src str {
    &source[tokens[first].token.range.start..tokens[last].token.range.end]
}

/// Recapitalizes `text`'s first character, leaving the rest untouched.
fn recapitalize(text: &str) -> String {
    let mut chars = text.chars();
    chars.next().map_or_else(String::new, |first| {
        let mut out: String = first.to_uppercase().collect();
        out.push_str(chars.as_str());
        out
    })
}

// ---------------------------------------------------------------------
// T4: active -> agentless passive.
// ---------------------------------------------------------------------

/// Subjects recoverable without an explicit agent phrase — the boundary
/// between "safe to demote" (no information lost, since the reader
/// already knows who "we" or "the team" is) and "unsafe to demote" (an
/// identifiable, previously-unmentioned agent like "the vendor" carries
/// information the passive would erase).
///
/// Matched against the subject's own full span text (its subtree, not
/// just its head token). The prototype this is ported from compared
/// against a single token's surface text instead, which means its two
/// multi-word entries (`"the team"`, `"our team"`) could never match
/// anything — a single token's text is never a two-word string. Matching
/// the full span keeps every entry in the table reachable, rather than
/// silently porting a table half of which is dead.
const GENERIC_SUBJ: &[&str] = &["we", "i", "you", "they", "one", "the team", "our team"];

/// Active clause -> agentless passive (T4): `"We deployed the change"` ->
/// `"The change was deployed"`.
///
/// This transducer deliberately *increases* a construction (the agentless
/// passive) that LLM output under-produces relative to human writing,
/// rather than removing one — every other transducer in this module
/// moves a count down.
///
/// # Licensing conditions
/// Every one of these must hold, or a candidate verb is skipped entirely:
/// - the verb has both an `nsubj` and a `dobj` child (both required —
///   there is no subject-less or object-less passivization here)
/// - the subject's own text is one of [`GENERIC_SUBJ`]
/// - the verb has no `auxpass` child (the clause is not already passive)
/// - the verb is not itself a `conj`, and none of its children is a verb
///   attached by `conj` — passivising one conjunct of a coordinated
///   predicate (`"We instrumented X, and discovered Y"`) would strand the
///   other conjunct with no subject of its own; this guard is load-bearing,
///   not a defensive extra
///
/// `be` takes the tense of the original verb (`VBD` -> was/were, `VBZ`/
/// `VBP` -> is/are) and agrees in number with the promoted object.
#[must_use]
pub fn t4_activize_to_passive(
    source: &str,
    tokens: &[TaggedToken],
    parse: &SentenceParse,
) -> Vec<Candidate> {
    let mut out = Vec::new();

    for (index, edge) in parse.edges().iter().enumerate() {
        let tag = tokens[index].pos.as_str();
        if !matches!(tag, "VBD" | "VBZ" | "VBP") {
            continue;
        }
        if edge.relation == DepRelation::Conj {
            continue; // this verb is itself a conjunct; see the module docs.
        }
        if children_with_relation(parse, index, DepRelation::AuxPass)
            .next()
            .is_some()
        {
            continue; // already passive.
        }
        let strands_a_conjunct = children_with_relation(parse, index, DepRelation::Conj)
            .any(|child| coarse_tag(&tokens[child.token].pos).as_ref() == "VB");
        if strands_a_conjunct {
            continue;
        }

        let Some(subj) = children_with_relation(parse, index, DepRelation::Nsubj).next() else {
            continue;
        };
        let Some(obj) = children_with_relation(parse, index, DepRelation::Dobj).next() else {
            continue;
        };

        let (subj_first, subj_last) = subtree_span(parse, subj.token);
        let subj_text = span_text(source, tokens, subj_first, subj_last).to_lowercase();
        if !GENERIC_SUBJ.contains(&subj_text.as_str()) {
            continue; // demoting an identifiable agent loses information.
        }

        let (obj_first, obj_last) = subtree_span(parse, obj.token);
        let obj_text = span_text(source, tokens, obj_first, obj_last);
        let obj_surface = token_text(source, tokens, obj.token).to_lowercase();
        let plural = matches!(tokens[obj.token].pos.as_str(), "NNS" | "NNPS")
            || matches!(obj_surface.as_str(), "they" | "these" | "those");
        let be = match (tag == "VBD", plural) {
            (true, true) => "were",
            (true, false) => "was",
            (false, true) => "are",
            (false, false) => "is",
        };

        let replacement = format!(
            "{} {be} {}",
            recapitalize(obj_text),
            past_participle(tokens[index].lemma.as_ref())
        );

        let mut delta = BTreeMap::new();
        delta.insert("agentless_passive", 1);
        delta.insert("first_person", -1);

        out.push(Candidate {
            kind: CandidateKind::ActivizeToPassive,
            range: tokens[subj_first].token.range.start..tokens[obj_last].token.range.end,
            replacement: replacement.into_boxed_str(),
            delta,
            confidence: 0.8,
        });
    }

    out
}

// ---------------------------------------------------------------------
// T5: nominalization unpacking.
// ---------------------------------------------------------------------

/// Nominalized-noun -> verb table, ported verbatim (23 entries). Fixed
/// and closed by design: a general suffix-based nominalization detector
/// (`"-tion"`, `"-ment"`, ...) would also match nouns with no natural
/// verb reading in this register (`"nation"`, `"moment"`), so the
/// original implementation's own boundary keeps this a literal, audited
/// lookup table rather than a productive rule.
const NOMINAL_VERB: &[(&str, &str)] = &[
    ("optimization", "optimizing"),
    ("reduction", "reducing"),
    ("creation", "creating"),
    ("implementation", "implementing"),
    ("integration", "integrating"),
    ("migration", "migrating"),
    ("deployment", "deploying"),
    ("improvement", "improving"),
    ("allocation", "allocating"),
    ("utilization", "using"),
    ("configuration", "configuring"),
    ("validation", "validating"),
    ("verification", "verifying"),
    ("execution", "executing"),
    ("generation", "generating"),
    ("adoption", "adopting"),
    ("expansion", "expanding"),
    ("introduction", "introducing"),
    ("elimination", "eliminating"),
    ("consolidation", "consolidating"),
    ("degradation", "degrading"),
    ("compression", "compressing"),
    ("duplication", "duplicating"),
];

/// Nominalization unpacking (T5): `"the optimization of X"` -> `"optimizing X"`.
///
/// Keeps the sentence verbal without inventing an agent — the same
/// restraint [`t4_activize_to_passive`] shows in the other direction, by
/// refusing to demote an agent it can't recover.
///
/// # Licensing conditions
/// - the noun's own lowercase text is a key in [`NOMINAL_VERB`]
/// - it has a `det` child spelled exactly `"the"`
/// - it has a `prep` child spelled exactly `"of"`, and that `prep` token
///   itself has a `pobj` child — the argument the unpacked verb takes
///
/// Recapitalizes the replacement when the determiner opened the sentence
/// (token index `0`).
#[must_use]
pub fn t5_nominalization(
    source: &str,
    tokens: &[TaggedToken],
    parse: &SentenceParse,
) -> Vec<Candidate> {
    let mut out = Vec::new();

    for index in 0..tokens.len() {
        // Penn `NN`/`NNS` exactly, not `coarse_tag`'s truncated "NN":
        // `coarse_tag` would also match `NNP`/`NNPS`, but the reference
        // this table's suffix/length conditions were audited against
        // excludes proper nouns from nominalization.
        if !matches!(tokens[index].pos.as_str(), "NN" | "NNS") {
            continue;
        }
        let lower = token_text(source, tokens, index).to_lowercase();
        let Some(&(_, verb)) = NOMINAL_VERB.iter().find(|&&(noun, _)| noun == lower) else {
            continue;
        };

        let Some(det) = children_with_relation(parse, index, DepRelation::Det)
            .find(|edge| token_text(source, tokens, edge.token).eq_ignore_ascii_case("the"))
        else {
            continue;
        };
        let Some(of_prep) = children_with_relation(parse, index, DepRelation::Prep)
            .find(|edge| token_text(source, tokens, edge.token).eq_ignore_ascii_case("of"))
        else {
            continue;
        };
        let Some(pobj) = children_with_relation(parse, of_prep.token, DepRelation::Pobj).next()
        else {
            continue;
        };

        let (arg_first, arg_last) = subtree_span(parse, pobj.token);
        let arg_text = span_text(source, tokens, arg_first, arg_last);

        let mut replacement = format!("{verb} {arg_text}");
        if det.token == 0 {
            replacement = recapitalize(&replacement);
        }

        let mut delta = BTreeMap::new();
        delta.insert("nominalization", -1);
        delta.insert("prepositions", -1);

        out.push(Candidate {
            kind: CandidateKind::NominalizationUnpack,
            range: tokens[det.token].token.range.start..tokens[arg_last].token.range.end,
            replacement: replacement.into_boxed_str(),
            delta,
            confidence: 0.85,
        });
    }

    out
}

/// Every T4/T5 candidate for one sentence, in source order.
///
/// Selection between candidates (including any that overlap) is a
/// caller's decision; this only concatenates and sorts the two
/// transducers' output.
#[must_use]
pub fn candidates(source: &str, tokens: &[TaggedToken], parse: &SentenceParse) -> Vec<Candidate> {
    let mut out = t4_activize_to_passive(source, tokens, parse);
    out.extend(t5_nominalization(source, tokens, parse));
    out.sort_by_key(|candidate| (candidate.range.start, candidate.range.end));
    out
}

// ---------------------------------------------------------------------
// Inflection: ported verbatim from the same prototype. `third_sg` is not
// called by either transducer above (T4 only ever needs
// `past_participle`; T5 does no inflection at all, since `NOMINAL_VERB`
// already spells out its verb forms), but it is ported alongside `past`
// and `past_participle` because the three functions and their two tables
// were a single, audited unit in the original: separating `third_sg` out
// would leave the port incomplete in a way that isn't visible until a
// participial transducer needs it later.
// ---------------------------------------------------------------------

/// Endings after which the regular third-person-singular/plural suffix is
/// `"-es"` rather than a bare `"-s"`. `"o"` sits alongside the true
/// sibilants (`"go"` -> `"goes"`, `"do"` -> `"does"`) because English
/// spelling extends the same epenthetic vowel to a trailing `"o"`, not
/// because `"o"` is phonetically a sibilant.
const SIBILANT_ENDINGS: &[&str] = &["s", "x", "z", "ch", "sh", "o"];

/// Irregular past-tense forms (44 entries), ported verbatim. `lemma +
/// "ed"` produces a real-looking but wrong English word often enough —
/// `"holded"`, `"builded"`, `"catched"` all read as plausible tokens, not
/// obvious garbage — that a silent fallback to the regular rule would be
/// a worse failure mode than a closed table simply not covering a lemma
/// outside it.
const IRREGULAR_PAST: &[(&str, &str)] = &[
    ("be", "was"),
    ("have", "had"),
    ("do", "did"),
    ("make", "made"),
    ("take", "took"),
    ("give", "gave"),
    ("find", "found"),
    ("run", "ran"),
    ("hold", "held"),
    ("lead", "led"),
    ("keep", "kept"),
    ("leave", "left"),
    ("mean", "meant"),
    ("send", "sent"),
    ("build", "built"),
    ("bring", "brought"),
    ("buy", "bought"),
    ("catch", "caught"),
    ("cut", "cut"),
    ("put", "put"),
    ("set", "set"),
    ("let", "let"),
    ("read", "read"),
    ("write", "wrote"),
    ("drive", "drove"),
    ("rise", "rose"),
    ("fall", "fell"),
    ("grow", "grew"),
    ("show", "showed"),
    ("see", "saw"),
    ("get", "got"),
    ("go", "went"),
    ("come", "came"),
    ("become", "became"),
    ("begin", "began"),
    ("choose", "chose"),
    ("draw", "drew"),
    ("feed", "fed"),
    ("meet", "met"),
    ("pay", "paid"),
    ("sell", "sold"),
    ("spend", "spent"),
    ("lose", "lost"),
    ("win", "won"),
    ("think", "thought"),
    ("teach", "taught"),
];

/// Irregular past-participle forms (44 entries), ported verbatim as a
/// table separate from [`IRREGULAR_PAST`] rather than derived from it:
/// several lemmas' past tense and past participle diverge (`"write"` ->
/// `"wrote"` / `"written"`, `"drive"` -> `"drove"` / `"driven"`, `"rise"`
/// -> `"rose"` / `"risen"`), so collapsing the two tables into one would
/// silently reuse the wrong form for exactly those verbs.
const IRREGULAR_PAST_PARTICIPLE: &[(&str, &str)] = &[
    ("be", "been"),
    ("have", "had"),
    ("do", "done"),
    ("make", "made"),
    ("take", "taken"),
    ("give", "given"),
    ("find", "found"),
    ("run", "run"),
    ("hold", "held"),
    ("lead", "led"),
    ("keep", "kept"),
    ("leave", "left"),
    ("mean", "meant"),
    ("send", "sent"),
    ("build", "built"),
    ("bring", "brought"),
    ("buy", "bought"),
    ("catch", "caught"),
    ("cut", "cut"),
    ("put", "put"),
    ("set", "set"),
    ("let", "let"),
    ("read", "read"),
    ("write", "written"),
    ("drive", "driven"),
    ("rise", "risen"),
    ("fall", "fallen"),
    ("grow", "grown"),
    ("show", "shown"),
    ("see", "seen"),
    ("get", "got"),
    ("go", "gone"),
    ("come", "come"),
    ("become", "become"),
    ("begin", "begun"),
    ("choose", "chosen"),
    ("draw", "drawn"),
    ("feed", "fed"),
    ("meet", "met"),
    ("pay", "paid"),
    ("sell", "sold"),
    ("spend", "spent"),
    ("lose", "lost"),
    ("win", "won"),
    ("think", "thought"),
    ("teach", "taught"),
];

/// `lemma` ends in `"y"` preceded by a consonant (so `"y"` -> `"i"`
/// before a vowel suffix: `"carry"` -> `"carries"`/`"carried"`), as
/// opposed to a vowel, which keeps the `"y"` (`"play"` ->
/// `"plays"`/`"played"`).
fn ends_with_consonant_y(lemma: &str) -> bool {
    let chars: Vec<char> = lemma.chars().collect();
    chars.len() > 1 && chars[chars.len() - 1] == 'y' && !is_vowel(chars[chars.len() - 2])
}

const fn is_vowel(c: char) -> bool {
    matches!(c, 'a' | 'e' | 'i' | 'o' | 'u')
}

/// The regular third-person-singular present / plural form of `lemma`.
///
/// (`"deploy"` -> `"deploys"`, `"watch"` -> `"watches"`, `"carry"` ->
/// `"carries"`). Not consulted by any irregular table — the reference
/// this is ported from never called it for third-person-singular
/// irregulars either, relying on [`IRREGULAR_PAST`]/
/// [`IRREGULAR_PAST_PARTICIPLE`] only for the past forms it actually
/// needed.
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

/// The past-tense form of `lemma`: the irregular table's entry if one
/// exists, otherwise the regular `"-d"`/`"-ied"`/`"-ed"` suffix rule.
#[must_use]
pub fn past(lemma: &str) -> String {
    if let Some(&(_, irregular)) = IRREGULAR_PAST.iter().find(|&&(base, _)| base == lemma) {
        return irregular.to_string();
    }
    if lemma.ends_with('e') {
        format!("{lemma}d")
    } else if ends_with_consonant_y(lemma) {
        format!("{}ied", &lemma[..lemma.len() - 1])
    } else {
        format!("{lemma}ed")
    }
}

/// The past-participle form of `lemma`: the irregular table's entry if
/// one exists, otherwise the same form [`past`] produces (true for every
/// regular English verb).
#[must_use]
pub fn past_participle(lemma: &str) -> String {
    IRREGULAR_PAST_PARTICIPLE
        .iter()
        .find(|&&(base, _)| base == lemma)
        .map_or_else(|| past(lemma), |&(_, participle)| participle.to_string())
}
