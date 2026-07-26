//! Rewrite transducers ported from a Python register-rewriting
//! prototype.
//!
//! Each proposes a candidate edit predicting an exact per-feature count
//! delta, with a confidence and the byte range it would replace. Five
//! transducers existed; only two are ported.
//!
//! [`t4_activize_to_passive`]/[`t5_nominalization`] depend on
//! `nsubj`/`dobj`/`det`/`prep`/`pobj`, resolved at 82-96% accuracy. The
//! other three depend on `acl`/`advcl`, resolved at only 52-58% —
//! firing wrongly half the time is worse than not firing, so they stay
//! unported.
//!
//! Functions here only propose candidates; none mutate `source`, choose
//! between overlaps, or apply anything — a caller's decision.

use std::collections::BTreeMap;
use std::ops::Range;

use friction_core::CoreError;
use friction_core::span::{Spanned, validate_range};
use friction_nlp::{
    DepEdge, DepRelation, FINITE_VERB_TAGS, SentenceParse, TaggedToken, coarse_tag,
};

/// A proposed rewrite: replace `range` with `replacement`, moving the
/// Biber feature counts in `delta` by the listed amount.
///
/// Deliberately separate from `friction_core::Patch` despite the same
/// shape: a `Patch` is a decision already applied, a `Candidate` is
/// unselected — merging them would make "chosen yet" untyped.
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    /// Which transducer produced this candidate.
    pub kind: CandidateKind,
    /// Byte range in the original source this candidate would replace.
    pub range: Range<usize>,
    /// The text to substitute for `range`.
    pub replacement: Box<str>,
    /// The Biber feature-count changes this candidate predicts, keyed
    /// by name. A [`BTreeMap`], not `HashMap` — alphabetical-by-name is
    /// a fixed, meaningful order, which this workspace always prefers.
    pub delta: BTreeMap<&'static str, i32>,
    /// Hand-set trust in this rewrite class, `[0.0, 1.0]`. A plain
    /// `f32`, not [`friction_nlp::Confidence`] — that type means a
    /// parse edge's margin over its alternative, not a fixed
    /// per-transducer level.
    pub confidence: f32,
}

impl Candidate {
    /// Validates `range` against `source`: in bounds, and both endpoints
    /// on a UTF-8 boundary. Mirrors `friction_core::Patch::validate` —
    /// same span-honesty obligation, same check.
    ///
    /// # Errors
    /// Returns [`CoreError`] if `range` is out of bounds or splits a
    /// UTF-8 character.
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
// well-formed tree, so a walk down from any token always terminates.
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
/// Not necessarily `root` itself: a subtree can extend left of its head
/// (a determiner) or right of it (a prepositional phrase), so this is
/// the min/max index reached, not `root`'s own.
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

/// Source text spanning token `first` through `last`, inclusive. Built
/// from the same per-token ranges every span here is validated against,
/// so it is byte-honest by construction, not by a separate check.
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

/// Subjects recoverable without an agent phrase: the reader already
/// knows who "we"/"the team" is, unlike an identifiable agent ("the
/// vendor") whose demotion erases information.
///
/// Matched against the subject's full span, not just its head token —
/// the prototype compared single tokens, so its multi-word entries
/// (`"the team"`, `"our team"`) could never match. `they` is
/// deliberately absent: it's anaphoric to an earlier antecedent, unlike
/// `we`/`i`/`you`/`one`, which are non-referential.
///
/// Measured on real prose: `"Browse the list of programmed channels to
/// ensure they match the uploaded file"` satisfies every condition, and
/// passivizing yields `"...to ensure the uploaded file is matched"`,
/// quietly dropping that *the channels* must match — a meaning change
/// this transducer isn't permitted to make.
const GENERIC_SUBJ: &[&str] = &["we", "i", "you", "one", "the team", "our team"];

/// Reflexive pronouns: never a licensed passive subject.
///
/// "I ... tore myself away" promotes to "Myself was torn away" —
/// ungrammatical, since a reflexive has no referent independent of the
/// subject the passive deletes. No table can paper over this; the
/// transducer refuses to fire. Personal pronouns are never a licensed
/// object to promote.
///
/// Two reasons: a ditransitive verb's indirect object is often
/// mislabeled `dobj`, promoting the wrong participant, and a pronoun
/// carries almost no information to promote. Measured on real prose:
/// `"see how much time and effort they save you"` passivizes to
/// `"...you are saved"` — the beneficiary is promoted while the real
/// object (`time and effort`) is stranded.
///
/// Separate from [`REFLEXIVE_OBJ`]: a reflexive is refused as grammar,
/// a personal pronoun as judgement — merging would hide that
/// difference.
const PERSONAL_PRONOUN_OBJ: &[&str] = &["me", "us", "you", "him", "her", "it", "them", "'em"];

const REFLEXIVE_OBJ: &[&str] = &[
    "myself",
    "yourself",
    "himself",
    "herself",
    "itself",
    "oneself",
    "ourselves",
    "yourselves",
    "themselves",
];

/// `true` if `text` contains a character that only ever appears as
/// Markdown structural syntax here (emphasis asterisks, link/reference
/// brackets, inline-code backticks).
///
/// Measured on real prose: a bracketed, bolded placeholder is bridged
/// into one prose run rather than excluded, so its literal markup
/// characters reach the dependency parser, which has no training signal
/// for Markdown. Refusing any span containing them is cheaper than
/// parsing Markdown correctly.
fn contains_markdown_structural_syntax(text: &str) -> bool {
    text.contains(['*', '[', ']', '`'])
}

/// `true` if `pos` is a tag a promoted passive subject can plausibly
/// be: a common or proper noun, a pronoun, or a number.
///
/// Guards against a mislabeled predicative complement — measured on
/// real prose where "felt most productive" (`JJ`) and "inch closer"
/// (`RBR`) were both accepted as objects, producing "Most productive
/// was feeled" and "Closer is inched".
fn is_plausible_object_pos(pos: &str) -> bool {
    matches!(pos, "NN" | "NNS" | "NNP" | "NNPS" | "PRP" | "CD")
}

/// `true` if the span from token `first` to token `last` sits inside an
/// unclosed `(` or `[` earlier in `source`, or crosses either bracket.
///
/// Depth-counted, not parity-counted like the straight-quote case
/// elsewhere: brackets are directional, so nesting has to be tracked
/// rather than assumed away.
fn within_bracketed_aside(source: &str, tokens: &[TaggedToken], first: usize, last: usize) -> bool {
    let start = tokens[first].token.range.start;
    let end = tokens[last].token.range.end;
    if source[start..end].contains(['(', ')', '[', ']']) {
        return true;
    }
    let mut round = 0i32;
    let mut square = 0i32;
    for c in source[..start].chars() {
        match c {
            '(' => round += 1,
            ')' => round -= 1,
            '[' => square += 1,
            ']' => square -= 1,
            _ => {}
        }
    }
    round > 0 || square > 0
}

/// Stative or linking-verb lemmas: never a licensed passive.
///
/// "Have" and "become"/"remain"/"seem"/"appear" are copulas taking a
/// predicate-nominal complement, not a true object: forcing the
/// transform produces nonsense ("about a day of downtime was had",
/// "problems are become").
///
/// "Want"/"wish"/"prefer"/"need" differ: ordinary transitive verbs, so
/// the transform is legal, but a passive of a desire reads bureaucratic
/// and loses who wanted it ("the surprising benefits we've found were
/// wanted", "Postgres' fine-grained control was preferred" — both
/// grammatical, both worse).
///
/// The distinction matters if revisited: the first group cannot be
/// passivized, the second should not be.
const STATIVE_OR_LINKING_LEMMA: &[&str] = &[
    "have", "become", "remain", "seem", "appear", "want", "wish", "prefer", "need",
];

/// Active clause -> agentless passive (T4): `"We deployed the change"` ->
/// `"The change was deployed"`.
///
/// Deliberately *increases* the agentless passive, which LLM output
/// under-produces, rather than removing a construction like every other
/// transducer here.
///
/// # Licensing conditions
/// All required, or the candidate verb is skipped:
/// - the verb has both an `nsubj` and a `dobj` child
/// - the subject's text is one of [`GENERIC_SUBJ`]
/// - the verb has no `auxpass` child (not already passive)
/// - the verb is not itself a `conj`, and none of its children is a verb
///   attached by `conj` — passivizing one conjunct would strand the
///   other with no subject (`"We instrumented X, and discovered Y"`)
///
/// `be` takes the original verb's tense and agrees in number with the
/// promoted object.
///
/// Documents [`t4_activize_to_passive`] as a whole; kept here, with
/// [`object_is_licensable`]'s docs, as one connected licensing check.
fn verb_is_licensable(
    source: &str,
    tokens: &[TaggedToken],
    parse: &SentenceParse,
    index: usize,
    edge: &DepEdge,
) -> bool {
    if edge.relation == DepRelation::Conj {
        return false; // this verb is itself a conjunct; see the module docs.
    }
    if children_with_relation(parse, index, DepRelation::AuxPass)
        .next()
        .is_some()
    {
        return false; // already passive.
    }
    let strands_a_conjunct = children_with_relation(parse, index, DepRelation::Conj)
        .any(|child| coarse_tag(&tokens[child.token].pos).as_ref() == "VB");
    if strands_a_conjunct {
        return false;
    }
    let lemma = tokens[index].lemma.as_ref();
    if STATIVE_OR_LINKING_LEMMA.contains(&lemma) {
        // Never passivizes idiomatically; see
        // STATIVE_OR_LINKING_LEMMA's own docs.
        return false;
    }
    if lemma.ends_with("ed") && !IRREGULAR_PAST.iter().any(|&(base, _)| base == lemma) {
        // A base-form verb essentially never ends in "-ed" ("embed" is
        // the rare exception, not worth the false negatives avoiding it
        // would cost); a lemma that does is almost always an unreduced
        // surface form ("pinpointed", "absorbed") the tagger failed to
        // lemmatize. Suffixing it again produced "pinpointeded" and
        // "absorbeded" against real prose — a made-up word, not merely
        // awkward. Refusing beats compounding a known upstream
        // inaccuracy.
        return false;
    }
    let tag = tokens[index].pos.as_str();
    let surface_lower = token_text(source, tokens, index).to_lowercase();
    if matches!(tag, "VBD" | "VBZ")
        && lemma.eq_ignore_ascii_case(&surface_lower)
        && !IRREGULAR_PAST.iter().any(|&(base, _)| base == lemma)
    {
        // A `VBD`/`VBZ` surface form is never identical to its base
        // lemma for a regular verb — unlike `VBP` ("we deploy"), where
        // surface == lemma legitimately. A `VBD`/`VBZ` token whose
        // lemma matches surface, and isn't a known invariant verb
        // (`cut`, `put`, `hit`, ...), signals the tagger failed to
        // reduce an irregular verb ("rewrote") to its base at all.
        // Suffixing the unreduced form produced "rewroted" against real
        // prose; refusing is safer than compounding the error.
        return false;
    }
    true
}

/// `true` if `obj_token`'s subtree (`obj_first..=obj_last`) is safe to
/// promote to subject position.
fn object_is_licensable(
    source: &str,
    tokens: &[TaggedToken],
    parse: &SentenceParse,
    verb_token: usize,
    obj_token: usize,
    obj_first: usize,
    obj_last: usize,
) -> bool {
    // The object must be a bare noun phrase, no post-modifier — the
    // largest source of bad output this transducer produces, and silent
    // meaning change rather than ungrammaticality, which no later gate
    // catches.
    //
    // A trailing prepositional or infinitival modifier may attach to
    // the verb, not the noun, so moving the whole subtree rewrites the
    // claim: `"inspected each board for knots and defects"` became
    // `"each board for knots and defects was inspected"`, dropping that
    // the inspection was *for* knots.
    //
    // Checked by relation, not a trailing preposition token, since the
    // failing cases include infinitival `to`. `Other` is permitted: it
    // collapses compounds, numerals and possessives, all noun-internal
    // and safe to move.
    let has_post_modifier = (obj_first..=obj_last).any(|index| {
        index != obj_token
            && parse.edge(index).is_some_and(|edge| {
                matches!(
                    edge.relation,
                    DepRelation::Prep
                        | DepRelation::Acl
                        | DepRelation::Advcl
                        | DepRelation::Xcomp
                        | DepRelation::Ccomp
                        | DepRelation::Csubj
                        | DepRelation::Mark
                        | DepRelation::Aux
                        | DepRelation::AuxPass
                )
            })
    });
    if has_post_modifier {
        return false;
    }

    // The object must contain at least one actual noun.
    //
    // A single head tag is exactly what a tagger is likeliest to get
    // wrong on an unusual span. Requiring a noun anywhere in the
    // subtree survives one bad tag: `"Relatively low-impact and
    // predictable"` was promoted on a mistagged head, producing
    // "Relatively low-impact and predictable was started."
    let has_noun = (obj_first..=obj_last)
        .any(|index| matches!(tokens[index].pos.as_str(), "NN" | "NNS" | "NNP" | "NNPS"));
    if !has_noun {
        return false;
    }

    // Never rewrite inside a bracketed or parenthesised aside: square
    // brackets mark a fill-in-the-blank placeholder, a parenthetical is
    // scoped to itself, and rewriting either reads as broken even when
    // locally correct (`"[Specific cabinet styles ... e.g., soft-close
    // drawers are desired, ...]"`, `"(Sketch 1 is envisioned)"`).
    //
    // Same reasoning as the guard against quoted text: the author is
    // displaying that material, not asserting it.
    if within_bracketed_aside(source, tokens, obj_first, obj_last) {
        return false;
    }

    // No preposition may stand between the verb and its object: a
    // preposition in that gap means the noun is really the object of
    // the *preposition*, mislabelled by the parser.
    //
    // Measured on real prose: `"As we continue down this path"` parsed
    // `path` as `continue`'s direct object, when it's really `down`'s.
    // Passivizing gave `"As this path is continued"` — grammatical, and
    // worse, applied to a relation that wasn't there.
    //
    // Particles (`RP`) are rejected too: `"we continue down this path"`
    // and `"we gave up the plan"` tag identically (`RP`, `dobj`), so no
    // structural rule separates them. Cost: the good phrasal passive
    // (`"the server was set up"`) — measured at three rewrites over the
    // corpus, all the stilted case this guard removes.
    let preposition_before_object = (verb_token + 1..obj_first).any(|index| {
        tokens
            .get(index)
            .is_some_and(|t| matches!(t.pos.as_str(), "IN" | "RP"))
    });
    if preposition_before_object {
        return false;
    }

    // Nothing that belongs to the verb may sit after the object: the
    // rewrite leaves the rest of the sentence untouched, safe only when
    // the object ends its clause. `"We encourage contributions to
    // Lumen"` became `"Contributions are encouraged to Lumen"`, where
    // `to Lumen` now reads as the destination of the encouraging, not
    // the contributions.
    //
    // Mirrors the post-modifier check above (refuses to *leave* what
    // that one refuses to *move*). Punctuation is exempt.
    let verb_child_after_object = parse.edges().iter().any(|edge| {
        edge.head == Some(verb_token)
            && edge.token > obj_last
            && edge.relation != DepRelation::Punct
            && tokens
                .get(edge.token)
                .is_some_and(|t| t.token.kind != friction_core::TokenKind::Punctuation)
    });
    if verb_child_after_object {
        return false;
    }
    let obj_surface = token_text(source, tokens, obj_token).to_lowercase();
    if REFLEXIVE_OBJ.contains(&obj_surface.as_str()) {
        return false; // no independent referent to promote; see REFLEXIVE_OBJ's own docs.
    }
    if PERSONAL_PRONOUN_OBJ.contains(&obj_surface.as_str()) {
        return false; // see PERSONAL_PRONOUN_OBJ's own docs.
    }
    if !is_plausible_object_pos(tokens[obj_token].pos.as_str()) {
        return false; // not a nominal; see is_plausible_object_pos's own docs.
    }

    // The object's subtree must not contain a finite verb, nor may the
    // token right after it — both guard the parser flattening an
    // embedded clause into a `dobj`: "we anticipate this reporting
    // workload will continue to grow" promoted the object, stranding
    // "will continue to grow" with no subject; a `dobj` subtree also
    // swallowed a full relative-clause chain elsewhere.
    let subtree_has_finite_verb =
        (obj_first..=obj_last).any(|index| FINITE_VERB_TAGS.contains(&tokens[index].pos.as_str()));
    if subtree_has_finite_verb {
        return false;
    }
    if tokens
        .get(obj_last + 1)
        .is_some_and(|next| FINITE_VERB_TAGS.contains(&next.pos.as_str()))
    {
        return false;
    }
    if matches!(tokens[obj_last].pos.as_str(), "IN" | "TO") {
        // A bare trailing preposition or infinitival "to" never ends a
        // genuine noun phrase: "a rough budget range **of**" left its
        // argument attached elsewhere, producing "... range of was
        // established" once promoted. Mirrors `t5_nominalization`'s
        // stranded-tail guard, on the object side.
        return false;
    }
    true
}

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
        if !verb_is_licensable(source, tokens, parse, index, edge) {
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
        if !object_is_licensable(source, tokens, parse, index, obj.token, obj_first, obj_last) {
            continue;
        }

        let obj_text = span_text(source, tokens, obj_first, obj_last);
        if contains_markdown_structural_syntax(span_text(source, tokens, subj_first, obj_last)) {
            continue; // see contains_markdown_structural_syntax's own docs.
        }
        let obj_surface = token_text(source, tokens, obj.token).to_lowercase();

        // "you" always takes plural agreement regardless of referent
        // count — measured where "you" as a promoted object produced
        // "You is encouraged" three times. "them" belongs with
        // "they"/"these"/"those" but was missing, producing "Them is
        // plucked".
        let plural = obj_surface == "you"
            || matches!(tokens[obj.token].pos.as_str(), "NNS" | "NNPS")
            || matches!(obj_surface.as_str(), "they" | "these" | "those" | "them");
        let be = match (tag == "VBD", plural) {
            (true, true) => "were",
            (true, false) => "was",
            (false, true) => "are",
            (false, false) => "is",
        };

        // Recapitalize only when this candidate's range opens the
        // sentence — unconditional recapitalization was a real bug:
        // most firings passivize a subordinate clause ("... when I
        // found an exception" -> "... when An exception was found").
        let promoted = if subj_first == 0 {
            recapitalize(obj_text)
        } else {
            obj_text.to_string()
        };
        let replacement = format!(
            "{promoted} {be} {}",
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

/// The complete set of function words either transducer may introduce
/// beyond the matched span or
/// [`past_participle`]/[`past`]/[`third_sg`].
///
/// Closed by design, for `friction-edit`'s register closure gate: every
/// "be" form [`t4_activize_to_passive`] can emit and nothing else.
/// Widening this silently would widen what the gate accepts without
/// anyone auditing the new word.
pub const PERMITTED_FUNCTION_WORDS: [&str; 4] = ["was", "were", "is", "are"];

/// The verb [`t5_nominalization`] would substitute for
/// `nominalization_lower` (already lowercased).
///
/// The read side of [`NOMINAL_VERB`], exposed so a caller validating
/// closure consults the same lookup rather than a second,
/// independently-maintained copy.
///
#[must_use]
pub fn nominal_verb_for(nominalization_lower: &str) -> Option<&'static str> {
    NOMINAL_VERB
        .iter()
        .find(|&&(noun, _)| noun == nominalization_lower)
        .map(|&(_, verb)| verb)
}

/// Nominalized-noun -> verb table, ported verbatim (23 entries). Fixed
/// and closed: a suffix detector (`"-tion"`, `"-ment"`) would also
/// match nouns with no verb reading (`"nation"`, `"moment"`), so this
/// stays a literal, audited table.
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
/// Keeps the sentence verbal without inventing an agent, the same
/// restraint [`t4_activize_to_passive`] shows in reverse.
///
/// # Licensing conditions
/// - the noun's lowercase text is a key in [`NOMINAL_VERB`]
/// - it has a `det` child spelled exactly `"the"`
/// - it has a `prep` child spelled `"of"`, whose `pobj` child is the
///   argument the unpacked verb takes
///
/// Recapitalizes when the determiner opened the sentence.
#[must_use]
pub fn t5_nominalization(
    source: &str,
    tokens: &[TaggedToken],
    parse: &SentenceParse,
) -> Vec<Candidate> {
    let mut out = Vec::new();

    for index in 0..tokens.len() {
        // Penn `NN`/`NNS` exactly, not `coarse_tag`'s truncated "NN",
        // which would also match `NNP`/`NNPS`: the reference this table
        // was audited against excludes proper nouns from
        // nominalization.
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

        // A modified nominalization has nowhere to put its modifier:
        // the rewrite deletes an adjective sitting between determiner
        // and argument — `"The seamless integration of SQS with other
        // AWS services"` became `"integrating SQS with other AWS
        // services"`, losing `seamless` outright, the one failure mode
        // this module must never produce.
        //
        // Carrying it across isn't an option either: the verbal form
        // needs a synthesized adverb (`seamlessly integrating`), a word
        // the input never contained. So the construction is declined.
        if children_with_relation(parse, index, DepRelation::Amod)
            .next()
            .is_some()
        {
            continue;
        }
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

        // Guard against a mis-attached compound-noun tail: "the
        // integration of the third-party analytics SDK" had its final
        // noun ("SDK") attached outside the `pobj` subtree, so a
        // subtree-only span silently dropped it — "integrating the
        // third-party analytics" is missing its object. A genuine
        // subtree boundary is never immediately followed by a bare
        // noun.
        if tokens
            .get(arg_last + 1)
            .is_some_and(|next| matches!(next.pos.as_str(), "NN" | "NNS" | "NNP" | "NNPS"))
        {
            continue;
        }

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
/// Selection between candidates, including overlapping ones, is a
/// caller's decision — this only concatenates and sorts both outputs.
#[must_use]
pub fn candidates(source: &str, tokens: &[TaggedToken], parse: &SentenceParse) -> Vec<Candidate> {
    let mut out = t4_activize_to_passive(source, tokens, parse);
    out.extend(t5_nominalization(source, tokens, parse));
    out.sort_by_key(|candidate| (candidate.range.start, candidate.range.end));
    out
}

// ---------------------------------------------------------------------
// Inflection: ported verbatim from the same prototype. `third_sg` is
// unused by either transducer (T5's `NOMINAL_VERB` spells out its own
// forms), but ships alongside `past`/`past_participle` since the three
// were one audited unit — splitting it out would leave the port
// incomplete until a participial transducer needs it.
// ---------------------------------------------------------------------

/// Endings after which the regular third-person-singular suffix is
/// `"-es"`, not `"-s"`. `"o"` joins the true sibilants (`"go"` ->
/// `"goes"`) because English spelling extends the same epenthetic vowel
/// to a trailing `"o"`, not because it's phonetically a sibilant.
const SIBILANT_ENDINGS: &[&str] = &["s", "x", "z", "ch", "sh", "o"];

/// Irregular past-tense forms (53 entries — the original 44 plus 9
/// invariant verbs missed initially: "hit", "cost", "cast", "shut",
/// "spread", "hurt", "quit", "burst", "shed", caught producing
/// "hited"). `lemma + "ed"` produces real-looking but wrong words often
/// enough — `"holded"`, `"builded"`, `"catched"` — that falling back
/// silently would be worse than a closed table simply missing a lemma.
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
    ("hit", "hit"),
    ("cost", "cost"),
    ("cast", "cast"),
    ("shut", "shut"),
    ("spread", "spread"),
    ("hurt", "hurt"),
    ("quit", "quit"),
    ("burst", "burst"),
    ("shed", "shed"),
    ("write", "wrote"),
    // Prefixed compounds of an already-listed irregular base, common in
    // technical prose ("rewrote the migration script") but missing from
    // the original 44-entry port; each inflects exactly like its base.
    ("rewrite", "rewrote"),
    ("overwrite", "overwrote"),
    ("undergo", "underwent"),
    ("override", "overrode"),
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

/// Irregular past-participle forms (53 entries, same addition as
/// [`IRREGULAR_PAST`]), kept separate rather than derived: several
/// lemmas' past tense and participle diverge (`"write"` ->
/// `"wrote"`/`"written"`, `"drive"` -> `"drove"`/`"driven"`), so
/// merging would silently reuse the wrong form.
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
    ("hit", "hit"),
    ("cost", "cost"),
    ("cast", "cast"),
    ("shut", "shut"),
    ("spread", "spread"),
    ("hurt", "hurt"),
    ("quit", "quit"),
    ("burst", "burst"),
    ("shed", "shed"),
    ("write", "written"),
    ("rewrite", "rewritten"),
    ("overwrite", "overwritten"),
    ("undergo", "undergone"),
    ("override", "overridden"),
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

/// `lemma` ends in `"y"` preceded by a consonant, so `"y"` -> `"i"`
/// before a vowel suffix (`"carry"` -> `"carries"`/`"carried"`), unlike
/// a preceding vowel, which keeps the `"y"` (`"play"` ->
/// `"plays"`/`"played"`).
fn ends_with_consonant_y(lemma: &str) -> bool {
    let chars: Vec<char> = lemma.chars().collect();
    chars.len() > 1 && chars[chars.len() - 1] == 'y' && !is_vowel(chars[chars.len() - 2])
}

const fn is_vowel(c: char) -> bool {
    matches!(c, 'a' | 'e' | 'i' | 'o' | 'u')
}

/// The regular third-person-singular present form of `lemma`
/// (`"deploy"` -> `"deploys"`, `"carry"` -> `"carries"`).
///
/// Consults no irregular table: the reference this is ported from used
/// [`IRREGULAR_PAST`]/[`IRREGULAR_PAST_PARTICIPLE`] for past forms
/// only.
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

/// The past-tense form of `lemma`.
///
/// The irregular table's entry if one exists, otherwise the regular
/// suffix rule, delegated to [`friction_nlp::inflect`] rather than
/// reimplemented here.
///
/// This module's own suffix rule originally had no consonant-doubling
/// logic (`"prefer"` -> `"prefered"` instead of `"preferred"`);
/// `inflect`'s `generate_past` already handles it and is tested.
/// `inflect` also checks its own overlapping irregular table first, an
/// independent safety net alongside [`IRREGULAR_PAST`].
#[must_use]
pub fn past(lemma: &str) -> String {
    if let Some(&(_, irregular)) = IRREGULAR_PAST.iter().find(|&&(base, _)| base == lemma) {
        return irregular.to_string();
    }
    friction_nlp::inflect("used", lemma).unwrap_or_else(|| format!("{lemma}ed"))
}

/// The past-participle form of `lemma`: the irregular table's entry if
/// one exists, otherwise the same form [`past`] produces (true for
/// every regular English verb).
#[must_use]
pub fn past_participle(lemma: &str) -> String {
    IRREGULAR_PAST_PARTICIPLE
        .iter()
        .find(|&&(base, _)| base == lemma)
        .map_or_else(|| past(lemma), |&(_, participle)| participle.to_string())
}
