//! Rewrite transducers for the register pass.
//!
//! Each proposes a candidate edit predicting an exact per-feature count
//! delta, with a confidence and the byte range it would replace. Two of
//! a possible five register transducers are implemented — the cut is
//! parser accuracy, not effort:
//!
//! [`t4_activize_to_passive`]/[`t5_nominalization`] depend on
//! `nsubj`/`dobj`/`det`/`prep`/`pobj`, resolved at 82-96% accuracy. The
//! other three candidates would depend on `acl`/`advcl`, resolved at
//! only 52-58%: firing wrongly half the time is worse than not firing,
//! so they don't exist.
//!
//! [`t6_em_dash`]/[`t7_semicolon`] are later additions: neither em
//! dashes nor semicolon splices came up in the research phase's
//! Biber-feature work, only later as measured Claude-family tells (see
//! `register-v1.toml`'s `[features.em_dash]`/`[features.semicolon]` and
//! `docs/research/FRONTIER_MODELS.md`). [`t7_semicolon`] reuses
//! [`independent_clause_follows`], T6's own subtree-anchored
//! independent-clause check, unchanged: the two features license the
//! same rewrite shape on the same evidence, just for different
//! punctuation.
//!
//! Functions here only propose candidates; none mutate `source`, choose
//! between overlaps, or apply anything: a caller's decision.

use std::collections::BTreeMap;
use std::ops::Range;

use friction_core::CoreError;
use friction_core::span::{Spanned, validate_range};
use friction_nlp::{
    DepEdge, DepRelation, FINITE_VERB_TAGS, SentenceParse, TaggedToken, coarse_tag, has_finite_verb,
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
    /// `f32`, not [`friction_nlp::Confidence`]: that type means a
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
    /// Produced by [`t6_em_dash`].
    EmDash,
    /// Produced by [`t7_semicolon`].
    Semicolon,
}

// ---------------------------------------------------------------------
// Subtree helpers. `SentenceParse` guarantees one edge per token and a
// well-formed tree, so a walk down from any token always terminates.
// ---------------------------------------------------------------------

/// Every edge in `parse` whose head is `head` and whose relation is
/// `relation`, in token order.
///
/// Looks children up via [`SentenceParse::children_of`] (`O(head`'s own
/// child count`)`) rather than scanning every edge in the sentence.
fn children_with_relation(
    parse: &SentenceParse,
    head: usize,
    relation: DepRelation,
) -> impl Iterator<Item = &DepEdge> {
    parse
        .children_of(head)
        .iter()
        .filter_map(move |&child| parse.edge(child))
        .filter(move |edge| edge.relation == relation)
}

/// Every token index in `root`'s own subtree: `root` itself, plus every
/// token reachable by repeatedly following `head` edges down from it.
///
/// Walks via [`SentenceParse::children_of`], an `O(1)` lookup per node,
/// instead of re-scanning every edge in the sentence per frontier node —
/// the previous shape made a candidate's subtree walk `O(tokens^2)`
/// (`O(tokens)` frontier nodes, each rescanning all `O(tokens)` edges);
/// this makes it `O(tokens)` total, since a tree's total parent-child
/// links equal its token count.
fn subtree_indices(parse: &SentenceParse, root: usize) -> Vec<usize> {
    let mut collected = vec![root];
    let mut frontier = vec![root];
    while let Some(current) = frontier.pop() {
        for &child in parse.children_of(current) {
            collected.push(child);
            frontier.push(child);
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

/// Subject-pronoun contractions: one token fusing a subject AND a finite
/// auxiliary (`it's` = `it is`). The shipped tagger keeps these whole and
/// tags them `PRP`, so a tag-based finite-verb scan is blind to the
/// finite verb inside — measured on real prose: "None of it is wrong,
/// exactly — it's machine register …" was comma-spliced because `it's`
/// read as a bare pronoun. Case-folded, straight-apostrophe forms; match
/// after folding U+2019.
const SUBJECT_CONTRACTIONS: &[&str] = &[
    "it's", "that's", "there's", "here's", "he's", "she's", "what's", "who's", "they're", "we're",
    "you're", "i'm", "i've", "they've", "we've", "you've", "i'll", "he'll", "she'll", "it'll",
    "we'll", "they'll", "you'll", "i'd", "he'd", "she'd", "it'd", "we'd", "they'd", "you'd",
];

/// Negated auxiliary contractions: a finite verb (plus its negation) in
/// one token, no subject fused. Same tagger blindness as
/// [`SUBJECT_CONTRACTIONS`] — `aren't` can surface untagged as a verb.
const NEGATED_AUX_CONTRACTIONS: &[&str] = &[
    "isn't",
    "aren't",
    "wasn't",
    "weren't",
    "don't",
    "doesn't",
    "didn't",
    "won't",
    "can't",
    "couldn't",
    "shouldn't",
    "wouldn't",
    "hasn't",
    "haven't",
    "hadn't",
    "ain't",
];

/// Token `index`'s surface, case-folded with curly apostrophes
/// straightened: the shape both contraction lists are written in.
fn folded_token_text(source: &str, tokens: &[TaggedToken], index: usize) -> String {
    token_text(source, tokens, index)
        .to_lowercase()
        .replace('\u{2019}', "'")
}

/// [`has_finite_verb`] plus contraction awareness: `true` if any token is
/// tag-finite OR is a contraction that embeds a finite auxiliary. The
/// widening is deliberately local to this module's transducers. The edit
/// gates keep the strict tag-based check they were calibrated with.
fn has_finite_verb_cx(source: &str, all: &[TaggedToken], range: Range<usize>) -> bool {
    has_finite_verb(&all[range.clone()])
        || range.into_iter().any(|i| {
            let text = folded_token_text(source, all, i);
            SUBJECT_CONTRACTIONS.contains(&text.as_str())
                || NEGATED_AUX_CONTRACTIONS.contains(&text.as_str())
        })
}

/// `true` if the tokens from `next` open an independent clause by any of
/// three signals: a fused subject+finite contraction as the very first
/// word (`— it's never called directly`), an imperative-initial bare
/// verb (`— pull in the published module instead`), or the parse-anchored
/// subject check ([`independent_clause_follows`]).
fn opens_independent_clause(
    source: &str,
    tokens: &[TaggedToken],
    parse: &SentenceParse,
    next: usize,
) -> bool {
    let first_is_fused_clause = next < tokens.len()
        && SUBJECT_CONTRACTIONS.contains(&folded_token_text(source, tokens, next).as_str());
    first_is_fused_clause
        || friction_nlp::is_imperative_initial(&tokens[next..])
        || independent_clause_follows(tokens, parse, next)
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
/// a single-token comparison would leave the multi-word entries
/// (`"the team"`, `"our team"`) unable to ever match. `they` is
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
/// subject the passive deletes. No table can paper over this. The
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
/// wanted", "Postgres' fine-grained control was preferred": both
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
        // "absorbeded" against real prose: a made-up word, not merely
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
    // (`"the server was set up"`): measured at three rewrites over the
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
        // sentence. Unconditional recapitalization was a real bug:
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

/// Nominalized-noun -> verb table (23 entries). Fixed and closed: a
/// suffix detector (`"-tion"`, `"-ment"`) would also match nouns with
/// no verb reading (`"nation"`, `"moment"`), so this stays a literal,
/// audited table.
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
        // which would also match `NNP`/`NNPS`: proper nouns are never
        // nominalizations, and the table above was audited under that
        // exclusion.
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
    out.extend(t6_em_dash(source, tokens, parse));
    out.extend(t7_semicolon(source, tokens, parse));
    out.sort_by_key(|candidate| (candidate.range.start, candidate.range.end));
    out
}

// ---------------------------------------------------------------------
// T6: em-dash reduction.
// ---------------------------------------------------------------------

/// `true` if `text` contains a literal backtick.
///
/// [`friction_parse::extract`] excludes a genuine inline-code span from
/// prose entirely (`Event::Code` is a leaf-excluded event, never text),
/// so an em dash inside real inline code never reaches this module's
/// input in the first place. This guard exists anyway, for the case the
/// exclusion doesn't cover -- prose bridged across an inline emphasis or
/// bracketed placeholder can still carry a literal backtick byte if one
/// was escaped in the source -- and it is cheap enough to apply
/// unconditionally rather than prove unreachable.
fn spans_inline_code(text: &str) -> bool {
    text.contains('`')
}

/// Every em-dash token index in `tokens`, in source order: a token whose
/// surface is exactly one U+2014 character.
///
/// A dash merged into a punctuation run with a directly adjacent mark
/// (`"word—,"` tokenizes as one multi-character token — see
/// `friction_nlp::tag_perceptron`'s tokenizer) is deliberately excluded:
/// rare in prose, and there is no safe way to isolate just the dash's own
/// byte out of a token whose surface is no longer only that character.
fn em_dash_tokens(source: &str, tokens: &[TaggedToken]) -> Vec<usize> {
    (0..tokens.len())
        .filter(|&index| token_text(source, tokens, index) == "\u{2014}")
        .collect()
}

/// `true` if the word right after the dash (`next`) opens the subject
/// phrase of a genuine independent clause -- not merely whether a finite
/// verb with a subject exists *somewhere* later in the sentence, which
/// would also match a subject buried inside a relative or subordinate
/// clause several levels down (measured on real prose: "... — including
/// writes made by other processes, which process-level caching **does**
/// not see unless you also configure ..." has a finite verb with its own
/// subject, "caching does", but "including" itself is a participle
/// opening a fragment, not an independent clause).
///
/// Scans for a token bearing `nsubj`/`nsubjPass` whose own subtree (see
/// [`subtree_span`]) *starts* exactly at `next` -- covering a modified
/// subject ("3 **workers** drain it" -- the subtree of "workers" starts
/// at "3") without also matching a subject deep inside an unrelated
/// embedded clause, whose subtree could never start there.
///
/// Two head shapes count, matching how this workspace's own parser
/// attaches a passive clause (see `RegisterCounts::agentless_passives`'s
/// own docs for the same attachment read): an active clause, where the
/// finite verb itself carries the subject child; and a passive clause,
/// where the *participle* is the head (carrying the subject) and the
/// finite auxiliary ("is"/"was") attaches to it as an `auxpass` child
/// instead of the other way around.
fn independent_clause_follows(tokens: &[TaggedToken], parse: &SentenceParse, next: usize) -> bool {
    (next..tokens.len()).any(|index| {
        let subject_head = parse.edge(index).and_then(|edge| {
            matches!(edge.relation, DepRelation::Nsubj | DepRelation::NsubjPass)
                .then_some(edge.head)
                .flatten()
        });
        let Some(head) = subject_head else {
            return false;
        };
        let heads_a_clause = FINITE_VERB_TAGS.contains(&tokens[head].pos.as_str())
            || children_with_relation(parse, head, DepRelation::AuxPass)
                .next()
                .is_some();
        heads_a_clause && subtree_span(parse, index).0 == next
    })
}

/// Case (a): a paired `" — X — "` interpolation where `X` (the tokens
/// strictly between the two dashes) carries no finite verb of its own:
/// a true parenthetical aside, not a second clause. Replaces the whole
/// span, dashes and their flanking spaces included, with `", X, "`.
///
/// Declines if either dash opens/closes the sentence (nothing to anchor
/// the surrounding comma to — the "empty side" the module docs warn
/// about), if `X` is empty, or if `X` has its own finite verb.
fn paired_parenthetical_candidate(
    source: &str,
    tokens: &[TaggedToken],
    first: usize,
    second: usize,
) -> Option<Candidate> {
    if first == 0 || second + 1 >= tokens.len() || second <= first + 1 {
        return None;
    }
    if has_finite_verb_cx(source, tokens, first + 1..second) {
        return None; // X carries its own clause; this isn't a parenthetical.
    }

    let range = tokens[first - 1].token.range.end..tokens[second + 1].token.range.start;
    if spans_inline_code(&source[range.clone()]) {
        return None;
    }

    let x_start = tokens[first + 1].token.range.start;
    let x_end = tokens[second - 1].token.range.end;
    let x_text = source[x_start..x_end].trim();
    if x_text.is_empty() {
        return None;
    }

    let mut delta = BTreeMap::new();
    delta.insert("em_dash", -2);
    // An interpolation that carries its own commas can't be set off by a
    // comma pair — its internal commas and the delimiting ones become
    // indistinguishable, flattening the aside into a run-on list.
    // Parentheses keep the boundary readable and are equally
    // punctuation-only.
    let replacement = if x_text.contains(',') {
        format!(" ({x_text}) ")
    } else {
        format!(", {x_text}, ")
    };
    Some(Candidate {
        kind: CandidateKind::EmDash,
        range,
        replacement: replacement.into_boxed_str(),
        delta,
        confidence: 0.9,
    })
}

/// The definition-lead-in case: a single em dash whose LEFT side carries
/// no finite verb — a bare term or noun-phrase lead ("**Rate limiting**
/// — the design calls for …"). Replaces the dash and its flanking spaces
/// with `": "`, the punctuation this pattern actually means.
fn lead_in_candidate(source: &str, tokens: &[TaggedToken], dash: usize) -> Option<Candidate> {
    let range = tokens[dash - 1].token.range.end..tokens[dash + 1].token.range.start;
    if spans_inline_code(&source[range.clone()]) {
        return None;
    }
    let mut delta = BTreeMap::new();
    delta.insert("em_dash", -1);
    Some(Candidate {
        kind: CandidateKind::EmDash,
        range,
        replacement: Box::from(": "),
        delta,
        confidence: 0.85,
    })
}

/// Case (b): a single em dash with no finite verb between it and the
/// sentence's end: an appositive/elaboration fragment, not a second
/// clause. Replaces the dash and its flanking spaces with `", "` — or
/// with `": "` when the fragment carries commas of its own: a bare comma
/// delimiter in front of a comma-bearing fragment flattens it into one
/// long false list ("pushing React core forward, faster, simpler, and
/// easier to work with": measured on real prose), the same collision the
/// paired case escapes with parentheses.
fn fragment_candidate(source: &str, tokens: &[TaggedToken], dash: usize) -> Option<Candidate> {
    let range = tokens[dash - 1].token.range.end..tokens[dash + 1].token.range.start;
    if spans_inline_code(&source[range.clone()]) {
        return None;
    }
    let fragment_text =
        &source[tokens[dash + 1].token.range.start..tokens[tokens.len() - 1].token.range.end];
    let replacement = if fragment_text.contains(',') {
        ": "
    } else {
        ", "
    };
    let mut delta = BTreeMap::new();
    delta.insert("em_dash", -1);
    Some(Candidate {
        kind: CandidateKind::EmDash,
        range,
        replacement: Box::from(replacement),
        delta,
        confidence: 0.85,
    })
}

/// Case (c): a single em dash followed by an independent clause (its own
/// finite verb and subject). Replaces the dash plus the word right after
/// it with `". "` plus that word recapitalized -- or, if the following
/// word can't be sensibly recapitalized, a fallback that needs no case
/// change at all.
///
/// # Closure gate interaction
///
/// `friction-edit::register::closure_violation` compares words by
/// running both the replacement and the original span through
/// `friction_match::token::tokenize_str`, which lowercases every word
/// token before comparing (`fold_token_text`). Recapitalizing "it" to
/// "It" is therefore invisible to that gate -- both fold back to "it" --
/// so this function needs no special-case registration for the
/// recapitalized form; the ordinary "already in the matched span" rule
/// covers it.
fn independent_clause_candidate(
    source: &str,
    tokens: &[TaggedToken],
    dash: usize,
) -> Option<Candidate> {
    let next = dash + 1;
    let range = tokens[dash - 1].token.range.end..tokens[next].token.range.end;
    if spans_inline_code(&source[range.clone()]) {
        return None;
    }

    let next_word = token_text(source, tokens, next);
    let first_char = next_word.chars().next()?;
    let replacement = if !first_char.is_alphabetic() {
        // A digit, backtick, or bracket can't be recapitalized, and a
        // semicolon needs no recapitalization at all -- the safe
        // fallback, rather than ever leaving a sentence starting with a
        // lowercase letter (or a bare symbol) after a period.
        format!("; {next_word}")
    } else if first_char.is_uppercase() {
        // Already capitalized (a proper noun): recapitalizing is a
        // no-op, so the trivial case just needs the period.
        format!(". {next_word}")
    } else {
        format!(". {}", recapitalize(next_word))
    };

    let mut delta = BTreeMap::new();
    delta.insert("em_dash", -1);
    Some(Candidate {
        kind: CandidateKind::EmDash,
        range,
        replacement: replacement.into_boxed_str(),
        delta,
        confidence: 0.8,
    })
}

/// Em-dash reduction (T6): homes the `em_dash` feature toward its
/// human band (measured at effectively zero -- see `register-v1.toml`)
/// by removing em dashes one sentence-level construction at a time.
///
/// Handles exactly two shapes, chosen by how many em-dash tokens (see
/// [`em_dash_tokens`]) the sentence has:
/// - two: [`paired_parenthetical_candidate`] (case a);
/// - one: [`fragment_candidate`] (case b) if no finite verb follows,
///   otherwise [`independent_clause_candidate`] (case c) if an
///   independent clause follows, otherwise no candidate (a finite verb
///   present but not heading its own subject is still governed by the
///   clause before the dash -- ambiguous which side it belongs to, so
///   this module declines rather than guess).
///
/// Zero em-dash tokens, or three or more, produce no candidate at all:
/// zero has nothing to rewrite, and three-plus is a shape (multiple
/// parentheticals, or a parenthetical plus a fragment) this module has
/// no principled way to decompose into an ordered pair of edits without
/// risking a wrong pairing across an unrelated third dash.
///
/// `parse` is required, mirroring [`t4_activize_to_passive`]/
/// [`t5_nominalization`]: a sentence whose parse failed never reaches any
/// transducer at all (`friction-edit::register::build_sentence_contexts`
/// drops it upstream), so there is no "unparsed sentence" case for this
/// function itself to handle.
#[must_use]
pub fn t6_em_dash(source: &str, tokens: &[TaggedToken], parse: &SentenceParse) -> Vec<Candidate> {
    match em_dash_tokens(source, tokens).as_slice() {
        [first, second] => paired_parenthetical_candidate(source, tokens, *first, *second)
            .into_iter()
            .collect(),
        [dash] => {
            let dash = *dash;
            if dash == 0 || dash + 1 >= tokens.len() {
                return Vec::new(); // nothing on one side to anchor the rewrite.
            }
            if !has_finite_verb_cx(source, tokens, 0..dash) {
                // A verbless left side is a definition lead-in ("**Rate
                // limiting** — the design calls for ..."), where the dash
                // separates a term from its explanation. A comma there is
                // wrong whatever follows: it either splices a clause onto
                // a bare noun phrase or misreads the lead as apposition.
                // The colon is the rewrite for this pattern, and it needs
                // no read on the right side at all. Which also shields
                // this case from tagger noise on the right side's verb.
                lead_in_candidate(source, tokens, dash)
                    .into_iter()
                    .collect()
            } else if opens_independent_clause(source, tokens, parse, dash + 1) {
                // Checked BEFORE the fragment branch: a fused
                // subject+finite contraction ("— it's machine register")
                // and an imperative ("— pull in the published module")
                // both read as verbless to the tag scan, and a comma
                // against either is a splice. Measured on real prose,
                // both shapes were produced by exactly that mistake.
                independent_clause_candidate(source, tokens, dash)
                    .into_iter()
                    .collect()
            } else if !has_finite_verb_cx(source, tokens, dash + 1..tokens.len()) {
                fragment_candidate(source, tokens, dash)
                    .into_iter()
                    .collect()
            } else {
                Vec::new() // finite verb present but no subject of its own; ambiguous, decline.
            }
        }
        _ => Vec::new(),
    }
}

// ---------------------------------------------------------------------
// T7: semicolon-splice reduction.
// ---------------------------------------------------------------------

/// Every semicolon token index in `tokens`, in source order: a token
/// whose surface is exactly one ASCII `;` (U+003B).
///
/// Never U+037E (the Greek question mark, a visual lookalike in most
/// fonts) — an exact surface match is definitionally scoped to that one
/// code point, the same convention [`em_dash_tokens`] uses for U+2014
/// against its own lookalike, U+2013.
fn semicolon_tokens(source: &str, tokens: &[TaggedToken]) -> Vec<usize> {
    (0..tokens.len())
        .filter(|&index| token_text(source, tokens, index) == ";")
        .collect()
}

/// The token-index bounds of every segment `semis` splits the sentence
/// into: before the first semicolon, between each consecutive pair, and
/// after the last — `semis.len() + 1` half-open ranges, each excluding
/// the semicolon tokens themselves, covering every remaining token
/// exactly once.
fn semicolon_segments(semis: &[usize], token_count: usize) -> Vec<Range<usize>> {
    let mut starts = Vec::with_capacity(semis.len() + 1);
    let mut ends = Vec::with_capacity(semis.len() + 1);
    starts.push(0);
    for &semi in semis {
        ends.push(semi);
        starts.push(semi + 1);
    }
    ends.push(token_count);
    starts.into_iter().zip(ends).map(|(s, e)| s..e).collect()
}

/// One semicolon's own candidate: a semicolon joining two independent
/// clauses becomes a sentence break (`". "`), the following word
/// recapitalized.
///
/// Declines if `semi` opens or closes the sentence (nothing to anchor
/// the rewrite to), the span crosses inline code, the clause right
/// after it isn't a genuine independent clause
/// ([`independent_clause_follows`], reused unchanged from T6), or the
/// following word can't be recapitalized. That last case has no
/// fallback, unlike T6's case (c): a semicolon substituting for an
/// unrecapitalizable word there falls back to the semicolon *because*
/// the source had a dash, a strictly worse-tell character, to trade
/// away. Here the source already has the semicolon — there is nothing
/// to substitute it for, so this declines outright rather than leave a
/// sentence starting on a lowercase letter or introduce a second
/// semicolon/comma with no license.
fn semicolon_candidate(
    source: &str,
    tokens: &[TaggedToken],
    parse: &SentenceParse,
    semi: usize,
) -> Option<Candidate> {
    if semi == 0 || semi + 1 >= tokens.len() {
        return None;
    }
    if !opens_independent_clause(source, tokens, parse, semi + 1) {
        return None;
    }

    let range = tokens[semi - 1].token.range.end..tokens[semi + 1].token.range.end;
    if spans_inline_code(&source[range.clone()]) {
        return None;
    }

    let next_word = token_text(source, tokens, semi + 1);
    let first_char = next_word.chars().next()?;
    if !first_char.is_alphabetic() {
        return None; // no sensible fallback for a semicolon; see this function's own docs.
    }
    let replacement = if first_char.is_uppercase() {
        // Already capitalized (a proper noun): recapitalizing is a
        // no-op, so the trivial case just needs the period.
        format!(". {next_word}")
    } else {
        format!(". {}", recapitalize(next_word))
    };

    let mut delta = BTreeMap::new();
    delta.insert("semicolon", -1);
    Some(Candidate {
        kind: CandidateKind::Semicolon,
        range,
        replacement: replacement.into_boxed_str(),
        delta,
        confidence: 0.8,
    })
}

/// Semicolon-splice reduction (T7).
///
/// Homes the `semicolon` feature toward its human band, nonzero
/// unlike T6's em-dash band (see `register-v1.toml`'s
/// `[features.semicolon]`), by turning a semicolon that joins two
/// independent clauses into a sentence break.
///
/// One rewrite shape only, unlike T6's four: a semicolon splicing two
/// independent clauses is the one construction this module can act on
/// with confidence. Every other established use (a serial-comma list's
/// own separator, an elliptical continuation with no clause of its own)
/// is left alone; declining rather than guessing at those is this
/// function's entire job.
///
/// # Serial-semicolon protection
///
/// Splits the sentence into segments at every semicolon (see
/// [`semicolon_segments`]). If any segment lacks a finite verb of its
/// own, the whole sentence produces no candidate at all: with a
/// verbless segment present, either the sentence has one semicolon and
/// an elliptical side, or two-or-more and at least one is a "super-comma"
/// list separator ("the A, which Bs; the C, which Ds; and the E") rather
/// than a clause boundary — and there is no principled way to tell,
/// from a verbless segment alone, which semicolon (if any) is the
/// genuine clause boundary. Splitting a super-comma list at any point
/// destroys it, so this declines every semicolon in the sentence rather
/// than risk picking the wrong one.
///
/// Only once every segment carries its own finite clause is each
/// semicolon evaluated — and independently licensed or declined — on
/// its own merits via [`semicolon_candidate`].
///
/// `parse` is required, mirroring [`t6_em_dash`]: a sentence whose parse
/// failed never reaches any transducer at all
/// (`friction-edit::register::build_sentence_contexts` drops it
/// upstream).
#[must_use]
pub fn t7_semicolon(source: &str, tokens: &[TaggedToken], parse: &SentenceParse) -> Vec<Candidate> {
    let semis = semicolon_tokens(source, tokens);
    if semis.is_empty() {
        return Vec::new();
    }

    // An introducing colon before the first semicolon marks a
    // colon-introduced enumeration ("runs in four stages: A lets X; B
    // covers Y; ..."), where semicolons are the construction's correct
    // coordinate separators even when every item carries its own finite
    // clause. Splitting any of them breaks the enumeration's symmetry —
    // measured on real prose: the band-edge stop once promoted exactly
    // one item to a sentence and left its siblings semicolon-joined.
    if (0..semis[0]).any(|i| token_text(source, tokens, i) == ":") {
        return Vec::new();
    }

    let segments = semicolon_segments(&semis, tokens.len());
    if segments
        .iter()
        .any(|segment| !has_finite_verb_cx(source, tokens, segment.clone()))
    {
        return Vec::new(); // a verbless segment: see this function's own docs.
    }

    semis
        .iter()
        .filter_map(|&semi| semicolon_candidate(source, tokens, parse, semi))
        .collect()
}

// ---------------------------------------------------------------------
// Inflection. `third_sg` is unused by either transducer (T5's
// `NOMINAL_VERB` spells out its own forms), but ships with
// `past`/`past_participle` since the three were audited as one unit — a
// participial transducer would need it, and splitting it out now would
// only fragment that audit.
// ---------------------------------------------------------------------

/// Endings after which the regular third-person-singular suffix is
/// `"-es"`, not `"-s"`. `"o"` joins the true sibilants (`"go"` ->
/// `"goes"`) because English spelling extends the same epenthetic vowel
/// to a trailing `"o"`, not because it's phonetically a sibilant.
const SIBILANT_ENDINGS: &[&str] = &["s", "x", "z", "ch", "sh", "o"];

/// Irregular past-tense forms (53 entries: the original 44 plus 9
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
/// Consults no irregular table: [`IRREGULAR_PAST`]/
/// [`IRREGULAR_PAST_PARTICIPLE`] cover past forms only, and English
/// third-singular formation is regular outside `be`/`have` (neither of
/// which is ever a `NOMINAL_VERB` target).
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
