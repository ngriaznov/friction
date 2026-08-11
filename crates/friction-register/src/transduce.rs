//! Rewrite transducers for the register pass.
//!
//! Each proposes a candidate edit predicting an exact per-feature count
//! delta and the byte range it would replace. Two of a possible five
//! register transducers are implemented — the cut is
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
//! `register-en-v1.toml`'s `[features.em_dash]`/`[features.semicolon]` and
//! `docs/research/FRONTIER_MODELS.md`). [`t7_semicolon`] reuses
//! [`independent_clause_follows`], T6's own subtree-anchored
//! independent-clause check, unchanged: the two features license the
//! same rewrite shape on the same evidence, just for different
//! punctuation.
//!
//! [`t9_past_progressive`] is a later addition of the same kind: not
//! from the original Biber-feature list either, but a measured
//! Claude-family tell (see `register-en-v1.toml`'s
//! `[features.past_progressive]` once wired). It licenses on tags and
//! lemmas alone, no subtree walk needed, since the construction it
//! targets is exactly two adjacent tokens.
//!
//! Functions here only propose candidates; none mutate `source`, choose
//! between overlaps, or apply anything: a caller's decision.

use std::collections::BTreeMap;
use std::ops::Range;

use friction_core::CoreError;
use friction_core::span::{Spanned, validate_range};
use friction_nlp::{
    DepEdge, DepRelation, FINITE_VERB_TAGS, LEXICON_EN, Lexicon, SentenceParse, TaggedToken,
    coarse_tag, has_finite_verb, is_imperative_initial,
};

/// A proposed rewrite: replace `range` with `replacement`, moving the
/// Biber feature counts in `delta` by the listed amount.
///
/// Deliberately separate from `friction_core::Patch` despite the same
/// shape: a `Patch` is a decision already applied, a `Candidate` is
/// unselected — merging them would make "chosen yet" untyped.
#[derive(Debug, Clone, PartialEq, Eq)]
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
    /// Produced by [`t9_past_progressive`]. Measured at 472.1 per million
    /// words in machine prose vs 198.9 in human prose, a 2.4x rate -- see
    /// `register-en-v1.toml`'s `[features.past_progressive]` once wired.
    PastProgressive,
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

/// Token `index`'s surface, case-folded with curly apostrophes
/// straightened: the shape both contraction lists are written in.
fn folded_token_text(source: &str, tokens: &[TaggedToken], index: usize) -> String {
    token_text(source, tokens, index)
        .to_lowercase()
        .replace('\u{2019}', "'")
}

/// [`has_finite_verb`] plus contraction awareness: `true` if any token is
/// tag-finite OR is a [`Lexicon::subject_contractions`](friction_nlp::Lexicon::subject_contractions)
/// or [`Lexicon::negated_aux_contractions`](friction_nlp::Lexicon::negated_aux_contractions)
/// entry. The widening is deliberately local to this module's
/// transducers. The edit gates keep the strict tag-based check they were
/// calibrated with.
///
/// Both lists exist because the shipped tagger fuses a contraction into
/// one token and tags it `PRP`/leaves it unrecognized, so a tag-based
/// finite-verb scan is blind to the finite verb inside: a subject
/// contraction fuses a subject AND a finite auxiliary (`it's` = `it is`)
/// — measured on real prose: "None of it is wrong, exactly — it's
/// machine register …" was comma-spliced because `it's` read as a bare
/// pronoun — while a negated-auxiliary contraction fuses a finite verb
/// plus its negation with no subject (`aren't` can surface untagged as a
/// verb). Both lists are case-folded, straight-apostrophe forms; match
/// after folding U+2019 (see [`folded_token_text`]).
fn has_finite_verb_cx(source: &str, all: &[TaggedToken], range: Range<usize>) -> bool {
    has_finite_verb(&all[range.clone()])
        || range.into_iter().any(|i| {
            let text = folded_token_text(source, all, i);
            LEXICON_EN.subject_contractions.contains(&text)
                || LEXICON_EN.negated_aux_contractions.contains(&text)
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
        && LEXICON_EN
            .subject_contractions
            .contains(&folded_token_text(source, tokens, next));
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
/// - the subject's text is one of `LEXICON_EN.generic_subjects`
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
    if LEXICON_EN.stative_verbs.contains(lemma) {
        // Stative or linking-verb lemmas: never a licensed passive.
        //
        // "Have" and "become"/"remain"/"seem"/"appear" are copulas
        // taking a predicate-nominal complement, not a true object:
        // forcing the transform produces nonsense ("about a day of
        // downtime was had", "problems are become").
        //
        // "Want"/"wish"/"prefer"/"need" differ: ordinary transitive
        // verbs, so the transform is legal, but a passive of a desire
        // reads bureaucratic and loses who wanted it ("the surprising
        // benefits we've found were wanted", "Postgres' fine-grained
        // control was preferred": both grammatical, both worse).
        //
        // The distinction matters if revisited: the first group cannot
        // be passivized, the second should not be.
        return false;
    }
    if lemma.ends_with("ed") && !LEXICON_EN.is_irregular_verb_base(lemma) {
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
        && !LEXICON_EN.is_irregular_verb_base(lemma)
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
    // Reflexive pronouns are never a licensed passive subject: "I ...
    // tore myself away" promotes to "Myself was torn away" —
    // ungrammatical, since a reflexive has no referent independent of
    // the subject the passive deletes. No table can paper over this;
    // the transducer refuses to fire.
    //
    // Personal pronouns are refused for a different reason — judgement,
    // not grammar — so kept as a separate check rather than merged into
    // the reflexive one, which would hide that difference. Two reasons:
    // a ditransitive verb's indirect object is often mislabeled `dobj`,
    // promoting the wrong participant, and a pronoun carries almost no
    // information to promote. Measured on real prose: "see how much
    // time and effort they save you" passivizes to "...you are saved"
    // — the beneficiary is promoted while the real object ("time and
    // effort") is stranded.
    let obj_surface = token_text(source, tokens, obj_token).to_lowercase();
    if LEXICON_EN.reflexive_pronouns.contains(&obj_surface) {
        return false;
    }
    if LEXICON_EN.object_pronouns.contains(&obj_surface) {
        return false;
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

/// `true` if the promoted object takes plural "be" agreement.
///
/// Agreement follows the whole promoted phrase, not just its head
/// token: a coordinated object is plural even when its head conjunct
/// is singular — the head-only check produced "a thumbnail, title,
/// price, and two badges **is** rendered" against real prose. An
/// "or"/"nor" list instead agrees with the conjunct nearest the verb
/// slot (the last one): "thumbnails or a badge **is** rendered" —
/// blanket-plural over disjunction would trade one agreement error for
/// another.
fn promoted_object_is_plural(
    source: &str,
    tokens: &[TaggedToken],
    parse: &SentenceParse,
    obj_token: usize,
    obj_first: usize,
    obj_last: usize,
) -> bool {
    let Some(last_conjunct) = children_with_relation(parse, obj_token, DepRelation::Conj)
        .map(|edge| edge.token)
        .max()
    else {
        return token_takes_plural_agreement(source, tokens, obj_token);
    };
    let mut coordinators = (obj_first..=obj_last)
        .filter(|&index| {
            parse
                .edge(index)
                .is_some_and(|edge| edge.relation == DepRelation::Cc)
        })
        .map(|index| token_text(source, tokens, index).to_lowercase())
        .peekable();
    let disjunctive =
        coordinators.peek().is_some() && coordinators.all(|cc| matches!(cc.as_str(), "or" | "nor"));
    if disjunctive {
        token_takes_plural_agreement(source, tokens, last_conjunct)
    } else {
        // "and" coordination — or an asyndetic comma list, which reads
        // the same way: more than one referent, plural.
        true
    }
}

/// `true` if this single token, standing as (or ending) a promoted
/// subject, takes plural "be" agreement.
///
/// "you" always takes plural agreement regardless of referent count —
/// measured where "you" as a promoted object produced "You is
/// encouraged" three times. "them" belongs with "they"/"these"/"those"
/// but was missing, producing "Them is plucked".
fn token_takes_plural_agreement(source: &str, tokens: &[TaggedToken], index: usize) -> bool {
    let surface = token_text(source, tokens, index).to_lowercase();
    surface == "you"
        || matches!(tokens[index].pos.as_str(), "NNS" | "NNPS")
        || matches!(surface.as_str(), "they" | "these" | "those" | "them")
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

        // Subjects recoverable without an agent phrase: the reader
        // already knows who "we"/"the team" is, unlike an identifiable
        // agent ("the vendor") whose demotion erases information.
        //
        // Matched against the subject's full span, not just its head
        // token — a single-token comparison would leave the multi-word
        // entries ("the team", "our team") unable to ever match.
        // "they" is deliberately absent from the lexicon's generic-
        // subject list: it's anaphoric to an earlier antecedent, unlike
        // "we"/"i"/"you"/"one", which are non-referential.
        //
        // Measured on real prose: "Browse the list of programmed
        // channels to ensure they match the uploaded file" satisfies
        // every condition, and passivizing yields "...to ensure the
        // uploaded file is matched", quietly dropping that *the
        // channels* must match — a meaning change this transducer
        // isn't permitted to make.
        let (subj_first, subj_last) = subtree_span(parse, subj.token);
        let subj_text = span_text(source, tokens, subj_first, subj_last).to_lowercase();
        if !LEXICON_EN.generic_subjects.contains(&subj_text) {
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
        let plural =
            promoted_object_is_plural(source, tokens, parse, obj.token, obj_first, obj_last);
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
/// The read side of `LEXICON_EN.nominal_verbs` (fixed and closed: a
/// suffix detector (`"-tion"`, `"-ment"`) would also match nouns with
/// no verb reading — `"nation"`, `"moment"` — so it stays a literal,
/// audited table), exposed so a caller validating closure consults the
/// same lookup rather than a second, independently-maintained copy.
#[must_use]
pub fn nominal_verb_for(nominalization_lower: &str) -> Option<&'static str> {
    let lex: &'static Lexicon = &LEXICON_EN;
    lex.nominal_verbs.get(nominalization_lower).map(Box::as_ref)
}

/// Nominalization unpacking (T5): `"the optimization of X"` -> `"optimizing X"`.
///
/// Keeps the sentence verbal without inventing an agent, the same
/// restraint [`t4_activize_to_passive`] shows in reverse.
///
/// # Licensing conditions
/// - the noun's lowercase text is a key in `LEXICON_EN.nominal_verbs`
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
        // nominalizations, and `LEXICON_EN.nominal_verbs` was audited
        // under that exclusion.
        if !matches!(tokens[index].pos.as_str(), "NN" | "NNS") {
            continue;
        }
        let lower = token_text(source, tokens, index).to_lowercase();
        let Some(verb) = LEXICON_EN.nominal_verbs.get(lower.as_str()) else {
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
    out.extend(t9_past_progressive(source, tokens, parse));
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
    })
}

/// Em-dash reduction (T6): homes the `em_dash` feature toward its
/// human band (measured at effectively zero -- see `register-en-v1.toml`)
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
    })
}

/// Semicolon-splice reduction (T7).
///
/// Homes the `semicolon` feature toward its human band, nonzero
/// unlike T6's em-dash band (see `register-en-v1.toml`'s
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
// T9: past-progressive simplification.
// ---------------------------------------------------------------------

/// `true` if `text`, case-folded, is "when" or "while" -- a temporal
/// subordinator whose clause the progressive aspect may be doing real
/// work against (see [`t9_past_progressive`]'s own docs).
fn is_temporal_frame_word(text: &str) -> bool {
    matches!(text.to_lowercase().as_str(), "when" | "while")
}

/// `true` if any token anywhere in the sentence is "when" or "while".
///
/// Sentence-wide, not clause-local: this module has no `advcl`
/// resolution accurate enough to scope the check to just the clause the
/// temporal subordinator introduces (see the module's own top-level docs
/// on why `acl`/`advcl` never gate a transducer here), so this errs
/// toward declining the whole sentence rather than risk homing a
/// progressive that *is* doing real aspectual work against a frame
/// elsewhere in it.
fn sentence_has_temporal_frame(source: &str, tokens: &[TaggedToken]) -> bool {
    (0..tokens.len()).any(|index| is_temporal_frame_word(token_text(source, tokens, index)))
}

/// `true` if `aux` is attached to `verb` by its own (non-passive) `aux`
/// edge: the periphrastic-progressive shape ("was **marking**", `aux` on
/// "marking" pointing at "was"), not a copula taking an adjectival
/// predicate that merely happens to carry a `VBG` tag ("the plan was
/// **promising**" -- an adjective, not a verb in progress, and not this
/// transducer's business to rewrite).
fn is_progressive_aux(parse: &SentenceParse, aux: usize, verb: usize) -> bool {
    children_with_relation(parse, verb, DepRelation::Aux).any(|edge| edge.token == aux)
}

/// One `"was"`/`"were"` + `VBG` pair's own candidate: the periphrastic
/// past progressive collapsed to the participle's simple past, via
/// [`friction_nlp::past`]. Capitalization transfers from the auxiliary:
/// a sentence-initial "Was"/"Were" produces a capitalized replacement,
/// same as every other recapitalizing rewrite in this module.
///
/// # Declines
/// - the sentence contains "when" or "while" anywhere
///   ([`sentence_has_temporal_frame`]): the progressive may be doing real
///   aspectual work against a temporal frame, and homing a register band
///   never licenses a meaning change
/// - the token right after `aux` isn't tagged `VBG` at all -- covers an
///   adverb between them ("was quietly marking"): collapsing it forces
///   an adverb-placement decision no table can make (before the new verb,
///   "quietly marked", or after, "marked quietly" -- either is a
///   judgment call this function isn't licensed to make), so it declines
///   rather than guess, the same way it declines any other non-`VBG`
///   token in that slot
/// - the participle is "being" or "going": neither is the plain
///   progressive this transducer targets -- "was being X" is a stative
///   copula construction, and "was going" is usually the "going to"
///   future periphrasis, not the progressive aspect
/// - `aux` isn't attached to `verb` by its own `aux` edge
///   ([`is_progressive_aux`])
/// - [`friction_nlp::past`] returns an empty string for the participle's
///   lemma (a defensive check against a malformed lemma reaching this
///   far; the shipped lexicon never produces one today)
fn past_progressive_candidate(
    source: &str,
    tokens: &[TaggedToken],
    parse: &SentenceParse,
    aux: usize,
) -> Option<Candidate> {
    let verb = aux + 1;
    let verb_token = tokens.get(verb).filter(|t| t.pos.as_str() == "VBG")?;
    let verb_surface_lower = token_text(source, tokens, verb).to_lowercase();
    if matches!(verb_surface_lower.as_str(), "being" | "going") {
        return None;
    }
    if !is_progressive_aux(parse, aux, verb) {
        return None;
    }
    // A participle sitting directly before a bare noun is reading as an
    // attributive adjective, whatever the parse says: "support were
    // compelling factors" carries a mis-attached `aux` edge in the
    // shipped parser's output, and collapsing it produced "compelled
    // factors" on a real corpus fixture during this feature's snapshot
    // review. A genuine progressive's bare-plural object ("was marking
    // scope shifts") declines too -- fail-closed, the designed reading.
    if tokens
        .get(verb + 1)
        .is_some_and(|t| t.pos.as_str().starts_with("NN"))
    {
        return None;
    }

    let past_form = friction_nlp::past(verb_token.lemma.as_ref());
    if past_form.is_empty() {
        return None;
    }

    let aux_surface = token_text(source, tokens, aux);
    let replacement = if aux_surface.starts_with(char::is_uppercase) {
        recapitalize(&past_form)
    } else {
        past_form
    };

    let mut delta = BTreeMap::new();
    delta.insert("past_progressive", -1);
    Some(Candidate {
        kind: CandidateKind::PastProgressive,
        range: tokens[aux].token.range.start..tokens[verb].token.range.end,
        replacement: replacement.into_boxed_str(),
        delta,
    })
}

/// Every `"was"`/`"were"` token index in `tokens`, in source order,
/// case-insensitive on the surface text.
fn past_progressive_aux_tokens(source: &str, tokens: &[TaggedToken]) -> Vec<usize> {
    (0..tokens.len())
        .filter(|&index| {
            matches!(
                token_text(source, tokens, index).to_lowercase().as_str(),
                "was" | "were"
            )
        })
        .collect()
}

/// Past-progressive simplification (T9): `"was marking"`/`"were
/// marking"` -> `"marked"`.
///
/// Homes the `past_progressive` feature toward its human band: machine
/// prose uses the periphrastic past progressive well past the human rate
/// (472.1 vs 198.9 per million words, a 2.4x rate -- see
/// `register-en-v1.toml`'s `[features.past_progressive]` once wired),
/// the progressive-softening tell -- spreading a plain past-tense claim
/// out across an auxiliary and a participle to sound more considered
/// than the flat simple past would.
///
/// One rewrite shape: an adjacent `"was"`/`"were"` + `VBG` pair collapses
/// to the participle's simple past. See [`past_progressive_candidate`]
/// for the full licensing/decline list. `parse` is required, mirroring
/// every other transducer here: a sentence whose parse failed never
/// reaches this function at all
/// (`friction-edit::register::build_sentence_contexts` drops it
/// upstream).
#[must_use]
pub fn t9_past_progressive(
    source: &str,
    tokens: &[TaggedToken],
    parse: &SentenceParse,
) -> Vec<Candidate> {
    if sentence_has_temporal_frame(source, tokens) {
        return Vec::new();
    }
    past_progressive_aux_tokens(source, tokens)
        .into_iter()
        .filter_map(|aux| past_progressive_candidate(source, tokens, parse, aux))
        .collect()
}

// ---------------------------------------------------------------------
// T10: "ensure that NP is/are VBN" -> imperative/infinitival "VB NP".
// ---------------------------------------------------------------------

/// One [`t10_ensures_that_restructure`] outcome for a single `ensure`
/// token: either a licensed rewrite, or the specific reason a genuinely
/// matched "ensure that NP is/are VBN" shape declined.
///
/// [`RestructureOutcome::Declined`] is returned only once the construction
/// has already matched (this function's own steps 1 through 6: an
/// `ensure` lemma, a `Ccomp` child tagged `VBN`, a literal "that", exactly
/// one qualifying `AuxPass` child, exactly one `NsubjPass` child) — a
/// shape that never reaches that point (a modal, intransitive, or
/// adjectival complement) produces no [`RestructureOutcome`] at all, the
/// same silent-decline convention [`t4_activize_to_passive`] uses for a
/// verb that was never a licensable candidate to begin with. A caller
/// wanting the corpus-backlog accounting the design calls for turns
/// [`RestructureOutcome::Declined`] into a held [`friction_core::Finding`]
/// and leaves a shape this function never returned anything for silent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RestructureOutcome {
    /// A licensed rewrite: replace `range` with `replacement`.
    Candidate {
        /// Byte range in the original source this candidate would
        /// replace.
        range: Range<usize>,
        /// The text to substitute for `range`.
        replacement: Box<str>,
    },
    /// The construction matched but a named guard declined it.
    Declined {
        /// Byte range of the matched "ensure ... VBN" run.
        range: Range<usize>,
        /// Which guard declined, for the held finding's message.
        reason: &'static str,
    },
}

/// The negation set this construction's own law uses: "any word that can
/// invert the participle's polarity" — a strict superset of
/// [`friction_register::features`](crate::features)'s own
/// `NEGATION_PARTICLES`, which deliberately excludes "never" for a
/// Biber-feature-counting reason that has no bearing here (a feature
/// count tolerates one missed instance; a passive-to-imperative collapse
/// over a missed "never" states the opposite of what the author wrote).
/// Declared locally rather than imported, per this module's own
/// evidence-scoping convention (see `PERMITTED_FUNCTION_WORDS`'s own
/// docs for the same "closed, local, audited" shape).
const T10_NEGATION_WORDS: [&str; 3] = ["not", "n't", "never"];

/// The four contracted "be" + "not" forms this construction's own
/// `AuxPass` step also recognizes as matching (not merely as later
/// negation evidence): the shipped tagger fuses a negated auxiliary into
/// one token (`friction_match::token`'s own tokenizer keeps "don't"/
/// "isn't" whole — see its `tokenize_str` doc example), so a literal
/// "is"/"are"/"was"/"were" surface check alone would silently miss the
/// single most dangerous negated shape this construction can see: parsed
/// against the shipped parser, "Ensure that telemetry isn't sent." keeps
/// the full passive shape (`NsubjPass`/`AuxPass`/`Ccomp`-tagged-`VBN`)
/// with "isn't" as the `AuxPass` token itself — verified against the
/// shipped tagger and parser before this list was written, not assumed.
const T10_NEGATED_AUXPASS_FORMS: [&str; 4] = ["isn't", "aren't", "wasn't", "weren't"];

/// `true` if `stem` (a candidate base-form verb with no trailing `"e"`)
/// is one of the narrow shapes where restoring a silent `"e"` is a
/// certainty, not a guess: a word ending in a bare consonant immediately
/// followed by `"l"`, with no vowel between them.
///
/// English has no base verb ending in an unvoweled `"...Cl"` cluster —
/// every real word of that shape ("enable", "handle", "settle",
/// "struggle") keeps the silent `"e"` that makes the trailing `"l"`
/// syllabic. This is the one silent-`"e"` shape decidable from spelling
/// alone; the wider class (`"resolve"` vs. `"want"`, both a bare
/// consonant preceded by a vowel after suffix-stripping) is not — see
/// [`t10_derive_base_verb`]'s own docs for why that wider class is left
/// alone rather than guessed at.
fn t10_ends_in_bare_consonant_l(stem: &str) -> bool {
    let chars: Vec<char> = stem.chars().collect();
    let Some(&last) = chars.last() else {
        return false;
    };
    let Some(&before_last) = chars.len().checked_sub(2).map(|i| &chars[i]) else {
        return false;
    };
    last == 'l'
        && !matches!(
            before_last.to_ascii_lowercase(),
            'a' | 'e' | 'i' | 'o' | 'u'
        )
}

/// Derives the base-form ("VB") verb `comp_surface` (a `VBN` token's own
/// surface text) inflects from, corroborated against
/// [`friction_nlp::past_participle`]'s forward direction — `None` if no
/// candidate round-trips, the fail-closed outcome [`t10_ensures_that_restructure`]
/// turns into a named decline rather than ever guessing.
///
/// `comp_lemma` (the tagger's own reverse-lemmatization) is trusted after
/// the round-trip check, with one correction: measured directly against
/// the shipped tagger, `friction_nlp::lemmatize`'s own candidate ordering
/// tries the naive `"-ed"`-stripped stem before the silent-`"e"`-restored
/// one, and for a real, common verb class ("enable", "disable" — the
/// design's own flagship example) the naive stem round-trips on its own
/// merits (no consonant-doubling rule intervenes to rule it out), so it
/// wins even though it isn't the true lemma. [`t10_ends_in_bare_consonant_l`]
/// catches exactly that shape and corrects it. The much wider class of
/// silent-`"e"` stems this doesn't catch (`"resolved"` -> `"resolv"`,
/// `"closed"` -> `"clos"`: both round-trip identically to a genuine
/// e-less verb like `"want"` -> `"wanted"`, and nothing short of a
/// dictionary separates them) is a disclosed, pre-existing limitation of
/// the shared lemmatizer, not something this function invents a fix for
/// — those instances still round-trip (just to the wrong verb), so this
/// function returns a value for them; see this crate's own worked corpus
/// check (25 of 79 sampled technical verbs mis-lemmatized this way) for
/// the measured scope of that residual risk.
fn t10_derive_base_verb(comp_lemma: &str, comp_surface: &str) -> Option<String> {
    let round_trips =
        |candidate: &str| past_participle(candidate).eq_ignore_ascii_case(comp_surface);
    if round_trips(comp_lemma) {
        if !comp_lemma.ends_with('e') && t10_ends_in_bare_consonant_l(comp_lemma) {
            let restored = format!("{comp_lemma}e");
            if round_trips(&restored) {
                return Some(restored);
            }
        }
        return Some(comp_lemma.to_string());
    }
    if !comp_lemma.ends_with('e') {
        let restored = format!("{comp_lemma}e");
        if round_trips(&restored) {
            return Some(restored);
        }
    }
    None
}

/// `true` if `subj`'s subtree (`first..=last`) is safe to promote to the
/// front of the new imperative/infinitival clause.
///
/// Reuses [`within_bracketed_aside`] unchanged (a promoted subject inside
/// a parenthetical aside is displayed material, not asserted, exactly
/// [`object_is_licensable`]'s own reasoning) and the same no-post-modifier
/// / has-a-real-noun shape that function checks inline, adapted rather
/// than called directly: `object_is_licensable`'s own
/// `preposition_before_object`/`verb_child_after_object` checks are
/// scoped to T4's active-verb-then-object word order, which doesn't hold
/// here — a passive subject precedes its participle, not the reverse.
fn t10_subject_is_licensable(
    source: &str,
    tokens: &[TaggedToken],
    parse: &SentenceParse,
    subj_token: usize,
    first: usize,
    last: usize,
) -> bool {
    let has_post_modifier = (first..=last).any(|index| {
        index != subj_token
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
    let has_noun = (first..=last)
        .any(|index| matches!(tokens[index].pos.as_str(), "NN" | "NNS" | "NNP" | "NNPS"));
    if !has_noun {
        return false;
    }
    !within_bracketed_aside(source, tokens, first, last)
}

/// The re-inflection the derived main verb takes in
/// [`T10OuterShape::FiniteSubject`], to agree with the outer subject in
/// exactly the way `ensure`'s own surface form already does.
///
/// English subject-verb agreement for `ensure` and for the verb this
/// transducer derives from the embedded participle are the SAME
/// agreement problem — same subject, same tense — so the outer verb's
/// own tag already encodes the answer; there is no need to separately
/// classify the subject's own number the way
/// [`promoted_object_is_plural`]/[`token_takes_plural_agreement`] do for
/// T4. Those two exist because T4 promotes an object into a BRAND NEW
/// subject position with no pre-existing verb form to copy; here the
/// subject was never displaced; its agreement is already sitting on
/// `ensure`/`ensures`/`ensured`; the derived verb only has to reproduce
/// it. `VBP`/`VB` both take the base form: English's non-3rd-person
/// present is invariant across "I"/"we"/"you"/"they", so no plural
/// question ever arises for that case either.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FiniteAgreement {
    /// `"ensure"` (`VB`/`VBP`): non-3rd-person-singular present.
    Base,
    /// `"ensures"` (`VBZ`): 3rd-person-singular present.
    ThirdSg,
    /// `"ensured"` (`VBD`): simple past, invariant across subject number.
    Past,
}

/// The licensed outer mood shape a matched `ensure` token sits in, or
/// `None` for every other shape (a modal-headed or otherwise unparsable
/// `ensure`) — see [`t10_ensures_that_restructure`]'s own docs for why
/// only these three ship.
enum T10OuterShape {
    /// `ensure` opens the sentence, tagged `VB`: a bare imperative.
    SentenceInitialImperative,
    /// `ensure` is immediately preceded by infinitival `to`.
    Infinitival,
    /// `ensure` is the finite verb of its own clause, with an `Nsubj`
    /// child of its own (`"This ensures that X is enabled."`,
    /// `"These flags ensure that X is enabled."`,
    /// `"These checks ensured that X was enabled."`). The outer
    /// subject's own tokens are never touched — only the span from
    /// `ensure` through the embedded participle is replaced — so this
    /// carries no subject span of its own, just the re-inflection the
    /// replacement verb needs.
    FiniteSubject(FiniteAgreement),
}

fn t10_outer_shape(
    source: &str,
    tokens: &[TaggedToken],
    parse: &SentenceParse,
    ensure_index: usize,
) -> Option<T10OuterShape> {
    let tag = tokens[ensure_index].pos.as_str();
    if tag == "VB" {
        if ensure_index == 0 && is_imperative_initial(tokens) {
            return Some(T10OuterShape::SentenceInitialImperative);
        }
        if ensure_index > 0
            && tokens[ensure_index - 1].pos.as_str() == "TO"
            && token_text(source, tokens, ensure_index - 1).eq_ignore_ascii_case("to")
        {
            return Some(T10OuterShape::Infinitival);
        }
        return None;
    }
    let agreement = match tag {
        "VBZ" => FiniteAgreement::ThirdSg,
        "VBD" => FiniteAgreement::Past,
        "VBP" => FiniteAgreement::Base,
        _ => return None,
    };
    if t10_has_finite_subject(source, tokens, parse, ensure_index) {
        return Some(T10OuterShape::FiniteSubject(agreement));
    }
    None
}

/// `true` if `ensure_index` (already known to be finitely tagged) has an
/// outer subject licensing the [`T10OuterShape::FiniteSubject`] shape.
///
/// Two attachment shapes, both verified against the shipped parser
/// rather than assumed:
/// - a genuine `Nsubj` child (`"It ensures ..."`, `"This method ensures
///   ..."`) — the ordinary case; or
/// - a bare demonstrative pronoun standing alone as the subject, with no
///   head noun of its own to modify (`"This ensures ..."`, `"That
///   ensures ..."`). The shipped parser attaches this shape's `"this"`/
///   `"that"`/`"these"`/`"those"` via a `Det` edge landing directly on
///   `ensure_index` itself, not `Nsubj` — measured consistently across
///   every real and synthetic fixture tried (a determiner attaches to
///   the noun it modifies; landing on the verb instead only happens when
///   there is no such noun, i.e. the determiner IS the subject). This is
///   also the design brief's own first-named example
///   (`"This ensures that X is enabled."`) and the dominant real-corpus
///   shape for this construction, so covering it is not an edge case.
fn t10_has_finite_subject(
    source: &str,
    tokens: &[TaggedToken],
    parse: &SentenceParse,
    ensure_index: usize,
) -> bool {
    let nsubj: Vec<usize> = children_with_relation(parse, ensure_index, DepRelation::Nsubj)
        .map(|edge| edge.token)
        .collect();
    if matches!(nsubj.as_slice(), [_]) {
        return true;
    }
    let det: Vec<usize> = children_with_relation(parse, ensure_index, DepRelation::Det)
        .map(|edge| edge.token)
        .collect();
    let [det] = det.as_slice() else {
        return false;
    };
    matches!(
        folded_token_text(source, tokens, *det).as_str(),
        "this" | "that" | "these" | "those"
    )
}

/// "Ensure that NP is/are VBN" -> imperative/infinitival "VB NP" (T10):
/// `"Ensure that the setting is enabled."` -> `"Enable the setting."`.
///
/// T4 run in reverse, scoped to one complement: T4 walks down from a
/// finite verb to its `Nsubj`/`Dobj` children and promotes the object to
/// an agentless passive subject; this walks down from `ensure`'s `Ccomp`
/// child to find an already-passive embedded clause (`NsubjPass` +
/// `AuxPass` children on a `VBN` head — the same two-head passive shape
/// [`independent_clause_follows`]'s own docs describe) and collapses it
/// to an active, subject-less imperative or infinitival.
///
/// Licensed on the evidence already gauntlet-CONFIRMED for the pack row
/// this transducer replaces (`able.ensures-that-short::ensure`,
/// `frame-rules-en-v1.toml:1211`: 453.1/M machine vs. 66.9/M human, a
/// 6.8x tilt) — a construction whose template could never realize (see
/// this crate's own design notes), not new evidence.
///
/// # Detection order
/// 1. A token whose lemma is `"ensure"`, tagged `VB`/`VBP`/`VBZ`/`VBD`
///    (`VBG` never carries a finite or imperative/infinitival mood this
///    transducer produces, so it's excluded up front rather than
///    reaching the outer-shape check only to always fail it).
/// 2. A `Ccomp` child — call it `ccomp_target`.
/// 3. Either:
///    - `ccomp_target` tagged `VBN`: the standard shape. It has exactly
///      one `AuxPass` child whose surface (case-folded, curly
///      apostrophes straightened) is `"is"`/`"are"`/`"was"`/`"were"` or
///      one of [`T10_NEGATED_AUXPASS_FORMS`] — never `"being"`/`"been"`,
///      mirroring [`t9_past_progressive`]'s own stative-copula
///      exclusion — call it `aux`, and exactly one `NsubjPass` child —
///      call it `subj`; or
///    - `ccomp_target` itself folds to one of those same surfaces: the
///      aux-headed shape (§A2). The shipped parser heads a past-tense
///      embedded passive at the auxiliary itself, not the participle,
///      whenever an adverb separates them — verified against the
///      shipped parser on the design brief's own worked example ("to
///      ensure that event listeners **were** properly cleaned up" —
///      `Ccomp` lands on `"were"`, `VBD`, not on `"cleaned"`) and a
///      dozen further real and synthetic fixtures. Consistently, in
///      every instance of this shape: the subject attaches to
///      `ccomp_target` as a plain `Nsubj` (not `NsubjPass`) — call it
///      `subj` — and the true participle attaches as exactly one direct
///      child tagged `VBN` — call it `comp`; `ccomp_target` itself
///      becomes `aux`. Every case where the shipped parser produced an
///      inconsistent attachment for this shape (a mistagged
///      `VBD` participle instead of `VBN`, most often on a singular
///      `"was"` subject) is excluded by construction: no `VBN` child
///      means no match, the ordinary silent decline.
///
///    Anything else (a `JJ` adjectival complement, a `VB`/`VBZ`
///    intransitive one under a modal, `ccomp_target` folding to neither
///    a recognized surface nor `VBN`) means this `ensure` was never a
///    passive-complement candidate at all: no finding, the same silent
///    decline [`t4_activize_to_passive`] gives a verb that never
///    qualified.
/// 4. Take `subj`'s full subtree span via [`subtree_span`].
/// 5. The token immediately after `ensure` is literally `"that"`
///    (case-insensitive) — verified against the shipped parser's own
///    fixtures before this was locked down: every tested example
///    attaches `Mark` from `ccomp_target` to that same token, so the
///    literal adjacency check and a `Mark`-of-`ccomp_target` check agree
///    on every case tried; the literal check is used here as the
///    simpler, equally correct implementation. Its absence (`"ensure X
///    is enabled"`, no `"that"`) is out of scope for v1 and produces no
///    finding.
///
/// Every shape reaching this point genuinely matched the construction.
/// From here, a failing guard is a named [`RestructureOutcome::Declined`],
/// never silence:
/// 6. **Adverb gap**: the token right after `aux` must be `comp` (the
///    participle) itself. A gap ("was **properly** cleaned up") forces
///    an adverb-placement decision no table can make —
///    [`t9_past_progressive`]'s own precedent for the identical shape.
///    Every real aux-headed (§A2) instance measured so far has an
///    adverb in this gap — the very thing that causes the shipped
///    parser to head the clause at the auxiliary in the first place —
///    so this shape is expected to decline here far more often than it
///    accepts; that's the fail-closed law working as intended, not a
///    sign the shape is unreachable.
/// 7. **Negation**: any token in `comp`'s own subtree, immediately
///    before it, or `aux` itself, case-folding to one of
///    [`T10_NEGATION_WORDS`] — or `aux` itself being one of
///    [`T10_NEGATED_AUXPASS_FORMS`] — declines. The single
///    highest-stakes guard in this module: missing it turns "ensure
///    telemetry is **not** sent" into "Send telemetry", the opposite of
///    what the author wrote, not merely a bad rewrite.
/// 8. **Subject quality**: [`t10_subject_is_licensable`] over `subj`'s
///    subtree, and [`contains_markdown_structural_syntax`] over the
///    whole candidate span.
/// 9. **Lemma derivation**: [`t10_derive_base_verb`] must produce a
///    round-tripping base form.
/// 10. **Outer shape**: [`t10_outer_shape`] must license a
///     sentence-initial imperative, an infinitival "to ensure", or a
///     finite outer subject (§A1: `ensure`/`ensures`/`ensured` re-shown
///     on the derived verb via [`FiniteAgreement`]). Every other shape —
///     a modal-headed `ensure`, or any construction
///     [`t10_outer_shape`] doesn't recognize — declines.
#[must_use]
pub fn t10_ensures_that_restructure(
    source: &str,
    tokens: &[TaggedToken],
    parse: &SentenceParse,
) -> Vec<RestructureOutcome> {
    let mut out = Vec::new();
    for index in 0..tokens.len() {
        if tokens[index].lemma.as_ref() != "ensure" {
            continue;
        }
        if !matches!(tokens[index].pos.as_str(), "VB" | "VBP" | "VBZ" | "VBD") {
            continue;
        }
        let Some(m) = t10_match_passive_complement(source, tokens, parse, index) else {
            continue; // steps 2-5 (see this function's own docs): never matched, silent.
        };
        out.push(t10_license_and_realize(source, tokens, parse, index, &m));
    }
    out
}

/// Steps 2 through 5 of [`t10_ensures_that_restructure`]'s own detection
/// order: locates `ensure`'s `Ccomp` child, matches either the standard
/// or the aux-headed (§A2) shape, requires a literal `"that"` right
/// after `ensure`, and requires exactly one qualifying subject child.
/// `None` for any failure here — the construction never matched, so
/// [`t10_ensures_that_restructure`] emits no [`RestructureOutcome`] at
/// all for this `ensure` token.
///
/// `comp` always names the true participle token (`VBN`) and `aux`
/// always names the finite "be" token, regardless of which shape
/// matched — the field names are the shape-independent contract every
/// downstream guard in [`t10_license_and_realize`] relies on.
struct T10Match {
    comp: usize,
    aux: usize,
    subj: usize,
    subj_first: usize,
    subj_last: usize,
    negated_auxpass_form: bool,
}

fn t10_match_passive_complement(
    source: &str,
    tokens: &[TaggedToken],
    parse: &SentenceParse,
    ensure_index: usize,
) -> Option<T10Match> {
    let ccomp_target = children_with_relation(parse, ensure_index, DepRelation::Ccomp)
        .next()?
        .token;
    // The rewrite span runs from `ensure` to the participle, so the
    // complement must sit after the verb in token order. A parse that
    // attaches `Ccomp` leftward is wrong about this construction, and
    // slicing across it would panic or mis-span — decline instead.
    if ccomp_target <= ensure_index {
        return None;
    }
    let has_literal_that = tokens
        .get(ensure_index + 1)
        .is_some_and(|_| token_text(source, tokens, ensure_index + 1).eq_ignore_ascii_case("that"));
    if !has_literal_that {
        return None;
    }

    let (comp, aux, subj) = if tokens[ccomp_target].pos.as_str() == "VBN" {
        // Standard shape: `Ccomp` lands directly on the participle; the
        // finite "be" attaches below it as an `AuxPass` child, and the
        // promoted subject as an `NsubjPass` child.
        let auxpass: Vec<usize> = children_with_relation(parse, ccomp_target, DepRelation::AuxPass)
            .map(|edge| edge.token)
            .collect();
        let [aux] = auxpass.as_slice() else {
            return None;
        };
        let nsubjpass: Vec<usize> =
            children_with_relation(parse, ccomp_target, DepRelation::NsubjPass)
                .map(|edge| edge.token)
                .collect();
        let [subj] = nsubjpass.as_slice() else {
            return None;
        };
        (ccomp_target, *aux, *subj)
    } else {
        // Aux-headed shape (§A2): see `t10_ensures_that_restructure`'s
        // own docs for the parser behavior this matches. `ccomp_target`
        // itself plays `aux`; its one `VBN` child is the true
        // participle; the subject attaches to `ccomp_target` as a plain
        // `Nsubj`, the one attachment consistently observed for this
        // shape.
        let aux_folded = folded_token_text(source, tokens, ccomp_target);
        let is_recognized_form = T10_NEGATED_AUXPASS_FORMS.contains(&aux_folded.as_str())
            || matches!(aux_folded.as_str(), "is" | "are" | "was" | "were");
        if !is_recognized_form {
            return None; // not a "be" surface at all; never a candidate.
        }
        let vbn_children: Vec<usize> = parse
            .children_of(ccomp_target)
            .iter()
            .copied()
            .filter(|&child| tokens[child].pos.as_str() == "VBN")
            .collect();
        let [participle] = vbn_children.as_slice() else {
            return None; // zero, or more than one: no consistent participle to name.
        };
        if *participle <= ccomp_target {
            return None; // the participle always follows its auxiliary; a leftward one is a parse this shape doesn't trust.
        }
        let nsubj: Vec<usize> = children_with_relation(parse, ccomp_target, DepRelation::Nsubj)
            .map(|edge| edge.token)
            .collect();
        let [subj] = nsubj.as_slice() else {
            return None;
        };
        (*participle, ccomp_target, *subj)
    };

    let aux_folded = folded_token_text(source, tokens, aux);
    let negated_auxpass_form = T10_NEGATED_AUXPASS_FORMS.contains(&aux_folded.as_str());
    if !negated_auxpass_form && !matches!(aux_folded.as_str(), "is" | "are" | "was" | "were") {
        return None; // "being"/"been"/anything else.
    }

    let (subj_first, subj_last) = subtree_span(parse, subj);
    if subj_last >= aux {
        return None; // malformed span.
    }

    Some(T10Match {
        comp,
        aux,
        subj,
        subj_first,
        subj_last,
        negated_auxpass_form,
    })
}

/// Steps 6 through 10 of [`t10_ensures_that_restructure`]'s own detection
/// order, run once the construction is known to have matched: the
/// adverb-gap, negation, subject-quality, Markdown, lemma-derivation, and
/// outer-mood-shape guards, each producing a named
/// [`RestructureOutcome::Declined`] on failure, and a
/// [`RestructureOutcome::Candidate`] only once every guard passes.
fn t10_license_and_realize(
    source: &str,
    tokens: &[TaggedToken],
    parse: &SentenceParse,
    ensure_index: usize,
    m: &T10Match,
) -> RestructureOutcome {
    let range = tokens[ensure_index].token.range.start..tokens[m.comp].token.range.end;
    let decline = |reason| RestructureOutcome::Declined {
        range: range.clone(),
        reason,
    };

    if m.aux + 1 != m.comp {
        return decline("adverb between the auxiliary and the participle");
    }

    let negated = m.negated_auxpass_form
        || subtree_indices(parse, m.comp)
            .iter()
            .any(|&i| T10_NEGATION_WORDS.contains(&folded_token_text(source, tokens, i).as_str()))
        || (m.comp > 0
            && T10_NEGATION_WORDS
                .contains(&folded_token_text(source, tokens, m.comp - 1).as_str()));
    if negated {
        return decline("negation would invert the sentence's meaning");
    }

    if !t10_subject_is_licensable(source, tokens, parse, m.subj, m.subj_first, m.subj_last) {
        return decline("the promoted noun phrase is not a licensable subject");
    }
    if contains_markdown_structural_syntax(span_text(source, tokens, ensure_index, m.comp)) {
        return decline("candidate span contains Markdown structural syntax");
    }

    let comp_surface = token_text(source, tokens, m.comp);
    let Some(base_verb) = t10_derive_base_verb(tokens[m.comp].lemma.as_ref(), comp_surface) else {
        return decline("the participle's lemma did not round-trip to a base verb");
    };

    let subj_text = span_text(source, tokens, m.subj_first, m.subj_last);
    let Some(shape) = t10_outer_shape(source, tokens, parse, ensure_index) else {
        return decline("no licensed outer mood shape (unsupported construction)");
    };
    let replacement = match shape {
        T10OuterShape::SentenceInitialImperative => {
            format!("{} {subj_text}", recapitalize(&base_verb))
        }
        T10OuterShape::Infinitival => format!("{base_verb} {subj_text}"),
        T10OuterShape::FiniteSubject(agreement) => {
            let derived = match agreement {
                FiniteAgreement::Base => base_verb,
                FiniteAgreement::ThirdSg => third_sg(&base_verb),
                FiniteAgreement::Past => past(&base_verb),
            };
            format!("{derived} {subj_text}")
        }
    };

    RestructureOutcome::Candidate {
        range,
        replacement: replacement.into_boxed_str(),
    }
}

// ---------------------------------------------------------------------
// T11: dobj-gated transitive substitution (R2: `surface` -> `reveal`,
// generalized over `LEXICON_EN.transitive_verbs`).
// ---------------------------------------------------------------------

/// A licensed [`t11_transitive_substitution`] rewrite: replace `range`
/// (exactly the matched verb token) with `replacement`.
///
/// No `Declined` variant, unlike [`RestructureOutcome`]: every guard
/// here (no `Dobj`, an implausible object POS, an intervening
/// preposition/particle, no noun in the object subtree) is part of
/// deciding whether this verb token is a genuine dobj-headed transitive
/// use at all — the same matching question [`t4_activize_to_passive`]
/// answers for its own verb candidates — not a downstream policy call on
/// an already-matched shape. A `surface`-lemma token that fails any of
/// these is exactly the shape the unmodified `vsub.surface::*`
/// report-only frame-pack rows already account for; this module adds no
/// second finding for the same decline (see this crate's own R2 design
/// notes on why the decline side needs no new code path).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubstitutionCandidate {
    /// Byte range of the matched verb token in the original source.
    pub range: Range<usize>,
    /// The text to substitute for `range`.
    pub replacement: Box<str>,
}

/// `true` if `obj_token`'s subtree (`obj_first..=obj_last`) is a safe,
/// genuine direct object for [`t11_transitive_substitution`]'s verb
/// candidate at `verb_index`.
///
/// Three counter-hardening guards against the parser mis-attachment
/// failure mode T4/T9 already measured on real prose (see this crate's
/// own R2 design notes), reused from [`object_is_licensable`] rather
/// than duplicated logic:
/// - the object subtree starts strictly after the verb — a `Dobj` the
///   parser attached leftward is a parse this transducer doesn't trust,
///   the same defensive ordering check [`t10_match_passive_complement`]
///   applies to its own `Ccomp`/participle attachments;
/// - [`is_plausible_object_pos`] on the `Dobj` token itself;
/// - no preposition or particle (`IN`/`RP`) between the verb and the
///   object — the exact shape that produced T4's `"As this path is
///   continued"` failure, and the single likeliest way `"surfaced as a
///   significant contributor"` could be mis-parsed with `contributor` as
///   a `dobj` of `surfaced` instead of `pobj` of `as`: this guard makes
///   that mis-parse harmless even if the parser gets the relation wrong;
/// - a real noun tag present in the object subtree.
fn t11_object_is_licensable(
    tokens: &[TaggedToken],
    verb_index: usize,
    obj_token: usize,
    obj_first: usize,
    obj_last: usize,
) -> bool {
    if obj_first <= verb_index {
        return false; // the object always follows its verb; see this function's own docs.
    }
    if !is_plausible_object_pos(tokens[obj_token].pos.as_str()) {
        return false;
    }
    let preposition_before_object = (verb_index + 1..obj_first).any(|index| {
        tokens
            .get(index)
            .is_some_and(|t| matches!(t.pos.as_str(), "IN" | "RP"))
    });
    if preposition_before_object {
        return false;
    }
    (obj_first..=obj_last)
        .any(|index| matches!(tokens[index].pos.as_str(), "NN" | "NNS" | "NNP" | "NNPS"))
}

/// Dobj-gated transitive substitution (T11): `"This surfaced two more
/// tests."` -> `"This revealed two more tests."`.
///
/// Generalizes over every key in `LEXICON_EN.transitive_verbs`, not just
/// `"surface"` — the next dobj-gated substitution this workspace ships
/// is a one-line table addition, not new transducer code (see this
/// crate's own R2 design notes).
///
/// # Licensing conditions
/// All required, or the verb token is skipped (silently — see
/// [`SubstitutionCandidate`]'s own docs on why there is no decline-side
/// finding here):
/// - the token's lemma is a key in `LEXICON_EN.transitive_verbs`, tagged
///   `VB`/`VBZ`/`VBG`/`VBN`/`VBD` — the five tags the corpus-attested
///   `vsub.surface::*` frame rows already establish as worth covering
///   for this lemma family
/// - it has a `Dobj` child
/// - [`t11_object_is_licensable`] over that object's subtree
///
/// Realized via [`friction_nlp::inflect`], the same function
/// `frame_rules`'s own `TplOp::Inflect` arm calls: the matched verb's own
/// surface morphology (tense, number, aspect) transfers onto the
/// replacement lemma unchanged, so `"surfaced"` -> `"revealed"`,
/// `"surfacing"` -> `"revealing"`, `"surfaces"` -> `"reveals"`.
#[must_use]
pub fn t11_transitive_substitution(
    source: &str,
    tokens: &[TaggedToken],
    parse: &SentenceParse,
) -> Vec<SubstitutionCandidate> {
    let mut out = Vec::new();
    for index in 0..tokens.len() {
        if !matches!(
            tokens[index].pos.as_str(),
            "VB" | "VBZ" | "VBG" | "VBN" | "VBD"
        ) {
            continue;
        }
        let Some(target_lemma) = LEXICON_EN
            .transitive_verbs
            .get(tokens[index].lemma.as_ref())
        else {
            continue;
        };
        let Some(dobj) = children_with_relation(parse, index, DepRelation::Dobj).next() else {
            continue; // intransitive here: the vsub.* report-only rows already cover it.
        };
        let (obj_first, obj_last) = subtree_span(parse, dobj.token);
        if !t11_object_is_licensable(tokens, index, dobj.token, obj_first, obj_last) {
            continue;
        }
        let surface = token_text(source, tokens, index);
        let Some(replacement) = friction_nlp::inflect(surface, target_lemma) else {
            continue; // defensive: `inflect` only fails on a non-alphabetic token, never expected here.
        };
        out.push(SubstitutionCandidate {
            range: tokens[index].token.range.clone(),
            replacement: replacement.into_boxed_str(),
        });
    }
    out
}

// ---------------------------------------------------------------------
// T12: sentence-final participial-closer split.
// ---------------------------------------------------------------------

/// `true` if `prefix` (the tokens strictly before the licensing comma)
/// is a complete clause on its own: a finite verb somewhere, or an
/// imperative-initial bare verb.
///
/// The narrower of the two checks [`crate::gates::clause_ok`] runs in
/// `friction-edit` (that one also falls back to
/// `has_ambiguous_s_verb`, a tagger-specific compensation this crate has
/// no access to and no need of here — a false negative just declines
/// the candidate, never a false accept). Named separately from
/// [`has_finite_verb_cx`] because a contraction-fused clause boundary
/// (`"it's"`) is `t6_em_dash`'s own concern for a dash immediately
/// following it, not this construction's: T12's comma always sits
/// before a `VBG`, never before a fused subject+aux.
/// The main clause's last past-tense finite verb (`VBD`) surface, if
/// any — the tense donor for T12's promoted verb. Present finite verbs
/// (`VBZ`/`VBP`) return [`None`] on purpose: [`third_sg`] already
/// produces the right present form for the literal `"That"` subject,
/// and a `VBZ` donor like `"keeps"` would transfer third-singular
/// through `inflect` identically — only past needs the donor.
fn t12_main_clause_tense_donor<'a>(source: &'a str, prefix: &[TaggedToken]) -> Option<&'a str> {
    let index = prefix
        .iter()
        .rposition(|token| token.pos.as_str() == "VBD")?;
    Some(token_text(source, prefix, index))
}

fn t12_main_clause_is_complete(prefix: &[TaggedToken]) -> bool {
    has_finite_verb(prefix) || is_imperative_initial(prefix)
}

/// Derives the base-form ("VB") verb `vbg_surface` (a `VBG` token's own
/// surface text) inflects from, corroborated by round-tripping
/// [`friction_nlp::inflect`]'s gerund generation back against
/// `vbg_surface` — `None` if no candidate round-trips.
///
/// The `VBG` analogue of [`t10_derive_base_verb`], reusing its exact
/// two-step shape (trust the tagger's own lemma once it round-trips,
/// with [`t10_ends_in_bare_consonant_l`]'s silent-`"e"` correction
/// preferred when it also round-trips) and its exact same disclosed
/// scope: the shipped lemmatizer strips a silent `"e"` before both
/// `"-ed"` and `"-ing"` the same way, so the same wider class this
/// doesn't catch there (`"resolved"` -> `"resolv"`) doesn't get caught
/// here either (`"resolving"` -> `"resolv"`, which then round-trips on
/// its own merits — see [`t10_derive_base_verb`]'s own docs for why
/// nothing short of a dictionary separates that from a genuine e-less
/// verb). Verified against the shipped tagger: `"making"` -> `"make"`
/// and `"handling"`/`"enabling"` -> `"handl"`/`"enabl"`, corrected to
/// `"handle"`/`"enable"` by the bare-consonant-`"l"` step, exactly
/// mirroring T10's own `"enabled"`/`"disabled"` fixtures.
fn t12_derive_base_verb(vbg_lemma: &str, vbg_surface: &str) -> Option<String> {
    let round_trips = |candidate: &str| {
        friction_nlp::inflect(vbg_surface, candidate)
            .is_some_and(|generated| generated.eq_ignore_ascii_case(vbg_surface))
    };
    if round_trips(vbg_lemma) {
        if !vbg_lemma.ends_with('e') && t10_ends_in_bare_consonant_l(vbg_lemma) {
            let restored = format!("{vbg_lemma}e");
            if round_trips(&restored) {
                return Some(restored);
            }
        }
        return Some(vbg_lemma.to_string());
    }
    if !vbg_lemma.ends_with('e') {
        let restored = format!("{vbg_lemma}e");
        if round_trips(&restored) {
            return Some(restored);
        }
    }
    None
}

/// Sentence-final participial-closer split (T12).
///
/// A comma-attached `VBG` adjunct hanging off a complete main clause
/// becomes its own sentence, with a literal `"That"` subject and the
/// participle re-conjugated to agree with it: `"The tool automates the
/// whole pipeline, making it easy to onboard new projects."` -> `"The
/// tool automates the whole pipeline. That makes it easy to onboard new
/// projects."`.
///
/// Targets `participial_closer_rate`, a structural metric this
/// workspace's own holdout measured discriminative for machine prose
/// (see this crate's own design notes); no other restructuring op
/// touches it.
///
/// # Detection order
/// 1. A token tagged `VBG`, immediately preceded by a literal `","`
///    token (never a raw-text `", "` scan: token adjacency survives
///    irregular internal spacing the same way every other candidate in
///    this module is located).
/// 2. That `VBG` token's own incoming edge is [`DepRelation::Advcl`] —
///    the parser's own signal that its (understood) subject is the
///    CLAUSE the comma closes, not a coordinated noun. This is the
///    guard `"a bag containing apples, oranges, and pears"` needs:
///    verified against the shipped parser, a coordinated conjunct in a
///    list (`"...linting, testing, and deploying."`) attaches by
///    [`DepRelation::Conj`], and a noun's own reduced-relative modifier
///    (`"...bag containing apples..."`) attaches by
///    [`DepRelation::Acl`] — never `Advcl` — so this one check rejects
///    both shapes structurally rather than by guessing from the comma
///    alone. Verified too on a harder case: `"The service processes
///    orders, invoices, and receipts, generating a report for each."`
///    still matches here (`"generating"` attaches `Advcl` to the root
///    despite the internal list commas before it), correctly, since
///    that final comma really does close the whole clause.
/// 3. Sentence-final: this `VBG` token's own subtree (see
///    [`subtree_span`]) must reach exactly the token before the
///    sentence's own terminal punctuation (`"."`/`"!"`/`"?"`, the last
///    token in the sentence). A mid-sentence participial
///    (`"..., making X, the team shipped ..."`) has more sentence after
///    its subtree ends, so this declines to match at all — the same
///    silent-decline convention T10 gives a shape that was never a
///    candidate.
///
/// Every shape reaching this point genuinely matched the construction.
/// From here, a failing guard is a named [`RestructureOutcome::Declined`]:
/// 4. **Main-clause completeness**: [`t12_main_clause_is_complete`] over
///    the tokens strictly before the comma.
/// 5. **Markdown**: [`contains_markdown_structural_syntax`] over the
///    matched comma-through-`VBG` span.
/// 6. **Morphology**: [`t12_derive_base_verb`] must produce a
///    round-tripping base form, re-conjugated third-singular-present via
///    [`third_sg`] to agree with the literal `"That"` subject (the same
///    static morphology table [`t10_ensures_that_restructure`]'s own
///    finite-subject shape uses for its own re-inflection).
///
/// No negation guard, unlike T10: this construction never inverts a
/// polarity. It only re-punctuates and re-subjects an already-written
/// clause, copying everything after the `VBG` itself (any `"not"` it
/// already contains included) forward unchanged.
#[must_use]
pub fn t12_participial_closer_split(
    source: &str,
    tokens: &[TaggedToken],
    parse: &SentenceParse,
) -> Vec<RestructureOutcome> {
    let mut out = Vec::new();
    let Some(last_index) = tokens.len().checked_sub(1) else {
        return out;
    };
    let sentence_final_punct = matches!(token_text(source, tokens, last_index), "." | "!" | "?");

    for index in 0..tokens.len() {
        if tokens[index].pos.as_str() != "VBG" {
            continue;
        }
        if index == 0 || token_text(source, tokens, index - 1) != "," {
            continue; // no comma directly precedes; never a candidate.
        }
        let is_advcl = parse
            .edge(index)
            .is_some_and(|edge| edge.relation == DepRelation::Advcl);
        if !is_advcl {
            continue; // a coordinated conjunct (`Conj`) or a noun modifier (`Acl`); see this function's own docs.
        }
        if !sentence_final_punct {
            continue; // no terminal punctuation to anchor the sentence-final check against.
        }
        let (_, subtree_last) = subtree_span(parse, index);
        if subtree_last != last_index - 1 {
            continue; // more sentence follows this participial's own subtree; not sentence-final.
        }

        let comma_index = index - 1;
        let range = tokens[comma_index].token.range.start..tokens[index].token.range.end;
        let decline = |reason| RestructureOutcome::Declined {
            range: range.clone(),
            reason,
        };

        if !t12_main_clause_is_complete(&tokens[..comma_index]) {
            out.push(decline(
                "the clause preceding the comma is not a complete clause on its own",
            ));
            continue;
        }
        if contains_markdown_structural_syntax(span_text(source, tokens, comma_index, index)) {
            out.push(decline(
                "candidate span contains Markdown structural syntax",
            ));
            continue;
        }

        let vbg_lemma = tokens[index].lemma.as_ref();
        let vbg_surface = token_text(source, tokens, index);
        let Some(base_verb) = t12_derive_base_verb(vbg_lemma, vbg_surface) else {
            out.push(decline(
                "the participle's lemma did not round-trip to a base verb",
            ));
            continue;
        };

        // The promoted verb agrees with the MAIN clause's own tense, not
        // a fixed present: "kept accumulating objects, leading to an
        // increase" must become "That led to", not "That leads to" — a
        // present-tense aside inside past narrative reads wrong. The
        // main clause's last finite verb donates its form through
        // `inflect` (lowercased first, so a sentence-initial donor's
        // capital never leaks into the replacement). A generation-side
        // irregular hole ("letted") needs no guard here: the seam gate
        // downstream checks ("that", <verb>) against the attestation
        // tables, and a form no human writes is exactly a seam no
        // corpus attests.
        let donor = t12_main_clause_tense_donor(source, &tokens[..comma_index]);
        let promoted = if let Some(donor) = donor {
            let Some(form) = friction_nlp::inflect(&donor.to_lowercase(), &base_verb) else {
                out.push(decline(
                    "the main clause's tense could not be transferred to the promoted verb",
                ));
                continue;
            };
            form
        } else {
            third_sg(&base_verb)
        };
        let replacement = format!(". That {promoted}");
        out.push(RestructureOutcome::Candidate {
            range,
            replacement: replacement.into_boxed_str(),
        });
    }

    out
}

// ---------------------------------------------------------------------
// T13: em-dash relative-chain split ("— which is why X").
// ---------------------------------------------------------------------

/// The exact three-token frame [`t13_em_dash_relative_chain_split`]
/// requires immediately after its one em dash, case-folded: `"which"`,
/// `"is"`, `"why"`, in that order.
///
/// One literal frame only, deliberately: the design brief calls for v1
/// to stay narrow rather than generalize to every relative-chain shape a
/// dash can introduce (`"— which means"`, `"— which suggests"`, ...);
/// widening this table is a follow-up, not part of this construction's
/// own scope.
const T13_FRAME_WORDS: [&str; 3] = ["which", "is", "why"];

/// Em-dash relative-chain split (T13): `"The service handles routing —
/// which is why the config stays small."` -> `"The service handles
/// routing. That is why the config stays small."`.
///
/// Targets the same `em_dash`-adjacent structural tell T6 reduces, for
/// the one shape T6 itself declines: T6's own `independent_clause_candidate`
/// (case c) already recapitalizes past a bare independent clause after a
/// single dash, but a clause OPENED by `"which"` is a relative pronoun,
/// not a genuine new sentence subject — capitalizing "which" to "Which"
/// reads as broken. This module targets that one specific relative-chain
/// frame instead, swapping the dash and its relative pronoun for a
/// period and a demonstrative: `"— which is"` -> `". That is"`.
///
/// # Detection order
/// 1. Exactly one em-dash token in the sentence (see [`em_dash_tokens`]).
///    Zero has nothing to rewrite; two or more is the CLOSED-pair shape
///    (`"A — which is why B — and C."`) this v1 op declines outright —
///    too entangled with whatever the paired dashes already delimit to
///    decompose safely (the same class of shape [`t6_em_dash`] itself
///    declines for three-or-more dashes, one narrower).
/// 2. The dash neither opens nor closes the sentence (something on both
///    sides to anchor the rewrite to), and the three tokens right after
///    it case-fold to [`T13_FRAME_WORDS`] in order.
/// 3. The tokens before the dash carry a finite verb of their own
///    ([`has_finite_verb_cx`]) — the OPEN case's own left-hand clause
///    requirement (`"A"` in the design brief's own `"A — which is why
///    B."` shape). A verbless left side is a different construction (a
///    definition lead-in, T6's own territory), not this one.
///
/// No `Declined` variant for a shape that never reaches this point (a
/// missing frame, a verbless left side, a closed pair): the same silent
/// convention T11 uses for a lemma that was never a licensed candidate
/// at all. From here, a failing guard IS a named
/// [`RestructureOutcome::Declined`]:
/// 4. **Markdown**: [`contains_markdown_structural_syntax`] over the
///    matched `"— which is"` span.
///
/// No morphology to derive: `"is"` never changes form (the frame's own
/// verb already carries the number the replacement keeps), so this
/// candidate is realized directly once every guard above passes.
#[must_use]
pub fn t13_em_dash_relative_chain_split(
    source: &str,
    tokens: &[TaggedToken],
) -> Vec<RestructureOutcome> {
    let dashes = em_dash_tokens(source, tokens);
    let [dash] = dashes.as_slice() else {
        return Vec::new(); // zero, or a closed pair (two or more): out of scope for v1.
    };
    let dash = *dash;
    if dash == 0 || dash + T13_FRAME_WORDS.len() >= tokens.len() {
        return Vec::new(); // nothing on one side to anchor the rewrite, or no room for the frame.
    }
    let frame_matches = T13_FRAME_WORDS
        .iter()
        .enumerate()
        .all(|(offset, &word)| folded_token_text(source, tokens, dash + 1 + offset) == word);
    if !frame_matches {
        return Vec::new();
    }
    if !has_finite_verb_cx(source, tokens, 0..dash) {
        return Vec::new(); // the left side isn't its own clause; not this construction.
    }

    let is_index = dash + 2; // T13_FRAME_WORDS[1] == "is".
    let range = tokens[dash - 1].token.range.end..tokens[is_index].token.range.end;
    if spans_inline_code(&source[range.clone()]) {
        return Vec::new();
    }
    if contains_markdown_structural_syntax(&source[range.clone()]) {
        return vec![RestructureOutcome::Declined {
            range,
            reason: "candidate span contains Markdown structural syntax",
        }];
    }

    vec![RestructureOutcome::Candidate {
        range,
        replacement: Box::from(". That is"),
    }]
}

// ---------------------------------------------------------------------
// Inflection. `third_sg` is used only by T10's finite-subject shape
// (§A1) and `past` by both T9 and T10; `past_participle` ships alongside
// them since all three were audited as one unit — they now live in
// `friction_nlp::inflect`, beside the `LEXICON_EN` tables they read, and
// are re-exported here unchanged so external callers of
// `friction_register::transduce::{past, past_participle, third_sg}` keep
// working.
// ---------------------------------------------------------------------

pub use friction_nlp::{past, past_participle, third_sg};

#[cfg(test)]
mod t10_private_tests {
    use super::{t10_derive_base_verb, t10_ends_in_bare_consonant_l};

    /// The flagship silent-`"e"` correction: measured directly against
    /// the shipped tagger, `"enabled"` reverse-lemmatizes to `"enabl"`,
    /// not `"enable"` (see [`t10_derive_base_verb`]'s own docs) —
    /// [`t10_derive_base_verb`] must recover the real base form anyway.
    #[test]
    fn derive_base_verb_restores_silent_e_for_bare_consonant_l_stems() {
        assert_eq!(
            t10_derive_base_verb("enabl", "enabled").as_deref(),
            Some("enable")
        );
        assert_eq!(
            t10_derive_base_verb("disabl", "disabled").as_deref(),
            Some("disable")
        );
    }

    /// A lemma the tagger already got right (ends in `"e"`, or a bare
    /// consonant that doesn't need silent-`"e"` restoration) round-trips
    /// unchanged.
    #[test]
    fn derive_base_verb_trusts_an_already_correct_lemma() {
        assert_eq!(
            t10_derive_base_verb("configure", "configured").as_deref(),
            Some("configure")
        );
        assert_eq!(
            t10_derive_base_verb("clean", "cleaned").as_deref(),
            Some("clean")
        );
        assert_eq!(
            t10_derive_base_verb("want", "wanted").as_deref(),
            Some("want")
        );
    }

    /// A lemma that doesn't round-trip at all (no candidate regenerates
    /// the given surface) yields `None` — the fail-closed outcome a
    /// caller turns into a named decline rather than a guess.
    #[test]
    fn derive_base_verb_declines_on_a_genuine_round_trip_failure() {
        assert_eq!(t10_derive_base_verb("xyz", "enabled"), None);
    }

    /// [`t10_ends_in_bare_consonant_l`] is the narrow, structurally
    /// certain shape (a bare consonant immediately before a trailing
    /// `"l"`, no vowel between them) — true for `"enabl"`/`"disabl"`,
    /// false for a stem where the character before the trailing `"l"`
    /// is already a vowel (`"cancel"`, which needs no correction) or
    /// for a stem not ending in `"l"` at all.
    #[test]
    fn ends_in_bare_consonant_l_is_narrow() {
        assert!(t10_ends_in_bare_consonant_l("enabl"));
        assert!(t10_ends_in_bare_consonant_l("disabl"));
        assert!(!t10_ends_in_bare_consonant_l("cancel"));
        assert!(!t10_ends_in_bare_consonant_l("want"));
        assert!(!t10_ends_in_bare_consonant_l("l"));
    }
}
