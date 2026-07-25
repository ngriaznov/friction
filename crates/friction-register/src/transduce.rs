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
use friction_nlp::{
    DepEdge, DepRelation, FINITE_VERB_TAGS, SentenceParse, TaggedToken, coarse_tag,
};

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
/// `they` is deliberately absent, against the reference's own table.
///
/// The other entries are non-referential in expository prose: `we` and `i`
/// are the author, `you` is the reader, `one` is people in general, and
/// demoting any of them removes a participant the reader can already
/// recover. `they` is anaphoric — it stands for a specific antecedent
/// somewhere earlier — so demoting it deletes information rather than
/// leaving it recoverable.
///
/// Measured on real prose: `"Browse the list of programmed channels to
/// ensure they match the uploaded file"` satisfies every other condition,
/// and passivizing it yields `"...to ensure the uploaded file is
/// matched"`, which quietly drops the claim that it is *the channels* that
/// must match. That is a meaning change, not a register change, and this
/// transducer is not permitted to make one.
const GENERIC_SUBJ: &[&str] = &["we", "i", "you", "one", "the team", "our team"];

/// Reflexive pronouns: never a licensed passive subject.
///
/// Measured directly against real prose: "I ... tore myself away" has a
/// generic subject ("I") and a direct object ("myself"), satisfying
/// every other T4 condition, but promoting a reflexive object produces
/// "Myself was torn away" — not merely awkward, ungrammatical, since a
/// reflexive pronoun has no independent referent to promote; it refers
/// back to the very subject the passive would delete. No fixed
/// replacement table paves over this the way [`GENERIC_SUBJ`]'s own
/// membership check does for the subject side, so this transducer simply
/// refuses to fire rather than launder a reflexive object into subject
/// position.
/// Personal pronouns are never a licensed object to promote.
///
/// Two independent reasons, either sufficient. A ditransitive verb takes
/// both an indirect and a direct object, and a parser routinely labels the
/// indirect one `dobj` — promoting it produces a sentence about the wrong
/// participant. And a pronoun object carries almost no information, so
/// moving it into subject position buys nothing even when it is the right
/// object.
///
/// Measured on real prose: `"see how much time and effort they save you"`
/// passes every structural guard, and passivizing on the labelled object
/// yields `"see how much time and effort you are saved"` — the promoted
/// participant is the beneficiary, while the real object (`time and
/// effort`) is left stranded in front of it.
///
/// Deliberately separate from [`REFLEXIVE_OBJ`] rather than merged into
/// one list: a reflexive is refused because it has no referent independent
/// of the subject, which is a fact about grammar. These are refused
/// because promoting them is either wrong or pointless, which is a
/// judgement about value. Merging the two would hide that difference from
/// anyone later deciding whether to relax one of them.
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
/// Markdown structural syntax in this workspace's prose (emphasis
/// asterisks, link/reference brackets, inline-code backticks).
///
/// Measured directly against real prose: a placeholder like
/// `"**[Proposed Cutover Date - e.g., October 26th, 2023]**"` is bridged
/// into one prose run rather than excluded the way inline code and link
/// URLs are (see `friction_parse::extract`'s own
/// `emphasis_markup_bridges_into_one_run` test), so its literal `*`/`[`/
/// `]` characters reach this transducer's dependency parse — which has no
/// training signal for Markdown syntax and produces an unreliable tree
/// over it. Refusing to promote a span containing any of these characters
/// is cheaper and more general than trying to parse Markdown correctly
/// here.
fn contains_markdown_structural_syntax(text: &str) -> bool {
    text.contains(['*', '[', ']', '`'])
}

/// `true` if `pos` is a part-of-speech tag a promoted passive subject can
/// plausibly be: a common or proper noun, a pronoun, or a number.
///
/// Guards against a parser mislabeling a predicative complement as a
/// `dobj` — measured directly against real prose where "felt most
/// productive" (an adjective complement of a copula-like verb, tag `JJ`)
/// and "inch closer" (an adverbial complement, tag `RBR`) were both
/// accepted as objects, producing "Most productive was feeled" and
/// "Closer is inched". A genuine direct object is always some kind of
/// nominal; anything else is a sign the parse mistook a complement for
/// one.
fn is_plausible_object_pos(pos: &str) -> bool {
    matches!(pos, "NN" | "NNS" | "NNP" | "NNPS" | "PRP" | "CD")
}

/// `true` if the span from token `first` to token `last` sits inside an
/// unclosed `(` or `[` earlier in `source`, or crosses either bracket.
///
/// Depth-counted rather than parity-counted, unlike the straight-quote
/// case elsewhere in this workspace: brackets are directional, so an
/// opener without its closer before the span is decisive on its own and
/// nesting has to be tracked rather than assumed away.
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
/// "Have" (possession/experience) and "become"/"remain"/"seem"/"appear"
/// (copulas taking a predicate-nominal complement, not a true object) are
/// excluded for the same reason: whatever a parser attaches as their
/// `dobj` is not an entity a passive can meaningfully promote, and every
/// one of these produces stilted or nonsensical output when forced
/// through the transform ("about a day of downtime was had", "problems
/// are become").
///
/// "Want"/"wish"/"prefer"/"need" are here for a related but distinct
/// reason, and it is worth keeping straight. They are ordinary transitive
/// verbs, so the transform is structurally legal; what fails is the
/// result. A passive of a desire reads as bureaucratic in a way the
/// active never does, and it also loses the one thing the sentence was
/// about — who wanted it. Measured on real prose: "the surprising
/// benefits we've found were wanted" and "Postgres' fine-grained control
/// was preferred" are both grammatical and both worse than what they
/// replaced.
///
/// That distinction matters if anyone later revisits this list: the first
/// group cannot be passivized, the second should not be.
const STATIVE_OR_LINKING_LEMMA: &[&str] = &[
    "have", "become", "remain", "seem", "appear", "want", "wish", "prefer", "need",
];

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
///
/// Split from [`t4_activize_to_passive`] itself only to keep that
/// function's own body a readable size: the verb-level guards below
/// ([`verb_is_licensable`]) and the object-level guards
/// ([`object_is_licensable`]) are still one connected licensing check,
/// read together with the loop that calls them.
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
        // Stative/linking verbs almost never passivize idiomatically in
        // English -- measured directly against real prose ("we had
        // about a day of downtime" -> "... was had"; "before they
        // become problems" -> "problems are become", where "become" is
        // a copula, not a transitive action verb, and its complement is
        // a predicate nominal a passive has nothing to promote away
        // from). Every other verb this transducer promotes is a
        // dynamic action verb, the class the agentless passive
        // construction is native to.
        return false;
    }
    if lemma.ends_with("ed") && !IRREGULAR_PAST.iter().any(|&(base, _)| base == lemma) {
        // A base-form English verb essentially never ends in "-ed" (the
        // rare genuine exception, "embed", is not worth the false-
        // negative risk this check accepts to avoid it); a lemma that
        // does is almost always a tagger reverse-lemmatization failure
        // that left an already-inflected surface form ("pinpointed",
        // "absorbed") in place of its base. Suffixing it again produced
        // "pinpointeded" and "absorbeded" against real prose -- not
        // awkward, a made-up word. Better to refuse than to compound a
        // known upstream inaccuracy into a worse one.
        return false;
    }
    let tag = tokens[index].pos.as_str();
    let surface_lower = token_text(source, tokens, index).to_lowercase();
    if matches!(tag, "VBD" | "VBZ")
        && lemma.eq_ignore_ascii_case(&surface_lower)
        && !IRREGULAR_PAST.iter().any(|&(base, _)| base == lemma)
    {
        // A `VBD` (simple past) or `VBZ` (third-person-singular present)
        // surface form is never identical to its own base lemma for a
        // regular verb -- unlike `VBP`, where "we deploy"/"they
        // encourage" legitimately has surface == base lemma and this
        // check must not fire. A `VBD`/`VBZ` token whose lemma matches
        // its surface and isn't one of the genuinely invariant verbs
        // this module's own irregular table already knows about
        // (`cut`, `put`, `hit`, ...) is a strong signal the tagger
        // simply failed to reduce an irregular or compound-irregular
        // verb ("rewrote") to its base form at all, rather than
        // evidence the lemma is correct. Regularly suffixing an
        // unreduced inflected form produced "rewroted" against real
        // prose; refusing is safer than compounding the inaccuracy.
        return false;
    }
    true
}

/// `true` if `obj_token`'s own subtree (spanning `obj_first..=obj_last`)
/// is safe to promote to subject position — see [`verb_is_licensable`]'s
/// own doc comment for why this is split out of
/// [`t4_activize_to_passive`] itself.
fn object_is_licensable(
    source: &str,
    tokens: &[TaggedToken],
    parse: &SentenceParse,
    verb_token: usize,
    obj_token: usize,
    obj_first: usize,
    obj_last: usize,
) -> bool {
    // The object must be a bare noun phrase: no post-modifier hanging off
    // it. A post-modified object is the single largest source of bad
    // output this transducer produces, and the damage is not
    // ungrammaticality -- it is silent meaning change, which no later gate
    // catches.
    //
    // Promotion moves the object's whole subtree to the front of the
    // clause. When that subtree ends in a prepositional phrase or an
    // infinitival clause, two things can go wrong and both were measured
    // on real prose. The modifier may actually attach to the verb rather
    // than the noun, in which case moving it rewrites what the sentence
    // claims: `"inspected each board for knots and defects"` became
    // `"each board for knots and defects was inspected"`, which no longer
    // says the inspection was *for* knots. Or the modifier genuinely
    // belongs to the noun, and hauling a long tail to the front strands
    // whatever followed it: `"wanted a more robust solution for ensuring
    // our background jobs completed reliably"` became `"...was wanted
    // reliably"`.
    //
    // Deliberately checked by relation rather than by looking for a
    // trailing preposition token. The failing cases include an infinitival
    // `to`, which is not a preposition, and the shared property is
    // structural -- something depends on the object other than the
    // determiners and modifiers that sit inside a simple noun phrase.
    // `Other` is permitted because this workspace's relation set collapses
    // compounds, numerals and possessives into it, all of which are
    // noun-internal and safe to move.
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
    // The head-tag check above trusts a single tag, which is exactly the
    // decision a tagger is most likely to get wrong on an unusual span.
    // Requiring a noun somewhere in the subtree is a second, independent
    // signal that survives one bad tag: `"Relatively low-impact and
    // predictable"` was promoted as an object because its head was
    // mistagged as a noun, and it produced `"Relatively low-impact and
    // predictable was started."` — a phrase with no noun in it at all is
    // never a noun phrase, whatever any single tag says.
    let has_noun = (obj_first..=obj_last)
        .any(|index| matches!(tokens[index].pos.as_str(), "NN" | "NNS" | "NNP" | "NNPS"));
    if !has_noun {
        return false;
    }

    // Never rewrite inside a bracketed or parenthesised aside.
    //
    // Square brackets in this genre mark a template placeholder the reader
    // is meant to fill in, and a parenthetical is an aside whose grammar
    // is scoped to itself. Rewriting into either produces text that reads
    // as broken even when the transformation is locally correct:
    // `"[Specific cabinet styles or features e.g., soft-close drawers are
    // desired, ...]"` rewrote a list item inside a fill-in-the-blank
    // placeholder, and `"(Sketch 1 is envisioned)"` rewrote across a
    // parenthesis boundary.
    //
    // Same reasoning as the existing guard against rewriting quoted text:
    // the author is displaying that material rather than asserting it, so
    // it is not their register to correct.
    if within_bracketed_aside(source, tokens, obj_first, obj_last) {
        return false;
    }

    // No preposition may stand between the verb and its object.
    //
    // A direct object sits adjacent to its verb, separated at most by the
    // determiners and modifiers inside its own noun phrase. A preposition
    // in that gap means the noun is the object of the *preposition*, and
    // the parser mislabelled it — which is a mistake to detect
    // structurally rather than one to trust.
    //
    // Measured on real prose: `"As we continue down this path"` parsed
    // with `path` as a direct object of `continue`, when it is really the
    // object of `down`. Passivizing it gave `"As this path is
    // continued"` — grammatical, and worse than what it replaced, because
    // the transform was applied to a relation that was not there.
    //
    // Particles (`RP`) are rejected alongside prepositions, which is the
    // less obvious half. `"we continue down this path"` and `"we gave up
    // the plan"` tag identically — `RP` particle, `dobj` noun — so no
    // structural rule separates them, and refusing both is the only
    // option that refuses the first. The apparent cost is the good
    // phrasal-verb passive (`"the server was set up"`); the measured cost
    // over the whole corpus was three rewrites, all three of them the
    // stilted case this guard exists to remove. The good phrasal case
    // does not survive the other conditions often enough to show up, so
    // the trade is real in principle and free in practice — worth
    // rechecking if those conditions ever loosen.
    let preposition_before_object = (verb_token + 1..obj_first).any(|index| {
        tokens
            .get(index)
            .is_some_and(|t| matches!(t.pos.as_str(), "IN" | "RP"))
    });
    if preposition_before_object {
        return false;
    }

    // Nothing that belongs to the verb may sit after the object.
    //
    // The rewrite replaces subject-through-object and leaves the rest of
    // the sentence untouched, which is only safe when the object ends its
    // clause. A verb argument sitting after it gets stranded in front of
    // nothing: `"We encourage contributions to Lumen"` became
    // `"Contributions are encouraged to Lumen"`, where `to Lumen`
    // modifies the verb and now reads as the destination of the
    // encouraging rather than of the contributions.
    //
    // This is the exact mirror of the post-modifier check above. That one
    // refuses to *move* material that should stay; this one refuses to
    // *leave* material that should move. Both failures come from the same
    // assumption -- that a clause is cleanly split at the object boundary
    // -- and only checking one side leaves the other class of damage
    // untouched, which is what the first pass over real prose showed.
    //
    // Punctuation is exempt: a trailing comma or period after the object
    // is not an argument and does not move.
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

    // The promoted object's own subtree must not contain a finite verb,
    // and the token immediately following it must not be one either.
    // Both guard the same real failure: the parser flattening an entire
    // embedded clause into what it labels a `dobj` -- measured against
    // real prose where "this reporting workload" was promoted from "we
    // anticipate this reporting workload will continue to grow",
    // stranding "will continue to grow" with no subject right after the
    // rewrite, and where a `dobj` subtree itself swallowed a full
    // relative-clause chain ("a signal ... that ... which ... when
    // ..."). A genuine simple NP object never contains, or directly
    // precedes, a finite verb.
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
        // The object subtree's own last token is a bare preposition or
        // infinitival "to" -- never how a genuine noun phrase ends.
        // Measured against real prose where a trailing preposition's
        // own argument attached elsewhere in the parse instead of under
        // it ("a rough budget range **of**" stopping right before the
        // dollar figure it should introduce), producing "... range of
        // was established" once promoted. Mirrors `t5_nominalization`'s
        // own stranded-tail guard, applied to the object side here
        // instead of the `pobj` argument side.
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

        // English "you" always takes plural verb agreement ("you are"/
        // "you were"), regardless of how many people it refers to --
        // measured against real prose where "you" as a promoted object
        // produced "You is encouraged" three times across the corpus.
        // "them" belongs in the same pronoun list as "they"/"these"/
        // "those" (all four are plural-referring pronouns none of which
        // carry an `NNS`-family tag) but was simply missing from it,
        // producing "Them is plucked".
        let plural = obj_surface == "you"
            || matches!(tokens[obj.token].pos.as_str(), "NNS" | "NNPS")
            || matches!(obj_surface.as_str(), "they" | "these" | "those" | "them");
        let be = match (tag == "VBD", plural) {
            (true, true) => "were",
            (true, false) => "was",
            (false, true) => "are",
            (false, false) => "is",
        };

        // Recapitalize the promoted object only when this candidate's own
        // range opens the sentence (`subj_first == 0`) -- exactly T5's
        // own convention (see `t5_nominalization`'s `det.token == 0`
        // check). Unconditional recapitalization was a real bug, not a
        // stylistic nuance: measured directly against real prose, most
        // T4 firings passivize a subordinate clause ("... when I found
        // an exception" -> "... when An exception was found"), where the
        // promoted object sits mid-sentence and must stay lowercase.
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
/// into a replacement that is not itself drawn from the matched span or
/// [`past_participle`]/[`past`]/[`third_sg`]'s own morphology.
///
/// Closed and exhaustive by design — see `friction-edit`'s register
/// closure gate, the one caller that consults this list: every "be" form
/// [`t4_activize_to_passive`] can possibly emit ("was"/"were" past,
/// "is"/"are" present) and nothing else. Widening this set silently would
/// widen what the closure gate accepts without anyone auditing the new
/// word, which is exactly the failure mode the gate exists to prevent.
pub const PERMITTED_FUNCTION_WORDS: [&str; 4] = ["was", "were", "is", "are"];

/// The verb [`t5_nominalization`] would substitute for `nominalization_lower`.
///
/// (Already lowercased.) The read side of [`NOMINAL_VERB`], the exact
/// table that licenses the transducer's own output, exposed so a caller
/// validating a candidate's closure (is this content word derivable from
/// the input?) consults the same lookup that produced it rather than a
/// second, independently-maintained copy.
#[must_use]
pub fn nominal_verb_for(nominalization_lower: &str) -> Option<&'static str> {
    NOMINAL_VERB
        .iter()
        .find(|&&(noun, _)| noun == nominalization_lower)
        .map(|&(_, verb)| verb)
}

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

        // Guard against a mis-attached compound-noun tail: measured
        // directly against real prose ("the integration of the
        // third-party analytics SDK"), where the parser attached the
        // phrase's own final noun ("SDK") to a token entirely outside
        // this construction rather than into the `pobj`'s subtree. A
        // subtree-only span then silently drops it -- "integrating the
        // third-party analytics" is missing its own object, not merely
        // awkward -- so this refuses to fire rather than ship a
        // content-losing rewrite. A genuine subtree boundary is never
        // immediately followed by a bare noun (a real one ends in
        // punctuation, a coordinator, or a verb); a following bare noun
        // is exactly the signature of a stranded compound-noun tail.
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

/// Irregular past-tense forms (53 entries -- the original 44 plus 9 common
/// invariant verbs the port initially missed: "hit", "cost", "cast",
/// "shut", "spread", "hurt", "quit", "burst", "shed" -- see this module's
/// own register-closure test suite, which caught "hit" producing
/// "hited" against real prose). Ported verbatim. `lemma +
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

/// Irregular past-participle forms (53 entries -- the same 9-entry
/// invariant-verb addition as [`IRREGULAR_PAST`]'s own doc comment
/// explains), ported verbatim as a table separate from [`IRREGULAR_PAST`]
/// rather than derived from it:
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

/// The past-tense form of `lemma`.
///
/// The irregular table's entry if one exists, otherwise the regular
/// `"-d"`/`"-ied"`/`"-ed"` suffix rule — delegated to
/// [`friction_nlp::inflect`] rather than reimplemented here a second
/// time.
///
/// This module's own regular-suffix rule originally had no consonant-
/// doubling logic (`"prefer"` -> `"prefered"` instead of `"preferred"`,
/// caught against real prose), which `friction_nlp::inflect`'s own
/// `generate_past` already implements and this crate's own inflection
/// tests already cover. Passing it the fixed surface `"used"` selects
/// exactly the past-tense form this function documents; `inflect` checks
/// its own (differently curated, but overlapping) irregular-verb table
/// first, a second independent safety net rather than a redundancy with
/// [`IRREGULAR_PAST`] above.
#[must_use]
pub fn past(lemma: &str) -> String {
    if let Some(&(_, irregular)) = IRREGULAR_PAST.iter().find(|&&(base, _)| base == lemma) {
        return irregular.to_string();
    }
    friction_nlp::inflect("used", lemma).unwrap_or_else(|| format!("{lemma}ed"))
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
