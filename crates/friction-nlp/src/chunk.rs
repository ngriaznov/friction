//! Clause segmentation and coordination chunking over a tagged token
//! stream.
//!
//! Two independent pieces live here:
//!
//! - Clause completeness: [`has_finite_verb`] / [`is_imperative_initial`]
//!   answer the whole-sentence question directly; [`chunk_clauses`] /
//!   [`is_complete_after_deletion`] answer the same question cheaply for a
//!   *hypothetical* deletion, without re-running the tagger on spliced
//!   text — each [`Clause`] records its own finite-verb token indices so a
//!   caller can subtract an index range instead.
//! - Coordination: [`coordination_groups`] detects flat `X, Y, and Z`
//!   lists directly over part-of-speech categories (see
//!   [`CoordCategory`]), the same pattern this crate's retired heuristic
//!   dependency parser used to detect [`crate::dep::DepRelation::Conj`]
//!   edges — kept here now that coordination chunking has no
//!   [`crate::dep::DepParser`] to route through: the arc-eager transition
//!   system that replaced the heuristic parser is a training-time oracle
//!   over already-known trees, not an inference-time parser this crate
//!   could call on arbitrary text.
//!
//! [`opens_with_binding_cue`] answers a third, related question (does this
//! sentence open with something a neighboring sentence's deletion could
//! leave dangling) but deliberately takes its "connective" word list as a
//! parameter — this crate sits below the pack layer and must not curate
//! that closed class itself.

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

use friction_core::TokenKind;

use crate::tag::TaggedToken;

/// Full Penn tags that count as a finite verb for clause completeness.
pub const FINITE_VERB_TAGS: [&str; 4] = ["VBZ", "VBP", "VBD", "MD"];

/// `true` if any token in `tokens` is tagged with one of
/// [`FINITE_VERB_TAGS`].
#[must_use]
pub fn has_finite_verb(tokens: &[TaggedToken]) -> bool {
    tokens
        .iter()
        .any(|t| FINITE_VERB_TAGS.contains(&t.pos.as_str()))
}

/// `true` if `tokens`' first token is tagged `VB` (a bare-infinitive verb
/// in sentence-initial position reads as an imperative).
#[must_use]
pub fn is_imperative_initial(tokens: &[TaggedToken]) -> bool {
    tokens.first().is_some_and(|t| t.pos.as_str() == "VB")
}

/// A closed list of subordinating conjunctions that open a subordinate
/// clause — a language fact (the same status as [`crate::lvc::LIGHT_VERBS`]
/// / [`crate::lvc::BE_FORMS`]), not pack curation, so it is hardcoded here
/// rather than threaded in by a caller.
const SUBORDINATORS: &[&str] = &[
    "because", "although", "though", "while", "since", "unless", "until", "before", "after", "if",
    "when", "whereas", "as",
];

/// `true` if `token` is a single `;`/`.`/`!`/`?` punctuation mark — a
/// strong boundary a clause never crosses.
fn is_strong_boundary(token: &TaggedToken) -> bool {
    token.token.kind == TokenKind::Punctuation
        && matches!(token.lemma.as_ref(), ";" | "." | "!" | "?")
}

/// `true` if `token` is a single `,` punctuation mark.
fn is_comma(token: &TaggedToken) -> bool {
    token.token.kind == TokenKind::Punctuation && token.lemma.as_ref() == ","
}

/// `true` if `token` is tagged `CC` (coordinating conjunction).
fn is_coordinator(token: &TaggedToken) -> bool {
    token.pos.as_str() == "CC"
}

/// `true` if a [`SUBORDINATORS`] match at token index `i` actually opens a
/// subordinate *clause* rather than a prepositional phrase — several
/// entries in that list (`while`, `before`, `after`, `since`, `as`, ...) are
/// also ordinary prepositions ("waited for **a while**", "here **since
/// Monday**"), and only a finite verb between the match and the next strong
/// boundary (or the stream's own end) distinguishes the two, the same gate
/// [`coordinator_splits`] already applies to its own coordinator match.
fn subordinator_opens_a_clause(tokens: &[TaggedToken], i: usize) -> bool {
    let right_end = ((i + 1)..tokens.len())
        .find(|&j| is_strong_boundary(&tokens[j]))
        .unwrap_or(tokens.len());
    tokens[(i + 1)..right_end]
        .iter()
        .any(|t| FINITE_VERB_TAGS.contains(&t.pos.as_str()))
}

/// Split points contributed by a comma-plus-coordinator that joins two
/// independent clauses (`"X did A, and X did B"`) as opposed to a
/// phrase-internal list (`"screws, bolts, and washers"`): the coordinator
/// must be preceded by a comma, and a finite verb must appear on both
/// sides of it before the next strong boundary (or the token stream's own
/// edge).
fn coordinator_splits(tokens: &[TaggedToken]) -> Vec<usize> {
    let mut splits = Vec::new();
    for c in 0..tokens.len() {
        if !is_coordinator(&tokens[c]) || c == 0 || !is_comma(&tokens[c - 1]) {
            continue;
        }
        let left_start = (0..c)
            .rev()
            .find(|&i| is_strong_boundary(&tokens[i]))
            .map_or(0, |i| i + 1);
        let right_end = ((c + 1)..tokens.len())
            .find(|&i| is_strong_boundary(&tokens[i]))
            .unwrap_or(tokens.len());
        let left_has_verb = tokens[left_start..c]
            .iter()
            .any(|t| FINITE_VERB_TAGS.contains(&t.pos.as_str()));
        let right_has_verb = tokens[(c + 1)..right_end]
            .iter()
            .any(|t| FINITE_VERB_TAGS.contains(&t.pos.as_str()));
        if left_has_verb && right_has_verb {
            splits.push(c);
        }
    }
    splits
}

/// One clause within a tagged token stream: a contiguous, non-overlapping
/// token range, plus which of its own tokens (by index into the *whole*
/// stream, not clause-relative) are finite verbs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Clause {
    /// This clause's token range, into the same stream [`chunk_clauses`]
    /// was called with.
    pub token_range: Range<usize>,
    /// Indices (into the whole stream) of this clause's finite-verb
    /// tokens.
    pub finite_verb_tokens: Vec<usize>,
    /// `true` if this clause's own first token is tagged `VB`.
    pub is_imperative_initial: bool,
}

/// A sentence's clause segmentation, in token-stream order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClauseChunks {
    pub clauses: Vec<Clause>,
}

/// Segments `tokens` into clauses.
///
/// Splits the stream at: a subordinator from the closed [`SUBORDINATORS`]
/// list that is gated by a finite verb on its right (see
/// [`subordinator_opens_a_clause`] — several subordinators are also
/// ordinary prepositions, and an ungated match would carve a verbless
/// prepositional phrase like "for a while" or "since Monday" off as its own
/// bogus clause); a comma-plus-coordinator that joins two independent
/// clauses (see [`coordinator_splits`]); and semicolons or sentence-strong
/// punctuation. Each resulting [`Clause`] records its own finite-verb token
/// indices, so [`is_complete_after_deletion`] can answer by index-set
/// subtraction instead of re-tagging spliced text.
#[must_use]
pub fn chunk_clauses(tokens: &[TaggedToken]) -> ClauseChunks {
    if tokens.is_empty() {
        return ClauseChunks {
            clauses: Vec::new(),
        };
    }

    let mut splits: BTreeSet<usize> = BTreeSet::from([0]);
    for (i, token) in tokens.iter().enumerate().skip(1) {
        if SUBORDINATORS.contains(&token.lemma.as_ref()) && subordinator_opens_a_clause(tokens, i) {
            splits.insert(i);
        }
    }
    for (i, token) in tokens.iter().enumerate() {
        if is_strong_boundary(token) && i + 1 < tokens.len() {
            splits.insert(i + 1);
        }
    }
    for c in coordinator_splits(tokens) {
        splits.insert(c);
    }

    let mut bounds: Vec<usize> = splits.into_iter().collect();
    bounds.push(tokens.len());

    let clauses = bounds
        .windows(2)
        .filter(|w| w[0] < w[1])
        .map(|w| {
            let range = w[0]..w[1];
            let finite_verb_tokens: Vec<usize> = range
                .clone()
                .filter(|&i| FINITE_VERB_TAGS.contains(&tokens[i].pos.as_str()))
                .collect();
            let is_imperative_initial = tokens[range.start].pos.as_str() == "VB";
            Clause {
                token_range: range,
                finite_verb_tokens,
                is_imperative_initial,
            }
        })
        .collect();

    ClauseChunks { clauses }
}

/// Would deleting token-index range `deleted` still leave the sentence
/// with a finite verb, or imperative-initial, without re-tagging?
///
/// Answers the same clause-completeness question [`has_finite_verb`] /
/// [`is_imperative_initial`] answer for a whole, already-tagged sentence,
/// but for a hypothetical edit: subtracts `deleted` from `chunks`'
/// precomputed finite-verb index sets, and checks whether the first
/// surviving token (by index, not re-tagged) is `VB`.
#[must_use]
pub fn is_complete_after_deletion(
    tokens: &[TaggedToken],
    chunks: &ClauseChunks,
    deleted: Range<usize>,
) -> bool {
    let has_remaining_finite_verb = chunks
        .clauses
        .iter()
        .flat_map(|c| c.finite_verb_tokens.iter())
        .any(|&i| !deleted.contains(&i));
    if has_remaining_finite_verb {
        return true;
    }
    (0..tokens.len())
        .find(|i| !deleted.contains(i))
        .is_some_and(|i| tokens[i].pos.as_str() == "VB")
}

/// A flat `X, Y, and Z` coordination group found by [`coordination_groups`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoordinationGroup {
    /// The list's first conjunct's head token index — the anchor every
    /// other conjunct's head attaches to (see
    /// [`crate::dep::DepRelation::Conj`]).
    pub anchor: usize,
    /// Every other conjunct's head token index, in token order.
    pub conjuncts: Vec<usize>,
    /// The smallest token range spanning `anchor` and every entry in
    /// `conjuncts`.
    pub token_range: Range<usize>,
}

/// A coarse category [`find_coordination`] scans over, folded from a
/// token's tagger-assigned part-of-speech tag plus (for punctuation, where
/// the tag alone is ambiguous) its own [`TokenKind`]/lemma — the same
/// signals [`is_comma`]/[`is_strong_boundary`]/[`is_coordinator`] above
/// already key off, rather than slicing surface text a second time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CoordCategory {
    /// A noun or pronoun: a candidate conjunct head.
    Nominal,
    /// A finite/base verb, modal, or participle: also a candidate conjunct
    /// head (coordination doesn't need to tell these apart the way
    /// subject/object/root detection would).
    VerbOrParticiple,
    /// A determiner, adjective, adverb, or numeral: skippable while
    /// scanning a segment for its head, and a last-resort head when a
    /// segment has nothing stronger.
    Modifier,
    /// A coordinating conjunction (`and`, `or`, `but`, ...).
    Coordinator,
    /// A comma.
    Comma,
    /// Sentence-internal strong punctuation (`.`, `;`, `:`, `!`, `?`) that
    /// bounds a clause.
    StrongBoundary,
    /// Anything not covered above (prepositions, particles, ...).
    Other,
}

/// Classifies one token's [`CoordCategory`].
fn coord_category(token: &TaggedToken) -> CoordCategory {
    if is_comma(token) {
        return CoordCategory::Comma;
    }
    if is_strong_boundary(token) {
        return CoordCategory::StrongBoundary;
    }
    if is_coordinator(token) {
        return CoordCategory::Coordinator;
    }
    match token.pos.as_str() {
        "PRP" | "WP" => CoordCategory::Nominal,
        "JJ" | "JJR" | "JJS" | "RB" | "RBR" | "RBS" | "DT" | "PDT" | "WDT" | "PRP$" | "CD" => {
            CoordCategory::Modifier
        }
        p if p.starts_with("NN") => CoordCategory::Nominal,
        p if p.starts_with("VB") => CoordCategory::VerbOrParticiple,
        _ => CoordCategory::Other,
    }
}

/// Every recognized coordination edge as `(dependent, anchor)`: a flat
/// `X, Y, and Z` list where `X`'s head becomes the anchor every other
/// conjunct's head attaches to.
fn find_coordination(categories: &[CoordCategory]) -> Vec<(usize, usize)> {
    let mut result = Vec::new();
    for c in 0..categories.len() {
        if !matches!(categories[c], CoordCategory::Coordinator) {
            continue;
        }
        if c == 0 || !matches!(categories[c - 1], CoordCategory::Comma) {
            continue;
        }
        let comma_b = c - 1;
        let Some(comma_a) = (0..comma_b)
            .rev()
            .find(|&i| matches!(categories[i], CoordCategory::Comma))
        else {
            continue;
        };

        let seg1_start = (0..comma_a)
            .rev()
            .find(|&i| {
                matches!(
                    categories[i],
                    CoordCategory::StrongBoundary | CoordCategory::Comma
                )
            })
            .map_or(0, |i| i + 1);
        let seg3_end = ((c + 1)..categories.len())
            .find(|&i| {
                matches!(
                    categories[i],
                    CoordCategory::StrongBoundary | CoordCategory::Comma
                )
            })
            .unwrap_or(categories.len());

        let Some(anchor) = head_of_segment(categories, seg1_start..comma_a) else {
            continue;
        };
        for segment in [(comma_a + 1)..comma_b, (c + 1)..seg3_end] {
            if let Some(head) = head_of_segment(categories, segment) {
                result.push((head, anchor));
            }
        }
    }
    result
}

/// The rightmost strong content word (noun, verb, or participle) in
/// `range`, falling back to the rightmost modifier (adjective/numeral) if
/// the segment has no stronger candidate. `None` for an empty or
/// all-function-word segment.
fn head_of_segment(categories: &[CoordCategory], range: Range<usize>) -> Option<usize> {
    range
        .clone()
        .rev()
        .find(|&i| {
            matches!(
                categories[i],
                CoordCategory::Nominal | CoordCategory::VerbOrParticiple
            )
        })
        .or_else(|| {
            range
                .rev()
                .find(|&i| matches!(categories[i], CoordCategory::Modifier))
        })
}

/// Groups coordination edges found in `tokens` by their shared anchor head.
#[must_use]
pub fn coordination_groups(tokens: &[TaggedToken]) -> Vec<CoordinationGroup> {
    let categories: Vec<CoordCategory> = tokens.iter().map(coord_category).collect();

    let mut by_anchor: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (dependent, anchor) in find_coordination(&categories) {
        if dependent != anchor {
            by_anchor.entry(anchor).or_default().push(dependent);
        }
    }

    by_anchor
        .into_iter()
        .map(|(anchor, mut conjuncts)| {
            conjuncts.sort_unstable();
            let min = conjuncts
                .iter()
                .copied()
                .chain([anchor])
                .min()
                .unwrap_or(anchor);
            let max = conjuncts
                .iter()
                .copied()
                .chain([anchor])
                .max()
                .unwrap_or(anchor);
            CoordinationGroup {
                anchor,
                conjuncts,
                token_range: min..(max + 1),
            }
        })
        .collect()
}

/// `true` if `range` partially overlaps any group in `groups`.
///
/// A partial overlap touches part of a coordinated list without covering
/// the whole thing, which would break a counted enumeration (`"three
/// things: X, Y, and Z"` with only `Y` deleted). A range that does not
/// intersect a group at all, or that fully contains it (deleting the
/// entire list), is not a partial overlap and does not count.
#[must_use]
pub fn overlaps_counted_enumeration(range: &Range<usize>, groups: &[CoordinationGroup]) -> bool {
    groups.iter().any(|g| {
        let intersects = range.start < g.token_range.end && g.token_range.start < range.end;
        if !intersects {
            return false;
        }
        let fully_contains_group =
            range.start <= g.token_range.start && g.token_range.end <= range.end;
        !fully_contains_group
    })
}

/// `true` if `tokens`' first token is a pronoun/demonstrative (`PRP`,
/// `PRP$`, `WP`, or `DT` immediately spelling `this`/`that`/`these`/
/// `those`), or its lowercased surface is a member of `connectives`.
///
/// `connectives` is supplied by the caller (a pack's own closed
/// connective list) rather than owned here — this crate sits below
/// `friction-packs` in the dependency graph and must not curate that class
/// itself; the pronoun/demonstrative check, by contrast, is pure grammar
/// and stays hardcoded.
#[must_use]
pub fn opens_with_binding_cue(
    source: &str,
    tokens: &[TaggedToken],
    connectives: &BTreeSet<&str>,
) -> bool {
    let Some(first) = tokens.first() else {
        return false;
    };
    match first.pos.as_str() {
        "PRP" | "PRP$" | "WP" => return true,
        "DT" => {
            let surface = friction_core::span::slice(source, &first.token.range)
                .unwrap_or("")
                .to_lowercase();
            if matches!(surface.as_str(), "this" | "that" | "these" | "those") {
                return true;
            }
        }
        _ => {}
    }
    connectives.contains(first.lemma.as_ref())
}

#[cfg(test)]
mod tests {
    use friction_core::{Token, TokenKind};

    use super::*;
    use crate::tag::PosTag;

    /// Builds a sentence's tagged tokens from `(surface, pos, lemma)`
    /// triples, laying out contiguous byte spans (one space between
    /// tokens) so the returned source and tokens agree — hand-built
    /// fixtures rather than a live-tagged sentence, so this module's tests
    /// exercise the clause/coordination *logic* directly rather than
    /// depending on a tagger's accuracy on arbitrary sentences.
    fn sentence(words: &[(&str, &str, &str)]) -> (String, Vec<TaggedToken>) {
        let mut source = String::new();
        let mut tokens = Vec::with_capacity(words.len());
        for &(surface, pos, lemma) in words {
            if !source.is_empty() {
                source.push(' ');
            }
            let start = source.len();
            source.push_str(surface);
            let end = source.len();
            let kind = if pos.chars().next().is_some_and(char::is_alphabetic) || pos == "CD" {
                if pos == "CD" {
                    TokenKind::Number
                } else {
                    TokenKind::Word
                }
            } else {
                TokenKind::Punctuation
            };
            tokens.push(TaggedToken {
                token: Token::new(start..end, kind),
                pos: PosTag::new(pos),
                lemma: lemma.into(),
            });
        }
        (source, tokens)
    }

    // ---------------------------------------------------------------
    // has_finite_verb / is_imperative_initial
    // ---------------------------------------------------------------

    #[test]
    fn has_finite_verb_true_for_a_sentence_with_a_finite_verb() {
        let (_, tokens) = sentence(&[
            ("The", "DT", "the"),
            ("tool", "NN", "tool"),
            ("scans", "VBZ", "scan"),
            ("the", "DT", "the"),
            ("project", "NN", "project"),
            (".", ".", "."),
        ]);
        assert!(has_finite_verb(&tokens));
    }

    #[test]
    fn has_finite_verb_false_for_a_verbless_fragment() {
        let (_, tokens) = sentence(&[
            ("A", "DT", "a"),
            ("great", "JJ", "great"),
            ("idea", "NN", "idea"),
            (".", ".", "."),
        ]);
        assert!(!has_finite_verb(&tokens));
    }

    #[test]
    fn is_imperative_initial_true_for_a_leading_bare_verb() {
        let (_, tokens) = sentence(&[
            ("Run", "VB", "run"),
            ("the", "DT", "the"),
            ("scanner", "NN", "scanner"),
            ("first", "RB", "first"),
            (".", ".", "."),
        ]);
        assert!(is_imperative_initial(&tokens));
    }

    #[test]
    fn is_imperative_initial_false_when_the_first_token_is_not_vb() {
        let (_, tokens) = sentence(&[
            ("The", "DT", "the"),
            ("scanner", "NN", "scanner"),
            ("runs", "VBZ", "run"),
            ("first", "RB", "first"),
            (".", ".", "."),
        ]);
        assert!(!is_imperative_initial(&tokens));
    }

    // ---------------------------------------------------------------
    // chunk_clauses
    // ---------------------------------------------------------------

    #[test]
    fn chunk_clauses_empty_input_produces_no_clauses() {
        let chunks = chunk_clauses(&[]);
        assert!(chunks.clauses.is_empty());
    }

    #[test]
    fn chunk_clauses_splits_a_subordinate_clause() {
        let (_, tokens) = sentence(&[
            ("The", "DT", "the"),
            ("build", "NN", "build"),
            ("failed", "VBD", "fail"),
            ("because", "IN", "because"),
            ("the", "DT", "the"),
            ("tests", "NNS", "test"),
            ("timed", "VBD", "time"),
            ("out", "RP", "out"),
            (".", ".", "."),
        ]);
        let chunks = chunk_clauses(&tokens);
        assert_eq!(chunks.clauses.len(), 2);
        assert_eq!(chunks.clauses[0].finite_verb_tokens.len(), 1);
        assert_eq!(chunks.clauses[1].finite_verb_tokens.len(), 1);
    }

    #[test]
    fn chunk_clauses_splits_coordinated_independent_clauses() {
        let (_, tokens) = sentence(&[
            ("The", "DT", "the"),
            ("team", "NN", "team"),
            ("shipped", "VBD", "ship"),
            ("the", "DT", "the"),
            ("release", "NN", "release"),
            (",", ",", ","),
            ("and", "CC", "and"),
            ("the", "DT", "the"),
            ("client", "NN", "client"),
            ("reviewed", "VBD", "review"),
            ("it", "PRP", "it"),
            (".", ".", "."),
        ]);
        let chunks = chunk_clauses(&tokens);
        assert_eq!(chunks.clauses.len(), 2);
        for clause in &chunks.clauses {
            assert!(
                !clause.finite_verb_tokens.is_empty(),
                "each coordinated clause should keep its own finite verb: {chunks:?}"
            );
        }
    }

    #[test]
    fn chunk_clauses_does_not_split_a_phrase_internal_and() {
        let (_, tokens) = sentence(&[
            ("The", "DT", "the"),
            ("kit", "NN", "kit"),
            ("includes", "VBZ", "include"),
            ("screws", "NNS", "screw"),
            (",", ",", ","),
            ("bolts", "NNS", "bolt"),
            (",", ",", ","),
            ("and", "CC", "and"),
            ("washers", "NNS", "washer"),
            (".", ".", "."),
        ]);
        let chunks = chunk_clauses(&tokens);
        assert_eq!(
            chunks.clauses.len(),
            1,
            "a flat noun list must not be mistaken for clause coordination: {chunks:?}"
        );
    }

    #[test]
    fn chunk_clauses_does_not_split_a_prepositional_use_of_a_subordinator() {
        // "while"/"before"/"after"/"since" are also ordinary prepositions;
        // with no finite verb on their right, a match must not carve off a
        // bogus verbless-fragment clause.
        let (_, tokens) = sentence(&[
            ("We", "PRP", "we"),
            ("waited", "VBD", "wait"),
            ("for", "IN", "for"),
            ("a", "DT", "a"),
            ("while", "NN", "while"),
            (".", ".", "."),
        ]);
        let chunks = chunk_clauses(&tokens);
        assert_eq!(
            chunks.clauses.len(),
            1,
            "a prepositional 'for a while' must not split into its own clause: {chunks:?}"
        );
    }

    #[test]
    fn chunk_clauses_splits_a_genuine_subordinate_clause_using_a_dual_role_word() {
        // The same word ("since") legitimately opens a subordinate clause
        // when a finite verb follows it.
        let (_, tokens) = sentence(&[
            ("The", "DT", "the"),
            ("team", "NN", "team"),
            ("celebrated", "VBD", "celebrate"),
            ("since", "IN", "since"),
            ("the", "DT", "the"),
            ("release", "NN", "release"),
            ("shipped", "VBD", "ship"),
            ("early", "RB", "early"),
            (".", ".", "."),
        ]);
        let chunks = chunk_clauses(&tokens);
        assert_eq!(chunks.clauses.len(), 2);
        assert_eq!(chunks.clauses[1].finite_verb_tokens.len(), 1);
    }

    #[test]
    fn chunk_clauses_splits_on_semicolons() {
        let (_, tokens) = sentence(&[
            ("The", "DT", "the"),
            ("build", "NN", "build"),
            ("failed", "VBD", "fail"),
            (";", ";", ";"),
            ("the", "DT", "the"),
            ("team", "NN", "team"),
            ("investigated", "VBD", "investigate"),
            ("immediately", "RB", "immediately"),
            (".", ".", "."),
        ]);
        let chunks = chunk_clauses(&tokens);
        assert_eq!(chunks.clauses.len(), 2);
    }

    // ---------------------------------------------------------------
    // is_complete_after_deletion
    // ---------------------------------------------------------------

    #[test]
    fn is_complete_after_deletion_true_when_a_finite_verb_survives() {
        let (_, tokens) = sentence(&[
            ("The", "DT", "the"),
            ("tool", "NN", "tool"),
            ("scans", "VBZ", "scan"),
            ("the", "DT", "the"),
            ("project", "NN", "project"),
            ("quickly", "RB", "quickly"),
            (".", ".", "."),
        ]);
        let chunks = chunk_clauses(&tokens);
        // Delete only the trailing adverb, leaving the finite verb intact.
        let last = tokens.len() - 2;
        assert!(is_complete_after_deletion(
            &tokens,
            &chunks,
            last..(last + 1)
        ));
    }

    #[test]
    fn is_complete_after_deletion_false_when_the_only_finite_verb_is_removed() {
        let (_, tokens) = sentence(&[
            ("The", "DT", "the"),
            ("tool", "NN", "tool"),
            ("scans", "VBZ", "scan"),
            ("the", "DT", "the"),
            ("project", "NN", "project"),
            (".", ".", "."),
        ]);
        let chunks = chunk_clauses(&tokens);
        let verb_index = tokens
            .iter()
            .position(|t| FINITE_VERB_TAGS.contains(&t.pos.as_str()))
            .expect("sentence has a finite verb");
        assert!(!is_complete_after_deletion(
            &tokens,
            &chunks,
            verb_index..(verb_index + 1)
        ));
    }

    #[test]
    fn is_complete_after_deletion_true_when_deletion_leaves_an_imperative() {
        let (_, tokens) = sentence(&[
            ("Please", "RB", "please"),
            ("run", "VB", "run"),
            ("the", "DT", "the"),
            ("scanner", "NN", "scanner"),
            (".", ".", "."),
        ]);
        let chunks = chunk_clauses(&tokens);
        // Delete "Please" (index 0), leaving "run the scanner." imperative.
        assert!(is_complete_after_deletion(&tokens, &chunks, 0..1));
    }

    // ---------------------------------------------------------------
    // coordination_groups / overlaps_counted_enumeration
    // ---------------------------------------------------------------

    #[test]
    fn coordination_groups_finds_the_triad_fixture() {
        let (_, tokens) = sentence(&[
            ("The", "DT", "the"),
            ("kit", "NN", "kit"),
            ("includes", "VBZ", "include"),
            ("screws", "NNS", "screw"),
            (",", ",", ","),
            ("bolts", "NNS", "bolt"),
            (",", ",", ","),
            ("and", "CC", "and"),
            ("washers", "NNS", "washer"),
            (".", ".", "."),
        ]);
        let groups = coordination_groups(&tokens);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].conjuncts.len(), 2);
    }

    #[test]
    fn coordination_groups_negative_case_phrase_internal_and_still_reported_as_flat_list() {
        // Not a "no coordination at all" case (a determiner-adjective-noun
        // phrase has no comma/coordinator shape to even attempt) -- the
        // negative case that matters here is that a plain modifier-noun
        // phrase with no list at all produces no coordination group.
        let (_, tokens) = sentence(&[
            ("The", "DT", "the"),
            ("quick", "JJ", "quick"),
            ("fox", "NN", "fox"),
            ("jumps", "VBZ", "jump"),
            (".", ".", "."),
        ]);
        let groups = coordination_groups(&tokens);
        assert!(groups.is_empty());
    }

    #[test]
    fn overlaps_counted_enumeration_flags_a_partial_deletion() {
        let (_, tokens) = sentence(&[
            ("The", "DT", "the"),
            ("kit", "NN", "kit"),
            ("includes", "VBZ", "include"),
            ("screws", "NNS", "screw"),
            (",", ",", ","),
            ("bolts", "NNS", "bolt"),
            (",", ",", ","),
            ("and", "CC", "and"),
            ("washers", "NNS", "washer"),
            (".", ".", "."),
        ]);
        let groups = coordination_groups(&tokens);
        assert_eq!(groups.len(), 1);
        let mid = groups[0].anchor + 1;
        assert!(overlaps_counted_enumeration(&(mid..(mid + 1)), &groups));
    }

    #[test]
    fn overlaps_counted_enumeration_ignores_a_disjoint_range() {
        let (_, tokens) = sentence(&[
            ("The", "DT", "the"),
            ("kit", "NN", "kit"),
            ("includes", "VBZ", "include"),
            ("screws", "NNS", "screw"),
            (",", ",", ","),
            ("bolts", "NNS", "bolt"),
            (",", ",", ","),
            ("and", "CC", "and"),
            ("washers", "NNS", "washer"),
            (".", ".", "."),
        ]);
        let groups = coordination_groups(&tokens);
        assert!(!overlaps_counted_enumeration(&(0..1), &groups));
    }

    #[test]
    fn overlaps_counted_enumeration_allows_deleting_the_whole_group() {
        let (_, tokens) = sentence(&[
            ("The", "DT", "the"),
            ("kit", "NN", "kit"),
            ("includes", "VBZ", "include"),
            ("screws", "NNS", "screw"),
            (",", ",", ","),
            ("bolts", "NNS", "bolt"),
            (",", ",", ","),
            ("and", "CC", "and"),
            ("washers", "NNS", "washer"),
            (".", ".", "."),
        ]);
        let groups = coordination_groups(&tokens);
        assert_eq!(groups.len(), 1);
        assert!(!overlaps_counted_enumeration(
            &groups[0].token_range,
            &groups
        ));
    }

    // ---------------------------------------------------------------
    // opens_with_binding_cue
    // ---------------------------------------------------------------

    #[test]
    fn opens_with_binding_cue_true_for_a_leading_pronoun() {
        let (source, tokens) = sentence(&[
            ("It", "PRP", "it"),
            ("handles", "VBZ", "handle"),
            ("the", "DT", "the"),
            ("rest", "NN", "rest"),
            ("automatically", "RB", "automatically"),
            (".", ".", "."),
        ]);
        let connectives = BTreeSet::new();
        assert!(opens_with_binding_cue(&source, &tokens, &connectives));
    }

    #[test]
    fn opens_with_binding_cue_true_for_a_leading_demonstrative() {
        let (source, tokens) = sentence(&[
            ("This", "DT", "this"),
            ("simplifies", "VBZ", "simplify"),
            ("the", "DT", "the"),
            ("setup", "NN", "setup"),
            ("considerably", "RB", "considerably"),
            (".", ".", "."),
        ]);
        let connectives = BTreeSet::new();
        assert!(opens_with_binding_cue(&source, &tokens, &connectives));
    }

    #[test]
    fn opens_with_binding_cue_true_for_a_caller_supplied_connective() {
        let (source, tokens) = sentence(&[
            ("However", "RB", "however"),
            (",", ",", ","),
            ("the", "DT", "the"),
            ("results", "NNS", "result"),
            ("held", "VBD", "hold"),
            ("up", "RP", "up"),
            (".", ".", "."),
        ]);
        let connectives = BTreeSet::from(["however"]);
        assert!(opens_with_binding_cue(&source, &tokens, &connectives));
    }

    #[test]
    fn opens_with_binding_cue_false_for_an_unrelated_opener() {
        let (source, tokens) = sentence(&[
            ("The", "DT", "the"),
            ("team", "NN", "team"),
            ("shipped", "VBD", "ship"),
            ("the", "DT", "the"),
            ("release", "NN", "release"),
            ("yesterday", "RB", "yesterday"),
            (".", ".", "."),
        ]);
        let connectives = BTreeSet::from(["however"]);
        assert!(!opens_with_binding_cue(&source, &tokens, &connectives));
    }

    #[test]
    fn opens_with_binding_cue_false_for_a_dt_that_is_not_demonstrative() {
        let (source, tokens) = sentence(&[
            ("A", "DT", "a"),
            ("team", "NN", "team"),
            ("shipped", "VBD", "ship"),
            ("it", "PRP", "it"),
            (".", ".", "."),
        ]);
        let connectives: BTreeSet<&str> = BTreeSet::new();
        assert!(!opens_with_binding_cue(&source, &tokens, &connectives));
    }

    #[test]
    fn opens_with_binding_cue_false_for_empty_tokens() {
        let connectives = BTreeSet::new();
        assert!(!opens_with_binding_cue("", &[], &connectives));
    }
}
