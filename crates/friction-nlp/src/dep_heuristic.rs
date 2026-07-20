//! [`HeuristicParser`]: a [`DepParser`] built from part-of-speech-pattern
//! approximations rather than a trained model.
//!
//! Always available — no model, no cache directory, no cargo feature.
//! Targets exactly three patterns well enough for the rules that consume
//! them: a sentence's nominal subject and direct object (for same-subject
//! comparison between adjacent sentences via [`crate::dep::same_subject`]),
//! a sentence-final participial modifier, and flat `X, Y, and Z`
//! coordination lists. It assumes the token slice it is given has no
//! separate whitespace tokens between words (as [`crate::NlpruleTagger`]
//! produces), and it only ever recognizes a relation it has positive
//! evidence for — every other token gets [`DepRelation::Other`] at low
//! confidence, which callers should read as "no opinion", not as a
//! negative claim.
//!
//! # Known limitations
//!
//! The coordination detector resolves each conjunct's head as its
//! rightmost content word, which is right for flat noun-phrase lists
//! (`"screws, bolts, and washers"`) but not for verb-phrase coordination
//! sharing a subject (`"reviewed the docs, updated the tests, and shipped
//! the release"`, where the correct head of each conjunct is its verb, not
//! its trailing object) — that case is out of scope for this
//! implementation.

use friction_core::span;

use crate::dep::{Confidence, DepEdge, DepParseError, DepParser, DepRelation, SentenceParse};
use crate::tag::TaggedToken;

/// A [`DepParser`] built from part-of-speech-pattern heuristics. See the
/// module docs for what it recognizes and its known limitations.
#[derive(Debug, Clone, Copy, Default)]
pub struct HeuristicParser;

impl HeuristicParser {
    /// Creates a new heuristic parser. Stateless: every instance behaves
    /// identically.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl DepParser for HeuristicParser {
    fn parse(&self, source: &str, tokens: &[TaggedToken]) -> Result<SentenceParse, DepParseError> {
        if tokens.is_empty() {
            return SentenceParse::new(Vec::new());
        }

        let categories: Vec<Category> = tokens
            .iter()
            .map(|token| classify(text_of(source, token), token.pos.as_str()))
            .collect();
        let root = find_root(&categories);

        // Default: every non-root token is unclassified, pointed at the
        // root with low confidence. Recognized patterns below overwrite
        // specific entries.
        let mut relations = vec![(DepRelation::Other, Some(root), Confidence::LOW); tokens.len()];
        relations[root] = (DepRelation::Other, None, Confidence::CERTAIN);

        if let Some((index, confidence)) = find_subject(&categories, root) {
            relations[index] = (DepRelation::Subject, Some(root), confidence);
        }
        if let Some((index, confidence)) = find_object(&categories, root) {
            relations[index] = (DepRelation::Object, Some(root), confidence);
        }
        if let Some((index, confidence)) = find_participial_modifier(&categories)
            && index != root
        {
            relations[index] = (DepRelation::ParticipialModifier, Some(root), confidence);
        }
        for (index, anchor, confidence) in find_coordination(&categories) {
            if index != anchor {
                relations[index] = (DepRelation::Coordination, Some(anchor), confidence);
            }
        }

        let edges = relations
            .into_iter()
            .enumerate()
            .map(|(index, (relation, head, confidence))| DepEdge {
                token: index,
                head,
                relation,
                confidence,
            })
            .collect();
        SentenceParse::new(edges)
    }
}

/// A coarse syntactic category, collapsed from a tagger's part-of-speech
/// string plus (for punctuation, where the tag alone is ambiguous) the
/// token's exact surface text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Category {
    /// A noun or pronoun: a candidate subject, object, or coordination
    /// head.
    Nominal,
    /// A finite or base-form verb, or a modal: a root candidate.
    Verb,
    /// A `-ing`/`-ed` participle used as a modifier rather than the
    /// sentence's finite verb.
    Participle,
    /// A determiner, adjective, adverb, or numeral: skippable while
    /// scanning outward from the root for a subject or object.
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

/// Classifies one token's coarse category from its exact surface `text`
/// and its tagger-assigned `pos` string.
///
/// Punctuation is classified from `text` rather than `pos`, since which
/// exact mark a punctuation tag represents is otherwise tagger-defined;
/// every other category is read from the tagger's Penn-Treebank-style
/// `pos` string.
fn classify(text: &str, pos: &str) -> Category {
    match text {
        "," => return Category::Comma,
        "." | ";" | ":" | "!" | "?" => return Category::StrongBoundary,
        _ => {}
    }
    match pos {
        "PRP" | "WP" => Category::Nominal,
        "VBG" | "VBN" => Category::Participle,
        "MD" => Category::Verb,
        "CC" => Category::Coordinator,
        "JJ" | "JJR" | "JJS" | "RB" | "RBR" | "RBS" | "DT" | "PDT" | "WDT" | "PRP$" | "CD" => {
            Category::Modifier
        }
        p if p.starts_with("NN") => Category::Nominal,
        p if p.starts_with("VB") => Category::Verb,
        _ => Category::Other,
    }
}

/// Slices `token`'s exact surface text out of `source`, or `""` if its span
/// is somehow invalid — this must never panic, only degrade to "no
/// signal".
fn text_of<'s>(source: &'s str, token: &TaggedToken) -> &'s str {
    span::slice(source, &token.token.range).unwrap_or("")
}

/// The sentence's root candidate: its first finite/base verb or modal, or
/// token `0` if the sentence has none (a verbless fragment still needs a
/// root to attach everything else to).
fn find_root(categories: &[Category]) -> usize {
    categories
        .iter()
        .position(|category| matches!(category, Category::Verb))
        .unwrap_or(0)
}

/// Scans outward from `root` in `direction`, skipping [`Category::Modifier`]
/// tokens, and returns the first [`Category::Nominal`] token found. Stops
/// (returning `None`) the moment it crosses anything else — a clause
/// boundary the heuristic should not reach past.
fn scan_for_nominal(
    categories: &[Category],
    indices: impl Iterator<Item = usize>,
) -> Option<(usize, Confidence)> {
    let mut skipped = 0usize;
    for index in indices {
        match categories[index] {
            Category::Nominal => {
                let confidence = if skipped == 0 { 0.9 } else { 0.7 };
                return Some((index, Confidence::new(confidence)));
            }
            Category::Modifier => skipped += 1,
            _ => return None,
        }
    }
    None
}

/// The nearest nominal token before `root`, skipping determiners and
/// adjectives, stopping at the first clause boundary.
fn find_subject(categories: &[Category], root: usize) -> Option<(usize, Confidence)> {
    scan_for_nominal(categories, (0..root).rev())
}

/// The nearest nominal token after `root`, skipping determiners and
/// adjectives, stopping at the first clause boundary.
fn find_object(categories: &[Category], root: usize) -> Option<(usize, Confidence)> {
    scan_for_nominal(categories, (root + 1)..categories.len())
}

/// A trailing participial phrase: the sentence's last comma (before any
/// trailing strong punctuation), immediately followed by a participle, with
/// a main clause before it.
fn find_participial_modifier(categories: &[Category]) -> Option<(usize, Confidence)> {
    let mut end = categories.len();
    while end > 0 && matches!(categories[end - 1], Category::StrongBoundary) {
        end -= 1;
    }
    let comma = (0..end)
        .rev()
        .find(|&i| matches!(categories[i], Category::Comma))?;
    if comma == 0 {
        return None;
    }
    let participle = comma + 1;
    if participle < end && matches!(categories[participle], Category::Participle) {
        Some((participle, Confidence::new(0.7)))
    } else {
        None
    }
}

/// Every recognized coordination edge as `(dependent, anchor, confidence)`:
/// a flat `X, Y, and Z` list where `X`'s head becomes the anchor every
/// other conjunct's head attaches to.
fn find_coordination(categories: &[Category]) -> Vec<(usize, usize, Confidence)> {
    let mut result = Vec::new();
    for c in 0..categories.len() {
        if !matches!(categories[c], Category::Coordinator) {
            continue;
        }
        if c == 0 || !matches!(categories[c - 1], Category::Comma) {
            continue;
        }
        let comma_b = c - 1;
        let Some(comma_a) = (0..comma_b)
            .rev()
            .find(|&i| matches!(categories[i], Category::Comma))
        else {
            continue;
        };

        let seg1_start = (0..comma_a)
            .rev()
            .find(|&i| matches!(categories[i], Category::StrongBoundary | Category::Comma))
            .map_or(0, |i| i + 1);
        let seg3_end = ((c + 1)..categories.len())
            .find(|&i| matches!(categories[i], Category::StrongBoundary | Category::Comma))
            .unwrap_or(categories.len());

        let Some(anchor) = head_of_segment(categories, seg1_start..comma_a) else {
            continue;
        };
        for segment in [(comma_a + 1)..comma_b, (c + 1)..seg3_end] {
            if let Some(head) = head_of_segment(categories, segment) {
                result.push((head, anchor, Confidence::new(0.8)));
            }
        }
    }
    result
}

/// The rightmost strong content word (noun, verb, or participle) in
/// `range`, falling back to the rightmost modifier (adjective/numeral) if
/// the segment has no stronger candidate. `None` for an empty or
/// all-function-word segment.
fn head_of_segment(categories: &[Category], range: std::ops::Range<usize>) -> Option<usize> {
    range
        .clone()
        .rev()
        .find(|&i| {
            matches!(
                categories[i],
                Category::Nominal | Category::Verb | Category::Participle
            )
        })
        .or_else(|| {
            range
                .rev()
                .find(|&i| matches!(categories[i], Category::Modifier))
        })
}

#[cfg(test)]
mod tests {
    use friction_core::{Token, TokenKind};

    use super::*;
    use crate::dep::same_subject;
    use crate::tag::PosTag;

    /// Builds a sentence's tagged tokens from `(surface, pos, lemma)`
    /// triples, laying out contiguous byte spans (one space between
    /// tokens) so `source` and `tokens` agree, the way a real tagger's
    /// output would.
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
            tokens.push(TaggedToken {
                token: Token::new(start..end, TokenKind::Word),
                pos: PosTag::new(pos),
                lemma: lemma.into(),
            });
        }
        (source, tokens)
    }

    /// An empty token slice parses to an empty, valid `SentenceParse`
    /// rather than erroring.
    #[test]
    fn parse_accepts_empty_sentence() {
        let parse = HeuristicParser::new().parse("", &[]).unwrap();
        assert!(parse.edges().is_empty());
    }

    /// A flat `X, Y, and Z` noun list is recognized as coordination, with
    /// the first conjunct's head noun as the shared anchor, and the
    /// subject/object of the governing clause are still found correctly.
    #[test]
    fn parse_recognizes_triad_coordination() {
        let (source, tokens) = sentence(&[
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
        let parse = HeuristicParser::new().parse(&source, &tokens).unwrap();

        let screws = 3;
        let bolts = 5;
        let washers = 8;

        assert_eq!(parse.edge(screws).unwrap().relation, DepRelation::Object);

        let coordinated: Vec<_> = parse.by_relation(DepRelation::Coordination).collect();
        assert_eq!(coordinated.len(), 2);
        for edge in &coordinated {
            assert_eq!(edge.head, Some(screws));
        }
        assert!(coordinated.iter().any(|e| e.token == bolts));
        assert!(coordinated.iter().any(|e| e.token == washers));

        assert_eq!(parse.edge(1).unwrap().relation, DepRelation::Subject); // "kit"
    }

    /// A sentence-final participial phrase attaches to the main clause's
    /// verb with `DepRelation::ParticipialModifier`.
    #[test]
    fn parse_recognizes_trailing_participial_modifier() {
        let (source, tokens) = sentence(&[
            ("The", "DT", "the"),
            ("team", "NN", "team"),
            ("shipped", "VBD", "ship"),
            ("the", "DT", "the"),
            ("release", "NN", "release"),
            (",", ",", ","),
            ("raising", "VBG", "raise"),
            ("concerns", "NNS", "concern"),
            ("about", "IN", "about"),
            ("scalability", "NN", "scalability"),
            (".", ".", "."),
        ]);
        let parse = HeuristicParser::new().parse(&source, &tokens).unwrap();

        let shipped = 2;
        let raising = 6;
        let edge = parse.edge(raising).unwrap();
        assert_eq!(edge.relation, DepRelation::ParticipialModifier);
        assert_eq!(edge.head, Some(shipped));

        assert_eq!(parse.edge(1).unwrap().relation, DepRelation::Subject); // "team"
        assert_eq!(parse.edge(4).unwrap().relation, DepRelation::Object); // "release"
    }

    /// A sentence with no comma-plus-participle tail has no participial
    /// modifier — the detector does not fire on an ordinary sentence.
    #[test]
    fn parse_does_not_hallucinate_participial_modifier() {
        let (source, tokens) = sentence(&[
            ("The", "DT", "the"),
            ("team", "NN", "team"),
            ("shipped", "VBD", "ship"),
            ("the", "DT", "the"),
            ("release", "NN", "release"),
            (".", ".", "."),
        ]);
        let parse = HeuristicParser::new().parse(&source, &tokens).unwrap();
        assert!(
            parse
                .by_relation(DepRelation::ParticipialModifier)
                .next()
                .is_none()
        );
    }

    /// Two adjacent sentences that repeat the same head noun as their
    /// subject are recognized as sharing a subject via
    /// [`crate::dep::same_subject`], driven entirely by `HeuristicParser`
    /// output.
    #[test]
    fn heuristic_parses_feed_same_subject_detection() {
        let (source_a, tokens_a) = sentence(&[
            ("The", "DT", "the"),
            ("team", "NN", "team"),
            ("shipped", "VBD", "ship"),
            ("the", "DT", "the"),
            ("release", "NN", "release"),
            (".", ".", "."),
        ]);
        let (source_b, tokens_b) = sentence(&[
            ("The", "DT", "the"),
            ("team", "NN", "team"),
            ("also", "RB", "also"),
            ("fixed", "VBD", "fix"),
            ("a", "DT", "a"),
            ("regression", "NN", "regression"),
            (".", ".", "."),
        ]);
        let parser = HeuristicParser::new();
        let parse_a = parser.parse(&source_a, &tokens_a).unwrap();
        let parse_b = parser.parse(&source_b, &tokens_b).unwrap();

        assert!(same_subject((&tokens_a, &parse_a), (&tokens_b, &parse_b)));
    }

    /// A verbless fragment (no `Category::Verb` token at all) still parses
    /// without panicking or erroring, falling back to token `0` as root.
    #[test]
    fn parse_handles_verbless_fragment_without_panicking() {
        let (source, tokens) = sentence(&[
            ("Great", "JJ", "great"),
            ("news", "NN", "news"),
            ("!", "!", "!"),
        ]);
        let parse = HeuristicParser::new().parse(&source, &tokens).unwrap();
        assert_eq!(parse.root().unwrap().token, 0);
    }
}
