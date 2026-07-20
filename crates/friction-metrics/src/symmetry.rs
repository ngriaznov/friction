//! Structural and symmetry metrics: surface coordination and list/clause
//! parallelism patterns that recur unusually often in LLM-authored prose.
//!
//! Every public function here is a pure function of a [`Document`] plus a
//! [`Tagger`]: identical inputs produce identical output on any machine, on
//! any run. A [`Document`]'s prose is segmented into sentences elsewhere
//! (`friction-nlp`'s `segment_document`) but is not yet part-of-speech
//! tagged — tokenization and tagging are one pass, done on demand by a
//! [`Tagger`] — so these functions tag each sentence's text themselves,
//! walking the document's prose units and their sentences in source order,
//! and never reach for a dependency parser: the patterns below are shallow
//! surface patterns over a part-of-speech tag sequence, precise enough on
//! their own that layering `HeuristicParser`'s clause-structure guesses on
//! top would only add another source of error without sharpening what
//! "same broad class" or "present participle" already mean directly from
//! tags.

use std::ops::Range;

use friction_core::span::{contains_range, slice};
use friction_core::{Block, BlockKind, Document, ProseUnit, TokenKind};
use friction_nlp::{TaggedToken, Tagger};

// ---------------------------------------------------------------------
// Shared: coarse part-of-speech classes and sentence tagging
// ---------------------------------------------------------------------

/// A coarse grammatical bucket a token's part-of-speech tag folds into for
/// the "same broad class" comparisons the metrics in this module make.
///
/// Folding is driven entirely by the tagger's Penn-Treebank-style tag
/// prefix (`"NN"`, `"VB"`, `"JJ"`, `"RB"` — the same convention
/// `friction-nlp`'s heuristic dependency parser uses for its own coarse
/// categories). A tag matching none of those prefixes — including the
/// tagger's own `"UNKNOWN"` sentinel for an out-of-vocabulary word — folds
/// to [`BroadClass::Other`], its own comparable bucket rather than a
/// wildcard that would trivially "match" everything.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BroadClass {
    /// Any noun tag (`NN`, `NNS`, `NNP`, `NNPS`).
    Noun,
    /// Any verb tag, including participles and modals (`VB`, `VBZ`, `VBG`,
    /// `VBN`, `VBD`, `VBP`, `MD`).
    Verb,
    /// Any adjective tag (`JJ`, `JJR`, `JJS`).
    Adjective,
    /// Any adverb tag (`RB`, `RBR`, `RBS`).
    Adverb,
    /// Everything else: determiners, prepositions, punctuation, the
    /// unknown-POS sentinel, and any tag not covered above.
    Other,
}

/// Folds `pos` into its [`BroadClass`]. See the type's docs for the exact
/// rule.
fn broad_class(pos: &str) -> BroadClass {
    if pos.starts_with("NN") {
        BroadClass::Noun
    } else if pos.starts_with("VB") || pos == "MD" {
        BroadClass::Verb
    } else if pos.starts_with("JJ") {
        BroadClass::Adjective
    } else if pos.starts_with("RB") {
        BroadClass::Adverb
    } else {
        BroadClass::Other
    }
}

/// A token counts as a coordination/list item's "content word" — a
/// candidate to represent that item's grammatical class — when its broad
/// class is anything but [`BroadClass::Other`]: a noun, verb, adjective, or
/// adverb. Determiners, prepositions, and punctuation are skipped while
/// looking for one.
fn content_class(token: &TaggedToken) -> Option<BroadClass> {
    match broad_class(token.pos.as_str()) {
        BroadClass::Other => None,
        class => Some(class),
    }
}

/// The exact surface text `token` addresses in `source`, or `""` if its
/// span is somehow invalid — this must never panic, only degrade to "no
/// signal", since a metric is expected to produce a number for any input.
fn token_text<'s>(source: &'s str, token: &TaggedToken) -> &'s str {
    slice(source, &token.token.range).unwrap_or("")
}

fn is_comma(token: &TaggedToken, source: &str) -> bool {
    token_text(source, token) == ","
}

/// Sentence-internal strong punctuation that bounds a clause: a period,
/// exclamation mark, question mark, semicolon, or colon.
fn is_strong_boundary(token: &TaggedToken, source: &str) -> bool {
    matches!(token_text(source, token), "." | "!" | "?" | ";" | ":")
}

/// A listy coordinator: exactly `"and"` or `"or"` (case-insensitive),
/// tagged as a coordinating conjunction. Deliberately excludes `"but"`,
/// `"nor"`, and any other coordinator — the pattern these metrics target is
/// a parallel *list*, not clause coordination in general.
fn is_list_coordinator(token: &TaggedToken, source: &str) -> bool {
    token.pos.as_str().starts_with("CC")
        && matches!(
            token_text(source, token).to_ascii_lowercase().as_str(),
            "and" | "or"
        )
}

/// Tags every sentence in `document` with `tagger`, one token vector per
/// sentence, walking prose units and then sentences in source (document)
/// order — the same order every metric below accumulates in, so float
/// accumulation is order-stable across runs.
fn tagged_sentences(document: &Document, tagger: &dyn Tagger) -> Vec<Vec<TaggedToken>> {
    document
        .prose()
        .iter()
        .flat_map(|unit| unit.sentences.iter())
        .map(|sentence| {
            let text = document
                .text(&sentence.range)
                .expect("Sentence ranges are already validated by Document::new");
            tagger.tag(text, sentence.range.start)
        })
        .collect()
}

// ---------------------------------------------------------------------
// Triad rate
// ---------------------------------------------------------------------

/// The start index of the segment immediately preceding `end`: scanning
/// backward from `end - 1`, one past the nearest comma or strong-boundary
/// token, or `0` if none is found before the sentence's start.
fn segment_left_bound(tokens: &[TaggedToken], source: &str, end: usize) -> usize {
    (0..end)
        .rev()
        .find(|&i| is_comma(&tokens[i], source) || is_strong_boundary(&tokens[i], source))
        .map_or(0, |i| i + 1)
}

/// The end index (exclusive) of the segment starting at `start`: the
/// index of the nearest comma or strong-boundary token at or after
/// `start`, or `tokens.len()` if none is found before the sentence ends.
fn segment_right_bound(tokens: &[TaggedToken], source: &str, start: usize) -> usize {
    (start..tokens.len())
        .find(|&i| is_comma(&tokens[i], source) || is_strong_boundary(&tokens[i], source))
        .unwrap_or(tokens.len())
}

/// The nearest comma strictly before `end`, provided the scan does not
/// first cross a strong-boundary token (which would mean any comma beyond
/// it belongs to a different clause, not this one). `None` if no such
/// comma exists.
fn nearest_comma_before(tokens: &[TaggedToken], source: &str, end: usize) -> Option<usize> {
    for i in (0..end).rev() {
        if is_comma(&tokens[i], source) {
            return Some(i);
        }
        if is_strong_boundary(&tokens[i], source) {
            return None;
        }
    }
    None
}

/// The rightmost content word's [`BroadClass`] within `tokens[range]`, or
/// `None` if the segment has no content word at all.
fn segment_head_class(tokens: &[TaggedToken], range: Range<usize>) -> Option<BroadClass> {
    range.rev().find_map(|i| content_class(&tokens[i]))
}

/// Counts triad coordination patterns in one sentence's tagged tokens: a
/// three-part list of the surface shape `"X, Y, and Z"` / `"X, Y, or Z"`.
///
/// A match requires, precisely:
///
/// - a coordinator token (see [`is_list_coordinator`]) immediately
///   preceded by a comma (`comma_b`);
/// - a second comma (`comma_a`) found by scanning backward from `comma_b`
///   without first crossing a strong-boundary token, with at least one
///   token strictly between `comma_a` and `comma_b` (a non-empty middle
///   item);
/// - a non-empty first item, running from the nearest comma/strong
///   boundary before `comma_a` (or the sentence start) up to `comma_a`;
/// - a non-empty third item, running from the coordinator to the nearest
///   following comma/strong boundary (or the sentence end);
/// - each of the three items has a detectable head: scanning its tokens
///   from the end, the rightmost content word (see [`content_class`]). An
///   item with no content word at all (e.g. only a determiner) disqualifies
///   the whole pattern.
///
/// The three heads must fold to the *same* [`BroadClass`] — three nouns,
/// three verbs, three adjectives, or three adverbs. A mixed-class
/// three-part list ("fast, quiet, and it works") is ordinary coordination,
/// not the parallel-triad tic this metric targets.
///
/// The scan is left-to-right and resumes strictly after the coordinator
/// token whether or not it matched, so a sentence can contribute more than
/// one triad only if it contains more than one such non-overlapping
/// pattern.
fn count_triads(tokens: &[TaggedToken], source: &str) -> usize {
    let mut count = 0;
    let mut c = 1;
    while c < tokens.len() {
        if !is_list_coordinator(&tokens[c], source) {
            c += 1;
            continue;
        }
        let comma_b = c - 1;
        if !is_comma(&tokens[comma_b], source) {
            c += 1;
            continue;
        }
        let Some(comma_a) = nearest_comma_before(tokens, source, comma_b) else {
            c += 1;
            continue;
        };
        if comma_b - comma_a < 2 {
            c += 1;
            continue;
        }
        let seg1_start = segment_left_bound(tokens, source, comma_a);
        if seg1_start >= comma_a {
            c += 1;
            continue;
        }
        let seg3_end = segment_right_bound(tokens, source, c + 1);
        if seg3_end <= c + 1 {
            c += 1;
            continue;
        }

        let head1 = segment_head_class(tokens, seg1_start..comma_a);
        let head2 = segment_head_class(tokens, (comma_a + 1)..comma_b);
        let head3 = segment_head_class(tokens, (c + 1)..seg3_end);
        if let (Some(a), Some(b), Some(z)) = (head1, head2, head3)
            && a == b
            && b == z
        {
            count += 1;
        }
        c += 1;
    }
    count
}

/// Rate of triad coordination patterns (`"X, Y, and Z"`) per sentence.
///
/// The total number of matches found by [`count_triads`] across every
/// sentence in `document`, divided by the document's sentence count. `0.0`
/// for a document with no sentences.
#[must_use]
pub fn triad_rate(document: &Document, tagger: &dyn Tagger) -> f64 {
    let source = document.source();
    let sentences = tagged_sentences(document, tagger);
    if sentences.is_empty() {
        return 0.0;
    }
    let mut triads = 0usize;
    for tokens in &sentences {
        triads += count_triads(tokens, source);
    }
    #[allow(clippy::cast_precision_loss)]
    let rate = triads as f64 / sentences.len() as f64;
    rate
}

// ---------------------------------------------------------------------
// Participial-closer rate
// ---------------------------------------------------------------------

/// `true` if `tokens` (one sentence) ends in a present-participle phrase
/// set off by a comma: stripping any trailing strong-boundary tokens
/// (`.`, `!`, `?`, `;`, `:`), the sentence's last remaining comma is
/// immediately followed by a token tagged exactly `"VBG"` (a present
/// participle — `"making"`, `"allowing"`, `"helping"`), and that comma is
/// not the sentence's very first token (there must be a main clause before
/// it).
///
/// Deliberately narrower than a general trailing-participle check: a past
/// participle (`"VBN"`, e.g. `", written quickly."`) does not qualify —
/// the LLM tic this targets is specifically the dangling present-participle
/// closer (`", making it easier to..."`, `", allowing developers to..."`),
/// not passive-voice modification.
fn is_participial_closer(tokens: &[TaggedToken], source: &str) -> bool {
    let mut end = tokens.len();
    while end > 0 && is_strong_boundary(&tokens[end - 1], source) {
        end -= 1;
    }
    if end == 0 {
        return false;
    }
    let Some(comma) = (0..end).rev().find(|&i| is_comma(&tokens[i], source)) else {
        return false;
    };
    if comma == 0 {
        return false;
    }
    let participle = comma + 1;
    participle < end && tokens[participle].pos.as_str() == "VBG"
}

/// Rate of participial-closer sentences (see [`is_participial_closer`]) per
/// sentence: the fraction of `document`'s sentences that end this way.
/// `0.0` for a document with no sentences.
#[must_use]
pub fn participial_closer_rate(document: &Document, tagger: &dyn Tagger) -> f64 {
    let source = document.source();
    let sentences = tagged_sentences(document, tagger);
    if sentences.is_empty() {
        return 0.0;
    }
    let hits = sentences
        .iter()
        .filter(|tokens| is_participial_closer(tokens, source))
        .count();
    #[allow(clippy::cast_precision_loss)]
    let rate = hits as f64 / sentences.len() as f64;
    rate
}

// ---------------------------------------------------------------------
// Bullet-stem parallelism
// ---------------------------------------------------------------------

/// Reconstructs each block's immediate parent index from `blocks`' flat,
/// pre-order (source-ordered) layout: block `i`'s parent is the nearest
/// still-open ancestor whose range contains it, `None` for a top-level
/// block.
///
/// This is a plain containment stack, correct because of what
/// `friction-parse` already guarantees about `blocks`: it is pre-order
/// (non-decreasing start), sibling ranges never overlap, and a child's
/// range is always fully contained in its parent's — this function derives
/// nothing beyond that guarantee.
fn block_parents(blocks: &[Block]) -> Vec<Option<usize>> {
    let mut parents = vec![None; blocks.len()];
    let mut stack: Vec<usize> = Vec::new();
    for (index, block) in blocks.iter().enumerate() {
        while let Some(&top) = stack.last() {
            if contains_range(&blocks[top].range, &block.range) {
                break;
            }
            stack.pop();
        }
        parents[index] = stack.last().copied();
        stack.push(index);
    }
    parents
}

/// The index of the [`BlockKind::ListItem`] block that `block_index` most
/// directly belongs to: `block_index` itself if it is already a list item,
/// otherwise its nearest list-item ancestor. `None` if neither
/// `block_index` nor any of its ancestors is a list item.
///
/// Used to attribute a `ProseUnit` to the list item that directly owns it
/// — as opposed to an outer item that merely contains a nested sublist
/// this prose actually belongs to.
fn innermost_list_item(
    block_index: usize,
    blocks: &[Block],
    parents: &[Option<usize>],
) -> Option<usize> {
    let mut current = Some(block_index);
    while let Some(index) = current {
        if blocks[index].kind == BlockKind::ListItem {
            return Some(index);
        }
        current = parents[index];
    }
    None
}

/// The direct child [`BlockKind::ListItem`] blocks of the list at
/// `list_index`, in source order. Items belonging to a sublist nested
/// inside one of these items are not included — `CommonMark`'s grammar
/// makes a list item a direct child of exactly one list, so `parents[i] ==
/// Some(list_index)` alone is sufficient to select `list_index`'s own
/// items.
fn list_items(list_index: usize, blocks: &[Block], parents: &[Option<usize>]) -> Vec<usize> {
    blocks
        .iter()
        .enumerate()
        .filter(|&(i, block)| block.kind == BlockKind::ListItem && parents[i] == Some(list_index))
        .map(|(i, _)| i)
        .collect()
}

/// The earliest [`ProseUnit`] directly owned by the list item at
/// `item_index` — i.e. whose [`innermost_list_item`] is exactly
/// `item_index`, not a nested sublist's item. `document.prose()` is
/// already in source order, so the first match is the item's own leading
/// text (its first paragraph, for a loose item; its own inline text, for a
/// tight one).
fn leading_prose<'d>(
    item_index: usize,
    document: &'d Document,
    blocks: &[Block],
    parents: &[Option<usize>],
) -> Option<&'d ProseUnit> {
    document
        .prose()
        .iter()
        .find(|unit| innermost_list_item(unit.block, blocks, parents) == Some(item_index))
}

/// The [`BroadClass`] of a list item's first token — its "stem" — for the
/// item at `item_index`: the item's leading prose text (see
/// [`leading_prose`]) is tagged with `tagger`, and the class of the first
/// tagged token whose lexical kind is [`TokenKind::Word`] (skipping any
/// leading punctuation, e.g. a bolded lead-in's `**`) is returned.
///
/// `None` if the item has no directly-owned prose at all, or that prose
/// tags to no word token — an item lacking a detectable stem, which
/// [`list_parallelism_score`] buckets on its own rather than folding into
/// any word class.
fn item_stem_class(
    item_index: usize,
    document: &Document,
    blocks: &[Block],
    parents: &[Option<usize>],
    tagger: &dyn Tagger,
) -> Option<BroadClass> {
    let unit = leading_prose(item_index, document, blocks, parents)?;
    let text = document
        .text(&unit.range)
        .expect("ProseUnit ranges are already validated by Document::new");
    tagger
        .tag(text, 0)
        .into_iter()
        .find(|tagged| tagged.token.kind == TokenKind::Word)
        .map(|tagged| broad_class(tagged.pos.as_str()))
}

/// Counts of a list's items by their stem's [`BroadClass`], plus a
/// separate bucket for items with no detectable stem at all (see
/// [`item_stem_class`]). A fixed six-way tally rather than a hash map: the
/// bucket set is small and known ahead of time, and the largest count is
/// order-independent regardless of which items were recorded in which
/// order.
#[derive(Debug, Clone, Copy, Default)]
struct StemClassCounts {
    noun: usize,
    verb: usize,
    adjective: usize,
    adverb: usize,
    other: usize,
    none: usize,
}

impl StemClassCounts {
    const fn record(&mut self, class: Option<BroadClass>) {
        match class {
            Some(BroadClass::Noun) => self.noun += 1,
            Some(BroadClass::Verb) => self.verb += 1,
            Some(BroadClass::Adjective) => self.adjective += 1,
            Some(BroadClass::Adverb) => self.adverb += 1,
            Some(BroadClass::Other) => self.other += 1,
            None => self.none += 1,
        }
    }

    /// The size of this tally's largest bucket.
    const fn largest(&self) -> usize {
        let mut max = self.noun;
        if self.verb > max {
            max = self.verb;
        }
        if self.adjective > max {
            max = self.adjective;
        }
        if self.adverb > max {
            max = self.adverb;
        }
        if self.other > max {
            max = self.other;
        }
        if self.none > max {
            max = self.none;
        }
        max
    }
}

/// The parallelism score for one list: the fraction of its direct items
/// (see [`list_items`]) whose stem shares the [`BroadClass`] most common
/// among that list's items — `1.0` when every item's stem is the same
/// class (fully parallel), lower as the list mixes classes. `None` for a
/// list with no direct items (not expected from `friction-parse`'s
/// output, but handled rather than dividing by zero).
fn list_parallelism_score(
    list_index: usize,
    document: &Document,
    blocks: &[Block],
    parents: &[Option<usize>],
    tagger: &dyn Tagger,
) -> Option<f64> {
    let items = list_items(list_index, blocks, parents);
    if items.is_empty() {
        return None;
    }
    let mut counts = StemClassCounts::default();
    for &item in &items {
        counts.record(item_stem_class(item, document, blocks, parents, tagger));
    }
    #[allow(clippy::cast_precision_loss)]
    let score = counts.largest() as f64 / items.len() as f64;
    Some(score)
}

/// Bullet-stem parallelism score.
///
/// For every markdown list in `document` (top-level or nested — a nested
/// list contributes its own score independent of its containing list),
/// [`list_parallelism_score`]; aggregated as the mean over all lists,
/// walked in source (document) order. `0.0` for a document with no lists.
#[must_use]
pub fn bullet_parallelism(document: &Document, tagger: &dyn Tagger) -> f64 {
    let blocks = document.blocks();
    let parents = block_parents(blocks);
    let mut total = 0.0;
    let mut count = 0usize;
    for (index, block) in blocks.iter().enumerate() {
        if !matches!(block.kind, BlockKind::List { .. }) {
            continue;
        }
        if let Some(score) = list_parallelism_score(index, document, blocks, &parents, tagger) {
            total += score;
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        #[allow(clippy::cast_precision_loss)]
        let mean = total / count as f64;
        mean
    }
}

#[cfg(test)]
mod tests {
    use friction_core::{Sentence, Token, TokenKind as CoreTokenKind};
    use friction_nlp::PosTag;

    use super::*;

    // -------------------------------------------------------------
    // Test helpers
    // -------------------------------------------------------------

    /// Builds a sentence's tagged tokens from `(surface, pos, lemma)`
    /// triples, laying out contiguous byte spans (one space between
    /// tokens, no leading/trailing space) so the returned source string and
    /// token ranges agree, the way a real tagger's output would.
    fn build_tokens(words: &[(&str, &str, &str)]) -> (String, Vec<TaggedToken>) {
        let mut source = String::new();
        let mut tokens = Vec::with_capacity(words.len());
        for &(surface, pos, lemma) in words {
            if !source.is_empty() && surface != "," && surface != "." {
                source.push(' ');
            }
            let start = source.len();
            source.push_str(surface);
            let end = source.len();
            tokens.push(TaggedToken {
                token: Token::new(start..end, CoreTokenKind::Word),
                pos: PosTag::new(pos),
                lemma: lemma.into(),
            });
        }
        (source, tokens)
    }

    /// A stub [`Tagger`] that returns a fixed token vector for an exact
    /// text match and an empty vector otherwise, so the metric functions
    /// above can be exercised through the public `Document`-plus-`Tagger`
    /// API with hand-picked, tagger-quality-independent POS sequences.
    struct StubTagger(Vec<(&'static str, Vec<TaggedToken>)>);

    impl Tagger for StubTagger {
        fn tag(&self, text: &str, _base_offset: usize) -> Vec<TaggedToken> {
            self.0
                .iter()
                .find(|(key, _)| *key == text)
                .map(|(_, tokens)| tokens.clone())
                .unwrap_or_default()
        }
    }

    /// Builds a `Document` with one paragraph block holding `sentences`
    /// (each an exact substring of `source`, contiguous and covering all
    /// of it) so `tagged_sentences` walks them in the given order.
    fn document_of(source: &str, sentence_ranges: &[Range<usize>]) -> Document {
        let blocks = vec![Block::new(BlockKind::Paragraph, 0..source.len())];
        let sentences = sentence_ranges
            .iter()
            .cloned()
            .map(|range| Sentence::new(range, Vec::new()))
            .collect();
        let prose = vec![ProseUnit::new(0, 0..source.len(), sentences)];
        Document::new(source, blocks, prose).expect("well-formed test document")
    }

    // -------------------------------------------------------------
    // triad_rate / count_triads
    // -------------------------------------------------------------

    /// A flat `"X, Y, and Z"` noun list, all three heads nouns: one triad.
    #[test]
    fn count_triads_matches_same_class_noun_triad() {
        let (source, tokens) = build_tokens(&[
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
        assert_eq!(count_triads(&tokens, &source), 1);
    }

    /// Heads of different broad classes (noun, verb, noun) do not count as
    /// a triad, even though the comma/coordinator shape matches.
    #[test]
    fn count_triads_rejects_mixed_class_heads() {
        let (source, tokens) = build_tokens(&[
            ("screws", "NNS", "screw"),
            (",", ",", ","),
            ("ran", "VBD", "run"),
            (",", ",", ","),
            ("and", "CC", "and"),
            ("washers", "NNS", "washer"),
            (".", ".", "."),
        ]);
        assert_eq!(count_triads(&tokens, &source), 0);
    }

    /// `"but"` is a coordinator but not a listy one — no triad, even with
    /// an otherwise-matching comma-comma shape.
    #[test]
    fn count_triads_rejects_non_listy_coordinator() {
        let (source, tokens) = build_tokens(&[
            ("screws", "NNS", "screw"),
            (",", ",", ","),
            ("bolts", "NNS", "bolt"),
            (",", ",", ","),
            ("but", "CC", "but"),
            ("washers", "NNS", "washer"),
            (".", ".", "."),
        ]);
        assert_eq!(count_triads(&tokens, &source), 0);
    }

    /// An empty middle item (two adjacent commas) disqualifies the match.
    #[test]
    fn count_triads_rejects_empty_middle_item() {
        let (source, tokens) = build_tokens(&[
            ("screws", "NNS", "screw"),
            (",", ",", ","),
            (",", ",", ","),
            ("and", "CC", "and"),
            ("washers", "NNS", "washer"),
            (".", ".", "."),
        ]);
        assert_eq!(count_triads(&tokens, &source), 0);
    }

    /// Three adjectives coordinated ("fast, quiet, and reliable") count as
    /// a triad too — the same-class rule is not noun-only.
    #[test]
    fn count_triads_matches_same_class_adjective_triad() {
        let (source, tokens) = build_tokens(&[
            ("It", "PRP", "it"),
            ("was", "VBD", "be"),
            ("fast", "JJ", "fast"),
            (",", ",", ","),
            ("quiet", "JJ", "quiet"),
            (",", ",", ","),
            ("and", "CC", "and"),
            ("reliable", "JJ", "reliable"),
            (".", ".", "."),
        ]);
        assert_eq!(count_triads(&tokens, &source), 1);
    }

    /// `triad_rate` divides the total triad count by the sentence count,
    /// across a two-sentence document, wired through the public
    /// `Document`-plus-`Tagger` API with a stub tagger.
    #[test]
    fn triad_rate_averages_over_sentences() {
        let (triad_text, triad_tokens) = build_tokens(&[
            ("screws", "NNS", "screw"),
            (",", ",", ","),
            ("bolts", "NNS", "bolt"),
            (",", ",", ","),
            ("and", "CC", "and"),
            ("washers", "NNS", "washer"),
            (".", ".", "."),
        ]);
        let (plain_text, plain_tokens) = build_tokens(&[
            ("It", "PRP", "it"),
            ("shipped", "VBD", "ship"),
            (".", ".", "."),
        ]);

        let source = format!("{triad_text} {plain_text}");
        let boundary = triad_text.len();
        let document = document_of(&source, &[0..boundary, (boundary + 1)..source.len()]);
        let tagger = StubTagger(vec![
            (triad_text.leak() as &str, triad_tokens),
            (plain_text.leak() as &str, plain_tokens),
        ]);

        assert!((triad_rate(&document, &tagger) - 0.5).abs() < f64::EPSILON);
    }

    /// A document with no sentences has a `triad_rate` of `0.0`, not a
    /// division-by-zero panic.
    #[test]
    fn triad_rate_zero_for_no_sentences() {
        let document = document_of("", &[]);
        let tagger = StubTagger(Vec::new());
        assert!(triad_rate(&document, &tagger).abs() < f64::EPSILON);
    }

    // -------------------------------------------------------------
    // participial_closer_rate / is_participial_closer
    // -------------------------------------------------------------

    /// A trailing `", raising concerns..."` present-participle phrase set
    /// off by a comma is recognized.
    #[test]
    fn is_participial_closer_recognizes_trailing_vbg_clause() {
        let (source, tokens) = build_tokens(&[
            ("The", "DT", "the"),
            ("team", "NN", "team"),
            ("shipped", "VBD", "ship"),
            ("the", "DT", "the"),
            ("release", "NN", "release"),
            (",", ",", ","),
            ("raising", "VBG", "raise"),
            ("concerns", "NNS", "concern"),
            (".", ".", "."),
        ]);
        assert!(is_participial_closer(&tokens, &source));
    }

    /// A past participle (`VBN`) after the comma does not qualify — only
    /// the present-participle (`VBG`) form does.
    #[test]
    fn is_participial_closer_rejects_past_participle() {
        let (source, tokens) = build_tokens(&[
            ("The", "DT", "the"),
            ("team", "NN", "team"),
            ("shipped", "VBD", "ship"),
            ("the", "DT", "the"),
            ("release", "NN", "release"),
            (",", ",", ","),
            ("delayed", "VBN", "delay"),
            ("twice", "RB", "twice"),
            (".", ".", "."),
        ]);
        assert!(!is_participial_closer(&tokens, &source));
    }

    /// A sentence with no comma at all has no participial closer.
    #[test]
    fn is_participial_closer_rejects_sentence_without_comma() {
        let (source, tokens) = build_tokens(&[
            ("The", "DT", "the"),
            ("team", "NN", "team"),
            ("shipped", "VBD", "ship"),
            ("the", "DT", "the"),
            ("release", "NN", "release"),
            (".", ".", "."),
        ]);
        assert!(!is_participial_closer(&tokens, &source));
    }

    /// A comma at the very start of the sentence has no main clause before
    /// it, so it does not qualify even though a `VBG` immediately follows.
    #[test]
    fn is_participial_closer_rejects_leading_comma() {
        let (source, tokens) = build_tokens(&[
            (",", ",", ","),
            ("raising", "VBG", "raise"),
            ("concerns", "NNS", "concern"),
            (".", ".", "."),
        ]);
        assert!(!is_participial_closer(&tokens, &source));
    }

    /// `participial_closer_rate` is the fraction of matching sentences,
    /// wired through the public API with a stub tagger.
    #[test]
    fn participial_closer_rate_computes_fraction_of_sentences() {
        let (closer_text, closer_tokens) = build_tokens(&[
            ("The", "DT", "the"),
            ("team", "NN", "team"),
            ("shipped", "VBD", "ship"),
            ("it", "PRP", "it"),
            (",", ",", ","),
            ("helping", "VBG", "help"),
            ("users", "NNS", "user"),
            (".", ".", "."),
        ]);
        let (shipped_text, shipped_tokens) = build_tokens(&[
            ("It", "PRP", "it"),
            ("shipped", "VBD", "ship"),
            (".", ".", "."),
        ]);
        let (worked_text, worked_tokens) = build_tokens(&[
            ("It", "PRP", "it"),
            ("worked", "VBD", "work"),
            (".", ".", "."),
        ]);

        let source = format!("{closer_text} {shipped_text} {worked_text}");
        let s1_end = closer_text.len();
        let s2_start = s1_end + 1;
        let s2_end = s2_start + shipped_text.len();
        let s3_start = s2_end + 1;
        let document = document_of(
            &source,
            &[0..s1_end, s2_start..s2_end, s3_start..source.len()],
        );
        let tagger = StubTagger(vec![
            (closer_text.leak() as &str, closer_tokens),
            (shipped_text.leak() as &str, shipped_tokens),
            (worked_text.leak() as &str, worked_tokens),
        ]);

        let rate = participial_closer_rate(&document, &tagger);
        assert!((rate - (1.0 / 3.0)).abs() < f64::EPSILON);
    }

    // -------------------------------------------------------------
    // bullet_parallelism
    // -------------------------------------------------------------

    /// A three-item tight list, every item's stem an imperative verb:
    /// fully parallel, score `1.0`.
    #[test]
    fn bullet_parallelism_scores_one_for_uniform_verb_stems() {
        let source = "- Configure the server\n- Run the migration\n- Restart the service\n";
        let document = friction_parse::parse(source).expect("valid markdown parses");

        let tagger = StubTagger(vec![
            (
                "Configure the server",
                vec![word_token(0, "Configure", "VB", "configure")],
            ),
            ("Run the migration", vec![word_token(0, "Run", "VB", "run")]),
            (
                "Restart the service",
                vec![word_token(0, "Restart", "VB", "restart")],
            ),
        ]);

        assert!((bullet_parallelism(&document, &tagger) - 1.0).abs() < f64::EPSILON);
    }

    /// A three-item list with one noun-stemmed item among two verb-stemmed
    /// ones: the largest class (verb, 2 of 3) sets the score.
    #[test]
    fn bullet_parallelism_scores_fraction_for_mixed_stems() {
        let source = "- Configuration steps\n- Run the migration\n- Restart the service\n";
        let document = friction_parse::parse(source).expect("valid markdown parses");

        let tagger = StubTagger(vec![
            (
                "Configuration steps",
                vec![word_token(0, "Configuration", "NN", "configuration")],
            ),
            ("Run the migration", vec![word_token(0, "Run", "VB", "run")]),
            (
                "Restart the service",
                vec![word_token(0, "Restart", "VB", "restart")],
            ),
        ]);

        let score = bullet_parallelism(&document, &tagger);
        assert!((score - (2.0 / 3.0)).abs() < f64::EPSILON);
    }

    /// A document with no lists at all scores `0.0`.
    #[test]
    fn bullet_parallelism_zero_for_no_lists() {
        let source = "Just a paragraph.\n";
        let document = friction_parse::parse(source).expect("valid markdown parses");
        let tagger = StubTagger(Vec::new());
        assert!(bullet_parallelism(&document, &tagger).abs() < f64::EPSILON);
    }

    /// Two sibling lists (a top-level list and, inside one of its items, a
    /// nested sublist) each contribute their own score to the mean — the
    /// nested sublist's items are not folded into the outer list's tally.
    #[test]
    fn bullet_parallelism_aggregates_nested_lists_separately() {
        let source = "- Configure the server\n  - alpha\n  - beta\n- Restart the service\n";
        let document = friction_parse::parse(source).expect("valid markdown parses");

        // Outer list: both items verb-stemmed -> outer score 1.0.
        // Inner list: both items noun-stemmed -> inner score 1.0.
        // Mean over the two lists: 1.0.
        let tagger = StubTagger(vec![
            (
                "Configure the server",
                vec![word_token(0, "Configure", "VB", "configure")],
            ),
            ("alpha", vec![word_token(0, "alpha", "NN", "alpha")]),
            ("beta", vec![word_token(0, "beta", "NN", "beta")]),
            (
                "Restart the service",
                vec![word_token(0, "Restart", "VB", "restart")],
            ),
        ]);

        assert!((bullet_parallelism(&document, &tagger) - 1.0).abs() < f64::EPSILON);
    }

    /// Builds a single [`TaggedToken`] for `surface` starting at byte
    /// `start`, classified as [`CoreTokenKind::Word`].
    fn word_token(start: usize, surface: &str, pos: &str, lemma: &str) -> TaggedToken {
        TaggedToken {
            token: Token::new(start..start + surface.len(), CoreTokenKind::Word),
            pos: PosTag::new(pos),
            lemma: lemma.into(),
        }
    }

    // -------------------------------------------------------------
    // End-to-end fixture through the real tagger
    // -------------------------------------------------------------

    /// All three metrics computed through the real segmenter and the real
    /// `nlprule`-backed tagger, on one hand-picked document, exercising the
    /// full pipeline rather than a stub. Expected values were confirmed
    /// against this tagger's actual output for this exact text.
    #[test]
    fn real_pipeline_end_to_end_fixture() {
        let source = "\
The kit includes screws, bolts, and washers. \
The team shipped the release, allowing customers to upgrade early.\n\
\n\
- Configure the server\n\
- Restart the service\n";

        let document = friction_parse::parse(source).expect("valid markdown parses");
        let document =
            friction_nlp::segment_document(&document, &friction_nlp::SrxSegmenter::new())
                .expect("segmentation succeeds");
        let tagger = friction_nlp::NlpruleTagger::new().expect("embedded model loads");

        // Four segmented sentences in all: the two paragraph sentences, plus
        // one per tight list item (a list item's text is its own segment
        // even though it carries no sentence-ending punctuation).
        //
        // - Sentence 1 ("The kit includes screws, bolts, and washers.") is
        //   a same-class (noun) triad.
        // - Sentence 2 ("...release, allowing customers to upgrade early.")
        //   is a present-participle closer.
        // - The list items tag as one verb-stemmed and one noun-stemmed
        //   ("Configure" VB, "Restart" NN, per this tagger's actual
        //   disambiguation on these two words in isolation), so the list's
        //   own parallelism score is 1 matching item out of 2.
        assert!((triad_rate(&document, &tagger) - 0.25).abs() < f64::EPSILON);
        assert!((participial_closer_rate(&document, &tagger) - 0.25).abs() < f64::EPSILON);
        assert!((bullet_parallelism(&document, &tagger) - 0.5).abs() < f64::EPSILON);
    }
}
