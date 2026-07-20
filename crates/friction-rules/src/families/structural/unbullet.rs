//! [`UnbulletRule`]: collapses a short, machine-flavored bulleted list into
//! one flowing prose sentence.
//!
//! # What qualifies
//!
//! A markdown list block qualifies only when *every* one of these holds:
//!
//! - **Top-level.** The list itself is not nested inside another list's
//!   item (or any other container) — see [`super::block_parents`]. A
//!   nested (sub-)list is always left untouched, unconditionally.
//! - **2 or 3 items.** [`MIN_ITEMS`]..=[`MAX_ITEMS`], counting only the
//!   list's own direct item children (a nested sublist's items don't
//!   count toward its containing list's count, and disqualify that
//!   containing list on their own — see the next point).
//! - **Each item is a bare fragment.** No block nests inside it at all —
//!   no sublist, no nested paragraph (a *loose* list item wraps its text
//!   in one), no code block. A list built from real fragments (`"- Fast"`,
//!   not `"- Fast\n\n  more detail"`) is the only shape this rule ever
//!   touches.
//! - **Each item's own prose is exactly one run.** More than one
//!   [`friction_core::ProseUnit`] sharing the item's block index means
//!   *something* inside the item's text was excluded from prose
//!   extraction (inline code, a link's URL) — conservatively disqualifies
//!   the whole list rather than risk reconstructing markup this rule was
//!   never designed to preserve.
//! - **Short.** At most [`MAX_ITEM_TOKENS`] word tokens (see
//!   [`super::count_word_tokens`]) and no backtick anywhere in the item's
//!   text (belt-and-suspenders against inline code — see the previous
//!   point for why a code span should already disqualify the item on its
//!   own).
//! - **Machine-flavored stem parallelism.** Every item's leading word
//!   resolves to the same coarse part-of-speech bucket (noun / verb /
//!   adjective / adverb / "no detectable stem at all" — the last is its
//!   own bucket, not folded into any word class, matching
//!   `friction_metrics::bullet_parallelism`'s own design) — the uniform,
//!   templated cadence ("Supports X", "Handles Y", "Provides Z") this rule
//!   exists to collapse.
//!
//! # Why this is Fix tier, not Suggest
//!
//! The fix only ever *joins* the qualifying items into one sentence, in
//! their original source order, using each item's own text verbatim (up to
//! the re-casing described below) — it never reorders, drops, or rewords a
//! single item. That is exactly the "merge adjacent text without
//! reordering" (plus "change punctuation and case") shape this workspace's
//! tier discipline allows as Fix: every proposition the list expressed is
//! still expressed, in the same order, just as one sentence instead of a
//! list.
//!
//! # Re-casing joined items
//!
//! Each bullet item was written to stand alone (`"Fast startup"`,
//! `"Validates input"`), so its own leading word is conventionally
//! capitalized the way any sentence-initial fragment is. Joined into the
//! middle of a larger sentence, that capital reads as a mid-sentence
//! capitalization error unless the word is a genuine proper noun.
//! [`decapitalize_unless_proper_noun`] lowercases an item's leading
//! character unless the tagger resolves it to a proper-noun tag
//! (`NNP`/`NNPS`), and [`decapitalized_items`] applies that to every item
//! before either branch below joins them — this is a case-only change to
//! each item's own first letter, nothing else about the item's text moves
//! or changes, so it stays within the same Fix-tier allowance the rest of
//! this rule already relies on. The joined *sentence's* own leading
//! character (whichever text ends up first in the replacement — an item's
//! own first letter in the standalone case, or the lead-in's own in the
//! colon-lead case) still needs to read as capitalized; each branch below
//! handles that separately, since only one of them has a lead-in sentence
//! to inherit the capital from.
//!
//! # Deriving the lead-in
//!
//! When the list is immediately preceded (see [`super::previous_sibling`])
//! by a paragraph whose *last* sentence ends in a colon — `"It
//! supports:"` — that colon sentence reads as the list's own lead-in, so
//! the fix consumes it too: the patch's range starts at that sentence's
//! own start (not the list's), and the replacement folds it in without the
//! colon: `"It supports:"` + items `["Fast", "Reliable"]` -> `"It supports
//! fast and reliable."` (both items lowercased — see "Re-casing joined
//! items" above — since the lead-in itself already supplies the sentence's
//! capital). Otherwise the fix only replaces the list block's own bytes
//! with a standalone sentence built from the items alone: `"Fast,
//! reliable, and documented."` (only the very first letter capitalized,
//! since this sentence has no lead-in of its own).
//!
//! # Conjunctive vs. disjunctive joining
//!
//! A flat `"and"` join asserts the reader should do (or the document
//! guarantees) *every* item — correct for a conjunctive breakdown
//! (`"Supports X"` / `"Handles Y"`), but a silent reversal of a
//! *disjunctive* lead-in (`"try one of the following:"` + items
//! `["Disable logging", "Disable caching"]` joined with `"and"` now reads
//! as an instruction to do both, not choose one). [`is_disjunctive_lead`]
//! checks the colon lead-in sentence itself for a small, fixed set of
//! disjunctive cue phrases (`"one of the following"`, `"either"`, ...);
//! when one is present, [`UnbulletRule::fix`] joins with `"or"` instead of
//! `"and"` — the same conservative, lexically-grounded signal
//! [`RitualConclusionRule`](crate::families::symmetry::RitualConclusionRule)'s
//! own Fix/Suggest split is built on, except here the signal is precise
//! enough to *correct* the join rather than merely gate it: the lead-in
//! sentence is already being extracted and folded into the replacement
//! verbatim, so checking its own wording for the cue that decides which
//! conjunction is correct costs nothing extra and drifts from `colon_lead`
//! only in how its result is used, never in what it matches. A standalone
//! list with no colon lead-in at all has no such signal to check and keeps
//! the conjunctive `"and"` default, exactly as before.

use std::ops::Range;

use friction_core::{
    Block, BlockKind, Document, Finding, MetricVector, Patch, RuleId, Tier, TokenKind,
};
use friction_nlp::Tagger;

use super::{block_parents, count_word_tokens, previous_sibling};
use crate::budget::Budget;
use crate::context::{GenreEnvelope, RuleContext};
use crate::rule::{Gate, Rule, RuleFamily};
use crate::strategy::StrategyRng;

/// This rule's stable identifier.
const RULE_ID: RuleId = RuleId::new("structural.unbullet");

/// The [`MetricVector`] field this rule gates on.
const GATED_METRIC: &str = "list_item_density";

/// See [`crate::families::connective`]'s `PER_FIX_EFFECT` for the exact
/// same reasoning: `gate` sees only the round's already-normalized
/// per-1000-token density, never the document's real token count, so it
/// cannot compute this rule's true per-fix effect on that density. `1.0`
/// is the natural, dimensionless stand-in — exact for a document near
/// 1000 tokens, conservative (never an overshoot risk) for a longer one.
const PER_FIX_EFFECT: f64 = 1.0;

/// A list qualifies only with exactly 2 or 3 direct items — see the module
/// docs' "What qualifies" section.
const MIN_ITEMS: usize = 2;
/// See [`MIN_ITEMS`].
const MAX_ITEMS: usize = 3;

/// The maximum word-token count (see [`super::count_word_tokens`]) a
/// single item's own text may have and still count as a "SHORT" fragment.
const MAX_ITEM_TOKENS: usize = 10;

/// A coarse grammatical bucket for a list item's leading word, used only to
/// compare items' stems against each other — see
/// `friction_metrics::symmetry`'s own `BroadClass` (private to that crate,
/// hence this small local copy) for the same folding rule and rationale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BroadClass {
    Noun,
    Verb,
    Adjective,
    Adverb,
    Other,
}

/// Folds a Penn-Treebank-style tag prefix into its [`BroadClass`].
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

/// `item`'s stem bucket: the [`BroadClass`] of its first
/// [`TokenKind::Word`] token, or `None` if it has none — its own bucket,
/// not folded into any word class (see the module docs).
fn item_stem_bucket(item: &str, tagger: &dyn Tagger) -> Option<BroadClass> {
    tagger
        .tag(item, 0)
        .into_iter()
        .find(|tagged| tagged.token.kind == TokenKind::Word)
        .map(|tagged| broad_class(tagged.pos.as_str()))
}

/// `true` if every item in `items` shares the same stem bucket (see
/// [`item_stem_bucket`]) — the "machine-flavored stem parallelism" gate.
/// Vacuously `true` for an empty slice (never reached in practice: callers
/// always check the 2-3 item count first).
fn items_share_stem_bucket(items: &[String], tagger: &dyn Tagger) -> bool {
    let mut buckets = items.iter().map(|item| item_stem_bucket(item, tagger));
    let Some(first) = buckets.next() else {
        return true;
    };
    buckets.all(|bucket| bucket == first)
}

/// `true` if the list at `list_index` sits directly at the document's
/// top level — i.e. is not nested inside any other block at all. See the
/// module docs' "Top-level" bullet for why this rule never touches a
/// nested list.
fn is_top_level_list(list_index: usize, parents: &[Option<usize>]) -> bool {
    parents[list_index].is_none()
}

/// The direct child [`BlockKind::ListItem`] indices of the list at
/// `list_index`, in source order.
fn list_item_indices(list_index: usize, blocks: &[Block], parents: &[Option<usize>]) -> Vec<usize> {
    blocks
        .iter()
        .enumerate()
        .filter(|&(i, block)| block.kind == BlockKind::ListItem && parents[i] == Some(list_index))
        .map(|(i, _)| i)
        .collect()
}

/// `true` if any block in `blocks` is a direct child of `item_index` — a
/// nested sublist, a loose item's wrapping paragraph, a nested code block,
/// anything at all. See the module docs' "bare fragment" bullet.
fn item_has_child_blocks(item_index: usize, parents: &[Option<usize>]) -> bool {
    parents.contains(&Some(item_index))
}

/// The item's own directly-owned prose text, or `None` if it has zero or
/// more than one [`friction_core::ProseUnit`] run — see the module docs'
/// "exactly one run" bullet.
fn item_prose_text(item_index: usize, document: &Document) -> Option<&str> {
    let mut runs = document
        .prose()
        .iter()
        .filter(|unit| unit.block == item_index);
    let only = runs.next()?;
    if runs.next().is_some() {
        return None;
    }
    document.text(&only.range).ok()
}

/// Runs every "What qualifies" check from the module docs against the list
/// at `list_index`, returning its items' cleaned text (trailing period
/// stripped, in source order) if the whole list qualifies, `None`
/// otherwise.
///
/// A pure function of `(list_index, document, blocks, parents, tagger)` —
/// called identically by both `scan` (to decide whether to emit a
/// [`Finding`]) and `fix` (to reconstruct the same items for the patch),
/// so the two can never drift apart.
fn candidate_list(
    list_index: usize,
    document: &Document,
    blocks: &[Block],
    parents: &[Option<usize>],
    tagger: &dyn Tagger,
) -> Option<Vec<String>> {
    if !is_top_level_list(list_index, parents) {
        return None;
    }
    let item_indices = list_item_indices(list_index, blocks, parents);
    if !(MIN_ITEMS..=MAX_ITEMS).contains(&item_indices.len()) {
        return None;
    }

    let mut items = Vec::with_capacity(item_indices.len());
    for item_index in item_indices {
        if item_has_child_blocks(item_index, parents) {
            return None;
        }
        let text = item_prose_text(item_index, document)?;
        let trimmed = text.trim();
        if trimmed.is_empty() || trimmed.contains('`') {
            return None;
        }
        if count_word_tokens(trimmed) > MAX_ITEM_TOKENS {
            return None;
        }
        let cleaned = trimmed.strip_suffix('.').unwrap_or(trimmed).trim_end();
        if cleaned.is_empty() {
            return None;
        }
        items.push(cleaned.to_string());
    }

    if !items_share_stem_bucket(&items, tagger) {
        return None;
    }
    Some(items)
}

/// Joins 2 or 3 items into one comma-and-`conjunction`-joined phrase
/// (`"and"` or `"or"` — see the module docs' "Conjunctive vs. disjunctive
/// joining" section), each item used verbatim. Panics only if
/// `items.len()` is outside `2..=3` — never reachable, since
/// [`candidate_list`] is the only producer of `items` and already enforces
/// that count.
fn join_items(items: &[String], conjunction: &str) -> String {
    match items {
        [a, b] => format!("{a} {conjunction} {b}"),
        [a, b, c] => format!("{a}, {b}, {conjunction} {c}"),
        _ => unreachable!("candidate_list only ever returns 2 or 3 items"),
    }
}

/// Uppercases `s`'s first character, leaving the rest untouched.
fn capitalize_first(s: &str) -> String {
    let mut chars = s.chars();
    chars.next().map_or_else(String::new, |first| {
        first.to_uppercase().collect::<String>() + chars.as_str()
    })
}

/// Lowercases `item`'s first character, unless `tagger` tags its leading
/// word as a proper noun (`NNP`/`NNPS`) — the only case a bullet item's own
/// capital letter is semantically load-bearing rather than an artifact of
/// sitting at the start of its own standalone fragment. Every bullet item
/// this rule collapses was written to read as a sentence-initial fragment
/// on its own (`"Fast startup"`, `"Validates input"`); once joined into the
/// middle of a larger sentence, only a genuine proper noun should still
/// read as capitalized — everything else is now mid-sentence prose and
/// [`sentence_from_items`]/[`UnbulletRule::fix`]'s colon-lead branch handle
/// making the *sentence's own* leading character uppercase separately (see
/// their own docs), so this only ever needs to decide whether to lower an
/// item's own leading letter, never whether to raise one.
///
/// A capitalized common word tagged in isolation (no surrounding sentence
/// context) sometimes reads as a proper noun to this tagger even when it
/// plainly is not one — the same context-free tagging quirk
/// [`RitualConclusionRule`](crate::families::symmetry::RitualConclusionRule)'s
/// own module docs call out. That failure mode only ever *keeps* a
/// capital this function could safely have lowered, never the reverse
/// (lowering a genuine proper noun), so it is the same one-directional-safe
/// bias this workspace's other conservative heuristics already take —
/// erring toward "leave a capital letter alone" is a cosmetic miss, never a
/// correctness bug.
fn decapitalize_unless_proper_noun(item: &str, tagger: &dyn Tagger) -> String {
    let is_proper_noun = tagger
        .tag(item, 0)
        .into_iter()
        .find(|tagged| tagged.token.kind == TokenKind::Word)
        .is_some_and(|tagged| tagged.pos.as_str().starts_with("NNP"));
    if is_proper_noun {
        return item.to_string();
    }
    let mut chars = item.chars();
    chars.next().map_or_else(String::new, |first| {
        first.to_lowercase().collect::<String>() + chars.as_str()
    })
}

/// [`decapitalize_unless_proper_noun`], applied to every item in `items` —
/// the shared step [`UnbulletRule::fix`]'s two branches (standalone
/// sentence, colon lead-in) both need before joining, so every item reads
/// as ordinary mid-sentence prose rather than keeping its standalone bullet
/// fragment's own capitalization (see the module docs' "Re-casing joined
/// items" section).
fn decapitalized_items(items: &[String], tagger: &dyn Tagger) -> Vec<String> {
    items
        .iter()
        .map(|item| decapitalize_unless_proper_noun(item, tagger))
        .collect()
}

/// The standalone-sentence fix: `items` (already [`decapitalized_items`])
/// joined conjunctively (`"and"`), capitalized and terminated — used when
/// the list has no colon lead-in to fold into (see [`colon_lead`]), and so
/// has no textual signal [`is_disjunctive_lead`] could check.
/// [`capitalize_first`] forces the sentence's own first character (item
/// one's own leading letter) uppercase regardless of what
/// [`decapitalize_unless_proper_noun`] did to it, since this sentence has
/// no lead-in of its own to inherit sentence-initial case from.
fn sentence_from_items(items: &[String]) -> String {
    format!("{}.", capitalize_first(&join_items(items, "and")))
}

/// Fixed, lowercase cue phrases that mark a colon lead-in sentence as
/// introducing a *disjunctive* ("pick one") list rather than a conjunctive
/// ("all of these") one — see the module docs' "Conjunctive vs.
/// disjunctive joining" section. Deliberately small and literal (no
/// attempt at broader paraphrase detection) — the same "only checkable
/// signals, not a claim of semantic completeness" posture this workspace's
/// other lexically-grounded heuristics take.
const DISJUNCTIVE_LEAD_CUES: &[&str] = &[
    "one of the following",
    "any of the following",
    "one of these",
    "any of these",
    "either",
];

/// `true` if `lead_text` (a colon lead-in sentence, colon already stripped
/// — see [`colon_lead`]) contains one of [`DISJUNCTIVE_LEAD_CUES`],
/// case-insensitively.
fn is_disjunctive_lead(lead_text: &str) -> bool {
    let lower = lead_text.to_ascii_lowercase();
    DISJUNCTIVE_LEAD_CUES.iter().any(|cue| lower.contains(cue))
}

/// If the list at `list_index` is immediately preceded (see
/// [`super::previous_sibling`]) by a paragraph whose last sentence ends in
/// a colon, returns that sentence's own start offset (in `document`'s
/// original source) and its text with the trailing colon stripped. `None`
/// otherwise — see the module docs' "Deriving the lead-in" section.
fn colon_lead(
    list_index: usize,
    document: &Document,
    blocks: &[Block],
    parents: &[Option<usize>],
) -> Option<(usize, String)> {
    let prev_index = previous_sibling(list_index, parents)?;
    if blocks[prev_index].kind != BlockKind::Paragraph {
        return None;
    }
    let last_unit = document
        .prose()
        .iter()
        .rfind(|unit| unit.block == prev_index)?;
    let last_sentence = last_unit.sentences.last()?;
    let text = document.text(&last_sentence.range).ok()?;
    let lead = text.trim_end().strip_suffix(':')?.trim_end();
    if lead.is_empty() {
        return None;
    }
    Some((last_sentence.range.start, lead.to_string()))
}

/// Locates the exact [`BlockKind::List`] block `finding` names: the one
/// whose own range equals `finding.range` — always exactly one, since
/// [`UnbulletRule::scan`] sets every finding's range to a list block's
/// range verbatim, and two distinct list blocks in one document can never
/// share a byte range.
fn list_block_index(finding_range: &Range<usize>, blocks: &[Block]) -> Option<usize> {
    blocks.iter().position(|block| {
        matches!(block.kind, BlockKind::List { .. }) && block.range == *finding_range
    })
}

/// Collapses a short, machine-flavored bulleted list into one prose
/// sentence.
///
/// Budgeted to bring [`GATED_METRIC`] back into the genre's envelope. See
/// the module docs for exactly which lists qualify and how the
/// replacement sentence is built.
#[derive(Debug, Clone, Copy, Default)]
pub struct UnbulletRule;

impl UnbulletRule {
    /// Creates the rule.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Rule for UnbulletRule {
    fn id(&self) -> RuleId {
        RULE_ID
    }

    fn family(&self) -> RuleFamily {
        RuleFamily::Structural
    }

    fn gate(&self, metrics: &MetricVector, envelope: &dyn GenreEnvelope) -> Gate {
        let Some(band) = envelope.band(GATED_METRIC) else {
            return Gate::Off;
        };
        let current = metrics.list_item_density;
        // Only the "too many list items" direction is this rule's to fix
        // — it only ever removes a list, never adds one, so a document
        // already inside the band, or below its floor, gates Off either
        // way.
        if current <= band.hi {
            return Gate::Off;
        }
        let budget = Budget::from_envelope_excess(current, band, PER_FIX_EFFECT);
        if budget.is_exhausted() {
            Gate::Off
        } else {
            Gate::Fix { budget }
        }
    }

    fn scan(&self, ctx: &RuleContext<'_>) -> Vec<Finding> {
        let document = ctx.document();
        let blocks = document.blocks();
        let parents = block_parents(blocks);
        let mut findings = Vec::new();
        for (index, block) in blocks.iter().enumerate() {
            if !matches!(block.kind, BlockKind::List { .. }) {
                continue;
            }
            if candidate_list(index, document, blocks, &parents, ctx.tagger()).is_some() {
                findings.push(Finding::new(
                    RULE_ID,
                    block.range.clone(),
                    "short bulleted list reads as a machine-flavored parallel breakdown; \
                     could join into one sentence",
                    Tier::Fix,
                ));
            }
        }
        findings
    }

    fn fix(
        &self,
        finding: &Finding,
        ctx: &RuleContext<'_>,
        _strategy_rng: &mut StrategyRng,
    ) -> Option<Patch> {
        let document = ctx.document();
        let blocks = document.blocks();
        let parents = block_parents(blocks);
        let list_index = list_block_index(&finding.range, blocks)?;
        let items = candidate_list(list_index, document, blocks, &parents, ctx.tagger())?;
        let items = decapitalized_items(&items, ctx.tagger());
        let list_range = blocks[list_index].range.clone();
        // A markdown list block's own range always ends in at least one
        // trailing newline (see the module docs' fixture notes); leaving
        // those bytes untouched, rather than folding them into the
        // replaced range, keeps this rule from silently eating the
        // document's own trailing-newline convention (or a blank line
        // before whatever follows).
        let end = list_range.end - trailing_newline_len(document.text(&list_range).ok()?);

        if let Some((lead_start, lead_text)) = colon_lead(list_index, document, blocks, &parents) {
            let conjunction = if is_disjunctive_lead(&lead_text) {
                "or"
            } else {
                "and"
            };
            // No `capitalize_first` here: every item (including the first)
            // is now mid-sentence prose continuing `lead_text`, which
            // already supplies its own sentence-initial capital — see the
            // module docs' "Re-casing joined items" section.
            let replacement = format!("{lead_text} {}.", join_items(&items, conjunction));
            Some(Patch::new(lead_start..end, replacement, RULE_ID, Tier::Fix))
        } else {
            Some(Patch::new(
                list_range.start..end,
                sentence_from_items(&items),
                RULE_ID,
                Tier::Fix,
            ))
        }
    }
}

/// The number of trailing `\n`/`\r` bytes at the end of `text`.
fn trailing_newline_len(text: &str) -> usize {
    text.len() - text.trim_end_matches(['\n', '\r']).len()
}

#[cfg(test)]
mod tests {
    use friction_core::{Envelope, TokenKind as CoreTokenKind};
    use friction_nlp::{PosTag, SrxSegmenter, TaggedToken};

    use super::*;
    use crate::context::MapEnvelope;

    /// A stub tagger that tags nothing — used for tests where stem
    /// parallelism is irrelevant (every item lands in the shared `None`
    /// bucket, which counts as parallel — see the module docs).
    struct NoopTagger;
    impl Tagger for NoopTagger {
        fn tag(&self, _text: &str, _base_offset: usize) -> Vec<TaggedToken> {
            Vec::new()
        }
    }

    /// A tagger that classifies each whitespace-delimited word by a
    /// deliberately simple suffix heuristic (`-s`/`-ing` -> a verb tag,
    /// everything else -> a noun tag), only accurate enough to give this
    /// module's tests two distinguishable stem buckets on demand — not a
    /// real part-of-speech tagger.
    struct SuffixTagger;
    impl Tagger for SuffixTagger {
        fn tag(&self, text: &str, base_offset: usize) -> Vec<TaggedToken> {
            let mut tokens = Vec::new();
            let mut cursor = 0usize;
            for word in text.split_whitespace() {
                let start = text[cursor..].find(word).expect("word found in text") + cursor;
                let end = start + word.len();
                cursor = end;
                let lower = word.to_ascii_lowercase();
                let pos =
                    if lower.ends_with("ing") || (lower.ends_with('s') && !lower.ends_with("ss")) {
                        "VBZ"
                    } else {
                        "NN"
                    };
                tokens.push(TaggedToken {
                    token: friction_core::Token::new(
                        (base_offset + start)..(base_offset + end),
                        CoreTokenKind::Word,
                    ),
                    pos: PosTag::new(pos),
                    lemma: lower.into(),
                });
            }
            tokens
        }
    }

    fn document(source: &str) -> Document {
        let parsed = friction_parse::parse(source).expect("valid markdown parses");
        friction_nlp::segment_document(&parsed, &SrxSegmenter::new())
            .expect("segmentation succeeds")
    }

    fn metrics_with_density(density: f64) -> MetricVector {
        MetricVector {
            list_item_density: density,
            ..MetricVector::default()
        }
    }

    // ---------------------------------------------------------------
    // gate()
    // ---------------------------------------------------------------

    #[test]
    fn gate_is_off_without_a_band() {
        let rule = UnbulletRule::new();
        let envelope = MapEnvelope::new();
        assert_eq!(
            rule.gate(&metrics_with_density(500.0), &envelope),
            Gate::Off
        );
    }

    #[test]
    fn gate_is_off_inside_band() {
        let rule = UnbulletRule::new();
        let envelope = MapEnvelope::new().with(GATED_METRIC, Envelope::new(0.0, 100.0));
        assert_eq!(rule.gate(&metrics_with_density(50.0), &envelope), Gate::Off);
    }

    #[test]
    fn gate_below_band_floor_is_also_off() {
        let rule = UnbulletRule::new();
        let envelope = MapEnvelope::new().with(GATED_METRIC, Envelope::new(10.0, 100.0));
        assert_eq!(rule.gate(&metrics_with_density(1.0), &envelope), Gate::Off);
    }

    /// Hand-computed: current 13.0, hi 10.0, effect 1.0 -> budget 3.
    #[test]
    fn gate_above_band_computes_hand_verified_budget() {
        let rule = UnbulletRule::new();
        let envelope = MapEnvelope::new().with(GATED_METRIC, Envelope::new(0.0, 10.0));
        assert_eq!(
            rule.gate(&metrics_with_density(13.0), &envelope),
            Gate::Fix {
                budget: Budget::new(3)
            }
        );
    }

    // ---------------------------------------------------------------
    // scan(): qualifying and disqualifying shapes
    // ---------------------------------------------------------------

    fn scan_source(source: &str) -> Vec<Finding> {
        let doc = document(source);
        let envelope = MapEnvelope::new();
        let ctx = RuleContext::new(&doc, &NoopTagger, "blog", &envelope);
        UnbulletRule::new().scan(&ctx)
    }

    #[test]
    fn scan_matches_a_qualifying_three_item_list() {
        let findings = scan_source("- Fast\n- Reliable\n- Documented\n");
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].tier, Tier::Fix);
    }

    #[test]
    fn scan_matches_a_qualifying_two_item_list() {
        let findings = scan_source("- Fast\n- Reliable\n");
        assert_eq!(findings.len(), 1);
    }

    #[test]
    fn scan_does_not_match_a_single_item_list() {
        assert!(scan_source("- Fast\n").is_empty());
    }

    #[test]
    fn scan_does_not_match_a_four_item_list() {
        assert!(scan_source("- Fast\n- Reliable\n- Documented\n- Cheap\n").is_empty());
    }

    #[test]
    fn scan_does_not_match_when_an_item_is_too_long() {
        let source = "- This particular item has clearly more than ten separate individual word tokens in it\n- Short\n";
        assert!(scan_source(source).is_empty());
    }

    #[test]
    fn scan_does_not_match_a_loose_list() {
        // Blank line between items makes each item's text live in a
        // nested paragraph -> item_has_child_blocks disqualifies it.
        assert!(scan_source("- Fast\n\n- Reliable\n\n- Documented\n").is_empty());
    }

    #[test]
    fn scan_does_not_match_an_item_containing_inline_code() {
        assert!(scan_source("- Uses `fast` mode\n- Reliable\n").is_empty());
    }

    /// The nested-lists invariant this module's docs and the family's
    /// fixture care both call out: an outer list whose item contains a
    /// sublist is never touched, and the inner sublist itself (nested
    /// inside another list's item) is never touched either.
    #[test]
    fn scan_never_matches_nested_lists() {
        let source = "- outer one\n  - inner a\n  - inner b\n- outer two\n";
        assert!(scan_source(source).is_empty());
    }

    #[test]
    fn scan_finds_multiple_qualifying_lists_in_one_document() {
        let source = "- Fast\n- Reliable\n\nSome prose in between.\n\n- Cheap\n- Simple\n";
        let findings = scan_source(source);
        assert_eq!(findings.len(), 2);
        assert!(findings[0].range.start < findings[1].range.start);
    }

    // ---------------------------------------------------------------
    // Stem parallelism
    // ---------------------------------------------------------------

    #[test]
    fn scan_matches_when_stems_are_parallel_under_a_real_tagger() {
        let doc = document("- Supports exports\n- Handles imports\n");
        let envelope = MapEnvelope::new();
        let ctx = RuleContext::new(&doc, &SuffixTagger, "blog", &envelope);
        assert_eq!(UnbulletRule::new().scan(&ctx).len(), 1);
    }

    #[test]
    fn scan_does_not_match_when_stems_are_not_parallel_under_a_real_tagger() {
        // "Supports" -> verb bucket (SuffixTagger), "Documentation" -> noun
        // bucket: mixed stems, not machine-flavored parallel.
        let doc = document("- Supports exports\n- Documentation\n");
        let envelope = MapEnvelope::new();
        let ctx = RuleContext::new(&doc, &SuffixTagger, "blog", &envelope);
        assert!(UnbulletRule::new().scan(&ctx).is_empty());
    }

    // ---------------------------------------------------------------
    // fix(): standalone sentence, colon lead-in
    // ---------------------------------------------------------------

    fn fix_first(source: &str) -> (String, Patch) {
        let doc = document(source);
        let envelope = MapEnvelope::new();
        let ctx = RuleContext::new(&doc, &NoopTagger, "blog", &envelope);
        let rule = UnbulletRule::new();
        let finding = &rule.scan(&ctx)[0];
        let mut rng = StrategyRng::from_seed(0);
        let patch = rule.fix(finding, &ctx, &mut rng).expect("expected a patch");
        let mut applied = source.to_string();
        applied.replace_range(patch.range.clone(), &patch.replacement);
        (applied, patch)
    }

    #[test]
    fn fix_builds_a_standalone_sentence_without_a_colon_lead() {
        let (applied, patch) = fix_first("- Fast\n- Reliable\n- Documented\n");
        assert_eq!(patch.tier, Tier::Fix);
        // Only the sentence's own leading letter stays capitalized —
        // items 2 and 3 are re-cased to ordinary mid-sentence prose (see
        // the module docs' "Re-casing joined items" section).
        assert_eq!(applied, "Fast, reliable, and documented.\n");
    }

    #[test]
    fn fix_folds_in_a_preceding_colon_lead_sentence() {
        let (applied, _patch) = fix_first("It supports:\n\n- Fast\n- Reliable\n- Documented\n");
        // Every item, including the first, is re-cased here: the lead-in
        // ("It supports") already supplies the sentence's own capital.
        assert_eq!(applied, "It supports fast, reliable, and documented.\n");
    }

    #[test]
    fn fix_only_consumes_the_colon_sentence_not_earlier_sentences() {
        let (applied, _patch) =
            fix_first("Intro info. It supports:\n\n- Fast\n- Reliable\n- Documented\n");
        assert_eq!(
            applied,
            "Intro info. It supports fast, reliable, and documented.\n"
        );
    }

    #[test]
    fn fix_ignores_a_preceding_paragraph_without_a_colon() {
        let (applied, _patch) = fix_first("Some text without a colon.\n\n- Fast\n- Reliable\n");
        assert_eq!(
            applied,
            "Some text without a colon.\n\nFast and reliable.\n"
        );
    }

    /// Regression test for the finding that a disjunctive ("pick one")
    /// lead-in got joined with "and", silently reversing the document's
    /// own instruction into "do both": `is_disjunctive_lead` recognizes the
    /// `"one of the following"` cue and joins with "or" instead.
    #[test]
    fn fix_joins_with_or_when_the_lead_in_is_disjunctive() {
        let (applied, _patch) = fix_first(
            "To resolve this, try one of the following:\n\n- Disable logging\n- Disable caching\n",
        );
        assert_eq!(
            applied,
            "To resolve this, try one of the following disable logging or disable caching.\n"
        );
    }

    /// A three-item disjunctive list also joins with "or", using the
    /// Oxford-comma template.
    #[test]
    fn fix_joins_three_items_with_or_when_the_lead_in_is_disjunctive() {
        let (applied, _patch) = fix_first(
            "Choose either of the following:\n\n- Disable logging\n- Disable caching\n- Disable retries\n",
        );
        assert_eq!(
            applied,
            "Choose either of the following disable logging, disable caching, or disable retries.\n"
        );
    }

    /// A colon lead-in with no disjunctive cue still joins conjunctively,
    /// exactly as before — this rule's default is unchanged when there is
    /// no textual signal to check.
    #[test]
    fn fix_still_joins_with_and_when_the_lead_in_has_no_disjunctive_cue() {
        let (applied, _patch) = fix_first("It supports:\n\n- Fast\n- Reliable\n- Documented\n");
        assert_eq!(applied, "It supports fast, reliable, and documented.\n");
    }

    #[test]
    fn fix_strips_a_trailing_period_from_an_item_before_joining() {
        let (applied, _patch) = fix_first("- Fast.\n- Reliable.\n");
        assert_eq!(applied, "Fast and reliable.\n");
    }

    /// Regression test for the finding that items after the first kept
    /// whatever capitalization they had as standalone bullet text, reading
    /// as broken mid-sentence title case: a real tagger's proper-noun tag
    /// is the only thing that keeps an item's own leading letter
    /// capitalized once joined.
    #[test]
    fn fix_lowercases_non_proper_items_but_keeps_a_proper_noun_capitalized() {
        let doc = document("- Runs on Kubernetes\n- Ships fast\n");
        let envelope = MapEnvelope::new();
        let tagger = friction_nlp::NlpruleTagger::new().expect("embedded model loads");
        let ctx = RuleContext::new(&doc, &tagger, "blog", &envelope);
        let rule = UnbulletRule::new();
        let finding = &rule.scan(&ctx)[0];
        let mut rng = StrategyRng::from_seed(0);
        let patch = rule.fix(finding, &ctx, &mut rng).expect("expected a patch");
        let mut applied = "- Runs on Kubernetes\n- Ships fast\n".to_string();
        applied.replace_range(patch.range, &patch.replacement);
        assert_eq!(applied, "Runs on Kubernetes and ships fast.\n");
    }

    // ---------------------------------------------------------------
    // Idempotence and determinism
    // ---------------------------------------------------------------

    #[test]
    fn fixing_a_document_is_idempotent() {
        let source = "- Fast\n- Reliable\n- Documented\n";
        let (applied, _) = fix_first(source);
        assert!(
            scan_source(&applied).is_empty(),
            "expected no findings left after fixing"
        );
    }

    #[test]
    fn fixing_the_same_source_twice_is_byte_identical() {
        let source = "- Fast\n- Reliable\n- Documented\n";
        let run = || fix_first(source).1.replacement;
        assert_eq!(run(), run());
    }
}
