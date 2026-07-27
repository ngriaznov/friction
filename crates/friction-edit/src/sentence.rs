//! Per-sentence orchestrator for the five operations.
//!
//! Fixed order: ritual, paired substitution, derivational pivot (loop,
//! max 2), gated span deletion, frame-gated `just`-deletion — followed
//! by a final recapitalization pass.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::ops::Range;

use friction_core::{Finding, Patch, RuleId, Tier};
use friction_match::token::{AnalysisTokenKind, tokenize_str};
use friction_nlp::Tagger;
use friction_nlp::lvc::{CandidateOutcome, classify_candidate};
use friction_packs::{AttestationPack, InventoryPack};

use crate::gates::{self, DeletionGateOutcome, capitalize, clause_ok, tag};
use crate::nearnoop::PivotBudget;
use crate::splice::SentenceSplicer;

const RULE_RITUAL: RuleId = RuleId::new("ritual.delete");
const RULE_SUB: RuleId = RuleId::new("sub.apply");
const RULE_PIVOT: RuleId = RuleId::new("pivot.lvc");
const RULE_SPAN: RuleId = RuleId::new("span.delete");
const RULE_FRAME_DEJUST: RuleId = RuleId::new("frame.dejust");
const RULE_RECAPITALIZE: RuleId = RuleId::new("edit.recapitalize");

/// Marker words [`run_frame_dejust`] may delete — a strict subset of
/// [`friction_match::frame::DISMISSIVE_MARKERS`]: `"only"` often carries
/// real quantity meaning ("is it only one replica, or…") and is
/// deliberately excluded here even though the detection channel treats it
/// as a valid frame marker.
const DEJUST_MARKERS: [&str; 3] = ["just", "merely", "simply"];

/// `true` if a replacement at `start` would open a line or a sentence,
/// looking past any inline markup in between.
///
/// A substitution replaces matched text with a fixed lowercase form, so
/// restoring a capital depends on where the match landed. Offset zero
/// alone missed real cases: `"* **Leverage Monitoring Tools
/// Effectively:**"` produced `"* **use Monitoring Tools..."` (markers
/// push the match off zero), and `"**...patterns.** Utilize tools"`
/// produced `"** use tools"` (the period sits behind a closing `**`).
///
/// Fix: walk back over whitespace/markup to the first real character.
/// Line start or terminal punctuation means open (needs a capital); a
/// letter or digit means mid-sentence. Not keyed on the segmenter's
/// boundaries, which miss a bolded list-item lead-in.
fn opens_line_or_sentence(text: &str, start: usize) -> bool {
    let Some(prefix) = text.get(..start) else {
        return false;
    };
    prefix
        .chars()
        .rev()
        .take_while(|c| *c != '\n')
        .find(|c| !c.is_whitespace() && !matches!(c, '*' | '_' | '`' | '#' | '>' | '~'))
        .is_none_or(|c| matches!(c, '.' | '!' | '?' | ':' | ';'))
}

/// The result of running the five-operation pipeline over one sentence.
#[derive(Debug, Default)]
pub struct SentenceOutcome {
    /// Accepted edits, as byte-honest patches against the original
    /// source.
    pub patches: Vec<Patch>,
    /// Gate-held candidates, Suggest-tier diagnostics only.
    pub held: Vec<Finding>,
    /// `true` if ritual deletion removed this sentence whole (skipping
    /// the other three operations).
    pub whole_sentence_deleted: bool,
}

/// A sentence's position within its prose unit.
///
/// Its own byte range, plus enough of its neighbours' ranges for a
/// whole-sentence ritual deletion's separator consumption and
/// discourse-binding check. Bundled to keep [`edit_sentence`]'s
/// parameter count reasonable.
#[derive(Debug, Clone)]
pub struct SentencePosition {
    /// This sentence's own byte range into `source`.
    pub range: Range<usize>,
    /// The previous sentence's end offset within the same prose unit,
    /// if any.
    pub prev_end: Option<usize>,
    /// The next sentence's own range within the same prose unit, if
    /// any.
    pub next_range: Option<Range<usize>>,
    /// `false` if this sentence is really the leading fragment of a
    /// prose unit split off its predecessor by an excluded construct
    /// (inline code, a link/image, raw HTML, a footnote reference, a
    /// task-list marker, or a block-quote continuation) with no
    /// sentence-terminal punctuation between them — the segmenter can't
    /// tell it's still mid-sentence, and capitalizing it would rewrite
    /// real prose. Always `true` past a unit's first sentence.
    pub is_sentence_start: bool,
    /// `true` if this sentence is an ordered list item's entire prose
    /// content — deleting it whole would break the item's enumeration,
    /// so ritual deletion is held instead.
    pub sole_content_of_ordered_list_item: bool,
}

/// The packs and tools every stage of [`edit_sentence`] shares —
/// bundled to keep its parameter count reasonable.
pub struct EditContext<'a> {
    pub inventory: &'a InventoryPack,
    pub attestation: &'a AttestationPack,
    pub tagger: &'a dyn Tagger,
    /// Document-local casing evidence for this pass's untouched source
    /// text (see [`DocumentCasing`]).
    pub casing: &'a DocumentCasing,
}

/// Document-local evidence that a word's lowercase spelling is the
/// author's deliberate choice, not a missing capital — collected from
/// every in-scope sentence's untouched text before any edit.
///
/// The recapitalization guard consults it for a sentence whose own
/// original lowercase opener survived editing: recurring lowercase
/// mid-sentence, or opening ≥2 sentences lowercase, marks it deliberate
/// (e.g. `mimalloc`) and holds the opener. A deletion-exposed opener
/// skips this check — every ordinary word occurs lowercase somewhere,
/// so it carries no authorial intent.
#[derive(Debug, Default)]
pub struct DocumentCasing {
    /// Words seen with a lowercase first letter anywhere but a genuine
    /// sentence start (including a continuation fragment's every word).
    mid_sentence_lowercase: BTreeSet<Box<str>>,
    /// How many genuine sentence starts each lowercase-initial word
    /// opens.
    sentence_initial_lowercase: BTreeMap<Box<str>, usize>,
}

impl DocumentCasing {
    /// Records one sentence's word tokens. `is_sentence_start` mirrors
    /// [`SentencePosition`]'s flag: `false` marks a continuation
    /// fragment whose every word, including the first, is really
    /// mid-sentence.
    pub fn record_sentence(&mut self, text: &str, is_sentence_start: bool) {
        let mut at_sentence_start = is_sentence_start;
        for token in tokenize_str(text, 0) {
            if token.kind != AnalysisTokenKind::Word {
                continue;
            }
            if token.text.chars().next().is_some_and(char::is_lowercase) {
                if at_sentence_start {
                    *self
                        .sentence_initial_lowercase
                        .entry(token.text.clone())
                        .or_insert(0) += 1;
                } else {
                    self.mid_sentence_lowercase.insert(token.text.clone());
                }
            }
            at_sentence_start = false;
        }
    }

    /// `true` if the document treats `word`'s lowercase spelling as
    /// deliberate: it recurs lowercase mid-sentence, opens ≥1 *other*
    /// sentence lowercase, or looks like an identifier (digit, `_`,
    /// `-`).
    fn deliberately_lowercase(&self, word: &str) -> bool {
        word.chars()
            .any(|c| c.is_ascii_digit() || c == '_' || c == '-')
            || self.mid_sentence_lowercase.contains(word)
            || self
                .sentence_initial_lowercase
                .get(word)
                .is_some_and(|&count| count >= 2)
    }
}

/// Per-sentence state from the untouched original text: the
/// clause-completeness baseline gates check against, and finite-verb
/// words (paired-substitution's tagger-weakness fallback; see
/// [`gates::original_finite_verb_words`]).
///
/// A pure function of `(tagger, original_text)` — nothing else
/// [`edit_sentence`] threads in (`ctx.inventory`/`ctx.attestation`,
/// `ctx.casing`, `pivot_budget`) feeds into computing it. That purity is
/// exactly what makes [`OriginalStateCache`] safe: two calls with the
/// same tagger and byte-identical `original_text` always recompute the
/// identical value, whatever pass or sentence position either call came
/// from.
#[derive(Clone)]
pub(crate) struct OriginalState {
    clause_ok: bool,
    verb_words: BTreeSet<Box<str>>,
}

/// Caches [`OriginalState`] by a sentence's exact original text, reused
/// across [`crate::document::edit_document`]'s bounded passes within one
/// call so pass 2 doesn't retag a sentence pass 1 already tagged
/// byte-identically.
///
/// Deliberately narrow: this caches only the untouched-original-sentence
/// tag/clause/verb-word computation above, never a sentence's final
/// patches. The five operations' actual decisions also depend on
/// `pivot_budget`, shared and mutated left-to-right across every
/// sentence in a pass (see [`crate::nearnoop::PivotBudget`]'s own docs)
/// — a sentence that produced zero patches in pass 1 can still see a
/// different `pivot_budget` value reach it in pass 2, if earlier
/// sentences behaved differently there, so skipping a sentence's full
/// pipeline based on its pass-1 outcome would risk silently reusing a
/// stale decision. Caching only this pure, budget-independent piece
/// keeps the two-pass loop exactly as behaviorally neutral as before,
/// just without the redundant retag.
///
/// Built once per [`crate::document::edit_document`] call, never reused
/// across documents or invocations — an [`edit_sentence`] caller owns the
/// instance and passes it in by `&mut`.
pub(crate) type OriginalStateCache = HashMap<Box<str>, OriginalState>;

/// `original_text`'s [`OriginalState`], from `cache` if a prior pass
/// already computed it for this exact sentence text, freshly tagged and
/// inserted otherwise.
fn original_state(
    original_text: &str,
    tagger: &dyn Tagger,
    cache: &mut OriginalStateCache,
) -> OriginalState {
    if let Some(cached) = cache.get(original_text) {
        return cached.clone();
    }
    let original_tags = tag(tagger, original_text);
    let computed = OriginalState {
        clause_ok: clause_ok(&original_tags, original_text),
        verb_words: gates::original_finite_verb_words(&original_tags, original_text),
    };
    cache.insert(Box::from(original_text), computed.clone());
    computed
}

/// Runs the five-operation pipeline over one sentence.
///
/// `pub(crate)`: only `crate::document::edit_document` calls this —
/// tightened from `pub` when [`OriginalStateCache`] (`pub(crate)`, since
/// [`OriginalState`] carries no reason to be part of this crate's public
/// surface) joined its parameter list.
#[must_use]
pub(crate) fn edit_sentence(
    source: &str,
    position: &SentencePosition,
    ctx: &EditContext<'_>,
    pivot_budget: &mut PivotBudget,
    original_cache: &mut OriginalStateCache,
) -> SentenceOutcome {
    let sentence_range = position.range.clone();
    let original_text = &source[sentence_range.clone()];
    if original_text.trim().is_empty() {
        return SentenceOutcome::default();
    }
    let original = original_state(original_text, ctx.tagger, original_cache);

    let mut held: Vec<Finding> = Vec::new();
    if let Some(outcome) = try_ritual_deletion(source, position, ctx, &mut held) {
        return outcome;
    }

    let mut splicer = SentenceSplicer::new(source, sentence_range.clone());

    run_substitution(&mut splicer, &mut held, &sentence_range, ctx, &original);
    run_pivot(&mut splicer, &mut held, &sentence_range, ctx, pivot_budget);
    run_deletion(&mut splicer, &mut held, &sentence_range, ctx, &original);
    run_frame_dejust(&mut splicer, &mut held, &sentence_range, pivot_budget);
    // Only a genuine sentence start gets its leading letter capitalized
    // — see `SentencePosition::is_sentence_start`.
    if position.is_sentence_start {
        recapitalize(&mut splicer, ctx.casing);
    }

    SentenceOutcome {
        patches: splicer.finish(),
        held,
        whole_sentence_deleted: false,
    }
}

/// Operation 1 (ritual deletion). Returns `Some` (whole-sentence
/// deletion) if a ritual frame matched with no hold, `None` otherwise —
/// pushing a Suggest-tier hold when a frame matched but couldn't be
/// acted on.
///
/// Unconditional on an actionable match, mirroring `fix_sentence`'s
/// ritual step: no discourse-binding check on the following sentence. (A
/// block-level ritual — a whole preview paragraph gated on
/// adjacent-structure coverage — is separate and unimplemented here, not
/// license to hold an ordinary sentence-level match.) Two holds apply:
/// - a frame matching only inside a double-quoted span is a mention, not
///   a use, of a ritual phrase ([`gates::in_quoted_span`]);
/// - a sentence that is a numbered list item's entire content is never
///   deleted whole — that would leave a bare item marker.
fn try_ritual_deletion(
    source: &str,
    position: &SentencePosition,
    ctx: &EditContext<'_>,
    held: &mut Vec<Finding>,
) -> Option<SentenceOutcome> {
    let sentence_range = position.range.clone();
    let original_text = &source[sentence_range.clone()];

    let mut quoted_frame = None;
    let mut acting_frame = None;
    for frame in ctx.inventory.ritual_frames() {
        for m in frame.pattern.find_iter(original_text) {
            if gates::in_quoted_span(original_text, &m.range()) {
                quoted_frame.get_or_insert(frame);
            } else {
                acting_frame = Some(frame);
                break;
            }
        }
        if acting_frame.is_some() {
            break;
        }
    }
    let frame = match (acting_frame, quoted_frame) {
        (Some(frame), _) => frame,
        (None, Some(frame)) => {
            held.push(Finding::new(
                RULE_RITUAL,
                sentence_range,
                format!("ritual {} held: matched inside quotation", frame.id),
                Tier::Suggest,
            ));
            return None;
        }
        (None, None) => return None,
    };
    if position.sole_content_of_ordered_list_item {
        held.push(Finding::new(
            RULE_RITUAL,
            sentence_range,
            format!(
                "ritual {} held: deleting it would empty a numbered list item",
                frame.id
            ),
            Tier::Suggest,
        ));
        return None;
    }

    let delete_range = match (position.prev_end, position.next_range.as_ref()) {
        (Some(prev_end), _) => prev_end..sentence_range.end,
        (None, Some(next_range)) => sentence_range.start..next_range.start,
        (None, None) => sentence_range,
    };
    Some(SentenceOutcome {
        patches: vec![Patch::new(delete_range, "", RULE_RITUAL, Tier::Fix)],
        held: std::mem::take(held),
        whole_sentence_deleted: true,
    })
}

/// Operation 3 (paired substitution): applies every matching pack
/// pattern, in pack order, gated by
/// [`gates::substitution_clause_gate`].
fn run_substitution(
    splicer: &mut SentenceSplicer<'_>,
    held: &mut Vec<Finding>,
    sentence_range: &Range<usize>,
    ctx: &EditContext<'_>,
    original: &OriginalState,
) {
    for pair in ctx.inventory.substitution_pairs() {
        let working = splicer.working_text();
        if !pair.pattern.is_match(&working) {
            continue;
        }

        let candidate = pair
            .pattern
            .replace_all(&working, pair.replacement.as_ref());
        let candidate_tags = tag(ctx.tagger, &candidate);
        let matched_text = pair.pattern.find(&working).map_or("", |m| m.as_str());
        if !gates::substitution_clause_gate(
            matched_text,
            &candidate,
            &candidate_tags,
            original.clause_ok,
            &original.verb_words,
        ) {
            held.push(Finding::new(
                RULE_SUB,
                sentence_range.clone(),
                format!(
                    "substitution {} held: clause incomplete after edit",
                    pair.id
                ),
                Tier::Suggest,
            ));
            continue;
        }

        let (quoted, ranges): (Vec<Range<usize>>, Vec<Range<usize>>) = pair
            .pattern
            .find_iter(&working)
            .map(|m| m.range())
            .partition(|range| gates::in_quoted_span(&working, range));
        if !quoted.is_empty() {
            held.push(Finding::new(
                RULE_SUB,
                sentence_range.clone(),
                format!("substitution {} held: matched inside quotation", pair.id),
                Tier::Suggest,
            ));
        }
        for range in ranges.into_iter().rev() {
            let mut replacement = pair.replacement.to_string();
            if opens_line_or_sentence(&working, range.start)
                && working[range.clone()]
                    .chars()
                    .next()
                    .is_some_and(char::is_uppercase)
            {
                replacement = capitalize(&replacement);
            }
            splicer.apply(range, &replacement, RULE_SUB, Tier::Fix);
        }
    }
}

/// Operation 4 (derivational pivot, loop, max 2): scans left to right
/// for a licensed light-verb construction, applying up to 2, gated by
/// `pivot_budget`.
fn run_pivot(
    splicer: &mut SentenceSplicer<'_>,
    held: &mut Vec<Finding>,
    sentence_range: &Range<usize>,
    ctx: &EditContext<'_>,
    pivot_budget: &mut PivotBudget,
) {
    let lexicon = ctx.inventory.lvc_lexicon();
    for _ in 0..2 {
        let working = splicer.working_text();
        let tokens = tag(ctx.tagger, &working);

        let mut licensed = None;
        let mut rejected = false;
        for i in 0..tokens.len() {
            match classify_candidate(&tokens, &working, i, lexicon) {
                CandidateOutcome::Rejected(_) => {
                    rejected = true;
                    break;
                }
                CandidateOutcome::NotLightVerb
                | CandidateOutcome::NoNominalFollows
                | CandidateOutcome::Unlicensed => {}
                CandidateOutcome::Licensed(construction) => {
                    if gates::in_quoted_span(&working, &construction.range) {
                        held.push(Finding::new(
                            RULE_PIVOT,
                            sentence_range.clone(),
                            "pivot held: matched inside quotation",
                            Tier::Suggest,
                        ));
                        continue;
                    }
                    licensed = Some(construction);
                    break;
                }
            }
        }
        if rejected {
            break;
        }
        let Some(construction) = licensed else {
            break;
        };

        if !pivot_budget.try_take() {
            held.push(Finding::new(
                RULE_PIVOT,
                sentence_range.clone(),
                "pivot held: near-no-op budget exhausted",
                Tier::Suggest,
            ));
            break;
        }

        let mut verb = construction.derived_verb.to_string();
        if construction.range.start == 0
            && working[..1].chars().next().is_some_and(char::is_uppercase)
        {
            verb = capitalize(&verb);
        }
        splicer.apply(construction.range, &verb, RULE_PIVOT, Tier::Fix);
    }
}

/// Operation 2 (gated span deletion): tries every deletion-span
/// pattern, in pack order, against the post substitution/pivot working
/// text.
fn run_deletion(
    splicer: &mut SentenceSplicer<'_>,
    held: &mut Vec<Finding>,
    sentence_range: &Range<usize>,
    ctx: &EditContext<'_>,
    original: &OriginalState,
) {
    for span in ctx.inventory.deletion_spans() {
        let working = splicer.working_text();
        let Some(m) = span.pattern.find(&working) else {
            continue;
        };
        if gates::in_quoted_span(&working, &m.range()) {
            held.push(Finding::new(
                RULE_SPAN,
                sentence_range.clone(),
                format!("deletion {} held: matched inside quotation", span.id),
                Tier::Suggest,
            ));
            continue;
        }
        let outcome = gates::check_deletion_gates(
            &working,
            m.range(),
            ctx.attestation,
            original.clause_ok,
            ctx.tagger,
        );
        match outcome {
            DeletionGateOutcome::Allowed => {
                splicer.apply(m.range(), "", RULE_SPAN, Tier::Fix);
            }
            DeletionGateOutcome::SeamNotAttested
            | DeletionGateOutcome::SkeletonNotAttested
            | DeletionGateOutcome::ClauseIncomplete => {
                held.push(Finding::new(
                    RULE_SPAN,
                    sentence_range.clone(),
                    format!("deletion {} held: {outcome:?}", span.id),
                    Tier::Suggest,
                ));
            }
        }
    }
}

/// Operation 5 (frame-gated `just`-deletion): inside a detected
/// `frame.contrast.question` span, deletes the licensing marker plus one
/// adjacent space when it is `just`/`merely`/`simply` — never `only` (see
/// [`DEJUST_MARKERS`]). Defangs the rhetorical dismissive framing while
/// preserving the genuine disjunctive question (SYNTHESIS.md §1).
///
/// Detection is [`friction_match::frame::find_contrast_question`], the
/// same function [`friction_match::Channel::Frame`] spans are built from
/// — this operation never re-implements the template. Runs on the working
/// text (post substitution/pivot/deletion), so a match is only found here
/// if the marker survived every earlier operation; after the marker is
/// deleted, the template has no marker left to match, so a later pass
/// over this same sentence never re-fires (idempotent by construction,
/// not by a separate check).
///
/// Shares [`PivotBudget`] with operation 4 — the one near-no-op budget
/// this pipeline has — rather than a second, uncalibrated counter of its
/// own.
fn run_frame_dejust(
    splicer: &mut SentenceSplicer<'_>,
    held: &mut Vec<Finding>,
    sentence_range: &Range<usize>,
    pivot_budget: &mut PivotBudget,
) {
    let working = splicer.working_text();
    let tokens = tokenize_str(&working, 0);
    let Some(question) = friction_match::frame::find_contrast_question(&tokens) else {
        return;
    };
    if !DEJUST_MARKERS.contains(&question.marker.as_ref()) {
        return;
    }
    if gates::in_quoted_span(&working, &question.marker_range) {
        held.push(Finding::new(
            RULE_FRAME_DEJUST,
            sentence_range.clone(),
            "frame.dejust held: matched inside quotation",
            Tier::Suggest,
        ));
        return;
    }
    if !pivot_budget.try_take() {
        held.push(Finding::new(
            RULE_FRAME_DEJUST,
            sentence_range.clone(),
            "frame.dejust held: near-no-op budget exhausted",
            Tier::Suggest,
        ));
        return;
    }

    let delete_range = extend_over_one_adjacent_space(&working, question.marker_range);
    splicer.apply(delete_range, "", RULE_FRAME_DEJUST, Tier::Fix);
}

/// Widens `range` by exactly one adjacent whitespace byte: the trailing
/// space if there is one, else the leading space — the marker always sits
/// strictly between two words in a licensed [`friction_match::frame::ContrastQuestion`]
/// match (between the auxiliary and the coordinating `or`), so exactly
/// one side has a space to consume. Mirrors the literal channel's own
/// `\bsimply\s+` trailing-whitespace convention (`friction_match::literal`),
/// generalized to also cover the marker's non-final position inside the
/// clause.
fn extend_over_one_adjacent_space(text: &str, range: Range<usize>) -> Range<usize> {
    if text[range.end..].starts_with(' ') {
        range.start..(range.end + 1)
    } else if text[..range.start].ends_with(' ') {
        (range.start - 1)..range.end
    } else {
        range
    }
}

/// Final step: if a deletion exposed a lowercase-initial opener, splice a one-char uppercase substitution there.
///
/// Three guards against rewriting prose the engine shouldn't touch:
/// - no accepted edits leaves the sentence as written — a
///   recapitalization-only patch would "fix" the author's own casing;
/// - position 0 is never touched (substitution already capitalizes its
///   own replacements when needed);
/// - a lowercase opener the author wrote at sentence start is held when
///   [`DocumentCasing`] marks it deliberate; one a leading deletion
///   newly exposed carries no such intent and is always capitalized.
fn recapitalize(splicer: &mut SentenceSplicer<'_>, casing: &DocumentCasing) {
    if !splicer.edited() {
        return;
    }
    let final_text = splicer.working_text();
    let Some(first) = final_text.chars().next() else {
        return;
    };
    if !first.is_lowercase() || !splicer.starts_with_original() {
        return;
    }
    if splicer.opening_intact()
        && let Some(opener) = first_word(&final_text)
        && casing.deliberately_lowercase(&opener)
    {
        return;
    }
    let upper = capitalize(&first.to_string());
    splicer.apply(0..first.len_utf8(), &upper, RULE_RECAPITALIZE, Tier::Fix);
}

/// The first word token of `text`, if any.
fn first_word(text: &str) -> Option<Box<str>> {
    tokenize_str(text, 0)
        .into_iter()
        .find(|t| t.kind == AnalysisTokenKind::Word)
        .map(|t| t.text)
}
