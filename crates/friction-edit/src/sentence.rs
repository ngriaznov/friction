//! Per-sentence orchestrator for the six operations.
//!
//! Fixed order: ritual, paired substitution, derivational pivot (loop,
//! max 2), gated span deletion, frame-gated `just`-deletion, frame
//! rewriting: followed by a final recapitalization pass. Frame-rule
//! *guards* constrain the frame channel itself (no frame rewrite may
//! touch a guarded span, and a guarded sentence is never
//! ritual-deleted whole); report-only frame rules emit their findings
//! from the untouched original text. See `run_frame_rewrite`'s docs
//! for why guards deliberately do not veto the construction-level
//! operations.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::ops::Range;

use friction_core::{Finding, Patch, RuleId, Tier};
use friction_match::token::{AnalysisTokenKind, tokenize_str};
use friction_nlp::Tagger;
use friction_nlp::lvc::{CandidateOutcome, classify_candidate};
use friction_packs::{AttestationPack, InventoryPack, RepairKind};
use std::sync::LazyLock;

use friction_match::frame_rewrite::{self, FrameIndex, FrameMatch};
use friction_nlp::TaggedToken;
use friction_packs::frame_bin::{FramePackView, PatOp, TplOp};
use friction_packs::frame_compile::CompiledKind;
use friction_packs::frame_rules::Tag as FrameTag;

use crate::gates::{self, DeletionGateOutcome, capitalize, clause_ok, tag};
use crate::nearnoop::PivotBudget;
use crate::splice::SentenceSplicer;

const RULE_RITUAL: RuleId = RuleId::new("ritual.delete");
const RULE_SUB: RuleId = RuleId::new("sub.apply");
const RULE_PIVOT: RuleId = RuleId::new("pivot.lvc");
const RULE_SPAN: RuleId = RuleId::new("span.delete");
const RULE_FRAME_DEJUST: RuleId = RuleId::new("frame.dejust");
const RULE_RECAPITALIZE: RuleId = RuleId::new("edit.recapitalize");

/// The embedded frame pack's anchor index, built once per process.
static FRAME_INDEX: LazyLock<FrameIndex> =
    LazyLock::new(|| FrameIndex::build(&friction_packs::FRAME.pack));

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

/// The packs and tools every stage of [`edit_sentence`] shares:
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
/// author's deliberate choice, not a missing capital, collected from
/// every in-scope sentence's untouched text before any edit.
///
/// The recapitalization guard consults it for a sentence whose own
/// original lowercase opener survived editing: recurring lowercase
/// mid-sentence, or opening ≥2 sentences lowercase, marks it deliberate
/// (e.g. `mimalloc`) and holds the opener. A deletion-exposed opener
/// skips this check: every ordinary word occurs lowercase somewhere, so
/// it carries no authorial intent.
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
    /// The original sentence's tags, kept for the frame phases: the
    /// guard/report scan always runs on the original text, and the
    /// rewrite scan reuses these when no earlier operation edited.
    tags: Vec<TaggedToken>,
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
    let computed = {
        let original_tags = tag(tagger, original_text);
        OriginalState {
            clause_ok: clause_ok(&original_tags, original_text),
            verb_words: gates::original_finite_verb_words(&original_tags, original_text),
            tags: original_tags,
        }
    };
    cache.insert(Box::from(original_text), computed.clone());
    computed
}

/// [`edit_sentence`]'s cache key: everything its full-pipeline output
/// actually depends on, besides `ctx` (fixed for the whole document) and
/// `pivot_budget`'s exact value (folded into `budget_regime` below).
///
/// `prev_gap`/`next_gap` are the byte GAP to each neighbour, not the
/// neighbour's own absolute position — `try_ritual_deletion` is the only
/// stage reading `SentencePosition::prev_end`/`next_range` at all, and
/// only to size a whole-sentence deletion's separator consumption, which
/// depends on the gap, never on where in the document it happens to sit.
/// A sentence whose own bytes are unchanged between pass 1 and pass 2 but
/// whose neighbour gap shifted (rare — none of the five operations edit
/// inter-sentence whitespace) simply misses the cache instead of reusing
/// a wrong range.
///
/// `budget_regime` is [`PivotBudget::remaining`] clamped to `0..=3`: at
/// most three [`PivotBudget::try_take`] calls ever happen inside one
/// sentence (the pivot loop's own two, plus `frame_dejust`'s one), and
/// once `remaining` covers however many of those this sentence will
/// attempt, every additional unit of headroom is unobservable — a
/// sentence that would attempt 2 takes behaves identically whether
/// `remaining` was 3 or 3,000,000 at entry. Clamping lets a budget-rich
/// document (the common case once near-no-op calibration exists) still
/// hit cache broadly, instead of every distinct `remaining` value forcing
/// a miss.
#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct GenerationKey {
    text: Box<str>,
    is_sentence_start: bool,
    sole_content_of_ordered_list_item: bool,
    prev_gap: Option<usize>,
    next_gap: Option<usize>,
    budget_regime: u32,
}

impl GenerationKey {
    fn for_position(source: &str, position: &SentencePosition, pivot_budget: PivotBudget) -> Self {
        let sentence_range = &position.range;
        Self {
            text: Box::from(&source[sentence_range.clone()]),
            is_sentence_start: position.is_sentence_start,
            sole_content_of_ordered_list_item: position.sole_content_of_ordered_list_item,
            prev_gap: position.prev_end.map(|end| sentence_range.start - end),
            next_gap: position
                .next_range
                .as_ref()
                .map(|next| next.start - sentence_range.end),
            budget_regime: pivot_budget.remaining().min(3),
        }
    }
}

/// One cached [`Patch`], in coordinates relative to its own sentence's
/// start: rebased to the sentence's actual position on every use,
/// including a cache hit from a different pass. Always `0..=len` (never
/// negative, never past the sentence's own end): [`SentenceSplicer`]
/// guarantees every one of these came from a `working_range` inside the
/// single `Chunk::Original` it was seeded with.
pub(crate) struct GeneratedPatch {
    rel_start: usize,
    rel_end: usize,
    replacement: Box<str>,
    rule: RuleId,
    tier: Tier,
}

/// One cached held [`Finding`]. No range: every held finding this module
/// produces carries `sentence_range` verbatim (never a narrower span —
/// true of every `held.push(Finding::new(RULE_*, sentence_range.clone(),
/// ..))` call below), so [`GeneratedOutcome::rebase`] rebuilds it from the
/// CALLER's own current `sentence_range` directly, with nothing to shift.
pub(crate) struct GeneratedHeld {
    rule: RuleId,
    message: Box<str>,
    tier: Tier,
}

/// [`edit_sentence`]'s full-pipeline output, budget-application included,
/// in a form indexed by [`GenerationKey`] and safe to replay against a
/// different sentence position (a cache hit is byte-identical to a fresh
/// [`generate_sentence`] call only because [`GenerationKey`] already pins
/// every input that could change the outcome. See its own docs).
pub(crate) enum GeneratedOutcome {
    /// [`try_ritual_deletion`] fired: the sentence (plus, in general, part
    /// of its surrounding gap) is deleted whole. `rel_start` can be
    /// negative (consuming the gap before the sentence) and `rel_end` can
    /// exceed the sentence's own length (consuming the gap after it) —
    /// see `try_ritual_deletion`'s own `delete_range` computation. Never
    /// touches `pivot_budget` (mirroring that function, which never
    /// calls [`PivotBudget::try_take`]) and never holds anything (see
    /// [`to_generated`]'s own docs).
    RitualDeleted { rel_start: isize, rel_end: isize },
    /// The five-stage pipeline ran to completion, whether or not it
    /// produced any patches. `budget_takes` is exactly how many
    /// [`PivotBudget::try_take`] calls the generating run made succeed —
    /// replaying it against a live budget on a cache hit reproduces the
    /// exact same left-to-right consumption a fresh call would have made,
    /// so a later sentence sharing the same budget sees identical state
    /// either way.
    Pipeline {
        patches: Vec<GeneratedPatch>,
        held: Vec<GeneratedHeld>,
        budget_takes: u32,
    },
}

impl GeneratedOutcome {
    /// Builds the cached form of `outcome`, which [`edit_sentence`] just
    /// produced for a sentence at `sentence_range` after making exactly
    /// `budget_takes` successful [`PivotBudget::try_take`] calls.
    fn from_outcome(
        outcome: &SentenceOutcome,
        sentence_range: &Range<usize>,
        budget_takes: u32,
    ) -> Self {
        if outcome.whole_sentence_deleted {
            // `try_ritual_deletion` returns exactly one patch and an
            // always-empty `held` (nothing is ever pushed to it before
            // the early return. See that function's own docs).
            let patch = &outcome.patches[0];
            #[allow(clippy::cast_possible_wrap)]
            return Self::RitualDeleted {
                rel_start: patch.range.start as isize - sentence_range.start as isize,
                rel_end: patch.range.end as isize - sentence_range.start as isize,
            };
        }
        let patches = outcome
            .patches
            .iter()
            .map(|patch| GeneratedPatch {
                rel_start: patch.range.start - sentence_range.start,
                rel_end: patch.range.end - sentence_range.start,
                replacement: Box::from(patch.replacement.as_str()),
                rule: patch.rule,
                tier: patch.tier,
            })
            .collect();
        let held = outcome
            .held
            .iter()
            .map(|finding| GeneratedHeld {
                rule: finding.rule,
                message: Box::from(finding.message.as_str()),
                tier: finding.tier,
            })
            .collect();
        Self::Pipeline {
            patches,
            held,
            budget_takes,
        }
    }

    /// Rebuilds the [`SentenceOutcome`] this cached entry represents for
    /// a call at `sentence_range`, consuming `pivot_budget` by exactly as
    /// much as the generating call did.
    fn rebase(
        &self,
        sentence_range: &Range<usize>,
        pivot_budget: &mut PivotBudget,
    ) -> SentenceOutcome {
        match self {
            Self::RitualDeleted { rel_start, rel_end } => {
                #[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
                let range = ((sentence_range.start as isize + rel_start) as usize)
                    ..((sentence_range.start as isize + rel_end) as usize);
                SentenceOutcome {
                    patches: vec![Patch::new(range, "", RULE_RITUAL, Tier::Fix)],
                    held: Vec::new(),
                    whole_sentence_deleted: true,
                }
            }
            Self::Pipeline {
                patches,
                held,
                budget_takes,
            } => {
                for _ in 0..*budget_takes {
                    // Always succeeds: `budget_takes` never exceeds the
                    // `budget_regime` the lookup key was built from, and a
                    // cache hit only happens when the live budget's own
                    // clamped `remaining` matches that same regime.
                    pivot_budget.try_take();
                }
                let patches = patches
                    .iter()
                    .map(|patch| {
                        Patch::new(
                            (sentence_range.start + patch.rel_start)
                                ..(sentence_range.start + patch.rel_end),
                            patch.replacement.to_string(),
                            patch.rule,
                            patch.tier,
                        )
                    })
                    .collect();
                let held = held
                    .iter()
                    .map(|finding| {
                        Finding::new(
                            finding.rule,
                            sentence_range.clone(),
                            finding.message.to_string(),
                            finding.tier,
                        )
                    })
                    .collect();
                SentenceOutcome {
                    patches,
                    held,
                    whole_sentence_deleted: false,
                }
            }
        }
    }
}

/// Caches [`GeneratedOutcome`] by [`GenerationKey`], reused across
/// [`crate::document::edit_document`]'s bounded passes within one call —
/// the pass-2 analogue of [`OriginalStateCache`], but covering the whole
/// per-sentence pipeline's output (patches, held findings, and budget
/// consumption) rather than just the untouched-original-text tag/clause
/// computation. Built once per [`crate::document::edit_document`] call,
/// never reused across documents or invocations.
pub(crate) type GenerationCache = HashMap<GenerationKey, GeneratedOutcome>;

/// Runs the five-operation pipeline over one sentence, memoizing
/// everything but recapitalization — patches, held findings, and how much
/// of `pivot_budget` it consumes — by [`GenerationKey`] in
/// `generation_cache`, then applies recapitalization fresh every call.
///
/// Most sentences are byte-identical, in the same position, seeing the
/// same budget headroom, between pass 1 and pass 2 (see
/// [`crate::document::edit_document`]'s own docs on why pass 2 exists at
/// all): [`GenerationKey`] pins every input [`generate_sentence`]'s
/// output actually depends on, so a pass-2 hit replays pass 1's already-
/// computed patches/held findings ([`GeneratedOutcome::rebase`]) instead
/// of re-tagging and re-walking the five stages from scratch. A miss
/// falls through to [`generate_sentence`] exactly as before, then caches
/// its result for a later pass or sentence to reuse.
///
/// Recapitalization is the one stage [`GenerationKey`] deliberately does
/// NOT cover: [`recapitalize`] reads `ctx.casing`
/// ([`EditContext::casing`]), a WHOLE-DOCUMENT summary
/// [`crate::document::edit_document`] rebuilds from scratch every pass —
/// two byte-identical sentences, same flags, same budget regime, can
/// still get different recapitalization outcomes across passes if some
/// OTHER sentence's edits changed what the document treats as a
/// deliberate lowercase opener. So it is never part of the cached value;
/// [`apply_recapitalize`] runs it fresh against every call's own
/// `ctx.casing`, cache hit or miss alike.
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
    generation_cache: &mut GenerationCache,
) -> SentenceOutcome {
    if source[position.range.clone()].trim().is_empty() {
        return SentenceOutcome::default();
    }

    let key = GenerationKey::for_position(source, position, *pivot_budget);
    let pre_recap = if let Some(cached) = generation_cache.get(&key) {
        cached.rebase(&position.range, pivot_budget)
    } else {
        let budget_before = pivot_budget.remaining();
        let outcome = generate_sentence(source, position, ctx, pivot_budget, original_cache);
        let budget_takes = budget_before
            .saturating_sub(pivot_budget.remaining())
            .min(3);
        generation_cache.insert(
            key,
            GeneratedOutcome::from_outcome(&outcome, &position.range, budget_takes),
        );
        outcome
    };

    // Ritual deletion never recapitalizes (see `try_ritual_deletion`'s own
    // early return in `generate_sentence`), and only a genuine sentence
    // start gets its leading letter capitalized — see
    // `SentencePosition::is_sentence_start`.
    if pre_recap.whole_sentence_deleted || !position.is_sentence_start {
        return pre_recap;
    }
    apply_recapitalize(source, &position.range, pre_recap, ctx.casing)
}

/// Applies [`recapitalize`] fresh, against `casing`, on top of
/// `pre_recap`'s already-decided (possibly cached) patches.
///
/// Reconstructs a real [`SentenceSplicer`] by replaying `pre_recap`'s
/// patches — DISJOINT by construction (no operation re-matches text an
/// earlier one introduced), so replaying them in DESCENDING
/// (rightmost-first) original-position order is always valid (each
/// replay target is still a single untouched `Chunk::Original` piece at
/// the moment it's applied) and reconstructs the exact same chunk
/// structure the original sequential run would have, regardless of what
/// order they were really produced in — [`crate::document::resolve`]
/// re-sorts every patch by position across the whole pass anyway, so
/// replay order was never significant to begin with. Runs the real
/// [`recapitalize`] on the reconstructed splicer (not a hand-rederived
/// equivalent), so this can't drift from what a fresh, uncached
/// [`generate_sentence`] call would have decided.
fn apply_recapitalize(
    source: &str,
    sentence_range: &Range<usize>,
    pre_recap: SentenceOutcome,
    casing: &DocumentCasing,
) -> SentenceOutcome {
    let mut splicer = SentenceSplicer::new(source, sentence_range.clone());
    let mut ordered = pre_recap.patches;
    ordered.sort_by_key(|patch| std::cmp::Reverse(patch.range.start));
    for patch in ordered {
        let local =
            (patch.range.start - sentence_range.start)..(patch.range.end - sentence_range.start);
        splicer.apply(local, &patch.replacement, patch.rule, patch.tier);
    }
    recapitalize(&mut splicer, casing);
    SentenceOutcome {
        patches: splicer.finish(),
        held: pre_recap.held,
        whole_sentence_deleted: false,
    }
}

/// The five-operation pipeline's actual computation for one sentence —
/// [`edit_sentence`]'s cache-miss path, unchanged from before
/// [`GenerationCache`] existed.
fn generate_sentence(
    source: &str,
    position: &SentencePosition,
    ctx: &EditContext<'_>,
    pivot_budget: &mut PivotBudget,
    original_cache: &mut OriginalStateCache,
) -> SentenceOutcome {
    let sentence_range = position.range.clone();
    let original_text = &source[sentence_range.clone()];
    let original = original_state(original_text, ctx.tagger, original_cache);

    let mut held: Vec<Finding> = Vec::new();

    // Frame guards and report-only rules scan the untouched original.
    // A guard protects its span from the frame channel's own rewrites
    // (and holds whole-sentence ritual deletion); it deliberately does
    // NOT block the construction-level operations (the derivational
    // pivot above all): a guard's evidence is the bare verb's
    // human-corpus tilt, while a pivot rewrites a light-verb
    // NOMINALIZATION construction whose machine tilt was measured
    // separately — unigram direction must not veto construction-level
    // evidence. A report rule's finding documents the original text,
    // not some half-edited working state.
    let frame_view = &friction_packs::FRAME.pack;
    let original_frame_matches =
        frame_rewrite::scan_sentence(frame_view, &FRAME_INDEX, &original.tags, original_text);
    let has_guarded_span = original_frame_matches
        .iter()
        .any(|m| frame_kind(frame_view, m) == CompiledKind::Guard);
    for report in original_frame_matches
        .iter()
        .filter(|m| frame_kind(frame_view, m) == CompiledKind::Report)
    {
        let rule = frame_view
            .rule(report.rule_index)
            .expect("matched rule in range");
        let message = if rule.m_pm > 0.0 || rule.h_pm > 0.0 {
            format!(
                "frame {}: measured {:.1}/M machine vs {:.1}/M human; report-only",
                rule.id, rule.m_pm, rule.h_pm
            )
        } else {
            // Pilot-bucket demotions carry pair support, not
            // per-million rates — printing 0.0/M would misread as
            // "unobserved".
            format!(
                "frame {}: report-only (see the frame-pack compile report)",
                rule.id
            )
        };
        held.push(Finding::new(
            RuleId::from(rule.id),
            report.bytes.start + sentence_range.start..report.bytes.end + sentence_range.start,
            message,
            Tier::Suggest,
        ));
    }

    if let Some(outcome) = try_ritual_deletion(source, position, ctx, &mut held) {
        // A whole-sentence ritual deletion erases everything in the
        // sentence: a guarded span included, which no evidence
        // licensed. Hold it instead when one exists.
        if !has_guarded_span {
            return outcome;
        }
        held.push(Finding::new(
            RULE_RITUAL,
            sentence_range.clone(),
            "ritual deletion held: sentence contains a frame-guarded span".to_string(),
            Tier::Suggest,
        ));
    }

    let mut splicer = SentenceSplicer::new(source, sentence_range.clone());

    run_substitution(&mut splicer, &mut held, &sentence_range, ctx, &original);
    run_pivot(
        &mut splicer,
        &mut held,
        &sentence_range,
        ctx,
        pivot_budget,
        &original,
    );
    run_deletion(&mut splicer, &mut held, &sentence_range, ctx, &original);
    run_frame_dejust(&mut splicer, &mut held, &sentence_range, pivot_budget);
    run_frame_rewrite(
        &mut splicer,
        &mut held,
        &sentence_range,
        ctx,
        &original,
        &original_frame_matches,
    );
    // Recapitalization is deliberately NOT run here — see
    // `edit_sentence`'s own docs on why it's applied separately, after
    // any [`GenerationCache`] lookup/insert, never as part of the cached
    // value.

    SentenceOutcome {
        patches: splicer.finish(),
        held,
        whole_sentence_deleted: false,
    }
}

/// Operation 1 (ritual deletion). Returns `Some` (whole-sentence
/// deletion) if a ritual frame matched with no hold, `None` otherwise:
/// pushing a Suggest-tier hold when a frame matched but couldn't be
/// acted on.
///
/// Unconditional on an actionable match, mirroring `fix_sentence`'s
/// ritual step: no discourse-binding check on the following sentence. (A
/// block-level ritual, a whole preview paragraph gated on
/// adjacent-structure coverage, is separate and unimplemented here, not
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
        // Same enforcement as the span loop above: a diagnostic-only
        // ritual frame is recorded, never executed. Before this check,
        // the shipped diagnostic-only frame ("fortunate enough to have
        // had the opportunity to") deleted its whole sentence -- real
        // content destroyed by an entry whose own notes forbid exactly
        // that.
        (Some(frame), _) if frame.repair == RepairKind::DiagnosticOnly => {
            held.push(Finding::new(
                RULE_RITUAL,
                sentence_range,
                format!(
                    "ritual {} recorded: diagnostic-only, no safe rewrite",
                    frame.id
                ),
                Tier::Suggest,
            ));
            return None;
        }
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
    let mut working = splicer.working_text();
    for pair in ctx.inventory.substitution_pairs() {
        debug_assert_eq!(
            working,
            splicer.working_text(),
            "hoisted working text out of sync with splicer chain"
        );
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
        if ranges.is_empty() {
            continue;
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
        // At least one non-quoted match just applied (the loop above is
        // only reached with a non-empty `ranges`): the chain changed, so
        // the next pattern in this same pass must see the updated text.
        working = splicer.working_text();
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
    original: &OriginalState,
) {
    let lexicon = ctx.inventory.lvc_lexicon();
    for _ in 0..2 {
        let working = splicer.working_text();
        // An unedited working text is byte-identical to the original the
        // guard phase already tagged — reuse those tags (the same
        // purity argument as `run_frame_rewrite`'s); only an edited
        // sentence (a prior substitution, or this loop's own first
        // pivot) pays for a fresh tag.
        let owned_tokens;
        let tokens: &[TaggedToken] = if splicer.edited() {
            owned_tokens = tag(ctx.tagger, &working);
            &owned_tokens
        } else {
            &original.tags
        };

        let mut licensed = None;
        let mut rejected = false;
        for i in 0..tokens.len() {
            match classify_candidate(tokens, &working, i, lexicon) {
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
    let mut working = splicer.working_text();
    for span in ctx.inventory.deletion_spans() {
        debug_assert_eq!(
            working,
            splicer.working_text(),
            "hoisted working text out of sync with splicer chain"
        );
        let Some(m) = span.pattern.find(&working) else {
            continue;
        };
        // The pack's contract for a diagnostic-only entry ("recorded
        // for honesty but has no safe closed-set rewrite; must never be
        // executed") is enforced here, not merely documented: the span
        // is reported and the gates are never consulted, so no gate
        // outcome can ever turn it into an edit.
        if span.repair == RepairKind::DiagnosticOnly {
            held.push(Finding::new(
                RULE_SPAN,
                sentence_range.clone(),
                format!(
                    "deletion {} recorded: diagnostic-only, no safe rewrite",
                    span.id
                ),
                Tier::Suggest,
            ));
            continue;
        }
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
                // The chain changed: the next span's pattern must match
                // against the post-deletion text.
                working = splicer.working_text();
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
    let token_refs: Vec<&_> = tokens.iter().collect();
    let Some(question) = friction_match::frame::find_contrast_question(&token_refs) else {
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

/// The compiled kind of a frame match's rule.
fn frame_kind(view: &FramePackView<'_>, m: &FrameMatch) -> CompiledKind {
    view.rule(m.rule_index).expect("matched rule in range").kind
}

/// Operation 6 (frame rewriting): applies the compiled frame-rule
/// program, rewrites and deletions with corpus-adjudicated evidence,
/// to the working text, after every other operation has had its turn.
///
/// Each resolved candidate runs the full existing gate stack before it
/// may edit: deletions go through [`gates::check_deletion_gates`]
/// unchanged; rewrites through [`check_rewrite_gates`], the same
/// seam-bigram / clause / skeleton battery phrased for a replacement
/// instead of an excision. A gate failure demotes the candidate to a
/// Suggest-tier finding naming the gate. Candidates are applied
/// rightmost-first so earlier (left) working-text coordinates stay
/// valid as the chunk chain reshapes. The splicer itself declines any
/// candidate overlapping a guard span.
fn run_frame_rewrite(
    splicer: &mut SentenceSplicer<'_>,
    held: &mut Vec<Finding>,
    sentence_range: &Range<usize>,
    ctx: &EditContext<'_>,
    original: &OriginalState,
    original_frame_matches: &[FrameMatch],
) {
    let view = &friction_packs::FRAME.pack;
    let working = splicer.working_text();
    // Tagging and scanning are the expensive parts, and an unedited
    // sentence is byte-identical to the original the guard phase
    // already tagged and scanned. Reuse both. Only an edited sentence
    // pays for a fresh tag and scan.
    let owned_tags;
    let owned_matches;
    let (tags, matches): (&[TaggedToken], &[FrameMatch]) = if splicer.edited() {
        owned_tags = tag(ctx.tagger, &working);
        owned_matches = frame_rewrite::scan_sentence(view, &FRAME_INDEX, &owned_tags, &working);
        (&owned_tags, &owned_matches)
    } else {
        (&original.tags, original_frame_matches)
    };
    let guard_spans: Vec<Range<usize>> = matches
        .iter()
        .filter(|m| frame_kind(view, m) == CompiledKind::Guard)
        .map(|m| m.bytes.clone())
        .collect();
    let edits: Vec<FrameMatch> = matches
        .iter()
        .filter(|m| {
            matches!(
                frame_kind(view, m),
                CompiledKind::Rewrite | CompiledKind::Delete
            ) && !guard_spans
                .iter()
                .any(|g| m.bytes.start < g.end && g.start < m.bytes.end)
        })
        .cloned()
        .collect();
    let mut resolved = frame_rewrite::resolve(view, edits);
    resolved.reverse();
    for m in resolved {
        apply_frame_candidate(
            splicer,
            held,
            sentence_range,
            ctx,
            original,
            &working,
            tags,
            m,
        );
    }
}

/// Gates and applies (or holds) one resolved frame candidate.
#[expect(
    clippy::too_many_arguments,
    reason = "per-candidate slice of run_frame_rewrite"
)]
fn apply_frame_candidate(
    splicer: &mut SentenceSplicer<'_>,
    held: &mut Vec<Finding>,
    sentence_range: &Range<usize>,
    ctx: &EditContext<'_>,
    original: &OriginalState,
    working: &str,
    tags: &[TaggedToken],
    m: FrameMatch,
) {
    let view = &friction_packs::FRAME.pack;
    let rule = view.rule(m.rule_index).expect("matched rule in range");
    let rule_id = RuleId::from(rule.id);
    let finding_range = m.bytes.start + sentence_range.start..m.bytes.end + sentence_range.start;
    if gates::in_quoted_span(working, &m.bytes) {
        held.push(Finding::new(
            rule_id,
            finding_range,
            format!("frame {} held: matched inside quotation", rule.id),
            Tier::Suggest,
        ));
        return;
    }
    match rule.kind {
        CompiledKind::Delete => {
            let outcome = gates::check_deletion_gates(
                working,
                m.bytes.clone(),
                ctx.attestation,
                original.clause_ok,
                ctx.tagger,
            );
            if outcome == DeletionGateOutcome::Allowed {
                let base = m.bytes;
                let widened = widen_deletion(working, base.clone());
                let (range, replacement) = repair_article(working, widened, String::new());
                if splicer.can_apply(&range) {
                    splicer.apply(range, &replacement, rule_id, Tier::Fix);
                } else if splicer.can_apply(&base) {
                    // The widening crossed into an earlier edit's
                    // replacement; the bare deletion is still sound.
                    splicer.apply(base, "", rule_id, Tier::Fix);
                }
            } else {
                held.push(Finding::new(
                    rule_id,
                    finding_range,
                    format!("frame {} held: {outcome:?}", rule.id),
                    Tier::Suggest,
                ));
            }
        }
        CompiledKind::Rewrite => {
            // A bare verb+slot pattern encodes a TRANSITIVE
            // substitution ("utilize X" -> "use X"); when the slot
            // opens with a preposition or particle the sentence is
            // a different construction (literal motion, a phrasal
            // verb) whose direction the corpus never adjudicated
            // ("Navigate to the backups directory" must not become
            // "Handle to ..."). Held, not rejected: the evidence
            // still flags the verb itself.
            if bare_verb_slot_over_preposition(&rule, &m, tags) {
                held.push(Finding::new(
                    rule_id,
                    finding_range,
                    format!(
                        "frame {} held: slot opens with a preposition (untested construction)",
                        rule.id
                    ),
                    Tier::Suggest,
                ));
                return;
            }
            let Some(replacement) = realize_template(view, &rule, &m, tags, working) else {
                held.push(Finding::new(
                    rule_id,
                    finding_range,
                    format!("frame {} held: template did not realize", rule.id),
                    Tier::Suggest,
                ));
                return;
            };
            if let Some(gate) = check_rewrite_gates(
                working,
                &m.bytes,
                &replacement,
                ctx.attestation,
                original.clause_ok,
                ctx.tagger,
            ) {
                held.push(Finding::new(
                    rule_id,
                    finding_range,
                    format!("frame {} held: {gate}", rule.id),
                    Tier::Suggest,
                ));
                return;
            }
            // A replacement that opens its line or sentence carries
            // the opener's capital itself: the splicer marks the
            // region as replacement text, which the recapitalization
            // pass deliberately never touches (same contract as paired
            // substitution).
            let replacement = if opens_line_or_sentence(working, m.bytes.start) {
                capitalize(&replacement)
            } else {
                replacement
            };
            let base = m.bytes;
            let (range, widened_replacement) =
                repair_article(working, base.clone(), replacement.clone());
            if splicer.can_apply(&range) {
                splicer.apply(range, &widened_replacement, rule_id, Tier::Fix);
            } else if splicer.can_apply(&base) {
                // Article widening crossed an earlier edit's boundary;
                // the un-widened rewrite is still sound.
                splicer.apply(base, &replacement, rule_id, Tier::Fix);
            }
        }
        CompiledKind::Guard | CompiledKind::Report => {}
    }
}

/// Whether `rule` is a bare verb+slot pattern whose matched slot opens
/// with a preposition or particle (`IN`/`TO`/`RP`): the transitive
/// substitution the rule encodes does not cover that construction.
fn bare_verb_slot_over_preposition(
    rule: &friction_packs::frame_bin::RuleView<'_>,
    m: &FrameMatch,
    tags: &[TaggedToken],
) -> bool {
    let ops: Vec<PatOp> = rule
        .pattern
        .clone()
        .map(|cell| cell.expect("embedded pack ops always decode").0)
        .collect();
    let bare_verb_slot = matches!(ops.as_slice(), [PatOp::Lit(_), PatOp::Slot(_)]);
    bare_verb_slot
        && m.slots.iter().flatten().next().is_some_and(|slot| {
            tags.get(slot.start)
                .is_some_and(|t| matches!(t.pos.as_str(), "IN" | "TO" | "RP"))
        })
}

/// Widens a deletion to swallow one adjacent space, so removing a word
/// never leaves a doubled space or a stranded one before punctuation
/// ("extraction seamlessly." must become "extraction.", never
/// "extraction ."). The leading space goes when what follows the match
/// is punctuation, whitespace, or the end; otherwise the trailing
/// space goes (a deleted sentence-opening word must not leave its
/// separator behind either).
fn widen_deletion(working: &str, range: Range<usize>) -> Range<usize> {
    let after = working[range.end..].chars().next();
    let before_is_space = working[..range.start].ends_with(' ');
    let after_needs_left_join =
        after.is_none_or(|c| c.is_whitespace() || matches!(c, '.' | ',' | '!' | '?' | ':' | ';'));
    if before_is_space && after_needs_left_join {
        return range.start - 1..range.end;
    }
    if after == Some(' ') {
        return range.start..range.end + 1;
    }
    range
}

/// Widens an edit to repair `a`/`an` agreement: when the token
/// directly before the edited range is the indefinite article and the
/// first word the reader now meets after it changed vowel-ness, the
/// article joins the edit and flips ("an invaluable resource" with the
/// adjective deleted must read "a resource", never "an resource").
fn repair_article(
    working: &str,
    range: Range<usize>,
    replacement: String,
) -> (Range<usize>, String) {
    let pre = &working[..range.start];
    let trimmed = pre.trim_end();
    let article = ["a", "an", "A", "An"].into_iter().find(|a| {
        trimmed.ends_with(a)
            && trimmed[..trimmed.len() - a.len()]
                .chars()
                .next_back()
                .is_none_or(|c| !c.is_alphanumeric())
    });
    let Some(article) = article else {
        return (range, replacement);
    };
    // The first alphabetic word the article now precedes: the
    // replacement's first word, or (for a deletion) the text after the
    // edited range.
    let next_text = if replacement.trim().is_empty() {
        working[range.end..].trim_start()
    } else {
        replacement.trim_start()
    };
    let Some(first) = next_text.chars().find(|c| c.is_alphabetic()) else {
        return (range, replacement);
    };
    let vowel = "aeiouAEIOU".contains(first);
    let fixed = match (article, vowel) {
        ("a", true) => "an",
        ("an", false) => "a",
        ("A", true) => "An",
        ("An", false) => "A",
        _ => return (range, replacement),
    };
    let article_start = trimmed.len() - article.len();
    let between = &working[trimmed.len()..range.start];
    let widened = article_start..range.end;
    let new_replacement = if replacement.trim().is_empty() {
        // Deletion: the corrected article stands alone; the text after
        // the range supplies the separator (append one only if it
        // doesn't).
        if working[range.end..].starts_with(char::is_whitespace) {
            fixed.to_string()
        } else {
            format!("{fixed} ")
        }
    } else {
        format!("{fixed}{between}{replacement}")
    };
    (widened, new_replacement)
}

/// Maps a pattern *element* index to its op-cell index (one group =
/// one element; the compiler records agreement targets by element).
fn frame_element_to_op(pattern_ops: &[(PatOp, bool)], element: u8) -> Option<usize> {
    let mut element_index = 0u8;
    let mut i = 0;
    while i < pattern_ops.len() {
        if element_index == element {
            return Some(i);
        }
        if matches!(pattern_ops[i].0, PatOp::GroupStart { .. }) {
            while i < pattern_ops.len() && !matches!(pattern_ops[i].0, PatOp::GroupEnd) {
                i += 1;
            }
        }
        i += 1;
        element_index += 1;
    }
    None
}

/// Renders a matched rule's template into replacement text.
///
/// Slot copies re-emit the captured working-text bytes verbatim;
/// inflected realizations go through [`friction_nlp::inflect`], with
/// the bare lemma as the fallback when no inflection applies. Returns
/// `None` when a copy references something the match did not capture.
/// The candidate is then held rather than spliced half-formed.
fn realize_template(
    view: &FramePackView<'_>,
    rule: &friction_packs::frame_bin::RuleView<'_>,
    m: &FrameMatch,
    tags: &[TaggedToken],
    working: &str,
) -> Option<String> {
    let pattern_ops: Vec<(PatOp, bool)> = rule
        .pattern
        .clone()
        .map(|cell| cell.expect("embedded pack ops always decode"))
        .collect();
    let token_text = |index: usize| -> &str {
        let range = tags[index].token.range.clone();
        &working[range]
    };
    let mut pieces: Vec<String> = Vec::new();
    for op in rule.template.clone() {
        let op = op.expect("embedded pack ops always decode");
        let piece = match op {
            TplOp::Lit(id) => view.word(id)?.to_string(),
            TplOp::Slot(slot) => {
                let range = m.slots[usize::from(slot.code())].clone()?;
                let start = tags[range.start].token.range.start;
                let end = tags[range.end - 1].token.range.end;
                working[start..end].to_string()
            }
            TplOp::TagCopy(tag) => {
                let index = pattern_ops.iter().enumerate().find_map(|(i, (op, _))| {
                    (matches!(op, PatOp::Tag(t) if *t == tag) && m.op_tokens[i].is_some())
                        .then(|| m.op_tokens[i].expect("checked Some"))
                })?;
                token_text(index).to_string()
            }
            TplOp::Inflect { lemma, agree_elem } => {
                let lemma = view.word(lemma)?;
                let agreed = frame_element_to_op(&pattern_ops, agree_elem)
                    .and_then(|op_index| m.op_tokens.get(op_index).copied().flatten())
                    .map(token_text);
                // `inflect` transfers the matched surface's
                // capitalization onto the realization, so a
                // sentence-opening match keeps its capital.
                agreed
                    .and_then(|surface| friction_nlp::inflect(surface, lemma))
                    .unwrap_or_else(|| lemma.to_string())
            }
            TplOp::InflectForm { lemma, tag } => {
                let lemma = view.word(lemma)?;
                friction_nlp::inflect(form_probe(tag), lemma).unwrap_or_else(|| lemma.to_string())
            }
            TplOp::ClassRealize { class, tag } => {
                // Realize the member the match contains (found by lemma
                // among the matched tokens), in the tag's own form —
                // the authored tag, not the matched token's number:
                // NNS[insight_n] over a singular match still pluralizes.
                let members = view.class_members(class)?;
                let matched_member = m.tokens.clone().find_map(|index| {
                    let lemma = tags[index].lemma.to_lowercase();
                    members
                        .iter()
                        .filter_map(|member| match member.as_slice() {
                            [only] => view.word(*only),
                            _ => None,
                        })
                        .find(|word| **word == *lemma)
                });
                let member = matched_member.or_else(|| {
                    members.first().and_then(|member| match member.as_slice() {
                        [only] => view.word(*only),
                        _ => None,
                    })
                })?;
                friction_nlp::inflect(form_probe(tag), member).unwrap_or_else(|| member.to_string())
            }
            TplOp::AltCopy(ordinal) => {
                let index = m
                    .group_tokens
                    .get(usize::from(ordinal))
                    .copied()
                    .flatten()?;
                token_text(index).to_string()
            }
            TplOp::Clitic(clitic) => clitic.surface().to_string(),
        };
        pieces.push(piece);
    }
    let mut out = String::new();
    for piece in pieces {
        let no_space_before = piece
            .chars()
            .next()
            .is_none_or(|c| c == '\'' || matches!(c, '.' | ',' | '!' | '?' | ':' | ';'));
        if !out.is_empty() && !no_space_before {
            out.push(' ');
        }
        out.push_str(&piece);
    }
    (!out.is_empty()).then_some(out)
}

/// A surface whose shape carries the inflection `tag` asks for, used
/// as [`friction_nlp::inflect`]'s form template when no matched token
/// supplies agreement.
const fn form_probe(tag: FrameTag) -> &'static str {
    match tag {
        FrameTag::Vbz | FrameTag::Nns => "uses",
        FrameTag::Vbg => "using",
        FrameTag::Vbn | FrameTag::Vbd => "used",
        _ => "use",
    }
}

/// The rewrite twin of [`gates::check_deletion_gates`]: seam bigrams
/// at both splice boundaries, the clause gate over the candidate
/// sentence, and skeleton attestation over the replaced window.
/// Returns the failing gate's name, or `None` when all pass.
///
/// The seams are taken around the *changed core*, not the whole
/// replacement: a template's slot copies re-emit the matched text
/// verbatim, so "Expedite the rollout" -> "speed up the rollout"
/// shares the suffix "the rollout" with its original, and the only
/// genuinely new adjacency is ("up", "the") — testing ("the",
/// "rollout") would demand the corpus attest a pair the author
/// already wrote.
pub(crate) fn check_rewrite_gates(
    working: &str,
    match_range: &Range<usize>,
    replacement: &str,
    attestation: &friction_packs::AttestationPack,
    original_clause_ok: bool,
    tagger: &dyn Tagger,
) -> Option<&'static str> {
    let pre = &working[..match_range.start];
    let post = &working[match_range.end..];
    let pre_tokens = tokenize_str(pre, 0);
    let post_tokens = tokenize_str(post, 0);
    let original_tokens = tokenize_str(&working[match_range.clone()], 0);
    let replacement_tokens = tokenize_str(replacement, 0);

    let equals = |a: &friction_match::token::AnalysisToken,
                  b: &friction_match::token::AnalysisToken| {
        a.text.eq_ignore_ascii_case(&b.text)
    };
    let mut common_prefix = 0;
    while common_prefix < original_tokens.len()
        && common_prefix < replacement_tokens.len()
        && equals(
            &original_tokens[common_prefix],
            &replacement_tokens[common_prefix],
        )
    {
        common_prefix += 1;
    }
    let mut common_suffix = 0;
    while common_suffix < original_tokens.len().saturating_sub(common_prefix)
        && common_suffix < replacement_tokens.len().saturating_sub(common_prefix)
        && equals(
            &original_tokens[original_tokens.len() - 1 - common_suffix],
            &replacement_tokens[replacement_tokens.len() - 1 - common_suffix],
        )
    {
        common_suffix += 1;
    }
    let core = &replacement_tokens[common_prefix..replacement_tokens.len() - common_suffix];

    // Seams around the changed core. The left context is the last
    // token before it (from the shared prefix, else the pre-text); the
    // right context is the first token after it (from the shared
    // suffix, else the post-text). Sentence edges and a terminal `.`
    // are auto-safe, mirroring the deletion gate's asymmetry.
    let left_context = original_tokens[..common_prefix]
        .last()
        .or_else(|| pre_tokens.last());
    let right_context = original_tokens[original_tokens.len() - common_suffix..]
        .first()
        .or_else(|| post_tokens.first());
    if let (Some(left), Some(first)) = (left_context, core.first())
        && !attestation.bigram().attests(&left.text, &first.text)
    {
        return Some("left seam not attested");
    }
    let right_is_terminal = right_context
        .is_none_or(|t| t.kind == AnalysisTokenKind::Punctuation && t.text.as_ref() == ".");
    if !right_is_terminal
        && let (Some(last), Some(right)) = (core.last(), right_context)
        && !attestation.bigram().attests(&last.text, &right.text)
    {
        return Some("right seam not attested");
    }

    let candidate = format!("{pre}{replacement}{post}");
    let candidate_tags = tag(tagger, &candidate);
    if !gates::clause_gate(&candidate_tags, &candidate, original_clause_ok) {
        return Some("clause incomplete after rewrite");
    }

    // Skeleton window over the changed core: the coarse-tag run
    // covering its tokens plus one boundary slot each side must be
    // attested in the human corpus, the same table the deletion gate
    // consults for its seam window.
    let core_start = match_range.start + core.first().map_or(replacement.len(), |t| t.range.start);
    let core_end = match_range.start + core.last().map_or(replacement.len(), |t| t.range.end);
    let lo = candidate_tags
        .iter()
        .filter(|t| t.token.range.end <= core_start)
        .count();
    let inside = candidate_tags
        .iter()
        .filter(|t| t.token.range.start >= core_start && t.token.range.end <= core_end)
        .count();
    let hi = lo + inside + 1;
    let coarse = gates::coarse_tag_window(&candidate_tags);
    let coarse_refs: Vec<&str> = coarse.iter().map(AsRef::as_ref).collect();
    if !attestation.skeleton().window_attested(
        &coarse_refs,
        lo,
        hi.min(coarse_refs.len().saturating_sub(1)),
    ) {
        return Some("skeleton not attested after rewrite");
    }
    None
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
