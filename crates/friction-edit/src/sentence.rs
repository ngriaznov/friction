//! Per-sentence orchestrator.
//!
//! The four operations, in ALGORITHMS.md §4's fixed order — ritual,
//! paired substitution, derivational pivot (loop, max 2), gated span
//! deletion — followed by a final recapitalization pass.

use std::collections::{BTreeMap, BTreeSet};
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
const RULE_RECAPITALIZE: RuleId = RuleId::new("edit.recapitalize");

/// The result of running the four-operation pipeline over one sentence.
#[derive(Debug, Default)]
pub struct SentenceOutcome {
    /// Accepted edits, as byte-honest patches against the document's
    /// original source.
    pub patches: Vec<Patch>,
    /// Gate-held candidates, Suggest-tier diagnostics only.
    pub held: Vec<Finding>,
    /// `true` if this sentence was deleted whole by the ritual operation
    /// (in which case none of the other three operations ran).
    pub whole_sentence_deleted: bool,
}

/// A sentence's position within its prose unit.
///
/// Its own byte range, plus enough of its neighbors' ranges to compute a
/// whole-sentence ritual deletion's adjacent-separator consumption and
/// discourse-binding check. Bundled into one type so [`edit_sentence`]
/// takes a reasonable number of parameters.
#[derive(Debug, Clone)]
pub struct SentencePosition {
    /// This sentence's own byte range into `source`.
    pub range: Range<usize>,
    /// The previous sentence's own end offset within the same prose unit,
    /// if any.
    pub prev_end: Option<usize>,
    /// The next sentence's own range within the same prose unit, if any.
    pub next_range: Option<Range<usize>>,
    /// `false` if this sentence is not actually a fresh sentence but the
    /// leading fragment of a prose unit that was split off its
    /// predecessor purely by an intervening excluded construct (inline
    /// code, a link/image boundary, raw HTML, a footnote reference, a
    /// task-list marker, or a non-adjacent structural gap like a
    /// block-quote continuation marker) with no sentence-terminal
    /// punctuation between them — i.e. the segmenter had no way to see
    /// that this "sentence" is really still mid-sentence. Recapitalizing
    /// such a fragment's first letter would rewrite real prose the same
    /// way capitalizing a genuine sentence-initial word does not. Always
    /// `true` for a sentence that isn't first in its prose unit — only a
    /// unit's own first sentence can be a continuation.
    pub is_sentence_start: bool,
    /// `true` if this sentence is the entire prose content of an item in
    /// an ordered (numbered) list — deleting it whole would leave a bare
    /// item marker and break the counted enumeration, so a whole-sentence
    /// ritual deletion is held instead.
    pub sole_content_of_ordered_list_item: bool,
}

/// The packs and tools every stage of [`edit_sentence`] shares — bundled
/// so the function's own parameter count stays reasonable.
pub struct EditContext<'a> {
    pub inventory: &'a InventoryPack,
    pub attestation: &'a AttestationPack,
    pub tagger: &'a dyn Tagger,
    /// Document-local casing evidence for this pass's untouched source
    /// text — see [`DocumentCasing`].
    pub casing: &'a DocumentCasing,
}

/// Document-local evidence that a word's lowercase spelling is the
/// author's deliberate choice rather than a missing capital.
///
/// Collected from every in-scope prose sentence of the current pass's
/// untouched text, before any sentence is edited.
///
/// The recapitalization guard consults it for a sentence whose *own
/// original* lowercase opener survived that sentence's edits: seeing the
/// same word lowercase mid-sentence elsewhere, or opening more than one
/// sentence in lowercase, marks the casing as deliberate (a product name
/// like `mimalloc`), so the opener is held rather than capitalized. An
/// opener a leading deletion newly exposed never consults this — every
/// ordinary word occurs lowercase mid-sentence somewhere, and such an
/// opener carries no authorial sentence-start casing intent at all.
#[derive(Debug, Default)]
pub struct DocumentCasing {
    /// Words seen with a lowercase first letter anywhere other than at a
    /// genuine sentence start (including every word of a mid-sentence
    /// continuation fragment).
    mid_sentence_lowercase: BTreeSet<Box<str>>,
    /// How many genuine sentence starts each lowercase-initial word
    /// opens.
    sentence_initial_lowercase: BTreeMap<Box<str>, usize>,
}

impl DocumentCasing {
    /// Records one sentence's word tokens. `is_sentence_start` is the
    /// same flag [`SentencePosition`] carries: `false` marks a
    /// continuation fragment, whose every word — the first included — is
    /// really mid-sentence text.
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
    /// deliberate: it also occurs lowercase mid-sentence, opens at least
    /// one *other* sentence in lowercase, or is shaped like an identifier
    /// (contains a digit, `_`, or `-`).
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

/// Per-sentence state computed once from the untouched original text and
/// threaded through every stage: the clause-completeness baseline every
/// gate's "enforced only when the original had one" clause checks
/// against, and the original's own finite-verb surface words (the
/// paired-substitution clause gate's tagger-weakness fallback; see
/// [`gates::original_finite_verb_words`]'s own docs).
struct OriginalState {
    clause_ok: bool,
    verb_words: BTreeSet<Box<str>>,
}

/// Runs the four-operation pipeline over one sentence.
#[must_use]
pub fn edit_sentence(
    source: &str,
    position: &SentencePosition,
    ctx: &EditContext<'_>,
    pivot_budget: &mut PivotBudget,
) -> SentenceOutcome {
    let sentence_range = position.range.clone();
    let original_text = &source[sentence_range.clone()];
    if original_text.trim().is_empty() {
        return SentenceOutcome::default();
    }
    let original_tags = tag(ctx.tagger, original_text);
    let original = OriginalState {
        clause_ok: clause_ok(&original_tags, original_text),
        verb_words: gates::original_finite_verb_words(&original_tags, original_text),
    };

    let mut held: Vec<Finding> = Vec::new();
    if let Some(outcome) = try_ritual_deletion(source, position, ctx, &mut held) {
        return outcome;
    }

    let mut splicer = SentenceSplicer::new(source, sentence_range.clone());

    run_substitution(&mut splicer, &mut held, &sentence_range, ctx, &original);
    run_pivot(&mut splicer, &mut held, &sentence_range, ctx, pivot_budget);
    run_deletion(&mut splicer, &mut held, &sentence_range, ctx, &original);
    // Only a genuine sentence start should ever have its leading letter
    // capitalized — a fragment that merely follows an excluded construct
    // (inline code, a link, ...) with no sentence-terminal punctuation
    // before it is not a sentence start, and recapitalizing it would
    // rewrite real prose (see `SentencePosition::is_sentence_start`'s own
    // docs).
    if position.is_sentence_start {
        recapitalize(&mut splicer, ctx.casing);
    }

    SentenceOutcome {
        patches: splicer.finish(),
        held,
        whole_sentence_deleted: false,
    }
}

/// Operation 1 (ritual deletion). Returns `Some` (a whole-sentence
/// deletion) if a ritual frame matched and no hold applies, `None` if the
/// sentence should proceed to the other three operations (pushing a
/// Suggest-tier hold onto `held` when a frame matched but could not be
/// acted on).
///
/// Unconditional on an actionable match, mirroring `fix_sentence`'s own
/// ritual step exactly: `for pat in RITUAL: if pat.search(s): return "",
/// [...]` — no discourse-binding check on the following sentence. (A
/// block-level ritual — a whole preview paragraph, gated on its content
/// being covered by adjacent structure — is a distinct case this function
/// does not implement at all; it is not a broader license to hold an
/// ordinary sentence-level ritual match.) Two holds do apply:
/// - a frame matching only inside a double-quoted span is a mention of a
///   ritual phrase, not a use of one ([`gates::in_quoted_span`]);
/// - a sentence that is the entire content of a numbered list item is
///   never deleted whole — that would leave a bare item marker and break
///   the counted enumeration.
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

/// Operation 3 (paired substitution): applies every pack pattern that
/// matches the current working text, in pack order, gating each on
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
            if range.start == 0
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

/// Operation 4 (derivational pivot, loop, max 2): scans the current
/// working text left to right for a licensed light-verb construction,
/// applying at most 2 across this sentence, gated by `pivot_budget`.
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

/// Operation 2 (gated span deletion): tries every deletion-span pattern,
/// in pack order, against the current (post substitution/pivot) working
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

/// Final step: if a deletion exposed a lowercase-initial opener, splice a
/// one-char uppercase substitution at that position.
///
/// Three guards keep this from rewriting prose the engine has no business
/// touching:
/// - a sentence with no accepted edits is left exactly as written — a
///   recapitalization-only patch would "fix" the author's own casing;
/// - replacement text at position 0 is never touched (the substitution
///   stage already capitalizes its own replacements when needed);
/// - a lowercase opener the author wrote at sentence start (the
///   sentence's original first word, untouched by this sentence's own
///   edits) is held when the document's own casing evidence marks it
///   deliberate — see [`DocumentCasing`]. An opener newly exposed by a
///   leading deletion carries no such intent and is capitalized
///   unconditionally.
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
