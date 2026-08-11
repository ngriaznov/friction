//! The restructure pass: parse-aware, per-instance-licensed rewrites with
//! no band and no rate to home toward.
//!
//! [`crate::register`] answers "is this document's *rate* of a cadence
//! feature outside its band" and greedily applies whichever candidate
//! moves that rate most. Restructuring rules answer a different
//! question — "does this one instance have a correct rewrite, or must it
//! decline" — decided per instance, the same policy
//! [`friction_match::frame_rewrite`]'s 920 pack rules already use.
//! Forcing a construction like `able.ensures-that-short::ensure` through
//! [`crate::register`]'s `Direction`/band machinery would make its
//! application conditional on a document-wide statistic, which is wrong:
//! "Ensure that the setting is enabled" becomes "Enable the setting"
//! regardless of how many other imperatives share its document.
//!
//! # Pass ordering
//!
//! [`run_restructure`] runs once, between the five/six-op convergence
//! loop and [`crate::register::run_register`] (see `document.rs`'s own
//! `edit_document`). Register must see the post-restructure text, since
//! restructuring removes exactly the spans register's own features count
//! (an agentless-passive-shaped `Ccomp`, a clause boundary). Placing it
//! before register is also provably idempotent with respect to
//! register's own transducers: T10's imperative/infinitival output is
//! subject-less (`t4_activize_to_passive` requires both an `Nsubj` and a
//! `Dobj`, which an imperative has neither of), and T11 drops the lemma
//! `surface` entirely, so `vsub.surface::*`'s own anchor cannot re-match
//! either. T10's finite-subject shape (§A1) is the one output shape this
//! argument doesn't cover unconditionally — it can produce a genuine
//! `Nsubj`+`Dobj` clause register's own T4 could, in principle, promote
//! back toward a passive — but that is the same judgment call T4 already
//! makes for any human- or machine-written sentence of that shape; it is
//! not a defect this pass introduces.
//!
//! # Selection: gate every candidate, apply every survivor
//!
//! No cross-candidate scoring, no conflict resolution beyond
//! [`crate::document::resolve`]'s ordinary overlap guard: each
//! construction is self-contained within one sentence, so there is
//! nothing to weigh candidates against.
//!
//! # Stale report-only findings (R2)
//!
//! `frame-rules-en-v1.toml`'s `vsub.surface::*` rows keep firing
//! report-only (`Tier::Suggest`) for every `surface`-lemma instance T11
//! doesn't fix, from the five/six-op loop's own bounded pass — untouched
//! by this module, exactly as designed (see `t11_transitive_substitution`'s
//! own docs). The `bounded_held.retain` call at the end of
//! [`run_restructure`] already dedups this for free: it drops every
//! `bounded_held` finding — T10's own `able.ensures-that-short` target
//! included — overlapping ANY patch this pass just accepted, T10's or
//! T11's alike, so a `surface`-lemma span this pass fixes never also
//! carries its own stale `vsub.surface::*` finding into `--suggest`
//! output.

use friction_core::span::ranges_overlap;
use friction_core::{Finding, Patch, RuleId, Tier};
use friction_match::token::prose_scope;
use friction_nlp::{DepParser, Segmenter, Tagger};
use friction_packs::AttestationPack;
use friction_register::transduce::{
    RestructureOutcome, t10_ensures_that_restructure, t11_transitive_substitution,
    t12_participial_closer_split, t13_em_dash_relative_chain_split,
};

use crate::document::{apply, ends_with_sentence_terminal_punctuation, resolve};
use crate::error::EditError;
use crate::gates::{clause_ok, in_quoted_span};
use crate::parse_ctx::build_sentence_contexts_where;
use crate::sentence::check_rewrite_gates;

/// The rule id every T10 patch and held finding carries. Cited by name in
/// `frame-rules-en-v1.toml`'s retired `able.ensures-that-short::ensure`
/// row, so a reader following that citation lands here.
const RULE_ENSURES_THAT: RuleId = RuleId::new("restructure.ensures_that");

/// The rule id every T11 (R2) patch and held finding carries. The
/// unmodified `vsub.surface::*` frame-pack rows keep their own separate
/// id and keep firing report-only for every instance T11 doesn't fix
/// (see this module's own `run_restructure` docs on the overlap dedup
/// that keeps a fixed span from also carrying a stale one of those).
const RULE_TRANSITIVE_SUBSTITUTION: RuleId = RuleId::new("restructure.transitive_substitution");

/// The rule id every T12 patch and held finding carries: the
/// sentence-final participial-closer split (see
/// `friction_register::transduce::t12_participial_closer_split`'s own
/// docs for the construction).
const RULE_PARTICIPIAL_CLOSER: RuleId = RuleId::new("restructure.participial_closer");

/// The rule id every T13 patch and held finding carries: the em-dash
/// relative-chain split (see
/// `friction_register::transduce::t13_em_dash_relative_chain_split`'s
/// own docs for the construction).
const RULE_EM_DASH_RELATIVE_CHAIN: RuleId = RuleId::new("restructure.em_dash_relative_chain");

/// Runs the restructure pass once over `source`.
///
/// `source` is expected to already be the five-operation pipeline's
/// converged output (see this module's own docs on pass ordering).
///
/// `bounded_held` is the five/six-op convergence loop's own final-pass
/// held findings, filtered in place: once this pass accepts a patch on a
/// span, an earlier held finding overlapping that span refers to bytes
/// about to be rewritten, so it's dropped rather than left to co-appear
/// beside the real fix in `--suggest` output.
///
/// # Errors
/// Returns [`EditError`] if `source` fails to parse or segment.
pub fn run_restructure(
    source: &str,
    syntax: friction_parse::Syntax,
    attestation: &AttestationPack,
    tagger: &dyn Tagger,
    parser: &dyn DepParser,
    segmenter: &dyn Segmenter,
    bounded_held: &mut Vec<Finding>,
) -> Result<(String, crate::document::PassReport), EditError> {
    let document = friction_parse::parse_with(source, syntax)?;
    let units = prose_scope(&document, segmenter);
    // Every T10/T11 candidate requires a literal trigger word ("ensure"
    // in some finite form; a `[transitive_verbs]` key in any of its
    // inflections), and every T12/T13 candidate requires a literal
    // trigger character/suffix of its own (see [`t12_trigger_present`]/
    // [`t13_trigger_present`]'s own docs), so a sentence with none of
    // these skips tagging and parsing entirely — see
    // `build_sentence_contexts_where`'s docs for why this is byte-safe
    // by construction. Stems: "ensur" covers ensure/ensures/ensured; a
    // table key minus its final "e" covers every regular inflection of
    // that key.
    let stems: Vec<String> = std::iter::once("ensur".to_owned())
        .chain(
            friction_nlp::LEXICON_EN
                .transitive_verbs
                .keys()
                .map(|k| k.trim_end_matches('e').to_owned()),
        )
        .collect();
    let sentences = build_sentence_contexts_where(source, &units, tagger, parser, |text| {
        let folded = text.to_ascii_lowercase();
        stems.iter().any(|s| folded.contains(s.as_str()))
            || t12_trigger_present(text)
            || t13_trigger_present(text)
    });

    let mut patches: Vec<Patch> = Vec::new();
    let mut held: Vec<Finding> = Vec::new();

    for ctx in &sentences {
        let text = &source[ctx.range.clone()];
        // Same fragment guards `register::collect_candidates` applies: a
        // range the segmenter cut short at an excluded construct is a
        // fragment, not a clause, and a run that continues a previous
        // one is missing its left context. T10's rewrites are
        // clause-sized, so both are unsafe input.
        if !ends_with_sentence_terminal_punctuation(text) || ctx.continues_previous {
            continue;
        }
        let original_clause_ok = clause_ok(&ctx.tokens, text);

        collect_t10(
            text,
            ctx,
            attestation,
            tagger,
            original_clause_ok,
            &mut patches,
            &mut held,
        );
        collect_t11(
            text,
            ctx,
            attestation,
            tagger,
            original_clause_ok,
            &mut patches,
            &mut held,
        );
        collect_t12(
            text,
            ctx,
            attestation,
            tagger,
            original_clause_ok,
            &mut patches,
            &mut held,
        );
        collect_t13(
            text,
            ctx,
            attestation,
            tagger,
            original_clause_ok,
            &mut patches,
            &mut held,
        );
    }

    let (accepted, dropped) = resolve(source, patches);
    let patches_applied = accepted.len();
    let next = apply(source, &accepted);

    bounded_held.retain(|finding| {
        !accepted
            .iter()
            .any(|patch| ranges_overlap(&patch.range, &finding.range))
    });

    Ok((
        next,
        crate::document::PassReport {
            patches_applied,
            patches_dropped: dropped,
            applied_patches: accepted,
            held,
        },
    ))
}

/// `true` if `text` contains a comma directly followed (after optional
/// whitespace) by a word ending in `"ing"` — the necessary text-level
/// condition for a [`t12_participial_closer_split`] candidate (a
/// comma-attached `VBG`).
///
/// Approximates the tagger's own `VBG` tag with a bare suffix check,
/// same convention as the stem list above: a false positive here only
/// costs a discarded tag+parse (the real POS/dependency checks inside
/// `t12_participial_closer_split` still gate the actual match), and a
/// false negative would silently skip a genuine candidate, which
/// English's own spelling regularity (every gerund/present participle
/// ends `"-ing"`) rules out.
fn t12_trigger_present(text: &str) -> bool {
    text.split(',').skip(1).any(|piece| {
        let word: String = piece
            .trim_start()
            .chars()
            .take_while(|c| c.is_alphanumeric())
            .collect();
        word.len() > 3 && word.to_ascii_lowercase().ends_with("ing")
    })
}

/// `true` if `text` contains an em dash — the necessary condition for a
/// [`t13_em_dash_relative_chain_split`] candidate, which always matches
/// at exactly one em-dash token.
fn t13_trigger_present(text: &str) -> bool {
    text.contains('\u{2014}')
}

/// One sentence's worth of T10 candidates: gates every
/// [`RestructureOutcome`] and pushes either a held [`Finding`] or an
/// accepted [`Patch`] — split out of [`run_restructure`] only to keep
/// that function's own line count down.
#[allow(clippy::too_many_arguments)] // mirrors run_restructure's own one assembly point
fn collect_t10(
    text: &str,
    ctx: &crate::parse_ctx::SentenceCtx,
    attestation: &AttestationPack,
    tagger: &dyn Tagger,
    original_clause_ok: bool,
    patches: &mut Vec<Patch>,
    held: &mut Vec<Finding>,
) {
    for outcome in t10_ensures_that_restructure(text, &ctx.tokens, &ctx.parse) {
        let (local_range, decline_reason, replacement) = match outcome {
            RestructureOutcome::Candidate { range, replacement } => {
                (range, None, Some(replacement))
            }
            RestructureOutcome::Declined { range, reason } => (range, Some(reason), None),
        };
        let doc_range = (ctx.range.start + local_range.start)..(ctx.range.start + local_range.end);

        if let Some(reason) = decline_reason {
            held.push(Finding::new(
                RULE_ENSURES_THAT,
                doc_range,
                format!("restructure held: {reason}"),
                Tier::Suggest,
            ));
            continue;
        }
        let replacement = replacement.expect("Candidate always carries a replacement");

        if in_quoted_span(text, &local_range) {
            held.push(Finding::new(
                RULE_ENSURES_THAT,
                doc_range,
                "restructure held: matched inside quotation",
                Tier::Suggest,
            ));
            continue;
        }

        match check_rewrite_gates(
            text,
            &local_range,
            &replacement,
            attestation,
            original_clause_ok,
            tagger,
        ) {
            Some(reason) => {
                held.push(Finding::new(
                    RULE_ENSURES_THAT,
                    doc_range,
                    format!("restructure held: {reason}"),
                    Tier::Suggest,
                ));
            }
            None => {
                patches.push(Patch::new(
                    doc_range,
                    replacement.to_string(),
                    RULE_ENSURES_THAT,
                    Tier::Fix,
                ));
            }
        }
    }
}

/// One sentence's worth of T11 candidates: no `Declined` variant to
/// match on — every structural guard already ran inside
/// [`t11_transitive_substitution`] itself (see its own docs on why the
/// decline side needs no accounting here), so every survivor reaching
/// this function only ever needs the same runtime gates [`collect_t10`]
/// applies. Split out for the same line-count reason as that function.
fn collect_t11(
    text: &str,
    ctx: &crate::parse_ctx::SentenceCtx,
    attestation: &AttestationPack,
    tagger: &dyn Tagger,
    original_clause_ok: bool,
    patches: &mut Vec<Patch>,
    held: &mut Vec<Finding>,
) {
    for candidate in t11_transitive_substitution(text, &ctx.tokens, &ctx.parse) {
        let doc_range =
            (ctx.range.start + candidate.range.start)..(ctx.range.start + candidate.range.end);

        if in_quoted_span(text, &candidate.range) {
            held.push(Finding::new(
                RULE_TRANSITIVE_SUBSTITUTION,
                doc_range,
                "restructure held: matched inside quotation",
                Tier::Suggest,
            ));
            continue;
        }

        match check_rewrite_gates(
            text,
            &candidate.range,
            &candidate.replacement,
            attestation,
            original_clause_ok,
            tagger,
        ) {
            Some(reason) => {
                held.push(Finding::new(
                    RULE_TRANSITIVE_SUBSTITUTION,
                    doc_range,
                    format!("restructure held: {reason}"),
                    Tier::Suggest,
                ));
            }
            None => {
                patches.push(Patch::new(
                    doc_range,
                    candidate.replacement.to_string(),
                    RULE_TRANSITIVE_SUBSTITUTION,
                    Tier::Fix,
                ));
            }
        }
    }
}

/// `true` if `replacement` opens with the sentence-splitting period
/// every T12/T13 replacement carries (`". That ..."`/`". That is"`),
/// which is how [`check_split_rewrite_gates`] locates the new sentence
/// boundary without re-parsing punctuation out of the candidate text.
fn starts_a_new_sentence(replacement: &str) -> bool {
    replacement.starts_with(". ")
}

/// A whole tag run's own coarse-tag skeleton (its own `<S>`/`<E>`
/// sentinels, not a caller's shared wrapped window) is attested as a
/// complete unit in the human corpus.
fn whole_sentence_skeleton_attested(
    tags: &[friction_nlp::TaggedToken],
    attestation: &AttestationPack,
) -> bool {
    if tags.is_empty() {
        return false;
    }
    let coarse = crate::gates::coarse_tag_window(tags);
    let coarse_refs: Vec<&str> = coarse.iter().map(AsRef::as_ref).collect();
    attestation
        .skeleton()
        .window_attested(&coarse_refs, 0, coarse_refs.len().saturating_sub(1))
}

/// T12/T13's own rewrite gate — the construction-specific counterpart to
/// [`check_rewrite_gates`], required because both ops introduce a
/// sentence-terminal period no rewrite this crate applies did before.
///
/// Measured directly against the shipped tagger and the embedded
/// attestation pack: [`check_rewrite_gates`]'s own skeleton half re-tags
/// the whole pre/replacement/post run as one blob, then requires a
/// coarse-tag window straddling the edit to be attested as part of a
/// single continuous human sentence. For every T12/T13 candidate tried
/// (a dozen distinct wordings across both ops), that window spans across
/// the very period this construction introduces — which asks the
/// training corpus for a genuine single sentence whose own skeleton has
/// a period followed by more words before its end, a shape a
/// correctly-segmented human sentence essentially never has by
/// definition (that IS where a segmenter cuts). Every candidate tried
/// declined on exactly this gate regardless of wording: not a
/// corpus-rarity problem, an architectural mismatch between a gate built
/// for T10/T11's own in-sentence-only rewrites and this construction's
/// own sentence-splitting one.
///
/// This function keeps [`check_rewrite_gates`]'s other two checks in
/// spirit — a seam bigram, a clause gate — but scopes both correctly
/// around the new boundary instead of across it:
/// - Right seam only. The left boundary is always an existing word
///   immediately followed by the period this construction itself
///   introduces, the same "terminal punctuation needs no bigram check"
///   shape [`crate::gates::check_deletion_gates`] already treats as
///   auto-safe on its own right edge, mirrored here on the left. The
///   right seam (the replacement's own last word against whatever
///   follows the matched span, unchanged) is the one genuinely new
///   adjacency this construction can introduce: T12's re-conjugated verb
///   against the untouched word after it. T13's copy-through "is" seam
///   is unchanged from the original, so it always passes trivially.
/// - Clause and skeleton on both resulting sentences independently, each
///   re-tagged and wrapped in its own start/end sentinels — the shape
///   the attestation pack was actually mined from. Verified directly
///   against the embedded pack: both of this crate's own tests' second
///   sentences ("That makes it easy to onboard new projects.", "That is
///   why the config stays small.") are attested this way, confirming
///   the construction's own output is genuinely human-shaped; the
///   blob-wide check above was simply asking it an unanswerable
///   question.
fn check_split_rewrite_gates(
    text: &str,
    match_range: &std::ops::Range<usize>,
    replacement: &str,
    attestation: &AttestationPack,
    tagger: &dyn Tagger,
) -> Option<&'static str> {
    debug_assert!(
        starts_a_new_sentence(replacement),
        "every T12/T13 replacement opens with its own sentence-splitting period"
    );
    let pre = &text[..match_range.start];
    let post = &text[match_range.end..];

    let replacement_tokens = friction_match::token::tokenize_str(replacement, 0);
    let post_tokens = friction_match::token::tokenize_str(post, 0);
    if let (Some(last), Some(right)) = (replacement_tokens.last(), post_tokens.first())
        && !attestation.bigram().attests(&last.text, &right.text)
    {
        return Some("right seam not attested");
    }

    let candidate = format!("{pre}{replacement}{post}");
    let split = pre.len() + 1; // one byte past the period `replacement` itself opens with.
    let Some(first_raw) = candidate.get(..split) else {
        return Some("malformed sentence-split boundary");
    };
    let Some(second_raw) = candidate.get(split..) else {
        return Some("malformed sentence-split boundary");
    };

    let first = first_raw.trim();
    let first_tags = crate::gates::tag(tagger, first);
    if !clause_ok(&first_tags, first) {
        return Some("first resulting sentence has no complete clause of its own");
    }
    if !whole_sentence_skeleton_attested(&first_tags, attestation) {
        return Some("first resulting sentence's skeleton is not attested");
    }

    let second = second_raw.trim();
    let second_tags = crate::gates::tag(tagger, second);
    if !clause_ok(&second_tags, second) {
        return Some("second resulting sentence has no complete clause of its own");
    }
    if !whole_sentence_skeleton_attested(&second_tags, attestation) {
        return Some("second resulting sentence's skeleton is not attested");
    }

    None
}

/// One sentence's worth of T12 candidates: gates every
/// [`RestructureOutcome`] and pushes either a held [`Finding`] or an
/// accepted [`Patch`] — mirrors [`collect_t10`]'s own shape
/// (`RestructureOutcome` match, quotation guard, then a rewrite gate),
/// using [`check_split_rewrite_gates`] in place of
/// [`check_rewrite_gates`] (see that function's own docs for why).
#[allow(clippy::too_many_arguments)] // mirrors collect_t10's own one assembly point
fn collect_t12(
    text: &str,
    ctx: &crate::parse_ctx::SentenceCtx,
    attestation: &AttestationPack,
    tagger: &dyn Tagger,
    _original_clause_ok: bool,
    patches: &mut Vec<Patch>,
    held: &mut Vec<Finding>,
) {
    for outcome in t12_participial_closer_split(text, &ctx.tokens, &ctx.parse) {
        let (local_range, decline_reason, replacement) = match outcome {
            RestructureOutcome::Candidate { range, replacement } => {
                (range, None, Some(replacement))
            }
            RestructureOutcome::Declined { range, reason } => (range, Some(reason), None),
        };
        let doc_range = (ctx.range.start + local_range.start)..(ctx.range.start + local_range.end);

        if let Some(reason) = decline_reason {
            held.push(Finding::new(
                RULE_PARTICIPIAL_CLOSER,
                doc_range,
                format!("restructure held: {reason}"),
                Tier::Suggest,
            ));
            continue;
        }
        let replacement = replacement.expect("Candidate always carries a replacement");

        if in_quoted_span(text, &local_range) {
            held.push(Finding::new(
                RULE_PARTICIPIAL_CLOSER,
                doc_range,
                "restructure held: matched inside quotation",
                Tier::Suggest,
            ));
            continue;
        }

        match check_split_rewrite_gates(text, &local_range, &replacement, attestation, tagger) {
            Some(reason) => {
                held.push(Finding::new(
                    RULE_PARTICIPIAL_CLOSER,
                    doc_range,
                    format!("restructure held: {reason}"),
                    Tier::Suggest,
                ));
            }
            None => {
                patches.push(Patch::new(
                    doc_range,
                    replacement.to_string(),
                    RULE_PARTICIPIAL_CLOSER,
                    Tier::Fix,
                ));
            }
        }
    }
}

/// One sentence's worth of T13 candidates: same shape as [`collect_t12`]
/// over [`t13_em_dash_relative_chain_split`]'s own output instead.
#[allow(clippy::too_many_arguments)] // mirrors collect_t10's own one assembly point
fn collect_t13(
    text: &str,
    ctx: &crate::parse_ctx::SentenceCtx,
    attestation: &AttestationPack,
    tagger: &dyn Tagger,
    _original_clause_ok: bool,
    patches: &mut Vec<Patch>,
    held: &mut Vec<Finding>,
) {
    for outcome in t13_em_dash_relative_chain_split(text, &ctx.tokens) {
        let (local_range, decline_reason, replacement) = match outcome {
            RestructureOutcome::Candidate { range, replacement } => {
                (range, None, Some(replacement))
            }
            RestructureOutcome::Declined { range, reason } => (range, Some(reason), None),
        };
        let doc_range = (ctx.range.start + local_range.start)..(ctx.range.start + local_range.end);

        if let Some(reason) = decline_reason {
            held.push(Finding::new(
                RULE_EM_DASH_RELATIVE_CHAIN,
                doc_range,
                format!("restructure held: {reason}"),
                Tier::Suggest,
            ));
            continue;
        }
        let replacement = replacement.expect("Candidate always carries a replacement");

        if in_quoted_span(text, &local_range) {
            held.push(Finding::new(
                RULE_EM_DASH_RELATIVE_CHAIN,
                doc_range,
                "restructure held: matched inside quotation",
                Tier::Suggest,
            ));
            continue;
        }

        match check_split_rewrite_gates(text, &local_range, &replacement, attestation, tagger) {
            Some(reason) => {
                held.push(Finding::new(
                    RULE_EM_DASH_RELATIVE_CHAIN,
                    doc_range,
                    format!("restructure held: {reason}"),
                    Tier::Suggest,
                ));
            }
            None => {
                patches.push(Patch::new(
                    doc_range,
                    replacement.to_string(),
                    RULE_EM_DASH_RELATIVE_CHAIN,
                    Tier::Fix,
                ));
            }
        }
    }
}
