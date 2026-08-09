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
//! register's own transducers: T10's output is a subject-less imperative
//! or infinitival (`t4_activize_to_passive` requires both an `Nsubj` and
//! a `Dobj`, which an imperative has neither of), so register can never
//! re-fire on restructure's own output.
//!
//! # Selection: gate every candidate, apply every survivor
//!
//! No cross-candidate scoring, no conflict resolution beyond
//! [`crate::document::resolve`]'s ordinary overlap guard: each
//! construction is self-contained within one sentence, so there is
//! nothing to weigh candidates against.

use friction_core::span::ranges_overlap;
use friction_core::{Finding, Patch, RuleId, Tier};
use friction_match::token::prose_scope;
use friction_nlp::{DepParser, Segmenter, Tagger};
use friction_packs::AttestationPack;
use friction_register::transduce::{RestructureOutcome, t10_ensures_that_restructure};

use crate::document::{apply, ends_with_sentence_terminal_punctuation, resolve};
use crate::error::EditError;
use crate::gates::{clause_ok, in_quoted_span};
use crate::parse_ctx::build_sentence_contexts;
use crate::sentence::check_rewrite_gates;

/// The rule id every T10 patch and held finding carries. Cited by name in
/// `frame-rules-en-v1.toml`'s retired `able.ensures-that-short::ensure`
/// row, so a reader following that citation lands here.
const RULE_ENSURES_THAT: RuleId = RuleId::new("restructure.ensures_that");

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
    let sentences = build_sentence_contexts(source, &units, tagger, parser);

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

        for outcome in t10_ensures_that_restructure(text, &ctx.tokens, &ctx.parse) {
            let (local_range, decline_reason, replacement) = match outcome {
                RestructureOutcome::Candidate { range, replacement } => {
                    (range, None, Some(replacement))
                }
                RestructureOutcome::Declined { range, reason } => (range, Some(reason), None),
            };
            let doc_range =
                (ctx.range.start + local_range.start)..(ctx.range.start + local_range.end);

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
