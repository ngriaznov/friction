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
    // inflections), so sentences without one skip tagging and parsing
    // entirely — see `build_sentence_contexts_where`'s docs for why
    // this is byte-safe by construction. Stems: "ensur" covers
    // ensure/ensures/ensured; a table key minus its final "e" covers
    // every regular inflection of that key.
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
