//! Shared gate helpers: the clause gate, and the gated-span-deletion seam
//! + skeleton gates.
//!
//! Every gate here operates on real, case-preserved text tagged directly
//! by the shipped tagger — the same convention `corpus-tool attest` used
//! to mine the [`friction_packs::AttestationPack`] this module queries,
//! rather than reconstructing a lowercased, pre-tokenized word list the
//! way the Python reference tags its own candidate spans. This keeps the
//! tag distribution measured here consistent with what the pack was
//! built from.

use std::collections::BTreeSet;
use std::ops::Range;

use friction_match::token::{AnalysisTokenKind, tokenize_str};
use friction_nlp::{
    FINITE_VERB_TAGS, TaggedToken, Tagger, coarse_tag, has_finite_verb, is_imperative_initial,
};
use friction_packs::AttestationPack;

/// `true` if `tokens` (tagged from `text`) satisfies clause-completeness
/// on its own: a finite verb somewhere, or an imperative-initial `VB`.
///
/// Falls back to [`has_ambiguous_s_verb`] alongside the strict Penn-tag
/// check — a measured compensation for the tagger's own weakness here,
/// not a loosening of what counts as a clause.
#[must_use]
pub fn clause_ok(tokens: &[TaggedToken], text: &str) -> bool {
    has_finite_verb(tokens) || is_imperative_initial(tokens) || has_ambiguous_s_verb(tokens, text)
}

/// A narrow, targeted compensation for a measured weakness in the
/// shipped tagger.
///
/// An `-s`-suffixed word immediately followed by an infinitival `to VB`
/// complement (`"needs to be placed"`-shaped) is unambiguous evidence of
/// a finite verb regardless of the token's own tag — English's `-s` is
/// genuinely ambiguous between plural noun and third-person-singular
/// verb, and this tagger resolves it wrong in this shape (measured:
/// `"the configuration file needs to be placed"` tags `needs` as `RB`,
/// not `VBZ`, in every context tried).
///
/// Deliberately narrow: a bare "ends in s, tagged NN/NNS/RB" check would
/// open a much larger false-positive class (`"a place to go"`, `"nothing
/// to lose"` never satisfy this, since neither ends in `s`).
#[must_use]
pub fn has_ambiguous_s_verb(tokens: &[TaggedToken], text: &str) -> bool {
    for i in 0..tokens.len().saturating_sub(2) {
        let pos = tokens[i].pos.as_str();
        if !matches!(pos, "NN" | "NNS" | "RB" | "RBS") {
            continue;
        }
        let surface = &text[tokens[i].token.range.clone()];
        if surface.len() < 3 || !surface.ends_with('s') {
            continue;
        }
        if tokens[i + 1].pos.as_str() != "TO" {
            continue;
        }
        if tokens[i + 2].pos.as_str() == "VB" {
            return true;
        }
    }
    false
}

/// The post-edit clause gate: `clause_ok(post_edit_tokens, text) ||
/// !original_had_a_clause` — enforced only when the original sentence had
/// one.
#[must_use]
pub fn clause_gate(post_edit_tokens: &[TaggedToken], text: &str, original_clause_ok: bool) -> bool {
    clause_ok(post_edit_tokens, text) || !original_clause_ok
}

/// Tags `text` from scratch (case-preserved, offset 0) — the one place
/// this crate re-runs the tagger mid-pipeline, matching the reference
/// engine's own re-tagging-per-candidate discipline.
#[must_use]
pub fn tag(tagger: &dyn Tagger, text: &str) -> Vec<TaggedToken> {
    tagger.tag(text, 0)
}

/// The outcome of checking a gated-span-deletion candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeletionGateOutcome {
    /// Every gate passed; the deletion is safe to apply.
    Allowed,
    /// The seam gate failed: the surviving left/right token pair is not
    /// attested at a human sentence boundary.
    SeamNotAttested,
    /// The seam gate passed, but the excised sentence's POS-skeleton
    /// window around the deletion point is not attested.
    SkeletonNotAttested,
    /// The seam and skeleton gates passed, but deleting the span would
    /// leave the sentence without a finite verb (when the original had
    /// one).
    ClauseIncomplete,
}

/// Checks every gate required for deleting
/// `match_range` (a byte range into `working_text`) with no replacement.
///
/// `original_clause_ok` is `clause_ok` computed once from the untouched
/// original sentence.
#[must_use]
pub fn check_deletion_gates(
    working_text: &str,
    match_range: Range<usize>,
    attestation: &AttestationPack,
    original_clause_ok: bool,
    tagger: &dyn Tagger,
) -> DeletionGateOutcome {
    let pre = &working_text[..match_range.start];
    let post = &working_text[match_range.end..];

    let pre_tokens = tokenize_str(pre, 0);
    let post_tokens = tokenize_str(post, 0);

    let sentence_initial = pre_tokens.is_empty();
    let left = pre_tokens.last().map_or("<s>", |t| &t.text);
    let right = post_tokens.first().map_or(".", |t| &t.text);
    // Only a literal `.` is auto-safe, matching the reference's `R ==
    // "."` exactly — `!`/`?` must clear the bigram check like any other
    // right token below.
    let right_is_terminal = post_tokens
        .first()
        .is_none_or(|t| t.kind == AnalysisTokenKind::Punctuation && t.text.as_ref() == ".");

    let seam_ok =
        sentence_initial || right_is_terminal || attestation.bigram().attests(left, right);
    if !seam_ok {
        return DeletionGateOutcome::SeamNotAttested;
    }

    // Excised-candidate text for tagging: pre and post, trimmed and
    // joined by one space (mirrors the reference's `pre.rstrip() + " "
    // + post.lstrip()`), so `lo`'s boundary offset is `pre_trimmed.len()`.
    let pre_trimmed = pre.trim_end();
    let post_trimmed = post.trim_start();
    let ct_text = match (pre_trimmed.is_empty(), post_trimmed.is_empty()) {
        (true, true) => String::new(),
        (true, false) => post_trimmed.to_string(),
        (false, true) => pre_trimmed.to_string(),
        (false, false) => format!("{pre_trimmed} {post_trimmed}"),
    };

    let ct_tags = tag(tagger, &ct_text);

    if !clause_gate(&ct_tags, &ct_text, original_clause_ok) {
        return DeletionGateOutcome::ClauseIncomplete;
    }

    let boundary = pre_trimmed.len();
    let lo = ct_tags
        .iter()
        .filter(|t| t.token.range.end <= boundary)
        .count();
    let hi = lo + 1;

    let mut coarse: Vec<String> = Vec::with_capacity(ct_tags.len() + 2);
    coarse.push("<S>".to_string());
    for t in &ct_tags {
        coarse.push(coarse_tag(&t.pos).to_string());
    }
    coarse.push("<E>".to_string());
    let coarse_refs: Vec<&str> = coarse.iter().map(String::as_str).collect();

    // Deliberately unshifted: `ref_engine.py::skeleton_ok` computes
    // `lo`/`hi` as indices into the UNWRAPPED token list, then reuses
    // them unshifted as indices into the `<S>`-prefixed WRAPPED sequence
    // — never correcting for the sentinel's leading slot. Transcribed
    // verbatim, not "fixed", because the accept fixtures were validated
    // against this exact window placement.
    if attestation.skeleton().window_attested(&coarse_refs, lo, hi) {
        DeletionGateOutcome::Allowed
    } else {
        DeletionGateOutcome::SkeletonNotAttested
    }
}

/// Lowercased surface text of every token in `tags` (a whole, untouched
/// original sentence) that carries a finite-verb tag.
///
/// Used by the paired-substitution clause gate as a fallback when
/// re-tagging the substituted candidate fails to find a finite verb: the
/// tagger is measurably less reliable right after a word swap (a fresh,
/// out-of-training-distribution context) than on the natural sentence it
/// trained on. Instead of trusting that re-tag, this checks whether the
/// match consumed one of the *original* sentence's finite-verb words —
/// if not, the original verb is untouched and completeness holds
/// trivially; if so, the pack's replacement was curated to still
/// function as a verb there (`substitution_pairs` closure-checks this at
/// build time), so consuming the original verb is the signal trusted.
#[must_use]
pub fn original_finite_verb_words(tags: &[TaggedToken], text: &str) -> BTreeSet<Box<str>> {
    tags.iter()
        .filter(|t| FINITE_VERB_TAGS.contains(&t.pos.as_str()))
        .map(|t| text[t.token.range.clone()].to_lowercase().into_boxed_str())
        .collect()
}

/// The paired-substitution clause gate.
///
/// `clause_ok` over the re-tagged candidate when that succeeds, falling
/// back to `original_verb_words` survival otherwise (see
/// [`original_finite_verb_words`]). Always passes when the original had
/// no clause to preserve.
#[must_use]
pub fn substitution_clause_gate(
    matched_text: &str,
    candidate_text: &str,
    candidate_tags: &[TaggedToken],
    original_clause_ok: bool,
    original_verb_words: &BTreeSet<Box<str>>,
) -> bool {
    if !original_clause_ok {
        return true;
    }
    if clause_ok(candidate_tags, candidate_text) {
        return true;
    }
    let matched_tokens = tokenize_str(matched_text, 0);
    matched_tokens
        .iter()
        .any(|t| original_verb_words.contains(t.text.as_ref()))
}

/// Capitalizes `text`'s first character, leaving the rest unchanged.
#[must_use]
pub fn capitalize(text: &str) -> String {
    let mut chars = text.chars();
    chars.next().map_or_else(String::new, |first| {
        let mut out: String = first.to_uppercase().collect();
        out.push_str(chars.as_str());
        out
    })
}

/// `true` if `range` falls inside (or crosses the boundary of) a
/// double-quoted span of `text`.
///
/// Quoted text is a mention — an example, a citation, someone else's
/// words — not the author's own register, so no operation may rewrite it.
///
/// Straight `"` marks are tracked by parity: an odd count before
/// `range.start` means the range starts inside a quotation. Curly
/// `\u{201c}`/`\u{201d}` marks are directional, tracked by nesting depth
/// instead. A matched slice containing any double-quote character
/// crosses a boundary and counts as quoted too — conservative in the
/// direction a hold should be.
#[must_use]
pub fn in_quoted_span(text: &str, range: &std::ops::Range<usize>) -> bool {
    let is_quote = |c: char| c == '"' || c == '\u{201c}' || c == '\u{201d}';
    if text
        .get(range.clone())
        .is_some_and(|slice| slice.chars().any(is_quote))
    {
        return true;
    }
    let Some(prefix) = text.get(..range.start) else {
        // A range that does not fall on character boundaries never came
        // from a scan of `text`; refuse to certify it as unquoted.
        return true;
    };
    let mut straight = 0usize;
    let mut curly_depth = 0isize;
    for c in prefix.chars() {
        match c {
            '"' => straight += 1,
            '\u{201c}' => curly_depth += 1,
            '\u{201d}' => curly_depth -= 1,
            _ => {}
        }
    }
    straight % 2 == 1 || curly_depth > 0
}
