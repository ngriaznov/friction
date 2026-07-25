//! Shared gate helpers: the clause gate, and the gated-span-deletion seam
//! + skeleton gates (ALGORITHMS.md §4.2 / §4.5).
//!
//! Every gate here operates on real, case-preserved text tagged directly
//! by the shipped tagger — the same convention `corpus-tool attest` used
//! to mine the [`friction_packs::AttestationPack`] this module queries
//! (see that module's own doc comment on tokenization/tagging
//! conventions), rather than reconstructing a lowercased, pre-tokenized
//! word list the way the Python reference tags its own candidate spans.
//! Tagging real text keeps the tag distribution this module measures
//! consistent with the tag distribution the pack itself was built from.

use std::collections::BTreeSet;
use std::ops::Range;

use friction_match::token::{AnalysisTokenKind, tokenize_str};
use friction_nlp::{
    FINITE_VERB_TAGS, TaggedToken, Tagger, coarse_tag, has_finite_verb, is_imperative_initial,
};
use friction_packs::AttestationPack;

/// `true` if `tokens` (tagged from `text`) satisfies the clause-
/// completeness test on its own: a finite verb somewhere, or an
/// imperative-initial `VB`.
///
/// Falls back to [`has_ambiguous_s_verb`] alongside the strict Penn-tag
/// check — a measured compensation for the shipped tagger's own
/// weakness on this exact ambiguity class, not a loosening of what
/// counts as a clause; see that function's own docs.
#[must_use]
pub fn clause_ok(tokens: &[TaggedToken], text: &str) -> bool {
    has_finite_verb(tokens) || is_imperative_initial(tokens) || has_ambiguous_s_verb(tokens, text)
}

/// A narrow, targeted compensation for a measured weakness in the shipped
/// tagger.
///
/// An `-s`-suffixed word immediately followed by an infinitival
/// `to VB` complement (`"needs to be placed"`, `"requires to run"`-shaped)
/// is unambiguous evidence of a finite verb in that slot regardless of
/// what tag the token itself carries — English's `-s` suffix is
/// genuinely ambiguous between a plural noun and a third-person-singular
/// verb, and this tagger sometimes resolves that ambiguity the wrong way
/// specifically in this shape (measured directly against the accept
/// fixtures: `"the configuration file needs to be placed"` tags `needs`
/// as `RB`, not `VBZ`, in every context tried, isolated or not).
///
/// Deliberately narrow to avoid the much larger false-positive class a
/// bare "ends in s, tagged NN/NNS/RB" check would open (an ordinary noun
/// phrase like `"a place to go"` or `"nothing to lose"` never satisfies
/// this — `place`/`nothing` don't end in `s` — so this only ever fires on
/// the specific morphological-ambiguity shape it targets).
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
/// one, per ALGORITHMS.md §4.5.
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

/// Checks every gate ALGORITHMS.md §4.2/§4.5 requires for deleting
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
    // "."` exactly — `!` and `?` are not sentence-final free passes and
    // must clear the bigram check like any other right token below.
    let right_is_terminal = post_tokens
        .first()
        .is_none_or(|t| t.kind == AnalysisTokenKind::Punctuation && t.text.as_ref() == ".");

    let seam_ok =
        sentence_initial || right_is_terminal || attestation.bigram().attests(left, right);
    if !seam_ok {
        return DeletionGateOutcome::SeamNotAttested;
    }

    // Build the excised-candidate text for tagging: pre and post,
    // trimmed and joined by exactly one space (mirrors the reference's
    // `pre.rstrip() + " " + post.lstrip()`), so the boundary byte offset
    // used to compute `lo` is exactly `pre_trimmed.len()`.
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

    // Deliberately unshifted: `ref_engine.py::skeleton_ok` computes `lo`/
    // `hi` as indices into the UNWRAPPED excised token list (`ct`), then
    // uses those same, unshifted values directly as indices into the
    // `<S>`-prefixed WRAPPED tag sequence — i.e. it never corrects for the
    // sentinel's own extra leading slot. This is transcribed verbatim
    // (not "fixed" to shift by one) because the target is `ref_engine.py`'s
    // exact byte-level behavior, and the accept fixtures were validated
    // against this exact window placement.
    if attestation.skeleton().window_attested(&coarse_refs, lo, hi) {
        DeletionGateOutcome::Allowed
    } else {
        DeletionGateOutcome::SkeletonNotAttested
    }
}

/// Lowercased surface text of every token in `tags` (a whole, untouched
/// original sentence's tagged tokens) that carries a finite-verb tag.
///
/// Used by the paired-substitution clause gate as a fallback when
/// re-tagging the freshly-substituted candidate sentence fails to find a
/// finite verb: the shipped tagger is measurably less reliable on a
/// sentence immediately after a word has been swapped for a different
/// one of the same part of speech (a fresh, out-of-training-distribution
/// local context) than it is on the original, natural sentence it was
/// trained on the shape of. Rather than trust that single re-tag, this
/// checks whether the substitution's own matched span consumed one of the
/// *original* sentence's finite-verb words in the first place — if it
/// did not, the original verb is untouched and clause completeness holds
/// trivially regardless of how the rewritten sentence retags; if it did,
/// the pack's replacement text was curated specifically to still function
/// as a verb in that slot (every `substitution_pairs` entry that touches
/// a verb replaces it with another verb-shaped word, checked for closure
/// at pack build time), so consuming the original verb is itself the
/// signal this fallback trusts.
#[must_use]
pub fn original_finite_verb_words(tags: &[TaggedToken], text: &str) -> BTreeSet<Box<str>> {
    tags.iter()
        .filter(|t| FINITE_VERB_TAGS.contains(&t.pos.as_str()))
        .map(|t| text[t.token.range.clone()].to_lowercase().into_boxed_str())
        .collect()
}

/// The paired-substitution clause gate.
///
/// `clause_ok` over the freshly re-tagged candidate sentence when that
/// succeeds, falling back to `original_verb_words` survival when it does
/// not (see [`original_finite_verb_words`]'s own docs for why). Always
/// passes when the original sentence had no clause to preserve.
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
/// Quoted text is a mention (an example, a citation, a string someone
/// else wrote), not the author's own register, so no operation may
/// rewrite it.
///
/// Straight `"` marks are tracked by parity: an odd count before
/// `range.start` means the range starts inside a quotation. Curly
/// `\u{201c}`/`\u{201d}` marks are directional, so they are tracked by
/// nesting depth instead. A matched slice that itself contains any
/// double-quote character crosses a quotation boundary and is treated as
/// quoted too — conservative in exactly the direction a hold should be.
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
