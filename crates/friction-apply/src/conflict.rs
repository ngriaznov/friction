//! Per-round patch conflict resolution and atomic application.

use friction_core::{Patch, span};
use friction_rules::RuleFamily;

/// A patch proposed by one rule this round, tagged with that rule's family
/// for conflict-resolution priority.
///
/// Public so a caller assembling its own round pipeline (rather than going
/// through [`crate::run_fixpoint`]) can drive [`resolve_round`] directly.
#[derive(Debug, Clone)]
pub struct Candidate {
    /// The proposed patch.
    pub patch: Patch,
    /// The family of the rule that produced it, used only for
    /// conflict-resolution priority (see [`RuleFamily::priority`]).
    pub family: RuleFamily,
}

/// Resolves one round's candidate patches against `source` into the
/// accepted, non-overlapping subset that will actually be applied.
///
/// Three steps, in order:
///
/// 1. **Validate.** Every candidate's patch range is checked against
///    `source` (in bounds, on a UTF-8 character boundary) via
///    [`Patch::validate`]. An invalid patch is rejected outright — dropped,
///    never panicked on — since it cannot be safely applied to `source` at
///    all, and can't meaningfully be compared for overlap either.
/// 2. **Sort.** The remaining, valid candidates are sorted by `(range.start
///    ascending, range length descending, family priority ascending, rule
///    id ascending, replacement text ascending)`. The first three keys are
///    "leftmost-longest, then rule priority" from the project's
///    conflict-resolution rule; the last two exist only to make the sort a
///    total order so the result never depends on the caller's original
///    `Vec` order, which is required for determinism when two candidates
///    are otherwise indistinguishable (extremely unlikely in practice, but
///    the tie-break exists precisely so that case can't silently produce
///    two different outputs on two different runs).
/// 3. **Greedy accept.** Walking the sorted list, a candidate is accepted
///    if its range does not overlap any already-accepted range, and
///    dropped otherwise. Because of the sort order, this is exactly
///    "leftmost patch wins; among same-start patches, the longest wins;
///    among same-start-and-length patches, higher family priority wins."
///
/// Returns the accepted patches (in the same sorted order — *not* yet
/// resorted into application order; see [`apply_patches`]) and a count of
/// how many candidates were dropped, for either reason above.
#[must_use]
pub fn resolve_round(source: &str, candidates: Vec<Candidate>) -> (Vec<Patch>, usize) {
    let mut dropped = 0usize;
    let mut valid: Vec<Candidate> = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        if candidate.patch.validate(source).is_ok() {
            valid.push(candidate);
        } else {
            dropped += 1;
        }
    }

    valid.sort_by(|a, b| {
        a.patch
            .range
            .start
            .cmp(&b.patch.range.start)
            .then_with(|| patch_len(&b.patch).cmp(&patch_len(&a.patch)))
            .then_with(|| a.family.priority().cmp(&b.family.priority()))
            .then_with(|| a.patch.rule.as_str().cmp(b.patch.rule.as_str()))
            .then_with(|| a.patch.replacement.cmp(&b.patch.replacement))
    });

    let mut accepted: Vec<Patch> = Vec::with_capacity(valid.len());
    for candidate in valid {
        let overlaps_accepted = accepted.iter().any(|accepted_patch| {
            span::ranges_overlap(&accepted_patch.range, &candidate.patch.range)
        });
        if overlaps_accepted {
            dropped += 1;
        } else {
            accepted.push(candidate.patch);
        }
    }

    (accepted, dropped)
}

/// A patch's replaced-range length in bytes.
const fn patch_len(patch: &Patch) -> usize {
    patch.range.end - patch.range.start
}

/// Applies `patches` to `source` in one atomic pass, returning the result.
///
/// `patches` must be pairwise non-overlapping (as guaranteed by
/// [`resolve_round`]'s accepted output) and every range must already be
/// valid for `source` (also guaranteed by [`resolve_round`]). Patches are
/// applied right-to-left (highest `range.start` first): because no two
/// ranges overlap, replacing a later range never shifts the byte offsets
/// of an earlier one still waiting to be applied, so every patch's range
/// stays valid against the *original* `source` offsets throughout the
/// pass — this is what "atomic" means here, not a transaction log, just an
/// application order that needs no offset bookkeeping at all.
///
/// Applying an empty `patches` slice returns `source` unchanged,
/// byte-for-byte.
#[must_use]
pub fn apply_patches(source: &str, patches: &[Patch]) -> String {
    let mut ordered: Vec<&Patch> = patches.iter().collect();
    ordered.sort_by_key(|patch| std::cmp::Reverse(patch.range.start));

    let mut result = source.to_string();
    for patch in ordered {
        result.replace_range(patch.range.clone(), patch.replacement.as_str());
    }
    result
}

#[cfg(test)]
mod tests {
    use friction_core::{RuleId, Tier};

    use super::*;

    fn patch(range: std::ops::Range<usize>, replacement: &str, rule: &'static str) -> Patch {
        Patch::new(range, replacement, RuleId::new(rule), Tier::Fix)
    }

    fn candidate(patch: Patch, family: RuleFamily) -> Candidate {
        Candidate { patch, family }
    }

    /// Disjoint candidates are all accepted, none dropped.
    #[test]
    fn resolve_round_accepts_disjoint_candidates() {
        let source = "The quick brown fox jumps.";
        let candidates = vec![
            candidate(patch(0..3, "A", "lexical.a"), RuleFamily::Lexical),
            candidate(patch(10..15, "B", "lexical.b"), RuleFamily::Lexical),
        ];
        let (accepted, dropped) = resolve_round(source, candidates);
        assert_eq!(accepted.len(), 2);
        assert_eq!(dropped, 0);
    }

    /// Of two overlapping candidates at the same start, the longer one
    /// wins ("leftmost-longest").
    #[test]
    fn resolve_round_prefers_longer_patch_at_same_start() {
        let source = "leveraging the pipeline";
        let candidates = vec![
            candidate(
                patch(0..10, "using", "lexical.leverage"),
                RuleFamily::Lexical,
            ),
            candidate(patch(0..3, "u", "lexical.stub"), RuleFamily::Lexical),
        ];
        let (accepted, dropped) = resolve_round(source, candidates);
        assert_eq!(accepted.len(), 1);
        assert_eq!(accepted[0].replacement, "using");
        assert_eq!(dropped, 1);
    }

    /// Of two overlapping candidates that start and end at the same
    /// point, the higher-priority family (lower `priority()`) wins —
    /// Structural over Lexical here.
    #[test]
    fn resolve_round_prefers_higher_priority_family_on_full_tie() {
        let source = "It leverages the pipeline heavily.";
        let candidates = vec![
            candidate(
                patch(3..13, "uses", "lexical.leverage"),
                RuleFamily::Lexical,
            ),
            candidate(
                patch(3..13, "relies on", "structural.rewrite"),
                RuleFamily::Structural,
            ),
        ];
        let (accepted, dropped) = resolve_round(source, candidates);
        assert_eq!(accepted.len(), 1);
        assert_eq!(accepted[0].replacement, "relies on");
        assert_eq!(dropped, 1);
    }

    /// An out-of-bounds candidate is dropped, not applied and not
    /// panicked on, and does not prevent an otherwise-valid candidate from
    /// being accepted.
    #[test]
    fn resolve_round_drops_invalid_range_without_panicking() {
        let source = "short";
        let candidates = vec![
            candidate(patch(0..999, "x", "lexical.bad"), RuleFamily::Lexical),
            candidate(patch(0..5, "SHORT", "lexical.ok"), RuleFamily::Lexical),
        ];
        let (accepted, dropped) = resolve_round(source, candidates);
        assert_eq!(accepted.len(), 1);
        assert_eq!(accepted[0].replacement, "SHORT");
        assert_eq!(dropped, 1);
    }

    /// A patch range that splits a UTF-8 character is dropped, not
    /// panicked on.
    #[test]
    fn resolve_round_drops_non_char_boundary_range() {
        let source = "café bar";
        // 'é' is a 2-byte UTF-8 character occupying bytes 3..5; byte 4
        // sits inside it, not on a character boundary.
        let candidates = vec![candidate(
            patch(0..4, "x", "lexical.bad"),
            RuleFamily::Lexical,
        )];
        let (accepted, dropped) = resolve_round(source, candidates);
        assert!(accepted.is_empty());
        assert_eq!(dropped, 1);
    }

    /// `resolve_round`'s output never overlaps, checked with
    /// `friction_core::find_overlaps` directly (the same detector the core
    /// crate ships), across a mix of overlapping and disjoint candidates.
    #[test]
    fn resolve_round_output_never_overlaps() {
        let source = "aaaaaaaaaaaaaaaaaaaa";
        let candidates = vec![
            candidate(patch(0..5, "A", "lexical.a"), RuleFamily::Lexical),
            candidate(patch(2..8, "B", "lexical.b"), RuleFamily::Lexical),
            candidate(patch(5..10, "C", "lexical.c"), RuleFamily::Lexical),
            candidate(patch(12..15, "D", "lexical.d"), RuleFamily::Lexical),
        ];
        let (accepted, _dropped) = resolve_round(source, candidates);
        assert!(friction_core::find_overlaps(&accepted).is_empty());
    }

    /// Resolving an empty candidate list yields no accepted patches and no
    /// drops.
    #[test]
    fn resolve_round_handles_empty_input() {
        let (accepted, dropped) = resolve_round("anything", Vec::new());
        assert!(accepted.is_empty());
        assert_eq!(dropped, 0);
    }

    /// `apply_patches` with no patches returns the input unchanged,
    /// byte-for-byte.
    #[test]
    fn apply_patches_with_no_patches_is_unchanged() {
        let source = "Hello, world! Café.";
        assert_eq!(apply_patches(source, &[]), source);
    }

    /// A single deletion patch removes exactly its range.
    #[test]
    fn apply_patches_applies_single_deletion() {
        let source = "It is worth noting that this works.";
        let deletion = patch(0..24, "", "lexical.filler");
        assert_eq!(apply_patches(source, &[deletion]), "this works.");
    }

    /// Multiple non-overlapping patches all apply correctly in one pass,
    /// regardless of the order they're given in — right-to-left internal
    /// ordering keeps every range's offsets valid against the original
    /// source.
    #[test]
    fn apply_patches_applies_multiple_patches_regardless_of_input_order() {
        let source = "It leverages a robust pipeline to facilitate delivery.";
        // "leverages" -> "uses", "robust" deleted (+ trailing space),
        // "facilitate" -> "enable".
        let a = patch(3..12, "uses", "lexical.leverage");
        let b = patch(15..22, "", "lexical.robust");
        let c = patch(34..44, "enable", "lexical.facilitate");
        let expected = "It uses a pipeline to enable delivery.";

        assert_eq!(
            apply_patches(source, &[a.clone(), b.clone(), c.clone()]),
            expected
        );
        // Same patches, reversed input order: identical result.
        assert_eq!(apply_patches(source, &[c, b, a]), expected);
    }

    /// A pure insertion patch (empty range) inserts without consuming any
    /// source text.
    #[test]
    fn apply_patches_applies_pure_insertion() {
        let source = "do not do that";
        let insertion = patch(0..0, "", "noop"); // sanity: empty replacement, empty range = no-op
        assert_eq!(apply_patches(source, &[insertion]), source);

        let real_insertion = Patch::new(2..2, "n't", RuleId::new("contraction.insert"), Tier::Fix);
        assert_eq!(apply_patches("do not", &[real_insertion]), "don't not");
    }
}
