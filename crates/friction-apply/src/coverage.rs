//! [`touched_original_ranges`]: mapping a (possibly multi-round) fix back
//! to which spans of the *original* input it ever touched.
//!
//! Every [`RoundReport::applied_patches`] range indexes that round's own
//! source text, not the original document past round 1 — round 2's
//! patches sit in coordinates shifted by whatever round 1 already
//! changed. Reporting tools that want to know "which original sentences
//! did the engine touch" (e.g. the human near-no-op report) need those
//! per-round ranges translated back into round-1 (original) coordinates.
//! This module does exactly that translation, using only the patches the
//! driver already recorded — no approximate text diffing involved, since
//! every edit and its exact position are already known precisely.

use std::ops::Range;

use friction_core::Patch;

use crate::driver::RoundReport;

/// One contiguous run of a round's *current* source text that either
/// traces back, byte-for-byte, to an identical-length span of the
/// original document (`original_start: Some(_)`), or was introduced by an
/// earlier round's patch and has no such correspondence (`None` — e.g. a
/// substitution's replacement text).
#[derive(Debug, Clone)]
struct Segment {
    current: Range<usize>,
    original_start: Option<usize>,
}

/// Every byte range of the *original* input that a fixpoint run touched.
///
/// "Touched" means fell within at least one applied patch's replaced
/// span, across every round — i.e. exactly the text a rule actually
/// touched, expressed once in original-document coordinates regardless of
/// how many rounds it took to converge.
///
/// `original_len` must be the byte length of the original `source` (round
/// 1's source); `rounds` must be the `FixpointReport::rounds` produced by
/// running the fixpoint driver over that same `source`, in order.
///
/// Returned ranges are sorted by `start` ascending and may overlap or be
/// adjacent (never merged) — callers checking "does sentence X overlap
/// any touched range" don't need them coalesced.
#[must_use]
pub fn touched_original_ranges(original_len: usize, rounds: &[RoundReport]) -> Vec<Range<usize>> {
    let mut segments = vec![Segment {
        current: 0..original_len,
        original_start: Some(0),
    }];
    let mut touched: Vec<Range<usize>> = Vec::new();

    for round in rounds {
        if round.applied_patches.is_empty() {
            continue;
        }
        segments = apply_round(&segments, &round.applied_patches, &mut touched);
    }

    touched.sort_by_key(|r| r.start);
    touched
}

/// Applies one round's non-overlapping `patches` to `segments`, recording
/// every touched original-coordinate sub-range into `touched` and
/// returning the updated segment list for the *next* round's source.
fn apply_round(
    segments: &[Segment],
    patches: &[Patch],
    touched: &mut Vec<Range<usize>>,
) -> Vec<Segment> {
    let mut sorted_patches: Vec<&Patch> = patches.iter().collect();
    sorted_patches.sort_by_key(|p| p.range.start);

    let cur_len = segments.last().map_or(0, |s| s.current.end);

    // Every position where either a segment or a patch starts or ends is
    // a cut point; walking consecutive cut points gives sub-intervals
    // that are each entirely inside exactly one segment and either fully
    // inside one patch or fully outside every patch.
    let mut cuts: Vec<usize> = vec![0, cur_len];
    for patch in &sorted_patches {
        cuts.push(patch.range.start);
        cuts.push(patch.range.end);
    }
    for seg in segments {
        cuts.push(seg.current.start);
        cuts.push(seg.current.end);
    }
    cuts.sort_unstable();
    cuts.dedup();

    let mut new_segments: Vec<Segment> = Vec::new();
    let mut delta: i64 = 0;
    let mut seg_idx = 0usize;
    let mut patch_idx = 0usize;

    for window in cuts.windows(2) {
        let (a, b) = (window[0], window[1]);
        if a >= b {
            continue;
        }

        while seg_idx + 1 < segments.len() && segments[seg_idx].current.end <= a {
            seg_idx += 1;
        }
        let seg = &segments[seg_idx];
        let seg_original_at_a = seg.original_start.map(|os| os + (a - seg.current.start));

        while patch_idx < sorted_patches.len() && sorted_patches[patch_idx].range.end <= a {
            patch_idx += 1;
        }
        let in_patch = patch_idx < sorted_patches.len()
            && sorted_patches[patch_idx].range.start <= a
            && b <= sorted_patches[patch_idx].range.end;

        if in_patch {
            let patch = sorted_patches[patch_idx];
            if let Some(orig_start) = seg_original_at_a {
                touched.push(orig_start..(orig_start + (b - a)));
            }
            // Emit the patch's replacement as one new segment, exactly
            // once, the first sub-interval we see for it (its own start).
            if a == patch.range.start {
                let new_start = (i64::try_from(a).unwrap_or(i64::MAX) + delta)
                    .try_into()
                    .unwrap_or(0);
                let replacement_len = patch.replacement.len();
                new_segments.push(Segment {
                    current: new_start..(new_start + replacement_len),
                    original_start: None,
                });
                let patch_len = patch.range.end - patch.range.start;
                delta += i64::try_from(replacement_len).unwrap_or(i64::MAX)
                    - i64::try_from(patch_len).unwrap_or(i64::MAX);
            }
        } else {
            let new_start = (i64::try_from(a).unwrap_or(i64::MAX) + delta)
                .try_into()
                .unwrap_or(0);
            new_segments.push(Segment {
                current: new_start..(new_start + (b - a)),
                original_start: seg_original_at_a,
            });
        }
    }

    new_segments
}

#[cfg(test)]
mod tests {
    use friction_core::{RuleId, Tier};

    use super::*;

    fn patch(range: Range<usize>, replacement: &str) -> Patch {
        Patch::new(range, replacement, RuleId::new("test.rule"), Tier::Fix)
    }

    fn round(applied_patches: Vec<Patch>) -> RoundReport {
        let patches_applied = applied_patches.len();
        RoundReport {
            round: 1,
            rules_fired: Vec::new(),
            findings: Vec::new(),
            patches_applied,
            patches_dropped: 0,
            applied_patches,
        }
    }

    /// No rounds, or rounds with no applied patches, touch nothing.
    #[test]
    fn no_patches_touches_nothing() {
        assert!(touched_original_ranges(20, &[]).is_empty());
        assert!(touched_original_ranges(20, &[round(vec![])]).is_empty());
    }

    /// A single round, single deletion patch touches exactly its own
    /// range (hand-computed: source "leverages the pipeline heavily",
    /// deleting "leverages " at 0..10 touches original bytes 0..10).
    #[test]
    fn single_round_single_patch_touches_its_own_range() {
        let rounds = vec![round(vec![patch(0..10, "")])];
        assert_eq!(touched_original_ranges(30, &rounds), vec![0..10]);
    }

    /// Two disjoint patches in one round each contribute their own
    /// original-coordinate range, sorted by start.
    #[test]
    fn two_disjoint_patches_in_one_round() {
        let rounds = vec![round(vec![patch(20..25, ""), patch(0..5, "")])];
        assert_eq!(touched_original_ranges(30, &rounds), vec![0..5, 20..25]);
    }

    /// The classic multi-round case: round 1 shortens the text before a
    /// second sentence, shifting its byte position; round 2's patch,
    /// expressed in round-2 (post-round-1) coordinates, must still map
    /// back to the *original* position of that second sentence.
    ///
    /// Hand-computed: original = "It is worth noting that this works. It
    /// leverages a pipeline.".
    ///
    /// Round 1 deletes "It is worth noting that " (bytes 0..24) — the
    /// filler phrase rule's own signature move — shrinking the text by 24
    /// bytes, so "leverages" (originally at bytes 39..48) now sits at
    /// round-2 position `39 - 24 = 15`..`48 - 24 = 24`.
    ///
    /// Round 2 replaces round-2-coordinates `15..24` ("leverages") with
    /// "uses" — this must map back to *original* `39..48`.
    #[test]
    fn multi_round_patch_maps_back_through_an_earlier_rounds_shift() {
        let original = "It is worth noting that this works. It leverages a pipeline.";
        assert_eq!(&original[0..24], "It is worth noting that ");
        assert_eq!(&original[39..48], "leverages");

        let round1 = round(vec![patch(0..24, "")]);
        let round2 = round(vec![patch(15..24, "uses")]);

        let touched = touched_original_ranges(original.len(), &[round1, round2]);
        assert_eq!(touched, vec![0..24, 39..48]);
    }

    /// A patch's replacement text (no original correspondence) getting
    /// further edited in a later round contributes nothing new — the
    /// original span it stands in for was already recorded touched when
    /// the first patch fired.
    #[test]
    fn editing_a_previous_patchs_replacement_adds_no_new_original_range() {
        // Round 1: replace original bytes 5..15 ("leveraging") with
        // "using" (5 bytes) at round-2 position 5..10.
        let round1 = round(vec![patch(5..15, "using")]);
        // Round 2: further edits that same replacement text in place
        // (contrived, but exercises the `original_start: None` path).
        let round2 = round(vec![patch(5..10, "USING")]);

        let touched = touched_original_ranges(30, &[round1, round2]);
        // Only the round-1 patch's original span is recorded; round 2's
        // patch sits entirely inside replacement text with no original
        // mapping, so it adds nothing.
        assert_eq!(touched, vec![5..15]);
    }

    /// A patch that entirely replaces the whole original document still
    /// maps back correctly (boundary case: `a == 0`, `b == original_len`).
    #[test]
    fn patch_spanning_the_whole_document() {
        let rounds = vec![round(vec![patch(0..10, "x")])];
        assert_eq!(touched_original_ranges(10, &rounds), vec![0..10]);
    }
}
