//! The shapes every detection channel emits, and how they merge.
//!
//! A [`MatchSpan`] carries no repair action — this crate detects and
//! reports, it never rewrites (see this crate's own top-level docs). And
//! unlike [`friction_core::Patch`] conflicts (where two patches genuinely
//! can't both apply, hence [`friction_core::find_overlaps`]), two
//! overlapping `MatchSpan`s from different channels are just two
//! independent, simultaneously-true observations about the same text —
//! there's nothing to arbitrate. [`merge_spans`] reflects that: it only
//! ever drops exact-duplicate `(range, channel, frame_id)` tuples
//! (defensive — two channels should never coincide exactly, but nothing
//! stops it structurally), and keeps every other overlap.

use std::ops::Range;

use friction_core::span::Spanned;
use friction_packs::ModelFamily;

/// Which detection channel produced a [`MatchSpan`].
///
/// Declaration order is this type's `Ord` order, which is also the
/// channel tie-break [`merge_spans`] sorts by.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Channel {
    /// Differential matching statistics (§1 of this crate's algorithm
    /// reference).
    Dms,
    /// The literal automaton / regex-fallback tell inventory (§2).
    Literal,
    /// Licensed light-verb constructions.
    Lvc,
}

/// A [`MatchSpan`]'s score.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MatchScore {
    /// DMS: `sum(d)` over the differential run — transcribed verbatim
    /// from the reference `spans()` scoring, always positive.
    Differential(i64),
    /// Literal / LVC: presence is the whole signal; neither channel's
    /// reference defines a graded per-match score.
    Present,
}

/// One detection finding: a byte range into the original document
/// source, which channel found it, that channel's own frame identifier,
/// and its score.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatchSpan {
    /// Byte range into the original document source.
    pub range: Range<usize>,
    /// Which channel produced this span.
    pub channel: Channel,
    /// Namespaced per channel: `"dms.<family>"`, the inventory pack's own
    /// entry id for Literal, `"lvc.<nominalization>"` for Lvc.
    pub frame_id: Box<str>,
    /// This span's score.
    pub score: MatchScore,
}

impl Spanned for MatchSpan {
    fn range(&self) -> Range<usize> {
        self.range.clone()
    }
}

/// One family's DMS document-level statistics.
#[derive(Debug, Clone)]
pub struct DmsFamilyReport {
    /// The family this report covers.
    pub family: ModelFamily,
    /// `mean(mM)` over every in-scope token, walking `family`'s
    /// automaton.
    pub mean_machine: f64,
    /// `mean(mH)` over every in-scope token, walking the human automaton.
    pub mean_human: f64,
    /// `mean_machine - mean_human`.
    pub differential: f64,
    /// The number of in-scope (prose) tokens the means above were
    /// computed over.
    pub token_count: usize,
}

/// The document-level DMS report: one [`DmsFamilyReport`] per family the
/// loaded pack defines.
#[derive(Debug, Clone)]
pub struct DmsReport {
    /// The family [`crate::MatchEngine`] was constructed with — the one
    /// [`crate::span::MatchSpan`]s in [`Channel::Dms`] were extracted
    /// against.
    pub target_family: ModelFamily,
    /// One entry per family the pack defines, [`ModelFamily::ALL`] order.
    pub families: Vec<DmsFamilyReport>,
}

/// The full result of scanning one document.
#[derive(Debug, Clone)]
pub struct DocumentReport {
    /// Every span found by every channel, merged and sorted (see
    /// [`merge_spans`]).
    pub spans: Vec<MatchSpan>,
    /// The document-level DMS report.
    pub dms: DmsReport,
}

/// Merges spans from every channel: drops exact-duplicate
/// `(range, channel, frame_id)` tuples (defensive), keeps every other
/// overlap (spans are independent findings, not competing edits — see
/// this module's own docs), and sorts by `(range.start, range.end,
/// channel, frame_id)` for byte-identical output on identical input.
pub(crate) fn merge_spans(channels: Vec<Vec<MatchSpan>>) -> Vec<MatchSpan> {
    let mut merged: Vec<MatchSpan> = channels.into_iter().flatten().collect();
    merged.sort_by(|a, b| {
        a.range
            .start
            .cmp(&b.range.start)
            .then(a.range.end.cmp(&b.range.end))
            .then(a.channel.cmp(&b.channel))
            .then(a.frame_id.cmp(&b.frame_id))
    });
    merged
        .dedup_by(|a, b| a.range == b.range && a.channel == b.channel && a.frame_id == b.frame_id);
    merged
}

#[cfg(test)]
mod tests {
    use super::*;

    fn span(range: Range<usize>, channel: Channel, frame_id: &str) -> MatchSpan {
        MatchSpan {
            range,
            channel,
            frame_id: frame_id.into(),
            score: MatchScore::Present,
        }
    }

    #[test]
    fn merge_spans_drops_exact_duplicates_across_channel_lists() {
        let a = vec![span(0..5, Channel::Literal, "sub.prior_to")];
        let b = vec![span(0..5, Channel::Literal, "sub.prior_to")];
        let merged = merge_spans(vec![a, b]);
        assert_eq!(merged.len(), 1);
    }

    #[test]
    fn merge_spans_keeps_overlapping_spans_from_different_channels() {
        let a = vec![span(0..10, Channel::Dms, "dms.qwen")];
        let b = vec![span(2..8, Channel::Literal, "sub.prior_to")];
        let merged = merge_spans(vec![a, b]);
        assert_eq!(merged.len(), 2);
    }

    #[test]
    fn merge_spans_sorts_by_start_then_end_then_channel_then_frame_id() {
        let spans = vec![
            span(5..10, Channel::Lvc, "lvc.decision"),
            span(0..5, Channel::Literal, "sub.prior_to"),
            span(0..5, Channel::Dms, "dms.qwen"),
            span(0..3, Channel::Literal, "span.simply"),
        ];
        let merged = merge_spans(vec![spans]);
        let starts_ends: Vec<(usize, usize, Channel)> = merged
            .iter()
            .map(|s| (s.range.start, s.range.end, s.channel))
            .collect();
        assert_eq!(
            starts_ends,
            vec![
                (0, 3, Channel::Literal),
                (0, 5, Channel::Dms),
                (0, 5, Channel::Literal),
                (5, 10, Channel::Lvc),
            ]
        );
    }

    #[test]
    fn merge_spans_is_deterministic_across_independent_calls() {
        let build = || {
            vec![
                vec![span(5..10, Channel::Lvc, "lvc.decision")],
                vec![span(0..5, Channel::Dms, "dms.qwen")],
            ]
        };
        assert_eq!(merge_spans(build()), merge_spans(build()));
    }
}
