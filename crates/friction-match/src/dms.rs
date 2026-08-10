//! Differential matching statistics: span extraction and the document-
//! level report.
//!
//! The span-extraction algorithm and its scoring are specified in
//! `docs/research/ALGORITHMS.md` §1.2-1.3. One difference from a naive
//! whole-file walk: a raw byte stream has no prose/non-prose distinction,
//! so a single automaton walk over it would never need to reset
//! mid-document. Once non-prose tokens are dropped (this crate's
//! prose-only guarantee), "the token stream" is no longer contiguous in
//! the source, so [`scan_units`]/[`document_report`] reset the walk (state
//! `v=0, l=0`) and run independently for each in-scope [`ScopedUnit`]: a
//! span's left-extension or continuation run can never bridge across a
//! heading/table gap into a byte range that slices through excluded
//! content.
//!
//! # One pooled machine automaton, not five per-family ones
//!
//! Family attribution is retired: the product question this channel
//! answers is only "does this read machine-generated", never "which
//! generator wrote it". Both [`scan_units`] and [`document_report`] walk
//! [`friction_packs::DmsIndexView::pooled_machine_sam`] — every family
//! stream `corpus-tool dms-pack` defines, concatenated in a fixed
//! alphabetical order with the document separator between streams (see
//! that method's own docs) — instead of fanning out over
//! [`ModelFamily::ALL`] and reporting the best/every family's walk
//! separately. The threshold/score logic in [`spans_from_curve`] is
//! unchanged; only the automaton it is applied to changed. Because a
//! pooled walk's matching-statistics curve at any position is the max
//! over what any single family's walk would have produced there (the
//! pooled automaton recognizes every substring any single family's
//! automaton does, plus none that crosses a separator), an unchanged
//! threshold applied to the pooled curve can only flag a superset of
//! what scanning every family separately and unioning the results would
//! have flagged — the intended semantics this round; recalibrating the
//! thresholds for the pooled distribution is explicitly out of scope.
//! `ModelFamily`/`family_sam` are kept for compatibility (`corpus-tool
//! dms-pack` still serializes every per-family section), but no longer
//! change what a scan reports; the CLI's own `--family` flag was retired
//! for the same reason.

use friction_packs::{DmsIndexView, SamView, VocabView};
// Only the native (non-wasm32) path in `document_report` below uses
// `.par_iter()` — see that function's own wasm32 fallback comment.
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use crate::span::{Channel, DmsMachineReport, DmsReport, MatchScore, MatchSpan};
use crate::token::ScopedUnit;

/// The constant frame id every [`Channel::Dms`] span carries. Family
/// attribution is retired — the product question is only "does this read
/// machine-generated", scanned against the one pooled machine-side
/// automaton [`friction_packs::DmsIndex::pooled_machine_sam`] builds (see
/// that method's own docs for why an unchanged threshold applied to a
/// pooled max-over-families walk can only flag a superset of what any
/// single family's walk flagged, the intended semantics this round).
const MACHINE_FRAME_ID: &str = "dms.machine";

/// The validated thresholds: a span starts where
/// `d[i] >= 3`, continues while `d[j] >= 2`, and requires length `>= 2`
/// tokens.
const START_THRESH: i64 = 3;
const CONT_THRESH: i64 = 2;
const MIN_LEN: usize = 2;

/// One extracted run in TOKEN-INDEX space (unit-local), before byte
/// translation: the raw algorithmic result, hand-verifiable against
/// the worked examples in this module's tests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DmsRun {
    pub score: i64,
    pub left: usize,
    pub end: usize,
}

/// Extracts differential runs from a pair of matching-statistics
/// curves: `d[i] = mM[i] - mH[i]`; a run starts where `d[i] >=
/// start_thresh`, continues while `d[j] >= cont_thresh`, requires
/// length `>= min_len`, is extended left by `mM[i]-1` tokens (clamped
/// to 0) to cover the matched phrase, and is scored by `sum(d[i..j])` —
/// the CONTINUE region only, NOT including the left-extension prefix.
/// Runs are returned sorted descending over `(score, left, end)`
/// tuples, so ties break by `left` then `end`, descending.
/// `m`/`h`/`d`/`i`/`j` mirror the algorithm write-up's own variable
/// names (docs/research/ALGORITHMS.md §1.2): kept identical so this
/// function stays hand-verifiable against its worked examples.
#[allow(clippy::many_single_char_names)]
pub fn spans_from_curve(
    m: &[u32],
    h: &[u32],
    start_thresh: i64,
    cont_thresh: i64,
    min_len: usize,
) -> Vec<DmsRun> {
    let d: Vec<i64> = m
        .iter()
        .zip(h)
        .map(|(&a, &b)| i64::from(a) - i64::from(b))
        .collect();

    let mut runs = Vec::new();
    let mut i = 0usize;
    while i < d.len() {
        if d[i] >= start_thresh {
            let mut j = i;
            while j < d.len() && d[j] >= cont_thresh {
                j += 1;
            }
            if j - i >= min_len {
                let left_extension = (i64::from(m[i]) - 1).max(0);
                #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
                let left = i.saturating_sub(left_extension as usize);
                let score: i64 = d[i..j].iter().sum();
                runs.push(DmsRun {
                    score,
                    left,
                    end: j,
                });
            }
            i = j;
        } else {
            i += 1;
        }
    }

    runs.sort_by(|a, b| {
        b.score
            .cmp(&a.score)
            .then(b.left.cmp(&a.left))
            .then(b.end.cmp(&a.end))
    });
    runs
}

/// Queries `unit`'s tokens through `vocab`, producing the `Option<u32>`
/// query [`SamView::matching_stats`] expects (`None` for an
/// out-of-vocabulary token, the `-1` sentinel).
fn query_ids(unit: &ScopedUnit<'_>, vocab: VocabView<'_>) -> Vec<Option<u32>> {
    unit.tokens
        .iter()
        .map(|token| vocab.id_of(&token.text))
        .collect()
}

/// Runs [`spans_from_curve`] (thresholds 3/2/2, per the validated run)
/// independently over each in-scope unit against `machine`'s pooled
/// automaton vs. `human`, translates unit-local token indices to absolute
/// byte ranges via `unit.tokens`, and tags every span
/// `frame_id = "dms.machine"` — the same constant for every span
/// regardless of which family's evidence a run's tokens actually came
/// from (see [`MACHINE_FRAME_ID`]'s own docs).
pub fn scan_units(
    units: &[ScopedUnit<'_>],
    machine: SamView<'_>,
    human: SamView<'_>,
    vocab: VocabView<'_>,
) -> Vec<MatchSpan> {
    let mut spans = Vec::new();
    let frame_id: Box<str> = MACHINE_FRAME_ID.into();

    for unit in units {
        if unit.tokens.is_empty() {
            continue;
        }
        let query = query_ids(unit, vocab);
        let mm = machine.matching_stats(&query);
        let mh = human.matching_stats(&query);

        for run in spans_from_curve(&mm, &mh, START_THRESH, CONT_THRESH, MIN_LEN) {
            let start = unit.tokens[run.left].range.start;
            let end = unit.tokens[run.end - 1].range.end;
            spans.push(MatchSpan {
                range: start..end,
                channel: Channel::Dms,
                frame_id: frame_id.clone(),
                score: MatchScore::Differential(run.score),
                message: None,
            });
        }
    }
    spans
}

/// Document-level report: flattens every in-scope unit's
/// independently-walked pooled-machine-vs-human matching-statistics
/// arrays and reports `mean(mM) - mean(mH)`.
///
/// Every unit's query, pooled-machine curve, and human curve are
/// independent of every other unit's, so this runs as one rayon
/// `par_iter` over `units`, reducing straight to the three running sums —
/// no per-family fan-out (see [`crate::span::DmsReport`]'s own docs for
/// why: the pooled scan reports one machine-vs-human statistic, not one
/// per family).
pub fn document_report(
    units: &[ScopedUnit<'_>],
    index: &DmsIndexView<'_>,
    vocab: VocabView<'_>,
) -> DmsReport {
    let machine_sam = index.pooled_machine_sam();
    let human_sam = index.human_sam();

    // rayon cannot spawn OS threads on wasm32-unknown-unknown; the
    // sequential `.fold()` below is byte-identical to `.reduce()` here
    // (the fold's running total is exactly what `.reduce()`'s associative,
    // commutative `u64`/`usize` addition already collapses to regardless
    // of split point — see `friction-edit::parse_ctx`'s own comment for
    // the determinism proof this whole workspace's rayon rollout rests
    // on).
    #[cfg(not(target_arch = "wasm32"))]
    let (machine_total, human_total, token_count) = units
        .par_iter()
        .filter(|unit| !unit.tokens.is_empty())
        .map(|unit| {
            let query = query_ids(unit, vocab);
            let mm = machine_sam.matching_stats(&query);
            let mh = human_sam.matching_stats(&query);
            let machine: u64 = mm.iter().map(|&v| u64::from(v)).sum();
            let human: u64 = mh.iter().map(|&v| u64::from(v)).sum();
            (machine, human, query.len())
        })
        .reduce(
            || (0u64, 0u64, 0usize),
            |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2),
        );
    #[cfg(target_arch = "wasm32")]
    let (machine_total, human_total, token_count) = units
        .iter()
        .filter(|unit| !unit.tokens.is_empty())
        .map(|unit| {
            let query = query_ids(unit, vocab);
            let mm = machine_sam.matching_stats(&query);
            let mh = human_sam.matching_stats(&query);
            let machine: u64 = mm.iter().map(|&v| u64::from(v)).sum();
            let human: u64 = mh.iter().map(|&v| u64::from(v)).sum();
            (machine, human, query.len())
        })
        .fold((0u64, 0u64, 0usize), |a, b| {
            (a.0 + b.0, a.1 + b.1, a.2 + b.2)
        });

    #[allow(clippy::cast_precision_loss)]
    let (mean_machine, mean_human) = if token_count == 0 {
        (0.0, 0.0)
    } else {
        (
            machine_total as f64 / token_count as f64,
            human_total as f64 / token_count as f64,
        )
    };

    DmsReport {
        machine: DmsMachineReport {
            mean_machine,
            mean_human,
            differential: mean_machine - mean_human,
            token_count,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-picked `m`/`h` arrays exercising a start-threshold hit, a
    /// continuation, a left-extension (`m[i] > 1`), and a below-`min_len`
    /// near-miss that must NOT produce a run.
    ///
    /// `d = m - h`:
    /// index:   0  1  2  3  4  5  6  7
    /// m:       1  4  5  0  3  0  0  0
    /// h:       0  0  0  0  0  0  0  0
    /// d:       1  4  5  0  3  0  0  0
    ///
    /// - `i=0`: `d[0]=1 < 3`, no start.
    /// - `i=1`: `d[1]=4 >= 3`, starts. Continue while `d[j]>=2`: `d[1]=4`,
    ///   `d[2]=5`, `d[3]=0` stops -> `j=3`. Length `3-1=2 >= min_len`.
    ///   `left = max(0, 1 - max(m[1]-1, 0)) = max(0, 1 - 3) = 0` (clamped).
    ///   `score = sum(d[1..3]) = 4+5 = 9`.
    /// - `i=3`: `d[3]=0 < 3`, `i` advances by 1 to 4.
    /// - `i=4`: `d[4]=3 >= 3`, starts. Continue: `d[4]=3`, `d[5]=0` stops
    ///   -> `j=5`. Length `5-4=1 < min_len=2` -> NOT a run (the near-miss).
    #[test]
    fn spans_from_curve_matches_hand_worked_example() {
        let m = [1, 4, 5, 0, 3, 0, 0, 0];
        let h = [0, 0, 0, 0, 0, 0, 0, 0];
        let runs = spans_from_curve(&m, &h, 3, 2, 2);
        assert_eq!(
            runs,
            vec![DmsRun {
                score: 9,
                left: 0,
                end: 3
            }]
        );
    }

    /// A case where `m[i] - 1 > 0` so the left-extension actually moves
    /// `left` before `i`, and the score must still only cover the
    /// CONTINUE region `d[i..end]`, not the extended `d[left..end]`.
    ///
    /// m:  0 0 4 5 0
    /// h:  0 0 0 0 0
    /// d:  0 0 4 5 0
    ///
    /// `i=2`: `d[2]=4>=3`, starts. Continue: `d[2]=4`, `d[3]=5`, `d[4]=0`
    /// stops -> `j=4`. Length `4-2=2>=min_len`.
    /// `left = max(0, 2 - max(m[2]-1,0)) = max(0, 2 - 3) = 0`.
    /// `score = sum(d[2..4]) = 4+5 = 9` — NOT `sum(d[0..4])`.
    #[test]
    fn spans_from_curve_scores_only_the_continue_region_not_the_left_extension() {
        let m = [0, 0, 4, 5, 0];
        let h = [0, 0, 0, 0, 0];
        let runs = spans_from_curve(&m, &h, 3, 2, 2);
        assert_eq!(runs.len(), 1);
        let run = &runs[0];
        assert!(run.left < 2, "left extension must move left before i=2");
        assert_eq!(run.score, 9);
    }

    #[test]
    fn spans_from_curve_sorts_by_score_descending() {
        // Two independent runs, separated by a `d < start_thresh` gap;
        // the second run scores higher and must sort first.
        let m = [4, 4, 0, 6, 6, 6];
        let h = [0, 0, 0, 0, 0, 0];
        let runs = spans_from_curve(&m, &h, 3, 2, 2);
        assert_eq!(runs.len(), 2);
        assert!(runs[0].score > runs[1].score);
        assert_eq!(runs[0].score, 18);
        assert_eq!(runs[1].score, 8);
    }

    #[test]
    fn spans_from_curve_empty_curve_produces_no_runs() {
        let runs = spans_from_curve(&[], &[], 3, 2, 2);
        assert!(runs.is_empty());
    }
}
