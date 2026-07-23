//! Shared ratio-threshold n-gram scoring core.
//!
//! Extracted out of `mine_inventory.rs` (the `mine-inventory` subcommand)
//! because the scoring math is pure over two labeled count buckets —
//! exactly the shape `commands::mine::ClassCounts` already documents as
//! intentionally reusable "for any two classes, not just the two this
//! module mines by default." `mine-paired` needs the identical
//! eps-ratio-with-a-frequency-gate algorithm over a *different* pair of
//! buckets (stock/antislop rather than machine/human); extracting it
//! once here avoids a second, silently-divergent copy of the scoring
//! algorithm.
//!
//! `pub(crate)`, not a CLI subcommand of its own — this module has no
//! `Args`/`run`, only the scoring primitives `mine_inventory` and
//! `mine_paired` both build their own gated, sorted, top-truncated report
//! sections on top of.

use std::collections::{BTreeMap, BTreeSet};

use crate::commands::mine::ClassCounts;

/// `(count_num/total_num + eps/total_num) / (count_den/total_den +
/// eps/total_den)` — the eps-ratio score, oriented so `count_num`/
/// `total_num` is the "favored" side. `None` if either total is zero (no
/// basis for a rate at all).
///
/// Moved verbatim from `mine_inventory.rs`; unchanged body.
pub fn ratio_score(
    count_num: u64,
    total_num: u64,
    count_den: u64,
    total_den: u64,
    eps: f64,
) -> Option<f64> {
    if total_num == 0 || total_den == 0 {
        return None;
    }
    #[allow(clippy::cast_precision_loss)]
    let (count_num, total_num, count_den, total_den) = (
        count_num as f64,
        total_num as f64,
        count_den as f64,
        total_den as f64,
    );
    let numerator = count_num / total_num + eps / total_num;
    let denominator = count_den / total_den + eps / total_den;
    Some(numerator / denominator)
}

/// `true` iff every space-separated token of `ngram` has a per-token
/// frequency (from `reference`) at or above `min_freq`.
///
/// Renamed from `mine_inventory.rs`'s `passes_human_freq_gate` — same
/// logic, generalized name since the "reference" side is now
/// caller-supplied: the human unigram table for `mine-inventory`, the
/// antislop unigram table for `mine-paired`.
pub fn passes_token_freq_gate(
    ngram: &str,
    reference: &BTreeMap<String, u64>,
    min_freq: u64,
) -> bool {
    ngram
        .split(' ')
        .all(|token| reference.get(token).copied().unwrap_or(0) >= min_freq)
}

/// One scored n-gram entry (either ratio-mining direction).
#[derive(Debug, Clone)]
pub struct RatioEntry {
    pub ngram: String,
    pub count_a: u64,
    pub count_b: u64,
    pub score: f64,
}

/// One n-gram order's `a`-favored and `b`-favored top lists.
pub struct RatioOrderReport {
    pub order: usize,
    pub total_a: u64,
    pub total_b: u64,
    pub a_favored: Vec<RatioEntry>,
    pub b_favored: Vec<RatioEntry>,
}

/// Builds one order's [`RatioOrderReport`] from `counts`, gating every
/// candidate n-gram on `reference_unigram`'s per-token frequency (both
/// directions) and this order's own per-direction count threshold.
///
/// `counts.llm` is bucket "a", `counts.human` is bucket "b" — reusing
/// `ClassCounts`'s own documented generic-reuse contract (see its doc
/// comment in `commands::mine`); callers supply their own labels when
/// rendering (machine/human for `mine-inventory`, stock/antislop for
/// `mine-paired`).
#[allow(clippy::too_many_arguments)]
pub fn build_ratio_order_report(
    order: usize,
    counts: &ClassCounts,
    reference_unigram: &BTreeMap<String, u64>,
    min_ref_freq: u64,
    min_count_a: u64,
    min_count_b: u64,
    top: usize,
    eps: f64,
) -> RatioOrderReport {
    let total_a: u64 = counts.llm.values().sum();
    let total_b: u64 = counts.human.values().sum();

    let mut vocab: BTreeSet<&str> = BTreeSet::new();
    vocab.extend(counts.llm.keys().map(String::as_str));
    vocab.extend(counts.human.keys().map(String::as_str));

    let mut a_favored = Vec::new();
    let mut b_favored = Vec::new();
    for ngram in vocab {
        if !passes_token_freq_gate(ngram, reference_unigram, min_ref_freq) {
            continue;
        }
        let count_a = counts.llm.get(ngram).copied().unwrap_or(0);
        let count_b = counts.human.get(ngram).copied().unwrap_or(0);

        if count_a >= min_count_a
            && let Some(score) = ratio_score(count_a, total_a, count_b, total_b, eps)
        {
            a_favored.push(RatioEntry {
                ngram: ngram.to_string(),
                count_a,
                count_b,
                score,
            });
        }
        if count_b >= min_count_b
            && let Some(score) = ratio_score(count_b, total_b, count_a, total_a, eps)
        {
            b_favored.push(RatioEntry {
                ngram: ngram.to_string(),
                count_a,
                count_b,
                score,
            });
        }
    }
    a_favored.sort_by(|x, y| {
        y.score
            .total_cmp(&x.score)
            .then_with(|| x.ngram.cmp(&y.ngram))
    });
    a_favored.truncate(top);
    b_favored.sort_by(|x, y| {
        y.score
            .total_cmp(&x.score)
            .then_with(|| x.ngram.cmp(&y.ngram))
    });
    b_favored.truncate(top);

    RatioOrderReport {
        order,
        total_a,
        total_b,
        a_favored,
        b_favored,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small fixture where bucket `a` clearly dominates one n-gram and
    /// bucket `b` clearly dominates another; `build_ratio_order_report`
    /// puts each in its own favored list with the right label mapping
    /// (`counts.llm` -> `a`, `counts.human` -> `b`), and neither list
    /// leaks the other's top entry.
    #[test]
    fn build_ratio_order_report_labels_a_as_llm_and_b_as_human() {
        let mut reference = BTreeMap::new();
        for tok in ["zeta", "widget", "cascade", "rare", "token", "combo"] {
            reference.insert(tok.to_string(), 100);
        }
        let counts = ClassCounts {
            llm: BTreeMap::from([("zeta widget".to_string(), 10)]),
            human: BTreeMap::from([("rare token".to_string(), 10)]),
        };
        let report = build_ratio_order_report(2, &counts, &reference, 0, 1, 1, 10, 0.4);
        assert_eq!(report.total_a, 10);
        assert_eq!(report.total_b, 10);
        assert_eq!(report.a_favored[0].ngram, "zeta widget");
        assert_eq!(report.a_favored[0].count_a, 10);
        assert_eq!(report.a_favored[0].count_b, 0);
        assert_eq!(report.b_favored[0].ngram, "rare token");
        assert_eq!(report.b_favored[0].count_b, 10);
    }

    /// A candidate n-gram whose token fails the reference frequency gate
    /// is excluded from both directions regardless of its raw counts.
    #[test]
    fn build_ratio_order_report_excludes_ngrams_failing_the_frequency_gate() {
        let mut reference = BTreeMap::new();
        reference.insert("common".to_string(), 100);
        // "rare" is absent from the reference entirely.
        let counts = ClassCounts {
            llm: BTreeMap::from([("common rare".to_string(), 50)]),
            human: BTreeMap::new(),
        };
        let report = build_ratio_order_report(2, &counts, &reference, 25, 1, 1, 10, 0.4);
        assert!(report.a_favored.is_empty());
        assert!(report.b_favored.is_empty());
    }
}
