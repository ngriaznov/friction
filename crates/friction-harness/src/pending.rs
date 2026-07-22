//! The manifest of byte-exact engine assertions this milestone defers.
//!
//! This milestone verifies closure, tell-span reduction, and score
//! improvement for every `accept` fixture instead of wiring a real
//! fix-producing engine and asserting exact byte equality against its
//! documented output. [`tests/pending_manifest.rs`](../../tests/pending_manifest.rs)
//! pins the count this function returns, so adding or resolving one of
//! these forces a deliberate update here rather than a silent count drift.

/// One (input, output) pair an `accept` fixture implies, that a real
/// fix-producing engine would eventually need to reproduce byte-for-byte.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PendingEngineAssertion {
    /// The `accept` fixture this pair belongs to.
    pub fixture_id: &'static str,
    /// Which pair, for a fixture with more than one (`pivot_constructed_cases`).
    pub pair_index: usize,
}

/// The full set of deferred byte-exact engine assertions.
///
/// One row per `(input, output)` pair this milestone's `accept` fixtures
/// imply — `regression1_getting_started` (1),
/// `composed_four_operation_paragraph` (1), `pivot_constructed_cases` (5,
/// one per pair), `pivot_real_corpus` (1), `near_noop_clean_text` (1) — 9
/// total.
#[must_use]
pub fn pending_engine_assertions() -> Vec<PendingEngineAssertion> {
    let mut assertions = vec![
        PendingEngineAssertion {
            fixture_id: "regression1_getting_started",
            pair_index: 0,
        },
        PendingEngineAssertion {
            fixture_id: "composed_four_operation_paragraph",
            pair_index: 0,
        },
    ];
    assertions.extend((0..5).map(|pair_index| PendingEngineAssertion {
        fixture_id: "pivot_constructed_cases",
        pair_index,
    }));
    assertions.push(PendingEngineAssertion {
        fixture_id: "pivot_real_corpus",
        pair_index: 0,
    });
    assertions.push(PendingEngineAssertion {
        fixture_id: "near_noop_clean_text",
        pair_index: 0,
    });
    assertions
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pending_engine_assertions_totals_nine() {
        assert_eq!(pending_engine_assertions().len(), 9);
    }
}
