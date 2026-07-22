//! Pins the count of deferred byte-exact engine assertions, so adding or
//! resolving one forces a deliberate update to
//! [`pending_engine_assertions`](friction_harness::pending::pending_engine_assertions).

use friction_harness::pending::pending_engine_assertions;

#[test]
fn deferred_byte_exact_engine_assertions_total_is_pinned() {
    assert_eq!(
        pending_engine_assertions().len(),
        9,
        "this milestone defers byte-exact engine output assertions for 9 (input, output) \
         pairs (regression1_getting_started x1, composed_four_operation_paragraph x1, \
         pivot_constructed_cases x5, pivot_real_corpus x1, near_noop_clean_text x1) — if \
         this count changed, update pending_engine_assertions() deliberately rather than \
         letting it drift"
    );
}
