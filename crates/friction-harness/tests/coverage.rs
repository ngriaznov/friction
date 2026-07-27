//! Every fixture id `docs/research/fixtures.json` defines must have a
//! dedicated test in `accept_fixtures.rs`/`reject_fixtures.rs`/
//! `rank_fixtures.rs`. This file is the first line of defense: a sorted
//! diff against a hardcoded expected id list, checked independently in
//! each of the three categories.
//!
//! Belt-and-suspenders: the per-id `match` dispatch in the three sibling
//! test files each end in an `other => panic!("unhandled fixture id:
//! {other}")` arm, so a JSON-side id addition fails loudly in two
//! independent places even before this file is touched.

use friction_harness::fixtures::FIXTURES;

const EXPECTED_ACCEPT_IDS: &[&str] = &[
    "regression1_getting_started",
    "composed_four_operation_paragraph",
    "pivot_constructed_cases",
    "pivot_real_corpus",
    "near_noop_clean_text",
    "frame_dejust_provenance_question",
];

const EXPECTED_REJECT_IDS: &[&str] = &[
    "pivot_trap_subject_genitive",
    "pivot_trap_modified_nominal",
    "pivot_trap_quantified",
    "pivot_trap_artifact_noun",
    "pivot_trap_non_light_verb",
    "pivot_trap_passive",
    "heading_pivot",
    "bridge_modal_insertion",
    "bridge_connective_insertion",
    "bridge_content_insertion",
    "bridge_inert_insertion",
    "word_salad_synthesis",
    "dejust_guard_just_as",
];

const EXPECTED_RANK_IDS: &[&str] = &["chatspeak_vs_human", "dumb_human_vs_conservative"];

/// Sorts both `actual` and a copy of `expected`, and panics with a
/// diff-style message if they disagree — used by all three checks below so
/// a mismatch is easy to read regardless of declaration order in either
/// side.
fn assert_same_ids(category: &str, actual: &[String], expected: &[&str]) {
    let mut actual_sorted = actual.to_vec();
    actual_sorted.sort();
    let mut expected_sorted: Vec<String> = expected.iter().map(|s| (*s).to_string()).collect();
    expected_sorted.sort();

    if actual_sorted != expected_sorted {
        let missing: Vec<&String> = expected_sorted
            .iter()
            .filter(|id| !actual_sorted.contains(id))
            .collect();
        let unexpected: Vec<&String> = actual_sorted
            .iter()
            .filter(|id| !expected_sorted.contains(id))
            .collect();
        panic!(
            "{category} fixture ids diverged from the expected list:\n\
             missing from fixtures.json (test expects but JSON lacks): {missing:?}\n\
             present in fixtures.json but not covered by a test: {unexpected:?}"
        );
    }
}

#[test]
fn every_accept_fixture_id_is_covered_by_a_test() {
    let actual: Vec<String> = FIXTURES.accept.iter().map(|f| f.id.clone()).collect();
    assert_same_ids("accept", &actual, EXPECTED_ACCEPT_IDS);
}

#[test]
fn every_reject_fixture_id_is_covered_by_a_test() {
    let actual: Vec<String> = FIXTURES.reject.iter().map(|f| f.id.clone()).collect();
    assert_same_ids("reject", &actual, EXPECTED_REJECT_IDS);
}

#[test]
fn every_rank_fixture_id_is_covered_by_a_test() {
    let actual: Vec<String> = FIXTURES.rank.iter().map(|f| f.id.clone()).collect();
    assert_same_ids("rank", &actual, EXPECTED_RANK_IDS);
}
