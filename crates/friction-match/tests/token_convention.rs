//! Pins `token::tokenize_str`'s token-boundary convention to
//! `friction_harness::clean::tokenize`'s: for every battery sentence
//! below, the two must produce element-for-element identical token text,
//! even though `tokenize_str` runs directly on raw, mixed-case,
//! un-cleaned bytes and `clean::tokenize` runs on `clean(text)`
//! lowercased.
//!
//! This is a dev-dependency behavioral equivalence test, not shared
//! code. See this crate's own top-level docs for why the two tokenizers
//! stay separate implementations.

use friction_harness::clean;
use friction_harness::fixtures::FIXTURES;
use friction_match::token::tokenize_str;

fn assert_same_token_text(input: &str) {
    let expected = clean::tokenize(input);
    let actual: Vec<String> = tokenize_str(input, 0)
        .into_iter()
        .map(|t| t.text.to_string())
        .collect();
    assert_eq!(actual, expected, "token text mismatch for input: {input:?}");
}

#[test]
fn matches_clean_tokenize_for_straight_and_curly_contractions() {
    assert_same_token_text("Don't stop, please.");
    assert_same_token_text("it\u{2019}s a \u{2018}test\u{2019} run.");
}

#[test]
fn matches_clean_tokenize_for_hyphenated_compounds() {
    assert_same_token_text("This is a state-of-the-art, well-known approach.");
}

#[test]
fn matches_clean_tokenize_for_decimals() {
    assert_same_token_text("Pi is about 3.14 today, or 2,718 in another sense.");
}

#[test]
fn matches_clean_tokenize_for_trailing_punctuation_runs() {
    assert_same_token_text("Wait!! Are you serious?! Really...");
}

#[test]
fn matches_clean_tokenize_for_mixed_case_prose() {
    assert_same_token_text("The QUICK Brown Fox Jumps Over The Lazy Dog.");
}

#[test]
fn matches_clean_tokenize_for_composed_four_operation_paragraph_fixture() {
    let input = FIXTURES
        .accept
        .iter()
        .find(|f| f.id == "composed_four_operation_paragraph")
        .and_then(|f| f.input.as_deref())
        .expect("fixture has an input");
    assert_same_token_text(input);
}

#[test]
fn matches_clean_tokenize_for_near_noop_clean_text_fixture() {
    let input = FIXTURES
        .accept
        .iter()
        .find(|f| f.id == "near_noop_clean_text")
        .and_then(|f| f.input.as_deref())
        .expect("fixture has an input");
    assert_same_token_text(input);
}
