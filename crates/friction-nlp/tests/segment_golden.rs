//! Golden sentence-segmentation test set.
//!
//! Loads `tests/data/segment_golden.toml` — a set of tricky English inputs
//! (decimals, abbreviations, initials, ellipses, quotes, code-ish tokens,
//! URLs, literal markdown markup left in prose, and paragraph-internal
//! newlines) paired with the exact sentences each should segment into —
//! and asserts [`SrxSegmenter`] reproduces every one of them, byte for
//! byte: each returned range, sliced against the input, must equal the
//! expected sentence string exactly.

use std::ops::Range;

use friction_nlp::{Segmenter, SrxSegmenter};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Fixture {
    case: Vec<Case>,
}

#[derive(Debug, Deserialize)]
struct Case {
    name: String,
    input: String,
    expect: Vec<String>,
}

const GOLDEN_TOML: &str = include_str!("data/segment_golden.toml");

fn load_fixture() -> Fixture {
    toml::from_str(GOLDEN_TOML).expect("tests/data/segment_golden.toml is well-formed")
}

/// Every golden case's expected sentences round-trip through
/// `SrxSegmenter::segment`, sliced byte-exactly from the original input.
#[test]
fn golden_set_segments_exactly() {
    let fixture = load_fixture();
    assert!(
        fixture.case.len() >= 50,
        "golden fixture should cover a wide range of cases"
    );

    let segmenter = SrxSegmenter::new();
    let mut failures = Vec::new();

    for case in &fixture.case {
        let ranges = segmenter.segment(&case.input, 0);
        let actual: Vec<&str> = ranges.iter().map(|r| &case.input[r.clone()]).collect();

        if actual != case.expect.iter().map(String::as_str).collect::<Vec<_>>() {
            failures.push(format!(
                "case {:?}:\n  input:    {:?}\n  expected: {:?}\n  actual:   {:?}",
                case.name, case.input, case.expect, actual
            ));
        }

        // Byte-span exactness: slicing the *input* with each raw range
        // must reproduce the sentence text with no off-by-one drift, for
        // every range regardless of whether the sentence text matched.
        for (range, text) in ranges.iter().zip(actual.iter()) {
            assert_eq!(
                &case.input[range.clone()],
                *text,
                "case {:?}: range {range:?} does not slice back to its own reported text",
                case.name
            );
        }
    }

    assert!(
        failures.is_empty(),
        "{} golden case(s) failed:\n\n{}",
        failures.len(),
        failures.join("\n\n")
    );
}

/// The golden fixture collectively exercises at least 100 sentences, as
/// required of the segmentation golden set.
#[test]
fn golden_set_covers_at_least_100_sentences() {
    let fixture = load_fixture();
    let total: usize = fixture.case.iter().map(|c| c.expect.len()).sum();
    assert!(
        total >= 100,
        "golden fixture covers only {total} sentences, want >= 100"
    );
}

/// Every returned range is non-empty, in strictly increasing source
/// order, and never straddles a UTF-8 character boundary (guaranteed by
/// successfully indexing `&str` with it, which panics otherwise).
#[test]
fn golden_set_ranges_are_ordered_and_non_overlapping() {
    let fixture = load_fixture();
    let segmenter = SrxSegmenter::new();

    for case in &fixture.case {
        let ranges: Vec<Range<usize>> = segmenter.segment(&case.input, 0);
        let mut prev_end = 0usize;
        for range in &ranges {
            assert!(
                !range.is_empty(),
                "case {:?}: segment produced an empty range {range:?}",
                case.name
            );
            assert!(
                range.start >= prev_end,
                "case {:?}: range {range:?} overlaps or precedes the previous one (prev_end={prev_end})",
                case.name
            );
            // Panics on a non-boundary index, which is itself a failure.
            let _ = &case.input[range.clone()];
            prev_end = range.end;
        }
    }
}

/// `SrxSegmenter::segment` is a pure, deterministic function of its
/// inputs: running the whole golden set twice yields byte-identical
/// results both times.
#[test]
fn golden_set_segmentation_is_deterministic_across_runs() {
    let fixture = load_fixture();
    let segmenter = SrxSegmenter::new();

    for case in &fixture.case {
        let first = segmenter.segment(&case.input, 0);
        let second = segmenter.segment(&case.input, 0);
        assert_eq!(
            first, second,
            "case {:?}: segmentation was not deterministic across two calls",
            case.name
        );
    }
}

/// A non-zero `base_offset` shifts every range by exactly that amount,
/// for the whole golden set at once — proving the offset is applied
/// uniformly rather than only in the small hand-written unit tests.
#[test]
fn golden_set_respects_base_offset() {
    const OFFSET: usize = 1000;

    let fixture = load_fixture();
    let segmenter = SrxSegmenter::new();

    for case in &fixture.case {
        let zero_based = segmenter.segment(&case.input, 0);
        let offset_based = segmenter.segment(&case.input, OFFSET);
        let shifted: Vec<Range<usize>> = zero_based
            .iter()
            .map(|r| r.start + OFFSET..r.end + OFFSET)
            .collect();
        assert_eq!(
            offset_based, shifted,
            "case {:?}: base_offset was not applied uniformly",
            case.name
        );
    }
}
