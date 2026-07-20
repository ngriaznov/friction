//! Fuzz target 3: `friction_cli::offset_to_line_col` (the byte-offset ->
//! `(line, column)` converter `friction check --format sarif` uses to
//! build every SARIF result's `physicalLocation.region`) must never panic
//! on arbitrary text plus an arbitrary offset, and its result must always
//! be a 1-based `(line, column)` pair within `text`'s actual bounds.
//!
//! The offset is deliberately *not* pre-validated against `text` (no
//! char-boundary check, no in-bounds check) before being handed to the
//! converter: `Finding`/`Patch` ranges are validated before a finding
//! ever reaches SARIF rendering in the real CLI pipeline, but this target
//! exists precisely to prove the converter itself — taken in isolation —
//! degrades safely (clamps, never panics) on an offset a caller failed to
//! validate, since a `usize` offset can land anywhere: in-bounds and on a
//! char boundary, in-bounds but mid-character (the interesting case for a
//! naive byte-slicing implementation), or past `text.len()` entirely.
#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Debug, Arbitrary)]
struct Input {
    text: String,
    offset: usize,
}

fuzz_target!(|input: Input| {
    let (line, col) = friction_cli::offset_to_line_col(&input.text, input.offset);

    assert!(line >= 1, "line must be 1-based, got {line}");
    assert!(col >= 1, "column must be 1-based, got {col}");

    // `text` has exactly `newline_count + 1` lines; the returned line
    // number must land on one of them, regardless of how far out of
    // bounds `offset` was.
    let total_lines = input.text.chars().filter(|&c| c == '\n').count() + 1;
    assert!(
        line <= total_lines,
        "line {line} exceeds text's {total_lines} lines for offset {} in {:?}",
        input.offset,
        input.text
    );
});
