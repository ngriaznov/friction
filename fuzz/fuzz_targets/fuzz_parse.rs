//! Fuzz target 1: `friction_parse::parse` must never panic on arbitrary
//! bytes, decoded as UTF-8 (lossily, so every byte string the fuzzer
//! generates yields *some* input string rather than being skipped).
//!
//! `parse`'s own contract (see `friction-parse/src/lib.rs`) is that it is
//! a total function over UTF-8 text — any malformed markdown/prose shape
//! degrades to a `ParseError`, never a panic — so this target's only
//! oracle is "did not panic"; both `Ok` and `Err` are accepted outcomes.
//! `parse` upholds that contract even when the underlying `pulldown-cmark`
//! parser itself panics (a real case this target found — see
//! `ParseError::UnderlyingParserPanicked`'s doc comment for the minimized
//! repro) by wrapping the call in `std::panic::catch_unwind` internally;
//! see `support::let_inner_catch_unwind_actually_catch` for why that
//! needs a little help to actually work under `cargo fuzz`.
#![no_main]

use libfuzzer_sys::fuzz_target;

#[path = "support.rs"]
mod support;

fuzz_target!(|data: &[u8]| {
    support::let_inner_catch_unwind_actually_catch();
    let source = String::from_utf8_lossy(data);
    let _ = friction_parse::parse(source.as_ref());
});
