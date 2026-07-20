//! Shared support for fuzz targets that exercise a `catch_unwind`-based
//! panic mitigation (currently `friction_parse::parse`'s own — see its
//! doc comment) and need it to actually take effect under `cargo fuzz`.
//! Included via `#[path = "support.rs"] mod support;` rather than a
//! library crate, since this fuzz package is bin-targets-only (see
//! `fuzz/Cargo.toml`'s own doc comment on why it stays outside the main
//! workspace).

/// `libfuzzer-sys`'s `fuzz_target!` macro installs a panic hook (once,
/// via its own `initialize()`, called before the first input) that
/// prints and then calls `std::process::abort()` *from inside the hook*
/// — i.e. before Rust's normal unwind machinery ever runs (see that
/// crate's own doc comment on `initialize`, which frames this as a
/// deliberate `HACK / FIXME` to make the fuzzer treat every panic as a
/// crash). That hook fires unconditionally for *any* panic, including
/// one `friction_parse::parse`'s own internal `catch_unwind` is written
/// to catch — so without this, a fuzz target calling `parse` (directly,
/// as `fuzz_parse.rs` does, or transitively through the fixpoint driver,
/// as `fuzz_fix_document.rs` does) could never actually observe `parse`
/// successfully turning a `pulldown-cmark` panic into a `ParseError` the
/// way it does under a normal (non-fuzz) build; every occurrence of that
/// already-triaged, already-fixed upstream bug (see
/// `ParseError::UnderlyingParserPanicked`'s own doc comment for the
/// minimized repro `fuzz_parse.rs` found) would keep reporting as a
/// fresh "crash" forever.
///
/// Restoring the default (non-aborting) hook here does not weaken either
/// target's own "no panics" oracle: a panic *nothing* catches still
/// unwinds all the way up to `libfuzzer-sys`'s own outer `catch_unwind`
/// (in its `test_input_wrap`), which aborts on `Err` regardless of which
/// hook is installed — so a genuine, unmitigated panic (anywhere else in
/// `friction-parse`, `friction-apply`, or any rule family) is still
/// reported as a crash exactly as before. Only a panic `parse` itself
/// already catches stops unwinding early and never reaches that outer
/// boundary.
pub fn let_inner_catch_unwind_actually_catch() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let _ = std::panic::take_hook();
    });
}
