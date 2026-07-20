//! Library surface for `friction-cli`.
//!
//! `main.rs` builds the `friction` binary directly from its own
//! (unexported) module tree; this library target exists only so an
//! external caller — currently
//! `fuzz/fuzz_targets/fuzz_sarif_line_col.rs` — can reuse a small, pure
//! piece of that tree without linking against a binary crate, which
//! Cargo does not support.
//!
//! `common.rs` is compiled twice (once into this library, once into the
//! `friction` binary target): an ordinary consequence of a package
//! sharing one module's source file between a `[lib]` and a `[[bin]]`,
//! not a functional difference between the two copies. It stays a
//! private module here (`mod`, not `pub mod`) so its own doc comments
//! — written for a binary crate's internal module, not a published
//! library's public API — don't have to satisfy public-API doc lints;
//! [`offset_to_line_col`] below is the one function this crate exports.
//! `#[allow(dead_code)]`: this library target only ever calls
//! `offset_to_line_col`, so every other item `common.rs` defines (used by
//! the `friction` binary target, which compiles the same file
//! separately) is legitimately unreachable dead code from *this* crate's
//! point of view — a property of splitting one source file across two
//! crate targets, not a real defect in either.
#[allow(dead_code)]
mod common;

/// Converts a 0-based byte offset into `source` to a 1-based `(line,
/// column)` pair, both counted in `char`s, not bytes.
///
/// A thin, public re-export of `common::offset_to_line_col` — the
/// function `friction-cli`'s SARIF renderer
/// (`src/sarif.rs::render`) uses to turn a [`friction_core::Finding`]'s
/// byte range into a SARIF `region`. See that private function's own
/// doc comment for the exact clamping/counting contract.
#[must_use]
pub fn offset_to_line_col(source: &str, offset: usize) -> (usize, usize) {
    common::offset_to_line_col(source, offset)
}
