//! Fuzz target 2: `FixEngine::fix_document` (the fixpoint driver: parse ->
//! metrics -> gate -> scan -> fix -> resolve conflicts -> apply, up to
//! `friction_apply::MAX_ROUNDS` rounds — see `friction-apply/src/fix.rs`
//! and `friction-apply/src/driver.rs`) must never panic on arbitrary
//! markdown-ish text, for any of the five genres `friction-cli` exposes.
//!
//! Four oracles, each an invariant the real driver already documents as
//! *guaranteed* (see `friction-apply/src/lib.rs`'s module docs) rather
//! than merely typical:
//!
//! 1. **No panic.** `fix_document` returning `Err` is an accepted
//!    outcome (a pathological input failing to segment, say); only a
//!    panic is a finding.
//! 2. **Output is valid UTF-8.** Guaranteed by `String`'s own type
//!    invariant, checked explicitly anyway per this target's brief.
//! 3. **Every applied patch's span was char-boundary-valid** against the
//!    text of the round it was applied to. Reconstructed round-by-round
//!    using the same public `friction_apply::apply_patches` the driver
//!    itself uses to go from one round's `RoundReport::applied_patches`
//!    to the next round's source, which also re-derives the driver's
//!    final output independently of the value `fix_document` returned —
//!    a second, free cross-check that the two never diverge.
//! 4. **Idempotence.** `fix_document(fix_document(x)) ==
//!    fix_document(x)`, byte-for-byte — the cheap oracle
//!    `friction-apply/tests/idempotence_sweep.rs` already runs over the
//!    whole corpus; this target runs it over the fuzzer's generated
//!    inputs instead.
//!
//! Every round starts with a `friction_parse::parse` call, which catches
//! its own `pulldown-cmark`-internal panics (see
//! `friction_parse::ParseError::UnderlyingParserPanicked`); this target
//! calls `support::let_inner_catch_unwind_actually_catch` for the same
//! reason `fuzz_parse.rs` does, so that mitigation actually takes effect
//! here too instead of every occurrence reporting as a fresh crash.
#![no_main]

use std::sync::OnceLock;

use arbitrary::Arbitrary;
use friction_apply::{FixEngine, apply_patches};
use libfuzzer_sys::fuzz_target;

#[path = "support.rs"]
mod support;

/// The five genres `friction-cli::common::Genre` exposes (see that
/// crate's `Genre::as_str`), duplicated here (rather than depended on)
/// since `friction-cli`'s `Genre` type lives on its binary's own
/// `--genre` value-enum surface, not its fuzz-only library target.
const GENRES: [&str; 5] = ["docs", "blog", "readme", "email", "forum"];

/// Keeps each fuzz iteration cheap: the fixpoint driver re-parses,
/// re-tags, and re-scans every registered rule up to `MAX_ROUNDS` times
/// per call, and this target calls it *twice* (the idempotence oracle).
/// 8 KiB is comfortably larger than every fixture in the corpus the
/// idempotence sweep already covers, while keeping a single iteration
/// fast enough for the fuzzer to explore broadly within a bounded soak.
const MAX_SOURCE_LEN: usize = 8192;

fn engine() -> &'static FixEngine {
    static ENGINE: OnceLock<FixEngine> = OnceLock::new();
    ENGINE.get_or_init(|| FixEngine::new().expect("the embedded English tagger model must load"))
}

#[derive(Debug, Arbitrary)]
struct Input {
    genre_idx: u8,
    source: String,
}

/// Char-boundary-truncates `s` to at most `max_len` bytes.
fn truncate_to(s: &mut String, max_len: usize) {
    if s.len() <= max_len {
        return;
    }
    let mut end = max_len;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    s.truncate(end);
}

/// Runs `fix_document` once, asserting it did not panic, its output is
/// valid UTF-8, and every patch it applied was char-boundary-valid
/// against the round it was applied to (oracles 2 and 3). Returns the
/// fixed text on success.
fn fix_once_checked(genre: &str, source: &str) -> Option<String> {
    let (output, report) = engine().fix_document(source, genre).ok()?;

    assert!(
        std::str::from_utf8(output.as_bytes()).is_ok(),
        "fix_document output was not valid UTF-8 for genre {genre:?}, source {source:?}"
    );

    let mut reconstructed = source.to_string();
    for round in &report.rounds {
        for patch in &round.applied_patches {
            assert!(
                patch.range.end <= reconstructed.len(),
                "round {}: patch {:?} out of bounds for a {}-byte source (genre {genre:?})",
                round.round,
                patch.range,
                reconstructed.len()
            );
            assert!(
                reconstructed.is_char_boundary(patch.range.start)
                    && reconstructed.is_char_boundary(patch.range.end),
                "round {}: patch {:?} is not char-boundary-valid against {:?} (genre {genre:?})",
                round.round,
                patch.range,
                reconstructed
            );
        }
        reconstructed = apply_patches(&reconstructed, &round.applied_patches);
    }
    assert_eq!(
        reconstructed, output,
        "replaying every round's applied_patches did not reproduce fix_document's own output \
         for genre {genre:?}, source {source:?}"
    );

    Some(output)
}

fuzz_target!(|input: Input| {
    support::let_inner_catch_unwind_actually_catch();
    let genre = GENRES[input.genre_idx as usize % GENRES.len()];
    let mut source = input.source;
    truncate_to(&mut source, MAX_SOURCE_LEN);

    let Some(once) = fix_once_checked(genre, &source) else {
        return;
    };

    let Some(twice) = fix_once_checked(genre, &once) else {
        panic!(
            "fix_document succeeded on `once` but failed on fix(fix(x)) for genre {genre:?}, \
             once = {once:?}"
        );
    };

    assert_eq!(
        once, twice,
        "fix_document is not idempotent for genre {genre:?}: source {source:?}"
    );
});
