//! `Engine::fix_document` must never panic on arbitrary markdown-ish text.
//!
//! Four oracles, each an invariant the engine documents as guaranteed
//! rather than merely typical:
//!
//! 1. **No panic.** Returning `Err` is an accepted outcome (a pathological
//!    input failing to segment); only a panic is a finding.
//! 2. **Output is valid UTF-8.** Guaranteed by `String`'s own invariant,
//!    checked anyway.
//! 3. **Every applied patch's span was char-boundary-valid** against the
//!    pass it was applied to. Replayed pass by pass with this target's own
//!    splice rather than the engine's, so the replay is an independent
//!    reimplementation: it also re-derives the final output, which catches
//!    a bug in the engine's own apply step instead of sharing it.
//! 4. **Idempotence.** The *third* full run is byte-identical to the
//!    second. Deliberately not `fix(fix(x)) == fix(x)`: a second run may
//!    legitimately clean up after the first run's deletions, so the
//!    guarantee is that the third changes nothing.
//!
//! Every pass starts with a `friction_parse::parse` call, which catches
//! its own `pulldown-cmark`-internal panics (see
//! `friction_parse::ParseError::UnderlyingParserPanicked`); this target
//! calls `support::let_inner_catch_unwind_actually_catch` for the same
//! reason `fuzz_parse.rs` does, so that mitigation takes effect here too
//! instead of every occurrence reporting as a fresh crash.
#![no_main]

use std::sync::OnceLock;

use arbitrary::Arbitrary;
use friction_core::Patch;
use friction_edit::Engine;
use libfuzzer_sys::fuzz_target;

#[path = "support.rs"]
mod support;

/// Keeps each iteration cheap: the engine re-parses, re-tags and re-scans
/// per pass, and this target calls it three times. 8 KiB is comfortably
/// larger than every corpus fixture while staying fast enough for the
/// fuzzer to explore broadly.
const MAX_SOURCE_LEN: usize = 8192;

fn engine() -> &'static Engine {
    static ENGINE: OnceLock<Engine> = OnceLock::new();
    ENGINE.get_or_init(|| Engine::new().expect("the embedded models and packs must load"))
}

#[derive(Debug, Arbitrary)]
struct Input {
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

/// Applies non-overlapping `patches` right-to-left.
///
/// Deliberately this target's own splice rather than the engine's: an
/// independent reimplementation makes oracle 3 a real cross-check instead
/// of a tautology that would share any bug in the engine's own version.
fn splice(source: &str, patches: &[Patch]) -> String {
    let mut ordered: Vec<&Patch> = patches.iter().collect();
    ordered.sort_by_key(|p| std::cmp::Reverse(p.range.start));
    let mut out = source.to_string();
    for patch in ordered {
        out.replace_range(patch.range.clone(), patch.replacement.as_str());
    }
    out
}

/// Runs one full `fix_document`, checking oracles 2 and 3. Returns the
/// fixed text, or `None` if the engine returned `Err`.
fn fix_once_checked(source: &str) -> Option<String> {
    let (output, report) = engine().fix_document(source).ok()?;

    assert!(
        std::str::from_utf8(output.as_bytes()).is_ok(),
        "output was not valid UTF-8 for source {source:?}"
    );

    let mut reconstructed = source.to_string();
    for (index, pass) in report.passes.iter().enumerate() {
        for patch in &pass.applied_patches {
            assert!(
                patch.range.end <= reconstructed.len(),
                "pass {index}: patch {:?} out of bounds for a {}-byte source",
                patch.range,
                reconstructed.len()
            );
            assert!(
                reconstructed.is_char_boundary(patch.range.start)
                    && reconstructed.is_char_boundary(patch.range.end),
                "pass {index}: patch {:?} is not char-boundary-valid against {reconstructed:?}",
                patch.range
            );
        }
        reconstructed = splice(&reconstructed, &pass.applied_patches);
    }
    assert_eq!(
        reconstructed, output,
        "replaying every pass's applied_patches did not reproduce the engine's own output \
         for source {source:?}"
    );

    Some(output)
}

fuzz_target!(|input: Input| {
    support::let_inner_catch_unwind_actually_catch();
    let mut source = input.source;
    truncate_to(&mut source, MAX_SOURCE_LEN);

    let Some(once) = fix_once_checked(&source) else {
        return;
    };
    let Some(twice) = fix_once_checked(&once) else {
        panic!("succeeded on the source but failed on the second run, once = {once:?}");
    };
    let Some(thrice) = fix_once_checked(&twice) else {
        panic!("succeeded on the second run but failed on the third, twice = {twice:?}");
    };

    assert_eq!(
        twice, thrice,
        "the third run must be a no-op: source {source:?}"
    );
});
