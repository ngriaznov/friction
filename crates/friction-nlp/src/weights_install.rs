//! Pre-init byte-installation seam shared by [`crate::tag_perceptron`] and
//! [`crate::dep_perceptron`]: the loading path a wasm build uses to supply
//! the perceptron weight artifacts at runtime (fetched separately, e.g.
//! from a CDN) instead of paying for them in every page load's binary —
//! see the `embedded-weights` feature in this crate's `Cargo.toml`.
//!
//! A native build never calls [`install`]: its `for_lang` always finds
//! `embedded-weights` on and reads straight from the `include_bytes!`'d
//! static, exactly as before this module existed — this seam is additive,
//! not a behavior change to the default build.
//!
//! # Format auto-detection
//!
//! Both artifacts ship in two shapes: the audited `json.gz` interchange
//! format (gzip magic `0x1f 0x8b`) and the derived, faster-to-load `.bin`
//! view format (starts with an 8-byte ASCII magic — see
//! [`crate::weights_bin`]). [`is_gzip`] is the one bit of sniffing needed
//! to route installed bytes to whichever loader already exists for that
//! shape; callers never have to say which one they're handing in.

use std::sync::OnceLock;

/// Errors installing pre-init weight bytes.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum InstallError {
    /// `install_*` was called more than once for the same artifact slot.
    ///
    /// This is the only "too late" condition this module can detect
    /// cheaply: unlike `friction_packs::DMS` (a `LazyLock` this crate's
    /// sibling can query with `LazyLock::get` before it's forced),
    /// neither [`crate::tag_perceptron::PerceptronTagger::for_lang`] nor
    /// [`crate::dep_perceptron::PerceptronParser::for_lang`] caches its
    /// result behind a singleton here — every call re-reads whatever is
    /// currently installed (or the embedded fallback). That means a
    /// *second* install is always a caller bug (the first install's bytes
    /// are what every future load already sees, so installing again can
    /// only be a mistake), and this variant catches exactly that.
    ///
    /// What this can **not** catch: several call sites elsewhere in this
    /// workspace (`friction-match`'s `overuse`/`jargon`/`lvc` modules,
    /// `friction-harness`) cache their *own* `OnceLock<PerceptronTagger>`
    /// built from a first call to `PerceptronTagger::new()`. Installing
    /// after one of those has already forced does not retroactively
    /// change its cached tagger. **Contract: call every `install_*`
    /// function before constructing (directly or indirectly) any tagger
    /// or parser in the process** — there is no cheap way to detect a
    /// late install across an arbitrary caller's own cache, so this is a
    /// documented contract, not an enforced one.
    #[error(
        "weights were already installed for this slot — install_* must be called at most once, \
         before any PerceptronTagger/PerceptronParser is built (including inside any caller's \
         own cached singleton)"
    )]
    AlreadyInstalled,
}

/// `true` if `bytes` starts with the gzip magic (`0x1f 0x8b`) — the audited
/// `json.gz` interchange format, distinguishing it from the derived `.bin`
/// view format (which starts with an unrelated 8-byte ASCII magic; see
/// [`crate::weights_bin::split_header`]'s own check for that half).
pub fn is_gzip(bytes: &[u8]) -> bool {
    bytes.starts_with(&[0x1f, 0x8b])
}

/// One artifact's pre-init override slot: leaked bytes ([`install`] below
/// explains why leaking is the right call here), set at most once.
pub struct OverrideSlot(OnceLock<&'static [u8]>);

impl OverrideSlot {
    pub const fn new() -> Self {
        Self(OnceLock::new())
    }

    /// Leaks `bytes` and stores the resulting `'static` slice, failing if
    /// this slot already holds one.
    ///
    /// Leaking (`Vec::leak`) rather than keeping `bytes` owned in the
    /// `OnceLock` itself is deliberate: [`crate::tag_perceptron::TaggerView`]/
    /// [`crate::dep_perceptron::ParserView`]'s zero-copy `ViewBackend` needs
    /// a `'static` borrow to live in a `PerceptronTagger`/`PerceptronParser`
    /// with no lifetime parameter (matching the embedded-static path's own
    /// shape) — the same reason `include_bytes!` gets a `'static` slice for
    /// free. A wasm module installs its weights once per page/worker
    /// lifetime and then runs until the page is torn down (freeing the
    /// whole arena at once), so the leak is bounded by something that was
    /// always going to reclaim it wholesale; a native build never calls
    /// `install_*` at all, so it never leaks anything.
    pub(crate) fn set(&self, bytes: Vec<u8>) -> Result<(), InstallError> {
        // Checked before leaking, not after: a redundant `set` on an
        // already-occupied slot (a caller-bug retry, or `init_engine`
        // reinstalling after a sibling slot failed) must not also leak
        // this call's payload — see this type's own doc comment.
        if self.0.get().is_some() {
            return Err(InstallError::AlreadyInstalled);
        }
        let leaked: &'static [u8] = Vec::leak(bytes);
        self.0
            .set(leaked)
            .map_err(|_| InstallError::AlreadyInstalled)
    }

    pub(crate) fn get(&self) -> Option<&'static [u8]> {
        self.0.get().copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_gzip_detects_the_magic_and_rejects_bin_headers() {
        assert!(is_gzip(&[0x1f, 0x8b, 0x08, 0x00]));
        assert!(!is_gzip(b"FRTAGWT1"));
        assert!(!is_gzip(b""));
        assert!(!is_gzip(&[0x1f]));
    }

    #[test]
    fn override_slot_rejects_a_second_install() {
        let slot = OverrideSlot::new();
        assert!(slot.get().is_none());
        slot.set(vec![1, 2, 3]).expect("first install succeeds");
        assert_eq!(slot.get(), Some(&[1u8, 2, 3][..]));
        let err = slot.set(vec![4, 5, 6]).unwrap_err();
        assert!(matches!(err, InstallError::AlreadyInstalled));
        // The first install's bytes are still what's served.
        assert_eq!(slot.get(), Some(&[1u8, 2, 3][..]));
    }
}
