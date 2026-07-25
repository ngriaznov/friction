//! [`SentenceSplicer`]: the byte-honest working-text primitive.
//!
//! It lets the per-sentence pipeline (see `crate::sentence`) match and
//! gate against a progressively-mutated working string — the same way
//! the reference engine's `fix_sentence` sequentially rewrites its own
//! `s` variable — while still emitting `Patch`es against the *original*
//! document bytes.
//!
//! A sentence's working text is modeled as an ordered chain of chunks:
//! either an untouched slice of the original source (still
//! byte-addressable back into the document) or a fixed replacement string
//! from an already-applied earlier-stage edit (no original-byte
//! correspondence). [`SentenceSplicer::apply`] always resolves a
//! working-text range against this chain, so a later stage can keep
//! matching against `working_text()` (the current mutated string) while
//! every accepted edit still lands as one honest `Patch` against the
//! document's own bytes.

use std::ops::Range;

use friction_core::{Patch, RuleId, Tier};

/// One piece of a sentence's working text.
#[derive(Debug, Clone)]
enum Chunk {
    /// An untouched slice of the original source — still byte-addressable.
    Original(Range<usize>),
    /// A fixed replacement string from an already-applied edit — no
    /// original-byte correspondence.
    Replaced(Box<str>),
}

impl Chunk {
    fn len(&self) -> usize {
        match self {
            Self::Original(range) => range.end - range.start,
            Self::Replaced(text) => text.len(),
        }
    }

    fn text<'a>(&'a self, source: &'a str) -> &'a str {
        match self {
            Self::Original(range) => &source[range.clone()],
            Self::Replaced(text) => text,
        }
    }
}

/// The byte-honest working-text splicer for one sentence.
///
/// `new` seeds the chunk chain with a single [`Chunk::Original`] spanning
/// the whole sentence; each accepted [`Self::apply`] call splits the chunk
/// containing `working_range` into up to three pieces (before / the new
/// replacement / after), recording one [`Patch`] against the document's
/// original bytes. `working_text()` always reflects every accepted edit so
/// far.
pub struct SentenceSplicer<'src> {
    source: &'src str,
    opening_start: usize,
    chunks: Vec<Chunk>,
    patches: Vec<Patch>,
}

impl<'src> SentenceSplicer<'src> {
    /// Seeds a new splicer over `sentence_range` (a byte range into
    /// `source`).
    #[must_use]
    pub fn new(source: &'src str, sentence_range: Range<usize>) -> Self {
        Self {
            source,
            opening_start: sentence_range.start,
            chunks: vec![Chunk::Original(sentence_range)],
            patches: Vec::new(),
        }
    }

    /// The current working text: every chunk's text, concatenated in
    /// order.
    #[must_use]
    pub fn working_text(&self) -> String {
        let mut out = String::new();
        for chunk in &self.chunks {
            out.push_str(chunk.text(self.source));
        }
        out
    }

    /// Replaces `working_range` (a byte range into [`Self::working_text`])
    /// with `replacement`, recording one [`Patch`] against the document's
    /// original bytes and reshaping the chunk chain.
    ///
    /// `working_range` must fall entirely within a single
    /// [`Chunk::Original`] piece — guaranteed in practice by pack
    /// disjointness (an operation never matches text an earlier operation
    /// just introduced). Returns `false` without mutating anything if this
    /// invariant is violated (an engine bug); debug builds additionally
    /// assert.
    pub fn apply(
        &mut self,
        working_range: Range<usize>,
        replacement: &str,
        rule: RuleId,
        tier: Tier,
    ) -> bool {
        if working_range.start > working_range.end {
            debug_assert!(false, "SentenceSplicer::apply: inverted range");
            return false;
        }

        let mut cursor = 0usize;
        for (index, chunk) in self.chunks.iter().enumerate() {
            let len = chunk.len();
            let chunk_range = cursor..(cursor + len);

            if working_range.start >= chunk_range.start && working_range.end <= chunk_range.end {
                let Chunk::Original(orig) = chunk else {
                    debug_assert!(
                        false,
                        "SentenceSplicer::apply: range falls inside an already-replaced chunk"
                    );
                    return false;
                };

                let rel_start = working_range.start - cursor;
                let rel_end = working_range.end - cursor;
                let abs_start = orig.start + rel_start;
                let abs_end = orig.start + rel_end;

                self.patches.push(Patch::new(
                    abs_start..abs_end,
                    replacement.to_string(),
                    rule,
                    tier,
                ));

                let before = orig.start..abs_start;
                let after = abs_end..orig.end;
                let mut next_chunks = Vec::with_capacity(self.chunks.len() + 2);
                for (other_index, other) in self.chunks.iter().enumerate() {
                    if other_index != index {
                        next_chunks.push(other.clone());
                        continue;
                    }
                    if !before.is_empty() {
                        next_chunks.push(Chunk::Original(before.clone()));
                    }
                    if !replacement.is_empty() {
                        next_chunks.push(Chunk::Replaced(replacement.into()));
                    }
                    if !after.is_empty() {
                        next_chunks.push(Chunk::Original(after.clone()));
                    }
                }
                self.chunks = next_chunks;
                return true;
            }
            cursor += len;
        }

        debug_assert!(
            false,
            "SentenceSplicer::apply: working_range not found within a single chunk"
        );
        false
    }

    /// `true` if at least one edit has been applied to this sentence.
    #[must_use]
    pub const fn edited(&self) -> bool {
        !self.patches.is_empty()
    }

    /// `true` if the working text's very first chunk is an untouched
    /// [`Chunk::Original`] slice (as opposed to an earlier edit's
    /// replacement text) — used by the recapitalization step to avoid
    /// splicing into replacement text a stage already capitalized itself.
    #[must_use]
    pub fn starts_with_original(&self) -> bool {
        matches!(self.chunks.first(), Some(Chunk::Original(_)))
    }

    /// `true` if the working text still opens with the sentence's own
    /// original first byte — i.e. no applied edit has deleted or replaced
    /// the sentence's opening. Distinguishes an opener the author wrote
    /// at sentence start from one a leading deletion newly exposed.
    #[must_use]
    pub fn opening_intact(&self) -> bool {
        matches!(self.chunks.first(), Some(Chunk::Original(r)) if r.start == self.opening_start)
    }

    /// Consumes the splicer, returning every accepted edit as a `Patch`
    /// against the document's original bytes, in the order they were
    /// applied.
    #[must_use]
    pub fn finish(self) -> Vec<Patch> {
        self.patches
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rule(id: &'static str) -> RuleId {
        RuleId::new(id)
    }

    #[test]
    fn working_text_starts_as_the_seeded_sentence_slice() {
        let source = "Hello there, world.";
        let splicer = SentenceSplicer::new(source, 0..source.len());
        assert_eq!(splicer.working_text(), source);
    }

    #[test]
    fn single_splice_records_a_patch_and_updates_working_text() {
        let source = "It utilizes the tool.";
        let mut splicer = SentenceSplicer::new(source, 0..source.len());
        // "utilizes" is at bytes 3..11.
        assert!(splicer.apply(3..11, "uses", rule("sub.utilize"), Tier::Fix));
        assert_eq!(splicer.working_text(), "It uses the tool.");
        let patches = splicer.finish();
        assert_eq!(patches.len(), 1);
        assert_eq!(patches[0].range, 3..11);
        assert_eq!(patches[0].replacement, "uses");
    }

    #[test]
    fn multi_splice_chains_correctly_left_to_right() {
        let source = "It utilizes a robust pipeline to facilitate delivery.";
        let mut splicer = SentenceSplicer::new(source, 0..source.len());
        assert!(splicer.apply(3..11, "uses", rule("r1"), Tier::Fix));
        // After the first splice, working text is
        // "It uses a robust pipeline to facilitate delivery." — "robust "
        // sits at working-text bytes 10..17.
        assert_eq!(&splicer.working_text()[10..17], "robust ");
        assert!(splicer.apply(10..17, "", rule("r2"), Tier::Fix));
        assert_eq!(
            splicer.working_text(),
            "It uses a pipeline to facilitate delivery."
        );

        let patches = splicer.finish();
        assert_eq!(patches.len(), 2);
        // Both patches must be honest against the ORIGINAL source.
        assert_eq!(patches[0].range, 3..11);
        assert_eq!(patches[1].range, 14..21);
        assert_eq!(&source[14..21], "robust ");
    }

    #[test]
    fn multi_splice_chains_right_to_left_too() {
        let source = "It utilizes a robust pipeline to facilitate delivery.";
        let mut splicer = SentenceSplicer::new(source, 0..source.len());
        // Delete "robust " first (original bytes 14..21).
        assert!(splicer.apply(14..21, "", rule("r1"), Tier::Fix));
        assert_eq!(
            splicer.working_text(),
            "It utilizes a pipeline to facilitate delivery."
        );
        // Now replace "utilizes" (still at working-text bytes 3..11).
        assert!(splicer.apply(3..11, "uses", rule("r2"), Tier::Fix));
        assert_eq!(
            splicer.working_text(),
            "It uses a pipeline to facilitate delivery."
        );

        let patches = splicer.finish();
        assert_eq!(patches.len(), 2);
        assert_eq!(patches[0].range, 14..21);
        assert_eq!(patches[1].range, 3..11);
    }

    #[test]
    fn apply_within_a_replaced_chunk_fails_without_panicking_in_release_shape() {
        let source = "It utilizes a tool.";
        let mut splicer = SentenceSplicer::new(source, 0..source.len());
        assert!(splicer.apply(3..11, "uses", rule("r1"), Tier::Fix));
        // Working text is now "It uses a tool."; bytes 3..7 sit inside the
        // just-inserted "uses" replacement chunk, not an Original chunk.
        // In a debug build this panics via debug_assert; test the return
        // value contract with catch_unwind so this test itself passes in
        // both debug and release profiles.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            splicer.apply(3..7, "x", rule("r2"), Tier::Fix)
        }));
        if let Ok(applied) = result {
            assert!(!applied);
        }
    }

    #[test]
    fn apply_at_sentence_start_and_end_works() {
        let source = "Simply run it now.";
        let mut splicer = SentenceSplicer::new(source, 0..source.len());
        assert!(splicer.apply(0..7, "", rule("span.simply"), Tier::Fix));
        assert_eq!(splicer.working_text(), "run it now.");
    }

    #[test]
    fn edited_tracks_whether_any_edit_applied() {
        let source = "It utilizes the tool.";
        let mut splicer = SentenceSplicer::new(source, 0..source.len());
        assert!(!splicer.edited());
        assert!(splicer.apply(3..11, "uses", rule("r1"), Tier::Fix));
        assert!(splicer.edited());
    }

    #[test]
    fn opening_intact_only_until_the_sentence_front_is_edited() {
        let source = "Simply run it now.";
        // Untouched: intact.
        let mut splicer = SentenceSplicer::new(source, 0..source.len());
        assert!(splicer.opening_intact());
        // A later-in-sentence edit leaves the opening intact.
        assert!(splicer.apply(14..17, "immediately", rule("r1"), Tier::Fix));
        assert!(splicer.opening_intact());
        // A leading deletion removes the original first byte: no longer
        // intact, though the working text still starts with original
        // bytes ("run it ...").
        assert!(splicer.apply(0..7, "", rule("r2"), Tier::Fix));
        assert!(!splicer.opening_intact());
        assert!(splicer.starts_with_original());
        // A replacement at position 0 is not an intact opening either.
        let mut replaced = SentenceSplicer::new(source, 0..source.len());
        assert!(replaced.apply(0..6, "Just", rule("r3"), Tier::Fix));
        assert!(!replaced.opening_intact());
        assert!(!replaced.starts_with_original());
    }

    #[test]
    fn finish_with_no_applies_returns_no_patches() {
        let source = "Nothing to change here.";
        let splicer = SentenceSplicer::new(source, 0..source.len());
        assert!(splicer.finish().is_empty());
    }
}
