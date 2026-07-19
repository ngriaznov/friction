//! Markdown parsing and prose extraction.
//!
//! Turns raw markdown source into a [`friction_core::Document`]: a
//! `pulldown-cmark`-backed block tree with exact byte ranges, plus the
//! prose extracted from it. `friction-nlp` is responsible for
//! segmenting that prose into sentences and tokens; every `ProseUnit`
//! produced here has an empty `sentences` vector, since prose extraction
//! leaves `ProseUnit` sentence/token segmentation minimal by design.
//!
//! # Prose extraction rules
//!
//! - **Excluded**: fenced and indented code blocks, inline code spans,
//!   link/image destination URLs (and their surrounding `[...]( ... )`
//!   markup), raw HTML blocks, footnote reference markers, and GFM
//!   task-list checkboxes.
//! - **Included as prose**: paragraphs, headings (excluding the `#`/setext
//!   markup and, for ATX headings, the trailing newline), the text nested
//!   in block quotes and list items, table cell text (the table's own
//!   pipe/dash structure is not prose), link and image *label* text, and
//!   emphasis/strong/strikethrough delimiter markup (`**`, `_`, `~~`),
//!   which is treated as ordinary text rather than excluded punctuation.
//!
//! A block can yield zero, one, or several `ProseUnit`s: whenever excluded
//! content (or block-quote/list continuation markup that pulldown-cmark's
//! inline event stream simply does not cover) interrupts a block's text,
//! the interruption splits it into separate maximal contiguous "prose
//! runs", each becoming its own `ProseUnit` referencing the same block
//! index. See the private `extract` module for the byte-level algorithm.
//!
//! # Round-trip guarantee
//!
//! [`Document::new`] validates every span recursively: every block
//! and prose range is in-bounds, on a UTF-8 char boundary, and (for prose)
//! contained in the block it was extracted from — `parse` propagates any
//! violation as [`ParseError`] rather than ever panicking or silently
//! truncating. The crate's `tests/roundtrip.rs` additionally proves the
//! stronger byte-exact property: concatenating the
//! document's outermost ("root") block ranges, with the untouched source
//! bytes between them, reproduces the original source exactly.
//!
//! # Determinism
//!
//! Extraction is a single deterministic left-to-right pass over
//! `pulldown-cmark`'s offset event stream; it holds no ambient state
//! (`Vec`s only, no hash-based collections) and never touches the clock or
//! ambient randomness. Identical source bytes always produce an identical
//! block/prose structure.

mod error;
mod extract;

use std::sync::Arc;

pub use error::ParseError;
use extract::extract;
use friction_core::Document;

/// Parses `source` into a [`Document`]: a markdown block tree with exact
/// byte ranges, plus the prose extracted from it.
///
/// This is a pure function of `source`'s bytes: identical input
/// always produces an identical `Document`.
///
/// # Errors
/// Returns [`ParseError`] if the extracted block/prose structure fails
/// [`Document`]'s span-honesty validation. This should not happen
/// for any input — `pulldown-cmark` is a total function over UTF-8 text —
/// but is surfaced as an error rather than a panic or `expect` so a bug in
/// the extraction logic degrades to a diagnosable error instead of a
/// crash.
pub fn parse(source: impl Into<Arc<str>>) -> Result<Document, ParseError> {
    let source = source.into();
    let (blocks, prose) = extract(&source);
    Document::new(source, blocks, prose).map_err(ParseError::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `parse` builds a `Document` whose `source()` is exactly the
    /// input, and which contains the expected block/prose shape for a
    /// trivial one-paragraph document.
    #[test]
    fn parse_builds_document_for_simple_paragraph() {
        let source = "Hello, world.\n";
        let doc = parse(source).expect("simple paragraph must parse");
        assert_eq!(doc.source(), source);
        assert_eq!(doc.blocks().len(), 1);
        assert_eq!(doc.prose().len(), 1);
        assert_eq!(doc.text(&doc.prose()[0].range).unwrap(), "Hello, world.");
    }

    /// An empty document parses to an empty, valid `Document`.
    #[test]
    fn parse_accepts_empty_source() {
        let doc = parse("").expect("empty source must parse");
        assert_eq!(doc.source(), "");
        assert!(doc.blocks().is_empty());
        assert!(doc.prose().is_empty());
    }
}
