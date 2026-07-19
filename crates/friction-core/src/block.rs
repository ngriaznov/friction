//! Markdown block-level AST nodes, populated by `friction-parse`.

use std::ops::Range;

use crate::span::Spanned;

/// A markdown block-level AST node together with its exact byte range in
/// the original document source.
///
/// `friction-parse` is responsible for producing these from
/// `pulldown-cmark`'s event stream; `friction-core` only defines the shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Block {
    /// The kind of markdown construct this block represents.
    pub kind: BlockKind,
    /// Byte range of this block in the original document source.
    pub range: Range<usize>,
}

impl Block {
    /// Creates a new block spanning `range` in the original source.
    #[must_use]
    pub const fn new(kind: BlockKind, range: Range<usize>) -> Self {
        Self { kind, range }
    }
}

impl Spanned for Block {
    fn range(&self) -> Range<usize> {
        self.range.clone()
    }
}

/// The kind of markdown block-level construct a [`Block`] represents.
///
/// Mirrors the block-level constructs `pulldown-cmark` can emit. Marked
/// `#[non_exhaustive]` so new constructs (e.g. frontmatter, definition
/// lists) can be added without a breaking change for downstream matches;
/// `friction-parse` constructs known variants directly by name.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum BlockKind {
    /// A paragraph of prose.
    Paragraph,
    /// An ATX or setext heading.
    Heading {
        /// Heading level, `1` (largest) through `6` (smallest).
        level: u8,
    },
    /// A block quote (`> ...`).
    BlockQuote,
    /// A fenced or indented code block. Its contents are excluded from
    /// prose extraction.
    CodeBlock {
        /// The fence info string (e.g. the language tag), if any.
        info: Option<Box<str>>,
    },
    /// An ordered or unordered list.
    List {
        /// `true` for an ordered (`1.`) list, `false` for unordered (`-`).
        ordered: bool,
        /// Starting number for an ordered list.
        start: Option<u64>,
    },
    /// A single list item within a [`BlockKind::List`].
    ListItem,
    /// A GFM table. The table's own structure (rows, cell boundaries) is
    /// excluded from prose extraction; cell text is prose.
    Table,
    /// A single row within a [`BlockKind::Table`].
    TableRow,
    /// A single cell within a [`BlockKind::TableRow`].
    TableCell {
        /// `true` if this cell is in the table's header row.
        header: bool,
    },
    /// A thematic break (`---`, `***`, `___`).
    ThematicBreak,
    /// A raw HTML block.
    HtmlBlock,
    /// A footnote definition.
    FootnoteDefinition {
        /// The footnote's label, e.g. `"1"` in `[^1]: ...`.
        label: Box<str>,
    },
    /// A block-level construct not covered by another variant, identified
    /// by an implementation-defined tag.
    Other(Box<str>),
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A block reports its own range via `Spanned`.
    #[test]
    fn block_range_matches_constructor() {
        let block = Block::new(BlockKind::Paragraph, 4..10);
        assert_eq!(block.range(), 4..10);
        assert_eq!(block.range, 4..10);
    }

    /// `BlockKind` variants with fields carry the expected data.
    #[test]
    fn block_kind_carries_variant_data() {
        let heading = BlockKind::Heading { level: 2 };
        assert!(matches!(heading, BlockKind::Heading { level: 2 }));

        let code = BlockKind::CodeBlock {
            info: Some("rust".into()),
        };
        assert!(
            matches!(code, BlockKind::CodeBlock { info: Some(ref lang) } if lang.as_ref() == "rust")
        );
    }
}
