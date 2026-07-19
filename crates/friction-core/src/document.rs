//! [`Document`]: source text plus its block structure and extracted prose.

use std::ops::Range;
use std::sync::Arc;

use crate::block::Block;
use crate::error::CoreError;
use crate::span::{self, Spanned};

/// A parsed document.
///
/// Holds the original source text, the markdown block structure over it
/// (produced by `friction-parse`), and the prose extracted from it,
/// segmented into sentences and tokens (`friction-nlp`).
///
/// All byte ranges anywhere in a `Document` — on [`Block`]s, [`ProseUnit`]s,
/// [`Sentence`]s, and [`Token`]s — are offsets into `source`, and
/// [`Document::new`] validates that invariant recursively at construction
/// time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Document {
    source: Arc<str>,
    blocks: Vec<Block>,
    prose: Vec<ProseUnit>,
}

impl Document {
    /// Builds a document from its source text, block structure, and
    /// extracted prose, validating every byte span: each block's
    /// range must lie within `source`; each prose unit's `block` index
    /// must reference an existing block and its range must be contained in
    /// that block's range; and each sentence's and token's range must be
    /// contained within its parent's range.
    ///
    /// # Errors
    /// Returns the first [`CoreError`] found while validating `blocks`
    /// (in order), then `prose` (in order, recursing into sentences and
    /// tokens).
    pub fn new(
        source: impl Into<Arc<str>>,
        blocks: Vec<Block>,
        prose: Vec<ProseUnit>,
    ) -> Result<Self, CoreError> {
        let source = source.into();
        for block in &blocks {
            span::validate_range(&source, &block.range)?;
        }
        for unit in &prose {
            unit.validate(&source, &blocks)?;
        }
        Ok(Self {
            source,
            blocks,
            prose,
        })
    }

    /// The document's original source text.
    #[must_use]
    pub fn source(&self) -> &str {
        &self.source
    }

    /// A cheaply-cloneable handle to the document's original source text,
    /// for callers that need to hold onto it independent of the
    /// `Document`'s own lifetime.
    #[must_use]
    pub fn source_arc(&self) -> Arc<str> {
        Arc::clone(&self.source)
    }

    /// The document's markdown block structure, in source order.
    #[must_use]
    pub fn blocks(&self) -> &[Block] {
        &self.blocks
    }

    /// The document's extracted prose: one [`ProseUnit`] per prose-bearing
    /// block.
    #[must_use]
    pub fn prose(&self) -> &[ProseUnit] {
        &self.prose
    }

    /// Returns the source text addressed by `range`.
    ///
    /// # Errors
    /// Returns [`CoreError`] if `range` is out of bounds for the source or
    /// splits a UTF-8 character.
    pub fn text(&self, range: &Range<usize>) -> Result<&str, CoreError> {
        span::slice(&self.source, range)
    }
}

/// Prose extracted from a single [`Block`], segmented into sentences.
///
/// `friction-parse` produces one `ProseUnit` per prose-bearing block, with
/// `range` covering the block's prose text; sentence segmentation
/// (`friction-nlp`) and tokenization fill in `sentences` in a later pass,
/// so an empty `sentences` vector is a valid, not-yet-segmented state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProseUnit {
    /// Index into the owning [`Document`]'s `blocks` of the block this
    /// prose was extracted from.
    pub block: usize,
    /// Byte range of this prose unit in the original source.
    pub range: Range<usize>,
    /// Sentences within this prose unit, in source order.
    pub sentences: Vec<Sentence>,
}

impl ProseUnit {
    /// Creates a new prose unit.
    #[must_use]
    pub const fn new(block: usize, range: Range<usize>, sentences: Vec<Sentence>) -> Self {
        Self {
            block,
            range,
            sentences,
        }
    }

    /// Validates this prose unit against `source` and its owning
    /// document's `blocks`: `block` must index an existing block, `range`
    /// must lie within that block's range, and every sentence (and its
    /// tokens, transitively) must be contained within `range`.
    ///
    /// # Errors
    /// Returns the first [`CoreError`] found.
    pub fn validate(&self, source: &str, blocks: &[Block]) -> Result<(), CoreError> {
        span::validate_range(source, &self.range)?;
        let owner = blocks
            .get(self.block)
            .ok_or(CoreError::BlockIndexOutOfBounds {
                index: self.block,
                len: blocks.len(),
            })?;
        if !span::contains_range(&owner.range, &self.range) {
            return Err(CoreError::RangeNotContained {
                inner: self.range.clone(),
                outer: owner.range.clone(),
            });
        }
        for sentence in &self.sentences {
            sentence.validate(source, &self.range)?;
        }
        Ok(())
    }
}

impl Spanned for ProseUnit {
    fn range(&self) -> Range<usize> {
        self.range.clone()
    }
}

/// A single sentence within a [`ProseUnit`], segmented into tokens.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sentence {
    /// Byte range of this sentence in the original source.
    pub range: Range<usize>,
    /// Tokens within this sentence, in source order.
    pub tokens: Vec<Token>,
}

impl Sentence {
    /// Creates a new sentence.
    #[must_use]
    pub const fn new(range: Range<usize>, tokens: Vec<Token>) -> Self {
        Self { range, tokens }
    }

    /// Validates this sentence against `source` and its enclosing prose
    /// unit's `parent` range: `range` must lie within `source` and be
    /// contained in `parent`, and every token must be contained in
    /// `range`.
    ///
    /// # Errors
    /// Returns the first [`CoreError`] found.
    pub fn validate(&self, source: &str, parent: &Range<usize>) -> Result<(), CoreError> {
        span::validate_range(source, &self.range)?;
        if !span::contains_range(parent, &self.range) {
            return Err(CoreError::RangeNotContained {
                inner: self.range.clone(),
                outer: parent.clone(),
            });
        }
        for token in &self.tokens {
            token.validate(source, &self.range)?;
        }
        Ok(())
    }
}

impl Spanned for Sentence {
    fn range(&self) -> Range<usize> {
        self.range.clone()
    }
}

/// A single lexical token within a [`Sentence`].
///
/// `Token` carries only a span and a coarse lexical [`TokenKind`]; richer
/// linguistic annotation (POS tags, lemmas, dependency edges) is layered on
/// top by `friction-nlp` rather than stored here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    /// Byte range of this token in the original source.
    pub range: Range<usize>,
    /// Coarse lexical classification of this token.
    pub kind: TokenKind,
}

impl Token {
    /// Creates a new token.
    #[must_use]
    pub const fn new(range: Range<usize>, kind: TokenKind) -> Self {
        Self { range, kind }
    }

    /// Validates this token against `source` and its enclosing sentence's
    /// `parent` range.
    ///
    /// # Errors
    /// Returns the first [`CoreError`] found.
    pub fn validate(&self, source: &str, parent: &Range<usize>) -> Result<(), CoreError> {
        span::validate_range(source, &self.range)?;
        if !span::contains_range(parent, &self.range) {
            return Err(CoreError::RangeNotContained {
                inner: self.range.clone(),
                outer: parent.clone(),
            });
        }
        Ok(())
    }
}

impl Spanned for Token {
    fn range(&self) -> Range<usize> {
        self.range.clone()
    }
}

/// Coarse lexical classification of a [`Token`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TokenKind {
    /// An alphabetic word, possibly hyphenated or contracted.
    Word,
    /// A numeric literal.
    Number,
    /// A punctuation mark.
    Punctuation,
    /// Inter-token whitespace.
    Whitespace,
    /// A symbol not covered by another variant (e.g. `&`, `%`, an emoji).
    Symbol,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::BlockKind;

    fn word(range: Range<usize>) -> Token {
        Token::new(range, TokenKind::Word)
    }

    /// A well-formed document round-trips its source and structure through
    /// `Document::new`.
    #[test]
    fn document_new_accepts_well_formed_structure() {
        let source = "Hello world. Bye.";
        let blocks = vec![Block::new(BlockKind::Paragraph, 0..source.len())];
        let sentence1 = Sentence::new(0..12, vec![word(0..5), word(6..11)]);
        let sentence2 = Sentence::new(13..17, vec![word(13..16)]);
        let prose = vec![ProseUnit::new(
            0,
            0..source.len(),
            vec![sentence1, sentence2],
        )];

        let doc = Document::new(source, blocks, prose).unwrap();
        assert_eq!(doc.source(), source);
        assert_eq!(doc.blocks().len(), 1);
        assert_eq!(doc.prose()[0].sentences.len(), 2);
        assert_eq!(doc.text(&(0..5)).unwrap(), "Hello");
    }

    /// A block range past the end of the source is rejected.
    #[test]
    fn document_new_rejects_out_of_bounds_block() {
        let source = "short";
        let blocks = vec![Block::new(BlockKind::Paragraph, 0..100)];
        let err = Document::new(source, blocks, Vec::new()).unwrap_err();
        assert!(matches!(err, CoreError::RangeOutOfBounds { .. }));
    }

    /// A `ProseUnit` referencing a non-existent block index is rejected.
    #[test]
    fn document_new_rejects_dangling_block_index() {
        let source = "hello";
        let blocks = vec![Block::new(BlockKind::Paragraph, 0..5)];
        let prose = vec![ProseUnit::new(3, 0..5, Vec::new())];
        let err = Document::new(source, blocks, prose).unwrap_err();
        assert!(matches!(
            err,
            CoreError::BlockIndexOutOfBounds { index: 3, len: 1 }
        ));
    }

    /// A prose unit whose range escapes its owning block's range is
    /// rejected.
    #[test]
    fn document_new_rejects_prose_escaping_block() {
        let source = "hello world";
        let blocks = vec![Block::new(BlockKind::Paragraph, 0..5)];
        let prose = vec![ProseUnit::new(0, 0..11, Vec::new())];
        let err = Document::new(source, blocks, prose).unwrap_err();
        assert!(matches!(err, CoreError::RangeNotContained { .. }));
    }

    /// A sentence whose range escapes its prose unit's range is rejected.
    #[test]
    fn prose_unit_validate_rejects_sentence_escaping_parent() {
        let source = "hello world";
        let blocks = vec![Block::new(BlockKind::Paragraph, 0..11)];
        let sentence = Sentence::new(0..11, Vec::new());
        let unit = ProseUnit::new(0, 0..5, vec![sentence]);
        let err = unit.validate(source, &blocks).unwrap_err();
        assert!(matches!(err, CoreError::RangeNotContained { .. }));
    }

    /// A token whose range escapes its sentence's range is rejected.
    #[test]
    fn sentence_validate_rejects_token_escaping_parent() {
        let source = "hello world";
        let sentence = Sentence::new(0..5, vec![word(0..11)]);
        let err = sentence.validate(source, &(0..11)).unwrap_err();
        assert!(matches!(err, CoreError::RangeNotContained { .. }));
    }

    /// `Spanned::range` is available uniformly across prose levels.
    #[test]
    fn spanned_trait_covers_prose_levels() {
        let sentence = Sentence::new(2..8, vec![word(2..5)]);
        let unit = ProseUnit::new(0, 0..10, vec![sentence.clone()]);
        assert_eq!(unit.range(), 0..10);
        assert_eq!(sentence.range(), 2..8);
        assert_eq!(word(2..5).range(), 2..5);
    }
}
