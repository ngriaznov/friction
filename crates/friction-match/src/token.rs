//! Span-carrying tokenization and the one allowlisted prose-scope gate.
//!
//! Two separate concerns live in this module, both described in this
//! crate's own top-level docs:
//!
//! - [`tokenize_str`]: a tiny word/punctuation tokenizer that runs
//!   directly against original, mixed-case source bytes and folds case and
//!   curly quotes at match time, so every returned range is already a
//!   valid range into the original source, never a range into some cleaned
//!   or rewritten copy. It is *behaviorally* pinned to
//!   `friction_harness::clean::tokenize`'s token-boundary convention by a
//!   dev-dependency equivalence test (`tests/token_convention.rs`), not by
//!   sharing that function's regex object: the two operate on different
//!   inputs (pre-lowercased whole-file text vs. raw, mixed-case,
//!   per-prose-block text) and are solving the same boundary problem for
//!   different callers.
//! - [`prose_scope`]: the single place in this crate that walks
//!   [`Document::blocks`]/[`Document::prose`] directly. Every detection
//!   channel consumes its output ([`ScopedUnit`]) and never touches
//!   [`Document`] itself, so the prose-only guarantee (headings, table
//!   structure, code, and any future non-prose block kind never produce a
//!   span) is enforced exactly once, centrally.

use std::ops::Range;
use std::sync::LazyLock;

use friction_core::{BlockKind, Document, ProseUnit};
use friction_nlp::Segmenter;
use regex::Regex;

/// `[A-Za-z\u{2018}\u{2019}']+|[.,;:!?]` — a word run (ASCII letters plus
/// straight or curly single quotes) or a single punctuation character.
///
/// Deliberately not `(?i)`: both cases of every ASCII letter are named
/// explicitly in the class, so no Unicode case-folding machinery is
/// engaged — this matches `clean::tokenize`'s own `[a-z']+` acting on
/// already-lowercased text exactly, rather than merely approximating it.
static WORD_OR_PUNCTUATION: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"[A-Za-z\u{2018}\u{2019}']+|[.,;:!?]").expect("token pattern is valid")
});

/// One word or single-punctuation-character token from the DMS/literal
/// analysis view.
///
/// Carries its exact byte range in the ORIGINAL document source. Text is
/// case-folded (curly single quotes folded to `'`) but the range always
/// addresses the untouched original bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnalysisToken {
    /// Byte range of this token in the original document source.
    pub range: Range<usize>,
    /// The token's case-folded, curly-quote-folded text — matches the
    /// vocabulary strings a DMS pack's own tokenization convention
    /// produces, not necessarily the exact original-source bytes at
    /// `range`.
    pub text: Box<str>,
    /// Coarse word/punctuation classification.
    pub kind: AnalysisTokenKind,
}

/// Which of the two token shapes [`tokenize_str`] recognizes an
/// [`AnalysisToken`] as.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnalysisTokenKind {
    /// A run of one or more letters/apostrophes.
    Word,
    /// A single `.,;:!?` character.
    Punctuation,
}

/// Folds `raw` (already known to be either a `WORD_OR_PUNCTUATION` word or
/// punctuation match) into its analysis text: curly single quotes folded
/// to `'`, ASCII letters lowercased.
fn fold_token_text(raw: &str) -> Box<str> {
    raw.chars()
        .map(|c| match c {
            '\u{2018}' | '\u{2019}' => '\'',
            other => other.to_ascii_lowercase(),
        })
        .collect::<String>()
        .into_boxed_str()
}

/// `true` if `raw` is a single terminal-punctuation character — the other
/// arm of [`WORD_OR_PUNCTUATION`]'s alternation, which is otherwise a run
/// of word characters.
fn is_punctuation_match(raw: &str) -> bool {
    let mut chars = raw.chars();
    matches!(chars.next(), Some('.' | ',' | ';' | ':' | '!' | '?')) && chars.next().is_none()
}

/// Tokenizes `text` into word-run (`[a-z']+`, curly quotes folded) and
/// single-terminal-punctuation (`.,;:!?`) tokens.
///
/// Matches every match boundary `friction_harness::clean::tokenize` would
/// produce for the same prose content — see `tests/token_convention.rs`
/// for the pinned equivalence. `text` is assumed already prose-extracted
/// (no fenced/inline code, no raw link URLs, friction-parse's contract).
/// Every returned range is shifted by `base_offset`.
#[must_use]
pub fn tokenize_str(text: &str, base_offset: usize) -> Vec<AnalysisToken> {
    WORD_OR_PUNCTUATION
        .find_iter(text)
        .map(|m| {
            let raw = m.as_str();
            let kind = if is_punctuation_match(raw) {
                AnalysisTokenKind::Punctuation
            } else {
                AnalysisTokenKind::Word
            };
            AnalysisToken {
                range: (base_offset + m.start())..(base_offset + m.end()),
                text: fold_token_text(raw),
                kind,
            }
        })
        .collect()
}

/// `true` for the block kinds detection may touch (gate: prose blocks
/// only). Allowlist, not blocklist — `BlockKind` is `#[non_exhaustive]`.
#[must_use]
pub const fn is_in_scope(kind: &BlockKind) -> bool {
    matches!(
        kind,
        BlockKind::Paragraph
            | BlockKind::BlockQuote
            | BlockKind::ListItem
            | BlockKind::FootnoteDefinition { .. }
    )
}

/// One in-scope prose unit, pre-segmented into sentence byte ranges and
/// pre-tokenized for the DMS/literal channels.
pub struct ScopedUnit<'doc> {
    /// The prose unit this scope was built from.
    pub unit: &'doc ProseUnit,
    /// This unit's text, i.e. `document.text(&unit.range)`.
    pub text: &'doc str,
    /// Sentence byte ranges within this unit, absolute into the original
    /// source (see [`Segmenter`]'s own contract).
    pub sentences: Vec<Range<usize>>,
    /// This unit's tokens, absolute into the original source.
    pub tokens: Vec<AnalysisToken>,
}

/// Filters `document.prose()` down to in-scope blocks ([`is_in_scope`]),
/// segments each surviving unit's text with `segmenter`, and tokenizes
/// its full text with [`tokenize_str`].
///
/// The ONE place the prose-only guarantee is enforced — every channel
/// iterates only over this function's output, never `document.prose()`
/// directly.
#[must_use]
pub fn prose_scope<'doc>(
    document: &'doc Document,
    segmenter: &dyn Segmenter,
) -> Vec<ScopedUnit<'doc>> {
    document
        .prose()
        .iter()
        .filter(|unit| {
            document
                .blocks()
                .get(unit.block)
                .is_some_and(|block| is_in_scope(&block.kind))
        })
        .map(|unit| {
            let text = document
                .text(&unit.range)
                .expect("ProseUnit ranges are already validated by Document::new");
            let sentences = segmenter.segment(text, unit.range.start);
            let tokens = tokenize_str(text, unit.range.start);
            ScopedUnit {
                unit,
                text,
                sentences,
                tokens,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenize_str_splits_words_and_punctuation_case_folded() {
        let tokens = tokenize_str("Don't Stop, Please.", 0);
        let texts: Vec<&str> = tokens.iter().map(|t| &*t.text).collect();
        assert_eq!(texts, vec!["don't", "stop", ",", "please", "."]);
    }

    #[test]
    fn tokenize_str_folds_curly_quotes_but_keeps_original_byte_range() {
        let text = "it\u{2019}s fine";
        let tokens = tokenize_str(text, 0);
        assert_eq!(&*tokens[0].text, "it's");
        // The curly apostrophe is a 3-byte UTF-8 character, so the token's
        // original-source range is longer than its folded text.
        assert_eq!(tokens[0].range, 0..text.find(' ').unwrap());
        assert_eq!(&text[tokens[0].range.clone()], "it\u{2019}s");
    }

    #[test]
    fn tokenize_str_shifts_every_range_by_base_offset() {
        let tokens = tokenize_str("word", 10);
        assert_eq!(tokens[0].range, 10..14);
    }

    #[test]
    fn is_in_scope_allowlists_only_prose_block_kinds() {
        assert!(is_in_scope(&BlockKind::Paragraph));
        assert!(is_in_scope(&BlockKind::BlockQuote));
        assert!(is_in_scope(&BlockKind::ListItem));
        assert!(is_in_scope(&BlockKind::FootnoteDefinition {
            label: "1".into()
        }));
        assert!(!is_in_scope(&BlockKind::Heading { level: 2 }));
        assert!(!is_in_scope(&BlockKind::Table));
        assert!(!is_in_scope(&BlockKind::TableRow));
        assert!(!is_in_scope(&BlockKind::TableCell { header: false }));
        assert!(!is_in_scope(&BlockKind::CodeBlock { info: None }));
        assert!(!is_in_scope(&BlockKind::HtmlBlock));
        assert!(!is_in_scope(&BlockKind::ThematicBreak));
        assert!(!is_in_scope(&BlockKind::List {
            ordered: false,
            start: None
        }));
        assert!(!is_in_scope(&BlockKind::Other("frontmatter".into())));
    }

    #[test]
    fn prose_scope_excludes_heading_blocks() {
        let document = friction_parse::parse("## Heading\n\nParagraph text.\n")
            .expect("valid markdown parses");
        let segmenter = friction_nlp::SrxSegmenter::new();
        let units = prose_scope(&document, &segmenter);
        assert_eq!(units.len(), 1);
        assert_eq!(units[0].text, "Paragraph text.");
    }
}
