//! Part-of-speech tagging: enriching sentence text with tagged
//! [`friction_core::Token`]s.
//!
//! [`Tagger`] is the boundary between this crate and its
//! implementation(s) — currently a single one, [`crate::NlpruleTagger`], in
//! [`crate::tag_nlprule`] — so other crates only ever need to depend on the
//! trait. Unlike [`crate::Segmenter`], a `Tagger` produces the token spans
//! themselves (not just enriches pre-existing ones): tokenization and POS
//! tagging are the same pass in every backend worth having, so splitting
//! them into two trait methods would only force implementations to agree
//! on a token boundary contract neither of them needs independently.

use friction_core::{Token, TokenKind};

/// Splits sentence text into tokens and tags each one with a
/// part-of-speech and a lemma.
///
/// Implementations receive the text to tag together with `base_offset`,
/// the byte position at which `text` begins in some larger source; every
/// token's range in the returned [`Vec`] is already shifted by
/// `base_offset`, so callers slice the *original* source with it directly
/// rather than `text`. Tokens are returned in source order.
///
/// Implementations must be deterministic: identical `text` and
/// `base_offset` always produce identical output, on any machine, on any
/// run.
pub trait Tagger {
    /// Tags every token found in `text`, each offset by `base_offset`.
    fn tag(&self, text: &str, base_offset: usize) -> Vec<TaggedToken>;
}

/// A part-of-speech tag.
///
/// A thin newtype over the tagger's own tag string (for
/// [`crate::NlpruleTagger`], the Penn-Treebank-style tags of its
/// dictionary, e.g. `"NN"`, `"VBZ"`, `"JJ"`) rather than a closed enum:
/// different taggers use different tagsets, and callers that only care
/// about a handful of coarse categories can match on
/// [`PosTag::as_str`] themselves.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PosTag(Box<str>);

impl PosTag {
    /// Wraps `tag` as a [`PosTag`].
    pub fn new(tag: impl Into<Box<str>>) -> Self {
        Self(tag.into())
    }

    /// The tag text.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// The sentinel tag used when a tagger has no dictionary entry (and no
    /// other usable candidate) for a token's surface text. Callers should
    /// treat this as "unclassified", not as a real syntactic category.
    #[must_use]
    pub fn unknown() -> Self {
        Self(Box::from("UNKNOWN"))
    }
}

impl std::fmt::Display for PosTag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// A [`Token`] enriched with a part-of-speech tag and a lemma.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaggedToken {
    /// The underlying token: byte span into the original source, plus its
    /// coarse lexical [`TokenKind`].
    pub token: Token,
    /// The token's part-of-speech tag.
    pub pos: PosTag,
    /// The token's dictionary (base) form, lowercase. Equal to the token's
    /// own surface text (case-folded) when the tagger has no lemma for it.
    pub lemma: Box<str>,
}

/// Classifies `text` — a single token's surface text — into a coarse
/// [`TokenKind`], the same classification every [`Tagger`] implementation
/// in this crate should use so `TokenKind` means the same thing regardless
/// of which one produced it.
///
/// This is a pure function of `text`'s characters (`char::is_alphabetic`,
/// `char::is_numeric`; no locale, no ambient state), so it is deterministic
/// by construction. An empty `text` classifies as [`TokenKind::Symbol`],
/// the least specific catch-all; well-behaved tokenizers never emit empty
/// token text.
#[must_use]
pub fn classify_token_kind(text: &str) -> TokenKind {
    let Some(first) = text.chars().next() else {
        return TokenKind::Symbol;
    };
    if text.chars().all(char::is_whitespace) {
        return TokenKind::Whitespace;
    }
    if first.is_numeric() && text.chars().all(|c| c.is_numeric() || c == '.' || c == ',') {
        return TokenKind::Number;
    }
    if text
        .chars()
        .all(|c| c.is_alphabetic() || c == '\'' || c == '-')
        && text.chars().any(char::is_alphabetic)
    {
        return TokenKind::Word;
    }
    if text.chars().all(is_prose_punctuation) {
        return TokenKind::Punctuation;
    }
    TokenKind::Symbol
}

/// A character used as ordinary prose punctuation — the narrower set
/// [`friction_core::TokenKind::Punctuation`]'s doc comment describes
/// (`.`, `,`, `!`, quotes, brackets, dashes...) as opposed to
/// [`friction_core::TokenKind::Symbol`]'s examples (`&`, `%`, an emoji):
/// those are common in prose but are not "a punctuation mark" in the
/// sense the surrounding sentence-structure rules care about.
const fn is_prose_punctuation(c: char) -> bool {
    matches!(
        c,
        '.' | ','
            | ';'
            | ':'
            | '!'
            | '?'
            | '\''
            | '"'
            | '('
            | ')'
            | '['
            | ']'
            | '{'
            | '}'
            | '-'
            | '\u{2013}'
            | '\u{2014}'
            | '\u{2026}'
            | '/'
            | '\u{2018}'
            | '\u{2019}'
            | '\u{201C}'
            | '\u{201D}'
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_token_kind_recognizes_words() {
        assert_eq!(classify_token_kind("leverage"), TokenKind::Word);
        assert_eq!(classify_token_kind("Leveraging"), TokenKind::Word);
        assert_eq!(classify_token_kind("don't"), TokenKind::Word);
        assert_eq!(classify_token_kind("state-of-the-art"), TokenKind::Word);
    }

    #[test]
    fn classify_token_kind_recognizes_numbers() {
        assert_eq!(classify_token_kind("42"), TokenKind::Number);
        assert_eq!(classify_token_kind("3.14"), TokenKind::Number);
        assert_eq!(classify_token_kind("1,000"), TokenKind::Number);
    }

    #[test]
    fn classify_token_kind_recognizes_punctuation() {
        assert_eq!(classify_token_kind(","), TokenKind::Punctuation);
        assert_eq!(classify_token_kind("."), TokenKind::Punctuation);
        assert_eq!(classify_token_kind("--"), TokenKind::Punctuation);
    }

    #[test]
    fn classify_token_kind_recognizes_whitespace() {
        assert_eq!(classify_token_kind(" "), TokenKind::Whitespace);
        assert_eq!(classify_token_kind("\t\n"), TokenKind::Whitespace);
    }

    #[test]
    fn classify_token_kind_falls_back_to_symbol() {
        assert_eq!(classify_token_kind("&"), TokenKind::Symbol);
        assert_eq!(classify_token_kind("%"), TokenKind::Symbol);
        assert_eq!(classify_token_kind(""), TokenKind::Symbol);
    }

    #[test]
    fn pos_tag_wraps_and_displays_its_text() {
        let tag = PosTag::new("VBZ");
        assert_eq!(tag.as_str(), "VBZ");
        assert_eq!(tag.to_string(), "VBZ");
        assert_eq!(PosTag::unknown().as_str(), "UNKNOWN");
    }
}
