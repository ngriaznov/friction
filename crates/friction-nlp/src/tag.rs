//! Part-of-speech tagging: enriching sentence text with tagged
//! [`friction_core::Token`]s.
//!
//! [`Tagger`] is the boundary between this crate and its
//! implementation(s) — currently just [`crate::PerceptronTagger`], in
//! `tag_perceptron` — so other crates only need to depend on the trait.
//! Unlike [`crate::Segmenter`], a `Tagger` produces the token spans
//! themselves, not just enriches pre-existing ones: tokenization and POS
//! tagging are the same pass in every backend worth having, so splitting
//! them into two trait methods would only force implementations to agree
//! on a token boundary contract neither needs independently.

use friction_core::{Token, TokenKind};

/// Splits sentence text into tokens and tags each one with a
/// part-of-speech and a lemma.
///
/// Implementations receive `base_offset`, the byte position at which
/// `text` begins in some larger source; every returned token's range is
/// already shifted by `base_offset`, so callers slice the *original*
/// source with it, not `text`. Tokens are returned in source order.
///
/// Implementations must be deterministic: identical `text` and
/// `base_offset` always produce identical output, on any machine or run.
///
/// `Send + Sync`: callers tag independent sentences concurrently (rayon
/// per-sentence parallelism in `friction-edit`'s register pass and
/// `friction-match`'s per-channel scans) through a single shared `&dyn
/// Tagger`. [`crate::PerceptronTagger`] satisfies this trivially — every
/// per-call scratch buffer is owned by the `tag` call itself, never
/// `self`, so `&self` is safe to share across threads.
pub trait Tagger: Send + Sync {
    /// Tags every token found in `text`, each offset by `base_offset`.
    fn tag(&self, text: &str, base_offset: usize) -> Vec<TaggedToken>;
}

/// A part-of-speech tag.
///
/// A thin newtype over the tagger's own tag string (for
/// [`crate::PerceptronTagger`], Penn-Treebank-style, e.g. `"NN"`, `"VBZ"`,
/// `"JJ"`) rather than a closed enum: different taggers use different
/// tagsets, and callers that only care about coarse categories can match
/// on [`PosTag::as_str`] themselves.
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

/// The coarse part-of-speech tag for `pos`.
///
/// Its full tag truncated to the first two characters (`"VBZ"` -> `"VB"`,
/// `"NNS"` -> `"NN"`), except a tag whose first character isn't
/// alphabetic — a punctuation tag, kept verbatim (truncating to one
/// character would lose information it never had to begin with).
#[must_use]
pub fn coarse_tag(pos: &PosTag) -> Box<str> {
    let full = pos.as_str();
    let Some(first) = full.chars().next() else {
        return Box::from(full);
    };
    if !first.is_alphabetic() {
        return Box::from(full);
    }
    let end = full
        .char_indices()
        .nth(2)
        .map_or(full.len(), |(index, _)| index);
    Box::from(&full[..end])
}

/// Classifies `text` — a single token's surface text — into a coarse
/// [`TokenKind`].
///
/// The same classification every [`Tagger`] implementation in this crate
/// should use, so `TokenKind` means the same thing regardless of which
/// one produced it.
///
/// A pure function of `text`'s characters (`char::is_alphabetic`,
/// `char::is_numeric`; no locale, no ambient state), so deterministic by
/// construction. An empty `text` classifies as [`TokenKind::Symbol`], the
/// least specific catch-all; well-behaved tokenizers never emit empty
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
/// [`friction_core::TokenKind::Punctuation`] describes (`.`, `,`, `!`,
/// quotes, brackets, dashes...) as opposed to
/// [`friction_core::TokenKind::Symbol`]'s examples (`&`, `%`, an emoji),
/// which are common in prose but aren't "a punctuation mark" in the sense
/// sentence-structure rules care about.
pub const fn is_prose_punctuation(c: char) -> bool {
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
    fn coarse_tag_truncates_to_two_characters() {
        assert_eq!(&*coarse_tag(&PosTag::new("VBZ")), "VB");
        assert_eq!(&*coarse_tag(&PosTag::new("NNS")), "NN");
        assert_eq!(&*coarse_tag(&PosTag::new("MD")), "MD");
    }

    #[test]
    fn coarse_tag_keeps_punctuation_tags_verbatim() {
        assert_eq!(&*coarse_tag(&PosTag::new(",")), ",");
        assert_eq!(&*coarse_tag(&PosTag::new("...")), "...");
    }

    #[test]
    fn pos_tag_wraps_and_displays_its_text() {
        let tag = PosTag::new("VBZ");
        assert_eq!(tag.as_str(), "VBZ");
        assert_eq!(tag.to_string(), "VBZ");
        assert_eq!(PosTag::unknown().as_str(), "UNKNOWN");
    }
}
