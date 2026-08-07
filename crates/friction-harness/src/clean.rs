//! Shared string-level cleaning and tokenization conventions.
//!
//! This is the exact normalization the fixture expectations in
//! `docs/research/` were produced under, pinned here so scoring uses the
//! same text shape the fixtures were validated against.
//!
//! This is the *string-level* half of the harness: [`tellspan`](crate::tellspan),
//! [`pivot`](crate::pivot), and [`closure`](crate::closure) all analyze text
//! through this module rather than through `friction-parse`'s real block tree:
//! the fixtures are string-level contracts, deliberately independent of any one
//! Markdown parser. [`fragment`](crate::fragment) is the one module in this crate
//! that deliberately does *not* use it. See that module's own docs.

use std::sync::LazyLock;

use friction_core::token_class::{LOWERCASE_WORD_APOSTROPHE_ASCII, PROSE_PUNCTUATION};
use regex::Regex;

/// `` (?s)```.*?``` `` — fenced code blocks, matched non-greedily across
/// newlines.
static FENCED_CODE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?s)```.*?```").expect("fenced-code pattern is valid"));

/// `` `[^`]*` `` — inline code spans.
static INLINE_CODE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"`[^`]*`").expect("inline-code pattern is valid"));

/// `\[([^\]]*)\]\([^)]*\)` — markdown links, replaced by their anchor text.
static LINKS: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\[([^\]]*)\]\([^)]*\)").expect("link pattern is valid"));

/// `[#>*_|]` — leftover markdown syntax characters, stripped to a space.
static SYNTAX_CHARS: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[#>*_|]").expect("syntax-char pattern is valid"));

/// A lowercase word run or a single punctuation character. Built from
/// `friction_core::token_class`'s shared
/// [`LOWERCASE_WORD_APOSTROPHE_ASCII`]/[`PROSE_PUNCTUATION`] fragments —
/// the single source of truth this pattern's shape stays in lockstep with
/// `friction_match::token::WORD_OR_PUNCTUATION` and
/// `friction_packs::validate::WORD_TOKEN` through (see that module's docs
/// for why all three must agree, and for the `_UNICODE` alternates staged
/// for a future non-English language: switching this site to
/// `LOWERCASE_WORD_APOSTROPHE_UNICODE` was measured against the full
/// corpus byte-fence and reverted — the English corpus carries
/// tokenization-visible non-ASCII letters in real prose, e.g. accented
/// proper nouns, so `\p{Ll}` moved `check`'s token counts on ~50 files).
static TOKEN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(&format!(
        "[{LOWERCASE_WORD_APOSTROPHE_ASCII}]+|[{PROSE_PUNCTUATION}]"
    ))
    .expect("token pattern is valid")
});

/// Case-preserving cleaning: curly single quotes to straight, fenced and
/// inline code stripped, links replaced by their anchor text, and the
/// leftover markdown syntax characters `#>*_|` stripped.
///
/// Normalizes only curly *single* quotes (`'`, `'`), not curly double
/// quotes: the fixture expectations were produced under exactly that
/// asymmetry, so "fixing" it here would silently invalidate them.
#[must_use]
pub fn clean(text: &str) -> String {
    let text = text.replace(['\u{2019}', '\u{2018}'], "'");
    let text = FENCED_CODE.replace_all(&text, " ");
    let text = INLINE_CODE.replace_all(&text, " ");
    let text = LINKS.replace_all(&text, "$1");
    SYNTAX_CHARS.replace_all(&text, " ").into_owned()
}

/// `clean(text)`, lowercased, split into `[a-z']+` word tokens and
/// `.,;:!?` punctuation tokens.
///
/// Mirrors `toks()` / `re.findall(r"[a-z']+|[.,;:!?]", text.lower())`.
#[must_use]
pub fn tokenize(text: &str) -> Vec<String> {
    let cleaned = clean(text).to_lowercase();
    TOKEN
        .find_iter(&cleaned)
        .map(|m| m.as_str().to_string())
        .collect()
}

/// `true` for a single `.,;:!?` character token.
#[must_use]
pub fn is_punctuation_token(token: &str) -> bool {
    let mut chars = token.chars();
    matches!(chars.next(), Some('.' | ',' | ';' | ':' | '!' | '?')) && chars.next().is_none()
}

/// Lookaround-free reimplementation of `re.split(r"(?<=[.!?])\s+", text)`.
///
/// Splits after a run of `.!?` that is followed by whitespace or
/// end-of-string, keeping the terminal punctuation with the preceding
/// sentence. Rust's `regex` crate has no lookbehind, so this is a small manual scan
/// over `char_indices` rather than a regex. ASCII terminal punctuation and
/// whitespace bytes are never continuation bytes of a multi-byte UTF-8
/// character, so the byte positions this function slices `text` with are
/// always char-boundary-safe.
#[must_use]
pub fn split_sentences(text: &str) -> Vec<&str> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let n = chars.len();
    let mut result = Vec::new();
    let mut start = 0usize;
    let mut i = 0usize;

    while i < n {
        if !matches!(chars[i].1, '.' | '!' | '?') {
            i += 1;
            continue;
        }
        // Consume the whole run of terminal punctuation.
        let mut j = i + 1;
        while j < n && matches!(chars[j].1, '.' | '!' | '?') {
            j += 1;
        }
        if j == n {
            // The run reaches end-of-string: the final sentence ends here,
            // nothing left to scan.
            break;
        }
        if chars[j].1.is_whitespace() {
            let run_end_byte = chars[j].0;
            result.push(&text[start..run_end_byte]);
            let mut k = j;
            while k < n && chars[k].1.is_whitespace() {
                k += 1;
            }
            start = if k < n { chars[k].0 } else { text.len() };
            i = k;
        } else {
            i = j;
        }
    }

    if start < text.len() {
        result.push(&text[start..]);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_normalizes_curly_single_quotes() {
        assert_eq!(clean("it\u{2019}s a \u{2018}test\u{2019}"), "it's a 'test'");
    }

    #[test]
    fn clean_strips_fenced_and_inline_code() {
        assert_eq!(clean("before ```code here``` after"), "before   after");
        assert_eq!(clean("use `foo()` please"), "use   please");
    }

    #[test]
    fn clean_replaces_links_with_anchor_text() {
        assert_eq!(
            clean("see [the docs](https://example.com) now"),
            "see the docs now"
        );
    }

    #[test]
    fn clean_strips_markdown_syntax_characters() {
        assert_eq!(
            clean("# Heading\n> quote *em* _also_ a|b"),
            "  Heading\n  quote  em   also  a b"
        );
    }

    #[test]
    fn tokenize_splits_words_and_punctuation() {
        assert_eq!(
            tokenize("Don't stop, please."),
            vec!["don't", "stop", ",", "please", "."]
        );
    }

    #[test]
    fn is_punctuation_token_recognizes_single_punctuation_chars() {
        for p in [".", ",", ";", ":", "!", "?"] {
            assert!(is_punctuation_token(p));
        }
        assert!(!is_punctuation_token("word"));
        assert!(!is_punctuation_token(".."));
        assert!(!is_punctuation_token(""));
    }

    #[test]
    fn split_sentences_splits_on_terminal_punctuation_and_whitespace() {
        let text = "First one. Second one!  Third?";
        assert_eq!(
            split_sentences(text),
            vec!["First one.", "Second one!", "Third?"]
        );
    }

    #[test]
    fn split_sentences_does_not_split_decimals() {
        let text = "Pi is about 3.14 today.";
        assert_eq!(split_sentences(text), vec!["Pi is about 3.14 today."]);
    }

    #[test]
    fn split_sentences_handles_empty_text() {
        assert!(split_sentences("").is_empty());
    }

    #[test]
    fn split_sentences_handles_no_trailing_punctuation() {
        let text = "One. Trailing fragment with no period";
        assert_eq!(
            split_sentences(text),
            vec!["One.", "Trailing fragment with no period"]
        );
    }
}
