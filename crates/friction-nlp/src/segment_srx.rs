//! SRX-rule-based [`Segmenter`] implementation.

use std::ops::Range;
use std::sync::OnceLock;

use srx::SRX;

use crate::segment::Segmenter;

/// The vendored English SRX ruleset, embedded at compile time.
///
/// See `data/friction-en.srx` (in this crate) for the ruleset itself, its
/// rule-by-rule rationale, and its license header. Embedding it via
/// `include_str!` means parsing it needs no file or network access at
/// build or run time — the bytes are baked into the compiled binary.
const RULESET_XML: &str = include_str!("../data/friction-en.srx");

/// Parses [`RULESET_XML`] and extracts its English rules, once, caching
/// the result for the lifetime of the process.
///
/// # Panics
/// Panics if [`RULESET_XML`] fails to parse as valid SRX or contains a
/// rule whose regex fails to compile. Both are invariants of the vendored
/// file this crate ships and controls; `segment_srx::tests::ruleset_parses`
/// below exercises exactly this parse so a broken ruleset fails CI rather
/// than surfacing here at first use.
fn english_rules() -> &'static srx::Rules {
    static RULES: OnceLock<srx::Rules> = OnceLock::new();
    RULES.get_or_init(|| {
        let srx: SRX = RULESET_XML
            .parse()
            .expect("vendored data/friction-en.srx is well-formed SRX with valid rule regexes");
        srx.language_rules("en")
    })
}

/// Sentence segmentation via a small, self-authored SRX (Segmentation
/// Rules eXchange) 2.0 ruleset for English, vendored under this crate's
/// `data/` directory and embedded at compile time.
///
/// Handles the common false-positive sources for naive
/// terminal-punctuation splitting: abbreviations ("Dr.", "etc.", "e.g."),
/// initials and letter-abbreviated acronyms ("J. R. R. Tolkien", "U.S."),
/// decimals ("3.14", never followed by whitespace so never a candidate
/// split point), and ellipses continuing into a lowercase word. See
/// `data/friction-en.srx`'s header comment for the full rule cascade and
/// its rationale, and `tests/segment_golden.rs` for the golden sentence
/// set it is verified against.
///
/// Stateless and `Copy`; construct with [`SrxSegmenter::new`] or
/// [`SrxSegmenter::default`].
#[derive(Debug, Clone, Copy, Default)]
pub struct SrxSegmenter;

impl SrxSegmenter {
    /// Creates a new SRX-backed segmenter.
    ///
    /// The ruleset itself is embedded in the binary; parsing it and
    /// compiling its rules' regexes happens lazily, once, on first use of
    /// any `SrxSegmenter` instance (see [`english_rules`]), not on every
    /// construction.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Segmenter for SrxSegmenter {
    fn segment(&self, text: &str, base_offset: usize) -> Vec<Range<usize>> {
        english_rules()
            .split_ranges(text)
            .into_iter()
            .filter_map(|range| trim_whitespace(text, range))
            .map(|range| base_offset + range.start..base_offset + range.end)
            .collect()
    }
}

/// Trims leading and trailing whitespace from `range` within `text`,
/// returning `None` if nothing but whitespace remains (the gap between
/// two paragraphs, or trailing whitespace after the last sentence).
///
/// The SRX split algorithm returns segments that include a sentence's
/// leading whitespace (the gap since the previous sentence, since the
/// break index sits at the *start* of that whitespace run); trimming
/// gives each sentence range the tight span a consumer expects, with no
/// leading or trailing space.
fn trim_whitespace(text: &str, range: Range<usize>) -> Option<Range<usize>> {
    let slice = text.get(range.clone())?;
    let leading = slice.len() - slice.trim_start().len();
    let start = range.start + leading;
    let end = start + slice.trim().len();
    (start < end).then_some(start..end)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The vendored ruleset parses and yields a non-empty English rule
    /// set; the sole guard against [`english_rules`]'s panic ever firing
    /// in a real build.
    #[test]
    fn ruleset_parses() {
        assert!(!english_rules().is_empty());
    }

    /// A single unpunctuated sentence segments to itself, trimmed.
    #[test]
    fn segments_single_sentence() {
        let segmenter = SrxSegmenter::new();
        let ranges = segmenter.segment("Hello world", 0);
        assert_eq!(ranges, vec![0..11]);
    }

    /// Two sentences split at the terminal period, each range excluding
    /// the whitespace between them.
    #[test]
    fn segments_two_sentences_without_leading_whitespace() {
        let segmenter = SrxSegmenter::new();
        let text = "First one. Second one.";
        let ranges = segmenter.segment(text, 0);
        assert_eq!(ranges.len(), 2);
        assert_eq!(&text[ranges[0].clone()], "First one.");
        assert_eq!(&text[ranges[1].clone()], "Second one.");
    }

    /// `base_offset` shifts every returned range, so it addresses a
    /// larger source the input text was sliced from.
    #[test]
    fn segment_applies_base_offset() {
        let segmenter = SrxSegmenter::new();
        let ranges = segmenter.segment("One. Two.", 100);
        assert_eq!(ranges, vec![100..104, 105..109]);
    }

    /// A known abbreviation followed by a capitalized word does not
    /// split.
    #[test]
    fn does_not_split_on_abbreviation() {
        let segmenter = SrxSegmenter::new();
        let ranges = segmenter.segment("I saw Dr. Smith today.", 0);
        assert_eq!(ranges.len(), 1);
    }

    /// Empty input yields no sentences.
    #[test]
    fn segments_empty_text_to_nothing() {
        let segmenter = SrxSegmenter::new();
        assert!(segmenter.segment("", 0).is_empty());
    }

    /// A digit following one of the ambiguous abbreviation/word tokens
    /// still suppresses the break ("No. 4" stays "No. 4", not "No." +
    /// "4").
    #[test]
    fn does_not_split_ambiguous_abbreviation_before_digit() {
        let segmenter = SrxSegmenter::new();
        let text = "Please review item No. 4 first. It affects the rest.";
        let ranges = segmenter.segment(text, 0);
        let sentences: Vec<&str> = ranges.iter().map(|r| &text[r.clone()]).collect();
        assert_eq!(
            sentences,
            vec!["Please review item No. 4 first.", "It affects the rest."]
        );
    }

    /// Tokens that double as ordinary standalone English words or names
    /// ("no", "Ed", "co", "fig") still end a sentence when a capitalized
    /// word follows, instead of being swallowed by the abbreviation
    /// no-break list.
    #[test]
    fn splits_ambiguous_abbreviation_tokens_used_as_ordinary_words() {
        let segmenter = SrxSegmenter::new();
        let cases: &[(&str, &[&str])] = &[
            (
                "Is this the answer? No. It is not.",
                &["Is this the answer?", "No.", "It is not."],
            ),
            (
                "Did it work? No. We need to try again.",
                &["Did it work?", "No.", "We need to try again."],
            ),
            (
                "The answer was no. The next question was harder.",
                &["The answer was no.", "The next question was harder."],
            ),
            (
                "I ran into Ed. He was in a hurry.",
                &["I ran into Ed.", "He was in a hurry."],
            ),
            (
                "We visited the co. It was closed.",
                &["We visited the co.", "It was closed."],
            ),
            (
                "She works at the fig. It grows well here.",
                &["She works at the fig.", "It grows well here."],
            ),
        ];

        for (text, expect) in cases {
            let ranges = segmenter.segment(text, 0);
            let sentences: Vec<&str> = ranges.iter().map(|r| &text[r.clone()]).collect();
            assert_eq!(&sentences, expect, "input: {text:?}");
        }
    }
}
