//! [`RuleContext`]: the per-round handle every [`crate::Rule`]'s `scan` and
//! `fix` step reads from.

use std::collections::BTreeMap;

use friction_core::{Document, Envelope, ProseUnit, Sentence};
use friction_nlp::{TaggedToken, Tagger};

/// A read-only view of one genre's per-metric human envelope bands.
///
/// `friction-rules` only defines this trait; producing a real
/// implementation for a genre — typically by loading a versioned
/// `friction-packs` envelope pack and indexing its `[<genre>.<metric>]`
/// tables — is the job of whatever builds a round's [`RuleContext`]
/// (`friction-apply`'s driver, ultimately `friction-cli`). [`MapEnvelope`]
/// is a small in-memory implementation for tests, and a template for a
/// pack-backed one.
pub trait GenreEnvelope {
    /// The envelope band for `metric` in this view's genre, or `None` if
    /// the pack has no band for it (e.g. an unrecognized name, or a metric
    /// this genre had no train-split human data for).
    fn band(&self, metric: &str) -> Option<Envelope>;
}

/// A [`GenreEnvelope`] backed by an in-memory table.
#[derive(Debug, Clone, Default)]
pub struct MapEnvelope(BTreeMap<&'static str, Envelope>);

impl MapEnvelope {
    /// An empty table: every [`GenreEnvelope::band`] lookup returns `None`.
    #[must_use]
    pub const fn new() -> Self {
        Self(BTreeMap::new())
    }

    /// Adds (or replaces) `metric`'s band, returning `self` for chaining.
    #[must_use]
    pub fn with(mut self, metric: &'static str, envelope: Envelope) -> Self {
        self.0.insert(metric, envelope);
        self
    }
}

impl GenreEnvelope for MapEnvelope {
    fn band(&self, metric: &str) -> Option<Envelope> {
        self.0.get(metric).copied()
    }
}

/// Everything a [`crate::Rule`]'s `scan`/`fix` step needs.
///
/// Built once per `friction-apply` round: the round's
/// parsed-and-sentence-segmented document, a tagger handle for on-demand
/// part-of-speech lookups, the document's genre, and that genre's envelope
/// bands.
///
/// `document` must already have been run through
/// [`friction_nlp::segment_document`] — a `RuleContext` does not segment
/// on a rule's behalf, so every rule sees exactly the same sentence
/// boundaries for the round, computed once.
#[derive(Clone, Copy)]
pub struct RuleContext<'a> {
    document: &'a Document,
    tagger: &'a dyn Tagger,
    genre: &'a str,
    envelope: &'a dyn GenreEnvelope,
}

impl<'a> RuleContext<'a> {
    /// Builds a context from its parts. `document` is expected to already
    /// be sentence-segmented (see the type's own docs).
    #[must_use]
    pub fn new(
        document: &'a Document,
        tagger: &'a dyn Tagger,
        genre: &'a str,
        envelope: &'a dyn GenreEnvelope,
    ) -> Self {
        Self {
            document,
            tagger,
            genre,
            envelope,
        }
    }

    /// The round's document (source, block structure, and segmented
    /// prose).
    #[must_use]
    pub const fn document(&self) -> &'a Document {
        self.document
    }

    /// The tagger handle for on-demand part-of-speech lookups; see
    /// [`RuleContext::tag_sentence`] for the common case of tagging one
    /// sentence.
    #[must_use]
    pub fn tagger(&self) -> &'a dyn Tagger {
        self.tagger
    }

    /// The document's genre (e.g. `"blog"`, `"docs"`), matching the key
    /// space a `friction-packs` envelope pack uses for its
    /// `[<genre>.<metric>]` tables.
    #[must_use]
    pub const fn genre(&self) -> &'a str {
        self.genre
    }

    /// This genre's envelope bands.
    #[must_use]
    pub fn envelope(&self) -> &'a dyn GenreEnvelope {
        self.envelope
    }

    /// Tags one sentence on demand: slices its source text out of
    /// `document` and runs it through `tagger`, offset so every returned
    /// token's span is already a byte range into the original document
    /// source (not into the sentence's own local text).
    ///
    /// # Panics
    /// Panics if `sentence` did not come from this context's own
    /// `document` — every `Sentence` reachable from `document.prose()`
    /// was already span-validated when `document` was constructed, so this
    /// cannot happen for a `sentence` obtained via
    /// [`RuleContext::sentences`].
    #[must_use]
    pub fn tag_sentence(&self, sentence: &Sentence) -> Vec<TaggedToken> {
        let text = self.document.text(&sentence.range).expect(
            "a Sentence's range was already validated against its Document at construction",
        );
        self.tagger.tag(text, sentence.range.start)
    }

    /// Every `(prose unit, sentence)` pair in the document, in source
    /// order — the iteration shape most rules' `scan` needs.
    pub fn sentences(&self) -> impl Iterator<Item = (&'a ProseUnit, &'a Sentence)> {
        self.document
            .prose()
            .iter()
            .flat_map(|unit| unit.sentences.iter().map(move |sentence| (unit, sentence)))
    }
}

#[cfg(test)]
mod tests {
    use friction_nlp::PosTag;

    use super::*;

    /// A stub tagger that tags every whitespace-delimited run as a plain
    /// noun, enough to exercise `RuleContext` wiring without depending on
    /// the real nlprule-backed tagger.
    struct WordTagger;

    impl Tagger for WordTagger {
        fn tag(&self, text: &str, base_offset: usize) -> Vec<TaggedToken> {
            let mut tokens = Vec::new();
            let mut start = None;
            for (i, c) in text.char_indices() {
                if c.is_whitespace() {
                    if let Some(s) = start.take() {
                        tokens.push(word(text, base_offset, s, i));
                    }
                } else if start.is_none() {
                    start = Some(i);
                }
            }
            if let Some(s) = start {
                tokens.push(word(text, base_offset, s, text.len()));
            }
            tokens
        }
    }

    fn word(text: &str, base_offset: usize, start: usize, end: usize) -> TaggedToken {
        TaggedToken {
            token: friction_core::Token::new(
                (base_offset + start)..(base_offset + end),
                friction_core::TokenKind::Word,
            ),
            pos: PosTag::new("NN"),
            lemma: text[start..end].to_ascii_lowercase().into(),
        }
    }

    /// `MapEnvelope` returns bands it was built with and `None` for
    /// anything else.
    #[test]
    fn map_envelope_looks_up_known_and_unknown_metrics() {
        let envelope = MapEnvelope::new()
            .with("triad_rate", Envelope::new(0.0, 0.3))
            .with("em_dash_density", Envelope::new(0.0, 5.0));
        assert_eq!(envelope.band("triad_rate"), Some(Envelope::new(0.0, 0.3)));
        assert_eq!(
            envelope.band("em_dash_density"),
            Some(Envelope::new(0.0, 5.0))
        );
        assert_eq!(envelope.band("not_a_metric"), None);
    }

    /// `RuleContext::sentences` walks every prose unit's sentences, in
    /// source order, pairing each with its owning prose unit.
    #[test]
    fn sentences_iterates_every_sentence_in_source_order() {
        let source = "First sentence. Second one.\n\nA new paragraph starts here.\n";
        let parsed = friction_parse::parse(source).expect("valid markdown parses");
        let srx = friction_nlp::SrxSegmenter::new();
        let with_sentences =
            friction_nlp::segment_document(&parsed, &srx).expect("segmentation succeeds");

        let envelope = MapEnvelope::new();
        let ctx = RuleContext::new(&with_sentences, &WordTagger, "blog", &envelope);

        let ranges: Vec<_> = ctx.sentences().map(|(_, s)| s.range.clone()).collect();
        assert!(ranges.len() >= 3, "expected at least 3 sentences");
        for pair in ranges.windows(2) {
            assert!(
                pair[0].start < pair[1].start,
                "sentences must come out in source order"
            );
        }
        assert_eq!(ctx.genre(), "blog");
    }

    /// `tag_sentence` tags a sentence's text and shifts every token's span
    /// by the sentence's own offset into the document.
    #[test]
    fn tag_sentence_shifts_token_spans_by_sentence_offset() {
        let source = "Intro.\n\nWe shipped it.\n";
        let parsed = friction_parse::parse(source).expect("valid markdown parses");
        let srx = friction_nlp::SrxSegmenter::new();
        let with_sentences =
            friction_nlp::segment_document(&parsed, &srx).expect("segmentation succeeds");

        let envelope = MapEnvelope::new();
        let ctx = RuleContext::new(&with_sentences, &WordTagger, "docs", &envelope);

        let (_, second_sentence) = ctx
            .sentences()
            .nth(1)
            .expect("document has at least two sentences");
        let tokens = ctx.tag_sentence(second_sentence);
        assert!(!tokens.is_empty());
        for tagged in &tokens {
            assert!(
                friction_core::span::contains_range(&second_sentence.range, &tagged.token.range),
                "tagged token {:?} must stay within its sentence's range {:?}",
                tagged.token.range,
                second_sentence.range
            );
        }
    }
}
