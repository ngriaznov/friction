//! [`Tagger`] implementation backed by the `nlprule` crate's English
//! tokenizer/tagger model.
//!
//! The model binary this crate embeds is downloaded and sha256-verified by
//! `build.rs` at *build* time, decompressed into `OUT_DIR`, and pulled into
//! the compiled binary with `include_bytes!` — nothing here ever touches
//! the network or the filesystem at runtime, so a `friction` binary built
//! once works offline forever after.

use friction_core::Token;
use nlprule::Tokenizer as NlpruleTokenizer;

use crate::tag::{PosTag, TaggedToken, Tagger, classify_token_kind};

/// The decompressed nlprule English tokenizer/tagger model. `build.rs`
/// writes it to `OUT_DIR` (see the pinned URL and sha256 there); this
/// bakes it into the compiled crate.
static MODEL_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/en_tokenizer.bin"));

/// Errors constructing an [`NlpruleTagger`].
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum TagError {
    /// The embedded model bytes failed to deserialize into a tokenizer.
    /// This points at a build-time problem with the pinned model (a
    /// corrupted build, or a `nlprule` version mismatch between what
    /// `build.rs` downloaded and what this crate links against) — it is
    /// not a runtime or network condition, since the bytes are compiled
    /// in.
    #[error("failed to load the embedded English tagger model: {0}")]
    ModelLoad(#[source] nlprule::Error),
}

/// A [`Tagger`] backed by `nlprule`'s English POS tagger and lemmatizer.
///
/// Splits sentence text into tokens the same way `nlprule` splits text for
/// its own rule matching (word-ish runs, contractions and hyphenated
/// compounds handled by its dictionary, punctuation as separate tokens),
/// tags each one, and picks the first non-bookkeeping part-of-speech
/// candidate `nlprule` offers as the token's tag (its disambiguation pass
/// already reorders candidates so the contextually-preferred one comes
/// first when it can tell). A token `nlprule`'s dictionary has no entry
/// for gets [`PosTag::unknown`] and its own surface text as its lemma,
/// rather than an error: an out-of-vocabulary word is an ordinary tagging
/// outcome, not a missing-model condition.
pub struct NlpruleTagger {
    tokenizer: NlpruleTokenizer,
}

impl NlpruleTagger {
    /// Loads the tagger from its embedded model.
    ///
    /// Also makes a best-effort attempt to pin `nlprule`'s internal
    /// rayon-parallel disambiguation pass to a single thread, per this
    /// project's determinism discipline for model-backed inference. The
    /// pass is already order-stable regardless of thread count — `nlprule`
    /// picks the leftmost matching disambiguation rule via
    /// `ParallelIterator::find_first`, which does not depend on which
    /// thread finishes first — so a failed pin (for instance because some
    /// other crate already initialized the process-global rayon pool) can
    /// only cost throughput, never determinism, and is silently ignored.
    ///
    /// # Errors
    /// Returns [`TagError::ModelLoad`] if the embedded model bytes fail to
    /// deserialize.
    pub fn new() -> Result<Self, TagError> {
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build_global();
        let tokenizer = NlpruleTokenizer::from_reader(MODEL_BYTES).map_err(TagError::ModelLoad)?;
        Ok(Self { tokenizer })
    }
}

impl Tagger for NlpruleTagger {
    fn tag(&self, text: &str, base_offset: usize) -> Vec<TaggedToken> {
        self.tokenizer
            .pipe(text)
            .flat_map(|sentence| {
                sentence
                    .tokens()
                    .iter()
                    .map(|token| to_tagged_token(token, base_offset))
                    .collect::<Vec<_>>()
            })
            .collect()
    }
}

/// Converts one `nlprule` token into a [`TaggedToken`] whose span is an
/// absolute byte offset into the original document (`nlprule`'s spans are
/// relative to whatever text it was handed, i.e. relative to `text` in
/// [`Tagger::tag`]).
fn to_tagged_token(nlp_token: &nlprule::types::Token<'_>, base_offset: usize) -> TaggedToken {
    let surface = nlp_token.word().as_str();
    let byte_span = nlp_token.span().byte();
    let range = (byte_span.start + base_offset)..(byte_span.end + base_offset);
    let token = Token::new(range, classify_token_kind(surface));

    let best = nlp_token
        .word()
        .tags()
        .iter()
        .find(|data| is_linguistic_pos(data.pos().as_str()));

    let (pos, lemma) = best.map_or_else(
        || (PosTag::unknown(), Box::from(surface)),
        |data| {
            (
                PosTag::new(data.pos().as_str()),
                Box::from(data.lemma().as_str()),
            )
        },
    );

    TaggedToken { token, pos, lemma }
}

/// `nlprule` appends bookkeeping pseudo-tags — an empty string, plus
/// `"UNKNOWN"`, `"SENT_START"`, `"SENT_END"` markers — after a word's real
/// linguistic candidates. This filters those out so [`to_tagged_token`]
/// only picks a genuine part-of-speech tag.
fn is_linguistic_pos(pos: &str) -> bool {
    !pos.is_empty() && !matches!(pos, "UNKNOWN" | "SENT_START" | "SENT_END")
}

#[cfg(test)]
mod tests {
    use std::sync::OnceLock;

    use friction_core::TokenKind;

    use super::*;

    fn tagger() -> &'static NlpruleTagger {
        static TAGGER: OnceLock<NlpruleTagger> = OnceLock::new();
        TAGGER.get_or_init(|| NlpruleTagger::new().expect("embedded model must load"))
    }

    /// Tagging a plain sentence recovers the expected surface spans, POS
    /// categories, and lemmas, with every span shifted by `base_offset`.
    #[test]
    fn tag_produces_expected_spans_pos_and_lemmas() {
        // Enough context ("over lazy dogs") for nlprule's disambiguation
        // pass to prefer the verbal ("jumping" as VBG) reading of the
        // gerund over the adjectival one ("a jumping bean") it cannot
        // rule out from a shorter sentence.
        let text = "The quick brown foxes are jumping over lazy dogs.";
        let base_offset = 100;
        let tagged = tagger().tag(text, base_offset);

        let by_surface = |surface: &str| -> &TaggedToken {
            tagged
                .iter()
                .find(|t| {
                    &text[t.token.range.start - base_offset..t.token.range.end - base_offset]
                        == surface
                })
                .unwrap_or_else(|| panic!("no tagged token for {surface:?} in {tagged:?}"))
        };

        let foxes = by_surface("foxes");
        assert_eq!(&*foxes.lemma, "fox");
        assert_eq!(foxes.pos.as_str(), "NNS");
        assert_eq!(foxes.token.kind, TokenKind::Word);
        assert_eq!(
            &text[foxes.token.range.start - base_offset..foxes.token.range.end - base_offset],
            "foxes"
        );

        let jumping = by_surface("jumping");
        assert_eq!(&*jumping.lemma, "jump");
        assert_eq!(jumping.pos.as_str(), "VBG");

        // Every span is shifted by base_offset and lies within the shifted
        // text bounds.
        for t in &tagged {
            assert!(t.token.range.start >= base_offset);
            assert!(t.token.range.end <= base_offset + text.len());
        }
    }

    /// Punctuation tokens classify as `TokenKind::Punctuation`.
    #[test]
    fn tag_classifies_punctuation_tokens() {
        let tagged = tagger().tag("Wait, really?", 0);
        let comma = tagged
            .iter()
            .find(|t| &"Wait, really?"[t.token.range.clone()] == ",")
            .expect("comma token present");
        assert_eq!(comma.token.kind, TokenKind::Punctuation);
    }

    /// A word entirely absent from the tagger's dictionary still produces
    /// a token: its own text as the lemma, and the unknown-POS sentinel,
    /// rather than an error or a panic.
    #[test]
    fn tag_degrades_gracefully_for_out_of_vocabulary_words() {
        let tagged = tagger().tag("Zxqvplarnfrobnicate is a word.", 0);
        let oov = &tagged[0];
        assert_eq!(&*oov.lemma, "Zxqvplarnfrobnicate");
        assert_eq!(oov.pos, PosTag::unknown());
    }

    /// An empty sentence tags to zero tokens rather than panicking.
    #[test]
    fn tag_accepts_empty_text() {
        assert!(tagger().tag("", 0).is_empty());
    }

    /// Tagging the same paragraph twice produces byte-identical output:
    /// the crate's determinism invariant, exercised with a paragraph
    /// varied enough (mixed sentence lengths, punctuation, a contraction,
    /// an out-of-vocabulary token) to catch nondeterminism that a single
    /// trivial sentence could hide.
    #[test]
    fn tagging_is_deterministic_across_repeated_runs() {
        let paragraph = "The team didn't leverage the framework's full potential. \
                          Zxqvplarnfrobnicate elements were, however, refactored twice. \
                          Meanwhile, 42 widgets shipped on time, and the client was pleased!";

        let first = tagger().tag(paragraph, 0);
        let second = tagger().tag(paragraph, 0);
        assert_eq!(first, second);

        // Independently-constructed taggers agree too, not just repeated
        // calls on the same instance.
        let fresh = NlpruleTagger::new().expect("embedded model must load");
        let third = fresh.tag(paragraph, 0);
        assert_eq!(first, third);
    }
}
