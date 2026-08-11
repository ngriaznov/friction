//! Shared seam-bigram sentence walk.
//!
//! The exact per-sentence tokenization and `"<s>"`-prefixed windowed
//! `(left, right)` pair emission `corpus-tool attest` mines the TRAIN
//! human bigram table with (see `friction_packs::attestation`'s own
//! module docs for the bigram side's tokenization convention:
//! `friction_harness::clean`'s `[a-z']+` word runs, single-character
//! `.,;:!?` punctuation, digits dropped).
//!
//! Factored out of `commands::attest` so `commands::pooled_attest`
//! (external pooled bigram counts, `friction_packs::pooled_attest`) walks
//! documents through literally the same code path rather than a
//! hand-copied approximation that could silently drift from it — the two
//! commands' mined evidence is only comparable at all if "what counts as
//! a seam pair" means the same thing on both sides.

/// The reserved sentence-start token prepended to every sentence's word
/// sequence before windowing, so a sentence's first real word always has
/// a left neighbor to form a bigram with.
///
/// Mirrors `friction_packs::attestation`'s own reserved `"<s>"` convention.
pub const SENTENCE_START: &str = "<s>";

/// Splits `text` into sentences and tokenizes each one.
///
/// `friction_harness::clean::clean` then `friction_harness::clean::split_sentences`,
/// then `friction_harness::clean::tokenize` per sentence, dropping any
/// sentence with no bigram tokens — the exact per-document walk
/// `corpus-tool attest`'s own bigram-side pass performs (see that
/// command's module docs: the bigram and skeleton passes are
/// independent, and a sentence contributing nothing to the bigram side is
/// simply skipped there).
#[must_use]
pub fn tokenized_sentences(text: &str) -> Vec<Vec<String>> {
    let cleaned = friction_harness::clean::clean(text);
    friction_harness::clean::split_sentences(&cleaned)
        .into_iter()
        .map(friction_harness::clean::tokenize)
        .filter(|tokens| !tokens.is_empty())
        .collect()
}

/// The consecutive `(left, right)` token pairs one sentence's own bigram
/// tokens produce.
///
/// The reserved [`SENTENCE_START`] sentinel is prepended so the
/// sentence's first real word always pairs with a left neighbor —
/// `corpus-tool attest`'s `<s>`-prefixed `windows(2)` walk, extracted so
/// every caller reads it off the one definition instead of re-deriving
/// it.
#[must_use]
pub fn seam_pairs(tokens: &[String]) -> Vec<(&str, &str)> {
    let mut seq: Vec<&str> = Vec::with_capacity(tokens.len() + 1);
    seq.push(SENTENCE_START);
    seq.extend(tokens.iter().map(String::as_str));
    seq.windows(2).map(|w| (w[0], w[1])).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenized_sentences_splits_and_tokenizes() {
        let sentences = tokenized_sentences("The fox runs. Quick.");
        assert_eq!(sentences.len(), 2);
        assert_eq!(sentences[0], vec!["the", "fox", "runs", "."]);
        assert_eq!(sentences[1], vec!["quick", "."]);
    }

    #[test]
    fn tokenized_sentences_drops_a_sentence_with_no_bigram_tokens() {
        // The trailing chunk "123" has no terminal punctuation of its own
        // (nothing left to split on) and digits are dropped entirely by
        // `tokenize`, so it tokenizes to an empty list and is dropped.
        let sentences = tokenized_sentences("The fox runs. 123");
        assert_eq!(sentences, vec![vec!["the", "fox", "runs", "."]]);
    }

    #[test]
    fn seam_pairs_prefixes_sentence_start_and_windows_by_two() {
        let tokens = vec!["the".to_string(), "fox".to_string(), ".".to_string()];
        let pairs = seam_pairs(&tokens);
        assert_eq!(pairs, vec![("<s>", "the"), ("the", "fox"), ("fox", ".")]);
    }

    #[test]
    fn seam_pairs_of_a_single_token_sentence_pairs_only_with_sentence_start() {
        let tokens = vec!["hello".to_string()];
        assert_eq!(seam_pairs(&tokens), vec![("<s>", "hello")]);
    }
}
