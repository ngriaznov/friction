//! Measures the trained perceptron tagger's held-out accuracy: overall tag
//! accuracy and full per-tag precision/recall/F1, on the gold file's own
//! `# split=test` sentences — the ones `examples/train_perceptron.rs` never
//! trains on (see `friction_nlp::train_support::parse_gold_file_split`).
//!
//! The gold file stores only `(word, tag)` pairs, not byte spans, so this
//! tool reconstructs a single-space-joined pseudo-text from the gold words
//! and re-tags *that* with the real, public [`PerceptronTagger::tag`] API
//! rather than the original text. This is exact, not approximate:
//! `tokenize()` (see `src/tag_perceptron.rs`) splits purely on character
//! class and never merges tokens across whitespace, so inserting one space
//! between two gold words can never change how many tokens result or where
//! they split. The reconstructed text retokenizes to the exact same token
//! sequence every time. A per-sentence length/order mismatch therefore
//! means a real bug, so this tool asserts against it rather than silently
//! dropping the sentence.
//!
//! ```text
//! cargo run -p friction-nlp --example eval_perceptron --features train-tooling -- \
//!     crates/friction-nlp/weights/gold_pos_en.tsv
//! ```

use std::collections::BTreeMap;
use std::path::PathBuf;

use friction_nlp::train_support::parse_gold_file_split;
use friction_nlp::{PerceptronTagger, Tagger};

/// A test sentence's `# split=test` marker value from `parse_gold_file_split`.
const TEST_SPLIT: &str = "test";

/// Below this many held-out gold occurrences, a tag's precision/recall are
/// reported but flagged as too thin to trust.
const MIN_TRUSTED_SUPPORT: usize = 50;

#[derive(Default)]
struct Confusion {
    /// `gold_totals[tag]`: how many held-out tokens are gold-tagged `tag`.
    gold_totals: BTreeMap<String, usize>,
    /// `pred_totals[tag]`: how many held-out tokens the tagger predicted
    /// `tag` for.
    pred_totals: BTreeMap<String, usize>,
    /// `correct[tag]`: how many held-out tokens gold-tagged `tag` the
    /// tagger also predicted `tag` for (this metric's true positives).
    correct: BTreeMap<String, usize>,
    total_tokens: usize,
    total_correct: usize,
}

impl Confusion {
    fn record(&mut self, gold: &str, predicted: &str) {
        self.total_tokens += 1;
        *self.gold_totals.entry(gold.to_string()).or_insert(0) += 1;
        *self.pred_totals.entry(predicted.to_string()).or_insert(0) += 1;
        if gold == predicted {
            self.total_correct += 1;
            *self.correct.entry(gold.to_string()).or_insert(0) += 1;
        }
    }

    fn all_tags(&self) -> Vec<String> {
        let mut tags: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        tags.extend(self.gold_totals.keys().cloned());
        tags.extend(self.pred_totals.keys().cloned());
        tags.into_iter().collect()
    }

    /// `(precision, recall, f1, support)` for `tag`; `None` fields are
    /// reported as `0.0` when their denominator is zero (a tag the model
    /// never predicts, or one absent from the held-out gold set).
    fn metrics(&self, tag: &str) -> (f64, f64, f64, usize) {
        let support = self.gold_totals.get(tag).copied().unwrap_or(0);
        let predicted_count = self.pred_totals.get(tag).copied().unwrap_or(0);
        let tp = self.correct.get(tag).copied().unwrap_or(0);
        #[allow(clippy::cast_precision_loss)]
        let precision = if predicted_count == 0 {
            0.0
        } else {
            tp as f64 / predicted_count as f64
        };
        #[allow(clippy::cast_precision_loss)]
        let recall = if support == 0 {
            0.0
        } else {
            tp as f64 / support as f64
        };
        let f1 = if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * precision * recall / (precision + recall)
        };
        (precision, recall, f1, support)
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let gold_path = PathBuf::from(
        args.next()
            .unwrap_or_else(|| "crates/friction-nlp/weights/gold_pos_en.tsv".to_string()),
    );

    let gold_text = std::fs::read_to_string(&gold_path)
        .unwrap_or_else(|err| panic!("failed to read {}: {err}", gold_path.display()));
    let sentences = parse_gold_file_split(&gold_text, TEST_SPLIT);
    assert!(
        !sentences.is_empty(),
        "gold file {} produced zero {TEST_SPLIT}-split sentences",
        gold_path.display()
    );
    let token_count: usize = sentences.iter().map(Vec::len).sum();
    eprintln!(
        "loaded {} {TEST_SPLIT}-split sentences, {} tokens from {}",
        sentences.len(),
        token_count,
        gold_path.display()
    );

    let tagger = PerceptronTagger::new().expect("embedded perceptron weight artifact must load");

    let mut confusion = Confusion::default();
    let mut mismatched_sentences = 0usize;

    for sentence in &sentences {
        let words: Vec<&str> = sentence.iter().map(|(word, _)| word.as_str()).collect();
        let pseudo_text = words.join(" ");
        let predicted = tagger.tag(&pseudo_text, 0);

        if predicted.len() != sentence.len() {
            mismatched_sentences += 1;
            eprintln!(
                "WARNING: retokenization produced {} tokens, gold has {} -- skipping sentence \
                 {:?}",
                predicted.len(),
                sentence.len(),
                pseudo_text
            );
            continue;
        }

        for (predicted_token, (gold_word, gold_tag)) in predicted.iter().zip(sentence.iter()) {
            let predicted_surface = &pseudo_text[predicted_token.token.range.clone()];
            assert_eq!(
                predicted_surface, gold_word,
                "retokenized surface {predicted_surface:?} does not match gold word \
                 {gold_word:?} in sentence {pseudo_text:?} -- retokenization invariant broken"
            );
            confusion.record(gold_tag, predicted_token.pos.as_str());
        }
    }

    assert_eq!(
        mismatched_sentences, 0,
        "{mismatched_sentences} sentence(s) failed to retokenize to the gold token count; this \
         indicates a real bug (see this tool's module docs), not an expected edge case"
    );

    #[allow(clippy::cast_precision_loss)]
    let overall_accuracy = confusion.total_correct as f64 / confusion.total_tokens.max(1) as f64;
    println!(
        "overall tag accuracy: {}/{} ({:.4}%)",
        confusion.total_correct,
        confusion.total_tokens,
        overall_accuracy * 100.0
    );
    println!();
    println!("per-tag precision/recall/F1 (support = held-out gold occurrences):");
    println!(
        "{:<8} {:>8} {:>8} {:>8} {:>8}  note",
        "tag", "precision", "recall", "f1", "support"
    );
    for tag in confusion.all_tags() {
        let (precision, recall, f1, support) = confusion.metrics(&tag);
        let note = if support < MIN_TRUSTED_SUPPORT {
            "too thin to trust"
        } else {
            ""
        };
        println!("{tag:<8} {precision:>8.4} {recall:>8.4} {f1:>8.4} {support:>8}  {note}");
    }

    println!();
    println!("tags of specific interest (VBG VBN NNP RB VBP VB IN WDT TO CC JJ NN):");
    for tag in [
        "VBG", "VBN", "NNP", "RB", "VBP", "VB", "IN", "WDT", "TO", "CC", "JJ", "NN",
    ] {
        let (precision, recall, f1, support) = confusion.metrics(tag);
        let note = if support < MIN_TRUSTED_SUPPORT {
            "too thin to trust"
        } else {
            ""
        };
        println!("{tag:<8} {precision:>8.4} {recall:>8.4} {f1:>8.4} {support:>8}  {note}");
    }
}
