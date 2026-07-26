//! Trains the averaged-perceptron tagger's vendored weight artifact from a
//! hand-curated gold-tag file, deterministically (fixed epoch count, fixed
//! sentence order — no shuffling — trading a little convergence quality for
//! exact reproducibility).
//!
//! ```text
//! cargo run -p friction-nlp --example train_perceptron --features train-tooling -- \
//!     weights/gold_pos_en.tsv weights/perceptron_en.json.gz
//! ```
//!
//! Requires the `train-tooling` feature; not part of the default build or
//! CI, run by a developer regenerating the vendored artifact after
//! extending the gold file. See `weights/NOTICE.md` for the gold file's own
//! provenance.

use std::collections::BTreeMap;
use std::path::PathBuf;

use friction_core::TokenKind;
use friction_nlp::classify_token_kind;
use friction_nlp::train_support::{
    AveragedPerceptron, features_for, parse_gold_file_split, punctuation_tag_for_diagnostics,
    save_weight_file, surfaces_of,
};

/// The split this tool trains on: the gold file's `# split=test` sentences
/// are held out, never seen by `train_one`, so `examples/eval_perceptron.rs`
/// can measure genuine generalization against them.
const TRAIN_SPLIT: &str = "train";

/// The deterministic passthrough tag a non-`Word` gold token must already
/// carry — computed the same way `PerceptronTagger::tag` assigns it at
/// inference, so a gold-file curation mistake (a punctuation line tagged
/// with something other than its deterministic tag) is caught here rather
/// than silently trained on.
fn expected_passthrough_tag(word: &str, kind: TokenKind) -> Option<String> {
    match kind {
        TokenKind::Number => Some("CD".to_string()),
        TokenKind::Punctuation => Some(punctuation_tag_for_diagnostics(word).to_string()),
        TokenKind::Symbol => Some("SYM".to_string()),
        _ => None,
    }
}

/// Fixed epoch count: deterministic, not tuned per run.
const EPOCHS: usize = 10;
/// A word needs at least this many gold occurrences before it is eligible
/// for the tagdict shortcut.
const TAGDICT_MIN_COUNT: u32 = 20;
/// ...and at least this fraction of its occurrences must share one tag.
const TAGDICT_MIN_PURITY: f64 = 0.97;

type GoldSentences = Vec<friction_nlp::train_support::GoldSentence>;

/// Only `Word`-token gold tags become model classes: the perceptron is
/// never asked to score a non-word token (see [`expected_passthrough_tag`]),
/// so its class set shouldn't carry punctuation/number/symbol passthrough
/// values either.
fn observed_word_classes(sentences: &GoldSentences) -> Vec<Box<str>> {
    let mut set: std::collections::BTreeSet<Box<str>> = std::collections::BTreeSet::new();
    for sentence in sentences {
        for (word, tag) in sentence {
            if classify_token_kind(word) == TokenKind::Word {
                set.insert(Box::from(tag.as_str()));
            }
        }
    }
    set.into_iter().collect()
}

/// Every `Word`-token `(lowercased surface, tag)` occurrence count, feeding
/// [`friction_nlp::train_support::AveragedPerceptron::finish`]'s tagdict.
fn word_tag_counts(sentences: &GoldSentences) -> BTreeMap<String, BTreeMap<String, u32>> {
    let mut counts: BTreeMap<String, BTreeMap<String, u32>> = BTreeMap::new();
    for sentence in sentences {
        for (word, tag) in sentence {
            if classify_token_kind(word) != TokenKind::Word {
                continue;
            }
            *counts
                .entry(word.to_lowercase())
                .or_default()
                .entry(tag.clone())
                .or_insert(0) += 1;
        }
    }
    counts
}

/// Every non-word gold entry must already carry the exact deterministic
/// passthrough tag inference will assign it — verified once, up front, so
/// a curation mistake fails loudly instead of quietly training on a token
/// it will never actually be asked to score.
fn assert_passthrough_tags_are_consistent(sentences: &GoldSentences) {
    for sentence in sentences {
        for (word, tag) in sentence {
            let kind = classify_token_kind(word);
            if let Some(expected) = expected_passthrough_tag(word, kind) {
                assert_eq!(
                    tag, &expected,
                    "non-word gold token {word:?} (kind {kind:?}) must carry its deterministic \
                     passthrough tag {expected:?}, found {tag:?}"
                );
            }
        }
    }
}

/// Runs [`EPOCHS`] fixed, unshuffled passes over `sentences`, training
/// `model` on every `Word` token and threading its own live prediction
/// (not the gold tag) into tag-history context — matching
/// `PerceptronTagger::tag`'s inference-time convention exactly.
fn train_epochs(model: &mut AveragedPerceptron, sentences: &GoldSentences) {
    for epoch in 0..EPOCHS {
        let mut correct = 0usize;
        let mut total = 0usize;
        for sentence in sentences {
            let surfaces = surfaces_of(sentence);
            let mut prev1 = "-START-".to_string();
            let mut prev2 = "-START2-".to_string();
            for (i, (word, gold)) in sentence.iter().enumerate() {
                let assigned = if classify_token_kind(word) == TokenKind::Word {
                    let features = features_for(&surfaces, word, i, &prev1, &prev2);
                    let guess = model.train_one(&features, gold);
                    total += 1;
                    if &guess == gold {
                        correct += 1;
                    }
                    guess
                } else {
                    gold.clone()
                };
                prev2 = prev1;
                prev1 = assigned;
            }
        }
        #[allow(clippy::cast_precision_loss)]
        let accuracy = 100.0 * correct as f64 / total.max(1) as f64;
        eprintln!("epoch {epoch}: training accuracy on Word tokens {accuracy:.2}%");
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let gold_path = PathBuf::from(
        args.next()
            .unwrap_or_else(|| "crates/friction-nlp/weights/gold_pos_en.tsv".to_string()),
    );
    let out_path = PathBuf::from(
        args.next()
            .unwrap_or_else(|| "crates/friction-nlp/weights/perceptron_en.json.gz".to_string()),
    );

    let gold_text = std::fs::read_to_string(&gold_path)
        .unwrap_or_else(|err| panic!("failed to read {}: {err}", gold_path.display()));
    let sentences = parse_gold_file_split(&gold_text, TRAIN_SPLIT);
    assert!(
        !sentences.is_empty(),
        "gold file {} produced zero {TRAIN_SPLIT}-split sentences",
        gold_path.display()
    );

    let token_count: usize = sentences.iter().map(Vec::len).sum();
    eprintln!(
        "loaded {} {TRAIN_SPLIT}-split sentences, {} tokens from {} (test-split sentences held \
         out; see examples/eval_perceptron.rs)",
        sentences.len(),
        token_count,
        gold_path.display()
    );

    let classes = observed_word_classes(&sentences);
    eprintln!("{} distinct word-token tags observed", classes.len());

    let counts = word_tag_counts(&sentences);
    assert_passthrough_tags_are_consistent(&sentences);

    let mut model = AveragedPerceptron::new(classes);
    train_epochs(&mut model, &sentences);

    let weight_file = model.finish(&counts, TAGDICT_MIN_COUNT, TAGDICT_MIN_PURITY);
    eprintln!(
        "tagdict entries: {}, feature count: {}",
        weight_file.tagdict.len(),
        weight_file.features.len()
    );

    save_weight_file(&weight_file, &out_path)
        .unwrap_or_else(|err| panic!("failed to write {}: {err}", out_path.display()));

    let bytes = std::fs::read(&out_path).expect("just-written artifact must be readable");
    eprintln!(
        "wrote {} ({} bytes, sha256 {})",
        out_path.display(),
        bytes.len(),
        sha256_hex(&bytes)
    );
}

fn sha256_hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    // A dependency-free sha256 would be a lot of code for a one-off
    // reporting line; shelling out to `shasum`/`sha256sum` (present on
    // every platform this workspace targets) is a pragmatic substitute.
    // Falls back to a placeholder rather than panicking if neither is on
    // PATH.
    for (cmd, args) in [("shasum", &["-a", "256"][..]), ("sha256sum", &[])] {
        if let Ok(mut child) = std::process::Command::new(cmd)
            .args(args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()
        {
            use std::io::Write as _;
            if let Some(stdin) = child.stdin.as_mut() {
                let _ = stdin.write_all(bytes);
            }
            if let Ok(output) = child.wait_with_output() {
                let text = String::from_utf8_lossy(&output.stdout);
                if let Some(hash) = text.split_whitespace().next() {
                    let mut out = String::new();
                    let _ = write!(out, "{hash}");
                    return out;
                }
            }
        }
    }
    "<sha256 unavailable: no shasum/sha256sum on PATH>".to_string()
}
