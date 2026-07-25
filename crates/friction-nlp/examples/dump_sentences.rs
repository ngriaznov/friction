//! Dumps candidate sentences for the dependency-parser gold-data pipeline:
//! every `class: human`, `split: train` document in `corpus/manifest.jsonl`,
//! across all five genres, run through the real `friction-parse` prose
//! extraction and [`SrxSegmenter`] segmentation — the same pipeline
//! `examples/train_perceptron.rs`'s own gold file and
//! `friction_harness::fragment` both use — then tagged with
//! [`PerceptronTagger`] and filtered to the tagger's own gold-sentence
//! shape (4-40 whitespace-separated words, pure ASCII, no `UNKNOWN`
//! sentinel).
//!
//! Output is newline-delimited JSON on stdout, one object per surviving
//! sentence, with byte offsets relative to that sentence's own text (not
//! the document). This is a read-only extraction step: it never writes
//! under `corpus/`.
//!
//! ```text
//! cargo run -p friction-nlp --example dump_sentences -- [corpus_dir]
//! ```
//!
//! `corpus_dir` defaults to `corpus`, resolved relative to the current
//! working directory (matching the crate-root-relative convention
//! `crates/friction-nlp/tests/corpus_determinism.rs` uses for the same
//! corpus).

use std::io::Write as _;
use std::path::{Path, PathBuf};

use friction_nlp::{PerceptronTagger, SrxSegmenter, Tagger, segment_document};
use serde::Serialize;

/// One manifest line's fields this tool actually needs. Parsed leniently
/// (unknown fields ignored) since this is a read-only consumer of a
/// manifest schema owned elsewhere (`corpus-tool::manifest`), not a
/// validator of it.
#[derive(Debug, serde::Deserialize)]
struct ManifestLine {
    id: String,
    class: String,
    genre: String,
    split: Option<String>,
}

/// One filtered token in the emitted sentence, with byte offsets relative
/// to that sentence's own `text` field.
#[derive(Serialize)]
struct TokenOut<'a> {
    surface: &'a str,
    pos: &'a str,
    start: usize,
    end: usize,
}

/// One emitted sentence: newline-delimited JSON, one object per line.
#[derive(Serialize)]
struct SentenceOut<'a> {
    doc_id: &'a str,
    genre: &'a str,
    sent_index: usize,
    text: &'a str,
    tokens: Vec<TokenOut<'a>>,
}

/// A whitespace-separated word count of 4..=40, matching the tagger's own
/// gold-sentence filter (see `weights/NOTICE.md`).
const MIN_WORDS: usize = 4;
const MAX_WORDS: usize = 40;

/// Resolves manifest entry `id`/`genre` to a corpus document path: the
/// primary location is `<corpus_dir>/human/<genre>/<id>.md`; some human
/// documents were quarantined instead and live at
/// `<corpus_dir>/quarantine/<genre>/<id>.md`.
fn resolve_doc_path(corpus_dir: &Path, genre: &str, id: &str) -> Option<PathBuf> {
    let primary = corpus_dir
        .join("human")
        .join(genre)
        .join(format!("{id}.md"));
    if primary.is_file() {
        return Some(primary);
    }
    let fallback = corpus_dir
        .join("quarantine")
        .join(genre)
        .join(format!("{id}.md"));
    if fallback.is_file() {
        return Some(fallback);
    }
    None
}

/// Reads `<corpus_dir>/manifest.jsonl` and returns every `class: human`,
/// `split: train` entry, sorted by `id` so document processing order (and
/// therefore this tool's stdout) is reproducible independent of the
/// manifest file's own line order.
fn train_human_entries(corpus_dir: &Path) -> Vec<ManifestLine> {
    let manifest_path = corpus_dir.join("manifest.jsonl");
    let text = std::fs::read_to_string(&manifest_path)
        .unwrap_or_else(|err| panic!("failed to read {}: {err}", manifest_path.display()));

    let mut entries: Vec<ManifestLine> = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .enumerate()
        .map(|(lineno, line)| {
            serde_json::from_str::<ManifestLine>(line).unwrap_or_else(|err| {
                panic!(
                    "{}:{}: failed to parse manifest line: {err}",
                    manifest_path.display(),
                    lineno + 1
                )
            })
        })
        .filter(|entry| entry.class == "human" && entry.split.as_deref() == Some("train"))
        .collect();

    entries.sort_by(|a, b| a.id.cmp(&b.id));
    entries
}

/// True if `text` passes the tagger's own gold-sentence shape: 4-40
/// whitespace-separated words, pure ASCII, and free of the `UNKNOWN`
/// sentinel among `tokens`.
fn passes_gold_filter(text: &str, tokens: &[friction_nlp::TaggedToken]) -> bool {
    let word_count = text.split_whitespace().count();
    if !(MIN_WORDS..=MAX_WORDS).contains(&word_count) {
        return false;
    }
    if !text.is_ascii() {
        return false;
    }
    if tokens.iter().any(|t| t.pos.as_str() == "UNKNOWN") {
        return false;
    }
    true
}

fn main() {
    let mut args = std::env::args().skip(1);
    let corpus_dir = PathBuf::from(args.next().unwrap_or_else(|| "corpus".to_string()));

    let entries = train_human_entries(&corpus_dir);
    eprintln!(
        "{} class=human split=train manifest entries across all genres",
        entries.len()
    );

    let mut unresolved: Vec<(String, String)> = Vec::new();
    let mut resolved: Vec<(ManifestLine, PathBuf)> = Vec::new();
    for entry in entries {
        match resolve_doc_path(&corpus_dir, &entry.genre, &entry.id) {
            Some(path) => resolved.push((entry, path)),
            None => unresolved.push((entry.id.clone(), entry.genre.clone())),
        }
    }

    if unresolved.is_empty() {
        eprintln!("resolved all {} entries to a file on disk", resolved.len());
    } else {
        eprintln!(
            "WARNING: {} of {} entries could not be resolved to a file under {}/human or {}/quarantine:",
            unresolved.len(),
            unresolved.len() + resolved.len(),
            corpus_dir.display(),
            corpus_dir.display()
        );
        for (id, genre) in &unresolved {
            eprintln!("  {id} (genre={genre})");
        }
    }

    let segmenter = SrxSegmenter::new();
    let tagger = PerceptronTagger::new().expect("embedded perceptron weight artifact must load");

    let stdout = std::io::stdout();
    let mut out = stdout.lock();

    let mut doc_count = 0usize;
    let mut total_sentences = 0usize;
    let mut kept_sentences = 0usize;
    let mut kept_tokens = 0usize;

    for (entry, path) in &resolved {
        let source = std::fs::read_to_string(path)
            .unwrap_or_else(|err| panic!("failed to read {}: {err}", path.display()));
        let document = friction_parse::parse(source.as_str())
            .unwrap_or_else(|err| panic!("failed to parse {}: {err}", path.display()));
        let sentenced = segment_document(&document, &segmenter)
            .unwrap_or_else(|err| panic!("failed to segment {}: {err}", path.display()));

        doc_count += 1;
        let mut sent_index = 0usize;

        for unit in sentenced.prose() {
            for sentence in &unit.sentences {
                let text = sentenced.text(&sentence.range).unwrap_or_else(|err| {
                    panic!("invalid sentence span in {}: {err}", path.display())
                });
                total_sentences += 1;

                let tokens = tagger.tag(text, 0);

                if passes_gold_filter(text, &tokens) {
                    let tokens_out: Vec<TokenOut<'_>> = tokens
                        .iter()
                        .map(|t| TokenOut {
                            surface: &text[t.token.range.clone()],
                            pos: t.pos.as_str(),
                            start: t.token.range.start,
                            end: t.token.range.end,
                        })
                        .collect();
                    kept_tokens += tokens_out.len();

                    let record = SentenceOut {
                        doc_id: &entry.id,
                        genre: &entry.genre,
                        sent_index,
                        text,
                        tokens: tokens_out,
                    };
                    serde_json::to_writer(&mut out, &record)
                        .expect("serializing a sentence record cannot fail");
                    out.write_all(b"\n").expect("stdout write must succeed");
                    kept_sentences += 1;
                }

                sent_index += 1;
            }
        }
    }

    eprintln!(
        "{doc_count} documents processed, {total_sentences} candidate sentences, \
         {kept_sentences} kept after the gold-shape filter ({kept_tokens} tokens)"
    );
}
