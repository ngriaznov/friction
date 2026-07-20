//! Cross-run determinism check: parsing, segmenting, and tagging the same
//! corpus documents must produce byte-identical token/tag streams every
//! time, in-process and across separate process invocations.
//!
//! Two variants:
//!
//! - A fast, always-on smoke check over the first 20 corpus documents
//!   (`corpus_determinism_smoke`), run as part of the normal test suite.
//! - The full corpus check (`corpus_determinism_full`), `#[ignore]`d by
//!   default since it runs the tagger over every document in the corpus
//!   fixture. Run it explicitly with:
//!
//!   ```text
//!   cargo test -p friction-nlp --test corpus_determinism -- --ignored --nocapture
//!   ```
//!
//!   which prints the resulting hash. To confirm stability across
//!   separate process invocations (not just repeated calls within one
//!   process), run that same command two or three times in a row and
//!   compare the printed hash by eye — each invocation hashes the corpus
//!   three times in-process on top of that.

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use friction_nlp::{NlpruleTagger, SrxSegmenter, Tagger, segment_document};
use sha2::{Digest, Sha256};

/// Root of the corpus fixture directory, resolved relative to this
/// crate's manifest directory so the test works regardless of the
/// process's current working directory.
fn corpus_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus")
}

/// Appends every `.md` file found (recursively) under `dir` to `out`, one
/// directory level at a time, sorting each level's entries by name before
/// descending. This never relies on the operating system's own directory
/// iteration order, which `std::fs::read_dir` explicitly does not
/// guarantee to be stable.
fn collect_md_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    let mut children: Vec<PathBuf> = entries
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .collect();
    children.sort();
    for path in children {
        if path.is_dir() {
            collect_md_files(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "md") {
            out.push(path);
        }
    }
}

/// Every corpus document this check runs over: every `.md` file under
/// `corpus/human`, `corpus/llm`, and `corpus/quarantine` (712 files as of
/// this writing), sorted by path for a deterministic, machine-independent
/// order.
///
/// `corpus/incoming` is excluded — it is a staging area for documents not
/// yet promoted into the labeled corpus, duplicating some of
/// `corpus/human`'s content — and `corpus/prompts` holds generation
/// config, not documents.
fn corpus_docs() -> Vec<PathBuf> {
    let root = corpus_root();
    let mut docs = Vec::new();
    for subdir in ["human", "llm", "quarantine"] {
        collect_md_files(&root.join(subdir), &mut docs);
    }
    docs.sort();
    docs
}

/// Runs parse -> segment -> tag over `path`'s contents and folds a
/// canonical, deterministic serialization of every resulting token into
/// `hasher`: one line per token, in document order, naming the source
/// file (relative to the corpus root, so the hash does not depend on
/// where the corpus happens to be checked out on a given machine),
/// sentence index, absolute byte span, lexical kind, POS tag, and lemma.
fn hash_document(
    root: &Path,
    path: &Path,
    segmenter: SrxSegmenter,
    tagger: &NlpruleTagger,
    hasher: &mut Sha256,
    line: &mut String,
) {
    let relative = path.strip_prefix(root).unwrap_or(path);
    let source = fs::read_to_string(path)
        .unwrap_or_else(|err| panic!("failed to read corpus doc {}: {err}", path.display()));
    let document = friction_parse::parse(source.as_str())
        .unwrap_or_else(|err| panic!("failed to parse corpus doc {}: {err}", path.display()));
    let sentenced = segment_document(&document, &segmenter)
        .unwrap_or_else(|err| panic!("failed to segment corpus doc {}: {err}", path.display()));

    for unit in sentenced.prose() {
        for (sentence_index, sentence) in unit.sentences.iter().enumerate() {
            let text = sentenced
                .text(&sentence.range)
                .unwrap_or_else(|err| panic!("sentence span invalid in {}: {err}", path.display()));
            for token in tagger.tag(text, sentence.range.start) {
                line.clear();
                let _ = writeln!(
                    line,
                    "{}\t{sentence_index}\t{}\t{}\t{:?}\t{}\t{}",
                    relative.display(),
                    token.token.range.start,
                    token.token.range.end,
                    token.token.kind,
                    token.pos.as_str(),
                    token.lemma,
                );
                hasher.update(line.as_bytes());
            }
        }
    }
}

/// Runs the full parse -> segment -> tag pipeline over every document in
/// `paths`, in order, and returns the lowercase hex sha256 of the
/// resulting canonical token stream.
fn hash_corpus(paths: &[PathBuf], segmenter: SrxSegmenter, tagger: &NlpruleTagger) -> String {
    let root = corpus_root();
    let mut hasher = Sha256::new();
    let mut line = String::new();
    for path in paths {
        hash_document(&root, path, segmenter, tagger, &mut hasher, &mut line);
    }
    let digest = hasher.finalize();
    digest
        .iter()
        .fold(String::with_capacity(64), |mut hex, byte| {
            let _ = write!(hex, "{byte:02x}");
            hex
        })
}

/// Fast, always-on variant: hashes the first 20 corpus documents (by the
/// same deterministic ordering the full check uses) three times
/// in-process and asserts every run agrees.
#[test]
fn corpus_determinism_smoke() {
    let docs = corpus_docs();
    assert!(
        docs.len() >= 20,
        "expected at least 20 corpus documents, found {}",
        docs.len()
    );
    let sample = &docs[..20];

    let segmenter = SrxSegmenter::new();
    let tagger = NlpruleTagger::new().expect("embedded model must load");

    let first = hash_corpus(sample, segmenter, &tagger);
    let second = hash_corpus(sample, segmenter, &tagger);
    let third = hash_corpus(sample, segmenter, &tagger);

    println!("corpus_determinism_smoke: {first} ({} docs)", sample.len());
    assert_eq!(first, second, "run 1 and run 2 disagree");
    assert_eq!(first, third, "run 1 and run 3 disagree");
}

/// Full corpus variant: hashes every document in the corpus fixture three
/// times in-process and asserts every run agrees, printing the resulting
/// hash. `#[ignore]`d by default since it runs the tagger over the whole
/// corpus; see this file's module docs for the command to run it
/// explicitly (including confirming stability across separate process
/// invocations, which an in-process check alone cannot rule out — for
/// instance, nondeterminism seeded from the process's environment or
/// address-space layout rather than from any per-call state).
#[test]
#[ignore = "runs the tagger over the full 712-document corpus; see module docs for the explicit invocation"]
fn corpus_determinism_full() {
    let docs = corpus_docs();
    assert!(
        docs.len() >= 712,
        "expected at least 712 corpus documents, found {}",
        docs.len()
    );

    let segmenter = SrxSegmenter::new();
    let tagger = NlpruleTagger::new().expect("embedded model must load");

    let first = hash_corpus(&docs, segmenter, &tagger);
    let second = hash_corpus(&docs, segmenter, &tagger);
    let third = hash_corpus(&docs, segmenter, &tagger);

    println!("corpus_determinism_full: {first} ({} docs)", docs.len());
    assert_eq!(first, second, "run 1 and run 2 disagree");
    assert_eq!(first, third, "run 1 and run 3 disagree");
}
