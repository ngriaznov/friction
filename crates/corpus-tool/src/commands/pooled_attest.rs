//! `corpus-tool pooled-attest` — builds `pooled-attest-v1`
//! (`friction_packs::pooled_attest`), the pooled seam-bigram membership
//! filter mined from locally-staged external human corpora.
//!
//! Widens [`friction_packs::AttestationPack`]'s TRAIN-only bigram table
//! (see that module's own "Widened evidence" docs, and
//! `friction_packs::pooled_attest`'s module docs for the full motivation
//! and mining-floor rationale).
//!
//! # Same shape as `human-evidence`, same sentence walk as `attest`
//!
//! Input handling mirrors `corpus-tool human-evidence`: every `--input`
//! is a `cleaned.jsonl` file (JSONL, one `{"text": ...}` object per line
//! — `"id"` is read but not used), already downloaded and staged by
//! whoever runs this (never a network fetch). Each document's text is
//! walked into sentences and `(left, right)` seam pairs through
//! `crate::seam_bigram` — the EXACT sentence-split/tokenize/windows(2)
//! convention `corpus-tool attest` mines the TRAIN human bigram table
//! with, so a pooled pair and a TRAIN pair always mean the same thing.
//!
//! # Mining floor: `>= 2` occurrences
//!
//! A `(left, right)` pair is kept only once it has recurred at least
//! twice across every staged input, counted independently per pair (not
//! per document — two occurrences in the same document count). See
//! `friction_packs::pooled_attest`'s module docs for why this floor
//! exists (the external corpora are unreviewed web text, unlike TRAIN)
//! and why it is a deliberate departure from the TRAIN table's own pure-
//! membership stance.
//!
//! # `--built-at`: caller-supplied, never system time
//!
//! The meta sidecar's `built_at` field is required and comes from the
//! caller (a `YYYY-MM-DD` string, by convention) rather than reading the
//! system clock: this keeps a rebuild from the same inputs reproducible
//! byte-for-byte regardless of when it happens to run, the same
//! determinism this workspace's other derived artifacts hold to.

use std::collections::{BTreeSet, HashMap};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use anyhow::Context as _;
use clap::Args as ClapArgs;
use friction_packs::pooled_attest::{build_pack_bytes, pair_key};

use crate::hashing::sha256_hex;
use crate::seam_bigram::{seam_pairs, tokenized_sentences};

/// A pair is kept in the pooled filter only once it has recurred at
/// least this many times across every staged input — see this module's
/// own docs for the rationale.
const MIN_OCCURRENCES: u32 = 2;

/// Arguments for `corpus-tool pooled-attest`.
#[derive(Debug, ClapArgs)]
pub struct Args {
    /// A `cleaned.jsonl` external corpus file (JSONL, one `{"id": ...,
    /// "text": ...}` object per line — `"id"` is read but not used).
    /// Repeatable; each file's own parent directory basename is recorded
    /// in the meta sidecar as that input's source label.
    #[arg(long = "input")]
    pub input: Vec<PathBuf>,
    /// Path to write the built filter to.
    #[arg(
        long,
        default_value = "crates/friction-packs/packs/pooled-attest-en-v1.bin"
    )]
    pub out_bin: PathBuf,
    /// Path to write the pack's TOML meta sidecar to.
    #[arg(
        long,
        default_value = "crates/friction-packs/packs/pooled-attest-en-v1.meta.toml"
    )]
    pub out_meta: PathBuf,
    /// Recorded verbatim in the meta sidecar's `built_at` field — see
    /// this module's own docs for why this is caller-supplied rather
    /// than the system clock.
    #[arg(long)]
    pub built_at: String,
    /// The minimum number of times a `(left, right)` pair must recur
    /// across every staged input to be kept. Exposed for experimentation;
    /// the shipped pack is built with the default.
    #[arg(long, default_value_t = MIN_OCCURRENCES)]
    pub min_occurrences: u32,
}

/// One `cleaned.jsonl` document: only the field this command reads.
#[derive(Debug, serde::Deserialize)]
struct CorpusDoc {
    #[serde(default)]
    #[allow(dead_code, reason = "kept for schema documentation, not read")]
    id: String,
    text: String,
}

/// One staged input file's raw bytes, parsed documents, and source label
/// (its parent directory's basename, falling back to the file's own stem
/// if it has no parent).
struct SourceInput {
    label: String,
    raw_bytes: Vec<u8>,
    docs: Vec<CorpusDoc>,
}

/// `path`'s source label: its parent directory's basename (e.g.
/// `/.../corpora/codereview-se/cleaned.jsonl` -> `"codereview-se"`),
/// falling back to the file's own stem when it has no parent (or the
/// parent has no basename — a bare relative filename).
fn source_label(path: &Path) -> String {
    path.parent()
        .and_then(Path::file_name)
        .or_else(|| path.file_stem())
        .map_or_else(
            || path.display().to_string(),
            |name| name.to_string_lossy().into_owned(),
        )
}

/// Reads and parses one `--input` file.
///
/// # Errors
/// Returns an error if `path` can't be read, isn't valid UTF-8, or a
/// line fails to parse as a `{"id": ..., "text": ...}` JSON object.
fn read_input(path: &Path) -> anyhow::Result<SourceInput> {
    let raw_bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let text = String::from_utf8(raw_bytes.clone())
        .with_context(|| format!("{} is not valid UTF-8", path.display()))?;
    let mut docs = Vec::new();
    for (lineno, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let doc: CorpusDoc = serde_json::from_str(line).with_context(|| {
            format!("{}:{}: not a valid corpus doc", path.display(), lineno + 1)
        })?;
        docs.push(doc);
    }
    Ok(SourceInput {
        label: source_label(path),
        raw_bytes,
        docs,
    })
}

/// One source's provenance/summary record for the meta sidecar.
struct SourceRecord {
    docs: usize,
    tokens: u64,
    sha256: String,
}

/// Counts every `(left, right)` seam pair across `input`'s documents,
/// using `crate::seam_bigram`'s exact sentence-walk/windowing convention
/// (see this module's own docs). Returns the pair counts (keyed by
/// [`pair_key`], so the caller's threshold filter and the eventual filter
/// build both read the identical string) plus the document's own total
/// bigram-token count (for the meta sidecar's per-source token count).
fn count_pairs(docs: &[CorpusDoc]) -> (HashMap<String, u32>, u64) {
    let mut counts: HashMap<String, u32> = HashMap::new();
    let mut tokens = 0u64;
    for doc in docs {
        for sentence in tokenized_sentences(&doc.text) {
            tokens += sentence.len() as u64;
            for (left, right) in seam_pairs(&sentence) {
                *counts.entry(pair_key(left, right)).or_insert(0) += 1;
            }
        }
    }
    (counts, tokens)
}

/// Runs `pooled-attest`.
///
/// # Errors
/// Returns an error if any `--input` file can't be read or parsed, if
/// filter construction fails (see
/// [`friction_packs::PackError::PooledAttestConstructionFailed`]), or if
/// `--out-bin`/`--out-meta` can't be written.
pub fn run(args: &Args) -> anyhow::Result<()> {
    let mut inputs: Vec<SourceInput> = args
        .input
        .iter()
        .map(|path| read_input(path))
        .collect::<anyhow::Result<_>>()?;
    inputs.sort_by(|a, b| a.label.cmp(&b.label));

    let total_docs: usize = inputs.iter().map(|i| i.docs.len()).sum();

    let mut total_counts: HashMap<String, u32> = HashMap::new();
    let mut source_records: Vec<(String, SourceRecord)> = Vec::with_capacity(inputs.len());
    for input in &inputs {
        let (counts, tokens) = count_pairs(&input.docs);
        for (key, count) in &counts {
            *total_counts.entry(key.clone()).or_insert(0) += count;
        }
        source_records.push((
            input.label.clone(),
            SourceRecord {
                docs: input.docs.len(),
                tokens,
                sha256: sha256_hex(&input.raw_bytes),
            },
        ));
    }

    let total_tokens: u64 = source_records.iter().map(|(_, r)| r.tokens).sum();

    let pair_keys: BTreeSet<String> = total_counts
        .into_iter()
        .filter(|(_, count)| *count >= args.min_occurrences)
        .map(|(key, _)| key)
        .collect();

    let built = build_pack_bytes(&pair_keys)?;

    if let Some(parent) = args.out_bin.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&args.out_bin, &built.bytes)
        .with_context(|| format!("writing {}", args.out_bin.display()))?;

    let bin_sha256 = sha256_hex(&built.bytes);
    let meta = render_meta(&MetaInput {
        key_count: built.key_count,
        pairs_sha256: &built.pairs_sha256,
        bin_sha256: &bin_sha256,
        bin_len: built.bytes.len(),
        min_occurrences: args.min_occurrences,
        built_at: &args.built_at,
        sources: &source_records,
        total_tokens,
    });
    if let Some(parent) = args.out_meta.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&args.out_meta, &meta)
        .with_context(|| format!("writing {}", args.out_meta.display()))?;

    println!(
        "pooled-attest: {} input file(s), {} doc(s), {} bigram token(s) -> {} distinct pair(s) \
         at >= {} occurrence(s) each ({} hash collision(s) dropped) written to {} ({} bytes) + {}",
        inputs.len(),
        total_docs,
        total_tokens,
        built.key_count,
        args.min_occurrences,
        built.hash_collisions,
        args.out_bin.display(),
        built.bytes.len(),
        args.out_meta.display(),
    );
    Ok(())
}

struct MetaInput<'a> {
    key_count: u64,
    pairs_sha256: &'a str,
    bin_sha256: &'a str,
    bin_len: usize,
    min_occurrences: u32,
    built_at: &'a str,
    sources: &'a [(String, SourceRecord)],
    total_tokens: u64,
}

/// TOML-escapes `text` for a basic string literal (backslash and quote
/// only: every value this command writes is plain ASCII).
fn toml_escape(text: &str) -> String {
    text.replace('\\', "\\\\").replace('"', "\\\"")
}

/// Renders the `pooled-attest-en-v1.meta.toml` sidecar: a `[pack]` table
/// with the two fields [`friction_packs::pooled_attest::PooledAttestPack::load`]
/// cross-checks (`version`, `key_count`) plus documentary provenance
/// (mining threshold, `built_at`, both hashes, source labels/token
/// counts) it doesn't re-parse — same tolerance convention as
/// `jargon-attest-en-v1.toml`/`human-evidence-en-v1.toml`'s own sidecars.
fn render_meta(input: &MetaInput<'_>) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# pooled-attest-v1 pack: a BinaryFuse16 filter over `(left, right)`\n\
         # seam-bigram pairs mined from staged external human corpora,\n\
         # widening friction_packs::AttestationPack's TRAIN-only bigram\n\
         # table (see that module's \"Widened evidence\" docs). Generated by\n\
         # `corpus-tool pooled-attest` — do not hand-edit.\n\
         #\n\
         # `friction_packs::pooled_attest::PooledAttestPack::load` cross-checks\n\
         # `version` and `key_count` below against the `.bin`'s own embedded\n\
         # header; every other field here is provenance for humans, not\n\
         # re-parsed."
    )
    .expect("write to String is infallible");
    writeln!(out).expect("write to String is infallible");
    writeln!(out, "[pack]").expect("write to String is infallible");
    writeln!(out, "version = \"pooled-attest-v1\"").expect("write to String is infallible");
    writeln!(out, "key_count = {}", input.key_count).expect("write to String is infallible");
    writeln!(out, "bin_sha256 = \"{}\"", input.bin_sha256).expect("write to String is infallible");
    writeln!(out, "bin_bytes = {}", input.bin_len).expect("write to String is infallible");
    writeln!(out, "pairs_sha256 = \"{}\"", input.pairs_sha256)
        .expect("write to String is infallible");
    writeln!(out, "min_occurrences = {}", input.min_occurrences)
        .expect("write to String is infallible");
    writeln!(out, "built_at = \"{}\"", toml_escape(input.built_at))
        .expect("write to String is infallible");
    writeln!(out, "total_tokens = {}", input.total_tokens).expect("write to String is infallible");
    writeln!(out, "filter_kind = \"xorf::BinaryFuse16\"").expect("write to String is infallible");
    writeln!(out, "false_positive_rate_approx = \"1/65536\"")
        .expect("write to String is infallible");
    writeln!(out, "hash_function = \"xxh64 (xxhash-rust), fixed seed\"")
        .expect("write to String is infallible");
    writeln!(
        out,
        "key_convention = \"left and right joined by U+001F (unit separator); \
         friction_packs::pooled_attest::pair_key\""
    )
    .expect("write to String is infallible");
    writeln!(
        out,
        "determinism = \"bit-identical .bin for a fixed, identically-ordered input \
         pair set: xorf's BinaryFuse16 construction retries a deterministic seed \
         sequence (splitmix64 from a fixed initial state), never external \
         randomness, and this builder always feeds keys in BTreeSet-ascending \
         order\""
    )
    .expect("write to String is infallible");
    writeln!(
        out,
        "collision_direction = \"safe: a false 'attested' only ever admits one \
         additional candidate seam past a gate, never suppresses a real one\""
    )
    .expect("write to String is infallible");

    for (label, record) in input.sources {
        out.push('\n');
        writeln!(out, "[pack.sources.\"{}\"]", toml_escape(label))
            .expect("write to String is infallible");
        writeln!(out, "docs = {}", record.docs).expect("write to String is infallible");
        writeln!(out, "tokens = {}", record.tokens).expect("write to String is infallible");
        writeln!(out, "sha256 = \"{}\"", toml_escape(&record.sha256))
            .expect("write to String is infallible");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_label_uses_the_parent_directory_basename() {
        assert_eq!(
            source_label(Path::new("/a/b/codereview-se/cleaned.jsonl")),
            "codereview-se"
        );
    }

    #[test]
    fn source_label_falls_back_to_the_file_stem_with_no_parent() {
        assert_eq!(source_label(Path::new("cleaned.jsonl")), "cleaned");
    }

    #[test]
    fn count_pairs_counts_every_occurrence_across_sentences() {
        let docs = vec![
            CorpusDoc {
                id: "d0".to_string(),
                text: "The fox runs. The fox jumps.".to_string(),
            },
            CorpusDoc {
                id: "d1".to_string(),
                text: "The fox sleeps.".to_string(),
            },
        ];
        let (counts, tokens) = count_pairs(&docs);
        // "<s>"->"the" occurs three times (once per sentence).
        assert_eq!(counts[&pair_key("<s>", "the")], 3);
        // "the"->"fox" occurs three times too.
        assert_eq!(counts[&pair_key("the", "fox")], 3);
        // "fox"->"runs" occurs once.
        assert_eq!(counts.get(&pair_key("fox", "runs")), Some(&1));
        assert!(tokens > 0);
    }

    #[test]
    fn run_applies_the_min_occurrences_floor() {
        let dir = tempfile::tempdir().expect("tempdir");
        let bucket_dir = dir.path().join("docs");
        std::fs::create_dir_all(&bucket_dir).expect("create bucket dir");
        // "please review" occurs twice (clears a floor of 2); "please
        // merge" occurs once (does not).
        let lines = [
            r#"{"id":"d0","text":"Please review this."}"#,
            r#"{"id":"d1","text":"Please review that."}"#,
            r#"{"id":"d2","text":"Please merge this."}"#,
        ];
        std::fs::write(bucket_dir.join("cleaned.jsonl"), lines.join("\n"))
            .expect("write cleaned.jsonl");

        let out_bin = dir.path().join("out.bin");
        let out_meta = dir.path().join("out.meta.toml");
        let args = Args {
            input: vec![bucket_dir.join("cleaned.jsonl")],
            out_bin: out_bin.clone(),
            out_meta: out_meta.clone(),
            built_at: "2026-08-11".to_string(),
            min_occurrences: 2,
        };
        run(&args).expect("run succeeds");

        let bytes: &'static [u8] = Box::leak(
            std::fs::read(&out_bin)
                .expect("read bin")
                .into_boxed_slice(),
        );
        let meta_text = std::fs::read_to_string(&out_meta).expect("read meta");
        let pack = friction_packs::pooled_attest::PooledAttestPack::load(bytes, &meta_text)
            .expect("built pack loads");
        assert!(pack.attests("please", "review"));
        assert!(
            !pack.attests("please", "merge"),
            "below the min-occurrences floor"
        );
        assert!(meta_text.contains("built_at = \"2026-08-11\""));
    }

    #[test]
    fn run_is_deterministic_across_two_runs_of_the_same_fixture() {
        let dir = tempfile::tempdir().expect("tempdir");
        let bucket_dir = dir.path().join("docs");
        std::fs::create_dir_all(&bucket_dir).expect("create bucket dir");
        let lines = [
            r#"{"id":"d0","text":"Please review this. Please review that."}"#,
            r#"{"id":"d1","text":"The fix looks good to me."}"#,
        ];
        std::fs::write(bucket_dir.join("cleaned.jsonl"), lines.join("\n"))
            .expect("write cleaned.jsonl");
        let input = vec![bucket_dir.join("cleaned.jsonl")];

        let out_bin_a = dir.path().join("a.bin");
        let out_meta_a = dir.path().join("a.meta.toml");
        run(&Args {
            input: input.clone(),
            out_bin: out_bin_a.clone(),
            out_meta: out_meta_a,
            built_at: "2026-08-11".to_string(),
            min_occurrences: 2,
        })
        .expect("run a succeeds");

        let out_bin_b = dir.path().join("b.bin");
        let out_meta_b = dir.path().join("b.meta.toml");
        run(&Args {
            input,
            out_bin: out_bin_b.clone(),
            out_meta: out_meta_b,
            built_at: "2026-08-11".to_string(),
            min_occurrences: 2,
        })
        .expect("run b succeeds");

        assert_eq!(
            std::fs::read(&out_bin_a).expect("read a"),
            std::fs::read(&out_bin_b).expect("read b"),
        );
    }

    #[test]
    fn render_meta_is_valid_toml_and_contains_cross_checked_fields() {
        let sources = vec![(
            "docs".to_string(),
            SourceRecord {
                docs: 3,
                tokens: 42,
                sha256: "deadbeef".to_string(),
            },
        )];
        let text = render_meta(&MetaInput {
            key_count: 7,
            pairs_sha256: "cafef00d",
            bin_sha256: "f00dcafe",
            bin_len: 1234,
            min_occurrences: 2,
            built_at: "2026-08-11",
            sources: &sources,
            total_tokens: 42,
        });
        assert!(text.contains("version = \"pooled-attest-v1\""));
        assert!(text.contains("key_count = 7"));
        assert!(text.contains("min_occurrences = 2"));
        assert!(text.contains("built_at = \"2026-08-11\""));
        assert!(text.contains("[pack.sources.\"docs\"]"));
        assert!(toml::from_str::<toml::Value>(&text).is_ok(), "{text}");
    }
}
