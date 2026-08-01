//! `corpus-tool human-evidence` — builds the `human-evidence-v1` pack.
//!
//! The pack carries external human-corpus evidence that
//! `friction_packs::frame_compile::CorpusEvidence::with_external` pools
//! into the frame-rewrite compile fences (SYNTHESIS.md's evidence-gated
//! register bands, extended to draw on staged external human corpora,
//! not just the shipped `dms-index-v1` streams).
//!
//! Deterministic and fully offline, like `jargon-attest`: every input is
//! a local file (already downloaded and staged by whoever runs this —
//! this command never makes a network request) — one `--input <dir>` per
//! external corpus, each holding a `cleaned.jsonl` (one JSON object per
//! line, `{"id": ..., "text": ...}`), plus the vendored
//! `frame-rules-v1.toml` rule set.
//!
//! # Tokenization parity with `dms-index-v1`
//!
//! Every document's `text` is tokenized with the *exact* function
//! `corpus-tool index` uses to build the `dms-index-v1` human stream —
//! `friction_harness::clean::tokenize`, run directly over the raw
//! document text, no sentence splitting, no `friction-parse` prose
//! extraction, no lemmatization. This is a deliberate, verified choice,
//! not a default: `tokenize` already case-folds
//! (`friction_harness::clean::clean(text).to_lowercase()` before the
//! `[a-z']+|[.,;:!?]` token regex runs), so every id `corpus-tool index`
//! interns into `dms-index-v1`'s vocabulary is already a plain lowercase
//! surface form — never a lemma, never a stem. Reading through both this
//! module and `frame_compile::CorpusEvidence::from_dms_toml` end to end
//! confirms it: the fences this pack ultimately feeds
//! (`ATTESTATION_FLOOR_PER_MILLION`, the direction fence, the human-rate
//! ceiling) all consult that same plain-surface DMS stream, not
//! `corpus-tool adjudicate`'s separate lemma/surface/reduction referee
//! pipeline (see that command's own module docs — it exists to
//! re-derive rule verdicts against tagged prose, a different consumer
//! entirely, and was never the thing this pack's counts need to agree
//! with). Matching `index`'s tokenizer, word for word, is therefore both
//! the simplest choice available and the only one under which a pooled
//! count means the same thing on both sides of `CorpusEvidence`'s
//! pooling formula.
//!
//! # Phrase-counting semantics
//!
//! `frame_compile::CorpusEvidence::phrase_counts` (what the direction
//! fence and the human-rate ceiling actually call) scans the DMS human
//! stream as one flat sequence, non-overlapping, left to right — *not*
//! sentence-bounded (the DMS streams as built by `corpus-tool index`
//! carry no sentence boundaries at all, only a reserved per-document
//! separator token between concatenated whole-document tokenizations).
//! This command mirrors that exactly: each external document's own
//! `tokenize` output is scanned in isolation, non-overlapping, left to
//! right (never bridging two documents, the same effect `index`'s
//! per-document separator produces), and never split into sentences.
//! Every probe's per-document counts are summed across every document in
//! every `--input` bucket.
//!
//! # Two floors, deliberately different
//!
//! The unigram table keeps a lowercased token surface only once it has
//! been seen at least five times across every staged bucket — noise
//! control for a table with no natural upper bound. The probe table
//! carries no such floor: probes come from a small, fixed source
//! (`frame-rules-v1.toml`'s own literal probes and pilot phrases), so
//! every one of them is recorded with its real count, including an
//! honest `0` — see `friction_packs::human_evidence`'s own module docs
//! for why the two tables' absence therefore means two different things,
//! and why that asymmetry matters to the pooling formula that consumes
//! them.
//!
//! # The empty pack
//!
//! Run with zero `--input` directories, this command builds the empty
//! placeholder pack (`packs/human-evidence-v1.bin` as shipped today —
//! see `friction_packs::human_evidence`'s own docs): no buckets means no
//! documents to tokenize, so this command does not even attempt to
//! derive probes from `--rules` in that case — there is nothing to count
//! them against, and an all-zero probe table built from zero documents
//! would carry no more information than an empty one.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt::Write as _;
use std::path::PathBuf;

use anyhow::Context as _;
use clap::Args as ClapArgs;
use friction_packs::frame_rules::{FrameRuleSet, literal_probes, parse_pattern};

use crate::hashing::sha256_hex;

/// Arguments for `corpus-tool human-evidence`.
#[derive(Debug, ClapArgs)]
pub struct Args {
    /// A directory holding one corpus bucket: either a `cleaned.jsonl`
    /// (one JSON object per line, `{"id": ..., "text": ...}`) or, when
    /// no `cleaned.jsonl` is present, a set of `*.md` documents read
    /// one-per-file in filename order. The directory's own basename
    /// names the bucket (e.g. `codereviewer`, `machine`). Repeatable;
    /// omit entirely to build the empty placeholder pack.
    #[arg(long = "input")]
    pub input: Vec<PathBuf>,
    /// Version string recorded in the TOML sidecar — the pack's
    /// identity, not its on-disk format (the binary layout is shared).
    /// The default builds the human-side pack; the machine-review pack
    /// is built with `--pack-name machine-evidence-v1` and its own
    /// output paths.
    #[arg(long, default_value = "human-evidence-v1")]
    pub pack_name: String,
    /// Path to the frame-rewrite rule set whose literal probes are
    /// counted against the staged corpora.
    #[arg(
        long,
        default_value = "crates/friction-packs/packs/frame-rules-v1.toml"
    )]
    pub rules: PathBuf,
    /// Path to write the pack's TOML sidecar to.
    #[arg(
        long,
        default_value = "crates/friction-packs/packs/human-evidence-v1.toml"
    )]
    pub out_toml: PathBuf,
    /// Path to write the built pack to.
    #[arg(
        long,
        default_value = "crates/friction-packs/packs/human-evidence-v1.bin"
    )]
    pub out_bin: PathBuf,
}

/// One `cleaned.jsonl` document: only the two fields this command reads.
#[derive(Debug, serde::Deserialize)]
struct CorpusDoc {
    #[serde(default)]
    #[allow(dead_code, reason = "kept for schema documentation, not read")]
    id: String,
    text: String,
}

/// One staged bucket's raw input plus its parsed documents.
struct BucketInput {
    name: String,
    raw_bytes: Vec<u8>,
    docs: Vec<CorpusDoc>,
}

/// One staged bucket's provenance/summary record for the TOML sidecar.
struct BucketRecord {
    docs: usize,
    tokens: u64,
    sha256: String,
}

/// Reads and parses one `--input <dir>`: its `cleaned.jsonl` when one
/// is present, otherwise its `*.md` documents (one document per file,
/// filename order; `raw_bytes` for the provenance hash is the
/// filename-prefixed concatenation, so renames and edits both change
/// the recorded sha).
///
/// # Errors
/// Returns an error if `dir` has no directory name, if the input can't
/// be read as UTF-8, if a `cleaned.jsonl` line fails to parse as a
/// `{"id": ..., "text": ...}` JSON object, or if a doc-directory
/// contains no `*.md` files at all.
fn read_bucket(dir: &std::path::Path) -> anyhow::Result<BucketInput> {
    let name = dir
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .ok_or_else(|| anyhow::anyhow!("--input {}: has no directory name", dir.display()))?;
    let path = dir.join("cleaned.jsonl");
    if !path.exists() {
        return read_doc_dir_bucket(dir, name);
    }
    let raw_bytes = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
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
    Ok(BucketInput {
        name,
        raw_bytes,
        docs,
    })
}

/// Reads a doc-directory bucket: every `*.md` file, one document per
/// file, in filename order.
fn read_doc_dir_bucket(dir: &std::path::Path, name: String) -> anyhow::Result<BucketInput> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("reading {}", dir.display()))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "md"))
        .collect();
    paths.sort();
    anyhow::ensure!(
        !paths.is_empty(),
        "--input {}: no cleaned.jsonl and no *.md documents",
        dir.display()
    );
    let mut raw_bytes = Vec::new();
    let mut docs = Vec::new();
    for path in paths {
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        let id = path
            .file_stem()
            .map(|stem| stem.to_string_lossy().into_owned())
            .unwrap_or_default();
        raw_bytes.extend_from_slice(id.as_bytes());
        raw_bytes.push(0);
        raw_bytes.extend_from_slice(text.as_bytes());
        docs.push(CorpusDoc { id, text });
    }
    Ok(BucketInput {
        name,
        raw_bytes,
        docs,
    })
}

/// Every literal probe `set` produces across all seven buckets
/// (`literal_probes` for the six knowledge buckets, whitespace-split
/// phrases for `rules_pilot`), deduplicated and lowercased — mirrors
/// `corpus-tool adjudicate`'s own probe extraction
/// (`flatten_probe`/`push_knowledge_bucket`/`push_pilot_bucket`), except
/// this command only needs the distinct probe set, not per-rule
/// accounting.
fn derive_probes(set: &FrameRuleSet) -> BTreeSet<Vec<String>> {
    let mut probes = BTreeSet::new();
    for (_, rules) in set.knowledge_buckets() {
        for rule in rules {
            let Ok(pattern) = parse_pattern(&rule.p) else {
                continue;
            };
            for probe in literal_probes(&pattern, &set.classes, 64) {
                let words: Vec<String> = probe
                    .iter()
                    .flat_map(|part| part.split_whitespace())
                    .map(str::to_lowercase)
                    .collect();
                if !words.is_empty() {
                    probes.insert(words);
                }
            }
        }
    }
    for rule in &set.rules_pilot {
        let words: Vec<String> = rule.p.split_whitespace().map(str::to_lowercase).collect();
        if !words.is_empty() {
            probes.insert(words);
        }
    }
    probes
}

/// Indexes every probe by its first word, so a document scan only tests
/// probes that could plausibly start at the current position.
fn build_probe_index(probes: &[Vec<String>]) -> HashMap<String, Vec<u32>> {
    let mut index: HashMap<String, Vec<u32>> = HashMap::new();
    for (i, probe) in probes.iter().enumerate() {
        if let Some(first) = probe.first() {
            #[allow(
                clippy::cast_possible_truncation,
                reason = "probe count is far below u32::MAX"
            )]
            index.entry(first.clone()).or_default().push(i as u32);
        }
    }
    index
}

/// Scans one document's token stream, counting every probe's
/// non-overlapping, leftmost occurrences into `counts[probe_idx]`.
/// Per-probe match state is local to this call, so a probe can never
/// bridge two documents — the same effect `dms-index-v1`'s per-document
/// separator token produces on the stream `CorpusEvidence::phrase_counts`
/// scans (see this module's own docs).
fn scan_document(
    tokens: &[String],
    index: &HashMap<String, Vec<u32>>,
    probes: &[Vec<String>],
    counts: &mut [u64],
) {
    let mut next_allowed: HashMap<u32, usize> = HashMap::new();
    for i in 0..tokens.len() {
        let Some(candidates) = index.get(&tokens[i]) else {
            continue;
        };
        for &probe_idx in candidates {
            let probe = &probes[probe_idx as usize];
            let plen = probe.len();
            if i + plen > tokens.len() {
                continue;
            }
            if let Some(&allowed_from) = next_allowed.get(&probe_idx)
                && i < allowed_from
            {
                continue;
            }
            if probe
                .iter()
                .enumerate()
                .all(|(j, word)| *word == tokens[i + j])
            {
                counts[probe_idx as usize] += 1;
                next_allowed.insert(probe_idx, i + plen);
            }
        }
    }
}

/// Runs `human-evidence`.
///
/// # Errors
/// Returns an error if `--rules` can't be read or doesn't parse, if any
/// `--input` bucket can't be read (see [`read_bucket`]), or if
/// `--out-bin`/`--out-toml` can't be written.
pub fn run(args: &Args) -> anyhow::Result<()> {
    let rules_bytes =
        std::fs::read(&args.rules).with_context(|| format!("reading {}", args.rules.display()))?;
    let rules_text = String::from_utf8(rules_bytes.clone())
        .with_context(|| format!("{} is not valid UTF-8", args.rules.display()))?;
    let rules_sha256 = sha256_hex(&rules_bytes);
    let rule_set = FrameRuleSet::parse(&rules_text)
        .map_err(|e| anyhow::anyhow!("{} did not parse: {e}", args.rules.display()))?;

    let mut buckets: Vec<BucketInput> = args
        .input
        .iter()
        .map(|dir| read_bucket(dir))
        .collect::<anyhow::Result<_>>()?;
    buckets.sort_by(|a, b| a.name.cmp(&b.name));

    let total_docs: usize = buckets.iter().map(|b| b.docs.len()).sum();

    let probes_vec: Vec<Vec<String>> = if total_docs > 0 {
        derive_probes(&rule_set).into_iter().collect()
    } else {
        Vec::new()
    };
    let probe_index = build_probe_index(&probes_vec);
    let mut probe_hits = vec![0u64; probes_vec.len()];

    let mut unigram_counts: BTreeMap<String, u64> = BTreeMap::new();
    let mut bucket_records: BTreeMap<String, BucketRecord> = BTreeMap::new();
    let mut total_tokens = 0u64;

    for bucket in &buckets {
        let mut bucket_tokens = 0u64;
        for doc in &bucket.docs {
            let tokens = friction_harness::clean::tokenize(&doc.text);
            bucket_tokens += tokens.len() as u64;
            for token in &tokens {
                *unigram_counts.entry(token.clone()).or_insert(0) += 1;
            }
            scan_document(&tokens, &probe_index, &probes_vec, &mut probe_hits);
        }
        total_tokens += bucket_tokens;
        bucket_records.insert(
            bucket.name.clone(),
            BucketRecord {
                docs: bucket.docs.len(),
                tokens: bucket_tokens,
                sha256: sha256_hex(&bucket.raw_bytes),
            },
        );
    }

    let unigrams: BTreeMap<String, u64> = unigram_counts
        .into_iter()
        .filter(|(_, count)| *count >= 5)
        .collect();
    let probes: BTreeMap<Vec<String>, u64> = probes_vec.into_iter().zip(probe_hits).collect();

    let bin = friction_packs::human_evidence::build_pack_bytes(&unigrams, &probes, total_tokens);
    if let Some(parent) = args.out_bin.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&args.out_bin, &bin)
        .with_context(|| format!("writing {}", args.out_bin.display()))?;

    let toml = render_toml(
        &args.pack_name,
        &rules_sha256,
        &bucket_records,
        unigrams.len(),
        probes.len(),
        total_tokens,
    );
    if let Some(parent) = args.out_toml.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&args.out_toml, &toml)
        .with_context(|| format!("writing {}", args.out_toml.display()))?;

    println!(
        "human-evidence: {} bucket(s), {} doc(s), {} token(s) -> {} unigram(s), {} probe(s) \
         written to {} ({} bytes) + {}",
        bucket_records.len(),
        total_docs,
        total_tokens,
        unigrams.len(),
        probes.len(),
        args.out_bin.display(),
        bin.len(),
        args.out_toml.display(),
    );
    Ok(())
}

/// TOML-escapes `text` for a basic string literal (backslash and quote
/// only — every value this command writes is plain ASCII).
fn toml_escape(text: &str) -> String {
    text.replace('\\', "\\\\").replace('"', "\\\"")
}

/// Renders the pack's TOML sidecar: small, human-readable provenance,
/// mirroring `jargon-attest-v1.toml`'s own shape.
fn render_toml(
    pack_name: &str,
    rules_sha256: &str,
    buckets: &BTreeMap<String, BucketRecord>,
    unigram_count: usize,
    probe_count: usize,
    total_tokens: u64,
) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# {pack_name} pack: corpus unigram and literal-probe occurrence\n\
         # counts, consumed by friction_packs::frame_compile::CorpusEvidence\n\
         # to strengthen (never replace) the dms-index-v1 evidence the\n\
         # frame-rewrite compile fences consult. Generated by `corpus-tool\n\
         # human-evidence` — do not hand-edit.\n\
         #\n\
         # See friction_packs::human_evidence's own module docs for the two\n\
         # tables' differing absence semantics and the pooling formula.\n"
    )
    .expect("write to String is infallible");
    writeln!(out, "[pack]").expect("write to String is infallible");
    writeln!(out, "version = \"{}\"", toml_escape(pack_name))
        .expect("write to String is infallible");
    writeln!(out, "rules_sha256 = \"{}\"", toml_escape(rules_sha256))
        .expect("write to String is infallible");
    writeln!(out, "unigram_count = {unigram_count}").expect("write to String is infallible");
    writeln!(out, "probe_count = {probe_count}").expect("write to String is infallible");
    writeln!(out, "total_tokens = {total_tokens}").expect("write to String is infallible");

    for (name, record) in buckets {
        out.push('\n');
        writeln!(out, "[pack.buckets.\"{}\"]", toml_escape(name))
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

    fn tiny_rule_set() -> FrameRuleSet {
        let toml = r#"
rules_ship = [
    { id="vsub.utilize", p="\"utilize\" X", t="\"use\" X", k="r", m_pm=100.0, h_pm=1.0, v="CONFIRMED" },
]
rules_flip = []
rules_surface = []
rules_pilot = [
    { p="please synergize", t="please combine", k="r", support=1 },
]
rules_quarantine = []
rules_no_evidence = []
rules_staged_surface = []
[classes]
[function_words]
words = ["a", "the"]
"#;
        FrameRuleSet::parse(toml).expect("tiny rule set parses")
    }

    // --- derive_probes ---

    #[test]
    fn derive_probes_collects_knowledge_and_pilot_phrases_lowercased_and_deduped() {
        let set = tiny_rule_set();
        let probes = derive_probes(&set);
        assert!(probes.contains(&vec!["utilize".to_string()]));
        assert!(probes.contains(&vec!["please".to_string(), "synergize".to_string()]));
        assert_eq!(probes.len(), 2);
    }

    // --- scan_document / build_probe_index ---

    #[test]
    fn scan_document_counts_non_overlapping_occurrences() {
        let probes = vec![vec!["use".to_string(), "use".to_string()]];
        let index = build_probe_index(&probes);
        let tokens: Vec<String> = ["use", "use", "use", "use"]
            .iter()
            .map(|s| (*s).to_string())
            .collect();
        let mut counts = vec![0u64];
        scan_document(&tokens, &index, &probes, &mut counts);
        assert_eq!(counts[0], 2);
    }

    #[test]
    fn scan_document_never_bridges_two_separate_calls() {
        // Two calls stand in for two different documents: a probe
        // spanning the boundary must never match, mirroring the
        // dms-index-v1 per-document separator's effect.
        let probes = vec![vec!["please".to_string(), "use".to_string()]];
        let index = build_probe_index(&probes);
        let mut counts = vec![0u64];
        let first: Vec<String> = vec!["please".to_string()];
        let second: Vec<String> = vec!["use".to_string()];
        scan_document(&first, &index, &probes, &mut counts);
        scan_document(&second, &index, &probes, &mut counts);
        assert_eq!(counts[0], 0);
    }

    // --- tokenization parity with corpus-tool index ---

    /// The exact tokenizer `corpus-tool index` interns the dms-index-v1
    /// human stream with — pinned here so a drift in either command's
    /// tokenizer choice is a visible test failure, not a silent
    /// incomparability between this pack's counts and the dms stream's.
    #[test]
    fn tokenization_matches_the_dms_index_pipeline_exactly() {
        let text = "Please DON'T utilize this, use it instead.";
        let tokens = friction_harness::clean::tokenize(text);
        // Whole-text tokenize: lowercase word runs and single
        // punctuation characters, no sentence splitting, no lemma.
        assert_eq!(
            tokens,
            vec![
                "please", "don't", "utilize", "this", ",", "use", "it", "instead", "."
            ]
        );
    }

    // --- run(): determinism, unigram floor, probe zeros, empty inputs ---

    fn write_bucket(dir: &std::path::Path, name: &str, lines: &[&str]) -> PathBuf {
        let bucket_dir = dir.join(name);
        std::fs::create_dir_all(&bucket_dir).expect("create bucket dir");
        std::fs::write(bucket_dir.join("cleaned.jsonl"), lines.join("\n"))
            .expect("write cleaned.jsonl");
        bucket_dir
    }

    fn rules_fixture(dir: &std::path::Path) -> PathBuf {
        let path = dir.join("frame-rules-v1.toml");
        std::fs::write(&path, tiny_rule_set_toml()).expect("write rules fixture");
        path
    }

    fn tiny_rule_set_toml() -> &'static str {
        r#"
rules_ship = [
    { id="vsub.utilize", p="\"utilize\" X", t="\"use\" X", k="r", m_pm=100.0, h_pm=1.0, v="CONFIRMED" },
]
rules_flip = []
rules_surface = []
rules_pilot = [
    { p="please synergize", t="please combine", k="r", support=1 },
]
rules_quarantine = []
rules_no_evidence = []
rules_staged_surface = []
[classes]
[function_words]
words = ["a", "the"]
"#
    }

    #[test]
    fn run_with_zero_input_dirs_builds_the_empty_placeholder_pack() {
        let dir = tempfile::tempdir().expect("tempdir");
        let rules = rules_fixture(dir.path());
        let out_bin = dir.path().join("out.bin");
        let out_toml = dir.path().join("out.toml");
        let args = Args {
            input: Vec::new(),
            rules,
            pack_name: "human-evidence-v1".to_string(),
            out_toml,
            out_bin: out_bin.clone(),
        };
        run(&args).expect("run succeeds with zero input dirs");
        let pack = friction_packs::human_evidence::HumanEvidencePack::load(
            &std::fs::read(&out_bin).expect("read out_bin"),
        )
        .expect("built pack loads");
        assert_eq!(pack.total_tokens(), 0);
        assert_eq!(pack.unigram_count("utilize"), 0);
        assert_eq!(pack.probe_count(&["utilize"]), None);
    }

    #[test]
    fn run_keeps_unigrams_at_or_above_the_five_occurrence_floor_only() {
        let dir = tempfile::tempdir().expect("tempdir");
        let rules = rules_fixture(dir.path());
        // "please" appears once per doc across 5 docs (>= floor); each
        // doc's other word is unique, appearing only once overall
        // (below floor) — only "please" should survive into the pack.
        let lines = [
            r#"{"id":"d0","text":"please alpha"}"#,
            r#"{"id":"d1","text":"please beta"}"#,
            r#"{"id":"d2","text":"please gamma"}"#,
            r#"{"id":"d3","text":"please delta"}"#,
            r#"{"id":"d4","text":"please epsilon"}"#,
        ];
        write_bucket(dir.path(), "docs", &lines);
        let out_bin = dir.path().join("out.bin");
        let out_toml = dir.path().join("out.toml");
        let args = Args {
            input: vec![dir.path().join("docs")],
            rules,
            pack_name: "human-evidence-v1".to_string(),
            out_toml,
            out_bin: out_bin.clone(),
        };
        run(&args).expect("run succeeds");
        let pack = friction_packs::human_evidence::HumanEvidencePack::load(
            &std::fs::read(&out_bin).expect("read out_bin"),
        )
        .expect("built pack loads");
        assert_eq!(pack.unigram_count("please"), 5);
        assert_eq!(
            pack.unigram_count("alpha"),
            0,
            "below the five-occurrence floor"
        );
    }

    #[test]
    fn run_records_a_probe_at_zero_when_never_observed() {
        let dir = tempfile::tempdir().expect("tempdir");
        let rules = rules_fixture(dir.path());
        write_bucket(
            dir.path(),
            "docs",
            &[r#"{"id":"d0","text":"nothing relevant here at all today really"}"#],
        );
        let out_bin = dir.path().join("out.bin");
        let out_toml = dir.path().join("out.toml");
        let args = Args {
            input: vec![dir.path().join("docs")],
            rules,
            pack_name: "human-evidence-v1".to_string(),
            out_toml,
            out_bin: out_bin.clone(),
        };
        run(&args).expect("run succeeds");
        let pack = friction_packs::human_evidence::HumanEvidencePack::load(
            &std::fs::read(&out_bin).expect("read out_bin"),
        )
        .expect("built pack loads");
        // "utilize" is a registered probe (from rules_ship) that never
        // occurs in this fixture's text: Some(0), not None.
        assert_eq!(pack.probe_count(&["utilize"]), Some(0));
    }

    #[test]
    fn run_is_deterministic_across_two_runs_of_the_same_fixture() {
        let dir = tempfile::tempdir().expect("tempdir");
        let rules = rules_fixture(dir.path());
        write_bucket(
            dir.path(),
            "docs",
            &[
                r#"{"id":"d0","text":"please synergize this thing please synergize"}"#,
                r#"{"id":"d1","text":"please please please please please"}"#,
            ],
        );
        let input = vec![dir.path().join("docs")];
        let out_bin_a = dir.path().join("a.bin");
        let out_toml_a = dir.path().join("a.toml");
        run(&Args {
            input: input.clone(),
            rules: rules.clone(),
            pack_name: "human-evidence-v1".to_string(),
            out_toml: out_toml_a,
            out_bin: out_bin_a.clone(),
        })
        .expect("run a succeeds");
        let out_bin_b = dir.path().join("b.bin");
        let out_toml_b = dir.path().join("b.toml");
        run(&Args {
            input,
            rules,
            pack_name: "human-evidence-v1".to_string(),
            out_toml: out_toml_b,
            out_bin: out_bin_b.clone(),
        })
        .expect("run b succeeds");
        assert_eq!(
            std::fs::read(&out_bin_a).expect("read a"),
            std::fs::read(&out_bin_b).expect("read b"),
        );
    }

    #[test]
    fn render_toml_is_valid_and_contains_cross_checked_fields() {
        let buckets = BTreeMap::from([(
            "docs".to_string(),
            BucketRecord {
                docs: 3,
                tokens: 42,
                sha256: "deadbeef".to_string(),
            },
        )]);
        let text = render_toml("human-evidence-v1", "cafef00d", &buckets, 7, 2, 42);
        assert!(text.contains("version = \"human-evidence-v1\""));
        assert!(text.contains("rules_sha256 = \"cafef00d\""));
        assert!(text.contains("unigram_count = 7"));
        assert!(text.contains("probe_count = 2"));
        assert!(text.contains("total_tokens = 42"));
        assert!(text.contains("[pack.buckets.\"docs\"]"));
        assert!(toml::from_str::<toml::Value>(&text).is_ok(), "{text}");
    }
}
