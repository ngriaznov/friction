//! `corpus-tool index` — builds the per-model-family and human token-id
//! streams over one shared vocabulary.
//!
//! `friction_packs::dms::DmsIndex` reconstructs those streams into
//! differential-matching-statistics suffix automata at load time.
//!
//! # Tokenization convention — deliberately different from `mine-inventory`
//!
//! This is a faithful transcription of `ref_dms.py`, which cleans and
//! tokenizes the *raw whole-file text* directly — no `friction-parse`
//! block-tree prose extraction at all. That is the exact pipeline the
//! reference's 16/16 held-out classification and its size/throughput
//! numbers were measured against, so changing the input representation
//! here would invalidate the transcription. `corpus-tool mine-inventory`,
//! by contrast, extracts prose via `friction-parse` first (see that
//! module's own doc comment for why its mining needs the real block
//! tree) — the two commands read the same corpus files through two
//! different, deliberately-not-unified pipelines.
//!
//! # Vocabulary and streams
//!
//! Every `train`-split `llm` document is grouped by [`family_of`] (a hard
//! error for an unmapped model name — see that function). Every
//! `train`-split `human` document forms one more pool. Docs within each
//! group are processed in ascending `id` order, and the five groups
//! themselves in the fixed order `[Qwen, Gemma, Llama, Granite, Human]` —
//! this fixed traversal order is what makes the shared vocabulary's
//! first-seen-order id assignment (and therefore the whole pack)
//! deterministic byte-for-byte across reruns of the same corpus.
//!
//! Each document contributes `friction_harness::clean::tokenize(raw
//! text)` (word and punctuation tokens, faithfully mirroring
//! `ref_dms.py::toks(clean(text))`) followed by one reserved separator
//! token, `"\u{0}"`, exactly matching `ref_dms.py`'s own `"\x00"`
//! document-boundary convention (it prevents a matching-statistics walk
//! from ever bridging two different documents). The separator is interned
//! into the vocabulary immediately after the reserved id-`0` placeholder,
//! so it is always id `1` — recorded redundantly as `separator_token_id`
//! in the pack header so a reader never has to search `[vocab].tokens`
//! for the one entry that is a literal NUL character.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::PathBuf;

use clap::Args as ClapArgs;
use friction_packs::{DmsIndex, ModelFamily};

use crate::corpus_layout::relpath;
use crate::hashing::sha256_hex;
use crate::manifest::{self, Class, ManifestRecord, Split};

/// The reserved separator token appended after every document's tokens —
/// `ref_dms.py`'s own `"\x00"` document-boundary convention.
const SEPARATOR_TOKEN: &str = "\u{0}";

/// Arguments for `corpus-tool index`.
#[derive(Debug, ClapArgs)]
pub struct Args {
    /// Corpus root directory.
    #[arg(long, default_value = "corpus")]
    pub corpus_dir: PathBuf,
    /// Path to write the versioned DMS index pack to.
    #[arg(long, default_value = "crates/friction-packs/packs/dms-index-v1.toml")]
    pub pack: PathBuf,
    /// Also build the five automata in-process and report dev-split
    /// (never holdout) per-family classification accuracy — a regression
    /// signal reproducing the validated 16/16 experiment, not part of the
    /// pack itself. Slower; off by default.
    #[arg(long, default_value_t = false)]
    pub calibrate: bool,
}

/// Maps a manifest `model.name` (an Ollama tag, e.g. `"gemma2:9b"`, or an
/// API model id, e.g. `"claude-opus-5"`) to its [`ModelFamily`] via a
/// case-insensitive substring match.
///
/// `None` for a name matching none of the five known families — `run`
/// treats that as a hard error (an unmapped model must force a deliberate
/// code change here, never a silent under-count).
#[must_use]
pub fn family_of(model_name: &str) -> Option<ModelFamily> {
    let lower = model_name.to_lowercase();
    if lower.contains("qwen") {
        Some(ModelFamily::Qwen)
    } else if lower.contains("gemma") {
        Some(ModelFamily::Gemma)
    } else if lower.contains("llama") {
        Some(ModelFamily::Llama)
    } else if lower.contains("granite") {
        Some(ModelFamily::Granite)
    } else if lower.contains("claude") || lower.contains("opus") || lower.contains("sonnet") {
        Some(ModelFamily::Claude)
    } else {
        None
    }
}

/// Interns token text into a shared, first-seen-order vocabulary.
#[derive(Debug, Default)]
struct VocabBuilder {
    tokens: Vec<String>,
    by_text: std::collections::HashMap<String, u32>,
}

impl VocabBuilder {
    /// A fresh builder with the reserved, unused id-`0` placeholder
    /// already inserted.
    fn new() -> Self {
        let mut builder = Self::default();
        builder.tokens.push("<unused-0>".to_string());
        builder
    }

    /// Returns `token`'s id, assigning it the next free id on first sight.
    fn intern(&mut self, token: &str) -> u32 {
        if let Some(&id) = self.by_text.get(token) {
            return id;
        }
        #[allow(clippy::cast_possible_truncation)]
        let id = self.tokens.len() as u32;
        self.tokens.push(token.to_string());
        self.by_text.insert(token.to_string(), id);
        id
    }
}

/// One family or human stream's built token-id sequence plus its doc
/// count, for the pack's `[pack.families]` summary.
struct StreamOutput {
    doc_count: usize,
    ids: Vec<u32>,
}

/// Runs `index`.
///
/// # Errors
///
/// Returns an error if the manifest or a referenced document can't be
/// read, if any `train`-split `llm` document's `model.name` does not map
/// to a known [`ModelFamily`] (see [`family_of`]), or if `--pack` can't be
/// written.
pub fn run(args: &Args) -> anyhow::Result<()> {
    let manifest_path = args.corpus_dir.join("manifest.jsonl");
    let manifest_bytes = std::fs::read(&manifest_path).unwrap_or_default();
    let records = manifest::read_manifest(&manifest_path)?.unwrap_or_default();

    let mut train: Vec<&ManifestRecord> = records
        .iter()
        .filter(|r| r.split == Some(Split::Train))
        .collect();
    train.sort_by(|a, b| a.id.cmp(&b.id));

    let mut by_family: BTreeMap<ModelFamily, Vec<&ManifestRecord>> = BTreeMap::new();
    let mut human_docs: Vec<&ManifestRecord> = Vec::new();

    for &record in &train {
        match record.class {
            Class::Human => human_docs.push(record),
            Class::Llm => {
                let model_name = record.model.as_ref().map_or("", |m| m.name.as_str());
                let family = family_of(model_name).ok_or_else(|| {
                    anyhow::anyhow!(
                        "index: unmapped model family for doc {} (model {model_name:?})",
                        record.id
                    )
                })?;
                by_family.entry(family).or_default().push(record);
            }
        }
    }
    for docs in by_family.values_mut() {
        docs.sort_by(|a, b| a.id.cmp(&b.id));
    }
    human_docs.sort_by(|a, b| a.id.cmp(&b.id));

    let mut vocab = VocabBuilder::new();
    // Interned immediately so the separator is always id 1, deterministic
    // regardless of which family's first document happens to run first.
    let separator_token_id = vocab.intern(SEPARATOR_TOKEN);

    let mut family_streams: BTreeMap<ModelFamily, StreamOutput> = BTreeMap::new();
    for family in ModelFamily::ALL {
        let docs = by_family.get(&family).cloned().unwrap_or_default();
        if docs.is_empty() {
            eprintln!(
                "warning: index: no train-split docs for model family \"{}\"; omitted from pack",
                family.as_str()
            );
            continue;
        }
        let ids = build_stream(args, &docs, &mut vocab, separator_token_id)?;
        family_streams.insert(
            family,
            StreamOutput {
                doc_count: docs.len(),
                ids,
            },
        );
    }

    let human_ids = build_stream(args, &human_docs, &mut vocab, separator_token_id)?;

    let header = PackHeader {
        corpus_manifest_sha256: sha256_hex(&manifest_bytes),
        train_human_doc_count: human_docs.len(),
        separator_token_id,
    };
    let pack_text = render_pack(&header, &vocab.tokens, &family_streams, &human_ids);

    if let Some(parent) = args.pack.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&args.pack, &pack_text)?;
    println!(
        "index: wrote {} vocab token(s), {} family stream(s), {} human doc(s) to {}",
        vocab.tokens.len(),
        family_streams.len(),
        human_docs.len(),
        args.pack.display()
    );

    if args.calibrate {
        run_calibration(args, &pack_text)?;
    }

    Ok(())
}

/// Reads, tokenizes, and interns every doc in `docs` (already sorted by
/// id), appending the separator token after each one.
fn build_stream(
    args: &Args,
    docs: &[&ManifestRecord],
    vocab: &mut VocabBuilder,
    separator_token_id: u32,
) -> anyhow::Result<Vec<u32>> {
    let mut ids = Vec::new();
    for record in docs {
        let path = args.corpus_dir.join(relpath(record));
        let bytes = std::fs::read(&path)
            .map_err(|e| anyhow::anyhow!("index: failed to read {}: {e}", path.display()))?;
        let text = String::from_utf8(bytes)
            .map_err(|e| anyhow::anyhow!("index: {} is not valid UTF-8: {e}", path.display()))?;
        for token in friction_harness::clean::tokenize(&text) {
            ids.push(vocab.intern(&token));
        }
        ids.push(separator_token_id);
    }
    Ok(ids)
}

struct PackHeader {
    corpus_manifest_sha256: String,
    train_human_doc_count: usize,
    separator_token_id: u32,
}

/// Escapes `text` into a valid double-quoted TOML basic-string literal.
///
/// Every token this module writes is plain ASCII except the one reserved
/// separator token (a literal NUL character), so this only ever has real
/// work to do for that one entry — but it is applied uniformly to every
/// token for safety, rather than special-casing the separator, since a
/// hostile or unusual corpus document could in principle tokenize to a
/// control character some other way `friction_harness::clean::tokenize`
/// does not anticipate today.
fn toml_quote(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for ch in text.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\u{8}' => out.push_str("\\b"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\u{c}' => out.push_str("\\f"),
            '\r' => out.push_str("\\r"),
            c if (c as u32) < 0x20 || c as u32 == 0x7F => {
                write!(out, "\\u{:04X}", c as u32).expect("write to String is infallible");
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Renders a comma-joined `ids` field value: see the module docs (and
/// `crates/friction-packs/packs/dms-index-v1.toml`'s own header comment,
/// once generated) for why this is one string rather than a native TOML
/// integer array.
fn render_ids_csv(ids: &[u32]) -> String {
    ids.iter().map(u32::to_string).collect::<Vec<_>>().join(",")
}

fn render_pack(
    header: &PackHeader,
    vocab_tokens: &[String],
    family_streams: &BTreeMap<ModelFamily, StreamOutput>,
    human_ids: &[u32],
) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# dms-index-v1 pack: per-model-family and human token-id streams over one shared \
         vocabulary, for reconstructing differential-matching-statistics suffix automata at \
         load time (see `friction_packs::dms`)."
    )
    .expect("write to String is infallible");
    writeln!(
        out,
        "# Generated by `corpus-tool index` — do not hand-edit."
    )
    .expect("write to String is infallible");
    writeln!(out).expect("write to String is infallible");

    writeln!(out, "[pack]").expect("write to String is infallible");
    writeln!(out, "version = \"dms-index-v1\"").expect("write to String is infallible");
    writeln!(
        out,
        "corpus_manifest_sha256 = {}",
        toml_quote(&header.corpus_manifest_sha256)
    )
    .expect("write to String is infallible");
    writeln!(
        out,
        "train_human_doc_count = {}",
        header.train_human_doc_count
    )
    .expect("write to String is infallible");
    writeln!(
        out,
        "separator_token_id = {} # documented, not literally embeddable as a readable TOML string",
        header.separator_token_id
    )
    .expect("write to String is infallible");
    writeln!(out).expect("write to String is infallible");

    writeln!(out, "[pack.families]").expect("write to String is infallible");
    for (family, stream) in family_streams {
        writeln!(
            out,
            "{} = {{ doc_count = {}, token_count = {} }}",
            family.as_str(),
            stream.doc_count,
            stream.ids.len()
        )
        .expect("write to String is infallible");
    }
    writeln!(out).expect("write to String is infallible");

    writeln!(out, "[vocab]").expect("write to String is infallible");
    writeln!(out, "tokens = [").expect("write to String is infallible");
    for token in vocab_tokens {
        writeln!(out, "    {},", toml_quote(token)).expect("write to String is infallible");
    }
    writeln!(out, "]").expect("write to String is infallible");

    writeln!(out).expect("write to String is infallible");
    writeln!(out, "[streams.human]").expect("write to String is infallible");
    writeln!(out, "ids = {}", toml_quote(&render_ids_csv(human_ids)))
        .expect("write to String is infallible");

    for (family, stream) in family_streams {
        writeln!(out).expect("write to String is infallible");
        writeln!(out, "[streams.{}]", family.as_str()).expect("write to String is infallible");
        writeln!(out, "ids = {}", toml_quote(&render_ids_csv(&stream.ids)))
            .expect("write to String is infallible");
    }

    out
}

/// `--calibrate`: parses the just-written pack back into a [`DmsIndex`]
/// and reports, per family, how many `dev`-split (never `holdout`) docs
/// the sign of `mean(mM) - mean(mH)` classifies correctly against the
/// human automaton — the same statistic and sign convention
/// `ref_dms.py`'s own held-out classification uses. Printed under an
/// explicitly labeled heading so it is never mistaken for a holdout
/// result; this is a recommended regression signal, not part of the pack
/// format itself.
fn run_calibration(args: &Args, pack_text: &str) -> anyhow::Result<()> {
    let index = DmsIndex::parse(pack_text)?;

    let manifest_path = args.corpus_dir.join("manifest.jsonl");
    let records = manifest::read_manifest(&manifest_path)?.unwrap_or_default();
    let mut dev: Vec<&ManifestRecord> = records
        .iter()
        .filter(|r| r.split == Some(Split::Dev))
        .collect();
    dev.sort_by(|a, b| a.id.cmp(&b.id));

    println!("dev calibration (not holdout):");
    for family in ModelFamily::ALL {
        let Some(family_sam) = index.family_sam(family) else {
            continue;
        };
        let mut correct = 0usize;
        let mut total = 0usize;
        for record in &dev {
            let expected_machine = match record.class {
                Class::Llm => {
                    family_of(record.model.as_ref().map_or("", |m| &m.name)) == Some(family)
                }
                Class::Human => true,
            };
            if !expected_machine && record.class == Class::Llm {
                continue;
            }
            let path = args.corpus_dir.join(relpath(record));
            let Ok(bytes) = std::fs::read(&path) else {
                continue;
            };
            let Ok(text) = String::from_utf8(bytes) else {
                continue;
            };
            let query: Vec<Option<u32>> = friction_harness::clean::tokenize(&text)
                .iter()
                .map(|t| index.vocab().id_of(t))
                .collect();
            if query.is_empty() {
                continue;
            }
            let m_stats = family_sam.matching_stats(&query);
            let h_stats = index.human_sam().matching_stats(&query);
            #[allow(clippy::cast_precision_loss)]
            let diff = (f64::from(m_stats.iter().sum::<u32>())
                - f64::from(h_stats.iter().sum::<u32>()))
                / query.len() as f64;
            let predicted_machine = diff > 0.0;
            let actual_machine = record.class == Class::Llm;
            if predicted_machine == actual_machine {
                correct += 1;
            }
            total += 1;
        }
        println!(
            "  {}: {correct}/{total} dev doc(s) classified correctly",
            family.as_str()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- family_of ---

    #[test]
    fn family_of_maps_every_real_manifest_model_name() {
        assert_eq!(family_of("gemma2:9b"), Some(ModelFamily::Gemma));
        assert_eq!(family_of("gemma3:4b"), Some(ModelFamily::Gemma));
        assert_eq!(family_of("granite4.1:3b"), Some(ModelFamily::Granite));
        assert_eq!(family_of("llama3.1:8b"), Some(ModelFamily::Llama));
        assert_eq!(family_of("llama3.2:3b"), Some(ModelFamily::Llama));
        assert_eq!(family_of("qwen2.5:7b-instruct"), Some(ModelFamily::Qwen));
    }

    #[test]
    fn family_of_is_case_insensitive() {
        assert_eq!(family_of("QWEN2.5:7B-INSTRUCT"), Some(ModelFamily::Qwen));
    }

    #[test]
    fn family_of_returns_none_for_unmapped_name() {
        assert_eq!(family_of("mystery-model:1b"), None);
    }

    // --- toml_quote ---

    #[test]
    fn toml_quote_escapes_the_reserved_separator_token() {
        assert_eq!(toml_quote(SEPARATOR_TOKEN), "\"\\u0000\"");
    }

    #[test]
    fn toml_quote_leaves_plain_ascii_tokens_alone() {
        assert_eq!(toml_quote("hello"), "\"hello\"");
        assert_eq!(toml_quote("don't"), "\"don't\"");
    }

    #[test]
    fn toml_quote_escapes_backslash_and_quote() {
        assert_eq!(toml_quote(r#"a"b\c"#), r#""a\"b\\c""#);
    }

    // --- render_ids_csv ---

    #[test]
    fn render_ids_csv_comma_joins() {
        assert_eq!(render_ids_csv(&[1, 2, 3]), "1,2,3");
        assert_eq!(render_ids_csv(&[]), "");
    }

    // --- VocabBuilder ---

    #[test]
    fn vocab_builder_reserves_id_zero_and_assigns_separator_id_one() {
        let mut vocab = VocabBuilder::new();
        assert_eq!(vocab.tokens[0], "<unused-0>");
        let sep_id = vocab.intern(SEPARATOR_TOKEN);
        assert_eq!(sep_id, 1);
    }

    #[test]
    fn vocab_builder_interns_first_seen_order_and_reuses_ids() {
        let mut vocab = VocabBuilder::new();
        let a1 = vocab.intern("the");
        let b1 = vocab.intern("quick");
        let a2 = vocab.intern("the");
        assert_eq!(a1, a2, "re-interning the same text returns the same id");
        assert_ne!(a1, b1);
        assert_eq!(vocab.tokens[a1 as usize], "the");
    }
}
