//! `corpus-tool jargon-attest` — builds `jargon-attest-v1`, the web-scale
//! compound attestation pack `friction_packs::JargonAttestPack` loads at
//! runtime (SYNTHESIS.md §4).
//!
//! Deterministic and fully offline: both inputs are local files (already
//! downloaded and staged by whoever runs this — this command never makes
//! a network request), normalized through the ONE shared function
//! (`friction_packs::jargon_attest::normalize_compound`) the runtime pack
//! also uses, deduped, sorted, and built into a `BinaryFuse8` filter via
//! `friction_packs::jargon_attest::build_pack_bytes` — see that module's
//! own docs for the determinism guarantee this command relies on.
//!
//! # Wikipedia title candidate extraction
//!
//! `--titles` is expected to already be pre-filtered to multiword (2-4
//! underscore-separated segments), concept-shaped `MediaWiki` page titles
//! (one raw title per line, e.g. `Data_fabric`) — this command does not
//! re-derive that shape from a raw Wikipedia dump. It does apply one
//! additional structural filter of its own: a line whose first character
//! is a lowercase ASCII letter is dropped. `MediaWiki` always capitalizes a
//! page title's first character (a raw dump title never legitimately
//! starts lowercase), so a lowercase-initial line is necessarily a
//! non-title artifact of however `--titles` was produced — in practice,
//! the raw `enwiki-latest-all-titles-in-ns0.gz` dump's own `page_title`
//! TSV header line, which survives naive "2-4 underscore segments"
//! pre-filtering (`page_title` -> `page`, `title`, two segments) because
//! that filter never checked capitalization. This command's own
//! word-count filter (after normalization, independent of the input
//! file's own pre-filtering) still applies on top of this.
//!
//! # `OpenAlex` topic candidate extraction
//!
//! `--openalex` is one topic display name per line, mixed case, no
//! underscores — no lowercase-initial filter (`OpenAlex` names are not
//! guaranteed title-case, and are CC0 either way).

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use anyhow::Context as _;
use clap::Args as ClapArgs;
use friction_packs::jargon_attest::{in_word_range, normalize_compound};

use crate::hashing::sha256_hex;

/// Arguments for `corpus-tool jargon-attest`.
#[derive(Debug, ClapArgs)]
pub struct Args {
    /// Path to the newline-separated, pre-filtered Wikipedia title corpus
    /// (raw `MediaWiki` titles, underscores as word separators — e.g.
    /// `Data_fabric`). Required: this pack always needs a real input
    /// corpus, never a network fetch.
    #[arg(long)]
    pub titles: PathBuf,
    /// Path to the newline-separated `OpenAlex` Topics display-name corpus
    /// (mixed case, space-separated — e.g. `Magnetic confinement fusion
    /// research`).
    #[arg(long)]
    pub openalex: PathBuf,
    /// The sha256 of the raw Wikipedia titles dump `--titles` was
    /// pre-filtered from (`enwiki-latest-all-titles-in-ns0.gz`), for the
    /// sidecar's provenance record. This command does not (re-)download
    /// or read that dump itself — hex digest supplied by the caller, who
    /// computed it once against the actual dump file.
    #[arg(long)]
    pub source_dump_sha256: String,
    /// Path to write the built filter to.
    #[arg(
        long,
        default_value = "crates/friction-packs/packs/jargon-attest-v1.bin"
    )]
    pub out_bin: PathBuf,
    /// Path to write the pack's TOML sidecar to.
    #[arg(
        long,
        default_value = "crates/friction-packs/packs/jargon-attest-v1.toml"
    )]
    pub out_sidecar: PathBuf,
}

/// One input source's contribution: how many raw lines were read, how
/// many survived this command's filters and were folded into the shared
/// key set, and the input file's own sha256 (for the sidecar).
#[derive(Debug, Clone, Copy)]
struct SourceStats {
    lines_read: usize,
    keys_kept: usize,
}

/// Runs `jargon-attest`.
///
/// Reads `--titles` and `--openalex`, normalizes and filters every line
/// (see this module's own docs for each source's extra rule), unions the
/// results into one sorted, deduped key set, builds the `BinaryFuse8`
/// filter, and writes both the `.bin` and its `.toml` sidecar.
///
/// # Errors
/// Returns an error if either input file can't be read, if filter
/// construction fails (see
/// [`friction_packs::PackError::JargonAttestConstructionFailed`]), or if
/// `--out-bin`/`--out-sidecar` can't be written.
pub fn run(args: &Args) -> anyhow::Result<()> {
    let titles_bytes = std::fs::read(&args.titles)
        .with_context(|| format!("reading {}", args.titles.display()))?;
    let titles_text = String::from_utf8(titles_bytes.clone())
        .with_context(|| format!("{} is not valid UTF-8", args.titles.display()))?;
    let titles_sha256 = sha256_hex(&titles_bytes);

    let openalex_bytes = std::fs::read(&args.openalex)
        .with_context(|| format!("reading {}", args.openalex.display()))?;
    let openalex_text = String::from_utf8(openalex_bytes.clone())
        .with_context(|| format!("{} is not valid UTF-8", args.openalex.display()))?;
    let openalex_sha256 = sha256_hex(&openalex_bytes);

    let mut keys: BTreeSet<String> = BTreeSet::new();
    let titles_stats = fold_wikipedia_titles(&titles_text, &mut keys);
    let openalex_stats = fold_openalex_topics(&openalex_text, &mut keys);

    let built = friction_packs::jargon_attest::build_pack_bytes(&keys)?;

    if let Some(parent) = args.out_bin.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&args.out_bin, &built.bytes)
        .with_context(|| format!("writing {}", args.out_bin.display()))?;

    let sidecar = render_sidecar(&SidecarInput {
        key_count: built.key_count,
        titles_path: &args.titles,
        titles_stats,
        titles_sha256: &titles_sha256,
        source_dump_sha256: &args.source_dump_sha256,
        openalex_path: &args.openalex,
        openalex_stats,
        openalex_sha256: &openalex_sha256,
    });
    if let Some(parent) = args.out_sidecar.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&args.out_sidecar, &sidecar)
        .with_context(|| format!("writing {}", args.out_sidecar.display()))?;

    println!(
        "jargon-attest: {} title line(s) ({} kept) + {} openalex line(s) ({} kept) -> {} distinct \
         key(s) ({} hash collision(s) dropped) written to {} ({} bytes) + {}",
        titles_stats.lines_read,
        titles_stats.keys_kept,
        openalex_stats.lines_read,
        openalex_stats.keys_kept,
        built.key_count,
        built.hash_collisions,
        args.out_bin.display(),
        built.bytes.len(),
        args.out_sidecar.display(),
    );
    Ok(())
}

/// Folds every qualifying line of `text` (raw Wikipedia titles) into
/// `keys`, applying the lowercase-initial-line drop (see module docs)
/// before normalization, then the shared 2-4-word range after it.
fn fold_wikipedia_titles(text: &str, keys: &mut BTreeSet<String>) -> SourceStats {
    let mut lines_read = 0usize;
    let mut keys_kept = 0usize;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        lines_read += 1;
        if line.starts_with(|c: char| c.is_ascii_lowercase()) {
            // Not a legitimate MediaWiki title (see module docs) —
            // e.g. the raw dump's own `page_title` TSV header.
            continue;
        }
        let normalized = normalize_compound(line);
        if in_word_range(&normalized) {
            keys.insert(normalized);
            keys_kept += 1;
        }
    }
    SourceStats {
        lines_read,
        keys_kept,
    }
}

/// Folds every qualifying line of `text` (`OpenAlex` topic display names)
/// into `keys` — normalize, then the shared 2-4-word range; no
/// capitalization filter (see module docs).
fn fold_openalex_topics(text: &str, keys: &mut BTreeSet<String>) -> SourceStats {
    let mut lines_read = 0usize;
    let mut keys_kept = 0usize;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        lines_read += 1;
        let normalized = normalize_compound(line);
        if in_word_range(&normalized) {
            keys.insert(normalized);
            keys_kept += 1;
        }
    }
    SourceStats {
        lines_read,
        keys_kept,
    }
}

/// `path`'s file name only (falling back to its full display form if it
/// has none) — the sidecar records provenance for humans reading a
/// checked-in file, not the ephemeral absolute path the builder happened
/// to run against.
fn input_basename(path: &Path) -> String {
    path.file_name().map_or_else(
        || path.display().to_string(),
        |name| name.to_string_lossy().into_owned(),
    )
}

struct SidecarInput<'a> {
    key_count: u64,
    titles_path: &'a Path,
    titles_stats: SourceStats,
    titles_sha256: &'a str,
    source_dump_sha256: &'a str,
    openalex_path: &'a Path,
    openalex_stats: SourceStats,
    openalex_sha256: &'a str,
}

/// Renders the `jargon-attest-v1.toml` sidecar: a `[pack]` table with the
/// two fields `JargonAttestPack::load` cross-checks (`version`,
/// `key_count`) plus documentary provenance fields it doesn't re-parse,
/// and one `[pack.sources.*]` table per input source recording licensing,
/// full input-file sha256, and (for Wikipedia) the raw source dump's own
/// sha256.
fn render_sidecar(input: &SidecarInput<'_>) -> String {
    format!(
        "# jargon-attest-v1 pack: a BinaryFuse8 filter over normalized, \
         multiword Wikipedia-title and OpenAlex-topic compound keys \
         (SYNTHESIS.md \u{a7}4). Generated by `corpus-tool jargon-attest` — do \
         not hand-edit.\n\
         #\n\
         # `friction_packs::JargonAttestPack::load` cross-checks `version` \
         and `key_count` below against the `.bin`'s own embedded header; \
         every other field here is provenance for humans, not re-parsed.\n\n\
         [pack]\n\
         version = \"jargon-attest-v1\"\n\
         key_count = {key_count}\n\
         filter_kind = \"xorf::BinaryFuse8\"\n\
         false_positive_rate_approx = \"0.4%\"\n\
         hash_function = \"xxh64 (xxhash-rust), fixed seed\"\n\
         hash_seed_hex = \"0x4652494354494f4e\"\n\
         normalization = \"case-fold; underscores and hyphens folded to a \
         single space each; whitespace runs collapsed to one space and \
         trimmed; kept only with 2-4 words after normalization \
         (friction_packs::jargon_attest::normalize_compound)\"\n\
         determinism = \"bit-identical .bin for a fixed, identically-ordered \
         input key set: xorf's BinaryFuse8 construction retries a \
         deterministic seed sequence (splitmix64 from a fixed initial \
         state), never external randomness, and this builder always feeds \
         keys in BTreeSet-ascending order — see \
         friction_packs::jargon_attest's own module docs and its \
         determinism_is_bit_identical_for_a_fixed_input_set test\"\n\
         collision_direction = \"safe: a false 'attested' only ever \
         suppresses a jargon.metaphor flag, never invents one\"\n\n\
         [pack.sources.wikipedia]\n\
         description = \"Multiword MediaWiki page titles, ns0 (articles), \
         pre-filtered to 2-4 underscore-separated segments\"\n\
         input_file = {titles_path:?}\n\
         input_file_sha256 = \"{titles_sha256}\"\n\
         input_lines_read = {titles_lines}\n\
         input_lines_kept = {titles_kept}\n\
         source_dump = \"enwiki-latest-all-titles-in-ns0.gz\"\n\
         source_dump_sha256 = \"{source_dump_sha256}\"\n\
         license = \"CC BY-SA 4.0 and GFDL\"\n\
         attribution = \"Wikipedia contributors, https://en.wikipedia.org/ \
         — titles only (no article text) are used, as bare compound keys \
         in a membership filter\"\n\n\
         [pack.sources.openalex]\n\
         description = \"OpenAlex Topics display names\"\n\
         input_file = {openalex_path:?}\n\
         input_file_sha256 = \"{openalex_sha256}\"\n\
         input_lines_read = {openalex_lines}\n\
         input_lines_kept = {openalex_kept}\n\
         license = \"CC0\"\n\
         attribution = \"OpenAlex (https://openalex.org/), CC0 — no \
         attribution required, credited here as a courtesy\"\n",
        key_count = input.key_count,
        titles_path = input_basename(input.titles_path),
        titles_sha256 = input.titles_sha256,
        titles_lines = input.titles_stats.lines_read,
        titles_kept = input.titles_stats.keys_kept,
        source_dump_sha256 = input.source_dump_sha256,
        openalex_path = input_basename(input.openalex_path),
        openalex_sha256 = input.openalex_sha256,
        openalex_lines = input.openalex_stats.lines_read,
        openalex_kept = input.openalex_stats.keys_kept,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fold_wikipedia_titles_drops_lowercase_initial_header_artifact() {
        let mut keys = BTreeSet::new();
        let stats = fold_wikipedia_titles("page_title\nData_fabric\nA\n", &mut keys);
        assert_eq!(stats.lines_read, 3);
        // "page_title" dropped (lowercase-initial); "A" dropped (1 word
        // after normalization, below the 2-word floor).
        assert_eq!(stats.keys_kept, 1);
        assert!(keys.contains("data fabric"));
        assert!(!keys.contains("page title"));
    }

    #[test]
    fn fold_wikipedia_titles_applies_word_range_after_hyphen_expansion() {
        let mut keys = BTreeSet::new();
        // 2 underscore segments, but "Cross-domain" internally hyphenates
        // to 2 words -> 3 words total after normalization: still in range.
        fold_wikipedia_titles("Cross-domain_resonance\n", &mut keys);
        assert!(keys.contains("cross domain resonance"));
    }

    #[test]
    fn fold_openalex_topics_keeps_lowercase_initial_lines() {
        let mut keys = BTreeSet::new();
        let stats = fold_openalex_topics("earthquake and tectonic studies\n", &mut keys);
        assert_eq!(stats.keys_kept, 1);
        assert!(keys.contains("earthquake and tectonic studies"));
    }

    #[test]
    fn render_sidecar_contains_cross_checked_fields_and_licensing() {
        let text = render_sidecar(&SidecarInput {
            key_count: 42,
            titles_path: Path::new("titles.txt"),
            titles_stats: SourceStats {
                lines_read: 10,
                keys_kept: 8,
            },
            titles_sha256: "deadbeef",
            source_dump_sha256: "cafef00d",
            openalex_path: Path::new("openalex.txt"),
            openalex_stats: SourceStats {
                lines_read: 5,
                keys_kept: 5,
            },
            openalex_sha256: "f00dcafe",
        });
        assert!(text.contains("version = \"jargon-attest-v1\""));
        assert!(text.contains("key_count = 42"));
        assert!(text.contains("CC BY-SA 4.0 and GFDL"));
        assert!(text.contains("CC0"));
        assert!(text.contains("cafef00d"));
        assert!(toml::from_str::<toml::Value>(&text).is_ok(), "{text}");
    }
}
