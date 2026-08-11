//! `corpus-tool general-evidence` — builds `general-evidence-v1`, the
//! optional (large, NOT-embedded) web-scale unigram/bigram membership pack
//! [`friction_packs::general_evidence`] resolves at runtime.
//!
//! Two actions, mirroring a classic map/reduce split for a corpus too
//! large to hold in memory at once:
//!
//! - [`mine`] streams plain-text `MediaWiki` dump content from stdin
//!   (the caller pipes `bzcat` — this command never decompresses or
//!   downloads anything itself), strips wiki markup, tokenizes, and
//!   flushes sorted `"key\tcount"` partial files to `--work` whenever the
//!   bigram table grows past `--flush-keys` distinct keys.
//! - [`pack`] k-way merges `--work`'s sorted partials (memory-bounded:
//!   one open reader per partial file, never the whole corpus in memory),
//!   applies `--min-count`, and builds the `.bin` via
//!   [`friction_packs::general_evidence::build_pack_bytes`].
//!
//! # Markup stripping
//!
//! Ported from this workspace's own `mine-bigrams.sh` staging script
//! (hand-rolled byte scans, not a regex crate, per this command's own
//! speed goal — this replaces what was a multi-hour `awk` pipeline):
//! templates (`{{...}}`, one nesting level), links (`[[target|display]]`
//! keeps `display`, `[[target]]` keeps `target`), tags (`<...>`),
//! entities (`&[a-z]+;`), and headings (`==+...==+`) are all stripped
//! before tokenization. Deliberately crude, same as the shell script it
//! replaces: the `--min-count` floor at pack time washes out whatever
//! tokenization noise a missed edge case leaves behind.
//!
//! # Bigrams never bridge two input lines
//!
//! Matching the original script's own `awk` (which counts adjacent
//! `$i`/`$(i+1)` fields within one input record, never across records):
//! [`mine`] tokenizes and counts each raw dump line independently, so a
//! bigram spanning the end of one line and the start of the next is never
//! recorded. A word shorter than 3 letters breaks the bigram chain at its
//! position (no bigram records it on either side) without discarding its
//! neighbors' own eligibility to pair with whatever comes after.

use std::cmp::Ordering;
use std::collections::{BTreeSet, BinaryHeap};
use std::fs::File;
use std::io::{BufRead, BufReader, Write as _};
use std::path::{Path, PathBuf};

use anyhow::Context as _;
use clap::{Args as ClapArgs, Subcommand};

use crate::hashing::sha256_hex;

/// Arguments for `corpus-tool general-evidence`.
#[derive(Debug, ClapArgs)]
pub struct Args {
    #[command(subcommand)]
    pub action: Action,
}

/// `general-evidence`'s two actions — see this module's own docs.
#[derive(Debug, Subcommand)]
pub enum Action {
    /// Mines unigram/bigram counts from stdin into sorted partial files.
    Mine(MineArgs),
    /// Merges partial files into the built `general-evidence-v1` pack.
    Pack(PackArgs),
}

/// Runs `general-evidence`.
///
/// # Errors
/// See [`mine`]/[`pack`]'s own docs.
pub fn run(args: &Args) -> anyhow::Result<()> {
    match &args.action {
        Action::Mine(args) => mine(args),
        Action::Pack(args) => pack(args),
    }
}

// ---------------------------------------------------------------------
// mine
// ---------------------------------------------------------------------

/// The default distinct-bigram-key count that triggers a partial flush —
/// matching `mine-bigrams.sh`'s own `FLUSH_KEYS` default.
const DEFAULT_FLUSH_KEYS: usize = 15_000_000;

/// Arguments for `corpus-tool general-evidence mine`.
#[derive(Debug, ClapArgs)]
pub struct MineArgs {
    /// Directory sorted partial count files are flushed to. Created if
    /// absent.
    #[arg(long)]
    pub work: PathBuf,
    /// Flush both tables' partials once the bigram table holds this many
    /// distinct keys.
    #[arg(long, default_value_t = DEFAULT_FLUSH_KEYS)]
    pub flush_keys: usize,
}

/// One dump line's worth of already-lowercased, ASCII-alpha-run tokens —
/// reused across lines (`Vec::clear` between calls) so [`mine`]'s hot
/// loop allocates no new `Vec` per line.
type TokenBuf = Vec<String>;

/// Splits `line` into maximal ASCII-alphabetic runs, lowercased — the
/// `tr -c 'A-Za-z\n' ' ' | tr 'A-Z' 'a-z'` step of `mine-bigrams.sh`,
/// ported as one byte scan instead of two `tr` passes plus a `split`.
/// Every run is kept regardless of length; [`mine`]'s own counting loop
/// applies the 3-character floor per word (see this module's own docs on
/// why a short word still breaks the bigram chain rather than being
/// dropped from the stream outright).
fn tokenize_line(line: &str, tokens: &mut TokenBuf) {
    tokens.clear();
    let bytes = line.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i].is_ascii_alphabetic() {
            let start = i;
            while i < bytes.len() && bytes[i].is_ascii_alphabetic() {
                i += 1;
            }
            // `bytes[start..i]` is pure ASCII: always a valid str slice
            // boundary regardless of what non-ASCII content surrounds it.
            tokens.push(line[start..i].to_ascii_lowercase());
        } else {
            i += 1;
        }
    }
}

/// Removes every `{{...}}` template span (one nesting level: the first
/// `"}}"` found after a `"{{"` closes it, matching `mine-bigrams.sh`'s
/// `s/\{\{[^{}]*\}\}//g`).
fn strip_templates(s: &str) -> String {
    strip_delimited_spans(s, "{{", "}}", |_inner| String::new())
}

/// Unwraps every `[[target]]`/`[[target|display]]` link span to its
/// display text (`display` if piped, `target` otherwise — matching
/// `mine-bigrams.sh`'s two link substitutions).
fn strip_links(s: &str) -> String {
    strip_delimited_spans(s, "[[", "]]", |inner| {
        inner
            .find('|')
            .map_or_else(|| inner.to_string(), |pos| inner[pos + 1..].to_string())
    })
}

/// Removes every `<...>` tag span (matching `mine-bigrams.sh`'s
/// `s/<[^>]*>//g`).
fn strip_tags(s: &str) -> String {
    strip_delimited_spans(s, "<", ">", |_inner| String::new())
}

/// One byte scan shared by [`strip_templates`]/[`strip_links`]/
/// [`strip_tags`]: copies `s` to a fresh `String`, replacing every
/// `open`...`close` span (first `close` found after an `open`, not
/// nesting-aware — same crude-but-cheap shape as the sed passes this
/// ports) with `replace_inner`'s result on the span's own inner text. An
/// unmatched trailing `open` (no `close` before the line ends) is kept
/// verbatim, mirroring a sed substitution that simply doesn't match past
/// that point.
fn strip_delimited_spans(
    s: &str,
    open: &str,
    close: &str,
    replace_inner: impl Fn(&str) -> String,
) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    loop {
        let Some(open_at) = rest.find(open) else {
            out.push_str(rest);
            return out;
        };
        out.push_str(&rest[..open_at]);
        let after_open = &rest[open_at + open.len()..];
        let Some(close_at) = after_open.find(close) else {
            out.push_str(&rest[open_at..]);
            return out;
        };
        out.push_str(&replace_inner(&after_open[..close_at]));
        rest = &after_open[close_at + close.len()..];
    }
}

/// Blanks every `&[a-z]+;` HTML/XML entity to a single space (matching
/// `mine-bigrams.sh`'s `s/&[a-z]+;/ /g`).
fn strip_entities(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'&' {
            let mut j = i + 1;
            while j < bytes.len() && bytes[j].is_ascii_lowercase() {
                j += 1;
            }
            if j > i + 1 && j < bytes.len() && bytes[j] == b';' {
                out.push(' ');
                i = j + 1;
                continue;
            }
        }
        let ch = s[i..]
            .chars()
            .next()
            .expect("i < bytes.len() implies a char follows");
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

/// Removes every `==+...==+` heading span (>= 2 `=` on each side, no `=`
/// in between — matching `mine-bigrams.sh`'s `s/==+[^=]*==+//g`).
fn strip_headings(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut pos = 0;
    while pos < bytes.len() {
        if bytes[pos] == b'=' {
            let open_start = pos;
            let mut open_end = pos;
            while open_end < bytes.len() && bytes[open_end] == b'=' {
                open_end += 1;
            }
            if open_end - open_start >= 2 {
                let mut close_start = open_end;
                while close_start < bytes.len() && bytes[close_start] != b'=' {
                    close_start += 1;
                }
                let mut close_end = close_start;
                while close_end < bytes.len() && bytes[close_end] == b'=' {
                    close_end += 1;
                }
                if close_end - close_start >= 2 {
                    pos = close_end;
                    continue;
                }
            }
        }
        let ch = s[pos..]
            .chars()
            .next()
            .expect("pos < bytes.len() implies a char follows");
        out.push(ch);
        pos += ch.len_utf8();
    }
    out
}

/// Runs every markup-stripping pass on `raw`, in the same order
/// `mine-bigrams.sh`'s `sed` script applies them.
fn strip_markup(raw: &str) -> String {
    let s = strip_templates(raw);
    let s = strip_links(&s);
    let s = strip_tags(&s);
    let s = strip_entities(&s);
    strip_headings(&s)
}

/// Tracks whether the line-by-line stdin scan is currently inside a
/// `<text ...>...</text>` element, across however many raw dump lines
/// that spans.
#[derive(Default)]
struct TextExtractor {
    in_text: bool,
}

impl TextExtractor {
    /// Feeds one raw dump line, calling `on_chunk` once per contiguous
    /// span of `<text>` element content found on this line (usually
    /// zero or one call; more than one only for a line that closes one
    /// page's `<text>` and opens another's, which real dumps never do
    /// but this handles anyway rather than assuming).
    fn feed_line(&mut self, line: &str, mut on_chunk: impl FnMut(&str)) {
        let mut rest = line;
        loop {
            if !self.in_text {
                let Some(tag_at) = rest.find("<text") else {
                    return;
                };
                let after_tag_start = &rest[tag_at + "<text".len()..];
                let Some(gt) = after_tag_start.find('>') else {
                    return;
                };
                if after_tag_start[..gt].ends_with('/') {
                    // Self-closing <text .../>: an empty page, no content.
                    rest = &after_tag_start[gt + 1..];
                    continue;
                }
                self.in_text = true;
                rest = &after_tag_start[gt + 1..];
            }
            if let Some(end_at) = rest.find("</text>") {
                on_chunk(&rest[..end_at]);
                self.in_text = false;
                rest = &rest[end_at + "</text>".len()..];
            } else {
                on_chunk(rest);
                return;
            }
        }
    }
}

/// Writes `counts` (already `BTreeMap`-sorted by key) as `"key\tcount\n"`
/// lines to a fresh `{work}/{prefix}-partial-{index:05}.txt`.
fn flush_partial(
    work: &Path,
    prefix: &str,
    index: u32,
    counts: &std::collections::BTreeMap<String, u64>,
) -> anyhow::Result<()> {
    let path = work.join(format!("{prefix}-partial-{index:05}.txt"));
    let mut out = std::io::BufWriter::new(
        File::create(&path).with_context(|| format!("creating {}", path.display()))?,
    );
    for (key, count) in counts {
        writeln!(out, "{key}\t{count}").with_context(|| format!("writing {}", path.display()))?;
    }
    out.flush()
        .with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

/// Runs `general-evidence mine`: reads plain-text `MediaWiki` dump content
/// from stdin, strips markup, tokenizes, and flushes sorted unigram/bigram
/// partial count files to `--work`.
///
/// # Errors
/// Returns an error if `--work` can't be created, if stdin can't be read,
/// or if a partial file can't be written.
// `reader` (stdin's lock) is deliberately held for the whole scan loop
// below, not tightened around each `read_line` call — this process reads
// nothing else from stdin, so there is no real contention to relieve.
#[allow(clippy::significant_drop_tightening)]
pub fn mine(args: &MineArgs) -> anyhow::Result<()> {
    std::fs::create_dir_all(&args.work)
        .with_context(|| format!("creating {}", args.work.display()))?;

    let mut unigram_counts: std::collections::BTreeMap<String, u64> =
        std::collections::BTreeMap::new();
    let mut bigram_counts: std::collections::BTreeMap<String, u64> =
        std::collections::BTreeMap::new();
    let mut part_index: u32 = 0;
    let mut lines_read: u64 = 0;
    let mut chunks_read: u64 = 0;

    let mut extractor = TextExtractor::default();
    let mut tokens: TokenBuf = Vec::new();
    let mut raw_line = String::new();
    let stdin = std::io::stdin();
    let mut reader = stdin.lock();

    loop {
        raw_line.clear();
        let n = reader.read_line(&mut raw_line).context("reading stdin")?;
        if n == 0 {
            break;
        }
        lines_read += 1;
        let line = raw_line.trim_end_matches(['\n', '\r']);

        // Cloning `line` per <text> chunk keeps the borrow checker happy
        // around `extractor`'s own `&mut self` — chunks per line are rare
        // (usually 0 or 1), so this isn't the hot path `tokenize_line`'s
        // buffer reuse targets.
        let mut chunks: Vec<String> = Vec::new();
        extractor.feed_line(line, |chunk| chunks.push(chunk.to_string()));

        for chunk in &chunks {
            chunks_read += 1;
            let cleaned = strip_markup(chunk);
            tokenize_line(&cleaned, &mut tokens);
            for word in tokens.iter().filter(|w| w.len() >= 3) {
                *unigram_counts.entry(word.clone()).or_insert(0) += 1;
            }
            for pair in tokens.windows(2) {
                let [w1, w2] = pair else {
                    unreachable!("windows(2) yields 2-element slices")
                };
                if w1.len() >= 3 && w2.len() >= 3 {
                    *bigram_counts.entry(format!("{w1} {w2}")).or_insert(0) += 1;
                }
            }
        }

        if bigram_counts.len() >= args.flush_keys {
            part_index += 1;
            flush_partial(&args.work, "unigram", part_index, &unigram_counts)?;
            flush_partial(&args.work, "bigram", part_index, &bigram_counts)?;
            unigram_counts.clear();
            bigram_counts.clear();
        }
    }

    if !unigram_counts.is_empty() || !bigram_counts.is_empty() {
        part_index += 1;
        flush_partial(&args.work, "unigram", part_index, &unigram_counts)?;
        flush_partial(&args.work, "bigram", part_index, &bigram_counts)?;
    }

    println!(
        "general-evidence mine: {lines_read} line(s), {chunks_read} <text> chunk(s) -> \
         {part_index} partial file pair(s) written to {}",
        args.work.display(),
    );
    Ok(())
}

// ---------------------------------------------------------------------
// pack
// ---------------------------------------------------------------------

/// Arguments for `corpus-tool general-evidence pack`.
#[derive(Debug, ClapArgs)]
pub struct PackArgs {
    /// Directory holding the partial count files `mine` wrote.
    #[arg(long)]
    pub work: PathBuf,
    /// A key must have been counted at least this many times (summed
    /// across every partial) to survive into the built filter.
    #[arg(long, default_value_t = 5)]
    pub min_count: u64,
    /// Separate floor for the unigram (general-vocabulary) table. The
    /// unigram filter's job is the channel's unknown-word SKIP, so a
    /// higher floor here trims corpus typos ("successfuly" clears >=5 in
    /// a 3B-token mine) without touching bigram attestation. Defaults to
    /// `--min-count` when not given.
    #[arg(long)]
    pub unigram_min_count: Option<u64>,
    /// Path to write the built `.bin` to. Deliberately has no default —
    /// unlike this crate's other packs, this one is NOT committed (see
    /// this module's own docs); a caller who wants the runtime seam
    /// (`friction_packs::general_evidence`) to pick it up automatically
    /// should point this at `$FRICTION_GENERAL_EVIDENCE` or
    /// `~/.cache/friction/general-evidence-en-v1.bin`.
    #[arg(long)]
    pub out: PathBuf,
    /// Path to write the pack's committed `.meta.toml` sidecar to —
    /// provenance only (this file is small and tracked; the `.bin` it
    /// describes is neither).
    #[arg(
        long,
        default_value = "crates/friction-packs/packs/general-evidence-en-v1.meta.toml"
    )]
    pub meta_out: PathBuf,
    /// Human-readable label for the mined corpus's provenance, recorded
    /// in the `.meta.toml` (e.g. `"enwiki-20260701-pages-articles"`).
    #[arg(long)]
    pub source_label: String,
    /// The build's own timestamp, recorded in the `.meta.toml` verbatim —
    /// supplied by the caller, never read from system time, so a rebuild
    /// from identical `--work` partials produces an identical `.meta.toml`
    /// (an ISO-8601 date/time is the natural choice, but this command
    /// doesn't parse or validate the string).
    #[arg(long)]
    pub built_at: String,
    /// Allow `--out` to resolve inside the repo working tree — normally
    /// refused (see this module's own docs: the `.bin` is a large runtime
    /// artifact and must never be committed).
    #[arg(long)]
    pub force_repo_out: bool,
}

/// A `key\tcount` partial file, opened with its next unread line already
/// parsed and buffered — the unit [`k_way_merge`]'s heap tracks.
struct PartialReader {
    reader: BufReader<File>,
    next: Option<(String, u64)>,
}

impl PartialReader {
    fn open(path: &Path) -> anyhow::Result<Self> {
        let reader = BufReader::new(
            File::open(path).with_context(|| format!("opening {}", path.display()))?,
        );
        let mut this = Self { reader, next: None };
        this.advance()?;
        Ok(this)
    }

    /// Reads this reader's next `"key\tcount"` line into `self.next`, or
    /// leaves it `None` at end of file.
    fn advance(&mut self) -> anyhow::Result<()> {
        let mut line = String::new();
        let n = self.reader.read_line(&mut line)?;
        if n == 0 {
            self.next = None;
            return Ok(());
        }
        let line = line.trim_end_matches('\n');
        let (key, count) = line
            .rsplit_once('\t')
            .with_context(|| format!("malformed partial line: {line:?}"))?;
        self.next = Some((key.to_string(), count.parse::<u64>()?));
        Ok(())
    }
}

/// One heap entry: the reader index it came from, so [`k_way_merge`] can
/// advance exactly that reader once this entry is popped. `Ord`/`PartialEq`
/// compare `key` only — `count`/`reader_idx` are payload, not ranking —
/// making this a min-heap on `key` (reversed against [`BinaryHeap`]'s own
/// max-heap default).
struct HeapItem {
    key: String,
    count: u64,
    reader_idx: usize,
}

impl PartialEq for HeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}
impl Eq for HeapItem {}
impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        other.key.cmp(&self.key)
    }
}
impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// K-way merges every already-sorted partial file in `paths`, summing a
/// key's count across every partial that has it, and returns the
/// `BTreeSet` of keys whose summed count is `>= min_count`.
///
/// Memory-bounded in the number of partial files (one open [`PartialReader`]
/// each) plus the surviving key set itself — never the full unmerged
/// corpus. `paths` need not be sorted by the caller; each file is
/// independently already key-sorted (by [`flush_partial`]), which is the
/// only ordering this merge relies on.
fn k_way_merge(paths: &[PathBuf], min_count: u64) -> anyhow::Result<BTreeSet<String>> {
    let mut readers: Vec<PartialReader> = paths
        .iter()
        .map(|p| PartialReader::open(p))
        .collect::<anyhow::Result<_>>()?;

    let mut heap: BinaryHeap<HeapItem> = BinaryHeap::new();
    for (idx, reader) in readers.iter().enumerate() {
        if let Some((key, count)) = &reader.next {
            heap.push(HeapItem {
                key: key.clone(),
                count: *count,
                reader_idx: idx,
            });
        }
    }

    let mut kept = BTreeSet::new();
    let mut pending: Option<(String, u64)> = None;
    while let Some(item) = heap.pop() {
        readers[item.reader_idx].advance()?;
        if let Some((key, count)) = &readers[item.reader_idx].next {
            heap.push(HeapItem {
                key: key.clone(),
                count: *count,
                reader_idx: item.reader_idx,
            });
        }
        match &mut pending {
            Some((pending_key, pending_total)) if *pending_key == item.key => {
                *pending_total += item.count;
            }
            _ => {
                if let Some((finished_key, finished_total)) =
                    pending.replace((item.key, item.count))
                    && finished_total >= min_count
                {
                    kept.insert(finished_key);
                }
            }
        }
    }
    if let Some((finished_key, finished_total)) = pending
        && finished_total >= min_count
    {
        kept.insert(finished_key);
    }
    Ok(kept)
}

/// Every file directly under `dir` whose name matches
/// `{prefix}-partial-*.txt`, sorted (filename order is also flush order,
/// which does not matter to [`k_way_merge`] — each file is independently
/// sorted).
fn partial_paths(dir: &Path, prefix: &str) -> anyhow::Result<Vec<PathBuf>> {
    let want_prefix = format!("{prefix}-partial-");
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("reading {}", dir.display()))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|p| {
            p.file_name().and_then(|n| n.to_str()).is_some_and(|name| {
                name.starts_with(&want_prefix)
                    && p.extension()
                        .is_some_and(|ext| ext.eq_ignore_ascii_case("txt"))
            })
        })
        .collect();
    paths.sort();
    Ok(paths)
}

/// Walks up from the current working directory looking for a `.git`
/// entry — a best-effort repo-root finder needing no new dependency (this
/// workspace has no `git2`/`gix` crate anywhere already).
fn find_repo_root() -> Option<PathBuf> {
    let mut dir = std::env::current_dir().ok()?;
    loop {
        if dir.join(".git").exists() {
            return Some(dir);
        }
        if !dir.pop() {
            return None;
        }
    }
}

/// Lexically resolves `path` to an absolute path (no filesystem access,
/// so this works even when `path` doesn't exist yet): joins it onto the
/// current working directory if relative, then collapses `.`/`..`
/// components textually. Deliberately not `fs::canonicalize` — that
/// requires the path to already exist, which `--out` (about to be
/// created) usually does not.
fn absolute_lexical(path: &Path) -> PathBuf {
    let base = if path.is_absolute() {
        PathBuf::new()
    } else {
        std::env::current_dir().unwrap_or_default()
    };
    let mut out = base;
    for component in path.components() {
        match component {
            std::path::Component::ParentDir => {
                out.pop();
            }
            std::path::Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// `true` if `out`'s lexically-resolved absolute path falls under
/// `repo_root`'s — the pure boundary check [`pack`]'s `--force-repo-out`
/// refusal is built on, factored out so it can be tested against
/// synthetic absolute paths without touching the real working directory
/// (see this module's own tests).
fn out_is_inside(repo_root: &Path, out: &Path) -> bool {
    absolute_lexical(out).starts_with(absolute_lexical(repo_root))
}

/// Runs `general-evidence pack`: k-way merges `--work`'s partials, applies
/// `--min-count`, builds the `.bin`, and writes it plus its `.meta.toml`
/// sidecar.
///
/// # Errors
/// Returns an error if `--work` can't be read, if a partial file is
/// malformed, if filter construction fails (see
/// [`friction_packs::PackError::GeneralEvidenceConstructionFailed`]), if
/// `--out` resolves inside the repo working tree and `--force-repo-out`
/// wasn't passed, or if `--out`/`--meta-out` can't be written.
pub fn pack(args: &PackArgs) -> anyhow::Result<()> {
    if !args.force_repo_out
        && let Some(repo_root) = find_repo_root()
        && out_is_inside(&repo_root, &args.out)
    {
        anyhow::bail!(
            "--out {} resolves inside the repo working tree ({}) — the general-evidence-v1 .bin \
             is a large (~25-45MB), optional runtime artifact and must not be committed; pass \
             --force-repo-out to override",
            args.out.display(),
            repo_root.display(),
        );
    }

    let unigram_paths = partial_paths(&args.work, "unigram")?;
    let bigram_paths = partial_paths(&args.work, "bigram")?;
    let unigrams = k_way_merge(
        &unigram_paths,
        args.unigram_min_count.unwrap_or(args.min_count),
    )?;
    let bigrams = k_way_merge(&bigram_paths, args.min_count)?;

    let built = friction_packs::general_evidence::build_pack_bytes(&unigrams, &bigrams)?;

    if let Some(parent) = args.out.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    std::fs::write(&args.out, &built.bytes)
        .with_context(|| format!("writing {}", args.out.display()))?;
    let bin_sha256_hex = sha256_hex(&built.bytes);

    let meta = render_meta_toml(&MetaInput {
        min_count: args.min_count,
        unigram_min_count: args.unigram_min_count.unwrap_or(args.min_count),
        unigram_key_count: built.unigram_key_count,
        bigram_key_count: built.bigram_key_count,
        source_label: &args.source_label,
        built_at: &args.built_at,
        bin_sha256_hex: &bin_sha256_hex,
    });
    if let Some(parent) = args.meta_out.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    std::fs::write(&args.meta_out, &meta)
        .with_context(|| format!("writing {}", args.meta_out.display()))?;

    println!(
        "general-evidence pack: {} unigram partial(s) + {} bigram partial(s) -> {} unigram \
         key(s) + {} bigram key(s) written to {} ({} bytes, sha256 {bin_sha256_hex}) + {}",
        unigram_paths.len(),
        bigram_paths.len(),
        built.unigram_key_count,
        built.bigram_key_count,
        args.out.display(),
        built.bytes.len(),
        args.meta_out.display(),
    );
    Ok(())
}

struct MetaInput<'a> {
    min_count: u64,
    unigram_min_count: u64,
    unigram_key_count: u64,
    bigram_key_count: u64,
    source_label: &'a str,
    built_at: &'a str,
    bin_sha256_hex: &'a str,
}

/// Renders the pack's `.meta.toml` sidecar: small, human-readable,
/// committed provenance for the large, git-ignored `.bin` it describes —
/// mirrors `human-evidence-en-v1.toml`'s own shape.
fn render_meta_toml(input: &MetaInput<'_>) -> String {
    format!(
        "# general-evidence-v1 pack metadata: committed provenance for the\n\
         # large (~25-45MB), OPTIONAL, NOT-committed .bin runtime artifact\n\
         # this describes (crates/friction-packs/packs/general-evidence-*.bin\n\
         # is git-ignored — see the repo root .gitignore). Generated by\n\
         # `corpus-tool general-evidence pack` — do not hand-edit.\n\
         #\n\
         # friction_packs::general_evidence::GeneralEvidencePack does NOT\n\
         # read this file at runtime (unlike jargon-attest-v1's .toml\n\
         # sidecar): the .bin's own header is fully self-describing. This\n\
         # file exists for humans auditing what a staged .bin was built\n\
         # from.\n\n\
         [pack]\n\
         version = \"general-evidence-v1\"\n\
         min_count = {min_count}\n\
         unigram_min_count = {unigram_min_count}\n\
         unigram_key_count = {unigram_key_count}\n\
         bigram_key_count = {bigram_key_count}\n\
         source_label = {source_label:?}\n\
         built_at = {built_at:?}\n\
         bin_sha256 = \"{bin_sha256_hex}\"\n",
        min_count = input.min_count,
        unigram_min_count = input.unigram_min_count,
        unigram_key_count = input.unigram_key_count,
        bigram_key_count = input.bigram_key_count,
        source_label = input.source_label,
        built_at = input.built_at,
        bin_sha256_hex = input.bin_sha256_hex,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- markup stripping ---

    #[test]
    fn strip_templates_removes_a_single_level_span() {
        assert_eq!(strip_templates("a {{cite|x=1}} b"), "a  b");
        assert_eq!(strip_templates("no templates here"), "no templates here");
    }

    #[test]
    fn strip_links_keeps_target_for_plain_links_and_display_for_piped() {
        assert_eq!(
            strip_links("see [[Rust (language)]] docs"),
            "see Rust (language) docs"
        );
        assert_eq!(
            strip_links("see [[Rust (language)|Rust]] docs"),
            "see Rust docs"
        );
        assert_eq!(
            strip_links("[[File:x.jpg|thumb|a caption]]"),
            "thumb|a caption",
            "only the first pipe splits target from display; the rest is display text"
        );
    }

    #[test]
    fn strip_tags_removes_inline_markup() {
        assert_eq!(strip_tags("a <ref name=\"x\">note</ref> b"), "a note b");
    }

    #[test]
    fn strip_entities_blanks_named_entities() {
        assert_eq!(strip_entities("Rust&amp;WASM"), "Rust WASM");
    }

    #[test]
    fn strip_headings_removes_heading_markers_only() {
        assert_eq!(strip_headings("==History==\ntext"), "\ntext");
        assert_eq!(
            strip_headings("a = b"),
            "a = b",
            "a bare '=' is not a heading"
        );
    }

    #[test]
    fn strip_markup_end_to_end_on_a_synthetic_wiki_line() {
        let raw = "{{cite}}A [[data fabric|fabric]] pattern <ref>x</ref> uses &amp; logic.";
        let cleaned = strip_markup(raw);
        assert_eq!(cleaned, "A fabric pattern x uses   logic.");
    }

    // --- tokenize_line ---

    #[test]
    fn tokenize_line_splits_on_non_alpha_and_lowercases() {
        let mut tokens = Vec::new();
        tokenize_line("Data-Fabric 2 patterns.", &mut tokens);
        assert_eq!(tokens, vec!["data", "fabric", "patterns"]);
    }

    #[test]
    fn tokenize_line_reuses_its_buffer() {
        let mut tokens = Vec::new();
        tokenize_line("one two", &mut tokens);
        tokenize_line("three", &mut tokens);
        assert_eq!(tokens, vec!["three"]);
    }

    // --- TextExtractor ---

    #[test]
    fn text_extractor_captures_content_spanning_multiple_lines() {
        let mut extractor = TextExtractor::default();
        let mut chunks = Vec::new();
        extractor.feed_line("<page><text xml:space=\"preserve\">first line", |c| {
            chunks.push(c.to_string());
        });
        extractor.feed_line("second line</text></page>", |c| chunks.push(c.to_string()));
        assert_eq!(chunks, vec!["first line", "second line"]);
    }

    #[test]
    fn text_extractor_handles_a_single_line_open_and_close() {
        let mut extractor = TextExtractor::default();
        let mut chunks = Vec::new();
        extractor.feed_line("<text>short article</text>", |c| chunks.push(c.to_string()));
        assert_eq!(chunks, vec!["short article"]);
    }

    #[test]
    fn text_extractor_skips_a_self_closing_text_tag() {
        let mut extractor = TextExtractor::default();
        let mut chunks = Vec::new();
        extractor.feed_line("<text bytes=\"0\" />", |c| chunks.push(c.to_string()));
        assert!(chunks.is_empty());
    }

    #[test]
    fn text_extractor_ignores_content_outside_text_elements() {
        let mut extractor = TextExtractor::default();
        let mut chunks = Vec::new();
        extractor.feed_line("<title>Some Page</title>", |c| chunks.push(c.to_string()));
        assert!(chunks.is_empty());
    }

    // --- k_way_merge / partial_paths / flush_partial ---

    fn write_lines(path: &Path, lines: &[(&str, u64)]) {
        use std::fmt::Write as _;
        let mut out = String::new();
        for (key, count) in lines {
            writeln!(out, "{key}\t{count}").expect("write to String is infallible");
        }
        std::fs::write(path, out).expect("write partial fixture");
    }

    #[test]
    fn k_way_merge_sums_across_partials_and_applies_min_count() {
        let dir = tempfile::tempdir().expect("tempdir");
        let a = dir.path().join("a.txt");
        let b = dir.path().join("b.txt");
        write_lines(&a, &[("alpha", 3), ("gamma", 1)]);
        write_lines(&b, &[("alpha", 2), ("beta", 5)]);
        let kept = k_way_merge(&[a, b], 5).expect("merge succeeds");
        assert_eq!(
            kept,
            BTreeSet::from(["alpha".to_string(), "beta".to_string()])
        );
    }

    #[test]
    fn k_way_merge_of_zero_partials_is_empty() {
        let kept = k_way_merge(&[], 1).expect("merge of nothing succeeds");
        assert!(kept.is_empty());
    }

    #[test]
    fn partial_paths_filters_by_prefix_and_extension() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("unigram-partial-00001.txt"), "").unwrap();
        std::fs::write(dir.path().join("bigram-partial-00001.txt"), "").unwrap();
        std::fs::write(dir.path().join("unigram-partial-00002.txt"), "").unwrap();
        std::fs::write(dir.path().join("notes.md"), "").unwrap();
        let paths = partial_paths(dir.path(), "unigram").expect("reads dir");
        assert_eq!(paths.len(), 2);
        assert!(paths.iter().all(|p| {
            p.file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .starts_with("unigram-partial-")
        }));
    }

    #[test]
    fn flush_partial_writes_sorted_tab_separated_lines() {
        let dir = tempfile::tempdir().expect("tempdir");
        let counts = std::collections::BTreeMap::from([
            ("zeta".to_string(), 2u64),
            ("alpha".to_string(), 9u64),
        ]);
        flush_partial(dir.path(), "unigram", 1, &counts).expect("flush succeeds");
        let text = std::fs::read_to_string(dir.path().join("unigram-partial-00001.txt")).unwrap();
        assert_eq!(text, "alpha\t9\nzeta\t2\n");
    }

    // --- repo-root refusal ---

    #[test]
    fn out_is_inside_detects_a_path_under_the_repo_root() {
        let repo_root = Path::new("/home/dev/friction");
        assert!(out_is_inside(
            repo_root,
            Path::new("/home/dev/friction/crates/friction-packs/packs/general-evidence-en-v1.bin")
        ));
        assert!(!out_is_inside(
            repo_root,
            Path::new("/home/dev/.cache/friction/general-evidence-en-v1.bin")
        ));
    }

    #[test]
    fn out_is_inside_normalizes_dot_dot_components() {
        let repo_root = Path::new("/home/dev/friction");
        assert!(!out_is_inside(
            repo_root,
            Path::new("/home/dev/friction/crates/../../.cache/friction/general-evidence-en-v1.bin")
        ));
    }

    // --- render_meta_toml ---

    #[test]
    fn render_meta_toml_contains_every_field_and_parses_as_toml() {
        let text = render_meta_toml(&MetaInput {
            min_count: 5,
            unigram_min_count: 5,
            unigram_key_count: 42,
            bigram_key_count: 7,
            source_label: "enwiki-20260701-pages-articles",
            built_at: "2026-07-01T00:00:00Z",
            bin_sha256_hex: "deadbeef",
        });
        assert!(text.contains("version = \"general-evidence-v1\""));
        assert!(text.contains("min_count = 5"));
        assert!(text.contains("unigram_key_count = 42"));
        assert!(text.contains("bigram_key_count = 7"));
        assert!(text.contains("enwiki-20260701-pages-articles"));
        assert!(text.contains("bin_sha256 = \"deadbeef\""));
        assert!(toml::from_str::<toml::Value>(&text).is_ok(), "{text}");
    }

    // --- mine/pack smoke test: a tiny in-repo fixture, no network ---

    /// A tiny synthetic `MediaWiki`-shaped dump, exercising templates,
    /// links, tags, entities, and a heading in one pass — small enough to
    /// hand-verify, real enough to catch a wiring mistake between `mine`
    /// and `pack`.
    const FIXTURE_DUMP: &str = "\
<mediawiki>
<page>
<title>Example</title>
<revision>
<text xml:space=\"preserve\">
==History==
{{infobox|foo=bar}}
The [[data fabric|fabric]] pattern uses <ref>citation</ref> logic &amp; design.
The fabric pattern recurs across every fabric pattern discussion.
</text>
</revision>
</page>
</mediawiki>
";

    #[test]
    fn mine_then_pack_round_trips_a_tiny_fixture_into_a_loadable_pack() {
        let dir = tempfile::tempdir().expect("tempdir");
        let work = dir.path().join("work");

        // Feed the fixture through the exact TextExtractor + strip_markup
        // + tokenize_line pipeline `mine` uses, bypassing stdin (this is
        // an in-process unit test, not a CLI integration test).
        std::fs::create_dir_all(&work).expect("create work dir");
        let mut unigram_counts: std::collections::BTreeMap<String, u64> =
            std::collections::BTreeMap::new();
        let mut bigram_counts: std::collections::BTreeMap<String, u64> =
            std::collections::BTreeMap::new();
        let mut extractor = TextExtractor::default();
        let mut tokens = Vec::new();
        for line in FIXTURE_DUMP.lines() {
            let mut chunks = Vec::new();
            extractor.feed_line(line, |c| chunks.push(c.to_string()));
            for chunk in &chunks {
                let cleaned = strip_markup(chunk);
                tokenize_line(&cleaned, &mut tokens);
                for word in tokens.iter().filter(|w| w.len() >= 3) {
                    *unigram_counts.entry(word.clone()).or_insert(0) += 1;
                }
                for pair in tokens.windows(2) {
                    let [w1, w2] = pair else { unreachable!() };
                    if w1.len() >= 3 && w2.len() >= 3 {
                        *bigram_counts.entry(format!("{w1} {w2}")).or_insert(0) += 1;
                    }
                }
            }
        }
        assert!(unigram_counts.get("fabric").copied().unwrap_or(0) >= 3);
        flush_partial(&work, "unigram", 1, &unigram_counts).expect("flush unigrams");
        flush_partial(&work, "bigram", 1, &bigram_counts).expect("flush bigrams");

        let out = dir.path().join("general-evidence-en-v1.bin");
        let meta_out = dir.path().join("general-evidence-en-v1.meta.toml");
        pack(&PackArgs {
            work,
            min_count: 3,
            unigram_min_count: None,
            out: out.clone(),
            meta_out: meta_out.clone(),
            source_label: "fixture".to_string(),
            built_at: "2026-01-01T00:00:00Z".to_string(),
            force_repo_out: true,
        })
        .expect("pack succeeds");

        let bin = std::fs::read(&out).expect("read built .bin");
        let loaded_pack = friction_packs::general_evidence::GeneralEvidencePack::load(Box::leak(
            bin.into_boxed_slice(),
        ))
        .expect("built pack loads");
        assert!(
            loaded_pack.has_word("fabric"),
            "seen >= 3 times, above --min-count 3"
        );
        assert!(
            !loaded_pack.has_word("history"),
            "seen once (in a stripped heading), below --min-count 3"
        );
        assert!(loaded_pack.has_bigram("fabric", "pattern"));

        let meta = std::fs::read_to_string(&meta_out).expect("read meta.toml");
        assert!(meta.contains("source_label = \"fixture\""));
    }
}
