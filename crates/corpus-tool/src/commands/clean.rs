//! `corpus-tool clean` — deterministic incoming-doc cleaning
//! pipeline (normalize to UTF-8 + LF, strip README boilerplate, keep
//! markdown structure, drop sub-300-word docs).

use std::path::PathBuf;

use clap::Args as ClapArgs;

use crate::hashing::word_count;

/// Docs under this many words (after cleaning) are dropped. Shared with
/// `ingest`, which applies the identical cleaning pipeline to incoming
/// human-corpus fragments.
pub(crate) const MIN_WORDS: usize = 300;

/// Arguments for `corpus-tool clean`.
#[derive(Debug, ClapArgs)]
pub struct Args {
    /// Directory of raw incoming documents, recursively scanned for
    /// `.md` files.
    #[arg(long)]
    pub incoming: PathBuf,
    /// Directory to write cleaned documents into, mirroring the incoming
    /// directory's relative layout.
    #[arg(long)]
    pub out: PathBuf,
}

/// Runs `clean`.
///
/// Reads every `.md` file under `--incoming` in sorted (deterministic)
/// order, normalizes it to UTF-8 + LF, strips common README boilerplate
/// (badge-image walls, standalone HTML nav/footer/layout tag lines) while
/// leaving markdown structure alone, and writes survivors under `--out`.
/// Docs under 300 words after cleaning are dropped (not written) and
/// reported.
///
/// # Errors
///
/// Returns an error on any I/O failure reading `--incoming` or writing
/// `--out`.
pub fn run(args: &Args) -> anyhow::Result<()> {
    let mut inputs: Vec<PathBuf> = walkdir::WalkDir::new(&args.incoming)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().is_file())
        .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "md"))
        .map(walkdir::DirEntry::into_path)
        .collect();
    inputs.sort();

    let mut kept = 0usize;
    let mut dropped = Vec::new();

    for input_path in &inputs {
        let relative = input_path
            .strip_prefix(&args.incoming)
            .expect("walkdir entries are always under the incoming root");
        let raw = std::fs::read(input_path)?;
        let text = normalize(&raw);
        let cleaned = strip_boilerplate(&text);
        let words = word_count(&cleaned);

        if words < MIN_WORDS {
            dropped.push((relative.to_path_buf(), words));
            continue;
        }

        let out_path = args.out.join(relative);
        if let Some(parent) = out_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&out_path, cleaned)?;
        kept += 1;
    }

    println!(
        "clean: kept {kept}, dropped {} (under {MIN_WORDS} words)",
        dropped.len()
    );
    for (path, words) in &dropped {
        println!("  dropped {} ({words} words)", path.display());
    }

    Ok(())
}

/// Decodes to UTF-8 (lossily, if the source isn't valid UTF-8), normalizes
/// all line endings to LF, and decodes raw HTML entities left over from
/// un-decoded source markup (see [`decode_entities`]).
pub(crate) fn normalize(raw: &[u8]) -> String {
    let text = String::from_utf8_lossy(raw);
    let text = text.replace("\r\n", "\n").replace('\r', "\n");
    decode_entities(&text)
}

/// Named HTML entities `decode_entities` recognizes, mapped to their
/// literal replacement text. Deliberately a small, fixed set — the
/// entities actually observed in this corpus's StackExchange-sourced
/// source markup — rather than the full HTML5 named-entity table.
const NAMED_ENTITIES: &[(&str, &str)] = &[
    ("amp", "&"),
    ("lt", "<"),
    ("gt", ">"),
    ("quot", "\""),
    ("apos", "'"),
    ("nbsp", "\u{a0}"),
    ("mdash", "\u{2014}"),
    ("ndash", "\u{2013}"),
    ("hellip", "\u{2026}"),
    ("rsquo", "\u{2019}"),
    ("lsquo", "\u{2018}"),
    ("rdquo", "\u{201d}"),
    ("ldquo", "\u{201c}"),
];

/// Upper bound on the number of full decode passes [`decode_entities`]
/// performs.
///
/// A single pass decodes every named entity in [`NAMED_ENTITIES`] and
/// every decimal (`&#39;`) or hex (`&#x27;`) numeric character reference
/// it finds — but a *double*-encoded source (`&amp;#39;`, i.e. the literal
/// `&` of `&#39;` was itself entity-encoded before this text was captured)
/// only exposes its inner `&#39;` after the outer `&amp;` layer is peeled
/// off, one pass at a time: pass 1 turns `&amp;#39;` into `&#39;`, pass 2
/// turns that into `'`. Looping to a fixpoint (stopping as soon as a pass
/// changes nothing) handles that and any deeper nesting without over- or
/// under-decoding; the bound just caps the work on adversarial input,
/// since real corpus text is never encoded more than twice.
const MAX_DECODE_PASSES: usize = 4;

/// Decodes HTML entities — [`NAMED_ENTITIES`] plus decimal/hex numeric
/// character references — to their literal characters, repeating to a
/// fixpoint (bounded by [`MAX_DECODE_PASSES`]) so a double-encoded entity
/// (`&amp;#39;`) fully decodes rather than stopping one layer short. An
/// unrecognized or malformed `&...;` sequence is left as-is.
pub(crate) fn decode_entities(text: &str) -> String {
    let mut current = text.to_string();
    for _ in 0..MAX_DECODE_PASSES {
        let next = decode_entities_once(&current);
        if next == current {
            break;
        }
        current = next;
    }
    current
}

/// One left-to-right decode pass over `text`: every recognized `&...;`
/// entity reference is replaced by its literal character(s); everything
/// else (including any `&` that isn't the start of a recognized entity)
/// is copied through unchanged.
fn decode_entities_once(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(amp_offset) = rest.find('&') {
        out.push_str(&rest[..amp_offset]);
        let tail = &rest[amp_offset..];
        if let Some((replacement, consumed)) = decode_one_entity(tail) {
            out.push_str(&replacement);
            rest = &tail[consumed..];
        } else {
            out.push('&');
            rest = &tail['&'.len_utf8()..];
        }
    }
    out.push_str(rest);
    out
}

/// Decodes at most one entity reference starting at the beginning of `s`
/// (which must start with `&`). Returns the decoded replacement text and
/// the number of bytes of `s` it consumes (from the leading `&` through
/// the trailing `;`), or `None` if `s` doesn't start with a recognized
/// entity — either the whole `&name;`/`&#NNN;`/`&#xHHH;` form, or the
/// name/codepoint isn't recognized/valid.
fn decode_one_entity(s: &str) -> Option<(String, usize)> {
    debug_assert!(s.starts_with('&'));
    let body = &s[1..];
    // A real entity is short; a `;` found far away is almost certainly an
    // unrelated stray `&`, not a malformed entity worth scanning for.
    let semicolon = body.char_indices().take(32).find(|&(_, c)| c == ';')?;
    let (semicolon_byte, _) = semicolon;
    let entity = &body[..semicolon_byte];
    let consumed = 1 + semicolon_byte + 1;

    if let Some(hex) = entity
        .strip_prefix("#x")
        .or_else(|| entity.strip_prefix("#X"))
    {
        let codepoint = u32::from_str_radix(hex, 16).ok()?;
        return char::from_u32(codepoint).map(|c| (c.to_string(), consumed));
    }
    if let Some(decimal) = entity.strip_prefix('#') {
        let codepoint: u32 = decimal.parse().ok()?;
        return char::from_u32(codepoint).map(|c| (c.to_string(), consumed));
    }
    NAMED_ENTITIES
        .iter()
        .find(|(name, _)| *name == entity)
        .map(|(_, replacement)| ((*replacement).to_string(), consumed))
}

/// Strips badge-image-wall and standalone HTML nav/footer/layout tag
/// lines, then collapses the resulting blank-line runs and trims leading
/// and trailing blank lines. Markdown prose lines are left untouched.
pub(crate) fn strip_boilerplate(text: &str) -> String {
    let mut out_lines: Vec<&str> = Vec::new();
    for line in text.lines() {
        if is_badge_line(line) || is_html_wrapper_line(line) {
            continue;
        }
        out_lines.push(line.trim_end());
    }

    let mut collapsed: Vec<&str> = Vec::with_capacity(out_lines.len());
    for line in out_lines {
        let blank = line.trim().is_empty();
        if blank
            && collapsed
                .last()
                .is_some_and(|prev: &&str| prev.trim().is_empty())
        {
            continue;
        }
        collapsed.push(line);
    }

    let start = collapsed
        .iter()
        .position(|l| !l.trim().is_empty())
        .unwrap_or(collapsed.len());
    let end = collapsed
        .iter()
        .rposition(|l| !l.trim().is_empty())
        .map_or(start, |i| i + 1);

    let mut out = collapsed[start..end].join("\n");
    out.push('\n');
    out
}

/// A bare markdown image, a linked ("shield") badge image, or a raw HTML
/// `<img>` tag, alone on its line — the atoms of a typical README badge
/// wall.
fn is_badge_line(line: &str) -> bool {
    let line = line.trim();
    is_markdown_image_only(line) || is_linked_markdown_image_only(line) || is_html_img_only(line)
}

fn is_markdown_image_only(line: &str) -> bool {
    line.starts_with("![") && line.ends_with(')')
}

fn is_linked_markdown_image_only(line: &str) -> bool {
    line.starts_with("[![") && line.ends_with(')')
}

fn is_html_img_only(line: &str) -> bool {
    line.starts_with("<img ") && line.ends_with('>')
}

/// HTML tag names treated as boilerplate wrapper elements when a line
/// consists solely of one such tag (open or close).
const WRAPPER_TAGS: [&str; 10] = [
    "div", "nav", "footer", "header", "p", "hr", "br", "a", "table", "center",
];

/// A standalone HTML wrapper tag line common in README boilerplate
/// (centering divs, nav/footer blocks, layout tables, raw `<hr>`/`<br>`).
fn is_html_wrapper_line(line: &str) -> bool {
    let trimmed = line.trim();
    if trimmed.len() < 2 || !trimmed.starts_with('<') || !trimmed.ends_with('>') {
        return false;
    }

    let inner = &trimmed[1..trimmed.len() - 1];
    let inner = inner.strip_prefix('/').unwrap_or(inner);
    let tag_name = inner
        .split_whitespace()
        .next()
        .unwrap_or(inner)
        .to_ascii_lowercase();
    WRAPPER_TAGS.contains(&tag_name.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// CRLF and bare-CR line endings normalize to LF.
    #[test]
    fn normalize_converts_crlf_and_cr_to_lf() {
        let text = normalize(b"line one\r\nline two\rline three\n");
        assert_eq!(text, "line one\nline two\nline three\n");
    }

    /// Invalid UTF-8 is lossily decoded rather than rejected.
    #[test]
    fn normalize_lossily_decodes_invalid_utf8() {
        let text = normalize(&[b'h', b'i', 0xFF, b'!']);
        assert!(text.starts_with("hi"));
        assert!(text.ends_with('!'));
    }

    /// A badge wall of bare and linked markdown images is
    /// stripped, keeping the prose that follows.
    #[test]
    fn strip_boilerplate_removes_badge_wall() {
        let input = "![Build](build.svg)\n[![Coverage](cov.svg)](https://example.com)\n\n# Title\n\nReal prose here.\n";
        let cleaned = strip_boilerplate(input);
        assert!(!cleaned.contains("build.svg"));
        assert!(!cleaned.contains("cov.svg"));
        assert!(cleaned.contains("# Title"));
        assert!(cleaned.contains("Real prose here."));
    }

    /// A centering `<div>`/`<p align>` wrapper block is stripped.
    #[test]
    fn strip_boilerplate_removes_html_wrapper_lines() {
        let input =
            "<div align=\"center\">\n<img src=\"logo.png\">\n</div>\n\n# Title\n\nBody text.\n";
        let cleaned = strip_boilerplate(input);
        assert!(!cleaned.contains("<div"));
        assert!(!cleaned.contains("</div>"));
        assert!(cleaned.contains("# Title"));
        assert!(cleaned.contains("Body text."));
    }

    /// Ordinary markdown structure (headings, lists, code
    /// fences) survives untouched.
    #[test]
    fn strip_boilerplate_keeps_markdown_structure() {
        let input = "# Title\n\n- item one\n- item two\n\n```rust\nfn main() {}\n```\n";
        let cleaned = strip_boilerplate(input);
        assert!(cleaned.contains("# Title"));
        assert!(cleaned.contains("- item one"));
        assert!(cleaned.contains("```rust"));
        assert!(cleaned.contains("fn main() {}"));
    }

    /// Cleaning the same input twice produces byte-identical output.
    #[test]
    fn strip_boilerplate_is_deterministic() {
        let input = "![Badge](b.svg)\n\n# Title\n\nProse.\n";
        assert_eq!(strip_boilerplate(input), strip_boilerplate(input));
    }

    /// Each named entity in [`NAMED_ENTITIES`] decodes to its literal
    /// character.
    #[test]
    fn decode_entities_decodes_named_entities() {
        assert_eq!(
            decode_entities("Tom &amp; Jerry: 1 &lt; 2 &gt; 0, &quot;quoted&quot;, doesn&apos;t"),
            "Tom & Jerry: 1 < 2 > 0, \"quoted\", doesn't"
        );
        assert_eq!(decode_entities("a&nbsp;b"), "a\u{a0}b");
        assert_eq!(decode_entities("wait&hellip;"), "wait\u{2026}");
        assert_eq!(decode_entities("em&mdash;dash"), "em\u{2014}dash");
        assert_eq!(decode_entities("en&ndash;dash"), "en\u{2013}dash");
        assert_eq!(
            decode_entities("&lsquo;quote&rsquo; &ldquo;double&rdquo;"),
            "\u{2018}quote\u{2019} \u{201c}double\u{201d}"
        );
    }

    /// Decimal numeric character references decode via their Unicode
    /// codepoint — the exact `&#39;` case the diagnosis calls out, plus a
    /// multi-digit codepoint.
    #[test]
    fn decode_entities_decodes_decimal_numeric_refs() {
        assert_eq!(decode_entities("doesn&#39;t"), "doesn't");
        assert_eq!(decode_entities("&#176;"), "\u{b0}");
    }

    /// Hex numeric character references (both `&#x` and `&#X`) decode via
    /// their Unicode codepoint.
    #[test]
    fn decode_entities_decodes_hex_numeric_refs() {
        assert_eq!(decode_entities("&#x27;"), "'");
        assert_eq!(decode_entities("&#X3BB;"), "\u{3bb}");
    }

    /// A double-encoded entity — the literal `&` of `&#39;` was itself
    /// entity-encoded to `&amp;` before this text was captured — fully
    /// decodes to the apostrophe, not just one layer to `&#39;`.
    #[test]
    fn decode_entities_fully_decodes_double_encoded_entity() {
        assert_eq!(decode_entities("doesn&amp;#39;t"), "doesn't");
        assert_eq!(decode_entities("&amp;amp;"), "&");
    }

    /// An unrecognized entity name, and a bare `&` not part of any
    /// entity, are both left untouched rather than dropped or mangled.
    #[test]
    fn decode_entities_leaves_unrecognized_and_bare_ampersands_untouched() {
        assert_eq!(decode_entities("&foo; &bar;"), "&foo; &bar;");
        assert_eq!(decode_entities("Q&A department"), "Q&A department");
        assert_eq!(decode_entities("R&D and A&B"), "R&D and A&B");
    }

    /// A malformed/out-of-range numeric reference (invalid codepoint) is
    /// left untouched rather than panicking or producing a replacement
    /// character.
    #[test]
    fn decode_entities_leaves_invalid_numeric_ref_untouched() {
        assert_eq!(decode_entities("&#xD800;"), "&#xD800;");
        assert_eq!(decode_entities("&#99999999;"), "&#99999999;");
    }

    /// Decoding the same input twice produces byte-identical output
    /// (idempotence is what lets the corpus maintenance pass be safely
    /// rerun).
    #[test]
    fn decode_entities_is_deterministic() {
        let input = "doesn&amp;#39;t &amp; &lt;tags&gt; &#xE9;clair";
        assert_eq!(decode_entities(input), decode_entities(input));
    }

    /// Already-decoded text (no entities at all) round-trips unchanged —
    /// running the maintenance pass on an already-clean doc is a no-op.
    #[test]
    fn decode_entities_is_idempotent_on_clean_text() {
        let input = "Plain prose with no entities, & this bare ampersand.";
        let once = decode_entities(input);
        assert_eq!(once, input);
        assert_eq!(decode_entities(&once), once);
    }

    /// `normalize` decodes entities as part of the same pass that
    /// normalizes line endings, matching what both `clean` and `ingest`
    /// apply to every doc.
    #[test]
    fn normalize_decodes_entities() {
        let text = normalize(b"line one\r\ndoesn&#39;t &amp; won&apos;t\n");
        assert_eq!(text, "line one\ndoesn't & won't\n");
    }
}
