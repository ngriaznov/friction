//! `corpus-tool clean` — deterministic incoming-doc cleaning
//! pipeline (normalize to UTF-8 + LF, strip README boilerplate, keep
//! markdown structure, drop sub-300-word docs).

use std::path::PathBuf;

use clap::Args as ClapArgs;

use crate::hashing::word_count;

const MIN_WORDS: usize = 300;

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

/// Decodes to UTF-8 (lossily, if the source isn't valid UTF-8) and
/// normalizes all line endings to LF.
fn normalize(raw: &[u8]) -> String {
    let text = String::from_utf8_lossy(raw);
    text.replace("\r\n", "\n").replace('\r', "\n")
}

/// Strips badge-image-wall and standalone HTML nav/footer/layout tag
/// lines, then collapses the resulting blank-line runs and trims leading
/// and trailing blank lines. Markdown prose lines are left untouched.
fn strip_boilerplate(text: &str) -> String {
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
}
