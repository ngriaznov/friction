//! Static-HTML prose extraction.
//!
//! Turns an HTML page into the same [`friction_core::Document`] shape the
//! markdown extractor produces: blocks with exact byte ranges into the
//! original source, plus the prose text runs found inside them. The
//! engine downstream is format-agnostic — it edits byte ranges and never
//! sees markup — so this module is the *whole* of HTML support: nothing
//! else in the workspace knows the input was HTML.
//!
//! # What counts as prose
//!
//! A maximal run of text between markup boundaries, in an element context
//! that is not excluded. Every tag, comment, doctype, CDATA section, and
//! *character reference* is a boundary that ends the current run — the
//! same "an interruption splits a block into separate prose runs" rule
//! the markdown extractor applies to inline code and link URLs. Splitting
//! at character references means a run's bytes are always literal text:
//! no entity ever needs decoding on the way in or re-encoding on the way
//! out, and an edit can never produce a range that slices through
//! `&amp;`. The cost is that a sentence interrupted by markup is analyzed
//! as fragments; the safety is that markup is untouchable by
//! construction.
//!
//! # Element handling
//!
//! - **Excluded contexts** ([`is_excluded_element`]): `head`, scripts,
//!   styles, code-ish elements (`pre`, `code`, `kbd`, `samp`, `var`,
//!   `tt`), embedded non-prose vocabularies (`svg`, `math`), and
//!   non-content machinery (`template`, `noscript`, `textarea`,
//!   `select`, `option`, `iframe`, `object`, `canvas`). Text inside any
//!   of them is never prose.
//! - **Raw-text elements** ([`is_raw_text_element`]): `script`, `style`,
//!   `textarea`, `title` content is scanned as opaque bytes to the
//!   matching close tag, per the HTML parsing spec's raw-text states —
//!   a `<` inside a script is not markup.
//! - **Void elements** ([`is_void_element`]): never pushed onto the open
//!   stack, so an unclosed `<br>` can't poison the context below it.
//! - **Block kinds**: a run inside `h1`-`h6` becomes a
//!   [`BlockKind::Heading`], inside `td`/`th` a [`BlockKind::TableCell`],
//!   inside `li` a [`BlockKind::ListItem`], inside `blockquote` a
//!   [`BlockKind::BlockQuote`], and anything else a
//!   [`BlockKind::Paragraph`] — chosen by innermost context, so the
//!   existing scope rules (`friction-match`'s `prose_scope`) give HTML
//!   the same semantics markdown already has: headings and table cells
//!   are out of edit scope, paragraph/list/quote text is in.
//!
//! Each prose run gets its own [`Block`] with an identical range — HTML
//! has no useful analogue of markdown's nested block tree for the
//! editing passes, and a one-to-one block-per-run structure satisfies
//! [`friction_core::Document`]'s containment validation exactly.
//!
//! # Malformed input
//!
//! This is a tokenizer with the standard mis-nesting recovery (a close
//! tag pops to its matching open element if one is on the stack, and is
//! otherwise ignored), not a spec-complete tree builder. The failure
//! direction is the safe one everywhere: anything not confidently prose
//! is left out of every prose run, and text that is never extracted is
//! text that can never be edited.

use friction_core::{Block, BlockKind, ProseUnit};

/// One open element on the context stack.
struct OpenElement {
    /// Lowercased tag name.
    name: String,
    /// `true` if this element (itself, not an ancestor) is one of
    /// [`is_excluded_element`]'s set — recorded at push time so pops can
    /// maintain the exclusion counter without re-deriving it.
    excluded: bool,
}

/// Ends the current text run (if any) at `end`, emitting a block/prose
/// pair for its whitespace-trimmed extent.
fn flush(
    source: &str,
    run_start: &mut Option<usize>,
    end: usize,
    stack: &[OpenElement],
    blocks: &mut Vec<Block>,
    prose: &mut Vec<ProseUnit>,
) {
    let Some(start) = run_start.take() else {
        return;
    };
    debug_assert!(start <= end);
    let raw = &source[start..end];
    let trimmed_front = raw.len() - raw.trim_start().len();
    let trimmed = raw.trim_end().trim_start();
    if trimmed.is_empty() {
        return;
    }
    let range = start + trimmed_front..start + trimmed_front + trimmed.len();
    let kind = kind_for_context(stack);
    let block_index = blocks.len();
    blocks.push(Block::new(kind, range.clone()));
    prose.push(ProseUnit::new(block_index, range, Vec::new()));
}

/// Extracts `(blocks, prose)` from HTML source — the read side
/// [`crate::parse_with`] hands to [`friction_core::Document::new`].
pub fn extract_html(source: &str) -> (Vec<Block>, Vec<ProseUnit>) {
    let bytes = source.as_bytes();
    let mut blocks: Vec<Block> = Vec::new();
    let mut prose: Vec<ProseUnit> = Vec::new();
    let mut stack: Vec<OpenElement> = Vec::new();
    let mut excluded_depth: usize = 0;

    let mut pos = 0;
    let mut run_start: Option<usize> = None;

    while pos < bytes.len() {
        match bytes[pos] {
            b'<' => {
                flush(source, &mut run_start, pos, &stack, &mut blocks, &mut prose);
                pos = consume_markup(
                    source,
                    pos,
                    &mut stack,
                    &mut excluded_depth,
                    &mut blocks,
                    &mut prose,
                );
            }
            b'&' => {
                if let Some(after) = character_reference_end(bytes, pos) {
                    // A recognized-shape character reference: a run
                    // boundary, exactly like inline code in markdown.
                    flush(source, &mut run_start, pos, &stack, &mut blocks, &mut prose);
                    pos = after;
                } else {
                    // A bare ampersand ("R&D") is ordinary prose text.
                    if excluded_depth == 0 && run_start.is_none() {
                        run_start = Some(pos);
                    }
                    pos += 1;
                }
            }
            _ => {
                if excluded_depth == 0 && run_start.is_none() {
                    run_start = Some(pos);
                }
                pos += 1;
            }
        }
    }
    flush(
        source,
        &mut run_start,
        bytes.len(),
        &stack,
        &mut blocks,
        &mut prose,
    );

    (blocks, prose)
}

/// The [`BlockKind`] for a text run, from the innermost relevant open
/// element — see the module docs for the mapping and why it reuses
/// markdown's kinds.
fn kind_for_context(stack: &[OpenElement]) -> BlockKind {
    for element in stack.iter().rev() {
        match element.name.as_str() {
            "h1" => return BlockKind::Heading { level: 1 },
            "h2" => return BlockKind::Heading { level: 2 },
            "h3" => return BlockKind::Heading { level: 3 },
            "h4" => return BlockKind::Heading { level: 4 },
            "h5" => return BlockKind::Heading { level: 5 },
            "h6" => return BlockKind::Heading { level: 6 },
            "td" => return BlockKind::TableCell { header: false },
            "th" => return BlockKind::TableCell { header: true },
            "li" => return BlockKind::ListItem,
            "blockquote" => return BlockKind::BlockQuote,
            _ => {}
        }
    }
    BlockKind::Paragraph
}

/// Consumes one markup construct starting at the `<` at `at`, updating
/// the element stack and exclusion depth, and returns the position just
/// past it. A `<` that opens no recognizable construct is consumed as a
/// single byte (it renders as text, but a lone stray `<` is not worth a
/// prose run of its own).
fn consume_markup(
    source: &str,
    at: usize,
    stack: &mut Vec<OpenElement>,
    excluded_depth: &mut usize,
    blocks: &mut Vec<Block>,
    prose: &mut Vec<ProseUnit>,
) -> usize {
    let bytes = source.as_bytes();
    let rest = &bytes[at..];

    if rest.starts_with(b"<!--") {
        return find_from(bytes, at + 4, b"-->").map_or(bytes.len(), |i| i + 3);
    }
    if rest.starts_with(b"<![CDATA[") {
        return find_from(bytes, at + 9, b"]]>").map_or(bytes.len(), |i| i + 3);
    }
    if rest.starts_with(b"<!") || rest.starts_with(b"<?") {
        // Doctype or processing instruction: skip to `>`.
        return find_from(bytes, at + 2, b">").map_or(bytes.len(), |i| i + 1);
    }
    if rest.starts_with(b"</") {
        let (name, after_name) = tag_name(source, at + 2);
        let end = find_from(bytes, after_name, b">").map_or(bytes.len(), |i| i + 1);
        if !name.is_empty() {
            close_element(&name, stack, excluded_depth);
        }
        return end;
    }
    if rest.len() > 1 && rest[1].is_ascii_alphabetic() {
        let (name, after_name) = tag_name(source, at + 1);
        let (tag_end, self_closed) = consume_attributes(bytes, after_name);
        if !self_closed && !is_void_element(&name) {
            let excluded = is_excluded_element(&name);
            if excluded {
                *excluded_depth += 1;
            }
            if is_raw_text_element(&name) {
                // Raw-text content: opaque bytes to the matching close
                // tag; the element never stays on the stack. One
                // exception inside the opacity: a `<script>` whose
                // `type` marks it a JSON *data block* (not executable
                // code) gets its string VALUES extracted as prose — see
                // [`scan_json_data_block`].
                if excluded {
                    *excluded_depth -= 1;
                }
                let end = raw_text_end(source, tag_end, &name);
                if name == "script"
                    && attribute_value(&source[at..tag_end], "type")
                        .is_some_and(|t| is_json_data_type(&t))
                {
                    let content_end = close_tag_start(source, tag_end, &name);
                    scan_json_data_block(source, tag_end, content_end, blocks, prose);
                }
                return end;
            }
            stack.push(OpenElement { name, excluded });
        }
        return tag_end;
    }
    // A stray `<` opening nothing: consume the one byte.
    at + 1
}

/// Pops the stack down through the innermost element named `name`, if
/// one is open — standard mis-nesting recovery. A close tag with no
/// matching open element is ignored.
fn close_element(name: &str, stack: &mut Vec<OpenElement>, excluded_depth: &mut usize) {
    let Some(position) = stack.iter().rposition(|e| e.name == name) else {
        return;
    };
    for popped in stack.drain(position..) {
        if popped.excluded {
            *excluded_depth -= 1;
        }
    }
}

/// Reads a lowercased tag name starting at `at`, returning it with the
/// position just past it.
fn tag_name(source: &str, at: usize) -> (String, usize) {
    let bytes = source.as_bytes();
    let mut end = at;
    while end < bytes.len() && (bytes[end].is_ascii_alphanumeric() || bytes[end] == b'-') {
        end += 1;
    }
    (source[at..end].to_ascii_lowercase(), end)
}

/// Consumes attributes from `at` to the tag's closing `>`, honoring
/// quoted values (a `>` inside `alt="a > b"` is data, not the tag end).
/// Returns the position past `>` and whether the tag was self-closed.
fn consume_attributes(bytes: &[u8], at: usize) -> (usize, bool) {
    let mut pos = at;
    let mut self_closed = false;
    while pos < bytes.len() {
        match bytes[pos] {
            b'>' => return (pos + 1, self_closed),
            b'"' | b'\'' => {
                let quote = bytes[pos];
                pos += 1;
                while pos < bytes.len() && bytes[pos] != quote {
                    pos += 1;
                }
                pos = (pos + 1).min(bytes.len());
                self_closed = false;
            }
            b'/' => {
                self_closed = true;
                pos += 1;
            }
            _ => {
                if bytes[pos] != b'/' && !bytes[pos].is_ascii_whitespace() {
                    self_closed = false;
                }
                pos += 1;
            }
        }
    }
    (bytes.len(), self_closed)
}

/// The position where `</name` begins for a raw-text element whose
/// content starts at `content_start`, scanning case-insensitively —
/// i.e. the end of the element's content. Unterminated raw text runs to
/// the end of the input.
fn close_tag_start(source: &str, content_start: usize, name: &str) -> usize {
    let lower = source.to_ascii_lowercase();
    let needle = format!("</{name}");
    lower[content_start..]
        .find(&needle)
        .map_or(source.len(), |found| content_start + found)
}

/// The position just past `</name ... >` for a raw-text element opened
/// at `content_start`, scanning case-insensitively. Unterminated raw
/// text swallows the rest of the input — the safe direction, since none
/// of it becomes prose.
fn raw_text_end(source: &str, content_start: usize, name: &str) -> usize {
    let close_start = close_tag_start(source, content_start, name);
    if close_start >= source.len() {
        return source.len();
    }
    find_from(source.as_bytes(), close_start, b">").map_or(source.len(), |i| i + 1)
}

/// The value of attribute `name` inside `tag` (the raw `<tag ...>` byte
/// slice), lowercased and unquoted, or `None` if absent. A minimal scan
/// sufficient for the one attribute this module reads (`type`).
fn attribute_value(tag: &str, name: &str) -> Option<String> {
    let lower = tag.to_ascii_lowercase();
    let bytes = lower.as_bytes();
    let mut search_from = 0;
    loop {
        let found = lower[search_from..].find(name)?;
        let start = search_from + found;
        search_from = start + name.len();
        // Must be a standalone attribute name: preceded by whitespace
        // and followed (after optional whitespace) by `=`.
        let preceded_ok = start > 0 && bytes[start - 1].is_ascii_whitespace();
        let mut pos = start + name.len();
        while pos < bytes.len() && bytes[pos].is_ascii_whitespace() {
            pos += 1;
        }
        if !preceded_ok || bytes.get(pos) != Some(&b'=') {
            continue;
        }
        pos += 1;
        while pos < bytes.len() && bytes[pos].is_ascii_whitespace() {
            pos += 1;
        }
        return match bytes.get(pos) {
            Some(&quote @ (b'"' | b'\'')) => {
                let value_start = pos + 1;
                let value_end = lower[value_start..]
                    .find(quote as char)
                    .map_or(lower.len(), |i| value_start + i);
                Some(lower[value_start..value_end].to_string())
            }
            Some(_) => {
                let value_start = pos;
                let value_end = lower[value_start..]
                    .find(|c: char| c.is_ascii_whitespace() || c == '>')
                    .map_or(lower.len(), |i| value_start + i);
                Some(lower[value_start..value_end].to_string())
            }
            None => None,
        };
    }
}

/// `true` if a `<script type="...">` value marks a JSON *data block* —
/// inert data the page reads, not executable code. Covers the two plain
/// JSON MIME types and the whole `+json` suffix family
/// (`application/ld+json`, `application/bento+json`, ...).
fn is_json_data_type(value: &str) -> bool {
    let value = value.trim();
    value == "application/json" || value == "text/json" || value.ends_with("+json")
}

/// Extracts prose from a JSON data block's string VALUES.
///
/// A hand-rolled forward lexer, not a JSON parser: it walks
/// `source[start..end]` finding string literals, treats a string as a
/// key exactly when the next non-whitespace byte after it is `:`, and
/// emits every other string's content as prose runs. Inside a value,
/// every backslash escape (`\"`, `\n`, `\uXXXX`, ...) is a run boundary
/// — the same rule character references get in HTML text, and with the
/// same consequence: a run's bytes are always literal text, no edit can
/// slice through an escape, and the inserted-word closure (plain ASCII
/// words only) can never produce bytes that need JSON escaping. The
/// document therefore stays valid JSON by construction.
///
/// Values without internal whitespace (ids, colors, font stacks come
/// close but keep their spaces) are skipped: a single token carries
/// nothing any operation could edit, and skipping it keeps
/// machine-value noise out of the register denominators.
fn scan_json_data_block(
    source: &str,
    start: usize,
    end: usize,
    blocks: &mut Vec<Block>,
    prose: &mut Vec<ProseUnit>,
) {
    let bytes = source.as_bytes();
    let mut pos = start;
    while pos < end {
        if bytes[pos] != b'"' {
            pos += 1;
            continue;
        }
        // A string literal: find its end, honoring escapes.
        let content_start = pos + 1;
        let mut cursor = content_start;
        while cursor < end && bytes[cursor] != b'"' {
            cursor += if bytes[cursor] == b'\\' { 2 } else { 1 };
        }
        let content_end = cursor.min(end);
        pos = (content_end + 1).min(end);

        // A key iff the next non-whitespace byte is `:`.
        let mut after = pos;
        while after < end && bytes[after].is_ascii_whitespace() {
            after += 1;
        }
        if bytes.get(after) == Some(&b':') {
            continue;
        }

        // Skip whole values with no internal whitespace.
        let value = &source[content_start..content_end];
        if !value.trim().contains(|c: char| c.is_ascii_whitespace()) {
            continue;
        }

        // Emit the value's escape-delimited runs.
        let mut run_start = content_start;
        let mut scan = content_start;
        while scan <= content_end {
            let boundary = scan == content_end || bytes[scan] == b'\\';
            if boundary {
                emit_json_run(source, run_start, scan, blocks, prose);
                if scan < content_end {
                    // `\uXXXX` is six bytes; every other escape is two.
                    scan += if bytes.get(scan + 1) == Some(&b'u') {
                        6
                    } else {
                        2
                    };
                } else {
                    scan += 1;
                }
                run_start = scan;
            } else {
                scan += 1;
            }
        }
    }
}

/// Emits one whitespace-trimmed prose run from a JSON string value, as
/// an in-scope paragraph block.
fn emit_json_run(
    source: &str,
    start: usize,
    end: usize,
    blocks: &mut Vec<Block>,
    prose: &mut Vec<ProseUnit>,
) {
    if start >= end {
        return;
    }
    let raw = &source[start..end];
    let trimmed_front = raw.len() - raw.trim_start().len();
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return;
    }
    let range = start + trimmed_front..start + trimmed_front + trimmed.len();
    let block_index = blocks.len();
    blocks.push(Block::new(BlockKind::Paragraph, range.clone()));
    prose.push(ProseUnit::new(block_index, range, Vec::new()));
}

/// The first position of `needle` in `bytes` at or after `from`.
fn find_from(bytes: &[u8], from: usize, needle: &[u8]) -> Option<usize> {
    if from >= bytes.len() {
        return None;
    }
    bytes[from..]
        .windows(needle.len())
        .position(|window| window == needle)
        .map(|i| from + i)
}

/// The end position (past the `;`) of a character reference starting at
/// the `&` at `at`, or `None` if the bytes there don't form one.
/// Recognizes `&name;`, `&#123;`, and `&#x1F;` shapes — the same three
/// the HTML spec defines.
fn character_reference_end(bytes: &[u8], at: usize) -> Option<usize> {
    let rest = &bytes[at + 1..];
    let body_len = if rest.starts_with(b"#x") || rest.starts_with(b"#X") {
        rest[2..]
            .iter()
            .take_while(|b| b.is_ascii_hexdigit())
            .count()
            + 2
    } else if rest.starts_with(b"#") {
        rest[1..].iter().take_while(|b| b.is_ascii_digit()).count() + 1
    } else {
        rest.iter()
            .take_while(|b| b.is_ascii_alphanumeric())
            .count()
    };
    let has_digits = match body_len {
        0 => false,
        _ if rest.starts_with(b"#x") || rest.starts_with(b"#X") => body_len > 2,
        _ if rest.starts_with(b"#") => body_len > 1,
        _ => rest[0].is_ascii_alphabetic(),
    };
    (has_digits && body_len <= 32 && rest.get(body_len) == Some(&b';'))
        .then(|| at + 1 + body_len + 1)
}

/// Elements whose entire content is never prose — see the module docs.
fn is_excluded_element(name: &str) -> bool {
    matches!(
        name,
        "head"
            | "script"
            | "style"
            | "pre"
            | "code"
            | "kbd"
            | "samp"
            | "var"
            | "tt"
            | "svg"
            | "math"
            | "template"
            | "noscript"
            | "textarea"
            | "select"
            | "option"
            | "iframe"
            | "object"
            | "canvas"
            | "title"
    )
}

/// Elements whose content the HTML spec tokenizes as raw text — a `<`
/// inside them is data, not markup.
fn is_raw_text_element(name: &str) -> bool {
    matches!(name, "script" | "style" | "textarea" | "title")
}

/// Void elements: no content, no close tag, never on the stack.
fn is_void_element(name: &str) -> bool {
    matches!(
        name,
        "area"
            | "base"
            | "br"
            | "col"
            | "embed"
            | "hr"
            | "img"
            | "input"
            | "link"
            | "meta"
            | "param"
            | "source"
            | "track"
            | "wbr"
    )
}

/// `true` if `source` looks like an HTML page rather than markdown.
///
/// The sniff `friction`'s CLI applies to stdin, where no file extension
/// can say. Only unambiguous page-level openers count: a markdown
/// document legitimately *containing* raw HTML almost never *starts*
/// with a doctype or `<html>`/`<head>`/`<body>` tag.
#[must_use]
pub fn looks_like_html(source: &str) -> bool {
    let trimmed = source.trim_start();
    let head = trimmed.get(..64).unwrap_or(trimmed);
    let lower = head.to_ascii_lowercase();
    lower.starts_with("<!doctype html")
        || lower.starts_with("<html")
        || lower.starts_with("<head")
        || lower.starts_with("<body")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prose_ranges(source: &str) -> Vec<(std::ops::Range<usize>, String)> {
        let (_, prose) = extract_html(source);
        prose
            .into_iter()
            .map(|unit| {
                let text = source[unit.range.clone()].to_string();
                (unit.range, text)
            })
            .collect()
    }

    #[test]
    fn extracts_paragraph_text_with_exact_offsets() {
        let source = "<html><body><p>Hello, world.</p></body></html>";
        let runs = prose_ranges(source);
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].1, "Hello, world.");
        assert_eq!(&source[runs[0].0.clone()], "Hello, world.");
    }

    #[test]
    fn inline_tags_split_runs_but_keep_all_text() {
        let source = "<p>the <em>fix</em> command runs</p>";
        let runs = prose_ranges(source);
        let texts: Vec<&str> = runs.iter().map(|(_, t)| t.as_str()).collect();
        assert_eq!(texts, vec!["the", "fix", "command runs"]);
    }

    #[test]
    fn head_script_style_and_code_are_never_prose() {
        let source = "<html><head><title>Page Title</title><style>p { color: red }</style>\
                      </head><body><script>let x = \"not prose\";</script>\
                      <p>Real prose.</p><pre>fn main() {}</pre>\
                      <p>Uses <code>friction fix</code> daily.</p></body></html>";
        let runs = prose_ranges(source);
        let texts: Vec<&str> = runs.iter().map(|(_, t)| t.as_str()).collect();
        assert_eq!(texts, vec!["Real prose.", "Uses", "daily."]);
    }

    #[test]
    fn raw_text_script_content_with_angle_brackets_is_opaque() {
        let source =
            "<body><script>if (a < b) { emit(\"<p>fake</p>\") }</script><p>after</p></body>";
        let runs = prose_ranges(source);
        let texts: Vec<&str> = runs.iter().map(|(_, t)| t.as_str()).collect();
        assert_eq!(texts, vec!["after"]);
    }

    #[test]
    fn character_references_split_runs_and_stay_untouched() {
        let source = "<p>ships &amp; runs</p><p>R&D budget</p><p>x &#8212; y</p>";
        let runs = prose_ranges(source);
        let texts: Vec<&str> = runs.iter().map(|(_, t)| t.as_str()).collect();
        assert_eq!(texts, vec!["ships", "runs", "R&D budget", "x", "y"]);
    }

    #[test]
    fn heading_and_table_cell_runs_carry_out_of_scope_kinds() {
        let source = "<h2>Getting Started</h2><table><tr><th>Name</th><td>value text</td></tr>\
                      </table><p>Body prose.</p>";
        let (blocks, prose) = extract_html(source);
        let kinds: Vec<&BlockKind> = prose.iter().map(|u| &blocks[u.block].kind).collect();
        assert_eq!(
            kinds,
            vec![
                &BlockKind::Heading { level: 2 },
                &BlockKind::TableCell { header: true },
                &BlockKind::TableCell { header: false },
                &BlockKind::Paragraph,
            ]
        );
    }

    #[test]
    fn list_items_and_blockquotes_carry_in_scope_kinds() {
        let source = "<ul><li>First point here.</li></ul><blockquote>Quoted line.</blockquote>";
        let (blocks, prose) = extract_html(source);
        let kinds: Vec<&BlockKind> = prose.iter().map(|u| &blocks[u.block].kind).collect();
        assert_eq!(kinds, vec![&BlockKind::ListItem, &BlockKind::BlockQuote]);
    }

    #[test]
    fn quoted_attribute_values_may_contain_angle_brackets() {
        let source = "<p><img alt=\"a > b\" src=\"x.png\">after the image</p>";
        let runs = prose_ranges(source);
        let texts: Vec<&str> = runs.iter().map(|(_, t)| t.as_str()).collect();
        assert_eq!(texts, vec!["after the image"]);
    }

    #[test]
    fn comments_doctype_and_cdata_are_skipped() {
        let source = "<!DOCTYPE html><!-- a <p>commented</p> paragraph --><p>kept</p>";
        let runs = prose_ranges(source);
        let texts: Vec<&str> = runs.iter().map(|(_, t)| t.as_str()).collect();
        assert_eq!(texts, vec!["kept"]);
    }

    #[test]
    fn void_elements_do_not_poison_context() {
        let source = "<p>before<br>after</p><p>with <img src=\"x\"> image</p>";
        let runs = prose_ranges(source);
        let texts: Vec<&str> = runs.iter().map(|(_, t)| t.as_str()).collect();
        assert_eq!(texts, vec!["before", "after", "with", "image"]);
    }

    #[test]
    fn unclosed_and_misnested_tags_recover() {
        // The stray `</em>` (never opened) is ignored; the unclosed `<b>`
        // never blocks the following paragraph's text.
        let source = "<p>one</em> two<b> three</p><p>four</p>";
        let runs = prose_ranges(source);
        let texts: Vec<&str> = runs.iter().map(|(_, t)| t.as_str()).collect();
        assert_eq!(texts, vec!["one", "two", "three", "four"]);
    }

    #[test]
    fn unterminated_script_swallows_the_tail_safely() {
        let source = "<p>kept</p><script>never closed";
        let runs = prose_ranges(source);
        let texts: Vec<&str> = runs.iter().map(|(_, t)| t.as_str()).collect();
        assert_eq!(texts, vec!["kept"]);
    }

    #[test]
    fn extraction_is_deterministic() {
        let source = "<h1>T</h1><p>Alpha &amp; beta.</p><ul><li>x</li></ul>";
        let (blocks_a, prose_a) = extract_html(source);
        let (blocks_b, prose_b) = extract_html(source);
        assert_eq!(blocks_a.len(), blocks_b.len());
        assert_eq!(
            prose_a.iter().map(|u| u.range.clone()).collect::<Vec<_>>(),
            prose_b.iter().map(|u| u.range.clone()).collect::<Vec<_>>(),
        );
    }

    #[test]
    fn json_data_block_values_become_prose_and_keys_do_not() {
        let source = r#"<script type="application/bento+json">
{"title": "The Results Deck", "id": "cover",
 "notes": "The comparison covers both branches.",
 "tags": ["read path", "single"]}
</script><script>let code = "executable strings stay out";</script>"#;
        let runs = prose_ranges(source);
        let texts: Vec<&str> = runs.iter().map(|(_, t)| t.as_str()).collect();
        // Keys ("title", "notes", ...) never appear; single-token values
        // ("cover", "single") are skipped; executable-script strings are
        // untouchable.
        assert_eq!(
            texts,
            vec![
                "The Results Deck",
                "The comparison covers both branches.",
                "read path",
            ]
        );
    }

    #[test]
    fn json_escapes_split_runs_and_are_never_prose() {
        let source = r#"<script type="application/json">
{"notes": "first line.\nSecond line with \"quoted\" text and — dash."}
</script>"#;
        let runs = prose_ranges(source);
        let texts: Vec<&str> = runs.iter().map(|(_, t)| t.as_str()).collect();
        // The literal `—` is raw text, not an escape: it stays inside
        // its run, which is what lets the em-dash operation see it with
        // its surrounding clauses.
        assert_eq!(
            texts,
            vec![
                "first line.",
                "Second line with",
                "quoted",
                "text and — dash."
            ]
        );
    }

    #[test]
    fn non_json_script_types_stay_opaque() {
        let source = r#"<script id="rt" type="bento/deflate-b64">eJzT08vP "not prose" x</script>"#;
        assert!(prose_ranges(source).is_empty());
    }

    #[test]
    fn attribute_value_reads_quoted_unquoted_and_absent() {
        assert_eq!(
            attribute_value(r#"<script type="application/json" id=x"#, "type"),
            Some("application/json".to_string())
        );
        assert_eq!(
            attribute_value("<script type=text/json>", "type"),
            Some("text/json".to_string())
        );
        assert_eq!(attribute_value("<script id=x>", "type"), None);
        // `data-type` must not satisfy a lookup for `type`.
        assert_eq!(
            attribute_value(r#"<script data-type="a+json">"#, "type"),
            None
        );
    }

    #[test]
    fn sniff_recognizes_page_openers_only() {
        assert!(looks_like_html("<!DOCTYPE html><html>..."));
        assert!(looks_like_html("  <html lang=\"en\">"));
        assert!(looks_like_html("<body>"));
        assert!(!looks_like_html(
            "# A markdown heading\n\n<div>raw html block</div>"
        ));
        assert!(!looks_like_html("Plain prose."));
        assert!(!looks_like_html(
            "<div>markdown may start with raw html</div>"
        ));
    }
}
