//! `pulldown-cmark` offset-event walk that builds a [`Block`] tree and
//! extracts prose runs into [`ProseUnit`]s.
//!
//! # Algorithm
//!
//! `pulldown-cmark`'s [`Parser::into_offset_iter`] yields `(Event, Range)`
//! pairs; for every block-level tag it hands us the *exact* byte range of
//! that construct (markup included) at `Start` time already — `Start` and
//! `End` carry an identical range for these tags, so a block never needs
//! its `End` event to know its own span. [`ExtractState::on_start`] turns
//! each block-level `Start` directly into a [`Block`] pushed onto a flat,
//! pre-order (and therefore non-decreasing-start, "in source order") list.
//!
//! At most one *prose session* is open at a time, scoped to the innermost
//! currently-open block that can hold inline content directly: a
//! paragraph, heading, table cell, or list item (loose list items wrap
//! their text in a nested paragraph instead, which opens its own session;
//! a tight list item's `Start(Item)` is immediately followed by inline
//! events with no such wrapper). Two prose-bearing containers can never be
//! open at once — `CommonMark`'s grammar doesn't nest them — so a single
//! [`Session`] slot suffices; encountering *any* nested block-level
//! `Start` while a session is open (e.g. a sublist inside a tight list
//! item) ends it immediately, since whatever content precedes the nested
//! block is everything that container will ever contribute directly.
//!
//! Within a session, [`Session::extend_touching`] merges leaf events
//! (`Text`/`SoftBreak`/`HardBreak`) into a run only when they sit exactly
//! byte-adjacent to the run so far; a gap — most often the `>` / `> `
//! block-quote continuation marker on a wrapped line, which never appears
//! in the inline event stream at all — ends the run and starts a new one.
//! Emphasis, strong, and strikethrough markup is different: their `**`,
//! `_`, `~~` delimiter bytes aren't covered by any leaf event either, but
//! they're still literal prose punctuation, not structure to exclude — so
//! [`Session::bridge`] force-merges across them (tracked via
//! [`Session::depth`], since nested delimiters must all bridge). A
//! backslash-escaped character (`\*`, `\_`, `` \` ``, `\.`, an escaped
//! table pipe `\|`, ...) is the same story at a single-byte scale:
//! `pulldown-cmark` emits a `Text` event for the escaped character alone,
//! starting *after* the backslash, so the backslash byte is covered by no
//! event either — but it's still one literal prose character (`\*` means
//! a literal `*`), not structure. [`Session::extend_touching`] therefore
//! special-cases exactly a one-byte gap whose sole byte is `\`: it bridges
//! across it instead of closing the run. Inline code, raw/inline HTML,
//! footnote references, and task-list checkboxes are genuinely excluded:
//! [`Session::close_current`] ends the run without extending across them.
//! Links and images are excluded structurally (`[`, `](url)`) but their
//! label *is* prose, so their `Start`/`End` force a real break on both
//! sides rather than bridging — the label becomes its own separate run.

use std::ops::Range;

use friction_core::{Block, BlockKind, ProseUnit};
use pulldown_cmark::{CodeBlockKind, Event, HeadingLevel, Options, Parser, Tag, TagEnd};

/// Markdown extensions enabled for prose extraction: GFM tables,
/// footnotes, strikethrough, task-list checkboxes, and heading attributes.
/// Deliberately excludes `ENABLE_SMART_PUNCTUATION` (would substitute
/// typographic characters, breaking the byte-exact round-trip guarantee
/// for prose ranges) and the experimental math/definition-list/wikilink
/// extensions.
const OPTIONS: Options = Options::ENABLE_TABLES
    .union(Options::ENABLE_FOOTNOTES)
    .union(Options::ENABLE_STRIKETHROUGH)
    .union(Options::ENABLE_TASKLISTS)
    .union(Options::ENABLE_HEADING_ATTRIBUTES);

/// Parses `source` into its markdown block tree and extracted prose runs.
/// See the module documentation for the extraction algorithm.
pub fn extract(source: &str) -> (Vec<Block>, Vec<ProseUnit>) {
    let mut state = ExtractState {
        source,
        ..ExtractState::default()
    };
    for (event, range) in Parser::new_ext(source, OPTIONS).into_offset_iter() {
        match event {
            Event::Start(tag) => state.on_start(&tag, range),
            Event::End(tag_end) => state.on_end(tag_end, range),
            Event::Text(_) | Event::SoftBreak | Event::HardBreak => state.on_leaf_prose(range),
            Event::Code(_)
            | Event::InlineMath(_)
            | Event::DisplayMath(_)
            | Event::Html(_)
            | Event::InlineHtml(_)
            | Event::FootnoteReference(_)
            | Event::TaskListMarker(_) => state.on_leaf_excluded(),
            Event::Rule => state.on_rule(range),
        }
    }
    state.finish()
}

/// A maximal contiguous run of "prose bytes" being accumulated for the one
/// currently-open prose-bearing block, plus the runs already closed out.
#[derive(Debug)]
struct Session {
    /// Index into the (still being built) block list of the block this
    /// session's runs will be attributed to.
    block: usize,
    /// Nesting depth of open emphasis/strong/strikethrough wrappers; while
    /// positive, leaf events force-bridge instead of requiring byte
    /// adjacency (see module docs).
    depth: u32,
    /// The run currently being extended, if any.
    current: Option<Range<usize>>,
    /// Runs already closed out for this session, in source order.
    runs: Vec<Range<usize>>,
}

impl Session {
    const fn new(block: usize) -> Self {
        Self {
            block,
            depth: 0,
            current: None,
            runs: Vec::new(),
        }
    }

    /// Extends the current run by `range` if it starts exactly where the
    /// current run ends, or if the single byte separating them is a
    /// backslash escaping `range`'s leading character (see module docs) —
    /// in which case the backslash is folded into the run too, since it's
    /// ordinary prose bytes, not structure. Otherwise closes the current
    /// run (if non-empty) and starts a new one at `range`.
    fn extend_touching(&mut self, range: Range<usize>, source: &str) {
        match &mut self.current {
            Some(run) if run.end == range.start => run.end = range.end,
            Some(run) if is_backslash_escape_gap(run.end, range.start, source) => {
                run.end = range.end;
            }
            Some(_) => {
                self.close_current();
                self.current = Some(range);
            }
            None => self.current = Some(range),
        }
    }

    /// Force-extends the current run's end to `range.end`, ignoring any
    /// gap (used for emphasis-like delimiter bytes not covered by any leaf
    /// event); starts a fresh run at `range` if none is open.
    const fn bridge(&mut self, range: Range<usize>) {
        match &mut self.current {
            Some(run) => run.end = range.end,
            None => self.current = Some(range),
        }
    }

    /// Closes the current run, pushing it to `runs` unless it's empty.
    fn close_current(&mut self) {
        if let Some(run) = self.current.take()
            && !run.is_empty()
        {
            self.runs.push(run);
        }
    }

    /// Closes any open run and turns every accumulated run into a
    /// [`ProseUnit`] attributed to this session's block.
    fn finish(mut self) -> Vec<ProseUnit> {
        self.close_current();
        self.runs
            .into_iter()
            .map(|range| ProseUnit::new(self.block, range, Vec::new()))
            .collect()
    }
}

/// `true` if `[gap_start, gap_end)` in `source` is exactly the single byte
/// `\` — the backslash of a `pulldown-cmark` backslash-escape, which is
/// never covered by any event of its own (see module docs).
fn is_backslash_escape_gap(gap_start: usize, gap_end: usize, source: &str) -> bool {
    gap_end == gap_start + 1 && source.as_bytes().get(gap_start) == Some(&b'\\')
}

/// Accumulator for a single left-to-right pass over the offset-event
/// stream (see [`extract`]).
#[derive(Debug, Default)]
struct ExtractState<'a> {
    blocks: Vec<Block>,
    prose: Vec<ProseUnit>,
    session: Option<Session>,
    /// `true` while inside a GFM `TableHead`, so nested `TableCell`s are
    /// marked as header cells.
    table_header: bool,
    /// The full document source, needed to detect backslash-escape gaps
    /// (see [`is_backslash_escape_gap`]) between otherwise-disjoint leaf
    /// events.
    source: &'a str,
}

impl ExtractState<'_> {
    fn finish(mut self) -> (Vec<Block>, Vec<ProseUnit>) {
        self.flush_session();
        (self.blocks, self.prose)
    }

    /// Ends the active session (if any), turning its accumulated runs into
    /// `ProseUnit`s.
    fn flush_session(&mut self) {
        if let Some(session) = self.session.take() {
            self.prose.extend(session.finish());
        }
    }

    fn push_block(&mut self, kind: BlockKind, range: Range<usize>) -> usize {
        let index = self.blocks.len();
        self.blocks.push(Block::new(kind, range));
        index
    }

    fn on_start(&mut self, tag: &Tag<'_>, range: Range<usize>) {
        if let Some(kind) = block_kind_for_start(tag, self.table_header) {
            // Any block-level `Start` — including one nested inside the
            // block a session is currently scanning, e.g. a sublist inside
            // a tight list item — ends that session: everything the
            // scanned block contributes directly to prose is already
            // behind us.
            self.flush_session();
            match tag {
                Tag::TableHead => self.table_header = true,
                Tag::TableRow => self.table_header = false,
                _ => {}
            }
            let index = self.push_block(kind, range);
            if is_prose_container(tag) {
                self.session = Some(Session::new(index));
            }
            return;
        }

        let Some(session) = &mut self.session else {
            return;
        };
        match tag {
            Tag::Emphasis | Tag::Strong | Tag::Strikethrough => {
                session.depth += 1;
                session.bridge(range.start..range.start);
            }
            Tag::Link { .. } | Tag::Image { .. } => session.close_current(),
            _ => {}
        }
    }

    fn on_end(&mut self, tag_end: TagEnd, range: Range<usize>) {
        match tag_end {
            TagEnd::Paragraph | TagEnd::Heading(_) | TagEnd::TableCell | TagEnd::Item => {
                self.flush_session();
            }
            TagEnd::Emphasis | TagEnd::Strong | TagEnd::Strikethrough => {
                if let Some(session) = &mut self.session {
                    session.bridge(range.end..range.end);
                    session.depth = session.depth.saturating_sub(1);
                }
            }
            TagEnd::Link | TagEnd::Image => {
                if let Some(session) = &mut self.session {
                    session.close_current();
                }
            }
            _ => {}
        }
    }

    /// `Text`/`SoftBreak`/`HardBreak`: prose leaf bytes.
    fn on_leaf_prose(&mut self, range: Range<usize>) {
        let source = self.source;
        let Some(session) = &mut self.session else {
            return;
        };
        if session.depth > 0 {
            session.bridge(range);
        } else {
            session.extend_touching(range, source);
        }
    }

    /// `Code`/`Html`/`InlineHtml`/`FootnoteReference`/`TaskListMarker`
    /// (and the unstable `InlineMath`/`DisplayMath`): excluded from prose.
    fn on_leaf_excluded(&mut self) {
        if let Some(session) = &mut self.session {
            session.close_current();
        }
    }

    fn on_rule(&mut self, range: Range<usize>) {
        self.flush_session();
        self.push_block(BlockKind::ThematicBreak, range);
    }
}

/// `true` for block-level tags whose direct children may be inline
/// (prose) content: paragraphs, headings, table cells, and list items
/// (tight lists only — loose items wrap their text in a nested paragraph,
/// which is itself prose-eligible).
const fn is_prose_container(tag: &Tag<'_>) -> bool {
    matches!(
        tag,
        Tag::Paragraph | Tag::Heading { .. } | Tag::TableCell | Tag::Item
    )
}

/// Maps a block-level `pulldown-cmark` [`Tag`] to its [`BlockKind`],
/// returning `None` for span-level (inline) tags, which are handled by
/// [`ExtractState::on_start`] directly.
fn block_kind_for_start(tag: &Tag<'_>, table_header: bool) -> Option<BlockKind> {
    match tag {
        Tag::Paragraph => Some(BlockKind::Paragraph),
        Tag::Heading { level, .. } => Some(BlockKind::Heading {
            level: heading_level_number(*level),
        }),
        Tag::BlockQuote(_) => Some(BlockKind::BlockQuote),
        Tag::CodeBlock(kind) => Some(BlockKind::CodeBlock {
            info: code_block_info(kind),
        }),
        Tag::HtmlBlock => Some(BlockKind::HtmlBlock),
        Tag::List(start) => Some(BlockKind::List {
            ordered: start.is_some(),
            start: *start,
        }),
        Tag::Item => Some(BlockKind::ListItem),
        Tag::FootnoteDefinition(label) => Some(BlockKind::FootnoteDefinition {
            label: label.as_ref().into(),
        }),
        Tag::Table(_) => Some(BlockKind::Table),
        Tag::TableHead | Tag::TableRow => Some(BlockKind::TableRow),
        Tag::TableCell => Some(BlockKind::TableCell {
            header: table_header,
        }),
        _ => None,
    }
}

/// `HeadingLevel` to the `1..=6` scale [`BlockKind::Heading`] documents.
const fn heading_level_number(level: HeadingLevel) -> u8 {
    match level {
        HeadingLevel::H1 => 1,
        HeadingLevel::H2 => 2,
        HeadingLevel::H3 => 3,
        HeadingLevel::H4 => 4,
        HeadingLevel::H5 => 5,
        HeadingLevel::H6 => 6,
    }
}

/// The fence info string for a fenced code block, or `None` for an
/// indented block or an empty (language-less) fence.
fn code_block_info(kind: &CodeBlockKind<'_>) -> Option<Box<str>> {
    match kind {
        CodeBlockKind::Indented => None,
        CodeBlockKind::Fenced(info) if info.is_empty() => None,
        CodeBlockKind::Fenced(info) => Some(Box::from(info.as_ref())),
    }
}

#[cfg(test)]
mod tests {
    use friction_core::Spanned;

    use super::*;

    fn ranges(units: &[ProseUnit]) -> Vec<Range<usize>> {
        units.iter().map(Spanned::range).collect()
    }

    /// A plain paragraph becomes one block and one prose run
    /// spanning exactly its text (no trailing newline).
    #[test]
    fn paragraph_is_single_prose_run() {
        let source = "Hello, world.\n";
        let (blocks, prose) = extract(source);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].kind, BlockKind::Paragraph);
        assert_eq!(ranges(&prose), vec![0..13]);
        assert_eq!(&source[0..13], "Hello, world.");
    }

    /// An ATX heading's prose excludes the `#` marker and trailing
    /// newline.
    #[test]
    fn atx_heading_excludes_marker() {
        let source = "## Title Here\n";
        let (blocks, prose) = extract(source);
        assert_eq!(blocks[0].kind, BlockKind::Heading { level: 2 });
        assert_eq!(&source[ranges(&prose)[0].clone()], "Title Here");
    }

    /// A setext heading's prose excludes the underline.
    #[test]
    fn setext_heading_excludes_underline() {
        let source = "Title\n=====\n\nBody.\n";
        let (_, prose) = extract(source);
        assert_eq!(&source[ranges(&prose)[0].clone()], "Title");
        assert_eq!(&source[ranges(&prose)[1].clone()], "Body.");
    }

    /// Emphasis/strong/strikethrough delimiters are bridged into
    /// one contiguous prose run rather than fragmenting it.
    #[test]
    fn emphasis_markup_bridges_into_one_run() {
        let source = "He is **bold** and *italic* and ~~gone~~ now.\n";
        let (_, prose) = extract(source);
        let r = ranges(&prose);
        assert_eq!(r.len(), 1, "expected one contiguous run, got {r:?}");
        assert_eq!(
            &source[r[0].clone()],
            "He is **bold** and *italic* and ~~gone~~ now."
        );
    }

    /// Inline code excludes the code span but keeps the
    /// surrounding text as separate prose runs.
    #[test]
    fn inline_code_splits_prose_run() {
        let source = "See `code` here.\n";
        let (_, prose) = extract(source);
        let r = ranges(&prose);
        assert_eq!(r.len(), 2);
        assert_eq!(&source[r[0].clone()], "See ");
        assert_eq!(&source[r[1].clone()], " here.");
        for run in &r {
            assert!(!source[run.clone()].contains("code"));
        }
    }

    /// Link text is prose; the URL and surrounding `[...] (...)`
    /// markup are excluded and end up in a run of their own.
    #[test]
    fn link_excludes_url_keeps_label() {
        let source = "See [the text](https://example.com/path) end.\n";
        let (_, prose) = extract(source);
        let r = ranges(&prose);
        let texts: Vec<&str> = r.iter().map(|range| &source[range.clone()]).collect();
        assert!(texts.contains(&"the text"));
        assert!(texts.iter().any(|t| t.contains("See")));
        assert!(texts.iter().any(|t| t.contains("end.")));
        assert!(!texts.iter().any(|t| t.contains("example.com")));
    }

    /// A fenced code block contributes no prose at all.
    #[test]
    fn fenced_code_block_excluded_entirely() {
        let source = "Before.\n\n```rust\nfn f() {}\n```\n\nAfter.\n";
        let (blocks, prose) = extract(source);
        assert!(
            blocks
                .iter()
                .any(|b| matches!(&b.kind, BlockKind::CodeBlock { info: Some(lang) } if lang.as_ref() == "rust"))
        );
        for unit in &prose {
            assert!(!source[unit.range.clone()].contains("fn f()"));
        }
    }

    /// An indented code block contributes no prose.
    #[test]
    fn indented_code_block_excluded_entirely() {
        let source = "Before.\n\n    fn f() {}\n\nAfter.\n";
        let (blocks, prose) = extract(source);
        assert!(
            blocks
                .iter()
                .any(|b| matches!(&b.kind, BlockKind::CodeBlock { info: None }))
        );
        for unit in &prose {
            assert!(!source[unit.range.clone()].contains("fn f()"));
        }
    }

    /// An HTML block contributes no prose.
    #[test]
    fn html_block_excluded_entirely() {
        let source = "<div>\nraw html\n</div>\n\nAfter.\n";
        let (blocks, prose) = extract(source);
        assert!(blocks.iter().any(|b| b.kind == BlockKind::HtmlBlock));
        for unit in &prose {
            assert!(!source[unit.range.clone()].contains("raw html"));
        }
        assert!(prose.iter().any(|u| &source[u.range.clone()] == "After."));
    }

    /// A block-quote continuation marker (`> ` on a wrapped line)
    /// is not covered by any inline event and correctly splits the
    /// paragraph's prose into two runs, excluding the marker itself.
    #[test]
    fn blockquote_continuation_marker_excluded() {
        let source = "> Quoted text here.\n> Second line.\n";
        let (_, prose) = extract(source);
        let r = ranges(&prose);
        assert_eq!(r.len(), 2);
        assert_eq!(&source[r[0].clone()], "Quoted text here.\n");
        assert_eq!(&source[r[1].clone()], "Second line.");
    }

    /// A tight list item's text is directly prose (no wrapping
    /// paragraph).
    #[test]
    fn tight_list_item_is_prose() {
        let source = "- one\n- two\n";
        let (blocks, prose) = extract(source);
        let item_count = blocks
            .iter()
            .filter(|b| b.kind == BlockKind::ListItem)
            .count();
        assert_eq!(item_count, 2);
        assert_eq!(prose.len(), 2);
        assert_eq!(&source[prose[0].range.clone()], "one");
        assert_eq!(&source[prose[1].range.clone()], "two");
    }

    /// A loose list item's text comes from its nested paragraph,
    /// not the item itself.
    #[test]
    fn loose_list_item_prose_comes_from_nested_paragraph() {
        let source = "- one\n\n- two\n\n- three\n";
        let (blocks, prose) = extract(source);
        let paragraph_count = blocks
            .iter()
            .filter(|b| b.kind == BlockKind::Paragraph)
            .count();
        assert_eq!(paragraph_count, 3);
        assert_eq!(prose.len(), 3);
        for unit in &prose {
            let owner = &blocks[unit.block];
            assert_eq!(owner.kind, BlockKind::Paragraph);
        }
    }

    /// Table cell text is prose; table structure is not. Header
    /// cells are flagged.
    #[test]
    fn table_cells_are_prose_with_header_flag() {
        let source = "| a | b |\n| --- | --- |\n| 1 | two |\n";
        let (blocks, prose) = extract(source);
        let header_cells = blocks
            .iter()
            .filter(|b| matches!(b.kind, BlockKind::TableCell { header: true }))
            .count();
        let body_cells = blocks
            .iter()
            .filter(|b| matches!(b.kind, BlockKind::TableCell { header: false }))
            .count();
        assert_eq!(header_cells, 2);
        assert_eq!(body_cells, 2);
        let texts: Vec<&str> = prose.iter().map(|u| &source[u.range.clone()]).collect();
        assert!(texts.contains(&"a"));
        assert!(texts.contains(&"b"));
        assert!(texts.contains(&"1"));
        assert!(texts.contains(&"two"));
    }

    /// A thematic break produces a block and never panics, and
    /// correctly flushes any (impossible, but defensively checked) open
    /// session.
    #[test]
    fn thematic_break_is_its_own_block() {
        let source = "Before.\n\n---\n\nAfter.\n";
        let (blocks, prose) = extract(source);
        assert!(blocks.iter().any(|b| b.kind == BlockKind::ThematicBreak));
        assert_eq!(prose.len(), 2);
    }

    /// A footnote reference marker is excluded; the footnote
    /// definition's own text is prose via its nested paragraph.
    #[test]
    fn footnote_reference_excluded_definition_is_prose() {
        let source = "See note.[^1]\n\n[^1]: The footnote text.\n";
        let (blocks, prose) = extract(source);
        assert!(blocks.iter().any(
            |b| matches!(&b.kind, BlockKind::FootnoteDefinition { label } if label.as_ref() == "1")
        ));
        let texts: Vec<&str> = prose.iter().map(|u| &source[u.range.clone()]).collect();
        assert!(texts.contains(&"See note."));
        assert!(texts.contains(&"The footnote text."));
        assert!(!texts.iter().any(|t| t.contains("[^1]")));
    }

    /// A GFM task-list checkbox marker is excluded from prose.
    #[test]
    fn task_list_marker_excluded() {
        let source = "- [ ] unchecked\n- [x] checked\n";
        let (_, prose) = extract(source);
        let texts: Vec<&str> = prose.iter().map(|u| &source[u.range.clone()]).collect();
        assert_eq!(texts, vec!["unchecked", "checked"]);
    }

    /// CRLF line endings inside a paragraph are preserved as
    /// ordinary prose bytes (no normalization).
    #[test]
    fn crlf_preserved_in_prose() {
        let source = "Line one.\r\nLine two.\r\n";
        let (_, prose) = extract(source);
        assert_eq!(prose.len(), 1);
        assert_eq!(&source[prose[0].range.clone()], "Line one.\r\nLine two.");
    }

    /// A sublist nested inside a tight list item ends that item's
    /// prose session; the item's own text (before the sublist) is still
    /// captured, and the sublist's items get their own prose runs.
    #[test]
    fn nested_list_inside_tight_item() {
        let source = "- outer\n  - inner\n- outer2\n";
        let (blocks, prose) = extract(source);
        let texts: Vec<&str> = prose.iter().map(|u| &source[u.range.clone()]).collect();
        assert!(texts.contains(&"outer"));
        assert!(texts.contains(&"inner"));
        assert!(texts.contains(&"outer2"));
        assert!(
            blocks
                .iter()
                .filter(|b| b.kind == BlockKind::ListItem)
                .count()
                >= 3
        );
    }

    /// A backslash-escaped punctuation character (`\*`) is bridged
    /// into the surrounding prose as one contiguous run, backslash
    /// included, rather than fragmenting the sentence into two disjoint
    /// `ProseUnit`s with the backslash byte belonging to neither.
    #[test]
    fn backslash_escaped_punctuation_bridges_into_one_run() {
        let source = "Use a\\*b to mean literal star.\n";
        let (_, prose) = extract(source);
        let r = ranges(&prose);
        assert_eq!(r.len(), 1, "expected one contiguous run, got {r:?}");
        assert_eq!(&source[r[0].clone()], "Use a\\*b to mean literal star.");
    }

    /// An escaped pipe inside a GFM table cell is bridged into the
    /// cell's prose run rather than splitting it and dropping the
    /// backslash.
    #[test]
    fn backslash_escaped_pipe_in_table_cell_bridges() {
        let source = "| a\\|b | c |\n|---|---|\n";
        let (_, prose) = extract(source);
        let texts: Vec<&str> = prose.iter().map(|u| &source[u.range.clone()]).collect();
        assert!(
            texts.contains(&"a\\|b"),
            "expected one run \"a\\|b\", got {texts:?}"
        );
    }

    /// Several backslash escapes in the same sentence all bridge,
    /// not just the first.
    #[test]
    fn multiple_backslash_escapes_all_bridge() {
        let source = "a\\*b\\_c\\`d\\.e\n";
        let (_, prose) = extract(source);
        let r = ranges(&prose);
        assert_eq!(r.len(), 1, "expected one contiguous run, got {r:?}");
        assert_eq!(&source[r[0].clone()], "a\\*b\\_c\\`d\\.e");
    }

    /// `is_backslash_escape_gap` only fires for an exact one-byte `\` gap
    /// — not for any other single-byte gap, and not for a multi-byte gap
    /// (e.g. the blockquote continuation marker, which must keep splitting
    /// runs as covered by `blockquote_continuation_marker_excluded`).
    #[test]
    fn is_backslash_escape_gap_requires_exact_one_byte_backslash() {
        assert!(super::is_backslash_escape_gap(
            5,
            6,
            "Use a\\b to mean literal star."
        ));
        assert!(!super::is_backslash_escape_gap(
            5,
            6,
            "Use a b to mean literal star."
        ));
        assert!(!super::is_backslash_escape_gap(0, 2, "\\\\b"));
    }

    /// Emoji, CJK, and combining-mark text is captured verbatim as
    /// a single prose run, with no char-boundary or byte-loss issues.
    #[test]
    fn unicode_text_captured_verbatim() {
        let source = "café 日本語 combining e\u{0301} emoji 🎉 end.\n";
        let (_, prose) = extract(source);
        assert_eq!(prose.len(), 1);
        assert_eq!(
            &source[prose[0].range.clone()],
            "café 日本語 combining e\u{0301} emoji 🎉 end."
        );
    }
}
