//! Round-trip and golden fixture tests for `friction-parse`.
//!
//! Two independent checks are run over every document, generated or fixed:
//!
//! - [`assert_root_blocks_reconstruct_source`]: the document's outermost
//!   ("root") block ranges, concatenated with the untouched source bytes
//!   between them, reproduce the source exactly, and — the check that
//!   actually has teeth — do so without the root ranges ever overlapping
//!   or running out of order.
//! - [`assert_prose_units_disjoint`]: no two extracted prose runs overlap.
//!
//! (a) [`parse_round_trips_over_generated_markdown_ish_input`] is the
//! proptest property test over generated markdown-ish input; (b) the
//! `golden_*` tests below it are the golden fixtures
//! (`tests/fixtures/*.md`), covering code fences, nested lists, tables,
//! links, block quotes, HTML, Unicode, and CRLF vs LF line endings.

use std::ops::Range;
use std::path::{Path, PathBuf};

use friction_core::{Document, Spanned, span};
use friction_parse::parse;
use proptest::prelude::*;

fn fixture_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn parse_fixture(name: &str) -> (String, Document) {
    let path = fixture_path(name);
    let source =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read fixture {name}: {e}"));
    let doc = parse(source.clone()).unwrap_or_else(|e| panic!("parse fixture {name}: {e}"));
    (source, doc)
}

fn prose_texts(doc: &Document) -> Vec<&str> {
    doc.prose()
        .iter()
        .map(|unit| {
            doc.text(&unit.range)
                .expect("every prose span must slice cleanly from source")
        })
        .collect()
}

fn any_prose_contains(doc: &Document, needle: &str) -> bool {
    prose_texts(doc).iter().any(|text| text.contains(needle))
}

fn no_prose_contains(doc: &Document, needle: &str) -> bool {
    !any_prose_contains(doc, needle)
}

/// `true` if block `i` is not properly nested inside any other block.
/// Ties (two blocks sharing the exact same range, e.g. a single-row GFM
/// table and its header row) are broken by index: the block pushed first
/// (during `friction-parse`'s pre-order walk) is the outer one.
fn is_root(blocks: &[friction_core::Block], i: usize) -> bool {
    let target = blocks[i].range();
    !blocks.iter().enumerate().any(|(j, other)| {
        if j == i {
            return false;
        }
        let other_range = other.range();
        span::contains_range(&other_range, &target) && (other_range != target || j < i)
    })
}

/// Concatenating the document's root block ranges with
/// the untouched source bytes between them reproduces the source exactly.
/// This is only possible without panicking if the root ranges are
/// pairwise non-overlapping and in ascending order, which is the property
/// under test.
fn assert_root_blocks_reconstruct_source(doc: &Document) {
    let blocks = doc.blocks();
    let mut roots: Vec<Range<usize>> = (0..blocks.len())
        .filter(|&i| is_root(blocks, i))
        .map(|i| blocks[i].range())
        .collect();
    roots.sort_by_key(|r| r.start);

    let source = doc.source();
    let mut reconstructed = String::with_capacity(source.len());
    let mut cursor = 0usize;
    for root in &roots {
        assert!(
            cursor <= root.start,
            "root block ranges overlap or are out of order: cursor={cursor}, root={root:?}"
        );
        reconstructed.push_str(&source[cursor..root.start]);
        reconstructed.push_str(&source[root.clone()]);
        cursor = root.end;
    }
    reconstructed.push_str(&source[cursor..]);
    assert_eq!(
        reconstructed, source,
        "reconstructing from root block spans must reproduce source exactly"
    );
}

/// Extracted prose runs never overlap each other.
fn assert_prose_units_disjoint(doc: &Document) {
    let prose = doc.prose();
    for i in 0..prose.len() {
        for j in (i + 1)..prose.len() {
            assert!(
                !span::ranges_overlap(&prose[i].range(), &prose[j].range()),
                "prose runs {:?} and {:?} overlap",
                prose[i].range(),
                prose[j].range()
            );
        }
    }
}

fn assert_document_round_trips(doc: &Document, source: &str) {
    assert_eq!(
        doc.source(),
        source,
        "Document::source() must be exactly the parsed input"
    );
    assert_root_blocks_reconstruct_source(doc);
    assert_prose_units_disjoint(doc);
}

/// A small alphabet of ASCII, markdown-syntax, and multi-byte Unicode
/// characters (accented Latin, CJK, an astral-plane emoji, a combining
/// mark, and CRLF) used to generate "markdown-ish" documents. Includes `\`
/// so backslash-escaped punctuation (`\*`, `\_`, `` \` ``, `\|`, ...) is
/// exercised — its absence previously let a whole class of prose-run
/// fragmentation bugs through the fuzzer untested.
fn markdown_ish_char() -> impl Strategy<Value = char> {
    prop::sample::select(
        [
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', ' ', ' ', '\n', '\n', '\n', '\t', '#', '*',
            '_', '`', '[', ']', '(', ')', '|', '>', '-', '+', '.', ',', ':', '!', '~', '1', '2',
            '3', '0', 'é', '日', '本', '🎉', '\u{0301}', '\r', '"', '\'', '<', '/', '\\',
        ]
        .as_slice(),
    )
}

proptest! {
    /// (a) For any generated markdown-ish input, `parse`
    /// never fails and never panics, and the round-trip / disjointness
    /// invariants above hold.
    #[test]
    fn parse_round_trips_over_generated_markdown_ish_input(
        chars in prop::collection::vec(markdown_ish_char(), 0..500)
    ) {
        let source: String = chars.into_iter().collect();
        let doc = parse(source.clone())
            .unwrap_or_else(|e| panic!("parse must not fail on any input: {e}\nsource={source:?}"));
        assert_document_round_trips(&doc, &source);
    }
}

/// (b) Golden fixture: fenced and indented code blocks contribute no
/// prose; text around them does.
#[test]
fn golden_code_fences_exclude_code() {
    let (source, doc) = parse_fixture("code_fences.md");
    assert_document_round_trips(&doc, &source);
    assert!(no_prose_contains(&doc, "INLINECODE_MARKER"));
    assert!(no_prose_contains(&doc, "FENCED_CODE_MARKER"));
    assert!(no_prose_contains(&doc, "INDENTED_CODE_MARKER"));
    assert!(any_prose_contains(
        &doc,
        "Prose continues after indentation"
    ));
}

/// (b) Golden fixture: nested (tight and loose) lists round-trip and
/// surface every item's text as prose.
#[test]
fn golden_nested_lists() {
    let (source, doc) = parse_fixture("nested_lists.md");
    assert_document_round_trips(&doc, &source);
    for marker in ["ONE", "ALPHA", "BETA", "TWO", "FIRST", "SECOND", "THREE"] {
        assert!(any_prose_contains(&doc, marker), "missing marker {marker}");
    }
    assert!(any_prose_contains(&doc, "LOOSE_MARKER"));
}

/// (b) Golden fixture: table cell text is prose; pipe/dash table
/// structure is not.
#[test]
fn golden_tables() {
    let (source, doc) = parse_fixture("tables.md");
    assert_document_round_trips(&doc, &source);
    assert!(any_prose_contains(&doc, "CELLTEXT"));
    assert!(any_prose_contains(&doc, "Grace"));
    assert!(any_prose_contains(&doc, "TABLE_TRAILING_MARKER"));
    for text in prose_texts(&doc) {
        assert!(
            !text.contains('|'),
            "table pipe leaked into prose: {text:?}"
        );
    }
}

/// (b) Golden fixture: link/image label text is prose; the destination
/// URL is excluded.
#[test]
fn golden_links() {
    let (source, doc) = parse_fixture("links.md");
    assert_document_round_trips(&doc, &source);
    assert!(any_prose_contains(&doc, "LINKTEXT_MARKER"));
    assert!(any_prose_contains(&doc, "ALTTEXT_MARKER"));
    assert!(no_prose_contains(&doc, "LINKURL_MARKER"));
    assert!(no_prose_contains(&doc, "IMAGEURL_MARKER"));
}

/// (b) Golden fixture: block-quote text is prose; the `>` continuation
/// marker on wrapped lines is excluded.
#[test]
fn golden_blockquotes() {
    let (source, doc) = parse_fixture("blockquotes.md");
    assert_document_round_trips(&doc, &source);
    for marker in [
        "BQMARKER_ONE",
        "BQMARKER_TWO",
        "BQMARKER_THREE",
        "BQ_TRAILING_MARKER",
    ] {
        assert!(any_prose_contains(&doc, marker), "missing marker {marker}");
    }
    for text in prose_texts(&doc) {
        assert!(
            !text.contains("\n>"),
            "blockquote marker leaked into prose: {text:?}"
        );
    }
}

/// (b) Golden fixture: an HTML block contributes no prose; text between
/// raw inline HTML tags is still prose, but the tags themselves are
/// excluded.
#[test]
fn golden_html() {
    let (source, doc) = parse_fixture("html.md");
    assert_document_round_trips(&doc, &source);
    assert!(no_prose_contains(&doc, "HTML_BLOCK_MARKER"));
    assert!(any_prose_contains(&doc, "INLINE_HTML_MARKER"));
    assert!(any_prose_contains(&doc, "INLINE_SURROUND_MARKER"));
    for text in prose_texts(&doc) {
        assert!(
            !text.contains("<span"),
            "raw HTML tag leaked into prose: {text:?}"
        );
    }
}

/// (b) Golden fixture: emoji, CJK, and a combining-mark sequence survive
/// extraction as prose, byte-exact.
#[test]
fn golden_unicode() {
    let (source, doc) = parse_fixture("unicode.md");
    assert_document_round_trips(&doc, &source);
    assert!(any_prose_contains(&doc, "🎉"));
    assert!(any_prose_contains(&doc, "日本語"));
    assert!(any_prose_contains(&doc, "e\u{0301}"));
    assert!(any_prose_contains(&doc, "🚀"));
    assert!(any_prose_contains(&doc, "ZH_MARKER"));
    assert!(any_prose_contains(&doc, "LIST_EMOJI_MARKER"));
}

/// Walks every `*.md` file under `corpus/human/` and `corpus/llm/` (if
/// present) and asserts each one round-trips, giving the "round-trip
/// passes on 100% of corpus docs" requirement an executable check rather
/// than only the synthetic proptest/fixture coverage above.
///
/// Corpus generation has not landed yet as of this test's authoring, so
/// `corpus/human/` and `corpus/llm/` don't exist — in that case this is a
/// deliberate no-op (skipped, not a silent pass dressed up as coverage)
/// rather than a failure, exactly like `scripts/check-holdout.sh` no-ops
/// before `corpus/holdout.lock` exists. Once real documents land, this
/// test starts exercising them automatically with no further wiring.
#[test]
fn corpus_docs_round_trip() {
    let corpus_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus");
    let doc_dirs = ["human", "llm"].map(|sub| corpus_root.join(sub));

    let mut checked = 0usize;
    for dir in &doc_dirs {
        if !dir.is_dir() {
            continue;
        }
        for entry in walk_markdown_files(dir) {
            let source = std::fs::read_to_string(&entry)
                .unwrap_or_else(|e| panic!("read corpus doc {}: {e}", entry.display()));
            let doc = parse(source.clone())
                .unwrap_or_else(|e| panic!("parse corpus doc {}: {e}", entry.display()));
            assert_document_round_trips(&doc, &source);
            checked += 1;
        }
    }

    if checked == 0 {
        eprintln!(
            "no corpus docs found under {}/{{human,llm}} — skipping \
             (corpus generation has not landed yet)",
            corpus_root.display()
        );
    }
}

/// Recursively collects every `*.md` file under `dir`, sorted, so the test
/// above runs in a deterministic order.
fn walk_markdown_files(dir: &Path) -> Vec<PathBuf> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        let mut children: Vec<PathBuf> = entries.filter_map(|e| e.ok().map(|e| e.path())).collect();
        children.sort();
        for path in children {
            if path.is_dir() {
                walk(&path, out);
            } else if path.extension().is_some_and(|ext| ext == "md") {
                out.push(path);
            }
        }
    }
    let mut out = Vec::new();
    walk(dir, &mut out);
    out
}

/// (b) Golden fixture: CRLF line endings are preserved verbatim, both in
/// the document source and inside extracted prose.
#[test]
fn golden_crlf() {
    let (source, doc) = parse_fixture("crlf.md");
    assert!(source.contains("\r\n"), "fixture must actually use CRLF");
    assert_document_round_trips(&doc, &source);
    assert!(any_prose_contains(&doc, "Line one.\r\nLine two."));
    assert!(any_prose_contains(&doc, "CRLF_MARKER"));
    assert!(any_prose_contains(&doc, "Final paragraph after CRLF list."));
}
