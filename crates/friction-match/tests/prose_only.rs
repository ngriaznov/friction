//! Regression coverage for the prose-only guarantee: headings, table
//! structure, and code never produce a span, while paragraphs, block
//! quotes, and list items do.

mod support;

use support::engine;

/// The exact regression case named in the build brief: the
/// `heading_pivot` reject-fixture text (`## ` markdown wrapping the
/// fixture's `input_block.text`) would license a derivational pivot if it
/// were prose — it must produce zero spans because it is a heading.
#[test]
fn heading_never_produces_a_span() {
    let source = "## Making the Decision\n";
    let document = friction_parse::parse(source).expect("valid markdown parses");
    let report = engine().scan(&document).expect("scan succeeds");
    assert!(
        report.spans.is_empty(),
        "expected no spans for a heading, got {:?}",
        report.spans
    );
}

#[test]
fn table_cell_never_produces_a_span() {
    let source = "\
| Step | Notes |
| --- | --- |
| One | prior to running, the tool performs an initialization of the database. |
";
    let document = friction_parse::parse(source).expect("valid markdown parses");
    let report = engine().scan(&document).expect("scan succeeds");
    assert!(
        report.spans.is_empty(),
        "expected no spans for a table cell, got {:?}",
        report.spans
    );
}

#[test]
fn code_block_never_produces_a_span() {
    let source = "\
```text
It is important to note that the tool performs an initialization of the database.
```
";
    let document = friction_parse::parse(source).expect("valid markdown parses");
    let report = engine().scan(&document).expect("scan succeeds");
    assert!(
        report.spans.is_empty(),
        "expected no spans for a fenced code block, got {:?}",
        report.spans
    );
}

#[test]
fn paragraph_and_blockquote_and_list_item_do_produce_spans() {
    let source = "\
Prior to release, please review the changelog.

> Prior to release, please review the changelog.

- Prior to release, please review the changelog.
";
    let document = friction_parse::parse(source).expect("valid markdown parses");
    let report = engine().scan(&document).expect("scan succeeds");
    assert_eq!(
        report.spans.len(),
        3,
        "expected one literal span per prose block, got {:?}",
        report.spans
    );
}
