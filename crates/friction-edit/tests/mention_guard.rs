//! Mention-vs-use guards: text inside double quotation marks is someone's
//! example or citation, and a numbered list item's prose is load-bearing
//! structure — the engine must hold (report) rather than edit either.
//!
//! Motivating failure: running the engine over documentation *about*
//! machine-register phrases rewrote quoted phrases themselves
//! ("performs validation of" -> "validates" inside a quote) and deleted
//! a quoted ritual-closer example, leaving a bare `1.` list marker.

use friction_edit::Engine;

fn engine() -> Engine {
    Engine::new().expect("embedded packs load")
}

/// Quoted machine-register phrases pass through byte-identical: the
/// substitution and deletion frames match, but every match is inside
/// straight double quotes, so each is held as a suggestion instead.
#[test]
fn quoted_substitution_and_deletion_mentions_are_left_verbatim() {
    let source = "The style guide bans \"will walk you through\" and \"in order to\" outright. \
                  It also flags \"quickly and easily\" as filler.\n";
    let (fixed, report) = engine().fix_document(source).expect("engine runs");
    assert_eq!(fixed, source, "quoted mentions must not be rewritten");
    assert!(
        report
            .passes
            .iter()
            .flat_map(|p| &p.held)
            .any(|f| f.message.contains("inside quotation")),
        "the quoted matches must surface as held findings"
    );
}

/// A quoted light-verb construction is not pivoted; the same construction
/// unquoted in the same document still is.
#[test]
fn quoted_pivot_mention_is_left_verbatim_while_unquoted_use_is_pivoted() {
    let source = "The paper cites \"the tool performs validation of the manifest\" as its \
                  example. The loader performs validation of the schema.\n";
    let (fixed, _report) = engine().fix_document(source).expect("engine runs");
    assert!(
        fixed.contains("\"the tool performs validation of the manifest\""),
        "the quoted construction must survive verbatim, got: {fixed}"
    );
    assert!(
        fixed.contains("The loader validates the schema."),
        "the unquoted construction must still pivot, got: {fixed}"
    );
}

/// A quoted ritual closer is a mention: the sentence containing it is not
/// deleted. Curly quotes count as quotes too.
#[test]
fn quoted_ritual_closer_is_not_deleted() {
    for (open, close) in [("\"", "\""), ("\u{201c}", "\u{201d}")] {
        let source = format!(
            "Closers like {open}If you have any questions, please reach out to our \
             team.{close} score highest. The detector flags them first.\n"
        );
        let (fixed, report) = engine().fix_document(&source).expect("engine runs");
        assert_eq!(
            fixed, source,
            "a quoted ritual mention must not delete its sentence"
        );
        assert!(
            report
                .passes
                .iter()
                .flat_map(|p| &p.held)
                .any(|f| f.message.contains("inside quotation")),
            "the quoted ritual must surface as a held finding"
        );
    }
}

/// A numbered list item whose only sentence matches a ritual frame keeps
/// its text — deleting it would leave a bare `1.` marker and break the
/// counted enumeration. The same sentence in a plain paragraph is still
/// deleted.
#[test]
fn ritual_deletion_never_empties_a_numbered_list_item() {
    let listed = "Steps:\n\n1. If you have any questions, please reach out to support.\n2. \
                  Restart the service.\n";
    let (fixed, report) = engine().fix_document(listed).expect("engine runs");
    assert_eq!(
        fixed, listed,
        "a numbered item's only sentence must not be deleted"
    );
    assert!(
        report
            .passes
            .iter()
            .flat_map(|p| &p.held)
            .any(|f| f.message.contains("numbered list item")),
        "the held ritual must be reported"
    );

    let paragraph = "The setup is done.\n\nIf you have any questions, please reach out to \
                     support.\n";
    let (fixed, _) = engine().fix_document(paragraph).expect("engine runs");
    assert!(
        !fixed.contains("reach out"),
        "the same ritual sentence in a paragraph must still be deleted, got: {fixed}"
    );
}

/// An unordered list item is not covered by the enumeration hold: no
/// counted sequence breaks, so the reference's unconditional ritual
/// deletion still applies.
#[test]
fn ritual_deletion_still_fires_in_unordered_list_items() {
    let source = "Notes:\n\n- If you have any questions, please reach out to support.\n- \
                  Restart the service.\n";
    let (fixed, _) = engine().fix_document(source).expect("engine runs");
    assert!(
        !fixed.contains("reach out"),
        "an unordered item's ritual sentence is still deletable, got: {fixed}"
    );
}
