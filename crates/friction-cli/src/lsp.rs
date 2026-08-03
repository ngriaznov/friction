//! `friction lsp` — the resident language server.
//!
//! One process, LSP over stdio, serving any editor (Zed and Neovim
//! natively, VS Code through a thin client). The packs and models are
//! the embedded ones every other subcommand uses; keeping the process
//! resident amortizes their first-touch parses to zero, so a lint is
//! pure scan time.
//!
//! # What a lint publishes
//!
//! Three diagnostic tiers, mirroring the CLI's own honesty split:
//!
//! - **Warning** — an edit the repair engine would apply (`fix`'s own
//!   patches). Each carries its replacement in the diagnostic's `data`,
//!   so the matching quick-fix is offered only where a licensed rewrite
//!   exists.
//! - **Hint** — a gate-held candidate (`fix --suggest`'s findings): a
//!   detected tell with no safe mechanical rewrite, explained.
//! - **Information** — a detection-only span (`check`'s DMS, jargon,
//!   overuse, and contrast-frame channels): machine register that needs
//!   a human hand.
//!
//! # Coordinates
//!
//! The repair pipeline is multi-pass and each pass's ranges index its
//! own input text, so only a prefix of the report maps onto the live
//! buffer: pass entries are included while every earlier pass applied
//! zero patches. An already-clean buffer therefore gets the full report
//! (including the register pass), and a dirty one shows its first
//! pass's edits — applying them re-lints, and later-pass edits surface
//! on the next round. Convergence is the editor loop itself.
//!
//! Positions are converted byte-offset → UTF-16 through [`LineIndex`],
//! the LSP default encoding; multi-byte and astral-plane text is pinned
//! by this module's tests.

use std::collections::HashMap;
use std::ops::Range;
use std::process::ExitCode;

use clap::Args as ClapArgs;
use lsp_server::{Connection, Message, Notification, Request, RequestId, Response};
use lsp_types::notification::{
    DidChangeTextDocument, DidCloseTextDocument, DidOpenTextDocument, DidSaveTextDocument,
    Notification as _, PublishDiagnostics,
};
use lsp_types::request::{CodeActionRequest, Request as _};
use lsp_types::{
    CodeAction, CodeActionKind, CodeActionOrCommand, CodeActionParams,
    CodeActionProviderCapability, Diagnostic, DiagnosticSeverity, NumberOrString, Position,
    PublishDiagnosticsParams, TextDocumentSyncCapability, TextDocumentSyncKind, TextEdit, Url,
    WorkspaceEdit,
};

use crate::common;
use crate::common::CliError;

/// Arguments for `friction lsp`.
#[derive(Debug, ClapArgs)]
pub struct LspArgs {}

/// Above this many bytes a buffer is linted on save only, never on
/// type: a multi-megabyte document costs visible fractions of a second
/// per scan, and per-keystroke latency is the one budget an editor
/// never forgives.
const ON_TYPE_LIMIT_BYTES: usize = 512 * 1024;

/// The diagnostic source label every published diagnostic carries.
const SOURCE: &str = "friction";

/// Runs the server over stdio until the client disconnects.
pub fn run(_args: &LspArgs) -> ExitCode {
    match run_inner() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("friction lsp: {err}");
            ExitCode::FAILURE
        }
    }
}

fn run_inner() -> Result<(), CliError> {
    // A language server lives for the editor session, so the engines
    // are deliberately leaked into `'static`: `MatchEngine` borrows the
    // tagger and segmenter it scans with, and a process-lifetime borrow
    // is the honest expression of that.
    let cli_engine: &'static common::Engine = Box::leak(Box::new(common::Engine::load()?));
    let edit_engine: &'static friction_edit::Engine =
        Box::leak(Box::new(friction_edit::Engine::new()?));
    let match_engine: &'static friction_match::MatchEngine<'static> =
        Box::leak(Box::new(friction_match::MatchEngine::new(
            &friction_packs::INVENTORY.pack,
            &friction_packs::DMS.pack,
            &friction_packs::JARGON.pack,
            &friction_packs::JARGON_ATTEST,
            &friction_packs::HUMAN_EVIDENCE,
            friction_packs::ModelFamily::Qwen, // label-only; the scan is pooled
            &cli_engine.tagger,
            &cli_engine.segmenter,
        )?));

    let (connection, io_threads) = Connection::stdio();
    let capabilities = serde_json::json!({
        "textDocumentSync": TextDocumentSyncCapability::Kind(TextDocumentSyncKind::FULL),
        "codeActionProvider": CodeActionProviderCapability::Simple(true),
    });
    let _initialize_params =
        connection
            .initialize(capabilities)
            .map_err(|e| CliError::ReadInput {
                path: "<lsp stdio>".to_string(),
                source: std::io::Error::other(e.to_string()),
            })?;

    main_loop(&connection, edit_engine, match_engine);
    drop(connection);
    let _ = io_threads.join();
    Ok(())
}

/// Server state: the open documents by URI.
struct Docs {
    open: HashMap<Url, String>,
}

fn main_loop(
    connection: &Connection,
    edit_engine: &friction_edit::Engine,
    match_engine: &friction_match::MatchEngine<'_>,
) {
    let mut docs = Docs {
        open: HashMap::new(),
    };
    for msg in &connection.receiver {
        match msg {
            Message::Request(req) => {
                if connection.handle_shutdown(&req).unwrap_or(true) {
                    return;
                }
                if let Some(resp) = handle_request(&req, &docs, edit_engine) {
                    let _ = connection.sender.send(Message::Response(resp));
                }
            }
            Message::Notification(note) => {
                handle_notification(note, &mut docs, connection, edit_engine, match_engine);
            }
            Message::Response(_) => {}
        }
    }
}

fn handle_notification(
    note: Notification,
    docs: &mut Docs,
    connection: &Connection,
    edit_engine: &friction_edit::Engine,
    match_engine: &friction_match::MatchEngine<'_>,
) {
    let mut lint_target: Option<(Url, bool)> = None;
    match note.method.as_str() {
        DidOpenTextDocument::METHOD => {
            if let Ok(p) =
                serde_json::from_value::<lsp_types::DidOpenTextDocumentParams>(note.params)
            {
                docs.open
                    .insert(p.text_document.uri.clone(), p.text_document.text);
                lint_target = Some((p.text_document.uri, true));
            }
        }
        DidChangeTextDocument::METHOD => {
            if let Ok(p) =
                serde_json::from_value::<lsp_types::DidChangeTextDocumentParams>(note.params)
            {
                // Full sync: the last change carries the whole text.
                if let Some(change) = p.content_changes.into_iter().last() {
                    docs.open.insert(p.text_document.uri.clone(), change.text);
                    lint_target = Some((p.text_document.uri, false));
                }
            }
        }
        DidSaveTextDocument::METHOD => {
            if let Ok(p) =
                serde_json::from_value::<lsp_types::DidSaveTextDocumentParams>(note.params)
            {
                lint_target = Some((p.text_document.uri, true));
            }
        }
        DidCloseTextDocument::METHOD => {
            if let Ok(p) =
                serde_json::from_value::<lsp_types::DidCloseTextDocumentParams>(note.params)
            {
                docs.open.remove(&p.text_document.uri);
                publish(connection, p.text_document.uri, Vec::new());
            }
        }
        _ => {}
    }
    if let Some((uri, saved)) = lint_target
        && let Some(text) = docs.open.get(&uri)
    {
        if !saved && text.len() > ON_TYPE_LIMIT_BYTES {
            return; // large buffer: lint on save only.
        }
        let diagnostics = lint(text, &uri, edit_engine, match_engine);
        publish(connection, uri, diagnostics);
    }
}

fn publish(connection: &Connection, uri: Url, diagnostics: Vec<Diagnostic>) {
    let params = PublishDiagnosticsParams {
        uri,
        diagnostics,
        version: None,
    };
    let _ = connection.sender.send(Message::Notification(Notification {
        method: PublishDiagnostics::METHOD.to_string(),
        params: serde_json::to_value(params).unwrap_or_default(),
    }));
}

fn handle_request(
    req: &Request,
    docs: &Docs,
    edit_engine: &friction_edit::Engine,
) -> Option<Response> {
    if req.method == CodeActionRequest::METHOD {
        let params: CodeActionParams = serde_json::from_value(req.params.clone()).ok()?;
        let actions = code_actions(&params, docs, edit_engine);
        return Some(ok_response(
            req.id.clone(),
            serde_json::to_value(actions).unwrap_or_default(),
        ));
    }
    None
}

const fn ok_response(id: RequestId, result: serde_json::Value) -> Response {
    Response {
        id,
        result: Some(result),
        error: None,
    }
}

/// The quick-fix for each in-range diagnostic that carries a
/// replacement, plus one whole-document "fix everything" source action
/// whenever the repair engine would change the buffer at all.
fn code_actions(
    params: &CodeActionParams,
    docs: &Docs,
    edit_engine: &friction_edit::Engine,
) -> Vec<CodeActionOrCommand> {
    let uri = &params.text_document.uri;
    let mut actions = Vec::new();
    for diagnostic in &params.context.diagnostics {
        let Some(replacement) = diagnostic
            .data
            .as_ref()
            .and_then(|d| d.get("replacement"))
            .and_then(|r| r.as_str())
        else {
            continue;
        };
        let title = if replacement.trim().is_empty() {
            format!("friction: delete ({})", code_text(diagnostic))
        } else {
            format!("friction: rewrite to \u{201c}{replacement}\u{201d}")
        };
        let edit = TextEdit {
            range: diagnostic.range,
            new_text: replacement.to_string(),
        };
        actions.push(CodeActionOrCommand::CodeAction(CodeAction {
            title,
            kind: Some(CodeActionKind::QUICKFIX),
            diagnostics: Some(vec![diagnostic.clone()]),
            edit: Some(WorkspaceEdit {
                changes: Some([(uri.clone(), vec![edit])].into()),
                ..WorkspaceEdit::default()
            }),
            ..CodeAction::default()
        }));
    }

    if let Some(text) = docs.open.get(uri)
        && let Ok((fixed, _)) = edit_engine.fix_document_with(text, syntax_of(uri, text))
        && fixed != *text
    {
        let index = LineIndex::new(text);
        let whole = lsp_types::Range {
            start: Position::new(0, 0),
            end: index.position(text.len()),
        };
        actions.push(CodeActionOrCommand::CodeAction(CodeAction {
            title: "friction: fix document".to_string(),
            kind: Some(CodeActionKind::SOURCE_FIX_ALL),
            edit: Some(WorkspaceEdit {
                changes: Some(
                    [(
                        uri.clone(),
                        vec![TextEdit {
                            range: whole,
                            new_text: fixed,
                        }],
                    )]
                    .into(),
                ),
                ..WorkspaceEdit::default()
            }),
            ..CodeAction::default()
        }));
    }
    actions
}

fn code_text(diagnostic: &Diagnostic) -> String {
    match &diagnostic.code {
        Some(NumberOrString::String(s)) => s.clone(),
        Some(NumberOrString::Number(n)) => n.to_string(),
        None => String::new(),
    }
}

/// Chooses the parse syntax the same way the CLI does for a path.
#[expect(
    clippy::case_sensitive_file_extension_comparisons,
    reason = "the path is lowercased on the line above the comparisons"
)]
fn syntax_of(uri: &Url, source: &str) -> friction_parse::Syntax {
    let path = uri.path().to_ascii_lowercase();
    if path.ends_with(".html") || path.ends_with(".htm") {
        friction_parse::Syntax::Html
    } else if path.ends_with(".md") || path.ends_with(".markdown") {
        friction_parse::Syntax::Markdown
    } else {
        // Plain text still parses as Markdown: CommonMark's plain
        // paragraphs are plain text, and the block scoping (skip code
        // fences and the like) only ever protects.
        let _ = source;
        friction_parse::Syntax::Markdown
    }
}

/// Runs the repair plan and the detection scan over `text` and renders
/// both as diagnostics in buffer coordinates.
fn lint(
    text: &str,
    uri: &Url,
    edit_engine: &friction_edit::Engine,
    match_engine: &friction_match::MatchEngine<'_>,
) -> Vec<Diagnostic> {
    let syntax = syntax_of(uri, text);
    let index = LineIndex::new(text);
    let mut out = Vec::new();

    if let Ok((_, report)) = edit_engine.fix_document_with(text, syntax) {
        // Only the report prefix whose input text IS the buffer maps
        // onto it — see the module docs' coordinate rule.
        let mut buffer_valid = true;
        for pass in &report.passes {
            if !buffer_valid {
                break;
            }
            for patch in &pass.applied_patches {
                out.push(patch_diagnostic(patch, &index));
            }
            for finding in &pass.held {
                out.push(finding_diagnostic(
                    finding.rule.as_str(),
                    finding.range.clone(),
                    &finding.message,
                    DiagnosticSeverity::HINT,
                    &index,
                ));
            }
            if pass.patches_applied > 0 {
                buffer_valid = false;
            }
        }
    }

    if let Ok(document) = friction_parse::parse_with(text.to_string(), syntax)
        && let Ok(report) = match_engine.scan(&document)
    {
        for span in &report.spans {
            let message = span.message.as_deref().map_or_else(
                || format!("machine register ({})", span.frame_id),
                String::from,
            );
            out.push(finding_diagnostic(
                &span.frame_id,
                span.range.clone(),
                &message,
                DiagnosticSeverity::INFORMATION,
                &index,
            ));
        }
    }

    out.sort_by_key(|d| (d.range.start.line, d.range.start.character));
    out
}

fn patch_diagnostic(patch: &friction_core::Patch, index: &LineIndex) -> Diagnostic {
    let message = if patch.replacement.trim().is_empty() {
        "machine register; friction would delete this".to_string()
    } else {
        format!(
            "machine register; friction would rewrite to \u{201c}{}\u{201d}",
            patch.replacement.trim()
        )
    };
    Diagnostic {
        range: index.range(&patch.range),
        severity: Some(DiagnosticSeverity::WARNING),
        code: Some(NumberOrString::String(patch.rule.as_str().to_string())),
        source: Some(SOURCE.to_string()),
        message,
        data: Some(serde_json::json!({ "replacement": patch.replacement })),
        ..Diagnostic::default()
    }
}

fn finding_diagnostic(
    code: &str,
    range: Range<usize>,
    message: &str,
    severity: DiagnosticSeverity,
    index: &LineIndex,
) -> Diagnostic {
    Diagnostic {
        range: index.range(&range),
        severity: Some(severity),
        code: Some(NumberOrString::String(code.to_string())),
        source: Some(SOURCE.to_string()),
        message: message.to_string(),
        ..Diagnostic::default()
    }
}

/// Byte-offset → LSP `Position` (UTF-16 code units, the protocol
/// default) conversion over one immutable text snapshot.
struct LineIndex {
    /// Byte offset where each line starts; `line_starts[0] == 0`.
    line_starts: Vec<usize>,
    len: usize,
    text: String,
}

impl LineIndex {
    fn new(text: &str) -> Self {
        let mut line_starts = vec![0];
        for (i, byte) in text.bytes().enumerate() {
            if byte == b'\n' {
                line_starts.push(i + 1);
            }
        }
        Self {
            line_starts,
            len: text.len(),
            text: text.to_string(),
        }
    }

    fn position(&self, offset: usize) -> Position {
        let offset = offset.min(self.len);
        let line = self
            .line_starts
            .partition_point(|&start| start <= offset)
            .saturating_sub(1);
        let line_start = self.line_starts[line];
        let column: usize = self.text[line_start..offset]
            .chars()
            .map(char::len_utf16)
            .sum();
        #[expect(clippy::cast_possible_truncation, reason = "lines fit u32")]
        Position::new(line as u32, column as u32)
    }

    fn range(&self, range: &Range<usize>) -> lsp_types::Range {
        lsp_types::Range {
            start: self.position(range.start),
            end: self.position(range.end),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_index_converts_multibyte_offsets_to_utf16_positions() {
        let text = "héllo\nwörld 🦀 end\n";
        let index = LineIndex::new(text);
        assert_eq!(index.position(0), Position::new(0, 0));
        // 'é' is 2 bytes, 1 UTF-16 unit: byte 3 is the first 'l'.
        assert_eq!(index.position(3), Position::new(0, 2));
        let line2 = text.find("wörld").expect("present");
        assert_eq!(index.position(line2), Position::new(1, 0));
        // '🦀' is 4 bytes and 2 UTF-16 units; "end" starts after it.
        let end = text.find("end").expect("present");
        assert_eq!(index.position(end), Position::new(1, 9));
        // Past-the-end clamps to the final position.
        assert_eq!(index.position(text.len() + 10), index.position(text.len()));
    }
}
