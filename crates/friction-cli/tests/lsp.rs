//! End-to-end coverage for `friction lsp`: spawns the real binary,
//! speaks LSP over its stdio, and pins the diagnostic tiers and code
//! actions against a document with known tells.

use std::io::{BufRead as _, BufReader, Read as _, Write as _};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use serde_json::{Value, json};

struct Server {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl Server {
    fn spawn() -> Self {
        let mut child = Command::new(assert_cmd::cargo::cargo_bin("friction"))
            .arg("lsp")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .expect("friction lsp spawns");
        let stdin = child.stdin.take().expect("stdin piped");
        let stdout = BufReader::new(child.stdout.take().expect("stdout piped"));
        Self {
            child,
            stdin,
            stdout,
        }
    }

    fn send(&mut self, payload: &Value) {
        let body = payload.to_string();
        write!(self.stdin, "Content-Length: {}\r\n\r\n{body}", body.len())
            .expect("write to server");
        self.stdin.flush().expect("flush");
    }

    fn recv(&mut self) -> Value {
        let mut length = 0usize;
        loop {
            let mut line = String::new();
            self.stdout.read_line(&mut line).expect("header line");
            let line = line.trim_end();
            if line.is_empty() {
                break;
            }
            if let Some(rest) = line.strip_prefix("Content-Length: ") {
                length = rest.parse().expect("length parses");
            }
        }
        let mut body = vec![0u8; length];
        self.stdout.read_exact(&mut body).expect("body bytes");
        serde_json::from_slice(&body).expect("body is JSON")
    }
}

const DOC: &str = "It is important to note that we utilize the cache.\n";
const URI: &str = "file:///tmp/friction-lsp-test.md";

#[test]
fn lsp_publishes_tiered_diagnostics_and_offers_fixes() {
    let mut server = Server::spawn();

    server.send(&json!({
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"capabilities": {}, "processId": null, "rootUri": null}
    }));
    let init = server.recv();
    assert!(
        init["result"]["capabilities"]["codeActionProvider"]
            .as_bool()
            .unwrap_or(false),
        "code actions must be advertised: {init}"
    );
    server.send(&json!({"jsonrpc": "2.0", "method": "initialized", "params": {}}));

    server.send(&json!({
        "jsonrpc": "2.0", "method": "textDocument/didOpen",
        "params": {"textDocument": {
            "uri": URI, "languageId": "markdown", "version": 1, "text": DOC
        }}
    }));
    let published = server.recv();
    assert_eq!(
        published["method"], "textDocument/publishDiagnostics",
        "didOpen must publish: {published}"
    );
    let diagnostics = published["params"]["diagnostics"]
        .as_array()
        .expect("diagnostics array")
        .clone();

    // The known tells: a deletable filler span (warning, with its
    // replacement in `data`) and the utilize->use rewrite.
    let fixable: Vec<&Value> = diagnostics
        .iter()
        .filter(|d| d["data"]["replacement"].is_string())
        .collect();
    assert!(
        fixable.iter().any(|d| d["code"] == "span.delete"),
        "the filler span must be a fixable warning: {diagnostics:?}"
    );
    assert!(
        fixable
            .iter()
            .any(|d| d["code"] == "sub.apply" && d["data"]["replacement"] == "use"),
        "utilize->use must carry its replacement: {diagnostics:?}"
    );
    // Held/report-only findings surface as non-fixable tiers alongside.
    assert!(
        diagnostics
            .iter()
            .any(|d| d["data"]["replacement"].is_null()),
        "detection-only findings must be published too: {diagnostics:?}"
    );

    // A quick-fix for the fixable diagnostic, and the whole-document
    // fix-all, both with real edits.
    let target = fixable[0].clone();
    server.send(&json!({
        "jsonrpc": "2.0", "id": 2, "method": "textDocument/codeAction",
        "params": {
            "textDocument": {"uri": URI},
            "range": target["range"],
            "context": {"diagnostics": [target]}
        }
    }));
    let response = server.recv();
    let actions = response["result"].as_array().expect("actions array");
    assert!(
        actions.iter().any(|a| a["kind"] == "quickfix"),
        "a replacement-bearing diagnostic must get a quickfix: {actions:?}"
    );
    let fix_all = actions
        .iter()
        .find(|a| a["kind"] == "source.fixAll")
        .expect("fix-all action present");
    let edits = fix_all["edit"]["changes"][URI].as_array().expect("edits");
    assert_eq!(
        edits[0]["newText"], "We use the cache.\n",
        "fix-all must produce the fully repaired document"
    );

    server.send(&json!({"jsonrpc": "2.0", "id": 3, "method": "shutdown", "params": null}));
    let _ = server.recv();
    server.send(&json!({"jsonrpc": "2.0", "method": "exit", "params": null}));
    let status = server.child.wait().expect("server exits");
    assert!(status.success(), "clean shutdown");
}
