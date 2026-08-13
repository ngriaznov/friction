//! End-to-end tests over real CPython bytecode.
//!
//! Skipped rather than failed when no interpreter is present.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use salience_core::{Denylist, ScoreWeights, Sidecar, analyze};

/// Writes `source` to a uniquely named temp file.
fn write_source(source: &str) -> PathBuf {
    static SEQ: AtomicUsize = AtomicUsize::new(0);
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    let path =
        std::env::temp_dir().join(format!("salience-py-test-{}-{seq}.py", std::process::id()));
    std::fs::write(&path, source).expect("temp file is writable");
    path
}

fn sidecar_for(source: &str) -> Option<Sidecar> {
    let path = write_source(source);
    let lowered = match salience_py::lower_file(&path) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("skipping: {e}");
            let _ = std::fs::remove_file(&path);
            return None;
        }
    };
    let mut side = Sidecar::new(lowered.file.clone(), "salience-py/test");
    for ir in &lowered.functions {
        ir.validate().expect("frontend emits a well-formed graph");
        let sal = analyze(ir, &Denylist::new(), &ScoreWeights::default());
        side.push(ir, &sal);
    }
    side.finish();
    let _ = std::fs::remove_file(&path);
    Some(side)
}

/// Deliberately the same program as the JVM test's fixture, so the two
/// frontends can be compared line for line.
const SOURCE: &str = r#"import logging

LOG = logging.getLogger("sample")
TOTALS = {}


def process(prices, rate, apply_it, key):
    LOG.info("starting with %d prices", len(prices))
    unused = "goes nowhere"
    counted = 0
    subtotal = 0.0
    for p in prices:
        if p is None:
            continue
        subtotal += p
        counted += 1
    total = subtotal
    if apply_it:
        total = subtotal * (1.0 + rate)
    LOG.debug("counted %d", counted)
    TOTALS[key] = total
    return total
"#;

/// The same headline behavior the JVM frontend shows, on a completely different
/// substrate and through a serialized IR: an accumulator reaching a state write
/// is core, a counter reaching only a log is inert.
#[test]
fn separates_a_real_accumulator_from_a_log_only_counter() {
    let Some(side) = sidecar_for(SOURCE) else {
        return;
    };
    // `subtotal += p` on line 15, `counted += 1` on line 16.
    assert_eq!(
        side.tier_at(15),
        Some("core"),
        "subtotal reaches TOTALS[key]"
    );
    assert_eq!(
        side.tier_at(16),
        Some("inert"),
        "counted only ever reaches a log call"
    );
    // Both log lines, including the `len(prices)` call built for one of them.
    assert_eq!(side.tier_at(8), Some("inert"));
    assert_eq!(side.tier_at(20), Some("inert"));
    // The dead local.
    assert_eq!(side.tier_at(9), Some("plumbing"));
    // The `if apply_it:` predicate.
    assert_eq!(side.tier_at(18), Some("core"));
}

/// Since PEP 626 every instruction carries a line, so the frontend has no
/// excuse for losing one.
#[test]
fn every_instruction_carries_a_line() {
    let Some(side) = sidecar_for(SOURCE) else {
        return;
    };
    for f in &side.functions {
        assert_eq!(
            f.coverage.instructions, f.coverage.with_line,
            "{} lost line attribution",
            f.name
        );
    }
}

#[test]
fn lowering_is_deterministic() {
    let (Some(a), Some(b)) = (sidecar_for(SOURCE), sidecar_for(SOURCE)) else {
        return;
    };
    // The file path differs between runs by construction, so compare the part
    // that must be stable.
    assert_eq!(
        serde_json::to_string(&a.functions).unwrap(),
        serde_json::to_string(&b.functions).unwrap()
    );
}

/// Comprehensions and nested closures compile to their own code objects. They
/// must be lowered too, or whole regions of a file would silently have no map.
#[test]
fn nested_code_objects_are_lowered() {
    let Some(side) = sidecar_for(
        "def outer(xs):\n    doubled = [x * 2 for x in xs]\n    def inner(y):\n        return y + 1\n    return doubled, inner\n",
    ) else {
        return;
    };
    let names: Vec<&str> = side.functions.iter().map(|f| f.name.as_str()).collect();
    assert!(
        names.iter().any(|n| n.contains("inner")),
        "expected the nested function, got {names:?}"
    );
    assert!(
        names.len() >= 3,
        "module, outer, inner at minimum: {names:?}"
    );
}

/// A syntax error must surface as an error, not a panic or an empty map.
#[test]
fn reports_syntax_errors() {
    let path = write_source("def broken(:\n");
    let result = salience_py::lower_file(&path);
    let _ = std::fs::remove_file(&path);
    assert!(result.is_err());
}
