//! End-to-end tests over real bytecode, compiled by `javac` at test time.
//!
//! Skipped rather than failed when no JDK is present, so the suite stays green
//! on machines that only build Rust.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

use salience_core::{Denylist, ScoreWeights, Sidecar, analyze};

/// Compiles `source` with debug info and returns the directory holding the
/// resulting classes, or `None` when `javac` is unavailable.
fn compile(name: &str, source: &str) -> Option<PathBuf> {
    // Tests run in parallel in one process, so the pid alone does not make the
    // directory unique — a counter is what keeps one test from deleting the
    // classes another is still reading.
    static SEQ: AtomicUsize = AtomicUsize::new(0);
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "salience-jvm-test-{name}-{}-{seq}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).ok()?;
    let java = dir.join(format!("{name}.java"));
    std::fs::write(&java, source).ok()?;

    // `-g` is what emits the LineNumberTable; without it the analysis still
    // runs but projects onto nothing, which is precisely what one test checks.
    let out = Command::new("javac").arg("-g").arg(&java).output().ok()?;
    if !out.status.success() {
        eprintln!("javac failed: {}", String::from_utf8_lossy(&out.stderr));
        return None;
    }
    Some(dir)
}

fn sidecar_for(dir: &Path, class: &str) -> Sidecar {
    let bytes = std::fs::read(dir.join(format!("{class}.class"))).expect("class file exists");
    let lowered = salience_jvm::lower_class(&bytes).expect("class parses");
    let mut side = Sidecar::new(
        lowered.source_file.clone().unwrap_or_default(),
        "salience-jvm/test",
    );
    for ir in &lowered.functions {
        ir.validate().expect("frontend emits a well-formed graph");
        let sal = analyze(ir, &Denylist::new(), &ScoreWeights::default());
        side.push(ir, &sal);
    }
    side.finish();
    side
}

const SOURCE: &str = r#"
import java.util.List;
import java.util.logging.Logger;

public class Sample {
    private static final Logger LOG = Logger.getLogger("sample");
    private double stored;

    public double process(List<Double> prices, double rate, boolean apply) {
        LOG.info("starting with " + prices.size() + " prices");
        String unused = "goes nowhere";
        int counted = 0;
        double subtotal = 0.0;
        for (Double p : prices) {
            if (p == null) {
                continue;
            }
            subtotal += p;
            counted++;
        }
        double total = subtotal;
        if (apply) {
            total = subtotal * (1.0 + rate);
        }
        LOG.fine("counted " + counted);
        this.stored = total;
        return total;
    }
}
"#;

/// The headline behavior, on real bytecode: an accumulator that reaches a state
/// write is core, while a counter that only ever reaches a log call is inert —
/// even though the two are syntactically identical and both loop-carried.
#[test]
fn separates_a_real_accumulator_from_a_log_only_counter() {
    let Some(dir) = compile("Sample", SOURCE) else {
        eprintln!("skipping: javac unavailable");
        return;
    };
    let side = sidecar_for(&dir, "Sample");

    // The two loop bodies: `subtotal += p` and `counted++`, adjacent lines,
    // syntactically identical, both loop-carried — and correctly separated.
    assert_eq!(
        side.tier_at(18),
        Some("core"),
        "subtotal reaches the field write"
    );
    assert_eq!(
        side.tier_at(19),
        Some("inert"),
        "counted only ever reaches a log call"
    );
    // Both log lines, including the `prices.size()` call built as an argument
    // for one of them.
    assert_eq!(side.tier_at(10), Some("inert"));
    assert_eq!(side.tier_at(25), Some("inert"));
    // The dead local.
    assert_eq!(side.tier_at(11), Some("plumbing"));
    // The `if (apply)` predicate.
    assert_eq!(side.tier_at(22), Some("core"));
    // The field write and the return: the frontier.
    assert_eq!(side.tier_at(26), Some("boundary"));
    assert_eq!(side.tier_at(27), Some("boundary"));

    let _ = std::fs::remove_dir_all(&dir);
}

/// Every instruction in a `-g` compiled body must carry a source line. If this
/// regresses, the line table is being read wrongly and every span is suspect.
#[test]
fn debug_compiled_bytecode_has_full_line_coverage() {
    let Some(dir) = compile("Sample", SOURCE) else {
        eprintln!("skipping: javac unavailable");
        return;
    };
    let side = sidecar_for(&dir, "Sample");
    for f in &side.functions {
        assert_eq!(
            f.coverage.instructions, f.coverage.with_line,
            "{} lost line attribution",
            f.name
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// Two runs over the same bytes must produce identical JSON, or the artifact
/// cannot be cached or diffed.
#[test]
fn lowering_is_deterministic() {
    let Some(dir) = compile("Sample", SOURCE) else {
        eprintln!("skipping: javac unavailable");
        return;
    };
    let a = serde_json::to_string(&sidecar_for(&dir, "Sample")).unwrap();
    let b = serde_json::to_string(&sidecar_for(&dir, "Sample")).unwrap();
    assert_eq!(a, b);
    let _ = std::fs::remove_dir_all(&dir);
}

/// Garbage in must not panic.
#[test]
fn rejects_non_class_bytes() {
    assert!(salience_jvm::lower_class(b"not a class file at all").is_err());
    assert!(salience_jvm::lower_class(&[]).is_err());
}
