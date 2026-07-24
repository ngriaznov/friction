//! Reports (does not assert) `Engine::fix_document`'s throughput on a
//! small fixture-sized input and a mid-size corpus document, warming the
//! tagger/pack loads outside the timed region.
//!
//! Run with `cargo run --release --example throughput_bench -p friction-edit`.

use std::time::Instant;

const SMALL: &str = "This guide will walk you through configuring the backup agent for your \
    staging environment. It is important to note that the agent performs validation of the \
    configuration file before each run. Once validation succeeds, the agent conducts an \
    analysis of the snapshot catalog and simply uploads any missing segments. By following \
    these steps, you can quickly and easily verify that your backups are consistent. If you \
    have any questions or require further assistance, please reach out to our support team.";

fn report(label: &str, engine: &friction_edit::Engine, input: &str) {
    // Warm-up run (first call may pay one-time lazy-init costs inside the
    // tagger/regex engines) outside the timed region.
    let _ = engine.fix_document(input);

    let start = Instant::now();
    let (_out, _report) = engine
        .fix_document(input)
        .expect("fix_document must succeed");
    let elapsed = start.elapsed();

    let bytes = input.len();
    let ms = elapsed.as_secs_f64() * 1000.0;
    #[allow(clippy::cast_precision_loss)]
    let bytes_f64 = bytes as f64;
    let bytes_per_ms = if ms > 0.0 {
        bytes_f64 / ms
    } else {
        f64::INFINITY
    };
    let ms_per_kb = if bytes > 0 {
        ms / (bytes_f64 / 1024.0)
    } else {
        0.0
    };
    println!(
        "{label}: {bytes} bytes in {ms:.3} ms  ({bytes_per_ms:.1} bytes/ms, {ms_per_kb:.3} ms/KB)"
    );
}

fn main() {
    let engine = friction_edit::Engine::new().expect("engine must load");

    report("composed_four_operation_paragraph (~700 B)", &engine, SMALL);

    // A mid-size document: repeat the small fixture to approximate a
    // 5-10 KB document without depending on a specific corpus file's
    // exact bytes (kept dependency-free here; corpus-backed sizing is
    // exercised by the crate's own idempotence test instead).
    let mid_size = SMALL.repeat(10);
    report("synthetic mid-size document (~7 KB)", &engine, &mid_size);

    println!(
        "target: sub-10 ms/KB end to end (ALGORITHMS.md section 5). Numbers above are measured, \
         not asserted — see this file's own doc comment."
    );
}
