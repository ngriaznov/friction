//! End-to-end throughput benchmark: how many bytes of input prose
//! `FixEngine::fix_document` processes per second, single-threaded.
//!
//! Run in release mode (an unoptimized build is not representative):
//!
//! ```text
//! cargo run -p friction-apply --release --example throughput_bench
//! ```
//!
//! # Workload
//!
//! The same 20-document, 5-genre, deterministically-selected corpus sample
//! `friction-cli`'s own end-to-end snapshot suite pins (see that crate's
//! `tests/snapshot.rs` module docs for the selection rule this list is a
//! copy of) — a realistic mix of genres and both human- and LLM-authored
//! prose, rather than one repeated document.
//!
//! # Method
//!
//! [`FixEngine::new`] (embedded tagger model load) and every document's
//! file read happen once, up front, outside the timed region — this
//! benchmark measures `fix_document` itself, not process startup or I/O.
//! The full 20-document set is then run through `fix_document` for
//! [`REPS`] repetitions, wall-clock timed with [`std::time::Instant`], and
//! throughput is reported as total input bytes processed (`REPS` times the
//! set's total byte length) divided by total elapsed seconds. Running the
//! same fixed set repeatedly, rather than once, keeps the measured region
//! long enough that fixed per-call overhead (e.g. one-time allocator
//! warm-up) does not dominate a single short run.

use std::path::{Path, PathBuf};
use std::time::Instant;

use friction_apply::FixEngine;

/// One benchmark document: its genre (needed to look up envelope bands)
/// and its path relative to the workspace root.
struct Doc {
    genre: &'static str,
    relpath: &'static str,
}

/// The 20 documents `friction-cli`'s end-to-end snapshot suite selects —
/// see this file's module docs.
const DOCS: &[Doc] = &[
    Doc {
        genre: "blog",
        relpath: "corpus/human/blog/016b54b46d29feb8.md",
    },
    Doc {
        genre: "blog",
        relpath: "corpus/human/blog/0589bf2932eba95a.md",
    },
    Doc {
        genre: "blog",
        relpath: "corpus/llm/blog/152f7fa1159f4910.md",
    },
    Doc {
        genre: "blog",
        relpath: "corpus/llm/blog/19f12335d308d0e0.md",
    },
    Doc {
        genre: "docs",
        relpath: "corpus/human/docs/01ec8967989205a2.md",
    },
    Doc {
        genre: "docs",
        relpath: "corpus/human/docs/08d07d7b04ccd440.md",
    },
    Doc {
        genre: "docs",
        relpath: "corpus/llm/docs/00533d2e3a398154.md",
    },
    Doc {
        genre: "docs",
        relpath: "corpus/llm/docs/0a0197006e9ca159.md",
    },
    Doc {
        genre: "email",
        relpath: "corpus/quarantine/email/026f8f57c3920652.md",
    },
    Doc {
        genre: "email",
        relpath: "corpus/human/email/0ac47db2525fd485.md",
    },
    Doc {
        genre: "email",
        relpath: "corpus/llm/email/062a9d6f268e8994.md",
    },
    Doc {
        genre: "email",
        relpath: "corpus/llm/email/29ff22ba7de18cf6.md",
    },
    Doc {
        genre: "forum",
        relpath: "corpus/quarantine/forum/0710f627ec229d97.md",
    },
    Doc {
        genre: "forum",
        relpath: "corpus/quarantine/forum/0a3d4030ac673c91.md",
    },
    Doc {
        genre: "forum",
        relpath: "corpus/llm/forum/0720879ca70251ed.md",
    },
    Doc {
        genre: "forum",
        relpath: "corpus/llm/forum/137c3759df60fa49.md",
    },
    Doc {
        genre: "readme",
        relpath: "corpus/human/readme/0217ca71eb7abfce.md",
    },
    Doc {
        genre: "readme",
        relpath: "corpus/human/readme/05f0fbd8371252a5.md",
    },
    Doc {
        genre: "readme",
        relpath: "corpus/llm/readme/04c4c78e378a7c8f.md",
    },
    Doc {
        genre: "readme",
        relpath: "corpus/llm/readme/0859e6e20fb1cf84.md",
    },
];

/// How many times the full 20-document set is run through `fix_document`
/// in the timed region. Chosen so the timed region comfortably clears a
/// second of wall-clock time even on a fast machine, keeping fixed
/// per-call overhead from dominating the measurement.
const REPS: usize = 50;

/// The workspace root, resolved from this crate's own manifest directory
/// (`crates/friction-apply`) two levels up.
fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("the workspace root exists two levels up from CARGO_MANIFEST_DIR")
}

fn main() {
    let root = workspace_root();

    // Load every document's text once, up front — never part of the timed
    // region.
    let sources: Vec<(String, &'static str)> = DOCS
        .iter()
        .map(|doc| {
            let path = root.join(doc.relpath);
            let text = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("{}: failed to read: {e}", path.display()));
            (text, doc.genre)
        })
        .collect();
    let total_bytes_per_pass: usize = sources.iter().map(|(text, _)| text.len()).sum();

    // Model load: a one-time setup cost, never part of the timed region.
    let engine = FixEngine::new().expect("embedded tagger model must load");

    // One untimed warm-up pass: first-touch page faults, allocator growth,
    // and any lazily-initialized static data get paid for here rather than
    // skewing the timed passes that follow.
    for (source, genre) in &sources {
        let _ = engine
            .fix_document(source, genre)
            .expect("corpus fixture must fix cleanly");
    }

    let start = Instant::now();
    for _ in 0..REPS {
        for (source, genre) in &sources {
            let (output, _report) = engine
                .fix_document(source, genre)
                .expect("corpus fixture must fix cleanly");
            std::hint::black_box(&output);
        }
    }
    let elapsed = start.elapsed();

    #[allow(clippy::cast_precision_loss)]
    let total_bytes = (total_bytes_per_pass * REPS) as f64;
    let seconds = elapsed.as_secs_f64();
    let bytes_per_sec = total_bytes / seconds;

    println!("documents:            {}", DOCS.len());
    println!("bytes per pass:       {total_bytes_per_pass}");
    println!("repetitions:          {REPS}");
    println!("total bytes:          {total_bytes:.0}");
    println!("elapsed:              {elapsed:?}");
    println!(
        "throughput:           {bytes_per_sec:.0} bytes/s ({:.3} MB/s)",
        bytes_per_sec / 1_000_000.0
    );
}
