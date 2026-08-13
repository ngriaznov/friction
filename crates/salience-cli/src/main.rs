//! `salience` — produce a per-line salience sidecar for a compiled class or a
//! Python source file.
//!
//! ```text
//! salience Foo.class                 # JSON sidecar on stdout
//! salience foo.py --annotate Foo.py  # tiered source view
//! salience Foo.class --stats         # tier histogram and timing
//! ```

#![forbid(unsafe_code)]
#![warn(clippy::pedantic)]
// `CPython`, `MokaIR`, `JSR`, `SMAP` and friends are proper nouns in prose,
// not identifiers.
// Line counts are small; the f64 cast in the histogram cannot lose anything.
#![allow(clippy::cast_precision_loss, clippy::doc_markdown)]

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use clap::{Parser, ValueEnum};
use salience_core::ir::FunctionIr;
use salience_core::{Denylist, ScoreWeights, Sidecar, analyze};

/// Output shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Format {
    /// The sidecar artifact as JSON.
    Json,
    /// One line per span, for reading in a terminal.
    Text,
}

#[derive(Debug, Parser)]
#[command(
    name = "salience",
    about = "Deterministic per-line salience tiering over function bodies",
    long_about = "Classifies every statement in every function body as core, boundary, \
plumbing or inert, and projects the result onto source lines.\n\n\
Accepts a JVM .class file or a Python source file. No model runs: every tier is \
the output of a dominance, loop or reachability query, and every span carries \
the reason that produced it."
)]
struct Args {
    /// The `.class` or `.py` file to analyze.
    input: PathBuf,

    /// Output format.
    #[arg(long, value_enum, default_value_t = Format::Json)]
    format: Format,

    /// Print the source with each line marked by tier. Requires the source file
    /// when analyzing bytecode, since a class file does not carry it.
    #[arg(long, value_name = "SOURCE")]
    annotate: Option<PathBuf>,

    /// Print a tier histogram and the analysis time.
    #[arg(long)]
    stats: bool,

    /// Do not treat any call as inert.
    #[arg(long)]
    no_denylist: bool,

    /// Additional inert call patterns, matched by substring. Repeatable.
    #[arg(long = "inert", value_name = "PATTERN")]
    inert: Vec<String>,

    /// Interpreter to use when lowering Python.
    #[arg(long, default_value = "python3")]
    python: String,
}

fn main() -> ExitCode {
    let args = Args::parse();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("salience: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    let mut denylist = if args.no_denylist {
        Denylist::empty()
    } else {
        Denylist::new()
    };
    for p in &args.inert {
        denylist = denylist.with(p.clone());
    }
    let weights = ScoreWeights::default();

    let ext = args
        .input
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or_default();

    let lower_started = Instant::now();
    let (generator, functions, note) = match ext {
        "class" => {
            let bytes = std::fs::read(&args.input)?;
            let lowered = salience_jvm::lower_class(&bytes)?;
            let note = if lowered.has_smap {
                Some(format!(
                    "{} carries a SourceDebugExtension (JSR-45/SMAP): some line numbers \
refer to an inlined declaration site in another file, not to this one",
                    lowered.binary_name
                ))
            } else {
                None
            };
            ("salience-jvm/mokapot".to_owned(), lowered.functions, note)
        }
        "py" => {
            let lowered = salience_py::lower_file_with(&args.input, &args.python)?;
            ("salience-py/dis".to_owned(), lowered.functions, None)
        }
        other => {
            return Err(
                format!("unsupported input type {other:?}: expected a .class or .py file").into(),
            );
        }
    };

    let file_label = functions
        .first()
        .map_or_else(|| args.input.display().to_string(), |f| f.id.file.clone());
    let mut sidecar = Sidecar::new(file_label, generator);

    let lowering = lower_started.elapsed();
    let analysis_started = Instant::now();
    let mut analyzed = 0usize;
    for ir in &functions {
        if let Err(e) = ir.validate() {
            eprintln!("salience: skipping {}: {e}", ir.id.name);
            continue;
        }
        if ir.is_empty() {
            continue;
        }
        let sal = analyze(ir, &denylist, &weights);
        sidecar.push(ir, &sal);
        analyzed += 1;
    }
    sidecar.finish();
    let analysis = analysis_started.elapsed();

    if let Some(note) = note {
        eprintln!("salience: note: {note}");
    }

    match args.format {
        Format::Json => println!("{}", serde_json::to_string_pretty(&sidecar)?),
        Format::Text => print_text(&sidecar),
    }

    if let Some(source) = &args.annotate {
        annotate(source, &sidecar)?;
    }

    if args.stats {
        print_stats(
            &sidecar,
            analyzed,
            functions.len(),
            lowering,
            analysis,
            &functions,
        );
    }
    Ok(())
}

/// One line per span.
fn print_text(sidecar: &salience_core::Sidecar) {
    for f in &sidecar.functions {
        println!(
            "\n{} {}  [{}/{} instructions carry a line]",
            f.name, f.signature, f.coverage.with_line, f.coverage.instructions
        );
        for s in &f.spans {
            let range = if s.start == s.end {
                format!("{}", s.start)
            } else {
                format!("{}-{}", s.start, s.end)
            };
            println!(
                "  {range:>9}  {:<9} {:.2}  {}",
                s.tier,
                s.score,
                s.reasons.first().map_or("", String::as_str)
            );
        }
    }
}

/// The source with a tier marker per line.
fn annotate(source: &Path, sidecar: &salience_core::Sidecar) -> std::io::Result<()> {
    let text = std::fs::read_to_string(source)?;
    println!("\n--- {} ---", source.display());
    for (i, line) in text.lines().enumerate() {
        let n = u32::try_from(i + 1).unwrap_or(u32::MAX);
        let marker = match sidecar.tier_at(n) {
            Some("core") => "CORE ",
            Some("boundary") => "BOUND",
            Some("plumbing") => "plumb",
            Some("inert") => "inert",
            _ => "     ",
        };
        println!("{marker} {n:>4} | {line}");
    }
    Ok(())
}

/// Histogram and timing.
fn print_stats(
    sidecar: &salience_core::Sidecar,
    analyzed: usize,
    total: usize,
    lowering: std::time::Duration,
    analysis: std::time::Duration,
    functions: &[FunctionIr],
) {
    let mut core = 0;
    let mut boundary = 0;
    let mut plumbing = 0;
    let mut inert = 0;
    for f in &sidecar.functions {
        core += f.summary.core;
        boundary += f.summary.boundary;
        plumbing += f.summary.plumbing;
        inert += f.summary.inert;
    }
    let lines = core + boundary + plumbing + inert;
    let instructions: usize = functions.iter().map(FunctionIr::len).sum();
    let with_line: usize = sidecar.functions.iter().map(|f| f.coverage.with_line).sum();

    eprintln!("\n--- stats ---");
    eprintln!("functions   {analyzed} analyzed / {total} lowered");
    eprintln!("instructions {instructions}, {with_line} with a source line");
    eprintln!("lines       {lines} tiered");
    if lines > 0 {
        let pct = |x: usize| (x as f64) * 100.0 / (lines as f64);
        eprintln!("  core      {core:>5}  {:.1}%", pct(core));
        eprintln!("  boundary  {boundary:>5}  {:.1}%", pct(boundary));
        eprintln!("  plumbing  {plumbing:>5}  {:.1}%", pct(plumbing));
        eprintln!("  inert     {inert:>5}  {:.1}%", pct(inert));
    }
    // Reported apart because they are paid at different times and by
    // different people. Lowering is I/O plus a parse — for Python it is a whole
    // interpreter subprocess — and happens once per file. Analysis is the part
    // that would run inside an editor hook, and the part the caching story is
    // about.
    eprintln!("lowering    {lowering:?}");
    eprintln!("analysis    {analysis:?}");
    if analyzed > 0 {
        eprintln!(
            "            {:?} per function",
            analysis / u32::try_from(analyzed).unwrap_or(1)
        );
    }
}
