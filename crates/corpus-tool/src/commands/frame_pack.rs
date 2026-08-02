//! `corpus-tool frame-pack` — compiles the frame-rewrite rule set and
//! builds its derived binary artifact.
//!
//! Runs the whole compile fence (see `friction_packs`' own
//! `frame_compile` module docs for the gauntlet) over the four
//! compile-eligible buckets of `frame-rules-v1.toml`, serializes the
//! surviving rule program with `friction_packs::frame_bin`, and prints
//! the full compile report: every compiled rule, every report-only
//! demotion with its reason, and every rejected rule with its reason.
//! The pack's contents stay fully explainable from the source TOML and
//! this command's output alone.
//!
//! Deterministic and fully offline: every input is a vendored artifact
//! already checked into `crates/friction-packs/packs/` — the rule set
//! itself, the DMS index TOML whose human stream supplies the
//! attestation word rates, and the human-evidence pack that pools
//! additional external human-corpus evidence into that same rate (see
//! `friction_packs::human_evidence`'s own module docs — today's shipped
//! pack is the empty placeholder, so it changes nothing until real
//! external corpora are staged and `corpus-tool human-evidence` is
//! re-run). The rules TOML's sha256 is recorded in the artifact header,
//! so a stale `.bin` (rules edited, this command not re-run) fails
//! loudly in `friction-packs`' own drift test instead of silently
//! serving an older rule program. **Re-run this command after every
//! `frame-rules-v1.toml` change, after every `corpus-tool index` run**
//! (regenerates the attestation rates), and after every `corpus-tool
//! human-evidence` run.

use std::path::PathBuf;

use anyhow::Context as _;
use clap::Args as ClapArgs;
use friction_packs::Sha256;
use friction_packs::frame_compile::{CorpusEvidence, compile};
use friction_packs::frame_rules::FrameRuleSet;
use friction_packs::human_evidence::HumanEvidencePack;

/// Arguments for `corpus-tool frame-pack`.
#[derive(Debug, ClapArgs)]
pub struct Args {
    /// Path to the vendored `frame-rules-v1.toml` rule set.
    #[arg(
        long,
        default_value = "crates/friction-packs/packs/frame-rules-v1.toml"
    )]
    pub rules: PathBuf,
    /// Path to the vendored `dms-index-v1.toml` pack (attestation
    /// rates come from its human stream).
    #[arg(long, default_value = "crates/friction-packs/packs/dms-index-v1.toml")]
    pub dms_toml: PathBuf,
    /// Path to the vendored `human-evidence-v1.bin` pack (pooled into
    /// the dms-index-v1 human rate: see this module's own docs).
    #[arg(
        long,
        default_value = "crates/friction-packs/packs/human-evidence-v1.bin"
    )]
    pub human_evidence: PathBuf,
    /// Path to the vendored `machine-evidence-v1.bin` pack (the machine
    /// half of the review-register evidence pair).
    #[arg(
        long,
        default_value = "crates/friction-packs/packs/machine-evidence-v1.bin"
    )]
    pub machine_evidence: PathBuf,
    /// Path to write the derived binary artifact to.
    #[arg(long, default_value = "crates/friction-packs/packs/frame-pack-v1.bin")]
    pub out_bin: PathBuf,
}

/// Runs `frame-pack`: compiles the fenced buckets, writes the
/// artifact, prints the compile report.
///
/// # Errors
/// Returns an error if any input can't be read or parsed, if the
/// compile hits a structural inconsistency (see
/// `friction_packs::frame_compile::FrameCompileError`), or if the
/// `.bin` can't be written.
pub fn run(args: &Args) -> anyhow::Result<()> {
    let rules_text = std::fs::read_to_string(&args.rules)
        .with_context(|| format!("reading {}", args.rules.display()))?;
    let dms_text = std::fs::read_to_string(&args.dms_toml)
        .with_context(|| format!("reading {}", args.dms_toml.display()))?;
    let human_evidence_bytes = std::fs::read(&args.human_evidence)
        .with_context(|| format!("reading {}", args.human_evidence.display()))?;
    let set = FrameRuleSet::parse(&rules_text)
        .map_err(|e| anyhow::anyhow!("{} did not parse: {e}", args.rules.display()))?;
    let external = HumanEvidencePack::load(&human_evidence_bytes)
        .map_err(|e| anyhow::anyhow!("{} did not load: {e}", args.human_evidence.display()))?;
    let machine_evidence_bytes = std::fs::read(&args.machine_evidence)
        .with_context(|| format!("reading {}", args.machine_evidence.display()))?;
    let external_machine = HumanEvidencePack::load(&machine_evidence_bytes)
        .map_err(|e| anyhow::anyhow!("{} did not load: {e}", args.machine_evidence.display()))?;
    let rates = CorpusEvidence::from_dms_toml(&dms_text)
        .map_err(|e| anyhow::anyhow!("{} evidence: {e}", args.dms_toml.display()))?
        .with_external(external)
        .with_external_machine(external_machine);

    let (pack, report) = compile(&set, &rates)
        .map_err(|e| anyhow::anyhow!("compiling {}: {e}", args.rules.display()))?;

    let sha_hex = Sha256::of_bytes(rules_text.as_bytes()).to_string();
    let packed = friction_packs::frame_bin::pack_frame_bin(&pack, &sha_hex);
    if let Some(parent) = args.out_bin.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&args.out_bin, &packed)
        .with_context(|| format!("writing {}", args.out_bin.display()))?;

    println!(
        "frame-pack: {} -> {} ({} bytes), source sha256 {sha_hex}",
        args.rules.display(),
        args.out_bin.display(),
        packed.len(),
    );
    println!(
        "compiled: {} | report-only: {} | rejected: {}",
        report.compiled.len(),
        report.demoted.len(),
        report.rejected.len()
    );
    println!("\n--- compiled ---");
    for (id, kind) in &report.compiled {
        println!("  {id} [{kind:?}]");
    }
    println!("\n--- report-only ---");
    for (id, reason) in &report.demoted {
        println!("  {id}: {reason}");
    }
    println!("\n--- rejected ---");
    for (id, reject) in &report.rejected {
        println!("  {id}: {reject}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use friction_packs::frame_compile::{CorpusEvidence, compile};
    use friction_packs::frame_rules::FrameRuleSet;
    use friction_packs::human_evidence::HumanEvidencePack;

    /// Compiling and packing the vendored artifacts twice is
    /// byte-identical, the determinism pin every derived-artifact
    /// command in this tool carries.
    #[test]
    fn packing_the_vendored_rules_twice_is_byte_identical() {
        let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()
            .expect("workspace root exists");
        let rules_text = std::fs::read_to_string(
            repo_root.join("crates/friction-packs/packs/frame-rules-v1.toml"),
        )
        .expect("vendored frame-rules-v1.toml exists");
        let dms_text = std::fs::read_to_string(
            repo_root.join("crates/friction-packs/packs/dms-index-v1.toml"),
        )
        .expect("vendored dms-index-v1.toml exists");
        let human_evidence_bytes =
            std::fs::read(repo_root.join("crates/friction-packs/packs/human-evidence-v1.bin"))
                .expect("vendored human-evidence-v1.bin exists");
        let set = FrameRuleSet::parse(&rules_text).expect("vendored rules parse");
        let external =
            HumanEvidencePack::load(&human_evidence_bytes).expect("vendored human evidence loads");
        let rates = CorpusEvidence::from_dms_toml(&dms_text)
            .expect("vendored evidence derives")
            .with_external(external);
        let sha = "0".repeat(64);
        let first = friction_packs::frame_bin::pack_frame_bin(
            &compile(&set, &rates).expect("vendored set compiles").0,
            &sha,
        );
        let second = friction_packs::frame_bin::pack_frame_bin(
            &compile(&set, &rates).expect("vendored set compiles").0,
            &sha,
        );
        assert_eq!(first, second);
    }
}
