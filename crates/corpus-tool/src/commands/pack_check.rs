//! `corpus-tool pack-check` — parses and validates an inventory pack.

use std::path::PathBuf;

use clap::Args as ClapArgs;

/// Arguments for `corpus-tool pack-check`.
#[derive(Debug, ClapArgs)]
pub struct Args {
    /// Path to the inventory pack TOML file to check.
    #[arg(long, default_value = "crates/friction-packs/packs/inventory-v1.toml")]
    pub pack: PathBuf,
}

/// Runs `pack-check`: reads and parses `--pack`, then runs
/// [`friction_packs::validate`] over it.
///
/// # Errors
///
/// Returns an error (with a distinct `"pack-check: structurally invalid"`
/// message prefix) if the file can't be read or doesn't parse as a
/// well-formed inventory pack. Returns an error naming the violation
/// count if parsing succeeds but [`friction_packs::validate`] reports one
/// or more violations.
pub fn run(args: &Args) -> anyhow::Result<()> {
    let text = std::fs::read_to_string(&args.pack)
        .map_err(|e| anyhow::anyhow!("pack-check: failed to read {}: {e}", args.pack.display()))?;

    let pack = friction_packs::InventoryPack::parse(&text)
        .map_err(|e| anyhow::anyhow!("pack-check: structurally invalid pack: {e}"))?;

    let violations = friction_packs::validate(&pack);
    if violations.is_empty() {
        println!(
            "pack-check: {} is clean (version={}, sha256={})",
            args.pack.display(),
            pack.version(),
            friction_packs::Sha256::of_bytes(text.as_bytes())
        );
        println!(
            "  deletion_spans={} substitution_pairs={} ritual_frames={} preview_frames={} \
             lvc_pairs={} output_frequency_bands={}",
            pack.deletion_spans().len(),
            pack.substitution_pairs().len(),
            pack.ritual_frames().len(),
            pack.preview_frames().len(),
            pack.lvc_pairs().len(),
            pack.output_frequency_bands().len(),
        );
        return Ok(());
    }

    for violation in &violations {
        println!("{violation}");
    }
    anyhow::bail!(
        "pack-check: {} has {} violation(s)",
        args.pack.display(),
        violations.len()
    );
}
