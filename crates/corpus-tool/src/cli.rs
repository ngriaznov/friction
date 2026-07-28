//! Top-level CLI definition and dispatch.

use clap::{Parser, Subcommand};

use crate::commands;

/// `corpus-tool`: manage the friction validation corpus.
#[derive(Debug, Parser)]
#[command(
    name = "corpus-tool",
    version,
    about = "Manage the friction validation corpus"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

/// Subcommands. Each variant delegates to a same-named module under
/// `crate::commands`, one module per subcommand, so a new subcommand slots
/// in without touching the others.
#[derive(Debug, Subcommand)]
enum Command {
    /// Validate the manifest and corpus files.
    Validate(commands::validate::Args),
    /// Print per-`(class, genre)` corpus statistics.
    Stats(commands::stats::Args),
    /// Compute the deterministic stratified train/dev/holdout split.
    Split(commands::split::Args),
    /// Freeze the holdout split into `<corpus_dir>/holdout.lock`.
    Seal(commands::seal::Args),
    /// Verify `<corpus_dir>/holdout.lock` against the manifest and files.
    HoldoutCheck(commands::holdout_check::Args),
    /// Clean an incoming raw-doc directory into the corpus layout.
    Clean(commands::clean::Args),
    /// Ingest incoming human-corpus docs + metadata into the manifest.
    Ingest(commands::ingest::Args),
    /// Remove one or more docs: drops the manifest record and corpus
    /// file, leaving the raw original under `corpus/incoming/` in place.
    Remove(commands::remove::Args),
    /// Maintenance pass: decode raw HTML entities left in already-ingested
    /// corpus docs, rewriting affected files in place and refreshing
    /// their manifest `sha256`.
    FixEntities(commands::fix_entities::Args),
    /// Generate the LLM corpus via Ollama.
    Generate(commands::generate::Args),
    /// Generate the stock/antislop paired mining corpus via Ollama.
    GeneratePaired(commands::generate_paired::Args),
    /// Estimate per-`(genre, metric)` human percentile bands from the
    /// train split and write a versioned envelope pack.
    Envelope(commands::envelope::Args),
    /// On the dev split, report how well the metric vector separates
    /// `llm` docs from `human` docs, per genre and per metric.
    Separate(commands::separate::Args),
    /// On the sealed holdout split (see `holdout-check`), report
    /// human-holdout vs llm-holdout (baseline) and human-holdout vs
    /// fixed-llm-holdout (after running the release `friction` binary)
    /// combined-score AUCs and distributions, per genre.
    SeparateHoldout(commands::separate_holdout::Args),
    /// On the train split, mine discriminative 1-/2-/3-gram phrases
    /// between `llm` and `human` prose.
    Mine(commands::mine::Args),
    /// On the train split, mine ratio-threshold literal n-grams,
    /// POS-skeleton patterns, block-position-conditioned frames, and
    /// light-verb-construction pair rates for the curated inventory pack.
    MineInventory(commands::mine_inventory::Args),
    /// On the stock/antislop paired mining corpus, mine ratio-threshold
    /// literal n-grams (2-/3-/4-gram only), with a read-only
    /// human-train cross-check column.
    MinePaired(commands::mine_paired::Args),
    /// Builds the per-model-family and human token-id streams (over one
    /// shared vocabulary) that a DMS suffix-automaton index reconstructs
    /// from.
    Index(commands::index::Args),
    /// Parses and validates an inventory pack (structural checks plus
    /// disjointness/closure/frequency-hygiene rules).
    PackCheck(commands::pack_check::Args),
    /// On the train split, measures how often a fixed literal replacement
    /// phrase already occurs naturally in the human corpus, for the
    /// inventory pack's output-frequency hygiene bands.
    OutputBands(commands::output_bands::Args),
    /// On the train split, builds the seam-bigram membership table and
    /// POS-skeleton n-gram sets for the attestation pack.
    Attest(commands::attest::Args),
    /// On the train split's docs genre, measures each document's
    /// per-1000-prose-word em-dash rate and reports the population's
    /// 10th/50th/90th percentile, for `register-v1.toml`'s
    /// `[features.em_dash]`.
    RegisterBands(commands::register_bands::Args),
    /// Builds `jargon-attest-v1`: a `BinaryFuse8` filter over normalized
    /// Wikipedia-title and OpenAlex-topic compound keys, for
    /// `friction-match`'s `jargon.metaphor` channel.
    JargonAttest(commands::jargon_attest::Args),
    /// Builds the derived `.bin` weight artifacts
    /// (`friction_nlp::PerceptronTagger`/`PerceptronParser` load at
    /// runtime) from the vendored, audited `json.gz` weight artifacts.
    WeightsPack(commands::weights_pack::Args),
    /// Builds the derived DMS binary artifact (`friction_packs::DMS`
    /// loads at runtime) from the vendored `dms-index-v1.toml`: every
    /// stream's suffix automaton pre-built and serialized flat, so
    /// process start pays no TOML parse or automaton construction.
    DmsPack(commands::dms_pack::Args),
}

/// Parses process arguments and runs the selected subcommand.
///
/// # Errors
///
/// Returns an error if the selected subcommand fails; the caller (`main`)
/// should treat this as a non-zero exit. See each `commands::*::run` for
/// what specifically can fail.
///
/// `generate` and `generate-paired` are the two subcommands that can
/// also make the *process* exit non-zero on success: if any job was
/// skipped because its model wasn't available in Ollama, each calls
/// `std::process::exit(commands::generate::EXIT_CODE_MODELS_SKIPPED)`
/// after printing the summary, rather than returning `Ok(())` — every
/// other subcommand's success is a plain `Ok(())`.
pub fn run() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Validate(args) => commands::validate::run(&args),
        Command::Stats(args) => commands::stats::run(&args),
        Command::Split(args) => commands::split::run(&args),
        Command::Seal(args) => commands::seal::run(&args),
        Command::HoldoutCheck(args) => commands::holdout_check::run(&args),
        Command::Clean(args) => commands::clean::run(&args),
        Command::Ingest(args) => commands::ingest::run(&args),
        Command::Remove(args) => commands::remove::run(&args),
        Command::FixEntities(args) => commands::fix_entities::run(&args),
        Command::Generate(args) => {
            let outcome = commands::generate::run(&args)?;
            if outcome.any_models_skipped() {
                std::process::exit(commands::generate::EXIT_CODE_MODELS_SKIPPED);
            }
            Ok(())
        }
        Command::GeneratePaired(args) => {
            let outcome = commands::generate_paired::run(&args)?;
            if outcome.any_models_skipped() {
                std::process::exit(commands::generate::EXIT_CODE_MODELS_SKIPPED);
            }
            Ok(())
        }
        Command::Envelope(args) => commands::envelope::run(&args),
        Command::Separate(args) => commands::separate::run(&args),
        Command::SeparateHoldout(args) => commands::separate_holdout::run(&args),
        Command::Mine(args) => commands::mine::run(&args),
        Command::MineInventory(args) => commands::mine_inventory::run(&args),
        Command::MinePaired(args) => commands::mine_paired::run(&args),
        Command::Index(args) => commands::index::run(&args),
        Command::PackCheck(args) => commands::pack_check::run(&args),
        Command::OutputBands(args) => commands::output_bands::run(&args),
        Command::Attest(args) => commands::attest::run(&args),
        Command::RegisterBands(args) => commands::register_bands::run(&args),
        Command::JargonAttest(args) => commands::jargon_attest::run(&args),
        Command::WeightsPack(args) => commands::weights_pack::run(&args),
        Command::DmsPack(args) => commands::dms_pack::run(&args),
    }
}
