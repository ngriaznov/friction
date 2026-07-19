//! Binary entry point for `corpus-tool`; see `corpus_tool::cli`.

fn main() -> anyhow::Result<()> {
    corpus_tool::cli::run()
}
