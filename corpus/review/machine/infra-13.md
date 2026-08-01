Solid first CI setup for the crate. Adding `cargo test`, `cargo clippy -- -D warnings`, and `cargo fmt --check` as separate jobs in `.github/workflows/ci.yml` covers the essentials, and running them in parallel rather than as sequential steps in one job is the right call for feedback speed — no reason to wait for clippy to finish before you know fmt failed.

The `Swatinem/rust-cache@v2` action is a good pick over hand-rolling `actions/cache` for the `target/` directory — it already knows how to key on `Cargo.lock` and handles the incremental-compilation cache invalidation quirks that people usually get wrong on a first attempt.

Two things I'd want addressed before merging:

The test job runs on `ubuntu-latest` only. If this crate is meant to be cross-platform (the `Cargo.toml` has conditional `cfg(windows)` blocks, so it looks like it is), you're not actually testing the Windows-specific code paths anywhere in CI. Worth adding a matrix over at least `ubuntu-latest` and `windows-latest`, macOS optionally.

`cargo clippy -- -D warnings` is good, but it's only running against the default feature set. If the crate has feature flags (I see `--all-features` isn't passed anywhere), you could have clippy-clean code on defaults while a feature-gated module has warnings nobody's ever seen. Add `--all-features` to both the clippy and test invocations, or at minimum a matrix entry for the significant feature combinations.

Minor: no `cargo audit` or equivalent dependency vulnerability scan. Not a blocker for a first CI pass, but worth a follow-up issue.

Nice to see `CARGO_TERM_COLOR: always` set in the workflow env — small thing but makes the Actions log output much more readable, and easy to forget.

Approving — the two gaps above (Windows testing, all-features clippy) are worth a fast follow-up but shouldn't hold up getting basic CI in place, which this workflow already does well.
