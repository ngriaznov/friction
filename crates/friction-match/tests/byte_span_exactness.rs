//! Byte-span honesty over real corpus documents.
//!
//! Every path embedded below was independently verified against
//! `corpus/manifest.jsonl` to have `"split": "train"` or `"split":
//! "dev"` — never `"holdout"` — and is hardcoded by path rather than
//! discovered by walking the corpus directory, so this test can never
//! accidentally pick up a holdout document even if the corpus layout
//! changes. `corpus/holdout.lock` is never read.

mod support;

use proptest::prelude::*;
use proptest::sample::select;

use friction_core::span::Spanned;
use friction_match::token::prose_scope;
use friction_nlp::SrxSegmenter;
use support::engine;

/// `(path, source)` for five human and five LLM `docs`-genre corpus
/// files, all `train` or `dev` split.
const DOCS: &[(&str, &str)] = &[
    (
        "corpus/human/docs/01ec8967989205a2.md",
        include_str!("../../../corpus/human/docs/01ec8967989205a2.md"),
    ),
    (
        "corpus/human/docs/043f761fc6dc1a88.md",
        include_str!("../../../corpus/human/docs/043f761fc6dc1a88.md"),
    ),
    (
        "corpus/human/docs/059d88d5de18c59c.md",
        include_str!("../../../corpus/human/docs/059d88d5de18c59c.md"),
    ),
    (
        "corpus/human/docs/08d07d7b04ccd440.md",
        include_str!("../../../corpus/human/docs/08d07d7b04ccd440.md"),
    ),
    (
        "corpus/human/docs/099e1e22b0439e17.md",
        include_str!("../../../corpus/human/docs/099e1e22b0439e17.md"),
    ),
    (
        "corpus/llm/docs/00533d2e3a398154.md",
        include_str!("../../../corpus/llm/docs/00533d2e3a398154.md"),
    ),
    (
        "corpus/llm/docs/0a0197006e9ca159.md",
        include_str!("../../../corpus/llm/docs/0a0197006e9ca159.md"),
    ),
    (
        "corpus/llm/docs/13f0ebf1b88ce101.md",
        include_str!("../../../corpus/llm/docs/13f0ebf1b88ce101.md"),
    ),
    (
        "corpus/llm/docs/21b549af281d2808.md",
        include_str!("../../../corpus/llm/docs/21b549af281d2808.md"),
    ),
    (
        "corpus/llm/docs/25e804f4db7f550c.md",
        include_str!("../../../corpus/llm/docs/25e804f4db7f550c.md"),
    ),
];

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))]

    /// For every emitted span: `source[span.range]` slices cleanly, is
    /// non-empty, and (sanity check beyond "doesn't panic") is fully
    /// contained within a single in-scope prose unit's range — the
    /// strongest form of the prose-only + byte-honesty guarantee
    /// combined, since it proves the per-unit-reset design in this
    /// crate's own docs actually holds and can never bridge across a
    /// block boundary.
    #[test]
    fn every_span_round_trips_and_stays_within_a_single_prose_unit(
        (path, source) in select(DOCS.to_vec())
    ) {
        let document = friction_parse::parse(source)
            .unwrap_or_else(|e| panic!("{path} must parse as markdown: {e}"));
        let report = engine()
            .scan(&document)
            .unwrap_or_else(|e| panic!("{path} must scan without error: {e}"));

        let segmenter = SrxSegmenter::new();
        let units = prose_scope(&document, &segmenter);

        for span in &report.spans {
            let text = document
                .text(&span.range())
                .unwrap_or_else(|e| panic!("{path}: span {span:?} must slice cleanly: {e}"));
            prop_assert!(!text.is_empty(), "{path}: span {span:?} sliced to empty text");
            prop_assert!(
                !text.contains("\n\n"),
                "{path}: span {span:?} crosses a paragraph boundary: {text:?}"
            );

            let contained = units
                .iter()
                .any(|unit| unit.unit.range.start <= span.range.start
                    && span.range.end <= unit.unit.range.end);
            prop_assert!(
                contained,
                "{path}: span {span:?} is not fully contained within any single in-scope prose unit"
            );
        }
    }
}
