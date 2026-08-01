//! Shared test-only helper: a `MatchEngine` built from the real embedded
//! packs, a real tagger, and a real segmenter, reused across this crate's
//! integration test binaries.

use std::sync::OnceLock;

use friction_match::MatchEngine;
use friction_nlp::{PerceptronTagger, SrxSegmenter};
use friction_packs::ModelFamily;

/// Builds a `MatchEngine` bound to the embedded inventory and DMS packs,
/// targeting `ModelFamily::Qwen` (the pack defines every family, so the
/// choice here is arbitrary — nothing in this crate decides *which*
/// family to scan against for a given input document; that is a caller
/// concern).
#[allow(dead_code)]
pub fn engine() -> MatchEngine<'static> {
    static TAGGER: OnceLock<PerceptronTagger> = OnceLock::new();
    static SEGMENTER: OnceLock<SrxSegmenter> = OnceLock::new();

    let tagger =
        TAGGER.get_or_init(|| PerceptronTagger::new().expect("embedded tagger model loads"));
    let segmenter = SEGMENTER.get_or_init(SrxSegmenter::new);

    MatchEngine::new(
        &friction_packs::INVENTORY.pack,
        &friction_packs::DMS.pack,
        &friction_packs::JARGON.pack,
        &friction_packs::JARGON_ATTEST,
        &friction_packs::HUMAN_EVIDENCE,
        ModelFamily::Qwen,
        tagger,
        segmenter,
    )
    .expect("engine builds from the embedded packs")
}
