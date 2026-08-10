//! `jargon.compound`'s arming fence, exercised in a process that never
//! installs `general-evidence-v1`.
//!
//! A genuinely separate `cargo test` binary from every other test file in
//! this crate (each file under `tests/` gets its own process) — the one
//! guarantee that lets this assert [`friction_packs::general_evidence`]
//! resolves to [`None`] without racing `jargon_compound.rs`'s own unit
//! test, which is the sole place in this crate's *other* (lib) test
//! binary allowed to install it (see that test's own docs).

mod support;

use friction_match::jargon_compound::JARGON_COMPOUND_ID;
use friction_nlp::{PerceptronTagger, SrxSegmenter};
use support::engine;

/// A document built entirely from ordinary noun-noun/adj-noun pairs that
/// would very likely flag as `jargon.compound` if any general-evidence
/// pack were resolved and their pairing came up unattested — the pack's
/// absence, not a lucky choice of attested vocabulary, is what must be
/// keeping this quiet.
const SOURCE: &str = "\
# Report

The team built a semantic ladder to rank every candidate result before \
the cross domain resonance study began. A rich tapestry of services \
underpins the whole pipeline, and the shared vocabulary drift kept \
surprising every reviewer.
";

#[test]
fn general_evidence_resolves_to_none_in_this_process() {
    assert!(
        friction_packs::general_evidence().is_none(),
        "no general-evidence-v1 override, env var, or cache file exists in this test process"
    );
}

#[test]
fn jargon_compound_scan_units_returns_nothing_when_the_pack_is_absent() {
    assert!(friction_packs::general_evidence().is_none());

    let document = friction_parse::parse(SOURCE).expect("valid markdown parses");
    let segmenter = SrxSegmenter::new();
    let pos_tagger = PerceptronTagger::new().expect("embedded tagger model loads");
    let units = friction_match::token::prose_scope(&document, &segmenter);
    let tagged_sentences =
        friction_match::tagging::tag_units(&units, document.source(), &pos_tagger);

    let spans = friction_match::jargon_compound::scan_units(
        &tagged_sentences,
        document.source(),
        &friction_packs::JARGON_ATTEST,
    );
    assert!(spans.is_empty(), "{spans:?}");
}

/// The engine-level guarantee this arming fence exists for: with the pack
/// absent, `MatchEngine::scan`'s merged report carries zero
/// `jargon.compound` spans — the channel's wiring into
/// [`friction_match::MatchEngine::scan`] contributes nothing, byte for
/// byte, until an operator actually installs or caches the artifact.
#[test]
fn match_engine_scan_reports_no_jargon_compound_spans_when_the_pack_is_absent() {
    assert!(friction_packs::general_evidence().is_none());

    let document = friction_parse::parse(SOURCE).expect("valid markdown parses");
    let report = engine().scan(&document).expect("scan runs to completion");
    assert!(
        report
            .spans
            .iter()
            .all(|span| &*span.frame_id != JARGON_COMPOUND_ID),
        "{:?}",
        report.spans
    );
}
