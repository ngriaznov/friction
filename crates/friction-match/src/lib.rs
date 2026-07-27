//! Fix-time detection: five independent channels (differential matching
//! statistics, a literal tell inventory, licensed light-verb
//! constructions, deterministic contrast-frame templates, and metaphor-
//! compound jargon detection) over a document's prose, reporting byte-
//! honest [`MatchSpan`]s. This crate never rewrites text — see
//! [`MatchEngine`]'s own docs.
//!
//! # Span honesty by construction, not by translation
//!
//! Every regex/automaton match this crate makes runs directly against the
//! *original* source bytes of a prose region — never against a cleaned or
//! rewritten copy. Case-folding and curly-quote handling are done via
//! case-insensitive matching and widened character classes at match time
//! (see [`token::tokenize_str`], [`literal::LiteralAutomaton`]), not via a
//! `clean()`-style byte-rewriting pre-pass. That means every match's byte
//! range *is already* a valid original-source range — there is no
//! cleaned-to-original span translation table to get wrong.
//!
//! `friction_harness::clean` stays exactly as it is: it is pinned to
//! `corpus-tool index`'s whole-file, no-spans pack-build pipeline, and
//! nothing in this crate depends on it at build time. This crate's own
//! [`token`] module owns a tiny span-carrying tokenizer instead, pinned to
//! `clean::tokenize`'s token-boundary convention by a dev-dependency
//! equivalence test (`tests/token_convention.rs`), not by shared code —
//! the two tokenizers solve the same boundary problem over different
//! inputs (pre-lowercased whole-file text vs. raw, mixed-case,
//! per-prose-block text), so sharing the regex object would be cosmetic,
//! not real deduplication.
//!
//! # Prose-blocks-only, enforced exactly once
//!
//! `friction-parse` extracts prose from headings and table cells too
//! (other consumers need that); this crate filters instead of assuming.
//! [`token::prose_scope`] is the one allowlisted function that walks
//! [`friction_core::Document::blocks`]/[`friction_core::Document::prose`];
//! every channel consumes its output and never touches [`Document`]
//! directly.
//!
//! # DMS runs per prose unit, independently
//!
//! See [`dms`]'s own module docs for why the automaton walk resets at
//! every in-scope prose unit rather than streaming the whole document as
//! one contiguous token stream.
//!
//! # DMS without the other two channels
//!
//! [`dms_spans_for_family`] is a standalone DMS-only entry point: given
//! prose units already scoped by [`token::prose_scope`], it returns one
//! family's spans without building a [`MatchEngine`] (which also compiles
//! the literal automaton and requires a tagger/segmenter for the LVC
//! channel). `friction fix` uses it to scan its own fixed output against
//! every pack family for paraphrase suggestions it detects but never
//! auto-edits — DMS stays banned as an edit judge (`friction-edit`'s own
//! crate docs: every gate is the curated inventory pack and the corpus-
//! attested seam-bigram/skeleton tables, never a metric or genre
//! envelope); this crate still only detects and reports, never rewrites.

mod dms;
mod error;
pub mod frame;
pub mod jargon;
mod literal;
mod lvc;
pub mod span;
pub mod token;

pub use error::MatchError;
pub use span::{Channel, DmsFamilyReport, DmsReport, DocumentReport, MatchScore, MatchSpan};

use friction_core::Document;
use friction_nlp::{Segmenter, Tagger};
use friction_packs::{DmsIndex, InventoryPack, JargonPack, ModelFamily};

use crate::token::ScopedUnit;

/// Built once (compiles the literal automaton once) and reused across
/// every document scanned against the same pack/family/tagger/segmenter.
///
/// This crate only detects and reports spans with frame ids — it never
/// rewrites text. A repair layer built on top of this crate's output is a
/// separate concern.
pub struct MatchEngine<'a> {
    inventory: &'a InventoryPack,
    dms: &'a DmsIndex,
    jargon: &'a JargonPack,
    target_family: ModelFamily,
    tagger: &'a dyn Tagger,
    segmenter: &'a dyn Segmenter,
    automaton: literal::LiteralAutomaton,
}

impl<'a> MatchEngine<'a> {
    /// Builds an engine bound to `inventory`, `dms`, `jargon`,
    /// `target_family`, `tagger`, and `segmenter` for its whole lifetime —
    /// the literal automaton is compiled exactly once here.
    ///
    /// # Errors
    /// [`MatchError::FamilyNotInPack`] if `dms` has no stream for
    /// `target_family`; [`MatchError::Automaton`] if the inventory's
    /// literal-eligible patterns fail to compile (should not happen for
    /// the embedded pack — covered by this crate's own tests).
    pub fn new(
        inventory: &'a InventoryPack,
        dms: &'a DmsIndex,
        jargon: &'a JargonPack,
        target_family: ModelFamily,
        tagger: &'a dyn Tagger,
        segmenter: &'a dyn Segmenter,
    ) -> Result<Self, MatchError> {
        if dms.family_sam(target_family).is_none() {
            return Err(MatchError::FamilyNotInPack(target_family));
        }
        let automaton = literal::LiteralAutomaton::build(inventory)?;
        Ok(Self {
            inventory,
            dms,
            jargon,
            target_family,
            tagger,
            segmenter,
            automaton,
        })
    }

    /// Scans `document`, running all five channels over the same
    /// prose-only, in-scope unit set ([`token::prose_scope`]), and
    /// merges their spans deterministically ([`span::merge_spans`]).
    ///
    /// # Errors
    /// [`MatchError::Core`] if a prose range fails to slice — not
    /// expected for any `Document` produced by `friction_parse::parse`,
    /// since its ranges are already validated at construction.
    pub fn scan(&self, document: &Document) -> Result<DocumentReport, MatchError> {
        let units = token::prose_scope(document, self.segmenter);

        let target_sam = self
            .dms
            .family_sam(self.target_family)
            .expect("MatchEngine::new already validated target_family has a stream");
        let human_sam = self.dms.human_sam();
        let vocab = self.dms.vocab();

        let dms_spans = dms::scan_units(&units, target_sam, self.target_family, human_sam, vocab);
        let dms_report = dms::document_report(&units, self.dms, self.target_family, vocab);

        let literal_ac_spans = self.automaton.scan_units(&units);
        let literal_fallback_spans =
            literal::regex_fallback_spans(self.inventory, &self.automaton, &units);

        let lexicon = self.inventory.lvc_lexicon();
        let lvc_spans = lvc::scan_units(&units, document.source(), self.tagger, lexicon);

        let frame_spans = frame::scan_units(&units);

        let jargon_spans = jargon::scan_units(&units, document.source(), self.tagger, self.jargon);

        let spans = span::merge_spans(vec![
            dms_spans,
            literal_ac_spans,
            literal_fallback_spans,
            lvc_spans,
            frame_spans,
            jargon_spans,
        ]);

        Ok(DocumentReport {
            spans,
            dms: dms_report,
        })
    }
}

/// Returns [`Channel::Dms`] spans for exactly `family` over `units`
/// (already scoped by [`token::prose_scope`]), scanning `dms`'s stream for
/// `family` against the same pack's human baseline stream.
///
/// `None` if `dms` has no stream for `family` — a caller-supplied override
/// pack may define fewer than [`ModelFamily::ALL`], and a caller checking
/// every family already expects to skip the ones a given pack lacks rather
/// than treat that as an error.
///
/// A standalone entry point for callers that need only the DMS channel
/// over every pack family (e.g. `friction fix`'s paraphrase-suggestion
/// report) without paying for [`MatchEngine`]'s literal-automaton
/// compilation or running the literal/LVC channels they don't need.
#[must_use]
pub fn dms_spans_for_family(
    units: &[ScopedUnit<'_>],
    dms: &DmsIndex,
    family: ModelFamily,
) -> Option<Vec<MatchSpan>> {
    let target_sam = dms.family_sam(family)?;
    let human_sam = dms.human_sam();
    let vocab = dms.vocab();
    Some(dms::scan_units(units, target_sam, family, human_sam, vocab))
}
