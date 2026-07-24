//! Fix-time detection: three independent channels (differential matching
//! statistics, a literal tell inventory, and licensed light-verb
//! constructions) over a document's prose, reporting byte-honest
//! [`MatchSpan`]s. This crate never rewrites text — see [`MatchEngine`]'s
//! own docs.
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

mod dms;
mod error;
mod literal;
mod lvc;
pub mod span;
pub mod token;

pub use error::MatchError;
pub use span::{Channel, DmsFamilyReport, DmsReport, DocumentReport, MatchScore, MatchSpan};

use friction_core::Document;
use friction_nlp::{Segmenter, Tagger};
use friction_packs::{DmsIndex, InventoryPack, ModelFamily};

/// Built once (compiles the literal automaton once) and reused across
/// every document scanned against the same pack/family/tagger/segmenter.
///
/// This crate only detects and reports spans with frame ids — it never
/// rewrites text. A repair layer built on top of this crate's output is a
/// separate concern.
pub struct MatchEngine<'a> {
    inventory: &'a InventoryPack,
    dms: &'a DmsIndex,
    target_family: ModelFamily,
    tagger: &'a dyn Tagger,
    segmenter: &'a dyn Segmenter,
    automaton: literal::LiteralAutomaton,
}

impl<'a> MatchEngine<'a> {
    /// Builds an engine bound to `inventory`, `dms`, `target_family`,
    /// `tagger`, and `segmenter` for its whole lifetime — the literal
    /// automaton is compiled exactly once here.
    ///
    /// # Errors
    /// [`MatchError::FamilyNotInPack`] if `dms` has no stream for
    /// `target_family`; [`MatchError::Automaton`] if the inventory's
    /// literal-eligible patterns fail to compile (should not happen for
    /// the embedded pack — covered by this crate's own tests).
    pub fn new(
        inventory: &'a InventoryPack,
        dms: &'a DmsIndex,
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
            target_family,
            tagger,
            segmenter,
            automaton,
        })
    }

    /// Scans `document`, running all three channels over the same
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

        let spans = span::merge_spans(vec![
            dms_spans,
            literal_ac_spans,
            literal_fallback_spans,
            lvc_spans,
        ]);

        Ok(DocumentReport {
            spans,
            dms: dms_report,
        })
    }
}
