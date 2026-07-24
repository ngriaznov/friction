//! The friction v4 repair engine: four operations (ritual deletion, paired
//! substitution, derivational pivot, gated span deletion) applied per
//! sentence, in that fixed order, gated by the curated inventory pack and
//! the corpus-attested seam-bigram/skeleton tables — never by a metric or
//! genre envelope.
//!
//! [`Engine`] is the top-level entry point: it loads the embedded tagger,
//! segmenter, inventory pack, and attestation pack once, and
//! [`Engine::fix_document`] runs the bounded two-pass pipeline
//! (`document::edit_document`) over a document's prose.
//!
//! No synthesis: every emitted [`friction_core::Patch`] either deletes,
//! substitutes a fixed pack string, or derives a verb via
//! `friction_nlp::lvc::conjugate`'s static morphology tables. When a gate
//! fails, the candidate is held (Suggest-tier) and the source bytes are
//! left untouched.

pub mod document;
mod error;
pub mod gates;
pub mod nearnoop;
pub mod sentence;
pub mod splice;

pub use document::{EditReport, PassReport, edit_document};
pub use error::EditError;

use friction_nlp::{PerceptronTagger, SrxSegmenter};
use friction_packs::{AttestationPack, InventoryPack};

/// The engine: the embedded tagger, segmenter, inventory pack, and
/// attestation pack, loaded once and reused across every
/// [`Engine::fix_document`] call.
pub struct Engine {
    inventory: &'static InventoryPack,
    attestation: &'static AttestationPack,
    tagger: PerceptronTagger,
    segmenter: SrxSegmenter,
}

impl Engine {
    /// Builds an engine over the embedded, process-lifetime inventory and
    /// attestation packs.
    ///
    /// # Errors
    /// Returns [`EditError::Tagger`] if the embedded tagger weights fail
    /// to load.
    pub fn new() -> Result<Self, EditError> {
        let tagger = PerceptronTagger::new()?;
        Ok(Self {
            inventory: &friction_packs::INVENTORY.pack,
            attestation: &friction_packs::ATTESTATION.pack,
            tagger,
            segmenter: SrxSegmenter::new(),
        })
    }

    /// Runs the bounded two-pass repair pipeline over `source`, returning
    /// the fixed text and a report of what happened each pass.
    ///
    /// # Errors
    /// Returns [`EditError`] if `source` fails to parse or segment.
    pub fn fix_document(&self, source: &str) -> Result<(String, EditReport), EditError> {
        document::edit_document(
            source,
            self.inventory,
            self.attestation,
            &self.tagger,
            &self.segmenter,
        )
    }

    /// `source`'s total prose word-token count, counted the same way this
    /// engine's own per-document pivot budget is scaled — exposed so a
    /// calibration tool measuring a natural pivot rate (patches per 1000
    /// words) uses the identical word-counting convention the budget it
    /// calibrates will later be constrained by.
    ///
    /// # Errors
    /// Returns [`EditError`] if `source` fails to parse or segment.
    pub fn word_count(&self, source: &str) -> Result<usize, EditError> {
        document::prose_word_count(source, &self.segmenter)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_loads_and_is_reusable() {
        let engine = Engine::new().expect("engine must load");
        let (out1, _) = engine.fix_document("Run the scanner.").unwrap();
        let (out2, _) = engine.fix_document("Run the scanner.").unwrap();
        assert_eq!(out1, out2);
    }

    #[test]
    fn near_noop_clean_text_is_byte_identical() {
        let engine = Engine::new().expect("engine must load");
        let input = "Run the scanner from the project root. Results stream in as they are \
                     found, and nothing is deleted without confirmation.";
        let (out, _) = engine.fix_document(input).unwrap();
        assert_eq!(out, input);
    }
}
