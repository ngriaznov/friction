//! The friction v4 repair engine: five operations (ritual deletion, paired
//! substitution, derivational pivot, gated span deletion, frame-gated
//! `just`-deletion) applied per sentence, in fixed order, gated by the
//! curated inventory pack and the corpus-attested seam-bigram/skeleton
//! tables — never by a metric or genre envelope.
//!
//! [`Engine`] is the top-level entry point: loads the embedded tagger,
//! segmenter, inventory pack, and attestation pack once; `fix_document`
//! runs the bounded two-pass pipeline (`document::edit_document`) over a
//! document's prose.
//!
//! No synthesis: every emitted [`friction_core::Patch`] either deletes,
//! substitutes a fixed pack string, or derives a verb via
//! `friction_nlp::lvc::conjugate`'s static morphology. A failed gate
//! holds the candidate (Suggest-tier) and leaves the source bytes untouched.

pub mod document;
mod error;
pub mod gates;
pub mod nearnoop;
pub mod register;
pub mod sentence;
pub mod splice;

pub use document::{EditReport, PassReport, edit_document};
pub use error::EditError;
pub use register::{RegisterSentenceTags, ReusableScan};

use friction_nlp::{PerceptronParser, PerceptronTagger, SrxSegmenter};
use friction_packs::{AttestationPack, InventoryPack, RegisterPack};

/// The engine: the embedded tagger, dependency parser, segmenter,
/// inventory pack, attestation pack, and register pack, loaded once and
/// reused across every [`Engine::fix_document`] call.
pub struct Engine {
    inventory: &'static InventoryPack,
    attestation: &'static AttestationPack,
    register_pack: &'static RegisterPack,
    tagger: PerceptronTagger,
    parser: PerceptronParser,
    segmenter: SrxSegmenter,
}

impl Engine {
    /// Builds an engine over the embedded, process-lifetime inventory,
    /// attestation, and register packs.
    ///
    /// # Errors
    /// Returns [`EditError::Tagger`] if the embedded tagger weights fail
    /// to load, or [`EditError::Parser`] if the embedded dependency-parser
    /// weights fail to load.
    pub fn new() -> Result<Self, EditError> {
        let tagger = PerceptronTagger::new()?;
        let parser = PerceptronParser::new()?;
        Ok(Self {
            inventory: &friction_packs::INVENTORY.pack,
            attestation: &friction_packs::ATTESTATION.pack,
            register_pack: &friction_packs::REGISTER.pack,
            tagger,
            parser,
            segmenter: SrxSegmenter::new(),
        })
    }

    /// Runs the bounded two-pass repair pipeline over `source`, followed
    /// by the register pass, returning the fixed text and a report of
    /// what happened each pass (the register pass is the report's final
    /// entry — see [`crate::document::edit_document`]'s own docs).
    ///
    /// # Errors
    /// Returns [`EditError`] if `source` fails to parse or segment.
    pub fn fix_document(&self, source: &str) -> Result<(String, EditReport), EditError> {
        self.fix_document_with(source, friction_parse::Syntax::Markdown)
    }

    /// [`Self::fix_document`] with the surface syntax chosen by the
    /// caller — every pass in the pipeline (including register's own
    /// re-parses) reads `source` and its intermediate texts as `syntax`.
    ///
    /// # Errors
    /// Returns [`EditError`] if `source` fails to parse or segment.
    pub fn fix_document_with(
        &self,
        source: &str,
        syntax: friction_parse::Syntax,
    ) -> Result<(String, EditReport), EditError> {
        document::edit_document(
            source,
            syntax,
            self.inventory,
            self.attestation,
            self.register_pack,
            &self.tagger,
            &self.parser,
            &self.segmenter,
        )
    }

    /// `source`'s total prose word-token count, counted the same way this
    /// engine's per-document pivot budget is scaled — exposed so a
    /// calibration tool measuring pivot rate (patches per 1000 words)
    /// uses the same word-counting convention the budget uses.
    ///
    /// # Errors
    /// Returns [`EditError`] if `source` fails to parse or segment.
    pub fn word_count(&self, source: &str) -> Result<usize, EditError> {
        document::prose_word_count(source, &self.segmenter)
    }

    /// This engine's embedded tagger, as a trait object.
    ///
    /// Lets a caller that already holds an [`Engine`] (`friction fix`'s
    /// paraphrase scan, which needs part-of-speech tags for its jargon
    /// channel — see `friction-cli`'s `fix` module) reuse this engine's
    /// already-loaded tagger instead of constructing a second
    /// `friction_nlp::PerceptronTagger` of its own.
    #[must_use]
    pub fn tagger(&self) -> &dyn friction_nlp::Tagger {
        &self.tagger
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
