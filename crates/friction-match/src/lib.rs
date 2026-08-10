//! Fix-time detection: six independent channels (differential matching
//! statistics, a literal tell inventory, licensed light-verb
//! constructions, deterministic contrast-frame templates, jargon
//! detection — both curated-lexeme metaphor compounds and, when
//! [`friction_packs::general_evidence`] is available, list-free
//! unattested compounds — and per-document word-overuse detection) over
//! a document's prose, reporting byte-honest [`MatchSpan`]s. This crate
//! never rewrites text — see [`MatchEngine`]'s own docs.
//!
//! # Span honesty by construction, not by translation
//!
//! Every regex/automaton match this crate makes runs directly against the
//! *original* source bytes of a prose region, never against a cleaned or
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
//! equivalence test (`tests/token_convention.rs`), not by shared code: the
//! two tokenizers solve the same boundary problem over different inputs
//! (pre-lowercased whole-file text vs. raw, mixed-case, per-prose-block
//! text), so sharing the regex object would be cosmetic, not real
//! deduplication.
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
//! [`dms_spans_pooled`] is a standalone DMS-only entry point: given prose
//! units already scoped by [`token::prose_scope`], it returns the pooled
//! machine-vs-human scan's spans without building a [`MatchEngine`]
//! (which also compiles the literal automaton and requires a
//! tagger/segmenter for the LVC channel). `friction fix` uses it to scan
//! its own fixed output for paraphrase suggestions it detects but never
//! auto-edits — DMS stays banned as an edit judge (`friction-edit`'s own
//! crate docs: every gate is the curated inventory pack and the corpus-
//! attested seam-bigram/skeleton tables, never a metric or genre
//! envelope); this crate still only detects and reports, never rewrites.

mod dms;
mod error;
pub mod frame;
pub mod frame_rewrite;
pub mod jargon;
pub mod jargon_compound;
mod literal;
mod lvc;
pub mod overuse;
pub mod span;
pub mod tagging;
pub mod token;

pub use error::MatchError;
pub use span::{Channel, DmsMachineReport, DmsReport, DocumentReport, MatchScore, MatchSpan};

use friction_core::Document;
use friction_nlp::{Segmenter, Tagger};
use friction_packs::{
    DmsIndexView, HumanEvidencePack, InventoryPack, JargonAttestPack, JargonPack,
};

use crate::token::ScopedUnit;

/// Built once (compiles the literal automaton once) and reused across
/// every document scanned against the same pack/tagger/segmenter.
///
/// This crate only detects and reports spans with frame ids: it never
/// rewrites text. A repair layer built on top of this crate's output is a
/// separate concern.
pub struct MatchEngine<'a> {
    inventory: &'a InventoryPack,
    dms: &'a DmsIndexView<'a>,
    jargon: &'a JargonPack,
    jargon_attest: &'a JargonAttestPack,
    human_evidence: &'a HumanEvidencePack,
    tagger: &'a dyn Tagger,
    segmenter: &'a dyn Segmenter,
    automaton: literal::LiteralAutomaton,
}

impl<'a> MatchEngine<'a> {
    /// Builds an engine bound to `inventory`, `dms`, `jargon`,
    /// `jargon_attest`, `human_evidence`, `tagger`, and `segmenter` for its
    /// whole lifetime: the literal automaton is compiled exactly once
    /// here. [`Self::scan`]'s DMS channel always walks
    /// [`friction_packs::DmsIndexView::pooled_machine_sam`] (see [`dms`]'s
    /// own module docs) — there is no per-family selection to make.
    ///
    /// # Errors
    /// [`MatchError::Automaton`] if the inventory's literal-eligible
    /// patterns fail to compile (should not happen for the embedded pack,
    /// covered by this crate's own tests).
    pub fn new(
        inventory: &'a InventoryPack,
        dms: &'a DmsIndexView<'a>,
        jargon: &'a JargonPack,
        jargon_attest: &'a JargonAttestPack,
        human_evidence: &'a HumanEvidencePack,
        tagger: &'a dyn Tagger,
        segmenter: &'a dyn Segmenter,
    ) -> Result<Self, MatchError> {
        let automaton = literal::LiteralAutomaton::build(inventory)?;
        Ok(Self {
            inventory,
            dms,
            jargon,
            jargon_attest,
            human_evidence,
            tagger,
            segmenter,
            automaton,
        })
    }

    /// Scans `document`, running all six channels over the same
    /// prose-only, in-scope unit set ([`token::prose_scope`]), and
    /// merges their spans deterministically ([`span::merge_spans`]).
    ///
    /// DMS, the literal automaton, its regex fallback, frame, the
    /// document-level DMS report, and the shared tagging step
    /// ([`tagging::tag_units`]) each read only `units`/`document` and this
    /// engine's own packs — never each other's output — so they run as
    /// six concurrent `rayon::scope` tasks (the DMS report is itself a
    /// `par_iter` over `units`, in [`dms::document_report`]). LVC, jargon,
    /// jargon-compound, and overuse all need the tagged sentences that
    /// step produces, so [`Self::tagged_channel_spans`] runs them
    /// afterward, over the same borrowed [`tagging::TaggedSentence`]
    /// slice — one shared tag pass instead of each channel tagging every
    /// sentence on its own (see [`tagging`]'s own module docs).
    /// [`span::merge_spans`] sorts its input before returning, so no
    /// task's completion order, in either stage, ever affects the
    /// returned bytes; the local variables from both stages are still
    /// assembled into `merge_spans`' `Vec` argument (and
    /// `DocumentReport`) in the same fixed order the prior sequential
    /// version used, so nothing here depends on that insensitivity to
    /// stay byte-identical across thread counts.
    ///
    /// # Errors
    /// [`MatchError::Core`] if a prose range fails to slice — not
    /// expected for any `Document` produced by `friction_parse::parse`,
    /// since its ranges are already validated at construction.
    pub fn scan(&self, document: &Document) -> Result<DocumentReport, MatchError> {
        let units = token::prose_scope(document, self.segmenter);

        let machine_sam = self.dms.pooled_machine_sam();
        let human_sam = self.dms.human_sam();
        let vocab = self.dms.vocab();
        let lexicon = self.inventory.lvc_lexicon();

        // rayon cannot spawn OS threads on wasm32-unknown-unknown (no
        // thread creation capability there) — `rayon::scope`/`rayon::join`
        // panic lazily the first time either tries to build the global
        // pool. The six independent computations and the two-stage
        // lvc/jargon/overuse join below run sequentially there instead;
        // this crate's own `rayon_thread_count_determinism` corpus tests
        // already prove `RAYON_NUM_THREADS=1` output is byte-identical to
        // any other thread count, and running these calls one after
        // another inline is exactly that one-thread case. The six
        // `let mut ... = <empty>` bindings `rayon::scope`'s closures
        // write through only exist on the native path — the wasm32
        // block below assigns each once, so it declares its own
        // (non-`mut`) bindings instead of sharing these.
        #[cfg(not(target_arch = "wasm32"))]
        let (dms_spans, dms_report, literal_ac_spans, literal_fallback_spans, frame_spans, tagged) = {
            let mut dms_spans = Vec::new();
            let mut dms_report = None;
            let mut literal_ac_spans = Vec::new();
            let mut literal_fallback_spans = Vec::new();
            let mut frame_spans = Vec::new();
            let mut tagged = Vec::new();
            rayon::scope(|s| {
                s.spawn(|_| {
                    dms_spans = dms::scan_units(&units, machine_sam, human_sam, vocab);
                });
                s.spawn(|_| {
                    dms_report = Some(dms::document_report(&units, self.dms, vocab));
                });
                s.spawn(|_| {
                    literal_ac_spans = self.automaton.scan_units(&units);
                });
                s.spawn(|_| {
                    literal_fallback_spans =
                        literal::regex_fallback_spans(self.inventory, &self.automaton, &units);
                });
                s.spawn(|_| {
                    frame_spans = frame::scan_units(&units);
                });
                s.spawn(|_| {
                    tagged = tagging::tag_units(&units, document.source(), self.tagger);
                });
            });
            (
                dms_spans,
                dms_report,
                literal_ac_spans,
                literal_fallback_spans,
                frame_spans,
                tagged,
            )
        };
        #[cfg(target_arch = "wasm32")]
        let (dms_spans, dms_report, literal_ac_spans, literal_fallback_spans, frame_spans, tagged) = (
            dms::scan_units(&units, machine_sam, human_sam, vocab),
            Some(dms::document_report(&units, self.dms, vocab)),
            self.automaton.scan_units(&units),
            literal::regex_fallback_spans(self.inventory, &self.automaton, &units),
            frame::scan_units(&units),
            tagging::tag_units(&units, document.source(), self.tagger),
        );

        let (lvc_spans, jargon_spans, jargon_compound_spans, overuse_spans) =
            self.tagged_channel_spans(&tagged, document.source(), lexicon);

        let spans = span::merge_spans(vec![
            dms_spans,
            literal_ac_spans,
            literal_fallback_spans,
            lvc_spans,
            frame_spans,
            jargon_spans,
            jargon_compound_spans,
            overuse_spans,
        ]);

        Ok(DocumentReport {
            spans,
            dms: dms_report.expect("every rayon::scope task, including the DMS report one, has run by the time scope() returns"),
        })
    }

    /// The four channels that need `tagged` ([`tagging::tag_units`]'s
    /// output) rather than the raw prose units [`Self::scan`]'s first
    /// stage already covers: LVC, jargon (`jargon.metaphor`),
    /// jargon-compound ([`jargon_compound::scan_units`]), and overuse, in
    /// that order — the fixed order [`Self::scan`] assembles into
    /// `merge_spans`' argument, so this returns them as a plain tuple
    /// rather than leaving the caller to destructure a nested `rayon::join`
    /// result.
    ///
    /// Three nested `rayon::join` pairs on native targets (jargon-compound
    /// and overuse share the innermost pair, since neither depends on the
    /// other); a plain sequential tuple on wasm32, where rayon cannot spawn
    /// OS threads — the same substitution [`Self::scan`]'s own first stage
    /// uses, and byte-identical for the same reason (see that stage's own
    /// comment).
    fn tagged_channel_spans(
        &self,
        tagged: &[tagging::TaggedSentence],
        document_text: &str,
        lexicon: &std::collections::BTreeMap<Box<str>, Box<str>>,
    ) -> (
        Vec<MatchSpan>,
        Vec<MatchSpan>,
        Vec<MatchSpan>,
        Vec<MatchSpan>,
    ) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let (lvc_spans, (jargon_spans, (jargon_compound_spans, overuse_spans))) = rayon::join(
                || lvc::scan_units(tagged, document_text, lexicon),
                || {
                    rayon::join(
                        || {
                            jargon::scan_units(
                                tagged,
                                document_text,
                                self.jargon,
                                self.jargon_attest,
                            )
                        },
                        || {
                            rayon::join(
                                || {
                                    jargon_compound::scan_units(
                                        tagged,
                                        document_text,
                                        self.jargon_attest,
                                    )
                                },
                                || overuse::scan_units(tagged, document_text, self.human_evidence),
                            )
                        },
                    )
                },
            );
            (
                lvc_spans,
                jargon_spans,
                jargon_compound_spans,
                overuse_spans,
            )
        }
        #[cfg(target_arch = "wasm32")]
        {
            (
                lvc::scan_units(tagged, document_text, lexicon),
                jargon::scan_units(tagged, document_text, self.jargon, self.jargon_attest),
                jargon_compound::scan_units(tagged, document_text, self.jargon_attest),
                overuse::scan_units(tagged, document_text, self.human_evidence),
            )
        }
    }
}

/// Returns [`Channel::Dms`] spans from the pooled machine-vs-human scan.
///
/// Scans `units` (already scoped by [`token::prose_scope`]) — every
/// present family stream `dms` defines, concatenated into one automaton
/// (see [`dms`]'s own module docs), against the same pack's human
/// baseline stream. Family attribution is retired: every span carries
/// the constant `"dms.machine"` frame id.
///
/// A standalone entry point for callers that need only the DMS channel
/// (e.g. `friction fix`'s paraphrase-suggestion report) without paying
/// for [`MatchEngine`]'s literal-automaton compilation or running the
/// literal/LVC channels they don't need.
#[must_use]
pub fn dms_spans_pooled(units: &[ScopedUnit<'_>], dms: &DmsIndexView<'_>) -> Vec<MatchSpan> {
    let machine_sam = dms.pooled_machine_sam();
    let human_sam = dms.human_sam();
    let vocab = dms.vocab();
    dms::scan_units(units, machine_sam, human_sam, vocab)
}
