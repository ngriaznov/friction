//! The public fix entry point: [`fix_document`] (a thin wrapper around
//! [`crate::run_fixpoint`] that supplies the fixed tranche-1 rule set) and
//! [`FixEngine`] (a ready-to-use, model-loaded-once handle for callers
//! that don't want to build a segmenter, tagger, and envelope pack
//! themselves for every document).

use friction_nlp::{NlpruleTagger, Segmenter, SrxSegmenter, TagError, Tagger};
use friction_packs::{ENVELOPE_V2, EnvelopePack};
use friction_rules::{
    ConnectiveSurgery, ContractionRule, FillerPhraseRule, GenreEnvelope, Rule, SubstitutionRule,
};

use crate::driver::{ApplyError, FixpointReport, run_fixpoint};

// Every tranche-1 rule is a zero-sized, `const fn new()`-constructible
// type (see each family's own module docs), so the registered set can be
// plain `'static` statics — no per-call allocation, no interior state to
// race across threads.
static CONNECTIVE_SURGERY: ConnectiveSurgery = ConnectiveSurgery::new();
static CONTRACTION: ContractionRule = ContractionRule::new();
static FILLER_PHRASE: FillerPhraseRule = FillerPhraseRule::new();
static SUBSTITUTION: SubstitutionRule = SubstitutionRule::new();

/// The full tranche-1 rule set (lexical, connective, contraction), in a
/// fixed, deterministic registration order.
///
/// Registration order here is documentation, not a correctness lever:
/// every candidate patch from every active rule is pooled and
/// conflict-resolved together by [`crate::resolve_round`] before anything
/// is applied, so the *output* of a round never depends on which order
/// `rules` were passed to [`run_fixpoint`] in (see that function's own
/// docs). The order below simply follows `RuleFamily::priority`'s
/// documented order (`Connective` before `Lexical` before `Contraction`,
/// the three families tranche-1 covers), with the two `Lexical` rules
/// ordered filler-then-substitution to match `friction_rules::families::
/// lexical`'s own module layout.
#[must_use]
pub fn registered_rules() -> [&'static dyn Rule; 4] {
    [
        &CONNECTIVE_SURGERY,
        &FILLER_PHRASE,
        &SUBSTITUTION,
        &CONTRACTION,
    ]
}

/// Runs the fixpoint driver over `source` with the full tranche-1
/// [`registered_rules`] set: parse -> metrics -> gate -> scan -> fix ->
/// resolve conflicts -> apply, up to [`crate::MAX_ROUNDS`] rounds.
///
/// A thin layer above [`run_fixpoint`] — the only thing this function
/// adds is supplying the fixed, deterministically-ordered rule set, so
/// every caller (CLI, corpus tooling, tests) exercises the exact same
/// rule list rather than each assembling its own.
///
/// `envelope` is expected to be backed by a loaded `friction-packs`
/// envelope pack for `genre` (see [`FixEngine`] for a convenience handle
/// that loads one, plus the model-backed `segmenter`/`tagger`, once and
/// reuses them across many calls).
///
/// # Errors
/// See [`run_fixpoint`].
pub fn fix_document(
    source: &str,
    genre: &str,
    envelope: &dyn GenreEnvelope,
    segmenter: &dyn Segmenter,
    tagger: &dyn Tagger,
) -> Result<(String, FixpointReport), ApplyError> {
    run_fixpoint(
        source,
        &registered_rules(),
        segmenter,
        tagger,
        genre,
        envelope,
    )
}

/// A [`GenreEnvelope`] view over one genre's slice of a loaded
/// [`EnvelopePack`].
struct PackEnvelope<'a> {
    pack: &'a EnvelopePack,
    genre: &'a str,
}

impl GenreEnvelope for PackEnvelope<'_> {
    fn band(&self, metric: &str) -> Option<friction_core::Envelope> {
        self.pack.band(self.genre, metric)
    }
}

/// Errors produced while building a [`FixEngine`].
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum EngineError {
    /// The embedded English part-of-speech tagger model failed to load.
    #[error("failed to load the embedded English tagger model: {0}")]
    Tagger(#[from] TagError),
}

/// A ready-to-use [`fix_document`] handle.
///
/// Loads the sentence segmenter, part-of-speech tagger, and the shipped
/// `envelope-v2` pack once (via [`FixEngine::new`]), then reuses all
/// three across as many [`FixEngine::fix_document`] calls as the caller
/// likes.
///
/// This is the shape corpus-scale callers (an idempotence sweep, a
/// near-no-op report, `friction-cli`'s eventual `fix` subcommand) want:
/// [`NlpruleTagger::new`] loads an embedded model and is not something to
/// pay for once per document.
pub struct FixEngine {
    segmenter: SrxSegmenter,
    tagger: NlpruleTagger,
    envelope_pack: &'static EnvelopePack,
}

impl FixEngine {
    /// Loads the tagger model and builds a `FixEngine` backed by the
    /// shipped `envelope-v2` pack ([`friction_packs::ENVELOPE_V2`]).
    ///
    /// # Errors
    /// Returns [`EngineError`] if the embedded English tagger model fails
    /// to load.
    pub fn new() -> Result<Self, EngineError> {
        Ok(Self {
            segmenter: SrxSegmenter::new(),
            tagger: NlpruleTagger::new()?,
            envelope_pack: &ENVELOPE_V2,
        })
    }

    /// Runs [`fix_document`] over `source` for `genre`, using this
    /// engine's already-loaded segmenter, tagger, and envelope pack.
    ///
    /// # Errors
    /// See [`run_fixpoint`].
    pub fn fix_document(
        &self,
        source: &str,
        genre: &str,
    ) -> Result<(String, FixpointReport), ApplyError> {
        let envelope = PackEnvelope {
            pack: self.envelope_pack,
            genre,
        };
        fix_document(source, genre, &envelope, &self.segmenter, &self.tagger)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stub tagger that tags nothing — fine for `fix_document`'s test,
    /// none of the four registered rules' gates in that test consult
    /// part-of-speech tags.
    struct NoopTagger;
    impl Tagger for NoopTagger {
        fn tag(&self, _text: &str, _base_offset: usize) -> Vec<friction_nlp::TaggedToken> {
            Vec::new()
        }
    }

    /// The registered rule set is exactly the four tranche-1 rules, each
    /// appearing once, by id.
    #[test]
    fn registered_rules_are_the_four_tranche_one_rules_once_each() {
        let mut ids: Vec<&str> = registered_rules().iter().map(|r| r.id().as_str()).collect();
        ids.sort_unstable();
        assert_eq!(
            ids,
            vec![
                "connective.surgery",
                "contraction.insert",
                "lexical.filler_phrase",
                "lexical.substitution",
            ]
        );
    }

    /// `fix_document` (the free function) actually runs the registered
    /// rules: a document that leans on a heavy sentence-initial
    /// connective and sits below the genre's contraction floor gets both
    /// addressed in one call, given bands wide enough to license it.
    #[test]
    fn fix_document_runs_the_registered_rule_set() {
        use friction_core::Envelope;
        use friction_nlp::SrxSegmenter;
        use friction_rules::MapEnvelope;

        let segmenter = SrxSegmenter::new();
        let tagger = NoopTagger;
        let envelope = MapEnvelope::new()
            .with("discourse_marker_density", Envelope::new(0.0, 0.0))
            .with("contraction_ratio", Envelope::new(0.9, 1.0));

        let (output, report) = fix_document(
            "Moreover, it is not ready yet.",
            "blog",
            &envelope,
            &segmenter,
            &tagger,
        )
        .expect("well-formed input must not fail");

        assert!(report.total_patches_applied() > 0);
        assert_ne!(output, "Moreover, it is not ready yet.");
    }
}
