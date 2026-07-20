//! The public fix entry point: [`fix_document`] (a thin wrapper around
//! [`crate::run_fixpoint`] that supplies the fixed, six-family
//! [`registered_rules`] set) and [`FixEngine`] (a ready-to-use,
//! model-loaded-once handle for callers that don't want to build a
//! segmenter, tagger, and envelope pack themselves for every document).

use friction_nlp::{NlpruleTagger, Segmenter, SrxSegmenter, TagError, Tagger};
use friction_packs::{ENVELOPE_V2, EnvelopePack};
use friction_rules::{
    BoldLabelStripRule, ConnectiveSurgery, ContractionRule, FillerPhraseRule, GenreEnvelope,
    HeaderMergeRule, NotJustButRule, ParticipialCloserRule, RitualConclusionRule, Rule,
    SentenceFuseRule, SentenceSplitRule, SubstitutionRule, TriadReductionRule, UnbulletRule,
};

use crate::driver::{ApplyError, FixpointReport, run_fixpoint};

// Every registered rule is a zero-sized, `const fn new()`-constructible
// type (see each family's own module docs), so the registered set can be
// plain `'static` statics — no per-call allocation, no interior state to
// race across threads.
static BOLD_LABEL_STRIP: BoldLabelStripRule = BoldLabelStripRule::new();
static HEADER_MERGE: HeaderMergeRule = HeaderMergeRule::new();
static UNBULLET: UnbulletRule = UnbulletRule::new();
static NOT_JUST_BUT: NotJustButRule = NotJustButRule::new();
static PARTICIPIAL_CLOSER: ParticipialCloserRule = ParticipialCloserRule::new();
static RITUAL_CONCLUSION: RitualConclusionRule = RitualConclusionRule::new();
static TRIAD_REDUCTION: TriadReductionRule = TriadReductionRule::new();
static CONNECTIVE_SURGERY: ConnectiveSurgery = ConnectiveSurgery::new();
static CONTRACTION: ContractionRule = ContractionRule::new();
static FILLER_PHRASE: FillerPhraseRule = FillerPhraseRule::new();
static SUBSTITUTION: SubstitutionRule = SubstitutionRule::new();
static SENTENCE_FUSE: SentenceFuseRule = SentenceFuseRule::new();
static SENTENCE_SPLIT: SentenceSplitRule = SentenceSplitRule::new();

/// The full six-family rule set (structural, symmetry, connective,
/// lexical, rhythm, contraction), in a fixed, deterministic registration
/// order.
///
/// Registration order here is documentation, not a correctness lever:
/// every candidate patch from every active rule is pooled and
/// conflict-resolved together by [`crate::resolve_round`] before anything
/// is applied, so the *output* of a round never depends on which order
/// `rules` were passed to [`run_fixpoint`] in (see that function's own
/// docs) — including the `Suggest`-only rules below, which never produce
/// a patch at all and so never participate in that resolution either way.
/// The order below simply follows [`friction_rules::RuleFamily::
/// priority`]'s documented order (`Structural, Symmetry, Connective,
/// Lexical, Rhythm, Contraction`), with each family's own rules ordered to
/// match that family's module (`pub use`) layout: structural as
/// bold-label-strip, header-merge, unbullet; symmetry as not-just-but,
/// participial-closer, ritual-conclusion, triad-reduction; lexical as
/// filler-then-substitution; rhythm as fuse-then-split.
///
/// Three rules here — [`NotJustButRule`] and [`TriadReductionRule`]
/// (always `Suggest` tier) and [`SentenceFuseRule`] (`Suggest` tier,
/// `Gate::Detect` only) — never produce a [`friction_core::Patch`]; they
/// are registered anyway so their findings are scanned and surfaced
/// through the normal round pipeline (see [`crate::driver::RoundReport::
/// findings`]), exactly like every other rule's diagnostics.
#[must_use]
pub fn registered_rules() -> [&'static dyn Rule; 13] {
    [
        &BOLD_LABEL_STRIP,
        &HEADER_MERGE,
        &UNBULLET,
        &NOT_JUST_BUT,
        &PARTICIPIAL_CLOSER,
        &RITUAL_CONCLUSION,
        &TRIAD_REDUCTION,
        &CONNECTIVE_SURGERY,
        &FILLER_PHRASE,
        &SUBSTITUTION,
        &SENTENCE_FUSE,
        &SENTENCE_SPLIT,
        &CONTRACTION,
    ]
}

/// Runs the fixpoint driver over `source` with the full six-family
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
    /// since the envelope bands that test supplies only put
    /// `lexical`/`connective`/`contraction` rules into `Gate::Fix` (every
    /// other family gates `Off` for lack of a band — see
    /// `registered_rules_are_the_thirteen_six_family_rules_once_each`'s
    /// sibling test below for the full set), and none of those three
    /// families' gates consult part-of-speech tags. Families that do
    /// (e.g. `symmetry.participial_closer`) are exercised against a real
    /// tagger in their own family tests instead.
    struct NoopTagger;
    impl Tagger for NoopTagger {
        fn tag(&self, _text: &str, _base_offset: usize) -> Vec<friction_nlp::TaggedToken> {
            Vec::new()
        }
    }

    /// The registered rule set is exactly the thirteen rules across all
    /// six families, each appearing once, by id.
    #[test]
    fn registered_rules_are_the_thirteen_six_family_rules_once_each() {
        let mut ids: Vec<&str> = registered_rules().iter().map(|r| r.id().as_str()).collect();
        ids.sort_unstable();
        assert_eq!(
            ids,
            vec![
                "connective.surgery",
                "contraction.insert",
                "lexical.filler_phrase",
                "lexical.substitution",
                "rhythm.fuse",
                "rhythm.split",
                "structural.bold_label_strip",
                "structural.header_merge",
                "structural.unbullet",
                "symmetry.not_just_but",
                "symmetry.participial_closer",
                "symmetry.ritual_conclusion",
                "symmetry.triad_reduction",
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

    /// A `Suggest`-tier finding from a registered rule
    /// (`symmetry.not_just_but`, which never proposes a patch — see that
    /// family's own module docs) survives all the way out through
    /// `fix_document`'s public [`FixpointReport`], with its original span
    /// and message intact, and applies no patch: the report is a
    /// diagnostics channel for `Suggest` findings, not just a record of
    /// what got fixed.
    #[test]
    fn suggest_tier_findings_surface_in_the_report_with_span_and_message() {
        use friction_core::{Envelope, Tier};
        use friction_nlp::SrxSegmenter;
        use friction_rules::MapEnvelope;

        let segmenter = SrxSegmenter::new();
        let tagger = NoopTagger;
        // A band whose `hi` sits at zero forces `symmetry.not_just_but`
        // out of its envelope on any match at all, gating `Detect` (see
        // that rule's own `gate`) — every other registered rule has no
        // band in this `MapEnvelope` and so gates `Off`, isolating the
        // one finding this test cares about.
        let envelope = MapEnvelope::new().with("not_just_but_rate", Envelope::new(0.0, 0.0));
        let source = "This release is not just fast but also reliable.";

        let (output, report) = fix_document(source, "blog", &envelope, &segmenter, &tagger)
            .expect("well-formed input must not fail");

        // No patch: a Suggest-only rule proposes none, and no other rule
        // was gated on at all.
        assert_eq!(output, source, "a Suggest-only finding must apply no patch");
        assert_eq!(report.total_patches_applied(), 0);

        let findings = &report.rounds[0].findings;
        let suggestion = findings
            .iter()
            .find(|f| f.rule.as_str() == "symmetry.not_just_but")
            .expect("symmetry.not_just_but must have scanned and surfaced a finding");
        assert_eq!(suggestion.tier, Tier::Suggest);
        assert_eq!(
            &source[suggestion.range.clone()],
            "not just fast but also",
            "the finding's span must index the original source text it flagged"
        );
        assert!(
            !suggestion.message.is_empty(),
            "a surfaced finding must carry a human-readable message"
        );
    }

    /// Regression test for the finding that `structural.unbullet`
    /// (priority 0, round 1) joining a three-item list into one
    /// serial-comma sentence, then `rhythm.split` (priority 4) re-parsing
    /// that sentence in round 2, used to compose into a comma splice: the
    /// only boundary `rhythm.split` could find was the final `", and "`,
    /// and splitting only there left the first two clauses joined by
    /// nothing but a bare comma. `rhythm.split`'s own
    /// `precedes_unresolved_comma` check (see
    /// `friction_rules::families::rhythm::split`'s module docs) now
    /// excludes exactly that boundary, so the joined sentence — over-long
    /// as it is — is left as one grammatically valid sentence rather than
    /// being cut into a splice.
    #[test]
    fn unbullet_then_split_across_rounds_never_produces_a_comma_splice() {
        use friction_core::Envelope;
        use friction_rules::MapEnvelope;

        let segmenter = SrxSegmenter::new();
        let tagger = NlpruleTagger::new().expect("embedded model loads");
        // Forces `structural.unbullet` (any list at all) and
        // `rhythm.split` (any document, however low its real
        // `sentence_length_cv`) both into `Gate::Fix`; every other
        // registered rule has no band here and gates `Off`, isolating
        // exactly the two-rule, two-round composition this test cares
        // about.
        let envelope = MapEnvelope::new()
            .with("list_item_density", Envelope::new(0.0, 0.0))
            .with("sentence_length_cv", Envelope::new(5.0, 10.0));
        let source = "- Configuration handles environment setup automatically for every new \
                       team member\n\
                       - Installation handles every dependency without manual intervention \
                       required\n\
                       - Kubernetes handles the rollout to every environment automatically \
                       once approved\n";

        let (output, _report) = fix_document(source, "docs", &envelope, &segmenter, &tagger)
            .expect("well-formed input must not fail");

        assert!(
            !output.contains("required. Kubernetes"),
            "must never split the serial list into a comma splice, got {output:?}"
        );
        assert_eq!(
            output,
            "Configuration handles environment setup automatically for every new team \
             member, installation handles every dependency without manual intervention \
             required, and Kubernetes handles the rollout to every environment \
             automatically once approved.\n",
            "structural.unbullet should still join the list; rhythm.split must decline \
             every boundary rather than introduce a splice"
        );
    }
}
