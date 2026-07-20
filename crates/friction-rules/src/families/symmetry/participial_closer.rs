//! Participial-closer rewriting: a sentence-final `", <VBG-phrase>"` clause
//! tacked onto an otherwise-complete sentence (`"The team shipped the
//! release, making it easier to onboard new users."`) — an LLM tic this
//! rule either deletes outright or promotes into its own sentence.
//!
//! # Detection
//!
//! [`find_closure`] mirrors `friction-metrics::symmetry::is_participial_
//! closer`'s exact predicate (private to that crate, so not importable —
//! see `families::symmetry`'s own module docs for why every submodule here
//! mirrors rather than imports): strip any trailing strong-boundary tokens
//! (`.`/`!`/`?`/`;`/`:`), find the sentence's last remaining comma, and
//! require a present participle (`VBG`) immediately after it, with at
//! least one token before the comma (a main clause to attach to). This
//! module's `find_closure_matches_metrics_predicate_on_shared_fixtures`
//! test pins that mirror down against hand-built token fixtures that
//! reproduce `friction-metrics::symmetry`'s own test cases exactly.
//!
//! # Mixed tier, per finding
//!
//! A sentence-final participial closer can be resolved two ways:
//!
//! - **Promote**: split the closer clause into its own sentence, with a
//!   `"This"` subject and the participle's own lemma inflected to agree
//!   with it (`friction_nlp::inflect`, third-person-singular present).
//!   `"...the release, making it easier to onboard new users."` ->
//!   `"...the release. This makes it easier to onboard new users."` This
//!   carries the closer clause's own claim forward intact — nothing is
//!   dropped — so it is the only strategy this rule ever applies
//!   automatically, and only when the participle has a clear object or
//!   complement immediately following it to promote (see
//!   [`has_object_like_continuation`]) and its lemma inflects
//!   unambiguously (see [`promote_patch`]).
//! - **Delete**: remove the comma through the closer clause, keeping the
//!   sentence's own trailing punctuation. `"...the release, making it
//!   easier to onboard new users."` -> `"...the release."` A purely
//!   syntactic scan cannot tell a decorative closer (safe to drop) from
//!   one carrying the sentence's *only* concrete claim (e.g. `"...,
//!   exposing a single point of failure in the power supply."`) — deleting
//!   the latter silently erases it. Because this rule cannot make that
//!   judgment safely, **delete is never applied automatically**: whenever
//!   promote is not available (no object-like continuation, or an
//!   unreliable lemma), the finding is [`friction_core::Tier::Suggest`]
//!   with no patch — the same "cannot rule out dropping a proposition"
//!   posture `TriadReductionRule`/`NotJustButRule` already take in this
//!   family — and [`ParticipialCloserRule::fix`] declines it.
//!
//! So this rule's tier is a per-finding runtime decision, not a fixed
//! per-rule constant — the same shape `RitualConclusionRule` uses.
//!
//! # Idempotence
//!
//! **Promote** turns the closer into an ordinary declarative sentence
//! (`"This <verb> ..."`) with no sentence-final comma-`VBG` shape of its
//! own, so a second scan cannot rematch it — checked directly by this
//! module's `fixing_a_document_is_idempotent` test. A **Suggest**-tier
//! finding never produces a patch at all, so it trivially cannot introduce
//! new matches either.
//!
//! # Exact, per-round budgeting
//!
//! Same shape as `families::contraction::ContractionRule`'s own "Exact,
//! per-round budgeting" section: [`Rule::gate`] runs before `scan` sees the
//! real document, so it cannot know [`GATED_METRIC`]'s exact per-fix effect
//! (`1 / sentence_count`, a number that depends on the document); it hands
//! back a generous safety-cap budget whenever the round is above the
//! envelope, and [`ParticipialCloserRule::fix`] — which does have the real
//! document — computes the exact number of fixes needed to bring the rate
//! back down to (never below) the envelope's ceiling, declining every
//! finding past that count, leftmost-first.

use friction_core::{Finding, MetricVector, Patch, RuleId, Sentence, Tier, span};
use friction_nlp::{TaggedToken, inflect};

use crate::budget::Budget;
use crate::context::{GenreEnvelope, RuleContext};
use crate::rule::{Gate, Rule, RuleFamily};
use crate::strategy::StrategyRng;

/// This rule's stable identifier.
const RULE_ID: RuleId = RuleId::new("symmetry.participial_closer");

/// The [`MetricVector`] field this rule gates on.
const GATED_METRIC: &str = "participial_closer_rate";

/// The per-round budget [`ParticipialCloserRule::gate`] hands back whenever
/// the round is above the envelope at all. See the module docs' "Exact,
/// per-round budgeting" section — the real limit is computed in `fix`, from
/// the real document.
const GATE_SAFETY_CAP: usize = 1000;

/// The exact surface text `token` addresses in `source`, or `""` if its
/// span is somehow invalid.
fn token_text<'s>(source: &'s str, token: &TaggedToken) -> &'s str {
    span::slice(source, &token.token.range).unwrap_or("")
}

fn is_comma(token: &TaggedToken, source: &str) -> bool {
    token_text(source, token) == ","
}

/// Sentence-internal strong punctuation that bounds a clause.
fn is_strong_boundary(token: &TaggedToken, source: &str) -> bool {
    matches!(token_text(source, token), "." | "!" | "?" | ";" | ":")
}

/// One sentence-final participial-closer match: the token indices of its
/// comma and participle, plus the index of the first trailing
/// strong-boundary token (or `tokens.len()` if the sentence has none).
#[allow(clippy::struct_field_names)]
struct ClosureMatch {
    comma_index: usize,
    participle_index: usize,
    boundary_index: usize,
}

impl ClosureMatch {
    /// Byte offset of the comma itself — the start of everything this
    /// rule's two strategies replace.
    fn comma_start(&self, tokens: &[TaggedToken]) -> usize {
        tokens[self.comma_index].token.range.start
    }

    /// Byte offset just past the participle token.
    fn participle_end(&self, tokens: &[TaggedToken]) -> usize {
        tokens[self.participle_index].token.range.end
    }

    /// Byte offset where the closer clause ends: the start of the first
    /// trailing strong-boundary token, or `sentence_end` if the sentence
    /// has none.
    fn boundary_start(&self, tokens: &[TaggedToken], sentence_end: usize) -> usize {
        tokens
            .get(self.boundary_index)
            .map_or(sentence_end, |t| t.token.range.start)
    }
}

/// Finds a sentence-final participial closer in one sentence's tagged
/// tokens. See the module docs' "Detection" section for the exact
/// predicate this mirrors.
fn find_closure(tokens: &[TaggedToken], source: &str) -> Option<ClosureMatch> {
    let mut end = tokens.len();
    while end > 0 && is_strong_boundary(&tokens[end - 1], source) {
        end -= 1;
    }
    if end == 0 {
        return None;
    }
    let comma_index = (0..end).rev().find(|&i| is_comma(&tokens[i], source))?;
    if comma_index == 0 {
        return None;
    }
    let participle_index = comma_index + 1;
    if participle_index >= end || tokens[participle_index].pos.as_str() != "VBG" {
        return None;
    }
    Some(ClosureMatch {
        comma_index,
        participle_index,
        boundary_index: end,
    })
}

/// `true` if the token immediately after the participle looks like the
/// start of an object or complement phrase (a determiner, pronoun, or
/// noun) — the conservative signal [`promote_patch`] requires before
/// attempting the **promote** strategy at all. A participle with nothing
/// of the sort right after it (typically the sentence's own end, or a
/// bare adverb) has no clear complement to carry into a promoted sentence
/// on its own — deleting it outright cannot be verified safe (see the
/// module docs' "Mixed tier, per finding" section), so both [`scan`] and
/// [`ParticipialCloserRule::fix`] treat that case as Suggest tier with no
/// patch.
fn has_object_like_continuation(tokens: &[TaggedToken], closure: &ClosureMatch) -> bool {
    tokens
        .get(closure.participle_index + 1)
        .is_some_and(|token| {
            let pos = token.pos.as_str();
            pos.starts_with("NN") || pos.starts_with("PRP") || pos == "DT"
        })
}

/// `true` if [`promote_patch`] can produce a content-preserving patch for
/// `closure` — the sole condition under which this rule ever fixes
/// automatically. Both [`ParticipialCloserRule::scan`] (to decide a
/// finding's tier) and [`ParticipialCloserRule::fix`] (to decide whether to
/// actually build the patch) call this rather than duplicating the check,
/// so the two can never disagree about which findings are safe to apply.
fn is_safely_promotable(tokens: &[TaggedToken], closure: &ClosureMatch) -> bool {
    has_object_like_continuation(tokens, closure) && promote_patch(tokens, closure).is_some()
}

/// The **promote** strategy's patch, or `None` if the participle's own
/// lemma does not look reliably lemmatized (see below) — the caller treats
/// that as "no safe fix" rather than falling back to deleting the clause
/// (see the module docs' "Mixed tier, per finding" section).
///
/// Replaces the comma through the participle itself with `". This
/// <inflected-verb>"`, leaving everything after the participle (the
/// object/complement phrase this strategy is only attempted when present,
/// plus the sentence's own trailing punctuation) untouched in the source.
/// The inflected verb comes from [`friction_nlp::inflect`]: `"uses"` (an
/// unambiguously third-person-singular-present, all-lowercase template
/// surface form) applied to the participle's own tagger-assigned lemma
/// reuses `inflect`'s own suffix/irregular-verb rules rather than
/// duplicating them, the same technique
/// `families::lexical::substitution::surface_forms` already uses in this
/// workspace.
///
/// If the participle's lemma itself still ends in `"ing"` — the shape a
/// tagger falls back to when it has no dictionary entry for the word (see
/// [`friction_nlp::TaggedToken::lemma`]'s own docs) — inflecting it would
/// silently produce a garbled result (`"makings"` instead of `"makes"`)
/// rather than a wrong-but-plausible one, so this is treated as "no safe
/// fix", not attempted.
fn promote_patch(tokens: &[TaggedToken], closure: &ClosureMatch) -> Option<Patch> {
    let participle = &tokens[closure.participle_index];
    if participle.lemma.ends_with("ing") {
        return None;
    }
    let inflected = inflect("uses", &participle.lemma)?;
    let start = closure.comma_start(tokens);
    let end = closure.participle_end(tokens);
    Some(Patch::new(
        start..end,
        format!(". This {inflected}"),
        RULE_ID,
        Tier::Fix,
    ))
}

/// Finds the sentence containing `finding`'s range, plus that sentence's
/// tagged tokens — used by `fix` to rebuild the closure match it needs to
/// construct a patch (patches are never cached across the `scan`/`fix`
/// split; see `crate::Rule::fix`'s own docs).
fn sentence_and_tokens_for<'a>(
    ctx: &RuleContext<'a>,
    finding: &Finding,
) -> Option<(&'a Sentence, Vec<TaggedToken>)> {
    ctx.sentences().find_map(|(_, sentence)| {
        span::contains_range(&sentence.range, &finding.range)
            .then(|| (sentence, ctx.tag_sentence(sentence)))
    })
}

/// Deletes, or promotes to its own sentence, a sentence-final
/// present-participle closer clause. See the module docs for both
/// strategies and this rule's exact per-round budgeting.
#[derive(Debug, Clone, Copy, Default)]
pub struct ParticipialCloserRule;

impl ParticipialCloserRule {
    /// Creates the rule.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Rule for ParticipialCloserRule {
    fn id(&self) -> RuleId {
        RULE_ID
    }

    fn family(&self) -> RuleFamily {
        RuleFamily::Symmetry
    }

    fn gate(&self, metrics: &MetricVector, envelope: &dyn GenreEnvelope) -> Gate {
        let Some(band) = envelope.band(GATED_METRIC) else {
            return Gate::Off;
        };
        if metrics.participial_closer_rate <= band.hi {
            return Gate::Off;
        }
        Gate::Fix {
            budget: Budget::new(GATE_SAFETY_CAP),
        }
    }

    fn scan(&self, ctx: &RuleContext<'_>) -> Vec<Finding> {
        let source = ctx.document().source();
        let mut findings = Vec::new();
        for (_, sentence) in ctx.sentences() {
            let tokens = ctx.tag_sentence(sentence);
            let Some(closure) = find_closure(&tokens, source) else {
                continue;
            };
            let start = closure.comma_start(&tokens);
            let end = closure.boundary_start(&tokens, sentence.range.end);
            let (tier, message) = if is_safely_promotable(&tokens, &closure) {
                (
                    Tier::Fix,
                    "sentence-final participial closer reads as an LLM tic; promoting it to its own sentence preserves the claim",
                )
            } else {
                (
                    Tier::Suggest,
                    "sentence-final participial closer reads as an LLM tic, but it has no clear object to promote; deleting it cannot be verified to preserve the sentence's claim, so this needs a human decision",
                )
            };
            findings.push(Finding::new(RULE_ID, start..end, message, tier));
        }
        findings
    }

    fn fix(
        &self,
        finding: &Finding,
        ctx: &RuleContext<'_>,
        _strategy_rng: &mut StrategyRng,
    ) -> Option<Patch> {
        // Only ever applies the content-preserving **promote** strategy —
        // see the module docs' "Mixed tier, per finding" section. A
        // Suggest-tier finding (no safe promotion available) always
        // declines here, the same defense-in-depth
        // `RitualConclusionRule::fix` and `TriadReductionRule::fix` apply
        // for their own Suggest-tier findings.
        if finding.tier != Tier::Fix {
            return None;
        }

        let band = ctx.envelope().band(GATED_METRIC)?;

        // See the module docs' "Exact, per-round budgeting" section: the
        // real per-round target count is computed here, from the real
        // document, not in `gate`. Only Fix-tier findings are ever
        // actually fixable, so budgeting and ranking both operate on that
        // subset — a Suggest-tier finding elsewhere in the document neither
        // consumes budget nor pushes a later Fix-tier finding out of it.
        let this_round: Vec<Finding> = self
            .scan(ctx)
            .into_iter()
            .filter(|f| f.tier == Tier::Fix)
            .collect();
        let sentence_count = ctx.sentences().count();
        if sentence_count == 0 {
            return None;
        }
        #[allow(clippy::cast_precision_loss)]
        let current = this_round.len() as f64 / sentence_count as f64;
        if current <= band.hi {
            return None;
        }
        let surplus = current - band.hi;
        #[allow(clippy::cast_precision_loss)]
        let sentence_count_f64 = sentence_count as f64;
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        let target_count = (surplus * sentence_count_f64).floor() as usize;
        if target_count == 0 {
            return None;
        }
        let rank = this_round.iter().position(|f| f.range == finding.range)?;
        if rank >= target_count {
            return None;
        }

        let (_sentence, tokens) = sentence_and_tokens_for(ctx, finding)?;
        let closure = find_closure(&tokens, ctx.document().source())?;
        if !is_safely_promotable(&tokens, &closure) {
            return None;
        }
        promote_patch(&tokens, &closure)
    }
}

#[cfg(test)]
mod tests {
    use friction_core::{Envelope, Sentence as CoreSentence, Token, TokenKind};
    use friction_nlp::PosTag;

    use super::*;
    use crate::context::MapEnvelope;

    // -----------------------------------------------------------------
    // Test helpers: hand-built tagged-token fixtures, mirroring
    // friction-metrics::symmetry's own `build_tokens` helper exactly so
    // this module's mirror of its predicate can be cross-checked against
    // the same inputs.
    // -----------------------------------------------------------------

    fn build_tokens(words: &[(&str, &str, &str)]) -> (String, Vec<TaggedToken>) {
        let mut source = String::new();
        let mut tokens = Vec::with_capacity(words.len());
        for &(surface, pos, lemma) in words {
            if !source.is_empty() && surface != "," && surface != "." {
                source.push(' ');
            }
            let start = source.len();
            source.push_str(surface);
            let end = source.len();
            tokens.push(TaggedToken {
                token: Token::new(start..end, TokenKind::Word),
                pos: PosTag::new(pos),
                lemma: lemma.into(),
            });
        }
        (source, tokens)
    }

    fn document(source: &str) -> friction_core::Document {
        let parsed = friction_parse::parse(source).expect("valid markdown parses");
        friction_nlp::segment_document(&parsed, &friction_nlp::SrxSegmenter::new())
            .expect("segmentation succeeds")
    }

    fn metrics_with_rate(rate: f64) -> MetricVector {
        MetricVector {
            participial_closer_rate: rate,
            ..MetricVector::default()
        }
    }

    /// A permissive envelope: `lo = hi = 1.0` puts the ceiling above any
    /// real fixture's rate, so `fix`'s document-derived target count never
    /// declines a finding purely for lack of a large-enough deficit —
    /// mirrors `families::contraction`'s own `permissive_envelope` test
    /// helper and its documented rationale.
    fn permissive_envelope() -> MapEnvelope {
        MapEnvelope::new().with(GATED_METRIC, Envelope::new(0.0, 0.0))
    }

    // -----------------------------------------------------------------
    // find_closure: mirrors friction-metrics::symmetry's own test cases
    // -----------------------------------------------------------------

    #[test]
    fn find_closure_matches_metrics_predicate_on_shared_fixtures() {
        // Same fixture as friction-metrics::symmetry's
        // `is_participial_closer_recognizes_trailing_vbg_clause`.
        let (source, tokens) = build_tokens(&[
            ("The", "DT", "the"),
            ("team", "NN", "team"),
            ("shipped", "VBD", "ship"),
            ("the", "DT", "the"),
            ("release", "NN", "release"),
            (",", ",", ","),
            ("raising", "VBG", "raise"),
            ("concerns", "NNS", "concern"),
            (".", ".", "."),
        ]);
        assert!(find_closure(&tokens, &source).is_some());
    }

    #[test]
    fn find_closure_rejects_past_participle() {
        // Mirrors `is_participial_closer_rejects_past_participle`.
        let (source, tokens) = build_tokens(&[
            ("The", "DT", "the"),
            ("team", "NN", "team"),
            ("shipped", "VBD", "ship"),
            ("the", "DT", "the"),
            ("release", "NN", "release"),
            (",", ",", ","),
            ("delayed", "VBN", "delay"),
            ("twice", "RB", "twice"),
            (".", ".", "."),
        ]);
        assert!(find_closure(&tokens, &source).is_none());
    }

    #[test]
    fn find_closure_rejects_sentence_without_comma() {
        // Mirrors `is_participial_closer_rejects_sentence_without_comma`.
        let (source, tokens) = build_tokens(&[
            ("The", "DT", "the"),
            ("team", "NN", "team"),
            ("shipped", "VBD", "ship"),
            ("the", "DT", "the"),
            ("release", "NN", "release"),
            (".", ".", "."),
        ]);
        assert!(find_closure(&tokens, &source).is_none());
    }

    #[test]
    fn find_closure_rejects_leading_comma() {
        // Mirrors `is_participial_closer_rejects_leading_comma`.
        let (source, tokens) = build_tokens(&[
            (",", ",", ","),
            ("raising", "VBG", "raise"),
            ("concerns", "NNS", "concern"),
            (".", ".", "."),
        ]);
        assert!(find_closure(&tokens, &source).is_none());
    }

    // -----------------------------------------------------------------
    // has_object_like_continuation
    // -----------------------------------------------------------------

    #[test]
    fn has_object_like_continuation_true_for_pronoun_object() {
        let (source, tokens) = build_tokens(&[
            ("It", "PRP", "it"),
            ("shipped", "VBD", "ship"),
            (",", ",", ","),
            ("making", "VBG", "make"),
            ("it", "PRP", "it"),
            ("easier", "JJR", "easy"),
            (".", ".", "."),
        ]);
        let closure = find_closure(&tokens, &source).expect("closure present");
        assert!(has_object_like_continuation(&tokens, &closure));
    }

    #[test]
    fn has_object_like_continuation_false_for_bare_participle() {
        let (source, tokens) = build_tokens(&[
            ("It", "PRP", "it"),
            ("shipped", "VBD", "ship"),
            (",", ",", ","),
            ("improving", "VBG", "improve"),
            (".", ".", "."),
        ]);
        let closure = find_closure(&tokens, &source).expect("closure present");
        assert!(!has_object_like_continuation(&tokens, &closure));
    }

    // -----------------------------------------------------------------
    // gate()
    // -----------------------------------------------------------------

    #[test]
    fn gate_is_off_without_a_band() {
        let rule = ParticipialCloserRule::new();
        let envelope = MapEnvelope::new();
        assert_eq!(rule.gate(&metrics_with_rate(0.5), &envelope), Gate::Off);
    }

    #[test]
    fn gate_is_off_inside_band() {
        let rule = ParticipialCloserRule::new();
        let envelope = MapEnvelope::new().with(GATED_METRIC, Envelope::new(0.0, 0.5));
        assert_eq!(rule.gate(&metrics_with_rate(0.2), &envelope), Gate::Off);
    }

    #[test]
    fn gate_above_band_returns_the_safety_cap_budget() {
        let rule = ParticipialCloserRule::new();
        let envelope = MapEnvelope::new().with(GATED_METRIC, Envelope::new(0.0, 0.1));
        assert_eq!(
            rule.gate(&metrics_with_rate(0.5), &envelope),
            Gate::Fix {
                budget: Budget::new(GATE_SAFETY_CAP)
            }
        );
    }

    // -----------------------------------------------------------------
    // scan() through the real pipeline (no tagger stub — this rule needs
    // real POS tags)
    // -----------------------------------------------------------------

    #[test]
    fn scan_finds_a_closer_through_the_real_tagger() {
        let source = "The team shipped the release, allowing customers to upgrade early.\n";
        let doc = document(source);
        let envelope = permissive_envelope();
        let tagger = friction_nlp::NlpruleTagger::new().expect("embedded model loads");
        let ctx = RuleContext::new(&doc, &tagger, "blog", &envelope);
        let findings = ParticipialCloserRule::new().scan(&ctx);
        assert_eq!(findings.len(), 1);
        assert_eq!(
            &source[findings[0].range.clone()],
            ", allowing customers to upgrade early"
        );
    }

    #[test]
    fn scan_finds_nothing_in_plain_prose() {
        let source = "The team shipped the release. It works well.\n";
        let doc = document(source);
        let envelope = permissive_envelope();
        let tagger = friction_nlp::NlpruleTagger::new().expect("embedded model loads");
        let ctx = RuleContext::new(&doc, &tagger, "blog", &envelope);
        assert!(ParticipialCloserRule::new().scan(&ctx).is_empty());
    }

    // -----------------------------------------------------------------
    // fix(): only ever promotes (never deletes) through the real pipeline
    // -----------------------------------------------------------------

    fn fix_first(source: &str) -> Option<String> {
        let doc = document(source);
        let envelope = permissive_envelope();
        let tagger = friction_nlp::NlpruleTagger::new().expect("embedded model loads");
        let ctx = RuleContext::new(&doc, &tagger, "blog", &envelope);
        let rule = ParticipialCloserRule::new();
        let finding = rule.scan(&ctx).into_iter().next().expect("a finding");
        let mut rng = StrategyRng::from_seed(0);
        let patch = rule.fix(&finding, &ctx, &mut rng)?;
        let mut applied = source.to_string();
        applied.replace_range(patch.range, &patch.replacement);
        Some(applied)
    }

    #[test]
    fn fix_with_a_clear_object_promotes_the_closer_into_its_own_sentence() {
        let source = "The team shipped the release, allowing customers to upgrade early.\n";
        let applied = fix_first(source).expect("expected a patch");
        assert_eq!(
            applied,
            "The team shipped the release. This allows customers to upgrade early.\n"
        );
    }

    /// Regression test for the finding that this rule's coin-flip
    /// **delete** strategy could silently discard the sentence's only
    /// concrete claim: a closer clause that carries real content (not
    /// decorative filler) but *does* have a clear object continuation is
    /// always promoted — never deleted — so the claim survives.
    #[test]
    fn fix_never_silently_drops_a_content_bearing_closer_with_an_object() {
        let source = "The outage lasted six hours, exposing a single point of failure in the primary datacenter's power supply.\n";
        let doc = document(source);
        let envelope = permissive_envelope();
        let tagger = friction_nlp::NlpruleTagger::new().expect("embedded model loads");
        let ctx = RuleContext::new(&doc, &tagger, "blog", &envelope);
        let rule = ParticipialCloserRule::new();
        let finding = rule.scan(&ctx).into_iter().next().expect("a finding");
        assert_eq!(
            finding.tier,
            Tier::Fix,
            "a closer with a clear object continuation is always safely promotable"
        );
        // Exercise every seed, not just one: this rule no longer draws a
        // strategy from `strategy_rng` at all, so every seed must produce
        // the exact same content-preserving patch.
        for seed in 0..50u64 {
            let mut rng = StrategyRng::from_seed(seed);
            let patch = rule
                .fix(&finding, &ctx, &mut rng)
                .expect("a safely-promotable finding always yields a patch");
            let mut applied = source.to_string();
            applied.replace_range(patch.range, &patch.replacement);
            assert_eq!(
                applied,
                "The outage lasted six hours. This exposes a single point of failure in the primary datacenter's power supply.\n",
                "seed {seed}: the outage's root-cause claim must survive the fix, never be silently deleted"
            );
        }
    }

    /// Without a clear object continuation, this rule cannot verify that
    /// deleting the closer clause is safe (it might be the sentence's only
    /// concrete claim, not decorative filler) — so it is Suggest tier and
    /// `fix` always declines, for every seed.
    #[test]
    fn fix_declines_when_there_is_no_clear_object_to_promote() {
        let source = "The team shipped the release, improving.\n";
        let doc = document(source);
        let envelope = permissive_envelope();
        let tagger = friction_nlp::NlpruleTagger::new().expect("embedded model loads");
        let ctx = RuleContext::new(&doc, &tagger, "blog", &envelope);
        let rule = ParticipialCloserRule::new();
        let finding = rule.scan(&ctx).into_iter().next().expect("a finding");
        assert_eq!(finding.tier, Tier::Suggest);
        for seed in 0..20u64 {
            let mut rng = StrategyRng::from_seed(seed);
            assert!(rule.fix(&finding, &ctx, &mut rng).is_none());
        }
    }

    // -----------------------------------------------------------------
    // Idempotence and determinism
    // -----------------------------------------------------------------

    #[test]
    fn fixing_a_document_is_idempotent() {
        // The second sentence's closer ("improving.") has no object
        // continuation, so it is Suggest tier and is never fixed — it is
        // expected to still be present, unchanged, after fixing (a
        // Suggest-tier finding is a stable diagnostic, not something a
        // second round could ever "re-find" as new; see the module docs'
        // "Idempotence" section). Idempotence here is specifically about
        // Fix-tier findings never reappearing after being fixed.
        let source = "The team shipped the release, allowing customers to upgrade early. \
            It also shipped a fix, improving.\n";
        let doc = document(source);
        let envelope = permissive_envelope();
        let tagger = friction_nlp::NlpruleTagger::new().expect("embedded model loads");
        let mut patches: Vec<Patch> = {
            let ctx = RuleContext::new(&doc, &tagger, "blog", &envelope);
            let rule = ParticipialCloserRule::new();
            rule.scan(&ctx)
                .iter()
                .filter_map(|finding| {
                    let mut rng = StrategyRng::seeded(source.as_bytes(), rule.id());
                    rule.fix(finding, &ctx, &mut rng)
                })
                .collect()
        };
        assert!(!patches.is_empty());
        patches.sort_by_key(|p| std::cmp::Reverse(p.range.start));

        let mut fixed = source.to_string();
        for patch in &patches {
            fixed.replace_range(patch.range.clone(), &patch.replacement);
        }

        let fixed_doc = document(&fixed);
        let ctx_after = RuleContext::new(&fixed_doc, &tagger, "blog", &envelope);
        let findings_after = ParticipialCloserRule::new().scan(&ctx_after);
        assert!(
            findings_after.iter().all(|f| f.tier != Tier::Fix),
            "expected no Fix-tier findings left after fixing, got {findings_after:?} in {fixed:?}"
        );
    }

    #[test]
    fn fixing_the_same_source_twice_is_byte_identical() {
        let source = "The team shipped the release, allowing customers to upgrade early.\n";
        let tagger = friction_nlp::NlpruleTagger::new().expect("embedded model loads");

        let run = || {
            let doc = document(source);
            let envelope = permissive_envelope();
            let ctx = RuleContext::new(&doc, &tagger, "blog", &envelope);
            let rule = ParticipialCloserRule::new();
            rule.scan(&ctx)
                .iter()
                .filter_map(|finding| {
                    let mut rng = StrategyRng::seeded(source.as_bytes(), rule.id());
                    rule.fix(finding, &ctx, &mut rng)
                })
                .collect::<Vec<_>>()
        };

        assert_eq!(run(), run());
    }

    /// Exercises `sentence_and_tokens_for`'s fallback path is unreachable
    /// for a well-formed finding — kept as a smoke test that the helper's
    /// `Sentence` type import compiles and behaves for a directly
    /// constructed sentence, independent of the real segmenter.
    #[test]
    fn sentence_and_tokens_for_finds_the_owning_sentence() {
        let source = "One. Two, allowing customers to upgrade early.\n";
        let doc = document(source);
        let envelope = permissive_envelope();
        let tagger = friction_nlp::NlpruleTagger::new().expect("embedded model loads");
        let ctx = RuleContext::new(&doc, &tagger, "blog", &envelope);
        let finding = ParticipialCloserRule::new()
            .scan(&ctx)
            .into_iter()
            .next()
            .expect("a finding");
        let (sentence, tokens): (&CoreSentence, Vec<TaggedToken>) =
            sentence_and_tokens_for(&ctx, &finding).expect("owning sentence found");
        assert!(span::contains_range(&sentence.range, &finding.range));
        assert!(!tokens.is_empty());
    }
}
