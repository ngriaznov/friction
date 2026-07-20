//! The fixpoint driver: repeated rounds of parse -> metrics -> gate -> scan
//! -> fix -> apply, bounded and reported.

use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap};

use friction_core::{Document, Finding, Patch, RuleId, Tier, span};
use friction_nlp::{Segmenter, TaggedToken, Tagger};
use friction_plan::Plan;
use friction_rules::{Gate, GenreEnvelope, Rule, RuleContext, RuleFamily, StrategyRng};

use crate::conflict::{Candidate, apply_patches, resolve_round};

/// A per-round memoizing wrapper over a [`Tagger`].
///
/// One round's document text is fixed for the round's whole duration (see
/// [`run_round`]), but several independent call sites each tag it from
/// scratch: `friction-metrics::compute`'s own tagger-dependent metrics
/// (`triad_rate`, `bullet_parallelism`, `participial_closer_rate` each walk
/// every sentence and tag it independently) and every active rule's own
/// `scan`/`fix` step (several of which call
/// [`friction_rules::RuleContext::tag_sentence`] once per sentence they
/// examine, and multiple rules often examine the same sentence). Left
/// unmemoized, a document's sentences each get tagged — the most expensive
/// step in the whole pipeline, since it runs the embedded `nlprule` model —
/// upwards of a dozen times over in a single round.
///
/// [`Tagger::tag`]'s own documented contract — identical `text` and
/// `base_offset` always produce identical output — is exactly what makes
/// memoizing it safe: this wrapper is constructed fresh inside
/// [`run_round`] and dropped at that round's end, so a cache entry can
/// never outlive (or be reused across) the text it was computed from.
struct CachingTagger<'a> {
    inner: &'a dyn Tagger,
    cache: RefCell<HashMap<(usize, String), Vec<TaggedToken>>>,
}

impl<'a> CachingTagger<'a> {
    fn new(inner: &'a dyn Tagger) -> Self {
        Self {
            inner,
            cache: RefCell::new(HashMap::new()),
        }
    }
}

impl Tagger for CachingTagger<'_> {
    fn tag(&self, text: &str, base_offset: usize) -> Vec<TaggedToken> {
        let key = (base_offset, text.to_string());
        if let Some(cached) = self.cache.borrow().get(&key) {
            return cached.clone();
        }
        let tagged = self.inner.tag(text, base_offset);
        self.cache.borrow_mut().insert(key, tagged.clone());
        tagged
    }
}

/// Up to this many rounds run before the fixpoint driver stops
/// unconditionally, even if the last round still produced patches.
///
/// A rule's budget is recomputed fresh every round from that round's
/// *actual* (re-measured) metric excess, never carried over or
/// compounded, and every registered rule only ever deletes, merges
/// adjacent text, splits a sentence at an existing boundary, or
/// substitutes toward a closed, non-cyclic target — so the total
/// "distance left to fix" strictly decreases round over round (a split
/// closes its own over-length gap by construction: neither half of a
/// split sentence is itself split-eligible again with the same excess)
/// and the driver is guaranteed to reach a genuine zero-patch round
/// eventually, for any finite document. `16` is not a theoretical bound
/// but an empirical one with real headroom: a full sweep over every
/// corpus document (`friction-apply`'s idempotence sweep fixture)
/// converges naturally within 9 rounds at the slowest, so 16 leaves each
/// an individual document could plausibly need without ever being the
/// *reason* a well-behaved document's fix is incomplete — see
/// `crate::fix_document`'s own idempotence guarantee, which this bound
/// exists to make actually hold rather than merely usually hold.
pub const MAX_ROUNDS: usize = 16;

/// Errors produced while running the fixpoint driver.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ApplyError {
    /// A round's source failed to parse as markdown.
    #[error("round {round}: parse failed: {source}")]
    Parse {
        /// The 1-indexed round the failure happened in.
        round: usize,
        /// Underlying parse error.
        #[source]
        source: friction_parse::ParseError,
    },
    /// A round's document failed to sentence-segment.
    #[error("round {round}: segmentation failed: {source}")]
    Segment {
        /// The 1-indexed round the failure happened in.
        round: usize,
        /// Underlying segmentation error.
        #[source]
        source: friction_nlp::SegmentError,
    },
}

/// One round's outcome: which rules fired, what they found, and how many
/// of their proposed patches were applied versus dropped.
#[derive(Debug, Clone)]
pub struct RoundReport {
    /// This round's 1-indexed number (between 1 and [`MAX_ROUNDS`]
    /// inclusive).
    pub round: usize,
    /// Ids of every rule that produced at least one *accepted* patch this
    /// round (i.e. survived both its own gate/budget and conflict
    /// resolution), sorted and deduplicated.
    pub rules_fired: Vec<RuleId>,
    /// Every finding surfaced by an active (non-`Off`-gated) rule's `scan`
    /// this round, in the order rules were scanned — diagnostic
    /// information independent of which findings ended up fixed.
    pub findings: Vec<Finding>,
    /// How many patches were applied this round (after conflict
    /// resolution).
    pub patches_applied: usize,
    /// How many candidate patches were dropped this round, either for
    /// failing span validation against the round's source or for losing
    /// conflict resolution to a higher-priority overlapping patch.
    pub patches_dropped: usize,
    /// The accepted patches actually applied this round (after conflict
    /// resolution), sorted by `range.start` ascending. Every range indexes
    /// *this round's own* source text — the original `source` passed to
    /// [`run_fixpoint`] for round 1, or the previous round's resulting
    /// text for round 2 onward — never the original document's bytes past
    /// round 1.
    ///
    /// Kept alongside the plain `patches_applied` count for callers that
    /// need to know exactly what changed and where, e.g. mapping a
    /// multi-round fix back to which spans of the *original* input were
    /// ever touched (see `crate::touched_original_ranges`).
    pub applied_patches: Vec<Patch>,
}

/// The full fixpoint driver's outcome: one [`RoundReport`] per round
/// actually run (at most [`MAX_ROUNDS`], fewer if a round produced zero
/// patches and the driver stopped early).
#[derive(Debug, Clone)]
pub struct FixpointReport {
    /// Per-round reports, in round order.
    pub rounds: Vec<RoundReport>,
}

impl FixpointReport {
    /// Total patches applied across every round.
    #[must_use]
    pub fn total_patches_applied(&self) -> usize {
        self.rounds.iter().map(|r| r.patches_applied).sum()
    }
}

/// Runs the fixpoint driver on `source`: up to [`MAX_ROUNDS`] rounds of
/// parse -> compute metrics -> gate every rule -> scan -> fix -> resolve
/// conflicts -> apply -> (re-parse for the next round).
///
/// Each round is independent: `source` is re-parsed from scratch every
/// round (never carried over as a mutated in-memory tree), so every
/// [`friction_core::Patch`] a rule proposes is guaranteed to carry a byte
/// range into that round's actual current text.
///
/// Stops after a round that applies zero patches (including possibly the
/// very first round, on a document already inside its envelope — the
/// common case) — the returned text at that point equals the previous
/// round's, so this is never observable as a "wasted" round from the
/// caller's side beyond one extra (cheap, patch-free) report entry.
///
/// `rules` is queried in the given order for gating and scanning, but the
/// result never depends on that order: every candidate patch, from every
/// rule, is pooled and conflict-resolved together by
/// [`crate::resolve_round`] before anything is applied.
///
/// # Errors
/// Returns [`ApplyError`] if any round's current text fails to parse or
/// sentence-segment. A well-formed input and well-behaved
/// `segmenter`/`tagger` cannot trigger this on the first round; a
/// misbehaving `Rule::fix` implementation that produces a patch making the
/// *next* round's text invalid markdown (impossible for pure text
/// substitution, since markdown's block grammar does not depend on inline
/// prose content in a way normal substitutions could break) is the only
/// realistic way a later round could.
pub fn run_fixpoint(
    source: &str,
    rules: &[&dyn Rule],
    segmenter: &dyn Segmenter,
    tagger: &dyn Tagger,
    genre: &str,
    envelope: &dyn GenreEnvelope,
) -> Result<(String, FixpointReport), ApplyError> {
    run_fixpoint_with_plan(source, rules, segmenter, tagger, genre, envelope, None)
}

/// Runs the fixpoint driver exactly like [`run_fixpoint`], with one
/// additional, optional input: `plan`.
///
/// `plan` is `None` -> `Some`, additive: passing `None` runs the exact
/// same code path [`run_fixpoint`] itself runs (that function is a thin
/// wrapper over this one, supplying `None`), so it reproduces
/// [`run_fixpoint`]'s output byte-for-byte, with no separate
/// implementation to drift out of sync — see
/// `crate::tests::plan_none_matches_run_fixpoint` for a regression test
/// pinning this down over a real fixture.
///
/// Passing `Some(plan)` layers one additional constraint on top of every
/// rule's own [`Rule::gate`]-computed budget, never in place of it: each
/// round, a family may contribute at most [`Plan::budget_for`]'s value
/// for that family to that round's candidate pool, counting only
/// candidates that actually got that far (i.e. already inside their own
/// rule's per-rule budget). A finding still gets scanned and surfaced in
/// [`RoundReport::findings`] even once its family's plan budget is
/// exhausted — only the fix pathway is capped, exactly like a per-rule
/// [`friction_rules::Budget`] of zero already behaves. Because the cap
/// applies to *candidates entering conflict resolution*, not to a
/// post-resolution count, the number of patches a family actually ends up
/// applying in a round can be lower than its plan budget (an overlap with
/// another family's patch can still drop one) but never higher.
///
/// # Errors
/// See [`run_fixpoint`].
pub fn run_fixpoint_with_plan(
    source: &str,
    rules: &[&dyn Rule],
    segmenter: &dyn Segmenter,
    tagger: &dyn Tagger,
    genre: &str,
    envelope: &dyn GenreEnvelope,
    plan: Option<&Plan>,
) -> Result<(String, FixpointReport), ApplyError> {
    let mut current = source.to_string();
    let mut rounds = Vec::with_capacity(MAX_ROUNDS);

    for round in 1..=MAX_ROUNDS {
        let (next, report) = run_round(
            &current, rules, segmenter, tagger, genre, envelope, round, plan,
        )?;
        let applied = report.patches_applied;
        rounds.push(report);
        current = next;
        if applied == 0 {
            break;
        }
    }

    Ok((current, FixpointReport { rounds }))
}

/// Runs a single round: parse, compute metrics, gate/scan/fix every rule,
/// resolve conflicts, and apply. See [`run_fixpoint_with_plan`] for the
/// driver this composes into.
#[allow(clippy::too_many_arguments)]
fn run_round(
    source: &str,
    rules: &[&dyn Rule],
    segmenter: &dyn Segmenter,
    tagger: &dyn Tagger,
    genre: &str,
    envelope: &dyn GenreEnvelope,
    round: usize,
    plan: Option<&Plan>,
) -> Result<(String, RoundReport), ApplyError> {
    let document =
        friction_parse::parse(source).map_err(|source| ApplyError::Parse { round, source })?;
    // Shared across every tagger-dependent step this round runs (metrics
    // computation below, then every rule's scan/fix) — see
    // `CachingTagger`'s own docs for why memoizing it here is safe.
    let caching_tagger = CachingTagger::new(tagger);
    let metrics = friction_metrics::compute(&document, segmenter, &caching_tagger);
    let with_sentences = friction_nlp::segment_document(&document, segmenter)
        .map_err(|source| ApplyError::Segment { round, source })?;
    let ctx = RuleContext::new(&with_sentences, &caching_tagger, genre, envelope);

    let mut findings = Vec::new();
    let mut candidates: Vec<Candidate> = Vec::new();
    let mut fired: BTreeSet<RuleId> = BTreeSet::new();
    // How many candidates each family has already contributed this round,
    // indexed by `RuleFamily::priority` (0..=5, one slot per family) —
    // only ever consulted when `plan` is `Some`; see `family_has_room`.
    let mut family_counts = [0usize; 6];

    for &rule in rules {
        match rule.gate(&metrics, envelope) {
            Gate::Off => {}
            Gate::Detect => {
                // Diagnostic-only this round: still worth surfacing what
                // was found, but `fix` is never called, regardless of any
                // individual finding's own tier.
                findings.extend(rule.scan(&ctx));
            }
            Gate::Fix { mut budget } => {
                for finding in rule.scan(&ctx) {
                    if finding.tier == Tier::Fix
                        && !budget.is_exhausted()
                        && family_has_room(plan, rule.family(), &family_counts)
                    {
                        let sentence_bytes = sentence_bytes_for(&with_sentences, &finding);
                        let mut rng = StrategyRng::seeded(sentence_bytes, rule.id());
                        if let Some(patch) = rule.fix(&finding, &ctx, &mut rng) {
                            // Defense-in-depth: only a Fix-tier patch is
                            // ever eligible for automatic application,
                            // regardless of the finding's own tier having
                            // passed that same check above.
                            if patch.tier == Tier::Fix {
                                budget = budget
                                    .take_one()
                                    .expect("budget was checked non-exhausted above");
                                family_counts[rule.family().priority() as usize] += 1;
                                fired.insert(rule.id());
                                candidates.push(Candidate {
                                    patch,
                                    family: rule.family(),
                                });
                            }
                        }
                    }
                    findings.push(finding);
                }
            }
        }
    }

    let (accepted, patches_dropped) = resolve_round(source, candidates);
    let next = apply_patches(source, &accepted);

    Ok((
        next,
        RoundReport {
            round,
            rules_fired: fired.into_iter().collect(),
            findings,
            patches_applied: accepted.len(),
            patches_dropped,
            applied_patches: accepted,
        },
    ))
}

/// `true` if `family` may still contribute another candidate this round:
/// always `true` when `plan` is `None` (the default, behavior-preserving
/// case — see [`run_fixpoint_with_plan`]'s own docs), otherwise `true`
/// only while `family`'s running count in `counts` (indexed by
/// [`RuleFamily::priority`]) is still below `plan`'s
/// [`Plan::budget_for`] value for that family.
fn family_has_room(plan: Option<&Plan>, family: RuleFamily, counts: &[usize; 6]) -> bool {
    plan.is_none_or(|plan| counts[family.priority() as usize] < plan.budget_for(family))
}

/// The source bytes of the sentence containing `finding`, for seeding a
/// [`StrategyRng`].
///
/// Falls back to `finding`'s own range if no sentence in `document`
/// contains it (e.g. a structural finding spanning a whole block) — always
/// a pure, deterministic function of `document`'s text either way, never a
/// panic.
fn sentence_bytes_for<'a>(document: &'a Document, finding: &Finding) -> &'a [u8] {
    for unit in document.prose() {
        for sentence in &unit.sentences {
            if span::contains_range(&sentence.range, &finding.range) {
                return document
                    .text(&sentence.range)
                    .expect("sentence ranges are already validated against the document")
                    .as_bytes();
            }
        }
    }
    document.text(&finding.range).map_or(&[], str::as_bytes)
}

#[cfg(test)]
mod tests {
    use friction_core::{Envelope, MetricVector, Patch};
    use friction_nlp::SrxSegmenter;
    use friction_rules::{Budget, MapEnvelope};

    use super::*;

    /// A stub tagger that tags nothing — fine for rules in these tests,
    /// none of which need part-of-speech information.
    struct NoopTagger;
    impl Tagger for NoopTagger {
        fn tag(&self, _text: &str, _base_offset: usize) -> Vec<friction_nlp::TaggedToken> {
            Vec::new()
        }
    }

    /// A rule that always gates `Fix` with a fixed budget, finds every
    /// occurrence of a literal phrase, and deletes it (plus one trailing
    /// space, if present) — a minimal stand-in for a lexical filler-phrase
    /// deletion rule.
    struct DeletePhraseRule {
        id: RuleId,
        phrase: &'static str,
        budget: usize,
    }

    impl Rule for DeletePhraseRule {
        fn id(&self) -> RuleId {
            self.id
        }

        fn family(&self) -> friction_rules::RuleFamily {
            friction_rules::RuleFamily::Lexical
        }

        fn gate(&self, _metrics: &MetricVector, _envelope: &dyn GenreEnvelope) -> Gate {
            Gate::Fix {
                budget: Budget::new(self.budget),
            }
        }

        fn scan(&self, ctx: &RuleContext<'_>) -> Vec<Finding> {
            let source = ctx.document().source();
            let mut findings = Vec::new();
            let mut start = 0;
            while let Some(offset) = source[start..].find(self.phrase) {
                let range = (start + offset)..(start + offset + self.phrase.len());
                findings.push(Finding::new(
                    self.id,
                    range.clone(),
                    "filler phrase",
                    Tier::Fix,
                ));
                start = range.end;
            }
            findings
        }

        fn fix(
            &self,
            finding: &Finding,
            ctx: &RuleContext<'_>,
            _strategy_rng: &mut StrategyRng,
        ) -> Option<Patch> {
            let source = ctx.document().source();
            let mut end = finding.range.end;
            if source[end..].starts_with(' ') {
                end += 1;
            }
            Some(Patch::new(finding.range.start..end, "", self.id, Tier::Fix))
        }
    }

    /// A rule that gates `Off` unconditionally — used to prove an `Off`
    /// rule contributes nothing to a round.
    struct AlwaysOffRule;
    impl Rule for AlwaysOffRule {
        fn id(&self) -> RuleId {
            RuleId::new("stub.off")
        }
        fn family(&self) -> friction_rules::RuleFamily {
            friction_rules::RuleFamily::Lexical
        }
        fn gate(&self, _metrics: &MetricVector, _envelope: &dyn GenreEnvelope) -> Gate {
            Gate::Off
        }
        fn scan(&self, _ctx: &RuleContext<'_>) -> Vec<Finding> {
            panic!("scan must never be called when gate() returned Off");
        }
        fn fix(
            &self,
            _f: &Finding,
            _ctx: &RuleContext<'_>,
            _rng: &mut StrategyRng,
        ) -> Option<Patch> {
            panic!("fix must never be called when gate() returned Off");
        }
    }

    /// A rule that gates `Detect` unconditionally and always proposes a
    /// fix if asked — used to prove `Detect` surfaces findings but never
    /// applies patches.
    struct AlwaysDetectRule;
    impl Rule for AlwaysDetectRule {
        fn id(&self) -> RuleId {
            RuleId::new("stub.detect")
        }
        fn family(&self) -> friction_rules::RuleFamily {
            friction_rules::RuleFamily::Symmetry
        }
        fn gate(&self, _metrics: &MetricVector, _envelope: &dyn GenreEnvelope) -> Gate {
            Gate::Detect
        }
        fn scan(&self, ctx: &RuleContext<'_>) -> Vec<Finding> {
            let source = ctx.document().source();
            if source.is_empty() {
                Vec::new()
            } else {
                vec![Finding::new(
                    self.id(),
                    0..source.len().min(1),
                    "would-be finding",
                    Tier::Suggest,
                )]
            }
        }
        fn fix(
            &self,
            _f: &Finding,
            _ctx: &RuleContext<'_>,
            _rng: &mut StrategyRng,
        ) -> Option<Patch> {
            panic!("fix must never be called when gate() returned Detect");
        }
    }

    fn run(source: &str, rules: &[&dyn Rule]) -> (String, FixpointReport) {
        let segmenter = SrxSegmenter::new();
        let tagger = NoopTagger;
        let envelope = MapEnvelope::new();
        run_fixpoint(source, rules, &segmenter, &tagger, "blog", &envelope)
            .expect("well-formed markdown input must not fail")
    }

    /// A rule gated `Off` never scans (enforced by the stub's own panic)
    /// and contributes no patches.
    #[test]
    fn off_gated_rule_never_scans_or_fixes() {
        let rule: &dyn Rule = &AlwaysOffRule;
        let (output, report) = run("Some ordinary prose.", &[rule]);
        assert_eq!(output, "Some ordinary prose.");
        assert_eq!(report.total_patches_applied(), 0);
    }

    /// A rule gated `Detect` surfaces its findings but never fixes them —
    /// output is unchanged and no patches are applied, even though the
    /// stub's `fix` would panic if ever called.
    #[test]
    fn detect_gated_rule_surfaces_findings_without_fixing() {
        let rule: &dyn Rule = &AlwaysDetectRule;
        let (output, report) = run("Some ordinary prose.", &[rule]);
        assert_eq!(output, "Some ordinary prose.");
        assert_eq!(report.total_patches_applied(), 0);
        assert_eq!(report.rounds[0].findings.len(), 1);
    }

    /// A `Fix`-gated rule with enough budget deletes every occurrence of
    /// its target phrase, converges (a later round finds nothing left to
    /// delete), and the driver reports which rule fired.
    #[test]
    fn fix_gated_rule_deletes_every_occurrence_and_converges() {
        let rule = DeletePhraseRule {
            id: RuleId::new("lexical.filler"),
            phrase: "it is worth noting that ",
            budget: 10,
        };
        let rule: &dyn Rule = &rule;
        let source = "it is worth noting that this works. it is worth noting that so does this.";
        let (output, report) = run(source, &[rule]);
        assert_eq!(output, "this works. so does this.");
        assert!(report.total_patches_applied() >= 2);
        assert!(
            report
                .rounds
                .iter()
                .any(|r| r.rules_fired.contains(&RuleId::new("lexical.filler")))
        );

        // Idempotence: fixing the fixed output again changes nothing.
        let (output_again, report_again) = run(&output, &[rule]);
        assert_eq!(output_again, output);
        assert_eq!(report_again.total_patches_applied(), 0);
    }

    /// A budget of `0` means the rule scans (findings are still reported)
    /// but fixes nothing.
    #[test]
    fn zero_budget_scans_without_fixing() {
        let rule = DeletePhraseRule {
            id: RuleId::new("lexical.filler"),
            phrase: "filler",
            budget: 0,
        };
        let rule: &dyn Rule = &rule;
        let (output, report) = run("some filler text here.", &[rule]);
        assert_eq!(output, "some filler text here.");
        assert_eq!(report.total_patches_applied(), 0);
        assert_eq!(report.rounds[0].findings.len(), 1);
    }

    /// A budget smaller than the number of findings in one round fixes
    /// only that many, leftmost first (scan returns findings in source
    /// order, and the driver walks them in that order) — the budget is
    /// per round, though, so a later round with a fresh budget picks up
    /// where the previous one left off, and the driver still converges to
    /// the same fixed point across rounds.
    #[test]
    fn partial_budget_fixes_only_that_many_leftmost_first() {
        let rule = DeletePhraseRule {
            id: RuleId::new("lexical.filler"),
            phrase: "filler ",
            budget: 1,
        };
        let rule: &dyn Rule = &rule;
        let source = "filler one and filler two.";
        let (output, report) = run(source, &[rule]);

        // Round 1 fixes only the leftmost occurrence...
        assert_eq!(report.rounds[0].patches_applied, 1);
        // ...round 2 (fresh budget) fixes the remaining one...
        assert_eq!(report.rounds[1].patches_applied, 1);
        // ...and by the final, fully-converged output, both are gone.
        assert_eq!(output, "one and two.");
    }

    /// The driver stops after at most `MAX_ROUNDS` rounds even if a rule
    /// would keep finding (and fixing) something every round forever.
    #[test]
    fn driver_stops_after_max_rounds() {
        // Deletes one character (`x`) at a time, one per round (budget 1),
        // from a run of `x`s long enough to outlast MAX_ROUNDS.
        struct EatOneXPerRound;
        impl Rule for EatOneXPerRound {
            fn id(&self) -> RuleId {
                RuleId::new("stub.eat_x")
            }
            fn family(&self) -> friction_rules::RuleFamily {
                friction_rules::RuleFamily::Lexical
            }
            fn gate(&self, _metrics: &MetricVector, _envelope: &dyn GenreEnvelope) -> Gate {
                Gate::Fix {
                    budget: Budget::new(1),
                }
            }
            fn scan(&self, ctx: &RuleContext<'_>) -> Vec<Finding> {
                let source = ctx.document().source();
                source
                    .find('x')
                    .map(|i| vec![Finding::new(self.id(), i..i + 1, "x", Tier::Fix)])
                    .unwrap_or_default()
            }
            fn fix(
                &self,
                finding: &Finding,
                _ctx: &RuleContext<'_>,
                _rng: &mut StrategyRng,
            ) -> Option<Patch> {
                Some(Patch::new(finding.range.clone(), "", self.id(), Tier::Fix))
            }
        }

        let rule: &dyn Rule = &EatOneXPerRound;
        let x_count = MAX_ROUNDS + 10; // more x's than MAX_ROUNDS can consume
        let source = "x".repeat(x_count);
        let (output, report) = run(&source, &[rule]);
        assert_eq!(report.rounds.len(), MAX_ROUNDS);
        assert_eq!(output, "x".repeat(x_count - MAX_ROUNDS));
    }

    /// Running the fixpoint driver with no rules at all leaves the input
    /// unchanged and produces a single zero-patch round.
    #[test]
    fn no_rules_leaves_input_unchanged() {
        let (output, report) = run("Untouched prose.", &[]);
        assert_eq!(output, "Untouched prose.");
        assert_eq!(report.rounds.len(), 1);
        assert_eq!(report.total_patches_applied(), 0);
    }

    /// A `Gate::Fix`-gated rule whose `scan` reports *both* a `Tier::Fix`
    /// finding and a `Tier::Suggest` finding for the same document (the
    /// shape `friction-rules::families::symmetry::RitualConclusionRule`
    /// needs: a single rule whose per-occurrence tier is a runtime decision,
    /// not fixed per rule) — confirms both survive into the round's
    /// `findings`, in scan order, while only the `Tier::Fix` one is ever
    /// turned into an applied patch. This is not new driver behavior (the
    /// `Gate::Fix` branch above already pushes every scanned finding,
    /// regardless of tier, unconditionally); this test exists to pin that
    /// behavior down explicitly for the mixed-tier-per-finding shape, so a
    /// future change to this branch can't silently start dropping the
    /// `Suggest` half.
    #[test]
    fn fix_gated_rule_surfaces_a_mixed_tier_finding_set_and_fixes_only_the_fix_tier_one() {
        struct MixedTierRule;
        impl Rule for MixedTierRule {
            fn id(&self) -> RuleId {
                RuleId::new("stub.mixed_tier")
            }
            fn family(&self) -> friction_rules::RuleFamily {
                friction_rules::RuleFamily::Symmetry
            }
            fn gate(&self, _metrics: &MetricVector, _envelope: &dyn GenreEnvelope) -> Gate {
                Gate::Fix {
                    budget: Budget::new(10),
                }
            }
            fn scan(&self, ctx: &RuleContext<'_>) -> Vec<Finding> {
                let source = ctx.document().source();
                vec![
                    Finding::new(self.id(), 0..1, "fixable", Tier::Fix),
                    Finding::new(
                        self.id(),
                        source.len() - 1..source.len(),
                        "suggest-only",
                        Tier::Suggest,
                    ),
                ]
            }
            fn fix(
                &self,
                finding: &Finding,
                _ctx: &RuleContext<'_>,
                _rng: &mut StrategyRng,
            ) -> Option<Patch> {
                assert_eq!(
                    finding.tier,
                    Tier::Fix,
                    "the driver must never call fix() for a Suggest-tier finding"
                );
                Some(Patch::new(finding.range.clone(), "X", self.id(), Tier::Fix))
            }
        }

        let rule: &dyn Rule = &MixedTierRule;
        let (_output, report) = run("ab", &[rule]);
        let round = &report.rounds[0];
        assert_eq!(round.findings.len(), 2, "both findings must be surfaced");
        assert_eq!(round.findings[0].tier, Tier::Fix);
        assert_eq!(round.findings[1].tier, Tier::Suggest);
        assert_eq!(
            round.patches_applied, 1,
            "only the Fix-tier finding should have produced an applied patch"
        );
    }

    /// Regression test: `run_fixpoint_with_plan(..., None)` reproduces
    /// `run_fixpoint`'s output byte-for-byte — including every round's
    /// fired-rule set, applied/dropped counts, and the applied patch
    /// count — on a golden, multi-round, multi-family fixture (structural
    /// unbullet, connective surgery, and contraction insertion all fire
    /// across four rounds). `run_fixpoint` is defined as calling this
    /// function with `None` (see its own docs), so this passes by
    /// construction today; the point of this test is to catch a future
    /// change that breaks that construction — e.g. a refactor that
    /// accidentally starts threading `plan` through some other path.
    #[test]
    fn plan_none_matches_pre_plan_output_on_a_golden_fixture() {
        let segmenter = SrxSegmenter::new();
        let tagger = NoopTagger;
        let envelope = MapEnvelope::new()
            .with("list_item_density", Envelope::new(0.0, 0.0))
            .with("discourse_marker_density", Envelope::new(0.0, 0.0))
            .with("contraction_ratio", Envelope::new(0.9, 1.0));
        let source = "- Moreover, it is not ready for every team member to use\n\
                       - Furthermore, it is not likely to work either\n";
        let rules = crate::registered_rules();

        let (without_plan, report_without_plan) =
            run_fixpoint(source, &rules, &segmenter, &tagger, "blog", &envelope)
                .expect("golden fixture must fix cleanly");
        let (with_none_plan, report_with_none_plan) =
            run_fixpoint_with_plan(source, &rules, &segmenter, &tagger, "blog", &envelope, None)
                .expect("golden fixture must fix cleanly");

        // Pin the actual output, not just the two paths' agreement with
        // each other: a future change to any registered rule that alters
        // this fixture's output has to update this assertion consciously.
        assert_eq!(
            without_plan,
            "It's not ready for every team member to use and it's not likely to work either.\n"
        );
        assert_eq!(
            without_plan, with_none_plan,
            "run_fixpoint_with_plan(..., None) must reproduce run_fixpoint's output exactly"
        );

        assert_eq!(
            report_without_plan.rounds.len(),
            report_with_none_plan.rounds.len()
        );
        for (a, b) in report_without_plan
            .rounds
            .iter()
            .zip(&report_with_none_plan.rounds)
        {
            assert_eq!(a.round, b.round);
            assert_eq!(a.rules_fired, b.rules_fired);
            assert_eq!(a.patches_applied, b.patches_applied);
            assert_eq!(a.patches_dropped, b.patches_dropped);
            assert_eq!(a.applied_patches, b.applied_patches);
        }
    }

    /// A [`Plan`] with a `0` budget for a rule's family blocks that
    /// family's fixes entirely, this round, while leaving its scan
    /// (findings surfaced for diagnostics) untouched — exactly the same
    /// shape [`zero_budget_scans_without_fixing`] already pins down for a
    /// rule's own per-round [`Budget`] of zero, now via the plan-level
    /// cap instead.
    #[test]
    fn plan_zero_budget_blocks_a_familys_fixes_but_not_its_scan() {
        let rule = DeletePhraseRule {
            id: RuleId::new("lexical.filler"),
            phrase: "filler ",
            budget: 10,
        };
        let rule: &dyn Rule = &rule;
        let source = "filler one and filler two.";

        // No bands at all -> every family's summed advisory budget is 0.
        let plan = Plan::build(&MetricVector::default(), &MapEnvelope::new());
        assert_eq!(plan.budget_for(RuleFamily::Lexical), 0);

        let segmenter = SrxSegmenter::new();
        let tagger = NoopTagger;
        let envelope = MapEnvelope::new();
        let (output, report) = run_fixpoint_with_plan(
            source,
            &[rule],
            &segmenter,
            &tagger,
            "blog",
            &envelope,
            Some(&plan),
        )
        .expect("well-formed markdown input must not fail");

        assert_eq!(output, source, "a zero plan budget must apply no patches");
        assert_eq!(report.total_patches_applied(), 0);
        assert_eq!(
            report.rounds[0].findings.len(),
            2,
            "scanning must still happen even though the family's plan budget is zero"
        );
    }

    /// A [`Plan`]'s per-family budget caps that family's applied patches
    /// for the round, even when the firing rule's own per-round
    /// [`Budget`] (computed independently by [`Rule::gate`]) would allow
    /// more — the plan layers an additional constraint on top, it never
    /// loosens the rule's own. The cap is per round, though (like a
    /// rule's own budget): a fresh round re-applies the same plan budget
    /// and picks up where the previous one left off.
    #[test]
    fn plan_caps_a_familys_applied_patches_per_round() {
        let rule = DeletePhraseRule {
            id: RuleId::new("lexical.filler"),
            phrase: "filler ",
            // Generous rule-level budget: not the limiting factor here.
            budget: 10,
        };
        let rule: &dyn Rule = &rule;
        let source = "filler one filler two filler three filler four filler five.";

        // Hand-computed: llm_favored_phrase_rate at 5.0 against a [0.0,
        // 2.0] band, per-fix effect 1.0 -> excess 3.0 -> budget 3.
        let plan = Plan::build(
            &MetricVector {
                llm_favored_phrase_rate: 5.0,
                ..MetricVector::default()
            },
            &MapEnvelope::new().with("llm_favored_phrase_rate", Envelope::new(0.0, 2.0)),
        );
        assert_eq!(plan.budget_for(RuleFamily::Lexical), 3);

        let segmenter = SrxSegmenter::new();
        let tagger = NoopTagger;
        let envelope = MapEnvelope::new();
        let (output, report) = run_fixpoint_with_plan(
            source,
            &[rule],
            &segmenter,
            &tagger,
            "blog",
            &envelope,
            Some(&plan),
        )
        .expect("well-formed markdown input must not fail");

        assert_eq!(
            report.rounds[0].patches_applied, 3,
            "plan budget of 3 must cap round 1, even though the rule's own budget of 10 \
             would allow all 5 occurrences"
        );
        assert_eq!(
            report.rounds[1].patches_applied, 2,
            "round 2's fresh cap picks up the 2 occurrences round 1 left behind"
        );
        assert_eq!(output, "one two three four five.");
    }
}
