//! Connective surgery: trims sentence-initial "heavy" transition words
//! ("Moreover,", "However,", "Consequently,", ...) when a document leans
//! on them more than the genre's human envelope does.
//!
//! Three meaning-preserving strategies compete for each occurrence, picked
//! deterministically per occurrence from the engine's hash-seeded
//! [`StrategyRng`] (never a fixed constant choice, never ambient
//! randomness — see this crate's docs):
//!
//! - **Delete**: drop the connective, its trailing comma, and the
//!   whitespace after it, recapitalizing the word that now opens the
//!   sentence. `"However, it still works."` -> `"It still works."`
//! - **Swap**: replace the connective (and its trailing comma) with a
//!   short coordinator from the same semantic class — "But" for a
//!   contrastive connective, "And" for an additive one, "So" for a
//!   consequential one. `"Moreover, it scales."` -> `"And it scales."`
//! - **Leave unchanged**: decline this occurrence entirely (`fix` returns
//!   `None`); the engine does not count this against the round's budget,
//!   so a later occurrence in the same round still gets a chance at it.
//!
//! Every occurrence in the document is reported as a [`Finding`] by
//! [`ConnectiveSurgery::scan`] in source order; `friction-apply`'s driver
//! is what stops calling `fix` once a round's budget is spent, walking
//! findings in exactly that order — so occurrences later in the document
//! are the ones left untouched when a budget runs out partway through.
//!
//! # Why only these connectives, and only followed by a comma
//!
//! [`CONNECTIVES`] deliberately only lists sentence-initial adverbial
//! connectives that are conventionally followed directly by a comma
//! (`"Moreover, ..."`, not `"Moreover you should..."`). Requiring that
//! comma as part of the match is not just cosmetic: it is what keeps this
//! rule from misfiring on an unrelated construction that merely starts
//! with the same word, e.g. `"In addition to the screws, the kit..."` (a
//! prepositional phrase, not the transition adverbial `"In addition,
//! ..."`) never matches, because no comma sits directly after `"In
//! addition"` there.
//!
//! # Idempotence
//!
//! Both fix strategies are idempotent by construction: **delete** removes
//! the connective outright, so a second pass over the same text has
//! nothing left to find; **swap**'s replacement text ("But"/"And"/"So") is
//! never itself a [`CONNECTIVES`] entry (checked by this module's own
//! `swap_targets_are_not_connectives` test — the "no RHS maps to another
//! LHS" closure this workspace requires of every substitution table), so a
//! swapped sentence can never be rematched as a heavy connective either.

use friction_core::{Finding, MetricVector, Patch, RuleId, Tier};

use crate::budget::Budget;
use crate::context::{GenreEnvelope, RuleContext};
use crate::rule::{Gate, Rule, RuleFamily};
use crate::strategy::StrategyRng;

/// This rule's stable identifier.
const RULE_ID: RuleId = RuleId::new("connective.surgery");

/// The [`friction_core::MetricVector`] field this rule gates on:
/// sentence-initial discourse-marker density, per 1000 word tokens.
const GATED_METRIC: &str = "discourse_marker_density";

/// How much fixing one occurrence is projected to move
/// [`GATED_METRIC`], for [`Budget::from_envelope_excess`].
///
/// The metric is already normalized to a rate *per 1000 word tokens*, and
/// nothing in a [`MetricVector`] (by design — see that type's docs) carries
/// the document's raw token count, so `gate` cannot reconstruct the exact
/// per-occurrence effect for *this particular* document. Using `1.0` — one
/// point of the metric's own per-1000-token scale — is the natural,
/// dimensionless unit given that constraint: for a document close to 1000
/// tokens this is close to the real effect; for a much longer document it
/// *understates* the real effect (one deletion moves a huge document's
/// density by less than a full point), which only ever makes the computed
/// budget conservative, never excessive — an under-shoot here just means
/// the next round's fresh `gate` call, re-measured against the actual
/// resulting text, tops the budget up further, exactly the behavior
/// [`Budget::from_envelope_excess`]'s own docs describe as the intended
/// failure mode of a conservative estimate.
const PER_FIX_EFFECT: f64 = 1.0;

/// The semantic class of a heavy connective, and the short coordinator its
/// **swap** strategy replaces it with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConnectiveClass {
    /// Signals a contrast with what preceded it ("However", "Nevertheless").
    Contrastive,
    /// Signals an addition to what preceded it ("Moreover", "Furthermore").
    Additive,
    /// Signals a consequence of what preceded it ("Therefore", "Thus").
    Consequential,
}

impl ConnectiveClass {
    /// The short coordinator **swap** replaces this class's connective
    /// with, already capitalized for sentence-initial use.
    const fn coordinator(self) -> &'static str {
        match self {
            Self::Contrastive => "But",
            Self::Additive => "And",
            Self::Consequential => "So",
        }
    }
}

/// One entry in [`CONNECTIVES`]: the connective's exact, sentence-initial
/// (title-case) surface form and its [`ConnectiveClass`].
struct ConnectiveEntry {
    marker: &'static str,
    class: ConnectiveClass,
}

/// Sentence-initial heavy connectives this rule targets, each paired with
/// the semantic class its **swap** strategy draws a coordinator from.
///
/// Matched case-sensitively (see [`match_connective`]) against exactly the
/// title-case surface form listed here — the form these connectives
/// overwhelmingly take when they open a sentence — and only when
/// immediately followed by a comma (see the module docs' "Why only these
/// connectives" section). No two entries here share a marker, and (see the
/// `swap_targets_are_not_connectives` test) no class's coordinator is
/// itself one of these markers.
const CONNECTIVES: &[ConnectiveEntry] = &[
    ConnectiveEntry {
        marker: "However",
        class: ConnectiveClass::Contrastive,
    },
    ConnectiveEntry {
        marker: "Nevertheless",
        class: ConnectiveClass::Contrastive,
    },
    ConnectiveEntry {
        marker: "Nonetheless",
        class: ConnectiveClass::Contrastive,
    },
    ConnectiveEntry {
        marker: "Moreover",
        class: ConnectiveClass::Additive,
    },
    ConnectiveEntry {
        marker: "Furthermore",
        class: ConnectiveClass::Additive,
    },
    ConnectiveEntry {
        marker: "Additionally",
        class: ConnectiveClass::Additive,
    },
    ConnectiveEntry {
        marker: "In addition",
        class: ConnectiveClass::Additive,
    },
    ConnectiveEntry {
        marker: "Consequently",
        class: ConnectiveClass::Consequential,
    },
    ConnectiveEntry {
        marker: "Therefore",
        class: ConnectiveClass::Consequential,
    },
    ConnectiveEntry {
        marker: "Thus",
        class: ConnectiveClass::Consequential,
    },
];

/// The strategy chosen for one occurrence — see the module docs for what
/// each does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Strategy {
    Delete,
    Swap,
    None,
}

/// Draws this occurrence's strategy from `strategy_rng`.
///
/// Weighted 45% delete / 45% swap / 10% leave-unchanged, via a single
/// `gen_range(100)` draw bucketed `[0, 45)` / `[45, 90)` / `[90, 100)`. The
/// two active strategies are weighted equally (neither is presented as the
/// "primary" fix), each favored well over the 10% left inert — enough
/// weight on "leave unchanged" that a dense run of occurrences doesn't
/// scrub every single one identically (which would just trade one uniform
/// tic for another), but not so much that a rule with real budget to spend
/// routinely leaves a document still over its envelope.
const fn choose_strategy(strategy_rng: &mut StrategyRng) -> Strategy {
    match strategy_rng.gen_range(100) {
        0..45 => Strategy::Delete,
        45..90 => Strategy::Swap,
        _ => Strategy::None,
    }
}

/// If `text` starts with one of [`CONNECTIVES`]'s markers immediately
/// followed by a comma, returns that entry.
///
/// `text` is expected to already have its leading whitespace stripped
/// (true of every [`friction_core::Sentence`]'s own text — segmenters
/// exclude a sentence's surrounding whitespace by contract). Matching is
/// case-sensitive against the marker's exact title-case form: this rule
/// only targets the conventional sentence-initial capitalization, not a
/// stray mid-sentence-fragment lowercase echo of the same word.
fn match_connective(text: &str) -> Option<&'static ConnectiveEntry> {
    CONNECTIVES
        .iter()
        .find(|entry| text.starts_with(entry.marker) && text[entry.marker.len()..].starts_with(','))
}

/// Byte offset of the first non-whitespace character in `source` at or
/// after `from`, or `source.len()` if none remains.
fn skip_whitespace(source: &str, from: usize) -> usize {
    let rest = &source[from..];
    let stop = rest
        .find(|c: char| !c.is_whitespace())
        .unwrap_or(rest.len());
    from + stop
}

/// The **delete** strategy's patch: removes the connective, its trailing
/// comma, and the whitespace after it in one splice, folding the
/// recapitalization of the word that now opens the sentence into the same
/// contiguous range when that word starts lowercase.
///
/// # Panics
/// Panics (via the invariant `expect` below) if `finding`'s range is not
/// immediately followed by a comma in `source` — cannot happen for a
/// `finding` produced by [`ConnectiveSurgery::scan`], whose match already
/// required exactly that.
fn delete_patch(source: &str, finding: &Finding) -> Patch {
    let comma_end = comma_end_after(source, finding.range.end);
    let ws_end = skip_whitespace(source, comma_end);

    let (patch_end, replacement) = match source[ws_end..].chars().next() {
        Some(c) if c.is_lowercase() => {
            let upper: String = c.to_uppercase().collect();
            (ws_end + c.len_utf8(), upper)
        }
        _ => (ws_end, String::new()),
    };

    Patch::new(
        finding.range.start..patch_end,
        replacement,
        RULE_ID,
        Tier::Fix,
    )
}

/// The **swap** strategy's patch: replaces the connective and its
/// trailing comma with `class`'s short coordinator, leaving the single
/// space that already separated the comma from the rest of the sentence
/// untouched (so `"But"`/`"And"`/`"So"` reads as a normal sentence-initial
/// coordinator, not `"But, ..."`).
///
/// # Panics
/// See [`delete_patch`] — same invariant, same reason it cannot fire.
fn swap_patch(source: &str, finding: &Finding, class: ConnectiveClass) -> Patch {
    let comma_end = comma_end_after(source, finding.range.end);
    Patch::new(
        finding.range.start..comma_end,
        class.coordinator(),
        RULE_ID,
        Tier::Fix,
    )
}

/// `end + 1`, the byte offset just past the comma [`ConnectiveSurgery::
/// scan`] already confirmed sits at `source[end]`.
fn comma_end_after(source: &str, end: usize) -> usize {
    debug_assert_eq!(
        source[end..].chars().next(),
        Some(','),
        "a Finding from ConnectiveSurgery::scan always sits immediately before a comma"
    );
    end + ','.len_utf8()
}

/// Sentence-initial connective-surgery: deletes, swaps, or leaves
/// unchanged each sentence-initial heavy connective, budgeted to bring
/// [`GATED_METRIC`] back into the genre's envelope.
#[derive(Debug, Clone, Copy, Default)]
pub struct ConnectiveSurgery;

impl ConnectiveSurgery {
    /// Creates the rule.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Rule for ConnectiveSurgery {
    fn id(&self) -> RuleId {
        RULE_ID
    }

    fn family(&self) -> RuleFamily {
        RuleFamily::Connective
    }

    fn gate(&self, metrics: &MetricVector, envelope: &dyn GenreEnvelope) -> Gate {
        let Some(band) = envelope.band(GATED_METRIC) else {
            return Gate::Off;
        };
        let current = metrics.discourse_marker_density;
        // Only the "too many connectives" direction is this rule's to
        // fix — it only ever deletes or swaps, never inserts one, so a
        // document already inside the band, or (unusually) below its
        // floor, gates Off either way: there is nothing this rule could
        // safely do about a deficit.
        if current <= band.hi {
            return Gate::Off;
        }
        let budget = Budget::from_envelope_excess(current, band, PER_FIX_EFFECT);
        if budget.is_exhausted() {
            Gate::Off
        } else {
            Gate::Fix { budget }
        }
    }

    fn scan(&self, ctx: &RuleContext<'_>) -> Vec<Finding> {
        let document = ctx.document();
        let mut findings = Vec::new();
        for (_, sentence) in ctx.sentences() {
            let Ok(text) = document.text(&sentence.range) else {
                continue;
            };
            if let Some(entry) = match_connective(text) {
                let start = sentence.range.start;
                let end = start + entry.marker.len();
                findings.push(Finding::new(
                    RULE_ID,
                    start..end,
                    format!(
                        "sentence-initial connective \"{}\" is over this genre's envelope",
                        entry.marker
                    ),
                    Tier::Fix,
                ));
            }
        }
        findings
    }

    fn fix(
        &self,
        finding: &Finding,
        ctx: &RuleContext<'_>,
        strategy_rng: &mut StrategyRng,
    ) -> Option<Patch> {
        let source = ctx.document().source();
        let marker_text = &source[finding.range.clone()];
        let entry = CONNECTIVES
            .iter()
            .find(|entry| entry.marker == marker_text)
            .expect("a Finding from ConnectiveSurgery::scan always names a CONNECTIVES marker");

        match choose_strategy(strategy_rng) {
            Strategy::Delete => Some(delete_patch(source, finding)),
            Strategy::Swap => Some(swap_patch(source, finding, entry.class)),
            Strategy::None => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use friction_core::Envelope;
    use friction_nlp::{SrxSegmenter, Tagger};

    use super::*;
    use crate::context::MapEnvelope;

    /// A stub tagger; none of this rule's logic consults POS tags.
    struct NoopTagger;
    impl Tagger for NoopTagger {
        fn tag(&self, _text: &str, _base_offset: usize) -> Vec<friction_nlp::TaggedToken> {
            Vec::new()
        }
    }

    fn document(source: &str) -> friction_core::Document {
        let parsed = friction_parse::parse(source).expect("valid markdown parses");
        friction_nlp::segment_document(&parsed, &SrxSegmenter::new())
            .expect("segmentation succeeds")
    }

    fn metrics_with_density(density: f64) -> MetricVector {
        MetricVector {
            discourse_marker_density: density,
            ..MetricVector::default()
        }
    }

    // ---------------------------------------------------------------
    // Closed substitution table
    // ---------------------------------------------------------------

    /// No two [`CONNECTIVES`] entries share a marker.
    #[test]
    fn connectives_markers_are_unique() {
        let mut markers: Vec<&str> = CONNECTIVES.iter().map(|e| e.marker).collect();
        let before = markers.len();
        markers.sort_unstable();
        markers.dedup();
        assert_eq!(markers.len(), before, "duplicate marker in CONNECTIVES");
    }

    /// The closed-table invariant this workspace requires of every
    /// substitution table: no **swap** replacement text ("But"/"And"/
    /// "So") is itself a [`CONNECTIVES`] lookup key. If it were, swapping
    /// one occurrence could hand the next round's `scan` a brand-new
    /// "finding" made of the rule's own previous output — breaking
    /// idempotence.
    #[test]
    fn swap_targets_are_not_connectives() {
        let coordinators = [
            ConnectiveClass::Contrastive.coordinator(),
            ConnectiveClass::Additive.coordinator(),
            ConnectiveClass::Consequential.coordinator(),
        ];
        for coordinator in coordinators {
            assert!(
                CONNECTIVES.iter().all(|entry| entry.marker != coordinator),
                "{coordinator} (a swap target) must not itself be a CONNECTIVES marker"
            );
            // Also confirm a sentence starting with the coordinator alone
            // never re-matches, the same check `scan` itself performs.
            let probe = format!("{coordinator}, more text follows.");
            assert!(
                match_connective(&probe).is_none(),
                "{probe:?} must not match as a heavy connective"
            );
        }
    }

    // ---------------------------------------------------------------
    // gate()
    // ---------------------------------------------------------------

    /// No envelope band for the gated metric: gate is `Off` regardless of
    /// the metric's value.
    #[test]
    fn gate_is_off_without_a_band() {
        let rule = ConnectiveSurgery::new();
        let envelope = MapEnvelope::new();
        let gate = rule.gate(&metrics_with_density(500.0), &envelope);
        assert_eq!(gate, Gate::Off);
    }

    /// A density inside the band gates `Off`.
    #[test]
    fn gate_is_off_inside_band() {
        let rule = ConnectiveSurgery::new();
        let envelope = MapEnvelope::new().with(GATED_METRIC, Envelope::new(0.0, 10.0));
        let gate = rule.gate(&metrics_with_density(5.0), &envelope);
        assert_eq!(gate, Gate::Off);
    }

    /// A density *below* the band's floor also gates `Off` — this rule
    /// only ever removes connectives, so it has no safe move for a
    /// document that already has too few.
    #[test]
    fn gate_is_off_below_band_floor() {
        let rule = ConnectiveSurgery::new();
        let envelope = MapEnvelope::new().with(GATED_METRIC, Envelope::new(5.0, 10.0));
        let gate = rule.gate(&metrics_with_density(1.0), &envelope);
        assert_eq!(gate, Gate::Off);
    }

    /// Hand-computed: current 13.0, band hi 10.0, `PER_FIX_EFFECT` 1.0 ->
    /// excess 3.0 -> budget 3.
    #[test]
    fn gate_above_band_computes_hand_verified_budget() {
        let rule = ConnectiveSurgery::new();
        let envelope = MapEnvelope::new().with(GATED_METRIC, Envelope::new(0.0, 10.0));
        let gate = rule.gate(&metrics_with_density(13.0), &envelope);
        assert_eq!(
            gate,
            Gate::Fix {
                budget: Budget::new(3)
            }
        );
    }

    /// A density only marginally above the band produces a zero budget
    /// (`floor` rounds down), which this rule reports as `Off` rather
    /// than `Fix { budget: Budget::ZERO }` — no point handing the driver
    /// a rule with literally nothing to spend.
    #[test]
    fn gate_with_zero_budget_is_off() {
        let rule = ConnectiveSurgery::new();
        let envelope = MapEnvelope::new().with(GATED_METRIC, Envelope::new(0.0, 10.0));
        let gate = rule.gate(&metrics_with_density(10.5), &envelope);
        assert_eq!(gate, Gate::Off);
    }

    // ---------------------------------------------------------------
    // scan()
    // ---------------------------------------------------------------

    /// `scan` finds every sentence-initial heavy connective, in source
    /// order, each finding's range covering exactly the connective word
    /// (not the comma or anything after it).
    #[test]
    fn scan_finds_every_occurrence_in_source_order() {
        let source = "Moreover, it scales. Plain sentence here. However, it still works.";
        let doc = document(source);
        let envelope = MapEnvelope::new();
        let ctx = RuleContext::new(&doc, &NoopTagger, "blog", &envelope);
        let rule = ConnectiveSurgery::new();

        let findings = rule.scan(&ctx);
        assert_eq!(findings.len(), 2);
        assert_eq!(&source[findings[0].range.clone()], "Moreover");
        assert_eq!(&source[findings[1].range.clone()], "However");
        assert!(findings[0].range.start < findings[1].range.start);
    }

    /// `"In addition to X, ..."` is a prepositional phrase, not the
    /// transition adverbial `"In addition, ..."` — no comma sits directly
    /// after `"In addition"`, so it must not match.
    #[test]
    fn scan_does_not_match_in_addition_to_as_a_connective() {
        let source = "In addition to screws, the kit includes bolts.";
        let doc = document(source);
        let envelope = MapEnvelope::new();
        let ctx = RuleContext::new(&doc, &NoopTagger, "blog", &envelope);
        let rule = ConnectiveSurgery::new();
        assert!(rule.scan(&ctx).is_empty());
    }

    /// A connective appearing mid-sentence (not sentence-initial) does not
    /// match.
    #[test]
    fn scan_does_not_match_mid_sentence_connective() {
        let source = "It works well, however, it is slow.";
        let doc = document(source);
        let envelope = MapEnvelope::new();
        let ctx = RuleContext::new(&doc, &NoopTagger, "blog", &envelope);
        let rule = ConnectiveSurgery::new();
        assert!(rule.scan(&ctx).is_empty());
    }

    // ---------------------------------------------------------------
    // fix() strategies, driven directly by a known seed
    // ---------------------------------------------------------------

    /// `StrategyRng::from_seed(0)`'s first `next_u64` is the value
    /// independently verified in `crate::strategy`'s own tests
    /// (`16_294_208_416_658_607_535`); `% 100 == 35`, which
    /// `choose_strategy`'s documented bucketing (`[0,45)` delete)
    /// resolves to **Delete** — hand-traceable from that already
    /// cross-checked reference value, not from running this rule's own
    /// code.
    #[test]
    fn fix_seed_zero_deletes_and_recapitalizes() {
        let source = "However, it still works.";
        let doc = document(source);
        let envelope = MapEnvelope::new();
        let ctx = RuleContext::new(&doc, &NoopTagger, "blog", &envelope);
        let rule = ConnectiveSurgery::new();

        let finding = &rule.scan(&ctx)[0];
        let mut rng = StrategyRng::from_seed(0);
        let patch = rule.fix(finding, &ctx, &mut rng).expect("seed 0 -> Delete");
        assert_eq!(patch.replacement, "I");
        assert_eq!(&source[patch.range.clone()], "However, i");

        let mut applied = source.to_string();
        applied.replace_range(patch.range, &patch.replacement);
        assert_eq!(applied, "It still works.");
    }

    /// A seed whose first draw lands in the `[45, 90)` bucket chooses
    /// **Swap**. Seed `1` (`from_seed`, bypassing the sentence/rule hash
    /// entirely) is used here purely as a fixed, arbitrary probe seed —
    /// its resulting `next_u64` is taken at face value from splitmix64's
    /// own definition, then reduced mod 100 by hand, exactly as
    /// `fix_seed_zero_deletes_and_recapitalizes` does for seed 0.
    #[test]
    fn fix_finds_a_swap_seed_and_applies_it() {
        let source = "Moreover, it scales well.";
        let doc = document(source);
        let envelope = MapEnvelope::new();
        let ctx = RuleContext::new(&doc, &NoopTagger, "blog", &envelope);
        let rule = ConnectiveSurgery::new();
        let finding = &rule.scan(&ctx)[0];

        // Scan a small range of arbitrary seeds for one that lands in the
        // documented Swap bucket, rather than hand-tracing splitmix64
        // arithmetic a second time — `choose_strategy`'s bucketing is
        // already exercised directly (independent of this rule) by
        // `crate::strategy`'s own reference-vector tests plus this
        // module's `strategy_distribution_over_synthetic_sentences_hits_
        // every_bucket` below.
        let seed = (0..1000u64)
            .find(|&s| {
                matches!(
                    choose_strategy(&mut StrategyRng::from_seed(s)),
                    Strategy::Swap
                )
            })
            .expect("some seed in 0..1000 lands in the Swap bucket");

        let mut rng = StrategyRng::from_seed(seed);
        let patch = rule
            .fix(finding, &ctx, &mut rng)
            .expect("chosen seed -> Swap");
        assert_eq!(patch.replacement, "And");
        assert_eq!(&source[patch.range.clone()], "Moreover,");

        let mut applied = source.to_string();
        applied.replace_range(patch.range, &patch.replacement);
        assert_eq!(applied, "And it scales well.");
    }

    /// A seed whose first draw lands in the `[90, 100)` bucket declines
    /// the finding (`fix` returns `None`) — same probing technique as
    /// `fix_finds_a_swap_seed_and_applies_it`.
    #[test]
    fn fix_finds_a_leave_unchanged_seed_and_declines() {
        let source = "Therefore, it must be true.";
        let doc = document(source);
        let envelope = MapEnvelope::new();
        let ctx = RuleContext::new(&doc, &NoopTagger, "blog", &envelope);
        let rule = ConnectiveSurgery::new();
        let finding = &rule.scan(&ctx)[0];

        let seed = (0..1000u64)
            .find(|&s| {
                matches!(
                    choose_strategy(&mut StrategyRng::from_seed(s)),
                    Strategy::None
                )
            })
            .expect("some seed in 0..1000 lands in the leave-unchanged bucket");

        let mut rng = StrategyRng::from_seed(seed);
        assert!(rule.fix(finding, &ctx, &mut rng).is_none());
    }

    /// Deleting a connective before a word that is *already* capitalized
    /// (e.g. a proper noun) does not touch that capitalization — only the
    /// connective, comma, and separating whitespace are removed.
    #[test]
    fn delete_does_not_recapitalize_an_already_capitalized_word() {
        let source = "However, Anna shipped it.";
        let doc = document(source);
        let envelope = MapEnvelope::new();
        let ctx = RuleContext::new(&doc, &NoopTagger, "blog", &envelope);
        let finding = &ConnectiveSurgery::new().scan(&ctx)[0];

        let patch = delete_patch(source, finding);
        assert_eq!(patch.replacement, "");
        let mut applied = source.to_string();
        applied.replace_range(patch.range, &patch.replacement);
        assert_eq!(applied, "Anna shipped it.");
    }

    // ---------------------------------------------------------------
    // Strategy distribution
    // ---------------------------------------------------------------

    /// Over 50 distinct synthetic sentences (varied enough that their
    /// `xxh64` seeds are, for this rule's purposes, effectively
    /// independent draws), every one of the three strategies is chosen at
    /// least once — the distribution isn't collapsed onto one or two of
    /// the three by some accidental bias in the bucketing.
    #[test]
    fn strategy_distribution_over_synthetic_sentences_hits_every_bucket() {
        use std::collections::BTreeSet;

        let mut seen: BTreeSet<&'static str> = BTreeSet::new();
        for i in 0..50u32 {
            let sentence = format!("Moreover, synthetic sentence number {i} appears here.");
            let mut rng = StrategyRng::seeded(sentence.as_bytes(), RULE_ID);
            let label = match choose_strategy(&mut rng) {
                Strategy::Delete => "delete",
                Strategy::Swap => "swap",
                Strategy::None => "none",
            };
            seen.insert(label);
        }
        assert_eq!(
            seen,
            BTreeSet::from(["delete", "swap", "none"]),
            "expected all three strategies to occur across 50 synthetic sentences, saw {seen:?}"
        );
    }

    // ---------------------------------------------------------------
    // Determinism
    // ---------------------------------------------------------------

    /// Fixing the same source twice, from scratch, with independently
    /// constructed contexts, yields byte-identical patches: same ranges,
    /// same replacement text, in the same order.
    #[test]
    fn fixing_the_same_source_twice_is_byte_identical() {
        let source = "Moreover, it scales. However, it is slow. Additionally, it is cheap.";

        let run = || {
            let doc = document(source);
            let envelope = MapEnvelope::new();
            let ctx = RuleContext::new(&doc, &NoopTagger, "blog", &envelope);
            let rule = ConnectiveSurgery::new();
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
}
