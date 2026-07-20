//! Contraction insertion: rewrites expanded auxiliary-verb phrases ("do
//! not", "it is", "we are", ...) into their contracted form ("don't",
//! "it's", "we're").
//!
//! This runs when a document's contraction ratio sits below its genre's
//! human envelope — i.e. it reads more formally than that genre's own
//! writers typically do.
//!
//! # Data source
//!
//! The `(expanded, contracted)` table itself is not duplicated here: it is
//! read straight from [`friction_metrics::contraction_pairs`], the exact
//! table [`friction_metrics::contraction_ratio`] counts over, so the
//! metric this rule gates on and the rewrite it performs can never quietly
//! drift apart (see this module's `compiled_pairs_matches_metrics_table`
//! test).
//!
//! # Exceptions (never contracted)
//!
//! Four conservative exceptions, checked before any match is turned into a
//! [`Finding`] — a false negative here just leaves a sentence untouched
//! (always safe); a false positive would change how a sentence reads, so
//! every one of these is deliberately cautious:
//!
//! - **Emphasis capitals**: any word in the matched phrase written in ALL
//!   CAPS (two or more letters, all uppercase) — `"do NOT"`, `"IS NOT
//!   ready"` — signals stress/emphasis the writer chose deliberately;
//!   contracting would erase it. A lone capital `"I"` never counts on its
//!   own (it is always capitalized as the pronoun, not for emphasis).
//! - **Sentence-final auxiliary**: nothing but punctuation follows the
//!   match before the sentence ends — `"Yes, it is."` — English does not
//!   let a contracted auxiliary carry a sentence's final stress this way
//!   (no fluent writer turns that into `"Yes, it's."`); applied uniformly
//!   across the whole table rather than only to the (linguistically the
//!   only *strictly* required) positive-auxiliary pairs, since being more
//!   conservative than the grammar strictly demands only ever costs one
//!   fewer contraction, never a wrong one.
//! - **Comma boundary**: the match is immediately followed by a comma —
//!   `"That is, roughly speaking, correct."`, `"It is, however, still
//!   running."` — these are parenthetical/explanatory uses of the phrase,
//!   not a simple copula, and contracting them reads oddly at best.
//! - **Cleft "that"**: for `"it is"`/`"that is"` specifically, immediately
//!   followed by the word `"that"` — `"It is that the numbers don't add
//!   up."` — conservatively left alone: this shape can read as a
//!   cleft/complementizer construction where the contracted form shifts
//!   emphasis in a way a plain copula substitution does not.
//!
//! # Exact, per-round budgeting
//!
//! [`Rule::gate`] decides `Off`/`Fix` from only the round's [`MetricVector`]
//! and the genre's envelope — it runs *before* `scan`, so it cannot see the
//! real document, and therefore cannot know [`GATED_METRIC`]'s real
//! per-fix effect: fixing one contractible occurrence moves
//! `contraction_ratio`'s numerator by exactly one, and its denominator
//! (`contracted + contractible`) does not change, so the exact effect is
//! `1 / denominator` — a number this rule cannot compute without the
//! document. Guessing that denominator with a fixed constant does not
//! work: a guess too small under-budgets a document with more contractible
//! occurrences than assumed (it can take far more rounds than are actually
//! available to close the gap), and a guess too large over-budgets a
//! document with fewer (risking a single round pushing the ratio past the
//! *far* edge of the envelope).
//!
//! So `gate` hands back a budget that is deliberately just a generous,
//! document-size-independent safety ceiling ([`GATE_SAFETY_CAP`]) whenever
//! the document is below its floor at all, and [`ContractionRule::fix`] —
//! which *does* have the real document, via `ctx` — does the exact
//! computation instead: it re-derives the document's true `contracted`/
//! `contractible` counts (the same [`friction_metrics::contraction_counts`]
//! the metric itself is built from, so this can never drift from what
//! `gate` was reacting to), works out exactly how many of this round's
//! findings — in source order, leftmost first, the same order [`Budget`]
//! is spent in everywhere else in this workspace — are needed to bring the
//! ratio up to (never past) the envelope floor, and declines every finding
//! past that count. The same floor-not-ceiling rounding
//! [`crate::budget::Budget::from_envelope_excess`] uses elsewhere applies
//! here too: if the document's real deficit is smaller than a single
//! fix's own effect, the exact count is `0` and this round changes
//! nothing, rather than force one fix that would overshoot the floor.
//!
//! # Genre flag
//!
//! [`GenreFlags::legalish`] is a genre-level switch, orthogonal to the
//! envelope-driven density gate: when set, this rule proposes nothing at
//! all for that genre, because a legal-register (or legal-adjacent)
//! genre's house style favors fully expanded auxiliaries as a matter of
//! convention, not as an LLM-introduced tic to correct. No genre in the
//! v1 frozen genre set (`docs`, `blog`, `readme`, `email`, `forum`) is
//! legal-register, so [`LEGALISH_GENRES`] is empty today; it exists as the
//! extension point a future genre would hook into, without threading a
//! new concept through the rest of the engine.
//!
//! # Idempotence
//!
//! Every `contracted` form in the table contains an apostrophe and every
//! `expanded` form is plain alphabetic words with none, so no rewrite this
//! rule makes can ever itself match one of [`COMPILED_PAIRS`]'s patterns
//! again — the "no substitution-table RHS is also an LHS" closure this
//! workspace requires of every table like this one, checked directly by
//! this module's `contracted_forms_never_match_as_expanded_phrases` test.

use std::ops::Range;
use std::sync::LazyLock;

use friction_core::{Finding, MetricVector, Patch, RuleId, Tier, span};
use regex::Regex;

use crate::budget::Budget;
use crate::context::{GenreEnvelope, RuleContext};
use crate::rule::{Gate, Rule, RuleFamily};
use crate::strategy::StrategyRng;

/// This rule's stable identifier.
const RULE_ID: RuleId = RuleId::new("contraction.insert");

/// The [`MetricVector`] field this rule gates on.
const GATED_METRIC: &str = "contraction_ratio";

/// The per-round budget [`ContractionRule::gate`] hands back whenever the
/// document is below its envelope floor at all.
///
/// Not a computed estimate of how many fixes are actually needed — see the
/// module docs' "Exact, per-round budgeting" section for why `gate` cannot
/// compute that (it never sees the real document) and where the real limit
/// actually gets enforced ([`ContractionRule::fix`], document-derived and
/// exact). This is only a generous upper safety bound, sized well above
/// how many contractible occurrences even a long document plausibly
/// carries, so it is never itself the reason a real document's fix falls
/// short.
const GATE_SAFETY_CAP: usize = 1000;

/// Genres whose house style keeps every auxiliary expanded — see the
/// module docs' "Genre flag" section. Empty for the v1 frozen genre set;
/// sorted (ASCII byte order) for the same reason every other lookup table
/// in this workspace is, even though it is empty today.
const LEGALISH_GENRES: [&str; 0] = [];

/// Genre-level behavior switches for this rule, orthogonal to its
/// envelope-driven density gate. See the module docs' "Genre flag"
/// section.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GenreFlags {
    /// When `true`, this rule proposes no findings at all for the genre.
    pub legalish: bool,
}

impl GenreFlags {
    /// Resolves this rule's genre flags for `genre`.
    #[must_use]
    pub fn for_genre(genre: &str) -> Self {
        Self {
            legalish: LEGALISH_GENRES.contains(&genre),
        }
    }
}

/// One compiled entry of [`friction_metrics::contraction_pairs`]: the
/// expanded phrase's own text, its contracted replacement, and the regex
/// that finds it in sentence text.
struct CompiledPair {
    expanded: &'static str,
    contracted: &'static str,
    pattern: Regex,
}

/// Every [`friction_metrics::contraction_pairs`] entry, each compiled once
/// into a case-insensitive, whole-word pattern.
static COMPILED_PAIRS: LazyLock<Vec<CompiledPair>> = LazyLock::new(|| {
    friction_metrics::contraction_pairs()
        .into_iter()
        .map(|(expanded, contracted)| CompiledPair {
            expanded,
            contracted,
            pattern: compile_pattern(expanded),
        })
        .collect()
});

/// Builds `expanded`'s matching pattern: its words, matched
/// case-insensitively, each still its own separate word (`\b...\b`),
/// joined by (and allowing) any run of whitespace between them —
/// sentence text can contain a raw markdown source line break where a
/// human writer would type a single space.
fn compile_pattern(expanded: &str) -> Regex {
    let joined = expanded
        .split_whitespace()
        .map(regex::escape)
        .collect::<Vec<_>>()
        .join(r"\s+");
    Regex::new(&format!(r"(?i)\b{joined}\b"))
        .expect("every contraction_pairs() expanded phrase compiles to a valid pattern")
}

/// One [`COMPILED_PAIRS`] match in a sentence's text: the local
/// (sentence-relative) byte range and which pair matched.
#[derive(Debug)]
struct RawMatch {
    local_range: Range<usize>,
    pair_index: usize,
}

/// A match's length in bytes.
const fn match_len(m: &RawMatch) -> usize {
    m.local_range.end - m.local_range.start
}

/// Every [`COMPILED_PAIRS`] match in `text`, unordered and not yet
/// deduplicated for overlap — see [`resolve_sentence_matches`].
fn scan_sentence(text: &str) -> Vec<RawMatch> {
    let mut matches = Vec::new();
    for (pair_index, pair) in COMPILED_PAIRS.iter().enumerate() {
        for m in pair.pattern.find_iter(text) {
            matches.push(RawMatch {
                local_range: m.start()..m.end(),
                pair_index,
            });
        }
    }
    matches
}

/// Greedily keeps the leftmost, longest, lowest-`pair_index`
/// non-overlapping subset of `matches` — the same "leftmost-longest" rule
/// `friction-apply`'s own conflict resolution uses, applied here within a
/// single rule's own candidates so `scan` never hands the driver two
/// overlapping findings for the same span. E.g. `"She is not late."`
/// matches both the `"she is"` and `"is not"` pairs, sharing the word
/// `"is"`; only the leftmost-starting one, `"she is"`, survives.
fn resolve_sentence_matches(mut matches: Vec<RawMatch>) -> Vec<RawMatch> {
    matches.sort_by(|a, b| {
        a.local_range
            .start
            .cmp(&b.local_range.start)
            .then_with(|| match_len(b).cmp(&match_len(a)))
            .then_with(|| a.pair_index.cmp(&b.pair_index))
    });
    let mut accepted: Vec<RawMatch> = Vec::with_capacity(matches.len());
    for candidate in matches {
        let overlaps = accepted
            .iter()
            .any(|kept| span::ranges_overlap(&kept.local_range, &candidate.local_range));
        if !overlaps {
            accepted.push(candidate);
        }
    }
    accepted
}

/// `true` if `word` is written in ALL CAPS: two or more alphabetic
/// characters, every one of them uppercase. A single capital letter (the
/// pronoun `"I"`) never counts.
fn is_all_caps_word(word: &str) -> bool {
    let mut letters = 0u32;
    let mut all_upper = true;
    for c in word.chars().filter(|c| c.is_alphabetic()) {
        letters += 1;
        if !c.is_uppercase() {
            all_upper = false;
        }
    }
    letters >= 2 && all_upper
}

/// `true` if any whitespace-separated word in `matched` is
/// [`is_all_caps_word`] — the emphasis-capitals exception.
fn has_all_caps_word(matched: &str) -> bool {
    matched.split_whitespace().any(is_all_caps_word)
}

/// `true` if `remainder` (the sentence's text right after a match)
/// contains no alphanumeric character at all — i.e. only punctuation (and
/// whitespace) remains before the sentence ends. The sentence-final
/// exception.
fn is_sentence_final(remainder: &str) -> bool {
    !remainder.chars().any(char::is_alphanumeric)
}

/// `true` if `s` begins with `word`, case-insensitively, immediately
/// followed by a non-alphanumeric character or the end of `s`.
fn starts_with_word(s: &str, word: &str) -> bool {
    let Some(candidate) = s.get(..word.len()) else {
        return false;
    };
    candidate.eq_ignore_ascii_case(word)
        && s[word.len()..]
            .chars()
            .next()
            .is_none_or(|c| !c.is_alphanumeric())
}

/// `true` if `m` falls under one of this rule's conservative exceptions —
/// see the module docs' "Exceptions" section for what each one is and why.
fn is_exempt(text: &str, m: &RawMatch) -> bool {
    let local = &m.local_range;
    let matched = &text[local.start..local.end];
    if has_all_caps_word(matched) {
        return true;
    }
    let remainder = &text[local.end..];
    if is_sentence_final(remainder) {
        return true;
    }
    if remainder.starts_with(',') {
        return true;
    }
    let expanded = COMPILED_PAIRS[m.pair_index].expanded;
    if matches!(expanded, "it is" | "that is") && starts_with_word(remainder.trim_start(), "that") {
        return true;
    }
    false
}

/// Finds which [`COMPILED_PAIRS`] entry `matched` (an already-matched
/// finding's exact source text) is an occurrence of, comparing
/// whitespace-normalized, case-insensitive words rather than the raw
/// bytes — `matched` may contain a line break where [`compile_pattern`]'s
/// pattern allowed one, but every table entry's `expanded` field is
/// written with plain single spaces.
fn identify_pair(matched: &str) -> Option<&'static CompiledPair> {
    let words: Vec<&str> = matched.split_whitespace().collect();
    COMPILED_PAIRS.iter().find(|pair| {
        let expected: Vec<&str> = pair.expanded.split_whitespace().collect();
        expected.len() == words.len()
            && expected
                .iter()
                .zip(&words)
                .all(|(e, w)| e.eq_ignore_ascii_case(w))
    })
}

/// Applies `matched`'s capitalization to `contracted`: if `matched`'s
/// first alphabetic character is uppercase, `contracted`'s first
/// character is capitalized too (the rest of `contracted` — already
/// lowercase in the table — is left as-is); otherwise `contracted` is
/// returned unchanged. `"Do not"` -> `"Don't"`, `"do not"` -> `"don't"`.
fn apply_capitalization(matched: &str, contracted: &str) -> String {
    let starts_upper = matched
        .chars()
        .find(|c| c.is_alphabetic())
        .is_some_and(char::is_uppercase);
    if !starts_upper {
        return contracted.to_string();
    }
    let mut chars = contracted.chars();
    chars.next().map_or_else(String::new, |first| {
        first.to_uppercase().chain(chars).collect()
    })
}

/// Inserts contractions ("do not" -> "don't") when a document reads too
/// formally for its genre.
///
/// See the module docs for the exceptions this rule always respects and
/// the [`GenreFlags`] hook.
#[derive(Debug, Clone, Copy, Default)]
pub struct ContractionRule;

impl ContractionRule {
    /// Creates the rule.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Rule for ContractionRule {
    fn id(&self) -> RuleId {
        RULE_ID
    }

    fn family(&self) -> RuleFamily {
        RuleFamily::Contraction
    }

    fn gate(&self, metrics: &MetricVector, envelope: &dyn GenreEnvelope) -> Gate {
        let Some(band) = envelope.band(GATED_METRIC) else {
            return Gate::Off;
        };
        let current = metrics.contraction_ratio;
        // Only the "too formal" direction is this rule's to fix — it only
        // ever inserts a contraction, never removes one, so a document
        // already inside the band, or (unusually) above its ceiling,
        // gates Off either way: there is nothing this rule could safely
        // do about a surplus.
        if current >= band.lo {
            return Gate::Off;
        }
        // See the module docs' "Exact, per-round budgeting" section: this
        // is a generous safety ceiling only, not the real per-round limit
        // — `fix` computes that exactly, from the real document.
        Gate::Fix {
            budget: Budget::new(GATE_SAFETY_CAP),
        }
    }

    fn scan(&self, ctx: &RuleContext<'_>) -> Vec<Finding> {
        if GenreFlags::for_genre(ctx.genre()).legalish {
            return Vec::new();
        }
        let document = ctx.document();
        let mut findings = Vec::new();
        for (_, sentence) in ctx.sentences() {
            let Ok(text) = document.text(&sentence.range) else {
                continue;
            };
            let candidates: Vec<RawMatch> = scan_sentence(text)
                .into_iter()
                .filter(|m| !is_exempt(text, m))
                .collect();
            for m in resolve_sentence_matches(candidates) {
                let local = &m.local_range;
                let start = sentence.range.start + local.start;
                let end = sentence.range.start + local.end;
                let phrase = &text[local.start..local.end];
                findings.push(Finding::new(
                    RULE_ID,
                    start..end,
                    format!("\"{phrase}\" could contract to match this genre's register"),
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
        _strategy_rng: &mut StrategyRng,
    ) -> Option<Patch> {
        // A contraction has exactly one meaning-preserving rewrite (the
        // canonical contracted form), so `_strategy_rng` goes unused — see
        // `Rule::fix`'s own docs: a rule with a single strategy is free to
        // ignore it.
        //
        // The real per-round limit lives here, not in `gate` — see the
        // module docs' "Exact, per-round budgeting" section. Recompute the
        // document's true contracted/contractible counts (the same ones
        // `contraction_ratio` — and so this round's `gate` decision — is
        // built from), work out exactly how many of this round's findings
        // are needed to bring the ratio up to (never past) the envelope
        // floor, and decline every finding beyond that count.
        let band = ctx.envelope().band(GATED_METRIC)?;
        let (contracted, contractible) = friction_metrics::contraction_counts(ctx.document());
        let denominator = contracted + contractible;
        if denominator == 0 {
            return None;
        }
        #[allow(clippy::cast_precision_loss)]
        let current = contracted as f64 / denominator as f64;
        if current >= band.lo {
            return None;
        }
        let deficit = band.lo - current;
        #[allow(clippy::cast_precision_loss)]
        let denominator_f64 = denominator as f64;
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        let target_count = (deficit * denominator_f64).floor() as usize;
        if target_count == 0 {
            return None;
        }

        // This round's findings, in the same source order `scan` always
        // returns them in — leftmost-first budget spending, matching every
        // other rule in this workspace.
        let this_round = self.scan(ctx);
        let rank = this_round.iter().position(|f| f.range == finding.range)?;
        if rank >= target_count {
            return None;
        }

        let matched = ctx.document().text(&finding.range).ok()?;
        let pair = identify_pair(matched)?;
        let replacement = apply_capitalization(matched, pair.contracted);
        Some(Patch::new(
            finding.range.clone(),
            replacement,
            RULE_ID,
            Tier::Fix,
        ))
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

    fn metrics_with_ratio(ratio: f64) -> MetricVector {
        MetricVector {
            contraction_ratio: ratio,
            ..MetricVector::default()
        }
    }

    fn scan_source(source: &str) -> (friction_core::Document, Vec<Finding>) {
        let doc = document(source);
        let envelope = MapEnvelope::new();
        let findings = {
            let ctx = RuleContext::new(&doc, &NoopTagger, "blog", &envelope);
            ContractionRule::new().scan(&ctx)
        };
        (doc, findings)
    }

    /// A `contraction_ratio` band wide enough that `fix`'s own
    /// document-derived target count (see the module docs' "Exact,
    /// per-round budgeting" section) never declines a finding a test wants
    /// fixed: `lo = hi = 1.0` puts the floor above any real document's
    /// ratio, so the computed deficit -- and therefore the target count --
    /// is always large enough to cover every contractible occurrence a
    /// unit test constructs. `scan`/`fix` tests that call `fix` directly
    /// (bypassing `friction-apply`'s own gate/budget wiring) want this
    /// rather than an empty [`MapEnvelope`], which would make every `fix`
    /// call decline for lack of a band.
    fn permissive_envelope() -> MapEnvelope {
        MapEnvelope::new().with(GATED_METRIC, Envelope::new(1.0, 1.0))
    }

    // ---------------------------------------------------------------
    // Closed substitution table / cross-crate consistency
    // ---------------------------------------------------------------

    /// `COMPILED_PAIRS` is built from — and only from —
    /// `friction_metrics::contraction_pairs`, in the same order: the
    /// cross-crate consistency check that keeps this rule and the metric
    /// it gates on from ever drifting out of sync, because there is only
    /// one table.
    #[test]
    fn compiled_pairs_matches_metrics_table() {
        let source = friction_metrics::contraction_pairs();
        assert_eq!(COMPILED_PAIRS.len(), source.len());
        for (compiled, (expanded, contracted)) in COMPILED_PAIRS.iter().zip(source.iter()) {
            assert_eq!(compiled.expanded, *expanded);
            assert_eq!(compiled.contracted, *contracted);
        }
    }

    /// No `contracted` form in the table is ever itself matched as an
    /// `expanded` phrase — the "no substitution-table RHS is also an LHS"
    /// closure property, checked directly rather than merely argued about
    /// in the module docs.
    #[test]
    fn contracted_forms_never_match_as_expanded_phrases() {
        for pair in COMPILED_PAIRS.iter() {
            let matches = scan_sentence(pair.contracted);
            assert!(
                matches.is_empty(),
                "{:?} (a contracted RHS) must not match any expanded-phrase pattern",
                pair.contracted
            );
        }
    }

    // ---------------------------------------------------------------
    // GenreFlags
    // ---------------------------------------------------------------

    /// None of the v1 frozen genre set is legal-register by default.
    #[test]
    fn genre_flags_default_off_for_frozen_genres() {
        for genre in ["docs", "blog", "readme", "email", "forum"] {
            assert!(
                !GenreFlags::for_genre(genre).legalish,
                "{genre} must default to legalish: false"
            );
        }
    }

    /// An unrecognized genre also defaults to `legalish: false` — the
    /// switch is opt-in, not opt-out.
    #[test]
    fn genre_flags_default_off_for_unknown_genre() {
        assert!(!GenreFlags::for_genre("some-future-genre").legalish);
    }

    // ---------------------------------------------------------------
    // gate()
    // ---------------------------------------------------------------

    /// No envelope band for the gated metric: gate is `Off` regardless of
    /// the metric's value.
    #[test]
    fn gate_is_off_without_a_band() {
        let rule = ContractionRule::new();
        let envelope = MapEnvelope::new();
        assert_eq!(rule.gate(&metrics_with_ratio(0.1), &envelope), Gate::Off);
    }

    /// A ratio inside the band gates `Off`.
    #[test]
    fn gate_is_off_inside_band() {
        let rule = ContractionRule::new();
        let envelope = MapEnvelope::new().with(GATED_METRIC, Envelope::new(0.2, 0.8));
        assert_eq!(rule.gate(&metrics_with_ratio(0.5), &envelope), Gate::Off);
    }

    /// A ratio *above* the band's ceiling also gates `Off` — this rule
    /// only ever inserts contractions, so it has no safe move for a
    /// document that is already more contracted than the genre's own
    /// writers typically are.
    #[test]
    fn gate_is_off_above_band_ceiling() {
        let rule = ContractionRule::new();
        let envelope = MapEnvelope::new().with(GATED_METRIC, Envelope::new(0.2, 0.8));
        assert_eq!(rule.gate(&metrics_with_ratio(0.95), &envelope), Gate::Off);
    }

    /// Below the band: `gate` hands back the generous safety-cap budget —
    /// see the module docs' "Exact, per-round budgeting" section for why
    /// the real per-round count is computed in `fix`, not here.
    #[test]
    fn gate_below_band_returns_the_safety_cap_budget() {
        let rule = ContractionRule::new();
        let envelope = MapEnvelope::new().with(GATED_METRIC, Envelope::new(0.4, 0.9));
        assert_eq!(
            rule.gate(&metrics_with_ratio(0.1), &envelope),
            Gate::Fix {
                budget: Budget::new(GATE_SAFETY_CAP)
            }
        );
    }

    /// A ratio only marginally below the band still gates `Fix` (`gate`
    /// alone cannot tell "marginal" from "far below" without the real
    /// document) — but see
    /// `fix_declines_every_finding_when_the_real_deficit_rounds_to_zero`
    /// below for where that marginal case actually gets resolved to no
    /// patches, exactly as it used to be resolved here before the fix.
    #[test]
    fn gate_marginally_below_band_still_gates_fix() {
        let rule = ContractionRule::new();
        let envelope = MapEnvelope::new().with(GATED_METRIC, Envelope::new(0.4, 0.9));
        assert_eq!(
            rule.gate(&metrics_with_ratio(0.35), &envelope),
            Gate::Fix {
                budget: Budget::new(GATE_SAFETY_CAP)
            }
        );
    }

    // ---------------------------------------------------------------
    // scan(): basic matching, ordering, overlap resolution
    // ---------------------------------------------------------------

    #[test]
    fn scan_finds_a_simple_expanded_phrase() {
        let (doc, findings) = scan_source("Do not stop.");
        assert_eq!(findings.len(), 1);
        assert_eq!(&doc.source()[findings[0].range.clone()], "Do not");
    }

    #[test]
    fn scan_finds_every_occurrence_in_source_order() {
        let (doc, findings) =
            scan_source("Do not stop. Plain sentence here. It will not work otherwise.");
        assert_eq!(findings.len(), 2);
        assert_eq!(&doc.source()[findings[0].range.clone()], "Do not");
        assert_eq!(&doc.source()[findings[1].range.clone()], "will not");
        assert!(findings[0].range.start < findings[1].range.start);
    }

    /// `"She is not late."` matches both the `"she is"` and `"is not"`
    /// pairs, overlapping on the shared word `"is"`; only the
    /// leftmost-starting match (`"She is"`) survives.
    #[test]
    fn scan_resolves_overlapping_pair_matches_leftmost_first() {
        let (doc, findings) = scan_source("She is not late.");
        assert_eq!(findings.len(), 1);
        assert_eq!(&doc.source()[findings[0].range.clone()], "She is");
    }

    /// A match spanning a markdown source line break (matched via `\s+`,
    /// not a literal single space) is still found and fixed as one
    /// contiguous span.
    #[test]
    fn scan_matches_phrase_split_by_a_line_break() {
        let (doc, findings) = scan_source("It is\nfine, we are told.");
        assert_eq!(findings.len(), 2);
        assert_eq!(&doc.source()[findings[0].range.clone()], "It is");
    }

    // ---------------------------------------------------------------
    // Exceptions
    // ---------------------------------------------------------------

    #[test]
    fn exception_all_caps_emphasis() {
        let (_, findings) = scan_source("Do NOT stop.");
        assert!(findings.is_empty());
    }

    /// A single capital `"I"` does not itself count as emphasis: `"I am
    /// ready."` still contracts.
    #[test]
    fn exception_lone_capital_i_is_not_emphasis() {
        let (doc, findings) = scan_source("I am ready.");
        assert_eq!(findings.len(), 1);
        assert_eq!(&doc.source()[findings[0].range.clone()], "I am");
    }

    #[test]
    fn exception_sentence_final_auxiliary() {
        let (_, findings) = scan_source("Yes, it is.");
        assert!(findings.is_empty());
    }

    #[test]
    fn exception_comma_boundary() {
        let (_, findings) = scan_source("It is, however, still running.");
        assert!(findings.is_empty());
    }

    /// The cleft-"that" exception is specific to `"it is"`/`"that is"`.
    #[test]
    fn exception_cleft_that_for_it_is() {
        let (_, findings) = scan_source("It is that simple.");
        assert!(findings.is_empty());
    }

    #[test]
    fn exception_cleft_that_for_that_is() {
        let (_, findings) = scan_source("That is that.");
        assert!(findings.is_empty());
    }

    /// A sentence with one exempt match and one ordinary match only fixes
    /// the ordinary one.
    #[test]
    fn exception_does_not_suppress_unrelated_matches_in_the_same_sentence() {
        let (doc, findings) = scan_source("It is that the numbers do not add up.");
        assert_eq!(findings.len(), 1);
        assert_eq!(&doc.source()[findings[0].range.clone()], "do not");
    }

    // ---------------------------------------------------------------
    // fix(): capitalization, pair identification
    // ---------------------------------------------------------------

    fn fix_first(source: &str) -> String {
        let doc = document(source);
        let envelope = permissive_envelope();
        let ctx = RuleContext::new(&doc, &NoopTagger, "blog", &envelope);
        let rule = ContractionRule::new();
        let finding = rule
            .scan(&ctx)
            .into_iter()
            .next()
            .expect("expected a finding");
        let mut rng = StrategyRng::from_seed(0);
        let patch = rule
            .fix(&finding, &ctx, &mut rng)
            .expect("expected a patch");
        assert_eq!(patch.tier, Tier::Fix);
        patch.replacement
    }

    #[test]
    fn fix_lowercase_preserves_case() {
        assert_eq!(fix_first("do not stop."), "don't");
    }

    #[test]
    fn fix_capitalized_preserves_case() {
        assert_eq!(fix_first("Do not stop."), "Don't");
    }

    #[test]
    fn fix_single_word_pair() {
        assert_eq!(fix_first("Cannot proceed."), "Can't");
    }

    #[test]
    fn fix_i_am_pair() {
        assert_eq!(fix_first("I am ready."), "I'm");
    }

    #[test]
    fn fix_applies_cleanly_to_source() {
        let source = "Do not stop working on it.";
        let doc = document(source);
        let envelope = permissive_envelope();
        let ctx = RuleContext::new(&doc, &NoopTagger, "blog", &envelope);
        let rule = ContractionRule::new();
        let finding = &rule.scan(&ctx)[0];
        let mut rng = StrategyRng::from_seed(0);
        let patch = rule.fix(finding, &ctx, &mut rng).expect("expected a patch");

        let mut applied = source.to_string();
        applied.replace_range(patch.range, &patch.replacement);
        assert_eq!(applied, "Don't stop working on it.");
    }

    // ---------------------------------------------------------------
    // Idempotence and determinism
    // ---------------------------------------------------------------

    /// Applying every finding this rule proposes for a document, then
    /// scanning the result again, finds nothing left to fix.
    #[test]
    fn fixing_a_document_is_idempotent() {
        let source =
            "Do not stop. It is not ready. We are not done. Cannot proceed. I will not wait.";
        let doc = document(source);
        let envelope = permissive_envelope();
        let mut patches: Vec<Patch> = {
            let ctx = RuleContext::new(&doc, &NoopTagger, "blog", &envelope);
            let rule = ContractionRule::new();
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

        let (_, findings_after) = scan_source(&fixed);
        assert!(
            findings_after.is_empty(),
            "expected no findings left after fixing, got {findings_after:?} in {fixed:?}"
        );
    }

    /// Scanning and fixing the same source twice, from independently
    /// constructed contexts, yields byte-identical patches.
    #[test]
    fn fixing_the_same_source_twice_is_byte_identical() {
        let source = "Do not stop. It is not ready. We are not done.";

        let run = || {
            let doc = document(source);
            let envelope = permissive_envelope();
            let ctx = RuleContext::new(&doc, &NoopTagger, "blog", &envelope);
            let rule = ContractionRule::new();
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

    // ---------------------------------------------------------------
    // fix(): exact, document-derived per-round target count
    // ---------------------------------------------------------------

    /// The convergence regression this fix exists for: a document with far
    /// more contractible occurrences than the old fixed per-fix-effect
    /// assumption (`10`) expected still gets exactly the right number of
    /// fixes computed from the *real* document, in a single `fix` pass —
    /// no dependency on a lucky guessed constant.
    #[test]
    fn fix_computes_the_exact_document_derived_target_count() {
        // 20 independent sentences, each with one contractible "is not";
        // denominator 20, current ratio 0.0, band lo 0.5 -> deficit 0.5 ->
        // target_count = floor(0.5 * 20) = 10 (hand-computed).
        let source = (0..20)
            .map(|i| format!("Sentence {i} is not ready."))
            .collect::<Vec<_>>()
            .join(" ");
        let doc = document(&source);
        let envelope = MapEnvelope::new().with(GATED_METRIC, Envelope::new(0.5, 0.9));
        let ctx = RuleContext::new(&doc, &NoopTagger, "forum", &envelope);
        let rule = ContractionRule::new();
        let findings = rule.scan(&ctx);
        assert_eq!(findings.len(), 20);

        let fixed_count = findings
            .iter()
            .filter(|finding| {
                let mut rng = StrategyRng::from_seed(0);
                rule.fix(finding, &ctx, &mut rng).is_some()
            })
            .count();
        assert_eq!(
            fixed_count, 10,
            "expected the hand-computed target count of 10 fixes, got {fixed_count}"
        );
    }

    /// The exact target count is spent leftmost-first, same as every other
    /// budget in this workspace: the first 10 (of 20) findings in source
    /// order get fixed, the rest decline.
    #[test]
    fn fix_spends_the_exact_target_count_leftmost_first() {
        let source = (0..20)
            .map(|i| format!("Sentence {i} is not ready."))
            .collect::<Vec<_>>()
            .join(" ");
        let doc = document(&source);
        let envelope = MapEnvelope::new().with(GATED_METRIC, Envelope::new(0.5, 0.9));
        let ctx = RuleContext::new(&doc, &NoopTagger, "forum", &envelope);
        let rule = ContractionRule::new();
        let findings = rule.scan(&ctx);

        let outcomes: Vec<bool> = findings
            .iter()
            .map(|finding| {
                let mut rng = StrategyRng::from_seed(0);
                rule.fix(finding, &ctx, &mut rng).is_some()
            })
            .collect();
        let expected: Vec<bool> = (0..20).map(|i| i < 10).collect();
        assert_eq!(outcomes, expected);
    }

    /// When the real document's deficit is smaller than a single fix's own
    /// exact effect (`1 / denominator`), the target count rounds down to
    /// zero and `fix` declines every finding this round — the same
    /// "accept staying marginally outside rather than force an overshoot"
    /// behavior `gate`'s own floor rounding used to provide before this
    /// fix, now computed exactly instead of estimated.
    ///
    /// Hand-computed: 3 contractible "It is" occurrences (denominator 3,
    /// current 0.0), band lo 0.3 -> deficit 0.3 -> `target_count` =
    /// floor(0.3 * 3) = floor(0.9) = 0.
    #[test]
    fn fix_declines_every_finding_when_the_real_deficit_rounds_to_zero() {
        let source = "It is nice. It is nice. It is nice.";
        let doc = document(source);
        let envelope = MapEnvelope::new().with(GATED_METRIC, Envelope::new(0.3, 0.9));
        let ctx = RuleContext::new(&doc, &NoopTagger, "forum", &envelope);
        let rule = ContractionRule::new();
        let findings = rule.scan(&ctx);
        assert_eq!(findings.len(), 3);

        for finding in &findings {
            let mut rng = StrategyRng::from_seed(0);
            assert!(
                rule.fix(finding, &ctx, &mut rng).is_none(),
                "expected every finding to decline when the real deficit rounds to zero"
            );
        }
    }

    /// `fix` declines outright (rather than panicking or fixing
    /// unconditionally) when the envelope has no band for
    /// `contraction_ratio` at all.
    #[test]
    fn fix_declines_without_a_band() {
        let source = "Do not stop.";
        let doc = document(source);
        let envelope = MapEnvelope::new();
        let ctx = RuleContext::new(&doc, &NoopTagger, "blog", &envelope);
        let rule = ContractionRule::new();
        let finding = &rule.scan(&ctx)[0];
        let mut rng = StrategyRng::from_seed(0);
        assert!(rule.fix(finding, &ctx, &mut rng).is_none());
    }

    // ---------------------------------------------------------------
    // GenreFlags wired into scan()
    // ---------------------------------------------------------------

    /// Every genre in the v1 frozen set behaves identically (not
    /// legalish): `scan` finds the same findings regardless of which one
    /// is passed.
    #[test]
    fn legalish_flag_off_leaves_scan_unaffected_for_every_frozen_genre() {
        let source = "Do not stop. It is not ready.";
        for genre in ["docs", "blog", "readme", "email", "forum"] {
            let doc = document(source);
            let envelope = MapEnvelope::new();
            let ctx = RuleContext::new(&doc, &NoopTagger, genre, &envelope);
            assert_eq!(
                ContractionRule::new().scan(&ctx).len(),
                2,
                "genre {genre} must not be treated as legalish"
            );
        }
    }
}
