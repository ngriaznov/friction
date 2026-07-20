//! [`SentenceSplitRule`]: carves an over-long sentence into two at its
//! strongest clause boundary.
//!
//! # When this rule fires
//!
//! Gated on [`GATED_METRIC`] (`sentence_length_cv`) sitting *below* the
//! genre's envelope floor — a document whose sentences are all suspiciously
//! close to the same length. Even then, an individual sentence is only a
//! candidate if it is also "over-long": above [`overlong_threshold_tokens`],
//! this genre's approximate train-human p90 sentence length. Both
//! conditions have to hold: a uniform-but-short document (every sentence
//! comfortably under the genre's typical length) has nothing to split, and
//! a document with one long sentence but otherwise healthy variety already
//! sits inside its envelope and gates `Off`.
//!
//! # Boundary selection
//!
//! Two boundary classes, in order of preference:
//!
//! 1. **Semicolon.** Any `;` with real content on both sides (not the
//!    sentence's own final mark) — semicolons overwhelmingly join two
//!    independent clauses in edited English, so no further check is
//!    needed.
//! 2. **Coordinator.** A literal `", and "`, `", but "`, or `", so "`,
//!    accepted only when both:
//!    - the tagger's next token after it is subject-like (a pronoun, a
//!      determiner, or a proper noun — see [`SUBJECT_LIKE_TAGS`]). This is
//!      what keeps the rule from splitting a flat noun-phrase list
//!      (`"screws, bolts, and washers"`, where "washers" tags as a plain
//!      common noun `NNS`) instead of genuine clause coordination
//!      (`"..., and the team shipped it"`, where "the" tags `DT`) — the
//!      same false-positive class [`friction_nlp::HeuristicParser`]'s own
//!      coordination detector is built to recognize, checked here from the
//!      opposite direction (reject a plain noun-phrase continuation)
//!      instead.
//!    - [`precedes_unresolved_comma`] finds no *other*, unaccounted-for
//!      comma between the start of this clause and the coordinator's own
//!      comma. Without this check, a natural three-clause serial list
//!      (`"A does X, B does Y, and C does Z."`) would only ever offer the
//!      *last* `", and "` as a candidate (the earlier `", "` between the
//!      first two clauses is never itself a `COORDINATORS` pattern), and
//!      splitting only there leaves the first two clauses joined by nothing
//!      but a bare comma — a comma splice this rule would have *introduced*
//!      into text that had none. See that function's own docs.
//!
//! Neither a semicolon match nor a passing coordinator match may begin
//! inside an open quotation ([`inside_open_quote`]) — splitting inside a
//! direct quotation would silently rewrite the exact words and punctuation
//! attributed to whoever is being quoted, which this rule's "punctuation
//! only, on the document's own prose" safety case (see "Fix" below) never
//! accounted for.
//!
//! A sentence's semicolons are tried first; the coordinator patterns are
//! only searched (which needs a tagger call `scan` would otherwise skip)
//! when the sentence has none. Among several candidates of the winning
//! class, the one nearest the sentence's own midpoint wins — a balanced
//! split, so the two halves are each as likely as possible to already sit
//! under the over-long threshold and not need a further round (see the
//! module docs' "Idempotence" section below for why an unbalanced,
//! always-pick-the-first choice is the wrong default here). At most one
//! boundary — one split — is proposed per sentence per round, honoring
//! this workspace's budget semantics.
//!
//! # Fix: punctuation (and, for `", but "`/`", so "`, the connective word)
//!
//! A semicolon or `", and "` boundary's marker is replaced by `". "`,
//! recapitalizing the word that now opens the second sentence if it starts
//! lowercase — the exact delete-and-recapitalize shape
//! [`crate::families::connective::ConnectiveSurgery`]'s own **delete**
//! strategy uses, with an inserted `". "` in place of nothing. Nothing is
//! reordered and no proposition is dropped, only a clause boundary's
//! punctuation and the following letter's case change — squarely within
//! what this workspace's tier discipline allows at `Fix` tier.
//!
//! A `", but "`/`", so "` boundary keeps its connective word instead of
//! deleting it: `". But "`/`". So "` replaces the whole marker, and the
//! word that used to open the second clause keeps its original case
//! unchanged. `"but"` and `"so"` carry a real discourse relation (contrast,
//! cause) the sentence would otherwise assert nothing about — dropping
//! them outright (as this rule used to) is more than a punctuation change,
//! so this rule now only ever changes case/punctuation *around* the
//! connective, never the connective itself.
//!
//! # Idempotence
//!
//! Idempotent by construction, the same way every substitution-table rule
//! in this workspace is: the patch's replacement text is always exactly
//! `". "`, or `". But "`/`". So "`, plus at most one recapitalized letter —
//! never a semicolon, never a comma — so it can never itself contain a
//! [`COORDINATORS`] pattern (which always starts with a comma) or a
//! semicolon this rule (or a later round of it) would match again at that
//! position (checked directly by this module's
//! `split_replacement_never_reintroduces_a_boundary_marker` test, the same
//! "no substitution-table RHS is also an LHS" closure this workspace
//! requires of every table like this one).
//!
//! A single round only ever performs *one* split per sentence, so a
//! sentence far enough over the threshold to need two splits takes two
//! rounds — each one re-measures `sentence_length_cv` and each sentence's
//! own length fresh, so this is ordinary budgeted convergence, not an
//! oscillation: nearest-to-middle boundary selection also means each split
//! shrinks its sentence roughly in half rather than lopping off a small
//! fragment and leaving a still-over-long remainder that would need many
//! more rounds to resolve (`split_prefers_the_more_balanced_boundary` and
//! this crate's own golden fixture `two_boundaries_needs_two_rounds`
//! exercise this directly).

use friction_core::{Finding, MetricVector, Patch, RuleId, Sentence, Tier};
use friction_nlp::TaggedToken;

use crate::budget::Budget;
use crate::context::{GenreEnvelope, RuleContext};
use crate::rule::{Gate, Rule, RuleFamily};
use crate::strategy::StrategyRng;

use super::token_count;

/// This rule's stable identifier.
const RULE_ID: RuleId = RuleId::new("rhythm.split");

/// The [`MetricVector`] field this rule gates on.
const GATED_METRIC: &str = "sentence_length_cv";

/// How much splitting one over-long sentence is projected to move
/// [`GATED_METRIC`], for [`Budget::from_envelope_excess`].
///
/// `gate` runs before `scan` (see [`Rule::gate`]'s own docs) and receives
/// only the round's already-computed [`MetricVector`], never the real
/// document — so, exactly like every other density-gated rule in this
/// workspace (see e.g. `crate::families::connective::ConnectiveSurgery`'s
/// own `PER_FIX_EFFECT` docs), it cannot compute this document's *exact*
/// per-split effect on a coefficient of variation that depends on every
/// sentence length in the document at once. `0.05` — five hundredths of a
/// point of CV, a small fraction of the roughly `0.5`-to-`1.3`-wide bands
/// the shipped envelope pack uses for this metric — is a deliberately
/// conservative flat estimate: an understatement only ever yields a
/// smaller budget than the document could truly support, never a larger
/// one, and a later round's fresh `gate` call (re-measured against the
/// actual result) tops up any shortfall.
const PER_FIX_EFFECT: f64 = 0.05;

/// Coordinating conjunctions this rule treats as a candidate clause
/// boundary when preceded by a comma and followed by a subject-like token
/// (see [`SUBJECT_LIKE_TAGS`]) — see the module docs' "Boundary selection"
/// section.
const COORDINATORS: &[&str] = &[", and ", ", but ", ", so "];

/// Part-of-speech tags (Penn-Treebank-style, matching
/// `friction_nlp::NlpruleTagger`'s dictionary) this rule accepts as
/// "subject-like" evidence that a coordinator boundary opens a genuine new
/// clause rather than continuing a flat noun-phrase list: personal and
/// wh-pronouns (`PRP`, `WP`), determiners (`DT`, `PDT`, `WDT`), and proper
/// nouns (`NNP`, `NNPS`). A plain common noun (`NN`/`NNS`) — the shape a
/// list's next item overwhelmingly takes (`"screws, bolts, and washers"`)
/// — is deliberately excluded.
const SUBJECT_LIKE_TAGS: &[&str] = &["PRP", "WP", "DT", "PDT", "WDT", "NNP", "NNPS"];

/// Each genre's approximate train-human p90 sentence length, in tokens —
/// the length above which a sentence counts as "over-long" and becomes a
/// splitting candidate.
///
/// `friction-packs`' shipped envelope pack carries only an `[lo, hi]` band
/// for `sentence_length_mean` itself, not a raw p90 statistic, so this
/// table is derived once, by hand, from that band's `hi` edge (the
/// envelope's own upper bound on human-typical mean sentence length, per
/// `envelope-v2.toml` as of writing) and a single documented multiplier:
/// prose sentence-length distributions are typically right-skewed, with
/// the 90th percentile commonly running somewhere around 1.6x-1.8x the
/// mean; `1.75` is a deliberately generic value from the middle of that
/// range, applied uniformly to every genre rather than claiming a false
/// precision this pack's data cannot support. Each entry below is
/// `round(mean_hi * 1.75)`:
///
/// | genre  | `sentence_length_mean.hi` | p90 threshold |
/// |--------|--------------------------:|---------------:|
/// | blog   | 22.612...                  | 40              |
/// | docs   | 11.370...                  | 20              |
/// | email  | 8.368...                   | 15              |
/// | forum  | 21.545...                  | 38              |
/// | readme | 8.028...                   | 14              |
///
/// A genre outside the v1 frozen set (`docs`, `blog`, `readme`, `email`,
/// `forum`) has no entry: [`overlong_threshold_tokens`] returns `None`,
/// and [`SentenceSplitRule::scan`] finds nothing for it rather than
/// guessing a threshold with no data behind it.
fn overlong_threshold_tokens(genre: &str) -> Option<usize> {
    match genre {
        "blog" => Some(40),
        "docs" => Some(20),
        "email" => Some(15),
        "forum" => Some(38),
        "readme" => Some(14),
        _ => None,
    }
}

/// Which class of clause boundary a [`Boundary`] is — see the module docs'
/// "Boundary selection" section.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BoundaryClass {
    Semicolon,
    Coordinator,
}

/// One candidate split point within a sentence.
#[derive(Debug, Clone, Copy)]
struct Boundary {
    /// Byte offset, into the document's original source, where the
    /// boundary marker begins.
    start: usize,
    /// The marker's own byte length (`1` for a semicolon; a
    /// [`COORDINATORS`] entry's length for a coordinator).
    marker_len: usize,
    class: BoundaryClass,
}

/// `true` if `remainder` (a sentence's text right after a candidate
/// boundary) contains no alphanumeric character at all — i.e. the boundary
/// sits at the very end of the sentence, with nothing left to split off as
/// a second clause. Same convention
/// `crate::families::contraction::ContractionRule`'s own sentence-final
/// exception uses.
fn is_sentence_final(remainder: &str) -> bool {
    !remainder.chars().any(char::is_alphanumeric)
}

/// `true` if byte offset `pos` in `text` (a sentence's own text) falls
/// inside a still-open quotation — a straight double quote (`"`) with an
/// odd number of unescaped occurrences before it, or a Unicode "smart"
/// opening quote (`\u{201C}`) that outnumbers its own closing quote
/// (`\u{201D}`) before it. Either signal alone is enough: a splitting
/// boundary that sits inside a direct quotation would silently rewrite the
/// exact words and punctuation attributed to whoever is being quoted (see
/// the module docs' "Boundary selection" section), which is never a "safe,
/// punctuation-only" change the way splitting the document's own prose is.
///
/// This is a byte-counting heuristic, not a real quote parser — a
/// literal, unpaired quotation mark elsewhere in the sentence (e.g. an
/// apostrophe-like usage this function does not attempt to special-case)
/// could in principle miscount, but the failure mode of a miscount is
/// always "decline a boundary this rule could safely have used", never
/// "split inside a quotation" — the same one-directional-safe bias
/// [`is_sentence_final`] and the subject-like check already take.
fn inside_open_quote(text: &str, pos: usize) -> bool {
    let before = &text[..pos];
    let straight_open = before.matches('"').count() % 2 == 1;
    let smart_open = before.matches('\u{201C}').count() > before.matches('\u{201D}').count();
    straight_open || smart_open
}

/// Every semicolon boundary in `text` (a sentence's own text), with real
/// (alphanumeric) content on both sides — see [`is_sentence_final`] for
/// the "real content after" check and the module docs for why a semicolon
/// needs no further (tagger-based) gating. Excludes any semicolon
/// [`inside_open_quote`] finds inside a direct quotation.
fn find_semicolon_boundaries(text: &str, sentence_start: usize) -> Vec<Boundary> {
    text.match_indices(';')
        .filter(|&(rel, _)| {
            text[..rel].chars().any(char::is_alphanumeric)
                && !is_sentence_final(&text[rel + 1..])
                && !inside_open_quote(text, rel)
        })
        .map(|(rel, _)| Boundary {
            start: sentence_start + rel,
            marker_len: ';'.len_utf8(),
            class: BoundaryClass::Semicolon,
        })
        .collect()
}

/// `true` if the first tagged token starting at or after byte offset `pos`
/// (into the document source `tagged`'s spans address) carries one of
/// [`SUBJECT_LIKE_TAGS`].
fn has_subject_like_token_at_or_after(tagged: &[TaggedToken], pos: usize) -> bool {
    tagged
        .iter()
        .find(|token| token.token.range.start >= pos)
        .is_some_and(|token| SUBJECT_LIKE_TAGS.contains(&token.pos.as_str()))
}

/// `true` if there is an earlier, *unaccounted-for* comma between the start
/// of `text` (a sentence's own text — or, more precisely, the start of the
/// current clause run: this function does not look back past a `.`/`!`/
/// `?`/`:`, though a single sentence should not usually contain one) and
/// `comma_rel` (the byte offset, in `text`, of a coordinator boundary's own
/// leading comma).
///
/// "Unaccounted-for" means that earlier comma is not itself immediately
/// followed by one of [`COORDINATORS`]' own trailing words (`"and "`,
/// `"but "`, `"so "`) — i.e. it is a bare, unconjoined comma splice
/// waypoint, the shape a natural three-clause serial list produces
/// (`"A does X, B does Y, and C does Z."`: the comma after `"X,"` has
/// nothing but `"B does Y"` after it). Accepting a coordinator boundary
/// with one of these sitting in front of it would leave everything before
/// that earlier comma and everything after it joined by nothing but a bare
/// comma once this rule's own split lands at the *later* boundary — a
/// comma splice this rule would have introduced into text that had none.
/// See the module docs' "Boundary selection" section.
fn precedes_unresolved_comma(text: &str, comma_rel: usize) -> bool {
    let scan_start = text[..comma_rel]
        .rfind(['.', '!', '?', ':'])
        .map_or(0, |i| i + 1);
    text[scan_start..comma_rel]
        .match_indices(',')
        .any(|(idx, _)| {
            let after = &text[scan_start + idx + 1..];
            !COORDINATORS
                .iter()
                .any(|pattern| after.starts_with(&pattern[1..]))
        })
}

/// The capitalized connective word [`Boundary::connective`] keeps for a
/// `", but "`/`", so "` pattern, or `None` for `", and "` (whose marker is
/// dropped entirely, same as a semicolon) — see the module docs' "Fix"
/// section.
fn connective_for(pattern: &str) -> Option<&'static str> {
    match pattern {
        ", but " => Some("But"),
        ", so " => Some("So"),
        _ => None,
    }
}

/// Every [`COORDINATORS`] boundary in `text` whose next tagged token is
/// subject-like and which does not risk leaving a comma splice behind — see
/// the module docs' "Boundary selection" section. `tagged` must be this
/// same sentence's tagged tokens (document-absolute spans, as
/// [`RuleContext::tag_sentence`] produces).
fn find_coordinator_boundaries(
    text: &str,
    sentence_start: usize,
    tagged: &[TaggedToken],
) -> Vec<Boundary> {
    let mut boundaries = Vec::new();
    for pattern in COORDINATORS {
        let mut search_from = 0usize;
        while let Some(rel) = text[search_from..].find(pattern) {
            let comma_rel = search_from + rel;
            let abs_start = sentence_start + comma_rel;
            let marker_end = abs_start + pattern.len();
            search_from += rel + pattern.len();
            if has_subject_like_token_at_or_after(tagged, marker_end)
                && !inside_open_quote(text, comma_rel)
                && !precedes_unresolved_comma(text, comma_rel)
            {
                boundaries.push(Boundary {
                    start: abs_start,
                    marker_len: pattern.len(),
                    class: BoundaryClass::Coordinator,
                });
            }
        }
    }
    boundaries
}

/// The boundary in `boundaries` nearest `sentence`'s own midpoint,
/// breaking a tie toward the earlier (lower-offset) one — see the module
/// docs' "Boundary selection" section for why this (not "the first
/// occurrence") is the right default.
fn nearest_to_middle(boundaries: &[Boundary], sentence: &Sentence) -> Option<Boundary> {
    let midpoint = sentence.range.start + (sentence.range.end - sentence.range.start) / 2;
    boundaries
        .iter()
        .copied()
        .min_by_key(|boundary| (boundary.start.abs_diff(midpoint), boundary.start))
}

/// The strongest boundary in `text` (a sentence's own text): every
/// semicolon boundary if there is at least one, otherwise every
/// coordinator boundary (which needs `tagged`, this sentence's tagged
/// tokens) — see the module docs' "Boundary selection" section.
fn strongest_boundary(text: &str, sentence: &Sentence, tagged: &[TaggedToken]) -> Option<Boundary> {
    let semicolons = find_semicolon_boundaries(text, sentence.range.start);
    if !semicolons.is_empty() {
        return nearest_to_middle(&semicolons, sentence);
    }
    let coordinators = find_coordinator_boundaries(text, sentence.range.start, tagged);
    nearest_to_middle(&coordinators, sentence)
}

/// Byte offset of the first non-whitespace character in `source` at or
/// after `from`, or `source.len()` if none remains. Same helper
/// `crate::families::connective::ConnectiveSurgery`'s own `skip_whitespace`
/// provides, duplicated locally per this workspace's small-per-module-
/// helper convention.
fn skip_whitespace(source: &str, from: usize) -> usize {
    let rest = &source[from..];
    let stop = rest
        .find(|c: char| !c.is_whitespace())
        .unwrap_or(rest.len());
    from + stop
}

/// Builds the split patch for a boundary starting at `boundary_start`
/// (byte offset into `source`) with marker length `marker_len` — see the
/// module docs' "Fix" section.
///
/// With no `connective`: replaces the marker and any whitespace after it
/// with `". "`, folding in the recapitalization of the next word when it
/// starts lowercase (a semicolon or `", and "` boundary).
///
/// With `connective` (`"But"`/`"So"`, for a `", but "`/`", so "` boundary):
/// replaces the marker and any whitespace after it with `". <connective> "`
/// instead — the word that used to open the second clause is left exactly
/// as it was in the source (never recapitalized, since it is no longer
/// sentence-initial: `"But"`/`"So"` is).
fn build_split_patch(
    source: &str,
    boundary_start: usize,
    marker_len: usize,
    connective: Option<&str>,
) -> Patch {
    let after_marker = boundary_start + marker_len;
    let ws_end = skip_whitespace(source, after_marker);
    if let Some(word) = connective {
        return Patch::new(
            boundary_start..ws_end,
            format!(". {word} "),
            RULE_ID,
            Tier::Fix,
        );
    }
    let tail = match source[ws_end..].chars().next() {
        Some(c) if c.is_lowercase() => c.to_uppercase().collect::<String>(),
        _ => String::new(),
    };
    let patch_end = ws_end + tail.chars().next().map_or(0, char::len_utf8);
    Patch::new(
        boundary_start..patch_end,
        format!(". {tail}"),
        RULE_ID,
        Tier::Fix,
    )
}

/// Carves an over-long sentence into two at its strongest clause boundary
/// when the document's sentence lengths read as suspiciously uniform. See
/// the module docs for the full gate/scan/fix shape.
#[derive(Debug, Clone, Copy, Default)]
pub struct SentenceSplitRule;

impl SentenceSplitRule {
    /// Creates the rule.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Rule for SentenceSplitRule {
    fn id(&self) -> RuleId {
        RULE_ID
    }

    fn family(&self) -> RuleFamily {
        RuleFamily::Rhythm
    }

    fn gate(&self, metrics: &MetricVector, envelope: &dyn GenreEnvelope) -> Gate {
        let Some(band) = envelope.band(GATED_METRIC) else {
            return Gate::Off;
        };
        let current = metrics.sentence_length_cv;
        // This rule only ever splits a sentence — which nudges the
        // document's sentence lengths *away* from a uniform middle, i.e.
        // it only ever has a safe move when the document reads as too
        // uniform (current below the band's floor). A CV already inside
        // the band, or (unusually) above its ceiling, gates Off either
        // way.
        if current >= band.lo {
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
        let Some(threshold) = overlong_threshold_tokens(ctx.genre()) else {
            return Vec::new();
        };
        let document = ctx.document();

        // (token length desc, sentence start asc, Finding) — see the
        // module docs' "Boundary selection" section for why processing
        // order is length-first rather than plain source order.
        let mut candidates: Vec<(usize, usize, Finding)> = Vec::new();
        for (_, sentence) in ctx.sentences() {
            let Ok(text) = document.text(&sentence.range) else {
                continue;
            };
            let len = token_count(text);
            if len <= threshold {
                continue;
            }
            let tagged = ctx.tag_sentence(sentence);
            let Some(boundary) = strongest_boundary(text, sentence, &tagged) else {
                continue;
            };
            let range = boundary.start..(boundary.start + boundary.marker_len);
            let boundary_kind = match boundary.class {
                BoundaryClass::Semicolon => "semicolon",
                BoundaryClass::Coordinator => "coordinator",
            };
            let finding = Finding::new(
                RULE_ID,
                range,
                format!(
                    "sentence is {len} tokens, above this genre's over-long threshold of \
                     {threshold}; splitting at the nearest-to-middle {boundary_kind} boundary"
                ),
                Tier::Fix,
            );
            candidates.push((len, sentence.range.start, finding));
        }

        candidates.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        candidates
            .into_iter()
            .map(|(_, _, finding)| finding)
            .collect()
    }

    fn fix(
        &self,
        finding: &Finding,
        ctx: &RuleContext<'_>,
        _strategy_rng: &mut StrategyRng,
    ) -> Option<Patch> {
        // A split boundary has exactly one meaning-preserving fix (replace
        // the marker with ". "/". But "/". So ", recapitalizing if needed),
        // so `_strategy_rng` goes unused — see `Rule::fix`'s own docs: a
        // rule with a single strategy is free to ignore it.
        let source = ctx.document().source();
        let marker = source.get(finding.range.clone())?;
        debug_assert!(
            marker == ";" || COORDINATORS.contains(&marker),
            "a Finding from SentenceSplitRule::scan always names a recognized boundary marker, \
             got {marker:?}"
        );
        let connective = connective_for(marker);
        Some(build_split_patch(
            source,
            finding.range.start,
            marker.len(),
            connective,
        ))
    }
}

#[cfg(test)]
mod tests {
    use friction_core::Envelope;
    use friction_nlp::{NlpruleTagger, SrxSegmenter};

    use super::*;
    use crate::context::MapEnvelope;

    fn document(source: &str) -> friction_core::Document {
        let parsed = friction_parse::parse(source).expect("valid markdown parses");
        friction_nlp::segment_document(&parsed, &SrxSegmenter::new())
            .expect("segmentation succeeds")
    }

    fn metrics_with_cv(cv: f64) -> MetricVector {
        MetricVector {
            sentence_length_cv: cv,
            ..MetricVector::default()
        }
    }

    fn tagger() -> NlpruleTagger {
        NlpruleTagger::new().expect("embedded tagger model loads")
    }

    fn apply(source: &str, patch: &Patch) -> String {
        let mut applied = source.to_string();
        applied.replace_range(patch.range.clone(), &patch.replacement);
        applied
    }

    // ---------------------------------------------------------------
    // overlong_threshold_tokens / genre table
    // ---------------------------------------------------------------

    #[test]
    fn overlong_threshold_covers_every_frozen_genre() {
        for genre in ["docs", "blog", "readme", "email", "forum"] {
            assert!(
                overlong_threshold_tokens(genre).is_some(),
                "{genre} must have a threshold"
            );
        }
        assert_eq!(overlong_threshold_tokens("some-future-genre"), None);
    }

    #[test]
    fn overlong_threshold_hand_computed_values() {
        assert_eq!(overlong_threshold_tokens("blog"), Some(40));
        assert_eq!(overlong_threshold_tokens("docs"), Some(20));
        assert_eq!(overlong_threshold_tokens("email"), Some(15));
        assert_eq!(overlong_threshold_tokens("forum"), Some(38));
        assert_eq!(overlong_threshold_tokens("readme"), Some(14));
    }

    // ---------------------------------------------------------------
    // gate()
    // ---------------------------------------------------------------

    #[test]
    fn gate_is_off_without_a_band() {
        let rule = SentenceSplitRule::new();
        let envelope = MapEnvelope::new();
        assert_eq!(rule.gate(&metrics_with_cv(0.3), &envelope), Gate::Off);
    }

    #[test]
    fn gate_is_off_inside_band() {
        let rule = SentenceSplitRule::new();
        let envelope = MapEnvelope::new().with(GATED_METRIC, Envelope::new(0.5, 1.0));
        assert_eq!(rule.gate(&metrics_with_cv(0.7), &envelope), Gate::Off);
    }

    #[test]
    fn gate_is_off_above_band_ceiling() {
        let rule = SentenceSplitRule::new();
        let envelope = MapEnvelope::new().with(GATED_METRIC, Envelope::new(0.5, 1.0));
        assert_eq!(rule.gate(&metrics_with_cv(1.5), &envelope), Gate::Off);
    }

    /// Hand-computed: current 0.4, band lo 0.61, `PER_FIX_EFFECT` 0.05 ->
    /// deficit ~0.21 -> budget floor(0.21 / 0.05) = 4 (a deficit picked
    /// comfortably clear of the exact `0.05` grid so the assertion is not
    /// sensitive to `f64` subtraction's own rounding).
    #[test]
    fn gate_below_band_computes_hand_verified_budget() {
        let rule = SentenceSplitRule::new();
        let envelope = MapEnvelope::new().with(GATED_METRIC, Envelope::new(0.61, 1.0));
        assert_eq!(
            rule.gate(&metrics_with_cv(0.4), &envelope),
            Gate::Fix {
                budget: Budget::new(4)
            }
        );
    }

    #[test]
    fn gate_with_zero_budget_is_off() {
        let rule = SentenceSplitRule::new();
        let envelope = MapEnvelope::new().with(GATED_METRIC, Envelope::new(0.6, 1.0));
        // deficit 0.6 - 0.599 = 0.001, floor(0.001 / 0.05) = 0.
        assert_eq!(rule.gate(&metrics_with_cv(0.599), &envelope), Gate::Off);
    }

    // ---------------------------------------------------------------
    // scan(): boundary selection
    // ---------------------------------------------------------------

    #[test]
    fn scan_finds_a_semicolon_boundary() {
        let source = "The quarterly review covered budget allocation across every \
                       department in critical detail; the team agreed on next steps for the year.";
        let doc = document(source);
        let envelope = MapEnvelope::new();
        let tagger = tagger();
        let ctx = RuleContext::new(&doc, &tagger, "docs", &envelope);
        let findings = SentenceSplitRule::new().scan(&ctx);
        assert_eq!(findings.len(), 1);
        assert_eq!(&source[findings[0].range.clone()], ";");
    }

    #[test]
    fn scan_finds_a_coordinator_boundary_before_a_pronoun() {
        let source = "The engineering team spent the entire quarter rebuilding the \
                       deployment pipeline from scratch, and it now handles ten times the \
                       load without breaking a sweat.";
        let doc = document(source);
        let envelope = MapEnvelope::new();
        let tagger = tagger();
        let ctx = RuleContext::new(&doc, &tagger, "docs", &envelope);
        let findings = SentenceSplitRule::new().scan(&ctx);
        assert_eq!(findings.len(), 1);
        assert_eq!(&source[findings[0].range.clone()], ", and ");
    }

    /// A `", and "` immediately before a plain common noun (continuing a
    /// flat noun-phrase list, not opening a clause) is not a candidate —
    /// even inside an otherwise over-long sentence with no other boundary,
    /// `scan` proposes nothing rather than splitting a noun-phrase list.
    #[test]
    fn scan_does_not_split_a_noun_phrase_coordination_list() {
        let source = "The comprehensive hardware kit for this particular installation \
                       includes screws, brackets, spacers, washers, and fasteners for every \
                       possible mounting scenario imaginable.";
        let doc = document(source);
        let envelope = MapEnvelope::new();
        let tagger = tagger();
        let ctx = RuleContext::new(&doc, &tagger, "docs", &envelope);
        assert!(SentenceSplitRule::new().scan(&ctx).is_empty());
    }

    /// Regression test: a semicolon that sits inside a direct quotation is
    /// never a candidate, even though the sentence as a whole is over-long
    /// and has no other boundary — splitting there would silently rewrite
    /// the exact words attributed to the quoted speaker (recapitalizing
    /// "nobody" to "Nobody" inside the quotation marks).
    #[test]
    fn scan_does_not_split_a_semicolon_inside_a_direct_quotation() {
        let source = "The manager said, \"the launch went fine; nobody on the team expected \
                       the traffic spike we saw this morning at all.\"";
        let doc = document(source);
        let envelope = MapEnvelope::new();
        let tagger = tagger();
        let ctx = RuleContext::new(&doc, &tagger, "docs", &envelope);
        assert!(SentenceSplitRule::new().scan(&ctx).is_empty());
    }

    /// Regression test: a coordinator boundary that sits inside a direct
    /// quotation is never a candidate either.
    #[test]
    fn scan_does_not_split_a_coordinator_inside_a_direct_quotation() {
        let source = "The manager said, \"the launch went fine, and nobody on the team \
                       expected the traffic spike we saw this morning at all whatsoever.\"";
        let doc = document(source);
        let envelope = MapEnvelope::new();
        let tagger = tagger();
        let ctx = RuleContext::new(&doc, &tagger, "docs", &envelope);
        assert!(SentenceSplitRule::new().scan(&ctx).is_empty());
    }

    /// Regression test for the finding that a natural three-clause serial
    /// list (`"A does X, B does Y, and C does Z."`) only ever offered the
    /// *last* `", and "` as a candidate, and splitting only there left the
    /// first two clauses joined by nothing but a bare comma — a comma
    /// splice this rule would have introduced into text that had none. Now
    /// `precedes_unresolved_comma` excludes that boundary, and there is no
    /// other candidate in this sentence, so `scan` proposes nothing rather
    /// than degrading a (grammatically valid, if stylistically dense)
    /// three-clause sentence into a splice.
    #[test]
    fn scan_does_not_split_a_three_clause_serial_list_into_a_comma_splice() {
        let source = "Configuration handles environment setup automatically for every new \
                       team member, Installation handles every dependency without manual \
                       intervention required, and Kubernetes handles the rollout to every \
                       environment automatically once approved.";
        let doc = document(source);
        let envelope = MapEnvelope::new();
        let tagger = tagger();
        let ctx = RuleContext::new(&doc, &tagger, "docs", &envelope);
        assert!(SentenceSplitRule::new().scan(&ctx).is_empty());
    }

    /// A sentence at or under the genre's over-long threshold is never a
    /// candidate, regardless of how many semicolons it contains.
    #[test]
    fn scan_ignores_a_sentence_at_or_under_the_threshold() {
        let source = "Short and fine; this stays put.";
        let doc = document(source);
        let envelope = MapEnvelope::new();
        let tagger = tagger();
        let ctx = RuleContext::new(&doc, &tagger, "docs", &envelope);
        assert!(SentenceSplitRule::new().scan(&ctx).is_empty());
    }

    /// Two over-long sentences are returned longest-first, not in source
    /// order — the shorter (but still over-long) sentence appears earlier
    /// in the source but later in `scan`'s output.
    #[test]
    fn scan_orders_findings_by_length_descending_then_position() {
        let shorter = "The rollout plan covers every region we support and every \
                        customer segment we serve across the whole organization; teams \
                        will coordinate closely.";
        let longer = "The engineering organization spent months redesigning the entire \
                       deployment pipeline from the ground up to support far higher \
                       throughput than before; the operations team documented every \
                       step of the migration along the way for future reference.";
        let source = format!("{shorter} {longer}");
        let doc = document(&source);
        let envelope = MapEnvelope::new();
        let tagger = tagger();
        let ctx = RuleContext::new(&doc, &tagger, "docs", &envelope);
        let findings = SentenceSplitRule::new().scan(&ctx);
        assert_eq!(findings.len(), 2);
        // The longer sentence's boundary sits at a later byte offset (it
        // comes second in the source), but it must be first in `scan`'s
        // output because it is the longer sentence.
        assert!(findings[0].range.start > findings[1].range.start);
    }

    // ---------------------------------------------------------------
    // fix()
    // ---------------------------------------------------------------

    #[test]
    fn fix_splits_at_a_semicolon_and_recapitalizes() {
        let source = "The quarterly review covered budget allocation across every \
                       department in critical detail; the team agreed on next steps for the year.";
        let doc = document(source);
        let envelope = MapEnvelope::new();
        let tagger = tagger();
        let ctx = RuleContext::new(&doc, &tagger, "docs", &envelope);
        let rule = SentenceSplitRule::new();
        let finding = &rule.scan(&ctx)[0];
        let mut rng = StrategyRng::from_seed(0);
        let patch = rule.fix(finding, &ctx, &mut rng).expect("expected a patch");
        assert_eq!(
            apply(source, &patch),
            "The quarterly review covered budget allocation across every department in \
             critical detail. The team agreed on next steps for the year."
        );
    }

    #[test]
    fn fix_splits_at_a_coordinator_and_recapitalizes() {
        let source = "The engineering team spent the entire quarter rebuilding the \
                       deployment pipeline from scratch, and it now handles ten times the \
                       load without breaking a sweat.";
        let doc = document(source);
        let envelope = MapEnvelope::new();
        let tagger = tagger();
        let ctx = RuleContext::new(&doc, &tagger, "docs", &envelope);
        let rule = SentenceSplitRule::new();
        let finding = &rule.scan(&ctx)[0];
        let mut rng = StrategyRng::from_seed(0);
        let patch = rule.fix(finding, &ctx, &mut rng).expect("expected a patch");
        assert_eq!(
            apply(source, &patch),
            "The engineering team spent the entire quarter rebuilding the deployment \
             pipeline from scratch. It now handles ten times the load without breaking a \
             sweat."
        );
    }

    /// Regression test: a `", but "` boundary keeps "But" as its own word
    /// instead of silently deleting the contrast it expresses.
    #[test]
    fn fix_splits_at_a_but_coordinator_and_keeps_the_connective() {
        let source = "The engineering team spent the entire quarter rebuilding the \
                       deployment pipeline from scratch, but it now handles ten times the \
                       load without breaking a sweat.";
        let doc = document(source);
        let envelope = MapEnvelope::new();
        let tagger = tagger();
        let ctx = RuleContext::new(&doc, &tagger, "docs", &envelope);
        let rule = SentenceSplitRule::new();
        let finding = &rule.scan(&ctx)[0];
        let mut rng = StrategyRng::from_seed(0);
        let patch = rule.fix(finding, &ctx, &mut rng).expect("expected a patch");
        assert_eq!(
            apply(source, &patch),
            "The engineering team spent the entire quarter rebuilding the deployment \
             pipeline from scratch. But it now handles ten times the load without breaking a \
             sweat."
        );
    }

    /// Regression test: a `", so "` boundary keeps "So" as its own word
    /// instead of silently deleting the causal link it expresses.
    #[test]
    fn fix_splits_at_a_so_coordinator_and_keeps_the_connective() {
        let source = "The deployment pipeline failed its integrity check twice in the same \
                       afternoon, so the on-call engineer rolled the whole release back \
                       immediately.";
        let doc = document(source);
        let envelope = MapEnvelope::new();
        let tagger = tagger();
        let ctx = RuleContext::new(&doc, &tagger, "docs", &envelope);
        let rule = SentenceSplitRule::new();
        let finding = &rule.scan(&ctx)[0];
        let mut rng = StrategyRng::from_seed(0);
        let patch = rule.fix(finding, &ctx, &mut rng).expect("expected a patch");
        assert_eq!(
            apply(source, &patch),
            "The deployment pipeline failed its integrity check twice in the same afternoon. \
             So the on-call engineer rolled the whole release back immediately."
        );
    }

    // ---------------------------------------------------------------
    // Idempotence
    // ---------------------------------------------------------------

    /// The split replacement (`". "` plus at most one recapitalized
    /// letter) can never itself contain a semicolon or a `COORDINATORS`
    /// phrase — the "no substitution-table RHS is also an LHS" closure
    /// this workspace requires of every table like this one, checked
    /// directly rather than merely argued about in the module docs.
    #[test]
    fn split_replacement_never_reintroduces_a_boundary_marker() {
        for (source, marker_start, marker_len, connective) in [
            ("before; after works.", 6usize, 1usize, None),
            ("clause, and it works.", 6usize, ", and ".len(), None),
            ("clause, but it fails.", 6usize, ", but ".len(), Some("But")),
            ("clause, so it ships.", 6usize, ", so ".len(), Some("So")),
        ] {
            let patch = build_split_patch(source, marker_start, marker_len, connective);
            assert!(!patch.replacement.contains(';'));
            for coordinator in COORDINATORS {
                assert!(!patch.replacement.contains(coordinator));
            }
        }
    }

    #[test]
    fn fixing_the_same_source_twice_is_byte_identical() {
        let source = "The quarterly review covered budget allocation across every \
                       department in critical detail; the team agreed on next steps for the year.";
        let run = || {
            let doc = document(source);
            let envelope = MapEnvelope::new();
            let tagger = tagger();
            let ctx = RuleContext::new(&doc, &tagger, "docs", &envelope);
            let rule = SentenceSplitRule::new();
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
