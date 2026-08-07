//! The register pass.
//!
//! After the five-operation passes converge, home four register features — nominalization,
//! agentless passive, em dash, semicolon — toward their human rate band by selecting and
//! applying [`friction_register::transduce`] candidates. The first two are Biber constructions;
//! the last two aren't Biber's, but the same band-and-transducer machinery applies to them
//! unchanged (see `register-en-v1.toml`'s `[features.em_dash]`/`[features.semicolon]`).
//!
//! # Termination is the band, not the median
//!
//! A feature is done once its rate sits inside `[low, high]`, never at the band's `median`.
//! Human writing is a distribution, not a point: converging every document onto one centroid
//! would be a stronger tell than the spread it started with, so [`run_register`] stops there
//! and never walks further.
//!
//! # Selection: greedy and re-evaluated
//!
//! Candidates are scored by how far their delta moves the feature's *rate*: count over prose
//! word total, not raw count. The best is applied, both totals updated from that pick's
//! measured effect, and the pool re-scored: a transducer's `delta` gives the count change
//! directly, but the word-count change is only known by measuring it, shifting the denominator
//! every later candidate is scored against.
//!
//! # Conflict is dependency-scope, not just span
//!
//! Two candidates conflict when their ranges overlap, when one's span contains the governor of
//! a token the other depends on, or when both trace to the same governing finite verb. Span
//! disjointness alone isn't enough: two edits on non-overlapping spans of a coordinated clause
//! can still mismatch voice across it, one conjunct passivized and the other not, sharing a
//! subject that no longer exists on one side. [`conflicts`] is the general, transducer-agnostic
//! guard against that; it doesn't rely on any single transducer's own narrower licensing to
//! rule it out.
//!
//! # Closure is structural
//!
//! Every candidate passes [`closure_violation`] before entering the pool: no path from
//! candidate to patch skips it. A content word must already appear in the matched span or be
//! reachable through the transducer's own inflection table; a function word must be one of the
//! four permitted (see [`friction_register::transduce::PERMITTED_FUNCTION_WORDS`]). A candidate
//! that fails is dropped and held, never applied.

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

use friction_core::span::ranges_overlap;
use friction_core::{Document, Finding, Patch, RuleId, Tier};
use friction_match::token::{AnalysisTokenKind, prose_scope, tokenize_str};
use friction_nlp::{DepParser, FINITE_VERB_TAGS, Segmenter, SentenceParse, TaggedToken, Tagger};
use friction_packs::{RegisterBand, RegisterPack};
use friction_register::features::{
    RegisterCounts, comma_and_offsets, contrast_closers, em_dashes, nominalizations,
    past_progressives, semicolons,
};
use friction_register::transduce::{
    self, CandidateKind, PERMITTED_FUNCTION_WORDS, past, past_participle, third_sg,
};
use rayon::prelude::*;

use crate::document::{apply, ends_with_sentence_terminal_punctuation, resolve};
use crate::error::EditError;
use crate::gates::in_quoted_span;

const RULE_PASSIVIZE: RuleId = RuleId::new("register.passivize");
const RULE_UNPACK: RuleId = RuleId::new("register.unpack");
const RULE_EM_DASH: RuleId = RuleId::new("register.em_dash");
const RULE_SEMICOLON: RuleId = RuleId::new("register.semicolon");
const RULE_COMMA_AND: RuleId = RuleId::new("register.comma_and");
const RULE_PAST_PROGRESSIVE: RuleId = RuleId::new("register.past_progressive");
/// `contrast_closer` is detect-only -- no transducer, no [`CandidateKind`]
/// (see the held-finding block at the end of [`run_register`]) -- so this
/// rule id is never returned by [`rule_for`], only used directly there.
const RULE_CONTRAST: RuleId = RuleId::new("register.contrast_closer");

const fn rule_for(kind: CandidateKind) -> RuleId {
    match kind {
        CandidateKind::ActivizeToPassive => RULE_PASSIVIZE,
        CandidateKind::NominalizationUnpack => RULE_UNPACK,
        CandidateKind::EmDash => RULE_EM_DASH,
        CandidateKind::Semicolon => RULE_SEMICOLON,
        CandidateKind::CommaAnd => RULE_COMMA_AND,
        CandidateKind::PastProgressive => RULE_PAST_PROGRESSIVE,
    }
}

/// One in-scope sentence's text range, tags, and dependency parse:
/// computed once and reused by every stage below (counting, candidate
/// generation, closure checking, conflict checking) so none re-tag or
/// re-parse.
struct SentenceCtx {
    /// This sentence's byte range in the document's original source.
    range: Range<usize>,
    /// Tagged tokens, offset 0 (local to this sentence's own text).
    tokens: Vec<TaggedToken>,
    /// This sentence's dependency parse, indexed the same way as
    /// `tokens`.
    parse: SentenceParse,
    /// This "sentence" opens a prose run split off mid-sentence by an
    /// excluded construct (same-block predecessor run with no
    /// sentence-terminal punctuation — the `crate::document` rule the
    /// four operations use for recapitalization). Its text is the tail
    /// of a clause, not a clause: a paired em-dash aside cut at an
    /// inline-code span leaves exactly one dash visible here, and
    /// rewriting it splits the pair ("run — [`code`]: rather than", a
    /// real defect measured on this repository's own comments).
    continues_previous: bool,
}

/// One sentence's byte range and part-of-speech tags, exposed via
/// [`ReusableScan`] for a caller that needs the same tags [`run_register`]
/// already computed.
///
/// `friction fix`'s paraphrase scan (`friction-cli`'s `fix` module) would
/// otherwise re-parse, re-scope, and re-tag the exact same bytes register
/// just finished with. `tokens` keeps [`run_register`]'s own LOCAL-offset
/// tagging convention
/// (`tagger.tag(text, 0)`, relative to `range`, not absolute into the
/// document) — this module's own candidate generation needs sentence-
/// local coordinates throughout (`friction_register::transduce`'s
/// functions all index against a per-sentence text slice), so keeping it
/// here means zero blast radius to that machinery. A caller needing
/// absolute offsets (the jargon channel's own convention — see
/// `friction_match::tagging`'s module docs) shifts each token's range by
/// `+ range.start`: an `O(tokens)` map, far cheaper than re-tagging.
#[derive(Debug, Clone)]
pub struct RegisterSentenceTags {
    /// This sentence's byte range, absolute into the document source.
    pub range: Range<usize>,
    /// This sentence's tagged tokens, LOCAL offset (relative to `range`).
    pub tokens: Vec<TaggedToken>,
}

/// Everything [`run_register`] already built for the text it received,
/// reusable by a caller scanning that SAME text afterward.
///
/// See [`RegisterSentenceTags`]'s own docs for why. Valid ONLY together
/// with a [`crate::document::PassReport`] whose
/// `patches_applied` is `0`: any accepted patch shifts every later byte
/// offset in the pass's OUTPUT text out from under this `document`/these
/// `sentences`, which describe the text register RECEIVED, not (once a
/// patch shifts it) the different text it returned. [`run_register`]
/// enforces this itself — it only ever returns `Some` alongside a
/// `patches_applied == 0` report — so a caller need only check for
/// `Some`, never re-derive the condition.
#[derive(Debug, Clone)]
pub struct ReusableScan {
    /// The parsed document register received (and, since it applied zero
    /// patches, the exact document it returned too).
    pub document: Document,
    /// Every in-scope sentence's range and local-offset tags.
    pub sentences: Vec<RegisterSentenceTags>,
}

impl From<SentenceCtx> for RegisterSentenceTags {
    fn from(ctx: SentenceCtx) -> Self {
        Self {
            range: ctx.range,
            tokens: ctx.tokens,
        }
    }
}

/// A transducer candidate in document-absolute coordinates, carrying
/// the extra bookkeeping the selection loop needs: which sentence it
/// came from (for conflict checking against another candidate from the
/// same sentence) and how many prose words applying it would add or
/// remove (for re-scoring the rate after each pick).
struct PositionedCandidate {
    sentence_index: usize,
    /// Byte range in that sentence's own local text (matches
    /// `SentenceCtx::tokens`' space).
    local_range: Range<usize>,
    /// The same range, shifted into the document's original source.
    doc_range: Range<usize>,
    replacement: Box<str>,
    kind: CandidateKind,
    delta: BTreeMap<&'static str, i32>,
    /// `word_count(replacement) - word_count(original matched span)`.
    word_delta: i64,
}

/// Which way a feature's rate needs to move to enter its band.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Direction {
    /// Below `low`: only [`CandidateKind::ActivizeToPassive`] moves
    /// this way (agentless passive).
    Increase,
    /// Above `high`: [`CandidateKind::NominalizationUnpack`]
    /// (nominalization), [`CandidateKind::EmDash`] (em dash),
    /// [`CandidateKind::Semicolon`] (semicolon), [`CandidateKind::CommaAnd`]
    /// (comma-and), and [`CandidateKind::PastProgressive`]
    /// (past-progressive) all move this way. `contrast_closer` also moves
    /// this way but has no `CandidateKind` at all -- it is detect-only
    /// (see the held-finding block at the end of [`run_register`]).
    Decrease,
}

/// Tags and parses every non-empty sentence in `units`, silently
/// dropping any sentence a per-sentence parse failure or empty trimmed
/// text excludes.
///
/// Every sentence's tag+parse is independent of every other's — neither
/// [`Tagger::tag`] nor [`DepParser::parse`] reads or writes anything but
/// its own `text` argument (see both traits' `Send + Sync` docs) — so
/// this runs over a flattened, order-preserving list of sentence ranges
/// with a rayon `par_iter`. `filter_map` over an indexed parallel
/// iterator, collected straight into a `Vec`, keeps the surviving
/// sentences in exactly the document order the sequential version
/// produced: rayon's collect for an `IndexedParallelIterator` reassembles
/// results by source position regardless of which thread computed which
/// element or how many threads ran, which is the byte-determinism this
/// module's callers (and `tests::rayon_thread_count_determinism`) depend
/// on.
fn build_sentence_contexts(
    source: &str,
    units: &[friction_match::token::ScopedUnit<'_>],
    tagger: &dyn Tagger,
    parser: &dyn DepParser,
) -> Vec<SentenceCtx> {
    let ranges: Vec<(Range<usize>, bool)> = units
        .iter()
        .enumerate()
        .flat_map(|(unit_index, unit)| {
            // A unit sharing its predecessor's block index is a later
            // run of the same prose session, split off by an excluded
            // construct (`friction_parse::extract`'s guarantee). Its
            // first "sentence" is a genuine sentence only if that
            // predecessor ended with sentence-terminal punctuation —
            // the same rule `crate::document` applies for
            // recapitalization. Same-block runs share a block kind, so
            // scope filtering never separates them and the scoped
            // predecessor is the true one.
            let continues_previous =
                unit_index
                    .checked_sub(1)
                    .map(|i| &units[i])
                    .is_some_and(|prev| {
                        prev.unit.block == unit.unit.block
                            && !ends_with_sentence_terminal_punctuation(prev.text)
                    });
            unit.sentences
                .iter()
                .enumerate()
                .map(move |(sentence_index, range)| {
                    (range.clone(), sentence_index == 0 && continues_previous)
                })
        })
        .collect();
    ranges
        .par_iter()
        .filter_map(|(range, continues_previous)| {
            build_sentence_ctx(source, range, tagger, parser, *continues_previous)
        })
        .collect()
}

/// Tags and parses one sentence, `None` if `range` fails to slice `source`
/// (never expected — `range` always comes from this same `source`'s own
/// segmentation), its trimmed text is empty, or the parse fails (allowed
/// by [`DepParser::parse`]'s contract, never expected from the shipped
/// perceptron parser): the four operations already ran and aren't rolled
/// back for a register-only problem, so a dropped sentence is silently
/// excluded rather than failing the whole document.
fn build_sentence_ctx(
    source: &str,
    range: &Range<usize>,
    tagger: &dyn Tagger,
    parser: &dyn DepParser,
    continues_previous: bool,
) -> Option<SentenceCtx> {
    let text = source.get(range.clone())?;
    if text.trim().is_empty() {
        return None;
    }
    let tokens = tagger.tag(text, 0);
    let parse = parser.parse(text, &tokens).ok()?;
    Some(SentenceCtx {
        range: range.clone(),
        tokens,
        parse,
        continues_previous,
    })
}

/// The document-absolute byte range of every remaining instance of a
/// Decrease feature, in document order.
///
/// Calls the same detector functions [`RegisterCounts::count`] takes its
/// lengths from, so the listed positions and the counted rate can't
/// quietly disagree. Instances overlapping an `accepted` rewrite are
/// dropped — those bytes are about to change, so pointing a finding at
/// them would misplace it (in the zero-patch convergence pass whose
/// findings the CLI actually reports, `accepted` is empty and this
/// filter is a no-op).
fn remainder_instance_ranges(
    sentences: &[SentenceCtx],
    source: &str,
    kind: CandidateKind,
    accepted: &[PositionedCandidate],
) -> Vec<Range<usize>> {
    let mut out: Vec<Range<usize>> = Vec::new();
    for ctx in sentences {
        let text = &source[ctx.range.clone()];
        match kind {
            CandidateKind::NominalizationUnpack => {
                for index in nominalizations(text, &ctx.tokens, &ctx.parse) {
                    let local = &ctx.tokens[index].token.range;
                    out.push(ctx.range.start + local.start..ctx.range.start + local.end);
                }
            }
            CandidateKind::EmDash => {
                for byte in em_dashes(text) {
                    let start = ctx.range.start + byte;
                    out.push(start..start + '\u{2014}'.len_utf8());
                }
            }
            CandidateKind::Semicolon => {
                for byte in semicolons(text) {
                    let start = ctx.range.start + byte;
                    out.push(start..start + 1);
                }
            }
            CandidateKind::CommaAnd => {
                for byte in comma_and_offsets(text) {
                    let start = ctx.range.start + byte;
                    out.push(start..start + ", and ".len());
                }
            }
            CandidateKind::PastProgressive => {
                for index in past_progressives(text, &ctx.tokens) {
                    let local = &ctx.tokens[index].token.range;
                    out.push(ctx.range.start + local.start..ctx.range.start + local.end);
                }
            }
            // An Increase feature never leaves "instances to remove", so
            // no remainder findings exist for it.
            CandidateKind::ActivizeToPassive => {}
        }
    }
    out.retain(|range| !accepted.iter().any(|c| ranges_overlap(&c.doc_range, range)));
    out
}

/// Emits one Suggest finding per remaining instance of a Decrease
/// feature that is still above its band after selection.
///
/// A Decrease feature can run out of licensed candidates while still
/// above its band — every remaining instance was a hold-don't-guess
/// decline at candidate-generation time. Each finding carries the
/// instance's own byte range, so the report says exactly which tokens
/// need a human hand instead of ending on an unlocatable count. An
/// instance whose candidate was gate-held already has a finding of its
/// own saying exactly why (a quotation guard, a closure failure), so it
/// is skipped here — a second, generic line on the same bytes would say
/// less than the one already there. For a nominalization whose verb the
/// derivational lexicon knows, the finding names that verb: the one
/// rewrite direction that can be stated without inventing a word.
#[allow(clippy::too_many_arguments)] // private emission helper for one call site
fn push_remainder_findings(
    held: &mut Vec<Finding>,
    sentences: &[SentenceCtx],
    source: &str,
    kind: CandidateKind,
    accepted: &[PositionedCandidate],
    feature: &str,
    feature_rate: f64,
    band_high: f64,
) {
    let explained: Vec<Range<usize>> = held
        .iter()
        .filter(|f| f.rule == rule_for(kind))
        .map(|f| f.range.clone())
        .collect();
    for range in remainder_instance_ranges(sentences, source, kind, accepted)
        .into_iter()
        .filter(|r| !explained.iter().any(|e| ranges_overlap(e, r)))
    {
        let (subject, verb_hint) = match kind {
            CandidateKind::NominalizationUnpack => {
                let noun = &source[range.clone()];
                let hint = friction_nlp::lvc::DERIVATIONAL_LEXICON
                    .get(noun.to_lowercase().as_str())
                    .map(|verb| format!(" (its verb is \"{verb}\")"))
                    .unwrap_or_default();
                (format!("\"{noun}\""), hint)
            }
            _ => ("this instance".to_string(), String::new()),
        };
        held.push(Finding {
            rule: rule_for(kind),
            range,
            message: format!(
                "{feature}: {subject} has no licensed rewrite and the document is still \
                 above the human band ({feature_rate:.2} > {band_high:.2} per 1000 words) \
                 — needs a human hand{verb_hint}"
            ),
            tier: Tier::Suggest,
        });
    }
}

/// Emits one Suggest finding per surviving `contrast_closer` instance
/// once the document's rate clears the band -- the see-saw tell's own
/// counterpart to [`push_remainder_findings`], called separately because
/// `contrast_closer` has no [`CandidateKind`] at all: no transducer
/// exists for it (there is no single-word substitute that states a
/// "X rather than Y" / "X, not Y" contrast without picking a side), so
/// every instance is a remainder from the moment it's counted, never a
/// pool candidate `select_and_apply` might have applied first.
///
/// Arms on the same Wilson-bound evidence bar every Decrease feature
/// uses ([`wilson_lower`] against `band.high`). `count`/`total_words` are
/// the FINAL values [`run_register`] has after every armed feature's own
/// selection loop: those loops may have shifted the word-count
/// denominator, but never `contrast_closer`'s own count, since nothing
/// upstream ever edits its instances.
fn push_contrast_closer_findings(
    held: &mut Vec<Finding>,
    sentences: &[SentenceCtx],
    source: &str,
    accepted: &[PositionedCandidate],
    count: i64,
    total_words: i64,
    band: RegisterBand,
) {
    if wilson_lower(count, total_words) <= band.high {
        return;
    }
    let feature_rate = rate(count, total_words);
    for ctx in sentences {
        let text = &source[ctx.range.clone()];
        for local in contrast_closers(text) {
            let range = ctx.range.start + local.start..ctx.range.start + local.end;
            if accepted
                .iter()
                .any(|c| ranges_overlap(&c.doc_range, &range))
            {
                continue;
            }
            held.push(Finding {
                rule: RULE_CONTRAST,
                range,
                message: format!(
                    "contrast_closer: a contrast closer (\"X rather than Y\" / \"X, not Y\") \
                     and the document is still above the human band ({feature_rate:.2} > \
                     {:.2} per 1000 words) -- no licensed rewrite: state one side; needs a \
                     human hand",
                    band.high
                ),
                tier: Tier::Suggest,
            });
        }
    }
}

/// The document-wide totals [`count_features`] returns: total prose
/// words, plus one count per register feature.
///
/// A named struct, not a positional tuple: the tuple this replaced grew
/// from five elements to eight across three features landing in one
/// change, and a positional `(i64, i64, i64, i64, i64, i64, i64, i64)`
/// at that width is a transposition bug waiting to happen at every call
/// site. `contrast_closer` gets a field like every other feature even
/// though it is detect-only downstream (see [`run_register`]'s final
/// held-finding block) -- the count itself is computed the same way
/// regardless of whether a transducer ever consumes it.
#[derive(Debug, Clone, Copy, Default)]
struct FeatureCounts {
    total_words: i64,
    nominalization: i64,
    agentless_passive: i64,
    em_dash: i64,
    semicolon: i64,
    contrast_closer: i64,
    comma_and: i64,
    past_progressive: i64,
}

/// The document-wide [`FeatureCounts`] over `sentences`.
fn count_features(sentences: &[SentenceCtx], source: &str) -> FeatureCounts {
    let mut counts = FeatureCounts::default();
    for ctx in sentences {
        let text = &source[ctx.range.clone()];
        let register_counts = RegisterCounts::count(text, &ctx.tokens, &ctx.parse);
        counts.nominalization += i64::try_from(register_counts.nominalization).unwrap_or(i64::MAX);
        counts.agentless_passive +=
            i64::try_from(register_counts.agentless_passive).unwrap_or(i64::MAX);
        counts.em_dash += i64::try_from(register_counts.em_dashes).unwrap_or(i64::MAX);
        counts.semicolon += i64::try_from(register_counts.semicolons).unwrap_or(i64::MAX);
        counts.contrast_closer += i64::try_from(contrast_closers(text).len()).unwrap_or(i64::MAX);
        counts.comma_and += i64::try_from(comma_and_offsets(text).len()).unwrap_or(i64::MAX);
        counts.past_progressive +=
            i64::try_from(past_progressives(text, &ctx.tokens).len()).unwrap_or(i64::MAX);
        counts.total_words += word_count(text);
    }
    counts
}

/// The document-wide em-dash rate (per 1000 prose words), computed
/// through the exact sentence-context/counting path [`run_register`]
/// itself uses.
///
/// Exposed for band-measurement tooling (`corpus-tool register-bands`):
/// remeasuring the corpus this way, rather than through a separate
/// counting path, is the contract that keeps a remeasured band honest
/// about what the runtime pass actually counts.
///
/// # Errors
/// Returns [`EditError`] if `source` fails to parse or segment.
pub fn measure_em_dash_rate(
    source: &str,
    tagger: &dyn Tagger,
    parser: &dyn DepParser,
    segmenter: &dyn Segmenter,
) -> Result<f64, EditError> {
    let document = friction_parse::parse(source)?;
    let units = prose_scope(&document, segmenter);
    let sentences = build_sentence_contexts(source, &units, tagger, parser);
    let counts = count_features(&sentences, source);
    Ok(rate(counts.em_dash, counts.total_words))
}

/// The document-wide semicolon rate (per 1000 prose words), computed
/// through the same path [`measure_em_dash_rate`] does, for the same
/// band-measurement contract.
///
/// # Errors
/// Returns [`EditError`] if `source` fails to parse or segment.
pub fn measure_semicolon_rate(
    source: &str,
    tagger: &dyn Tagger,
    parser: &dyn DepParser,
    segmenter: &dyn Segmenter,
) -> Result<f64, EditError> {
    let document = friction_parse::parse(source)?;
    let units = prose_scope(&document, segmenter);
    let sentences = build_sentence_contexts(source, &units, tagger, parser);
    let counts = count_features(&sentences, source);
    Ok(rate(counts.semicolon, counts.total_words))
}

/// The document-wide contrast-closer rate (per 1000 prose words), computed
/// through the same path [`measure_em_dash_rate`] does, for the same
/// band-measurement contract.
///
/// # Errors
/// Returns [`EditError`] if `source` fails to parse or segment.
pub fn measure_contrast_closer_rate(
    source: &str,
    tagger: &dyn Tagger,
    parser: &dyn DepParser,
    segmenter: &dyn Segmenter,
) -> Result<f64, EditError> {
    let document = friction_parse::parse(source)?;
    let units = prose_scope(&document, segmenter);
    let sentences = build_sentence_contexts(source, &units, tagger, parser);
    let counts = count_features(&sentences, source);
    Ok(rate(counts.contrast_closer, counts.total_words))
}

/// The document-wide comma-and rate (per 1000 prose words), computed
/// through the same path [`measure_em_dash_rate`] does, for the same
/// band-measurement contract.
///
/// # Errors
/// Returns [`EditError`] if `source` fails to parse or segment.
pub fn measure_comma_and_rate(
    source: &str,
    tagger: &dyn Tagger,
    parser: &dyn DepParser,
    segmenter: &dyn Segmenter,
) -> Result<f64, EditError> {
    let document = friction_parse::parse(source)?;
    let units = prose_scope(&document, segmenter);
    let sentences = build_sentence_contexts(source, &units, tagger, parser);
    let counts = count_features(&sentences, source);
    Ok(rate(counts.comma_and, counts.total_words))
}

/// The document-wide past-progressive rate (per 1000 prose words),
/// computed through the same path [`measure_em_dash_rate`] does, for the
/// same band-measurement contract.
///
/// # Errors
/// Returns [`EditError`] if `source` fails to parse or segment.
pub fn measure_past_progressive_rate(
    source: &str,
    tagger: &dyn Tagger,
    parser: &dyn DepParser,
    segmenter: &dyn Segmenter,
) -> Result<f64, EditError> {
    let document = friction_parse::parse(source)?;
    let units = prose_scope(&document, segmenter);
    let sentences = build_sentence_contexts(source, &units, tagger, parser);
    let counts = count_features(&sentences, source);
    Ok(rate(counts.past_progressive, counts.total_words))
}

/// Runs the register pass once over `source`.
///
/// `source` is expected to already be the five-operation pipeline's converged output. Register
/// never interleaves with those passes.
///
/// The third element is [`ReusableScan`]'s reuse-or-rebuild handoff:
/// `Some` exactly when the returned [`crate::document::PassReport`]'s
/// `patches_applied` is `0` (this call's own document/sentences are then
/// still valid for the text it returned), `None` otherwise.
///
/// # Errors
/// Returns [`EditError`] if `source` fails to parse or segment.
pub fn run_register(
    source: &str,
    syntax: friction_parse::Syntax,
    register_pack: &RegisterPack,
    tagger: &dyn Tagger,
    parser: &dyn DepParser,
    segmenter: &dyn Segmenter,
) -> Result<(String, crate::document::PassReport, Option<ReusableScan>), EditError> {
    let document = friction_parse::parse_with(source, syntax)?;
    let units = prose_scope(&document, segmenter);
    let sentences = build_sentence_contexts(source, &units, tagger, parser);
    let counts = count_features(&sentences, source);
    let mut total_words = counts.total_words;

    let mut held: Vec<Finding> = Vec::new();
    if total_words == 0 {
        let reusable = ReusableScan {
            document,
            sentences: sentences
                .into_iter()
                .map(RegisterSentenceTags::from)
                .collect(),
        };
        return Ok((source.to_string(), empty_pass(), Some(reusable)));
    }

    let features = feature_plan(register_pack, counts);

    let needed: Vec<CandidateKind> = features
        .iter()
        .filter(|(_, fix, ..)| fix.is_some())
        .map(|&(.., kind)| kind)
        .collect();
    let mut pool: Vec<PositionedCandidate> = Vec::new();
    collect_needed_candidates(&sentences, source, &needed, &mut pool, &mut held);

    let mut accepted: Vec<PositionedCandidate> = Vec::new();

    for (feature, fix, direction, count, kind) in features {
        if let Some(band) = fix {
            let (final_count, new_words) = select_and_apply(
                &mut pool,
                &sentences,
                feature,
                band,
                direction,
                count,
                total_words,
                &mut accepted,
            );
            total_words = new_words;
            if direction == Direction::Decrease
                && wilson_lower(final_count, total_words) > band.high
            {
                push_remainder_findings(
                    &mut held,
                    &sentences,
                    source,
                    kind,
                    &accepted,
                    feature,
                    rate(final_count, total_words),
                    band.high,
                );
            }
        }
    }

    // `contrast_closer` is DETECT-ONLY: no transducer, no `CandidateKind`
    // exists for it, so nothing here was ever a candidate for
    // `select_and_apply` -- see `push_contrast_closer_findings`'s own
    // docs for why and how it arms.
    if let Some(band) = register_pack.band("contrast_closer") {
        push_contrast_closer_findings(
            &mut held,
            &sentences,
            source,
            &accepted,
            counts.contrast_closer,
            total_words,
            band,
        );
    }

    let patches: Vec<Patch> = accepted
        .iter()
        .map(|c| {
            Patch::new(
                c.doc_range.clone(),
                c.replacement.to_string(),
                rule_for(c.kind),
                Tier::Fix,
            )
        })
        .collect();
    let (resolved, dropped) = resolve(source, patches);
    let patches_applied = resolved.len();
    let next = apply(source, &resolved);

    // Only valid to hand back when nothing shifted the byte layout this
    // `document`/these `sentences` were built against — see
    // `ReusableScan`'s own docs.
    let reusable = (patches_applied == 0).then(|| ReusableScan {
        document,
        sentences: sentences
            .into_iter()
            .map(RegisterSentenceTags::from)
            .collect(),
    });

    Ok((
        next,
        crate::document::PassReport {
            patches_applied,
            patches_dropped: dropped,
            applied_patches: resolved,
            held,
        },
        reusable,
    ))
}

fn empty_pass() -> crate::document::PassReport {
    crate::document::PassReport::default()
}

/// One row per register feature, in application order: name, the band
/// when (and only when) the document's rate sits outside it, the
/// direction the rate must move, the starting count, and the candidate
/// kind that moves it.
///
/// `Some(band)` *is* "this feature needs work", so the value that
/// decided that also carries the band a later step needs. A separate
/// `bool` alongside the `Option` would leave two values that must
/// agree, and reading the band back out would mean asserting they do.
///
/// `contrast_closer` has no row here despite being a [`FeatureCounts`]
/// field: it has no `CandidateKind`, so it never enters
/// [`select_and_apply`]'s pool -- see the held-finding block at the end
/// of [`run_register`] instead.
fn feature_plan(
    register_pack: &RegisterPack,
    counts: FeatureCounts,
) -> [(
    &'static str,
    Option<RegisterBand>,
    Direction,
    i64,
    CandidateKind,
); 6] {
    let FeatureCounts {
        total_words,
        nominalization,
        agentless_passive,
        em_dash,
        semicolon,
        comma_and,
        past_progressive,
        contrast_closer: _,
    } = counts;
    [
        // Arming compares a Wilson confidence bound against the band, not
        // the raw point estimate — see [`WILSON_Z`]'s docs. A Decrease
        // feature arms only when even the LOWER bound clears `band.high`;
        // the Increase feature arms only when even the UPPER bound sits
        // below `band.low`. Either way, the document must be confidently
        // outside the band, not just measured outside it on a sample too
        // small to know.
        (
            "agentless_passive",
            register_pack
                .band("agentless_passive")
                .filter(|band| wilson_upper(agentless_passive, total_words) < band.low),
            Direction::Increase,
            agentless_passive,
            CandidateKind::ActivizeToPassive,
        ),
        (
            "nominalization",
            register_pack
                .band("nominalization")
                .filter(|band| wilson_lower(nominalization, total_words) > band.high),
            Direction::Decrease,
            nominalization,
            CandidateKind::NominalizationUnpack,
        ),
        (
            "em_dash",
            register_pack
                .band("em_dash")
                .filter(|band| wilson_lower(em_dash, total_words) > band.high),
            Direction::Decrease,
            em_dash,
            CandidateKind::EmDash,
        ),
        (
            "semicolon",
            register_pack
                .band("semicolon")
                .filter(|band| wilson_lower(semicolon, total_words) > band.high),
            Direction::Decrease,
            semicolon,
            CandidateKind::Semicolon,
        ),
        (
            "comma_and",
            register_pack
                .band("comma_and")
                .filter(|band| wilson_lower(comma_and, total_words) > band.high),
            Direction::Decrease,
            comma_and,
            CandidateKind::CommaAnd,
        ),
        (
            "past_progressive",
            register_pack
                .band("past_progressive")
                .filter(|band| wilson_lower(past_progressive, total_words) > band.high),
            Direction::Decrease,
            past_progressive,
            CandidateKind::PastProgressive,
        ),
    ]
}

/// Runs [`collect_candidates`] for exactly the features that need fixing
/// -- split out of [`run_register`] itself only to keep that function's
/// own line count down; `needed` is the kinds whose feature sits outside
/// its band, in application order. This helper never needs the bands
/// themselves, only which kinds to gather.
fn collect_needed_candidates(
    sentences: &[SentenceCtx],
    source: &str,
    needed: &[CandidateKind],
    pool: &mut Vec<PositionedCandidate>,
    held: &mut Vec<Finding>,
) {
    for &kind in needed {
        collect_candidates(sentences, source, kind, pool, held);
    }
}

/// Word-token count of `text` (same convention as
/// [`crate::document::prose_word_count`]).
fn word_count(text: &str) -> i64 {
    let count = tokenize_str(text, 0)
        .iter()
        .filter(|t| t.kind == AnalysisTokenKind::Word)
        .count();
    i64::try_from(count).unwrap_or(i64::MAX)
}

/// `count` per 1000 `words`; `0.0` for a zero total (never hit in
/// practice — [`run_register`] returns early when `total_words == 0`).
fn rate(count: i64, words: i64) -> f64 {
    if words <= 0 {
        return 0.0;
    }
    #[allow(clippy::cast_precision_loss)]
    let (count, words) = (count as f64, words as f64);
    count / words * 1000.0
}

/// The z-score both Wilson bounds are computed at: 1.96, the
/// conventional 95% figure.
///
/// The point of the bounds is evidence-proportional arming: a 58-word
/// comment with 4 nominalizations has a point rate of 69/1000, well
/// above the 50.56 band, but its lower bound is ~27/1000 — the sample
/// simply cannot support the claim, so the feature stays quiet. The
/// same density sustained over 700 words clears the bound and arms.
/// This replaces any fixed word-count floor: a short text is never
/// refused, it is held to a higher evidentiary bar that decays smoothly
/// toward the band as length grows (and a zero band still arms on a
/// single instance at any length — the lower bound of 1-in-anything is
/// above zero).
const WILSON_Z: f64 = 1.96;

/// The Wilson score interval's lower bound on a feature's rate, per
/// 1000 prose words — what a Decrease feature's arming compares against
/// `band.high` instead of the raw point estimate.
///
/// Closed-form and deterministic: IEEE-754 `sqrt` is exactly rounded,
/// so the same `(count, words)` produces the same bits on every
/// platform — the byte-determinism bar every path in this crate meets.
fn wilson_lower(count: i64, words: i64) -> f64 {
    wilson_bounds(count, words).0
}

/// The Wilson score interval's upper bound on a feature's rate, per
/// 1000 prose words — the mirrored test for the one Increase feature:
/// agentless passive arms only when even the upper bound sits below
/// `band.low`, i.e. the document is confidently *under*-using the
/// construction rather than just short.
fn wilson_upper(count: i64, words: i64) -> f64 {
    wilson_bounds(count, words).1
}

/// Both Wilson bounds, as rates per 1000 prose words.
fn wilson_bounds(count: i64, words: i64) -> (f64, f64) {
    if words <= 0 {
        return (0.0, 0.0);
    }
    #[allow(clippy::cast_precision_loss)]
    let (k, n) = (count as f64, words as f64);
    let p = k / n;
    let z2 = WILSON_Z * WILSON_Z;
    let denominator = 1.0 + z2 / n;
    let centre = p + z2 / (2.0 * n);
    let spread = WILSON_Z * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt();
    let lower = ((centre - spread) / denominator).max(0.0);
    let upper = ((centre + spread) / denominator).min(1.0);
    (lower * 1000.0, upper * 1000.0)
}

/// Generates every `kind` candidate over every sentence, gates each one
/// (quoted-span, then closure), and pushes the survivors into `pool`.
fn collect_candidates(
    sentences: &[SentenceCtx],
    source: &str,
    kind: CandidateKind,
    pool: &mut Vec<PositionedCandidate>,
    held: &mut Vec<Finding>,
) {
    for (sentence_index, ctx) in sentences.iter().enumerate() {
        let text = &source[ctx.range.clone()];
        // A range the segmenter cut short at an excluded construct
        // (inline code, a link) rather than real sentence-terminal
        // punctuation is a fragment, not a clause — seen in a corpus
        // doc where a Sphinx `:meth:`code`` cross-reference split a
        // sentence right before its own inline-code span, leaving a
        // dangling `"...of :meth:"` fragment whose parser-assigned
        // `pobj` was the reference-role name itself, producing
        // grammatically broken output when rewritten. The four
        // operations tolerate this fragment shape for smaller edits;
        // register's rewrites are clause-sized, so only a complete
        // sentence is safe input.
        if !ends_with_sentence_terminal_punctuation(text) {
            continue;
        }
        // The symmetric guard at the start: a "sentence" opening a run
        // that was split off mid-clause by an excluded construct is the
        // tail of a sentence, not a sentence (see
        // `SentenceCtx::continues_previous`). Register's rewrites are
        // clause-sized; a dash or semicolon here is missing its left
        // context — most concretely, the opening dash of a paired
        // aside — so decline rather than rewrite half a construction.
        if ctx.continues_previous {
            continue;
        }
        let candidates = match kind {
            CandidateKind::ActivizeToPassive => {
                transduce::t4_activize_to_passive(text, &ctx.tokens, &ctx.parse)
            }
            CandidateKind::NominalizationUnpack => {
                transduce::t5_nominalization(text, &ctx.tokens, &ctx.parse)
            }
            CandidateKind::EmDash => transduce::t6_em_dash(text, &ctx.tokens, &ctx.parse),
            CandidateKind::Semicolon => transduce::t7_semicolon(text, &ctx.tokens, &ctx.parse),
            CandidateKind::CommaAnd => transduce::t8_comma_and(text, &ctx.tokens, &ctx.parse),
            CandidateKind::PastProgressive => {
                transduce::t9_past_progressive(text, &ctx.tokens, &ctx.parse)
            }
        };

        for candidate in candidates {
            if candidate.validate(text).is_err() {
                continue;
            }
            let doc_range =
                (ctx.range.start + candidate.range.start)..(ctx.range.start + candidate.range.end);

            if in_quoted_span(text, &candidate.range) {
                held.push(Finding::new(
                    rule_for(kind),
                    doc_range,
                    "register held: matched inside quotation",
                    Tier::Suggest,
                ));
                continue;
            }

            let span_tokens: Vec<&TaggedToken> = ctx
                .tokens
                .iter()
                .filter(|t| {
                    candidate.range.start <= t.token.range.start
                        && t.token.range.end <= candidate.range.end
                })
                .collect();
            let original_span_text = &text[candidate.range.clone()];
            let offending = closure_violation(
                kind,
                &candidate.replacement,
                original_span_text,
                &span_tokens,
            );
            if !offending.is_empty() {
                held.push(Finding::new(
                    rule_for(kind),
                    doc_range,
                    format!(
                        "register held: closure violation, introduces {}",
                        offending.join(", ")
                    ),
                    Tier::Suggest,
                ));
                continue;
            }

            let word_delta = word_count(&candidate.replacement) - word_count(original_span_text);
            pool.push(PositionedCandidate {
                sentence_index,
                local_range: candidate.range.clone(),
                doc_range,
                replacement: candidate.replacement,
                kind,
                delta: candidate.delta,
                word_delta,
            });
        }
    }
}

/// The closure gate: every content word in `replacement` must already
/// appear in `original_span_text`, be reachable via `kind`'s inflection
/// table from a lemma/surface in the matched span, or be one of
/// [`PERMITTED_FUNCTION_WORDS`]. Returns the offending tokens (sorted,
/// deduplicated), empty if none.
fn closure_violation(
    kind: CandidateKind,
    replacement: &str,
    original_span_text: &str,
    span_tokens: &[&TaggedToken],
) -> Vec<Box<str>> {
    let original_words: BTreeSet<Box<str>> = tokenize_str(original_span_text, 0)
        .into_iter()
        .filter(|t| t.kind == AnalysisTokenKind::Word)
        .map(|t| t.text)
        .collect();

    let mut derived: BTreeSet<Box<str>> = BTreeSet::new();
    match kind {
        CandidateKind::ActivizeToPassive => {
            for token in span_tokens {
                let lemma = token.lemma.as_ref();
                derived.insert(past(lemma).to_lowercase().into_boxed_str());
                derived.insert(past_participle(lemma).to_lowercase().into_boxed_str());
                derived.insert(third_sg(lemma).to_lowercase().into_boxed_str());
            }
        }
        CandidateKind::NominalizationUnpack => {
            for token in span_tokens {
                let surface_lower = original_span_text
                    .get(shift(token, original_span_text, span_tokens))
                    .unwrap_or_default()
                    .to_lowercase();
                if let Some(verb) = transduce::nominal_verb_for(&surface_lower) {
                    derived.insert(verb.to_lowercase().into_boxed_str());
                }
                // Lemma licensed too: for a bare singular noun it's
                // already the lowercase surface, but checking both is
                // free and covers a lemma that normalizes differently.
                if let Some(verb) = transduce::nominal_verb_for(token.lemma.as_ref()) {
                    derived.insert(verb.to_lowercase().into_boxed_str());
                }
            }
        }
        CandidateKind::PastProgressive => {
            // T9's only derived form: the participle's simple past, via
            // the same `past` table T4 draws from -- "was marking" ->
            // "marked", where "marked" is absent from
            // `original_span_text` ("was marking") itself.
            for token in span_tokens {
                derived.insert(past(token.lemma.as_ref()).to_lowercase().into_boxed_str());
            }
        }
        CandidateKind::EmDash | CandidateKind::Semicolon | CandidateKind::CommaAnd => {
            // No derived words: every T6/T7/T8 replacement is punctuation
            // (",", ". ", "; ", ": ") plus, at most, a word already
            // present in `original_span_text` (T6's interpolated middle,
            // T8's recapitalized word after "and", either kind's
            // recapitalized or unchanged following word) -- there is no
            // inflection table to consult, unlike the other two kinds.
        }
    }

    let mut offending: Vec<Box<str>> = tokenize_str(replacement, 0)
        .into_iter()
        .filter(|t| t.kind == AnalysisTokenKind::Word)
        .map(|t| t.text)
        .filter(|word| {
            !original_words.contains(word)
                && !PERMITTED_FUNCTION_WORDS.contains(&word.as_ref())
                && !derived.contains(word)
        })
        .collect();
    offending.sort();
    offending.dedup();
    offending
}

/// `token`'s byte range, shifted to slice `original_span_text` (which
/// starts at the matched span, not the sentence). `span_tokens` is
/// consulted only for its min start; by [`collect_candidates`]'s
/// filter, every token in it is already inside the span.
fn shift(
    token: &TaggedToken,
    original_span_text: &str,
    span_tokens: &[&TaggedToken],
) -> Range<usize> {
    let span_start = span_tokens
        .iter()
        .map(|t| t.token.range.start)
        .min()
        .unwrap_or(token.token.range.start);
    let start = token.token.range.start.saturating_sub(span_start);
    let end = token.token.range.end.saturating_sub(span_start);
    start.min(original_span_text.len())..end.min(original_span_text.len())
}

/// Walks up `start`'s dependency-head chain to the nearest finite-verb
/// token, returning its index — `None` if the chain reaches root
/// without passing through one (a non-finite fragment, or `start`
/// already at root).
fn governing_finite_verb(
    parse: &SentenceParse,
    tokens: &[TaggedToken],
    start: usize,
) -> Option<usize> {
    let mut current = start;
    let mut steps = 0usize;
    loop {
        if FINITE_VERB_TAGS.contains(&tokens[current].pos.as_str()) {
            return Some(current);
        }
        match parse.edge(current).and_then(|edge| edge.head) {
            Some(head) if steps < tokens.len() => {
                current = head;
                steps += 1;
            }
            _ => return None,
        }
    }
}

/// The token indices in `tokens` whose own range falls entirely inside
/// `range`.
fn tokens_in(tokens: &[TaggedToken], range: &Range<usize>) -> Vec<usize> {
    tokens
        .iter()
        .enumerate()
        .filter(|(_, t)| range.start <= t.token.range.start && t.token.range.end <= range.end)
        .map(|(i, _)| i)
        .collect()
}

/// `true` if `a` and `b` may not both apply (the three conditions from
/// this module's "Conflict" section). Only the last two need
/// same-sentence data: different sentences never share governance, and
/// their ranges can't overlap since sentence ranges are disjoint by
/// construction.
fn conflicts(a: &PositionedCandidate, b: &PositionedCandidate, sentences: &[SentenceCtx]) -> bool {
    if ranges_overlap(&a.doc_range, &b.doc_range) {
        return true;
    }
    if a.sentence_index != b.sentence_index {
        return false;
    }
    let ctx = &sentences[a.sentence_index];
    let a_tokens = tokens_in(&ctx.tokens, &a.local_range);
    let b_tokens = tokens_in(&ctx.tokens, &b.local_range);

    let a_governs_b = a_tokens.iter().any(|&ta| {
        b_tokens
            .iter()
            .any(|&tb| ctx.parse.edge(tb).and_then(|e| e.head) == Some(ta))
    });
    let b_governs_a = b_tokens.iter().any(|&tb| {
        a_tokens
            .iter()
            .any(|&ta| ctx.parse.edge(ta).and_then(|e| e.head) == Some(tb))
    });
    if a_governs_b || b_governs_a {
        return true;
    }

    let a_verbs: BTreeSet<usize> = a_tokens
        .iter()
        .filter_map(|&t| governing_finite_verb(&ctx.parse, &ctx.tokens, t))
        .collect();
    let b_verbs: BTreeSet<usize> = b_tokens
        .iter()
        .filter_map(|&t| governing_finite_verb(&ctx.parse, &ctx.tokens, t))
        .collect();
    a_verbs.intersection(&b_verbs).next().is_some()
}

/// Greedily selects candidates for `feature` from `pool` until its rate
/// enters `band` or no remaining non-conflicting candidate improves it,
/// applying each pick's exact effect (`delta[feature]`, measured
/// `word_delta`) to the running `(count, words)` before re-scoring.
/// Winners move from `pool` into `accepted`; `pool` keeps whatever the
/// other feature's phase might still need.
#[allow(clippy::too_many_arguments)]
fn select_and_apply(
    pool: &mut Vec<PositionedCandidate>,
    sentences: &[SentenceCtx],
    feature: &'static str,
    band: RegisterBand,
    direction: Direction,
    mut count: i64,
    mut words: i64,
    accepted: &mut Vec<PositionedCandidate>,
) -> (i64, i64) {
    loop {
        let current_rate = rate(count, words);
        if band.contains(current_rate) {
            break;
        }

        let mut best: Option<(usize, f64)> = None;
        for (index, candidate) in pool.iter().enumerate() {
            let Some(&delta_feature) = candidate.delta.get(feature) else {
                continue;
            };
            if accepted
                .iter()
                .any(|taken| conflicts(taken, candidate, sentences))
            {
                continue;
            }
            let new_count = count + i64::from(delta_feature);
            let new_words = words + candidate.word_delta;
            let new_rate = rate(new_count, new_words);
            let score = match direction {
                Direction::Increase => new_rate - current_rate,
                Direction::Decrease => current_rate - new_rate,
            };
            if score > 0.0 && best.is_none_or(|(_, best_score)| score > best_score) {
                best = Some((index, score));
            }
        }

        let Some((index, _)) = best else {
            break; // no remaining candidate improves this feature's rate.
        };
        let winner = pool.remove(index);
        count += i64::from(*winner.delta.get(feature).unwrap_or(&0));
        words += winner.word_delta;
        accepted.push(winner);
    }
    (count, words)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use friction_core::{Token, TokenKind};
    use friction_nlp::{Confidence, DepEdge, DepRelation, PosTag};

    use super::*;

    // --- Wilson bounds: evidence-proportional arming ---

    /// The motivating case: a 58-word comment with 4 nominalizations has
    /// a point rate of 68.97/1000, above the 50.56 band, but its lower
    /// bound (~27/1000) is not, so the feature must stay quiet. Seven
    /// instances in the same 58 words (~121/1000 point rate, lower bound
    /// ~60/1000) is real evidence and must arm.
    #[test]
    fn wilson_lower_holds_short_texts_to_a_higher_bar() {
        let band_high = 50.56;
        assert!(rate(4, 58) > band_high, "the point estimate IS above band");
        assert!(wilson_lower(4, 58) < band_high, "…but the evidence isn't");
        assert!((wilson_lower(4, 58) - 27.2).abs() < 0.5);
        assert!(
            wilson_lower(7, 58) > band_high,
            "7 instances in 58 words is"
        );
    }

    /// A zero band still arms on a single instance at any length — the
    /// lower bound of one-in-anything is above zero — so em-dash
    /// rewriting keeps working on short snippets with no special case.
    #[test]
    fn wilson_lower_of_one_instance_clears_a_zero_band_at_any_length() {
        for words in [5, 20, 100, 10_000] {
            assert!(wilson_lower(1, words) > 0.0, "words={words}");
        }
    }

    /// The semicolon band (high 3.66/1000): one semicolon in 40 words is
    /// a strong density signal and arms. One in 200 words is consistent
    /// with human use and doesn't.
    #[test]
    fn wilson_lower_separates_dense_from_incidental_semicolons() {
        assert!(wilson_lower(1, 40) > 3.66);
        assert!(wilson_lower(1, 200) < 3.66);
    }

    /// The Increase mirror: a 58-word text with zero passives has an
    /// upper bound of ~62/1000 — the sample can't establish confident
    /// under-use, so passivization must not arm on a short comment.
    #[test]
    fn wilson_upper_does_not_call_a_short_text_underpassivized() {
        assert!((wilson_upper(0, 58) - 62.1).abs() < 0.5);
        assert!(wilson_upper(0, 58) > 8.0);
    }

    /// Degenerate and determinism cases: an empty document yields
    /// `(0, 0)`, and the closed form is bit-stable across calls.
    #[test]
    fn wilson_bounds_are_deterministic_and_safe_on_empty_input() {
        assert_eq!(wilson_bounds(0, 0), (0.0, 0.0));
        assert_eq!(wilson_bounds(4, 58), wilson_bounds(4, 58));
        let (lower, upper) = wilson_bounds(4, 58);
        assert!(lower < rate(4, 58) && rate(4, 58) < upper);
    }

    // --- Band-measurement determinism ---

    /// Every `measure_*_rate` function, run twice on the same text through
    /// the same tagger/parser/segmenter, must return bit-identical `f64`s
    /// -- the byte-determinism bar this crate holds everywhere else (see
    /// `build_sentence_contexts`'s own docs on why `corpus-tool
    /// register-bands` can trust a remeasurement not to silently diverge
    /// from a previous run over the same corpus).
    type Measure = fn(&str, &dyn Tagger, &dyn DepParser, &dyn Segmenter) -> Result<f64, EditError>;

    // Exact `f64` equality is the correct check here, not an approximation
    // bug: the claim under test IS bit-identical output from two calls
    // with identical inputs, not "close enough".
    #[allow(clippy::float_cmp)]
    #[test]
    fn measure_functions_are_deterministic() {
        let tagger = friction_nlp::PerceptronTagger::new().expect("tagger loads");
        let parser = friction_nlp::PerceptronParser::new().expect("parser loads");
        let segmenter = friction_nlp::SrxSegmenter::new();
        let text = "The scanner reads each file once, and it never touches what it reads. It \
                     was checking every line before it moved to the next file. The team chose \
                     caching rather than a cache-aside pattern for the lookup; the report is \
                     clear, not vague, about what changed — even so.";

        let measures: [(&str, Measure); 5] = [
            ("em_dash", measure_em_dash_rate),
            ("semicolon", measure_semicolon_rate),
            ("contrast_closer", measure_contrast_closer_rate),
            ("comma_and", measure_comma_and_rate),
            ("past_progressive", measure_past_progressive_rate),
        ];
        for (name, measure) in measures {
            let a = measure(text, &tagger, &parser, &segmenter).expect("measurement succeeds");
            let b = measure(text, &tagger, &parser, &segmenter).expect("measurement succeeds");
            assert_eq!(a, b, "{name} must be deterministic across repeated calls");
        }
    }

    /// A candidate introducing a word outside the input, the inflection
    /// tables, and the permitted function-word set must be rejected by
    /// the closure gate.
    #[test]
    fn closure_violation_rejects_a_candidate_introducing_an_unlicensed_word() {
        // "We deployed the change." -- a genuine T4 firing produces
        // "The change was deployed", every word traceable to the span
        // or the four-word set. Splicing in "surely" (in neither the
        // span, the inflection tables, nor the permitted set) must be
        // flagged.
        let source = "We deployed the change.";
        let tokens = [
            tagged(source, "We", "PRP", "we"),
            tagged(source, "deployed", "VBD", "deploy"),
            tagged(source, "the", "DT", "the"),
            tagged(source, "change", "NN", "change"),
        ];
        let span_tokens: Vec<&TaggedToken> = tokens.iter().collect();
        let original_span_text = "We deployed the change";

        let offending = closure_violation(
            CandidateKind::ActivizeToPassive,
            "The change was surely deployed",
            original_span_text,
            &span_tokens,
        );
        assert_eq!(offending, vec![Box::from("surely")]);
    }

    /// A genuine T4 replacement -- every content word traced to the
    /// matched span, the sole function word ("was") permitted -- passes
    /// closure cleanly.
    #[test]
    fn closure_violation_accepts_a_genuine_t4_replacement() {
        let tokens = [
            tagged("We deployed the change", "We", "PRP", "we"),
            tagged("We deployed the change", "deployed", "VBD", "deploy"),
            tagged("We deployed the change", "the", "DT", "the"),
            tagged("We deployed the change", "change", "NN", "change"),
        ];
        let span_tokens: Vec<&TaggedToken> = tokens.iter().collect();
        let offending = closure_violation(
            CandidateKind::ActivizeToPassive,
            "The change was deployed",
            "We deployed the change",
            &span_tokens,
        );
        assert!(
            offending.is_empty(),
            "expected no violations, got {offending:?}"
        );
    }

    /// A genuine T5 replacement's derived verb ("optimizing", absent
    /// from "the optimization of the query") is licensed via
    /// `nominal_verb_for`, the exact table that produced it.
    #[test]
    fn closure_violation_accepts_a_genuine_t5_replacement() {
        let source = "the optimization of the query";
        let tokens = [
            tagged(source, "the", "DT", "the"),
            tagged(source, "optimization", "NN", "optimization"),
            tagged(source, "of", "IN", "of"),
            tagged(source, "the", "DT", "the"),
            tagged(source, "query", "NN", "query"),
        ];
        let span_tokens: Vec<&TaggedToken> = tokens.iter().collect();
        let offending = closure_violation(
            CandidateKind::NominalizationUnpack,
            "optimizing the query",
            source,
            &span_tokens,
        );
        assert!(
            offending.is_empty(),
            "expected no violations, got {offending:?}"
        );
    }

    /// Builds one `TaggedToken` for `surface` in `full_text` -- a test
    /// helper, so a fixed linear scan per call is fine.
    fn tagged(full_text: &str, surface: &str, pos: &str, lemma: &str) -> TaggedToken {
        let start = full_text
            .find(surface)
            .expect("surface occurs in fixture text");
        TaggedToken {
            token: Token::new(start..start + surface.len(), TokenKind::Word),
            pos: PosTag::new(pos),
            lemma: Box::from(lemma),
        }
    }

    /// `governing_finite_verb` finds the nearest finite-verb ancestor
    /// for a subject/object token, and `None` for a fragment with no
    /// finite verb at all.
    #[test]
    fn governing_finite_verb_walks_up_to_the_nearest_finite_verb() {
        // "We deployed the change." -- deployed(1) is root/finite.
        // We(0) is nsubj of 1; the(2) is det of change(3); change(3) is
        // dobj of 1.
        let edges = vec![
            DepEdge {
                token: 0,
                head: Some(1),
                relation: DepRelation::Nsubj,
                confidence: Confidence::CERTAIN,
            },
            DepEdge {
                token: 1,
                head: None,
                relation: DepRelation::Root,
                confidence: Confidence::CERTAIN,
            },
            DepEdge {
                token: 2,
                head: Some(3),
                relation: DepRelation::Det,
                confidence: Confidence::CERTAIN,
            },
            DepEdge {
                token: 3,
                head: Some(1),
                relation: DepRelation::Dobj,
                confidence: Confidence::CERTAIN,
            },
        ];
        let parse = SentenceParse::new(edges).unwrap();
        let tokens = [
            tagged("We deployed the change", "We", "PRP", "we"),
            tagged("We deployed the change", "deployed", "VBD", "deploy"),
            tagged("We deployed the change", "the", "DT", "the"),
            tagged("We deployed the change", "change", "NN", "change"),
        ];
        assert_eq!(governing_finite_verb(&parse, &tokens, 0), Some(1));
        assert_eq!(governing_finite_verb(&parse, &tokens, 3), Some(1));
        assert_eq!(governing_finite_verb(&parse, &tokens, 1), Some(1));
    }

    /// Two candidates whose byte ranges overlap conflict outright, no
    /// parse needed.
    #[test]
    fn conflicts_detects_overlapping_ranges() {
        let ctx = SentenceCtx {
            range: 0..10,
            tokens: vec![tagged(
                "placeholder text",
                "placeholder",
                "NN",
                "placeholder",
            )],
            parse: SentenceParse::new(vec![DepEdge {
                token: 0,
                head: None,
                relation: DepRelation::Root,
                confidence: Confidence::CERTAIN,
            }])
            .unwrap(),
            continues_previous: false,
        };
        let a = candidate_at(0, 0..5, 0..5);
        let b = candidate_at(0, 3..8, 3..8);
        assert!(conflicts(&a, &b, &[ctx]));
    }

    /// Two candidates in different sentences never conflict, regardless
    /// of their (necessarily disjoint) ranges.
    #[test]
    fn conflicts_is_false_across_different_sentences() {
        let ctx0 = fragment_ctx(0..10);
        let ctx1 = fragment_ctx(10..20);
        let a = candidate_at(0, 0..5, 0..5);
        let b = candidate_at(1, 10..15, 0..5);
        assert!(!conflicts(&a, &b, &[ctx0, ctx1]));
    }

    fn fragment_ctx(range: Range<usize>) -> SentenceCtx {
        SentenceCtx {
            range,
            tokens: vec![tagged("x", "x", "NN", "x")],
            parse: SentenceParse::new(vec![DepEdge {
                token: 0,
                head: None,
                relation: DepRelation::Root,
                confidence: Confidence::CERTAIN,
            }])
            .unwrap(),
            continues_previous: false,
        }
    }

    fn candidate_at(
        sentence_index: usize,
        doc_range: Range<usize>,
        local_range: Range<usize>,
    ) -> PositionedCandidate {
        PositionedCandidate {
            sentence_index,
            local_range,
            doc_range,
            replacement: Box::from(""),
            kind: CandidateKind::ActivizeToPassive,
            delta: BTreeMap::new(),
            word_delta: 0,
        }
    }
}
