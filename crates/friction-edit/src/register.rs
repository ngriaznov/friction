//! The register pass.
//!
//! After the five-operation passes converge, home four register features — nominalization,
//! agentless passive, em dash, semicolon — toward their human rate band by selecting and
//! applying [`friction_register::transduce`] candidates. The first two are Biber constructions;
//! the last two aren't Biber's, but the same band-and-transducer machinery applies to them
//! unchanged (see `register-v1.toml`'s `[features.em_dash]`/`[features.semicolon]`).
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
//! Candidates are scored by how far their delta moves the feature's *rate* — count over prose
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
//! Every candidate passes [`closure_violation`] before entering the pool — no path from
//! candidate to patch skips it. A content word must already appear in the matched span or be
//! reachable through the transducer's own inflection table; a function word must be one of the
//! four permitted (see [`friction_register::transduce::PERMITTED_FUNCTION_WORDS`]). A candidate
//! that fails is dropped and held, never applied.

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

use friction_core::span::ranges_overlap;
use friction_core::{Finding, Patch, RuleId, Tier};
use friction_match::token::{AnalysisTokenKind, prose_scope, tokenize_str};
use friction_nlp::{DepParser, FINITE_VERB_TAGS, Segmenter, SentenceParse, TaggedToken, Tagger};
use friction_packs::{RegisterBand, RegisterPack};
use friction_register::features::RegisterCounts;
use friction_register::transduce::{
    self, CandidateKind, PERMITTED_FUNCTION_WORDS, past, past_participle, third_sg,
};

use crate::document::{apply, ends_with_sentence_terminal_punctuation, resolve};
use crate::error::EditError;
use crate::gates::in_quoted_span;

const RULE_PASSIVIZE: RuleId = RuleId::new("register.passivize");
const RULE_UNPACK: RuleId = RuleId::new("register.unpack");
const RULE_EM_DASH: RuleId = RuleId::new("register.em_dash");
const RULE_SEMICOLON: RuleId = RuleId::new("register.semicolon");

const fn rule_for(kind: CandidateKind) -> RuleId {
    match kind {
        CandidateKind::ActivizeToPassive => RULE_PASSIVIZE,
        CandidateKind::NominalizationUnpack => RULE_UNPACK,
        CandidateKind::EmDash => RULE_EM_DASH,
        CandidateKind::Semicolon => RULE_SEMICOLON,
    }
}

/// One in-scope sentence's text range, tags, and dependency parse —
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
    /// (nominalization), [`CandidateKind::EmDash`] (em dash), and
    /// [`CandidateKind::Semicolon`] (semicolon) all move this way.
    Decrease,
}

/// Tags and parses every non-empty sentence in `units`, silently
/// dropping any sentence a per-sentence parse failure or empty trimmed
/// text excludes.
fn build_sentence_contexts(
    source: &str,
    units: &[friction_match::token::ScopedUnit<'_>],
    tagger: &dyn Tagger,
    parser: &dyn DepParser,
) -> Vec<SentenceCtx> {
    let mut sentences = Vec::new();
    for unit in units {
        for range in &unit.sentences {
            let Some(text) = source.get(range.clone()) else {
                continue;
            };
            if text.trim().is_empty() {
                continue;
            }
            let tokens = tagger.tag(text, 0);
            let Ok(parse) = parser.parse(text, &tokens) else {
                // A per-sentence parse failure (never expected from the
                // shipped perceptron parser, but allowed by
                // `DepParser::parse`'s contract) drops that sentence
                // rather than failing the whole document — the four
                // operations already ran and aren't rolled back for a
                // register-only problem.
                continue;
            };
            sentences.push(SentenceCtx {
                range: range.clone(),
                tokens,
                parse,
            });
        }
    }
    sentences
}

/// The document-wide `(total_words, nominalization_count,
/// agentless_passive_count, em_dash_count, semicolon_count)` totals over
/// `sentences`.
fn count_features(sentences: &[SentenceCtx], source: &str) -> (i64, i64, i64, i64, i64) {
    let mut total_words: i64 = 0;
    let mut count_nominalization: i64 = 0;
    let mut count_agentless_passive: i64 = 0;
    let mut count_em_dash: i64 = 0;
    let mut count_semicolon: i64 = 0;
    for ctx in sentences {
        let text = &source[ctx.range.clone()];
        let counts = RegisterCounts::count(text, &ctx.tokens, &ctx.parse);
        count_nominalization += i64::try_from(counts.nominalization).unwrap_or(i64::MAX);
        count_agentless_passive += i64::try_from(counts.agentless_passive).unwrap_or(i64::MAX);
        count_em_dash += i64::try_from(counts.em_dashes).unwrap_or(i64::MAX);
        count_semicolon += i64::try_from(counts.semicolons).unwrap_or(i64::MAX);
        total_words += word_count(text);
    }
    (
        total_words,
        count_nominalization,
        count_agentless_passive,
        count_em_dash,
        count_semicolon,
    )
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
    let (total_words, _, _, count_em_dash, _) = count_features(&sentences, source);
    Ok(rate(count_em_dash, total_words))
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
    let (total_words, _, _, _, count_semicolon) = count_features(&sentences, source);
    Ok(rate(count_semicolon, total_words))
}

/// Runs the register pass once over `source`.
///
/// `source` is expected to already be the five-operation pipeline's converged output; register
/// never interleaves with those passes.
///
/// # Errors
/// Returns [`EditError`] if `source` fails to parse or segment.
pub fn run_register(
    source: &str,
    register_pack: &RegisterPack,
    tagger: &dyn Tagger,
    parser: &dyn DepParser,
    segmenter: &dyn Segmenter,
) -> Result<(String, crate::document::PassReport), EditError> {
    let document = friction_parse::parse(source)?;
    let units = prose_scope(&document, segmenter);
    let sentences = build_sentence_contexts(source, &units, tagger, parser);
    let (
        mut total_words,
        count_nominalization,
        count_agentless_passive,
        count_em_dash,
        count_semicolon,
    ) = count_features(&sentences, source);

    let mut held: Vec<Finding> = Vec::new();
    if total_words == 0 {
        return Ok((source.to_string(), empty_pass()));
    }

    let features = feature_plan(
        register_pack,
        total_words,
        count_agentless_passive,
        count_nominalization,
        count_em_dash,
        count_semicolon,
    );

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
            // A Decrease feature can run out of licensed candidates while
            // still above its band — every remaining instance was a
            // hold-don't-guess decline at candidate-generation time, which
            // leaves no per-span finding of its own. Surface ONE
            // document-level Suggest finding so the report says why the
            // visible tells remain, instead of ending silent.
            if direction == Direction::Decrease && rate(final_count, total_words) > band.high {
                held.push(Finding {
                    rule: rule_for(kind),
                    range: 0..0,
                    message: format!(
                        "{feature}: {final_count} instance(s) remain and the document is still \
                         above the human band ({:.2} > {:.2} per 1000 words) — none has a \
                         licensed rewrite, so each needs a human hand",
                        rate(final_count, total_words),
                        band.high
                    ),
                    tier: Tier::Suggest,
                });
            }
        }
    }
    let _ = total_words;

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

    Ok((
        next,
        crate::document::PassReport {
            patches_applied,
            patches_dropped: dropped,
            applied_patches: resolved,
            held,
        },
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
fn feature_plan(
    register_pack: &RegisterPack,
    total_words: i64,
    count_agentless_passive: i64,
    count_nominalization: i64,
    count_em_dash: i64,
    count_semicolon: i64,
) -> [(
    &'static str,
    Option<RegisterBand>,
    Direction,
    i64,
    CandidateKind,
); 4] {
    [
        (
            "agentless_passive",
            register_pack
                .band("agentless_passive")
                .filter(|band| rate(count_agentless_passive, total_words) < band.low),
            Direction::Increase,
            count_agentless_passive,
            CandidateKind::ActivizeToPassive,
        ),
        (
            "nominalization",
            register_pack
                .band("nominalization")
                .filter(|band| rate(count_nominalization, total_words) > band.high),
            Direction::Decrease,
            count_nominalization,
            CandidateKind::NominalizationUnpack,
        ),
        (
            "em_dash",
            register_pack
                .band("em_dash")
                .filter(|band| rate(count_em_dash, total_words) > band.high),
            Direction::Decrease,
            count_em_dash,
            CandidateKind::EmDash,
        ),
        (
            "semicolon",
            register_pack
                .band("semicolon")
                .filter(|band| rate(count_semicolon, total_words) > band.high),
            Direction::Decrease,
            count_semicolon,
            CandidateKind::Semicolon,
        ),
    ]
}

/// Runs [`collect_candidates`] for exactly the features that need fixing
/// -- split out of [`run_register`] itself only to keep that function's
/// own line count down; `needed` is the kinds whose feature sits outside
/// its band, in application order — this helper never needs the bands
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
        let candidates = match kind {
            CandidateKind::ActivizeToPassive => {
                transduce::t4_activize_to_passive(text, &ctx.tokens, &ctx.parse)
            }
            CandidateKind::NominalizationUnpack => {
                transduce::t5_nominalization(text, &ctx.tokens, &ctx.parse)
            }
            CandidateKind::EmDash => transduce::t6_em_dash(text, &ctx.tokens, &ctx.parse),
            CandidateKind::Semicolon => transduce::t7_semicolon(text, &ctx.tokens, &ctx.parse),
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
        CandidateKind::EmDash | CandidateKind::Semicolon => {
            // No derived words: every T6/T7 replacement is punctuation
            // (",", ". ", "; ", ": ") plus, at most, a word already
            // present in `original_span_text` (T6's interpolated middle,
            // either kind's recapitalized or unchanged following word)
            // -- there is no inflection table to consult, unlike the
            // other two kinds.
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
        // "We deployed the change." -- deployed(1) is root/finite;
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
