//! The curated tell-span inventory pack (`inventory-v1`): deletion spans,
//! substitution pairs, ritual frames, a (currently empty) preview-frame
//! family, licensed light-verb-construction pairs, guard-token closed
//! classes, a small closed function-word allowance, and per-frame output-
//! frequency hygiene bands.
//!
//! This module is the read side only: [`InventoryPack::parse`] turns the
//! pack's TOML shape into typed, validated, deterministically-sorted
//! in-memory tables. `crates/friction-packs/packs/inventory-en-v1.toml`
//! itself is hand-edited, not generated. See that file's own header
//! comment for the curation discipline.

use std::collections::{BTreeMap, BTreeSet};

use rayon::prelude::*;
use regex::Regex;
use serde::Deserialize;

use crate::PackError;

/// Where a `deletion_spans` pattern is anchored within its sentence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Anchor {
    /// The pattern requires `^` (start of sentence).
    SentenceInitial,
    /// The pattern carries no positional anchor.
    MidSentence,
    /// The pattern requires the sentence's end.
    Trailing,
}

impl Anchor {
    fn parse(id: &str, value: &str) -> Result<Self, PackError> {
        match value {
            "sentence_initial" => Ok(Self::SentenceInitial),
            "mid_sentence" => Ok(Self::MidSentence),
            "trailing" => Ok(Self::Trailing),
            other => Err(PackError::InventoryUnknownAnchor {
                id: id.to_string(),
                value: other.to_string(),
            }),
        }
    }
}

/// A `deletion_spans`/`ritual_frames`/`preview_frames` entry's repair operation.
///
/// `substitution_pairs` entries do not carry this type at all — their
/// repair is structurally always "substitute" (parsed and discarded at
/// load time, never stored as a field), so a [`SubstitutionPair`] can
/// never reach a [`RepairKind::DiagnosticOnly`] state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepairKind {
    /// The matched span is deleted with no replacement text.
    Delete,
    /// Recorded for honesty but has no safe closed-set rewrite; must
    /// never be executed as a rewrite rule.
    DiagnosticOnly,
}

impl RepairKind {
    fn parse(id: &str, value: &str) -> Result<Self, PackError> {
        match value {
            "delete" => Ok(Self::Delete),
            "diagnostic_only" => Ok(Self::DiagnosticOnly),
            other => Err(PackError::InventoryUnknownRepairKind {
                id: id.to_string(),
                value: other.to_string(),
            }),
        }
    }
}

/// The unit an [`OutputFrequencyBand`]'s rate is expressed in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrequencyUnit {
    /// Occurrences per 1000 word tokens.
    PerThousandWords,
}

impl FrequencyUnit {
    fn parse(frame: &str, value: &str) -> Result<Self, PackError> {
        match value {
            "per_thousand_words" => Ok(Self::PerThousandWords),
            other => Err(PackError::InventoryUnknownFrequencyUnit {
                frame: frame.to_string(),
                value: other.to_string(),
            }),
        }
    }
}

/// A gated span deletion: whole match removed, no replacement text.
#[derive(Debug, Clone)]
pub struct DeletionSpan {
    pub id: Box<str>,
    pub pattern: Regex,
    pub pattern_source: Box<str>,
    pub anchor: Anchor,
    pub repair: RepairKind,
    pub source: Box<str>,
    pub notes: Option<Box<str>>,
}

/// A substitution pair: matched span replaced with a fixed, plain-register
/// string.
#[derive(Debug, Clone)]
pub struct SubstitutionPair {
    pub id: Box<str>,
    pub pattern: Regex,
    pub pattern_source: Box<str>,
    /// The fixed replacement text. `repair` is structurally always
    /// "substitute" for this family — not a stored field, see
    /// [`RepairKind`]'s own docs.
    pub replacement: Box<str>,
    pub source: Box<str>,
    pub notes: Option<Box<str>>,
    /// Content-word tokens [`crate::validate::literal_tokens_from_pattern`]'s
    /// heuristic can't recover from `pattern_source` cleanly (e.g.
    /// alternation): the closure check's escape hatch for a pattern the
    /// heuristic can't parse.
    pub declared_literal_tokens: Vec<Box<str>>,
    /// Content-word tokens in `replacement` a curator hand-attests are
    /// licensed, with a non-empty `notes` explaining why — see
    /// [`crate::validate::check_closure`].
    pub attested_replacement_tokens: Vec<Box<str>>,
}

/// A whole-sentence ritual frame: deleted whole, or, for a
/// `diagnostic_only` entry, recorded without a rewrite because none is
/// safe.
#[derive(Debug, Clone)]
pub struct RitualFrame {
    pub id: Box<str>,
    pub pattern: Regex,
    pub pattern_source: Box<str>,
    pub scope: Box<str>,
    pub repair: RepairKind,
    pub source: Box<str>,
    pub notes: Option<Box<str>>,
    pub train_machine_count: Option<u32>,
    pub train_human_count: Option<u32>,
    pub train_stock_count: Option<u32>,
    pub train_antislop_count: Option<u32>,
}

/// A block-position-conditioned preview-frame rule. New family, empty in
/// the shipped pack today (see that file's own header comment).
#[derive(Debug, Clone)]
pub struct PreviewFrame {
    pub id: Box<str>,
    pub pattern: Regex,
    pub pattern_source: Box<str>,
    pub requires_header_lemma_coverage: bool,
    pub repair: RepairKind,
    pub source: Box<str>,
    pub notes: Option<Box<str>>,
}

/// A licensed light-verb/nominalization pair.
#[derive(Debug, Clone)]
pub struct LvcPair {
    /// Must be one of `perform`/`conduct`/`make`/`do` — checked against
    /// `friction_nlp::lvc::LIGHT_VERBS` at parse time.
    pub light_verb_base: Box<str>,
    /// Unique key across `lvc_pairs`.
    pub nominalization: Box<str>,
    pub derived_verb: Box<str>,
    pub source: Box<str>,
    pub train_construction_count_machine: u32,
    pub train_construction_count_human: u32,
    pub train_verb_count_machine: u32,
    pub train_verb_count_human: u32,
}

/// The four closed English syntactic guard-token classes.
#[derive(Debug, Clone, Default)]
pub struct GuardTokens {
    pub negation: Vec<Box<str>>,
    pub quantifiers: Vec<Box<str>>,
    pub modals: Vec<Box<str>>,
    pub connectives: Vec<Box<str>>,
}

/// A per-frame output-frequency hygiene band: the natural rate a curator
/// measured in the train-human corpus, plus a hygiene ceiling.
#[derive(Debug, Clone)]
pub struct OutputFrequencyBand {
    /// Exact literal replacement text, matched case-insensitively.
    pub frame: Box<str>,
    pub unit: FrequencyUnit,
    /// Measured natural rate in the train-human corpus.
    pub observed_rate: f64,
    /// Hygiene ceiling; must be `>= observed_rate`.
    pub max: f64,
    pub sample_doc_count: u32,
    /// Auditable method label, e.g. `"human_train_rate_x3"`.
    pub method: Box<str>,
    pub notes: Option<Box<str>>,
}

/// A parsed, validated, deterministically-sorted `inventory-v1` pack.
#[derive(Debug, Clone)]
pub struct InventoryPack {
    version: Box<str>,
    corpus_manifest_sha256: Box<str>,
    paired_manifest_sha256: Box<str>,
    deletion_spans: Vec<DeletionSpan>,
    substitution_pairs: Vec<SubstitutionPair>,
    ritual_frames: Vec<RitualFrame>,
    preview_frames: Vec<PreviewFrame>,
    lvc_pairs: Vec<LvcPair>,
    guard_tokens: GuardTokens,
    closure_function_word_allowance: Vec<Box<str>>,
    output_frequency_bands: Vec<OutputFrequencyBand>,
    lvc_lexicon: BTreeMap<Box<str>, Box<str>>,
}

impl InventoryPack {
    /// Parses `toml`'s `inventory-v1`-shaped contents into a validated
    /// pack.
    ///
    /// # Errors
    /// Returns [`PackError::Toml`] if `toml` is not valid TOML in the
    /// expected shape. Returns one of the `Inventory*` [`PackError`]
    /// variants for a structural violation this parse enforces. See each
    /// variant's own docs.
    pub fn parse(toml: &str) -> Result<Self, PackError> {
        // The four regex-carrying families compile in parallel: regex
        // construction (~200 µs a pattern, ~31 patterns) is this parse's
        // entire cost, paid on every process's first pack touch.
        // Determinism: an indexed `into_par_iter().map(..).collect()`
        // yields the same Vec<Result> in declaration order regardless of
        // thread count, and `first_error` then picks the winning error
        // sequentially: same value AND same error selection as the old
        // serial loop, byte for byte.
        fn first_error<T: Send>(results: Vec<Result<T, PackError>>) -> Result<Vec<T>, PackError> {
            results.into_iter().collect()
        }

        let raw: RawPack = toml::from_str(toml).map_err(PackError::from)?;
        let mut deletion_spans: Vec<DeletionSpan> = first_error(
            raw.deletion_spans
                .into_par_iter()
                .map(DeletionSpan::try_from_raw)
                .collect(),
        )?;
        let mut substitution_pairs: Vec<SubstitutionPair> = first_error(
            raw.substitution_pairs
                .into_par_iter()
                .map(SubstitutionPair::try_from_raw)
                .collect(),
        )?;
        let mut ritual_frames: Vec<RitualFrame> = first_error(
            raw.ritual_frames
                .into_par_iter()
                .map(RitualFrame::try_from_raw)
                .collect(),
        )?;
        let mut preview_frames: Vec<PreviewFrame> = first_error(
            raw.preview_frames
                .into_par_iter()
                .map(PreviewFrame::try_from_raw)
                .collect(),
        )?;
        let mut lvc_pairs: Vec<LvcPair> = raw
            .lvc_pairs
            .into_iter()
            .map(LvcPair::try_from_raw)
            .collect::<Result<_, _>>()?;
        let mut output_frequency_bands: Vec<OutputFrequencyBand> = raw
            .output_frequency_bands
            .into_iter()
            .map(OutputFrequencyBand::try_from_raw)
            .collect::<Result<_, _>>()?;

        deletion_spans.sort_by(|a, b| a.id.cmp(&b.id));
        substitution_pairs.sort_by(|a, b| a.id.cmp(&b.id));
        ritual_frames.sort_by(|a, b| a.id.cmp(&b.id));
        preview_frames.sort_by(|a, b| a.id.cmp(&b.id));
        lvc_pairs.sort_by(|a, b| a.nominalization.cmp(&b.nominalization));
        output_frequency_bands.sort_by(|a, b| a.frame.cmp(&b.frame));

        let mut seen_ids: BTreeSet<&str> = BTreeSet::new();
        for id in deletion_spans
            .iter()
            .map(|e| e.id.as_ref())
            .chain(substitution_pairs.iter().map(|e| e.id.as_ref()))
            .chain(ritual_frames.iter().map(|e| e.id.as_ref()))
            .chain(preview_frames.iter().map(|e| e.id.as_ref()))
        {
            if !seen_ids.insert(id) {
                return Err(PackError::InventoryDuplicateId { id: id.to_string() });
            }
        }

        let mut seen_noms: BTreeSet<&str> = BTreeSet::new();
        for pair in &lvc_pairs {
            if !seen_noms.insert(pair.nominalization.as_ref()) {
                return Err(PackError::InventoryDuplicateLvcNominalization {
                    nominalization: pair.nominalization.to_string(),
                });
            }
        }

        let lvc_lexicon: BTreeMap<Box<str>, Box<str>> = lvc_pairs
            .iter()
            .map(|p| (p.nominalization.clone(), p.derived_verb.clone()))
            .collect();

        Ok(Self {
            version: raw.pack.version.into_boxed_str(),
            corpus_manifest_sha256: raw.pack.corpus_manifest_sha256.into_boxed_str(),
            paired_manifest_sha256: raw.pack.paired_manifest_sha256.into_boxed_str(),
            deletion_spans,
            substitution_pairs,
            ritual_frames,
            preview_frames,
            lvc_pairs,
            guard_tokens: raw.guard_tokens.into(),
            closure_function_word_allowance: raw
                .closure_function_word_allowance
                .into_iter()
                .map(String::into_boxed_str)
                .collect(),
            output_frequency_bands,
            lvc_lexicon,
        })
    }

    #[must_use]
    pub fn version(&self) -> &str {
        &self.version
    }

    #[must_use]
    pub fn corpus_manifest_sha256(&self) -> &str {
        &self.corpus_manifest_sha256
    }

    #[must_use]
    pub fn paired_manifest_sha256(&self) -> &str {
        &self.paired_manifest_sha256
    }

    #[must_use]
    pub fn deletion_spans(&self) -> &[DeletionSpan] {
        &self.deletion_spans
    }

    #[must_use]
    pub fn substitution_pairs(&self) -> &[SubstitutionPair] {
        &self.substitution_pairs
    }

    #[must_use]
    pub fn ritual_frames(&self) -> &[RitualFrame] {
        &self.ritual_frames
    }

    #[must_use]
    pub fn preview_frames(&self) -> &[PreviewFrame] {
        &self.preview_frames
    }

    #[must_use]
    pub fn lvc_pairs(&self) -> &[LvcPair] {
        &self.lvc_pairs
    }

    #[must_use]
    pub const fn guard_tokens(&self) -> &GuardTokens {
        &self.guard_tokens
    }

    #[must_use]
    pub fn closure_function_word_allowance(&self) -> &[Box<str>] {
        &self.closure_function_word_allowance
    }

    #[must_use]
    pub fn output_frequency_bands(&self) -> &[OutputFrequencyBand] {
        &self.output_frequency_bands
    }

    /// `nominalization -> derived_verb`, precomputed at parse time from
    /// `lvc_pairs`.
    #[must_use]
    pub const fn lvc_lexicon(&self) -> &BTreeMap<Box<str>, Box<str>> {
        &self.lvc_lexicon
    }
}

/// Rewrites every literal space in `pattern_source` into `\s+`, leaving
/// spaces inside a character class and escaped spaces alone.
///
/// Patterns are authored as running text (`will walk you through`), but the
/// text they run against is Markdown, and Markdown is routinely hard-wrapped
/// at a fixed column. A wrap puts a newline in the middle of a phrase, so a
/// pattern written with literal spaces silently stops matching: `"if you
/// have any questions or require further\nassistance"` matched nothing at
/// all, and the failure is invisible — the engine simply reports no finding,
/// which is indistinguishable from clean text.
///
/// Done here rather than by authoring `\s+` in each pattern by hand: a
/// pattern is a statement about words, the wrapping is an artifact of how
/// the file happens to be stored, and 31 hand-written separators is 31
/// chances to forget one.
fn whitespace_flexible(pattern_source: &str) -> String {
    let mut out = String::with_capacity(pattern_source.len());
    let mut chars = pattern_source.chars();
    let mut in_class = false;
    while let Some(c) = chars.next() {
        match c {
            '\\' => {
                out.push(c);
                if let Some(escaped) = chars.next() {
                    out.push(escaped);
                }
            }
            '[' => {
                in_class = true;
                out.push(c);
            }
            ']' => {
                in_class = false;
                out.push(c);
            }
            ' ' if !in_class => out.push_str(r"\s+"),
            _ => out.push(c),
        }
    }
    out
}

fn compile_pattern(id: &str, pattern_source: &str) -> Result<Regex, PackError> {
    // `dot_matches_new_line` for the same reason: a pattern using `.` to
    // skip over intervening words means "any words", and a hard wrap in the
    // middle of those words should not end the match.
    regex::RegexBuilder::new(&whitespace_flexible(pattern_source))
        .dot_matches_new_line(true)
        .build()
        .map_err(|e| PackError::InventoryInvalidPattern {
            id: id.to_string(),
            pattern: pattern_source.to_string(),
            message: e.to_string(),
        })
}

impl DeletionSpan {
    fn try_from_raw(raw: RawDeletionSpan) -> Result<Self, PackError> {
        let pattern = compile_pattern(&raw.id, &raw.pattern)?;
        let anchor = Anchor::parse(&raw.id, &raw.anchor)?;
        let repair = RepairKind::parse(&raw.id, &raw.repair)?;
        Ok(Self {
            id: raw.id.into_boxed_str(),
            pattern,
            pattern_source: raw.pattern.into_boxed_str(),
            anchor,
            repair,
            source: raw.source.into_boxed_str(),
            notes: raw.notes.map(String::into_boxed_str),
        })
    }
}

impl SubstitutionPair {
    fn try_from_raw(raw: RawSubstitutionPair) -> Result<Self, PackError> {
        if raw.repair != "substitute" {
            return Err(PackError::InventoryInvalidRepairForFamily {
                id: raw.id,
                family: "substitution_pairs",
                found: raw.repair,
            });
        }
        let pattern = compile_pattern(&raw.id, &raw.pattern)?;
        Ok(Self {
            id: raw.id.into_boxed_str(),
            pattern,
            pattern_source: raw.pattern.into_boxed_str(),
            replacement: raw.replacement.into_boxed_str(),
            source: raw.source.into_boxed_str(),
            notes: raw.notes.map(String::into_boxed_str),
            declared_literal_tokens: raw
                .declared_literal_tokens
                .into_iter()
                .map(String::into_boxed_str)
                .collect(),
            attested_replacement_tokens: raw
                .attested_replacement_tokens
                .into_iter()
                .map(String::into_boxed_str)
                .collect(),
        })
    }
}

impl RitualFrame {
    fn try_from_raw(raw: RawRitualFrame) -> Result<Self, PackError> {
        let pattern = compile_pattern(&raw.id, &raw.pattern)?;
        let repair = RepairKind::parse(&raw.id, &raw.repair)?;
        Ok(Self {
            id: raw.id.into_boxed_str(),
            pattern,
            pattern_source: raw.pattern.into_boxed_str(),
            scope: raw.scope.into_boxed_str(),
            repair,
            source: raw.source.into_boxed_str(),
            notes: raw.notes.map(String::into_boxed_str),
            train_machine_count: raw.train_machine_count,
            train_human_count: raw.train_human_count,
            train_stock_count: raw.train_stock_count,
            train_antislop_count: raw.train_antislop_count,
        })
    }
}

impl PreviewFrame {
    fn try_from_raw(raw: RawPreviewFrame) -> Result<Self, PackError> {
        let pattern = compile_pattern(&raw.id, &raw.pattern)?;
        let repair = RepairKind::parse(&raw.id, &raw.repair)?;
        Ok(Self {
            id: raw.id.into_boxed_str(),
            pattern,
            pattern_source: raw.pattern.into_boxed_str(),
            requires_header_lemma_coverage: raw.requires_header_lemma_coverage,
            repair,
            source: raw.source.into_boxed_str(),
            notes: raw.notes.map(String::into_boxed_str),
        })
    }
}

impl LvcPair {
    fn try_from_raw(raw: RawLvcPair) -> Result<Self, PackError> {
        let resolves = friction_nlp::lvc::DERIVATIONAL_LEXICON
            .get(raw.nominalization.as_str())
            .is_some_and(|&derived| derived == raw.derived_verb);
        if !resolves {
            return Err(PackError::InventoryUnresolvableLvcPair {
                nominalization: raw.nominalization,
                derived_verb: raw.derived_verb,
            });
        }
        let is_base_form = friction_nlp::lvc::LIGHT_VERBS
            .get(raw.light_verb_base.as_str())
            .is_some_and(|&form| form == friction_nlp::lvc::LightVerbForm::Base);
        if !is_base_form {
            return Err(PackError::InventoryUnknownLightVerbBase {
                nominalization: raw.nominalization,
                light_verb_base: raw.light_verb_base,
            });
        }
        Ok(Self {
            light_verb_base: raw.light_verb_base.into_boxed_str(),
            nominalization: raw.nominalization.into_boxed_str(),
            derived_verb: raw.derived_verb.into_boxed_str(),
            source: raw.source.into_boxed_str(),
            train_construction_count_machine: raw.train_construction_count_machine,
            train_construction_count_human: raw.train_construction_count_human,
            train_verb_count_machine: raw.train_verb_count_machine,
            train_verb_count_human: raw.train_verb_count_human,
        })
    }
}

impl OutputFrequencyBand {
    fn try_from_raw(raw: RawOutputFrequencyBand) -> Result<Self, PackError> {
        let unit = FrequencyUnit::parse(&raw.frame, &raw.unit)?;
        Ok(Self {
            frame: raw.frame.into_boxed_str(),
            unit,
            observed_rate: raw.observed_rate,
            max: raw.max,
            sample_doc_count: raw.sample_doc_count,
            method: raw.method.into_boxed_str(),
            notes: raw.notes.map(String::into_boxed_str),
        })
    }
}

impl From<RawGuardTokens> for GuardTokens {
    fn from(raw: RawGuardTokens) -> Self {
        Self {
            negation: raw
                .negation
                .into_iter()
                .map(String::into_boxed_str)
                .collect(),
            quantifiers: raw
                .quantifiers
                .into_iter()
                .map(String::into_boxed_str)
                .collect(),
            modals: raw.modals.into_iter().map(String::into_boxed_str).collect(),
            connectives: raw
                .connectives
                .into_iter()
                .map(String::into_boxed_str)
                .collect(),
        }
    }
}

// --- Raw (pre-validation) TOML shapes ---

#[derive(Debug, Deserialize)]
struct RawPackMeta {
    version: String,
    corpus_manifest_sha256: String,
    paired_manifest_sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawDeletionSpan {
    id: String,
    pattern: String,
    anchor: String,
    repair: String,
    source: String,
    #[serde(default)]
    notes: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawSubstitutionPair {
    id: String,
    pattern: String,
    replacement: String,
    repair: String,
    source: String,
    #[serde(default)]
    notes: Option<String>,
    #[serde(default)]
    declared_literal_tokens: Vec<String>,
    #[serde(default)]
    attested_replacement_tokens: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawRitualFrame {
    id: String,
    pattern: String,
    scope: String,
    repair: String,
    source: String,
    #[serde(default)]
    notes: Option<String>,
    #[serde(default)]
    train_machine_count: Option<u32>,
    #[serde(default)]
    train_human_count: Option<u32>,
    #[serde(default)]
    train_stock_count: Option<u32>,
    #[serde(default)]
    train_antislop_count: Option<u32>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawPreviewFrame {
    id: String,
    pattern: String,
    #[serde(default)]
    requires_header_lemma_coverage: bool,
    repair: String,
    source: String,
    #[serde(default)]
    notes: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawLvcPair {
    light_verb_base: String,
    nominalization: String,
    derived_verb: String,
    source: String,
    train_construction_count_machine: u32,
    train_construction_count_human: u32,
    train_verb_count_machine: u32,
    train_verb_count_human: u32,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawOutputFrequencyBand {
    frame: String,
    unit: String,
    observed_rate: f64,
    max: f64,
    sample_doc_count: u32,
    method: String,
    #[serde(default)]
    notes: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct RawGuardTokens {
    #[serde(default)]
    negation: Vec<String>,
    #[serde(default)]
    quantifiers: Vec<String>,
    #[serde(default)]
    modals: Vec<String>,
    #[serde(default)]
    connectives: Vec<String>,
}

/// The TOML shape of the pack as a whole. Deliberately **not**
/// `#[serde(deny_unknown_fields)]` — matches `dms.rs`'s precedent of
/// tolerating free-form top-level metadata (`mine_inventory_report_path`
/// etc.).
#[derive(Debug, Deserialize)]
struct RawPack {
    pack: RawPackMeta,
    #[serde(default)]
    closure_function_word_allowance: Vec<String>,
    #[serde(default)]
    deletion_spans: Vec<RawDeletionSpan>,
    #[serde(default)]
    substitution_pairs: Vec<RawSubstitutionPair>,
    #[serde(default)]
    ritual_frames: Vec<RawRitualFrame>,
    #[serde(default)]
    preview_frames: Vec<RawPreviewFrame>,
    #[serde(default)]
    lvc_pairs: Vec<RawLvcPair>,
    #[serde(default)]
    guard_tokens: RawGuardTokens,
    #[serde(default)]
    output_frequency_bands: Vec<RawOutputFrequencyBand>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal, hermetic, valid `inventory-v1`-shaped pack: one entry
    /// per family (deletion span, substitution pair, ritual frame, lvc
    /// pair), guard tokens, and no preview frames or output-frequency
    /// bands: every field this module's own structural checks care
    /// about, nothing from the real shipped pack.
    const MINIMAL_PACK: &str = r#"
        closure_function_word_allowance = ["a", "an", "the", "to", "of"]

        [pack]
        version = "inventory-v1"
        corpus_manifest_sha256 = "deadbeef"
        paired_manifest_sha256 = "deadbeef"

        [[deletion_spans]]
        id = "span.simply"
        pattern = '(?i)\bsimply\s+'
        anchor = "mid_sentence"
        repair = "delete"
        source = "seed"

        [[substitution_pairs]]
        id = "sub.prior_to"
        pattern = '(?i)\bprior to\b'
        replacement = "before"
        repair = "substitute"
        source = "seed"
        notes = "attested"
        attested_replacement_tokens = ["before"]

        [[ritual_frames]]
        id = "ritual.we_hope"
        pattern = "(?i)we hope (this|you)"
        scope = "sentence"
        repair = "delete"
        source = "seed"

        [[lvc_pairs]]
        light_verb_base = "perform"
        nominalization = "initialization"
        derived_verb = "initialize"
        source = "seed"
        train_construction_count_machine = 0
        train_construction_count_human = 0
        train_verb_count_machine = 1
        train_verb_count_human = 1

        [guard_tokens]
        negation = ["not", "never"]
        quantifiers = ["all", "some"]
        modals = ["can", "will"]
        connectives = ["because", "so"]
    "#;

    #[test]
    fn parse_sorts_every_family_by_its_key_regardless_of_file_order() {
        let toml = r#"
            [pack]
            version = "inventory-v1"
            corpus_manifest_sha256 = "x"
            paired_manifest_sha256 = "x"

            [[deletion_spans]]
            id = "span.z"
            pattern = "z"
            anchor = "mid_sentence"
            repair = "delete"
            source = "seed"

            [[deletion_spans]]
            id = "span.a"
            pattern = "a"
            anchor = "mid_sentence"
            repair = "delete"
            source = "seed"

            [[lvc_pairs]]
            light_verb_base = "perform"
            nominalization = "validation"
            derived_verb = "validate"
            source = "seed"
            train_construction_count_machine = 0
            train_construction_count_human = 0
            train_verb_count_machine = 0
            train_verb_count_human = 0

            [[lvc_pairs]]
            light_verb_base = "perform"
            nominalization = "assessment"
            derived_verb = "assess"
            source = "seed"
            train_construction_count_machine = 0
            train_construction_count_human = 0
            train_verb_count_machine = 0
            train_verb_count_human = 0
        "#;
        let pack = InventoryPack::parse(toml).unwrap();
        let ids: Vec<&str> = pack.deletion_spans().iter().map(|s| &*s.id).collect();
        assert_eq!(ids, vec!["span.a", "span.z"]);
        let noms: Vec<&str> = pack
            .lvc_pairs()
            .iter()
            .map(|p| &*p.nominalization)
            .collect();
        assert_eq!(noms, vec!["assessment", "validation"]);
    }

    #[test]
    fn parse_compiles_every_pattern_into_a_working_regex() {
        let pack = InventoryPack::parse(MINIMAL_PACK).unwrap();
        assert!(pack.deletion_spans()[0].pattern.is_match("simply do it"));
        assert!(pack.substitution_pairs()[0].pattern.is_match("prior to"));
        assert!(pack.ritual_frames()[0].pattern.is_match("we hope this"));
    }

    #[test]
    fn parse_rejects_invalid_regex() {
        let toml = r#"
            [pack]
            version = "inventory-v1"
            corpus_manifest_sha256 = "x"
            paired_manifest_sha256 = "x"

            [[deletion_spans]]
            id = "span.bad"
            pattern = "("
            anchor = "mid_sentence"
            repair = "delete"
            source = "seed"
        "#;
        let err = InventoryPack::parse(toml).unwrap_err();
        assert!(matches!(err, PackError::InventoryInvalidPattern { .. }));
    }

    #[test]
    fn parse_rejects_unknown_anchor() {
        let toml = r#"
            [pack]
            version = "inventory-v1"
            corpus_manifest_sha256 = "x"
            paired_manifest_sha256 = "x"

            [[deletion_spans]]
            id = "span.bad"
            pattern = "x"
            anchor = "somewhere"
            repair = "delete"
            source = "seed"
        "#;
        let err = InventoryPack::parse(toml).unwrap_err();
        assert!(matches!(err, PackError::InventoryUnknownAnchor { .. }));
    }

    #[test]
    fn parse_rejects_unknown_repair_kind_on_deletion_span() {
        let toml = r#"
            [pack]
            version = "inventory-v1"
            corpus_manifest_sha256 = "x"
            paired_manifest_sha256 = "x"

            [[deletion_spans]]
            id = "span.bad"
            pattern = "x"
            anchor = "mid_sentence"
            repair = "rewrite"
            source = "seed"
        "#;
        let err = InventoryPack::parse(toml).unwrap_err();
        assert!(matches!(err, PackError::InventoryUnknownRepairKind { .. }));
    }

    #[test]
    fn parse_rejects_substitution_pair_repair_other_than_substitute() {
        let toml = r#"
            [pack]
            version = "inventory-v1"
            corpus_manifest_sha256 = "x"
            paired_manifest_sha256 = "x"

            [[substitution_pairs]]
            id = "sub.bad"
            pattern = "x"
            replacement = "y"
            repair = "diagnostic_only"
            source = "seed"
        "#;
        let err = InventoryPack::parse(toml).unwrap_err();
        assert!(matches!(
            err,
            PackError::InventoryInvalidRepairForFamily { .. }
        ));
    }

    #[test]
    fn parse_rejects_unknown_frequency_unit() {
        let toml = r#"
            [pack]
            version = "inventory-v1"
            corpus_manifest_sha256 = "x"
            paired_manifest_sha256 = "x"

            [[output_frequency_bands]]
            frame = "covers"
            unit = "per_day"
            observed_rate = 0.1
            max = 0.3
            sample_doc_count = 10
            method = "test"
        "#;
        let err = InventoryPack::parse(toml).unwrap_err();
        assert!(matches!(
            err,
            PackError::InventoryUnknownFrequencyUnit { .. }
        ));
    }

    #[test]
    fn parse_rejects_duplicate_id_across_families() {
        let toml = r#"
            [pack]
            version = "inventory-v1"
            corpus_manifest_sha256 = "x"
            paired_manifest_sha256 = "x"

            [[deletion_spans]]
            id = "dup.id"
            pattern = "x"
            anchor = "mid_sentence"
            repair = "delete"
            source = "seed"

            [[ritual_frames]]
            id = "dup.id"
            pattern = "y"
            scope = "sentence"
            repair = "delete"
            source = "seed"
        "#;
        let err = InventoryPack::parse(toml).unwrap_err();
        assert!(matches!(err, PackError::InventoryDuplicateId { .. }));
    }

    #[test]
    fn parse_rejects_duplicate_lvc_nominalization() {
        let toml = r#"
            [pack]
            version = "inventory-v1"
            corpus_manifest_sha256 = "x"
            paired_manifest_sha256 = "x"

            [[lvc_pairs]]
            light_verb_base = "perform"
            nominalization = "validation"
            derived_verb = "validate"
            source = "seed"
            train_construction_count_machine = 0
            train_construction_count_human = 0
            train_verb_count_machine = 0
            train_verb_count_human = 0

            [[lvc_pairs]]
            light_verb_base = "conduct"
            nominalization = "validation"
            derived_verb = "validate"
            source = "seed"
            train_construction_count_machine = 0
            train_construction_count_human = 0
            train_verb_count_machine = 0
            train_verb_count_human = 0
        "#;
        let err = InventoryPack::parse(toml).unwrap_err();
        assert!(matches!(
            err,
            PackError::InventoryDuplicateLvcNominalization { .. }
        ));
    }

    #[test]
    fn parse_rejects_lvc_pair_not_resolvable_in_derivational_lexicon() {
        let toml = r#"
            [pack]
            version = "inventory-v1"
            corpus_manifest_sha256 = "x"
            paired_manifest_sha256 = "x"

            [[lvc_pairs]]
            light_verb_base = "perform"
            nominalization = "not_a_real_nominalization"
            derived_verb = "nope"
            source = "seed"
            train_construction_count_machine = 0
            train_construction_count_human = 0
            train_verb_count_machine = 0
            train_verb_count_human = 0
        "#;
        let err = InventoryPack::parse(toml).unwrap_err();
        assert!(matches!(
            err,
            PackError::InventoryUnresolvableLvcPair { .. }
        ));
    }

    #[test]
    fn parse_rejects_lvc_pair_with_unknown_light_verb_base() {
        let toml = r#"
            [pack]
            version = "inventory-v1"
            corpus_manifest_sha256 = "x"
            paired_manifest_sha256 = "x"

            [[lvc_pairs]]
            light_verb_base = "performs"
            nominalization = "initialization"
            derived_verb = "initialize"
            source = "seed"
            train_construction_count_machine = 0
            train_construction_count_human = 0
            train_verb_count_machine = 0
            train_verb_count_human = 0
        "#;
        let err = InventoryPack::parse(toml).unwrap_err();
        assert!(matches!(
            err,
            PackError::InventoryUnknownLightVerbBase { .. }
        ));
    }

    #[test]
    fn parse_is_deterministic() {
        let a = InventoryPack::parse(MINIMAL_PACK).unwrap();
        let b = InventoryPack::parse(MINIMAL_PACK).unwrap();
        let ids = |p: &InventoryPack| -> Vec<String> {
            p.deletion_spans()
                .iter()
                .map(|s| s.id.to_string())
                .chain(p.substitution_pairs().iter().map(|s| s.id.to_string()))
                .chain(p.ritual_frames().iter().map(|s| s.id.to_string()))
                .chain(p.lvc_pairs().iter().map(|s| s.nominalization.to_string()))
                .collect()
        };
        assert_eq!(ids(&a), ids(&b));
        let probe = "simply do it";
        assert_eq!(
            a.deletion_spans()[0].pattern.is_match(probe),
            b.deletion_spans()[0].pattern.is_match(probe)
        );
    }

    #[test]
    fn lvc_lexicon_reflects_every_licensed_pair() {
        let pack = InventoryPack::parse(MINIMAL_PACK).unwrap();
        assert_eq!(
            pack.lvc_lexicon().get("initialization").map(|v| &**v),
            Some("initialize")
        );
        assert_eq!(pack.lvc_lexicon().len(), pack.lvc_pairs().len());
    }

    #[test]
    fn guard_tokens_parses_all_four_closed_classes() {
        let pack = InventoryPack::parse(MINIMAL_PACK).unwrap();
        let guard = pack.guard_tokens();
        assert_eq!(&*guard.negation[0], "not");
        assert_eq!(&*guard.quantifiers[0], "all");
        assert_eq!(&*guard.modals[0], "can");
        assert_eq!(&*guard.connectives[0], "because");
    }

    #[test]
    fn preview_frames_defaults_to_empty_when_section_absent() {
        let pack = InventoryPack::parse(MINIMAL_PACK).unwrap();
        assert!(pack.preview_frames().is_empty());
    }

    #[test]
    fn closure_function_word_allowance_defaults_to_empty_when_absent() {
        let toml = r#"
            [pack]
            version = "inventory-v1"
            corpus_manifest_sha256 = "x"
            paired_manifest_sha256 = "x"
        "#;
        let pack = InventoryPack::parse(toml).unwrap();
        assert!(pack.closure_function_word_allowance().is_empty());
    }
}
