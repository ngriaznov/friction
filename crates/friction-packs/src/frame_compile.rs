//! Compiles the adjudicated frame-rules buckets into the runtime rule
//! program, applying the whole rejection gauntlet.
//!
//! Only the four fenced buckets (`ship`, `flip`, `surface`, `pilot`)
//! are ever offered to this compiler, and it still rejects individual
//! rules freely: every exclusion is recorded in the [`CompileReport`]
//! with its [`Reject`] reason, so the derived pack's contents are fully
//! explainable from the source TOML alone. An unparseable pattern in a
//! fenced bucket fails the whole compile instead — that never
//! represents an evidence decision, only a corrupted artifact. (A
//! duplicated id is *not* structural: the shipped surface bucket
//! carries a handful of duplicated rows, so the first occurrence wins
//! and later ones soft-reject.)
//!
//! # The gauntlet, in check order
//!
//! 1. **Parseability** — pattern and target must parse (hard failure).
//! 2. **Guard shape** — a guard whose target is not `"="` is rejected.
//! 3. **Defined classes** — a pattern class position naming a class the
//!    set does not define can never match; rejected.
//! 4. **Authored target** — `t="REVIEW"` and `VB[?]` placeholders mark
//!    targets that were never authored; the rule demotes to
//!    report-only (it still surfaces findings, it just never edits).
//! 5. **Shadowing** — a surface rule whose pattern is an inflection of
//!    its own compiling parent's pattern is dead weight at runtime
//!    (literal matching is lemma-level, so the parent already covers
//!    it); rejected.
//! 6. **Anchor** — every compiling rule needs at least one obligatory
//!    literal token to anchor the scan; rejected otherwise.
//! 7. **Attestation** — every static content word a rewrite target can
//!    emit must clear the human-corpus frequency floor
//!    ([`ATTESTATION_FLOOR_PER_MILLION`]) or be in the declared
//!    function-word set. Inflected realizations are exempt (their
//!    surface comes from the inflection tables); the lemma being
//!    realized is not.
//! 8. **Closure** — a rewrite/delete target that can emit another
//!    compiling rewrite/delete rule's anchor sequence would re-match on
//!    a second pass; rejected. Guard sources are exempt: a guard
//!    produces no output, so re-matching one is a no-op, and the
//!    engine's third-pass canary proves idempotence end-to-end.
//!
//! # Target realization
//!
//! A rewrite whose pattern and target both open with a literal token
//! compiles that leading target literal as an *agreeing inflection*
//! ([`CTpl::Inflect`]): literal pattern tokens match at lemma level, so
//! "notify X" matches "notified X", and a plain literal "tell" would
//! break tense — the runtime realizes "tell" in the matched token's
//! form ("told") through the inflection tables, falling back to the
//! bare lemma when no inflection applies.

use std::collections::{BTreeMap, BTreeSet};

use serde::Deserialize;

use crate::frame_rules::{
    Clitic, FrameRule, FrameRuleSet, PatElem, PilotRule, RuleKind, Slot, Tag, Target, TplElem,
    parse_pattern, parse_target,
};

/// The human-corpus frequency floor (occurrences per million tokens) a
/// static target word must clear when it is not in the declared
/// function-word set.
pub const ATTESTATION_FLOOR_PER_MILLION: f64 = 20.0;

/// A compiled rule's operation kind (the source `r`/`d`/`g` letters
/// plus the compile-time demotion target).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompiledKind {
    /// Rewrite the match through the template.
    Rewrite,
    /// Delete the match.
    Delete,
    /// Protect the match from all edits this pass.
    Guard,
    /// Emit a suggestion finding only — never edit. Flip-bucket guards
    /// and unauthored-target rules land here.
    Report,
}

/// One compiled pattern element (interned, class-resolved).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CPat {
    /// Lemma-matched literal, as an interner id.
    Lit(u32),
    /// Bare tag position.
    Tag(Tag),
    /// Tag position restricted to a class's members.
    Class {
        /// Required tag.
        tag: Tag,
        /// Index into [`CompiledPack::classes`].
        class: u16,
    },
    /// Content slot (captures for target copies).
    Slot(Slot),
    /// Match must start its sentence.
    SentStart,
    /// Match must end its sentence.
    SentEnd,
    /// Contraction clitic.
    Clitic(Clitic),
    /// Optional single element.
    Opt(Box<Self>),
    /// Alternation group.
    Group {
        /// Alternatives in written order.
        alts: Vec<Vec<Self>>,
        /// Whether the group may match empty.
        optional: bool,
    },
}

/// One compiled template element.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CTpl {
    /// Emit this interned literal.
    Lit(u32),
    /// Copy a slot's captured bytes.
    Slot(Slot),
    /// Copy the token matched at the pattern element with this tag.
    TagCopy(Tag),
    /// Realize the interned lemma in the form of the token matched at
    /// pattern element `agree_elem`, falling back to the bare lemma.
    Inflect {
        /// Interner id of the lemma to realize.
        lemma: u32,
        /// Pattern element index whose matched token supplies the form.
        agree_elem: u8,
    },
    /// Realize the interned lemma in this tag's own form (no agreeing
    /// pattern element exists).
    InflectForm {
        /// Interner id of the lemma to realize.
        lemma: u32,
        /// The form to realize.
        tag: Tag,
    },
    /// Realize the class member the match contains, in this tag's form;
    /// first member if the match contains none.
    ClassRealize {
        /// Index into [`CompiledPack::classes`].
        class: u16,
        /// The form to realize.
        tag: Tag,
    },
    /// Copy the token matched by the pattern's alternation group with
    /// this ordinal (0-based, in pattern order).
    AltCopy(u8),
    /// Emit this clitic.
    Clitic(Clitic),
}

/// One rule as compiled into the pack.
#[derive(Debug, Clone, PartialEq)]
pub struct CompiledRule {
    /// The source rule id (pilot rules get `pilot.<slug>`).
    pub id: String,
    /// Operation kind after compile-time demotions.
    pub kind: CompiledKind,
    /// Compiled pattern.
    pub pattern: Vec<CPat>,
    /// Compiled template (empty for delete/guard/report).
    pub template: Vec<CTpl>,
    /// Element index range of the anchor run within `pattern`.
    pub anchor_elems: (u16, u16),
    /// The anchor's interned lemma-id sequence.
    pub anchor_ids: Vec<u32>,
    /// Conflict tie-break weight (higher wins after leftmost-longest).
    pub support: u32,
    /// Whether literals match lowercased surface forms instead of
    /// lemmas (pilot rules — their phrases are inflected surfaces).
    pub surface_match: bool,
    /// Measured machine per-million rate (explain output).
    pub m_pm: f64,
    /// Measured human per-million rate (explain output).
    pub h_pm: f64,
}

/// The compiled pack, pre-serialization.
#[derive(Debug, Clone, PartialEq)]
pub struct CompiledPack {
    /// Sorted, deduplicated word table; ids are indices.
    pub interner: Vec<String>,
    /// Class member lists: `classes[i]` is a list of members, each a
    /// sequence of interner ids (multi-word members were hyphenated in
    /// the source and match as phrases).
    pub classes: Vec<Vec<Vec<u32>>>,
    /// Class names, parallel to `classes` (explain/report output).
    pub class_names: Vec<String>,
    /// The compiled rules, in source order (ship, flip, surface,
    /// pilot), which is also rule-id-stable order for conflict ties.
    pub rules: Vec<CompiledRule>,
}

/// Why a rule was excluded from the pack.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum Reject {
    /// The pattern references a class the set does not define.
    #[error("pattern references undefined class {0:?}")]
    UndefinedClass(String),
    /// A guard rule whose target is not `"="`.
    #[error("guard target must be \"=\", found {0:?}")]
    GuardTargetNotEquals(String),
    /// The pattern has no obligatory literal token to anchor on.
    #[error("no obligatory literal token to anchor the scan")]
    ZeroLiteralAnchor,
    /// A static target word failed human-corpus attestation.
    #[error("target word {word:?} is unattested ({per_million:.1}/M < {floor:.0}/M)")]
    AttestationFailed {
        /// The failing word.
        word: String,
        /// Its measured human-corpus rate.
        per_million: f64,
        /// The floor it failed.
        floor: f64,
    },
    /// The target can emit another compiling rule's anchor.
    #[error("target can emit the anchor of rule {other:?}")]
    ClosureViolation {
        /// The rule whose anchor the target contains.
        other: String,
    },
    /// A surface rule fully covered by its compiling parent's
    /// lemma-level match.
    #[error("shadowed by compiling parent rule {parent:?}")]
    ShadowedByLemmaRule {
        /// The parent rule id.
        parent: String,
    },
    /// A later row reusing an earlier row's id (first occurrence wins).
    #[error("duplicate of an earlier rule id")]
    DuplicateId,
}

/// The compile outcome: what shipped, what demoted, what fell.
#[derive(Debug, Clone, Default)]
pub struct CompileReport {
    /// Rules compiled as written (id, kind).
    pub compiled: Vec<(String, CompiledKind)>,
    /// Rules demoted to report-only, with the reason text.
    pub demoted: Vec<(String, String)>,
    /// Rules excluded entirely, with reasons.
    pub rejected: Vec<(String, Reject)>,
}

/// Structural failures that abort the whole compile.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum FrameCompileError {
    /// A fenced rule's pattern or target failed to parse.
    #[error("rule {id}: {message}")]
    Unparseable {
        /// The offending rule id.
        id: String,
        /// The parse error text.
        message: String,
    },
    /// The attestation source (DMS human stream TOML) failed to parse.
    #[error("human-rate source did not parse: {0}")]
    RateSource(String),
}

/// Builds the human-corpus word-rate table the attestation fence needs
/// from a `dms-index-v1` TOML's shared vocabulary and human stream.
///
/// Deriving the rates from that committed artifact (rather than from a
/// live corpus walk) keeps the compile a pure function of in-repo
/// bytes, so the pack drift test can re-run it byte-identically.
///
/// # Errors
/// Returns [`FrameCompileError::RateSource`] if the TOML lacks the
/// `[vocab]`/`[streams.human]` shape or the id stream is malformed.
pub fn human_rates_from_dms_toml(
    toml_text: &str,
) -> Result<BTreeMap<String, f64>, FrameCompileError> {
    #[derive(Deserialize)]
    struct RawVocab {
        tokens: Vec<String>,
    }
    #[derive(Deserialize)]
    struct RawStream {
        ids: String,
    }
    #[derive(Deserialize)]
    struct RawStreams {
        human: RawStream,
    }
    #[derive(Deserialize)]
    struct RawPack {
        vocab: RawVocab,
        streams: RawStreams,
    }
    let raw: RawPack =
        toml::from_str(toml_text).map_err(|e| FrameCompileError::RateSource(e.to_string()))?;
    let mut counts: BTreeMap<u32, u64> = BTreeMap::new();
    let mut total: u64 = 0;
    for id_text in raw.streams.human.ids.split(',') {
        let id: u32 = id_text
            .trim()
            .parse()
            .map_err(|_| FrameCompileError::RateSource(format!("bad stream id {id_text:?}")))?;
        *counts.entry(id).or_default() += 1;
        total += 1;
    }
    if total == 0 {
        return Err(FrameCompileError::RateSource("empty human stream".into()));
    }
    #[expect(
        clippy::cast_precision_loss,
        reason = "corpus counts are far below 2^52"
    )]
    let rate = |count: u64| count as f64 / total as f64 * 1_000_000.0;
    let mut rates = BTreeMap::new();
    for (id, count) in counts {
        if let Some(token) = raw.vocab.tokens.get(id as usize) {
            // Duplicate vocab texts resolve by accumulation: the rate of
            // a word is the rate of all ids spelling it.
            *rates.entry(token.to_lowercase()).or_insert(0.0) += rate(count);
        }
    }
    Ok(rates)
}

/// Compiles the fenced buckets of `set` against the attestation table.
///
/// # Errors
/// Returns [`FrameCompileError`] only for structural inconsistencies
/// (duplicate ids, unparseable fenced rules); evidence-based exclusions
/// land in the report instead.
pub fn compile(
    set: &FrameRuleSet,
    human_rates: &BTreeMap<String, f64>,
) -> Result<(CompiledPack, CompileReport), FrameCompileError> {
    let mut report = CompileReport::default();
    let mut interner = Interner {
        function_words: set
            .function_words
            .words
            .iter()
            .map(|word| word.to_lowercase())
            .collect(),
        ..Interner::default()
    };
    let mut classes = ClassTable::new(&set.classes, &mut interner);

    let drafts = collect_drafts(set, &mut report)?;
    let ship_parents: BTreeSet<&str> = set.rules_ship.iter().map(|r| r.id.as_str()).collect();

    // Individual fences. Each draft either compiles, demotes, or falls;
    // the closure fence needs every survivor's anchor, so anchors are
    // resolved first and closure runs as a second pass.
    let mut survivors: Vec<CompiledRule> = Vec::new();
    for draft in drafts {
        match compile_one(
            draft,
            &mut interner,
            &mut classes,
            &ship_parents,
            human_rates,
        ) {
            Outcome::Compiled(rule, demotion) => {
                match demotion {
                    Some(reason) => report.demoted.push((rule.id.clone(), reason)),
                    None => report.compiled.push((rule.id.clone(), rule.kind)),
                }
                survivors.push(rule);
            }
            Outcome::Rejected(id, reject) => report.rejected.push((id, reject)),
        }
    }

    // Closure fence: no rewrite/delete target may be able to emit any
    // compiling rewrite/delete rule's anchor sequence (guards produce
    // no output and report rules never edit — both exempt, see the
    // module docs).
    let anchors: Vec<(String, Vec<u32>)> = survivors
        .iter()
        .filter(|r| matches!(r.kind, CompiledKind::Rewrite | CompiledKind::Delete))
        .map(|r| (r.id.clone(), r.anchor_ids.clone()))
        .collect();
    let mut closed: Vec<CompiledRule> = Vec::new();
    for rule in survivors {
        if matches!(rule.kind, CompiledKind::Rewrite) {
            let emitted = static_emission_runs(&rule.template);
            let violation = anchors.iter().find(|(other_id, anchor)| {
                !anchor.is_empty()
                    && emitted
                        .iter()
                        .any(|run| run.windows(anchor.len()).any(|w| w == anchor.as_slice()))
                    && *other_id != rule.id
            });
            if let Some((other_id, _)) = violation {
                report.compiled.retain(|(id, _)| id != &rule.id);
                report.demoted.retain(|(id, _)| id != &rule.id);
                report.rejected.push((
                    rule.id.clone(),
                    Reject::ClosureViolation {
                        other: other_id.clone(),
                    },
                ));
                continue;
            }
        }
        closed.push(rule);
    }

    Ok((
        CompiledPack {
            interner: interner.words,
            classes: classes.members,
            class_names: classes.names,
            rules: closed,
        },
        report,
    ))
}

/// Parses every fenced rule into a draft, soft-rejecting duplicated
/// ids (first occurrence wins — the shipped surface bucket carries a
/// handful of duplicated rows). A parse failure here is structural:
/// the fenced buckets were verified parseable when adjudicated.
fn collect_drafts(
    set: &FrameRuleSet,
    report: &mut CompileReport,
) -> Result<Vec<Draft>, FrameCompileError> {
    let mut seen_ids = BTreeSet::new();
    let mut drafts: Vec<Draft> = Vec::new();
    for (bucket, rule) in fenced_rules(set) {
        if !seen_ids.insert(rule.id.clone()) {
            report.rejected.push((rule.id.clone(), Reject::DuplicateId));
            continue;
        }
        let pattern = parse_pattern(&rule.p).map_err(|e| FrameCompileError::Unparseable {
            id: rule.id.clone(),
            message: e.to_string(),
        })?;
        // The unauthored-target placeholder is uppercase in the source
        // (quoted or bare) and must never reach the parser or a splice.
        let target = if rule.t.trim_matches('"') == "REVIEW" {
            Target::Template(Vec::new())
        } else {
            parse_target(&rule.t).map_err(|e| FrameCompileError::Unparseable {
                id: rule.id.clone(),
                message: e.to_string(),
            })?
        };
        drafts.push(Draft {
            id: rule.id.clone(),
            bucket,
            kind: RuleKind::parse(&rule.k).expect("kinds validated at load"),
            raw_target: rule.t.clone(),
            pattern,
            target,
            support: support_weight(rule.m_pm),
            surface_match: false,
            m_pm: rule.m_pm,
            h_pm: rule.h_pm,
        });
    }
    for rule in &set.rules_pilot {
        let draft = pilot_draft(rule);
        if !seen_ids.insert(draft.id.clone()) {
            report.rejected.push((draft.id, Reject::DuplicateId));
            continue;
        }
        drafts.push(draft);
    }
    Ok(drafts)
}

/// A rule mid-compilation.
struct Draft {
    id: String,
    bucket: Bucket,
    kind: RuleKind,
    raw_target: String,
    pattern: Vec<PatElem>,
    target: Target,
    support: u32,
    surface_match: bool,
    m_pm: f64,
    h_pm: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Bucket {
    Ship,
    Flip,
    Surface,
    Pilot,
}

enum Outcome {
    /// Compiled, possibly demoted to report-only (reason attached).
    Compiled(CompiledRule, Option<String>),
    Rejected(String, Reject),
}

/// The fenced knowledge buckets in fence order.
fn fenced_rules(set: &FrameRuleSet) -> impl Iterator<Item = (Bucket, &FrameRule)> {
    set.rules_ship
        .iter()
        .map(|rule| (Bucket::Ship, rule))
        .chain(set.rules_flip.iter().map(|rule| (Bucket::Flip, rule)))
        .chain(set.rules_surface.iter().map(|rule| (Bucket::Surface, rule)))
}

/// Conflict tie-break weight from a measured machine rate.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "rates are non-negative and far below u32::MAX / 10"
)]
fn support_weight(m_pm: f64) -> u32 {
    (m_pm * 10.0).round() as u32
}

/// Builds a pilot rule's draft: surface-matched literal phrase, target
/// pre-inflected by the miner, id slugged from the phrase.
fn pilot_draft(rule: &PilotRule) -> Draft {
    let slug: String = rule
        .p
        .split_whitespace()
        .collect::<Vec<_>>()
        .join("-")
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '-')
        .collect();
    let pattern = rule
        .p
        .split_whitespace()
        .map(|w| PatElem::Lit(w.to_lowercase()))
        .collect();
    let target = if rule.t.is_empty() {
        Target::Delete
    } else {
        Target::Template(
            rule.t
                .split_whitespace()
                .map(|w| TplElem::Lit(w.to_lowercase()))
                .collect(),
        )
    };
    Draft {
        id: format!("pilot.{slug}"),
        bucket: Bucket::Pilot,
        kind: RuleKind::parse(&rule.k).expect("kinds validated at load"),
        raw_target: rule.t.clone(),
        pattern,
        target,
        support: rule.support * 10,
        surface_match: true,
        m_pm: 0.0,
        h_pm: 0.0,
    }
}

/// Runs every per-rule fence except closure.
fn compile_one(
    draft: Draft,
    interner: &mut Interner,
    classes: &mut ClassTable,
    ship_parents: &BTreeSet<&str>,
    human_rates: &BTreeMap<String, f64>,
) -> Outcome {
    // Guard shape: a fenced guard must protect, not rewrite.
    if draft.kind == RuleKind::Guard
        && draft.bucket != Bucket::Flip
        && draft.target != Target::Guard
    {
        return Outcome::Rejected(draft.id, Reject::GuardTargetNotEquals(draft.raw_target));
    }

    // Defined classes (pattern side).
    if let Some(name) = first_undefined_class(&draft.pattern, classes) {
        return Outcome::Rejected(draft.id, Reject::UndefinedClass(name));
    }

    // Shadowing: surface rules whose compiling parent already covers
    // them at lemma level.
    if draft.bucket == Bucket::Surface
        && let Some(parent) = draft
            .id
            .split_once("::")
            .map(|(parent, _)| parent.to_string())
        && ship_parents.contains(parent.as_str())
    {
        return Outcome::Rejected(draft.id, Reject::ShadowedByLemmaRule { parent });
    }

    // Anchor: the longest run of consecutive obligatory literals.
    let Some((anchor_start, anchor_len)) = anchor_run(&draft.pattern) else {
        return Outcome::Rejected(draft.id, Reject::ZeroLiteralAnchor);
    };

    // Demotions to report-only: flip-bucket guards (measured
    // machine-tilted — protecting them would shelter machine-leaning
    // text) and unauthored targets.
    let demotion = if draft.bucket == Bucket::Flip {
        Some("guard measured machine-tilted; reports instead of protecting".to_string())
    } else if draft.raw_target.trim_matches('"') == "REVIEW" {
        Some("target was never authored (REVIEW placeholder)".to_string())
    } else if template_has_placeholder(&draft.target) {
        Some("target realization was never authored (`?` placeholder)".to_string())
    } else {
        None
    };

    let kind = if demotion.is_some() {
        CompiledKind::Report
    } else {
        match draft.kind {
            RuleKind::Rewrite => CompiledKind::Rewrite,
            RuleKind::Delete => CompiledKind::Delete,
            RuleKind::Guard => CompiledKind::Guard,
        }
    };

    // Compile the template (report/guard/delete rules have none), then
    // attest every static word it can emit.
    let template = if kind == CompiledKind::Rewrite {
        compile_template(&draft, interner, classes)
    } else {
        Vec::new()
    };
    if kind == CompiledKind::Rewrite
        && let Some(reject) = attest_template(&template, interner, classes, human_rates)
    {
        return Outcome::Rejected(draft.id, reject);
    }

    let pattern: Vec<CPat> = draft
        .pattern
        .iter()
        .map(|elem| compile_pat(elem, interner, classes))
        .collect();
    let anchor_ids = draft.pattern[anchor_start..anchor_start + anchor_len]
        .iter()
        .map(|elem| match elem {
            PatElem::Lit(text) => interner.intern(text),
            _ => unreachable!("anchor runs contain only literals"),
        })
        .collect();

    #[expect(clippy::cast_possible_truncation, reason = "patterns are tokens-long")]
    let rule = CompiledRule {
        id: draft.id,
        kind,
        pattern,
        template,
        anchor_elems: (anchor_start as u16, anchor_len as u16),
        anchor_ids,
        support: draft.support,
        surface_match: draft.surface_match,
        m_pm: draft.m_pm,
        h_pm: draft.h_pm,
    };
    Outcome::Compiled(rule, demotion)
}

/// Finds the longest run of consecutive obligatory literal elements;
/// returns `(start, len)`, leftmost winning ties.
fn anchor_run(pattern: &[PatElem]) -> Option<(usize, usize)> {
    let mut best: Option<(usize, usize)> = None;
    let mut run_start = 0;
    let consider = |start: usize, end: usize, best: &mut Option<(usize, usize)>| {
        let len = end - start;
        if len > 0 && best.is_none_or(|(_, best_len)| len > best_len) {
            *best = Some((start, len));
        }
    };
    for (i, elem) in pattern.iter().enumerate() {
        if !matches!(elem, PatElem::Lit(_)) {
            consider(run_start, i, &mut best);
            run_start = i + 1;
        }
    }
    consider(run_start, pattern.len(), &mut best);
    best
}

/// First class name referenced by the pattern that the set does not
/// define, if any.
fn first_undefined_class(pattern: &[PatElem], classes: &ClassTable) -> Option<String> {
    pattern.iter().find_map(|elem| match elem {
        PatElem::TagClass { class, .. } if classes.id_of(class).is_none() => Some(class.clone()),
        PatElem::Optional(inner) => first_undefined_class(std::slice::from_ref(inner), classes),
        PatElem::Group { alts, .. } => alts
            .iter()
            .find_map(|alt| first_undefined_class(alt, classes)),
        _ => None,
    })
}

/// Whether the target carries a `TAG[?]` placeholder.
fn template_has_placeholder(target: &Target) -> bool {
    match target {
        Target::Template(elems) => elems
            .iter()
            .any(|elem| matches!(elem, TplElem::TagArg { arg, .. } if arg == "?")),
        Target::Guard | Target::Delete => false,
    }
}

/// Compiles a rewrite draft's parsed target into template ops.
fn compile_template(draft: &Draft, interner: &mut Interner, classes: &ClassTable) -> Vec<CTpl> {
    let Target::Template(elems) = &draft.target else {
        return Vec::new();
    };
    // Sentence sentinels are positional constraints, not emissions —
    // they simply drop out of the template.
    let mut ops = Vec::new();
    let mut group_ordinal: u8 = 0;
    let leading_lit_agreement = leading_literal_agreement(draft);
    for (i, elem) in elems.iter().enumerate() {
        match elem {
            TplElem::SentStart | TplElem::SentEnd => {}
            TplElem::Lit(text) => {
                let id = interner.intern(text);
                if Some(i) == leading_lit_agreement.map(|(tpl_idx, _)| tpl_idx) {
                    let (_, pat_elem) = leading_lit_agreement.expect("checked Some");
                    ops.push(CTpl::Inflect {
                        lemma: id,
                        agree_elem: pat_elem,
                    });
                } else {
                    ops.push(CTpl::Lit(id));
                }
            }
            TplElem::Slot(slot) => ops.push(CTpl::Slot(*slot)),
            TplElem::TagCopy(tag) => ops.push(CTpl::TagCopy(*tag)),
            TplElem::TagArg { tag, arg } => {
                if let Some(class) = classes.id_of(arg) {
                    ops.push(CTpl::ClassRealize { class, tag: *tag });
                } else {
                    // A lemma to realize. Agreement source: the
                    // pattern's BE element if it has one (finite verb
                    // agreement), else the tag's own form.
                    let lemma = interner.intern(arg);
                    match be_element(&draft.pattern) {
                        Some(pos) => ops.push(CTpl::Inflect {
                            lemma,
                            agree_elem: pos,
                        }),
                        None => ops.push(CTpl::InflectForm { lemma, tag: *tag }),
                    }
                }
            }
            TplElem::AltCopy(_) => {
                ops.push(CTpl::AltCopy(group_ordinal));
                group_ordinal += 1;
            }
            TplElem::Clitic(clitic) => ops.push(CTpl::Clitic(*clitic)),
        }
    }
    ops
}

/// If pattern and target both open with a literal (ignoring sentence
/// sentinels) and they differ, the target's leading literal realizes as
/// an inflection agreeing with the matched pattern token. Returns
/// `(template index, pattern element index)`.
fn leading_literal_agreement(draft: &Draft) -> Option<(usize, u8)> {
    if draft.surface_match {
        return None;
    }
    let Target::Template(elems) = &draft.target else {
        return None;
    };
    let not_sentinel_pat = |elem: &&PatElem| !matches!(elem, PatElem::SentStart | PatElem::SentEnd);
    let not_sentinel_tpl = |elem: &&TplElem| !matches!(elem, TplElem::SentStart | TplElem::SentEnd);
    let (pat_idx, pat_first) = draft
        .pattern
        .iter()
        .enumerate()
        .find(|(_, elem)| not_sentinel_pat(elem))?;
    let (tpl_idx, tpl_first) = elems
        .iter()
        .enumerate()
        .find(|(_, e)| not_sentinel_tpl(e))?;
    match (pat_first, tpl_first) {
        (PatElem::Lit(source), TplElem::Lit(target)) if source != target => {
            u8::try_from(pat_idx).ok().map(|pat| (tpl_idx, pat))
        }
        _ => None,
    }
}

/// Position of the pattern's `BE` element, if it has exactly one.
fn be_element(pattern: &[PatElem]) -> Option<u8> {
    let mut positions = pattern
        .iter()
        .enumerate()
        .filter(|(_, elem)| matches!(elem, PatElem::Tag(Tag::Be)));
    let (pos, _) = positions.next()?;
    if positions.next().is_some() {
        return None;
    }
    u8::try_from(pos).ok()
}

/// Attests every static word the template can emit. Slot/tag/group
/// copies re-emit input (exempt); inflected realizations attest their
/// lemma (the surface form comes from the tables).
fn attest_template(
    template: &[CTpl],
    interner: &Interner,
    classes: &ClassTable,
    human_rates: &BTreeMap<String, f64>,
) -> Option<Reject> {
    let function_words = &interner.function_words;
    let check = |id: u32| {
        let word = &interner.words[id as usize];
        if function_words.contains(word.as_str()) {
            return None;
        }
        let per_million = human_rates.get(word).copied().unwrap_or(0.0);
        (per_million < ATTESTATION_FLOOR_PER_MILLION).then(|| Reject::AttestationFailed {
            word: word.clone(),
            per_million,
            floor: ATTESTATION_FLOOR_PER_MILLION,
        })
    };
    template.iter().find_map(|op| match op {
        CTpl::Lit(id) | CTpl::Inflect { lemma: id, .. } | CTpl::InflectForm { lemma: id, .. } => {
            check(*id)
        }
        CTpl::ClassRealize { class, .. } => classes.members[*class as usize]
            .iter()
            .flatten()
            .find_map(|id| check(*id)),
        CTpl::Slot(_) | CTpl::TagCopy(_) | CTpl::AltCopy(_) | CTpl::Clitic(_) => None,
    })
}

/// Compiles one parsed pattern element.
fn compile_pat(elem: &PatElem, interner: &mut Interner, classes: &mut ClassTable) -> CPat {
    match elem {
        PatElem::Lit(text) => CPat::Lit(interner.intern(text)),
        PatElem::Tag(tag) => CPat::Tag(*tag),
        PatElem::TagClass { tag, class } => CPat::Class {
            tag: *tag,
            class: classes.id_of(class).expect("checked by the class fence"),
        },
        PatElem::Slot(slot) => CPat::Slot(*slot),
        PatElem::SentStart => CPat::SentStart,
        PatElem::SentEnd => CPat::SentEnd,
        PatElem::Clitic(clitic) => CPat::Clitic(*clitic),
        PatElem::Optional(inner) => CPat::Opt(Box::new(compile_pat(inner, interner, classes))),
        PatElem::Group { alts, optional } => CPat::Group {
            alts: alts
                .iter()
                .map(|alt| {
                    alt.iter()
                        .map(|inner| compile_pat(inner, interner, classes))
                        .collect()
                })
                .collect(),
            optional: *optional,
        },
    }
}

/// The static lemma-id runs a template can emit: consecutive literal
/// emissions, broken by any input-derived element (which can be
/// anything, so it conservatively breaks adjacency rather than
/// wildcarding it). Inflected realizations contribute their lemma id —
/// the closure check compares lemma-level anchors, and the realization
/// of a lemma matches that lemma at lemma level.
fn static_emission_runs(template: &[CTpl]) -> Vec<Vec<u32>> {
    let mut runs = vec![Vec::new()];
    for op in template {
        match op {
            CTpl::Lit(id)
            | CTpl::Inflect { lemma: id, .. }
            | CTpl::InflectForm { lemma: id, .. } => {
                runs.last_mut().expect("runs starts non-empty").push(*id);
            }
            _ => runs.push(Vec::new()),
        }
    }
    runs.retain(|run| !run.is_empty());
    runs
}

/// The word interner. Ids are append-order (first use wins), which is
/// deterministic because compilation visits rules in fixed bucket
/// order; the serializer sorts the table and remaps every reference
/// once when writing the pack, so the on-disk interner supports binary
/// search while this build-side one stays cheap.
#[derive(Debug, Default)]
struct Interner {
    words: Vec<String>,
    index: BTreeMap<String, u32>,
    function_words: BTreeSet<String>,
}

impl Interner {
    fn intern(&mut self, word: &str) -> u32 {
        if let Some(&id) = self.index.get(word) {
            return id;
        }
        let id = u32::try_from(self.words.len()).expect("interner far below u32::MAX");
        self.words.push(word.to_string());
        self.index.insert(word.to_string(), id);
        id
    }
}

/// The class tables, members interned.
#[derive(Debug)]
struct ClassTable {
    names: Vec<String>,
    members: Vec<Vec<Vec<u32>>>,
}

impl ClassTable {
    fn new(classes: &BTreeMap<String, Vec<String>>, interner: &mut Interner) -> Self {
        let mut names = Vec::new();
        let mut members = Vec::new();
        for (name, words) in classes {
            names.push(name.clone());
            members.push(
                words
                    .iter()
                    .map(|word| {
                        word.to_lowercase()
                            .split('-')
                            .map(|part| interner.intern(part))
                            .collect()
                    })
                    .collect(),
            );
        }
        Self { names, members }
    }

    fn id_of(&self, name: &str) -> Option<u16> {
        self.names
            .iter()
            .position(|n| n == name)
            .map(|i| u16::try_from(i).expect("class count far below u16::MAX"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame_rules::FrameRuleSet;

    /// A rates table where everything common is attested.
    fn generous_rates(words: &[&str]) -> BTreeMap<String, f64> {
        words.iter().map(|w| ((*w).to_string(), 100.0)).collect()
    }

    fn tiny_set(rules_ship: &str) -> FrameRuleSet {
        let toml = format!(
            r#"
{rules_ship}
rules_flip = []
rules_surface = []
rules_pilot = []
rules_quarantine = []
rules_no_evidence = []
rules_staged_surface = []
[classes]
insight_n = ["insight", "takeaway"]
[function_words]
words = ["a", "an", "the", "to", "of"]
"#
        );
        FrameRuleSet::parse(&toml).expect("tiny set must parse")
    }

    /// The leading-literal rewrite compiles to an agreeing inflection,
    /// not a bare literal — lemma-level matching would otherwise break
    /// tense on inflected matches.
    #[test]
    fn leading_literal_target_compiles_as_agreeing_inflection() {
        let set = tiny_set(
            r#"rules_ship = [
{ id="vsub.utilize", p="\"utilize\" X", t="\"use\" X", k="r", m_pm=100.0, h_pm=10.0, v="CONFIRMED" },
]"#,
        );
        let (pack, report) =
            compile(&set, &generous_rates(&["use"])).expect("compile must succeed");
        assert_eq!(report.compiled.len(), 1);
        assert_eq!(pack.rules.len(), 1);
        let rule = &pack.rules[0];
        assert!(matches!(
            rule.template[0],
            CTpl::Inflect { agree_elem: 0, .. }
        ));
        assert!(matches!(rule.template[1], CTpl::Slot(Slot::X)));
    }

    /// An unattested target word rejects the rule with the measured
    /// rate in the reason.
    #[test]
    fn unattested_target_word_rejects_the_rule() {
        let set = tiny_set(
            r#"rules_ship = [
{ id="vsub.commence", p="\"commence\" X", t="\"begined\" X", k="r", m_pm=100.0, h_pm=0.0, v="CONFIRMED" },
]"#,
        );
        let (pack, report) = compile(&set, &generous_rates(&["use"])).expect("compile");
        assert!(pack.rules.is_empty());
        assert!(matches!(
            &report.rejected[0].1,
            Reject::AttestationFailed { word, .. } if word == "begined"
        ));
    }

    /// `REVIEW` placeholder targets demote to report-only instead of
    /// ever splicing the placeholder into text.
    #[test]
    fn review_placeholder_demotes_to_report_only() {
        let set = tiny_set(
            r#"rules_ship = [
{ id="col.seamless-integration", p="\"seamless\" \"integration\"", t="\"REVIEW\"", k="r", m_pm=50.0, h_pm=1.0, v="CONFIRMED" },
]"#,
        );
        let (pack, report) = compile(&set, &generous_rates(&[])).expect("compile");
        assert_eq!(pack.rules.len(), 1);
        assert_eq!(pack.rules[0].kind, CompiledKind::Report);
        assert_eq!(report.demoted.len(), 1);
        assert!(pack.rules[0].template.is_empty());
    }

    /// Undefined pattern classes reject; defined ones compile.
    #[test]
    fn undefined_pattern_class_rejects() {
        let set = tiny_set(
            r#"rules_ship = [
{ id="lst.key-points", p="\"key\" NNS[point-class]", t="NNS", k="r", m_pm=50.0, h_pm=1.0, v="CONFIRMED" },
{ id="met.insights", p="\"actionable\" NN[insight_n]", t="NNS[insight_n]", k="r", m_pm=50.0, h_pm=1.0, v="CONFIRMED" },
]"#,
        );
        let (pack, report) =
            compile(&set, &generous_rates(&["insight", "takeaway"])).expect("compile");
        assert_eq!(pack.rules.len(), 1, "only the defined-class rule survives");
        assert_eq!(pack.rules[0].id, "met.insights");
        assert!(matches!(
            &report.rejected[0].1,
            Reject::UndefinedClass(name) if name == "point-class"
        ));
    }

    /// A target able to emit another rule's anchor is a closure
    /// violation: pass two would re-match inside pass one's output.
    #[test]
    fn closure_violation_rejects_the_emitting_rule() {
        let set = tiny_set(
            r#"rules_ship = [
{ id="vsub.a", p="\"expedite\" X", t="\"hasten\" X", k="r", m_pm=100.0, h_pm=1.0, v="CONFIRMED" },
{ id="vsub.b", p="\"hasten\" X", t="\"speed\" X", k="r", m_pm=100.0, h_pm=1.0, v="CONFIRMED" },
]"#,
        );
        let (pack, report) = compile(&set, &generous_rates(&["hasten", "speed"])).expect("compile");
        assert_eq!(pack.rules.len(), 1);
        assert_eq!(pack.rules[0].id, "vsub.b");
        assert!(matches!(
            &report.rejected[0].1,
            Reject::ClosureViolation { other } if other == "vsub.b"
        ));
    }

    /// A duplicated id keeps its first row and soft-rejects the rest —
    /// the shipped surface bucket carries a handful of duplicated rows.
    #[test]
    fn duplicate_id_keeps_first_row_and_rejects_later_ones() {
        let set = tiny_set(
            r#"rules_ship = [
{ id="vsub.x", p="\"utilize\" X", t="\"use\" X", k="r", m_pm=100.0, h_pm=1.0, v="CONFIRMED" },
{ id="vsub.x", p="\"leverage\" X", t="\"use\" X", k="r", m_pm=100.0, h_pm=1.0, v="CONFIRMED" },
]"#,
        );
        let (pack, report) = compile(&set, &generous_rates(&["use"])).expect("compile");
        assert_eq!(pack.rules.len(), 1);
        assert_eq!(
            report.rejected,
            vec![("vsub.x".to_string(), Reject::DuplicateId)]
        );
    }

    /// The real artifact compiles: fences hold, and the report accounts
    /// for every fenced rule exactly once.
    #[test]
    fn shipped_rule_set_compiles_with_full_accounting() {
        let set = FrameRuleSet::parse(include_str!("../packs/frame-rules-v1.toml"))
            .expect("shipped rules parse");
        let rates = human_rates_from_dms_toml(include_str!("../packs/dms-index-v1.toml"))
            .expect("shipped dms index parses");
        let (pack, report) = compile(&set, &rates).expect("shipped set must compile");
        let fenced = set.rules_ship.len()
            + set.rules_flip.len()
            + set.rules_surface.len()
            + set.rules_pilot.len();
        assert_eq!(
            report.compiled.len() + report.demoted.len() + report.rejected.len(),
            fenced,
            "every fenced rule lands in exactly one report section"
        );
        assert_eq!(
            pack.rules.len(),
            report.compiled.len() + report.demoted.len(),
            "pack holds exactly the compiled + demoted rules"
        );
        assert!(
            pack.rules.iter().any(|r| r.kind == CompiledKind::Rewrite),
            "some rewrites must survive"
        );
        assert!(
            pack.rules.iter().any(|r| r.kind == CompiledKind::Guard),
            "ship guards must survive"
        );
        // The known-broken generated inflections must all have fallen
        // at the attestation fence.
        for word in ["begined", "buyed", "telled", "haves", "leted"] {
            assert!(
                !pack.interner.contains(&word.to_string())
                    || pack.rules.iter().all(|r| {
                        r.template.iter().all(|op| !matches!(op,
                            CTpl::Lit(id) | CTpl::Inflect { lemma: id, .. } | CTpl::InflectForm { lemma: id, .. }
                                if pack.interner[*id as usize] == word))
                    }),
                "broken form {word:?} must not be emittable"
            );
        }
    }

    /// Compilation is a pure function: two runs, identical output.
    #[test]
    fn compile_is_deterministic() {
        let set = FrameRuleSet::parse(include_str!("../packs/frame-rules-v1.toml"))
            .expect("shipped rules parse");
        let rates = human_rates_from_dms_toml(include_str!("../packs/dms-index-v1.toml"))
            .expect("shipped dms index parses");
        let (pack_a, _) = compile(&set, &rates).expect("compile a");
        let (pack_b, _) = compile(&set, &rates).expect("compile b");
        assert_eq!(pack_a, pack_b);
    }
}
