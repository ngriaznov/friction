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
//! 1. **Parseability**: pattern and target must parse (hard failure).
//! 2. **Guard shape** — a guard whose target is not `"="` is rejected.
//! 3. **Defined classes**: a pattern class position naming a class the
//!    set does not define can never match; rejected.
//! 4. **Authored target** — `t="REVIEW"` and `VB[?]` placeholders mark
//!    targets that were never authored. The rule demotes to
//!    report-only (it still surfaces findings, it just never edits).
//! 5. **Shadowing**: a surface rule whose pattern is an inflection of
//!    its own compiling parent's pattern is dead weight at runtime
//!    (literal matching is lemma-level, so the parent already covers
//!    it); rejected.
//! 6. **Anchor**: every compiling rule needs at least one obligatory
//!    literal token to anchor the scan; rejected otherwise.
//! 7. **Attestation** — every static content word a rewrite target can
//!    emit must clear the human-corpus frequency floor
//!    ([`ATTESTATION_FLOOR_PER_MILLION`]) or be in the declared
//!    function-word set. Inflected realizations are exempt (their
//!    surface comes from the inflection tables); the lemma being
//!    realized is not.
//! 8. **Closure**: a rewrite/delete target that can emit another
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
//! break tense: the runtime realizes "tell" in the matched token's form
//! ("told") through the inflection tables, falling back to the bare
//! lemma when no inflection applies.

use std::collections::{BTreeMap, BTreeSet};

use serde::Deserialize;

use crate::frame_rules::{
    Clitic, FrameRule, FrameRuleSet, PatElem, PilotRule, RuleKind, Slot, Tag, Target, TplElem,
    Verdict, classify, parse_pattern, parse_target,
};
use crate::human_evidence::HumanEvidencePack;

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
    /// Emit a suggestion finding only: never edit. Flip-bucket guards
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
    /// lemmas (pilot rules. Their phrases are inflected surfaces).
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

/// The corpus evidence the compile fences consult.
///
/// Derived from the committed `dms-index-v1` TOML so the compile stays a
/// pure function of in-repo bytes (the pack drift test re-runs it
/// byte-identically): per-word human occurrence counts for the
/// attestation floor, and both sides' token-id streams for the direction
/// fence's phrase counts (the machine side is every family stream
/// concatenated).
///
/// Optionally strengthened with a `corpus-tool human-evidence` pack
/// ([`Self::with_external`]): additional human-side unigram and
/// literal-probe counts mined from staged external human corpora. Every
/// accessor below pools the external pack in automatically when one is
/// attached, so every fence that reads this struct (the attestation
/// floor, the direction fence, the human-rate ceiling, the guard
/// direction check) is pooled for free, with no fence-specific special
/// case: `pooled = (dms_count + ext_count) / (dms_total + ext_total) *
/// 1_000_000`.
///
/// # Absence asymmetry
///
/// The external pack's two tables answer "absent" differently, and this
/// struct's accessors honor that difference rather than flattening it
/// (see [`crate::human_evidence`]'s own docs for why each table means
/// what it means): a unigram absent from the external pack is folded in
/// as a real `0` count *with* the external total still added to the
/// denominator (its absence is itself evidence, fewer than the
/// builder's five-occurrence floor). A phrase absent from the external
/// pack's probe table is left out of the pooled count *and* its total
/// entirely — that absence means the phrase was never one of
/// `frame-rules-en-v1.toml`'s own literal probes when the pack was built,
/// so nothing about it was ever measured, and pooling a `0` would be a
/// fabricated data point, not a real one.
///
/// With no external pack attached (or the shipped empty pack, whose
/// tables and total are all empty/zero), every accessor is numerically
/// identical to the dms-only computation — pooling a pack with nothing
/// in it changes nothing.
#[derive(Debug, Clone)]
pub struct CorpusEvidence {
    /// Lowercased word → DMS human-stream occurrence count (raw, not yet
    /// per-million — see [`Self::human_rate_per_million`] for the pooled
    /// per-million rate every fence actually reads).
    human_word_counts: BTreeMap<String, u64>,
    word_ids: BTreeMap<String, u32>,
    human_stream: Vec<u32>,
    machine_stream: Vec<u32>,
    external: Option<HumanEvidencePack>,
    external_machine: Option<HumanEvidencePack>,
}

impl CorpusEvidence {
    /// Parses a `dms-index-v1` TOML's shared vocabulary and streams.
    ///
    /// # Errors
    /// Returns [`FrameCompileError::RateSource`] if the TOML lacks the
    /// `[vocab]`/`[streams.*]` shape or an id stream is malformed.
    pub fn from_dms_toml(toml_text: &str) -> Result<Self, FrameCompileError> {
        #[derive(Deserialize)]
        struct RawVocab {
            tokens: Vec<String>,
        }
        #[derive(Deserialize, Default)]
        struct RawStream {
            ids: String,
        }
        #[derive(Deserialize)]
        struct RawStreams {
            human: RawStream,
            #[serde(default)]
            qwen: Option<RawStream>,
            #[serde(default)]
            gemma: Option<RawStream>,
            #[serde(default)]
            llama: Option<RawStream>,
            #[serde(default)]
            granite: Option<RawStream>,
            #[serde(default)]
            claude: Option<RawStream>,
        }
        #[derive(Deserialize)]
        struct RawPack {
            vocab: RawVocab,
            streams: RawStreams,
        }
        let raw: RawPack =
            toml::from_str(toml_text).map_err(|e| FrameCompileError::RateSource(e.to_string()))?;
        let parse_stream = |ids: &str| -> Result<Vec<u32>, FrameCompileError> {
            ids.split(',')
                .map(|t| {
                    t.trim()
                        .parse()
                        .map_err(|_| FrameCompileError::RateSource(format!("bad stream id {t:?}")))
                })
                .collect()
        };
        let human_stream = parse_stream(&raw.streams.human.ids)?;
        if human_stream.is_empty() {
            return Err(FrameCompileError::RateSource("empty human stream".into()));
        }
        let mut machine_stream = Vec::new();
        for family in [
            &raw.streams.qwen,
            &raw.streams.gemma,
            &raw.streams.llama,
            &raw.streams.granite,
            &raw.streams.claude,
        ]
        .into_iter()
        .flatten()
        {
            machine_stream.extend(parse_stream(&family.ids)?);
        }

        let mut counts: BTreeMap<u32, u64> = BTreeMap::new();
        for id in &human_stream {
            *counts.entry(*id).or_default() += 1;
        }
        let mut human_word_counts: BTreeMap<String, u64> = BTreeMap::new();
        let mut word_ids: BTreeMap<String, u32> = BTreeMap::new();
        for (index, token) in raw.vocab.tokens.iter().enumerate() {
            let id = u32::try_from(index).expect("vocab far below u32::MAX");
            let word = token.to_lowercase();
            // Duplicate vocab texts: counts accumulate; the id map keeps
            // the first (matching the runtime vocab's first-id-wins).
            if let Some(count) = counts.get(&id) {
                *human_word_counts.entry(word.clone()).or_insert(0) += count;
            }
            word_ids.entry(word).or_insert(id);
        }
        Ok(Self {
            human_word_counts,
            word_ids,
            human_stream,
            machine_stream,
            external: None,
            external_machine: None,
        })
    }

    /// Attaches a `corpus-tool human-evidence` pack, pooled into every
    /// accessor below from this point on (see this struct's own docs for
    /// the pooling formula and the two tables' absence asymmetry).
    #[must_use]
    pub fn with_external(mut self, external: HumanEvidencePack) -> Self {
        self.external = Some(external);
        self
    }

    /// Attaches the machine half of the review-register evidence pair
    /// (the `machine-evidence-v1` pack, built over the committed
    /// LLM-generated review documents). Consulted only through
    /// [`Self::review_pair_counts`]: never pooled into the human-side
    /// accessors above.
    #[must_use]
    pub fn with_external_machine(mut self, machine: HumanEvidencePack) -> Self {
        self.external_machine = Some(machine);
        self
    }

    /// The review-register evidence pair for one anchor phrase:
    /// `(machine_count, human_count, machine_total, human_total)` — the
    /// machine side from the generated-review pack, the human side from
    /// the external human pack's probe table.
    ///
    /// This is the second *register-matched* pair the two-sided fences
    /// may consult (both sides are review-register prose), with the DMS
    /// pair from [`Self::phrase_counts_register_matched`]. `None` unless
    /// both packs are attached, both are non-empty, and both probe
    /// tables measured the phrase — a pair with a missing side is no
    /// pair at all.
    #[must_use]
    pub fn review_pair_counts(&self, words: &[&str]) -> Option<(u64, u64, u64, u64)> {
        let machine = self.external_machine.as_ref()?;
        let human = self.external.as_ref()?;
        if machine.total_tokens() == 0 || human.total_tokens() == 0 {
            return None;
        }
        let m = machine.probe_count(words)?;
        let h = human.probe_count(words)?;
        Some((m, h, machine.total_tokens(), human.total_tokens()))
    }

    /// A minimal evidence table for tests: the given words attested at
    /// a generous rate, with empty streams (so the direction fence
    /// never trips) and no external pack attached.
    #[must_use]
    pub fn for_tests(attested: &[&str]) -> Self {
        Self {
            human_word_counts: attested.iter().map(|w| ((*w).to_string(), 1)).collect(),
            word_ids: BTreeMap::new(),
            human_stream: vec![0; 1],
            machine_stream: Vec::new(),
            external: None,
            external_machine: None,
        }
    }

    /// The pooled human-corpus per-million rate for one word: the DMS
    /// human stream's own count, plus the external pack's unigram count
    /// when one is attached, over the sum of both totals. See this
    /// struct's own docs for the pooling formula and why a unigram
    /// absent from the external pack still folds its total in (unlike a
    /// probe).
    #[must_use]
    pub fn human_rate_per_million(&self, word: &str) -> f64 {
        let word = word.to_lowercase();
        let dms_count = self.human_word_counts.get(&word).copied().unwrap_or(0);
        let ext = self.external.as_ref();
        let ext_count = ext.map_or(0, |pack| pack.unigram_count(&word));
        let ext_total = ext.map_or(0, HumanEvidencePack::total_tokens);
        pooled_rate_per_million(
            dms_count + ext_count,
            self.human_stream.len() as u64 + ext_total,
        )
    }

    /// The per-million rate a word must be judged against for target
    /// attestation: the *best* single-bucket rate (DMS human stream, or
    /// the external pack over its own total), never the pooled rate.
    ///
    /// Attestation asks "do humans use this word at all?" — an
    /// existence question, answered per corpus. Pooling would let a
    /// large register-narrow external corpus dilute a word that is
    /// perfectly ordinary in the DMS corpus's registers below the
    /// floor ("food" is rare in code-review prose; that must not
    /// un-attest it for a travel-register rule), silently converting
    /// the floor from "unknown to humans" into "unknown to whichever
    /// register dominates the token count".
    #[must_use]
    pub fn human_attestation_rate_per_million(&self, word: &str) -> f64 {
        let word = word.to_lowercase();
        let dms_count = self.human_word_counts.get(&word).copied().unwrap_or(0);
        let dms_rate = pooled_rate_per_million(dms_count, self.human_stream.len() as u64);
        let ext_rate = self.external.as_ref().map_or(0.0, |pack| {
            pooled_rate_per_million(pack.unigram_count(&word), pack.total_tokens())
        });
        dms_rate.max(ext_rate)
    }

    /// Pooled evidence for one anchor phrase: `(machine_count,
    /// human_count, machine_total, human_total)` — exactly the four raw
    /// numbers [`crate::frame_rules::classify`] takes, already resolved
    /// to the same measurement universe.
    ///
    /// `machine_count`/`machine_total` are always DMS-only (the external
    /// pack is human corpora only, by construction). `human_count`/
    /// `human_total` pool the external pack's PROBE-table entry and its
    /// own token total — but only when `words` was actually registered
    /// as a probe when that pack was built. An unregistered phrase keeps
    /// both DMS-only: the alternative (an un-pooled numerator over a
    /// pooled denominator) would silently deflate every unregistered
    /// phrase's apparent rate, which is exactly what this struct's own
    /// documented absence asymmetry says a probe's absence must not do
    /// (see this struct's own docs). A phrase containing any word the
    /// DMS vocabulary does not know occurs zero times on the DMS side by
    /// construction.
    #[must_use]
    pub fn phrase_counts(&self, words: &[&str]) -> (u64, u64, u64, u64) {
        let (machine, dms_human) = self.dms_phrase_counts(words);
        let machine_total = self.machine_stream.len() as u64;
        let dms_human_total = self.human_stream.len() as u64;
        match self.external.as_ref().and_then(|pack| {
            pack.probe_count(words)
                .map(|count| (count, pack.total_tokens()))
        }) {
            Some((ext_human, ext_total)) => (
                machine,
                dms_human + ext_human,
                machine_total,
                dms_human_total + ext_total,
            ),
            None => (machine, dms_human, machine_total, dms_human_total),
        }
    }

    /// Register-matched phrase evidence: `(machine_count, human_count,
    /// machine_total, human_total)`, both sides DMS-only.
    ///
    /// Two-sided verdicts, direction errors and guard confirmation,
    /// compare a machine rate against a human rate, which is only
    /// meaningful when both sides sample the same register mix. The DMS
    /// streams do (their genres were paired by construction); an
    /// external human corpus does not (it brings its own register, with
    /// no machine counterpart), so pooling it into one side of a ratio
    /// manufactures verdicts out of register difference — "enable" is
    /// human-tilted in docs prose and rare in review comments, and only
    /// the first comparison says anything about machines. One-sided
    /// checks (attestation, the human-rate ceiling) are the ones that
    /// may draw on external corpora.
    #[must_use]
    pub fn phrase_counts_register_matched(&self, words: &[&str]) -> (u64, u64, u64, u64) {
        let (machine, human) = self.dms_phrase_counts(words);
        (
            machine,
            human,
            self.machine_stream.len() as u64,
            self.human_stream.len() as u64,
        )
    }

    /// The DMS-only phrase counts [`Self::phrase_counts`] pools external
    /// evidence on top of.
    fn dms_phrase_counts(&self, words: &[&str]) -> (u64, u64) {
        let ids: Option<Vec<u32>> = words
            .iter()
            .map(|w| self.word_ids.get(&w.to_lowercase()).copied())
            .collect();
        let Some(ids) = ids else {
            return (0, 0);
        };
        if ids.is_empty() {
            return (0, 0);
        }
        let count = |stream: &[u32]| {
            let mut n = 0u64;
            let mut i = 0;
            while i + ids.len() <= stream.len() {
                if stream[i..i + ids.len()] == ids[..] {
                    n += 1;
                    i += ids.len();
                } else {
                    i += 1;
                }
            }
            n
        };
        (count(&self.machine_stream), count(&self.human_stream))
    }

    /// Each side's pooled token total: `(machine, human)`. The machine
    /// total is always DMS-only; the human total folds in the external
    /// pack's own token total whenever one is attached, regardless of
    /// any particular phrase — a general corpus-size stat, not the
    /// phrase-specific denominator [`Self::phrase_counts`] already
    /// bundles (see that method's own docs for why the two can differ
    /// for a phrase absent from the external pack's probe table).
    #[must_use]
    pub fn totals(&self) -> (u64, u64) {
        let ext_total = self
            .external
            .as_ref()
            .map_or(0, HumanEvidencePack::total_tokens);
        (
            self.machine_stream.len() as u64,
            self.human_stream.len() as u64 + ext_total,
        )
    }
}

/// `count` occurrences over `total` tokens, as occurrences per million —
/// `0.0` for a zero total rather than a division by zero.
#[expect(
    clippy::cast_precision_loss,
    reason = "corpus counts are far below 2^52"
)]
fn pooled_rate_per_million(count: u64, total: u64) -> f64 {
    if total == 0 {
        0.0
    } else {
        count as f64 / total as f64 * 1_000_000.0
    }
}

/// Compiles the fenced buckets of `set` against the attestation table.
///
/// # Errors
/// Returns [`FrameCompileError`] only for structural inconsistencies
/// (duplicate ids, unparseable fenced rules); evidence-based exclusions
/// land in the report instead.
pub fn compile(
    set: &FrameRuleSet,
    evidence: &CorpusEvidence,
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
    let ship_parents: BTreeMap<&str, Vec<PatElem>> = set
        .rules_ship
        .iter()
        .filter_map(|r| Some((r.id.as_str(), parse_pattern(&r.p).ok()?)))
        .collect();

    // Individual fences. Each draft either compiles, demotes, or falls.
    // The closure fence needs every survivor's anchor, so anchors are
    // resolved first and closure runs as a second pass.
    let mut survivors: Vec<CompiledRule> = Vec::new();
    for draft in drafts {
        match compile_one(draft, &mut interner, &mut classes, &ship_parents, evidence) {
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
    // no output and report rules never edit: both exempt, see the
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
/// ids (first occurrence wins. The shipped surface bucket carries a
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
    ship_parents: &BTreeMap<&str, Vec<PatElem>>,
    evidence: &CorpusEvidence,
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

    // Shadowing: a surface rule adds nothing when its pattern is its
    // compiling parent's pattern with literals merely re-inflected —
    // literal matching is lemma-level, so the parent already covers
    // every such match. A surface with a *different* structure (extra
    // sentinels, groups, tags) is a refinement, not a shadow, and
    // compiles with its parent; overlaps resolve by leftmost-longest
    // at runtime.
    if draft.bucket == Bucket::Surface
        && let Some(parent) = draft
            .id
            .split_once("::")
            .map(|(parent, _)| parent.to_string())
        && ship_parents
            .get(parent.as_str())
            .is_some_and(|parent_pattern| is_inflection_shadow(&draft.pattern, parent_pattern))
    {
        return Outcome::Rejected(draft.id, Reject::ShadowedByLemmaRule { parent });
    }

    // Anchor: the longest run of consecutive obligatory anchorable
    // elements — literals, plus single-member-class tag positions
    // (see `anchorable_word`).
    let Some((anchor_start, anchor_len)) = anchor_run(&draft.pattern, classes, interner) else {
        return Outcome::Rejected(draft.id, Reject::ZeroLiteralAnchor);
    };
    let anchor_words: Vec<String> = draft.pattern[anchor_start..anchor_start + anchor_len]
        .iter()
        .map(|elem| {
            anchorable_word(elem, classes, interner)
                .expect("anchor runs contain only anchorable elements")
        })
        .collect();

    // Demotions to report-only: flip-bucket guards (measured
    // machine-tilted: protecting them would shelter machine-leaning
    // text) and unauthored targets.
    let demotion = if draft.bucket == Bucket::Flip {
        Some("guard measured machine-tilted; reports instead of protecting".to_string())
    } else if draft.raw_target.trim_matches('"') == "REVIEW" {
        Some("target was never authored (REVIEW placeholder)".to_string())
    } else if template_has_placeholder(&draft.target) {
        Some("target realization was never authored (`?` placeholder)".to_string())
    } else if draft.kind != RuleKind::Guard
        && let Some(reason) = anchor_evidence_demotion(&anchor_words, evidence)
    {
        Some(reason)
    } else if draft.kind == RuleKind::Guard
        && let Some((m, h)) = machine_tilted_guard(&anchor_words, evidence)
    {
        // A guard whose own anchor measures machine-tilted would
        // shelter machine-leaning text: the flip bucket's exact
        // definition, applied to every guard at compile time rather
        // than only to the rows the original referee caught.
        Some(format!(
            "guard anchor measured machine-tilted on the shipped corpora ({m} machine vs {h} human occurrences); reports instead of protecting"
        ))
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
        && let Some(reject) = attest_template(&template, interner, classes, evidence)
    {
        return Outcome::Rejected(draft.id, reject);
    }

    let pattern: Vec<CPat> = draft
        .pattern
        .iter()
        .map(|elem| compile_pat(elem, interner, classes))
        .collect();
    let anchor_ids = anchor_words
        .iter()
        .map(|word| interner.intern(word))
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

/// Whether `surface` matches exactly the same shapes as `parent`,
/// element for element, with literals allowed to differ only by
/// inflection of the parent's lemma (checked through the inflection
/// tables: `surface_lit` must be `parent_lit` realized in
/// `surface_lit`'s own form).
fn is_inflection_shadow(surface: &[PatElem], parent: &[PatElem]) -> bool {
    surface.len() == parent.len()
        && surface.iter().zip(parent).all(|(s, p)| match (s, p) {
            (PatElem::Lit(surface_lit), PatElem::Lit(parent_lit)) => {
                surface_lit == parent_lit
                    || friction_nlp::inflect(surface_lit, parent_lit).as_deref()
                        == Some(surface_lit.as_str())
            }
            _ => s == p,
        })
}

/// The one word this pattern element pins every match to, if any: a
/// literal's own text, or — for a `TAG[class]` position whose class
/// has exactly one single-word member — that member. Either shape lets
/// the element serve in an anchor run: the index prefilter matches the
/// word at lemma level, and for a tag-classed element the verifier
/// still enforces the tag before the rule can fire (the verb-gated
/// `vsub.harness` split is the motivating case: noun "harness" reaches
/// the verifier and is refused there).
fn anchorable_word(elem: &PatElem, classes: &ClassTable, interner: &Interner) -> Option<String> {
    match elem {
        PatElem::Lit(text) => Some(text.clone()),
        PatElem::TagClass { class, .. } => classes
            .single_word_member(class)
            .map(|id| interner.words[id as usize].clone()),
        _ => None,
    }
}

/// Finds the longest run of consecutive obligatory anchorable elements
/// (see [`anchorable_word`]); returns `(start, len)`, leftmost winning
/// ties.
fn anchor_run(
    pattern: &[PatElem],
    classes: &ClassTable,
    interner: &Interner,
) -> Option<(usize, usize)> {
    let mut best: Option<(usize, usize)> = None;
    let mut run_start = 0;
    let consider = |start: usize, end: usize, best: &mut Option<(usize, usize)>| {
        let len = end - start;
        if len > 0 && best.is_none_or(|(_, best_len)| len > best_len) {
            *best = Some((start, len));
        }
    };
    for (i, elem) in pattern.iter().enumerate() {
        if anchorable_word(elem, classes, interner).is_none() {
            consider(run_start, i, &mut best);
            run_start = i + 1;
        }
    }
    consider(run_start, pattern.len(), &mut best);
    best
}

/// The per-million rate above which an anchor is too common in human
/// prose to auto-edit: at 100/M a rule fires on ordinary human text
/// about once every ten thousand words, which is exactly the
/// near-no-op noise the human-holdout criterion forbids — the same
/// reasoning that turned the human-dense verbs into guards.
const HUMAN_RATE_CEILING_PER_MILLION: f64 = 100.0;

/// Why the anchor's own measured evidence demotes an edit rule to
/// report-only, if it does. Two fences, both re-derived from corpus
/// bytes on every compile (the trust model applied to the delivered
/// buckets — never hand-curation):
///
/// - **direction**: an anchor solidly HUMAN-tilted here (three or more
///   human occurrences and the referee's own direction-error verdict)
///   marks a rewrite that would push text toward the machine register.
///   Measured register-matched (DMS streams only) — a two-sided ratio
///   over mismatched registers is meaningless, see
///   [`CorpusEvidence::phrase_counts_register_matched`];
/// - **human-rate ceiling**: an anchor above
///   [`HUMAN_RATE_CEILING_PER_MILLION`] in human prose fires on human
///   text too often to auto-edit no matter how machine-tilted its
///   ratio is — absolute human frequency, not the ratio, is what sets
///   the human-holdout edit noise.
fn anchor_evidence_demotion(anchor_words: &[String], evidence: &CorpusEvidence) -> Option<String> {
    let words: Vec<&str> = anchor_words.iter().map(String::as_str).collect();
    #[expect(
        clippy::cast_precision_loss,
        reason = "corpus counts are far below 2^52"
    )]
    let rate = |count: u64, total: u64| {
        if total == 0 {
            0.0
        } else {
            count as f64 / total as f64 * 1_000_000.0
        }
    };
    let (m, h, m_total, h_total) = evidence.phrase_counts_register_matched(&words);
    if h >= 3 && classify(RuleKind::Rewrite, m, h, m_total, h_total) == Verdict::DirectionError {
        return Some(format!(
            "anchor measured human-tilted on the shipped corpora ({m} machine vs {h} human occurrences)"
        ));
    }
    // Second register-matched pair: generated-LLM review prose vs the
    // external human review corpora. The machine side is small, so its
    // count enters with a rule-of-three cushion (+3): absence from ~10^5
    // generated tokens only bounds the machine rate, it does not
    // establish "humans-only" — the verdict must hold even against the
    // machine rate's plausible upper bound before it may demote.
    if let Some((m2, h2, m2_total, h2_total)) = evidence.review_pair_counts(&words)
        && h2 >= 3
        && classify(RuleKind::Rewrite, m2 + 3, h2, m2_total, h2_total) == Verdict::DirectionError
    {
        return Some(format!(
            "anchor measured human-tilted in the review register ({m2} machine vs {h2} human occurrences)"
        ));
    }
    // The ceiling is one-sided (human frequency alone), so it may pool
    // every human corpus we have, including external registers.
    let (_, pooled_h, _, pooled_h_total) = evidence.phrase_counts(&words);
    let h_rate = rate(pooled_h, pooled_h_total);
    (h_rate >= HUMAN_RATE_CEILING_PER_MILLION).then(|| {
        format!("anchor too common in human prose to auto-edit ({h_rate:.0}/M human); report-only")
    })
}

/// The guard anchor's `(machine, human)` counts when they classify as
/// solidly machine-tilted (the confirmation bar) — `None` otherwise.
fn machine_tilted_guard(anchor_words: &[String], evidence: &CorpusEvidence) -> Option<(u64, u64)> {
    let words: Vec<&str> = anchor_words.iter().map(String::as_str).collect();
    // Guard confirmation is a two-sided verdict: register-matched
    // evidence only (see `CorpusEvidence::phrase_counts_register_matched`).
    // Either matched pair may confirm — the DMS streams, or the
    // review-register pair (whose machine counts are real occurrences,
    // positive evidence needing no rule-of-three cushion).
    let (m, h, m_total, h_total) = evidence.phrase_counts_register_matched(&words);
    if classify(RuleKind::Rewrite, m, h, m_total, h_total) == Verdict::Confirmed {
        return Some((m, h));
    }
    evidence
        .review_pair_counts(&words)
        .and_then(|(m2, h2, m2_total, h2_total)| {
            (classify(RuleKind::Rewrite, m2, h2, m2_total, h2_total) == Verdict::Confirmed)
                .then_some((m2, h2))
        })
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
    // Sentence sentinels are positional constraints, not emissions,
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
                    // A lemma to realize. Agreement source, in
                    // preference order: the pattern's BE element if it
                    // has exactly one (finite-verb agreement); else
                    // the pattern's leading literal (lemma-matched, so
                    // "embarked on this journey" realizes VB[begin] as
                    // "began", never bare "begin"); else the tag's own
                    // form.
                    let lemma = interner.intern(arg);
                    let agree = be_element(&draft.pattern).or_else(|| {
                        draft
                            .pattern
                            .iter()
                            .position(|elem| !matches!(elem, PatElem::SentStart | PatElem::SentEnd))
                            .filter(|&i| matches!(draft.pattern[i], PatElem::Lit(_)))
                            .and_then(|i| u8::try_from(i).ok())
                    });
                    match agree {
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
    evidence: &CorpusEvidence,
) -> Option<Reject> {
    let function_words = &interner.function_words;
    let check = |id: u32| {
        let word = &interner.words[id as usize];
        if function_words.contains(word.as_str()) {
            return None;
        }
        let per_million = evidence.human_attestation_rate_per_million(word);
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
/// wildcarding it). Inflected realizations contribute their lemma id.
/// The closure check compares lemma-level anchors, and the realization
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

    /// The class's one single-word member id, when the class has
    /// exactly one member and that member is exactly one word — the
    /// shape that lets a `TAG[class]` position anchor a scan like a
    /// literal would (see [`anchorable_word`]).
    fn single_word_member(&self, name: &str) -> Option<u32> {
        let members = &self.members[usize::from(self.id_of(name)?)];
        match members.as_slice() {
            [member] => match member.as_slice() {
                [only] => Some(*only),
                _ => None,
            },
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame_rules::FrameRuleSet;

    /// An evidence table where the given words are attested and the
    /// direction fence never trips.
    fn generous_rates(words: &[&str]) -> CorpusEvidence {
        CorpusEvidence::for_tests(words)
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
        let set = FrameRuleSet::parse(include_str!("../packs/frame-rules-en-v1.toml"))
            .expect("shipped rules parse");
        let rates = CorpusEvidence::from_dms_toml(include_str!("../packs/dms-index-en-v1.toml"))
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
        let set = FrameRuleSet::parse(include_str!("../packs/frame-rules-en-v1.toml"))
            .expect("shipped rules parse");
        let rates = CorpusEvidence::from_dms_toml(include_str!("../packs/dms-index-en-v1.toml"))
            .expect("shipped dms index parses");
        let (pack_a, _) = compile(&set, &rates).expect("compile a");
        let (pack_b, _) = compile(&set, &rates).expect("compile b");
        assert_eq!(pack_a, pack_b);
    }

    // --- external evidence pooling ---

    use crate::human_evidence::{HumanEvidencePack, build_pack_bytes as build_human_evidence_bin};

    /// A tiny synthetic `dms-index-v1`-shaped TOML: two human occurrences
    /// of "use", one machine occurrence, a two-word human-side phrase.
    const SMALL_DMS_TOML: &str = r#"
        [vocab]
        tokens = ["<unused-0>", "use", "please", "leverage", "synergy"]

        [streams.human]
        ids = "1,2,1,3,4"

        [streams.qwen]
        ids = "3,4"
    "#;

    /// A pooled word's rate is the sum of both sides' counts over the
    /// sum of both sides' totals — not either side's rate averaged.
    #[test]
    fn human_rate_per_million_pools_dms_and_external_unigram_counts() {
        let unigrams = BTreeMap::from([("use".to_string(), (3u64, 0u32))]);
        let external =
            HumanEvidencePack::load(&build_human_evidence_bin(&unigrams, &BTreeMap::new(), 5))
                .expect("built pack loads");
        let evidence = CorpusEvidence::from_dms_toml(SMALL_DMS_TOML)
            .expect("small pack parses")
            .with_external(external);
        // dms: "use" occurs 2 times over 5 human tokens; external: 3
        // times over 5 tokens. Pooled: (2+3)/(5+5)*1e6 = 500_000.0.
        let rate = evidence.human_rate_per_million("use");
        assert!((rate - 500_000.0).abs() < 1e-9, "pooled rate was {rate}");
    }

    /// A word absent from the external pack's unigram table still folds
    /// the external total into the denominator (absence means "fewer
    /// than the floor", not "not measured") — the rate strictly
    /// decreases relative to dms-only, it never silently ignores the
    /// external corpus's existence.
    #[test]
    fn human_rate_per_million_folds_in_external_total_even_when_the_word_is_unmeasured() {
        let external = HumanEvidencePack::load(&build_human_evidence_bin(
            &BTreeMap::new(),
            &BTreeMap::new(),
            5,
        ))
        .expect("built pack loads");
        let evidence = CorpusEvidence::from_dms_toml(SMALL_DMS_TOML)
            .expect("small pack parses")
            .with_external(external);
        // dms: "use" occurs 2/5. Pooled: (2+0)/(5+5)*1e6 = 200_000.0,
        // strictly less than the dms-only 400_000.0.
        let rate = evidence.human_rate_per_million("use");
        assert!((rate - 200_000.0).abs() < 1e-9, "pooled rate was {rate}");
    }

    /// A phrase registered in the external pack's probe table pools its
    /// count and the pack's total into both the numerator and the
    /// denominator together.
    #[test]
    fn phrase_counts_pools_a_registered_probe_numerator_and_denominator_together() {
        let probes = BTreeMap::from([(vec!["please".to_string()], 7u64)]);
        let external =
            HumanEvidencePack::load(&build_human_evidence_bin(&BTreeMap::new(), &probes, 5))
                .expect("built pack loads");
        let evidence = CorpusEvidence::from_dms_toml(SMALL_DMS_TOML)
            .expect("small pack parses")
            .with_external(external);
        // dms: "please" occurs 1/5 human, 0/2 machine. Pooled human:
        // (1+7) over (5+5); machine stays dms-only (0 over 2).
        assert_eq!(evidence.phrase_counts(&["please"]), (0, 8, 2, 10));
    }

    /// A phrase never registered as a probe when the external pack was
    /// built pools nothing — both its count and its total stay
    /// dms-only, never an unpooled numerator over a pooled denominator
    /// (which would silently deflate its apparent rate).
    #[test]
    fn phrase_counts_leaves_an_unregistered_phrase_entirely_dms_only() {
        let probes = BTreeMap::from([(vec!["please".to_string()], 7u64)]);
        let external =
            HumanEvidencePack::load(&build_human_evidence_bin(&BTreeMap::new(), &probes, 5))
                .expect("built pack loads");
        let evidence = CorpusEvidence::from_dms_toml(SMALL_DMS_TOML)
            .expect("small pack parses")
            .with_external(external);
        // "use" was never one of this pack's probes (only "please"
        // was): dms-only on both sides: 2 human over 5, 0 machine over
        // 2.
        assert_eq!(evidence.phrase_counts(&["use"]), (0, 2, 2, 5));
    }

    /// `totals()` is a general corpus-size stat: it folds in the
    /// external total whenever a pack is attached, independent of any
    /// particular phrase (unlike `phrase_counts`'s bundled denominator).
    #[test]
    fn totals_folds_in_the_external_total_unconditionally() {
        let external = HumanEvidencePack::load(&build_human_evidence_bin(
            &BTreeMap::new(),
            &BTreeMap::new(),
            5,
        ))
        .expect("built pack loads");
        let evidence = CorpusEvidence::from_dms_toml(SMALL_DMS_TOML)
            .expect("small pack parses")
            .with_external(external);
        assert_eq!(evidence.totals(), (2, 10));
    }

    /// The empty pack (the shipped placeholder — see
    /// `crate::human_evidence`'s own docs) is a no-op: every pooled
    /// accessor equals the dms-only computation exactly, and compiling
    /// the shipped rule set with it attached produces a byte-identical
    /// pack to compiling without any external evidence at all.
    #[test]
    fn empty_external_pack_leaves_compile_output_unchanged() {
        let set = FrameRuleSet::parse(include_str!("../packs/frame-rules-en-v1.toml"))
            .expect("shipped rules parse");
        let dms_toml = include_str!("../packs/dms-index-en-v1.toml");
        let without_external = CorpusEvidence::from_dms_toml(dms_toml).expect("dms index parses");
        let with_empty_external = CorpusEvidence::from_dms_toml(dms_toml)
            .expect("dms index parses")
            .with_external(HumanEvidencePack::empty());
        let (pack_a, report_a) = compile(&set, &without_external).expect("compile a");
        let (pack_b, report_b) = compile(&set, &with_empty_external).expect("compile b");
        assert_eq!(pack_a, pack_b);
        assert_eq!(report_a.compiled, report_b.compiled);
        assert_eq!(report_a.demoted, report_b.demoted);
        assert_eq!(report_a.rejected, report_b.rejected);
    }
}
