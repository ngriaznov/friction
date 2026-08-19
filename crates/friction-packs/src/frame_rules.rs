//! Source model for the adjudicated frame-rewrite rule set.
//!
//! Covers `packs/frame-rules-en-v1.toml`: seven evidence buckets, the
//! class tables, the declared function-word set, and the
//! pattern/target grammar the rules are written in.
//!
//! This module is the *tooling-side* read path. The runtime never parses
//! this TOML — it reads the derived zero-copy pack instead; this model
//! feeds `corpus-tool adjudicate` (the evidence referee) and the pack
//! compiler. Rule patterns are therefore kept as raw strings at load
//! time and parsed per rule via [`parse_pattern`] / [`parse_target`]:
//! the staged buckets deliberately carry rules the compiler must
//! *reject*, so one malformed staged pattern must surface as that rule's
//! own verdict, never as a load failure that blocks the whole set.
//!
//! # The compile fence
//!
//! Only the `ship`, `flip`, `surface`, and `pilot` buckets are ever
//! candidates for pack compilation. `quarantine`, `no_evidence`, and
//! `staged_surface` are adjudication evidence: they exist so the referee
//! can re-derive every verdict from the corpora, and so a rule whose
//! evidence improves can be promoted by re-running the referee, never by
//! hand-editing a bucket.
//!
//! # Verdict policy
//!
//! [`classify`] is the single verdict function. Its thresholds are the
//! adjudication policy for the whole feature: a rewrite/delete rule is
//! confirmed at ratio ≥ 2 with at least two machine-side occurrences,
//! direction-errored at ratio ≤ 0.67, weak between; a guard is confirmed
//! whenever the human side matches at least as densely as the machine
//! side; a rule unobserved on both sides has no evidence. Ratios compare
//! per-million rates so the two corpus sides need not be the same size;
//! the occurrence floor uses raw counts so a single stray hit can never
//! confirm a rule on its own.

use std::collections::BTreeMap;

use serde::Deserialize;

/// One adjudicated rule row, pattern and target still raw.
#[derive(Debug, Clone, Deserialize)]
pub struct FrameRule {
    /// Stable rule id (`family.name` or `family.name::surface`).
    pub id: String,
    /// Raw source pattern in the grammar [`parse_pattern`] accepts.
    pub p: String,
    /// Raw target ladder: ordered candidate target templates, primary
    /// (today's sole target) first (`""` = delete, `"="` = guard —
    /// meaningful only as a one-element ladder; see `frame_compile`'s
    /// own docs for how the compiler treats a delete/guard-kind row
    /// that carries more than one). A bare TOML string (`t="\"use\" X"`)
    /// parses as the one-element ladder every rule authored so far
    /// carries; an array (`t=["\"improve\" X", "\"boost\" X"]`) parses
    /// as an ordered ladder — at apply time the runtime tries each
    /// candidate in order and fires the first whose realization clears
    /// the full gate stack, holding (as today) only if none does.
    #[serde(deserialize_with = "deserialize_target_ladder")]
    pub t: Vec<String>,
    /// Rule kind letter: `r` rewrite, `d` delete, `g` guard.
    pub k: String,
    /// Measured machine-corpus occurrences per million tokens.
    pub m_pm: f64,
    /// Measured human-corpus occurrences per million tokens.
    pub h_pm: f64,
    /// The adjudicated verdict label as recorded by the referee.
    pub v: String,
    /// Optional authoring note (e.g. "target needs per-collocation
    /// authoring at compile review").
    #[serde(default)]
    pub n: Option<String>,
}

/// Deserializes [`FrameRule::t`]'s bare-string-or-array shape into a
/// non-empty ladder. Rejects an empty array explicitly: a rule with no
/// primary target at all is a malformed row, not a zero-length ladder.
fn deserialize_target_ladder<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Raw {
        One(String),
        Many(Vec<String>),
    }
    let ladder = match Raw::deserialize(deserializer)? {
        Raw::One(target) => vec![target],
        Raw::Many(targets) => targets,
    };
    if ladder.is_empty() {
        return Err(serde::de::Error::custom(
            "target ladder must carry at least one target",
        ));
    }
    Ok(ladder)
}

/// One pilot-bucket rule: a plain literal phrase pair with pair-mined
/// support instead of per-million corpus measurements.
#[derive(Debug, Clone, Deserialize)]
pub struct PilotRule {
    /// Literal source phrase, space-separated surface tokens.
    pub p: String,
    /// Literal replacement (`""` = delete).
    pub t: String,
    /// Rule kind letter: `r` rewrite, `d` delete.
    pub k: String,
    /// Number of aligned corruption pairs exhibiting this edit.
    pub support: u32,
}

/// The whole `frame-rules-en-v1.toml` artifact, buckets in fence order.
#[derive(Debug, Clone, Deserialize)]
pub struct FrameRuleSet {
    /// Corpus-confirmed knowledge rules (compile).
    pub rules_ship: Vec<FrameRule>,
    /// Authored as guards, measured machine-tilted (compile as
    /// report-only: never as guards, never as rewrites).
    pub rules_flip: Vec<FrameRule>,
    /// Materialized surfaces of confirmed frames (compile).
    pub rules_surface: Vec<FrameRule>,
    /// Pair-mined literal rules (compile); also the miner fixture.
    pub rules_pilot: Vec<PilotRule>,
    /// Direction-error or weak differential (never compile).
    pub rules_quarantine: Vec<FrameRule>,
    /// Unobserved on both corpus sides (never compile).
    pub rules_no_evidence: Vec<FrameRule>,
    /// Surfaces of unconfirmed frames (never compile; promoted only by
    /// re-adjudication).
    pub rules_staged_surface: Vec<FrameRule>,
    /// Class name → member lemmas (multi-word members hyphenated).
    pub classes: BTreeMap<String, Vec<String>>,
    /// The declared function-word set: the only non-slot, non-table
    /// words a target may introduce without human-corpus attestation.
    pub function_words: FunctionWords,
}

/// Wrapper for the `[function_words]` table.
#[derive(Debug, Clone, Deserialize)]
pub struct FunctionWords {
    /// The words themselves, lowercase.
    pub words: Vec<String>,
}

impl FrameRuleSet {
    /// Parses the `frame-rules-en-v1.toml` text into the bucket model.
    ///
    /// # Errors
    /// Returns [`FrameRulesError::Toml`] if the text is not valid TOML
    /// for this shape, and [`FrameRulesError::UnknownKind`] if any rule
    /// carries a kind letter outside `r`/`d`/`g`.
    pub fn parse(text: &str) -> Result<Self, FrameRulesError> {
        let set: Self = toml::from_str(text).map_err(|e| FrameRulesError::Toml(e.to_string()))?;
        for rule in set.knowledge_buckets().flat_map(|(_, rules)| rules) {
            RuleKind::parse(&rule.k).ok_or_else(|| FrameRulesError::UnknownKind {
                id: rule.id.clone(),
                kind: rule.k.clone(),
            })?;
        }
        for rule in &set.rules_pilot {
            RuleKind::parse(&rule.k).ok_or_else(|| FrameRulesError::UnknownKind {
                id: format!("pilot:{}", rule.p),
                kind: rule.k.clone(),
            })?;
        }
        Ok(set)
    }

    /// All non-pilot buckets with their names, fence order.
    pub fn knowledge_buckets(&self) -> impl Iterator<Item = (&'static str, &[FrameRule])> {
        [
            ("ship", self.rules_ship.as_slice()),
            ("flip", self.rules_flip.as_slice()),
            ("surface", self.rules_surface.as_slice()),
            ("quarantine", self.rules_quarantine.as_slice()),
            ("no_evidence", self.rules_no_evidence.as_slice()),
            ("staged_surface", self.rules_staged_surface.as_slice()),
        ]
        .into_iter()
    }
}

/// A rule's operation kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuleKind {
    /// Rewrite the matched span through the target template.
    Rewrite,
    /// Delete the matched span.
    Delete,
    /// Protect the matched span from all rewriting.
    Guard,
}

impl RuleKind {
    /// Parses the single-letter kind used by the rule rows.
    pub fn parse(letter: &str) -> Option<Self> {
        match letter {
            "r" => Some(Self::Rewrite),
            "d" => Some(Self::Delete),
            "g" => Some(Self::Guard),
            _ => None,
        }
    }
}

/// Coarse tag positions the pattern grammar can name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[expect(missing_docs, reason = "tag inventory mirrors the coarse tagset")]
pub enum Tag {
    Nn,
    Nns,
    Vb,
    Vbz,
    Vbg,
    Vbn,
    Vbd,
    Adj,
    Adjs,
    Adv,
    Dt,
    Md,
    Prp,
    PrpPoss,
    Aux,
    Be,
    Uh,
    Punct,
    Nom,
}

impl Tag {
    fn parse(word: &str) -> Option<Self> {
        Some(match word {
            "NN" => Self::Nn,
            "NNS" => Self::Nns,
            "VB" => Self::Vb,
            "VBZ" => Self::Vbz,
            "VBG" => Self::Vbg,
            "VBN" => Self::Vbn,
            "VBD" => Self::Vbd,
            "ADJ" => Self::Adj,
            "ADJS" => Self::Adjs,
            "ADV" => Self::Adv,
            "DT" => Self::Dt,
            "MD" => Self::Md,
            "PRP" => Self::Prp,
            "PRP$" => Self::PrpPoss,
            "AUX" => Self::Aux,
            "BE" => Self::Be,
            "UH" => Self::Uh,
            "PUNCT" => Self::Punct,
            "NOM" => Self::Nom,
            _ => return None,
        })
    }

    /// Every tag in stable serialization order — [`Self::code`] is the
    /// index here, so this array is part of the pack format.
    pub const ALL: [Self; 19] = [
        Self::Nn,
        Self::Nns,
        Self::Vb,
        Self::Vbz,
        Self::Vbg,
        Self::Vbn,
        Self::Vbd,
        Self::Adj,
        Self::Adjs,
        Self::Adv,
        Self::Dt,
        Self::Md,
        Self::Prp,
        Self::PrpPoss,
        Self::Aux,
        Self::Be,
        Self::Uh,
        Self::Punct,
        Self::Nom,
    ];

    /// The tag's stable one-byte serialization code.
    #[must_use]
    pub fn code(self) -> u8 {
        Self::ALL
            .iter()
            .position(|t| *t == self)
            .map(|i| u8::try_from(i).expect("tag count fits u8"))
            .expect("every tag is in ALL")
    }

    /// Decodes [`Self::code`]'s value.
    #[must_use]
    pub fn from_code(code: u8) -> Option<Self> {
        Self::ALL.get(usize::from(code)).copied()
    }
}

/// Contraction clitic tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[expect(missing_docs, reason = "one variant per clitic surface form")]
pub enum Clitic {
    S,
    Nt,
    Ve,
    Re,
    Ll,
}

impl Clitic {
    fn parse(word: &str) -> Option<Self> {
        Some(match word {
            "CLIT_S" => Self::S,
            "CLIT_NT" => Self::Nt,
            "CLIT_VE" => Self::Ve,
            "CLIT_RE" => Self::Re,
            "CLIT_LL" => Self::Ll,
            _ => return None,
        })
    }

    /// Every clitic in stable serialization order — [`Self::code`] is
    /// the index here, so this array is part of the pack format.
    pub const ALL: [Self; 5] = [Self::S, Self::Nt, Self::Ve, Self::Re, Self::Ll];

    /// The clitic's stable one-byte serialization code.
    #[must_use]
    pub fn code(self) -> u8 {
        Self::ALL
            .iter()
            .position(|c| *c == self)
            .map(|i| u8::try_from(i).expect("clitic count fits u8"))
            .expect("every clitic is in ALL")
    }

    /// Decodes [`Self::code`]'s value.
    #[must_use]
    pub fn from_code(code: u8) -> Option<Self> {
        Self::ALL.get(usize::from(code)).copied()
    }

    /// The clitic's surface text, apostrophe included.
    #[must_use]
    pub const fn surface(self) -> &'static str {
        match self {
            Self::S => "'s",
            Self::Nt => "n't",
            Self::Ve => "'ve",
            Self::Re => "'re",
            Self::Ll => "'ll",
        }
    }
}

/// Typed content slots, copied verbatim into targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[expect(missing_docs, reason = "slot names are the grammar's own")]
pub enum Slot {
    X,
    Y,
    Z,
}

impl Slot {
    fn parse(word: &str) -> Option<Self> {
        Some(match word {
            "X" => Self::X,
            "Y" => Self::Y,
            "Z" => Self::Z,
            _ => return None,
        })
    }

    /// The slot's stable one-byte serialization code (also its index).
    #[must_use]
    pub const fn code(self) -> u8 {
        match self {
            Self::X => 0,
            Self::Y => 1,
            Self::Z => 2,
        }
    }

    /// Decodes [`Self::code`]'s value.
    #[must_use]
    pub const fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::X),
            1 => Some(Self::Y),
            2 => Some(Self::Z),
            _ => None,
        }
    }
}

/// One parsed element of a source pattern.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PatElem {
    /// Lemma-matched literal token (stored lowercase).
    Lit(String),
    /// Coarse tag position.
    Tag(Tag),
    /// Tag position constrained to a class table's members.
    TagClass {
        /// The tag the position must carry.
        tag: Tag,
        /// The class-table name (resolved against the set's `classes`
        /// by the consumer; an unknown name is a compile rejection, not
        /// a parse error).
        class: String,
    },
    /// Typed content slot.
    Slot(Slot),
    /// `<S>` — the match must start the sentence.
    SentStart,
    /// `</S>` — the match must end the sentence.
    SentEnd,
    /// Contraction clitic token.
    Clitic(Clitic),
    /// A single optional element (`elem?`).
    Optional(Box<Self>),
    /// Alternation group `(a b | c)`; each alternative is a sequence.
    Group {
        /// The alternatives, in written order.
        alts: Vec<Vec<Self>>,
        /// Whether the whole group carries a `?`.
        optional: bool,
    },
}

/// A parsed target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Target {
    /// `t="="` — protect the matched span from all rewriting.
    Guard,
    /// `t=""` — delete the matched span.
    Delete,
    /// A replacement template.
    Template(Vec<TplElem>),
}

/// One parsed element of a target template.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TplElem {
    /// Emit this literal token.
    Lit(String),
    /// Copy a slot's captured bytes verbatim.
    Slot(Slot),
    /// Copy the token the pattern matched at its (unique) position with
    /// this tag.
    TagCopy(Tag),
    /// `TAG[arg]` — realize `arg` in this tag's form. Whether `arg` is
    /// a lemma, a class name, or an unauthored placeholder (`?`) is
    /// resolved by the compiler against the class tables. The parser
    /// records it verbatim.
    TagArg {
        /// The form to realize.
        tag: Tag,
        /// The bracketed argument, verbatim.
        arg: String,
    },
    /// Copy the token matched by the pattern's (positionally
    /// corresponding) alternation group.
    AltCopy(Vec<Tag>),
    /// Sentence-start sentinel (targets keep pattern sentinels).
    SentStart,
    /// Sentence-end sentinel.
    SentEnd,
    /// Emit this clitic.
    Clitic(Clitic),
}

/// Errors from loading the rule set or parsing a single rule.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum FrameRulesError {
    /// The TOML text did not match the bucket model.
    #[error("frame-rules TOML did not parse: {0}")]
    Toml(String),
    /// A rule row carried a kind letter outside `r`/`d`/`g`.
    #[error("rule {id}: unknown kind letter {kind:?}")]
    UnknownKind {
        /// The offending rule's id.
        id: String,
        /// The letter found.
        kind: String,
    },
    /// A pattern or target failed to tokenize/parse.
    #[error("pattern element {at}: {message}")]
    Pattern {
        /// Byte offset of the offending token in the raw string.
        at: usize,
        /// What went wrong.
        message: String,
    },
}

fn pattern_error(at: usize, message: impl Into<String>) -> FrameRulesError {
    FrameRulesError::Pattern {
        at,
        message: message.into(),
    }
}

/// Raw lexical token of the pattern grammar, before element
/// classification.
#[derive(Debug, Clone, PartialEq)]
enum Lexeme {
    Quoted(String),
    Word(String),
    GroupOpen,
    GroupClose,
    Pipe,
    Question,
}

/// Splits a raw pattern/target string into lexemes with byte offsets.
fn lex(raw: &str) -> Result<Vec<(usize, Lexeme)>, FrameRulesError> {
    let bytes = raw.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b' ' | b'\t' => i += 1,
            b'"' => {
                let end = raw[i + 1..]
                    .find('"')
                    .map(|off| i + 1 + off)
                    .ok_or_else(|| pattern_error(i, "unterminated quoted literal"))?;
                out.push((i, Lexeme::Quoted(raw[i + 1..end].to_lowercase())));
                i = end + 1;
            }
            b'(' => {
                out.push((i, Lexeme::GroupOpen));
                i += 1;
            }
            b')' => {
                out.push((i, Lexeme::GroupClose));
                i += 1;
            }
            b'|' => {
                out.push((i, Lexeme::Pipe));
                i += 1;
            }
            b'?' => {
                out.push((i, Lexeme::Question));
                i += 1;
            }
            _ => {
                // A bare word: runs to whitespace or a structural
                // character, except that a bracket argument may contain
                // anything up to its closing bracket (e.g.
                // `VBZ[verbalize(NOM)]`).
                let start = i;
                let mut depth = 0usize;
                while i < bytes.len() {
                    match bytes[i] {
                        b'[' => depth += 1,
                        b']' => depth = depth.saturating_sub(1),
                        b' ' | b'\t' | b'?' | b'(' | b')' | b'|' if depth == 0 => break,
                        _ => {}
                    }
                    i += 1;
                }
                out.push((start, Lexeme::Word(raw[start..i].to_string())));
            }
        }
    }
    Ok(out)
}

/// Classifies one bare word as a pattern element.
fn pattern_atom(at: usize, word: &str) -> Result<PatElem, FrameRulesError> {
    if word == "<S>" {
        return Ok(PatElem::SentStart);
    }
    if word == "</S>" {
        return Ok(PatElem::SentEnd);
    }
    if let Some(slot) = Slot::parse(word) {
        return Ok(PatElem::Slot(slot));
    }
    if let Some(clitic) = Clitic::parse(word) {
        return Ok(PatElem::Clitic(clitic));
    }
    if let Some((tag_part, rest)) = word.split_once('[') {
        let tag = Tag::parse(tag_part)
            .ok_or_else(|| pattern_error(at, format!("unknown tag {tag_part:?}")))?;
        let class = rest
            .strip_suffix(']')
            .ok_or_else(|| pattern_error(at, "unterminated [class] bracket"))?;
        return Ok(PatElem::TagClass {
            tag,
            class: class.to_string(),
        });
    }
    if let Some(tag) = Tag::parse(word) {
        return Ok(PatElem::Tag(tag));
    }
    Err(pattern_error(at, format!("unrecognized element {word:?}")))
}

/// Parses a rule's source pattern.
///
/// # Errors
/// Returns [`FrameRulesError::Pattern`] on any unknown tag, unterminated
/// quote/bracket, nested group, or empty alternative.
pub fn parse_pattern(raw: &str) -> Result<Vec<PatElem>, FrameRulesError> {
    let lexemes = lex(raw)?;
    let mut elems: Vec<PatElem> = Vec::new();
    let mut i = 0;
    while i < lexemes.len() {
        let (at, lexeme) = &lexemes[i];
        match lexeme {
            Lexeme::Quoted(text) => {
                // A quoted literal may contain spaces ("seamless
                // integration"): it matches as a phrase, one element
                // per token.
                elems.extend(text.split_whitespace().map(|w| PatElem::Lit(w.to_string())));
                i += 1;
            }
            Lexeme::Word(word) => {
                // Bare alternation without parens (`VB|VBZ`) lexes as
                // word, pipe, word: gather it into a group.
                if matches!(lexemes.get(i + 1), Some((_, Lexeme::Pipe))) {
                    let mut alts = vec![vec![pattern_atom(*at, word)?]];
                    i += 1;
                    while matches!(lexemes.get(i), Some((_, Lexeme::Pipe))) {
                        let Some((next_at, Lexeme::Word(next))) = lexemes.get(i + 1) else {
                            return Err(pattern_error(*at, "dangling | in bare alternation"));
                        };
                        alts.push(vec![pattern_atom(*next_at, next)?]);
                        i += 2;
                    }
                    elems.push(PatElem::Group {
                        alts,
                        optional: false,
                    });
                } else {
                    elems.push(pattern_atom(*at, word)?);
                    i += 1;
                }
            }
            Lexeme::GroupOpen => {
                let (group, consumed) = parse_group(&lexemes[i..], *at)?;
                elems.push(group);
                i += consumed;
            }
            Lexeme::Question => {
                let previous = elems
                    .pop()
                    .ok_or_else(|| pattern_error(*at, "? with no preceding element"))?;
                elems.push(match previous {
                    PatElem::Group { alts, .. } => PatElem::Group {
                        alts,
                        optional: true,
                    },
                    other => PatElem::Optional(Box::new(other)),
                });
                i += 1;
            }
            Lexeme::GroupClose | Lexeme::Pipe => {
                return Err(pattern_error(*at, "structural token outside a group"));
            }
        }
    }
    Ok(elems)
}

/// Parses one parenthesized group starting at `lexemes[0]`
/// (`GroupOpen`); returns the element and how many lexemes it consumed.
fn parse_group(
    lexemes: &[(usize, Lexeme)],
    open_at: usize,
) -> Result<(PatElem, usize), FrameRulesError> {
    let mut alts: Vec<Vec<PatElem>> = vec![Vec::new()];
    let mut i = 1;
    loop {
        let Some((at, lexeme)) = lexemes.get(i) else {
            return Err(pattern_error(open_at, "unterminated group"));
        };
        match lexeme {
            Lexeme::GroupClose => {
                i += 1;
                break;
            }
            Lexeme::Pipe => {
                alts.push(Vec::new());
                i += 1;
            }
            Lexeme::Quoted(text) => {
                alts.last_mut()
                    .expect("alts starts non-empty")
                    .extend(text.split_whitespace().map(|w| PatElem::Lit(w.to_string())));
                i += 1;
            }
            Lexeme::Word(word) => {
                alts.last_mut()
                    .expect("alts starts non-empty")
                    .push(pattern_atom(*at, word)?);
                i += 1;
            }
            Lexeme::Question => {
                let current = alts.last_mut().expect("alts starts non-empty");
                let previous = current
                    .pop()
                    .ok_or_else(|| pattern_error(*at, "? with no preceding element"))?;
                current.push(PatElem::Optional(Box::new(previous)));
                i += 1;
            }
            Lexeme::GroupOpen => {
                return Err(pattern_error(*at, "nested groups are not supported"));
            }
        }
    }
    if alts.iter().any(Vec::is_empty) {
        return Err(pattern_error(open_at, "empty alternative in group"));
    }
    let optional = matches!(lexemes.get(i), Some((_, Lexeme::Question)));
    if optional {
        i += 1;
    }
    Ok((PatElem::Group { alts, optional }, i))
}

/// Parses a rule's target string.
///
/// # Errors
/// Returns [`FrameRulesError::Pattern`] on any construct a target cannot
/// carry (optionals, non-tag group members, unknown words).
pub fn parse_target(raw: &str) -> Result<Target, FrameRulesError> {
    if raw.is_empty() {
        return Ok(Target::Delete);
    }
    if raw == "=" {
        return Ok(Target::Guard);
    }
    let elems = parse_pattern(raw)?;
    let template = elems
        .into_iter()
        .map(target_elem)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Target::Template(template))
}

/// Converts one parsed pattern element into its target meaning.
fn target_elem(elem: PatElem) -> Result<TplElem, FrameRulesError> {
    Ok(match elem {
        PatElem::Lit(text) => TplElem::Lit(text),
        PatElem::Slot(slot) => TplElem::Slot(slot),
        PatElem::Tag(tag) => TplElem::TagCopy(tag),
        PatElem::TagClass { tag, class } => TplElem::TagArg { tag, arg: class },
        PatElem::SentStart => TplElem::SentStart,
        PatElem::SentEnd => TplElem::SentEnd,
        PatElem::Clitic(clitic) => TplElem::Clitic(clitic),
        PatElem::Group { alts, optional } => {
            if optional {
                return Err(pattern_error(0, "optional group in a target"));
            }
            let tags = alts
                .into_iter()
                .map(|alt| match alt.as_slice() {
                    [PatElem::Tag(tag)] => Ok(*tag),
                    _ => Err(pattern_error(
                        0,
                        "target group alternative is not a bare tag",
                    )),
                })
                .collect::<Result<Vec<_>, _>>()?;
            TplElem::AltCopy(tags)
        }
        PatElem::Optional(_) => return Err(pattern_error(0, "optional element in a target")),
    })
}

/// Extracts a pattern's literal probes for corpus adjudication: the
/// longest obligatory run of consecutive literal-bearing elements,
/// class positions expanded to their members.
///
/// Optional elements, groups, tags, slots, and sentinels all break a
/// run. They make the tokens around them non-adjacent in at least one
/// realization. Returns one probe (a lemma sequence) per class-member
/// combination of the winning run, leftmost run winning ties; the
/// expansion is capped at `max_probes` (further members are dropped
/// deterministically, in member order). A pattern with no obligatory
/// literal content returns no probes.
pub fn literal_probes(
    pattern: &[PatElem],
    classes: &BTreeMap<String, Vec<String>>,
    max_probes: usize,
) -> Vec<Vec<String>> {
    let mut best: &[PatElem] = &[];
    let mut start = 0;
    for (i, elem) in pattern.iter().enumerate() {
        let breaks = !matches!(elem, PatElem::Lit(_) | PatElem::TagClass { .. });
        if breaks {
            if i - start > best.len() {
                best = &pattern[start..i];
            }
            start = i + 1;
        }
    }
    if pattern.len() - start > best.len() {
        best = &pattern[start..];
    }
    let mut probes: Vec<Vec<String>> = vec![Vec::new()];
    for elem in best {
        match elem {
            PatElem::Lit(text) => {
                for probe in &mut probes {
                    probe.push(text.clone());
                }
            }
            PatElem::TagClass { class, .. } => {
                let members = classes.get(class).cloned().unwrap_or_default();
                if members.is_empty() {
                    // Unknown or empty class: the position matches
                    // nothing knowable — treat as a run break by
                    // discarding the probes (conservative: the caller
                    // sees no probe rather than a wrong one).
                    return Vec::new();
                }
                probes = probes
                    .iter()
                    .flat_map(|probe| {
                        members.iter().map(move |member| {
                            let mut extended = probe.clone();
                            // Multi-word members are stored hyphenated.
                            // They match as phrases.
                            extended.extend(member.split('-').map(str::to_string));
                            extended
                        })
                    })
                    .take(max_probes)
                    .collect();
            }
            _ => unreachable!("run contains only Lit/TagClass"),
        }
    }
    probes.retain(|probe| !probe.is_empty());
    probes
}

/// An adjudication verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// Machine-tilted at ratio ≥ 2 with ≥ 2 machine occurrences.
    Confirmed,
    /// A guard whose human side is at least as dense as its machine
    /// side. The protection is warranted.
    GuardConfirmed,
    /// A guard measured machine-tilted: protecting it would shelter
    /// machine-tilted text. Compiles as report-only, never as a guard.
    GuardWrong,
    /// Human-tilted at ratio ≤ 0.67: rewriting away from it would push
    /// text toward the machine register.
    DirectionError,
    /// Differential too small (or a single unmatched
    /// machine hit).
    Weak,
    /// Unobserved on both corpus sides.
    NoEvidence,
}

/// Classifies one rule's measured evidence. `m`/`h` are raw occurrence
/// counts; `m_total`/`h_total` are the corpus sides' token counts.
#[must_use]
pub fn classify(kind: RuleKind, m: u64, h: u64, m_total: u64, h_total: u64) -> Verdict {
    if m == 0 && h == 0 {
        return Verdict::NoEvidence;
    }
    #[expect(
        clippy::cast_precision_loss,
        reason = "corpus counts are far below 2^52"
    )]
    let per_million = |count: u64, total: u64| {
        if total == 0 {
            0.0
        } else {
            count as f64 / total as f64 * 1_000_000.0
        }
    };
    let m_pm = per_million(m, m_total);
    let h_pm = per_million(h, h_total);
    if kind == RuleKind::Guard {
        return if h > 0 && m_pm / h_pm <= 1.0 {
            Verdict::GuardConfirmed
        } else {
            Verdict::GuardWrong
        };
    }
    let ratio = if h_pm > 0.0 {
        m_pm / h_pm
    } else {
        f64::INFINITY
    };
    if ratio >= 2.0 && m >= 2 {
        Verdict::Confirmed
    } else if ratio <= 0.67 {
        Verdict::DirectionError
    } else {
        Verdict::Weak
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shipped rule-set artifact itself: parses, holds every rule
    /// the original adjudication delivered (bucket membership moves
    /// with the evidence — `corpus-tool adjudicate --regenerate` — but
    /// rules are never lost), and keeps its class and function-word
    /// tables intact.
    #[test]
    fn shipped_rule_set_parses_and_keeps_every_rule() {
        let set = FrameRuleSet::parse(include_str!("../packs/frame-rules-en-v1.toml"))
            .expect("shipped frame-rules-en-v1.toml must parse");
        let total = set.rules_ship.len()
            + set.rules_flip.len()
            + set.rules_surface.len()
            + set.rules_pilot.len()
            + set.rules_quarantine.len()
            + set.rules_no_evidence.len()
            + set.rules_staged_surface.len();
        // 3,333 delivered rules plus 55 authored arrivals (the
        // honestly/honest/load-bearing batch and two corpus-mined
        // batches) plus the five-way verb-tag split of vsub.harness
        // (net +4) minus two stale report-only holds
        // (col.unwavering-commitment, col.seamless-integration) that
        // duplicated already-shipped adjective-deletion rewrites on
        // the identical byte range, plus the five-way detection-only
        // vsub.surface family (the verb "surface", transitivity-mixed,
        // measured after the sonnet-5 corpus round) — bumped only when
        // rules are deliberately added or removed, never by
        // regeneration. 3395 -> 3394: `able.ensures-that-short::ensure`
        // retired from `rules_ship` (its template could never realize --
        // see this file's own `frame-rules-en-v1.toml` comment at the
        // retired row) once `friction-edit`'s restructure pass took over
        // its evidence (`restructure.ensures_that`). 3394 -> 3398: the
        // sentence-initial discourse-marker audit (see the pack's own
        // "SENTENCE-INITIAL DISCOURSE-MARKER AUDIT" preamble comment)
        // measured four additive-connector openers with no prior row —
        // similarly/likewise/what-is-more/in-addition — and staged all
        // four under `conn.additive` in `rules_staged_surface`; every
        // one measures human-tilted or no-evidence, so none compile.
        // 3398 -> 3402: adv.deliberately-{parenthetical,pre-colon,trailing}
        // + advg.deliberately (grep-measured 27x machine tilt).
        // 3402 -> 3412: col.load-bearing-x's ten collocation-scoped
        // rewrites (metaphorical heads only; the literal architectural
        // sense can never match).
        // 3412 -> 3414: int.exactly-what + intg.exactly (exactly-the-same
        // was tried and re-measured human-tilted by the compile fence —
        // see the pack row's own note).
        assert_eq!(total, 3414, "regeneration moves rules, never loses them");
        assert_eq!(set.rules_pilot.len(), 21, "pilot rules never move");
        assert_eq!(set.classes.len(), 35);
        assert_eq!(set.function_words.words.len(), 37);
        for (name, rules) in set.knowledge_buckets() {
            assert!(!rules.is_empty(), "bucket {name} must not be empty");
        }
    }

    /// Every pattern and target in the four compiling buckets parses.
    /// (Staged buckets are allowed to carry unparseable rows: those
    /// become per-rule rejections, not load failures.)
    #[test]
    fn every_compiling_bucket_pattern_and_target_parses() {
        let set = FrameRuleSet::parse(include_str!("../packs/frame-rules-en-v1.toml"))
            .expect("shipped frame-rules-en-v1.toml must parse");
        for rule in [&set.rules_ship, &set.rules_flip, &set.rules_surface]
            .into_iter()
            .flatten()
        {
            parse_pattern(&rule.p)
                .unwrap_or_else(|e| panic!("rule {} pattern {:?}: {e}", rule.id, rule.p));
            for t in &rule.t {
                parse_target(t).unwrap_or_else(|e| panic!("rule {} target {t:?}: {e}", rule.id));
            }
        }
    }

    /// The grammar corner cases the rule set actually uses.
    #[test]
    fn pattern_grammar_parses_the_observed_constructs() {
        // Multi-element optional group with a literal and a slot.
        let p = parse_pattern(r#"<S> "look" "no" "further" ("than" X)? PUNCT </S>"#)
            .expect("group pattern must parse");
        assert!(matches!(
            p.as_slice(),
            [
                PatElem::SentStart,
                PatElem::Lit(_),
                PatElem::Lit(_),
                PatElem::Lit(_),
                PatElem::Group { optional: true, .. },
                PatElem::Tag(Tag::Punct),
                PatElem::SentEnd,
            ]
        ));

        // Bare alternation without parens.
        let p = parse_pattern(r#""gracefully" VB|VBZ"#).expect("bare alternation must parse");
        assert_eq!(
            p[1],
            PatElem::Group {
                alts: vec![vec![PatElem::Tag(Tag::Vb)], vec![PatElem::Tag(Tag::Vbz)]],
                optional: false,
            }
        );

        // Optional literal, class position, clitic, bracket-with-parens.
        let p = parse_pattern(r#"BE "designed" "to" "provide" DT "comprehensive"? "overview""#)
            .expect("optional literal must parse");
        assert!(
            matches!(&p[5], PatElem::Optional(inner) if **inner == PatElem::Lit("comprehensive".into()))
        );
        let p = parse_pattern(r#""now" CLIT_VE? VBN[cover-class] X"#).expect("clitic class");
        assert!(
            matches!(&p[1], PatElem::Optional(inner) if **inner == PatElem::Clitic(Clitic::Ve))
        );
        assert_eq!(
            p[2],
            PatElem::TagClass {
                tag: Tag::Vbn,
                class: "cover-class".into()
            }
        );
        let p = parse_pattern("VBZ[verbalize(NOM)]").expect("nested parens inside bracket");
        assert_eq!(
            p[0],
            PatElem::TagClass {
                tag: Tag::Vbz,
                class: "verbalize(NOM)".into()
            }
        );
    }

    /// Target parsing distinguishes guard, delete, copies, and
    /// realization arguments.
    #[test]
    fn target_grammar_parses_the_observed_constructs() {
        assert_eq!(parse_target("="), Ok(Target::Guard));
        assert_eq!(parse_target(""), Ok(Target::Delete));
        let Target::Template(t) = parse_target(r"VBZ[cover] X").expect("template") else {
            panic!("expected template");
        };
        assert_eq!(
            t[0],
            TplElem::TagArg {
                tag: Tag::Vbz,
                arg: "cover".into()
            }
        );
        assert_eq!(t[1], TplElem::Slot(Slot::X));
        let Target::Template(t) = parse_target("(ADJ|VB)").expect("alt copy") else {
            panic!("expected template");
        };
        assert_eq!(t[0], TplElem::AltCopy(vec![Tag::Adj, Tag::Vb]));
        let Target::Template(t) = parse_target("NNS").expect("tag copy") else {
            panic!("expected template");
        };
        assert_eq!(t[0], TplElem::TagCopy(Tag::Nns));
    }

    /// Wraps one `rules_ship` array literal in the rest of the bucket
    /// scaffold, mirroring `frame_compile`'s own `tiny_set` test helper.
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
[function_words]
words = ["a", "an", "the", "to", "of"]
"#
        );
        FrameRuleSet::parse(&toml).expect("tiny set must parse")
    }

    /// A bare TOML string `t` — every rule authored so far — parses as
    /// the one-element ladder the compiler treats identically to today.
    #[test]
    fn bare_string_target_parses_as_a_one_element_ladder() {
        let set = tiny_set(
            r#"rules_ship = [
{ id="vsub.utilize", p="\"utilize\" X", t="\"use\" X", k="r", m_pm=100.0, h_pm=10.0, v="CONFIRMED" },
]"#,
        );
        assert_eq!(set.rules_ship[0].t, vec!["\"use\" X".to_string()]);
    }

    /// An array `t` parses as an ordered ladder, primary first.
    #[test]
    fn array_target_parses_as_an_ordered_ladder() {
        let set = tiny_set(
            r#"rules_ship = [
{ id="vsub.utilize", p="\"utilize\" X", t=["\"use\" X", "\"apply\" X"], k="r", m_pm=100.0, h_pm=10.0, v="CONFIRMED" },
]"#,
        );
        assert_eq!(
            set.rules_ship[0].t,
            vec!["\"use\" X".to_string(), "\"apply\" X".to_string()]
        );
    }

    /// An empty array `t` is a load error: a rule needs a primary
    /// target, never zero.
    #[test]
    fn empty_array_target_is_a_load_error() {
        let toml = r#"
rules_ship = [
{ id="vsub.utilize", p="\"utilize\" X", t=[], k="r", m_pm=100.0, h_pm=10.0, v="CONFIRMED" },
]
rules_flip = []
rules_surface = []
rules_pilot = []
rules_quarantine = []
rules_no_evidence = []
rules_staged_surface = []
[classes]
[function_words]
words = []
"#;
        assert!(FrameRuleSet::parse(toml).is_err());
    }

    /// Probe extraction picks the longest obligatory literal run and
    /// expands class positions.
    #[test]
    fn literal_probes_pick_longest_run_and_expand_classes() {
        let classes = BTreeMap::from([(
            "note_v".to_string(),
            vec!["note".to_string(), "point-out".to_string()],
        )]);
        let p = parse_pattern(r#""it" BE "important" "to" X"#).expect("pattern");
        assert_eq!(
            literal_probes(&p, &classes, 64),
            vec![vec!["important".to_string(), "to".to_string()]]
        );
        // Class expansion, hyphenated member as phrase.
        let p = parse_pattern(r#""duly" VB[note_v]"#).expect("pattern");
        assert_eq!(
            literal_probes(&p, &classes, 64),
            vec![
                vec!["duly".to_string(), "note".to_string()],
                vec!["duly".to_string(), "point".to_string(), "out".to_string()],
            ]
        );
        // Optional elements break runs.
        let p = parse_pattern(r#""a" "comprehensive"? "overview""#).expect("pattern");
        assert_eq!(
            literal_probes(&p, &classes, 64),
            vec![vec!["a".to_string()]]
        );
        // No literal content -> no probes.
        let p = parse_pattern("DT NN").expect("pattern");
        assert!(literal_probes(&p, &classes, 64).is_empty());
    }

    /// The verdict thresholds, pinned at their boundaries: two machine
    /// occurrences against zero human confirm; one does not.
    #[test]
    fn classify_pins_the_adjudication_boundaries() {
        use RuleKind::{Guard, Rewrite};
        let m_total = 340_000;
        let h_total = 340_000;
        assert_eq!(
            classify(Rewrite, 0, 0, m_total, h_total),
            Verdict::NoEvidence
        );
        assert_eq!(
            classify(Rewrite, 2, 0, m_total, h_total),
            Verdict::Confirmed
        );
        assert_eq!(classify(Rewrite, 1, 0, m_total, h_total), Verdict::Weak);
        assert_eq!(
            classify(Rewrite, 10, 4, m_total, h_total),
            Verdict::Confirmed
        );
        assert_eq!(classify(Rewrite, 10, 6, m_total, h_total), Verdict::Weak);
        assert_eq!(
            classify(Rewrite, 2, 3, m_total, h_total),
            Verdict::DirectionError
        );
        assert_eq!(
            classify(Guard, 5, 5, m_total, h_total),
            Verdict::GuardConfirmed
        );
        assert_eq!(classify(Guard, 6, 5, m_total, h_total), Verdict::GuardWrong);
        assert_eq!(classify(Guard, 3, 0, m_total, h_total), Verdict::GuardWrong);
        assert_eq!(classify(Guard, 0, 0, m_total, h_total), Verdict::NoEvidence);
        // Different corpus sizes: rates, not raw counts, set the ratio.
        assert_eq!(
            classify(Rewrite, 4, 4, 340_000, 680_000),
            Verdict::Confirmed
        );
    }
}
