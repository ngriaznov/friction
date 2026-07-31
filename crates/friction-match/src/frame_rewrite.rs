//! Frame-rewrite matching: anchored scan and pattern verification over
//! tagged sentences, against the compiled frame pack.
//!
//! This module is the *detection* half of the frame-rewrite operation:
//! a pure function of one tagged sentence plus the pack — no edits, no
//! gates, no splicing (those live in `friction-edit`, which consumes
//! the [`FrameMatch`]es this module returns).
//!
//! # Scan shape
//!
//! Every compiled rule carries an anchor: its longest obligatory
//! literal run, as interned word ids. [`FrameIndex`] buckets rules by
//! their anchor's first word id; the per-sentence scan resolves each
//! token to word ids once (lemma and lowercased surface — a literal
//! matches either, the same equivalence the adjudication referee
//! counts by), walks the sentence left to right, and only attempts
//! full verification for rules whose anchor could start at the current
//! token. Anchor candidacy is a few binary searches per token; full
//! verification only runs on plausible sites.
//!
//! # Verification semantics
//!
//! The verifier walks the rule's pattern ops with a cursor, anchored so
//! the anchor run lands on the candidate tokens, with a fixed,
//! deterministic policy everywhere the grammar allows choice:
//!
//! - **Optionals are greedy**: an optional element consumes its token
//!   when it matches there, and is skipped otherwise (never both — no
//!   optional-driven backtracking).
//! - **Alternatives are first-match**: a group tries its alternatives
//!   in written order and commits to the first that matches.
//! - **Slots are lazy**: `X`/`Y`/`Z` capture the shortest non-empty
//!   token run that lets the rest of the pattern match, found by
//!   linear extension (each extension re-tries the pattern tail once —
//!   patterns are a handful of elements, so this stays trivially
//!   cheap).
//! - **Clitics match fused or free**: the tokenizer keeps contractions
//!   as one word ("we've"), so a clitic op matches zero-width when the
//!   previously matched token ends with its surface, or consumes one
//!   token whose surface is exactly the clitic.
//!
//! # Conflict policy
//!
//! [`resolve`] applies the fixed policy over a sentence's raw matches:
//! leftmost first, then longest, then higher support, then rule id
//! ascending — and the survivors are non-overlapping, kept left to
//! right. No scoring, no search: the policy is a total order, so the
//! result is deterministic for any input.

use std::collections::BTreeMap;
use std::ops::Range;

use friction_nlp::TaggedToken;
use friction_packs::frame_bin::{FramePackView, PatOp, RuleView};
use friction_packs::frame_rules::{Clitic, Slot, Tag};

/// One verified frame-rule match within a single sentence.
#[derive(Debug, Clone)]
pub struct FrameMatch {
    /// Index of the rule in the pack.
    pub rule_index: u32,
    /// The matched token range within the sentence (exclusive end).
    pub tokens: Range<usize>,
    /// The matched byte range, absolute into the document source.
    pub bytes: Range<usize>,
    /// Captured slot token ranges, indexed by [`Slot::code`].
    pub slots: [Option<Range<usize>>; 3],
    /// For each pattern op index, the sentence token it matched (ops
    /// that consumed no token — sentinels, skipped optionals, group
    /// delimiters, fused clitics — hold `None`).
    pub op_tokens: Vec<Option<usize>>,
    /// For each alternation group ordinal, the single token its
    /// matched alternative consumed, when it consumed exactly one.
    pub group_tokens: Vec<Option<usize>>,
}

/// Anchor-first lookup over a frame pack: rule indexes bucketed by
/// their anchor's first word id. Built once per process.
#[derive(Debug)]
pub struct FrameIndex {
    by_first_anchor_id: BTreeMap<u32, Vec<u32>>,
}

impl FrameIndex {
    /// Builds the index over every rule in `view`.
    #[must_use]
    pub fn build(view: &FramePackView<'_>) -> Self {
        let mut by_first_anchor_id: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
        for index in 0..view.rule_count() {
            let rule = view.rule(index).expect("index in range");
            if let Some(first) = rule.anchor.clone().next() {
                by_first_anchor_id.entry(first).or_default().push(index);
            }
        }
        Self { by_first_anchor_id }
    }

    /// The rules whose anchor starts with `word_id`.
    fn candidates(&self, word_id: u32) -> &[u32] {
        self.by_first_anchor_id
            .get(&word_id)
            .map_or(&[], Vec::as_slice)
    }
}

/// One sentence token resolved against the pack's interner.
struct ResolvedToken {
    /// Interner id of the lowercased surface, if interned — the only
    /// id surface-matched (pilot) rules may use: their targets are
    /// literal inflected forms, so a lemma-level match would splice a
    /// wrong tense.
    surface_id: Option<u32>,
    /// Every id this token can stand for at lemma level: the tagged
    /// lemma, the surface, and tag-independent inflection reductions
    /// (`-ed`/`-ing`/`-s` stems). The reductions cover tagger
    /// mis-tags — "utilized" tagged `VBP` keeps its surface as its
    /// lemma, and without the fallback a lemma-matched literal rule
    /// would silently miss it.
    ids: Vec<u32>,
}

impl ResolvedToken {
    fn matches(&self, id: u32, surface_only: bool) -> bool {
        if surface_only {
            self.surface_id == Some(id)
        } else {
            self.ids.contains(&id)
        }
    }
}

/// Scans one tagged sentence for frame-rule matches.
///
/// Returns raw matches (all rules, all positions) — callers apply
/// [`resolve`] for the non-overlapping edit set, and may keep raw
/// matches for report-only findings.
///
/// `text` is the full document source (token spans are absolute).
#[must_use]
pub fn scan_sentence(
    view: &FramePackView<'_>,
    index: &FrameIndex,
    tokens: &[TaggedToken],
    text: &str,
) -> Vec<FrameMatch> {
    let resolved: Vec<ResolvedToken> = tokens
        .iter()
        .map(|token| {
            let surface = surface_of(token, text).to_lowercase();
            let surface_id = view.word_id(&surface);
            let mut ids: Vec<u32> = surface_id.into_iter().collect();
            let mut push = |word: &str| {
                if let Some(id) = view.word_id(word)
                    && !ids.contains(&id)
                {
                    ids.push(id);
                }
            };
            push(&token.lemma.to_lowercase());
            // Irregular forms reduce through the shared tables...
            for pos in ["VBD", "VBG", "VBZ"] {
                push(&friction_nlp::lemmatize(&surface, pos));
            }
            // ...but the regular reverse lemmatization there keeps ONE
            // stem per form, and when two stems both round-trip
            // ("commenc"/"commence" for "commenced") it can keep the
            // wrong one. Here the pack's own interner is the
            // dictionary: generate every plausible suffix reduction
            // and let interner membership arbitrate — a stem that is
            // no rule's word can never matter to a scan, and a stem
            // that is one is exactly the reading the rule wants.
            for candidate in suffix_reductions(&surface) {
                push(&candidate);
            }
            ResolvedToken { surface_id, ids }
        })
        .collect();

    let mut matches = Vec::new();
    for at in 0..tokens.len() {
        let mut tried: Vec<u32> = Vec::new();
        for id in resolved[at].ids.clone() {
            for &rule_index in index.candidates(id) {
                if tried.contains(&rule_index) {
                    continue;
                }
                tried.push(rule_index);
                let rule = view.rule(rule_index).expect("indexed rule in range");
                if anchor_fits(&rule, &resolved, at)
                    && let Some(m) = verify(view, &rule, rule_index, tokens, &resolved, text, at)
                {
                    matches.push(m);
                }
            }
        }
    }
    // A token whose lemma and surface both resolve and both index the
    // same rule can verify it twice; keep the first of any duplicates.
    matches.dedup_by(|a, b| a.rule_index == b.rule_index && a.tokens == b.tokens);
    matches
}

/// Applies the conflict policy over a sentence's raw matches.
///
/// Leftmost, then longest, then support descending, then rule id
/// ascending; survivors are non-overlapping, committed left to right.
/// Only edit-capable kinds should compete — callers filter guards and
/// report-only rules first.
#[must_use]
pub fn resolve(view: &FramePackView<'_>, mut matches: Vec<FrameMatch>) -> Vec<FrameMatch> {
    matches.sort_by(|a, b| {
        a.bytes
            .start
            .cmp(&b.bytes.start)
            .then_with(|| b.bytes.end.cmp(&a.bytes.end))
            .then_with(|| {
                let support =
                    |m: &FrameMatch| view.rule(m.rule_index).expect("rule in range").support;
                support(b).cmp(&support(a))
            })
            .then_with(|| {
                let id = |m: &FrameMatch| view.rule(m.rule_index).expect("rule in range").id;
                id(a).cmp(id(b))
            })
    });
    let mut kept: Vec<FrameMatch> = Vec::new();
    for candidate in matches {
        let overlaps = kept
            .iter()
            .any(|k| candidate.bytes.start < k.bytes.end && k.bytes.start < candidate.bytes.end);
        if !overlaps {
            kept.push(candidate);
        }
    }
    kept
}

/// The token's surface text.
fn surface_of<'a>(token: &TaggedToken, text: &'a str) -> &'a str {
    &text[token.token.range.clone()]
}

/// Every plausible base-form reduction of a lowercased surface:
/// `-ied`/`-ed`/`-ing` (with doubled-consonant undoubling and `-e`
/// restoration) and `-ies`/`-es`/`-s`. Deliberately over-generates —
/// the caller keeps only candidates the pack's interner knows.
fn suffix_reductions(surface: &str) -> Vec<String> {
    let mut out = Vec::new();
    let undoubled = |stem: &str| {
        let chars: Vec<char> = stem.chars().collect();
        let n = chars.len();
        (n >= 2 && chars[n - 1] == chars[n - 2] && !"aeiou".contains(chars[n - 1]))
            .then(|| chars[..n - 1].iter().collect::<String>())
    };
    if let Some(stem) = surface.strip_suffix("ied") {
        out.push(format!("{stem}y"));
    }
    for suffix in ["ed", "ing"] {
        if let Some(stem) = surface.strip_suffix(suffix) {
            if let Some(un) = undoubled(stem) {
                out.push(un);
            }
            out.push(stem.to_string());
            out.push(format!("{stem}e"));
        }
    }
    if let Some(stem) = surface.strip_suffix("ies") {
        out.push(format!("{stem}y"));
    }
    if let Some(stem) = surface.strip_suffix("es") {
        out.push(stem.to_string());
    }
    if let Some(stem) = surface.strip_suffix('s') {
        out.push(stem.to_string());
    }
    out
}

/// Whether `rule`'s full anchor id run matches starting at token `at`.
fn anchor_fits(rule: &RuleView<'_>, resolved: &[ResolvedToken], at: usize) -> bool {
    rule.anchor.clone().enumerate().all(|(offset, id)| {
        resolved
            .get(at + offset)
            .is_some_and(|token| token.matches(id, rule.surface_match))
    })
}

/// A decoded pattern op with its flags, position-addressable.
struct DecodedOp {
    op: PatOp,
    optional: bool,
}

/// The verifier's mutable capture state.
struct Captures {
    slots: [Option<Range<usize>>; 3],
    op_tokens: Vec<Option<usize>>,
    group_tokens: Vec<Option<usize>>,
}

/// The immutable state one verification walk shares: the rule's ops,
/// the sentence, and the anchor pin.
struct Verifier<'a> {
    view: &'a FramePackView<'a>,
    ops: &'a [DecodedOp],
    tokens: &'a [TaggedToken],
    resolved: &'a [ResolvedToken],
    text: &'a str,
    surface_only: bool,
    anchor_pin: (usize, usize),
}

/// Verifies `rule` with its anchor starting at token `anchor_at`;
/// builds the match on success.
fn verify(
    view: &FramePackView<'_>,
    rule: &RuleView<'_>,
    rule_index: u32,
    tokens: &[TaggedToken],
    resolved: &[ResolvedToken],
    text: &str,
    anchor_at: usize,
) -> Option<FrameMatch> {
    let ops: Vec<DecodedOp> = rule
        .pattern
        .clone()
        .map(|cell| {
            let (op, optional) = cell.expect("embedded pack ops always decode");
            DecodedOp { op, optional }
        })
        .collect();
    // `anchor_elems` counts pattern *elements*; groups flatten to
    // multiple cells, but an anchor is always a run of plain literal
    // cells outside any group, so walking cells counting element
    // starts recovers the cell index.
    let anchor_op = element_to_op_index(&ops, rule.anchor_elems.0)?;

    // The ops before the anchor must match ending exactly where the
    // anchor starts. Slots make that prefix variable-width, so try
    // every plausible start position, latest (shortest prefix) first,
    // requiring the anchor to land on `anchor_at`.
    let earliest = anchor_at.saturating_sub(max_prefix_tokens(&ops, anchor_op));
    for start in (earliest..=anchor_at).rev() {
        let verifier = Verifier {
            view,
            ops: &ops,
            tokens,
            resolved,
            text,
            surface_only: rule.surface_match,
            anchor_pin: (anchor_op, anchor_at),
        };
        let mut captures = Captures {
            slots: [None, None, None],
            op_tokens: vec![None; ops.len()],
            group_tokens: Vec::new(),
        };
        if let Some(end) = verifier.match_ops(0, start, &mut captures) {
            let first_token = captures.op_tokens.iter().flatten().min().copied();
            let last_token = captures.op_tokens.iter().flatten().max().copied();
            let (Some(first), Some(last)) = (first_token, last_token) else {
                return None;
            };
            return Some(FrameMatch {
                rule_index,
                tokens: first..end,
                bytes: tokens[first].token.range.start..tokens[last].token.range.end,
                slots: captures.slots,
                op_tokens: captures.op_tokens,
                group_tokens: captures.group_tokens,
            });
        }
    }
    None
}

/// Walks op cells to find the cell index of pattern element
/// `element` (0-based over elements, where one group = one element).
fn element_to_op_index(ops: &[DecodedOp], element: u16) -> Option<usize> {
    let mut element_index = 0u16;
    let mut i = 0;
    while i < ops.len() {
        if element_index == element {
            return Some(i);
        }
        match ops[i].op {
            PatOp::GroupStart { .. } => {
                let mut depth = 1;
                i += 1;
                while i < ops.len() && depth > 0 {
                    match ops[i].op {
                        PatOp::GroupStart { .. } => depth += 1,
                        PatOp::GroupEnd => depth -= 1,
                        _ => {}
                    }
                    i += 1;
                }
            }
            _ => i += 1,
        }
        element_index += 1;
    }
    None
}

/// Upper bound on tokens the ops before `anchor_op` can consume: one
/// token per op, plus a fixed allowance when a slot sits in the
/// prefix (slots are bounded by the sentence anyway; a small cap
/// keeps the backward walk cheap).
fn max_prefix_tokens(ops: &[DecodedOp], anchor_op: usize) -> usize {
    let has_slot = ops[..anchor_op]
        .iter()
        .any(|op| matches!(op.op, PatOp::Slot(_)));
    anchor_op + if has_slot { 12 } else { 0 }
}

impl Verifier<'_> {
    /// Matches `ops[op_at..]` against tokens starting at `token_at`;
    /// returns the token position after the last consumed token on
    /// success. The anchor pin requires the op at `anchor_pin.0` to
    /// consume the token at `anchor_pin.1`.
    fn match_ops(&self, op_at: usize, token_at: usize, captures: &mut Captures) -> Option<usize> {
        let Some(decoded) = self.ops.get(op_at) else {
            return Some(token_at);
        };
        if op_at == self.anchor_pin.0 && token_at != self.anchor_pin.1 {
            return None;
        }
        match &decoded.op {
            PatOp::SentStart => (token_at == 0)
                .then(|| self.match_ops(op_at + 1, token_at, captures))
                .flatten(),
            PatOp::SentEnd => {
                // Trailing sentence punctuation does not block the
                // sentinel.
                let rest_is_punct = self.tokens[token_at..]
                    .iter()
                    .all(|t| !surface_of(t, self.text).chars().any(char::is_alphanumeric));
                rest_is_punct
                    .then(|| self.match_ops(op_at + 1, self.tokens.len(), captures))
                    .flatten()
            }
            PatOp::Slot(slot) => self.match_slot(op_at, token_at, *slot, captures),
            PatOp::GroupStart { .. } => self.match_group(op_at, token_at, captures),
            PatOp::AltSep | PatOp::GroupEnd => {
                unreachable!("group delimiters are consumed by match_group")
            }
            PatOp::Clitic(clitic) => self.match_clitic(op_at, token_at, *clitic, captures),
            PatOp::Lit(_) | PatOp::Tag(_) | PatOp::Class { .. } => {
                if self.single_op_matches(decoded.op, token_at) {
                    captures.op_tokens[op_at] = Some(token_at);
                    let advanced = self.match_ops(op_at + 1, token_at + 1, captures);
                    if advanced.is_some() {
                        return advanced;
                    }
                    captures.op_tokens[op_at] = None;
                }
                if decoded.optional {
                    // Greedy already tried consuming; fall back to
                    // skipping.
                    return self.match_ops(op_at + 1, token_at, captures);
                }
                None
            }
        }
    }

    /// Whether one single-token op matches the token at `at`.
    fn single_op_matches(&self, op: PatOp, at: usize) -> bool {
        if at >= self.tokens.len() {
            return false;
        }
        match op {
            PatOp::Lit(id) => self.resolved[at].matches(id, self.surface_only),
            PatOp::Tag(tag) => tag_matches(tag, &self.tokens[at], self.text),
            PatOp::Class { tag, class } => {
                tag_matches(tag, &self.tokens[at], self.text)
                    && self.view.class_members(class).is_some_and(|members| {
                        members.iter().any(|member| match member.as_slice() {
                            // Multi-word class members match as
                            // phrases; a single-token position can only
                            // satisfy a one-word member.
                            [only] => self.resolved[at].matches(*only, self.surface_only),
                            _ => false,
                        })
                    })
            }
            _ => false,
        }
    }

    /// Lazy slot matching: capture the shortest non-empty token run
    /// that lets the pattern tail match.
    fn match_slot(
        &self,
        op_at: usize,
        token_at: usize,
        slot: Slot,
        captures: &mut Captures,
    ) -> Option<usize> {
        for end in (token_at + 1)..=self.tokens.len() {
            let slot_index = usize::from(slot.code());
            let previous = captures.slots[slot_index].replace(token_at..end);
            captures.op_tokens[op_at] = Some(end - 1);
            if let Some(after) = self.match_ops(op_at + 1, end, captures) {
                return Some(after);
            }
            captures.op_tokens[op_at] = None;
            captures.slots[slot_index] = previous;
        }
        None
    }

    /// Group matching: first alternative that lets the tail match wins.
    fn match_group(&self, op_at: usize, token_at: usize, captures: &mut Captures) -> Option<usize> {
        let optional = self.ops[op_at].optional;
        let ordinal = captures.group_tokens.len();
        captures.group_tokens.push(None);
        let mut alt_ranges: Vec<Range<usize>> = Vec::new();
        let mut alt_start = op_at + 1;
        let mut i = op_at + 1;
        let group_end = loop {
            match self.ops.get(i).map(|d| &d.op) {
                Some(PatOp::AltSep) => {
                    alt_ranges.push(alt_start..i);
                    alt_start = i + 1;
                    i += 1;
                }
                Some(PatOp::GroupEnd) => {
                    alt_ranges.push(alt_start..i);
                    break i;
                }
                Some(_) => i += 1,
                None => unreachable!("compiled groups always close"),
            }
        };
        for alt in &alt_ranges {
            if let Some(after_alt) = self.match_alt_run(alt.clone(), token_at, captures) {
                if after_alt == token_at + 1 {
                    captures.group_tokens[ordinal] = Some(token_at);
                }
                if let Some(end) = self.match_ops(group_end + 1, after_alt, captures) {
                    return Some(end);
                }
                captures.group_tokens[ordinal] = None;
            }
        }
        if optional && let Some(end) = self.match_ops(group_end + 1, token_at, captures) {
            return Some(end);
        }
        captures.group_tokens.pop();
        None
    }

    /// Matches one group alternative's ops as a consecutive run (no
    /// slots or nested groups inside alternatives — the grammar
    /// forbids them).
    fn match_alt_run(
        &self,
        range: Range<usize>,
        token_at: usize,
        captures: &mut Captures,
    ) -> Option<usize> {
        let mut pos = token_at;
        let mut consumed: Vec<usize> = Vec::new();
        for i in range {
            let decoded = &self.ops[i];
            if self.single_op_matches(decoded.op, pos) {
                captures.op_tokens[i] = Some(pos);
                consumed.push(i);
                pos += 1;
            } else if !decoded.optional {
                for &j in &consumed {
                    captures.op_tokens[j] = None;
                }
                return None;
            }
        }
        Some(pos)
    }

    /// Clitic matching: zero-width when the previous token's surface
    /// ends with the clitic (fused contraction), one token when a
    /// standalone token is exactly the clitic.
    fn match_clitic(
        &self,
        op_at: usize,
        token_at: usize,
        clitic: Clitic,
        captures: &mut Captures,
    ) -> Option<usize> {
        let surface = clitic.surface();
        let standalone = self
            .tokens
            .get(token_at)
            .is_some_and(|t| surface_of(t, self.text).eq_ignore_ascii_case(surface));
        if standalone {
            captures.op_tokens[op_at] = Some(token_at);
            if let Some(end) = self.match_ops(op_at + 1, token_at + 1, captures) {
                return Some(end);
            }
            captures.op_tokens[op_at] = None;
        }
        let fused = token_at > 0
            && surface_of(&self.tokens[token_at - 1], self.text)
                .to_lowercase()
                .ends_with(surface);
        if fused || self.ops[op_at].optional {
            return self.match_ops(op_at + 1, token_at, captures);
        }
        None
    }
}

/// Whether a pack tag admits this token.
fn tag_matches(tag: Tag, token: &TaggedToken, text: &str) -> bool {
    let pos = token.pos.as_str();
    let surface = surface_of(token, text);
    match tag {
        Tag::Nn => pos == "NN" || pos == "NNP",
        Tag::Nns => pos == "NNS" || pos == "NNPS",
        Tag::Vb => pos == "VB" || pos == "VBP",
        Tag::Vbz => pos == "VBZ",
        Tag::Vbg => pos == "VBG",
        Tag::Vbn => pos == "VBN",
        Tag::Vbd => pos == "VBD",
        Tag::Adj => pos == "JJ" || pos == "JJR",
        Tag::Adjs => pos == "JJS",
        Tag::Adv => pos == "RB" || pos == "RBR" || pos == "RBS",
        Tag::Dt => pos == "DT",
        Tag::Md => pos == "MD",
        Tag::Prp => pos == "PRP",
        Tag::PrpPoss => pos == "PRP$",
        Tag::Aux => pos == "MD" || matches!(&*token.lemma, "be" | "have" | "do"),
        Tag::Be => &*token.lemma == "be",
        Tag::Uh => pos == "UH",
        Tag::Punct => !surface.chars().any(char::is_alphanumeric),
        // No compiling rule carries a NOM position (nominalization
        // detection lives in the register channel); staged rules that
        // do will need a detector wired here when promoted.
        Tag::Nom => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use friction_nlp::{PerceptronTagger, Tagger};
    use friction_packs::FRAME;
    use friction_packs::frame_compile::CompiledKind;

    fn tagged(text: &str) -> Vec<TaggedToken> {
        let tagger = PerceptronTagger::new().expect("embedded tagger loads");
        tagger.tag(text, 0)
    }

    fn scan(text: &str) -> (Vec<FrameMatch>, Vec<TaggedToken>) {
        let view = &FRAME.pack;
        let index = FrameIndex::build(view);
        let tokens = tagged(text);
        let matches = scan_sentence(view, &index, &tokens, text);
        (matches, tokens)
    }

    fn rule_ids(matches: &[FrameMatch]) -> Vec<&'static str> {
        matches
            .iter()
            .map(|m| FRAME.pack.rule(m.rule_index).expect("rule").id)
            .collect()
    }

    /// A lemma-matched literal rule fires on an inflected surface: the
    /// pattern literal is "utilize", the text says "utilized".
    #[test]
    fn lemma_matched_literal_fires_on_inflected_surface() {
        let (matches, _) = scan("We utilized the cache for speed.");
        assert!(
            rule_ids(&matches).contains(&"vsub.utilize"),
            "found: {:?}",
            rule_ids(&matches)
        );
    }

    /// A guard rule matches its protected verb.
    #[test]
    fn guard_rule_matches_its_verb() {
        let (matches, _) = scan("The compiler will generate the code.");
        assert!(
            rule_ids(&matches).contains(&"vguard.generate"),
            "found: {:?}",
            rule_ids(&matches)
        );
    }

    /// The multi-word collocation report rule matches as a phrase.
    #[test]
    fn multi_word_collocation_matches_as_phrase() {
        let (matches, _) = scan("This provides seamless integration with the API.");
        assert!(
            rule_ids(&matches).contains(&"col.seamless-integration"),
            "found: {:?}",
            rule_ids(&matches)
        );
    }

    /// A slot rule captures the trailing content: "utilize X".
    #[test]
    fn slot_captures_trailing_content() {
        let (matches, tokens) = scan("Teams utilize caching everywhere.");
        let m = matches
            .iter()
            .find(|m| FRAME.pack.rule(m.rule_index).expect("rule").id == "vsub.utilize")
            .expect("vsub.utilize matches");
        let x = m.slots[0].clone().expect("X captured");
        assert!(!x.is_empty());
        assert!(x.start > 0, "slot starts after the verb");
        let _ = tokens;
    }

    /// Sentence sentinels hold: the sentinelled `::lit` refinement
    /// fires only in its sentence-initial shape, while its ship parent
    /// (the bare phrase, authored without sentinels) may match
    /// anywhere — the edit-time gates arbitrate the parent's matches.
    #[test]
    fn sentence_initial_rule_respects_the_sentinel() {
        // rit.look-no-further::lit: <S> "look" "no" "further" ("than" X)? PUNCT </S>
        let (matches, _) = scan("Look no further!");
        assert!(
            rule_ids(&matches).contains(&"rit.look-no-further::lit"),
            "found: {:?}",
            rule_ids(&matches)
        );
        let (mid, _) = scan("You should look no further today maybe.");
        assert!(
            !rule_ids(&mid).contains(&"rit.look-no-further::lit"),
            "the sentinelled refinement must not fire mid-sentence, found: {:?}",
            rule_ids(&mid)
        );
    }

    /// The conflict policy keeps the leftmost-longest non-overlapping
    /// set and orders deterministically. Only edit-capable rules
    /// compete — the engine hands guards and report-only rules to
    /// their own channels before resolving, since a report rule must
    /// never shadow a rewrite out of its edit.
    #[test]
    fn resolve_keeps_leftmost_longest_non_overlapping() {
        let (matches, _) = scan("We utilize and leverage the cache.");
        let matches: Vec<FrameMatch> = matches
            .into_iter()
            .filter(|m| {
                matches!(
                    FRAME.pack.rule(m.rule_index).expect("rule").kind,
                    CompiledKind::Rewrite | CompiledKind::Delete
                )
            })
            .collect();
        let resolved = resolve(&FRAME.pack, matches);
        let ids = resolved
            .iter()
            .map(|m| FRAME.pack.rule(m.rule_index).expect("rule").id)
            .collect::<Vec<_>>();
        assert!(ids.contains(&"vsub.utilize"), "found: {ids:?}");
        assert!(ids.contains(&"vsub.leverage"), "found: {ids:?}");
        for pair in resolved.windows(2) {
            assert!(
                pair[0].bytes.end <= pair[1].bytes.start,
                "non-overlapping, ordered"
            );
        }
    }

    /// Scanning the same sentence twice is identical (determinism).
    #[test]
    fn scanning_twice_is_identical() {
        let text = "We utilized the cache to streamline the workflow.";
        let (a, _) = scan(text);
        let (b, _) = scan(text);
        let key = |ms: &[FrameMatch]| {
            ms.iter()
                .map(|m| (m.rule_index, m.bytes.clone()))
                .collect::<Vec<_>>()
        };
        assert_eq!(key(&a), key(&b));
    }
}
