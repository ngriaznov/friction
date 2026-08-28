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
//! their anchor's first word id in a dense array indexed by that id
//! (pack word ids are dense ranks, so this is a direct slice index,
//! not a tree lookup); two-or-more-word anchors additionally carry
//! their second word id, so the scan can reject a candidate by
//! peeking one token ahead before paying for full verification. The
//! per-sentence scan resolves each token to word ids once (lemma and
//! lowercased surface — a literal matches either, the same
//! equivalence the adjudication referee counts by), walks the
//! sentence left to right, and only attempts full verification for
//! rules whose anchor could plausibly start at the current token.
//! Anchor candidacy is an O(1) bucket lookup plus, for multi-word
//! anchors, one cheap next-token id check; full verification only
//! runs on sites that survive both.
//!
//! # Verification semantics
//!
//! The verifier walks the rule's pattern ops with a cursor, anchored so
//! the anchor run lands on the candidate tokens, with a fixed,
//! deterministic policy everywhere the grammar allows choice:
//!
//! - **Optionals are greedy**: an optional element consumes its token
//!   when it matches there, and is skipped otherwise (never both, no
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

/// One anchor-bucket entry: a candidate rule, plus — when its anchor
/// runs two or more words — the second anchor word's id and whether
/// that word must match by surface only (mirrors the rule's own
/// `surface_match`, so the discriminator can never be looser than the
/// verifier it's screening for).
#[derive(Debug, Clone, Copy)]
struct AnchorCandidate {
    rule_index: u32,
    second: Option<u32>,
    surface_only: bool,
}

/// Anchor-first lookup over a frame pack: rule indexes bucketed by
/// their anchor's first word id.
///
/// The bucket array is indexed directly by that id (pack word ids are
/// dense ranks assigned by the compiler, and the vocabulary is small,
/// so this trades a little unused capacity for O(1) lookup instead of
/// a tree). Built once per process.
#[derive(Debug)]
pub struct FrameIndex {
    by_first_anchor_id: Vec<Vec<AnchorCandidate>>,
    /// Every interned word, keyed by text, built once so
    /// [`scan_sentence`]'s per-token resolution — several lookups per
    /// token, every token of every sentence — hits a hash lookup
    /// instead of [`FramePackView::word_id`]'s binary search over the
    /// packed string table each time. `FxHashMap` (rustc's own
    /// seedless, deterministic hasher): this workspace's
    /// reproducibility doctrine rules out a randomly-seeded hasher
    /// (`ahash`/`foldhash`) on a hot path, and std's default `SipHash`'s
    /// `DoS` resistance buys nothing for compiled-in pack vocabulary.
    word_ids: rustc_hash::FxHashMap<Box<str>, u32>,
}

impl FrameIndex {
    /// Builds the index over every rule in `view`.
    #[must_use]
    pub fn build(view: &FramePackView<'_>) -> Self {
        let mut by_first_anchor_id: Vec<Vec<AnchorCandidate>> =
            vec![Vec::new(); view.word_count() as usize];
        for index in 0..view.rule_count() {
            let rule = view.rule(index).expect("index in range");
            let mut anchor = rule.anchor.clone();
            let Some(first) = anchor.next() else {
                continue;
            };
            let second = anchor.next();
            if let Some(bucket) = by_first_anchor_id.get_mut(first as usize) {
                bucket.push(AnchorCandidate {
                    rule_index: index,
                    second,
                    surface_only: rule.surface_match,
                });
            }
        }
        let mut word_ids = rustc_hash::FxHashMap::with_capacity_and_hasher(
            view.word_count() as usize,
            rustc_hash::FxBuildHasher,
        );
        for id in 0..view.word_count() {
            let word = view.word(id).expect("id in range");
            word_ids.insert(Box::from(word), id);
        }
        Self {
            by_first_anchor_id,
            word_ids,
        }
    }

    /// The rules whose anchor starts with `word_id`.
    fn candidates(&self, word_id: u32) -> &[AnchorCandidate] {
        self.by_first_anchor_id
            .get(word_id as usize)
            .map_or(&[], Vec::as_slice)
    }

    /// The interned id of `w`, from the index's own hash map — see the
    /// `word_ids` field docs for why this exists alongside
    /// [`FramePackView::word_id`] rather than calling through to it.
    fn word_id(&self, w: &str) -> Option<u32> {
        self.word_ids.get(w).copied()
    }
}

/// One sentence token resolved against the pack's interner.
struct ResolvedToken {
    /// Interner id of the lowercased surface, if interned: the only
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
    // Scratch buffers reused across every token in the sentence instead
    // of each token allocating its own lowercased `String` (surface,
    // lemma) and its own `Vec<String>` of suffix-reduction candidates —
    // this loop is sequential (a single `.map().collect()`, not
    // `par_iter`), so one buffer per role is safe to reuse and clear on
    // each token.
    let mut surface_buf = String::new();
    let mut lemma_buf = String::new();
    let mut reduction_buf = String::new();
    let resolved: Vec<ResolvedToken> = tokens
        .iter()
        .map(|token| {
            lower_into(&mut surface_buf, surface_of(token, text));
            let surface_id = index.word_id(&surface_buf);
            let mut ids: Vec<u32> = surface_id.into_iter().collect();
            let mut push = |word: &str| {
                if let Some(id) = index.word_id(word)
                    && !ids.contains(&id)
                {
                    ids.push(id);
                }
            };
            lower_into(&mut lemma_buf, &token.lemma);
            push(&lemma_buf);
            // Irregular forms reduce through the shared table directly —
            // one hash lookup instead of `lemmatize`'s lowercase-then-
            // linear-scan repeated once per VBD/VBG/VBZ (the irregular
            // check inside `lemmatize` is pos-independent anyway, so
            // three calls only ever found the same base three times).
            if let Some(base) = friction_nlp::irregular_verb_base(&surface_buf) {
                push(base);
            }
            // `lemmatize`'s regular reverse-derivation (VBD/VBZ, and
            // VBG's plain "-ing" strip) picks ONE stem per form via a
            // round-trip check, and when two stems both round-trip
            // ("commenc"/"commence" for "commenced") it can keep the
            // wrong one. Here the pack's own interner is the
            // dictionary instead: `for_each_suffix_reduction` below
            // generates every plausible suffix reduction and lets
            // interner membership arbitrate — a stem that is no rule's
            // word can never matter to a scan, and a stem that is one
            // is exactly the reading the rule wants. Shares
            // `suffix_reductions`'s exact candidate logic (see that
            // function's own docs on why runtime and the adjudication
            // referee can never disagree) via `for_each_suffix_reduction`,
            // without collecting a `Vec<String>` this hot path only
            // ever throws away after one `word_id` lookup each.
            //
            // The one candidate `for_each_suffix_reduction` does NOT
            // generate that `lemmatize`'s VBG path does: the
            // "-ying" -> "-ie" restoration for the tie/die/lie/vie
            // family (a short y-stem after stripping "-ing" — the
            // suffix-reduction loop only ever strips "-ing" down to
            // the bare y-stem, never rebuilds the "-ie" ending).
            // Fall back to it directly so it isn't lost.
            if let Some(stem) = surface_buf.strip_suffix("ing")
                && stem.chars().count() <= 2
                && let Some(base) = surface_buf.strip_suffix("ying")
            {
                reduction_buf.clear();
                reduction_buf.push_str(base);
                reduction_buf.push_str("ie");
                push(&reduction_buf);
            }
            for_each_suffix_reduction(&surface_buf, &mut reduction_buf, push);
            ResolvedToken { surface_id, ids }
        })
        .collect();

    let mut matches = Vec::new();
    // Hoisted out of the token loop and cleared per token instead of
    // reallocated: this is `scan_sentence`'s innermost bookkeeping,
    // paid once per token either way, so reusing the backing buffer
    // saves one allocation per token instead of per rule-dedup reset.
    let mut tried: Vec<u32> = Vec::new();
    for at in 0..tokens.len() {
        tried.clear();
        for idx in 0..resolved[at].ids.len() {
            let id = resolved[at].ids[idx];
            for candidate in index.candidates(id) {
                if tried.contains(&candidate.rule_index) {
                    continue;
                }
                // Second-anchor-word discriminator: for multi-word
                // anchors, reject before paying for a rule fetch and
                // full verification unless the next token can stand
                // for the anchor's second word. Single-word anchors
                // (`second == None`) and pilot rules (surface-only)
                // fall through to their pre-existing behavior.
                if let Some(second) = candidate.second
                    && !resolved
                        .get(at + 1)
                        .is_some_and(|next| next.matches(second, candidate.surface_only))
                {
                    continue;
                }
                let rule_index = candidate.rule_index;
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
/// Only edit-capable kinds should compete. Callers filter guards and
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

/// Lowercases `raw` into `buf` (cleared first) instead of allocating a new
/// `String` — exactly [`str::to_lowercase`]'s own per-`char`
/// [`char::to_lowercase`] mapping, so this produces byte-identical output
/// to `raw.to_lowercase()`, just written into scratch space [`scan_sentence`]
/// reuses across every token in a sentence rather than allocating fresh
/// per token.
fn lower_into(buf: &mut String, raw: &str) {
    buf.clear();
    for c in raw.chars() {
        buf.extend(c.to_lowercase());
    }
}

/// Undoes a doubled final consonant in `stem`, writing the shortened form
/// into `buf` (cleared first) and returning `true` — `false`, `buf`
/// untouched, when `stem` has no doubled consonant to undo. Shared by
/// [`for_each_suffix_reduction`]'s `-ed`/`-ing` branch and, through it,
/// [`suffix_reductions`].
fn undoubled_into(stem: &str, buf: &mut String) -> bool {
    let chars: Vec<char> = stem.chars().collect();
    let n = chars.len();
    if n >= 2 && chars[n - 1] == chars[n - 2] && !"aeiou".contains(chars[n - 1]) {
        buf.clear();
        buf.extend(chars[..n - 1].iter());
        true
    } else {
        false
    }
}

/// Calls `f` once per plausible base-form reduction of a lowercased
/// `surface`, in the same order [`suffix_reductions`] collects them —
/// covers `-ied`/`-ed`/`-ing` (with doubled-consonant undoubling and
/// `-e` restoration) and `-ies`/`-es`/`-s`, deliberately over-generating
/// (callers keep only candidates a dictionary knows: the pack's interner
/// at match time, the probe vocabulary in the adjudication referee).
///
/// [`suffix_reductions`] is defined in terms of this function (collecting
/// each `f` call into an owned `Vec<String>`), so referee and runtime
/// share the exact same reduction logic either way — this generator form
/// exists only so [`scan_sentence`]'s hot path, which needs nothing but
/// an interner `word_id` lookup per candidate, never keeps a candidate
/// past that lookup. `buf` is scratch space the caller owns and reuses
/// across tokens; each candidate is either a subslice of `surface`
/// (borrowed directly, no allocation) or built into `buf` and yielded as
/// `&*buf`.
fn for_each_suffix_reduction(surface: &str, buf: &mut String, mut f: impl FnMut(&str)) {
    if let Some(stem) = surface.strip_suffix("ied") {
        buf.clear();
        buf.push_str(stem);
        buf.push('y');
        f(buf);
    }
    for suffix in ["ed", "ing"] {
        if let Some(stem) = surface.strip_suffix(suffix) {
            if undoubled_into(stem, buf) {
                f(buf);
            }
            f(stem);
            buf.clear();
            buf.push_str(stem);
            buf.push('e');
            f(buf);
        }
    }
    if let Some(stem) = surface.strip_suffix("ies") {
        buf.clear();
        buf.push_str(stem);
        buf.push('y');
        f(buf);
    }
    if let Some(stem) = surface.strip_suffix("es") {
        f(stem);
    }
    if let Some(stem) = surface.strip_suffix('s') {
        f(stem);
    }
}

/// Every plausible base-form reduction of a lowercased surface.
///
/// Covers `-ied`/`-ed`/`-ing` (with doubled-consonant undoubling and
/// `-e` restoration) and `-ies`/`-es`/`-s`. Deliberately
/// over-generates: callers keep only candidates a dictionary knows
/// (the pack's interner at match time; the probe vocabulary in the
/// adjudication referee, which shares this function so referee and
/// runtime can never disagree about what counts as an occurrence).
#[must_use]
pub fn suffix_reductions(surface: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut buf = String::new();
    for_each_suffix_reduction(surface, &mut buf, |candidate| {
        out.push(candidate.to_string());
    });
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
const fn element_to_op_index(ops: &[DecodedOp], element: u16) -> Option<usize> {
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
    /// slots or nested groups inside alternatives. The grammar
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
        let (matches, _) = scan("This gives customers a strong value proposition.");
        assert!(
            rule_ids(&matches).contains(&"col.value-proposition"),
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
    /// anywhere. The edit-time gates arbitrate the parent's matches.
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
    /// compete. The engine hands guards and report-only rules to
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

    /// The rule index of the pack rule with id `id`.
    fn find_rule_index(view: &FramePackView<'_>, id: &str) -> u32 {
        (0..view.rule_count())
            .find(|&i| view.rule(i).expect("rule").id == id)
            .unwrap_or_else(|| panic!("rule {id} not found in pack"))
    }

    /// [`FrameIndex`] stores each multi-word anchor's second word id
    /// alongside the rule, and the scan uses it to reject a candidate
    /// before ever invoking the verifier: a sentence where the
    /// anchor's first word appears but the second does not must not
    /// match, even though the first word alone would have put the
    /// rule in play.
    #[test]
    fn two_word_anchor_discriminator_rejects_before_verify() {
        let view = &FRAME.pack;
        let index = FrameIndex::build(view);
        let rule_index = find_rule_index(view, "col.value-proposition");
        let rule = view.rule(rule_index).expect("rule");
        let mut anchor = rule.anchor.clone();
        let first = anchor.next().expect("anchor has a first word");
        let second = anchor.next().expect("anchor has a second word");
        let candidate = index
            .candidates(first)
            .iter()
            .find(|c| c.rule_index == rule_index)
            .expect("rule indexed under its anchor's first word");
        assert_eq!(
            candidate.second,
            Some(second),
            "discriminator carries the anchor's second word"
        );

        // "value" appears, but not followed by "proposition": the
        // discriminator rejects before verification runs.
        let (rejected, _) = scan("This gives customers a strong value assessment.");
        assert!(
            !rule_ids(&rejected).contains(&"col.value-proposition"),
            "found: {:?}",
            rule_ids(&rejected)
        );

        // Both anchor words present: the rule still fires.
        let (accepted, _) = scan("This gives customers a strong value proposition.");
        assert!(
            rule_ids(&accepted).contains(&"col.value-proposition"),
            "found: {:?}",
            rule_ids(&accepted)
        );
    }

    /// A single-word anchor carries no discriminator, so the
    /// next-token check never applies to it — candidacy behaves
    /// exactly as before the discriminator was added.
    #[test]
    fn single_word_anchor_has_no_discriminator() {
        let view = &FRAME.pack;
        let index = FrameIndex::build(view);
        let rule_index = find_rule_index(view, "vsub.utilize");
        let first = view
            .rule(rule_index)
            .expect("rule")
            .anchor
            .clone()
            .next()
            .expect("anchor has a first word");
        let candidate = index
            .candidates(first)
            .iter()
            .find(|c| c.rule_index == rule_index)
            .expect("rule indexed under its anchor's first word");
        assert_eq!(candidate.second, None);

        let (matches, _) = scan("We utilize the cache.");
        assert!(
            rule_ids(&matches).contains(&"vsub.utilize"),
            "found: {:?}",
            rule_ids(&matches)
        );
    }

    /// A pilot (surface-matched) two-word anchor discriminates by
    /// surface id, mirroring the verifier's own pilot semantics:
    /// never a lemma-level identity, since pilot targets are exact
    /// inflected forms.
    #[test]
    fn pilot_rule_discriminator_uses_surface_id_only() {
        let view = &FRAME.pack;
        let index = FrameIndex::build(view);
        let rule_index = find_rule_index(view, "pilot.such-as");
        let rule = view.rule(rule_index).expect("rule");
        assert!(rule.surface_match, "pilot rules are surface-matched");
        let mut anchor = rule.anchor.clone();
        let first = anchor.next().expect("anchor has a first word");
        let second = anchor.next().expect("anchor has a second word");
        let candidate = index
            .candidates(first)
            .iter()
            .find(|c| c.rule_index == rule_index)
            .expect("rule indexed under its anchor's first word");
        assert_eq!(candidate.second, Some(second));
        assert!(
            candidate.surface_only,
            "pilot candidate discriminates by surface id"
        );

        let (rejected, _) = scan("We processed data such that errors dropped.");
        assert!(
            !rule_ids(&rejected).contains(&"pilot.such-as"),
            "found: {:?}",
            rule_ids(&rejected)
        );

        let (accepted, _) = scan("We used tools such as calculators.");
        assert!(
            rule_ids(&accepted).contains(&"pilot.such-as"),
            "found: {:?}",
            rule_ids(&accepted)
        );
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
