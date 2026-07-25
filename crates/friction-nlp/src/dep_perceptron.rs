//! [`PerceptronParser`]: an averaged-structured-perceptron [`DepParser`]
//! that drives [`crate::dep_arceager::Configuration`] at inference time —
//! the classifier the module docs on `dep_arceager` describe as "later, an
//! actual `DepParser` implementation".
//!
//! # Action space
//!
//! Every decision is one of 42 actions: [`Transition::Shift`],
//! [`Transition::Reduce`], and [`Transition::LeftArc`]/[`Transition::RightArc`]
//! over each of the 20 non-[`DepRelation::Root`] relations (`Root` is never a
//! legal arc label — see that variant's own docs). [`action_index`]/
//! [`action_at`] fix a single, deterministic enumeration of that space
//! ([`RELATIONS`], in [`DepRelation`]'s own declaration order) so "the
//! lowest action index" is a stable, load-bearing tie-break rather than an
//! accident of iteration order.
//!
//! # Feature templates
//!
//! Fixed, and load-bearing: the vendored artifact
//! (`weights/parser_en.json.gz`) was trained against the exact template list
//! below, keyed by a template tag so two templates whose *values* happen to
//! coincide never collide. Changing this list invalidates that artifact.
//! `s0`/`s1` are the top two stack items, `b0`/`b1`/`b2` the first three
//! buffer items (all by token index into the sentence); `w` is a token's
//! lowercased surface text, `t` its Penn tag. A position that does not exist
//! (an empty stack slot, a buffer past the sentence end, a token with no
//! assigned dependent yet) contributes the sentinel [`NONE`] rather than
//! being omitted, so every token always emits the same fixed-length feature
//! list regardless of configuration shape.
//!
//! - Unigrams: `s0.w`, `s0.t`, `s0.wt` (`w+t`), `s1.w`, `s1.t`, `s1.wt`,
//!   `b0.w`, `b0.t`, `b0.wt`, `b1.w`, `b1.t`, `b1.wt`, `b2.t`
//! - Pairs: `s0b0.wtwt` (`s0.w+s0.t+b0.w+b0.t`), `s0b0.wtw`, `s0b0.wwt`,
//!   `s0b0.wtt`, `s0b0.twt`, `s0b0.ww`, `s0b0.tt`, `b0b1.tt`
//! - Triples: `s0b0b1.ttt`, `s1s0b0.ttt`
//! - Children: `s0.lc.t`/`s0.lc.rel` (tag/relation of `s0`'s leftmost
//!   assigned dependent), `s0.rc.t`/`s0.rc.rel` (rightmost), `b0.lc.t`/
//!   `b0.lc.rel` (`b0`'s leftmost)
//! - Distance: `dist.s0t.b0t` — `b0`'s token index minus `s0`'s, bucketed
//!   into `1`, `2`, `3`, `4`, `5`, `6-10`, `>10` (or [`NONE`] if either side
//!   is absent), conjoined with `s0.t+b0.t`
//!
//! Thirty feature strings per decision, always, in that fixed order —
//! [`extract_features`] is the single source of truth for the exact format
//! of each one.
//!
//! # Training: structured perceptron with early update
//!
//! [`train_support::train_sentence`] decodes greedily from a fresh
//! [`Configuration`], scoring every step with the model's own live (not yet
//! averaged) weights, masked to [`Configuration::is_allowed`]. The instant
//! the model's masked argmax disagrees with
//! [`crate::dep_arceager::oracle`]'s transition, the oracle's action is
//! promoted and the model's wrong pick demoted for every feature of that
//! step, and the sentence is abandoned right there — the rest of it, which
//! the model never got to see because it already went wrong, contributes no
//! further signal. A sentence the model gets entirely right contributes one
//! full oracle-length walk of updates-that-never-fire (since agreement never
//! triggers a bump). Final weights are the standard lazy-averaged perceptron
//! (see [`train_support::AveragedPerceptron`]), the same accumulate-by-
//! timestamp trick [`crate::tag_perceptron`]'s tagger trainer uses, adapted
//! from string class names to a fixed `u8` action index instead of
//! duplicating a second copy of that same lazy-averaging arithmetic
//! unchanged.
//!
//! Only sentences whose gold tree [`crate::dep_arceager::derive`] can
//! actually reproduce ever reach training — see that function's own docs
//! for why a non-derivable gold tree (in practice, non-projective) has to be
//! dropped rather than trained on: walking it with the oracle would silently
//! teach the model a transition sequence that does not reconstruct the gold
//! parse it was supposedly learning.
//!
//! # Determinism
//!
//! No randomness anywhere in this module, at training or at inference:
//! [`train_support`]'s epoch loop walks `weights/gold_dep_en.conllu`'s
//! sentences in the file's own on-disk order (no shuffling), matching
//! `examples/train_perceptron.rs`'s own choice for the same reason —
//! reproducibility that does not depend on a seed at all beats
//! reproducibility that depends on getting a seed right forever. At
//! inference, [`best_allowed_action`] breaks every tie toward the lowest
//! action index, and feature-list order is the fixed order documented
//! above, so summation order (and therefore the resulting scores) is
//! identical on every run.

use std::collections::HashMap;
use std::io::Read as _;

use serde::{Deserialize, Serialize};

use crate::dep::{DepParseError, DepParser, DepRelation, SentenceParse};
use crate::dep_arceager::{Configuration, Transition};
use crate::tag::TaggedToken;

/// The vendored, gzip-compressed dependency-parser weight artifact. See
/// `weights/NOTICE.md` for provenance and the reproduction command;
/// `examples/train_parser.rs` (behind the `train-tooling` feature)
/// regenerates it.
static WEIGHTS_BYTES: &[u8] = include_bytes!("../weights/parser_en.json.gz");

/// Sentinel feature value for an absent stack/buffer position, an absent
/// assigned dependent, or an incomputable distance bucket. Distinct from any
/// real lowercased word or Penn tag, so a feature built from it can never be
/// confused with a genuine one.
const NONE: &str = "<none>";

/// The 20 relations [`Transition::LeftArc`]/[`Transition::RightArc`] may
/// carry, in [`DepRelation`]'s own declaration order — the fixed
/// enumeration [`action_index`]/[`action_at`] build the 42-way action space
/// from. [`DepRelation::Root`] is excluded: it is never a legal arc label
/// (see that variant's own docs).
const RELATIONS: [DepRelation; 20] = [
    DepRelation::Acl,
    DepRelation::Advcl,
    DepRelation::Agent,
    DepRelation::Amod,
    DepRelation::Aux,
    DepRelation::AuxPass,
    DepRelation::Cc,
    DepRelation::Ccomp,
    DepRelation::Conj,
    DepRelation::Csubj,
    DepRelation::Det,
    DepRelation::Dobj,
    DepRelation::Mark,
    DepRelation::Nsubj,
    DepRelation::NsubjPass,
    DepRelation::Pobj,
    DepRelation::Prep,
    DepRelation::Xcomp,
    DepRelation::Punct,
    DepRelation::Other,
];

/// Total action count: [`Transition::Shift`], [`Transition::Reduce`], plus
/// one [`Transition::LeftArc`] and one [`Transition::RightArc`] per
/// [`RELATIONS`] entry.
const NUM_ACTIONS: usize = 2 + 2 * RELATIONS.len();

/// `rel`'s position in [`RELATIONS`].
///
/// Only [`action_index`] calls this (in turn only needed by training and by
/// this module's own round-trip tests — inference only ever goes
/// [`action_at`]'s direction, index to transition, never the reverse), so
/// it is gated the same way `action_index` is.
///
/// # Panics
/// Panics if `rel` is [`DepRelation::Root`] — never a legal arc label, so
/// never legally reachable here; every caller in this module only ever
/// passes a relation carried by [`Transition::LeftArc`]/
/// [`Transition::RightArc`], which [`crate::dep_arceager::oracle`] never
/// labels `Root` (see that function's own docs).
#[cfg(any(test, feature = "train-tooling"))]
fn relation_index(rel: DepRelation) -> usize {
    RELATIONS
        .iter()
        .position(|&candidate| candidate == rel)
        .expect("DepRelation::Root is never a legal LeftArc/RightArc label")
}

/// `transition`'s position in the fixed 42-way action enumeration —
/// [`action_at`]'s exact inverse. Only training
/// ([`train_support::train_sentence`], scoring the oracle's own transition)
/// and this module's own round-trip tests ever need to go this direction;
/// inference only ever goes index -> transition via [`action_at`].
#[cfg(any(test, feature = "train-tooling"))]
fn action_index(transition: Transition) -> usize {
    match transition {
        Transition::Shift => 0,
        Transition::Reduce => 1,
        Transition::LeftArc(rel) => 2 + 2 * relation_index(rel),
        Transition::RightArc(rel) => 3 + 2 * relation_index(rel),
    }
}

/// The [`Transition`] at position `index` in the fixed 42-way action
/// enumeration — [`action_index`]'s exact inverse.
///
/// # Panics
/// Panics if `index` is out of range for the fixed [`NUM_ACTIONS`]-entry
/// action space — every caller in this module only ever passes an index
/// produced by [`action_index`] or scanned from a fixed `0..NUM_ACTIONS`
/// range.
fn action_at(index: usize) -> Transition {
    match index {
        0 => Transition::Shift,
        1 => Transition::Reduce,
        i if i < NUM_ACTIONS => {
            let rel = RELATIONS[(i - 2) / 2];
            if i % 2 == 0 {
                Transition::LeftArc(rel)
            } else {
                Transition::RightArc(rel)
            }
        }
        i => panic!("action index {i} out of range for a {NUM_ACTIONS}-action space"),
    }
}

/// One stack/buffer position's feature-relevant view: lowercased surface
/// text and Penn tag, or [`NONE`] for both if the position does not exist.
struct View<'a> {
    w: &'a str,
    t: &'a str,
}

fn at<'a>(words: &'a [Box<str>], tags: &'a [Box<str>], idx: Option<usize>) -> View<'a> {
    idx.map_or(View { w: NONE, t: NONE }, |i| View {
        w: &words[i],
        t: &tags[i],
    })
}

/// `head`'s leftmost and rightmost assigned dependents in `config`, by
/// token index (not by attachment order) — a linear scan over every token,
/// since a dependent may be anywhere in the sentence and
/// [`Configuration`] exposes no reverse (head -> children) index of its
/// own, only the forward `token -> (head, relation)` one
/// ([`Configuration::arc`]).
fn children_of(config: &Configuration, head: usize) -> (Option<usize>, Option<usize>) {
    let mut left = None;
    let mut right = None;
    for token in 0..config.token_count() {
        if let Some((token_head, _)) = config.arc(token)
            && token_head == head
        {
            if left.is_none() {
                left = Some(token);
            }
            right = Some(token);
        }
    }
    (left, right)
}

/// Appends `{prefix}.t=` and `{prefix}.rel=` features describing `child`'s
/// tag and its own arc relation (the relation attaching it to its head,
/// i.e. exactly the label a caller wants when `child` is being reported as
/// someone else's dependent) — [`NONE`] for both if `child` is absent.
fn push_child_features(
    feats: &mut Vec<Box<str>>,
    prefix: &str,
    config: &Configuration,
    tags: &[Box<str>],
    child: Option<usize>,
) {
    let (tag, relation) = child.map_or((NONE, NONE), |index| {
        let relation = config.arc(index).map_or(NONE, |(_, rel)| rel.as_str());
        (tags[index].as_ref(), relation)
    });
    feats.push(Box::from(format!("{prefix}.t={tag}")));
    feats.push(Box::from(format!("{prefix}.rel={relation}")));
}

/// `diff` (a buffer-minus-stack token-index distance, always positive since
/// the buffer only ever holds indices past every stacked one) bucketed into
/// `1`, `2`, `3`, `4`, `5`, `6-10`, `>10`.
const fn bucket_distance(diff: usize) -> &'static str {
    match diff {
        1 => "1",
        2 => "2",
        3 => "3",
        4 => "4",
        5 => "5",
        6..=10 => "6-10",
        _ => ">10",
    }
}

/// Builds one configuration's fixed, 30-entry feature list. See the module
/// docs for the exact template catalogue this implements; this function is
/// that catalogue's single source of truth.
///
/// `words`/`tags` are the whole sentence's lowercased surface text and Penn
/// tags, indexed by token — the same slices for every call across one
/// sentence's decode, since only `config` changes step to step.
fn extract_features(
    words: &[Box<str>],
    tags: &[Box<str>],
    config: &Configuration,
) -> Vec<Box<str>> {
    let stack = config.stack();
    let s0 = stack.last().copied();
    let s1 = if stack.len() >= 2 {
        Some(stack[stack.len() - 2])
    } else {
        None
    };
    let b0 = config.buffer_front();
    let n = config.token_count();
    let b1 = b0.and_then(|i| (i + 1 < n).then_some(i + 1));
    let b2 = b0.and_then(|i| (i + 2 < n).then_some(i + 2));

    let s0v = at(words, tags, s0);
    let s1v = at(words, tags, s1);
    let b0v = at(words, tags, b0);
    let b1v = at(words, tags, b1);
    let b2v = at(words, tags, b2);

    let mut feats = Vec::with_capacity(30);

    // Unigrams.
    feats.push(Box::from(format!("s0.w={}", s0v.w)));
    feats.push(Box::from(format!("s0.t={}", s0v.t)));
    feats.push(Box::from(format!("s0.wt={}|{}", s0v.w, s0v.t)));
    feats.push(Box::from(format!("s1.w={}", s1v.w)));
    feats.push(Box::from(format!("s1.t={}", s1v.t)));
    feats.push(Box::from(format!("s1.wt={}|{}", s1v.w, s1v.t)));
    feats.push(Box::from(format!("b0.w={}", b0v.w)));
    feats.push(Box::from(format!("b0.t={}", b0v.t)));
    feats.push(Box::from(format!("b0.wt={}|{}", b0v.w, b0v.t)));
    feats.push(Box::from(format!("b1.w={}", b1v.w)));
    feats.push(Box::from(format!("b1.t={}", b1v.t)));
    feats.push(Box::from(format!("b1.wt={}|{}", b1v.w, b1v.t)));
    feats.push(Box::from(format!("b2.t={}", b2v.t)));

    // Pairs.
    feats.push(Box::from(format!(
        "s0b0.wtwt={}|{}|{}|{}",
        s0v.w, s0v.t, b0v.w, b0v.t
    )));
    feats.push(Box::from(format!("s0b0.wtw={}|{}|{}", s0v.w, s0v.t, b0v.w)));
    feats.push(Box::from(format!("s0b0.wwt={}|{}|{}", s0v.w, b0v.w, b0v.t)));
    feats.push(Box::from(format!("s0b0.wtt={}|{}|{}", s0v.w, s0v.t, b0v.t)));
    feats.push(Box::from(format!("s0b0.twt={}|{}|{}", s0v.t, b0v.w, b0v.t)));
    feats.push(Box::from(format!("s0b0.ww={}|{}", s0v.w, b0v.w)));
    feats.push(Box::from(format!("s0b0.tt={}|{}", s0v.t, b0v.t)));
    feats.push(Box::from(format!("b0b1.tt={}|{}", b0v.t, b1v.t)));

    // Triples.
    feats.push(Box::from(format!(
        "s0b0b1.ttt={}|{}|{}",
        s0v.t, b0v.t, b1v.t
    )));
    feats.push(Box::from(format!(
        "s1s0b0.ttt={}|{}|{}",
        s1v.t, s0v.t, b0v.t
    )));

    // Children.
    let (s0_left, s0_right) = s0.map_or((None, None), |head| children_of(config, head));
    push_child_features(&mut feats, "s0.lc", config, tags, s0_left);
    push_child_features(&mut feats, "s0.rc", config, tags, s0_right);
    let b0_left = b0.and_then(|head| children_of(config, head).0);
    push_child_features(&mut feats, "b0.lc", config, tags, b0_left);

    // Distance.
    let dist_bucket = match (s0, b0) {
        (Some(si), Some(bi)) => bucket_distance(bi - si),
        _ => NONE,
    };
    feats.push(Box::from(format!(
        "dist.s0t.b0t={}|{}|{}",
        dist_bucket, s0v.t, b0v.t
    )));

    feats
}

/// The highest-scoring action among those [`Configuration::is_allowed`]
/// permits, breaking ties toward the lowest action index — `None` only if
/// no action is legal at all, which never happens while `config` is not
/// [`Configuration::is_terminal`] (see that method's own docs: a non-empty
/// buffer always makes [`Transition::Shift`] legal, and an exhausted buffer
/// that is not terminal always makes [`Transition::Reduce`] legal).
fn best_allowed_action(scores: &[f32; NUM_ACTIONS], config: &Configuration) -> Option<usize> {
    let mut best: Option<(usize, f32)> = None;
    for (index, &score) in scores.iter().enumerate() {
        if !config.is_allowed(action_at(index)) {
            continue;
        }
        let better = match best {
            None => true,
            Some((_, best_score)) => score > best_score,
        };
        if better {
            best = Some((index, score));
        }
    }
    best.map(|(index, _)| index)
}

/// On-disk shape of the dependency-parser weight artifact: a `BTreeMap` so
/// the serialized JSON is stable and diffable across re-training runs, and a
/// sparse per-feature action list rather than a dense one, since most
/// features only ever fire for a handful of the 42 actions.
///
/// `pub` (not exported at the crate root) purely so
/// `train_support::AveragedPerceptron::finish` and
/// [`PerceptronParser::from_weight_file`] can hand one to each other and to
/// `examples/train_parser.rs` — mirrors [`crate::tag_perceptron`]'s own
/// `WeightFile`, including that same crate-internal-only reachability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightFile {
    /// feature string -> sparse `[(action index into the fixed 42-way
    /// space, weight)]`.
    pub features: std::collections::BTreeMap<Box<str>, Vec<(u8, f32)>>,
}

/// Runtime weight table: one dense 42-entry weight array per feature
/// string, built once at load from a [`WeightFile`]'s sparse rows.
struct WeightTable {
    by_feature: HashMap<Box<str>, [f32; NUM_ACTIONS]>,
}

impl WeightTable {
    fn from_file(file: WeightFile) -> Self {
        let mut by_feature = HashMap::with_capacity(file.features.len());
        for (feature, sparse) in file.features {
            let mut dense = [0f32; NUM_ACTIONS];
            for (index, weight) in sparse {
                if let Some(slot) = dense.get_mut(usize::from(index)) {
                    *slot = weight;
                }
            }
            by_feature.insert(feature, dense);
        }
        Self { by_feature }
    }

    /// Sums each feature's dense per-action weight array, in `features`'
    /// own fixed order — float summation order is identical across runs
    /// because iteration order here is fixed by the caller-supplied slice,
    /// never by hash-map iteration (mirrors
    /// [`crate::tag_perceptron`]'s `WeightTable::score`).
    fn score(&self, features: &[Box<str>]) -> [f32; NUM_ACTIONS] {
        let mut scores = [0f32; NUM_ACTIONS];
        for feature in features {
            if let Some(dense) = self.by_feature.get(feature) {
                for (slot, weight) in scores.iter_mut().zip(dense.iter()) {
                    *slot += weight;
                }
            }
        }
        scores
    }
}

/// Errors constructing a [`PerceptronParser`].
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum PerceptronParseError {
    /// The embedded weight artifact failed to gunzip.
    #[error("failed to decompress the embedded dependency-parser weight artifact: {0}")]
    Decompress(#[source] std::io::Error),
    /// The decompressed artifact was not valid JSON in the expected shape.
    #[error("failed to parse the embedded dependency-parser weight artifact: {0}")]
    Parse(#[source] serde_json::Error),
}

/// A [`DepParser`] backed by a from-scratch-trained averaged structured
/// perceptron. See the module docs for the action space, feature templates,
/// training procedure, and determinism guarantees.
pub struct PerceptronParser {
    weights: WeightTable,
}

impl PerceptronParser {
    /// Loads the parser from its embedded, gzip-compressed weight artifact.
    ///
    /// # Errors
    /// [`PerceptronParseError::Decompress`] if the embedded bytes are not
    /// valid gzip; [`PerceptronParseError::Parse`] if the decompressed JSON
    /// does not match the expected shape. Neither should happen for the
    /// vendored artifact this crate ships.
    pub fn new() -> Result<Self, PerceptronParseError> {
        let mut decoder = flate2::read::GzDecoder::new(WEIGHTS_BYTES);
        let mut json = String::new();
        decoder
            .read_to_string(&mut json)
            .map_err(PerceptronParseError::Decompress)?;
        let file: WeightFile = serde_json::from_str(&json).map_err(PerceptronParseError::Parse)?;
        Ok(Self {
            weights: WeightTable::from_file(file),
        })
    }

    /// Builds a parser directly from an in-memory [`WeightFile`], bypassing
    /// the embedded artifact entirely.
    ///
    /// Exists so `examples/train_parser.rs` can evaluate a freshly trained
    /// model — through this exact same inference path — before that model
    /// is written to disk and compiled back in as the vendored artifact;
    /// gated behind `train-tooling` so a production build never exposes a
    /// way to load an untrusted weight file.
    #[cfg(feature = "train-tooling")]
    #[must_use]
    pub fn from_weight_file(file: WeightFile) -> Self {
        Self {
            weights: WeightTable::from_file(file),
        }
    }
}

impl DepParser for PerceptronParser {
    fn parse(&self, source: &str, tokens: &[TaggedToken]) -> Result<SentenceParse, DepParseError> {
        let words: Vec<Box<str>> = tokens
            .iter()
            .map(|token| Box::from(source[token.token.range.clone()].to_lowercase()))
            .collect();
        let tags: Vec<Box<str>> = tokens
            .iter()
            .map(|token| Box::from(token.pos.as_str()))
            .collect();

        let mut config = Configuration::new(tokens.len());
        while !config.is_terminal() {
            let features = extract_features(&words, &tags, &config);
            let scores = self.weights.score(&features);
            match best_allowed_action(&scores, &config) {
                Some(index) => config.apply(action_at(index)),
                // Unreachable while `!config.is_terminal()` (see
                // `best_allowed_action`'s own docs) — defensive rather than
                // a panic, since a `DepParser` implementation must never
                // panic on any input (see `DepParser::parse`'s contract).
                None => break,
            }
        }

        SentenceParse::new(config.finish())
    }
}

/// Training-support helpers.
///
/// Gold-file parsing, the early-update structured-perceptron trainer, and
/// weight-artifact save/load. Exists only behind the `train-tooling`
/// feature, mirroring [`crate::tag_perceptron::train_support`].
#[cfg(feature = "train-tooling")]
pub mod train_support {
    use std::collections::BTreeMap;
    use std::io::Write as _;

    use super::{
        NONE, NUM_ACTIONS, PerceptronParseError, WeightFile, action_at, action_index,
        best_allowed_action, extract_features,
    };
    use crate::dep::{
        Confidence, DepEdge, DepParseError, DepRelation, ParseDepRelationError, SentenceParse,
    };
    use crate::dep_arceager::{Configuration, oracle};

    /// One gold-parsed sentence loaded from `weights/gold_dep_en.conllu`.
    pub struct GoldSentence {
        /// `"train"` or `"test"`, read verbatim from the sentence's `# split
        /// = ` comment line.
        pub split: Box<str>,
        /// Every token's lowercased surface text, in order.
        pub words: Vec<Box<str>>,
        /// Every token's Penn tag, in order.
        pub tags: Vec<Box<str>>,
        /// The gold dependency tree itself.
        pub parse: SentenceParse,
    }

    /// Errors parsing `weights/gold_dep_en.conllu`.
    #[derive(Debug, thiserror::Error)]
    pub enum GoldParseError {
        /// A token row did not have the expected five tab-separated
        /// columns.
        #[error("malformed gold row {0:?}: expected 5 tab-separated columns")]
        MalformedRow(Box<str>),
        /// A row's head-index column was not a valid non-negative integer.
        #[error("invalid head index {0:?} in gold row")]
        InvalidHead(Box<str>),
        /// A row's relation column was not one of [`DepRelation`]'s
        /// canonical names.
        #[error(transparent)]
        Relation(#[from] ParseDepRelationError),
        /// The assembled edge list failed [`SentenceParse::new`]'s own
        /// consistency checks.
        #[error(transparent)]
        Sentence(#[from] DepParseError),
    }

    /// One parsed token row, before it is folded into a [`GoldSentence`].
    struct Row {
        surface: Box<str>,
        tag: Box<str>,
        /// 1-based; `0` means "this token is the sentence root".
        head: usize,
        relation: Box<str>,
    }

    /// Parses the tab-separated CoNLL-U-like gold format.
    ///
    /// `gold_dep_en.conllu` uses: `# sent_id = ...` / `# split = train|test`
    /// / `# text = ...` comment lines, then one `index<TAB>surface<TAB>
    /// penn_pos<TAB>head_index<TAB>relation` row per token (1-based indices,
    /// head `0` for root), a blank line ending each sentence.
    ///
    /// # Errors
    /// The first [`GoldParseError`] encountered — a malformed row, an
    /// unparseable head index, an unrecognized relation name, or an
    /// internally-inconsistent edge list (a self-head, an out-of-bounds
    /// head).
    pub fn parse_gold_file(text: &str) -> Result<Vec<GoldSentence>, GoldParseError> {
        let mut sentences = Vec::new();
        let mut split: Option<Box<str>> = None;
        let mut rows: Vec<Row> = Vec::new();

        for line in text.lines() {
            let line = line.trim_end();
            if line.is_empty() {
                if !rows.is_empty() {
                    sentences.push(build_gold_sentence(
                        split.take(),
                        &std::mem::take(&mut rows),
                    )?);
                }
                continue;
            }
            if let Some(rest) = line.strip_prefix("# split = ") {
                split = Some(Box::from(rest));
                continue;
            }
            if line.starts_with('#') {
                continue;
            }
            let mut cols = line.split('\t');
            let _index = cols.next();
            let malformed = || GoldParseError::MalformedRow(Box::from(line));
            let surface = cols.next().ok_or_else(malformed)?;
            let tag = cols.next().ok_or_else(malformed)?;
            let head_str = cols.next().ok_or_else(malformed)?;
            let relation = cols.next().ok_or_else(malformed)?;
            let head: usize = head_str
                .parse()
                .map_err(|_| GoldParseError::InvalidHead(Box::from(head_str)))?;
            rows.push(Row {
                surface: Box::from(surface.to_lowercase()),
                tag: Box::from(tag),
                head,
                relation: Box::from(relation),
            });
        }
        if !rows.is_empty() {
            sentences.push(build_gold_sentence(split.take(), &rows)?);
        }
        Ok(sentences)
    }

    fn build_gold_sentence(
        split: Option<Box<str>>,
        rows: &[Row],
    ) -> Result<GoldSentence, GoldParseError> {
        let words = rows.iter().map(|row| row.surface.clone()).collect();
        let tags = rows.iter().map(|row| row.tag.clone()).collect();
        let edges = rows
            .iter()
            .enumerate()
            .map(|(index, row)| {
                let head = if row.head == 0 {
                    None
                } else {
                    Some(row.head - 1)
                };
                let relation: DepRelation = row.relation.parse()?;
                Ok(DepEdge {
                    token: index,
                    head,
                    relation,
                    confidence: Confidence::CERTAIN,
                })
            })
            .collect::<Result<Vec<_>, GoldParseError>>()?;
        let parse = SentenceParse::new(edges)?;
        Ok(GoldSentence {
            split: split.unwrap_or_default(),
            words,
            tags,
            parse,
        })
    }

    /// A minimal from-scratch averaged structured perceptron.
    ///
    /// Scores the fixed 42-way action space and follows the same lazy-
    /// averaging trick as
    /// [`crate::tag_perceptron::train_support::AveragedPerceptron`]
    /// (accumulate `weight * (iterations held)` and divide by the total
    /// iteration count at the end), keyed by a `u8` action index instead of
    /// a string class name since this classifier's class set is fixed at
    /// compile time rather than discovered from data.
    #[derive(Default)]
    pub struct AveragedPerceptron {
        weights: BTreeMap<Box<str>, BTreeMap<u8, f64>>,
        totals: BTreeMap<(Box<str>, u8), f64>,
        timestamps: BTreeMap<(Box<str>, u8), u64>,
        iterations: u64,
    }

    impl AveragedPerceptron {
        #[must_use]
        pub fn new() -> Self {
            Self::default()
        }

        /// Sums every feature's live (non-averaged) per-action weights,
        /// cast to `f32` at the point of summation — the same precision
        /// [`super::WeightTable::score`] scores with at inference, so
        /// training-time masked prediction sees the same rounding behavior
        /// the shipped artifact will.
        fn score(&self, features: &[Box<str>]) -> [f32; NUM_ACTIONS] {
            let mut scores = [0f32; NUM_ACTIONS];
            for feature in features {
                if let Some(per_action) = self.weights.get(feature) {
                    for (&action, &weight) in per_action {
                        #[allow(clippy::cast_possible_truncation)]
                        {
                            scores[usize::from(action)] += weight as f32;
                        }
                    }
                }
            }
            scores
        }

        fn bump(&mut self, feature: &str, action: u8, delta: f64) {
            let key: (Box<str>, u8) = (Box::from(feature), action);
            let current = self
                .weights
                .get(feature)
                .and_then(|m| m.get(&action))
                .copied()
                .unwrap_or(0.0);
            let last_ts = self.timestamps.get(&key).copied().unwrap_or(0);
            // Training runs for a handful of epochs over a few thousand
            // sentences, so `iterations` never comes close to 2^52 — this
            // conversion never loses meaningful precision.
            #[allow(clippy::cast_precision_loss)]
            let held = (self.iterations - last_ts) as f64;
            let total = self.totals.entry(key.clone()).or_insert(0.0);
            *total = held.mul_add(current, *total);
            self.timestamps.insert(key, self.iterations);
            self.weights
                .entry(Box::from(feature))
                .or_default()
                .insert(action, current + delta);
        }

        /// One early-update training step: scores `features` under
        /// `config`'s legal-action mask using the model's current live
        /// weights, and if the masked argmax disagrees with `gold_index`,
        /// bumps `gold_index` up and the wrong prediction down for every
        /// feature. Returns the predicted action index.
        fn train_one(
            &mut self,
            features: &[Box<str>],
            gold_index: usize,
            config: &Configuration,
        ) -> Option<usize> {
            self.iterations += 1;
            let scores = self.score(features);
            let predicted = best_allowed_action(&scores, config);
            if predicted != Some(gold_index) {
                for feature in features {
                    self.bump(
                        feature,
                        u8::try_from(gold_index).expect("gold_index < NUM_ACTIONS <= 255"),
                        1.0,
                    );
                    if let Some(p) = predicted {
                        self.bump(
                            feature,
                            u8::try_from(p).expect("p < NUM_ACTIONS <= 255"),
                            -1.0,
                        );
                    }
                }
            }
            predicted
        }

        /// Averages every weight the standard way and emits the on-disk
        /// [`WeightFile`] shape.
        #[must_use]
        pub fn finish(mut self) -> WeightFile {
            let mut features: BTreeMap<Box<str>, Vec<(u8, f32)>> = BTreeMap::new();
            let keys: Vec<Box<str>> = self.weights.keys().cloned().collect();
            for feature in keys {
                let per_action = self.weights.remove(&feature).unwrap_or_default();
                let mut sparse = Vec::new();
                for (action, weight) in per_action {
                    let key = (feature.clone(), action);
                    let last_ts = self.timestamps.get(&key).copied().unwrap_or(0);
                    let total = self.totals.get(&key).copied().unwrap_or(0.0);
                    // See the identical justification in `bump` above.
                    #[allow(clippy::cast_precision_loss)]
                    let held = (self.iterations - last_ts) as f64;
                    let total = held.mul_add(weight, total);
                    #[allow(clippy::cast_precision_loss)]
                    let averaged = total / (self.iterations.max(1) as f64);
                    if averaged.abs() > f64::EPSILON {
                        #[allow(clippy::cast_possible_truncation)]
                        sparse.push((action, averaged as f32));
                    }
                }
                if !sparse.is_empty() {
                    features.insert(feature, sparse);
                }
            }
            WeightFile { features }
        }
    }

    /// Runs one sentence through greedy oracle-guided decoding with early update.
    ///
    /// See the module docs' "Training" section. Stops the instant `model`'s
    /// masked prediction disagrees with the oracle, having already applied
    /// that step's weight update. Returns whether the model matched the
    /// oracle all the way to a terminal configuration (i.e. this sentence
    /// needed no update at all this epoch).
    pub fn train_sentence(
        model: &mut AveragedPerceptron,
        words: &[Box<str>],
        tags: &[Box<str>],
        gold: &SentenceParse,
    ) -> bool {
        let mut config = Configuration::new(gold.edges().len());
        loop {
            let Some(oracle_transition) = oracle(gold, &config) else {
                return true;
            };
            let features = extract_features(words, tags, &config);
            let gold_index = action_index(oracle_transition);
            let predicted = model.train_one(&features, gold_index, &config);
            if predicted != Some(gold_index) {
                return false;
            }
            config.apply(oracle_transition);
        }
    }

    /// Re-exposes [`action_at`] for a training tool's own diagnostics
    /// (e.g. reporting a per-action breakdown of update frequency) without
    /// duplicating the fixed action enumeration.
    #[must_use]
    pub fn action_at_index(index: usize) -> crate::dep_arceager::Transition {
        action_at(index)
    }

    /// Re-exposes [`NONE`], the feature sentinel, for a training tool's own
    /// diagnostics.
    pub const NONE_SENTINEL: &str = NONE;

    /// Gzip-compresses `file` as JSON and writes it to `path`.
    ///
    /// # Errors
    /// Any I/O or serialization failure.
    pub fn save_weight_file(file: &WeightFile, path: &std::path::Path) -> std::io::Result<()> {
        let json = serde_json::to_string(file).expect("WeightFile always serializes");
        let out = std::fs::File::create(path)?;
        let mut encoder = flate2::write::GzEncoder::new(out, flate2::Compression::best());
        encoder.write_all(json.as_bytes())?;
        encoder.finish()?;
        Ok(())
    }

    /// Loads a gzip-compressed [`WeightFile`] from `path` — used by a
    /// training tool's own reproducibility check (train twice, compare the
    /// two on-disk artifacts).
    ///
    /// # Errors
    /// [`PerceptronParseError::Decompress`] / [`PerceptronParseError::Parse`].
    pub fn load_weight_file(path: &std::path::Path) -> Result<WeightFile, PerceptronParseError> {
        let bytes = std::fs::read(path).map_err(PerceptronParseError::Decompress)?;
        let mut decoder = flate2::read::GzDecoder::new(bytes.as_slice());
        let mut json = String::new();
        std::io::Read::read_to_string(&mut decoder, &mut json)
            .map_err(PerceptronParseError::Decompress)?;
        serde_json::from_str(&json).map_err(PerceptronParseError::Parse)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// [`action_index`]/[`action_at`] round-trip for every one of the 42
    /// actions, and the enumeration is exactly 42 entries — the load-bearing
    /// contract the shipped artifact was trained against.
    #[test]
    fn action_index_and_action_at_round_trip_every_action() {
        assert_eq!(NUM_ACTIONS, 42);
        for index in 0..NUM_ACTIONS {
            let transition = action_at(index);
            assert_eq!(
                action_index(transition),
                index,
                "round-trip failed for index {index}"
            );
        }
    }

    /// [`action_index`] never collides two distinct actions onto the same
    /// index.
    #[test]
    fn action_index_is_injective() {
        let mut seen = std::collections::HashSet::new();
        for index in 0..NUM_ACTIONS {
            assert!(seen.insert(action_index(action_at(index))));
        }
    }

    /// [`bucket_distance`]'s boundaries land exactly where documented.
    #[test]
    fn bucket_distance_boundaries() {
        assert_eq!(bucket_distance(1), "1");
        assert_eq!(bucket_distance(5), "5");
        assert_eq!(bucket_distance(6), "6-10");
        assert_eq!(bucket_distance(10), "6-10");
        assert_eq!(bucket_distance(11), ">10");
        assert_eq!(bucket_distance(1000), ">10");
    }

    /// [`extract_features`] always returns exactly 30 features, even for
    /// the empty-stack, single-token-buffer initial configuration where
    /// every "child" and most "neighbor" positions are absent.
    #[test]
    fn extract_features_always_returns_thirty_features() {
        let words: Vec<Box<str>> = vec![Box::from("the"), Box::from("cat"), Box::from("sat")];
        let tags: Vec<Box<str>> = vec![Box::from("DT"), Box::from("NN"), Box::from("VBD")];
        let config = Configuration::new(3);
        let feats = extract_features(&words, &tags, &config);
        assert_eq!(feats.len(), 30);
        // Sentinel values show up for the empty stack.
        assert!(feats.contains(&Box::from("s0.w=<none>")));
        assert!(feats.contains(&Box::from("s1.w=<none>")));
        // The buffer front is populated.
        assert!(feats.contains(&Box::from("b0.w=the")));
        assert!(feats.contains(&Box::from("b0.t=DT")));
    }

    /// Loading the embedded weight artifact never panics — covers the
    /// gunzip/JSON round trip end to end, independent of how accurate the
    /// resulting model is.
    #[test]
    fn embedded_weight_artifact_loads() {
        PerceptronParser::new().expect("embedded weights must load");
    }
}
