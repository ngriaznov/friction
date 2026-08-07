//! [`PerceptronParser`]: an averaged-structured-perceptron [`DepParser`]
//! that drives [`crate::dep_arceager::Configuration`] at inference time —
//! the classifier `dep_arceager`'s module docs describe as "later, an
//! actual `DepParser` implementation".
//!
//! # Action space
//!
//! Every decision is one of 42 actions: [`Transition::Shift`],
//! [`Transition::Reduce`], and [`Transition::LeftArc`]/[`Transition::RightArc`]
//! over each of the 20 non-[`DepRelation::Root`] relations. [`action_index`]/
//! [`action_at`] fix a single, deterministic enumeration ([`RELATIONS`], in
//! [`DepRelation`]'s declaration order), so "lowest action index" is a
//! stable tie-break, not an accident of iteration order.
//!
//! # Feature templates
//!
//! Fixed and load-bearing: the vendored artifact
//! (`weights/parser_en.json.gz`) was trained against the exact list below,
//! keyed by a template tag so coincidentally equal *values* never collide.
//! Changing this list invalidates the artifact. `s0`/`s1` are the top two
//! stack items, `b0`/`b1`/`b2` the first three buffer items; `w` is a
//! token's lowercased text, `t` its Penn tag. An absent position
//! contributes [`NONE`] rather than being omitted, so every step emits
//! the same fixed-length list.
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
//! Thirty feature strings per decision, always, in this fixed order —
//! [`extract_features`] is the single source of truth for the format.
//!
//! # Training: structured perceptron with early update
//!
//! [`train_support::train_sentence`] decodes greedily from a fresh
//! [`Configuration`], scoring each step with the model's live (unaveraged)
//! weights masked to [`Configuration::is_allowed`]. The instant the masked
//! argmax disagrees with [`crate::dep_arceager::oracle`], the oracle's
//! action is promoted and the wrong pick demoted for every feature of that
//! step, and the sentence is abandoned there. The rest contributes no
//! signal, since the model never saw it. A sentence the model gets
//! entirely right contributes a full walk of updates that never fire.
//! Final weights use the standard lazy-averaged perceptron (see
//! [`train_support::AveragedPerceptron`]), the same
//! accumulate-by-timestamp trick [`crate::tag_perceptron`]'s trainer uses,
//! adapted to a fixed `u8` action index.
//!
//! Only sentences [`crate::dep_arceager::derive`] can actually reproduce
//! reach training — a non-derivable (in practice non-projective) gold
//! tree would silently teach a transition sequence that doesn't
//! reconstruct the parse it was supposed to learn.
//!
//! # Determinism
//!
//! No randomness anywhere: [`train_support`]'s epoch loop walks
//! `weights/gold_dep_en.conllu` in its own on-disk order (no shuffling),
//! matching `examples/train_perceptron.rs`. At inference,
//! [`best_allowed_action`] breaks ties toward the lowest action index,
//! and feature order is fixed, so summation, and the resulting scores,
//! is identical every run.
//!
//! # Two loading paths, one inference surface
//!
//! Like [`crate::tag_perceptron::PerceptronTagger`], [`PerceptronParser::new`]
//! (fast, embedded `.bin`) and [`PerceptronParser::from_json_gz`] (slow,
//! audited-source) build different internal representations — a zero-copy
//! [`ViewBackend`] versus a `HashMap`-based [`OwnedBackend`] — behind the
//! shared [`Backend`] trait, so [`DepParser::parse`]'s per-step scoring
//! loop is written once regardless of which backend loaded.
//!
//! # The embedded `.bin`'s version-2 payload: a serialized hash table
//!
//! [`PerceptronParser::new`] loads `parser_en.bin` (embedded as
//! [`WEIGHTS_BIN`]) with zero per-key construction — see
//! `crate::weights_bin`'s module docs for the general scheme and
//! [`crate::tag_perceptron`]'s own module docs for the sibling artifact
//! that motivated it. This module's payload (the bytes after
//! `crate::weights_bin`'s shared header), built by [`build_view_payload`]
//! and read back by [`ParserView::from_payload`], is simpler than the
//! tagger's — no classes or tagdict, just one feature table:
//!
//! - `stride: u32` — always [`NUM_ACTIONS`] (42), recorded and checked at
//!   load time as a cheap sanity check that a payload really is this
//!   artifact's shape
//! - `features_capacity: u32`, then the features hash table's slots as a
//!   length-prefixed section (feature string -> a
//!   [`crate::weights_bin::sparse_row`] byte offset into the values
//!   section)
//! - the features values section: every feature's sparse `(action_index:
//!   u16, weight: f32)` row (see [`crate::weights_bin::build_sparse_row`])
//!   — sparse, not one dense `f32` per action, since the average feature
//!   only carries a handful of nonzero action weights (measured: ~2.8 of
//!   42) and a dense row would multiply artifact size roughly 15x for no
//!   benefit
//! - the string pool: every feature string, in the fixed order it was
//!   pushed during packing (`WeightFile.features`'s own ascending
//!   `BTreeMap` order)
//!
//! Determinism: [`build_view_payload`] feeds the table builder entries in
//! that same fixed ascending order, so packing the same `WeightFile`
//! twice produces bit-identical bytes — pinned by
//! `pack_perceptron_parser_bin_is_deterministic` below.

use std::collections::HashMap;
use std::io::Read as _;

use friction_core::Lang;
use serde::{Deserialize, Serialize};

use crate::dep::{DepParseError, DepParser, DepRelation, SentenceParse};
use crate::dep_arceager::{Configuration, Transition};
use crate::tag::TaggedToken;

/// The vendored, gzip-compressed dependency-parser weight artifact: the
/// audited interchange format from the training pipeline. See
/// `weights/NOTICE.md` for provenance and the reproduction command;
/// `examples/train_parser.rs` (behind the `train-tooling` feature)
/// regenerates it.
///
/// `cfg(test)`-only: normal startup uses the faster [`WEIGHTS_BIN`] path
/// instead (see that constant's own docs), so a production binary has no
/// reason to also carry this now-redundant copy. The only in-tree callers
/// of [`PerceptronParser::from_json_gz`] and `pack_perceptron_parser_bin`
/// with these exact bytes are this module's own parity tests below;
/// `corpus-tool weights-pack` calls `pack_perceptron_parser_bin` with
/// bytes it reads fresh from disk instead, since regenerating this
/// artifact is exactly the case where the newly-retrained `json.gz` on
/// disk and whatever is currently compiled into this static disagree.
#[cfg(test)]
static WEIGHTS_JSON_GZ: &[u8] = include_bytes!("../weights/parser_en.json.gz");

/// The derived binary weight artifact [`PerceptronParser::new`] loads at
/// normal startup: a version-2 serialized hash-table view (this module's
/// own docs describe the exact byte layout), embedded raw — see
/// `crate::weights_bin`'s module docs for the shared header shape and
/// `weights/NOTICE.md`'s "derived artifacts" section for how it's
/// produced (`corpus-tool weights-pack`) from `WEIGHTS_JSON_GZ`.
static WEIGHTS_BIN: &[u8] = include_bytes!("../weights/parser_en.bin");

/// This artifact's 8-byte magic, distinct from [`crate::tag_perceptron`]'s
/// own: so one can never be silently loaded in place of the other.
const ARTIFACT_MAGIC: [u8; 8] = *b"FRPARWT1";

/// The sha256 (lowercase hex) of the `WEIGHTS_JSON_GZ` bytes currently
/// embedded above, fixed at compile time. [`PerceptronParser::new`]
/// compares this against `parser_en.bin`'s own recorded source sha256
/// (from its header — see `crate::weights_bin`) so a `.bin` left stale
/// after a `parser_en.json.gz` regeneration fails loudly at load time
/// instead of silently drifting. Pinned by
/// `embedded_json_gz_sha256_matches_compiled_in_constant` below, which
/// recomputes it directly from the embedded bytes.
const SOURCE_JSON_GZ_SHA256: &str =
    "60e3c2ebcc98474769c07747ce7a297dc1c0c6ad120fe287cfc6e132ce3d4b08";

/// Sentinel for an absent stack/buffer position, dependent, or distance
/// bucket: distinct from any real lowercased word or Penn tag, so a
/// feature built from it is never confused with a genuine one.
const NONE: &str = "<none>";

/// The 20 relations [`Transition::LeftArc`]/[`Transition::RightArc`] may
/// carry, in [`DepRelation`]'s declaration order — the enumeration
/// [`action_index`]/[`action_at`] build the 42-way action space from.
/// [`DepRelation::Root`] is excluded: never a legal arc label.
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
/// Only [`action_index`] calls this, for training and round-trip tests
/// (inference only goes index-to-transition via [`action_at`]), so it's
/// gated the same way.
///
/// # Panics
/// Panics if `rel` is [`DepRelation::Root`]: every caller passes a
/// relation carried by [`Transition::LeftArc`]/[`Transition::RightArc`],
/// and [`crate::dep_arceager::oracle`] never labels those `Root`.
#[cfg(any(test, feature = "train-tooling"))]
fn relation_index(rel: DepRelation) -> usize {
    RELATIONS
        .iter()
        .position(|&candidate| candidate == rel)
        .expect("DepRelation::Root is never a legal LeftArc/RightArc label")
}

/// `transition`'s position in the fixed 42-way action enumeration —
/// [`action_at`]'s exact inverse. Only training and round-trip tests need
/// this direction. Inference only goes index -> transition.
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
/// Panics if `index` is out of range for [`NUM_ACTIONS`] — every caller
/// passes an index produced by [`action_index`] or scanned from
/// `0..NUM_ACTIONS`.
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

fn at<'a>(words: &'a [Box<str>], tags: &'a [&'a str], idx: Option<usize>) -> View<'a> {
    idx.map_or(View { w: NONE, t: NONE }, |i| View {
        w: &words[i],
        t: tags[i],
    })
}

/// `head`'s leftmost and rightmost assigned dependents in `config`, by
/// token index — an O(1) lookup: [`Configuration`] maintains the bound
/// incrementally as arcs are created (see [`Configuration::children_of`]),
/// so this no longer needs its own scan.
fn children_of(config: &Configuration, head: usize) -> (Option<usize>, Option<usize>) {
    config.children_of(head)
}

/// Emits `{prefix}.t=` and `{prefix}.rel=` features for `child`'s tag and
/// its arc relation (the label attaching it to its own head) — [`NONE`]
/// for both if `child` is absent. Each feature is written into `buf`
/// (cleared first) and handed to `emit` as a borrow — see [`build_features`]
/// for why a shared, reused buffer replaces a `format!` allocation per
/// feature.
fn push_child_features(
    buf: &mut String,
    emit: &mut impl FnMut(&str),
    prefix: &str,
    config: &Configuration,
    tags: &[&str],
    child: Option<usize>,
) {
    let (tag, relation) = child.map_or((NONE, NONE), |index| {
        let relation = config.arc(index).map_or(NONE, |(_, rel)| rel.as_str());
        (tags[index], relation)
    });
    // Direct byte pushes, not `write!` — see `build_features`' own note.
    buf.clear();
    buf.push_str(prefix);
    buf.push_str(".t=");
    buf.push_str(tag);
    emit(buf.as_str());
    buf.clear();
    buf.push_str(prefix);
    buf.push_str(".rel=");
    buf.push_str(relation);
    emit(buf.as_str());
}

/// `diff` (buffer-minus-stack distance, always positive — the buffer only
/// holds indices past every stacked one) bucketed into `1`, `2`, `3`, `4`,
/// `5`, `6-10`, `>10`.
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

/// Builds one configuration's fixed, 30-entry feature list, invoking
/// `emit` once per feature in the fixed order the module docs describe —
/// this function is the single source of truth for the template
/// catalogue.
///
/// Every feature is formatted into `buf` (cleared right before) rather
/// than a fresh `String`, so `emit` only ever sees one live borrow at a
/// time. Callers that drive this once per parse step with the *same*
/// `buf` across the whole sentence (as [`DepParser::parse`]'s scoring
/// loop does) get zero-allocation feature construction after `buf`
/// stabilizes at its high-water mark; [`extract_features`] below is the
/// owned, allocating wrapper training and tests use instead.
///
/// `words`/`tags` are the whole sentence's lowercased text and tags,
/// indexed by token — the same slices across one sentence's decode, since
/// only `config` changes step to step.
fn build_features(
    words: &[Box<str>],
    tags: &[&str],
    config: &Configuration,
    buf: &mut String,
    mut emit: impl FnMut(&str),
) {
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

    // Direct byte pushes, not `write!` — every part is already a `&str`,
    // and the `core::fmt` machinery measured ~17% of a large fix run's
    // instructions across the two perceptrons' feature builders.
    // `push_str` concatenation of the identical parts produces
    // byte-identical feature keys with none of that overhead.
    macro_rules! feat {
        ($($part:expr),+ $(,)?) => {{
            buf.clear();
            $(buf.push_str($part);)+
            emit(buf.as_str());
        }};
    }

    // Unigrams.
    feat!("s0.w=", s0v.w);
    feat!("s0.t=", s0v.t);
    feat!("s0.wt=", s0v.w, "|", s0v.t);
    feat!("s1.w=", s1v.w);
    feat!("s1.t=", s1v.t);
    feat!("s1.wt=", s1v.w, "|", s1v.t);
    feat!("b0.w=", b0v.w);
    feat!("b0.t=", b0v.t);
    feat!("b0.wt=", b0v.w, "|", b0v.t);
    feat!("b1.w=", b1v.w);
    feat!("b1.t=", b1v.t);
    feat!("b1.wt=", b1v.w, "|", b1v.t);
    feat!("b2.t=", b2v.t);

    // Pairs.
    feat!("s0b0.wtwt=", s0v.w, "|", s0v.t, "|", b0v.w, "|", b0v.t);
    feat!("s0b0.wtw=", s0v.w, "|", s0v.t, "|", b0v.w);
    feat!("s0b0.wwt=", s0v.w, "|", b0v.w, "|", b0v.t);
    feat!("s0b0.wtt=", s0v.w, "|", s0v.t, "|", b0v.t);
    feat!("s0b0.twt=", s0v.t, "|", b0v.w, "|", b0v.t);
    feat!("s0b0.ww=", s0v.w, "|", b0v.w);
    feat!("s0b0.tt=", s0v.t, "|", b0v.t);
    feat!("b0b1.tt=", b0v.t, "|", b1v.t);

    // Triples.
    feat!("s0b0b1.ttt=", s0v.t, "|", b0v.t, "|", b1v.t);
    feat!("s1s0b0.ttt=", s1v.t, "|", s0v.t, "|", b0v.t);

    // Children.
    let (s0_left, s0_right) = s0.map_or((None, None), |head| children_of(config, head));
    push_child_features(buf, &mut emit, "s0.lc", config, tags, s0_left);
    push_child_features(buf, &mut emit, "s0.rc", config, tags, s0_right);
    let b0_left = b0.and_then(|head| children_of(config, head).0);
    push_child_features(buf, &mut emit, "b0.lc", config, tags, b0_left);

    // Distance.
    let dist_bucket = match (s0, b0) {
        (Some(si), Some(bi)) => bucket_distance(bi - si),
        _ => NONE,
    };
    feat!("dist.s0t.b0t=", dist_bucket, "|", s0v.t, "|", b0v.t);
}

/// Owned, allocating wrapper over [`build_features`]: collects every
/// feature into a freshly boxed `Vec<Box<str>>`, one heap allocation per
/// feature (30 per call) — used by training (which needs owned keys to
/// update a `BTreeMap`-backed weight table across many sentences) and by
/// this module's own tests. [`DepParser::parse`]'s hot loop does not call
/// this: it drives [`build_features`] directly with a reused scratch
/// buffer and an immediate per-feature scoring callback, so no `Box<str>`
/// or intermediate `String` is ever allocated per feature there. Gated
/// like [`relation_index`]/[`action_index`] above: nothing in a normal,
/// non-training build calls this anymore.
#[cfg(any(test, feature = "train-tooling"))]
fn extract_features(
    words: &[Box<str>],
    tags: &[Box<str>],
    config: &Configuration,
) -> Vec<Box<str>> {
    let mut feats = Vec::with_capacity(30);
    let mut buf = String::with_capacity(64);
    // Training/tests still pass owned `Box<str>` tags (mirroring the gold
    // file's own storage); only the hot inference path in `DepParser::parse`
    // avoids that allocation. Borrow here instead of threading `Box<str>`
    // back into `build_features`.
    let tag_refs: Vec<&str> = tags.iter().map(Box::as_ref).collect();
    build_features(words, &tag_refs, config, &mut buf, |s| {
        feats.push(Box::from(s));
    });
    feats
}

/// The highest-scoring action [`Configuration::is_allowed`] permits, ties
/// broken toward the lowest index — `None` only if no action is legal,
/// which never happens while `config` is not terminal (a non-empty buffer
/// always makes `Shift` legal; an exhausted, non-terminal buffer always
/// makes `Reduce` legal).
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

/// On-disk shape of the weight artifact: `BTreeMap` for stable, diffable
/// JSON across re-training runs; sparse per-feature action list since
/// most features only fire for a handful of the 42 actions.
///
/// `pub` but not exported at the crate root, purely so
/// `train_support::AveragedPerceptron::finish`,
/// [`PerceptronParser::from_weight_file`], and `examples/train_parser.rs`
/// can pass it around — mirrors [`crate::tag_perceptron`]'s own
/// `WeightFile`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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

    /// Adds `feature`'s dense per-action weight array (if any) into
    /// `scores` — [`Backend::accumulate`]'s single-feature step, called
    /// once per feature straight out of [`build_features`]'s reused
    /// scratch buffer (no owned `Vec<Box<str>>` of features ever built on
    /// this path). Summation order — and so the resulting scores — is
    /// identical across runs since it's fixed by the caller's fixed
    /// feature-template order, never by hash-map iteration (mirrors
    /// [`crate::tag_perceptron`]'s own `WeightTable::score`).
    fn accumulate_one(&self, feature: &str, scores: &mut [f32; NUM_ACTIONS]) {
        if let Some(dense) = self.by_feature.get(feature) {
            for (slot, weight) in scores.iter_mut().zip(dense.iter()) {
                *slot += weight;
            }
        }
    }
}

/// A zero-copy view over a version-2 parser payload's embedded bytes —
/// see this module's own docs for the exact byte layout
/// [`build_view_payload`] writes and [`Self::from_payload`] reads back.
/// Borrows directly from the `'static` embedded artifact ([`WEIGHTS_BIN`]);
/// building one does no per-key work at all.
struct ParserView<'a> {
    features: crate::weights_bin::HashView<'a>,
    values: &'a [u8],
}

impl<'a> ParserView<'a> {
    /// Parses `payload` (the bytes [`crate::weights_bin::split_header`]
    /// returned) into a view.
    ///
    /// # Errors
    /// [`crate::weights_bin::WeightsBinError::MalformedPayload`] if any
    /// section is missing or truncated, if an embedded hash table's slot
    /// buffer doesn't match its own recorded capacity, or if the
    /// recorded stride doesn't match this build's [`NUM_ACTIONS`].
    fn from_payload(payload: &'a [u8]) -> Result<Self, crate::weights_bin::WeightsBinError> {
        use crate::weights_bin::{Cursor, HashView};

        let mut cursor = Cursor::new(payload);
        let stride = cursor.read_u32()? as usize;
        if stride != NUM_ACTIONS {
            return Err(crate::weights_bin::WeightsBinError::MalformedPayload(
                "payload's recorded stride does not match this build's NUM_ACTIONS",
            ));
        }
        let features_capacity = cursor.read_u32()?;
        let features_slots = cursor.read_section()?;
        let values = cursor.read_section()?;
        let pool = cursor.read_section()?;

        let features = HashView::new(features_capacity, features_slots, pool)?;
        Ok(Self { features, values })
    }

    /// Adds `feature`'s sparse per-action weight row (if any) into
    /// `scores` — reads directly from the payload's bytes, no owned key
    /// required, so [`Backend::accumulate`] can drive this with a
    /// borrowed feature straight out of a reused scratch buffer.
    fn accumulate_one(&self, feature: &[u8], scores: &mut [f32; NUM_ACTIONS]) {
        if let Some(row_offset) = self.features.get(feature) {
            for chunk in crate::weights_bin::sparse_row(self.values, row_offset).chunks_exact(6) {
                let (action_index, weight) = crate::weights_bin::decode_sparse_entry(chunk);
                scores[usize::from(action_index)] += weight;
            }
        }
    }

    /// Sums every matched feature's sparse per-action weight row over a
    /// caller-supplied feature list — mirrors [`WeightTable::accumulate_one`]'s
    /// fixed-order summation, just reading each row's nonzero entries from
    /// the payload instead of a pre-built dense array. Only this module's
    /// own full-vocabulary parity tests drive the view this way (a list of
    /// already-owned features); [`PerceptronParser::parse`]'s hot loop
    /// instead calls [`Self::accumulate_one`] directly from
    /// [`build_features`]'s callback.
    #[cfg(test)]
    fn score(&self, features: &[Box<str>]) -> [f32; NUM_ACTIONS] {
        let mut scores = [0f32; NUM_ACTIONS];
        for feature in features {
            self.accumulate_one(feature.as_bytes(), &mut scores);
        }
        scores
    }
}

/// Builds a version-2 parser payload's bytes (everything after
/// `crate::weights_bin`'s shared header) from `file` — this module's own
/// docs describe the exact byte layout. The inverse of
/// [`ParserView::from_payload`].
///
/// Feeds the feature table builder entries in `file.features`'s own
/// ascending-key `BTreeMap` order, so calling this twice on an equal
/// `WeightFile` produces bit-identical bytes (pinned by
/// `pack_perceptron_parser_bin_is_deterministic` below).
fn build_view_payload(file: WeightFile) -> Vec<u8> {
    let WeightFile { features } = file;
    let mut pool = Vec::new();
    let mut values = Vec::new();
    let mut feature_entries: Vec<(&str, u32, u32, u32)> = Vec::with_capacity(features.len());
    for (feature, sparse) in &features {
        let translated: Vec<(u16, f32)> = sparse
            .iter()
            .map(|&(action_index, weight)| (u16::from(action_index), weight))
            .collect();
        let row_offset = crate::weights_bin::build_sparse_row(&mut values, &translated);
        let (key_off, key_len) = crate::weights_bin::push_key(&mut pool, feature);
        feature_entries.push((feature.as_ref(), key_off, key_len, row_offset));
    }
    let (features_capacity, features_slots) =
        crate::weights_bin::build_open_table(&feature_entries);

    let mut out = Vec::new();
    crate::weights_bin::write_u32(
        &mut out,
        u32::try_from(NUM_ACTIONS).expect("NUM_ACTIONS fits u32"),
    );
    crate::weights_bin::write_u32(&mut out, features_capacity);
    crate::weights_bin::write_section(&mut out, &features_slots);
    crate::weights_bin::write_section(&mut out, &values);
    crate::weights_bin::write_section(&mut out, &pool);
    out
}

/// Shared inference surface both loading paths implement identically:
/// [`PerceptronParser::new`]'s zero-copy [`ViewBackend`] and
/// [`PerceptronParser::from_json_gz`]'s owned [`OwnedBackend`]. Only one
/// method: unlike the tagger, a parser step's output is already a plain
/// action index (no class string, no tagdict bypass to resolve). Takes
/// one feature at a time (rather than a collected slice) so
/// [`DepParser::parse`]'s hot loop can drive it straight from
/// [`build_features`]'s per-feature callback — no owned `Vec<Box<str>>`
/// of features is ever built on that path.
trait Backend {
    fn accumulate(&self, feature: &str, scores: &mut [f32; NUM_ACTIONS]);
}

/// [`Backend`] over an in-memory, `HashMap`-based [`WeightTable`] — built
/// by [`PerceptronParser::from_json_gz`], the slower audited-source path.
struct OwnedBackend {
    weights: WeightTable,
}

impl Backend for OwnedBackend {
    fn accumulate(&self, feature: &str, scores: &mut [f32; NUM_ACTIONS]) {
        self.weights.accumulate_one(feature, scores);
    }
}

/// [`Backend`] over a zero-copy [`ParserView`] — built by
/// [`PerceptronParser::new`], the fast path every normal caller wants.
struct ViewBackend {
    view: ParserView<'static>,
}

impl Backend for ViewBackend {
    fn accumulate(&self, feature: &str, scores: &mut [f32; NUM_ACTIONS]) {
        self.view.accumulate_one(feature.as_bytes(), scores);
    }
}

/// Errors constructing a [`PerceptronParser`].
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum PerceptronParseError {
    /// A `json.gz` artifact failed to gunzip.
    #[error("failed to decompress the dependency-parser weight artifact: {0}")]
    Decompress(#[source] std::io::Error),
    /// A decompressed `json.gz` artifact was not valid JSON in the
    /// expected shape.
    #[error("failed to parse the dependency-parser weight artifact: {0}")]
    Parse(#[source] serde_json::Error),
    /// The embedded `.bin` artifact's header was malformed — see
    /// [`crate::weights_bin::WeightsBinError`]'s variants.
    #[error(transparent)]
    Header(#[from] crate::weights_bin::WeightsBinError),
    /// The embedded `.bin` artifact's recorded source sha256 doesn't
    /// match [`SOURCE_JSON_GZ_SHA256`], the compile-time sha256 of the
    /// currently embedded `parser_en.json.gz`: the `.bin` is stale (the
    /// `json.gz` was regenerated but `corpus-tool weights-pack` wasn't
    /// re-run) and must not be trusted.
    #[error(
        "the embedded parser_en.bin's recorded source sha256 ({found}) does not match \
         parser_en.json.gz's compiled-in sha256 ({expected}) — regenerate parser_en.bin with \
         `corpus-tool weights-pack`"
    )]
    StaleBin {
        found: Box<str>,
        expected: &'static str,
    },
}

/// Gunzips `bytes` and parses the result as a [`WeightFile`]. The one
/// shared JSON-loading step behind [`PerceptronParser::from_json_gz`] and
/// [`pack_perceptron_parser_bin`].
fn parse_json_gz(bytes: &[u8]) -> Result<WeightFile, PerceptronParseError> {
    let mut decoder = flate2::read::GzDecoder::new(bytes);
    let mut json = String::new();
    decoder
        .read_to_string(&mut json)
        .map_err(PerceptronParseError::Decompress)?;
    serde_json::from_str(&json).map_err(PerceptronParseError::Parse)
}

/// Builds the derived `.bin` artifact's bytes.
///
/// Takes a `json.gz` weight artifact's raw bytes and its own sha256
/// (lowercase hex, computed by the caller — `corpus-tool weights-pack`
/// over the file it just read). Parses `json_gz_bytes` the same way
/// [`PerceptronParser::from_json_gz`] does, builds the version-2 view
/// payload via [`build_view_payload`], and prepends the shared
/// [`crate::weights_bin`] header carrying `source_sha256_hex` — the exact
/// bytes `PerceptronParser::new` later validates against
/// [`SOURCE_JSON_GZ_SHA256`]. The payload is embedded raw (uncompressed):
/// see [`crate::tag_perceptron::pack_perceptron_tagger_bin`]'s own docs
/// for why (the same reasoning applies to both artifacts).
///
/// # Errors
/// [`PerceptronParseError::Decompress`] / [`PerceptronParseError::Parse`]
/// — see [`PerceptronParser::from_json_gz`].
pub fn pack_perceptron_parser_bin(
    json_gz_bytes: &[u8],
    source_sha256_hex: &str,
) -> Result<Vec<u8>, PerceptronParseError> {
    let file = parse_json_gz(json_gz_bytes)?;
    let payload = build_view_payload(file);
    let mut out = crate::weights_bin::write_header(ARTIFACT_MAGIC, source_sha256_hex);
    out.extend_from_slice(&payload);
    Ok(out)
}

/// A [`DepParser`] backed by a from-scratch-trained averaged structured
/// perceptron.
///
/// See the module docs for the action space, feature templates, training
/// procedure, determinism guarantees, and the two loading paths' shared
/// [`Backend`] surface.
pub struct PerceptronParser {
    backend: Box<dyn Backend + Send + Sync>,
}

impl PerceptronParser {
    /// Loads the parser for [`Lang::En`], the only supported language in
    /// v1 — shorthand for `Self::for_lang(Lang::En)`.
    ///
    /// # Errors
    /// See [`Self::for_lang`].
    pub fn new() -> Result<Self, PerceptronParseError> {
        Self::for_lang(Lang::En)
    }

    /// Loads the parser for `lang`, from its embedded, derived `.bin`
    /// artifact — a version-2 serialized hash-table view (see this
    /// module's docs for the byte layout), gated on a source-sha256
    /// header check against the embedded `json.gz`
    /// (`crate::weights_bin`'s module docs). This is the fast path every
    /// normal caller wants: [`ParserView::from_payload`] only slices the
    /// embedded `'static` bytes, with no `HashMap` built and no
    /// per-feature array allocated. See [`Self::from_json_gz`] for the
    /// slower, audited-source path.
    ///
    /// The match is exhaustive over [`Lang`]: adding a language variant
    /// without a retrained artifact for it is a compile error here, not a
    /// runtime surprise.
    ///
    /// # Errors
    /// [`PerceptronParseError::Header`] if the embedded `.bin`'s header or
    /// payload is malformed; [`PerceptronParseError::StaleBin`] if its
    /// recorded source sha256 doesn't match the currently embedded
    /// `json.gz`. Neither should happen for the vendored artifact.
    pub fn for_lang(lang: Lang) -> Result<Self, PerceptronParseError> {
        match lang {
            Lang::En => {
                let (source_sha256, payload) =
                    crate::weights_bin::split_header(WEIGHTS_BIN, ARTIFACT_MAGIC)?;
                if source_sha256 != SOURCE_JSON_GZ_SHA256 {
                    return Err(PerceptronParseError::StaleBin {
                        found: Box::from(source_sha256),
                        expected: SOURCE_JSON_GZ_SHA256,
                    });
                }
                let view = ParserView::from_payload(payload)?;
                Ok(Self {
                    backend: Box::new(ViewBackend { view }),
                })
            }
        }
    }

    /// Loads the parser directly from `json.gz` bytes (the audited
    /// interchange format `weights/NOTICE.md` describes), bypassing the
    /// derived `.bin` artifact entirely. Used by [`pack_perceptron_parser_bin`]
    /// (the `corpus-tool weights-pack` converter) and by this module's own
    /// parity test, which asserts this path and [`Self::new`] agree.
    ///
    /// # Errors
    /// [`PerceptronParseError::Decompress`] if `bytes` isn't valid gzip;
    /// [`PerceptronParseError::Parse`] if the decompressed JSON doesn't
    /// match the expected shape.
    pub fn from_json_gz(bytes: &[u8]) -> Result<Self, PerceptronParseError> {
        let file = parse_json_gz(bytes)?;
        Ok(Self {
            backend: Box::new(OwnedBackend {
                weights: WeightTable::from_file(file),
            }),
        })
    }

    /// Builds a parser directly from an in-memory [`WeightFile`], bypassing
    /// the embedded artifact.
    ///
    /// Exists so `examples/train_parser.rs` can evaluate a freshly trained
    /// model through this exact inference path before it's written to disk
    /// as the vendored artifact; gated behind `train-tooling` so a
    /// production build can't load an untrusted weight file.
    #[cfg(feature = "train-tooling")]
    #[must_use]
    pub fn from_weight_file(file: WeightFile) -> Self {
        Self {
            backend: Box::new(OwnedBackend {
                weights: WeightTable::from_file(file),
            }),
        }
    }
}

impl DepParser for PerceptronParser {
    fn parse(&self, source: &str, tokens: &[TaggedToken]) -> Result<SentenceParse, DepParseError> {
        let words: Vec<Box<str>> = tokens
            .iter()
            .map(|token| Box::from(source[token.token.range.clone()].to_lowercase()))
            .collect();
        // Borrowed, not boxed: `TaggedToken::pos` already owns its string,
        // so a fresh allocation per token here would just copy data the
        // sentence already has.
        let tags: Vec<&str> = tokens.iter().map(|token| token.pos.as_str()).collect();

        let mut config = Configuration::new(tokens.len());
        // Reused across every step of this sentence: `build_features`
        // clears and rewrites it per feature, so after the first step or
        // two grows it to its high-water mark, the rest of the sentence
        // formats features with zero further allocation. Fused with
        // scoring below (`backend.accumulate` per feature) instead of
        // collecting an owned `Vec<Box<str>>` and scoring it afterward —
        // see `build_features`'s and `Backend::accumulate`'s docs.
        let mut buf = String::with_capacity(64);
        while !config.is_terminal() {
            let mut scores = [0f32; NUM_ACTIONS];
            build_features(&words, &tags, &config, &mut buf, |feature| {
                self.backend.accumulate(feature, &mut scores);
            });
            match best_allowed_action(&scores, &config) {
                Some(index) => config.apply(action_at(index)),
                // Unreachable while `!config.is_terminal()` (see
                // `best_allowed_action`'s docs) — defensive, not a panic,
                // since a `DepParser` must never panic on any input.
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
    /// `gold_dep_en.conllu` uses `# sent_id = ...` / `# split = train|test`
    /// / `# text = ...` comment lines, then one `index<TAB>surface<TAB>
    /// penn_pos<TAB>head_index<TAB>relation` row per token (1-based, head
    /// `0` for root), with a blank line ending each sentence.
    ///
    /// # Errors
    /// The first [`GoldParseError`] encountered: malformed row, unparseable
    /// head index, unrecognized relation, or an inconsistent edge list
    /// (self-head, out-of-bounds head).
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
    /// Scores the fixed 42-way action space, following the same
    /// lazy-averaging trick as
    /// [`crate::tag_perceptron::train_support::AveragedPerceptron`]
    /// (accumulate `weight * iterations held`, divide by total iterations
    /// at the end) — keyed by a `u8` action index rather than a string
    /// class name, since this classifier's class set is fixed at compile
    /// time.
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
        /// cast to `f32` at summation — the same precision
        /// [`super::WeightTable::score`] uses at inference, so
        /// training-time prediction sees the rounding behavior the shipped
        /// artifact will.
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
            // A handful of epochs over a few thousand sentences never gets
            // `iterations` near 2^52: no meaningful precision lost here.
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
        /// `config`'s legal-action mask, and if the masked argmax
        /// disagrees with `gold_index`, bumps `gold_index` up and the
        /// wrong prediction down for every feature. Returns the predicted
        /// action index.
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

    /// Runs one sentence through greedy oracle-guided decoding with early
    /// update. See the module docs' "Training" section.
    ///
    /// Stops the instant `model`'s masked prediction disagrees with the
    /// oracle, having already applied that step's update. Returns whether
    /// the model matched the oracle all the way to a terminal
    /// configuration (no update needed this epoch).
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

    /// Re-exposes [`action_at`] for a training tool's diagnostics (e.g. a
    /// per-action update-frequency breakdown) without duplicating the
    /// enumeration.
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
    /// training tool's reproducibility check (train twice, compare
    /// artifacts).
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

    /// [`action_index`]/[`action_at`] round-trip for all 42 actions, and
    /// the enumeration is exactly 42 entries — the contract the shipped
    /// artifact was trained against.
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
    /// the empty-stack initial configuration where most positions are
    /// absent.
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
    /// `.bin` header/view round trip, independent of model accuracy.
    #[test]
    fn embedded_weight_artifact_loads() {
        PerceptronParser::new().expect("embedded weights must load");
    }

    // ---------------------------------------------------------------
    // Derived `.bin` artifact: header, staleness guard, and JSON/binary
    // loading-path parity (see `weights/NOTICE.md`'s "derived artifacts"
    // section and `crate::weights_bin`'s module docs).
    // ---------------------------------------------------------------

    /// [`SOURCE_JSON_GZ_SHA256`] is a hand-copied compile-time constant —
    /// pin it against the embedded `json.gz`'s own actual sha256 so it
    /// can never silently drift from what's really embedded.
    #[test]
    fn embedded_json_gz_sha256_matches_compiled_in_constant() {
        use std::fmt::Write as _;

        use sha2::{Digest as _, Sha256};
        let digest = Sha256::digest(WEIGHTS_JSON_GZ);
        let hex = digest
            .iter()
            .fold(String::with_capacity(64), |mut acc, byte| {
                write!(acc, "{byte:02x}").expect("writing to a String never fails");
                acc
            });
        assert_eq!(hex, SOURCE_JSON_GZ_SHA256);
    }

    /// The embedded `.bin`'s header names the right artifact and the
    /// `json.gz` it was derived from.
    #[test]
    fn embedded_bin_header_matches_embedded_json_gz() {
        let (source_sha256, _payload) =
            crate::weights_bin::split_header(WEIGHTS_BIN, ARTIFACT_MAGIC)
                .expect("embedded .bin header must parse");
        assert_eq!(source_sha256, SOURCE_JSON_GZ_SHA256);
    }

    /// A `.bin` whose header claims a different source sha256 is
    /// distinguishable from one whose header matches — the staleness
    /// check `PerceptronParser::new` runs is a plain string compare
    /// against the compile-time constant, exercised directly here. The
    /// payload itself is still well-formed and loadable — staleness is
    /// purely a header-metadata mismatch, never payload corruption.
    #[test]
    fn a_bin_with_a_different_recorded_sha_is_detected_as_stale() {
        let file = parse_json_gz(WEIGHTS_JSON_GZ).expect("embedded json.gz parses");
        let payload = build_view_payload(file);
        let stale_sha = "0".repeat(64);
        let mut bytes = crate::weights_bin::write_header(ARTIFACT_MAGIC, &stale_sha);
        bytes.extend_from_slice(&payload);
        let (source_sha256, decoded_payload) =
            crate::weights_bin::split_header(&bytes, ARTIFACT_MAGIC).expect("header parses");
        assert_ne!(source_sha256, SOURCE_JSON_GZ_SHA256);
        ParserView::from_payload(decoded_payload).expect("payload still parses");
    }

    /// Full-vocabulary equivalence: every feature the embedded `json.gz`
    /// describes agrees with what the embedded `.bin`'s zero-copy view
    /// returns — not just a battery of sentences (below), the *entire*
    /// vocabulary, so a rare feature no battery sentence happens to
    /// trigger can't hide a translation bug. Also checks that keys
    /// genuinely absent from the source file are reported absent by the
    /// view, not aliased onto some other entry.
    ///
    /// Exact float equality is intentional: the view's `f32` weights are
    /// read verbatim (via `from_le_bytes`) from the same bits
    /// `serde_json` parsed from the source `json.gz`, so any bit
    /// difference is a real translation bug, not float-precision noise.
    #[test]
    #[allow(clippy::float_cmp)]
    fn view_matches_the_json_weight_file_across_its_entire_vocabulary() {
        let file = parse_json_gz(WEIGHTS_JSON_GZ).expect("embedded json.gz parses");
        let (source_sha256, payload) =
            crate::weights_bin::split_header(WEIGHTS_BIN, ARTIFACT_MAGIC).expect("header parses");
        assert_eq!(source_sha256, SOURCE_JSON_GZ_SHA256);
        let view = ParserView::from_payload(payload).expect("payload parses");

        for (feature, sparse) in &file.features {
            let mut expected = [0f32; NUM_ACTIONS];
            for &(action_index, weight) in sparse {
                expected[usize::from(action_index)] = weight;
            }
            let scores = view.score(std::slice::from_ref(feature));
            assert_eq!(scores, expected, "feature {feature:?}");
        }

        let absent_features = ["zzz-never-a-feature-zzz", "s0.w=zzzznonexistentzzz"];
        for absent in absent_features {
            assert!(
                !file.features.contains_key(absent),
                "test bug: {absent:?} is actually a real feature"
            );
            let owned = Box::from(absent);
            let scores = view.score(std::slice::from_ref(&owned));
            assert_eq!(scores, [0f32; NUM_ACTIONS], "absent feature {absent:?}");
        }
    }

    /// Behavioral parity, one level up: a parser built via each loading
    /// path derives identical dependency edges over a fixed battery of
    /// sentences (tagged by the embedded tagger, itself irrelevant to
    /// which parser loading path is under test here).
    #[test]
    fn json_and_bin_paths_parse_a_sentence_battery_identically() {
        use crate::Tagger as _;

        let tagger = crate::PerceptronTagger::new().expect("embedded tagger loads");
        let from_bin = PerceptronParser::new().expect("embedded .bin loads");
        let from_json =
            PerceptronParser::from_json_gz(WEIGHTS_JSON_GZ).expect("embedded json.gz loads");
        let battery = [
            "The quick brown foxes are jumping over lazy dogs.",
            "Run the scanner from the project root.",
            "Results stream in as they are produced.",
            "Cannot connect to the database right now.",
            "State-of-the-art solutions rarely stay that way for long.",
            "Wait, really? That seems unlikely.",
            "She'll fix it before the release, won't she?",
            "Version 3.14 shipped on 2024-01-01.",
            "Neither the client nor the server retried the request.",
            "This endpoint sends an Alt-Svc header field to clients.",
            "Please don't remove the trailing comma.",
            "We're deprecating the legacy API next quarter.",
            "A cross-domain resonance pattern emerged from the data.",
            "The team's velocity improved after the refactor.",
            "Install the package, then restart the daemon.",
            "It is unclear whether the fix addressed the root cause.",
            "Only administrators may modify these settings.",
            "The function returns early if the input is empty.",
            "He might have missed the deadline entirely.",
            "The report was filed late, but nobody complained.",
        ];
        for sentence in battery {
            let tagged_tokens = tagger.tag(sentence, 0);
            let bin_parse = from_bin
                .parse(sentence, &tagged_tokens)
                .expect("embedded .bin parses");
            let json_parse = from_json
                .parse(sentence, &tagged_tokens)
                .expect("embedded json.gz parses");
            assert_eq!(bin_parse, json_parse, "parse mismatch for {sentence:?}");
        }
    }

    /// [`pack_perceptron_parser_bin`] is deterministic: converting the
    /// same `json.gz` bytes twice (with the same recorded source sha256)
    /// produces bit-identical `.bin` bytes.
    #[test]
    fn pack_perceptron_parser_bin_is_deterministic() {
        let sha = "a".repeat(64);
        let first = pack_perceptron_parser_bin(WEIGHTS_JSON_GZ, &sha).expect("packs");
        let second = pack_perceptron_parser_bin(WEIGHTS_JSON_GZ, &sha).expect("packs");
        assert_eq!(first, second);
    }
}
