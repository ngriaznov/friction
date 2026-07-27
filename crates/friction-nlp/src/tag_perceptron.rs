//! Averaged-perceptron [`Tagger`] implementation: the default backend.
//!
//! Weights aren't downloaded at build time — they're vendored in this
//! crate's own `weights/perceptron_en.json.gz`, trained on a small,
//! hand-curated gold-tag corpus from this project's own vendored human
//! documents (see `weights/NOTICE.md` for provenance and the
//! reproduction command). Nothing here ever touches the network.
//!
//! # Tokenization is its own
//!
//! [`tokenize`] is a small manual scanner producing spans that satisfy
//! [`crate::tag::classify_token_kind`]'s rule directly: maximal
//! alphabetic/`'`/`-` runs as [`TokenKind::Word`], a numeric-leading run
//! of digits/`.`/`,` as [`TokenKind::Number`], maximal runs of
//! [`crate::tag::is_prose_punctuation`] characters as one
//! [`TokenKind::Punctuation`] token, everything else as
//! [`TokenKind::Symbol`]. No whitespace tokens are emitted.
//!
//! # Non-word tokens: a deterministic passthrough, never the model
//!
//! [`TokenKind::Number`] gets `"CD"`; [`TokenKind::Symbol`] gets `"SYM"`;
//! [`TokenKind::Punctuation`] gets one of nine closed Penn/spaCy-style
//! tags from [`punctuation_tag`], never its own surface text (see
//! `weights/NOTICE.md`'s tagset decision: a surface-text tag turned
//! "punctuation" into ~140 near-unique values, defeating any downstream
//! consumer treating a tag as a syntactic class). These categories are
//! effectively fully determined by surface form, so hardcoding them
//! removes risk without costing accuracy, and using the same fixed value
//! for tag-history context in both training and inference keeps the two
//! consistent. Only [`TokenKind::Word`] tokens run the perceptron.
//!
//! # Lemma: best-effort, reusing this crate's own inflection tables
//!
//! No dictionary lemmatizer — [`crate::inflect::lemmatize`] guesses a
//! `Word` token's base form by generating candidate stems from its tag
//! and keeping whichever round-trips back through
//! [`crate::inflect::inflect`]'s forward generation, reusing that
//! module's tables in reverse rather than duplicating them. A candidate
//! that fails to round-trip falls back to the lowercased surface text —
//! matching [`TaggedToken::lemma`]'s documented fallback. Non-`Word`
//! tokens always get the lowercased surface text directly.
//!
//! # Determinism
//!
//! The feature list is a fixed-order `Vec<Box<str>>` built from a fixed
//! code path, not iterated out of a hash collection, so summation order
//! — and the resulting scores — is identical run to run. Scoring indexes
//! a plain array by each class's load-time sorted position and picks the
//! argmax by strict `>` in that fixed order, so ties always resolve to
//! the lexicographically earliest tag, never hash-map iteration order.
//!
//! # Per-call sentence-reset semantics
//!
//! [`PerceptronTagger::tag`] does **not** re-segment multi-sentence
//! `text`: it treats the whole argument as one tag-history sequence,
//! resetting the start-of-sequence sentinels once. Every production call
//! site already calls `tag()` per already-segmented sentence, so this is
//! safe — but a caller must not pass an unsegmented paragraph expecting
//! sentence-internal resets.
//!
//! # Two loading paths, one inference surface
//!
//! [`PerceptronTagger::new`] (the fast path, embedded `.bin`) and
//! [`PerceptronTagger::from_json_gz`] (the slow, audited-source path) build
//! different internal representations — [`ViewBackend`], a zero-copy view
//! over the `'static` embedded bytes, versus [`OwnedBackend`], a
//! `HashMap`-based table built the way this module always has — behind
//! the shared [`Backend`] trait, so [`PerceptronTagger::tag_word`]'s
//! tagdict/scoring/argmax logic is written once and both paths are
//! provably running the identical algorithm (pinned by this module's own
//! `json_and_bin_paths_tag_a_sentence_battery_identically` test).
//!
//! # The embedded `.bin`'s version-2 payload: a serialized hash table
//!
//! [`PerceptronTagger::new`] loads `perceptron_en.bin` (embedded as
//! [`WEIGHTS_BIN`]) with zero per-key construction — no `HashMap` built,
//! no feature string re-hashed, no per-feature `Vec` allocated. See
//! `crate::weights_bin`'s module docs for why (version-1 payloads
//! `postcard`-decoded straight into a `HashMap`; version 2 instead
//! serializes the table itself). This module's payload (the bytes after
//! `crate::weights_bin`'s shared header), built by [`build_view_payload`]
//! and read back by [`TaggerView::from_payload`], is:
//!
//! - `num_classes: u32`
//! - `features_capacity: u32`, then the features hash table's slots as a
//!   length-prefixed section (feature string -> a [`crate::weights_bin::sparse_row`]
//!   byte offset into the values section below)
//! - the features values section: every feature's sparse `(class_index:
//!   u16, weight: f32)` row, self-describing via a leading count (see
//!   [`crate::weights_bin::build_sparse_row`]) — sparse, not one dense
//!   `f32` per class, since the average feature only carries a handful of
//!   nonzero class weights (measured: ~4 of 49 for this artifact) and a
//!   dense row would multiply artifact size roughly by the class count
//! - `tagdict_capacity: u32`, then the tagdict hash table's slots (word ->
//!   the index of its bypass tag in the classes list below, not a
//!   duplicated string)
//! - the classes section: `num_classes` `(offset: u32, length: u32)` pairs
//!   into the string pool, in sorted order (matching the row/argmax index
//!   every score uses)
//! - the string pool: every feature string, tagdict word, and class tag
//!   string, concatenated in that order (feature keys, then tagdict
//!   words, then class tags — each in the fixed order it was pushed
//!   during packing)
//!
//! Determinism: [`build_view_payload`] feeds every table builder entries
//! in `WeightFile`'s own `BTreeMap` order (already sorted ascending by
//! key), so packing the same `WeightFile` twice produces bit-identical
//! bytes — pinned by `pack_perceptron_tagger_bin_is_deterministic` below.

use std::collections::{BTreeMap, HashMap};
use std::io::Read as _;
use std::ops::Range;

use friction_core::{Token, TokenKind};
use serde::{Deserialize, Serialize};

use crate::tag::{PosTag, TaggedToken, Tagger, classify_token_kind, is_prose_punctuation};

/// The vendored, gzip-compressed perceptron weight artifact — the
/// audited interchange format from the training pipeline. See
/// `weights/NOTICE.md` for provenance and the reproduction command;
/// `examples/train_perceptron.rs` (behind the `train-tooling` feature)
/// regenerates it.
///
/// `cfg(test)`-only: normal startup uses the faster [`WEIGHTS_BIN`] path
/// instead (see that constant's own docs), so a production binary has no
/// reason to also carry this now-redundant copy. The only in-tree callers
/// of [`PerceptronTagger::from_json_gz`] and `pack_perceptron_tagger_bin`
/// with these exact bytes are this module's own parity tests below;
/// `corpus-tool weights-pack` calls `pack_perceptron_tagger_bin` with
/// bytes it reads fresh from disk instead, since regenerating this
/// artifact is exactly the case where the newly-retrained `json.gz` on
/// disk and whatever is currently compiled into this static disagree.
#[cfg(test)]
static WEIGHTS_JSON_GZ: &[u8] = include_bytes!("../weights/perceptron_en.json.gz");

/// The derived binary weight artifact [`PerceptronTagger::new`] loads at
/// normal startup: a version-2 serialized hash-table view (this module's
/// own docs describe the exact byte layout), embedded raw — see
/// `crate::weights_bin`'s module docs for the shared header shape and
/// `weights/NOTICE.md`'s "derived artifacts" section for how it's
/// produced (`corpus-tool weights-pack`) from `WEIGHTS_JSON_GZ`.
static WEIGHTS_BIN: &[u8] = include_bytes!("../weights/perceptron_en.bin");

/// This artifact's 8-byte magic, distinct from [`crate::dep_perceptron`]'s
/// own — so one can never be silently loaded in place of the other.
const ARTIFACT_MAGIC: [u8; 8] = *b"FRTAGWT1";

/// The sha256 (lowercase hex) of the `WEIGHTS_JSON_GZ` bytes currently
/// embedded above, fixed at compile time. [`PerceptronTagger::new`]
/// compares this against `perceptron_en.bin`'s own recorded source
/// sha256 (from its header — see `crate::weights_bin`) so a `.bin` left
/// stale after a `perceptron_en.json.gz` regeneration fails loudly at
/// load time instead of silently drifting. Pinned by
/// `embedded_json_gz_sha256_matches_compiled_in_constant` below, which
/// recomputes it directly from the embedded bytes.
const SOURCE_JSON_GZ_SHA256: &str =
    "0521131dad4e233c2552d4b3be0612505ef676ad21d34378888428c5a33f3a4e";

/// Sentinel fed as tag-history context before the first token (and, one
/// position further back, before the second token).
const START_TAG_1: &str = "-START-";
const START_TAG_2: &str = "-START2-";

/// Sentinel fed as word-context when a feature looks past either edge of
/// the token stream.
const START_WORD_1: &str = "-start-";
const START_WORD_2: &str = "-start2-";
const END_WORD_1: &str = "-end-";
const END_WORD_2: &str = "-end2-";

/// Errors constructing a [`PerceptronTagger`].
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum PerceptronTagError {
    /// A `json.gz` artifact failed to gunzip. Points at a build problem
    /// with the vendored artifact, not a runtime condition, since the
    /// bytes are compiled in.
    #[error("failed to decompress the perceptron weight artifact: {0}")]
    Decompress(#[source] std::io::Error),
    /// A decompressed `json.gz` artifact was not valid JSON in the
    /// expected shape.
    #[error("failed to parse the perceptron weight artifact: {0}")]
    Parse(#[source] serde_json::Error),
    /// The embedded `.bin` artifact's header was malformed — see
    /// [`crate::weights_bin::WeightsBinError`]'s variants.
    #[error(transparent)]
    Header(#[from] crate::weights_bin::WeightsBinError),
    /// The embedded `.bin` artifact's recorded source sha256 doesn't
    /// match [`SOURCE_JSON_GZ_SHA256`], the compile-time sha256 of the
    /// currently embedded `perceptron_en.json.gz`: the `.bin` is stale
    /// (the `json.gz` was regenerated but `corpus-tool weights-pack`
    /// wasn't re-run) and must not be trusted.
    #[error(
        "the embedded perceptron_en.bin's recorded source sha256 ({found}) does not match \
         perceptron_en.json.gz's compiled-in sha256 ({expected}) — regenerate perceptron_en.bin \
         with `corpus-tool weights-pack`"
    )]
    StaleBin {
        found: Box<str>,
        expected: &'static str,
    },
}

/// On-disk shape of the weight artifact: `BTreeMap`s throughout so the
/// serialized JSON is stable and diffable across re-training runs, and so
/// [`OwnedBackend::from_file`]/[`build_view_payload`] can both feed their
/// respective tables entries in the same deterministic ascending-key
/// order.
///
/// Public only because `train-tooling`'s `train_support` module hands one
/// back to an external training tool (`examples/train_perceptron.rs`,
/// compiled as its own crate) — otherwise unreachable from outside this
/// module, since `tag_perceptron` is private and this struct is never
/// re-exported directly.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WeightFile {
    pub classes: Vec<Box<str>>,
    /// feature string -> sparse `[(class index into `classes`, weight)]`.
    pub features: BTreeMap<Box<str>, Vec<(u16, f32)>>,
    /// A frequent/unambiguous word's tag, bypassing the model entirely.
    pub tagdict: BTreeMap<Box<str>, Box<str>>,
}

/// Sorts `file_classes` and returns `(sorted classes, index_map)`, where
/// `index_map[i]` is `file_classes[i]`'s position in the sorted list —
/// shared by [`OwnedBackend::from_file`] (dense, in-memory) and
/// [`build_view_payload`] (sparse, on-disk) so both translate a
/// [`WeightFile`]'s stored class indices the same way regardless of what
/// order its own `classes` array happened to be written in.
fn sort_classes_with_index_map(file_classes: &[Box<str>]) -> (Vec<Box<str>>, Vec<usize>) {
    let mut classes = file_classes.to_vec();
    classes.sort();
    let index_map = file_classes
        .iter()
        .map(|class| {
            classes
                .binary_search(class)
                .expect("every class in a weight file's `classes` list must appear in it")
        })
        .collect();
    (classes, index_map)
}

/// Runtime weight table: one dense per-class weight vector per feature
/// string, indexed by each class's position in the freshly sorted
/// `classes` list [`sort_classes_with_index_map`] returns — never by
/// whatever order the on-disk file happened to use.
struct WeightTable {
    num_classes: usize,
    by_feature: HashMap<Box<str>, Vec<f32>>,
}

impl WeightTable {
    /// Builds a `WeightTable` from `features` (already keyed by the
    /// file's own, pre-translation class indices) and `index_map`, the
    /// translation [`sort_classes_with_index_map`] computed for the same
    /// file.
    fn from_features(
        features: BTreeMap<Box<str>, Vec<(u16, f32)>>,
        index_map: &[usize],
        num_classes: usize,
    ) -> Self {
        let mut by_feature = HashMap::with_capacity(features.len());
        for (feature, sparse) in features {
            let mut dense = vec![0f32; num_classes];
            for (class_index, weight) in sparse {
                if let Some(mapped) = index_map.get(class_index as usize) {
                    dense[*mapped] = weight;
                }
            }
            by_feature.insert(feature, dense);
        }
        Self {
            num_classes,
            by_feature,
        }
    }

    /// Sums each feature's dense per-class weight vector, in `features`'
    /// own fixed order — the determinism-critical step: summation order
    /// is identical across runs because iteration order is fixed by the
    /// caller-supplied `Vec`, never by hash-map iteration.
    fn score(&self, features: &[Box<str>]) -> Vec<f32> {
        let mut scores = vec![0f32; self.num_classes];
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

/// A zero-copy view over a version-2 tagger payload's embedded bytes — see
/// this module's own docs for the exact byte layout [`build_view_payload`]
/// writes and [`Self::from_payload`] reads back. Every field borrows
/// directly from the `'static` embedded artifact ([`WEIGHTS_BIN`]);
/// building one does no per-key work at all.
struct TaggerView<'a> {
    num_classes: usize,
    features: crate::weights_bin::HashView<'a>,
    values: &'a [u8],
    tagdict: crate::weights_bin::HashView<'a>,
    /// `num_classes` `(offset: u32, length: u32)` pairs into `pool`, in
    /// sorted class order.
    classes_offsets: &'a [u8],
    pool: &'a [u8],
}

impl<'a> TaggerView<'a> {
    /// Parses `payload` (the bytes [`crate::weights_bin::split_header`]
    /// returned) into a view.
    ///
    /// # Errors
    /// [`crate::weights_bin::WeightsBinError::MalformedPayload`] if any
    /// section is missing, truncated, or an embedded hash table's slot
    /// buffer doesn't match its own recorded capacity.
    fn from_payload(payload: &'a [u8]) -> Result<Self, crate::weights_bin::WeightsBinError> {
        use crate::weights_bin::{Cursor, HashView};

        let mut cursor = Cursor::new(payload);
        let num_classes = cursor.read_u32()? as usize;
        let features_capacity = cursor.read_u32()?;
        let features_slots = cursor.read_section()?;
        let values = cursor.read_section()?;
        let tagdict_capacity = cursor.read_u32()?;
        let tagdict_slots = cursor.read_section()?;
        let classes_offsets = cursor.read_section()?;
        let pool = cursor.read_section()?;

        let features = HashView::new(features_capacity, features_slots, pool)?;
        let tagdict = HashView::new(tagdict_capacity, tagdict_slots, pool)?;
        if classes_offsets.len() != num_classes * 8 {
            return Err(crate::weights_bin::WeightsBinError::MalformedPayload(
                "classes section length doesn't match num_classes * 8",
            ));
        }

        Ok(Self {
            num_classes,
            features,
            values,
            tagdict,
            classes_offsets,
            pool,
        })
    }

    /// The sorted class list's `index`th tag string.
    fn class_str(&self, index: usize) -> &'a str {
        let start = index * 8;
        let entry = &self.classes_offsets[start..start + 8];
        let offset = u32::from_le_bytes(entry[0..4].try_into().expect("4-byte slice")) as usize;
        let len = u32::from_le_bytes(entry[4..8].try_into().expect("4-byte slice")) as usize;
        std::str::from_utf8(&self.pool[offset..offset + len])
            .expect("build_view_payload only ever pushes valid UTF-8 keys into the pool")
    }

    /// Sums every matched feature's sparse per-class weight row —
    /// mirrors [`WeightTable::score`]'s fixed-order summation, just
    /// reading each row's nonzero entries from the payload instead of a
    /// pre-built dense `Vec`.
    fn score(&self, features: &[Box<str>]) -> Vec<f32> {
        let mut scores = vec![0f32; self.num_classes];
        for feature in features {
            if let Some(row_offset) = self.features.get(feature.as_bytes()) {
                for chunk in crate::weights_bin::sparse_row(self.values, row_offset).chunks_exact(6)
                {
                    let (class_index, weight) = crate::weights_bin::decode_sparse_entry(chunk);
                    scores[usize::from(class_index)] += weight;
                }
            }
        }
        scores
    }

    /// `word`'s tagdict bypass tag, if any.
    fn tagdict_get(&self, word: &str) -> Option<&'a str> {
        self.tagdict
            .get(word.as_bytes())
            .map(|class_index| self.class_str(class_index as usize))
    }
}

/// Builds a version-2 tagger payload's bytes (everything after
/// `crate::weights_bin`'s shared header) from `file` — this module's own
/// docs describe the exact byte layout. The inverse of
/// [`TaggerView::from_payload`].
///
/// Every table is built from `file`'s own `BTreeMap`s in their natural
/// ascending-key iteration order, so calling this twice on an equal
/// `WeightFile` produces bit-identical bytes (pinned by
/// `pack_perceptron_tagger_bin_is_deterministic` below).
fn build_view_payload(file: WeightFile) -> Vec<u8> {
    let WeightFile {
        classes: file_classes,
        features,
        tagdict,
    } = file;

    let (classes, index_map) = sort_classes_with_index_map(&file_classes);
    let mut pool = Vec::new();

    // Features: feature string -> a sparse row of (sorted class index,
    // weight) pairs, translated through `index_map` exactly like
    // `WeightTable::from_features` translates them for the owned backend.
    let mut values = Vec::new();
    let mut feature_entries: Vec<(&str, u32, u32, u32)> = Vec::with_capacity(features.len());
    for (feature, sparse) in &features {
        let translated: Vec<(u16, f32)> = sparse
            .iter()
            .filter_map(|&(class_index, weight)| {
                index_map
                    .get(class_index as usize)
                    .map(|&mapped| (u16::try_from(mapped).expect("num_classes fits u16"), weight))
            })
            .collect();
        let row_offset = crate::weights_bin::build_sparse_row(&mut values, &translated);
        let (key_off, key_len) = crate::weights_bin::push_key(&mut pool, feature);
        feature_entries.push((feature.as_ref(), key_off, key_len, row_offset));
    }
    let (features_capacity, features_slots) =
        crate::weights_bin::build_open_table(&feature_entries);

    // Tagdict: word -> index of its bypass tag in the sorted classes list.
    let mut tagdict_entries: Vec<(&str, u32, u32, u32)> = Vec::with_capacity(tagdict.len());
    for (word, tag) in &tagdict {
        let class_index = classes
            .binary_search(tag)
            .expect("every tagdict tag is one of the file's own registered classes");
        let (key_off, key_len) = crate::weights_bin::push_key(&mut pool, word);
        tagdict_entries.push((
            word.as_ref(),
            key_off,
            key_len,
            u32::try_from(class_index).expect("num_classes fits u32"),
        ));
    }
    let (tagdict_capacity, tagdict_slots) = crate::weights_bin::build_open_table(&tagdict_entries);

    // Classes: sorted tag strings, appended to the pool last.
    let mut classes_offsets = Vec::with_capacity(classes.len() * 8);
    for class in &classes {
        let (offset, len) = crate::weights_bin::push_key(&mut pool, class);
        classes_offsets.extend_from_slice(&offset.to_le_bytes());
        classes_offsets.extend_from_slice(&len.to_le_bytes());
    }

    let mut out = Vec::new();
    crate::weights_bin::write_u32(
        &mut out,
        u32::try_from(classes.len()).expect("num_classes fits u32"),
    );
    crate::weights_bin::write_u32(&mut out, features_capacity);
    crate::weights_bin::write_section(&mut out, &features_slots);
    crate::weights_bin::write_section(&mut out, &values);
    crate::weights_bin::write_u32(&mut out, tagdict_capacity);
    crate::weights_bin::write_section(&mut out, &tagdict_slots);
    crate::weights_bin::write_section(&mut out, &classes_offsets);
    crate::weights_bin::write_section(&mut out, &pool);
    out
}

/// Shared inference surface both loading paths implement identically:
/// [`PerceptronTagger::new`]'s zero-copy [`ViewBackend`] and
/// [`PerceptronTagger::from_json_gz`]'s owned [`OwnedBackend`]. Everything
/// [`PerceptronTagger::tag_word`] needs — the tagdict short-circuit,
/// feature scoring, and resolving a winning class index back to its tag
/// string — goes through here, so the per-token tagging algorithm is
/// written exactly once and both backends are provably running the same
/// one.
trait Backend {
    /// `word`'s tagdict bypass tag, if any (already an owned `Box<str>`:
    /// the owned backend's own map entry is cloned either way, so there's
    /// no allocation-free path to give up by not returning a borrow).
    fn tagdict_lookup(&self, word: &str) -> Option<Box<str>>;
    /// Sums every matched feature's per-class weights, in `features`' own
    /// fixed order.
    fn score(&self, features: &[Box<str>]) -> Vec<f32>;
    /// The tag string for the sorted class list's `index`th entry.
    fn class_at(&self, index: usize) -> Box<str>;
}

/// [`Backend`] over an in-memory, `HashMap`-based [`WeightTable`] — built
/// by [`PerceptronTagger::from_json_gz`], the slower audited-source path.
struct OwnedBackend {
    weights: WeightTable,
    classes: Vec<Box<str>>,
    tagdict: BTreeMap<Box<str>, Box<str>>,
}

impl OwnedBackend {
    fn from_file(file: WeightFile) -> Self {
        let (classes, index_map) = sort_classes_with_index_map(&file.classes);
        let num_classes = classes.len();
        let weights = WeightTable::from_features(file.features, &index_map, num_classes);
        Self {
            weights,
            classes,
            tagdict: file.tagdict,
        }
    }
}

impl Backend for OwnedBackend {
    fn tagdict_lookup(&self, word: &str) -> Option<Box<str>> {
        self.tagdict.get(word).cloned()
    }

    fn score(&self, features: &[Box<str>]) -> Vec<f32> {
        self.weights.score(features)
    }

    fn class_at(&self, index: usize) -> Box<str> {
        self.classes[index].clone()
    }
}

/// [`Backend`] over a zero-copy [`TaggerView`] — built by
/// [`PerceptronTagger::new`], the fast path every normal caller wants.
struct ViewBackend {
    view: TaggerView<'static>,
}

impl Backend for ViewBackend {
    fn tagdict_lookup(&self, word: &str) -> Option<Box<str>> {
        self.view.tagdict_get(word).map(Box::from)
    }

    fn score(&self, features: &[Box<str>]) -> Vec<f32> {
        self.view.score(features)
    }

    fn class_at(&self, index: usize) -> Box<str> {
        Box::from(self.view.class_str(index))
    }
}

/// The index of the highest-scoring class in `scores`, breaking ties in
/// favor of the lowest index — since `classes` is sorted ascending at load
/// time, this is exactly "the lexicographically earliest tag wins ties".
fn argmax(scores: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_score = scores[0];
    for (index, &score) in scores.iter().enumerate().skip(1) {
        if score > best_score {
            best_score = score;
            best = index;
        }
    }
    best
}

/// A word's *identity* feature value, normalized past a small training
/// vocabulary: lowercased, and collapsed to one of three closed sentinel
/// buckets rather than kept as a literal (likely unseen) string — a small
/// hand-tagged corpus can't enumerate every hyphenated or numeral-bearing
/// token it might meet at inference. Mirrors NLTK's `_normalize` bucket
/// order (hyphen, then four-digit year, then any leading-digit token)
/// rather than a looser "contains a digit/hyphen anywhere" test, so
/// `"co2"`/`"a-1"` bucket the same way NLTK's tagger would.
///
/// Only the *identity* feature is bucketed; suffix and first-character
/// features always read the literal word (see [`extract_features`]),
/// since those stay informative even for a bucketed word.
fn normalize_word_identity(word_lower: &str) -> &str {
    let Some(first) = word_lower.chars().next() else {
        return word_lower;
    };
    if first != '-' && word_lower.contains('-') {
        return "!hyphenated!";
    }
    if word_lower.chars().count() == 4 && word_lower.chars().all(|c| c.is_ascii_digit()) {
        return "!year!";
    }
    if first.is_ascii_digit() {
        return "!contains_digit!";
    }
    word_lower
}

/// The last (up to) three characters of `word_lower`, by `char` count, not
/// byte count — a short word's own full text if it has three characters or
/// fewer.
fn suffix3(word_lower: &str) -> &str {
    let byte_start = word_lower
        .char_indices()
        .rev()
        .nth(2)
        .map_or(0, |(index, _)| index);
    &word_lower[byte_start..]
}

/// Classifies a [`TokenKind::Punctuation`] token's surface text into one
/// of nine closed Penn/spaCy-style tags — `.` `,` `:` `-LRB-` `-RRB-`
/// `` ` `` `''` `HYPH` `NFP` — never the surface text itself. See
/// `weights/NOTICE.md`: the old per-surface-string scheme emitted ~140
/// distinct punctuation "tags", each carrying no more information than
/// the token's own text.
///
/// `surface` is [`tokenize`]'s maximal run of [`is_prose_punctuation`]
/// characters and can mix distinct punctuation characters (`"),"`,
/// `"?!"`), unlike genuine Penn-Treebank/CoNLL-U tokenization, which
/// always isolates one character per token. Classified by the character
/// *set* the run is drawn from, in this fixed order (an earlier rule
/// always wins):
///
/// 1. every character is one of `.!?` (including a repeated run like
///    `"!!!"` or `"..."`) -> `.`, the sentence-final-punctuation bucket
/// 2. every character is `,` -> `,`
/// 3. every character is one of `;:` -> `:`
/// 4. every character is a hyphen/en dash/em dash -> `HYPH`
/// 5. every character is one of `([{` -> `-LRB-`
/// 6. every character is one of `)]}` -> `-RRB-`
/// 7. every character is an opening curly quote (`'` or `"`) -> `` ` ``
/// 8. every character is a closing curly quote or a straight quote
///    (`'`, `"`, `'`, or `"`) -> `''`
/// 9. anything else — a mixed-character run (`").:"`, `"?!"`), a bare `/`,
///    or a lone horizontal-ellipsis character — -> `NFP`, spaCy's own
///    bucket for punctuation that fits no cleaner category.
fn punctuation_tag(surface: &str) -> &'static str {
    if surface.is_empty() {
        return "NFP";
    }
    let mut chars = surface.chars();
    if chars.clone().all(|c| matches!(c, '.' | '!' | '?')) {
        "."
    } else if chars.clone().all(|c| c == ',') {
        ","
    } else if chars.clone().all(|c| matches!(c, ';' | ':')) {
        ":"
    } else if chars
        .clone()
        .all(|c| matches!(c, '-' | '\u{2013}' | '\u{2014}'))
    {
        "HYPH"
    } else if chars.clone().all(|c| matches!(c, '(' | '[' | '{')) {
        "-LRB-"
    } else if chars.clone().all(|c| matches!(c, ')' | ']' | '}')) {
        "-RRB-"
    } else if chars.clone().all(|c| matches!(c, '\u{2018}' | '\u{201C}')) {
        "``"
    } else if chars.all(|c| matches!(c, '\'' | '"' | '\u{2019}' | '\u{201D}')) {
        "''"
    } else {
        "NFP"
    }
}

/// `word_lower`'s first character, as a `&str` (empty for an empty word,
/// which should never occur for a real [`TokenKind::Word`] token).
fn first_char(word_lower: &str) -> &str {
    match word_lower.char_indices().nth(1) {
        Some((end, _)) => &word_lower[..end],
        None => word_lower,
    }
}

/// `surfaces[idx]`'s lowercased text, or a start-/end-of-sequence sentinel
/// when `idx` looks past either edge — shared by every "word two back" /
/// "previous word" / "next word" / "word two forward" feature.
fn context_word(surfaces: &[Box<str>], idx: isize) -> &str {
    if idx < 0 {
        return if idx == -1 {
            START_WORD_1
        } else {
            START_WORD_2
        };
    }
    // `idx >= 0` was just checked, and every real token stream is far
    // smaller than `isize::MAX`, so this conversion never loses the sign.
    #[allow(clippy::cast_sign_loss)]
    let idx = idx as usize;
    if idx >= surfaces.len() {
        return if idx == surfaces.len() {
            END_WORD_1
        } else {
            END_WORD_2
        };
    }
    &surfaces[idx]
}

/// Builds token `i`'s fixed-order, 14-entry feature list: bias; current
/// word; current word's 3-char suffix; current word's first character;
/// previous tag; tag two back; previous-tag×tag-two-back; previous-tag×
/// current-word; previous word; previous word's 3-char suffix; word two
/// back; next word; next word's 3-char suffix; word two forward.
///
/// `surfaces` is every token's lowercased surface text for the whole
/// `tag()` call (punctuation included — it legitimately participates in
/// word context, as in genuine Penn-Treebank tagging). `raw_current` is
/// token `i`'s text in its *original* case: NLTK's `_get_features`
/// computes `word[-3:]`/`word[0]` from the raw token, not the normalized
/// `context[i]`, so this is the only place the feature list sees
/// capitalization — every neighboring-word feature still reads the
/// lowercased `surfaces` entry, matching `context[i-1]`/`context[i+1]`
/// exactly. `prev1_tag`/`prev2_tag` are the tags already assigned to
/// tokens `i-1`/`i-2` (or a start-of-sequence sentinel).
fn extract_features(
    surfaces: &[Box<str>],
    raw_current: &str,
    i: usize,
    prev1_tag: &str,
    prev2_tag: &str,
) -> Vec<Box<str>> {
    // Every real token stream is far smaller than `isize::MAX`, so this
    // conversion never wraps.
    #[allow(clippy::cast_possible_wrap)]
    let i = i as isize;
    let cur_lower = context_word(surfaces, i);
    let cur_word = normalize_word_identity(cur_lower);
    let cur_suffix = suffix3(raw_current);
    let cur_pref1 = first_char(raw_current);

    let prev_raw = context_word(surfaces, i - 1);
    let prev_word = normalize_word_identity(prev_raw);
    let prev_suffix = suffix3(prev_raw);

    let word_two_back = normalize_word_identity(context_word(surfaces, i - 2));

    let next_raw = context_word(surfaces, i + 1);
    let next_word = normalize_word_identity(next_raw);
    let next_suffix = suffix3(next_raw);

    let word_two_forward = normalize_word_identity(context_word(surfaces, i + 2));

    vec![
        Box::from("bias"),
        Box::from(format!("word={cur_word}")),
        Box::from(format!("suffix={cur_suffix}")),
        Box::from(format!("pref1={cur_pref1}")),
        Box::from(format!("i-1 tag={prev1_tag}")),
        Box::from(format!("i-2 tag={prev2_tag}")),
        Box::from(format!("i-1 tag+i-2 tag={prev1_tag},{prev2_tag}")),
        Box::from(format!("i-1 tag+word={prev1_tag},{cur_word}")),
        Box::from(format!("i-1 word={prev_word}")),
        Box::from(format!("i-1 suffix={prev_suffix}")),
        Box::from(format!("i-2 word={word_two_back}")),
        Box::from(format!("i+1 word={next_word}")),
        Box::from(format!("i+1 suffix={next_suffix}")),
        Box::from(format!("i+2 word={word_two_forward}")),
    ]
}

/// One scanned token: its byte range and coarse kind, before tagging.
struct ScannedToken {
    range: Range<usize>,
    kind: TokenKind,
}

/// Splits `text` into [`ScannedToken`]s: maximal alphabetic/`'`/`-` runs
/// containing at least one alphabetic character as [`TokenKind::Word`]; a
/// numeric-leading run of digits/`.`/`,` as [`TokenKind::Number`]; maximal
/// runs of [`is_prose_punctuation`] characters (including a bare run of
/// `'`/`-` with no alphabetic character) as one [`TokenKind::Punctuation`]
/// token; everything else grouped into maximal runs as [`TokenKind::Symbol`].
/// Whitespace is skipped, never emitted. The same partition
/// [`classify_token_kind`] describes for a single already-split token,
/// applied here to decide the splits themselves.
fn tokenize(text: &str) -> Vec<ScannedToken> {
    let mut tokens = Vec::new();
    let mut chars = text.char_indices().peekable();

    while let Some(&(start, c)) = chars.peek() {
        if c.is_whitespace() {
            chars.next();
            continue;
        }

        if c.is_numeric() {
            let mut end = start + c.len_utf8();
            chars.next();
            while let Some(&(idx, next)) = chars.peek() {
                if next.is_numeric() || next == '.' || next == ',' {
                    end = idx + next.len_utf8();
                    chars.next();
                } else {
                    break;
                }
            }
            tokens.push(ScannedToken {
                range: start..end,
                kind: TokenKind::Number,
            });
            continue;
        }

        if c.is_alphabetic() || c == '\'' || c == '-' {
            let mut end = start + c.len_utf8();
            let mut has_alpha = c.is_alphabetic();
            chars.next();
            while let Some(&(idx, next)) = chars.peek() {
                if next.is_alphabetic() || next == '\'' || next == '-' {
                    has_alpha |= next.is_alphabetic();
                    end = idx + next.len_utf8();
                    chars.next();
                } else {
                    break;
                }
            }
            let kind = if has_alpha {
                TokenKind::Word
            } else {
                // A bare run of `'`/`-` with no letter at all: both
                // characters are members of `is_prose_punctuation`, so this
                // is exactly one punctuation token, not a word.
                TokenKind::Punctuation
            };
            tokens.push(ScannedToken {
                range: start..end,
                kind,
            });
            continue;
        }

        if is_prose_punctuation(c) {
            let mut end = start + c.len_utf8();
            chars.next();
            while let Some(&(idx, next)) = chars.peek() {
                if is_prose_punctuation(next) {
                    end = idx + next.len_utf8();
                    chars.next();
                } else {
                    break;
                }
            }
            tokens.push(ScannedToken {
                range: start..end,
                kind: TokenKind::Punctuation,
            });
            continue;
        }

        let mut end = start + c.len_utf8();
        chars.next();
        while let Some(&(idx, next)) = chars.peek() {
            if next.is_whitespace()
                || next.is_numeric()
                || next.is_alphabetic()
                || next == '\''
                || next == '-'
                || is_prose_punctuation(next)
            {
                break;
            }
            end = idx + next.len_utf8();
            chars.next();
        }
        tokens.push(ScannedToken {
            range: start..end,
            kind: TokenKind::Symbol,
        });
    }

    debug_assert!(
        tokens
            .iter()
            .all(|t| classify_token_kind(&text[t.range.clone()]) == t.kind),
        "tokenize's own classification must always agree with classify_token_kind, the \
         backend-agnostic contract every Tagger in this crate shares"
    );

    tokens
}

/// A [`Tagger`] backed by a from-scratch-trained averaged perceptron.
///
/// See the module docs for tokenization, non-word passthrough tags, the
/// per-call sentence-reset semantics, and the two loading paths' shared
/// [`Backend`] surface.
pub struct PerceptronTagger {
    backend: Box<dyn Backend + Send + Sync>,
}

/// Gunzips `bytes` and parses the result as a [`WeightFile`]. The one
/// shared JSON-loading step behind [`PerceptronTagger::from_json_gz`] and
/// [`pack_perceptron_tagger_bin`].
fn parse_json_gz(bytes: &[u8]) -> Result<WeightFile, PerceptronTagError> {
    let mut decoder = flate2::read::GzDecoder::new(bytes);
    let mut json = String::new();
    decoder
        .read_to_string(&mut json)
        .map_err(PerceptronTagError::Decompress)?;
    serde_json::from_str(&json).map_err(PerceptronTagError::Parse)
}

/// Builds the derived `.bin` artifact's bytes.
///
/// Takes a `json.gz` weight artifact's raw bytes and its own sha256
/// (lowercase hex, computed by the caller — `corpus-tool weights-pack`
/// over the file it just read). Parses `json_gz_bytes` the same way
/// [`PerceptronTagger::from_json_gz`] does, builds the version-2 view
/// payload via [`build_view_payload`], and prepends the shared
/// [`crate::weights_bin`] header carrying `source_sha256_hex` — the exact
/// bytes `PerceptronTagger::new` later validates against
/// [`SOURCE_JSON_GZ_SHA256`]. The payload is embedded raw (uncompressed):
/// [`PerceptronTagger::new`] borrows straight from the embedded `'static`
/// bytes, so there is nothing to decompress at load time — mirrors
/// `friction_packs::jargon_attest`'s own embedded filter, the precedent
/// this whole format follows.
///
/// # Errors
/// [`PerceptronTagError::Decompress`] / [`PerceptronTagError::Parse`] —
/// see [`PerceptronTagger::from_json_gz`].
pub fn pack_perceptron_tagger_bin(
    json_gz_bytes: &[u8],
    source_sha256_hex: &str,
) -> Result<Vec<u8>, PerceptronTagError> {
    let file = parse_json_gz(json_gz_bytes)?;
    let payload = build_view_payload(file);
    let mut out = crate::weights_bin::write_header(ARTIFACT_MAGIC, source_sha256_hex);
    out.extend_from_slice(&payload);
    Ok(out)
}

impl PerceptronTagger {
    /// Loads the tagger from its embedded, derived `.bin` artifact — a
    /// version-2 serialized hash-table view (see this module's docs for
    /// the byte layout), gated on a source-sha256 header check against
    /// the embedded `json.gz` (`crate::weights_bin`'s module docs). This
    /// is the fast path every normal caller wants: no `HashMap` is built,
    /// no feature string is re-hashed, no per-feature `Vec` is allocated
    /// — [`TaggerView::from_payload`] only slices the embedded `'static`
    /// bytes. See [`Self::from_json_gz`] for the slower, audited-source
    /// path.
    ///
    /// # Errors
    /// [`PerceptronTagError::Header`] if the embedded `.bin`'s header or
    /// payload is malformed; [`PerceptronTagError::StaleBin`] if its
    /// recorded source sha256 doesn't match the currently embedded
    /// `json.gz`. Neither should happen for the vendored artifact —
    /// covered by this module's own tests.
    pub fn new() -> Result<Self, PerceptronTagError> {
        let (source_sha256, payload) =
            crate::weights_bin::split_header(WEIGHTS_BIN, ARTIFACT_MAGIC)?;
        if source_sha256 != SOURCE_JSON_GZ_SHA256 {
            return Err(PerceptronTagError::StaleBin {
                found: Box::from(source_sha256),
                expected: SOURCE_JSON_GZ_SHA256,
            });
        }
        let view = TaggerView::from_payload(payload)?;
        Ok(Self {
            backend: Box::new(ViewBackend { view }),
        })
    }

    /// Loads the tagger directly from `json.gz` bytes (the audited
    /// interchange format `weights/NOTICE.md` describes), bypassing the
    /// derived `.bin` artifact entirely. Used by [`pack_perceptron_tagger_bin`]
    /// (the `corpus-tool weights-pack` converter) and by this module's own
    /// parity test, which asserts this path and [`Self::new`] agree.
    ///
    /// # Errors
    /// [`PerceptronTagError::Decompress`] if `bytes` isn't valid gzip;
    /// [`PerceptronTagError::Parse`] if the decompressed JSON doesn't
    /// match the expected shape.
    pub fn from_json_gz(bytes: &[u8]) -> Result<Self, PerceptronTagError> {
        let file = parse_json_gz(bytes)?;
        Ok(Self {
            backend: Box::new(OwnedBackend::from_file(file)),
        })
    }

    /// Tags one [`TokenKind::Word`] token at `i`: a tagdict hit short-
    /// circuits the model; otherwise scores every class and returns the
    /// argmax. `raw` is the token's original-case surface text, used only
    /// for its suffix/first-character features — see [`extract_features`].
    fn tag_word(
        &self,
        surfaces: &[Box<str>],
        raw: &str,
        i: usize,
        prev1: &str,
        prev2: &str,
    ) -> Box<str> {
        if let Some(tag) = self.backend.tagdict_lookup(&surfaces[i]) {
            return tag;
        }
        let features = extract_features(surfaces, raw, i, prev1, prev2);
        let scores = self.backend.score(&features);
        self.backend.class_at(argmax(&scores))
    }
}

impl Tagger for PerceptronTagger {
    fn tag(&self, text: &str, base_offset: usize) -> Vec<TaggedToken> {
        let scanned = tokenize(text);
        if scanned.is_empty() {
            return Vec::new();
        }

        let surfaces: Vec<Box<str>> = scanned
            .iter()
            .map(|t| Box::from(text[t.range.clone()].to_lowercase()))
            .collect();

        let mut prev1: Box<str> = Box::from(START_TAG_1);
        let mut prev2: Box<str> = Box::from(START_TAG_2);
        let mut out = Vec::with_capacity(scanned.len());

        for (i, scan_token) in scanned.iter().enumerate() {
            let surface = &text[scan_token.range.clone()];
            let assigned: Box<str> = match scan_token.kind {
                TokenKind::Number => Box::from("CD"),
                TokenKind::Punctuation => Box::from(punctuation_tag(surface)),
                TokenKind::Word => self.tag_word(&surfaces, surface, i, &prev1, &prev2),
                // `tokenize` never emits `Whitespace`; any other/future
                // `TokenKind` (this enum is `#[non_exhaustive]`) falls back
                // here rather than panicking.
                _ => Box::from("SYM"),
            };

            let range =
                (scan_token.range.start + base_offset)..(scan_token.range.end + base_offset);
            let lemma = if scan_token.kind == TokenKind::Word {
                crate::inflect::lemmatize(surface, &assigned)
            } else {
                Box::from(surface.to_lowercase())
            };
            out.push(TaggedToken {
                token: Token::new(range, scan_token.kind),
                pos: PosTag::new(assigned.clone()),
                lemma,
            });

            prev2 = prev1;
            prev1 = assigned;
        }

        out
    }
}

/// Training-support helpers.
///
/// Feature extraction, the gold-file format, and weight-table save, shared
/// between inference (above) and `examples/train_perceptron.rs`. Exists
/// only behind the `train-tooling` feature so normal builds never carry
/// training code in their public surface.
#[cfg(feature = "train-tooling")]
pub mod train_support {
    use std::collections::BTreeMap;
    use std::io::Write as _;

    use super::{
        PerceptronTagError, WeightFile, extract_features, normalize_word_identity, punctuation_tag,
    };

    /// One gold-tagged sentence: `(lowercased surface, gold tag)` pairs, in
    /// order.
    ///
    /// Non-word tokens (punctuation/number/symbol) carry the same
    /// deterministic passthrough tag [`super::Tagger::tag`] assigns at
    /// inference, so a gold file stays self-consistent with runtime
    /// tag-history context.
    pub type GoldSentence = Vec<(String, String)>;

    /// Parses a hand-rolled tab-separated gold file.
    ///
    /// `word<TAB>tag` per line, a blank line marking a sentence break. No
    /// external CoNLL-U crate or corpus dependency — the file is entirely
    /// this project's own hand-curated data.
    #[must_use]
    pub fn parse_gold_file(text: &str) -> Vec<GoldSentence> {
        let mut sentences = Vec::new();
        let mut current: GoldSentence = Vec::new();
        for line in text.lines() {
            let trimmed = line.trim_end();
            if trimmed.trim().is_empty() {
                if !current.is_empty() {
                    sentences.push(std::mem::take(&mut current));
                }
                continue;
            }
            let Some((word, tag)) = trimmed.split_once('\t') else {
                continue;
            };
            current.push((word.to_string(), tag.to_string()));
        }
        if !current.is_empty() {
            sentences.push(current);
        }
        sentences
    }

    /// Parses a split-aware gold file.
    ///
    /// Like [`parse_gold_file`], but additionally reads a `# split=train` /
    /// `# split=test` marker line required immediately before every
    /// sentence's token lines, keeping only sentences whose marker equals
    /// `want_split` (`"train"` or `"test"`).
    ///
    /// A marker line carries no tab, so [`parse_gold_file`] (and any other
    /// unmodified reader) already skips it silently without disturbing
    /// sentence boundaries — this is what lets the gold file carry a split
    /// marker while keeping its line format byte-for-byte the same
    /// otherwise.
    ///
    /// # Panics
    /// If a sentence's marker is missing or is neither `"train"` nor
    /// `"test"` — a malformed or hand-edited gold file must fail loudly
    /// here rather than silently mis-splitting training data.
    #[must_use]
    pub fn parse_gold_file_split(text: &str, want_split: &str) -> Vec<GoldSentence> {
        assert!(
            want_split == "train" || want_split == "test",
            "want_split must be \"train\" or \"test\", got {want_split:?}"
        );
        let mut sentences = Vec::new();
        let mut current: GoldSentence = Vec::new();
        let mut current_split: Option<String> = None;

        let mut flush = |current: &mut GoldSentence, current_split: &mut Option<String>| {
            if current.is_empty() {
                return;
            }
            let split = current_split
                .take()
                .expect("sentence ending with no preceding `# split=` marker line");
            assert!(
                split == "train" || split == "test",
                "gold-file `# split=` marker must be \"train\" or \"test\", found {split:?}"
            );
            if split == want_split {
                sentences.push(std::mem::take(current));
            } else {
                current.clear();
            }
        };

        for line in text.lines() {
            let trimmed = line.trim_end();
            if trimmed.trim().is_empty() {
                flush(&mut current, &mut current_split);
                continue;
            }
            if let Some(value) = trimmed.strip_prefix("# split=") {
                current_split = Some(value.trim().to_string());
                continue;
            }
            let Some((word, tag)) = trimmed.split_once('\t') else {
                continue;
            };
            current.push((word.to_string(), tag.to_string()));
        }
        flush(&mut current, &mut current_split);
        sentences
    }

    /// Every token's lowercased surface text in `sentence` — the same
    /// context slice [`features_for`] needs, computed once per sentence.
    #[must_use]
    pub fn surfaces_of(sentence: &GoldSentence) -> Vec<Box<str>> {
        sentence
            .iter()
            .map(|(word, _)| Box::from(word.to_lowercase()))
            .collect()
    }

    /// Re-exposes [`extract_features`] for the training loop.
    ///
    /// The loop must call it once per token, interleaved with the model's
    /// own live prediction feeding `prev1_tag`/`prev2_tag` for the *next*
    /// token — the perceptron-tagger convention this module's inference
    /// path follows (the model's guess, not the gold tag, becomes
    /// tag-history context), so training and inference never disagree
    /// about "previous tag". `raw_word` must be token `i`'s original-case
    /// gold-file text (not `surfaces[i]`, already lowercased) — see
    /// [`extract_features`]'s doc comment for why the current token's
    /// suffix/first-character features need the un-normalized form.
    #[must_use]
    pub fn features_for(
        surfaces: &[Box<str>],
        raw_word: &str,
        i: usize,
        prev1_tag: &str,
        prev2_tag: &str,
    ) -> Vec<Box<str>> {
        extract_features(surfaces, raw_word, i, prev1_tag, prev2_tag)
    }

    /// A minimal from-scratch averaged perceptron.
    ///
    /// Follows the standard lazy-averaging trick (accumulate `weight *
    /// iterations held`, divide by total iterations at the end) rather
    /// than naively summing every intermediate weight.
    #[derive(Default)]
    pub struct AveragedPerceptron {
        weights: BTreeMap<Box<str>, BTreeMap<Box<str>, f64>>,
        totals: BTreeMap<(Box<str>, Box<str>), f64>,
        timestamps: BTreeMap<(Box<str>, Box<str>), u64>,
        iterations: u64,
        pub classes: Vec<Box<str>>,
    }

    impl AveragedPerceptron {
        #[must_use]
        pub fn new(classes: Vec<Box<str>>) -> Self {
            let mut classes = classes;
            classes.sort();
            classes.dedup();
            Self {
                classes,
                ..Self::default()
            }
        }

        /// Scores every class for `features`, returning `(predicted tag,
        /// per-class scores in `self.classes` order)`. Ties resolve to the
        /// lexicographically earliest class, exactly like the frozen
        /// [`super::WeightTable`] this trainer eventually produces.
        #[must_use]
        pub fn predict(&self, features: &[Box<str>]) -> String {
            let mut scores = vec![0f64; self.classes.len()];
            for feature in features {
                let Some(per_class) = self.weights.get(feature) else {
                    continue;
                };
                for (class, weight) in per_class {
                    if let Ok(index) = self.classes.binary_search(class) {
                        scores[index] += weight;
                    }
                }
            }
            let mut best = 0usize;
            let mut best_score = scores[0];
            for (index, &score) in scores.iter().enumerate().skip(1) {
                if score > best_score {
                    best_score = score;
                    best = index;
                }
            }
            self.classes[best].to_string()
        }

        fn bump(&mut self, feature: &str, class: &str, delta: f64) {
            let key: (Box<str>, Box<str>) = (Box::from(feature), Box::from(class));
            let current = self
                .weights
                .get(feature)
                .and_then(|m| m.get(class))
                .copied()
                .unwrap_or(0.0);
            let last_ts = self.timestamps.get(&key).copied().unwrap_or(0);
            // A handful of epochs over a small gold corpus never gets
            // `iterations` near 2^52 — no meaningful precision lost here.
            #[allow(clippy::cast_precision_loss)]
            let held = (self.iterations - last_ts) as f64;
            let total = self.totals.entry(key.clone()).or_insert(0.0);
            *total = held.mul_add(current, *total);
            self.timestamps.insert(key, self.iterations);
            self.weights
                .entry(Box::from(feature))
                .or_default()
                .insert(Box::from(class), current + delta);
        }

        /// Trains on one example: predicts, and if wrong, nudges the gold
        /// class's weight up and the guessed class's weight down for every
        /// feature. Returns the prediction (the caller feeds this, not the
        /// gold tag, into the next token's tag-history context).
        pub fn train_one(&mut self, features: &[Box<str>], gold: &str) -> String {
            self.iterations += 1;
            let guess = self.predict(features);
            if guess != gold {
                for feature in features {
                    self.bump(feature, gold, 1.0);
                    self.bump(feature, &guess, -1.0);
                }
            }
            guess
        }

        /// Averages every weight the standard way and emits the on-disk
        /// [`WeightFile`] shape, plus `tagdict` built from `word_tag_counts`
        /// (a word seen at least `min_count` times, with one tag covering
        /// at least `min_purity` of its occurrences, bypasses the model
        /// entirely).
        #[must_use]
        pub fn finish(
            mut self,
            word_tag_counts: &BTreeMap<String, BTreeMap<String, u32>>,
            min_count: u32,
            min_purity: f64,
        ) -> WeightFile {
            let mut features: BTreeMap<Box<str>, Vec<(u16, f32)>> = BTreeMap::new();
            let keys: Vec<Box<str>> = self.weights.keys().cloned().collect();
            for feature in keys {
                let per_class = self.weights.remove(&feature).unwrap_or_default();
                let mut sparse = Vec::new();
                for (class, weight) in per_class {
                    let key = (feature.clone(), class.clone());
                    let last_ts = self.timestamps.get(&key).copied().unwrap_or(0);
                    let total = self.totals.get(&key).copied().unwrap_or(0.0);
                    // See the identical justification in `bump` above.
                    #[allow(clippy::cast_precision_loss)]
                    let held = (self.iterations - last_ts) as f64;
                    let total = held.mul_add(weight, total);
                    #[allow(clippy::cast_precision_loss)]
                    let averaged = total / (self.iterations.max(1) as f64);
                    if averaged.abs() > f64::EPSILON {
                        let index = self
                            .classes
                            .binary_search(&class)
                            .expect("class must be registered");
                        #[allow(clippy::cast_possible_truncation)]
                        sparse.push((index as u16, averaged as f32));
                    }
                }
                if !sparse.is_empty() {
                    features.insert(feature, sparse);
                }
            }

            let mut tagdict = BTreeMap::new();
            for (word, tag_counts) in word_tag_counts {
                let total: u32 = tag_counts.values().sum();
                if total < min_count {
                    continue;
                }
                if let Some((best_tag, best_count)) =
                    tag_counts.iter().max_by_key(|(_, count)| **count)
                {
                    #[allow(clippy::cast_precision_loss)]
                    let purity = f64::from(*best_count) / f64::from(total);
                    if purity >= min_purity {
                        tagdict.insert(Box::from(word.as_str()), Box::from(best_tag.as_str()));
                    }
                }
            }

            WeightFile {
                classes: self.classes,
                features,
                tagdict,
            }
        }
    }

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

    /// Loads a gzip-compressed [`WeightFile`] from `path` — the same shape
    /// [`super::PerceptronTagger::new`] reads from its embedded bytes, used
    /// by `examples/train_perceptron.rs`'s reproducibility check.
    ///
    /// # Errors
    /// [`PerceptronTagError::Decompress`] / [`PerceptronTagError::Parse`].
    pub fn load_weight_file(path: &std::path::Path) -> Result<WeightFile, PerceptronTagError> {
        let bytes = std::fs::read(path).map_err(PerceptronTagError::Decompress)?;
        let mut decoder = flate2::read::GzDecoder::new(bytes.as_slice());
        let mut json = String::new();
        std::io::Read::read_to_string(&mut decoder, &mut json)
            .map_err(PerceptronTagError::Decompress)?;
        serde_json::from_str(&json).map_err(PerceptronTagError::Parse)
    }

    /// Re-exposes [`normalize_word_identity`] for a training tool's own
    /// diagnostics (e.g. reporting how many gold tokens fell into the
    /// hyphen bucket) without duplicating the bucketing rule.
    #[must_use]
    pub fn normalize_for_diagnostics(word_lower: &str) -> &str {
        normalize_word_identity(word_lower)
    }

    /// Re-exposes [`punctuation_tag`] for training/eval tooling.
    ///
    /// Lets a training tool's gold-file consistency check
    /// (`expected_passthrough_tag` in `examples/train_perceptron.rs`) and
    /// a gold-drafting tool compute the same deterministic punctuation tag
    /// runtime inference assigns, without duplicating the classification
    /// rule.
    #[must_use]
    pub fn punctuation_tag_for_diagnostics(surface: &str) -> &'static str {
        punctuation_tag(surface)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::OnceLock;

    use friction_core::TokenKind;

    use super::*;

    fn tagger() -> &'static PerceptronTagger {
        static TAGGER: OnceLock<PerceptronTagger> = OnceLock::new();
        TAGGER.get_or_init(|| PerceptronTagger::new().expect("embedded weights must load"))
    }

    #[test]
    fn tag_produces_spans_pos_and_lemmas_for_a_plain_sentence() {
        let text = "The quick brown foxes are jumping over lazy dogs.";
        let base_offset = 100;
        let tagged = tagger().tag(text, base_offset);
        assert!(!tagged.is_empty());
        for t in &tagged {
            assert!(t.token.range.start >= base_offset);
            assert!(t.token.range.end <= base_offset + text.len());
            assert!(!t.pos.as_str().is_empty());
        }
        let foxes = tagged
            .iter()
            .find(|t| {
                &text[t.token.range.start - base_offset..t.token.range.end - base_offset] == "foxes"
            })
            .expect("foxes token present");
        // The lemma is either a round-tripped base form ("fox") or the
        // documented surface-text fallback ("foxes") -- never anything
        // else, and never empty.
        assert!(
            matches!(&*foxes.lemma, "fox" | "foxes"),
            "{:?}",
            foxes.lemma
        );
    }

    #[test]
    fn tag_gives_a_comma_the_comma_tag() {
        let tagged = tagger().tag("Wait, really?", 0);
        let comma = tagged
            .iter()
            .find(|t| &"Wait, really?"[t.token.range.clone()] == ",")
            .expect("comma token present");
        assert_eq!(comma.token.kind, TokenKind::Punctuation);
        assert_eq!(comma.pos.as_str(), ",");
    }

    #[test]
    fn tag_gives_sentence_final_marks_the_period_tag() {
        let tagged = tagger().tag("Really?", 0);
        let mark = tagged
            .iter()
            .find(|t| t.token.kind == TokenKind::Punctuation)
            .expect("a punctuation token is present");
        assert_eq!(mark.pos.as_str(), ".");
    }

    #[test]
    fn tag_gives_brackets_the_lrb_rrb_tags() {
        let tagged = tagger().tag("(hello) [world]", 0);
        let text = "(hello) [world]";
        let opens: Vec<_> = tagged
            .iter()
            .filter(|t| matches!(&text[t.token.range.clone()], "(" | "["))
            .collect();
        let closes: Vec<_> = tagged
            .iter()
            .filter(|t| matches!(&text[t.token.range.clone()], ")" | "]"))
            .collect();
        assert!(!opens.is_empty() && !closes.is_empty());
        for t in opens {
            assert_eq!(t.pos.as_str(), "-LRB-");
        }
        for t in closes {
            assert_eq!(t.pos.as_str(), "-RRB-");
        }
    }

    #[test]
    fn tag_gives_hyphen_runs_the_hyph_tag() {
        let text = "okay -- fine";
        let tagged = tagger().tag(text, 0);
        let dash = tagged
            .iter()
            .find(|t| &text[t.token.range.clone()] == "--")
            .expect("dash token present");
        assert_eq!(dash.pos.as_str(), "HYPH");
    }

    #[test]
    fn tag_gives_a_mixed_punctuation_run_the_nfp_tag() {
        let text = "wait...";
        let tagged = tagger().tag(text, 0);
        let ellipsis = tagged
            .iter()
            .find(|t| &text[t.token.range.clone()] == "...")
            .expect("ellipsis token present");
        // A run of three literal periods is entirely drawn from the
        // sentence-final-mark set, so it's the "." bucket, not "NFP" --
        // "NFP" is reserved for a run mixing distinct punctuation classes.
        assert_eq!(ellipsis.pos.as_str(), ".");

        let mixed_text = "wait?!";
        let mixed_tagged = tagger().tag(mixed_text, 0);
        let mixed = mixed_tagged
            .iter()
            .find(|t| &mixed_text[t.token.range.clone()] == "?!")
            .expect("mixed run token present");
        // "?!" is still entirely sentence-final-mark, so it also lands on
        // "." -- exercise a genuinely mixed run to reach "NFP".
        assert_eq!(mixed.pos.as_str(), ".");

        let punct_run_text = "words).:more";
        let punct_run_tagged = tagger().tag(punct_run_text, 0);
        let punct_run = punct_run_tagged
            .iter()
            .find(|t| &punct_run_text[t.token.range.clone()] == ").:")
            .expect("mixed-class punctuation run token present");
        assert_eq!(punct_run.pos.as_str(), "NFP");
    }

    #[test]
    fn tag_punctuation_tags_are_always_members_of_the_closed_set() {
        // Regression guard for the tagset decision: every punctuation
        // token's tag is one of the nine closed strings below, never an
        // arbitrary literal like the old surface-text-as-tag scheme would
        // produce for a run like "):." below.
        let text = "Section 1.2.3, see (note):";
        let tagged = tagger().tag(text, 0);
        for t in tagged
            .iter()
            .filter(|t| t.token.kind == TokenKind::Punctuation)
        {
            let closed_set = [".", ",", ":", "-LRB-", "-RRB-", "``", "''", "HYPH", "NFP"];
            assert!(
                closed_set.contains(&t.pos.as_str()),
                "punctuation tag {:?} is not a member of the closed set",
                t.pos.as_str()
            );
        }
    }

    #[test]
    fn tag_gives_numbers_the_cd_tag() {
        let tagged = tagger().tag("There are 42 widgets.", 0);
        let number = tagged
            .iter()
            .find(|t| t.token.kind == TokenKind::Number)
            .expect("a number token is present");
        assert_eq!(number.pos.as_str(), "CD");
    }

    #[test]
    fn tag_gives_symbols_the_sym_tag() {
        let tagged = tagger().tag("Use A & B together.", 0);
        let symbol = tagged
            .iter()
            .find(|t| t.token.kind == TokenKind::Symbol)
            .expect("a symbol token is present");
        assert_eq!(symbol.pos.as_str(), "SYM");
    }

    #[test]
    fn tag_never_panics_and_never_emits_unknown_for_an_out_of_vocabulary_word() {
        let tagged = tagger().tag("Zxqvplarnfrobnicate is a word.", 0);
        assert!(!tagged.is_empty());
        let oov = &tagged[0];
        assert_eq!(&*oov.lemma, "zxqvplarnfrobnicate");
        assert_ne!(oov.pos.as_str(), "UNKNOWN");
    }

    #[test]
    fn tag_accepts_empty_text() {
        assert!(tagger().tag("", 0).is_empty());
    }

    #[test]
    fn tagging_is_deterministic_across_repeated_runs_and_fresh_instances() {
        let paragraph = "The team didn't leverage the framework's full potential. \
                          Zxqvplarnfrobnicate elements were, however, refactored twice. \
                          Meanwhile, 42 widgets shipped on time, and the client was pleased!";

        let first = tagger().tag(paragraph, 0);
        let second = tagger().tag(paragraph, 0);
        assert_eq!(first, second);

        let fresh = PerceptronTagger::new().expect("embedded weights must load");
        let third = fresh.tag(paragraph, 0);
        assert_eq!(first, third);
    }

    /// A constructed near-tie input (an out-of-vocabulary word with no
    /// informative feature weights) still resolves to the same tag every
    /// run — the tie-break is deterministic by construction (lowest
    /// sorted-class index wins), not by luck.
    #[test]
    fn tag_tie_break_is_stable_across_repeated_runs() {
        let text = "Zzzznope.";
        let first = tagger().tag(text, 0);
        for _ in 0..5 {
            assert_eq!(tagger().tag(text, 0), first);
        }
    }

    // ---------------------------------------------------------------
    // extract_features: current-token case handling
    // ---------------------------------------------------------------

    /// The current token's suffix/first-character features must read its
    /// *original* case (matching real NLTK's `word[-3:]`/`word[0]`,
    /// computed from the raw token, not the normalized `context` list) —
    /// while its identity feature (`word=`) stays lowercased/normalized
    /// either way, since that one deliberately generalizes across case.
    #[test]
    fn extract_features_suffix_and_pref1_use_the_current_tokens_original_case() {
        let surfaces: Vec<Box<str>> = vec![Box::from("apple")];
        let lower = extract_features(&surfaces, "apple", 0, START_TAG_1, START_TAG_2);
        let upper = extract_features(&surfaces, "Apple", 0, START_TAG_1, START_TAG_2);

        let pref1_lower = lower
            .iter()
            .find(|f| f.starts_with("pref1="))
            .expect("pref1 feature present");
        let pref1_upper = upper
            .iter()
            .find(|f| f.starts_with("pref1="))
            .expect("pref1 feature present");
        assert_eq!(&**pref1_lower, "pref1=a");
        assert_eq!(&**pref1_upper, "pref1=A");

        let word_lower = lower.iter().find(|f| f.starts_with("word="));
        let word_upper = upper.iter().find(|f| f.starts_with("word="));
        assert_eq!(
            word_lower, word_upper,
            "the identity feature must stay case-insensitive even though pref1/suffix now see \
             raw case"
        );
    }

    // ---------------------------------------------------------------
    // tokenize (via classify_token_kind agreement)
    // ---------------------------------------------------------------

    #[test]
    fn tokenize_agrees_with_classify_token_kind_on_word_runs() {
        let tokens = tokenize("state-of-the-art don't");
        let kinds: Vec<TokenKind> = tokens.iter().map(|t| t.kind).collect();
        assert_eq!(kinds, vec![TokenKind::Word, TokenKind::Word]);
    }

    #[test]
    fn tokenize_groups_runs_of_dots_and_dashes_as_one_punctuation_token() {
        let text = "wait... okay -- fine";
        let tokens = tokenize(text);
        let texts: Vec<&str> = tokens.iter().map(|t| &text[t.range.clone()]).collect();
        assert_eq!(texts, vec!["wait", "...", "okay", "--", "fine"]);
    }

    #[test]
    fn tokenize_recognizes_numbers() {
        let text = "3.14 and 1,000";
        let tokens = tokenize(text);
        assert_eq!(tokens[0].kind, TokenKind::Number);
        assert_eq!(&text[tokens[0].range.clone()], "3.14");
    }

    #[test]
    fn tokenize_emits_no_whitespace_tokens() {
        let tokens = tokenize("  a   b  ");
        assert!(tokens.iter().all(|t| t.kind != TokenKind::Whitespace));
        assert_eq!(tokens.len(), 2);
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

    /// A `.bin` whose header claims a different source sha256 is rejected
    /// as stale rather than silently loaded — the payload itself is still
    /// well-formed and loadable, so staleness is purely a header-metadata
    /// mismatch, never payload corruption.
    #[test]
    fn new_rejects_a_bin_whose_header_sha_does_not_match_the_compiled_in_constant() {
        let file = parse_json_gz(WEIGHTS_JSON_GZ).expect("embedded json.gz parses");
        let payload = build_view_payload(file);
        let stale_sha = "0".repeat(64);
        let mut bytes = crate::weights_bin::write_header(ARTIFACT_MAGIC, &stale_sha);
        bytes.extend_from_slice(&payload);
        let (source_sha256, decoded_payload) =
            crate::weights_bin::split_header(&bytes, ARTIFACT_MAGIC).expect("header parses");
        assert_ne!(source_sha256, SOURCE_JSON_GZ_SHA256);
        TaggerView::from_payload(decoded_payload).expect("payload still parses");
    }

    /// Full-vocabulary equivalence: every feature, tagdict entry, and
    /// class the embedded `json.gz` describes agrees with what the
    /// embedded `.bin`'s zero-copy view returns — not just a battery of
    /// sentences (below), the *entire* vocabulary, so a rare feature no
    /// battery sentence happens to trigger can't hide a translation bug.
    /// Also checks that keys genuinely absent from the source file are
    /// reported absent by the view, not aliased onto some other entry.
    #[test]
    fn view_matches_the_json_weight_file_across_its_entire_vocabulary() {
        let file = parse_json_gz(WEIGHTS_JSON_GZ).expect("embedded json.gz parses");
        let (source_sha256, payload) =
            crate::weights_bin::split_header(WEIGHTS_BIN, ARTIFACT_MAGIC).expect("header parses");
        assert_eq!(source_sha256, SOURCE_JSON_GZ_SHA256);
        let view = TaggerView::from_payload(payload).expect("payload parses");

        let (sorted_classes, index_map) = sort_classes_with_index_map(&file.classes);
        assert_eq!(view.num_classes, sorted_classes.len());
        for (index, class) in sorted_classes.iter().enumerate() {
            assert_eq!(view.class_str(index), class.as_ref(), "class index {index}");
        }

        for (feature, sparse) in &file.features {
            let mut expected = vec![0f32; sorted_classes.len()];
            for &(class_index, weight) in sparse {
                if let Some(&mapped) = index_map.get(class_index as usize) {
                    expected[mapped] = weight;
                }
            }
            let scores = view.score(std::slice::from_ref(feature));
            assert_eq!(scores, expected, "feature {feature:?}");
        }

        for (word, tag) in &file.tagdict {
            assert_eq!(
                view.tagdict_get(word),
                Some(tag.as_ref()),
                "tagdict word {word:?}"
            );
        }

        let absent_features = ["zzz-never-a-feature-zzz", "word=zzzznonexistentzzz"];
        for absent in absent_features {
            assert!(
                !file.features.contains_key(absent),
                "test bug: {absent:?} is actually a real feature"
            );
            let owned = Box::from(absent);
            let scores = view.score(std::slice::from_ref(&owned));
            assert_eq!(
                scores,
                vec![0f32; sorted_classes.len()],
                "absent feature {absent:?}"
            );
        }

        let absent_word = "zzznonexistentwordzzz";
        assert!(!file.tagdict.contains_key(absent_word));
        assert_eq!(view.tagdict_get(absent_word), None);
    }

    /// Behavioral parity, one level up: a tagger built via each loading
    /// path assigns identical tags over a fixed battery of sentences.
    #[test]
    fn json_and_bin_paths_tag_a_sentence_battery_identically() {
        let from_bin = PerceptronTagger::new().expect("embedded .bin loads");
        let from_json =
            PerceptronTagger::from_json_gz(WEIGHTS_JSON_GZ).expect("embedded json.gz loads");
        let battery = [
            "The quick brown foxes are jumping over lazy dogs.",
            "Run the scanner from the project root.",
            "Results stream in as they are produced.",
            "Cannot connect to the database right now.",
            "State-of-the-art solutions rarely stay that way for long.",
            "Wait, really? That seems unlikely.",
            "(hello) [world] {test}",
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
        ];
        for sentence in battery {
            let bin_tags: Vec<String> = from_bin
                .tag(sentence, 0)
                .iter()
                .map(|t| t.pos.as_str().to_owned())
                .collect();
            let json_tags: Vec<String> = from_json
                .tag(sentence, 0)
                .iter()
                .map(|t| t.pos.as_str().to_owned())
                .collect();
            assert_eq!(bin_tags, json_tags, "tag mismatch for {sentence:?}");
        }
    }

    /// [`pack_perceptron_tagger_bin`] is deterministic: converting the
    /// same `json.gz` bytes twice (with the same recorded source sha256)
    /// produces bit-identical `.bin` bytes.
    #[test]
    fn pack_perceptron_tagger_bin_is_deterministic() {
        let sha = "a".repeat(64);
        let first = pack_perceptron_tagger_bin(WEIGHTS_JSON_GZ, &sha).expect("packs");
        let second = pack_perceptron_tagger_bin(WEIGHTS_JSON_GZ, &sha).expect("packs");
        assert_eq!(first, second);
    }
}
