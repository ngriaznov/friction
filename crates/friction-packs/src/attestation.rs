//! Attestation pack: a seam-bigram membership table plus POS-skeleton
//! n-gram sets mined from TRAIN-split human docs.
//!
//! `corpus-tool attest` mines both tables from `friction_harness::clean`'s
//! tokenization convention (bigram side) and the shipped part-of-speech
//! tagger's coarse tags (skeleton side); this module is the read side,
//! turning the versioned TOML pack it writes (`attestation-en-v1.toml`,
//! embedded and exposed pre-parsed as [`crate::ATTESTATION`]) into two
//! small, queryable in-memory structures — the same "embedded TOML ->
//! in-memory struct" shape [`crate::DmsIndex`] already established.
//!
//! # What the two tables are for
//!
//! [`BigramTable`] answers one boolean question: has this exact
//! left/right word pair been observed adjacent to each other anywhere in
//! the TRAIN human corpus (the reserved `"<s>"` token stands for
//! sentence-start, so a left of `"<s>"` asks "does a human sentence ever
//! open with this word"). [`SkeletonSet`] answers a similar boolean
//! question one level more abstract: has this exact run of coarse
//! part-of-speech tags (`<S>`/`<E>` sentinels included) been observed as
//! a human sentence's skeleton. Both are membership-only (no counts, no
//! frequencies) attestation is a question of existence, and a frequency
//! threshold would just be a second, hidden calibration knob.
//!
//! # Independent tokenizations, on purpose
//!
//! The bigram table's tokens and the skeleton set's tags are built from
//! two different tokenizations of the same sentence text, and this
//! module does not try to force them into positional correspondence with
//! each other:
//!
//! - the bigram side uses `friction_harness::clean::tokenize`'s
//!   convention (`[a-z']+` word runs, single-character `.,;:!?`
//!   punctuation, digits silently dropped) — the exact convention
//!   `friction_match::token::tokenize_str` already shares (see that
//!   module's own doc comment), so a caller building `(left, right)` from
//!   a real document's matched span and one built from `corpus-tool
//!   attest`'s mining pass always agree on where a "word" begins and
//!   ends;
//! - the skeleton side uses the shipped tagger's own tokenizer (which
//!   groups punctuation runs into one token, keeps hyphenated words
//!   whole, and tags digits `CD` instead of dropping them) — the same
//!   design `corpus-tool mine-inventory`'s own `skeleton_word_runs`
//!   already established: tag the sentence text directly and walk the
//!   tagger's own token stream, never re-paired against a second,
//!   independently-tokenized word list.
//!
//! Trying to zip these two streams together by position is a real
//! correctness trap (a run of `"..."` is three tokens on the bigram side
//! and one on the skeleton side; a hyphenated word is four tokens on the
//! bigram side and one on the skeleton side) that this design sidesteps
//! entirely by never attempting it: nothing in either table's query API
//! needs the two streams to have walked in lockstep, only that each
//! table was built, and is queried, through its own single convention
//! consistently.
//!
//! # Widened evidence: the pooled seam filter
//!
//! [`BigramTable::attests`] is the ONE entry point every gate in
//! `friction-edit` calls. [`AttestationPack::with_pooled`] attaches
//! [`crate::pooled_attest::PooledAttestPack`] — a much larger membership
//! structure mined from external human corpora, see that module's own
//! docs for why the TRAIN-only table alone is a measured recall
//! bottleneck — to a `BigramTable`, after which `attests` answers `true`
//! on either the exact TRAIN hit OR a pooled-filter hit. Every existing
//! caller of `attests` (`friction-edit`'s deletion and rewrite gates)
//! sees the widened evidence automatically, with no call-site change:
//! see `attests`'s own docs for the exact precedence. Only
//! [`crate::registry::ATTESTATION`] attaches the real embedded pooled
//! pack; [`AttestationPack::parse`] and [`AttestationPack::from_bin`]
//! both leave `pooled` unset (`None`), so every other caller (offline
//! `corpus-tool` passes, this module's own tests, a hand-built pack in a
//! unit test elsewhere) keeps exactly its historical, pooled-absent
//! behavior.

use std::collections::{BTreeMap, BTreeSet};

use serde::Deserialize;

use crate::pooled_attest::PooledAttestPack;
use crate::{PackError, Sha256};

/// A pack's word vocabulary (bigram side) or coarse-tag vocabulary
/// (skeleton side): `tokens[i]` is the text for id `i`.
///
/// Deliberately not [`crate::Vocab`] (the DMS pack's own vocab type,
/// `pub(self)`-scoped to `dms.rs`): the two packs' vocabularies serve
/// different lookup shapes and have no reason to share an implementation
/// beyond the coincidence that both are "text interned to a small `u32`
/// id space".
#[derive(Debug, Clone, Default)]
struct TokenVocab {
    tokens: Vec<Box<str>>,
    by_text: BTreeMap<Box<str>, u32>,
}

impl TokenVocab {
    fn from_tokens(tokens: Vec<String>) -> Self {
        let tokens: Vec<Box<str>> = tokens.into_iter().map(String::into_boxed_str).collect();
        let mut by_text: BTreeMap<Box<str>, u32> = BTreeMap::new();
        for (index, token) in tokens.iter().enumerate() {
            #[allow(clippy::cast_possible_truncation)]
            let id = index as u32;
            by_text.entry(token.clone()).or_insert(id);
        }
        Self { tokens, by_text }
    }

    fn id_of(&self, text: &str) -> Option<u32> {
        self.by_text.get(text).copied()
    }

    #[cfg(test)]
    fn token_of(&self, id: u32) -> Option<&str> {
        self.tokens.get(id as usize).map(AsRef::as_ref)
    }

    const fn len(&self) -> usize {
        self.tokens.len()
    }
}

// ---------------------------------------------------------------------
// Derived binary artifact (`attestation-en-v1.bin`).
//
// The TOML parse above stays the audited source-of-truth read side, used
// offline by `corpus-tool` — but it paid ~25 ms on every process start
// (full `toml::from_str` over 1.8 MB, then B-tree materialization of
// ~16k vocab tokens, ~125k bigram edges, and ~96k skeleton codes): the
// single largest slice of `friction`'s fixed startup cost. This is the
// same cure the DMS index already received (`crate::dms_bin`): do the
// work once, offline, in `corpus-tool attest-pack`, and serialize the
// finished tables in a layout the runtime reads as a zero-copy view over
// the `include_bytes!`'d static: header validation plus slice splits at
// load, no per-entry work.
//
// Layout (all integers little-endian; every multi-byte field is read via
// `from_le_bytes` on a byte slice — alignment-free by construction, so
// no `unsafe` is ever needed):
//
// - **Header**: 8-byte magic `FRATTEST`, `u16` format version, 64-byte
//   lowercase-hex sha256 of the source TOML (the same staleness guard
//   `dms_bin` records: the registry drift test re-packs the TOML and
//   compares bytes).
// - **Near-no-op calibration**: `u8` presence flag; when present, the
//   threshold's `f64` bits, `u32` sample doc count, `u8` dev-check flag
//   (+ its `f64` bits when set), and a `u32`-length-prefixed method
//   string. `f64::to_le_bytes`/`from_le_bytes` round-trip bit-exactly.
// - **Word vocab**, then later **tag vocab**, both in `dms_bin`'s vocab
//   shape: `u32` token count `n`; `n + 1` cumulative `u32` byte offsets
//   into a string pool; `u32` pool length + the pool; and a first-id-wins
//   duplicate-collapsed lookup table of ids sorted by token text (`u32`
//   count + that many `u32` ids) for `id_of`'s binary search — exactly
//   [`TokenVocab`]'s `entry().or_insert(first)` resolution, precomputed.
// - **Bigram edges**: `u32` distinct-left count (precomputed for
//   [`BigramTable::distinct_lefts`]), `u32` edge count, then each edge
//   packed `(left << 32) | right` as a `u64`, strictly ascending — the
//   flat equivalent of the owned `BTreeMap<u32, BTreeSet<u32>>`, whose
//   sorted iteration order is exactly this packed order.
// - **Skeleton codes**: `tag5` then `tag4`, each a `u32` count plus that
//   many strictly-ascending `u64` codes (the owned `BTreeSet<u64>`s,
//   flattened in order).
//
// Determinism: `pack_attestation_bin` is a pure function of the TOML
// text — every section serializes a `BTreeMap`/`BTreeSet` iteration.
// Equivalence: each view query is the owned query with the B-tree probe
// replaced by a binary search over the identical sorted content, which
// cannot change any membership answer; pinned by this module's own
// round-trip tests and the CLI's byte-identical snapshot battery.
// ---------------------------------------------------------------------

/// The 8-byte magic every attestation binary artifact starts with.
const BIN_MAGIC: [u8; 8] = *b"FRATTEST";

/// The artifact's on-disk format version: bumped if the layout above
/// ever changes shape.
const BIN_FORMAT_VERSION: u16 = 1;

/// A sha256 digest is recorded as lowercase hex (matching
/// [`crate::Sha256`]'s `Display`) — same choice as `dms_bin`'s header.
const SHA256_HEX_LEN: usize = 64;

/// Reads the little-endian `u32` at logical index `index` of a `u32`
/// array serialized as raw bytes.
fn u32_at(region: &[u8], index: usize) -> u32 {
    let start = index * 4;
    u32::from_le_bytes(
        region[start..start + 4]
            .try_into()
            .expect("caller-validated 4-byte slice"),
    )
}

/// Reads the little-endian `u64` at logical index `index` of a `u64`
/// array serialized as raw bytes.
fn u64_at(region: &[u8], index: usize) -> u64 {
    let start = index * 8;
    u64::from_le_bytes(
        region[start..start + 8]
            .try_into()
            .expect("caller-validated 8-byte slice"),
    )
}

/// `true` if the strictly-ascending serialized `u64` array in `region`
/// contains `value` — the binary-search counterpart of the owned
/// `BTreeSet<u64>::contains`.
fn sorted_u64_contains(region: &[u8], value: u64) -> bool {
    let mut lo: usize = 0;
    let mut hi: usize = region.len() / 8;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        match u64_at(region, mid).cmp(&value) {
            std::cmp::Ordering::Less => lo = mid + 1,
            std::cmp::Ordering::Greater => hi = mid,
            std::cmp::Ordering::Equal => return true,
        }
    }
    false
}

/// A tiny forward-only cursor for [`AttestationPack::from_bin`]'s
/// sequential section reads — deliberately its own copy rather than a
/// shared abstraction with `dms_bin`'s (same "no forced sharing"
/// reasoning as [`TokenVocab`] vs `crate::Vocab` above), erroring in
/// this module's own [`PackError`] vocabulary.
struct BinCursor {
    bytes: &'static [u8],
    pos: usize,
}

impl BinCursor {
    const fn new(bytes: &'static [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn take(&mut self, len: usize) -> Result<&'static [u8], PackError> {
        let end = self
            .pos
            .checked_add(len)
            .filter(|&end| end <= self.bytes.len())
            .ok_or(PackError::AttestationBinMalformed(
                "section ran past the end of the artifact",
            ))?;
        let slice = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn read_u8(&mut self) -> Result<u8, PackError> {
        Ok(self.take(1)?[0])
    }

    fn read_u32(&mut self) -> Result<u32, PackError> {
        let slice = self.take(4)?;
        Ok(u32::from_le_bytes(
            slice.try_into().expect("checked 4-byte slice"),
        ))
    }

    fn read_f64(&mut self) -> Result<f64, PackError> {
        let slice = self.take(8)?;
        Ok(f64::from_le_bytes(
            slice.try_into().expect("checked 8-byte slice"),
        ))
    }

    const fn is_exhausted(&self) -> bool {
        self.pos == self.bytes.len()
    }
}

/// Zero-copy view over a serialized vocab section: [`TokenVocab`]'s exact
/// lookup surface (first-id-wins for duplicate texts included), over
/// borrowed bytes.
#[derive(Debug, Clone, Copy)]
struct VocabView {
    /// Total token-id count (duplicates included) — [`TokenVocab::len`].
    count: u32,
    /// `count + 1` cumulative byte offsets into `pool`.
    offsets: &'static [u8],
    pool: &'static [u8],
    /// Token ids sorted by token text, duplicates collapsed first-id-wins.
    sorted_count: u32,
    sorted_ids: &'static [u8],
}

impl VocabView {
    /// The token text for `id`, as bytes — the comparison key
    /// [`Self::id_of`]'s binary search probes with.
    fn token_bytes(&self, id: u32) -> &'static [u8] {
        let start = u32_at(self.offsets, id as usize) as usize;
        let end = u32_at(self.offsets, id as usize + 1) as usize;
        &self.pool[start..end]
    }

    /// The id for `text`, or `None` if out-of-vocab —
    /// [`TokenVocab::id_of`]'s contract.
    fn id_of(&self, text: &str) -> Option<u32> {
        let key = text.as_bytes();
        let mut lo: u32 = 0;
        let mut hi: u32 = self.sorted_count;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let id = u32_at(self.sorted_ids, mid as usize);
            match self.token_bytes(id).cmp(key) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => return Some(id),
            }
        }
        None
    }

    /// The text for `id`, or `None` if `id` is not a valid id —
    /// [`TokenVocab::token_of`]'s contract.
    #[cfg(test)]
    fn token_of(&self, id: u32) -> Option<&'static str> {
        (id < self.count).then(|| {
            std::str::from_utf8(self.token_bytes(id))
                .expect("pack side wrote every token from a &str")
        })
    }

    const fn len(&self) -> usize {
        self.count as usize
    }
}

/// Seam-bigram membership table: has this exact `(left, right)` word pair
/// been observed adjacent to each other anywhere in the TRAIN human corpus.
///
/// OR, when [`AttestationPack::with_pooled`] has attached one, in the
/// pooled external-corpus filter (see this module's own "Widened
/// evidence" docs above).
///
/// Two representations behind one query surface: `Owned` is what
/// [`AttestationPack::parse`] builds from the source TOML (the offline
/// `corpus-tool` path), `View` is what [`AttestationPack::from_bin`]
/// splits zero-copy out of the embedded derived artifact (the runtime
/// path). Every query answers identically from either: the view's
/// binary searches walk the same sorted content the owned B-trees hold.
/// `pooled` is orthogonal to that split — either representation can carry
/// one, attached after construction by [`AttestationPack::with_pooled`].
#[derive(Debug, Clone)]
pub struct BigramTable {
    repr: BigramRepr,
    /// The widened evidence filter, if [`AttestationPack::with_pooled`]
    /// attached one. `None` for every table built by [`AttestationPack::parse`]
    /// or [`AttestationPack::from_bin`] directly — see this module's
    /// "Widened evidence" docs for who actually attaches this.
    pooled: Option<&'static PooledAttestPack>,
}

#[derive(Debug, Clone)]
enum BigramRepr {
    Owned {
        vocab: TokenVocab,
        edges: BTreeMap<u32, BTreeSet<u32>>,
    },
    View {
        vocab: VocabView,
        distinct_lefts: u32,
        /// Strictly-ascending `(left << 32) | right` packed edges.
        edges: &'static [u8],
    },
}

/// Packs a bigram edge the way the binary artifact stores it.
const fn pack_edge(left_id: u32, right_id: u32) -> u64 {
    ((left_id as u64) << 32) | right_id as u64
}

impl BigramTable {
    /// `true` if `right` was observed immediately following `left` in at
    /// least one TRAIN human sentence. `"<s>"` is a valid `left` (the
    /// reserved sentence-start token: `attests("<s>", w)` asks "does a
    /// human sentence ever open with `w`").
    ///
    /// `false` for a `left` or `right` outside this pack's own vocabulary
    /// (an out-of-vocabulary word can never have been attested, by
    /// definition): not an error, mirroring the mining algorithm's own
    /// boolean membership test — but only for the EXACT table's own
    /// answer; a pooled hit (see below) is vocabulary-independent, since
    /// [`crate::pooled_attest::PooledAttestPack`] hashes raw token text
    /// directly rather than through this table's own id space.
    ///
    /// When [`AttestationPack::with_pooled`] has attached a pooled filter,
    /// this is `exact_hit OR pooled_hit`: the exact TRAIN-table check
    /// above runs first (and short-circuits on a hit, so the pooled
    /// filter's ~1/65536 false-positive rate is never even consulted for
    /// a pair the exact table already knows), falling through to the
    /// pooled filter only on an exact miss. No pooled filter attached
    /// (the common case for every caller except the real embedded
    /// registry pack — see this module's own "Widened evidence" docs)
    /// means this is exactly the exact-table-only answer it always was.
    #[must_use]
    pub fn attests(&self, left: &str, right: &str) -> bool {
        let exact_hit = match &self.repr {
            BigramRepr::Owned { vocab, edges } => {
                let Some(left_id) = vocab.id_of(left) else {
                    return self.pooled_hit(left, right);
                };
                let Some(right_id) = vocab.id_of(right) else {
                    return self.pooled_hit(left, right);
                };
                edges
                    .get(&left_id)
                    .is_some_and(|rights| rights.contains(&right_id))
            }
            BigramRepr::View { vocab, edges, .. } => {
                let Some(left_id) = vocab.id_of(left) else {
                    return self.pooled_hit(left, right);
                };
                let Some(right_id) = vocab.id_of(right) else {
                    return self.pooled_hit(left, right);
                };
                sorted_u64_contains(edges, pack_edge(left_id, right_id))
            }
        };
        exact_hit || self.pooled_hit(left, right)
    }

    /// The pooled filter's own answer for `(left, right)`, or `false` when
    /// [`AttestationPack::with_pooled`] never attached one. Broken out so
    /// both the out-of-vocabulary early return and the exact-miss
    /// fallthrough in [`Self::attests`] read through the same one line.
    fn pooled_hit(&self, left: &str, right: &str) -> bool {
        self.pooled
            .is_some_and(|pooled| pooled.attests(left, right))
    }

    /// The number of distinct left tokens with at least one attested
    /// right token (including `"<s>"` if any sentence's first word was
    /// mined).
    ///
    /// Parity-test introspection only; no runtime consumer.
    #[must_use]
    pub fn distinct_lefts(&self) -> usize {
        match &self.repr {
            BigramRepr::Owned { edges, .. } => edges.len(),
            BigramRepr::View { distinct_lefts, .. } => *distinct_lefts as usize,
        }
    }

    /// The total number of `(left, right)` pairs attested across every
    /// left token.
    ///
    /// Parity-test introspection only; no runtime consumer.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        match &self.repr {
            BigramRepr::Owned { edges, .. } => edges.values().map(BTreeSet::len).sum(),
            BigramRepr::View { edges, .. } => edges.len() / 8,
        }
    }

    /// This table's word vocabulary size (including the reserved `"<s>"`
    /// entry).
    ///
    /// Parity-test introspection only; no runtime consumer.
    #[must_use]
    pub const fn vocab_len(&self) -> usize {
        match &self.repr {
            BigramRepr::Owned { vocab, .. } => vocab.len(),
            BigramRepr::View { vocab, .. } => vocab.len(),
        }
    }
}

/// POS-skeleton n-gram set: has this exact run of coarse part-of-speech
/// tags (`<S>`/`<E>` sentinels included) been observed as (part of) a
/// TRAIN human sentence's skeleton.
///
/// Same two-representation shape as [`BigramTable`]: see its own docs.
#[derive(Debug, Clone)]
pub struct SkeletonSet {
    repr: SkeletonRepr,
}

#[derive(Debug, Clone)]
enum SkeletonRepr {
    Owned {
        tag_vocab: TokenVocab,
        tag5: BTreeSet<u64>,
        tag4: BTreeSet<u64>,
    },
    View {
        tag_vocab: VocabView,
        /// Strictly-ascending packed 5-gram codes.
        tag5: &'static [u8],
        /// Strictly-ascending packed 4-gram codes.
        tag4: &'static [u8],
    },
}

impl SkeletonSet {
    /// This set's tag-vocabulary id for `tag`, whichever representation
    /// holds it.
    fn tag_id_of(&self, tag: &str) -> Option<u32> {
        match &self.repr {
            SkeletonRepr::Owned { tag_vocab, .. } => tag_vocab.id_of(tag),
            SkeletonRepr::View { tag_vocab, .. } => tag_vocab.id_of(tag),
        }
    }

    /// Whether `code` is an attested 5-gram window.
    fn tag5_contains(&self, code: u64) -> bool {
        match &self.repr {
            SkeletonRepr::Owned { tag5, .. } => tag5.contains(&code),
            SkeletonRepr::View { tag5, .. } => sorted_u64_contains(tag5, code),
        }
    }

    /// Whether `code` is an attested 4-gram window.
    fn tag4_contains(&self, code: u64) -> bool {
        match &self.repr {
            SkeletonRepr::Owned { tag4, .. } => tag4.contains(&code),
            SkeletonRepr::View { tag4, .. } => sorted_u64_contains(tag4, code),
        }
    }

    /// Packs a window of tag text into this set's base-`vocab_len` `u64`
    /// encoding, or `None` if any tag in `window` is outside this set's
    /// own tag vocabulary (an out-of-vocabulary tag can never have been
    /// attested).
    fn pack_window(&self, window: &[&str]) -> Option<u64> {
        let base = u64::try_from(self.tag_vocab_len()).ok()?;
        let mut value: u64 = 0;
        for &tag in window {
            let id = self.tag_id_of(tag)?;
            value = value.checked_mul(base)?.checked_add(u64::from(id))?;
        }
        Some(value)
    }

    /// `true` if EVERY 5-gram tag window (falling back to that window's
    /// own 4-gram prefix when the window doesn't clear a boundary-safe
    /// 5-gram check — see below) starting in `[lo.saturating_sub(3),
    /// (tags.len().saturating_sub(4)).min(hi.saturating_add(3)))` was
    /// attested in the TRAIN human corpus — a universal, not existential,
    /// quantifier: the loop rejects on the *first* unattested window in
    /// range rather than accepting on the first attested one. The upper
    /// bound is `min(tags.len() - 4, hi + 3)`, not merely `hi + 3`
    /// clamped to the sequence end: capping the range so every checked
    /// start index has a full 4-gram available avoids a spurious reject
    /// from an incomplete window right at the sequence's own tail. The
    /// loop always checks at least one window at `lo.saturating_sub(3)`
    /// even when that bound computation would otherwise produce an empty
    /// range.
    ///
    /// `tags` is the caller's full wrapped tag sequence, typically the
    /// sentence's coarse tags wrapped in the `<S>`/`<E>` sentinels;
    /// `lo`/`hi` are token indices into that same sequence bounding the
    /// candidate span under consideration. A tag in `tags` outside this
    /// set's own vocabulary makes every window it appears in
    /// unattestable (never a match), matching the bigram table's own
    /// out-of-vocabulary-is-always-a-miss convention.
    #[must_use]
    pub fn window_attested(&self, tags: &[&str], lo: usize, hi: usize) -> bool {
        let n = tags.len();
        if n == 0 {
            return false;
        }
        let start = lo.saturating_sub(3);
        let bound = n.saturating_sub(4).min(hi.saturating_add(3));
        let end = bound.max(start + 1);

        for i in start..end {
            if i >= n {
                break;
            }
            let five_attested = i + 5 <= n
                && self
                    .pack_window(&tags[i..i + 5])
                    .is_some_and(|code| self.tag5_contains(code));
            if five_attested {
                continue;
            }
            let four_attested = i + 4 <= n
                && self
                    .pack_window(&tags[i..i + 4])
                    .is_some_and(|code| self.tag4_contains(code));
            if four_attested {
                continue;
            }
            return false;
        }
        true
    }

    /// The number of distinct attested 5-gram tag windows.
    ///
    /// Parity-test introspection only; no runtime consumer.
    #[must_use]
    pub fn tag5_len(&self) -> usize {
        match &self.repr {
            SkeletonRepr::Owned { tag5, .. } => tag5.len(),
            SkeletonRepr::View { tag5, .. } => tag5.len() / 8,
        }
    }

    /// The number of distinct attested 4-gram tag windows.
    ///
    /// Parity-test introspection only; no runtime consumer.
    #[must_use]
    pub fn tag4_len(&self) -> usize {
        match &self.repr {
            SkeletonRepr::Owned { tag4, .. } => tag4.len(),
            SkeletonRepr::View { tag4, .. } => tag4.len() / 8,
        }
    }

    /// This set's coarse-tag vocabulary size (including the `<S>`/`<E>`
    /// sentinels).
    ///
    /// Parity-test introspection only; no runtime consumer.
    #[must_use]
    pub const fn tag_vocab_len(&self) -> usize {
        match &self.repr {
            SkeletonRepr::Owned { tag_vocab, .. } => tag_vocab.len(),
            SkeletonRepr::View { tag_vocab, .. } => tag_vocab.len(),
        }
    }

    /// The tag text for `id`, or `None` if `id` is not a valid id in this
    /// set's own tag vocabulary. Exposed only for tests/debugging that
    /// need to render a packed code back to readable tags.
    #[cfg(test)]
    fn tag_of(&self, id: u32) -> Option<&str> {
        match &self.repr {
            SkeletonRepr::Owned { tag_vocab, .. } => tag_vocab.token_of(id),
            SkeletonRepr::View { tag_vocab, .. } => tag_vocab.token_of(id),
        }
    }
}

/// The near-no-op pivot-rate calibration, once measured against a real
/// engine.
///
/// Filled in by a stage-2 calibration pass, not by stage-1
/// `corpus-tool attest`; `None` until then — a pack with no `[near_noop]`
/// table parses successfully with this field absent, exactly matching
/// stage 1's own documented output shape.
#[derive(Debug, Clone, PartialEq)]
pub struct NearNoOpCalibration {
    pub threshold_per_1000_words: f64,
    pub sample_doc_count: u32,
    pub dev_check_max_per_1000_words: Option<f64>,
    pub method: Box<str>,
}

/// A parsed `attestation-v1` pack: [`BigramTable`] plus [`SkeletonSet`],
/// and an optional [`NearNoOpCalibration`] once stage 2 has filled it in.
#[derive(Debug, Clone)]
pub struct AttestationPack {
    bigram: BigramTable,
    skeleton: SkeletonSet,
    near_noop: Option<NearNoOpCalibration>,
}

impl AttestationPack {
    /// This pack's bigram membership table.
    #[must_use]
    pub const fn bigram(&self) -> &BigramTable {
        &self.bigram
    }

    /// This pack's POS-skeleton n-gram set.
    #[must_use]
    pub const fn skeleton(&self) -> &SkeletonSet {
        &self.skeleton
    }

    /// Attaches `pooled` to this pack's bigram table, widening
    /// [`BigramTable::attests`] to fall back to the pooled filter on an
    /// exact-table miss — see this module's own "Widened evidence" docs.
    ///
    /// Consuming builder style (`self -> Self`), not `&mut self`: an
    /// `AttestationPack` is otherwise built once and never mutated in
    /// place (both [`Self::parse`] and [`Self::from_bin`] hand back a
    /// finished value), so this reads at the one real call site
    /// ([`crate::registry::ATTESTATION`]) as "the freshly-loaded pack,
    /// with pooled evidence attached" rather than a separate mutation
    /// step a caller could forget or reorder.
    #[must_use]
    pub const fn with_pooled(mut self, pooled: &'static PooledAttestPack) -> Self {
        self.bigram.pooled = Some(pooled);
        self
    }

    /// This pack's near-no-op pivot-rate calibration, if stage 2 has run.
    #[must_use]
    pub const fn near_noop(&self) -> Option<&NearNoOpCalibration> {
        self.near_noop.as_ref()
    }

    /// Parses an `attestation-v1`-shaped pack: a `[vocab]` table, a
    /// `[bigram]` table (`lefts`/`rights` comma/semicolon-joined id
    /// lists), a `[skeleton]` table (`tags` array plus `tag5`/`tag4`
    /// comma-joined packed-code lists), and an optional `[near_noop]`
    /// table.
    ///
    /// # Errors
    /// Returns [`PackError::Toml`] if `toml` is not valid TOML in the
    /// expected shape. Returns [`PackError::AttestationMalformedId`] if a
    /// `[bigram]` id list contains a value that doesn't parse as `u32`,
    /// [`PackError::AttestationIdOutOfRange`] if a `[bigram]` id
    /// references a token past the end of `[vocab].tokens`,
    /// [`PackError::AttestationBigramShapeMismatch`] if `lefts` and
    /// `rights` don't name the same number of groups, and
    /// [`PackError::AttestationMalformedSkeletonCode`] if a
    /// `[skeleton].tag5`/`tag4` entry doesn't parse as `u64`.
    pub fn parse(toml: &str) -> Result<Self, PackError> {
        let raw: RawPack = toml::from_str(toml).map_err(PackError::from)?;

        let vocab = TokenVocab::from_tokens(raw.vocab.tokens);
        let bigram = parse_bigram(&raw.bigram, vocab)?;

        let tag_vocab = TokenVocab::from_tokens(raw.skeleton.tags);
        let tag5 = parse_skeleton_codes(&raw.skeleton.tag5)?;
        let tag4 = parse_skeleton_codes(&raw.skeleton.tag4)?;
        let skeleton = SkeletonSet {
            repr: SkeletonRepr::Owned {
                tag_vocab,
                tag5,
                tag4,
            },
        };

        let near_noop = raw.near_noop.map(|raw| NearNoOpCalibration {
            threshold_per_1000_words: raw.threshold_per_1000_words,
            sample_doc_count: raw.sample_doc_count,
            dev_check_max_per_1000_words: raw.dev_check_max_per_1000_words,
            method: raw.method.into_boxed_str(),
        });

        Ok(Self {
            bigram,
            skeleton,
            near_noop,
        })
    }

    /// Loads a [`pack_attestation_bin`]-written derived artifact as a
    /// zero-copy view (header/section validation plus slice splits, no
    /// per-entry materialization) returning the pack alongside the
    /// source TOML's sha256 hex recorded in the artifact header.
    ///
    /// `bytes` is `&'static` because the view borrows it for the life of
    /// the process — the runtime caller is [`crate::ATTESTATION`]'s
    /// `include_bytes!`'d static.
    ///
    /// # Errors
    /// Returns [`PackError::AttestationBinMalformed`] for any structural
    /// inconsistency (bad magic/version, a section running past the end,
    /// a lookup table or code array out of order). Never expected for
    /// the vendored embedded artifact: covered by this module's
    /// round-trip tests and the registry's drift test.
    pub fn from_bin(bytes: &'static [u8]) -> Result<(Self, &'static str), PackError> {
        let mut cursor = BinCursor::new(bytes);
        if cursor.take(8)? != BIN_MAGIC {
            return Err(PackError::AttestationBinMalformed("unrecognized magic"));
        }
        let version = u16::from_le_bytes(cursor.take(2)?.try_into().expect("checked 2-byte slice"));
        if version != BIN_FORMAT_VERSION {
            return Err(PackError::AttestationBinMalformed(
                "unsupported format version",
            ));
        }
        let sha_hex = std::str::from_utf8(cursor.take(SHA256_HEX_LEN)?)
            .map_err(|_| PackError::AttestationBinMalformed("sha256 is not valid UTF-8"))?;

        let near_noop = read_near_noop(&mut cursor)?;

        let vocab = read_vocab(&mut cursor)?;
        let distinct_lefts = cursor.read_u32()?;
        let edge_count = cursor.read_u32()? as usize;
        let edges = cursor.take(edge_count * 8)?;
        if validate_edges(edges)? != distinct_lefts {
            return Err(PackError::AttestationBinMalformed(
                "recorded distinct-left count disagrees with the edge array",
            ));
        }

        let tag_vocab = read_vocab(&mut cursor)?;
        let tag5_count = cursor.read_u32()? as usize;
        let tag5 = cursor.take(tag5_count * 8)?;
        validate_ascending_u64(tag5, "tag5 codes are not strictly ascending")?;
        let tag4_count = cursor.read_u32()? as usize;
        let tag4 = cursor.take(tag4_count * 8)?;
        validate_ascending_u64(tag4, "tag4 codes are not strictly ascending")?;

        if !cursor.is_exhausted() {
            return Err(PackError::AttestationBinMalformed(
                "trailing bytes after the final section",
            ));
        }

        Ok((
            Self {
                bigram: BigramTable {
                    repr: BigramRepr::View {
                        vocab,
                        distinct_lefts,
                        edges,
                    },
                    pooled: None,
                },
                skeleton: SkeletonSet {
                    repr: SkeletonRepr::View {
                        tag_vocab,
                        tag5,
                        tag4,
                    },
                },
                near_noop,
            },
            sha_hex,
        ))
    }
}

/// Packs `toml_text` (an `attestation-v1` pack) into the binary layout
/// documented at the top of this module's artifact section — the build
/// side `corpus-tool attest-pack` runs offline.
///
/// A pure function of the TOML text: every section serializes a
/// `BTreeMap`/`BTreeSet` iteration, so packing the same text twice is
/// bit-identical (pinned by this module's tests).
///
/// # Errors
/// Returns [`PackError`] if `toml_text` is not a valid `attestation-v1`
/// pack — see [`AttestationPack::parse`]. Serialization itself cannot
/// fail.
pub fn pack_attestation_bin(toml_text: &str) -> Result<Vec<u8>, PackError> {
    let pack = AttestationPack::parse(toml_text)?;
    let sha_hex = Sha256::of_bytes(toml_text.as_bytes()).to_string();

    let mut out = Vec::new();
    out.extend_from_slice(&BIN_MAGIC);
    out.extend_from_slice(&BIN_FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(sha_hex.as_bytes());

    write_near_noop(&mut out, pack.near_noop());

    let BigramRepr::Owned { vocab, edges } = &pack.bigram.repr else {
        unreachable!("parse always builds the owned representation");
    };
    write_vocab(&mut out, vocab);
    write_u32(&mut out, u32_from(edges.len()));
    let edge_count: usize = edges.values().map(BTreeSet::len).sum();
    write_u32(&mut out, u32_from(edge_count));
    for (&left, rights) in edges {
        for &right in rights {
            out.extend_from_slice(&pack_edge(left, right).to_le_bytes());
        }
    }

    let SkeletonRepr::Owned {
        tag_vocab,
        tag5,
        tag4,
    } = &pack.skeleton.repr
    else {
        unreachable!("parse always builds the owned representation");
    };
    write_vocab(&mut out, tag_vocab);
    for set in [tag5, tag4] {
        write_u32(&mut out, u32_from(set.len()));
        for &code in set {
            out.extend_from_slice(&code.to_le_bytes());
        }
    }

    Ok(out)
}

/// Serializes one vocab section: see the artifact layout docs above.
fn write_vocab(out: &mut Vec<u8>, vocab: &TokenVocab) {
    write_u32(out, u32_from(vocab.tokens.len()));
    let mut pool: Vec<u8> = Vec::new();
    let mut offsets: Vec<u32> = Vec::with_capacity(vocab.tokens.len() + 1);
    offsets.push(0);
    for token in &vocab.tokens {
        pool.extend_from_slice(token.as_bytes());
        offsets.push(u32_from(pool.len()));
    }
    for offset in &offsets {
        write_u32(out, *offset);
    }
    write_u32(out, u32_from(pool.len()));
    out.extend_from_slice(&pool);
    // `by_text` IS the first-id-wins duplicate collapse in byte-string
    // order — `TokenVocab::from_tokens` built it with
    // `entry().or_insert(first)` — so its values, in iteration order,
    // are exactly the sorted lookup table `VocabView::id_of` needs.
    write_u32(out, u32_from(vocab.by_text.len()));
    for &id in vocab.by_text.values() {
        write_u32(out, id);
    }
}

/// Serializes the near-no-op calibration section.
fn write_near_noop(out: &mut Vec<u8>, near_noop: Option<&NearNoOpCalibration>) {
    let Some(calibration) = near_noop else {
        out.push(0);
        return;
    };
    out.push(1);
    out.extend_from_slice(&calibration.threshold_per_1000_words.to_le_bytes());
    write_u32(out, calibration.sample_doc_count);
    match calibration.dev_check_max_per_1000_words {
        None => out.push(0),
        Some(value) => {
            out.push(1);
            out.extend_from_slice(&value.to_le_bytes());
        }
    }
    write_u32(out, u32_from(calibration.method.len()));
    out.extend_from_slice(calibration.method.as_bytes());
}

/// Appends `value` little-endian.
fn write_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

/// `usize -> u32` with the one honest justification every count in this
/// artifact shares.
fn u32_from(value: usize) -> u32 {
    u32::try_from(value).expect("an attestation pack's counts stay far below u32::MAX")
}

/// Reads the near-no-op calibration section.
fn read_near_noop(cursor: &mut BinCursor) -> Result<Option<NearNoOpCalibration>, PackError> {
    match cursor.read_u8()? {
        0 => Ok(None),
        1 => {
            let threshold_per_1000_words = cursor.read_f64()?;
            let sample_doc_count = cursor.read_u32()?;
            let dev_check_max_per_1000_words = match cursor.read_u8()? {
                0 => None,
                1 => Some(cursor.read_f64()?),
                _ => {
                    return Err(PackError::AttestationBinMalformed(
                        "near-no-op dev-check flag is not 0 or 1",
                    ));
                }
            };
            let method_len = cursor.read_u32()? as usize;
            let method = std::str::from_utf8(cursor.take(method_len)?)
                .map_err(|_| PackError::AttestationBinMalformed("method is not valid UTF-8"))?;
            Ok(Some(NearNoOpCalibration {
                threshold_per_1000_words,
                sample_doc_count,
                dev_check_max_per_1000_words,
                method: Box::from(method),
            }))
        }
        _ => Err(PackError::AttestationBinMalformed(
            "near-no-op flag is not 0 or 1",
        )),
    }
}

/// Reads and structurally validates one vocab section: offsets monotone
/// and spanning the pool exactly, every lookup id in range, lookup texts
/// strictly ascending (what makes [`VocabView::id_of`]'s binary search
/// correct).
fn read_vocab(cursor: &mut BinCursor) -> Result<VocabView, PackError> {
    let count = cursor.read_u32()?;
    let offsets = cursor.take((count as usize + 1) * 4)?;
    let pool_len = cursor.read_u32()? as usize;
    let pool = cursor.take(pool_len)?;

    let mut prev: u32 = 0;
    for i in 0..=count as usize {
        let offset = u32_at(offsets, i);
        if offset < prev || offset as usize > pool_len {
            return Err(PackError::AttestationBinMalformed(
                "vocab offsets are not monotone within the pool",
            ));
        }
        prev = offset;
    }
    if u32_at(offsets, 0) != 0 || u32_at(offsets, count as usize) as usize != pool_len {
        return Err(PackError::AttestationBinMalformed(
            "vocab offsets do not span the pool",
        ));
    }

    let sorted_count = cursor.read_u32()?;
    if sorted_count > count {
        return Err(PackError::AttestationBinMalformed(
            "vocab lookup table is larger than the vocab",
        ));
    }
    let sorted_ids = cursor.take(sorted_count as usize * 4)?;
    let view = VocabView {
        count,
        offsets,
        pool,
        sorted_count,
        sorted_ids,
    };
    let mut prev_token: Option<&[u8]> = None;
    for i in 0..sorted_count as usize {
        let id = u32_at(sorted_ids, i);
        if id >= count {
            return Err(PackError::AttestationBinMalformed(
                "vocab lookup id out of range",
            ));
        }
        let token = view.token_bytes(id);
        if prev_token.is_some_and(|prev| prev >= token) {
            return Err(PackError::AttestationBinMalformed(
                "vocab lookup table is not strictly ascending",
            ));
        }
        prev_token = Some(token);
    }
    Ok(view)
}

/// Validates the packed edge array is strictly ascending, returning its
/// distinct-left count for the header cross-check.
fn validate_edges(edges: &[u8]) -> Result<u32, PackError> {
    let n = edges.len() / 8;
    let mut prev: Option<u64> = None;
    let mut distinct_lefts: u32 = 0;
    for i in 0..n {
        let value = u64_at(edges, i);
        if prev.is_some_and(|p| p >= value) {
            return Err(PackError::AttestationBinMalformed(
                "bigram edges are not strictly ascending",
            ));
        }
        if prev.is_none_or(|p| (p >> 32) != (value >> 32)) {
            distinct_lefts += 1;
        }
        prev = Some(value);
    }
    Ok(distinct_lefts)
}

/// Validates a packed skeleton-code array is strictly ascending.
fn validate_ascending_u64(region: &[u8], message: &'static str) -> Result<(), PackError> {
    let n = region.len() / 8;
    let mut prev: Option<u64> = None;
    for i in 0..n {
        let value = u64_at(region, i);
        if prev.is_some_and(|p| p >= value) {
            return Err(PackError::AttestationBinMalformed(message));
        }
        prev = Some(value);
    }
    Ok(())
}

fn parse_bigram(raw: &RawBigram, vocab: TokenVocab) -> Result<BigramTable, PackError> {
    let lefts = parse_csv_u32("bigram.lefts", &raw.lefts)?;
    let right_groups = parse_bigram_rights(&raw.rights)?;

    if lefts.len() != right_groups.len() {
        return Err(PackError::AttestationBigramShapeMismatch {
            lefts: lefts.len(),
            rights: right_groups.len(),
        });
    }

    let mut edges: BTreeMap<u32, BTreeSet<u32>> = BTreeMap::new();
    for (left_id, rights) in lefts.into_iter().zip(right_groups) {
        if left_id as usize >= vocab.len() {
            return Err(PackError::AttestationIdOutOfRange {
                field: "bigram.lefts",
                id: left_id,
                vocab_len: vocab.len(),
            });
        }
        for &right_id in &rights {
            if right_id as usize >= vocab.len() {
                return Err(PackError::AttestationIdOutOfRange {
                    field: "bigram.rights",
                    id: right_id,
                    vocab_len: vocab.len(),
                });
            }
        }
        edges.entry(left_id).or_default().extend(rights);
    }

    Ok(BigramTable {
        repr: BigramRepr::Owned { vocab, edges },
        pooled: None,
    })
}

/// Parses `rights` (the `;`-delimited, per-left comma-joined id groups)
/// into one `Vec<u32>` per group. An entirely empty field (both `lefts`
/// and `rights` empty: a degenerate, zero-bigram pack) parses to zero
/// groups rather than one spurious empty group.
fn parse_bigram_rights(rights: &str) -> Result<Vec<Vec<u32>>, PackError> {
    let trimmed = rights.trim();
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }
    trimmed
        .split(';')
        .map(|group| parse_csv_u32("bigram.rights", group))
        .collect()
}

/// Parses a comma-separated `u32` list. An empty (post-trim) string
/// parses to an empty list rather than a single malformed empty field —
/// mirrors `friction_packs::dms`'s own `parse_ids_csv` convention.
fn parse_csv_u32(field: &'static str, value: &str) -> Result<Vec<u32>, PackError> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }
    trimmed
        .split(',')
        .map(|part| {
            part.trim()
                .parse::<u32>()
                .map_err(|_| PackError::AttestationMalformedId {
                    field,
                    value: part.to_string(),
                })
        })
        .collect()
}

/// Parses a comma-separated `u64` packed-skeleton-code list into a set.
fn parse_skeleton_codes(value: &str) -> Result<BTreeSet<u64>, PackError> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Ok(BTreeSet::new());
    }
    trimmed
        .split(',')
        .map(|part| {
            part.trim()
                .parse::<u64>()
                .map_err(|_| PackError::AttestationMalformedSkeletonCode {
                    value: part.to_string(),
                })
        })
        .collect()
}

#[derive(Debug, Deserialize)]
struct RawVocab {
    tokens: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct RawBigram {
    lefts: String,
    rights: String,
}

#[derive(Debug, Deserialize)]
struct RawSkeleton {
    tags: Vec<String>,
    tag5: String,
    tag4: String,
}

#[derive(Debug, Deserialize)]
struct RawNearNoop {
    threshold_per_1000_words: f64,
    sample_doc_count: u32,
    #[serde(default)]
    dev_check_max_per_1000_words: Option<f64>,
    method: String,
}

/// The TOML shape of an `attestation-v1` pack as a whole. `[pack]`'s
/// metadata fields (`version`, `corpus_manifest_sha256`,
/// `train_human_doc_count`) are not named here (nothing in this module
/// needs them to build a working [`AttestationPack`]) and are simply
/// ignored by serde rather than rejected.
#[derive(Debug, Deserialize)]
struct RawPack {
    vocab: RawVocab,
    bigram: RawBigram,
    skeleton: RawSkeleton,
    #[serde(default)]
    near_noop: Option<RawNearNoop>,
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_PACK: &str = r#"
        [pack]
        version = "attestation-v1"
        corpus_manifest_sha256 = "deadbeef"
        train_human_doc_count = 2

        [vocab]
        tokens = ["<s>", ".", "fox", "quick", "the"]

        [bigram]
        # "<s>" -> "the" (id 0 -> id 4), "the" -> "quick" (4 -> 3),
        # "quick" -> "fox" (3 -> 2), "fox" -> "." (2 -> 1)
        lefts = "0,2,3,4"
        rights = "4;1;2;3"

        [skeleton]
        tags = ["<S>", "<E>", ".", "DT", "JJ", "NN"]
        # <S> DT JJ NN . <E> packed base-6:
        # ids: <S>=0 DT=3 JJ=4 NN=5 .=2 <E>=1
        # window1 = [0,3,4,5,2] -> 0*6^4+3*6^3+4*6^2+5*6+2 = 0+648+144+30+2=824
        # window2 = [3,4,5,2,1] -> 3*6^4+4*6^3+5*6^2+2*6+1 = 3888+864+180+12+1=4945
        tag5 = "824,4945"
        # 4-gram prefixes for the same two windows:
        # [0,3,4,5] -> 0*216+3*36+4*6+5 = 108+24+5=137
        # [3,4,5,2] -> 3*216+4*36+5*6+2 = 648+144+30+2=824
        tag4 = "137,824"
    "#;

    #[test]
    fn parse_builds_a_working_pack_with_no_near_noop_table() {
        let pack = AttestationPack::parse(SAMPLE_PACK).expect("sample pack parses");
        assert!(pack.near_noop().is_none());
        assert_eq!(pack.bigram().vocab_len(), 5);
        assert_eq!(pack.skeleton().tag_vocab_len(), 6);
    }

    #[test]
    fn bigram_attests_reports_exact_membership() {
        let pack = AttestationPack::parse(SAMPLE_PACK).expect("sample pack parses");
        assert!(pack.bigram().attests("<s>", "the"));
        assert!(pack.bigram().attests("the", "quick"));
        assert!(pack.bigram().attests("quick", "fox"));
        assert!(pack.bigram().attests("fox", "."));
        assert!(!pack.bigram().attests("<s>", "fox"), "never attested");
        assert!(
            !pack.bigram().attests("the", "banana"),
            "out-of-vocab right is always a miss"
        );
        assert!(
            !pack.bigram().attests("banana", "the"),
            "out-of-vocab left is always a miss"
        );
    }

    #[test]
    fn bigram_counts_are_exact() {
        let pack = AttestationPack::parse(SAMPLE_PACK).expect("sample pack parses");
        assert_eq!(pack.bigram().distinct_lefts(), 4);
        assert_eq!(pack.bigram().edge_count(), 4);
    }

    #[test]
    fn skeleton_window_attested_finds_the_exact_5gram() {
        let pack = AttestationPack::parse(SAMPLE_PACK).expect("sample pack parses");
        let tags = ["<S>", "DT", "JJ", "NN", ".", "<E>"];
        // The 5-gram at start index 0 (<S> DT JJ NN .) is attested.
        assert!(pack.skeleton().window_attested(&tags, 0, 0));
        // The 5-gram at start index 1 (DT JJ NN . <E>) is also attested.
        assert!(pack.skeleton().window_attested(&tags, 1, 1));
    }

    #[test]
    fn skeleton_window_attested_falls_back_to_4gram_prefix_near_a_boundary() {
        let pack = AttestationPack::parse(SAMPLE_PACK).expect("sample pack parses");
        // A 4-token tail with no room for a full 5-gram at its own start
        // index: only the tag4 set can attest it.
        let tags = ["<S>", "DT", "JJ", "NN"];
        assert!(pack.skeleton().window_attested(&tags, 0, 0));
    }

    #[test]
    fn skeleton_window_attested_rejects_an_unseen_run() {
        let pack = AttestationPack::parse(SAMPLE_PACK).expect("sample pack parses");
        let tags = ["<S>", "NN", "DT", "JJ", ".", "<E>"];
        assert!(!pack.skeleton().window_attested(&tags, 0, 0));
    }

    #[test]
    fn skeleton_window_attested_handles_an_out_of_vocabulary_tag() {
        let pack = AttestationPack::parse(SAMPLE_PACK).expect("sample pack parses");
        let tags = ["<S>", "ZZ", "JJ", "NN", ".", "<E>"];
        assert!(!pack.skeleton().window_attested(&tags, 0, 0));
    }

    #[test]
    fn skeleton_counts_and_tag_lookup_are_exact() {
        let pack = AttestationPack::parse(SAMPLE_PACK).expect("sample pack parses");
        assert_eq!(pack.skeleton().tag5_len(), 2);
        assert_eq!(pack.skeleton().tag4_len(), 2);
        assert_eq!(pack.skeleton().tag_of(0), Some("<S>"));
        assert_eq!(pack.skeleton().tag_of(99), None);
    }

    /// Float fields round-trip through the same deterministic TOML/Rust
    /// decimal-literal parser on both sides, so a direct equality
    /// comparison is safe here — `#[allow(clippy::float_cmp)]` documents
    /// that this is a deliberate exact-round-trip check, not a
    /// numerically-computed comparison that should use a tolerance.
    #[test]
    #[allow(clippy::float_cmp)]
    fn parse_reads_a_present_near_noop_table() {
        let with_calibration = format!(
            "{SAMPLE_PACK}\n[near_noop]\nthreshold_per_1000_words = 1.5\n\
             sample_doc_count = 264\ndev_check_max_per_1000_words = 1.2\n\
             method = \"train-max-times-1.15\"\n"
        );
        let pack = AttestationPack::parse(&with_calibration).expect("pack with calibration parses");
        let calibration = pack.near_noop().expect("near_noop present");
        assert_eq!(calibration.threshold_per_1000_words, 1.5);
        assert_eq!(calibration.sample_doc_count, 264);
        assert_eq!(calibration.dev_check_max_per_1000_words, Some(1.2));
        assert_eq!(&*calibration.method, "train-max-times-1.15");
    }

    #[test]
    fn parse_reads_a_present_near_noop_table_with_no_dev_check() {
        let with_calibration = format!(
            "{SAMPLE_PACK}\n[near_noop]\nthreshold_per_1000_words = 1.5\n\
             sample_doc_count = 264\nmethod = \"train-max-times-1.15\"\n"
        );
        let pack = AttestationPack::parse(&with_calibration).expect("pack with calibration parses");
        let calibration = pack.near_noop().expect("near_noop present");
        assert_eq!(calibration.dev_check_max_per_1000_words, None);
    }

    #[test]
    fn parse_rejects_malformed_toml() {
        assert!(AttestationPack::parse("not [ valid toml").is_err());
    }

    #[test]
    fn parse_rejects_non_numeric_bigram_id() {
        let bad = r#"
            [vocab]
            tokens = ["<s>", "the"]
            [bigram]
            lefts = "0,not-a-number"
            rights = "1;1"
            [skeleton]
            tags = ["<S>", "<E>"]
            tag5 = ""
            tag4 = ""
        "#;
        let err = AttestationPack::parse(bad).expect_err("non-numeric bigram id must be rejected");
        assert!(matches!(err, PackError::AttestationMalformedId { .. }));
    }

    #[test]
    fn parse_rejects_bigram_id_out_of_range_for_vocab() {
        let bad = r#"
            [vocab]
            tokens = ["<s>", "the"]
            [bigram]
            lefts = "0"
            rights = "99"
            [skeleton]
            tags = ["<S>", "<E>"]
            tag5 = ""
            tag4 = ""
        "#;
        let err = AttestationPack::parse(bad).expect_err("out-of-range bigram id must be rejected");
        assert!(matches!(err, PackError::AttestationIdOutOfRange { .. }));
    }

    #[test]
    fn parse_rejects_mismatched_lefts_and_rights_group_counts() {
        let bad = r#"
            [vocab]
            tokens = ["<s>", "the", "fox"]
            [bigram]
            lefts = "0,1"
            rights = "1"
            [skeleton]
            tags = ["<S>", "<E>"]
            tag5 = ""
            tag4 = ""
        "#;
        let err = AttestationPack::parse(bad)
            .expect_err("mismatched lefts/rights groups must be rejected");
        assert!(matches!(
            err,
            PackError::AttestationBigramShapeMismatch { .. }
        ));
    }

    #[test]
    fn parse_rejects_non_numeric_skeleton_code() {
        let bad = r#"
            [vocab]
            tokens = ["<s>"]
            [bigram]
            lefts = ""
            rights = ""
            [skeleton]
            tags = ["<S>", "<E>"]
            tag5 = "not-a-number"
            tag4 = ""
        "#;
        let err =
            AttestationPack::parse(bad).expect_err("non-numeric skeleton code must be rejected");
        assert!(matches!(
            err,
            PackError::AttestationMalformedSkeletonCode { .. }
        ));
    }

    #[test]
    fn parse_handles_an_entirely_empty_bigram_table_gracefully() {
        let pack = r#"
            [vocab]
            tokens = ["<s>"]
            [bigram]
            lefts = ""
            rights = ""
            [skeleton]
            tags = ["<S>", "<E>"]
            tag5 = ""
            tag4 = ""
        "#;
        let parsed = AttestationPack::parse(pack).expect("empty bigram/skeleton table parses");
        assert_eq!(parsed.bigram().edge_count(), 0);
        assert_eq!(parsed.skeleton().tag5_len(), 0);
        assert!(!parsed.bigram().attests("<s>", "<s>"));
    }

    #[test]
    fn parsing_the_same_bytes_twice_is_deterministic() {
        let a = AttestationPack::parse(SAMPLE_PACK).expect("first parse");
        let b = AttestationPack::parse(SAMPLE_PACK).expect("second parse");
        assert_eq!(a.bigram().edge_count(), b.bigram().edge_count());
        assert_eq!(a.skeleton().tag5_len(), b.skeleton().tag5_len());
        assert!(a.bigram().attests("<s>", "the"));
        assert!(b.bigram().attests("<s>", "the"));
    }

    /// Leaks a packed artifact for the `&'static` lifetime
    /// [`AttestationPack::from_bin`] requires — test-only, mirroring the
    /// runtime's `include_bytes!` static.
    fn leak_bin(toml: &str) -> &'static [u8] {
        Box::leak(
            pack_attestation_bin(toml)
                .expect("test pack packs")
                .into_boxed_slice(),
        )
    }

    #[test]
    fn packing_the_same_toml_twice_is_bit_identical() {
        assert_eq!(
            pack_attestation_bin(SAMPLE_PACK).expect("sample pack packs"),
            pack_attestation_bin(SAMPLE_PACK).expect("sample pack packs"),
        );
    }

    #[test]
    fn bin_round_trip_answers_identically_to_the_owned_parse() {
        let owned = AttestationPack::parse(SAMPLE_PACK).expect("sample pack parses");
        let (view, sha_hex) = AttestationPack::from_bin(leak_bin(SAMPLE_PACK))
            .expect("freshly packed artifact parses");

        assert_eq!(
            sha_hex,
            Sha256::of_bytes(SAMPLE_PACK.as_bytes()).to_string()
        );
        assert_eq!(view.bigram().vocab_len(), owned.bigram().vocab_len());
        assert_eq!(view.bigram().edge_count(), owned.bigram().edge_count());
        assert_eq!(
            view.bigram().distinct_lefts(),
            owned.bigram().distinct_lefts()
        );
        assert_eq!(view.skeleton().tag5_len(), owned.skeleton().tag5_len());
        assert_eq!(view.skeleton().tag4_len(), owned.skeleton().tag4_len());
        assert_eq!(
            view.skeleton().tag_vocab_len(),
            owned.skeleton().tag_vocab_len()
        );
        assert_eq!(view.near_noop(), owned.near_noop());

        // Every in-vocab pair, both directions, plus out-of-vocab probes:
        // the view's binary searches and the owned B-trees must agree on
        // all of them.
        for left in ["<s>", ".", "fox", "quick", "the", "banana"] {
            for right in ["<s>", ".", "fox", "quick", "the", "banana"] {
                assert_eq!(
                    view.bigram().attests(left, right),
                    owned.bigram().attests(left, right),
                    "attests({left:?}, {right:?}) must agree"
                );
            }
        }

        let attested = ["<S>", "DT", "JJ", "NN", ".", "<E>"];
        let unattested = ["<S>", "NN", "DT", "JJ", ".", "<E>"];
        let out_of_vocab = ["<S>", "ZZ", "JJ", "NN", ".", "<E>"];
        for tags in [&attested, &unattested, &out_of_vocab] {
            for lo in 0..tags.len() {
                assert_eq!(
                    view.skeleton().window_attested(tags, lo, lo),
                    owned.skeleton().window_attested(tags, lo, lo),
                    "window_attested({tags:?}, {lo}) must agree"
                );
            }
        }
    }

    #[test]
    fn bin_round_trip_preserves_a_present_near_noop_table() {
        let with_calibration = format!(
            "{SAMPLE_PACK}\n[near_noop]\nthreshold_per_1000_words = 1.5\n\
             sample_doc_count = 264\ndev_check_max_per_1000_words = 1.2\n\
             method = \"train-max-times-1.15\"\n"
        );
        let owned = AttestationPack::parse(&with_calibration).expect("pack parses");
        let (view, _) = AttestationPack::from_bin(leak_bin(&with_calibration))
            .expect("freshly packed artifact parses");
        assert_eq!(view.near_noop(), owned.near_noop());
    }

    #[test]
    fn from_bin_rejects_a_truncated_artifact() {
        let bytes = pack_attestation_bin(SAMPLE_PACK).expect("sample pack packs");
        let truncated: &'static [u8] =
            Box::leak(bytes[..bytes.len() - 1].to_vec().into_boxed_slice());
        assert!(matches!(
            AttestationPack::from_bin(truncated),
            Err(PackError::AttestationBinMalformed(_))
        ));
    }

    #[test]
    fn from_bin_rejects_a_bad_magic() {
        let mut bytes = pack_attestation_bin(SAMPLE_PACK).expect("sample pack packs");
        bytes[0] ^= 0xFF;
        let corrupted: &'static [u8] = Box::leak(bytes.into_boxed_slice());
        assert!(matches!(
            AttestationPack::from_bin(corrupted),
            Err(PackError::AttestationBinMalformed("unrecognized magic"))
        ));
    }

    #[test]
    fn duplicate_left_ids_union_their_right_sets_rather_than_overwriting() {
        let pack = r#"
            [vocab]
            tokens = ["<s>", "a", "b", "c"]
            [bigram]
            lefts = "1,1"
            rights = "2;3"
            [skeleton]
            tags = ["<S>", "<E>"]
            tag5 = ""
            tag4 = ""
        "#;
        let parsed = AttestationPack::parse(pack).expect("duplicate-left pack parses");
        assert!(parsed.bigram().attests("a", "b"));
        assert!(parsed.bigram().attests("a", "c"));
        assert_eq!(parsed.bigram().edge_count(), 2);
    }

    /// Leaks a built [`crate::pooled_attest::PooledAttestPack`] for the
    /// `&'static` lifetime [`AttestationPack::with_pooled`] requires —
    /// test-only, same leak-for-`'static` convention [`leak_bin`] uses.
    fn leak_pooled(pairs: &[(&str, &str)]) -> &'static PooledAttestPack {
        use crate::pooled_attest::{build_pack_bytes, pair_key};
        let keys: BTreeSet<String> = pairs.iter().map(|(l, r)| pair_key(l, r)).collect();
        let built = build_pack_bytes(&keys).expect("small synthetic pooled pack builds");
        let bin: &'static [u8] = Box::leak(built.bytes.into_boxed_slice());
        let meta = format!(
            "[pack]\nversion = \"pooled-attest-v1\"\nkey_count = {}\n",
            built.key_count
        );
        Box::leak(Box::new(
            PooledAttestPack::load(bin, &meta).expect("synthetic pooled pack loads"),
        ))
    }

    /// The widening this whole module's "Widened evidence" docs describe:
    /// a pair absent from the exact TRAIN table is still `false` with no
    /// pooled filter attached, and becomes `true` once
    /// [`AttestationPack::with_pooled`] attaches a pooled filter that
    /// contains it — while a pair the exact table already knows is
    /// unaffected either way.
    #[test]
    fn with_pooled_widens_attests_on_an_exact_miss_only() {
        let pack = AttestationPack::parse(SAMPLE_PACK).expect("sample pack parses");
        assert!(
            !pack.bigram().attests("will", "upgrade"),
            "fixture assumption: the small sample table never attested this pair"
        );

        let pooled = leak_pooled(&[("will", "upgrade")]);
        let widened = pack.with_pooled(pooled);
        assert!(
            widened.bigram().attests("will", "upgrade"),
            "the pooled filter now attests the pair the exact table missed"
        );
        // An exact hit is untouched by attaching a pooled filter that
        // does NOT itself contain that pair.
        assert!(widened.bigram().attests("<s>", "the"));
        // A pair neither table has ever seen stays unattested.
        assert!(!widened.bigram().attests("never", "seen"));
    }

    /// `"<s>"` queries the pooled path identically to any other left
    /// token — no special-casing on either side (see
    /// `crate::pooled_attest`'s own module docs).
    #[test]
    fn with_pooled_widens_a_sentence_start_seam() {
        let pack = AttestationPack::parse(SAMPLE_PACK).expect("sample pack parses");
        assert!(
            !pack.bigram().attests("<s>", "banana"),
            "fixture assumption: never attested in the exact table"
        );
        let pooled = leak_pooled(&[("<s>", "banana")]);
        let widened = pack.with_pooled(pooled);
        assert!(widened.bigram().attests("<s>", "banana"));
    }
}
