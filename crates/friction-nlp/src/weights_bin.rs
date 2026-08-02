//! Shared on-disk header for the derived, `postcard`-encoded weight
//! artifacts (`weights/parser_en.bin`, `weights/perceptron_en.bin`).
//!
//! The `json.gz` files in this crate's `weights/` directory stay the
//! audited interchange artifact from the training pipeline (see
//! `weights/NOTICE.md`); the `.bin` files are a derived, faster-to-load
//! encoding of the exact same data, produced by `corpus-tool
//! weights-pack` (see [`crate::tag_perceptron::pack_perceptron_tagger_bin`]
//! / [`crate::dep_perceptron::pack_perceptron_parser_bin`]).
//!
//! Both files share one fixed-length header, built by [`write_header`] and
//! validated by [`split_header`]: an 8-byte ASCII magic (distinct per
//! artifact — [`crate::tag_perceptron`] and [`crate::dep_perceptron`] each
//! define their own, so one can never be silently loaded in place of the
//! other), a little-endian `u16` format version, and the 64-byte
//! lowercase-hex sha256 of the `json.gz` this `.bin` was derived from —
//! immediately followed by the payload (see below for what the payload
//! itself now contains).
//!
//! Recording the source sha256 lets a stale `.bin` (source `json.gz`
//! regenerated, `corpus-tool weights-pack` not re-run) fail loudly at
//! process init rather than silently drifting from the audited source:
//! each artifact's own `new()` compares [`split_header`]'s returned
//! sha256 against a compile-time constant of its own currently embedded
//! `json.gz`, not against a value computed at load time (hashing a
//! multi-megabyte file on every startup would undo the whole point of
//! shipping a faster-to-load artifact).
//!
//! # Format version 2: a serialized hash table, not a serialized `HashMap`
//!
//! Version 1's payload was a `postcard`-encoded
//! [`crate::tag_perceptron::WeightFile`]/[`crate::dep_perceptron::WeightFile`]:
//! loading it still meant building a real `std::collections::HashMap` from scratch:
//! hashing and inserting every one of tens to hundreds of thousands of feature-string
//! keys, each carrying its own heap allocation. That construction, not the postcard
//! decode itself, dominated startup cost.
//!
//! Version 2's payload instead serializes the hash table's own on-disk
//! representation, so loading it is a handful of slices over the
//! `include_bytes!`'d static: zero per-key work. This module provides the
//! shared primitive both [`crate::tag_perceptron`] and
//! [`crate::dep_perceptron`] build their own payload layout from (each
//! documents its exact byte layout in its own module docs):
//!
//! - a fixed-seed, open-addressed hash table ([`hash_table_capacity`],
//!   [`build_open_table`], [`HashView`]) mapping a byte-string key to one
//!   `u32` value, linear-probed;
//! - a contiguous string pool ([`push_key`]) every table's keys are
//!   appended to, referenced by `(offset, length)` rather than duplicated;
//! - a variable-length sparse value row ([`build_sparse_row`],
//!   [`sparse_row`]) for the feature -> per-class/per-action weight lists,
//!   since most features only carry a handful of nonzero weights out of
//!   the full class/action space (storing a full dense row per feature
//!   would multiply artifact size by roughly the class count).
//!
//! Every multi-byte field is read via `from_le_bytes` on a byte slice —
//! alignment-free by construction, so no `unsafe` casting is ever needed
//! (this workspace denies `unsafe_code` at the workspace level).
//!
//! Construction ([`build_open_table`]) is a pure function of `(entries,
//! their fixed feed order)`: same fixed [`HASH_SEED`], same linear-probe
//! order, same capacity formula. Every packer feeds entries in a
//! `BTreeMap`'s ascending-key order, so packing the same `WeightFile`
//! twice produces bit-identical `.bin` bytes — mirrored by each artifact
//! module's own `pack_*_bin_is_deterministic` test.

/// The on-disk format every header currently writes. Bumped when the
/// payload's own encoding scheme changes crate-wide (version 1: a
/// `postcard`-encoded `WeightFile`; version 2: the serialized hash-table
/// view this module now describes) — every artifact using this shared
/// header moves together, so unlike the per-artifact magic (which
/// distinguishes *which* artifact a `.bin` is, not what scheme its payload
/// uses), this field is shared.
const FORMAT_VERSION: u16 = 2;

/// A sha256 digest is recorded as lowercase hex (matching
/// `corpus_tool::hashing::sha256_hex`'s output shape), not raw digest
/// bytes, so a hex dump of the header stays human-legible.
const SHA256_HEX_LEN: usize = 64;

/// The fixed-length prefix every `.bin` starts with: 8-byte magic + 2-byte
/// little-endian version + 64-byte hex sha256.
///
/// `pub` (rather than `pub(crate)`, likewise for [`write_header`] and
/// [`split_header`] below): `weights_bin` is a private module, so nothing
/// here is reachable from outside the crate by path regardless — `pub`
/// just avoids clippy's `redundant_pub_crate` (an item can't be "more
/// public than necessary" when its only path to the outside is already
/// sealed one level up).
pub const HEADER_LEN: usize = 8 + 2 + SHA256_HEX_LEN;

/// Errors reading a derived binary weight artifact's header.
///
/// `pub` purely so [`PerceptronTagError`](crate::PerceptronTagError)/
/// [`PerceptronParseError`](crate::PerceptronParseError) — both `pub`
/// enums — can wrap it in a `#[from]` variant without tripping the
/// `private_interfaces` lint; see [`HEADER_LEN`]'s docs for why this
/// doesn't actually widen this module's external surface.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum WeightsBinError {
    /// Shorter than [`HEADER_LEN`]: not a header this module wrote.
    #[error("weight artifact truncated: {len} byte(s), need at least {min}")]
    Truncated { len: usize, min: usize },
    /// The first 8 bytes didn't match the caller's expected magic.
    #[error("weight artifact has an unrecognized magic")]
    BadMagic,
    /// The version field isn't [`FORMAT_VERSION`].
    #[error("weight artifact format version {found} is unsupported (expected {expected})")]
    VersionMismatch { found: u16, expected: u16 },
    /// The sha256 field isn't valid UTF-8 — never expected for a header
    /// this module itself wrote, only for corrupted or hand-edited input.
    #[error("weight artifact's recorded sha256 is not valid UTF-8")]
    BadSha256,
    /// A version-2 hash-table-view payload was structurally inconsistent
    /// (a length field pointing past the end of its own slice, a stored
    /// key length that doesn't fit, ...) — never expected for the
    /// vendored artifact (covered by this crate's own parity tests), only
    /// for corrupted or hand-edited input.
    #[error("weight artifact payload is malformed: {0}")]
    MalformedPayload(&'static str),
}

/// Builds the fixed-length header for `magic` (8 ASCII bytes, distinct per
/// artifact) and `source_sha256_hex` (the source `json.gz`'s sha256, as
/// lowercase hex — exactly [`SHA256_HEX_LEN`] characters).
///
/// # Panics
/// Panics if `source_sha256_hex` isn't exactly [`SHA256_HEX_LEN`] bytes:
/// a caller bug (every sha256 hex digest is exactly 64 characters), never
/// a runtime condition.
pub fn write_header(magic: [u8; 8], source_sha256_hex: &str) -> Vec<u8> {
    assert_eq!(
        source_sha256_hex.len(),
        SHA256_HEX_LEN,
        "a sha256 hex digest is always exactly {SHA256_HEX_LEN} characters"
    );
    let mut out = Vec::with_capacity(HEADER_LEN);
    out.extend_from_slice(&magic);
    out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(source_sha256_hex.as_bytes());
    out
}

/// Validates `bytes`'s header against `magic` and splits it into
/// `(source_sha256_hex, payload)`. Does not check `source_sha256_hex`
/// against any expected value: the caller compares it against its own
/// compile-time constant (see this module's docs for why).
///
/// # Errors
/// [`WeightsBinError::Truncated`], [`WeightsBinError::BadMagic`],
/// [`WeightsBinError::VersionMismatch`], or
/// [`WeightsBinError::BadSha256`]: see each variant's own docs.
pub fn split_header(bytes: &[u8], magic: [u8; 8]) -> Result<(&str, &[u8]), WeightsBinError> {
    if bytes.len() < HEADER_LEN {
        return Err(WeightsBinError::Truncated {
            len: bytes.len(),
            min: HEADER_LEN,
        });
    }
    if bytes[..8] != magic {
        return Err(WeightsBinError::BadMagic);
    }
    let version = u16::from_le_bytes(
        bytes[8..10]
            .try_into()
            .expect("bytes.len() >= HEADER_LEN guarantees 2 bytes at [8..10]"),
    );
    if version != FORMAT_VERSION {
        return Err(WeightsBinError::VersionMismatch {
            found: version,
            expected: FORMAT_VERSION,
        });
    }
    let sha_hex =
        std::str::from_utf8(&bytes[10..HEADER_LEN]).map_err(|_| WeightsBinError::BadSha256)?;
    Ok((sha_hex, &bytes[HEADER_LEN..]))
}

/// The fixed xxh64 seed every open-addressed hash-table key ([`HashView`]/
/// [`build_open_table`]) is hashed with, on both the build side
/// (`corpus-tool weights-pack`) and the query side (runtime lookups).
///
/// Fixed rather than std's randomized-per-process `SipHash` default, for
/// the same reason [`crate`]'s sibling crate `friction_packs::jargon_attest`
/// documents for its own filter: a randomized hash would make table
/// construction (and so the packed `.bin` bytes) vary run to run, and the
/// artifact would no longer be reproducible from its source `json.gz`.
/// `b"FRWEIGHT"` read as a big-endian `u64` — arbitrary, but fixed,
/// documented, and distinct from `jargon_attest::HASH_SEED` (the two
/// crates' hash tables never need to agree with each other).
pub const HASH_SEED: u64 = 0x4652_5745_4947_4854;

/// The `key_len` value every empty slot is initialized to — no real key
/// is ever this long, so it doubles as the "this slot is empty" sentinel
/// [`build_open_table`] writes and [`HashView::get`] checks for.
const EMPTY_KEY_LEN: u32 = u32::MAX;

/// Bytes per slot: `key_off: u32`, `key_len: u32`, `value: u32`, each
/// little-endian.
const SLOT_LEN: usize = 12;

/// The power-of-two slot count for an open-addressed table holding
/// `count` entries, sized so `count` never exceeds ~0.7 of the total slot
/// count (a fixed, deterministic integer computation — `count * 10 <=
/// capacity * 7` — rather than a floating-point load-factor division, so
/// the result can never differ across platforms).
///
/// Always at least `1`, so [`HashView::get`]'s `capacity - 1` probe mask
/// is always well-defined even for a table with zero entries.
#[must_use]
pub fn hash_table_capacity(count: usize) -> u32 {
    let mut capacity: u64 = 1;
    #[allow(clippy::cast_possible_truncation)] // guarded by the loop condition below
    let count = count as u64;
    while count * 10 > capacity * 7 {
        capacity *= 2;
    }
    u32::try_from(capacity)
        .expect("a weight artifact's key count is always far below u32::MAX slots")
}

/// Appends `key`'s UTF-8 bytes to `pool` and returns its `(offset, length)`
/// — the one step every open-table builder below uses to place a key into
/// a crate-wide contiguous string pool.
#[must_use]
pub fn push_key(pool: &mut Vec<u8>, key: &str) -> (u32, u32) {
    let offset = u32::try_from(pool.len())
        .expect("a derived weight artifact's string pool stays far under 4 GiB");
    pool.extend_from_slice(key.as_bytes());
    let len =
        u32::try_from(key.len()).expect("a single feature/word key stays far under 4 GiB long");
    (offset, len)
}

/// Builds an open-addressed hash table's slot bytes over `entries` —
/// `(key, key_offset, key_length, value)`, each `key_offset`/`key_length`
/// already resolved via [`push_key`] against the pool the caller will
/// embed alongside these slots.
///
/// `entries` must be fed in a fixed, deterministic order for the result to
/// be reproducible — every caller in this crate feeds a `BTreeMap`'s
/// ascending-key order. Ties (there are none: every entry's key is
/// distinct) aside, linear probing from each key's `hash & (capacity -
/// 1)` slot makes the final layout a pure function of `(entries, their
/// feed order)`.
///
/// # Panics
/// Panics if `entries` somehow overflows the table built for its own
/// length (never happens: [`hash_table_capacity`] always sizes the table
/// to keep the load factor under 1).
#[must_use]
pub fn build_open_table(entries: &[(&str, u32, u32, u32)]) -> (u32, Vec<u8>) {
    let capacity = hash_table_capacity(entries.len());
    let mut slots = vec![0xFFu8; capacity as usize * SLOT_LEN];
    let mask = u64::from(capacity - 1);
    for &(key, key_off, key_len, value) in entries {
        let hash = xxhash_rust::xxh64::xxh64(key.as_bytes(), HASH_SEED);
        #[allow(clippy::cast_possible_truncation)] // hash & mask fits capacity's own u32 range
        let mut idx = (hash & mask) as u32;
        loop {
            let start = idx as usize * SLOT_LEN;
            let cur_len = u32::from_le_bytes(
                slots[start + 4..start + 8]
                    .try_into()
                    .expect("4-byte slice"),
            );
            if cur_len == EMPTY_KEY_LEN {
                slots[start..start + 4].copy_from_slice(&key_off.to_le_bytes());
                slots[start + 4..start + 8].copy_from_slice(&key_len.to_le_bytes());
                slots[start + 8..start + 12].copy_from_slice(&value.to_le_bytes());
                break;
            }
            // `& mask` bounds the result to capacity's own u32 range.
            #[allow(clippy::cast_possible_truncation)]
            {
                idx = ((u64::from(idx) + 1) & mask) as u32;
            }
        }
    }
    (capacity, slots)
}

/// A zero-copy view over one [`build_open_table`]-built hash table: its
/// `slots` (exactly `capacity` times [`SLOT_LEN`] bytes) plus the `pool`
/// its keys were pushed into (see [`push_key`]).
#[derive(Debug, Clone, Copy)]
pub struct HashView<'a> {
    capacity: u32,
    slots: &'a [u8],
    pool: &'a [u8],
}

impl<'a> HashView<'a> {
    /// Wraps `slots`/`pool` as a lookup view.
    ///
    /// # Errors
    /// [`WeightsBinError::MalformedPayload`] if `slots.len()` isn't
    /// exactly `capacity` times [`SLOT_LEN`].
    pub const fn new(
        capacity: u32,
        slots: &'a [u8],
        pool: &'a [u8],
    ) -> Result<Self, WeightsBinError> {
        if slots.len() != capacity as usize * SLOT_LEN {
            return Err(WeightsBinError::MalformedPayload(
                "hash table slot buffer length doesn't match its own recorded capacity",
            ));
        }
        Ok(Self {
            capacity,
            slots,
            pool,
        })
    }

    /// Looks up `key`, returning its stored `u32` value if present.
    ///
    /// Hashes `key` with the same [`HASH_SEED`] [`build_open_table`] used,
    /// then linearly probes from `hash & (capacity - 1)` until it finds a
    /// matching key, an empty slot (definitive miss — construction never
    /// leaves a gap before a key's true home slot in its probe sequence),
    /// or has probed every slot (defensive bound; never reached in
    /// practice since the table's load factor is always under 1).
    #[must_use]
    pub fn get(&self, key: &[u8]) -> Option<u32> {
        let mask = u64::from(self.capacity - 1);
        let hash = xxhash_rust::xxh64::xxh64(key, HASH_SEED);
        #[allow(clippy::cast_possible_truncation)]
        let mut idx = (hash & mask) as u32;
        for _probe in 0..self.capacity {
            let start = idx as usize * SLOT_LEN;
            let slot = &self.slots[start..start + SLOT_LEN];
            let key_len = u32::from_le_bytes(slot[4..8].try_into().expect("4-byte slice"));
            if key_len == EMPTY_KEY_LEN {
                return None;
            }
            if key_len as usize == key.len() {
                let key_off = u32::from_le_bytes(slot[0..4].try_into().expect("4-byte slice"));
                let start_p = key_off as usize;
                if &self.pool[start_p..start_p + key.len()] == key {
                    let value = u32::from_le_bytes(slot[8..12].try_into().expect("4-byte slice"));
                    return Some(value);
                }
            }
            // `& mask` bounds the result to capacity's own u32 range.
            #[allow(clippy::cast_possible_truncation)]
            {
                idx = ((u64::from(idx) + 1) & mask) as u32;
            }
        }
        None
    }
}

/// Appends one feature's sparse per-class/per-action weight row to
/// `values`: a little-endian `u16` entry count, then that many
/// little-endian `(class_or_action_index: u16, weight: f32)` pairs (6
/// bytes each) — self-describing, so a single `u32` byte offset into
/// `values` (this row's start) is enough for [`sparse_row`] to read it
/// back with no separate length table. Returns that starting offset.
///
/// A dense fixed-width row (one `f32` per class/action, always) was
/// rejected: most features only carry a handful of nonzero weights out of
/// the full class/action space, so a dense row would multiply artifact
/// size by roughly the class/action count for no benefit. See each
/// artifact module's own docs for the measured sparsity.
///
/// # Panics
/// Panics if `entries.len()` or `values.len()` somehow exceed `u16::MAX`/
/// `u32::MAX` — never happens for any real weight file (a few dozen
/// classes/actions per feature; a `.bin` payload far under 4 GiB).
#[must_use]
pub fn build_sparse_row(values: &mut Vec<u8>, entries: &[(u16, f32)]) -> u32 {
    let offset = u32::try_from(values.len()).expect(
        "a derived weight artifact's values region stays far \
            under 4 GiB",
    );
    let count = u16::try_from(entries.len())
        .expect("a single feature carries far fewer than 65536 nonzero class/action weights");
    values.extend_from_slice(&count.to_le_bytes());
    for &(index, weight) in entries {
        values.extend_from_slice(&index.to_le_bytes());
        values.extend_from_slice(&weight.to_le_bytes());
    }
    offset
}

/// Reads one [`build_sparse_row`]-written row back: `values[offset..]`'s
/// leading `u16` count, then a byte slice of that many `(u16, f32)`
/// pairs — callers `chunks_exact(6)` it and decode each half with
/// `from_le_bytes`, matching the encoding [`build_sparse_row`] wrote.
///
/// # Panics
/// Panics if `offset` doesn't point at a valid row (never happens for a
/// value this crate's own hash-table lookup returned: every stored
/// `value` is a `build_sparse_row` return value from packing the same
/// payload).
#[must_use]
pub fn sparse_row(values: &[u8], offset: u32) -> &[u8] {
    let start = offset as usize;
    let count = u16::from_le_bytes(values[start..start + 2].try_into().expect("2-byte slice"));
    &values[start + 2..start + 2 + usize::from(count) * 6]
}

/// One decoded entry from a [`sparse_row`] slice's `chunks_exact(6)`.
#[must_use]
pub fn decode_sparse_entry(chunk: &[u8]) -> (u16, f32) {
    let index = u16::from_le_bytes(chunk[0..2].try_into().expect("2-byte slice"));
    let weight = f32::from_le_bytes(chunk[2..6].try_into().expect("4-byte slice"));
    (index, weight)
}

/// Appends `value` as little-endian bytes — [`Cursor::read_u32`]'s exact
/// write-side counterpart.
pub fn write_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

/// Appends `bytes` as a `u32`-length-prefixed section — [`Cursor::read_section`]'s
/// exact write-side counterpart.
///
/// # Panics
/// Panics if `bytes.len()` exceeds `u32::MAX` — never happens for any
/// section this crate packs (a `.bin` payload stays far under 4 GiB).
pub fn write_section(out: &mut Vec<u8>, bytes: &[u8]) {
    write_u32(
        out,
        u32::try_from(bytes.len()).expect("a packed section stays far under 4 GiB"),
    );
    out.extend_from_slice(bytes);
}

/// A tiny forward-only cursor over a byte slice, shared by both artifact
/// modules' version-2 payload readers so the sequence of "read a
/// length-prefixed section" steps isn't duplicated per artifact.
pub struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    #[must_use]
    pub const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    /// Reads one little-endian `u32`, advancing past it.
    ///
    /// # Errors
    /// [`WeightsBinError::MalformedPayload`] if fewer than 4 bytes remain.
    pub fn read_u32(&mut self) -> Result<u32, WeightsBinError> {
        let slice = self.take(4)?;
        Ok(u32::from_le_bytes(
            slice.try_into().expect("checked 4-byte slice"),
        ))
    }

    /// Reads exactly `len` bytes, advancing past them.
    ///
    /// # Errors
    /// [`WeightsBinError::MalformedPayload`] if fewer than `len` bytes
    /// remain.
    pub fn take(&mut self, len: usize) -> Result<&'a [u8], WeightsBinError> {
        let end = self
            .pos
            .checked_add(len)
            .filter(|&end| end <= self.bytes.len())
            .ok_or(WeightsBinError::MalformedPayload(
                "payload section ran past the end of its own artifact",
            ))?;
        let slice = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    /// Reads a `u32`-length-prefixed byte slice (the shape every variable-
    /// length section in a version-2 payload uses), advancing past both.
    ///
    /// # Errors
    /// [`WeightsBinError::MalformedPayload`] — see [`Self::read_u32`]/
    /// [`Self::take`].
    pub fn read_section(&mut self) -> Result<&'a [u8], WeightsBinError> {
        let len = self.read_u32()? as usize;
        self.take(len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MAGIC: [u8; 8] = *b"TESTMAG1";
    const SHA: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    #[test]
    fn write_then_split_round_trips() {
        let mut bytes = write_header(MAGIC, SHA);
        bytes.extend_from_slice(b"payload");
        let (sha, payload) = split_header(&bytes, MAGIC).expect("valid header");
        assert_eq!(sha, SHA);
        assert_eq!(payload, b"payload");
    }

    #[test]
    fn write_header_length_is_exactly_header_len() {
        let bytes = write_header(MAGIC, SHA);
        assert_eq!(bytes.len(), HEADER_LEN);
    }

    #[test]
    fn split_header_rejects_truncated_input() {
        let err = split_header(b"short", MAGIC).unwrap_err();
        assert!(matches!(err, WeightsBinError::Truncated { .. }));
    }

    #[test]
    fn split_header_rejects_wrong_magic() {
        let bytes = write_header(*b"OTHRMAG1", SHA);
        let err = split_header(&bytes, MAGIC).unwrap_err();
        assert!(matches!(err, WeightsBinError::BadMagic));
    }

    #[test]
    fn split_header_rejects_wrong_version() {
        let mut bytes = write_header(MAGIC, SHA);
        bytes[8..10].copy_from_slice(&99u16.to_le_bytes());
        let err = split_header(&bytes, MAGIC).unwrap_err();
        assert!(matches!(
            err,
            WeightsBinError::VersionMismatch { found: 99, .. }
        ));
    }

    #[test]
    #[should_panic(expected = "always exactly 64 characters")]
    fn write_header_panics_on_wrong_length_sha() {
        let _ = write_header(MAGIC, "short");
    }

    // ---------------------------------------------------------------
    // Version-2 hash-table-view primitives.
    // ---------------------------------------------------------------

    #[test]
    fn hash_table_capacity_stays_above_0_7_load_factor() {
        for count in [0usize, 1, 2, 7, 8, 1000, 54_421, 324_838] {
            let capacity = hash_table_capacity(count);
            assert!(capacity.is_power_of_two());
            assert!(
                (count as u64) * 10 <= u64::from(capacity) * 7,
                "count={count} capacity={capacity}"
            );
            // A tighter table wouldn't satisfy the same bound (except at
            // count 0, where capacity 1 is already the floor).
            if capacity > 1 {
                assert!((count as u64) * 10 > u64::from(capacity / 2) * 7);
            }
        }
    }

    fn build_and_view<'a>(
        pool: &'a mut Vec<u8>,
        pairs: &[(&str, u32)],
    ) -> (u32, Vec<u8>, &'a [u8]) {
        let entries: Vec<(&str, u32, u32, u32)> = pairs
            .iter()
            .map(|&(key, value)| {
                let (off, len) = push_key(pool, key);
                (key, off, len, value)
            })
            .collect();
        let (capacity, slots) = build_open_table(&entries);
        (capacity, slots, pool.as_slice())
    }

    #[test]
    fn hash_view_round_trips_every_inserted_key() {
        let pairs = [
            ("word=the", 1u32),
            ("word=cat", 2),
            ("suffix=ing", 3),
            ("bias", 4),
            ("i-1 tag=NN", 5),
        ];
        let mut pool = Vec::new();
        let (capacity, slots, pool_ref) = build_and_view(&mut pool, &pairs);
        let view = HashView::new(capacity, &slots, pool_ref).expect("valid slots length");
        for (key, value) in pairs {
            assert_eq!(view.get(key.as_bytes()), Some(value), "key {key:?}");
        }
    }

    #[test]
    fn hash_view_returns_none_for_an_absent_key() {
        let pairs = [("alpha", 1u32), ("beta", 2)];
        let mut pool = Vec::new();
        let (capacity, slots, pool_ref) = build_and_view(&mut pool, &pairs);
        let view = HashView::new(capacity, &slots, pool_ref).expect("valid slots length");
        assert_eq!(view.get(b"gamma"), None);
        assert_eq!(view.get(b"al"), None);
        assert_eq!(view.get(b""), None);
    }

    #[test]
    fn hash_view_handles_an_empty_table() {
        let (capacity, slots) = build_open_table(&[]);
        let view = HashView::new(capacity, &slots, b"").expect("valid slots length");
        assert_eq!(view.get(b"anything"), None);
    }

    #[test]
    fn hash_view_rejects_a_slots_length_mismatch() {
        let err = HashView::new(4, &[0u8; 10], b"").unwrap_err();
        assert!(matches!(err, WeightsBinError::MalformedPayload(_)));
    }

    #[test]
    fn build_open_table_is_deterministic_for_a_fixed_feed_order() {
        let pairs = [("z", 1u32), ("a", 2), ("m", 3), ("q", 4)];
        let mut pool1 = Vec::new();
        let (cap1, slots1, _) = build_and_view(&mut pool1, &pairs);
        let mut pool2 = Vec::new();
        let (cap2, slots2, _) = build_and_view(&mut pool2, &pairs);
        assert_eq!(cap1, cap2);
        assert_eq!(slots1, slots2);
        assert_eq!(pool1, pool2);
    }

    #[test]
    fn sparse_row_round_trips_several_class_weight_pairs() {
        let mut values = Vec::new();
        let offset = build_sparse_row(&mut values, &[(3u16, 1.5f32), (7, -2.25), (0, 0.0)]);
        let row = sparse_row(&values, offset);
        let decoded: Vec<(u16, f32)> = row.chunks_exact(6).map(decode_sparse_entry).collect();
        assert_eq!(decoded, vec![(3, 1.5), (7, -2.25), (0, 0.0)]);
    }

    #[test]
    fn sparse_row_round_trips_an_empty_row() {
        let mut values = Vec::new();
        let offset = build_sparse_row(&mut values, &[]);
        let row = sparse_row(&values, offset);
        assert!(row.is_empty());
    }

    #[test]
    fn multiple_sparse_rows_do_not_overlap() {
        let mut values = Vec::new();
        let first = build_sparse_row(&mut values, &[(1u16, 1.0f32)]);
        let second = build_sparse_row(&mut values, &[(2u16, 2.0f32), (3, 3.0)]);
        let first_row: Vec<(u16, f32)> = sparse_row(&values, first)
            .chunks_exact(6)
            .map(decode_sparse_entry)
            .collect();
        let second_row: Vec<(u16, f32)> = sparse_row(&values, second)
            .chunks_exact(6)
            .map(decode_sparse_entry)
            .collect();
        assert_eq!(first_row, vec![(1, 1.0)]);
        assert_eq!(second_row, vec![(2, 2.0), (3, 3.0)]);
    }

    #[test]
    fn cursor_reads_u32_and_length_prefixed_sections_in_sequence() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&42u32.to_le_bytes());
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(b"abc");
        let mut cursor = Cursor::new(&bytes);
        assert_eq!(cursor.read_u32().expect("first u32"), 42);
        assert_eq!(cursor.read_section().expect("section"), b"abc");
    }

    #[test]
    fn write_u32_and_write_section_round_trip_through_cursor() {
        let mut bytes = Vec::new();
        write_u32(&mut bytes, 7);
        write_section(&mut bytes, b"hello");
        write_section(&mut bytes, b"");
        let mut cursor = Cursor::new(&bytes);
        assert_eq!(cursor.read_u32().expect("u32"), 7);
        assert_eq!(cursor.read_section().expect("section"), b"hello");
        assert_eq!(cursor.read_section().expect("empty section"), b"");
    }

    #[test]
    fn cursor_rejects_reads_past_the_end() {
        let bytes = [0u8; 3];
        let mut cursor = Cursor::new(&bytes);
        let err = cursor.read_u32().unwrap_err();
        assert!(matches!(err, WeightsBinError::MalformedPayload(_)));
    }
}
