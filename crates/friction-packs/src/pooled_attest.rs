//! `pooled-attest-v1`: pooled external seam-bigram membership, widening
//! [`crate::attestation::BigramTable`]'s evidence beyond the TRAIN human
//! split.
//!
//! # Motivation
//!
//! [`crate::attestation::BigramTable`] (the table `friction-edit`'s
//! runtime rewrite/deletion gates consult) is mined by `corpus-tool
//! attest` from the TRAIN human split alone — roughly 300k tokens.
//! Measured on real candidate edits, that table is the recall bottleneck,
//! not the candidates it's asked to attest: an ordinary seam like `("will",
//! "upgrade")` can fail simply because the small TRAIN corpus never
//! happened to write it, not because the seam is actually unusual English.
//!
//! This module mines a SECOND, much larger membership structure from
//! locally-staged external human corpora (`corpus-tool pooled-attest`,
//! currently ~43M tokens of code-review prose — two orders of magnitude
//! more than TRAIN) and exposes it as a compact `BinaryFuse16` filter.
//! [`crate::attestation::AttestationPack::with_pooled`] is the one seam
//! that wires it into [`crate::attestation::BigramTable::attests`]: every
//! gate that already calls `attests` sees the widened evidence for free,
//! with no per-call-site change (see that method's own docs).
//!
//! # Membership, not exact-table replacement
//!
//! This pack never replaces the exact TRAIN table — it only ever adds a
//! second, independent "have I plausibly seen this pair" signal `attests`
//! consults after the exact table misses. A pair present in the exact
//! table but absent from the pooled filter (small overlap corpora, or a
//! pair too rare to clear the `>= 2` mining floor below) is unaffected:
//! the exact table's own `true` always short-circuits before the pooled
//! filter is even queried.
//!
//! # Count-`>= 2` floor: a deliberate deviation from the train table
//!
//! [`crate::attestation::BigramTable`]'s TRAIN table is pure membership —
//! any pair observed even once is attested, because TRAIN is curated,
//! reviewed prose (`corpus-tool ingest`'s own provenance discipline). The
//! external corpora this module mines are NOT reviewed in that sense —
//! large, unedited web-scale text (Stack Exchange code review threads,
//! GitHub PR comments) — so a singleton adjacency is exactly the shape a
//! typo, a copy-paste artifact, or a one-off garbled sentence produces.
//! Requiring the pair to recur (`>= 2` independent occurrences, mined by
//! `corpus-tool pooled-attest`) is "existence with a witness": cheap
//! insurance against seeding the gate's widened evidence with noise,
//! at the cost of very slightly narrowing what a single, larger-but-
//! unreviewed corpus alone would have attested. This is a real,
//! deliberate difference from the TRAIN table's pure-membership stance,
//! not an oversight — the two tables' inputs have different trust levels
//! and are held to different bars accordingly.
//!
//! # Collision direction is safe by construction
//!
//! A `BinaryFuse16` has no false negatives and a false positive rate of
//! ~1/65536 (16-bit fingerprints; ~2.2 bytes/key on disk). A false
//! positive here means a `(left, right)` pair that was never actually
//! mined nonetheless reads as "pooled-attested" — which only ever admits
//! one additional candidate seam past a gate that would otherwise have
//! held it (a Suggest-tier finding instead of an applied edit, per
//! `friction-edit`'s fail-closed design). At a 1-in-65536 rate this is
//! harmless. The reverse error (a real, repeatedly-observed pair the
//! filter fails to attest) is categorically impossible for a Bloom/fuse
//! filter — the shipped pack's only false-negative source is corpus
//! coverage (a pair the staged corpora never happened to contain), the
//! same honest limitation every attestation table in this crate carries.
//!
//! # One key convention, shared, so drift is impossible
//!
//! [`pair_key`] is the single definition of "this pack's lookup key": the
//! left and right token joined by the ASCII `\x1f` (unit separator)
//! control character, never whitespace or an empty separator. `\x1f`
//! cannot occur inside a `friction_harness::clean::tokenize` token (that
//! tokenizer only ever emits `[a-z']+` word runs or single `.,;:!?`
//! punctuation characters), so `pair_key` is injective: `("a b", "c")` and
//! `("a", "b c")` join to `"a b\x1fc"` and `"a\x1fb c"` respectively —
//! distinguishable strings — where a bare-concatenation join (`"a bc"` /
//! `"a bc"`) would have collided. `corpus-tool pooled-attest` (mining) and
//! [`PooledAttestPack::attests`] (query) both build their hash input
//! through this one function, and both hash with the same fixed
//! [`HASH_SEED`] via [`hash_key`] — never std's randomized `SipHash`,
//! which would make a rebuild's filter answers vary run to run.
//!
//! `"<s>"` (the reserved sentence-start token — see
//! [`crate::attestation`]'s own module docs) is a perfectly ordinary
//! `left` value here: it is hashed like any other token, with no special
//! casing, so `attests("<s>", w)` ("does a human sentence ever open with
//! `w`") answers identically through the pooled path as through the exact
//! table.
//!
//! # Determinism guarantee
//!
//! Same discipline as [`crate::jargon_attest`] (see that module's own
//! docs for the full argument): `xorf`'s `BinaryFuse16::try_from` retries
//! from a fixed initial `splitmix64` state, never ambient randomness, and
//! this module always feeds keys in a [`BTreeSet`]'s ascending order (see
//! [`build_pack_bytes`]) — building the filter twice from the same input
//! key set produces bit-identical `.bin` bytes, pinned by this module's
//! own `determinism_is_bit_identical_for_a_fixed_input_set` test.
//!
//! # Serialization
//!
//! Hand-rolled, mirroring [`crate::jargon_attest`]'s
//! [`xorf::DmaSerializable`]-based layout with two additions
//! ([`crate::attestation`]'s own header shape): an explicit format
//! version and the source evidence's own sha256, both ahead of the key
//! count. See [`HEADER_LEN`] for the exact byte layout.
//!
//! # Fingerprint alignment on the read side
//!
//! `xorf::BinaryFuse16Ref::from_dma` requires its fingerprint byte slice
//! to be 2-byte aligned (`FilterRef::FINGERPRINT_ALIGNMENT`) — internally
//! it reinterprets those bytes in place as `&[u16]`. A raw
//! `include_bytes!` static gives no such guarantee (only 1-byte
//! alignment), unlike [`crate::jargon_attest`]'s `BinaryFuse8` (1-byte
//! fingerprints, no alignment requirement at all). [`PooledAttestPack::load`]
//! fixes this with one load-time copy: the embedded fingerprint bytes are
//! read into a freshly allocated `Vec<u16>` (2-aligned by the allocator's
//! own `u16` layout, no `unsafe` required — this workspace denies
//! `unsafe_code` crate-wide), leaked for the process lifetime, and handed
//! to `from_dma` as a `&'static [u8]` view via `zerocopy::IntoBytes`. This
//! costs one copy of a ~3-4 MB artifact at process start, not a
//! per-query cost.

use std::collections::BTreeSet;

use xorf::{BinaryFuse16, BinaryFuse16Ref, DmaSerializable, Filter, FilterRef};
use zerocopy::IntoBytes;

use crate::PackError;

/// The fixed xxh64 seed every pooled pair key is hashed with.
///
/// `u64::from_be_bytes(*b"POOLSEAM")` — arbitrary but fixed and
/// self-documenting, and (by construction, since the two seeds' input
/// bytes differ) distinct from [`crate::jargon_attest::HASH_SEED`]: two
/// packs sharing a hash seed would let a jargon-compound key and a
/// seam-pair key collide into the same fingerprint space if the two
/// filters were ever compared or merged, which they never are today, but
/// there's no reason to leave that door open.
pub const HASH_SEED: u64 = u64::from_be_bytes(*b"POOLSEAM");

/// The ASCII "unit separator" control character `pair_key` joins `left`
/// and `right` with.
///
/// See this module's own docs for why a separator (rather than bare
/// concatenation) is required for injectivity, and why this specific
/// character can never appear inside either half.
pub const PAIR_SEPARATOR: char = '\u{1f}';

/// Builds this pack's canonical lookup key for a `(left, right)` seam pair.
///
/// The ONE function both [`build_pack_bytes`] (via `corpus-tool
/// pooled-attest`'s mining pass) and [`PooledAttestPack::attests`] (the
/// runtime query) route through, so the two sides can never drift apart
/// on what a key means.
///
/// # Examples
/// ```
/// assert_eq!(
///     friction_packs::pooled_attest::pair_key("will", "upgrade"),
///     "will\u{1f}upgrade"
/// );
/// ```
#[must_use]
pub fn pair_key(left: &str, right: &str) -> String {
    let mut key = String::with_capacity(left.len() + 1 + right.len());
    key.push_str(left);
    key.push(PAIR_SEPARATOR);
    key.push_str(right);
    key
}

/// Hashes an already-[`pair_key`]d string with the fixed [`HASH_SEED`] —
/// the one hash both [`build_pack_bytes`] and [`PooledAttestPack::attests`]
/// use.
#[must_use]
pub fn hash_key(key: &str) -> u64 {
    xxhash_rust::xxh64::xxh64(key.as_bytes(), HASH_SEED)
}

/// 8-byte magic every `pooled-attest-v1` `.bin` starts with.
const MAGIC: &[u8; 8] = b"POOLFUS1";

/// The artifact's on-disk format version: bumped if the header/section
/// layout below ever changes shape (the filter's mined content — a
/// corpus refresh — can change without bumping this).
const FORMAT_VERSION: u16 = 1;

/// A sha256 digest is recorded as lowercase hex (matching
/// [`crate::Sha256`]'s `Display`) — same choice as
/// [`crate::attestation`]'s artifact header.
const SHA256_HEX_LEN: usize = 64;

/// The fixed-length prefix every `.bin` starts with: [`MAGIC`] (8 bytes),
/// [`FORMAT_VERSION`] (2 bytes, little-endian), the source evidence's
/// sha256 hex digest ([`SHA256_HEX_LEN`] bytes — see [`BuiltPack::pairs_sha256`]),
/// a little-endian key count (8 bytes), then `xorf`'s own filter
/// descriptor ([`DmaSerializable::DESCRIPTOR_LEN`]). Everything after this
/// is raw little-endian `u16` fingerprint bytes.
const HEADER_LEN: usize =
    8 + 2 + SHA256_HEX_LEN + 8 + <BinaryFuse16 as DmaSerializable>::DESCRIPTOR_LEN;

/// The result of [`build_pack_bytes`]: the serialized `.bin` contents plus
/// the bookkeeping `corpus-tool pooled-attest` needs for its `.meta.toml`
/// sidecar and summary output.
#[derive(Debug, Clone)]
pub struct BuiltPack {
    /// The full `.bin` file contents (header + descriptor + fingerprints).
    pub bytes: Vec<u8>,
    /// The number of distinct hashes actually inserted into the filter —
    /// may be marginally less than the input key set's length if
    /// [`hash_key`] collided (see `hash_collisions`); this is the count
    /// [`PooledAttestPack::load`] cross-checks the meta sidecar's
    /// `[pack].key_count` against.
    pub key_count: u64,
    /// How many input keys were dropped because their hash collided with
    /// an already-inserted key's hash. Expected `0` at any realistic
    /// pooled-corpus scale (a 64-bit hash space) — recorded, not silently
    /// ignored, mirroring [`crate::jargon_attest::BuiltPack::hash_collisions`].
    pub hash_collisions: u64,
    /// The sha256 hex digest of the exact mined evidence this filter was
    /// built from: `pair_keys`' own `BTreeSet`-ascending members, each
    /// followed by a `\n`, concatenated. Recorded in the `.bin` header
    /// (this artifact's own "source sha") and cross-referenced in the
    /// meta sidecar, the same staleness guard `crate::attestation`'s
    /// source-TOML sha256 provides for that pack.
    pub pairs_sha256: String,
}

/// Builds a `pooled-attest-v1` pack's `.bin` bytes from `pair_keys`.
///
/// Already [`pair_key`]-joined, already deduped and sorted (a
/// [`BTreeSet`]'s natural state; `corpus-tool pooled-attest` folds every
/// `(left, right)` pair that cleared its `>= 2`-occurrence mining floor
/// into exactly this shape before calling here).
///
/// Keys are hashed ([`hash_key`]) in `pair_keys`'s own ascending iteration
/// order and fed to `xorf::BinaryFuse16::try_from` in that exact order —
/// see this module's own determinism-guarantee docs for why that's what
/// makes two builds from the same corpus produce bit-identical bytes.
///
/// # Errors
/// [`PackError::PooledAttestConstructionFailed`] if `xorf`'s bounded
/// retry loop never finds a working seed arrangement — see
/// [`crate::jargon_attest::build_pack_bytes`]'s own docs for how unlikely
/// this is in practice (the identical condition, for `BinaryFuse8` there).
pub fn build_pack_bytes(pair_keys: &BTreeSet<String>) -> Result<BuiltPack, PackError> {
    let mut seen_hashes: BTreeSet<u64> = BTreeSet::new();
    let mut keys: Vec<u64> = Vec::with_capacity(pair_keys.len());
    let mut hash_collisions = 0u64;
    for key in pair_keys {
        let hash = hash_key(key);
        if seen_hashes.insert(hash) {
            keys.push(hash);
        } else {
            hash_collisions += 1;
        }
    }

    let filter = BinaryFuse16::try_from(&keys).map_err(|message| {
        PackError::PooledAttestConstructionFailed {
            message: message.to_string(),
        }
    })?;

    let mut descriptor = vec![0u8; <BinaryFuse16 as DmaSerializable>::DESCRIPTOR_LEN];
    filter.dma_copy_descriptor_to(&mut descriptor);

    // Explicit little-endian re-encoding of every fingerprint, not
    // `DmaSerializable::dma_fingerprints`'s raw native-endian byte view:
    // this workspace's every other multi-byte artifact field round-trips
    // through `to_le_bytes`/`from_le_bytes`, and doing the same here means
    // the read side never has to reason about the build machine's
    // endianness (every realistic target — x86_64, aarch64, wasm32 — is
    // little-endian anyway, so this changes no byte in practice; it just
    // makes the format's own contract explicit rather than incidental).
    let mut fingerprint_bytes = Vec::with_capacity(filter.fingerprints.len() * 2);
    for fp in &filter.fingerprints {
        fingerprint_bytes.extend_from_slice(&fp.to_le_bytes());
    }

    let mut pairs_source = String::new();
    for key in pair_keys {
        pairs_source.push_str(key);
        pairs_source.push('\n');
    }
    let pairs_sha256 = crate::Sha256::of_bytes(pairs_source.as_bytes()).to_string();

    let key_count = keys.len();
    let mut bytes = Vec::with_capacity(HEADER_LEN + fingerprint_bytes.len());
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    bytes.extend_from_slice(pairs_sha256.as_bytes());
    bytes.extend_from_slice(&u64::try_from(key_count).unwrap_or(u64::MAX).to_le_bytes());
    bytes.extend_from_slice(&descriptor);
    bytes.extend_from_slice(&fingerprint_bytes);

    Ok(BuiltPack {
        bytes,
        key_count: key_count as u64,
        hash_collisions,
        pairs_sha256,
    })
}

/// A parsed, validated `pooled-attest-v1` pack.
///
/// A `BinaryFuse16Ref` view over the embedded `.bin`'s fingerprints (see
/// this module's own docs on the one load-time alignment copy that makes
/// the view possible), plus the metadata [`Self::load`] cross-checked it
/// against.
#[derive(Debug, Clone)]
pub struct PooledAttestPack {
    filter: BinaryFuse16Ref<'static>,
    key_count: u64,
    pairs_sha256: Box<str>,
}

impl PooledAttestPack {
    /// Parses `bin` (this pack's embedded `.bin`, `include_bytes!`'d by
    /// the caller — see [`crate::registry`]) and `meta_toml` (its
    /// `.meta.toml` sidecar, `include_str!`'d the same way) into a
    /// validated pack.
    ///
    /// Validates: `bin` is at least [`HEADER_LEN`] bytes and starts with
    /// [`MAGIC`]; its recorded format version is [`FORMAT_VERSION`]; its
    /// fingerprint section's byte length is even (every fingerprint is a
    /// `u16`); the sidecar's `[pack].version` is exactly
    /// `"pooled-attest-v1"`; the sidecar's `[pack].key_count` matches the
    /// `.bin`'s own embedded key count.
    ///
    /// # Errors
    /// [`PackError::PooledAttestTruncated`], [`PackError::PooledAttestBadMagic`],
    /// [`PackError::PooledAttestUnsupportedVersion`],
    /// [`PackError::PooledAttestOddFingerprintLength`], [`PackError::Toml`],
    /// [`PackError::PooledAttestVersionMismatch`], or
    /// [`PackError::PooledAttestKeyCountMismatch`]: see each variant's own
    /// docs.
    pub fn load(bin: &'static [u8], meta_toml: &str) -> Result<Self, PackError> {
        if bin.len() < HEADER_LEN {
            return Err(PackError::PooledAttestTruncated {
                len: bin.len(),
                min: HEADER_LEN,
            });
        }
        if &bin[..8] != MAGIC {
            return Err(PackError::PooledAttestBadMagic);
        }
        let version = u16::from_le_bytes(bin[8..10].try_into().expect("checked 2-byte slice"));
        if version != FORMAT_VERSION {
            return Err(PackError::PooledAttestUnsupportedVersion(version));
        }
        let pairs_sha256 = std::str::from_utf8(&bin[10..10 + SHA256_HEX_LEN])
            .map_err(|_| PackError::PooledAttestTruncated {
                len: bin.len(),
                min: HEADER_LEN,
            })?
            .to_string();
        let key_count_at = 10 + SHA256_HEX_LEN;
        let key_count = u64::from_le_bytes(
            bin[key_count_at..key_count_at + 8]
                .try_into()
                .expect("checked 8-byte slice"),
        );
        let descriptor_at = key_count_at + 8;
        let descriptor = &bin[descriptor_at..HEADER_LEN];
        let fingerprint_raw = &bin[HEADER_LEN..];

        if !fingerprint_raw.len().is_multiple_of(2) {
            return Err(PackError::PooledAttestOddFingerprintLength(
                fingerprint_raw.len(),
            ));
        }

        let meta: RawMeta = toml::from_str(meta_toml).map_err(PackError::from)?;
        if meta.pack.version != "pooled-attest-v1" {
            return Err(PackError::PooledAttestVersionMismatch {
                found: meta.pack.version,
            });
        }
        if meta.pack.key_count != key_count {
            return Err(PackError::PooledAttestKeyCountMismatch {
                meta: meta.pack.key_count,
                embedded: key_count,
            });
        }

        let fingerprint_bytes = align_fingerprints(fingerprint_raw);
        let filter = BinaryFuse16Ref::from_dma(descriptor, fingerprint_bytes);
        Ok(Self {
            filter,
            key_count,
            pairs_sha256: pairs_sha256.into_boxed_str(),
        })
    }

    /// The number of keys embedded in the filter.
    #[must_use]
    pub const fn key_count(&self) -> u64 {
        self.key_count
    }

    /// The sha256 hex digest of the exact mined evidence this pack's
    /// filter was built from — see [`BuiltPack::pairs_sha256`].
    #[must_use]
    pub fn pairs_sha256(&self) -> &str {
        &self.pairs_sha256
    }

    /// `true` if `(left, right)` hashes into this filter — the pooled-side
    /// half of [`crate::attestation::BigramTable::attests`]'s widened
    /// query. Callers pass raw, already-tokenized text exactly as they
    /// would to the exact table (including `"<s>"` as a `left` value);
    /// [`pair_key`] is the one place normalization/joining happens.
    #[must_use]
    pub fn attests(&self, left: &str, right: &str) -> bool {
        self.filter.contains(&hash_key(&pair_key(left, right)))
    }
}

/// Copies `raw` (this pack's little-endian fingerprint bytes, as embedded
/// — see [`build_pack_bytes`]'s explicit LE re-encoding) into a freshly
/// allocated, leaked `Vec<u16>` and returns a `&'static [u8]` view over
/// that same allocation.
///
/// This is the one load-time fix for `BinaryFuse16Ref::from_dma`'s 2-byte
/// alignment requirement (see this module's own docs): a `Vec<u16>`'s
/// backing allocation is 2-aligned by construction (the allocator sizes
/// it for `u16`'s own layout), where the raw `include_bytes!` static `raw`
/// is sliced from carries no such guarantee. `zerocopy::IntoBytes::as_bytes`
/// performs the `&[u16] -> &[u8]` reinterpretation safely (no `unsafe` in
/// this crate — see `Cargo.toml`'s own comment on this dependency), and
/// leaking the `Vec` is correct here for the same reason
/// `crate::registry`'s embedded-pack statics leak nothing at all: this
/// runs once, inside a `LazyLock`, for the life of the process.
fn align_fingerprints(raw: &[u8]) -> &'static [u8] {
    let mut owned: Vec<u16> = Vec::with_capacity(raw.len() / 2);
    for chunk in raw.as_chunks::<2>().0 {
        owned.push(u16::from_le_bytes(*chunk));
    }
    let leaked: &'static [u16] = Vec::leak(owned);
    leaked.as_bytes()
}

/// The meta sidecar TOML's `[pack]` table — only the two fields this
/// reader cross-checks against the `.bin` header; every other field
/// (source labels, token counts, mining threshold, `built_at`, ...) is
/// provenance for humans and intentionally not re-parsed here, matching
/// [`crate::jargon_attest`]'s sidecar-tolerance convention.
#[derive(Debug, serde::Deserialize)]
struct RawMetaPack {
    version: String,
    key_count: u64,
}

#[derive(Debug, serde::Deserialize)]
struct RawMeta {
    pack: RawMetaPack,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pair_key_joins_with_the_unit_separator() {
        assert_eq!(pair_key("will", "upgrade"), "will\u{1f}upgrade");
        assert_eq!(pair_key("<s>", "the"), "<s>\u{1f}the");
    }

    #[test]
    fn pair_key_is_injective_across_a_naive_word_boundary_shift() {
        // Without a separator, ("a b", "c") and ("a", "b c") would both
        // join to "abc"; the separator keeps them distinct strings.
        assert_ne!(pair_key("a b", "c"), pair_key("a", "b c"));
    }

    #[test]
    fn hash_key_is_deterministic_and_seed_dependent() {
        assert_eq!(hash_key("will\u{1f}upgrade"), hash_key("will\u{1f}upgrade"));
        assert_ne!(hash_key("will\u{1f}upgrade"), hash_key("upgrade\u{1f}will"));
    }

    #[test]
    fn hash_seed_differs_from_jargon_attests_seed() {
        assert_ne!(HASH_SEED, crate::jargon_attest::HASH_SEED);
    }

    fn sample_keys() -> BTreeSet<String> {
        [
            pair_key("<s>", "the"),
            pair_key("will", "upgrade"),
            pair_key("upgrade", "the"),
            pair_key("the", "system"),
            pair_key("system", "."),
            pair_key("please", "review"),
        ]
        .into_iter()
        .collect()
    }

    fn sample_meta(key_count: u64) -> String {
        format!("[pack]\nversion = \"pooled-attest-v1\"\nkey_count = {key_count}\n")
    }

    #[test]
    fn build_pack_bytes_round_trips_through_load_and_attests() {
        let keys = sample_keys();
        let built = build_pack_bytes(&keys).expect("small sample builds");
        assert_eq!(built.hash_collisions, 0);
        assert_eq!(built.key_count, keys.len() as u64);

        let bin: &'static [u8] = Box::leak(built.bytes.into_boxed_slice());
        let meta = sample_meta(built.key_count);
        let pack = PooledAttestPack::load(bin, &meta).expect("round-trips");

        assert!(pack.attests("<s>", "the"), "<s> is an ordinary left value");
        assert!(pack.attests("will", "upgrade"));
        assert!(pack.attests("upgrade", "the"));
        assert!(!pack.attests("the", "will"), "reversed pair never inserted");
        assert!(!pack.attests("never", "seen"));
        assert_eq!(pack.pairs_sha256(), built.pairs_sha256);
    }

    /// Determinism guarantee this module's header claims: building twice
    /// from the same (small, synthetic) input set produces bit-identical
    /// `.bin` bytes — see [`crate::jargon_attest`]'s identical test for the
    /// full argument this mirrors.
    #[test]
    fn determinism_is_bit_identical_for_a_fixed_input_set() {
        let keys = sample_keys();
        let first = build_pack_bytes(&keys).expect("builds");
        let second = build_pack_bytes(&keys).expect("builds");
        assert_eq!(first.bytes, second.bytes);
        assert_eq!(first.key_count, second.key_count);
        assert_eq!(first.hash_collisions, second.hash_collisions);
        assert_eq!(first.pairs_sha256, second.pairs_sha256);
    }

    #[test]
    fn load_rejects_truncated_bin() {
        let err = PooledAttestPack::load(b"short", &sample_meta(0)).unwrap_err();
        assert!(matches!(err, PackError::PooledAttestTruncated { .. }));
    }

    #[test]
    fn load_rejects_bad_magic() {
        let keys = sample_keys();
        let mut built = build_pack_bytes(&keys).expect("builds");
        built.bytes[0] = b'X';
        let bin: &'static [u8] = Box::leak(built.bytes.into_boxed_slice());
        let err = PooledAttestPack::load(bin, &sample_meta(built.key_count)).unwrap_err();
        assert!(matches!(err, PackError::PooledAttestBadMagic));
    }

    #[test]
    fn load_rejects_unsupported_format_version() {
        let keys = sample_keys();
        let mut built = build_pack_bytes(&keys).expect("builds");
        built.bytes[8] = 0xFF;
        built.bytes[9] = 0xFF;
        let bin: &'static [u8] = Box::leak(built.bytes.into_boxed_slice());
        let err = PooledAttestPack::load(bin, &sample_meta(built.key_count)).unwrap_err();
        assert!(matches!(err, PackError::PooledAttestUnsupportedVersion(_)));
    }

    #[test]
    fn load_rejects_meta_version_mismatch() {
        let keys = sample_keys();
        let built = build_pack_bytes(&keys).expect("builds");
        let bin: &'static [u8] = Box::leak(built.bytes.into_boxed_slice());
        let bad_meta = "[pack]\nversion = \"pooled-attest-v0\"\nkey_count = 6\n";
        let err = PooledAttestPack::load(bin, bad_meta).unwrap_err();
        assert!(matches!(err, PackError::PooledAttestVersionMismatch { .. }));
    }

    #[test]
    fn load_rejects_meta_key_count_mismatch() {
        let keys = sample_keys();
        let built = build_pack_bytes(&keys).expect("builds");
        let bin: &'static [u8] = Box::leak(built.bytes.into_boxed_slice());
        let err = PooledAttestPack::load(bin, &sample_meta(999)).unwrap_err();
        assert!(matches!(
            err,
            PackError::PooledAttestKeyCountMismatch { .. }
        ));
    }

    #[test]
    fn load_rejects_a_truncated_fingerprint_section() {
        let keys = sample_keys();
        let built = build_pack_bytes(&keys).expect("builds");
        let meta = sample_meta(built.key_count);
        let truncated: &'static [u8] = Box::leak(
            built.bytes[..built.bytes.len() - 1]
                .to_vec()
                .into_boxed_slice(),
        );
        let err = PooledAttestPack::load(truncated, &meta).unwrap_err();
        assert!(matches!(
            err,
            PackError::PooledAttestOddFingerprintLength(_)
        ));
    }
}
