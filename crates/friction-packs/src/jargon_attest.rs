//! `jargon-attest-v1`: web-scale compound attestation (SYNTHESIS.md §4).
//!
//! Replaces a hand-curated allowlist with a real attestation oracle — a
//! [`xorf::BinaryFuse8`] filter over ~2M case-folded, multiword concept
//! keys mined from Wikipedia article titles (CC BY-SA 4.0 + GFDL) and
//! `OpenAlex` Topics display names (CC0), embedded in the binary as a
//! versioned pack (`packs/jargon-attest-v1.bin` + its `.toml` sidecar).
//!
//! # Collision direction is safe by construction
//!
//! A `BinaryFuse8` has no false negatives and a small (~0.4%) false
//! positive rate.
//!
//! A false positive here means a compound that was never actually a
//! Wikipedia title or `OpenAlex` topic nonetheless reads as "attested" —
//! which only ever *suppresses* a [`crate::jargon`] `jargon.metaphor`
//! flag, never invents one. The reverse error (a real title absent from
//! the table, table-lag or corpus-gap) is the documented, honest
//! limitation SYNTHESIS.md §4 already calls out for this whole detection
//! channel, not something this filter adds.
//!
//! # One normalization function, shared, so drift is impossible
//!
//! [`normalize_compound`] is the single definition of "this pack's lookup
//! key".
//!
//! `corpus-tool jargon-attest` runs every raw Wikipedia/`OpenAlex` title
//! through it while building [`build_pack_bytes`]'s input set, and
//! [`crate::jargon`]'s `jargon.metaphor` channel runs every candidate
//! compound span through the *same* function before calling
//! [`JargonAttestPack::is_attested`]. Both sides also hash with the same
//! fixed [`HASH_SEED`] via [`hash_key`] — never std's randomized
//! `SipHash`, which would make the filter's membership answers vary
//! per-process and the `.bin` artifact meaningless to rebuild
//! reproducibly.
//!
//! # Determinism guarantee
//!
//! `xorf`'s `BinaryFuse8::try_from` is not naively deterministic in
//! general: construction can fail on an unlucky hash arrangement and
//! retries with a new seed.
//!
//! But `xorf` derives that seed sequence from a **fixed initial state**
//! (`splitmix64` seeded from the constant `1`, advanced by its own
//! deterministic retry loop) rather than from ambient randomness
//! (`rand::thread_rng` or similar) — verified against `xorf` 0.12.0's
//! source (`src/prelude/bfuse.rs`'s `bfuse_from_impl!` macro). Combined
//! with a fixed key *order* (this module always feeds keys in a
//! [`BTreeSet`]'s ascending order — see [`build_pack_bytes`]), the entire
//! construction is a pure function of the input key set: **building the
//! filter twice from the same input produces bit-identical `.bin`
//! bytes**, pinned by this module's own
//! `determinism_is_bit_identical_for_a_fixed_input_set` test. This is a
//! *stronger* guarantee than "same input, same semantic answers" — no
//! fallback to the weaker guarantee was needed.
//!
//! # Serialization: hand-rolled, zero extra dependencies
//!
//! Rather than pull in `xorf`'s `serde` feature (which drags in
//! `serde_bytes` for no real benefit here), this module serializes
//! through [`xorf::DmaSerializable`] instead.
//!
//! `DmaSerializable` is already part of `xorf`'s default surface; this
//! module wraps it in a tiny hand-rolled layout: an 8-byte magic, a
//! little-endian `u64` key count, `xorf`'s own fixed-length filter
//! descriptor, then the raw fingerprint bytes. At load time
//! [`xorf::FilterRef::from_dma`] reconstructs a zero-copy
//! [`xorf::BinaryFuse8Ref`] directly over the `include_bytes!`'d slice —
//! no parsing pass, no allocation beyond the filter's own descriptor.

use std::collections::BTreeSet;

use xorf::{BinaryFuse8, BinaryFuse8Ref, DmaSerializable, Filter, FilterRef};

use crate::PackError;

/// The fixed xxh64 seed every normalized compound key is hashed with.
///
/// Used on both the build side ([`build_pack_bytes`]) and the query side
/// ([`JargonAttestPack::is_attested`]). `b"FRICTION"` read as a
/// big-endian `u64` — arbitrary, but fixed and documented, unlike std's
/// randomized-per-process `SipHash` default (which this module explicitly
/// does not use — see the module header's determinism discussion).
pub const HASH_SEED: u64 = 0x4652_4943_5449_4F4E;

/// A normalized compound key is kept only with this many space-separated
/// words.
///
/// This is the builder's inclusion filter (`corpus-tool`'s
/// `jargon-attest` subcommand). Documented here but not re-checked by
/// [`JargonAttestPack::is_attested`] itself — see that method's own docs
/// for why the runtime doesn't re-apply it.
pub const MIN_WORDS: usize = 2;
/// See [`MIN_WORDS`].
pub const MAX_WORDS: usize = 4;

/// Normalizes `raw` into this pack's canonical lookup key.
///
/// Folds underscores AND hyphens to a single space each, case-folds,
/// then collapses every whitespace run (including ones the previous
/// steps just produced, e.g. `"a_-_team"` -> `"a   team"` before this
/// step) to one space and trims leading/trailing space.
///
/// This is the ONE normalization both the builder (over raw
/// underscore-separated Wikipedia titles and mixed-case `OpenAlex` topic
/// names) and [`crate::jargon`]'s `jargon.metaphor` channel (over a raw
/// source-text compound span, hyphens intact) run their input through —
/// see this module's header for why sharing the function itself, not
/// just the rule in prose, is what makes the two sides impossible to
/// drift apart.
///
/// # Examples
/// ```
/// assert_eq!(friction_packs::jargon_attest::normalize_compound("Data_fabric"), "data fabric");
/// assert_eq!(
///     friction_packs::jargon_attest::normalize_compound("Cross-domain_resonance"),
///     "cross domain resonance"
/// );
/// ```
#[must_use]
pub fn normalize_compound(raw: &str) -> String {
    raw.chars()
        .map(|c| if c == '_' || c == '-' { ' ' } else { c })
        .collect::<String>()
        .to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// `true` if `normalized`'s word count falls in [`MIN_WORDS`]..=[`MAX_WORDS`].
/// Only ever called on already-[`normalize_compound`]d strings.
#[must_use]
pub fn in_word_range(normalized: &str) -> bool {
    (MIN_WORDS..=MAX_WORDS).contains(&normalized.split_whitespace().count())
}

/// Hashes an already-[`normalize_compound`]d key with the fixed
/// [`HASH_SEED`] — the one hash both [`build_pack_bytes`] and
/// [`JargonAttestPack::is_attested`] use.
#[must_use]
pub fn hash_key(normalized: &str) -> u64 {
    xxhash_rust::xxh64::xxh64(normalized.as_bytes(), HASH_SEED)
}

/// 8-byte magic identifying this pack's `.bin` layout.
///
/// `JRGNFUS1` (jargon fuse filter, layout version 1). Bumped only if the
/// on-disk *layout* changes; the filter's semantic content can change (a
/// corpus refresh) without bumping this.
const MAGIC: &[u8; 8] = b"JRGNFUS1";

/// The fixed-length prefix every `.bin` starts with.
///
/// Three parts back to back: [`MAGIC`] (8 bytes), a little-endian key
/// count (8 bytes), then `xorf`'s own filter descriptor
/// ([`DmaSerializable::DESCRIPTOR_LEN`]). Everything after this is raw
/// fingerprint bytes.
const HEADER_LEN: usize = 8 + 8 + <BinaryFuse8 as DmaSerializable>::DESCRIPTOR_LEN;

/// The result of [`build_pack_bytes`]: the serialized `.bin` contents plus
/// the bookkeeping `corpus-tool jargon-attest` needs for its sidecar TOML
/// and summary output.
#[derive(Debug, Clone)]
pub struct BuiltPack {
    /// The full `.bin` file contents (header + descriptor + fingerprints).
    pub bytes: Vec<u8>,
    /// The number of distinct hashes actually inserted into the filter —
    /// may be one or two less than the input set's length if
    /// [`hash_key`] collided (see `hash_collisions`); this is the count
    /// [`JargonAttestPack::load`] cross-checks the sidecar's
    /// `[pack].key_count` against.
    pub key_count: u64,
    /// How many input keys were dropped because their hash collided with
    /// an already-inserted key's hash. Expected to be `0` for real input
    /// sizes (a 64-bit hash space makes this astronomically unlikely
    /// below tens of millions of keys) — recorded, not silently ignored,
    /// so a build that does hit one is visible in the builder's own
    /// summary output rather than a quietly undercounted pack.
    pub hash_collisions: u64,
}

/// Builds a `jargon-attest-v1` pack's `.bin` bytes from `normalized_keys`
/// — already [`normalize_compound`]d, already deduped and sorted (a
/// [`BTreeSet`]'s natural state).
///
/// Keys are hashed ([`hash_key`]) in `normalized_keys`'s own ascending
/// iteration order and fed to `xorf::BinaryFuse8::try_from` in that exact
/// order: the filter's construction is a pure function of `(key set, key
/// order)` (see this module's header), so iterating a sorted set in a
/// fixed direction is what makes two builds from the same corpus produce
/// bit-identical bytes.
///
/// # Errors
/// [`PackError::JargonAttestConstructionFailed`] if `xorf`'s bounded
/// retry loop never finds a working seed arrangement — see that variant's
/// own docs for how unlikely this is in practice.
pub fn build_pack_bytes(normalized_keys: &BTreeSet<String>) -> Result<BuiltPack, PackError> {
    let mut seen_hashes: BTreeSet<u64> = BTreeSet::new();
    let mut keys: Vec<u64> = Vec::with_capacity(normalized_keys.len());
    let mut hash_collisions = 0u64;
    for normalized in normalized_keys {
        let hash = hash_key(normalized);
        if seen_hashes.insert(hash) {
            keys.push(hash);
        } else {
            hash_collisions += 1;
        }
    }

    let filter = BinaryFuse8::try_from(&keys).map_err(|message| {
        PackError::JargonAttestConstructionFailed {
            message: message.to_string(),
        }
    })?;

    let mut descriptor = vec![0u8; <BinaryFuse8 as DmaSerializable>::DESCRIPTOR_LEN];
    filter.dma_copy_descriptor_to(&mut descriptor);
    let fingerprints = filter.dma_fingerprints();

    let key_count = keys.len();
    let mut bytes = Vec::with_capacity(HEADER_LEN + fingerprints.len());
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&u64::try_from(key_count).unwrap_or(u64::MAX).to_le_bytes());
    bytes.extend_from_slice(&descriptor);
    bytes.extend_from_slice(fingerprints);

    Ok(BuiltPack {
        bytes,
        key_count: key_count as u64,
        hash_collisions,
    })
}

/// A parsed, validated `jargon-attest-v1` pack: a zero-copy
/// [`BinaryFuse8Ref`] over the embedded `.bin`'s fingerprints, plus the
/// metadata [`JargonAttestPack::load`] cross-checked it against.
#[derive(Debug, Clone)]
pub struct JargonAttestPack {
    filter: BinaryFuse8Ref<'static>,
    key_count: u64,
    version: Box<str>,
}

impl JargonAttestPack {
    /// Parses `bin` (this pack's embedded `.bin`, `include_bytes!`'d by
    /// the caller — see [`crate::registry`]) and `sidecar_toml` (its
    /// `.toml` sidecar, `include_str!`'d the same way) into a validated
    /// pack.
    ///
    /// Validates: `bin` is at least [`HEADER_LEN`] bytes and starts with
    /// [`MAGIC`] (byte-length sanity); the sidecar's `[pack].version` is
    /// exactly `"jargon-attest-v1"`; the sidecar's `[pack].key_count`
    /// matches the `.bin`'s own embedded key count (the two files agree
    /// on what build they describe).
    ///
    /// # Errors
    /// [`PackError::JargonAttestTruncated`], [`PackError::JargonAttestBadMagic`],
    /// [`PackError::Toml`], [`PackError::JargonAttestVersionMismatch`], or
    /// [`PackError::JargonAttestKeyCountMismatch`] — see each variant's
    /// own docs.
    pub fn load(bin: &'static [u8], sidecar_toml: &str) -> Result<Self, PackError> {
        if bin.len() < HEADER_LEN {
            return Err(PackError::JargonAttestTruncated {
                len: bin.len(),
                min: HEADER_LEN,
            });
        }
        if &bin[..8] != MAGIC {
            return Err(PackError::JargonAttestBadMagic);
        }
        let key_count = u64::from_le_bytes(
            bin[8..16]
                .try_into()
                .expect("bin.len() >= HEADER_LEN guarantees 8 bytes at [8..16]"),
        );
        let descriptor = &bin[16..HEADER_LEN];
        let fingerprints = &bin[HEADER_LEN..];

        let sidecar: RawSidecar = toml::from_str(sidecar_toml).map_err(PackError::from)?;
        if sidecar.pack.version != "jargon-attest-v1" {
            return Err(PackError::JargonAttestVersionMismatch {
                found: sidecar.pack.version,
            });
        }
        if sidecar.pack.key_count != key_count {
            return Err(PackError::JargonAttestKeyCountMismatch {
                sidecar: sidecar.pack.key_count,
                embedded: key_count,
            });
        }

        let filter = BinaryFuse8Ref::from_dma(descriptor, fingerprints);
        Ok(Self {
            filter,
            key_count,
            version: sidecar.pack.version.into_boxed_str(),
        })
    }

    /// The number of keys embedded in the filter.
    #[must_use]
    pub const fn key_count(&self) -> u64 {
        self.key_count
    }

    /// The sidecar's own recorded `[pack].version` (always
    /// `"jargon-attest-v1"` for a pack that loaded successfully).
    #[must_use]
    pub fn version(&self) -> &str {
        &self.version
    }

    /// `true` if `normalized_compound` — already run through
    /// [`normalize_compound`] by the caller (see [`crate::jargon`]'s call
    /// site) — hashes into this filter.
    ///
    /// No word-count check here: a compound outside
    /// [`MIN_WORDS`]..=[`MAX_WORDS`] words was never inserted at build
    /// time, so it simply won't be found (modulo the filter's own ~0.4%
    /// false-positive rate, which is the collision direction this
    /// module's header already establishes as safe) — re-checking the
    /// range here would only reject that tiny false-positive slice, at
    /// the cost of a second range check that could itself drift from the
    /// builder's.
    #[must_use]
    pub fn is_attested(&self, normalized_compound: &str) -> bool {
        self.filter.contains(&hash_key(normalized_compound))
    }
}

/// The sidecar TOML's `[pack]` table — only the two fields this reader
/// cross-checks against the `.bin` header; every other field
/// (`hash_function`, `build_date`, `[pack.sources.*]`, ...) is
/// provenance for humans and intentionally not re-parsed here (not
/// `#[serde(deny_unknown_fields)]`, matching this crate's other pack
/// readers' tolerance of a free-form `[pack]` metadata table).
#[derive(Debug, serde::Deserialize)]
struct RawSidecarPack {
    version: String,
    key_count: u64,
}

#[derive(Debug, serde::Deserialize)]
struct RawSidecar {
    pack: RawSidecarPack,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_compound_folds_underscore_case_and_whitespace() {
        assert_eq!(normalize_compound("Data_fabric"), "data fabric");
        assert_eq!(normalize_compound("Semantic wells"), "semantic wells");
        assert_eq!(
            normalize_compound("Cross-domain_resonance"),
            "cross domain resonance"
        );
    }

    #[test]
    fn normalize_compound_collapses_runs_hyphen_and_underscore_mixed() {
        // "A_-_team" -> "A   team" (three chars, each folded to a space)
        // -> collapsed -> "a team".
        assert_eq!(normalize_compound("A_-_team"), "a team");
    }

    #[test]
    fn normalize_compound_is_idempotent() {
        let once = normalize_compound("Cross-domain_resonance");
        let twice = normalize_compound(&once);
        assert_eq!(once, twice);
    }

    #[test]
    fn in_word_range_matches_min_and_max() {
        assert!(!in_word_range("soup"));
        assert!(in_word_range("topical soup"));
        assert!(in_word_range("a very topical soup"));
        assert!(!in_word_range("a very very topical soup"));
    }

    #[test]
    fn hash_key_is_deterministic_and_seed_dependent() {
        assert_eq!(hash_key("data fabric"), hash_key("data fabric"));
        assert_ne!(hash_key("data fabric"), hash_key("service mesh"));
    }

    fn sample_keys() -> BTreeSet<String> {
        [
            "data fabric",
            "primordial soup",
            "resonance frequency",
            "service mesh",
            "inverted index",
            "color harmony",
        ]
        .into_iter()
        .map(String::from)
        .collect()
    }

    fn sample_sidecar(key_count: u64) -> String {
        format!("[pack]\nversion = \"jargon-attest-v1\"\nkey_count = {key_count}\n")
    }

    #[test]
    fn build_pack_bytes_round_trips_through_load_and_is_attested() {
        let keys = sample_keys();
        let built = build_pack_bytes(&keys).expect("small sample builds");
        assert_eq!(built.hash_collisions, 0);
        assert_eq!(built.key_count, keys.len() as u64);

        let bin: &'static [u8] = Box::leak(built.bytes.into_boxed_slice());
        let sidecar = sample_sidecar(built.key_count);
        let pack = JargonAttestPack::load(bin, &sidecar).expect("round-trips");

        for key in &keys {
            assert!(pack.is_attested(key), "{key} should be attested");
        }
        assert!(!pack.is_attested("semantic wells"));
        assert!(!pack.is_attested("cross domain resonance"));
        assert!(!pack.is_attested("topical soup"));
    }

    /// The determinism guarantee this module's header claims: building
    /// twice from the same (small, synthetic) input set produces
    /// bit-identical `.bin` bytes, because `xorf`'s seed-retry sequence is
    /// a pure function of a fixed initial state and this module always
    /// feeds keys in the same (`BTreeSet`-ascending) order.
    #[test]
    fn determinism_is_bit_identical_for_a_fixed_input_set() {
        let keys = sample_keys();
        let first = build_pack_bytes(&keys).expect("builds");
        let second = build_pack_bytes(&keys).expect("builds");
        assert_eq!(first.bytes, second.bytes);
        assert_eq!(first.key_count, second.key_count);
        assert_eq!(first.hash_collisions, second.hash_collisions);
    }

    #[test]
    fn load_rejects_truncated_bin() {
        let err = JargonAttestPack::load(b"short", &sample_sidecar(0)).unwrap_err();
        assert!(matches!(err, PackError::JargonAttestTruncated { .. }));
    }

    #[test]
    fn load_rejects_bad_magic() {
        let keys = sample_keys();
        let mut built = build_pack_bytes(&keys).expect("builds");
        built.bytes[0] = b'X';
        let bin: &'static [u8] = Box::leak(built.bytes.into_boxed_slice());
        let err = JargonAttestPack::load(bin, &sample_sidecar(built.key_count)).unwrap_err();
        assert!(matches!(err, PackError::JargonAttestBadMagic));
    }

    #[test]
    fn load_rejects_sidecar_version_mismatch() {
        let keys = sample_keys();
        let built = build_pack_bytes(&keys).expect("builds");
        let bin: &'static [u8] = Box::leak(built.bytes.into_boxed_slice());
        let bad_sidecar = "[pack]\nversion = \"jargon-attest-v0\"\nkey_count = 6\n";
        let err = JargonAttestPack::load(bin, bad_sidecar).unwrap_err();
        assert!(matches!(err, PackError::JargonAttestVersionMismatch { .. }));
    }

    #[test]
    fn load_rejects_sidecar_key_count_mismatch() {
        let keys = sample_keys();
        let built = build_pack_bytes(&keys).expect("builds");
        let bin: &'static [u8] = Box::leak(built.bytes.into_boxed_slice());
        let err = JargonAttestPack::load(bin, &sample_sidecar(999)).unwrap_err();
        assert!(matches!(
            err,
            PackError::JargonAttestKeyCountMismatch { .. }
        ));
    }
}
