//! `general-evidence-v1`: web-scale general-vocabulary attestation.
//!
//! Two [`xorf::BinaryFuse8`] filters in one `.bin` — a unigram membership
//! table and a bigram membership table — mined from a large general
//! corpus (`corpus-tool general-evidence mine`/`pack`; a Wikipedia dump
//! in the shipped build's own case, though this module has no opinion on
//! the source). Same shape as [`crate::jargon_attest`]: deterministic
//! `BTreeSet`-ordered construction, bounded retry inherited from `xorf`,
//! no false negatives.
//!
//! # NOT embedded, by design
//!
//! Unlike every other pack this crate ships, this one is never
//! `include_bytes!`'d. At ~25-45 MB it would roughly double this crate's
//! binary size for a signal most callers can already do without — the
//! packs that ARE embedded ([`crate::JARGON_ATTEST`], [`crate::DMS`], …)
//! already cover the load-bearing detection channels. [`general_evidence`]
//! resolves an OPTIONAL runtime artifact instead (see "Availability seam"
//! below), and its absence changes nothing else in this crate: every
//! existing embedded pack, and everything built from it, stays exactly as
//! it was before this module existed.
//!
//! # Two independent filters, one normalization
//!
//! [`normalize_word`] is the one word-shape rule both tables share: trim,
//! lowercase, then keep only if the result is ASCII-alphabetic and at
//! least 3 characters. [`build_pack_bytes`]'s caller (`corpus-tool
//! general-evidence pack`) applies this same shape (plus a `count >= 5`
//! floor) when it decides which mined words survive into the `BTreeSet`s
//! this module builds from; [`GeneralEvidencePack::has_word`] and
//! [`GeneralEvidencePack::has_bigram`] re-apply it to every query so an
//! unnormalized caller input (mixed case, punctuation-attached, a
//! 1-2-letter fragment) can never even reach the filter — it fails closed
//! to `false` before hashing anything, never a spurious lookup.
//!
//! # Header layout
//!
//! A fixed prefix, then two filter sections back to back — see
//! [`HEADER_LEN`] and the per-filter block comment on [`build_pack_bytes`]
//! for the exact byte layout. Every multi-byte field is `from_le_bytes`,
//! alignment-free (this workspace denies `unsafe_code`), matching
//! `crate::dms_bin`'s own convention. The header carries an 8-byte magic,
//! a `u16` format version, a 64-byte lowercase-hex sha256 of the two input
//! `BTreeSet`s' own content (this pack's "source metadata" — there is no
//! source TOML to hash, unlike [`crate::DMS`]/[`crate::FRAME`], so the
//! mined key sets themselves stand in for it), and each filter's own key
//! count — everything [`GeneralEvidencePack::validate`] needs to catch a
//! corrupt or truncated file without constructing anything.
//!
//! # Availability seam
//!
//! [`general_evidence`] resolves, once, in this order: an
//! [`install_general_evidence_bin`] override; else (native builds only —
//! `cfg(not(target_arch = "wasm32"))`, since wasm has no filesystem) the
//! file named by the `FRICTION_GENERAL_EVIDENCE` environment variable;
//! else `~/.cache/friction/general-evidence-en-v1.bin` (`$HOME` read via
//! `std::env::var_os`, no `dirs`-style dependency); else [`None`]. A
//! missing or corrupt candidate is skipped silently (no `eprintln!` — this
//! is library code, and this pack's whole point is that its absence is a
//! normal, unremarkable state, not a warning-worthy one); a caller that
//! wants a loud diagnosis of *why* a specific file didn't load calls
//! [`GeneralEvidencePack::validate`] on its bytes directly. The override
//! seam mirrors `friction_nlp::weights_install`'s `OverrideSlot` (leak
//! once, `OnceLock`-guarded, checked-before-leak) — reimplemented locally
//! rather than exposed across that crate's boundary for this one caller;
//! see [`InstallError::AlreadyInstalled`] for what "too late" this module
//! can and cannot detect.

use std::collections::BTreeSet;
use std::sync::OnceLock;

use xorf::{BinaryFuse8, BinaryFuse8Ref, DmaSerializable, Filter, FilterRef};

use crate::{PackError, Sha256};

/// The fixed xxh64 seed every normalized word/bigram key is hashed with.
///
/// Independent of [`crate::jargon_attest::HASH_SEED`] (a different pack,
/// a different key namespace; there is no reason the two should share a
/// seed). `b"GENEVID1"` read as a big-endian `u64`.
pub const HASH_SEED: u64 = 0x4745_4E45_5649_4431;

/// Hashes an already-[`normalize_word`]d key with the fixed [`HASH_SEED`].
///
/// `normalized` is a single word, or a `"word1 word2"` bigram key — the
/// one hash both [`build_pack_bytes`] and [`GeneralEvidencePack`]'s
/// queries use.
#[must_use]
pub fn hash_key(normalized: &str) -> u64 {
    xxhash_rust::xxh64::xxh64(normalized.as_bytes(), HASH_SEED)
}

/// Normalizes `raw` into this pack's canonical word shape.
///
/// Trim, then lowercase, then keep only if the result is ASCII-alphabetic
/// (`is_ascii_lowercase` on every byte, post-lowering) and at least 3
/// characters long — [`None`] otherwise.
///
/// The one normalization [`GeneralEvidencePack::has_word`]/
/// [`GeneralEvidencePack::has_bigram`] run every query word through
/// before it can even reach a hash lookup, and the shape `corpus-tool
/// general-evidence`'s mining/packing steps hold every surviving word to
/// (plus their own `count >= 5` floor, which this function has no
/// opinion on — it is a per-word shape check, not a corpus-frequency
/// one).
///
/// # Examples
/// ```
/// assert_eq!(friction_packs::general_evidence::normalize_word("  Utilize "), Some("utilize".to_string()));
/// assert_eq!(friction_packs::general_evidence::normalize_word("ok"), None, "below the 3-char floor");
/// assert_eq!(friction_packs::general_evidence::normalize_word("well-known"), None, "not ASCII-alphabetic");
/// ```
#[must_use]
pub fn normalize_word(raw: &str) -> Option<String> {
    let normalized = raw.trim().to_lowercase();
    if normalized.len() < 3 || !normalized.bytes().all(|b| b.is_ascii_lowercase()) {
        None
    } else {
        Some(normalized)
    }
}

/// The 8-byte magic every `general-evidence-v1` `.bin` starts with.
const MAGIC: &[u8; 8] = b"GENEVID1";

/// On-disk format version: bumped only if the layout changes shape.
const FORMAT_VERSION: u16 = 1;

/// A sha256 digest is recorded as lowercase hex (matching [`Sha256`]'s
/// `Display`), the same choice as `crate::dms_bin`'s header.
const SHA256_HEX_LEN: usize = 64;

/// The fixed-length prefix: magic + version + source sha256 hex + the two
/// filters' own key counts.
const HEADER_LEN: usize = 8 + 2 + SHA256_HEX_LEN + 8 + 8;

/// The result of [`build_pack_bytes`]: the serialized `.bin` contents plus
/// the bookkeeping `corpus-tool general-evidence pack` needs for its
/// `.meta.toml` sidecar and summary output.
#[derive(Debug, Clone)]
pub struct BuiltPack {
    /// The full `.bin` file contents.
    pub bytes: Vec<u8>,
    /// The lowercase-hex sha256 of the two input `BTreeSet`s' own content
    /// — the same value embedded in the `.bin`'s header as its "source
    /// metadata sha" (see this module's own docs).
    pub sha256_hex: String,
    /// The number of distinct unigram keys actually inserted into the
    /// unigram filter (may be less than `unigrams.len()` if [`hash_key`]
    /// collided — astronomically unlikely at real input sizes, same
    /// reasoning as [`crate::jargon_attest::build_pack_bytes`]).
    pub unigram_key_count: u64,
    /// The bigram filter's own key count — see `unigram_key_count`.
    pub bigram_key_count: u64,
}

/// Builds a `general-evidence-v1` pack's `.bin` bytes from `unigrams` and
/// `bigrams`.
///
/// Both already [`normalize_word`]-shaped (unigrams) or
/// `"word1 word2"`-shaped (bigrams, each half already `normalize_word`-
/// shaped), already deduped and sorted (a [`BTreeSet`]'s natural state),
/// and already past whatever corpus-frequency floor the caller applies
/// (this function has no opinion on frequency — see [`normalize_word`]'s
/// docs).
///
/// Byte layout (mirrors [`crate::jargon_attest::build_pack_bytes`]'s
/// single-filter version, extended to two): [`MAGIC`], [`FORMAT_VERSION`]
/// (`u16` LE), the source sha256 hex (64 ASCII bytes), the unigram and
/// bigram key counts (`u64` LE each), then per filter — unigram first,
/// bigram second — a `DmaSerializable::DESCRIPTOR_LEN`-byte descriptor
/// and a `u64` LE fingerprint-byte-length, then finally both filters'
/// raw fingerprint bytes back to back (unigram's, then bigram's). Writing
/// each filter's own fingerprint length right after its descriptor (never
/// relying on some derived length) is what lets [`GeneralEvidencePack::load`]
/// slice the trailing region unambiguously without touching `xorf`'s
/// internal descriptor encoding.
///
/// Each filter is built independently: keys are hashed ([`hash_key`]) in
/// `BTreeSet`-ascending order and fed to `xorf::BinaryFuse8::try_from` in
/// that order, so (per [`crate::jargon_attest`]'s own determinism
/// discussion, which applies unchanged to `xorf`'s construction here)
/// building from the same two input sets twice produces bit-identical
/// bytes.
///
/// # Errors
/// [`PackError::GeneralEvidenceConstructionFailed`] if `xorf`'s bounded
/// retry loop never finds a working seed arrangement for either filter —
/// see that variant's own docs for how unlikely this is in practice.
pub fn build_pack_bytes(
    unigrams: &BTreeSet<String>,
    bigrams: &BTreeSet<String>,
) -> Result<BuiltPack, PackError> {
    let sha256_hex = source_metadata_sha256_hex(unigrams, bigrams);
    let unigram_filter = build_filter(unigrams, "unigram")?;
    let bigram_filter = build_filter(bigrams, "bigram")?;

    let mut bytes = Vec::with_capacity(
        HEADER_LEN
            + 2 * (<BinaryFuse8 as DmaSerializable>::DESCRIPTOR_LEN + 8)
            + unigram_filter.fingerprints.len()
            + bigram_filter.fingerprints.len(),
    );
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    debug_assert_eq!(
        sha256_hex.len(),
        SHA256_HEX_LEN,
        "Sha256::Display is always 64 hex chars"
    );
    bytes.extend_from_slice(sha256_hex.as_bytes());
    bytes.extend_from_slice(&unigram_filter.key_count.to_le_bytes());
    bytes.extend_from_slice(&bigram_filter.key_count.to_le_bytes());
    bytes.extend_from_slice(&unigram_filter.descriptor);
    bytes.extend_from_slice(
        &u64::try_from(unigram_filter.fingerprints.len())
            .expect("fingerprint byte length fits u64")
            .to_le_bytes(),
    );
    bytes.extend_from_slice(&bigram_filter.descriptor);
    bytes.extend_from_slice(
        &u64::try_from(bigram_filter.fingerprints.len())
            .expect("fingerprint byte length fits u64")
            .to_le_bytes(),
    );
    bytes.extend_from_slice(&unigram_filter.fingerprints);
    bytes.extend_from_slice(&bigram_filter.fingerprints);

    Ok(BuiltPack {
        bytes,
        sha256_hex,
        unigram_key_count: unigram_filter.key_count,
        bigram_key_count: bigram_filter.key_count,
    })
}

/// One filter's built pieces: [`build_pack_bytes`] builds the unigram and
/// bigram filters through this same helper, `table` naming which one (for
/// [`PackError::GeneralEvidenceConstructionFailed`]'s message only).
struct BuiltFilter {
    descriptor: Vec<u8>,
    fingerprints: Vec<u8>,
    key_count: u64,
}

fn build_filter(keys: &BTreeSet<String>, table: &'static str) -> Result<BuiltFilter, PackError> {
    let mut seen_hashes: BTreeSet<u64> = BTreeSet::new();
    let mut hashed: Vec<u64> = Vec::with_capacity(keys.len());
    for key in keys {
        let hash = hash_key(key);
        if seen_hashes.insert(hash) {
            hashed.push(hash);
        }
    }

    let filter = BinaryFuse8::try_from(&hashed).map_err(|message| {
        PackError::GeneralEvidenceConstructionFailed {
            table,
            message: message.to_string(),
        }
    })?;

    let mut descriptor = vec![0u8; <BinaryFuse8 as DmaSerializable>::DESCRIPTOR_LEN];
    filter.dma_copy_descriptor_to(&mut descriptor);
    let fingerprints = filter.dma_fingerprints().to_vec();
    let key_count = u64::try_from(hashed.len()).expect("key count fits u64");

    Ok(BuiltFilter {
        descriptor,
        fingerprints,
        key_count,
    })
}

/// The sha256 embedded in the header as this pack's "source metadata": a
/// deterministic hash of the two input `BTreeSet`s' own sorted content
/// (`unigrams` joined by `\n`, a `\0BIGRAMS\0` separator, then `bigrams`
/// joined by `\n`) — there is no source TOML to hash the way
/// [`crate::DMS`]/[`crate::FRAME`] do, so the mined key sets themselves
/// stand in for it. A pure function of the two sets' content: rebuilding
/// from the same sets reproduces the same digest (pinned by this module's
/// own determinism test).
fn source_metadata_sha256_hex(unigrams: &BTreeSet<String>, bigrams: &BTreeSet<String>) -> String {
    let mut buf = Vec::new();
    for word in unigrams {
        buf.extend_from_slice(word.as_bytes());
        buf.push(b'\n');
    }
    buf.extend_from_slice(b"\0BIGRAMS\0");
    for bigram in bigrams {
        buf.extend_from_slice(bigram.as_bytes());
        buf.push(b'\n');
    }
    Sha256::of_bytes(&buf).to_string()
}

/// The borrowed sections [`parse_sections`] splits a `.bin`'s bytes into —
/// generic over the input slice's lifetime so [`GeneralEvidencePack::validate`]
/// can run the exact same structural checks [`GeneralEvidencePack::load`]
/// does over a borrowed (not necessarily `'static`) slice.
struct Sections<'a> {
    source_sha256_hex: &'a str,
    unigram_key_count: u64,
    bigram_key_count: u64,
    unigram_descriptor: &'a [u8],
    unigram_fingerprints: &'a [u8],
    bigram_descriptor: &'a [u8],
    bigram_fingerprints: &'a [u8],
}

/// A cursor over the artifact's bytes, erroring (never panicking) on
/// truncation — same shape as `crate::human_evidence`'s private `Reader`,
/// duplicated locally rather than shared across the two modules'
/// otherwise-unrelated parsers (that module's own `write_string_table`
/// comment documents this crate's convention for small format-local
/// helpers).
struct Reader<'a> {
    bytes: &'a [u8],
}

impl<'a> Reader<'a> {
    const fn take(&mut self, len: usize, section: &'static str) -> Result<&'a [u8], PackError> {
        if self.bytes.len() < len {
            return Err(PackError::GeneralEvidenceTruncated { section });
        }
        let (taken, rest) = self.bytes.split_at(len);
        self.bytes = rest;
        Ok(taken)
    }

    fn u16(&mut self, section: &'static str) -> Result<u16, PackError> {
        Ok(u16::from_le_bytes(
            self.take(2, section)?.try_into().expect("2-byte split"),
        ))
    }

    fn u64(&mut self, section: &'static str) -> Result<u64, PackError> {
        Ok(u64::from_le_bytes(
            self.take(8, section)?.try_into().expect("8-byte split"),
        ))
    }
}

/// Parses `bin`'s header and slices out both filters' descriptor and
/// fingerprint sections, without constructing anything — the shared core
/// [`GeneralEvidencePack::load`] and [`GeneralEvidencePack::validate`]
/// both run.
///
/// # Errors
/// [`PackError::GeneralEvidenceTruncated`] if `bin` runs out of bytes
/// (mid-header, mid-section, or trailing bytes left over after both
/// filters' sections are accounted for) or if the source sha256 hex
/// section isn't well-formed hex; [`PackError::GeneralEvidenceBadMagic`]
/// if it doesn't start with [`MAGIC`]; [`PackError::GeneralEvidenceUnsupportedVersion`]
/// if its recorded format version isn't [`FORMAT_VERSION`].
fn parse_sections(bin: &[u8]) -> Result<Sections<'_>, PackError> {
    let mut r = Reader { bytes: bin };
    let magic = r.take(8, "header")?;
    if magic != MAGIC {
        return Err(PackError::GeneralEvidenceBadMagic);
    }
    let version = r.u16("header")?;
    if version != FORMAT_VERSION {
        return Err(PackError::GeneralEvidenceUnsupportedVersion(version));
    }
    let sha_bytes = r.take(SHA256_HEX_LEN, "source_sha256_hex")?;
    let source_sha256_hex =
        std::str::from_utf8(sha_bytes).map_err(|_| PackError::GeneralEvidenceTruncated {
            section: "source_sha256_hex",
        })?;
    Sha256::parse_hex(source_sha256_hex).map_err(|_| PackError::GeneralEvidenceTruncated {
        section: "source_sha256_hex",
    })?;

    let unigram_key_count = r.u64("header")?;
    let bigram_key_count = r.u64("header")?;

    let unigram_descriptor = r.take(
        <BinaryFuse8 as DmaSerializable>::DESCRIPTOR_LEN,
        "unigram descriptor",
    )?;
    let unigram_fingerprints_len = r.u64("unigram fingerprints length")?;
    let bigram_descriptor = r.take(
        <BinaryFuse8 as DmaSerializable>::DESCRIPTOR_LEN,
        "bigram descriptor",
    )?;
    let bigram_fingerprints_len = r.u64("bigram fingerprints length")?;

    let unigram_fingerprints = r.take(
        usize::try_from(unigram_fingerprints_len).map_err(|_| {
            PackError::GeneralEvidenceTruncated {
                section: "unigram fingerprints",
            }
        })?,
        "unigram fingerprints",
    )?;
    let bigram_fingerprints = r.take(
        usize::try_from(bigram_fingerprints_len).map_err(|_| {
            PackError::GeneralEvidenceTruncated {
                section: "bigram fingerprints",
            }
        })?,
        "bigram fingerprints",
    )?;
    if !r.bytes.is_empty() {
        return Err(PackError::GeneralEvidenceTruncated {
            section: "trailing bytes",
        });
    }

    Ok(Sections {
        source_sha256_hex,
        unigram_key_count,
        bigram_key_count,
        unigram_descriptor,
        unigram_fingerprints,
        bigram_descriptor,
        bigram_fingerprints,
    })
}

/// A parsed, validated `general-evidence-v1` pack: two zero-copy
/// [`BinaryFuse8Ref`]s (unigram, bigram) over one `.bin`'s fingerprint
/// bytes.
///
/// Always the `'static`-borrowing zero-copy shape, never an owned
/// representation — unlike [`crate::human_evidence::HumanEvidencePack`],
/// this pack has no embedded-artifact call site that would need an owned
/// `load` for a non-`'static` slice (see this module's own "Availability
/// seam" docs: every resolution path either leaks its bytes or was never
/// going to reclaim them before process exit anyway).
#[derive(Debug, Clone)]
pub struct GeneralEvidencePack {
    unigram: BinaryFuse8Ref<'static>,
    bigram: BinaryFuse8Ref<'static>,
    unigram_key_count: u64,
    bigram_key_count: u64,
    source_sha256_hex: Box<str>,
}

impl GeneralEvidencePack {
    /// Parses `bin` (this pack's `.bin`, `'static` because every resolved
    /// source either leaks — see [`install_general_evidence_bin`] and
    /// [`general_evidence`]'s file-loading path — or was `include_bytes!`'d,
    /// though this pack is never embedded in practice) into a validated
    /// pack.
    ///
    /// # Errors
    /// See [`parse_sections`]'s docs — every error this can return comes
    /// from that shared structural check.
    pub fn load(bin: &'static [u8]) -> Result<Self, PackError> {
        let sections = parse_sections(bin)?;
        Ok(Self {
            unigram: BinaryFuse8Ref::from_dma(
                sections.unigram_descriptor,
                sections.unigram_fingerprints,
            ),
            bigram: BinaryFuse8Ref::from_dma(
                sections.bigram_descriptor,
                sections.bigram_fingerprints,
            ),
            unigram_key_count: sections.unigram_key_count,
            bigram_key_count: sections.bigram_key_count,
            source_sha256_hex: sections.source_sha256_hex.into(),
        })
    }

    /// Runs [`Self::load`]'s exact structural validation over `bin`
    /// without requiring a `'static` slice and without keeping anything
    /// around afterward — the loud check [`general_evidence`]'s silent
    /// resolution intentionally skips (see this module's own "Availability
    /// seam" docs). A caller staging a freshly downloaded/copied `.bin`
    /// calls this first to fail loudly on a truncated download, a bad
    /// magic, or an unsupported format version, instead of finding out
    /// only when [`general_evidence`] silently returns [`None`] later.
    ///
    /// # Errors
    /// See [`parse_sections`]'s docs.
    pub fn validate(bin: &[u8]) -> Result<(), PackError> {
        parse_sections(bin).map(|_| ())
    }

    /// The number of keys embedded in the unigram filter.
    #[must_use]
    pub const fn unigram_key_count(&self) -> u64 {
        self.unigram_key_count
    }

    /// The number of keys embedded in the bigram filter.
    #[must_use]
    pub const fn bigram_key_count(&self) -> u64 {
        self.bigram_key_count
    }

    /// This pack's "source metadata" sha256 — see this module's own docs
    /// for what it hashes.
    #[must_use]
    pub fn source_sha256_hex(&self) -> &str {
        &self.source_sha256_hex
    }

    /// `true` if `word`, run through [`normalize_word`], hashes into the
    /// unigram filter. `false` (never a lookup) if `word` doesn't survive
    /// normalization — a 2-letter fragment or anything not ASCII-alphabetic
    /// was never a candidate key at build time either.
    #[must_use]
    pub fn has_word(&self, word: &str) -> bool {
        normalize_word(word).is_some_and(|normalized| self.unigram.contains(&hash_key(&normalized)))
    }

    /// `true` if `w1` and `w2`, each independently run through
    /// [`normalize_word`] then joined as `"w1 w2"`, hashes into the bigram
    /// filter. `false` (never a lookup) if either word fails normalization
    /// — see [`Self::has_word`]'s own docs for why that's a closed-world
    /// "no" rather than an error.
    #[must_use]
    pub fn has_bigram(&self, w1: &str, w2: &str) -> bool {
        let (Some(a), Some(b)) = (normalize_word(w1), normalize_word(w2)) else {
            return false;
        };
        let key = format!("{a} {b}");
        self.bigram.contains(&hash_key(&key))
    }
}

/// Errors installing pre-init `general-evidence-v1` bytes — see
/// [`install_general_evidence_bin`].
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum InstallError {
    /// [`general_evidence`] was already resolved (from an earlier install,
    /// or its own first touch — the file/cache-path resolution runs
    /// exactly once and is cached) by the time
    /// [`install_general_evidence_bin`] ran.
    ///
    /// Detection is best-effort, not atomic — the same caveat
    /// `friction_packs::registry`'s `install_dms_index_bin` documents for
    /// its own `LazyLock::get` check: a concurrent first touch of
    /// [`general_evidence`] between the check and the store can still
    /// slip past it. Contract: call this before anything (on any thread)
    /// calls [`general_evidence`].
    #[error(
        "general-evidence-v1 was already resolved (installed override, or general_evidence()'s \
         own first touch) before this call — install_general_evidence_bin must run before \
         anything calls `friction_packs::general_evidence()`"
    )]
    AlreadyInstalled,
}

/// This pack's pre-init override slot: leaked bytes, set at most once —
/// the same shape as `friction_nlp::weights_install::OverrideSlot`
/// (checked-before-leak, `'static` because a zero-copy
/// [`BinaryFuse8Ref`] needs a `'static` borrow to live in a
/// [`GeneralEvidencePack`] with no lifetime parameter), reimplemented
/// locally rather than exposed across that crate's boundary for this one
/// caller — see this module's own "Availability seam" docs.
struct OverrideSlot(OnceLock<&'static [u8]>);

impl OverrideSlot {
    const fn new() -> Self {
        Self(OnceLock::new())
    }

    fn set(&self, bytes: Vec<u8>) -> Result<(), InstallError> {
        if self.0.get().is_some() {
            return Err(InstallError::AlreadyInstalled);
        }
        let leaked: &'static [u8] = Vec::leak(bytes);
        self.0
            .set(leaked)
            .map_err(|_| InstallError::AlreadyInstalled)
    }

    fn get(&self) -> Option<&'static [u8]> {
        self.0.get().copied()
    }
}

static OVERRIDE: OverrideSlot = OverrideSlot::new();

/// [`general_evidence`]'s cache: resolved (or definitively absent) at
/// most once per process.
static RESOLVED: OnceLock<Option<GeneralEvidencePack>> = OnceLock::new();

/// Installs `bytes` as the `general-evidence-v1` pack, in place of the
/// default file-based resolution.
///
/// The seam a caller (an embedder that fetched the artifact itself, a
/// test harness) uses to supply it directly instead of relying on
/// `FRICTION_GENERAL_EVIDENCE`/the `~/.cache` fallback.
///
/// # Errors
/// [`InstallError::AlreadyInstalled`] if [`general_evidence`] was already
/// resolved (by an earlier install or its own first touch) before this
/// call — see that variant's own docs.
pub fn install_general_evidence_bin(bytes: Vec<u8>) -> Result<(), InstallError> {
    if RESOLVED.get().is_some() {
        return Err(InstallError::AlreadyInstalled);
    }
    OVERRIDE.set(bytes)
}

/// The resolved `general-evidence-v1` pack, if one is available.
///
/// See this module's own "Availability seam" docs for the resolution
/// order. Resolved once and cached; every call after the first returns
/// the same answer for the life of the process.
#[must_use]
pub fn general_evidence() -> Option<&'static GeneralEvidencePack> {
    RESOLVED.get_or_init(resolve).as_ref()
}

/// [`general_evidence`]'s resolution, run at most once
/// ([`RESOLVED`]'s `get_or_init`).
fn resolve() -> Option<GeneralEvidencePack> {
    if let Some(bytes) = OVERRIDE.get() {
        return GeneralEvidencePack::load(bytes).ok();
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let env_path = std::env::var_os("FRICTION_GENERAL_EVIDENCE").map(std::path::PathBuf::from);
        let cache_path = default_cache_path();
        resolve_from_paths(env_path, cache_path)
    }
    #[cfg(target_arch = "wasm32")]
    {
        None
    }
}

/// The file-based half of [`resolve`], factored out so it can be
/// exercised directly against synthetic paths (see this module's own
/// tests) without touching real environment variables or `$HOME` — a
/// real call site only ever reaches this through [`resolve`], which
/// supplies the real `FRICTION_GENERAL_EVIDENCE`/`$HOME`-derived paths.
///
/// A set `env_path` is AUTHORITATIVE: explicit intent beats the ambient
/// cache, so when the variable names a path that doesn't exist or fails
/// [`GeneralEvidencePack::load`], the answer is [`None`] — never a
/// fall-through to `cache_path`. This is also the documented disable
/// switch (`FRICTION_GENERAL_EVIDENCE=/nonexistent` guarantees the
/// channel stays dark, which byte-exact test suites rely on). With no
/// env var at all, the cache candidate is tried; a corrupt or missing
/// cache file silently resolves to [`None`], per this module's own
/// "Availability seam" docs.
#[cfg(not(target_arch = "wasm32"))]
fn resolve_from_paths(
    env_path: Option<std::path::PathBuf>,
    cache_path: Option<std::path::PathBuf>,
) -> Option<GeneralEvidencePack> {
    env_path.map_or_else(
        || cache_path.and_then(|path| load_from_path(&path)),
        |path| load_from_path(&path),
    )
}

/// Reads and leaks `path`'s bytes, then parses them — [`None`] if the
/// file can't be read (most commonly: doesn't exist) or fails
/// [`GeneralEvidencePack::load`]. The leak is bounded by the process
/// lifetime, same as [`OverrideSlot::set`]'s.
#[cfg(not(target_arch = "wasm32"))]
fn load_from_path(path: &std::path::Path) -> Option<GeneralEvidencePack> {
    let bytes = std::fs::read(path).ok()?;
    let leaked: &'static [u8] = Vec::leak(bytes);
    GeneralEvidencePack::load(leaked).ok()
}

/// The dirs-free `~/.cache/friction/general-evidence-en-v1.bin` fallback
/// path, or [`None`] if `$HOME` isn't set (or is empty) in this process's
/// environment.
#[cfg(not(target_arch = "wasm32"))]
fn default_cache_path() -> Option<std::path::PathBuf> {
    let home = std::env::var_os("HOME")?;
    if home.is_empty() {
        return None;
    }
    Some(
        std::path::Path::new(&home)
            .join(".cache")
            .join("friction")
            .join("general-evidence-en-v1.bin"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_word_trims_lowercases_and_enforces_shape() {
        assert_eq!(normalize_word("  Utilize "), Some("utilize".to_string()));
        assert_eq!(normalize_word("OK"), None, "below the 3-char floor");
        assert_eq!(normalize_word("a1b"), None, "not ascii-alphabetic");
        assert_eq!(normalize_word("naïve"), None, "not ascii-alphabetic");
        assert_eq!(normalize_word(""), None);
    }

    #[test]
    fn hash_key_is_deterministic_and_seed_dependent() {
        assert_eq!(hash_key("utilize"), hash_key("utilize"));
        assert_ne!(hash_key("utilize"), hash_key("leverage"));
        assert_ne!(
            hash_key("utilize"),
            crate::jargon_attest::hash_key("utilize"),
            "a different fixed seed from jargon-attest-v1's"
        );
    }

    fn sample_unigrams() -> BTreeSet<String> {
        ["utilize", "leverage", "synergy", "paradigm", "robust"]
            .into_iter()
            .map(String::from)
            .collect()
    }

    fn sample_bigrams() -> BTreeSet<String> {
        ["utilize this", "leverage synergy", "robust paradigm"]
            .into_iter()
            .map(String::from)
            .collect()
    }

    #[test]
    fn build_pack_bytes_round_trips_through_load_and_has_word_has_bigram() {
        let unigrams = sample_unigrams();
        let bigrams = sample_bigrams();
        let built = build_pack_bytes(&unigrams, &bigrams).expect("small sample builds");
        assert_eq!(built.unigram_key_count, unigrams.len() as u64);
        assert_eq!(built.bigram_key_count, bigrams.len() as u64);

        let bin: &'static [u8] = Box::leak(built.bytes.into_boxed_slice());
        let pack = GeneralEvidencePack::load(bin).expect("round-trips");
        assert_eq!(pack.source_sha256_hex(), built.sha256_hex);
        assert_eq!(pack.unigram_key_count(), unigrams.len() as u64);
        assert_eq!(pack.bigram_key_count(), bigrams.len() as u64);

        for word in &unigrams {
            assert!(pack.has_word(word), "{word} should be present");
        }
        assert!(pack.has_word("UTILIZE"), "has_word normalizes case");
        assert!(
            pack.has_word("  leverage  "),
            "has_word normalizes whitespace"
        );
        assert!(!pack.has_word("unattested"));
        assert!(
            !pack.has_word("ok"),
            "below the 3-char floor: false, never a lookup"
        );

        for bigram in &bigrams {
            let (w1, w2) = bigram
                .split_once(' ')
                .expect("sample bigrams are two words");
            assert!(pack.has_bigram(w1, w2), "{bigram} should be present");
        }
        assert!(
            !pack.has_bigram("utilize", "leverage"),
            "not one of the built bigrams"
        );
        assert!(
            !pack.has_bigram("ok", "leverage"),
            "left word below the 3-char floor"
        );
    }

    #[test]
    fn build_pack_bytes_is_deterministic() {
        let unigrams = sample_unigrams();
        let bigrams = sample_bigrams();
        let a = build_pack_bytes(&unigrams, &bigrams).expect("builds");
        let b = build_pack_bytes(&unigrams, &bigrams).expect("builds");
        assert_eq!(a.bytes, b.bytes);
        assert_eq!(a.sha256_hex, b.sha256_hex);
    }

    #[test]
    fn build_pack_bytes_handles_empty_sets() {
        let built = build_pack_bytes(&BTreeSet::new(), &BTreeSet::new()).expect("empty sets build");
        assert_eq!(built.unigram_key_count, 0);
        assert_eq!(built.bigram_key_count, 0);
        let bin: &'static [u8] = Box::leak(built.bytes.into_boxed_slice());
        let pack = GeneralEvidencePack::load(bin).expect("empty pack loads");
        assert!(!pack.has_word("utilize"));
        assert!(!pack.has_bigram("utilize", "this"));
    }

    #[test]
    fn load_rejects_truncated_bin() {
        let err = GeneralEvidencePack::load(b"short").unwrap_err();
        assert!(matches!(err, PackError::GeneralEvidenceTruncated { .. }));
    }

    #[test]
    fn load_rejects_bad_magic() {
        let built = build_pack_bytes(&sample_unigrams(), &sample_bigrams()).expect("builds");
        let mut bytes = built.bytes;
        bytes[0] = b'X';
        let bin: &'static [u8] = Box::leak(bytes.into_boxed_slice());
        let err = GeneralEvidencePack::load(bin).unwrap_err();
        assert!(matches!(err, PackError::GeneralEvidenceBadMagic));
    }

    #[test]
    fn load_rejects_unsupported_version() {
        let built = build_pack_bytes(&sample_unigrams(), &sample_bigrams()).expect("builds");
        let mut bytes = built.bytes;
        bytes[8] = 0xFF;
        bytes[9] = 0xFF;
        let bin: &'static [u8] = Box::leak(bytes.into_boxed_slice());
        let err = GeneralEvidencePack::load(bin).unwrap_err();
        assert!(matches!(
            err,
            PackError::GeneralEvidenceUnsupportedVersion(_)
        ));
    }

    #[test]
    fn load_rejects_truncated_fingerprint_section() {
        let built = build_pack_bytes(&sample_unigrams(), &sample_bigrams()).expect("builds");
        let mut bytes = built.bytes;
        bytes.truncate(bytes.len() - 3);
        let bin: &'static [u8] = Box::leak(bytes.into_boxed_slice());
        let err = GeneralEvidencePack::load(bin).unwrap_err();
        assert!(matches!(err, PackError::GeneralEvidenceTruncated { .. }));
    }

    #[test]
    fn load_rejects_trailing_bytes() {
        let built = build_pack_bytes(&sample_unigrams(), &sample_bigrams()).expect("builds");
        let mut bytes = built.bytes;
        bytes.push(0);
        let bin: &'static [u8] = Box::leak(bytes.into_boxed_slice());
        let err = GeneralEvidencePack::load(bin).unwrap_err();
        assert!(matches!(
            err,
            PackError::GeneralEvidenceTruncated {
                section: "trailing bytes"
            }
        ));
    }

    #[test]
    fn validate_agrees_with_load_without_requiring_a_static_slice() {
        let built = build_pack_bytes(&sample_unigrams(), &sample_bigrams()).expect("builds");
        // A plain owned Vec, deliberately not leaked to 'static: `validate`
        // must not require that.
        assert!(GeneralEvidencePack::validate(&built.bytes).is_ok());
        assert!(GeneralEvidencePack::validate(b"short").is_err());
    }

    /// The override-slot install seam's happy path and its rejection of a
    /// second install — exercised on a *fresh* local instance rather than
    /// the module's real global [`OVERRIDE`]/[`RESOLVED`] statics, so this
    /// test can't be order-dependent on any other test in this binary
    /// that might touch [`general_evidence`] (same isolation
    /// `friction_nlp::weights_install`'s own `OverrideSlot` test uses).
    #[test]
    fn override_slot_rejects_a_second_install() {
        let slot = OverrideSlot::new();
        assert!(slot.get().is_none());
        slot.set(vec![1, 2, 3]).expect("first install succeeds");
        assert_eq!(slot.get(), Some(&[1u8, 2, 3][..]));
        let err = slot.set(vec![4, 5, 6]).unwrap_err();
        assert!(matches!(err, InstallError::AlreadyInstalled));
        assert_eq!(
            slot.get(),
            Some(&[1u8, 2, 3][..]),
            "first install's bytes still served"
        );
    }

    /// The full public seam end to end: [`install_general_evidence_bin`]
    /// then [`general_evidence`] then a second install rejected — this is
    /// the ONE test in this module allowed to touch the real global
    /// [`OVERRIDE`]/[`RESOLVED`] statics (see [`override_slot_rejects_a_second_install`]
    /// for why every other install-seam test uses a fresh local instance
    /// instead).
    #[test]
    fn install_general_evidence_bin_then_general_evidence_then_double_install_rejected() {
        let built = build_pack_bytes(&sample_unigrams(), &sample_bigrams()).expect("builds");
        install_general_evidence_bin(built.bytes).expect("first install succeeds");

        let pack = general_evidence().expect("installed pack resolves");
        assert!(pack.has_word("utilize"));
        assert!(pack.has_bigram("utilize", "this"));

        let err = install_general_evidence_bin(vec![1, 2, 3]).unwrap_err();
        assert!(matches!(err, InstallError::AlreadyInstalled));
    }

    #[test]
    fn resolve_from_paths_treats_a_set_env_candidate_as_authoritative() {
        let dir = tempfile_dir();
        let built = build_pack_bytes(&sample_unigrams(), &sample_bigrams()).expect("builds");
        let cache_path = dir.join("cache.bin");
        std::fs::write(&cache_path, &built.bytes).expect("write cache candidate");

        // Set-but-missing env path: the cache must NOT be consulted —
        // this is the disable switch byte-exact suites depend on.
        let missing_env = dir.join("does-not-exist.bin");
        assert!(resolve_from_paths(Some(missing_env), Some(cache_path.clone())).is_none());

        // No env var at all: the cache candidate is used.
        let pack = resolve_from_paths(None, Some(cache_path)).expect("cache candidate loads");
        assert!(pack.has_word("utilize"));
    }

    #[test]
    fn resolve_from_paths_answers_none_for_a_corrupt_env_candidate() {
        let dir = tempfile_dir();
        let built = build_pack_bytes(&sample_unigrams(), &sample_bigrams()).expect("builds");
        let env_path = dir.join("env.bin");
        std::fs::write(&env_path, b"not a real pack").expect("write corrupt env candidate");
        let cache_path = dir.join("cache.bin");
        std::fs::write(&cache_path, &built.bytes).expect("write cache candidate");

        // The explicit env candidate is authoritative even when corrupt:
        // silently None, never a fall-through to the ambient cache.
        assert!(resolve_from_paths(Some(env_path), Some(cache_path)).is_none());
    }

    #[test]
    fn resolve_from_paths_is_none_when_no_candidate_exists() {
        let dir = tempfile_dir();
        assert!(resolve_from_paths(Some(dir.join("a.bin")), Some(dir.join("b.bin"))).is_none());
        assert!(resolve_from_paths(None, None).is_none());
    }

    /// A minimal dependency-free temp-directory helper: this crate has no
    /// `tempfile` dev-dependency (unlike `corpus-tool`), and these tests
    /// only need one throwaway directory each, cleaned up on drop.
    fn tempfile_dir() -> TempDir {
        let mut path = std::env::temp_dir();
        let unique = format!(
            "friction-general-evidence-test-{}-{:?}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time is after the epoch")
                .as_nanos()
        );
        path.push(unique);
        std::fs::create_dir_all(&path).expect("create temp dir");
        TempDir(path)
    }

    struct TempDir(std::path::PathBuf);

    impl std::ops::Deref for TempDir {
        type Target = std::path::Path;
        fn deref(&self) -> &std::path::Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}
