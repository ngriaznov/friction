//! The unified pack-registry entry point: loads a pack's TOML bytes into
//! its typed, parsed form, alongside its declared version and a recorded
//! sha256 of the exact source bytes it was parsed from.

use std::sync::LazyLock;

use crate::attestation::AttestationPack;
use crate::dms::DmsIndex;
use crate::dms_bin::DmsIndexView;
use crate::inventory::InventoryPack;
use crate::jargon::JargonPack;
use crate::jargon_attest::JargonAttestPack;
use crate::register::RegisterPack;
use crate::validate::validate;
use crate::{PackError, Sha256};

/// A parsed pack, alongside its declared version string and the sha256 of
/// the exact source bytes it was parsed from.
#[derive(Debug, Clone)]
pub struct LoadedPack<T> {
    pub pack: T,
    pub version: Box<str>,
    pub sha256: Sha256,
}

/// The embedded `inventory-v1.toml` source — 32 KB, cheap to embed, like
/// `ENVELOPE_V2_TOML`.
const INVENTORY_V1_TOML: &str = include_str!("../packs/inventory-v1.toml");

/// The embedded derived DMS artifact — every stream's suffix automaton
/// pre-built by `corpus-tool dms-pack` and serialized flat (see
/// [`crate::dms_bin`]'s module docs for the layout and why). The source
/// `dms-index-v1.toml` stays in `packs/` as the audited artifact
/// `corpus-tool index` writes, but is no longer embedded or parsed at
/// runtime: this crate's `dms_bin` drift test re-packs it and compares
/// bytes, so the two cannot silently diverge.
const DMS_INDEX_V1_BIN: &[u8] = include_bytes!("../packs/dms-index-v1.bin");

/// The built-in inventory pack, parsed once from the embedded
/// `inventory-v1.toml` and reused for the life of the process.
///
/// # Panics
/// Panics if the embedded `inventory-v1.toml` fails to parse, or if it
/// parses but fails audit (disjointness/closure/frequency-hygiene/
/// guard-token-exposure) — a bug in this crate's own vendored data
/// (covered by this crate's inventory and validate tests), not a
/// condition any caller can recover from by retrying.
pub static INVENTORY: LazyLock<LoadedPack<InventoryPack>> = LazyLock::new(|| {
    let pack = InventoryPack::parse(INVENTORY_V1_TOML)
        .expect("embedded inventory-v1.toml must parse: see this crate's inventory/validate tests");
    let violations = validate(&pack);
    assert!(
        violations.is_empty(),
        "embedded inventory-v1.toml failed audit: {violations:#?}"
    );
    LoadedPack {
        version: pack.version().into(),
        sha256: Sha256::of_bytes(INVENTORY_V1_TOML.as_bytes()),
        pack,
    }
});

/// Parses a `dms-index-v1`-shaped pack's TOML bytes (loaded by the caller
/// — today, `corpus-tool index --calibrate`) into a [`LoadedPack`], the
/// same `sha256`-recording discipline [`INVENTORY`] uses.
///
/// This is the TOML-source path, used offline by `corpus-tool`; the
/// runtime detection pass reads [`DMS`], the derived binary artifact's
/// zero-copy view, instead.
///
/// # Errors
/// Returns [`PackError`] if `toml` does not parse as a valid
/// `dms-index-v1` pack — see [`DmsIndex::parse`].
pub fn load_dms_pack(toml: &str) -> Result<LoadedPack<DmsIndex>, PackError> {
    let pack = DmsIndex::parse(toml)?;
    Ok(LoadedPack {
        version: "dms-index-v1".into(),
        sha256: Sha256::of_bytes(toml.as_bytes()),
        pack,
    })
}

/// The built-in DMS index, viewed zero-copy over the embedded derived
/// artifact and reused for the life of the process.
///
/// This used to parse the ~2.5 MB `dms-index-v1.toml` and build every
/// suffix automaton on first touch — the whole of `friction`'s measured
/// fixed startup cost (~90 ms). The view does a handful of slice splits
/// instead; the construction now happens once, offline, in `corpus-tool
/// dms-pack`. `sha256` records the **source TOML's** digest (carried in
/// the artifact's own header), preserving this registry's provenance
/// contract: the checksum still names the audited source bytes, exactly
/// as it did when the TOML was parsed directly.
///
/// # Panics
/// Panics if the embedded artifact fails to parse — a bug in this
/// crate's own vendored data (covered by `dms_bin`'s round-trip and
/// drift tests), not a runtime condition.
pub static DMS: LazyLock<LoadedPack<DmsIndexView<'static>>> = LazyLock::new(|| {
    let pack = DmsIndexView::parse(DMS_INDEX_V1_BIN).expect("embedded dms-index-v1.bin must parse");
    let sha256 = Sha256::parse_hex(pack.source_sha256_hex())
        .expect("a parsed artifact's recorded sha256 is always 64 hex characters");
    LoadedPack {
        version: "dms-index-v1".into(),
        sha256,
        pack,
    }
});

/// The embedded `attestation-v1.toml` source — written by `corpus-tool
/// attest`.
const ATTESTATION_V1_TOML: &str = include_str!("../packs/attestation-v1.toml");

/// The built-in attestation pack, parsed once from the embedded
/// `attestation-v1.toml` and reused for the life of the process.
///
/// # Panics
/// Panics if the embedded `attestation-v1.toml` fails to parse — a bug in
/// this crate's own vendored data (covered by this crate's attestation
/// tests), not a condition any caller can recover from by retrying.
pub static ATTESTATION: LazyLock<LoadedPack<AttestationPack>> = LazyLock::new(|| {
    let pack = AttestationPack::parse(ATTESTATION_V1_TOML)
        .expect("embedded attestation-v1.toml must parse: see this crate's attestation tests");
    LoadedPack {
        version: "attestation-v1".into(),
        sha256: Sha256::of_bytes(ATTESTATION_V1_TOML.as_bytes()),
        pack,
    }
});

/// The embedded `register-v1.toml` source — written by hand (measured
/// off-line from the train-split `docs` genre; see that file's own header
/// comment), not generated by a `corpus-tool` subcommand.
const REGISTER_V1_TOML: &str = include_str!("../packs/register-v1.toml");

/// The built-in register pack, parsed once from the embedded
/// `register-v1.toml` and reused for the life of the process.
///
/// # Panics
/// Panics if the embedded `register-v1.toml` fails to parse — a bug in
/// this crate's own vendored data (covered by this crate's register
/// tests), not a condition any caller can recover from by retrying.
pub static REGISTER: LazyLock<LoadedPack<RegisterPack>> = LazyLock::new(|| {
    let pack = RegisterPack::parse(REGISTER_V1_TOML)
        .expect("embedded register-v1.toml must parse: see this crate's register tests");
    LoadedPack {
        version: "register-v1".into(),
        sha256: Sha256::of_bytes(REGISTER_V1_TOML.as_bytes()),
        pack,
    }
});

/// The embedded `jargon-v1.toml` source — the curated metaphor-lexeme
/// pack, hand-curated (not generated by a `corpus-tool` subcommand); see
/// that file's own header comment.
const JARGON_V1_TOML: &str = include_str!("../packs/jargon-v1.toml");

/// The built-in jargon pack, parsed once from the embedded
/// `jargon-v1.toml` and reused for the life of the process.
///
/// # Panics
/// Panics if the embedded `jargon-v1.toml` fails to parse — a bug in
/// this crate's own vendored data (covered by this crate's jargon
/// tests), not a condition any caller can recover from by retrying.
pub static JARGON: LazyLock<LoadedPack<JargonPack>> = LazyLock::new(|| {
    let pack = JargonPack::parse(JARGON_V1_TOML)
        .expect("embedded jargon-v1.toml must parse: see this crate's jargon tests");
    LoadedPack {
        version: "jargon-v1".into(),
        sha256: Sha256::of_bytes(JARGON_V1_TOML.as_bytes()),
        pack,
    }
});

/// The embedded `jargon-attest-v1.bin` — the web-scale compound
/// attestation filter (`BinaryFuse8` over ~2M normalized Wikipedia-title
/// and OpenAlex-topic keys), generated by `corpus-tool jargon-attest`;
/// see that pack's own sidecar for full provenance.
const JARGON_ATTEST_V1_BIN: &[u8] = include_bytes!("../packs/jargon-attest-v1.bin");

/// The embedded `jargon-attest-v1.toml` sidecar — version/key-count
/// cross-check plus human-readable provenance (sources, licensing,
/// normalization/hash spec) for [`JARGON_ATTEST_V1_BIN`]; see that file
/// itself.
const JARGON_ATTEST_V1_TOML: &str = include_str!("../packs/jargon-attest-v1.toml");

/// The built-in jargon attestation pack, loaded once from the embedded
/// `.bin` + `.toml` sidecar and reused for the life of the process.
///
/// `friction-match`'s `jargon.metaphor` channel's real attestation oracle
/// (SYNTHESIS.md §4), replacing what used to be a hand-curated exception
/// list alone.
///
/// # Panics
/// Panics if the embedded `.bin`/`.toml` pair fails to load — a bug in
/// this crate's own vendored data (covered by this crate's
/// `jargon_attest` tests), not a condition any caller can recover from by
/// retrying.
pub static JARGON_ATTEST: LazyLock<JargonAttestPack> = LazyLock::new(|| {
    JargonAttestPack::load(JARGON_ATTEST_V1_BIN, JARGON_ATTEST_V1_TOML)
        .expect("embedded jargon-attest-v1 pack must load: see this crate's jargon_attest tests")
});

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inventory_static_loads_and_validates_cleanly() {
        assert_eq!(&*INVENTORY.version, "inventory-v1");
        let violations = crate::validate::validate(&INVENTORY.pack);
        assert!(
            violations.is_empty(),
            "embedded inventory pack has violations: {violations:#?}"
        );
    }

    #[test]
    fn inventory_sha256_matches_recomputed_hash_of_the_embedded_bytes() {
        let recomputed = Sha256::of_bytes(INVENTORY_V1_TOML.as_bytes());
        assert_eq!(INVENTORY.sha256, recomputed);
        assert!(recomputed.verify(INVENTORY_V1_TOML.as_bytes()));
    }

    const SAMPLE_DMS_PACK: &str = r#"
        [vocab]
        tokens = ["<unused-0>", "the", "quick", "fox"]

        [streams.human]
        ids = "1,2,3"
    "#;

    #[test]
    fn load_dms_pack_parses_a_hermetic_pack_and_records_its_sha256() {
        let loaded = load_dms_pack(SAMPLE_DMS_PACK).expect("sample pack parses");
        assert_eq!(&*loaded.version, "dms-index-v1");
        assert_eq!(loaded.pack.vocab().len(), 4);
        assert_eq!(loaded.sha256, Sha256::of_bytes(SAMPLE_DMS_PACK.as_bytes()));
    }

    /// The in-repo source TOML, loaded at test time only — the runtime
    /// embeds the derived `.bin`, and these tests are what keep the two
    /// from diverging.
    const DMS_INDEX_V1_TOML: &str = include_str!("../packs/dms-index-v1.toml");

    #[test]
    fn dms_static_loads_and_defines_every_family() {
        assert_eq!(&*DMS.version, "dms-index-v1");
        for family in crate::ModelFamily::ALL {
            assert!(
                DMS.pack.family_sam(family).is_some(),
                "embedded dms-index-v1.bin has no stream for {family}"
            );
        }
    }

    #[test]
    fn dms_sha256_matches_recomputed_hash_of_the_source_toml() {
        let recomputed = Sha256::of_bytes(DMS_INDEX_V1_TOML.as_bytes());
        assert_eq!(DMS.sha256, recomputed);
        assert!(recomputed.verify(DMS_INDEX_V1_TOML.as_bytes()));
    }

    /// The drift guard: re-packing the in-repo TOML must reproduce the
    /// embedded `.bin` byte for byte. Fails when `corpus-tool index`
    /// regenerates the TOML but `corpus-tool dms-pack` wasn't re-run —
    /// the exact staleness the artifact's recorded source sha256 exists
    /// to catch.
    #[test]
    fn dms_bin_is_freshly_packed_from_the_source_toml() {
        let repacked =
            crate::pack_dms_index_bin(DMS_INDEX_V1_TOML).expect("in-repo dms-index-v1.toml packs");
        assert_eq!(
            repacked, DMS_INDEX_V1_BIN,
            "packs/dms-index-v1.bin is stale: re-run `corpus-tool dms-pack`"
        );
    }

    #[test]
    fn parsing_the_same_inventory_bytes_twice_is_deterministic() {
        let a = InventoryPack::parse(INVENTORY_V1_TOML).unwrap();
        let b = InventoryPack::parse(INVENTORY_V1_TOML).unwrap();
        let ids_a: Vec<&str> = a.deletion_spans().iter().map(|s| &*s.id).collect();
        let ids_b: Vec<&str> = b.deletion_spans().iter().map(|s| &*s.id).collect();
        assert_eq!(ids_a, ids_b);
        let noms_a: Vec<&str> = a.lvc_pairs().iter().map(|p| &*p.nominalization).collect();
        let noms_b: Vec<&str> = b.lvc_pairs().iter().map(|p| &*p.nominalization).collect();
        assert_eq!(noms_a, noms_b);
    }

    #[test]
    fn attestation_static_loads() {
        assert_eq!(&*ATTESTATION.version, "attestation-v1");
        assert!(ATTESTATION.pack.bigram().vocab_len() > 0);
        assert!(ATTESTATION.pack.skeleton().tag_vocab_len() > 0);
    }

    #[test]
    fn attestation_sha256_matches_recomputed_hash_of_the_embedded_bytes() {
        let recomputed = Sha256::of_bytes(ATTESTATION_V1_TOML.as_bytes());
        assert_eq!(ATTESTATION.sha256, recomputed);
        assert!(recomputed.verify(ATTESTATION_V1_TOML.as_bytes()));
    }

    #[test]
    fn parsing_the_same_attestation_bytes_twice_is_deterministic() {
        let a = AttestationPack::parse(ATTESTATION_V1_TOML).unwrap();
        let b = AttestationPack::parse(ATTESTATION_V1_TOML).unwrap();
        assert_eq!(a.bigram().edge_count(), b.bigram().edge_count());
        assert_eq!(a.skeleton().tag5_len(), b.skeleton().tag5_len());
        assert_eq!(a.skeleton().tag4_len(), b.skeleton().tag4_len());
    }

    #[test]
    fn register_static_loads() {
        assert_eq!(&*REGISTER.version, "register-v1");
        assert!(REGISTER.pack.band("nominalization").is_some());
        assert!(REGISTER.pack.band("agentless_passive").is_some());
        assert!(REGISTER.pack.band("em_dash").is_some());
        assert!(REGISTER.pack.band("semicolon").is_some());
    }

    #[test]
    fn register_sha256_matches_recomputed_hash_of_the_embedded_bytes() {
        let recomputed = Sha256::of_bytes(REGISTER_V1_TOML.as_bytes());
        assert_eq!(REGISTER.sha256, recomputed);
        assert!(recomputed.verify(REGISTER_V1_TOML.as_bytes()));
    }

    #[test]
    fn jargon_static_loads() {
        assert_eq!(&*JARGON.version, "jargon-v1");
        assert!(!JARGON.pack.lexemes().is_empty());
        assert!(!JARGON.pack.attested_exceptions().is_empty());
        assert!(JARGON.pack.is_head_word("well"));
        assert!(JARGON.pack.is_head_word("wells"));
        assert!(JARGON.pack.is_attested_exception("service fabric"));
    }

    /// `echo` is explicitly excluded (see the pack's own header comment:
    /// human-favored on this corpus, and a shell command besides) — this
    /// pins that exclusion as a regression test, not just prose.
    #[test]
    fn jargon_static_never_lists_echo() {
        assert!(!JARGON.pack.is_head_word("echo"));
        for lexeme in JARGON.pack.lexemes() {
            assert_ne!(&*lexeme.lexeme, "echo");
            assert_ne!(lexeme.plural.as_deref(), Some("echo"));
        }
    }

    /// Every lexeme and every attested exception carries non-empty
    /// `notes` — enforced structurally by [`JargonPack::parse`], pinned
    /// here against the real embedded pack rather than only a synthetic
    /// one.
    #[test]
    fn jargon_static_every_entry_has_notes() {
        for lexeme in JARGON.pack.lexemes() {
            assert!(!lexeme.notes.trim().is_empty(), "{}", lexeme.lexeme);
        }
        for exception in JARGON.pack.attested_exceptions() {
            assert!(!exception.notes.trim().is_empty(), "{}", exception.compound);
        }
    }

    #[test]
    fn jargon_sha256_matches_recomputed_hash_of_the_embedded_bytes() {
        let recomputed = Sha256::of_bytes(JARGON_V1_TOML.as_bytes());
        assert_eq!(JARGON.sha256, recomputed);
        assert!(recomputed.verify(JARGON_V1_TOML.as_bytes()));
    }

    #[test]
    fn jargon_attest_static_loads() {
        assert_eq!(JARGON_ATTEST.version(), "jargon-attest-v1");
        // ~2M Wikipedia titles alone; a low bound well clear of any
        // plausible truncation/misbuild while not pinning the exact
        // count (which drifts if the corpus is ever refreshed).
        assert!(JARGON_ATTEST.key_count() > 1_000_000);
    }

    /// Spot-checks verified against the raw `jargon-attest-v1` input data
    /// (see `crates/corpus-tool/src/commands/jargon_attest.rs`'s own
    /// tests and this pack's sidecar for provenance) — real Wikipedia
    /// titles / `OpenAlex` topics the filter must attest.
    #[test]
    fn jargon_attest_static_positive_spot_checks() {
        for compound in [
            "data fabric",
            "primordial soup",
            "resonance frequency",
            "service mesh",
            "inverted index",
        ] {
            assert!(
                JARGON_ATTEST.is_attested(compound),
                "{compound:?} should be attested"
            );
        }
    }

    /// Spot-checks verified NOT present in either source as of the
    /// 2026-07-27 measurement this task's own acceptance criteria pin —
    /// invented pseudo-jargon the filter must not suppress.
    #[test]
    fn jargon_attest_static_negative_spot_checks() {
        for compound in ["semantic wells", "cross domain resonance", "topical soup"] {
            assert!(
                !JARGON_ATTEST.is_attested(compound),
                "{compound:?} should NOT be attested"
            );
        }
    }
}
