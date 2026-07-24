//! The unified pack-registry entry point: loads a pack's TOML bytes into
//! its typed, parsed form, alongside its declared version and a recorded
//! sha256 of the exact source bytes it was parsed from.

use std::sync::LazyLock;

use crate::dms::DmsIndex;
use crate::inventory::InventoryPack;
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

/// The embedded `dms-index-v1.toml` source — ~1.9 MB.
const DMS_INDEX_V1_TOML: &str = include_str!("../packs/dms-index-v1.toml");

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
/// Deliberately **not** embedded here: the pack is ~1.9 MB and not yet
/// wired into any runtime detection pass — see [`crate::dms`]'s own docs.
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

/// The built-in DMS index, parsed once from the embedded
/// `dms-index-v1.toml` and reused for the life of the process.
///
/// ~1.9 MB embedded — the pack this crate's own `dms` module docs said
/// was "not yet wired into any runtime detection pass"; `friction-match`
/// is that pass.
///
/// # Panics
/// Panics if the embedded pack fails to parse — a bug in this crate's own
/// vendored data, not a runtime condition.
pub static DMS: LazyLock<LoadedPack<DmsIndex>> = LazyLock::new(|| {
    let pack = DmsIndex::parse(DMS_INDEX_V1_TOML).expect("embedded dms-index-v1.toml must parse");
    LoadedPack {
        version: "dms-index-v1".into(),
        sha256: Sha256::of_bytes(DMS_INDEX_V1_TOML.as_bytes()),
        pack,
    }
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

    #[test]
    fn dms_static_loads_and_defines_every_family() {
        assert_eq!(&*DMS.version, "dms-index-v1");
        for family in crate::ModelFamily::ALL {
            assert!(
                DMS.pack.family_sam(family).is_some(),
                "embedded dms-index-v1.toml has no stream for {family}"
            );
        }
    }

    #[test]
    fn dms_sha256_matches_recomputed_hash_of_the_embedded_bytes() {
        let recomputed = Sha256::of_bytes(DMS_INDEX_V1_TOML.as_bytes());
        assert_eq!(DMS.sha256, recomputed);
        assert!(recomputed.verify(DMS_INDEX_V1_TOML.as_bytes()));
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
}
