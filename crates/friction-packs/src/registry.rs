//! The unified pack-registry entry point: loads a pack's TOML bytes into
//! its typed, parsed form, with its declared version and a recorded
//! sha256 of the exact source bytes it was parsed from.

use std::sync::LazyLock;

use friction_core::Lang;

use crate::attestation::AttestationPack;
use crate::dms::DmsIndex;
use crate::dms_bin::DmsIndexView;
use crate::frame_bin::FramePackView;
use crate::human_evidence::HumanEvidencePack;
use crate::inventory::InventoryPack;
use crate::jargon::JargonPack;
use crate::jargon_attest::JargonAttestPack;
use crate::register::RegisterPack;
use crate::validate::validate;
use crate::{PackError, Sha256};

/// A parsed pack, with its declared version string and the sha256 of the
/// exact source bytes it was parsed from.
#[derive(Debug, Clone)]
pub struct LoadedPack<T> {
    pub pack: T,
    pub version: Box<str>,
    pub sha256: Sha256,
}

/// The embedded `inventory-en-v1.toml` source — 32 KB, cheap to embed, like
/// `ENVELOPE_V2_TOML`.
const INVENTORY_V1_TOML: &str = include_str!("../packs/inventory-en-v1.toml");

/// The embedded derived DMS artifact — every stream's suffix automaton
/// pre-built by `corpus-tool dms-pack` and serialized flat (see
/// [`crate::dms_bin`]'s module docs for the layout and why). The source
/// `dms-index-en-v1.toml` stays in `packs/` as the audited artifact
/// `corpus-tool index` writes, but is no longer embedded or parsed at
/// runtime: this crate's `dms_bin` drift test re-packs it and compares
/// bytes, so the two cannot silently diverge.
const DMS_INDEX_V1_BIN: &[u8] = include_bytes!("../packs/dms-index-en-v1.bin");

/// The embedded derived frame-rewrite pack — the compiled rule program
/// `corpus-tool frame-pack` builds from `frame-rules-en-v1.toml` after
/// running the whole compile fence. ~115 KB.
const FRAME_PACK_V1_BIN: &[u8] = include_bytes!("../packs/frame-pack-en-v1.bin");

/// The built-in inventory pack, parsed once from the embedded
/// `inventory-en-v1.toml` and reused for the life of the process.
///
/// # Panics
/// Panics if the embedded `inventory-en-v1.toml` fails to parse, or if it
/// parses but fails audit (disjointness/closure/frequency-hygiene/
/// guard-token-exposure) — a bug in this crate's own vendored data
/// (covered by this crate's inventory and validate tests), not a
/// condition any caller can recover from by retrying.
pub static INVENTORY: LazyLock<LoadedPack<InventoryPack>> = LazyLock::new(|| {
    let pack = InventoryPack::parse(INVENTORY_V1_TOML).expect(
        "embedded inventory-en-v1.toml must parse: see this crate's inventory/validate tests",
    );
    let violations = validate(&pack);
    assert!(
        violations.is_empty(),
        "embedded inventory-en-v1.toml failed audit: {violations:#?}"
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
/// This used to parse the ~2.5 MB `dms-index-en-v1.toml` and build every
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
    let pack =
        DmsIndexView::parse(DMS_INDEX_V1_BIN).expect("embedded dms-index-en-v1.bin must parse");
    let sha256 = Sha256::parse_hex(pack.source_sha256_hex())
        .expect("a parsed artifact's recorded sha256 is always 64 hex characters");
    LoadedPack {
        version: "dms-index-v1".into(),
        sha256,
        pack,
    }
});

/// The built-in frame-rewrite rule program, viewed zero-copy over the
/// embedded derived artifact and reused for the life of the process.
///
/// The source `frame-rules-en-v1.toml` never parses at runtime: the whole
/// compile fence — attestation against the DMS human stream included —
/// ran once, offline, in `corpus-tool frame-pack`. `sha256` records the
/// **source TOML's** digest (carried in the artifact's own header), the
/// same provenance contract as [`DMS`].
///
/// # Panics
/// Panics if the embedded artifact fails to parse — a bug in this
/// crate's own vendored data (covered by `frame_bin`'s round-trip and
/// this module's drift tests), not a runtime condition.
pub static FRAME: LazyLock<LoadedPack<FramePackView<'static>>> = LazyLock::new(|| {
    let pack =
        FramePackView::parse(FRAME_PACK_V1_BIN).expect("embedded frame-pack-en-v1.bin must parse");
    let sha256 = Sha256::parse_hex(pack.source_sha256_hex())
        .expect("a parsed artifact's recorded sha256 is always 64 hex characters");
    LoadedPack {
        version: "frame-pack-v1".into(),
        sha256,
        pack,
    }
});

/// The embedded derived attestation artifact — the bigram/skeleton
/// tables pre-flattened by `corpus-tool attest-pack` (see the artifact
/// layout docs in [`crate::attestation`]). The source
/// `attestation-en-v1.toml` stays in `packs/` as the audited artifact
/// `corpus-tool attest` writes, but is no longer embedded or parsed at
/// runtime: this module's drift test re-packs it and compares bytes, so
/// the two cannot silently diverge.
const ATTESTATION_V1_BIN: &[u8] = include_bytes!("../packs/attestation-en-v1.bin");

/// The built-in attestation pack, viewed zero-copy over the embedded
/// derived artifact and reused for the life of the process.
///
/// This used to parse the ~1.8 MB `attestation-en-v1.toml` on first touch —
/// measured ~25 ms, the largest single slice of `friction`'s fixed
/// startup cost after [`DMS`]'s own conversion. The view does header
/// validation plus slice splits instead; the flattening now happens
/// once, offline, in `corpus-tool attest-pack`. `sha256` records the
/// **source TOML's** digest (carried in the artifact's own header),
/// preserving this registry's provenance contract — the same shape as
/// [`DMS`] and [`FRAME`].
///
/// # Panics
/// Panics if the embedded artifact fails to parse: a bug in this
/// crate's own vendored data (covered by this crate's attestation
/// round-trip and drift tests), not a runtime condition.
pub static ATTESTATION: LazyLock<LoadedPack<AttestationPack>> = LazyLock::new(|| {
    let (pack, sha_hex) = AttestationPack::from_bin(ATTESTATION_V1_BIN)
        .expect("embedded attestation-en-v1.bin must parse: see this crate's attestation tests");
    LoadedPack {
        version: "attestation-v1".into(),
        sha256: Sha256::parse_hex(sha_hex)
            .expect("a parsed artifact's recorded sha256 is always 64 hex characters"),
        pack,
    }
});

/// The embedded `register-en-v1.toml` source — written by hand (measured
/// off-line from the train-split `docs` genre; see that file's own header
/// comment), not generated by a `corpus-tool` subcommand.
const REGISTER_V1_TOML: &str = include_str!("../packs/register-en-v1.toml");

/// The built-in register pack, parsed once from the embedded
/// `register-en-v1.toml` and reused for the life of the process.
///
/// # Panics
/// Panics if the embedded `register-en-v1.toml` fails to parse — a bug in
/// this crate's own vendored data (covered by this crate's register
/// tests), not a condition any caller can recover from by retrying.
pub static REGISTER: LazyLock<LoadedPack<RegisterPack>> = LazyLock::new(|| {
    let pack = RegisterPack::parse(REGISTER_V1_TOML)
        .expect("embedded register-en-v1.toml must parse: see this crate's register tests");
    LoadedPack {
        version: "register-v1".into(),
        sha256: Sha256::of_bytes(REGISTER_V1_TOML.as_bytes()),
        pack,
    }
});

/// The embedded `jargon-en-v1.toml` source — the curated metaphor-lexeme
/// pack, hand-curated (not generated by a `corpus-tool` subcommand); see
/// that file's own header comment.
const JARGON_V1_TOML: &str = include_str!("../packs/jargon-en-v1.toml");

/// The built-in jargon pack, parsed once from the embedded
/// `jargon-en-v1.toml` and reused for the life of the process.
///
/// # Panics
/// Panics if the embedded `jargon-en-v1.toml` fails to parse — a bug in
/// this crate's own vendored data (covered by this crate's jargon
/// tests), not a condition any caller can recover from by retrying.
pub static JARGON: LazyLock<LoadedPack<JargonPack>> = LazyLock::new(|| {
    let pack = JargonPack::parse(JARGON_V1_TOML)
        .expect("embedded jargon-en-v1.toml must parse: see this crate's jargon tests");
    LoadedPack {
        version: "jargon-v1".into(),
        sha256: Sha256::of_bytes(JARGON_V1_TOML.as_bytes()),
        pack,
    }
});

/// The embedded `jargon-attest-en-v1.bin` — the web-scale compound
/// attestation filter (`BinaryFuse8` over ~2M normalized Wikipedia-title
/// and OpenAlex-topic keys), generated by `corpus-tool jargon-attest`;
/// see that pack's own sidecar for full provenance.
const JARGON_ATTEST_V1_BIN: &[u8] = include_bytes!("../packs/jargon-attest-en-v1.bin");

/// The embedded `jargon-attest-en-v1.toml` sidecar — version/key-count
/// cross-check plus human-readable provenance (sources, licensing,
/// normalization/hash spec) for [`JARGON_ATTEST_V1_BIN`]; see that file
/// itself.
const JARGON_ATTEST_V1_TOML: &str = include_str!("../packs/jargon-attest-en-v1.toml");

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

/// The embedded `human-evidence-en-v1.bin` — external human-corpus unigram
/// and literal-probe occurrence counts, generated by `corpus-tool
/// human-evidence` from locally-staged external corpora; see
/// [`crate::human_evidence`]'s own module docs for the pack's two tables.
const HUMAN_EVIDENCE_V1_BIN: &[u8] = include_bytes!("../packs/human-evidence-en-v1.bin");

/// The embedded `machine-evidence-en-v1.bin` — the machine half of the
/// review-register evidence pair: probe counts over the committed
/// LLM-generated review documents in `corpus/review/machine/`, built by
/// the same `corpus-tool human-evidence` command with
/// `--pack-name machine-evidence-v1`. Register-matched against
/// [`HUMAN_EVIDENCE`]'s review-register buckets, never against the DMS
/// streams — see `crate::frame_compile`'s register-matching docs.
const MACHINE_EVIDENCE_V1_BIN: &[u8] = include_bytes!("../packs/machine-evidence-en-v1.bin");

/// The built-in human evidence pack.
///
/// Parsed once from the embedded `human-evidence-en-v1.bin` and reused for
/// the life of the process.
/// `crate::frame_compile::CorpusEvidence::with_external` pools it into
/// every frame-rewrite compile fence: see that method's own docs.
///
/// # Panics
/// Panics if the embedded `human-evidence-en-v1.bin` fails to parse — a bug
/// in this crate's own vendored data (covered by this crate's
/// `human_evidence` tests), not a condition any caller can recover from
/// by retrying.
pub static HUMAN_EVIDENCE: LazyLock<HumanEvidencePack> = LazyLock::new(|| {
    HumanEvidencePack::load_static(HUMAN_EVIDENCE_V1_BIN).expect(
        "embedded human-evidence-en-v1.bin must load: see this crate's human_evidence tests",
    )
});

/// The built-in machine-review evidence pack (same on-disk format as
/// [`HUMAN_EVIDENCE`], machine-side content), parsed once and reused
/// for the life of the process.
///
/// # Panics
/// Panics if the embedded `machine-evidence-en-v1.bin` fails to parse — a
/// bug in this crate's own vendored data, not a condition any caller
/// can recover from by retrying.
pub static MACHINE_EVIDENCE: LazyLock<HumanEvidencePack> = LazyLock::new(|| {
    HumanEvidencePack::load_static(MACHINE_EVIDENCE_V1_BIN).expect(
        "embedded machine-evidence-en-v1.bin must load: see this crate's human_evidence tests",
    )
});

/// Every language-scoped pack this crate ships, bundled behind one
/// [`Lang`]-keyed access path.
///
/// **This is the access path for language-scoped packs.** A caller that
/// wants "the inventory pack for the language this document is being
/// analyzed as" reaches for `PackSet::for_lang(lang).inventory`, never
/// [`INVENTORY`] directly — the individual statics (`INVENTORY`, `DMS`,
/// `JARGON`, `JARGON_ATTEST`, `ATTESTATION`, `REGISTER`, `FRAME`,
/// `HUMAN_EVIDENCE`, `MACHINE_EVIDENCE`) still exist and still do the
/// actual embedding and parsing; they are the English backing store this
/// struct bundles references to, kept `pub` because this crate's own
/// tests (and the drift/freshness guards below) still address them by
/// name.
///
/// v1 ships exactly one bundle ([`Lang::En`], built from those same nine
/// statics), so `for_lang` cannot yet observe two bundles disagreeing —
/// but every consumer that goes through it instead of the bare statics is
/// already written the way a second language's bundle will need it read.
pub struct PackSet {
    /// The curated tell-span inventory (deletion spans, substitution
    /// pairs, ritual frames, licensed LVC pairs).
    pub inventory: &'static LoadedPack<InventoryPack>,
    /// The DMS (differential machine score) index: per-family and pooled
    /// suffix automata over the human/machine token streams.
    pub dms: &'static LoadedPack<DmsIndexView<'static>>,
    /// The curated metaphor-lexeme pack the jargon channel flags against.
    pub jargon: &'static LoadedPack<JargonPack>,
    /// The web-scale compound attestation filter backing the jargon
    /// channel's real attestation oracle.
    pub jargon_attest: &'static JargonAttestPack,
    /// The seam-bigram and POS-skeleton attestation tables the repair
    /// engine's gates consult.
    pub attestation: &'static LoadedPack<AttestationPack>,
    /// The per-1000-word human rate bands (nominalization, agentless
    /// passive, em dash, semicolon, ...) the register pass measures
    /// against.
    pub register: &'static LoadedPack<RegisterPack>,
    /// The compiled contrast-frame collocation-rewrite rule program.
    pub frame: &'static LoadedPack<FramePackView<'static>>,
    /// External human-corpus unigram/literal-probe evidence, pooled into
    /// every frame-rewrite compile fence.
    pub human_evidence: &'static HumanEvidencePack,
    /// The machine-side half of the review-register evidence pair,
    /// register-matched against `human_evidence`'s review buckets.
    pub machine_evidence: &'static HumanEvidencePack,
}

/// The built-in [`PackSet`] for [`Lang::En`], bundling references to the
/// nine statics above — built once, on first use, and reused for the life
/// of the process. Building it does not itself force any of the
/// individual statics; each is still parsed lazily on its own first
/// touch, same as before this bundle existed.
static PACK_SET_EN: LazyLock<PackSet> = LazyLock::new(|| PackSet {
    inventory: &INVENTORY,
    dms: &DMS,
    jargon: &JARGON,
    jargon_attest: &JARGON_ATTEST,
    attestation: &ATTESTATION,
    register: &REGISTER,
    frame: &FRAME,
    human_evidence: &HUMAN_EVIDENCE,
    machine_evidence: &MACHINE_EVIDENCE,
});

impl PackSet {
    /// Selects the built-in [`PackSet`] for `lang` — `Lang::En` ->
    /// [`PACK_SET_EN`].
    ///
    /// The match is exhaustive over [`Lang`]: adding a language variant
    /// without a pack bundle for it is a compile error here, not a
    /// runtime surprise.
    #[must_use]
    pub fn for_lang(lang: Lang) -> &'static Self {
        match lang {
            Lang::En => &PACK_SET_EN,
        }
    }
}

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
    const DMS_INDEX_V1_TOML: &str = include_str!("../packs/dms-index-en-v1.toml");

    #[test]
    fn dms_static_loads_and_defines_every_family() {
        assert_eq!(&*DMS.version, "dms-index-v1");
        for family in crate::ModelFamily::ALL {
            assert!(
                DMS.pack.family_sam(family).is_some(),
                "embedded dms-index-en-v1.bin has no stream for {family}"
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
        let repacked = crate::pack_dms_index_bin(DMS_INDEX_V1_TOML)
            .expect("in-repo dms-index-en-v1.toml packs");
        assert_eq!(
            repacked, DMS_INDEX_V1_BIN,
            "packs/dms-index-en-v1.bin is stale: re-run `corpus-tool dms-pack`"
        );
    }

    /// The in-repo frame rules TOML, loaded at test time only.
    const FRAME_RULES_V1_TOML: &str = include_str!("../packs/frame-rules-en-v1.toml");

    #[test]
    fn frame_static_loads_with_rules_of_every_kind() {
        assert_eq!(&*FRAME.version, "frame-pack-v1");
        assert!(FRAME.pack.rule_count() > 0);
        let kinds: std::collections::BTreeSet<u8> = (0..FRAME.pack.rule_count())
            .map(|i| FRAME.pack.rule(i).expect("rule in range").kind as u8)
            .collect();
        assert_eq!(
            kinds.len(),
            4,
            "rewrite, delete, guard, and report rules all present"
        );
    }

    #[test]
    fn frame_sha256_matches_recomputed_hash_of_the_source_toml() {
        let recomputed = Sha256::of_bytes(FRAME_RULES_V1_TOML.as_bytes());
        assert_eq!(FRAME.sha256, recomputed);
    }

    /// The drift guard: re-compiling and re-packing the in-repo rules
    /// (with attestation rates re-derived from the in-repo DMS TOML,
    /// pooled with the in-repo human-evidence pack) must reproduce the
    /// embedded `.bin` byte for byte. Fails when any of the three
    /// sources changes without `corpus-tool frame-pack` re-run.
    ///
    /// Every input is committed, in-repo bytes — the human-evidence pack
    /// included, even though its own provenance traces back to external
    /// corpora: once built and checked in, re-packing against it is a
    /// pure function of this repository's own bytes, same as the other
    /// two inputs.
    #[test]
    fn frame_pack_is_freshly_compiled_from_the_source_toml() {
        let set = crate::frame_rules::FrameRuleSet::parse(FRAME_RULES_V1_TOML)
            .expect("in-repo frame-rules-en-v1.toml parses");
        let rates = crate::frame_compile::CorpusEvidence::from_dms_toml(DMS_INDEX_V1_TOML)
            .expect("in-repo dms-index-en-v1.toml evidence derives")
            .with_external(
                crate::human_evidence::HumanEvidencePack::load(HUMAN_EVIDENCE_V1_BIN)
                    .expect("in-repo human-evidence-en-v1.bin loads"),
            )
            .with_external_machine(
                crate::human_evidence::HumanEvidencePack::load(MACHINE_EVIDENCE_V1_BIN)
                    .expect("in-repo machine-evidence-en-v1.bin loads"),
            );
        let (pack, _) =
            crate::frame_compile::compile(&set, &rates).expect("in-repo rule set compiles");
        let sha_hex = Sha256::of_bytes(FRAME_RULES_V1_TOML.as_bytes()).to_string();
        let repacked = crate::frame_bin::pack_frame_bin(&pack, &sha_hex);
        assert_eq!(
            repacked, FRAME_PACK_V1_BIN,
            "packs/frame-pack-en-v1.bin is stale: re-run `corpus-tool frame-pack`"
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

    /// The in-repo source TOML, loaded at test time only — the runtime
    /// embeds the derived `.bin`, and these tests are what keep the two
    /// from diverging (same arrangement as [`DMS_INDEX_V1_TOML`]).
    const ATTESTATION_V1_TOML: &str = include_str!("../packs/attestation-en-v1.toml");

    #[test]
    fn attestation_static_loads() {
        assert_eq!(&*ATTESTATION.version, "attestation-v1");
        assert!(ATTESTATION.pack.bigram().vocab_len() > 0);
        assert!(ATTESTATION.pack.skeleton().tag_vocab_len() > 0);
    }

    #[test]
    fn attestation_sha256_matches_recomputed_hash_of_the_source_toml() {
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

    /// The drift guard: re-packing the in-repo TOML must reproduce the
    /// embedded `.bin` byte for byte. Fails when `corpus-tool attest`
    /// (or its `--calibrate-near-noop` stage) regenerates the TOML but
    /// `corpus-tool attest-pack` wasn't re-run — the exact staleness the
    /// artifact's recorded source sha256 exists to catch.
    #[test]
    fn attestation_bin_is_freshly_packed_from_the_source_toml() {
        let repacked = crate::pack_attestation_bin(ATTESTATION_V1_TOML)
            .expect("in-repo attestation-en-v1.toml packs");
        assert_eq!(
            repacked, ATTESTATION_V1_BIN,
            "packs/attestation-en-v1.bin is stale: re-run `corpus-tool attest-pack`"
        );
    }

    /// The embedded view and a fresh TOML parse answer identically:
    /// spot-checked over real vocabulary the fix pipeline queries.
    #[test]
    fn attestation_view_agrees_with_a_fresh_toml_parse() {
        let owned = AttestationPack::parse(ATTESTATION_V1_TOML).unwrap();
        let view = &ATTESTATION.pack;
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
        for (left, right) in [
            ("<s>", "the"),
            ("the", "same"),
            ("of", "the"),
            ("is", "a"),
            ("never", "attested-nonsense-token"),
            ("qzxwv", "qzxwv"),
        ] {
            assert_eq!(
                view.bigram().attests(left, right),
                owned.bigram().attests(left, right),
                "attests({left:?}, {right:?}) must agree"
            );
        }
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
    /// `notes` — enforced by [`JargonPack::parse`], pinned
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

    /// The shipped pack carries the staged code-review corpora (see the
    /// `human-evidence-en-v1.toml` sidecar for buckets and input hashes):
    /// the total is pinned exactly so a staged-corpora refresh is a
    /// deliberate, visible change to this test, not a silent drift.
    #[test]
    fn human_evidence_static_loads_the_staged_review_corpora() {
        assert_eq!(HUMAN_EVIDENCE.total_tokens(), 49_747_909);
        // "the" clears the unigram floor in any English corpus this
        // size; a word never written in review prose stays a real zero.
        assert!(HUMAN_EVIDENCE.unigram_count("the") > 1_000_000);
        assert_eq!(HUMAN_EVIDENCE.unigram_count("qzxwv"), 0);
    }

    /// `PackSet::for_lang(Lang::En)` bundles references to exactly the
    /// same nine statics a direct caller would otherwise reach for.
    #[test]
    fn pack_set_for_lang_en_bundles_the_english_statics() {
        let set = PackSet::for_lang(Lang::En);
        assert!(std::ptr::eq(set.inventory, &raw const *INVENTORY));
        assert!(std::ptr::eq(set.dms, &raw const *DMS));
        assert!(std::ptr::eq(set.jargon, &raw const *JARGON));
        assert!(std::ptr::eq(set.jargon_attest, &raw const *JARGON_ATTEST));
        assert!(std::ptr::eq(set.attestation, &raw const *ATTESTATION));
        assert!(std::ptr::eq(set.register, &raw const *REGISTER));
        assert!(std::ptr::eq(set.frame, &raw const *FRAME));
        assert!(std::ptr::eq(set.human_evidence, &raw const *HUMAN_EVIDENCE));
        assert!(std::ptr::eq(
            set.machine_evidence,
            &raw const *MACHINE_EVIDENCE
        ));
    }

    /// Two calls to `for_lang` return the exact same bundle (the
    /// `LazyLock` singleton, not a fresh allocation per call).
    #[test]
    fn pack_set_for_lang_is_a_stable_singleton() {
        assert!(std::ptr::eq(
            PackSet::for_lang(Lang::En),
            PackSet::for_lang(Lang::En)
        ));
    }
}
