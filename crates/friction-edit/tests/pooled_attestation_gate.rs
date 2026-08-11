//! Gate-level proof that `friction_packs::pooled_attest` widens the
//! deletion gate's seam check: the exact TRAIN-only bigram table alone
//! holds a candidate on `SeamNotAttested`, and attaching the REAL
//! committed pooled pack (mined from external human corpora — see
//! `friction_packs::pooled_attest`'s own module docs) makes the exact
//! same candidate clear the gate.
//!
//! Uses the actual committed `attestation-en-v1.toml` and
//! `pooled-attest-en-v1.{bin,meta.toml}` (`include_str!`/`include_bytes!`'d
//! directly, not the crate's embedded `.bin` views — [`AttestationPack::parse`]
//! on the source TOML gives an "unwidened" baseline pack with no `with_pooled`
//! call anywhere near it), so this is real mined evidence end to end, not a
//! synthetic fixture: `("will", "upgrade")` is the exact seam
//! SYNTHESIS-motivation calls out as an "ordinary" pair the small TRAIN
//! corpus never happened to attest.

use friction_edit::gates::{self, DeletionGateOutcome};
use friction_nlp::PerceptronTagger;
use friction_packs::AttestationPack;
use friction_packs::pooled_attest::PooledAttestPack;

/// The exact TOML `corpus-tool attest` writes and
/// `friction_packs::registry::ATTESTATION` embeds a packed derivative
/// of — read directly here so this test builds its own, independent
/// (unwidened) `AttestationPack` rather than reaching for the crate's
/// already-`with_pooled`-attached singleton.
const ATTESTATION_TOML: &str = include_str!("../../friction-packs/packs/attestation-en-v1.toml");

/// The real, committed pooled-attest artifact `corpus-tool pooled-attest`
/// mined from the staged external corpora (~43M tokens) — the same bytes
/// `friction_packs::registry::POOLED_ATTEST` embeds.
const POOLED_BIN: &[u8] = include_bytes!("../../friction-packs/packs/pooled-attest-en-v1.bin");
const POOLED_META: &str = include_str!("../../friction-packs/packs/pooled-attest-en-v1.meta.toml");

/// A sentence whose only deletable span ("surely ") leaves the surviving
/// seam exactly `("will", "upgrade")`.
const TEXT: &str = "The team will surely upgrade the system.";

/// The byte range of "surely " within [`TEXT`] — deleting it leaves
/// "The team will " + "upgrade the system.".
fn surely_range() -> std::ops::Range<usize> {
    let start = TEXT.find("surely ").expect("fixture contains \"surely \"");
    start..start + "surely ".len()
}

#[test]
fn pooled_evidence_turns_a_held_deletion_into_an_allowed_one() {
    let pre = &TEXT[..surely_range().start];
    let post = &TEXT[surely_range().end..];
    assert!(
        pre.ends_with("will "),
        "fixture: left seam token is \"will\""
    );
    assert!(
        post.starts_with("upgrade "),
        "fixture: right seam token is \"upgrade\""
    );

    let tagger = PerceptronTagger::new().expect("embedded tagger loads");
    let original_tags = gates::tag(&tagger, TEXT);
    let original_clause_ok = gates::clause_ok(&original_tags, TEXT);

    let baseline = AttestationPack::parse(ATTESTATION_TOML)
        .expect("the committed attestation-en-v1.toml parses");
    assert!(
        !baseline.bigram().attests("will", "upgrade"),
        "fixture assumption: the TRAIN-only exact table has never attested \
         this ordinary pair — the exact recall gap SYNTHESIS's motivation \
         calls out"
    );

    let baseline_outcome =
        gates::check_deletion_gates(TEXT, surely_range(), &baseline, original_clause_ok, &tagger);
    assert_eq!(
        baseline_outcome,
        DeletionGateOutcome::SeamNotAttested,
        "fixture assumption: the deletion is held on the unwidened table"
    );

    let pooled = PooledAttestPack::load(POOLED_BIN, POOLED_META)
        .expect("the committed pooled-attest artifact loads");
    assert!(
        pooled.attests("will", "upgrade"),
        "fixture assumption: the real mined pooled evidence attests this pair"
    );
    let widened = baseline.with_pooled(Box::leak(Box::new(pooled)));

    let widened_outcome =
        gates::check_deletion_gates(TEXT, surely_range(), &widened, original_clause_ok, &tagger);
    assert_eq!(
        widened_outcome,
        DeletionGateOutcome::Allowed,
        "the pooled table's real evidence must turn a previously-held \
         deletion into an allowed one"
    );
}
