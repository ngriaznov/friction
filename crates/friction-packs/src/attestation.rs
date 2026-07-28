//! Attestation pack: a seam-bigram membership table plus POS-skeleton
//! n-gram sets mined from TRAIN-split human docs.
//!
//! `corpus-tool attest` mines both tables from `friction_harness::clean`'s
//! tokenization convention (bigram side) and the shipped part-of-speech
//! tagger's coarse tags (skeleton side); this module is the read side,
//! turning the versioned TOML pack it writes (`attestation-v1.toml`,
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
//! a human sentence's skeleton. Both are membership-only — no counts, no
//! frequencies — attestation is a question of existence, and a
//! frequency threshold would just be a second, hidden calibration knob.
//!
//! # Independent tokenizations, on purpose
//!
//! The bigram table's tokens and the skeleton set's tags are built from
//! two genuinely different tokenizations of the same sentence text, and
//! this module does not try to force them into positional correspondence
//! with each other:
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

use std::collections::{BTreeMap, BTreeSet};

use serde::Deserialize;

use crate::PackError;

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

/// Seam-bigram membership table: has this exact `(left, right)` word pair
/// been observed adjacent to each other anywhere in the TRAIN human
/// corpus.
#[derive(Debug, Clone)]
pub struct BigramTable {
    vocab: TokenVocab,
    edges: BTreeMap<u32, BTreeSet<u32>>,
}

impl BigramTable {
    /// `true` if `right` was observed immediately following `left` in at
    /// least one TRAIN human sentence. `"<s>"` is a valid `left` (the
    /// reserved sentence-start token: `attests("<s>", w)` asks "does a
    /// human sentence ever open with `w`").
    ///
    /// `false` for a `left` or `right` outside this pack's own vocabulary
    /// (an out-of-vocabulary word can never have been attested, by
    /// definition) — not an error, mirroring the mining algorithm's own
    /// boolean membership test.
    #[must_use]
    pub fn attests(&self, left: &str, right: &str) -> bool {
        let Some(left_id) = self.vocab.id_of(left) else {
            return false;
        };
        let Some(right_id) = self.vocab.id_of(right) else {
            return false;
        };
        self.edges
            .get(&left_id)
            .is_some_and(|rights| rights.contains(&right_id))
    }

    /// The number of distinct left tokens with at least one attested
    /// right token (including `"<s>"` if any sentence's first word was
    /// mined).
    #[must_use]
    pub fn distinct_lefts(&self) -> usize {
        self.edges.len()
    }

    /// The total number of `(left, right)` pairs attested across every
    /// left token.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.edges.values().map(BTreeSet::len).sum()
    }

    /// This table's word vocabulary size (including the reserved `"<s>"`
    /// entry).
    #[must_use]
    pub const fn vocab_len(&self) -> usize {
        self.vocab.len()
    }
}

/// POS-skeleton n-gram set: has this exact run of coarse part-of-speech
/// tags (`<S>`/`<E>` sentinels included) been observed as (part of) a
/// TRAIN human sentence's skeleton.
#[derive(Debug, Clone)]
pub struct SkeletonSet {
    tag_vocab: TokenVocab,
    tag5: BTreeSet<u64>,
    tag4: BTreeSet<u64>,
}

impl SkeletonSet {
    /// Packs a window of tag text into this set's base-`vocab_len` `u64`
    /// encoding, or `None` if any tag in `window` is outside this set's
    /// own tag vocabulary (an out-of-vocabulary tag can never have been
    /// attested).
    fn pack_window(&self, window: &[&str]) -> Option<u64> {
        let base = u64::try_from(self.tag_vocab.len()).ok()?;
        let mut value: u64 = 0;
        for &tag in window {
            let id = self.tag_vocab.id_of(tag)?;
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
                    .is_some_and(|code| self.tag5.contains(&code));
            if five_attested {
                continue;
            }
            let four_attested = i + 4 <= n
                && self
                    .pack_window(&tags[i..i + 4])
                    .is_some_and(|code| self.tag4.contains(&code));
            if four_attested {
                continue;
            }
            return false;
        }
        true
    }

    /// The number of distinct attested 5-gram tag windows.
    #[must_use]
    pub fn tag5_len(&self) -> usize {
        self.tag5.len()
    }

    /// The number of distinct attested 4-gram tag windows.
    #[must_use]
    pub fn tag4_len(&self) -> usize {
        self.tag4.len()
    }

    /// This set's coarse-tag vocabulary size (including the `<S>`/`<E>`
    /// sentinels).
    #[must_use]
    pub const fn tag_vocab_len(&self) -> usize {
        self.tag_vocab.len()
    }

    /// The tag text for `id`, or `None` if `id` is not a valid id in this
    /// set's own tag vocabulary. Exposed only for tests/debugging that
    /// need to render a packed code back to readable tags.
    #[cfg(test)]
    fn tag_of(&self, id: u32) -> Option<&str> {
        self.tag_vocab.token_of(id)
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
            tag_vocab,
            tag5,
            tag4,
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

    Ok(BigramTable { vocab, edges })
}

/// Parses `rights` (the `;`-delimited, per-left comma-joined id groups)
/// into one `Vec<u32>` per group. An entirely empty field (both `lefts`
/// and `rights` empty — a degenerate, zero-bigram pack) parses to zero
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
/// `train_human_doc_count`) are not named here — nothing in this module
/// needs them to build a working [`AttestationPack`] — and are simply
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
}
