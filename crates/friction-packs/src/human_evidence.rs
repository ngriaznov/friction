//! `human-evidence-v1`: external human-side corpus evidence.
//!
//! Mined from staged external human corpora, this pack pools into
//! [`crate::frame_compile::CorpusEvidence`] to strengthen (never replace)
//! the `dms-index-v1` human signal the frame-rewrite compile fences
//! consult.
//!
//! Same architectural shape as [`crate::jargon_attest`]: `corpus-tool
//! human-evidence` reads locally-staged external inputs (never a network
//! request), tokenizes them through the exact same code path
//! `corpus-tool index` uses for the `dms-index-v1` human stream
//! (`friction_harness::clean::tokenize`, whole-document text, no sentence
//! splitting — see that command's own module docs for why the two
//! representations must match for pooled counts to mean the same thing),
//! and hands the counts to [`build_pack_bytes`] here for deterministic
//! serialization.
//!
//! # Two tables, two absence meanings
//!
//! The unigram table only keeps a word once it has been seen at least
//! five times (`corpus-tool human-evidence`'s own floor, to bound the
//! table and keep single-document noise out) — so a word's *absence*
//! from this table is itself real information ("fewer than five
//! occurrences"), not "never measured". [`Self::unigram_count`] returns a
//! plain `u64`, `0` for an absent word, reflecting that.
//!
//! The probe table has no such floor: it always contains exactly the
//! literal probes `frame-rules-v1.toml` produced when the pack was
//! built (every knowledge-bucket rule's [`crate::frame_rules::literal_probes`]
//! output plus every pilot rule's whitespace-split phrase), each with its
//! real measured count — including an honest `0`. A phrase's *absence*
//! here means the opposite of the unigram case: it was never one of the
//! rule set's own probes when this pack was built, so nothing was ever
//! counted for it. [`Self::probe_count`] returns `Option<u64>` to keep
//! these two meanings distinct at the type level; [`crate::frame_compile::CorpusEvidence`]
//! is the one place that turns each into the right pooling behavior.
//!
//! # The shipped empty pack
//!
//! The external human corpora this pack draws from live outside the
//! repository (large, living datasets, not vendored). Until they are
//! staged, `packs/human-evidence-v1.bin` is the *empty* pack — built by
//! running `corpus-tool human-evidence` with zero `--input` directories:
//! an empty unigram table, an empty probe table, `total_tokens = 0`. Every
//! pooling accessor on [`crate::frame_compile::CorpusEvidence`] is a
//! no-op against an empty pack by construction (pooling `0` extra count
//! and `0` extra total changes nothing), so wiring this pack into the
//! frame-pack compile today changes no compiled rule.

use std::collections::{BTreeMap, BTreeSet};

use crate::PackError;

/// The 8-byte magic every `human-evidence-v1` `.bin` starts with.
const MAGIC: &[u8; 8] = b"HUMANEV1";

/// On-disk format version — bumped only if the layout changes shape.
const FORMAT_VERSION: u16 = 1;

/// A parsed `human-evidence-v1` pack: pooled unigram and literal-probe
/// occurrence counts mined from staged external human corpora, plus the
/// external token total the pooling formula's denominator needs.
///
/// Parsed into plain owned collections rather than a zero-copy view —
/// this pack stays small (a bounded rule set's own probes, plus whatever
/// unigrams clear the five-occurrence floor), so the `dms_bin`/`frame_bin`
/// zero-copy discipline buys nothing here (see those modules' own docs for
/// when it does).
#[derive(Debug, Clone, Default)]
pub struct HumanEvidencePack {
    total_tokens: u64,
    unigrams: BTreeMap<String, u64>,
    probes: BTreeMap<Vec<String>, u64>,
}

impl HumanEvidencePack {
    /// The empty pack: zero unigrams, zero probes, zero total tokens —
    /// what `corpus-tool human-evidence` produces from zero `--input`
    /// directories, and what every pooling accessor treats as a no-op.
    #[must_use]
    pub fn empty() -> Self {
        Self::default()
    }

    /// Parses a `human-evidence-v1` `.bin` written by
    /// [`build_pack_bytes`].
    ///
    /// # Errors
    /// Returns [`PackError::HumanEvidenceTruncated`] if `bin` runs out of
    /// bytes mid-section, [`PackError::HumanEvidenceBadMagic`] if it
    /// doesn't start with this pack's magic, or
    /// [`PackError::HumanEvidenceUnsupportedVersion`] if its recorded
    /// format version isn't [`FORMAT_VERSION`].
    pub fn load(bin: &[u8]) -> Result<Self, PackError> {
        let mut r = Reader { bytes: bin };
        let magic = r.take(8, "header")?;
        if magic != MAGIC {
            return Err(PackError::HumanEvidenceBadMagic);
        }
        let version = r.u16("header")?;
        if version != FORMAT_VERSION {
            return Err(PackError::HumanEvidenceUnsupportedVersion(version));
        }
        let total_tokens = r.u64("header")?;

        let (word_count, word_offsets, word_pool) = read_string_table(&mut r, "interner")?;
        let word_at = |id: u32| string_at(word_offsets, word_pool, word_count, id, "interner");

        let unigram_count = r.u32("unigrams")?;
        let mut unigrams = BTreeMap::new();
        for _ in 0..unigram_count {
            let id = r.u32("unigrams")?;
            let count = r.u64("unigrams")?;
            unigrams.insert(word_at(id)?.to_string(), count);
        }

        let probe_count = r.u32("probes")?;
        let mut probes = BTreeMap::new();
        for _ in 0..probe_count {
            let len = r.u16("probes")?;
            let mut words = Vec::with_capacity(usize::from(len));
            for _ in 0..len {
                let id = r.u32("probes")?;
                words.push(word_at(id)?.to_string());
            }
            let count = r.u64("probes")?;
            probes.insert(words, count);
        }

        Ok(Self {
            total_tokens,
            unigrams,
            probes,
        })
    }

    /// This pack's external human-side token total — the pooling
    /// formula's external denominator contribution (see
    /// [`crate::frame_compile::CorpusEvidence`]'s own docs).
    #[must_use]
    pub const fn total_tokens(&self) -> u64 {
        self.total_tokens
    }

    /// `word`'s external occurrence count, `0` if absent — absence
    /// genuinely means "fewer than the builder's five-occurrence floor",
    /// a real (if imprecise) zero, never "not measured" (see this
    /// module's own docs).
    #[must_use]
    pub fn unigram_count(&self, word: &str) -> u64 {
        self.unigrams
            .get(&word.to_lowercase())
            .copied()
            .unwrap_or(0)
    }

    /// `words`' external non-overlapping occurrence count, if `words` was
    /// one of `frame-rules-v1.toml`'s own literal probes when this pack
    /// was built. `None` means exactly that it was not — "never
    /// measured", not "measured zero" (see this module's own docs).
    #[must_use]
    pub fn probe_count(&self, words: &[&str]) -> Option<u64> {
        let key: Vec<String> = words.iter().map(|w| w.to_lowercase()).collect();
        self.probes.get(&key).copied()
    }
}

/// Builds a `human-evidence-v1` pack's `.bin` bytes from already-pooled
/// evidence.
///
/// `unigrams` and `probes` are consumed in their own (`BTreeMap`)
/// ascending order, and the shared word interner is built from a
/// [`BTreeSet`] of every word either table references — so this function
/// is a pure function of its inputs: identical maps always produce
/// bit-identical bytes (pinned by this module's own
/// `build_pack_bytes_is_deterministic` test).
#[must_use]
pub fn build_pack_bytes(
    unigrams: &BTreeMap<String, u64>,
    probes: &BTreeMap<Vec<String>, u64>,
    total_tokens: u64,
) -> Vec<u8> {
    let mut words: BTreeSet<&str> = BTreeSet::new();
    for word in unigrams.keys() {
        words.insert(word.as_str());
    }
    for probe in probes.keys() {
        for word in probe {
            words.insert(word.as_str());
        }
    }
    let words: Vec<&str> = words.into_iter().collect();
    let ids: BTreeMap<&str, u32> = words
        .iter()
        .enumerate()
        .map(|(i, w)| (*w, u32::try_from(i).expect("word count fits u32")))
        .collect();

    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(&total_tokens.to_le_bytes());
    write_string_table(&mut out, words.iter().copied());

    out.extend_from_slice(
        &u32::try_from(unigrams.len())
            .expect("unigram count fits u32")
            .to_le_bytes(),
    );
    for (word, count) in unigrams {
        out.extend_from_slice(&ids[word.as_str()].to_le_bytes());
        out.extend_from_slice(&count.to_le_bytes());
    }

    out.extend_from_slice(
        &u32::try_from(probes.len())
            .expect("probe count fits u32")
            .to_le_bytes(),
    );
    for (probe, count) in probes {
        out.extend_from_slice(
            &u16::try_from(probe.len())
                .expect("probe word count fits u16")
                .to_le_bytes(),
        );
        for word in probe {
            out.extend_from_slice(&ids[word.as_str()].to_le_bytes());
        }
        out.extend_from_slice(&count.to_le_bytes());
    }
    out
}

/// Writes one string table: `u32` count, `count + 1` cumulative `u32`
/// offsets, `u32` pool length, pool bytes — same shape as
/// `frame_bin::write_string_table`, duplicated locally rather than shared
/// across the two modules' otherwise-unrelated serializers.
fn write_string_table<'a>(out: &mut Vec<u8>, strings: impl Iterator<Item = &'a str>) {
    let strings: Vec<&str> = strings.collect();
    out.extend_from_slice(
        &u32::try_from(strings.len())
            .expect("string count fits u32")
            .to_le_bytes(),
    );
    let mut offset = 0u32;
    out.extend_from_slice(&offset.to_le_bytes());
    for s in &strings {
        offset += u32::try_from(s.len()).expect("string length fits u32");
        out.extend_from_slice(&offset.to_le_bytes());
    }
    out.extend_from_slice(&offset.to_le_bytes());
    for s in &strings {
        out.extend_from_slice(s.as_bytes());
    }
}

/// A cursor over the artifact's bytes, erroring (never panicking) on
/// truncation.
struct Reader<'a> {
    bytes: &'a [u8],
}

impl<'a> Reader<'a> {
    const fn take(&mut self, len: usize, section: &'static str) -> Result<&'a [u8], PackError> {
        if self.bytes.len() < len {
            return Err(PackError::HumanEvidenceTruncated { section });
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

    fn u32(&mut self, section: &'static str) -> Result<u32, PackError> {
        Ok(u32::from_le_bytes(
            self.take(4, section)?.try_into().expect("4-byte split"),
        ))
    }

    fn u64(&mut self, section: &'static str) -> Result<u64, PackError> {
        Ok(u64::from_le_bytes(
            self.take(8, section)?.try_into().expect("8-byte split"),
        ))
    }

    fn str(&mut self, len: usize, section: &'static str) -> Result<&'a str, PackError> {
        std::str::from_utf8(self.take(len, section)?)
            .map_err(|_| PackError::HumanEvidenceTruncated { section })
    }
}

/// Reads one string table written by [`write_string_table`].
fn read_string_table<'a>(
    r: &mut Reader<'a>,
    section: &'static str,
) -> Result<(u32, &'a [u8], &'a str), PackError> {
    let count = r.u32(section)?;
    let offsets = r.take((count as usize + 1) * 4, section)?;
    let pool_len = r.u32(section)?;
    let pool = r.str(pool_len as usize, section)?;
    Ok((count, offsets, pool))
}

/// The string with this id from an offsets-and-pool table.
fn string_at<'a>(
    offsets: &[u8],
    pool: &'a str,
    count: u32,
    id: u32,
    section: &'static str,
) -> Result<&'a str, PackError> {
    if id >= count {
        return Err(PackError::HumanEvidenceTruncated { section });
    }
    let at = id as usize * 4;
    let start = u32::from_le_bytes(offsets[at..at + 4].try_into().expect("4-byte split")) as usize;
    let end =
        u32::from_le_bytes(offsets[at + 4..at + 8].try_into().expect("4-byte split")) as usize;
    pool.get(start..end)
        .ok_or(PackError::HumanEvidenceTruncated { section })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_unigrams() -> BTreeMap<String, u64> {
        BTreeMap::from([
            ("leverage".to_string(), 12),
            ("utilize".to_string(), 40),
            (".".to_string(), 900),
        ])
    }

    fn sample_probes() -> BTreeMap<Vec<String>, u64> {
        BTreeMap::from([
            (
                vec!["is".to_string(), "a".to_string(), "testament".to_string()],
                3,
            ),
            (vec!["duly".to_string(), "note".to_string()], 0),
        ])
    }

    #[test]
    fn build_and_load_round_trips_every_table() {
        let bin = build_pack_bytes(&sample_unigrams(), &sample_probes(), 12_345);
        let pack = HumanEvidencePack::load(&bin).expect("pack loads");
        assert_eq!(pack.total_tokens(), 12_345);
        assert_eq!(pack.unigram_count("leverage"), 12);
        assert_eq!(
            pack.unigram_count("UTILIZE"),
            40,
            "lookup is case-insensitive"
        );
        assert_eq!(pack.unigram_count("never-seen"), 0);
        assert_eq!(
            pack.probe_count(&["is", "a", "testament"]),
            Some(3),
            "a registered probe with real occurrences"
        );
        assert_eq!(
            pack.probe_count(&["duly", "note"]),
            Some(0),
            "a registered probe measured at zero is Some(0), not None"
        );
        assert_eq!(
            pack.probe_count(&["never", "registered"]),
            None,
            "an unregistered phrase is None, not Some(0)"
        );
    }

    #[test]
    fn build_pack_bytes_is_deterministic() {
        let a = build_pack_bytes(&sample_unigrams(), &sample_probes(), 12_345);
        let b = build_pack_bytes(&sample_unigrams(), &sample_probes(), 12_345);
        assert_eq!(a, b);
    }

    #[test]
    fn empty_pack_has_zero_everything() {
        let pack = HumanEvidencePack::empty();
        assert_eq!(pack.total_tokens(), 0);
        assert_eq!(pack.unigram_count("anything"), 0);
        assert_eq!(pack.probe_count(&["anything"]), None);
    }

    #[test]
    fn build_pack_bytes_from_empty_maps_round_trips_to_the_empty_pack() {
        let bin = build_pack_bytes(&BTreeMap::new(), &BTreeMap::new(), 0);
        let pack = HumanEvidencePack::load(&bin).expect("empty pack loads");
        assert_eq!(pack.total_tokens(), 0);
        assert_eq!(pack.unigram_count("x"), 0);
        assert_eq!(pack.probe_count(&["x"]), None);
    }

    #[test]
    fn load_rejects_truncated_bin() {
        let err = HumanEvidencePack::load(b"short").unwrap_err();
        assert!(matches!(err, PackError::HumanEvidenceTruncated { .. }));
    }

    #[test]
    fn load_rejects_bad_magic() {
        let mut bin = build_pack_bytes(&sample_unigrams(), &sample_probes(), 1);
        bin[0] = b'X';
        let err = HumanEvidencePack::load(&bin).unwrap_err();
        assert!(matches!(err, PackError::HumanEvidenceBadMagic));
    }

    #[test]
    fn load_rejects_unsupported_version() {
        let mut bin = build_pack_bytes(&sample_unigrams(), &sample_probes(), 1);
        bin[8] = 0xFF;
        bin[9] = 0xFF;
        let err = HumanEvidencePack::load(&bin).unwrap_err();
        assert!(matches!(err, PackError::HumanEvidenceUnsupportedVersion(_)));
    }
}
