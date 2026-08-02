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
//! table and keep single-document noise out): so a word's *absence* from
//! this table is itself real information ("fewer than five occurrences"),
//! not "never measured". [`Self::unigram_count`] returns a plain `u64`,
//! `0` for an absent word, reflecting that.
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

/// On-disk format version: bumped only if the layout changes shape.
/// Version 2 added the per-unigram burst envelope (the highest
/// per-document rate any contributing document sustained for that word
/// — see [`HumanEvidencePack::burst_envelope_per_million`]).
const FORMAT_VERSION: u16 = 2;

/// A parsed `human-evidence-v1` pack: pooled unigram and literal-probe
/// occurrence counts mined from staged external human corpora, plus the
/// external token total the pooling formula's denominator needs.
///
/// Two representations behind one query surface. [`Self::load`] builds
/// plain owned collections from any byte slice — the offline
/// `corpus-tool` path. [`Self::load_static`] instead borrows the
/// embedded artifact directly — the pack long outgrew the "stays small,
/// zero-copy buys nothing" assumption this module originally made (the
/// shipped table now spans ~50M staged review-corpus tokens, and
/// re-materializing its interner into owned `String`s measured ~10 ms of
/// the CLI's fixed startup), and the serialized layout happens to need
/// no second artifact for the view: the interner is written from a
/// [`BTreeSet`], hence byte-ascending and binary-searchable in place;
/// unigram entries are written in unigram-key order, hence id-ascending
/// (interner order IS key order) and binary-searchable in place; only
/// the variable-length probe entries need a load-built offset index,
/// and the probe table is bounded by the rule set's own size.
#[derive(Debug, Clone)]
pub struct HumanEvidencePack {
    repr: Repr,
}

#[derive(Debug, Clone)]
enum Repr {
    Owned {
        total_tokens: u64,
        unigrams: BTreeMap<String, (u64, u32)>,
        probes: BTreeMap<Vec<String>, u64>,
    },
    View(ViewRepr),
}

/// [`HumanEvidencePack::load_static`]'s zero-copy representation: raw
/// sections of the embedded artifact, plus the one small index the
/// variable-stride probe section needs.
#[derive(Debug, Clone)]
struct ViewRepr {
    total_tokens: u64,
    word_count: u32,
    /// `word_count + 1` cumulative `u32` byte offsets into `word_pool`.
    word_offsets: &'static [u8],
    word_pool: &'static str,
    /// [`UNIGRAM_ENTRY_LEN`]-byte `(id, count, burst)` entries, written
    /// in unigram-key order: id-ascending, because the interner's
    /// byte-ascending id assignment makes interner order and key order
    /// the same order.
    unigrams: &'static [u8],
    /// Byte offset of each probe entry into `probes`, entry order:
    /// entries are variable-stride, so binary search needs this index.
    probe_index: Box<[u32]>,
    probes: &'static [u8],
}

/// Bytes per serialized unigram entry: `id: u32`, `count: u64`,
/// `burst: u32`.
const UNIGRAM_ENTRY_LEN: usize = 16;

impl Default for HumanEvidencePack {
    fn default() -> Self {
        Self {
            repr: Repr::Owned {
                total_tokens: 0,
                unigrams: BTreeMap::new(),
                probes: BTreeMap::new(),
            },
        }
    }
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
            let burst = r.u32("unigrams")?;
            unigrams.insert(word_at(id)?.to_string(), (count, burst));
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
            repr: Repr::Owned {
                total_tokens,
                unigrams,
                probes,
            },
        })
    }

    /// [`Self::load`]'s zero-copy counterpart for the embedded artifact:
    /// borrows every section in place instead of re-materializing the
    /// interner into owned `String`s and B-trees, turning the ~10 ms
    /// owned load into structural validation plus one small probe-offset
    /// index. Query answers are identical (pinned by this module's
    /// owned-vs-view parity test).
    ///
    /// # Errors
    /// The same [`PackError`] family as [`Self::load`], plus
    /// `HumanEvidenceTruncated` for any ordering violation the view's
    /// binary searches rely on (interner not byte-ascending, unigram ids
    /// not ascending, probe keys not ascending) — never expected for a
    /// [`build_pack_bytes`]-written artifact, whose `BTreeMap`/`BTreeSet`
    /// iteration produces exactly those orders.
    pub fn load_static(bin: &'static [u8]) -> Result<Self, PackError> {
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
        for id in 1..word_count {
            let prev = string_at(word_offsets, word_pool, word_count, id - 1, "interner")?;
            let this = string_at(word_offsets, word_pool, word_count, id, "interner")?;
            if prev >= this {
                return Err(PackError::HumanEvidenceTruncated {
                    section: "interner order",
                });
            }
        }

        let unigram_count = r.u32("unigrams")?;
        let unigrams = r.take(unigram_count as usize * UNIGRAM_ENTRY_LEN, "unigrams")?;
        let mut prev_id: Option<u32> = None;
        for i in 0..unigram_count as usize {
            let id = u32::from_le_bytes(
                unigrams[i * UNIGRAM_ENTRY_LEN..i * UNIGRAM_ENTRY_LEN + 4]
                    .try_into()
                    .expect("4-byte split"),
            );
            if id >= word_count || prev_id.is_some_and(|prev| prev >= id) {
                return Err(PackError::HumanEvidenceTruncated {
                    section: "unigram order",
                });
            }
            prev_id = Some(id);
        }

        let probe_count = r.u32("probes")?;
        let probes = r.bytes;
        let mut probe_index = Vec::with_capacity(probe_count as usize);
        let mut prev_entry: Option<u32> = None;
        for _ in 0..probe_count {
            let offset =
                u32::try_from(probes.len() - r.bytes.len()).expect("probe section length fits u32");
            let len = r.u16("probes")?;
            for _ in 0..len {
                let id = r.u32("probes")?;
                if id >= word_count {
                    return Err(PackError::HumanEvidenceTruncated {
                        section: "probe ids",
                    });
                }
            }
            r.u64("probes")?;
            if let Some(prev) = prev_entry
                && compare_probe_entries(probes, prev, offset) != std::cmp::Ordering::Less
            {
                return Err(PackError::HumanEvidenceTruncated {
                    section: "probe order",
                });
            }
            prev_entry = Some(offset);
            probe_index.push(offset);
        }
        if !r.bytes.is_empty() {
            return Err(PackError::HumanEvidenceTruncated {
                section: "trailing bytes",
            });
        }

        Ok(Self {
            repr: Repr::View(ViewRepr {
                total_tokens,
                word_count,
                word_offsets,
                word_pool,
                unigrams,
                probe_index: probe_index.into_boxed_slice(),
                probes,
            }),
        })
    }

    /// This pack's external human-side token total — the pooling
    /// formula's external denominator contribution (see
    /// [`crate::frame_compile::CorpusEvidence`]'s own docs).
    #[must_use]
    pub const fn total_tokens(&self) -> u64 {
        match &self.repr {
            Repr::Owned { total_tokens, .. } => *total_tokens,
            Repr::View(view) => view.total_tokens,
        }
    }

    /// The `(count, burst)` unigram entry for an already-lowercased
    /// word, from whichever representation holds it.
    fn unigram_entry(&self, lower: &str) -> Option<(u64, u32)> {
        match &self.repr {
            Repr::Owned { unigrams, .. } => unigrams.get(lower).copied(),
            Repr::View(view) => view.unigram_entry(lower),
        }
    }

    /// `word`'s external occurrence count, `0` if absent: absence
    /// genuinely means "fewer than the builder's five-occurrence floor",
    /// a real (if imprecise) zero, never "not measured" (see this
    /// module's own docs).
    #[must_use]
    pub fn unigram_count(&self, word: &str) -> u64 {
        self.unigram_entry(&word.to_lowercase())
            .map_or(0, |(count, _)| count)
    }

    /// `word`'s burst envelope: the highest per-document per-million rate
    /// any single contributing document (of at least the builder's
    /// document-length floor) sustained for `word`. `None` when `word` is
    /// absent from the unigram table — no envelope evidence at all, which
    /// a consumer must treat as "cannot arm", never as "envelope zero".
    ///
    /// This is what makes a per-document overuse verdict topic-robust:
    /// human documents burst their own topic words too, so a topical
    /// word's envelope is naturally high, while a word no human document
    /// ever leaned on keeps a low one — the per-word analogue of the
    /// `envelope-v2` genre bands.
    #[must_use]
    pub fn burst_envelope_per_million(&self, word: &str) -> Option<u32> {
        self.unigram_entry(&word.to_lowercase())
            .map(|(_, burst)| burst)
    }

    /// `words`' external non-overlapping occurrence count, if `words` was
    /// one of `frame-rules-v1.toml`'s own literal probes when this pack
    /// was built. `None` means exactly that it was not: "never measured",
    /// not "measured zero" (see this module's own docs).
    #[must_use]
    pub fn probe_count(&self, words: &[&str]) -> Option<u64> {
        // Hyphens split exactly as the builder splits probe keys (and as
        // `friction_harness::clean::tokenize` splits the streams they
        // were counted against). A caller quoting "double-checking"
        // means the same two stream tokens the pack measured.
        let key: Vec<String> = words
            .iter()
            .flat_map(|w| w.split('-'))
            .filter(|w| !w.is_empty())
            .map(str::to_lowercase)
            .collect();
        match &self.repr {
            Repr::Owned { probes, .. } => probes.get(&key).copied(),
            Repr::View(view) => view.probe_count(&key),
        }
    }
}

impl ViewRepr {
    /// The interned id for an already-lowercased word — a binary search
    /// over the byte-ascending interner, [`load_static`]'s counterpart
    /// of the owned `BTreeMap` key walk.
    ///
    /// [`load_static`]: HumanEvidencePack::load_static
    fn word_id(&self, lower: &str) -> Option<u32> {
        let mut lo: u32 = 0;
        let mut hi: u32 = self.word_count;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let word = string_at(self.word_offsets, self.word_pool, self.word_count, mid, "")
                .expect("load_static validated every interner entry");
            match word.cmp(lower) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => return Some(mid),
            }
        }
        None
    }

    fn unigram_entry(&self, lower: &str) -> Option<(u64, u32)> {
        let id = self.word_id(lower)?;
        let mut lo: usize = 0;
        let mut hi: usize = self.unigrams.len() / UNIGRAM_ENTRY_LEN;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let at = mid * UNIGRAM_ENTRY_LEN;
            let entry_id =
                u32::from_le_bytes(self.unigrams[at..at + 4].try_into().expect("4-byte split"));
            match entry_id.cmp(&id) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => {
                    let count = u64::from_le_bytes(
                        self.unigrams[at + 4..at + 12]
                            .try_into()
                            .expect("8-byte split"),
                    );
                    let burst = u32::from_le_bytes(
                        self.unigrams[at + 12..at + 16]
                            .try_into()
                            .expect("4-byte split"),
                    );
                    return Some((count, burst));
                }
            }
        }
        None
    }

    fn probe_count(&self, key: &[String]) -> Option<u64> {
        // A key word outside the interner can't be part of any stored
        // probe — the owned B-tree would simply miss too.
        let mut key_ids = Vec::with_capacity(key.len());
        for word in key {
            key_ids.push(self.word_id(word)?);
        }

        // Interner ids ascend in word order, so lexicographic id-sequence
        // order IS the owned `BTreeMap<Vec<String>>` key order: binary
        // search over the load-built entry offsets.
        let mut lo: usize = 0;
        let mut hi: usize = self.probe_index.len();
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let at = self.probe_index[mid] as usize;
            match compare_probe_entry_to_key(self.probes, at, &key_ids) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => {
                    let len = u16::from_le_bytes(
                        self.probes[at..at + 2].try_into().expect("2-byte split"),
                    ) as usize;
                    let count_at = at + 2 + len * 4;
                    return Some(u64::from_le_bytes(
                        self.probes[count_at..count_at + 8]
                            .try_into()
                            .expect("8-byte split"),
                    ));
                }
            }
        }
        None
    }
}

/// Compares the probe entry at `at` against a query id sequence —
/// element-wise on ids, then by length, exactly `Vec<String>`'s
/// lexicographic order once ids stand in for words.
fn compare_probe_entry_to_key(probes: &[u8], at: usize, key_ids: &[u32]) -> std::cmp::Ordering {
    let len = u16::from_le_bytes(probes[at..at + 2].try_into().expect("2-byte split")) as usize;
    for (i, &key_id) in key_ids.iter().enumerate() {
        if i >= len {
            return std::cmp::Ordering::Less;
        }
        let id_at = at + 2 + i * 4;
        let entry_id =
            u32::from_le_bytes(probes[id_at..id_at + 4].try_into().expect("4-byte split"));
        match entry_id.cmp(&key_id) {
            std::cmp::Ordering::Equal => {}
            other => return other,
        }
    }
    len.cmp(&key_ids.len())
}

/// Compares two serialized probe entries — [`load_static`]'s
/// ascending-order validation.
///
/// [`load_static`]: HumanEvidencePack::load_static
fn compare_probe_entries(probes: &[u8], a: u32, b: u32) -> std::cmp::Ordering {
    let b_at = b as usize;
    let b_len = u16::from_le_bytes(probes[b_at..b_at + 2].try_into().expect("2-byte split"));
    let mut b_ids = Vec::with_capacity(usize::from(b_len));
    for i in 0..usize::from(b_len) {
        let id_at = b_at + 2 + i * 4;
        b_ids.push(u32::from_le_bytes(
            probes[id_at..id_at + 4].try_into().expect("4-byte split"),
        ));
    }
    compare_probe_entry_to_key(probes, a as usize, &b_ids)
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
    unigrams: &BTreeMap<String, (u64, u32)>,
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
    for (word, (count, burst)) in unigrams {
        out.extend_from_slice(&ids[word.as_str()].to_le_bytes());
        out.extend_from_slice(&count.to_le_bytes());
        out.extend_from_slice(&burst.to_le_bytes());
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

    fn sample_unigrams() -> BTreeMap<String, (u64, u32)> {
        BTreeMap::from([
            ("leverage".to_string(), (12, 4_000)),
            ("utilize".to_string(), (40, 9_000)),
            (".".to_string(), (900, 70_000)),
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

    /// The owned load and the zero-copy static view answer every query
    /// identically — the parity pin [`HumanEvidencePack::load_static`]'s
    /// docs promise.
    #[test]
    fn load_static_agrees_with_owned_load_on_every_query() {
        let bin = build_pack_bytes(&sample_unigrams(), &sample_probes(), 12_345);
        let owned = HumanEvidencePack::load(&bin).expect("pack loads");
        let leaked: &'static [u8] = Box::leak(bin.into_boxed_slice());
        let view = HumanEvidencePack::load_static(leaked).expect("static view loads");

        assert_eq!(view.total_tokens(), owned.total_tokens());
        for word in [
            "leverage",
            "UTILIZE",
            ".",
            "never-seen",
            "is",
            "testament",
            "",
        ] {
            assert_eq!(
                view.unigram_count(word),
                owned.unigram_count(word),
                "unigram_count({word:?}) must agree"
            );
            assert_eq!(
                view.burst_envelope_per_million(word),
                owned.burst_envelope_per_million(word),
                "burst_envelope_per_million({word:?}) must agree"
            );
        }
        let probes: [&[&str]; 6] = [
            &["is", "a", "testament"],
            &["duly", "note"],
            &["never", "registered"],
            &["is", "a"],
            &["is", "a", "testament", "to"],
            &["duly-note"],
        ];
        for probe in probes {
            assert_eq!(
                view.probe_count(probe),
                owned.probe_count(probe),
                "probe_count({probe:?}) must agree"
            );
        }
    }

    #[test]
    fn load_static_from_empty_maps_round_trips_to_the_empty_pack() {
        let bin = build_pack_bytes(&BTreeMap::new(), &BTreeMap::new(), 0);
        let leaked: &'static [u8] = Box::leak(bin.into_boxed_slice());
        let pack = HumanEvidencePack::load_static(leaked).expect("empty view loads");
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
