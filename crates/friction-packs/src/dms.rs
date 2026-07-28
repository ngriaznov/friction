//! DMS (differential matching statistics) index: the per-model-family
//! suffix automata a fix-time detection pass compares a document against.
//!
//! `corpus-tool index` only ever writes token-id streams plus a shared
//! vocabulary to a versioned TOML pack (`dms-index-v1.toml`); this module
//! is the read side, turning that pack into in-memory, queryable suffix
//! automata — the same "embedded TOML -> in-memory struct" shape
//! [`crate::EnvelopePack`] already established for this crate. A
//! detection crate built on top of this one (`friction-match`) wires a
//! [`DmsIndex`] into a running fix-time detection pass.
//!
//! [`Sam`] is a faithful transcription of the suffix-automaton reference
//! (`next`/`link`/`len` arrays, the exact clone-handling branch in
//! `extend`, and the matching-statistics walk), over `u32` token ids
//! rather than characters.

use std::collections::{BTreeMap, HashMap};

use serde::Deserialize;

use crate::PackError;

/// The five generator families a train-split `llm` document can belong to.
///
/// Per this workspace's model-manifest convention. A sixth family showing
/// up in the corpus manifest is a hard error at index-build time
/// (`corpus-tool index`'s own `family_of`), not a silent gap here — this
/// enum is exhaustive by design.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ModelFamily {
    Qwen,
    Gemma,
    Llama,
    Granite,
    Claude,
}

impl ModelFamily {
    /// Every family, in the fixed processing order `corpus-tool index`
    /// uses for deterministic vocabulary assignment.
    pub const ALL: [Self; 5] = [
        Self::Qwen,
        Self::Gemma,
        Self::Llama,
        Self::Granite,
        Self::Claude,
    ];

    /// The lowercase name used as this family's `[streams.<name>]` TOML
    /// table key.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Qwen => "qwen",
            Self::Gemma => "gemma",
            Self::Llama => "llama",
            Self::Granite => "granite",
            Self::Claude => "claude",
        }
    }
}

impl std::fmt::Display for ModelFamily {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A pack's shared token vocabulary.
///
/// `tokens[i]` is the text for token id `i`; id `0` is a reserved, unused
/// placeholder (see `corpus-tool index`'s own doc comment for why), so
/// array index equals id everywhere with no off-by-one arithmetic.
#[derive(Debug, Clone, Default)]
pub struct Vocab {
    tokens: Vec<Box<str>>,
    by_text: BTreeMap<Box<str>, u32>,
}

impl Vocab {
    /// Builds a vocab from its `tokens[i] == text for id i` array.
    ///
    /// If the same text appears at more than one index (not expected of a
    /// well-formed pack, whose ids are first-seen-order-unique, but not
    /// rejected here either), [`Vocab::id_of`] resolves to the *first*
    /// index that text appears at.
    #[must_use]
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

    /// The id for `token`'s exact text, or `None` if it is out-of-vocab —
    /// the `-1` case `ref_dms.py::ids_query` maps an unseen word to.
    #[must_use]
    pub fn id_of(&self, token: &str) -> Option<u32> {
        self.by_text.get(token).copied()
    }

    /// The text for `id`, or `None` if `id` is not a valid id in this
    /// vocab.
    #[must_use]
    pub fn token_of(&self, id: u32) -> Option<&str> {
        self.tokens.get(id as usize).map(AsRef::as_ref)
    }

    /// The number of token ids this vocab defines (including the reserved
    /// unused id `0`).
    #[must_use]
    pub const fn len(&self) -> usize {
        self.tokens.len()
    }

    /// `true` if this vocab defines no token ids at all.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

/// Suffix automaton over a `u32` token-id stream.
///
/// Faithful transcription of `ref_dms.py::SAM`: `next`/`link`/`len`
/// arrays, the exact clone-handling branch in `extend`. Rust's `Option<u32>`
/// stands in for Python's `-1` sentinel: `link[0]` (the root's own link)
/// is `None`, every other state's link is always `Some`.
///
/// `next` is a `HashMap` per state rather than a `BTreeMap`: unlike every
/// other collection in this codebase that is *iterated* to produce
/// output (documented explicitly in `friction-parse`'s extraction code
/// and elsewhere, where a `HashMap` would make output order
/// machine-dependent), `Sam::next` is only ever **point-queried**
/// (`next[v].get(&c)`) during both construction and matching — its
/// internal bucket order is never observed by any caller. The automaton
/// built from a fixed, ordered `ids` sequence is therefore bit-for-bit
/// reproducible regardless of the map's internal layout, while a
/// `HashMap` gives the point-lookup performance the "tens of MB/s per
/// core" throughput target needs. Do not "fix" this to a `BTreeMap` for
/// consistency with the rest of the codebase — that consistency does not
/// apply here, and it would cost real throughput for no benefit.
#[derive(Debug, Clone)]
pub struct Sam {
    next: Vec<HashMap<u32, u32>>,
    link: Vec<Option<u32>>,
    len: Vec<u32>,
    last: u32,
}

impl Default for Sam {
    fn default() -> Self {
        Self::new()
    }
}

impl Sam {
    /// A fresh automaton containing only the root state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            next: vec![HashMap::new()],
            link: vec![None],
            len: vec![0],
            last: 0,
        }
    }

    /// Builds a suffix automaton over `ids`, in order — `ref_dms.py`'s own
    /// `for c in stream: sam.extend(c)` loop.
    #[must_use]
    pub fn build(ids: &[u32]) -> Self {
        let mut sam = Self::new();
        for &id in ids {
            sam.extend(id);
        }
        sam
    }

    /// Extends the automaton by one token id. Transcribed 1:1 from
    /// `ref_dms.py::SAM.extend`, with Python's `-1` sentinel (`p == -1`,
    /// meaning "walked off the root") represented as `p: Option<u32> ==
    /// None`.
    fn extend(&mut self, c: u32) {
        #[allow(clippy::cast_possible_truncation)]
        let cur = self.next.len() as u32;
        self.next.push(HashMap::new());
        self.len.push(self.len[self.last as usize] + 1);
        self.link.push(None);

        let mut p: Option<u32> = Some(self.last);
        while let Some(pv) = p {
            if self.next[pv as usize].contains_key(&c) {
                break;
            }
            self.next[pv as usize].insert(c, cur);
            p = self.link[pv as usize];
        }

        match p {
            None => self.link[cur as usize] = Some(0),
            Some(pv) => {
                let q = self.next[pv as usize][&c];
                if self.len[pv as usize] + 1 == self.len[q as usize] {
                    self.link[cur as usize] = Some(q);
                } else {
                    #[allow(clippy::cast_possible_truncation)]
                    let clone = self.next.len() as u32;
                    let cloned_next = self.next[q as usize].clone();
                    self.next.push(cloned_next);
                    self.len.push(self.len[pv as usize] + 1);
                    self.link.push(self.link[q as usize]);

                    let mut p2: Option<u32> = Some(pv);
                    while let Some(p2v) = p2 {
                        if self.next[p2v as usize].get(&c) == Some(&q) {
                            self.next[p2v as usize].insert(c, clone);
                            p2 = self.link[p2v as usize];
                        } else {
                            break;
                        }
                    }
                    self.link[q as usize] = Some(clone);
                    self.link[cur as usize] = Some(clone);
                }
            }
        }
        self.last = cur;
    }

    /// The number of states, including the root — the serialization bound
    /// `crate::dms_bin`'s packer walks.
    pub(crate) const fn state_count(&self) -> usize {
        self.len.len()
    }

    /// State `s`'s `len` value (longest string reaching it).
    pub(crate) fn len_of(&self, s: usize) -> u32 {
        self.len[s]
    }

    /// State `s`'s suffix link (`None` only for the root).
    pub(crate) fn link_of(&self, s: usize) -> Option<u32> {
        self.link[s]
    }

    /// State `s`'s outgoing transitions, sorted by label — the one place
    /// `next`'s `HashMap` is ever *iterated*, and only after an explicit
    /// sort, so the machine-dependent bucket order documented on [`Sam`]
    /// never reaches the serialized artifact.
    pub(crate) fn sorted_transitions(&self, s: usize) -> Vec<(u32, u32)> {
        let mut transitions: Vec<(u32, u32)> =
            self.next[s].iter().map(|(&c, &to)| (c, to)).collect();
        transitions.sort_unstable();
        transitions
    }

    /// The longest match ENDING at each position of `query`, walking the
    /// automaton exactly as `ref_dms.py::matching_stats` does. `None` in
    /// `query` is the `-1` sentinel: a query token absent from the shared
    /// vocabulary entirely (out-of-vocab), which always forces `l` to `0`
    /// and the walk back to the root state, the same as a known id this
    /// automaton's alphabet simply never saw following the current state.
    #[must_use]
    pub fn matching_stats(&self, query: &[Option<u32>]) -> Vec<u32> {
        let mut v: u32 = 0;
        let mut l: u32 = 0;
        let mut out = Vec::with_capacity(query.len());

        for &c in query {
            let direct = c.and_then(|id| self.next[v as usize].get(&id).copied());
            if let Some(next_v) = direct {
                v = next_v;
                l += 1;
            } else {
                while v != 0 {
                    let has_transition =
                        c.is_some_and(|id| self.next[v as usize].contains_key(&id));
                    if has_transition {
                        break;
                    }
                    v = self.link[v as usize]
                        .expect("every non-root state has a link, by construction in extend");
                    l = self.len[v as usize];
                }
                let landed = c.and_then(|id| self.next[v as usize].get(&id).copied());
                if let Some(next_v) = landed {
                    v = next_v;
                    l += 1;
                } else {
                    v = 0;
                    l = 0;
                }
            }
            out.push(l);
        }
        out
    }
}

/// The TOML shape of `[vocab]`.
#[derive(Debug, Deserialize)]
struct RawVocab {
    tokens: Vec<String>,
}

/// The TOML shape of one `[streams.<name>]` table.
#[derive(Debug, Deserialize)]
struct RawStream {
    /// Comma-separated `u32` token ids — see `corpus-tool index`'s doc
    /// comment for why this is one big string rather than a native TOML
    /// integer array.
    ids: String,
}

/// The TOML shape of `[streams]`: the always-present `human` stream plus
/// the five optional per-family streams.
#[derive(Debug, Deserialize)]
struct RawStreams {
    human: RawStream,
    #[serde(default)]
    qwen: Option<RawStream>,
    #[serde(default)]
    gemma: Option<RawStream>,
    #[serde(default)]
    llama: Option<RawStream>,
    #[serde(default)]
    granite: Option<RawStream>,
    #[serde(default)]
    claude: Option<RawStream>,
}

/// The TOML shape of a `dms-index-v1` pack as a whole. The `[pack]`
/// metadata table and `[pack.families]` doc/token-count summaries are not
/// named here (nothing in this module needs them to build a working
/// [`DmsIndex`]) and are simply ignored by serde rather than rejected.
#[derive(Debug, Deserialize)]
struct RawPack {
    vocab: RawVocab,
    streams: RawStreams,
}

/// A parsed DMS index pack: the shared vocabulary plus one suffix
/// automaton per stream (`human`, and whichever of the five
/// [`ModelFamily`] streams the pack defines).
#[derive(Debug, Clone)]
pub struct DmsIndex {
    vocab: Vocab,
    human: Sam,
    families: BTreeMap<ModelFamily, Sam>,
}

impl DmsIndex {
    /// Parses a `dms-index-v1`-shaped pack: a `[vocab]` table plus a
    /// `[streams.human]` table and any of the five `[streams.<family>]`
    /// tables, building a [`Vocab`] and one [`Sam`] per stream present.
    ///
    /// # Errors
    /// Returns [`PackError::Toml`] if `toml` is not valid TOML in the
    /// expected shape (missing `[vocab]`/`[streams.human]`, or a
    /// `tokens`/`ids` field of the wrong type). Returns
    /// [`PackError::DmsMalformedIds`] if a stream's `ids` field contains a
    /// value that doesn't parse as `u32`, and
    /// [`PackError::DmsIdOutOfRange`] if a stream references a token id
    /// past the end of the pack's own `[vocab]` table.
    pub fn parse(toml: &str) -> Result<Self, PackError> {
        let raw: RawPack = toml::from_str(toml).map_err(PackError::from)?;
        let vocab = Vocab::from_tokens(raw.vocab.tokens);

        let human = Self::build_stream_sam(&vocab, "human", &raw.streams.human.ids)?;

        let mut families: BTreeMap<ModelFamily, Sam> = BTreeMap::new();
        let raw_family_streams = [
            (ModelFamily::Qwen, &raw.streams.qwen),
            (ModelFamily::Gemma, &raw.streams.gemma),
            (ModelFamily::Llama, &raw.streams.llama),
            (ModelFamily::Granite, &raw.streams.granite),
            (ModelFamily::Claude, &raw.streams.claude),
        ];
        for (family, raw_stream) in raw_family_streams {
            if let Some(stream) = raw_stream {
                let sam = Self::build_stream_sam(&vocab, family.as_str(), &stream.ids)?;
                families.insert(family, sam);
            }
        }

        Ok(Self {
            vocab,
            human,
            families,
        })
    }

    /// Parses one stream's `ids` field and builds its [`Sam`], validating
    /// every id against `vocab` first.
    fn build_stream_sam(vocab: &Vocab, stream_name: &str, ids_csv: &str) -> Result<Sam, PackError> {
        let ids = parse_ids_csv(stream_name, ids_csv)?;
        for &id in &ids {
            if id as usize >= vocab.len() {
                return Err(PackError::DmsIdOutOfRange {
                    stream: stream_name.to_string(),
                    id,
                    vocab_len: vocab.len(),
                });
            }
        }
        Ok(Sam::build(&ids))
    }

    /// This pack's shared token vocabulary.
    #[must_use]
    pub const fn vocab(&self) -> &Vocab {
        &self.vocab
    }

    /// The human-corpus suffix automaton.
    #[must_use]
    pub const fn human_sam(&self) -> &Sam {
        &self.human
    }

    /// The suffix automaton for `family`, or `None` if this pack defines
    /// no stream for that family.
    #[must_use]
    pub fn family_sam(&self, family: ModelFamily) -> Option<&Sam> {
        self.families.get(&family)
    }

    /// Every family stream this pack defines, in [`ModelFamily`] order —
    /// the iteration `crate::dms_bin`'s packer serializes streams in.
    pub(crate) fn family_sams(&self) -> impl Iterator<Item = (ModelFamily, &Sam)> {
        self.families.iter().map(|(&family, sam)| (family, sam))
    }
}

/// Parses a comma-separated `u32` list (the `ids` field's own encoding —
/// see `corpus-tool index`'s doc comment). An empty (post-trim) string
/// parses to an empty id list rather than a single malformed empty field.
fn parse_ids_csv(stream_name: &str, csv: &str) -> Result<Vec<u32>, PackError> {
    let trimmed = csv.trim();
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }
    trimmed
        .split(',')
        .map(|part| {
            part.trim()
                .parse::<u32>()
                .map_err(|_| PackError::DmsMalformedIds {
                    stream: stream_name.to_string(),
                    value: part.to_string(),
                })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- ModelFamily ---

    #[test]
    fn model_family_as_str_round_trips_for_every_family() {
        assert_eq!(ModelFamily::Qwen.as_str(), "qwen");
        assert_eq!(ModelFamily::Gemma.as_str(), "gemma");
        assert_eq!(ModelFamily::Llama.as_str(), "llama");
        assert_eq!(ModelFamily::Granite.as_str(), "granite");
        assert_eq!(ModelFamily::Claude.as_str(), "claude");
        assert_eq!(ModelFamily::ALL.len(), 5);
    }

    // --- Vocab ---

    #[test]
    fn vocab_id_of_and_token_of_round_trip() {
        let vocab = Vocab::from_tokens(vec![
            "<unused-0>".to_string(),
            "the".to_string(),
            "a".to_string(),
        ]);
        assert_eq!(vocab.id_of("the"), Some(1));
        assert_eq!(vocab.id_of("a"), Some(2));
        assert_eq!(vocab.id_of("nonexistent"), None);
        assert_eq!(vocab.token_of(1), Some("the"));
        assert_eq!(vocab.token_of(99), None);
        assert_eq!(vocab.len(), 3);
        assert!(!vocab.is_empty());
    }

    // --- Sam: hand-computed matching statistics ---

    /// Corpus `[1,2,3,2,3]` ("a b c b c"). Querying the corpus back
    /// against its own automaton gives a strictly growing match length at
    /// every position, ending at the full length: every prefix of the
    /// corpus is (trivially) a substring of the corpus.
    ///
    /// Cross-checked by re-running `ref_dms.py`'s own `SAM`/
    /// `matching_stats` (verbatim, offline) against this exact input:
    /// `[1, 2, 3, 4, 5]`.
    #[test]
    fn matching_stats_self_query_grows_to_full_length() {
        let sam = Sam::build(&[1, 2, 3, 2, 3]);
        let query: Vec<Option<u32>> = vec![Some(1), Some(2), Some(3), Some(2), Some(3)];
        assert_eq!(sam.matching_stats(&query), vec![1, 2, 3, 4, 5]);
    }

    /// Same corpus. A query id genuinely absent from the corpus (`9`,
    /// never extended into the automaton at all) produces `0` at that
    /// position and forces a reset to the root — matching lengths after
    /// it recover from empty match state, not from wherever the walk was
    /// before the miss.
    ///
    /// Cross-checked against `ref_dms.py`: `[1, 2, 0, 1, 2]`.
    #[test]
    fn matching_stats_unseen_id_produces_zero_and_resets() {
        let sam = Sam::build(&[1, 2, 3, 2, 3]);
        let query: Vec<Option<u32>> = vec![Some(1), Some(2), Some(9), Some(2), Some(3)];
        assert_eq!(sam.matching_stats(&query), vec![1, 2, 0, 1, 2]);
    }

    /// Same corpus and query shape as the unseen-id case, but the missing
    /// position is `None` (the out-of-vocabulary / `-1` sentinel) rather
    /// than a known-but-absent id. Produces the identical curve — both
    /// are "no transition for this token", just for different reasons.
    ///
    /// Cross-checked against `ref_dms.py`: `[1, 2, 0, 1, 2]`.
    #[test]
    fn matching_stats_unknown_token_produces_zero_and_resets() {
        let sam = Sam::build(&[1, 2, 3, 2, 3]);
        let query: Vec<Option<u32>> = vec![Some(1), Some(2), None, Some(2), Some(3)];
        assert_eq!(sam.matching_stats(&query), vec![1, 2, 0, 1, 2]);
    }

    /// A second, independent hand/reference-checked example: corpus
    /// `[1,2,1,2,3]` ("a b a b c"), queried with `[2,1,2,3,1]` ("b a b c
    /// a"). The match grows through `"baba"`'s longest run
    /// (`b`,`ba`,`bab`,`babc`... — actually the corpus contains `"abab"`,
    /// so `"b"`,`"ab"`... see the reference cross-check) and drops back
    /// on the final `a` because `"ca"` never occurs in the corpus, only a
    /// bare `"a"` does.
    ///
    /// Cross-checked against `ref_dms.py`: `[1, 2, 3, 4, 1]`.
    #[test]
    fn matching_stats_matches_second_reference_cross_check() {
        let sam = Sam::build(&[1, 2, 1, 2, 3]);
        let query: Vec<Option<u32>> = vec![Some(2), Some(1), Some(2), Some(3), Some(1)];
        assert_eq!(sam.matching_stats(&query), vec![1, 2, 3, 4, 1]);
    }

    // --- Sam: determinism ---

    /// Building from the same id slice twice and querying the same query
    /// twice gives identical output both times — this is exactly what
    /// makes the `HashMap`-for-`next` choice safe (see this module's own
    /// doc comment on [`Sam`]).
    #[test]
    fn sam_build_and_matching_stats_are_deterministic() {
        let ids = [4, 1, 2, 3, 2, 3, 1, 4, 2];
        let sam_a = Sam::build(&ids);
        let sam_b = Sam::build(&ids);
        let query: Vec<Option<u32>> = ids.iter().map(|&i| Some(i)).collect();
        assert_eq!(
            sam_a.matching_stats(&query),
            sam_b.matching_stats(&query),
            "two independently built automata over the same ids must agree"
        );
        assert_eq!(
            sam_a.matching_stats(&query),
            sam_a.matching_stats(&query),
            "querying the same automaton twice must be byte-identical"
        );
    }

    // --- DmsIndex::parse ---

    const SAMPLE_PACK: &str = r#"
        [pack]
        version = "dms-index-v1"

        [vocab]
        tokens = ["<unused-0>", "the", "quick", "fox"]

        [streams.human]
        ids = "1,2,3"

        [streams.qwen]
        ids = "2,3,1"
    "#;

    #[test]
    fn parse_builds_vocab_and_working_sams_for_present_streams() {
        let index = DmsIndex::parse(SAMPLE_PACK).expect("sample pack parses");
        assert_eq!(index.vocab().len(), 4);
        assert_eq!(index.vocab().id_of("quick"), Some(2));

        let human = index.human_sam();
        let query: Vec<Option<u32>> = vec![Some(1), Some(2), Some(3)];
        assert_eq!(human.matching_stats(&query), vec![1, 2, 3]);

        assert!(index.family_sam(ModelFamily::Qwen).is_some());
        assert!(index.family_sam(ModelFamily::Gemma).is_none());
    }

    #[test]
    fn parse_rejects_non_numeric_ids() {
        let bad = r#"
            [vocab]
            tokens = ["<unused-0>", "the"]

            [streams.human]
            ids = "1,not-a-number"
        "#;
        let err = DmsIndex::parse(bad).expect_err("non-numeric id must be rejected");
        assert!(matches!(err, PackError::DmsMalformedIds { .. }));
    }

    #[test]
    fn parse_rejects_id_out_of_range_for_vocab() {
        let bad = r#"
            [vocab]
            tokens = ["<unused-0>", "the"]

            [streams.human]
            ids = "1,99"
        "#;
        let err = DmsIndex::parse(bad).expect_err("out-of-range id must be rejected");
        assert!(matches!(err, PackError::DmsIdOutOfRange { .. }));
    }

    #[test]
    fn parse_rejects_malformed_toml() {
        assert!(DmsIndex::parse("not [ valid toml").is_err());
    }

    #[test]
    fn parse_rejects_missing_required_table() {
        let bad = r#"
            [vocab]
            tokens = ["<unused-0>"]
        "#;
        assert!(DmsIndex::parse(bad).is_err());
    }

    #[test]
    fn parse_handles_empty_stream_gracefully() {
        let pack = r#"
            [vocab]
            tokens = ["<unused-0>"]

            [streams.human]
            ids = ""
        "#;
        let index = DmsIndex::parse(pack).expect("empty ids stream parses");
        assert_eq!(index.human_sam().matching_stats(&[]), Vec::<u32>::new());
    }
}
