//! Averaged-perceptron [`Tagger`] implementation: the default backend.
//!
//! Unlike [`crate::tag_nlprule::NlpruleTagger`], this tagger's weights are
//! not downloaded at build time — they are vendored directly in this
//! crate's own `weights/perceptron_en.json.gz`, trained on a small,
//! hand-curated gold-tag corpus drawn from this project's own vendored
//! human documents (see `weights/NOTICE.md` for provenance: source
//! documents, token count, annotation method, and the reproduction
//! command). Nothing here ever touches the network, at build time or at
//! run time.
//!
//! # Tokenization is its own, not `nlprule`'s
//!
//! [`tokenize`] is a small manual scanner producing spans that satisfy
//! [`crate::tag::classify_token_kind`]'s tokenization rule directly:
//! maximal alphabetic/`'`/`-` runs as [`TokenKind::Word`], a numeric-leading
//! run of digits/`.`/`,` as [`TokenKind::Number`], maximal runs of
//! [`crate::tag::is_prose_punctuation`] characters grouped as one
//! [`TokenKind::Punctuation`] token, and everything else grouped as
//! [`TokenKind::Symbol`]. No whitespace tokens are emitted.
//!
//! # Non-word tokens: a deterministic passthrough, never the model
//!
//! [`TokenKind::Number`] gets `"CD"`; [`TokenKind::Punctuation`] gets its
//! own literal surface text as its tag (matching genuine Penn-Treebank/
//! CoNLL-U convention for punctuation); [`TokenKind::Symbol`] gets `"SYM"`.
//! These categories are effectively fully determined by surface form in
//! Penn-Treebank-style tagging, so hardcoding them removes risk without
//! costing accuracy — and using this exact same fixed value for tag-history
//! context during both training and inference keeps the two consistent.
//! Only [`TokenKind::Word`] tokens ever run the perceptron.
//!
//! # Lemma: best-effort, reusing this crate's own inflection tables
//!
//! There is no dictionary lemmatizer here — [`crate::inflect::lemmatize`]
//! guesses a `Word` token's base form by generating candidate stems from
//! its own tag (gerund/past/plural-or-third-singular) and keeping whichever
//! one round-trips back to the surface text through
//! [`crate::inflect::inflect`]'s own forward generation, reusing that
//! module's tables and rules in reverse rather than duplicating them or
//! adding new ones. A candidate that fails to round-trip (irregular in a
//! way the tables don't cover, or simply a base-form word to begin with)
//! falls back to the lowercased surface text — matching
//! [`TaggedToken::lemma`]'s own documented fallback ("equal to the token's
//! own surface text... when the tagger has no lemma for it"). Non-`Word`
//! tokens always get the lowercased surface text directly.
//!
//! # Determinism
//!
//! The feature list for each token is a fixed-order `Vec<Box<str>>`, built
//! from a fixed code path rather than iterated out of a hash collection, so
//! summation order (and therefore the resulting float scores) is identical
//! run to run. Scoring indexes a plain array by each class's load-time
//! sorted position and picks the argmax by strict `>` comparison scanning
//! in that fixed order, so ties always resolve to the lexicographically
//! earliest tag — never to hash-map iteration order.
//!
//! # Per-call sentence-reset semantics
//!
//! Unlike [`crate::tag_nlprule::NlpruleTagger::tag`], which re-segments
//! multi-sentence `text` internally, [`PerceptronTagger::tag`] does **not**
//! re-segment: it treats the whole `text` argument as one tag-history
//! sequence, resetting the start-of-sequence sentinels once. Every
//! production call site in this workspace already calls `tag()` per
//! already-segmented sentence, so this is a safe simplification — but a
//! caller must not pass an unsegmented paragraph expecting sentence-internal
//! resets.

use std::collections::{BTreeMap, HashMap};
use std::io::Read as _;
use std::ops::Range;

use friction_core::{Token, TokenKind};
use serde::{Deserialize, Serialize};

use crate::tag::{PosTag, TaggedToken, Tagger, classify_token_kind, is_prose_punctuation};

/// The vendored, gzip-compressed perceptron weight artifact. See
/// `weights/NOTICE.md` for provenance and the reproduction command;
/// `examples/train_perceptron.rs` (behind the `train-tooling` feature)
/// regenerates it.
static WEIGHTS_BYTES: &[u8] = include_bytes!("../weights/perceptron_en.json.gz");

/// Sentinel fed as tag-history context before the first token (and, one
/// position further back, before the second token).
const START_TAG_1: &str = "-START-";
const START_TAG_2: &str = "-START2-";

/// Sentinel fed as word-context when a feature looks past either edge of
/// the token stream.
const START_WORD_1: &str = "-start-";
const START_WORD_2: &str = "-start2-";
const END_WORD_1: &str = "-end-";
const END_WORD_2: &str = "-end2-";

/// Errors constructing a [`PerceptronTagger`].
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum PerceptronTagError {
    /// The embedded weight artifact failed to gunzip. Points at a build
    /// problem with the vendored artifact, not a runtime condition, since
    /// the bytes are compiled in.
    #[error("failed to decompress the embedded perceptron weight artifact: {0}")]
    Decompress(#[source] std::io::Error),
    /// The decompressed artifact was not valid JSON in the expected shape.
    #[error("failed to parse the embedded perceptron weight artifact: {0}")]
    Parse(#[source] serde_json::Error),
}

/// On-disk shape of the weight artifact: `BTreeMap`s throughout so the
/// serialized JSON is stable and diffable across re-training runs.
///
/// Public (rather than crate-private) only because the `train-tooling`
/// feature's `train_support` module hands one back to an external training
/// tool (`examples/train_perceptron.rs`, compiled as its own crate) — the
/// type is otherwise unreachable from outside this module, since
/// `tag_perceptron` itself is a private module and this struct is never
/// re-exported directly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightFile {
    pub classes: Vec<Box<str>>,
    /// feature string -> sparse `[(class index into `classes`, weight)]`.
    pub features: BTreeMap<Box<str>, Vec<(u16, f32)>>,
    /// A frequent/unambiguous word's tag, bypassing the model entirely.
    pub tagdict: BTreeMap<Box<str>, Box<str>>,
}

/// Runtime weight table: one dense per-class weight vector per feature
/// string, indexed by each class's position in [`PerceptronTagger`]'s own
/// (freshly sorted, at load) `classes` list — never by whatever order the
/// on-disk file happened to use.
struct WeightTable {
    num_classes: usize,
    by_feature: HashMap<Box<str>, Vec<f32>>,
}

impl WeightTable {
    /// Builds a `(sorted classes, WeightTable)` pair from a [`WeightFile`],
    /// translating each stored class index through a lookup into the
    /// freshly sorted class list so the table is correct regardless of
    /// what order the file's own `classes` array was written in.
    fn from_file(file: WeightFile) -> (Vec<Box<str>>, Self) {
        let WeightFile {
            classes: file_classes,
            features,
            ..
        } = file;

        let mut classes = file_classes.clone();
        classes.sort();

        let index_map: Vec<usize> = file_classes
            .iter()
            .map(|class| {
                classes
                    .binary_search(class)
                    .expect("every class in a weight file's `classes` list must appear in it")
            })
            .collect();

        let num_classes = classes.len();
        let mut by_feature = HashMap::with_capacity(features.len());
        for (feature, sparse) in features {
            let mut dense = vec![0f32; num_classes];
            for (class_index, weight) in sparse {
                if let Some(mapped) = index_map.get(class_index as usize) {
                    dense[*mapped] = weight;
                }
            }
            by_feature.insert(feature, dense);
        }

        (
            classes,
            Self {
                num_classes,
                by_feature,
            },
        )
    }

    /// Sums each feature's dense per-class weight vector, in `features`'
    /// own fixed order — the determinism-critical step: float summation
    /// order is identical across runs because iteration order here is
    /// fixed by the caller-supplied `Vec`, never by hash-map iteration.
    fn score(&self, features: &[Box<str>]) -> Vec<f32> {
        let mut scores = vec![0f32; self.num_classes];
        for feature in features {
            if let Some(dense) = self.by_feature.get(feature) {
                for (slot, weight) in scores.iter_mut().zip(dense.iter()) {
                    *slot += weight;
                }
            }
        }
        scores
    }
}

/// The index of the highest-scoring class in `scores`, breaking ties in
/// favor of the lowest index — since `classes` is sorted ascending at load
/// time, this is exactly "the lexicographically earliest tag wins ties".
fn argmax(scores: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_score = scores[0];
    for (index, &score) in scores.iter().enumerate().skip(1) {
        if score > best_score {
            best_score = score;
            best = index;
        }
    }
    best
}

/// A word's *identity* feature value, normalized for generalization past a
/// small training vocabulary: lowercased, and — because a small hand-tagged
/// corpus cannot enumerate every hyphenated compound or numeral-bearing
/// token it might meet at inference time — collapsed to one of three
/// closed, documented sentinel buckets, rather than kept as a literal (and
/// likely unseen) string. Mirrors real NLTK's own `_normalize` bucket
/// order and conditions (hyphen, then four-digit year, then any other
/// leading-digit token) rather than a looser "contains a digit anywhere" /
/// "contains a hyphen anywhere" test, so a word like `"co2"` or `"a-1"`
/// buckets the same way NLTK's tagger would.
///
/// Only the *identity* feature is bucketed this way; suffix and
/// first-character features always read the literal word (see
/// [`extract_features`]), since those stay informative even for a bucketed
/// word.
fn normalize_word_identity(word_lower: &str) -> &str {
    let Some(first) = word_lower.chars().next() else {
        return word_lower;
    };
    if first != '-' && word_lower.contains('-') {
        return "!hyphenated!";
    }
    if word_lower.chars().count() == 4 && word_lower.chars().all(|c| c.is_ascii_digit()) {
        return "!year!";
    }
    if first.is_ascii_digit() {
        return "!contains_digit!";
    }
    word_lower
}

/// The last (up to) three characters of `word_lower`, by `char` count, not
/// byte count — a short word's own full text if it has three characters or
/// fewer.
fn suffix3(word_lower: &str) -> &str {
    let byte_start = word_lower
        .char_indices()
        .rev()
        .nth(2)
        .map_or(0, |(index, _)| index);
    &word_lower[byte_start..]
}

/// `word_lower`'s first character, as a `&str` (empty for an empty word,
/// which should never occur for a real [`TokenKind::Word`] token).
fn first_char(word_lower: &str) -> &str {
    match word_lower.char_indices().nth(1) {
        Some((end, _)) => &word_lower[..end],
        None => word_lower,
    }
}

/// `surfaces[idx]`'s lowercased text, or a start-/end-of-sequence sentinel
/// when `idx` looks past either edge — shared by every "word two back" /
/// "previous word" / "next word" / "word two forward" feature.
fn context_word(surfaces: &[Box<str>], idx: isize) -> &str {
    if idx < 0 {
        return if idx == -1 {
            START_WORD_1
        } else {
            START_WORD_2
        };
    }
    // `idx >= 0` was just checked, and every real token stream is far
    // smaller than `isize::MAX`, so this conversion never loses the sign.
    #[allow(clippy::cast_sign_loss)]
    let idx = idx as usize;
    if idx >= surfaces.len() {
        return if idx == surfaces.len() {
            END_WORD_1
        } else {
            END_WORD_2
        };
    }
    &surfaces[idx]
}

/// Builds token `i`'s fixed-order, 14-entry feature list: bias; current
/// word; current word's 3-char suffix; current word's first character;
/// previous tag; tag two back; previous-tag×tag-two-back; previous-tag×
/// current-word; previous word; previous word's 3-char suffix; word two
/// back; next word; next word's 3-char suffix; word two forward.
///
/// `surfaces` is every token's lowercased surface text in the whole `tag()`
/// call (word or not — punctuation legitimately participates in word
/// context, the same way it does in genuine Penn-Treebank tagging).
/// `raw_current` is token `i`'s surface text in its *original* case: real
/// NLTK's own `_get_features` computes `word[-3:]`/`word[0]` from the raw,
/// un-normalized token (only the identity feature — `context[i]` — reads
/// the lowercased form), so this is the one place in the whole feature list
/// that ever sees capitalization; every neighboring-word suffix/identity
/// feature below deliberately still reads the lowercased `surfaces` entry,
/// matching NLTK's `context[i-1]`/`context[i+1]` exactly. `prev1_tag`/
/// `prev2_tag` are the tags already assigned to tokens `i-1`/`i-2` (or a
/// start-of-sequence sentinel).
fn extract_features(
    surfaces: &[Box<str>],
    raw_current: &str,
    i: usize,
    prev1_tag: &str,
    prev2_tag: &str,
) -> Vec<Box<str>> {
    // Every real token stream is far smaller than `isize::MAX`, so this
    // conversion never wraps.
    #[allow(clippy::cast_possible_wrap)]
    let i = i as isize;
    let cur_lower = context_word(surfaces, i);
    let cur_word = normalize_word_identity(cur_lower);
    let cur_suffix = suffix3(raw_current);
    let cur_pref1 = first_char(raw_current);

    let prev_raw = context_word(surfaces, i - 1);
    let prev_word = normalize_word_identity(prev_raw);
    let prev_suffix = suffix3(prev_raw);

    let word_two_back = normalize_word_identity(context_word(surfaces, i - 2));

    let next_raw = context_word(surfaces, i + 1);
    let next_word = normalize_word_identity(next_raw);
    let next_suffix = suffix3(next_raw);

    let word_two_forward = normalize_word_identity(context_word(surfaces, i + 2));

    vec![
        Box::from("bias"),
        Box::from(format!("word={cur_word}")),
        Box::from(format!("suffix={cur_suffix}")),
        Box::from(format!("pref1={cur_pref1}")),
        Box::from(format!("i-1 tag={prev1_tag}")),
        Box::from(format!("i-2 tag={prev2_tag}")),
        Box::from(format!("i-1 tag+i-2 tag={prev1_tag},{prev2_tag}")),
        Box::from(format!("i-1 tag+word={prev1_tag},{cur_word}")),
        Box::from(format!("i-1 word={prev_word}")),
        Box::from(format!("i-1 suffix={prev_suffix}")),
        Box::from(format!("i-2 word={word_two_back}")),
        Box::from(format!("i+1 word={next_word}")),
        Box::from(format!("i+1 suffix={next_suffix}")),
        Box::from(format!("i+2 word={word_two_forward}")),
    ]
}

/// One scanned token: its byte range and coarse kind, before tagging.
struct ScannedToken {
    range: Range<usize>,
    kind: TokenKind,
}

/// Splits `text` into [`ScannedToken`]s: maximal alphabetic/`'`/`-` runs
/// containing at least one alphabetic character as [`TokenKind::Word`]; a
/// numeric-leading run of digits/`.`/`,` as [`TokenKind::Number`]; maximal
/// runs of [`is_prose_punctuation`] characters (including a bare run of
/// `'`/`-` with no alphabetic character at all) as one
/// [`TokenKind::Punctuation`] token; everything else grouped into maximal
/// runs as [`TokenKind::Symbol`]. Whitespace is skipped, never emitted as a
/// token. This is the same partition [`classify_token_kind`] describes for
/// a single already-split token, applied here to decide the splits
/// themselves.
fn tokenize(text: &str) -> Vec<ScannedToken> {
    let mut tokens = Vec::new();
    let mut chars = text.char_indices().peekable();

    while let Some(&(start, c)) = chars.peek() {
        if c.is_whitespace() {
            chars.next();
            continue;
        }

        if c.is_numeric() {
            let mut end = start + c.len_utf8();
            chars.next();
            while let Some(&(idx, next)) = chars.peek() {
                if next.is_numeric() || next == '.' || next == ',' {
                    end = idx + next.len_utf8();
                    chars.next();
                } else {
                    break;
                }
            }
            tokens.push(ScannedToken {
                range: start..end,
                kind: TokenKind::Number,
            });
            continue;
        }

        if c.is_alphabetic() || c == '\'' || c == '-' {
            let mut end = start + c.len_utf8();
            let mut has_alpha = c.is_alphabetic();
            chars.next();
            while let Some(&(idx, next)) = chars.peek() {
                if next.is_alphabetic() || next == '\'' || next == '-' {
                    has_alpha |= next.is_alphabetic();
                    end = idx + next.len_utf8();
                    chars.next();
                } else {
                    break;
                }
            }
            let kind = if has_alpha {
                TokenKind::Word
            } else {
                // A bare run of `'`/`-` with no letter at all: both
                // characters are members of `is_prose_punctuation`, so this
                // is exactly one punctuation token, not a word.
                TokenKind::Punctuation
            };
            tokens.push(ScannedToken {
                range: start..end,
                kind,
            });
            continue;
        }

        if is_prose_punctuation(c) {
            let mut end = start + c.len_utf8();
            chars.next();
            while let Some(&(idx, next)) = chars.peek() {
                if is_prose_punctuation(next) {
                    end = idx + next.len_utf8();
                    chars.next();
                } else {
                    break;
                }
            }
            tokens.push(ScannedToken {
                range: start..end,
                kind: TokenKind::Punctuation,
            });
            continue;
        }

        let mut end = start + c.len_utf8();
        chars.next();
        while let Some(&(idx, next)) = chars.peek() {
            if next.is_whitespace()
                || next.is_numeric()
                || next.is_alphabetic()
                || next == '\''
                || next == '-'
                || is_prose_punctuation(next)
            {
                break;
            }
            end = idx + next.len_utf8();
            chars.next();
        }
        tokens.push(ScannedToken {
            range: start..end,
            kind: TokenKind::Symbol,
        });
    }

    debug_assert!(
        tokens
            .iter()
            .all(|t| classify_token_kind(&text[t.range.clone()]) == t.kind),
        "tokenize's own classification must always agree with classify_token_kind, the \
         backend-agnostic contract every Tagger in this crate shares"
    );

    tokens
}

/// A [`Tagger`] backed by a from-scratch-trained averaged perceptron. See
/// the module docs for tokenization, non-word passthrough tags, and the
/// per-call sentence-reset semantics.
pub struct PerceptronTagger {
    weights: WeightTable,
    /// Sorted once at load; fixes both scoring-array indices and tie-break
    /// order (argmax scans this in order, so ties resolve to the
    /// lexicographically earliest entry).
    classes: Vec<Box<str>>,
    tagdict: BTreeMap<Box<str>, Box<str>>,
}

impl PerceptronTagger {
    /// Loads the tagger from its embedded, gzip-compressed weight
    /// artifact.
    ///
    /// # Errors
    /// [`PerceptronTagError::Decompress`] if the embedded bytes are not
    /// valid gzip; [`PerceptronTagError::Parse`] if the decompressed JSON
    /// does not match the expected shape. Neither should happen for the
    /// vendored artifact this crate ships — covered by this module's own
    /// tests.
    pub fn new() -> Result<Self, PerceptronTagError> {
        let mut decoder = flate2::read::GzDecoder::new(WEIGHTS_BYTES);
        let mut json = String::new();
        decoder
            .read_to_string(&mut json)
            .map_err(PerceptronTagError::Decompress)?;
        let file: WeightFile = serde_json::from_str(&json).map_err(PerceptronTagError::Parse)?;
        let tagdict = file.tagdict.clone();
        let (classes, weights) = WeightTable::from_file(file);
        Ok(Self {
            weights,
            classes,
            tagdict,
        })
    }

    /// Tags one [`TokenKind::Word`] token at `i`: a tagdict hit short-
    /// circuits the model entirely; otherwise scores every class and
    /// returns the argmax. `raw` is the token's original-case surface text,
    /// used only for its suffix/first-character features — see
    /// [`extract_features`].
    fn tag_word(
        &self,
        surfaces: &[Box<str>],
        raw: &str,
        i: usize,
        prev1: &str,
        prev2: &str,
    ) -> Box<str> {
        if let Some(tag) = self.tagdict.get(surfaces[i].as_ref()) {
            return tag.clone();
        }
        let features = extract_features(surfaces, raw, i, prev1, prev2);
        let scores = self.weights.score(&features);
        self.classes[argmax(&scores)].clone()
    }
}

impl Tagger for PerceptronTagger {
    fn tag(&self, text: &str, base_offset: usize) -> Vec<TaggedToken> {
        let scanned = tokenize(text);
        if scanned.is_empty() {
            return Vec::new();
        }

        let surfaces: Vec<Box<str>> = scanned
            .iter()
            .map(|t| Box::from(text[t.range.clone()].to_lowercase()))
            .collect();

        let mut prev1: Box<str> = Box::from(START_TAG_1);
        let mut prev2: Box<str> = Box::from(START_TAG_2);
        let mut out = Vec::with_capacity(scanned.len());

        for (i, scan_token) in scanned.iter().enumerate() {
            let surface = &text[scan_token.range.clone()];
            let assigned: Box<str> = match scan_token.kind {
                TokenKind::Number => Box::from("CD"),
                TokenKind::Punctuation => Box::from(surface),
                TokenKind::Word => self.tag_word(&surfaces, surface, i, &prev1, &prev2),
                // `tokenize` never emits `Whitespace`; any other/future
                // `TokenKind` (this enum is `#[non_exhaustive]`) falls back
                // here rather than panicking.
                _ => Box::from("SYM"),
            };

            let range =
                (scan_token.range.start + base_offset)..(scan_token.range.end + base_offset);
            let lemma = if scan_token.kind == TokenKind::Word {
                crate::inflect::lemmatize(surface, &assigned)
            } else {
                Box::from(surface.to_lowercase())
            };
            out.push(TaggedToken {
                token: Token::new(range, scan_token.kind),
                pos: PosTag::new(assigned.clone()),
                lemma,
            });

            prev2 = prev1;
            prev1 = assigned;
        }

        out
    }
}

/// Training-support helpers.
///
/// Feature extraction, the gold-file format, and weight-table save, shared
/// between inference (above) and `examples/train_perceptron.rs`. Exists
/// only behind the `train-tooling` feature so normal builds never carry
/// training code in their public surface.
#[cfg(feature = "train-tooling")]
pub mod train_support {
    use std::collections::BTreeMap;
    use std::io::Write as _;

    use super::{PerceptronTagError, WeightFile, extract_features, normalize_word_identity};

    /// One gold-tagged sentence: `(lowercased surface, gold tag)` pairs.
    ///
    /// In order. Non-word tokens (punctuation/number/symbol) carry the
    /// exact same deterministic passthrough tag [`super::Tagger::tag`]
    /// assigns them at inference, so a gold file is entirely self-consistent
    /// with runtime tag-history context.
    pub type GoldSentence = Vec<(String, String)>;

    /// Parses a hand-rolled tab-separated gold file.
    ///
    /// `word<TAB>tag` per line, a blank line marking a sentence break. No
    /// external CoNLL-U crate needed, and no external corpus dependency —
    /// the file is entirely this project's own hand-curated data.
    #[must_use]
    pub fn parse_gold_file(text: &str) -> Vec<GoldSentence> {
        let mut sentences = Vec::new();
        let mut current: GoldSentence = Vec::new();
        for line in text.lines() {
            let trimmed = line.trim_end();
            if trimmed.trim().is_empty() {
                if !current.is_empty() {
                    sentences.push(std::mem::take(&mut current));
                }
                continue;
            }
            let Some((word, tag)) = trimmed.split_once('\t') else {
                continue;
            };
            current.push((word.to_string(), tag.to_string()));
        }
        if !current.is_empty() {
            sentences.push(current);
        }
        sentences
    }

    /// Every token's lowercased surface text in `sentence` — the same
    /// context slice [`features_for`] needs, computed once per sentence.
    #[must_use]
    pub fn surfaces_of(sentence: &GoldSentence) -> Vec<Box<str>> {
        sentence
            .iter()
            .map(|(word, _)| Box::from(word.to_lowercase()))
            .collect()
    }

    /// Re-exposes [`extract_features`] for the training loop.
    ///
    /// The loop must call it once per token, interleaved with the model's
    /// own live prediction feeding `prev1_tag`/`prev2_tag` for the *next*
    /// token — this is exactly the perceptron-tagger convention this
    /// module's own inference path follows (the model's guess, not the
    /// gold tag, becomes tag-history context), so training and inference
    /// never disagree about what "previous tag" means. `raw_word` must be
    /// token `i`'s original-case gold-file text (not `surfaces[i]`, which
    /// is already lowercased) — see [`extract_features`]'s own doc comment
    /// for why the current token's suffix/first-character features need
    /// the un-normalized form to match training and inference.
    #[must_use]
    pub fn features_for(
        surfaces: &[Box<str>],
        raw_word: &str,
        i: usize,
        prev1_tag: &str,
        prev2_tag: &str,
    ) -> Vec<Box<str>> {
        extract_features(surfaces, raw_word, i, prev1_tag, prev2_tag)
    }

    /// A minimal from-scratch averaged perceptron.
    ///
    /// Follows the standard lazy-averaging trick (accumulate `weight *
    /// (iterations held)` and divide by the total iteration count at the
    /// end) rather than naively summing every intermediate weight.
    #[derive(Default)]
    pub struct AveragedPerceptron {
        weights: BTreeMap<Box<str>, BTreeMap<Box<str>, f64>>,
        totals: BTreeMap<(Box<str>, Box<str>), f64>,
        timestamps: BTreeMap<(Box<str>, Box<str>), u64>,
        iterations: u64,
        pub classes: Vec<Box<str>>,
    }

    impl AveragedPerceptron {
        #[must_use]
        pub fn new(classes: Vec<Box<str>>) -> Self {
            let mut classes = classes;
            classes.sort();
            classes.dedup();
            Self {
                classes,
                ..Self::default()
            }
        }

        /// Scores every class for `features`, returning `(predicted tag,
        /// per-class scores in `self.classes` order)`. Ties resolve to the
        /// lexicographically earliest class, exactly like the frozen
        /// [`super::WeightTable`] this trainer eventually produces.
        #[must_use]
        pub fn predict(&self, features: &[Box<str>]) -> String {
            let mut scores = vec![0f64; self.classes.len()];
            for feature in features {
                let Some(per_class) = self.weights.get(feature) else {
                    continue;
                };
                for (class, weight) in per_class {
                    if let Ok(index) = self.classes.binary_search(class) {
                        scores[index] += weight;
                    }
                }
            }
            let mut best = 0usize;
            let mut best_score = scores[0];
            for (index, &score) in scores.iter().enumerate().skip(1) {
                if score > best_score {
                    best_score = score;
                    best = index;
                }
            }
            self.classes[best].to_string()
        }

        fn bump(&mut self, feature: &str, class: &str, delta: f64) {
            let key: (Box<str>, Box<str>) = (Box::from(feature), Box::from(class));
            let current = self
                .weights
                .get(feature)
                .and_then(|m| m.get(class))
                .copied()
                .unwrap_or(0.0);
            let last_ts = self.timestamps.get(&key).copied().unwrap_or(0);
            // Training runs for a handful of epochs over a small gold
            // corpus, so `iterations` never comes close to 2^52 — this
            // conversion never loses meaningful precision.
            #[allow(clippy::cast_precision_loss)]
            let held = (self.iterations - last_ts) as f64;
            let total = self.totals.entry(key.clone()).or_insert(0.0);
            *total = held.mul_add(current, *total);
            self.timestamps.insert(key, self.iterations);
            self.weights
                .entry(Box::from(feature))
                .or_default()
                .insert(Box::from(class), current + delta);
        }

        /// Trains on one example: predicts, and if wrong, nudges the gold
        /// class's weight up and the guessed class's weight down for every
        /// feature. Returns the prediction (the caller feeds this, not the
        /// gold tag, into the next token's tag-history context).
        pub fn train_one(&mut self, features: &[Box<str>], gold: &str) -> String {
            self.iterations += 1;
            let guess = self.predict(features);
            if guess != gold {
                for feature in features {
                    self.bump(feature, gold, 1.0);
                    self.bump(feature, &guess, -1.0);
                }
            }
            guess
        }

        /// Averages every weight the standard way and emits the on-disk
        /// [`WeightFile`] shape, plus `tagdict` built from `word_tag_counts`
        /// (a word seen at least `min_count` times, with one tag covering
        /// at least `min_purity` of its occurrences, bypasses the model
        /// entirely).
        #[must_use]
        pub fn finish(
            mut self,
            word_tag_counts: &BTreeMap<String, BTreeMap<String, u32>>,
            min_count: u32,
            min_purity: f64,
        ) -> WeightFile {
            let mut features: BTreeMap<Box<str>, Vec<(u16, f32)>> = BTreeMap::new();
            let keys: Vec<Box<str>> = self.weights.keys().cloned().collect();
            for feature in keys {
                let per_class = self.weights.remove(&feature).unwrap_or_default();
                let mut sparse = Vec::new();
                for (class, weight) in per_class {
                    let key = (feature.clone(), class.clone());
                    let last_ts = self.timestamps.get(&key).copied().unwrap_or(0);
                    let total = self.totals.get(&key).copied().unwrap_or(0.0);
                    // See the identical justification in `bump` above.
                    #[allow(clippy::cast_precision_loss)]
                    let held = (self.iterations - last_ts) as f64;
                    let total = held.mul_add(weight, total);
                    #[allow(clippy::cast_precision_loss)]
                    let averaged = total / (self.iterations.max(1) as f64);
                    if averaged.abs() > f64::EPSILON {
                        let index = self
                            .classes
                            .binary_search(&class)
                            .expect("class must be registered");
                        #[allow(clippy::cast_possible_truncation)]
                        sparse.push((index as u16, averaged as f32));
                    }
                }
                if !sparse.is_empty() {
                    features.insert(feature, sparse);
                }
            }

            let mut tagdict = BTreeMap::new();
            for (word, tag_counts) in word_tag_counts {
                let total: u32 = tag_counts.values().sum();
                if total < min_count {
                    continue;
                }
                if let Some((best_tag, best_count)) =
                    tag_counts.iter().max_by_key(|(_, count)| **count)
                {
                    #[allow(clippy::cast_precision_loss)]
                    let purity = f64::from(*best_count) / f64::from(total);
                    if purity >= min_purity {
                        tagdict.insert(Box::from(word.as_str()), Box::from(best_tag.as_str()));
                    }
                }
            }

            WeightFile {
                classes: self.classes,
                features,
                tagdict,
            }
        }
    }

    /// Gzip-compresses `file` as JSON and writes it to `path`.
    ///
    /// # Errors
    /// Any I/O or serialization failure.
    pub fn save_weight_file(file: &WeightFile, path: &std::path::Path) -> std::io::Result<()> {
        let json = serde_json::to_string(file).expect("WeightFile always serializes");
        let out = std::fs::File::create(path)?;
        let mut encoder = flate2::write::GzEncoder::new(out, flate2::Compression::best());
        encoder.write_all(json.as_bytes())?;
        encoder.finish()?;
        Ok(())
    }

    /// Loads a gzip-compressed [`WeightFile`] from `path` — the same shape
    /// [`super::PerceptronTagger::new`] reads from its embedded bytes, used
    /// by `examples/train_perceptron.rs`'s own reproducibility check.
    ///
    /// # Errors
    /// [`PerceptronTagError::Decompress`] / [`PerceptronTagError::Parse`].
    pub fn load_weight_file(path: &std::path::Path) -> Result<WeightFile, PerceptronTagError> {
        let bytes = std::fs::read(path).map_err(PerceptronTagError::Decompress)?;
        let mut decoder = flate2::read::GzDecoder::new(bytes.as_slice());
        let mut json = String::new();
        std::io::Read::read_to_string(&mut decoder, &mut json)
            .map_err(PerceptronTagError::Decompress)?;
        serde_json::from_str(&json).map_err(PerceptronTagError::Parse)
    }

    /// Re-exposes [`normalize_word_identity`] for a training tool's own
    /// diagnostics (e.g. reporting how many gold tokens fell into the
    /// hyphen bucket) without duplicating the bucketing rule.
    #[must_use]
    pub fn normalize_for_diagnostics(word_lower: &str) -> &str {
        normalize_word_identity(word_lower)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::OnceLock;

    use friction_core::TokenKind;

    use super::*;

    fn tagger() -> &'static PerceptronTagger {
        static TAGGER: OnceLock<PerceptronTagger> = OnceLock::new();
        TAGGER.get_or_init(|| PerceptronTagger::new().expect("embedded weights must load"))
    }

    #[test]
    fn tag_produces_spans_pos_and_lemmas_for_a_plain_sentence() {
        let text = "The quick brown foxes are jumping over lazy dogs.";
        let base_offset = 100;
        let tagged = tagger().tag(text, base_offset);
        assert!(!tagged.is_empty());
        for t in &tagged {
            assert!(t.token.range.start >= base_offset);
            assert!(t.token.range.end <= base_offset + text.len());
            assert!(!t.pos.as_str().is_empty());
        }
        let foxes = tagged
            .iter()
            .find(|t| {
                &text[t.token.range.start - base_offset..t.token.range.end - base_offset] == "foxes"
            })
            .expect("foxes token present");
        // The lemma is either a successfully round-tripped base form
        // ("fox", if tagged NNS/VBZ) or the documented surface-text
        // fallback ("foxes") -- never anything else, and never empty.
        assert!(
            matches!(&*foxes.lemma, "fox" | "foxes"),
            "{:?}",
            foxes.lemma
        );
    }

    #[test]
    fn tag_gives_punctuation_its_own_literal_text_as_its_tag() {
        let tagged = tagger().tag("Wait, really?", 0);
        let comma = tagged
            .iter()
            .find(|t| &"Wait, really?"[t.token.range.clone()] == ",")
            .expect("comma token present");
        assert_eq!(comma.token.kind, TokenKind::Punctuation);
        assert_eq!(comma.pos.as_str(), ",");
    }

    #[test]
    fn tag_gives_numbers_the_cd_tag() {
        let tagged = tagger().tag("There are 42 widgets.", 0);
        let number = tagged
            .iter()
            .find(|t| t.token.kind == TokenKind::Number)
            .expect("a number token is present");
        assert_eq!(number.pos.as_str(), "CD");
    }

    #[test]
    fn tag_gives_symbols_the_sym_tag() {
        let tagged = tagger().tag("Use A & B together.", 0);
        let symbol = tagged
            .iter()
            .find(|t| t.token.kind == TokenKind::Symbol)
            .expect("a symbol token is present");
        assert_eq!(symbol.pos.as_str(), "SYM");
    }

    #[test]
    fn tag_never_panics_and_never_emits_unknown_for_an_out_of_vocabulary_word() {
        let tagged = tagger().tag("Zxqvplarnfrobnicate is a word.", 0);
        assert!(!tagged.is_empty());
        let oov = &tagged[0];
        assert_eq!(&*oov.lemma, "zxqvplarnfrobnicate");
        assert_ne!(oov.pos.as_str(), "UNKNOWN");
    }

    #[test]
    fn tag_accepts_empty_text() {
        assert!(tagger().tag("", 0).is_empty());
    }

    #[test]
    fn tagging_is_deterministic_across_repeated_runs_and_fresh_instances() {
        let paragraph = "The team didn't leverage the framework's full potential. \
                          Zxqvplarnfrobnicate elements were, however, refactored twice. \
                          Meanwhile, 42 widgets shipped on time, and the client was pleased!";

        let first = tagger().tag(paragraph, 0);
        let second = tagger().tag(paragraph, 0);
        assert_eq!(first, second);

        let fresh = PerceptronTagger::new().expect("embedded weights must load");
        let third = fresh.tag(paragraph, 0);
        assert_eq!(first, third);
    }

    /// A constructed near-tie input (an out-of-vocabulary word with no
    /// informative feature weights at all) still resolves to the exact
    /// same tag on every run — the tie-break is deterministic by
    /// construction (lowest sorted-class index wins), not by luck.
    #[test]
    fn tag_tie_break_is_stable_across_repeated_runs() {
        let text = "Zzzznope.";
        let first = tagger().tag(text, 0);
        for _ in 0..5 {
            assert_eq!(tagger().tag(text, 0), first);
        }
    }

    // ---------------------------------------------------------------
    // extract_features: current-token case handling
    // ---------------------------------------------------------------

    /// The current token's suffix/first-character features must read its
    /// *original* case (matching real NLTK's own `word[-3:]`/`word[0]`,
    /// computed from the raw token, not the normalized `context` list) —
    /// while its identity feature (`word=`) stays lowercased/normalized
    /// either way, since that one deliberately generalizes across case.
    #[test]
    fn extract_features_suffix_and_pref1_use_the_current_tokens_original_case() {
        let surfaces: Vec<Box<str>> = vec![Box::from("apple")];
        let lower = extract_features(&surfaces, "apple", 0, START_TAG_1, START_TAG_2);
        let upper = extract_features(&surfaces, "Apple", 0, START_TAG_1, START_TAG_2);

        let pref1_lower = lower
            .iter()
            .find(|f| f.starts_with("pref1="))
            .expect("pref1 feature present");
        let pref1_upper = upper
            .iter()
            .find(|f| f.starts_with("pref1="))
            .expect("pref1 feature present");
        assert_eq!(&**pref1_lower, "pref1=a");
        assert_eq!(&**pref1_upper, "pref1=A");

        let word_lower = lower.iter().find(|f| f.starts_with("word="));
        let word_upper = upper.iter().find(|f| f.starts_with("word="));
        assert_eq!(
            word_lower, word_upper,
            "the identity feature must stay case-insensitive even though pref1/suffix now see \
             raw case"
        );
    }

    // ---------------------------------------------------------------
    // tokenize (via classify_token_kind agreement)
    // ---------------------------------------------------------------

    #[test]
    fn tokenize_agrees_with_classify_token_kind_on_word_runs() {
        let tokens = tokenize("state-of-the-art don't");
        let kinds: Vec<TokenKind> = tokens.iter().map(|t| t.kind).collect();
        assert_eq!(kinds, vec![TokenKind::Word, TokenKind::Word]);
    }

    #[test]
    fn tokenize_groups_runs_of_dots_and_dashes_as_one_punctuation_token() {
        let text = "wait... okay -- fine";
        let tokens = tokenize(text);
        let texts: Vec<&str> = tokens.iter().map(|t| &text[t.range.clone()]).collect();
        assert_eq!(texts, vec!["wait", "...", "okay", "--", "fine"]);
    }

    #[test]
    fn tokenize_recognizes_numbers() {
        let text = "3.14 and 1,000";
        let tokens = tokenize(text);
        assert_eq!(tokens[0].kind, TokenKind::Number);
        assert_eq!(&text[tokens[0].range.clone()], "3.14");
    }

    #[test]
    fn tokenize_emits_no_whitespace_tokens() {
        let tokens = tokenize("  a   b  ");
        assert!(tokens.iter().all(|t| t.kind != TokenKind::Whitespace));
        assert_eq!(tokens.len(), 2);
    }
}
