//! Rhythm and shape metrics: sentence-length burstiness, paragraph shape,
//! and dash/semicolon punctuation density.
//!
//! # Token definition
//!
//! Every function in this module that counts tokens — sentence length in
//! tokens, and the "per 1000 tokens" densities — uses the same definition:
//! a token is a maximal run of non-whitespace characters, exactly what
//! [`str::split_whitespace`] yields. This is deliberately independent of
//! any tagger or dependency parser: it is a pure function of a sentence's
//! own source text, so these metrics never depend on whether tokenization
//! or POS tagging has been run over the document.
//!
//! # Determinism
//!
//! Every function here is a pure function of its [`Document`] argument.
//! Observations (sentence lengths, sentence-per-paragraph counts, dash/
//! semicolon occurrences) are always gathered by walking `document.prose()`
//! and each unit's `sentences` in order — never through a `HashMap` or any
//! other unordered collection — and floating-point sums fold left to
//! right over that same order, so identical input always produces
//! bit-identical output on any platform.
//!
//! # Degenerate cases
//!
//! No function in this module ever returns `NaN` or `inf`. See
//! [`RhythmStats`]'s docs for the mean/stddev/CV conventions on empty and
//! single-observation inputs, and [`em_dash_density`]/[`semicolon_density`]
//! for the empty-document (zero-token) convention.

use friction_core::{Document, Sentence};

/// Mean, population standard deviation, and coefficient of variation
/// (`stddev / mean`) over a list of numeric observations, plus the
/// observation count they were computed from.
///
/// # Degenerate cases
///
/// - **Zero observations** (`n == 0`, e.g. a document with no sentences):
///   `mean`, `stddev`, and `cv` are all `0.0`. There is no shape to report
///   for an empty input, and `0.0` keeps every value a plain, usable
///   number instead of a `NaN` that would poison anything computed from it
///   downstream (a [`friction_core::MetricVector`] field, an envelope
///   comparison, ...).
/// - **A single observation** (`n == 1`): `stddev` is exactly `0.0` (one
///   point has no spread to measure), so `cv` is `0.0` too — this falls
///   out of the same formula used for `n > 1`, it is not a special case.
/// - **`mean` is exactly `0.0`** (every observation is `0.0`): `cv` is
///   defined as `0.0` rather than `stddev / 0.0`, which would otherwise be
///   `NaN` (when `stddev` is also `0.0`) or `inf`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RhythmStats {
    /// Arithmetic mean of the observations.
    pub mean: f64,
    /// Population standard deviation: divides the sum of squared
    /// deviations by `n`, not `n - 1`. These metrics describe the shape of
    /// the document or paragraph actually observed, not an estimate of
    /// some larger population it was sampled from, so the population
    /// (not sample) formula is the correct one.
    pub stddev: f64,
    /// Coefficient of variation, `stddev / mean`. See "Degenerate cases"
    /// on this type for when `mean` is `0.0`.
    pub cv: f64,
    /// Number of observations `mean`, `stddev`, and `cv` were computed
    /// from.
    pub n: usize,
}

impl RhythmStats {
    /// Computes mean, population standard deviation, and coefficient of
    /// variation over `values`, summing left to right in the order given
    /// (index `0` first) so the result is bit-for-bit reproducible
    /// regardless of platform or thread count.
    fn from_observations(values: &[f64]) -> Self {
        let n = values.len();
        if n == 0 {
            return Self {
                mean: 0.0,
                stddev: 0.0,
                cv: 0.0,
                n: 0,
            };
        }
        #[allow(clippy::cast_precision_loss)]
        let n_f64 = n as f64;
        let mean = values.iter().sum::<f64>() / n_f64;
        let variance = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n_f64;
        let stddev = variance.sqrt();
        let cv = if mean == 0.0 { 0.0 } else { stddev / mean };
        Self {
            mean,
            stddev,
            cv,
            n,
        }
    }
}

/// Counts the tokens in `text`: maximal runs of non-whitespace characters
/// (see the module docs for why this definition, rather than a tagger's,
/// is used here).
fn token_count(text: &str) -> usize {
    text.split_whitespace().count()
}

/// The source text of `sentence`, sliced from `document`.
///
/// # Panics
/// Never, for any `sentence` that actually belongs to `document`: every
/// sentence range reachable from `document.prose()` was already validated
/// (in-bounds, on a UTF-8 boundary, contained in its parent) by
/// [`Document::new`] at construction time, so [`Document::text`] cannot
/// fail for it.
fn sentence_text<'doc>(document: &'doc Document, sentence: &Sentence) -> &'doc str {
    document
        .text(&sentence.range)
        .expect("sentence ranges are validated by Document::new at construction")
}

/// Every sentence length in `document`, in tokens, in source order
/// (walking each prose unit, then each of its sentences, in order).
fn document_sentence_lengths(document: &Document) -> Vec<f64> {
    document
        .prose()
        .iter()
        .flat_map(|unit| &unit.sentences)
        .map(|sentence| {
            #[allow(clippy::cast_precision_loss)]
            let len = token_count(sentence_text(document, sentence)) as f64;
            len
        })
        .collect()
}

/// Sentence length (in tokens) [`RhythmStats`] over every sentence in
/// `document`, document-wide.
///
/// # Example (hand-computed)
///
/// Three sentences with lengths 3, 10, and 3 tokens:
/// `mean = 16/3 ≈ 5.333333`; squared deviations `(3 − 16/3)² = 49/9`,
/// `(10 − 16/3)² = 196/9`, `(3 − 16/3)² = 49/9`, summing to `294/9`;
/// `variance = (294/9) / 3 = 294/27 ≈ 10.888889`; `stddev = √variance ≈
/// 3.299832`; `cv = stddev / mean ≈ 0.618718`. See the unit test of the
/// same name for the executable version of this derivation.
#[must_use]
pub fn sentence_length_by_document(document: &Document) -> RhythmStats {
    RhythmStats::from_observations(&document_sentence_lengths(document))
}

/// Sentence length (in tokens) [`RhythmStats`] computed separately for
/// each paragraph in `document`, in source order.
///
/// A paragraph is one [`friction_core::ProseUnit`] — a maximal contiguous
/// run of prose text, as produced by `friction-parse`. Paragraphs with no
/// sentences (e.g. a heading whose text segmented to zero complete
/// sentences) contribute no entry to the returned `Vec`: there is no
/// sentence-length shape to report for them, and inventing an all-zero
/// entry would misrepresent an actual zero-sentence paragraph as if it
/// were a paragraph with one zero-token sentence.
#[must_use]
pub fn sentence_length_by_paragraph(document: &Document) -> Vec<RhythmStats> {
    document
        .prose()
        .iter()
        .filter(|unit| !unit.sentences.is_empty())
        .map(|unit| {
            let lengths: Vec<f64> = unit
                .sentences
                .iter()
                .map(|sentence| {
                    #[allow(clippy::cast_precision_loss)]
                    let len = token_count(sentence_text(document, sentence)) as f64;
                    len
                })
                .collect();
            RhythmStats::from_observations(&lengths)
        })
        .collect()
}

/// Paragraph-shape [`RhythmStats`]: mean and coefficient of variation of
/// sentences-per-paragraph.
///
/// Computed over every paragraph in `document` that has at least one
/// sentence (see [`sentence_length_by_paragraph`] for why zero-sentence
/// paragraphs are excluded), in source order.
///
/// `stddev` is reported alongside `mean`/`cv` like every [`RhythmStats`],
/// but the metric vector only surfaces `mean` and `cv` for paragraph
/// shape.
#[must_use]
pub fn paragraph_shape(document: &Document) -> RhythmStats {
    let counts: Vec<f64> = document
        .prose()
        .iter()
        .filter_map(|unit| {
            if unit.sentences.is_empty() {
                None
            } else {
                #[allow(clippy::cast_precision_loss)]
                let n = unit.sentences.len() as f64;
                Some(n)
            }
        })
        .collect();
    RhythmStats::from_observations(&counts)
}

/// Em-dash density: occurrences per 1000 tokens, over every sentence in
/// `document`, document-wide.
///
/// Two surface forms each count as one em-dash occurrence:
/// - the literal em dash character (`—`, U+2014);
/// - a double-hyphen surrogate (`--`, exactly two ASCII hyphen-minus
///   characters — a run of one or of three-or-more does not count) that
///   sits between two words: the nearest non-space character on each side
///   (looking one character past a single flanking space, if there is
///   one) must be alphanumeric. This covers both the spaced
///   (`"word -- word"`) and unspaced (`"word--word"`) typewriter
///   conventions for an em dash, while excluding a `--` at the very start
///   or end of a sentence (no word on that side) and a `---` thematic-
///   break-style run.
///
/// `tokens` is the same whitespace-token count [`sentence_length_by_document`]
/// uses. An empty document (zero tokens) has density `0.0`, not `NaN`.
///
/// # Example (hand-computed)
///
/// `"Speed matters — quality matters too."` (6 tokens, one literal em
/// dash) followed by `"It works fine--somehow; trust me."` (5 tokens, one
/// double-hyphen surrogate between "fine" and "somehow", one semicolon):
/// 11 tokens total, 2 em-dash occurrences, density `= 2 × 1000 / 11 ≈
/// 181.818182`. See the unit test of the same name for the executable
/// version, including [`semicolon_density`]'s `1 × 1000 / 11 ≈ 90.909091`
/// on the same fixture.
#[must_use]
pub fn em_dash_density(document: &Document) -> f64 {
    density_per_1000_tokens(document, count_em_dashes)
}

/// Semicolon density: `;` occurrences per 1000 tokens, over every sentence
/// in `document`, document-wide. See [`em_dash_density`] for the token
/// definition and the empty-document (zero-token) convention.
#[must_use]
pub fn semicolon_density(document: &Document) -> f64 {
    density_per_1000_tokens(document, count_semicolons)
}

/// Shared walk for the two "occurrences per 1000 tokens" densities:
/// accumulates both `count_in`'s occurrence count and the token count over
/// every sentence in `document`, in source order, then divides once at the
/// end.
fn density_per_1000_tokens(document: &Document, count_in: impl Fn(&str) -> usize) -> f64 {
    let mut occurrences: usize = 0;
    let mut tokens: usize = 0;
    for unit in document.prose() {
        for sentence in &unit.sentences {
            let text = sentence_text(document, sentence);
            occurrences += count_in(text);
            tokens += token_count(text);
        }
    }
    if tokens == 0 {
        return 0.0;
    }
    #[allow(clippy::cast_precision_loss)]
    let density = occurrences as f64 * 1000.0 / tokens as f64;
    density
}

/// Counts `;` characters in `text`.
fn count_semicolons(text: &str) -> usize {
    text.chars().filter(|&c| c == ';').count()
}

/// Counts em-dash occurrences in `text`: see [`em_dash_density`] for the
/// exact rule.
fn count_em_dashes(text: &str) -> usize {
    let chars: Vec<char> = text.chars().collect();
    let mut count = 0;
    let mut i = 0;
    while i < chars.len() {
        match chars[i] {
            '\u{2014}' => {
                count += 1;
                i += 1;
            }
            '-' => {
                let start = i;
                let mut end = i;
                while end < chars.len() && chars[end] == '-' {
                    end += 1;
                }
                if end - start == 2 && is_word_flanked(&chars, start, end) {
                    count += 1;
                }
                i = end;
            }
            _ => i += 1,
        }
    }
    count
}

/// Whether the hyphen run `chars[start..end]` has an alphanumeric
/// character immediately outside it on both sides (skipping at most one
/// flanking space per side).
fn is_word_flanked(chars: &[char], start: usize, end: usize) -> bool {
    flank_before(chars, start).is_some_and(char::is_alphanumeric)
        && flank_after(chars, end).is_some_and(char::is_alphanumeric)
}

/// The character just before `chars[start]`, skipping one space if that is
/// what immediately precedes it. `None` if `start` is the beginning of
/// `chars`, or a skipped space is itself the beginning.
fn flank_before(chars: &[char], start: usize) -> Option<char> {
    if start == 0 {
        return None;
    }
    let c = chars[start - 1];
    if c == ' ' {
        start.checked_sub(2).map(|i| chars[i])
    } else {
        Some(c)
    }
}

/// The character just after `chars[end - 1]`, skipping one space if that
/// is what immediately follows it. `None` if `end` is the end of `chars`,
/// or a skipped space is itself the end.
fn flank_after(chars: &[char], end: usize) -> Option<char> {
    let c = *chars.get(end)?;
    if c == ' ' {
        chars.get(end + 1).copied()
    } else {
        Some(c)
    }
}

#[cfg(test)]
mod tests {
    use friction_core::{Block, BlockKind, Document, ProseUnit, Sentence};

    use super::{
        RhythmStats, em_dash_density, paragraph_shape, semicolon_density,
        sentence_length_by_document, sentence_length_by_paragraph,
    };

    /// Builds a one-block, one-paragraph document out of `sentences`
    /// (already-joined source text) and the byte range of each sentence
    /// within it, entirely through `friction-core`'s own constructors —
    /// independent of `friction-parse`/`friction-nlp`, so these fixtures
    /// exercise only this module's arithmetic, not any other agent's
    /// still-moving segmentation behavior.
    fn doc_single_paragraph(source: &str, sentence_ranges: &[std::ops::Range<usize>]) -> Document {
        let block = Block::new(BlockKind::Paragraph, 0..source.len());
        let sentences = sentence_ranges
            .iter()
            .cloned()
            .map(|range| Sentence::new(range, Vec::new()))
            .collect();
        let prose = ProseUnit::new(0, 0..source.len(), sentences);
        Document::new(source, vec![block], vec![prose]).expect("fixture must be well-formed")
    }

    /// A document with `paragraphs`, each a `(text, sentence_ranges)` pair
    /// already positioned at its own byte offset within `source`; one
    /// `Block`/`ProseUnit` per paragraph.
    fn doc_multi_paragraph(
        source: &str,
        paragraphs: &[(std::ops::Range<usize>, &[std::ops::Range<usize>])],
    ) -> Document {
        let blocks: Vec<Block> = paragraphs
            .iter()
            .map(|(range, _)| Block::new(BlockKind::Paragraph, range.clone()))
            .collect();
        let prose: Vec<ProseUnit> = paragraphs
            .iter()
            .enumerate()
            .map(|(i, (range, sentence_ranges))| {
                let sentences = sentence_ranges
                    .iter()
                    .cloned()
                    .map(|r| Sentence::new(r, Vec::new()))
                    .collect();
                ProseUnit::new(i, range.clone(), sentences)
            })
            .collect();
        Document::new(source, blocks, prose).expect("fixture must be well-formed")
    }

    const EPSILON: f64 = 1e-6;

    /// Hand-computed: three sentences of length 3, 10, and 3 tokens.
    ///
    /// `mean = 16/3`; squared deviations `49/9, 196/9, 49/9` sum to
    /// `294/9`; `variance = (294/9) / 3 = 294/27`; `stddev = sqrt(294/27)`;
    /// `cv = stddev / mean`. Worked out on paper in the module docs of
    /// [`sentence_length_by_document`].
    #[test]
    fn sentence_length_by_document_matches_hand_computation() {
        let s1 = "The cat sat.";
        let s2 = "It was a very old cat, and it liked naps.";
        let s3 = "Naps are great.";
        let source = format!("{s1} {s2} {s3}");
        let s1_range = 0..s1.len();
        let s2_start = s1_range.end + 1;
        let s2_range = s2_start..s2_start + s2.len();
        let s3_start = s2_range.end + 1;
        let s3_range = s3_start..s3_start + s3.len();

        let doc = doc_single_paragraph(&source, &[s1_range, s2_range, s3_range]);
        let stats = sentence_length_by_document(&doc);

        let expected_mean: f64 = 16.0 / 3.0;
        let expected_variance: f64 = 294.0 / 27.0;
        let expected_stddev = expected_variance.sqrt();
        let expected_cv = expected_stddev / expected_mean;

        assert_eq!(stats.n, 3);
        assert!((stats.mean - expected_mean).abs() < EPSILON);
        assert!((stats.stddev - expected_stddev).abs() < EPSILON);
        assert!((stats.cv - expected_cv).abs() < EPSILON);
    }

    /// Hand-computed: two paragraphs. The first has sentences of length 3
    /// and 10 tokens (`mean = 6.5`, deviations `±3.5`, `variance =
    /// (12.25 + 12.25) / 2 = 12.25`, `stddev = 3.5` exactly, `cv =
    /// 3.5 / 6.5 = 7/13`); the second has a single sentence of length 3
    /// (`stddev = 0`, `cv = 0`, the single-observation convention).
    // The single-observation convention is an exact-zero guarantee (see
    // `RhythmStats`' docs), not an approximation, so exact equality is the
    // right assertion here.
    #[allow(clippy::float_cmp)]
    #[test]
    fn sentence_length_by_paragraph_matches_hand_computation() {
        let s1 = "The cat sat.";
        let s2 = "It was a very old cat, and it liked naps.";
        let s3 = "Naps are great.";
        let para1 = format!("{s1} {s2}");
        let para2 = s3.to_string();
        let source = format!("{para1}\n\n{para2}");

        let s1_range = 0..s1.len();
        let s2_start = s1_range.end + 1;
        let s2_range = s2_start..s2_start + s2.len();
        let para1_range = 0..para1.len();

        let para2_start = para1.len() + 2;
        let s3_range = para2_start..para2_start + s3.len();
        let para2_range = para2_start..para2_start + para2.len();

        let doc = doc_multi_paragraph(
            &source,
            &[
                (para1_range, &[s1_range, s2_range]),
                (para2_range, &[s3_range]),
            ],
        );
        let stats = sentence_length_by_paragraph(&doc);

        assert_eq!(stats.len(), 2);

        assert_eq!(stats[0].n, 2);
        assert!((stats[0].mean - 6.5).abs() < EPSILON);
        assert!((stats[0].stddev - 3.5).abs() < EPSILON);
        assert!((stats[0].cv - 7.0 / 13.0).abs() < EPSILON);

        assert_eq!(stats[1].n, 1);
        assert!((stats[1].mean - 3.0).abs() < EPSILON);
        assert_eq!(stats[1].stddev, 0.0);
        assert_eq!(stats[1].cv, 0.0);
    }

    /// Hand-computed: three paragraphs with 2, 4, and 3 sentences.
    /// `mean = 9/3 = 3`; deviations `-1, 1, 0`; squared `1, 1, 0` sum to
    /// `2`; `variance = 2/3`; `stddev = sqrt(2/3)`; `cv = stddev / 3`.
    #[test]
    fn paragraph_shape_matches_hand_computation() {
        // Sentence text content doesn't matter for this metric; only the
        // per-paragraph sentence *count* does. Each sentence is the
        // trivial one-token text "a." at its own two-byte slot, so
        // sentence `i` sits at byte range `3*i .. 3*i + 2` (a space
        // separates consecutive slots): paragraph 1 gets sentences 0-1 (2
        // sentences), paragraph 2 gets sentences 2-5 (4 sentences),
        // paragraph 3 gets sentences 6-8 (3 sentences).
        let source = "a. a. a. a. a. a. a. a. a.";
        let sentence = |i: usize| (3 * i)..(3 * i + 2);
        let sentences: Vec<std::ops::Range<usize>> = (0..9).map(sentence).collect();

        let para1_sentences = &sentences[0..2];
        let para2_sentences = &sentences[2..6];
        let para3_sentences = &sentences[6..9];
        let para1_range = para1_sentences[0].start..para1_sentences[1].end;
        let para2_range = para2_sentences[0].start..para2_sentences[3].end;
        let para3_range = para3_sentences[0].start..para3_sentences[2].end;

        let doc = doc_multi_paragraph(
            source,
            &[
                (para1_range, para1_sentences),
                (para2_range, para2_sentences),
                (para3_range, para3_sentences),
            ],
        );
        let stats = paragraph_shape(&doc);

        let expected_variance: f64 = 2.0 / 3.0;
        let expected_stddev = expected_variance.sqrt();
        let expected_cv = expected_stddev / 3.0;

        assert_eq!(stats.n, 3);
        assert!((stats.mean - 3.0).abs() < EPSILON);
        assert!((stats.stddev - expected_stddev).abs() < EPSILON);
        assert!((stats.cv - expected_cv).abs() < EPSILON);
    }

    /// Hand-computed: `"Speed matters — quality matters too."` (6 tokens,
    /// one literal em dash) plus `"It works fine--somehow; trust me."` (5
    /// tokens, one double-hyphen surrogate between "fine" and "somehow",
    /// one semicolon). 11 tokens total: em-dash density `= 2*1000/11`,
    /// semicolon density `= 1*1000/11`.
    #[test]
    fn dash_and_semicolon_density_match_hand_computation() {
        let s1 = "Speed matters — quality matters too.";
        let s2 = "It works fine--somehow; trust me.";
        let source = format!("{s1} {s2}");
        let s1_range = 0..s1.len();
        let s2_start = s1_range.end + 1;
        let s2_range = s2_start..s2_start + s2.len();

        let doc = doc_single_paragraph(&source, &[s1_range, s2_range]);

        let expected_em_dash = 2.0 * 1000.0 / 11.0;
        let expected_semicolon = 1.0 * 1000.0 / 11.0;

        assert!((em_dash_density(&doc) - expected_em_dash).abs() < EPSILON);
        assert!((semicolon_density(&doc) - expected_semicolon).abs() < EPSILON);
    }

    /// A leading `--` (no word before it) is not a surrogate em dash; a
    /// triple hyphen `---` is not either (wrong run length); an unspaced
    /// `word--word` is.
    #[test]
    fn em_dash_surrogate_requires_a_two_hyphen_run_between_words() {
        let s1 = "--not a dash here.";
        let s2 = "A triple---hyphen run does not count.";
        let s3 = "But word--word does.";
        let source = format!("{s1} {s2} {s3}");
        let s1_range = 0..s1.len();
        let s2_start = s1_range.end + 1;
        let s2_range = s2_start..s2_start + s2.len();
        let s3_start = s2_range.end + 1;
        let s3_range = s3_start..s3_start + s3.len();

        let doc = doc_single_paragraph(&source, &[s1_range, s2_range, s3_range]);

        // Exactly one surrogate occurrence ("word--word"), across however
        // many tokens the three sentences contain; check the numerator
        // directly by reconstructing density's own token divisor.
        let expected_tokens: usize = [s1, s2, s3]
            .iter()
            .map(|s| s.split_whitespace().count())
            .sum();
        #[allow(clippy::cast_precision_loss)]
        let expected = 1.0 * 1000.0 / expected_tokens as f64;
        assert!((em_dash_density(&doc) - expected).abs() < EPSILON);
    }

    /// An empty document (no prose at all) yields the all-zero degenerate
    /// `RhythmStats` and zero densities, never `NaN`.
    // The empty-input convention is an exact-zero guarantee, not an
    // approximation.
    #[allow(clippy::float_cmp)]
    #[test]
    fn empty_document_is_degenerate_not_nan() {
        let doc = Document::new("", Vec::new(), Vec::new()).expect("empty document is valid");

        let doc_stats = sentence_length_by_document(&doc);
        assert_eq!(
            doc_stats,
            RhythmStats {
                mean: 0.0,
                stddev: 0.0,
                cv: 0.0,
                n: 0,
            }
        );
        assert!(sentence_length_by_paragraph(&doc).is_empty());

        let shape_stats = paragraph_shape(&doc);
        assert_eq!(shape_stats.n, 0);
        assert_eq!(shape_stats.mean, 0.0);
        assert_eq!(shape_stats.cv, 0.0);

        assert_eq!(em_dash_density(&doc), 0.0);
        assert_eq!(semicolon_density(&doc), 0.0);
    }

    /// A document with exactly one sentence has `stddev == 0.0` and
    /// `cv == 0.0` by the general formula, not a special case.
    // The single-observation convention is an exact-zero guarantee, not an
    // approximation.
    #[allow(clippy::float_cmp)]
    #[test]
    fn single_sentence_document_has_zero_stddev_and_cv() {
        let s1 = "Only one sentence lives here.";
        #[allow(clippy::single_range_in_vec_init)]
        let doc = doc_single_paragraph(s1, &[0..s1.len()]);
        let stats = sentence_length_by_document(&doc);
        assert_eq!(stats.n, 1);
        assert!((stats.mean - 5.0).abs() < EPSILON);
        assert_eq!(stats.stddev, 0.0);
        assert_eq!(stats.cv, 0.0);
    }
}
