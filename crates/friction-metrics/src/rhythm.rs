//! Rhythm and shape metrics: sentence-length burstiness, paragraph shape,
//! and dash/semicolon punctuation density.
//!
//! # Token definition
//!
//! Every token count here uses the same definition — a maximal run of
//! non-whitespace characters, what [`str::split_whitespace`] yields,
//! independent of any tagger or parser (a pure function of source text).
//!
//! # Determinism
//!
//! Every function is a pure function of its [`Document`] argument:
//! observations are walked from `document.prose()`/`sentences` in order,
//! never via a `HashMap` or other unordered collection, and
//! floating-point sums fold left to right, so identical input always
//! produces bit-identical output.
//!
//! # Degenerate cases
//!
//! No function here returns `NaN` or `inf`. See [`RhythmStats`] for
//! empty/single-observation conventions, and
//! [`em_dash_density`]/[`semicolon_density`] for the empty-document
//! convention.

use friction_core::{Document, Sentence};

/// Mean, population standard deviation, and coefficient of variation
/// (`stddev / mean`) over a list of observations, plus the count they
/// were computed from.
///
/// # Degenerate cases
///
/// - **`n == 0`**: `mean`/`stddev`/`cv` are `0.0` — avoids `NaN`
///   poisoning downstream computations.
/// - **`n == 1`**: `stddev` is `0.0` (no spread), so `cv` is too — same
///   formula as `n > 1`, not a special case.
/// - **`mean == 0.0`**: `cv` is `0.0`, not `stddev / 0.0` (`NaN`/`inf`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RhythmStats {
    /// Arithmetic mean of the observations.
    pub mean: f64,
    /// Population standard deviation: divides by `n`, not `n - 1` — this
    /// is the document/paragraph actually observed, not a sample.
    pub stddev: f64,
    /// Coefficient of variation, `stddev / mean`. See "Degenerate cases"
    /// on this type for when `mean` is `0.0`.
    pub cv: f64,
    /// Observation count `mean`, `stddev`, and `cv` were computed from.
    pub n: usize,
}

impl RhythmStats {
    /// Computes mean, population standard deviation, and coefficient of
    /// variation over `values`, summing left to right (index `0` first)
    /// for bit-for-bit reproducibility.
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

/// Counts tokens in `text`: maximal runs of non-whitespace characters (see
/// module docs for why, not a tagger's definition).
fn token_count(text: &str) -> usize {
    text.split_whitespace().count()
}

/// The source text of `sentence`, sliced from `document`.
///
/// # Panics
/// Never, for any `sentence` belonging to `document`: [`Document::new`]
/// validates every sentence range (in-bounds, UTF-8 boundary, contained
/// in its parent) at construction, so [`Document::text`] cannot fail.
fn sentence_text<'doc>(document: &'doc Document, sentence: &Sentence) -> &'doc str {
    document
        .text(&sentence.range)
        .expect("sentence ranges are validated by Document::new at construction")
}

/// Every sentence length in `document`, in tokens, source order (each
/// prose unit, then its sentences).
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
/// `document`.
///
/// # Example (hand-computed)
///
/// Lengths 3, 10, 3: `mean = 16/3 ≈ 5.333333`, `variance = 294/27 ≈
/// 10.888889`, `stddev ≈ 3.299832`, `cv ≈ 0.618718`. Unit test of the
/// same name: executable version.
#[must_use]
pub fn sentence_length_by_document(document: &Document) -> RhythmStats {
    RhythmStats::from_observations(&document_sentence_lengths(document))
}

/// Sentence length (in tokens) [`RhythmStats`] computed separately for
/// each paragraph in `document`, in source order.
///
/// A paragraph is one [`friction_core::ProseUnit`]. Paragraphs with no
/// sentences (e.g. a heading) get no entry: an all-zero entry would
/// misrepresent a zero-sentence paragraph as one with a single
/// zero-token sentence.
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

/// Paragraph-shape [`RhythmStats`] (mean/cv of sentences-per-paragraph)
/// over paragraphs in `document` with at least one sentence (see
/// [`sentence_length_by_paragraph`] for why), in source order.
///
/// `stddev` is reported like any [`RhythmStats`], but the metric vector
/// only surfaces `mean` and `cv` here.
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
/// `document`.
///
/// Counts as one occurrence: the literal em dash (`—`, U+2014), or a
/// double-hyphen surrogate (`--`, exactly two ASCII hyphens, no more or
/// fewer) between two words: the nearest non-space character on each
/// side (skipping at most one flanking space) must be alphanumeric.
/// Covers `"word -- word"` / `"word--word"`, excludes a leading/trailing
/// `--` and a `---` run.
///
/// `tokens` is the same count [`sentence_length_by_document`] uses; an
/// empty document has density `0.0`, not `NaN`.
///
/// # Example (hand-computed)
///
/// `"Speed matters — quality matters too."` + `"It works fine--somehow;
/// trust me."`: 11 tokens, 2 em-dash occurrences, density `= 2 × 1000/11
/// ≈ 181.818182`. Unit test of the same name also covers
/// [`semicolon_density`]'s `1 × 1000/11 ≈ 90.909091` on this fixture.
#[must_use]
pub fn em_dash_density(document: &Document) -> f64 {
    density_per_1000_tokens(document, count_em_dashes)
}

/// Semicolon density: `;` occurrences per 1000 tokens, over every
/// sentence in `document`. See [`em_dash_density`] for the token
/// definition and empty-document convention.
#[must_use]
pub fn semicolon_density(document: &Document) -> f64 {
    density_per_1000_tokens(document, count_semicolons)
}

/// Shared walk for the two "per 1000 tokens" densities: accumulates
/// `count_in`'s occurrence count and the token count over every sentence,
/// then divides once at the end.
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

/// Whether `chars[start..end]` (the hyphen run) has an alphanumeric
/// character just outside it on both sides (skipping at most one
/// flanking space).
fn is_word_flanked(chars: &[char], start: usize, end: usize) -> bool {
    flank_before(chars, start).is_some_and(char::is_alphanumeric)
        && flank_after(chars, end).is_some_and(char::is_alphanumeric)
}

/// Character just before `chars[start]` (skips one space). `None` at the
/// start of `chars`, or if the skipped space is itself the start.
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

/// Character just after `chars[end - 1]` (skips one space). `None` at the
/// end of `chars`, or if the skipped space is itself the end.
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

    /// Builds a one-block, one-paragraph document from `sentences` (joined
    /// text) and each sentence's byte range, via `friction-core`'s own
    /// constructors — independent of `friction-parse`/`friction-nlp`.
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
    /// positioned in `source`; one `Block`/`ProseUnit` per paragraph.
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

    /// Same 3/10/3-token fixture as the hand-computed derivation in
    /// [`sentence_length_by_document`]'s doc comment.
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

    /// Two paragraphs: first has sentences 3, 10 (`mean = 6.5`, `stddev =
    /// 3.5` exactly, `cv = 7/13`); second has one sentence of length 3
    /// (`stddev = 0`, `cv = 0`, single-observation convention).
    // Exact-zero guarantee, not an approximation.
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

    /// Three paragraphs with 2, 4, and 3 sentences: `mean = 3`;
    /// `variance = 2/3`; `stddev = sqrt(2/3)`; `cv = stddev / 3`.
    #[test]
    fn paragraph_shape_matches_hand_computation() {
        // Sentence text doesn't matter, only per-paragraph sentence
        // *count*. Each sentence is one-token "a." at byte range
        // `3*i .. 3*i + 2`. Paragraphs: sentences 0-1, 2-5, 6-8.
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

    /// Same fixture as the hand-computed example in [`em_dash_density`]'s
    /// doc comment.
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

        // One surrogate occurrence ("word--word"); check the numerator
        // directly by reconstructing density's own token divisor.
        let expected_tokens: usize = [s1, s2, s3]
            .iter()
            .map(|s| s.split_whitespace().count())
            .sum();
        #[allow(clippy::cast_precision_loss)]
        let expected = 1.0 * 1000.0 / expected_tokens as f64;
        assert!((em_dash_density(&doc) - expected).abs() < EPSILON);
    }

    /// An empty document yields the all-zero degenerate `RhythmStats` and
    /// zero densities, never `NaN`.
    // Exact-zero guarantee, not an approximation.
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
    // Exact-zero guarantee, not an approximation.
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
