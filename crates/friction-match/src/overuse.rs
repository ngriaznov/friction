//! Per-document word-overuse detection (a real user complaint: "pin" used
//! constantly in one review).
//!
//! An LLM document can hammer a single verb or adverb — a word the writer
//! keeps reaching for — at a rate no individual human review ever
//! sustains, even though the word itself is unremarkable (unlike
//! [`crate::jargon`]'s invented compounds, there is nothing wrong with
//! "pin" as a word). This channel catches that: it compares each
//! candidate word's rate WITHIN THIS ONE DOCUMENT against that word's
//! burst envelope — the densest any single human document in
//! [`friction_packs::HUMAN_EVIDENCE`]'s 49.7M-token corpus ever used it
//! ([`HumanEvidencePack::burst_envelope_per_million`]), and flags only
//! when the document's own confidence interval exceeds what any human
//! document ever sustained.
//!
//! Nouns are exempt by design — see [`is_lexical_choice_tag`]: a human
//! email about Haskell says "haskell" at five hundred times its
//! corpus-wide rate because that is its topic, and no threshold
//! separates a topic noun's burst from a stylistic one. Judging only
//! the parts of speech a writer chooses (rather than the ones the
//! subject dictates) is what makes a per-document rate comparison mean
//! "overuse" at all, which is also why this channel consumes the tagged
//! sentences [`crate::tagging::tag_units`] produces instead of raw
//! tokens.
//!
//! # Why a Wilson lower bound, not a raw ratio
//!
//! A raw point-rate comparison would flag any short document the moment
//! one content word happens to repeat a handful of times — four
//! occurrences of an ordinary word in a 60-word paragraph is not evidence
//! of overuse, it is just what short prose does. [`wilson_lower_bound_per_million`]
//! is the same evidence-proportional arming mechanism this codebase
//! already uses for register-band arming
//! (`friction_edit::register`'s `wilson_lower`/`wilson_upper`, guarding
//! nominalization/em-dash/semicolon/agentless-passive bands): the document
//! must be confidently, not just measurably, above the human rate. A short
//! document's lower bound sits far below its point rate and stays quiet;
//! the same density sustained over a longer document clears the bound and
//! arms. [`MIN_OCCURRENCES`] additionally floors out documents too short
//! to say anything at all.
//!
//! This channel cannot import that helper directly — `friction-edit`
//! depends on `friction-match`, not the reverse (see that crate's own
//! `Cargo.toml`) — so [`wilson_lower_bound_per_million`] is the same
//! closed-form Wilson score interval, re-derived here at a per-million
//! (not per-1000-word) scale to match [`friction_packs::HumanEvidencePack`]'s
//! own rate convention.
//!
//! # One constant id, like `jargon.metaphor`
//!
//! Every flagged span carries the same [`OVERUSE_WORD_ID`] regardless of
//! which word triggered it — the same choice [`crate::jargon`] makes and
//! for the same reason (see [`crate::jargon::JARGON_METAPHOR_ID`]'s own
//! docs): the specific word is per-occurrence detail, not a second
//! classification axis this crate's span shape needs to carry. Since the
//! word, its count, and both rates would otherwise have nowhere else to
//! live, each span's [`MatchSpan::message`] carries them instead.
//!
//! # Detection-only
//!
//! Same status as `jargon.metaphor` (see that module's own docs): there is
//! no deterministic true replacement for "the document uses this word too
//! much" — which word to swap in and where is a judgment call, not a
//! mechanical rewrite. This channel has no entry in `friction-edit`'s
//! repair engine and never will on its own; a flagged span is reported,
//! never rewritten.

use std::collections::BTreeMap;
use std::ops::Range;

use friction_packs::HumanEvidencePack;

use crate::span::{Channel, MatchScore, MatchSpan};
use crate::tagging::TaggedSentence;

/// This channel's one `frame_id` — constant across every flagged word; see
/// this module's own docs for why.
pub const OVERUSE_WORD_ID: &str = "overuse.word";

/// A candidate word must be at least this many ASCII letters long.
/// Shorter than this and a token is more likely a function word or an
/// abbreviation than a genuine lexical choice; the length floor keeps
/// two-letter noise out of the candidate set before any rate math runs.
const MIN_WORD_LEN: usize = 3;

/// A word must occur at least this many times in the document before it is
/// even considered — below this, [`wilson_lower_bound_per_million`]'s own
/// evidence-proportional arming would almost never clear a burst
/// envelope anyway (a handful of instances produces a lower bound close
/// to zero), but checking the raw count first skips the bound
/// computation entirely for the overwhelming majority of words in any
/// document, which occur once or twice.
const MIN_OCCURRENCES: u64 = 4;

/// The factor a document's Wilson-lower-bound rate must exceed a word's
/// burst envelope by before this channel flags it. The envelope is
/// already the most extreme per-document rate any human document in the
/// reference corpus sustained (an observed maximum, not a typical
/// value) so the headroom is thin: the claim being made is "denser than
/// any human document ever", and 1.2x keeps sampling wobble on both
/// sides of that comparison from deciding it.
const ENVELOPE_HEADROOM: f64 = 1.2;

/// The z-score [`wilson_lower_bound_per_million`] computes at — 1.96, the
/// conventional 95% figure, matching `friction_edit::register::WILSON_Z`
/// (duplicated, not imported: see this module's own docs on the
/// dependency direction).
const WILSON_Z: f64 = 1.96;

/// `true` for a candidate word's TEXT shape: at least [`MIN_WORD_LEN`]
/// characters and made ENTIRELY of ASCII letters (any case — the raw
/// source slice arrives cased; [`tally`] folds it for the tally key). A
/// contraction like "don't" is excluded here rather than by the
/// tokenizer, since a contraction is not a lexical choice comparable to
/// [`friction_packs::HumanEvidencePack`]'s own unigram entries: it is
/// the kind of high-frequency grammatical glue this channel's whole
/// point is to exclude.
fn is_candidate_shape(text: &str) -> bool {
    text.len() >= MIN_WORD_LEN && text.bytes().all(|b| b.is_ascii_alphabetic())
}

/// Both Wilson score interval bounds have the closed form used by
/// `friction_edit::register::wilson_bounds`; this channel only ever needs
/// the LOWER bound (arming requires the document to be confidently ABOVE
/// the human rate, never confidently below it), scaled to a per-million
/// rate to match [`friction_packs::HumanEvidencePack`]'s own convention
/// instead of per-1000-words.
///
/// `0.0` for `total == 0` (never hit in practice — [`scan_units`] returns
/// early when a document has no prose tokens at all). Closed-form and
/// deterministic: IEEE-754 `sqrt` is exactly rounded, so identical
/// `(count, total)` inputs produce identical bits on every platform.
#[must_use]
fn wilson_lower_bound_per_million(count: u64, total: u64) -> f64 {
    if total == 0 {
        return 0.0;
    }
    #[allow(clippy::cast_precision_loss)]
    let (k, n) = (count as f64, total as f64);
    let p = k / n;
    let z2 = WILSON_Z * WILSON_Z;
    let denominator = 1.0 + z2 / n;
    let centre = p + z2 / (2.0 * n);
    let spread = WILSON_Z * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt();
    let lower = ((centre - spread) / denominator).max(0.0);
    lower * 1_000_000.0
}

/// `count` per million `total` — the plain point-rate `wilson_lower_bound_per_million`
/// arms against, and the figure this channel's message reports for the
/// document's own side (see [`overuse_message`]).
#[must_use]
fn rate_per_million(count: u64, total: u64) -> f64 {
    if total == 0 {
        return 0.0;
    }
    #[allow(clippy::cast_precision_loss)]
    let (k, n) = (count as f64, total as f64);
    k / n * 1_000_000.0
}

/// The measured-rates message every flagged occurrence of `word` shares,
/// e.g. `"'pin' appears 6 times (~9800/M vs 210/M human; report-only)"` —
/// matching this codebase's existing measured-rate wording
/// (`friction_edit::register`'s remainder findings report a feature's own
/// rate against its band the same way: a number, a comparison, a plain
/// English gloss of what that means).
fn overuse_message(word: &str, count: u64, doc_rate: f64, envelope: u32) -> String {
    let times = if count == 1 { "time" } else { "times" };
    format!(
        "'{word}' appears {count} {times} (~{doc_rate:.0}/M; densest human document: \
         {envelope}/M; report-only)"
    )
}

/// The smallest document (in tagger tokens, punctuation included) this
/// channel will judge at all. Below a paragraph-and-change of text,
/// "this word repeats" is a mention cluster, not a usage pattern — a
/// five-token fragment can push a Wilson lower bound over any
/// multiplier with a single repeated word, and no honest per-document
/// rate exists at that size. The same short-text-stays-quiet principle
/// as `friction_edit::register`'s Wilson arming, applied as a hard
/// floor because this channel's statistic is document-global.
const MIN_DOC_TOKENS: u64 = 120;

/// `true` for the part-of-speech tags whose words this channel judges:
/// finite/base/participial verbs (`VB`, `VBD`, `VBN`, `VBP`, `VBZ`) and
/// adverbs (`RB*`): the words style chooses and no topic dictates.
/// Every excluded class fell to the human-corpus evidence gate, not to
/// taste: nouns because a human email about Haskell says "haskell" at
/// hundreds of times its corpus-wide rate (a topic, not a tic);
/// adjectives because attributive topic adjectives burst the same way
/// ("regional" five times in one human email about regional
/// operations); and gerunds (`VBG`) because a gerund heads noun phrases
/// and carries a topic exactly like a noun does ("coaching" eight times
/// in one human document about coaching). Repeatedly reaching for the
/// same finite verb or adverb, by contrast, is a stylistic fingerprint
/// aboutness cannot explain.
fn is_lexical_choice_tag(tag: &str) -> bool {
    (tag.starts_with("VB") && tag != "VBG") || tag.starts_with("RB")
}

/// One candidate word's tally while [`scan_units`] walks the document:
/// every occurrence's byte range, in source order (tagged sentences come
/// from [`crate::tagging::tag_units`] in document order).
type Occurrences = BTreeMap<Box<str>, Vec<Range<usize>>>;

/// Pass one of [`scan_units`], shared with the test-side margin probe:
/// every candidate word's occurrence ranges plus the document's total
/// tagger-token count (word and punctuation tokens both, the closest
/// available analogue of [`friction_packs::HumanEvidencePack`]'s own
/// words-and-punctuation token basis).
///
/// A word qualifies only when EVERY one of its occurrences carries a
/// lexical-choice tag: unanimity, not majority. The tagger guesses at
/// words outside its dictionary from context, and its guesses for tech
/// vocabulary drift between noun and adjective readings ("nodejs" as
/// `JJ` before "runtime", `NN` elsewhere); a word the tagger cannot
/// identify consistently has no trustworthy part of speech at all, and
/// this channel's topic-noun exemption only means something when
/// uncertain words fail closed. A genuinely chosen verb or modifier
/// tags consistently across a document; a mis-tagged topic word does
/// not.
fn tally(tagged: &[TaggedSentence], source: &str) -> (Occurrences, u64) {
    let mut occurrences: Occurrences = BTreeMap::new();
    // `total_tokens` alone decides whether `scan_units` judges this
    // document at all ([`MIN_DOC_TOKENS`]) and is knowable from each
    // sentence's token *count* without visiting a single token — so it's
    // computed first and checked before any per-token work (slicing,
    // case-folding, map lookups) runs at all. A no-op for the two
    // benchmark documents (both are far larger than the floor), but a
    // real, unconditional skip for every short document this channel
    // would have discarded via `total_tokens < MIN_DOC_TOKENS` anyway.
    let total_tokens: u64 = tagged.iter().map(|s| s.tokens.len() as u64).sum();
    if total_tokens < MIN_DOC_TOKENS {
        return (occurrences, total_tokens);
    }

    let mut nonlexical: std::collections::BTreeSet<Box<str>> = std::collections::BTreeSet::new();
    // Reused across every candidate token instead of allocating a fresh
    // lowercased `String` per occurrence: `to_ascii_lowercase` output
    // is written into this buffer once per token, and only turned into
    // an owned `Box<str>` key on a new word — the common case (a
    // repeated word) looks the existing entry up by the borrowed buffer
    // contents and pushes into its `Vec` without allocating at all.
    // `is_candidate_shape` already restricts a candidate to ASCII
    // letters, so ASCII case-folding here is exactly `to_lowercase`'s
    // result for every string this function ever sees.
    let mut lower = String::new();
    for sentence in tagged {
        for token in &sentence.tokens {
            let text = &source[token.token.range.clone()];
            if !is_candidate_shape(text) {
                continue;
            }
            let key: &str = if text.bytes().all(|b| b.is_ascii_lowercase()) {
                text
            } else {
                lower.clear();
                lower.extend(text.chars().map(|c| c.to_ascii_lowercase()));
                &lower
            };
            if is_lexical_choice_tag(token.pos.as_str()) {
                if let Some(ranges) = occurrences.get_mut(key) {
                    ranges.push(token.token.range.clone());
                } else {
                    occurrences.insert(key.into(), vec![token.token.range.clone()]);
                }
            } else if !nonlexical.contains(key) {
                nonlexical.insert(key.into());
            }
        }
    }
    occurrences.retain(|word, _| !nonlexical.contains(word));
    (occurrences, total_tokens)
}

/// Reports every `overuse.word` occurrence across `units`.
///
/// A two-pass whole-document scan, unlike [`crate::jargon::scan_units`]'s
/// per-sentence independence — "does this document overuse this word"
/// is inherently a document-wide question (`n` and `T` are both totals
/// over the whole document), so there is no independent per-sentence unit
/// of work to parallelize the way jargon's per-sentence compound search
/// has.
///
/// Pass one tallies every candidate word's occurrences and the document's
/// total prose token count `T` (word AND punctuation tokens both count,
/// matching [`friction_packs::HumanEvidencePack::total_tokens`]'s own
/// `friction_harness::clean::tokenize` convention, see that pack's module
/// docs, so a document rate and a human rate are computed on the same
/// basis). Pass two arms each candidate against
/// [`OVERUSE_RATIO_MULTIPLIER`] and, for every word that arms, emits one
/// span per occurrence with a shared [`overuse_message`].
///
/// Returned in ascending byte-range order — candidate words are visited in
/// [`BTreeMap`] (word-text) order, not document order, so the per-word
/// spans are sorted once at the end rather than relying on iteration order
/// to already be document order.
#[must_use]
pub fn scan_units(
    tagged: &[TaggedSentence],
    source: &str,
    evidence: &HumanEvidencePack,
) -> Vec<MatchSpan> {
    let (occurrences, total_tokens) = tally(tagged, source);
    if total_tokens < MIN_DOC_TOKENS {
        return Vec::new();
    }

    let mut spans = Vec::new();
    for (word, ranges) in occurrences {
        let count = ranges.len() as u64;
        if count < MIN_OCCURRENCES {
            continue;
        }
        let Some(envelope) = evidence.burst_envelope_per_million(&word) else {
            continue;
        };
        let lower_bound = wilson_lower_bound_per_million(count, total_tokens);
        if lower_bound <= ENVELOPE_HEADROOM * f64::from(envelope) {
            continue;
        }

        let doc_rate = rate_per_million(count, total_tokens);
        let message: Box<str> = overuse_message(&word, count, doc_rate, envelope).into();
        spans.extend(ranges.into_iter().map(|range| MatchSpan {
            range,
            channel: Channel::Overuse,
            frame_id: OVERUSE_WORD_ID.into(),
            score: MatchScore::Present,
            message: Some(message.clone()),
        }));
    }

    spans.sort_by(|a, b| {
        a.range
            .start
            .cmp(&b.range.start)
            .then(a.range.end.cmp(&b.range.end))
    });
    spans
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::sync::OnceLock;

    use friction_core::{Block, BlockKind, Document, ProseUnit};
    use friction_nlp::{PerceptronTagger, SrxSegmenter};
    use friction_packs::build_human_evidence_bin;

    use super::*;

    /// The real embedded tagger, shared across tests exactly as
    /// `jargon.rs`'s own tests share theirs. The POS gate is part of the
    /// detection rule, so these tests run real tags, not hand-picked ones.
    fn tagger() -> &'static PerceptronTagger {
        static TAGGER: OnceLock<PerceptronTagger> = OnceLock::new();
        TAGGER.get_or_init(|| PerceptronTagger::new().expect("embedded model must load"))
    }

    /// A small synthetic evidence pack, standing in for the real embedded
    /// `HUMAN_EVIDENCE`: `pin`, `tweak`, and `adjust` at low human rates
    /// (clearing [`HUMAN_RATE_CEILING_PER_MILLION`] easily), `the` at
    /// function-word tier (must never flag regardless of count), and
    /// nothing else — so any other word is "absent from the table" and
    /// must never flag either.
    fn synthetic_evidence() -> HumanEvidencePack {
        let total = 1_000_000u64;
        // Burst envelopes: what the densest human document ever did with
        // each word. Low for the style words the positive tests hammer,
        // high for "really"/"the" (documents repeat glue words freely).
        let unigrams = BTreeMap::from([
            ("pin".to_string(), (210, 3_000)),
            ("tweak".to_string(), (5, 2_000)),
            ("adjust".to_string(), (5, 2_000)),
            ("really".to_string(), (8_000, 60_000)),
            ("the".to_string(), (60_000, 120_000)),
        ]);
        let bin = build_human_evidence_bin(&unigrams, &BTreeMap::new(), total);
        HumanEvidencePack::load(&bin).expect("synthetic pack loads")
    }

    fn scoped_document(source: &str) -> Document {
        let blocks = vec![Block::new(BlockKind::Paragraph, 0..source.len())];
        let prose = vec![ProseUnit::new(0, 0..source.len(), Vec::new())];
        Document::new(source, blocks, prose).expect("document is well-formed")
    }

    fn spans_for(source: &str, evidence: &HumanEvidencePack) -> Vec<MatchSpan> {
        let document = scoped_document(source);
        let segmenter = SrxSegmenter::new();
        let units = crate::token::prose_scope(&document, &segmenter);
        let tagged = crate::tagging::tag_units(&units, source, tagger());
        scan_units(&tagged, source, evidence)
    }

    /// Filler prose whose every content word is absent from
    /// [`synthetic_evidence`]'s table (and so can never flag), long
    /// enough to lift a test document over [`MIN_DOC_TOKENS`].
    fn filler() -> String {
        "The team reviewed every change and merged the branch without \
         much discussion, because nothing in it touched a public \
         interface or a stored schema. "
            .repeat(6)
    }

    #[test]
    fn is_candidate_shape_rejects_short_and_non_alphabetic_tokens() {
        assert!(is_candidate_shape("pin"));
        assert!(is_candidate_shape("Pin"));
        assert!(!is_candidate_shape("it"));
        assert!(!is_candidate_shape("don't"));
        assert!(!is_candidate_shape("a1b"));
    }

    #[test]
    fn lexical_choice_tags_cover_finite_verbs_and_adverbs_only() {
        for tag in ["VB", "VBZ", "VBD", "VBN", "VBP", "RB", "RBR", "RBS"] {
            assert!(is_lexical_choice_tag(tag), "{tag}");
        }
        for tag in [
            "VBG", "JJ", "JJR", "NN", "NNS", "NNP", "DT", "IN", "PRP", "MD", "CD",
        ] {
            assert!(!is_lexical_choice_tag(tag), "{tag}");
        }
    }

    #[test]
    fn wilson_lower_bound_is_deterministic_and_safe_on_empty_input() {
        assert!(wilson_lower_bound_per_million(0, 0).abs() < f64::EPSILON);
        assert!(
            (wilson_lower_bound_per_million(6, 500) - wilson_lower_bound_per_million(6, 500)).abs()
                < f64::EPSILON
        );
    }

    /// A review that hammers verb-position "pin" far past its ~210/M
    /// human rate — the real complaint this channel exists for. Every
    /// verb occurrence flags; the topic-noun sense would not (see
    /// [`negative_topic_nouns_never_flag`]).
    #[test]
    fn positive_pin_used_constantly_flags_every_occurrence() {
        let evidence = synthetic_evidence();
        let source = format!(
            "{}You should pin the schema in the first test. Please pin \
             the response shape too, and we pin the offsets before every \
             assertion. They pin the wire contract, so also pin the \
             headers, and then pin the ports in the last suite.",
            filler()
        );
        let spans = spans_for(&source, &evidence);
        assert_eq!(spans.len(), 6, "{spans:?}");
        for span in &spans {
            assert_eq!(&*span.frame_id, OVERUSE_WORD_ID);
            assert_eq!(span.channel, Channel::Overuse);
            assert_eq!(&source[span.range.clone()], "pin");
            let message = span
                .message
                .as_deref()
                .expect("overuse span carries a message");
            assert!(message.starts_with("'pin' appears"), "{message}");
            assert!(message.contains("human"), "{message}");
            assert!(message.contains("report-only"), "{message}");
        }
    }

    /// A document about pins — the same word at the same density, but in
    /// noun position throughout — never flags: topic nouns are aboutness,
    /// not overuse (see [`is_lexical_choice_tag`]).
    #[test]
    fn negative_topic_nouns_never_flag() {
        let evidence = synthetic_evidence();
        let source = format!(
            "{}The pin arrived quickly and the pin looked great on the \
             jacket. A second pin came in the box, and that pin matched \
             the first pin, so the pin collection grew.",
            filler()
        );
        let spans = spans_for(&source, &evidence);
        assert!(spans.is_empty(), "{spans:?}");
    }

    /// Spans come back in ascending byte order even though candidates are
    /// visited in word-text (`BTreeMap`) order internally.
    #[test]
    fn positive_spans_are_sorted_by_position_not_word() {
        // Two distinct low-rate verbs, interleaved in the source, whose
        // BTreeMap (alphabetical) visitation order disagrees with their
        // document order: "tweak" is visited last internally but its
        // first occurrence comes before "adjust"'s later ones.
        let evidence = synthetic_evidence();
        let source = format!(
            "{}We tweak the layout first and adjust the padding after. \
             Then we tweak the margins and adjust the gutters again. \
             Later we tweak the colors and adjust the contrast once more. \
             Finally we tweak the spacing and adjust the baseline grid.",
            filler()
        );
        let spans = spans_for(&source, &evidence);
        assert!(!spans.is_empty(), "expected both words to arm");
        let mut sorted = spans.clone();
        sorted.sort_by_key(|a| a.range.start);
        assert_eq!(spans, sorted);
    }

    /// Fewer than [`MIN_OCCURRENCES`] never flags, no matter how far above
    /// the human rate the point estimate sits.
    #[test]
    fn negative_below_min_occurrences_never_flags() {
        let evidence = synthetic_evidence();
        let source = format!(
            "{}You should pin the schema here, and later we pin the \
             response shape, and that is the whole ask.",
            filler()
        );
        assert!(spans_for(&source, &evidence).is_empty());
    }

    /// A word absent from the evidence table (fewer than five human
    /// occurrences, or genuinely never seen) never flags, regardless of
    /// how many times the document repeats it — "no reliable human rate"
    /// must never be silently treated as "human rate near zero".
    #[test]
    fn negative_word_absent_from_evidence_table_never_flags() {
        let evidence = synthetic_evidence();
        let source = format!(
            "{}We refactor the parser, then refactor the printer, then \
             refactor the loader, and finally refactor the walker, and \
             we refactor the runner and refactor the setup too.",
            filler()
        );
        assert!(spans_for(&source, &evidence).is_empty());
    }

    /// A word at or above [`HUMAN_RATE_CEILING_PER_MILLION`], however
    /// lexical its tag, never flags: at that frequency it is
    /// structural glue, and every document repeats it.
    #[test]
    fn negative_ceiling_tier_word_never_flags() {
        let evidence = synthetic_evidence();
        let source = format!(
            "{}This is really solid, really careful, really thorough \
             work, really well tested, really well documented, and \
             really easy to follow, really.",
            filler()
        );
        assert!(spans_for(&source, &evidence).is_empty());
    }

    /// A document below [`MIN_DOC_TOKENS`] never arms, no matter how
    /// dense a repeat is — a fragment's point rate is not a usage
    /// pattern, the same short-text-stays-quiet behavior
    /// `friction_edit::register`'s own Wilson arming documents, applied
    /// here as a hard floor.
    #[test]
    fn negative_below_min_doc_tokens_never_arms() {
        let evidence = synthetic_evidence();
        let source = "We pin this, we pin that, we pin those, we pin these.";
        let spans = spans_for(source, &evidence);
        assert!(spans.is_empty(), "{spans:?}");
    }

    /// The embedded corpus reader other tests in this crate use — mirrors
    /// `jargon.rs`'s own test-doc-building helpers, reading the repo's
    /// `corpus/human/` tree at test time rather than embedding a copy.
    fn corpus_human_root() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus/human")
    }

    /// The evidence gate: runs the full detection rule, with the REAL
    /// embedded `HUMAN_EVIDENCE` pack, over every document under
    /// `corpus/human/` (every genre) and asserts zero findings.
    ///
    /// This is what licenses [`OVERUSE_RATIO_MULTIPLIER`] and
    /// [`MIN_OCCURRENCES`]: `HUMAN_EVIDENCE` is itself pooled FROM this
    /// same corpus (see that pack's own module docs), so a human document
    /// flagging here would mean the constants are miscalibrated against
    /// the very data that defines "human rate": a true near-no-op canary,
    /// not just a spot check. Skips a document that fails to parse rather
    /// than failing the gate on an unrelated parser issue (never expected
    /// for the corpus's own markdown, but not this test's concern either
    /// way).
    #[test]
    fn overuse_word_evidence_gate_zero_findings_on_every_human_corpus_document() {
        let root = corpus_human_root();
        assert!(
            root.is_dir(),
            "corpus/human/ must exist at {}",
            root.display()
        );
        let segmenter = SrxSegmenter::new();
        let mut scanned = 0usize;
        let mut worst: Option<(String, f64)> = None;

        for entry in walk_markdown_files(&root) {
            let Ok(source) = std::fs::read_to_string(&entry) else {
                continue;
            };
            let Ok(document) = friction_parse::parse(source.as_str()) else {
                continue;
            };
            let units = crate::token::prose_scope(&document, &segmenter);
            let tagged = crate::tagging::tag_units(&units, document.source(), tagger());
            let spans = scan_units(&tagged, document.source(), &friction_packs::HUMAN_EVIDENCE);
            assert!(
                spans.is_empty(),
                "human corpus document {} produced overuse findings: {spans:?}",
                entry.display()
            );
            scanned += 1;

            let margin = closest_margin_to_arming(
                &tagged,
                document.source(),
                &friction_packs::HUMAN_EVIDENCE,
            );
            if let Some(margin) = margin
                && worst.as_ref().is_none_or(|(_, best)| margin > *best)
            {
                worst = Some((entry.display().to_string(), margin));
            }
        }

        assert!(scanned > 0, "expected at least one corpus/human/ document");
        if let Some((path, margin)) = worst {
            eprintln!(
                "overuse evidence gate: closest human document to arming was {path} \
                 at {margin:.3}x its arming threshold (1.0x would flag)"
            );
        }
    }

    /// For every candidate word in `units`, `lower_bound / (multiplier *
    /// human_rate)` — how close that word came to arming, as a fraction of
    /// the threshold (`1.0` would have flagged). `None` if `units` has no
    /// eligible candidate at all. Diagnostic-only: computed the same way
    /// [`scan_units`] itself gates, minus the final `>=` comparison,
    /// purely so the evidence-gate test can report its own margin instead
    /// of only asserting pass/fail.
    fn closest_margin_to_arming(
        tagged: &[TaggedSentence],
        source: &str,
        evidence: &HumanEvidencePack,
    ) -> Option<f64> {
        let (occurrences, total_tokens) = tally(tagged, source);
        if total_tokens < MIN_DOC_TOKENS {
            return None;
        }

        let mut best: Option<f64> = None;
        for (word, ranges) in occurrences {
            let count = ranges.len() as u64;
            if count < MIN_OCCURRENCES {
                continue;
            }
            let Some(envelope) = evidence.burst_envelope_per_million(&word) else {
                continue;
            };
            if envelope == 0 {
                continue;
            }
            let lower_bound = wilson_lower_bound_per_million(count, total_tokens);
            let margin = lower_bound / (ENVELOPE_HEADROOM * f64::from(envelope));
            if best.is_none_or(|b| margin > b) {
                best = Some(margin);
            }
        }
        best
    }

    /// Every `.md` file under `root`, recursively, in a stable
    /// (directory-walk) order: good enough for a test that only needs
    /// "every file", not a specific order.
    fn walk_markdown_files(root: &Path) -> Vec<std::path::PathBuf> {
        let mut out = Vec::new();
        let mut stack = vec![root.to_path_buf()];
        while let Some(dir) = stack.pop() {
            let Ok(read_dir) = std::fs::read_dir(&dir) else {
                continue;
            };
            for entry in read_dir.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                } else if path.extension().is_some_and(|ext| ext == "md") {
                    out.push(path);
                }
            }
        }
        out
    }
}
