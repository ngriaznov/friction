//! Honest agreement numbers between the default [`PerceptronTagger`] and
//! the `nlprule`-backed comparison tagger, plus the one hard bar that
//! actually matters operationally: the two taggers must drive the pivot
//! detector to the identical accept/reject verdict on every sentence in
//! this workspace's own literal fixture set.
//!
//! Gated behind the `nlprule` feature (needs both taggers loaded);
//! regenerate `tests/data/perceptron_parity_report.md` by running:
//!
//! ```text
//! cargo test -p friction-nlp --features nlprule --test perceptron_parity -- --nocapture
//! ```
//!
//! and committing the resulting diff — a living, honest artifact, not a
//! hidden number. Raw tag-agreement assertions are deliberately weak (this
//! test reports the real numbers rather than gating the build on a tuned
//! target); the pivot-verdict-parity assertion is the one hard check.

// This whole test needs both taggers loaded, so it compiles to nothing at
// all under default features rather than merely skipping its one test at
// runtime — `cargo test`/`cargo clippy --all-targets` without `--features
// nlprule` never even sees `NlpruleTagger` referenced.
#![cfg(feature = "nlprule")]

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;

use friction_core::TokenKind;
use friction_harness::fixtures::{CHATSPEAK_MD, FIXTURES, GOOD_DOCS_MD};
use friction_harness::pivot::match_pivot;
use friction_nlp::lvc::DERIVATIONAL_LEXICON;
use friction_nlp::{NlpruleTagger, PerceptronTagger, Segmenter, SrxSegmenter, TaggedToken, Tagger};

/// Full Penn tags that count as a finite verb for clause completeness.
const FINITE_VERB_TAGS: [&str; 4] = ["VBZ", "VBP", "VBD", "MD"];

/// Mirrors `tests/segment_golden.rs`'s own fixture shape, minus the fields
/// this test does not need.
#[derive(serde::Deserialize)]
struct GoldenFixture {
    case: Vec<GoldenCase>,
}

#[derive(serde::Deserialize)]
struct GoldenCase {
    expect: Vec<String>,
}

/// Every sentence this test pools its metrics over: every `accept`/`reject`
/// fixture's own input text (including `pivot_constructed_cases`' five
/// pairs and `heading_pivot`'s block text), `segment_golden.toml`'s 98
/// golden sentences, and `chatspeak.md`/`good_docs.md` segmented with the
/// real sentence segmenter.
fn push_segmented(sentences: &mut Vec<String>, segmenter: SrxSegmenter, text: &str) {
    for range in segmenter.segment(text, 0) {
        let s = text[range].trim();
        if !s.is_empty() {
            sentences.push(s.to_string());
        }
    }
}

fn sentence_pool() -> Vec<String> {
    let mut sentences = Vec::new();
    let segmenter = SrxSegmenter::new();

    for fixture in &FIXTURES.accept {
        if let Some(input) = &fixture.input {
            push_segmented(&mut sentences, segmenter, input);
        }
        if let Some(pairs) = &fixture.pairs {
            for (input, _) in pairs {
                push_segmented(&mut sentences, segmenter, input);
            }
        }
    }
    for fixture in &FIXTURES.reject {
        if let Some(input) = &fixture.input {
            push_segmented(&mut sentences, segmenter, input);
        }
        if let Some(block) = &fixture.input_block {
            push_segmented(&mut sentences, segmenter, &block.text);
        }
    }

    let golden_toml = include_str!("data/segment_golden.toml");
    let parsed: GoldenFixture = toml::from_str(golden_toml).expect("segment_golden.toml parses");
    for case in parsed.case {
        sentences.extend(case.expect);
    }

    push_segmented(&mut sentences, segmenter, CHATSPEAK_MD);
    push_segmented(&mut sentences, segmenter, GOOD_DOCS_MD);

    sentences.sort();
    sentences.dedup();
    sentences
}

/// Tags `sentence` with both taggers and returns `(perceptron tokens,
/// nlprule tokens)`.
fn tag_both(
    sentence: &str,
    perceptron: &PerceptronTagger,
    nlprule: &NlpruleTagger,
) -> (Vec<TaggedToken>, Vec<TaggedToken>) {
    (perceptron.tag(sentence, 0), nlprule.tag(sentence, 0))
}

/// `nlprule`'s own dictionary tags a noun's countability as a suffix on
/// the base Penn tag (`"NN:UN"`, `"NN:U"`, ...) — a real distinction
/// `nlprule` draws that this workspace's plain-Penn-tag convention does
/// not. Comparing those raw strings against the perceptron's plain `"NN"`
/// would report every single noun as a "disagreement", which is not an
/// honest picture of tagging accuracy — it is entirely an artifact of the
/// two tagsets' granularity, not a real difference of opinion. This strips
/// exactly that suffix before any comparison in this file, the same
/// normalization `weights/NOTICE.md` documents applying to the gold
/// training data itself.
fn normalize_nlprule_tag(tag: &str) -> &str {
    tag.split(':').next().unwrap_or(tag)
}

/// Aligns two tag streams by exact byte range (both were tagged from the
/// same `base_offset`), returning `(perceptron tag, nlprule tag)` pairs
/// only for spans both taggers agree are a single token — the two
/// tokenizers occasionally disagree on span boundaries (nlprule splits a
/// possessive `"'s"` as its own token; this crate's tokenizer keeps it
/// attached), and those tokens are honestly excluded from the comparison
/// rather than silently mismatched against the wrong neighbor.
fn align<'a>(
    perceptron: &'a [TaggedToken],
    nlprule: &'a [TaggedToken],
) -> Vec<(&'a TaggedToken, &'a TaggedToken)> {
    let mut by_range: BTreeMap<(usize, usize), &TaggedToken> = BTreeMap::new();
    for token in nlprule {
        by_range.insert((token.token.range.start, token.token.range.end), token);
    }
    perceptron
        .iter()
        .filter_map(|p| {
            by_range
                .get(&(p.token.range.start, p.token.range.end))
                .map(|&n| (p, n))
        })
        .collect()
}

/// One overall or restricted agreement measurement.
struct AgreementStat {
    label: &'static str,
    agree: usize,
    total: usize,
}

impl AgreementStat {
    fn rate(&self) -> f64 {
        if self.total == 0 {
            f64::NAN
        } else {
            #[allow(clippy::cast_precision_loss)]
            let rate = self.agree as f64 / self.total as f64;
            rate
        }
    }
}

/// Accumulates the four tag-agreement measurements over every sentence in
/// `sentences`, aligning each sentence's two tag streams by exact byte
/// range first (see [`align`]).
fn collect_agreement_stats(
    sentences: &[String],
    perceptron: &PerceptronTagger,
    nlprule: &NlpruleTagger,
) -> Vec<AgreementStat> {
    let mut overall = (0usize, 0usize);
    let mut finite_verb = (0usize, 0usize);
    let mut jj = (0usize, 0usize);
    let mut sentence_initial_vb = (0usize, 0usize);

    for sentence in sentences {
        let (ptoks, ntoks) = tag_both(sentence, perceptron, nlprule);
        let aligned = align(&ptoks, &ntoks);

        for (p, n) in &aligned {
            if p.token.kind != TokenKind::Word {
                continue;
            }
            let n_tag = normalize_nlprule_tag(n.pos.as_str());
            let agrees = p.pos.as_str() == n_tag;
            overall.1 += 1;
            overall.0 += usize::from(agrees);
            if FINITE_VERB_TAGS.contains(&n_tag) {
                finite_verb.1 += 1;
                finite_verb.0 += usize::from(agrees);
            }
            if n_tag == "JJ" {
                jj.1 += 1;
                jj.0 += usize::from(agrees);
            }
        }

        if let (Some(p_first), Some(n_first)) = (ptoks.first(), ntoks.first())
            && p_first.token.range == n_first.token.range
            && normalize_nlprule_tag(n_first.pos.as_str()) == "VB"
        {
            sentence_initial_vb.1 += 1;
            sentence_initial_vb.0 += usize::from(p_first.pos.as_str() == "VB");
        }
    }

    vec![
        AgreementStat {
            label: "overall exact tag agreement",
            agree: overall.0,
            total: overall.1,
        },
        AgreementStat {
            label: "finite-verb-set agreement (VBZ/VBP/VBD/MD)",
            agree: finite_verb.0,
            total: finite_verb.1,
        },
        AgreementStat {
            label: "JJ agreement",
            agree: jj.0,
            total: jj.1,
        },
        AgreementStat {
            label: "sentence-initial VB agreement",
            agree: sentence_initial_vb.0,
            total: sentence_initial_vb.1,
        },
    ]
}

/// Derivational-lexicon NN-vs-VB confusion table: does the perceptron ever
/// tag a licensed nominalization as anything other than `NN` when seen on
/// its own?
fn collect_lexicon_rows(
    perceptron: &PerceptronTagger,
    nlprule: &NlpruleTagger,
) -> Vec<(&'static str, String, String)> {
    DERIVATIONAL_LEXICON
        .keys()
        .map(|&nominalization| {
            let p_tag = perceptron
                .tag(nominalization, 0)
                .first()
                .map_or_else(|| "?".to_string(), |t| t.pos.as_str().to_string());
            let n_tag = nlprule.tag(nominalization, 0).first().map_or_else(
                || "?".to_string(),
                |t| normalize_nlprule_tag(t.pos.as_str()).to_string(),
            );
            (nominalization, p_tag, n_tag)
        })
        .collect()
}

/// The one hard assertion's own data: every sentence where `match_pivot`'s
/// outcome differs between the two taggers.
fn collect_verdict_mismatches(
    sentences: &[String],
    perceptron: &PerceptronTagger,
    nlprule: &NlpruleTagger,
) -> Vec<(String, String, String)> {
    sentences
        .iter()
        .filter_map(|sentence| {
            let p_outcome = match_pivot(sentence, perceptron);
            let n_outcome = match_pivot(sentence, nlprule);
            (p_outcome != n_outcome).then(|| {
                (
                    sentence.clone(),
                    format!("{p_outcome:?}"),
                    format!("{n_outcome:?}"),
                )
            })
        })
        .collect()
}

#[test]
fn perceptron_parity_report_and_pivot_verdict_hard_bar() {
    let perceptron = PerceptronTagger::new().expect("embedded perceptron weights load");
    let nlprule = NlpruleTagger::new().expect("embedded nlprule model loads");

    let sentences = sentence_pool();
    assert!(
        sentences.len() >= 100,
        "expected a substantial sentence pool, got {}",
        sentences.len()
    );

    let stats = collect_agreement_stats(&sentences, &perceptron, &nlprule);
    let lexicon_rows = collect_lexicon_rows(&perceptron, &nlprule);
    let verdict_mismatches = collect_verdict_mismatches(&sentences, &perceptron, &nlprule);

    let report = render_report(&sentences, &stats, &lexicon_rows, &verdict_mismatches);
    println!("{report}");
    let report_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/perceptron_parity_report.md");
    std::fs::write(&report_path, &report)
        .unwrap_or_else(|err| panic!("failed to write {}: {err}", report_path.display()));

    assert!(
        verdict_mismatches.is_empty(),
        "perceptron vs. nlprule pivot verdict mismatch on {} sentence(s) -- see \
         tests/data/perceptron_parity_report.md for details:\n{verdict_mismatches:#?}",
        verdict_mismatches.len()
    );
}

fn render_report(
    sentences: &[String],
    stats: &[AgreementStat],
    lexicon_rows: &[(&str, String, String)],
    verdict_mismatches: &[(String, String, String)],
) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "# Perceptron vs. nlprule tagger parity report");
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "Regenerated by `cargo test -p friction-nlp --features nlprule --test \
         perceptron_parity -- --nocapture`. Sentence pool: {} sentences (fixtures.json \
         accept/reject inputs, segment_golden.toml's 98 golden sentences, and the \
         chatspeak/good_docs sample documents segmented with the real sentence segmenter).",
        sentences.len()
    );
    let _ = writeln!(out);
    let _ = writeln!(out, "## Tag agreement (aligned tokens only)");
    let _ = writeln!(out);
    let _ = writeln!(out, "| metric | agree | total | rate |");
    let _ = writeln!(out, "|---|---|---|---|");
    for stat in stats {
        let _ = writeln!(
            out,
            "| {} | {} | {} | {:.4} |",
            stat.label,
            stat.agree,
            stat.total,
            stat.rate()
        );
    }
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "These are honest agreement numbers, not a gate: the perceptron is trained on a \
         small hand-curated corpus (see `weights/NOTICE.md`) and is expected to disagree \
         with nlprule on real cases, especially rarer tags. Nothing above is asserted by \
         this test."
    );
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "## Derivational-lexicon nominalization tags (perceptron vs. nlprule, standalone)"
    );
    let _ = writeln!(out);
    let _ = writeln!(out, "| nominalization | perceptron | nlprule | agree |");
    let _ = writeln!(out, "|---|---|---|---|");
    for (word, p_tag, n_tag) in lexicon_rows {
        let agree = if p_tag == n_tag { "yes" } else { "NO" };
        let _ = writeln!(out, "| {word} | {p_tag} | {n_tag} | {agree} |");
    }
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "## Pivot accept/reject verdict parity (hard assertion)"
    );
    let _ = writeln!(out);
    if verdict_mismatches.is_empty() {
        let _ = writeln!(
            out,
            "All {} sentences: `match_pivot` returns the identical outcome under both \
             taggers.",
            sentences.len()
        );
    } else {
        let _ = writeln!(out, "| sentence | perceptron outcome | nlprule outcome |");
        let _ = writeln!(out, "|---|---|---|");
        for (sentence, p, n) in verdict_mismatches {
            let _ = writeln!(out, "| {sentence} | {p} | {n} |");
        }
    }
    out
}
