//! LLM-side effectiveness preview (informational only, no tuning): runs
//! `fix_document` over llm-class TRAIN docs and reports each genre's mean
//! combined out-of-envelope score (`friction_packs::EnvelopePack::
//! combined_score`, the same normalized-exceedance-over-included-metrics
//! score `corpus-tool separate` reports) before vs. after fixing, against
//! the shipped `envelope-v2` pack.
//!
//! This is a preview, not a gate: nothing here tunes a rule or a pack,
//! and no assertion fails the run based on the numbers it prints.
//!
//! Run with:
//!
//! ```text
//! cargo run -p friction-apply --release --example llm_effectiveness_preview
//! ```

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use corpus_tool::corpus_layout::relpath;
use corpus_tool::manifest::{Class, Genre, ManifestRecord, Split, read_manifest};
use friction_apply::FixEngine;
use friction_nlp::{NlpruleTagger, SrxSegmenter};
use friction_packs::ENVELOPE_V2;

/// The fixed set of five genres, in report order.
const GENRES: [Genre; 5] = [
    Genre::Docs,
    Genre::Blog,
    Genre::Readme,
    Genre::Email,
    Genre::Forum,
];

#[derive(Debug, Clone, Copy, Default)]
struct GenreScores {
    docs: usize,
    /// Sum of each doc's combined score before fixing (only docs with a
    /// defined combined score count toward `docs`/these sums).
    before_total: f64,
    after_total: f64,
}

impl GenreScores {
    fn mean_before(self) -> f64 {
        if self.docs == 0 {
            0.0
        } else {
            #[allow(clippy::cast_precision_loss)]
            let m = self.before_total / self.docs as f64;
            m
        }
    }

    fn mean_after(self) -> f64 {
        if self.docs == 0 {
            0.0
        } else {
            #[allow(clippy::cast_precision_loss)]
            let m = self.after_total / self.docs as f64;
            m
        }
    }
}

fn corpus_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus")
}

fn llm_train_records(corpus_dir: &Path) -> Vec<ManifestRecord> {
    let manifest_path = corpus_dir.join("manifest.jsonl");
    let mut records: Vec<ManifestRecord> = read_manifest(&manifest_path)
        .expect("manifest.jsonl must read")
        .expect("manifest.jsonl must exist")
        .into_iter()
        .filter(|r| r.class == Class::Llm && r.split == Some(Split::Train))
        .collect();
    records.sort_by(|a, b| a.id.cmp(&b.id));
    records
}

fn main() {
    let corpus_dir = corpus_dir();
    let engine = FixEngine::new().expect("embedded tagger model must load");
    let segmenter = SrxSegmenter::new();
    let tagger = NlpruleTagger::new().expect("embedded tagger model must load");

    let mut per_genre: [GenreScores; 5] = [GenreScores::default(); 5];
    let mut skipped_no_band = 0usize;

    for record in llm_train_records(&corpus_dir) {
        let idx = GENRES
            .iter()
            .position(|g| *g == record.genre)
            .expect("GENRES lists every corpus_tool::manifest::Genre variant");
        let path = corpus_dir.join(relpath(&record));
        let source = fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("{}: failed to read: {e}", path.display()));
        let genre_str = record.genre.to_string();

        let before_doc =
            friction_parse::parse(source.clone()).expect("corpus doc must parse as markdown");
        let before_metrics = friction_metrics::compute(&before_doc, &segmenter, &tagger);

        let (after_text, _report) = engine
            .fix_document(&source, &genre_str)
            .expect("corpus doc must run through fix_document");
        let after_doc =
            friction_parse::parse(after_text.clone()).expect("fix_document output must parse");
        let after_metrics = friction_metrics::compute(&after_doc, &segmenter, &tagger);

        let (Some(before_score), Some(after_score)) = (
            ENVELOPE_V2.combined_score(&genre_str, &before_metrics),
            ENVELOPE_V2.combined_score(&genre_str, &after_metrics),
        ) else {
            skipped_no_band += 1;
            continue;
        };

        per_genre[idx].docs += 1;
        per_genre[idx].before_total += before_score;
        per_genre[idx].after_total += after_score;
    }

    let mut report = String::new();
    writeln!(
        report,
        "genre,docs,mean_combined_score_before,mean_combined_score_after"
    )
    .expect("write to String is infallible");
    for (genre, scores) in GENRES.iter().zip(&per_genre) {
        writeln!(
            report,
            "{genre},{},{:.4},{:.4}",
            scores.docs,
            scores.mean_before(),
            scores.mean_after()
        )
        .expect("write to String is infallible");
    }
    print!("{report}");
    if skipped_no_band > 0 {
        println!("# {skipped_no_band} doc(s) skipped: no combined score for their genre");
    }
}
