//! Integration tests for `corpus-tool mine-paired`.
//!
//! Entirely hermetic: every corpus here (both the synthetic paired
//! corpus and the synthetic `corpus/human` train docs used for the
//! cross-check column) is built fresh in a tempdir — never the real
//! `corpus/` directory, never `corpus-tool generate-paired`, never
//! Ollama.

mod common;

use corpus_tool::commands::mine_paired::{self, Args};
use corpus_tool::manifest::{Genre, ModelInfo, Split};
use corpus_tool::paired_manifest::{self, PairedManifestRecord, Side};

fn args(
    corpus_dir: &std::path::Path,
    paired_dir: &std::path::Path,
    report: std::path::PathBuf,
) -> Args {
    Args {
        corpus_dir: corpus_dir.to_path_buf(),
        paired_dir: paired_dir.to_path_buf(),
        report,
        eps: 0.4,
        min_antislop_token_freq: 8,
        min_stock_count_n2: 8,
        min_stock_count_n3: 5,
        min_stock_count_n4: 4,
        top: 50,
    }
}

fn paired_record(pair_id: &str, side: Side, genre: Genre, sha256: String) -> PairedManifestRecord {
    PairedManifestRecord {
        pair_id: pair_id.to_string(),
        side,
        genre,
        model: ModelInfo {
            name: "test-model".to_string(),
            quantization: "q4_k_m".to_string(),
        },
        model_digest: "sha256:deadbeef".to_string(),
        prompt_id: "p001".to_string(),
        seed: 1,
        temperature: 0.7,
        num_predict: 512,
        reproducible: true,
        sha256,
    }
}

fn write_paired_manifest_raw(path: &std::path::Path, records: &[PairedManifestRecord]) {
    let lines: Vec<String> = records
        .iter()
        .map(|r| serde_json::to_string(r).expect("serialize fixture paired record"))
        .collect();
    std::fs::write(path, lines.join("\n") + "\n").expect("write fixture paired manifest");
}

/// A block of filler text repeating five words in separate sentences (so
/// they never form a 3-gram with each other), used to push each word's
/// antislop-side frequency comfortably above the default gate of 8, on
/// the antislop side of one pair.
fn antislop_filler_block() -> String {
    "Zeta is fine. Widget is fine. Cascade is fine. Rare is fine. Token is fine. \n".repeat(9)
}

/// A block planting two literal 3-grams on the stock side: `"zeta widget
/// cascade"` (whose every token also appears in the antislop filler, so
/// it clears the per-token frequency gate) and `"rare token combo"`
/// (whose third token, `"combo"`, never appears on the antislop side at
/// all, so the whole 3-gram must be excluded by the frequency gate
/// regardless of its stock-side count).
fn stock_plant_block() -> String {
    "Zeta widget cascade happened again. Rare token combo occurred often. \n".repeat(10)
}

/// Positive case: a 3-gram appearing 10 times in the synthetic stock doc
/// and 0 times in the synthetic antislop doc, with every constituent
/// token's antislop-side frequency pushed above the default gate (9 >=
/// 8), surfaces as the top stock-favored 3-gram with a score above 1.0.
///
/// Negative case, in the same fixture: a 3-gram (`"rare token combo"`)
/// that clears the stock-count threshold just as easily is correctly
/// excluded from every table because one of its tokens (`"combo"`) never
/// reaches the antislop-frequency gate.
#[test]
fn mine_paired_surfaces_planted_trigram_stock_favored_and_excludes_gated_one() {
    let dir = tempfile::tempdir().unwrap();
    let paired_dir = dir.path().join("paired");

    let stock_text = stock_plant_block();
    let antislop_text = antislop_filler_block();
    let sha_stock = common::write_doc(&paired_dir, "stock/docs/pair1.md", &stock_text);
    let sha_antislop = common::write_doc(&paired_dir, "antislop/docs/pair1.md", &antislop_text);

    let stock_rec = paired_record("pair1", Side::Stock, Genre::Docs, sha_stock);
    let antislop_rec = paired_record("pair1", Side::Antislop, Genre::Docs, sha_antislop);
    write_paired_manifest_raw(
        &paired_dir.join("manifest.jsonl"),
        &[stock_rec, antislop_rec],
    );

    let report_path = dir.path().join("report.md");
    mine_paired::run(&args(dir.path(), &paired_dir, report_path.clone())).unwrap();
    let report = std::fs::read_to_string(&report_path).unwrap();

    let section_start = report
        .find("## 3-gram")
        .expect("report must have a 3-gram section");
    let section_end = report[section_start..]
        .find("## 4-gram")
        .map_or(report.len(), |offset| section_start + offset);
    let section = &report[section_start..section_end];
    let stock_favored_start = section
        .find("### stock-favored")
        .expect("3-gram section must have a stock-favored subsection");
    let antislop_favored_start = section
        .find("### antislop-favored")
        .expect("3-gram section must have an antislop-favored subsection");
    let stock_favored_table = &section[stock_favored_start..antislop_favored_start];

    assert!(
        stock_favored_table.contains("zeta widget cascade"),
        "expected the planted trigram in the 3-gram stock-favored table:\n{stock_favored_table}"
    );
    assert!(
        !stock_favored_table.contains("rare token combo"),
        "the gated trigram (token \"combo\" never appears in antislop) must not appear:\n\
         {stock_favored_table}"
    );
    assert!(
        !section.contains("rare token combo"),
        "the gated trigram must not appear anywhere in the 3-gram section (either direction)"
    );

    let row_line = stock_favored_table
        .lines()
        .find(|line| line.contains("zeta widget cascade"))
        .expect("planted trigram row must be present");
    let score_field = row_line
        .trim_start_matches('|')
        .split('|')
        .nth(3)
        .expect("row has a score column")
        .trim();
    let score: f64 = score_field.parse().expect("score column must parse as f64");
    assert!(
        score > 1.0,
        "expected score > 1.0, got {score} (row: {row_line})"
    );
}

/// A candidate whose tokens are one occurrence short of the default
/// `--min-antislop-token-freq` gate (7 < 8) is excluded from the report
/// even though it clears the stock-count threshold easily.
#[test]
fn mine_paired_respects_min_antislop_token_freq_gate() {
    let dir = tempfile::tempdir().unwrap();
    let paired_dir = dir.path().join("paired");

    // "alpha"/"beta" occur 7 times each on the antislop side — one short
    // of the default gate of 8.
    let antislop_text = "Alpha beta filler sentence here. \n".repeat(7);
    let stock_text = "Alpha beta combo appears often here. \n".repeat(10);

    let sha_stock = common::write_doc(&paired_dir, "stock/docs/pair1.md", &stock_text);
    let sha_antislop = common::write_doc(&paired_dir, "antislop/docs/pair1.md", &antislop_text);

    let stock_rec = paired_record("pair1", Side::Stock, Genre::Docs, sha_stock);
    let antislop_rec = paired_record("pair1", Side::Antislop, Genre::Docs, sha_antislop);
    write_paired_manifest_raw(
        &paired_dir.join("manifest.jsonl"),
        &[stock_rec, antislop_rec],
    );

    let report_path = dir.path().join("report.md");
    mine_paired::run(&args(dir.path(), &paired_dir, report_path.clone())).unwrap();
    let report = std::fs::read_to_string(&report_path).unwrap();

    let section_start = report
        .find("## 2-gram")
        .expect("report must have a 2-gram section");
    let section_end = report[section_start..]
        .find("## 3-gram")
        .map_or(report.len(), |offset| section_start + offset);
    let section = &report[section_start..section_end];
    assert!(
        !section.contains("alpha beta"),
        "expected \"alpha beta\" to be gated out (antislop freq 7 < 8):\n{section}"
    );
}

/// The `human_train_count` column reflects a read-only cross-check
/// against `corpus/human` train-split docs only: a surfaced n-gram
/// planted in a train-split human doc shows up with a nonzero count, but
/// the same marker planted in a dev-split human doc never leaks in —
/// mirrors `mine_inventory_respects_train_only_data_protocol`.
#[test]
fn mine_paired_report_includes_human_train_cross_check_column() {
    let dir = tempfile::tempdir().unwrap();
    let paired_dir = dir.path().join("paired");

    let stock_text = stock_plant_block();
    let antislop_text = antislop_filler_block();
    let sha_stock = common::write_doc(&paired_dir, "stock/docs/pair1.md", &stock_text);
    let sha_antislop = common::write_doc(&paired_dir, "antislop/docs/pair1.md", &antislop_text);
    let stock_rec = paired_record("pair1", Side::Stock, Genre::Docs, sha_stock);
    let antislop_rec = paired_record("pair1", Side::Antislop, Genre::Docs, sha_antislop);
    write_paired_manifest_raw(
        &paired_dir.join("manifest.jsonl"),
        &[stock_rec, antislop_rec],
    );

    // A train-split human doc plants the surfaced trigram 4 times.
    let train_text = "Zeta widget cascade is a well documented term. \n".repeat(4);
    let sha_train = common::write_doc(dir.path(), "human/docs/h001.md", &train_text);
    let mut train_record = common::human_record("h001", Genre::Docs, sha_train);
    train_record.split = Some(Split::Train);

    // A dev-split human doc plants a unique marker that must never leak
    // into the report.
    let dev_text = "Devonly marker phrase repeated here. \n".repeat(10);
    let sha_dev = common::write_doc(dir.path(), "human/docs/h002.md", &dev_text);
    let mut dev_record = common::human_record("h002", Genre::Docs, sha_dev);
    dev_record.split = Some(Split::Dev);

    common::write_manifest_raw(
        &dir.path().join("manifest.jsonl"),
        &[train_record, dev_record],
    );

    let report_path = dir.path().join("report.md");
    mine_paired::run(&args(dir.path(), &paired_dir, report_path.clone())).unwrap();
    let report = std::fs::read_to_string(&report_path).unwrap();

    assert!(
        !report.to_lowercase().contains("devonly"),
        "a dev-split human doc's marker word must never appear in the report"
    );

    let row_line = report
        .lines()
        .find(|line| line.contains("zeta widget cascade"))
        .expect("planted trigram row must be present");
    // Columns: | n-gram | stock count | antislop count | score | human_train_count |
    let human_train_count_field = row_line
        .trim_start_matches('|')
        .split('|')
        .nth(4)
        .expect("row has a human_train_count column")
        .trim();
    assert_eq!(
        human_train_count_field, "4",
        "expected human_train_count=4 for the planted trigram: {row_line}"
    );
}

/// Every antislop-favored section carries the standing banner stating
/// that its entries are diagnostic only, never a substitution-pair
/// replacement source.
#[test]
fn mine_paired_antislop_favored_section_carries_the_never_a_replacement_source_banner() {
    let dir = tempfile::tempdir().unwrap();
    let paired_dir = dir.path().join("paired");
    let report_path = dir.path().join("report.md");
    mine_paired::run(&args(dir.path(), &paired_dir, report_path.clone())).unwrap();
    let report = std::fs::read_to_string(&report_path).unwrap();

    assert!(report.contains("### antislop-favored"));
    assert!(report.contains("never a source of substitution-pair replacement text"));
    assert!(report.contains("must be hand-picked from corpus/human train docs"));
}

/// Running `mine-paired` twice against the same corpus produces
/// byte-identical report markdown.
#[test]
fn mine_paired_report_is_deterministic() {
    let dir = tempfile::tempdir().unwrap();
    let paired_dir = dir.path().join("paired");
    let stock_text = stock_plant_block();
    let antislop_text = antislop_filler_block();
    let sha_stock = common::write_doc(&paired_dir, "stock/docs/pair1.md", &stock_text);
    let sha_antislop = common::write_doc(&paired_dir, "antislop/docs/pair1.md", &antislop_text);
    let stock_rec = paired_record("pair1", Side::Stock, Genre::Docs, sha_stock);
    let antislop_rec = paired_record("pair1", Side::Antislop, Genre::Docs, sha_antislop);
    write_paired_manifest_raw(
        &paired_dir.join("manifest.jsonl"),
        &[stock_rec, antislop_rec],
    );

    let report_a = dir.path().join("a.md");
    let report_b = dir.path().join("b.md");
    mine_paired::run(&args(dir.path(), &paired_dir, report_a.clone())).unwrap();
    mine_paired::run(&args(dir.path(), &paired_dir, report_b.clone())).unwrap();

    let text_a = std::fs::read_to_string(&report_a).unwrap();
    let text_b = std::fs::read_to_string(&report_b).unwrap();
    assert_eq!(text_a, text_b);
}

/// An empty paired corpus (no `corpus/paired/manifest.jsonl` at all)
/// still writes a well-formed header-only report rather than erroring.
#[test]
fn mine_paired_empty_paired_corpus_writes_header_only_report() {
    let dir = tempfile::tempdir().unwrap();
    let paired_dir = dir.path().join("paired");
    let report_path = dir.path().join("report.md");
    mine_paired::run(&args(dir.path(), &paired_dir, report_path.clone())).unwrap();
    let report = std::fs::read_to_string(&report_path).unwrap();

    assert!(report.contains("# Paired stock/antislop mining report"));
    assert!(report.contains("0 stock doc(s), 0 antislop doc(s)"));
    assert!(report.contains("## 2-gram"));
    assert!(report.contains("## 3-gram"));
    assert!(report.contains("## 4-gram"));
}

/// Unused-but-imported: `paired_manifest` re-export sanity — reading
/// back a manifest this test file wrote round-trips through the crate's
/// own reader (guards against a schema mismatch between the fixture
/// helpers here and `PairedManifestRecord`).
#[test]
fn fixture_paired_manifest_round_trips_through_the_crate_reader() {
    let dir = tempfile::tempdir().unwrap();
    let paired_dir = dir.path().join("paired");
    std::fs::create_dir_all(&paired_dir).unwrap();
    let record = paired_record("pair1", Side::Stock, Genre::Docs, "deadbeef".to_string());
    write_paired_manifest_raw(
        &paired_dir.join("manifest.jsonl"),
        std::slice::from_ref(&record),
    );

    let read_back = paired_manifest::read_paired_manifest(&paired_dir.join("manifest.jsonl"))
        .unwrap()
        .unwrap();
    assert_eq!(read_back, vec![record]);
}
