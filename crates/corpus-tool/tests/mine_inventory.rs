//! Integration tests for `corpus-tool mine-inventory`.
//!
//! Entirely hermetic: every corpus here is a small synthetic fixture
//! built in a tempdir, never the real `corpus/` directory.

mod common;

use corpus_tool::commands::mine_inventory::{self, Args};
use corpus_tool::manifest::{Genre, Split};

fn args(corpus_dir: &std::path::Path, report: std::path::PathBuf) -> Args {
    Args {
        corpus_dir: corpus_dir.to_path_buf(),
        report,
        eps: 0.4,
        min_human_token_freq: 25,
        min_machine_count_n2: 8,
        min_machine_count_n3: 5,
        min_machine_count_n4: 4,
        top: 50,
        min_lvc_count: 2,
        skeleton: false,
    }
}

/// A block of filler text repeating five words in separate sentences (so
/// they never form a 3-gram with each other), used to push each word's
/// human-corpus frequency comfortably above the default gate of 25.
fn human_filler_block() -> String {
    "Zeta is fine. Widget is fine. Cascade is fine. Rare is fine. Token is fine. \n".repeat(26)
}

/// A block planting two literal 3-grams: `"zeta widget cascade"` (whose
/// every token also appears in the human filler, so it clears the
/// per-token frequency gate) and `"rare token combo"` (whose third token,
/// `"combo"`, never appears in the human corpus at all, so the whole
/// 3-gram must be excluded by the frequency gate regardless of its
/// machine-side count).
fn machine_plant_block() -> String {
    "Zeta widget cascade happened again. Rare token combo occurred often. \n".repeat(10)
}

/// Positive case: a 3-gram appearing 10 times in the synthetic machine
/// docs and 0 times in the synthetic human docs, with every constituent
/// token's human-corpus frequency pushed above the default gate (26 >=
/// 25), surfaces as the top machine-favored 3-gram with a score above
/// 1.0.
///
/// Negative case, in the same fixture: a 3-gram (`"rare token combo"`)
/// that clears the machine-count threshold just as easily is correctly
/// excluded from every table because one of its tokens (`"combo"`) never
/// reaches the human-frequency gate (it has zero human-corpus
/// occurrences at all).
#[test]
fn mine_inventory_surfaces_planted_trigram_and_excludes_gated_one() {
    let dir = tempfile::tempdir().unwrap();

    let human_text = human_filler_block();
    let machine_text = machine_plant_block();
    let sha_h = common::write_doc(dir.path(), "human/docs/h001.md", &human_text);
    let sha_l = common::write_doc(dir.path(), "llm/docs/l001.md", &machine_text);

    let mut h = common::human_record("h001", Genre::Docs, sha_h);
    h.split = Some(Split::Train);
    let mut l = common::llm_record("l001", Genre::Docs, sha_l);
    l.split = Some(Split::Train);
    common::write_manifest_raw(&dir.path().join("manifest.jsonl"), &[h, l]);

    let report_path = dir.path().join("report.md");
    mine_inventory::run(&args(dir.path(), report_path.clone())).unwrap();
    let report = std::fs::read_to_string(&report_path).unwrap();

    // Extract the 3-gram machine-favored table's rows and find the
    // planted phrase's row.
    let section_start = report
        .find("### 3-gram")
        .expect("report must have a 3-gram section");
    let section_end = report[section_start..]
        .find("### 4-gram")
        .map_or(report.len(), |offset| section_start + offset);
    let section = &report[section_start..section_end];
    let machine_favored_start = section
        .find("#### machine-favored")
        .expect("3-gram section must have a machine-favored subsection");
    let human_favored_start = section
        .find("#### human-favored")
        .expect("3-gram section must have a human-favored subsection");
    let machine_favored_table = &section[machine_favored_start..human_favored_start];

    assert!(
        machine_favored_table.contains("zeta widget cascade"),
        "expected the planted trigram in the 3-gram machine-favored table:\n{machine_favored_table}"
    );
    assert!(
        !machine_favored_table.contains("rare token combo"),
        "the gated trigram (token \"combo\" has zero human-corpus frequency) must not appear:\n\
         {machine_favored_table}"
    );
    assert!(
        !section.contains("rare token combo"),
        "the gated trigram must not appear anywhere in the 3-gram section (either direction)"
    );

    // The planted row's score must be above 1.0 (machine-favored).
    let row_line = machine_favored_table
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

/// `mine-inventory` reads `train`-split records only: a `dev`-split and a
/// `holdout`-split human doc (each with its own unique, otherwise-
/// unused marker word repeated enough to clear the frequency gate) never
/// contribute to the report, and no `holdout.lock` file is created or
/// needed for this to hold — this command has no reason to ever look
/// for one.
#[test]
fn mine_inventory_respects_train_only_data_protocol() {
    let dir = tempfile::tempdir().unwrap();

    let train_text = "Trainonly word appears here. \n".repeat(30);
    let dev_text = "Devonly word appears here. \n".repeat(30);
    let holdout_text = "Holdoutonly word appears here. \n".repeat(30);

    let sha_train = common::write_doc(dir.path(), "human/docs/h001.md", &train_text);
    let sha_dev = common::write_doc(dir.path(), "human/docs/h002.md", &dev_text);
    let sha_holdout = common::write_doc(dir.path(), "human/docs/h003.md", &holdout_text);

    let mut train = common::human_record("h001", Genre::Docs, sha_train);
    train.split = Some(Split::Train);
    let mut dev = common::human_record("h002", Genre::Docs, sha_dev);
    dev.split = Some(Split::Dev);
    let mut holdout = common::human_record("h003", Genre::Docs, sha_holdout);
    holdout.split = Some(Split::Holdout);
    common::write_manifest_raw(&dir.path().join("manifest.jsonl"), &[train, dev, holdout]);

    assert!(
        !dir.path().join("holdout.lock").exists(),
        "test fixture deliberately has no holdout.lock"
    );

    let report_path = dir.path().join("report.md");
    mine_inventory::run(&args(dir.path(), report_path.clone())).unwrap();
    let report = std::fs::read_to_string(&report_path).unwrap();

    assert!(
        report.contains("1 document(s) (1 human, 0 llm)"),
        "expected exactly the one train-split doc to be counted:\n{report}"
    );
    assert!(
        !report.to_lowercase().contains("devonly"),
        "a dev-split doc's marker word must never appear in the report"
    );
    assert!(
        !report.to_lowercase().contains("holdoutonly"),
        "a holdout-split doc's marker word must never appear in the report"
    );
    assert!(!dir.path().join("holdout.lock").exists());
}

/// Running `mine-inventory` twice against the same corpus produces
/// byte-identical report markdown.
#[test]
fn mine_inventory_report_is_deterministic() {
    let dir = tempfile::tempdir().unwrap();
    let human_text = human_filler_block();
    let machine_text = machine_plant_block();
    let sha_h = common::write_doc(dir.path(), "human/docs/h001.md", &human_text);
    let sha_l = common::write_doc(dir.path(), "llm/docs/l001.md", &machine_text);
    let mut h = common::human_record("h001", Genre::Docs, sha_h);
    h.split = Some(Split::Train);
    let mut l = common::llm_record("l001", Genre::Docs, sha_l);
    l.split = Some(Split::Train);
    common::write_manifest_raw(&dir.path().join("manifest.jsonl"), &[h, l]);

    let report_a = dir.path().join("a.md");
    let report_b = dir.path().join("b.md");
    mine_inventory::run(&args(dir.path(), report_a.clone())).unwrap();
    mine_inventory::run(&args(dir.path(), report_b.clone())).unwrap();

    let text_a = std::fs::read_to_string(&report_a).unwrap();
    let text_b = std::fs::read_to_string(&report_b).unwrap();
    assert_eq!(text_a, text_b);
}

/// An empty corpus still writes a well-formed report rather than
/// erroring.
#[test]
fn mine_inventory_empty_corpus_writes_header_only_report() {
    let dir = tempfile::tempdir().unwrap();
    let report_path = dir.path().join("report.md");
    mine_inventory::run(&args(dir.path(), report_path.clone())).unwrap();
    let report = std::fs::read_to_string(&report_path).unwrap();
    assert!(report.contains("# Inventory mining report"));
    assert!(report.contains("0 document(s) (0 human, 0 llm)"));
}

/// The seed light-verb-construction table reports all 31
/// `DERIVATIONAL_LEXICON` entries (a real corpus sentence exercising one
/// of them shows up with a nonzero `constr_M` count), and the
/// new-candidate discovery section is present, even on a corpus too
/// small to discover anything.
#[test]
fn mine_inventory_lvc_section_reports_seed_pairs_and_candidate_section() {
    let dir = tempfile::tempdir().unwrap();
    let human_text = human_filler_block();
    let machine_text = "The system performs an initialization of the database. \n".repeat(3);
    let sha_h = common::write_doc(dir.path(), "human/docs/h001.md", &human_text);
    let sha_l = common::write_doc(dir.path(), "llm/docs/l001.md", &machine_text);
    let mut h = common::human_record("h001", Genre::Docs, sha_h);
    h.split = Some(Split::Train);
    let mut l = common::llm_record("l001", Genre::Docs, sha_l);
    l.split = Some(Split::Train);
    common::write_manifest_raw(&dir.path().join("manifest.jsonl"), &[h, l]);

    let report_path = dir.path().join("report.md");
    mine_inventory::run(&args(dir.path(), report_path.clone())).unwrap();
    let report = std::fs::read_to_string(&report_path).unwrap();

    assert!(report.contains("## Light-verb-construction pairs"));
    assert!(report.contains("### Seed pairs"));
    assert!(report.contains("### New-candidate discovery"));
    // "initialization" is a seed nominalization and the machine text
    // plants exactly the licensed construction shape three times.
    let seed_section_start = report.find("### Seed pairs").unwrap();
    let seed_section_end = report[seed_section_start..]
        .find("### New-candidate discovery")
        .map_or(report.len(), |o| seed_section_start + o);
    let seed_section = &report[seed_section_start..seed_section_end];
    let row = seed_section
        .lines()
        .find(|line| line.contains("| initialization |"))
        .expect("initialization row must be present");
    assert!(
        row.contains("| initialization | initialize | 3 |"),
        "expected constr_M=3 for the planted construction: {row}"
    );
}

/// The POS-skeleton mining pass runs end to end (`--skeleton true`,
/// the default) without crashing, and its section header appears.
#[test]
fn mine_inventory_skeleton_pass_runs_and_reports_a_section() {
    let dir = tempfile::tempdir().unwrap();
    let human_text = human_filler_block();
    let machine_text = machine_plant_block();
    let sha_h = common::write_doc(dir.path(), "human/docs/h001.md", &human_text);
    let sha_l = common::write_doc(dir.path(), "llm/docs/l001.md", &machine_text);
    let mut h = common::human_record("h001", Genre::Docs, sha_h);
    h.split = Some(Split::Train);
    let mut l = common::llm_record("l001", Genre::Docs, sha_l);
    l.split = Some(Split::Train);
    common::write_manifest_raw(&dir.path().join("manifest.jsonl"), &[h, l]);

    let mut with_skeleton = args(dir.path(), dir.path().join("report.md"));
    with_skeleton.skeleton = true;
    mine_inventory::run(&with_skeleton).unwrap();
    let report = std::fs::read_to_string(&with_skeleton.report).unwrap();
    assert!(report.contains("## POS-skeleton mining"));
}
