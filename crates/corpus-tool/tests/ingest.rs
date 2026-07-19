//! Integration tests for `corpus-tool ingest`.

mod common;

use corpus_tool::commands::ingest::{self, Args};
use corpus_tool::manifest::{self, Class, Split};

fn args(incoming: &std::path::Path, corpus_dir: &std::path::Path) -> Args {
    Args {
        incoming: incoming.to_path_buf(),
        corpus_dir: corpus_dir.to_path_buf(),
    }
}

fn write_fragment(incoming: &std::path::Path, meta_file: &str, line: &str) {
    use std::io::Write as _;
    let path = incoming.join(meta_file);
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .unwrap();
    writeln!(file, "{line}").unwrap();
}

fn fragment_json(file: &str, genre: &str, source: &str, license: &str) -> String {
    format!(
        "{{\"file\":\"{file}\",\"genre\":\"{genre}\",\"source\":\"{source}\",\
         \"license\":\"{license}\",\"license_evidence\":\"evidence\",\
         \"provenance_evidence\":\"provenance\",\"title\":\"Title\"}}"
    )
}

/// A well-formed fragment with a normal (non-CC-BY-SA) license is cleaned,
/// written under `human/<genre>/`, and gets a full manifest record.
#[test]
fn ingest_writes_normal_license_doc_under_human() {
    let dir = tempfile::tempdir().unwrap();
    let incoming = dir.path().join("incoming");
    let corpus_dir = dir.path().join("corpus");
    let body = common::filler_words(320);
    common::write_doc(&incoming, "docs/a.md", &format!("# A\n\n{body}\n"));
    write_fragment(
        &incoming,
        "meta-a.jsonl",
        &fragment_json("docs/a.md", "docs", "https://example.com/a", "MIT"),
    );

    ingest::run(&args(&incoming, &corpus_dir)).unwrap();

    let records = manifest::read_manifest(&corpus_dir.join("manifest.jsonl"))
        .unwrap()
        .unwrap();
    assert_eq!(records.len(), 1);
    let record = &records[0];
    assert_eq!(record.class, Class::Human);
    assert_eq!(record.license, "MIT");
    assert_eq!(record.lang, "en");
    assert_eq!(record.split, None);
    assert!(!record.style_prompted);
    assert!(record.model.is_none());
    assert!(record.gen_config.is_none());
    assert_eq!(record.provenance_evidence.as_deref(), Some("provenance"));
    assert!(
        corpus_dir
            .join(format!("human/docs/{}.md", record.id))
            .exists()
    );
}

/// A CC-BY-SA-licensed doc is written under `quarantine/<genre>/`, not
/// `human/<genre>/`, even though its manifest `class` still says `human`.
#[test]
fn ingest_quarantines_cc_by_sa_doc() {
    let dir = tempfile::tempdir().unwrap();
    let incoming = dir.path().join("incoming");
    let corpus_dir = dir.path().join("corpus");
    let body = common::filler_words(320);
    common::write_doc(&incoming, "forum/q.md", &format!("# Q\n\n{body}\n"));
    write_fragment(
        &incoming,
        "meta-forum.jsonl",
        &fragment_json(
            "forum/q.md",
            "forum",
            "https://stackoverflow.com/q/1",
            "CC-BY-SA-4.0",
        ),
    );

    ingest::run(&args(&incoming, &corpus_dir)).unwrap();

    let records = manifest::read_manifest(&corpus_dir.join("manifest.jsonl"))
        .unwrap()
        .unwrap();
    assert_eq!(records.len(), 1);
    let record = &records[0];
    assert_eq!(record.class, Class::Human);
    assert!(
        !corpus_dir
            .join(format!("human/forum/{}.md", record.id))
            .exists()
    );
    assert!(
        corpus_dir
            .join(format!("quarantine/forum/{}.md", record.id))
            .exists()
    );
}

/// A fragment whose cleaned doc falls under 300 words is dropped: no file
/// is written and no manifest record is created.
#[test]
fn ingest_drops_docs_under_300_words() {
    let dir = tempfile::tempdir().unwrap();
    let incoming = dir.path().join("incoming");
    let corpus_dir = dir.path().join("corpus");
    let short_body = common::filler_words(50);
    common::write_doc(
        &incoming,
        "blog/short.md",
        &format!("# Short\n\n{short_body}\n"),
    );
    write_fragment(
        &incoming,
        "meta-blog.jsonl",
        &fragment_json("blog/short.md", "blog", "https://example.com/short", "MIT"),
    );

    ingest::run(&args(&incoming, &corpus_dir)).unwrap();

    let records = manifest::read_manifest(&corpus_dir.join("manifest.jsonl")).unwrap();
    assert!(records.unwrap_or_default().is_empty());
}

/// A fragment with a missing/empty license, `license_evidence`, or
/// `provenance_evidence` is refused: no doc is ingested for it.
#[test]
fn ingest_refuses_fragments_missing_required_evidence() {
    let dir = tempfile::tempdir().unwrap();
    let incoming = dir.path().join("incoming");
    let corpus_dir = dir.path().join("corpus");
    let body = common::filler_words(320);
    common::write_doc(&incoming, "email/e.md", &format!("# E\n\n{body}\n"));
    write_fragment(
        &incoming,
        "meta-email.jsonl",
        "{\"file\":\"email/e.md\",\"genre\":\"email\",\"source\":\"https://example.com/e\",\
         \"license\":\"MIT\",\"license_evidence\":\"\",\"provenance_evidence\":\"p\",\"title\":\"E\"}",
    );

    ingest::run(&args(&incoming, &corpus_dir)).unwrap();

    let records = manifest::read_manifest(&corpus_dir.join("manifest.jsonl")).unwrap();
    assert!(records.unwrap_or_default().is_empty());
}

/// A license outside the canonical set is refused rather than ingested
/// under an unnormalized spelling.
#[test]
fn ingest_refuses_unrecognized_license() {
    let dir = tempfile::tempdir().unwrap();
    let incoming = dir.path().join("incoming");
    let corpus_dir = dir.path().join("corpus");
    let body = common::filler_words(320);
    common::write_doc(&incoming, "readme/r.md", &format!("# R\n\n{body}\n"));
    write_fragment(
        &incoming,
        "meta-readme.jsonl",
        &fragment_json("readme/r.md", "readme", "https://example.com/r", "ISC"),
    );

    ingest::run(&args(&incoming, &corpus_dir)).unwrap();

    let records = manifest::read_manifest(&corpus_dir.join("manifest.jsonl")).unwrap();
    assert!(records.unwrap_or_default().is_empty());
}

/// Running `ingest` twice over the same incoming directory is
/// incremental: the second run adds nothing new (same manifest, same doc
/// count), since every fragment's derived id is already present.
#[test]
fn ingest_is_incremental_across_reruns() {
    let dir = tempfile::tempdir().unwrap();
    let incoming = dir.path().join("incoming");
    let corpus_dir = dir.path().join("corpus");
    let body = common::filler_words(320);
    common::write_doc(&incoming, "docs/a.md", &format!("# A\n\n{body}\n"));
    write_fragment(
        &incoming,
        "meta-a.jsonl",
        &fragment_json("docs/a.md", "docs", "https://example.com/a", "MIT"),
    );

    ingest::run(&args(&incoming, &corpus_dir)).unwrap();
    let first = manifest::read_manifest(&corpus_dir.join("manifest.jsonl"))
        .unwrap()
        .unwrap();

    ingest::run(&args(&incoming, &corpus_dir)).unwrap();
    let second = manifest::read_manifest(&corpus_dir.join("manifest.jsonl"))
        .unwrap()
        .unwrap();

    assert_eq!(first, second);
    assert_eq!(second.len(), 1);
}

/// Two fragments that share one `source` URL (e.g. two essays from the
/// same anthology page) both get ingested under distinct, stable ids
/// rather than colliding.
#[test]
fn ingest_disambiguates_fragments_sharing_one_source() {
    let dir = tempfile::tempdir().unwrap();
    let incoming = dir.path().join("incoming");
    let corpus_dir = dir.path().join("corpus");
    let body = common::filler_words(320);
    common::write_doc(&incoming, "blog/one.md", &format!("# One\n\n{body}\n"));
    common::write_doc(&incoming, "blog/two.md", &format!("# Two\n\n{body}\n"));
    write_fragment(
        &incoming,
        "meta-blog.jsonl",
        &fragment_json("blog/one.md", "blog", "https://example.com/anthology", "PD"),
    );
    write_fragment(
        &incoming,
        "meta-blog.jsonl",
        &fragment_json("blog/two.md", "blog", "https://example.com/anthology", "PD"),
    );

    ingest::run(&args(&incoming, &corpus_dir)).unwrap();

    let records = manifest::read_manifest(&corpus_dir.join("manifest.jsonl"))
        .unwrap()
        .unwrap();
    assert_eq!(records.len(), 2);
    assert_ne!(records[0].id, records[1].id);
}

/// `split` remains `null` for freshly ingested docs — split assignment is
/// a separate, later step (`corpus-tool split`), never done by `ingest`.
#[test]
fn ingest_leaves_split_unassigned() {
    let dir = tempfile::tempdir().unwrap();
    let incoming = dir.path().join("incoming");
    let corpus_dir = dir.path().join("corpus");
    let body = common::filler_words(320);
    common::write_doc(&incoming, "docs/a.md", &format!("# A\n\n{body}\n"));
    write_fragment(
        &incoming,
        "meta-a.jsonl",
        &fragment_json("docs/a.md", "docs", "https://example.com/a", "MIT"),
    );

    ingest::run(&args(&incoming, &corpus_dir)).unwrap();

    let records = manifest::read_manifest(&corpus_dir.join("manifest.jsonl"))
        .unwrap()
        .unwrap();
    assert_eq!(records[0].split, Option::<Split>::None);
}
