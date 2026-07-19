//! Corpus directory layout conventions.
//!
//! ```text
//! <corpus_dir>/manifest.jsonl
//! <corpus_dir>/<class>/<genre>/<id>.md
//! <corpus_dir>/quarantine/<genre>/<id>.md   (CC-BY-SA material)
//! <corpus_dir>/holdout.lock                 (sealed holdout manifest)
//! ```
//!
//! Path resolution is derived entirely from manifest fields — no extra
//! schema field is needed: a record is quarantined exactly when
//! its `license` names CC-BY-SA (`StackExchange` answers in v1).

use crate::manifest::ManifestRecord;

/// True if a record's license marks it as CC-BY-SA quarantined material:
/// measured for the separation report, never redistributed in
/// the shipped pack.
pub fn is_quarantined(record: &ManifestRecord) -> bool {
    record
        .license
        .trim()
        .to_ascii_uppercase()
        .starts_with("CC-BY-SA")
}

/// Path to a record's document, relative to the corpus root.
///
/// Always forward-slash joined regardless of host OS, so manifest- and
/// lock-file text built from it is byte-identical across platforms —
/// callers that need an actual filesystem path still join this
/// onto a `PathBuf` corpus dir, which `Path::join` handles correctly on
/// every supported OS.
pub fn relpath(record: &ManifestRecord) -> String {
    let dir = if is_quarantined(record) {
        "quarantine".to_string()
    } else {
        record.class.to_string()
    };
    format!("{dir}/{}/{}.md", record.genre, record.id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::{Class, Genre};

    fn record(id: &str, class: Class, genre: Genre, license: &str) -> ManifestRecord {
        ManifestRecord {
            id: id.to_string(),
            class,
            genre,
            source: "test".to_string(),
            model: None,
            prompt_id: None,
            license: license.to_string(),
            lang: "en".to_string(),
            split: None,
            sha256: "deadbeef".to_string(),
            provenance_evidence: None,
            style_prompted: false,
            gen_config: None,
        }
    }

    /// A normally licensed doc resolves under `<class>/<genre>/`.
    #[test]
    fn relpath_places_normal_doc_under_class_and_genre() {
        let r = record("h001", Class::Human, Genre::Readme, "MIT");
        assert_eq!(relpath(&r), "human/readme/h001.md");
    }

    /// A CC-BY-SA doc is quarantined under `quarantine/<genre>/`,
    /// regardless of its class.
    #[test]
    fn relpath_quarantines_cc_by_sa_doc_by_genre_only() {
        let r = record("se042", Class::Human, Genre::Forum, "CC-BY-SA");
        assert_eq!(relpath(&r), "quarantine/forum/se042.md");
    }

    /// The CC-BY-SA check is case-insensitive and tolerant of a
    /// trailing version suffix (e.g. "CC-BY-SA-4.0").
    #[test]
    fn is_quarantined_is_case_insensitive_and_matches_version_suffix() {
        assert!(is_quarantined(&record(
            "a",
            Class::Human,
            Genre::Forum,
            "cc-by-sa-4.0"
        )));
        assert!(!is_quarantined(&record(
            "a",
            Class::Human,
            Genre::Forum,
            "MIT"
        )));
        assert!(!is_quarantined(&record(
            "a",
            Class::Human,
            Genre::Forum,
            "CC-BY-4.0"
        )));
    }
}
