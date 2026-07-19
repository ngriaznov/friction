//! `corpus-tool ingest` — folds collector-supplied raw human-corpus
//! candidates (`--incoming <dir>/<genre>/*.md` plus `meta-*.jsonl`
//! metadata fragments in the same directory) into the real corpus.
//!
//! Applies the identical cleaning transform as `clean`, drops docs that
//! fall under 300 words afterward, assigns each survivor a deterministic
//! id, writes it to its layout-correct path (quarantined automatically
//! when the license is CC-BY-SA, per `crate::corpus_layout`), and appends
//! a manifest record. Fragments with missing/empty or unrecognized
//! license fields, missing evidence, an unknown genre, a missing source
//! file, or a metadata collision are refused rather than ingested, and
//! listed in the run summary. Reruns are incremental: a fragment whose
//! derived id is already in the manifest is skipped without touching the
//! filesystem again.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use anyhow::Context;
use clap::Args as ClapArgs;
use serde::Deserialize;

use crate::commands::clean::{MIN_WORDS, normalize, strip_boilerplate};
use crate::commands::generate::parse_genre;
use crate::corpus_layout::relpath;
use crate::hashing::{sha256_hex, word_count};
use crate::manifest::{self, Class, ManifestRecord};

/// Arguments for `corpus-tool ingest`.
#[derive(Debug, ClapArgs)]
pub struct Args {
    /// Directory of incoming human-corpus docs (`<genre>/*.md`) and their
    /// `meta-*.jsonl` metadata fragments.
    #[arg(long, default_value = "corpus/incoming/human")]
    pub incoming: PathBuf,
    /// Corpus root directory.
    #[arg(long, default_value = "corpus")]
    pub corpus_dir: PathBuf,
}

/// One collector-supplied metadata record, one per incoming doc. Required
/// fields absent from a line are simply missing Rust values (not a parse
/// error) so a business-rule refusal can be reported instead of a hard
/// crash on one bad fragment.
#[derive(Debug, Deserialize)]
struct MetaFragment {
    file: String,
    genre: String,
    source: String,
    #[serde(default)]
    license: Option<String>,
    #[serde(default)]
    license_evidence: Option<String>,
    #[serde(default)]
    provenance_evidence: Option<String>,
    #[serde(default)]
    title: Option<String>,
}

/// A fragment refused rather than ingested, with a plain-language reason.
#[derive(Debug, Clone)]
struct Refusal {
    file: String,
    reason: String,
}

/// A doc that was cleaned but fell under the word-count floor.
#[derive(Debug, Clone)]
struct Dropped {
    file: String,
    title: Option<String>,
    words: usize,
}

/// Runs `ingest`.
///
/// # Errors
///
/// Returns an error if `--incoming` can't be listed, a `meta-*.jsonl`
/// fragment file can't be read, a line in one fails to parse as JSON at
/// all (a structural problem distinct from a missing field, which is a
/// refusal instead), or the manifest can't be read or written.
pub fn run(args: &Args) -> anyhow::Result<()> {
    let (fragments, mut refusals) = load_fragments(&args.incoming)?;
    let ids = assign_ids(&fragments);

    let manifest_path = args.corpus_dir.join("manifest.jsonl");
    let mut records = manifest::read_manifest(&manifest_path)?.unwrap_or_default();
    let manifest_ids: BTreeSet<String> = records.iter().map(|r| r.id.clone()).collect();

    let mut dropped = Vec::new();
    let mut ingested = 0usize;
    let mut skipped_existing = 0usize;

    for (fragment, id) in fragments.iter().zip(ids.iter()) {
        if manifest_ids.contains(id) {
            skipped_existing += 1;
            continue;
        }

        let (license, provenance_evidence) = match validate_fragment(fragment) {
            Ok(fields) => fields,
            Err(reason) => {
                refusals.push(Refusal {
                    file: fragment.file.clone(),
                    reason,
                });
                continue;
            }
        };
        let genre = match parse_genre(&fragment.genre) {
            Ok(genre) => genre,
            Err(err) => {
                refusals.push(Refusal {
                    file: fragment.file.clone(),
                    reason: format!("unknown genre: {err}"),
                });
                continue;
            }
        };

        let raw_path = args.incoming.join(&fragment.file);
        let raw = match std::fs::read(&raw_path) {
            Ok(raw) => raw,
            Err(err) => {
                refusals.push(Refusal {
                    file: fragment.file.clone(),
                    reason: format!("source file not found ({err})"),
                });
                continue;
            }
        };

        let cleaned = strip_boilerplate(&normalize(&raw));
        let words = word_count(&cleaned);
        if words < MIN_WORDS {
            dropped.push(Dropped {
                file: fragment.file.clone(),
                title: fragment.title.clone(),
                words,
            });
            continue;
        }

        let record = ManifestRecord {
            id: id.clone(),
            class: Class::Human,
            genre,
            source: fragment.source.clone(),
            model: None,
            prompt_id: None,
            license,
            lang: "en".to_string(),
            split: None,
            sha256: sha256_hex(cleaned.as_bytes()),
            provenance_evidence: Some(provenance_evidence),
            style_prompted: false,
            gen_config: None,
        };

        let out_path = args.corpus_dir.join(relpath(&record));
        if let Some(parent) = out_path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating {}", parent.display()))?;
        }
        std::fs::write(&out_path, &cleaned)
            .with_context(|| format!("writing {}", out_path.display()))?;

        records.push(record);
        manifest::write_manifest(&manifest_path, &records)?;
        ingested += 1;
    }

    refusals.sort_by(|a, b| a.file.cmp(&b.file));
    dropped.sort_by(|a, b| a.file.cmp(&b.file));

    print_summary(ingested, skipped_existing, &dropped, &refusals);

    Ok(())
}

/// Checks the fields required beyond the license/genre/file-presence
/// checks handled inline in `run`: a non-empty, canonically-normalized
/// license, and non-empty license and provenance evidence. Returns
/// `(normalized_license, provenance_evidence)` on success, or a
/// plain-language refusal reason.
fn validate_fragment(fragment: &MetaFragment) -> Result<(String, String), String> {
    let license = non_empty(fragment.license.as_deref()).ok_or("missing license")?;
    // Required to be present, but (unlike `license` and
    // `provenance_evidence`) not itself stored in the manifest.
    non_empty(fragment.license_evidence.as_deref()).ok_or("missing license_evidence")?;
    let provenance_evidence =
        non_empty(fragment.provenance_evidence.as_deref()).ok_or("missing provenance_evidence")?;

    let normalized = normalize_license(license)
        .ok_or_else(|| format!("unrecognized license: {license}"))?
        .to_string();

    Ok((normalized, provenance_evidence.to_string()))
}

fn non_empty(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|s| !s.is_empty())
}

/// Maps collector-supplied license spellings to the canonical set the
/// corpus manifest uses; anything else is rejected (returns `None`).
fn normalize_license(raw: &str) -> Option<&'static str> {
    let key: String = raw
        .trim()
        .to_ascii_lowercase()
        .chars()
        .map(|c| if c == ' ' || c == '_' { '-' } else { c })
        .collect();

    match key.as_str() {
        "mit" => Some("MIT"),
        "apache-2.0" | "apache2.0" | "apache-2" | "apache2" | "apache" => Some("Apache-2.0"),
        "bsd-2-clause" | "bsd2" | "bsd-2" => Some("BSD-2-Clause"),
        "bsd-3-clause" | "bsd3" | "bsd-3" => Some("BSD-3-Clause"),
        "cc-by-4.0" | "ccby4.0" | "cc-by" => Some("CC-BY-4.0"),
        "cc-by-3.0" | "ccby3.0" => Some("CC-BY-3.0"),
        "cc0-1.0" | "cc0" | "cc0-1" => Some("CC0-1.0"),
        "pd" | "public-domain" | "publicdomain" => Some("PD"),
        "cc-by-sa-3.0" | "ccbysa3.0" => Some("CC-BY-SA-3.0"),
        "cc-by-sa-4.0" | "ccbysa4.0" => Some("CC-BY-SA-4.0"),
        _ => None,
    }
}

/// Assigns each of `fragments` its deterministic doc id, one-to-one and in
/// the same order: normally the first 16 hex chars of `sha256(source)`.
///
/// A pure function of the fragment list itself — never of the manifest's
/// current contents — so the same incoming fragment set always gets the
/// same id assignment, whether this is the first run or the tenth,
/// regardless of how much of it has already been ingested (that's decided
/// separately, by checking each assigned id against the manifest).
///
/// Two fragments occasionally share one `source` (e.g. several essays
/// pulled from the same anthology page) — an id collision would either
/// violate the manifest's uniqueness invariant or silently overwrite one
/// doc's file with another's. `fragments` is assumed already sorted by
/// `file` (as `load_fragments` returns it): the first fragment (in that
/// fixed order) among any that share a source keeps the plain
/// `sha256(source)` id; every later one is disambiguated by mixing its
/// own (unique) file path into the hash input, so no two fragments ever
/// collide and the choice never depends on anything but the fragment set
/// and its fixed order.
fn assign_ids(fragments: &[MetaFragment]) -> Vec<String> {
    let mut by_primary: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (index, fragment) in fragments.iter().enumerate() {
        let primary = sha256_hex(fragment.source.as_bytes())[..16].to_string();
        by_primary.entry(primary).or_default().push(index);
    }

    let mut ids = vec![String::new(); fragments.len()];
    for (primary, indices) in by_primary {
        let mut indices = indices.into_iter();
        let first = indices.next().expect("by_primary groups are never empty");
        ids[first] = primary;
        for index in indices {
            let fragment = &fragments[index];
            ids[index] = sha256_hex(format!("{}\0{}", fragment.source, fragment.file).as_bytes())
                [..16]
                .to_string();
        }
    }
    ids
}

/// Reads every `meta-*.jsonl` fragment file directly under `incoming`
/// (sorted by filename), parses each non-blank line, and returns the
/// fragments in deterministic `file`-sorted order alongside refusals for
/// any duplicate `file` reference (kept: first in sort order).
fn load_fragments(incoming: &std::path::Path) -> anyhow::Result<(Vec<MetaFragment>, Vec<Refusal>)> {
    let mut meta_paths: Vec<PathBuf> = std::fs::read_dir(incoming)
        .with_context(|| format!("reading {}", incoming.display()))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            let stem_matches = path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("meta-"));
            stem_matches && path.extension().is_some_and(|ext| ext == "jsonl")
        })
        .collect();
    meta_paths.sort();

    let mut fragments = Vec::new();
    for path in &meta_paths {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        for (idx, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let fragment: MetaFragment = serde_json::from_str(line).with_context(|| {
                format!("{}:{}: invalid metadata fragment", path.display(), idx + 1)
            })?;
            fragments.push(fragment);
        }
    }
    fragments.sort_by(|a, b| a.file.cmp(&b.file));

    let mut refusals = Vec::new();
    let mut seen = BTreeSet::new();
    fragments.retain(|fragment| {
        if seen.insert(fragment.file.clone()) {
            true
        } else {
            refusals.push(Refusal {
                file: fragment.file.clone(),
                reason: "duplicate metadata fragment for this file".to_string(),
            });
            false
        }
    });

    Ok((fragments, refusals))
}

fn print_summary(
    ingested: usize,
    skipped_existing: usize,
    dropped: &[Dropped],
    refusals: &[Refusal],
) {
    println!(
        "ingest: {ingested} ingested, {skipped_existing} already in manifest, \
         {} dropped (under {MIN_WORDS} words), {} refused",
        dropped.len(),
        refusals.len(),
    );
    for doc in dropped {
        let title = doc.title.as_deref().unwrap_or("untitled");
        println!("  dropped {} \"{title}\" ({} words)", doc.file, doc.words);
    }
    for refusal in refusals {
        println!("  refused {}: {}", refusal.file, refusal.reason);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bare_fragment(file: &str, source: &str) -> MetaFragment {
        MetaFragment {
            file: file.to_string(),
            genre: "blog".to_string(),
            source: source.to_string(),
            license: None,
            license_evidence: None,
            provenance_evidence: None,
            title: None,
        }
    }

    /// `assign_ids` matches a hand-computed golden vector — the first 16
    /// hex chars of `sha256(source)` — pinning the exact derivation for a
    /// fragment with no id collision.
    #[test]
    fn assign_ids_matches_known_vector() {
        let fragments = [bare_fragment("blog/one.md", "https://example.com/post")];
        assert_eq!(assign_ids(&fragments), ["061128ccde4ce470"]);
    }

    /// Assigning ids twice for the same fragment list is field-for-field
    /// identical (no ambient state), matching the "stable across reruns"
    /// contract — the property that lets `run` recompute the same id for
    /// an already-ingested fragment and correctly recognize it as such.
    #[test]
    fn assign_ids_is_deterministic() {
        let fragments = [
            bare_fragment("blog/one.md", "https://example.com/a"),
            bare_fragment("blog/two.md", "https://example.com/b"),
        ];
        assert_eq!(assign_ids(&fragments), assign_ids(&fragments));
    }

    /// When two fragments share one `source`, the first (in `file` order)
    /// keeps the plain `sha256(source)` id and the second gets a distinct
    /// id derived by mixing in its own file path, rather than colliding.
    #[test]
    fn assign_ids_disambiguates_shared_source() {
        let fragments = [
            bare_fragment("blog/one.md", "https://example.com/post"),
            bare_fragment("blog/two.md", "https://example.com/post"),
        ];
        let ids = assign_ids(&fragments);
        assert_ne!(ids[0], ids[1]);
        assert_eq!(ids[0], "061128ccde4ce470");
        assert_eq!(ids[1], "55f9a733550c97a0");
    }

    /// Three-way and higher collisions on one `source` still resolve to
    /// all-distinct ids, not just pairwise-distinct.
    #[test]
    fn assign_ids_disambiguates_three_way_collision() {
        let fragments = [
            bare_fragment("blog/one.md", "https://example.com/post"),
            bare_fragment("blog/three.md", "https://example.com/post"),
            bare_fragment("blog/two.md", "https://example.com/post"),
        ];
        let ids = assign_ids(&fragments);
        let unique: BTreeSet<&String> = ids.iter().collect();
        assert_eq!(unique.len(), 3);
    }

    /// The canonical spellings all normalize to themselves.
    #[test]
    fn normalize_license_accepts_canonical_spellings() {
        for license in [
            "MIT",
            "Apache-2.0",
            "BSD-2-Clause",
            "BSD-3-Clause",
            "CC-BY-4.0",
            "CC-BY-3.0",
            "CC0-1.0",
            "PD",
            "CC-BY-SA-3.0",
            "CC-BY-SA-4.0",
        ] {
            assert_eq!(normalize_license(license), Some(license));
        }
    }

    /// Common collector variants (case, spacing) map to the same
    /// canonical spelling as the exact form.
    #[test]
    fn normalize_license_maps_collector_variants() {
        assert_eq!(normalize_license("mit"), Some("MIT"));
        assert_eq!(normalize_license(" apache 2.0 "), Some("Apache-2.0"));
        assert_eq!(normalize_license("cc0"), Some("CC0-1.0"));
        assert_eq!(normalize_license("public domain"), Some("PD"));
        assert_eq!(normalize_license("CC-BY-SA-4.0"), Some("CC-BY-SA-4.0"));
    }

    /// A license outside the canonical set (e.g. one seen in real
    /// collector output that isn't on the approved list) is rejected.
    #[test]
    fn normalize_license_rejects_unknown_license() {
        assert_eq!(normalize_license("ISC"), None);
        assert_eq!(normalize_license("WTFPL"), None);
        assert_eq!(normalize_license(""), None);
    }

    fn fragment(license: &str, license_evidence: &str, provenance: &str) -> MetaFragment {
        MetaFragment {
            file: "docs/a.md".to_string(),
            genre: "docs".to_string(),
            source: "https://example.com/a".to_string(),
            license: Some(license.to_string()),
            license_evidence: Some(license_evidence.to_string()),
            provenance_evidence: Some(provenance.to_string()),
            title: Some("A".to_string()),
        }
    }

    /// A fragment with every required field present and a recognized
    /// license validates and normalizes cleanly.
    #[test]
    fn validate_fragment_accepts_complete_fragment() {
        let f = fragment("mit", "repo LICENSE file", "git commit 2020-01-01");
        let (license, provenance) = validate_fragment(&f).unwrap();
        assert_eq!(license, "MIT");
        assert_eq!(provenance, "git commit 2020-01-01");
    }

    /// Missing (empty-string) license, `license_evidence`, or
    /// `provenance_evidence` is each refused with a distinct, plain reason.
    #[test]
    fn validate_fragment_refuses_missing_required_fields() {
        let mut f = fragment("MIT", "evidence", "evidence");
        f.license = Some(String::new());
        assert_eq!(validate_fragment(&f).unwrap_err(), "missing license");

        let mut f = fragment("MIT", "evidence", "evidence");
        f.license_evidence = Some("   ".to_string());
        assert_eq!(
            validate_fragment(&f).unwrap_err(),
            "missing license_evidence"
        );

        let mut f = fragment("MIT", "evidence", "evidence");
        f.provenance_evidence = None;
        assert_eq!(
            validate_fragment(&f).unwrap_err(),
            "missing provenance_evidence"
        );
    }

    /// A license outside the canonical set is refused with the offending
    /// raw value quoted back, not silently coerced.
    #[test]
    fn validate_fragment_refuses_unrecognized_license() {
        let f = fragment("ISC", "evidence", "evidence");
        assert_eq!(
            validate_fragment(&f).unwrap_err(),
            "unrecognized license: ISC"
        );
    }

    /// `load_fragments` reports a refusal (not a crash) when two
    /// fragments claim the same `file`, keeping the first in
    /// `file`-sorted order.
    #[test]
    fn load_fragments_refuses_duplicate_file_entries() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("meta-a.jsonl"),
            "{\"file\":\"docs/a.md\",\"genre\":\"docs\",\"source\":\"https://x/1\",\"license\":\"MIT\",\"license_evidence\":\"e\",\"provenance_evidence\":\"p\",\"title\":\"A\"}\n\
             {\"file\":\"docs/a.md\",\"genre\":\"docs\",\"source\":\"https://x/2\",\"license\":\"MIT\",\"license_evidence\":\"e\",\"provenance_evidence\":\"p\",\"title\":\"A2\"}\n",
        )
        .unwrap();

        let (fragments, refusals) = load_fragments(dir.path()).unwrap();
        assert_eq!(fragments.len(), 1);
        assert_eq!(fragments[0].source, "https://x/1");
        assert_eq!(refusals.len(), 1);
        assert_eq!(
            refusals[0].reason,
            "duplicate metadata fragment for this file"
        );
    }
}
