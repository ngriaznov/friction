//! SARIF 2.1.0 output for `friction check --format sarif`.
//!
//! Builds a minimal, schema-valid SARIF log: one `tool.driver.rules[]`
//! entry per distinct [`RuleId`] actually present among a run's findings
//! (with a plain-language description — see [`rule_description`]), and one
//! `results[]` entry per finding, with a `physicalLocation.region` derived
//! from the finding's byte range via [`crate::common::offset_to_line_col`].
//!
//! `tests/sarif_schema.rs` validates this module's output against the
//! vendored SARIF 2.1.0 JSON schema (`tests/data/sarif-schema-2.1.0.json`)
//! with the `jsonschema` crate.

use std::collections::BTreeSet;

use friction_core::{Finding, Tier};
use serde::Serialize;

use crate::common::offset_to_line_col;

/// The SARIF schema URI this log declares conformance to.
const SCHEMA_URI: &str = "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json";

/// A plain-language description for a `friction-rules` rule id, for
/// `tool.driver.rules[].shortDescription`.
///
/// Covers every id in `friction_apply::registered_rules()`; an id not
/// found here (impossible for the shipped rule set, but not a panic
/// either) falls back to a generic description built from the id itself.
#[must_use]
fn rule_description(id: &str) -> &'static str {
    match id {
        "connective.surgery" => {
            "Rewrites or removes an overused sentence-initial connective (e.g. \"Moreover,\", \
             \"Furthermore,\")."
        }
        "contraction.insert" => {
            "Inserts a contraction (e.g. \"do not\" -> \"don't\") where the genre's human \
             baseline expects one."
        }
        "lexical.filler_phrase" => {
            "Deletes a discourse-filler phrase that adds no propositional content (e.g. \"it \
             is worth noting that\")."
        }
        "lexical.substitution" => {
            "Substitutes an overused LLM-favored word or phrase for a more natural alternative \
             (e.g. \"leverage\" -> \"use\")."
        }
        "rhythm.fuse" => {
            "Fuses short, choppy consecutive sentences to restore natural sentence-length \
             variation."
        }
        "rhythm.split" => "Splits an overly long sentence at an existing grammatical boundary.",
        "structural.bold_label_strip" => {
            "Strips a bolded lead-in label from a paragraph or list item (e.g. \"**Note:**\")."
        }
        "structural.header_merge" => {
            "Merges an over-segmented heading or section into the surrounding prose."
        }
        "structural.unbullet" => {
            "Converts a short, parallel bullet list into a single serial-comma sentence."
        }
        "symmetry.not_just_but" => {
            "Flags an overused \"not just X but (also) Y\" coordination pattern."
        }
        "symmetry.participial_closer" => {
            "Flags a sentence-final participial closer clause (e.g. \", ensuring reliability.\")."
        }
        "symmetry.ritual_conclusion" => {
            "Flags a ritual opening or closing marker (e.g. \"In conclusion,\", \"Overall,\")."
        }
        "symmetry.triad_reduction" => {
            "Flags an overused three-item coordination pattern (\"X, Y, and Z\")."
        }
        _ => "A friction rule finding.",
    }
}

/// SARIF's `level` for one finding's [`Tier`]: `Tier::Fix` (an
/// automatically-applicable change exists, so `friction fix` resolves it
/// on its own) is `"warning"`; `Tier::Suggest` (needs a human's judgment,
/// so it is the one a reader most needs to see) is the higher `"error"`.
///
/// This ordering (`Suggest` outranks `Fix`) must agree with
/// [`crate::diagnostics`]'s `miette::Severity` ordering for the same
/// `Tier`s, so that `check`'s `--format text` and `--format sarif`
/// outputs never disagree about which findings are more urgent — see
/// `crate::diagnostics`'s `severity_ordering_agrees_with_sarif_level_ordering`
/// test, which pins both orderings together.
pub const fn level_for(tier: Tier) -> &'static str {
    match tier {
        Tier::Fix => "warning",
        Tier::Suggest => "error",
    }
}

#[derive(Debug, Serialize)]
struct Log {
    #[serde(rename = "$schema")]
    schema: &'static str,
    version: &'static str,
    runs: Vec<Run>,
}

#[derive(Debug, Serialize)]
struct Run {
    tool: Tool,
    results: Vec<SarifResult>,
}

#[derive(Debug, Serialize)]
struct Tool {
    driver: Driver,
}

#[derive(Debug, Serialize)]
struct Driver {
    name: &'static str,
    #[serde(rename = "informationUri")]
    information_uri: &'static str,
    version: &'static str,
    rules: Vec<RuleDescriptor>,
}

#[derive(Debug, Serialize)]
struct RuleDescriptor {
    id: String,
    #[serde(rename = "shortDescription")]
    short_description: Message,
}

#[derive(Debug, Serialize)]
struct Message {
    text: String,
}

#[derive(Debug, Serialize)]
struct SarifResult {
    #[serde(rename = "ruleId")]
    rule_id: String,
    level: &'static str,
    message: Message,
    locations: Vec<Location>,
}

#[derive(Debug, Serialize)]
struct Location {
    #[serde(rename = "physicalLocation")]
    physical_location: PhysicalLocation,
}

#[derive(Debug, Serialize)]
struct PhysicalLocation {
    #[serde(rename = "artifactLocation")]
    artifact_location: ArtifactLocation,
    region: Region,
}

#[derive(Debug, Serialize)]
struct ArtifactLocation {
    uri: String,
}

#[derive(Debug, Serialize)]
struct Region {
    #[serde(rename = "startLine")]
    start_line: usize,
    #[serde(rename = "startColumn")]
    start_column: usize,
    #[serde(rename = "endLine")]
    end_line: usize,
    #[serde(rename = "endColumn")]
    end_column: usize,
}

/// Renders `findings` (already scanned against `source`) as a SARIF 2.1.0
/// log, one `results[]` entry per finding in the order given (`check`
/// hands this a list already sorted by `(range.start, rule, range.end)` —
/// see `crate::scan::scan`) and one `tool.driver.rules[]` entry per
/// distinct rule id among them, sorted lexicographically.
///
/// `artifact_uri` is used verbatim as every result's
/// `physicalLocation.artifactLocation.uri` — the caller-given path (or
/// `"<stdin>"`; see `crate::common::display_path`), never resolved to an
/// absolute filesystem path.
#[must_use]
pub fn render(findings: &[Finding], source: &str, artifact_uri: &str) -> String {
    let mut rule_ids: BTreeSet<&str> = BTreeSet::new();
    for finding in findings {
        rule_ids.insert(finding.rule.as_str());
    }

    let rules = rule_ids
        .into_iter()
        .map(|id| RuleDescriptor {
            id: id.to_string(),
            short_description: Message {
                text: rule_description(id).to_string(),
            },
        })
        .collect();

    let results = findings
        .iter()
        .map(|finding| {
            let (start_line, start_column) = offset_to_line_col(source, finding.range.start);
            let (end_line, end_column) = offset_to_line_col(source, finding.range.end);
            SarifResult {
                rule_id: finding.rule.as_str().to_string(),
                level: level_for(finding.tier),
                message: Message {
                    text: finding.message.clone(),
                },
                locations: vec![Location {
                    physical_location: PhysicalLocation {
                        artifact_location: ArtifactLocation {
                            uri: artifact_uri.to_string(),
                        },
                        region: Region {
                            start_line,
                            start_column,
                            end_line,
                            end_column,
                        },
                    },
                }],
            }
        })
        .collect();

    let log = Log {
        schema: SCHEMA_URI,
        version: "2.1.0",
        runs: vec![Run {
            tool: Tool {
                driver: Driver {
                    name: "friction",
                    information_uri: env!("CARGO_PKG_REPOSITORY"),
                    version: env!("CARGO_PKG_VERSION"),
                    rules,
                },
            },
            results,
        }],
    };

    serde_json::to_string_pretty(&log).expect("SARIF log serializes: every field is plain data")
}

#[cfg(test)]
mod tests {
    use friction_core::RuleId;

    use super::*;

    /// An empty finding list still renders a well-formed, empty-results
    /// SARIF log.
    #[test]
    fn render_handles_no_findings() {
        let json = render(&[], "hello", "file.md");
        let value: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        assert_eq!(value["version"], "2.1.0");
        assert!(value["runs"][0]["results"].as_array().unwrap().is_empty());
        assert!(
            value["runs"][0]["tool"]["driver"]["rules"]
                .as_array()
                .unwrap()
                .is_empty()
        );
    }

    /// One finding renders one result with a 1-based line/column region
    /// and a matching rule descriptor, deduplicated across repeats of the
    /// same rule.
    #[test]
    fn render_maps_one_finding_to_one_result_with_deduped_rules() {
        let source = "one\ntwo not just fast but also reliable.";
        let range = source.find("not just").unwrap()..source.find("also reliable").unwrap() + 14;
        let findings = vec![
            Finding::new(
                RuleId::new("symmetry.not_just_but"),
                range.clone(),
                "not just X but also Y",
                Tier::Suggest,
            ),
            Finding::new(
                RuleId::new("symmetry.not_just_but"),
                range,
                "not just X but also Y (again)",
                Tier::Suggest,
            ),
        ];
        let json = render(&findings, source, "doc.md");
        let value: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        let results = value["runs"][0]["results"].as_array().unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0]["ruleId"], "symmetry.not_just_but");
        assert_eq!(results[0]["level"], "error");
        let region = &results[0]["locations"][0]["physicalLocation"]["region"];
        assert_eq!(region["startLine"], 2);
        assert_eq!(region["startColumn"], 5);

        let rules = value["runs"][0]["tool"]["driver"]["rules"]
            .as_array()
            .unwrap();
        assert_eq!(rules.len(), 1, "the rule id repeats but must appear once");
        assert_eq!(rules[0]["id"], "symmetry.not_just_but");
    }

    /// `Tier::Fix` maps to SARIF level `"warning"`, `Tier::Suggest` to
    /// the higher `"error"` — the same relative ordering as
    /// `crate::diagnostics`'s `miette::Severity` mapping for the same
    /// tiers (see `severity_ordering_agrees_with_sarif_level_ordering` in
    /// that module).
    #[test]
    fn level_for_maps_tiers() {
        assert_eq!(level_for(Tier::Fix), "warning");
        assert_eq!(level_for(Tier::Suggest), "error");
    }
}
