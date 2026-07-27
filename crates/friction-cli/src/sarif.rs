//! SARIF 2.1.0 output for `friction check --format sarif`.
//!
//! Builds a minimal, schema-valid SARIF log: one `tool.driver.rules[]`
//! entry per distinct frame id actually present among a run's detection
//! spans (with a plain-language description derived from the channel that
//! produced it — see [`channel_description`]), and one `results[]` entry
//! per span, with a `physicalLocation.region` derived from the span's
//! byte range via [`crate::common::LineIndex`].
//!
//! `tests/sarif_schema.rs` validates this module's output against the
//! vendored SARIF 2.1.0 JSON schema (`tests/data/sarif-schema-2.1.0.json`)
//! with the `jsonschema` crate.

use std::collections::BTreeMap;

use friction_match::{Channel, MatchSpan};
use serde::Serialize;

use crate::common::LineIndex;

/// The SARIF schema URI this log declares conformance to.
const SCHEMA_URI: &str = "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json";

/// A plain-language description of what detection `channel` covers, for
/// `tool.driver.rules[].shortDescription`.
#[must_use]
const fn channel_description(channel: Channel) -> &'static str {
    match channel {
        Channel::Dms => {
            "Differential matching statistics: a span whose local token \
             continuations are attested far more often in this generator \
             family's corpus than in the human baseline."
        }
        Channel::Literal => {
            "A curated literal tell: a fixed filler phrase, ritual opening/\
             closing, or lexical substitution pattern from the inventory \
             pack."
        }
        Channel::Lvc => {
            "A licensed light-verb construction (e.g. \"performs validation \
             of\") that the pack allows deriving back to its underlying \
             verb (e.g. \"validates\")."
        }
        Channel::Frame => {
            "A deterministic contrast-frame template: a dismissive-foil \
             interrogative (\"is X just A, or B?\") or a declarative \
             epanorthosis (\"not just X — it's Y\")."
        }
        Channel::Jargon => {
            "A curated metaphor lexeme (e.g. \"resonance\", \"tapestry\", \
             \"well\") heading a noun compound (\"semantic wells\", \
             \"cross-domain resonance\") — a narrow, high-precision slice \
             of invented pseudo-terminology, detection-only."
        }
    }
}

/// SARIF's `level` for every span this crate reports: `"warning"` — a
/// span is a candidate `friction fix` can act on, matching that engine's
/// own `Tier::Fix` framing, not necessarily something that needs a
/// human's judgment on its own (the repair engine's own gates decide, per
/// candidate, whether it actually applies or gets held).
///
/// This must agree with [`crate::diagnostics::render_spans`]'s
/// `miette::Severity` for the same spans, so `check`'s `--format text` and
/// `--format sarif` outputs never disagree about severity.
pub const LEVEL: &str = "warning";

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

/// Renders `spans` (already scanned against `source`) as a SARIF 2.1.0
/// log, one `results[]` entry per span in the order given (`check` hands
/// this a list already sorted by `friction_match::span::merge_spans`) and
/// one `tool.driver.rules[]` entry per distinct frame id among them,
/// sorted lexicographically.
///
/// `artifact_uri` is used verbatim as every result's
/// `physicalLocation.artifactLocation.uri` — the caller-given path (or
/// `"<stdin>"`; see `crate::common::display_path`), never resolved to an
/// absolute filesystem path.
#[must_use]
pub fn render(spans: &[MatchSpan], source: &str, artifact_uri: &str) -> String {
    let mut channel_by_frame_id: BTreeMap<&str, Channel> = BTreeMap::new();
    for span in spans {
        channel_by_frame_id
            .entry(&span.frame_id)
            .or_insert(span.channel);
    }

    let rules = channel_by_frame_id
        .into_iter()
        .map(|(id, channel)| RuleDescriptor {
            id: id.to_string(),
            short_description: Message {
                text: channel_description(channel).to_string(),
            },
        })
        .collect();

    let lines = LineIndex::new(source);
    let results = spans
        .iter()
        .map(|span| {
            let (start_line, start_column) = lines.line_col(source, span.range.start);
            let (end_line, end_column) = lines.line_col(source, span.range.end);
            SarifResult {
                rule_id: span.frame_id.to_string(),
                level: LEVEL,
                message: Message {
                    text: format!("{:?} tell: {}", span.channel, span.frame_id),
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
    use friction_match::MatchScore;

    use super::*;

    fn span(range: std::ops::Range<usize>, channel: Channel, frame_id: &str) -> MatchSpan {
        MatchSpan {
            range,
            channel,
            frame_id: frame_id.into(),
            score: MatchScore::Present,
        }
    }

    /// An empty span list still renders a well-formed, empty-results
    /// SARIF log.
    #[test]
    fn render_handles_no_spans() {
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

    /// One span renders one result with a 1-based line/column region and
    /// a matching rule descriptor, deduplicated across repeats of the
    /// same frame id.
    #[test]
    fn render_maps_one_span_to_one_result_with_deduped_rules() {
        let source = "one\ntwo simply validates the input.";
        let range = source.find("simply").unwrap()..source.find("validates").unwrap();
        let spans = vec![
            span(range.clone(), Channel::Literal, "span.simply"),
            span(range, Channel::Literal, "span.simply"),
        ];
        let json = render(&spans, source, "doc.md");
        let value: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        let results = value["runs"][0]["results"].as_array().unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0]["ruleId"], "span.simply");
        assert_eq!(results[0]["level"], "warning");
        let region = &results[0]["locations"][0]["physicalLocation"]["region"];
        assert_eq!(region["startLine"], 2);
        assert_eq!(region["startColumn"], 5);

        let rules = value["runs"][0]["tool"]["driver"]["rules"]
            .as_array()
            .unwrap();
        assert_eq!(rules.len(), 1, "the frame id repeats but must appear once");
        assert_eq!(rules[0]["id"], "span.simply");
    }
}
