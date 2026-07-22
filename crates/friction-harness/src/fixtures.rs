//! Verbatim regression-fixture data, embedded at compile time from
//! `docs/research/fixtures.json` and the sample documents alongside it.
//!
//! Nothing here re-authors or re-derives fixture content: the JSON and the
//! two markdown sample files are loaded byte-for-byte via `include_str!`
//! and parsed into a typed shape the rest of this crate can address by
//! field instead of by ad hoc JSON indexing.

use std::sync::LazyLock;

use serde::Deserialize;

/// The embedded fixture manifest, loaded verbatim from
/// `docs/research/fixtures.json` (path relative to this file, which lives
/// at `crates/friction-harness/src/fixtures.rs`).
const FIXTURES_JSON: &str = include_str!("../../../docs/research/fixtures.json");

/// The chat-speak rank-fixture sample document, loaded verbatim from
/// `docs/research/samples/chatspeak.md`.
pub const CHATSPEAK_MD: &str = include_str!("../../../docs/research/samples/chatspeak.md");

/// The well-written-docs rank-fixture sample document, loaded verbatim from
/// `docs/research/samples/good_docs.md`.
pub const GOOD_DOCS_MD: &str = include_str!("../../../docs/research/samples/good_docs.md");

/// The parsed, embedded fixture manifest, parsed once and reused for the
/// life of the process.
///
/// # Panics
/// Panics if the embedded `fixtures.json` fails to parse — a bug in this
/// crate's own vendored copy of that file (covered by this module's own
/// tests), not a condition any caller can recover from by retrying.
pub static FIXTURES: LazyLock<Fixtures> = LazyLock::new(|| {
    serde_json::from_str(FIXTURES_JSON)
        .expect("embedded fixtures.json must parse: see fixtures module tests")
});

/// Resolves a `worse_file`/`better_file` value from a rank fixture (a path
/// relative to `docs/research/`, e.g. `"samples/chatspeak.md"`) to its
/// embedded text.
///
/// Centralizing this mapping — rather than re-authoring or re-reading
/// either sample file anywhere else in this crate — is what keeps the
/// sample docs loaded exactly once, verbatim.
#[must_use]
pub fn sample_by_path(relative: &str) -> Option<&'static str> {
    match relative {
        "samples/chatspeak.md" => Some(CHATSPEAK_MD),
        "samples/good_docs.md" => Some(GOOD_DOCS_MD),
        _ => None,
    }
}

/// The top-level shape of `docs/research/fixtures.json`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Fixtures {
    /// The manifest's own free-text description of its categories.
    pub comment: String,
    /// Fixtures the engine must produce exactly the documented output for.
    pub accept: Vec<AcceptFixture>,
    /// Fixtures the engine must make no edit to (or must not produce the
    /// documented bad output for).
    pub reject: Vec<RejectFixture>,
    /// Fixtures the scorer must order one text ahead of another for.
    pub rank: Vec<RankFixture>,
    /// Known misses recorded for future inventory growth, not asserted by
    /// this crate's tests.
    pub inventory_frontier: InventoryFrontier,
}

/// One `accept` fixture: an input this milestone's checks (closure,
/// tell-span reduction, score improvement) must all hold for, given its
/// documented `output`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AcceptFixture {
    /// The fixture's stable identifier.
    pub id: String,
    /// The input text, for single-pair fixtures.
    #[serde(default)]
    pub input: Option<String>,
    /// The expected output text, for single-pair fixtures.
    #[serde(default)]
    pub output: Option<String>,
    /// Free-text notes about this fixture, if any.
    #[serde(default)]
    pub notes: Option<String>,
    /// Where this fixture's input was drawn from, if recorded.
    #[serde(default)]
    pub provenance: Option<String>,
    /// `(input, output)` pairs, for multi-pair fixtures (`pivot_constructed_cases`).
    #[serde(default)]
    pub pairs: Option<Vec<(String, String)>>,
}

/// A single labeled block of input text (used by `heading_pivot`, whose
/// input is a heading rather than a paragraph).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputBlock {
    /// The block's markdown kind, e.g. `"heading"`.
    #[serde(rename = "type")]
    pub kind: String,
    /// The block's own text.
    pub text: String,
}

/// One `reject` fixture: an input the engine must never apply the
/// documented bad edit to, gated by the named mechanism.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RejectFixture {
    /// The fixture's stable identifier.
    pub id: String,
    /// The input text, for fixtures whose input is ordinary prose.
    #[serde(default)]
    pub input: Option<String>,
    /// The input text, for fixtures whose input is a non-prose block
    /// (`heading_pivot`).
    #[serde(default)]
    pub input_block: Option<InputBlock>,
    /// A single documented bad output, for fixtures with exactly one.
    #[serde(default)]
    pub bad_output: Option<String>,
    /// Multiple documented bad outputs, for fixtures with more than one.
    #[serde(default)]
    pub bad_outputs: Option<Vec<String>>,
    /// The correct edit, for fixtures that also document one.
    #[serde(default)]
    pub good_output: Option<String>,
    /// Free-text description of which gate this fixture exercises.
    pub gate: String,
}

impl RejectFixture {
    /// Normalizes `bad_output`/`bad_outputs` into one iterable: exactly one
    /// of the two is ever set, per the source JSON, so callers never need
    /// to special-case the shape.
    #[must_use]
    pub fn bad_outputs(&self) -> Vec<&str> {
        match (&self.bad_output, &self.bad_outputs) {
            (Some(single), None) => vec![single.as_str()],
            (None, Some(many)) => many.iter().map(String::as_str).collect(),
            (None, None) => Vec::new(),
            (Some(_), Some(_)) => {
                unreachable!("fixtures.json never sets both bad_output and bad_outputs")
            }
        }
    }
}

/// One `rank` fixture: the scorer must order `better` (or `better_file`)
/// ahead of `worse` (or `worse_file`).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RankFixture {
    /// The fixture's stable identifier.
    pub id: String,
    /// The worse text, given inline.
    #[serde(default)]
    pub worse: Option<String>,
    /// The better text, given inline.
    #[serde(default)]
    pub better: Option<String>,
    /// The worse text, as a path into `docs/research/samples/`.
    #[serde(default)]
    pub worse_file: Option<String>,
    /// The better text, as a path into `docs/research/samples/`.
    #[serde(default)]
    pub better_file: Option<String>,
    /// Free-text notes about this fixture, if any.
    #[serde(default)]
    pub notes: Option<String>,
}

/// The manifest's own record of known inventory misses — data only, never
/// asserted against by this crate's tests.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InventoryFrontier {
    /// Free-text description of what these examples represent.
    pub comment: String,
    /// The example phrases themselves.
    pub examples: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The embedded manifest parses under `deny_unknown_fields`, so a
    /// future field added to `fixtures.json` this crate hasn't been
    /// extended for fails here immediately.
    #[test]
    fn fixtures_json_parses() {
        assert!(!FIXTURES.accept.is_empty());
        assert!(!FIXTURES.reject.is_empty());
        assert!(!FIXTURES.rank.is_empty());
    }

    /// Pins `inventory_frontier.examples`'s actual content (not just its
    /// structural presence): if a documented frontier phrase were deleted
    /// or corrupted in the embedded manifest, this catches it.
    #[test]
    fn inventory_frontier_examples_match_the_documented_phrases() {
        assert_eq!(
            FIXTURES.inventory_frontier.examples,
            vec![
                "we'll delve into",
                "our development journey",
                "fortunate enough to have had the opportunity to",
            ]
        );
    }

    /// `sample_by_path` resolves both rank-fixture sample paths to their
    /// embedded text, and rejects anything else.
    #[test]
    fn sample_by_path_resolves_known_paths_only() {
        assert_eq!(sample_by_path("samples/chatspeak.md"), Some(CHATSPEAK_MD));
        assert_eq!(sample_by_path("samples/good_docs.md"), Some(GOOD_DOCS_MD));
        assert_eq!(sample_by_path("samples/nope.md"), None);
    }

    /// `bad_outputs` normalizes both shapes fixtures.json uses.
    #[test]
    fn bad_outputs_normalizes_single_and_multi_shapes() {
        let single = RejectFixture {
            id: "x".into(),
            input: None,
            input_block: None,
            bad_output: Some("one".into()),
            bad_outputs: None,
            good_output: None,
            gate: "g".into(),
        };
        assert_eq!(single.bad_outputs(), vec!["one"]);

        let multi = RejectFixture {
            id: "y".into(),
            input: None,
            input_block: None,
            bad_output: None,
            bad_outputs: Some(vec!["a".into(), "b".into()]),
            good_output: None,
            gate: "g".into(),
        };
        assert_eq!(multi.bad_outputs(), vec!["a", "b"]);
    }
}
