//! `corpus/prompts/<genre>.toml` — the prompt catalog consumed by
//! `corpus-tool generate`.
//!
//! Format: `[[prompts]]` tables with `id`, `text`, `topic`, one file per
//! genre named after the genre (`corpus/prompts/blog.toml`, ...).

use std::path::Path;

use anyhow::Context;
use serde::Deserialize;

use crate::manifest::Genre;

/// One generation prompt.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Prompt {
    pub id: String,
    pub text: String,
    pub topic: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PromptFile {
    #[serde(default)]
    prompts: Vec<Prompt>,
}

/// Reads and parses `<prompts_dir>/<genre>.toml`, returning its prompts
/// sorted by `id` (ascending) for deterministic downstream planning.
///
/// # Errors
///
/// Returns a clear, contextual error — not a panic — if the file doesn't
/// exist yet (prompt files may not be authored for every genre from day
/// one) or fails to parse.
pub fn load(prompts_dir: &Path, genre: Genre) -> anyhow::Result<Vec<Prompt>> {
    let path = prompts_dir.join(format!("{genre}.toml"));
    let text = std::fs::read_to_string(&path).with_context(|| {
        format!(
            "corpus-tool generate: prompt file {} not found; author it \
             ([[prompts]] entries with id/text/topic) before generating docs for genre \"{genre}\"",
            path.display()
        )
    })?;
    let mut file: PromptFile =
        toml::from_str(&text).with_context(|| format!("parsing prompt file {}", path.display()))?;
    file.prompts.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(file.prompts)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A well-formed prompt file parses and is sorted by id
    /// regardless of file order.
    #[test]
    fn load_parses_and_sorts_by_id() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("blog.toml"),
            r#"
            [[prompts]]
            id = "blog-002"
            text = "second"
            topic = "t2"

            [[prompts]]
            id = "blog-001"
            text = "first"
            topic = "t1"
            "#,
        )
        .unwrap();

        let prompts = load(dir.path(), Genre::Blog).unwrap();
        assert_eq!(prompts.len(), 2);
        assert_eq!(prompts[0].id, "blog-001");
        assert_eq!(prompts[1].id, "blog-002");
    }

    /// A missing prompt file is a clear error (mentions the
    /// genre), not a panic.
    #[test]
    fn load_missing_file_returns_clear_error_mentioning_genre() {
        let dir = tempfile::tempdir().unwrap();
        let err = load(dir.path(), Genre::Forum).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("forum"), "message was: {msg}");
    }

    /// An unrecognized field in a `[[prompts]]`
    /// entry is a hard parse error.
    #[test]
    fn load_rejects_unknown_field() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("email.toml"),
            r#"
            [[prompts]]
            id = "email-001"
            text = "hi"
            topic = "t"
            oops = "typo"
            "#,
        )
        .unwrap();
        assert!(load(dir.path(), Genre::Email).is_err());
    }

    /// An empty (no `[[prompts]]` tables) file parses to an empty list
    /// rather than erroring.
    #[test]
    fn load_empty_file_returns_empty_list() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("docs.toml"), "").unwrap();
        assert!(load(dir.path(), Genre::Docs).unwrap().is_empty());
    }
}
