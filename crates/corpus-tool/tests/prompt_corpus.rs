//! Every `corpus/prompts/*.toml` file in the repo must parse, each
//! genre file individually has at least 40 prompts, and prompt ids are
//! unique across all genre files.
//!
//! This exercises the real, committed `corpus/prompts/` directory (not a
//! tempdir fixture) via `corpus_tool::prompts::load`, so a malformed or
//! under-populated prompt catalog fails `cargo test` directly.

use std::collections::BTreeSet;
use std::path::Path;

use corpus_tool::manifest::Genre;
use corpus_tool::prompts;

const ALL_GENRES: [Genre; 5] = [
    Genre::Docs,
    Genre::Blog,
    Genre::Readme,
    Genre::Email,
    Genre::Forum,
];

/// `corpus/prompts/<genre>.toml` parses for every frozen genre,
/// *each genre file individually* has at least 40 prompts (this
/// is a per-genre minimum, not a total across genres — a genre could drop
/// to a handful of prompts while others stayed large and still pass a
/// total-only check), and the union of prompt ids across all genre files
/// is unique.
#[test]
fn all_genre_prompt_files_parse_with_at_least_40_unique_ids_per_genre() {
    let prompts_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus/prompts");
    assert!(
        prompts_dir.is_dir(),
        "expected corpus/prompts/ to exist at {}",
        prompts_dir.display()
    );

    let mut all_ids = BTreeSet::new();
    let mut total = 0usize;

    for genre in ALL_GENRES {
        let loaded = prompts::load(&prompts_dir, genre)
            .unwrap_or_else(|e| panic!("corpus/prompts/{genre}.toml failed to parse: {e:#}"));
        assert!(
            loaded.len() >= 40,
            "corpus/prompts/{genre}.toml has {} prompts; each genre requires >= 40 prompts",
            loaded.len()
        );
        for prompt in &loaded {
            assert!(
                all_ids.insert(prompt.id.clone()),
                "duplicate prompt id {:?} (genre {genre}) — ids must be unique across all corpus/prompts/*.toml files",
                prompt.id
            );
        }
        total += loaded.len();
    }

    assert_eq!(
        all_ids.len(),
        total,
        "prompt id uniqueness check and total count diverged unexpectedly"
    );
}
