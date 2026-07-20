//! Shared plumbing for `check`, `fix`, and `explain`: genre/format value
//! types, input reading (file or stdin), envelope pack loading, and the
//! segmenter/tagger/envelope handle every subcommand needs to run the
//! rule engine.

use std::fmt;
use std::io::{Read as _, Write as _};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::ValueEnum;
use friction_core::Envelope;
use friction_nlp::{NlpruleTagger, SrxSegmenter, TagError};
use friction_packs::{ENVELOPE_V2, EnvelopePack, PackError};
use friction_rules::GenreEnvelope;

/// The genres a document may be classified as, matching
/// `friction-packs`' envelope-pack genre keys exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "lower")]
pub enum Genre {
    /// Prose documentation (READMEs excluded — see [`Genre::Readme`]).
    Docs,
    /// Blog-style prose.
    Blog,
    /// A project README.
    Readme,
    /// Email prose.
    Email,
    /// Forum-post prose.
    Forum,
}

impl Genre {
    /// The default genre used when `--genre` is omitted: `docs`.
    pub const DEFAULT: Self = Self::Docs;

    /// This genre's key in a `friction-packs` envelope pack's
    /// `[<genre>.<metric>]` tables (e.g. `"docs"`, `"blog"`).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Docs => "docs",
            Self::Blog => "blog",
            Self::Readme => "readme",
            Self::Email => "email",
            Self::Forum => "forum",
        }
    }
}

impl fmt::Display for Genre {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Resolves an optional `--genre` flag to a concrete [`Genre`], printing a
/// note to stderr (once) when it had to default.
///
/// Shared by every subcommand that takes genre-scoped input (`check`,
/// `fix`, `explain`) so a defaulted genre is announced identically
/// everywhere, whether the input came from a file or stdin.
#[must_use]
pub fn resolve_genre(explicit: Option<Genre>) -> Genre {
    explicit.unwrap_or_else(|| {
        eprintln!(
            "friction: note: no --genre given; defaulting to {:?}",
            Genre::DEFAULT.as_str()
        );
        Genre::DEFAULT
    })
}

/// Output shapes shared by `check`/`fix`/`explain`. Not every subcommand
/// supports every variant (`fix` and `explain` reject `--format sarif`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Default)]
#[value(rename_all = "lower")]
pub enum Format {
    /// Human-readable tables and (for `check`) miette diagnostics.
    #[default]
    Text,
    /// Stable, serde-derived JSON.
    Json,
    /// SARIF 2.1.0 (`check` only).
    Sarif,
}

/// Errors shared by every subcommand: reading input, parsing, loading a
/// pack, or building the NLP engine.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum CliError {
    /// The input could not be read (file I/O or stdin).
    #[error("could not read input {path:?}: {source}")]
    ReadInput {
        /// The path (or `-` for stdin) that failed to read.
        path: String,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// The input could not be written back out (`--in-place`).
    #[error("could not write output {path:?}: {source}")]
    WriteOutput {
        /// The path that failed to write.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// `--pack` named a file that could not be read.
    #[error("could not read pack {path:?}: {source}")]
    ReadPack {
        /// The `--pack` path that failed to read.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// `--pack` named a file that failed to parse as an envelope pack.
    #[error("pack {path:?} is not a valid envelope pack: {source}")]
    ParsePack {
        /// The `--pack` path that failed to parse.
        path: PathBuf,
        /// Underlying pack-parse error.
        #[source]
        source: PackError,
    },
    /// The embedded English part-of-speech tagger model failed to load.
    #[error("failed to load the embedded English tagger model: {0}")]
    Tagger(#[from] TagError),
    /// The input failed to parse as markdown.
    #[error("{0}")]
    Parse(#[from] friction_parse::ParseError),
    /// The input failed to sentence-segment.
    #[error("{0}")]
    Segment(#[from] friction_nlp::SegmentError),
    /// The fixpoint driver failed.
    #[error("{0}")]
    Apply(#[from] friction_apply::ApplyError),
    /// `--format sarif` was requested for a subcommand that does not
    /// support it.
    #[error("--format sarif is only supported by `friction check`")]
    SarifUnsupported,
    /// `--in-place` was requested with stdin (`-`) input, which has no
    /// file to write back to.
    #[error("--in-place requires a real file path, not stdin (-)")]
    InPlaceStdin,
}

impl CliError {
    /// Prints this error to stderr and returns the exit code every
    /// subcommand's `main.rs` arm uses for a hard failure.
    #[must_use]
    pub fn report(&self) -> ExitCode {
        eprintln!("friction: error: {self}");
        ExitCode::from(2)
    }
}

/// Reads `path`'s full contents as UTF-8 text, or stdin's if `path` is
/// exactly `-`.
///
/// # Errors
/// Returns [`CliError::ReadInput`] on any I/O failure (including invalid
/// UTF-8, surfaced as an [`std::io::Error`] of kind `InvalidData` the same
/// way `std::fs::read_to_string` reports it).
pub fn read_input(path: &str) -> Result<String, CliError> {
    if path == "-" {
        let mut buf = String::new();
        std::io::stdin()
            .read_to_string(&mut buf)
            .map_err(|source| CliError::ReadInput {
                path: path.to_string(),
                source,
            })?;
        Ok(buf)
    } else {
        std::fs::read_to_string(path).map_err(|source| CliError::ReadInput {
            path: path.to_string(),
            source,
        })
    }
}

/// Atomically replaces `path`'s contents with `contents`: writes to a
/// sibling `<name>.partial` temp file first, then `fs::rename`s it into
/// place, so a process interrupted mid-write (disk full, `SIGKILL`,
/// power loss) between the write and the rename never truncates or
/// corrupts `path` — either the rename never happens and `path` is
/// untouched, or it happens after the full new contents are already
/// durably on disk. Mirrors `setup.rs`'s `ensure_cached`, which writes
/// downloaded artifacts the same way for the same reason.
///
/// Plain `std::fs::write` would instead open `path` with truncation
/// *before* writing the new bytes, discarding the original content up
/// front — a mid-write failure would then leave `path` empty or holding a
/// partial document with no way to recover the original.
///
/// # Errors
/// Returns [`CliError::WriteOutput`] if the temp file cannot be created
/// or written, or if the rename fails.
pub fn write_in_place(path: &Path, contents: &str) -> Result<(), CliError> {
    let err = |source| CliError::WriteOutput {
        path: path.to_path_buf(),
        source,
    };

    let mut tmp_name = path.file_name().map_or_else(
        || std::ffi::OsString::from("friction-fix"),
        std::ffi::OsStr::to_os_string,
    );
    tmp_name.push(".partial");
    let tmp_path = path.with_file_name(tmp_name);

    let mut file = std::fs::File::create(&tmp_path).map_err(err)?;
    file.write_all(contents.as_bytes()).map_err(err)?;
    file.sync_all().map_err(err)?;
    drop(file);

    std::fs::rename(&tmp_path, path).map_err(err)
}

/// A display label for `path` suitable for diagnostics and SARIF
/// `artifactLocation.uri`: `path` itself, verbatim — never resolved to an
/// absolute path (deterministic output must never embed the invoking
/// machine's filesystem layout), and `"<stdin>"` for stdin input.
#[must_use]
pub fn display_path(path: &str) -> &str {
    if path == "-" { "<stdin>" } else { path }
}

/// The envelope pack a run should use: either the embedded `envelope-v2`
/// pack, or a caller-supplied `--pack` override.
pub enum Pack {
    /// The embedded, shipped pack (`friction_packs::ENVELOPE_V2`).
    Embedded,
    /// A pack loaded from a `--pack` override file.
    Loaded(EnvelopePack),
}

impl Pack {
    /// Loads `override_path` as an envelope pack if given, else selects
    /// the embedded pack.
    ///
    /// # Errors
    /// Returns [`CliError::ReadPack`] or [`CliError::ParsePack`] if
    /// `override_path` is given but cannot be read or parsed.
    pub fn load(override_path: Option<&Path>) -> Result<Self, CliError> {
        match override_path {
            None => Ok(Self::Embedded),
            Some(path) => {
                let text = std::fs::read_to_string(path).map_err(|source| CliError::ReadPack {
                    path: path.to_path_buf(),
                    source,
                })?;
                let pack = EnvelopePack::parse(&text).map_err(|source| CliError::ParsePack {
                    path: path.to_path_buf(),
                    source,
                })?;
                Ok(Self::Loaded(pack))
            }
        }
    }

    /// Borrows the underlying pack, whichever source it came from.
    #[must_use]
    pub fn as_pack(&self) -> &EnvelopePack {
        match self {
            Self::Embedded => &ENVELOPE_V2,
            Self::Loaded(pack) => pack,
        }
    }
}

/// A [`GenreEnvelope`] view over one genre's slice of a [`Pack`].
///
/// Mirrors `friction-apply::fix::PackEnvelope`, which is private to that
/// crate — `friction-cli` needs the same small adapter for its own
/// `--pack`-overridable pack, so it is redefined here rather than exposed
/// as a public dependency between the two crates for a two-line struct.
pub struct PackEnvelope<'a> {
    pack: &'a EnvelopePack,
    genre: &'a str,
}

impl<'a> PackEnvelope<'a> {
    /// Builds a view over `pack`'s bands for `genre`.
    #[must_use]
    pub const fn new(pack: &'a EnvelopePack, genre: &'a str) -> Self {
        Self { pack, genre }
    }
}

impl GenreEnvelope for PackEnvelope<'_> {
    fn band(&self, metric: &str) -> Option<Envelope> {
        self.pack.band(self.genre, metric)
    }
}

/// The loaded segmenter and part-of-speech tagger every subcommand needs
/// to run the rule engine, built once per process.
pub struct Engine {
    /// The sentence segmenter.
    pub segmenter: SrxSegmenter,
    /// The part-of-speech tagger (loads the embedded English model).
    pub tagger: NlpruleTagger,
}

impl Engine {
    /// Loads the segmenter and tagger.
    ///
    /// # Errors
    /// Returns [`CliError::Tagger`] if the embedded English tagger model
    /// fails to load.
    pub fn load() -> Result<Self, CliError> {
        Ok(Self {
            segmenter: SrxSegmenter::new(),
            tagger: NlpruleTagger::new()?,
        })
    }
}

/// Converts a 0-based byte offset into `source` to a 1-based `(line,
/// column)` pair, both counted in Unicode scalar values (`char`s), not
/// bytes — the unit SARIF's `region.startColumn`/`endColumn` and most
/// editors use.
///
/// `offset` past `source.len()` clamps to the position one past the last
/// character, rather than panicking: a defensive fallback for a
/// pathological byte range no well-formed [`friction_core::Finding`]
/// should ever carry. An in-bounds `offset` that lands mid-character
/// (splitting a multi-byte UTF-8 sequence) is likewise walked back to the
/// nearest preceding character boundary rather than panicking on the
/// slice below — `offset` is caller-supplied and not guaranteed to be
/// `source`-char-boundary-valid the way a validated [`friction_core::
/// Finding`]/[`friction_core::Patch`] range already is (see
/// `friction_core::span::validate_range`), so this function has to
/// tolerate it on its own.
#[must_use]
pub fn offset_to_line_col(source: &str, offset: usize) -> (usize, usize) {
    let mut offset = offset.min(source.len());
    while !source.is_char_boundary(offset) {
        offset -= 1;
    }
    let mut line = 1usize;
    let mut col = 1usize;
    for ch in source[..offset].chars() {
        if ch == '\n' {
            line += 1;
            col = 1;
        } else {
            col += 1;
        }
    }
    (line, col)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn genre_as_str_matches_pack_keys() {
        assert_eq!(Genre::Docs.as_str(), "docs");
        assert_eq!(Genre::Blog.as_str(), "blog");
        assert_eq!(Genre::Readme.as_str(), "readme");
        assert_eq!(Genre::Email.as_str(), "email");
        assert_eq!(Genre::Forum.as_str(), "forum");
    }

    #[test]
    fn display_path_labels_stdin() {
        assert_eq!(display_path("-"), "<stdin>");
        assert_eq!(display_path("foo.md"), "foo.md");
    }

    /// `write_in_place` replaces the target file's contents and leaves no
    /// leftover `.partial` temp file behind on success.
    #[test]
    fn write_in_place_replaces_contents_and_cleans_up_temp_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("doc.md");
        std::fs::write(&path, "original").expect("seed file");

        write_in_place(&path, "replaced").expect("write_in_place succeeds");

        assert_eq!(
            std::fs::read_to_string(&path).expect("read back"),
            "replaced"
        );
        assert!(
            !dir.path().join("doc.md.partial").exists(),
            "the temp file must not survive a successful rename"
        );
    }

    /// Regression test for the mid-write data-loss hazard `write_in_place`
    /// exists to prevent: if the write fails before the atomic rename, the
    /// original file must be left byte-for-byte untouched — never
    /// truncated or partially overwritten the way a direct
    /// `std::fs::write` to the destination would leave it.
    #[cfg(unix)]
    #[test]
    fn write_in_place_leaves_the_original_file_untouched_on_failure() {
        use std::os::unix::fs::PermissionsExt as _;

        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("doc.md");
        std::fs::write(&path, "original content").expect("seed file");

        // Make the directory unwritable so creating the sibling `.partial`
        // temp file fails before the destination is ever touched.
        let readonly = std::fs::Permissions::from_mode(0o555);
        std::fs::set_permissions(dir.path(), readonly).expect("chmod dir read-only");

        let result = write_in_place(&path, "replaced content");

        // Restore write permission so the tempdir can clean itself up.
        std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o755))
            .expect("restore dir permissions");

        assert!(
            result.is_err(),
            "the write should fail with a read-only directory"
        );
        assert_eq!(
            std::fs::read_to_string(&path).expect("original file still readable"),
            "original content",
            "a failed write must never touch the original file's contents"
        );
    }

    #[test]
    fn offset_to_line_col_hand_computed() {
        let source = "abc\ndef\nghi";
        assert_eq!(offset_to_line_col(source, 0), (1, 1));
        assert_eq!(offset_to_line_col(source, 3), (1, 4));
        assert_eq!(offset_to_line_col(source, 4), (2, 1));
        assert_eq!(offset_to_line_col(source, 8), (3, 1));
        assert_eq!(offset_to_line_col(source, source.len()), (3, 4));
    }

    /// Regression test for a fuzz-found crash
    /// (`fuzz/fuzz_targets/fuzz_sarif_line_col.rs`): an in-bounds
    /// `offset` that splits a multi-byte character used to panic on
    /// `source[..offset]` (`byte index N is not a char boundary`) instead
    /// of clamping back to the nearest preceding one the way an
    /// out-of-bounds offset already clamped to `source.len()`.
    #[test]
    fn offset_to_line_col_walks_back_a_mid_character_offset_to_the_preceding_boundary() {
        // '\u{58b}' ("֋") is 2 bytes (0xd6 0x8b); offset 1 lands inside it.
        let source = "\u{58b}";
        assert_eq!(offset_to_line_col(source, 1), offset_to_line_col(source, 0));

        // The exact fuzzer-minimized failing case: three copies of the
        // same 2-byte character interleaved with 1-byte ASCII, offset 7
        // landing mid-character.
        let source = "^^\u{58b}^?\u{58b}~/\u{58b}";
        assert_eq!(offset_to_line_col(source, 7), (1, 6));
    }

    #[test]
    fn offset_to_line_col_counts_chars_not_bytes() {
        // 'é' is 2 bytes; the offset just past it is byte 3, but that is
        // still column 2 (one character consumed).
        let source = "é x";
        let e_byte_len = 'é'.len_utf8();
        assert_eq!(offset_to_line_col(source, e_byte_len), (1, 2));
    }
}
