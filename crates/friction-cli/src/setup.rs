//! `friction setup`: downloads and sha256-verifies `friction-packs`'
//! pinned NLP artifact registry into a local, XDG-compliant cache
//! directory.
//!
//! Idempotent: an artifact already present in the cache with a matching
//! checksum is left alone and reported as cached rather than re-fetched.
//! Every downloaded artifact is checksummed against its registry entry
//! before being moved into place; a mismatch is a hard error, never a
//! silently-accepted file. With `--require`, nothing is downloaded at
//! all — the cache is only checked, which is what a CI job that
//! pre-warmed the cache out of band wants to run to confirm nothing is
//! missing.

use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::Args;
use directories::ProjectDirs;
use friction_packs::Artifact;

/// Arguments for `friction setup`.
#[derive(Debug, Args)]
pub struct SetupArgs {
    /// Only check the cache for valid, already-downloaded artifacts;
    /// download nothing. Exits non-zero and lists anything missing or
    /// invalid — for CI parity with a pipeline that pre-warms the cache
    /// out of band.
    #[arg(long)]
    require: bool,

    /// Override the cache directory (defaults to the XDG-compliant cache
    /// directory for `friction`).
    #[arg(long, value_name = "DIR")]
    cache_dir: Option<PathBuf>,
}

/// Errors produced while resolving the cache directory, downloading an
/// artifact, or writing it to disk.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
enum SetupError {
    /// No cache directory could be resolved for the current platform (no
    /// resolvable home directory).
    #[error("could not resolve a cache directory for this platform (no home directory found)")]
    NoCacheDir,

    /// The cache directory (or an artifact's subdirectory within it)
    /// could not be created.
    #[error("could not create cache directory {path}: {source}")]
    CreateDir {
        /// The directory that could not be created.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },

    /// `--require` found no cached, valid copy of an artifact.
    #[error("not cached at {path}")]
    NotCached {
        /// Where the artifact was expected.
        path: PathBuf,
    },

    /// A cached or freshly-downloaded file's sha256 does not match the
    /// registry entry's pinned checksum.
    #[error("checksum mismatch for {path} (expected sha256 {expected})")]
    ChecksumMismatch {
        /// The file that failed verification.
        path: PathBuf,
        /// The checksum the registry pins for this artifact.
        expected: friction_packs::Sha256,
    },

    /// The artifact could not be downloaded.
    #[error("download of {url} failed: {source}")]
    Download {
        /// The URL that was requested.
        url: String,
        /// Underlying `ureq` error.
        #[source]
        source: Box<ureq::Error>,
    },

    /// The downloaded (or verified) bytes could not be written into the
    /// cache.
    #[error("could not write {path}: {source}")]
    Write {
        /// The file that could not be written.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
}

/// Runs `friction setup`.
pub fn run(args: &SetupArgs) -> ExitCode {
    let registry = &*friction_packs::REGISTRY;

    if registry.is_empty() {
        println!("friction setup: no downloadable artifacts are registered; nothing to do.");
        return ExitCode::SUCCESS;
    }

    let cache_dir = match resolve_cache_dir(args.cache_dir.clone()) {
        Ok(dir) => dir,
        Err(err) => {
            eprintln!("friction setup: {err}");
            return ExitCode::FAILURE;
        }
    };

    let mut missing = Vec::new();
    for artifact in registry {
        let path = artifact_path(&cache_dir, artifact);
        let result = if args.require {
            check_cached(&path, artifact)
        } else {
            ensure_cached(&path, artifact)
        };

        match result {
            Ok(Status::AlreadyCached) => {
                println!(
                    "{} {}: cached ({})",
                    artifact.name,
                    artifact.pack_version,
                    path.display()
                );
            }
            Ok(Status::Downloaded) => {
                println!(
                    "{} {}: downloaded ({})",
                    artifact.name,
                    artifact.pack_version,
                    path.display()
                );
            }
            Err(err) if args.require => {
                println!(
                    "{} {}: MISSING ({err})",
                    artifact.name, artifact.pack_version
                );
                missing.push(format!("{} {}", artifact.name, artifact.pack_version));
            }
            Err(err) => {
                eprintln!(
                    "friction setup: failed to fetch {} {}: {err}",
                    artifact.name, artifact.pack_version
                );
                return ExitCode::FAILURE;
            }
        }
    }

    if args.require && !missing.is_empty() {
        eprintln!(
            "friction setup --require: {} artifact(s) missing or invalid: {}",
            missing.len(),
            missing.join(", ")
        );
        return ExitCode::FAILURE;
    }

    ExitCode::SUCCESS
}

/// Whether an artifact was already present and valid, or had to be
/// downloaded this run.
enum Status {
    /// A valid cached copy was already present.
    AlreadyCached,
    /// No valid cached copy was present, so it was downloaded and
    /// verified.
    Downloaded,
}

/// Resolves the cache directory to use: `override_dir` if given, else the
/// platform's XDG-compliant (or platform-equivalent) cache directory for
/// `friction`. Creates it if it does not exist.
fn resolve_cache_dir(override_dir: Option<PathBuf>) -> Result<PathBuf, SetupError> {
    let dir = match override_dir {
        Some(dir) => dir,
        None => ProjectDirs::from("", "", "friction")
            .ok_or(SetupError::NoCacheDir)?
            .cache_dir()
            .to_path_buf(),
    };
    fs::create_dir_all(&dir).map_err(|source| SetupError::CreateDir {
        path: dir.clone(),
        source,
    })?;
    Ok(dir)
}

/// The path an artifact is cached at: `<cache_dir>/<name>/<pack_version>/<url's basename>`.
fn artifact_path(cache_dir: &Path, artifact: &Artifact) -> PathBuf {
    let filename = artifact
        .url
        .rsplit('/')
        .next()
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| artifact.name.as_ref());
    cache_dir
        .join(artifact.name.as_ref())
        .join(artifact.pack_version.as_ref())
        .join(filename)
}

/// Checks whether `path` already holds a byte-valid copy of `artifact`,
/// without performing any network I/O.
fn check_cached(path: &Path, artifact: &Artifact) -> Result<Status, SetupError> {
    let bytes = fs::read(path).map_err(|_source| SetupError::NotCached {
        path: path.to_path_buf(),
    })?;
    if artifact.sha256.verify(&bytes) {
        Ok(Status::AlreadyCached)
    } else {
        Err(SetupError::ChecksumMismatch {
            path: path.to_path_buf(),
            expected: artifact.sha256,
        })
    }
}

/// Ensures `path` holds a byte-valid copy of `artifact`, downloading and
/// verifying it first if the cache is empty or holds a stale/corrupt copy.
fn ensure_cached(path: &Path, artifact: &Artifact) -> Result<Status, SetupError> {
    if let Ok(bytes) = fs::read(path)
        && artifact.sha256.verify(&bytes)
    {
        return Ok(Status::AlreadyCached);
    }

    let bytes = download(artifact)?;
    if !artifact.sha256.verify(&bytes) {
        return Err(SetupError::ChecksumMismatch {
            path: path.to_path_buf(),
            expected: artifact.sha256,
        });
    }

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|source| SetupError::CreateDir {
            path: parent.to_path_buf(),
            source,
        })?;
    }

    // Write to a sibling temp file first and rename into place, so a
    // process interrupted mid-write never leaves a partial file at `path`
    // for a later idempotent run to mistake for a complete one (its
    // checksum would fail to verify anyway, but there is no reason to rely
    // on that alone).
    let mut tmp_name = path.file_name().map_or_else(
        || std::ffi::OsString::from("artifact"),
        std::ffi::OsStr::to_os_string,
    );
    tmp_name.push(".partial");
    let tmp_path = path.with_file_name(tmp_name);

    let mut file = fs::File::create(&tmp_path).map_err(|source| SetupError::Write {
        path: tmp_path.clone(),
        source,
    })?;
    file.write_all(&bytes).map_err(|source| SetupError::Write {
        path: tmp_path.clone(),
        source,
    })?;
    drop(file);
    fs::rename(&tmp_path, path).map_err(|source| SetupError::Write {
        path: path.to_path_buf(),
        source,
    })?;

    Ok(Status::Downloaded)
}

/// Downloads `artifact.url`'s exact bytes, bounding the response body to
/// slightly more than `artifact.size_bytes` so an unexpectedly huge
/// response is rejected rather than exhausting memory.
fn download(artifact: &Artifact) -> Result<Vec<u8>, SetupError> {
    let limit = artifact.size_bytes.saturating_add(4096);
    let mut response =
        ureq::get(artifact.url.as_ref())
            .call()
            .map_err(|source| SetupError::Download {
                url: artifact.url.to_string(),
                source: Box::new(source),
            })?;
    response
        .body_mut()
        .with_config()
        .limit(limit)
        .read_to_vec()
        .map_err(|source| SetupError::Download {
            url: artifact.url.to_string(),
            source: Box::new(source),
        })
}
