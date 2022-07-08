/* ************************************************************************ **
** This file is part of rsp2, and is licensed under EITHER the MIT license  **
** or the Apache 2.0 license, at your option.                               **
**                                                                          **
**     http://www.apache.org/licenses/LICENSE-2.0                           **
**     http://opensource.org/licenses/MIT                                   **
**                                                                          **
** Be aware that not all of rsp2 is provided under this permissive license, **
** and that the project as a whole is licensed under the GPL 3.0.           **
** ************************************************************************ */

use std::path::{Path, PathBuf};
use std::fs::{self, File};
use std::io::{BufReader};

pub use crate::cp_mv::{cp_a, mv, Copy, Move};
mod cp_mv;

pub use crate::tempdir::{ActualTempDir, TempDir};
mod tempdir;

#[macro_use] extern crate log;

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum FsError {
    #[error("{}: {}: {}", .path.display(), .message, .source)]
    OnePath {
        path: PathBuf,
        message: String,
        #[source]
        source: std::io::Error,
    },
    #[error("could not {} '{}' to '{}': {}", .verb, .src.display(), .dest.display(), .source)]
    TwoPath {
        src: PathBuf,
        dest: PathBuf,
        verb: &'static str,
        #[source]
        source: std::io::Error,
    },
    #[error("{}", .0)]
    Custom(String),
}

impl FsError {
    pub fn for_path(path: impl AsRef<Path>, msg: impl AsRef<str>, source: std::io::Error) -> Self {
        FsError::OnePath {
            path: path.as_ref().to_owned(),
            message: msg.as_ref().to_owned(),
            source,
        }
    }

    pub fn two_path(src:  impl AsRef<Path>, dest: impl AsRef<Path>, verb: &'static str, source: std::io::Error) -> Self {
        FsError::TwoPath {
            src: src.as_ref().to_owned(),
            dest: dest.as_ref().to_owned(),
            verb, source,
        }
    }
}

pub type FsResult<T> = Result<T, FsError>;

/// Wrapper around `File::open` that adds context.
pub fn open(path: impl AsRef<Path>) -> FsResult<File>
{
    File::open(path.as_ref())
        .map_err(|e| FsError::for_path(path, "could not open file", e))
}

/// Wrapper around `File::open` that adds context and makes a `BufReader`.
pub fn open_text(path: impl AsRef<Path>) -> FsResult<BufReader<File>>
{ open(path).map(BufReader::new) }

/// Wrapper around `File::create` that adds context.
pub fn create(path: impl AsRef<Path>) -> FsResult<File>
{
    File::create(path.as_ref())
        .map_err(|e| FsError::for_path(path, "could not create file", e))
}

/// Wrapper around `std::fs::write` that adds context.
pub fn write(path: impl AsRef<Path>, contents: impl AsRef<[u8]>) -> FsResult<()>
{
    std::fs::write(path.as_ref(), contents)
        .map_err(|e| FsError::for_path(path, "could not write file", e))
}

/// Wrapper around `std::fs::copy` that adds context.
pub fn copy(src: impl AsRef<Path>, dest: impl AsRef<Path>) -> FsResult<()>
{
    let (src, dest) = (src.as_ref(), dest.as_ref());
    fs::copy(src, dest)
        .map(|_| ()) // number of bytes; don't care
        .map_err(|e| FsError::two_path(src, dest, "copy", e))
}

/// Wrapper around `std::fs::create_dir` that adds context.
pub fn create_dir(dir: impl AsRef<Path>) -> FsResult<()>
{
    fs::create_dir(dir.as_ref())
        .map_err(|e| FsError::for_path(dir, "could not create directory", e))
}

/// Wrapper around `std::fs::canonicalize` that adds context.
pub fn canonicalize(dir: impl AsRef<Path>) -> FsResult<PathBuf>
{
    fs::canonicalize(dir.as_ref())
        .map_err(|e| FsError::for_path(dir, "could not normalize", e))
}

/// Canonicalizes a path where the final component need not exist.
///
/// NOTE: will behave strangely for paths that end in ..
///       due to how Path::parent is defined
pub fn canonicalize_parent(path: impl AsRef<Path>) -> FsResult<PathBuf>
{
    let path = path.as_ref();
    if path.exists() {
        return canonicalize(path);
    }
    match split(path) {
        None => Ok(path.into()),
        Some((parent, name)) => {
            canonicalize(parent).map(|p| p.join(name))
        },
    }
}

fn split(path: &Path) -> Option<(&Path, &::std::ffi::OsStr)>
{ path.file_name().map(|name| (path.parent().unwrap(), name)) }

/// Wrapper around `std::fs::remove_file` that adds context.
pub fn remove_dir(dir: impl AsRef<Path>) -> FsResult<()>
{
    fs::remove_dir(dir.as_ref())
        .map_err(|e| FsError::for_path(dir, "could not remove directory", e))
}

/// Wrapper around `std::fs::remove_file` that adds context.
pub fn remove_file(dir: impl AsRef<Path>) -> FsResult<()>
{
    fs::remove_file(dir.as_ref())
        .map_err(|e| FsError::for_path(dir, "could not remove file", e))
}

// Error-chaining wrapper around `hard_link`
pub fn hard_link(src: impl AsRef<Path>, dest: impl AsRef<Path>) -> FsResult<()>
{
    let (src, dest) = (src.as_ref(), dest.as_ref());
    fs::hard_link(src, dest)
        .map_err(|e| FsError::two_path(src, dest, "hard-link", e))
}

/// Simulates `rm -rf`.
///
/// Properties:
/// * Deletes files and folders alike.
/// * Does not require the path or its ancestors to exist.
/// * **Does** fail if other problems occur (e.g. insufficient permissions).
/// * Does **not** follow symbolic links.
pub fn rm_rf(path: impl AsRef<Path>) -> FsResult<()>
{
    use std::io::ErrorKind;

    let path = path.as_ref();

    // directoryness is only checked *after* failed deletion, to reduce race conditions
    match fs::remove_file(path) {
        Ok(()) => { return Ok(()); },
        Err(e) => {
            match (e.kind(), path.is_dir()) {
                (ErrorKind::NotFound, _) => { return Ok(()); },
                (_, true) => {},  // continue to directory code
                _ => return Err(FsError::for_path(path, "could not delete", e)),
            }
        }
    }

    match fs::remove_dir_all(path) {
        Ok(()) => Ok(()),
        Err(e) => {
            match e.kind() {
                ErrorKind::NotFound => Ok(()),
                _ => Err(FsError::for_path(path, "could not delete", e)),
            }
        }
    }
}
