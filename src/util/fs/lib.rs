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

use ::std::path::{Path, PathBuf};
use ::std::fs::{self, File};
use ::std::io::{BufReader};

pub use crate::cp_mv::{cp_a, mv, Copy, Move};
mod cp_mv;

pub use crate::tempdir::{ActualTempDir, TempDir};
mod tempdir;

#[macro_use]
extern crate failure;
extern crate tempdir as tempdir_crate;
#[macro_use]
extern crate log;

use ::failure::ResultExt;

pub type FailResult<T> = Result<T, ::failure::Error>;

/// Wrapper around `File::open` that adds context.
pub fn open(path: impl AsRef<Path>) -> FailResult<File>
{
    File::open(path.as_ref())
        .with_context(|e| format!("{}: could not open file: {}", path.as_ref().display(), e))
        .map_err(Into::into)
}

/// Wrapper around `File::open` that adds context and makes a `BufReader`.
pub fn open_text(path: impl AsRef<Path>) -> FailResult<BufReader<File>>
{ open(path).map(BufReader::new) }

/// Wrapper around `File::create` that adds context.
pub fn create(path: impl AsRef<Path>) -> FailResult<File>
{
    File::create(path.as_ref())
        .with_context(|e| format!("{}: could not create file: {}", path.as_ref().display(), e))
        .map_err(Into::into)
}

/// Wrapper around `std::fs::write` that adds context.
pub fn write(path: impl AsRef<Path>, contents: impl AsRef<[u8]>) -> FailResult<()>
{
    std::fs::write(path.as_ref(), contents)
        .with_context(|e| format!("{}: could not write file: {}", path.as_ref().display(), e))
        .map_err(Into::into)
}

/// Wrapper around `std::fs::copy` that adds context.
pub fn copy(src: impl AsRef<Path>, dest: impl AsRef<Path>) -> FailResult<()>
{
    let (src, dest) = (src.as_ref(), dest.as_ref());
    fs::copy(src, dest)
        .map(|_| ()) // number of bytes; don't care
        .with_context(|e|
            format!("could not copy file '{}' to '{}': {}",
                src.display(), dest.display(), e))
        .map_err(Into::into)
}

/// Wrapper around `std::fs::create_dir` that adds context.
pub fn create_dir(dir: impl AsRef<Path>) -> FailResult<()>
{
    fs::create_dir(dir.as_ref())
        .with_context(|e| format!("{}: could not create directory: {}", dir.as_ref().display(), e))
        .map_err(Into::into)
}

/// Wrapper around `std::fs::canonicalize` that adds context.
pub fn canonicalize(dir: impl AsRef<Path>) -> FailResult<PathBuf>
{
    fs::canonicalize(dir.as_ref())
        .with_context(|e| format!("{}: could not normalize: {}", dir.as_ref().display(), e))
        .map_err(Into::into)
}

/// Canonicalizes a path where the final component need not exist.
///
/// NOTE: will behave strangely for paths that end in ..
///       due to how Path::parent is defined
pub fn canonicalize_parent(path: impl AsRef<Path>) -> FailResult<PathBuf>
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
pub fn remove_dir(dir: impl AsRef<Path>) -> FailResult<()>
{
    fs::remove_dir(dir.as_ref())
        .with_context(|e| format!("{}: could not remove directory: {}", dir.as_ref().display(), e))
        .map_err(Into::into)
}

/// Wrapper around `std::fs::remove_file` that adds context.
pub fn remove_file(dir: impl AsRef<Path>) -> FailResult<()>
{
    fs::remove_file(dir.as_ref())
        .with_context(|e| format!("{}: could not remove file: {}", dir.as_ref().display(), e))
        .map_err(Into::into)
}

// Error-chaining wrapper around `hard_link`
pub fn hard_link(src: impl AsRef<Path>, dest: impl AsRef<Path>) -> FailResult<()>
{
    let (src, dest) = (src.as_ref(), dest.as_ref());
    fs::hard_link(src, dest)
        .with_context(|e|
            format!("could not hard-link '{}' to '{}': {}",
                src.display(), dest.display(), e))
        .map_err(Into::into)
}

/// Simulates `rm -rf`.
///
/// Properties:
/// * Deletes files and folders alike.
/// * Does not require the path or its ancestors to exist.
/// * **Does** fail if other problems occur (e.g. insufficient permissions).
/// * Does **not** follow symbolic links.
pub fn rm_rf(path: impl AsRef<Path>) -> FailResult<()>
{
    use ::std::io::ErrorKind;

    let path = path.as_ref();

    // directoryness is only checked *after* failed deletion, to reduce race conditions
    match fs::remove_file(path) {
        Ok(()) => { return Ok(()); },
        Err(e) => {
            match (e.kind(), path.is_dir()) {
                (ErrorKind::NotFound, _) => { return Ok(()); },
                (ErrorKind::Other, true) => {},
                _ => bail!("{}: could not delete: {}", path.display(), e),
            }
        }
    }

    match fs::remove_dir_all(path) {
        Ok(()) => { return Ok(()); },
        Err(e) => {
            match (e.kind(), path.is_file()) {
                (ErrorKind::NotFound, _) => { return Ok(()); },
                (ErrorKind::Other, true) => {
                    bail!("{}: could not delete: we're being trolled", path.display());
                },
                _ => bail!("{}: could not delete: {}", path.display(), e),
            }
        }
    }
}
