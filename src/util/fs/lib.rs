use ::std::path::{Path, PathBuf};
use ::std::fs::{self, File};
use ::std::io::{self, BufReader};

pub use cp_mv::{cp_a, mv, Copy, Move};
mod cp_mv;

pub use tempdir::{ActualTempDir, TempDir};
mod tempdir;

#[macro_use]
extern crate error_chain;
extern crate tempdir as tempdir_crate;
#[macro_use]
extern crate log;

error_chain! {
    foreign_links {
        Io(io::Error);
    }
}

/// Wrapper around `File::open` that adds context.
pub fn open<P: AsRef<Path>>(path: P) -> Result<File>
{
    File::open(path.as_ref())
        .chain_err(|| format!("while opening file: '{}'", path.as_ref().display()))
}

/// Wrapper around `File::open` that adds context and makes a `BufReader`.
pub fn open_text<P: AsRef<Path>>(path: P) -> Result<BufReader<File>>
{ open(path).map(BufReader::new) }

/// Wrapper around `File::create` that adds context.
pub fn create<P: AsRef<Path>>(path: P) -> Result<File>
{
    File::create(path.as_ref())
        .chain_err(|| format!("could not create file: '{}'", path.as_ref().display()))
}

/// Wrapper around `std::fs::copy` that adds context.
pub fn copy<P: AsRef<Path>, Q: AsRef<Path>>(src: P, dest: Q) -> Result<()>
{
    let (src, dest) = (src.as_ref(), dest.as_ref());
    fs::copy(src, dest)
        .map(|_| ()) // number of bytes; don't care
        .chain_err(||
            format!("could not copy file '{}' to '{}'",
                src.display(), dest.display()))
}

/// Wrapper around `std::fs::create_dir` that adds context.
pub fn create_dir<P: AsRef<Path>>(dir: P) -> Result<()>
{
    fs::create_dir(dir.as_ref())
        .chain_err(|| format!("could not create directory '{}'", dir.as_ref().display()))
}

/// Wrapper around `std::fs::canonicalize` that adds context.
pub fn canonicalize<P: AsRef<Path>>(dir: P) -> Result<PathBuf>
{
    fs::canonicalize(dir.as_ref())
        .chain_err(|| format!("could not normalize: '{}'", dir.as_ref().display()))
}

/// Canonicalizes a path where the final component need not exist.
///
/// NOTE: will behave strangely for paths that end in ..
///       due to how Path::parent is defined
pub fn canonicalize_parent<P: AsRef<Path>>(path: P) -> Result<PathBuf>
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
pub fn remove_dir<P: AsRef<Path>>(dir: P) -> Result<()>
{
    fs::remove_dir(dir.as_ref())
        .chain_err(|| format!("could not remove directory: '{}'", dir.as_ref().display()))
}

/// Wrapper around `std::fs::remove_file` that adds context.
pub fn remove_file<P: AsRef<Path>>(dir: P) -> Result<()>
{
    fs::remove_file(dir.as_ref())
        .chain_err(|| format!("could not remove file: '{}'", dir.as_ref().display()))
}

// Error-chaining wrapper around `hard_link`
pub fn hard_link<P: AsRef<Path>, Q: AsRef<Path>>(src: P, dest: Q) -> Result<()>
{
    let (src, dest) = (src.as_ref(), dest.as_ref());
    fs::hard_link(src, dest)
        .chain_err(||
            format!("could not hard-link '{}' to '{}'",
                src.display(), dest.display()))
}

/// Simulates `rm -rf`.
///
/// Properties:
/// * Deletes files and folders alike.
/// * Does not require the path or its ancestors to exist.
/// * **Does** fail if other problems occur (e.g. insufficient permissions).
/// * Does **not** follow symbolic links.
pub fn rm_rf<P: AsRef<Path>>(path: P) -> Result<()>
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
                _ => return Err(e).chain_err(|| format!("could not delete: {}", path.display())),
            }
        }
    }

    match fs::remove_dir_all(path) {
        Ok(()) => { return Ok(()); },
        Err(e) => {
            match (e.kind(), path.is_file()) {
                (ErrorKind::NotFound, _) => { return Ok(()); },
                (ErrorKind::Other, true) => {
                    bail!("could not delete '{}'; we're being trolled", path.display());
                },
                _ => return Err(e).chain_err(|| format!("could not delete: {}", path.display())),
            }
        }
    }
}
