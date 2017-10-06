//! Shim around the `tempdir` crate which doesn't delete the
//! directories on unwind, in order to facilitate debugging.
extern crate tempdir;
pub use tempdir::TempDir as ActualTempDir;

use ::std::io::Result;
use ::std::path::{Path, PathBuf};

/// Wrapper around `tempdir::TempDir` that does not destroy the directory on unwind.
#[derive(Debug)]
pub struct TempDir(Option<ActualTempDir>);

impl From<ActualTempDir> for TempDir {
    fn from(tmp: ActualTempDir) -> Self { TempDir(Some(tmp)) }
}

/// Forward everything to the tempdir crate.
impl TempDir {
    pub fn new(prefix: &str) -> Result<TempDir> {
        ActualTempDir::new(prefix).map(Self::from)
    }

    pub fn new_in<P: AsRef<Path>>(tmpdir: P, prefix: &str) -> Result<TempDir> {
        ActualTempDir::new_in(tmpdir, prefix).map(Self::from)
    }

    pub fn path(&self) -> &Path { self.0.as_ref().unwrap().path() }
    pub fn into_path(mut self) -> PathBuf { self.0.take().unwrap().into_path() }
    pub fn close(mut self) -> Result<()> { self.0.take().unwrap().close() }
}

impl AsRef<Path> for TempDir {
    fn as_ref(&self) -> &Path { self.0.as_ref().unwrap().as_ref() }
}

/// Leaks the inner TempDir if we are unwinding.
impl Drop for TempDir {
    fn drop(&mut self) {
        if ::std::thread::panicking() {
            ::std::mem::forget(self.0.take())
        }
    }
}
