//! Shim around the `tempdir` crate which doesn't delete the
//! directories on unwind, in order to facilitate debugging.
extern crate tempdir;
extern crate rsp2_fs_util as fsx;
#[macro_use]
extern crate log;

pub use tempdir::TempDir as ActualTempDir;

use ::std::io::Result;
use ::std::path::{Path, PathBuf};
use ::std::ffi::{OsStr, OsString};

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
            let temp = match self.0.take() {
                Some(temp) => temp.into_path(),
                None => {
                    error!("A TempDir was double-dropped during panic");
                    return; // avoid double-panic
                },
            };
            let temp: &Path = temp.as_ref();

            // Either move the directory to a user-specified location
            // or else leak it in its current location.

            let dest = match non_empty_env("RSP2_SAVETEMP") {
                None => {
                    info!("successfully leaked tempdir at {}", temp.display());
                    return;
                },
                Some(env_dest) => match ::std::env::current_dir() {
                    Err(e) => {
                        warn!("could not read current directory during panic: {}", e);
                        return; // avoid double-panic
                    },
                    Ok(cwd) => cwd.join(env_dest),
                },
            };
            let dest: &Path = dest.as_ref();

            let name = match temp.file_name() {
                None => {
                    warn!("could not gettemp dir name during panic");
                    return; // avoid double-panic
                },
                Some(path) => path,
            };

            // create $RSP2_SAVETEMP if it doesn't exist yet.
            if let Err(e) = ::std::fs::create_dir(dest) {
                if !dest.exists() {
                    warn!("failed to create '{}' during panic: {}", dest.display(), e);
                    return;
                }
            }

            let dest_file = dest.join(name);
            let dest_file: &Path = dest_file.as_ref();

            match ::fsx::mv(temp, dest_file) {
                Err(e) => warn!("failed to move during panic: from '{}' to '{}': {}", temp.display(), dest_file.display(), e),
                Ok(_) => info!("recovered tempdir during panic: {}", dest_file.display()),
            }
        }
    }
}

fn non_empty_env<S: AsRef<OsStr>>(key: S) -> Option<OsString> {
    match ::std::env::var_os(key) {
        None => None,
        Some(s) => match s.is_empty() {
            true => None,
            false => Some(s),
        }
    }
}
