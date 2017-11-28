use ::std::path::{Path, PathBuf};
use ::std::io::Result as IoResult;


/// RAII type to temporarily enter a directory.
///
/// The recommended usage is actually not to rely on the implicit destructor
/// (which panics on failure), but to instead explicitly call `.pop()`.
/// The advantage of doing so over just manually calling 'set_current_dir'
/// is the unused variable lint can help remind you to call `pop`.
///
/// Usage is highly discouraged in multithreaded contexts where
/// another thread may need to access the filesystem.
#[must_use]
pub struct PushDir(Option<PathBuf>);
pub fn push_dir<P: AsRef<Path>>(path: P) -> IoResult<PushDir> {
    let old = ::std::env::current_dir()?;
    ::std::env::set_current_dir(path)?;
    Ok(PushDir(Some(old)))
}

impl PushDir {
    /// Explicitly destroy the PushDir.
    ///
    /// This lets you handle the IO error, and has an advantage over
    /// manual calls to 'env::set_current_dir' in that the compiler will
    pub fn pop(mut self) -> IoResult<()> {
        ::std::env::set_current_dir(self.0.take().unwrap())
    }
}

impl Drop for PushDir {
    fn drop(&mut self) {
        if let Some(d) = self.0.take() {
            if let Err(e) = ::std::env::set_current_dir(d) {
                // uh oh.
                panic!("automatic popdir failed: {}", e);
            }
        }
    }
}


pub(crate) fn zip_eq<As, Bs>(a: As, b: Bs) -> ::std::iter::Zip<As::IntoIter, Bs::IntoIter>
    where
        As: IntoIterator, As::IntoIter: ExactSizeIterator,
        Bs: IntoIterator, Bs::IntoIter: ExactSizeIterator,
{
    let (a, b) = (a.into_iter(), b.into_iter());
    assert_eq!(a.len(), b.len());
    a.zip(b)
}
