use ::{Result, ResultExt};

use ::std::path::Path;
use ::std::fs;
use ::std::fs::File;
use ::std::io::BufReader;

pub(crate) fn open<P: AsRef<Path>>(path: P) -> Result<File>
{
    File::open(path.as_ref())
        .chain_err(|| format!("Could not open file: '{}'", path.as_ref().display()))
}

pub(crate) fn create<P: AsRef<Path>>(path: P) -> Result<File>
{
    File::create(path.as_ref())
        .chain_err(|| format!("Could not create file: '{}'", path.as_ref().display()))
}

pub(crate) fn open_text<P: AsRef<Path>>(path: P) -> Result<BufReader<File>>
{ open(path).map(BufReader::new) }

pub(crate) fn copy<P: AsRef<Path>, Q: AsRef<Path>>(src: P, dest: Q) -> Result<()>
{
    let (src, dest) = (src.as_ref(), dest.as_ref());
    fs::copy(src, dest)
        .map(|_| ()) // number of bytes; don't care
        .chain_err(||
            format!("while copying '{}' to '{}'",
                src.display(), dest.display()))
}

// Error-chaining wrapper around `hard_link`
pub(crate) fn link_or_bust<P: AsRef<Path>, Q: AsRef<Path>>(src: P, dest: Q) -> Result<()>
{
    let (src, dest) = (src.as_ref(), dest.as_ref());
    fs::hard_link(src, dest)
        .chain_err(||
            format!("while linking '{}' to '{}'",
                src.display(), dest.display()))
}

// More useful wrapper around `hard_link` which:
// - falls back to copying if the destination is on another filesystem.
// - does not fail if the destination already exists
//
// Returns 'false' if the file already existed.
pub(crate) fn smart_link<P: AsRef<Path>, Q: AsRef<Path>>(src: P, dest: Q) -> Result<bool>
{
    let (src, dest) = (src.as_ref(), dest.as_ref());
    link_or_bust(src, dest)
        .map(|_| true)
        .or_else(|_| Ok({
            // if the file already existed, the link will have failed.
            // Check this before continuing because we don't want to
            //   potentially overwrite a link with a copy.
            if dest.exists() {
                return Ok(false);
            }

            // assume the error was due to being on a different filesystem.
            // (Even if not, we will probably just get the same error)
            copy(src, dest)?;
            true
        }))
}
