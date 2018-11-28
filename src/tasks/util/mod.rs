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

use ::std::sync::atomic::AtomicUsize;
use ::std::sync::atomic::Ordering::SeqCst;
use ::std::sync::Arc;
use ::std::fmt;
use ::rsp2_array_types::{V3};

pub use self::rayon_cond::CondIterator;
mod rayon_cond;

//--------------------------------------------------------

#[derive(Clone)]
pub(crate) struct AtomicCounter(Arc<AtomicUsize>);

impl AtomicCounter {
    pub fn new() -> Self { AtomicCounter(Arc::new(AtomicUsize::new(0))) }
    pub fn get(&self) -> usize { self.0.load(SeqCst) }
    pub fn inc(&self) -> usize { self.0.fetch_add(1, SeqCst) }
}

impl fmt::Display for AtomicCounter {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        <usize as fmt::Display>::fmt(&self.get(), f)
    }
}

//--------------------------------------------------------

pub(crate) fn transpose_iter_to_vec<Tss, Ts, T>(input: Tss) -> Vec<Vec<T>>
where
    Tss: IntoIterator<Item=Ts>,
    Ts: ExactSizeIterator<Item=T>,
{
    let mut input = input.into_iter();
    let mut out: Vec<_> = input.next()
        .expect("can't take transpose with no rows") // width is degenerate
        .map(|x| vec![x]).collect();

    for row in input {
        for (dest, x) in zip_eq!(&mut out, row) {
            dest.push(x);
        }
    }
    out
}

//--------------------------------------------------------

pub(crate) fn index_of_nearest(carts: &[V3], needle: V3, tol: f64) -> Option<usize>
{
    carts.into_iter()
        .map(|v| (v - needle).sqnorm())
        .enumerate()
        .filter(|&(_, sq)| sq <= tol)
        .min_by(|&(_, v1), &(_, v2)| v1.partial_cmp(&v2).expect("NaN"))
        .map(|(i, _)| i)
}

#[allow(unused)]
pub(crate) fn index_of_shortest(carts: &[V3], tol: f64) -> Option<usize>
{ index_of_nearest(carts, V3([0.0; 3]), tol) }

//--------------------------------------------------------

pub(crate) use self::lockfile::{LockfilePath, LockfileGuard};
mod lockfile {
    use crate::FailResult;
    use ::std::fs::{OpenOptions};
    use ::std::io;
    use ::path_abs::{PathArc, PathFile, FileWrite};

    /// Handle with methods for creating a lockfile without race conditions.
    #[derive(Debug, Clone)]
    pub struct LockfilePath(pub PathArc);

    /// RAII guard for a lockfile.
    #[derive(Debug)]
    pub struct LockfileGuard(PathFile);

    #[allow(dead_code)]
    impl LockfilePath {
        pub fn try_lock(&self) -> FailResult<Option<LockfileGuard>> {
            // 'create_new' is the magic sauce for avoiding race conditions
            let mut options = OpenOptions::new();
            options.write(true);
            options.create_new(true);

            match FileWrite::open(&self.0, options) {
                Err(e) => {
                    match e.io_error().kind() {
                        io::ErrorKind::AlreadyExists => Ok(None),
                        _ => bail!(e),
                    }
                },
                Ok(_) => Ok(Some(LockfileGuard(self.0.canonicalize()?.into_file()?))),
            }
        }

        /// Retries until locking is successful. This could deadlock.
        pub fn lock(&self) -> FailResult<Option<LockfileGuard>> {
            let mut lock = self.try_lock()?;
            while lock.is_none() {
                ::std::thread::sleep(Default::default());
                lock = self.try_lock()?;
            }
            Ok(lock)
        }
    }

    impl ::std::ops::Deref for LockfilePath {
        type Target = PathArc;
        fn deref(&self) -> &PathArc { &self.0 }
    }

    #[allow(dead_code)]
    impl LockfileGuard {
        pub fn drop(mut self) -> FailResult<()>
        { self._drop() }

        fn _drop(&mut self) -> FailResult<()>
        {
            // clone because goddammit path_abs
            self.0.clone().remove().map_err(Into::into)
        }
    }

    impl Drop for LockfileGuard {
        fn drop(&mut self) {
            let _ = self._drop();
        }
    }
}

//--------------------------------------------------------

pub trait VeclikeIterator: Iterator + ExactSizeIterator + DoubleEndedIterator + ::std::iter::FusedIterator {}
impl<I> VeclikeIterator for I
    where I: Iterator + ExactSizeIterator + DoubleEndedIterator + ::std::iter::FusedIterator {}

//--------------------------------------------------------

pub mod ext_traits {
    use ::path_abs::PathDir;
    use ::std::path::Path;
    use ::std::fmt;

    extension_trait!{
        <'a> pub ArgMatchesExt<'a> for ::clap::ArgMatches<'a> {
            // For when the value ought to exist because it was 'required(true)'
            // (and therefore clap would have panicked if it were missing)
            fn expect_value_of(&self, s: &str) -> String
            { self.value_of(s).unwrap_or_else(|| panic!("BUG! ({} was required)", s)).into() }

            fn expect_values_of(&self, s: &str) -> Vec<String>
            { self.values_of(s).unwrap_or_else(|| panic!("BUG! ({} was required)", s)).map(Into::into).collect() }
        }
    }

    extension_trait! {
        pub <T, E> OptionResultExt<T, E> for Option<Result<T, E>> {
            fn fold_ok(self) -> Result<Option<T>, E> {
                self.map_or(Ok(None), |r| r.map(Some))
            }
        }
    }

    extension_trait! {
        pub <T: fmt::Debug> OptionExpectNoneExt<T> for Option<T> {
            fn expect_none(self, msg: &str) {
                if self.is_some() {
                    panic!("expect_none on {:?}: {}", self, msg);
                }
            }
        }
    }

    extension_trait! {
        pub <A: AsRef<Path>> PathNiceExt for A {
            // make a path "nice" for display, *if possible*
            fn nice(&self) -> String {
                self.nice_or_bust()
                    .unwrap_or_else(|| format!("{}", self.as_ref().display()))
            }

            fn nice_or_bust(&self) -> Option<String> {
                let cwd = PathDir::current_dir().ok()?;
                let absolute = cwd.join(self.as_ref());

                // (just bail if it's not a child. "../../../other/place" would hardly be nice.)
                let relative = absolute.as_path().strip_prefix(&cwd).ok()?;
                Some(format!("{}", relative.display()))
            }
        }
    }
}
