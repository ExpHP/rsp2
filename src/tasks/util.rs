use ::std::sync::atomic::AtomicUsize;
use ::std::sync::atomic::Ordering::SeqCst;
use ::std::sync::Arc;
use ::std::fmt;

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

pub(crate) fn zip_eq<As, Bs>(a: As, b: Bs) -> ::std::iter::Zip<As::IntoIter, Bs::IntoIter>
where
    As: IntoIterator, As::IntoIter: ExactSizeIterator,
    Bs: IntoIterator, Bs::IntoIter: ExactSizeIterator,
{
    let (a, b) = (a.into_iter(), b.into_iter());
    assert_eq!(a.len(), b.len());
    a.zip(b)
}

//--------------------------------------------------------

pub(crate) fn index_of_nearest(carts: &[[f64; 3]], needle: &[f64; 3], tol: f64) -> Option<usize>
{
    use ::rsp2_array_utils::{arr_from_fn, dot};
    carts.into_iter()
        .map(|v| arr_from_fn(|k| v[k] - needle[k]))
        .map(|v: [_; 3]| dot(&v, &v))
        .enumerate()
        .filter(|&(_, sq)| sq <= tol)
        .min_by(|&(_, v1), &(_, v2)| v1.partial_cmp(&v2).expect("NaN"))
        .map(|(i, _)| i)
}

#[allow(unused)]
pub(crate) fn index_of_shortest(carts: &[[f64; 3]], tol: f64) -> Option<usize>
{ index_of_nearest(carts, &[0.0; 3], tol) }

//--------------------------------------------------------
pub(crate) use self::lockfile::{LockfilePath, LockfileGuard};
mod lockfile {
    use ::Result;
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
        pub fn try_lock(&self) -> Result<Option<LockfileGuard>> {
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
        pub fn lock(&self) -> Result<Option<LockfileGuard>> {
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
        pub fn drop(mut self) -> Result<()>
        { self._drop() }

        fn _drop(&mut self) -> Result<()>
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
