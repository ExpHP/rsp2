use ::std::path::{Path, PathBuf};
use ::std::io::Result as IoResult;

/// Trait for wrapping newtypes that might be behind pointers.
///
/// This makes it easier to forward the implementation of a trait
/// method that takes `&self`.
///
/// # Safety
/// This trait has default method implementations which unsafely
/// cast pointers such as `Type<&T>` to `&Type<T>`.
///
/// Implementing this for anything but a newtype (`type Newtype<T>(pub T);`)
/// is **strongly inadvisable.**  The following are the *known* possible
/// sources of danger:
///
/// * It may be possible to synthesize invalid lifetimes or types.
/// * Behavior is undefined if the size (or worse, the sizedness)
///   of `T` and `Self` do not match.
pub(crate) unsafe trait IsNewtype<T: ?Sized> {
    /// The inverse of `&self.0` for a newtype.
    fn wrap_ref(x: &T) -> &Self
    { unsafe { transmute_copy_ref(x) } }

    /// The inverse of `&mut self.0` for a newtype.
    fn wrap_mut(x: &mut T) -> &mut Self
    { unsafe { transmute_copy_mut(x) } }
}

/// `transmute_copy` restricted to references.
///
/// Behavior is undefined if A and B differ in sizedness.
/// (and of course, plenty of other undefined behavior is possible
///  depending on what type B is...)
unsafe fn transmute_copy_ref<A: ?Sized, B: ?Sized>(borrow: &A) -> &B
{
    #![allow(unused_unsafe)]
    let ptr = borrow as *const A;
    // we can't just cast ptr to *const B because they may differ in fatness
    let ptrptr = (&ptr) as *const *const A as *const *const B;
    let ptr = unsafe { *ptrptr };
    unsafe { ptr.as_ref().unwrap() }
}

/// `transmute_copy` restricted to mutable references.
///
/// Behavior is undefined if A and B differ in sizedness.
/// (and of course, plenty of other undefined behavior is possible
///  depending on what type B is...)
unsafe fn transmute_copy_mut<A: ?Sized, B: ?Sized>(borrow: &mut A) -> &mut B
{
    #![allow(unused_unsafe)]
    let ptr = borrow as *mut A;
    // we can't just cast ptr to *mut B because they may differ in fatness
    let ptrptr = (&ptr) as *const *mut A as *const *mut B;
    let ptr = unsafe { *ptrptr };
    unsafe { ptr.as_mut().unwrap() }
}

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
