use ::std::path::Path;
use ::std::io::Result as IoResult;
use ::std::path::PathBuf;
use ::rsp2_tempdir::TempDir;

/// AsRef<Path> with more general impls on smart pointer types.
///
/// (for instance, `Box<AsPath>` and `Rc<TempDir>` both implement
///  the trait)
pub trait AsPath {
    fn as_path(&self) -> &Path;
}

macro_rules! as_path_impl {
    (@AsRef [$($generics:tt)*] $Type:ty)
    => {
        impl<$($generics)*> AsPath for $Type {
            fn as_path(&self) -> &Path { self.as_ref() }
        }
    };
    (@Deref [$($generics:tt)*] $Type:ty)
    => {
        impl<$($generics)*> AsPath for $Type {
            fn as_path(&self) -> &Path { (&**self).as_path() }
        }
    };
    ( $(
        (by $tag:tt) [$($generics:tt)*] $Type:ty;
    )+ )
    => {
        $( as_path_impl!{@$tag [$($generics)*] $Type} )*
    };
}

as_path_impl!{
    (by AsRef) [] ::std::path::Path;
    (by AsRef) [] ::std::path::PathBuf;
    (by AsRef) [] ::rsp2_tempdir::TempDir;
    (by AsRef) [] ::std::ffi::OsString;
    (by AsRef) [] ::std::ffi::OsStr;
    (by AsRef) [] str;
    (by AsRef) [] String;
    (by AsRef) ['a] ::std::path::Iter<'a>;
    (by AsRef) ['a] ::std::path::Components<'a>;
    (by Deref) ['p, P: AsPath + ?Sized] &'p mut P;
    (by Deref) [P: AsPath + ?Sized] Box<P>;
    (by Deref) [P: AsPath + ?Sized] ::std::rc::Rc<P>;
    (by Deref) [P: AsPath + ?Sized] ::std::sync::Arc<P>;
    (by Deref) ['p, P: AsPath + ToOwned + ?Sized] ::std::borrow::Cow<'p, P>;
}

impl<'p, P: AsPath + ?Sized> AsPath for &'p P
{ fn as_path(&self) -> &Path { P::as_path(self) } }

/// Trait for types that own a temporary directory, which can be
/// released (to prevent automatic deletion) or explicitly closed
/// to catch IO errors (which would be ignored on drop).
///
/// This is really just an implementation detail, and you should not
/// worry about it. All types that implement this expose it through
/// the `close()` and `into_path()` inherent methods, so you do not
/// need to import it.
pub trait HasTempDir: AsPath {
    /// Provides `close()` in generic contexts
    fn temp_dir_close(self) -> IoResult<()>;
    /// Provides `into_path()` in generic contexts
    fn temp_dir_into_path(self) -> PathBuf;
}

impl HasTempDir for TempDir {
    fn temp_dir_close(self) -> IoResult<()> { self.close() }
    fn temp_dir_into_path(self) -> PathBuf { self.into_path() }
}
