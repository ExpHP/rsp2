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

use crate::FailResult;
use std::path::Path;
use std::path::PathBuf;
use path_abs::{FileRead, FileWrite};

/// AsRef<Path> with more general impls on smart pointer types.
///
/// (for instance, `Box<AsPath>` and `Rc<TempDir>` both implement the trait)
pub trait AsPath {
    fn as_path(&self) -> &Path;

    fn to_path_buf(&self) -> PathBuf
    { self.as_path().to_path_buf() }

    fn join(&self, path: impl AsPath) -> PathBuf
    where Self: Sized, // generic functions are not object-safe
    { self.as_path().join(path.as_path()) }

//    fn exists(&self) -> bool
//    { self.as_path().exists() }
}

pub trait DirLike: AsPath {
    fn create_file(&self, name: &Path) -> FailResult<FileWrite>
    { Ok(FileWrite::create(self.as_path().join(name))?) }

    fn open(&self, name: &Path) -> FailResult<FileRead>
    { Ok(FileRead::open(self.as_path().join(name))?) }

    fn append_file(&self, name: &Path) -> FailResult<FileWrite>
    { Ok(FileWrite::open_append(self.as_path().join(name.as_path()))?) }

    // // A form of 'join' where path is verified to be a relative path.
    // // (however, it makes no further guarantee that `self.join(path)`
    // // points to a location inside `self`)
    // fn _descend(&self, path: &Path) -> Result<PathArc>
    // {
    //     if path.is_absolute() {
    //         bail!("Did not expect an absolute path: {}", path.display());
    //     }
    //     PathArc::new(self.as_path().join(path))
    // }

//    fn exists(&self) -> bool
//    { self.as_path().exists() }
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
    (by AsRef) [] std::path::Path;
    (by Deref) [] std::path::PathBuf;
    (by AsRef) [] rsp2_fs_util::ActualTempDir;
    (by AsRef) [] rsp2_fs_util::TempDir;
    (by AsRef) [] std::ffi::OsString;
    (by AsRef) [] std::ffi::OsStr;
    (by AsRef) [] str;
    (by AsRef) [] String;
    (by AsRef) ['a] std::path::Iter<'a>;
    (by AsRef) ['a] std::path::Components<'a>;
    (by Deref) ['p, P: AsPath + ?Sized] &'p mut P;
    (by Deref) [P: AsPath + ?Sized] Box<P>;
    (by Deref) [P: AsPath + ?Sized] std::rc::Rc<P>;
    (by Deref) [P: AsPath + ?Sized] std::sync::Arc<P>;
    (by Deref) ['p, P: AsPath + ToOwned + ?Sized] std::borrow::Cow<'p, P>;
    (by AsRef) [] path_abs::PathAbs;
    (by AsRef) [] path_abs::PathFile;
    (by AsRef) [] path_abs::PathDir;
}

impl<'p, P: AsPath + ?Sized> AsPath for &'p P
{ fn as_path(&self) -> &Path { P::as_path(self) } }

