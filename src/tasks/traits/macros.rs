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

#[macro_export]
macro_rules! impl_dirlike_boilerplate {
    (
        type: {$Type:ident<_>}
        member: self.$member:ident
        other_members: [$(self.$other_members:ident),*]
    ) => {
        // all dir-likes implement HasTempDir if possible
        #[allow(dead_code)]
        impl<P: HasTempDir> HasTempDir for $Type<P> {
            fn temp_dir_close(self) -> IoResult<()>
            { self.$member.temp_dir_close() }

            fn temp_dir_into_path(self) -> PathBuf
            { self.$member.temp_dir_into_path() }

            fn temp_dir_recover(self)
            { self.$member.temp_dir_recover() }
        }

        // all dir-likes implement AsPath
        impl<P: AsPath> AsPath for $Type<P> {
            fn as_path(&self) -> &Path { self.dir.as_path() }
        }

        // all dir-likes expose inherent methods that are aliases
        // for the HasTempDir and AsPath methods
        #[allow(dead_code)]
        impl<P: HasTempDir> $Type<P> {
            /// Explicitly close the temporary directory, deleting it.
            ///
            /// This also happens when the object is dropped, but in that
            /// case it is not possible to detect errors.
            pub fn close(self) -> IoResult<()> { self.temp_dir_close() }

            /// Move the directory to the given path, which must not exist.
            ///
            /// Currently, there is no recourse if the operation fails;
            /// the directory is simply lost. In the future, this may take
            /// '&mut self' and poison the object once the move has succeeded.
            pub fn relocate(self, path: impl AsPath)
            -> FailResult<$Type<PathBuf>>
            {Ok({
                // (use something that supports cross-filesystem moves)
                rsp2_fs_util::mv(self.path(), path.as_path())?;

                self.map_dir(|old| {
                    // forget the TempDir
                    let _ = old.temp_dir_into_path();
                    // store the new path
                    path.as_path().to_owned()
                })
            })}

            /// Recover the tempdir if a closure returns Err.
            pub fn try_with_recovery<B, E, F>(self, f: F) -> Result<(Self, B), E>
            where F: FnOnce(&Self) -> Result<B, E>,
            { self.temp_dir_try_with_recovery(f) }
        }

        #[allow(dead_code)]
        impl<P: AsPath> $Type<P> {
            pub fn path(&self) -> &Path { self.as_path() }

            /// Apply a function to change the type of the directory.
            /// For example, when `P = TempDir`, one could use `.map_dir(Rc::new)`
            ///  to enable cloning of the object.
            pub fn map_dir<Q, F>(self, f: F) -> $Type<Q>
            where
                Q: AsPath,
                F: FnOnce(P) -> Q,
            {
                let $member = f(self.$member);
                $(let $other_members = self.$other_members;)*
                $Type { $member, $($other_members),* }
            }

            /// Box the path, erasing its type.
            pub fn boxed(self) -> $Type<Box<dyn AsPath>>
            where P: 'static,
            { self.map_dir(move |d| Box::new(d) as _) }
        }
    };
}
