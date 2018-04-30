/// For deriving Save and Load through
#[macro_export]
macro_rules! derive_filetype_wrapper {
    (impl$par:tt [$($Trait:ty),+] for $Ty:ty as $Wrap:ident $bnd:tt)
    => {
        $(
            derive_filetype_wrapper!{
                @one($Wrap, $Trait) impl$par _ for $Ty $bnd
            }
        )+
    };

    (@one($Wrap:ident, Load) impl[$($par:tt)*] _ for $Ty:ty [where $($bnd:tt)*])
    => {
        impl<$($par)*> Load for $Ty where $($bnd)* {
            fn load<P: ::AsPath>(path: P) -> ::FailResult<Self>
            { Load::load().map(|$Wrap(x)| x) }
        }
    };

    (@one($Wrap:ident, Save) impl[$($par:tt)*] _ for $Ty:ty [where $($bnd:tt)*])
    => {
        impl<$($par)*> Save for $Ty where $($bnd)* {
            fn save<P: ::AsPath>(&self, path: P) -> ::FailResult<Self>
            { $Wrap::wrap_ref(self).save(path) }
        }
    };
}

/// Macro to use when implementing `alternate::{Fn, FnMut, FnOnce}`.
///
/// This macro will automatically generate the impls for the traits
/// lower on the Fn hierarchy.
#[macro_export]
macro_rules! derive_alternate_fn {
    // FnOnce
    (
        impl[$($par:tt)*] FnOnce<$Arg:ty> for $Type:ty
        $([where $($bnd:tt)*])*
        { $($fn_once_body:tt)* }
    )
    => {
        impl<$($par)*> $crate::alternate::FnOnce<$Arg> for $Type $(where $($bnd)*)*
        { $($fn_once_body)* }
    };

    // FnMut (+ FnOnce)
    (
        impl[$($par:tt)*] FnMut<$Arg:ty> for $Type:ty
        $([where $($bnd:tt)*])*
        {
            // explicitly captured so it can be moved to the FnOnce body
            type Output = $Output:ty;
            $($fn_mut_body:tt)*
        }
    )
    => {
        impl<$($par)*> $crate::alternate::FnMut<$Arg> for $Type $(where $($bnd)*)*
        {
            $($fn_mut_body)*
        }

        impl<$($par)*> $crate::alternate::FnOnce<$Arg> for $Type $(where $($bnd)*)*
        {
            type Output = $Output;
            fn call_once(mut self, args: $Args) -> Self::Output
            { $crate::alternate::FnMut::call(&mut self, args) }
        }
    };

    // Fn (+ FnMut + FnOnce)
    (
        impl[$($par:tt)*] Fn<$Arg:ty> for $Type:ty
        $([where $($bnd:tt)*])*
        {
            // explicitly captured so it can be moved to the FnOnce body
            type Output = $Output:ty;
            $($fn_body:tt)*
        }
    )
    => {
        impl<$($par)*> $crate::alternate::Fn<$Arg> for $Type $(where $($bnd)*)*
        {
            $($fn_body)*
        }

        impl<$($par)*> $crate::alternate::FnMut<$Arg> for $Type $(where $($bnd)*)*
        {
            fn call_mut(&mut self, args: $Arg) -> Self::Output
            { $crate::alternate::Fn::call(self, args) }
        }

        impl<$($par)*> $crate::alternate::FnOnce<$Arg> for $Type $(where $($bnd)*)*
        {
            type Output = $Output;
            fn call_once(self, args: $Arg) -> Self::Output
            { $crate::alternate::Fn::call(&self, args) }
        }
    };
}

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

            /// Convert into a PathBuf, disabling this object's destructor.
            ///
            /// To retain the ability to call the other methods on this type,
            /// see the `keep()` method.
            pub fn into_path(self) -> PathBuf { self.temp_dir_into_path() }

            /// Move the directory to the given path, which must not exist.
            ///
            /// Currently, there is no recourse if the operation fails;
            /// the directory is simply lost. In the future, this may take
            /// '&mut self' and poison the object once the move has succeeded.
            pub fn relocate<Q: AsPath>(self, path: Q)
            -> FailResult<$Type<PathBuf>>
            {Ok({
                // (use something that supports cross-filesystem moves)
                mv(self.path(), path.as_path())?;

                self.map_dir(|old| {
                    // forget the TempDir
                    let _ = old.temp_dir_into_path();
                    // store the new path
                    path.as_path().to_owned()
                })
            })}
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
            pub fn boxed(self) -> $Type<Box<AsPath>>
            where P: 'static,
            { self.map_dir(move |d| Box::new(d) as _) }
        }
    };
}
