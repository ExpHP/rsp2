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
#![feature(rust_2018_preview)]

#[macro_use] mod macros;
mod functional;
mod iter;
#[cfg(test)] mod test_util;

// Functional operations on arrays.
pub use ::functional::{map_arr, try_map_arr, opt_map_arr};
pub use ::functional::{arr_from_fn, try_arr_from_fn, opt_arr_from_fn};

// NOTE: There is deliberately no fold operation.  You can iterate over arrays.
//       We even have a move iter!

// Iterators on arrays
pub use ::iter::ArrayMoveIterExt;

mod traits {
    /// Marker trait for a built-in array type, i.e. `[T; n]`.
    ///
    /// This is a workaround for the current lack of generic constants in rust.
    /// Its methods are arbitrarily chosen to meet the needs of internal implementations
    /// and honestly it would be best if you just forgot that this even exists. Capiche?
    ///
    /// It is unsafe so that various properties can be trusted by unsafe code.
    /// For instance, `align_of::<Self>() == align_of<Self::Element>()` and
    /// `size_of::<Self>() == Self::array_len() * size_of<Self::Element>().
    ///
    /// The complete set of properties relied on by unsafe code is unspecified
    /// and may change; Please don't implement this for your own types.
    pub unsafe trait IsArray: Sized
    {
        /// `T` from the array type `[T; n]`.
        type Element;

        /// `n` from the array type `[T; n]`.
        fn array_len() -> usize;
        /// Perform the `&[T; n] -> &[T]` coercion.
        fn array_as_slice(&self) -> &[Self::Element];
        /// Perform the `&mut [T; n] -> &mut [T]` coercion.
        fn array_as_mut_slice(&mut self) -> &mut [Self::Element];
    }

    /// A poor-man's type family that can be used to construct
    /// the type `[B; n]` from `[A; n]`.
    pub trait WithElement<B>: IsArray {
        /// The type `[B; n]`.
        type Type: IsArray<Element=B> + WithElement<Self::Element, Type=Self>;
    }

    macro_rules! impl_is_array {
        {$n:expr} => {
            unsafe impl<T> IsArray for [T; $n] {
                type Element = T;

                #[inline(always)] fn array_len() -> usize { $n }
                #[inline(always)] fn array_as_slice(&self) -> &[T] { self }
                #[inline(always)] fn array_as_mut_slice(&mut self) -> &mut [T] { self }
            }

            impl<A, B> WithElement<B> for [A; $n] {
                type Type = [B; $n];
            }
        };
    }

    each_array_size!{ impl_is_array!{0...32} }
}
