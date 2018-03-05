// NOTE: This is really part of rsp2-array-types.

//! Small fixed-size matrix types, compatible with `V2`/`V3`/`V4`
//!
//! This library primarily uses a row-based formalism; matrices are conceptually
//! understood to be containers of row-vectors. This formalism is most useful when
//! most vectors used are row vectors (in which case most matrix-vector multiplication
//! has the matrix on the right)

use ::traits::{Semiring, Ring, Field};
use ::traits::internal::{PrimitiveSemiring, PrimitiveRing, PrimitiveFloat};
use super::types::*;
use super::{Unvee, Envee};

// Some math functions are delegated to older utilities in this crate
use ::small_mat::MatrixDeterminantExt as DetImpl;
use ::small_mat::MatrixInverseExt as InvImpl;

// ---------------------------------------------------------------------------
// ------------------------------ PUBLIC API ---------------------------------

/// Construct a matrix from a function on indices.
#[inline(always)]
pub fn from_fn<M: FromFn<F>, B, F>(f: F) -> M
where F: FnMut(usize, usize) -> B,
{ FromFn::from_fn(f) }

/// Construct a matrix from a 2D array (of rows).
#[inline(always)]
pub fn from_array<A: IntoMatrix>(arr: A) -> A::Matrix
{ arr.into_matrix() }

/// Construct an identity matrix.
#[inline(always)]
pub fn eye<M: Eye>() -> M
{ Eye::eye() }

gen_each!{
    @{Mnn_Mn_Vn_n}
    impl_square_inherent_wrappers!(
        {$Mnn:ident $Mn:ident $Vn:ident $n:tt}
    ) => {
        impl<X> $Mnn<X> {
            /// Matrix inverse.
            #[inline(always)]
            pub fn inv(&self) -> Self
            where Self: Inv,
            { <Self as Inv>::inv(self) }

            /// Matrix determinant.
            #[inline(always)]
            pub fn det(&self) -> DetT<Self>
            where Self: Det,
            { <Self as Det>::det(self) }
        }
    }
}

gen_each!{
    @{Mn_n}
    impl_general_inherent_wrappers!(
        {$Mr:ident $r:tt}
    ) => {
        impl<V> $Mr<V> {
            /// Matrix transpose. (does not conjugate)
            #[inline(always)]
            pub fn t(&self) -> TransposeT<Self>
            where Self: Transpose,
            { <Self as Transpose>::t(self) }

            /// Cast into a plain `[[T; m]; n]`.
            #[inline(always)]
            pub fn into_array(self) -> ArrayT<Self>
            where Self: IntoArray,
            { <Self as IntoArray>::into_array(self) }

            /// Cast into a plain `&[[T; m]; n]`.
            #[inline(always)]
            pub fn as_array(&self) -> &ArrayT<Self>
            where Self: IntoArray,
            { <Self as IntoArray>::as_array(self) }

            /// Cast into a plain `&mut [[T; m]; n]`.
            #[inline(always)]
            pub fn as_array_mut(&mut self) -> &mut ArrayT<Self>
            where Self: IntoArray,
            { <Self as IntoArray>::as_array_mut(self) }
        }

    }
}

gen_each!{
    @{Mn_n}
    @{Vn_n}
    impl_general_inherent_wrappers_with_scalar!(
        {$Mr:ident $r:tt}
        {$Vc:ident $c:tt}
    ) => {
        impl<X> $Mr<$Vc<X>> {
            /// Map each scalar element of a matrix.
            #[inline(always)]
            pub fn map<B, F>(&self, f: F) -> $Mr<$Vc<B>>
            where X: Copy, F: FnMut(X) -> B,
            { from_array(::map_mat(self.into_array(), f)) }

            /// Apply a fallible function to each scalar element, with short-circuiting.
            #[inline(always)]
            pub fn try_map<E, B, F>(&self, f: F) -> Result<$Mr<$Vc<B>>, E>
            where X: Copy, F: FnMut(X) -> Result<B, E>,
            { ::try_map_mat(self.into_array(), f).map(from_array) }

            /// Apply a fallible function to each scalar element, with short-circuiting.
            #[inline(always)]
            pub fn opt_map<B, F>(&self, f: F) -> Option<$Mr<$Vc<B>>>
            where X: Copy, F: FnMut(X) -> Option<B>,
            { ::opt_map_mat(self.into_array(), f).map(from_array) }
        }
    }
}

// -------------------------- END PUBLIC API ---------------------------------
// The rest is implementation and boiler boiler boiiiiler boiilerplaaaaate
// ---------------------------------------------------------------------------

/// Implementation detail of the free function `mat::from_fn`.
///
/// > **_Fuggedaboudit._**
pub trait FromFn<F>: Sized {
    fn from_fn(f: F) -> Self;
}

gen_each!{
    @{Mn_n}
    @{Vn_n}
    impl_transpose!(
        {$Mr:ident $r:tt}
        {$Vc:ident $c:tt}
    ) => {
        impl<X, F> FromFn<F> for $Mr<$Vc<X>>
          where F: FnMut(usize, usize) -> X,
        {
            #[inline]
            fn from_fn(f: F) -> Self
            {
                let m: nd![_; $r; $c] = ::mat_from_fn(f);
                from_array(m)
            }
        }
    }
}

// ---------------------------------------------------------------------------

/// Implementation detail of the free function `mat::from_array`.
///
/// > **_Fuggedaboudit._**
pub trait IntoMatrix: Sized {
    type Matrix;

    fn into_matrix(self) -> Self::Matrix;
}

pub type ArrayT<M> = <M as IntoArray>::Array;
/// Implementation detail of the inherent method `{M2,M3,M4}::into_array`.
///
/// > **_Fuggedaboudit._**
pub trait IntoArray: Sized {
    type Array;

    fn into_array(self) -> Self::Array;
    fn as_array(&self) -> &Self::Array;
    fn as_array_mut(&mut self) -> &mut Self::Array;
}

gen_each!{
    @{Mn_n}
    @{Vn_n}
    impl_transpose!(
        {$Mr:ident $r:tt}
        {$Vc:ident $c:tt}
    ) => {
        impl<X> IntoMatrix for nd![X; $r; $c] {
            type Matrix = $Mr<$Vc<X>>;

            #[inline(always)]
            fn into_matrix(self) -> Self::Matrix
            { $Mr(self.envee()) }
        }

        impl<X> IntoArray for $Mr<$Vc<X>> {
            type Array = nd![X; $r; $c];

            #[inline(always)]
            fn into_array(self) -> Self::Array
            { self.0.unvee() }

            #[inline(always)]
            fn as_array(&self) -> &Self::Array
            { self.0.unvee_ref() }

            #[inline(always)]
            fn as_array_mut(&mut self) -> &mut Self::Array
            { self.0.unvee_mut() }
        }
    }
}

// ---------------------------------------------------------------------------

/// Implementation detail of the free function `mat::eye`.
///
/// > **_Fuggedaboudit._**
pub trait Eye: Sized {
    fn eye() -> Self;
}

impl<X: Semiring> Eye for M22<X>
  where X: PrimitiveSemiring
{
    #[inline(always)]
    fn eye() -> Self { M2([
        V2([ X::one(), X::zero()]),
        V2([X::zero(),  X::one()]),
    ])}
}

impl<X: Semiring> Eye for M33<X>
  where X: PrimitiveSemiring
{
    #[inline(always)]
    fn eye() -> Self { M3([
        V3([ X::one(), X::zero(), X::zero()]),
        V3([X::zero(),  X::one(), X::zero()]),
        V3([X::zero(), X::zero(),  X::one()]),
    ])}
}

impl<X: Semiring> Eye for M44<X>
  where X: PrimitiveSemiring
{
    #[inline(always)]
    fn eye() -> Self { M4([
        V4([ X::one(), X::zero(), X::zero(), X::zero()]),
        V4([X::zero(),  X::one(), X::zero(), X::zero()]),
        V4([X::zero(), X::zero(),  X::one(), X::zero()]),
        V4([X::zero(), X::zero(), X::zero(),  X::one()]),
    ])}
}

// ---------------------------------------------------------------------------

/// Output of `det`. Probably a scalar type.
pub type DetT<A> = <A as Det>::Output;

/// Implementation detail of the inherent method `{M22,M33,M44}::det`.
///
/// > **_Fuggedaboudit._**
pub trait Det {
    type Output;

    fn det(&self) -> Self::Output;
}

gen_each!{
    [ {M22 2} {M33 3} ]
    impl_det!(
        {$Mnn:ident $n:tt}
    ) => {
        impl<X: Ring> Det for $Mnn<X>
          where X: PrimitiveRing,
        {
            type Output = X;

            #[inline]
            fn det(&self) -> Self::Output
            { self.as_array().determinant() }
        }
    }
}

// ---------------------------------------------------------------------------

/// Implementation detail of the inherent method `{M22,M33,M44}::inv`.
///
/// > **_Fuggedaboudit._**
pub trait Inv {
    fn inv(&self) -> Self;
}

gen_each!{
    [ {M22 M2} {M33 M3} ]
    impl_det!(
        {$Mnn:ident $Mn:ident}
    ) => {
        // delegate to rsp2-array-utils
        impl<X: Field> Inv for $Mnn<X>
          where X: PrimitiveFloat,
        {
            #[inline]
            fn inv(&self) -> Self
            { from_array(self.as_array().inverse()) }
        }
    }
}

// ---------------------------------------------------------------------------

/// Output of `transpose`. Probably a matrix with the dimensions flipped.
pub type TransposeT<A> = <A as Transpose>::Output;

/// Implementation detail of the inherent method `{M2,M3,M4}::transpose`.
///
/// > **_Fuggedaboudit._**
pub trait Transpose {
    type Output;

    fn t(&self) -> Self::Output;
}

gen_each!{
    @{Mn_n}
    @{Vn_n}
    impl_transpose!(
        {$Mr:ident $r:tt}
        {$Vc:ident $c:tt}
    ) => {
        impl<X: Copy> Transpose for $Mr<$Vc<X>> {
            type Output = M![$c, V![$r, X]];

            #[inline]
            fn t(&self) -> Self::Output
            { from_fn(|r, c| self[c][r]) }
        }
    }
}
