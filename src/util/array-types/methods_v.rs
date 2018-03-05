
// NOTE: This is really part of rsp2-array-types.

use ::std::ops::Mul;
use ::traits::{Semiring, Field};
use ::traits::internal::{PrimitiveSemiring, PrimitiveFloat};

use super::types::*;

// ---------------------------------------------------------------------------
// ------------------------------ PUBLIC API ---------------------------------

/// Construct a fixed-size vector from a function on indices.
#[inline(always)]
pub fn from_fn<V: FromFn<F>, B, F>(f: F) -> V
where F: FnMut(usize) -> B,
{ FromFn::from_fn(f) }

gen_each!{
    @{Vn}
    impl_v_inherent_wrappers!(
        {$Vn:ident}
    ) => {
        impl<X> $Vn<X> {
            /// Construct a fixed-size vector from a function on indices.
            #[inline(always)]
            pub fn from_fn<B, F>(f: F) -> Self
            where Self: FromFn<F>, F: FnMut(usize) -> B,
            { FromFn::from_fn(f) }

            /// Get the inner product of two vectors.
            #[inline(always)]
            pub fn dot<B>(&self, other: &B) -> DotT<Self, B>
            where Self: Dot<B>,
            { <Self as Dot<B>>::dot(self, other) }

            /// Get the vector's squared magnitude.
            #[inline(always)]
            pub fn sqnorm(&self) -> SqnormT<Self>
            where Self: Sqnorm,
            { <Self as Sqnorm>::sqnorm(self) }

            /// Get the vector's magnitude.
            #[inline(always)]
            pub fn norm(&self) -> NormT<Self>
            where Self: Norm,
            { <Self as Norm>::norm(self) }

            /// Get the shortest angle (as a value in `[0, pi]`) between this vector and another.
            #[inline(always)]
            pub fn angle_to(&self, other: &Self) -> AngleToT<Self>
            where Self: AngleTo,
            { <Self as AngleTo>::angle_to(self, other) }

            /// Get the part of the vector that is parallel to `r`.
            #[inline(always)]
            pub fn par(&self, r: &Self) -> Self
            where Self: Par,
            { <Self as Par>::par(self, r) }

            /// Get the part of the vector that is perpendicular to `r`.
            #[inline(always)]
            pub fn perp(&self, r: &Self) -> Self
            where Self: Perp,
            { <Self as Perp>::perp(self, r) }

            /// Apply a function to each element.
            #[inline(always)]
            pub fn map<B, F>(self, f: F) -> $Vn<B>
            where F: FnMut(X) -> B,
            { $Vn(::rsp2_array_utils::map_arr(self.0, f)) }

            /// Apply a fallible function to each element, with short-circuiting.
            #[inline(always)]
            pub fn try_map<E, B, F>(self, f: F) -> Result<$Vn<B>, E>
            where F: FnMut(X) -> Result<B, E>,
            { ::rsp2_array_utils::try_map_arr(self.0, f).map($Vn) }

            /// Apply a fallible function to each element, with short-circuiting.
            #[inline(always)]
            pub fn opt_map<B, F>(self, f: F) -> Option<$Vn<B>>
            where F: FnMut(X) -> Option<B>,
            { ::rsp2_array_utils::opt_map_arr(self.0, f).map($Vn) }
        }
    }
}

/// Inner product of vectors.
///
/// This is basically just `{V2,V3,V4}::dot` as a free function,
/// because everyone loves symmetry.
#[inline(always)]
pub fn dot<A, B>(a: &A, b: &B) -> DotT<A, B>
  where A: Dot<B>,
{ a.dot(b) }

// -------------------------- END PUBLIC API ---------------------------------
// The rest is implementation and boiler boiler boiiiiler boiilerplaaaaate
// ---------------------------------------------------------------------------

/// Implementation detail of the free function `vee::from_fn`.
///
/// > **_Fuggedaboudit._**
pub trait FromFn<F>: Sized {
    fn from_fn(f: F) -> Self;
}

gen_each!{
    @{Vn}
    impl_from_fn!(
        {$Vn:ident}
    ) => {
        impl<X, F> FromFn<F> for $Vn<X>
          where F: FnMut(usize) -> X,
        {
            #[inline]
            fn from_fn(f: F) -> Self
            { $Vn(::rsp2_array_utils::arr_from_fn(f)) }
        }
    }
}

// ---------------------------------------------------------------------------

/// Output of `dot`.  Probably the scalar type of the vector.
pub type DotT<A, B> = <A as Dot<B>>::Output;

/// Implementation detail of the inherent method `{V2,V3,V4}::dot`.
///
/// > **_Fuggedaboudit._**
pub trait Dot<B: ?Sized> {
    type Output;

    fn dot(&self, b: &B) -> Self::Output;
}

gen_each!{
    @{Vn_n}
    impl_v_dot!( {$Vn:ident $n:tt} ) => {
        impl<X: Semiring> Dot<$Vn<X>> for $Vn<X>
          where X: PrimitiveSemiring,
        {
            type Output = X;

            #[inline]
            fn dot(&self, other: &$Vn<X>) -> Self::Output
            { (0..$n).map(|i| self[i] * other[i]).sum() }
        }
    }
}

// ---------------------------------------------------------------------------

/// Output of `sqnorm`.  Probably the (real) scalar type of the vector.
pub type SqnormT<A> = DotT<A, A>;

/// Implementation detail of the inherent method `{V2,V3,V4}::sqnorm`.
///
/// > **_Fuggedaboudit._**
pub trait Sqnorm: Dot<Self> {
    #[inline(always)]
    fn sqnorm(&self) -> SqnormT<Self>
    { self.dot(self) }
}

/// Output of `norm`.  Probably the (real) scalar type of the vector.
pub type NormT<A> = <A as Norm>::Output;

/// Implementation detail of the inherent method `{V2,V3,V4}::norm`.
///
/// > **_Fuggedaboudit._**
pub trait Norm {
    type Output;

    fn norm(&self) -> Self::Output;
}

gen_each!{
    @{Vn}
    impl_v_norm!( {$Vn:ident} ) => {
        impl<X> Sqnorm for $Vn<X>
          where $Vn<X>: Dot<$Vn<X>>,
        { }

        impl<X: Field> Norm for $Vn<X>
          where X: PrimitiveFloat,
        {
            type Output = X;

            #[inline]
            fn norm(&self) -> Self::Output
            { self.sqnorm().sqrt() }
        }
    }
}

// ---------------------------------------------------------------------------

/// Output of `angle_to`. Probably a float with the same precision as the vector.
pub type AngleToT<A> = <A as AngleTo>::Output;

/// Implementation detail of the inherent method `{V2,V3,V4}::angle_to`.
///
/// > **_Fuggedaboudit._**
pub trait AngleTo {
    type Output;

    fn angle_to(&self, other: &Self) -> Self::Output;
}

gen_each!{
    @{Vn}
    [{f32} {f64}]
    impl_v_angle_to!( {$Vn:ident} {$X:ty} ) => {
        impl AngleTo for $Vn<$X> {
            type Output = $X;

            #[inline]
            fn angle_to(&self, other: &Self) -> Self::Output
            { <$X>::acos(self.dot(other) / <$X>::sqrt(self.sqnorm() * other.sqnorm())) }
        }
    }
}

// ---------------------------------------------------------------------------

/// Implementation detail of the inherent method `{V2,V3,V4}::par`.
///
/// > **_Fuggedaboudit._**
pub trait Par {
    fn par(&self, r: &Self) -> Self;
}

/// Implementation detail of the inherent method `{V2,V3,V4}::perp`.
///
/// > **_Fuggedaboudit._**
pub trait Perp {
    fn perp(&self, r: &Self) -> Self;
}

gen_each!{
    @{Vn}
    impl_v_par!( {$Vn:ident} ) => {
        impl<X: Field> Par for $Vn<X>
          where
            X: PrimitiveFloat,
            for<'b> &'b Self: Mul<X, Output=Self>,
        {
            #[inline]
            fn par(&self, r: &$Vn<X>) -> Self
            { r * (self.dot(r) / r.dot(r)) }
        }

        impl<X: Field> Perp for $Vn<X>
          where
            X: PrimitiveFloat,
            for<'b> &'b Self: Mul<X, Output=Self>,
        {
            #[inline]
            fn perp(&self, r: &$Vn<X>) -> Self
            { self - self.par(r) }
        }
    }
}

// ---------------------------------------------------------------------------

// slice-of-array integration.

// because `x.nest::<[_; 3]>().envee()` (turbofish required) really sucks.

// NOTE: IsSliceomorphic was never intended to be implemented by crates outside
//       slice_of_array, and the requirements for safety are left "unspecified".
//
//       That said, speaking as the maintainer of slice_of_array, I will say that
//       the following fact is sufficient for safety:
//
//       * It is safe to pointer-cast between `&[V3<X>]` and `&[[X; 3]]`.
//
//       - ML

gen_each!{
    @{Vn_n}
    impl_is_sliceomorphic!( {$Vn:ident $n:tt} ) => {
        unsafe impl<X> ::slice_of_array::IsSliceomorphic for $Vn<X> {
            type Element = X;

            #[inline(always)]
            fn array_len() -> usize { $n }
        }
    }
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn angle() {
        let a: V3 = V3([0.5, 0.0,  0.0]);
        let b: V3 = V3([2.0, 0.0, -2.0]);

        assert_close!(45.0, a.angle_to(&b).to_degrees());
    }

    #[test]
    fn prop_perp_plus_par() {
        for _ in 0..10 {
            let a: V3 = V3(::rand::random());
            let b: V3 = V3(::rand::random());
            (a.perp(&b) + a.par(&b) - a).iter().for_each(|&x| {
                assert_close!(abs=1e-10, 0.0, x);
            });
        }
    }

    #[test]
    fn prop_perp_is_perp() {
        for _ in 0..10 {
            let a: V3 = V3(::rand::random());
            let b: V3 = V3(::rand::random());
            assert_close!(abs=1e-10, 0.0, a.perp(&b).dot(&b));
        }
    }
}
