
use ::traits::{Semiring, Ring, Field};
use ::traits::internal::{PrimitiveSemiring, PrimitiveRing, PrimitiveFloat};

use super::types::*;

// ---------------------------------------------------------------------------
// ------------------------------ PUBLIC API ---------------------------------

/// Construct a fixed-size vector from a function on indices.
///
/// The length of the vector will be inferred solely from how
/// it is used. There is also a static method form of this for
/// easily supplying a type hint. (e.g. `V3::from_fn`)
#[inline(always)]
pub fn from_fn<V: FromFn<F>, B, F>(f: F) -> V
where F: FnMut(usize) -> B,
{ FromFn::from_fn(f) }

/// Get a zero vector. (using type inference)
#[inline(always)]
pub fn zero<V: Zero>() -> V
where V: Zero,
{ Zero::zero() }

gen_each!{
    @{Vn}
    impl_v_inherent_wrappers!(
        {$Vn:ident}
    ) => {
        impl<X> $Vn<X> {
            /// Get a zero vector.
            ///
            /// This is also available as the free function `vee::zero`;
            /// this static method just provides an easy way to supply a type hint.
            #[inline(always)]
            pub fn zero() -> Self
            where Self: Zero,
            { Zero::zero() }

            /// Construct a fixed-size vector from a function on indices.
            ///
            /// This is also available as the free function `vee::from_fn`;
            /// this static method just provides an easy way to supply a type hint.
            #[inline(always)]
            pub fn from_fn<B, F>(f: F) -> Self
            where Self: FromFn<F>, F: FnMut(usize) -> B,
            { FromFn::from_fn(f) }

            /// Get the inner product of two vectors.
            ///
            /// This is also available as the free function `dot`, and to be honest
            /// there's really no reason to prefer this form...
            #[deprecated = "use `dot(&a, &b)` (also `vee::dot(&a, &b)`)"]
            #[inline(always)]
            pub fn dot(&self, other: &Self) -> ScalarT<Self>
            where Self: Dot,
            { Dot::dot(self, other) }

            /// Get the vector's squared magnitude.
            #[inline(always)]
            pub fn sqnorm(&self) -> ScalarT<Self>
            where Self: Dot,
            { Dot::dot(self, self) }

            /// Get the vector's magnitude.
            #[inline(always)]
            pub fn norm(&self) -> ScalarT<Self>
            where Self: Dot, ScalarT<Self>: PrimitiveFloat,
            { self.sqnorm().sqrt() }

            /// Normalize the vector.
            #[inline(always)]
            pub fn unit(&self) -> Self
            where X: Field + PrimitiveFloat,
            { self / self.norm() }

            /// Get the shortest angle (as a value in `[0, pi]`) between this vector and another.
            #[inline(always)]
            pub fn angle_to(&self, other: &Self) -> ScalarT<Self>
            where X: Semiring + PrimitiveFloat,
            {
                let arg = dot(self, other) / X::sqrt(self.sqnorm() * other.sqnorm());
                let out = X::acos(arg.min(X::one()).max(-X::one()));
                out
            }

            /// Get the part of the vector that is parallel to `r`.
            #[inline]
            pub fn par(&self, r: &Self) -> Self
            where X: Field + PrimitiveFloat,
            { r * (dot(self, r) / dot(r, r)) }

            /// Get the part of the vector that is perpendicular to `r`.
            ///
            /// Be aware that chained calls to `perp` can have **spectacularly bad**
            /// numerical stability issues; you cannot trust that `c.perp(a).perp(b)`
            /// is even *remotely* orthogonal to `a`.
            /// (for 3d vectors, try `c.par(a.cross(b))` instead.)
            #[inline]
            pub fn perp(&self, r: &Self) -> Self
            where X: Field + PrimitiveFloat,
            { self - self.par(r) }

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

impl<X: Ring> V3<X>
where X: PrimitiveRing
{
    /// Cross-product. Only defined on 3-dimensional vectors.
    #[inline]
    pub fn cross(&self, other: &Self) -> Self {
        V3([
            self[1] * other[2] - self[2] * other[1],
            self[2] * other[0] - self[0] * other[2],
            self[0] * other[1] - self[1] * other[0],
        ])
    }
}

/// Inner product of vectors.
///
/// This is basically just `{V2,V3,V4}::dot` as a free function,
/// because everyone loves symmetry.
#[inline(always)]
pub fn dot<V>(a: &V, b: &V) -> ScalarT<V>
where V: Dot,
{ Dot::dot(a, b) }

/// Element type of the vector.
pub type ScalarT<V> = <V as IsV>::Scalar;
/// Trait that provides associated types for `V2, V3, V4`.
pub trait IsV {
    type Scalar;
}

gen_each!{
    @{Vn}
    impl_from_fn!(
        {$Vn:ident}
    ) => {
        impl<X> IsV for $Vn<X>
        { type Scalar = X; }
    }
}

// -------------------------- END PUBLIC API ---------------------------------
// The rest is implementation and boiler boiler boiiiiler boiilerplaaaaate
// ---------------------------------------------------------------------------

/// Implementation detail of the free function `vee::zero`.
///
/// > **_Fuggedaboudit._**
///
/// Without this, the free function `zero` could not be generic over different
/// sizes of V.
pub trait Zero: Sized {
    fn zero() -> Self;
}

gen_each!{
    @{Vn_n}
    impl_from_fn!(
        {$Vn:ident $n:tt}
    ) => {
        impl<X: Semiring> Zero for $Vn<X>
        where X: PrimitiveSemiring,
        {
            #[inline]
            fn zero() -> Self
            { $Vn([X::zero(); $n]) }
        }
    }
}

// ---------------------------------------------------------------------------

/// Implementation detail of the free function `vee::from_fn`.
///
/// > **_Fuggedaboudit._**
///
/// Without this, the free function `from_fn` could not be generic over different
/// sizes of V.
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

/// Implementation detail of the inherent method `{V2,V3,V4}::dot`.
///
/// > **_Fuggedaboudit._**
///
/// Without this, the free function `dot` could not be generic over different
/// sizes of V.
pub trait Dot: IsV {
    fn dot(&self, b: &Self) -> ScalarT<Self>;
}

gen_each!{
    @{Vn_n}
    impl_v_dot!( {$Vn:ident $n:tt} ) => {
        impl<X: Semiring> Dot for $Vn<X>
          where X: PrimitiveSemiring,
        {
            #[inline]
            fn dot(&self, other: &$Vn<X>) -> ScalarT<Self>
            { (1..$n).fold(self[0] * other[0], |s, i| s + self[i] * other[i]) }
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
        let b: V3 = V3([8.0, 0.0, -8.0]);

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
    fn prop_par_is_par() {
        for _ in 0..10 {
            let a: V3 = V3(::rand::random());
            let b: V3 = V3(::rand::random());
            let par = a.par(&b);
            assert_close!(abs=1e-4, 0.0, par.angle_to(&b));
        }
    }

    #[test]
    fn prop_perp_is_perp() {
        for _ in 0..10 {
            let a: V3 = V3(::rand::random());
            let b: V3 = V3(::rand::random());
            assert_close!(abs=1e-10, 0.0, dot(&a.perp(&b), &b));
        }
    }
}
