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

use crate::traits::{Semiring, Ring, Field};
use crate::traits::internal::{PrimitiveSemiring, PrimitiveRing, PrimitiveFloat};

use super::types::*;

use num_traits::Zero;

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
where V: Zero + IsV,
{ Zero::zero() }

gen_each!{
    @{Vn_n}
    for_each!(
        {$Vn:ident $n:expr}
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

            /// Construct a fixed-size vector from a function on indices.
            #[inline(always)]
            pub fn try_from_fn<B, E, F>(f: F) -> Result<Self, E>
            where Self: TryFromFn<F, E>, F: FnMut(usize) -> Result<B, E>,
            { TryFromFn::try_from_fn(f) }

            /// Get the inner product of two vectors.
            ///
            /// It is recommended you write this as `V3::dot(a, b)`, rather than `a.dot(b)`.
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

            /// Get a basis vector.
            #[inline(always)]
            pub fn axis_unit(i: usize) -> Self
            where
                Self: Zero,
                X: Semiring + PrimitiveSemiring,
            {
                let mut v = Self::zero();
                *v.get_mut(i)
                    .unwrap_or_else(|| panic!("Invalid axis for {}d vector: {}", $n, i)) = X::one();
                v
            }

            /// Generate a randomly-oriented unit vector whose direction comes from a uniform
            /// distribution.
            #[inline(always)]
            pub fn random_unit() -> Self
            where Self: RandomUnit,
            { RandomUnit::random_unit() }

            /// Generate a randomly-oriented unit vector whose direction comes from a uniform
            /// distribution.
            #[inline(always)]
            pub fn random_unit_with(rng: impl rand::Rng) -> Self
            where Self: RandomUnit,
            { RandomUnit::random_unit_with(rng) }

            /// Get the shortest angle (as a value in `[0, pi]`) between this vector and another.
            #[inline(always)]
            pub fn angle_to(&self, other: &Self) -> ScalarT<Self>
            where X: Semiring + PrimitiveFloat,
            {
                let arg = dot(self, other) / X::sqrt(self.sqnorm() * other.sqnorm());
                let out = X::acos(arg.min(X::one()).max(-X::one()));
                out
            }

            /// Perform elementwise multiplication, or multiplication of a vector by a diagonal
            /// matrix.
            #[inline(always)]
            pub fn mul_diag(&self, other: &Self) -> Self
            where X: Semiring + PrimitiveSemiring,
            { Self::from_fn(|i| self[i] * other[i]) }

            /// Get the part of the vector that is parallel to `r`.
            #[inline]
            pub fn par(&self, r: &Self) -> Self
            where X: Field + PrimitiveFloat,
            { r * ($Vn::dot(self, r) / $Vn::dot(r, r)) }

            /// Get the part of the vector that is perpendicular to `r`.
            ///
            /// Be aware that chained calls to `perp` can have **spectacularly bad**
            /// numerical stability issues; you cannot trust that `c.perp(a).perp(b)`
            /// is even *remotely* orthogonal to `a` unless `b` is orthogonal to `a`.
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
            { rsp2_array_utils::try_map_arr(self.0, f).map($Vn) }

            /// Apply a fallible function to each element, with short-circuiting.
            #[inline(always)]
            pub fn opt_map<B, F>(self, f: F) -> Option<$Vn<B>>
            where F: FnMut(X) -> Option<B>,
            { rsp2_array_utils::opt_map_arr(self.0, f).map($Vn) }
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
    for_each!(
        {$Vn:ident}
    ) => {
        impl<X> IsV for $Vn<X>
        { type Scalar = X; }
    }
}

// -------------------------- END PUBLIC API ---------------------------------
// The rest is implementation and boiler boiler boiiiiler boiilerplaaaaate
// ---------------------------------------------------------------------------

gen_each!{
    @{Vn_n}
    for_each!(
        {$Vn:ident $n:tt}
    ) => {
        impl<X: Semiring> Zero for $Vn<X>
        where X: PrimitiveSemiring,
        {
            #[inline]
            fn zero() -> Self
            { $Vn([X::zero(); $n]) }

            #[inline]
            fn is_zero(&self) -> bool
            { self.iter().all(Zero::is_zero) }
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
    for_each!(
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

/// Implementation detail of the free function `vee::try_rom_fn`.
///
/// > **_Fuggedaboudit._**
///
/// Without this, the free function `try_from_fn` could not be generic over different
/// sizes of V.
pub trait TryFromFn<F, E>: Sized {
    fn try_from_fn(f: F) -> Result<Self, E>;
}

gen_each!{
    @{Vn}
    for_each!(
        {$Vn:ident}
    ) => {
        impl<X, F, E> TryFromFn<F, E> for $Vn<X>
          where F: FnMut(usize) -> Result<X, E>,
        {
            #[inline]
            fn try_from_fn(f: F) -> Result<$Vn<X>, E>
            { ::rsp2_array_utils::try_arr_from_fn(f).map($Vn) }
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
    for_each!( {$Vn:ident $n:tt} ) => {
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

/// Implementation detail of the inherent method `{V2,V3,V4}::random_unit`.
///
/// > **_Fuggedaboudit._**
pub trait RandomUnit: IsV + Sized {
    #[inline]
    fn random_unit() -> Self
    { RandomUnit::random_unit_with(rand::thread_rng()) }

    fn random_unit_with(rng: impl rand::Rng) -> Self;
}

// http://mathworld.wolfram.com/CirclePointPicking.html
impl<X: Field> RandomUnit for V2<X>
 where X: PrimitiveFloat,
{
    #[inline]
    fn random_unit_with(mut rng: impl rand::Rng) -> Self
    {
        loop {
            let x1 = X::uniform_with(&mut rng, (-X::one(), X::one()));
            let x2 = X::uniform_with(&mut rng, (-X::one(), X::one()));
            let denom = x1*x1 + x2*x2;
            if denom >= X::one() {
                continue;
            }
            let x = (x1*x1 - x2*x2) / denom;
            let y = X::two()*x1*x2 / denom;
            return V2([x, y]);
        }
    }
}

// http://mathworld.wolfram.com/CirclePointPicking.html
impl<X: Field> RandomUnit for V3<X>
 where X: PrimitiveFloat,
{
    #[inline]
    fn random_unit_with(mut rng: impl rand::Rng) -> Self
    {
        loop {
            let x1 = X::uniform_with(&mut rng, (-X::one(), X::one()));
            let x2 = X::uniform_with(&mut rng, (-X::one(), X::one()));
            let sqsum = x1*x1 + x2*x2;
            if sqsum >= X::one() {
                continue;
            }
            let root = X::sqrt(X::one() - sqsum);
            let x = X::two() * x1 * root;
            let y = X::two() * x2 * root;
            let z = X::one() - X::two() * sqsum;
            return V3([x, y, z]);
        }
    }
}

// ---------------------------------------------------------------------------

// slice-of-array integration.

// because `x.nest::<[_; 3]>().envee()` (turbofish required) really sucks.

gen_each!{
    @{Vn_n}
    for_each!( {$Vn:ident $n:tt} ) => {
        unsafe impl<X> slice_of_array::IsSliceomorphic for $Vn<X> {
            type Element = X;
            const LEN: usize = $n;
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

    #[test]
    fn random_unit_norm() {
        for _ in 0..10 {
            assert_close!(abs=1e-10, 1.0, V2::random_unit().sqnorm());
            assert_close!(abs=1e-10, 1.0, V3::random_unit().sqnorm());
        }
    }
}
