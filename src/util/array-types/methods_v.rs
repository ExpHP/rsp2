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

// This version is generic over the output type.
#[inline]
pub(crate) fn from_fn<V: TryFromFn<Elem=B>, B, F>(mut f: F) -> V
where F: FnMut(usize) -> B,
{ V::try_from_fn(|n| Ok::<_, ()>(f(n))).ok().unwrap() }

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
            pub fn from_fn<F>(f: F) -> Self
            where F: FnMut(usize) -> X,
            { from_fn(f) }

            /// Construct a fixed-size vector from a function on indices.
            #[inline(always)]
            pub fn try_from_fn<E, F>(f: F) -> Result<Self, E>
            where F: FnMut(usize) -> Result<X, E>,
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
            #[inline]
            pub fn map<B, F>(self, mut f: F) -> $Vn<B>
            where F: FnMut(X) -> B,
            { self.try_map(|x| Ok::<_, ()>(f(x))).ok().unwrap() }

            /// Apply a fallible function to each element, with short-circuiting.
            #[inline(always)]
            pub fn try_map<E, B, F>(self, f: F) -> Result<$Vn<B>, E>
            where F: FnMut(X) -> Result<B, E>,
            { TryMap::try_map(self, f) }

            /// Apply a fallible function to each element, with short-circuiting.
            #[inline]
            pub fn opt_map<B, F>(self, mut f: F) -> Option<$Vn<B>>
            where F: FnMut(X) -> Option<B>,
            {
                // hand the problem off to our "sufficiently smart compiler"
                self.try_map(|x| f(x).ok_or(Err::<B, ()>(()))).ok()
            }
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

/// Implementation detail of `V3::try_from_fn`.
pub(crate) trait TryFromFn: Sized {
    type Elem;

    fn try_from_fn<E>(f: impl FnMut(usize) -> Result<Self::Elem, E>) -> Result<Self, E>;
}

impl<A> TryFromFn for V2<A> {
    type Elem = A;

    #[inline]
    fn try_from_fn<E>(mut f: impl FnMut(usize) -> Result<Self::Elem, E>) -> Result<Self, E> {
        Ok(V2([f(0)?, f(1)?]))
    }
}

impl<A> TryFromFn for V3<A> {
    type Elem = A;

    #[inline]
    fn try_from_fn<E>(mut f: impl FnMut(usize) -> Result<Self::Elem, E>) -> Result<Self, E> {
        Ok(V3([f(0)?, f(1)?, f(2)?]))
    }
}

impl<A> TryFromFn for V4<A> {
    type Elem = A;

    #[inline]
    fn try_from_fn<E>(mut f: impl FnMut(usize) -> Result<Self::Elem, E>) -> Result<Self, E> {
        Ok(V4([f(0)?, f(1)?, f(2)?, f(3)?]))
    }
}

// ---------------------------------------------------------------------------

/// Implementation detail of the inherent method `V3::try_map`.
pub(crate) trait TryMap<B>: Sized {
    type Elem;
    type Output;

    fn try_map<E>(self, f: impl FnMut(Self::Elem) -> Result<B, E>) -> Result<Self::Output, E>;
}

impl<A, B> TryMap<B> for V2<A> {
    type Elem = A;
    type Output = V2<B>;

    #[inline]
    fn try_map<E>(self, mut f: impl FnMut(Self::Elem) -> Result<B, E>) -> Result<Self::Output, E> {
        let V2([a, b]) = self;
        Ok(V2([f(a)?, f(b)?]))
    }
}

impl<A, B> TryMap<B> for V3<A> {
    type Elem = A;
    type Output = V3<B>;

    #[inline]
    fn try_map<E>(self, mut f: impl FnMut(Self::Elem) -> Result<B, E>) -> Result<Self::Output, E> {
        let V3([a, b, c]) = self;
        Ok(V3([f(a)?, f(b)?, f(c)?]))
    }
}

impl<A, B> TryMap<B> for V4<A> {
    type Elem = A;
    type Output = V4<B>;

    #[inline]
    fn try_map<E>(self, mut f: impl FnMut(Self::Elem) -> Result<B, E>) -> Result<Self::Output, E> {
        let V4([a, b, c, d]) = self;
        Ok(V4([f(a)?, f(b)?, f(c)?, f(d)?]))
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

// stdlib integration

gen_each!{
    @{Vn_n}
    for_each!( {$Vn:ident $n:tt} ) => {
        impl<X: Semiring> std::iter::Sum for $Vn<X>
        where X: PrimitiveSemiring,
        {
            fn sum<I: Iterator<Item=$Vn<X>>>(iter: I) -> Self {
                iter.fold($Vn::zero(), |a, b| a + b)
            }
        }

        impl<'a, X: Semiring> std::iter::Sum<&'a $Vn<X>> for $Vn<X>
        where X: PrimitiveSemiring,
        {
            fn sum<I: Iterator<Item=&'a $Vn<X>>>(iter: I) -> Self {
                iter.fold($Vn::zero(), |a, b| a + b)
            }
        }
    }
}

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
