
// NOTE: This is really part of rsp2-array-types.

use ::std::ops::{Add, Sub, AddAssign, SubAssign, Neg};
use ::std::ops::{Mul, Div, Rem, MulAssign, DivAssign, RemAssign};
use ::std::ops::{Deref, DerefMut};
use ::traits::{Semiring, Ring, Field};
use ::traits::internal::{PrimitiveSemiring, PrimitiveRing, PrimitiveFloat};
use ::arr_from_fn;
use ::std::mem;

/// A 2-dimensional vector with operations for linear algebra.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[derive(Serialize, Deserialize)]
pub struct V2<X=f64>(pub [X; 2]);

/// A 3-dimensional vector with operations for linear algebra.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[derive(Serialize, Deserialize)]
pub struct V3<X=f64>(pub [X; 3]);

/// A 4-dimensional vector with operations for linear algebra.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[derive(Serialize, Deserialize)]
pub struct V4<X=f64>(pub [X; 4]);

// ---------------------------------------------------------------------------

pub type Iter<'a, X> = ::std::slice::Iter<'a, X>;
pub type IterMut<'a, X> = ::std::slice::IterMut<'a, X>;

gen_each!{
    @{Vn_n}
    impl_v_deref!(
        {$Vn:ident $n:tt}
    ) => {
        impl<X> Deref for $Vn<X> {
            type Target = [X; $n];

            #[inline(always)]
            fn deref(&self) -> &Self::Target
            { &self.0 }
        }

        impl<X> DerefMut for $Vn<X> {
            #[inline(always)]
            fn deref_mut(&mut self) -> &mut Self::Target
            { &mut self.0 }
        }

        // Fix a paper cut not solved by Deref, which is that many methods
        // take `I: IntoIterator`.
        impl<'a, X> IntoIterator for &'a $Vn<X> {
            type Item = &'a X;
            type IntoIter = Iter<'a, X>;

            #[inline(always)]
            fn into_iter(self) -> Self::IntoIter
            { self.0.iter() }
        }

        impl<'a, X> IntoIterator for &'a mut $Vn<X> {
            type Item = &'a mut X;
            type IntoIter = IterMut<'a, X>;

            #[inline(always)]
            fn into_iter(self) -> Self::IntoIter
            { self.0.iter_mut() }
        }
    }
}

// ---------------------------------------------------------------------------

// NOTE: Operator impls are deliberately between same-typed vectors,
//       rather than e.g. V3<T> and V3<U> where T: Add<U>.
//
//       The reason for this is that the having such generic bounds
//       tends to influence the design of the rest of the library
//       towards a design that is actually impossible to implement.
gen_each!{
    @{Vn}
    [ [(   ) (   )] [('a,) (&'a)] ]
    [ [(   ) (   )] [('b,) (&'b)] ]
    impl_v_add_sub!(
        {$Vn:ident}
        [ ($($lt_a:tt)*) ($($ref_a:tt)*) ]
        [ ($($lt_b:tt)*) ($($ref_b:tt)*) ]
    ) => {
        impl<$($lt_a)* $($lt_b)* X: Semiring> Add<$($ref_b)* $Vn<X>> for $($ref_a)*$Vn<X>
          where X: PrimitiveSemiring,
        {
            type Output = $Vn<X>;

            fn add(self, other: $($ref_b)* $Vn<X>) -> Self::Output
            { $Vn(arr_from_fn(|k| self[k] + other[k])) }
        }

        impl<$($lt_a)* $($lt_b)* X: Ring> Sub<$($ref_b)* $Vn<X>> for $($ref_a)*$Vn<X>
          where X: PrimitiveRing,
        {
            type Output = $Vn<X>;

            fn sub(self, other: $($ref_b)* $Vn<X>) -> Self::Output
            { $Vn(arr_from_fn(|k| self[k] - other[k])) }
        }
    }
}

gen_each!{
    @{Vn}
    [ [(   ) (   )] [('b,) (&'b)] ]
    impl_v_add_sub_assign!(
        {$Vn:ident}
        [ ($($lt_b:tt)*) ($($ref_b:tt)*) ]
    ) => {
        impl<$($lt_b)* X: Semiring> AddAssign<$($ref_b)* $Vn<X>> for $Vn<X>
          where X: PrimitiveSemiring,
        {
            fn add_assign(&mut self, other: $($ref_b)* $Vn<X>)
            { *self = &*self + other; }
        }

        impl<$($lt_b)* X: Ring> SubAssign<$($ref_b)* $Vn<X>> for $Vn<X>
          where X: PrimitiveRing,
        {
            fn sub_assign(&mut self, other: $($ref_b)* $Vn<X>)
            { *self = &*self - other; }
        }
    }
}


// ---------------------------------------------------------------------------

gen_each!{
    @{Vn}
    [ [(   ) (   )] [('a,) (&'a)] ]
    impl_v_unops!(
        {$Vn:ident}
        [ ($($lt_a:tt)*) ($($ref_a:tt)*) ]
    ) => {
        impl<$($lt_a)* X: Ring> Neg for $($ref_a)* $Vn<X>
          where X: PrimitiveRing,
        {
            type Output = $Vn<X>;

            fn neg(self) -> Self::Output
            { $Vn(arr_from_fn(|k| -self.0[k])) }
        }
    }
}

// ---------------------------------------------------------------------------

gen_each!{
    @{Vn}
    // NOTE: these impls are explitly done for each semiring type rather than
    //       being generic over X: Semiring so that the orphan rules don't prevent
    //       us from impling `scalar * vector` multiplication
    @{semiring}
    [ [(   ) (   )] [('a,) (&'a)] ]
    impl_v_scalar_ops!(
        {$Vn:ident}
        {$X:ty}
        [ ($($lt_a:tt)*) ($($ref_a:tt)*) ]
    ) => {
        impl<$($lt_a)*> Mul<$($ref_a)* $Vn<$X>> for $X {
            type Output = $Vn<$X>;

            #[inline(always)]
            fn mul(self, vector: $($ref_a)* $Vn<$X>) -> Self::Output
            { vector * self }
        }

        impl<$($lt_a)*> Mul<$X> for $($ref_a)* $Vn<$X> {
            type Output = $Vn<$X>;

            fn mul(self, scalar: $X) -> Self::Output
            { $Vn(arr_from_fn(|k| self[k] * scalar)) }
        }

        impl<$($lt_a)*> Div<$X> for $($ref_a)* $Vn<$X> {
            type Output = $Vn<$X>;

            fn div(self, scalar: $X) -> Self::Output
            { $Vn(arr_from_fn(|k| self[k] / scalar)) }
        }

        impl<$($lt_a)*> Rem<$X> for $($ref_a)* $Vn<$X> {
            type Output = $Vn<$X>;

            fn rem(self, scalar: $X) -> Self::Output
            { $Vn(arr_from_fn(|k| self[k] % scalar)) }
        }
    }
}

gen_each!{
    @{Vn}
    @{semiring}
    impl_v_scalar_ops_assign!(
        {$Vn:ident}
        {$X:ty}
    ) => {
        impl MulAssign<$X> for $Vn<$X> {
            fn mul_assign(&mut self, scalar: $X)
            { *self = &*self * scalar; }
        }

        impl DivAssign<$X> for $Vn<$X> {
            fn div_assign(&mut self, scalar: $X)
            { *self = &*self / scalar; }
        }

        impl RemAssign<$X> for $Vn<$X> {
            fn rem_assign(&mut self, scalar: $X)
            { *self = &*self % scalar; }
        }
    }
}

// ---------------------------------------------------------------------------
// Inherent methods.
//
// Most of them delegate to traits so that inherent impls can be generic without having
// insane looking signatures.

gen_each!{
    @{Vn}
    impl_v_inherent_wrappers!(
        {$Vn:ident}
    ) => {
        impl<X> $Vn<X> {
            /// Get the inner product of two vectors.
            pub fn dot<B>(&self, other: &B) -> DotT<Self, B>
            where Self: Dot<B>,
            { <Self as Dot<B>>::dot(self, other) }

            /// Get the vector's squared magnitude.
            pub fn sqnorm(&self) -> SqnormT<Self>
            where Self: Sqnorm,
            { <Self as Sqnorm>::sqnorm(self) }

            /// Get the vector's magnitude.
            pub fn norm(&self) -> NormT<Self>
            where Self: Norm,
            { <Self as Norm>::norm(self) }

            /// Get the shortest angle (as a value in `[0, pi]`) between this vector and another.
            pub fn angle_to(&self, other: &Self) -> AngleToT<Self>
            where Self: AngleTo,
            { <Self as AngleTo>::angle_to(self, other) }

            /// Get the part of the vector that is parallel to `r`.
            pub fn par(&self, r: &Self) -> Self
            where Self: Par,
            { <Self as Par>::par(self, r) }

            /// Get the part of the vector that is perpendicular to `r`.
            pub fn perp(&self, r: &Self) -> Self
            where Self: Perp,
            { <Self as Perp>::perp(self, r) }

            /// Apply a function to each element.
            pub fn map<B, F>(self, f: F) -> $Vn<B>
            where F: FnMut(X) -> B,
            { $Vn(::map_arr(self.0, f)) }

            /// Apply a fallible function to each element, with short-circuiting.
            pub fn try_map<E, B, F>(self, f: F) -> Result<$Vn<B>, E>
            where F: FnMut(X) -> Result<B, E>,
            { ::try_map_arr(self.0, f).map($Vn) }

            /// Apply a fallible function to each element, with short-circuiting.
            pub fn opt_map<E, B, F>(self, f: F) -> Option<$Vn<B>>
            where F: FnMut(X) -> Option<B>,
            { ::opt_map_arr(self.0, f).map($Vn) }
        }
    }
}

// ---------------------------------------------------------------------------

/// Output of `dot`.  Probably the scalar type of the vector.
pub type DotT<A, B> = <A as Dot<B>>::Output;
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

            fn dot(&self, other: &$Vn<X>) -> Self::Output
            { (0..$n).map(|i| self[i] * other[i]).sum() }
        }
    }
}

/// Inner product of vectors.
///
/// This is basically just `{V2,V3,V4}::dot` as a free function,
/// because everyone loves symmetry.
pub fn dot<A, B>(a: &A, b: &B) -> DotT<A, B>
  where A: Dot<B>,
{ a.dot(b) }

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
            fn par(&self, r: &$Vn<X>) -> Self
            { r * (self.dot(r) / r.dot(r)) }
        }

        impl<X: Field> Perp for $Vn<X>
          where
            X: PrimitiveFloat,
            for<'b> &'b Self: Mul<X, Output=Self>,
        {
            fn perp(&self, r: &$Vn<X>) -> Self
            { self - self.par(r) }
        }
    }
}

// ---------------------------------------------------------------------------

/// Zero-cost transformations from sequences of arrays into sequences of `Vn`.
///
/// # Safety
///
/// The default impls effectively perform `transmute`, and some of the generic
/// impls assume that it is safe to perform pointer casts between Self and `Self::En`.
/// (this may be done even on pointers to pointers, or smart pointers and etc.)
pub unsafe trait Envee {
    type En: ?Sized;

    /// Casts a sequence of arrays into `V2`/`V3`/`V4`s.
    fn envee(self) -> Self::En
    where Self: Sized, Self::En: Sized
    { unsafe { mem::transmute_copy(&mem::ManuallyDrop::new(self)) } }
}

/// Zero-cost transformations from sequences of `Vn` into sequences of arrays.
///
/// # Safety
///
/// The default impls effectively perform `transmute`, and some of the generic
/// impls assume that it is safe to perform pointer casts between Self and `Self::Un`.
/// (this may be done even on pointers to pointers, or smart pointers and etc.)
pub unsafe trait Unvee {
    type Un: ?Sized;

    /// Casts a sequence of `V2`/`V3`/`V4`s into arrays.
    fn unvee(self) -> Self::Un
    where Self: Sized, Self::Un: Sized
    { unsafe { mem::transmute_copy(&mem::ManuallyDrop::new(self)) } }
}

gen_each!{
    @{Vn_n}
    impl_envee_unvee_slice!( {$Vn:ident $n:tt} ) => {
        unsafe impl<X> Envee for [[X;$n]] { type En = [$Vn<X>]; }
        unsafe impl<X> Unvee for [$Vn<X>] { type Un = [[X;$n]]; }

        unsafe impl<X> Envee for Vec<[X;$n]> { type En = Vec<$Vn<X>>; }
        unsafe impl<X> Unvee for Vec<$Vn<X>> { type Un = Vec<[X;$n]>; }
    }
}

gen_each!{
    @{Vn_n}
    @{0...8}
    impl_envee_unvee_array!( {$Vn:ident $n:tt} {$k:tt} ) => {
        unsafe impl<X> Envee for [[X;$n]; $k] { type En = [$Vn<X>; $k]; }
        unsafe impl<X> Unvee for [$Vn<X>; $k] { type Un = [[X;$n]; $k]; }
    }
}

mod envee_generic_impls {
    use super::*;

    use ::std::rc::{Rc, Weak as RcWeak};
    use ::std::sync::{Arc, Weak as ArcWeak};
    use ::std::cell::RefCell;

    gen_each!{
        [ {Envee En} {Unvee Un} ]
        impl_envee_unvee_generic!( {$Envee:ident $En:ident} ) => {
            unsafe impl<'a, V: $Envee + ?Sized> $Envee for &'a V      { type $En = &'a V::$En; }
            unsafe impl<'a, V: $Envee + ?Sized> $Envee for &'a mut V  { type $En = &'a mut V::$En; }
            unsafe impl<    V: $Envee + ?Sized> $Envee for Box<V>     { type $En = Box<V::$En>; }
            unsafe impl<    V: $Envee + ?Sized> $Envee for Rc<V>      { type $En = Rc<V::$En>; }
            unsafe impl<    V: $Envee + ?Sized> $Envee for RcWeak<V>  { type $En = RcWeak<V::$En>; }
            unsafe impl<    V: $Envee + ?Sized> $Envee for Arc<V>     { type $En = Arc<V::$En>; }
            unsafe impl<    V: $Envee + ?Sized> $Envee for ArcWeak<V> { type $En = ArcWeak<V::$En>; }
            unsafe impl<    V: $Envee + ?Sized> $Envee for RefCell<V> { type $En = RefCell<V::$En>; }
        }
    }
}

/// Casts a sequence of arrays into `V2`/`V3`/`V4`s.
///
/// This is basically just `Envee::envee` as a free function,
/// because sometimes it's easier to use that way.
pub fn envee<V: Envee>(v: V) -> V::En where V::En: Sized { v.envee() }

/// Casts a sequence of `V2`/`V3`/`V4`s into arrays.
///
/// This is basically just `Unvee::unvee` as a free function,
/// because sometimes it's easier to use that way.
pub fn unvee<V: Unvee>(v: V) -> V::Un where V::Un: Sized { v.unvee() }

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
