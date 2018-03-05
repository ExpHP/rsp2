
// NOTE: This is really part of rsp2-array-types.

use ::std::ops::{Add, Sub, AddAssign, SubAssign, Neg};
use ::std::ops::{Mul, Div, Rem, MulAssign, DivAssign, RemAssign};
use ::traits::{Semiring, Ring};
use ::traits::internal::{PrimitiveSemiring, PrimitiveRing};

use super::*;

// ---------------------------------------------------------------------------
// vector-vector ops

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
        // vector + vector
        impl<$($lt_a)* $($lt_b)* X: Semiring> Add<$($ref_b)* $Vn<X>> for $($ref_a)*$Vn<X>
          where X: PrimitiveSemiring,
        {
            type Output = $Vn<X>;

            #[inline]
            fn add(self, other: $($ref_b)* $Vn<X>) -> Self::Output
            { vee::from_fn(|k| self[k] + other[k]) }
        }

        // vector - vector
        impl<$($lt_a)* $($lt_b)* X: Ring> Sub<$($ref_b)* $Vn<X>> for $($ref_a)*$Vn<X>
          where X: PrimitiveRing,
        {
            type Output = $Vn<X>;

            #[inline]
            fn sub(self, other: $($ref_b)* $Vn<X>) -> Self::Output
            { vee::from_fn(|k| self[k] - other[k]) }
        }
    }
}

// ---------------------------------------------------------------------------
// vector unary ops

gen_each!{
    @{Vn}
    [ [(   ) (   )] [('a,) (&'a)] ]
    impl_v_unops!(
        {$Vn:ident}
        [ ($($lt_a:tt)*) ($($ref_a:tt)*) ]
    ) => {
        // -vector
        impl<$($lt_a)* X: Ring> Neg for $($ref_a)* $Vn<X>
          where X: PrimitiveRing,
        {
            type Output = $Vn<X>;

            #[inline]
            fn neg(self) -> Self::Output
            { vee::from_fn(|k| -self.0[k]) }
        }
    }
}

// ---------------------------------------------------------------------------
// vector-scalar ops

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
        // scalar * vector
        impl<$($lt_a)*> Mul<$($ref_a)* $Vn<$X>> for $X {
            type Output = $Vn<$X>;

            #[inline(always)]
            fn mul(self, vector: $($ref_a)* $Vn<$X>) -> Self::Output
            { vector * self }
        }

        // vector * scalar
        impl<$($lt_a)*> Mul<$X> for $($ref_a)* $Vn<$X> {
            type Output = $Vn<$X>;

            #[inline]
            fn mul(self, scalar: $X) -> Self::Output
            { vee::from_fn(|k| self[k] * scalar) }
        }

        // vector / scalar
        impl<$($lt_a)*> Div<$X> for $($ref_a)* $Vn<$X> {
            type Output = $Vn<$X>;

            #[inline]
            fn div(self, scalar: $X) -> Self::Output
            { vee::from_fn(|k| self[k] / scalar) }
        }

        // vector % scalar
        impl<$($lt_a)*> Rem<$X> for $($ref_a)* $Vn<$X> {
            type Output = $Vn<$X>;

            #[inline]
            fn rem(self, scalar: $X) -> Self::Output
            { vee::from_fn(|k| self[k] % scalar) }
        }
    }
}

// ---------------------------------------------------------------------------
// assign ops (general)

gen_each!{
    @{Vn}
    impl_v_assign_ops!(
        {$Vn:ident}
    ) => {
        // vector += vector;
        impl<X, B> AddAssign<B> for $Vn<X> where for<'a> &'a Self: Add<B, Output=Self> {
            #[inline(always)]
            fn add_assign(&mut self, rhs: B)
            { *self = &*self + rhs; }
        }

        // vector -= vector;
        impl<X, B> SubAssign<B> for $Vn<X> where for<'a> &'a Self: Sub<B, Output=Self> {
            #[inline(always)]
            fn sub_assign(&mut self, rhs: B)
            { *self = &*self - rhs; }
        }

        // vector *= scalar;
        // vector *= matrix;   (how fortunate that we primarily use row vectors!)
        impl<X, B> MulAssign<B> for $Vn<X> where for<'a> &'a Self: Mul<B, Output=Self> {
            #[inline(always)]
            fn mul_assign(&mut self, rhs: B)
            { *self = &*self * rhs; }
        }

        // vector /= scalar;
        impl<X, B> DivAssign<B> for $Vn<X> where for<'a> &'a Self: Div<B, Output=Self> {
            #[inline(always)]
            fn div_assign(&mut self, rhs: B)
            { *self = &*self / rhs; }
        }

        // vector %= scalar;
        impl<X, B> RemAssign<B> for $Vn<X> where for<'a> &'a Self: Rem<B, Output=Self> {
            #[inline(always)]
            fn rem_assign(&mut self, rhs: B)
            { *self = &*self % rhs; }
        }
    }
}

// ---------------------------------------------------------------------------

// vector * matrix
gen_each!{
    [ [(   ) (   )] [('v,) (&'v)] ]
    [{2} {3} {4}]
    [{2} {3} {4}]
    impl_mat_vec_mul!( [ ($($lt_v:tt)*) ($($ref_v:tt)*) ] {$r:tt} {$c:tt} ) => {
        // matrix * column vector
        impl<$($lt_v)* 'm, X: Semiring> Mul<$($ref_v)* V![$c, X]> for &'m M![$r, V![$c, X]]
          where X: PrimitiveSemiring,
        {
            type Output = V![$r, X];

            #[inline]
            fn mul(self, other: $($ref_v)* V![$c, X]) -> Self::Output {
                let matrix = self;
                let vector = other;
                vee::from_fn(|r| (0..$c).map(|i| matrix[r][i] * vector[i]).sum())
            }
        }

        // row vector * matrix
        impl<$($lt_v)* 'm, X: Semiring> Mul<&'m M![$r, V![$c, X]]> for $($ref_v)* V![$r, X]
          where X: PrimitiveSemiring,
        {
            type Output = V![$c, X];

            #[inline]
            fn mul(self, other: &'m M![$r, V![$c, X]]) -> Self::Output {
                let vector = self;
                let matrix = other;
                vee::from_fn(|c| (0..$r).map(|i| vector[i] * matrix[i][c]).sum())
            }
        }
    }
}

gen_each!{
    [{2} {3} {4}]
    [{2} {3} {4}]
    [{2} {3} {4}]
    impl_mat_mat_mul!( {$r:tt} {$k:tt} {$c:tt} ) => {
        // matrix * matrix
        impl<'a, 'b, X: Semiring> Mul<&'b M![$k, V![$c, X]]> for &'a M![$r, V![$k, X]]
          where X: PrimitiveSemiring,
        {
            type Output = M![$r, V![$c, X]];

            #[inline]
            fn mul(self, other: &'b M![$k, V![$c, X]]) -> Self::Output {
                mat::from_fn(|r,c| (0..$k).map(|i| self[r][i] * other[i][c]).sum())
            }
        }
    }
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mat_mat() {
        let eye2 = mat::from_array([[1, 0], [0, 1i32]]);
        let eye3 = mat::from_array([[1, 0, 0], [0, 1, 0], [0, 0, 1i32]]);

        let a = mat::from_array([
            [1, 2, 3],
            [4, 5, 6],
        ]);

        let b = mat::from_array([
            [1,  1],
            [1, -1],
            [0,  1],
        ]);

        let a_dot_b = mat::from_array([
            [3, 2],
            [9, 5],
        ]);

        assert_eq!(a, &eye2 * &a);
        assert_eq!(a, &a * &eye3);
        assert_eq!(a_dot_b, &a * &b);
    }

    #[test]
    fn mat_vec() {
        let m = mat::from_array([
            [1, 2, 3],
            [4, 5, 6],
        ]);
        assert_eq!(V2([1, 7]), &m * V3([4, -3, 1]));
        assert_eq!(V3([-8, -7, -6]), V2([4, -3]) * &m);

        // try with the other dimension longer so that we make sure the sums
        // are over the right indices
        let m = m.t();
        assert_eq!(V2([1, 7]), V3([4, -3, 1]) * &m);
        assert_eq!(V3([-8, -7, -6]), &m * V2([4, -3]));
    }
}
