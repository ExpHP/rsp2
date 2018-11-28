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

use std::ops::{Add, Sub, AddAssign, SubAssign, Neg};
use std::ops::{Mul, Div, MulAssign, DivAssign};
use std::fmt;
use crate::traits::{Semiring, Ring, Field};
use crate::traits::internal::{PrimitiveSemiring, PrimitiveRing, PrimitiveFloat};

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
    for_each!(
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
// matrix addition

gen_each!{
    @{Mn_n}
    @{Vn}
    [ [(   ) (   )] [('a,) (&'a)] ]
    [ [(   ) (   )] [('b,) (&'b)] ]
    for_each!(
        {$Mr:ident $r:tt}
        {$Vc:ident}
        [ ($($lt_a:tt)*) ($($ref_a:tt)*) ]
        [ ($($lt_b:tt)*) ($($ref_b:tt)*) ]
    ) => {
        // matrix + matrix
        impl<$($lt_a)* $($lt_b)* X: Semiring> Add<$($ref_b)* $Mr<$Vc<X>>> for $($ref_a)* $Mr<$Vc<X>>
        where X: PrimitiveSemiring,
        {
            type Output = $Mr<$Vc<X>>;

            #[inline]
            fn add(self, other: $($ref_b)* $Mr<$Vc<X>>) -> Self::Output
            { mat::from_fn(|r, c| self[r][c] + other[r][c]) }
        }

        // matrix - matrix
        impl<$($lt_a)* $($lt_b)* X: Ring> Sub<$($ref_b)* $Mr<$Vc<X>>> for $($ref_a)* $Mr<$Vc<X>>
        where X: PrimitiveRing,
        {
            type Output = $Mr<$Vc<X>>;

            #[inline]
            fn sub(self, other: $($ref_b)* $Mr<$Vc<X>>) -> Self::Output
            { mat::from_fn(|r, c| self[r][c] - other[r][c]) }
        }
    }
}

// ---------------------------------------------------------------------------
// unary ops

gen_each!{
    @{Vn}
    [ [(   ) (   )] [('a,) (&'a)] ]
    for_each!(
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

gen_each!{
    @{Mn_n}
    @{Vn}
    [ [(   ) (   )] [('a,) (&'a)] ]
    for_each!(
        {$Mr:ident $r:tt}
        {$Vc:ident}
        [ ($($lt_a:tt)*) ($($ref_a:tt)*) ]
    ) => {
        // -matrix
        impl<$($lt_a)* X: Ring> Neg for $($ref_a)* $Mr<$Vc<X>>
        where X: PrimitiveRing,
        {
            type Output = $Mr<$Vc<X>>;

            #[inline]
            fn neg(self) -> Self::Output
            { mat::from_fn(|r, c| -self.0[r][c]) }
        }
    }
}

// ---------------------------------------------------------------------------
// vector-scalar ops

// scalar `op` vector
gen_each!{
    @{Vn}
    // NOTE: the orphan rules prevent us from impl-ing these ops "for X" so
    //       we must generate a separate impl for each Semiring type rather than
    //       being generic over X: Semiring
    @{semiring}
    [ [(   ) (   )] [('a,) (&'a)] ]
    for_each!(
        {$Vn:ident}
        {$X:ty}
        [ ($($lt_a:tt)*) ($($ref_a:tt)*) ]
    ) => {
        // scalar * vector
        impl<$($lt_a)*> Mul<$($ref_a)* $Vn<$X>> for $X {
            type Output = $Vn<$X>;

            #[inline(always)]
            fn mul(self, vector: $($ref_a)* $Vn<$X>) -> Self::Output
            {
                // NOTE: we know precisely the set of types this is implemented for,
                //       and they all have commutative Mul impls.
                vector * self
            }
        }
    }
}

// vector `op` scalar
gen_each!{
    @{Vn}
    [ [(   ) (   )] [('a,) (&'a)] ]
    for_each!(
        {$Vn:ident}
        [ ($($lt_a:tt)*) ($($ref_a:tt)*) ]
    ) => {
        // vector * scalar
        impl<$($lt_a)* X: Semiring> Mul<X> for $($ref_a)* $Vn<X>
        where X: PrimitiveSemiring,
        {
            type Output = $Vn<X>;

            #[inline]
            fn mul(self, scalar: X) -> Self::Output
            { vee::from_fn(|k| self[k] * scalar) }
        }

        // vector / scalar
        impl<$($lt_a)* X: Field> Div<X> for $($ref_a)* $Vn<X>
        where X: PrimitiveFloat,
        {
            type Output = $Vn<X>;

            #[inline]
            fn div(self, scalar: X) -> Self::Output
            { vee::from_fn(|k| self[k] / scalar) }
        }

        // No modulus, which hardly makes sense for vectors anyways
        // except for the special case of `% 1.0`.
    }
}

// ---------------------------------------------------------------------------
// matrix-scalar ops

// scalar `op` matrix
gen_each!{
    @{Mn_n}
    @{Vn}
    // NOTE: the orphan rules prevent us from impl-ing these ops "for X" so
    //       we must generate a separate impl for each Semiring type rather than
    //       being generic over X: Semiring
    @{semiring}
    [ [(   ) (   )] [('a,) (&'a)] ]
    for_each!(
        {$Mr:ident $r:tt}
        {$Vc:ident}
        {$X:ty}
        [ ($($lt_a:tt)*) ($($ref_a:tt)*) ]
    ) => {
        // scalar * matrix
        impl<$($lt_a)*> Mul<$($ref_a)* $Mr<$Vc<$X>>> for $X {
            type Output = $Mr<$Vc<$X>>;

            #[inline(always)]
            fn mul(self, matrix: $($ref_a)* $Mr<$Vc<$X>>) -> Self::Output
            {
                // NOTE: we know precisely the set of types this is implemented for,
                //       and they all have commutative Mul impls.
                matrix * self
            }
        }
    }
}

// matrix `op` scalar
gen_each!{
    @{Mn_n}
    @{Vn}
    [ [(   ) (   )] [('a,) (&'a)] ]
    for_each!(
        {$Mr:ident $r:tt}
        {$Vc:ident}
        [ ($($lt_a:tt)*) ($($ref_a:tt)*) ]
    ) => {
        // matrix * scalar
        impl<$($lt_a)* X: Semiring> Mul<X> for $($ref_a)* $Mr<$Vc<X>>
        where X: PrimitiveSemiring,
        {
            type Output = $Mr<$Vc<X>>;

            #[inline]
            fn mul(self, scalar: X) -> Self::Output
            { mat::from_fn(|r, c| self.0[r][c] * scalar) }
        }

        // matrix / scalar
        impl<$($lt_a)* X: Field> Div<X> for $($ref_a)* $Mr<$Vc<X>>
        where X: PrimitiveFloat,
        {
            type Output = $Mr<$Vc<X>>;

            #[inline]
            fn div(self, scalar: X) -> Self::Output
            { mat::from_fn(|r, c| self.0[r][c] / scalar) }
        }
    }
}

// ---------------------------------------------------------------------------
// assign ops (general)

gen_each!{
    [
        {V2 X} {V3 X} {V4 X}
        {M2 V} {M3 V} {M4 V}
    ]
    for_each!(
        {$Cn:ident $T:ident}
    ) => {
        // vector += vector;
        // matrix += matrix;
        impl<$T, B> AddAssign<B> for $Cn<$T> where for<'a> &'a Self: Add<B, Output=Self> {
            #[inline(always)]
            fn add_assign(&mut self, rhs: B)
            { *self = &*self + rhs; }
        }

        // vector -= vector;
        // matrix -= matrix;
        impl<$T, B> SubAssign<B> for $Cn<$T> where for<'a> &'a Self: Sub<B, Output=Self> {
            #[inline(always)]
            fn sub_assign(&mut self, rhs: B)
            { *self = &*self - rhs; }
        }

        // vector *= scalar;
        // vector *= matrix;   (how fortunate that we primarily use row vectors!)
        // matrix *= scalar;
        // matrix *= matrix;
        impl<$T, B> MulAssign<B> for $Cn<$T> where for<'a> &'a Self: Mul<B, Output=Self> {
            #[inline(always)]
            fn mul_assign(&mut self, rhs: B)
            { *self = &*self * rhs; }
        }

        // vector /= scalar;
        // matrix /= scalar;
        impl<$T, B> DivAssign<B> for $Cn<$T> where for<'a> &'a Self: Div<B, Output=Self> {
            #[inline(always)]
            fn div_assign(&mut self, rhs: B)
            { *self = &*self / rhs; }
        }
    }
}

// ---------------------------------------------------------------------------

// vector * matrix
gen_each!{
    [{2} {3} {4}]
    [{2} {3} {4}]
    [ [(   ) (   )] [('v,) (&'v)] ]
    [ [(   ) (   )] [('m,) (&'m)] ]
    for_each!(
        {$r:tt}
        {$c:tt}
        [ ($($lt_v:tt)*) ($($ref_v:tt)*) ]
        [ ($($lt_m:tt)*) ($($ref_m:tt)*) ]
    ) => {
        // matrix * column vector
        impl<$($lt_v)* $($lt_m)* X: Semiring> Mul<$($ref_v)* V![$c, X]> for $($ref_m)* M![$r, V![$c, X]]
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
        impl<$($lt_v)* $($lt_m)* X: Semiring> Mul<$($ref_m)* M![$r, V![$c, X]]> for $($ref_v)* V![$r, X]
        where X: PrimitiveSemiring,
        {
            type Output = V![$c, X];

            #[inline]
            fn mul(self, other: $($ref_m)* M![$r, V![$c, X]]) -> Self::Output {
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
    [ [(   ) (   )] [('a,) (&'a)] ]
    [ [(   ) (   )] [('b,) (&'b)] ]
    for_each!(
        {$r:tt}
        {$k:tt}
        {$c:tt}
        [ ($($lt_a:tt)*) ($($ref_a:tt)*) ]
        [ ($($lt_b:tt)*) ($($ref_b:tt)*) ]
    ) => {
        // matrix * matrix
        impl<$($lt_a)* $($lt_b)* X: Semiring> Mul<$($ref_b)* M![$k, V![$c, X]]> for $($ref_a)* M![$r, V![$k, X]]
        where X: PrimitiveSemiring,
        {
            type Output = M![$r, V![$c, X]];

            #[inline]
            fn mul(self, other: $($ref_b)* M![$k, V![$c, X]]) -> Self::Output {
                mat::from_fn(|r,c| (0..$k).map(|i| self[r][i] * other[i][c]).sum())
            }
        }
    }
}

// ---------------------------------------------------------------------------

// fmt traits apply the format to each element for convenience.
gen_each!{
    [
        {V2 X} {V3 X} {V4 X}
        {M2 V} {M3 V} {M4 V}
    ]
    [
        // Note: the inclusion of Display in this list is a necessary evil, because
        //       there's no other way to get output like `[1.0000, 0.3333]`,
        //       which is kind of, you know, THE motivating use-case.
        {Binary} {LowerExp} {LowerHex} {Display}
        {Octal} {Pointer} {UpperExp} {UpperHex}
    ]
    for_each!(
        {$Cn:ident $T:ident}
        {$Format:ident}
    ) => {
        impl<$T: fmt::$Format> fmt::$Format for $Cn<$T> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "[")?;
                fmt::$Format::fmt(&self[0], f)?;
                for x in &self[1..] {
                    write!(f, ", ")?;
                    fmt::$Format::fmt(x, f)?;
                }
                write!(f, "]")?;
                Ok(())
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
