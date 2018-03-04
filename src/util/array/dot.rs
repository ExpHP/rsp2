// NOTE:
//  This file has a huge impact on compilation time due to the large
//    number of generated impls.
//  When editing this file you MUST benchmark compilation time before/after.

use ::{arr_from_fn, mat_from_fn};
use ::traits::{IsArray, Semiring};
use ::traits::internal::PrimitiveSemiring;

#[deprecated = "Use V2/V3/V4 (alternative for vec * matrix is coming)"]
pub fn dot<A,B,C>(a: &A, b: &B) -> C
  where A: Dot<B, Output=C>,
{ Dot::dot(a, b) }

/// Inner product of vectors and/or matrices.
pub trait Dot<M2>: IsArray
{
    type Output;

    /// Obtain a dot product by value.
    fn dot(a: &Self, b: &M2) -> Self::Output;
}

// // TODO
// pub trait WriteDot<A, B>: Sized
//   where A: Dot<B, Output=Self>,
// { fn write_dot(&mut self, a: &A, b: &B); }

gen_each!{
    @{1...16}
    impl_vec_vec_dot!({$k:expr})
    => {
        impl<T: Semiring> Dot<[T; $k]> for [T; $k]
          where T: ::traits::internal::PrimitiveSemiring,
        {
            type Output = T;
            fn dot(a: &Self, b: &Self) -> T {
                (0..$k).map(|i| a[i] * b[i]).sum()
            }
        }
    };
}

gen_each!{
    [{2} {3} {4}]
    [{2} {3} {4}]
    impl_vec_mat_dot!({$n:expr} {$k:expr})
    => {
        // (m, k) x (k, 1) -> (m, 1)
        impl<T: Semiring> Dot<nd![T; $k]> for nd![T; $n; $k]
          where T: PrimitiveSemiring,
        {
            type Output = nd![T; $n];
            fn dot(a: &Self, b: &nd![T; $k]) -> nd![T; $n] {
                arr_from_fn(|r| (0..$k).map(|i| a[r][i] * b[i]).sum())
            }
        }

        // (1,k) x (k, m)
        impl<T: Semiring> Dot<nd![T; $k; $n]> for nd![T; $k]
          where T: PrimitiveSemiring,
        {
            type Output = nd![T; $n];
            fn dot(a: &Self, b: &nd![T; $k; $n]) -> nd![T; $n] {
                arr_from_fn(|c| (0..$k).map(|i| a[i] * b[i][c]).sum())
            }
        }
    };
}

gen_each!{
    // Compile time is a big problem here.
    [{2} {3} {4}]
    [{2} {3} {4}]
    [{2} {3} {4}]
    impl_vec_mat_dot!({$r:expr} {$k:expr} {$c:expr})
    => {
        // (r, k) x (k, c) -> (r, c)
        impl<T: Semiring> Dot<nd![T; $k; $c]> for nd![T; $r; $k]
          where T: PrimitiveSemiring,
        {
            type Output = nd![T; $r; $c];
            fn dot(a: &Self, b: &nd![T; $k; $c]) -> nd![T; $r; $c] {
                mat_from_fn(|r,c| (0..$k).map(|i| a[r][i] * b[i][c]).sum())
            }
        }
    };
}


#[cfg(test)]
mod tests {
    use super::dot;

    #[test]
    fn mat_mat() {
        let eye2 = [[1, 0], [0, 1i32]];
        let eye3 = [[1, 0, 0], [0, 1, 0], [0, 0, 1i32]];

        let a = [
            [1, 2, 3],
            [4, 5, 6],
        ];

        let b = [
            [1,  1],
            [1, -1],
            [0,  1],
        ];

        let a_dot_b = [
            [3, 2],
            [9, 5],
        ];

        assert_eq!(a, dot(&eye2, &a));
        assert_eq!(a, dot(&a, &eye3));
        assert_eq!(a_dot_b, dot(&a, &b));
    }

    #[test]
    fn mat_vec() {
        let m = [
            [1, 2, 3],
            [4, 5, 6],
        ];
        assert_eq!([1, 7], dot(&m, &[4, -3, 1]));
        assert_eq!([-8, -7, -6], dot(&[4, -3], &m));
    }

    #[test]
    fn vec_vec() {
        assert_eq!(32, dot(&[1,2,3], &[4,5,6]));
    }
}
