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

use ::num_integer::Integer;
use ::rsp2_array_types::{M33, M22, mat};

// FIXME: (low priority) sloppy/rushed api, poor coverage of implementors
//
//        It was originally written as generic code in array-types, but then after moving it
//        to this crate I discovered that array-types does not expose PrimitiveRing and etc.
//        Everything here was changed to a single integer type so that I didn't need to write a
//        bunch of overly abstract bounds like `Mat33<X>: Mul<Mat33<X>, Output = Mat33<X>>`.

/// Result of Hermite Normal Form decomposition, with some extra tidbits.
///
/// See the documentation of `Hnf::hnf` regarding conventions.
#[derive(Debug, Clone)]
pub struct HnfDecomp<M> {
    // FIXME probably shouldn't use public fields, for the sake of invariant protection.
    //       (Then again, why would a consumer even want to mutate the fields?)

    /// The Hermite Normal Form matrix.
    pub hnf: M,

    /// The unimodular matrix that transforms the original matrix into `hnf`.
    pub transform: M,

    /// The inverse of `transform`.
    pub transform_inv: M,
}

// used for intermediate results of the computation that are not yet HNF.
struct UnimodularState<M> {
    /// The transformed matrix.
    matrix: M,

    /// The unimodular matrix that transforms the original matrix into `transformed`.
    transform: M,

    /// The inverse of `transform`.
    transform_inv: M,
}

fn invertible_hnf_33(input: M33<i64>) -> HnfDecomp<M33<i64>> {
    assert_ne!(input.det(), 0);
    let UnimodularState { matrix, transform, transform_inv } = {
        UnimodularState::start(input)
            // simulate euclid's algorithm on pairs of rows
            .make_gcd(2, 0, 2)
            .make_reduced(2, 0, 2) // make the element zero
            .make_gcd(2, 1, 2)
            .make_reduced(2, 1, 2)
            .make_gcd(1, 0, 1)
            .make_reduced(1, 0, 1)
            .maybe_flip_sign()
            // canonicalize by reducing
            .make_reduced(1, 2, 1)
            .make_reduced(0, 1, 0)
            .make_reduced(0, 2, 0)
    };

    let out = HnfDecomp { hnf: matrix, transform, transform_inv };

    if cfg!(debug_assertions) {
        out.validate(input);
    }
    out
}

impl HnfDecomp<M33<i64>>
{
    // This tests all necessary conditions to ensure that the result is correct.
    fn validate(&self, original: M33<i64>) {
        use std::cmp::Ordering;

        let HnfDecomp { hnf, transform, transform_inv } = *self;
        for r in 0..3 {
            for c in 0..3 {
                // reminder: pivots are on the main diagonal because we restricted ourselves
                //           to invertible matrices
                match r.cmp(&c) {
                    Ordering::Less => assert_eq!(hnf[r][c], 0, "{:?}", hnf),
                    Ordering::Equal => assert!(hnf[r][c] > 0, "{:?}", hnf),
                    Ordering::Greater => {
                        assert!(0 <= hnf[r][c] && hnf[r][c] < hnf[c][c], "{:?}", hnf)
                    },
                }
            }
        }
        assert_eq!(transform * original, hnf);
        assert_eq!(transform_inv * hnf, original);
    }
}

impl UnimodularState<M33<i64>>
{
    fn start(m: M33<i64>) -> Self {
        UnimodularState {
            matrix: m,
            transform: M33::eye(),
            transform_inv: M33::eye(),
        }
    }

    fn transform_by(self, unimodular: M33<i64>, unimodular_inv: M33<i64>) -> Self {
        debug_assert_eq!(unimodular * unimodular_inv, M33::eye());
        UnimodularState {
            matrix: unimodular * self.matrix,
            transform: unimodular * self.transform,
            transform_inv: self.transform_inv * unimodular_inv,
        }
    }

    // Apply a unimodular transformation such that m[gcd_row][col] becomes the gcd of
    // the original values of m[gcd_row][col] and m[other_row][col]
    fn make_gcd(self, col: usize, other_row: usize, gcd_row: usize) -> Self {
        use ::rsp2_numtheory::extended_gcd;

        let matrix = self.matrix;
        let data_g = extended_gcd(matrix[other_row][col], matrix[gcd_row][col]);
        let (u10, u11) = data_g.coeffs;

        // That gives us one row of a 2x2 unimodular operation.  Constraining its determinant to 1
        // gives us  u00*u11 - u01*u10 = 1 as an equation for the other row. Look at this carefully,
        // and see that it is a Bezout equation for the other two elements!
        let data_u = extended_gcd(u11, u10);

        // FIXME: We have a bug; when `matrix[other_row][col] == matrix[gcd_row][col] == 0`,
        //        extended_gcd returns (0, 0) for the Bezout coefficients, in which case they
        //        fail to be coprime. (and in fact are unsuitable for use in the rest of this
        //        function)
        debug_assert_eq!(data_u.gcd, 1, "bezout coefficients not coprime?!");
        let (u00, neg_u01) = data_u.coeffs;
        let u01 = -neg_u01;

        let unimodular = m22_to_m33(mat::from_array([[u00, u01], [u10, u11]]), other_row, gcd_row);
        let unimodular_inv = m22_to_m33(mat::from_array([[u11, -u01], [-u10, u00]]), other_row, gcd_row);
        let out = self.transform_by(unimodular, unimodular_inv);

        debug_assert_eq!(out.transform * out.transform_inv, M33::eye());

        assert_eq!(out.matrix[gcd_row][col], data_g.gcd);
        out
    }

    // reduces m[reduce_row][col] modulo m[gcd_row][col]
    fn make_reduced(self, col: usize, reduce_row: usize, gcd_row: usize) -> Self {
        let UnimodularState { mut matrix, mut transform, mut transform_inv } = self;

        assert!(matrix[gcd_row][col] > 0);
        let mult = -(matrix[reduce_row][col].div_floor(&matrix[gcd_row][col]));

        // simple row addition
        matrix[reduce_row] += mult * matrix[gcd_row];
        transform[reduce_row] += mult * transform[gcd_row];

        // expanding  1 == transform_inv * transform
        //              == transform_inv * row_add_op_inv * row_add_op * transform
        //              == transform_inv' * transform'
        // and expressing row_add_op as (1 + n * |dest><src|), it is easy to show
        // that  transform_inv'^T = (1 - n * |src><dest|) * transform_inv^T.
        //
        // That is, to update transform_inv, we must negate the multiplier, swap src and dest,
        // AND operate on the columns. Tricky!
        mutate_transpose(&mut transform_inv, |t| {
            t[gcd_row] -= mult * t[reduce_row]
        });

        debug_assert!(0 <= matrix[reduce_row][col] && matrix[reduce_row][col] < matrix[gcd_row][col]);

        debug_assert_eq!(transform * transform_inv, M33::eye());

        UnimodularState { matrix, transform, transform_inv }
    }

    // possibly performs a single sign flip to ensure the determinant is positive at the very end
    fn maybe_flip_sign(self) -> Self {
        let UnimodularState { mut matrix, mut transform, mut transform_inv } = self;

        // assuming that gcd and zero reduction has been performed on columns 1 and 2,
        // we should already be lower triangular with m11 and m22 guaranteed positive.
        //
        // Hence, the sign of m00 is the sign of the determinant.
        debug_assert_eq!(matrix[0][1], 0);
        debug_assert_eq!(matrix[0][2], 0);
        debug_assert_eq!(matrix[1][2], 0);
        debug_assert!(matrix[1][1] > 0);
        debug_assert!(matrix[2][2] > 0);
        if matrix[0][0] < 0 {
            matrix[0] *= -1;
            transform[0] *= -1;
            mutate_transpose(&mut transform_inv, |t| t[0] *= -1);
        }

        debug_assert_eq!(transform * transform_inv, M33::eye());

        UnimodularState { matrix, transform, transform_inv }
    }
}

fn m22_to_m33<X>(m: M22<X>, axis_0: usize, axis_1: usize) -> M33<X>
where
    X: Copy,
    M33<X>: ::num_traits::One,
{
    let mut out = M33::eye();
    out[axis_0][axis_0] = m[0][0];
    out[axis_0][axis_1] = m[0][1];
    out[axis_1][axis_0] = m[1][0];
    out[axis_1][axis_1] = m[1][1];
    out
}

fn mutate_transpose<X, F>(m: &mut M33<X>, f: F)
where
    X: Copy, // FIXME
    F: FnOnce(&mut M33<X>),
{
    let mut t = m.t();
    f(&mut t);
    *m = t.t();
}

/// Provides conversions into Hermite Normal Form.
pub trait Hnf: Sized {
    /// Convert self into Hermite Normal Form.
    ///
    /// This is currently only supported on a limited subset of matrices—in particular,
    /// they must be invertible.  This is because, even though non-invertible matrices have
    /// a well-defined HNF, the author of this code does not need to deal with them and thus does
    /// not want to have to worry about the locations of pivots.
    ///
    /// # Conventions
    ///
    /// The convention followed is row-based; a unimodular matrix `transform` is applied on the
    /// *left-hand side* of the matrix to convert it into a lower triangular form with positive
    /// determinant.
    ///
    /// # Panics
    ///
    /// Currently panics on non-invertible matrices.  Sorry.
    #[inline(never)]
    fn hnf(self) -> Self
    { self.hnf_decomp().hnf }

    /// Convert self into HNF and obtain the unimodular transform.
    ///
    /// The output satisfies:
    ///
    /// ```text,ignore
    /// transform * original == hnf
    ///
    /// transform_inv * hnf == original
    /// ```
    fn hnf_decomp(self) -> HnfDecomp<Self>;
}

impl Hnf for M33<i64> {
    fn hnf_decomp(self) -> HnfDecomp<Self> {
        assert_ne!(
            self.det(), 0,
            "hnf is currently only implemented on invertible matrices, sorry",
        );

        invertible_hnf_33(self)
    }
}

#[cfg(test)]
#[deny(unused)]
mod tests {
    use super::*;
    use ::rand::{thread_rng, Rng};

    #[test]
    fn invertible_hnf_33() {
        let matrices = {
            (0..)
                .map(|_| M33::from_fn(|_,_| thread_rng().gen_range(-20, 20 + 1)))
                // (mind: these random matrices are rarely degenerate, so simply deleting
                //        this next line is NOT enough to reliably extend this test to
                //        non-invertible matrices)
                .filter(|m| m.det() != 0)
                .take(30)
        };

        for m in matrices {
            eprintln!("testing {:?}", m);
            let decomp = m.hnf_decomp();
            eprintln!("    --> {:?}", decomp);
            decomp.validate(m);
        }
    }
}
