use num_traits::Zero;

use iter::SparseIterator;
use mat::{CscMat, CsrMat};
use math::SparseSparseMath;
use math::{SelfAdd, SelfMul};
use traits::Abs;
use traits::PartialMinMax;

// This currently has a bunch of miscellaneous ideas, which need some sort of API wrapped
//  around them

//-----------------------------------------------------------

#[inline]
fn multiplied_dim(dim1: (usize, usize), dim2: (usize, usize)) -> (usize, usize) {
    let (m, n1) = dim1;
    let (n2, p) = dim2;
    if n1 == n2 {
        (m, p)
    } else {
        panic!("incompatible shapes: {:?} and {:?}", dim1, dim2)
    }
}

fn mul_csr_csc_to_csr<T>(a: CsrMat<T>, b: CscMat<T>, zero_tol: &T) -> CsrMat<T>
where
    T: SelfMul + SelfAdd + Abs + Zero + Clone + PartialOrd,
{
    let dim = multiplied_dim(a.dim(), b.dim());

    let cap = ::std::cmp::min(a.nnz(), b.nnz());
    let mut vec = Vec::with_capacity(cap);

    for (i, row) in a.rows().enumerate() {
        for (j, col) in b.columns().enumerate() {
            // sparse dot product
            let row = row.sparse_iter().sparse_cloned();
            let col = col.sparse_iter().sparse_cloned();
            let value = row.sparse_sparse_mul(col)
                .map(|(_, x)| x)
                .fold(T::zero(), |a, b| a + b);

            // written as !(x <= y) so that it is also true for NaN
            // FIXME: probably should do something similar in other places where I "sparsify" values
            if !(&(value.clone().abs()) <= zero_tol) {
                vec.push(((i, j), value));
            }
        }
    }

    // vec is sorted in row-major order, and fully in-bounds
    unsafe { CsrMat::from_row_major_iter_unchecked(dim, vec) }
}

fn mul_csr_csc_to_csc<T>(a: CsrMat<T>, b: CscMat<T>, zero_tol: &T) -> CscMat<T>
where
    T: SelfMul + SelfAdd + Abs + Zero + Clone + PartialOrd,
{
    // A B = (B^T A^T)^T
    mul_csr_csc_to_csr(b.transpose(), a.transpose(), zero_tol).transpose()
}

//--------------------------------------------------
// a terrible proposed interface which uses result types, without hardly any of the actual benefits
//  of result types; the results currently only exist to give you an opportunity to choose your
//  output format

// I don't imagine that this will scale very well. :/

/// Trait for matrix-matrix multiplication.
///
/// Some types may choose to return "result" types which require a conversion method in order to
///  obtain a result.
///
/// The currently intended semantics are to panic for incompatible shapes.
pub trait MatMatMul<RHS> {
    type Output;
    fn mat_mul(self, rhs: RHS) -> Self::Output;
}

impl<T> MatMatMul<CscMat<T>> for CsrMat<T>
where
    T: Zero,
{
    type Output = CsrCscMulResult<T>;
    fn mat_mul(self, rhs: CscMat<T>) -> CsrCscMulResult<T> {
        // panic early on bad dimension
        let _ = multiplied_dim(self.dim(), rhs.dim());

        CsrCscMulResult {
            zero_tol: Zero::zero(),
            lhs: self,
            rhs: rhs,
        }
    }
}

//--------------------------------------------------

/// Multiplication result type `CsrMat` and `CscMat`.
#[derive(Clone, Debug, PartialEq)]
#[must_use = "Result types do nothing until evaluated"]
pub struct CsrCscMulResult<T> {
    pub zero_tol: T,
    pub lhs: CsrMat<T>,
    pub rhs: CscMat<T>,
}

impl<T> CsrCscMulResult<T>
where
    T: SelfMul + SelfAdd + Abs + Zero + Clone + PartialOrd + ::std::fmt::Debug,
{
    // NOTE: This takes a borrow for consistency with other prune methods, even though in this
    //   particular case we really *do* want a value.
    /// Remove explicit zeros (within a given tolerance)
    pub fn prune(self, zero_tol: &T) -> CsrCscMulResult<T> {
        assert!(
            zero_tol >= &Zero::zero(),
            "invalid zero-tolerance: {:?}",
            *zero_tol
        );

        // `.prune(small_value)` should have no effect after `.prune(larger_value)`
        let zero_tol = zero_tol
            .partial_max(&self.zero_tol)
            .expect("received unorderable value")
            .clone();
        CsrCscMulResult {
            zero_tol: zero_tol,
            ..self
        }
    }

    /// Get the shape of the multiplied result
    #[inline]
    pub fn dim(&self) -> (usize, usize) {
        multiplied_dim(self.lhs.dim(), self.rhs.dim())
    }

    /// Get the transpose (trivially, as another `CsrCscMulResult`).
    #[inline(always)]
    pub fn transpose(self) -> CsrCscMulResult<T> {
        CsrCscMulResult {
            lhs: self.rhs.transpose(),
            rhs: self.lhs.transpose(),
            ..self
        }
    }

    /// Evaluate as a `CsrMat`
    pub fn into_csr(self) -> CsrMat<T> {
        assert!(
            self.zero_tol >= Zero::zero(),
            "invalid zero-tolerance: {:?}",
            self.zero_tol
        );
        mul_csr_csc_to_csr(self.lhs, self.rhs, &self.zero_tol)
    }

    /// Evaluate as a `CscMat`
    pub fn into_csc(self) -> CscMat<T> {
        assert!(
            self.zero_tol >= Zero::zero(),
            "invalid zero-tolerance: {:?}",
            self.zero_tol
        );
        mul_csr_csc_to_csc(self.lhs, self.rhs, &self.zero_tol)
    }
}

//--------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use SparseVec;
    use mat::{CscMat, CsrMat};

    use num_traits::{One, Zero};
    use std::ops::Sub;

    #[test]
    fn test_csr_csc_pruning() {
        // create a 2x2 matrix filled with `x`, by multiplying together two matrices which are
        //  designed to produce such a result through cancellations.
        fn make_prod<T: Copy>(x: T) -> CsrCscMulResult<T>
        where
            T: One + Zero + PartialEq + Sub<T, Output = T>,
        {
            let one = <T as One>::one();
            let csr = CsrMat::from_rows(
                2,
                vec![
                    SparseVec::from_dense(vec![one, x - one]),
                    SparseVec::from_dense(vec![x - one, one]),
                ],
            );
            let csc = CscMat::from_columns(
                2,
                vec![
                    SparseVec::from_dense(vec![one, one]),
                    SparseVec::from_dense(vec![one, one]),
                ],
            );
            csr.mat_mul(csc)
        }

        // Exact zeros are always pruned.
        let prod = make_prod(0i64); // use integral type to ensure exact zeros
        assert_eq!(prod.clone().into_csr().nnz(), 0);
        assert_eq!(prod.clone().into_csc().nnz(), 0);

        // Approximate zeros are NOT pruned by default.
        let prod = make_prod(1e-10);
        assert_eq!(prod.clone().into_csr().nnz(), 4);
        assert_eq!(prod.clone().into_csc().nnz(), 4);

        // Still not large enough tolerance...
        let prod = make_prod(1e-10).prune(&1e-12);
        assert_eq!(prod.clone().into_csr().nnz(), 4);
        assert_eq!(prod.clone().into_csc().nnz(), 4);

        // That should do it.
        let prod = make_prod(1e-10).prune(&1e-7);
        assert_eq!(prod.clone().into_csr().nnz(), 0);
        assert_eq!(prod.clone().into_csc().nnz(), 0);

        // And pruning again with a smaller tolerance should not "undo" it.
        let prod = make_prod(1e-10).prune(&1e-7).prune(&1e-12);
        assert_eq!(prod.clone().into_csr().nnz(), 0);
        assert_eq!(prod.clone().into_csc().nnz(), 0);
    }

    // FIXME there should be more tests
}
