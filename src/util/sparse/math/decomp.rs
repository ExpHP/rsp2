use num_traits::{One, Zero};
use std::ops::{Add, Div, Mul, Sub};
pub struct LdlDecomp<T> {
    pub lmat: CsrMat<T>,
    pub dvec: Vec<T>,
}

use iter::IntoSparseIterator;
use iter::ShapedSparseIterator;
use iter::SparseIterator;
use mat::CsrMat;
use math::{SparseDenseMath, SparseSparseMath};
use traits::Abs;
use traits::DenseIndex;
use traits::Shape;
use vec::SparseVec;

// helper macro for ldl conversion
// performs an inner product between two sparse vectors and a dense vector, over indices < k.
//
// it's a macro instead of a fn to save me the pain of having to update trait bounds as things change
macro_rules! ldl_inner_prod {
    ($k:expr, $s1:expr, $s2:expr, $d:expr) => {
        $s1.slice(0..$k)
            .sparse_iter()
            .sparse_cloned()
            .sparse_sparse_mul($s2.slice(0..$k).sparse_iter().sparse_cloned())
            .sparse_dense_mul(&$d[..$k])
            .map(|(_, x)| x)
            .fold(Zero::zero(), |a, b| a + b)
    };
}

// And thus we see how horribly inadequate our abstractions still are.

pub fn csr_into_ldl<T>(csr: CsrMat<T>, zero_tol: &T) -> LdlDecomp<T>
where
    T: Clone
        + PartialOrd
        + Zero
        + One
        + Abs
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + Sub<T, Output = T>
        + Div<T, Output = T>,
{
    // TODO can we cheaply confirm that csr is actually hermitian?
    // Perhaps the cost of the decomposition is so great that the cost of computing the transpose
    //  to check symmetry would be negligible.

    assert!(csr.is_square());
    let dim = csr.dim().0;

    // get lower triangle and prune explicit zeros
    let mat = CsrMat::from_rows(
        dim,
        csr.rows().enumerate().map(
            |(i, row)| {
                row.slice(0..i + 1)
                    .to_sparse_vec()
                    .prune(zero_tol)
                    .into_sparse_iter()
                    .as_shaped(dim)
            }, // FIXME hack to extend shape back
        ),
    );

    // It's a bit of a shame how complicated the following snippet is, because the algorithm
    //  for this is actually beautifully concise when written as a fully in-place algorithm
    //  on dense matrices; Each element of the input matrix is only ever needed once, and
    //  that is when you are computing the element of L (or D) at the same position.
    // Unfortunately, sparse matrices, aliasing rules, and my incompetence have conspired to make
    //  this function incomprehensible!

    let mut lmat = CsrMat::zero((0, dim));
    let mut diag = vec![];

    for (i, old_row) in mat.rows().enumerate() {
        // scope so we can immutably borrow lmat
        let row = {
            let xs = old_row.slice(0..i).sparse_iter().sparse_cloned().densify();
            let lrows = lmat.rows();

            // Compute elements left of the diagonal for this row
            let mut new_row = SparseVec::zero(0);
            for (k, (x, lrow)) in xs.zip(lrows).enumerate() {
                assert_eq!(new_row.dim(), k);

                let new_x = (x - ldl_inner_prod!(k, new_row, lrow, diag)) / diag[k].clone();

                // must prune zeros as they come, otherwise this becomes O(n^3)
                let new_x = if &new_x.clone().abs() <= zero_tol {
                    Zero::zero()
                } else {
                    new_x
                };

                new_row.push_dense(new_x);
            }

            // Give L a unit diagonal
            new_row.push_dense(One::one());
            debug_assert_eq!(new_row.dim(), i + 1);

            // Rest of row is zero
            new_row.reshape(dim);

            // Compute next element of D
            let d = old_row.get_dense(i) - ldl_inner_prod!(i, new_row, new_row, diag);
            diag.push(d);

            new_row
        };
        lmat.push_row(row);
    }

    debug_assert_eq!(lmat.dim().0, dim);
    debug_assert_eq!(lmat.dim().1, dim);
    debug_assert_eq!(diag.len(), dim);
    LdlDecomp {
        lmat: lmat,
        dvec: diag,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mat::CsrMat;
    use vec::SparseVec;

    #[test]
    fn test_ldl() {
        // Wikipedia example
        let csr = CsrMat::from_rows(
            3,
            vec![
                SparseVec::from_dense(vec![4., 12., -16.]),
                SparseVec::from_dense(vec![12., 37., -43.]),
                SparseVec::from_dense(vec![-16., -43., 98.]),
            ],
        );
        let ldl = csr_into_ldl(csr, &1e-14f64);

        let expected_dvec = vec![4., 1., 9.];
        let expected_lmat = vec![vec![1., 0., 0.], vec![3., 1., 0.], vec![-4., 5., 1.]];

        assert_close!(abs=1e-12, ldl.dvec, expected_dvec);
        for (actual, expected) in ldl.lmat.rows().zip(expected_lmat) {
            assert_close!(abs=1e-12, actual.to_dense(), expected);
        }
    }
}

// FIXME this is hax so I can use the lib; I don't know how I want to provide this API
impl<T> CsrMat<T>
where
    T: Clone
        + PartialOrd
        + Zero
        + One
        + Abs
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + Sub<T, Output = T>
        + Div<T, Output = T>,
{
    pub fn into_ldl(self, zero_tol: &T) -> LdlDecomp<T> {
        csr_into_ldl(self, zero_tol)
    }
}
