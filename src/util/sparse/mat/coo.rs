//! COOrdinate-storage sparse matrices.

use std::ops::Add;
use mat::{CscMat, CsrMat};

use ::rsp2_soa_ops::{Perm};

/// A sparse format which simply stores triplets of `(row, column, value)`.
///
/// The COO format is a suitable format for the initial construction of sparse
/// matrices.  However, it is a poor format for most other operations, and
/// should be converted to a different type of matrix as soon as possible.
///
/// COO matrices are allowed to store multiple elements at the same coordinates,
/// with the "true" value at that position being the sum of the values.  This is
/// in line with its intended use-case as a format for sparse matrix construction.
///
/// Equality testing between two COO matrices is "dumb" and expects the coordinates
/// to match exactly, duplicates and all.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CooMat<T> {
    dim: (usize, usize),
    vec: Vec<((usize, usize), T)>,
}

// Helper macro for validating positions in an iterator against
// the matrix dimension.  Returns the iterator.
// (well, technically, returns an Inspect adapter, but who can tell the difference?)
macro_rules! check_positions {
    ($dim:expr, $iter:expr) => {
        $iter.inspect(|&(ref pos, _)| {
            assert!(
                pos.0 < $dim.0 && pos.1 < $dim.1,
                "Element position {:?} invalid for dimension {:?}!",
                pos,
                $dim
            );
        })
    };
}

impl<T> CooMat<T> {
    /// Create a zero matrix.
    #[inline]
    pub fn new(dim: (usize, usize)) -> CooMat<T> {
        let vec = vec![];
        CooMat { dim, vec }
    }

    /// Sort the elements by row, then column.
    #[inline]
    pub fn sort_row_major(&mut self) {
        self.vec.sort_by_key(|&((r, c), _)| (r, c))
    }

    /// Sort the elements by column, then row.
    #[inline]
    pub fn sort_column_major(&mut self) {
        self.vec.sort_by_key(|&((r, c), _)| (c, r))
    }

    /// Transpose the matrix, consuming it.
    #[cfg_attr(feature = "nightly", must_use = "not an in-place operation!")]
    #[inline]
    pub fn transpose(self) -> Self {
        let CooMat { dim: (nr, nc), mut vec } = self;

        for (pos, _) in &mut vec {
            *pos = (pos.1, pos.0);
        }
        let dim = (nc, nr);
        CooMat { dim, vec }
    }

    /// Sums sequential elements at the same position.
    ///
    /// Item format is `((row,col), value)`
    #[inline]
    pub fn into_reduced_iter(self) -> IntoReducedIter<T>
    where
        T: Add<T, Output = T>,
    {
        IntoReducedIter::new(self.vec)
    }

    /// Convert into CSR format.
    ///
    /// Runtime complexity is `O(n log n)` with `n` being the number of explicit entries in the
    /// matrix (including duplicates), as it must sort the entries first.
    pub fn into_csr(mut self) -> CsrMat<T>
    where
        T: Add<T, Output = T>,
    {
        self.sort_row_major();
        unsafe { CsrMat::from_row_major_iter_unchecked(self.dim, self.into_reduced_iter()) }
    }

    /// Convert into CSC format.
    ///
    /// Runtime complexity is `O(n log n)` with `n` being the number of explicit entries in the
    /// matrix (including duplicates), as it must sort the entries first.
    pub fn into_csc(mut self) -> CscMat<T>
    where
        T: Add<T, Output = T>,
    {
        self.sort_column_major();
        unsafe { CscMat::from_column_major_iter_unchecked(self.dim, self.into_reduced_iter()) }
    }

    /// Construct from an iterator of `((row,col), value)` tuples.
    ///
    /// Note that a dimension argument is required, which prevents providing this through
    ///  the core library's `FromIterator`. The positions in the iterator will be validated
    ///  against the provided dimension.
    // TODO: that kind of issue is exactly the reason why ShapedSparseIterator exists.
    // Consider possible solutions. Certainly, one option would be to make Shape parametric
    // over its index type, but... ick.
    pub fn from_iter<I>(dim: (usize, usize), it: I) -> Self
    where
        I: IntoIterator<Item = ((usize, usize), T)>,
    {
        let vec = check_positions!(dim, it.into_iter()).collect();
        CooMat { dim, vec }
    }

    /// Modifies the indices in such a way as to effectively permute the rows of the dense matrix.
    #[cfg_attr(feature = "nightly", must_use = "not an in-place operation!")]
    pub fn permute_rows(mut self, perm: &Perm) -> Self {
        assert_eq!(self.dim.0, perm.len());

        let perm_inv = perm.inverted();
        for ((row, _), _) in &mut self.vec {
            *row = perm_inv[*row] as usize;
        }
        self
    }

    /// Modifies the indices in such a way as to effectively permute the columns of the dense
    /// matrix.
    #[cfg_attr(feature = "nightly", must_use = "not an in-place operation!")]
    pub fn permute_columns(mut self, perm: &Perm) -> Self {
        assert_eq!(self.dim.1, perm.len());

        let perm_inv = perm.inverted();
        for ((_, col), _) in &mut self.vec {
            *col = perm_inv[*col] as usize;
        }
        self
    }
}

impl<T> Extend<((usize, usize), T)> for CooMat<T> {
    fn extend<I>(&mut self, it: I)
    where
        I: IntoIterator<Item = ((usize, usize), T)>,
    {
        let dim = self.dim.clone();
        self.vec.extend(check_positions!(dim, it.into_iter()));
    }
}

//-------------------------------------------------------------

#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
pub struct IntoReducedIter<T> {
    iter: ::std::iter::Peekable<::std::vec::IntoIter<((usize, usize), T)>>,
}

impl<T> IntoReducedIter<T> {
    #[inline]
    pub fn new(vec: Vec<((usize, usize), T)>) -> Self {
        IntoReducedIter {
            iter: vec.into_iter().peekable(),
        }
    }
}

impl<T> Iterator for IntoReducedIter<T>
where
    T: Add<T, Output = T>,
{
    type Item = ((usize, usize), T);
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().and_then(|(pos, mut val)| {
            while let Some(&(next_pos, _)) = self.iter.peek() {
                if next_pos == pos {
                    val = val + self.iter.next().unwrap().1;
                } else {
                    break;
                }
            }
            Some((pos, val))
        })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        // any number of elements may be reduced together
        let (lo, hi) = self.iter.size_hint();
        (::std::cmp::min(lo, 1), hi)
    }
}

// No ExactSizeIterator; elements may be reduced together
// No DoubleEndedIterator; too much trouble since we need to peek

//-------------------------------------------------------------

#[test]
fn test_reduced_iters() {
    let vec = vec![
        ((1, 4), 5i32),
        ((0, 2), 3),
        ((1, 2), -2),
        ((0, 4), 2),
        ((1, 2), 12),
    ];
    let mat = CooMat::from_iter((5, 5), vec);

    // column major
    let mut cmajor = mat.clone();
    cmajor.sort_column_major();
    let mut it = cmajor.into_reduced_iter();
    assert_eq!(it.next(), Some(((0, 2), 3)));
    assert_eq!(it.next(), Some(((1, 2), 10))); // this element is reduced from a sum
    assert_eq!(it.next(), Some(((0, 4), 2)));
    assert_eq!(it.next(), Some(((1, 4), 5)));
    assert_eq!(it.next(), None);
    assert_eq!(it.next(), None);

    // row major
    let mut rmajor = mat.clone();
    rmajor.sort_row_major();
    let mut it = rmajor.into_reduced_iter();
    assert_eq!(it.next(), Some(((0, 2), 3)));
    assert_eq!(it.next(), Some(((0, 4), 2)));
    assert_eq!(it.next(), Some(((1, 2), 10))); // this element is reduced from a sum
    assert_eq!(it.next(), Some(((1, 4), 5)));
    assert_eq!(it.next(), None);
    assert_eq!(it.next(), None);

    // possible edge case: no elements
    let mat = CooMat::<i32>::from_iter((5, 5), vec![]);
    let out: Vec<_> = mat.into_reduced_iter().collect();
    assert_eq!(out, vec![]);

    // possible edge case: first or last element must be reduced
    let vec = vec![((1, 2), 3i32); 5]; // just make 5 identical elements
    let mut it = CooMat::from_iter((5, 5), vec).into_reduced_iter();
    assert_eq!(it.next(), Some(((1, 2), 15)));
    assert_eq!(it.next(), None);
}

#[should_panic(expected = "invalid for dimension")]
#[test]
fn test_extend_bad_row() {
    CooMat::<i32>::new((32, 48)).extend(vec![((32, 16), 17)]);
}

#[should_panic(expected = "invalid for dimension")]
#[test]
fn test_extend_bad_col() {
    CooMat::<i32>::new((32, 48)).extend(vec![((0, 49), 17)]);
}

#[should_panic(expected = "invalid for dimension")]
#[test]
fn test_from_iter_bad_row() {
    CooMat::<i32>::from_iter((32, 48), vec![((32, 16), 17)]);
}

#[should_panic(expected = "invalid for dimension")]
#[test]
fn test_from_iter_bad_col() {
    CooMat::<i32>::from_iter((32, 48), vec![((0, 49), 17)]);
}

#[test]
fn test_to_cs() {
    // ALWAYS. ALWAYS. ALWAYS. TEST. ZERO LENGTH.
    // JUST DO ITTTTTTT!
    let coo = CooMat::<i32>::from_iter((0, 0), vec![]);
    assert_eq!(coo.clone().into_csr(), CsrMat::eye(0));
    assert_eq!(coo.clone().into_csc(), CscMat::eye(0));

    // check that dimensions are assigned properly
    let coo = CooMat::<i32>::from_iter((28, 79), vec![]);
    assert_eq!(coo.clone().into_csr(), CsrMat::zero((28, 79)));
    assert_eq!(coo.clone().into_csc(), CscMat::zero((28, 79)));

    // elements that require reduction
    let coo = CooMat::from_iter((1, 1), vec![((0, 0), 5i32); 4]);
    assert_eq!(coo.clone().into_csr(), CsrMat::from_diagonal(vec![20]));
    assert_eq!(coo.clone().into_csc(), CscMat::from_diagonal(vec![20]));

    // a full 2x2 matrix
    let coo = CooMat::from_iter(
        (2, 2),
        vec![((0, 0), 5i32), ((0, 1), 2), ((1, 0), 3), ((1, 1), 4)],
    );
    let rows: Vec<_> = coo.clone()
        .into_csr()
        .rows()
        .map(|x| x.to_dense())
        .collect();
    let cols: Vec<_> = coo.clone()
        .into_csc()
        .columns()
        .map(|x| x.to_dense())
        .collect();
    assert_eq!(rows, vec![vec![5, 2], vec![3, 4]]);
    assert_eq!(cols, vec![vec![5, 3], vec![2, 4]]);
}

#[test]
fn test_permute() {
    let coo = CooMat::<i32>::from_iter((0, 0), vec![]);
    assert_eq!(coo.clone().permute_columns(&Perm::eye(0)), coo.clone());
    assert_eq!(coo.clone().permute_rows(&Perm::eye(0)), coo.clone());

    let coo = CooMat::<i32>::from_iter((3, 4), vec![((1, 1), 9)]);
    assert_eq!(
        coo.clone().permute_rows(&Perm::eye(3).shift_right(1)),
        CooMat::<i32>::from_iter((3, 4), vec![((2, 1), 9)]),
    );

    let coo = CooMat::<i32>::from_iter((3, 4), vec![((1, 1), 9)]);
    assert_eq!(
        coo.clone().permute_columns(&Perm::eye(4).shift_right(1)),
        CooMat::<i32>::from_iter((3, 4), vec![((1, 2), 9)]),
    );
}

#[test]
fn test_transpose() {
    let coo = CooMat::<i32>::from_iter((0, 0), vec![]);
    assert_eq!(coo.clone().transpose(), coo.clone());

    let coo = CooMat::<i32>::from_iter((3, 4), vec![((1, 0), 9)]);
    assert_eq!(
        coo.clone().transpose(),
        CooMat::<i32>::from_iter((4, 3), vec![((0, 1), 9)]),
    );
}
