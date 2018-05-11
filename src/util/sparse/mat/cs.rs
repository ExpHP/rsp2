//! Compressed Sparse Row (CSR) and Compressed Sparse Column (CSC) matrices

use iter::IntoSparseIterator;
use iter::SparseIterator;
use iter::{ShapedSparseIterator, SortedSparseIterator, UniqueSparseIterator};
use slice::{RawSparseSlice, SparseSlice};
use traits::Shape;

use ::num_traits::One;

/// A matrix in Compressed Sparse Row (CSR) format.
// TODO: I should probably cite a paper here or something.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct CsrMat<T> {
    dim: (usize, usize),
    val: Vec<T>,
    idx: Vec<usize>,
    ptr: Vec<usize>, // "insertion indices" between rows
}

/// A matrix in Compressed Sparse Column (CSC) format.
// TODO: I should probably cite a paper here or something.
#[derive(Clone, PartialEq, Eq, Debug)]
#[allow(non_snake_case)]
pub struct CscMat<T> {
    T: CsrMat<T>, // self.T is the transpose, geddit?  Please don't hurt me
}

// fun fact: you can, in fact, substitute a macro name into a macro, and invoke it as a macro,
//  allowing you to be "generic" over assert! and debug_assert!
macro_rules! validate_csr {
    (always, $csr:expr) => {
        validate_csr!([assert, assert_eq], $csr)
    };
    (debug, $csr:expr) => {
        validate_csr!([debug_assert, debug_assert_eq], $csr)
    };
    ([$assert:ident, $assert_eq:ident], $csr:expr) => {{
        let csr = $csr;
        {
            let &CsrMat {
                ref ptr,
                ref val,
                ref idx,
                ref dim,
            } = &csr;

            // ptr is a sorted list of "endpoints", of length nrows + 1
            $assert!(ptr.len() == dim.0 + 1);
            $assert!(ptr[0] == 0usize);
            $assert!(ptr.last().unwrap() == &idx.len()); // safe unwrap; see len check
            $assert!(is_sorted(ptr));

            // idx is a concatenated set of strictly sorted lists, whose endpoints are in ptr
            $assert!(idx.len() == val.len());
            $assert!(
                ptr.windows(2)
                    .all(|win| is_strictly_sorted(&idx[win[0]..win[1]]))
            );
            // ensure columns are in range
            $assert!(idx.len() == 0 || idx.iter().max().expect("len > 0") < &dim.1);
        }
        csr
    }};
}

impl<T> CsrMat<T> {
    /// Build a CSR matrix from its components, which are:
    ///  * `dim`: A tuple `(nrows, ncols)`
    ///  * `val`: A vector of explicit values stored in the matrix, ordered by row then column.
    ///  * `col`: The column of each value in `val`.
    ///  * `ptr`: A list of indices for the endpoints of each row. That is, row `i` consists of
    ///           the elements from `val` and `col` in the range `ptr[i]..ptr[i+1]`.  It must
    ///           contain `nrows+1` elements, starting with `0` and ending in `val.len()`.
    ///
    /// This validates all of the class invariants, with O(`val.len()`) total complexity.
    pub fn from_parts(dim: (usize, usize), val: Vec<T>, col: Vec<usize>, ptr: Vec<usize>) -> Self {
        validate_csr!(
            always,
            CsrMat {
                dim: dim,
                val: val,
                idx: col,
                ptr: ptr
            }
        )
    }

    /// `from_parts` with checking disabled in release builds.
    ///
    /// This is unsafe in order to give `CsrMat` the freedom to make unsafe optimizations based
    /// on its class invariants.
    #[cfg_attr(not(debug_assertions), inline)]
    pub unsafe fn from_parts_unchecked(
        dim: (usize, usize),
        val: Vec<T>,
        col: Vec<usize>,
        ptr: Vec<usize>,
    ) -> Self {
        validate_csr!(
            debug,
            CsrMat {
                dim: dim,
                val: val,
                idx: col,
                ptr: ptr
            }
        )
    }

    /// Build from an iterator of `((row,col), value)`.
    ///
    /// The iterator must satisfy the following:
    ///
    /// * Elements are sorted by row, then column.
    /// * No `(row,col)` position is specified twice.
    /// * `row < dim.0`, `col < dim.1`
    // FIXME: No direct tests (only tested indirectly through CooMat::to_csr)
    pub fn from_row_major_iter<I>(dim: (usize, usize), iter: I) -> Self
    where
        I: IntoIterator<Item = ((usize, usize), T)>,
    {
        validate_csr!(always, CsrMat::__from_row_major_iter(dim, iter))
    }

    /// Like `from_row_major_iter`, but assumes that all requirements are met.
    #[cfg_attr(not(debug_assertions), inline)]
    pub unsafe fn from_row_major_iter_unchecked<I>(dim: (usize, usize), iter: I) -> Self
    where
        I: IntoIterator<Item = ((usize, usize), T)>,
    {
        validate_csr!(debug, CsrMat::__from_row_major_iter(dim, iter))
    }

    fn __from_row_major_iter<I>(dim: (usize, usize), iter: I) -> Self
    where
        I: IntoIterator<Item = ((usize, usize), T)>,
    {
        let iter = iter.into_iter();
        let cap = iter.size_hint().0;
        let mut val = Vec::with_capacity(cap);
        let mut idx = Vec::with_capacity(cap);
        let mut ptr = Vec::with_capacity(dim.0 + 1);
        ptr.push(0); // beginning index of first row

        for ((row, col), x) in iter {
            // terminate the previous row and add any empty rows, if necessary
            while ptr.len() <= row {
                ptr.push(idx.len())
            }

            idx.push(col);
            val.push(x);
        }

        // terminate the last filled row and add empty rows to end
        while ptr.len() <= dim.0 {
            ptr.push(idx.len())
        }

        CsrMat {
            dim: dim,
            val: val,
            idx: idx,
            ptr: ptr,
        }
    }

    /// Construct from rows.
    ///
    /// Note that the `width` field must be provided explicitly rather than being inferred from
    /// the input rows; this is because it cannot be inferred in the case of 0 rows.
    pub fn from_rows<I>(width: usize, iter: I) -> Self
    where
        I: IntoIterator,
        I::Item: IntoSparseIterator<Value = T>,
        <I::Item as IntoSparseIterator>::IntoSparseIter:
            ShapedSparseIterator<Value = T> + UniqueSparseIterator + SortedSparseIterator,
    {
        let mut csr = CsrMat::zero((0, width));
        for row in iter.into_iter() {
            csr.push_row(row);
        }
        csr
    }

    /// Append a row.
    ///
    /// Panics if the row is of incorrect width.
    pub fn push_row<I>(&mut self, row: I)
    where
        I: IntoSparseIterator<Value = T>,
        I::IntoSparseIter:
            ShapedSparseIterator<Value = T> + UniqueSparseIterator + SortedSparseIterator,
    {
        let row = row.into_sparse_iter();
        assert_eq!(row.dim(), self.dim.1);

        // FIXME: This does not validate the invariants of CsrMat, so it is technically
        //  unsafe.  (or rather, it WOULD be if CsrMat actually bothered to use any unsafe
        //  optimizations, which is an option I want to keep open)

        // TODO: Hmmm, the whole point of having Sparse{Vec,Slice} keep individual members
        //  for `val` and `pos` was so that operations involving them could be vectorized,
        //  but it seems as though the SparseIterator abstraction gets in the way possibly.
        //  Probably no big deal, but think more about it/do benchmarks?
        // In any case, the zipped nature of the iterator precludes the use of `extend`.
        let (lo, _) = row.size_hint();
        self.val.reserve(lo);
        self.idx.reserve(lo);

        for (k, x) in row {
            self.val.push(x);
            self.idx.push(k);
        }
        self.ptr.push(self.val.len());
        self.dim.0 += 1;
    }

    /// Constructs a zero matrix
    #[inline]
    pub fn zero(dim: (usize, usize)) -> Self {
        CsrMat {
            dim: dim,
            val: vec![],
            idx: vec![],
            ptr: vec![0; dim.0 + 1],
        }
    }

    /// Constructs an identity matrix
    #[inline]
    pub fn eye(n: usize) -> Self
    where
        T: One + Clone,
    {
        CsrMat::from_diagonal(vec![One::one(); n])
    }

    /// Returns true if the two dimensions of the matrix are equal.
    ///
    /// Note that, under this definition, an e.g. `0 x n` matrix for `n != 0` is
    /// not considered square.
    #[inline]
    pub fn is_square(&self) -> bool {
        self.dim.0 == self.dim.1
    }

    /// Constructs a diagonal matrix
    #[inline]
    pub fn from_diagonal<I>(diag: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let diag = diag.into_iter().collect::<Vec<_>>();
        let n = diag.len();
        CsrMat {
            dim: (n, n),
            val: diag,
            idx: (0..n).collect(),
            ptr: (0..n + 1).collect(),
        }
    }

    /// Get the shape tuple `(nrows, ncols)`
    #[inline(always)]
    pub fn dim(&self) -> (usize, usize) {
        self.dim
    }

    /// Get the number of explicitly-stored elements.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.val.len()
    }

    /// Get the ratio of explicitly stored elements.
    ///
    /// Returns `None` when the dimension is zero.
    pub fn density(&self) -> Option<f64> {
        if self.dim.0 == 0 || self.dim.1 == 0 {
            None
        } else {
            let used = self.nnz() as f64;
            let all = self.dim.0 as f64 * self.dim.1 as f64;
            let ratio = (used / all).min(1f64); // in case of errors in mantissa precision
            Some(ratio)
        }
    }

    /// Iterate over the rows as `SparseSlice`s
    #[inline]
    pub fn rows(&self) -> Rows<T> {
        Rows {
            csr: &self,
            ptr_windows: self.ptr.windows(2),
        }
    }

    /// Convert into CSC. (O(`val.len()`) operation)
    // FIXME Clone bound not ideal
    #[inline]
    pub fn into_csc(self) -> CscMat<T>
    where
        T: Clone,
    {
        invert_cs_sparsity(self)
    }

    /// Return the matrix itself!
    #[inline(always)]
    pub fn into_csr(self) -> CsrMat<T> {
        self
    }

    /// Get the transpose, in CSC form.
    ///
    /// This is an O(1) operation which makes no transformations to the data.
    /// If you want the transpose in CSR format, you will need to explicitly convert
    ///  it afterwards: `mat.transpose().into_csr()`
    #[inline]
    pub fn transpose(self) -> CscMat<T> {
        CscMat { T: self }
    }
}

impl<T> CscMat<T> {
    /// Build a CSC matrix from its components, which are:
    ///
    ///  * `dim`: A tuple `(nrows, ncols)`
    ///  * `val`: A vector of explicit values stored in the matrix, ordered by column then row.
    ///  * `row`: The row of each value in `val`.
    ///  * `ptr`: A list of indices for the endpoints of each column. That is, column `i` consists
    ///           of the elements from `val` and `row` in the range `ptr[i]..ptr[i+1]`.  It must
    ///           contain `ncols+1` elements, starting with `0` and ending in `val.len()`.
    ///
    /// This validates all of the class invariants, with O(`val.len()`) total complexity.
    #[inline(always)]
    pub fn from_parts(dim: (usize, usize), val: Vec<T>, row: Vec<usize>, ptr: Vec<usize>) -> Self {
        CscMat {
            T: CsrMat::from_parts((dim.1, dim.0), val, row, ptr),
        }
    }

    /// `from_parts` with checking disabled in release builds.
    ///
    /// This is unsafe in order to give `CscMat` the freedom to make unsafe optimizations based
    /// on its class invariants.
    #[inline(always)]
    pub unsafe fn from_parts_unchecked(
        dim: (usize, usize),
        val: Vec<T>,
        row: Vec<usize>,
        ptr: Vec<usize>,
    ) -> Self {
        CscMat {
            T: CsrMat::from_parts_unchecked((dim.1, dim.0), val, row, ptr),
        }
    }

    /// Build from an iterator of `((row,col), value)`.
    ///
    /// The iterator must satisfy the following:
    ///
    /// * Elements are sorted by column, then row.
    /// * No `(row,col)` position is specified twice.
    /// * `row < dim.0`, `col < dim.1`
    // FIXME: No direct tests (only tested indirectly through CooMat::to_csc)
    pub fn from_column_major_iter<I>(dim: (usize, usize), iter: I) -> Self
    where
        I: IntoIterator<Item = ((usize, usize), T)>,
    {
        // transpose inputs
        let dim = (dim.1, dim.0);
        let iter = iter.into_iter().map(|((r, c), x)| ((c, r), x));
        CscMat {
            T: CsrMat::from_row_major_iter(dim, iter),
        }
    }

    /// Like `from_column_major_iter`, but assumes that all requirements are met.
    // FIXME: No direct tests (only tested indirectly through CooMat::to_csc)
    pub unsafe fn from_column_major_iter_unchecked<I>(dim: (usize, usize), iter: I) -> Self
    where
        I: IntoIterator<Item = ((usize, usize), T)>,
    {
        // transpose inputs
        let dim = (dim.1, dim.0);
        let iter = iter.into_iter().map(|((r, c), x)| ((c, r), x));
        CscMat {
            T: CsrMat::from_row_major_iter_unchecked(dim, iter),
        }
    }

    /// Construct from columns.
    ///
    /// Note that the `height` field must be provided explicitly rather than being inferred from
    /// the input columns; this is because it cannot be inferred in the case of 0 columns.
    #[inline(always)]
    pub fn from_columns<I>(height: usize, iter: I) -> Self
    where
        I: IntoIterator,
        I::Item: IntoSparseIterator<Value = T>,
        <I::Item as IntoSparseIterator>::IntoSparseIter:
            ShapedSparseIterator<Value = T> + UniqueSparseIterator + SortedSparseIterator,
    {
        CscMat {
            T: CsrMat::from_rows(height, iter),
        }
    }

    /// Constructs a zero matrix
    #[inline]
    pub fn zero(dim: (usize, usize)) -> Self {
        CscMat {
            T: CsrMat::zero((dim.1, dim.0)),
        }
    }

    /// Constructs an identity matrix
    #[inline(always)]
    pub fn eye(n: usize) -> Self
    where
        T: One + Clone,
    {
        CscMat { T: CsrMat::eye(n) }
    }

    /// Constructs a diagonal matrix
    #[inline(always)]
    pub fn from_diagonal<I>(diag: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        CscMat {
            T: CsrMat::from_diagonal(diag),
        }
    }

    /// Returns true if the two dimensions of the matrix are equal.
    ///
    /// Note that, under this definition, an e.g. `0 x n` matrix for `n != 0` is
    /// not considered square.
    #[inline(always)]
    pub fn is_square(&self) -> bool {
        self.T.is_square()
    }

    /// Get the shape tuple `(nrows, ncols)`
    #[inline(always)]
    pub fn dim(&self) -> (usize, usize) {
        (self.T.dim.1, self.T.dim.0)
    }

    /// Get the number of explicitly-stored elements.
    #[inline(always)]
    pub fn nnz(&self) -> usize {
        self.T.nnz()
    }

    /// Get the ratio of explicitly stored elements.
    ///
    /// Returns `None` when the dimension is zero.
    #[inline(always)]
    pub fn density(&self) -> Option<f64> {
        self.T.density()
    }

    /// Iterate over the columns as `SparseSlice`s.
    ///
    /// Don't mind the name of the return type. (cough)
    #[inline(always)]
    pub fn columns(&self) -> Rows<T> {
        self.T.rows()
    }

    /// Convert into CSR. (O(`val.len()`) operation)
    // FIXME Clone bound not ideal
    #[inline]
    pub fn into_csr(self) -> CsrMat<T>
    where
        T: Clone,
    {
        invert_cs_sparsity(self.T).T
    }

    /// Return the matrix itself!
    #[inline(always)]
    pub fn into_csc(self) -> CscMat<T> {
        self
    }

    /// Get the transpose, in CSR format.
    ///
    /// This is an O(1) operation which makes no transformations to the data.
    /// If you want the transpose in CSC format, you will need to explicitly convert
    ///  it afterwards: `mat.transpose().into_csc()`
    #[inline(always)]
    pub fn transpose(self) -> CsrMat<T> {
        self.T
    }
}

//-------------------------------------------------

fn invert_cs_sparsity<T: Clone>(csr: CsrMat<T>) -> CscMat<T> {
    // It should be possible to do this with far fewer memory allocations, if necessary;
    // instead of building a table of nested vectors, you could make two passes through `idx`, one
    // to count the # of occurrences of each column (and construct ptr), and the second to write
    // the values (keeping track of offsets for the next element in each column).

    // reverse lookup table for elements by column
    let mut column_table: Vec<Vec<(usize, T)>> =
        ::std::iter::repeat(vec![]).take(csr.dim.1).collect();
    for (i, row) in csr.rows().enumerate() {
        for (k, val) in row.sparse_iter().sparse_cloned() {
            column_table[k].push((i, val));
        }
    }

    // CSC components
    let mut val = Vec::with_capacity(csr.val.len());
    let mut row = Vec::with_capacity(csr.val.len());
    let mut ptr = Vec::with_capacity(csr.dim.1 + 1);
    ptr.push(0);
    for column in column_table {
        let next = ptr.last().unwrap() + column.len();
        ptr.push(next);
        for (i, x) in column {
            row.push(i);
            val.push(x);
        }
    }

    unsafe { CscMat::from_parts_unchecked(csr.dim, val, row, ptr) }
}

//-------------------------------------------------

#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
#[derive(Clone)]
pub struct Rows<'a, T: 'a> {
    csr: &'a CsrMat<T>,
    ptr_windows: ::std::slice::Windows<'a, usize>,
}

impl<'a, T> Iterator for Rows<'a, T> {
    type Item = SparseSlice<'a, T>;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.ptr_windows.next().map(|win|
		// COPY-PASTA ALERT vvvv
		unsafe {
			RawSparseSlice {
				dim: self.csr.dim.1,
				val: &self.csr.val[win[0]..win[1]],
				src_pos: &self.csr.idx[win[0]..win[1]],
				offset: 0, // the values in idx ARE the column indices; no offset necessary
			}.to_sparse_slice_unchecked()
		})
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.ptr_windows.size_hint()
    }
}

impl<'a, T> DoubleEndedIterator for Rows<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.ptr_windows.next_back().map(|win|
		// COPY-PASTA ALERT ^^^^
		unsafe {
			RawSparseSlice {
				dim: self.csr.dim.1,
				val: &self.csr.val[win[0]..win[1]],
				src_pos: &self.csr.idx[win[0]..win[1]],
				offset: 0, // the values in idx ARE the column indices; no offset necessary
			}.to_sparse_slice_unchecked()
		})
    }
}

impl<'a, T> ExactSizeIterator for Rows<'a, T> {}

//-------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use test::black_box;

    #[test]
    fn test_from_parts_success() {
        // ALWAYS. TEST. SIZE. ZERO.
        // (sorry, still sour over `slice()`)
        black_box(CsrMat::<i32>::from_parts((0, 0), vec![], vec![], vec![0]));
        black_box(CscMat::<i32>::from_parts((0, 0), vec![], vec![], vec![0]));

        // Zero in the primary dimension.
        black_box(CsrMat::<i32>::from_parts((0, 5), vec![], vec![], vec![0]));
        black_box(CscMat::<i32>::from_parts((5, 0), vec![], vec![], vec![0]));

        // Zero in the secondary dimension
        black_box(CsrMat::<i32>::from_parts(
            (5, 0),
            vec![],
            vec![],
            vec![0; 6],
        ));
        black_box(CscMat::<i32>::from_parts(
            (0, 5),
            vec![],
            vec![],
            vec![0; 6],
        ));

        // An identity matrix
        black_box(CsrMat::from_parts(
            (3, 3),
            vec![1f32; 3],
            vec![0, 1, 2],
            vec![0, 1, 2, 3],
        ));
        black_box(CscMat::from_parts(
            (3, 3),
            vec![1f32; 3],
            vec![0, 1, 2],
            vec![0, 1, 2, 3],
        ));

        // A flipped identity matrix (a case where col isn't sorted)
        black_box(CsrMat::from_parts(
            (3, 3),
            vec![1f32; 3],
            vec![2, 1, 0],
            vec![0, 1, 2, 3],
        ));
        black_box(CscMat::from_parts(
            (3, 3),
            vec![1f32; 3],
            vec![2, 1, 0],
            vec![0, 1, 2, 3],
        ));

        // A filled 2x2 matrix (for kicks)
        black_box(CsrMat::from_parts(
            (2, 2),
            vec![1f32; 4],
            vec![0, 1, 0, 1],
            vec![0, 2, 4],
        ));
        black_box(CscMat::from_parts(
            (2, 2),
            vec![1f32; 4],
            vec![0, 1, 0, 1],
            vec![0, 2, 4],
        ));
    }

    // Tests for invalid input.
    // Not going to try to make these tests comprehensive because I'd be here all day, but let's
    // try what we can, alternating between Csc and Csr
    #[test]
    #[should_panic(expected = "assertion failed")]
    fn test_from_parts_fail_1() {
        // ptr empty
        black_box(CsrMat::<i32>::from_parts((0, 0), vec![], vec![], vec![]));
    }
    #[test]
    #[should_panic(expected = "assertion failed")]
    fn test_from_parts_fail_2() {
        // ptr not long enough
        black_box(CscMat::<i32>::from_parts(
            (0, 5),
            vec![],
            vec![],
            vec![0; 5],
        ));
    }
    #[test]
    #[should_panic(expected = "assertion failed")]
    fn test_from_parts_fail_3() {
        // wrong number of columns
        black_box(CsrMat::from_parts(
            (1, 7),
            vec![1i32, 2],
            vec![1, 2, 3],
            vec![0, 2],
        ));
    }
    #[test]
    #[should_panic(expected = "assertion failed")]
    fn test_from_parts_fail_4() {
        // index in secondary dimension too large
        black_box(CscMat::from_parts((4, 1), vec![1u16], vec![9], vec![0, 1]));
    }
    #[test]
    #[should_panic(expected = "assertion failed")]
    fn test_from_parts_fail_5() {
        // a row with a repeated index
        black_box(CsrMat::from_parts(
            (1, 5),
            vec![1f32, 2.],
            vec![2, 2],
            vec![0, 2],
        ));
    }

    #[test]
    fn test_eye() {
        assert_eq!(
            CsrMat::from_parts((3, 3), vec![1f32; 3], vec![0, 1, 2], vec![0, 1, 2, 3]),
            CsrMat::eye(3)
        );
    }

    #[test]
    fn test_dim() {
        assert_eq!(
            CsrMat::from_parts((3, 5), vec![1f32], vec![2], vec![0, 0, 1, 1]).dim(),
            (3, 5)
        );
        assert_eq!(
            CscMat::from_parts((3, 5), vec![1f32], vec![2], vec![0, 0, 1, 1, 1, 1]).dim(),
            (3, 5)
        );
    }

    #[test]
    fn test_nnz() {
        assert_eq!(CsrMat::<f32>::zero((0, 7)).nnz(), 0);
        assert_eq!(CscMat::<f32>::zero((0, 7)).nnz(), 0);
        assert_eq!(
            CsrMat::from_parts((3, 5), vec![1f32], vec![2], vec![0, 0, 1, 1]).nnz(),
            1
        );
        assert_eq!(
            CscMat::from_parts((3, 5), vec![1f32], vec![2], vec![0, 0, 1, 1, 1, 1]).nnz(),
            1
        );
    }

    #[test]
    fn test_density() {
        assert_eq!(CsrMat::<f64>::eye(0).density(), None);
        assert_eq!(CscMat::<f64>::eye(0).density(), None);
        assert_close!(rel=1e-12, CsrMat::<u16>::eye(1).density().unwrap(), 1.0);
        assert_close!(rel=1e-12, CscMat::<u16>::eye(1).density().unwrap(), 1.0);
        assert_close!(rel=1e-12, CsrMat::<i32>::eye(3).density().unwrap(), 1./3.);
        assert_close!(rel=1e-12, CscMat::<i32>::eye(3).density().unwrap(), 1./3.);
        assert_close!(rel=1e-12, CsrMat::<f32>::zero((3, 3)).density().unwrap(), 0.0);
        assert_close!(rel=1e-12, CscMat::<f32>::zero((3, 3)).density().unwrap(), 0.0);
    }

    #[test]
    fn test_rows() {
        // Size 0
        let mat = CsrMat::<f64>::eye(0);
        let mut rows = mat.rows().map(|x| x.to_dense());
        assert_eq!(rows.next(), None);

        // Diagonal
        let mat = CsrMat::from_diagonal(vec![7, 8, 9]);
        let mut rows = mat.rows().map(|x| x.to_dense());
        assert_eq!(rows.next(), Some(vec![7, 0, 0]));
        assert_eq!(rows.next(), Some(vec![0, 8, 0]));
        assert_eq!(rows.next(), Some(vec![0, 0, 9]));
        assert_eq!(rows.next(), None);

        // Something that isn't symmetric; an L shape of 1s
        let mat = CsrMat::from_parts((2, 2), vec![1.; 3], vec![0, 0, 1], vec![0, 1, 3]);
        let mut rows = mat.rows().map(|x| x.to_dense());
        assert_eq!(rows.next(), Some(vec![1., 0.]));
        assert_eq!(rows.next(), Some(vec![1., 1.]));
        assert_eq!(rows.next(), None);

        let tmat = mat.clone().transpose();
        let mut tcols = tmat.columns().map(|x| x.to_dense());
        assert_eq!(tcols.next(), Some(vec![1., 0.]));
        assert_eq!(tcols.next(), Some(vec![1., 1.]));
        assert_eq!(tcols.next(), None);
    }

    #[test]
    fn test_sparsity_change() {
        // Thank goodness for that all-caps note above which keeps reminding me to test size zero.
        let mat1 = CscMat::<i32>::eye(0);
        let mat2 = mat1.clone().transpose().into_csc();
        assert_eq!(mat1, mat2);

        // A sparsity change should leave diagonal matrices unaffected
        let mat1 = CscMat::from_diagonal(vec![1, 2, 4, 8, 16]);
        let mat2 = mat1.clone().transpose().into_csc();
        assert_eq!(mat1, mat2);

        // Off-diagonal elements; change sparsity to compute transpose of an L shape
        let mat = CsrMat::from_parts((2, 2), vec![1.; 3], vec![0, 0, 1], vec![0, 1, 3]);
        let mat = mat.transpose().into_csr();
        let mut rows = mat.rows().map(|x| x.to_dense());
        assert_eq!(rows.next(), Some(vec![1., 1.]));
        assert_eq!(rows.next(), Some(vec![0., 1.]));
        assert_eq!(rows.next(), None);

        // Something non-square; let's change between row and column vectors
        let row = CsrMat::from_parts((1, 3), vec![7., 8., 9.], vec![0, 1, 2], vec![0, 3]);
        let col = CsrMat::from_parts((3, 1), vec![7., 8., 9.], vec![0, 0, 0], vec![0, 1, 2, 3]);
        assert_eq!(row, col.clone().transpose().into_csr());
        assert_eq!(col, row.clone().transpose().into_csr());
    }
}

//-------------------------------------------------

#[inline]
fn is_sorted<T: PartialOrd>(xs: &[T]) -> bool {
    xs.windows(2).all(|win| win[0] <= win[1])
}

#[inline]
fn is_strictly_sorted<T: PartialOrd>(xs: &[T]) -> bool {
    xs.windows(2).all(|win| win[0] < win[1])
}
