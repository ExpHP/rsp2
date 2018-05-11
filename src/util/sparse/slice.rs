use num_traits::Zero;

use iter::{ShapedSparseIterator, SortedSparseIterator, SparseIterator, UniqueSparseIterator};
use traits::DenseIndex;
use traits::GenericRange;
use traits::Shape;
use vec::SparseVec;

/// A view into a SparseVec.
///
/// `SparseSlice<T>` is to `SparseVec<T>` as `&[T]` is to `Vec<T>`.
/// Slices are taken with respect to the dense item position, rather than sparse item index.
#[derive(Copy, Clone, Debug)]
pub struct SparseSlice<'a, T: 'a> {
    dim: usize,
    val: &'a [T],

    // Positions in the slice are generally less than positions in the source, but we want to
    // be `Copy`, so we cannot own our own Vec of positions.  Instead, we store an offset.
    // The relation is:  `pos_in_slice[i] = src_pos[i] - offset`
    src_pos: &'a [usize],
    offset: usize,
}

// Because I'll lose my mind if I have to move `SparseSlice` to the `vec` module just so
// I can construct one.
/// A struct with the layout of `SparseSlice`.
pub struct RawSparseSlice<'a, T: 'a> {
    pub dim: usize,
    pub val: &'a [T],
    pub src_pos: &'a [usize],
    pub offset: usize,
}

// FIXME inconsistency with other API parts, which just have a `from_parts_unchecked`
impl<'a, T> RawSparseSlice<'a, T> {
    /// Calling code must guarantee:
    ///  * `val` and `src_pos` are the same length.
    ///  * `src_pos` is sorted, without duplicates.
    ///  * `0 <= (x - offset) < dim` for all `x` in `src_pos`
    ///
    /// This is unsafe to give `SparseSlice` the freedom to make unsafe optimizations based on
    ///  its invariants.
    #[inline]
    pub unsafe fn to_sparse_slice_unchecked(self) -> SparseSlice<'a, T> {
        debug_assert!(self.val.len() == self.src_pos.len());
        debug_assert!(self.src_pos.windows(2).all(|w| w[0] < w[1]));

        // NOTE TO SELF: I cannot be trusted with `unwrap_or`
        debug_assert!(self.src_pos.len() == 0 || *self.src_pos.first().unwrap() >= self.offset);
        debug_assert!(
            self.src_pos.len() == 0 || *self.src_pos.last().unwrap() < self.dim + self.offset
        );
        ::std::mem::transmute(self)
    }
}

// FIXME lots of code duplication with SparseVec with only subtle changes (generally speaking,
//  just the lifetime parameter, and adding offsets to positions); this is asking for trouble

impl<'a, T> SparseSlice<'a, T> {
    /// Densify into a standard vector.
    ///
    /// Each item will be placed into its appropriate position, with the omitted elements
    /// filled with zeros.
    ///
    /// This is moderately more efficient than `self.sparse_iter().cloned().densify()`.
    // NOTE: That said, this function serves little use outside of debugging,
    //   so whether said optimization is actually justified remains to be seen...
    pub fn to_dense(&self) -> Vec<T>
    where
        T: Zero + Clone,
    {
        let mut vec = vec![Zero::zero(); self.dim];
        for (x, i) in zip!(self.val, self.src_pos) {
            vec[*i - self.offset] = x.clone();
        }
        vec
    }

    /// Iterate over the explicitly stored values and positions.
    ///
    /// The returned iterator has item type `(usize, &T)`.
    pub fn sparse_iter(&self) -> SparseIter<'a, T> {
        SparseIter {
            dim: self.dim,
            it: zip!(self.src_pos.iter().cloned(), self.val.iter()),
            offset: self.offset,
        }
    }

    /// Clone the contents of this slice into a new `SparseVec`.
    pub fn to_sparse_vec(&self) -> SparseVec<T>
    where
        T: Clone,
    {
        SparseVec::from_sparse_iter(self.sparse_iter().sparse_cloned())
    }

    /// The number of explicit elements contained.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.val.len()
    }

    // NOTE: `as_positions` is not possible because the positions in src_pos are offset.
    // To avoid asymmetry in the API, `as_values` is omitted as well.
}

impl<'a, T> Shape for SparseSlice<'a, T> {
    #[inline(always)]
    fn dim(&self) -> usize {
        self.dim
    }
}

// Must manually implement this so that it accounts for offsets
impl<'a, T: PartialEq<T>> PartialEq for SparseSlice<'a, T> {
    fn eq(&self, other: &SparseSlice<T>) -> bool {
        if self.dim() != other.dim() {
            return false;
        }
        if self.nnz() != other.nnz() {
            return false;
        }
        self.sparse_iter().eq(other.sparse_iter())
    }
}

impl<'a, T: Eq> Eq for SparseSlice<'a, T> {}

//----------------------------------------------------------

/// Implementation detail of `SparseSlice::sparse_iter`
pub struct SparseIter<'a, T: 'a> {
    dim: usize,
    it: ::std::iter::Zip<
        ::std::iter::Cloned<::std::slice::Iter<'a, usize>>,
        ::std::slice::Iter<'a, T>,
    >,
    offset: usize,
}

impl<'a, T> Iterator for SparseIter<'a, T> {
    type Item = (usize, &'a T);
    fn next(&mut self) -> Option<Self::Item> {
        self.it.next().map(|(i, x)| (i - self.offset, x))
    }
}

impl<'a, T> SparseIterator for SparseIter<'a, T> {
    type Value = &'a T;
}

// Trait checklist:
// [O] ExactSizeIterator
// [O] DoubleEndedIterator
// [O] ShapedSparseIterator
// [O] UniqueSparseIterator
// [O] SortedSparseIterator

impl<'a, T> ExactSizeIterator for SparseIter<'a, T> {
    fn len(&self) -> usize {
        self.it.len()
    }
}

impl<'a, T> DoubleEndedIterator for SparseIter<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.it.next_back().map(|(i, x)| (i - self.offset, x))
    }
}

impl<'a, T> Shape for SparseIter<'a, T> {
    fn dim(&self) -> usize {
        self.dim
    }
}

impl<'a, T> ShapedSparseIterator for SparseIter<'a, T> {}
impl<'a, T> UniqueSparseIterator for SparseIter<'a, T> {}
impl<'a, T> SortedSparseIterator for SparseIter<'a, T> {}

//----------------------------------------------------------
// Indexing

impl<'a, T> SparseSlice<'a, T> {
    // Same return value as `binary_search` for slices. This exists so that we can try switching out
    // to linear searches, or perhaps switching between methods based on `nnz`
    #[inline]
    fn find_idx(&self, pos: usize) -> Result<usize, usize> {
        // debug_assert because callers should already be doing their own checks
        debug_assert!(pos <= self.dim());
        self.src_pos.binary_search(&(pos + self.offset))
    }

    // merges the Ok/Err cases of find_idx
    #[inline]
    fn find_insertion_idx(&self, pos: usize) -> usize {
        match self.find_idx(pos) {
            Ok(x) | Err(x) => x,
        }
    }

    /// Index by dense position.
    ///
    /// Returns `#Some(&T)` if the position contains an explicit value (including explicit
    /// zeros), else `None`.
    /// Note that, due to the sparse format, there is an inherent cost in searching for the item.
    ///
    /// # Panics
    ///
    /// Panics if `pos >= self.dim()`.
    // TODO code example
    pub fn get_explicit(&self, pos: usize) -> Option<&T> {
        self.check_index(pos);
        match self.find_idx(pos) {
            Ok(idx) => Some(&self.val[idx]),
            Err(_) => None,
        }
    }

    /// Obtain a view into a region of the `SparseSlice`.
    ///
    /// The reason this is not provided via the `Index` trait is because the returned object
    /// cannot be represented as a borrow.
    ///
    /// # Panics
    ///
    /// Matches the panic semantics of `Vec as Index<Range<usize>>`.
    pub fn slice<R>(&'a self, pos: R) -> SparseSlice<'a, T>
    where
        R: GenericRange<usize>,
    {
        let pos = pos.with_defaults(0, self.dim());
        assert!(
            pos.start <= pos.end,
            "slice position starts at {} but ends at {}",
            pos.start,
            pos.end
        );
        assert!(
            pos.end <= self.dim(),
            "position {} out of range for sparse slice of dim {}",
            pos.end,
            self.dim()
        );
        let idx = self.find_insertion_idx(pos.start)..self.find_insertion_idx(pos.end);
        SparseSlice {
            dim: pos.len(),
            val: &self.val[idx.clone()],
            src_pos: &self.src_pos[idx],
            offset: self.offset + pos.start,
        }
    }

    #[inline]
    fn check_index(&self, pos: usize) {
        assert!(
            pos < self.dim(),
            "index out of bounds: the dim is {} but the position is {}",
            self.dim(),
            pos
        );
    }
}

impl<'a, T> DenseIndex<usize> for SparseSlice<'a, T>
where
    T: Clone + Zero,
{
    type Value = T;
    #[inline]
    fn get_dense(&self, pos: usize) -> T {
        match self.get_explicit(pos) {
            None => Zero::zero(),
            Some(borrow) => borrow.clone(),
        }
    }
}

//----------------------------------------------------------

#[cfg(test)]
mod tests {
    use traits::DenseIndex;
    use traits::Shape;
    use vec::SparseVec;

    #[test]
    fn test_slice() {
        let dense = vec![0i32, 2, 0, 0, 0, 0, 0, 4, 0, 0, 12, 0, 0, 0];
        let sparse = SparseVec::from_dense(dense);

        // slice with zero offset
        let slice = sparse.slice(0..6);
        let slc_dense = vec![0, 2, 0, 0, 0, 0];
        assert_eq!(slice.to_dense(), slc_dense);
        assert_eq!(slice.to_sparse_vec(), SparseVec::from_dense(slc_dense));
        assert_eq!(slice.dim(), 6);
        assert_eq!(slice.nnz(), 1);

        // zero length slice
        let slice = sparse.slice(3..3);
        let slc_dense = vec![];
        assert_eq!(slice.to_dense(), slc_dense);
        assert_eq!(slice.to_sparse_vec(), SparseVec::from_dense(slc_dense));
        assert_eq!(slice.dim(), 0);
        assert_eq!(slice.nnz(), 0);

        // full slice
        let slice = sparse.slice(0..sparse.dim());
        assert_eq!(slice.dim(), sparse.dim());
        assert_eq!(slice.nnz(), sparse.nnz());

        // slice with nonzero offset
        let slice = sparse.slice(3..9);
        let slc_dense = vec![0, 0, 0, 0, 4, 0];
        assert_eq!(slice.to_dense(), slc_dense);
        assert_eq!(slice.to_sparse_vec(), SparseVec::from_dense(slc_dense));
        assert_eq!(slice.dim(), 6);
        assert_eq!(slice.nnz(), 1);

        // slice of slice, with nonzero offsets
        let slice = slice.slice(3..6);
        let slc_dense = vec![0, 4, 0];
        assert_eq!(slice.to_dense(), slc_dense);
        assert_eq!(slice.to_sparse_vec(), SparseVec::from_dense(slc_dense));
        assert_eq!(slice.dim(), 3);
        assert_eq!(slice.nnz(), 1);
    }

    #[test]
    fn test_dense_index_ops() {
        // put an implicit zero, an explicit nonzero, and an explicit zero
        //  in the first three elements starting from index 7
        let sparse = SparseVec::from_parts_strictly_sorted(16, vec![2i32, 0], vec![8, 9]);
        let sparse = sparse.slice(7..10);
        assert_eq!(sparse.nnz(), 2);
        assert_eq!(sparse.get_dense(0), 0);
        assert_eq!(sparse.get_dense(1), 2);
        assert_eq!(sparse.get_dense(2), 0);

        assert_eq!(sparse.get_explicit(0), None);
        assert_eq!(sparse.get_explicit(1), Some(&2));
        assert_eq!(sparse.get_explicit(2), Some(&0));
    }

    // For testing PartialEq implementations.
    // Checks both eq and ne, to make sure they are opposites.
    macro_rules! check_partial_eq {
        ($expect:expr, $a:expr, $b:expr) => {
            assert_eq!($a == $b, $expect);
            assert_eq!($a != $b, !$expect);
        };
    }

    #[test]
    fn test_slice_partial_eq() {
        // zero length
        let a = SparseVec::<i32>::from_dense(vec![]);
        check_partial_eq!(true, a.slice(..), a.slice(..));

        let a = SparseVec::from_dense(vec![0i32; 20]);
        check_partial_eq!(false, a.slice(..5), a.slice(..6)); // shape matters
        check_partial_eq!(true, a.slice(..10), a.slice(10..)); // offset does not

        // explicit zeros matter
        let a = SparseVec::from_dense(vec![2u32, 3, 6, 0, 0, 2, 4]);
        let mut b = a.clone();
        assert_eq!(a.slice(..).to_dense(), b.slice(..).to_dense());
        check_partial_eq!(true, a.slice(..), b.slice(..));
        b.set_explicit(3, 0);
        assert_eq!(a.slice(..).to_dense(), b.slice(..).to_dense()); // dense form is unchanged...
        check_partial_eq!(false, a.slice(..), b.slice(..)); // ...but slices are no longer equal

        // compare effective positions, not those stored.
        // i.e. the following slices look different internally due to the position offset,
        //  but both are equal to `SparseVec::from_dense(vec![0,1,2,3]).slice(..)`
        let a = SparseVec::from_dense(vec![4i16, 0, 1, 2, 3, 8]);
        let b = SparseVec::from_dense(vec![4i16, 0, 0, 1, 2, 3, 9]);
        check_partial_eq!(true, a.slice(1..5), b.slice(2..6));
    }

    #[test]
    fn test_slice_polymorphism() {
        let dense = vec![0i32, 2, 0, 0, 0, 0, 0, 4, 0, 0, 12, 0, 0, 0];
        let sparse = SparseVec::from_dense(dense.clone());

        // test on vec
        assert_eq!(sparse.slice(..), sparse.slice(0..14));
        assert_eq!(sparse.slice(..8), sparse.slice(0..8));
        assert_eq!(sparse.slice(7..), sparse.slice(7..14));

        // test on slice
        let sparse = sparse.slice(1..13);
        assert_eq!(sparse.slice(..), sparse.slice(0..12));
        assert_eq!(sparse.slice(..8), sparse.slice(0..8));
        assert_eq!(sparse.slice(7..), sparse.slice(7..12));
    }
}
