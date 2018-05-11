



use num_traits::Zero;

#[cfg(test)]
use test::Bencher;

use iter::IntoSparseIterator;
use iter::SparseIterator;
use iter::{ShapedSparseIterator, SortedSparseIterator, UniqueSparseIterator};
use slice::RawSparseSlice;
use slice::SparseSlice;
use traits::Abs;
use traits::DenseIndex;
use traits::GenericRange;
use traits::Shape;

/// A `Vec` stored in a sparse format (without zeros).
#[derive(Clone, Debug)]
pub struct SparseVec<T> {
    dim: usize,
    val: Vec<T>,
    pos: Vec<usize>,
}

impl<T> PartialEq for SparseVec<T>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.dim == other.dim && self.val == other.val && self.pos == other.pos
    }
}

impl<T> SparseVec<T> {
    /// Construct a `SparseVec` from its components.
    ///
    /// The arguments are as follows:
    ///
    ///  * `dim`: The length of the dense Vec which the `SparseVec` represents.
    ///  * `val`: The explicitly stored values, in order of position.
    ///  * `pos`: A vector parallel to `val` containing the positions of each value.
    ///
    /// # Panics
    ///
    /// Positions must be unique, in order, and strictly less than `dim`, or the function
    ///  will panic. Also panics if the length of `pos` and `val` do not match.`
    pub fn from_parts(dim: usize, val: Vec<T>, pos: Vec<usize>) -> Self {
        assert_eq!(pos.len(), val.len());
        assert!(
            pos.windows(2).all(|win| win[0] < win[1]),
            "Input must be sorted, without duplicates"
        );
        assert!(
            pos.last().and_then(|x| Some(x < &dim)).unwrap_or(true),
            "element index exceeds dimension"
        );
        unsafe { SparseVec::from_parts_unchecked(dim, val, pos) }
    }

    /// Construct a `SparseVec` from its components.
    ///
    /// The arguments are the same as `from_parts`, but this function will not check that the
    ///  invariants of SparseVec are upheld.
    ///
    /// This is unsafe to give `SparseVec` the freedom to make unsafe optimizations based on
    ///  its invariants.
    #[inline]
    pub unsafe fn from_parts_unchecked(dim: usize, val: Vec<T>, pos: Vec<usize>) -> Self {
        SparseVec {
            dim: dim,
            val: val,
            pos: pos,
        }
    }

    /// Iterate over the explicitly stored values and positions.
    ///
    /// The returned iterator has item type `(usize, &T)`.
    #[inline]
    pub fn sparse_iter<'a>(&'a self) -> SparseIter<'a, T> {
        SparseIter {
            dim: self.dim,
            it: zip!(self.pos.iter().cloned(), self.val.iter()),
        }
    }

    /// Construct from a `SparseIter`
    pub fn from_sparse_iter<I>(it: I) -> SparseVec<T>
    where
        I: ShapedSparseIterator<Value = T> + SortedSparseIterator + UniqueSparseIterator,
    {
        let cap = it.size_hint().0;
        let mut pos = Vec::with_capacity(cap);
        let mut val = Vec::with_capacity(cap);
        let dim = it.dim();
        for (i, x) in it {
            debug_assert!(i < dim, "ShapedSparseVec guarantee violated");
            debug_assert!(
                pos.last().map(|k| k < &i).unwrap_or(true),
                "SortedSparseVec or UniqueSparseVec guarantee violated"
            );
            pos.push(i);
            val.push(x);
        }
        SparseVec {
            dim: dim,
            pos: pos,
            val: val,
        }
    }

    /// Construct from a dense vector, dropping any zero values.
    pub fn from_dense(vec: Vec<T>) -> Self
    where
        T: Zero + PartialEq,
    {
        let dim = vec.len();
        let zero = Zero::zero();
        let mut val = vec![];
        let mut pos = vec![];
        for (i, x) in vec.into_iter().enumerate().filter(|&(_, ref x)| x != &zero) {
            val.push(x);
            pos.push(i);
        }
        SparseVec {
            dim: dim,
            val: val,
            pos: pos,
        }
    }

    /// Densify into a standard vector.
    ///
    /// Each item will be placed into its appropriate position, with the omitted elements
    /// filled with zeros.
    ///
    /// This is moderately more efficient than `self.into_sparse_iter().densify()`.
    // NOTE: That said, this function serves little use outside of debugging,
    //   so whether said optimization is actually justified remains to be seen...
    pub fn into_dense(self) -> Vec<T>
    where
        T: Zero + Clone,
    {
        let mut vec = vec![Zero::zero(); self.dim];
        for (x, i) in zip!(self.val, self.pos) {
            vec[i] = x;
        }
        vec
    }

    /// Create a zero vector of a specified dimension.
    #[inline]
    pub fn zero(dim: usize) -> SparseVec<T> {
        SparseVec {
            dim: dim,
            val: vec![],
            pos: vec![],
        }
    }

    // FIXME this should be on SparseIterator instead
    /// Drop explicitly-stored zeros and values close to zero.
    pub fn prune(self, zero_tol: &T) -> Self
    where
        T: Clone + PartialOrd + Abs,
    {
        let SparseVec { dim, pos, val } = self;
        let mut newval = Vec::with_capacity(pos.len());
        let mut newpos = Vec::with_capacity(pos.len());

        let mut skip_count = 0;
        for (i, x) in zip!(pos, val) {
            if &x.clone().abs() >= zero_tol {
                newval.push(x);
                newpos.push(i - skip_count);
            } else {
                skip_count += 1;
            }
        }
        SparseVec {
            dim: dim,
            val: newval,
            pos: newpos,
        }
    }

    // TODO reconsider; was hastily added
    pub fn last_dense(&self) -> Option<T>
    where
        T: Zero + Clone,
    {
        if self.dim == 0 {
            return None;
        };

        match self.pos.last() {
            Some(i) if i + 1 == self.dim => Some(self.val.last().unwrap().clone()),
            _ => Some(Zero::zero()),
        }
    }

    /// Change the (dense) length of the vector.
    ///
    /// If the new dimension is shorter than the old dimension, elements beyond the
    /// new end are simply dropped.  If the new dimension is longer, the extra space
    /// is filled with implicit zeros.
    pub fn reshape(&mut self, dim: usize) {
        if self.dim > dim {
            let idx = self.find_insertion_idx(dim);
            self.val.truncate(idx);
            self.pos.truncate(idx);
        }
        self.dim = dim;
    }

    // TODO reconsider; was hastily added
    #[inline]
    pub fn push_dense(&mut self, x: T)
    where
        T: Zero + PartialEq,
    {
        if &x != &Zero::zero() {
            self.pos.push(self.dim);
            self.val.push(x);
        }

        self.dim += 1;
    }

    /// The number of elements stored, including any explicitly-stored zeroes.
    ///
    /// Contrast with `Shape::dim()`, which refers to the length of the *dense* array that the
    /// SparseVec represents (including implicit zeros)
    #[inline]
    pub fn nnz(&self) -> usize {
        self.val.len()
    }

    /// Obtain a reference to the slice of sorted positions.
    ///
    /// This is a slice of length `self.nnz()` containing the dense positions (in
    /// sorted order) of the explicit values in the vector.
    #[inline]
    pub fn as_positions(&self) -> &[usize] {
        &self.pos[..]
    }

    /// Obtain a reference to the slice of explicit values.
    ///
    /// This is a slice of length `self.nnz()` of values corresponding in order to
    /// the positions in `self.as_positions()`
    #[inline]
    pub fn as_values(&self) -> &[T] {
        &self.val[..]
    }
}

impl<T> Shape for SparseVec<T> {
    #[inline(always)]
    fn dim(&self) -> usize {
        self.dim
    }
}

//----------------------------------------------------------

impl<T> IntoSparseIterator for SparseVec<T> {
    type Value = T;
    type IntoSparseIter = IntoSparseIter<T>;

    #[inline]
    fn into_sparse_iter(self) -> IntoSparseIter<T> {
        IntoSparseIter {
            dim: self.dim,
            it: zip!(self.pos, self.val),
        }
    }
}

/// Implementation detail of `SparseVec::into_sparse_iter`
pub struct IntoSparseIter<T> {
    dim: usize,
    it: ::std::iter::Zip<::std::vec::IntoIter<usize>, ::std::vec::IntoIter<T>>,
}

/// Implementation detail of `SparseVec::sparse_iter`
pub struct SparseIter<'a, T: 'a> {
    dim: usize,
    it: ::std::iter::Zip<
        ::std::iter::Cloned<::std::slice::Iter<'a, usize>>,
        ::std::slice::Iter<'a, T>,
    >,
}

//----------------------------------------------------------
// Indexing

impl<T> SparseVec<T> {
    // Same return value as `binary_search` for slices. This exists so that we can try switching out
    // to linear searches, or perhaps switching between methods based on `nnz`
    #[inline]
    fn find_idx(&self, pos: usize) -> Result<usize, usize> {
        // debug_assert because callers should already be doing their own checks
        debug_assert!(pos <= self.dim());
        self.as_positions().binary_search(&pos)
    }

    // merges the Ok/Err cases of find_idx
    #[inline]
    fn find_insertion_idx(&self, pos: usize) -> usize {
        match self.find_idx(pos) {
            Ok(x) | Err(x) => x,
        }
    }

    // FIXME: It is totally possible to return references now that this produces an
    //  Option.  However, I'm not sure about the name! (need to consider in the context
    //  of set_value and unset_value as well)
    /// Index by dense position.
    ///
    /// Returns `Some(T)` if the position contains an explicit value (including explicit
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

    /// Set an explicit value at a dense position.
    ///
    /// Returns `Some(old_value)` if an explicit value previously existed at that location.
    /// Note that, due to the sparse format, there is an inherent cost in searching for where
    /// to place the item.
    ///
    /// # Panics
    ///
    /// Panics if `pos >= self.dim()`.
    // TODO code example
    pub fn set_explicit(&mut self, pos: usize, value: T) -> Option<T> {
        self.check_index(pos);
        match self.find_idx(pos) {
            Ok(idx) => Some(::std::mem::replace(&mut self.val[idx], value)),
            Err(idx) => {
                self.pos.insert(idx, pos);
                self.val.insert(idx, value);
                None
            }
        }
    }

    /// Unset the explicit value at a dense position, if one exists.
    ///
    /// Returns `Some(old_value)` with the removed value if one existed at that position.
    /// An implicit zero will be left in its place.
    /// Note that, due to the sparse format, there is an inherent cost in searching for the item.
    ///
    /// # Panics
    ///
    /// Panics if `pos >= self.dim()`.
    // TODO code example
    pub fn remove_explicit(&mut self, pos: usize) -> Option<T> {
        self.check_index(pos);
        match self.find_idx(pos) {
            Ok(idx) => {
                self.pos.remove(idx);
                Some(self.val.remove(idx))
            }
            Err(_) => None,
        }
    }

    /// Set an explicit nonzero value or an implicit zero.
    ///
    /// This invokes either `set_explicit` or `remove_explicit` based on whether the input
    /// value is zero.
    /// Note that, due to the sparse format, there is an inherent cost in searching for where
    /// to place the item.
    ///
    /// # Panics
    ///
    /// Panics if `pos >= self.dim()`.
    // FIXME
    pub fn set_dense(&mut self, pos: usize, value: T)
    where
        T: PartialEq + Zero,
    {
        if &value == &Zero::zero() {
            self.remove_explicit(pos);
        } else {
            self.set_explicit(pos, value);
        }
    }

    /// Obtain a view into a region of the `SparseVec`.
    ///
    /// The resulting `SparseSlice` object provides a similar interface to `SparseVec`.
    ///
    /// There is no `slice_mut`, and one cannot be reasonably provided due to the nature of
    /// operations on sparse vectors; even just setting an element may require inserting
    /// values into the underlying vector.
    ///
    /// The reason this is not provided via the `Index` trait is because the returned object
    /// cannot be represented as a borrow.
    ///
    /// # Panics
    ///
    /// Matches the panic semantics of `Vec as Index<Range<usize>>`.
    pub fn slice<'a, R>(&'a self, pos: R) -> SparseSlice<'a, T>
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
        let raw = RawSparseSlice {
            dim: pos.len(),
            val: &self.val[idx.clone()],
            src_pos: &self.pos[idx],
            offset: pos.start,
        };
        unsafe { raw.to_sparse_slice_unchecked() }
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

impl<T> DenseIndex<usize> for SparseVec<T>
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

#[test]
fn test_dense_index_ops() {
    // test function for {get,set,remove}_{explicit,dense}

    // an implicit zero, an explicit nonzero, an explicit zero
    let sparse = SparseVec::from_parts(3, vec![2i32, 0], vec![1, 2]);
    assert_eq!(sparse.nnz(), 2);
    assert_eq!(sparse.get_dense(0), 0);
    assert_eq!(sparse.get_dense(1), 2);
    assert_eq!(sparse.get_dense(2), 0);

    assert_eq!(sparse.get_explicit(0), None);
    assert_eq!(sparse.get_explicit(1), Some(&2));
    assert_eq!(sparse.get_explicit(2), Some(&0));

    // two implicit zeros, two explicit nonzeros, an explicit zero
    let mut sparse = SparseVec::from_parts(5, vec![8i32, 9, 0], vec![2, 3, 4]);
    assert_eq!(sparse.set_explicit(0, 1), None); // implicit -> explicit
    assert_eq!(sparse.set_explicit(1, 0), None); // implicit -> explicit zero
    assert_eq!(sparse.set_explicit(2, 1), Some(8)); // explicit -> explicit
    assert_eq!(sparse.set_explicit(3, 0), Some(9)); // explicit -> explicit zero
    assert_eq!(sparse.set_explicit(4, 1), Some(0)); // test on existing explicit zero
    assert_eq!(
        sparse,
        SparseVec::from_parts(5, vec![1, 0, 1, 0, 1], vec![0, 1, 2, 3, 4])
    );

    // two implicit zeros, two explicit nonzeros, an explicit zero
    let mut sparse = SparseVec::from_parts(5, vec![8i32, 9, 0], vec![2, 3, 4]);
    sparse.set_dense(0, 1); // implicit -> explicit
    sparse.set_dense(1, 0); // implicit -> implicit
    sparse.set_dense(2, 1); // explicit -> explicit
    sparse.set_dense(3, 0); // explicit -> implicit
    sparse.set_dense(4, 1); // test on existing explicit zero
    assert_eq!(
        sparse,
        SparseVec::from_parts(5, vec![1, 1, 1], vec![0, 2, 4])
    );

    // two implicit zeros, two explicit nonzeros, an explicit zero
    let mut sparse = SparseVec::from_parts(5, vec![8i32, 9, 0], vec![2, 3, 4]);
    assert_eq!(sparse.remove_explicit(1), None); // something implicit
    assert_eq!(sparse.remove_explicit(2), Some(8)); // something explicit
    assert_eq!(sparse, SparseVec::from_parts(5, vec![9, 0], vec![3, 4]));
}

#[test]
#[should_panic(expected = "but ends at")]
fn test_bad_slice1() {
    SparseVec::from_dense(vec![0, 1, 0, 2, 0, 3, 0, 4]).slice(6..5);
}

#[test]
#[should_panic(expected = "out of range")]
fn test_bad_slice2() {
    SparseVec::from_dense(vec![0, 1, 0, 2, 0, 3, 0, 4]).slice(6..12);
}

//----------------------------------------------------------

impl<T> Iterator for IntoSparseIter<T> {
    type Item = (usize, T);
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.it.next()
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.it.size_hint()
    }
}

impl<'a, T> Iterator for SparseIter<'a, T> {
    type Item = (usize, &'a T);
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.it.next()
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.it.size_hint()
    }
}

impl<T> SparseIterator for IntoSparseIter<T> {
    type Value = T;
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

impl<T> ExactSizeIterator for IntoSparseIter<T> {
    fn len(&self) -> usize {
        self.it.len()
    }
}
impl<'a, T> ExactSizeIterator for SparseIter<'a, T> {
    fn len(&self) -> usize {
        self.it.len()
    }
}

impl<T> DoubleEndedIterator for IntoSparseIter<T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.it.next_back()
    }
}
impl<'a, T> DoubleEndedIterator for SparseIter<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.it.next_back()
    }
}

impl<T> Shape for IntoSparseIter<T> {
    fn dim(&self) -> usize {
        self.dim
    }
}
impl<'a, T> Shape for SparseIter<'a, T> {
    fn dim(&self) -> usize {
        self.dim
    }
}

impl<T> ShapedSparseIterator for IntoSparseIter<T> {}
impl<T> UniqueSparseIterator for IntoSparseIter<T> {}
impl<T> SortedSparseIterator for IntoSparseIter<T> {}
impl<'a, T> ShapedSparseIterator for SparseIter<'a, T> {}
impl<'a, T> UniqueSparseIterator for SparseIter<'a, T> {}
impl<'a, T> SortedSparseIterator for SparseIter<'a, T> {}

//----------------------------------------------------------

#[test]
fn test_vec_into_from_dense() {
    // something with zeros at endpoints
    let s = SparseVec::from_parts(10, vec![7f32, 4., 5.], vec![3, 6, 7]);
    let d = vec![0., 0., 0., 7., 0., 0., 4., 5., 0., 0.];
    assert_eq!(s.clone().into_dense(), d.clone());
    assert_eq!(SparseVec::from_dense(d), s);

    // something with occupied endpoints
    let s = SparseVec::from_parts(10, vec![7f32, 4.], vec![0, 9]);
    let d = vec![7., 0., 0., 0., 0., 0., 0., 0., 0., 4.];
    assert_eq!(s.clone().into_dense(), d.clone());
    assert_eq!(SparseVec::from_dense(d), s);

    // explicit zeros are lost on a round-trip conversion to dense and back
    let s1 = SparseVec::from_parts(5, vec![0.], vec![3]);
    let d1 = s1.clone().into_dense();
    let s2 = SparseVec::from_dense(d1.clone());
    let d2 = s2.clone().into_dense();
    assert_eq!(d1, d2);
    assert_eq!(s1.nnz(), 1);
    assert_eq!(s2.nnz(), 0);
}

#[test]
#[should_panic(expected = "exceeds dimension")]
fn test_bad_vec1() {
    SparseVec::from_parts(12, vec![3., 7.], vec![2, 12]);
}

#[test]
#[should_panic(expected = "duplicate")]
fn test_bad_vec2() {
    SparseVec::from_parts(12, vec![3., 1., 2., 7.], vec![2, 3, 3, 4]);
}

#[test]
#[should_panic(expected = "sorted")]
fn test_bad_vec3() {
    SparseVec::from_parts(12, vec![3., 1., 2., 7.], vec![2, 7, 3, 4]);
}

#[test]
#[should_panic(expected = "assertion failed")]
fn test_bad_vec4() {
    SparseVec::from_parts(12, vec![3., 1., 2.], vec![2, 7, 3, 4]);
}

#[test]
fn test_reshape() {
    let mut dense = vec![0u32, 1, 6, 0, 2, 3, 0, 0, 7];
    let mut sparse = SparseVec::from_dense(dense.clone());
    assert_eq!(sparse.dim(), 9);
    assert_eq!(sparse.nnz(), 5);

    // same length
    let mut s2 = sparse.clone();
    s2.reshape(sparse.dim());
    assert_eq!(sparse, s2);

    // smaller
    sparse.reshape(5); // vec![0,1,6,0,2]
    dense.truncate(5);
    assert_eq!(sparse.dim(), 5);
    assert_eq!(sparse.nnz(), 3);
    assert_eq!(sparse.clone().into_dense(), dense);

    // longer
    sparse.reshape(9); // vec![0,1,6,0,2,0,0,0,0]
    dense.extend(::std::iter::repeat(0).take(4));
    assert_eq!(sparse.dim(), 9);
    assert_eq!(sparse.nnz(), 3);
    assert_eq!(sparse.clone().into_dense(), dense);

    // to zero
    sparse.reshape(0);
    assert_eq!(sparse.dim(), 0);
    assert_eq!(sparse.nnz(), 0);
    assert_eq!(sparse.clone().into_dense(), vec![]);

    // from zero
    sparse.reshape(1);
    assert_eq!(sparse.clone().into_dense(), vec![0]);
}
//
//// Checking that into_dense() is faster than densify.
//#[bench]
//fn bench_todense_densify_plus_clone(b: &mut Bencher) {
//    let v = SparseVec::from_parts(32, vec![2., 7., 0., 0., 1.], vec![5, 9, 15, 16, 25]);
//    b.iter(|| v.clone().into_sparse_iter().densify().collect::<Vec<f64>>())
//}
//
//// This is here as a reference point to give an idea of how much of the cost is due to the
////  `clone()` required for benchmarking purposes
//#[bench]
//fn bench_todense_densify(b: &mut Bencher) {
//    let v = SparseVec::from_parts(32, vec![2., 7., 0., 0., 1.], vec![5, 9, 15, 16, 25]);
//    b.iter(|| {
//        v.sparse_iter()
//            .densify_with(&0.)
//            .cloned()
//            .collect::<Vec<f64>>()
//    })
//}
//
//#[bench]
//fn bench_todense_into_dense_plus_clone(b: &mut Bencher) {
//    let v = SparseVec::from_parts(32, vec![2., 7., 0., 0., 1.], vec![5, 9, 15, 16, 25]);
//    b.iter(|| v.clone().into_dense())
//}
