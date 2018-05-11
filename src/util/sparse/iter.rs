use num_traits::Zero;
use std::cmp::Ordering;
use std::iter::Peekable;

use traits::Shape;

#[cfg(test)]
use vec::SparseVec;

//----------------------------------------------------------

/// An iterator which generalizes 1D "sparse" storage types.
///
/// A `SparseIterator` is an iterator whose items are of the form `(pos, value)`, where `pos` is an
/// index into a hypothetical "dense" array.
///
/// Generally speaking, functions with names like `sparse_map` or `sparse_cloned` differ from the
/// corresponding functions on `Iterator` in that the `sparse` versions operate only on the value
/// part of the item tuple (whereas the `Iterator` methods will operate on the entire tuple).
// TODO: `map_values`, `clone_values`, instead of `sparse_*` ... whaddya think?
pub trait SparseIterator: Iterator<Item = (usize, <Self as SparseIterator>::Value)> {
    type Value;

    /// Creates a sparse iterator which clones the values it yields.
    ///
    /// More specifically, it maps an iterator of `(usize, &T)` to `(usize, T)`.
    /// (contrast with `Iterator::cloned` which expects `&(usize, T)`, which is not a valid
    ///  item type for a SparseIterator)
    #[inline]
    fn sparse_cloned<'a, T: 'a>(self) -> SparseCloned<Self>
    where
        Self: Sized + SparseIterator<Value = &'a T>,
        T: Clone,
    {
        SparseCloned { it: self }
    }

    /// Creates a new sparse iterator that applies a function to the `value` part of each
    /// `(pos, value)` item in this sparse iterator.  That is, for each `(pos, value)` item
    /// in this iterator, the new iterator will return `(pos, f(value))`.
    ///
    /// Contrast with `Iterator::map`, which would invoke `f((pos, value))`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rsp2_sparse_matrix::{SparseVec,SparseIterator};
    ///
    /// let a = SparseVec::from_dense(vec![0,3,0,0,4]);
    /// let mut it = a.sparse_iter().sparse_map(|&x| 2 * x);
    /// assert_eq!(it.next(), Some((1,6)));
    /// assert_eq!(it.next(), Some((4,8)));
    /// assert!(it.next().is_none());
    /// ```
    #[inline]
    fn sparse_map<B, F>(self, f: F) -> SparseMap<Self, F>
    where
        Self: Sized,
        F: FnMut(Self::Value) -> B,
    {
        SparseMap { iter: self, f: f }
    }

    /// Creates a sparse iterator that applies the predicate to the `value` part of each
    /// `(pos, value)` item in this sparse iterator.  It will only return the `(pos, value)`
    /// items for which `predicate(&value)` evaluates to `true`.
    ///
    /// Contrast with `Iterator::filter`, which would evaluate `predicate(&(pos, value))`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rsp2_sparse_matrix::{SparseVec,SparseIterator};
    ///
    /// let a = SparseVec::from_dense(vec![0,3,0,0,4]);
    /// let mut it = a.sparse_iter().sparse_filter(|&x| *x > 3);
    /// assert_eq!(it.next(), Some((4,&4)));
    /// assert!(it.next().is_none());
    /// ```
    #[inline]
    fn sparse_filter<P>(self, predicate: P) -> SparseFilter<Self, P>
    where
        Self: Sized,
        P: FnMut(&Self::Value) -> bool,
    {
        SparseFilter {
            iter: self,
            predicate,
        }
    }

    /// Creates a sparse iterator that both filters and maps `(pos, value)` items based on `value`.
    /// If `f(value)` returns `None`, that item is skipped.  Otherwise, if `f(value)` returns
    /// `Some(y)`, then the new item is `(pos, y)`.
    ///
    /// Contrast with `Iterator::filter_map`, which would invoke `f((pos, value))`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rsp2_sparse_matrix::{SparseVec,SparseIterator};
    ///
    /// let a = SparseVec::from_dense(vec![0,3,0,0,4]);
    /// let mut it = a.sparse_iter().sparse_filter_map(|&x| if x > 3 {Some(2 * x)} else {None});
    /// assert_eq!(it.next(), Some((4,8)));
    /// assert!(it.next().is_none());
    /// ```
    #[inline]
    fn sparse_filter_map<B, F>(self, f: F) -> SparseFilterMap<Self, F>
    where
        Self: Sized,
        F: FnMut(Self::Value) -> Option<B>,
    {
        SparseFilterMap { iter: self, f }
    }

    /// Iterate over elements contained in each of two sparse iterators.
    ///
    /// This is a form of "zipped" iterator over positions that are contained in *both* sparse
    /// iterators; e.g. if `self` contains the position-value pair `(pos,a)` and `other` contains
    /// `(pos,b)`, then the intersection will contain `(pos, (a,b))`.  Positions only contained
    /// in one iterator are skipped.
    ///
    /// # Examples
    ///
    /// ```
    /// use rsp2_sparse_matrix::{SparseVec,SparseIterator,IntoSparseIterator};
    ///
    /// let a = SparseVec::from_dense(vec![0i64, 0, 1, 2]);
    /// let b = SparseVec::from_dense(vec![0i64, 4, 3, 0]);
    /// let mut it = a.into_sparse_iter().sparse_intersection(b);
    ///
    /// // only position 2 is nonzero in both vectors
    /// assert_eq!(it.next(), Some((2usize, (1, 3))));
    /// assert_eq!(it.next(), None);
    /// ```
    #[inline]
    fn sparse_intersection<B>(self, other: B) -> SparseIntersection<Self, B::IntoSparseIter>
    where
        Self: Sized + UniqueSparseIterator + SortedSparseIterator,
        B: IntoSparseIterator,
        B::IntoSparseIter: UniqueSparseIterator + SortedSparseIterator,
    {
        SparseIntersection::new(self, other.into_sparse_iter())
    }

    /// Iterate over elements contained either of two sparse iterators.
    ///
    /// This is a form of "zipped" iterator over positions that are contained in *at least one*
    /// sparse iterator; e.g. if `self` contains the position-value pair `(pos,a)` and `other` does
    /// not contain `pos`, then the intersection will contain `(pos, Left(a))`.
    ///
    /// The item type is (`usize`, [`UnionValue`](enum.UnionValue.html)).
    ///
    /// # Examples
    ///
    /// ```
    /// use rsp2_sparse_matrix::{SparseVec,SparseIterator,IntoSparseIterator};
    /// use rsp2_sparse_matrix::UnionValue::*;
    ///
    /// let a = SparseVec::from_dense(vec![0i64, 0, 1, 2]);
    /// let b = SparseVec::from_dense(vec![0i64, 4, 3, 0]);
    /// let mut it = a.into_sparse_iter().sparse_union(b);
    ///
    /// // positions 1, 2, and 3 are each contained in at least one vector
    /// assert_eq!(it.next(), Some((1usize, Right(4))));
    /// assert_eq!(it.next(), Some((2usize, Both(1,3))));
    /// assert_eq!(it.next(), Some((3usize, Left(2))));
    /// assert_eq!(it.next(), None);
    /// ```
    fn sparse_union<B>(self, other: B) -> SparseUnion<Self, B::IntoSparseIter>
    where
        Self: Sized + UniqueSparseIterator + SortedSparseIterator,
        B: IntoSparseIterator,
        B::IntoSparseIter: UniqueSparseIterator + SortedSparseIterator,
    {
        SparseUnion::new(self, other.into_sparse_iter())
    }

    /// A sparse iterator adapter which implements `ShapedSparseIterator`.
    ///
    /// This is useful for adding shape information back after the use of adaptors such as
    /// `SparseIntersection` which cannot provide such information.  It is a logical error to
    /// call `as_shaped` with a dimension that is insufficient for the items in the iterator.
    /// (i.e. it must be the case that `dim > pos` for each `(pos, value)` item).
    // FIXME should perhaps be `declare_shaped`.  `as_` generally denotes a member
    // borrowing method.
    #[inline]
    fn as_shaped(self, dim: usize) -> Shaped<Self>
    where
        Self: Sized,
    {
        Shaped {
            iter: self,
            dim: dim,
        }
    }

    /// Add `SortedSparseIterator` to an iterator that lacks it.
    ///
    /// Call this method on a sparse iterator to declare that it meets the contract
    /// of `SortedSparseIterator`, for use in functions that require the trait.
    /// This is useful for e.g. garden-variety iterators wrapped in a `SparseWrapper`.
    ///
    /// It is a logical error to call `declare_sorted` on an iterator whose indices
    /// are not actually sorted.
    #[inline]
    fn declare_sorted(self) -> DeclareSorted<Self>
    where
        Self: Sized,
    {
        DeclareSorted { iter: self }
    }

    /// Add `UniqueSparseIterator` to an iterator that lacks it.
    ///
    /// Call this method on a sparse iterator to declare that it meets the contract
    /// of `UniqueSparseIterator`, for use in functions that require the trait.
    /// This is useful for e.g. garden-variety iterators wrapped in a `SparseWrapper`.
    ///
    /// It is a logical error to call `declare_unique` on an iterator whose indices
    /// are not actually unique.
    #[inline]
    fn declare_unique(self) -> DeclareUnique<Self>
    where
        Self: Sized,
    {
        DeclareUnique { iter: self }
    }
}

/// A `SparseIterator` which represents a sequence of a known size.
///
/// `ShapedSparseIterator` describes sparse iterators where the length of the complete, dense
/// sequence they represent is known and is provided by `Shape::dim`.  This makes it possible to
/// "densify" them into a traditional iterator by filling in the missing elements with zeros.
/// (it is not possible to do this for `SparseIterator` in general because the number of zeros
///  after the last nonzero element is not known)
///
/// Types which implement both `SparseIterator` and `Shape` do not automatically implement
/// `ShapedSparseIterator`.  This is just to be conservative for the time being, until the
/// nature of the problem is better understood.  Furthermore, there is an additional contract
/// associated with `ShapedSparseIterator`:
/// **Implementors of `ShapedSparseIterator` must guarantee that, for any `(pos, value)` item in
/// the sparse sequence, `pos < self.dim()`.**
///
/// Note that `Self: ExactSizeIterator` and `Self: ShapedSparseIterator` are completely
/// orthogonal; you can have either one without the other. Because `self` is a `SparseIterator`,
/// `ExactSizeIterator::len` actually refers to the number of nonzero elements.
/// Another way to look at it is that `self.dim() == self.densify().len()`.
///
/// Because `ShapedSparseIterator` is not an unsafe trait, invalid implementations of the trait
/// must not be able to cause undefined behavior.  Therefore, it is incorrect to make unsafe
/// optimizations based solely on the presence of the trait.
pub trait ShapedSparseIterator: SparseIterator + Shape {
    /// Produce an iterator over the "dense" sequence.
    ///
    /// For each item `(p, v)` in the original sparse sequence, the `p`th item in the densified
    /// sequence will be `v`.  Missing elements are filled with zeros.
    fn densify(self) -> Densify<Self, Self::Value>
    where
        Self: Sized + SortedSparseIterator + UniqueSparseIterator,
        Self::Value: Zero,
    {
        self.densify_with(Zero::zero())
    }

    /// Densify with a custom fill value.
    fn densify_with(mut self, fill: Self::Value) -> Densify<Self, Self::Value>
    where
        Self: Sized + SortedSparseIterator + UniqueSparseIterator,
    {
        let first = self.next();
        Densify {
            a: self,
            pos: 0,
            peek: first,
            zero: fill,
        }
    }
}

/// Marker trait for sparse iterators sorted by position.
///
/// A `SparseIterator` implementing this trait promises that, if `(pos1, val1)` and `(pos2, val2)`
/// are successive items from the iterator, then `pos1 <= pos2`.
///
/// Because `SortedSparseIterator` is not an unsafe trait, invalid implementations of the trait
/// must not be able to cause undefined behavior.  Therefore, it is incorrect to make unsafe
/// optimizations based solely on the presence of the trait.
pub trait SortedSparseIterator: SparseIterator {}

/// Marker trait for sparse iterators with no duplicate positions.
///
/// A `SparseIterator` implementing this trait promises that no `pos` value ever appears more than
/// once among its `(pos, val)` tuples.
///
/// Because `UniqueSparseIterator` is not an unsafe trait, invalid implementations of the trait
/// must not be able to cause undefined behavior.  Therefore, it is incorrect to make unsafe
/// optimizations based solely on the presence of the trait.
pub trait UniqueSparseIterator: SparseIterator {}

//----------------------------------------------------------

/// Types from which a `SparseIterator` can be produced.
///
/// This gives some functions a slightly friendlier API, in the same manner that
///  `IntoIter` does for `zip`.  No, there is no `sparse for`. :P
pub trait IntoSparseIterator {
    type Value;
    type IntoSparseIter: SparseIterator<Value = Self::Value>;
    fn into_sparse_iter(self) -> Self::IntoSparseIter;
}

impl<I: SparseIterator> IntoSparseIterator for I {
    type Value = <I as SparseIterator>::Value;
    type IntoSparseIter = I;
    #[inline]
    fn into_sparse_iter(self) -> I {
        self
    }
}

//----------------------------------------------------------

/// Wraps a regular iterator into a sparse iterator.
///
/// This allows the construction of sparse iterators from a much wider variety
/// of sources.  The iterator must produce `(index, value)` pairs just as a
/// standard `SparseIterator` would.
///
/// `SparseWrapper` itself does not directly implement any of the sparse
/// iterator marker traits, because it knows nothing about the indices in
/// the wrapped `Iterator`.  If you know that the indices in your iterator
/// meet the contract of these marker traits, you can add them by calling
/// methods: `as_shaped`, `declare_unique`, and `declare_sorted`.
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
#[derive(Clone)]
pub struct SparseWrapper<I: Iterator<Item = (usize, T)>, T> {
    pub iter: I,
}

// Trait checklist:
// [O] ExactSizeIterator
// [O] DoubleEndedIterator
// [X] ShapedSparseIterator - for reasons explained in docstring
// [X] UniqueSparseIterator
// [X] SortedSparseIterator

impl<I: Iterator<Item = (usize, T)>, T> Iterator for SparseWrapper<I, T> {
    type Item = I::Item;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}
impl<I: ExactSizeIterator<Item = (usize, T)>, T> ExactSizeIterator for SparseWrapper<I, T> {}
impl<I: DoubleEndedIterator<Item = (usize, T)>, T> DoubleEndedIterator for SparseWrapper<I, T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back()
    }
}

//----------------------------------------------------------

/// Implementation detail of `SparseIterator::as_shaped`.
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
#[derive(Clone)]
pub struct Shaped<I> {
    dim: usize,
    iter: I,
}

/// Implementation detail of `SparseIterator::declare_sorted`.
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
#[derive(Clone)]
pub struct DeclareSorted<I> {
    iter: I,
}

/// Implementation detail of `SparseIterator::declare_unique`.
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
#[derive(Clone)]
pub struct DeclareUnique<I> {
    iter: I,
}

impl<I: SparseIterator> Shaped<I> {
    #[inline]
    fn validate(&self, item: Option<<Self as Iterator>::Item>) -> Option<<Self as Iterator>::Item> {
        if let Some((i, x)) = item {
            debug_assert!(i < self.dim);
            Some((i, x))
        } else {
            None
        }
    }
}

impl<I: SparseIterator> Iterator for Shaped<I> {
    type Item = I::Item;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let x = self.iter.next();
        self.validate(x)
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}
impl<I: SparseIterator> Iterator for DeclareSorted<I> {
    type Item = I::Item;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}
impl<I: SparseIterator> Iterator for DeclareUnique<I> {
    type Item = I::Item;
    #[inline] fn next(&mut self) -> Option<Self::Item> { self.iter.next() }
    #[inline] fn size_hint(&self) -> (usize, Option<usize>) { self.iter.size_hint() }
}

impl<I: SparseIterator> SparseIterator for Shaped<I> { type Value = I::Value; }
impl<I: SparseIterator> SparseIterator for DeclareSorted<I> { type Value = I::Value; }
impl<I: SparseIterator> SparseIterator for DeclareUnique<I> { type Value = I::Value; }

// Trait checklist:
// [O] ExactSizeIterator
// [O] DoubleEndedIterator
// [O] ShapedSparseIterator
// [O] UniqueSparseIterator
// [O] SortedSparseIterator

impl<I: SparseIterator + ExactSizeIterator> ExactSizeIterator for Shaped<I> {
    #[inline] fn len(&self) -> usize { self.iter.len() }
}
impl<I: SparseIterator + ExactSizeIterator> ExactSizeIterator for DeclareSorted<I> {
    #[inline] fn len(&self) -> usize { self.iter.len() }
}
impl<I: SparseIterator + ExactSizeIterator> ExactSizeIterator for DeclareUnique<I> {
    #[inline] fn len(&self) -> usize { self.iter.len() }
}

impl<I: SparseIterator + DoubleEndedIterator> DoubleEndedIterator for Shaped<I> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        let x = self.iter.next_back();
        self.validate(x)
    }
}
impl<I: SparseIterator + DoubleEndedIterator> DoubleEndedIterator for DeclareSorted<I> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back()
    }
}
impl<I: SparseIterator + DoubleEndedIterator> DoubleEndedIterator for DeclareUnique<I> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back()
    }
}

impl<I: SparseIterator> Shape for Shaped<I> {
    #[inline] fn dim(&self) -> usize { self.dim }
}
impl<I: Shape> Shape for DeclareSorted<I> {
    #[inline] fn dim(&self) -> usize { self.iter.dim() }
}
impl<I: Shape> Shape for DeclareUnique<I> {
    #[inline] fn dim(&self) -> usize { self.iter.dim() }
}

impl<I: SparseIterator> ShapedSparseIterator for Shaped<I> {}
impl<I: ShapedSparseIterator> ShapedSparseIterator for DeclareSorted<I> {}
impl<I: ShapedSparseIterator> ShapedSparseIterator for DeclareUnique<I> {}

impl<I: SparseIterator> SortedSparseIterator for DeclareSorted<I> {}
impl<I: SortedSparseIterator> SortedSparseIterator for Shaped<I> {}
impl<I: SortedSparseIterator> SortedSparseIterator for DeclareUnique<I> {}

impl<I: SparseIterator> UniqueSparseIterator for DeclareUnique<I> {}
impl<I: UniqueSparseIterator> UniqueSparseIterator for Shaped<I> {}
impl<I: UniqueSparseIterator> UniqueSparseIterator for DeclareSorted<I> {}

//----------------------------------------------------------

#[test]
fn test_as_shaped() {
    // let's make a simple passthrough iterator which doesn't implement ShapedIterator
    struct MyIter {
        iter: ::vec::IntoSparseIter<i32>,
    }
    impl MyIter {
        fn new(s: SparseVec<i32>) -> Self {
            MyIter {
                iter: s.into_sparse_iter(),
            }
        }
    }
    impl Iterator for MyIter {
        type Item = (usize, i32);
        fn next(&mut self) -> Option<Self::Item> {
            self.iter.next()
        }
    }
    impl DoubleEndedIterator for MyIter {
        fn next_back(&mut self) -> Option<Self::Item> {
            self.iter.next_back()
        }
    }
    impl SparseIterator for MyIter {
        type Value = i32;
    }
    impl UniqueSparseIterator for MyIter {}
    impl SortedSparseIterator for MyIter {}

    // and a test which checks that the new iterator densifies identically to the original vector
    // when wrapped using `as_shaped(s.dim())`
    fn test(s: SparseVec<i32>, myiter: MyIter) {
        let dense = myiter.as_shaped(s.dim()).densify().collect::<Vec<_>>();
        assert_eq!(dense, s.into_dense());
    }

    // dim = 0
    let s = SparseVec::<i32>::from_parts_strictly_sorted(0, vec![], vec![]);
    let myiter = MyIter::new(s.clone());
    test(s, myiter);

    // nnz = 0
    let s = SparseVec::<i32>::from_parts_strictly_sorted(10, vec![], vec![]);
    let myiter = MyIter::new(s.clone());
    test(s, myiter);

    // forward
    let s = SparseVec::from_parts_strictly_sorted(10, vec![7i32, 4], vec![3, 6]);
    let myiter = MyIter::new(s.clone());
    test(s.clone(), myiter);

    // check items returned by `rev` (in case of copypasta error)
    let myiterrev = MyIter::new(s.clone()).rev();
    let siterrev = s.into_sparse_iter().rev();
    assert!(siterrev.eq(myiterrev));
}

// FIXME no tests for `declare_sorted`, `declare_unique`

//----------------------------------------------------------

/// Implementation detail of `SparseIterator::sparse_cloned`.
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
#[derive(Clone)]
pub struct SparseCloned<I> {
    it: I,
}

impl<'a, I, T> Iterator for SparseCloned<I>
where
    T: 'a + Clone,
    I: SparseIterator<Value = &'a T>,
{
    type Item = (usize, T);
    #[inline]
    fn next(&mut self) -> Option<(usize, T)> {
        self.it.next().map(|(i, x)| (i, x.clone()))
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.it.size_hint()
    }
}

impl<'a, I, T> SparseIterator for SparseCloned<I>
where
    T: 'a + Clone,
    I: SparseIterator<Value = &'a T>,
{
    type Value = T;
}

// Trait checklist:
// [O] ExactSizeIterator
// [O] DoubleEndedIterator
// [O] ShapedSparseIterator
// [O] UniqueSparseIterator
// [O] SortedSparseIterator

impl<'a, I, T> ExactSizeIterator for SparseCloned<I>
where
    T: 'a + Clone,
    I: SparseIterator<Value = &'a T> + ExactSizeIterator,
{
    #[inline] fn len(&self) -> usize { self.it.len() }
}

impl<'a, I, T> DoubleEndedIterator for SparseCloned<I>
where
    T: 'a + Clone,
    I: SparseIterator<Value = &'a T> + DoubleEndedIterator,
{
    #[inline]
    fn next_back(&mut self) -> Option<(usize, T)> {
        self.it.next_back().map(|(i, x)| (i, x.clone()))
    }
}

impl<I> Shape for SparseCloned<I>
where
    I: Shape,
{
    #[inline] fn dim(&self) -> usize { self.it.dim() }
}

impl<'a, I, T> ShapedSparseIterator for SparseCloned<I>
where
    T: 'a + Clone,
    I: ShapedSparseIterator<Value = &'a T>,
{}
impl<'a, I, T> UniqueSparseIterator for SparseCloned<I>
where
    T: 'a + Clone,
    I: UniqueSparseIterator<Value = &'a T>,
{}
impl<'a, I, T> SortedSparseIterator for SparseCloned<I>
where
    T: 'a + Clone,
    I: SortedSparseIterator<Value = &'a T>,
{}

#[test]
fn test_sparse_cloned() {
    // dim = 0
    let s = SparseVec::<i32>::from_parts_strictly_sorted(0, vec![], vec![]);
    let mut iter = s.sparse_iter().sparse_cloned();
    assert_eq!(iter.dim(), 0);
    assert_eq!(iter.len(), 0);
    assert_eq!(iter.next(), None);
    assert_eq!(iter.next(), None);

    // nnz = 0
    let s = SparseVec::<i32>::from_parts_strictly_sorted(10, vec![], vec![]);
    let mut iter = s.sparse_iter().sparse_cloned();
    assert_eq!(iter.dim(), 10);
    assert_eq!(iter.len(), 0);
    assert_eq!(iter.next(), None);
    assert_eq!(iter.next(), None);

    // forward
    let s = SparseVec::from_parts_strictly_sorted(10, vec![7i32, 4], vec![3, 6]);
    let mut iter = s.sparse_iter().sparse_cloned();
    assert_eq!(iter.dim(), 10);
    assert_eq!(iter.len(), 2);
    assert_eq!(iter.next(), Some((3, 7)));
    assert_eq!(iter.next(), Some((6, 4)));
    assert_eq!(iter.next(), None);
    assert_eq!(iter.next(), None);

    // reverse
    let mut iter = s.sparse_iter().sparse_cloned().rev();
    assert_eq!(iter.next(), Some((6, 4)));
    assert_eq!(iter.next(), Some((3, 7)));
    assert_eq!(iter.next(), None);
    assert_eq!(iter.next(), None);
}

//----------------------------------------------------------

/// Implementation detail of `ShapedSparseIterator::densify`.
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
#[derive(Clone)]
pub struct Densify<A, T> {
    a: A,
    pos: usize,
    peek: Option<(usize, T)>,
    zero: T,
}

impl<A, T: Clone> Iterator for Densify<A, T>
where
    A: ShapedSparseIterator<Value = T>,
{
    type Item = T;

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }

    fn next(&mut self) -> Option<Self::Item> {
        // I feel like the logic here is more complicated than it needs to be...

        let result;
        match self.peek.take() {
            Some((peek_pos, peek_val)) => {
                // current pos is a nonzero item
                if self.pos == peek_pos {
                    self.peek = self.a.next();
                    result = Some(peek_val);

                // current pos is between nonzero items
                } else {
                    debug_assert!(self.pos < peek_pos);
                    self.peek = Some((peek_pos, peek_val));
                    result = Some(self.zero.clone());
                }

                self.pos += 1;
            }

            None => {
                // beyond last nonzero item, but not beyond end
                if self.pos < self.a.dim() {
                    self.pos += 1;
                    result = Some(self.zero.clone());

                // beyond end
                } else {
                    result = None;
                }
            }
        }
        result
    }
}

// Trait checklist:
// [O] ExactSizeIterator
// [X] DoubleEndedIterator - Too nasty.

impl<A, T: Clone> ExactSizeIterator for Densify<A, T>
where
    A: ShapedSparseIterator<Value = T>,
{
    #[inline]
    fn len(&self) -> usize {
        self.a.dim()
    }
}

#[test]
fn test_densify_traits() {
    // ExactSizeIterator
    let s = SparseVec::from_parts_strictly_sorted(5, vec![1f32, 2.], vec![1, 3]);
    assert_eq!(s.into_sparse_iter().densify().len(), 5);
}

//----------------------------------------------------------

/// Implementation detail of `SparseIterator::sparse_union`.
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
pub struct SparseUnion<A: Iterator, B: Iterator> {
    a: Peekable<A>,
    b: Peekable<B>,
}

impl<A: Iterator, B: Iterator> SparseUnion<A, B> {
    #[inline]
    fn new(a: A, b: B) -> Self {
        SparseUnion {
            a: a.peekable(),
            b: b.peekable(),
        }
    }
}

/// Value type of [`SparseIterator::sparse_union`](trait.SparseIterator.html#method.sparse_union)
///
/// Represents a value from at least one of the iterators.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum UnionValue<L, R> {
    Left(L),
    Right(R),
    Both(L, R),
}

impl<A, B, T, U> Iterator for SparseUnion<A, B>
where
    A: SparseIterator<Value = T>,
    B: SparseIterator<Value = U>,
{
    type Item = (usize, UnionValue<T, U>);
    fn next(&mut self) -> Option<Self::Item> {
        use self::UnionValue::*;

        let &mut SparseUnion {
            ref mut a,
            ref mut b,
        } = self;

        let a_pos = a.peek().map(|&(i, _)| i);
        let b_pos = b.peek().map(|&(i, _)| i);

        // define some shorthand for extracting a value and advancing one of the
        //  iterators, to tidy up the ensuing match
        let aa = &mut || a.next().unwrap().1;
        let bb = &mut || b.next().unwrap().1;

        match (a_pos, b_pos) {
            (None, None) => None,
            (Some(i), None) => Some((i, Left(aa()))),
            (None, Some(k)) => Some((k, Right(bb()))),
            (Some(i), Some(k)) => Some(match i.cmp(&k) {
                Ordering::Less => (i, Left(aa())),
                Ordering::Greater => (k, Right(bb())),
                Ordering::Equal => (i, Both(aa(), bb())),
            }),
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (a_lo, a_hi) = self.a.size_hint();
        let (b_lo, b_hi) = self.b.size_hint();

        // Worst case scenario, all positions are mismatched resulting in a_hi + a_lo
        //  elements. Technically, UniqueSparseIterator together with the fact that indices
        //  are of a fixed type (usize) does place a hard upper bound of `::std::usize::MAX`
        //  on the number of items... but just in case tuple positions ever happen, I'd rather
        //  not use `saturating_add` for now
        let hi = if let (Some(a_hi), Some(b_hi)) = (a_hi, b_hi) {
            a_hi.checked_add(b_hi)
        } else {
            None
        }; // at least one is "unbounded"

        // If every position in the shorter iterator is also a position in the longer iterator,
        // we have the minimum case of `shorter.len()` items.
        let lo = ::std::cmp::min(a_lo, b_lo);
        (lo, hi)
    }
}

// Trait checklist:
// [X] ExactSizeIterator - Impossible.
// [X] DoubleEndedIterator - Possible! (...but ugly)
// [X] ShapedSparseIterator - Impossible. (it's an all or nothing choice)
// [O] UniqueSparseIterator
// [O] SortedSparseIterator

impl<A, B> SparseIterator for SparseUnion<A, B>
where
    A: SortedSparseIterator + UniqueSparseIterator,
    B: SortedSparseIterator + UniqueSparseIterator,
{
    type Value = UnionValue<A::Value, B::Value>;
}

impl<A, B> SortedSparseIterator for SparseUnion<A, B>
where
    A: SortedSparseIterator + UniqueSparseIterator,
    B: SortedSparseIterator + UniqueSparseIterator,
{}

impl<A, B> UniqueSparseIterator for SparseUnion<A, B>
where
    A: SortedSparseIterator + UniqueSparseIterator,
    B: SortedSparseIterator + UniqueSparseIterator,
{}

//----------------------------------------------------------

/// Implementation detail of `SparseIterator::sparse_intersection`.
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
#[derive(Clone)]
pub struct SparseIntersection<A, B> {
    a: A,
    b: B,
}
impl<A, B> SparseIntersection<A, B> {
    #[inline]
    fn new(a: A, b: B) -> Self
    where
        A: SparseIterator,
        B: SparseIterator,
    {
        SparseIntersection { a: a, b: b }
    }
}

impl<A, B, T, U> Iterator for SparseIntersection<A, B>
where
    A: SparseIterator<Value = T>,
    B: SparseIterator<Value = U>,
{
    type Item = (usize, (T, U));
    fn next(&mut self) -> Option<Self::Item> {
        let &mut SparseIntersection {
            ref mut a,
            ref mut b,
        } = self;
        if let (Some((i0, a0)), Some((k0, b0))) = (a.next(), b.next()) {
            let (mut icur, mut acur) = (i0, a0);
            let (mut kcur, mut bcur) = (k0, b0);
            loop {
                // scan to an index present in both vectors
                while icur < kcur {
                    if let Some((inext, anext)) = a.next() {
                        acur = anext;
                        icur = inext;
                    } else {
                        return None;
                    } // stop immediately if either iterator ends
                }
                while kcur < icur {
                    if let Some((knext, bnext)) = b.next() {
                        bcur = bnext;
                        kcur = knext;
                    } else {
                        return None;
                    }
                }
                // at this point, kcur >= icur
                if icur == kcur {
                    return Some((icur, (acur, bcur)));
                }
            }
        // at least one iterator had no elements remaining
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (_, aupper) = self.a.size_hint();
        let (_, bupper) = self.b.size_hint();
        (0, ::std::cmp::min(aupper, bupper)) // any number of items may not match, so no lower bound
    }
}

// Trait checklist:
// [X] ExactSizeIterator - Impossible.
// [X] DoubleEndedIterator - Possible! (just not trivial)  FIXME
// [X] ShapedSparseIterator - Impossible. (it's an all or nothing choice)
// [O] UniqueSparseIterator
// [O] SortedSparseIterator

impl<A, B> SparseIterator for SparseIntersection<A, B>
where
    A: SortedSparseIterator + UniqueSparseIterator,
    B: SortedSparseIterator + UniqueSparseIterator,
{
    type Value = (A::Value, B::Value);
}

impl<A, B> SortedSparseIterator for SparseIntersection<A, B>
where
    A: SortedSparseIterator + UniqueSparseIterator,
    B: SortedSparseIterator + UniqueSparseIterator,
{}

impl<A, B> UniqueSparseIterator for SparseIntersection<A, B>
where
    A: SortedSparseIterator + UniqueSparseIterator,
    B: SortedSparseIterator + UniqueSparseIterator,
{}

//----------------------------------------------------------

// these structs and impls are pretty much carbon copies of Map, Filter,
//  and FilterMap from the core library

/// Implementation detail of `SparseIterator::sparse_map`
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
#[derive(Clone)]
pub struct SparseMap<I, F> {
    iter: I,
    f: F,
}

/// Implementation detail of `SparseIterator::sparse_filter`
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
#[derive(Clone)]
pub struct SparseFilter<I, P> {
    iter: I,
    predicate: P,
}

/// Implementation detail of `SparseIterator::sparse_filter_map`
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
#[derive(Clone)]
pub struct SparseFilterMap<I, F> {
    iter: I,
    f: F,
}

impl<B, I: SparseIterator, F> Iterator for SparseMap<I, F>
where
    F: FnMut(I::Value) -> B,
{
    type Item = (usize, B);
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(i, x)| (i, (self.f)(x)))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<B, I: SparseIterator + DoubleEndedIterator, F> DoubleEndedIterator for SparseMap<I, F>
where
    F: FnMut(I::Value) -> B,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back().map(|(i, x)| (i, (self.f)(x)))
    }
}

impl<I: SparseIterator, P> Iterator for SparseFilter<I, P>
where
    P: FnMut(&I::Value) -> bool,
{
    type Item = (usize, I::Value);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        for (i, x) in self.iter.by_ref() {
            if (self.predicate)(&x) {
                return Some((i, x));
            }
        }
        None
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (_, upper) = self.iter.size_hint();
        (0, upper) // can't know a lower bound, due to the predicate
    }
}

impl<I: SparseIterator + DoubleEndedIterator, P> DoubleEndedIterator for SparseFilter<I, P>
where
    P: FnMut(&I::Value) -> bool,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        for (i, x) in self.iter.by_ref().rev() {
            if (self.predicate)(&x) {
                return Some((i, x));
            }
        }
        None
    }
}

impl<B, I: SparseIterator, F> Iterator for SparseFilterMap<I, F>
where
    F: FnMut(I::Value) -> Option<B>,
{
    type Item = (usize, B);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        for (i, x) in self.iter.by_ref() {
            if let Some(y) = (self.f)(x) {
                return Some((i, y));
            }
        }
        None
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (_, upper) = self.iter.size_hint();
        (0, upper) // can't know a lower bound, due to the predicate
    }
}

impl<B, I: SparseIterator + DoubleEndedIterator, F> DoubleEndedIterator for SparseFilterMap<I, F>
where
    F: FnMut(I::Value) -> Option<B>,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        for (i, x) in self.iter.by_ref().rev() {
            if let Some(y) = (self.f)(x) {
                return Some((i, y));
            }
        }
        None
    }
}

impl<B, I: SparseIterator + ExactSizeIterator, F> ExactSizeIterator for SparseMap<I, F>
where
    F: FnMut(I::Value) -> B,
{}

// NOTE: No ExactSizeIterator for the filtering adaptors; size impossible to know.

impl<I: Shape, F> Shape for SparseMap<I, F> {
    #[inline] fn dim(&self) -> usize { self.iter.dim() }
}
impl<B, I: SparseIterator, F> SparseIterator for SparseMap<I, F>
where
    F: FnMut(I::Value) -> B,
{
    type Value = B;
}
impl<B, I: ShapedSparseIterator, F> ShapedSparseIterator for SparseMap<I, F>
where
    F: FnMut(I::Value) -> B,
{}
impl<B, I: UniqueSparseIterator, F> UniqueSparseIterator for SparseMap<I, F>
where
    F: FnMut(I::Value) -> B,
{}
impl<B, I: SortedSparseIterator, F> SortedSparseIterator for SparseMap<I, F>
where
    F: FnMut(I::Value) -> B,
{}

impl<I: Shape, P> Shape for SparseFilter<I, P> {
    #[inline] fn dim(&self) -> usize { self.iter.dim() }
}
impl<I: SparseIterator, P> SparseIterator for SparseFilter<I, P>
where
    P: FnMut(&I::Value) -> bool,
{
    type Value = I::Value;
}
impl<I: ShapedSparseIterator, P> ShapedSparseIterator for SparseFilter<I, P>
where
    P: FnMut(&I::Value) -> bool,
{}
impl<I: UniqueSparseIterator, P> UniqueSparseIterator for SparseFilter<I, P>
where
    P: FnMut(&I::Value) -> bool,
{}
impl<I: SortedSparseIterator, P> SortedSparseIterator for SparseFilter<I, P>
where
    P: FnMut(&I::Value) -> bool,
{}

impl<I: Shape, F> Shape for SparseFilterMap<I, F> {
    #[inline] fn dim(&self) -> usize { self.iter.dim() }
}
impl<B, I: SparseIterator, F> SparseIterator for SparseFilterMap<I, F>
where
    F: FnMut(I::Value) -> Option<B>,
{
    type Value = B;
}
impl<B, I: ShapedSparseIterator, F> ShapedSparseIterator for SparseFilterMap<I, F>
where
    F: FnMut(I::Value) -> Option<B>,
{}
impl<B, I: UniqueSparseIterator, F> UniqueSparseIterator for SparseFilterMap<I, F>
where
    F: FnMut(I::Value) -> Option<B>,
{}
impl<B, I: SortedSparseIterator, F> SortedSparseIterator for SparseFilterMap<I, F>
where
    F: FnMut(I::Value) -> Option<B>,
{}

#[test]
fn test_sparse_map() {
    fn f(x: i32) -> i32 {
        2 * x
    }

    // dim = 0
    let s = SparseVec::<i32>::from_parts_strictly_sorted(0, vec![], vec![]);
    let mut iter = s.clone().into_sparse_iter().sparse_map(f);
    assert_eq!(iter.dim(), 0);
    assert_eq!(iter.len(), 0);
    assert_eq!(iter.next(), None);
    assert_eq!(iter.next(), None);

    // nnz = 0
    let s = SparseVec::<i32>::from_parts_strictly_sorted(10, vec![], vec![]);
    let mut iter = s.clone().into_sparse_iter().sparse_map(f);
    assert_eq!(iter.dim(), 10);
    assert_eq!(iter.len(), 0);
    assert_eq!(iter.next(), None);
    assert_eq!(iter.next(), None);

    // forward
    let s = SparseVec::from_parts_strictly_sorted(10, vec![7i32, -4, 5], vec![3, 6, 7]);
    let mut iter = s.clone().into_sparse_iter().sparse_map(f);
    assert_eq!(iter.dim(), 10);
    assert_eq!(iter.len(), 3);
    assert_eq!(iter.next(), Some((3, 14)));
    assert_eq!(iter.next(), Some((6, -8)));
    assert_eq!(iter.next(), Some((7, 10)));
    assert_eq!(iter.next(), None);
    assert_eq!(iter.next(), None);

    // reverse
    let mut iter = s.clone().into_sparse_iter().sparse_map(f).rev();
    assert_eq!(iter.next(), Some((7, 10)));
    assert_eq!(iter.next(), Some((6, -8)));
    assert_eq!(iter.next(), Some((3, 14)));
    assert_eq!(iter.next(), None);
    assert_eq!(iter.next(), None);
}

#[test]
fn test_sparse_filter() {
    fn f(x: &i32) -> bool {
        x > &0
    }

    // dim = 0
    let s = SparseVec::<i32>::from_parts_strictly_sorted(0, vec![], vec![]);
    let mut iter = s.clone().into_sparse_iter().sparse_filter(f);
    assert_eq!(iter.dim(), 0);
    assert_eq!(iter.next(), None);
    assert_eq!(iter.next(), None);

    // nnz = 0
    let s = SparseVec::<i32>::from_parts_strictly_sorted(10, vec![], vec![]);
    let mut iter = s.clone().into_sparse_iter().sparse_filter(f);
    assert_eq!(iter.dim(), 10);
    assert_eq!(iter.next(), None);
    assert_eq!(iter.next(), None);

    // forward
    let s = SparseVec::from_parts_strictly_sorted(10, vec![7i32, -4, 5], vec![3, 6, 7]);
    let mut iter = s.clone().into_sparse_iter().sparse_filter(f);
    assert_eq!(iter.dim(), 10);
    assert_eq!(iter.next(), Some((3, 7)));
    assert_eq!(iter.next(), Some((7, 5)));
    assert_eq!(iter.next(), None);
    assert_eq!(iter.next(), None);

    // reverse
    let mut iter = s.clone().into_sparse_iter().sparse_filter(f).rev();
    assert_eq!(iter.next(), Some((7, 5)));
    assert_eq!(iter.next(), Some((3, 7)));
    assert_eq!(iter.next(), None);
    assert_eq!(iter.next(), None);
}

#[test]
fn test_sparse_filter_map() {
    fn f(x: i32) -> Option<i32> {
        if x > 0 {
            Some(2 * x)
        } else {
            None
        }
    }

    // dim = 0
    let s = SparseVec::<i32>::from_parts_strictly_sorted(0, vec![], vec![]);
    let mut iter = s.clone().into_sparse_iter().sparse_filter_map(f);
    assert_eq!(iter.dim(), 0);
    assert_eq!(iter.next(), None);
    assert_eq!(iter.next(), None);

    // nnz = 0
    let s = SparseVec::<i32>::from_parts_strictly_sorted(10, vec![], vec![]);
    let mut iter = s.clone().into_sparse_iter().sparse_filter_map(f);
    assert_eq!(iter.dim(), 10);
    assert_eq!(iter.next(), None);
    assert_eq!(iter.next(), None);

    // forward
    let s = SparseVec::from_parts_strictly_sorted(10, vec![7i32, -4, 5], vec![3, 6, 7]);
    let mut iter = s.clone().into_sparse_iter().sparse_filter_map(f);
    assert_eq!(iter.dim(), 10);
    assert_eq!(iter.next(), Some((3, 14)));
    assert_eq!(iter.next(), Some((7, 10)));
    assert_eq!(iter.next(), None);
    assert_eq!(iter.next(), None);

    // reverse
    let mut iter = s.clone().into_sparse_iter().sparse_filter_map(f).rev();
    assert_eq!(iter.next(), Some((7, 10)));
    assert_eq!(iter.next(), Some((3, 14)));
    assert_eq!(iter.next(), None);
    assert_eq!(iter.next(), None);
}

//----------------------------------------------------------

#[test]
fn test_densify() {
    // dim = 0
    let s = SparseVec::<f32>::from_parts_strictly_sorted(0, vec![], vec![]);
    let d = vec![];
    assert_eq!(s.into_sparse_iter().densify().collect::<Vec<_>>(), d);

    // nnz = 0
    let s = SparseVec::<f32>::from_parts_strictly_sorted(10, vec![], vec![]);
    let d = vec![0.; 10];
    assert_eq!(s.into_sparse_iter().densify().collect::<Vec<_>>(), d);

    // something with zeros at endpoints
    let s = SparseVec::from_parts_strictly_sorted(10, vec![7f32, 4., 5.], vec![3, 6, 7]);
    let d = vec![0., 0., 0., 7., 0., 0., 4., 5., 0., 0.];
    assert_eq!(s.into_sparse_iter().densify().collect::<Vec<_>>(), d);

    // something with occupied endpoints
    let s = SparseVec::from_parts_strictly_sorted(10, vec![7f32, 4.], vec![0, 9]);
    let d = vec![7., 0., 0., 0., 0., 0., 0., 0., 0., 4.];
    assert_eq!(s.into_sparse_iter().densify().collect::<Vec<_>>(), d);
}

#[test]
fn test_sparse_intersection() {
    macro_rules! sparse_intersection {
        ($a:expr, $b:expr) => {
            $a.into_sparse_iter().sparse_intersection($b)
        };
    }

    // sparse_intersection with an empty vec
    let a = SparseVec::<f32>::from_dense(vec![]);
    let b = SparseVec::<f32>::from_dense(vec![0f32, 1.]);
    assert_eq!(
        sparse_intersection!(a.clone(), a.clone()).collect::<Vec<_>>(),
        vec![]
    );
    assert_eq!(
        sparse_intersection!(a.clone(), b.clone()).collect::<Vec<_>>(),
        vec![]
    );
    assert_eq!(sparse_intersection!(b, a).collect::<Vec<_>>(), vec![]);

    // mismatched entries, and mutually skipped entries
    let a = SparseVec::from_dense(vec![1f32, 0., 0., 1., 0.]);
    let b = SparseVec::from_dense(vec![0f32, 1., 0., 0., 2.]);
    assert_eq!(
        sparse_intersection!(a.clone(), b.clone()).collect::<Vec<_>>(),
        vec![]
    );
    assert_eq!(sparse_intersection!(b, a).collect::<Vec<_>>(), vec![]);

    // consecutive matches. Also, match on first item
    let a = SparseVec::from_dense(vec![1f32, 2., 4.]);
    let b = SparseVec::from_dense(vec![4f32, 5., 6.]);
    let c: Vec<_> = sparse_intersection!(a, b).collect();
    assert_eq!(c, vec![(0, (1., 4.)), (1, (2., 5.)), (2, (4., 6.))]);

    // match after non-matches
    let a = SparseVec::from_dense(vec![0f32, 0., 1., 8., 1.]);
    let b = SparseVec::from_dense(vec![1f32, 0., 0., 9., 0.]);
    let c: Vec<_> = sparse_intersection!(a.clone(), b.clone()).collect();
    let d: Vec<_> = sparse_intersection!(b, a).collect();
    assert_eq!(c, vec![(3, (8., 9.))]);
    assert_eq!(d, vec![(3, (9., 8.))]);
}

// TODO holy crap man where's your test on SparseUnion
