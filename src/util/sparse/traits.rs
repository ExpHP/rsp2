use std::ops::{Range, RangeFrom, RangeFull, RangeTo};

use iter::{ShapedSparseIterator, SortedSparseIterator, UniqueSparseIterator};

/// Trait for objects which represent some sort of "dense" array of fixed size.
///
/// This is used by math operations for performing compatibility checks on their operands.
/// (i.e. to prohibit the addition of vectors of different shape), and it is used by certain
/// iterators to allow shape information to be retained even after such operations are performed.
pub trait Shape {
    /// The length of the "dense" vector represented by the object.
    ///
    /// Many vector math operations (such as pointwise operations, or a dot product) will panic
    /// if the two objects have different `dim`s.
    fn dim(&self) -> usize;
}

pub trait MathableSparseIterator:
    SortedSparseIterator + UniqueSparseIterator + ShapedSparseIterator + Shape
{
}

impl<T> MathableSparseIterator for T
where
    T: SortedSparseIterator + UniqueSparseIterator + ShapedSparseIterator + Shape,
{
}

impl<T> Shape for Vec<T> {
    #[inline]
    fn dim(&self) -> usize {
        self.len()
    }
}

impl<'a, T> Shape for &'a [T] {
    #[inline]
    fn dim(&self) -> usize {
        self.len()
    }
}

/*
pub trait Slice<R>: Shape {
    type Output: Shape;
    fn slice(&self, range: R) -> Output;
}


pub trait VecLike:
    Slice<Range<usize>> + Slice<RangeTo<usize>>
    + Slice<RangeFrom<usize>> + Slice<RangeFull>
    + Shape + PointwiseAdd
 {

}
*/

pub trait Abs {
    fn abs(self) -> Self;
}
impl Abs for i8 { #[inline(always)] fn abs(self) -> i8 { self.abs() } }
impl Abs for i16 { #[inline(always)] fn abs(self) -> i16 { self.abs() } }
impl Abs for i32 { #[inline(always)] fn abs(self) -> i32 { self.abs() } }
impl Abs for i64 { #[inline(always)] fn abs(self) -> i64 { self.abs() } }
impl Abs for isize { #[inline(always)] fn abs(self) -> isize { self.abs() } }
impl Abs for f32 { #[inline(always)] fn abs(self) -> f32 { self.abs() } }
impl Abs for f64 { #[inline(always)] fn abs(self) -> f64 { self.abs() } }
// NOTE: Couldn't quite put my finger on it, but I felt a bit uncomfortable placing this
//        on unsigned types, so I didn't.  We'll see whether there's a legitimate reason
//        to do so in the future.

/// Types from which an object of value type can be extracted by position.
///
/// This provides a shared interface for obtaining items by index from both
/// `Vec<T>` and `&[T]`.  Some sparse types may also implement this, if they
/// can provide a reasonably efficient implementation.
///
/// This will always return values, never references. As a result, most
/// implementors will almost certainly require that `T: Clone`.
pub trait DenseIndex<I> {
    type Value;

    /// Get an element, or `panic!` if out of bounds.
    #[inline(always)]
    fn get_dense(&self, index: I) -> Self::Value;
}

impl<T: Clone> DenseIndex<usize> for Vec<T> {
    type Value = T;
    #[inline(always)]
    fn get_dense(&self, pos: usize) -> T {
        self[pos].clone()
    }
}

impl<'a, T: Clone> DenseIndex<usize> for &'a [T] {
    type Value = T;
    #[inline(always)]
    fn get_dense(&self, pos: usize) -> T {
        self[pos].clone()
    }
}

pub trait GenericRange<I> {
    fn with_defaults(self, start: I, end: I) -> Range<I>;
}

impl<I> GenericRange<I> for Range<I> {
    #[inline(always)]
    fn with_defaults(self, _: I, _: I) -> Range<I> {
        self
    }
}
impl<I> GenericRange<I> for RangeFrom<I> {
    #[inline(always)]
    fn with_defaults(self, _: I, end: I) -> Range<I> {
        Range {
            start: self.start,
            end: end,
        }
    }
}
impl<I> GenericRange<I> for RangeTo<I> {
    #[inline(always)]
    fn with_defaults(self, start: I, _: I) -> Range<I> {
        Range {
            start: start,
            end: self.end,
        }
    }
}
impl<I> GenericRange<I> for RangeFull {
    #[inline(always)]
    fn with_defaults(self, start: I, end: I) -> Range<I> {
        Range {
            start: start,
            end: end,
        }
    }
}

//--------------------------------------------------

/// Provides min/max functions for `PartialOrd` types.
///
/// This is useful in generic contexts between types which are `Ord` and types which are "mostly"
/// `Ord`; that is, with the removal of a few miscreant values (such as `nan` for floating point
/// types), they would form a total order.
///
/// These simply emulate the behavior of `min` and `max`, and produce `None` in the case where
///  `partial_cmp` returns `None`).
pub trait PartialMinMax: PartialOrd<Self> {
    /// Returns the lesser of the two elements, if one can be determined.
    fn partial_min(self: Self, rhs: Self) -> Option<Self>
    where
        Self: Sized,
    {
        use std::cmp::Ordering::*;
        match self.partial_cmp(&rhs) {
            None => None,
            Some(Less) => Some(self),
            Some(Equal) => Some(rhs), // based on what ::std::cmp::min does
            Some(Greater) => Some(rhs),
        }
    }

    /// Returns the greater of the two elements, if one can be determined.
    fn partial_max(self: Self, rhs: Self) -> Option<Self>
    where
        Self: Sized,
    {
        use std::cmp::Ordering::*;
        match self.partial_cmp(&rhs) {
            None => None,
            Some(Less) => Some(rhs),
            Some(Equal) => Some(self), // based on what ::std::cmp::max does
            Some(Greater) => Some(self),
        }
    }
}

impl<T> PartialMinMax for T
where
    T: PartialOrd<T>,
{
}
