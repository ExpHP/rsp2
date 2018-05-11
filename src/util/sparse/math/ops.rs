use std::ops::{Add, Mul, Neg, Sub};

use iter::IntoSparseIterator;
use iter::ShapedSparseIterator;
use iter::SortedSparseIterator;
use iter::SparseIterator;
use iter::UnionValue;
use iter::UniqueSparseIterator;
use traits::DenseIndex;
use traits::MathableSparseIterator;
use traits::Shape;

// I really really wanted to have general traits for pointwise ops which work
//  regardless of sparse/dense, but I don't think there is really any hope of
//  doing so without trait specialization or negative bounds.

//-------------------------------------------------------
// Math traits with no type parameters or associated types
//
// I don't think it's possible to have macros which are generic over trait bounds of an arbitrary
// form. The following traits can be used as bounds in a macro if you match them as `ident`.

/// A type which can be added to itself.
///
/// This trait exists to overcome an apparent limitation of the macro system with respect to
/// trait bounds, as macros are used to implement the sparse math iterator adapters.
pub trait SelfAdd: Sized + Add<Self, Output = Self> {}

/// A type which can be multiplied with itself.
///
/// This trait exists to overcome an apparent limitation of the macro system with respect to
/// trait bounds, as macros are used to implement the sparse math iterator adapters.
pub trait SelfMul: Sized + Mul<Self, Output = Self> {}

/// A type which can be subtracted from itself and negated.
///
/// This trait exists to overcome an apparent limitation of the macro system with respect to
/// trait bounds, as macros are used to implement the sparse math iterator adapters.
pub trait SelfSub: Sized + Sub<Self, Output = Self> + Neg<Output = Self> {}

impl<T: Add<T, Output = T>> SelfAdd for T {}
impl<T: Mul<T, Output = T>> SelfMul for T {}
impl<T: Sub<T, Output = T> + Neg<Output = T>> SelfSub for T {}

//-------------------------------------------------------
// sparse-sparse operations

// adapter types

/// Implementation detail of `sparse_sparse_add`
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
pub struct SparseSparseAdd<A: Iterator, B: Iterator> {
    iter: ::iter::Shaped<::iter::SparseUnion<A, B>>,
}

/// Implementation detail of `sparse_sparse_sub`
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
pub struct SparseSparseSub<A: Iterator, B: Iterator> {
    iter: ::iter::Shaped<::iter::SparseUnion<A, B>>,
}

/// Implementation detail of `sparse_sparse_mul`
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
pub struct SparseSparseMul<A, B> {
    iter: ::iter::Shaped<::iter::SparseIntersection<A, B>>,
}

// impls for these guys are so similar that we can abstract out the differences with a macro
macro_rules! sparse_sparse_impls {
    ($Result:ident, $TBound:ident, $zip_method:ident, $value_func:expr) => {
        impl<A, B> $Result<A, B>
        where
            A: MathableSparseIterator,
            B: MathableSparseIterator<Value = A::Value>,
            A::Value: $TBound,
        {
            #[inline]
            pub fn new(a: A, b: B) -> Self
            where
                A: Sized,
                B: Sized,
            {
                assert!(
                    a.dim() == b.dim(),
                    "Dimension mismatch: {} vs {}",
                    a.dim(),
                    b.dim()
                );
                let dim = a.dim();
                $Result {
                    iter: a.$zip_method(b).as_shaped(dim),
                }
            }
        }

        impl<A, B> Iterator for $Result<A, B>
        where
            A: MathableSparseIterator,
            B: MathableSparseIterator<Value = A::Value>,
            A::Value: $TBound,
        {
            type Item = (usize, A::Value);
            #[inline]
            fn next(&mut self) -> Option<Self::Item> {
                self.iter
                    .next()
                    .and_then(|(i, x)| Some((i, $value_func(x))))
            }
            #[inline]
            fn size_hint(&self) -> (usize, Option<usize>) {
                self.iter.size_hint()
            }
        }

        impl<A, B> Shape for $Result<A, B>
        where
            A: MathableSparseIterator,
            B: MathableSparseIterator,
        {
            #[inline]
            fn dim(&self) -> usize {
                self.iter.dim()
            }
        }

        impl<A, B> SparseIterator for $Result<A, B>
        where
            A: MathableSparseIterator,
            B: MathableSparseIterator<Value = A::Value>,
            A::Value: $TBound,
        {
            type Value = A::Value;
        }

        impl<A, B> ShapedSparseIterator for $Result<A, B>
        where
            A: MathableSparseIterator,
            B: MathableSparseIterator<Value = A::Value>,
            A::Value: $TBound,
        {
        }

        impl<A, B> SortedSparseIterator for $Result<A, B>
        where
            A: MathableSparseIterator,
            B: MathableSparseIterator<Value = A::Value>,
            A::Value: $TBound,
        {
        }

        impl<A, B> UniqueSparseIterator for $Result<A, B>
        where
            A: MathableSparseIterator,
            B: MathableSparseIterator<Value = A::Value>,
            A::Value: $TBound,
        {
        }
    };
}

sparse_sparse_impls!(SparseSparseMul, SelfMul, sparse_intersection, |(x, y)| x * y);

sparse_sparse_impls!(SparseSparseAdd, SelfAdd, sparse_union, |val| match val {
    UnionValue::Left(x) | UnionValue::Right(x) => x,
    UnionValue::Both(x, y) => x + y,
});

// For some reason I get this if I define this one as a closure inside the macro:
//    math.rs:217:28: 217:29 error: the type of this value must be known in this context
//    math.rs:217          UnionValue::Right(y) => -y,
//                                                 ^~
#[inline]
fn do_sub<T>(val: UnionValue<T, T>) -> T
where
    T: Sub<T, Output = T> + Neg<Output = T>,
{
    match val {
        UnionValue::Left(x) => x,
        UnionValue::Right(y) => -y,
        UnionValue::Both(x, y) => x - y,
    }
}
sparse_sparse_impls!(SparseSparseSub, SelfSub, sparse_union, do_sub);

/// Pointwise math between two sparse iterators.
///
/// Currently, all ergonomics have been thrown out the window.  There is no way
/// to be generic over this and other "pointwise" math traits.  Furthermore, both
/// operands need to be sparse iterators (not containers) of value type `T` (not
/// `&T`).
///
/// # Examples
///
/// ```
/// use sparse::{SparseVec,SparseSparseMath,IntoSparseIterator};
///
/// let a = SparseVec::from_dense(vec![0i64, 0, 1, 2]).into_sparse_iter();
/// let b = SparseVec::from_dense(vec![0i64, 4, 3, 0]).into_sparse_iter();
/// let c = SparseVec::from_dense(vec![0i64, 0, 0, 1]).into_sparse_iter();
/// let mut it = a.sparse_sparse_mul(b).sparse_sparse_add(c);
///
/// // a*b + c == [0, 0, 3, 1]
/// assert_eq!(it.next(), Some((2usize, 3)));
/// assert_eq!(it.next(), Some((3usize, 1)));
/// assert_eq!(it.next(), None);
/// ```
pub trait SparseSparseMath<RHS>
where
    Self: MathableSparseIterator + Sized,
    RHS: IntoSparseIterator<Value = Self::Value>,
    RHS::IntoSparseIter: MathableSparseIterator<Value = Self::Value>,
{
    fn sparse_sparse_mul(self, rhs: RHS) -> SparseSparseMul<Self, RHS::IntoSparseIter>
    where
        Self::Value: SelfMul,
    {
        SparseSparseMul::new(self, rhs.into_sparse_iter())
    }
    fn sparse_sparse_add(self, rhs: RHS) -> SparseSparseAdd<Self, RHS::IntoSparseIter>
    where
        Self::Value: SelfAdd,
    {
        SparseSparseAdd::new(self, rhs.into_sparse_iter())
    }
    fn sparse_sparse_sub(self, rhs: RHS) -> SparseSparseSub<Self, RHS::IntoSparseIter>
    where
        Self::Value: SelfSub,
    {
        SparseSparseSub::new(self, rhs.into_sparse_iter())
    }
}

impl<A, B> SparseSparseMath<B> for A
where
    A: MathableSparseIterator,
    B: IntoSparseIterator<Value = A::Value>,
    B::IntoSparseIter: MathableSparseIterator<Value = A::Value>,
{
}

//-------------------------------------------------------
// sparse-dense operations with a sparse result

// adapter types

/// Implementation detail of `sparse_dense_mul`
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
pub struct SparseDenseMul<A, B> {
    sparse: A,
    dense: B,
}

macro_rules! sparse_dense_impls {
    ($Result:ident, $TBound:ident, $value_func:expr) => {
        // constructor

        impl<A, B> $Result<A, B>
        where
            A: MathableSparseIterator,
            B: DenseIndex<usize, Value = A::Value> + Shape,
            A::Value: $TBound,
        {
            #[inline]
            pub fn new(a: A, b: B) -> Self
            where
                A: Sized,
                B: Sized,
            {
                assert!(
                    a.dim() == b.dim(),
                    "Dimension mismatch: {} vs {}",
                    a.dim(),
                    b.dim()
                );
                $Result {
                    sparse: a,
                    dense: b,
                }
            }
        }

        impl<A, B> Iterator for $Result<A, B>
        where
            A: MathableSparseIterator,
            B: DenseIndex<usize, Value = A::Value> + Shape,
            A::Value: $TBound + Clone, // FIXME remove Clone bound
        {
            type Item = (usize, A::Value);
            #[inline]
            fn next(&mut self) -> Option<Self::Item> {
                // FIXME: This *always* clones the dense value even if `dense` is an owned `vec`.
                //  I'd like to avoid this if possible.
                self.sparse
                    .next()
                    .and_then(|(i, x)| Some((i, $value_func(x, self.dense.get_dense(i)))))
            }
            #[inline]
            fn size_hint(&self) -> (usize, Option<usize>) {
                self.sparse.size_hint()
            }
        }

        impl<A, B> DoubleEndedIterator for $Result<A, B>
        where
            A: MathableSparseIterator + DoubleEndedIterator,
            B: DenseIndex<usize, Value = A::Value> + Shape,
            A::Value: $TBound + Clone, // FIXME remove Clone bound
        {
            fn next_back(&mut self) -> Option<Self::Item> {
                self.sparse
                    .next_back()
                    .and_then(|(i, x)| Some((i, $value_func(x, self.dense.get_dense(i)))))
            }
        }

        // trivial impls

        impl<A, B> ExactSizeIterator for $Result<A, B>
        where
            A: MathableSparseIterator + ExactSizeIterator,
            B: DenseIndex<usize, Value = A::Value> + Shape,
            A::Value: $TBound + Clone, // FIXME remove Clone bound
        {
        }

        impl<A, B> Shape for $Result<A, B>
        where
            A: MathableSparseIterator,
            B: DenseIndex<usize, Value = A::Value> + Shape,
        {
            #[inline]
            fn dim(&self) -> usize {
                self.sparse.dim()
            }
        }

        impl<A, B> SparseIterator for $Result<A, B>
        where
            A: MathableSparseIterator,
            B: DenseIndex<usize, Value = A::Value> + Shape,
            A::Value: $TBound + Clone, // FIXME remove Clone bound
        {
            type Value = A::Value;
        }

        impl<A, B> ShapedSparseIterator for $Result<A, B>
        where
            A: MathableSparseIterator,
            B: DenseIndex<usize, Value = A::Value> + Shape,
            A::Value: $TBound + Clone, // FIXME remove Clone bound
        {
        }

        impl<A, B> SortedSparseIterator for $Result<A, B>
        where
            A: MathableSparseIterator,
            B: DenseIndex<usize, Value = A::Value> + Shape,
            A::Value: $TBound + Clone, // FIXME remove Clone bound
        {
        }

        impl<A, B> UniqueSparseIterator for $Result<A, B>
        where
            A: MathableSparseIterator,
            B: DenseIndex<usize, Value = A::Value> + Shape,
            A::Value: $TBound + Clone, // FIXME remove Clone bound
        {
        }
    };
}

sparse_dense_impls!(SparseDenseMul, SelfMul, |x, y| x * y);

/// Pointwise math between a sparse iterator and dense object.
///
/// Currently, all ergonomics have been thrown out the window.  There is no way
/// to be generic over this and other "pointwise" math traits.  `Self` must be
/// a sparse iterator (not container) of value type `T` (not `&T`).
///
/// # Examples
///
/// ```
/// use sparse::{SparseVec,SparseDenseMath,IntoSparseIterator};
///
/// // let a = SparseVec::from_dense(vec![0i64, 0, 1, 2]).into_sparse_iter();
/// // let b = vec![0i64, 4, 3, 0];
/// // let mut it = a.sparse_dense_mul(b);
/// let mut a = SparseVec::from_dense(vec![0i64, 0, 1, 2]).into_sparse_iter()
///     .sparse_dense_mul(vec![0, 4, 3, 0]);
///
/// assert_eq!(a.next(), Some((2usize, 3)));
/// assert_eq!(a.next(), Some((3usize, 0)));
/// assert_eq!(a.next(), None);
/// ```
pub trait SparseDenseMath<RHS>
where
    Self: MathableSparseIterator + Sized,
    RHS: DenseIndex<usize, Value = Self::Value> + Shape,
{
    /// Multiply a sparse iterator and dense object.
    ///
    /// The output type is sparse.
    fn sparse_dense_mul(self, rhs: RHS) -> SparseDenseMul<Self, RHS>
    where
        Self::Value: SelfMul,
    {
        SparseDenseMul::new(self, rhs)
    }

    // I have no interest (yet) in implementing add or sub, which would
    // have to produce objects with primarily dense semantics.
}

impl<A, B> SparseDenseMath<B> for A
where
    A: MathableSparseIterator,
    B: DenseIndex<usize, Value = A::Value> + Shape,
{
}

//-----------------------------------------------------------

#[cfg(test)]
mod tests {
    use iter::IntoSparseIterator;
    use iter::ShapedSparseIterator;
    use iter::SparseIterator;
    use math::SparseDenseMath;
    use math::SparseSparseMath;
    use vec::SparseVec;
    use test::black_box;

    #[test]
    fn test_ss_with_sparse_vec() {
        // testing usage with sparse vec argument
        let a = SparseVec::from_dense(vec![1i32, 2, 3])
            .into_sparse_iter()
            .sparse_sparse_mul(SparseVec::from_dense(vec![0, 2, 0]))
            .sparse_sparse_add(SparseVec::from_dense(vec![1, 0, 0]))
            .sparse_sparse_sub(SparseVec::from_dense(vec![2, 0, 0]));
        let a = SparseVec::from_sparse_iter(a);
        assert_eq!(a.into_dense(), vec![-1, 4, 0]);
    }

    #[test]
    fn test_ss_mul() {
        // Dense product
        let a = SparseVec::from_dense(vec![1u32, 2, 3]);
        let b = SparseVec::from_dense(vec![3u32, 5, 7]);
        let c = a.into_sparse_iter().sparse_sparse_mul(b.into_sparse_iter());
        assert_eq!(c.densify().collect::<Vec<_>>(), vec![3u32, 10, 21]);

        // Alternating zeros
        let a = SparseVec::from_dense(vec![0u32, 7, 0, 13]);
        let b = SparseVec::from_dense(vec![3u32, 0, 11, 0]);
        let c = a.into_sparse_iter().sparse_sparse_mul(b.into_sparse_iter());
        assert_eq!(c.densify().collect::<Vec<_>>(), vec![0, 0, 0, 0]);

        // A zero operand
        let a = SparseVec::from_dense(vec![1u32, 2, 3]);
        let b = SparseVec::from_dense(vec![0u32, 0, 0]);
        let c = a.clone()
            .into_sparse_iter()
            .sparse_sparse_mul(b.clone().into_sparse_iter());
        let d = b.into_sparse_iter().sparse_sparse_mul(a.into_sparse_iter());
        assert_eq!(c.densify().collect::<Vec<_>>(), vec![0, 0, 0]);
        assert_eq!(d.densify().collect::<Vec<_>>(), vec![0, 0, 0]);

        // chained operation
        let a = SparseVec::from_dense(vec![1u32, 2, 3]);
        let a = a.clone()
            .into_sparse_iter()
            .sparse_sparse_mul(a.clone().into_sparse_iter())
            .sparse_sparse_mul(a.into_sparse_iter());
        assert_eq!(a.densify().collect::<Vec<_>>(), vec![1, 8, 27]);
    }

    #[test]
    fn test_ss_add() {
        // Dense product
        let a = SparseVec::from_dense(vec![1u32, 2, 3]);
        let b = SparseVec::from_dense(vec![3u32, 5, 7]);
        let c = a.into_sparse_iter().sparse_sparse_add(b.into_sparse_iter());
        assert_eq!(c.densify().collect::<Vec<_>>(), vec![4u32, 7, 10]);

        // Alternating zeros
        let a = SparseVec::from_dense(vec![0u32, 7, 0, 13]);
        let b = SparseVec::from_dense(vec![3u32, 0, 11, 0]);
        let c = a.into_sparse_iter().sparse_sparse_add(b.into_sparse_iter());
        assert_eq!(c.densify().collect::<Vec<_>>(), vec![3, 7, 11, 13]);

        // A zero operand
        let a = SparseVec::from_dense(vec![1u32, 2, 3]);
        let b = SparseVec::from_dense(vec![0u32, 0, 0]);
        let c = SparseVec::from_sparse_iter(
            a.clone()
                .into_sparse_iter()
                .sparse_sparse_add(b.clone().into_sparse_iter()),
        );
        let d = SparseVec::from_sparse_iter(
            b.into_sparse_iter()
                .sparse_sparse_add(a.clone().into_sparse_iter()),
        );
        assert_eq!(c, a);
        assert_eq!(d, a);
    }

    #[test]
    fn test_ss_sub() {
        // Dense product
        let a = SparseVec::from_dense(vec![1i32, 2, 3]);
        let b = SparseVec::from_dense(vec![3i32, 5, 7]);
        let c = a.into_sparse_iter().sparse_sparse_sub(b.into_sparse_iter());
        assert_eq!(c.densify().collect::<Vec<_>>(), vec![-2, -3, -4]);

        // Alternating zeros
        let a = SparseVec::from_dense(vec![0i32, 7, 0, 13]);
        let b = SparseVec::from_dense(vec![3i32, 0, 11, 0]);
        let c = a.into_sparse_iter().sparse_sparse_sub(b.into_sparse_iter());
        assert_eq!(c.densify().collect::<Vec<_>>(), vec![-3, 7, -11, 13]);

        // A zero operand
        let a = SparseVec::from_dense(vec![1i32, 2, 3]);
        let b = SparseVec::from_dense(vec![0i32, 0, 0]);
        let c = SparseVec::from_sparse_iter(
            a.clone()
                .into_sparse_iter()
                .sparse_sparse_sub(b.clone().into_sparse_iter()),
        );
        let d = SparseVec::from_sparse_iter(
            b.into_sparse_iter()
                .sparse_sparse_sub(a.clone().into_sparse_iter()),
        );
        assert_eq!(c, a);
        assert_eq!(
            d,
            SparseVec::from_sparse_iter(a.into_sparse_iter().sparse_map(|x| -x))
        );
    }

    #[test]
    fn test_sd_mul() {
        // Basic test (where sparse vec is "dense")
        let mut a = SparseVec::from_dense(vec![1u32, 2, 3])
            .into_sparse_iter()
            .sparse_dense_mul(vec![3, 5, 7]);
        assert_eq!(a.next(), Some((0, 3)));
        assert_eq!(a.next(), Some((1, 10)));
        assert_eq!(a.next(), Some((2, 21)));
        assert_eq!(a.next(), None);
        assert_eq!(a.next(), None);

        // Check `ExactSizeIterator`
        let mut a = SparseVec::from_dense(vec![0u32, 7, 0, 2])
            .into_sparse_iter()
            .sparse_dense_mul(vec![3u32, 2, 11, 6]);
        assert_eq!(a.len(), 2);
        // Check `DoubleEndedIterator`
        assert_eq!(a.next_back(), Some((3, 12)));
        assert_eq!(a.next_back(), Some((1, 14)));
        assert_eq!(a.next_back(), None);
        assert_eq!(a.next(), None);

        // A zero sparse operand
        let mut a = SparseVec::from_dense(vec![0u32, 0, 0])
            .into_sparse_iter()
            .sparse_dense_mul(vec![1u32, 2, 3]);
        assert_eq!(a.next(), None);

        // chained operation
        let a = SparseVec::from_dense(vec![1u32, 2, 3])
            .into_sparse_iter()
            .sparse_dense_mul(vec![0, 2, 3])
            .sparse_dense_mul(vec![0, 2, 0]);
        assert_eq!(a.densify().collect::<Vec<_>>(), vec![0, 8, 0]);
    }

    #[should_panic(expected = "Dimension mismatch")]
    #[test]
    fn test_ss_mul_shape_mismatch() {
        let a = SparseVec::from_dense(vec![1u32, 2, 3]);
        let b = SparseVec::from_dense(vec![3u32, 5]);
        let c = a.into_sparse_iter().sparse_sparse_mul(b.into_sparse_iter());
        let c = SparseVec::from_sparse_iter(c);
        black_box(c);
    }

    #[should_panic(expected = "Dimension mismatch")]
    #[test]
    fn test_ss_add_shape_mismatch() {
        let a = SparseVec::from_dense(vec![1u32, 2, 3]);
        let b = SparseVec::from_dense(vec![3u32, 5]);
        let c = a.into_sparse_iter().sparse_sparse_add(b.into_sparse_iter());
        let c = SparseVec::from_sparse_iter(c);
        black_box(c);
    }

    #[should_panic(expected = "Dimension mismatch")]
    #[test]
    fn test_ss_sub_shape_mismatch() {
        let a = SparseVec::from_dense(vec![1i32, 2, 3]);
        let b = SparseVec::from_dense(vec![3i32, 5]);
        let c = a.into_sparse_iter().sparse_sparse_sub(b.into_sparse_iter());
        let c = SparseVec::from_sparse_iter(c);
        black_box(c);
    }

    #[should_panic(expected = "Dimension mismatch")]
    #[test]
    fn test_sd_mul_shape_mismatch() {
        let a = SparseVec::from_dense(vec![1i32, 2, 3])
            .into_sparse_iter()
            .sparse_dense_mul(vec![3i32, 5]);
        let a = SparseVec::from_sparse_iter(a);
        black_box(a);
    }

}
