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

// File adapted for use by rsp2. Originally from rust-lang/rust,
// where it was similarly dual-licensed.

use std::fmt::{Debug, Display};
use std::iter::{self, FromIterator};
use std::slice;
use std::marker::PhantomData;
use std::hash::Hash;
use std::borrow::{Borrow, BorrowMut};
use std::ops::{Deref, DerefMut, Index, IndexMut, Range, RangeFull, RangeFrom};
use std::fmt;
use std::vec;
use std::mem;

/// Represents some newtyped `usize` wrapper.
///
/// (purpose: avoid mixing indexes for different bitvector domains.)
///
/// # Safety
///
/// `unsafe` code in other crates is allowed to trust a number of properties
/// of index types: (these properties are provided by the macro-generated impls)
///
/// * `new` and `index` are identical in behavior to `mem::transmute`.
/// * Methods of `Clone`, `PartialEq`, `Eq`, `PartialOrd`, `Ord`, and `Hash` behave
///   identically to how they would for usizes.
/// * `Debug` impls do not panic.
pub unsafe trait Idx: Copy + 'static + Eq + Debug + Display + Ord + Hash + Send + Sync {
    fn new(idx: usize) -> Self;
    fn index(self) -> usize;
    #[inline(always)]
    fn next(self) -> Self { Self::new(self.index() + 1) }
}

unsafe impl Idx for usize {
    #[inline]
    fn new(idx: usize) -> Self { idx }
    #[inline]
    fn index(self) -> usize { self }
}

#[macro_export]
macro_rules! newtype_index {
    ( $( #[ $($attr:tt)+ ] )* $type:ident) => (

        $( #[ $($attr)+ ] )*
        // NOTE: unsafe code relies on these being derived in many places.
        #[derive(Copy, Clone)]
        // NOTE: unsafe code relies on these being derived for safely transmuting
        //       BTreeMaps and HashMaps.
        #[derive(PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub struct $type(pub usize);

        unsafe impl $crate::Idx for $type {
            #[inline(always)]
            fn new(value: usize) -> Self {
                $type(value)
            }

            #[inline(always)]
            fn index(self) -> usize {
                self.0
            }
        }

        impl ::std::fmt::Display for $type {
            fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
                ::std::fmt::Display::fmt(&self.0, f)
            }
        }

        impl ::std::fmt::Debug for $type {
            fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
                ::std::fmt::Debug::fmt(&self.0, f)
            }
        }
    );
}

/// A Vec or slice that uses newtype indices.
///
/// `V` is only ever `[T]` or `Vec<T>`.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Indexed<I: Idx, V: ?Sized> {
    _marker: PhantomData<fn(&I)>,
    pub raw: V,
}

pub type IndexVec<I, T> = Indexed<I, Vec<T>>;

// Whether `Indexed` is `Send` depends only on the data,
// not the phantom data.
unsafe impl<I: Idx, V> Send for Indexed<I, V> where V: Send {}

impl<I: Idx, V: ?Sized + fmt::Debug> fmt::Debug for Indexed<I, V> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&self.raw, fmt)
    }
}

#[inline]
pub fn iota<I: Idx>(start: I) -> std::iter::Map<RangeFrom<usize>, impl Fn(usize) -> I> {
    (start.index()..).map(I::new)
}

// NOTE: I don't want this to take arbitrary RangeBounds because it would either have
//       to use dynamic polymorphism, or panic on RangeFrom.
#[inline]
pub fn range<I: Idx>(range: Range<I>) -> std::iter::Map<Range<usize>, impl Fn(usize) -> I> {
    (range.start.index()..range.end.index()).map(I::new)
}

/// # Construction
impl<I: Idx, T> IndexVec<I, T> {
    #[inline]
    pub fn from_raw(raw: Vec<T>) -> Self {
        IndexVec { raw, _marker: PhantomData }
    }

    #[inline]
    pub fn new() -> Self {
        IndexVec::from_raw(Vec::new())
    }

    #[inline]
    pub fn from_elem<S>(elem: T, universe: &Indexed<I, [S]>) -> Self
    where T: Clone,
    {
        IndexVec::from_raw(vec![elem; universe.len()])
    }

    #[inline]
    pub fn from_elem_n(elem: T, n: usize) -> Self
    where T: Clone,
    {
        IndexVec::from_raw(vec![elem; n])
    }

    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        IndexVec::from_raw(Vec::with_capacity(capacity))
    }
}

/// # Construction
impl<I: Idx, T> Indexed<I, [T]> {
    #[inline]
    pub fn from_raw_ref(raw: &[T]) -> &Self {
        evert_ref(Indexed { raw, _marker: PhantomData })
    }

    #[inline]
    pub fn from_raw_mut(raw: &mut [T]) -> &mut Self {
        evert_mut(Indexed { raw, _marker: PhantomData })
    }
}

/// # Length-changing operations
impl<I: Idx, T> Indexed<I, Vec<T>> {
    #[inline]
    pub fn push(&mut self, d: T) -> I {
        let idx = I::new(self.len());
        self.raw.push(d);
        idx
    }

    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        self.raw.pop()
    }

    #[inline]
    pub fn resize(&mut self, new_len: usize, value: T)
    where T: Clone,
    {
        self.raw.resize(new_len, value)
    }

    #[inline]
    pub fn shrink_to_fit(&mut self) {
        self.raw.shrink_to_fit()
    }

    #[inline]
    pub fn truncate(&mut self, a: usize) {
        self.raw.truncate(a)
    }
}

/// # Simple access
impl<I: Idx, T> Indexed<I, [T]> {
    #[inline]
    pub fn len(&self) -> usize {
        self.raw.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.raw.is_empty()
    }

    #[inline]
    pub fn last(&self) -> Option<I> {
        self.len().checked_sub(1).map(I::new)
    }

    #[inline]
    pub fn binary_search(&self, value: &T) -> Result<I, I>
        where T: Ord,
    {
        match self.raw.binary_search(value) {
            Ok(i) => Ok(Idx::new(i)),
            Err(i) => Err(Idx::new(i)),
        }
    }

    #[inline]
    pub fn get(&self, index: I) -> Option<&T> {
        self.raw.get(index.index())
    }

    #[inline]
    pub fn get_mut(&mut self, index: I) -> Option<&mut T> {
        self.raw.get_mut(index.index())
    }

    /// Return mutable references to two distinct elements, a and b. Panics if a == b.
    #[inline]
    pub fn pick2_mut(&mut self, a: I, b: I) -> (&mut T, &mut T) {
        let (ai, bi) = (a.index(), b.index());
        assert_ne!(ai, bi);

        if ai < bi {
            let (c1, c2) = self.raw.split_at_mut(bi);
            (&mut c1[ai], &mut c2[0])
        } else {
            let (c2, c1) = self.pick2_mut(b, a);
            (c1, c2)
        }
    }
}

/// # By-value iteration
impl<I: Idx, T> Indexed<I, Vec<T>> {
    #[inline]
    pub fn into_iter(self) -> vec::IntoIter<T> {
        self.raw.into_iter()
    }

    /// Iterate with indices.
    ///
    /// # Panics
    ///
    /// It is guaranteed that calling `next` on this this iterator **will not panic.**
    #[inline]
    pub fn into_iter_enumerated(self)
        -> iter::Map<
            iter::Enumerate<vec::IntoIter<T>>,
            impl FnMut((usize, T)) -> (I, T),
        >
    {
        self.raw.into_iter().enumerate()
            .map(|(i, x)| (I::new(i), x))
    }
}

/// # Non-consuming iteration
impl<I: Idx, T> Indexed<I, [T]> {
    #[inline]
    pub fn iter(&self) -> slice::Iter<T> {
        self.raw.iter()
    }

    /// Iterate with indices.
    ///
    /// # Panics
    ///
    /// It is guaranteed that calling `next` on this this iterator **will not panic.**
    #[inline]
    pub fn iter_enumerated<'a>(&'a self)
        -> iter::Map<
            iter::Enumerate<slice::Iter<'a, T>>,
            impl FnMut((usize, &'a T)) -> (I, &'a T),
        >
    {
        self.raw.iter().enumerate()
            .map(|(i, x)| (I::new(i), x))
    }

    /// Iterate over the indices.
    ///
    /// # Panics
    ///
    /// It is guaranteed that calling `next` on this this iterator **will not panic.**
    #[inline]
    pub fn indices(&self) -> iter::Map<Range<usize>, impl FnMut(usize) -> I> {
        (0..self.len()).map(I::new)
    }

    #[inline]
    pub fn iter_mut(&mut self) -> slice::IterMut<T> {
        self.raw.iter_mut()
    }

    /// Iterate with indices.
    ///
    /// # Panics
    ///
    /// It is guaranteed that calling `next` on this this iterator **will not panic.**
    #[inline]
    pub fn iter_enumerated_mut<'a>(&'a mut self)
        -> iter::Map<
            iter::Enumerate<slice::IterMut<'a, T>>,
            impl FnMut((usize, &'a mut T)) -> (I, &'a mut T),
        >
    {
        self.raw.iter_mut().enumerate()
            .map(|(i, x)| (I::new(i), x))
    }

//    #[inline]
//    pub fn drain<'a, R: RangeBounds<usize>>(
//        &'a mut self,
//        range: R,
//    ) -> impl Iterator<Item=T> + 'a {
//        self.raw.drain(range)
//    }
//
//    #[inline]
//    pub fn drain_enumerated<'a, R: RangeBounds<usize>>(
//        &'a mut self,
//        range: R,
//    ) -> impl Iterator<Item=(I, T)> + 'a {
//        self.raw.drain(range).enumerate()
//            .map(|(i, x)| (I::new(i), x))
//    }
}

//--------------------------------------------------------

/// # Conversion
impl<I: Idx, T> Indexed<I, Vec<T>> {
    #[inline(always)]
    pub fn into_index_type<Ix: Idx>(self) -> Indexed<Ix, Vec<T>> {
        Indexed::from_raw(self.raw)
    }
}

/// # Conversion
impl<I: Idx, T> Indexed<I, [T]> {
    #[inline(always)]
    pub fn as_index_type<Ix: Idx>(&self) -> &Indexed<Ix, [T]> {
        Indexed::from_raw_ref(&self.raw)
    }

    #[inline(always)]
    pub fn as_index_type_mut<Ix: Idx>(&mut self) -> &mut Indexed<Ix, [T]> {
        Indexed::from_raw_mut(&mut self.raw)
    }
}

//--------------------------------------------------------

impl<I: Idx, V: Deref<Target=[T]>, T> Deref for Indexed<I, V> {
    type Target = Indexed<I, [T]>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        Indexed::from_raw_ref(self.raw.deref())
    }
}

impl<I: Idx, V: DerefMut<Target=[T]>, T> DerefMut for Indexed<I, V> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        Indexed::from_raw_mut(self.raw.deref_mut())
    }
}

//--------------------------------------------------------

impl<I: Idx, T> Index<I> for Indexed<I, [T]> {
    type Output = T;

    #[inline]
    fn index(&self, index: I) -> &T {
        &self.raw[index.index()]
    }
}

impl<I: Idx, T> IndexMut<I> for Indexed<I, [T]> {
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut T {
        &mut self.raw[index.index()]
    }
}

impl<I: Idx, T> Index<RangeFull> for Indexed<I, [T]> {
    type Output = Self;

    #[inline]
    fn index(&self, _: RangeFull) -> &Indexed<I, [T]> {
        Indexed::from_raw_ref(&self.raw[..])
    }
}

impl<I: Idx, T> IndexMut<RangeFull> for Indexed<I, [T]> {
    #[inline]
    fn index_mut(&mut self, _: RangeFull) -> &mut Indexed<I, [T]> {
        Indexed::from_raw_mut(&mut self.raw[..])
    }
}

//--------------------------------------------------------

impl<I: Idx, T> Default for Indexed<I, Vec<T>> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<I: Idx, T> Extend<T> for Indexed<I, Vec<T>> {
    #[inline]
    fn extend<J: IntoIterator<Item = T>>(&mut self, iter: J) {
        self.raw.extend(iter);
    }
}

impl<I: Idx, T> FromIterator<T> for Indexed<I, Vec<T>> {
    #[inline]
    fn from_iter<J>(iter: J) -> Self where J: IntoIterator<Item=T> {
        Indexed::from_raw(FromIterator::from_iter(iter))
    }
}

//--------------------------------------------------------

impl<I: Idx, T> IntoIterator for Indexed<I, Vec<T>> {
    type Item = T;
    type IntoIter = vec::IntoIter<T>;

    #[inline]
    fn into_iter(self) -> vec::IntoIter<T> {
        self.raw.into_iter()
    }
}

impl<'a, I: Idx, T: 'a> IntoIterator for &'a Indexed<I, [T]> {
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;

    #[inline]
    fn into_iter(self) -> slice::Iter<'a, T> {
        self.raw.iter()
    }
}

impl<'a, I: Idx, T: 'a> IntoIterator for &'a Indexed<I, Vec<T>> {
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;

    #[inline]
    fn into_iter(self) -> slice::Iter<'a, T> {
        self.raw.iter()
    }
}

impl<'a, I: Idx, T: 'a> IntoIterator for &'a mut Indexed<I, [T]> {
    type Item = &'a mut T;
    type IntoIter = slice::IterMut<'a, T>;

    #[inline]
    fn into_iter(self) -> slice::IterMut<'a, T> {
        self.raw.iter_mut()
    }
}

impl<'a, I: Idx, T: 'a> IntoIterator for &'a mut Indexed<I, Vec<T>> {
    type Item = &'a mut T;
    type IntoIter = slice::IterMut<'a, T>;

    #[inline]
    fn into_iter(self) -> slice::IterMut<'a, T> {
        self.raw.iter_mut()
    }
}

//--------------------------------------------------------

impl<I: Idx, T> Borrow<Indexed<I, [T]>> for Indexed<I, Vec<T>> {
    #[inline]
    fn borrow(&self) -> &Indexed<I, [T]> { self }
}

impl<I: Idx, T> BorrowMut<Indexed<I, [T]>> for Indexed<I, Vec<T>> {
    #[inline]
    fn borrow_mut(&mut self) -> &mut Indexed<I, [T]> { self }
}

impl<I: Idx, T: Clone> ToOwned for Indexed<I, [T]> {
    type Owned = Indexed<I, Vec<T>>;

    #[inline]
    fn to_owned(&self) -> Self::Owned { IndexVec::from_raw(self.raw.to_owned()) }
}

//--------------------------------------------------------

// unsafe transformations

#[inline]
fn evert_ref<'a, I: Idx, V: ?Sized>(it: Indexed<I, &'a V>) -> &'a Indexed<I, V> {
    unsafe { mem::transmute(it) }
}

#[inline]
fn evert_mut<'a, I: Idx, V: ?Sized>(it: Indexed<I, &'a mut V>) -> &'a mut Indexed<I, V> {
    unsafe { mem::transmute(it) }
}
