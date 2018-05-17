//
// File adapted for use by rsp2. Originally from rust-lang/rust.
//
// -----------------------------------------------------------------
// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::fmt::Debug;
use std::iter::{self, FromIterator};
use std::slice;
use std::marker::PhantomData;
use std::hash::Hash;
use std::borrow::{Borrow, BorrowMut};
use std::ops::{Deref, DerefMut, Index, IndexMut, Range, RangeFull};
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
/// * `new` and `index` are identical in behavior to `mem::transmute`, and
///   it is safe to transmute a container of one type into another.
/// * Methods of `Clone`, `PartialEq`, `Eq`, `PartialOrd`, and `Ord` behave
///   identically to how they would for usizes.
/// * `Hash` and `Debug` impls must not panic.
pub unsafe trait Idx: Copy + 'static + Eq + Debug + Ord + Hash + Send + Sync {
    fn new(idx: usize) -> Self;
    fn index(self) -> usize;
}

unsafe impl Idx for usize {
    #[inline]
    fn new(idx: usize) -> Self { idx }
    #[inline]
    fn index(self) -> usize { self }
}

#[macro_export]
macro_rules! newtype_index {
    ($type:ident) => (
        #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub struct $type(usize);

        unsafe impl Idx for $type {
            #[inline]
            fn new(value: usize) -> Self {
                $type(value)
            }

            #[inline]
            fn index(self) -> usize {
                self.0
            }
        }

        impl $crate::IsNewtypeIdx for $type {}

        impl ::std::fmt::Display for $type {
            fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
                ::std::fmt::Display::fmt(&self.0, f)
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

// Whether `Indexed` is `Send` depends only on the data,
// not the phantom data.
unsafe impl<I: Idx, V> Send for Indexed<I, V> where V: Send {}

impl<I: Idx, V: ?Sized + fmt::Debug> fmt::Debug for Indexed<I, V> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&self.raw, fmt)
    }
}

/// # Construction
impl<I: Idx, T> Indexed<I, Vec<T>> {
    #[inline]
    pub fn from_raw(raw: Vec<T>) -> Self {
        Indexed { raw, _marker: PhantomData }
    }

    #[inline]
    pub fn new() -> Self {
        Indexed::from_raw(Vec::new())
    }

    #[inline]
    pub fn from_elem<S>(elem: T, universe: &Indexed<I, [S]>) -> Self
    where T: Clone,
    {
        Indexed::from_raw(vec![elem; universe.len()])
    }

    #[inline]
    pub fn from_elem_n(elem: T, n: usize) -> Self
    where T: Clone,
    {
        Indexed::from_raw(vec![elem; n])
    }

    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Indexed::from_raw(Vec::with_capacity(capacity))
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
        assert!(ai != bi);

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

/// Either `[X]` (for `I = usize`) or `Indexed<I, [X]>`.
pub type SliceType<I, X> = <() as IndexFamily<I, X>>::Slice;

/// Either `Vec<X>` (for `I = usize`) or `Indexed<I, Vec<X>>`.
pub type OwnedType<I, X> = <() as IndexFamily<I, X>>::Owned;

/// Can be used to retrofit support for Indexed into old vec/slice-based interfaces.
///
/// The associated types `Slice` and `Owned` are `[X]` and `Vec<X>` for `I = usize`,
/// and `Indexed<I, _>` for anything else.  That makes them suitable for retrofitting
/// functions that used to return slices or vecs.
///
/// Thanks to this (and liberal use of `usize` defaults for index type parameters),
/// adoption of `Indexed` can be done incrementally.
///
/// ...the annoying bit is that it frequently needs to be written explicitly in where bounds.
pub trait IndexFamily<I: Idx, X> {
    type Slice: ?Sized;
    type Owned;

    // For converting the output type of a method that lies at the interface between
    // code generic over `I: Idx` and code that is not.
    fn ref_from_indexed(slice: &Indexed<I, [X]>) -> &Self::Slice;
    fn mut_from_indexed(slice: &mut Indexed<I, [X]>) -> &mut Self::Slice;
    fn owned_from_indexed(vec: Indexed<I, Vec<X>>) -> Self::Owned;

    // `OwnedType<I, X>` as an input arg does not work the way you might hope. (rust won't
    // be able to infer `I` and `X`.)  You'll have to use `impl IntoIndexed`, even though
    // it doesn't constrain the input type as well as `Vec` used to.
//    fn indexed_from_owned(vec: Self::Owned) -> Indexed<I, Vec<X>>;

    // I can't think of why these would ever be needed. Take `impl AsIndexed`
    // or `impl AsIndexedMut`, which have no foreseeable disadvantages.
//    fn indexed_from_ref(slice: &Self::Slice) -> &Indexed<I, [X]>;
//    fn indexed_from_mut(slice: &mut Self::Slice) -> &mut Indexed<I, [X]>;
}

impl<X> IndexFamily<usize, X> for () {
    type Slice = [X];
    type Owned = Vec<X>;

    #[inline] fn ref_from_indexed(slice: &Indexed<usize, [X]>) -> &Self::Slice
    { &slice.raw }
    #[inline] fn mut_from_indexed(slice: &mut Indexed<usize, [X]>) -> &mut Self::Slice
    { &mut slice.raw }
    #[inline] fn owned_from_indexed(vec: Indexed<usize, Vec<X>>) -> Self::Owned
    { vec.raw }

//    #[inline] fn indexed_from_ref(slice: &Self::Slice) -> &Indexed<usize, [X]>
//    { Indexed::from_raw_ref(slice) }
//    #[inline] fn indexed_from_mut(slice: &mut Self::Slice) -> &mut Indexed<usize, [X]>
//    { Indexed::from_raw_mut(slice) }
//    #[inline] fn indexed_from_owned(vec: Self::Owned) -> Indexed<usize, Vec<X>>
//    { Indexed::from_raw(vec) }
}

/// Implemented by newtype indices to allow `IndexFamily` to distinguish them from `usize`.
///
/// **Note:** Whenever you see a type error that "I: IsNewtypeIdx is not satisfied", there is
/// literally a 0% chance that adding this bound is the solution.  Some possible causes of
/// this error:
///
/// * Did you forget to add a `IndexFamily<I, X>` bound?  (these bounds are required in virtually
///   one-to-one correspondence with appearances of `SliceType`/`VecType`s)
/// * Did you write `SliceType<I, [X]>` instead of `SliceType<I, X>`?
/// * If you're calling a method of a type that has an `IndexFamily` bound, and you are writing
///   code that is generic over `I: Idx`, look for a version of the method without the bound.
///   (its name might contain the word 'indexed', and it will return `Indexed` where the original
///    method returned `SliceType` or `OwnedType`.  This will make your life 1000x times easier
///    than trying to work with `SliceType` or `OwnedType` in a generic context.)
pub trait IsNewtypeIdx: Idx {}

impl<I: IsNewtypeIdx, X> IndexFamily<I, X> for () {
    type Slice = Indexed<I, [X]>;
    type Owned = Indexed<I, Vec<X>>;

    #[inline] fn ref_from_indexed(slice: &Indexed<I, [X]>) -> &Self::Slice { slice }
    #[inline] fn mut_from_indexed(slice: &mut Indexed<I, [X]>) -> &mut Self::Slice { slice }
    #[inline] fn owned_from_indexed(vec: Indexed<I, Vec<X>>) -> Self::Owned { vec }

//    #[inline] fn indexed_from_ref(slice: &Self::Slice) -> &Indexed<I, [X]> { slice }
//    #[inline] fn indexed_from_mut(slice: &mut Self::Slice) -> &mut Indexed<I, [X]> { slice }
//    #[inline] fn indexed_from_owned(vec: Self::Owned) -> Indexed<I, Vec<X>> { vec }
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
    fn borrow(&self) -> &Indexed<I, [T]> { self.as_indexed() }
}

impl<I: Idx, T> BorrowMut<Indexed<I, [T]>> for Indexed<I, Vec<T>> {
    #[inline]
    fn borrow_mut(&mut self) -> &mut Indexed<I, [T]> { self.as_indexed_mut() }
}

impl<I: Idx, T: Clone> ToOwned for Indexed<I, [T]> {
    type Owned = Indexed<I, Vec<T>>;

    #[inline]
    fn to_owned(&self) -> Self::Owned { self.into_indexed() }
}

//--------------------------------------------------------

/// A trait recommended for use in argument lists where you require an `&Indexed<I, [T]>`.
///
/// It converts `&[T]` into `&Indexed<usize, [T]>`, helping support legacy code, and is also
/// polymorphic over by-value versus by-ref, which is just generally nice.
///
/// Use as `impl AsIndexed<I, T>` in an argument list.
pub trait AsIndexed: HasIndexType {
    type Elem;

    fn as_indexed(&self) -> &Indexed<Self::Index, [Self::Elem]>;
}

/// A trait recommended for use in argument lists where you require a `&mut Indexed<I, [T]>`.
///
/// It converts `&mut [T]` into `&mut Indexed<usize, [T]>`, helping support legacy code,
/// and is also polymorphic over by-value versus by-ref, which is just generally nice.
///
/// Use as `impl AsIndexedMut<I, T>` in an argument list.
pub trait AsIndexedMut: AsIndexed {
    fn as_indexed_mut(&mut self) -> &mut Indexed<Self::Index, [Self::Elem]>;
}

/// A trait recommended for use in argument lists where you require a `Indexed<I, Vec<T>>`.
///
/// It converts `Vec<T>` into `Indexed<usize, Vec<T>>`, helping support legacy code,
/// and is also polymorphic over by-value (no-op) versus by-ref (copy), which is just
/// generally nice.
///
/// Use as `impl AsIndexedMut<I, T>` in an argument list.
pub trait IntoIndexed: HasIndexType {
    type Elem;

    fn into_indexed(self) -> Indexed<Self::Index, Vec<Self::Elem>>;
}

impl<T> AsIndexed for [T] {
    type Elem = T;

    #[inline]
    fn as_indexed(&self) -> &Indexed<usize, [Self::Elem]> {
        Indexed::from_raw_ref(self)
    }
}

impl<T> AsIndexedMut for [T] {
    #[inline]
    fn as_indexed_mut(&mut self) -> &mut Indexed<usize, [Self::Elem]> {
        Indexed::from_raw_mut(self)
    }
}

impl<'a, T> IntoIndexed for &'a [T]
where
    T: Clone,
{
    type Elem = T;

    #[inline]
    fn into_indexed(self) -> Indexed<usize, Vec<Self::Elem>> {
        Indexed::from_raw(self.to_owned())
    }
}

impl<'a, T> IntoIndexed for &'a Vec<T>
where
    T: Clone,
{
    type Elem = T;

    #[inline]
    fn into_indexed(self) -> Indexed<usize, Vec<Self::Elem>> {
        Indexed::from_raw(self.to_owned())
    }
}

impl<T> AsIndexed for Vec<T> {
    type Elem = T;

    #[inline]
    fn as_indexed(&self) -> &Indexed<usize, [Self::Elem]> {
        self[..].as_indexed()
    }
}

impl<T> AsIndexedMut for Vec<T> {
    #[inline]
    fn as_indexed_mut(&mut self) -> &mut Indexed<usize, [Self::Elem]> {
        self[..].as_indexed_mut()
    }
}

impl<T> IntoIndexed for Vec<T> {
    type Elem = T;

    #[inline]
    fn into_indexed(self) -> Indexed<usize, Vec<Self::Elem>> {
        Indexed::from_raw(self)
    }
}

impl<I: Idx, T> AsIndexed for Indexed<I, [T]> {
    type Elem = T;

    #[inline]
    fn as_indexed(&self) -> &Indexed<Self::Index, [Self::Elem]> {
        self
    }
}

impl<I: Idx, T> AsIndexedMut for Indexed<I, [T]> {
    #[inline]
    fn as_indexed_mut(&mut self) -> &mut Indexed<Self::Index, [Self::Elem]> {
        self
    }
}

impl<'a, I: Idx, T> IntoIndexed for &'a Indexed<I, [T]>
where
    T: Clone,
{
    type Elem = T;

    #[inline]
    fn into_indexed(self) -> Indexed<Self::Index, Vec<Self::Elem>> {
        Indexed::from_raw(self.raw.to_owned())
    }
}

impl<'a, I: Idx, T> IntoIndexed for &'a Indexed<I, Vec<T>>
where
    T: Clone,
{
    type Elem = T;

    #[inline]
    fn into_indexed(self) -> Indexed<Self::Index, Vec<Self::Elem>> {
        Indexed::from_raw(self.raw.to_owned())
    }
}

impl<I: Idx, T> AsIndexed for Indexed<I, Vec<T>> {
    type Elem = T;

    #[inline]
    fn as_indexed(&self) -> &Indexed<Self::Index, [Self::Elem]> {
        self
    }
}

impl<I: Idx, T> AsIndexedMut for Indexed<I, Vec<T>> {
    #[inline]
    fn as_indexed_mut(&mut self) -> &mut Indexed<Self::Index, [Self::Elem]> {
        self
    }
}

impl<I: Idx, T> IntoIndexed for Indexed<I, Vec<T>> {
    type Elem = T;

    #[inline]
    fn into_indexed(self) -> Indexed<Self::Index, Vec<Self::Elem>> {
        self
    }
}

impl<'a, V: ?Sized + AsIndexed> AsIndexed for &'a V {
    type Elem = V::Elem;

    #[inline]
    fn as_indexed(&self) -> &Indexed<Self::Index, [Self::Elem]> {
        (**self).as_indexed()
    }
}

impl<'a, V: ?Sized + AsIndexed> AsIndexed for &'a mut V {
    type Elem = V::Elem;

    #[inline]
    fn as_indexed(&self) -> &Indexed<Self::Index, [Self::Elem]> {
        (**self).as_indexed()
    }
}

impl<'a, V: ?Sized + AsIndexedMut> AsIndexedMut for &'a mut V {
    #[inline]
    fn as_indexed_mut(&mut self) -> &mut Indexed<Self::Index, [Self::Elem]> {
        (**self).as_indexed_mut()
    }
}

// NOTE: this also supports &&&V and &&&&V and etc. through induction,
// though in practice only &&V ever shows up.
impl<'a, 'b, V: ?Sized> IntoIndexed for &'a &'b V
where &'b V: IntoIndexed,
{
    type Elem = <&'b V as IntoIndexed>::Elem;

    #[inline]
    fn into_indexed(self) -> Indexed<Self::Index, Vec<Self::Elem>> {
        (**self).into_indexed()
    }
}

#[test]
fn test_impl_existence() {
    newtype_index!{Foo}

    fn check_as_indexed(_v: impl AsIndexed) {}
    fn check_as_indexed_mut(_v: impl AsIndexedMut) {}
    fn check_into_indexed(_v: impl IntoIndexed) {}

    let mut vec = vec![()];
    let mut indexed_vec_1: Indexed<usize, Vec<()>> = vec![()].into_iter().collect();
    let mut indexed_vec_2: Indexed<Foo, Vec<()>> = vec![()].into_iter().collect();
    check_as_indexed(vec.clone());
    check_as_indexed(indexed_vec_1.clone());
    check_as_indexed(indexed_vec_2.clone());
    check_as_indexed(&vec);
    check_as_indexed(&indexed_vec_1);
    check_as_indexed(&indexed_vec_2);
    check_as_indexed(&vec[..]);
    check_as_indexed(&indexed_vec_1[..]);
    check_as_indexed(&indexed_vec_2[..]);

    check_as_indexed_mut(vec.clone());
    check_as_indexed_mut(indexed_vec_1.clone());
    check_as_indexed_mut(indexed_vec_2.clone());
    check_as_indexed_mut(&mut vec);
    check_as_indexed_mut(&mut indexed_vec_1);
    check_as_indexed_mut(&mut indexed_vec_2);
    check_as_indexed_mut(&mut vec[..]);
    check_as_indexed_mut(&mut indexed_vec_1[..]);
    check_as_indexed_mut(&mut indexed_vec_2[..]);

    check_into_indexed(vec.clone());
    check_into_indexed(indexed_vec_1.clone());
    check_into_indexed(indexed_vec_2.clone());
    check_into_indexed(&vec);
    check_into_indexed(&indexed_vec_1);
    check_into_indexed(&indexed_vec_2);
    check_into_indexed(&vec[..]);
    check_into_indexed(&indexed_vec_1[..]);
    check_into_indexed(&indexed_vec_2[..]);

    // types that frequently end up in function arguments after Extract Method refactoring...
    check_as_indexed(&&vec[..]);
    check_as_indexed(&&indexed_vec_1[..]);
    check_as_indexed(&&indexed_vec_2[..]);
    check_as_indexed_mut(&mut &mut vec[..]);
    check_as_indexed_mut(&mut &mut indexed_vec_1[..]);
    check_as_indexed_mut(&mut &mut indexed_vec_2[..]);
    check_into_indexed(&&vec[..]);
    check_into_indexed(&&indexed_vec_1[..]);
    check_into_indexed(&&indexed_vec_2[..]);
}

//--------------------------------------------------------

pub trait HasIndexType {
    type Index: Idx;
}

impl<T> HasIndexType for [T] {
    type Index = usize;
}

impl<T> HasIndexType for Vec<T> {
    type Index = usize;
}

impl<I: Idx, V: ?Sized> HasIndexType for Indexed<I, V> {
    type Index = I;
}

impl<'a, V: ?Sized + HasIndexType> HasIndexType for &'a V {
    type Index = V::Index;
}

impl<'a, V: ?Sized + HasIndexType> HasIndexType for &'a mut V {
    type Index = V::Index;
}

impl<V: ?Sized + HasIndexType> HasIndexType for Box<V> {
    type Index = V::Index;
}

impl<A, B, I: Idx> HasIndexType for (A, B)
where
    A: HasIndexType<Index=I>,
    B: HasIndexType<Index=I>,
{
    type Index = I;
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
