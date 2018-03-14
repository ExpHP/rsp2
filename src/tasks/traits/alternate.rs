//! Copies of Rust's Fn traits, with the difference that they can be
//! implemented manually on a type without unstable compiler features.

// These traits basically just let us use more functions in more places without
// sacrificing too much DRY-ness, and without having to wait for existential
// types to get stabilized.
pub use ::std::prelude::v1::{
    Fn as StdFn,
    FnMut as StdFnMut,
    FnOnce as StdFnOnce,
};

use ::std::ops::{Deref, DerefMut};
use ::frunk::hlist::{HList, HNil, HCons};

/// Alternate `Fn` that can be manually implemented on stable.
///
/// It is automatically implemented for all types that implement
/// the standard library's `Fn`.  The advantage is that it may
/// also be implemented for explicitly defined "unboxed closures".
pub trait Fn<Args: HList>: FnMut<Args> {
    fn call(&self, args: Args) -> Self::Output;

    /// Get a borrowed form of the closure.
    #[inline(always)]
    fn by_ref(&self) -> Ref<Self> { Ref(self) }
}

/// Alternate `FnMut` that can be manually implemented on stable.
///
/// It is automatically implemented for all types that implement
/// the standard library's `FnMut`.  The advantage is that it may
/// also be implemented for explicitly defined "unboxed closures".
pub trait FnMut<Args: HList>: FnOnce<Args> {
    fn call_mut(&mut self, args: Args) -> Self::Output;

    /// Get a borrowed form of the closure.
    #[inline(always)]
    fn by_ref_mut(&mut self) -> RefMut<Self> { RefMut(self) }
}

pub type CallT<F, Args> = <F as FnOnce<Args>>::Output;

/// Alternate FnOnce that can be manually implemented on stable.
///
/// It is automatically implemented for all types that implement
/// the standard library's `FnOnce`.  The advantage is that it may
/// also be implemented for explicitly defined "unboxed closures".
pub trait FnOnce<Args: HList> {
    type Output;

    fn call_once(self, args: Args) -> Self::Output;
}

// ---------------

macro_rules! derive_fn_from_std {
    ($([$([$a:ident : $A:ident])*])*) => {
        $(
            impl<F, R, $($A,)*> FnOnce<Hlist![$($A),*]> for F where F: StdFnOnce($($A),*) -> R {
                type Output = R;

                #[inline(always)]
                fn call_once(self, hlist_pat![$($a),*]: Hlist![$($A),*]) -> Self::Output
                { self($($a),*) }
            }

            impl<F, R, $($A,)*> FnMut<Hlist![$($A),*]> for F where F: StdFnMut($($A),*) -> R {
                #[inline(always)]
                fn call_mut(&mut self, hlist_pat![$($a),*]: Hlist![$($A),*]) -> Self::Output
                { self($($a),*) }
            }

            impl<F, R, $($A,)*> Fn<Hlist![$($A),*]> for F where F: StdFn($($A),*) -> R {
                #[inline(always)]
                fn call(&self, hlist_pat![$($a),*]: Hlist![$($A),*]) -> Self::Output
                { self($($a),*) }
            }
        )*
    }
}

derive_fn_from_std!{
    []
    [[a0:A0]]
    [[a0:A0][a1:A1]]
    [[a0:A0][a1:A1][a2:A2]]
    [[a0:A0][a1:A1][a2:A2][a3:A3]]
    [[a0:A0][a1:A1][a2:A2][a3:A3][a4:A4]]
    [[a0:A0][a1:A1][a2:A2][a3:A3][a4:A4][a5:A5]]
    [[a0:A0][a1:A1][a2:A2][a3:A3][a4:A4][a5:A5][a6:A6]]
    [[a0:A0][a1:A1][a2:A2][a3:A3][a4:A4][a5:A5][a6:A6][a7:A7]]
    [[a0:A0][a1:A1][a2:A2][a3:A3][a4:A4][a5:A5][a6:A6][a7:A7][a8:A8]]
    [[a0:A0][a1:A1][a2:A2][a3:A3][a4:A4][a5:A5][a6:A6][a7:A7][a8:A8][a9:A9]]
    [[a0:A0][a1:A1][a2:A2][a3:A3][a4:A4][a5:A5][a6:A6][a7:A7][a8:A8][a9:A9][a10:A10]]
    [[a0:A0][a1:A1][a2:A2][a3:A3][a4:A4][a5:A5][a6:A6][a7:A7][a8:A8][a9:A9][a10:A10][a11:A11]]
}

// ---------------

pub trait Curry<A>: Sized {
    //    fn curry_ref(&self, a: A) -> Partial<Ref<Self>, A>;
    //    { self.by_ref().curry() }
    //    fn curry_mut(&mut self, a: A) -> Partial<RefMut<Self>, A>
    //    { self.by_ref_mut().curry() }
    fn curry(self, a: A) -> Partial<Self, A>;
}

// NOTE: Unable to constrain F here.
//       You could write `F: FnOnce<HCons<A, Rest>>`, but where does Rest come from?
impl<A, F> Curry<A> for F
{
    #[inline(always)]
    fn curry(self, a: A) -> Partial<Self, A>
    { Partial {
        function: self,
        arg: a,
    }}
}

#[derive(Debug, Copy, Clone)]
pub struct Partial<F, A> {
    function: F,
    arg: A,
}

impl<F, A, Rest: HList> FnOnce<Rest> for Partial<F, A>
    where F: FnOnce<HCons<A, Rest>>
{
    type Output = F::Output;

    #[inline(always)]
    fn call_once(self, list: Rest) -> Self::Output
    { self.function.call_once(list.prepend(self.arg)) }
}

// ---------------
// Can't have generic impls for &F and &mut F thanks to the blanket impl,
// so we'll work around it with newtypes.

/// A borrowed form of an `alternate::Fn`.
#[derive(Debug, Copy, Clone)]
pub struct Ref<'a, F: ?Sized + 'a>(&'a F);

/// A borrowed form of an `alternate::FnMut`.
#[derive(Debug)]
pub struct RefMut<'a, F: ?Sized + 'a>(&'a mut F);

impl<'a, F: ?Sized> Deref for Ref<'a, F> {
    type Target = F;

    fn deref(&self) -> &Self::Target { &*self.0 }
}

impl<'a, F: ?Sized> Deref for RefMut<'a, F> {
    type Target = F;

    fn deref(&self) -> &Self::Target { &*self.0 }
}

impl<'a, F: ?Sized> DerefMut for RefMut<'a, F> {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut *self.0 }
}

impl<'a, F: ?Sized, Args: HList> FnOnce<Args> for Ref<'a, F>
    where F: Fn<Args>,
{
    type Output = F::Output;

    fn call_once(self, args: Args) -> Self::Output
    { self.0.call(args) }
}

impl<'a, F: ?Sized, Args: HList> FnMut<Args> for Ref<'a, F>
    where F: Fn<Args>,
{
    fn call_mut(&mut self, args: Args) -> Self::Output
    { self.0.call(args) }
}

impl<'a, F: ?Sized, Args: HList> Fn<Args> for Ref<'a, F>
    where F: Fn<Args>,
{
    fn call(&self, args: Args) -> Self::Output
    { self.0.call(args) }
}

impl<'a, F: ?Sized, Args: HList> FnOnce<Args> for RefMut<'a, F>
    where F: FnMut<Args>,
{
    type Output = F::Output;

    fn call_once(mut self, args: Args) -> Self::Output
    { self.0.call_mut(args) }
}

impl<'a, F: ?Sized, Args: HList> FnMut<Args> for RefMut<'a, F>
    where F: FnMut<Args>,
{
    fn call_mut(&mut self, args: Args) -> Self::Output
    { self.0.call_mut(args) }
}
