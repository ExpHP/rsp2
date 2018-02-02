//! # What is this?
//!
//! A failure.
//!
//! Some dumb abstraction I thought of to try and help make
//! it easier to split large tasks into multiple binaries.
//! It was supposed to help abstract over data that either exists
//! in memory (because it was just computed) or on the filesystem
//! (because it was from a previous separate binary).
//!
//! When I tried to actually implement it for some types, however,
//! I discovered that it was unusable.  I don't remember all the
//! details, but I do remember that the combinators were worthless
//! (due to ambiguities in trait resolution).
//!
//! # Why keep it?
//!
//! It was a terrible idea, but it's here just in case I try to
//! find it again later, so I can remind myself just how terrible
//! it is.

use ::{Result};
use ::traits::AsPath;
use ::traits::Load;
use ::alternate;

use ::std::sync::{Arc, Weak};

pub trait Source<T> {
    /// Get the value.
    fn get(&mut self) -> Result<T>;

    /// Get a cached value without performing any potentially dangerous
    /// IO or expensive computations.
    ///
    /// Different sources may have different conditions under which this
    /// returns `Some`.
    fn get_cached(&self) -> Option<T>;

    /// Guarantee that future calls to `get_cached()` return `Some`
    /// (barring arbitrary mutation to `self`).
    ///
    /// This does not guarantee that multiple calls to `get_cached()`
    /// will produce `Arc`s tied to the same reference count.
    fn ensure_cached(&mut self) -> Result<()>;
}

//-----------------------------------------------------

/// A simple source for a value already loaded in memory.
#[derive(Debug, Clone)]
pub struct ValueSource<T>(pub T);

impl<T: Clone> Source<T> for ValueSource<T>
{
    fn get(&mut self) -> Result<T>
    { Ok(self.0.clone()) }

    fn get_cached(&self) -> Option<T>
    { Some(self.0.clone()) }

    fn ensure_cached(&mut self) -> Result<()>
    { Ok(()) }
}

//-----------------------------------------------------


/// A source built from a deferred, repeatable computation.
///
/// Once computed, the value is cached internally by a weak reference,
/// so that at most one copy of the output exists in memory at any
/// given time.
pub struct FnSource<T, F> {
    func: F,
    // Weak pointer into the last value returned.
    cache: Option<Weak<T>>,
    // A strong pointer to implement `Source::retain`.
    retainer: Option<Arc<T>>,
}

impl<T, F> FnSource<T, F>
where F: alternate::FnMut<(), Output=Result<T>>,
{
    pub fn new(func: F) -> Self
    { FnSource {
        func,
        cache: None,
        retainer: None,
    }}
}

impl<T, F> Source<Arc<T>> for FnSource<T, F>
where F: alternate::FnMut<(), Output=Result<T>>,
{
    fn get(&mut self) -> Result<Arc<T>>
    {Ok({
        match self.get_cached() {
            Some(ptr) => ptr,
            None => {
                let ptr = Arc::new(self.func.call_mut(())?);
                self.cache = Some(Arc::downgrade(&ptr));
                ptr
            }
        }
    })}

    fn get_cached(&self) -> Option<Arc<T>>
    { self.cache.as_ref().and_then(Weak::upgrade) }

    fn ensure_cached(&mut self) -> Result<()>
    {Ok({
        self.retainer = Some(self.get()?.clone());
    })}
}

//-----------------------------------------------------

/// A source built from a deferred computation.
///
/// Once it is computed, the value is stored.
/// On the other hand, if the computation produces an `Err`,
/// the value is poisoned and cannot be retried.
#[derive(Debug, Clone)]
pub struct OnceSource<T, F>(Option<OnceSourceImpl<T, F>>);

// this layer of indirection allows poisoning the value
#[derive(Debug, Clone)]
enum OnceSourceImpl<T, F> {
    Once(F),
    Done(Arc<T>),
}

impl<T, F> OnceSource<T, F>
where F: alternate::FnOnce<(), Output=Result<T>>,
{
    pub fn new(func: F) -> Self
    { OnceSource(Some(OnceSourceImpl::Once(func))) }
}

impl<T, F> Source<Arc<T>> for OnceSource<T, F>
where F: ::alternate::FnOnce<(), Output=Result<T>>,
{
    fn get(&mut self) -> Result<Arc<T>>
    {Ok({
        self.ensure_cached()?;
        self.get_cached().expect("bug!")
    })}

    fn get_cached(&self) -> Option<Arc<T>>
    { match self.0.as_ref() {
        None => panic!("Called `get_cached()` on a poisoned `OnceSource`"),
        Some(&OnceSourceImpl::Once(_)) => None,
        Some(&OnceSourceImpl::Done(ref ptr)) => Some(ptr.clone()),
    }}

    fn ensure_cached(&mut self) -> Result<()>
    {Ok({
        self.0 = Some(match self.0.take() {
            None => panic!("Called `ensure_cached()` on a poisoned `OnceSource`"),
            Some(OnceSourceImpl::Once(func)) => {
                OnceSourceImpl::Done(Arc::new(func.call_once(())?))
            },
            Some(OnceSourceImpl::Done(value)) => OnceSourceImpl::Done(value),
        });
    })}
}

//-----------------------------------------------------

/// Output type of `path_source`.
pub type PathSource<T, P> = FnSource<T, unboxed::LoadFromPath<T, P>>;

pub fn path_source<T, P>(path: P) -> PathSource<T, P>
where
    P: AsPath,
    T: Load,
{
    FnSource::new(
        unboxed::LoadFromPath { path, _markers: Default::default() },
    )
}

// NOTE: Experimental.
// Product type combinators.
//
// They wrap the source. (wrapping the output and having a bare source
// *seems* like a good idea until you realize that the mere existence of
// such an impl destroys type inference literally everywhere)
pub struct Each<S>(pub S);
pub struct Multi<S>(pub S);

macro_rules! derive_tuple_source {
    ($([$a:ident : $A:ident, $s:ident : $S:ident],)*)
    => {
        impl<$($A,)* $($S,)*> Source<($($A,)*)> for Each<($($S,)*)>
        where $($S: Source<$A>,)*
        {
            fn get(&mut self) -> Result<($($A,)*)>
            {Ok({
                let Each(($(ref mut $s,)*)) = *self;
                $(
                    let $a = $s.get()?;
                )*
                ($($a,)*)
            })}

            fn get_cached(&self) -> Option<($($A,)*)>
            {
                let Each(($(ref $s,)*)) = *self;
                $(
                    let $a = match $s.get_cached() {
                        None => return None,
                        Some(x) => x,
                    };
                )*
                Some(($($a,)*))
            }

            fn ensure_cached(&mut self) -> Result<()>
            {Ok({
                let Each(($(ref mut $s,)*)) = *self;
                $(
                    $s.ensure_cached()?;
                )*
            })}
        }

        impl<$($A,)* S> Source<($($A,)*)> for Multi<S>
        where $(S: Source<$A>,)*
        {
            fn get(&mut self) -> Result<($($A,)*)>
            {Ok({
                $(
                    let $a = self.0.get()?;
                )*
                ($($a,)*)
            })}

            fn get_cached(&self) -> Option<($($A,)*)>
            {
                $(
                    let $a = match self.0.get_cached() {
                        None => return None,
                        Some(x) => x,
                    };
                )*
                Some(($($a,)*))
            }

            fn ensure_cached(&mut self) -> Result<()>
            {Ok({
                $(
                    <S as Source<$A>>::ensure_cached(&mut self.0)?;
                )*
            })}
        }
    }
}

macro_rules! derive_all_tuple_sources {
    () => {
        derive_tuple_source!();
    };

    (
        [$a0:ident : $A0:ident, $s0:ident : $S0:ident],
        $([$a:ident : $A:ident, $s:ident : $S:ident],)*
    )
    => {
        derive_tuple_source!{
            [$a0 : $A0, $s0 : $S0],
            $([$a : $A, $s : $S],)*
        }

        derive_all_tuple_sources!{
            $([$a : $A, $s : $S],)*
        }
    }
}

derive_all_tuple_sources! {
    [a8: A8, s8: S8],
    [a7: A7, s7: S7],
    [a6: A6, s6: S6],
    [a5: A5, s5: S5],
    [a4: A4, s4: S4],
    [a3: A3, s3: S3],
    [a2: A2, s2: S2],
    [a1: A1, s1: S1],
}

mod unboxed {
    use super::*;

    /// Unboxed closure form of `move || Load::load(path)`
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct LoadFromPath<T, P> {
        pub(crate) path: P,
        pub(crate) _markers: ::std::marker::PhantomData<T>,
    }

    derive_alternate_fn!{
        impl[P, T] Fn<()> for LoadFromPath<T, P>
        [ where
            P: AsPath,
            T: Load,
        ] {
            type Output = Result<T>;
            fn call(&self, ():()) -> Result<T>
            { Load::load(&self.path) }
        }
    }
}
