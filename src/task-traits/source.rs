use ::AsPath;
use ::{Result, StdResult, Error};
use ::Load;
use ::alternate;

use ::std::sync::{Arc, Weak};

pub trait Source<T> {
    type Error;

    /// Get the value.
    fn get(&mut self) -> StdResult<Arc<T>, Self::Error>;

    /// Get a cached value without performing any potentially dangerous
    /// IO or expensive computations.
    ///
    /// Different sources may have different conditions under which this
    /// returns `Some`, though  The method `ensure_cached` allows .
    fn get_cached(&self) -> Option<Arc<T>>;

    /// Guarantee that future calls to `get_cached()` return `Some`
    /// (barring arbitrary mutation to `self`).
    ///
    /// This does not guarantee that multiple calls to `get_cached()`
    /// will produce `Arc`s tied to the same reference count.
    fn ensure_cached(&mut self) -> StdResult<(), Self::Error>;
}

//-----------------------------------------------------

/// A simple source for a value already loaded in memory.
#[derive(Debug, Clone)]
pub struct ValueSource<T>(pub Arc<T>);

impl<T> Source<T> for ValueSource<T>
{
    type Error = Error;

    fn get(&mut self) -> Result<Arc<T>>
    { Ok(self.0.clone()) }

    fn get_cached(&self) -> Option<Arc<T>>
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

impl<T, F, E> FnSource<T, F>
where F: alternate::FnMut<(), Output=StdResult<T, E>>,
{
    pub fn new(func: F) -> Self
    { FnSource {
        func,
        cache: None,
        retainer: None,
    }}
}

impl<T, F, E> Source<T> for FnSource<T, F>
where F: alternate::FnMut<(), Output=StdResult<T, E>>,
{
    type Error = E;

    fn get(&mut self) -> StdResult<Arc<T>, E>
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

    fn ensure_cached(&mut self) -> StdResult<(), E>
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

impl<T, F, E> OnceSource<T, F>
where F: alternate::FnOnce<(), Output=StdResult<T, E>>,
{
    pub fn new(func: F) -> Self
    { OnceSource(Some(OnceSourceImpl::Once(func))) }
}

impl<T, F, E> Source<T> for OnceSource<T, F>
where F: ::alternate::FnOnce<(), Output=StdResult<T, E>>,
{
    type Error = E;

    fn get(&mut self) -> StdResult<Arc<T>, E>
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

    fn ensure_cached(&mut self) -> StdResult<(), E>
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
// Product type combinator.
//
// A tuple of Sources `(S1, S2, S3, S4)` is a Source of
// `(Arc<A1>, Arc<A2>, Arc<A3>, Arc<A4>)`, which of course
// gets wrapped in another `Arc`, resulting in...
// a lot of Arcs.
//
// The Helper just packs an additional type parameter to allow
// selecting the error type.
pub struct Helper<T, E>(T, ::std::marker::PhantomData<E>);
macro_rules! derive_tuple_source {
    ($([$a:ident : $A:ident, $s:ident : $S:ident],)*)
    => {
        impl<$($A,)* $($S,)* E> Source<($(Arc<$A>,)*)> for Helper<($($S,)*), E>
        where
            $($S: Source<$A>,)*
            $(E: From<<$S as Source<$A>>::Error>,)*
        {
            type Error = E;

            fn get(&mut self) -> StdResult<Arc<($(Arc<$A>,)*)>, E>
            {Ok({
                let Helper(($(ref mut $s,)*), _) = *self;
                $(
                    let $a = $s.get()?;
                )*
                Arc::new(($($a,)*))
            })}

            fn get_cached(&self) -> Option<Arc<($(Arc<$A>,)*)>>
            {
                let Helper(($(ref $s,)*), _) = *self;
                $(
                    let $a = match $s.get_cached() {
                        None => return None,
                        Some(x) => x,
                    };
                )*
                Some(Arc::new(($($a,)*)))
            }

            fn ensure_cached(&mut self) -> StdResult<(), E>
            {Ok({
                let Helper(($(ref mut $s,)*), _) = *self;
                $(
                    $s.ensure_cached()?;
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

    rsp2_derive_alternate_fn!{
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
