//! Copies of Rust's Fn traits, with the difference that they can be
//! implemented manually on a type without unstable compiler features.

// These traits basically just let us use more functions in more places without
// sacrificing too much DRYness, and without having to wait for existential
// types to get stabilized.
pub use ::std::prelude::v1::{
    Fn as StdFn,
    FnMut as StdFnMut,
    FnOnce as StdFnOnce,
};

/// Alternate `Fn` that can be manually implemented on stable.
///
/// It is automatically implemented for all types that implement
/// the standard library's `Fn`.  The advantage is that it may
/// also be implemented for explicitly defined "unboxed closures".
pub trait Fn<Args>: FnMut<Args> {
    fn call(&self, args: Args) -> Self::Output;
}

/// Alternate `FnMut` that can be manually implemented on stable.
///
/// It is automatically implemented for all types that implement
/// the standard library's `FnMut`.  The advantage is that it may
/// also be implemented for explicitly defined "unboxed closures".
pub trait FnMut<Args>: FnOnce<Args> {
    fn call_mut(&mut self, args: Args) -> Self::Output;
}

/// Alternate FnOnce that can be manually implemented on stable.
///
/// It is automatically implemented for all types that implement
/// the standard library's `FnOnce`.  The advantage is that it may
/// also be implemented for explicitly defined "unboxed closures".
pub trait FnOnce<Args> {
    type Output;
    fn call_once(self, args: Args) -> Self::Output;
}

impl<F, R, Args> FnOnce<Args> for F where F: StdFnOnce(Args) -> R {
    type Output = R;
    fn call_once(self, args: Args) -> Self::Output { self(args) }
}

impl<F, R, Args> FnMut<Args> for F where F: StdFnMut(Args) -> R {
    fn call_mut(&mut self, args: Args) -> Self::Output { self(args) }
}

impl<F, R, Args> Fn<Args> for F where F: StdFn(Args) -> R {
    fn call(&self, args: Args) -> Self::Output { self(args) }
}
