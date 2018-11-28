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

extern crate failure;
use std::fmt;

pub const DEFAULT_NONZERO_TOL: f64 = 1e-9;

#[macro_export]
macro_rules! assert_close {
    ($($t:tt)*) => {
        assert_close_impl!{
            @parsing [assert_close] [$($t)*]
            [[@rel $crate::DEFAULT_NONZERO_TOL] [@abs 0.0]]
        }
    };
}

/// Like `assert_close`, but produces a result.
#[macro_export]
macro_rules! check_close {
    ($($t:tt)*) => {
        assert_close_impl!{
            @parsing [check_close] [$($t)*]
            [[@rel $crate::DEFAULT_NONZERO_TOL] [@abs 0.0]]
        }
    };
}

#[macro_export]
macro_rules! debug_assert_close {
    ($($t:tt)*) => {{
        #[cfg(debug_assertions)] {
            assert_close!{$($t)*}
        }
    }};
}

#[doc(hidden)]
#[macro_export]
macro_rules! assert_close_impl {
    // the reason for $mac is that both assert_close! and check_close! need to parse the
    // tolerances.  (assert_close! cannot just let check_close! handle this, because
    // assert_close! needs to know where the format arguments begin)
    (@parsing [$mac:ident] [rel=$tol:expr, $($rest:tt)*] [$($assignment:tt)*]) => {
        assert_close_impl!(@parsing [$mac] [$($rest)*] [$($assignment)* [@rel $tol]]);
    };
    (@parsing [$mac:ident] [abs=$tol:expr, $($rest:tt)*] [$($assignment:tt)*]) => {
        assert_close_impl!(@parsing [$mac] [$($rest)*] [$($assignment)* [@abs $tol]]);
    };
    (@parsing [$mac:ident] [$a:expr, $b:expr $(,)*] $assignments:tt) => {
        assert_close_impl!(@expand [$mac] $assignments [@comp $a, $b] [@fmt])
    };
    (@parsing [$mac:ident] [$a:expr, $b:expr, $($fmt:tt)+] $assignments:tt) => {
        assert_close_impl!(@expand [$mac] $assignments [@comp $a, $b] [@fmt $($fmt)+])
    };

    (@expand [check_close] [$($assignment:tt)*] [@comp $a:expr, $b:expr] [@fmt $($fmt:tt)+] ) => {
        compile_error!{"check_close only takes 2 positional arguments"}
    };
    (@expand [check_close] [$($assignment:tt)*] [@comp $a:expr, $b:expr] [@fmt] ) => {{
        assert_close_impl!{
            @expand [let result =] [$($assignment)*] [@comp $a, $b] [@fmt]
        }
        result.0
    }};

    (@expand [assert_close] [$($assignment:tt)*] [@comp $a:expr, $b:expr] [@fmt] ) => {
        assert_close_impl!{@expand [assert_close] [$($assignment)*] [@comp $a, $b] [@fmt "not nearly equal!"]}
    };
    (@expand [assert_close] [$($assignment:tt)*] [@comp $a:expr, $b:expr] [@fmt $($fmt:tt)+] ) => {{
        assert_close_impl!{
            @expand [let result =] [$($assignment)*] [@comp $a, $b] [@fmt]
        }
        if let (Err(e), a, b, abs, rel) = result {
            panic!(
                "{} (tolerances: rel={:e}, abs={:e})\n left: {:?}\nright: {:?}\n{}\n",
                format!($($fmt)*), rel, abs, a, b, e,
            );
        }
    }};

    // NOTE: This must expand to statements instead of a block so that temporaries in $a and $b
    //       are assigned reasonable lifetimes.
    (@expand [let $result:ident =] [$($assignment:tt)*] [@comp $a:expr, $b:expr] [@fmt] ) => {
        let a = $a;
        let b = $b;
        let abs: f64;
        let rel: f64;
        let $result =
        ( // a tuple to include other things for use in assert_close!'s formatting
            #[allow(unused_mut)]
            #[allow(unused_assignments)]
            {
                let mut tmp_abs;
                let mut tmp_rel;
                $(
                    assert_close_impl!{@stmt::assign [tmp_abs, tmp_rel] $assignment}
                )*
                abs = tmp_abs;
                rel = tmp_rel;

                $crate::CheckClose::check_close(&a, &b, $crate::Tolerances { abs, rel })
            },
            // (the other tuple items)
            a, b, abs, rel,
        );
    };
    (@stmt::assign [$abs:ident, $rel:ident] [@abs $tol:expr]) => { $abs = $tol; };
    (@stmt::assign [$abs:ident, $rel:ident] [@rel $tol:expr]) => { $rel = $tol; };
}

#[doc(hidden)]
#[allow(non_snake_case)]
#[inline]
pub fn __is_close(a: f64, b: f64, Tolerances { abs, rel }: Tolerances) -> bool {
    // Implementation from Python 3.5.
    // https://hg.python.org/cpython/file/tip/Modules/mathmodule.c#l1993
    assert!(rel >= 0.0);
    assert!(abs >= 0.0);

    // catch infinities of same sign
    if a == b { return true; }

    // catch infinities of opposite sign, avoiding infinite relative tolerance
    if a.is_infinite() || b.is_infinite() { return false; }

    // case for general values and NaN.
    (a - b).abs() < abs.max(rel * a.abs()).max(rel * b.abs())
}

#[derive(Debug, Copy, Clone)]
pub struct Tolerances<T = f64> {
    pub abs: T,
    pub rel: T
}

#[derive(Debug)]
pub struct CheckCloseError<T = f64> {
    pub values: (T, T),
    pub tol: Tolerances<T>,
}

// avoid `T: Fail` bound
impl<T: fmt::Debug + Send + Sync + 'static> failure::Fail for CheckCloseError<T> {
    fn backtrace(&self) -> Option <&::failure::Backtrace> { None }
    fn cause(&self) -> Option <&::failure::Fail> { None }
}

impl<T: fmt::Debug> fmt::Display for CheckCloseError<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let (left, right) = &self.values;
        write!(f, "failed at:
  left: {:?}
 right: {:?}
   tol: {:?}", left, right, self.tol)
    }
}

pub trait CheckClose<Rhs: ?Sized = Self>: {
    type Scalar;

    /// Test that all values of self and other are close.
    fn check_close(&self, other: &Rhs, tol: Tolerances) -> Result<(), CheckCloseError<Self::Scalar>>;
}

impl CheckClose for f64 {
    type Scalar = f64;

    #[inline]
    fn check_close(&self, other: &Self, tol: Tolerances) -> Result<(), CheckCloseError<Self::Scalar>>
    {
        if __is_close(*self, *other, tol) {
            Ok(())
        } else {
            Err(CheckCloseError {
                values: (*self, *other),
                tol,
            })
        }
    }
}

impl<'a, T: ?Sized + CheckClose> CheckClose for &'a T {
    type Scalar = T::Scalar;

    fn check_close(&self, other: &Self, tol: Tolerances) -> Result<(), CheckCloseError<Self::Scalar>>
    { CheckClose::check_close(*self, *other, tol) }
}

impl<T: CheckClose> CheckClose for [T] {
    type Scalar = T::Scalar;

    fn check_close(&self, other: &Self, tol: Tolerances) -> Result<(), CheckCloseError<Self::Scalar>>
    {
        assert_eq!(self.len(), other.len());
        self.iter().zip(other)
            .map(|(a, b)| a.check_close(b, tol))
            .collect()
    }
}

impl<T: CheckClose> CheckClose for Vec<T> {
    type Scalar = T::Scalar;

    fn check_close(&self, other: &Self, tol: Tolerances) -> Result<(), CheckCloseError<Self::Scalar>>
    { (&self[..]).check_close(&other[..], tol) }
}

impl<T: CheckClose> CheckClose<[T]> for Vec<T> {
    type Scalar = T::Scalar;

    fn check_close(&self, other: &[T], tol: Tolerances) -> Result<(), CheckCloseError<Self::Scalar>>
    { (&self[..]).check_close(&other[..], tol) }
}

impl<T: CheckClose> CheckClose<Vec<T>> for [T] {
    type Scalar = T::Scalar;

    fn check_close(&self, other: &Vec<T>, tol: Tolerances) -> Result<(), CheckCloseError<Self::Scalar>>
    { (&self[..]).check_close(&other[..], tol) }
}

macro_rules! gen_array_impls {
    ($($n:tt)*) => {
        $(
        impl<T: CheckClose> CheckClose for [T; $n] {
            type Scalar = T::Scalar;

            fn check_close(&self, other: &Self, tol: Tolerances) -> Result<(), CheckCloseError<Self::Scalar>>
            { (&self[..]).check_close(&other[..], tol) }
        }
        )*
    };
}

gen_array_impls! {
     0  1  2  3  4  5  6  7  8  9
    10 11 12 13 14 15 16 17 18 19
    20 21 22 23 24 25 26 27 28 29
    30 31 32
}

#[cfg(test)]
#[deny(unused)]
mod tests {
    #[test]
    fn macro_output_can_compile() {
        assert_close!(1.0, 1.0);
        assert_close!(abs=1e-8, 1.0, 1.0);
        assert_close!(rel=1e-8, abs=1e-8, 1.0, 1.0);
        assert_close!(1.0, 1.0,);
        assert_close!(abs=1e-8, 1.0, 1.0,);
        assert_close!(rel=1e-8, abs=1e-8, 1.0, 1.0,);
    }

    #[test]
    fn bad_parse_regression() {
        #[derive(Debug)] struct S;
        impl S { fn x(self) -> S { self } }
        impl crate::CheckClose for S {
            type Scalar = f64;
            fn check_close(&self, _: &S, _: crate::Tolerances) -> Result<(), crate::CheckCloseError<Self::Scalar>> { Ok(()) }
        }
        assert_close!(
            abs=1e-10,
            S.x().x().x(),
            S.x().x().x(),
        );
        debug_assert_close!(
            abs=1e-10,
            S.x().x().x(),
            S.x().x().x(),
        );
        assert_close!(
            abs=1e-10,
            S.x().x().x(),
            S.x().x().x(),
            "{}", "hello",
        );
    }

    #[test]
    #[should_panic]
    fn not_close() {
        assert_close!(abs=0.0, rel=0.0, 1.0, 1.1);
    }

    #[test]
    #[cfg_attr(debug_assertions, should_panic)]
    fn debug_not_close() {
        debug_assert_close!(abs=0.0, rel=0.0, 1.0, 1.1);
    }

    #[test]
    fn reasonable_lifetimes_of_temporaries() {
        let vec = vec![1.0, 1.0];
        assert_close!(
            &vec.iter().cloned().collect::<Vec<_>>()[..],
            &vec.iter().cloned().collect::<Vec<_>>()[..],
        );
    }
}
