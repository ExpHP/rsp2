#[macro_export]
macro_rules! assert_close {
    ($($t:tt)*) => {assert_close_impl!{@parsing [$($t)*] [[@rel 1e-9] [@abs 0.0]]}};
}

#[macro_export]
macro_rules! assert_close_impl {
    (@parsing [rel=$tol:expr, $($rest:tt)*] [$($assignment:tt)*]) => {
        assert_close_impl!(@parsing [$($rest)*] [$($assignment)* [@rel $tol]]);
    };
    (@parsing [abs=$tol:expr, $($rest:tt)*] [$($assignment:tt)*]) => {
        assert_close_impl!(@parsing [$($rest)*] [$($assignment)* [@abs $tol]]);
    };
    (@parsing [$a:expr, $b:expr] $assignments:tt) => {
        assert_close_impl!(@expand $assignments [@comp $a, $b] [@fmt "not nearly equal!"])
    };
    (@parsing [$a:expr, $b:expr $(,$rest:expr)+] $assignments:tt) => {
        assert_close_impl!(@expand $assignments [@comp $a, $b] [@fmt $($rest),+])
    };
    (@expand [$($assignment:tt)*] [@comp $a:expr, $b:expr] [@fmt $($fmt:expr),+] ) => {
        #[allow(unused_mut)]
        #[allow(unused_assignments)]
        {
            let a: f64 = $a;
            let b: f64 = $b;

            let mut abs: f64;
            let mut rel: f64;
            $(
                assert_close_impl!{@stmt::assign [abs, rel] $assignment}
            )*

            if !$crate::__is_close(a, b, abs, rel) {
                panic!(
                "{} (tolerances: rel={}, abs={})\n left: {}\nright: {}",
                 format!($($fmt),*), rel, abs, a, b);
            }
        }
    };
    (@stmt::assign [$abs:ident, $rel:ident] [@abs $tol:expr]) => { $abs = $tol; };
    (@stmt::assign [$abs:ident, $rel:ident] [@rel $tol:expr]) => { $rel = $tol; };
}

/// Ignore this.
#[allow(non_snake_case)]
pub fn __is_close(a: f64, b: f64, abs: f64, rel: f64) -> bool {
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