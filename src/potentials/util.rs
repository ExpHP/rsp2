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

use rsp2_minimize::numerical;
use rsp2_array_types::V3;

use slice_of_array::prelude::*;

#[cfg(test)]
pub(crate) fn uniform(a: f64, b: f64) -> f64 { ::rand::random::<f64>() * (b - a) + a }

#[allow(dead_code)] // used in method calls that are normally commented out
pub(crate) fn try_num_grad_v3<E>(
    interval: f64,
    point: V3,
    mut value_fn: impl FnMut(V3) -> Result<f64, E>,
) -> Result<V3, E> {
    numerical::try_gradient(interval, None, &point.0, |v| value_fn(v.to_array()))
        .map(|x| x.to_array())
}

#[allow(dead_code)] // used in method calls that are normally commented out
pub(crate) fn num_grad_v3(
    interval: f64,
    point: V3,
    mut value_fn: impl FnMut(V3) -> f64,
) -> V3 {
    numerical::gradient(interval, None, &point.0, |v| value_fn(v.to_array())).to_array()
}

//-------------------------------------------------------------------------------------------

/// Switches from 0 to 1 as x goes from `interval.0` to `interval.1`.
#[inline(always)] // elide direction check hopefully since intervals should be constant
pub(crate) fn switch(
    interpolate: impl FnOnce(f64) -> (f64, f64),
    interval: (f64, f64),
    x: f64,
) -> (f64, f64) {
    match IntervalSide::classify(interval, x) {
        IntervalSide::Left => (0.0, 0.0),
        IntervalSide::Inside => {
            let width = interval.1 - interval.0;
            let alpha = (x - interval.0) / width;
            let (value, d_alpha) = interpolate(alpha);
            (value, d_alpha / width)
        },
        IntervalSide::Right => (1.0, 0.0),
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) enum IntervalSide { Left, Inside, Right }
impl IntervalSide {
    /// Determine if a value is before the beginning or after the end of a directed interval
    /// (directed as in, `interval.1 < interval.0` is ok and flips the classifications of ±∞)
    ///
    /// Neither endpoint is considered to lie in the interval.
    ///
    /// Output is unspecified if `interval.0 == x == interval.1`.
    #[inline(always)] // elide direction check hopefully since intervals should be constant
    pub(crate) fn classify(interval: (f64, f64), x: f64) -> Self {
        if interval.0 < interval.1 {
            // interval is (min, max)
            match x {
                x if x <= interval.0 => IntervalSide::Left,
                x if interval.1 <= x => IntervalSide::Right,
                _ => IntervalSide::Inside,
            }
        } else {
            // interval is (max, min)
            match x {
                x if interval.0 <= x => IntervalSide::Left,
                x if x <= interval.1 => IntervalSide::Right,
                _ => IntervalSide::Inside,
            }
        }
    }
}

#[test]
fn switch_direction() {
    assert_eq!(switch::poly5((1.5, 2.0), 1.0).0, 0.0);
    assert_eq!(switch::poly5((1.5, 2.0), 2.5).0, 1.0);
    assert_eq!(switch::poly5((2.0, 1.5), 1.0).0, 1.0);
    assert_eq!(switch::poly5((2.0, 1.5), 2.5).0, 0.0);
}

#[test]
fn switch_middle() {
    assert_close!(switch::poly5((1.5, 2.0), 1.75).0, 0.5);
    assert_close!(switch::poly5((2.0, 1.5), 1.75).0, 0.5);
}

#[test]
fn switch_endpoint() {
    for _ in 0..10 {
        let a = uniform(-10.0, 10.0);
        let b = uniform(-10.0, 10.0);
        assert_eq!(switch::poly5((a, b), a).0, 0.0);
        assert_eq!(switch::poly5((a, b), b).0, 1.0);
    }
}

pub(crate) mod switch {
    #[allow(dead_code)]
    pub(crate) fn poly3(interval: (f64, f64), x: f64) -> (f64, f64) {
        super::switch(raw_poly3, interval, x)
    }

    pub(crate) fn poly5(interval: (f64, f64), x: f64) -> (f64, f64) {
        super::switch(raw_poly5, interval, x)
    }

    // Solution to:  y[0] = 0;  y'[0] = 0
    //               y[1] = 1;  y'[1] = 0
    //
    // If you use this as an interpolation function for `switch`, then there will
    // be very large errors in numerical derivatives computed by central difference
    // methods around `x=0` or `x=1`; a 5-point stencil will report a derivative
    // of about `step` at these points rather than `0`.
    pub(crate) fn raw_poly3(x: f64) -> (f64, f64) {
        let value = x*(3.0*x - 2.0*x*x);
        let d_x = 6.0*x*(1.0 - x);
        (value, d_x)
    }

    // Solution to:  y[0] = 0;  y'[0] = y''[0] = 0;
    //               y[1] = 1;  y'[1] = y''[1] = 0;
    //
    // If you use this as an interpolation function for `switch`, then there will
    // be moderately large errors in numerical derivatives computed by central
    // difference methods around `x=0` or `x=1`; a 5-point stencil will report a
    // derivative of about `10 * step**3` at these points rather than `0`.
    pub(crate) fn raw_poly5(x: f64) -> (f64, f64) {
        let value = (x*x*x)*(10.0 + x*(-15.0 + x*6.0));
        let d_x = (30.0*x*x)*(1.0 + x*(-2.0 + x));
        (value, d_x)
    }

    #[test]
    fn switch_num_deriv() {
        use super::*;
        use rsp2_minimize::numerical::{self};

        for _ in 0..20 {
            // an interval with non-unit length to check the scaling of the derivative,
            // but we'll only check x well inside the interval due to analytical errors
            // in the numerical derivative near the interval endpoints.
            let interval = (-1.0, 2.0);
            let x = uniform(0.0, 1.0);

            let (_, d_x) = switch::poly3(interval, x);
            assert_close!(
                rel=1e-10, abs=1e-10, d_x,
                numerical::slope(1e-3, None, x, |x| switch::poly3(interval, x).0),
            );

            let (_, d_x) = switch::poly5(interval, x);
            assert_close!(
                rel=1e-10, abs=1e-10, d_x,
                numerical::slope(1e-3, None, x, |x| switch::poly5(interval, x).0),
            );
        }
    }
}
