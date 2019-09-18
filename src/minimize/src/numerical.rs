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

//! Utilities for numeric differentiation.
//!
//! These are publicly exported from rsp2_minimize because it is
//! likely that other crates which use rsp2_minimize may want to
//! use these to debug the potentials they are giving to this crate.

/// Approximation method for a numerical 1D derivative.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DerivativeKind {
    /// n-point stencil. `n` must be odd. Only implemented for `n = 3, 5, 7, 9`.
    Stencil(u32),
}

impl DerivativeKind {
    /// Alias for `DerivativeKind::Stencil(3)`.
    #[allow(bad_style)]
    pub const CentralDifference: Self = DerivativeKind::Stencil(3);
}

impl Default for DerivativeKind {
    fn default() -> DerivativeKind {
        DerivativeKind::Stencil(5)
    }
}

enum Never {}

/// Compute a numerical derivative using finite differences.
pub fn slope(
    interval_width: f64,
    kind: Option<DerivativeKind>,
    point: f64,
    mut value_fn: impl FnMut(f64) -> f64,
) -> f64 {
    try_slope::<Never, _>(interval_width, kind, point, |x| Ok(value_fn(x)))
        .unwrap_or_else(|e| match e {})
}

/// Compute a numerical second derivative using finite differences.
pub fn diff_2(
    interval_width: f64,
    kind: Option<DerivativeKind>,
    point: f64,
    mut value_fn: impl FnMut(f64) -> f64,
) -> f64 {
    try_diff_2::<Never, _>(interval_width, kind, point, |x| Ok(value_fn(x)))
        .unwrap_or_else(|e| match e {})
}

#[inline(always)]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    zip_eq!(a, b).map(|(&a, &b)| a * b).sum()
}

macro_rules! stencil_sum {
    ($value_fn:expr, $point:expr, $step:expr, [
        $((offset: $sign:tt $offset:expr, coeff: $(+)?$coeff:expr),)*
    ]) => {{
        let mut value_fn = $value_fn;
        let point = $point;
        let step = $step;
        let values = [
            $(value_fn(point $sign $offset * step)?,)+
        ];
        let coeffs = [$($coeff),*];
        dot(&values, &coeffs)
    }};
}

/// `slope` for functions that can fail.
pub fn try_slope<E, F>(
    step: f64,
    kind: Option<DerivativeKind>,
    point: f64,
    value_fn: F,
) -> Result<f64, E>
where
    F: FnMut(f64) -> Result<f64, E>,
{
    // http://www.holoborodko.com/pavel/numerical-methods/numerical-derivative/central-differences/
    match kind.unwrap_or_default() {
        DerivativeKind::Stencil(3) => {
            let numer = stencil_sum!(value_fn, point, step, [
                (offset: -1.0, coeff: -1.0),
                (offset: +1.0, coeff: +1.0),
            ]);
            let denom = 2.0 * step;
            Ok(numer / denom)
        },

        DerivativeKind::Stencil(5) => {
            let numer = stencil_sum!(value_fn, point, step, [
                (offset: -2.0, coeff: +1.0),
                (offset: -1.0, coeff: -8.0),
                (offset: +1.0, coeff: +8.0),
                (offset: +2.0, coeff: -1.0),
            ]);
            let denom = 12.0 * step;
            Ok(numer / denom)
        },

        DerivativeKind::Stencil(7) => {
            let numer = stencil_sum!(value_fn, point, step, [
                (offset: -3.0, coeff: -1.0),
                (offset: -2.0, coeff: +9.0),
                (offset: -1.0, coeff: -45.0),
                (offset: +1.0, coeff: +45.0),
                (offset: +2.0, coeff: -9.0),
                (offset: +3.0, coeff: +1.0),
            ]);
            let denom = 60.0 * step;
            Ok(numer / denom)
        },

        DerivativeKind::Stencil(9) => {
            let numer = stencil_sum!(value_fn, point, step, [
                (offset: -4.0, coeff: +3.0),
                (offset: -3.0, coeff: -32.0),
                (offset: -2.0, coeff: +168.0),
                (offset: -1.0, coeff: -672.0),
                (offset: +1.0, coeff: +672.0),
                (offset: +2.0, coeff: -168.0),
                (offset: +3.0, coeff: +32.0),
                (offset: +4.0, coeff: -3.0),
            ]);
            let denom = 840.0 * step;
            Ok(numer / denom)
        },

        DerivativeKind::Stencil(n@0) |
        DerivativeKind::Stencil(n@1) |
        DerivativeKind::Stencil(n) if n % 2 == 0 => {
            panic!("{}-point stencil does not exist", n);
        },

        DerivativeKind::Stencil(n) => {
            panic!("{}-point stencil is not implemented", n);
        },
    }
}

/// `diff_2` for functions that can fail.
pub fn try_diff_2<E, F>(
    step: f64,
    kind: Option<DerivativeKind>,
    point: f64,
    value_fn: F,
) -> Result<f64, E>
where
    F: FnMut(f64) -> Result<f64, E>,
{
    // http://www.holoborodko.com/pavel/numerical-methods/numerical-derivative/central-differences/#comment-1719
    match kind.unwrap_or_default() {
        DerivativeKind::Stencil(3) => {
            let numer = stencil_sum!(value_fn, point, step, [
                (offset: -1.0, coeff: +1.0),
                (offset: -0.0, coeff: -2.0),
                (offset: +1.0, coeff: +1.0),
            ]);
            let denom = step * step;
            Ok(numer / denom)
        },

        DerivativeKind::Stencil(5) => {
            let numer = stencil_sum!(value_fn, point, step, [
                (offset: -2.0, coeff: -1.0),
                (offset: -1.0, coeff: +16.0),
                (offset: -0.0, coeff: -30.0),
                (offset: +1.0, coeff: +16.0),
                (offset: +2.0, coeff: -1.0),
            ]);
            let denom = 12.0 * (step * step);
            Ok(numer / denom)
        },

        DerivativeKind::Stencil(7) => {
            let numer = stencil_sum!(value_fn, point, step, [
                (offset: -3.0, coeff: +2.0),
                (offset: -2.0, coeff: -27.0),
                (offset: -1.0, coeff: +270.0),
                (offset: -0.0, coeff: -490.0),
                (offset: +1.0, coeff: +270.0),
                (offset: +2.0, coeff: -27.0),
                (offset: +3.0, coeff: +2.0),
            ]);
            let denom = 180.0 * (step * step);
            Ok(numer / denom)
        },

        DerivativeKind::Stencil(n@0) |
        DerivativeKind::Stencil(n@1) |
        DerivativeKind::Stencil(n) if n % 2 == 0 => {
            panic!("{}-point stencil does not exist", n);
        },

        DerivativeKind::Stencil(n) => {
            panic!("{}-point stencil second derivative is not implemented", n);
        },
    }
}

/// Numerically compute a gradient.
///
/// This independently performs a slope check along each individual
/// axis of the input.  The number of function calls it makes will
/// be linearly proportional to the input size. This might be
/// prohibitively expensive!!
pub fn gradient(
    interval_width: f64,
    kind: Option<DerivativeKind>,
    point: &[f64],
    mut value_fn: impl FnMut(&[f64]) -> f64,
) -> Vec<f64> {
    try_gradient::<Never, _>(interval_width, kind, point, |x| Ok(value_fn(x)))
        .unwrap_or_else(|e| match e {})
}

/// `gradient` for functions that can fail.
pub fn try_gradient<E, F>(
    interval_width: f64,
    kind: Option<DerivativeKind>,
    point: &[f64],
    mut value_fn: F,
) -> Result<Vec<f64>, E>
where
    F: FnMut(&[f64]) -> Result<f64, E>,
{
    let kind = kind.unwrap_or_default();
    point.iter().enumerate()
        .map(|(i, &center)| {
            let mut point = point.to_vec(); // reset modifications
            try_slope(
                interval_width,
                Some(kind),
                center,
                |x| { point[i] = x; value_fn(&point) },
            )
        })
        .collect()
}

//---------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    use crate::test::one_dee::{Differentiable1d, Polynomial};
    use crate::util::random::uniform;

    #[test]
    fn num_diff() {
        for n in vec![3, 5, 7, 9] {
            for _ in 0..10 {
                // n-point stencil is exact for polynomials up to order n-1
                let poly = Polynomial::random(n - 1, 2.0);
                let x = uniform(-10.0, 10.0);

                let expected = poly.derivative().evaluate(x);
                let actual = slope(1e-1, Some(DerivativeKind::Stencil(n)), x, |x| poly.evaluate(x));
                // NOTE: 1e-10 fails at a rate of around ~1 in 1e6
                assert_close!(abs=1e-9, rel=1e-9, expected, actual, "{}-point", n);
            }
        }
    }

    #[test]
    fn num_diff_2() {
        for n in vec![3, 5, 7] {
            for _ in 0..10 {
                // n-point stencil is exact for polynomials up to order n-1
                let poly = Polynomial::random(n - 1, 2.0);
                let x = uniform(-10.0, 10.0);

                let expected = poly.derivative().derivative().evaluate(x);
                let actual = diff_2(1e-1, Some(DerivativeKind::Stencil(n)), x, |x| poly.evaluate(x));
                // NOTE: 1e-9 fails at a rate of around ~1 in 1e7
                assert_close!(abs=1e-8, rel=1e-8, expected, actual, "{}-point", n);
            }
        }
    }
}
