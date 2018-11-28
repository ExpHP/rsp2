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

/// `slope` for functions that can fail.
pub fn try_slope<E, F>(
    interval_width: f64,
    kind: Option<DerivativeKind>,
    point: f64,
    mut value_fn: F,
) -> Result<f64, E>
where
    F: FnMut(f64) -> Result<f64, E>,
{
    #[inline(always)]
    fn dot(a: &[f64], b: &[f64]) -> f64 {
        zip_eq!(a, b).map(|(&a, &b)| a * b).sum()
    }

    // http://www.holoborodko.com/pavel/numerical-methods/numerical-derivative/central-differences/
    match kind.unwrap_or_default() {
        DerivativeKind::CentralDifference => {
            let values = [
                value_fn(point - 1.0 * interval_width)?,
                value_fn(point + 1.0 * interval_width)?,
            ];
            let coeffs = [-1.0, 1.0];
            let denom = 2.0 * interval_width;
            Ok(dot(&values, &coeffs) / denom)
        },

        DerivativeKind::Stencil(5) => {
            let values = [
                value_fn(point - 2.0 * interval_width)?,
                value_fn(point - 1.0 * interval_width)?,
                value_fn(point + 1.0 * interval_width)?,
                value_fn(point + 2.0 * interval_width)?,
            ];
            let coeffs = [1.0, -8.0, 8.0, -1.0];
            let denom = 12.0 * interval_width;
            Ok(dot(&values, &coeffs) / denom)
        },

        DerivativeKind::Stencil(7) => {
            let values = [
                value_fn(point - 3.0 * interval_width)?,
                value_fn(point - 2.0 * interval_width)?,
                value_fn(point - 1.0 * interval_width)?,
                value_fn(point + 1.0 * interval_width)?,
                value_fn(point + 2.0 * interval_width)?,
                value_fn(point + 3.0 * interval_width)?,
            ];
            let coeffs = [-1.0, 9.0, -45.0, 45.0, -9.0, 1.0];
            let denom = 60.0 * interval_width;
            Ok(dot(&values, &coeffs) / denom)
        },

        DerivativeKind::Stencil(9) => {
            let values = [
                value_fn(point - 4.0 * interval_width)?,
                value_fn(point - 3.0 * interval_width)?,
                value_fn(point - 2.0 * interval_width)?,
                value_fn(point - 1.0 * interval_width)?,
                value_fn(point + 1.0 * interval_width)?,
                value_fn(point + 2.0 * interval_width)?,
                value_fn(point + 3.0 * interval_width)?,
                value_fn(point + 4.0 * interval_width)?,
            ];
            let coeffs = [3.0, -32.0, 168.0, -672.0, 672.0, -168.0, 32.0, -3.0];
            let denom = 840.0 * interval_width;
            Ok(dot(&values, &coeffs) / denom)
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

#[test]
fn num_diff() {
    for n in vec![3, 5, 7, 9u32] {
        for _ in 0..10 {
            // n-point stencil is exact for polynomials up to order n-1
            let poly = {
                std::iter::repeat_with(|| uniform(-2.0, 2.0))
                    .take(n as usize) // order n-1 means n coeffs
                    .collect::<Vec<_>>()
            };
            let x = uniform(-10.0, 10.0);

            let expected = polyval_dec(polyder_dec(poly.iter().cloned()), x);
            let actual = slope(1e-1, Some(DerivativeKind::Stencil(n)), x, |x| {
                polyval_dec(poly.iter().cloned(), x)
            });
            // NOTE: 1e-10 fails at a rate of around ~1 in 1e6
            assert_close!(abs=1e-9, rel=1e-9, expected, actual, "{}-point", n);
        }
    }
}

#[cfg(test)]
fn uniform(a: f64, b: f64) -> f64 { rand::random::<f64>() * (b - a) + a }

#[cfg(test)]
fn polyder_dec(
    coeffs: impl DoubleEndedIterator<Item=f64> + ExactSizeIterator + Clone,
) -> impl DoubleEndedIterator<Item=f64> + ExactSizeIterator + Clone
{ coeffs.rev().skip(1).enumerate().map(|(n, x)| (n + 1) as f64 * x).rev() }

#[cfg(test)]
#[inline(always)]
fn polyval_dec(coeffs: impl Iterator<Item=f64>, x: f64) -> f64 {
    coeffs.fold(0.0, |acc, c| acc * x + c)
}
