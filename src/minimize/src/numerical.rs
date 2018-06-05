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

/// Computation method for a numerical 1D derivative.
#[derive(Copy, Clone, Debug)]
pub enum DerivativeKind {
    /// Central difference approximation.
    CentralDifference,
}

impl Default for DerivativeKind {
    fn default() -> DerivativeKind {
        DerivativeKind::CentralDifference
    }
}

///// Get all binomial coefficients of order n, as double-precision floats.
/////
///// Beware the easy fencepost error: The output will have `len = n + 1`.
/////
///// Binomial coefficients grow extremely fast.  You will start seeing inexact results
///// as early as `n >= 57`, where they grow too large to fit in the floating point mantissa.
//fn binom_coefficients(n: usize) -> Vec<f64> {
//    let mut binoms = Vec::with_capacity(n + 1);
//    binoms.push(1.0); // binom coeffs of order zero
//
//    // beyond order zero
//    for order in 1..=n {
//        // There is a well-known relation between the binoms of order n and those of order n - 1
//        // (see Pascal's triangle)
//        binoms.push(1.0);
//        for i in 1..order {
//            binoms[i] += binoms[i + 1];
//        }
//    }
//    binoms
//}
//
//// I don't trust that garbage I just wrote
//#[test]
//fn test_binom_coefficients() {
//    // go go gadget pascal's triangle
//    assert_eq!(binom_coefficients(0), vec![1.0]);
//    assert_eq!(binom_coefficients(1), vec![1.0, 1.0]);
//    assert_eq!(binom_coefficients(2), vec![1.0, 2.0, 1.0]);
//    assert_eq!(binom_coefficients(3), vec![1.0, 3.0, 3.0, 1.0]);
//    assert_eq!(binom_coefficients(4), vec![1.0, 4.0, 6.0, 4.0, 1.0]);
//    assert_eq!(binom_coefficients(5), vec![1.0, 5.0, 10., 10., 5.0, 1.0]);
//}

/// Compute a numerical derivative using finite differences.
pub fn slope<E, F>(
    interval_width: f64,
    kind: Option<DerivativeKind>,
    point: f64,
    mut value_fn: F,
) -> Result<f64, E>
where
    F: FnMut(f64) -> Result<f64, E>,
{
    match kind.unwrap_or_default() {
        DerivativeKind::CentralDifference => {
            let val_plus = value_fn(point + 0.5 * interval_width)?;
            let val_minus = value_fn(point - 0.5 * interval_width)?;
            Ok((val_plus - val_minus) / interval_width)
        },
    }
}

/// Numerically compute a gradient.
///
/// This independently performs a slope check along each individual
/// axis of the input.  The number of function calls it makes will
/// be linearly proportional to the input size. This might be
/// prohibitively expensive!!
pub fn gradient<E, F>(
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
            let mut point = point.to_vec();
            ::numerical::slope(
                interval_width,
                Some(kind),
                center,
                |x| { point[i] = x; value_fn(&point) },
            )
        })
        .collect()
}
