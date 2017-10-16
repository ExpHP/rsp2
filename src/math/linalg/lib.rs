extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::Array;

/// Fit a polynomial.
///
/// Order is the maximum fitted coefficient of x.
/// That is, it is one less than the size of the returned vector.
pub fn polyfit(order: u32, data: &[(f64, f64)]) -> Vec<f64> {
    #![allow(bad_style)]
    use ::ndarray_linalg::Solve;

    // NOTE: this is probably a fair bit weaker than numpy's implementation.
    // We also do not obtain any useful measure of error.
    //
    // We just solve `X a = y` by pseudo inverse.

    let X = Array::from_shape_fn([data.len(), order as usize + 1], |(i, k)| data[i].0.powi(k as _));
    let y = Array::from_vec(data.iter().map(|&(_, y)| y).collect());
    let yp = X.t().dot(&y);
    let XTX = X.t().dot(&X);

    XTX.solve_into(yp).unwrap().to_vec()
}

