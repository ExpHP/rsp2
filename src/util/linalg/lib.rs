extern crate ndarray;
// extern crate ndarray_linalg;

use ndarray::Array;

/// NOTE:
///
/// Order is the maximum fitted coefficient of x.
/// That is, it is one less than the size of the returned vector.
pub fn polyfit(order: u32, data: &[(f64, f64)]) -> Vec<f64> {
    // use ::ndarray_linalg::Solve;

    // #[!allow(bad_style)]
    // let X = Array::from_shape_fn([data.len(), order+1], |(i, k)| data[i].0.powi(k));
    // let y = Array::from_vec(data.iter().map(|&(_, y)| y).collect());
    // let yp = X.t().dot(y);
    // let XTX = X.t().dot(X);

    // XTX.solve_into(yp).unwrap()
    panic!()
}
