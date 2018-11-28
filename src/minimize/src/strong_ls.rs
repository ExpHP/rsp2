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

//! A simple-ish linesearch for a point that satisfies the strong Wolfe conditions.
//!
//! Based on code originally written by Colin Daniels.  I can't recall if that
//! in turn was based off of anything.

use ::either::{Either, Left, Right};

#[derive(Debug, Clone, PartialEq)]
pub struct Settings {
    pub iteration_limit: u32,
    pub armijo_coeff: f64,
    pub curvature_coeff: f64,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            iteration_limit: 8,
            armijo_coeff: 1e-4,
            curvature_coeff: 1e-1,
        }
    }
}

impl Settings { pub fn new() -> Settings { Default::default() } }

impl Settings { pub fn validate(&self) { /* TODO */ } }

/// Holds information about linesearch boundaries
#[derive(Debug, Copy, Clone)]
struct Bound { alpha: f64, value: f64, slope: f64 }

pub trait DiffFn1D<E>: FnMut(f64) -> Result<(f64, f64), E> { }
impl<E, F> DiffFn1D<E> for F
where F: FnMut(f64) -> Result<(f64, f64), E> { }

/// `linesearch` error type
#[derive(Debug, Fail)]
#[fail(display = "{}", kind)]
pub struct LinesearchError {
    backtrace: ::failure::Backtrace,
    kind: ErrorKind,
}

#[derive(Debug, Fail)]
pub enum ErrorKind {
    #[fail(display = "Initial slope was positive: {}", slope)]
    Uphill { slope: f64 },

    #[doc(hidden)]
    #[fail(display = "impossible!")]
    _Hidden,
}

impl From<ErrorKind> for LinesearchError {
    fn from(kind: ErrorKind) -> Self {
        let backtrace = ::failure::Backtrace::new();
        LinesearchError { backtrace, kind }
    }
}

pub fn linesearch<E, F>(
    settings: &Settings,
    mut alpha: f64,
    mut compute: F,
) -> Result<f64, Either<LinesearchError, E>>
where F: FnMut(f64) -> Result<(f64, f64), E>,
{
    let mut compute = |alpha| compute(alpha).map_err(Right);

    let (mut value, mut slope) = compute(0.0)?;

    assert!(alpha > 0.0, "non-positive initial alpha: {}", alpha);
    if slope > 0.0 {
        return Err(Left(ErrorKind::Uphill { slope }.into()));
    }

    let initial_value = value;

    // Right hand side quantities for the wolfe condition linesearch.
    // - sufficient decrease
    let armijo = settings.armijo_coeff * slope.abs();
    // - the curvature condition
    let curvature = settings.curvature_coeff * slope.abs();

    // lower and upper bounds for minimum finding
    // (hard lower bound, soft upper bound)
    let mut low = Bound { alpha: 0.0, value, slope };
    let mut high = Bound { alpha: 0.0, value: 0.0, slope: 0.0 };

    // running minimum, initialize with the information from alpha = 0
    let mut min_point = (value, 0.0);

    for _ in 0..settings.iteration_limit {
        // check for errors in alpha
        if !alpha.is_finite() {
            return Ok(min_point.1)
        };

        // update value and slope
        let tup = compute(alpha)?;
        value = tup.0;
        slope = tup.1;

        // check the wolfe conditions
        if value <= initial_value - alpha * armijo { // armijo
            if slope.abs() <= curvature { // curvature
                return Ok(alpha);
            }
        }

        // update running minimum
        if value < min_point.0 {
            min_point = (value, alpha);
        }

        // update the bounding interval for the minimum
        if value < low.value && slope < 0.0 && alpha < high.alpha {
            low = Bound {alpha, value, slope};
        } else {
            high = Bound {alpha, value, slope};
        }

        alpha = guess_min(low, high);
    }

    // return the alpha that gave us the lowest value
    // note: This could be zero!
    Ok(min_point.1)
}

/// guess a minimum from two input bounds
fn guess_min(low: Bound, high: Bound) -> f64
{
    // NOTE: I was expecting unit tests to reveal that this was accepting many
    //       absurd output values from cubic_min, but so far I haven't seen
    //       this to actually be the case.
    // TODO: If I'm that adamant about it, I should check with the lammps potential,
    //        using those structures that tend to produce linesearch failures.
    match cubic_min(low, high) {
        a if a.is_normal() => a,
        _ => quadratic_min(low, high),
    }
}

/// Get the analytical minimum of a cubic fit to two input bounds.
/// Becomes numerically unstable as the third-order coefficient approaches 0.
// FIXME: In my tests I'm not actually seeing that this becomes particularly unstable;
//        it tends to produce either NaN, or something with about the same magnitude
//        as the quadratic minimum x value. (I was expecting to see extreme-but-normal values appear)
fn cubic_min(low: Bound, high: Bound) -> f64
{
    let d_value = high.value - low.value;
    let d_alpha = high.alpha - low.alpha;
    let d_alpha2 = d_alpha * d_alpha;
    let d_alpha3 = d_alpha * d_alpha2;

    // get cubic coefficients
    // f(x) = ax^3 + bx^2 + cx + d
    let a = ((high.slope + low.slope) * d_alpha - 2.0 * d_value) / d_alpha3;
    let b = (3.0 * d_value - (high.slope + 2.0 * low.slope) * d_alpha) / d_alpha2;
    let c = low.slope;

    // The derivative of f has two roots.
    // For real a, b, c, the + root is always the minimum
    low.alpha + ((b * b - 3.0 * a * c).sqrt() - b) / (3.0 * a)
}

/// get the analytical mininum of a quadratic fit to two input bounds
fn quadratic_min(low: Bound, high: Bound) -> f64
{
    let d_value = high.value - low.value;
    let d_alpha = high.alpha - low.alpha;
    let d_alpha2 = d_alpha * d_alpha;

    // get quadratic coefficients
    // f(x) = bx^2 + cx + d
    let b = (d_value - low.slope * d_alpha) / d_alpha2;
    let c = low.slope;

    // The derivative of f has a single root
    low.alpha - c / (2.0 * b)
}

#[cfg(test)]
mod tests {
    use super::Bound;
    use super::{linesearch, Settings, };

    use ::test::one_dee::prelude::*;
    use ::test::one_dee::Polynomial;

    fn init_logger() {
        let _ = ::env_logger::try_init();
    }

    #[derive(Debug,Copy,Clone,Hash,PartialEq,Eq,PartialOrd,Ord)]
    enum Never {}

    fn diff_fn(
        f: impl Differentiable1d,
    ) -> impl Fn(f64) -> Result<(f64, f64), Never> {
        let deriv = f.derivative();
        move |x| Ok::<_,Never>((f.evaluate(x), deriv.evaluate(x)))
    }

    fn bound_fn(
        f: impl Differentiable1d,
    ) -> impl Fn(f64) -> Bound {
        let deriv = f.derivative();
        move |x| Bound {
            alpha: x,
            value: f.evaluate(x),
            slope: deriv.evaluate(x),
        }
    }

    #[test]
    fn cubic_min_analytic() {
        {
            // Third coefficient positive, second coefficient positive.
            let poly = Polynomial::from_coeffs(&[1000.0, -800.0, 52.0, 3.0]);
            let root1 =   5.27987138188865; // local minimum
            let root2 = -16.8354269374442;  // local maximum
            assert_close!(abs=1e-8, poly.derivative().evaluate(root1), 0.0);
            assert_close!(abs=1e-8, poly.derivative().evaluate(root2), 0.0);

            // simple; facing towards minimum.
            let get_bound = bound_fn(poly.clone());
            assert_close!(root1, super::cubic_min(get_bound(0.0), get_bound(1.0)));
            // low.alpha not equal to 0, d_alpha not equal to 1
            assert_close!(root1, super::cubic_min(get_bound(4.5), get_bound(1.25)));

            // flipping x, so that minimum is the lesser root
            let get_bound = bound_fn(poly.scale_x(-1.0));
            assert_close!(-root1, super::cubic_min(get_bound(0.0), get_bound(1.0)));

            // both extrema in same direction from x=0
            let get_bound = bound_fn(poly.recenter(-20.0));
            assert_close!(20.0 + root1, super::cubic_min(get_bound(0.0), get_bound(1.0)));

            // Third coefficient negative, second coefficient negative
            let get_bound = bound_fn(poly.scale_y(-1.0));
            assert_close!(root2, super::cubic_min(get_bound(0.0), get_bound(1.0)));
        }

        {
            let poly = Polynomial::from_coeffs(&[1000.0, 800.0, 52.0, -3.0]);
            let root1 = -5.27987138188865; // local minimum
            let root2 = 16.8354269374442;  // local maximum
            assert_close!(abs=1e-10, poly.derivative().evaluate(root1), 0.0);
            assert_close!(abs=1e-10, poly.derivative().evaluate(root2), 0.0);

            // Third coefficient negative, second coefficient positive
            let get_bound = bound_fn(poly.clone());
            assert_close!(root1, super::cubic_min(get_bound(0.0), get_bound(1.0)));

            // Third coefficient positive, second coefficient negative
            let get_bound = bound_fn(poly.scale_y(-1.0));
            assert_close!(root2, super::cubic_min(get_bound(0.0), get_bound(1.0)));
        }
    }

    #[test]
    fn quadratic_min_analytic() {
        let poly = Polynomial::from_coeffs(&[1000.0, -800.0, -52.0]);
        let root = -7.69230769230769; // local minimum
        assert_close!(abs=1e-10, poly.derivative().evaluate(root), 0.0);

        // simple
        let get_bound = bound_fn(poly);
        assert_close!(root, super::quadratic_min(get_bound(0.0), get_bound(1.0)));
        // low.alpha not equal to 0, d_alpha not equal to 1
        assert_close!(root, super::quadratic_min(get_bound(4.5), get_bound(1.25)));
    }

    #[test]
    fn guess_min_analytic_3rd_order() {
        // NOTE: This test is subject to spurious failures due to difficulty
        //       in selecting decent tolerances.
        //       It has been tested to fail less than once per two billion loop iterations.

        for _ in 0..20 {
        // for _ in 0..2_000_000_000 { // NOTE: for testing spurious failures
            let coeffs: [f64; 4];
            loop {
                let adjust = |x| 10.0 * (x - 0.5); // transform [0.0, 1.0] into [-5.0, 5.0]
                let (c0,c1,c2,c3) = ::rand::random::<(f64, f64, f64, f64)>();
                let (c0,c1,c2,c3) = (adjust(c0), adjust(c1), adjust(c2), adjust(c3));
                // reduce spurious failures
                if c1.abs() < 1e-3 { continue; }
                if c3.abs() < 1e-3 { continue; }
                // check discriminant for real roots
                if c2 * c2 - 3.0 * c3 * c1 > 0.0 {
                    coeffs = [c0, c1, c2, c3];
                    break;
                }
            }
            let poly = Polynomial::from_coeffs(&coeffs);
            let get_bound = bound_fn(poly.clone());
            let guess = super::guess_min(get_bound(0.0), get_bound(1.0));

            // looks minimal?
            assert_close!(abs=1e-4, poly.derivative().evaluate(guess), 0.0, "{:?}", poly);
            assert!(false
                || poly.evaluate(guess) <= poly.evaluate(guess * (1.0 - 1e-2))
                || poly.evaluate(guess) <= poly.evaluate(guess * (1.0 - 1e-5))
                , "{:?}", poly,
                );
            assert!(false
                || poly.evaluate(guess) <= poly.evaluate(guess * (1.0 + 1e-2))
                || poly.evaluate(guess) <= poly.evaluate(guess * (1.0 + 1e-5))
                , "{:?}", poly,
                );
        }
    }

    // FIXME: need a test for checking that guess_min is capable of falling back
    //        to the 2nd order approx when the 3rd order approx is unstable.
    //
    //        I tried writing a test which uses randomly generated degree-2
    //        polynomials, but for those cases it seems that the third order
    //        approx always either produces NaN, or else something with a
    //        reasonable-looking magnitude that is hard to differentiate from
    //        the actual second order approx (without computing both and
    //        comparing the potential, at least).
    //
    //        In other words, the current behavior IS suboptimal for 2nd order
    //        polynomials, but perhaps only moderately; and I don't think
    //        there's really anything we can do about it.

    #[test]
    fn start_on_maximum() {
        init_logger();

        // Here is a function which has a maximum located precisely at x == 0.
        // When evaluated at x == 0, the slope will come out exactly equal to 0.
        //
        // Despite these hostile circumstances, linesearch should not have much of
        // an issue escaping from this maximum and locating a better point.
        // The only issue is that it may have trouble triggering the wolfe conditions.
        //
        // This is where a quadratic-only approximation would fail miserably.
        let poly = Polynomial::from_coeffs(&[0.0, 0.0, -1.0, 0.0, 1.0]);
        // it shouldn't produce an error
        let out = linesearch(&Settings::default(), 0.125, diff_fn(poly.clone())).unwrap();
        assert!(!out.is_nan());
        // it should be able to find at least one point better than those we gave it
        assert!(poly.evaluate(out) < poly.evaluate(0.125));
    }
}
