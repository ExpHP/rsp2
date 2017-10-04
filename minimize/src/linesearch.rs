use ::ordered_float::NotNaN; // sigh


pub struct Settings {
    pub iteration_limit: u32,
    pub no_force_tol: f64,
    pub weak_force_tol: f64,
    pub armijo_threshold: f64,
    pub curvature_threshold: f64,
}

impl Settings { pub fn new() -> Settings { Default::default() } }
impl Default for Settings {
    fn default() -> Settings {
        Settings {
            iteration_limit: 8,
            no_force_tol: 0.0,
            weak_force_tol: 0.0,
            armijo_threshold: 1e-4,
            curvature_threshold: 1e-1,
        }
    }
}

/// Holds information about linesearch boundaries
#[derive(Debug, Copy, Clone)]
struct Bound { alpha: f64, value: f64, slope: f64 }

impl Bound {
    /// Produce an approximate value by integrating the slope
    /// from a reference point.
    fn with_value_approx_from(self, from: Bound) -> Bound {
        // assume slope is linear w.r.t. alpha, solve for the
        // coefficients and integrate. Intuitively, this comes out
        // equivalent to taking the mean slope over the whole interval.
        let d_alpha = self.alpha - from.alpha;
        let d_value = 0.5 * (self.slope + from.slope) * d_alpha;
        Bound {
            alpha: self.alpha,
            value: from.value + d_value,
            slope: self.slope,
        }
    }
}

type ResultWithControlFlow<V, R, E> = Result<V, Result<R, E>>;
fn resolve_early_returns<R, E>(r: ResultWithControlFlow<R, R, E>) -> Result<R, E> {
    match r {
        Ok(x) => Ok(x), // standard return
        Err(Ok(x)) => Ok(x), // early return, masquerading as an error
        Err(Err(e)) => Err(e), // legitimate error
    }
}

pub trait DiffFn1D<E>: FnMut(f64) -> Result<(f64, f64), E> { }
impl<E, F> DiffFn1D<E> for F
where F: FnMut(f64) -> Result<(f64, f64), E> { }

/// `linesearch` error type
#[derive(Debug, Clone)]
pub enum Error<E> {
    /// General variant for unrecoverable errors that are not worth panicking on,
    /// and that are probably not worth matching on the receiver.
    Generic(String),

    /// Slope at the initial point was positive.
    Uphill,

    /// Failed to find any point that was better than the initial point.
    ///
    /// Since positive slope produces a different error, this may
    /// indicate disagreement between potential and slope.
    NoImprovement,

    /// Linesearch produced a guess alpha which is not finite.
    ///
    /// This might happen if two points are sampled which both have a slope of
    /// exactly zero.
    NonFiniteAlpha,

    /// The function produced an error.
    ComputeError(E),
}

impl<E> Error<E> {
    fn string(s: &str) -> Error<E> { Error::Generic(s.to_string()) }
}

pub fn linesearch<E, F>(
    settings: &Settings,
    initial_alpha: f64,
    compute: F,
) -> Result<f64, Error<E>>
where F: FnMut(f64) -> Result<(f64, f64), E>,
{
    let ugly = do_linesearch(settings, (0.0, initial_alpha), compute);
    resolve_early_returns(ugly)
}

struct NoEarlyExit;

pub fn really_old<E, F>(
    settings: &Settings,
    mut alpha: f64,
    mut compute: F,
) -> Result<f64, E>
where F: FnMut(f64) -> Result<(f64, f64), E>,
{
    let (mut value, mut slope) = compute(0.0)?;

    if alpha <= 0.0
        {panic!("linesearch initial alpha <= 0")};
    if slope > 0.0
        {panic!("uphill linesearch")};


    let initial_value = value;

    // Right hand side quantities for the wolfe condition linesearch.
    // - sufficient decrease
    let armijo = settings.armijo_threshold * slope.abs();
    // - the curvature condition
    let curvature = settings.curvature_threshold * slope.abs();

    // lower and upper bounds for minimum finding
    // (hard lower bound, soft upper bound)
    let mut low = Bound {alpha: 0.0, value, slope};
    let mut high = Bound {alpha: 0.0, value: 0.0, slope: 0.0};

    // running minimum, initialize with the information from alpha = 0
    let mut min_point = (value, 0.0);

    for iteration in 0..settings.iteration_limit
    {
        // check for errors in alpha
        if !alpha.is_finite()
            {return Ok(min_point.1)};

        // update value and slope
        let tup = compute(alpha)?;
        value = tup.0;
        slope = tup.1;

        // check the wolfe conditions
        if value <= initial_value - alpha * armijo {    // armijo
            if slope.abs() <= curvature { // curvature
                return Ok(alpha)}};

        // update running minimum
        if value < min_point.0 {
            min_point = (value, alpha);
        }

        // update the bounding interval for the minimum
        if value < low.value && slope < 0.0 && alpha < high.alpha
        {
            low = Bound {alpha, value, slope};
        } else {
            high = Bound {alpha, value, slope};
        }

        // get the new alpha
        let minimum = cubic_min(low, high);
        if minimum.is_normal() {
            alpha = minimum;
        } else {
            alpha = quadratic_min(low, high);
        }
    }

    // return the alpha that gave us the lowest value
    // note: This could be zero!
    return Ok(min_point.1);
}

// -------------------------------------------------------------------------- //
// Expressing this algorithm in a DRY fashion is an exercise in humility.
// You'll just have to accept these two functions with a not-entirely-clean
// separation of responsibilities.

// The heart of the matter is that the first few computations require special
// treatment to "get things started", but we still want them to be eligible
// for satisfying the exit conditions (this affects control flow). The former
// requirement begs for an internal iterator over positions, while anything
// involving control flow is usually best served by external iteration.
//
// Ultimately, an internal iterator is used, with a callback that sometimes
// slips successful early-return values into an Err, causing the iterator
// to short-circuit.

// First, the client of the internal iterator, which:
// * Defines how bounds are computed
// * Decides when to use weak-force methods
// * Implements the Wolfe conditions
fn do_linesearch<E, F>(
    settings: &Settings,
    interval: (f64, f64),
    mut compute: F,
) -> ResultWithControlFlow<f64, f64, Error<E>>
where F: FnMut(f64) -> Result<(f64, f64), E>,
{
    // Special case equal points because they are dangerous to curve fitting.
    if interval.0 == interval.1 { return Ok(interval.0); }

    let mut compute_bound = move |alpha| -> ResultWithControlFlow<Bound, f64, Error<E>> {
        let result = compute(alpha);
        // (legitimate errors will be Err(Err(e)))
        let result = result.map_err(|e| Err(Error::ComputeError(e)));
        let (value, slope) = result?;
        Ok(Bound { alpha, value, slope })
    };

    // Data at interval endpoints.
    // There is a notion of interval.0 being the "starting point,"
    //  regardless of which has the greater alpha.
    // This is the bound used for exit criteria.
    let from = compute_bound(interval.0)?;
    let to = compute_bound(interval.1)?;

    // How strong is the force?
    let is_weak = {
        let force_scale = f64::max(from.slope.abs(), to.slope.abs());
        let  work_scale = force_scale * (to.alpha - from.alpha).abs();
        let value_scale = f64::min(from.value.abs(), to.value.abs());

        // Case of no force; specifically prefer to return the starting point.
        if force_scale <= settings.no_force_tol {
            return Ok(from.alpha);
        }
        // "Weak force" when there is a force, but it is so weak relative to value
        // that we cannot reliably compare and subtract values.
        work_scale < settings.weak_force_tol * value_scale
    };

    // Redefine value to be zero at `from.alpha` for weak force.
    // This is accomplished in part through a function that may adjusts a bound's
    //  value to an approximate value relative to another bound.
    // This function should be idempotent, in case we want to be able to re-rescale
    //  a bound against a better reference point to improve the approximation.
    let from = if is_weak { Bound { value: 0.0, ..from } } else { from };
    let mut rescale: Box<FnMut(Bound, Bound) -> Bound> = {
        if is_weak {
            // Weak force; the best we can do is approximate value by using slope.
            Box::new(|bound: Bound, other| bound.with_value_approx_from(other))
        } else {
            // Strong force; we can trust the values themselves.
            // (NOTE: while consistency would be nice, we can't subtract the
            //        initial value here as that would fail to be idempotent)
            Box::new(|bound, _| bound)
        }
    };
    let to = rescale(to, from);

    // From this point onwards we will frequently be interested in whether a
    // computed bound is "good enough" to exit. Not only that, but once it
    // happens, we'll want to actually exit!
    //
    // `Err(Ok(x))` is dedicated to this purpose, allowing these early returns
    // to be handled through Result machinery. Please don't hurt me.
    let exit_if_good_enough = {
        let armijo = settings.armijo_threshold * from.slope.abs();
        let curvature_tol = settings.curvature_threshold * from.slope.abs();
        let from_alpha = from.alpha;
        let from_value = from.value;

        move |Bound { alpha, value, slope }| {
            let good =
                value < from_value - armijo * (alpha - from_alpha).abs()
                && slope.abs() < curvature_tol;

            match good {
                true => Err(Ok(alpha)), // early return
                false => Ok(()),
            }
        }
    };

    // The second computed point may very well be good enough; consider it.
    exit_if_good_enough(to)?;

    //-------------------------------------------------------------------------
    // From this point onward, we no longer care about which side of
    // the interval is the "start".  Resolve them into min/max.
    let (lo, hi) = if from.alpha < to.alpha { (from, to) } else { (to, from) };

    let each_computation = match is_weak {
        true => each_computation_weak,
        false => each_computation_strong,
    };

    each_computation(
        settings,
        lo, hi,
        |alpha| {
            let bound = compute_bound(alpha)?;
            exit_if_good_enough(bound)?;
            Ok(bound)
        },
    )
}

// Next, the coroutine-ish thing itself.
// Responsibilities are:
// * deciding which points to compute
//   (not including the initial computation at alpha=0)
// * deciding how to update the bounds of the search interval
fn each_computation_strong<R, E, F>(
    settings: &Settings,
    mut lo: Bound,
    mut hi: Bound,
    mut function: F,
) -> ResultWithControlFlow<f64, R, Error<E>>
where
  F: FnMut(f64) -> ResultWithControlFlow<Bound, R, Error<E>>,
{
    assert!(lo.value.is_finite());
    assert!(hi.value.is_finite());

    let mut previous_states = ::std::collections::HashMap::new();
    let mut history = vec![lo, hi];
    'a: loop { break {
        for iteration in 0..settings.iteration_limit {
            // Invariant: each bound has a unique alpha.
            // This aleviates some concerns of stability in guesses for new alphas.
            assert!(lo.alpha < hi.alpha);
            {
                let key = (NotNaN::new(lo.alpha).unwrap(), NotNaN::new(hi.alpha).unwrap());
                if let Some(old_iter) = previous_states.insert(key, iteration) {
                    debug!("Exiting after deja vu! (Iterations {}, {}) {:?}", old_iter+1, iteration+1, [lo, hi]);
                    break 'a;
                }
            }

            if lo.slope > 0.0 && hi.slope < 0.0 {
                debug!("Straddling a maximum! {:?}", [lo, hi]);
            }

            let alpha = guess_min(lo, hi);
            if !alpha.is_finite() {
                warn!("Exiting after non-finite alpha. ({:?} -> {})", [lo, hi], alpha);
                break 'a;
            }

            #[derive(Debug,Copy,Clone,PartialEq,PartialOrd,Eq,Ord)]
            enum L { Before, Between, After }
            let loc = {
                if [lo.alpha, hi.alpha].contains(&alpha) {
                    debug!("Exiting after repeating a bound. ({:?} -> {})", [lo, hi], alpha);
                    // This *could* be thanks to uncannily good convergence, together with
                    // something else (e.g. starting on a maximum) making the Wolfe conditions
                    // unsatisfiable.  Go search the history.
                    break 'a;
                }
                else if alpha < lo.alpha { L::Before }
                else if alpha < hi.alpha { L::Between }
                else { L::After }
            };

            let cur = function(alpha)?;
            history.push(cur);
            trace!("LS: i: {:>2}  a: {:<23e}  s: {:<23e}  v: {:<23}",
                iteration+1, cur.alpha, cur.slope, cur.value);

            #[derive(Debug,Copy,Clone,PartialEq,PartialOrd,Eq,Ord)]
            enum S { Pos, Neg }
            impl S { fn of(x: f64) -> S { if x < 0.0 { S::Neg } else { S::Pos }} }
            let (s_lo, s_hi, s_cur) = (S::of(lo.slope), S::of(hi.slope), S::of(cur.slope));

            let new_bounds;
            {
                // Make sure interval is at least bounded by (-, +) slopes.
                if (s_lo, s_hi) != (S::Neg, S::Pos) {
                    match (loc, (s_lo, s_hi)) {
                        (L::Before, (S::Neg, S::Neg))
                        | (L::After, (S::Pos, S::Pos))
                        => debug!("Shifting search in an unusual direction. ({:?} -> {:?})", [lo, hi], cur),

                        _ => {},
                    };

                    new_bounds = match loc {
                        L::After => { (hi, cur) },
                        L::Before => { (cur, lo) },
                        L::Between => {
                            debug!("Unexpectedly shrinking search interval! ({:?} -> {:?})", [lo, hi], cur);
                            (lo, cur) // arbitrary choice
                        },
                    }

                } else {
                    // Bounded by (-, +) slopes.
                    // Use guesses to rapidly disect the interval.
                    new_bounds = match (loc, s_cur) {
                        (L::Before, _) | (L::After, _)
                        => {
                            warn!("Exiting after refusing to grow interval. ({:?} -> {:?})", [lo, hi], cur);
                            break 'a;
                        },

                        (L::Between, S::Neg) => (cur, hi),
                        (L::Between, S::Pos) => (lo, cur),
                    }
                }
            };
            lo = new_bounds.0;
            hi = new_bounds.1;
        }
        debug!("Hit max iterations");
    }} // 'a

    Ok(history.iter().cloned()
        .min_by_key(|bound| NotNaN::new(bound.value).expect("bug!"))
        .expect("bug")
        .alpha)
}

fn each_computation_weak<R, E, F>(
    settings: &Settings,
    lo: Bound,
    hi: Bound,
    function: F,
) -> ResultWithControlFlow<f64, R, Error<E>>
where
  F: FnMut(f64) -> ResultWithControlFlow<Bound, R, Error<E>>,
{
    // assert!(lo.value.is_finite());
    // assert!(hi.value.is_finite());

    // let mut previous_states = ::std::collections::HashMap::new();
    // let mut history = vec![lo, hi];
    // let mut bounds = vec![lo, cur, hi];
    // 'a: loop { break {
    //     for iteration in 0..settings.iteration_limit {
    //         trace!("i: {:2}  bounds: {:?}", iteration + 1, [lo, hi]);
    //         // Invariant: each bound has a unique alpha.
    //         // This aleviates some concerns of stability in guesses for new alphas.
    //         assert!(lo.alpha < hi.alpha);
    //         {
    //             let key = (NotNaN::new(lo.alpha).unwrap(), NotNaN::new(hi.alpha).unwrap());
    //             if let Some(old_iter) = previous_states.insert(key, iteration) {
    //                 debug!("Exiting after deja vu! (Iterations {}, {}) {:?}", old_iter+1, iteration+1, [lo, hi]);
    //                 break 'a;
    //             }
    //         }

    //         if lo.slope > 0.0 && hi.slope < 0.0 {
    //             debug!("Straddling a maximum! {:?}", [lo, hi]);
    //         }

    //         let alpha = guess_min(lo, hi);
    //         if !alpha.is_finite() {
    //             warn!("Exiting after non-finite alpha. ({:?} -> {})", [lo, hi], alpha);
    //             break 'a;
    //         }

    //         #[derive(Debug,Copy,Clone,PartialEq,PartialOrd,Eq,Ord)]
    //         enum L { Before, Between, After }
    //         let loc = {
    //             if [lo.alpha, hi.alpha].contains(&alpha) {
    //                 debug!("Exiting after repeating a bound. ({:?} -> {})", [lo, hi], alpha);
    //                 // This *could* be thanks to uncannily good convergence, together with
    //                 // something else (e.g. starting on a maximum) making the Wolfe conditions
    //                 // unsatisfiable.  Go search the history.
    //                 break 'a;
    //             }
    //             else if alpha < lo.alpha { L::Before }
    //             else if alpha < hi.alpha { L::Between }
    //             else { L::After }
    //         };

    //         let cur = function(alpha)?;
    //         history.push(cur);

    //         #[derive(Debug,Copy,Clone,PartialEq,PartialOrd,Eq,Ord)]
    //         enum S { Pos, Neg }
    //         impl S { fn of(x: f64) -> S { if x < 0.0 { S::Neg } else { S::Pos }} }
    //         let (s_lo, s_hi, s_cur) = (S::of(lo.slope), S::of(hi.slope), S::of(cur.slope));

    //         let new_bounds;
    //         {
    //             // Make sure interval is at least bounded by (-, +) slopes.
    //             if (s_lo, s_hi) != (S::Neg, S::Pos) {
    //                 match (loc, (s_lo, s_hi)) {
    //                     (L::Before, (S::Neg, S::Neg))
    //                     | (L::After, (S::Pos, S::Pos))
    //                     => debug!("Shifting search in an unusual direction. ({:?} -> {:?})", [lo, hi], cur),

    //                     _ => {},
    //                 };

    //                 new_bounds = match loc {
    //                     L::After => { (hi, cur) },
    //                     L::Before => { (cur, lo) },
    //                     L::Between => {
    //                         debug!("Unexpectedly shrinking search interval! ({:?} -> {:?})", [lo, hi], cur);
    //                         (lo, cur) // arbitrary choice
    //                     },
    //                 }

    //             } else {
    //                 // Bounded by (-, +) slopes.
    //                 // Use guesses to rapidly disect the interval.
    //                 new_bounds = match (loc, s_cur) {
    //                     (L::Before, _) | (L::After, _)
    //                     => {
    //                         warn!("Exiting after refusing to grow interval. ({:?} -> {:?})", [lo, hi], cur);
    //                         break 'a;
    //                     },

    //                     (L::Between, S::Neg) => (cur, hi),
    //                     (L::Between, S::Pos) => (lo, cur),
    //                 }
    //             }
    //         };
    //         lo = new_bounds.0;
    //         hi = new_bounds.1;
    //     }
    //     debug!("Hit max iterations");
    // }} // 'a

    unimplemented!()
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

/// Get the analytical minimum of a cubic fit to two input bounds
/// using only slope.  Requires three points.
fn cubic_min_from_slope(bounds: [Bound; 3]) -> f64
{
    use ::sp2_array_utils::{mat_from_fn, dot, vec_from_fn};
    use ::sp2_array_utils::math::prelude::*;
    // Find quadratic coefficients of slope
    // [ 1  x1  x1*x1 ] [ c0 ]    [ m1 ]
    // [ 1  x2  x2*x2 ] [ c1 ] == [ m2 ]
    // [ 1  x3  x3*x3 ] [ c2 ]    [ m3 ]
    let mat: [[_; 3]; 3] = mat_from_fn(|r,c| bounds[r].alpha.powi(c as i32));
    let slopes: [_; 3] = vec_from_fn(|k| bounds[k].slope);
    let coeffs: [_; 3] = dot(&mat.inverse(), &slopes);

    // Slope has two roots; for real coefficients,
    // the + root always corresponds to the minimum value.
    let (a, b, c) = (coeffs[2], coeffs[1], coeffs[0]);
    (-b + (b*b - 4.0*a*c).sqrt()) / (2.0 * a)
}

/*
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
*/

/// Tests if a number is "sub-normal" (not the IEE-754 term), which is true
/// for zero as well as subnormals (*that's* the IEE-754 term).
///
/// (I'm not deliberately trying to be clever; I just can't come up with a
///  better name for these)
fn is_sub_normal(x: f64) -> bool {
    x.is_finite() && !x.is_normal()
}

#[cfg(test)]
mod tests {
    use super::Bound;
    use super::{linesearch, Settings, };

    use ::test_functions::one_dee::prelude::*;
    use ::test_functions::one_dee::Polynomial;

    fn init_logger() {
        let _ = ::env_logger::init();
    }

    #[derive(Debug,Copy,Clone,Hash,PartialEq,Eq,PartialOrd,Ord)]
    enum Never {}

    // Macro as HOF
    //  Input:   Differential1d
    //  Output:  Fn(f64) -> Result<(f64, f64), Never>
    macro_rules! diff_fn {
        ($f:expr) => {{
            let f = $f.clone();
            let deriv = f.derivative();
            move |x| Ok::<_,Never>((f.evaluate(x), deriv.evaluate(x)))
        }};
    }

    // Macro as HOF
    //  Input:   Differential1d
    //  Output:  Fn(f64) -> Bound
    macro_rules! bound_fn {
        ($f:expr) => {{
            let f = $f.clone();
            let deriv = f.derivative();
            move |x| Bound {
                alpha: x,
                value: f.evaluate(x),
                slope: deriv.evaluate(x),
            }
        }};
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
            let get_bound = bound_fn!(poly);
            assert_close!(root1, super::cubic_min(get_bound(0.0), get_bound(1.0)));
            // low.alpha not equal to 0, d_alpha not equal to 1
            assert_close!(root1, super::cubic_min(get_bound(4.5), get_bound(1.25)));

            // flipping x, so that minimum is the lesser root
            let get_bound = bound_fn!(poly.scale_x(-1.0));
            assert_close!(-root1, super::cubic_min(get_bound(0.0), get_bound(1.0)));

            // both extrema in same direction from x=0
            let get_bound = bound_fn!(poly.recenter(-20.0));
            assert_close!(20.0 + root1, super::cubic_min(get_bound(0.0), get_bound(1.0)));

            // Third coefficient negative, second coefficient negative
            let get_bound = bound_fn!(poly.scale_y(-1.0));
            assert_close!(root2, super::cubic_min(get_bound(0.0), get_bound(1.0)));
        }

        {
            let poly = Polynomial::from_coeffs(&[1000.0, 800.0, 52.0, -3.0]);
            let root1 = -5.27987138188865; // local minimum
            let root2 = 16.8354269374442;  // local maximum
            assert_close!(abs=1e-10, poly.derivative().evaluate(root1), 0.0);
            assert_close!(abs=1e-10, poly.derivative().evaluate(root2), 0.0);

            // Third coefficient negative, second coefficient positive
            let get_bound = bound_fn!(poly);
            assert_close!(root1, super::cubic_min(get_bound(0.0), get_bound(1.0)));

            // Third coefficient positive, second coefficient negative
            let get_bound = bound_fn!(poly.scale_y(-1.0));
            assert_close!(root2, super::cubic_min(get_bound(0.0), get_bound(1.0)));
        }
    }

    #[test]
    fn quadratic_min_analytic() {
        let poly = Polynomial::from_coeffs(&[1000.0, -800.0, -52.0]);
        let root = -7.69230769230769; // local minimum
        assert_close!(abs=1e-10, poly.derivative().evaluate(root), 0.0);

        // simple
        let get_bound = bound_fn!(poly);
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
            let get_bound = bound_fn!(poly);
            let guess = super::guess_min(get_bound(0.0), get_bound(1.0));

            // looks minimal?
            assert_close!(abs=1e-4, poly.derivative().evaluate(guess), 0.0, "{:?}", poly);
            assert!(false
                || poly.evaluate(guess) <= poly.evaluate(guess * (1.0 - 1e-2))
                || poly.evaluate(guess) <= poly.evaluate(guess * (1.0 - 1e-5))
                , "{:?}", poly
                );
            assert!(false
                || poly.evaluate(guess) <= poly.evaluate(guess * (1.0 + 1e-2))
                || poly.evaluate(guess) <= poly.evaluate(guess * (1.0 + 1e-5))
                , "{:?}", poly
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
        let out = linesearch(&Settings::default(), 0.125, diff_fn!(poly)).unwrap();
        assert!(!out.is_nan());
        // it should be able to find at least one point better than those we gave it
        assert!(poly.evaluate(out) < poly.evaluate(0.125));
    }
}