//! # Citations:
//!
//! * Hager, W.W. and Zhang, H. (2004), A new conjugate gradient method with
//! guaranteed descent and an efficient line search, University of Florida, Department
//! of Mathematics, November 17, 2003 (theory and comparisons), revised July 3, 2004.

// #[derive(Debug,Copy,Clone,PartialEq,Eq,PartialOrd,Ord,Hash)]
// enum Sign { Negative, Zero, Positive }
// use self::SlopeClass::*;
// impl Sign {
//     pub fn of(x: f64) -> Sign {
//         match x.partial_cmp(0f64).expect("sign of NaN!") {
//             Ordering::Less => Sign::Negative,
//             Ordering::Equal => Sign::Zero,
//             Ordering::Greater => Sign::Positive,
//         }
//     }

//     pub fn fuzzy(tol: f64, x: f64) -> Sign {
//         if x.abs() < tol.abs() { Sign::Zero }
//         else { Sign::of(x) }
//     }
// }


/// Holds information about linesearch boundaries
#[derive(Debug, Copy, Clone)]
struct Bound { alpha: f64, value: f64, slope: f64 }

type Interval = (Bound, Bound);
impl Bound {
    fn strictly_downhill(&self) -> bool { self.slope < 0.0 }
}

fn linterp(theta: f64, (a, b): (f64, f64)) -> f64 {
    (1.0 - theta) * a + theta * b
}

#[derive(Debug,Clone,PartialEq,PartialOrd)]
pub struct Params {
    // FIXME these need legit, meaningful names
    delta: f64,   // armijo coefficient
    sigma: f64,   // curvature coefficient
    epsilon: f64, // estimate of relative error in value
    theta: f64,   // bisection point for step U3
    gamma: f64,   // ???
    eta: f64,     // ???
    sanic: f64,   // interval width growth factor during expansion search
}

impl Params { pub fn new() -> Params { Default::default() } }
impl Default for Params {
    fn default() -> Params {
        Params {
            delta: 0.1,
            sigma: 0.9,
            epsilon: 1e-10,
            theta: 0.5,
            gamma: 2.0/3.0,
            eta: 0.01,
            sanic: (1.0 + 5f64.sqrt()) / 2.0,
        }
    }
}

impl Params {
    pub fn validate(&self) {
        let Params { delta, sigma, epsilon, theta, gamma, eta, sanic } = *self;
        assert!(0.0 < delta && delta < 0.5);
        assert!(delta <= sigma && sigma < 1.0);
        assert!(0.0 <= epsilon);
        assert!(0.0 < theta && theta < 1.0);
        assert!(0.0 < gamma && gamma < 1.0);
        assert!(0.0 < eta);
        assert!(1.0 < sanic);
    }
}


struct Hager {
    params: Params,
    initial: Bound,
}

#[derive(Debug,Copy,Clone,PartialEq,Eq,PartialOrd,Ord,Hash)]
enum How {
    InitialGuess,
    SameSlopeBisect,
    SameSlopeExpand,
    DoubleSecant1,
    DoubleSecant2,
    PlainBisect,
    Hack_IsLsState, // FIXME
}

impl How {
    pub fn as_str(&self) -> &'static str {
        match *self {
            How::InitialGuess => "INI",
            How::SameSlopeBisect => "SSB",
            How::SameSlopeExpand => "SSE",
            How::DoubleSecant1 => "DS1",
            How::DoubleSecant2 => "DS2",
            How::PlainBisect => "BSC",
            How::Hack_IsLsState => panic!(),
        }
    }
}

type ShortCircuitResult<O, SC, E> = Result<O, Result<SC, E>>;

pub fn linesearch<E, F>(
    params: &Params,
    initial_alpha: f64,
    mut compute: F,
) -> Result<f64, E>
where F: FnMut(f64) -> Result<(f64, f64), E>,
{
    let compute = |alpha| {
        let (value, slope) = compute(alpha)?;
        Ok(Bound { alpha, value, slope })
    };

    // I highly doubt that statically known function will help optimize
    // linesearches very much, and boxing helps us handle negative slope.
    let mut compute: Box<FnMut(f64) -> Result<Bound, E>> = Box::new(compute);
    let mut initial = compute(0.0)?;

    if initial.slope > 0.0 {
        debug!("Positive initial slope, turning around. (slope = {:e})", initial.slope);
        compute = Box::new(move |alpha| {
            compute(-alpha).map(|bound| Bound {
                // make alpha positive again
                alpha: { assert_eq!(bound.alpha, -alpha); alpha },
                value: bound.value,
                // slope reflects sign of dx
                slope: -bound.slope,
            })
        });

        assert_eq!(initial.alpha, 0.0);
        initial = Bound {
            alpha: 0.0,
            value: initial.value,
            slope: -initial.slope,
        }
    }

    let initial = initial; // un-mut

    // HACK: dumb edge case for zero slope;
    // allowing this through would make it even trickier to find an initial
    // interval that satisfies the opposite slope condition, and the only
    // cases where we could possibly ever accomplish anything by continuing
    // is the case where we initially lie exactly at a maximum or saddle point.
    //
    // This will never happen in practice for the functions we care about,
    // and is therefore simply not worth the effort.  Just give up.
    if initial.slope.abs() < 1e-200 {
        warn!("Bailing out immediately due to zero initial slope.");
        return Ok(initial.alpha);
    }

    Hager {
        params: params.clone(),
        initial,
    }.linesearch(
        initial_alpha,
        // FIXME change other methods to accept Box since their current
        //  signatures are misleading and force us to do eta expansion here
        |x| compute(x),
    ).map(|x| x.alpha)
}

// NOTE: Not one of these methods mutates self.
//       Hager serves as a context and nothing more.
impl Hager {

    /// Linesearch entry point.
    ///
    /// # Citation:
    /// Hager 2004, p. 184
    fn linesearch<E, F>(
        &self,
        start_alpha: f64,
        mut compute: F,
    ) -> Result<Bound, E>
    where F: FnMut(f64) -> Result<Bound, E>,
    {
        // We do a nasty trick with control flow here.
        //
        // ...Let me explain.
        //
        // Ideally, we should always return as soon as we have a bound that
        // meets the wolfe conditions; but this algorithm uses many different
        // strategies for choosing points to evaluate, and managing this gets
        // horrendous fast.
        // FIXME: (This isn't always necessarily true; I've observed that exiting
        //         super quick like this can lead to slow convergence)

        // So... we put the Wolfe condition tests inside the compute function
        // itself, and allow it to use `Err(Ok(x))` to signal a successful return.
        // This gets processed by all of the existing machinery we already use to
        // short-circuit on legitimate errors, resulting in a strangely pleasant
        // separation of concerns...
        // ...until we start making other methods aware of what we have done
        // allowing them to do horrible things like bail out of arbitrary closures.
        // Which we do.

        // FIXME
        trace!("                                   entry slope: {:<23e}", compute(start_alpha)?.slope);

        // This IIFE acts as a poor-man's `catch` block.  It scopes the
        // `?` operators inside so we can post-process the result.
        let result: ShortCircuitResult<(), Bound, E> = (|| {
            assert!(start_alpha > 0.0);

            let width = |(lo,hi): Interval| hi.alpha - lo.alpha;
            let midpoint = |(lo,hi): Interval| 0.5 * (hi.alpha + lo.alpha);

            // Wrap the compute function with things we want to do on
            // every computed point (including the successful return).
            let mut slow_exit = 2; // HACK
            let mut slow_non_exits = 0; // HACK AW GEEZE
            let slow_non_exit_limit = -14; // HACK WOW SO HACK
            let mut compute = {
                let mut computations = 0;
                move |alpha, how: How| {
                    computations += 1;

                    // Legitimate errors are now `Err(Err(e))`.
                    let bound = compute(alpha).map_err(Err)?;
                    if how == How::Hack_IsLsState {
                        if slow_exit > 0 {                //              HACK HACK
                            slow_exit -= 1;               //           HACK HACK
                        } else {                          //          HACKHACK    HACK   HACK   HACK
                            slow_non_exits -= 1;          //           HACK HACK
                        }                                 //              HACK HACK
                    } else {
                        trace!("LS: i: {:>2} ({})  a: {:<23e}  s: {:<23e}  v: {:<23}",
                            computations, how.as_str(), alpha, bound.slope, bound.value);
                    }

                    if self.should_accept(bound) && (slow_exit < 0 || slow_exit == 0 && how == How::Hack_IsLsState) {
                        // NOTE: this returns from the entire algorithm
                        return Err(Ok(bound));
                    }

                    // HACK iteration limit because the slow exit strategy can get indefinitely
                    //      stuck with a small interval that keeps guessing one of the bounds
                    //      (it goes DS1 -> BSC -> DS1 -> BSC -> DS1 -> ...)
                    if slow_non_exits < slow_non_exit_limit && how == How::Hack_IsLsState {
                        return Err(Ok(bound));
                    }

                    // NOTE: this only returns to the caller of 'compute'
                    Ok(bound)
                }
            };

            // The author abandons us to our own devices for a moment here:
            let mut cur = self.seek_initial_interval(start_alpha, &mut compute)?;

            // The rest is the algo actually presented on page 184 of the paper.
            loop {
                self.validate_opposite_slope(cur);

                compute(cur.0.alpha, How::Hack_IsLsState)?;

                let new = self.double_secant_strategy(cur, &mut compute)?;
                cur = match width(new) <= self.params.gamma * width(cur) {
                    true => new,
                    false => {
                        let mid = compute(midpoint(new), How::PlainBisect)?;
                        self.update_interval(new, mid, &mut compute)?
                    }
                }
            }

            // this is for you, the reader. Not rustc.
            #[allow(unreachable_code)]
            { unreachable!(); }
        })();

        // Get the actual result that we tucked away in `Err`.
        let result = result.err().expect("buggg");
        if let Ok(Bound { value, slope, .. }) = result {
            // FIXME
            trace!("                                    exit slope: {:<23e}", slope);
        }
        result
    }

    /// (4.4) in Hager (2004), the "opposite slope condition".
    ///
    /// These invariants are upheld by *almost* every interval
    /// constructed by the algorithm.
    fn validate_opposite_slope(&self, (lo, hi): Interval) {
        assert!(lo.alpha < hi.alpha);
        assert!(lo.strictly_downhill());
        assert!(!hi.strictly_downhill());
        assert!(self.reasonable_value(lo));
        // no condition on hi value
    }

    /// This locates an initial interval that satisfies the opposite slope condition.
    fn seek_initial_interval<E, F>(
        &self,
        start_alpha: f64,
        mut compute: F,
    ) -> ShortCircuitResult<Interval, Bound, E>
    where F: FnMut(f64, How) -> ShortCircuitResult<Bound, Bound, E>,
    {
        // Hager (2004) provides no strategy for this.
        // Or rather, it says that we can sample phi(alpha) for
        //   "various choices of alpha"... Harumph.

        // "Most" of our work is already done.
        // All that really remains is to locate an upper bound that is not downhill.
        let mut lo = self.initial;

        let mut cur = compute(start_alpha, How::InitialGuess)?;

        // We could handle this similar to `update_interval`,
        // except that we have no `hi` ready to be returned in the case of (U2).
        // Take care of this case straight away.
        while cur.strictly_downhill() && self.reasonable_value(cur) {
            assert!(lo.alpha < cur.alpha, "lo.alpha < cur.alpha failed:  {:e} vs {:e}", lo.alpha, cur.alpha);
            assert!(lo.strictly_downhill());
            assert!(self.reasonable_value(lo));

            // We're on an interval that appears to be entirely downhill.
            // Travel down along the line until something happens.
            //
            // Pick a point at least twice as far from lo.alpha as cur.alpha is.
            //  (the parameter is >= 1.0 so the linterp argument is >= 2.0)
            let next_alpha = linterp(1.0 + self.params.sanic, (lo.alpha, cur.alpha));
            lo = cur;
            cur = compute(next_alpha, How::SameSlopeExpand)?;
        }

        // the rest plays out like 'update_interval'
        let out = match (cur.strictly_downhill(), self.reasonable_value(cur)) {
            (false, _) => (lo, cur),
            (true, true) => unreachable!(), // taken care of by above loop
            (true, false) => self.funky_loop_in_u3((lo, cur), &mut compute)?,
        };
        self.validate_opposite_slope(out);
        Ok(out)
    }

    /// Takes values computed at the endpoints of an interval together with
    /// one additional point, and attempts to construct a shorter interval
    ///  that satisfies the opposite slope conditions.
    ///
    /// # Citation:
    /// Hager 2004, p. 182
    fn update_interval<E, F>(
        &self,
        input: Interval, // Current boundaries, written as 'a' and 'b' in the paper.
        guess: Bound, // Guess for next bound, written as 'c' in the paper
        mut compute: F,
    ) -> Result<Interval, E>
    where F: FnMut(f64, How) -> Result<Bound, E>,
    {
        self.validate_opposite_slope(input);
        let (lo, hi) = input;

        if !(lo.alpha < guess.alpha && guess.alpha < hi.alpha) {
            debug!("update_interval: Exit by strange guess (U0),  ({:e}, {:e}) vs {:e}",
                lo.alpha, hi.alpha, guess.alpha);
            return Ok((lo, hi));
        }

        let out = match (guess.strictly_downhill(), self.reasonable_value(guess)) {
            // easy cases where guess is already suitable for one of the bounds
            (true, true) => (guess, hi),   // condition (U2), p. 182
            (false, _) => (lo, guess),     // condition (U1), p. 182

            // tough case; bisect until we have a good interval again.
            (true, false) => self.funky_loop_in_u3((lo, guess), &mut compute)?,
        };

        let is_improper_subset = |a: Interval, b: Interval|
            b.0.alpha <= a.0.alpha && a.1.alpha <= b.1.alpha;

        assert!(is_improper_subset(out, input));
        self.validate_opposite_slope(out);
        Ok(out)
    }

    /// An unnamed loop that shows up in Hager (2004) step U3.
    ///
    /// It takes an interval where both ends point downhill but the upper end
    /// has a greater value (suggesting that the function follows some sort
    /// of rotated 'S' shape), and returns an interval satisfying the opposite
    /// slope condition.
    fn funky_loop_in_u3<E, F>(
        &self,
        (mut lo, mut hi): Interval,
        mut compute: F,
    ) -> Result<Interval, E>
    where F: FnMut(f64, How) -> Result<Bound, E>,
    {
        debug!("update_interval: Beginning same-slope bisection strategy.");

        let out = loop {
            // NOTE: These invariants differ in the marked ways
            //        from the opposite slope condition.
            assert!(lo.alpha < hi.alpha);
            assert!(lo.strictly_downhill());
            assert!(hi.strictly_downhill()); // <-- flipped condition
            assert!(self.reasonable_value(lo));
            assert!(!self.reasonable_value(hi)); // <-- new condition

            // TODO: I'm not yet convinced by the paper's argument for why this
            //       loop terminates ("The loop embedded in U3a-c [...]").
            //       It seems to me that it could potentially reach a point where
            //       'lo.alpha < mid.alpha < hi.alpha' fails before exiting.
            let mid_alpha = linterp(self.params.theta, (lo.alpha, hi.alpha));
            let mid = compute(mid_alpha, How::SameSlopeBisect)?;
            match (mid.strictly_downhill(), self.reasonable_value(mid)) {
                (true, true) => lo = mid,
                (true, false) => hi = mid,
                (false, _) => break (lo, mid),
            }
        }; // let out = loop { ... }

        self.validate_opposite_slope(out);
        Ok(out)
    }

    /// # Citation:
    /// Hager 2004, p. 184
    fn double_secant_strategy<E, F>(
        &self,
        (lo, hi): Interval,
        mut compute: F,
    ) -> Result<Interval, E>
    where F: FnMut(f64, How) -> Result<Bound, E>,
    {
        self.validate_opposite_slope((lo, hi));

        let secant = |a: Bound, b: Bound| {
            // NOTE: it is not necessarily true that 'a.alpha < b.alpha'
            // NOTE: this is currently allowed to return nan; see below
            let numer = a.alpha * b.slope - b.alpha * a.slope;
            let denom = b.slope - a.slope;
            numer / denom
        };

        let first = compute(secant(lo, hi), How::DoubleSecant1)?;
        let (new_lo, new_hi) = self.update_interval((lo, hi), first, &mut compute)?;
        self.validate_opposite_slope((new_lo, new_hi));

        // Checks if either of the "easy cases" in update_interval were met.
        // (TODO: can't this also occur by failing condition (U0) via equality?)
        // (NOTE: this does not occur in the "hard case" because in that case
        //         the initial guess is unsuitable for either bound)
        let accepted_as_lo = first.alpha == new_lo.alpha;
        let accepted_as_hi = first.alpha == new_hi.alpha;
        if accepted_as_lo || accepted_as_hi {
            // The paper uses considerations of curvature to pick bounds for the next secant
            // which are most likely to bracket the zero from the side opposite the first secant.
            let second_alpha = match (accepted_as_lo, accepted_as_hi) {
                (false, false) => unreachable!(),
                (false, true) => secant(hi, new_hi),
                (true, false) => secant(lo, new_lo),
                (true, true) => panic!("impossible due to opposite slope invariant")
            };

            // HACK: This is_finite() test is an experimental "quick fix".
            //       Sometimes when approaching the minimum with fast exit disabled,
            //       we can end up with 'hi == new_hi' or 'lo == new_lo',
            //       and so the second secant is NaN.
            //
            //       I tried heading this issue off at a couple of different places;
            //       but simply skipping the second interval update has been the most
            //       robust.
            //
            //       I am uncomfortable that this could potentially mask real errors;
            //       but most should still trigger another assertion elsewhere.
            if second_alpha.is_finite() {
                let second = compute(second_alpha, How::DoubleSecant2)?;
                return self.update_interval((new_lo, new_hi), second, &mut compute);
            }
        }
        // skip the second `update_interval`
        Ok((new_lo, new_hi))
    }

    /// Standard Wolfe conditions.
    fn wolfe_conditions(&self, bound: Bound) -> bool {
        let Bound { alpha, value, slope } = bound;
        let Bound { slope: zero_slope, value: zero_value, .. } = self.initial;
        let Params { delta, sigma, .. } = self.params;
        assert!(zero_slope < 0.0);

        let armijo = value - zero_value <= alpha * delta * zero_slope;

        // This is the standard curvature condition, which permits all
        // positive slopes (NOTE: zero_slope is negative) as opposed to
        // the "strong" curvature condition which caps the absolute value.
        let curvature = slope >= sigma * zero_slope;

        armijo && curvature
    }

    /// An alternate set of permissible exit conditions suitable for
    /// use when extremely close to a minimum.
    ///
    /// # Citation:
    /// Hager 2004, p. 181
    fn approx_wolfe_conditions(&self, bound: Bound) -> bool {
        let Bound { alpha, value, slope } = bound;
        let Bound { slope: zero_slope, .. } = self.initial;
        let Params { delta, sigma, .. } = self.params;
        assert!(zero_slope < 0.0);

        // Slope-based approximation of armijo condition,
        // equivalent up to second order in alpha.
        let armijo_slope = slope <= (2.0 * delta - 1.0) * zero_slope;

        // This is the standard curvature condition, which permits all
        // positive slopes (NOTE: zero_slope is negative) as opposed to
        // the "strong" curvature condition which caps the absolute value.
        let curvature = slope >= sigma * zero_slope;

        armijo_slope && curvature
    }

    /// (4.2) in Hager (2004)
    ///
    /// A far weaker condition on value than the armijo condition.
    ///
    /// This condition does permit the value to increase, but only so long as we can
    /// comfortably write it off as "numerical error".
    fn reasonable_value(&self, Bound {value, ..}: Bound) -> bool {
        let Params { epsilon, .. } = self.params;
        let Bound { value: zero_value, .. } = self.initial;

        value - zero_value <= epsilon * zero_value.abs()
    }

    /// Exit criteria of Hager's linesearch.
    ///
    /// # Citation:
    /// Hager 2004, p. 181
    fn should_accept(&self, bound: Bound) -> bool {
        #![allow(non_snake_case)]
        let T1 = self.wolfe_conditions(bound);
        let T2 = self.approx_wolfe_conditions(bound) && self.reasonable_value(bound);
        T1 || T2
    }
}

#[deny(dead_code)]
#[cfg(test)]
mod tests {
    use super::linesearch;

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

    // NOTE: The current linesearch can't handle this one.
    //       This is accepted as a "known limitation."
    #[cfg(nope)]
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
        let out = linesearch(&Default::default(), 0.125, diff_fn!(poly)).unwrap();
        assert!(!out.is_nan());
        // it should be able to find at least one point better than those we gave it
        assert!(poly.evaluate(out) < poly.evaluate(0.125));
    }

    #[test]
    fn start_near_maximum() {
        init_logger();

        // Here is a function which has a maximum located precisely at x == 0.
        // When evaluated at x == 0, the slope will come out exactly equal to 0.
        let poly = Polynomial::from_coeffs(&[0.0, 0.0, -1.0, 0.0, 1.0]);

        // ...but that's too evil.  Let's start very close to--but not quite at--zero.
        let poly = poly.recenter(1e-7);

        // It shouldn't produce an error
        let out = linesearch(&Default::default(), 0.125, diff_fn!(poly)).unwrap();
        assert!(!out.is_nan());
        // it should be able to find at least one point better than those we gave it
        assert!(poly.evaluate(out) < poly.evaluate(0.125));
    }

    // FIXME test turning around on initially positive slope
}
