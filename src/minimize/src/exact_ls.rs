use ::failure::Backtrace;

#[derive(Debug, Fail)]
#[fail(display = "{}", kind)]
pub struct GoldenSearchError {
    backtrace: Backtrace,
    kind: ErrorKind,
}

#[derive(Debug, Fail)]
pub enum ErrorKind {
    #[fail(display = "The input bound was too extreme: {}", _0)]
    BadBound(f64),
    #[fail(display = "Golden search encountered value larger than endpoints: {:?} vs {}", endvals, value)]
    GsBadValue {
        endvals: (f64, f64),
        value: f64,
    },
    #[fail(display = "The function appears to have no minimum")]
    NoMinimum,
    #[fail(display = "The function produced an inscrutible value: {}", _0)]
    FunctionOutput(f64),
    #[doc(hidden)]
    #[fail(display = "impossible!")]
    _Hidden,
}

impl From<ErrorKind> for GoldenSearchError {
    fn from(kind: ErrorKind) -> Self {
        let backtrace = Backtrace::new();
        GoldenSearchError { backtrace, kind }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub struct Value(pub f64);
#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub struct Slope(pub f64);
#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub struct ValueBound { pub alpha: f64, pub value: f64 }
#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub struct SlopeBound { pub alpha: f64, pub slope: f64 }
#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub struct Bound { pub alpha: f64, pub value: f64, pub slope: f64 }

pub type Interval = (f64, f64);
pub type SlopeInterval = (SlopeBound, SlopeBound);
pub type ValueFn<'a, E> = FnMut(f64) -> Result<Value, E> + 'a;
pub type SlopeFn<'a, E> = FnMut(f64) -> Result<Slope, E> + 'a;
pub type OneDeeFn<'a, E> = FnMut(f64) -> Result<(Value, Slope), E> + 'a;

fn check_mirroring_assumption(x0: f64) -> Result<(), GoldenSearchError> {
    // Assumption:
    //
    // Given an IEEE-754 floating point number of any precision
    // for which (2*x) is finite (x may be subnormal):
    //
    //      2 * x - x == x
    //
    // This has been validated by brute force for all f32
    // on x86_64 architecture.
    //
    // This assumption allows us to take a function evaluated
    // at 'x0' and change its argument to '2*x0 - x', knowing
    // that the value at 'x0' (and more importantly, the sign
    // of the slope) has been identically preserved.
    if 2.0 * x0 - x0 != x0 {
        return Err(ErrorKind::BadBound(x0).into());
    }
    Ok(())
}

pub fn linesearch<E, F>(
    from: f64,
    initial_step: f64,
    mut compute: F,
) -> Result<Result<SlopeBound, E>, GoldenSearchError>
where F: FnMut(f64) -> Result<Slope, E>
{
    // early wrapping:
    //  - SlopeBound for internal use
    //  - Detect nonsensical slopes
    //  - Result<Slope, Result<TheirError, OurError>> for easy short-circuiting
    let compute = move |alpha| {
        let slope = compute(alpha).map_err(Ok)?;
        if !slope.0.is_finite() {
            return Err(Err(ErrorKind::FunctionOutput(slope.0).into()));
        }
        trace!("LS-iter:  a: {:<23e}  s: {:<23e}", alpha, slope.0);
        Ok(SlopeBound { alpha, slope: slope.0 })
    };

    // make it possible to conditionally wrap the closure into another.
    let mut compute: Box<FnMut(f64) -> Result<SlopeBound, Result<E, GoldenSearchError>>>
        = Box::new(compute);

    nest_err(|| {
        let mut a = compute(from)?;
        if a.slope > 0.0 {
            check_mirroring_assumption(a.alpha).map_err(Err)?;
            let center = a.alpha;
            compute = Box::new(move |alpha|
                compute(2.0 * center - alpha)
                    .map(|SlopeBound { slope, .. }|
                        SlopeBound { alpha, slope: -slope })
            );
            a.slope *= -1.0;
        };
        let b = compute(from + initial_step)?;

        let (a, b) = find_initial((a, b), &mut *compute)?;
        let bound = bisect((a, b), &mut *compute)?;
        trace!("LS-exit:  a: {:<23e}  v: {:<23e}", bound.alpha, bound.slope);
        Ok(bound)
    })
}

fn find_initial<E>(
    (a, mut b): SlopeInterval,
    compute: &mut FnMut(f64) -> Result<SlopeBound, Result<E, GoldenSearchError>>,
) -> Result<SlopeInterval, Result<E, GoldenSearchError>>
{
    assert!(a.slope <= 0.0);
    while b.slope < 0.0 {
        // double the interval width
        let new_alpha = b.alpha + (b.alpha - a.alpha);
        if !new_alpha.is_finite() {
            return Err(Err(ErrorKind::NoMinimum.into()));
        }
        b = compute(new_alpha)?;
    }
    Ok((a, b))
}

fn bisect<E>(
    (mut lo, mut hi): SlopeInterval,
    compute: &mut FnMut(f64) -> Result<SlopeBound, E>,
) -> Result<SlopeBound, E>
{
    assert!(lo.alpha <= hi.alpha);
    loop {
        // We do allow both endpoints to have zero slope.
        assert!(lo.slope <= 0.0);
        assert!(hi.slope >= 0.0);

        let alpha = 0.5 * (lo.alpha + hi.alpha);
        if !(lo.alpha < alpha && alpha < hi.alpha) {
            return Ok(lo);
        }

        let bound = compute(alpha)?;

        // NOTE: If slope is uniformly zero, we'll shrink down to just 'lo'.
        match bound.slope >= 0.0 {
            true => hi = bound,
            false => lo = bound,
        }
    }
}


pub mod golden {
    pub use self::stop_condition::Rpn as StopCondition;

    pub(crate) use self::stop_condition::Objectives;

    pub mod stop_condition {
        use ::stop_condition::prelude::*;

        #[derive(Debug, Clone, PartialEq)]
        pub(crate) struct Objectives {
            /// x values at a, b, and d
            pub alphas: (f64, f64, f64),
            /// The number of full iterations that have occurred.
            pub iterations: u32,
        }

        #[derive(Serialize, Deserialize)]
        #[derive(Debug, Copy, Clone, PartialEq)]
        pub enum Simple {
            #[serde(rename = "interval-size")] IntervalSize(f64),
            #[serde(rename =    "iterations")] Iterations(u32),
        }

        impl ShouldStop<Objectives> for Simple {
            fn should_stop(&self, objs: &Objectives) -> bool {
                let Objectives { iterations, alphas: (a, _b, d) } = *objs;
                match *self {
                    Simple::IntervalSize(tol) => (a - d).abs() <= tol,
                    Simple::Iterations(n) => iterations >= n,
                }
            }
        }

        impl Default for Cereal {
            fn default() -> Self {
                // always false. (Even with this, `golden` will always terminate)
                from_json!({"any": []})
            }
        }

        pub type Cereal = ::stop_condition::Cereal<Simple>;
        pub type Rpn = ::stop_condition::Rpn<Simple>;

        mod tests {
            #[test]
            fn test_serialized_repr() {
                use super::Simple::Iterations;
                use ::stop_condition::Cereal::{Simple,Logical};
                use ::stop_condition::LogicalExpression::All;
                use ::serde_json::to_value;
                assert_eq!(
                    to_value(Simple(Iterations(5))).unwrap(),
                    json!({"iterations": 5})
                );
                assert_eq!(
                    to_value(Logical(All(vec![Simple(Iterations(5))]))).unwrap(),
                    json!({"all": [{"iterations": 5}]})
                );
            }
        }
    }
}

/// Builder for golden search
#[derive(Debug, Clone)]
pub struct Golden {
    stop_condition: golden::StopCondition,
}

impl Golden {
    pub fn new() -> Self
    {
        let stop_condition = golden::StopCondition::from_cereal(&Default::default());
        Golden { stop_condition }
    }

    pub fn stop_condition(&mut self, cereal: &golden::stop_condition::Cereal) -> &mut Self
    { self.stop_condition = golden::StopCondition::from_cereal(cereal); self }

    // Revelations:
    //  1. In common implementations of the algorithm (such as those on wikipedia)
    //     the values of the function at the endpoints are never used.
    //     Hence **it is only necessary to save one y value.**
    //     However, we save more because we don't trust the function's accuracy.
    //  2. TECHNICALLY the step function doesn't even even need to use phi;
    //      one could record 'b' and derive the second endpoint as 'c = d - b + a'.
    //     But I don't know if that is numerically stable, so we will do what
    //     the wikipedia implementations do and recompute b and c every iter.
    pub fn run<E, F>(
        &self,
        interval: (f64, f64),
        mut compute: F,
    // NOTE: cannot return a bound due to issue mentioned in body
    ) -> Result<Result<f64, E>, GoldenSearchError>
    where F: FnMut(f64) -> Result<Value, E>
    {
        nest_err(|| {
            // early wrapping:
            //  - ValueBound for internal use
            //  - Result<Value, Result<TheirError, OurError>> for easy short-circuiting
            let mut compute = move |alpha| {
                let value = compute(alpha).map_err(Ok)?;
                if !value.0.is_finite() {
                    return Err(Err(ErrorKind::FunctionOutput(value.0).into()));
                }
                trace!("GS-iter:  a: {:<23e}  v: {:<23e}", alpha, value.0);
                Ok(ValueBound { alpha, value: value.0 })
            };

            let phi: f64 = (1.0 + 5f64.sqrt()) / 2.0;
            let get_mid_xs = |a, d| {
                let dist = (d - a) / (1.0 + phi);
                (a + dist, d - dist)
            };

            let (mut state, mut history) = {
                // endpoints. (note: we allow d.alpha < a.alpha)
                let a = compute(interval.0)?;
                let d = compute(interval.1)?;

                // inner point closer to a
                let b = compute(get_mid_xs(a.alpha, d.alpha).0)?;

                let history = vec![a, d, b];
                ((a, b, d), history)
            };

            let mut iterations = 0;
            let stop_reason = loop {
                // Golden search will usually stop long before this,
                // after squeezing every last bit out of the f64 mantissa.
                if iterations > 300 { panic!("GS never stopped!"); }

                let (a, mut b, d) = state;

                // user-supplied stop condition
                {
                    use ::stop_condition::ShouldStop;
                    let objectives = golden::Objectives {
                        alphas: (a.alpha, b.alpha, d.alpha),
                        iterations: iterations,
                    };

                    if self.stop_condition.should_stop(&objectives) {
                        break "met user-supplied stop condition";
                    }
                }

                // Forcibly stop when it is obvious that the value is no longer numerically reliable.
                // (our interval looks like it has the wrong curvature)
                if b.value > a.value.max(d.value) {
                    break "noise > signal, or wrong curvature";
                }

                // re-adjust b, purportedly to avoid systematic issues with precision
                // that can cause infinite loops. (I dunno. ask whoever edits wikipedia)
                //
                // NOTE: Technically this desynchronizes the alpha of our Bounds from
                //  the values, so at the end we cannot return a bound.
                let (b_alpha, c_alpha) = get_mid_xs(a.alpha, d.alpha);
                b.alpha = b_alpha;

                // Forcibly stop if the interval has shrunk to nothing.
                {
                    let mut test = false;
                    // Has b crossed c?
                    test = test || (c_alpha - b.alpha) * (d.alpha - a.alpha) <= 0.0;
                    // Are any two alphas equal?
                    test = test || a.alpha == b_alpha;
                    test = test || c_alpha == d.alpha;
                    if test {
                        break "empty interval";
                    }
                }

                let c = compute(c_alpha)?;

                history.push(c);
                state = match b.value < c.value {
                    true => (c, b, a),
                    false => (b, c, d),
                };

                iterations += 1;
            }; // let stop_reason = { ... }

            //history.sort_on_key(|bound| NotNaN::new(bound.alpha).unwrap());
            let (_a, b, _d) = state;
            trace!("GS-stop:  a: {:<23e}  ({})", b.alpha, stop_reason);
            Ok(b.alpha)
        })

    }
}

// (NOTE: takes an IIFE so that ? can be used inside of it)
fn nest_err<A, B, C, F>(f: F)-> Result<Result<A, B>, C>
where F: FnOnce() -> Result<A, Result<B, C>>
{
    match f() {
        Ok(x) => Ok(Ok(x)),
        Err(Ok(e)) => Ok(Err(e)),
        Err(Err(e)) => Err(e),
    }
}
