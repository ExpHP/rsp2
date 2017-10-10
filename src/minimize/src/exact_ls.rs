error_chain!{
    types {
        Error, ErrorKind, ResultExt, LsResult;
    }
    errors {
        BadBound(b: f64) {
            description("An input bound was too extreme")
            display("The input bound was too extreme: {}", b)
        }
        NoMinimum {
            description("The function appears to have no minimum")
            display("The function appears to have no minimum", )
        }
        FunctionOutput(b: f64) {
            description("The function produced an inscrutible value")
            display("The function produced an inscrutible value: {}", b)
        }
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

fn check_mirroring_assumption(x0: f64) -> LsResult<()> {
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
    ensure!(2.0 * x0 - x0 == x0, ErrorKind::BadBound(x0));
    Ok(())
}

pub fn linesearch<E, F>(
    from: f64,
    initial_step: f64,
    mut compute: F,
) -> LsResult<Result<SlopeBound, E>>
where F: FnMut(f64) -> Result<Slope, E>
{
    // early wrapping:
    //  - SlopeBound for internal use
    //  - Detect nonsensical slopes
    //  - Result<Slope, Result<TheirError, OurError>> for easy short-circuiting
    let compute = move |alpha| {
        let slope = compute(alpha).map_err(Ok)?;
        ensure!(slope.0.is_finite(), Err(ErrorKind::FunctionOutput(slope.0).into()));
        trace!("LS-iter:  a: {:<23e}  s: {:<23e}", alpha, slope.0);
        Ok(SlopeBound { alpha, slope: slope.0 })
    };

    // make it possible to conditionally wrap the closure into another.
    let mut compute: Box<FnMut(f64) -> Result<SlopeBound, Result<E, Error>>>
        = Box::new(compute);

    // IIFE to intercept this result so we can transform it at the end
    match (|| {
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
        bisect((a, b), &mut *compute)
    })() {
        Ok(bound) => {
            trace!("LS-exit:  a: {:<23e}  s: {:<23e}", bound.alpha, bound.slope);
            Ok(Ok(bound))
        },
        Err(Ok(theirs)) => Ok(Err(theirs)),
        Err(Err(ours)) => Err(ours),
    }
}

fn find_initial<E>(
    (a, mut b): SlopeInterval,
    compute: &mut FnMut(f64) -> Result<SlopeBound, Result<E, Error>>,
) -> Result<SlopeInterval, Result<E, Error>>
{
    assert!(a.slope <= 0.0);
    while b.slope < 0.0 {
        // double the interval width
        let new_alpha = b.alpha + (b.alpha - a.alpha);
        ensure!(new_alpha.is_finite(), Err(ErrorKind::NoMinimum.into()));
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
