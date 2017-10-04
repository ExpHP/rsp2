
use ::sp2_slice_math::{vnorm, vdot, V, v, vnormalize};
use ::stop_condition::prelude::*;

use ::itertools::Itertools;
use ::ordered_float::NotNaN;
use ::std::fmt::Write;
use ::std::collections::VecDeque;

#[derive(Serialize, Deserialize)]
#[derive(Debug, Clone, PartialEq)]
pub struct Settings {
    #[serde(rename = "stop-condition")]
    stop_condition: StopConditionSettings,
}

impl Settings {
    pub fn has_verbosity(&self, level: i32) -> bool { true } // FIXME
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Objectives {
    /// Signed change in potential over the previous iteration.
    /// (`None` before the first iteration)
    pub delta_value: Option<f64>,
    /// Max atomic force for the current structure.
    pub grad_max: f64,
    /// Norm of force for the current structure.
    /// (This scales with sqrt(N), which makes it pretty useless actually.
    ///  we should fix that...)
    pub grad_norm: f64,
    /// Norm of force, rescaled as an intensive property.
    pub grad_rms: f64,
    /// The iteration that is about to occur.
    /// This index is 1-based since we're counting the fence segments,
    ///  not the fenceposts.
    pub iterations: u32,
}

pub use self::stop_condition::Rpn as StopCondition;
pub use self::stop_condition::Cereal as StopConditionSettings;
pub mod stop_condition {
    use ::stop_condition::prelude::*;
    use super::Objectives;

    #[derive(Serialize, Deserialize)]
    #[derive(Debug, Copy, Clone, PartialEq)]
    pub enum Simple {
        #[serde(rename = "value-delta")] ValueDelta(f64),
        #[serde(rename =    "grad-max")] GradientMax(f64),
        #[serde(rename =   "grad-norm")] GradientNorm(f64),
        #[serde(rename =    "grad-rms")] GradientRms(f64),
        #[serde(rename =  "iterations")] Iterations(u32),
    }

    impl ShouldStop<Objectives> for Simple {
        fn should_stop(&self, objs: &Objectives) -> bool {
            match *self {
                Simple::ValueDelta(tol) => match objs.delta_value {
                    Some(x) => x.abs() <= tol,
                    None => false, // first iteration; can't test dv
                },
                Simple::GradientMax(tol) => objs.grad_max <= tol,
                Simple::GradientNorm(tol) => objs.grad_norm <= tol,
                Simple::GradientRms(tol) => objs.grad_rms <= tol,
                Simple::Iterations(n) => objs.iterations >= n,
            }
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

pub trait DiffFn<E>: FnMut(&[f64]) -> Result<(f64, Vec<f64>), E> { }
impl<E, F> DiffFn<E> for F
where F: FnMut(&[f64]) -> Result<(f64, Vec<f64>), E> { }

#[derive(Serialize, Deserialize)]
#[derive(Debug, Clone)]
pub enum Error<E> {
    /// General variant for unrecoverable errors that are not worth panicking on,
    /// and that are probably not worth matching on the caller.
    Generic(String),
    /// The potential produced an error.
    ComputeError(E),
}
impl<E> Error<E> {
    fn string(s: &str) -> Error<E> { Error::Generic(s.to_string()) }
}
impl<E> From<E> for Failure<E> {
    fn from(e: E) -> Self { Failure::from_error(Error::ComputeError(e)) }
}

use linesearch::Error as LsError;
impl<E> From<LsError<E>> for Failure<E> {
    fn from(e: LsError<E>) -> Self { Failure::from_error(match e {
        LsError::ComputeError(e) => Error::ComputeError(e),
        LsError::Generic(s) => Error::Generic(s),
        LsError::NoImprovement => Error::Generic("linesearch failure".to_string()),
        LsError::Uphill => panic!("bug! (acgsd tried to linesearch uphill!)"),
        LsError::NonFiniteAlpha => panic!("unused code path; tested elsewhere")
    })}
}

/// An error type extended with some additional data.
#[derive(Serialize, Deserialize)]
#[derive(Debug, Clone)]
pub struct Failure<E> {
    /// The best position found prior to the failure, for those who feel exceptionally lucky.
    ///
    /// Might not always be available due to corners cut in error branches
    /// inside the acgsd implementation.
    pub best_position: Option<Vec<f64>>,
    pub error: Error<E>,
}

impl<E> Failure<E> {
    // Used by `?` in acgsd
    fn from_error(e: Error<E>) -> Self {
        Failure {
            // Best we can do at this high level.  To recover best positions
            // in more cases, we'll need to replace a lot of `?` with explicit returns.
            best_position: None,
            error: e,
        }
    }
}


#[derive(Serialize, Deserialize)]
#[derive(Debug, Clone)]
pub struct Output {
    pub iterations: i64,
    pub position: Vec<f64>,
    pub gradient: Vec<f64>,
    pub value: f64,
    // ensures addition of new fields is backwards compatible
    #[serde(skip)]
    #[allow(non_snake_case)]
    __no_full_destructure: (),
}

#[inline(never)]
pub fn acgsd<E, F: DiffFn<E>>(
    settings: &Settings,
    initial_position: &[f64],
    mut compute: F,
) -> Result<Output, Failure<E>>
where F: FnMut(&[f64]) -> Result<(f64, Vec<f64>), E>
{
    use ::util::cache::MinCacheBy;
    let stop_condition = self::stop_condition::Rpn::from_cereal(&settings.stop_condition);

    let mut compute_point = |position: &[f64]| {
        let position = position.to_vec();
        let (value, gradient) = compute(&position)?;
        // FIXME type annotation is only due to dumb conversions
        //       performed where map_err would be more robust
        Ok::<_,E>(Point {position, value, gradient})
    };

// /////////////////////////////////////////////////////////////////////////////
// Types                                                                      //
// /////////////////////////////////////////////////////////////////////////////

    #[derive(Debug,Clone)]
    struct Point {
        position: Vec<f64>,
        gradient: Vec<f64>,
        value: f64,
    };

    #[derive(Debug,Clone)]
    struct Saved {
        alpha: f64,
        position: Vec<f64>,
        gradient: Vec<f64>,
        value: f64,
    };

    impl Saved {
        pub fn into_point(self) -> Point {
            let Saved { position, gradient, value, .. } = self;
            Point { position, gradient, value }
        }
        pub fn to_point(&self) -> Point { self.clone().into_point() }
    }

    #[derive(Debug,Clone)]
    struct Last {
        direction: Vec<f64>,   // direction searched (normalized)

        // NOTE: These next three are all zero when linesearch has failed.
        //       This can be a problem for d_value in particular.
        d_value: f64,          // change in value
        d_position: Vec<f64>,  // change in position
        d_gradient: Vec<f64>,  // change in gradient

        ls_failed: bool,       // linesearch failed?
    }

// /////////////////////////////////////////////////////////////////////////////
// Loop start                                                                 //
// /////////////////////////////////////////////////////////////////////////////

    // These are all updated only at the end of an iteration.
    let mut last_saved = {
        let point = compute_point(initial_position)?;
        let Point { position, value, gradient } = point;
        Saved { alpha: 1.0, position, value, gradient }
    };

    // Describes the previous iteration
    let mut last_last = None::<Last>; // FIXME name
    // Record of previous directions
    let mut past_directions: VecDeque<Vec<f64>> = Default::default();

    // deliberately spelt plural as it counts how many have elapsed
    for iterations in 0.. {

        // Move these out so we can freely borrow from them without needing
        //  to scope the borrows.
        let saved = last_saved;
        let last = last_last;

// /////////////////////////////////////////////////////////////////////////////
// Closures for things commonly done during an iteration                      //
// /////////////////////////////////////////////////////////////////////////////

        // Compute at a position relative to saved.position
        let mut compute_in_dir = |alpha, direction: &[f64]| {
            let V(position): V<Vec<f64>> = v(&saved.position) + alpha * v(direction);
            compute_point(&position)
        };

        let warning = |msg: &str, alpha, point: Point|
        {
            if settings.has_verbosity(1) {
                println!("{}", msg);
                println!("Iterations: {}", iterations);
                println!("     Alpha: {}", alpha);
                println!("     Value: {}", point.value);
                println!(" Grad Norm: {}
                ", vnorm(&point.gradient));
                // if (additional_output)
                //     additional_output(cerr);
            }
        };

        // use as 'return fatal(...);'
        // this will return or throw based on the 'except_on_fail' setting
        let fatal = |msg: &str, alpha, point| {
            let msg = format!("ACGSD Failed: {}", msg);
            warning(&msg, alpha, point);
            panic!("ACGSD Failed: {}", msg);
            // FIXME
            //
            // if (settings.except_on_fail)
            //     throw runtime_error(msg);
            // return point.position;
        };

// /////////////////////////////////////////////////////////////////////////////
// Per-iteration output                                                       //
// /////////////////////////////////////////////////////////////////////////////

        if settings.has_verbosity(2) {
            let d_value = last.as_ref().map(|l| l.d_value).unwrap_or(0.0);
            let grad_mag = vnorm(&saved.gradient);
            print!(" i: {:>6}", iterations);
            print!("  v: {:18.14}", saved.value);
            print!(" dv: {:13.7e}", d_value);
            print!("  g: {:13.7e}", grad_mag);

            let cosines = {
                let mut s = String::new();
                let mut dirs = past_directions.iter();
                if dirs.len() >= 2 {
                    write!(&mut s, "  cosines:").unwrap();
                    let latest = dirs.next().unwrap();
                    for other in dirs {
                        write!(&mut s, " {:>6.2}", vdot(latest, other)).unwrap();
                    }
                }
                s
            };
            print!("{:<28}", cosines);
            use ::reporting::Bins;
            let grad_data = saved.gradient.iter().map(|&x| NotNaN::new(x.abs()).unwrap()).collect_vec();
            let &grad_max = grad_data.iter().max().unwrap();
            let grad_fracs = {
                if grad_max == NotNaN::new(0.0).unwrap() { grad_data }
                else { grad_data.iter().map(|&x| x / grad_max).collect() }
            };
            let divs = vec![0.0, 0.05, 0.40, 0.80, 1.0].into_iter().map(|x| NotNaN::new(x).unwrap()).collect_vec();
            let bins = Bins::from_iter(divs, grad_fracs);
            print!(" {:40} {}", bins.display(), *bins.as_counts().last().unwrap() );
            println!();
        }

        // // call the output function if applicable
        // if settings.intermediate_output_interval > 0 &&
        //     iterations % settings.intermediate_output_interval == 0
        // {
        //     settings.output_fn(saved.position)
        // };

// /////////////////////////////////////////////////////////////////////////////
// Evaluate exit conditions                                                   //
// /////////////////////////////////////////////////////////////////////////////

        { // scope
            let gnorm = vnorm(&saved.gradient);
            let objectives = Objectives {
                grad_norm: gnorm,
                grad_rms: gnorm / (saved.gradient.len() as f64).sqrt(),
                grad_max: max_norm(&saved.gradient),
                delta_value: last.as_ref().and_then(
                    |old|
                        if old.ls_failed { None }
                        else { Some(old.d_value) }
                ),
                iterations: iterations as u32, // FIXME remove cast?
            };

            if stop_condition.should_stop(&objectives) {
                if settings.has_verbosity(1) {
                    println!("ACGSD Finished.");
                    println!("Iterations: {}", objectives.iterations);
                    println!("     Value: {}", saved.value);
                    println!(" Delta Val: {}", objectives.delta_value.unwrap_or(0.0));
                    println!(" Grad Norm: {}", objectives.grad_norm);
                    println!("  Grad Max: {}", objectives.grad_max);
                }

                let Point { position, value, gradient } = saved.to_point();
                return Ok(Output { iterations, position, value, gradient, __no_full_destructure: () });
            }
        } // scope

////////////////////////////////////////////////////////////////////////////////
// Calculate the search direction.                                            //
////////////////////////////////////////////////////////////////////////////////

        // whether or not we will use steepest descent
        // note: force us to use steepest descent if linesearch failed
        // last iteration
        //
        // an loop/break is used for complex control flow
        let direction = 'use_dir: loop { break {

            // Consider the direction  'beta * dx - g'
            if let &Some(Last{
                ls_failed: false,
                ref d_position,
                ref d_gradient,
                ..
            }) = &last
            {
                let beta = calc_beta_acgsd(&saved.gradient, d_position, d_gradient);

                let V(direction) = beta * v(d_position) - v(&saved.gradient);

                // use this direction unless it is almost directly uphill
                if !should_revert_acgsd(&saved.gradient, &direction) {
                    break 'use_dir direction;
                }
            }

            // Fallback to steepest descent:  '-g'
            if settings.has_verbosity(2) {
                println!("Using steepest descent.");
            }

            let V(direction) = -v(&saved.gradient);
            direction
        }}; // 'use_dir: loop { break { ... } }

        // NOTE: The original source scaled alpha instead of normalizing
        //       direction, which seems to be a fruitless optimization
        //       that only serves to amplify the mental workload.
        let V(direction) = vnormalize(&direction).unwrap_or_else(|_| {
            // could happen for zero force; let's soldier on.
            warn!("non-normalizable direction; using arbitrary direction");
            let mut vec = vec![0.0; direction.len()];
            vec[0] = 1.0;
            v(vec)
        }); // FIXME use Err()

////////////////////////////////////////////////////////////////////////////////
// Perform the linesearch.                                                    //
////////////////////////////////////////////////////////////////////////////////

        // (these cache the best computation by linesearch)
        let mut ls_alpha = 0.0;
        let mut ls_point = saved.to_point();

        // Linesearch along direction for a better point.
        // It is possible that no better point will be found, in which case
        //  the displacement returned will naturally be zero.
        // FIXME type annotation is only due to dumb conversions
        //       performed where map_err would be more robust
        let next_alpha = ::new_linesearch::linesearch
            ::<::linesearch::Error<E>, _>
            (
            &Default::default(),
            saved.alpha,
            {
                let mut cache = ::std::collections::HashMap::new();
                let direction = &direction;
                let compute_in_dir = &mut compute_in_dir;
                let ls_point = &mut ls_point;
                let ls_alpha = &mut ls_alpha;
                let last = &last;
                move |alpha| {
                    let key = ::ordered_float::NotNaN::new(alpha).unwrap();
                    // can't use entry api due to Result
                    if !cache.contains_key(&key) {
                        let point = compute_in_dir(alpha, direction)
                            .map_err(::linesearch::Error::ComputeError)?; // HACK
                        let slope = vdot(&point.gradient, direction);

                        // update cache, checking values to predict which
                        //  point linesearch will prefer to use.
                        // (future additions to linesearch may make this less reliable)
                        if point.value < ls_point.value {
                            *ls_alpha = alpha;
                            *ls_point = point.clone();
                        }

                        if let Some(Last { ls_failed: true, .. }) = *last {
                            if settings.has_verbosity(1) {
                                print!("LS: a: {:.14e}", alpha);
                                print!("\tv: {:14e}", point.value);
                                print!("\ts: {:14e}", slope);
                                println!("");
                            }
                        }
                        cache.insert(key, (point.value, slope));
                    }
                    Ok(cache[&key])
                }
            }
        )?;
        let next_point = match next_alpha {
            a if a == ls_alpha => ls_point, // extraneous computation avoided!
            a => compute_in_dir(a, &direction)?,
        };

        // if the linesearch failed, note it and try
        //  again next iteration with steepest descent
        let ls_failed = (next_alpha == 0.0);
        if ls_failed {
            if let Some(Last { ls_failed: true, .. }) = last {
                return fatal("linesearch failure (second)", saved.alpha, saved.to_point());
                // return fatal(
                //     "linesearch failure (second)", saved.alpha, saved.point(),
                //     [&](ostream &out) {
                //         let test_dir = saved.gradient;
                //         vnormalize(test_dir);

                //         double numerical = util::central_difference<9, double>(
                //             [&](double a) {
                //                 return compute_in_dir(a, test_dir).value;
                //             }, 0.0, 1e-3);

                //         out << "Numerical gradient: "
                //             << setprecision(14) << numerical << endl;
                //     }
                // );
            } else {
                warning(
                    "Linesearch failure, switching to steepest descent",
                    saved.alpha, saved.to_point());
            }
        }

////////////////////////////////////////////////////////////////////////////////
// Update quantities following the linesearch.                                //
////////////////////////////////////////////////////////////////////////////////

        // NOTE: If linesearch failed, then next_alpha was zero and so
        //       next_point is saved.point(). Hence, all of these will be zero.
        let d_value = next_point.value - saved.value;
        let V(d_position) = v(&next_point.position) - v(&saved.position);
        let V(d_gradient) = v(&next_point.gradient) - v(&saved.gradient);

        past_directions.push_front(direction.clone());
        if past_directions.len() > 4 {
            past_directions.pop_back();
        };
        last_last = Some(Last { direction, d_value, d_position, d_gradient, ls_failed });
        last_saved = match ls_failed {
            true => saved.clone(),
            false => {
                let alpha = next_alpha;
                let Point { position, value, gradient } = next_point;
                Saved { alpha, position, value, gradient }
            },
        };
    }
    panic!("too many iterations")
}

/// calculate beta (ACGSD)
fn calc_beta_acgsd(gradient: &[f64], delta_x: &[f64], delta_g: &[f64]) -> f64
{
    // inner products needed for beta
    let dg_dx = vdot(delta_g, delta_x); // dg.dx
    let g_dg = vdot(gradient, delta_g); // g.dg
    let g_dx = vdot(gradient, delta_x); // g.dx

    (g_dg / dg_dx) * (1.0 - g_dx / dg_dx)
}

fn max_norm(v: &[f64]) -> f64 {
    let mut acc = 0f64;
    for x in v { acc = acc.max(x.abs()); }
    acc
}

/// whether the search should revert to steepest descent (ACGSD)
fn should_revert_acgsd(gradient: &[f64], direction: &[f64]) -> bool {
    // Continue as long as the search direction is at least slightly downhill.
    // Otherwise, revert.
    vdot(gradient, direction) > -1e-3 * vnorm(gradient) * vnorm(direction)
}

#[cfg(test)]
mod tests {

    use ::util::Never;
    use ::itertools::Itertools;
    use super::Settings;

    type NoFailResult = Result<(f64, Vec<f64>), Never>;

    // For some point `P`, get a potential function of the form `V(X) = (X - P).(X - P)`.
    // This is a macro pending stabilization of `impl Trait`
    macro_rules! quadratic_test_fn {
        ($target: expr) => {{
            use ::test_functions::one_dee::prelude::*;
            use ::test_functions::one_dee::Polynomial;

            let target = {$target}.to_vec();
            let polys = target.iter().map(|&x| Polynomial::x_n(2).recenter(-x)).collect_vec();
            let derivs = polys.iter().map(|p| p.derivative()).collect_vec();
            move |x: &[f64]| Ok::<_, Never>((
                izip!(x, polys.clone()).map(|(&x, p)| p.evaluate(x)).sum(),
                izip!(x, derivs.clone()).map(|(&x, d)| d.evaluate(x)).collect(),
            ))
        }};
    }

    macro_rules! scale_y {
        ($scale: expr, $f: expr) => {{
            let scale = $scale;
            let f = $f;
            move |x: &[f64]| {
                let (value, gradient): (f64, Vec<f64>) = f(x)?;
                Ok::<_, Never>((
                    scale * value,
                    gradient.into_iter().map(|x| x * scale).collect_vec(),
                ))
            }
        }}
    }

    // A high-level "it works" test.
    #[test]
    fn simple_quadratic() {
        use ::util::random::uniform_n;

        let target = uniform_n(15, -10.0, 10.0);
        let initial_point = uniform_n(15, -10.0, 10.0);
        let settings: Settings = from_json!({"stop-condition": {"grad-max": 1e-11}});
        let result = super::acgsd(&settings, &initial_point, quadratic_test_fn!(&target)).unwrap();
        for (a,b) in izip!(result.position, target) {
            assert_close!(a, b);
        }
    }

    // Test that tolerance tests can succeed as early as they ought to be capable of,
    //  by using absurdly large tolerances.
    #[test]
    fn insta_finish() {
        use ::util::random::uniform_n;

        // constant potential
        let point = uniform_n(18, -10.0, 10.0);
        let target = uniform_n(18, -10.0, 10.0);

        let s = from_json!({"stop-condition": {"grad-max": 1e20}});
        assert_eq!(super::acgsd(&s, &point, quadratic_test_fn!(&target)).unwrap().iterations, 0);

        let s = from_json!({"stop-condition": {"grad-norm": 1e20}});
        assert_eq!(super::acgsd(&s, &point, quadratic_test_fn!(&target)).unwrap().iterations, 0);

        // note: value-delta can only be tested after at least one iteration
        let s = from_json!({"stop-condition": {"value-delta": 1e20}});
        assert_eq!(super::acgsd(&s, &point, quadratic_test_fn!(&target)).unwrap().iterations, 1);
    }

    // A test to make sure some physicist doesn't stick an ill-conceived
    // comparison against absolute magnitude somewhere. Did it work?
    //
    // (For reference, the minimum and maximum normal floating point numbers
    //   have values around 1e-300 and 1e+300.  Which is to say, when it comes
    //   to uniform scale factors, WE HAVE WIGGLE ROOM.)
    #[test]
    fn scale_agnosticity() {
        use ::util::random::uniform_n;

        let target = uniform_n(15, -10.0, 10.0);
        let start = uniform_n(15, -10.0, 10.0);

        // Scale the "simple_quadratic" test by uniform scale factors

        let settings = from_json!({"stop-condition": {"grad-max": 1e-50}});
        let potential = scale_y!(1e-40, quadratic_test_fn!(&target));
        let result = super::acgsd(&settings, &start, potential).unwrap();
        for (&from, x, &targ) in izip!(&start, result.position, &target) {
            assert_ne!(from, x);
            assert_close!(abs=0.0, rel=1e-9, x, targ);
        }

        let settings = from_json!({"stop-condition": {"grad-max": 1e+30}});
        let potential = scale_y!(1e+40, quadratic_test_fn!(&target));
        let result = super::acgsd(&settings, &start, potential).unwrap();
        for (&from, x, &targ) in izip!(&start, result.position, &target) {
            assert_ne!(from, x);
            assert_close!(abs=0.0, rel=1e-9, x, targ);
        }
    }

    #[test]
    fn zero_force_stability() {
        // constant potential
        fn potential(_: &[f64]) -> NoFailResult { Ok((0.0, vec![0.0; 18])) }
        let point = ::util::random::uniform_n(18, -10.0, 10.0);

        // FIXME there should be a setting for whether linesearch failure is considered successful;
        //       in the case of this test, it is.
        //       For now, we limit ourselves to 1 iteration just to avoid angering the dragon.

        // zero force can be an edge case for numerical stability;
        // check that the position hasn't become NaN.
        let settings = from_json!({"stop-condition": {"iterations": 1}});
        assert_eq!(super::acgsd(&settings, &point, potential).unwrap().position, point);
    }

    #[test]
    fn test_iterations_stop_condition() {
        use ::util::random::uniform_n;
        let target = uniform_n(15, -10.0, 10.0);
        let start = uniform_n(15, -10.0, 10.0);

        // interations: 0 should cause no change in position despite not being at the minimum
        let settings = from_json!({"stop-condition": {"iterations": 0}});
        let result = super::acgsd(&settings, &start, quadratic_test_fn!(&target)).unwrap();
        assert_eq!(result.iterations, 0);
        assert_eq!(result.position, start);

        // interations: 1 should cause a change in position
        let settings = from_json!({"stop-condition": {"iterations": 1}});
        let result = super::acgsd(&settings, &start, quadratic_test_fn!(&target)).unwrap();
        assert_eq!(result.iterations, 1);
        assert_ne!(result.position, start);
    }

    #[test]
    fn trid() {
        use ::test_functions::n_dee::{Trid, OnceDifferentiable};
        use ::util::random::uniform_n;
        let d = 10;
        for _ in 0..10 {
            let max_coord = (d*d) as f64;
            let output = super::acgsd(
                &from_json!({"stop-condition": {"grad-rms": 1e-8}}),
                &uniform_n(d, -max_coord, max_coord),
                |p| Ok::<_,Never>(Trid(d).diff(p))
            ).unwrap();

            assert_close!(rel=1e-5, output.value, Trid(d).min_value());
            for (a,b) in izip!(output.position, Trid(d).min_position()) {
                assert_close!(rel=1e-5, a, b);
            }
        }
    }

    #[test]
    fn lj() {
        use ::sp2_slice_math::{v,V};
        use ::test_functions::n_dee::{HyperLennardJones, OnceDifferentiable, Sum};
        use ::util::random as urand;
        let d = 10;

        for _ in 0..10 {
            // Place two fixed "atoms" so that their minimal potential surfaces
            // intersect at a single point.
            let V(desired_min_point) = urand::uniform(1.0, 20.0) * v(urand::direction(d));
            let radius_1 = urand::uniform(2.0, 3.0);
            let radius_2 = urand::uniform(2.0, 3.0);
            let dipole_dir = urand::direction(d);
            let V(point_1) = v(&desired_min_point) + radius_1 * v(&dipole_dir);
            let V(point_2) = v(&desired_min_point) - radius_2 * v(&dipole_dir);

            let mut diff = Sum(
                HyperLennardJones { ndim: d, min_radius: radius_1, min_value: -13.0 }
                    .displace(point_1.clone()),
                HyperLennardJones { ndim: d, min_radius: radius_2, min_value:  -7.0 }
                    .displace(point_2.clone()),
            );

            // a point somewhere in the box whose corners are the atoms
            let start = urand::uniform_box(&point_1, &point_2);
            let settings = from_json!({"stop-condition": {"grad-rms": 1e-10}});
            let output = super::acgsd(&settings, &start, |p| Ok::<_,Never>(diff.diff(p))).unwrap();

            assert_close!(rel=1e-12, output.value, -20.0);
        }
    }
}