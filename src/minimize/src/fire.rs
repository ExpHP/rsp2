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

#![allow(non_snake_case)]

// HERE BE DRAGONS

use std::fmt;

use either::{Either, Left, Right};
use rsp2_slice_math::{vnorm, vdot, V, v, vnormalize};

#[derive(Serialize, Deserialize)]
#[derive(Debug, Clone, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub struct Params {
    /// Force damping coefficient (alpha from original FIRE paper).
    #[serde(default = "params__damping_init")] pub damping_init: f64,
    /// Value less than 1 used to decrease alpha.
    #[serde(default = "params__damping_reduction")] pub damping_reduction: f64,
    /// Prevent speedup for this many steps before.
    #[serde(default = "params__speedup_latency")] pub speedup_latency: u64,
    /// Value less than 1 used when timestep is increased.
    #[serde(default = "params__speedup_factor")] pub speedup_factor: f64,
    /// Value less than 1 used when timestep is reduced.
    #[serde(default = "params__slowdown_factor")] pub slowdown_factor: f64,
    #[serde(default = "params__integrator")] pub integrator: Integrator,
    pub timestep_max: f64,
    /// Uniform mass for all atoms.  Only affects the difference in scale
    /// between velocity and force.  This should generally be a value of
    /// vaguely similar magnitude to the forces.
    #[serde(default = "params__mass")] pub mass: f64,
}

impl Params {
    pub fn new(timestep_max: f64) -> Params {
        from_json!({ "timestep-max": timestep_max })
    }
}

#[derive(Serialize, Deserialize)]
#[derive(Debug, Clone, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum Integrator {
    /// Semi-implicit Euler integration recommended by
    /// <https://www.sciencedirect.com/science/article/pii/S0927025620300756>
    EulerImplicit {},
}

fn params__damping_init() -> f64 { 0.1 }
fn params__damping_reduction() -> f64 { 0.99 }
fn params__speedup_latency() -> u64 { 5 }
fn params__speedup_factor() -> f64 { 1.1 }
fn params__slowdown_factor() -> f64 { 0.5 }
fn params__integrator() -> Integrator { Integrator::EulerImplicit {} }
fn params__mass() -> f64 { 1. }

pub use crate::cg::StopCondition;

//==================================================================================================
// Errors

#[derive(Debug, Fail)]
pub enum FireError {
    #[doc(hidden)]
    #[fail(display = "impossible!")]
    _Hidden,
}

/// A wrapper type used to allow `?` to work on errors from the user function.
// (FIXME: honestly though we're probably better off without it so that
//         we can fill in the `best_position` field)
struct ComputeError<E>(E);
impl<E> From<FireError> for Failure<E> {
    fn from(e: FireError) -> Self {
        Failure {
            best_position: None,
            error: Left(e),
        }
    }
}
impl<E> From<ComputeError<E>> for Failure<E> {
    fn from(ComputeError(e): ComputeError<E>) -> Self {
        Failure {
            best_position: None,
            error: Right(e),
        }
    }
}

/// An error type extended with some additional data.
#[derive(Debug)]
pub struct Failure<E> {
    /// The best position found prior to the failure, for those who feel exceptionally lucky.
    ///
    /// Might not always be available due to corners cut in error branches
    /// inside the acgsd implementation.
    pub best_position: Option<Vec<f64>>,
    pub error: Either<FireError, E>,
}

impl<E: fmt::Display> fmt::Display for Failure<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result
    { self.error.fmt(f) }
}

//==================================================================================================
// Builder API

// Alias to indicate that the reason for a field having type Option is that it has no default.
// (it must be supplied)
type Required<T> = Option<T>;

/// Provides a significantly greater level of customization for performing conjugate gradient.
///
/// NOTE: A freshly-constructed `Builder` has no stop condition; you must call
/// [`Builder::stop_condition`] before calling [`Builder::run`].
#[must_use]
pub struct Builder {
    build_stop_condition: Required<Box<dyn BuildAlgorithmStateFn<Output=bool>>>,
    build_output_fns: Vec<Box<dyn BuildAlgorithmStateFn<Output=()>>>,
    params: Params,
}

impl Builder {
    pub fn new(params: &Params) -> Self {
        Builder {
            build_stop_condition: None,
            build_output_fns: vec![],
            params: params.clone(),
        }
    }

    /// Set up an arbitrary function for logging output each iteration.
    ///
    /// This will exist alongside any previously existing output functions.
    pub fn output_fn(&mut self, f: impl BuildAlgorithmStateFn<Output=()> + 'static) -> &mut Self {
        self.build_output_fns.push(Box::new(f)); self
    }

    /// Set up a "standard" output function that writes some formatted lines on each iteration.
    /// The output format may change.
    ///
    /// Example usage: `builder.basic_output_fn(|s| println!("{}", s))`
    ///
    /// This will exist alongside any previously existing output functions.
    pub fn basic_output_fn(&mut self, emit: impl Clone + FnMut(fmt::Arguments<'_>) + 'static) -> &mut Self {
        self.output_fn(crate::cg::get_basic_output_fn(emit))
    }

    pub fn stop_condition(&mut self, f: impl BuildAlgorithmStateFn<Output=bool> + 'static) -> &mut Self {
        self.build_stop_condition = Some(Box::new(f)); self
    }
}

impl Clone for Builder {
    fn clone(&self) -> Self {
        Builder {
            build_output_fns: self.build_output_fns.iter().map(|x| objekt::clone_box(&**x)).collect(),
            build_stop_condition: self.build_stop_condition.as_ref().map(|x| objekt::clone_box(&**x)),
            params: self.params.clone(),
        }
    }
}

//==================================================================================================
// closures in builder API

/// FIXME the fields of this don't totally make sense for FIRE,
///       but if I make a separate type then we need separate output fns...
pub use crate::cg::AlgorithmState;
pub use crate::cg::BuildAlgorithmStateFn;

//==================================================================================================
// Primary public API

#[derive(Serialize, Deserialize)]
#[derive(Debug, Clone)]
pub struct Output {
    pub iterations: u64,
    pub position: Vec<f64>,
    pub velocity: Vec<f64>,
    pub gradient: Vec<f64>,
    pub value: f64,
    pub timestep: f64,
    // ensures addition of new fields is backwards compatible
    #[serde(skip)]
    #[allow(non_snake_case)]
    __no_full_destructure: (),
}

impl Builder {
    pub fn run<F: DiffFn>(
        &self,
        initial_position: &[f64],
        compute: F,
    ) -> Result<Output, Failure<F::Error>>
    { fire(self, initial_position, compute) }
}

pub use crate::cg::DiffFn;

//==================================================================================================

// Types used inside the implementation of acgsd.
#[derive(Debug, Clone)]
pub(crate) struct Saved {
    pub(crate) fsm: FireFsm,
    pub(crate) position: Vec<f64>,
    pub(crate) gradient: Vec<f64>,
    pub(crate) velocity: Vec<f64>,
    pub(crate) value: f64,
}

use fsm::*;
mod fsm {
    use super::*;

    #[derive(Debug, Clone)]
    pub struct FireFsm {
        params: Params,
        damping_coeff: f64,
        timestep: f64,
        num_good_steps: u64,
    }
    #[derive(Debug, Clone)]
    pub struct FireFsmOutput {
        pub timestep: f64,
        pub damping_coeff: f64,
        pub should_reset_velocity: bool,
    }

    impl FireFsm {
        pub fn new(params: &Params) -> Self {
            FireFsm {
                params: params.clone(),
                damping_coeff: params.damping_init,
                timestep: params.timestep_max * 0.1,
                num_good_steps: 0,
            }
        }

        pub fn output(&self) -> FireFsmOutput {
            let should_reset_velocity = self.num_good_steps == 0;
            FireFsmOutput {
                timestep: self.timestep,
                damping_coeff: self.damping_coeff,
                should_reset_velocity,
            }
        }

        pub fn check_grad_dot_vel(mut self, dot: f64) -> FireFsm {
            if dot < 0.0 {
                self.num_good_steps += 1;
                if self.num_good_steps >= self.params.speedup_latency {
                    self.timestep *= self.params.speedup_factor;
                    self.damping_coeff *= self.params.damping_reduction;
                }
            } else {
                self.num_good_steps = 0;
                self.timestep *= self.params.slowdown_factor;
                self.damping_coeff = self.params.damping_init;
            }
            self.timestep = f64::min(self.timestep, self.params.timestep_max);
            self
        }
    }
}

#[inline(never)]
fn fire<F: DiffFn>(
    builder: &Builder,
    initial_position: &[f64],
    mut diff_fn: F,
) -> Result<Output, Failure<F::Error>>
{
    let params = &builder.params;

    let mut stop_condition = match builder.build_stop_condition {
        Some(ref f) => f.build(),
        None => panic!("'stop_condition' was not supplied to the CG Builder!"),
    };

    let mut output_functions: Vec<_> = {
        builder.build_output_fns.iter().map(|x| x.build()).collect()
    };

// /////////////////////////////////////////////////////////////////////////////
// Loop start                                                                 //
// /////////////////////////////////////////////////////////////////////////////

    let mut last_saved = {
        let (value, gradient) = diff_fn.compute(&initial_position).map_err(ComputeError)?;
        Saved {
            position: initial_position.to_vec(),
            velocity: vec![0.0; initial_position.len()],
            value,
            gradient,
            fsm: FireFsm::new(params),
        }
    };

    let dummy_direction = {
        let mut v = vec![0.0; initial_position.len()];
        v[0] = 1.0;
        v
    };

    // Remembers all values.
    let mut value_history = vec![last_saved.value];

    // deliberately spelt plural as it counts how many have elapsed
    for iterations in 0.. {
        // Move these out so we can freely borrow from them without needing
        //  to scope the borrows.
        let mut saved = last_saved;

// /////////////////////////////////////////////////////////////////////////////
// Per-iteration output & evaluate exit conditions                            //
// /////////////////////////////////////////////////////////////////////////////

        {
            let state = AlgorithmState {
                iterations,
                value: saved.value,
                gradient: &saved.gradient,
                position: &saved.position,
                direction: match iterations {
                    0 => None,
                    _ => Some(&dummy_direction), // FIXME dummy value
                },
                alpha: saved.fsm.output().timestep,
                __no_full_destructure: (),
            };

            for f in &mut output_functions {
                f(state.clone());
            }

            if stop_condition(state) {
                // FIXME this no longer belongs here, but...
                info!("FIRE Finished.");
                info!("Iterations: {}", iterations);
                info!("     Value: {}", saved.value);
                info!(" Grad Norm: {:e}", vnorm(&saved.gradient));
                info!("  Grad Max: {:e}", saved.gradient.iter().cloned().map(f64::abs).fold(0.0, f64::max));

                return Ok(Output {
                    iterations,
                    position: saved.position,
                    velocity: saved.velocity,
                    gradient: saved.gradient,
                    value: saved.value,
                    timestep: saved.fsm.output().timestep,
                    __no_full_destructure: (),
                });
            }
        } // scope

////////////////////////////////////////////////////////////////////////////////
// Do time integration.                                                       //
////////////////////////////////////////////////////////////////////////////////

        let FireFsmOutput {
            damping_coeff,
            timestep,
            should_reset_velocity,
        } = saved.fsm.output();

        if should_reset_velocity {
            saved.velocity = vec![0.0; saved.position.len()];
        }

        let (next_position, next_velocity, next_value, next_gradient);
        match params.integrator {
            Integrator::EulerImplicit {} => {

                let V(md_vel) = v(&saved.velocity) - (timestep / params.mass) * v(&saved.gradient);
                let md_vel_norm = vnorm(&md_vel);
                let V(grad_unit) = vnormalize(&saved.gradient).unwrap_or_else(|_| {
                    warn!("non-normalizable force; using arbitrary direction");
                    V(dummy_direction.clone())
                });
                // Fire adjustment
                next_velocity = {
                    (1.0 - damping_coeff) * v(md_vel)
                    - (damping_coeff * md_vel_norm) * v(&grad_unit)
                }.0;
                next_position = (v(&saved.position) + timestep * v(&next_velocity)).0;

                let diff = diff_fn.compute(&next_position).map_err(ComputeError)?;
                next_value = diff.0;
                next_gradient = diff.1;
            },
        }

        diff_fn.check(&next_position).map_err(ComputeError)?;

        last_saved = Saved {
            fsm: saved.fsm.check_grad_dot_vel(vdot(&next_gradient, &next_velocity)),
            position: next_position,
            gradient: next_gradient,
            velocity: next_velocity,
            value: next_value,
        };
        value_history.push(next_value);
    }
    panic!("too many iterations")
}

#[cfg(test)]
mod tests {
    use super::{Builder};
    use crate::util::Never;
    use itertools::Itertools;

    fn do_fire<F: super::DiffFn>(
        params: &super::Params,
        stop_condition: &super::StopCondition,
        initial_position: &[f64],
        compute: F,
    ) -> Result<super::Output, super::Failure<F::Error>> {
        Builder::new(params)
            .stop_condition(stop_condition.to_function())
            .run(initial_position, compute)
    }

    #[allow(unused)]
    fn do_fire_log<F: super::DiffFn>(
        params: &super::Params,
        stop_condition: &super::StopCondition,
        initial_position: &[f64],
        compute: F,
    ) -> Result<super::Output, super::Failure<F::Error>> {
        Builder::new(params)
            .stop_condition(stop_condition.to_function())
            .basic_output_fn(|args| println!("{}", args))
            .run(initial_position, compute)
    }


    type NoFailResult = Result<(f64, Vec<f64>), Never>;

    // For some point `P`, get a potential function of the form `V(X) = (X - P).(X - P)`.
    // This is a macro pending stabilization of `impl Trait`
    fn quadratic_test_fn(
        target: &[f64],
    ) -> impl FnMut(&[f64]) -> Result<(f64, Vec<f64>), Never> {
        use crate::test::one_dee::prelude::*;
        use crate::test::one_dee::Polynomial;

        let target = target.to_vec();
        let polys = target.iter().map(|&x| Polynomial::x_n(2).recenter(-x)).collect_vec();
        let derivs = polys.iter().map(|p| p.derivative()).collect_vec();
        move |x: &[f64]| Ok::<_, Never>((
            izip!(x, polys.clone()).map(|(&x, p)| p.evaluate(x)).sum(),
            izip!(x, derivs.clone()).map(|(&x, d)| d.evaluate(x)).collect(),
        ))
    }

    fn scale_y<E>(
        scale: f64,
        mut f: impl FnMut(&[f64]) -> Result<(f64, Vec<f64>), E>,
    ) -> impl FnMut(&[f64]) -> Result<(f64, Vec<f64>), E> {
        move |x: &[f64]| {
            let (value, gradient): (f64, Vec<f64>) = f(x)?;
            Ok((
                scale * value,
                gradient.into_iter().map(|x| x * scale).collect_vec(),
            ))
        }
    }

    // A high-level "it works" test.
    #[test]
    fn simple_quadratic() {
        use crate::util::random::uniform_n;

        let target = uniform_n(15, -10.0, 10.0);
        let initial_point = uniform_n(15, -10.0, 10.0);
        let result = do_fire(
            &from_json!({"timestep-max": 1e-2}),
            &from_json!({"grad-max": 1e-11}),
            &initial_point,
            quadratic_test_fn(&target),
        ).unwrap();
        assert_close!(result.position, target);
    }

    // Test that tolerance tests can succeed as early as they ought to be capable of,
    //  by using absurdly large tolerances.
    #[test]
    fn insta_finish() {
        use crate::util::random::uniform_n;

        // a potential where we are not initially at the minimum
        let point = uniform_n(18, -10.0, 10.0);
        let target = uniform_n(18, -10.0, 10.0);

        let params = from_json!({"timestep-max": 1e-2});
        let s = from_json!({"grad-max": 1e20});
        assert_eq!(do_fire(&params, &s, &point, quadratic_test_fn(&target)).unwrap().iterations, 0);

        let s = from_json!({"grad-norm": 1e20});
        assert_eq!(do_fire(&params, &s, &point, quadratic_test_fn(&target)).unwrap().iterations, 0);

        // note: value-delta can only be tested after at least one iteration
        let s = from_json!({
            "value-delta": {
                "rel-greater-than": -1e100,
                "steps-ago": 2,
            }
        });
        assert_eq!(do_fire(&params, &s, &point, quadratic_test_fn(&target)).unwrap().iterations, 2);
    }

    // A test to make sure some physicist doesn't stick an ill-conceived
    // comparison against absolute magnitude somewhere. Did it work?
    //
    // (For reference, the minimum and maximum normal floating point numbers
    //   have values around 1e-300 and 1e+300.  Which is to say, when it comes
    //   to uniform scale factors, WE HAVE WIGGLE ROOM.)
    #[test]
    fn scale_agnosticity() {
        use crate::util::random::uniform_n;

        let target = uniform_n(15, -10.0, 10.0);
        let start = uniform_n(15, -10.0, 10.0);


        // Scale the "simple_quadratic" test by uniform scale factors.
        //
        // Note FIRE is not totally scale-agnostic;

        let params = from_json!({
            "timestep-max": 1e-2,
            "mass": 1e-50,
        });
        let stop_condition = from_json!({"grad-max": 1e-50});
        let potential = scale_y(1e-40, quadratic_test_fn(&target));
        let result = do_fire(&params, &stop_condition, &start, potential).unwrap();
        for (&from, x, &targ) in izip!(&start, result.position, &target) {
            assert_ne!(from, x);
            assert_close!(abs=0.0, rel=1e-9, x, targ);
        }

        let params = from_json!({
            "timestep-max": 1e-2,
            "mass": 1e+30,
        });
        let stop_condition = from_json!({"grad-max": 1e+30});
        let potential = scale_y(1e+40, quadratic_test_fn(&target));
        let result = do_fire(&params, &stop_condition, &start, potential).unwrap();
        for (&from, x, &targ) in izip!(&start, result.position, &target) {
            assert_ne!(from, x);
            assert_close!(abs=0.0, rel=1e-9, x, targ);
        }
    }

    #[test]
    fn zero_force_stability() {
        // constant potential
        fn potential(_: &[f64]) -> NoFailResult { Ok((0.0, vec![0.0; 18])) }
        let point = crate::util::random::uniform_n(18, -10.0, 10.0);

        let params = from_json!({"timestep-max": 1e-2});

        // FIXME there should be a setting for whether linesearch failure is considered successful;
        //       in the case of this test, it is.
        //       For now, we limit ourselves to 1 iteration just to avoid angering the dragon.

        // zero force can be an edge case for numerical stability;
        // check that the position hasn't become NaN.
        let stop_condition = from_json!({"iterations": 1});
        assert_eq!(do_fire(&params, &stop_condition, &point, potential).unwrap().position, point);
    }

    #[test]
    fn test_iterations_stop_condition() {
        use crate::util::random::uniform_n;
        let target = uniform_n(15, -10.0, 10.0);
        let start = uniform_n(15, -10.0, 10.0);

        let params = from_json!({"timestep-max": 1e-2});

        // interations: 0 should cause no change in position despite not being at the minimum
        let stop_condition = from_json!({"iterations": 0});
        let result = do_fire(&params, &stop_condition, &start, quadratic_test_fn(&target)).unwrap();
        assert_eq!(result.iterations, 0);
        assert_eq!(result.position, start);

        // interations: 1 should cause a change in position
        let stop_condition = from_json!({"iterations": 1});
        let result = do_fire(&params, &stop_condition, &start, quadratic_test_fn(&target)).unwrap();
        assert_eq!(result.iterations, 1);
        assert_ne!(result.position, start);
    }

    #[test]
    fn trid() {
        use crate::test::n_dee::{Trid, OnceDifferentiable};
        use crate::util::random::uniform_n;
        let d = 10;
        for _ in 0..10 {
            let max_coord = (d*d) as f64;
            let output = do_fire(
                &from_json!({"timestep-max": 1e-2}),
                &from_json!({"grad-rms": 1e-8}),
                &uniform_n(d, -max_coord, max_coord),
                |p: &_| Ok::<_,Never>(Trid(d).diff(p))
            ).unwrap();

            assert_close!(rel=1e-5, output.value, Trid(d).min_value());
            assert_close!(rel=1e-5, output.position, Trid(d).min_position());
        }
    }

    // FIXME: FIRE does TERRIBLE on this one and I'm not sure why
//    #[test]
//    fn lj() {
//        use rsp2_slice_math::{v, V};
//        use crate::test::n_dee::{HyperLennardJones, OnceDifferentiable, Sum};
//        use crate::util::random as urand;
//        let d = 10;
//
//        for _ in 0..10 {
//            let params = from_json!({
//                "timestep-max": 1e-2,
//            });
//
//            // Place two fixed "atoms" so that their minimal potential surfaces
//            // intersect at a single point.
//            let V(desired_min_point) = urand::uniform(1.0, 20.0) * v(urand::direction(d));
//            let radius_1 = urand::uniform(2.0, 3.0);
//            let radius_2 = urand::uniform(2.0, 3.0);
//            let dipole_dir = urand::direction(d);
//            let V(point_1) = v(&desired_min_point) + radius_1 * v(&dipole_dir);
//            let V(point_2) = v(&desired_min_point) - radius_2 * v(&dipole_dir);
//
//            let mut diff = Sum(
//                HyperLennardJones { ndim: d, min_radius: radius_1, min_value: -13.0 }
//                    .displace(point_1.clone()),
//                HyperLennardJones { ndim: d, min_radius: radius_2, min_value:  -7.0 }
//                    .displace(point_2.clone()),
//            );
//
//            // a point somewhere in the box whose corners are the atoms
//            let start = urand::uniform_box(&point_1, &point_2);
//            let stop_condition = from_json!({
//                "any": [
//                    // FIRE is not capable of converging as precisely as CG here
//                    {"grad-rms": 1e-2},
//                    {"iterations": 1000},
//                ],
//            });
//            let diff_fn = |p: &_| Ok::<_,Never>(diff.diff(p));
//            let output = do_fire_log(&params, &stop_condition, &start, diff_fn).unwrap();
//            assert_ne!(output.iterations, 1000);
//
//            assert_close!(rel=1e-3, output.value, -20.0);
//        }
//    }
}
