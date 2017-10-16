// HERE BE DRAGONS

use ::sp2_slice_math::{vnorm, vdot, V, v, vnormalize};
use ::stop_condition::prelude::*;

use ::itertools::Itertools;
use ::ordered_float::NotNaN;
use ::std::fmt::Write;
use ::std::collections::VecDeque;

use ::either::{Either, Left, Right};

pub use self::settings::Settings;
pub mod settings {
    //! Please do not manually construct anything in here.
    //! None of this is future-safe.
    //!
    //! Deserialize from a JSON literal instead.

    pub use super::stop_condition::Cereal as StopCondition;

    #[derive(Serialize, Deserialize)]
    #[derive(Debug, Clone, PartialEq)]
    #[serde(rename_all="kebab-case")]
    pub struct Settings {
        #[serde()] pub(super) stop_condition: StopCondition,
        #[serde(default)] pub(super) beta: Beta,
        #[serde(default)] pub(super) linesearch: Linesearch,
        #[serde(default="defaults::alpha_guess_first")]
        pub(super) alpha_guess_first: f64,
        #[serde(default)] pub(super) alpha_guess_max: Option<f64>,
    }

    #[derive(Serialize, Deserialize)]
    #[derive(Debug, Clone, PartialEq)]
    #[serde(rename_all="kebab-case")]
    pub enum Beta {
        Acgsd {
            /// The direction searched will always point at least this much downhill.
            #[serde(default="defaults::beta__acgsd__downhill_min")]
            downhill_min: f64,
        },

        Hager {
            // NOTE: This seems to play a role similar to Acgsd.downhill_min
            //        but it isn't *quite* the same...
            /// This... I'm not sure what this does.
            #[serde(default="defaults::beta__hager__eta")]
            eta: f64,
        },
    }

    #[derive(Serialize, Deserialize)]
    #[derive(Debug, Clone, PartialEq)]
    #[serde(rename_all="kebab-case")]
    pub enum Linesearch {
        Acgsd(::linesearch::Settings),
        Hager(::hager_ls::Settings),
    }

    impl Beta {
        pub fn validate(&self) {
            match *self {
                Beta::Acgsd { downhill_min } => {
                    assert!(0.0 < downhill_min && downhill_min <= 1.0);
                },
                Beta::Hager { eta } => {
                    assert!(0.0 < eta);
                },
            }
        }
    }

    impl Linesearch {
        pub fn validate(&self) {
            match *self {
                Linesearch::Acgsd(ref settings) => settings.validate(),
                Linesearch::Hager(ref settings) => settings.validate(),
            }
        }
    }

    impl Default for Beta {
        fn default() -> Self { from_json!({"hager": {}}) }
    }
    #[test] fn test_beta_default() { Beta::default(); }

    impl Default for Linesearch {
        fn default() -> Self { from_json!({"hager": {}}) }
    }
    #[test] fn test_linesearch_default() { Linesearch::default(); }

    // Default functions, since literals aren't supported (serde gh #368)
    mod defaults {
        #![allow(non_snake_case)]
        pub(crate) fn alpha_guess_first() -> f64 { 1.0 }
        pub(crate) fn beta__acgsd__downhill_min() -> f64 { 1e-3 }
        pub(crate) fn beta__hager__eta() -> f64 { 1e-2 }
    }
}

impl Settings {
    pub fn has_verbosity(&self, _level: i32) -> bool { true } // FIXME
}

/* ****************************************

# Hagar CG_DESCENT formulas

## As written (pages 170-171)

$$
% To the person reading this source code:
% No, there isn't anywhere you should look in particular to find
% a rendered copy of this, and I have not included a complete prelude.
% Just go find a live MathJAX renderer (here's one: http://www.hostmath.com/)
\newcommand{\vec}[1]{\mathbf{#1}}
\newcommand{\norm}[1]{\left\Vert#1\right\Vert}
\newcommand{\paren}[1]{\left(#1\right)}
\newcommand{\braced}[1]{\left\{#1\right\}}
\newcommand{\a}[1]{\alpha_{#1}}
\newcommand{\x}[1]{\vec x_{#1}}
\newcommand{\g}[1]{\vec g_{#1}}
\newcommand{\d}[1]{\vec d_{#1}}
\newcommand{\gdiff}[1]{\Delta\g{#1}}
\newcommand{\betaN}[1]{\beta^N_{#1}}
\newcommand{\betaNp}[1]{\beta'^N_{#1}}
\newcommand{\betaNbar}[1]{\overline{\beta}{}_{#1}^N}
\newcommand{\T}{^\intercal}
\newcommand{\minTwo}[2]{\min\!\paren{#1,\, #2}}
\newcommand{\maxTwo}[2]{\max\!\paren{#1,\, #2}}
\newcommand{\dhat}[1]{\boldsymbol{\hat{\mathbf{d}}}_{#1}}
\DeclareMathOperator{\normalize}{normalize}

\begin{align}
\x{k+1} &= \x{k} + \alpha_k \d{k} \\
\gdiff{k} &= \g{k+1} - \g{k} \\
\d{0} &= - \g{0} \\
\d{k+1} &= - \g{k+1} + \maxTwo{\betaN{k}}{\eta_k} \d{k}  \\
\betaN{k} &=
    \frac{1}{\d{k}\T\gdiff{k}}
    \paren{
        \gdiff{k} - 2\d{k} \frac{\norm{\gdiff{k}}^2}{\d{k}\T \gdiff{k}}
    }\T \g{k+1} \\
\eta_k &= \frac{-1}{\norm{\d{k}} \minTwo{\eta}{\norm{\g{k}}}}
\end{align}
$$

$alpha_k$ is the step size (which has no formula, but is found
according to linesearch and must satisfy certain conditions)
and $\mathbf{g}_k$ is the gradient at $\mathbf{x}_k$.

The quantities above are numbered so that in each case, the first
value has index zero.  This is not very useful for us, so we
renumber them such that the value with index k is always computed
during the kth iteration. To allow "fencepost quantities" such as
position, gradient and value to start at index 0, the iterations
are numbered starting from 1.

\begin{align}
           &&     \x{0} & ~\text{given.} \\
(k \ge 1). &&     \x{k} &= \x{k-1} + \alpha_k \d{k} \\
(k \ge 1). && \gdiff{k} &= \g{k} - \g{k-1} \\
           &&     \d{1} &= - \g{0} \\
(k \ge 2). &&     \d{k} &= - \g{k-1} + \maxTwo{\betaN{k}}{\eta_k} \d{k-1}  \\
(k \ge 2). && \betaN{k} &=
    \frac{1}{\d{k-1}\T\gdiff{k-1}}
    \paren{
        \gdiff{k-1} - 2\d{k-1} \frac{\norm{\gdiff{k-1}}^2}{\d{k-1}\T \gdiff{k-1}}
    }\T \g{k-1} \\

(k \ge 2). &&    \eta_k &= \frac{-1}{\norm{\d{k-1}} \minTwo{\eta}{\norm{\g{k-2}}}}
\end{align}

Now we see that $\eta_k$ actually depends on the gradient from two
positions ago, i.e. at the *beginning* of the previous iteration.
Sneaky!

Things get even simpler if we normalize our direction vector $\mathbf d_k$.
We perform the following substitutions:

* \( \mathbf{d}_k \to d_k \hat{\mathbf{d}}_k \)
* \( \alpha_k \to d_k^{-1} \alpha'_k \)
* \( \beta^N_k \to  d_k \beta'^N_k \)
* \( \eta_k \to  d_k \eta'_k \)

We end up with

\begin{align}
           &&      \x{0} & ~\text{given.} \\
(k \ge 1). &&      \x{k} &= \x{k-1} + \alpha'_k \dhat{k} \\
(k \ge 1). &&  \gdiff{k} &= \g{k} - \g{k-1} \\
           &&      \dhat{1} &= \normalize\paren{ - \g{0} }\\
(k \ge 2). &&      \dhat{k} &= \normalize\paren{ - \g{k-1} + \maxTwo{\betaNp{k}}{\eta'_k} \dhat{k-1}}  \\
(k \ge 2). && \betaNp{k} &=
    \frac{1}{\dhat{k-1}\T\gdiff{k-1}}
    \paren{
        \gdiff{k-1} - 2\dhat{k-1} \frac{\norm{\gdiff{k-1}}^2}{\dhat{k-1}\T \gdiff{k-1}}
    }\T \g{k-1} \\

(k \ge 2). &&    \eta'_k &= \frac{-1}{\minTwo{\eta}{\norm{\g{k-2}}}}
\end{align}

Ultimately, the only observable impact of this substitution outside the
simpler formulas is that the relative scale of $\alpha_k/\alpha_{k-1}$ changes
to include scale factors that were previously reflected in the direction vector;
this can impact the quality of a starting guess for $\alpha$.

However, the decision to normalize direction was, in fact, originally motivated
by existing code that used $\alpha_{k-1} \cdot \norm{\d{k-1}} / \norm{\d{k}}$ as
its starting guess for $\alpha{k}$!  Under our revised scheme, the initial guess
for $\alpha'_k$ becomes simply $\alpha'_{k-1}$, and therefore the would-be norm
of the direction vector prior to normalization truly is of no concern to us.

...one more thing.  You can normalize $\Delta \mathbf g$ as well.
You get

$$
\betaNp{k} &=
    \paren{\frac
        {c_k\dghat{k-1} - 2 \dhat{k-1}}
        {c_k^2}
    }\T\g{k-1},
    \qquad c_k = \dhat{k-1}\T \dghat{k-1}.\\
$$

*/

pub use self::stop_condition::Rpn as StopCondition;

pub(crate) use self::stop_condition::Objectives;

pub mod stop_condition {
    use ::stop_condition::prelude::*;

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

pub mod hager_beta {

    pub struct Input<'a, 'b, 'c> {
        pub eta: f64,
        pub last_direction: &'a [f64],
        pub last_d_gradient: &'b [f64],
        pub from_gradient: &'c [f64],
    }

    pub fn compute(input: Input) -> f64 {
        use sp2_slice_math::{v, V, vnormalize, vnorm, vdot, BadNorm};

        let Input {
            eta, last_direction, last_d_gradient, from_gradient
        } = input;

        let V(last_from_gradient) = v(from_gradient) - v(last_d_gradient);

        // In a notation more evocative of the mathematical form.
        // (note: _km corresponds to k-1, _kmm corresponds to k-2)
        let d_km = last_direction;
        let V(dg_km_hat) = match vnormalize(last_d_gradient) {
            Ok(x) => x,
            // Zero or infinite norm.
            // I don't think this branch will ever be entered;
            // a successful linesearch guarantees that the gradient has changed,
            // and this method is not called on linesearch failure.
            Err(BadNorm(norm)) => {
                // Satisfying the wolfe conditions
                warn!("`d_gradient` bad norm: {}! Doing steepest descent.", norm);
                return 0.0;
            },
        };
        let dg_km_hat = &dg_km_hat[..];
        let g_km = from_gradient;
        let g_kmm = &last_from_gradient[..];

        let eta_k = -1.0 / eta.min(vnorm(g_kmm));

        let c_k = vdot(d_km, dg_km_hat);
        let V(beta_bra) = c_k * v(dg_km_hat) - 2.0 * v(d_km);
        let beta_k = vdot(&beta_bra, g_km) / (c_k * c_k);

        beta_k.max(eta_k)
    }
}

pub trait DiffFn<E>: FnMut(&[f64]) -> Result<(f64, Vec<f64>), E> { }
impl<E, F> DiffFn<E> for F
where F: FnMut(&[f64]) -> Result<(f64, Vec<f64>), E> { }

/// `linesearch` error type
error_chain! {
    types {
        Error, ErrorKind, ResultExt, CgResult;
    }
    links {
        LsError(::linesearch::Error, ::linesearch::ErrorKind);
    }
    errors { }
}

struct ComputeError<E>(E);
impl<E> From<Error> for Failure<E> {
    fn from(e: Error) -> Self {
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
    pub error: Either<Error, E>,
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

// These *could* be declared inside acgsd()
// but I think the function is quite long enough.
pub(crate) mod internal_types {

    #[derive(Debug, Clone)]
    pub(crate) struct Point {
        pub(crate) position: Vec<f64>,
        pub(crate) gradient: Vec<f64>,
        pub(crate) value: f64,
    }

    #[derive(Debug, Clone)]
    pub(crate) struct Saved {
        pub(crate) alpha: f64,
        pub(crate) position: Vec<f64>,
        pub(crate) gradient: Vec<f64>,
        pub(crate) value: f64,
    }

    impl Saved {
        pub(crate) fn into_point(self) -> Point {
            let Saved { position, gradient, value, .. } = self;
            Point { position, gradient, value }
        }
        pub(crate) fn to_point(&self) -> Point { self.clone().into_point() }
    }

    #[derive(Debug, Clone)]
    pub(crate) struct Last {
        pub(crate) direction: Vec<f64>, // direction searched (normalized)

        // NOTE: These next three are all zero when linesearch has failed.
        //       This can be a problem for d_value in particular.
        pub(crate) d_value: f64,          // change in value
        pub(crate) d_position: Vec<f64>,  // change in position
        pub(crate) d_gradient: Vec<f64>,  // change in gradient

        pub(crate) ls_failed: bool,       // linesearch failed?
    }
}

#[inline(never)]
pub fn acgsd<E, F: DiffFn<E>>(
    settings: &Settings,
    initial_position: &[f64],
    mut compute: F,
) -> Result<Output, Failure<E>>
where F: FnMut(&[f64]) -> Result<(f64, Vec<f64>), E>
{
    use self::internal_types::{Point, Saved, Last};

    let stop_condition = self::stop_condition::Rpn::from_cereal(&settings.stop_condition);

    let mut compute_point = |position: &[f64]| {
        let position = position.to_vec();
        let (value, gradient) = compute(&position).map_err(ComputeError)?;
        Ok(Point {position, value, gradient})
    };

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
            print!(" dv: {:>14.7e}", d_value);
            print!("  g: {:>13.7e}", grad_mag);

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
                direction: ref last_direction,
                d_gradient: ref last_d_gradient,
                d_position: ref last_d_position,
                ..
            }) = &last
            {
                let from_gradient = &saved.gradient[..];

                // FIXME messy garbage, these two methods are probably far more
                //       similar than the current code makes them appear
                match settings.beta {
                    settings::Beta::Hager { eta } => {
                        use self::hager_beta::Input;

                        let beta = hager_beta::compute(Input {
                            eta, last_direction, last_d_gradient, from_gradient,
                        });
                        let V(direction): V<Vec<f64>> = beta * v(last_direction) - v(from_gradient);
                        break 'use_dir direction;
                    },

                    settings::Beta::Acgsd { downhill_min } => {
                        let beta = calc_beta_acgsd(&saved.gradient, last_d_position, last_d_gradient);

                        let V(direction) = beta * v(last_d_position) - v(&saved.gradient);

                        // use this direction as long as it is downhill enough
                        if vdot(from_gradient, &direction) <= -downhill_min * vnorm(from_gradient) * vnorm(&direction) {
                            break 'use_dir direction;
                        }
                    },
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
        let next_alpha = {
            // (FIXME: we memoize the compute function because the linesearch
            //         is currently a bit messy and sometimes asks for a point
            //         more than once.  These are really issues with the linesearch,
            //         and ought not to be the caller's concern)
            let mut memoized: Box<FnMut(f64) -> Result<(f64, f64), ComputeError<E>>>
                = ::util::cache::hash_memoize_result_by_key(
                    |&alpha| ::ordered_float::NotNaN::new(alpha).unwrap(),
                    |alpha| {
                        let point = compute_in_dir(alpha, &direction)?;
                        let slope = vdot(&point.gradient, &direction);

                        // update cache, checking values to predict which
                        //  point linesearch will prefer to use.
                        // (future additions to linesearch may make this less reliable)
                        if point.value < ls_point.value {
                            ls_alpha = alpha;
                            ls_point = point.clone();
                        }

                        if let Some(Last { ls_failed: true, .. }) = last {
                            if settings.has_verbosity(1) {
                                print!("LS: a: {:.14e}", alpha);
                                print!("\tv: {:14e}", point.value);
                                print!("\ts: {:14e}", slope);
                                println!("");
                            }
                        }
                        Ok((point.value, slope))
                    },
                );

            // NOTE: Under our scheme where direction is normalized,
            //       the previous alpha itself is a suitable guess.
            let guess_alpha =
                saved.alpha
                .min(settings.alpha_guess_max.unwrap_or(::std::f64::INFINITY));

            match settings.linesearch {
                settings::Linesearch::Acgsd(ref settings) => {
                    match ::linesearch::linesearch(
                        settings,
                        saved.alpha,
                        &mut *memoized,
                    ) {
                        Ok(x) => x,
                        Err(Left(e)) => Err(Error::from(e))?,
                        Err(Right(e)) => Err(e)?,
                    }
                },
                settings::Linesearch::Hager(ref settings) => {
                    ::hager_ls::linesearch(
                        settings,
                        saved.alpha,
                        &mut *memoized,
                    )?
                },
            }

        }; // let next_alpha = { ... }
        let next_point = match next_alpha {
            a if a == ls_alpha => ls_point, // extraneous computation avoided!
            a => compute_in_dir(a, &direction)?,
        };

        // if the linesearch failed, note it and try
        //  again next iteration with steepest descent
        let ls_failed = next_alpha == 0.0;
        if ls_failed {
            if let Some(Last { ls_failed: true, .. }) = last {
                return fatal("linesearch failure (second)", saved.alpha, saved.to_point());
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
        use ::sp2_slice_math::{v, V};
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