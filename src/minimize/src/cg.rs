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

// HERE BE DRAGONS

use ::std::fmt::Write;
use ::std::collections::VecDeque;
use ::std::fmt;

use ::either::{Either, Left, Right};
use ::rsp2_slice_math::{vnorm, vdot, V, v, vnormalize};

pub mod settings {
    //! Please do not manually construct anything in here.
    //! None of this is future-safe.
    //!
    //! Deserialize from a JSON literal instead.

    pub use super::StopCondition;

    #[derive(Debug, Clone, PartialEq)]
    pub enum Beta {
        Acgsd {
            /// The direction searched will always point at least this much downhill.
            downhill_min: f64,
        },

        Hager {
            // NOTE: This seems to play a role similar to Acgsd.downhill_min
            //        but it isn't *quite* the same...
            /// This... I'm not sure what this does.
            eta: f64,
        },
    }

    impl Beta {
        pub fn new_acgsd() -> Self {
            Beta::Acgsd {
                downhill_min: 1e-3,
            }
        }

        pub fn new_hager() -> Self {
            Beta::Hager {
                eta: 1e-2,
            }
        }
    }

    #[derive(Debug, Clone, PartialEq)]
    pub enum Linesearch {
        Acgsd(::strong_ls::Settings),
        Hager(::hager_ls::Settings),
    }

    /// Behavior when a linesearch along the steepest descent direction fails.
    /// (this is phenomenally rare for the Hager linesearch method, and when it
    ///  does occur it may very well be due to exceptionally good convergence,
    ///  rather than any sort of actual failure)
    #[derive(Debug, Clone, PartialEq)]
    pub enum OnLsFailure {
        /// Treat a second linesearch failure as a successful stop condition.
        Succeed,
        /// Succeed, but log a warning.
        Warn,
        /// Return `Err(_)`.
        Fail,
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
            match self {
                Linesearch::Acgsd(settings) => settings.validate(),
                Linesearch::Hager(settings) => settings.validate(),
            }
        }
    }
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

pub use self::stop_condition::StopCondition;
pub mod stop_condition {
    use super::*;
    use ::stop_condition::prelude::*;

    #[derive(Debug, Clone, PartialEq)]
    pub(crate) struct Objectives<'a> {
        /// All computed values so far.
        pub values: &'a [f64],
        pub grad_max: f64,
        pub grad_norm: f64,
        pub grad_rms: f64,
        pub iterations: u64,
    }

    #[derive(Serialize, Deserialize)]
    #[derive(Debug, Copy, Clone, PartialEq)]
    pub enum Simple {
        /// This compares signed values, not magnitudes.
        ///
        /// What this means is that:
        /// - positive threshold says "stop if value increases more than this"
        /// - negative threshold says "continue as long as value has decreased
        ///                             by at least this much"
        ///
        /// (negative is useful when alternating between CG and other methods
        ///  of minimization)
        #[serde(rename = "value-delta")] ValueDelta {
            #[serde(rename = "rel-greater-than")] delta: f64,
            #[serde(rename =        "steps-ago")] steps_ago: u32,
        },
        /// Max absolute value of grad.
        #[serde(rename =    "grad-max")] GradientMax(f64),
        /// Norm of grad for the current structure.
        /// (Beware, this scales with sqrt(N)...)
        #[serde(rename =   "grad-norm")] GradientNorm(f64),
        /// Norm of grad, rescaled as an intensive property.
        #[serde(rename =    "grad-rms")] GradientRms(f64),
        /// The number of full iterations that have occurred.
        #[serde(rename =  "iterations")] Iterations(u64),
    }

    // Relative difference.
    //
    // This won't return NaN for finite inputs, although it
    // WILL be infinite if exactly one of the two operands is zero
    fn rel_sub(a: f64, b: f64) -> f64 {
        if a == b { return 0.0; }
        (a - b) / a.abs().min(b.abs())
    }

    impl<'a> ShouldStop<Objectives<'a>> for Simple {
        fn should_stop(&self, objs: &Objectives<'a>) -> bool {
            match *self {
                Simple::ValueDelta { delta: min_change, steps_ago } => {
                    if let Some(i) = objs.values.len().checked_sub(steps_ago as usize + 1) {
                        if rel_sub(*objs.values.last().unwrap(), objs.values[i]) >= min_change {
                            return true;
                        }
                    }
                    false
                },
                Simple::GradientMax(tol) => objs.grad_max <= tol,
                Simple::GradientNorm(tol) => objs.grad_norm <= tol,
                Simple::GradientRms(tol) => objs.grad_rms <= tol,
                Simple::Iterations(n) => objs.iterations >= n,
            }
        }
    }

    /// Configuration for the built-in implementation of stop conditions.
    ///
    /// The recommended method for constructing one of these is to deserialize it from JSON.  The
    /// set of basic conditions is documented in `#[serde]` annotations on [`Simple`], and there
    /// are additionally `"all"` and `"any"` meta-conditions that are provided by
    /// [`::stop_condition::Cereal`].
    ///
    /// If this doesn't fit your needs, you can use CG's Builder API to define your own arbitrary
    /// stop conditions as functions of [`AlgorithmState`].
    pub type StopCondition = ::stop_condition::Cereal<Simple>;

    impl StopCondition {
        /// Convert to the more general form accepted by the Builder API.
        pub fn to_function(&self) -> impl Clone + FnMut(AlgorithmState<'_>) -> bool {
            let mut value_history = vec![];
            let rpn = crate::stop_condition::Rpn::from_cereal(self);

            move |state: AlgorithmState<'_>| {
                value_history.push(state.value);

                let gnorm = vnorm(&state.gradient);
                rpn.should_stop(&Objectives {
                    grad_norm: gnorm,
                    grad_rms: gnorm / (state.gradient.len() as f64).sqrt(),
                    grad_max: max_norm(&state.gradient),
                    values: &value_history[..],
                    iterations: state.iterations,
                })
            }
        }
    }

    mod tests {
        #[test]
        fn test_serialized_repr() {
            use super::Simple::Iterations;
            use ::stop_condition::Cereal::{Simple,Logical};
            use ::stop_condition::LogicalExpression::All;
            use ::serde_json::to_value;
            assert_eq!(
                to_value(Simple(Iterations(5))).unwrap(),
                json!({"iterations": 5}),
            );
            assert_eq!(
                to_value(Logical(All(vec![Simple(Iterations(5))]))).unwrap(),
                json!({"all": [{"iterations": 5}]}),
            );
        }
    }
}

pub mod hager_beta {
    pub struct Input<'a> {
        pub eta: f64,
        pub last_direction: &'a [f64],
        pub last_d_gradient: &'a [f64],
        pub from_gradient: &'a [f64],
    }

    pub fn compute(input: Input) -> f64 {
        use rsp2_slice_math::{v, V, vnormalize, vnorm, vdot, BadNorm};

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

/// calculate beta (ACGSD)
fn calc_beta_acgsd(gradient: &[f64], delta_x: &[f64], delta_g: &[f64]) -> f64 {
    // inner products needed for beta
    let dg_dx = vdot(delta_g, delta_x); // dg.dx
    let g_dg = vdot(gradient, delta_g); // g.dg
    let g_dx = vdot(gradient, delta_x); // g.dx

    (g_dg / dg_dx) * (1.0 - g_dx / dg_dx)
}

//==================================================================================================
// Errors

use ::strong_ls::LinesearchError;
#[derive(Debug, Fail)]
pub enum AcgsdError {
    #[fail(display = "Linesearch failed: {}", _0)]
    Linesearch(#[fail(cause)] LinesearchError),

    #[doc(hidden)]
    #[fail(display = "impossible!")]
    _Hidden,
}

impl From<LinesearchError> for AcgsdError {
    fn from(error: LinesearchError) -> Self {
        AcgsdError::Linesearch(error)
    }
}

/// A wrapper type used to allow `?` to work on errors from the user function.
// (FIXME: honestly though we're probably better off without it so that
//         we can fill in the `best_position` field)
struct ComputeError<E>(E);
impl<E> From<AcgsdError> for Failure<E> {
    fn from(e: AcgsdError) -> Self {
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
    pub error: Either<AcgsdError, E>,
}

impl<E: fmt::Display> fmt::Display for Failure<E> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result
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
    beta: Required<settings::Beta>,
    linesearch: Required<settings::Linesearch>,
    on_ls_failure: settings::OnLsFailure,
    alpha_guess_first: f64,
    alpha_guess_max: f64,
    build_output_fns: Vec<Box<dyn BuildAlgorithmStateFn<Output=()>>>,
}

impl Builder {
    fn new() -> Self {
        Builder {
            build_stop_condition: None,
            beta: None,
            linesearch: None,
            on_ls_failure: settings::OnLsFailure::Fail,
            alpha_guess_first: 1.0,
            alpha_guess_max: std::f64::INFINITY,
            build_output_fns: vec![],
        }
    }

    /// Perform conjugate gradient using the default configuration for CG-DESCENT, and with a stop
    /// condition that can be deserialized from JSON.
    ///
    /// CG-DESCENT is a flavor of CG brought forth by William Hager that is robust to cases where
    /// rounding errors in the value can exceed the expected change over a linestep, which may be true
    /// for extremely large systems.  This method puts more faith in the gradient than the value, and
    /// so the value will sometimes increase over a step.
    pub fn new_acgsd() -> Self {
        let mut me = Self::new();
        me.beta = Some(settings::Beta::new_acgsd());
        me.linesearch = Some(settings::Linesearch::Acgsd(Default::default()));
        me
    }

    /// Perform conjugate gradient using the default configuration for ACGSD, and with a stop condition
    /// that can be deserialized from JSON.
    ///
    /// This was the first flavor of CG implemented in RSP2.  It is based off of earlier code written by
    /// Colin Daniels, which is presumably an implementation of Neculai Andrei's ACGSD.  Conceptually,
    /// it is far, far simpler than `cg_descent`.
    pub fn new_hager() -> Self {
        let mut me = Self::new();
        me.beta = Some(settings::Beta::new_hager());
        me.linesearch = Some(settings::Linesearch::Hager(Default::default()));
        me
    }

    pub fn beta(&mut self, value: settings::Beta) -> &mut Self {
        self.beta = Some(value); self
    }

    pub fn linesearch(&mut self, value: settings::Linesearch) -> &mut Self {
        self.linesearch = Some(value); self
    }

    pub fn on_ls_failure(&mut self, value: settings::OnLsFailure) -> &mut Self {
        self.on_ls_failure = value; self
    }

    pub fn alpha_guess_first(&mut self, value: f64) -> &mut Self {
        self.alpha_guess_first = value; self
    }

    pub fn alpha_guess_max(&mut self, value: f64) -> &mut Self {
        self.alpha_guess_max = value; self
    }

    /// Set up an arbitrary function for logging output each iteration.
    ///
    /// This will exist alongside any previously existing output functions.
    pub fn output_fn(&mut self, f: impl BuildAlgorithmStateFn<Output=()> + 'static) -> &mut Self {
        self.build_output_fns.push(Box::new(f));self
    }

    /// Set up a "standard" output function that writes some formatted lines on each iteration.
    /// The output format may change.
    ///
    /// This will exist alongside any previously existing output functions.
    pub fn basic_output_fn(&mut self, emit: impl Clone + FnMut(fmt::Arguments<'_>) + 'static) -> &mut Self {
        self.output_fn(get_basic_output_fn(emit))
    }

    pub fn stop_condition(&mut self, f: impl BuildAlgorithmStateFn<Output=bool> + 'static) -> &mut Self {
        self.build_stop_condition = Some(Box::new(f)); self
    }
}

impl Clone for Builder {
    fn clone(&self) -> Self {
        Builder {
            build_output_fns: self.build_output_fns.iter().map(|x| objekt::clone_box(&**x)).collect(),
            beta: self.beta.clone(),
            linesearch: self.linesearch.clone(),
            on_ls_failure: self.on_ls_failure.clone(),
            alpha_guess_first: self.alpha_guess_first.clone(),
            alpha_guess_max: self.alpha_guess_max.clone(),
            build_stop_condition: self.build_stop_condition.as_ref().map(|x| objekt::clone_box(&**x)),
        }
    }
}

//==================================================================================================
// closures in builder API

#[derive(Debug, Clone)]
pub struct AlgorithmState<'a> {
    /// Complete iterations *so far.*  The first call will be with `iterations: 0`,
    /// and the potential will have been computed once (but no linesearch will have
    /// been performed).
    pub iterations: u64,
    pub position: &'a [f64],
    pub gradient: &'a [f64],
    pub value: f64,
    /// The initial guess for line step size in the upcoming line search.
    ///
    /// (for iterations after the first iteration, this is equal to the step size taken in the
    ///  previous iteration)
    pub alpha: f64,
    /// Direction traveled last iteration. `None` on the first iteration.
    pub direction: Option<&'a [f64]>,
    // ensures addition of new fields is backwards compatible
    #[allow(non_snake_case)]
    __no_full_destructure: (),
}

/// Trait for producing fresh instances of an `AlgorithmStateFn`.
///
/// Don't worry too much about this; this trait exists simply so that a single Builder can be cloned
/// or used for multiple `acgsd` calls.
///
/// As a convenience, this is implemented for all cloneable `AlgorithmStateFn`s, so in general any
/// closure that takes `AlgorithmState` will do. (as long as calls to a clone of the closure do not
/// affect the "freshness" of the original closure; i.e. don't track prior inputs in an
/// `Rc<RefCell<_>>`, or at least don't use them in a manner which affects the output!)
pub trait BuildAlgorithmStateFn: objekt::Clone {
    type Output;

    /// Produce a fresh instance of the AlgorithmStateFn, with no history of calls made to it yet.
    fn build(&self) -> Box<dyn FnMut(AlgorithmState<'_>) -> Self::Output>;
}

impl<F, B> BuildAlgorithmStateFn for F
where
    F: FnMut(AlgorithmState<'_>) -> B,
    F: Clone + 'static,
{
    type Output = B;

    fn build(&self) -> Box<dyn FnMut(AlgorithmState<'_>) -> B> { Box::new(self.clone()) }
}

pub fn get_basic_output_fn(
    mut emit: impl Clone + FnMut(fmt::Arguments<'_>),
) -> impl Clone + FnMut(AlgorithmState<'_>) {
    let mut last_value = None;
    let mut past_directions = VecDeque::<Vec<_>>::new();

    move |state: AlgorithmState<'_>| {
        let d_value = last_value.as_ref().map(|prev| state.value - prev).unwrap_or(0.0);
        let grad_mag = vnorm(&state.gradient);
        emit(format_args!(" i: {i:>6}  v: {v:18.14} dv: {dv:+8.2e}  g: {g:>12.7e}  {cos:<24}",
               i = state.iterations,
               v = state.value,
               dv = d_value,
               g = grad_mag,
               cos = {
                   let mut s = String::new();
                   if !past_directions.is_empty() {
                       write!(&mut s, "cos:").unwrap();
                       let latest = state.direction.expect("(BUG)");
                       for other in &past_directions {
                           write!(&mut s, " {:>+5.2}", vdot(latest, other)).unwrap();
                       }
                   }
                   s
               },
        ));

        // Record data for next call.
        last_value = Some(state.value);

        if let Some(direction) = state.direction {
            let max_cosines = 3;
            let mut buf = match past_directions.len().cmp(&max_cosines) {
                std::cmp::Ordering::Less => vec![],
                std::cmp::Ordering::Equal => past_directions.pop_back().unwrap(),
                std::cmp::Ordering::Greater => unreachable!(),
            };
            buf.clear();
            buf.extend(direction.iter().cloned());
            past_directions.push_front(buf);
        }
    }
}

//==================================================================================================
// Primary public API

#[derive(Serialize, Deserialize)]
#[derive(Debug, Clone)]
pub struct Output {
    pub iterations: u64,
    pub position: Vec<f64>,
    pub gradient: Vec<f64>,
    pub value: f64,
    // ensures addition of new fields is backwards compatible
    #[serde(skip)]
    #[allow(non_snake_case)]
    __no_full_destructure: (),
}

pub trait DiffFn<E>: FnMut(&[f64]) -> Result<(f64, Vec<f64>), E> { }
impl<E, F> DiffFn<E> for F
where F: FnMut(&[f64]) -> Result<(f64, Vec<f64>), E> { }

impl Builder {
    pub fn run<E, F: DiffFn<E>>(
        &self,
        initial_position: &[f64],
        compute: F,
    ) -> Result<Output, Failure<E>>
    where F: FnMut(&[f64]) -> Result<(f64, Vec<f64>), E>
    { cg(self, initial_position, compute) }
}

/// Perform conjugate gradient using the default configuration for CG-DESCENT, and with a
/// stop condition that can be deserialized from JSON.
///
/// See [`Builder::new_hager()`] for more information.
pub fn cg_descent<E, F: DiffFn<E>>(
    stop_condition: &StopCondition,
    initial_position: &[f64],
    compute: F,
) -> Result<Output, Failure<E>>
where F: FnMut(&[f64]) -> Result<(f64, Vec<f64>), E>
{
    Builder::new_hager()
        .stop_condition(stop_condition.to_function())
        .run(initial_position, compute)
}

/// Perform conjugate gradient using the default configuration for ACGSD, and with a stop condition
/// that can be deserialized from JSON.
///
/// See [`Builder::new_acgsd()`] for more information.
pub fn acgsd<E, F: DiffFn<E>>(
    stop_condition: &StopCondition,
    initial_position: &[f64],
    compute: F,
) -> Result<Output, Failure<E>>
    where F: FnMut(&[f64]) -> Result<(f64, Vec<f64>), E>
{
    Builder::new_acgsd()
        .stop_condition(stop_condition.to_function())
        .run(initial_position, compute)
}

//==================================================================================================

// Types used inside the implementation of acgsd.
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
fn cg<E, F: DiffFn<E>>(
    builder: &Builder,
    initial_position: &[f64],
    mut compute: F,
) -> Result<Output, Failure<E>>
where F: FnMut(&[f64]) -> Result<(f64, Vec<f64>), E>
{
    use self::internal_types::{Point, Saved, Last};

    let mut stop_condition = match builder.build_stop_condition {
        Some(ref f) => f.build(),
        None => panic!("'stop_condition' was not supplied to the CG Builder!"),
    };
    let beta_settings = builder.beta.clone().expect("'beta' was not supplied to the CG Builder!");
    let ls_settings = builder.linesearch.clone().expect("'linesearch' was not supplied to the CG Builder!");

    let mut output_functions: Vec<_> = {
        builder.build_output_fns.iter().map(|x| x.build()).collect()
    };

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
        Saved { alpha: builder.alpha_guess_first, position, value, gradient }
    };

    // Remembers all values.
    let mut value_history = vec![last_saved.value];
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
            warn!("{}", msg);
            warn!("Iterations: {}", iterations);
            warn!("     Alpha: {}", alpha);
            warn!("     Value: {}", point.value);
            warn!(" Grad Norm: {}", vnorm(&point.gradient));
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

        // use as 'return success(...);'
        // Constructs a successful return value.
        let success = |Point { position, value, gradient }| {
            Ok(Output { iterations, position, value, gradient, __no_full_destructure: () })
        };

// /////////////////////////////////////////////////////////////////////////////
// Per-iteration output & evaluate exit conditions                            //
// /////////////////////////////////////////////////////////////////////////////

        {
            let state = AlgorithmState {
                iterations,
                value: saved.value,
                gradient: &saved.gradient,
                position: &saved.position,
                direction: last.as_ref().map(|last| &last.direction[..]),
                alpha: saved.alpha,
                __no_full_destructure: (),
            };

            for f in &mut output_functions {
                f(state.clone());
            }

            if stop_condition(state) {
                // FIXME this no longer belongs here, but...
                info!("ACGSD Finished.");
                info!("Iterations: {}", iterations);
                info!("     Value: {}", saved.value);
                info!(" Grad Norm: {:e}", vnorm(&saved.gradient));
                info!("  Grad Max: {:e}", saved.gradient.iter().cloned().map(f64::abs).fold(0.0, f64::max));

                return success(saved.to_point());
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
            if let Some(Last{
                ls_failed: false,
                direction: last_direction,
                d_gradient: last_d_gradient,
                d_position: last_d_position,
                ..
            }) = &last
            {
                let from_gradient = &saved.gradient[..];

                // FIXME messy garbage, these two methods are probably far more
                //       similar than the current code makes them appear
                match beta_settings {
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
            debug!("Using steepest descent. (i: {})", iterations + 1);

            let V(direction) = -v(&saved.gradient);
            direction
        }}; // 'use_dir: loop { break { ... } }

        // NOTE: The original source scaled alpha instead of normalizing
        //       direction, which seems to be a fruitless optimization
        //       that only serves to amplify the mental workload.
        let V(direction) = vnormalize(&direction).unwrap_or_else(|_| {
            // FIXME: The fixed vector used here could interact poorly with drift
            //        cancellation.

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
                    |&alpha| ::ordered_float::NotNan::new(alpha).unwrap(),
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

                        Ok((point.value, slope))
                    },
                );

            // NOTE: Under our scheme where direction is normalized,
            //       the previous alpha itself is a suitable guess.
            let guess_alpha = saved.alpha.min(builder.alpha_guess_max);

            match &ls_settings {
                settings::Linesearch::Acgsd(settings) => {
                    match ::strong_ls::linesearch(settings, guess_alpha, &mut *memoized) {
                        Ok(x) => x,
                        Err(Left(e)) => Err(AcgsdError::from(e))?,
                        Err(Right(e)) => Err(e)?,
                    }
                },
                settings::Linesearch::Hager(settings) => {
                    ::hager_ls::linesearch(settings, guess_alpha, &mut *memoized)?
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
                match builder.on_ls_failure {
                    settings::OnLsFailure::Succeed => {
                        return success(saved.to_point());
                    },
                    settings::OnLsFailure::Warn => {
                        warning("linesearch failure (second)", saved.alpha, saved.to_point());
                        return success(saved.to_point());
                    },
                    settings::OnLsFailure::Fail => {
                        return fatal("linesearch failure (second)", saved.alpha, saved.to_point());
                    },
                }
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

        value_history.push(next_point.value);
    }
    panic!("too many iterations")
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

    type NoFailResult = Result<(f64, Vec<f64>), Never>;

    // For some point `P`, get a potential function of the form `V(X) = (X - P).(X - P)`.
    // This is a macro pending stabilization of `impl Trait`
    fn quadratic_test_fn(
        target: &[f64],
    ) -> impl FnMut(&[f64]) -> Result<(f64, Vec<f64>), Never> {
        use ::test::one_dee::prelude::*;
        use ::test::one_dee::Polynomial;

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
        use ::util::random::uniform_n;

        let target = uniform_n(15, -10.0, 10.0);
        let initial_point = uniform_n(15, -10.0, 10.0);
        let stop_condition = from_json!({"grad-max": 1e-11});
        let result = super::cg_descent(&stop_condition, &initial_point, quadratic_test_fn(&target)).unwrap();
        assert_close!(result.position, target);
    }

    // Test that tolerance tests can succeed as early as they ought to be capable of,
    //  by using absurdly large tolerances.
    #[test]
    fn insta_finish() {
        use ::util::random::uniform_n;

        // a potential where we are not initially at the minimum
        let point = uniform_n(18, -10.0, 10.0);
        let target = uniform_n(18, -10.0, 10.0);

        let s = from_json!({"grad-max": 1e20});
        assert_eq!(super::cg_descent(&s, &point, quadratic_test_fn(&target)).unwrap().iterations, 0);

        let s = from_json!({"grad-norm": 1e20});
        assert_eq!(super::cg_descent(&s, &point, quadratic_test_fn(&target)).unwrap().iterations, 0);

        // note: value-delta can only be tested after at least one iteration
        let s = from_json!({
            "value-delta": {
                "rel-greater-than": -1e100,
                "steps-ago": 2,
            }
        });
        assert_eq!(super::cg_descent(&s, &point, quadratic_test_fn(&target)).unwrap().iterations, 2);
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

        let stop_condition = from_json!({"grad-max": 1e-50});
        let potential = scale_y(1e-40, quadratic_test_fn(&target));
        let result = super::cg_descent(&stop_condition, &start, potential).unwrap();
        for (&from, x, &targ) in izip!(&start, result.position, &target) {
            assert_ne!(from, x);
            assert_close!(abs=0.0, rel=1e-9, x, targ);
        }

        let stop_condition = from_json!({"grad-max": 1e+30});
        let potential = scale_y(1e+40, quadratic_test_fn(&target));
        let result = super::cg_descent(&stop_condition, &start, potential).unwrap();
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
        let stop_condition = from_json!({"iterations": 1});
        assert_eq!(super::cg_descent(&stop_condition, &point, potential).unwrap().position, point);
    }

    #[test]
    fn test_iterations_stop_condition() {
        use ::util::random::uniform_n;
        let target = uniform_n(15, -10.0, 10.0);
        let start = uniform_n(15, -10.0, 10.0);

        // interations: 0 should cause no change in position despite not being at the minimum
        let stop_condition = from_json!({"iterations": 0});
        let result = super::cg_descent(&stop_condition, &start, quadratic_test_fn(&target)).unwrap();
        assert_eq!(result.iterations, 0);
        assert_eq!(result.position, start);

        // interations: 1 should cause a change in position
        let stop_condition = from_json!({"iterations": 1});
        let result = super::cg_descent(&stop_condition, &start, quadratic_test_fn(&target)).unwrap();
        assert_eq!(result.iterations, 1);
        assert_ne!(result.position, start);
    }

    #[test]
    fn trid() {
        use ::test::n_dee::{Trid, OnceDifferentiable};
        use ::util::random::uniform_n;
        let d = 10;
        for _ in 0..10 {
            let max_coord = (d*d) as f64;
            let output = super::cg_descent(
                &from_json!({"grad-rms": 1e-8}),
                &uniform_n(d, -max_coord, max_coord),
                |p| Ok::<_,Never>(Trid(d).diff(p))
            ).unwrap();

            assert_close!(rel=1e-5, output.value, Trid(d).min_value());
            assert_close!(rel=1e-5, output.position, Trid(d).min_position());
        }
    }

    #[test]
    fn lj() {
        use ::rsp2_slice_math::{v, V};
        use ::test::n_dee::{HyperLennardJones, OnceDifferentiable, Sum};
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
            let stop_condition = from_json!({"grad-rms": 1e-10});
            let output = super::cg_descent(&stop_condition, &start, |p| Ok::<_,Never>(diff.diff(p))).unwrap();

            assert_close!(rel=1e-12, output.value, -20.0);
        }
    }
}
