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

//! rsp2 defers to scipy for solving sparse eigenvalue problems
//! (which in turn uses ARPACK)
//!
//! There are a multitude of reasons why this strategy is better than,
//! say, calling ARPACK directly. (and believe me, I tried):
//!
//! * ARPACK's API presents a "reverse communication" interface that is
//!   irrevocably thread-unsafe. This presents an interesting challenge
//!   to a rust crate author, as it is basically *impossible* for any non-application
//!   crate to expose a safe, public function that uses ARPACK.
//!   (unless a thread safety mechanism is embedded into the same crate that emits
//!    the linker flag, but these crates are not normally supposed to contain anything
//!    opinionated!)
//!   This is a complete non-issue for calling a python script, as the easiest way to
//!   do so already involves creating a new process. :wink:
//!
//! * Wrapping fortran APIs in rust is already difficult as it is.
//!   1-based indices and column major layout are huge mental hurdles for non-Fortran
//!   programmers. Meanwhile, scipy presents an interface that is already aligned with
//!   our conventions.
//!
//! * ARPACK may be well-documented, but using it is *difficult* for the very
//!   same reason that it is so powerful.  You need to write a "driver" that calls
//!   multiple methods of ARPACK and basically has an entire conversation with it.
//!   It took me 4 hours to translate their simplest driver (dssimp) into rust, and
//!   even that simple driver does things like:
//!   * supplying uninitialized input arguments (knowing that they will never be used)
//!   * doubly-mutable aliasing (the array v is given to `dseupd` twice, and both of
//!     those parameters are designated as output)
//!   Scipy's built-in driver might not be able to solve every problem, but I couldn't
//!   trust myself to write anything better.
//!
//! So there you have it.

use ::FailResult;
use ::math::basis::{Basis3, Ket3};

use ::slice_of_array::prelude::*;
use super::{call_script_and_communicate};

// Conversion factor phonopy uses to scale the eigenvalues to THz angular momentum.
//    = sqrt(eV/amu)/angstrom/(2*pi)/THz
const SQRT_EIGENVALUE_TO_THZ: f64 = 15.6333043006705;
//    = THz / (c / cm)
const THZ_TO_WAVENUMBER: f64 = 33.3564095198152;
const SQRT_EIGENVALUE_TO_WAVENUMBER: f64 = SQRT_EIGENVALUE_TO_THZ * THZ_TO_WAVENUMBER;

pub(super) const PY_CHECK_SCIPY_AVAILABILITY: &'static str = indoc!(r#"
    #!/usr/bin/env python3
    import numpy as np
    import scipy.sparse
    import scipy.sparse.linalg as spla
"#);

const PY_CALL_EIGSH: &'static str = include_str!("call-eigsh.py");

#[derive(Serialize)]
#[serde(rename_all = "kebab-case")]
struct Input {
    matrix: ::math::dynmat::Cereal,
    kw: PyKw,
    // permits non-convergence exceptions
    allow_fewer_solutions: bool,
}

#[allow(dead_code)]
#[derive(Serialize)]
enum Which {
    #[serde(rename = "LM")] LargestMagnitude,
    #[serde(rename = "SM")] SmallestMagnitude,
    #[serde(rename = "LA")] MostPositive,
    #[serde(rename = "SA")] MostNegative,
    #[serde(rename = "BE")] HalfAndHalf,
}

#[allow(dead_code)]
#[derive(Serialize)]
enum ShiftInvertMode {
    /// `w'[i] = 1 / (w[i] - sigma)`
    #[serde(rename = "normal")] Normal,

    /// `w'[i] = w[i] / (w[i] - sigma)`
    ///
    /// Requires the matrix to be positive definite.
    #[serde(rename = "buckling")] Buckling,

    /// `w'[i] = (w[i] + sigma) / (w[i] - sigma)`
    #[serde(rename = "cayley")] Cayley,
}

#[derive(Serialize, Default)]
struct PyKw {
    #[serde(rename = "k", skip_serializing_if = "Option::is_none")]
    pub how_many: Option<usize>,
    #[serde(rename = "sigma", skip_serializing_if = "Option::is_none")]
    pub shift_invert_target: Option<f64>,
    #[serde(rename = "mode", skip_serializing_if = "Option::is_none")]
    pub shift_invert_mode: Option<ShiftInvertMode>,
    #[serde(rename = "v0", skip_serializing_if = "Option::is_none")]
    pub initial_vec: Option<Vec<f64>>,
    #[serde(rename = "ncv", skip_serializing_if = "Option::is_none")]
    pub num_lanczos_vectors: Option<usize>,
    #[serde(rename = "which", skip_serializing_if = "Option::is_none")]
    pub which: Option<Which>,
    #[serde(rename = "maxiter", skip_serializing_if = "Option::is_none")]
    pub max_iter: Option<usize>,
    #[serde(rename = "tol", skip_serializing_if = "Option::is_none")]
    pub tol: Option<f64>,
}

//-------------------------------------------------------------------------------
// calling scripts

// Returns:
// - frequencies (not eigenvalues)
// - eigenvectors
fn call_eigsh(input: &Input) -> FailResult<(Vec<f64>, Basis3)> {

    let (vals, (real, imag)) = call_script_and_communicate(PY_CALL_EIGSH, input)?;
    // annotate types to select Deserialize impl
    let _: &Vec<f64> = &vals;
    let _: &Vec<Vec<f64>> = &real;
    let _: &Vec<Vec<f64>> = &imag;

    let mut kets = vec![];
    for (real, imag) in zip_eq!(real, imag) {
        let real = real.nest().to_vec();
        let imag = imag.nest().to_vec();
        kets.push(Ket3 { real, imag });
    }
    let freqs = vals.into_iter().map(eigenvalue_to_frequency).collect();

    Ok((freqs, Basis3(kets)))
}

#[derive(Debug, Fail)]
#[fail(display = "an error occurred importing numpy and scipy")]
pub struct ScipyAvailabilityError;

//-------------------------------------------------------------------------------

use ::math::dynmat::DynamicalMatrix;

impl DynamicalMatrix {
    /// Requesting more than this number of eigensolutions will fail.
    ///
    /// (inherent limitation of the method used by ARPACK)
    pub fn max_sparse_eigensolutions(&self) -> usize {
        3 * self.0.dim.0 - 2
    }

    /// Clip `how_many` for the max possible value for sparse solver methods.
    pub fn clip_how_many(&self, how_many: usize) -> usize {
        usize::min(how_many, self.max_sparse_eigensolutions())
    }

    pub fn compute_negative_eigensolutions(&self) -> FailResult<(Vec<f64>, Basis3)> {
        trace!("Computing most negative eigensolutions.");

        let how_many = 5;

        let most_negative = call_eigsh(&Input {
            matrix: self.cereal(),
            allow_fewer_solutions: true,
            kw: PyKw {
                which: Some(Which::MostNegative),
                how_many: Some(self.clip_how_many(how_many)),
                // A fixed, hard limit.  The idea here is that large negative eigenvalues
                // will hopefully be found quickly, and small negative eigenvalues may take
                // unreasonably long to converge without shift-invert mode.
                max_iter: Some(30),
                ..Default::default()
            },
        })?;

        if most_negative.0.len() > 0 {
            return Ok(most_negative);
        }

        // None converged.  Perhaps there are no negative modes, and the
        // acoustic ones may be causing trouble for convergence.

        // Shift-invert should be able to find the acoustics quickly, and possibly
        // small negative modes in addition to them.
        trace!("Nothing converged! Computing low-frequency negative eigensolutions using shift-invert mode.");
        let small_negative = call_eigsh(&Input {
            matrix: self.cereal(),
            allow_fewer_solutions: true,
            kw: PyKw {
                // NOTE: this will favor with greatest precedence the acoustic eigenvalues,
                //       followed by those which are increasingly negative.
                shift_invert_target: Some(0.0), // something larger than reasonable for acoustics
                shift_invert_mode: Some(ShiftInvertMode::Normal),
                which: Some(Which::MostNegative),

                // (look for more modes, to give some headroom beyond the acoustics)
                how_many: Some(self.clip_how_many(how_many + 5)),

                // (from what I can tell after a couple of runs on a 2000 atom system, a majority
                //  of eigensolutions are found in the first few iterations, and almost nothing
                //  extra is found by continuing to search afterward.  That said, we might as well
                //  do a few dozen iters, because my exploration was not very intensive, and the
                //  initial LU decomposition for shift-inversion is by far the most expensive step)
                max_iter: Some(30),

                ..Default::default()
            },
        })?;

        match small_negative.0.len() {
            0 => {
                // This is surprising; we'd expect to at least see acoustics!
                warn!("Nothing was found in the shift-invert search!")
            },
            _ => trace!("Done using shift-invert mode."),
        }
        Ok(small_negative)
    }

    pub fn compute_most_extreme_eigensolutions(&self, how_many: usize) -> FailResult<(Vec<f64>, Basis3)> {
        call_eigsh(&Input {
            matrix: self.cereal(),
            allow_fewer_solutions: false,
            kw: PyKw {
                how_many: Some(how_many),
                ..Default::default()
            },
        })
    }
}

fn eigenvalue_to_frequency(val: f64) -> f64 {
    f64::sqrt(f64::abs(val)) * f64::signum(val) * SQRT_EIGENVALUE_TO_WAVENUMBER
}
