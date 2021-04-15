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

use crate::FailResult;
use crate::math::basis::{Basis3, GammaBasis3};
use std::sync::Arc;

#[allow(unused)] // rustc bug
use slice_of_array::prelude::*;
use super::{call_script_and_communicate, Script};

pub(super) const PY_CHECK_SCIPY_AVAILABILITY: Script = Script::String(indoc!(r#"
    #!/usr/bin/env python3
    import numpy as np
    import scipy.sparse
    import scipy.sparse.linalg as spla
"#));

// NOTE: These modules require the python package in `rsp2-python` to be added to `PYTHONPATH`.
//
//       This is handled early on in rsp2 entry points.
const PY_CALL:     Script = Script::Module("rsp2.internals.scipy_eigsh.call");
const PY_NEGATIVE: Script = Script::Module("rsp2.internals.scipy_eigsh.negative");

type Frequency = f64;

mod scripts {
    use super::*;
    use crate::filetypes::eigensols;

    #[derive(Serialize)]
    #[serde(rename_all = "kebab-case")]
    pub(super) struct Eigsh {
        pub(super) matrix: rsp2_dynmat::Cereal,
        pub(super) kw: PyKw,
        // permits non-convergence exceptions
        pub(super) allow_fewer_solutions: bool,
    }

    #[allow(unused)]
    impl Eigsh {
        pub(super) fn invoke(self) -> FailResult<(Vec<Frequency>, Basis3)> {
            call_script_and_communicate(PY_CALL, self)
                .and_then(read_py_output)
        }

        pub(super) fn invoke_gamma(self) -> FailResult<(Vec<Frequency>, GammaBasis3)> {
            call_script_and_communicate(PY_CALL, self)
                .and_then(read_py_output_gamma)
        }
    }

    #[derive(Serialize)]
    #[serde(rename_all = "kebab-case")]
    pub(super) struct Negative {
        pub(super) matrix: rsp2_dynmat::Cereal,
        pub(super) max_solutions: usize,
        pub(super) shift_invert_attempts: u32,
        pub(super) dense: bool,
    }

    #[allow(unused)]
    impl Negative {
        pub(super) fn invoke(self) -> FailResult<(Vec<Frequency>, Basis3)> {
            call_script_and_communicate(PY_NEGATIVE, self)
                .and_then(read_py_output)
        }

        pub(super) fn invoke_gamma(self) -> FailResult<(Vec<Frequency>, GammaBasis3)> {
            call_script_and_communicate(PY_NEGATIVE, self)
                .and_then(read_py_output_gamma)
        }
    }

    fn read_py_output(raw: eigensols::Raw) -> FailResult<(Vec<Frequency>, Basis3)> {
        let esols = raw.into_eigensols()?;
        Ok((esols.frequencies, esols.eigenvectors))
    }

    fn read_py_output_gamma(raw: eigensols::Raw) -> FailResult<(Vec<Frequency>, GammaBasis3)> {
        let (freqs, evecs) = read_py_output(raw)?;
        let evecs = evecs.into_gamma_basis3().ok_or_else(|| failure::err_msg("Eigensols not real!"))?;
        Ok((freqs, evecs))
    }
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

#[derive(Debug, Fail)]
#[fail(display = "an error occurred importing numpy and scipy")]
pub struct ScipyAvailabilityError;

//-------------------------------------------------------------------------------

use rsp2_dynmat::DynamicalMatrix;

/// Intended to be used during relaxation.
///
/// *Attempts* to produce a set of eigenkets containing many or all of the non-acoustic modes of
/// negative eigenvalue (possibly along with other modes that do not meet this condition);
/// however, it may very well miss some.
///
/// If none of the modes produced are negative, then it is safe (-ish) to assume that the matrix
/// has no such eigenmodes.  (At least, that is the intent!)
pub fn compute_negative_eigensolutions_gamma(
    dynmat: &DynamicalMatrix,
    max_solutions: usize,
    shift_invert_attempts: u32,
) -> FailResult<(Vec<f64>, GammaBasis3)> {
    trace!("Computing most negative eigensolutions.");
    scripts::Negative {
        matrix: dynmat.cereal(),
        max_solutions,
        shift_invert_attempts,
        dense: false,
    }.invoke_gamma()
}

/// Produce all eigensolutions of a dynamical matrix at gamma.
pub fn compute_eigensolutions_dense_gamma(dynmat: &DynamicalMatrix) -> (Vec<f64>, GammaBasis3) {
    use crate::math::basis::GammaKet3;

    let (eigenvalues, eigenvectors) = dynmat.compute_eigensolutions_dense_gamma();

    let frequencies = eigenvalues.eigenvalues.into_iter().map(crate::filetypes::eigensols::eigenvalue_to_frequency).collect();
    let eigenvectors = GammaBasis3(Arc::new(eigenvectors.into_iter().map(GammaKet3).collect()));
    (frequencies, eigenvectors)
}
