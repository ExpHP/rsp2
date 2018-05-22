
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

use ::rsp2_newtype_indices::cast_index;
use ::rsp2_array_types::M33;
use ::slice_of_array::prelude::*;

use ::std::process;

const PY_NOOP: &'static str = indoc!(r#"
    #!/usr/bin/env python3
"#);

const PY_CHECK_SCIPY_AVAILABILITY: &'static str = indoc!(r#"
    #!/usr/bin/env python3
    import numpy as np
    import scipy.sparse
    import scipy.sparse.linalg as spla
"#);

const PY_CALL_EIGSH: &'static str = indoc!(r#"
    #!/usr/bin/env python3

    import json
    import sys
    import numpy as np
    import scipy.sparse
    import scipy.sparse.linalg as spla

    d = json.load(sys.stdin)
    json.dump(d, open('/tmp/zzzzzzzzzzzzzzzw', 'w'))
    kw = d.pop('kw')
    m = d.pop('matrix')
    assert not d

    data = np.array(m['complex_blocks'])
    json.dump(data.tolist(), open('/tmp/zzzzzzzzzzzzzzzx', 'w'))
    data = 1.0 * data[:, 0] + 1.0j * data[:, 1]
    json.dump(data.real.tolist(), open('/tmp/zzzzzzzzzzzzzzzy', 'w'))
    assert data.ndim == 3 and data.shape[1] == data.shape[2] == 3
    m = scipy.sparse.bsr_matrix(
        (data, m['col'], m['row_ptr']),
        shape=tuple(3*x for x in m['dim']),
    ).tocsc()
    json.dump(m.todense().real.tolist(), open('/tmp/zzzzzzzzzzzzzzzz', 'w'))

    (vals, vecs) = scipy.sparse.linalg.eigsh(m, **kw)

    real = vecs.real.T.tolist()
    imag = vecs.imag.T.tolist()
    vals = vals.tolist()
    json.dump((vals, (real, imag)), sys.stdout)
    print()
"#);

#[derive(Serialize)]
struct Input<'a> {
    matrix: Matrix<'a>,
    kw: PyKw,
}

#[derive(Serialize)]
struct Matrix<'a> {
    // these should be suitable for col_ptr and row_ind (i.e. no additional factor of 3).
    pub dim: (usize, usize),

    // CSR format (or technically Block CSR).
    pub complex_blocks: &'a [[M33; 2]],
    pub col: &'a [usize],
    pub row_ptr: &'a [usize],
}

#[derive(Serialize)]
enum Which {
    #[serde(rename = "LM")] LargestMagnitude,
    #[serde(rename = "SM")] SmallestMagnitude,
    #[serde(rename = "LA")] MostPositive,
    #[serde(rename = "SA")] MostNegative,
    #[serde(rename = "BE")] HalfAndHalf,
}

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

pub fn check_scipy_availability() -> FailResult<()> {
    call_script_and_check_success(PY_NOOP, PythonExecutionError)?;
    call_script_and_check_success(PY_CHECK_SCIPY_AVAILABILITY, ScipyAvailabilityError)?;
    Ok(())
}

fn call_eigsh(input: &Input<'_>) -> FailResult<(Vec<f64>, Basis3)> {
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
    Ok((vals, Basis3(kets)))
}

#[derive(Debug, Fail)]
#[fail(display = "an error occurred running the most trivial python script")]
pub struct PythonExecutionError;

#[derive(Debug, Fail)]
#[fail(display = "an error occurred importing numpy and scipy")]
pub struct ScipyAvailabilityError;

fn call_script_and_check_success<E: ::failure::Fail>(
    script: &'static str,
    error: E,
) -> FailResult<()>
{Ok({
    use ::std::process::Stdio;

    let tmp = ::rsp2_fs_util::TempDir::new("rsp2")?;
    let path = tmp.path().join("script.py");

    ::std::fs::write(&path, script)?;

    let mut cmd = process::Command::new("python3");
    cmd.arg(path);

    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());
    let mut child = cmd.spawn()?;
    let child_stdout = child.stdout.take().unwrap();
    let child_stderr = child.stderr.take().unwrap();

    let stdout_worker = ::stdout::log_worker(child_stdout);
    let stderr_worker = ::stderr::log_worker(child_stderr);

    if !child.wait()?.success() {
        throw!(error);
    }

    let _ = stdout_worker.join();
    let _ = stderr_worker.join();
})}

fn call_script_and_communicate<In, Out>(
    script: &'static str,
    stdin_data: &In,
) -> FailResult<Out>
where
    In: ::serde::Serialize,
    Out: for<'de> ::serde::Deserialize<'de>,
{Ok({
    use ::std::process::Stdio;

    let tmp = ::rsp2_fs_util::TempDir::new("rsp2")?;
    let path = tmp.path().join("script.py");

    ::std::fs::write(&path, script)?;

    let mut cmd = process::Command::new("python3");
    cmd.arg(path);
    cmd.stdin(Stdio::piped());
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());

    let mut child = cmd.spawn()?;
    let child_stdin = child.stdin.take().unwrap();
    let child_stdout = child.stdout.take().unwrap();
    let child_stderr = child.stderr.take().unwrap();

    let stderr_worker = ::stderr::log_worker(child_stderr);

    ::serde_json::to_writer(child_stdin, stdin_data)?;
    let stdout = ::serde_json::from_reader(child_stdout)?;

    if !child.wait()?.success() {
        bail!("an error occurred in a python script");
    }

    let _ = stderr_worker.join();

    stdout
})}

//-------------------------------------------------------------------------------

use ::math::dynmat::DynamicalMatrix;

impl DynamicalMatrix {
    fn py_matrix(&self) -> Matrix<'_> {
        // validate it because scipy apparently does not.
        // (I would think it could be done at *light speed* compared to all the magic
        //  python type-hackery they have to deal with, but whatever)
        self.0.validate().unwrap();
        Matrix {
            dim: self.0.dim,
            complex_blocks: &self.0.val,
            col: cast_index(&self.0.col),
            row_ptr: &self.0.row_ptr.raw,
        }
    }

    pub fn compute_most_negative_eigensolutions(&self, how_many: usize) -> FailResult<(Vec<f64>, Basis3)> {
        call_eigsh(&Input {
            matrix: self.py_matrix(),
            kw: PyKw {
                how_many: Some(how_many),
                which: Some(Which::MostNegative),
                ..Default::default()
            },
        })
    }

    pub fn compute_most_extreme_eigensolutions(&self, how_many: usize) -> FailResult<(Vec<f64>, Basis3)> {
        call_eigsh(&Input {
            matrix: self.py_matrix(),
            kw: PyKw {
                how_many: Some(how_many),
                ..Default::default()
            },
        })
    }
}
