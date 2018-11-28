/* ********************************************************************** **
**  This file is part of rsp2.                                            **
**                                                                        **
**  rsp2 is free software: you can redistribute it and/or modify it under **
**  the terms of the GNU General Public License as published by the Free  **
**  Software Foundation, either version 3 of the License, or (at your     **
**  option) any later version.                                            **
**                                                                        **
**      http://www.gnu.org/licenses/                                      **
**                                                                        **
** Do note that, while the whole of rsp2 is licensed under the GPL, many  **
** parts of it are licensed under more permissive terms.                  **
** ********************************************************************** */

use crate::FailResult;

use ::std::os::raw::{c_int};
use ::std::fmt;
use ::failure::Backtrace;

macro_rules! api_trace {
    ($($t:tt)*) => { log!(target: crate::API_TRACE_TARGET, crate::API_TRACE_LEVEL, $($t)*) };
}

/// An error thrown by the LAMMPS C API.
#[derive(Debug, Fail)]
pub struct LammpsError {
    pub(crate) backtrace: Backtrace,
    pub(crate) severity: Severity,
    pub(crate) message: String,
}

impl fmt::Display for LammpsError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f, "LAMMPS threw {}: {}",
            match self.severity {
                Severity::Recoverable => "an exception",
                Severity::Fatal => "a fatal exception",
            },
            self.message,
        )
    }
}

#[macro_use]
mod c_enum_macros;

pub(crate) use self::plain::LammpsOwner;
mod plain;

#[cfg(feature = "mpi")]
pub(crate) mod mpi;
#[cfg(feature = "mpi")]
pub(crate) mod mpi_helper;

c_enums!{
    [pub(crate)] enum ComputeStyle {
        Global = 0,
        PerAtom = 1,
        Local = 2,
    }

    [] enum ComputeType {
        Scalar = 0,
        Vector = 1,
        Array = 2, // 2D
    }

    [] enum ScatterGatherDatatype {
        Integer = 0,
        Float = 1,
    }

    [pub] enum Severity {
        Recoverable = 1,
        Fatal = 2,
    }
}

derive_into_from_as_cast!{
    ComputeStyle as c_int;
    ComputeType as c_int;
    ScatterGatherDatatype as c_int;
}

// (struct with named fields to create fewer independent places where
//  things could be written in the wrong order)
#[derive(Debug, Copy, Clone)]
pub(crate) struct Skews {
    pub(crate) xy: f64,
    pub(crate) yz: f64,
    pub(crate) xz: f64,
}

/// Common interface for the low-level API, which wraps the C api with more rusty signatures.
///
/// The design is fairly conservative, trying to make as few design choices
/// as necessary.  As a result, some exposed functions are still unsafe.
///
/// # Why is this messy thing a trait?
///
/// For better or worse, this thing is basically the former set of inherent methods of LammpsOwner,
/// recast as a trait to allow abstracting over the presence or absence of MPI:
///
/// * There is `LammpsOwner`, which directly wraps C API functions.
/// * ...and there's now `MpiLammpsOwner`, a conditionally-compiled type which uses `MpiOnDemand`
///   to factor out all logic for non-root processes into a single, moderately ugly event loop.
///
/// Adding this layer of indirection vastly simplifies the implementation of the higher-level APIs
/// in this crate so that everything can continue to be written as though it is always executed on
/// a single process.
///
/// # Why is every argument taken as an owned type?
///
/// For ease of codegen.  Notice that the `Input` enum in `low_level::mpi` must consist entirely
/// of owned data, because it must implement `Broadcast`.
pub(crate) trait LowLevelApi {
    fn command(&mut self, cmd: String) -> FailResult<()>;

    fn get_natoms(&mut self) -> usize;

    /// Initializes all positions in a manner which ensures that they are
    /// associated with the correct processors.
    ///
    /// This appears on LowLevelApi as a workaround to help minimize MPI overhead.
    fn init_atoms(&mut self, carts: Vec<[f64; 3]>, types: Vec<i64>) -> FailResult<()>;

    /// Set the lattice.
    ///
    /// * Their documentation says "assumes `domain->set_initial_box()` has been invoked previously".
    ///   (basically, this means we must call the `create_box` command.)
    /// * Because their implementation does not trap for exceptions, it clearly
    ///   accepts boxes that would not otherwise be allowed by lammps.
    ///   I don't know if violation of these invariants can trigger UB, but again,
    ///   we might as well just assume the worst.
    unsafe fn reset_box(
        &mut self,
        low: [f64; 3],
        high: [f64; 3],
        skews: Skews,
    ) -> FailResult<()>;

    /// Read a scalar compute, possibly computing it in the process.
    ///
    /// NOTE: There are warnings in extract_compute about making sure it is valid
    ///       to run the compute.  I'm not sure what it means, and it sounds to me
    ///       like this could possibly actually cause UB; I just have no idea how.
    unsafe fn extract_compute_0d(&mut self, name: String) -> FailResult<f64>;

    /// Read a vector compute, possibly computing it in the process.
    ///
    /// NOTE: There are warnings in extract_compute about making sure it is valid
    ///       to run the compute.  I'm not sure what it means, and it sounds to me
    ///       like this could possibly actually cause UB; I just have no idea how.
    unsafe fn extract_compute_1d(
        &mut self,
        name: String,
        style: ComputeStyle,
        len: usize,
    ) -> FailResult<Vec<f64>>;

    /// Gather an integer property across all atoms.
    ///
    /// Unsafe because an incorrect 'count' or a non-integer field may cause an out-of-bounds read.
    #[allow(unused)] // FIXME issue #4
    unsafe fn gather_atoms_i(&mut self, name: String, count: usize) -> FailResult<Vec<i64>>;

    /// Gather a floating property across all atoms.
    ///
    /// Unsafe because an incorrect 'count' or a non-floating field may cause an out-of-bounds read.
    unsafe fn gather_atoms_f(&mut self, name: String, count: usize) -> FailResult<Vec<f64>>;

    /// Write an integer property across all atoms.
    ///
    /// Unsafe because a non-integer field may copy data of the wrong size,
    /// and data of inappropriate length could cause an out of bounds write.
    unsafe fn scatter_atoms_i(&mut self, name: String, data: Vec<i64>) -> FailResult<()>;

    /// Write a floating property across all atoms.
    ///
    /// Unsafe because a non-floating field may copy data of the wrong size,
    /// and data of inappropriate length could cause an out of bounds write.
    unsafe fn scatter_atoms_f(&mut self, name: String, data: Vec<f64>) -> FailResult<()>;
}

impl dyn LowLevelApi {
    /// Repeatedly invokes `lammps_command`.
    ///
    /// That is to say, it does NOT invoke `lammps_command_list`.
    /// (Though one should sincerely *hope* this difference does not matter...)
    pub(crate) fn commands<S: ToString>(&mut self, cmds: impl IntoIterator<Item=S>) -> FailResult<()>
    { cmds.into_iter().try_for_each(|s| self.command(s.to_string())) }
}
