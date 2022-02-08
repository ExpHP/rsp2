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

#[cfg(feature = "mpi")]
use crate::mpi;
use crate::FailResult;
use std::os::raw::{c_int, c_void, c_double, c_char};
use crate::low_level::{ComputeStyle, ComputeType, Skews, LowLevelApi, Severity, ScatterGatherDatatype};

// Lammps exposes no API to obtain the error message length so we have to guess.
const MAX_ERROR_BYTES: usize = 4096;

/// A light wrapper around a LAMMPS instance which handles ownership
/// concerns and provides an interface that uses rust primitive types.
///
/// This implements the low-level API in a manner which directly wraps the C functions,
/// making it suitable for either of the following:
///
/// - For methods to be called on the only process, in a non-MPI setup.
/// - For methods to be called at the same time with the same arguments
///   on all processes, when MPI is used.
///
/// It is expressly NOT CLONE.
#[derive(Debug)]
pub(crate) struct LammpsOwner {
    // Pointer to LAMMPS instance.
    // - The 'static lifetime indicates that we own this.
    // - The lack of Clone prevents double-freeing.
    // - Box is not used because it is not allocated by Rust.
    ptr: &'static mut c_void,
    // Lammps holds some fingers into the argv we give it,
    // so we gotta make sure they don't get freed too early.
    _argv: CArgv,
}

impl LammpsOwner {
    /// # MPI
    ///
    /// The Lammps API generally does not specify what happens on non-root processes, so for now
    /// we follow rather conservative rules:
    ///
    /// This method should be called on all processes with the same arguments.
    /// Furthermore, after calling it, any other public method on it should be called on all
    /// processes with the same arguments in the same order.
    /// **This includes `Drop::drop`!**
    ///
    /// # Safety
    ///
    /// Construction of LammpsOwner is inherently unsafe because it is unsafe
    /// to use multiple instances simultaneously on separate threads.
    #[cfg(feature = "mpi")]
    pub(in crate::low_level) unsafe fn with_mpi<C: mpi::Communicator>(comm: &C, argv: &[&str]) -> FailResult<Self>
    {Ok({
        let mut argv = CArgv::from_strs(argv);
        let mut ptr: *mut c_void = std::ptr::null_mut();

        unsafe {
            lammps_sys::lammps_open(
                argv.len().try_to_c_int()?,
                argv.as_argv_ptr(),
                mpi::AsRaw::as_raw(comm),
                &mut ptr,
            );
        }

        let ptr = unsafe {
            ptr.as_mut()
        }.ok_or_else(|| format_err!("Lammps initialization failed"))?;

        LammpsOwner { _argv: argv, ptr }
    })}

    /// Construct without an MPI communicator.
    ///
    /// This is merely a wrapper around `lammps_open_no_mpi`.
    ///
    /// # MPI
    ///
    /// I don't know what happens if you use this when MPI is initialized.
    ///
    /// # Safety
    ///
    /// Construction of LammpsOwner is inherently unsafe because it is unsafe
    /// to use multiple instances simultaneously on separate threads.
    pub(crate) unsafe fn new(argv: &[&str]) -> FailResult<Self>
    {Ok({
        let mut argv = CArgv::from_strs(&argv);
        let mut ptr: *mut c_void = std::ptr::null_mut();

        unsafe {
            lammps_sys::lammps_open_no_mpi(
                argv.len().try_to_c_int()?,
                argv.as_argv_ptr(),
                &mut ptr,
            );
        }

        let ptr = unsafe {
            ptr.as_mut()
        }.ok_or_else(|| format_err!("Lammps initialization failed"))?;

        LammpsOwner { _argv: argv, ptr }
    })}
}


impl Drop for LammpsOwner {
    fn drop(&mut self) {
        // NOTE: not lammps_free!
        unsafe { lammps_sys::lammps_close(self.ptr); }
    }
}

//------------------------------

impl LowLevelApi for LammpsOwner {
    fn command(&mut self, cmd: String) -> FailResult<()>
    {Ok({
        let cmd = &cmd;

        api_trace!("lammps_command({:p}, {})", self.ptr, cmd);

        // FIXME: I still don't know if I'm supposed to free the output or not.
        // NOTE:  This returns "the command name" as a 'char *'.
        //        I pored over the Lammps source, and I, uh... *think* it's just
        //        a pointer into our string (which has had a null terminator
        //        forcefully thrust into it).  But I'm not sure.  - ML
        let ret = unsafe {
            with_temporary_c_str(cmd, |cmd| {
                lammps_sys::lammps_command(self.ptr, cmd)
            })
        };

        // NOTE: supposing that ret points into our argument (which has been
        //       freed), it is no longer safe to dereference.
        self.pop_error_as_result()?;

        assert!(!ret.is_null(), "lammps_command threw no exception, but returned null?!");
    })}

    fn get_natoms(&mut self) -> usize {
        api_trace!("lammps_get_natoms({:p})", self.ptr);
        let out = unsafe { lammps_sys::lammps_get_natoms(self.ptr) } as usize;
        self.assert_no_error();
        out
    }

    unsafe fn reset_box(
        &mut self,
        mut low: [f64; 3],
        mut high: [f64; 3],
        skews: Skews,
    ) -> FailResult<()>
    {Ok({
        let Skews { xy, yz, xz } = skews;
        api_trace!(
            "lammps_reset_box({:p}, {:?}, {:?}, {}, {}, {})",
            self.ptr, low, high, xy, yz, xz,
        );

        lammps_sys::lammps_reset_box(
            self.ptr,
            low.as_mut_ptr(),
            high.as_mut_ptr(),
            xy, yz, xz,
        );

        // NOTE: the version of the source I'm looking at does not trap for C++ exceptions,
        //       so this will never trigger, except perhaps in future versions of lammps...
        self.pop_error_as_result()?;
    })}

    fn init_atoms(&mut self, carts: Vec<[f64; 3]>, types: Vec<i64>) -> FailResult<()>
    {Ok({
        use slice_of_array::prelude::*;

        let mut carts = carts.flat().iter().map(|&x| x as c_double).collect::<Vec<_>>();
        let mut types = types.iter().map(|&x| x.try_to_c_int()).collect::<FailResult<Vec<_>>>()?;
        unsafe {
            lammps_sys::lammps_create_atoms(
                self.ptr,
                types.len().try_to_c_int()?, // int n
                std::ptr::null_mut(), // tagint *id
                types.as_mut_ptr(), // int *type
                carts.as_mut_ptr(), // double *x
                std::ptr::null_mut(), // double *v
                std::ptr::null_mut(), // imageint *image
                1, // int shrinkexceed
            );
        }
        self.pop_error_as_result()?;
    })}

    // shims to inherent methods so we can write unsafe helpers closer to code that uses them
    unsafe fn extract_compute_0d(&mut self, name: String) -> FailResult<f64>
    { self.impl_extract_compute_0d(&name) }

    unsafe fn extract_compute_1d(&mut self,
        name: String,
        style: ComputeStyle,
        len: usize,
    ) -> FailResult<Vec<f64>>
    { self.impl_extract_compute_1d(&name, style, len) }

    unsafe fn gather_atoms_i(&mut self, name: String, count: usize) -> FailResult<Vec<i64>>
    { self.impl_gather_atoms_i(&name, count) }

    unsafe fn gather_atoms_f(&mut self, name: String, count: usize) -> FailResult<Vec<f64>>
    { self.impl_gather_atoms_f(&name, count) }

    unsafe fn scatter_atoms_i(&mut self, name: String, data: Vec<i64>) -> FailResult<()>
    { self.impl_scatter_atoms_i(&name, &data) }

    unsafe fn scatter_atoms_f(&mut self, name: String, data: Vec<f64>) -> FailResult<()>
    { self.impl_scatter_atoms_f(&name, &data) }
}

//------------------------------
// error API (used internally)
//
// NOTE: Every call to an extern "C" function must be immediately
//       followed by one of these methods.
impl LammpsOwner {
    // (this is our '?')
    pub(in crate::low_level) fn pop_error_as_result(&mut self) -> Result<(), crate::LammpsError>
    {
        match self.pop_error() {
            None => Ok(()),
            Some((severity, message)) => {
                let backtrace = failure::Backtrace::new();
                Err(crate::LammpsError { severity, message, backtrace })
            },
        }
    }

    // (this is our 'unwrap')
    pub(in crate::low_level) fn assert_no_error(&mut self)
    {
        self.pop_error_as_result().unwrap_or_else(|e| {
            panic!("Unexpected error from LAMMPS: {}", e);
        });
    }

    // Read an error from the Lammps API if there is one.
    // (This removes the error, so that a second call will produce None.)
    pub(in crate::low_level) fn pop_error(&mut self) -> Option<(Severity, String)>
    {
        use lammps_sys::{lammps_get_last_error_message, lammps_has_error};

        api_trace!("lammps_has_error({:p})", self.ptr);
        let has_error = unsafe { lammps_has_error(self.ptr) } != 0;
        if !has_error {
            return None;
        };

        // +1 to guarantee a nul
        let mut buf = vec![0u8; MAX_ERROR_BYTES + 1];

        let severity_int = unsafe {
            api_trace!("lammps_get_last_error_message({:p}, (out), {})", self.ptr, MAX_ERROR_BYTES);
            lammps_get_last_error_message(
                self.ptr,
                buf.as_mut_ptr() as *mut c_char,
                MAX_ERROR_BYTES.try_to_c_int().unwrap(),
            )
        } as u32;
        let severity = Severity::from_int(severity_int).expect("lammps-wrap bug!");

        // truncate to written content
        let nul = buf.iter().position(|&c| c == b'\0').expect("lammps-wrap bug!");
        buf.truncate(nul);

        // (NOTE: the thought here was: if our guess for a maximum length
        //        happened to be right in the middle of a character,
        //        then we should cut off the invalid part...
        //        ...but now that I think about it, who says the error
        //        is even encoded in utf8?)
        let message = string_from_utf8_prefix(buf);
        Some((severity, message))
    }
}

//------------------------------

/// # Scatter/gather
impl LammpsOwner {
    // Gather an integer property across all atoms.
    //
    // unsafe because an incorrect 'count' or a non-integer field may cause an out-of-bounds read.
    #[allow(unused)] // FIXME issue #4
    unsafe fn impl_gather_atoms_i(&mut self, name: &str, count: usize) -> FailResult<Vec<i64>>
    {Ok({
        self.__gather_atoms_c_ty::<c_int>(name, ScatterGatherDatatype::Integer, count)?
            .into_iter().map(|x| x as i64).collect()
    })}

    // Gather a floating property across all atoms.
    //
    // unsafe because an incorrect 'count' or a non-floating field may cause an out-of-bounds read.
    unsafe fn impl_gather_atoms_f(&mut self, name: &str, count: usize) -> FailResult<Vec<f64>>
    {Ok({
        self.__gather_atoms_c_ty::<c_double>(name, ScatterGatherDatatype::Float, count)?
            .into_iter().map(|x| x as f64).collect()
    })}

    // unsafe because an incorrect 'count', 'ty', or 'T' may cause an out-of-bounds read.
    //
    // I would very much like for this and the rest of the __gather_atoms family to
    // return Option::None on failure.  Unfortunately, Lammps doesn't want to talk to us
    // about the problems it has with us.
    unsafe fn __gather_atoms_c_ty<T:Default + Clone>(
        &mut self,
        name: &str,
        ty: ScatterGatherDatatype,
        count: usize,
    ) -> FailResult<Vec<T>>
    {Ok({
        let natoms = self.get_natoms();
        let mut out = vec![T::default(); count * natoms];
        self.__gather_atoms_to_buf_c_ty(name, ty, &mut out)?;

        // I'm not sure if there is any way at all for us to verify that the operation
        // actually succeeded without screenscraping diagnostic output from LAMMPS.
        // The function only prints a warning on failure and does not set the error state.
        //   - ML
        let yolo = out;
        yolo
    })}

    // unsafe because an incorrect 'count', 'ty', or 'T' may cause an out-of-bounds read.
    //
    // I would very much like for this and the rest of the __gather_atoms family to
    // return Option::None on failure.  Unfortunately, Lammps doesn't want to talk to us
    // about the problems it has with us.
    unsafe fn __gather_atoms_to_buf_c_ty<T>(
        &mut self,
        name: &str,
        ty: ScatterGatherDatatype,
        buf: &mut [T],
    ) -> FailResult<()>
    {Ok({
        let natoms = self.get_natoms();

        assert_eq!(buf.len() % natoms, 0);
        let count = (buf.len() / natoms).try_to_c_int()?;
        let ty = ty.into();

        api_trace!("lammps_gather_atoms({:p}, {}, {}, {}, (out))", self.ptr, name, ty, count);

        with_temporary_c_str(name, |name| {
            lammps_sys::lammps_gather_atoms(
                self.ptr, name, ty, count,
                buf.as_mut_ptr() as *mut c_void,
            );
        });

        // NOTE: Known cases where this is Err:
        // * None so far.
        self.pop_error_as_result()?;
    })}

    // Write an integer property across all atoms.
    //
    // unsafe because a non-integer field may copy data of the wrong size,
    // and data of inappropriate length could cause an out of bounds write.
    unsafe fn impl_scatter_atoms_i(&mut self, name: &str, data: &[i64]) -> FailResult<()>
    {Ok({
        let mut cdata: Vec<_> = data.iter().map(|&x| x.try_to_c_int()).collect::<FailResult<_>>()?;
        self.__scatter_atoms_checked_c_ty(name, ScatterGatherDatatype::Integer, &mut cdata)?;
    })}

    // Write a floating property across all atoms.
    //
    // unsafe because a non-floating field may copy data of the wrong size,
    // and data of inappropriate length could cause an out of bounds write.
    unsafe fn impl_scatter_atoms_f(&mut self, name: &str, data: &[f64]) -> FailResult<()>
    {Ok({
        let mut cdata: Vec<_> = data.iter().map(|&x| x as c_double).collect();
        self.__scatter_atoms_checked_c_ty(name, ScatterGatherDatatype::Float, &mut cdata)?;
    })}

    // Variant of `__scatter_atoms_c_ty` that checks for evidence of an error
    // having occurred.
    unsafe fn __scatter_atoms_checked_c_ty<T: BitCheck + Clone>(
        &mut self,
        name: &str,
        ty: ScatterGatherDatatype,
        data: &mut [T],
    ) -> FailResult<()>
    {Ok({
        assert!(data.len() > 0, "No data to scatter!?"); // always at least one atom
        self.__scatter_atoms_c_ty(name, ty, data)?;

        // check if the operation succeeded by reading back and checking the first element
        // as a heuristic. Note this will never give a false positive, but it may give
        // false negatives (i.e. "should-be" errors that aren't detected)
        let expected = data[0].clone();
        data[0] = T::different_bits(expected.clone());

        assert!(!data[0].bit_eq(&expected));
        self.__gather_atoms_to_buf_c_ty(name, ty, data)?;
        assert!(data[0].bit_eq(&expected), "detected failure in LAMMPS scatter_atoms or gather_atoms");
    })}

    // unsafe because an incorrect 'ty' or 'T' may cause an out-of-bounds write.
    //
    // NOTE: if the operation fails on the LAMMPS side of things, this variant will
    //       quietly return `Ok(())`.  See the `_checked` variant.
    unsafe fn __scatter_atoms_c_ty<T>(
        &mut self,
        name: &str,
        ty: ScatterGatherDatatype,
        data: &mut [T]
    ) -> FailResult<()>
    {Ok({

        let natoms = self.get_natoms();
        assert_eq!(data.len() % natoms, 0);
        let count = (data.len() / natoms).try_to_c_int()?;
        let ty = ty.into();

        api_trace!("lammps_scatter_atoms({:p}, {}, {}, {}, (data))", self.ptr, name, ty, count);

        with_temporary_c_str(name, |name| {
            lammps_sys::lammps_scatter_atoms(
                self.ptr, name, ty, count,
                data.as_mut_ptr() as *mut c_void,
            );
        });

        // NOTE: Known cases where this is Err:
        // * None so far.
        self.pop_error_as_result()?;
    })}
}

// FIXME: remove once TryInto is stable
trait ToCInt: Sized + std::fmt::Display + Ord {
    fn max_value() -> u64;
    fn cast_to_c_int(self) -> c_int;
    fn cast_from_c_int(x: c_int) -> Self;
    fn try_to_c_int(self) -> FailResult<c_int> {
        // (first check is const, please optimize out...)
        if (c_int::max_value() as u64) < Self::max_value() {
            if self > Self::cast_from_c_int(c_int::max_value()) {
                bail!("attempt to overflow c_int with value {}", self);
            }
        }
        Ok(self.cast_to_c_int())
    }
}

impl ToCInt for usize {
    #[inline(always)] fn max_value() -> u64 { Self::max_value() as u64 }
    #[inline(always)] fn cast_to_c_int(self) -> c_int { self as c_int }
    #[inline(always)] fn cast_from_c_int(x: c_int) -> Self { x as Self }
}

impl ToCInt for i64 {
    #[inline(always)] fn max_value() -> u64 { Self::max_value() as u64 }
    #[inline(always)] fn cast_to_c_int(self) -> c_int { self as c_int }
    #[inline(always)] fn cast_from_c_int(x: c_int) -> Self { x as Self }
}

/// Used to implement the heuristic for detecting errors in gather/scatter
trait BitCheck {
    /// A comparison function that is reflexive even if Self is `PartialEq`.
    fn bit_eq(&self, other: &Self) -> bool;

    /// Produce a value such that `!self.bit_eq(self.different_bits())`
    fn different_bits(self) -> Self;
}

impl BitCheck for c_double {
    fn bit_eq(&self, other: &Self) -> bool
    { self.to_bits() == other.to_bits() }

    fn different_bits(self) -> Self
    {
        // Because all NaNs have an exponent filled with ones (signalling or not),
        // this satisfies the `!self.bit_eq(self.different_bits())` property even
        // if `from_bits` masks away signalling NaNs
        Self::from_bits(!self.to_bits())
    }
}

impl BitCheck for c_int {
    fn bit_eq(&self, other: &Self) -> bool
    { self == other }

    fn different_bits(self) -> Self
    { !self }
}

//------------------------------

/// # Computes
impl LammpsOwner {
    // Read a scalar compute, possibly computing it in the process.
    //
    // NOTE: There are warnings in extract_compute about making sure it is valid
    //       to run the compute.  I'm not sure what it means, and it sounds to me
    //       like this could possibly actually cause UB; I just have no idea how.
    unsafe fn impl_extract_compute_0d(&mut self, name: &str) -> FailResult<f64>
    {Ok({
        self.__extract_compute_any_d(name, ComputeStyle::Global, ComputeType::Scalar)?
            .clone()
    })}

    // Read a vector compute, possibly computing it in the process.
    //
    // NOTE: There are warnings in extract_compute about making sure it is valid
    //       to run the compute.  I'm not sure what it means, and it sounds to me
    //       like this could possibly actually cause UB; I just have no idea how.
    unsafe fn impl_extract_compute_1d(
        &mut self,
        name: &str,
        style: ComputeStyle,
        len: usize,
    ) -> FailResult<Vec<f64>>
    {Ok({
        assert_ne!(len, 0); // because extract_compute_any_d returns &T
        let p = self.__extract_compute_any_d(name, style, ComputeType::Vector)?;

        std::slice::from_raw_parts(p, len)
            .iter().map(|&c| c as f64).collect()
    })}

    // CAUTION: Note the unbound lifetime!!! This is only to factor out the null check;
    //          the output is NOT valid for arbitrary lifetimes!
    //          (so make sure to copy the output into an owned form ASAP)
    unsafe fn __extract_compute_any_d<'unbound>(
        &mut self,
        name: &str,
        style: ComputeStyle,
        ty: ComputeType,
    ) -> FailResult<&'unbound c_double>
    {Ok({
        let style: c_int = style.into();
        let ty: c_int = ty.into();

        api_trace!(
            "lammps_extract_compute({:p}, {}, {}, {})",
            self.ptr, name, style, ty,
        );

        let out_ptr = with_temporary_c_str(name, |name| {
            unsafe { lammps_sys::lammps_extract_compute(self.ptr, name, style, ty)}
        }) as *mut c_double;


        // NOTE: Known cases where this produces Err:
        // * None so far.
        self.pop_error_as_result()?;

        // NOTE: Known cases where the pointer is NULL:
        // * (bug in lammps-wrap) Name provided does not belong to a compute.
        out_ptr.as_ref()
            .unwrap_or_else(|| panic!("Could not extract {:?}", name))
    })}
}

//--------------------------------------
// ffi utilz

// Get the longest valid utf8 prefix as a String.
// This function never fails.
fn string_from_utf8_prefix(buf: Vec<u8>) -> String
{
    String::from_utf8(buf)
        .unwrap_or_else(|e| {
            let valid_len = e.utf8_error().valid_up_to();
            let mut bytes = e.into_bytes();
            bytes.truncate(valid_len);

            String::from_utf8(bytes).expect("bug!")
        })
}

// Temporarily allocate a C string for the duration of a closure.
//
// The closure may make arbitrary modifications to the string's
// content (including writes of interior NUL bytes), but must not
// write beyond the `s.len() + 1` allocated bytes for the C string.
fn with_temporary_c_str<B, F>(s: &str, f: F) -> B
    where F: FnOnce(*mut c_char) -> B
{
    // It is not safe to use CString here; LAMMPS may write NUL bytes
    // that change the length of the string.
    let mut bytes = s.to_string().into_bytes();
    bytes.push(0);
    f(bytes.as_mut_ptr() as *mut c_char)
}

use self::black_hole::BlackHole;
mod black_hole {
    use std::fmt;

    /// Contains something that is dangerous to obtain references to.
    ///
    /// It will never be seen again (except to be dropped).
    pub struct BlackHole<T>(T);
    impl<T> BlackHole<T> {
        pub fn entrap(x: T) -> BlackHole<T> { BlackHole(x) }
    }

    impl<T> fmt::Debug for BlackHole<T> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "BlackHole(_)")
        }
    }
}

/// An argv for C ffi, allocated and managed by rust code.
///
/// It is safe to move this around without fear of invalidating
/// pointers; only dropping the CArgv will invalidate the pointers.
/// It is expressly NOT CLONE.
#[derive(Debug)]
pub(crate) struct CArgv {
    ptrs: Vec<*mut c_char>,
    _allocs: BlackHole<Vec<Box<[u8]>>>,
}

impl CArgv {
    pub(crate) fn from_strs(strs: &[&str]) -> Self {
        let mut allocs = strs.iter().map(|&s| {
            let mut bytes = s.to_string().into_bytes();
            bytes.push(0);
            bytes.into_boxed_slice() as Box<[_]>
        }).collect::<Vec<_>>();

        let ptrs = allocs.iter_mut()
            .map(|x| x.as_mut_ptr()) // to *mut [u8]
            .map(|x: *mut u8| x) // CoerceUnsized to *mut u8
            .map(|x| x as *mut c_char)
            .collect();

        CArgv { _allocs: BlackHole::entrap(allocs), ptrs }
    }

    pub(crate) fn len(&self) -> usize
    { self.ptrs.len() }

    // Exposes the argv pointer.
    //
    // NOTE: the returned pointers (both the strings and the vector holding
    //       them) will become invalid when the CArgv is dropped.
    pub(crate) fn as_argv_ptr(&mut self) -> *mut *mut c_char
    { self.ptrs.as_mut_ptr() }
}

// FIXME disgusting and shouldn't be necessary, but I'm pretty sure it's safe.
//
// This is needed by MpiLammpsOwner because of the fact that the LammpsOwner instance
// in this case is actually stored in the Builder (which is intended to be Send + Sync)
unsafe impl Send for CArgv {}
