
use ::{Result, Error, ErrorKind};

use ::std::ffi::{CString};
use ::std::sync::Mutex;
use ::std::os::raw::{c_int, c_void, c_double, c_char};

// Lammps exposes no API to obtain the error message length so we have to guess.
const MAX_ERROR_BYTES: usize = 4096;

macro_rules! c_enums {
    (
        $(
            [$($vis:tt)*] enum $Type:ident {
                // tt so it can double as expr and pat
                $($Variant:ident = $value:tt,)+
            }
        )+
    ) => {
        $(
            #[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
            $($vis)* enum $Type {
                $($Variant = $value,)+
            }

            impl $Type {
                #[allow(unused)]
                pub fn from_int(x: u32) -> Result<$Type>
                { match x {
                    $($value => Ok($Type::$Variant),)+
                    _ => bail!("Invalid value {} for {}", x, stringify!($Type)),
                }}
            }
        )+
    };
}

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

macro_rules! derive_into_from_as_cast {
    ($($A:ty as $B:ty;)*)
    => { $(
        impl From<$A> for $B {
            fn from(a: $A) -> $B { a as $B }
        }
    )* };
}

derive_into_from_as_cast!{
    ComputeStyle as c_int;
    ComputeType as c_int;
    ScatterGatherDatatype as c_int;
}

/// A light wrapper around a LAMMPS instance which handles ownership
/// concerns and provides an interface that uses rust primitive types.
///
/// The design is fairly conservative, trying to make as few design choices
/// as necessary.  As a result, some exposed functions are still unsafe.
/// The expectation is that another, higher-level wrapper will be built
/// around this.
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
    argv: CArgv,
}

lazy_static! {
    /// HACK to work around the segfaults that appear to result from
    /// lammps being instantiated from multiple threads at the same time.
    ///
    /// This mutex is grabbed during the creation and destruction.
    ///
    /// NOTE: This is a leaf in the lock hierarchy (we never attempt to
    ///  grab other locks while holding this lock).
    static ref INSTANTIATION_LOCK: Mutex<()> = Default::default();
}

impl Drop for LammpsOwner {
    fn drop(&mut self) {
        let _guard = INSTANTIATION_LOCK.lock();

        // NOTE: not lammps_free!
        unsafe { ::lammps_sys::lammps_close(self.ptr); }
    }
}

impl LammpsOwner {
    pub fn new(argv: &[&str]) -> Result<LammpsOwner>
    {Ok({
        let mut argv = CArgv::from_strs(argv)?;
        let mut ptr: *mut c_void = ::std::ptr::null_mut();

        {
            // this will not deadlock because we never attempt to
            // acquire other locks while we hold this lock.
            let _guard = INSTANTIATION_LOCK.lock();
            unsafe {
                ::lammps_sys::lammps_open_no_mpi(
                    argv.len() as c_int,
                    argv.as_argv_ptr(),
                    &mut ptr,
                );
            }
        }

        let ptr = unsafe {
            ptr.as_mut()
        }.ok_or_else(|| err!("Lammps initialization failed"))?;

        LammpsOwner { argv, ptr }
    })}
}

mod cli { // name shows up in log output
    pub fn trace(cmd: &str) {
        trace!("{}", cmd);
    }
}

//------------------------------
// the basics
impl LammpsOwner {
    /// Invokes `lammps_command`.
    pub fn command(&mut self, cmd: &str) -> Result<()>
    {Ok({
        cli::trace(cmd);

        // FIXME: I still don't know if I'm supposed to free the output or not.
        // NOTE:  This returns "the command name" as a 'char *'.
        //        I pored over the Lammps source, and I, uh... *think* it's just
        //        a pointer into our string (which has had a null terminator
        //        forcefully thrust into it).  But I'm not sure.  - ML
        let ret = unsafe {
            with_temporary_c_str(cmd, |cmd| {
                ::lammps_sys::lammps_command(self.ptr, cmd)
            })? // NulError
        };

        // NOTE: supposing that ret points into our argument (which has been
        //       freed), it is no longer safe to dereference.
        self.pop_error_as_result()?;

        assert!(!ret.is_null(), "lammps_command threw no exception, but returned null?!");
    })}

    /// Repeatedly invokes `lammps_command`.
    ///
    /// That is to say, it does NOT invoke `lammps_command_list`.
    /// (Though one should sincerely *hope* this difference does not matter...)
    pub fn commands<S: ToString>(&mut self, cmds: &[S]) -> Result<()>
    {Ok({
        for s in cmds { self.command(&s.to_string())?; }
    })}

    pub fn get_natoms(&mut self) -> usize
    {
        let out = unsafe { ::lammps_sys::lammps_get_natoms(self.ptr) } as usize;
        self.assert_no_error();
        out
    }
}

//------------------------------
// error API (used internally)
//
// NOTE: Every call to an extern "C" function must be immediately
//       followed by one of these methods.
impl LammpsOwner {

    // (this is our '?')
    fn pop_error_as_result(&mut self) -> Result<()>
    {Ok({
        match self.pop_error() {
            None => {},
            Some((severity, s)) => bail!(ErrorKind::Lammps(severity, s)),
        }
    })}

    // (this is our 'unwrap')
    fn assert_no_error(&mut self)
    {
        self.pop_error_as_result().unwrap_or_else(|e| {
            use ::error_chain::ChainedError;
            panic!("Unexpected error from LAMMPS: {}", e.display_chain());
        });
    }

    // Read an error from the Lammps API if there is one.
    // (This removes the error, so that a second call will produce None.)
    fn pop_error(&mut self) -> Option<(Severity, String)>
    {
        use ::lammps_sys::{lammps_get_last_error_message, lammps_has_error};

        let has_error = unsafe { lammps_has_error(self.ptr) } != 0;
        if !has_error {
            return None;
        };

        // +1 to guarantee a nul
        let mut buf = vec![0u8; MAX_ERROR_BYTES + 1];

        let severity_int = unsafe {
            lammps_get_last_error_message(
                self.ptr,
                buf.as_mut_ptr() as *mut c_char,
                MAX_ERROR_BYTES as c_int,
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
// scatter/gather
impl LammpsOwner {

    // Gather an integer property across all atoms.
    //
    // unsafe because an incorrect 'count' or a non-integer field may cause an out-of-bounds read.
    pub unsafe fn gather_atoms_i(&mut self, name: &str, count: usize) -> Result<Vec<i64>>
    {Ok({
        self.__gather_atoms_c_ty::<c_int>(name, ScatterGatherDatatype::Integer, count)?
            .into_iter().map(|x| x as i64).collect()
    })}

    // Gather a floating property across all atoms.
    //
    // unsafe because an incorrect 'count' or a non-floating field may cause an out-of-bounds read.
    pub unsafe fn gather_atoms_f(&mut self, name: &str, count: usize) -> Result<Vec<f64>>
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
    ) -> Result<Vec<T>>
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
    ) -> Result <()>
    {Ok({
        let natoms = self.get_natoms();

        assert_eq!(buf.len() % natoms, 0);
        let count = buf.len() / natoms;

        with_temporary_c_str(name, |name| {
            ::lammps_sys::lammps_gather_atoms(
                self.ptr, name, ty.into(), count as c_int,
                buf.as_mut_ptr() as *mut c_void,
            );
        })?;

        // NOTE: Known cases where this is Err:
        // * None so far.
        self.pop_error_as_result()?;
    })}

    // Write an integer property across all atoms.
    //
    // unsafe because a non-integer field may copy data of the wrong size,
    // and data of inappropriate length could cause an out of bounds write.
    pub unsafe fn scatter_atoms_i(&mut self, name: &str, data: &[i64]) -> Result<()>
    {Ok({
        let mut cdata: Vec<_> = data.iter().map(|&x| x as c_int).collect();
        self.__scatter_atoms_checked_c_ty(name, ScatterGatherDatatype::Integer, &mut cdata)?;
    })}

    // Write a floating property across all atoms.
    //
    // unsafe because a non-floating field may copy data of the wrong size,
    // and data of inappropriate length could cause an out of bounds write.
    pub unsafe fn scatter_atoms_f(&mut self, name: &str, data: &[f64]) -> Result<()>
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
    ) -> Result<()>
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
    ) -> Result<()>
    {Ok({
        let natoms = self.get_natoms();
        assert_eq!(data.len() % natoms, 0);
        let count = data.len() / natoms;

        with_temporary_c_str(name, |name| {
            ::lammps_sys::lammps_scatter_atoms(
                self.ptr, name, ty.into(), count as c_int,
                data.as_mut_ptr() as *mut c_void,
            );
        })?;

        // NOTE: Known cases where this is Err:
        // * None so far.
        self.pop_error_as_result()?;
    })}

    //------------------------------
    // computes

    // Read a scalar compute, possibly computing it in the process.
    //
    // NOTE: There are warnings in extract_compute about making sure it is valid
    //       to run the compute.  I'm not sure what it means, and it sounds to me
    //       like this could possibly actually cause UB; I just have no idea how.
    pub unsafe fn extract_compute_0d(&mut self, name: &str) -> Result<f64>
    {Ok({
        let out_ptr = with_temporary_c_str(name, |name| {
            unsafe { ::lammps_sys::lammps_extract_compute(
                self.ptr, name,
                ComputeStyle::Global.into(),
                ComputeType::Scalar.into(),
            )}
        })? as *mut c_double;

        // NOTE: Known cases where this produces Err:
        // * None so far.
        self.pop_error_as_result()?;

        // NOTE: Known cases where the pointer is NULL:
        // * (bug in lammps-wrap) Name provided does not belong to a compute.
        unsafe { out_ptr.as_ref() }
            .cloned()
            .ok_or_else(|| Error::from(format!("could not extract {:?}", name)))?
    })}

    // Read a vector compute, possibly computing it in the process.
    //
    // NOTE: There are warnings in extract_compute about making sure it is valid
    //       to run the compute.  I'm not sure what it means, and it sounds to me
    //       like this could possibly actually cause UB; I just have no idea how.
    pub unsafe fn extract_compute_1d(
        &mut self,
        name: &str,
        style: ComputeStyle,
        len: usize,
    ) -> Result<Vec<f64>>
    {Ok({
        let out_ptr = with_temporary_c_str(name, |name| {
            unsafe { ::lammps_sys::lammps_extract_compute(
                self.ptr, name,
                style.into(),
                ComputeType::Vector.into(),
            )}
        })? as *mut c_double;

        // NOTE: See extract_compute_0d for a breakdown of the error cases.
        self.pop_error_as_result()?;
        let p =
            out_ptr.as_ref()
            .ok_or_else(|| err!("Could not extract {:?}", name))?;

        ::std::slice::from_raw_parts(p, len)
            .iter().map(|&c| c as f64).collect()
    })}
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

// Temporarily allocate a c string for the duration of a closure.
unsafe fn with_temporary_c_str<B, F>(s: &str, f: F) -> Result<B>
where F: FnOnce(*mut c_char) -> B
{Ok({
    let p = CString::new(s)?.into_raw();
    let out = f(p);
    let _ = CString::from_raw(p); // free memory
    out
})}

/// An argv for C ffi, allocated and managed by rust code.
///
/// It is safe to move this around without fear of invalidating
/// pointers; only dropping the CArgv will invalidate the pointers.
/// It is expressly NOT CLONE.
#[derive(Debug)]
pub(crate) struct CArgv(Vec<*mut c_char>);

impl CArgv {
    pub(crate) fn from_strs(strs: &[&str]) -> Result<Self> {
        // do all the nul-checking up front before we leak anything
        let strs: Vec<_> = strs.iter().map(|&s| Ok(CString::new(s)?)).collect::<Result<_>>()?;

        Ok(CArgv(strs.into_iter().map(|s| s.into_raw()).collect()))
    }

    pub(crate) fn len(&self) -> usize { self.0.len() }

    // Exposes the argv pointer.
    //
    // NOTE: For safety, it is important not to modify the inner '*mut' pointers
    //       to point to different memory.  This is not possible without the use
    //       of unsafe code.
    pub(crate) fn as_argv_ptr(&mut self) -> *mut *mut c_char { self.0.as_mut_ptr() }
}

impl Drop for CArgv {
    fn drop(&mut self) {
        // Unleak each inner pointer to free its memory.
        while let Some(s) = self.0.pop() {
            // Assuming the inner pointers were never modified,
            // this is safe because each pointer was allocated by rust.
            unsafe { let _ = CString::from_raw(s); }
        }
    }
}
