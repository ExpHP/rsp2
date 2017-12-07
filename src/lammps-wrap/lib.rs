/* FIXME need GNU GPL header
 */

#![allow(unused_unsafe)]

extern crate slice_of_array;
extern crate rsp2_structure;
extern crate lammps_sys;
#[macro_use] extern crate log;
#[macro_use] extern crate error_chain;
#[macro_use] extern crate lazy_static;
extern crate chrono;

error_chain! {
    foreign_links {
        NulError(::std::ffi::NulError);
    }

    errors {
        Lammps(severity: Severity, message: String) {
            description("LAMMPS threw an exception"),
            display("LAMMPS threw {}: {}",
                match *severity {
                    Severity::Recoverable => "an exception",
                    Severity::Fatal => "a fatal exception",
                },
                message
            ),
        }
    }
}

// constructs an Error, as opposed to bail!() which diverges with a Result.
// Useful in `ok_or_else`.
macro_rules! err {
    ($($t:tt)+)
    => { Error::from(format!($($t)+)) }
}

pub type StdResult<T, E> = ::std::result::Result<T, E>;

use ::std::os::raw::{c_void, c_char, c_int, c_double};
use ::std::ffi::CString;
use ::std::sync::Mutex;
use ::std::path::{Path, PathBuf};
use ::slice_of_array::prelude::*;
use ::rsp2_structure::{CoordStructure, Lattice};

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
    [] enum ComputeStyle {
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
struct LammpsOwner {
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

impl LammpsOwner {

    //------------------------------
    // the basics

    /// Invokes `lammps_command`.
    ///
    /// # Panics
    ///
    /// Panics on embedded NUL (`'\0'`) characters, but that's the least of your concerns.
    ///
    /// If the command is ill-formed, Lammps may **abort** the process.  As in,
    /// `MpiAbort`, `MpiFinalize`, and everybody's favorite, `libc::exit`.
    /// Good luck, and have fun!
    ///
    /// In some cases, it might panic instead of aborting.  This occurs when we can visibly
    /// detect that Lammps did not like the command (e.g. it returned NULL).  This might
    /// be changed to Result::Err later, but for now, we panic because I have no idea if and
    /// when this ever actually happens.  - ML
    // TODO: Looks like we can change (some of?) these aborts into detectable errors
    //        by defining LAMMPS_EXCEPTIONS at build time,
    //        which introduces `lammps_has_error` and `lammps_get_last_error_message`
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

    // convenience wrapper
    // NOTE: repeatedly invokes `lammps_command`, not `lammps_command_list`
    pub fn commands<S: AsRef<str>>(&mut self, cmds: &[S]) -> Result<()>
    {Ok({
        for s in cmds { self.command(s.as_ref())?; }
    })}

    pub fn get_natoms(&mut self) -> usize
    {
        let out = unsafe { ::lammps_sys::lammps_get_natoms(self.ptr) } as usize;
        self.assert_no_error();
        out
    }

    //------------------------------
    // error API (used internally)

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

    //------------------------------
    // scatter/gather

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

        with_temporary_c_str(name, |name| {
            ::lammps_sys::lammps_gather_atoms(
                self.ptr, name, ty.into(), count as c_int,
                out.as_mut_ptr() as *mut c_void,
            );
        })?;

        // NOTE: Known cases where this is Err:
        // * None so far.
        self.pop_error_as_result()?;

        // I'm not sure if there is any way at all for us to verify that the operation
        // actually succeeded without screenscraping diagnostic output from LAMMPS.
        // The function only prints a warning on failure and does not set the error state.
        //   - ML
        let yolo = out;
        yolo
    })}

    // Write an integer property across all atoms.
    //
    // unsafe because a non-integer field may copy data of the wrong size,
    // and data of inappropriate length could cause an out of bounds write.
    pub unsafe fn scatter_atoms_i(&mut self, name: &str, data: &[i64]) -> Result<()>
    {Ok({
        let mut cdata: Vec<_> = data.iter().map(|&x| x as c_int).collect();
        self.__scatter_atoms_c_ty(name, ScatterGatherDatatype::Integer, &mut cdata)?;
    })}

    // Write a floating property across all atoms.
    //
    // unsafe because a non-floating field may copy data of the wrong size,
    // and data of inappropriate length could cause an out of bounds write.
    unsafe fn scatter_atoms_f(&mut self, name: &str, data: &[f64]) -> Result<()>
    {Ok({
        let mut cdata: Vec<_> = data.iter().map(|&x| x as c_double).collect();
        self.__scatter_atoms_c_ty(name, ScatterGatherDatatype::Float, &mut cdata)?;
    })}

    // unsafe because an incorrect 'ty' or 'T' may cause an out-of-bounds write.
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

        // I'm not sure if there is any way at all for us to verify that the operation
        // actually succeeded without screenscraping diagnostic output from LAMMPS.
        // The function only prints a warning on failure and does not set the error state.
        //   - ML
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

pub struct Lammps {
    /// Put Lammps behind a RefCell so we can paper over things like `get_natoms(&mut self)`
    /// without needing to manually verify that no mutation occurs in the Lammps source.
    ///
    /// We don't want to slam user code with runtime violations of RefCell's aliasing checks.
    /// Thankfully, `RefCell` isn't `Sync`, so we only need to worry about single-threaded
    /// violations. I don't think these will be able to occur so long as we are careful with
    /// our API:
    ///
    /// - Be careful providing methods which return a value that extends the lifetime
    ///    of an immutably-borrowed (`&self`) receiver.
    /// - Be careful providing methods that borrow `&self` and take a callback.
    ///
    /// (NOTE: I'm not entirely sure if this is correct.)
    ptr: ::std::cell::RefCell<LammpsOwner>,

    /// The currently computed structure, encapsulated in a helper type
    /// that tracks dirtiness and helps us decide when we need to call lammps
    structure: MaybeDirty<CoordStructure>,
}

struct MaybeDirty<T> {
    // NOTE: Possible states for the members:
    //
    //        dirty:       clean:       when
    //        Some(s)       None       is dirty, and has never been clean.
    //        Some(s)      Some(s)     is dirty, but has been clean in the past.
    //         None        Some(s)     is currently clean.

    /// new data that has not been marked clean.
    dirty: Option<T>,
    /// the last data marked clean.
    clean: Option<T>,
}

impl<T> MaybeDirty<T> {
    pub fn new_dirty(x: T) -> MaybeDirty<T> {
        MaybeDirty {
            dirty: Some(x),
            clean: None,
        }
    }

    pub fn is_dirty(&self) -> bool
    { self.dirty.is_some() }

    pub fn last_clean(&self) -> Option<&T>
    { self.clean.as_ref() }

    pub fn get(&self) -> &T
    { self.dirty.as_ref().or(self.last_clean()).unwrap() }

    /// Get a mutable reference. This automatically marks the value as dirty.
    pub fn get_mut(&mut self) -> &mut T
    where T: Clone,
    {
        if self.dirty.is_none() {
            self.dirty = self.clean.clone();
        }
        self.dirty.as_mut().unwrap()
    }

    pub fn mark_clean(&mut self)
    {
        assert!(self.dirty.is_some() || self.clean.is_some());

        if self.dirty.is_some() {
            self.clean = self.dirty.take();
        }

        assert!(self.dirty.is_none());
        assert!(self.clean.is_some());
    }

    // test if f(x) is dirty by equality
    // HACK
    // this is only provided to help work around borrow checker issues
    pub fn is_projection_dirty<K, F>(&self, mut f: F) -> bool
    where
        K: PartialEq,
        F: FnMut(&T) -> K,
    {
        match (&self.clean, &self.dirty) {
            (&Some(ref clean), &Some(ref dirty)) => f(clean) != f(dirty),
            (&None, &Some(_)) => true,
            (&Some(_), &None) => false,
            (&None, &None) => unreachable!(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Builder {
    append_log: Option<PathBuf>,
    threaded: bool,
}

impl Default for Builder {
    fn default() -> Self
    { Builder::new() }
}

impl Builder {
    pub fn new() -> Self
    { Builder {
        append_log: None,
        threaded: true,
    }}

    pub fn append_log<P: AsRef<Path>>(&mut self, path: P) -> &mut Self
    { self.append_log = Some(path.as_ref().to_owned()); self }

    pub fn threaded(&mut self, value: bool) -> &mut Self
    { self.threaded = value; self }

    pub fn initialize_carbon(&self, structure: CoordStructure) -> Result<Lammps>
    { Lammps::from_builder_carbon(self, structure) }
}

impl Lammps {

    fn from_builder_carbon(builder: &Builder, structure: CoordStructure) -> Result<Lammps>
    {Ok({
        // Lammps script based on code from Colin Daniels.

        let carts = structure.to_carts();

        let lmp = ::LammpsOwner::new(&["lammps",
            "-screen", "none",
            "-log", "none", // logs opened from CLI are truncated, but we want to append
        ])?;
        let ptr = ::std::cell::RefCell::new(lmp);
        let me = Lammps { ptr, structure: MaybeDirty::new_dirty(structure) };

        if let Some(ref log_file) = builder.append_log
        {
            use ::std::io::prelude::*;
            if let Ok(mut f) =
                ::std::fs::OpenOptions::new()
                    .write(true)
                    .create(true)
                    .append(true)
                    .open(log_file)
            {
                let _ = writeln!(f, "---------------------------------------------");
                let _ = writeln!(f, "---- Begin run at {}", ::chrono::Local::now());
                let _ = writeln!(f, "---------------------------------------------");
            }

            me.ptr.borrow_mut().command(
                &format!("log {} append", log_file.display()),
            )?;
        }

        me.ptr.borrow_mut().commands(&[
            "package omp 0",
            "units metal",                  // Angstroms, picoseconds, eV
            match builder.threaded {
                true => "processors * * *",
                false => "processors 1 1 1",
            },
            "atom_style atomic",            // attributes to store per-atom
            "thermo_modify lost error",     // don't let atoms disappear without telling us
            "atom_modify map array",        // store all positions in an array
            "atom_modify sort 0 0.0",       // don't reorder atoms during simulation
        ])?;

        // (mostly) garbage initial lattice
        me.ptr.borrow_mut().commands(&[
            "boundary p p p",               // (p)eriodic, (f)ixed, (s)hrinkwrap
            "box tilt small",               // triclinic
            // NOTE: Initial skew factors must be zero to simplify
            //       reasoning about order in send_lmp_lattice.
            "region sim prism 0 2 0 2 0 2 0 0 0", // garbage garbage garbage
        ])?;

        {
            let n_atom_types = 2;
            me.ptr.borrow_mut().command(&format!("create_box {} sim", n_atom_types))?;
            me.ptr.borrow_mut().commands(&[
                "mass 1 1.0",
                "mass 2 12.01",
            ])?;
        }

        // garbage initial positions
        {
            let this_atom_type = 2;
            let seed = 0xbeef;
            me.ptr.borrow_mut().command(
                &format!("create_atoms {} random {} {} NULL remap yes",
                this_atom_type, carts.len(), seed))?;
        }

        {
            let sigma_scale = 3.0; // LJ Range (x3.4 A)
            let lj = 1;            // on/off
            let torsion = 0;       // on/off
            //let lj_scale = 1.0;
            me.ptr.borrow_mut().commands(&[
                &format!("pair_style airebo/omp {} {} {}", sigma_scale, lj, torsion),
                &format!("pair_coeff * * CH.airebo H C"), // read potential info
                //&format!("pair_coeff * * lj/scale {}", lj_scale), // set lj potential scaling factor (HACK)
            ])?;
        }

        // set up computes
        me.ptr.borrow_mut().commands(&[
            &format!("compute RSP2_PE all pe"),
            &format!("compute RSP2_Pressure all pressure NULL virial"),
        ])?;

        me
    })}

    //-------------------------------------------
    // modifying the system

    pub fn set_structure(&mut self, structure: CoordStructure) -> Result<()>
    {Ok({
        *self.structure.get_mut() = structure;
    })}

    pub fn set_carts(&mut self, carts: &[[f64; 3]]) -> Result<()>
    {Ok({
        self.structure.get_mut().set_carts(carts.to_vec());
    })}

    pub fn set_lattice(&mut self, lattice: Lattice) -> Result<()>
    {Ok({
        self.structure.get_mut().set_lattice(&lattice);
    })}

    //-------------------------------------------
    // sending input to lammps and running the main computation

    // This will rerun computations in lammps, but only if things have changed.
    //
    // At the end, (cached, updated) == (Some(_), None)
    fn update_computation(&mut self) -> Result<()>
    {Ok({
        if self.structure.is_dirty() {
            self.structure.get_mut().ensure_carts();

            // only send data that has changed from the cache.
            // This is done because it appears that lammps does some form of
            //  caching as well (lattice modifications in particular appear
            //  to increase the amount of computational work)
            if self.structure.is_projection_dirty(|s| s.lattice().clone()) {
                self.send_lmp_lattice()?;
            }

            if self.structure.is_projection_dirty(|s| s.to_carts()) {
                self.send_lmp_carts()?;
            }

            self.ptr.borrow_mut().command("run 0")?;
            self.structure.mark_clean();
        }
    })}

    fn send_lmp_carts(&mut self) -> Result<()>
    {Ok({
        let carts = self.structure.get().to_carts();
        let natoms = self.ptr.borrow_mut().get_natoms();
        assert_eq!(natoms, carts.len());

        unsafe { self.ptr.borrow_mut().scatter_atoms_f("x", carts.flat()) }?;
    })}

    fn send_lmp_lattice(&mut self) -> Result<()>
    {Ok({

        // From the documentation on 'change_box command':
        //
        //     Because the keywords used in this command are applied one at a time
        //     to the simulation box and the atoms in it, care must be taken with
        //     triclinic cells to avoid exceeding the limits on skew after each
        //     transformation in the sequence. If skew is exceeded before the final
        //     transformation this can be avoided by changing the order of the sequence,
        //     or breaking the transformation into two or more smaller transformations.
        //
        // There is nothing I can really say here that would not come across
        // as terribly, terribly rude. - ML

        let cur = self.structure.get().lattice().matrix();
        assert_eq!(0f64, cur[0][1], "non-triangular lattices not yet supported");
        assert_eq!(0f64, cur[0][2], "non-triangular lattices not yet supported");
        assert_eq!(0f64, cur[1][2], "non-triangular lattices not yet supported");

        #[derive(Debug, Copy, Clone, PartialEq, Eq)]
        enum Elem { Diagonal(&'static str), OffDiag(&'static str) }
        impl Elem {
            // tests whether a change to a given lattice matrix element
            //  could possibly cause us to crash if performed too soon.
            // This simple scheme allows us to handle MOST cases.
            pub fn should_defer(&self, old: f64, new: f64) -> bool
            { match *self {
                Elem::Diagonal(_) => new < old,
                Elem::OffDiag(_) => old.abs() < new.abs(),
            }}

            pub fn format(&self, value: f64) -> String
            { match *self {
                Elem::Diagonal(name) => format!("{} final 0 {}", name, value),
                Elem::OffDiag(name) => format!("{} final {}", name, value),
            }}
        }

        // NOTE: The order that these are declared are the order they will
        //       be set when no previous lattice has been sent.
        //       This order is safe for the garbage lattice with zero skew.
        let elems = &[
            (Elem::Diagonal("x"), (0, 0)),
            (Elem::Diagonal("y"), (1, 1)),
            (Elem::Diagonal("z"), (2, 2)),
            (Elem::OffDiag("xy"), (1, 0)),
            (Elem::OffDiag("xz"), (2, 0)),
            (Elem::OffDiag("yz"), (2, 1)),
        ];

        // Tragically, because we need to recall the last matrix that was set
        //  in order to determine what operations would cause lammps to crash,
        // we do need to match on whether or not a lattice has been saved,
        // which is exactly the sort of thing that MaybeDirty was supposed to
        // help prevent against.
        match self.structure.last_clean() {
            None => {
                let commands: Vec<_> =
                    elems.iter().map(|&(elem, (r, c))| {
                        format!("change_box all {}", elem.format(cur[r][c]))
                    }).collect();
                self.ptr.borrow_mut().commands(&commands)?;
            },

            Some(last) => {
                let last = last.lattice().matrix();
                // Look for cases that would defeat our simple ordering scheme.
                // (these are cases where both a diagonal and an off-diagonal would
                //   be classified as "defer until later", which is not enough information)
                // An example would be simultaneously skewing and shrinking a box.
                for r in 0..3 {
                    for c in 0..3 {
                        if Elem::Diagonal("").should_defer(cur[c][c], last[c][c]) &&
                            Elem::OffDiag("").should_defer(cur[r][c], last[r][c])
                        {
                            bail!("Tragically, you cannot simultaneously decrease a lattice \
                                diagonal element while increasing one of its off-diagonals. \
                                I'm sorry. You just can't.");
                        }
                    }
                }

                let mut do_early = vec![];
                let mut do_later = vec![];
                elems.iter().for_each(|&(elem, (r, c))| {
                    match elem.should_defer(last[r][c], cur[r][c]) {
                        true => { do_later.push((elem, (r, c))); },
                        false => { do_early.push((elem, (r, c))); },
                    }
                });

                let commands: Vec<_> =
                    None.into_iter()
                    .chain(do_early)
                    .chain(do_later)
                    .map(|(elem, (r, c))| {
                        format!("change_box all {}", elem.format(cur[r][c]))
                    })
                    .collect();

                self.ptr.borrow_mut().commands(&commands)?
            },
        }
    })}

    //-------------------------------------------
    // gathering output from lammps
    //
    // NOTE: Every method here should call update_computation().
    //       Don't worry about being sloppy with redundant calls;
    //       the method was designed for such usage.

    pub fn compute(&mut self) -> Result<(f64, Vec<[f64; 3]>)>
    {Ok({
        self.update_computation()?;

        (self.compute_value()?, self.compute_grad()?)
    })}

    pub fn compute_value(&mut self) -> Result<f64>
    {Ok({
        self.update_computation()?;

        unsafe { self.ptr.borrow_mut().extract_compute_0d("RSP2_PE") }?
    })}

    pub fn compute_force(&mut self) -> Result<Vec<[f64; 3]>>
    {Ok({
        self.update_computation()?;

        let grad = unsafe { self.ptr.borrow_mut().gather_atoms_f("f", 3)? };
        grad.nest().to_vec()
    })}

    pub fn compute_grad(&mut self) -> Result<Vec<[f64; 3]>>
    {Ok({
        self.update_computation()?;

        let mut grad = self.compute_force()?;
        for x in grad.flat_mut() {
            *x *= -1.0;
        }
        grad
    })}

    pub fn compute_pressure(&mut self) -> Result<[f64; 6]>
    {Ok({
        self.update_computation()?;

        // as_array().clone() doesn't manage type inference here as well as deref...
        *unsafe {
            self.ptr.borrow_mut().extract_compute_1d("RSP2_Pressure", ComputeStyle::Global, 6)
        }?.as_array()
    })}
}

use memory::CArgv;
mod memory {
    use ::std::os::raw::c_char;
    use ::std::ffi::{CString, NulError};

    /// An argv for C ffi, allocated and managed by rust code.
    ///
    /// It is safe to move this around without fear of invalidating
    /// pointers; only dropping the CArgv will invalidate the pointers.
    /// It is expressly NOT CLONE.
    #[derive(Debug)]
    pub(crate) struct CArgv(Vec<*mut c_char>);

    impl CArgv {
        pub(crate) fn from_strs(strs: &[&str]) -> Result<Self, NulError> {
            // do all the nul-checking up front before we leak anything
            let strs: Vec<_> = strs.iter().map(|&s| CString::new(s)).collect::<Result<_,_>>()?;

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
}

#[cfg(test)]
mod tests {
    use super::*;

    // NOTE: this is now mostly tested indirectly
    //       through the other crates that use it.

    // get a fresh Lammps instance on which arbitrary functions can be called.
    fn arbitrary_initialized_lammps() -> Lammps
    {
        use ::rsp2_structure::Coords;
        let structure = CoordStructure::new_coords(
            Lattice::eye(),
            Coords::Fracs(vec![[0.0; 3]]),
        );
        Builder::new().initialize_carbon(structure).unwrap()
    }

    macro_rules! assert_matches {
        ($pat:pat $(if $cond:expr)*, $val:expr $(,)*)
        => {
            match $val {
                $pat $(if $cond)* => {},
                ref e => {
                    panic!(
                        "assert_matches! failed:\nExpected:{}\n  Actual:{:?}",
                        stringify!($pat), e);
                },
            }
        };
    }

    #[test]
    fn exceptions()
    {
        let lmp = arbitrary_initialized_lammps();
        assert_matches!(
            Err(Error(ErrorKind::Lammps(Severity::Recoverable, _), _)),
            unsafe { lmp.ptr.borrow_mut().commands(&[
                // try to change to block with a nonzero skew
                "change_box all xy final 0.25",
                "change_box all ortho",
            ])}
        );
    }
}
