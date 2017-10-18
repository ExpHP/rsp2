/* FIXME need GNU GPL header
 */

#![allow(unused_unsafe)]

extern crate slice_of_array;
extern crate rsp2_structure;
extern crate lammps_sys;
#[macro_use] extern crate log;
extern crate chrono;

pub type StdResult<T, E> = ::std::result::Result<T, E>;
pub type Result<T> = StdResult<T, Error>;

use ::std::os::raw::{c_void, c_int, c_double};
use ::std::ffi::CString;
use ::slice_of_array::prelude::*;
use ::rsp2_structure::{CoordStructure, Lattice};

#[derive(Debug,Copy,Clone,PartialEq,Eq,PartialOrd,Ord,Hash)]
enum ComputeStyle {
    Global = 0,
    PerAtom = 1,
    Local = 2,
}

#[derive(Debug,Copy,Clone,PartialEq,Eq,PartialOrd,Ord,Hash)]
enum ComputeType {
    Scalar = 0,
    Vector = 1,
    Array = 2, // 2D
}

#[derive(Debug,Copy,Clone,PartialEq,Eq,PartialOrd,Ord,Hash)]
enum ScatterGatherDatatype {
    Integer = 0,
    Float = 1,
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

impl Drop for LammpsOwner {
    fn drop(&mut self) {
        // NOTE: not lammps_free!
        unsafe { ::lammps_sys::lammps_close(self.ptr); }
    }
}

#[derive(Debug,Copy,Clone,PartialEq,Eq,PartialOrd,Ord,Hash)]
pub struct Error; // TODO

impl LammpsOwner {
    // FIXME current signature is a lie, we never return errors
    pub fn new(argv: &[&str]) -> Result<LammpsOwner> {
        let mut argv = CArgv::from_strs(argv).unwrap(); // FIXME: unwrap out of laziness
        let mut ptr: *mut c_void = ::std::ptr::null_mut();
        unsafe {
            ::lammps_sys::lammps_open_no_mpi(
                argv.len() as c_int,
                argv.as_argv_ptr(),
                &mut ptr,
            );
        }

        Ok(LammpsOwner {
            argv,
            // FIXME should probably produce some sort of Err
            ptr: unsafe { ptr.as_mut() }.expect("Lammps initialization failed"),
        })
    }
}

mod cli { // name shows up in log output
    pub fn trace(cmd: &str) {
        trace!("{}", cmd);
    }
}

impl LammpsOwner {
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
    pub fn command(&mut self, cmd: &str) -> Result<()> {
        cli::trace(cmd);
        let cmd = CString::new(cmd).expect("embedded NUL!").into_raw();
        unsafe {
            // FIXME: I still don't know if I'm supposed to free the output or not.
            // NOTE:  This returns "the command name" as a 'char *'.
            //        I pored over the Lammps source, and I, uh... *think* it's just
            //        a pointer into our string (which has had a null terminator
            //        forcefully thrust into it).  But I'm not sure.  - ML
            let ret = ::lammps_sys::lammps_command(self.ptr, cmd);
            assert!(!ret.is_null());
            let _ = CString::from_raw(cmd);
        }
        Ok(())
    }

    // convenience wrapper
    // NOTE: repeatedly invokes `lammps_command`, not `lammps_command_list`
    pub fn commands<S: AsRef<str>>(&mut self, cmds: &[S]) -> Result<()> {
        for s in cmds { self.command(s.as_ref())?; }
        Ok(())
    }

    pub fn get_natoms(&mut self) -> usize {
        unsafe { ::lammps_sys::lammps_get_natoms(self.ptr) as usize }
    }

    // Gather an integer property across all atoms.
    //
    // unsafe because an incorrect 'count' or a non-integer field may cause an out-of-bounds read.
    pub unsafe fn gather_atoms_i(&mut self, name: &str, count: usize) -> Vec<i64> {
        self.__gather_atoms_c_ty::<c_int>(name, ScatterGatherDatatype::Integer, count)
            .into_iter().map(|x| x as i64).collect()
    }

    // Gather a floating property across all atoms.
    //
    // unsafe because an incorrect 'count' or a non-floating field may cause an out-of-bounds read.
    pub unsafe fn gather_atoms_f(&mut self, name: &str, count: usize) -> Vec<f64> {
        self.__gather_atoms_c_ty::<c_double>(name, ScatterGatherDatatype::Float, count)
            .into_iter().map(|x| x as f64).collect()
    }

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
    ) -> Vec<T>
    {
        let name = CString::new(name).expect("embedded NUL!").into_raw();
        let natoms = self.get_natoms();

        let mut out = vec![T::default(); count * natoms];
        ::lammps_sys::lammps_gather_atoms(
            self.ptr, name, ty.into(), count as c_int,
            out.as_mut_ptr() as *mut c_void,
        );

        let _ = CString::from_raw(name); // free memory

        // I'm not sure if there is any way at all for us to verify that the operation
        // actually succeeded without screenscraping diagnostic output from LAMMPS.
        // The function returns nothing, and prints a warning on failure.  - ML
        let yolo = out;
        yolo
    }

    // Write an integer property across all atoms.
    //
    // unsafe because a non-integer field may copy data of the wrong size,
    // and data of inappropriate length could cause an out of bounds write.
    pub unsafe fn scatter_atoms_i(&mut self, name: &str, data: &[i64]) {
        let mut cdata: Vec<_> = data.iter().map(|&x| x as c_int).collect();
        self.__scatter_atoms_c_ty(name, ScatterGatherDatatype::Integer, &mut cdata);
    }

    // Write a floating property across all atoms.
    //
    // unsafe because a non-floating field may copy data of the wrong size,
    // and data of inappropriate length could cause an out of bounds write.
    unsafe fn scatter_atoms_f(&mut self, name: &str, data: &[f64]) {
        let mut cdata: Vec<_> = data.iter().map(|&x| x as c_double).collect();
        self.__scatter_atoms_c_ty(name, ScatterGatherDatatype::Float, &mut cdata);
    }

    // unsafe because an incorrect 'ty' or 'T' may cause an out-of-bounds write.
    unsafe fn __scatter_atoms_c_ty<T>(
        &mut self,
        name: &str,
        ty: ScatterGatherDatatype,
        data: &mut [T]
    )
    {
        let name = CString::new(name).expect("embedded NUL!").into_raw();
        let natoms = self.get_natoms();
        assert_eq!(data.len() % natoms, 0);
        let count = data.len() / natoms;

        ::lammps_sys::lammps_scatter_atoms(
            self.ptr, name, ty.into(), count as c_int,
            data.as_mut_ptr() as *mut c_void,
        );

        let _ = CString::from_raw(name); // free memory

        // I'm not sure if there is any way at all for us to verify that the operation
        // actually succeeded without screenscraping diagnostic output from LAMMPS.
        // The function returns nothing, and prints a warning on failure.  - ML
    }

    // Read a scalar compute, possibly computing it in the process.
    //
    // NOTE: There are warnings in extract_compute about making sure it is valid
    //       to run the compute.  I'm not sure what it means, and it sounds to me
    //       like this could possibly actually cause UB; I just have no idea how.
    pub unsafe fn extract_compute_0d(&mut self, name: &str) -> Option<f64> {
        let id = CString::new(name).expect("internal NUL").into_raw();
        let out_ptr = unsafe { ::lammps_sys::lammps_extract_compute(
            self.ptr,
            id,
            ComputeStyle::Global.into(),
            ComputeType::Scalar.into(),
        ) };
        unsafe { (out_ptr as *mut c_double).as_ref() }.cloned()
    }

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
    ) -> Option<Vec<f64>>
    {
        let id = CString::new(name).expect("internal NUL").into_raw();
        let out_ptr = unsafe { ::lammps_sys::lammps_extract_compute(
            self.ptr,
            id,
            style.into(),
            ComputeType::Vector.into(),
        ) } as *mut c_double;

        out_ptr.as_ref().map(|p|
            ::std::slice::from_raw_parts(p, len)
                .iter().map(|&c| c as f64).collect()
        )
    }
}

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

impl Lammps {

    pub fn new_carbon(structure: CoordStructure) -> Result<Lammps>
    {Ok({
        // Lammps script based on code from Colin Daniels.

        let carts = structure.to_carts();


        let lmp = ::LammpsOwner::new(&["lammps",
            "-screen", "none",
            "-log", "none", // logs opened from CLI are truncated, but we want to append
        ])?;
        let ptr = ::std::cell::RefCell::new(lmp);
        let me = Lammps { ptr, structure: MaybeDirty::new_dirty(structure) };

        let log_file = "lammps.log";
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
        }

        me.ptr.borrow_mut().commands(&[
            &format!("log {} append", log_file)[..],
            "package omp 0",
            "units metal",                  // Angstroms, picoseconds, eV
            "processors * * *",             // automatic processor mapping
            "atom_style atomic",            // attributes to store per-atom
            "thermo_modify lost error",     // don't let atoms disappear without telling us
            "atom_modify map array",        // store all positions in an array
            "atom_modify sort 0 0.0",       // don't reorder atoms during simulation
        ])?;

        // garbage initial lattice
        me.ptr.borrow_mut().commands(&[
            "boundary p p p",               // (p)eriodic, (f)ixed, (s)hrinkwrap
            "box tilt large",               // triclinic
            "region sim prism 0 2 0 2 0 2 1 1 1", // garbage garbage garbage
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

        unsafe { self.ptr.borrow_mut().scatter_atoms_f("x", carts.flat()); };
    })}

    fn send_lmp_lattice(&mut self) -> Result<()>
    {Ok({
        let lattice = self.structure.get().lattice().matrix();
        assert_eq!(0f64, lattice[0][1], "non-triangular lattices not yet supported");
        assert_eq!(0f64, lattice[0][2], "non-triangular lattices not yet supported");
        assert_eq!(0f64, lattice[1][2], "non-triangular lattices not yet supported");

        self.ptr.borrow_mut().commands(&[
            &format!("change_box all x final 0 {}", lattice[0][0]),
            &format!("change_box all y final 0 {}", lattice[1][1]),
            &format!("change_box all z final 0 {}", lattice[2][2]),
            &format!("change_box all xy final {}", lattice[1][0]),
            &format!("change_box all xz final {}", lattice[2][0]),
            &format!("change_box all yz final {}", lattice[2][1]),
        ])?;
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

        unsafe { self.ptr.borrow_mut().extract_compute_0d("RSP2_PE") }.unwrap()
    })}

    pub fn compute_grad(&mut self) -> Result<Vec<[f64; 3]>>
    {Ok({
        self.update_computation()?;

        let grad = {
            let mut grad = unsafe { self.ptr.borrow_mut().gather_atoms_f("f", 3) };
            for x in &mut grad { *x *= -1.0 };
            grad
        };
        grad.nest().to_vec()
    })}

    pub fn compute_pressure(&mut self) -> Result<[f64; 6]>
    {Ok({
        self.update_computation()?;

        // as_array().clone() doesn't manage type inference here as well as deref...
        *unsafe {
            self.ptr.borrow_mut().extract_compute_1d("RSP2_Pressure", ComputeStyle::Global, 6)
        }.unwrap().as_array()
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
    // NOTE: this is now mostly tested indirectly
    //       through the other crates that use it.
}
