/* FIXME need GNU GPL header
 */

#![allow(unused_unsafe)]

extern crate sp2_array_utils;
extern crate lammps_sys;
extern crate ndarray;
#[macro_use] extern crate log;

pub mod structure_tools;
//pub mod algo;

pub mod metropolis; // HACK only here to compile-test

use ::std::os::raw::{c_void, c_int, c_char, c_double};
use ::std::ffi::{CString, CStr, NulError};
use ::sp2_array_utils::slice::prelude::*;

#[derive(Debug,Copy,Clone,PartialEq,Eq,PartialOrd,Ord,Hash)]
#[allow(unused)]
enum ComputeStyle {
	Global = 0,
	PerAtom = 1,
	Local = 2,
}

#[derive(Debug,Copy,Clone,PartialEq,Eq,PartialOrd,Ord,Hash)]
#[allow(unused)]
enum ComputeType {
	Scalar = 0,
	Vector = 1,
	Array = 2, // 2D
}

#[derive(Debug,Copy,Clone,PartialEq,Eq,PartialOrd,Ord,Hash)]
#[allow(unused)]
enum ScatterGatherDatatype {
	Integer = 0,
	Float = 1,
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
	pub fn new(argv: &[&str]) -> Result<LammpsOwner, Error> {
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
	pub fn command(&mut self, cmd: &str) -> Result<(), Error> {
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
	pub fn commands<S: AsRef<str>>(&mut self, cmds: &[S]) -> Result<(), Error> {
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
			self.ptr, name, ty as c_int, count as c_int,
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
			self.ptr, name, ty as c_int, count as c_int,
			data.as_mut_ptr() as *mut c_void,
		);

		let _ = CString::from_raw(name); // free memory

		// I'm not sure if there is any way at all for us to verify that the operation
		// actually succeeded without screenscraping diagnostic output from LAMMPS.
		// The function returns nothing, and prints a warning on failure.  - ML
	}

	// Read a scalar compute, possibly computing it in the process.
	pub unsafe fn extract_compute_0d(&mut self, name: &str) -> Option<f64> {
		let id = CString::new(name).expect("internal NUL").into_raw();
		let out_ptr = unsafe { ::lammps_sys::lammps_extract_compute(
			self.ptr,
			id,
			ComputeStyle::Global as c_int,
			ComputeType::Scalar as c_int,
		) };
		unsafe { (out_ptr as *mut c_double).as_ref() }.cloned()
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
}

impl Lammps {
	pub fn new_carbon(lattice: &[[f64; 3]; 3], carts: &[[f64; 3]]) -> Result<Lammps, Error> {
		let lmp = ::LammpsOwner::new(&["lammps", "-screen", "none"])?;
		let me = Lammps{ ptr: std::cell::RefCell::new(lmp) };

		me.ptr.borrow_mut().commands(&[
			"package omp 0",
			"units metal",                  // Angstroms, picoseconds, eV
        	"processors * * *",             // automatic processor mapping
        	"atom_style atomic",            // attributes to store per-atom
        	"thermo_modify lost error",     // don't let atoms disappear without telling us
        	"atom_modify map array",        // store all positions in an array
        	"atom_modify sort 0 0.0",       // don't reorder atoms during simulation
			"boundary p p p",               // (p)eriodic, (f)ixed, (s)hrinkwrap
		])?;
		
		let xx = lattice[0][0];
		assert_eq!(0f64, lattice[0][1], "non-triangular lattices not yet supported");
		assert_eq!(0f64, lattice[0][2], "non-triangular lattices not yet supported");
		let xy = lattice[1][0];
		let yy = lattice[1][1];
		assert_eq!(0f64, lattice[1][2], "non-triangular lattices not yet supported");
		let xz = lattice[2][0];
		let yz = lattice[2][1];
		let zz = lattice[2][2];

		// NOTE: sp2 used "region sim block" (and did not emit "box tilt large")
		//       for orthogonal vectors, which I can only assume makes it faster
		me.ptr.borrow_mut().commands(&[
            "box tilt large",
            &format!("region sim prism 0 {xx} 0 {yy} 0 {zz} {xy} {xz} {yz}",
				xx=xx, yy=yy, zz=zz, xy=xy, xz=xz, yz=yz),
        ])?;

		let n_atom_types = 2;
        me.ptr.borrow_mut().command(&format!("create_box {} sim", n_atom_types))?;
    	me.ptr.borrow_mut().commands(&[
			"mass 1 1.0",
			"mass 2 12.01"
		])?;

		let this_atom_type = 2;
		let seed = 0xbeef;
		me.ptr.borrow_mut().command(&format!("create_atoms {} random {} {} NULL remap yes",
			this_atom_type, carts.len(), seed))?;

		let sigma_scale = 3.0; // LJ Range (x3.4 A)
		let lj = 1;            // on/off
		let torsion = 0;       // on/off
		//let lj_scale = 1.0;
		me.ptr.borrow_mut().commands(&[
        	&format!("pair_style airebo/omp {} {} {}", sigma_scale, lj, torsion),
            &format!("pair_coeff * * CH.airebo H C"), // read potential info
        	//&format!("pair_coeff * * lj/scale {}", lj_scale), // set lj potential scaling factor (HACK)
        	&format!("compute MyPE all pe"), // set up a compute for energy
		])?;

		Ok(me)
	}

	pub fn compute(&mut self, carts: &[[f64; 3]]) -> Result<(f64, Vec<[f64; 3]>), Error> {
		let natoms = self.ptr.borrow_mut().get_natoms();
		assert_eq!(natoms, carts.len());

		unsafe { self.ptr.borrow_mut().scatter_atoms_f("x", carts.flat()); };

		self.ptr.borrow_mut().command("run 0")?;

		let value = unsafe { self.ptr.borrow_mut().extract_compute_0d("MyPE") }.unwrap();
		let grad = {
			let mut grad = unsafe { self.ptr.borrow_mut().gather_atoms_f("f", 3) };
			for x in &mut grad { *x *= -1.0 };
			grad
		};	
		Ok((value, grad.nest().to_vec()))
	}
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
	use sp2_array_utils::slice::prelude::*;

	fn lammps() -> Result<super::Lammps, super::Error> {
		let a = 2.46;
		let (xx,  _,  _) = (a, 0.0, 0.0);
		let (xy, yy,  _) = (-a/2.0, a*0.5*3f64.sqrt(), 0.0);
		let (xz, yz, zz) = (0.0, 0.0, 10.0);
		let fracs = vec![
			0.0, 0.0, 0.0,
			2./3., 1./3., 0.0
		];
		let (((xx,yy,zz),(xy,xz,yz)), fracs) = ::structure_tools::diagonal_supercell((7,7,1), ((xx,yy,zz), (xy,xz,yz)), &fracs);
		let carts = ::structure_tools::cartesian(((xx,yy,zz),(xy,xz,yz)), &fracs);

		let lattice = [[xx, 0., 0.], [xy, yy, 0.], [xz, yz, zz]];
		super::Lammps::new_carbon(&lattice, carts.nest())
	}

	#[test]
	fn lel() {
		//let mut lmp = ::Lammps::new(&["lammps", "-screen", "none"]).unwrap();
		let mut lmp = lammps();
        setup_lammps(&mut lmp);
		let force = unsafe { lmp.gather_atoms_f("f", 3) };
		assert!(true, "{:?}", force);
	}
}
