mod wrapper;
pub use wrapper::*;
//#include!(concat!(env!("OUT_DIR"), "/wrapper.rs"));

/*extern crate libc;
use libc::{c_char, c_void, c_int}

//fn lammps_open(c_int, *const *const char, MPI_Comm, *mut *mut c_void); // _/o\_
extern "C" fn lammps_open_no_mpi(c_int, *mut *mut c_char, *mut *mut c_void);
extern "C" fn lammps_close(*mut c_void);
extern "C" fn lammps_file(*mut c_void, *const c_char);
extern "C" fn lammps_command(*mut c_void, *const c_char) -> *const c_char;

extern "C" fn lammps_extract_global(*mut c_void, *const c_char) -> *const c_void;
extern "C" fn lammps_extract_atom(*mut c_void, *const c_char) -> *const c_void;
extern "C" fn lammps_extract_compute(*mut c_void, *const c_char, c_int, c_int) -> *const c_void;
extern "C" fn lammps_extract_fix(*mut c_void, *const c_char, c_int, c_int, c_int, c_int) -> *const c_void;
extern "C" fn lammps_extract_variable(*mut c_void, *const c_char, *const c_char) -> *const c_void;

extern "C" fn lammps_get_natoms(*const c_void) -> c_int;
extern "C" fn lammps_gather_atoms(*const c_void, *const c_char, c_int, c_int, *mut c_void);
extern "C" fn lammps_scatter_atoms(*const c_void, *const c_char, c_int, c_int, *mut c_void);
*/
