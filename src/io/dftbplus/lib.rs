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
#![allow(unused_unsafe)]
#![deny(unused_must_use)]

use dftbplus_sys as ffi;

use rsp2_structure::{Coords, Element};
use rsp2_array_types::V3;
use rsp2_fs_util as fsx;

#[allow(unused)] // rustc bug
use slice_of_array::prelude::*;
use std::io::prelude::*;
use std::path::{Path, PathBuf};
use std::os::raw::c_double;
use std::fmt;
use std::mem;

const ANGSTROM_TO_BOHR: f64 = 1.8897259886;
const HARTREE_TO_EV: f64 = 27.21138602;

pub type FailResult<T> = Result<T, failure::Error>;

#[derive(Debug, Clone)]
pub struct Hsd(String);

impl std::str::FromStr for Hsd {
    type Err = failure::Error;

    fn from_str(s: &str) -> FailResult<Self> {
        // (we don't actually bother parsing HSD)
        Ok(Hsd(s.to_string()))
    }
}

impl fmt::Display for Hsd {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

mod low_level {
    use super::*;

    #[derive(Debug)]
    pub struct Instance {
        p_dftb: ffi::DftbPlus,
        p_input: ffi::DftbPlusInput,
    }

    impl Instance {
        pub unsafe fn initialize(
            input_path: &Path,
            output_path: &Path,
        ) -> FailResult<Self> {
            // The backing memory for these can be safely deallocated at the end
            // because dftb+ copies the strings.
            let input_path = path_to_c_string(&input_path)?;
            let output_path = path_to_c_string(&output_path)?;

            let mut p_dftb: ffi::DftbPlus = unsafe { mem::uninitialized() };
            let mut p_input: ffi::DftbPlusInput = unsafe { mem::uninitialized() };

            // FIXME: how to detect errors that occur in DFTB+?
            // It looks to me like we are helplessly doomed to invoke undefined behavior...
            println!("========== Init DFTB+ ========");
            unsafe { ffi::dftbp_init(&mut p_dftb, output_path.as_ptr()); }
            unsafe { ffi::dftbp_get_input_from_file(&mut p_dftb, input_path.as_ptr(), &mut p_input); }
            unsafe { ffi::dftbp_process_input(&mut p_dftb, &mut p_input); }

            Ok(Instance { p_dftb, p_input })
        }

        fn num_atoms(&mut self) -> usize {
            unsafe { ffi::dftbp_get_nr_atoms(&mut self.p_dftb) as usize }
        }

        /// Units: Angstrom
        pub fn set_coords(
            &mut self,
            coords: &Coords,
        ) -> FailResult<()> {
            assert_eq!(self.num_atoms(), coords.len());

            let angstroms_to_bohrs = |vs: &[V3]| {
                vs.iter().map(|v| v * ANGSTROM_TO_BOHR).collect::<Vec<_>>()
            };
            let carts = angstroms_to_bohrs(&coords.to_carts());
            let lattice_vecs = angstroms_to_bohrs(coords.lattice().vectors());

            unsafe {
                ffi::dftbp_set_coords_and_lattice_vecs(
                    &mut self.p_dftb,
                    carts.flat().as_ptr() as *const c_double,
                    lattice_vecs.flat().as_ptr() as *const c_double,
                )
            }

            Ok(())
        }

        /// Units: eV
        pub fn get_energy(&mut self) -> FailResult<f64> {
            let mut hartree = std::f64::NAN as c_double;
            unsafe {
                ffi::dftbp_get_energy(&mut self.p_dftb, &mut hartree);
            }

            Ok(hartree as f64 * HARTREE_TO_EV)
        }

        /// Units: eV/Angstrom
        pub fn get_grad(&mut self) -> FailResult<Vec<V3>> {
            let mut hartree_per_bohr = vec![V3([std::f64::NAN as c_double; 3]); self.num_atoms()];

            unsafe {
                ffi::dftbp_get_gradients(
                    &mut self.p_dftb,
                    hartree_per_bohr.flat_mut().as_mut_ptr() as *mut c_double,
                );
            }

            let grad = {
                hartree_per_bohr.into_iter()
                    .map(|v| v.map(|x| x as f64))
                    .map(|v| v * (HARTREE_TO_EV * ANGSTROM_TO_BOHR))
                    .collect()
            };
            Ok(grad)
        }
    }

    impl Drop for Instance {
        fn drop(&mut self) {
            println!("========== Drop DFTB+ ========");
            unsafe { ffi::dftbp_final(&mut self.p_dftb); }
        }
    }
}

#[derive(Debug)]
pub struct DftbPlus {
    instance: low_level::Instance,
    tempdir: fsx::TempDir,
    elements: Vec<Element>,
}

#[derive(Debug, Clone)]
pub struct Builder {
    hsd: Hsd,
    elements: Option<Vec<Element>>,
    initial_coords: Option<Coords>,
    append_log: Option<PathBuf>,
}

impl Builder {
    pub fn from_hsd(hsd: &Hsd) -> Builder {
        Builder {
            hsd: hsd.clone(),
            initial_coords: None,
            elements: None,
            append_log: None,
        }
    }

    pub fn elements(&mut self, elements: &[Element]) -> &mut Self
    { self.elements = Some(elements.to_vec()); self }

    pub fn initial_coords(&mut self, coords: &Coords) -> &mut Self
    { self.initial_coords = Some(coords.clone()); self }

    pub fn append_log(&mut self, path: impl AsRef<Path>) -> &mut Self
    { self.append_log = Some(path.as_ref().to_owned()); self }

    pub fn build(&self) -> FailResult<DftbPlus> {
        DftbPlus::from_builder(self)
    }
}

#[cfg(unix)]
fn path_to_c_string(path: &Path) -> FailResult<std::ffi::CString> {
    Ok(std::ffi::CString::new(format!("{}", path.display()))?)
}

#[cfg(not(unix))]
fn path_to_c_string(path: &Path) -> FailResult<std::ffi::CString> {
    compile_error!("This crate only supports unix.");
    unreachable!()
}

impl DftbPlus {
    fn from_builder(builder: &Builder) -> FailResult<Self> {
        let Builder { hsd, elements, initial_coords, append_log } = builder;

        let elements = elements.as_ref().expect("Elements were not set on DftbPlus builder!");
        let initial_coords = initial_coords.as_ref().expect("Initial coords were not set on DftbPlus builder!");

        let outfile_name = {
            append_log.as_ref().map_or_else(|| PathBuf::from("/dev/null"), ToOwned::to_owned)
        };

        let tempdir = fsx::TempDir::new("rsp2")?;

        let primary_input_path = tempdir.path().join("dftb_in.hsd");

        {
            let mut file = fsx::create(&primary_input_path)?;
            write!(&mut file, r#"\
<<+ "{tmp}/user.hsd"
<<+ "{tmp}/generated.hsd"
"#, tmp=tempdir.path().display())?;
        }

        {
            let mut file = fsx::create(tempdir.path().join("user.hsd"))?;
            writeln!(&mut file, "{}", hsd)?;
        }

        {
            let mut file = fsx::create(tempdir.path().join("generated.hsd"))?;
            writeln!(&mut file, "{}", StructureHsd {
                coords: initial_coords,
                elements: elements,
                prefix: Some("!"),
            })?;
            writeln!(&mut file, "!Driver = {{ }}")?;
        }

        // we don't need to keep outfile_name around; DFTB+ copies it into its own buffer.
        let instance = unsafe {
            low_level::Instance::initialize(
                &primary_input_path,
                &outfile_name,
            )?
        };
        Ok(DftbPlus {
            instance,
            tempdir,
            elements: elements.to_vec(),
        })
    }


    /// The elements of each site.
    ///
    /// Once the `DftbPlus` instance is built, this can never be changed.
    pub fn elements(&self) -> &[Element] { &self.elements }

    pub fn set_coords(&mut self, coords: &Coords) -> FailResult<()> {
        self.instance.set_coords(coords)
    }

    pub fn compute_value(&mut self) -> FailResult<f64> {
        self.instance.get_energy()
    }

    pub fn compute_grad(&mut self) -> FailResult<Vec<V3>> {
        self.instance.get_grad()
    }
}


//---------------------------------------------------------------------------------------------

/// Has a Display impl that writes the HSD representation of a Coords.
///
/// Includes the leading `"Geometry = "` and is multiple lines long,
/// but does not include a trailing newline.
///
/// The prefix is applied to each tag, e.g. "*" or "!" to control parser behavior.
#[derive(Debug, Clone)]
struct StructureHsd<'a> {
    pub coords: &'a Coords,
    pub elements: &'a [Element],
    pub prefix: Option<&'a str>,
}

impl<'a> fmt::Display for StructureHsd<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use std::collections::BTreeSet;

        let StructureHsd { coords, elements, prefix } = self;
        let prefix = prefix.unwrap_or("");

        assert_eq!(coords.len(), elements.len());

        let unique_elems = {
            elements.iter().cloned()
                .collect::<BTreeSet<_>>().into_iter() // sorted unique
                .collect::<Vec<_>>() // fast linear scanning
        };

        writeln!(f, "{}Geometry = {{", prefix)?;
        writeln!(f, "  {}Periodic = Yes", prefix)?;

        write!(f, "  {}TypeNames = {{", prefix)?;
        for elem in &unique_elems {
            write!(f, " \"{}\"", elem.symbol())?;
        }
        write!(f, " }}")?;
        writeln!(f)?;

        writeln!(f, "  {}LatticeVectors [Angstrom] = {{", prefix)?;
        for &V3([x, y, z]) in coords.lattice().vectors() {
            writeln!(f, "    {:?} {:?} {:?}", x, y, z)?;
        }
        writeln!(f, "  }}")?;

        writeln!(f, "  {}TypesAndCoordinates [Angstrom] = {{", prefix)?;
        for (&elem, V3([x, y, z])) in elements.iter().zip(coords.to_carts()) {
            let elem_index = unique_elems.iter().position(|x| x == &elem).unwrap();

            writeln!(f, "    {} {:?} {:?} {:?}", elem_index + 1, x, y, z)?;
        }
        writeln!(f, "  }}")?;

        write!(f, "}}")?;
        Ok(())
    }
}
