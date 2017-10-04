extern crate sp2_structure;

#[macro_use] extern crate serde_derive;
extern crate serde_yaml;

pub type IoError = ::std::io::Error;
pub type YamlError = ::serde_yaml::Error;

use ::std::any::Any; // ew
use ::std::fmt::Debug;

#[derive(Debug)]
pub enum Error {
    IoError(IoError),
    ComputeError(Box<Any + Send>),
    YamlError(YamlError),

    #[doc(hidden)]
    NoTotalMatchPlease,
}

impl From<IoError> for Error {
    fn from(e: IoError) -> Self { Error::IoError(e) }
}

impl From<YamlError> for Error {
    fn from(e: YamlError) -> Self { Error::YamlError(e) }
}

// newtype for coherent From instances
struct ComputeError<E>(E);
impl<E: Send + 'static> From<ComputeError<E>> for Error {
    fn from(e: ComputeError<E>) -> Self { Error::ComputeError(Box::new(e.0)) }
}

mod disp_yaml {
    use ::Error;

    use ::std::io::prelude::*;
    use ::sp2_structure::Structure;

    mod cereal {
        #[derive(Deserialize)]
        pub(super) struct DispYaml {
            pub displacements: Vec<Displacement>,
            pub lattice: [[f64; 3]; 3],
            pub points: Vec<Point>,
        }

        #[derive(Deserialize)]
        pub(super) struct Displacement {
            pub atom: u32,
            pub direction: [f64; 3],
            pub displacement: [f64; 3],
        }

        #[derive(Deserialize)]
        pub(super) struct Point {
            pub symbol: String,
            pub coordinates: [f64; 3],
            pub mass: f64,
        }
    }

    /// A parsed disp.yaml
    pub struct DispYaml {
        pub structure: Structure<Meta>,
        pub displacements: Vec<(usize, [f64; 3])>
    }

    /// Atomic metadata from disp.yaml
    pub struct Meta {
        pub symbol: String,
        pub mass: f64,
    }

    pub fn read_disp_yaml<R: Read>(r: R) -> Result<DispYaml, Error>
    {
        use ::sp2_structure::{Coords, Lattice};
        use self::cereal::{Point, Displacement, DispYaml as RawDispYaml};

        let RawDispYaml { displacements, lattice, points } = ::serde_yaml::from_reader(r)?;

        let (carts, meta) =
            points.into_iter()
            .map(|Point { symbol, coordinates, mass }| (coordinates, Meta { symbol, mass }))
            .unzip();

        let structure = Structure::new(Lattice::new(lattice), Coords::Carts(carts), meta);

        let displacements =
            displacements.into_iter()
            // phonopy numbers from 1
            .map(|Displacement { atom, displacement, .. }| ((atom - 1) as usize, displacement))
            .collect();

        Ok(DispYaml { structure, displacements })
    }
}

mod force_constants {
    use ::{Error, ComputeError};

    use ::std::io::prelude::*;
    use ::sp2_structure::{Structure, Coords};

    /// Write a FORCE_CONSTANTS file.
    ///
    /// Adapted from code by Colin Daniels.
    pub fn write_from_fn<M, E: Send + 'static, W, F>(
        mut w: W,
        mut structure: Structure<M>,
        displacements: &[(usize, [f64; 3])],
        mut compute: F,
    ) -> Result<(), Error>
    where
        W: Write,
        F: FnMut(&Structure<M>) -> Result<Vec<[f64; 3]>, E>,
    {
        writeln!(&mut w, "{}", structure.num_atoms())?;
        writeln!(&mut w, "{}", displacements.len())?;
        writeln!(&mut w, "")?;

        let orig_coords = structure.to_carts();

        for &(atom, disp) in displacements {
            writeln!(&mut w, "{}", atom + 1)?; // NOTE: phonopy indexes atoms from 1
            writeln!(&mut w, "{:e} {:e} {:e}", disp[0], disp[1], disp[2])?;

            let mut coords = orig_coords.clone();
            for k in 0..3 {
                coords[atom][k] += disp[k];
            }

            structure.set_coords(Coords::Carts(coords));
            let force = compute(&structure).map_err(ComputeError)?;

            assert_eq!(force.len(), structure.num_atoms());
            for row in force {
                writeln!(&mut w, "{:e} {:e} {:e}", row[0], row[1], row[2])?;
            }

            // blank line for easier reading
            writeln!(&mut w, "")?;
        }
        Ok(())
    }
}
