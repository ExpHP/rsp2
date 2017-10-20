pub(crate) type Displacements = Vec<(usize, [f64; 3])>;
pub(crate) use disp_yaml::DispYaml;
pub mod disp_yaml {
    use ::Error;
    use ::Displacements;

    use ::std::io::prelude::*;
    use ::rsp2_structure::Structure;

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
        pub displacements: Displacements,
    }

    /// Atomic metadata from disp.yaml
    pub struct Meta {
        pub symbol: String,
        pub mass: f64,
    }

    pub fn read<R: Read>(r: R) -> Result<DispYaml, Error>
    {
        use ::rsp2_structure::{Coords, Lattice};
        use self::cereal::{Point, Displacement, DispYaml as RawDispYaml};

        let RawDispYaml { displacements, lattice, points } = ::serde_yaml::from_reader(r)?;

        let (carts, meta) =
            points.into_iter()
            .map(|Point { symbol, coordinates, mass }|
                (coordinates, Meta { symbol, mass }))
            .unzip();

        let structure = Structure::new(Lattice::new(&lattice), Coords::Fracs(carts), meta);

        let displacements =
            displacements.into_iter()
            // phonopy numbers from 1
            .map(|Displacement { atom, displacement, .. }| ((atom - 1) as usize, displacement))
            .collect();

        Ok(DispYaml { structure, displacements })
    }
}

pub mod force_sets {
    // Adapted from code by Colin Daniels.

    use ::Result;
    use ::std::result::Result as StdResult;

    use ::std::io::prelude::*;
    use ::rsp2_structure::{Structure, Coords};

    /// Given a function that computes gradient, automates the process
    /// of producing displaced structures and calling the function.
    pub fn compute_from_grad<M, E, F>(
        structure: Structure<M>,
        displacements: &[(usize, [f64; 3])],
        mut compute_grad: F,
    ) -> StdResult<Vec<Vec<[f64; 3]>>, E>
    where F: FnMut(&Structure<M>) -> StdResult<Vec<[f64; 3]>, E>,
    {
        use slice_of_array::prelude::*;

        self::compute(structure, displacements, |s| {
            let mut force = compute_grad(s)?;
            for x in force.flat_mut() {
                *x *= -1.0;
            }
            Ok(force)
        })
    }

    /// Given a function that computes forces, automates the process
    /// of producing displaced structures and calling the function.
    pub fn compute<M, E, F>(
        mut structure: Structure<M>,
        displacements: &[(usize, [f64; 3])],
        mut compute: F,
    ) -> StdResult<Vec<Vec<[f64; 3]>>, E>
    where F: FnMut(&Structure<M>) -> StdResult<Vec<[f64; 3]>, E>,
    {
        let orig_coords = structure.to_carts();

        displacements.iter().map(|&(atom, disp)| {
            let mut coords = orig_coords.clone();
            for k in 0..3 {
                coords[atom][k] += disp[k];
            }
            structure.set_coords(Coords::Carts(coords));

            let force = compute(&structure)?;
            assert_eq!(force.len(), structure.num_atoms());
            Ok(force)
        }).collect()
    }

    /// Write a FORCE_SETS file.
    pub fn write<M, W, V>(
        mut w: W,
        structure: &Structure<M>, // only used for natoms  _/o\_
        displacements: &[(usize, [f64; 3])],
        force_sets: &[V],
    ) -> Result<()>
    where
        W: Write,
        V: ::std::borrow::Borrow<[[f64; 3]]>,
    {
        assert_eq!(force_sets.len(), displacements.len());
        writeln!(w, "{}", structure.num_atoms())?;
        writeln!(w, "{}", displacements.len())?;
        writeln!(w, "")?;

        for (&(atom, disp), force) in displacements.iter().zip(force_sets) {
            writeln!(w, "{}", atom + 1)?; // NOTE: phonopy indexes atoms from 1
            writeln!(w, "{:e} {:e} {:e}", disp[0], disp[1], disp[2])?;

            assert_eq!(force.borrow().len(), structure.num_atoms());
            for row in force.borrow() {
                writeln!(w, "{:e} {:e} {:e}", row[0], row[1], row[2])?;
            }

            // blank line for easier reading
            writeln!(w, "")?;
        }
        Ok(())
    }
}
