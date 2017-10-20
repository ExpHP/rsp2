pub(crate) type Displacements = Vec<(usize, [f64; 3])>;
pub(crate) use self::disp_yaml::DispYaml;
pub mod disp_yaml {
    use ::Error;
    use super::Displacements;

    use ::std::io::prelude::*;
    use ::rsp2_structure::{Structure, Coords};

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

    pub fn displaced_structures<'a, M: Clone + 'a>(
        mut structure: Structure<M>,
        displacements: &'a [(usize, [f64; 3])],
    ) -> Box<Iterator<Item=Structure<M>> + 'a>
    {
        let orig_coords = structure.to_carts();

        Box::new(displacements.iter().map(move |&(atom, disp)| {
            let mut coords = orig_coords.clone();
            for k in 0..3 {
                coords[atom][k] += disp[k];
            }
            structure.set_coords(Coords::Carts(coords));
            structure.clone()
        }))
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

pub mod symmetry_yaml {
    use ::Result;

    use ::std::io::prelude::*;

    mod cereal {
        #[derive(Deserialize)]
        pub(super) struct SymmetryYaml {
            pub space_group_type: String,
            pub space_group_number: u32,
            pub point_group_type: String,
            pub space_group_operations: Vec<Operation>,
        }

        #[derive(Deserialize)]
        pub(super) struct Operation {
            pub rotation: [[i16; 3]; 3],
            pub translation: [f64; 3],
        }
    }

    /// A parsed --sym output
    pub struct SymmetryYaml {
        pub space_group_type: String,
        pub space_group_number: u32,
        pub point_group_type: String,
        pub space_group_operations: Vec<Operation>,
        _more: (),
    }

    /// Spacegroup operator from disp.yaml
    pub struct Operation {
        pub rotation: [[i16; 3]; 3],
        pub translation_twelfths: [i16; 3],
    }

    pub fn read<R: Read>(r: R) -> Result<SymmetryYaml>
    {Ok({
        parse(::serde_yaml::from_reader(r)?)?
    })}

    // monomorphic
    fn parse(yaml: cereal::SymmetryYaml) -> Result<SymmetryYaml>
    {Ok({
        use self::cereal::SymmetryYaml as RawYaml;
        use self::cereal::Operation as RawOperation;

        let RawYaml {
            space_group_type,
            space_group_number,
            point_group_type,
            space_group_operations,
        } = yaml;

        let space_group_operations =
            space_group_operations.into_iter().map(|op| {Ok({
                let RawOperation { rotation, translation } = op;

                let translation_twelfths = {
                    let t = translation;
                    let f = |mut x: f64| -> Result<i16> {Ok({
                        x *= 12.0;
                        // generous tolerance due to low precision in file
                        ensure!((x - x.round()).abs() < 1e-4,
                            "translation not divisible by 12: {:?}", translation);

                        x.round() as i16
                    })};

                    [f(t[0])?, f(t[1])?, f(t[2])?]
                };

                Operation { rotation, translation_twelfths }
             })}).collect::<Result<_>>()?;

        SymmetryYaml {
            space_group_type,
            space_group_number,
            point_group_type,
            space_group_operations,
            _more: (),
        }
    })}
}

pub mod force_sets {
    // Adapted from code by Colin Daniels.

    use ::Result;

    use ::std::io::prelude::*;
    use ::rsp2_structure::Structure;

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
