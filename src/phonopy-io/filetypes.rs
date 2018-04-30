use ::FailResult;
use ::std::io::prelude::*;
use ::std::collections::HashMap;

use ::rsp2_structure::Structure;
use ::rsp2_array_types::{V3, M33};

// why is this pub(crate)? I don't remember...
pub(crate) type Displacements = Vec<(usize, V3)>;
pub mod disp_yaml {
    use super::*;

    mod cereal {
        use super::*;

        #[derive(Deserialize)]
        pub(super) struct DispYaml {
            pub displacements: Vec<Displacement>,
            pub lattice: M33,
            pub points: Vec<Point>,
        }

        #[derive(Deserialize)]
        pub(super) struct Displacement {
            pub atom: u32,
            pub direction: V3,
            pub displacement: V3,
        }

        #[derive(Deserialize)]
        pub(super) struct Point {
            pub symbol: String,
            pub coordinates: V3,
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

    // helper for generating displaced structures
    pub fn apply_displacement<M: Clone>(
        structure: &Structure<M>,
        (atom, disp): (usize, V3),
    ) -> Structure<M>
    {
        let mut structure = structure.clone();
        structure.carts_mut()[atom] += disp;
        structure
    }

    pub fn read<R: Read>(mut r: R) -> FailResult<DispYaml>
    { _read(&mut r) }

    // Monomorphic to ensure that all the yaml parsing code is generated inside this crate
    pub fn _read(r: &mut Read) -> FailResult<DispYaml>
    {
        use ::rsp2_structure::{CoordsKind, Lattice};
        use self::cereal::{Point, Displacement, DispYaml as RawDispYaml};

        let RawDispYaml { displacements, lattice, points } = ::serde_yaml::from_reader(r)?;

        let (carts, meta): (_, Vec<_>) =
            points.into_iter()
                .map(|Point { symbol, coordinates, mass }|
                    (coordinates, Meta { symbol, mass }))
                .unzip();

        let structure = Structure::new(Lattice::new(&lattice), CoordsKind::Fracs(carts), meta);

        let displacements =
            displacements.into_iter()
                // phonopy numbers from 1
                .map(|Displacement { atom, displacement, .. }| ((atom - 1) as usize, displacement))
                .collect();

        Ok(DispYaml { structure, displacements })
    }
}

/// Type representing a phonopy conf file.
///
/// In reality, valid conf files are only a subset of this.
/// For instance, I would be wary of inserting a value that contains
///   a `'#'` (the comment delimiter).
pub type Conf = HashMap<String, String>;
pub mod conf {
    use super::*;

    pub fn read<R: BufRead>(file: R) -> FailResult<Conf>
    {Ok({
        // NOTE: This was just thrown together based on what I assume
        //       the format of phonopy's `conf` files is.
        //       I haven't bothered to look at phonopy's own source for
        //       reading the files, nor have I looked to see if there
        //       is clear and unambiguous documentation somewhere for the spec.
        //         - ML
        let mut out = HashMap::new();
        for line in file.lines() {
            let mut line = &line?[..];

            if line.trim().is_empty() {
                continue;
            }

            if let Some(i) = line.bytes().position(|c| c == b'#') {
                line = &line[..i];
            }

            if let Some(i) = line.bytes().position(|c| c == b'=') {
                let key = line[..i].trim();
                let value = line[i + 1..].trim();
                out.insert(key.to_string(), value.to_string());
            } else {
                bail!("Can't read conf line: {:?}", line)
            }
        }
        out
    })}

    pub fn write<W: Write>(mut w: W, conf: &Conf) -> FailResult<()>
    {Ok({
        for (key, val) in conf {
            ensure!(key.bytes().all(|c| c != b'='), "'=' in conf key");
            writeln!(w, "{} = {}", key, val)?
        }
    })}
}

pub mod symmetry_yaml {
    use super::*;

    /// Spacegroup operator from disp.yaml
    #[derive(Deserialize)]
    pub struct Operation {
        pub rotation: M33<i32>,
        pub translation: V3,
    }

    /// A parsed --sym output
    #[derive(Deserialize)]
    pub struct SymmetryYaml {
        pub space_group_type: String,
        pub space_group_number: u32,
        pub point_group_type: String,
        pub space_group_operations: Vec<Operation>,
        #[serde(skip)]
        _more: (),
    }

    // NOTE: this is currently entirely unvalidated.
    pub fn read<R: Read>(mut r: R) -> FailResult<SymmetryYaml>
    { _read(&mut r) }

    // Monomorphic to ensure that all the yaml parsing code is generated inside this crate
    pub fn _read(r: &mut Read) -> FailResult<SymmetryYaml>
    {Ok({ ::serde_yaml::from_reader(r)? })}
}

pub mod force_sets {
    use super::*;
    // Adapted from code by Colin Daniels.

    /// Write a FORCE_SETS file.
    pub fn write<W, Vs>(
        mut w: W,
        displacements: &[(usize, V3)],
        force_sets: Vs,
    ) -> FailResult<()>
    where
        W: Write,
        Vs: IntoIterator,
        <Vs as IntoIterator>::IntoIter: ExactSizeIterator,
        <Vs as IntoIterator>::Item: AsRef<[V3]>,
    {
        let mut force_sets = force_sets.into_iter().peekable();

        assert_eq!(force_sets.len(), displacements.len());
        let n_atom = force_sets.peek().expect("no displacements!?").as_ref().len();

        writeln!(w, "{}", n_atom)?;
        writeln!(w, "{}", displacements.len())?;
        writeln!(w, "")?;

        for (&(atom, disp), force) in displacements.iter().zip(force_sets) {
            writeln!(w, "{}", atom + 1)?; // NOTE: phonopy indexes atoms from 1
            writeln!(w, "{:e} {:e} {:e}", disp[0], disp[1], disp[2])?;

            assert_eq!(force.as_ref().len(), n_atom);
            for row in force.as_ref() {
                writeln!(w, "{:e} {:e} {:e}", row[0], row[1], row[2])?;
            }

            // blank line for easier reading
            writeln!(w, "")?;
        }
        Ok(())
    }
}

pub mod sparse_sets {
    use super::*;

    #[derive(Serialize, Deserialize)]
    #[derive(Debug, Clone)]
    #[serde(rename_all = "kebab-case")]
    pub struct SparseSets {
        natom: usize,
        force_sets: Vec<Forces>,
    }

    #[derive(Serialize, Deserialize)]
    #[derive(Debug, Clone)]
    #[serde(rename_all = "kebab-case")]
    pub struct Forces {
        atom: usize,
        displacement: V3,
        partners: Vec<usize>,
        vectors: Vec<V3>,
    }

    /// Write a SPARSE_SETS file.
    pub fn write<W, Vs>(
        mut w: W,
        displacements: &[(usize, V3)],
        force_sets: Vs,
    ) -> FailResult<()>
        where
            W: Write,
            Vs: IntoIterator,
            <Vs as IntoIterator>::IntoIter: ExactSizeIterator,
            <Vs as IntoIterator>::Item: AsRef<[V3]>,
    {
        let mut force_sets = force_sets.into_iter().peekable();
        let n_atom = force_sets.peek().expect("no force sets!?").as_ref().len();

        assert_eq!(force_sets.len(), displacements.len());
        let force_sets =
            displacements.iter().zip(force_sets)
                .map(|(&(atom, displacement), force)| {
                    assert_eq!(force.as_ref().len(), n_atom);

                    // sparsify
                    let (partners, vectors) =
                        force.as_ref().iter().cloned().enumerate()
                            .filter(|&(_, x)| x != V3([0.0; 3]))
                            .map(|(i, x)| (i + 1, x)) // to 1-based
                            .unzip();

                    Forces {
                        atom: atom + 1, // to 1-based
                        displacement,
                        partners,
                        vectors,
                    }
                })
                .collect();

        _write(&mut w, &SparseSets {
            natom: n_atom,
            force_sets,
        })
    }

    fn _write(w: &mut Write, sparse_sets: &SparseSets) -> FailResult<()>
    { Ok(::serde_json::to_writer(w, sparse_sets)?) }
}
