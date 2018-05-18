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

    pub fn read(mut r: impl Read) -> FailResult<DispYaml>
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

    pub fn read(file: impl BufRead) -> FailResult<Conf>
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

    pub fn write(mut w: impl Write, conf: &Conf) -> FailResult<()>
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
    pub fn read(mut r: impl Read) -> FailResult<SymmetryYaml>
    { _read(&mut r) }

    // Monomorphic to ensure that all the yaml parsing code is generated inside this crate
    pub fn _read(r: &mut Read) -> FailResult<SymmetryYaml>
    {Ok({ ::serde_yaml::from_reader(r)? })}
}

pub mod force_sets {
    use super::*;

    pub struct ForceSets {
        pub displacements: Vec<(usize, V3)>,
        pub force_sets: Vec<Vec<V3>>,
    }

    /// Write a FORCE_SETS file.
    pub fn write<Vs>(
        mut w: impl Write,
        displacements: &[(usize, V3)],
        force_sets: Vs,
    ) -> FailResult<()>
    where
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

        for (&(atom, V3([dx, dy, dz])), force) in displacements.iter().zip(force_sets) {
            writeln!(w, "{}", atom + 1)?; // NOTE: phonopy indexes atoms from 1
            writeln!(w, "{:e} {:e} {:e}", dx, dy, dz)?;

            assert_eq!(force.as_ref().len(), n_atom);
            for V3([fx, fy, fz]) in force.as_ref() {
                writeln!(w, "{:e} {:e} {:e}", fx, fy, fz)?;
            }

            // blank line for easier reading
            writeln!(w, "")?;
        }
        Ok(())
    }

    /// Read a FORCE_SETS file.
    pub fn read(w: impl BufRead) -> FailResult<ForceSets> {
        // Blank lines are ignored.
        let mut lines = w.lines().filter(|x| match x {
            Ok(s) => s.trim() != "",
            _ => true,
        });
        let mut next_line = move |expected: &str| -> FailResult<_> {
            match lines.next() {
                None => bail!("Expected {}, got EOL", expected),
                Some(Err(e)) => Err(e)?,
                Some(Ok(line)) => Ok(line),
            }
        };

        let n_atom: usize = next_line("atom count line")?.trim().parse()?;
        let n_disp: usize = next_line("disp count line")?.trim().parse()?;

        let parse_v3 = |s: &str| -> FailResult<V3> {
            let v = s.split_whitespace().map(|s| Ok(s.parse()?)).collect::<FailResult<Vec<f64>>>()?;
            match &v[..] {
                &[x, y, z] => Ok(V3([x, y, z])),
                _ => bail!("expected line of 3 floats, got {:?}", s),
            }
        };

        let mut displacements = Vec::with_capacity(n_disp);
        let mut force_sets: Vec<_> = (0..n_disp).map(|_| Vec::with_capacity(n_atom)).collect();
        for force_set in &mut force_sets {
            let displaced = next_line("displaced atom line")?.trim().parse::<usize>()? - 1;
            let displacement = parse_v3(&next_line("displacement vector line")?)?;
            displacements.push((displaced, displacement));
            for _ in 0..n_atom {
                force_set.push(parse_v3(&next_line("force line")?)?);
            }
        }
        Ok(ForceSets { displacements, force_sets })
    }

    #[test]
    fn it_can_read_what_it_writes() {
        let displacements = vec![
            (0, V3([1.0, 0.0, 0.0])),
            (2, V3([0.0, 1.0, 0.0])),
        ];
        let forces = vec![vec![V3([0.0, 0.2, 0.3]); 4]; 2];

        let mut buf = vec![];
        write(&mut buf, &displacements, &forces).unwrap();
        let force_sets = read(::std::io::BufReader::new(&buf[..])).unwrap();

        assert_eq!(displacements, force_sets.displacements);
        assert_eq!(forces, force_sets.force_sets);
    }
}


