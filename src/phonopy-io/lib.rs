extern crate sp2_kets;
extern crate sp2_structure;
extern crate sp2_structure_io;
extern crate sp2_slice_of_array;
extern crate sp2_byte_tools_plus_float as byte_tools;

#[macro_use] extern crate error_chain;
#[macro_use] extern crate nom;
#[macro_use] extern crate serde_derive;
extern crate serde_yaml;
extern crate tempdir;

pub type IoError = ::std::io::Error;
pub type YamlError = ::serde_yaml::Error;
pub type Shareable = Send + Sync + 'static;

mod npy;

error_chain!{
    foreign_links {
        Io(::std::io::Error);
        Yaml(::serde_yaml::Error);
    }

    errors {
        PhonopyExitCode(code: u32) {
            description("phonopy exited unsuccessfully"),
            display("phonopy exited unsuccessfully ({})", code),
        }
    }
}

pub(crate) type Displacements = Vec<(usize, [f64; 3])>;
pub(crate) use disp_yaml::DispYaml;
mod disp_yaml {
    use ::Error;
    use ::Displacements;

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
        pub displacements: Displacements,
    }

    /// Atomic metadata from disp.yaml
    pub struct Meta {
        pub symbol: String,
        pub mass: f64,
    }

    pub fn read<R: Read>(r: R) -> Result<DispYaml, Error>
    {
        use ::sp2_structure::{Coords, Lattice};
        use self::cereal::{Point, Displacement, DispYaml as RawDispYaml};

        let RawDispYaml { displacements, lattice, points } = ::serde_yaml::from_reader(r)?;

        let (carts, meta) =
            points.into_iter()
            .map(|Point { symbol, coordinates, mass }|
                (coordinates, Meta { symbol, mass }))
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
    use ::Result;

    use ::std::io::prelude::*;
    use ::sp2_structure::{Structure, Coords};

    /// Write a FORCE_CONSTANTS file.
    ///
    /// Adapted from code by Colin Daniels.
    pub fn write_from_fn<M, W, V>(
        mut w: W,
        mut structure: Structure<M>,
        displacements: &[(usize, [f64; 3])],
        mut forces: &[V],
    ) -> Result<()>
    where
        W: Write,
        V: ::std::borrow::Borrow<[[f64; 3]]>,
    {
        assert_eq!(forces.len(), displacements.len());
        writeln!(w, "{}", structure.num_atoms())?;
        writeln!(w, "{}", displacements.len())?;
        writeln!(w, "")?;

        let orig_coords = structure.to_carts();

        for (&(atom, disp), force) in displacements.iter().zip(forces) {
            writeln!(w, "{}", atom + 1)?; // NOTE: phonopy indexes atoms from 1
            writeln!(w, "{:e} {:e} {:e}", disp[0], disp[1], disp[2])?;

            let mut coords = orig_coords.clone();
            for k in 0..3 {
                coords[atom][k] += disp[k];
            }

            structure.set_coords(Coords::Carts(coords));

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

mod cmd {
    use ::Result;
    use ::Displacements;
    use ::DispYaml;

    use ::sp2_structure::CoordStructure;

    use ::tempdir::TempDir;
    use ::std::process::Command;
    use ::std::io::prelude::*;
    use ::std::fs;
    use ::std::fs::File;
    use ::std::path::Path;
    use ::std::collections::HashMap;

    fn write_conf<W>(mut w: W, conf: &HashMap<String, String>) -> Result<()>
    where W: Write,
    {
        for (key, val) in conf {
            ensure!(key.bytes().all(|c| c != b'='), "'=' in conf key");
            writeln!(w, "{} = {}", key, val)?
        }
        Ok(())
    }

    fn phonopy_displacements_carbon(
        conf: &HashMap<String, String>,
        structure: CoordStructure,
    ) -> Result<(Displacements, TempDir)>
    {
        use ::sp2_structure_io::poscar;

        let tmp = TempDir::new("sp2-rs")?;
        let displacements = {

            let tmp = tmp.path();

            write_conf(
                File::create(tmp.join("phonopy.conf"))?,
                &conf,
            )?;

            poscar::dump_carbon(
                File::create(tmp.join("POSCAR"))?,
                "blah",
                &structure,
            )?;

            if let Some(code) =
                Command::new("phonopy")
                .arg("--displacement")
                .arg("phonopy.conf")
                .current_dir(&tmp)
                .status()?.code()
            {
                bail!("Phonopy exited with code {}", code);
            }

            let DispYaml {
                displacements, ..
            } = ::disp_yaml::read(File::open(tmp.join("disp.yaml"))?)?;

            displacements
        };

        Ok((displacements, tmp))
    }

    fn phonopy_eigenvectors<P>(
        conf: &HashMap<String, String>,
        disp_dir: &P,
    ) -> Result<Vec<Vec<[f64; 3]>>>
    where P: AsRef<Path>,
    {
        use ::sp2_slice_of_array::prelude::*;

        let disp_dir = disp_dir.as_ref();

        let tmp = TempDir::new("sp2-rs")?;
        let tmp = tmp.path();

        let mut conf = conf.clone();
        conf.insert("BAND".to_string(), "0 0 0 1 0 0".to_string());
        conf.insert("BAND_POINTS".to_string(), "2".to_string());
        write_conf(File::create(tmp.join("phonopy.conf"))?, &conf)?;

        fs::copy(disp_dir.join("POSCAR"), tmp.join("POSCAR"))?;

        if let Some(code) =
            Command::new("phonopy")
            .env("NPY_EIGENVECTOR_HACK", "1")
            .arg("--eigenvectors")
            .arg("phonopy.conf")
            .current_dir(&tmp)
            .status()?.code()
        {
            bail!("Phonopy exited with code {}", code);
        }

        let bases = ::npy::read(File::open("eigenvector.npy")?)?;
        let basis = bases.into_iter().next().unwrap();
        let out = basis.iter().map(|ev| Ok(
            ev.iter().map(|c| {
                ensure!(c.imag == 0.0, "non-real eigenvector");
                Ok(c.real)
            }).collect::<Result<Vec<_>>>()?.nest().to_vec()
        )).collect::<Result<_>>();
        out
    }
}
