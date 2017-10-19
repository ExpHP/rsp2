extern crate rsp2_kets;
extern crate rsp2_structure;
extern crate rsp2_structure_io;
extern crate rsp2_byte_tools_plus_float as byte_tools;
extern crate slice_of_array;

#[macro_use] extern crate error_chain;
#[macro_use] extern crate nom;
#[macro_use] extern crate log;
#[macro_use] extern crate serde_derive;
extern crate serde_yaml;
extern crate rsp2_tempdir as tempdir;

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

pub mod cmd {
    use ::Result;
    use ::Displacements;
    use ::DispYaml;

    use ::rsp2_structure::CoordStructure;

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

    pub fn phonopy_displacements_carbon(
        conf: &HashMap<String, String>,
        structure: CoordStructure,
    ) -> Result<(CoordStructure, Displacements, TempDir)>
    {
        use ::rsp2_structure_io::poscar;

        let tmp = TempDir::new("rsp2-rs")?;
        let (displacements, superstructure) = {

            let tmp = tmp.path();
            trace!("Entered '{}'...", tmp.display());

            write_conf(
                File::create(tmp.join("phonopy.conf"))?,
                &conf,
            )?;

            poscar::dump_carbon(
                File::create(tmp.join("POSCAR"))?,
                "blah",
                &structure,
            )?;

            trace!("Calling phonopy for displacements...");
            {
                let mut command = Command::new("phonopy");
                command
                    .arg("--displacement")
                    .arg("phonopy.conf")
                    .current_dir(&tmp);

                ::log_stdio_and_wait(command)?;
            }

            trace!("Parsing disp.yaml...");
            let DispYaml {
                displacements, structure: superstructure
            } = ::disp_yaml::read(File::open(tmp.join("disp.yaml"))?)?;

            (displacements, superstructure)
        };

        Ok((superstructure.map_metadata(|_| ()), displacements, tmp))
    }

    pub fn phonopy_gamma_eigensystem<P>(
        conf: &HashMap<String, String>,
        force_sets: Vec<Vec<[f64; 3]>>,
        disp_dir: &P,
    ) -> Result<(Vec<f64>, Vec<Vec<[f64; 3]>>)>
    where P: AsRef<Path>,
    {
        use ::slice_of_array::prelude::*;

        let disp_dir = disp_dir.as_ref();

        let tmp = TempDir::new("rsp2-rs")?;
        let tmp = tmp.path();
        trace!("Entered '{}'...", tmp.display());

        let mut conf = conf.clone();
        conf.insert("BAND".to_string(), "0 0 0 1 0 0".to_string());
        conf.insert("BAND_POINTS".to_string(), "2".to_string());
        write_conf(File::create(tmp.join("phonopy.conf"))?, &conf)?;

        fs::copy(disp_dir.join("POSCAR"), tmp.join("POSCAR"))?;

        trace!("Parsing disp.yaml...");
        let DispYaml {
            displacements, structure: superstructure,
        } = ::disp_yaml::read(File::open(disp_dir.join("disp.yaml"))?)?;

        trace!("Writing FORCE_SETS...");
        ::force_sets::write(
            File::create(tmp.join("FORCE_SETS"))?,
            &superstructure,
            &displacements,
            &force_sets,
        )?;

        trace!("Calling phonopy for eigenvectors...");
        {
            let mut command = Command::new("phonopy");
            command
                .env("EIGENVECTOR_NPY_HACK", "1")
                .arg("--eigenvectors")
                .arg("phonopy.conf")
                .current_dir(&tmp);

            ::log_stdio_and_wait(command)?;
        }

        trace!("Reading eigenvectors...");
        let bases = ::npy::read_eigenvector_npy(File::open(tmp.join("eigenvector.npy"))?)?;
        trace!("Reading eigenvalues...");
        let freqs = ::npy::read_eigenvalue_npy(File::open(tmp.join("eigenvalue.npy"))?)?;

        // eigensystem at first kpoint (gamma)
        let basis = bases.into_iter().next().unwrap();
        let freqs = freqs.into_iter().next().unwrap();

        trace!("Getting real..."); // :P
        let evecs = basis.iter().map(|ev| Ok(
            ev.iter().map(|c| {
                // gamma kets are real
                ensure!(c.imag == 0.0, "non-real eigenvector");
                Ok(c.real)
            }).collect::<Result<Vec<_>>>()?.nest().to_vec()
        )).collect::<Result<_>>()?;
        trace!("Done computing eigensystem");
        Ok((freqs, evecs))
    }
}

fn log_stdio_and_wait(mut cmd: ::std::process::Command) -> Result<()>
{Ok({
    use ::std::process::Stdio;
    use ::std::io::{BufRead, BufReader};

    let mut child = cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    let stdout_worker = {
        let f = BufReader::new(child.stdout.take().unwrap());
        ::std::thread::spawn(move || -> Result<()> {Ok({
            for line in f.lines() {
                ::stdout::log(&(line?[..]));
            }
        })})
    };

    let stderr_worker = {
        let f = BufReader::new(child.stderr.take().unwrap());
        ::std::thread::spawn(move || -> Result<()> {Ok({
            for line in f.lines() {
                ::stderr::log(&(line?[..]));
            }
        })})
    };

    ensure!(child.wait()?.success(), "Phonopy failed.");

    let _ = stdout_worker.join();
    let _ = stderr_worker.join();
})}

/// This module only exists to have its name appear in logs.
/// It marks phonopy's stdout.
mod stdout {
    pub fn log(s: &str)
    { info!("{}", s) }
}

/// This module only exists to have its name appear in logs.
/// It marks phonopy's stderr.
mod stderr {
    pub fn log(s: &str)
    { warn!("{}", s) }
}
