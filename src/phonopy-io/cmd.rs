use ::errors::*;
use ::Displacements;
use ::DispYaml;
use ::As3;

use ::rsp2_structure::{CoordStructure, ElementStructure};
use ::rsp2_structure::{FracRot, FracTrans, FracOp};
use ::rsp2_kets::Basis;

use ::tempdir::TempDir;
use ::std::process::Command;
use ::std::io::prelude::*;
use ::std::fs;
use ::std::fs::File;
use ::std::path::Path;
use ::std::collections::HashMap;
use ::slice_of_array::prelude::*;

fn write_conf<W>(mut w: W, conf: &HashMap<String, String>) -> Result<()>
where W: Write,
{
    for (key, val) in conf {
        ensure!(key.bytes().all(|c| c != b'='), "'=' in conf key");
        writeln!(w, "{} = {}", key, val)?
    }
    Ok(())
}

#[derive(Debug, Clone, Default)]
pub struct Builder {
    symprec: Option<f64>,
    conf: HashMap<String, String>,
}


impl Builder {
    pub fn new() -> Self
    { Default::default() }

    pub fn symmetry_tolerance(mut self, x: f64) -> Self
    {
        self.symprec = Some(x);
        self
    }

    pub fn conf<K: AsRef<str>, V: AsRef<str>>(mut self, key: K, value: V) -> Self
    {
        self.conf.insert(key.as_ref().to_owned(), value.as_ref().to_owned());
        self
    }

    pub fn supercell_dim<V: As3<u32>>(self, dim: V) -> Self
    {
        self.conf("DIM", {
            let (a, b, c) = dim.as_3();
            format!("{} {} {}", a, b, c)
        })
    }

    fn args_from_settings(&self) -> Vec<String>
    {
        let mut out = vec![];
        if let Some(tol) = self.symprec {
            out.push(format!("--tolerance"));
            out.push(format!("{:e}", tol));
        }
        out
    }
}

impl Builder {
    // FIXME computation functions and builder setters in the same namespace
    //       with the same naming convention. this is weird/confusing
    pub fn displacements(
        &self,
        structure: ElementStructure,
    ) -> Result<(CoordStructure, Displacements, TempDir)>
    {
        use ::rsp2_structure_io::poscar;

        let tmp = TempDir::new("rsp2")?;
        let (displacements, superstructure) = {

            let tmp = tmp.path();
            trace!("Entered '{}'...", tmp.display());

            write_conf(File::create(tmp.join("phonopy.conf"))?, &self.conf)?;

            poscar::dump(File::create(tmp.join("POSCAR"))?, "blah", &structure)?;

            trace!("Calling phonopy for displacements...");
            {
                let mut command = Command::new("phonopy");
                command
                    .args(self.args_from_settings())
                    .arg("phonopy.conf")
                    .arg("--displacement")
                    .current_dir(&tmp);

                log_stdio_and_wait(command)?;
            }

            trace!("Parsing disp.yaml...");
            let DispYaml {
                displacements, structure: superstructure
            } = ::disp_yaml::read(File::open(tmp.join("disp.yaml"))?)?;

            (displacements, superstructure)
        };

        Ok((superstructure.map_metadata_into(|_| ()), displacements, tmp))
    }

    pub fn gamma_eigensystem<I>(
        self: &Self,
        force_sets: I,
        disp_dir: &AsRef<Path>,
    ) -> Result<(Vec<f64>, Vec<Vec<[f64; 3]>>)>
    where
        I: IntoIterator,
        I::Item: AsRef<[[f64; 3]]>,
    {Ok({
        let (freqs, basis) =
            self.eigensystems(force_sets, &[[0.0, 0.0, 0.0]], disp_dir)?
                .into_iter().next().unwrap();

        trace!("Getting real..."); // :P
        let evecs = basis.iter().map(|ev| Ok(
            ev.iter().map(|c| {
                ensure!(c.imag == 0.0, "non-real gamma eigenvector");
                Ok(c.real)
            }).collect::<Result<Vec<_>>>()?.nest().to_vec()
        )).collect::<Result<_>>()?;
        trace!("Done computing eigensystem at gamma");
        (freqs, evecs)
    })}

    pub fn eigensystems<I>(
        self: &Self,
        force_sets: I, // e.g. Vec<Vec<[f64; 3]>> or &[&[[f64; 3]]]
        q_points: &[[f64; 3]],
        disp_dir: &AsRef<Path>,
    ) -> Result<Vec<(Vec<f64>, Basis)>>
    where
        I: IntoIterator,
        I::Item: AsRef<[[f64; 3]]>,
    {
        let force_sets: Vec<_> = force_sets.into_iter().collect();
        let force_sets: Vec<_> = force_sets.iter().map(|x| x.as_ref()).collect();

        fn monomorphic_iife(
            me: &Builder,
            force_sets: &[&[[f64; 3]]],
            q_points: &[[f64; 3]],
            disp_dir: &AsRef<Path>,
        ) -> Result<Vec<(Vec<f64>, Basis)>>
        {Ok({
            let disp_dir = disp_dir.as_ref();

            let tmp = TempDir::new("rsp2")?;
            let tmp = tmp.path();
            trace!("Entered '{}'...", tmp.display());

            let mut me = me.clone();
            // Append a dummy qpoint so that each of our points begin a line segment.
            me = me.conf("BAND", band_string(q_points) + " 0 0 0");
            me = me.conf("BAND_POINTS", "2");
            write_conf(File::create(tmp.join("phonopy.conf"))?, &me.conf)?;

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
                    .args(me.args_from_settings())
                    .arg("phonopy.conf")
                    .arg("--eigenvectors")
                    .env("EIGENVECTOR_NPY_HACK", "1")
                    .current_dir(&tmp);

                log_stdio_and_wait(command)?;
            }

            trace!("Reading eigenvectors...");
            let bases = ::npy::read_eigenvector_npy(File::open(tmp.join("eigenvector.npy"))?)?;
            trace!("Reading eigenvalues...");
            let freqs = ::npy::read_eigenvalue_npy(File::open(tmp.join("eigenvalue.npy"))?)?;
            trace!("Done computing eigensystems");

            // extract the beginning of each line segment
            fn every_other<I: Iterator>(mut it: I) -> Vec<I::Item>
            {
                let mut out = vec![it.next().expect("no kpoints?!")];
                while let Some(x) = it.nth(2 - 1) {
                    out.push(x);
                }
                out
            }

            every_other(freqs.into_iter().zip(bases))
        })}
        monomorphic_iife(self, &force_sets[..], q_points, disp_dir)
    }

    pub fn symmetry(
        &self,
        structure: &ElementStructure,
    ) -> Result<(Vec<FracOp>)>
    {Ok({
        use ::rsp2_structure_io::poscar;
        use ::filetypes::symmetry_yaml;

        let tmp = TempDir::new("rsp2")?;
        let tmp = tmp.path();
        trace!("Entered '{}'...", tmp.display());

        write_conf(File::create(tmp.join("phonopy.conf"))?, &self.conf)?;

        poscar::dump(File::create(tmp.join("POSCAR"))?, "blah", &structure)?;

        trace!("Calling phonopy for symmetry...");
        check_status(Command::new("phonopy")
            .args(self.args_from_settings())
            .arg("phonopy.conf")
            .arg("--sym")
            .current_dir(&tmp)
            .stdout(File::create(tmp.join("symmetry.yaml"))?)
            .status()?)?;

        trace!("Done calling phonopy");

        // check if input structure was primitive
        {
            let prim = poscar::load(File::open(tmp.join("PPOSCAR"))?)?;

            let ratio = structure.lattice().volume() / prim.lattice().volume();
            let ratio = round_checked(ratio, 1e-4)?;

            // sorry, supercells are just not supported... yet.
            //
            // (In the future we may be able to instead return an object
            //  which will allow the spacegroup operators of the primitive
            //  to be applied in meaningful ways to the superstructure.)
            ensure!(ratio == 1, ErrorKind::NonPrimitiveStructure);
        }

        let yaml = symmetry_yaml::read(File::open(tmp.join("symmetry.yaml"))?)?;
        yaml.space_group_operations.into_iter()
            .map(|op| Ok({
                let rotation = FracRot::new(&op.rotation);
                let translation = FracTrans::from_floats(&op.translation)?;
                FracOp::new(&rotation, &translation)
            }))
            .collect::<Result<_>>()?
    })}
}

fn round_checked(x: f64, tol: f64) -> Result<i32>
{Ok({
    let r = x.round();
    ensure!((r - x).abs() < tol, "not nearly integral: {}", x);
    r as i32
})}

fn check_status(status: ::std::process::ExitStatus) -> Result<()>
{Ok({
    ensure!(status.success(), ErrorKind::PhonopyFailed(status));
})}

fn log_stdio_and_wait(mut cmd: ::std::process::Command) -> Result<()>
{Ok({
    use ::std::process::Stdio;
    use ::std::io::{BufRead, BufReader};

    debug!("$ {:?}", cmd);

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

    check_status(child.wait()?)?;

    let _ = stdout_worker.join();
    let _ = stderr_worker.join();
})}

fn band_string(ks: &[[f64; 3]]) -> String
{
    ks.flat().iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" ")
}
