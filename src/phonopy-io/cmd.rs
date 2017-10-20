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

            log_stdio_and_wait(command)?;
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
where P: AsRef<Path>
{ _phonopy_gamma_eigensystem(conf, force_sets, disp_dir.as_ref()) }

// monomorphic
#[inline(never)]
fn _phonopy_gamma_eigensystem(
    conf: &HashMap<String, String>,
    force_sets: Vec<Vec<[f64; 3]>>,
    disp_dir: &Path,
) -> Result<(Vec<f64>, Vec<Vec<[f64; 3]>>)>
{
    use ::slice_of_array::prelude::*;

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

        log_stdio_and_wait(command)?;
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
