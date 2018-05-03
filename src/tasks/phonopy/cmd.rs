//! Filesystem-based API.
//!
//! This API is centered around two types of objects:
//!
//! - Wrappers around directory types (which are generic in the
//!   directory type and may contain TempDirs, PathBufs, or even
//!   borrowed &Paths), representing a directory where some
//!   computation was run.
//! - Builders for configuring the next computation.
//!
//! The directory wrapper types allow the user to be able to
//! explicitly "keep" a phonopy output directory, and potentially
//! even reuse it in a future computation. The inability to do these
//! things was the biggest limitation of the previous API, which never
//! exposed its own temporary directories.

use ::FailResult;
use ::{IoResult};
use ::As3;

use super::{MissingFileError, PhonopyFailed};
use super::{Conf, DispYaml, SymmetryYaml, QPositions, Args};
use ::traits::{AsPath, HasTempDir, Save, Load};

use ::rsp2_structure_io::poscar;
use ::std::io::prelude::*;
use ::std::process::Command;
use ::std::path::{Path, PathBuf};
use ::rsp2_fs_util::{mv, TempDir};

use ::rsp2_kets::Basis;
use ::rsp2_fs_util::{open, create, open_text, copy, hard_link};
use ::rsp2_structure::{ElementStructure, Element};
use ::rsp2_structure::{FracRot, FracTrans, FracOp};
use ::rsp2_phonopy_io::npy;

use ::rsp2_array_types::{V3};

use ::slice_of_array::prelude::*;

const THZ_TO_WAVENUMBER: f64 = 33.35641;

#[derive(Debug, Clone)]
pub struct Builder {
    symprec: Option<f64>,
    conf: Conf,
}

impl Default for Builder {
    fn default() -> Self {
        Builder {
            symprec: None,
            conf: Default::default(),
        }
    }
}

const FNAME_OTHER_SETTINGS: &'static str = "rsp2-phonopy.json";
const FNAME_SETTINGS_ARGS: &'static str = "disp.args";
const FNAME_HELPER_SCRIPT: &'static str = "phonopy";
const FNAME_CONF_DISPS: &'static str = "disp.conf";
const FNAME_CONF_BANDS: &'static str = "band.conf";
#[allow(unused)]
const FNAME_CONF_SYMMETRY: &'static str = "phonopy.conf";
#[allow(unused)]
const FNAME_OUT_SYMMETRY: &'static str = "symmetry.yaml";

impl Builder {
    pub fn new() -> Self
    { Default::default() }

    pub fn symmetry_tolerance(mut self, x: f64) -> Self
    { self.symprec = Some(x); self }

    pub fn conf<K: AsRef<str>, V: AsRef<str>>(mut self, key: K, value: V) -> Self
    { self.conf.0.insert(key.as_ref().to_owned(), value.as_ref().to_owned()); self }

    /// Extend with configuration lines from a phonopy .conf file.
    /// If the file defines a value that was already set, the new
    ///  value from the file will take precedence.
    // FIXME: FailResult<Self>... oh, THAT's why the general recommendation is for &mut Self
    #[allow(unused)]
    pub fn conf_from_file<R: BufRead>(self, file: R) -> FailResult<Self>
    {Ok({
        let mut me = self;
        for (key, value) in ::rsp2_phonopy_io::conf::read(file)? {
            me = me.conf(key, value);
        }
        me
    })}

    pub fn supercell_dim<V: As3<u32>>(self, dim: V) -> Self
    {
        self.conf("DIM", {
            let (a, b, c) = dim.as_3();
            format!("{} {} {}", a, b, c)
        })
    }

    fn args_from_settings(&self) -> Args
    {
        let mut out = vec![];
        if let Some(tol) = self.symprec {
            out.push(format!("--tolerance"));
            out.push(format!("{:e}", tol));
        }
        out.into()
    }
}

impl Builder {

    fn finalize_config(&self, structure: &ElementStructure) -> Self
    {
        use ::itertools::Itertools;

        // FIXME: Mass hack
        self.clone()
            .conf(
                "MASS",
                structure.metadata().iter()
                    .map(|&s| ::common::element_mass(s))
                    .join(" ")
            )
    }

    pub fn displacements(
        &self,
        structure: &ElementStructure,
    ) -> FailResult<DirWithDisps<TempDir>>
    {
        self.finalize_config(structure)
            ._displacements(structure)
    }

    fn _displacements(
        &self,
        structure: &ElementStructure,
    ) -> FailResult<DirWithDisps<TempDir>>
    {Ok({
        let dir = TempDir::new("rsp2")?;
        {
            let dir = dir.path();
            trace!("Displacement dir: '{}'...", dir.display());

            let extra_args = self.args_from_settings();
            self.conf.save(dir.join(FNAME_CONF_DISPS))?;
            poscar::dump(create(dir.join("POSCAR"))?, "blah", &structure)?;
            extra_args.save(dir.join(FNAME_SETTINGS_ARGS))?;

            {
                use std::os::unix::fs::OpenOptionsExt;
                use ::failure::ResultExt;
                use ::util::ext_traits::PathNiceExt;
                let path = dir.join(FNAME_HELPER_SCRIPT);
                ::std::fs::OpenOptions::new()
                    .create(true)
                    .write(true)
                    .mode(0o777)
                    .open(&path)
                    .with_context(|e| format!("{}: error creating file: {}", path.nice(), e))?
                    .write_all(format!(r#"#!/bin/sh

# This script simulates rsp2's calls to phonopy by adding
# additional args that are not in any conf file.
#
# rsp2 itself does not use this file; it's for you!
phonopy \
    $(cat {}) \
    --fc-format=hdf5 \
    --readfc \
    "$@"
"#, FNAME_SETTINGS_ARGS).as_bytes())?;
            }

            trace!("Calling phonopy for displacements...");
            {
                let mut command = Command::new("phonopy");
                command
                    .args(&extra_args.0)
                    .arg(FNAME_CONF_DISPS)
                    .arg("--displacement")
                    .current_dir(&dir);

                log_stdio_and_wait(command, None)?;
            }
        };

        DirWithDisps::from_existing(dir)?
    })}

    // FIXME: Should return a new DirWithSymmetry type.
    pub fn symmetry(
        &self,
        structure: &ElementStructure,
    ) -> FailResult<(Vec<FracOp>)>
    {
        self.finalize_config(structure)
            ._symmetry(structure)
    }

    fn _symmetry(
        &self,
        structure: &ElementStructure,
    ) -> FailResult<(Vec<FracOp>)>
    {Ok({
        let tmp = TempDir::new("rsp2")?;
        let tmp = tmp.path();
        trace!("Entered '{}'...", tmp.display());

        self.conf.save(tmp.join(FNAME_CONF_SYMMETRY))?;

        poscar::dump(create(tmp.join("POSCAR"))?, "blah", &structure)?;

        trace!("Calling phonopy for symmetry...");
        check_status(Command::new("phonopy")
            .args(self.args_from_settings().0)
            .arg(FNAME_CONF_SYMMETRY)
            .arg("--sym")
            .current_dir(&tmp)
            .stdout(create(tmp.join(FNAME_OUT_SYMMETRY))?)
            .status()?)?;

        trace!("Done calling phonopy");

        // check if input structure was primitive
        {
            let prim = poscar::load(open(tmp.join("PPOSCAR"))?)?;

            let ratio = structure.lattice().volume() / prim.lattice().volume();
            let ratio = round_checked(ratio, 1e-4)?;

            // sorry, supercells are just not supported... yet.
            //
            // (In the future we may be able to instead return an object
            //  which will allow the spacegroup operators of the primitive
            //  to be applied in meaningful ways to the superstructure.)
            ensure!(ratio == 1, "attempted to compute symmetry of a supercell");
        }

        let yaml = SymmetryYaml::load(tmp.join(FNAME_OUT_SYMMETRY))?;
        yaml.space_group_operations.into_iter()
            .map(|op| Ok({
                let rotation = FracRot::new(&op.rotation);
                let translation = FracTrans::from_floats(&op.translation)?;
                FracOp::new(&rotation, &translation)
            }))
            .collect::<FailResult<_>>()?
    })}
}

/// Represents a directory with the following data:
/// - `POSCAR`: The input structure
/// - `disp.yaml`: Phonopy file with displacements
/// - configuration settings which impact the selection of displacements
///   - `--tol`, `--dim`
///
/// Generally, the next step is to supply the force sets, turning this
/// into a DirWithForces.
///
/// # Note
///
/// Currently, the implementation is rather optimistic that files in
/// the directory have not been tampered with since its creation.
/// As a result, some circumstances which probably should return `Error`
/// may instead cause a panic, or may not be detected as early as possible.
#[derive(Debug, Clone)]
pub struct DirWithDisps<P: AsPath> {
    pub(crate) dir: P,
    // These are cached in memory from `disp.yaml` due to the likelihood
    // that code using `DirWithDisps` will need them.
    pub(crate) superstructure: ElementStructure,
    pub(crate) displacements: Vec<(usize, V3)>,
}

impl<P: AsPath> DirWithDisps<P> {
    pub fn from_existing(dir: P) -> FailResult<Self>
    {Ok({
        for name in &[
            "POSCAR",
            "disp.yaml",
            FNAME_CONF_DISPS,
            FNAME_SETTINGS_ARGS,
            FNAME_OTHER_SETTINGS,
        ] {
            let path = dir.as_path().join(name);
            if !path.exists() {
                throw!(MissingFileError::new("DirWithDisps", &dir, name.to_string()));
            }
        }

        trace!("Parsing disp.yaml...");
        let DispYaml {
            displacements, structure: superstructure
        } = Load::load(dir.as_path().join("disp.yaml"))?;
        let superstructure = superstructure.try_map_metadata_into(|d| {
            match Element::from_symbol(&d.symbol[..]) {
                Some(e) => Ok(e),
                None => bail!("invalid symbol in disp.yaml: {:?}", d.symbol),
            }
        })?;

        DirWithDisps { dir, superstructure, displacements }
    })}

    #[allow(unused)]
    pub fn superstructure(&self) -> &ElementStructure
    { &self.superstructure }
    pub fn displacements(&self) -> &[(usize, V3)]
    { &self.displacements }

    /// Due to the "StreamingIterator problem" this iterator must return
    /// clones of the structure... though it seems unlikely that this cost
    /// is anything to worry about compared to the cost of computing the
    /// forces on said structure.
    pub fn displaced_structures<'a>(&'a self) -> Box<Iterator<Item=ElementStructure> + 'a>
    { Box::new({
        use ::rsp2_phonopy_io::disp_yaml::apply_displacement;
        self.displacements
            .iter()
            .map(move |&disp| apply_displacement(&self.superstructure, disp))
    })}

    /// Write FORCE_SETS to create a `DirWithForces`
    /// (which may serve as a template for one or more band computations).
    ///
    /// This variant creates a new temporary directory.
    pub fn make_force_dir<Vs>(self, forces: Vs) -> FailResult<DirWithForces<TempDir>>
    where
        Vs: IntoIterator,
        <Vs as IntoIterator>::IntoIter: ExactSizeIterator,
        <Vs as IntoIterator>::Item: AsRef<[V3]>,
    {Ok({
        let out = TempDir::new("rsp2")?;
        trace!("Force sets dir: '{}'...", out.path().display());

        self.make_force_dir_in_dir(forces, out)?
    })}

    /// Write FORCE_SETS to create a `DirWithForces`
    /// (which may serve as a template for one or more band computations).
    ///
    /// This variant uses the specified path.
    pub fn make_force_dir_in_dir<Vs, Q>(self, forces: Vs, path: Q) -> FailResult<DirWithForces<Q>>
    where
        Q: AsPath,
        Vs: IntoIterator,
        <Vs as IntoIterator>::IntoIter: ExactSizeIterator,
        <Vs as IntoIterator>::Item: AsRef<[V3]>,
    {Ok({
        self.prepare_force_dir(forces, path.as_path())?;
        DirWithForces::from_existing(path)?
    })}

    /// Creates the files expected by DirWithForces::from_existing
    fn prepare_force_dir<Vs>(self, forces: Vs, path: &Path) -> FailResult<()>
    where
        Vs: IntoIterator,
        <Vs as IntoIterator>::IntoIter: ExactSizeIterator,
        <Vs as IntoIterator>::Item: AsRef<[V3]>,
    {Ok({
        let disp_dir = self.path();

        for name in &[
            "POSCAR",
            "disp.yaml",
            FNAME_CONF_DISPS,
            FNAME_SETTINGS_ARGS,
            FNAME_OTHER_SETTINGS,
        ] {
            copy(disp_dir.join(name), path.join(name))?;
        }
        let _ = copy(disp_dir.join(FNAME_HELPER_SCRIPT), path.join(FNAME_HELPER_SCRIPT));

        trace!("Writing FORCE_SETS...");
        ::rsp2_phonopy_io::force_sets::write(
            create(path.join("FORCE_SETS"))?,
            &self.displacements,
            forces,
        )?;
    })}
}

/// Represents a directory with the following data:
/// - `POSCAR`: The input structure
/// - `FORCE_SETS`: Phonopy file with forces for displacements
/// - configuration settings which impact the selection of displacements
///   - `--tol`, `--dim`
/// - OtherSettings
///
/// It may also contain a force constants file.
///
/// Generally, one produces a `DirWithBands` from this by selecting some
/// q-points to sample.
///
/// # Note
///
/// Currently, the implementation is rather optimistic that files in
/// the directory have not been tampered with since its creation.
/// As a result, some circumstances which probably should return `Error`
/// may instead cause a panic, or may not be detected as early as possible.
#[derive(Debug, Clone)]
pub struct DirWithForces<P: AsPath> {
    dir: P,
    cache_force_constants: bool,
}

impl<P: AsPath> DirWithForces<P> {
    pub fn from_existing(dir: P) -> FailResult<Self>
    {Ok({
        // Sanity check
        for name in &[
            "POSCAR",
            "FORCE_SETS",
            FNAME_SETTINGS_ARGS,
            FNAME_CONF_DISPS,
        ] {
            let path = dir.as_path().join(name);
            if !path.exists() {
                throw!(MissingFileError::new("DirWithForces", &dir, name.to_string()));
            }
        }
        DirWithForces { dir, cache_force_constants: true }
    })}

    #[allow(unused)]
    pub fn structure(&self) -> FailResult<ElementStructure>
    { Ok(poscar::load(open_text(self.path().join("POSCAR"))?)?) }

    /// Enable/disable caching of force constants.
    ///
    /// When enabled (the default), force constants are copied back from
    /// the first successful `DirWithBands` created, and are reused in
    /// subsequent band computations. (these copies are hard links if possible)
    #[allow(unused)]
    pub fn cache_force_constants(&mut self, b: bool) -> &mut Self
    { self.cache_force_constants = b; self }

    /// Compute bands in a temp directory.
    ///
    /// Returns an object used to configure the computation.
    pub fn build_bands(&self) -> BandsBuilder<P>
    { BandsBuilder::init(self) }
}

// NOTE: I'm not sure if you should be allowed to clone this.
//       (correctness depends on how the build function uses the filesystem)
//
//       Dang filesystem; it's like global variables all over again.
#[derive(Debug)]
pub struct BandsBuilder<'moveck, 'p, P: AsPath + 'p> {
    dir_with_forces: &'p DirWithForces<P>,
    eigenvectors: bool,
    // Part of a trick to simulate "moving" a value with a `&mut self` function
    _move: ::std::marker::PhantomData<&'moveck ()>,
}

impl<'moveck, 'p, P: AsPath> BandsBuilder<'moveck, 'p, P> {
    fn init(dir_with_forces: &'p DirWithForces<P>) -> Self
    { BandsBuilder {
        dir_with_forces,
        eigenvectors: false,
        _move: Default::default(),
    }}

    pub fn eigenvectors(&mut self, b: bool) -> &mut Self
    { self.eigenvectors = b; self }

    /// Consume the builder, run phonopy, and produce a DirWithBands.
    ///
    /// This uses a special trick to simulate a move of the builder (statically preventing
    /// you from using it again) with a receiver of type `&mut self`. Basically, by associating
    /// the `&mut self` borrow with a lifetime parameter of the struct itself, the duration of
    /// the borrow is extended to cover the entire rest of the DirWithBands' existence.
    pub fn compute(&'moveck mut self, q_points: &[V3]) -> FailResult<DirWithBands<TempDir>>
    {Ok({
        let dir = TempDir::new("rsp2")?;

        // scope to temporarily shadow `dir` with a &Path, which is easier to work with.
        {
            let src = self.dir_with_forces.as_path();
            let dir = dir.as_path();

            let fc_filename = "force_constants.hdf5";

            for name in &[
                "POSCAR",
                FNAME_CONF_DISPS,
                FNAME_SETTINGS_ARGS,
                FNAME_OTHER_SETTINGS,
            ] {
                copy(src.join(name), dir.join(name))?;
            }

            for name in &[
                "FORCE_SETS",
            ] {
                copy_or_link(src.join(name), dir.join(name))?;
            }

            if src.join(fc_filename).exists() {
                copy_or_link(src.join(fc_filename), dir.join(fc_filename))?;
            }
            let _ = copy(src.join(FNAME_HELPER_SCRIPT), dir.join(FNAME_HELPER_SCRIPT));

            QPositions(q_points.into()).save(dir.join("q-positions.json"))?;

            // band.conf
            {
                // Carry over settings from displacements.
                let Conf(mut conf) = Load::load(dir.join(FNAME_CONF_DISPS))?;

                // Append a dummy qpoint so that each of our points begin a line segment.
                conf.insert("BAND".to_string(), band_string(q_points) + " 0 0 0");
                // NOTE: using 1 band point does result in a (currently inconsequential) divide-by-zero
                //       warning in phonopy, but cuts the filesize of band output by half compared to 2
                //       points. Considering how large the band output is, I'll take my chances!
                //          - ML
                conf.insert("BAND_POINTS".to_string(), "1".to_string());
                conf.insert("EIGENVECTORS".to_string(), match self.eigenvectors {
                    true => ".TRUE.".to_string(),
                    false => ".FALSE.".to_string(),
                });
                Conf(conf).save(dir.join(FNAME_CONF_BANDS))?;
            }

            trace!("Calling phonopy for bands...");
            {
                let mut command = Command::new("phonopy");
                command
                    .args(Args::load(dir.join(FNAME_SETTINGS_ARGS))?.0)
                    .arg(FNAME_CONF_BANDS)
                    .arg("--fc-format=hdf5")
                    .arg("--band-format=hdf5")
                    .arg(match dir.join(fc_filename).exists() {
                        true => "--readfc",
                        false => "--writefc",
                    })
                    .current_dir(&dir);

                log_stdio_and_wait(command, None)?;
            }

            trace!("Converting bands...");
            {
                let mut command = Command::new("python3");
                command.current_dir(&dir);

                // ayyyyup.
                log_stdio_and_wait(command, Some("
import numpy as np
import h5py

band = h5py.File('band.hdf5')
np.save('eigenvector.npy', band['eigenvector'])
np.save('eigenvalue.npy', band['frequency'])
np.save('q-position.npy', band['path'])
np.save('q-distance.npy', band['distance'])

del band
import os; os.unlink('band.hdf5')
".to_string()))?;
            }

            if self.dir_with_forces.cache_force_constants {
                cache_link(dir.join(fc_filename), src.join(fc_filename))?;
            }
        } // end of scope that borrows dir

        DirWithBands::from_existing(dir)?
    })}
}

/// Represents a directory with the following data:
/// - input structure
/// - q-points
/// - eigenvalues (and possibly eigenvectors)
/// - OtherSettings
///
/// # Note
///
/// Currently, the implementation is rather optimistic that files in
/// the directory have not been tampered with since its creation.
/// As a result, some circumstances which probably should return `Error`
/// may instead cause a panic, or may not be detected as early as possible.
#[derive(Debug, Clone)]
pub struct DirWithBands<P: AsPath> {
    dir: P,
}


impl<P: AsPath> DirWithBands<P> {
    pub fn from_existing(dir: P) -> FailResult<Self>
    {Ok({
        // Sanity check
        for name in &[
            "POSCAR",
            "FORCE_SETS",
            "eigenvalue.npy",
            "q-positions.json",
        ] {
            let path = dir.as_path().join(name);
            if !path.exists() {
                throw!(MissingFileError::new("DirWithBands", &dir, name.to_string()))
            }
        }

        DirWithBands { dir }
    })}

    pub fn structure(&self) -> FailResult<ElementStructure>
    { Ok(poscar::load(open_text(self.path().join("POSCAR"))?)?) }

    pub fn q_positions(&self) -> FailResult<Vec<V3>>
    { Ok(QPositions::load(self.path().join("q-positions.json"))?.0) }

    /// This will be `None` if `.eigenvectors(true)` was not set prior
    /// to the band computation.
    pub fn eigenvectors(&self) -> FailResult<Option<Vec<Basis>>>
    {Ok({
        let path = self.path().join("eigenvector.npy");
        if path.exists() {
            trace!("Reading eigenvectors...");
            Some(npy::read_eigenvector_npy(open(path)?)?)
        } else { None }
    })}

    pub fn eigenvalues(&self) -> FailResult<Vec<Vec<f64>>>
    {Ok({
        use ::rsp2_slice_math::{v};
        trace!("Reading eigenvectors...");
        let file = open(self.path().join("eigenvalue.npy"))?;
        npy::read_eigenvalue_npy(file)?
            .into_iter()
            .map(|evs| (v(evs) * THZ_TO_WAVENUMBER).0)
            .collect()
    })}
}

//-----------------------------

fn band_string(ks: &[V3]) -> String
{ ks.flat().iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" ") }

#[allow(unused)]
fn round_checked(x: f64, tol: f64) -> FailResult<i32>
{Ok({
    let r = x.round();
    ensure!((r - x).abs() < tol, "not nearly integral: {}", x);
    r as i32
})}

pub(crate) fn log_stdio_and_wait(
    mut cmd: ::std::process::Command,
    stdin: Option<String>,
) -> FailResult<()>
{Ok({
    use ::std::process::Stdio;
    use ::std::io::{BufRead, BufReader};

    if stdin.is_some() {
        cmd.stdin(Stdio::piped());
    }

    debug!("$ {:?}", cmd);

    let mut child = cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    if let Some(text) = stdin {
        child.stdin.take().unwrap().write_all(text.as_bytes())?;
    }

    let stdout_worker = {
        let f = BufReader::new(child.stdout.take().unwrap());
        ::std::thread::spawn(move || -> FailResult<()> {Ok({
            for line in f.lines() {
                ::stdout::log(&(line?[..]));
            }
        })})
    };

    let stderr_worker = {
        let f = BufReader::new(child.stderr.take().unwrap());
        ::std::thread::spawn(move || -> FailResult<()> {Ok({
            for line in f.lines() {
                ::stderr::log(&(line?[..]));
            }
        })})
    };

    check_status(child.wait()?)?;

    let _ = stdout_worker.join();
    let _ = stderr_worker.join();
})}

fn check_status(status: ::std::process::ExitStatus) -> Result<(), PhonopyFailed>
{
    if status.success() { Ok(()) }
    else {
        let backtrace = ::failure::Backtrace::new();
        Err(PhonopyFailed { backtrace, status })
    }
}

// Wrapper around `hard_link` which:
// - falls back to copying if the destination is on another filesystem.
// - does not fail if the destination exists (it is simply left behind)
//
// The use case is where this may be used on many identical source files
// for the same destination, possibly concurrently, in order to cache the
// file for further reuse.
pub fn cache_link<P: AsRef<Path>, Q: AsRef<Path>>(src: P, dest: Q) -> FailResult<()>
{
    let (src, dest) = (src.as_ref(), dest.as_ref());
    hard_link(src, dest)
        .map(|_| ())
        .or_else(|_| Ok({
            // if the file already existed, the link will have failed.
            // Check this before continuing because we don't want to
            //   potentially overwrite a link with a copy.
            if dest.exists() {
                return Ok(());
            }

            // assume the error was due to being on a different filesystem.
            // (Even if not, we will probably just get the same error)
            copy(src, dest)?;
        }))
}

// Like `cache_link` except it fails if the destination exists.
pub fn copy_or_link<P: AsRef<Path>, Q: AsRef<Path>>(src: P, dest: Q) -> FailResult<()>
{
    let (src, dest) = (src.as_ref(), dest.as_ref());
    hard_link(src, dest)
        .map(|_| ())
        .or_else(|_| Ok({
            // assume that, if the error was due to anything other than cross-device linking,
            // then we'll get the same error again when we try to copy.
            copy(src, dest)?;
        }))
}

//-----------------------------

impl_dirlike_boilerplate!{
    type: {DirWithDisps<_>}
    member: self.dir
    other_members: [self.displacements, self.superstructure]
}

impl_dirlike_boilerplate!{
    type: {DirWithForces<_>}
    member: self.dir
    other_members: [self.cache_force_constants]
}

impl_dirlike_boilerplate!{
    type: {DirWithBands<_>}
    member: self.dir
    other_members: []
}

#[cfg(test)]
#[deny(unused)]
mod tests {
    use super::*;

    // check use cases I need to work
    #[allow(unused)]
    fn things_expected_to_impl_aspath() {
        use ::std::rc::Rc;
        use ::std::sync::Arc;
        panic!("This is a compiletest; it should not be executed.");

        // clonable TempDirs
        let x: DirWithBands<TempDir> = panic!();
        let x = x.map_dir(Rc::new);
        let _ = x.clone();

        // sharable TempDirs
        let x: DirWithBands<TempDir> = panic!();
        let x = x.map_dir(Arc::new);
        let _: &(Send + Sync) = &x;

        // erased types, for conditional deletion
        let _: DirWithBands<Box<AsPath>> = x.map_dir(|e| Box::new(e) as _);
    }
}
