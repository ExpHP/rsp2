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

use ::{FailOk, FailResult};
use ::{IoResult};

use super::{MissingFileError, PhonopyFailed};
use super::{Conf, DispYaml, SymmetryYaml, QPositions, Args, ForceSets};
use ::traits::{AsPath, HasTempDir, Save, Load};
use ::math::basis::Basis3;
use ::meta::{Mass, Element};
use ::meta::prelude::*;
use ::hlist_aliases::*;

use ::std::io::prelude::*;
use ::std::rc::Rc;
use ::std::process::Command;
use ::std::path::{Path, PathBuf};
use ::rsp2_fs_util::{TempDir};

use ::rsp2_fs_util::{open, create, copy, hard_link};
use ::rsp2_structure::{Coords, CartOp};
use ::rsp2_structure::supercell::{SupercellToken};
use ::rsp2_soa_ops::{Permute, Perm};

use ::rsp2_structure_io::Poscar;
use ::rsp2_phonopy_io::npy;

use ::rsp2_array_types::{V3, Unvee};

use ::slice_of_array::prelude::*;
use ::itertools::Itertools;
use ::failure::ResultExt;

const THZ_TO_WAVENUMBER: f64 = 33.35641;

/// An object passed around in the highest-level code (`::cmd`) to record
/// various bits of configuration for phonopy.  Information is added when
/// convenient, and methods are provided to call phonopy.
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

// filenames invented by rsp2
//
// (we don't bother with constants for fixed filenames used by phonopy, like "POSCAR")
const FNAME_SETTINGS_ARGS: &'static str = "disp.args";
const FNAME_HELPER_SCRIPT: &'static str = "phonopy";
const FNAME_CONF_DISPS: &'static str = "disp.conf";
const FNAME_CONF_BANDS: &'static str = "band.conf";
const FNAME_OUT_SYMMETRY: &'static str = "symmetry.yaml";

impl Builder {
    pub fn new() -> Self
    { Default::default() }

    pub fn symmetry_tolerance(mut self, x: f64) -> Self
    { self.symprec = Some(x); self }

    pub fn conf(mut self, key: impl AsRef<str>, value: impl AsRef<str>) -> Self
    { self.conf.0.insert(key.as_ref().to_owned(), value.as_ref().to_owned()); self }

    /// Extend with configuration lines from a phonopy .conf file.
    /// If the file defines a value that was already set, the new
    ///  value from the file will take precedence.
    // FIXME: FailResult<Self>... oh, THAT's why the general recommendation is for &mut Self
    #[allow(unused)]
    pub fn conf_from_file(self, file: impl BufRead) -> FailResult<Self>
    {Ok({
        let mut me = self;
        for (key, value) in ::rsp2_phonopy_io::conf::read(file)? {
            me = me.conf(key, value);
        }
        me
    })}

    pub fn supercell_dim(self, dim: [u32; 3]) -> Self
    { self.conf("DIM", dim.iter().join(" ")) }

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
    /// Make last-second adjustments to the config that are only possible once
    /// the structure and metadata are known.
    fn finalize_config(&self, meta: HList1<Rc<[Mass]>>) -> Self
    {
        let masses: Rc<[Mass]> = meta.pick();
        self.clone().conf("MASS", masses.iter().join(" "))
    }

    pub fn displacements(
        &self,
        coords: &Coords,
        meta: HList2<
            Rc<[Element]>,
            Rc<[Mass]>,
        >,
    ) -> FailResult<DirWithDisps<TempDir>>
    {
        self.finalize_config(meta.sift())
            ._displacements(coords, meta.sift())
    }

    // this pattern of having a second impl method is to simulate rebinding
    // the output of `finalize_config` to `self`. (otherwise, we'd be forced
    // to have a `self: &Self` in scope with the incorrect config, which would
    // be a massive footgun)
    fn _displacements(
        &self,
        coords: &Coords,
        meta: HList2<
            Rc<[Element]>,
            Rc<[Mass]>,
        >,
    ) -> FailResult<DirWithDisps<TempDir>>
    {Ok({
        let elements: Rc<[Element]> = meta.pick();
        let dir = TempDir::new("rsp2")?;
        let dir = dir.try_with_recovery(|dir| FailOk({
            let dir = dir.path();
            trace!("Displacement dir: '{}'...", dir.display());

            let extra_args = self.args_from_settings();
            self.conf.save(dir.join(FNAME_CONF_DISPS))?;
            Poscar {
                comment: "blah", coords, elements,
            }.save(dir.join("POSCAR"))?;
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

            {
                trace!("Calling phonopy for displacements...");
                let mut command = Command::new("phonopy");
                command
                    .args(&extra_args.0)
                    .arg(FNAME_CONF_DISPS)
                    .arg("--displacement")
                    .current_dir(&dir);

                log_stdio_and_wait(command, None)?;
            }

            {
                trace!("Producing {}...", FNAME_OUT_SYMMETRY);
                let mut command = Command::new("phonopy");
                command
                    .args(&extra_args.0)
                    .arg(FNAME_CONF_DISPS)
                    .arg("--sym")
                    .current_dir(&dir)
                    .stdout(create(dir.join(FNAME_OUT_SYMMETRY))?);

                check_status(command.status()?)?;

                //---------------------------
                // NOTE: Even though integer-based FracTrans is gone, this limitation is
                //       still necessary because we're only capable of hashing the rotations
                //       when using GroupTree. (meaning they must be unique, meaning there
                //       must be no pure translations)
                //       (though maybe it would be better to check for pure translations *there*,
                //        rather than checking PPOSCAR *here*)
                //---------------------------
                //
                // check if input structure was primitive
                let Poscar { coords: prim, .. } = Poscar::load(dir.join("PPOSCAR"))?;

                let ratio = coords.lattice().volume() / prim.lattice().volume();
                let ratio = round_checked(ratio, 1e-4)?;

                ensure!(ratio == 1, "attempted to compute symmetry of a supercell");
            }
        }))?.0; // let dir = dir.try_with_recovery(...)

        DirWithDisps::from_existing(dir)?
    })}
}

/// Represents a directory with the following data:
/// - `POSCAR`: The input structure
/// - `disp.yaml`: Phonopy file with displacements
/// - configuration settings which impact the selection of displacements
///   - `--tol`, `--dim`
/// - `symmetry.yaml`: Output of `phonopy --sym`, currently present
///   only for validation purposes. (we use spglib)
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
    pub(crate) displacements: Vec<(usize, V3)>,
    // These are cached in memory from `disp.yaml` due to the likelihood
    // that code using `DirWithDisps` will need them.
    pub(crate) super_coords: Coords,
    pub(crate) super_meta: HList2<Rc<[Element]>, Rc<[Mass]>>,
}

impl<P: AsPath> DirWithDisps<P> {
    pub fn from_existing(dir: P) -> FailResult<Self>
    {Ok({
        for name in &[
            "POSCAR",
            "disp.yaml",
            FNAME_CONF_DISPS,
            FNAME_SETTINGS_ARGS,
            FNAME_OUT_SYMMETRY,
        ] {
            let path = dir.as_path().join(name);
            if !path.exists() {
                throw!(MissingFileError::new("DirWithDisps", &dir, name.to_string()));
            }
        }

        trace!("Parsing disp.yaml...");
        let DispYaml {
            displacements, coords, elements, masses,
        } = Load::load(dir.as_path().join("disp.yaml"))?;
        let elements: Rc<[_]> = elements.into();
        let masses: Rc<[_]> = masses.into_iter().map(Mass).collect::<Vec<_>>().into();
        let meta = hlist![elements, masses];

        DirWithDisps {
            dir,
            displacements,
            super_coords: coords,
            super_meta: meta.sift(),
        }
    })}

    /// The structure you provided as input.
    pub fn primitive_structure(&self) -> FailResult<(Coords, HList2<Rc<[Element]>, Rc<[Mass]>>)>
    { read_input_structure_with_mass(self) }

    /// Get the structure from `disp.yaml`.
    ///
    /// # Note
    ///
    /// This superstructure was generated by phonopy, and the atoms may be in
    /// a different order than most supercells in rsp2 (those produced with SupercellToken).
    #[allow(unused)]
    pub fn superstructure(&self) -> (&Coords, HList2<Rc<[Element]>, Rc<[Mass]>>)
    { (&self.super_coords, self.super_meta.clone()) }

    /// Get displacements.  *The atom indices are for phonopy's supercell!*
    pub fn displacements(&self) -> &[(usize, V3)]
    { &self.displacements }

    /// Get coordinate sets modified with displacements from `disp.yaml`.
    ///
    /// # Note
    ///
    /// The base structure will have been produced by phonopy.  The ordering of sites
    /// may differ from most supercells in rsp2 (those produced with SupercellToken).
    /// The corresponding metadata can be obtained from `self.superstructure()`.
    //
    // Due to the "StreamingIterator problem" this iterator must return
    // clones of the structure... though it seems unlikely that this cost
    // is anything to worry about compared to the cost of computing the
    // forces on said structure.
    pub fn displaced_coord_sets<'a>(&'a self) -> impl Iterator<Item=Coords> + 'a
    { Box::new({
        use ::rsp2_phonopy_io::disp_yaml::apply_displacement;
        self.displacements
            .iter()
            .map(move |&disp| apply_displacement(&self.super_coords, disp))
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

    /// Gets the spacegroup operators.
    pub fn symmetry(&self) -> FailResult<Vec<CartOp>>
    {
        use cmd::python::SpgDataset;

        // phonopy prints symmetry.yaml at too little precision.
        // Use spglib.
        let (coords, meta) = self.primitive_structure()?;

        // (this deliberately does not accomodate things such as differing masses
        //  for the same element ("false masses"), because phonopy does not.)
        let elements: Rc<[Element]> = meta.pick();
        let types: Vec<_> = elements.iter().map(|e| e.atomic_number()).collect();

        let symprec = self._symmetry_precision()?;
        let dataset = SpgDataset::compute(&coords, &types, symprec)?;

        // This is... yeah.
        let phonopy_sg = self._phonopy_sg_number()?;
        ensure!(
            dataset.space_group_number == phonopy_sg,
            "(BUG in rsp2) spglib found a different spacegroup than phonopy! \
            This must mean rsp2 supplied the wrong input.");

        unimplemented!("get float sg ops")
    }

    fn _symmetry_precision(&self) -> FailResult<f64>
    {Ok({
        // FIXME (I never intended for rsp2 to be parsing this file
        //        to recover configuration settings!)
        let args = Args::load(self.join(FNAME_SETTINGS_ARGS))?;
        ensure!(
            &args.0[0] == "--tolerance", "unexpected in {}: {:?}",
            FNAME_SETTINGS_ARGS,
            args.0[0],
        );
        args.0[1].parse::<f64>()
            .with_context(|e| format!("could not parse tolerance: {}", e))?
    })}

    // Although we ultimately use `spglib` (since it gives fuller precision for
    // the translations), the intent is still to get the spacegroup used *by phonopy*
    // (as otherwise we might end up with e.g. underdetermined force constants)
    //
    // So we call `phonopy --sym` for the sole purpose of validating that the spacegroup
    // returned is the same. This could fail if our method of assigning integer atom types
    // differed from phonopy (e.g. are masses checked?).
    fn _phonopy_sg_number(&self) -> FailResult<u32>
    { Ok(SymmetryYaml::load(self.join(FNAME_OUT_SYMMETRY))?.space_group_number) }
}

/// Like Gramma used to make.
pub struct Rsp2StyleDisplacements {
    /// A supercell following rsp2's conventions (not phonopy's)
    pub super_coords: Coords,
    pub super_meta: HList2<Rc<[Element]>, Rc<[Mass]>>,
    /// Describes the relationship between the input structure and `superstructure`.
    pub sc: SupercellToken,
    /// Permutation that rearranges phonopy's superstructure to match `superstructure`.
    pub perm_from_phonopy: Perm,

    /// Displacements that use indices into the primitive structure.
    ///
    /// You are free to just use this field and ignore the rest (which merely come
    /// for "free" with it). This field should be compatible with superstructures
    /// of any size, and obviously does not depend on the convention for ordering
    /// sites in a supercell.
    pub prim_displacements: Vec<(usize, V3)>,
}

impl<P: AsPath> DirWithDisps<P> {
    /// Produce a variety of data describing the displacements in terms of rsp2's conventions
    /// (whereas most other methods on `DirWithDisps` use phonopy's conventions).
    pub fn rsp2_style_displacements(&self) -> FailResult<Rsp2StyleDisplacements> {
        let (prim_coords, prim_meta) = self.primitive_structure()?;
        let sc_dims = {
            let conf = Conf::load(self.path().join(FNAME_CONF_DISPS))?;
            get_sc_dim(&conf)?.ok_or_else(|| format_err!("DIM is required"))?
        };

        let (our_super_coords, sc) = ::rsp2_structure::supercell::diagonal(sc_dims).build(&prim_coords);
        let our_super_meta = prim_meta.map(hlist![
            |x: Rc<[_]>| -> Rc<[_]> { sc.replicate(&x[..]).into() },
            |x: Rc<[_]>| -> Rc<[_]> { sc.replicate(&x[..]).into() },
        ]);

        // cmon, big money, big money....
        // if these assertions always succeed, it will save us a
        // good deal of implementation work.
        let perm_from_phonopy;
        {
            let (phonopy_super_coords, phonopy_super_meta) = self.superstructure();

            perm_from_phonopy = phonopy_super_coords.perm_to_match(&our_super_coords, 1e-10)?;

            // make phonopy match us
            let phonopy_super_coords = phonopy_super_coords.clone().permuted_by(&perm_from_phonopy);
            let phonopy_super_meta = phonopy_super_meta.permuted_by(&perm_from_phonopy);

            let err_msg = "\
                phonopy's superstructure does not match rsp2's conventions! \
                Unfortunately, support for this scenario is not yet implemented.\
            ";
            assert_eq!(our_super_meta, phonopy_super_meta, "{}", err_msg);
            assert_close!(
                abs=1e-10,
                our_super_coords.lattice(), phonopy_super_coords.lattice(),
                "{}", err_msg,
            );
            let lattice = our_super_coords.lattice();
            let diffs = {
                zip_eq!(our_super_coords.to_carts(), phonopy_super_coords.to_carts())
                    .map(|(a, b)| (a - b) / lattice)
                    .map(|v| v.map(|x| x - x.round()))
                    .map(|v| v * lattice)
                    .collect::<Vec<_>>()
            };
            assert_close!(
                abs=1e-10,
                vec![[0.0; 3]; diffs.len()],
                diffs.unvee(),
                "{}", err_msg,
            );
        }

        let prim_displacements = {
            let primitive_atoms = sc.atom_primitive_atoms();
            self.displacements().iter()
                .map(|&(phonopy_idx, disp)| {
                    let our_super_idx = perm_from_phonopy.permute_index(phonopy_idx);
                    let our_prim_idx = primitive_atoms[our_super_idx];
                    (our_prim_idx, disp)
                })
                .collect::<Vec<_>>()
        };

        Ok(Rsp2StyleDisplacements {
            super_coords: our_super_coords,
            super_meta: our_super_meta,
            sc,
            prim_displacements,
            perm_from_phonopy,
        })
    }
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
    pub fn structure(&self) -> FailResult<(Coords, HList2<Rc<[Element]>, Rc<[Mass]>>)>
    { read_input_structure_with_mass(self) }

    /// Read FORCE_SETS.
    ///
    /// # Note
    ///
    /// The displaced atom indices will be zero based; however, they will follow
    /// phonopy's conventions for ordering the supercell.
    pub fn force_sets(&self) -> FailResult<ForceSets>
    { Ok(ForceSets::load(self.path().join("FORCE_SETS"))?) }

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

        let dir = dir.try_with_recovery(|dir| FailOk({
            let src = self.dir_with_forces.as_path();
            let dir = dir.as_path();

            let fc_filename = "force_constants.hdf5";

            for name in &[
                "POSCAR",
                FNAME_CONF_DISPS,
                FNAME_SETTINGS_ARGS,
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
        }))?.0; // let dir = try_with_recovery(...)

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

    pub fn structure(&self) -> FailResult<(Coords, HList2<Rc<[Element]>, Rc<[Mass]>>)>
    { read_input_structure_with_mass(self) }

    pub fn q_positions(&self) -> FailResult<Vec<V3>>
    { Ok(QPositions::load(self.path().join("q-positions.json"))?.0) }

    /// This will be `None` if `.eigenvectors(true)` was not set prior
    /// to the band computation.
    pub fn eigenvectors(&self) -> FailResult<Option<Vec<Basis3>>>
    {Ok({
        let path = self.path().join("eigenvector.npy");
        if path.exists() {
            trace!("Reading eigenvectors...");
            let bases = npy::read_eigenvector_npy(open(path)?)?;
            let bases = bases.into_iter().map(Basis3::from_basis).collect();
            Some(bases)
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

    pub fn eigensystem_at(&self, q: V3) -> FailResult<(Vec<f64>, Basis3)>
    {Ok({
        let index = ::util::index_of_nearest(&self.q_positions()?, q, 1e-4);
        let index = match index {
            Some(i) => i,
            None => bail!("Bands do not include kpoint:\n  dir: {}\npoint: {:?}",
        self.path().display(), q),
        };

        // (a little wasteful...)
        let evals = self.eigenvalues()?.remove(index);
        let evecs = match self.eigenvectors()? {
            None => bail!("Directory has no eigenvectors: {}", self.path().display()),
            Some(mut evs) => evs.remove(index),
        };

        (evals, evecs)
    })}
}

//-----------------------------
// helpers

fn get_sc_dim(conf: &Conf) -> FailResult<Option<[u32; 3]>> {
    use ::util::ext_traits::OptionResultExt;
    conf.0.get("DIM")
        .map(|s| {
            let words = s.split_whitespace().map(str::parse).collect::<Result<Vec<_>, _>>()?;
            match &words[..] {
                &[a, b, c] => Ok([a, b, c]),
                _ => bail!("DIM does not contain three integers!"),
            }
        })
        .fold_ok()
}

fn masses_from_conf(conf: &Conf) -> Option<FailResult<Rc<[Mass]>>>
{Some({
    conf.0.get("MASS")?
        .split_whitespace()
        .map(|s| s.parse().map(Mass))
        .collect::<Result<Vec<_>, _>>()
        .map_err(Into::into)
        .map(Into::into)
})}

/// Reads "POSCAR", the input structure, in a directory with disp.conf.
///
/// **Cannot** be used to read other files in a POSCAR format such as PPOSCAR,
/// because the masses would be incorrect.
fn read_input_structure_with_mass(
    dir: impl AsPath,
) -> FailResult<(Coords, HList2<Rc<[Element]>, Rc<[Mass]>>)>
{
    let conf = Conf::load(dir.join(FNAME_CONF_DISPS))?;
    let masses = {
        masses_from_conf(&conf)
            .ok_or_else(|| format_err!("No MASS in {}!", FNAME_CONF_DISPS))??
    };
    let Poscar { coords, elements, .. } = Poscar::load(dir.join("POSCAR"))?;
    Ok((coords, hlist![elements.into(), masses]))
}

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

    let stdout_worker = ::stdout::spawn_log_worker(child.stdout.take().unwrap());
    let stderr_worker = ::stderr::spawn_log_worker(child.stderr.take().unwrap());

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
pub fn cache_link(src: impl AsPath, dest: impl AsPath) -> FailResult<()>
{
    let (src, dest) = (src.as_path(), dest.as_path());
    hard_link(src, dest)
        .map(|_| ())
        .or_else(|_| Ok({
            // if the file already existed, the link will have failed.
            // Check this before continuing because we don't want to
            //   potentially overwrite a link with a copy.
            if dest.is_file() { // true for files and symlinks to files
                return Ok(());
            }

            // assume the error was due to being on a different filesystem.
            // (Even if not, we will probably just get the same error)
            copy(src, dest)?;
        }))
}

// Like `cache_link` except it fails if the destination exists.
//
// FIXME: (2018-05-27)  Uhhhhhhmm. That description is misleading. The implementation
//                      clearly clobbers existing destination files with `copy`.
//                      (but between the description and the implementation, I cannot
//                       recall which was the original intent!)
pub fn copy_or_link(src: impl AsPath, dest: impl AsPath) -> FailResult<()>
{
    let (src, dest) = (src.as_path(), dest.as_path());
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
    other_members: [self.displacements, self.super_coords, self.super_meta]
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
