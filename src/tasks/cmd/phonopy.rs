/* ************************************************************************ **
** This file is part of rsp2, and is licensed under EITHER the MIT license  **
** or the Apache 2.0 license, at your option.                               **
**                                                                          **
**     http://www.apache.org/licenses/LICENSE-2.0                           **
**     http://opensource.org/licenses/MIT                                   **
**                                                                          **
** Be aware that not all of rsp2 is provided under this permissive license, **
** and that the project as a whole is licensed under the GPL 3.0.           **
** ************************************************************************ */

//! Ressurected from the grave, this extremely subdued form of rsp2's phonopy code now
//! only exists to help compare outputs.

use crate::FailResult;
use crate::traits::{AsPath, Save, Load};
use crate::meta::{self, prelude::*};
use crate::hlist_aliases::*;
use crate::errors::DisplayPathNice;

use crate::cmd::SupercellSpecExt;

use rsp2_tasks_config as cfg;
use std::io::prelude::*;
use std::rc::Rc;
use std::process::Command;
use std::path::{Path};

use rsp2_fs_util::{TempDir};
use rsp2_fs_util as fsx;
use rsp2_structure::{Coords};
use rsp2_structure::supercell::{SupercellToken};
use rsp2_soa_ops::{Permute, Perm};
use rsp2_structure_io::Poscar;

use rsp2_array_types::{V3, Unvee};

use failure::Backtrace;
#[allow(unused)] // rustc bug
use itertools::Itertools;

//--------------------------------------------------------

// filenames invented by rsp2
//
// (we don't bother with constants for fixed filenames used by phonopy, like "POSCAR")
const FNAME_SETTINGS_ARGS: &'static str = "disp.args";
const FNAME_CONF_DISPS: &'static str = "disp.conf";
const FNAME_OUT_SYMMETRY: &'static str = "symmetry.yaml";

//--------------------------------------------------------

// Directory types in this module follow a pattern of having the datatype constructed
// after all files have been made; this is thrown when that is not upheld.
#[derive(Debug, Fail)]
#[fail(display = "Directory '{}' is missing required file '{}' for '{}'", dir, filename, ty)]
pub(crate) struct MissingFileError {
    backtrace: Backtrace,
    ty: &'static str,
    dir: DisplayPathNice,
    filename: String,
}

#[derive(Debug, Fail)]
#[fail(display = "phonopy failed with status {}", status)]
pub(crate) struct PhonopyFailed {
    backtrace: Backtrace,
    pub status: std::process::ExitStatus,
}

impl MissingFileError {
    fn new(ty: &'static str, dir: &dyn AsPath, filename: String) -> Self {
        let backtrace = Backtrace::new();
        let dir = DisplayPathNice(dir.as_path().to_owned().into());
        MissingFileError { backtrace, ty, dir, filename }
    }
}

//--------------------------------------------------------

type SymmetryYaml = rsp2_phonopy_io::SymmetryYaml;
impl Load for SymmetryYaml {
    fn load(path: impl AsPath) -> FailResult<Self>
    { Ok(rsp2_phonopy_io::symmetry_yaml::read(fsx::open(path.as_path())?)?) }
}

//--------------------------------------------------------

type PhonopyDispYaml = rsp2_phonopy_io::PhonopyDispYaml;
impl Load for PhonopyDispYaml {
    fn load(path: impl AsPath) -> FailResult<Self>
    { Ok(rsp2_phonopy_io::phonopy_disp_yaml::read(fsx::open(path.as_path())?)?) }
}

//--------------------------------------------------------

// this is a type alias so we wrap it
#[derive(Debug, Clone, Default)]
pub struct Conf(pub rsp2_phonopy_io::Conf);
impl Load for Conf {
    fn load(path: impl AsPath) -> FailResult<Self>
    { Ok(rsp2_phonopy_io::conf::read(fsx::open_text(path.as_path())?).map(Conf)?) }
}

impl Save for Conf {
    fn save(&self, path: impl AsPath) -> FailResult<()>
    { Ok(rsp2_phonopy_io::conf::write(fsx::create(path.as_path())?, &self.0)?) }
}

//--------------------------------------------------------

/// Type representing extra CLI arguments.
///
/// Used internally to store things that must be preserved between
/// runs but cannot be set in conf files, like e.g. `--tolerance`
#[derive(Serialize, Deserialize)]
#[derive(Debug, Clone, Default)]
pub(crate) struct Args(Vec<String>);

impl<S, Ss> From<Ss> for Args
where
    S: AsRef<str>,
    Ss: IntoIterator<Item=S>,
{
    fn from(args: Ss) -> Self
    { Args(args.into_iter().map(|s| s.as_ref().to_owned()).collect()) }
}

impl Load for Args {
    fn load(path: impl AsPath) -> FailResult<Self>
    {
        use path_abs::FileRead;
        use crate::util::ext_traits::PathNiceExt;
        let path = path.as_path();

        let text = FileRead::open(path)?.read_string()?;
        if let Some(args) = shlex::split(&text) {
            Ok(Args(args))
        } else {
            bail!("Bad args at {}", path.nice())
        }
    }
}

impl Save for Args {
    fn save(&self, path: impl AsPath) -> FailResult<()>
    {
        use path_abs::FileWrite;
        let mut file = FileWrite::create(path.as_path())?;
        for arg in &self.0 {
            writeln!(file, "{}", shlex::quote(arg))?;
        }
        Ok(())
    }
}

//--------------------------------------------------------

mod builder {
    use super::*;

    /// FIXME: This no longer needs to exist but it's just easier to keep around.
    ///
    /// It used to be passed around through a decent amount of the high-level code in rsp2_tasks::cmd,
    /// so that different parts of the configuration could be set at times that were convenient.
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
            for (key, value) in rsp2_phonopy_io::conf::read(file)? {
                me = me.conf(key, value);
            }
            me
        })}

        pub fn supercell_dim(self, dim: [u32; 3]) -> Self
        { self.conf("DIM", dim.iter().join(" ")) }

        pub fn diagonal_disps(self, value: bool) -> Self
        { self.conf("DIAG", fortran_bool(value)) }

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
        fn finalize_config(&self, meta: HList1<meta::SiteMasses>) -> Self
        {
            let masses: meta::SiteMasses = meta.pick();
            self.clone().conf("MASS", masses.iter().join(" "))
        }

        pub(super) fn displacements(
            &self,
            coords: &Coords,
            meta: HList2<
                meta::SiteMasses,
                meta::SiteElements,
            >,
        ) -> FailResult<DirWithDisps>
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
                meta::SiteElements,
                meta::SiteMasses,
            >,
        ) -> FailResult<DirWithDisps>
        {Ok({
            let elements: meta::SiteElements = meta.pick();

            let dir = TempDir::new_labeled("rsp2", "phonopy")?;
            {
                let dir = dir.path();
                trace!("Displacement dir: '{}'...", dir.display());

                let extra_args = self.args_from_settings();
                self.conf.save(dir.join(FNAME_CONF_DISPS))?;
                Poscar {
                    comment: "blah", coords, elements,
                }.save(dir.join("POSCAR"))?;
                extra_args.save(dir.join(FNAME_SETTINGS_ARGS))?;

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
                        .arg("--symmetry")
                        .current_dir(&dir)
                        .stdout(fsx::create(dir.join(FNAME_OUT_SYMMETRY))?);

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

            }
            DirWithDisps::from_existing(dir)?
        })}
    }
}

//--------------------------------------------------------

// NOTE: The original motivation for this type's existence was to be generic over
//       a path type P, which could be a TempDir, or a borrowed path (in the case
//       of opening a previously-existing directory), or etc.
//
//       Now that it isn't used outside of this module anymore, it could probably
//       be trashed... but it just seems easier to leave it alone for now.

/// Represents a directory with the following data:
/// - `POSCAR`: The input structure
/// - `phonopy_disp.yaml`: Phonopy file with displacements
/// - configuration settings which impact the selection of displacements
///   - `--tol`, `--dim`
/// - `symmetry.yaml`: Output of `phonopy --symmetry`, currently present
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
#[derive(Debug)]
struct DirWithDisps {
    dir: TempDir,
    displacements: Vec<(usize, V3)>,
    // These are cached in memory from `phonopy_disp.yaml` due to the likelihood
    // that code using `DirWithDisps` will need them.
    super_coords: Coords,
    #[allow(dead_code)]
    super_meta: HList2<meta::SiteElements, meta::SiteMasses>,
}

impl AsPath for DirWithDisps {
    fn as_path(&self) -> &Path {
        self.dir.as_path()
    }
}

impl DirWithDisps {
    fn from_existing(dir: TempDir) -> FailResult<Self>
    {Ok({
        for name in &[
            "POSCAR",
            "phonopy_disp.yaml",
            FNAME_CONF_DISPS,
            FNAME_SETTINGS_ARGS,
            FNAME_OUT_SYMMETRY,
        ] {
            let path = dir.as_path().join(name);
            if !path.exists() {
                throw!(MissingFileError::new("DirWithDisps", &dir, name.to_string()));
            }
        }

        trace!("Parsing phonopy_disp.yaml...");
        let PhonopyDispYaml {
            displacements, coords, elements, masses,
        } = Load::load(dir.as_path().join("phonopy_disp.yaml"))?;
        let elements: Rc<[_]> = elements.into();
        let masses: Rc<[_]> = masses.into_iter().map(meta::Mass).collect::<Vec<_>>().into();
        let meta = hlist![elements, masses];

        DirWithDisps {
            dir,
            displacements,
            super_coords: coords,
            super_meta: meta,
        }
    })}

    /// Get the structure from `phonopy_disp.yaml`.
    ///
    /// # Note
    ///
    /// This superstructure was generated by phonopy, and the atoms may be in
    /// a different order than most supercells in rsp2 (those produced with SupercellToken).
    #[allow(unused)]
    fn super_coords(&self) -> &Coords
    { &self.super_coords }

    /// Get displacements.  *The atom indices are for phonopy's supercell!*
    fn displacements(&self) -> &[(usize, V3)]
    { &self.displacements }

    // Although we ultimately use `spglib` (since it gives fuller precision for
    // the translations), the intent is still to get the spacegroup used *by phonopy*
    // (as otherwise we might end up with e.g. underdetermined force constants)
    //
    // So we call `phonopy --sym` for the sole purpose of validating that the spacegroup
    // returned is the same (or a subgroup). This could fail if our method of assigning
    // integer atom types differed from phonopy (e.g. are masses checked?).
    fn phonopy_sg_op_count(&self) -> FailResult<usize>
    { Ok(SymmetryYaml::load(self.dir.join(FNAME_OUT_SYMMETRY))?.space_group_operations.len()) }
}

/// A smattering of information about the displacements chosen by phonopy, and how they
/// relate to rsp2's conventions.
pub struct PhonopyDisplacements {
    /// The original displacements exactly as they were chosen by phonopy.
    pub phonopy_super_displacements: Vec<(usize, V3)>,

    /// Permutation that rearranges phonopy's superstructure to match `superstructure`.
    ///
    /// I.e. `phonopy_superstructure.permuted_by(&perm_from_phonopy) ≈ superstructure`,
    /// modulo lattice point translations.
    pub coperm_from_phonopy: Perm,

    /// Displacements that use indices into the primitive structure.
    ///
    /// You are free to just use this field and ignore the rest (which merely come
    /// for "free" with it). This field should be compatible with superstructures
    /// of any size, and obviously does not depend on the convention for ordering
    /// sites in a supercell.
    pub prim_displacements: Vec<(usize, V3)>,

    /// Number of spacegroup operations detected by phonopy when it generated its displacements.
    ///
    /// If this is larger than the amount found by rsp2, the force constants will end up
    /// missing some terms.
    pub spacegroup_op_count: usize,
}

/// Produce a variety of data describing the displacements in terms of rsp2's conventions
/// (whereas most other methods on `DirWithDisps` use phonopy's conventions).
pub fn phonopy_displacements(
    settings: &cfg::Phonons,
    prim_coords: &Coords,
    prim_meta: HList2<
        meta::SiteElements,
        meta::SiteMasses,
    >,
    sc: &SupercellToken,
    // supercell coordinates in rsp2's ordering convention, as created by `sc`
    our_super_coords: &Coords,
) -> FailResult<PhonopyDisplacements> {
    let displacement_distance = settings.displacement_distance.expect("(bug) missing displacement-distance should have been caught earlier");
    let symmetry_tolerance = settings.symmetry_tolerance.expect("(bug) missing symmetry-tolerance should have been caught earlier");
    let dir = {
        let mut builder = {
            builder::Builder::new()
                // HACK: Give phonopy a slightly smaller symprec to ensure that, in case
                // rsp2 and phonopy find different spacegroups, phonopy should find the
                // smaller one.
                .symmetry_tolerance(symmetry_tolerance * 0.99)
                .conf("DISPLACEMENT_DISTANCE", format!("{:e}", displacement_distance))
                .supercell_dim(settings.supercell.dim_for_unitcell(prim_coords.lattice()))
        };
        if let cfg::PhononDispFinder::Phonopy { diag } = settings.disp_finder {
            builder = builder.diagonal_disps(diag);
        }
        builder.displacements(prim_coords, prim_meta.sift())?
    };
    let sc_dims = sc.as_diagonal().expect("non-diagonal supercell for phonopy not implemented");
    assert_eq!(settings.supercell.dim_for_unitcell(prim_coords.lattice()), sc_dims);

    // cmon, big money, big money....
    // if these assertions always succeed, it will save us a
    // good deal of implementation work.
    let perm_from_phonopy;
    {
        let phonopy_super_coords = Poscar::load(dir.join("SPOSCAR"))?.coords;

        perm_from_phonopy = phonopy_super_coords.perm_to_match(&our_super_coords, 1e-10)?;

        // make phonopy match us
        let phonopy_super_coords = phonopy_super_coords.clone().permuted_by(&perm_from_phonopy);

        let err_msg = "\
            phonopy's superstructure does not match rsp2's conventions! \
            Unfortunately, support for this scenario is not yet implemented.\
        ";
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
        dir.displacements().iter()
            .map(|&(phonopy_idx, disp)| {
                let our_super_idx = perm_from_phonopy.permute_index(phonopy_idx);
                let our_prim_idx = primitive_atoms[our_super_idx];
                (our_prim_idx, disp)
            })
            .collect::<Vec<_>>()
    };

    Ok(PhonopyDisplacements {
        prim_displacements,
        coperm_from_phonopy: perm_from_phonopy,
        phonopy_super_displacements: dir.displacements().to_vec(),
        spacegroup_op_count: dir.phonopy_sg_op_count()?,
    })
}

/// Struct to simulate named arguments
pub struct PhonopyForceSets<'a> {
    /// The original displacements exactly as they were chosen by phonopy.
    pub phonopy_super_displacements: &'a [(usize, V3)],

    /// Which cell (as defined in the docs for [`SupercellToken`]) did rsp2 choose for each displacement?
    pub rsp2_displaced_site_cells: &'a [[u32; 3]],

    /// Perm that rearranges phonopy's superstructure to match rsp2's superstructure.
    pub coperm_from_phonopy: &'a Perm,

    pub sc: &'a SupercellToken,
}

impl PhonopyForceSets<'_> {
    pub fn write(
        self,
        w: impl Write,
        force_sets: &Vec<std::collections::BTreeMap<usize, V3>>,
    ) -> FailResult<()> {
        let PhonopyForceSets {
            coperm_from_phonopy,
            phonopy_super_displacements,
            rsp2_displaced_site_cells,
            sc,
        } = self;
        let num_atoms = coperm_from_phonopy.len();

        // rsp2 and phonopy agree about the primitive sites that were displaced,
        // but the supercell data may differ in two ways:
        //
        // * Different images may have been chosen to be displaced.
        // * The supercell atoms may be in a different order.

        // permutation that turns our metadata into phonopy's metadata
        let ref deperm_from_phonopy = coperm_from_phonopy.inverted();
        let deperm_to_phonopy = coperm_from_phonopy; // inverse of inverse

        let site_cells = sc.atom_cells();

        let ref phonopy_displaced_site_cells: Vec<[u32; 3]> = {
            phonopy_super_displacements.iter().map(|&(phonopy_site, _)| {
                // FIXME I can't quite figure out whether this should use the
                //       coperm or the deperm, but the deperm seems to make the most sense.
                //
                // Currently it doesn't seem to matter which one we use, because the permutation
                // between rsp2 and phonopy always seems to be involutory; that is, it is equal to
                // its own inverse.
                if coperm_from_phonopy != deperm_from_phonopy {
                    warn_once!("Untested code path: 94111e7c-3afe-4838-948a-108580e8d252");
                }
                let rsp2_site = deperm_from_phonopy.permute_index(phonopy_site);
                site_cells[rsp2_site]
            }).collect()
        };

        let phonopy_force_sets: Vec<Vec<V3>> = {
            zip_eq!(force_sets, phonopy_displaced_site_cells, rsp2_displaced_site_cells)
                .map(|(rsp2_row, &phonopy_cell, &rsp2_cell)| {
                    // First, perform a translation that translates rsp2_cell to phonopy_cell,
                    // so that the correct site is displaced.
                    let rsp2_latt = sc.lattice_point_from_cell(rsp2_cell);
                    let phonopy_latt = sc.lattice_point_from_cell(phonopy_cell);
                    let translation_latt = phonopy_latt - rsp2_latt;

                    let deperm = sc.lattice_point_translation_deperm(translation_latt);

                    // Then, permute into phonopy's supercell convention

                    let deperm = deperm.then(deperm_to_phonopy);

                    // Apply this permutation to the columns while densifying
                    let mut phonopy_row = vec![V3::zero(); num_atoms];
                    for (&our_index, &vector) in rsp2_row {
                        let phonopy_index = deperm.permute_index(our_index);
                        phonopy_row[phonopy_index] = vector;
                    }

                    phonopy_row
                }).collect()
        };

        rsp2_phonopy_io::force_sets::write(w, phonopy_super_displacements, phonopy_force_sets)
    }
}

//-----------------------------
// helpers

fn round_checked(x: f64, tol: f64) -> FailResult<i32>
{Ok({
    let r = x.round();
    ensure!((r - x).abs() < tol, "not nearly integral: {}", x);
    r as i32
})}

fn fortran_bool(b: bool) -> &'static str {
    match b {
        true => ".TRUE.",
        false => ".FALSE.",
    }
}

pub(crate) fn log_stdio_and_wait(
    mut cmd: std::process::Command,
    stdin: Option<String>,
) -> FailResult<()>
{Ok({
    use std::process::Stdio;

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

    let stdout_worker = crate::stdout::spawn_log_worker(child.stdout.take().unwrap());
    let stderr_worker = crate::stderr::spawn_log_worker(child.stderr.take().unwrap());

    check_status(child.wait()?)?;

    let _ = stdout_worker.join();
    let _ = stderr_worker.join();
})}

fn check_status(status: std::process::ExitStatus) -> Result<(), PhonopyFailed>
{
    if status.success() { Ok(()) }
    else {
        let backtrace = failure::Backtrace::new();
        Err(PhonopyFailed { backtrace, status })
    }
}

//-----------------------------
