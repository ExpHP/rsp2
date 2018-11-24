/* ********************************************************************** **
**  This file is part of rsp2.                                            **
**                                                                        **
**  rsp2 is free software: you can redistribute it and/or modify it under **
**  the terms of the GNU General Public License as published by the Free  **
**  Software Foundation, either version 3 of the License, or (at your     **
**  option) any later version.                                            **
**                                                                        **
**      http://www.gnu.org/licenses/                                      **
**                                                                        **
** Do note that, while the whole of rsp2 is licensed under the GPL, many  **
** parts of it are licensed under more permissive terms.                  **
** ********************************************************************** */

// HERE BE DRAGONS

pub(crate) mod integrate_2d;


use self::ev_analyses::GammaSystemAnalysis;
use self::param_optimization::ScalableCoords;
mod ev_analyses;

use self::trial::TrialDir;
pub(crate) mod trial;

pub(crate) use ::filetypes::stored_structure::StoredStructure;

use self::relaxation::EvLoopDiagonalizer;
pub(crate) use self::relaxation::DidEvChasing;
mod relaxation;
mod acoustic_search;
mod param_optimization;

pub(crate) mod python;

use ::{FailResult, FailOk};
use ::rsp2_tasks_config::{self as cfg, Settings, NormalizationMode, SupercellSpec};
use ::traits::{AsPath, Load, Save};
use ::phonopy::{DirWithBands, DirWithDisps, DirWithForces};
use ::rsp2_lammps_wrap::LammpsOnDemand;

use ::meta::{self, prelude::*};
use ::util::ext_traits::{OptionResultExt, PathNiceExt};
use ::math::{
    basis::{Basis3},
    bands::{ScMatrix},
    dynmat::{ForceConstants},
};
use self::acoustic_search::ModeKind;

use ::path_abs::{PathAbs, PathArc, PathFile, PathDir};
use ::rsp2_structure::consts::CARBON;
use ::rsp2_slice_math::{vnorm};

use ::slice_of_array::prelude::*;
use ::rsp2_array_utils::arr_from_fn;
use ::rsp2_array_types::{V3, Unvee};
use ::rsp2_structure::{Coords, Lattice};
use ::rsp2_structure::{
    layer::LayersPerUnitCell,
    bonds::FracBonds,
};
use ::phonopy::Builder as PhonopyBuilder;

use ::rsp2_fs_util::{rm_rf, hard_link};

use ::std::{
    path::{PathBuf},
    io::{Write},
    ffi::{OsStr, OsString},
    collections::{BTreeMap},
    rc::{Rc},
    fmt,
};

use ::itertools::Itertools;
use ::hlist_aliases::*;
use ::potential::{PotentialBuilder, DiffFn, CommonMeta};
use ::traits::save::Json;
use ::threading::Threading;

const SAVE_BANDS_DIR: &'static str = "gamma-bands";

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum StructureFileType {
    Poscar,
    LayersYaml,
    StoredStructure,
}

pub enum EvLoopStructureKind {
    Initial,
    PreEvChase(Iteration),
    PostEvChase(Iteration),
    Final,
}

// FIXME hide the member and forbid the value 0
//
/// Number of an iteration of the eigenvector loop.
///
/// Iterations are numbered starting from 1.
#[derive(Debug, Copy, Clone)]
pub struct Iteration(pub u32);

impl fmt::Display for Iteration {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result
    { fmt::Display::fmt(&self.0, f) }
}

impl TrialDir {
    pub(crate) fn run_relax_with_eigenvectors(
        self,
        on_demand: Option<LammpsOnDemand>,
        settings: &Settings,
        file_format: StructureFileType,
        input: &PathAbs,
        // shameful HACK
        stop_after_dynmat: bool,
    ) -> FailResult<()>
    {Ok({
        let pot = PotentialBuilder::from_root_config(&self, on_demand, &settings);

        let (optimizable_coords, meta) = {
            read_optimizable_structure(
                settings.layer_search.as_ref(),
                settings.masses.as_ref(),
                file_format, input,
            )?
        };

        let original_coords = {
            // (can't reliably get bonds until the lattice parameter is correct)
            let meta = meta.clone().prepend(None::<meta::FracBonds>);
            ::cmd::param_optimization::optimize_layer_parameters(
                &settings.scale_ranges,
                &pot,
                optimizable_coords,
                meta.sift(),
            )?.construct()
        };

        let meta = meta.prepend({
            settings.bond_radius.map(|bond_radius| FailOk({
                Rc::new(FracBonds::from_brute_force(&original_coords, bond_radius)?)
            })).fold_ok()?
        });

        self.write_stored_structure(
            &self.structure_path(EvLoopStructureKind::Initial),
            "Initial structure (after lattice optimization)",
            &original_coords, meta.sift(),
        )?;

        // FIXME: Prior to the addition of the sparse solver, the code used to do something like:
        //
        //       let _do_not_drop_the_bands_dir = final_bands_dir
        //
        // IIRC, this unsightly hack was to allow RSP2_SAVETEMP to recover bands directories if code
        // after ev_analysis info panics, even if save_bands is disabled.
        //
        // Unfortunately, the addition of a second solver has only made this more convoluted,
        // and so now the guard is an option. (`None` when the sparse solver is used)
        let _do_not_drop_the_bands_dir: Option<DirWithBands<Box<AsPath>>>;

        let (coords, ev_analysis) = {
            use self::cfg::PhononEigenSolver::*;

            // macro to simulate a generic closure.
            // (we can't use dynamic polymorphism due to the differing associated types)
            macro_rules! do_ev_loop {
                ($diagonalizer:expr) => {{
                    let diagonalizer = $diagonalizer;
                    self.do_main_ev_loop(
                        settings, &*pot, diagonalizer, original_coords, meta.sift(),
                        stop_after_dynmat,
                    )
                }}
            }

            match settings.phonons.eigensolver {
                Phonopy { save_bands } => {
                    let save_bands = match save_bands {
                        true => Some(self.save_bands_dir()),
                        false => None,
                    };
                    let (coords, ev_analysis, final_bands_dir) = {
                        do_ev_loop!(PhonopyDiagonalizer { save_bands })?
                    };
                    _do_not_drop_the_bands_dir = Some(final_bands_dir);
                    (coords, ev_analysis)
                },
                Sparse { shift_invert_attempts } => {
                    let (coords, ev_analysis, final_iteration) = {
                        do_ev_loop!(SparseDiagonalizer { shift_invert_attempts })?
                    };
                    _do_not_drop_the_bands_dir = None;

                    // HACK: Put last gamma dynmat at a predictable path.
                    let final_iteration = final_iteration.expect("ev-loop should have iterations!");
                    rm_rf(self.join("gamma-dynmat.json"))?;
                    hard_link(
                        self.gamma_dynmat_path(final_iteration),
                        self.final_gamma_dynmat_path(),
                    )?;

                    (coords, ev_analysis)
                },
            }
        };

        self.write_stored_structure(
            &self.structure_path(EvLoopStructureKind::Final),
            "Final structure",
            &coords, meta.sift(),
        )?;

        write_eigen_info_for_machines(&ev_analysis, self.create_file("eigenvalues.final")?)?;

        write_ev_analysis_output_files(&self, &ev_analysis)?;
        self.write_summary_file(settings, &*pot, &ev_analysis)?;
    })}
}

pub(crate) fn write_ev_analysis_output_files(
    dir: &PathDir,
    eva: &GammaSystemAnalysis,
) -> FailResult<()>
{Ok({
    use path_abs::FileWrite;

    if let (Some(frequency), Some(raman)) = (&eva.ev_frequencies, &eva.ev_raman_tensors) {
        #[derive(Serialize)]
        #[serde(rename_all = "kebab-case")]
        struct Output {
            frequency: Vec<f64>,
            average_3d: Vec<f64>,
            backscatter: Vec<f64>,
        }
        use ::math::bond_polarizability::LightPolarization::*;
        ::serde_json::to_writer(FileWrite::create(dir.join("raman.json"))?, &Output {
            frequency: frequency.0.to_vec(),
            average_3d: raman.0.iter().map(|t| t.integrate_intensity(&Average)).collect(),
            backscatter: raman.0.iter().map(|t| t.integrate_intensity(&BackscatterZ)).collect(),
        })?;
    }

    if let (Some(sc_mats), Some(unfold_probs)) = (&eva.layer_sc_mats, &eva.unfold_probs) {
        #[derive(Serialize)]
        #[serde(rename_all = "kebab-case")]
        struct Output {
            layer_sc_dims: Vec<[u32; 3]>,
            layer_q_indices: Vec<Vec<[u32; 3]>>,
            layer_ev_q_probs: Vec<Vec<Vec<f64>>>,
        }

        ::serde_json::to_writer(FileWrite::create(dir.join("unfold.json"))?, &Output {
            layer_sc_dims: sc_mats.iter().map(|m| m.periods).collect(),
            layer_q_indices: {
                unfold_probs.layer_unfolders.iter()
                    .map(|u| u.q_indices().to_vec())
                    .collect()
            },
            layer_ev_q_probs: unfold_probs.layer_ev_q_probs.clone(),
        })?;
    }
})}

impl TrialDir {
    // log when writing stored structures, especially during loops
    // (to remove any doubt about the iteration number)
    fn write_stored_structure(
        &self,
        dir: impl AsPath,
        poscar_headline: &str,
        coords: &Coords,
        meta: HList5<
            meta::SiteElements,
            meta::SiteMasses,
            Option<meta::SiteLayers>,
            Option<meta::LayerScMatrices>,
            Option<meta::FracBonds>,
        >,
    ) -> FailResult<()>
    {Ok({
        let path = self.join(dir);

        trace!("Writing '{}'", path.nice());
        StoredStructure {
            title: poscar_headline.into(),
            coords: coords.clone(),
            elements: meta.pick(),
            masses: meta.pick(),
            layers: meta.pick(),
            layer_sc_matrices: meta.pick(),
            frac_bonds: meta.pick(),
        }.save(path)?
    })}

    /// Read a .structure directory.
    /// Relative paths are intepreted relative to the trial directory.
    fn read_stored_structure_data(
        &self,
        dir: impl AsPath,
    ) -> FailResult<(
        Coords,
        HList5<
            meta::SiteElements,
            meta::SiteMasses,
            Option<meta::SiteLayers>,
            Option<meta::LayerScMatrices>,
            Option<meta::FracBonds>,
        >,
    )>
    {Ok({
        let stored = self.read_stored_structure(dir)?;
        let meta = stored.meta();
        (stored.coords, meta)
    })}

    /// Read a .structure directory.
    /// Relative paths are intepreted relative to the trial directory.
    fn read_stored_structure(&self, dir: impl AsPath) -> FailResult<StoredStructure>
    { Load::load(self.join(dir.as_path())) }
}

struct PhonopyDiagonalizer {
    save_bands: Option<PathAbs>,
}
impl EvLoopDiagonalizer for PhonopyDiagonalizer {
    // I think this is used to keep TempDir bands alive after diagonalization, so that
    //  they can be recovered by RSP2_TEMPDIR if a panic occurs, even without `save_bands`.
    // (to be honest, the code has gone through so many changes in requirements that I can
    //  no longer remember; I just put this here to preserve behavior during a refactor.)
    type ExtraOut = DirWithBands<Box<AsPath>>;

    fn do_post_relaxation_computations(
        &self,
        _trial: &TrialDir,
        settings: &Settings,
        pot: &PotentialBuilder,
        stored: &StoredStructure,
        _stop_after_dynmat: bool, // HACK
        _iteration: Option<Iteration>,
    ) -> FailResult<(Vec<f64>, Basis3, DirWithBands<Box<AsPath>>)>
    {Ok({
        let meta = stored.meta();
        let coords = stored.coords.clone();

        let phonopy = phonopy_builder_from_settings(&settings.phonons, coords.lattice());
        match settings.phonons.disp_finder {
            cfg::PhononDispFinder::Phonopy { diag } => {
                let _ = diag; // already handled during builder construction
            },
            cfg::PhononDispFinder::Rsp2 { .. } => {
                bail!("'disp-finder: rsp2' and 'eigensolver: phonopy' are incompatible");
            },
        }

        let bands_dir = self.create_bands_dir(&settings.threading, pot, &phonopy, &coords, meta.sift())?;
        let (evals, evecs) = bands_dir.eigensystem_at(Q_GAMMA)?;

        (evals, evecs, bands_dir)
    })}

    fn allow_unfold_bands(&self) -> bool {
        // phonopy diagonalizer is only ever used on small systems
        true
    }
}

impl PhonopyDiagonalizer {
    // Create a DirWithBands that may or may not be a TempDir (based on config)
    fn create_bands_dir(
        &self,
        cfg_threading: &cfg::Threading,
        pot: &PotentialBuilder,
        phonopy: &PhonopyBuilder,
        coords: &Coords,
        meta: HList3<
            meta::SiteElements,
            meta::SiteMasses,
            Option<meta::FracBonds>,
        >,
    ) -> FailResult<DirWithBands<Box<AsPath>>>
    {Ok({
        let disp_dir = phonopy.displacements(coords, meta.sift())?;
        let force_sets = do_force_sets_at_disps_for_phonopy(pot, cfg_threading, &disp_dir)?;

        let bands_dir = disp_dir
            .make_force_dir(&force_sets)?
            .build_bands()
            .eigenvectors(true)
            .compute(&[Q_GAMMA])?;

        if let Some(save_dir) = &self.save_bands {
            rm_rf(save_dir)?;
            bands_dir.relocate(save_dir.clone())?.boxed()
        } else {
            bands_dir.boxed()
        }
    })}
}

struct SparseDiagonalizer {
    shift_invert_attempts: u32,
}
impl EvLoopDiagonalizer for SparseDiagonalizer {
    type ExtraOut = Option<Iteration>;

    fn do_post_relaxation_computations(
        &self,
        _trial: &TrialDir,
        settings: &Settings,
        pot: &PotentialBuilder,
        stored: &StoredStructure,
        stop_after_dynmat: bool,
        iteration: Option<Iteration>,
    ) -> FailResult<(Vec<f64>, Basis3, Option<Iteration>)>
    {Ok({
        let prim_coords = &stored.coords;
        let prim_meta: HList4<
            meta::SiteElements,
            meta::SiteMasses,
            Option<meta::SiteLayers>,
            Option<meta::FracBonds>
        > = stored.meta().sift();

        let compute_deperms = |coords: &_, cart_ops: &_| {
            ::rsp2_structure::find_perm::spacegroup_deperms(
                coords,
                cart_ops,
                // larger than SYMPREC because the coords we see may may be slightly
                // different from what spglib saw, but not so large that we risk pairing
                // the wrong atoms
                settings.phonons.symmetry_tolerance * 3.0,
            )
        };

        // (FIXME)
        // Q: Gee, golly ExpHP! Why does 'eigensolver: sparse' need to care about 'disp_finder'?
        //    Couldn't this be factored out so that these two config sections are handled more
        // orthogonally?
        //
        // A: Because AHHHHHHHHHGGGHGHGH
        // A: Because, if you look closely, the two eigensolvers actually use different schemes for
        //    ordering the sites in the supercell. `eigensolver: phonopy` needs to use the same
        //    order that phonopy uses, because that's what `BandsBuilder` takes, while
        //    `eigensolver: sparse` requires order to match the conventions set by SupercellToken.
        //    (and even if BandsBuilder was made more lenient to allow reordering, we would still
        //     have the silly problem that `PhonopyDiagonalizer` needs the `DirWithDisps`)
        let (super_coords, prim_displacements, sc, cart_ops) = match settings.phonons.disp_finder {
            cfg::PhononDispFinder::Phonopy { diag: _ } => {
                let phonopy = phonopy_builder_from_settings(&settings.phonons, prim_coords.lattice());
                let disp_dir = phonopy.displacements(prim_coords, prim_meta.sift())?;

                let ::phonopy::Rsp2StyleDisplacements {
                    super_coords, sc, prim_displacements, ..
                } = disp_dir.rsp2_style_displacements()?;

                let cart_ops = disp_dir.symmetry()?;
                (super_coords, prim_displacements, sc, cart_ops)
            },
            cfg::PhononDispFinder::Rsp2 { ref directions } => {
                use self::python::SpgDataset;

                let sc_dim = settings.phonons.supercell.dim_for_unitcell(prim_coords.lattice());
                let (super_coords, sc) = ::rsp2_structure::supercell::diagonal(sc_dim).build(prim_coords);

                trace!("Computing symmetry");
                let cart_ops = {
                    let atom_types: Vec<u32> = {
                        let elements: meta::SiteElements = prim_meta.pick();
                        elements.iter().map(|e| e.atomic_number()).collect()
                    };
                    SpgDataset::compute(prim_coords, &atom_types, settings.phonons.symmetry_tolerance)?
                        .cart_ops()
                };

                trace!("Computing deperms in primitive cell");
                let prim_deperms = compute_deperms(&prim_coords, &cart_ops)?;
                let prim_stars = ::math::stars::compute_stars(&prim_deperms);

                let prim_displacements = ::math::displacements::compute_displacements(
                    directions,
                    cart_ops.iter().map(|c| {
                        c.int_rot(prim_coords.lattice()).expect("bad operator from spglib!?")
                    }),
                    &prim_stars,
                    &prim_coords,
                    settings.phonons.displacement_distance,
                );
                (super_coords, prim_displacements, sc, cart_ops)
            },
        };

        let super_meta = {
            // macro to generate a closure, because generic closures don't exist
            macro_rules! f {
                () => { |x: Rc<[_]>| -> Rc<[_]> {
                    sc.replicate(&x[..]).into()
                }};
            }
            // replicating the primitive cell bonds is not worth the trouble
            let super_bonds = settings.bond_radius.map(|bond_radius| FailOk({
                Rc::new(FracBonds::from_brute_force(&super_coords, bond_radius)?)
            })).fold_ok()?;
            prim_meta.clone().map(hlist![
                f!(),
                f!(),
                |opt: Option<_>| opt.map(f!()),
                |_: Option<meta::FracBonds>| { super_bonds },
            ])
        };

        let super_displacements: Vec<_> = {
            prim_displacements.iter()
                .map(|&(prim, disp)| {
                    let atom = sc.atom_from_cell(prim, ForceConstants::DESIGNATED_CELL);
                    (atom, disp)
                })
                .collect()
        };
        trace!("num spacegroup ops: {}", cart_ops.len());
        trace!("num displacements:  {}", super_displacements.len());
        let force_sets = do_force_sets_at_disps_for_sparse(
            pot,
            &settings.threading,
            &super_displacements,
            &super_coords,
            super_meta.sift(),
        )?;
//        { // FIXME add special log flag
//            writeln!(_trial.create_file("force-sets")?, "{:?}", force_sets).unwrap();
//        }

        let cart_rots: Vec<_> = {
            cart_ops.iter().map(|c| c.cart_rot()).collect()
        };

        trace!("Computing deperms in supercell");
        let super_deperms = compute_deperms(&super_coords, &cart_ops)?;

        if log_enabled!(target: "rsp2_tasks::special::visualize_sparse_forces", ::log::Level::Trace) {
            visualize_sparse_force_sets(
                &_trial,
                &super_coords,
                &super_displacements,
                &super_deperms,
                &cart_ops,
                &force_sets,
            )?;
        }

        trace!("Computing sparse force constants");
        let force_constants = ::math::dynmat::ForceConstants::compute_required_rows(
            &super_displacements,
            &force_sets,
            &cart_rots,
            &super_deperms,
            &sc,
        )?;

        trace!("Computing sparse dynamical matrix");

        let dynmat = {
            force_constants
                .gamma_dynmat(&sc, prim_meta.pick())
                .hermitianize()
        };
        // HACK
        // Nasty side-effect, but it's best to write this now (rather than letting the caller
        // take care of it) so that it is there for debugging if we run into an error before
        // this function finishes.
        if let Some(iteration) = iteration {
            // we can't write NPZ easily from rust, so write JSON and convert via python
            Json(dynmat.cereal()).save(_trial.uncompressed_gamma_dynmat_path(iteration))?;

            ::cmd::python::convert::dynmat(
                _trial.uncompressed_gamma_dynmat_path(iteration),
                _trial.gamma_dynmat_path(iteration),
                ::cmd::python::convert::Mode::Delete,
            )?;
        }
        // EVEN NASTIER HACK
        // ...I'd let this speak for itself, but it really can't.
        if stop_after_dynmat {
            return Err(StoppedAfterDynmat.into());
        }

        {
            let max_size = (|(a, b)| a * b)(dynmat.0.dim);
            let nnz = dynmat.0.nnz();
            let density = nnz as f64 / max_size as f64;
            trace!("nnz: {} out of {} blocks (matrix density: {:.3e})", nnz, max_size, density);
        }
        trace!("Diagonalizing dynamical matrix");
        let (evals, evecs) = {
            pot.eco_mode(|| { // don't let MPI processes compete with python's threads
                dynmat.compute_negative_eigensolutions(self.shift_invert_attempts)
            })?
        };
        trace!("Done diagonalizing dynamical matrix");
        (evals, evecs, iteration)
    })}

    fn allow_unfold_bands(&self) -> bool {
        // sparse diagonalizer is used on large systems where UnfoldProbs is prohibitively
        // expensive to compute, and it's not even very useful when we're only looking at
        // imaginary/acoustic modes.
        false
    }
}

// HACK
// used to simulate a sort of unwind on successful exiting.
// Take a look at the places where it's used and try not to throw up.
//
// In case you haven't guessed, I've given up all hope on keeping rsp2-tasks maintainable.
// I need to rip out all of the legacy and/or experimental features I'm not using or start fresh.
#[derive(Debug, Fail)]
#[fail(display = "stopped after dynmat.  THIS IS NOT AN ACTUAL ERROR. THIS IS A DUMB HACK.")]
pub(crate) struct StoppedAfterDynmat;

use ::rsp2_soa_ops::{Perm, Permute};
use ::rsp2_structure::CartOp;
// FIXME incorrect for nontrivial supercells. Should use primitive stars and translate
//       the displaced atom to the correct image after rotation. (this would be easiest to
//       do using the functionality in the ForceConstants code)
fn visualize_sparse_force_sets(
    trial: &TrialDir,
    super_coords: &Coords,
    super_disps: &[(usize, V3)],
    super_deperms: &[Perm],
    cart_ops: &[CartOp],
    force_sets: &[BTreeMap<usize, V3>],
) -> FailResult<()>
{Ok({
    use ::rsp2_structure::consts;
    use ::rsp2_structure_io::Poscar;

    let subdir = trial.join("visualize-forces");
    if let Ok(dir) = PathDir::new(&subdir) {
        dir.remove_all()?;
    }
    let subdir = PathDir::create(subdir)?;

    let super_stars = ::math::stars::compute_stars(super_deperms);

    for (disp_i, (&(displaced, _), map)) in zip_eq!(super_disps, force_sets).enumerate() {
        let disp_dir = PathDir::create(subdir.join(format!("{:04}", disp_i)))?;

        let mut original_elements = vec![consts::CARBON; super_coords.len()];
        for &affected in map.keys() {
            original_elements[affected] = consts::OXYGEN;
        }
        original_elements[displaced] = consts::NITROGEN;

        let star_i = super_stars.assignments()[displaced];
        for &oper_i in super_stars[star_i].opers_from_rep(displaced) {
            let coords = cart_ops[oper_i].transform(&super_coords);
            let coords = coords.permuted_by(&super_deperms[oper_i]);
            let elements = original_elements.clone().permuted_by(&super_deperms[oper_i]);

            // (coalesce groups so vesta doesn't barf)
            let perm = Perm::argsort(&elements);
            let coords = coords.permuted_by(&perm);
            let elements = elements.permuted_by(&perm);

            // HACK
            let mut coords = coords;
            for v in coords.carts_mut() {
                v[2] *= -1.0;
            }

            Poscar {
                comment: format!("displacement {}, sg operator {}", disp_i, oper_i),
                coords,
                elements,
            }.save(disp_dir.join(format!("{:03}.vasp", oper_i)))?;
        }

    }
})}


// wrapper around the gamma_system_analysis module which handles all the newtype conversions
fn run_gamma_system_analysis(
    coords: &Coords,
    meta: HList5<
        meta::SiteElements,
        meta::SiteMasses,
        Option<meta::SiteLayers>,
        Option<meta::LayerScMatrices>,
        Option<meta::FracBonds>,
    >,
    evals: &[f64],
    evecs: &Basis3,
    mode_classifications: Option<Rc<[ModeKind]>>,
    // can set to false to forcibly disable this expensive operation even
    // if all necessary data is available
    unfold_bands: bool,
) -> FailResult<GammaSystemAnalysis> {
    use self::ev_analyses::*;

    // FIXME it would be nice if gamma_analysis used the same metadata types as this part
    //       of the code does...
    let hlist_pat![
        site_elements, site_masses, site_layers, layer_sc_matrices, frac_bonds,
    ] = meta;

    let cart_bonds = frac_bonds.as_ref().map(|b| b.to_cart_bonds(coords));

    gamma_system_analysis::Input {
        site_layers: site_layers,
        layer_sc_mats: layer_sc_matrices,
        site_masses: Some(site_masses.clone()),
        ev_classifications: mode_classifications.map(|x| EvClassifications(x.to_vec())),
        site_elements: Some(site_elements),
        site_coords: Some(coords.clone()),
        ev_frequencies: Some(EvFrequencies(evals.to_vec())),
        ev_eigenvectors: Some(EvEigenvectors(evecs.clone())),
        bonds: cart_bonds.map(Bonds),
        permission_to_unfold_bands: if unfold_bands { Some(PermissionToUnfoldBands) } else { None },
    }.compute()
}

fn write_eigen_info_for_humans(
    analysis: &GammaSystemAnalysis,
    writeln: &mut FnMut(String) -> FailResult<()>,
) -> FailResult<()>
{
    analysis.make_columns(ev_analyses::ColumnsMode::ForHumans)
        .expect("(bug) no columns, not even frequency?")
        .into_iter().map(writeln).collect()
}

fn write_eigen_info_for_machines(
    analysis: &GammaSystemAnalysis,
    mut file: impl Write,
) -> FailResult<()>
{
    analysis.make_columns(ev_analyses::ColumnsMode::ForMachines)
        .expect("(bug) no columns, not even frequency?")
        .into_iter().map(|s| FailOk(writeln!(file, "{}", s)?)).collect()
}

impl TrialDir {
    fn write_summary_file(
        &self,
        settings: &Settings,
        pot: &PotentialBuilder,
        ev_analysis: &GammaSystemAnalysis,
    ) -> FailResult<()> {Ok({
        use ::ui::cfg_merging::{make_nested_mapping, no_summary, merge_summaries};

        #[derive(Serialize)]
        struct EnergyPerAtom {
            initial: f64,
            final_: f64,
            before_ev_chasing: f64,
        }

        // FIXME: Rather than assuming these files are here, this should perhaps
        //        be done by saving structures into strongly typed objects
        //        for the analysis module
        let mut out = vec![];
        out.push(ev_analysis.make_summary(settings));
        out.push({
            let f = |kind| FailOk({
                let s = self.structure_path(kind);
                let (coords, meta) = self.read_stored_structure_data(&s)?;

                let na = coords.num_atoms() as f64;
                pot.one_off().compute_value(&coords, meta.sift())? / na
            });

            let initial = f(EvLoopStructureKind::Initial)?;
            let final_ = f(EvLoopStructureKind::Final)?;
            let before_ev_chasing = f(EvLoopStructureKind::PreEvChase(Iteration(1)))?;

            let cereal = EnergyPerAtom { initial, final_, before_ev_chasing };
            let value = ::serde_yaml::to_value(&cereal)?;
            make_nested_mapping(&["energy-per-atom"], value)
        });

        let summary = out.into_iter().fold(no_summary(), merge_summaries);
        ::serde_yaml::to_writer(self.create_file("summary.yaml")?, &summary)?;
    })}
}

fn phonopy_builder_from_settings(
    settings: &cfg::Phonons,
    lattice: &Lattice,
) -> PhonopyBuilder {
    let mut phonopy = {
        PhonopyBuilder::new()
            .symmetry_tolerance(settings.symmetry_tolerance)
            .conf("DISPLACEMENT_DISTANCE", format!("{:e}", settings.displacement_distance))
            .supercell_dim(settings.supercell.dim_for_unitcell(lattice))
    };
    if let cfg::PhononDispFinder::Phonopy { diag } = settings.disp_finder {
        phonopy = phonopy.diagonal_disps(diag);
    }
    phonopy
}

fn do_force_sets_at_disps_for_phonopy<P: AsPath + Send + Sync>(
    pot: &PotentialBuilder,
    threading: &cfg::Threading,
    disp_dir: &DirWithDisps<P>,
) -> FailResult<Vec<Vec<V3>>>
{Ok({
    use ::std::io::prelude::*;

    trace!("Computing forces at displacements");

    let counter = ::util::AtomicCounter::new();
    let num_displacements = disp_dir.displacements().len();
    let (initial_coords, meta) = disp_dir.superstructure();

    let force_sets = use_potential_maybe_with_rayon(
        pot,
        initial_coords,
        meta.sift(),
        (threading == &cfg::Threading::Rayon).into(),
        disp_dir.displaced_coord_sets().collect::<Vec<_>>(),
        move |diff_fn: &mut DiffFn<_>, meta, coords: Coords| FailOk({
            let i = counter.inc();
            eprint!("\rdisp {} of {}", i + 1, num_displacements);
            ::std::io::stderr().flush().unwrap();

            diff_fn.compute_force(&coords, meta)?
        }),
    )?;
    eprintln!();
    trace!("Done computing forces at displacements");
    force_sets
})}

fn do_force_sets_at_disps_for_sparse(
    pot: &PotentialBuilder,
    _threading: &cfg::Threading,
    displacements: &[(usize, V3)],
    coords: &Coords,
    meta: CommonMeta,
) -> FailResult<Vec<BTreeMap<usize, V3>>>
{Ok({
    use ::std::io::prelude::*;

    trace!("Computing forces at displacements");

    let mut disp_fn = pot.initialize_disp_fn(&coords, meta.sift())?;

    // this no longer has the option of using rayon because the speed gain from
    // disabling neighbor list updates in LAMMPS is far greater
    let force_sets = {
        displacements.iter()
            .enumerate()
            .map(|(i, &disp)| {
                eprint!("\rdisp {} of {}", i + 1, displacements.len());
                ::std::io::stderr().flush().unwrap();

                disp_fn.compute_sparse_force_delta(disp)
            })
            .collect::<Result<_, _>>()?
    };
    eprintln!();
    trace!("Done computing forces at displacements");
    force_sets
})}

// use a potential on many similar structures, possibly using Rayon.
//
// When rayon is not used, a single DiffFn is reused.
fn use_potential_maybe_with_rayon<Inputs, Input, F, Output>(
    pot: &PotentialBuilder,
    coords_for_initialize: &Coords,
    meta: CommonMeta,
    threading: Threading,
    inputs: Inputs,
    compute: F,
) -> FailResult<Vec<Output>>
where
    Input: Send,
    Output: Send,
    Inputs: IntoIterator<Item=Input>,
    Inputs: ::rayon::iter::IntoParallelIterator<Item=Input>,
    F: Fn(&mut DiffFn<CommonMeta>, CommonMeta, Input) -> FailResult<Output> + Sync + Send,
{ match threading {
    Threading::Parallel => {
        use ::rayon::prelude::*;
        let get_meta = meta.sendable();
        inputs.into_par_iter()
            .map(|x| compute(&mut pot.one_off(), get_meta(), x))
            .collect()
    },
    Threading::Serial => {
        // save the cost of repeated DiffFn initialization since we don't need Send.
        let mut diff_fn = pot.initialize_diff_fn(coords_for_initialize, meta.clone())?;
        inputs.into_iter()
            .map(|x| compute(&mut diff_fn, meta.clone(), x))
            .collect()
    },
}}

//-----------------------------------

// HACK:
// These are only valid for a hexagonal system represented
// with the [[a, 0], [-a/2, a sqrt(3)/2]] lattice convention
#[allow(unused)] const Q_GAMMA: V3 = V3([0.0, 0.0, 0.0]);
#[allow(unused)] const Q_K: V3 = V3([1f64/3.0, 1.0/3.0, 0.0]);
#[allow(unused)] const Q_K_PRIME: V3 = V3([2.0 / 3f64, 2.0 / 3f64, 0.0]);

//=================================================================

impl TrialDir {
    pub(crate) fn run_energy_surface(
        self,
        on_demand: Option<LammpsOnDemand>,
        settings: &cfg::EnergyPlotSettings,
        input: &PathDir,
    ) -> FailResult<()>
    {Ok({
        // support either a force dir or a bands dir as input
        let bands_dir = match DirWithBands::from_existing(input.to_path_buf()) {
            // accept a bands dir
            Ok(dir) => dir.boxed(),
            Err(e) => {
                // try computing gamma bands from a force dir
                use ::phonopy::MissingFileError;

                // bail out on anything other than MissingFileError
                // (this check is awkwardly written; I couldn't see a better way
                //  to handle borrow-checking correctly)
                let _ = e.downcast::<MissingFileError>()?;
                DirWithForces::from_existing(&input)?
                    .build_bands()
                    .eigenvectors(true)
                    .compute(&[Q_GAMMA])?
                    .boxed()
            },
        };

        let (coords, meta) = bands_dir.structure()?;
        let (evals, evecs) = bands_dir.eigensystem_at(Q_GAMMA)?;

        let plot_ev_indices = {
            use ::rsp2_tasks_config::EnergyPlotEvIndices::*;

            let (i, j) = match settings.ev_indices {
                Shear => {
                    // FIXME: This should just find layers, check that there's two
                    //        and then move one along x and y instead
                    panic!("energy plot using shear modes is no longer supported")
                },
                These(i, j) => (i, j),
            };

            // (in case of confusion about 0-based/1-based indices)
            trace!("X: Eigensolution {:>3}, frequency {}", i, evals[i]);
            trace!("Y: Eigensolution {:>3}, frequency {}", j, evals[j]);
            (i, j)
        };

        let get_real_ev = |i: usize| {
            let ev = evecs.0[i].as_real_checked();
            settings.normalization.normalize(ev)
        };

        let pot = PotentialBuilder::from_config_parts(
            Some(&self),
            on_demand,
            &settings.threading,
            &settings.lammps_update_style,
            &settings.lammps_processor_axis_mask,
            &settings.potential,
        );

        let [xmin, xmax] = settings.xlim;
        let [ymin, ymax] = settings.ylim;
        let [w, h] = settings.dim;
        let data = {
            ::cmd::integrate_2d::integrate_two_eigenvectors(
                (w, h),
                &coords.to_carts(),
                (xmin..xmax, ymin..ymax),
                (&get_real_ev(plot_ev_indices.0), &get_real_ev(plot_ev_indices.1)),
                {
                    use ::std::sync::atomic::{AtomicUsize, Ordering};
                    let counter = AtomicUsize::new(0);
                    let get_meta = meta.sendable();

                    move |pos| {FailOk({
                        let i = counter.fetch_add(1, Ordering::SeqCst);
                        // println!("{:?}", pos.flat().iter().sum::<f64>());

                        eprint!("\rdatapoint {:>6} of {}", i, w * h);
                        pot.one_off()
                            .compute_grad(
                                &coords.clone().with_carts(pos.to_vec()),
                                get_meta().sift(),
                            )?
                    })}
                }
            )?
        };
        eprintln!();

        let chunked: Vec<_> = data.chunks(w).collect();
        ::serde_json::to_writer_pretty(self.create_file("out.json")?, &chunked)?;
    })}
}

//=================================================================

// These were historically inherent methods, but the type was relocated to another crate
extension_trait!{
    SupercellSpecExt for SupercellSpec {
        fn dim_for_unitcell(&self, prim: &Lattice) -> [u32; 3] {
            match *self {
                SupercellSpec::Dim(d) => d,
                SupercellSpec::Target(targets) => {
                    let unit_lengths = prim.norms();
                    arr_from_fn(|k| {
                        (targets[k] / unit_lengths[k]).ceil().max(1.0) as u32
                    })
                },
            }
        }
    }
}

extension_trait! {
    NormalizationModeExt for NormalizationMode {
        fn norm(&self, ev: &[V3]) -> f64
        {
            let atom_rs = || ev.iter().map(V3::norm).collect::<Vec<_>>();

            match *self {
                NormalizationMode::CoordNorm => vnorm(ev.unvee().flat()),
                NormalizationMode::AtomMean => {
                    let rs = atom_rs();
                    rs.iter().sum::<f64>() / (rs.len() as f64)
                },
                NormalizationMode::AtomRms => {
                    let rs = atom_rs();
                    vnorm(&rs) / (rs.len() as f64).sqrt()
                },
                NormalizationMode::AtomMax => {
                    let rs = atom_rs();
                    rs.iter().cloned()
                        .max_by(|a, b| a.partial_cmp(b).expect("NaN?!")).expect("zero-dim ev?!")
                },
            }
        }

        fn normalize(&self, ev: &[V3]) -> Vec<V3>
        {
            let norm = self.norm(ev);

            let mut ev = ev.to_vec();
            for v in &mut ev {
                *v /= norm;
            }
            ev
        }
    }
}

//=================================================================

impl TrialDir {
    pub(crate) fn run_save_bands_after_the_fact(
        self,
        on_demand: Option<LammpsOnDemand>,
        settings: &Settings,
    ) -> FailResult<()>
    {Ok({
        let pot = PotentialBuilder::from_root_config(&self, on_demand, &settings);

        let (coords, meta) = self.read_stored_structure_data(&self.structure_path(EvLoopStructureKind::Final))?;

        let phonopy = phonopy_builder_from_settings(&settings.phonons, coords.lattice());

        // NOTE: there is no information needed from settings.phonons.eigensolver, so we can freely
        //       run this binary this even on computations that originally used sparse methods.
        PhonopyDiagonalizer {
            save_bands: Some(self.save_bands_dir()),
        }.create_bands_dir(&settings.threading, &pot, &phonopy, &coords, meta.sift())?;
    })}
}

impl TrialDir {
    pub fn save_bands_dir(&self) -> PathAbs
    {
        let path: PathArc = self.join(SAVE_BANDS_DIR).into();
        assert!(path.is_absolute());
        PathAbs::mock(path) // FIXME self.join ought to give a PathAbs to begin with
    }
}

//=================================================================

impl TrialDir {
    pub(crate) fn rerun_ev_analysis(
        self,
        on_demand: Option<LammpsOnDemand>,
        settings: &Settings,
        stored: StoredStructure,
    ) -> FailResult<()>
    {Ok({
        let pot = PotentialBuilder::from_root_config(&self, on_demand, &settings);

        // !!!!!!!!!!!!!!!!!
        //  FIXME FIXME BAD
        // !!!!!!!!!!!!!!!!!
        // The logic of constructing a Diagonalizer from a config is copied and pasted
        // from the code that uses do_ev_loop.  The only way to make this DRY is to
        // make a trait that simulates `for<T: EvLoopDiagonalizer> Fn(T) -> (B, T::ExtraOut)`
        // with a generic associated method
        //
        // (UPDATE: I tried that and it wasn't enough, because the main code couldn't recover
        //          the bands_dir without being able to dispatch based on the Diagonalizer type.
        //          Maybe we just need to give up on the associated type (the purpose of which
        //          is currently just to facilitate debugging a bit)).
        //
        let (evals, evecs) = {
            use self::cfg::PhononEigenSolver::*;

            // macro to simulate a generic closure.
            // (we can't use dynamic polymorphism due to the differing associated types)
            macro_rules! use_diagonalizer {
                ($diagonalizer:expr) => {{
                    $diagonalizer.do_post_relaxation_computations(
                        &self, settings, &*pot, &stored, false, None,
                    )
                }}
            }

            match settings.phonons.eigensolver {
                Phonopy { save_bands } => {
                    let save_bands = match save_bands {
                        true => Some(self.save_bands_dir()),
                        false => None,
                    };
                    let (evals, evecs, _) = {
                        use_diagonalizer!(PhonopyDiagonalizer { save_bands })?
                    };
                    (evals, evecs)
                },
                Sparse { shift_invert_attempts } => {
                    let (evals, evecs, _) = {
                        use_diagonalizer!(SparseDiagonalizer { shift_invert_attempts } )?
                    };
                    (evals, evecs)
                },
            }
        };
        // ^^^^^^^^^^^^^^^
        // !!!!!!!!!!!!!!!!!
        //  FIXME FIXME BAD
        // !!!!!!!!!!!!!!!!!

        trace!("============================");
        trace!("Finished diagonalization");

        trace!("Classifying eigensolutions");
        let classifications = acoustic_search::perform_acoustic_search(
            &pot, &evals, &evecs, &stored.coords, stored.meta().sift(), &settings.acoustic_search,
        )?;

        trace!("Computing eigensystem info");

        let ev_analysis = run_gamma_system_analysis(
            &stored.coords, stored.meta().sift(),
            &evals, &evecs, Some(classifications),
            true, // unfold bands
        )?;

        write_eigen_info_for_humans(&ev_analysis, &mut |s| FailOk(info!("{}", s)))?;

        write_ev_analysis_output_files(&self, &ev_analysis)?;
    })}
}

//=================================================================

pub(crate) fn run_sparse_analysis(
    structure: StoredStructure,
    evals: &[f64],
    evecs: &Basis3,
) -> FailResult<GammaSystemAnalysis>
{Ok({
    trace!("Computing eigensystem info");
    let ev_analysis = run_gamma_system_analysis(
        &structure.coords,
        structure.meta().sift(),
        &evals, &evecs,
        None, // ev_classifications
        true, // unfold_bands
    )?;

    write_eigen_info_for_humans(&ev_analysis, &mut |s| FailOk(info!("{}", s)))?;

    ev_analysis
})}

//=================================================================

pub(crate) fn run_plot_vdw(
    on_demand: Option<LammpsOnDemand>,
    pot: &cfg::Potential,
    z: f64,
    rs: &[f64],
) -> FailResult<()>
{Ok({
    use ::rsp2_structure::{CoordsKind};
    let threading = cfg::Threading::Lammps;

    let lammps_update_style = cfg::LammpsUpdateStyle::Fast { sync_positions_every: 1 };
    let processor_axis_mask = [true; 3];
    let pot = PotentialBuilder::from_config_parts(None, on_demand, &threading, &lammps_update_style, &processor_axis_mask, pot);

    let lattice = {
        let a = rs.iter().fold(0.0, |a, &b| f64::max(a, b)) + 20.0;
        let c = z + 20.0;
        Lattice::orthorhombic(a, a, c)
    };
    let (direction, direction_perp) = {
        // a randomly generated direction, but the same between runs
        let dir = V3([0.2268934438759319, 0.5877271759538497, 0.0]).unit();
        let perp = V3([dir[1], -dir[0], 0.0]); // dir rotated by 90
        (dir, perp)
    };
    let get_coords = |r: f64| {
        let rho = f64::sqrt(r * r - z * z);
        assert!(rho.is_finite());

        let pos = rho * direction + V3([0.0, 0.0, z]);
        Coords::new(lattice.clone(), CoordsKind::Carts(vec![V3::zero(), pos]))
    };

    let masses: meta::SiteMasses = vec![::common::default_element_mass(CARBON).unwrap(); 2].into();
    let elements: meta::SiteElements = vec![CARBON; 2].into();
    let bonds = None::<meta::FracBonds>;
    let meta = hlist![masses, elements, bonds];

    let mut diff_fn = pot.initialize_diff_fn(&get_coords(rs[0]), meta.sift())?;

    println!("# R Value Fpar Fperp Fz");
    for &r in rs {
        let coords = get_coords(r);
        let (value, grad) = diff_fn.compute(&coords, meta.sift())?;
        let f = -grad[1];
        let f_par = V3::dot(&f, &direction);
        let f_perp = V3::dot(&f, &direction_perp);
        println!("  {} {} {:e} {:e} {:e}", r, value, f_par, f_perp, f[2]);
    }
})}

pub(crate) fn run_converge_vdw(
    on_demand: Option<LammpsOnDemand>,
    pot: &cfg::Potential,
    z: f64,
    (r_min, r_max): (f64, f64),
) -> FailResult<()>
{Ok({
    use ::rsp2_structure::{CoordsKind};
    let threading = cfg::Threading::Lammps;

    let lammps_update_style = cfg::LammpsUpdateStyle::Fast { sync_positions_every: 1 };
    let processor_axis_mask = [true; 3];
    let pot = PotentialBuilder::from_config_parts(None, on_demand, &threading, &lammps_update_style, &processor_axis_mask, pot);

    let lattice = Lattice::orthorhombic(40.0, 40.0, 40.0);
    let direction = {
        // a randomly generated direction, but the same between runs
        V3([0.2268934438759319, 0.5877271759538497, 0.0]).unit()
    };
    let get_coords = |r: f64| {
        let rho = f64::sqrt(r * r - z * z);
        assert!(rho.is_finite());

        let pos = rho * direction + V3([0.0, 0.0, z]);
        Coords::new(lattice.clone(), CoordsKind::Carts(vec![V3::zero(), pos]))
    };

    let masses: meta::SiteMasses = vec![::common::default_element_mass(CARBON).unwrap(); 2].into();
    let elements: meta::SiteElements = vec![CARBON; 2].into();
    let bonds = None::<meta::FracBonds>;
    let meta = hlist![masses, elements, bonds];

    let mut diff_fn = pot.initialize_diff_fn(&get_coords(r_min), meta.sift())?;

    let value_1 = diff_fn.compute_value(&get_coords(r_min), meta.sift())?;
    let value_2 = diff_fn.compute_value(&get_coords(r_max), meta.sift())?;
    let d_value = value_2 - value_1;

    use ::rsp2_minimize::test::n_dee::{work};
    let path = work::PathConfig::Fixed(vec![
        get_coords(r_min).to_carts().flat().to_vec(),
        get_coords(r_max).to_carts().flat().to_vec(),
    ]).generate();
    println!("# Density Work DValue");
    for density in work::RefinementMode::Double.densities().take(20) {
        let work = work::compute_work_along_path(
            ::potential::Rsp2MinimizeDiffFnShim {
                ndim: 6,
                diff_fn: pot.initialize_flat_diff_fn(&get_coords(r_min), meta.sift())?,
            },
            &path.with_density(density),
        );
        println!("# {} {} {}", density, work, d_value);
    }
})}

//=================================================================

// FIXME refactor once it's working, this is way too long
pub(crate) fn run_dynmat_test(phonopy_dir: &PathDir) -> FailResult<()>
{Ok({
    // Make a supercell, and determine how our ordering of the supercell differs from phonopy.
    let forces_dir = DirWithForces::from_existing(phonopy_dir)?;
    let disp_dir = DirWithDisps::from_existing(phonopy_dir)?;
    let ::phonopy::Rsp2StyleDisplacements {
        super_coords, sc, perm_from_phonopy, ..
    } = disp_dir.rsp2_style_displacements()?;

    let (_, prim_meta) = disp_dir.primitive_structure()?;
    let space_group = disp_dir.symmetry()?;

    let space_group_deperms: Vec<_> = {
        ::rsp2_structure::find_perm::spacegroup_deperms(
            &super_coords,
            &space_group,
            1e-1, // FIXME should be slightly larger than configured tol,
                  //       but I forgot where that is stored.
        )?
    };

    let (force_sets, super_displacements): (Vec<_>, Vec<_>) = {
        let ::phonopy::ForceSets {
            force_sets: phonopy_force_sets,
            displacements: phonopy_super_displacements,
        } = forces_dir.force_sets()?;

        zip_eq!(phonopy_force_sets, phonopy_super_displacements)
            .map(|(fset, (phonopy_displaced, disp))| {
                let displaced = perm_from_phonopy.permute_index(phonopy_displaced);
                let fset = {
                    fset.into_iter().enumerate()
                        .map(|(atom, force)| (perm_from_phonopy.permute_index(atom), force))
                        .collect()
                };
                (fset, (displaced, disp))
            }).unzip()
    };

    let cart_rots: Vec<_> = {
        space_group.iter().map(|c| c.cart_rot()).collect()
    };

    trace!("Computing designated rows of force constants...");
    let force_constants = ::math::dynmat::ForceConstants::compute_required_rows(
        &super_displacements,
        &force_sets,
        &cart_rots,
        &space_group_deperms,
        &sc,
    ).unwrap();
    trace!("Done computing designated rows of force constants.");

    {
//        let dense = force_constants.permuted_by(&perm_to_phonopy).to_dense_matrix();
//        println!("{:?}", dense); // FINALLY: THE MOMENT OF TRUTH
    }

    {
        trace!("Computing dynamical matrix...");
        let our_dynamical_matrix = force_constants.gamma_dynmat(&sc, prim_meta.pick()).hermitianize();
        trace!("Done computing dynamical matrix.");
        println!("{:?}", our_dynamical_matrix.0.to_coo().map(|c| c.0).into_dense());

        trace!("Computing eigensolutions...");
        let shift_invert_attempts = 4;
        let (low, _low_basis) = our_dynamical_matrix.compute_negative_eigensolutions(shift_invert_attempts)?;
        let (high, _high_basis) = our_dynamical_matrix.compute_most_extreme_eigensolutions(3)?;
        trace!("Done computing eigensolutions...");

        println!("{:?}", low);
        println!("{:?}", high);
        let _ = our_dynamical_matrix;

    }
})}

//=================================================================

// #[allow(warnings)]
// pub fn make_force_sets(
//     conf: Option<&AsRef<Path>>,
//     poscar: &AsRef<Path>,
//     outdir: &AsRef<Path>,
// ) -> Result<()>
// {ok({
//     use ::rsp2_structure_io::poscar;
//     use ::std::io::BufReader;

//     let potential = panic!("TODO: potential in make_force_sets");
//     unreachable!();

//     let mut phonopy = PhonopyBuilder::new();
//     if let Some(conf) = conf {
//         phonopy = phonopy.conf_from_file(BufReader::new(open(conf)?))?;
//     }

//     let structure = poscar::load(open(poscar)?)?;

//     let pot = LammpsBuilder::new(&cfg::Threading::Lammps, &potential);

//     create_dir(&outdir)?;
//     {
//         // dumb/lazy solution to ensuring all output files go in the dir
//         let cwd_guard = push_dir(outdir)?;
//         GlobalLogger::default()
//             .path("rsp2.log")
//             .apply()?;

//         poscar::dump(create("./input.vasp")?, "", &structure)?;

//         let disp_dir = phonopy.displacements(&structure)?;
//         let force_sets = do_force_sets_at_disps(&pot, &cfg::Threading::Rayon, &disp_dir)?;
//         disp_dir.make_force_dir_in_dir(&force_sets, ".")?;

//         cwd_guard.pop()?;
//     }
// })}

//=================================================================

impl TrialDir {
    /// Used to figure out which iteration we're on when starting from the
    /// post-diagonalization part of the EV loop for sparse.
    pub(crate) fn find_iteration_for_ev_chase(&self) -> FailResult<Iteration> {
        use ::cmd::EvLoopStructureKind::*;

        for iteration in (1..).map(Iteration) {
            let pre_chase = self.structure_path(PreEvChase(iteration));
            let post_chase = self.structure_path(PostEvChase(iteration));
            let eigensols = self.eigensols_path(iteration);
            if !pre_chase.exists() {
                bail!("{}: does not exist", pre_chase.nice());
            }
            if !post_chase.exists() {
                if !eigensols.exists() {
                    bail!("\
                        {}: does not exist.  Did you perform diagonalization? \
                        (try the python module rsp2.cli.negative_modes)
                    ", eigensols.nice())
                }
                return Ok(iteration);
            }
        }
        unreachable!()
    }

    pub(crate) fn run_after_diagonalization(
        &self,
        on_demand: Option<LammpsOnDemand>,
        settings: &Settings,
        iteration: Iteration,
    ) -> FailResult<DidEvChasing> {
        use ::cmd::EvLoopStructureKind::*;
        use ::filetypes::Eigensols;

        let pot = PotentialBuilder::from_root_config(&self, on_demand, &settings);

        let (coords, meta) = self.read_stored_structure_data(&self.structure_path(PreEvChase(iteration)))?;
        let Eigensols {
            frequencies: evals,
            eigenvectors: evecs,
        } = Load::load(self.join(self.eigensols_path(iteration)))?;

        // this binary is only ever used with the sparse diagonalizer
        let allow_unfold_bands = false;

        let (_, coords, did_ev_chasing) = self.do_ev_loop_stuff_after_diagonalization(
            settings, &pot, meta.sift(), iteration,
            coords, &evals, &evecs,
            allow_unfold_bands,
        )?;

        if let DidEvChasing(true) = did_ev_chasing {
            // FIXME: This is confusing and side-effectful.
            //        File-saving should be done at the highest level possible, not in a function
            //        called from multiple places.
            let _coords_already_saved = self.do_ev_loop_stuff_before_diagonalization(
                settings, &pot, meta.sift(), Iteration(iteration.0 + 1), coords,
            )?;
        }

        {
            let mut f = self.create_file(format!("did-ev-chasing-{:02}", iteration))?;
            match did_ev_chasing {
                DidEvChasing(true) => writeln!(f, "1")?,
                DidEvChasing(false) => writeln!(f, "0")?,
            }
        }
        Ok(did_ev_chasing)
    }
}

//=================================================================

// Reads a POSCAR or layers.yaml into an intermediate form which can have its
// parameters optimized before producing a structure. (it also returns some other
// layer-related data).
//
// Be aware that layers.yaml files usually *require* optimization.
//
// A POSCAR will have its layers searched if the relevant config section is provided.
pub(crate) fn read_optimizable_structure(
    layer_cfg: Option<&cfg::LayerSearch>,
    mass_cfg: Option<&cfg::Masses>,
    file_format: StructureFileType,
    input: impl AsPath,
) -> FailResult<(
    // FIXME train wreck output
    // TODO could contain bonds read from .structure
    ScalableCoords,
    HList4<
        meta::SiteElements,
        meta::SiteMasses,
        Option<meta::SiteLayers>,
        Option<meta::LayerScMatrices>,
    >
)> {
    use ::meta::Layer;

    let input = input.as_path();

    let out_coords: ScalableCoords;
    let out_elements: meta::SiteElements;
    let out_masses: meta::SiteMasses;
    let out_layers: Option<meta::SiteLayers>;
    let out_sc_mats: Option<meta::LayerScMatrices>;
    match file_format {
        StructureFileType::Poscar => {
            use ::rsp2_structure_io::Poscar;

            let Poscar { coords, elements, .. } = Load::load(input.as_path())?;
            out_elements = elements.into();
            out_masses = masses_by_config(mass_cfg, out_elements.clone())?;

            if let Some(cfg) = layer_cfg {
                let layers = perform_layer_search(cfg, &coords)?;
                out_coords = ScalableCoords::from_layer_search_results(coords, cfg, &layers);
                out_layers = Some(layers.by_atom().into_iter().map(Layer).collect::<Vec<_>>().into());
                // We could do a primitive cell search, but anything using our results would have
                //  trouble interpreting our results if we chose a different cell from expected.
                // Thus, anything using sc matrices requires them to be supplied in advance.
                out_sc_mats = None;
            } else {
                out_coords = ScalableCoords::from_unlayered(coords);
                out_layers = None;
                out_sc_mats = None;
            }
        },
        StructureFileType::LayersYaml => {
            use ::rsp2_structure_io::layers_yaml::load;
            use ::rsp2_structure_io::layers_yaml::load_layer_sc_info;

            let layer_builder = load(PathFile::new(input)?.read()?)?;

            out_sc_mats = Some({
                load_layer_sc_info(PathFile::new(input)?.read()?)?
                    .into_iter()
                    .map(|(matrix, periods, _)| ScMatrix::new(&matrix, &periods))
                    .collect_vec()
                    .into()
            });

            out_layers = Some(layer_builder.atom_layers().into_iter().map(Layer).collect::<Vec<_>>().into());
            out_elements = vec![CARBON; layer_builder.num_atoms()].into();
            out_masses = masses_by_config(mass_cfg, out_elements.clone())?;
            out_coords = ScalableCoords::KnownLayers { layer_builder };

            if let Some(_) = layer_cfg {
                trace!("{} is in layers.yaml format, so layer-search config is ignored", input.nice())
            }
        },
        StructureFileType::StoredStructure => {
            let StoredStructure {
                coords, elements, layers, masses, layer_sc_matrices, ..
            } = Load::load(input.as_path())?;

            out_elements = elements;
            out_layers = layers;
            out_masses = masses;
            out_sc_mats = layer_sc_matrices;

            // (FIXME: just having the layer indices metadata isn't good enough; we need
            //         to be able to get contiguous layers, e.g. using rsp2_structure::Layers.
            //
            //         That said, I don't see any good reason to be optimizing layer seps
            //         on structures read in this format)
            out_coords = ScalableCoords::from_unlayered(coords);

            if let Some(_) = mass_cfg {
                trace!("{} is in directory format, so mass config is ignored", input.nice())
            }
            if let Some(_) = layer_cfg {
                trace!("{} is in directory format, so layer-search config is ignored", input.nice())
            }
        },
    }
    Ok((out_coords, hlist![out_elements, out_masses, out_layers, out_sc_mats]))
}

pub(crate) fn perform_layer_search(
    cfg: &cfg::LayerSearch,
    coords: &Coords,
) -> FailResult<LayersPerUnitCell>
{Ok({
    trace!("Finding layers...");
    let &cfg::LayerSearch {
        normal, threshold,
        count: expected_count,
    } = cfg;

    let layers = {
        ::rsp2_structure::layer::find_layers(coords, V3(normal), threshold)?
            .per_unit_cell()
            .expect("Structure is not layered?")
    };

    if let Some(expected) = expected_count {
        assert_eq!(
            layers.len() as u32, expected,
            "Layer count discovered does not match expected 'count' in config!"
        );
    }

    layers
})}

/// Implements the behavior of the `"masses"` config section.
///
/// When the section is omitted, default masses are used.
fn masses_by_config(
    cfg_masses: Option<&cfg::Masses>,
    elements: meta::SiteElements,
) -> FailResult<meta::SiteMasses>
{Ok({
    use ::meta::Mass;

    elements.iter().cloned()
        .map(|element| match cfg_masses {
            Some(cfg::Masses(map)) => {
                map.get(element.symbol())
                    .cloned()
                    .map(Mass)
                    .ok_or_else(|| {
                        format_err!("No mass in config for element {}", element.symbol())
                    })
            },
            None => ::common::default_element_mass(element),
        })
        .collect::<Result<Vec<_>, _>>()?.into()
})}

/// Heuristically accept either a trial directory, or a structure directory within one
pub(crate) fn resolve_trial_or_structure_path(
    path: &PathAbs,
    default_subdir: impl AsRef<OsStr>, // used when a trial directory is supplied
) -> FailResult<(TrialDir, StoredStructure)>
{Ok({
    let structure_name: OsString;
    let trial_path: PathDir;
    if StoredStructure::path_is_structure(&path) {
        let parent = path.parent().ok_or_else(|| format_err!("no parent for structure dir"))?;
        structure_name = path.file_name().unwrap().into();
        trial_path = PathDir::new(parent)?;
    } else {
        trial_path = PathDir::new(path)?;
        structure_name = default_subdir.as_ref().into();
    }

    let trial = TrialDir::from_existing(&trial_path)?;
    let structure = trial.read_stored_structure(structure_name)?;
    (trial, structure)
})}

impl TrialDir {
    pub fn structure_path(&self, kind: EvLoopStructureKind) -> PathBuf {
        self.join(match kind {
            EvLoopStructureKind::Initial => format!("initial.structure"),
            EvLoopStructureKind::Final => format!("final.structure"),
            EvLoopStructureKind::PreEvChase(n) => format!("ev-loop-{:02}.1.structure", n),
            EvLoopStructureKind::PostEvChase(n) => format!("ev-loop-{:02}.2.structure", n),
        })
    }

    pub fn eigensols_path(&self, iteration: Iteration) -> PathBuf
    { self.join(format!("ev-loop-modes-{:02}.json", iteration)) }

    pub fn uncompressed_gamma_dynmat_path(&self, iteration: Iteration) -> PathBuf
    { self.join(format!("gamma-dynmat-{:02}.json", iteration)) }

    pub fn gamma_dynmat_path(&self, iteration: Iteration) -> PathBuf
    { self.join(format!("gamma-dynmat-{:02}.npz", iteration)) }

    pub fn final_gamma_dynmat_path(&self) -> PathBuf
    { self.join("gamma-dynmat.npz") }
}
