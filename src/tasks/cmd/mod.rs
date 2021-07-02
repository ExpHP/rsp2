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

pub(crate) use crate::filetypes::stored_structure::StoredStructure;

pub(crate) use self::relaxation::DidEvChasing;
mod relaxation;
mod acoustic_search;
mod param_optimization;

pub(crate) mod python;

mod phonopy;

use crate::{FailResult, FailOk};
use rsp2_tasks_config::{self as cfg, Settings, SupercellSpec};
use crate::traits::{AsPath, Load, Save, save::Json};
use rsp2_lammps_wrap::LammpsOnDemand;
use rsp2_dynmat::{ForceConstants, DynamicalMatrix};

use crate::meta::{self, prelude::*};
use crate::util::ext_traits::{OptionResultExt, PathNiceExt};
use crate::math::{
    basis::{GammaBasis3, EvDirection},
    bands::{ScMatrix},
};
use self::acoustic_search::ModeKind;

use path_abs::{PathAbs, PathDir, FileRead};
use rsp2_structure::consts::CARBON;

use slice_of_array::prelude::*;
use rsp2_array_types::{V3, M33};
use rsp2_structure::{Coords, Lattice};
use rsp2_structure::{
    layer::LayersPerUnitCell,
    bonds::FracBonds,
    Element,
};

use rsp2_fs_util::{create, rm_rf, hard_link};

use std::{
    path::{PathBuf},
    io::{Write},
    ffi::{OsStr, OsString},
    collections::{BTreeMap},
    rc::{Rc},
    fmt,
};

use num_complex::Complex64;
use itertools::Itertools;
use crate::hlist_aliases::*;
use crate::potential::{PotentialBuilder, DiffFn, CommonMeta, EcoModeProof};

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum StructureFileType {
    Poscar,
    LayersYaml,
    StoredStructure,
    Xyz,
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
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result
    { fmt::Display::fmt(&self.0, f) }
}

// HACK
// used to simulate a sort of unwind on successful exiting.
//
// In case you haven't guessed, I've given up all hope on keeping rsp2-tasks maintainable.
// I need to rip out all of the legacy and/or experimental features I'm not using or start fresh.
#[derive(Debug, Fail)]
#[fail(display = "stopped after dynmat.  THIS IS NOT AN ACTUAL ERROR. THIS IS A DUMB HACK. \
You should not see this message.")]
pub(crate) struct StoppedEarly;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum StopAfter { Cg, Dynmat, DontStop }

impl TrialDir {
    pub(crate) fn run_relax_with_eigenvectors(
        self,
        on_demand: Option<LammpsOnDemand>,
        settings: &Settings,
        file_format: StructureFileType,
        input: &PathAbs,
        // shameful HACK
        stop_after: StopAfter,
    ) -> FailResult<()>
    {Ok({
        match (stop_after, &settings.phonons) {
            (StopAfter::Dynmat, None) |
            (StopAfter::DontStop, None) => bail!("`phonons` config section is required"),
            (StopAfter::Cg, None) => {},
            (_, Some(_)) => {},
        }

        let pot = <dyn PotentialBuilder>::from_root_config(Some(&self), on_demand, &settings)?;

        let (optimizable_coords, mut meta) = {
            read_optimizable_structure(
                settings.layer_search.as_ref(),
                settings.masses.as_ref(),
                file_format, input,
            )?
        };

        let original_coords = {
            // (can't reliably get bonds until the lattice parameter is correct)
            crate::cmd::param_optimization::optimize_layer_parameters(
                &settings.scale_ranges,
                &pot,
                optimizable_coords,
                meta.sift(),
            )?.construct()
        };

        // Compute the bonds only if they were not part of the input.
        trace!{"Computing intralayer bonds..."}
        let bonds: &mut Option<meta::FracBonds> = meta.get_mut();
        if bonds.is_none() {
            *bonds = settings.bond_radius.map(|bond_radius| FailOk({
                Rc::new(FracBonds::compute(&original_coords, bond_radius)?)
            })).fold_ok()?
        }

        self.write_stored_structure(
            &self.structure_path(EvLoopStructureKind::Initial),
            "Initial structure (after lattice optimization)",
            &original_coords, meta.sift(),
        )?;

        let (coords, ev_analysis) = {
            let (coords, stuff) = {
                self.do_main_ev_loop(
                    settings, &*pot, original_coords, meta.sift(),
                    stop_after,
                )?
            };

            // HACK: Put last gamma dynmat at a predictable path.
            rm_rf(self.join("gamma-dynmat.json"))?;
            if let Some((ev_analysis, final_iteration)) = stuff {
                hard_link(
                    self.gamma_dynmat_path(final_iteration),
                    self.final_gamma_dynmat_path(),
                )?;
                (coords, Some(ev_analysis))
            } else {
                (coords, None)
            }
        };

        self.write_stored_structure(
            &self.structure_path(EvLoopStructureKind::Final),
            "Final structure",
            &coords, meta.sift(),
        )?;

        if let Some(ev_analysis) = ev_analysis {
            write_eigen_info_for_machines(&ev_analysis, self.create_file("eigenvalues.final")?)?;

            write_ev_analysis_output_files(&self, &ev_analysis)?;
            self.write_summary_file(settings, &*pot, &ev_analysis)?;
        }
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
            raman_tensor: Vec<M33>,
            average_3d: Vec<f64>,
            backscatter: Vec<f64>,
        }
        use crate::math::bond_polarizability::LightPolarization::*;
        serde_json::to_writer(FileWrite::create(dir.join("raman.json"))?, &Output {
            frequency: frequency.0.to_vec(),
            raman_tensor: raman.0.iter().map(|t| t.tensor().clone()).collect(),
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

        serde_json::to_writer(FileWrite::create(dir.join("unfold.json"))?, &Output {
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

fn do_compute_dynmat(
    trial_dir: Option<&TrialDir>,
    settings: &Settings,
    phonons_settings: &cfg::Phonons,
    pot: &dyn PotentialBuilder,
    qpoint_pfrac: V3,
    prim_coords: &Coords,
    prim_meta: HList4<
        meta::SiteElements,
        meta::SiteMasses,
        Option<meta::SiteLayers>,
        Option<meta::FracBonds>,
    >,
) -> FailResult<DynamicalMatrix>
{
    if phonons_settings.analytic_hessian {
        return do_compute_dynmat_with_hessian(
            settings, phonons_settings, pot, qpoint_pfrac, prim_coords, prim_meta,
        );
    }

    let displacement_distance = phonons_settings.displacement_distance.expect("missing displacement-distance should have been caught sooner");
    let symprec = phonons_settings.symmetry_tolerance.expect("missing symmetry-tolerance should have been caught sooner");

    // Here exists a great deal of logic for dealing with supercells.
    // Ideally it would be factored out somehow to be less in your face,
    // so that this function doesn't have so many responsibilities.
    //
    // (but last time I tried to do so, I found the resulting function signature too obnoxious)
    //
    // For now, we just make heavy use of scoping to keep the number of names in scope smallish.
    let (ref super_coords, ref sc) = {
        let sc_dim = phonons_settings.supercell.dim_for_unitcell(prim_coords.lattice());
        trace!("Constructing supercell (dim: {:?})", sc_dim);
        rsp2_structure::supercell::diagonal(sc_dim).build(prim_coords)
    };

    let cart_ops = if symprec == 0.0 {
        trace!("Not computing symmetry (symmetry-tolerance = 0)");
        info!(" Spacegroup: P1 (1)");
        info!("Point group: 1");
        vec![CartOp::eye()]
    } else {
        use self::python::SpgDataset;

        trace!("Computing symmetry");
        let atom_types: Vec<u32> = {
            let elements: meta::SiteElements = prim_meta.pick();
            elements.iter().map(|e| e.atomic_number()).collect()
        };

        let spg = SpgDataset::compute(prim_coords, &atom_types, symprec)?;
        info!(" Spacegroup: {} ({})", spg.international_symbol, spg.spacegroup_number);
        info!("Point group: {}", spg.point_group);

        spg.cart_ops()
    };

    let mut phonopy_info = None;
    let prim_displacements = match phonons_settings.disp_finder {
        cfg::PhononDispFinder::Phonopy { diag: _ } => {
            let the_phonopy_info = self::phonopy::phonopy_displacements(
                &phonons_settings, prim_coords, prim_meta.sift(), sc, super_coords,
            )?;

            let prim_displacements = the_phonopy_info.prim_displacements.clone();

            match the_phonopy_info.spacegroup_op_count.cmp(&cart_ops.len()) {
                // note: this case should never fire because we give phonopy a smaller symprec
                std::cmp::Ordering::Greater => panic!(
                    "Phonopy found more spacegroup operations than rsp2! ({} > {}). \
                    This makes it impossible to reconstruct the force constants.",
                    the_phonopy_info.spacegroup_op_count,
                    cart_ops.len(),
                ),
                std::cmp::Ordering::Less => warn!(
                    "Phonopy found fewer spacegroup operations than rsp2! ({} < {}). \
                    This is surprising, but shouldn't cause any problems...",
                    the_phonopy_info.spacegroup_op_count,
                    cart_ops.len(),
                ),
                std::cmp::Ordering::Equal => {},
            }

            phonopy_info = Some(the_phonopy_info);

            prim_displacements
        },
        cfg::PhononDispFinder::Rsp2 { ref directions } => {
            trace!("Computing deperms in primitive cell");

            let prim_deperms = do_compute_deperms(&phonons_settings, &prim_coords, &cart_ops)?;
            let prim_stars = crate::math::stars::compute_stars(&prim_deperms);

            let prim_displacements = crate::math::displacements::compute_displacements(
                directions,
                cart_ops.iter().map(|c| {
                    c.int_rot(prim_coords.lattice()).expect("bad operator from spglib!?")
                }),
                &prim_stars,
                &prim_coords,
                displacement_distance,
            );

            prim_displacements
        },
    };

    let ref rsp2_displaced_site_cells = vec![ForceConstants::DESIGNATED_CELL; prim_displacements.len()];
    let super_displacements: Vec<_> = {
        zip_eq!(&prim_displacements, rsp2_displaced_site_cells)
            .map(|(&(prim, disp), &cell)| {
                let atom = sc.atom_from_cell(prim, cell);
                (atom, disp)
            })
            .collect()
    };

    let super_meta = replicate_meta_for_force_constants(settings, &super_coords, &sc, prim_meta.sift())?;

    trace!("Computing deperms in supercell");
    let super_deperms = do_compute_deperms(&phonons_settings, &super_coords, &cart_ops)?;

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

    let debug_files_root = match trial_dir {
        Some(d) => d.as_path().to_owned(),
        None => std::env::current_dir()?,
    };

    // Log target for comparison to phonopy
    if log_enabled!(target: "rsp2_tasks::special::phonopy_force_sets", log::Level::Trace) {
        use self::phonopy::{PhonopyDisplacements, PhonopyForceSets};

        if let Some(phonopy_info) = &phonopy_info {
            trace!("Creating FORCE_SETS file for rsp2_tasks::special::phonopy_force_sets");
            let w = create(debug_files_root.join("FORCE_SETS"))?;

            let PhonopyDisplacements {
                phonopy_super_displacements, coperm_from_phonopy, ..
            } = phonopy_info;

            if let Err(e) = (PhonopyForceSets {
                phonopy_super_displacements,
                coperm_from_phonopy,
                rsp2_displaced_site_cells,
                sc,
            }).write(w, &force_sets) {
                warn!("Error writing phonopy force sets: {}", e);
            }
        } else {
            warn_once!("\
                rsp2_tasks::special::phonopy_force_sets tracing was enabled, but cannot \
                do anything because the phonopy disp-finder was not used.\
            ");
        }
    }

    if log_enabled!(target: "rsp2_tasks::special::visualize_sparse_forces", log::Level::Trace) {
        trace!("Creating force log files for rsp2_tasks::special::visualize_sparse_forces=trace");
        visualize_sparse_force_sets(
            &debug_files_root,
            &super_coords,
            &super_displacements,
            &super_deperms,
            &cart_ops,
            &force_sets,
        )?;
    }

    trace!("Computing sparse force constants");
    let force_constants = ForceConstants::compute_required_rows(
        &super_displacements,
        &force_sets,
        &cart_rots,
        &super_deperms,
        &sc,
    )?;
    let force_constants = impose_sum_rule(phonons_settings, &sc, force_constants);

    if log_enabled!(target: "rsp2_tasks::special::phonopy_force_constants", log::Level::Trace) {
        if let Some(phonopy_info) = &phonopy_info {
            trace!("Creating rsp2-fcs.json file for rsp2_tasks::special::phonopy_force_constants");
            let deperm_to_phonopy = &phonopy_info.coperm_from_phonopy; // inverse of inverse
            let phonopy_fcs = {
                force_constants.to_super_force_constants_with_zeroed_rows(&sc)
                    .permuted_by(deperm_to_phonopy)
            };
            if let Err(e) = Json(phonopy_fcs.to_dense_matrix()).save(debug_files_root.join("rsp2-fcs.json")) {
                warn!("Error writing force constants debug file: {}", e);
            }
        } else {
            warn_once!("\
                rsp2_tasks::special::phonopy_force_constants tracing was enabled, but cannot \
                do anything because the phonopy disp-finder was not used.\
            ");
        }
    }

    trace!("Computing sparse dynamical matrix");
    let dynmat = {
        let qpoint_cart = qpoint_pfrac * &prim_coords.lattice().reciprocal();
        let masses: meta::SiteMasses = prim_meta.pick();
        let masses = masses.iter().map(|&meta::Mass(m)| m).collect::<Vec<_>>();
        force_constants
            .dynmat_at_cart_q(&super_coords, qpoint_cart, &sc, &masses)
            .hermitianize()
    };
    trace!("Done computing dynamical matrix");

    // Log target for tests/resources/force-constants fc files
    if log_enabled!(target: "rsp2_tasks::special::fc_test_files", log::Level::Trace) {
        trace!("Creating force log files for rsp2_tasks::special::fc_test_files=trace");
        save_force_sets_for_tests(
            &debug_files_root,
            &prim_coords,
            prim_meta.sift(),
            &super_coords,
            &sc,
            &cart_ops,
            &super_displacements,
            &force_sets,
            &dynmat,
        );
    }

    Ok(dynmat)
}

// Vastly simpler than do_compute_dynmat
fn do_compute_dynmat_with_hessian(
    settings: &Settings,
    phonons_settings: &cfg::Phonons,
    pot: &dyn PotentialBuilder,
    qpoint_pfrac: V3,
    prim_coords: &Coords,
    prim_meta: HList4<
        meta::SiteElements,
        meta::SiteMasses,
        Option<meta::SiteLayers>,
        Option<meta::FracBonds>,
    >,
) -> FailResult<DynamicalMatrix>
{
    trace!("Constructing supercell");
    let (ref super_coords, ref sc) = {
        let sc_dim = phonons_settings.supercell.dim_for_unitcell(prim_coords.lattice());
        rsp2_structure::supercell::diagonal(sc_dim).build(prim_coords)
    };

    let super_meta = replicate_meta_for_force_constants(settings, &super_coords, &sc, prim_meta.sift())?;

    let force_constants = do_force_constants_using_hessian(pot, &super_coords, super_meta.sift(), &sc)?;
    let force_constants = impose_sum_rule(phonons_settings, &sc, force_constants);

    trace!("Computing sparse dynamical matrix");
    let dynmat = {
        let qpoint_cart = qpoint_pfrac * &prim_coords.lattice().reciprocal();
        let masses: meta::SiteMasses = prim_meta.pick();
        let masses = masses.iter().map(|&meta::Mass(m)| m).collect::<Vec<_>>();
        force_constants
            .dynmat_at_cart_q(&super_coords, qpoint_cart, &sc, &masses)
            .hermitianize()
    };
    trace!("Done computing dynamical matrix");

    Ok(dynmat)
}

fn replicate_meta_for_force_constants(
    settings: &Settings,
    super_coords: &Coords,
    sc: &SupercellToken,
    prim_meta: HList4<
        meta::SiteElements,
        meta::SiteMasses,
        Option<meta::SiteLayers>,
        Option<meta::FracBonds>,
    >,
) -> FailResult<HList4<
    meta::SiteElements,
    meta::SiteMasses,
    Option<meta::SiteLayers>,
    Option<meta::FracBonds>,
>> {
    // macro to generate a closure, because generic closures don't exist
    macro_rules! f {
        () => { |x: Rc<[_]>| -> Rc<[_]> {
            sc.replicate(&x[..]).into()
        }};
    }

    // deriving these from the primitive cell bonds is not worth the trouble
    trace!("Computing bonds in supercell");
    let super_bonds = settings.bond_radius.map(|bond_radius| FailOk({
        Rc::new(FracBonds::compute(&super_coords, bond_radius)?)
    })).fold_ok()?;

    Ok(prim_meta.clone().map(hlist![
        f!(),
        f!(),
        |opt: Option<_>| opt.map(f!()),
        |_: Option<meta::FracBonds>| { super_bonds },
    ]))
}

fn impose_sum_rule(
    phonon_settings: &cfg::Phonons,
    sc: &SupercellToken,
    force_constants: ForceConstants,
) -> ForceConstants {
    match phonon_settings.sum_rule {
        None => force_constants,

        Some(cfg::PhononSumRule::TranslationalLikePhonopy { level })  => {
            trace!("Imposing translational acoustic sum rule");
            warn!("\
                Using the implementation based on phonopy's. This effectively causes the force \
                constants to become dense. Other parts of rsp2 may generate spurious warnings!\
            ");
            force_constants.impose_translational_invariance(&sc, level)
        }
    }
}

fn do_compute_deperms(
    phonon_settings: &cfg::Phonons,
    coords: &Coords,
    cart_ops: &[CartOp],
) -> FailResult<Vec<Perm>> {
    rsp2_structure::find_perm::spacegroup_deperms(
        coords,
        cart_ops,
        // larger than SYMPREC because the coords we see may may be slightly
        // different from what spglib saw, but not so large that we risk pairing
        // the wrong atoms
        //
        // the case of symmetry_tolerance = 0 is explicitly supported by the method
        phonon_settings.symmetry_tolerance.expect("(BUG!) should have been caught earlier") * 3.0,
    )
}

fn do_diagonalize_dynmat(
    phonons_settings: &cfg::Phonons,
    dynmat: DynamicalMatrix,
    // don't let MPI processes compete with python's threads
    _: EcoModeProof<'_>,
) -> FailResult<(Vec<f64>, GammaBasis3)>
{Ok({
    {
        let max_size = (|(a, b)| a * b)(dynmat.0.dim);
        let nnz = dynmat.0.nnz();
        let density = nnz as f64 / max_size as f64;
        trace!("nnz: {} out of {} blocks (matrix density: {:.3e})", nnz, max_size, density);
    }
    trace!("Diagonalizing dynamical matrix");
    let (freqs, evecs) = {
        match phonons_settings.eigensolver {
            cfg::PhononEigensolver::Phonopy(cfg::AlwaysFail(never, _)) => match never {},
            cfg::PhononEigensolver::Rsp2 { .. } => panic!("(BUG!) setting phonons.eigensolver is not normalized!"),
            cfg::PhononEigensolver::Dense {} => {
                // FIXME: the location of this function is misleading;
                //        it doesn't actually use eigsh.
                python::scipy_eigsh::compute_eigensolutions_dense_gamma(&dynmat)
            },
            cfg::PhononEigensolver::Sparse { how_many, shift_invert_attempts } => {
                python::scipy_eigsh::compute_negative_eigensolutions_gamma(
                    &dynmat,
                    how_many,
                    shift_invert_attempts,
                )?
            },
        }
    };
    trace!("Done diagonalizing dynamical matrix");
    (freqs, evecs)
})}

impl TrialDir {
    fn write_animations(
        &self,
        animate_settings: &cfg::Animate,
        coords: &Coords,
        meta: HList2<
            meta::SiteElements,
            meta::SiteMasses,
        >,
        iteration: Iteration,
        bad_directions: impl IntoIterator<Item=(f64, EvDirection)>,
        all_directions: impl IntoIterator<Item=(f64, EvDirection)>,
    ) -> FailResult<()> {Ok({
        let &cfg::Animate { ref format, ref which, max_count } = animate_settings;

        let mode_info: Vec<(f64, EvDirection)> = {
            match which {
                cfg::AnimateWhich::All => all_directions.into_iter().collect(),
                cfg::AnimateWhich::Negative => bad_directions.into_iter().collect(),
            }
        };
        let mode_info = match max_count {
            None => &mode_info[..],
            Some(count) => &mode_info[..usize::min(count, mode_info.len())],
        };

        let out_path = self.animation_path(iteration, format);
        match format {
            cfg::AnimateFormat::VSim { } => {
                use rsp2_structure_io::v_sim;

                let mut metadata = v_sim::AsciiMetadata::new();
                for &(frequency, ref direction) in mode_info {
                    metadata.add_phonon(v_sim::Phonon {
                        qpoint_frac: V3::zero(),
                        energy: frequency,
                        displacements: {
                            zip_eq!(&direction.real, &direction.imag)
                                .map(|(re, im)| V3::from_fn(|i| Complex64::new(re[i], im[i])))
                                .collect()
                        },
                    });
                }

                let elements: meta::SiteElements = meta.pick();
                v_sim::VSimAscii {
                    comment: "V_Sim ascii input generated by rsp2",
                    coords,
                    elements,
                    metadata,
                }.save(out_path)?
            },
        }
    })}
}

use rsp2_soa_ops::{Perm, Permute};
use rsp2_structure::CartOp;
use rsp2_structure::supercell::SupercellToken;

// FIXME incorrect for nontrivial supercells. Should use primitive stars and translate
//       the displaced atom to the correct image after rotation. (this would be easiest to
//       do using the functionality in the ForceConstants code)
fn visualize_sparse_force_sets(
    debug_files_root: impl AsPath,
    super_coords: &Coords,
    super_disps: &[(usize, V3)],
    super_deperms: &[Perm],
    cart_ops: &[CartOp],
    force_sets: &[BTreeMap<usize, V3>],
) -> FailResult<()>
{Ok({
    use rsp2_structure::consts;
    use rsp2_structure_io::Poscar;

    let subdir = debug_files_root.join("visualize-forces");
    if let Ok(dir) = PathDir::new(&subdir) {
        dir.remove_all()?;
    }
    let subdir = PathDir::create(subdir)?;

    let super_stars = crate::math::stars::compute_stars(super_deperms);

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

fn save_force_sets_for_tests(
    debug_files_root: impl AsPath,
    prim_coords: &Coords,
    prim_meta: HList1<meta::SiteMasses>,
    super_coords: &Coords,
    sc: &rsp2_structure::supercell::SupercellToken,
    cart_ops: &[CartOp],
    super_displacements: &[(usize, V3<f64>)],
    force_sets: &Vec<BTreeMap<usize, V3<f64>>>,
    dynmat: &DynamicalMatrix,
) {
    #[derive(Serialize, Deserialize)]
    #[serde(rename_all = "kebab-case")]
    struct Primitive {
        structure: Coords,
        masses: Vec<meta::Mass>,
        cart_ops: Vec<CartOp>,
    }

    #[derive(Serialize, Deserialize)]
    struct ForceSets {
        #[serde(rename = "sc-dims")] sc_dims: [u32; 3],
        #[serde(rename = "structure")] super_coords: Coords,
        #[serde(rename = "force-sets")] force_sets: Vec<Vec<(usize, V3)>>, // [disp] -> [(super, V3)]
        #[serde(rename = "displacements")] super_displacements: Vec<(usize, V3)>, // [disp] -> (super, V3)
    }

    #[derive(Serialize, Deserialize)]
    struct DenseDynmat {
        real: Vec<Vec<M33>>,
        imag: Vec<Vec<M33>>,
    }

    let prim_masses: meta::SiteMasses = prim_meta.pick();

    let primitive = Primitive {
        structure: prim_coords.clone(),
        masses: prim_masses.to_vec(),
        cart_ops: cart_ops.to_vec(),
    };

    let force_sets = ForceSets {
        sc_dims: sc.periods(),
        super_coords: super_coords.clone(),
        force_sets: force_sets.clone().into_iter().map(|row| row.into_iter().collect()).collect(),
        super_displacements: super_displacements.to_vec(),
    };

    let dense_dynmat = DenseDynmat {
        real: dynmat.0.to_coo().map(|c| c.0).into_dense(),
        imag: dynmat.0.to_coo().map(|c| c.1).into_dense(),
    };

    if let Err(e) = Json(primitive).save(debug_files_root.join("primitive.json")) {
        warn!("{}", e);
    }
    if let Err(e) = Json(force_sets).save(debug_files_root.join("super.json")) {
        warn!("{}", e);
    }
    if let Err(e) = Json(dense_dynmat).save(debug_files_root.join("dynmat.json")) {
        warn!("{}", e);
    }
}

// wrapper around the gamma_system_analysis module which handles all the newtype conversions
fn do_gamma_system_analysis(
    coords: &Coords,
    meta: HList5<
        meta::SiteElements,
        meta::SiteMasses,
        Option<meta::SiteLayers>,
        Option<meta::LayerScMatrices>,
        Option<meta::FracBonds>,
    >,
    freqs: &[f64],
    evecs: &GammaBasis3,
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
        ev_frequencies: Some(EvFrequencies(freqs.to_vec())),
        ev_eigenvectors: Some(EvEigenvectors(evecs.clone())),
        bonds: cart_bonds.map(Bonds),
        request_to_unfold_bands: if unfold_bands { Some(RequestToUnfoldBands) } else { None },
    }.compute()
}

fn write_eigen_info_for_humans(
    analysis: &GammaSystemAnalysis,
    writeln: &mut dyn FnMut(String) -> FailResult<()>,
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
        pot: &dyn PotentialBuilder,
        ev_analysis: &GammaSystemAnalysis,
    ) -> FailResult<()> {Ok({
        use crate::ui::cfg_merging::{make_nested_mapping, no_summary, merge_summaries};

        #[derive(Serialize)]
        #[serde(rename_all = "kebab-case")]
        struct EnergyPerAtom {
            initial: f64,
            #[serde(rename = "final")]
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
            let value = serde_yaml::to_value(&cereal)?;
            make_nested_mapping(&["energy-per-atom"], value)
        });

        let summary = out.into_iter().fold(no_summary(), merge_summaries);
        serde_yaml::to_writer(self.create_file("summary.yaml")?, &summary)?;
    })}
}

// FIXME: rename.
//
// Also notice that the rsp2 eigensolver apparently never did disps in parallel,
// because I wanted to make it reuse the DispFn.
fn do_force_sets_at_disps_for_sparse(
    pot: &dyn PotentialBuilder,
    _threading: &cfg::Threading,
    displacements: &[(usize, V3)],
    coords: &Coords,
    meta: CommonMeta,
) -> FailResult<Vec<BTreeMap<usize, V3>>>
{Ok({
    use std::io::prelude::*;

    trace!("Computing forces at displacements");

    let mut disp_fn = pot.initialize_disp_fn(&coords, meta.sift())?;

    // this no longer has the option of using rayon because the speed gain from
    // disabling neighbor list updates in LAMMPS is far greater
    let force_sets = {
        displacements.iter()
            .enumerate()
            .map(|(i, &disp)| {
                eprint!("\rdisp {} of {}", i + 1, displacements.len());
                std::io::stderr().flush().unwrap();

                disp_fn.compute_sparse_force_delta(disp)
            })
            .collect::<Result<_, _>>()?
    };
    eprintln!();
    trace!("Done computing forces at displacements");
    force_sets
})}

fn do_force_constants_using_hessian(
    pot: &dyn PotentialBuilder,
    coords: &Coords,
    meta: CommonMeta,
    sc: &SupercellToken,
) -> FailResult<ForceConstants>
{Ok({
    trace!("Computing analytic hessian");

    let mut ddiff_fn = match pot.initialize_pairwise_ddiff_fn(&coords, meta.sift())? {
        Some(ddiff_fn) => ddiff_fn,
        None => bail!("The chosen potential does not implement an analytic Hessian!"),
    };

    let (_, items) = ddiff_fn.compute(coords, meta)?;

    let primitive_atoms = sc.atom_primitive_atoms();
    let cells = sc.atom_cells();

    let (mut val, mut row, mut col) = (vec![], vec![], vec![]);
    for (bond_grad, hessian) in items {
        let super_from = bond_grad.minus_site;
        let super_to = bond_grad.plus_site;
        let prim_from = primitive_atoms[super_from];
        let prim_to = primitive_atoms[super_to];
        if cells[super_from] == ForceConstants::DESIGNATED_CELL {
            row.push(rsp2_dynmat::PrimI(prim_from));
            col.push(rsp2_dynmat::SuperI(super_from));
            val.push(hessian);

            row.push(rsp2_dynmat::PrimI(prim_from));
            col.push(rsp2_dynmat::SuperI(super_to));
            val.push(-hessian);
        }
        if cells[super_to] == ForceConstants::DESIGNATED_CELL {
            row.push(rsp2_dynmat::PrimI(prim_to));
            col.push(rsp2_dynmat::SuperI(super_to));
            val.push(hessian);

            row.push(rsp2_dynmat::PrimI(prim_to));
            col.push(rsp2_dynmat::SuperI(super_from));
            val.push(-hessian);
        }
    }

    let dim = (sc.num_primitive_atoms(), sc.num_supercell_atoms());
    ForceConstants(rsp2_sparse::RawCoo { dim, row, col, val }.to_csr())
})}

//=================================================================

pub struct EnergySurfaceArgs {
    density: usize,
    extend_border: bool,
    layer: meta::Layer,
}

impl crate::ui::cli_deserialize::CliDeserialize for EnergySurfaceArgs {
    fn _augment_clap_app<'a, 'b>(app: clap::App<'a, 'b>) -> clap::App<'a, 'b> {
        app.args(&[
            arg!( density [--density]=NPOINTS "number of points along each lattice vector"),
            arg!( extend_border [--extend-border] "\
                Include an extra point in each direction around the chosen range. \
                (these points are not counted in the density). Useful to ensure there are values \
                at all points along the edges if the data is to be resampled in some way.\
            "),
            arg!( layer [--layer]=LAYER "select which layer moves"),
        ])
    }

    fn _resolve_args(m: &clap::ArgMatches<'_>) -> FailResult<Self> {
        Ok(EnergySurfaceArgs {
            density: m.value_of("density").unwrap_or("100").parse()?,
            extend_border: m.is_present("extend_border"),
            layer: meta::Layer(m.value_of("layer").unwrap_or("0").parse()?),
        })
    }
}

pub(crate) fn run_shear_plot(
    on_demand: Option<LammpsOnDemand>,
    settings: &cfg::EnergyPlotSettings,
    structure: StoredStructure,
    plot_args: EnergySurfaceArgs,
    output_path: impl AsPath,
) -> FailResult<()>
{Ok({
    use rsp2_array_types::M22;

    let meta = structure.meta();
    let coords = structure.coords;

    let EnergySurfaceArgs { density, extend_border, layer: translated_layer } = plot_args;

    let pot = <dyn PotentialBuilder>::from_config_parts(
        None,
        on_demand,
        &settings.threading,
        &settings.lammps,
        &settings.potential,
    )?;

    let lattice_matrix_22 = {
        let matrix = coords.lattice().matrix();
        if (0..2).any(|k| matrix[2][k] != 0.0 || matrix[k][2] != 0.0) {
            bail!("Structure must be planar in xy plane.");
        }

        M22::from_fn(|r, c| matrix[r][c])
    };

    let site_layers: meta::SiteLayers = match meta.pick() {
        None => bail!("The structure for a shear plot must store layers."),
        Some(x) => x,
    };

    let mask: Vec<bool> = {
        site_layers.iter().map(|&x| x == translated_layer).collect()
    };

    // Vector that translates a layer by a lattice basis vector.
    let get_translation_vector = |k: usize| {
        let lattice_vector = coords.lattice().vectors()[k];

        let mut out = vec![V3::zero(); coords.len()];
        for (i, &mask_bit) in mask.iter().enumerate() {
            if mask_bit {
                out[i] = lattice_vector;
            }
        }
        out
    };

    let a_density = density;
    let b_density = density;
    let a_range = 0.0..1.0;
    let b_range = 0.0..1.0;
    let data = {
        crate::cmd::integrate_2d::integrate_two_directions(
            [a_density, b_density],
            &coords.to_carts(),
            [a_range, b_range],
            [extend_border, extend_border],
            [&get_translation_vector(0), &get_translation_vector(1)],
            {
                use std::sync::atomic::{AtomicUsize, Ordering};
                let coords = coords.clone();
                let counter = AtomicUsize::new(0);

                let mut diff_fn = pot.initialize_diff_fn(&coords, meta.sift())?;

                move |pos| {FailOk({
                    let i = counter.fetch_add(1, Ordering::SeqCst);
                    // println!("{:?}", pos.flat().iter().sum::<f64>());

                    eprint!("\rdatapoint {:>6} of {}", i, a_density * b_density);
                    diff_fn.compute_grad(
                        &coords.clone().with_carts(pos.to_vec()),
                        meta.sift(),
                    )?
                })}
            }
        )?
    };
    eprintln!();

    #[derive(Serialize)]
    #[serde(rename_all = "kebab-case")]
    struct Output {
        lattice: M22,
        index: Vec<[i32; 2]>,
        cart_translation: Vec<[f64; 2]>,
        energy_per_atom: Vec<f64>,
    }

    let output = Output {
        lattice: lattice_matrix_22,
        index: data.indices,
        cart_translation: {
            data.coords.iter()
                .map(|&[a, b]| {
                    let V3([x, y, z]) = V3([a, b, 0.0]) * coords.lattice();
                    assert_eq!(z, 0.0);
                    [x, y]
                })
                .collect()
        },
        energy_per_atom: data.values.iter().map(|v| v / (coords.len() as f64)).collect(),
    };

    Json(&output).save(output_path)?;
})}

//=================================================================

// These were historically inherent methods, but the type was relocated to another crate
extension_trait!{
    SupercellSpecExt for SupercellSpec {
        fn dim_for_unitcell(&self, prim: &Lattice) -> [u32; 3] {
            match *self {
                SupercellSpec::Dim(d) => d,
                SupercellSpec::Target(targets) => {
                    let unit_lengths = prim.norms();
                    V3::from_fn(|k| {
                        (targets[k] / unit_lengths[k]).ceil().max(1.0) as u32
                    }).0
                },
            }
        }
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
        let pot = <dyn PotentialBuilder>::from_root_config(Some(&self), on_demand, &settings)?;

        let phonons_settings = match &settings.phonons {
            Some(x) => x,
            None => bail!("`phonons` settings is required to do EV analysis"),
        };

        let qpoint = V3::zero();
        let dynmat = do_compute_dynmat(
            Some(&self), settings, phonons_settings,
            &*pot, qpoint, &stored.coords, stored.meta().sift(),
        )?;
        // Don't write the dynamical matrix; unclear where to put it.
        let (freqs, evecs) = pot.eco_mode(|eco_proof| {
            do_diagonalize_dynmat(&phonons_settings, dynmat, eco_proof)
        })?;

        trace!("============================");
        trace!("Finished diagonalization");

        trace!("Classifying eigensolutions");
        let classifications = acoustic_search::perform_acoustic_search(
            &pot, &freqs, &evecs, &stored.coords, stored.meta().sift(), &settings.acoustic_search,
        )?;

        trace!("Computing eigensystem info");

        let ev_analysis = do_gamma_system_analysis(
            &stored.coords, stored.meta().sift(),
            &freqs, &evecs, Some(classifications),
            true, // unfold bands
        )?;

        write_eigen_info_for_humans(&ev_analysis, &mut |s| FailOk(info!("{}", s)))?;

        write_ev_analysis_output_files(&self, &ev_analysis)?;
    })}
}

//=================================================================

pub(crate) fn run_sparse_analysis(
    structure: StoredStructure,
    freqs: &[f64],
    evecs: &GammaBasis3,
) -> FailResult<GammaSystemAnalysis>
{Ok({
    trace!("Computing eigensystem info");
    let ev_analysis = do_gamma_system_analysis(
        &structure.coords,
        structure.meta().sift(),
        &freqs, &evecs,
        None, // ev_classifications
        true, // unfold_bands
    )?;

    write_eigen_info_for_humans(&ev_analysis, &mut |s| FailOk(info!("{}", s)))?;

    ev_analysis
})}

pub(crate) fn run_dynmat_analysis(
    settings: &Settings,
    structure: StoredStructure,
    on_demand: Option<LammpsOnDemand>,
    dynmat: DynamicalMatrix,
) -> FailResult<GammaSystemAnalysis>
{
    eco_mode_without_potential(settings, on_demand, |eco_proof| Ok({
        let phonons_settings = match &settings.phonons {
            Some(x) => x,
            None => bail!("`phonons` config section is required for rsp2-dynmat-analysis"),
        };

        let (freqs, evecs) = do_diagonalize_dynmat(phonons_settings, dynmat, eco_proof)?;

        trace!("Computing eigensystem info");
        let ev_analysis = do_gamma_system_analysis(
            &structure.coords,
            structure.meta().sift(),
            &freqs, &evecs,
            None,  // ev_classifications
            false, // unfold_bands
        )?;

        write_eigen_info_for_humans(&ev_analysis, &mut |s| FailOk(info!("{}", s)))?;

        ev_analysis
    }))
}

//=================================================================

pub(crate) fn run_plot_vdw(
    on_demand: Option<LammpsOnDemand>,
    pot: &cfg::ValidatedPotential,
    z: f64,
    rs: &[f64],
) -> FailResult<()>
{Ok({
    use rsp2_structure::{CoordsKind};
    let threading = cfg::Threading::Lammps;

    let lammps = cfg::Lammps {
        update_style: cfg::LammpsUpdateStyle::Fast { sync_positions_every: 1 }.into(),
        processor_axis_mask: [true; 3].into(),
    };
    let pot = <dyn PotentialBuilder>::from_config_parts(None, on_demand, &threading, &lammps, pot)?;

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

    let masses: meta::SiteMasses = vec![crate::common::default_element_mass(CARBON).unwrap(); 2].into();
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
    pot: &cfg::ValidatedPotential,
    z: f64,
    (r_min, r_max): (f64, f64),
) -> FailResult<()>
{Ok({
    use rsp2_structure::{CoordsKind};
    let threading = cfg::Threading::Lammps;

    let lammps = cfg::Lammps {
        update_style: cfg::LammpsUpdateStyle::Fast { sync_positions_every: 1 }.into(),
        processor_axis_mask: [true; 3].into(),
    };
    let pot = <dyn PotentialBuilder>::from_config_parts(None, on_demand, &threading, &lammps, pot)?;

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

    let masses: meta::SiteMasses = vec![crate::common::default_element_mass(CARBON).unwrap(); 2].into();
    let elements: meta::SiteElements = vec![CARBON; 2].into();
    let bonds = None::<meta::FracBonds>;
    let meta = hlist![masses, elements, bonds];

    let mut diff_fn = pot.initialize_diff_fn(&get_coords(r_min), meta.sift())?;

    let value_1 = diff_fn.compute_value(&get_coords(r_min), meta.sift())?;
    let value_2 = diff_fn.compute_value(&get_coords(r_max), meta.sift())?;
    let d_value = value_2 - value_1;

    use rsp2_minimize::test::n_dee::{work};
    let path = work::PathConfig::Fixed(vec![
        get_coords(r_min).to_carts().flat().to_vec(),
        get_coords(r_max).to_carts().flat().to_vec(),
    ]).generate();
    println!("# Density Work DValue");
    for density in work::RefinementMode::Double.densities().take(20) {
        let work = work::compute_work_along_path(
            crate::potential::DiffFnWorkShim {
                ndim: 6,
                diff_fn: pot.initialize_cg_diff_fn(&get_coords(r_min), meta.sift())?,
            },
            &path.with_density(density),
        );
        println!("# {} {} {}", density, work, d_value);
    }
})}

//=================================================================

pub(crate) fn run_single_force_computation(
    on_demand: Option<LammpsOnDemand>,
    settings: &Settings,
    poscar: rsp2_structure_io::Poscar,
) -> FailResult<Vec<V3>>
{Ok({
    let rsp2_structure_io::Poscar { comment: _, coords, elements } = poscar;
    let elements: meta::SiteElements = elements.into();
    let masses = masses_by_config(settings.masses.as_ref(), elements.clone())?;

    let meta = hlist![elements, masses];
    let meta = meta.prepend({
        settings.bond_radius.map(|bond_radius| FailOk({
            Rc::new(FracBonds::compute(&coords, bond_radius)?)
        })).fold_ok()?
    });

    let pot = <dyn PotentialBuilder>::from_root_config(None, on_demand, &settings)?;

    pot.one_off().compute_force(&coords, meta.sift())?
})}

//=================================================================

pub(crate) fn run_layer_mode_frequencies(
    on_demand: Option<LammpsOnDemand>,
    settings: &Settings,
    structure: StoredStructure,
    step: f64,
    lattice_vector: Option<usize>,
) -> FailResult<()>
{Ok({
    use rsp2_minimize::numerical;
    use crate::util::{only_unique_value, OnlyUniqueResult};

    if let Some(lattice_vector) = lattice_vector {
        assert!(lattice_vector < 3, "--lattice-vector must be 0, 1, or 2");
    }

    let meta = structure.meta();
    let original_coords = structure.coords;

    let shift_layer_0 = |axis: usize, amount: f64, is_lattice_vector: bool| FailOk({
        let layers: meta::SiteLayers = match meta.pick() {
            Some(layers) => layers,
            None => bail!("structure must have layers"),
        };

        let displacement = match is_lattice_vector {
            true => original_coords.lattice().vectors()[axis].unit() * amount,
            false => {
                let mut displacement = V3::zero();
                displacement[axis] = amount;
                displacement
            },
        };

        let mut carts = original_coords.to_carts();
        for (cart, layer) in zip_eq!(&mut carts, &layers[..]) {
            if layer == &meta::Layer(0) {
                *cart += displacement;
            }
        }

        original_coords.with_carts(carts)
    });

    // this is what they do in 10.1016/j.physleta.2019.05.025
    let masses: meta::SiteMasses = meta.pick();
    let reduced_mass = match only_unique_value(masses.iter()) {
        OnlyUniqueResult::Ok(meta::Mass(x)) => 0.5 * x,
        OnlyUniqueResult::Conflict(_, _) => panic!("all atoms must have equal mass"),
        OnlyUniqueResult::NoValues => unreachable!(),
    };

    let pot = <dyn PotentialBuilder>::from_root_config(None, on_demand, &settings)?;
    let mut diff_fn = pot.initialize_diff_fn(&original_coords, meta.sift())?;

    warn!("Don't quote these values, only compare them!  (I'm not sure about the prefactor...)");
    let diff_2_to_wavenumber = |diff_2: f64| {
        // FIXME really not sure about the prefactor
        let sqrt_eigenvalue_to_thz = 15.6333043006705; // = sqrt(eV/amu)/angstrom/(2*pi)/THz
        let thz_to_wavenumber = 33.3564095198152; // = THz / (c / cm)
        let wavenumber = f64::sqrt(diff_2.abs() / reduced_mass) / (2.0 * std::f64::consts::PI) * sqrt_eigenvalue_to_thz * thz_to_wavenumber;
        diff_2.signum() * wavenumber
    };
    match lattice_vector {
        None => {
            let is_lattice_vector = false;
            let wavenumbers = V3::try_from_fn(|axis| FailOk({
                let diff_2 = numerical::try_diff_2(
                    step,
                    Some(numerical::DerivativeKind::Stencil(5)),
                    0.0,
                    |distance| {
                        let coords = shift_layer_0(axis, distance, is_lattice_vector)?;
                        diff_fn.compute_value(&coords, meta.sift())
                    },
                )?;
                diff_2_to_wavenumber(diff_2)
            }))?;
            serde_json::to_writer(std::io::stdout(), &wavenumbers)?;
            println!();
        },

        Some(axis) => {
            let is_lattice_vector = true;
            let diff_2 = numerical::try_diff_2(
                step,
                Some(numerical::DerivativeKind::Stencil(5)),
                0.0,
                |distance| {
                    let coords = shift_layer_0(axis, distance, is_lattice_vector)?;
                    diff_fn.compute_value(&coords, meta.sift())
                },
            )?;
            let wavenumber = diff_2_to_wavenumber(diff_2);
            serde_json::to_writer(std::io::stdout(), &wavenumber)?;
            println!();
        },
    }
})}

//=================================================================

pub(crate) fn run_dynmat_at_q(
    on_demand: Option<LammpsOnDemand>,
    settings: &Settings,
    qpoint_frac: V3,
    structure: StoredStructure,
) -> FailResult<DynamicalMatrix> {
    let pot = <dyn PotentialBuilder>::from_root_config(None, on_demand, &settings)?;

    let phonons_settings = match &settings.phonons {
        Some(x) => x,
        None => bail!("`phonons` config section is required to compute dynamical matrices"),
    };

    let meta = structure.meta();
    let coords = structure.coords;

    do_compute_dynmat(None, settings, phonons_settings, &pot, qpoint_frac, &coords, meta.sift())
}

//=================================================================

impl TrialDir {
    /// Used to figure out which iteration we're on when starting from the
    /// post-diagonalization part of the EV loop for sparse.
    pub(crate) fn find_iteration_for_ev_chase(&self, will_diagonalize: bool) -> FailResult<Iteration> {
        use crate::cmd::EvLoopStructureKind::*;

        for iteration in (1..).map(Iteration) {
            let pre_chase = self.structure_path(PreEvChase(iteration));
            let post_chase = self.structure_path(PostEvChase(iteration));
            let eigensols = self.eigensols_path(iteration);
            if !pre_chase.exists() {
                bail!("{}: does not exist", pre_chase.nice());
            }
            if !post_chase.exists() {
                if !(eigensols.exists() || will_diagonalize) {
                    bail!("\
                        {}: does not exist, and --diagonalize not supplied. \
                        Did you perform diagonalization? \
                        (try the python module rsp2.cli.negative_modes)\
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
        prev_iteration: Iteration,
        will_diagonalize: bool,
    ) -> FailResult<DidEvChasing> {
        use crate::cmd::EvLoopStructureKind::*;
        use crate::filetypes::Eigensols;

        let phonons_settings = match &settings.phonons {
            Some(x) => x,
            None => bail!("`rsp2-run-after-diagonalization` cannot be used without a `phonons:` config section"),
        };

        let pot = <dyn PotentialBuilder>::from_root_config(Some(&self), on_demand, &settings)?;

        let (coords, meta) = self.read_stored_structure_data(&self.structure_path(PreEvChase(prev_iteration)))?;

        let (freqs, evecs) = {
            if will_diagonalize {
                trace!("Diagonalizing due to --diagonalize.");
                pot.eco_mode(|proof| {
                    let dynmat = DynamicalMatrix::load(self.join(self.gamma_dynmat_path(prev_iteration)))?;
                    do_diagonalize_dynmat(phonons_settings, dynmat, proof)
                })?
            } else {
                let Eigensols {
                    frequencies, eigenvectors,
                } = Load::load(self.join(self.eigensols_path(prev_iteration)))?;

                let eigenvectors = eigenvectors.into_gamma_basis3().ok_or_else(|| {
                    failure::err_msg("expected real eigensols!")
                })?;

                (frequencies, eigenvectors)
            }
        };

        let (_, mut coords, did_ev_chasing) = self.do_ev_loop_stuff_after_diagonalization(
            settings, &pot, meta.sift(), prev_iteration,
            coords, &freqs, &evecs,
        )?;

        {
            let mut f = self.create_file(format!("did-ev-chasing-{:02}", prev_iteration))?;
            match did_ev_chasing {
                DidEvChasing(true) => writeln!(f, "1")?,
                DidEvChasing(false) => writeln!(f, "0")?,
            }
        }

        let next_iteration = Iteration(prev_iteration.0 + 1);
        if let DidEvChasing(true) = did_ev_chasing {
            coords = self.do_ev_loop_stuff_before_dynmat(
                settings, &pot, meta.sift(), Some(next_iteration), coords,
            )?;
        }

        let qpoint = V3::zero();
        let dynmat = do_compute_dynmat(Some(self), settings, phonons_settings, &pot, qpoint, &coords, meta.sift())?;
        dynmat.save(self.gamma_dynmat_path(next_iteration))?;

        Ok(did_ev_chasing)
    }
}

//=================================================================

pub enum LayerScMode { Auto, Assign, Multiply, None }

pub fn run_make_supercell(
    structure: StoredStructure,
    dims_str: &str,
    layer_sc_mode: LayerScMode,
    output: impl AsPath,
) -> FailResult<()> {
    let StoredStructure {
        title, mut coords, mut elements, mut layers, mut masses,
        mut layer_sc_matrices, frac_bonds,
    } = structure;

    if let Some(_) = frac_bonds {
        // TODO: support this properly.
        warn!("\
            Supercells of bond graphs are not yet implemented, so the created supercell will be \
            missing a bond graph.  (don't worry too much about this; rsp2 will typically \
            generate a new bond graph when run on the output).\
        ");
    };

    let sc_dim = parse_sc_dims_argument(dims_str)?;
    let (super_coords, sc) = rsp2_structure::supercell::diagonal(sc_dim).build(&coords);
    coords = super_coords;

    elements = sc.replicate(&elements).into();
    masses = sc.replicate(&masses).into();
    layers = layers.as_mut().map(|x| sc.replicate(&x).into());
    layer_sc_matrices = match (layer_sc_mode, layer_sc_matrices) {
        (LayerScMode::None, _) => None,
        (LayerScMode::Multiply, None) => {
            bail!("--layer-scs=multiply requires existing layer SC matrices.")
        },
        (LayerScMode::Auto, Some(old_mats)) |
        (LayerScMode::Multiply, Some(old_mats)) => Some({
            old_mats.into_iter().map(|old| old.multiply_diagonal(&V3(sc_dim)))
                .collect::<Vec<_>>().into()
        }),
        (LayerScMode::Auto, None) |
        (LayerScMode::Assign, _) => {
            let layers = match layers {
                None => bail!("--layer-scs=assign requires layer indices to be stored."),
                Some(ref layers) => layers,
            };

            let num_layers = layers.iter().max().unwrap().0 as usize + 1;
            Some({
                std::iter::repeat(ScMatrix::from_diagonal(&V3(sc_dim))).take(num_layers)
                    .collect::<Vec<_>>().into()
            })
        },
    };

    StoredStructure {
        title, coords, elements, layers, masses,
        layer_sc_matrices, frac_bonds,
    }.save(output)
}

fn parse_sc_dims_argument(arg: &str) -> FailResult<[u32; 3]> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Cereal {
        Scalar(i32),
        Vector([i32; 3]),
        Matrix([[i32; 3]; 3])
    }

    match serde_json::from_str(arg)? {
        Cereal::Scalar(x) if x <= 0 => bail!("A scalar supercell must be non-negative."),
        Cereal::Scalar(x) => Ok([x as u32; 3]),
        Cereal::Vector(v) => {
            if v.iter().any(|&x| x <= 0) {
                bail!("A vector supercell must consist of positive integers.")
            } else {
                let [a, b, c] = v;
                Ok([a as u32, b as u32, c as u32])
            }
        },
        Cereal::Matrix(_) => bail!("General matrix supercells are not yet implemented."),
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
    HList5<
        meta::SiteElements,
        meta::SiteMasses,
        Option<meta::SiteLayers>,
        Option<meta::LayerScMatrices>,
        Option<meta::FracBonds>,
    >,
)> {
    use crate::meta::Layer;

    let input = input.as_path();

    let out_coords: ScalableCoords;
    let out_elements: meta::SiteElements;
    let out_masses: meta::SiteMasses;
    let out_layers: Option<meta::SiteLayers>;
    let out_sc_mats: Option<meta::LayerScMatrices>;
    let out_bonds: Option<meta::FracBonds>;
    match file_format {
        StructureFileType::Xyz => {
            use rsp2_structure_io::Xyz;

            let Xyz { carts, elements, .. } = Load::load(input.as_path())?;
            out_elements = elements.into();
            out_masses = masses_by_config(mass_cfg, out_elements.clone())?;

            let vacuum_sep = 30.0;
            let coords = Coords::from_molecule(&carts, vacuum_sep);

            if let Some(cfg) = layer_cfg {
                let layers = perform_layer_search(cfg, &coords)?;
                // Technically no scaling should ever be needed for XYZ files,
                // but this is the type we return...
                out_coords = ScalableCoords::from_layer_search_results(coords, cfg, &layers);
                out_layers = Some(layers.by_atom().into_iter().map(Layer).collect::<Vec<_>>().into());
                // XYZ is not periodic
                out_sc_mats = None;
            } else {
                out_coords = ScalableCoords::from_unlayered(coords);
                out_layers = None;
                out_sc_mats = None;
            }
            out_bonds = None;
        },
        StructureFileType::Poscar => {
            use rsp2_structure_io::Poscar;

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
            out_bonds = None;
        },
        StructureFileType::LayersYaml => {
            use rsp2_structure_io::layers_yaml::load;
            use rsp2_structure_io::layers_yaml::load_layer_sc_info;

            let layer_builder = load(FileRead::open(input)?)?;

            out_sc_mats = Some({
                load_layer_sc_info(FileRead::open(input)?)?
                    .into_iter()
                    .map(|(matrix, periods, _)| ScMatrix::new(&matrix, &periods))
                    .collect_vec()
                    .into()
            });

            out_layers = Some(layer_builder.atom_layers().into_iter().map(Layer).collect::<Vec<_>>().into());
            //println!("{:?}",layer_builder.atoms);
            let mut elements = vec![];
            for atom in layer_builder.atoms.clone(){
                elements.push(Element::get_from_symbol(&atom));
            }
            out_elements = elements.into();
            out_masses = masses_by_config(mass_cfg, out_elements.clone())?;
            out_coords = ScalableCoords::KnownLayers { layer_builder };
            out_bonds = None; // Determine bonds AFTER parameter optimization.

            if let Some(_) = layer_cfg {
                trace!("{} is in layers.yaml format, so layer-search config is ignored", input.nice())
            }
        },
        StructureFileType::StoredStructure => {
            let StoredStructure {
                coords, elements, layers, masses, layer_sc_matrices, frac_bonds, ..
            } = Load::load(input.as_path())?;

            out_elements = elements;
            out_layers = layers;
            out_masses = masses;
            out_sc_mats = layer_sc_matrices;
            out_bonds = frac_bonds;

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
    Ok((out_coords, hlist![out_elements, out_masses, out_layers, out_sc_mats, out_bonds]))
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
        rsp2_structure::layer::find_layers(coords, V3(normal), threshold)?
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
pub(crate) fn masses_by_config(
    cfg_masses: Option<&cfg::Masses>,
    elements: meta::SiteElements,
) -> FailResult<meta::SiteMasses>
{Ok({
    use crate::meta::Mass;

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
            None => crate::common::default_element_mass(element),
        })
        .collect::<Result<Vec<_>, _>>()?.into()
})}

// Run a callback in eco mode without needing to create a PotentialBuilder.
fn eco_mode_without_potential<B, F>(
    settings: &Settings,
    on_demand: Option<LammpsOnDemand>,
    continuation: F,
) -> FailResult<B>
where F: FnOnce(EcoModeProof<'_>) -> FailResult<B>,
{
    // can't use rsp2_lammps_wrap::potential::None due to Meta type mismatch
    //
    // FIXME: This is dumb; creating a dummy potential just so we can make a builder
    // so we can call this method. LammpsOnDemand should expose an `eco_mode` method.
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
    pub struct NoPotential;
    impl rsp2_lammps_wrap::Potential for NoPotential {
        type Meta = CommonMeta;

        fn atom_types(&self, _: &Coords, _: &Self::Meta) -> Vec<rsp2_lammps_wrap::AtomType>
        { unreachable!() }

        fn init_info(&self, _: &Coords, _: &Self::Meta) -> rsp2_lammps_wrap::InitInfo
        { unreachable!() }

        fn molecule_ids(&self, _: &Coords, _: &Self::Meta) -> Option<Vec<usize>>
        { unreachable!() }
    }

    let trial_dir = None;
    let pot: &dyn PotentialBuilder = &crate::potential::lammps::Builder::new(
        trial_dir, on_demand, &settings.threading, &settings.lammps,
        NoPotential,
    )?;

    pot.eco_mode(|eco_proof| continuation(eco_proof))
}

/// Heuristically accept either a trial directory, or a structure directory within one
pub(crate) fn resolve_trial_or_structure_path(
    path: &PathAbs,
    default_subdir: impl AsRef<OsStr>, // used when a trial directory is supplied
) -> FailResult<(TrialDir, StoredStructure)>
{Ok({
    let structure_name: OsString;
    let trial_path: PathDir;
    if StoredStructure::path_is_structure(&path) {
        let parent = path.as_path().parent().ok_or_else(|| format_err!("no parent for structure dir"))?;
        structure_name = path.as_path().file_name().unwrap().into();
        trial_path = PathDir::new(parent)?;
    } else {
        trial_path = PathDir::new(path)?;
        structure_name = default_subdir.as_ref().into();
    }

    let trial = TrialDir::from_existing(trial_path.as_path())?;
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

    pub fn snapshot_structure_path(&self) -> PathBuf
    { self.join("snapshot.structure") }

    pub fn eigensols_path(&self, iteration: Iteration) -> PathBuf
    { self.join(format!("ev-loop-modes-{:02}.json", iteration)) }

    pub fn modified_settings_path(&self, iteration: Iteration) -> PathBuf
    { self.join(format!("ev-loop-modes-{:02}.yaml", iteration)) }

    pub fn gamma_dynmat_path(&self, iteration: Iteration) -> PathBuf
    { self.join(format!("gamma-dynmat-{:02}.npz", iteration)) }

    pub fn final_gamma_dynmat_path(&self) -> PathBuf
    { self.join("gamma-dynmat.npz") }

    pub fn animation_path(&self, iteration: Iteration, format: &cfg::AnimateFormat) -> PathBuf
    { match format {
        cfg::AnimateFormat::VSim {} => self.join(format!("ev-loop-modes-{:02}.ascii", iteration)),
    }}
}
