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

use crate::{FailResult, FailOk};
use rsp2_tasks_config::{self as cfg, Settings, NormalizationMode, SupercellSpec};
use crate::traits::{AsPath, Load, Save};
use rsp2_lammps_wrap::LammpsOnDemand;

use crate::meta::{self, prelude::*};
use crate::util::ext_traits::{OptionResultExt, PathNiceExt};
use crate::math::{
    basis::{Basis3, EvDirection},
    bands::{ScMatrix},
    dynmat::{ForceConstants},
};
use self::acoustic_search::ModeKind;

use path_abs::{PathAbs, PathFile, PathDir};
use rsp2_structure::consts::CARBON;
use rsp2_slice_math::{vnorm};

use slice_of_array::prelude::*;
use rsp2_array_utils::arr_from_fn;
use rsp2_array_types::{V3, Unvee};
use rsp2_structure::{Coords, Lattice};
use rsp2_structure::{
    layer::LayersPerUnitCell,
    bonds::FracBonds,
};

use rsp2_fs_util::{rm_rf, hard_link};

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
use crate::potential::{PotentialBuilder, DiffFn, CommonMeta};
use crate::traits::save::Json;

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
        let pot = PotentialBuilder::from_root_config(&self, on_demand, &settings)?;

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
            crate::cmd::param_optimization::optimize_layer_parameters(
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

        let (coords, ev_analysis) = {
            let (coords, ev_analysis, final_iteration) = {
                self.do_main_ev_loop(
                    settings, &*pot, original_coords, meta.sift(),
                    stop_after_dynmat,
                )?
            };

            // HACK: Put last gamma dynmat at a predictable path.
            let final_iteration = final_iteration.expect("ev-loop should have iterations!");
            rm_rf(self.join("gamma-dynmat.json"))?;
            hard_link(
                self.gamma_dynmat_path(final_iteration),
                self.final_gamma_dynmat_path(),
            )?;

            (coords, ev_analysis)
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
        use crate::math::bond_polarizability::LightPolarization::*;
        serde_json::to_writer(FileWrite::create(dir.join("raman.json"))?, &Output {
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

    fn do_post_relaxation_computations(
        &self,
        settings: &Settings,
        pot: &dyn PotentialBuilder,
        stored: &StoredStructure,
        stop_after_dynmat: bool,
        iteration: Option<Iteration>,
    ) -> FailResult<(Vec<f64>, Basis3, Option<Iteration>)>
    {Ok({
        // FIXME: A great deal of logic in here exists for dealing with supercells,
        //        which are pointless when we only compute at the gamma point.
        //
        //        I don't want to tear those considerations out since it was difficult to write,
        //        but it ought to be factored out somehow to be less in your face, and especially
        //        so that it isn't in a function that is clearly related to relaxation.

        let prim_coords = &stored.coords;
        let prim_meta: HList4<
            meta::SiteElements,
            meta::SiteMasses,
            Option<meta::SiteLayers>,
            Option<meta::FracBonds>,
        > = stored.meta().sift();

        let compute_deperms = |coords: &_, cart_ops: &_| {
            rsp2_structure::find_perm::spacegroup_deperms(
                coords,
                cart_ops,
                // larger than SYMPREC because the coords we see may may be slightly
                // different from what spglib saw, but not so large that we risk pairing
                // the wrong atoms
                settings.phonons.symmetry_tolerance * 3.0,
            )
        };

        let (super_coords, prim_displacements, sc, cart_ops) = match settings.phonons.disp_finder {
            cfg::PhononDispFinder::Phonopy(cfg::AlwaysFail(never, _)) => match never { },
            cfg::PhononDispFinder::Rsp2 { ref directions } => {
                use self::python::SpgDataset;

                let sc_dim = settings.phonons.supercell.dim_for_unitcell(prim_coords.lattice());
                let (super_coords, sc) = rsp2_structure::supercell::diagonal(sc_dim).build(prim_coords);

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
                let prim_stars = crate::math::stars::compute_stars(&prim_deperms);

                let prim_displacements = crate::math::displacements::compute_displacements(
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

        // Generate supercell metadata by repeating the unit cell metadata.
        let super_meta = {
            // macro to generate a closure, because generic closures don't exist
            macro_rules! f {
                () => { |x: Rc<[_]>| -> Rc<[_]> {
                    sc.replicate(&x[..]).into()
                }};
            }

            // deriving these from the primitive cell bonds is not worth the trouble
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

        if log_enabled!(target: "rsp2_tasks::special::visualize_sparse_forces", log::Level::Trace) {
            visualize_sparse_force_sets(
                self,
                &super_coords,
                &super_displacements,
                &super_deperms,
                &cart_ops,
                &force_sets,
            )?;
        }

        trace!("Computing sparse force constants");
        let force_constants = crate::math::dynmat::ForceConstants::compute_required_rows(
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
            Json(dynmat.cereal()).save(self.uncompressed_gamma_dynmat_path(iteration))?;

            crate::cmd::python::convert::dynmat(
                self.uncompressed_gamma_dynmat_path(iteration),
                self.gamma_dynmat_path(iteration),
                crate::cmd::python::convert::Mode::Delete,
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
        let (freqs, evecs) = {
            pot.eco_mode(|| { // don't let MPI processes compete with python's threads
                match settings.phonons.eigensolver {
                    cfg::PhononEigensolver::Phonopy(cfg::AlwaysFail(never, _)) => match never {},
                    cfg::PhononEigensolver::Rsp2 { dense: true, .. } => {
                        dynmat.compute_eigensolutions_dense()
                    },
                    cfg::PhononEigensolver::Rsp2 { dense: false, how_many, shift_invert_attempts } => {
                        dynmat.compute_negative_eigensolutions(
                            how_many,
                            shift_invert_attempts,
                        )
                    },
                }
            })?
        };
        trace!("Done diagonalizing dynamical matrix");
        (freqs, evecs, iteration)
    })}

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
        let cfg::Animate { format, which } = animate_settings;

        let mode_info: Vec<(f64, EvDirection)> = {
            match which {
                cfg::AnimateWhich::All => all_directions.into_iter().collect(),
                cfg::AnimateWhich::Negative => bad_directions.into_iter().collect(),
            }
        };

        let out_path = self.animation_path(iteration, format);
        match format {
            cfg::AnimateFormat::VSim { } => {
                use rsp2_structure_io::v_sim;

                let mut metadata = v_sim::AsciiMetadata::new();
                for (frequency, direction) in mode_info {
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

// HACK
// used to simulate a sort of unwind on successful exiting.
// Take a look at the places where it's used and try not to throw up.
//
// In case you haven't guessed, I've given up all hope on keeping rsp2-tasks maintainable.
// I need to rip out all of the legacy and/or experimental features I'm not using or start fresh.
#[derive(Debug, Fail)]
#[fail(display = "stopped after dynmat.  THIS IS NOT AN ACTUAL ERROR. THIS IS A DUMB HACK.")]
pub(crate) struct StoppedAfterDynmat;

use rsp2_soa_ops::{Perm, Permute};
use rsp2_structure::CartOp;
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
    use rsp2_structure::consts;
    use rsp2_structure_io::Poscar;

    let subdir = trial.join("visualize-forces");
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
    freqs: &[f64],
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

//=================================================================

impl TrialDir {
    pub(crate) fn run_energy_surface(
        self,
        on_demand: Option<LammpsOnDemand>,
        settings: &cfg::EnergyPlotSettings,
        _input: &PathDir,
    ) -> FailResult<()>
    {Ok({
        // (IIFEs to silence dead code warnings.)
        let (coords, meta): (Coords, CommonMeta) = {|| unimplemented!("\
            rsp2-shear-plot has not yet been fixed to accept non-phonopy-based \
            input files.\
        ")}();
        let (freqs, evecs): (Vec<f64>, Basis3) = {|| unreachable!()}();

        let plot_ev_indices = {
            use rsp2_tasks_config::EnergyPlotEvIndices::*;

            let (i, j) = match settings.ev_indices {
                Shear => {
                    // FIXME: This should just find layers, check that there's two
                    //        and then move one along x and y instead
                    panic!("energy plot using shear modes is no longer supported")
                },
                These(i, j) => (i, j),
            };

            // (in case of confusion about 0-based/1-based indices)
            trace!("X: Eigensolution {:>3}, frequency {}", i, freqs[i]);
            trace!("Y: Eigensolution {:>3}, frequency {}", j, freqs[j]);
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
            &settings.lammps,
            &settings.potential,
        )?;

        let [xmin, xmax] = settings.xlim;
        let [ymin, ymax] = settings.ylim;
        let [w, h] = settings.dim;
        let data = {
            crate::cmd::integrate_2d::integrate_two_eigenvectors(
                (w, h),
                &coords.to_carts(),
                (xmin..xmax, ymin..ymax),
                (&get_real_ev(plot_ev_indices.0), &get_real_ev(plot_ev_indices.1)),
                {
                    use std::sync::atomic::{AtomicUsize, Ordering};
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
        serde_json::to_writer_pretty(self.create_file("out.json")?, &chunked)?;
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
    pub(crate) fn rerun_ev_analysis(
        self,
        on_demand: Option<LammpsOnDemand>,
        settings: &Settings,
        stored: StoredStructure,
    ) -> FailResult<()>
    {Ok({
        let pot = PotentialBuilder::from_root_config(&self, on_demand, &settings)?;

        let (freqs, evecs, _) = {
            self.do_post_relaxation_computations(
                settings, &*pot, &stored, false, None,
            )?
        };

        trace!("============================");
        trace!("Finished diagonalization");

        trace!("Classifying eigensolutions");
        let classifications = acoustic_search::perform_acoustic_search(
            &pot, &freqs, &evecs, &stored.coords, stored.meta().sift(), &settings.acoustic_search,
        )?;

        trace!("Computing eigensystem info");

        let ev_analysis = run_gamma_system_analysis(
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
    evecs: &Basis3,
) -> FailResult<GammaSystemAnalysis>
{Ok({
    trace!("Computing eigensystem info");
    let ev_analysis = run_gamma_system_analysis(
        &structure.coords,
        structure.meta().sift(),
        &freqs, &evecs,
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
    use rsp2_structure::{CoordsKind};
    let threading = cfg::Threading::Lammps;

    let lammps = cfg::Lammps {
        update_style: cfg::LammpsUpdateStyle::Fast { sync_positions_every: 1 }.into(),
        processor_axis_mask: [true; 3].into(),
    };
    let pot = PotentialBuilder::from_config_parts(None, on_demand, &threading, &lammps, pot)?;

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
    pot: &cfg::Potential,
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
    let pot = PotentialBuilder::from_config_parts(None, on_demand, &threading, &lammps, pot)?;

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
            crate::potential::Rsp2MinimizeDiffFnShim {
                ndim: 6,
                diff_fn: pot.initialize_flat_diff_fn(&get_coords(r_min), meta.sift())?,
            },
            &path.with_density(density),
        );
        println!("# {} {} {}", density, work, d_value);
    }
})}

//=================================================================

impl TrialDir {
    /// Used to figure out which iteration we're on when starting from the
    /// post-diagonalization part of the EV loop for sparse.
    pub(crate) fn find_iteration_for_ev_chase(&self) -> FailResult<Iteration> {
        use crate::cmd::EvLoopStructureKind::*;

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
        use crate::cmd::EvLoopStructureKind::*;
        use crate::filetypes::Eigensols;

        let pot = PotentialBuilder::from_root_config(&self, on_demand, &settings)?;

        let (coords, meta) = self.read_stored_structure_data(&self.structure_path(PreEvChase(iteration)))?;
        let Eigensols {
            frequencies: freqs,
            eigenvectors: evecs,
        } = Load::load(self.join(self.eigensols_path(iteration)))?;

        let (_, coords, did_ev_chasing) = self.do_ev_loop_stuff_after_diagonalization(
            settings, &pot, meta.sift(), iteration,
            coords, &freqs, &evecs,
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

pub fn run_make_supercell(
    structure: StoredStructure,
    dims_str: &str,
    output: impl AsPath,
) -> FailResult<()> {
    let StoredStructure {
        title, mut coords, mut elements, mut layers, mut masses,
        mut layer_sc_matrices, mut frac_bonds,
    } = structure;

    if let Some(_) = frac_bonds.take() {
        // TODO: support this properly.
        warn!("\
            Supercells of bond graphs are not yet implemented, so the created supercell will be \
            missing a bond graph.  (don't worry too much about this; rsp2 will typically \
            generate a new bond graph when run on the output).\
        ");
    };
    if let Some(_) = frac_bonds.take() {
        // TODO: support this properly.
        warn!("\
            Layer SC matrices will be lost.  This means that some layer-specific data won't \
            be included in rsp2's output. (like band unfolding)
        ");
    };

    let sc_dim = parse_sc_dims_argument(dims_str)?;
    let (super_coords, sc) = rsp2_structure::supercell::diagonal(sc_dim).build(&coords);
    coords = super_coords;

    elements = sc.replicate(&elements).into();
    masses = sc.replicate(&masses).into();
    layers = layers.as_mut().map(|x| sc.replicate(&x).into());
    layer_sc_matrices = layer_sc_matrices.map(|x| sc.replicate(&x).into());

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
                bail!("A vector supercell must consist of non-negative integers.")
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
    HList4<
        meta::SiteElements,
        meta::SiteMasses,
        Option<meta::SiteLayers>,
        Option<meta::LayerScMatrices>,
    >
)> {
    use crate::meta::Layer;

    let input = input.as_path();

    let out_coords: ScalableCoords;
    let out_elements: meta::SiteElements;
    let out_masses: meta::SiteMasses;
    let out_layers: Option<meta::SiteLayers>;
    let out_sc_mats: Option<meta::LayerScMatrices>;
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
        },
        StructureFileType::LayersYaml => {
            use rsp2_structure_io::layers_yaml::load;
            use rsp2_structure_io::layers_yaml::load_layer_sc_info;

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
fn masses_by_config(
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

    pub fn modified_settings_path(&self, iteration: Iteration) -> PathBuf
    { self.join(format!("ev-loop-modes-{:02}.yaml", iteration)) }

    pub fn uncompressed_gamma_dynmat_path(&self, iteration: Iteration) -> PathBuf
    { self.join(format!("gamma-dynmat-{:02}.json", iteration)) }

    pub fn gamma_dynmat_path(&self, iteration: Iteration) -> PathBuf
    { self.join(format!("gamma-dynmat-{:02}.npz", iteration)) }

    pub fn final_gamma_dynmat_path(&self) -> PathBuf
    { self.join("gamma-dynmat.npz") }

    pub fn animation_path(&self, iteration: Iteration, format: &cfg::AnimateFormat) -> PathBuf
    { match format {
        cfg::AnimateFormat::VSim {} => self.join(format!("ev-loop-modes-{:02}.ascii", iteration)),
    }}
}
