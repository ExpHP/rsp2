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

use self::potential::{PotentialBuilder, DiffFn};
mod potential;

use self::ev_analyses::GammaSystemAnalysis;
use self::param_optimization::ScalableCoords;
mod ev_analyses;

use self::trial::TrialDir;
pub(crate) mod trial;

pub(crate) use self::stored_structure::StoredStructure;
pub(crate) mod stored_structure;

use self::relaxation::EvLoopDiagonalizer;
mod relaxation;
mod acoustic_search;
mod param_optimization;

pub(crate) mod python;

use ::{FailResult, FailOk};
use ::rsp2_tasks_config::{self as cfg, Settings, NormalizationMode, SupercellSpec};
use ::traits::{AsPath};
use ::phonopy::{DirWithBands, DirWithDisps, DirWithForces};

use ::traits::{Load, Save};
use ::meta::prelude::*;
use ::meta::{Element, Mass, Layer};
use ::util::ext_traits::{OptionResultExt, PathNiceExt};
use ::math::basis::Basis3;
use ::math::bonds::{FracBonds};
use self::acoustic_search::ModeKind;

use ::path_abs::{PathAbs, PathArc, PathFile, PathDir};
use ::rsp2_structure::consts::CARBON;
use ::rsp2_slice_math::{vnorm};

use ::slice_of_array::prelude::*;
use ::rsp2_array_utils::arr_from_fn;
use ::rsp2_array_types::{V3, Unvee};
use ::rsp2_structure::{Coords, Lattice};
use ::rsp2_structure::layer::LayersPerUnitCell;
use ::phonopy::Builder as PhonopyBuilder;
use ::math::bands::ScMatrix;

use ::rsp2_fs_util::{rm_rf};

use ::std::io::{Write};
use ::std::rc::Rc;

use ::itertools::Itertools;

use ::hlist_aliases::*;
use std::collections::BTreeMap;
use math::dynmat::ForceConstants;
use cmd::potential::CommonMeta;
use math::dynmat::DynamicalMatrix;
use traits::save::Json;
use std::ffi::OsStr;
use std::ffi::OsString;

const SAVE_BANDS_DIR: &'static str = "gamma-bands";

// FIXME needs a better home
#[derive(Copy, Clone, PartialEq, Eq)]
pub enum StructureFileType {
    Poscar,
    LayersYaml,
    StoredStructure,
}

impl TrialDir {
    pub(crate) fn run_relax_with_eigenvectors(
        self,
        settings: &Settings,
        file_format: StructureFileType,
        input: &PathAbs,
    ) -> FailResult<()>
    {Ok({
        let pot = PotentialBuilder::from_root_config(&self, &settings);

        let (optimizable_coords, atom_elements, atom_masses, atom_layers, layer_sc_mats) = {
            read_optimizable_structure(
                settings.layer_search.as_ref(),
                settings.masses.as_ref(),
                file_format, input,
            )?
        };
        let meta = hlist![atom_elements, atom_masses, atom_layers];

        let original_coords = {
            ::cmd::param_optimization::optimize_layer_parameters(
                &settings.scale_ranges,
                &pot,
                optimizable_coords,
                meta.sift(),
            )?.construct()
        };

        self.write_stored_structure(
            "initial.structure", "Initial structure (after lattice optimization)",
            &original_coords, meta.sift(), layer_sc_mats.clone(),
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
                        settings, &*pot, layer_sc_mats.clone(),
                        diagonalizer, original_coords, meta.sift(),
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
                    let (coords, ev_analysis, dynmat) = {
                        do_ev_loop!(SparseDiagonalizer { shift_invert_attempts })?
                    };
                    _do_not_drop_the_bands_dir = None;

                    // HACK
                    Json(dynmat.cereal()).save(self.join("gamma-dynmat.json"))?;

                    (coords, ev_analysis)
                },
            }
        };

        self.write_stored_structure(
            "final.structure", "Final structure",
            &coords, meta.sift(), layer_sc_mats.clone(),
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
            layer_sc_dims: sc_mats.0.iter().map(|m| m.periods).collect(),
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
        dir_name: &str,
        poscar_headline: &str,
        coords: &Coords,
        meta: HList3<
            Rc<[Element]>,
            Rc<[Mass]>,
            Option<Rc<[Layer]>>,
        >,
        layer_sc_matrices: Option<Rc<[ScMatrix]>>, // FIXME awkward
    ) -> FailResult<()>
    {Ok({
        let path = self.join(dir_name);

        // TODO: should take this as an input argument, and also pass it around through
        //       relaxation and potential
        let frac_bonds = {
            // HACK
            // (no code is supposed to be reading settings.yaml outside of entry points,
            //  but I don't want to add a Settings argument. The proper fix is to fix
            //  the above todo)
            let settings: Settings = self.read_settings()?;
            settings.bond_radius.map(|bond_radius| FailOk({
                trace!("needlessly computing frac bonds (FIXME)");
                FracBonds::from_brute_force_very_dumb(coords, bond_radius)?
            })).fold_ok()?
        };

        trace!("Writing '{}'", path.nice());
        StoredStructure {
            title: poscar_headline.into(),
            coords: coords.clone(),
            elements: meta.pick(),
            masses: meta.pick(),
            layers: meta.pick(),
            layer_sc_matrices,
            frac_bonds,
        }.save(path)?
    })}

    fn read_stored_structure_data(
        &self,
        dir_name: &str,
    ) -> FailResult<(
        Coords,
        HList3<Rc<[Element]>, Rc<[Mass]>, Option<Rc<[Layer]>>>,
        Option<Rc<[ScMatrix]>>,
    )>
    {Ok({
        let stored = self.read_stored_structure(dir_name)?;
        let meta = stored.meta();
        (stored.coords, meta, stored.layer_sc_matrices)
    })}

    fn read_stored_structure(&self, dir_name: impl AsRef<OsStr>) -> FailResult<StoredStructure>
    { Load::load(self.join(dir_name.as_ref())) }
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
}

impl PhonopyDiagonalizer {
    // Create a DirWithBands that may or may not be a TempDir (based on config)
    fn create_bands_dir(
        &self,
        cfg_threading: &cfg::Threading,
        pot: &PotentialBuilder,
        phonopy: &PhonopyBuilder,
        coords: &Coords,
        meta: HList2<
            Rc<[Element]>,
            Rc<[Mass]>,
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
    type ExtraOut = DynamicalMatrix;

    fn do_post_relaxation_computations(
        &self,
        _trial: &TrialDir,
        settings: &Settings,
        pot: &PotentialBuilder,
        stored: &StoredStructure,
    ) -> FailResult<(Vec<f64>, Basis3, DynamicalMatrix)>
    {Ok({
        let prim_coords = &stored.coords;
        let prim_meta = stored.meta();

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
                        let elements: Rc<[Element]> = prim_meta.pick();
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
            prim_meta.clone().map(hlist![
                f!(),
                f!(),
                |opt: Option<_>| opt.map(f!()),
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
        {
            writeln!(_trial.create_file("force-sets")?, "{:?}", force_sets).unwrap();
        }

        let cart_rots: Vec<_> = {
            cart_ops.iter().map(|c| c.cart_rot()).collect()
        };

        trace!("Computing deperms in supercell");
        let super_deperms = compute_deperms(&super_coords, &cart_ops)?;

        // XXX
        visualize_sparse_force_sets(
            &_trial,
            &super_coords,
            &super_displacements,
            &super_deperms,
            &cart_ops,
            &force_sets,
        )?;

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
        Json(dynmat.cereal()).save(_trial.join("gamma-dynmat.json"))?;

        {
            let max_size = (|(a, b)| a * b)(dynmat.0.dim);
            let nnz = dynmat.0.nnz();
            let density = nnz as f64 / max_size as f64;
            trace!("nnz: {} out of {} blocks (matrix density: {:.3e})", nnz, max_size, density);
        }
        trace!("Diagonalizing dynamical matrix");
        let (evals, evecs) = dynmat.compute_negative_eigensolutions(self.shift_invert_attempts)?;
        trace!("Done diagonalizing dynamical matrix");
        (evals, evecs, dynmat)
    })}
}


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
    meta: HList3<
        Rc<[Element]>,
        Rc<[Mass]>,
        Option<Rc<[Layer]>>,
    >,
    layer_sc_mats: Option<Rc<[ScMatrix]>>,
    evals: &[f64],
    evecs: &Basis3,
    bonds: Option<&FracBonds>,
    mode_classifications: Option<Rc<[ModeKind]>>,
) -> FailResult<GammaSystemAnalysis> {
    use self::ev_analyses::*;

    let cart_bonds = bonds.as_ref().map(|b| b.to_cart_bonds(coords));

    let atom_elements: Rc<[Element]> = meta.pick();
    let atom_masses: Rc<[Mass]> = meta.pick();
    let atom_masses: Vec<f64> = atom_masses.iter().map(|&Mass(x)| x).collect();
    let atom_layers: Option<Rc<[Layer]>> = meta.pick();
    let atom_layers = atom_layers.map(|v| v.iter().map(|&Layer(n)| n).collect());

    gamma_system_analysis::Input {
        atom_layers: atom_layers.map(AtomLayers),
        layer_sc_mats: layer_sc_mats.map(|x| LayerScMatrices(x.to_vec())),
        atom_masses: Some(AtomMasses(atom_masses)),
        ev_classifications: mode_classifications.map(|x| EvClassifications(x.to_vec())),
        atom_elements: Some(AtomElements(atom_elements.to_vec())),
        atom_coords: Some(AtomCoordinates(coords.clone())),
        ev_frequencies: Some(EvFrequencies(evals.to_vec())),
        ev_eigenvectors: Some(EvEigenvectors(evecs.clone())),
        bonds: cart_bonds.map(Bonds),
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
            let f = |s: &str| FailOk({
                let (coords, meta, _) = self.read_stored_structure_data(s)?;

                let na = coords.num_atoms() as f64;
                pot.one_off().compute_value(&coords, meta.sift())? / na
            });

            let initial = f(&"initial.structure")?;
            let final_ = f(&"final.structure")?;
            let before_ev_chasing = f(&"ev-loop-01.1.structure")?;

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
        &meta.sift(),
        threading == &cfg::Threading::Rayon,
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
    meta: HList2<
        Rc<[Element]>,
        Rc<[Mass]>,
    >,
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
    meta: &CommonMeta,
    use_rayon: bool,
    inputs: Inputs,
    compute: F,
) -> FailResult<Vec<Output>>
where
    Input: Send,
    Output: Send,
    Inputs: IntoIterator<Item=Input>,
    Inputs: ::rayon::iter::IntoParallelIterator<Item=Input>,
    F: Fn(&mut DiffFn<CommonMeta>, CommonMeta, Input) -> FailResult<Output> + Sync + Send,
{
    if use_rayon {
        use ::rayon::prelude::*;
        let get_meta = meta.sendable();
        inputs.into_par_iter()
            .map(|x| compute(&mut pot.one_off(), get_meta(), x))
            .collect()
    } else {
        // save the cost of repeated DiffFn initialization since we don't need Send.
        let mut diff_fn = pot.initialize_diff_fn(coords_for_initialize, meta.clone())?;
        inputs.into_iter()
            .map(|x| compute(&mut diff_fn, meta.clone(), x))
            .collect()
    }
}

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
                    let me = &self;

                    move |pos| {FailOk({
                        let i = counter.fetch_add(1, Ordering::SeqCst);
                        // println!("{:?}", pos.flat().iter().sum::<f64>());

                        eprint!("\rdatapoint {:>6} of {}", i, w * h);
                        PotentialBuilder::from_config_parts(me, &settings.threading, &settings.lammps_update_style, &settings.potential)
                            .one_off()
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
        settings: &Settings,
    ) -> FailResult<()>
    {Ok({
        let pot = PotentialBuilder::from_root_config(&self, &settings);

        let (coords, meta, _) = self.read_stored_structure_data("final.structure")?;

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
        settings: &Settings,
        stored: StoredStructure,
    ) -> FailResult<()>
    {Ok({
        let pot = PotentialBuilder::from_root_config(&self, &settings);

        let bonds = settings.bond_radius.map(|bond_radius| FailOk({
            trace!("Computing bonds");
            FracBonds::from_brute_force_very_dumb(&stored.coords, bond_radius)?
        })).fold_ok()?;

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
                        &self, settings, &*pot, &stored,
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
            &stored.coords,
            stored.meta().sift(),
            stored.layer_sc_matrices.clone(),
            &evals, &evecs,
            bonds.as_ref(),
            Some(classifications),
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
        structure.layer_sc_matrices.clone(),
        &evals, &evecs,
        structure.frac_bonds.as_ref(),
        None, // ev_classifications
    )?;

    write_eigen_info_for_humans(&ev_analysis, &mut |s| FailOk(info!("{}", s)))?;

    ev_analysis
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
    Rc<[Element]>,
    Rc<[Mass]>,
    Option<Rc<[Layer]>>,
    Option<Rc<[ScMatrix]>>,
)> {
    let input = input.as_path();

    let out_coords: ScalableCoords;
    let out_elements: Rc<[Element]>;
    let out_masses: Rc<[Mass]>;
    let out_layers: Option<Rc<[Layer]>>;
    let out_sc_mats: Option<Rc<[ScMatrix]>>;
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
            use ::cmd::stored_structure::StoredStructure;

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
    Ok((out_coords, out_elements, out_masses, out_layers, out_sc_mats))
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
    elements: Rc<[Element]>,
) -> FailResult<Rc<[Mass]>>
{Ok({
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
            None => ::common::default_element_mass(element).map(Mass),
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
