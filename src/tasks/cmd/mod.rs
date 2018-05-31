// HERE BE DRAGONS

pub(crate) mod integrate_2d;

use self::potential::{PotentialBuilder, DiffFn};
mod potential;

use self::ev_analyses::GammaSystemAnalysis;
use self::param_optimization::ScalableCoords;
mod ev_analyses;

use self::trial::TrialDir;
pub(crate) mod trial;

use self::stored_structure::StoredStructure;
mod stored_structure;

use self::relaxation::EvLoopDiagonalizer;
mod relaxation;
mod acoustic_search;
mod scipy_eigsh;
mod param_optimization;

use ::{FailResult, FailOk};
use ::rsp2_tasks_config::{self as cfg, Settings, NormalizationMode, SupercellSpec};
use ::traits::{AsPath};
use ::phonopy::{DirWithBands, DirWithDisps, DirWithForces, DirWithSymmetry};

use ::traits::{Load, Save};
use ::meta::prelude::*;
use ::meta::{Element, Mass, Layer};
use ::util::ext_traits::{OptionResultExt, PathNiceExt};
use ::math::basis::Basis3;
use ::math::bonds::{FracBonds};

use ::path_abs::{PathAbs, PathArc, PathFile, PathDir};
use ::rsp2_structure::consts::CARBON;
use ::rsp2_slice_math::{vnorm};

use ::slice_of_array::prelude::*;
use ::rsp2_array_utils::arr_from_fn;
use ::rsp2_array_types::{V3, M33, Unvee};
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
        let pot = PotentialBuilder::from_config(&settings.threading, &settings.potential);

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
            &original_coords, meta.sift(), &layer_sc_mats,
        )?;
        let phonopy = phonopy_builder_from_settings(&settings.phonons, original_coords.lattice());

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
                        settings, &*pot, &layer_sc_mats,
                        &phonopy, diagonalizer, original_coords, meta.sift(),
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
                Sparse { max_count } => {
                    let (coords, ev_analysis, ()) = {
                        do_ev_loop!(SparseDiagonalizer { max_count })?
                    };
                    _do_not_drop_the_bands_dir = None;
                    (coords, ev_analysis)
                },
            }
        };

        self.write_stored_structure(
            "final.structure", "Final structure",
            &coords, meta.sift(), &layer_sc_mats,
        )?;

        write_eigen_info_for_machines(&ev_analysis, self.create_file("eigenvalues.final")?)?;

        self.write_ev_analysis_output_files(settings, &*pot, &ev_analysis)?;
    })}

    fn write_ev_analysis_output_files(
        &self,
        settings: &Settings,
        pot: &PotentialBuilder,
        eva: &GammaSystemAnalysis,
    ) -> FailResult<()>
    {Ok({
        if let (Some(frequency), Some(raman)) = (&eva.ev_frequencies, &eva.ev_raman_tensors) {
            #[derive(Serialize)]
            #[serde(rename_all = "kebab-case")]
            struct Output {
                frequency: Vec<f64>,
                average_3d: Vec<f64>,
                backscatter: Vec<f64>,
            }
            use ::math::bond_polarizability::LightPolarization::*;
            ::serde_json::to_writer(self.create_file("raman.json")?, &Output {
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

            ::serde_json::to_writer(self.create_file("unfold.json")?, &Output {
                layer_sc_dims: sc_mats.0.iter().map(|m| m.periods).collect(),
                layer_q_indices: {
                    unfold_probs.layer_unfolders.iter()
                        .map(|u| u.q_indices().to_vec())
                        .collect()
                },
                layer_ev_q_probs: unfold_probs.layer_ev_q_probs.clone(),
            })?;
        }

        self.write_summary_file(settings, pot, eva)?;
    })}

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
        layer_sc_mats: &Option<Vec<ScMatrix>>, // FIXME awkward
    ) -> FailResult<()>
    {Ok({
        let path = self.join(dir_name);

        trace!("Writing '{}'", path.nice());
        StoredStructure {
            title: poscar_headline.into(),
            coords: coords.clone(),
            elements: meta.pick(),
            masses: meta.pick(),
            layers: meta.pick(),
            layer_sc_matrices: layer_sc_mats.clone(),
        }.save(path)?
    })}

    fn read_stored_structure_data(
        &self,
        dir_name: &str,
    ) -> FailResult<(
        Coords,
        HList3<Rc<[Element]>, Rc<[Mass]>, Option<Rc<[Layer]>>>,
        Option<Vec<ScMatrix>>,
    )>
    {Ok({
        let stored = self.read_stored_structure(dir_name)?;
        let meta = stored.meta();
        (stored.coords, meta, stored.layer_sc_matrices)
    })}

    fn read_stored_structure(&self, dir_name: &str) -> FailResult<StoredStructure>
    { Load::load(self.join(dir_name)) }
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
        settings: &Settings,
        pot: &PotentialBuilder,
        phonopy: &PhonopyBuilder,
        stored: &StoredStructure,
    ) -> FailResult<(Vec<f64>, Basis3, DirWithBands<Box<AsPath>>)>
    {Ok({
        let meta = stored.meta();
        let coords = stored.coords.clone();

        let bands_dir = self.create_bands_dir(&settings.threading, pot, phonopy, &coords, meta.sift())?;
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
    max_count: usize,
}
impl EvLoopDiagonalizer for SparseDiagonalizer {
    type ExtraOut = ();

    fn do_post_relaxation_computations(
        &self,
        settings: &Settings,
        pot: &PotentialBuilder,
        phonopy: &PhonopyBuilder,
        stored: &StoredStructure,
    ) -> FailResult<(Vec<f64>, Basis3, ())>
    {Ok({
        let coords = &stored.coords;
        let meta = stored.meta();

        let frac_ops = phonopy.symmetry(coords, meta.sift())?.frac_ops()?;
        let cart_rots: Vec<M33> = {
            frac_ops.iter()
                .map(|oper| oper.to_rot().cart(&coords.lattice()))
                .collect()
        };

        let ::phonopy::Rsp2StyleDisplacements {
            super_coords, sc, prim_displacements, ..
        } = phonopy.displacements(coords, meta.sift())?.rsp2_style_displacements()?;

        let space_group_deperms: Vec<_> = {
            ::rsp2_structure::find_perm::of_spacegroup_for_general(
                &super_coords,
                &frac_ops,
                &coords.lattice(),
                // larger than SYMPREC because the coords we see may may be slightly
                // different from what spglib saw, but not so large that we risk pairing
                // the wrong atoms
                settings.phonons.symmetry_tolerance * 3.0,
            )?.into_iter().map(|p| p.inverted()).collect()
        };

        let super_meta = {
            // macro to generate a closure, because generic closures don't exist
            macro_rules! f {
                () => { |x: Rc<[_]>| -> Rc<[_]> {
                    sc.replicate(&x[..]).into()
                }};
            }
            meta.clone().map(hlist![
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
        let force_sets = do_force_sets_at_disps_for_sparse(
            pot,
            &settings.threading,
            &super_displacements,
            &super_coords,
            super_meta.sift(),
        )?;

        let force_constants = ::math::dynmat::ForceConstants::compute_required_rows(
            &super_displacements,
            &force_sets,
            &cart_rots,
            &space_group_deperms,
            &sc,
        )?;

        let dynmat = {
            force_constants
                .gamma_dynmat(&sc, meta.pick())
                .hermitianize()
        };

//        HACK
//        Json(&dynmat.cereal()).save(self.join("dynmat.json"))?;

        let how_many = self.max_count.min(dynmat.max_sparse_eigensolutions());
        let (evals, evecs) = dynmat.compute_most_negative_eigensolutions(how_many)?;
        (evals, evecs, ())
    })}
}

// wrapper around the gamma_system_analysis module which handles all the newtype conversions
fn run_gamma_system_analysis(
    settings: &Settings,
    pot: &PotentialBuilder,
    coords: &Coords,
    meta: HList3<
        Rc<[Element]>,
        Rc<[Mass]>,
        Option<Rc<[Layer]>>,
    >,
    layer_sc_mats: &Option<Vec<ScMatrix>>,
    evals: &[f64],
    evecs: &Basis3,
    bonds: &Option<FracBonds>,
) -> FailResult<GammaSystemAnalysis> {
    use self::ev_analyses::*;

    let cart_bonds = bonds.as_ref().map(|b| b.to_cart_bonds(coords));

    // (this is more or less part of the analysis, but it is not done there
    //  because it requires a couple more calls to the potential)
    let classifications = acoustic_search::perform_acoustic_search(
        pot, &evals, &evecs, &coords, meta.sift(), &settings.acoustic_search,
    )?;

    let atom_elements: Rc<[Element]> = meta.pick();
    let atom_masses: Rc<[Mass]> = meta.pick();
    let atom_masses: Vec<f64> = atom_masses.iter().map(|&Mass(x)| x).collect();
    let atom_layers: Option<Rc<[Layer]>> = meta.pick();
    let atom_layers = atom_layers.map(|v| v.iter().map(|&Layer(n)| n).collect());

    gamma_system_analysis::Input {
        atom_layers: atom_layers.map(AtomLayers),
        layer_sc_mats: layer_sc_mats.clone().map(LayerScMatrices),
        atom_masses: Some(AtomMasses(atom_masses)),
        ev_classifications: Some(EvClassifications(classifications)),
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
    PhonopyBuilder::new()
        .symmetry_tolerance(settings.symmetry_tolerance)
        .conf("DISPLACEMENT_DISTANCE", format!("{:e}", settings.displacement_distance))
        .supercell_dim(settings.supercell.dim_for_unitcell(lattice))
        .conf("DIAG", ".FALSE.")
}

fn do_force_sets_at_disps_for_phonopy<P: AsPath + Send + Sync>(
    pot: &PotentialBuilder,
    threading: &cfg::Threading,
    disp_dir: &DirWithDisps<P>,
) -> FailResult<Vec<Vec<V3>>>
{Ok({
    use ::std::io::prelude::*;
    use ::rayon::prelude::*;

    trace!("Computing forces at displacements");

    let counter = ::util::AtomicCounter::new();
    let num_displacements = disp_dir.displacements().len();
    let compute = move |diff_fn: &mut DiffFn<_>, coords: &Coords, meta: HList2<Rc<[Element]>, Rc<[Mass]>>| FailOk({
        let i = counter.inc();
        eprint!("\rdisp {} of {}", i + 1, num_displacements);
        ::std::io::stderr().flush().unwrap();

        diff_fn.compute_force(coords, meta.sift())?
    });

    // FIXME duplicated logic
    let force_sets = match threading {
        &cfg::Threading::Lammps |
        &cfg::Threading::Serial => {
            let (initial_coords, meta) = disp_dir.superstructure();
            let mut diff_fn = pot.initialize_diff_fn(initial_coords, meta.sift())?;

            disp_dir.displaced_coord_sets()
                .map(|coords| compute(&mut diff_fn, &coords, meta.sift()))
                .collect::<FailResult<Vec<_>>>()?
        },
        &cfg::Threading::Rayon => {
            let (_, meta) = disp_dir.superstructure();
            let get_meta = meta.sendable();

            disp_dir.displaced_coord_sets()
                .collect::<Vec<_>>()
                .into_par_iter()
                .map(|coords| compute(&mut pot.one_off(), &coords, get_meta().sift()))
                .collect::<FailResult<Vec<_>>>()?
        },
    };
    eprintln!();
    force_sets
})}

fn do_force_sets_at_disps_for_sparse(
    pot: &PotentialBuilder,
    threading: &cfg::Threading,
    displacements: &[(usize, V3)],
    coords: &Coords,
    meta: HList2<
        Rc<[Element]>,
        Rc<[Mass]>,
    >,
) -> FailResult<Vec<BTreeMap<usize, V3>>>
{Ok({
    use ::std::io::prelude::*;
    use ::rayon::prelude::*;

    trace!("Computing forces at displacements");

    let counter = ::util::AtomicCounter::new();
    let num_displacements = displacements.len();

    let compute = move |diff_fn: &mut DiffFn<_>, displacement: (usize, V3), meta: HList2<Rc<[Element]>, Rc<[Mass]>>| FailOk({
        let i = counter.inc();
        eprint!("\rdisp {} of {}", i + 1, num_displacements);
        ::std::io::stderr().flush().unwrap();

        diff_fn.compute_sparse_force_set(coords, meta.sift(), displacement)?
    });

    // FIXME duplicated logic
    let force_sets = match threading {
        &cfg::Threading::Lammps |
        &cfg::Threading::Serial => {
            let mut diff_fn = pot.initialize_diff_fn(coords, meta.sift())?;
            displacements.iter()
                .map(|&disp| compute(&mut diff_fn, disp, meta.sift()))
                .collect::<FailResult<Vec<_>>>()?
        },
        &cfg::Threading::Rayon => {
            let get_meta = meta.sendable();
            displacements.par_iter()
                .map(|&disp| compute(&mut pot.one_off(), disp, get_meta().sift()))
                .collect::<FailResult<Vec<_>>>()?
        },
    };
    eprintln!();
    force_sets
})}

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

                    move |pos| {FailOk({
                        let i = counter.fetch_add(1, Ordering::SeqCst);
                        // println!("{:?}", pos.flat().iter().sum::<f64>());

                        eprint!("\rdatapoint {:>6} of {}", i, w * h);
                        PotentialBuilder::from_config(&settings.threading, &settings.potential)
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
        let pot = PotentialBuilder::from_config(&settings.threading, &settings.potential);

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
    ) -> FailResult<()>
    {Ok({
        let pot = PotentialBuilder::from_config(&settings.threading, &settings.potential);

        let stored = self.read_stored_structure("./final.structure")?;
        let phonopy = phonopy_builder_from_settings(&settings.phonons, stored.coords.lattice());

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
                        settings, &*pot, &phonopy, &stored,
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
                Sparse { max_count } => {
                    let (evals, evecs, _) = {
                        use_diagonalizer!(SparseDiagonalizer { max_count } )?
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

        trace!("Computing eigensystem info");

        let ev_analysis = run_gamma_system_analysis(
            &settings, &pot,
            &stored.coords,
            stored.meta().sift(),
            &stored.layer_sc_matrices,
            &evals, &evecs, &bonds,
        )?;

        self.write_ev_analysis_output_files(settings, &*pot, &ev_analysis)?;
    })}
}

//=================================================================

// FIXME refactor once it's working, this is way too long
pub(crate) fn run_dynmat_test(phonopy_dir: &PathDir) -> FailResult<()>
{Ok({
    ::cmd::scipy_eigsh::check_scipy_availability()?;

    // Make a supercell, and determine how our ordering of the supercell differs from phonopy.
    let symmetry_dir = DirWithSymmetry::from_existing(phonopy_dir)?;
    let forces_dir = DirWithForces::from_existing(phonopy_dir)?;
    let disp_dir = DirWithDisps::from_existing(phonopy_dir)?;
    let ::phonopy::Rsp2StyleDisplacements {
        super_coords, sc, perm_from_phonopy, ..
    } = disp_dir.rsp2_style_displacements()?;

    let (prim_coords, prim_meta) = disp_dir.primitive_structure()?;
    let space_group = symmetry_dir.frac_ops()?;
    let prim_lattice = prim_coords.lattice().clone();

    let space_group_deperms: Vec<_> = {
        ::rsp2_structure::find_perm::of_spacegroup_for_general(
            &super_coords,
            &space_group,
            &prim_lattice,
            1e-1, // FIXME should be slightly larger than configured tol,
                  //       but I forgot where that is stored.
        )?.into_iter().map(|p| p.inverted()).collect()
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

    let cart_rots: Vec<M33> = {
        space_group.iter()
            .map(|oper| oper.to_rot().cart(&prim_lattice))
            .collect()
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
        let (low, _low_basis) = our_dynamical_matrix.compute_most_negative_eigensolutions(3)?;
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
    ScalableCoords,
    Rc<[Element]>,
    Rc<[Mass]>,
    Option<Rc<[Layer]>>,
    Option<Vec<ScMatrix>>,
)> {
    let input = input.as_path();

    let out_coords: ScalableCoords;
    let out_elements: Rc<[Element]>;
    let out_masses: Rc<[Mass]>;
    let out_layers: Option<Rc<[Layer]>>;
    let out_sc_mats: Option<Vec<ScMatrix>>;
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
