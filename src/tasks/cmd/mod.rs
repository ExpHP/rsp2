// HERE BE DRAGONS

pub(crate) mod integrate_2d;

use self::potential::{PotentialBuilder, DiffFn};
mod potential;

use self::ev_analyses::GammaSystemAnalysis;
use self::param_optimization::ScalableCoords;
mod ev_analyses;

use self::trial::TrialDir;
pub(crate) mod trial;

mod acoustic_search;
mod relaxation;
mod scipy_eigsh;
mod param_optimization;

use ::{FailResult, FailOk};
use ::rsp2_tasks_config::{self as cfg, Settings, NormalizationMode, SupercellSpec};
use ::traits::{AsPath};
use ::phonopy::{DirWithBands, DirWithDisps, DirWithForces, DirWithSymmetry};

use ::meta::prelude::*;
use ::meta::{Element, Mass};
use ::util::ext_traits::{OptionResultExt, PathNiceExt};
use ::math::basis::Basis3;
use ::math::bonds::{FracBonds};

use ::path_abs::{PathArc, PathFile, PathDir};
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

const SAVE_BANDS_DIR: &'static str = "gamma-bands";

// cli args aren't in Settings, so they're just here.
pub struct CliArgs {
    pub save_bands: bool,
}

// FIXME needs a better home
#[derive(Copy, Clone, PartialEq, Eq)]
pub enum StructureFileType {
    Poscar,
    LayersYaml,
}

impl TrialDir {
    pub(crate) fn run_relax_with_eigenvectors(
        self,
        settings: &Settings,
        file_format: StructureFileType,
        input: &PathFile,
        cli: CliArgs,
    ) -> FailResult<()>
    {Ok({
        let pot = PotentialBuilder::from_config(&settings.threading, &settings.potential);

        let (optimizable_coords, atom_elements, atom_layers, layer_sc_mats) = {
            read_optimizable_structure(settings.layer_search.as_ref(), file_format, input)?
        };
        let atom_masses = masses_by_config(settings.masses.as_ref(), atom_elements.clone())?;
        let meta = hlist![atom_elements, atom_masses];

        let original_coords = {
            ::cmd::param_optimization::optimize_layer_parameters(
                &settings.scale_ranges,
                &pot,
                optimizable_coords,
                meta.sift(),
            )?.construct()
        };

        self.write_poscar(
            "initial.vasp", "Initial structure (after lattice optimization)",
            &original_coords, meta.sift(),
        )?;
        let phonopy = phonopy_builder_from_settings(&settings.phonons, original_coords.lattice());

        let (coords, ev_analysis, final_bands_dir) = self.do_main_ev_loop(
            settings, &cli, &*pot, &atom_layers, &layer_sc_mats,
            &phonopy, original_coords, meta.sift(),
        )?;
        let _do_not_drop_the_bands_dir = final_bands_dir;

        self.write_poscar("final.vasp", "Final structure", &coords, meta.sift())?;

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

    // log when writing poscar files, especially during loops
    // (to remove any doubt about the iteration number)
    fn write_poscar(
        &self,
        filename: &str,
        headline: &str,
        coords: &Coords,
        meta: HList1<
            Rc<[Element]>,
        >,
    ) -> FailResult<()>
    {Ok({
        use ::rsp2_structure_io::Poscar;

        let elements: Rc<[Element]> = meta.pick();

        let file = self.create_file(filename)?;
        trace!("Writing '{}'", file.path().nice());
        Poscar {
            comment: headline,
            coords: coords,
            elements: elements,
        }.to_writer(file)?
    })}

    fn do_post_relaxation_computations(
        &self,
        settings: &Settings,
        save_bands: Option<&PathArc>,
        pot: &PotentialBuilder,
        aux_info: aux_info::Info,
        phonopy: &PhonopyBuilder,
        coords: &Coords,
        meta: HList2<
            Rc<[Element]>,
            Rc<[Mass]>,
        >,
    ) -> FailResult<(DirWithBands<Box<AsPath>>, Vec<f64>, Basis3, GammaSystemAnalysis)>
    {Ok({

        let bands_dir = do_diagonalize(
            pot, &settings.threading, phonopy, coords, meta.sift(), save_bands, &[Q_GAMMA],
        )?;
        let (evals, evecs) = bands_dir.eigensystem_at(Q_GAMMA)?;

        trace!("============================");
        trace!("Finished diagonalization");

        // FIXME: only the CartBonds need to be recomputed each iteration;
        //       we could keep the FracBonds around between iterations.
        let bonds = settings.bond_radius.map(|bond_radius| FailOk({
            trace!("Computing bonds");
            FracBonds::from_brute_force_very_dumb(&coords, bond_radius)?
                .to_cart_bonds(&coords)
        })).fold_ok()?;

        trace!("Computing eigensystem info");

        let ev_analysis = {
            use self::ev_analyses::*;

            let classifications = acoustic_search::perform_acoustic_search(
                pot, &evals, &evecs, &coords, meta.sift(), &settings.acoustic_search,
            )?;

            let aux_info::Info { atom_layers, atom_masses, layer_sc_mats } = aux_info;
            let atom_elements: Rc<[Element]> = meta.pick();

            gamma_system_analysis::Input {
                atom_masses,
                atom_layers,
                layer_sc_mats,
                ev_classifications: Some(EvClassifications(classifications)),
                atom_elements:      Some(AtomElements(atom_elements.to_vec())),
                atom_coords:        Some(AtomCoordinates(coords.clone())),
                ev_frequencies:     Some(EvFrequencies(evals.clone())),
                ev_eigenvectors:    Some(EvEigenvectors(evecs.clone())),
                bonds:              bonds.map(Bonds),
            }.compute()?
        };
        (bands_dir, evals, evecs, ev_analysis)
    })}
}

fn do_diagonalize(
    pot: &PotentialBuilder,
    threading: &cfg::Threading,
    phonopy: &PhonopyBuilder,
    coords: &Coords,
    meta: HList2<
        Rc<[Element]>,
        Rc<[Mass]>,
    >,
    save_bands: Option<&PathArc>,
    points: &[V3],
) -> FailResult<DirWithBands<Box<AsPath>>>
{Ok({
    let disp_dir = phonopy.displacements(coords, meta.sift())?;
    let force_sets = do_force_sets_at_disps(pot, &threading, &disp_dir)?;

    let bands_dir = disp_dir
        .make_force_dir(&force_sets)?
        .build_bands()
        .eigenvectors(true)
        .compute(&points)?;

    if let Some(save_dir) = save_bands {
        rm_rf(save_dir)?;
        bands_dir.relocate(save_dir)?.boxed()
    } else {
        bands_dir.boxed()
    }
})}

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
            let f = |(coords, meta): (Coords, ::frunk::HCons<_, _>)| FailOk({
                let na = coords.num_atoms() as f64;
                pot.one_off().compute_value(&coords, meta.sift())? / na
            });
            let f_path = |s: &AsPath| FailOk(f(read_poscar(settings.masses.as_ref(), self.read_file(s)?)?)?);

            let initial = f_path(&"initial.vasp")?;
            let final_ = f_path(&"final.vasp")?;
            let before_ev_chasing = f_path(&"structure-01.1.vasp")?;

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

fn do_force_sets_at_disps<P: AsPath + Send + Sync>(
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
    let compute = move |coords: &Coords, meta: HList2<Rc<[Element]>, Rc<[Mass]>>| FailOk({
        let i = counter.inc();
        eprint!("\rdisp {} of {}", i + 1, num_displacements);
        ::std::io::stderr().flush().unwrap();

        pot.one_off().compute_force(coords, meta.sift())?
    });

    let force_sets = match threading {
        &cfg::Threading::Lammps |
        &cfg::Threading::Serial => {
            let (_, meta) = disp_dir.superstructure();
            disp_dir.displaced_coord_sets()
                .map(|coords| compute(&coords, meta.sift()))
                .collect::<FailResult<Vec<_>>>()?
        },
        &cfg::Threading::Rayon => {
            let (_, meta) = disp_dir.superstructure();
            let get_meta = meta.sendable();
            disp_dir.displaced_coord_sets()
                .collect::<Vec<_>>()
                .into_par_iter()
                .map(|coords| compute(&coords, get_meta().sift()))
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

// auxilliary info for rerunning updated analysis code on old trial directories
mod aux_info {
    use super::*;
    use self::ev_analyses::*;

    const FILENAME: &'static str = "aux-analysis-info.json";

    impl TrialDir {
        pub(crate) fn save_analysis_aux_info(&self, aux: &Info) -> FailResult<()>
        { Ok(::serde_json::to_writer(self.create_file(FILENAME)?, &aux)?) }

        pub(crate) fn load_analysis_aux_info(&self) -> FailResult<Info>
        {Ok({
            let file = self.read_file(FILENAME)?;
            let de = &mut ::serde_json::Deserializer::from_reader(file);
            ::serde_ignored::deserialize(de, |path| {
                panic!("Incompatible {}: unrecognized entry: {}", FILENAME, path)
            })?
        })}
    }

    #[derive(Clone, Deserialize, Serialize)]
    pub struct Info {
        pub atom_layers:   Option<AtomLayers>,
        pub atom_masses:   Option<AtomMasses>,
        pub layer_sc_mats: Option<LayerScMatrices>,
    }
}

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

        let (coords, meta) = read_poscar(settings.masses.as_ref(), self.read_file("./final.vasp")?)?;
        let phonopy = phonopy_builder_from_settings(&settings.phonons, coords.lattice());
        do_diagonalize(
            &*pot, &settings.threading, &phonopy, &coords, meta.sift(),
            Some(&self.save_bands_dir()),
            &[Q_GAMMA, Q_K],
        )?;
    })}
}

impl TrialDir {
    pub fn save_bands_dir(&self) -> PathArc
    { self.join(SAVE_BANDS_DIR).into() }
}

//=================================================================

impl TrialDir {
    pub(crate) fn rerun_ev_analysis(
        self,
        settings: &Settings,
    ) -> FailResult<()>
    {Ok({
        let pot = PotentialBuilder::from_config(&settings.threading, &settings.potential);

        let (coords, meta) = read_poscar(settings.masses.as_ref(), self.read_file("./final.vasp")?)?;
        let phonopy = phonopy_builder_from_settings(&settings.phonons, coords.lattice());

        let aux_info = self.load_analysis_aux_info()?;

        let save_bands = None;
        let (_, _, _, ev_analysis) = self.do_post_relaxation_computations(
            settings, save_bands, &*pot, aux_info, &phonopy, &coords, meta.sift(),
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
        super_coords, sc, prim_displacements, perm_from_phonopy, ..
    } = disp_dir.rsp2_style_displacements()?;

    let (prim_coords, prim_meta) = symmetry_dir.structure()?;
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

    let original_force_sets: Vec<_> = {
        let ::phonopy::ForceSets {
            force_sets: phonopy_force_sets, ..
        } = forces_dir.force_sets()?;

        phonopy_force_sets.into_iter()
            .map(|vec| {
                vec.into_iter().enumerate()
                    .map(|(atom, v3)| (perm_from_phonopy.permute_index(atom), v3))
                    .collect()
            }).collect()
    };

    let cart_rots: Vec<M33> = {
        space_group.iter()
            .map(|oper| oper.to_rot().cart(&prim_lattice))
            .collect()
    };

    trace!("Computing designated rows of force constants...");
    let force_constants = ::math::dynmat::ForceConstants::like_phonopy(
        &prim_displacements,
        &original_force_sets,
        &cart_rots,
        &space_group_deperms,
        &sc,
    ).unwrap();
    trace!("Done computing designated rows of force constants.");

    {
        // let dense = force_constants.permuted_by(&perm_to_phonopy).to_dense_matrix();
        // println!("{:?}", dense); // FINALLY: THE MOMENT OF TRUTH
    }

    {
        trace!("Computing dynamical matrix...");
        let our_dynamical_matrix = force_constants.gamma_dynmat(&sc, prim_meta.pick()).hermitianize();
        trace!("Done computing dynamical matrix.");
        println!("{:?}", our_dynamical_matrix.0.to_coo().map(|c| c.0).into_dense());


        trace!("Computing eigensolutions...");
        let (low, _low_basis) = our_dynamical_matrix.compute_most_negative_eigensolutions(30)?;
        let (high, _high_basis) = our_dynamical_matrix.compute_most_extreme_eigensolutions(30)?;
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
    layer_search: Option<&cfg::LayerSearch>,
    file_format: StructureFileType,
    input: &PathFile,
) -> FailResult<(ScalableCoords, Rc<[Element]>, Option<Vec<usize>>, Option<Vec<ScMatrix>>)> {
    let (output, atom_elements, atom_layers, layer_sc_mats);
    match file_format {
        StructureFileType::Poscar => {
            use ::rsp2_structure_io::Poscar;

            let Poscar { coords, elements, .. } = Poscar::from_reader(input.read()?)?;

            atom_elements = elements.into();

            if let Some(cfg) = layer_search {
                let layers = perform_layer_search(cfg, &coords)?;
                output = ScalableCoords::from_layer_search_results(coords, cfg, &layers);
                atom_layers = Some(layers.by_atom());
                // We could do a primitive cell search, but anything using our results would have
                //  trouble interpreting our results if we chose a different cell from expected.
                // Thus, anything using sc matrices requires them to be supplied in advance.
                layer_sc_mats = None;
            } else {
                output = ScalableCoords::from_unlayered(coords);
                atom_layers = None;
                layer_sc_mats = None;
            }
        },
        StructureFileType::LayersYaml => {
            use ::rsp2_structure_io::layers_yaml::load;
            use ::rsp2_structure_io::layers_yaml::load_layer_sc_info;

            let layer_builder = load(input.read()?)?;

            layer_sc_mats = Some({
                load_layer_sc_info(input.read()?)?
                    .into_iter()
                    .map(|(matrix, periods, _)| ScMatrix::new(&matrix, &periods))
                    .collect_vec()
            });

            atom_layers = Some(layer_builder.atom_layers());
            atom_elements = vec![CARBON; layer_builder.num_atoms()];

            output = ScalableCoords::KnownLayers { layer_builder };
        },
    }
    let atom_elements = atom_elements.into();
    Ok((output, atom_elements, atom_layers, layer_sc_mats))
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

fn read_poscar(
    cfg_masses: Option<&cfg::Masses>,
    r: impl ::std::io::Read,
) -> FailResult<(Coords, HList2<Rc<[Element]>, Rc<[Mass]>>)> {
    use ::rsp2_structure_io::Poscar;

    let Poscar { coords, elements, .. } = Poscar::from_reader(r)?;
    let elements: Rc<[Element]> = elements.into();
    let masses = masses_by_config(cfg_masses, elements.clone())?;
    Ok((coords, hlist![elements, masses]))
}

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
            None => ::common::element_mass(element).map(Mass),
        })
        .collect::<Result<Vec<_>, _>>()?.into()
})}
