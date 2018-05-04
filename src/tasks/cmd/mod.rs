// HERE BE DRAGONS

pub(crate) mod integrate_2d;

use self::lammps::{LammpsBuilder};
mod lammps;

use self::ev_analyses::GammaSystemAnalysis;
mod ev_analyses;

use self::trial::TrialDir;
pub(crate) mod trial;

mod acoustic_search;
mod relaxation;

use ::{FailResult, FailOk};
use ::rsp2_tasks_config::{self as cfg, Settings, NormalizationMode, SupercellSpec};
use ::traits::{AsPath};
use ::phonopy::{DirWithBands, DirWithDisps, DirWithForces};

use ::util::{tup2};
use ::util::ext_traits::{OptionResultExt, PathNiceExt};
use ::math::basis::Basis3;
use ::math::bonds::Bonds;

use ::path_abs::{PathArc, PathFile, PathDir};
use ::rsp2_structure::consts::CARBON;
use ::rsp2_slice_math::{vnorm};

use ::slice_of_array::prelude::*;
use ::rsp2_array_utils::arr_from_fn;
use ::rsp2_array_types::{V3, Unvee};
use ::rsp2_structure::{CoordStructure, ElementStructure, Structure};
use ::rsp2_structure::{Lattice};
use ::rsp2_structure::supercell;
use ::phonopy::Builder as PhonopyBuilder;
use ::math::bands::ScMatrix;
use ::rsp2_structure_io::poscar;

use ::rsp2_fs_util::{rm_rf};

use ::std::io::{Write};

use ::itertools::Itertools;

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
        let lmp = LammpsBuilder::new(&settings.threading, &settings.potential.kind);

        // FIXME: rather than doing this here, other rsp2 binaries ought to be able
        //        to support the other file formats as well.
        //
        //        That said, the REASON for doing it here is that, currently, a
        //        number of optional computational steps are toggled on/off based
        //        on the input file format.
        let (original_structure, atom_layers, layer_sc_mats) = read_structure_file(
            Some(settings), file_format, input, Some(&lmp),
        )?;

        self.write_poscar("initial.vasp", "Initial structure", &original_structure)?;
        let phonopy = phonopy_builder_from_settings(&settings.phonons, original_structure.lattice());

        let (structure, ev_analysis, final_bands_dir) = self.do_main_ev_loop(
            settings, &cli, &lmp, &atom_layers, &layer_sc_mats,
            &phonopy, original_structure,
        )?;
        let _do_not_drop_the_bands_dir = final_bands_dir;

        self.write_poscar("final.vasp", "Final structure", &structure)?;

        write_eigen_info_for_machines(&ev_analysis, self.create_file("eigenvalues.final")?)?;

        self.write_ev_analysis_output_files(settings, &lmp, &ev_analysis)?;
    })}

    fn write_ev_analysis_output_files(
        &self,
        settings: &Settings,
        lmp: &LammpsBuilder,
        eva: &GammaSystemAnalysis,
    ) -> FailResult<()>
    {Ok({
        if let (&Some(ref frequency), &Some(ref raman)) = (&eva.ev_frequencies, &eva.ev_raman_tensors) {
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

        if let (&Some(ref sc_mats), &Some(ref unfold_probs)) = (&eva.layer_sc_mats, &eva.unfold_probs) {
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

        self.write_summary_file(settings, lmp, eva)?;
    })}

    // log when writing poscar files, especially during loops
    // (to remove any doubt about the iteration number)
    fn write_poscar(
        &self,
        filename: &str,
        headline: &str,
        structure: &ElementStructure,
    ) -> FailResult<()>
    {Ok({
        use ::rsp2_structure_io::poscar;
        let file = self.create_file(filename)?;
        trace!("Writing '{}'", file.path().nice());
        poscar::dump(file, headline, &structure)?;
    })}

    fn do_post_relaxation_computations(
        &self,
        settings: &Settings,
        save_bands: Option<&PathArc>,
        lmp: &LammpsBuilder,
        aux_info: aux_info::Info,
        phonopy: &PhonopyBuilder,
        structure: &ElementStructure,
    ) -> FailResult<(DirWithBands<Box<AsPath>>, Vec<f64>, Basis3, GammaSystemAnalysis)>
    {Ok({

        let bands_dir = do_diagonalize(
            lmp, &settings.threading, phonopy, structure, save_bands, &[Q_GAMMA],
        )?;
        let (evals, evecs) = read_eigensystem(&bands_dir, &Q_GAMMA)?;

        trace!("============================");
        trace!("Finished diagonalization");

        // NOTE: in order to reuse the results of the bond search between iterations,
        //       we would need to store image indices so that the correct cartesian
        //       vectors can be computed.
        //       For now, we just do it all from scratch each time.
        let bonds = settings.bond_radius.map(|bond_radius| FailOk({
            trace!("Computing bonds");
            Bonds::from_brute_force_very_dumb(&structure, bond_radius)?
        })).fold_ok()?;

        trace!("Computing eigensystem info");

        let ev_analysis = {
            use self::ev_analyses::*;

            let classifications = acoustic_search::perform_acoustic_search(
                &lmp, &evals, &evecs, &structure, settings,
            )?;

            let aux_info::Info { atom_layers, atom_masses, layer_sc_mats } = aux_info;

            gamma_system_analysis::Input {
                atom_masses,
                atom_layers,
                layer_sc_mats,
                ev_classifications: Some(EvClassifications(classifications)),
                atom_elements:      Some(AtomElements(structure.metadata().to_vec())),
                atom_coords:        Some(AtomCoordinates(structure.map_metadata_to(|_| ()))),
                ev_frequencies:     Some(EvFrequencies(evals.clone())),
                ev_eigenvectors:    Some(EvEigenvectors(evecs.clone())),
                bonds:              bonds.map(Bonds),
            }.compute()?
        };
        (bands_dir, evals, evecs, ev_analysis)
    })}
}

#[allow(unused)]
fn do_diagonalize_at_gamma(
    lmp: &LammpsBuilder,
    threading: &cfg::Threading,
    phonopy: &PhonopyBuilder,
    structure: &ElementStructure,
    save_bands: Option<&PathArc>,
) -> FailResult<(Vec<f64>, Basis3)>
{Ok({
    let dir = do_diagonalize(lmp, threading, phonopy, structure, save_bands, &[Q_GAMMA])?;
    read_eigensystem(&dir, &Q_GAMMA)?
})}

fn do_diagonalize(
    lmp: &LammpsBuilder,
    threading: &cfg::Threading,
    phonopy: &PhonopyBuilder,
    structure: &ElementStructure,
    save_bands: Option<&PathArc>,
    points: &[V3],
) -> FailResult<DirWithBands<Box<AsPath>>>
{Ok({
    let disp_dir = phonopy.displacements(&structure)?;
    let force_sets = do_force_sets_at_disps(&lmp, &threading, &disp_dir)?;

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

fn write_eigen_info_for_machines<W: Write>(
    analysis: &GammaSystemAnalysis,
    mut file: W,
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
        lmp: &LammpsBuilder,
        ev_analysis: &GammaSystemAnalysis,
    ) -> FailResult<()> {Ok({
        use ::ui::cfg_merging::{make_nested_mapping, no_summary, merge_summaries};
        use ::rsp2_structure_io::poscar;

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
            let f = |structure: ElementStructure| FailOk({
                let na = structure.num_atoms() as f64;
                lmp.build(structure)?.compute_value()? / na
            });
            let f_path = |s: &AsPath| FailOk(f(poscar::load(self.read_file(s)?)?)?);

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
    lmp: &LammpsBuilder,
    threading: &cfg::Threading,
    disp_dir: &DirWithDisps<P>,
) -> FailResult<Vec<Vec<V3>>>
{Ok({
    use ::std::io::prelude::*;
    use ::rayon::prelude::*;

    trace!("Computing forces at displacements");

    let counter = ::util::AtomicCounter::new();
    let compute = move |structure| FailOk({
        let i = counter.inc();
        eprint!("\rdisp {} of {}", i + 1, disp_dir.displacements().len());
        ::std::io::stderr().flush().unwrap();

        lmp.build(structure)?.compute_force()?
    });

    let force_sets = match threading {
        &cfg::Threading::Lammps |
        &cfg::Threading::Serial => {
            disp_dir.displaced_structures().map(compute).collect::<FailResult<Vec<_>>>()?
        },
        &cfg::Threading::Rayon => {
            let structures = disp_dir.displaced_structures().collect::<Vec<_>>();
            structures.into_par_iter().map(compute).collect::<FailResult<Vec<_>>>()?
        },
    };
    eprintln!();
    force_sets
})}

fn read_eigensystem<P: AsPath>(
    bands_dir: &DirWithBands<P>,
    q: &V3,
) -> FailResult<(Vec<f64>, Basis3)>
{Ok({
    let index = ::util::index_of_nearest(&bands_dir.q_positions()?, q, 1e-4);
    let index = match index {
        Some(i) => i,
        None => bail!("Bands do not include kpoint:\n  dir: {}\npoint: {:?}",
            bands_dir.path().display(), q),
    };

    let evals = bands_dir.eigenvalues()?.remove(index);
    let evecs = match bands_dir.eigenvectors()? {
        None => bail!("Directory has no eigenvectors: {}", bands_dir.path().display()),
        Some(mut evs) => Basis3::from_basis(evs.remove(index)),
    };

    (evals, evecs)
})}

//-----------------------------------

// HACK:
// These are only valid for a hexagonal system represented
// with the [[a, 0], [-a/2, a sqrt(3)/2]] lattice convention
#[allow(unused)] const Q_GAMMA: V3 = V3([0.0, 0.0, 0.0]);
#[allow(unused)] const Q_K: V3 = V3([1f64/3.0, 1.0/3.0, 0.0]);
#[allow(unused)] const Q_K_PRIME: V3 = V3([2.0 / 3f64, 2.0 / 3f64, 0.0]);

// HACK: These adapters are temporary to help the existing code
//       (written only with carbon in mind) adapt to Structure.
#[allow(unused)]
fn carbon(structure: &CoordStructure) -> ElementStructure
{
    // I want it PAINTED BLACK!
    structure.map_metadata_to(|_| CARBON)
}

#[allow(unused)] // if this isn't used, I'm *glad*
fn uncarbon(structure: &ElementStructure) -> CoordStructure
{
    assert!(structure.metadata().iter().all(|&e| e == CARBON),
        "if you want to use this code on non-carbon stuff, you better \
        do something about all the carbon-specific code.  Look for calls \
        to `carbon()`");
    structure.map_metadata_to(|_| ())
}

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

        let structure = bands_dir.structure()?;
        let (evals, evecs) = read_eigensystem(&bands_dir, &Q_GAMMA)?;

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

        let (xmin, xmax) = tup2(settings.xlim);
        let (ymin, ymax) = tup2(settings.ylim);
        let (w, h) = tup2(settings.dim);
        let data = {
            ::cmd::integrate_2d::integrate_two_eigenvectors(
                (w, h),
                &structure.to_carts(),
                (xmin..xmax, ymin..ymax),
                (&get_real_ev(plot_ev_indices.0), &get_real_ev(plot_ev_indices.1)),
                {
                    use ::std::sync::atomic::{AtomicUsize, Ordering};
                    let counter = AtomicUsize::new(0);

                    move |pos| {FailOk({
                        let i = counter.fetch_add(1, Ordering::SeqCst);
                        // println!("{:?}", pos.flat().iter().sum::<f64>());

                        eprint!("\rdatapoint {:>6} of {}", i, w * h);
                        let lmp = LammpsBuilder::new(&settings.threading, &settings.potential);
                        let mut lmp = lmp.build(structure.clone())?;
                        lmp.set_carts(&pos)?;
                        lmp.compute_grad()?
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
        use ::rsp2_structure_io::poscar;

        let lmp = LammpsBuilder::new(&settings.threading, &settings.potential.kind);

        let structure = poscar::load(self.read_file("./final.vasp")?)?;
        let phonopy = phonopy_builder_from_settings(&settings.phonons, structure.lattice());
        do_diagonalize(
            &lmp, &settings.threading, &phonopy, &structure,
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
        use ::rsp2_structure_io::poscar;

        let lmp = LammpsBuilder::new(&settings.threading, &settings.potential.kind);

        let structure = poscar::load(self.read_file("./final.vasp")?)?;
        let phonopy = phonopy_builder_from_settings(&settings.phonons, structure.lattice());

        let aux_info = self.load_analysis_aux_info()?;

        let save_bands = None;
        let (_, _, _, ev_analysis) = self.do_post_relaxation_computations(
            settings, save_bands, &lmp, aux_info, &phonopy, &structure,
        )?;

        self.write_ev_analysis_output_files(settings, &lmp, &ev_analysis)?;
    })}
}

//=================================================================

impl TrialDir {
    pub(crate) fn run_dynmat_test(
        self,
        settings: &Settings,
        file_format: StructureFileType,
        input: &PathFile,
        cli: CliArgs,
    ) -> FailResult<()>
    {Ok({
        let lmp = LammpsBuilder::new(&settings.threading, &settings.potential.kind);

        // FIXME: rather than doing this here, other rsp2 binaries ought to be able
        //        to support the other file formats as well.
        //
        //        That said, the REASON for doing it here is that, currently, a
        //        number of optional computational steps are toggled on/off based
        //        on the input file format.
        let (prim_structure, _atom_layers, _layer_sc_mats) = read_structure_file(
            Some(settings), file_format, input, Some(&lmp),
        )?;

        self.write_poscar("initial.vasp", "Initial structure", &prim_structure)?;
        let phonopy = phonopy_builder_from_settings(&settings.phonons, prim_structure.lattice());
        let disp_dir = phonopy.displacements(&prim_structure)?;

        let (superstructure, sc_token) = {
            let sc_dims = settings.phonons.supercell.dim_for_unitcell(prim_structure.lattice());
            assert!(
                sc_dims.iter().all(|&x| x % 2 == 1),
                "even supercell sizes not supported here"
            );

            let (our_superstructure, sc_token) = {
                supercell::diagonal(sc_dims).build(prim_structure)
            };
            let phonopy_superstructure = disp_dir.superstructure();

            // cmon, big money, big money....
            // if these assertions always succeed, it will save us a
            // good deal of implementation work.
            let err_msg = "\
                phonopy's superstructure does not match rsp2's conventions! \
                Unfortunately, support for this scenario is not yet implemented.\
            ";
            assert_close!(
                abs=1e-10,
                our_superstructure.lattice().matrix().unvee(),
                phonopy_superstructure.lattice().matrix().unvee(),
                "{}", err_msg,
            );
            assert_close!(
                abs=1e-10,
                our_superstructure.to_carts().unvee(),
                phonopy_superstructure.to_carts().unvee(),
                "{}", err_msg,
            );
            let _ = phonopy_superstructure;
            (our_superstructure, sc_token)
        };

        let _ = superstructure;
        let _ = sc_token;
        let _ = cli;

        unimplemented!();
        let our_dynamical_matrix = {
        };

        // make phonopy compute dynamical matrix (as a gold standard)
        // TODO: how to get the dynamical matrix from phonopy?
        let phonopy_bands_dir = {
            //force_dir
            //    .build_bands()
            //    .compute(&[Q_GAMMA])?
        };

        let _ = our_dynamical_matrix;
        let _ = phonopy_bands_dir;
    })}
}

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

//     let lmp = LammpsBuilder::new(&cfg::Threading::Lammps, &potential);

//     create_dir(&outdir)?;
//     {
//         // dumb/lazy solution to ensuring all output files go in the dir
//         let cwd_guard = push_dir(outdir)?;
//         GlobalLogger::default()
//             .path("rsp2.log")
//             .apply()?;

//         poscar::dump(create("./input.vasp")?, "", &structure)?;

//         let disp_dir = phonopy.displacements(&structure)?;
//         let force_sets = do_force_sets_at_disps(&lmp, &cfg::Threading::Rayon, &disp_dir)?;
//         disp_dir.make_force_dir_in_dir(&force_sets, ".")?;

//         cwd_guard.pop()?;
//     }
// })}

//=================================================================

pub(crate) fn read_structure_file(
    // FIXME this option is dumb; basically, it's so that code can use this function
    //       even without needing an entire settings struct.
    settings: Option<&Settings>,
    file_format: StructureFileType,
    input: &PathFile,
    // will be used to optimize parameters for layers.yaml if provided.
    // will be ignored for poscars.
    // FIXME: that's really inconsistent and dumb.
    //        (a function like this really shouldn't do ANY optimization,
    //         but I don't see any other place to do it without mega refactoring...)
    lmp: Option<&LammpsBuilder>,
) -> FailResult<(ElementStructure, Option<Vec<usize>>, Option<Vec<ScMatrix>>)> {
    let (original_structure, atom_layers, layer_sc_mats);
    match file_format {
        StructureFileType::Poscar => {
            original_structure = poscar::load(input.read()?)?;
            atom_layers = None;
            layer_sc_mats = None;
        },
        StructureFileType::LayersYaml => {
            use ::rsp2_structure_gen::load_layers_yaml;
            use ::rsp2_structure_gen::layer_sc_info_from_layers_yaml;

            let mut layer_builder = load_layers_yaml(input.read()?)?;
            if let (Some(settings), Some(lmp)) = (settings, lmp) {
                layer_builder = self::relaxation::optimize_layer_parameters(
                    &settings.scale_ranges, lmp, layer_builder,
                )?;
            }
            original_structure = carbon(&layer_builder.assemble());

            layer_sc_mats = Some({
                layer_sc_info_from_layers_yaml(input.read()?)?
                    .into_iter()
                    .map(|(matrix, periods, _)| ScMatrix::new(&matrix, &periods))
                    .collect_vec()
            });

            // FIXME: This is entirely unnecessary. We just read layers.yaml; we should
            //        be able to get the layer assignments without a search like this!
            atom_layers = Some({
                trace!("Finding layers");
                let layers =
                    ::rsp2_structure::find_layers(&original_structure, &V3([0, 0, 1]), 0.25)?
                        .per_unit_cell().expect("Structure is not layered?");

                if let Some(settings) = settings {
                    if let Some(expected) = settings.layers {
                        assert_eq!(expected, layers.len() as u32);
                    }
                }
                layers.by_atom()
            });
        },
    }
    Ok((original_structure, atom_layers, layer_sc_mats))
}
