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

use ::errors::{Error, ErrorKind, Result, ok};
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
use ::phonopy::Builder as PhonopyBuilder;
use ::math::bands::ScMatrix;

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
    ) -> Result<()>
    {ok({
        use ::rsp2_structure_io::poscar;

        let lmp = LammpsBuilder::new(&settings.threading, &settings.potential.kind);

        // FIXME: rather than doing this here, other rsp2 binaries ought to be able
        //        to support the other file formats as well.
        //
        //        That said, the REASON for doing it here is that, currently, a
        //        number of optional computational steps are toggled on/off based
        //        on the input file format.
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

                let layer_builder = load_layers_yaml(input.read()?)?;
                let layer_builder = self::relaxation::optimize_layer_parameters(
                    &settings.scale_ranges, &lmp, layer_builder,
                )?;
                original_structure = carbon(&layer_builder.assemble());

                layer_sc_mats = Some({
                    layer_sc_info_from_layers_yaml(input.read()?)?
                        .into_iter()
                        .map(|(matrix, periods, _)| {
                            ::math::bands::ScMatrix::new(&matrix, &periods)
                        }).collect_vec()
                });

                // FIXME: This is entirely unnecessary. We just read layers.yaml; we should
                //        be able to get the layer assignments without a search like this!
                atom_layers = Some({
                    trace!("Finding layers");
                    let layers =
                        ::rsp2_structure::find_layers(&original_structure, &V3([0, 0, 1]), 0.25)?
                            .per_unit_cell().expect("Structure is not layered?");

                    if let Some(expected) = settings.layers {
                        assert_eq!(expected, layers.len() as u32);
                    }
                    layers.by_atom()
                });
            },
        }

        self.write_poscar("initial.vasp", "Initial structure", &original_structure)?;
        let phonopy = phonopy_builder_from_settings(&settings.phonons, &original_structure);
        let phonopy = phonopy.use_sparse_sets(settings.tweaks.sparse_sets);

        let (structure, ev_analysis, final_bands_dir) = self.do_main_ev_loop(
            settings, &cli, &lmp, &atom_layers, &layer_sc_mats,
            &phonopy, original_structure,
        )?;
        let _do_not_drop_the_bands_dir = final_bands_dir;

        self.write_poscar("final.vasp", "Final structure", &structure)?;

        write_eigen_info_for_machines(&ev_analysis, self.create_file("eigenvalues.final")?)?;

        // HACK
        if let (&Some(ref freqs), &Some(ref raman)) = (&ev_analysis.ev_frequencies, &ev_analysis.ev_raman_intensities) {
            let pairs = ::util::zip_eq(freqs.0.to_vec(), raman.0.to_vec()).collect_vec();
            ::serde_json::to_writer(self.create_file("raman.json")?, &pairs)?;
        }

        self.write_summary_file(settings, &lmp, &ev_analysis)?;
    })}

    fn write_poscar(&self, filename: &str, headline: &str, structure: &ElementStructure) -> Result<()>
    {ok({
        use ::rsp2_structure_io::poscar;
        let file = self.create_file(filename)?;
        trace!("Writing '{}'", file.path().nice());
        poscar::dump(file, headline, &structure)?;
    })}

    fn do_post_relaxation_computations(
        &self,
        settings: &Settings,
        cli: &CliArgs,
        lmp: &LammpsBuilder,
        atom_layers: &Option<Vec<usize>>,
        layer_sc_mats: &Option<Vec<ScMatrix>>,
        phonopy: &PhonopyBuilder,
        structure: &ElementStructure,
    ) -> Result<(DirWithBands<Box<AsPath>>, Vec<f64>, Basis3, GammaSystemAnalysis)>
    {ok({

        let bands_dir = do_diagonalize(
            lmp, &settings.threading, phonopy, structure,
            &(match cli.save_bands {
                true => Some(self.save_bands_dir()),
                false => None,
            }),
            &[Q_GAMMA],
        )?;
        let (evals, evecs) = read_eigensystem(&bands_dir, &Q_GAMMA)?;

        trace!("============================");
        trace!("Finished diagonalization");

        // NOTE: in order to reuse the results of the bond search between iterations,
        //       we would need to store image indices so that the correct cartesian
        //       vectors can be computed.
        //       For now, we just do it all from scratch each time.
        let bonds = settings.bond_radius.map(|bond_radius| ok({
            trace!("Computing bonds");
            Bonds::from_brute_force_very_dumb(&structure, bond_radius)?
        })).fold_ok()?;

        trace!("Computing eigensystem info");

        let ev_analysis = {
            use self::ev_analyses::*;

            let classifications = acoustic_search::perform_acoustic_search(
                &lmp, &evals, &evecs, &structure, settings,
            )?;

            let masses = {
                structure.metadata().iter()
                    .map(|&s| ::common::element_mass(s))
                    .collect()
            };

            gamma_system_analysis::Input {
                ev_classifications: &Some(EvClassifications(classifications)),
                atom_masses:        &Some(AtomMasses(masses)),
                atom_elements:      &Some(AtomElements(structure.metadata().to_vec())),
                atom_coords:        &Some(AtomCoordinates(structure.map_metadata_to(|_| ()))),
                atom_layers:        &atom_layers.clone().map(AtomLayers),
                layer_sc_mats:      &layer_sc_mats.clone().map(LayerScMatrices),
                ev_frequencies:     &Some(EvFrequencies(evals.clone())),
                ev_eigenvectors:    &Some(EvEigenvectors(evecs.clone())),
                bonds: &bonds.map(Bonds),
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
    save_bands: &Option<PathArc>,
) -> Result<(Vec<f64>, Basis3)>
{Ok({
    let dir = do_diagonalize(lmp, threading, phonopy, structure, save_bands, &[Q_GAMMA])?;
    read_eigensystem(&dir, &Q_GAMMA)?
})}

fn do_diagonalize(
    lmp: &LammpsBuilder,
    threading: &cfg::Threading,
    phonopy: &PhonopyBuilder,
    structure: &ElementStructure,
    save_bands: &Option<PathArc>,
    points: &[V3],
) -> Result<DirWithBands<Box<AsPath>>>
{ok({
    let disp_dir = phonopy.displacements(&structure)?;
    let force_sets = do_force_sets_at_disps(&lmp, &threading, &disp_dir)?;

    let bands_dir = disp_dir
        .make_force_dir(&force_sets)?
        .build_bands()
        .eigenvectors(true)
        .compute(&points)?;

    if let Some(ref save_dir) = *save_bands {
        rm_rf(save_dir)?;
        bands_dir.relocate(save_dir)?.boxed()
    } else {
        bands_dir.boxed()
    }
})}

fn write_eigen_info_for_humans(
    analysis: &GammaSystemAnalysis,
    writeln: &mut FnMut(String) -> Result<()>,
) -> Result<()>
{
    analysis.make_columns(ev_analyses::ColumnsMode::ForHumans)
        .expect("(bug) no columns, not even frequency?")
        .into_iter().map(writeln).collect()
}

fn write_eigen_info_for_machines<W: Write>(
    analysis: &GammaSystemAnalysis,
    mut file: W,
) -> Result<()>
{
    analysis.make_columns(ev_analyses::ColumnsMode::ForMachines)
        .expect("(bug) no columns, not even frequency?")
        .into_iter().map(|s| ok(writeln!(file, "{}", s)?)).collect()
}

impl TrialDir {
    fn write_summary_file(
        &self,
        settings: &Settings,
        lmp: &LammpsBuilder,
        ev_analysis: &GammaSystemAnalysis,
    ) -> Result<()> {ok({
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
            let f = |structure: ElementStructure| ok({
                let na = structure.num_atoms() as f64;
                lmp.build(structure)?.compute_value()? / na
            });
            let f_path = |s: &AsPath| ok(f(poscar::load(self.open(s)?)?)?);

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

fn phonopy_builder_from_settings<M>(
    settings: &cfg::Phonons,
    // the structure is needed to resolve the correct supercell size
    structure: &Structure<M>,
) -> PhonopyBuilder {
    PhonopyBuilder::new()
        .symmetry_tolerance(settings.symmetry_tolerance)
        .conf("DISPLACEMENT_DISTANCE", format!("{:e}", settings.displacement_distance))
        .supercell_dim(settings.supercell.dim_for_unitcell(structure.lattice()))
        .conf("DIAG", ".FALSE.")
}

fn do_force_sets_at_disps<P: AsPath + Send + Sync>(
    lmp: &LammpsBuilder,
    threading: &cfg::Threading,
    disp_dir: &DirWithDisps<P>,
) -> Result<Vec<Vec<V3>>>
{ok({
    use ::std::io::prelude::*;
    use ::rayon::prelude::*;

    trace!("Computing forces at displacements");

    let counter = ::util::AtomicCounter::new();
    let compute = move |structure| ok({
        let i = counter.inc();
        eprint!("\rdisp {} of {}", i + 1, disp_dir.displacements().len());
        ::std::io::stderr().flush().unwrap();

        lmp.build(structure)?.compute_force()?
    });

    let force_sets = match threading {
        &cfg::Threading::Lammps |
        &cfg::Threading::Serial => {
            disp_dir.displaced_structures().map(compute).collect::<Result<Vec<_>>>()?
        },
        &cfg::Threading::Rayon => {
            let structures = disp_dir.displaced_structures().collect::<Vec<_>>();
            structures.into_par_iter().map(compute).collect::<Result<Vec<_>>>()?
        },
    };
    eprintln!();
    force_sets
})}

fn read_eigensystem<P: AsPath>(
    bands_dir: &DirWithBands<P>,
    q: &V3,
) -> Result<(Vec<f64>, Basis3)>
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

impl TrialDir {
    pub(crate) fn run_energy_surface(
        self,
        settings: &cfg::EnergyPlotSettings,
        input: &PathDir,
    ) -> Result<()>
    {ok({
        // support either a force dir or a bands dir as input
        let bands_dir = match DirWithBands::from_existing(input.to_path_buf()) {
            // accept a bands dir
            Ok(dir) => dir.boxed(),
            // try computing gamma bands from a force dir
            Err(Error(ErrorKind::MissingFile(..), _)) => {
                DirWithForces::from_existing(&input)?
                    .build_bands()
                    .eigenvectors(true)
                    .compute(&[Q_GAMMA])?
                    .boxed()
            },
            Err(e) => return Err(e),
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

                    move |pos| {ok({
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
    ) -> Result<()>
    {Ok({
        use ::rsp2_structure_io::poscar;

        let lmp = LammpsBuilder::new(&settings.threading, &settings.potential.kind);

        let original = poscar::load(self.open("./final.vasp")?)?;
        let phonopy = phonopy_builder_from_settings(&settings.phonons, &original);
        let phonopy = phonopy.use_sparse_sets(settings.tweaks.sparse_sets);
        do_diagonalize(
            &lmp, &settings.threading, &phonopy, &original,
            &Some(self.save_bands_dir()),
            &[Q_GAMMA, Q_K],
        )?;
    })}
}

impl TrialDir {
    pub fn save_bands_dir(&self) -> PathArc
    { self.join(SAVE_BANDS_DIR).into() }
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
