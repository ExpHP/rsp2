// HERE BE DRAGONS

pub(crate) mod integrate_2d;

use self::lammps::{Lammps, LammpsBuilder};
mod lammps;

mod ev_analyses;
//mod ev_analyses3;
//use self::ev_analyses3 as ev_analyses;

use self::ev_analyses::GammaSystemAnalysis;
//mod ev_analyses;

use self::trial::TrialDir;
pub(crate) mod trial;

use ::errors::{Error, ErrorKind, Result, ok};
use ::rsp2_tasks_config::{self as cfg, Settings, NormalizationMode, SupercellSpec};
use ::traits::{AsPath};
use ::phonopy::{DirWithBands, DirWithDisps, DirWithForces};

use ::util::ext_traits::{OptionResultExt, PathNiceExt};
use ::types::{Basis3};
use ::math::bonds::Bonds;

use ::path_abs::{PathFile, PathDir};
use ::rsp2_structure::consts::CARBON;
use ::rsp2_slice_math::{v, V, vdot, vnorm};

use ::slice_of_array::prelude::*;
use ::rsp2_array_utils::arr_from_fn;
use ::rsp2_array_types::{V3, Unvee};
use ::rsp2_structure::supercell::{self, SupercellToken};
use ::rsp2_structure::{CoordStructure, ElementStructure, Structure};
use ::rsp2_structure::{Lattice, Coords};
use ::rsp2_structure_gen::Assemble;
use ::phonopy::Builder as PhonopyBuilder;
use ::math::bands::ScMatrix;

use ::rsp2_fs_util::{rm_rf};

use ::std::io::{Write};
use ::std::path::{Path};

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

        // let mut original = poscar::load(open(input)?)?;
        // original.scale_vecs(&settings.hack_scale);

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

                let builder = load_layers_yaml(input.read()?)?;
                let builder = optimize_layer_parameters(&settings.scale_ranges, &lmp, builder)?;
                original_structure = carbon(&builder.assemble());

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

        let mut from_structure = original_structure;
        let mut loop_state = EvLoopFsm::new(&settings.ev_loop);

        // NOTE: we use break with value
        let (structure, ev_analysis) = loop {
            // move out of from_structure so that Rust's control-flow analysis
            // will make sure we put something back.
            let structure = from_structure;
            let iteration = loop_state.iteration;

            trace!("============================");
            trace!("Begin relaxation # {}", iteration);

            let structure = do_relax(&lmp, &settings.cg, &settings.potential, structure)?;
            
            trace!("============================");

            self.write_poscar(
                &format!("structure-{:02}.1.vasp", iteration),
                &format!("Structure after CG round {}", iteration),
                &structure,
            )?;

            let (bands_dir, evals, evecs, ev_analysis) = self.do_post_relaxation_computations(
                settings, &cli, &lmp, &atom_layers, &layer_sc_mats,
                &phonopy, &structure,
            )?;
            let _do_not_drop_the_bands_dir = bands_dir;

            {
                let file = self.create_file(format!("eigenvalues.{:02}", iteration))?;
                write_eigen_info_for_machines(&ev_analysis, file)?;
                write_eigen_info_for_humans(&ev_analysis, &mut |s| ok(info!("{}", s)))?;
            }

            let (structure, did_chasing) = self.maybe_do_ev_chasing(
                &settings, &lmp, structure, &ev_analysis, &evals, &evecs,
            )?;

            self.write_poscar(
                &format!("structure-{:02}.2.vasp", iteration),
                &format!("Structure after eigenmode-chasing round {}", iteration),
                &structure,
            )?;

            warn_on_improvable_lattice_params(&lmp, &structure)?;

            match loop_state.step(did_chasing) {
                EvLoopStatus::KeepGoing => {
                    from_structure = structure;
                    continue;
                },
                EvLoopStatus::Done => {
                    break (structure, ev_analysis);
                },
                EvLoopStatus::ItsBadGuys(msg) => {
                    bail!("{}", msg);
                },
            }
            // unreachable
        }; // (structure, ev_analysis, final_bands_dir)

        self.write_poscar("final.vasp", "Final structure", &structure)?;

        write_eigen_info_for_machines(&ev_analysis, self.create_file("eigenvalues.final")?)?;

        // let (k_evals, k_evecs) = read_eigensystem(&bands_dir, &Q_K)?;
        // let kinfos = get_k_eigensystem_info(
        //     &k_evals, &k_evecs, &layers[..], &structure, Some(&layer_sc_mats),
        // );

        // HACK
        if let (&Some(ref freqs), &Some(ref raman)) = (&ev_analysis.ev_frequencies, &ev_analysis.ev_raman_intensities) {
            let pairs = ::util::zip_eq(freqs.0.to_vec(), raman.0.to_vec()).collect_vec();
            ::serde_json::to_writer(self.create_file("raman.json")?, &pairs)?;
        }

        // write_summary_file(settings, &lmp, &einfos, &kinfos)?;
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
            match cli.save_bands {
                true => Some(SAVE_BANDS_DIR.as_ref()),
                false => None,
            },
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

            let masses = {
                structure.metadata().iter()
                    .map(|&s| ::common::element_mass(s))
                    .collect()
            };

            gamma_system_analysis::Input {
                atom_masses:     &Some(AtomMasses(masses)),
                atom_elements:   &Some(AtomElements(structure.metadata().to_vec())),
                atom_coords:     &Some(AtomCoordinates(structure.map_metadata_to(|_| ()))),
                atom_layers:     &atom_layers.clone().map(AtomLayers),
                layer_sc_mats:   &layer_sc_mats.clone().map(LayerScMatrices),
                ev_frequencies:  &Some(EvFrequencies(evals.clone())),
                ev_eigenvectors: &Some(EvEigenvectors(evecs.clone())),
                bonds: &bonds.map(Bonds),
            }.compute()?
        };
        (bands_dir, evals, evecs, ev_analysis)
    })}


    fn maybe_do_ev_chasing(
        &self,
        settings: &Settings,
        lmp: &LammpsBuilder,
        structure: ElementStructure,
        ev_analysis: &GammaSystemAnalysis,
        evals: &[f64],
        evecs: &Basis3,
    ) -> Result<(ElementStructure, DidEvChasing)>
    {ok({
        let structure = structure;
        let bad_evs: Vec<_> = {
            let acousticness = ev_analysis.ev_acousticness.as_ref().expect("(bug) always computed!");
            izip!(1.., evals, &evecs.0, &acousticness.0)
                .take_while(|&(_, &freq, _, _)| freq < 0.0)
                .filter(|&(_, _, _, &acousticness)| acousticness < 0.95)
                .map(|(i, freq, evec, _)| {
                    let name = format!("band {} ({})", i, freq);
                    (name, evec.as_real_checked())
                }).collect()
        };

        match bad_evs.len() {
            0 => (structure, DidEvChasing(false)),
            n => {
                trace!("Chasing {} bad eigenvectors...", n);
                let structure = do_eigenvector_chase(
                    &lmp,
                    &settings.ev_chase,
                    &settings.potential,
                    structure,
                    &bad_evs[..],
                )?;
                (structure, DidEvChasing(true))
            }
        }
    })}
}

struct EvLoopFsm {
    config: cfg::EvLoop,
    iteration: u32,
    all_ok_count: u32,
}

pub enum EvLoopStatus {
    KeepGoing,
    Done,
    ItsBadGuys(&'static str),
}

impl EvLoopFsm {
    pub fn new(config: &cfg::EvLoop) -> Self
    { EvLoopFsm {
        config: config.clone(),
        iteration: 1,
        all_ok_count: 0,
    }}

    pub fn step(&mut self, did: DidEvChasing) -> EvLoopStatus {
        self.iteration += 1;
        match did {
            DidEvChasing(true) => {
                self.all_ok_count = 0;
                if self.iteration > self.config.max_iter {
                    if self.config.fail {
                        EvLoopStatus::ItsBadGuys("Too many relaxation steps!")
                    } else {
                        warn!("Too many relaxation steps!");
                        EvLoopStatus::Done
                    }
                } else {
                    EvLoopStatus::KeepGoing
                }
            },
            DidEvChasing(false) => {
                self.all_ok_count += 1;
                if self.all_ok_count >= self.config.min_positive_iter {
                    EvLoopStatus::Done
                } else {
                    EvLoopStatus::KeepGoing
                }
            },
        }
    }
}

pub struct DidEvChasing(bool);

fn do_relax(
    lmp: &LammpsBuilder,
    cg_settings: &cfg::Acgsd,
    potential_settings: &cfg::Potential,
    structure: ElementStructure,
) -> Result<ElementStructure>
{ok({
    let sc_dims = tup3(potential_settings.supercell.dim_for_unitcell(structure.lattice()));
    let (supercell, sc_token) = supercell::diagonal(sc_dims, structure);

    let mut lmp = lmp.with_modified_inner(|b| b.threaded(true)).build(supercell.clone())?;
    let relaxed_flat = ::rsp2_minimize::acgsd(
        cg_settings,
        supercell.to_carts().flat(),
        &mut *lammps_flat_diff_fn(&mut lmp),
    ).unwrap().position;

    let supercell = supercell.with_coords(Coords::Carts(relaxed_flat.nest().to_vec()));
    multi_threshold_deconstruct(sc_token, 1e-10, 1e-3, supercell)?
})}

#[allow(unused)]
fn do_diagonalize_at_gamma(
    lmp: &LammpsBuilder,
    threading: &cfg::Threading,
    phonopy: &PhonopyBuilder,
    structure: &ElementStructure,
    save_bands: Option<&Path>,
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
    save_bands: Option<&Path>,
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

    if let Some(save_dir) = save_bands {
        rm_rf(save_dir)?;
        bands_dir.relocate(save_dir)?.boxed()
    } else {
        bands_dir.boxed()
    }
})}

fn do_eigenvector_chase(
    lmp: &LammpsBuilder,
    chase_settings: &cfg::EigenvectorChase,
    potential_settings: &cfg::Potential,
    mut structure: ElementStructure,
    bad_evecs: &[(String, &[V3])],
) -> Result<ElementStructure>
{ok({
    match *chase_settings {
        cfg::EigenvectorChase::OneByOne => {
            for &(ref name, evec) in bad_evecs {
                let (alpha, new_structure) = do_minimize_along_evec(lmp, potential_settings, structure, &evec[..])?;
                info!("Optimized along {}, a = {:e}", name, alpha);

                structure = new_structure;
            }
            structure
        },
        cfg::EigenvectorChase::Acgsd(ref cg_settings) => {
            let evecs: Vec<_> = bad_evecs.iter().map(|&(_, ev)| ev).collect();
            do_cg_along_evecs(
                lmp,
                cg_settings,
                potential_settings,
                structure,
                &evecs[..],
            )?
        },
    }
})}

fn do_cg_along_evecs<V, I>(
    lmp: &LammpsBuilder,
    cg_settings: &cfg::Acgsd,
    potential_settings: &cfg::Potential,
    structure: ElementStructure,
    evecs: I,
) -> Result<ElementStructure>
where
    V: AsRef<[V3]>,
    I: IntoIterator<Item=V>,
{ok({
    let evecs: Vec<_> = evecs.into_iter().collect();
    let refs: Vec<_> = evecs.iter().map(|x| x.as_ref()).collect();
    _do_cg_along_evecs(lmp, cg_settings, potential_settings, structure, &refs)?
})}

fn _do_cg_along_evecs(
    lmp: &LammpsBuilder,
    cg_settings: &cfg::Acgsd,
    potential_settings: &cfg::Potential,
    structure: ElementStructure,
    evecs: &[&[V3]],
) -> Result<ElementStructure>
{ok({
    let sc_dims = tup3(potential_settings.supercell.dim_for_unitcell(structure.lattice()));
    let (mut supercell, sc_token) = supercell::diagonal(sc_dims, structure);
    let evecs: Vec<_> = evecs.iter().map(|ev| sc_token.replicate(ev)).collect();

    let flat_evecs: Vec<_> = evecs.iter().map(|ev| ev.flat()).collect();
    let init_pos = supercell.to_carts();

    let mut lmp = lmp.with_modified_inner(|b| b.threaded(true)).build(supercell.clone())?;
    let relaxed_coeffs = ::rsp2_minimize::acgsd(
        cg_settings,
        &vec![0.0; evecs.len()],
        &mut *lammps_constrained_diff_fn(&mut lmp, init_pos.flat(), &flat_evecs),
    ).unwrap().position;

    let final_flat_pos = flat_constrained_position(init_pos.flat(), &relaxed_coeffs, &flat_evecs);
    supercell.carts_mut().copy_from_slice(final_flat_pos.nest());
    multi_threshold_deconstruct(sc_token, 1e-10, 1e-3, supercell)?
})}

fn do_minimize_along_evec(
    lmp: &LammpsBuilder,
    settings: &cfg::Potential,
    structure: ElementStructure,
    evec: &[V3],
) -> Result<(f64, ElementStructure)>
{ok({
    let sc_dims = tup3(settings.supercell.dim_for_unitcell(structure.lattice()));
    let (structure, sc_token) = supercell::diagonal(sc_dims, structure);
    let evec = sc_token.replicate(evec);
    let mut lmp = lmp.with_modified_inner(|b| b.threaded(true)).build(structure.clone())?;

    let from_structure = structure;
    let direction = &evec[..];
    let from_pos = from_structure.to_carts();
    let pos_at_alpha = |alpha| {
        let V(pos) = v(from_pos.flat()) + alpha * v(direction.flat());
        pos
    };
    let alpha = ::rsp2_minimize::exact_ls(0.0, 1e-4, |alpha| {
        let gradient = lammps_flat_diff_fn(&mut lmp)(&pos_at_alpha(alpha))?.1;
        let slope = vdot(&gradient[..], direction.flat());
        ok(::rsp2_minimize::exact_ls::Slope(slope))
    })??.alpha;
    let pos = pos_at_alpha(alpha);
    let structure = from_structure.with_coords(Coords::Carts(pos.nest().to_vec()));

    (alpha, multi_threshold_deconstruct(sc_token, 1e-10, 1e-3, structure)?)
})}

fn warn_on_improvable_lattice_params(
    lmp: &LammpsBuilder,
    structure: &ElementStructure,
) -> Result<()>
{Ok({
    const SCALE_AMT: f64 = 1e-6;
    let mut lmp = lmp.build(structure.clone())?;
    let center_value = lmp.compute_value()?;

    let shrink_value = {
        let mut structure = structure.clone();
        structure.scale_vecs(&[1.0 - SCALE_AMT, 1.0 - SCALE_AMT, 1.0]);
        lmp.set_structure(structure)?;
        lmp.compute_value()?
    };

    let enlarge_value = {
        let mut structure = structure.clone();
        structure.scale_vecs(&[1.0 + SCALE_AMT, 1.0 + SCALE_AMT, 1.0]);
        lmp.set_structure(structure)?;
        lmp.compute_value()?
    };

    if shrink_value.min(enlarge_value) < center_value {
        warn!("Better value found at nearby lattice parameter:");
        warn!(" Smaller: {}", shrink_value);
        warn!(" Current: {}", center_value);
        warn!("  Larger: {}", enlarge_value);
    }
})}

fn multi_threshold_deconstruct(
    sc_token: SupercellToken,
    warn: f64,
    fail: f64,
    supercell: ElementStructure,
) -> Result<ElementStructure>
{ok({
    match sc_token.deconstruct(warn, supercell.clone()) {
        Ok(x) => x,
        Err(e) => {
            warn!("{}", e);
            sc_token.deconstruct(fail, supercell)?
        }
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

pub(crate) fn optimize_layer_parameters(
    settings: &cfg::ScaleRanges,
    lmp: &LammpsBuilder,
    mut builder: Assemble,
) -> Result<Assemble>
{ok({
    pub use ::rsp2_minimize::exact_ls::{Value, Golden};
    use ::rsp2_tasks_config::{ScaleRanges, ScaleRange, ScaleRangesLayerSepStyle};
    use ::std::cell::RefCell;

    let ScaleRanges {
        parameter: ref parameter_spec,
        layer_sep: ref layer_sep_spec,
        warn: warn_threshold,
        layer_sep_style,
        repeat_count,
    } = *settings;

    let n_seps = builder.layer_seps().len();

    // abuse RefCell for some DRYness
    let builder = RefCell::new(builder);
    {
        let builder = &builder;

        let optimizables = {
            let mut optimizables: Vec<(_, Box<Fn(f64)>)> = vec![];
            optimizables.push((
                (format!("lattice parameter"), parameter_spec.clone()),
                Box::new(|s| {
                    builder.borrow_mut().scale = s;
                }),
            ));

            match layer_sep_style {
                ScaleRangesLayerSepStyle::Individual => {
                    for i in 0..n_seps {
                        optimizables.push((
                            (format!("layer sep {}", i), layer_sep_spec.clone()),
                            Box::new(move |s| {
                                builder.borrow_mut().layer_seps()[i] = s;
                            }),
                        ));
                    }
                },
                ScaleRangesLayerSepStyle::Uniform => {
                    optimizables.push((
                        (format!("layer sep"), layer_sep_spec.clone()),
                        Box::new(move |s| {
                            for dest in builder.borrow_mut().layer_seps() {
                                *dest = s;
                            }
                        })
                    ))
                },
            }
            optimizables
        };

        // Set reasonable values before initializing LAMMPS
        //
        // exact specs: set the value
        // range specs: start with reasonable defaults ('guess' in config)
        for &((_, ref spec), ref setter) in &optimizables {
            match *spec {
                ScaleRange::Exact(value) |
                ScaleRange::Range { range: _, guess: Some(value) } => {
                    setter(value);
                },
                ScaleRange::Range { range: _, guess: None } => {},
            }
        }

        let get_value = || ok({
            lmp.build(carbon(&builder.borrow().assemble()))?.compute_value()?
        });

        // optimize them one-by-one.
        //
        // Repeat the whole process repeat_count times.
        // In future iterations, parameters other than the one currently being
        // relaxed may be set to different, better values, which may in turn
        // cause different values to be chosen for the earlier parameters.
        for _ in 0..repeat_count {
            for &((ref name, ref spec), ref setter) in &optimizables {
                trace!("Optimizing {}", name);

                let best = match *spec {
                    ScaleRange::Exact(value) => value,
                    ScaleRange::Range { guess: _, range } => {
                        let best = Golden::new()
                            .stop_condition(&from_json!({"interval-size": 1e-7}))
                            .run(range, |a| {
                                setter(a);
                                get_value().map(Value)
                            })??; // ?!??!!!?

                        if let Some(thresh) = warn_threshold {
                            // use signed differences so that all values outside violate the threshold
                            let lo = range.0.min(range.1);
                            let hi = range.0.max(range.1);
                            if (best - range.0).min(range.1 - best) / (range.1 - range.0) < thresh {
                                warn!("Relaxed value of '{}' is suspiciously close to limits!", name);
                                warn!("  lo: {:e}", lo);
                                warn!(" val: {:e}", best);
                                warn!("  hi: {:e}", hi);
                            }
                        }

                        info!("Optimized {}: {} (from {:?})", name, best, range);
                        best
                    },
                }; // let best = { ... }

                setter(best);
            } // for ... in optimizables
        } // for ... in repeat_count
    } // RefCell borrow scope

    builder.into_inner()
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

fn lammps_flat_diff_fn<'a>(lmp: &'a mut Lammps)
-> Box<FnMut(&[f64]) -> Result<(f64, Vec<f64>)> + 'a>
{
    Box::new(move |pos| ok({
        lmp.set_carts(pos.nest())?;
        lmp.compute().map(|(v, g)| (v, g.unvee().flat().to_vec()))?
    }))
}

fn flat_constrained_position(
    flat_init_pos: &[f64],
    coeffs: &[f64],
    flat_evecs: &[&[f64]],
) -> Vec<f64>
{
    let flat_d_pos = dot_vec_mat_dumb(coeffs, flat_evecs);
    let V(flat_pos): V<Vec<_>> = v(flat_init_pos) + v(flat_d_pos);
    flat_pos
}

// cg differential function along a restricted set of eigenvectors.
//
// There will be one coordinate for each eigenvector.
fn lammps_constrained_diff_fn<'a>(
    lmp: &'a mut Lammps,
    flat_init_pos: &'a [f64],
    flat_evs: &'a [&[f64]],
)
-> Box<FnMut(&[f64]) -> Result<(f64, Vec<f64>)> + 'a>
{
    let mut compute_from_3n_flat = lammps_flat_diff_fn(lmp);

    Box::new(move |coeffs| ok({
        assert_eq!(coeffs.len(), flat_evs.len());

        // This is dead simple.
        // The kth element of the new gradient is the slope along the kth ev.
        // The change in position is a sum over contributions from each ev.
        // These relationships have a simple expression in terms of
        //   the matrix whose columns are the selected eigenvectors.
        // (though the following is transposed for our row-centric formalism)
        let flat_pos = flat_constrained_position(flat_init_pos, coeffs, flat_evs);

        let (value, flat_grad) = compute_from_3n_flat(&flat_pos)?;

        let grad = dot_mat_vec_dumb(flat_evs, &flat_grad);
        (value, grad)
    }))
}

//----------------------
// a slice of slices is a really dumb representation for a matrix
// but we do not require performance where this is used, so whatever

fn dot_vec_mat_dumb(vec: &[f64], mat: &[&[f64]]) -> Vec<f64>
{
    assert_eq!(mat.len(), vec.len());
    assert_ne!(mat.len(), 0, "cannot determine width of matrix with no rows");
    let init = v(vec![0.0; mat[0].len()]);
    let V(out) = izip!(mat, vec)
        .fold(init, |acc, (&row, &alpha)| acc + alpha * v(row));
    out
}

fn dot_mat_vec_dumb(mat: &[&[f64]], vec: &[f64]) -> Vec<f64>
{ mat.iter().map(|row| vdot(vec, row)).collect() }

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
            Some(SAVE_BANDS_DIR.as_ref()),
            &[Q_GAMMA, Q_K],
        )?;
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

fn tup2<T:Copy>(arr: [T; 2]) -> (T, T) { (arr[0], arr[1]) }
fn tup3<T:Copy>(arr: [T; 3]) -> (T, T, T) { (arr[0], arr[1], arr[2]) }
