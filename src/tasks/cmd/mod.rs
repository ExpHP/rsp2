// HERE BE DRAGONS

pub(crate) mod integrate_2d;

use self::lammps::{Lammps, LammpsBuilder};
mod lammps;

use self::trial::TrialDir;
pub(crate) mod trial;

use ::errors::{Error, ErrorKind, Result, ok};
use ::rsp2_tasks_config::{self as cfg, Settings, NormalizationMode, SupercellSpec};
use ::traits::{AsPath};
use ::phonopy::{DirWithBands, DirWithDisps, DirWithForces};

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
use ::rsp2_structure::{Part, Partition};
use ::rsp2_structure_gen::Assemble;
use ::phonopy::Builder as PhonopyBuilder;

use ::rsp2_fs_util::{rm_rf};

use ::std::io::{Write};
use ::std::path::{Path, PathBuf};

use ::itertools::Itertools;

const SAVE_BANDS_DIR: &'static str = "gamma-bands";

// cli args aren't in Settings, so they're just here.
pub struct CliArgs {
    pub save_bands: bool,
}

impl TrialDir {
    pub(crate) fn run_relax_with_eigenvectors(
        self,
        settings: &Settings,
        input: &PathFile,
        cli: CliArgs,
    ) -> Result<()>
    {ok({
        use ::rsp2_structure_io::poscar;

        let lmp = LammpsBuilder::new(&settings.threading, &settings.potential.kind);

        // let mut original = poscar::load(open(input)?)?;
        // original.scale_vecs(&settings.hack_scale);

        // For the time being this now only supports layers.yaml format.
        let (original, layer_sc_mats) = {
            use ::rsp2_structure_gen::load_layers_yaml;
            use ::rsp2_structure_gen::layer_sc_info_from_layers_yaml;

            let builder = load_layers_yaml(input.read()?)?;
            let builder = optimize_layer_parameters(&settings.scale_ranges, &lmp, builder)?;
            let structure = carbon(&builder.assemble());

            let layer_sc_mats =
                layer_sc_info_from_layers_yaml(input.read()?)?.into_iter()
                .map(|(matrix, periods, _)| ::math::bands::ScMatrix::new(&matrix, &periods))
                .collect_vec();

            (structure, layer_sc_mats)
        };

        poscar::dump(self.create_file("initial.vasp")?, "", &original)?;
        let phonopy = phonopy_builder_from_settings(&settings.phonons, &original);
        let phonopy = phonopy.use_sparse_sets(settings.tweaks.sparse_sets);

        // (we can expect that the layers assignments will not change, so we do this early...)
        trace!("Finding layers");
        let layers = {
            let layers =
                ::rsp2_structure::find_layers(&original, &V3([0, 0, 1]), 0.25)?
                    .per_unit_cell().expect("Structure is not layered?");

            if let Some(expected) = settings.layers {
                assert_eq!(expected, layers.len() as u32);
            }
            layers.by_atom()
        };

        // (just dump bond info for now)
        if let Some(bond_radius) = settings.bond_radius {
            trace!("Computing bonds, just for kicks.");
            let bonds = Bonds::from_brute_force_very_dumb(&original, bond_radius)?;
            ::serde_yaml::to_writer(self.create_file("bonds.yaml")?, &bonds)?;
        };

        let mut from_structure = original.clone();
        let mut iteration = 1;
        // HACK to stop one iteration AFTER all non-acoustics are positive
        let mut all_ok_count = 0;
        let (structure, einfos) = loop { // NOTE: we use break with value

            let structure = do_relax(&lmp, &settings.cg, &settings.potential, from_structure)?;
            let bands_dir = do_diagonalize(
                &lmp, &settings.threading, &phonopy, &structure,
                match cli.save_bands {
                    true => Some(SAVE_BANDS_DIR.as_ref()),
                    false => None,
                },
                &[Q_GAMMA, Q_K],
            )?;

            let (evals, evecs) = read_eigensystem(&bands_dir, &Q_GAMMA)?;

            trace!("============================");
            trace!("Finished relaxation # {}", iteration);

            {
                let file = self.create_file(format!("structure-{:02}.1.vasp", iteration))?;
                trace!("Writing '{}'", file.path().nice()?);
                poscar::dump(
                    file,
                    &format!("Structure after CG round {}", iteration),
                    &structure)?;
            }

            trace!("Computing eigensystem info");
            let einfos = get_gamma_eigensystem_info(
                &evals, &evecs, &layers[..], &structure, Some(&layer_sc_mats),
            );
            {
                let file = self.create_file(format!("eigenvalues.{:02}", iteration))?;
                write_eigen_info_for_machines(&einfos, file)?;
                write_eigen_info_for_humans(&einfos, &mut |s| ok(info!("{}", s)))?;
            }

            let mut structure = structure;
            let bad_evs: Vec<_> = izip!(1.., &einfos, &evecs.0)
                .filter(|&(_, ref info, _)| info.frequency < 0.0 && !info.is_acoustic())
                .map(|(i, info, evec)| {
                    let name = format!("band {} ({})", i, info.frequency);
                    (name, evec.as_real_checked())
                }).collect();

            if !bad_evs.is_empty() {
                trace!("Chasing eigenvectors...");
                structure = do_eigenvector_chase(
                    &lmp,
                    &settings.ev_chase,
                    &settings.potential,
                    structure,
                    &bad_evs[..],
                )?;
            }

            {
                let file = self.create_file(format!("./structure-{:02}.2.vasp", iteration))?;
                trace!("Writing '{}'", file.path().nice()?);
                poscar::dump(
                    file,
                    &format!("Structure after eigenmode-chasing round {}", iteration),
                    &structure)?;
            }

            warn_on_improvable_lattice_params(&lmp, &structure)?;

            if bad_evs.is_empty() {
                all_ok_count += 1;
                if all_ok_count >= settings.ev_loop.min_positive_iter {
                    break (structure, einfos /*, bands_dir */);
                }
            }

            // -----------------------------
            // mutate loop variables

            iteration += 1; // REMINDER: not a for loop due to 'break value;'

            // Possibly fail after too many iterations
            // (we don't want to continue with negative non-acoustics)
            if iteration > settings.ev_loop.max_iter {
                if settings.ev_loop.fail {
                    error!("Too many relaxation steps!");
                    bail!("Too many relaxation steps!");
                } else {
                    warn!("Too many relaxation steps!");
                    break (structure, einfos /*, bands_dir */);
                }
            }

            from_structure = structure;
        }; // (structure, einfos, final_bands_dir)


        poscar::dump(self.create_file("final_vasp")?, "", &structure)?;

        write_eigen_info_for_machines(&einfos, self.create_file("eigenvalues.final")?)?;

        // let (k_evals, k_evecs) = read_eigensystem(&bands_dir, &Q_K)?;
        // let kinfos = get_k_eigensystem_info(
        //     &k_evals, &k_evecs, &layers[..], &structure, Some(&layer_sc_mats),
        // );

        // write_summary_file(settings, &lmp, &einfos, &kinfos)?;
        self.write_summary_file(settings, &lmp, &einfos)?;
    })}
}

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

// FIXME write_gamma_info
fn write_eigen_info_for_humans(
    einfos: &[EigenmodeInfo],
    writeln: &mut FnMut(&::std::fmt::Display) -> Result<()>,
) -> Result<()>
{ok({
    use ::ansi_term::Colour::{Red, Cyan, Yellow, Black};
    use ::ui::color::{ColorByRange, DisplayProb};

    let color_range = ColorByRange::new(vec![
        (0.999, Cyan.bold()),
        (0.9, Cyan.normal()),
        (0.1, Yellow.normal()),
        (1e-4, Red.bold()),
        (1e-10, Red.normal()),
    ], Black.normal());
    let dp = |x: f64| color_range.paint_as(&x, DisplayProb(x));
    let pol = |x: f64| color_range.paint(x);

    let nlayer = einfos.iter().next().unwrap().layer_gamma_probs.as_ref().map(|v| v.len()).unwrap_or(0);
    writeln(&format_args!("{:27}  {:^7}  {:^7}  {} [{:^4}, {:^4}, {:^4}]",
        "# Frequency (cm^-1)", "Acoustc", "Layer",
        (1..nlayer+1).map(|i| format!("GammaL{}", i)).join(" "),
        "X", "Y", "Z"))?;
    for item in einfos.iter() {
        let eval = item.frequency;
        let acou = dp(item.acousticness);
        let layer = dp(item.layer_acousticness);
        let (x, y, z) = tup3(item.polarization);
        let gammas = item.layer_gamma_probs.as_ref().unwrap_or(&vec![]).iter().map(|&p| dp(p)).join(" ");

        writeln(&format_args!("{:27}  {}  {}  {} [{:4.2}, {:4.2}, {:4.2}]",
            eval, acou, layer, gammas, pol(x), pol(y), pol(z)))?;
    }
})}

fn write_eigen_info_for_machines<W: Write>(
    einfos: &[EigenmodeInfo],
    mut file: W,
) -> Result<()>
{ok({
    writeln!(file, "{:27}  {:4}  {:4}  {:^4} {:^4} {:^4}",
        "# Frequency (cm^-1)", "Acou", "Layr", "X", "Y", "Z")?;
    for item in einfos.iter() {
        // don't use DisplayProb, keep things parsable
        let eval = item.frequency;
        let acou = item.acousticness;
        let layer = item.layer_acousticness;
        let (x, y, z) = tup3(item.polarization);
        writeln!(file, "{:27}  {:4.2}  {:4.2}  {:4.2} {:4.2} {:4.2}",
            eval, acou, layer, x, y, z)?;
    }
})}

impl TrialDir {
    fn write_summary_file(
        &self,
        settings: &Settings,
        lmp: &LammpsBuilder,
        gamma_infos: &[EigenmodeInfo],
    ) -> Result<()>
    {Ok({
        use self::summary::{Modes, GammaModes, Summary, EnergyPerAtom};
        use ::rsp2_structure_io::poscar;

        fn filtered_freqs<Pred>(modes: &[EigenmodeInfo], pred: Pred) -> Vec<f64>
        where Pred: Fn(&EigenmodeInfo) -> bool
        { modes.iter().filter(|&mode| pred(mode)).map(EigenmodeInfo::frequency).collect() }

        let acoustic = filtered_freqs(gamma_infos, EigenmodeInfo::is_acoustic);
        let shear = filtered_freqs(gamma_infos, EigenmodeInfo::is_shear);
        let layer_breathing = filtered_freqs(gamma_infos, EigenmodeInfo::is_layer_breathing);

        // FIXME: this mess is due to an awkward choice of representation;
        //        we have an array of structures where either all of the
        //        'layer_gamma_props' members are present, or none of them are.
        let layer_gammas = gamma_infos.iter().next().unwrap().layer_gamma_probs.as_ref().map(|_| {
            gamma_infos.iter().enumerate()
                .filter(|&(_,x)| x.layer_gamma_probs.as_ref().unwrap()[0] > settings.layer_gamma_threshold)
                .map(|(i,x)| (i, x.frequency()))
                .collect()
        });
        let gamma_modes = GammaModes { acoustic, shear, layer_breathing, layer_gammas };

        let modes = Modes {
            gamma: gamma_modes,
        };

        let energy_per_atom = {
            let f = |structure: ElementStructure| ok({
                let na = structure.num_atoms() as f64;
                lmp.build(structure)?.compute_value()? / na
            });
            let f_path = |s: &AsPath| ok(f(poscar::load(self.open(s)?)?)?);

            let initial = f_path(&"initial.vasp")?;
            let final_ = f_path(&"final.vasp")?;
            let before_ev_chasing = f_path(&"structure-01.1.vasp")?;
            EnergyPerAtom { initial, final_, before_ev_chasing }
        };

        let summary = Summary { modes, energy_per_atom };

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


pub fn get_gamma_eigensystem_info<L: Ord + Clone, M>(
    evals: &[f64],
    evecs: &Basis3,
    layers: &[L],
    // (M gets tossed out, but we don't take Structure<L> because it could
    //  unintentionally accept Element as our layer type)
    structure: &Structure<M>,
    // unfolding is only done when this is provided
    layer_supercell_matrices: Option<&[::math::bands::ScMatrix]>,
) -> Vec<EigenmodeInfo>
{
    use ::math::bands::GammaUnfolder;

    for evec in &evecs.0 {
        assert!(
            evec.imag.flat().iter().all(|&x| x == 0.0),
            "clearly not a gamma eigenket!");
    }

    let part = Part::from_ord_keys(layers);
    let layer_structures = structure
            .map_metadata_to(|_| ())
            .into_unlabeled_partitions(&part)
            .collect_vec();

    // precomputed data applicable to all kets
    let layer_gamma_unfolders: Option<Vec<GammaUnfolder>> =
        layer_supercell_matrices.map(|layer_supercell_matrices| {
            izip!(&layer_structures, layer_supercell_matrices)
                .map(|(layer_structure, layer_sc_mat)| {
                    GammaUnfolder::from_config(
                        &from_json!({
                            "fbz": "reciprocal-cell",
                            "sampling": { "plain": [4, 4, 1] },
                        }),
                        &layer_structure,
                        layer_sc_mat,
                    )
                }).collect()
        });

    // now do each ket
    let mut out = vec![];
    for (&frequency, evec) in izip!(evals, &evecs.0) {
        // split the evs up by layer (without renormalizing)
        let layer_evecs: Vec<_> = evec.clone().into_unlabeled_partitions(&part).collect();

        let acousticness = evec.acousticness();
        let layer_acousticness = layer_evecs.iter().map(|ev| ev.acousticness()).sum();
        let polarization = evec.polarization();

        let layer_gamma_probs = layer_gamma_unfolders.as_ref().map(|unfolders| {
            izip!(&unfolders[..], &layer_evecs)
                .map(|(unfolder, ket)| {
                    let probs = unfolder.unfold_phonon(ket.to_ket().as_ref());
                    izip!(unfolder.q_indices(), probs)
                        .find(|&(idx, _)| idx == &[0, 0, 0])
                        .unwrap().1
                }).collect()
        });

        out.push(EigenmodeInfo {
            frequency, acousticness, layer_acousticness, polarization, layer_gamma_probs,
        })
    }
    out
}

// HACK:
// These are only valid for a hexagonal system represented
// with the [[a, 0], [-a/2, a sqrt(3)/2]] lattice convention
#[allow(unused)] const Q_GAMMA: V3 = V3([0.0, 0.0, 0.0]);
#[allow(unused)] const Q_K: V3 = V3([1f64/3.0, 1.0/3.0, 0.0]);
#[allow(unused)] const Q_K_PRIME: V3 = V3([2.0 / 3f64, 2.0 / 3f64, 0.0]);

/// Properties for characterizing an eigenvector in layered graphene.
pub struct EigenmodeInfo {
    pub frequency: f64,
    pub acousticness: f64,
    pub layer_acousticness: f64,
    pub polarization: [f64; 3], // not a vector
    pub layer_gamma_probs: Option<Vec<f64>>,
}

impl EigenmodeInfo {
    pub fn frequency(&self) -> f64
    { self.frequency }

    pub fn is_acoustic(&self) -> bool
    { self.acousticness > 0.95 }

    pub fn is_xy_polarized(&self) -> bool
    { self.polarization[0] + self.polarization[1] > 0.9 }

    pub fn is_z_polarized(&self) -> bool
    { self.polarization[2] > 0.9 }

    pub fn is_layer_acoustic(&self) -> bool
    { self.layer_acousticness > 0.95 && !self.is_acoustic() }

    pub fn is_shear(&self) -> bool
    { self.is_layer_acoustic() && self.is_xy_polarized() }

    pub fn is_layer_breathing(&self) -> bool
    { self.is_layer_acoustic() && self.is_z_polarized() }
}

mod summary {
    #[derive(Serialize, Deserialize)]
    #[serde(rename_all="kebab-case")]
    pub struct Summary {
        pub energy_per_atom: EnergyPerAtom,
        pub modes: Modes,
    }

    #[derive(Serialize, Deserialize)]
    #[serde(rename_all="kebab-case")]
    pub struct EnergyPerAtom {
        pub initial: f64,
        pub before_ev_chasing: f64,
        #[serde(rename = "final")]
        pub final_: f64,
    }

    #[derive(Serialize, Deserialize)]
    #[serde(rename_all="kebab-case")]
    pub struct Modes {
        pub gamma: GammaModes,
        // Even though there's only one member,
        // this struct remains for legacy purposes
        // (to preserve the shape of the output YAML)
    }

    #[derive(Serialize, Deserialize)]
    #[serde(rename_all="kebab-case")]
    pub struct GammaModes {
        pub acoustic: Vec<f64>,
        pub shear: Vec<f64>,
        pub layer_breathing: Vec<f64>,
        pub layer_gammas: Option<Vec<(usize, f64)>>,
    }

    #[derive(Serialize, Deserialize)]
    #[serde(rename_all="kebab-case")]
    pub struct KModes {
        pub layer_ks: Option<Vec<(usize, f64)>>,
    }
}

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
                    trace!("Finding layers");
                    let (layers, nlayer) = {
                        let layers =
                            ::rsp2_structure::find_layers(&structure, &V3([0, 0, 1]), 0.25)?
                                .per_unit_cell().expect("Structure is not layered?");
                        (layers.by_atom(), layers.len())
                    };
                    assert!(nlayer >= 2);

                    trace!("Computing eigensystem info");
                    let einfos = get_gamma_eigensystem_info(&evals, &evecs, &layers, &structure, None);

                    write_eigen_info_for_humans(&einfos, &mut |s| ok(info!("{}", s)))?;

                    let mut iter = einfos
                        .iter().enumerate()
                        .filter(|&(_, info)| info.is_shear())
                        .map(|(i, _)| i);

                    match (iter.next(), iter.next()) {
                        (Some(a), Some(b)) => (a, b),
                        _ => panic!("Expected at least two shear modes"),
                    }
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

extension_trait! {
    PathBufExt for PathBuf {
        // make a path "nice" for display
        fn nice(&self) -> Result<String> {
            let cwd = PathDir::current_dir()?;
            let absolute = cwd.join(self);
            match absolute.as_path().strip_prefix(&cwd) {
                Ok(path) => Ok(format!("{}", path.display())),
                Err(_) => Ok(format!("{}", self.display())),
            }
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
