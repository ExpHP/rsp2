// HERE BE DRAGONS

const THZ_TO_WAVENUMBER: f64 = 33.35641;

use ::errors::{Result, ResultExt, ok};
use ::config::Settings;
use ::util::push_dir;
use ::logging::setup_global_logger;

use ::rsp2_structure::consts::CARBON;
use ::rsp2_slice_math::{v, V, vdot};

use ::slice_of_array::prelude::*;
use ::rsp2_structure::supercell::{self, SupercellToken};
use ::rsp2_structure::{Coords, CoordStructure, Structure, ElementStructure};
use ::rsp2_structure_gen::Assemble;
use ::rsp2_lammps_wrap::Lammps;
use ::rsp2_lammps_wrap::Builder as LammpsBuilder;
use ::rsp2_phonopy_io::Builder as PhonopyBuilder;

use ::std::io::{BufReader};
use ::std::path::{Path, PathBuf};
use ::std::hash::Hash;
use ::std::fs::{self, File};

use ::itertools::Itertools;

pub fn run_relax_with_eigenvectors(
    settings: &Settings,
    input: &AsRef<Path>,
    outdir: &AsRef<Path>,
    // cli args aren't in Settings, so they're just here.
    save_forces: bool,
) -> Result<()>
{ok({
    use ::std::io::prelude::*;
    use ::rsp2_structure_io::poscar;

    let lmp = make_lammps_builder(&settings.threading);
    let input = canonicalize(input.as_ref())?;

    create_dir(&outdir)?;
    {
        // dumb/lazy solution to ensuring all output files go in the dir
        let cwd_guard = push_dir(outdir)?;

        setup_global_logger(Some(&"rsp2.log"))?;

        // let mut original = poscar::load(open(input)?)?;
        // original.scale_vecs(&settings.hack_scale);
        let (original, layer_sc_mats) = {
            use ::rsp2_structure_gen::load_layers_yaml;
            use ::rsp2_structure_gen::layer_sc_info_from_layers_yaml;

            let builder = load_layers_yaml(open(&input)?)?;
            let builder = optimize_layer_parameters(&settings.scale_ranges, &lmp, builder)?;
            let structure = carbon(&builder.assemble());

            let layer_sc_info = layer_sc_info_from_layers_yaml(open(&input)?)?;
            let layer_sc_mats = layer_sc_info
                .into_iter().map(|(matrix, periods, _)| ::bands::ScMatrix::new(&matrix, &periods))
                .collect_vec();

            (structure, layer_sc_mats)
        };

        poscar::dump(create("./initial.vasp")?, "", &original)?;
        let phonopy = phonopy_builder_from_settings(&settings.phonons, &original);

        let mut from_structure = original.clone();
        let mut iteration = 1;
        // HACK to stop one iteration AFTER all non-acoustics are positive
        let mut all_ok_count = 0;
        let (structure, einfos, _evecs) = loop { // NOTE: we use break with value

            let structure = do_relax(&lmp, &settings.cg, &settings.potential, from_structure)?;
            let (evals, evecs) = do_diagonalize(&lmp, &phonopy, &structure)?;

            trace!("============================");
            trace!("Finished relaxation # {}", iteration);

            trace!("Finding layers");
            let (layers, nlayer) = ::rsp2_structure::assign_layers(&structure, &[0, 0, 1], 0.25)?;
            if let Some(expected) = settings.layers {
                assert_eq!(nlayer, expected);
            }

            {
                let fname = format!("./structure-{:02}.1.vasp", iteration);
                trace!("Writing '{}'", &fname);
                poscar::dump(
                    create(fname)?,
                    &format!("Structure after CG round {}", iteration),
                    &structure)?;
            }

            trace!("Computing eigensystem info");
            let einfos = get_eigensystem_info(
                &evals, &evecs, &layers[..], Some(&structure), Some(&layer_sc_mats),
            );
            {
                let mut file = create(format!("eigenvalues.{:02}", iteration))?;
                write_eigen_info(&einfos, &mut |s| writeln!(file, "{}", s).map_err(Into::into))?;
            }
            write_eigen_info(&einfos, &mut |s| ok(info!("{}", s)))?;

            let mut structure = structure;
            let bad_evs: Vec<_> = izip!(1.., &einfos, &evecs)
                .filter(|&(_, ref info, _)| info.frequency < 0.0 && !info.is_acoustic())
                .map(|(i, info, evec)| {
                    let name = format!("band {} ({})", i, info.frequency);
                    (name, &evec[..])
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
                let fname = format!("./structure-{:02}.2.vasp", iteration);
                trace!("Writing '{}'", &fname);
                poscar::dump(
                    create(fname)?,
                    &format!("Structure after eigenmode-chasing round {}", iteration),
                    &structure)?;
            }

            {
                const SCALE_AMT: f64 = 1e-6;
                let mut lmp = lmp.initialize_carbon(uncarbon(&structure))?;
                let center_value = lmp.compute_value()?;

                let shrink_value = {
                    let mut structure = structure.clone();
                    structure.scale_vecs(&[1.0 - SCALE_AMT, 1.0 - SCALE_AMT, 1.0]);
                    lmp.set_structure(uncarbon(&structure))?;
                    lmp.compute_value()?
                };

                let enlarge_value = {
                    let mut structure = structure.clone();
                    structure.scale_vecs(&[1.0 + SCALE_AMT, 1.0 + SCALE_AMT, 1.0]);
                    lmp.set_structure(uncarbon(&structure))?;
                    lmp.compute_value()?
                };

                if shrink_value.min(enlarge_value) < center_value {
                    warn!("Better value found at nearby lattice parameter: {:?}",
                        (shrink_value, center_value, enlarge_value))
                }
            }

            if bad_evs.is_empty() {
                all_ok_count += 1;
                if all_ok_count >= 3 {
                    // (clone is HACK because bad_evs borrows from evecs)
                    break (structure, einfos, evecs.clone());
                }
            }

            from_structure = structure;
            iteration += 1; // NOTE: not a for loop due to 'break value;'
        }; // (structure, einfos, evecs)

        {
            let mut f = create("eigenvalues.final")?;
            writeln!(f, "{:27}  {:4}  {:4}  {:^4} {:^4} {:^4}",
                "# Frequency (cm^-1)", "Acou", "Layr", "X", "Y", "Z")?;
            for item in einfos.iter() {
                // don't use DisplayProb, keep things readible
                let eval = item.frequency;
                let acou = item.acousticness;
                let layer = item.layer_acousticness;
                let (x, y, z) = tup3(item.polarization);
                writeln!(f, "{:27}  {:4.2}  {:4.2}  {:4.2} {:4.2} {:4.2}",
                    eval, acou, layer, x, y, z)?;
            }
        }

        poscar::dump(create("./final.vasp")?, "", &structure)?;

        if save_forces {
            let disp_dir = phonopy.displacements(&structure)?;
            let force_sets = do_force_sets_at_disps(&lmp, &disp_dir)?;
            let force_dir = disp_dir.make_force_dir(&force_sets)?;
            copy(force_dir.path().join("FORCE_SETS"), "FORCE_SETS")?;
        }

        { // write summary file
            use self::summary::{Modes, Summary, EnergyPerAtom};
            use self::eigen_info::Item;

            let acoustic = einfos.iter().filter(|x| x.is_acoustic()).map(Item::frequency).collect();
            let shear = einfos.iter().filter(|x| x.is_shear()).map(Item::frequency).collect();
            let layer_breathing = einfos.iter().filter(|x| x.is_layer_breathing()).map(Item::frequency).collect();
            let layer_gammas = einfos.iter().next().unwrap().layer_gamma_probs.as_ref().map(|_| {
                einfos.iter().enumerate()
                    .filter(|&(_,x)| x.layer_gamma_probs.as_ref().unwrap()[0] > settings.layer_gamma_threshold)
                    .map(|(i,x)| (i, x.frequency()))
                    .collect()
            });
            let modes = Modes { acoustic, shear, layer_breathing, layer_gammas };

            let energy_per_atom = {
                let f = |structure: ElementStructure| ok({
                    let na = structure.num_atoms() as f64;
                    lmp.initialize_carbon(uncarbon(&structure))?.compute_value()? / na
                });
                let f_path = |s: &AsRef<Path>| ok(f(poscar::load(open(s)?)?)?);

                let initial = f(original.clone())?;
                let final_ = f(structure)?;
                let before_ev_chasing = f_path(&"structure-01.1.vasp")?;
                EnergyPerAtom { initial, final_, before_ev_chasing }
            };

            let summary = Summary { modes, energy_per_atom };

            ::serde_yaml::to_writer(create("summary.yaml")?, &summary)?;
        }

        cwd_guard.pop()?;
    }
})}

fn do_relax(
    lmp: &LammpsBuilder,
    cg_settings: &::config::Acgsd,
    potential_settings: &::config::Potential,
    structure: ElementStructure,
) -> Result<ElementStructure>
{ok({
    let sc_dims = tup3(potential_settings.supercell.dim_for_unitcell(structure.lattice()));
    let (supercell, sc_token) = supercell::diagonal(sc_dims, structure);

    let mut lmp = lmp.clone().threaded(true).initialize_carbon(uncarbon(&supercell))?;
    let relaxed_flat = ::rsp2_minimize::acgsd(
        cg_settings,
        supercell.to_carts().flat(),
        &mut *lammps_flat_diff_fn(&mut lmp),
    ).unwrap().position;

    let supercell = supercell.with_coords(Coords::Carts(relaxed_flat.nest().to_vec()));
    multi_threshold_deconstruct(sc_token, 1e-10, 1e-3, supercell)?
})}

fn do_diagonalize(
    lmp: &LammpsBuilder,
    phonopy: &PhonopyBuilder,
    structure: &ElementStructure,
) -> Result<(Vec<f64>, Vec<Vec<[f64; 3]>>)>
{ok({
    let disp_dir = phonopy.displacements(&structure)?;
    let force_sets = do_force_sets_at_disps(&lmp, &disp_dir)?;

    match disp_dir
        .make_force_dir(&force_sets)?
        .build_bands()
        .eigenvectors(true)
        .compute(&[[0.0; 3]])?
        .gamma_eigensystem()?
    {
        Some((eval, Some(evec))) => {
            let V(eval) = THZ_TO_WAVENUMBER * v(eval);
            (eval, evec)
        },
        _ => unreachable!(),
    }
})}

fn do_eigenvector_chase(
    lmp: &LammpsBuilder,
    chase_settings: &::config::EigenvectorChase,
    potential_settings: &::config::Potential,
    mut structure: ElementStructure,
    bad_evecs: &[(String, &[[f64; 3]])],
) -> Result<ElementStructure>
{ok({
    match *chase_settings {
        ::config::EigenvectorChase::OneByOne => {
            for &(ref name, evec) in bad_evecs {
                let (alpha, new_structure) = do_minimize_along_evec(lmp, potential_settings, structure, &evec[..])?;
                info!("Optimized along {}, a = {:e}", name, alpha);

                structure = new_structure;
            }
            structure
        },
        ::config::EigenvectorChase::Acgsd(ref cg_settings) => {
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
    cg_settings: &::config::Acgsd,
    potential_settings: &::config::Potential,
    structure: ElementStructure,
    evecs: I,
) -> Result<ElementStructure>
where
    V: AsRef<[[f64; 3]]>,
    I: IntoIterator<Item=V>,
{ok({
    let evecs: Vec<_> = evecs.into_iter().collect();
    let refs: Vec<_> = evecs.iter().map(|x| x.as_ref()).collect();
    _do_cg_along_evecs(lmp, cg_settings, potential_settings, structure, &refs)?
})}

fn _do_cg_along_evecs(
    lmp: &LammpsBuilder,
    cg_settings: &::config::Acgsd,
    potential_settings: &::config::Potential,
    structure: ElementStructure,
    evecs: &[&[[f64; 3]]],
) -> Result<ElementStructure>
{ok({
    let sc_dims = tup3(potential_settings.supercell.dim_for_unitcell(structure.lattice()));
    let (supercell, sc_token) = supercell::diagonal(sc_dims, structure);
    let evecs: Vec<_> = evecs.iter().map(|ev| sc_token.replicate(ev)).collect();

    let flat_evecs: Vec<_> = evecs.iter().map(|ev| ev.flat()).collect();
    let init_pos = supercell.to_carts();

    let mut lmp = lmp.clone().threaded(true).initialize_carbon(uncarbon(&supercell))?;
    let relaxed_flat = ::rsp2_minimize::acgsd(
        cg_settings,
        &vec![0.0; evecs.len()],
        &mut *lammps_constrained_diff_fn(
            &mut lmp,
            init_pos.flat(),
            &flat_evecs,
            ),
    ).unwrap().position;

    let supercell = supercell.with_coords(Coords::Carts(relaxed_flat.nest().to_vec()));
    multi_threshold_deconstruct(sc_token, 1e-10, 1e-3, supercell)?
})}

fn do_minimize_along_evec(
    lmp: &LammpsBuilder,
    settings: &::config::Potential,
    structure: ElementStructure,
    evec: &[[f64; 3]],
) -> Result<(f64, ElementStructure)>
{ok({
    let sc_dims = tup3(settings.supercell.dim_for_unitcell(structure.lattice()));
    let (structure, sc_token) = supercell::diagonal(sc_dims, structure);
    let evec = sc_token.replicate(evec);
    let mut lmp = lmp.clone().threaded(true).initialize_carbon(uncarbon(&structure))?;

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

fn write_eigen_info(
    einfos: &EigenInfo,
    writeln: &mut FnMut(&::std::fmt::Display) -> Result<()>,
) -> Result<()>
{ok({
    use ::ansi_term::Colour::{Red, Cyan, Yellow, Black};
    use ::color::{ColorByRange, DisplayProb};

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

fn phonopy_builder_from_settings<M>(
    settings: &::config::Phonons,
    // the structure is needed to resolve the correct supercell size
    structure: &Structure<M>,
) -> PhonopyBuilder
{
    PhonopyBuilder::new()
        .symmetry_tolerance(settings.symmetry_tolerance)
        .conf("DISPLACEMENT_DISTANCE", format!("{:e}", settings.displacement_distance))
        .supercell_dim(settings.supercell.dim_for_unitcell(structure.lattice()))
        .conf("DIAG", ".FALSE.")
}

fn do_force_sets_at_disps<P: AsRef<Path>>(
    lmp: &LammpsBuilder,
    disp_dir: &::rsp2_phonopy_io::DirWithDisps<P>,
) -> Result<Vec<Vec<[f64; 3]>>>
{ok({
    use ::std::io::prelude::*;

    trace!("Computing forces at displacements");

    let mut i = 0;
    let force_sets = disp_dir.displaced_structures()
        .map(|structure| ok({
            i += 1;
            eprint!("\rdisp {} of {}", i, disp_dir.displacements().len());
            ::std::io::stderr().flush().unwrap();

            lmp.initialize_carbon(structure)?.compute_force()?
        })).collect::<Result<Vec<_>>>()?;
    eprintln!();
    force_sets
})}

//-----------------------------------

pub fn optimize_layer_parameters(
    settings: &::config::ScaleRanges,
    lmp: &LammpsBuilder,
    mut builder: Assemble,
) -> Result<Assemble>
{ok({
    pub use ::rsp2_minimize::exact_ls::{Value, Golden};
    use ::config::{ScaleRanges, ScaleRange};
    use ::std::cell::RefCell;

    // full destructure so we don't miss anything
    let ScaleRanges {
        parameter: ScaleRange { guess: parameter_guess, range: parameter_range },
        layer_sep: ScaleRange { guess: layer_sep_guess, range: layer_sep_range },
    } = *settings;

    let n_seps = builder.layer_seps().len();

    // abuse RefCell for some DRYness
    let builder = RefCell::new(builder);
    {
        let builder = &builder;

        let optimizables = {
            let mut optimizables: Vec<(_, Box<Fn(f64)>)> = vec![];
            optimizables.push((
                (format!("lattice parameter"), parameter_guess, parameter_range),
                Box::new(|s| {
                    builder.borrow_mut().scale = s;
                }),
            ));

            for i in 0..n_seps {
                optimizables.push((
                    (format!("layer sep {}", i), layer_sep_guess, layer_sep_range),
                    Box::new(move |s| {
                        builder.borrow_mut().layer_seps()[i] = s;
                    }),
                ));
            }
            optimizables
        };

        // try to start with reasonable defaults ('guess' in config)
        for &((_, guess, _), ref setter) in &optimizables {
            if let Some(guess) = guess {
                setter(guess);
            }
        }

        //let mut lmp = lmp.initialize_carbon(builder.borrow().assemble())?;
        let get_value = || ok({
            // use ::std::hash::{Hash, Hasher};
            lmp.initialize_carbon(builder.borrow().assemble())?.compute_value()?
        });

        // optimize them one-by-one.
        for &((ref name, _, range), ref setter) in &optimizables {
            trace!("Optimizing {}", name);

            let best = Golden::new()
                .stop_condition(&from_json!({"interval-size": 1e-7}))
                .run(range, |a| {
                    setter(a);
                    get_value().map(Value)
                })??; // ?!??!!!?

            info!("Optimized {}: {} (from {:?})", name, best, range);
            setter(best);
        }
    }

    builder.into_inner()
})}


//-----------------------------------

pub type EigenInfo = Vec<eigen_info::Item>;

pub fn get_eigensystem_info<L: Eq + Clone + Hash>(
    evals: &[f64],
    evecs: &[Vec<[f64; 3]>],
    layers: &[L],
    structure: Option<&ElementStructure>,
    layer_supercell_matrices: Option<&[::bands::ScMatrix]>,
) -> EigenInfo
{
    use ::rsp2_eigenvector_classify::{keyed_acoustic_basis, polarization};

    let layers: Vec<_> = layers.iter().cloned().map(Some).collect();
    let layer_acoustics = keyed_acoustic_basis(&layers[..], &[1,1,1]);
    let acoustics = keyed_acoustic_basis(&vec![Some(()); evecs[0].len()], &[1,1,1]);

    let mut out = vec![];
    for (&eval, evec) in izip!(evals, evecs) {

        let gamma_probs = match (structure, layer_supercell_matrices) {
            (Some(structure), Some(layer_supercell_matrices)) => {Some({
                use ::rsp2_kets::{Rect, Ket};
                let ket: Ket = evec.flat().iter().cloned().map(Rect::from).collect();

                layer_supercell_matrices.iter().map(|sc_mat| {
                    ::bands::unfold_phonon(
                        &from_json!({
                            "fbz": "reciprocal-cell",
                            "sampling": { "plain": [4, 4, 1] },
                        }),
                        &structure.map_metadata_to(|_| ()),
                        &[0.0, 0.0, 0.0], // eigenvector q
                        ket.as_ref(),
                        sc_mat,
                    ).iter()
                        .find(|&&(idx, _)| idx == [0, 0, 0])
                        .unwrap().1
                }).collect()
            })},
            _ => None,
        };

        out.push(eigen_info::Item {
            frequency: eval,
            acousticness: acoustics.probability(&evec),
            layer_acousticness: layer_acoustics.probability(&evec),
            polarization: polarization(&evec[..]),
            layer_gamma_probs: gamma_probs,
        })
    }
    out
}

pub mod eigen_info {
    pub struct Item {
        pub frequency: f64,
        pub acousticness: f64,
        pub layer_acousticness: f64,
        pub polarization: [f64; 3],
        pub layer_gamma_probs: Option<Vec<f64>>,
    }

    impl Item {
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
        pub acoustic: Vec<f64>,
        pub shear: Vec<f64>,
        pub layer_breathing: Vec<f64>,
        pub layer_gammas: Option<Vec<(usize, f64)>>,
    }
}

// HACK: These adapters are temporary to help the existing code
//       (written only with carbon in mind) adapt to ElementStructure.
#[allow(unused)]
fn carbon(structure: &CoordStructure) -> ElementStructure
{
    // I want it PAINTED BLACK!
    structure.map_metadata_to(|_| CARBON)
}

fn uncarbon(structure: &ElementStructure) -> CoordStructure
{
    assert!(structure.metadata().iter().all(|&e| e == CARBON),
        "if you want to use this code on non-carbon stuff, you better \
        do something about all the carbon-specific code.  Look for calls \
        to `carbon()`");
    structure.map_metadata_to(|_| ())
}

fn make_lammps_builder(threading: &::config::Threading) -> LammpsBuilder
{
    let mut lmp = LammpsBuilder::new();
    lmp.append_log("lammps.log");
    lmp.threaded(*threading == ::config::Threading::Lammps);
    lmp
}

fn lammps_flat_diff_fn<'a>(lmp: &'a mut Lammps)
-> Box<FnMut(&[f64]) -> Result<(f64, Vec<f64>)> + 'a>
{
    Box::new(move |pos| ok({
        lmp.set_carts(pos.nest())?;
        lmp.compute().map(|(v,g)| (v, g.flat().to_vec()))?
    }))
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
        let flat_d_pos = dot_vec_mat_dumb(coeffs, flat_evs);
        let V(flat_pos): V<Vec<_>> = v(flat_init_pos) + v(flat_d_pos);

        let (value, flat_grad) = compute_from_3n_flat(&flat_pos)?;

        let grad = dot_mat_vec_dumb(flat_evs, &flat_grad);
        (value, grad)
    }))
}

//----------------------
// a slice of slices is a really dumb representation for a matrix
// but hey; what works, works

fn dot_vec_mat_dumb(vec: &[f64], mat: &[&[f64]]) -> Vec<f64>
{ mat.iter().map(|row| vdot(vec, row)).collect() }

fn dot_mat_vec_dumb(mat: &[&[f64]], vec: &[f64]) -> Vec<f64>
{
    assert_eq!(mat.len(), vec.len());
    assert_ne!(mat.len(), 0, "cannot determine width of matrix with no rows");
    let init = v(vec![0.0; mat[0].len()]);
    let V(out) = izip!(mat, vec)
        .fold(init, |acc, (&row, &alpha)| acc + alpha * v(row));
    out
}

//=================================================================

pub fn run_symmetry_test(input: &Path) -> Result<()>
{ok({
    use ::rsp2_structure_io::poscar;

    setup_global_logger(None)?;

    let poscar = poscar::load(open(input)?)?;
    let symmops = PhonopyBuilder::new().symmetry(&poscar)?;
    ::rsp2_structure::dumb_symmetry_test(&poscar.map_metadata_to(|_| ()), &symmops, 1e-6)?;
})}

//=================================================================

pub fn get_energy_surface(
    settings: &::config::EnergyPlotSettings,
    input: &AsRef<Path>,
    outdir: &AsRef<Path>,
) -> Result<()>
{ok({
    use ::rsp2_structure_io::poscar;

    let structure = &poscar::load(open(input)?)?;

    let lmp = make_lammps_builder(&settings.threading);
    let phonopy = phonopy_builder_from_settings(&settings.phonons, &structure);

    create_dir(&outdir)?;
    {
        // dumb/lazy solution to ensuring all output files go in the dir
        let cwd_guard = push_dir(outdir)?;
        setup_global_logger(Some(&"rsp2.log"))?;

        poscar::dump(create("./input.vasp")?, "", &structure)?;

        let (evals, evecs) = do_diagonalize(&lmp, &phonopy, &structure)?;

        trace!("Finding layers");
        let (layers, nlayer) = ::rsp2_structure::assign_layers(&structure, &[0, 0, 1], 0.25)?;
        assert_eq!(nlayer, 2);

        trace!("Computing eigensystem info");
        let einfos = get_eigensystem_info(&evals, &evecs, &layers, None, None);

        write_eigen_info(&einfos, &mut |s| ok(info!("{}", s)))?;

        let shear_evecs = {
            let mut iter = einfos
                .iter().enumerate()
                .filter(|&(_, info)| info.is_shear())
                .map(|(i, _)| evecs[i].clone());

            match (iter.next(), iter.next()) {
                (Some(a), Some(b)) => (a, b),
                _ => panic!("Expected at least two shear modes"),
            }
        };

        const W: usize = 200;
        const H: usize = 200;
        let data = {
            let mut lmp = lmp.initialize_carbon(uncarbon(&structure))?;

            ::integrate_2d::integrate_two_eigenvectors(
                (W, H),
                &structure.to_carts(),
                (-10.0..10.0, -10.0..10.0),
                (&shear_evecs.0, &shear_evecs.1),
                {
                    let mut i = 0;
                    move |pos| {ok({
                        i += 1;
                        eprint!("\rdatapoint {:>6} of {}", i, W * H);
                        lmp.set_carts(&pos)?;
                        lmp.compute_grad()?
                    })}
                }
            )?
        };
        let chunked: Vec<_> = data.chunks(W).collect();
        ::serde_json::to_writer_pretty(create("out.json")?, &chunked)?;
        cwd_guard.pop()?;
    }

})}


//=================================================================

pub fn make_force_sets(
    conf: Option<&AsRef<Path>>,
    poscar: &AsRef<Path>,
    outdir: &AsRef<Path>,
) -> Result<()>
{ok({
    use ::rsp2_structure_io::poscar;
    use ::std::io::BufReader;

    let mut phonopy = PhonopyBuilder::new();
    if let Some(conf) = conf {
        phonopy = phonopy.conf_from_file(BufReader::new(open(conf)?))?;
    }

    let structure = poscar::load(open(poscar)?)?;

    let lmp = make_lammps_builder(&::config::Threading::Lammps);

    create_dir(&outdir)?;
    {
        // dumb/lazy solution to ensuring all output files go in the dir
        let cwd_guard = push_dir(outdir)?;
        setup_global_logger(Some(&"rsp2.log"))?;

        poscar::dump(create("./input.vasp")?, "", &structure)?;

        let disp_dir = phonopy.displacements(&structure)?;
        let force_sets = do_force_sets_at_disps(&lmp, &disp_dir)?;
        disp_dir.make_force_dir_in_dir(&force_sets, ".")?;

        cwd_guard.pop()?;
    }
})}

//=================================================================
// error chaining helpers that tell us what file had a problem

pub(crate) fn open<P: AsRef<Path>>(path: P) -> Result<File>
{
    File::open(path.as_ref())
        .chain_err(|| format!("while opening file: '{}'", path.as_ref().display()))
}

pub(crate) fn create<P: AsRef<Path>>(path: P) -> Result<File>
{
    File::create(path.as_ref())
        .chain_err(|| format!("while creating file: '{}'", path.as_ref().display()))
}

#[allow(unused)]
pub(crate) fn open_text<P: AsRef<Path>>(path: P) -> Result<BufReader<File>>
{ open(path).map(BufReader::new) }

pub(crate) fn copy<P: AsRef<Path>, Q: AsRef<Path>>(src: P, dest: Q) -> Result<()>
{
    let (src, dest) = (src.as_ref(), dest.as_ref());
    fs::copy(src, dest)
        .map(|_| ()) // number of bytes; don't care
        .chain_err(||
            format!("while copying '{}' to '{}'",
                src.display(), dest.display()))
}

pub(crate) fn create_dir<P: AsRef<Path>>(dir: P) -> Result<()>
{
    fs::create_dir(dir.as_ref())
        .chain_err(|| format!("while creating directory '{}'", dir.as_ref().display()))
}

pub(crate) fn canonicalize<P: AsRef<Path>>(dir: P) -> Result<PathBuf>
{
    fs::canonicalize(dir.as_ref())
        .chain_err(|| format!("while looking for '{}'", dir.as_ref().display()))
}

//=================================================================

fn tup3<T:Copy>(arr: [T; 3]) -> (T, T, T) { (arr[0], arr[1], arr[2]) }
