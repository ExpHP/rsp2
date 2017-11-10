// HERE BE DRAGONS

const THZ_TO_WAVENUMBER: f64 = 33.35641;

use ::{StdResult, Result, Error};
use ::config::Settings;
use ::util::push_dir;
use ::logging::setup_global_logger;

use ::rsp2_structure::consts::CARBON;
use ::rsp2_slice_math::{v, V, vdot};

use ::slice_of_array::prelude::*;
use ::rsp2_structure::supercell::{self, SupercellToken};
use ::rsp2_structure::{Coords, CoordStructure};
use ::rsp2_structure::{ElementStructure};
use ::rsp2_structure_gen::Assemble;
use ::rsp2_lammps_wrap::Lammps;
use ::rsp2_lammps_wrap::Builder as LammpsBuilder;
use ::rsp2_phonopy_io::cmd::Builder as PhonopyBuilder;

use ::std::path::Path;
use ::std::hash::Hash;

type LmpError = ::rsp2_lammps_wrap::Error;

pub fn run_relax_with_eigenvectors(
    settings: &Settings,
    input: &AsRef<Path>,
    outdir: &AsRef<Path>,
) -> Result<()>
{Ok({
    use ::std::io::prelude::*;
    use ::rsp2_structure_io::poscar;
    use ::std::fs::File;

    let lmp = make_lammps_builder(&settings.threading);

    ::std::fs::create_dir(&outdir)?;
    {
        // dumb/lazy solution to ensuring all output files go in the dir
        let cwd_guard = push_dir(outdir)?;

        setup_global_logger(Some(&"rsp2.log"))?;

        // let mut original = poscar::load(File::open(input)?)?;
        // original.scale_vecs(&settings.hack_scale);
        let (original, layer_1_sc_mat) = {
            use ::rsp2_structure_gen::load_layers_yaml;
            use ::rsp2_structure_gen::layer_sc_info_from_layers_yaml;

            let builder = load_layers_yaml(File::open(input)?)?;
            let builder = optimize_layer_parameters(
                &settings.scale_ranges,
                &lmp,
                builder,
            )?;
            let structure = carbon(&builder.assemble());

            let layer_sc_info = layer_sc_info_from_layers_yaml(File::open(input)?)?;
            let (matrix, periods) = layer_sc_info[0];
            let layer_1_sc_mat = ::bands::ScMatrix::new(&matrix, &periods);

            (structure, layer_1_sc_mat)
        };

        poscar::dump(File::create("./initial.vasp")?, "", &original)?;

        let mut from_structure = original.clone();
        let mut iteration = 1;
        // HACK to stop one iteration AFTER all non-acoustics are positive
        let mut all_ok_count = 0;
        let (structure, einfos, _evecs) = loop { // NOTE: we use break with value

            let structure = do_relax(&lmp, &settings.cg, &settings.potential, from_structure)?;
            let (evals, evecs) = do_diagonalize(&lmp, &settings.phonons, structure.clone())?;

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
                    File::create(fname)?,
                    &format!("Structure after CG round {}", iteration),
                    &structure)?;
            }

            trace!("Computing eigensystem info");
            let einfos = get_eigensystem_info(
                &evals, &evecs, &layers[..], Some(&structure), Some(&layer_1_sc_mat),
            );
            {
                let mut file = File::create(format!("eigenvalues.{:02}", iteration))?;
                write_eigen_info(&einfos, &mut |s| writeln!(file, "{}", s).map_err(Into::into))?;
            }
            write_eigen_info(&einfos, &mut |s| Ok::<_, Error>(info!("{}", s)))?;

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
                    File::create(fname)?,
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
            let mut f = File::create("eigenvalues.final")?;
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

        poscar::dump(File::create("./final.vasp")?, "", &structure)?;

        { // write summary file
            use self::summary::{Modes, Summary, EnergyPerAtom};
            use self::eigen_info::Item;

            let acoustic = einfos.iter().filter(|x| x.is_acoustic()).map(Item::frequency).collect();
            let shear = einfos.iter().filter(|x| x.is_shear()).map(Item::frequency).collect();
            let layer_breathing = einfos.iter().filter(|x| x.is_layer_breathing()).map(Item::frequency).collect();
            let modes = Modes { acoustic, shear, layer_breathing };

            let energy_per_atom = {
                let f = |structure: ElementStructure| {Ok::<_, Error>({
                    let na = structure.num_atoms() as f64;
                    lmp.initialize_carbon(uncarbon(&structure))?.compute_value()? / na
                })};
                let f_path = |s: &AsRef<Path>| Ok::<_, Error>(f(poscar::load(File::open(s)?)?)?);

                let initial = f(original.clone())?;
                let final_ = f(structure)?;
                let before_ev_chasing = f_path(&"structure-01.1.vasp")?;
                EnergyPerAtom { initial, final_, before_ev_chasing }
            };

            let summary = Summary { modes, energy_per_atom };

            ::serde_yaml::to_writer(File::create("summary.yaml")?, &summary)?;
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
{Ok({
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
    settings: &::config::Phonons,
    structure: ElementStructure,
) -> Result<(Vec<f64>, Vec<Vec<[f64; 3]>>)>
{Ok({
    use ::rsp2_phonopy_io::disp_yaml::apply_displacement;
    use ::std::io::prelude::*;

    let phonopy = PhonopyBuilder::new()
        .symmetry_tolerance(settings.symmetry_tolerance)
        .conf("DISPLACEMENT_DISTANCE", format!("{:e}", settings.displacement_distance))
        .supercell_dim(settings.supercell.dim_for_unitcell(structure.lattice()))
        .conf("HDF5", ".TRUE.")
        .conf("DIAG", ".FALSE.") // maybe?
        ;

    let (superstructure, displacements, disp_token) = phonopy.displacements(structure)?;

    trace!("Computing forces at displacements");

    let mut i = 0;
    let force_sets =
        displacements.iter()
        .map(|disp| Ok({
            i += 1;
            eprint!("\rdisp {} of {}", i, displacements.len());
            ::std::io::stderr().flush().unwrap();
            let superstructure = apply_displacement(&superstructure, *disp);

            lmp.initialize_carbon(superstructure)?.compute_force()?
        })).collect::<Result<Vec<_>>>()?;

    // NOTE: Here's how you *could* do them in parallel,
    //       but attempting to do so with lammps leads to segfaults.
    #[cfg(nope)]
    {
        // bigger chunks = fewer opportunities for starved worker threads
        // smaller chunks = more frequent visual feedback
        const CHUNK_SIZE: usize = 100;
        let superstructure = superstructure.send();
        let force_sets =
            displacements
            .chunks(CHUNK_SIZE)
            .enumerate()
            .map(|(chunk_index, chunk)| {Ok({
                let buf: Vec<_> =
                    chunk.par_iter()
                    .map(|&disp| {Ok({
                        let superstructure = superstructure.clone().recv();
                        let superstructure = apply_displacement(&superstructure, disp);
                        lmp.initialize_carbon(superstructure.clone())?.compute_force()?
                    })})
                    .collect::<Result<_>>()?;

                eprintln!(
                    "Completed {:>6} displacements (of {})",
                    CHUNK_SIZE * (chunk_index + 1), displacements.len());

                buf
            })})
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .fold(
                Vec::with_capacity(displacements.len()),
                |mut acc, chunk| { acc.extend(chunk); acc }
            );
    }

    let (eval, evec) = phonopy.gamma_eigensystem(force_sets, &disp_token)?;
    let V(eval) = THZ_TO_WAVENUMBER * v(eval);
    (eval, evec)
})}

fn do_eigenvector_chase(
    lmp: &LammpsBuilder,
    chase_settings: &::config::EigenvectorChase,
    potential_settings: &::config::Potential,
    mut structure: ElementStructure,
    bad_evecs: &[(String, &[[f64; 3]])],
) -> Result<ElementStructure>
{Ok({
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
{Ok({
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
{Ok({
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
{Ok({
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
    let alpha = ::rsp2_minimize::exact_ls::<LmpError, _>(0.0, 1e-4, |alpha| {
        let gradient = lammps_flat_diff_fn(&mut lmp)(&pos_at_alpha(alpha))?.1;
        let slope = vdot(&gradient[..], direction.flat());
        Ok(::rsp2_minimize::exact_ls::Slope(slope))
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
) -> StdResult<ElementStructure, ::rsp2_structure::Error>
{Ok({
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
{Ok({
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

    writeln(&format_args!("{:27}  {:^7}  {:^7}  {:^7} [{:^4}, {:^4}, {:^4}]",
        "# Frequency (cm^-1)", "Acoustc", "Layer",
        "GammaL1", "X", "Y", "Z"))?;
    for item in einfos.iter() {
        let eval = item.frequency;
        let acou = dp(item.acousticness);
        let layer = dp(item.layer_acousticness);
        let (x, y, z) = tup3(item.polarization);
        let gamma: Box<::std::fmt::Display> = match item.layer_1_gamma_prob {
            Some(prob) => Box::new(dp(prob)),
            None => Box::new("-------"),
        };
        writeln(&format_args!("{:27}  {}  {}  {} [{:4.2}, {:4.2}, {:4.2}]",
            eval, acou, layer, gamma, pol(x), pol(y), pol(z)))?;
    }
})}

//-----------------------------------

pub fn optimize_layer_parameters(
    settings: &::config::ScaleRanges,
    lmp: &LammpsBuilder,
    mut builder: Assemble,
) -> Result<Assemble>
{Ok({
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
        let get_value = || Ok::<_, Error>({
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
    layer_1_supercell_matrix: Option<&::bands::ScMatrix>,
) -> EigenInfo
{
    use ::rsp2_eigenvector_classify::{keyed_acoustic_basis, polarization};

    let layers: Vec<_> = layers.iter().cloned().map(Some).collect();
    let layer_acoustics = keyed_acoustic_basis(&layers[..], &[1,1,1]);
    let acoustics = keyed_acoustic_basis(&vec![Some(()); evecs[0].len()], &[1,1,1]);

    let mut out = vec![];
    for (&eval, evec) in izip!(evals, evecs) {

        let gamma_prob = match (structure, layer_1_supercell_matrix) {
            (Some(structure), Some(layer_1_supercell_matrix)) => {Some({
                use ::rsp2_kets::{Rect, Ket};
                let ket: Ket = evec.flat().iter().cloned().map(Rect::from).collect();

                ::bands::unfold_phonon(
                    &from_json!({
                        "fbz": "reciprocal-cell",
                        "sampling": { "plain": [4, 4, 1] },
                    }),
                    &structure.map_metadata_to(|_| ()),
                    &[0.0, 0.0, 0.0], // eigenvector q
                    ket.as_ref(),
                    layer_1_supercell_matrix,
                ).iter()
                    .find(|&&(idx, _)| idx == [0, 0, 0])
                    .unwrap().1
            })},
            _ => None,
        };

        out.push(eigen_info::Item {
            frequency: eval,
            acousticness: acoustics.probability(&evec),
            layer_acousticness: layer_acoustics.probability(&evec),
            polarization: polarization(&evec[..]),
            layer_1_gamma_prob: gamma_prob,
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
        pub layer_1_gamma_prob: Option<f64>,
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
-> Box<FnMut(&[f64]) -> StdResult<(f64, Vec<f64>), LmpError> + 'a>
{
    Box::new(move |pos| {
        lmp.set_carts(pos.nest())?;
        lmp.compute().map(|(v,g)| (v, g.flat().to_vec()))
    })
}

// cg differential function along a restricted set of eigenvectors.
//
// There will be one coordinate for each eigenvector.
fn lammps_constrained_diff_fn<'a>(
    lmp: &'a mut Lammps,
    flat_init_pos: &'a [f64],
    flat_evs: &'a [&[f64]],
)
-> Box<FnMut(&[f64]) -> StdResult<(f64, Vec<f64>), LmpError> + 'a>
{
    let mut compute_from_3n_flat = lammps_flat_diff_fn(lmp);

    Box::new(move |coeffs| {Ok({
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
    })})
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
{Ok({
    use ::rsp2_structure_io::poscar;
    use ::std::fs::File;

    setup_global_logger(None)?;

    let poscar = poscar::load(File::open(input)?)?;
    let symmops = PhonopyBuilder::new().symmetry(&poscar)?;
    ::rsp2_structure::dumb_symmetry_test(&poscar.map_metadata_to(|_| ()), &symmops, 1e-6)?;
})}

//=================================================================

pub fn get_energy_surface(
    settings: &::config::EnergyPlotSettings,
    input: &AsRef<Path>,
    outdir: &AsRef<Path>,
) -> Result<()>
{Ok({
    use ::rsp2_structure_io::poscar;
    use ::std::fs::File;

    let structure = &poscar::load(File::open(input)?)?;

    let lmp = make_lammps_builder(&settings.threading);

    ::std::fs::create_dir(&outdir)?;
    {
        // dumb/lazy solution to ensuring all output files go in the dir
        let cwd_guard = push_dir(outdir)?;
        setup_global_logger(Some(&"rsp2.log"))?;

        poscar::dump(File::create("./input.vasp")?, "", &structure)?;

        let (evals, evecs) = do_diagonalize(&lmp, &settings.phonons, structure.clone())?;

        trace!("Finding layers");
        let (layers, nlayer) = ::rsp2_structure::assign_layers(&structure, &[0, 0, 1], 0.25)?;
        assert_eq!(nlayer, 2);

        trace!("Computing eigensystem info");
        let einfos = get_eigensystem_info(&evals, &evecs, &layers, None, None);

        write_eigen_info(&einfos, &mut |s| Ok::<_, Error>(info!("{}", s)))?;

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

            ::integrate_2d::integrate_two_eigenvectors::<Error,_>(
                (W, H),
                &structure.to_carts(),
                (-10.0..10.0, -10.0..10.0),
                (&shear_evecs.0, &shear_evecs.1),
                {
                    let mut i = 0;
                    move |pos| {Ok({
                        i += 1;
                        eprint!("\rdatapoint {:>6} of {}", i, W * H);
                        lmp.set_carts(&pos)?;
                        lmp.compute_grad()?
                    })}
                }
            )?
        };
        let chunked: Vec<_> = data.chunks(W).collect();
        ::serde_json::to_writer_pretty(File::create("out.json")?, &chunked)?;
        cwd_guard.pop()?;
    }

})}


//=================================================================

pub fn make_force_sets(
    conf: Option<&AsRef<Path>>,
    poscar: &AsRef<Path>,
    outdir: &AsRef<Path>,
) -> Result<()>
{Ok({
    use ::rsp2_structure_io::poscar;
    use ::std::fs::File;
    use ::std::io::BufReader;
    use ::rsp2_phonopy_io::disp_yaml::apply_displacement;

    let mut phonopy = PhonopyBuilder::new();
    if let Some(conf) = conf {
        phonopy = phonopy.conf_from_file(BufReader::new(File::open(conf)?))?;
    }

    let structure = poscar::load(File::open(poscar)?)?;

    let lmp = make_lammps_builder(&::config::Threading::Lammps);

    ::std::fs::create_dir(&outdir)?;
    {
        // dumb/lazy solution to ensuring all output files go in the dir
        let cwd_guard = push_dir(outdir)?;
        setup_global_logger(Some(&"rsp2.log"))?;

        poscar::dump(File::create("./input.vasp")?, "", &structure)?;

        let (superstructure, displacements, _disp_token) = phonopy.displacements(structure)?;

        trace!("Computing forces at displacements");

        let mut i = 0;
        let force_sets =
            displacements.iter()
            .map(|disp| Ok({
                use ::std::io::prelude::*;
                i += 1;
                eprint!("\rdisp {} of {}", i, displacements.len());
                ::std::io::stderr().flush().unwrap();
                let superstructure = apply_displacement(&superstructure, *disp);

                lmp.initialize_carbon(superstructure)?.compute_force()?
            })).collect::<Result<Vec<_>>>()?;

        ::rsp2_phonopy_io::force_sets::write(
            File::create("FORCE_SETS")?,
            &superstructure,
            &displacements,
            &force_sets,
        )?;

        cwd_guard.pop()?;
    }
})}

//=================================================================

fn tup3<T:Copy>(arr: [T; 3]) -> (T, T, T) { (arr[0], arr[1], arr[2]) }
