// HERE BE DRAGONS

const THZ_TO_WAVENUMBER: f64 = 33.35641;

use ::{Never, StdResult, Result};
use ::config::Settings;
use ::util::push_dir;
use ::logging::setup_global_logger;

use ::rsp2_structure::consts::CARBON;
use ::rsp2_slice_math::{v, V, vdot};

use ::slice_of_array::prelude::*;
use ::rsp2_structure::supercell::{self, SupercellToken};
use ::rsp2_structure::{Coords, CoordStructure};
use ::rsp2_structure::{ElementStructure};
use ::rsp2_lammps_wrap::Lammps;
use ::std::path::Path;
use ::std::hash::Hash;
use ::std::io::prelude::*;

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

    let mut original = poscar::load(File::open(input)?)?;
    original.scale_vecs(&settings.hack_scale); // HACK
    let original = original;

    ::std::fs::create_dir(&outdir)?;
    {
        // dumb/lazy solution to ensuring all output files go in the dir
        let cwd_guard = push_dir(outdir)?;

        poscar::dump(File::create("./initial.vasp")?, "", &original)?;

        setup_global_logger(Some(&"rsp2.log"))?;

        let mut from_structure = original.clone();
        let mut iteration = 1;
        // HACK to stop one iteration AFTER all non-acoustics are positive
        let mut all_ok_count = 0;
        let (structure, einfos, _evecs) = loop { // NOTE: we use break with value
            let structure = do_relax(settings, from_structure)?;
            let (evals, evecs) = do_diagonalize(&settings.phonons, structure.clone())?;

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
            let einfos = get_eigensystem_info(&evals, &evecs, &layers[..]);
            {
                let mut file = File::create(format!("eigenvalues.{:02}", iteration))?;
                write_eigen_info(&einfos, &mut |s| writeln!(file, "{}", s).map_err(Into::into))?;
            }
            write_eigen_info(&einfos, &mut |s| Ok::<_, Never>(info!("{}", s)))?;

            let mut all_ok = true;
            let mut structure = structure;
            for (i, info, evec) in izip!(1.., &einfos, &evecs) {
                if info.frequency < 0.0 && !info.is_acoustic() {
                    if all_ok {
                        trace!("Optimizing along bands...");
                        all_ok = false;
                    }
                    let (alpha, new_structure) = do_minimize_along_evec(settings, structure, &evec[..])?;
                    info!("Optimized along band {} ({}), a = {:e}", i, info.frequency, alpha);

                    structure = new_structure;
                }
            }

            {
                let fname = format!("./structure-{:02}.2.vasp", iteration);
                trace!("Writing '{}'", &fname);
                poscar::dump(
                    File::create(fname)?,
                    &format!("Structure after eigenmode-chasing round {}", iteration),
                    &structure)?;
            }

            if all_ok {
                all_ok_count += 1;
                if all_ok_count >= 3 {
                    break (structure, einfos, evecs);
                }
            }

            from_structure = structure;
            iteration += 1;
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
                let f = |structure: ElementStructure| {Ok::<_, Never>({
                    let na = structure.num_atoms() as f64;
                    Lammps::new_carbon(uncarbon(&structure))?.compute_value()? / na
                })};
                let f_path = |s: &AsRef<Path>| Ok::<_, Never>(f(poscar::load(File::open(s)?)?)?);

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
    settings: &Settings,
    structure: ElementStructure,
) -> Result<ElementStructure>
{Ok({
    let sc_dims = tup3(settings.supercell_relax.dim_for_unitcell(structure.lattice()));
    let (supercell, sc_token) = supercell::diagonal(sc_dims, structure);

    // FIXME confusing for Lammps::new_carbon to take initial position
    let mut lmp = Lammps::new_carbon(uncarbon(&supercell))?;
    let relaxed_flat = ::rsp2_minimize::acgsd(
        &settings.cg,
        &supercell.to_carts().flat(),
        &mut *lammps_flat_diff_fn(&mut lmp),
    ).unwrap().position;

    let supercell = supercell.with_coords(Coords::Carts(relaxed_flat.nest().to_vec()));
    multi_threshold_deconstruct(sc_token, 1e-10, 1e-3, supercell)?
})}

fn do_diagonalize(
    settings: &::config::Phonons,
    structure: ElementStructure,
) -> Result<(Vec<f64>, Vec<Vec<[f64; 3]>>)>
{Ok({

    let phonopy = ::rsp2_phonopy_io::cmd::Builder::new()
        .symmetry_tolerance(settings.symmetry_tolerance)
        .conf("DISPLACEMENT_DISTANCE", format!("{:e}", settings.displacement_distance))
        .conf("DIM", {
            let (a, b, c) = tup3(settings.supercell.dim_for_unitcell(structure.lattice()));
            format!("{} {} {}", a, b, c)
        })
        .conf("HDF5", ".TRUE.")
        .conf("DIAG", ".FALSE.") // maybe?
        ;

    let (superstructure, displacements, disp_token) = phonopy.displacements(structure)?;

    trace!("Computing forces at displacements");
    let mut lmp = Lammps::new_carbon(superstructure.clone())?;
    let mut i = 0;
    let force_sets =
        ::rsp2_phonopy_io::disp_yaml::displaced_structures(superstructure, &displacements)
        .map(|s| Ok({
            // TODO rayon vs lammps threads here
            i += 1;
            eprint!("\rdisp {} of {}", i, displacements.len());
            ::std::io::stdout().flush().unwrap();

            lmp.set_structure(s)?;
            let grad = lmp.compute_grad()?;
            let V(force) = -1.0 * v(grad.flat());
            force.nest().to_vec()
        })).collect::<Result<Vec<_>>>()?;
    eprintln!();

    let (eval, evec) = phonopy.gamma_eigensystem(force_sets, &disp_token)?;
    let V(eval) = THZ_TO_WAVENUMBER * v(eval);
    (eval, evec)
})}

fn do_minimize_along_evec(
    settings: &Settings,
    structure: ElementStructure,
    evec: &[[f64; 3]],
) -> Result<(f64, ElementStructure)>
{Ok({
    let sc_dims = tup3(settings.supercell_relax.dim_for_unitcell(structure.lattice()));
    let (structure, sc_token) = supercell::diagonal(sc_dims, structure);
    let evec = sc_token.replicate(evec);
    let mut lmp = Lammps::new_carbon(uncarbon(&structure))?;

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

    writeln(&format_args!("{:27}  {:^7}  {:^7} [{:^4}, {:^4}, {:^4}]",
        "# Frequency (cm^-1)", "Acoustc", "Layer", "X", "Y", "Z"))?;
    for item in einfos.iter() {
        let eval = item.frequency;
        let acou = dp(item.acousticness);
        let layer = dp(item.layer_acousticness);
        let (x, y, z) = tup3(item.polarization);
        writeln(&format_args!("{:27}  {}  {} [{:4.2}, {:4.2}, {:4.2}]",
            eval, acou, layer, pol(x), pol(y), pol(z)))?;
    }
})}

//-----------------------------------

pub type EigenInfo = Vec<eigen_info::Item>;

pub fn get_eigensystem_info<L: Eq + Clone + Hash>(
    evals: &[f64],
    evecs: &[Vec<[f64; 3]>],
    layers: &[L],
) -> EigenInfo
{
    use ::rsp2_eigenvector_classify::{keyed_acoustic_basis, polarization};

    let layers: Vec<_> = layers.iter().cloned().map(Some).collect();
    let layer_acoustics = keyed_acoustic_basis(&layers[..], &[1,1,1]);
    let acoustics = keyed_acoustic_basis(&vec![Some(()); evecs[0].len()], &[1,1,1]);

    let mut out = vec![];
    for (&eval, evec) in izip!(evals, evecs) {
        out.push(eigen_info::Item {
            frequency: eval,
            acousticness: acoustics.probability(&evec),
            layer_acousticness: layer_acoustics.probability(&evec),
            polarization: polarization(&evec[..]),
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

fn lammps_flat_diff_fn<'a>(lmp: &'a mut Lammps)
-> Box<FnMut(&[f64]) -> StdResult<(f64, Vec<f64>), LmpError> + 'a>
{
    Box::new(move |pos| {
        lmp.set_carts(pos.nest())?;
        lmp.compute().map(|(v,g)| (v, g.flat().to_vec()))
    })
}

//=================================================================

pub fn run_symmetry_test(input: &Path) -> Result<()>
{Ok({
    use ::rsp2_structure_io::poscar;
    use ::std::fs::File;

    setup_global_logger(None)?;

    let poscar = poscar::load(File::open(input)?)?;
    let symmops = ::rsp2_phonopy_io::cmd::Builder::new().symmetry(&poscar)?;
    ::rsp2_structure::dumb_symmetry_test(&poscar.map_metadata_to(|_| ()), &symmops, 1e-6)?;
})}

//=================================================================

pub fn get_energy_surface(
    settings: &::config::Phonons,
    input: &AsRef<Path>,
    outdir: &AsRef<Path>,
) -> Result<()>
{Ok({
    use ::std::io::prelude::*;
    use ::rsp2_structure_io::poscar;
    use ::std::fs::File;

    let mut structure = &poscar::load(File::open(input)?)?;

    ::std::fs::create_dir(&outdir)?;
    {
        // dumb/lazy solution to ensuring all output files go in the dir
        let cwd_guard = push_dir(outdir)?;
        setup_global_logger(Some(&"rsp2.log"))?;

        poscar::dump(File::create("./input.vasp")?, "", &structure)?;

        let (evals, evecs) = do_diagonalize(settings, structure.clone())?;

        trace!("Computing eigensystem info");
        let einfos = get_eigensystem_info(&evals, &evecs, &vec![(); structure.num_atoms()][..]);

        write_eigen_info(&einfos, &mut |s| Ok::<_, Never>(info!("{}", s)))?;

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

        let mut lmp = Lammps::new_carbon(uncarbon(&structure))?;
        ::integrate_2d::integrate_two_eigenvectors::<Never,_>(
            (200, 200),
            &structure.to_carts(),
            (-1.0..1.0, -1.0..1.0),
            (&shear_evecs.0, &shear_evecs.1),
            |pos| {Ok({
                lmp.set_carts(&pos)?;
                lmp.compute_grad()?
            })}
        )?;
        cwd_guard.pop()?;
    }
})}


//=================================================================

fn tup3<T:Copy>(arr: [T; 3]) -> (T, T, T) { (arr[0], arr[1], arr[2]) }
