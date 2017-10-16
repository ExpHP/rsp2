// HERE BE DRAGONS

extern crate sp2_lammps_wrap;
extern crate sp2_minimize;
extern crate sp2_structure;
extern crate sp2_structure_io;
extern crate sp2_phonopy_io;
extern crate sp2_array_utils;
extern crate sp2_slice_math;
extern crate sp2_tempdir;
extern crate sp2_eigenvector_classify;
#[macro_use] extern crate sp2_util_macros;

extern crate rand;
extern crate env_logger;
extern crate slice_of_array;
extern crate serde;
extern crate serde_json;
#[macro_use] extern crate serde_derive;
#[macro_use] extern crate log;
#[macro_use] extern crate itertools;

const THZ_TO_WAVENUMBER: f64 = 33.35641;

use ::sp2_array_utils::vec_from_fn;
use ::slice_of_array::prelude::*;
use ::sp2_structure::{supercell, Coords, CoordStructure, Lattice};
use ::sp2_lammps_wrap::Lammps;
use ::std::path::Path;

type LmpError = ::sp2_lammps_wrap::Error;

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct Settings {
    supercell_relax: SupercellSpec,
    supercell_phonopy: SupercellSpec,
    displacement_distance: f64, // 1e-3
    neg_frequency_threshold: f64, // 1e-3
    hack_scale: [f64; 3], // HACK
    cg: ::sp2_minimize::acgsd::Settings,
}

// fn array_sum(arrs: &[[f64; 3]]) -> [f64; 3] {
//     let mut acc = [0.0, 0.0, 0.0];
//     for arr in arrs {
//         acc = vec_from_fn(|k| acc[k] + arr[k]);
//     }
//     acc
// }

// fn array_mean(arrs: &[[f64; 3]]) -> [f64; 3] {
//     assert!(arrs.len() > 0);
//     let out = array_sum(arrs);
//     vec_from_fn(|k| out[k] / arrs.len() as f64)
// }

// fn remove_mean_shift(a: &mut [[f64; 3]], b: &[[f64; 3]]) {
//     let shifts = v(a.flat()) - v(b.flat());
//     let mean = array_mean(shifts.nest());
//     for row in a {
//         *row = vec_from_fn(|k| row[k] - mean[k]);
//     }
// }

fn lammps_flat_diff_fn<'a>(lmp: &'a mut Lammps)
-> Box<FnMut(&[f64]) -> Result<(f64, Vec<f64>), LmpError> + 'a>
{
    Box::new(move |pos| {
        lmp.set_carts(pos.nest())?;
        lmp.compute().map(|(v,g)| (v, g.flat().to_vec()))
    })
}

pub enum Panic {}
impl<E: ::std::fmt::Debug> From<E> for Panic {
    fn from(e: E) -> Panic { Err::<(),_>(e).unwrap(); unreachable!() }
}

// fn numerical_lattice_param_slope(structure: &CoordStructure, mask: [f64; 3]) -> [f64; 3]
// {
//     vec![
//         -f64::exp2(-47.),
//         -f64::exp2(-48.),
//         -f64::exp2(-49.),
//         -f64::exp2(-50.),
//         -f64::exp2(-51.),
//         f64::exp2(-51.),
//         f64::exp2(-50.),
//         f64::exp2(-49.),
//         f64::exp2(-48.),
//         f64::exp2(-47.),
//     ].into_iter().map(|x| {
//         let scales = [1.0 + mask[0] * x, 1.0 + mask[1] * x, 1.0 + mask[2] * x];
//         let mut structure = structure.clone();
//         structure.scale_vecs(&scales);
//         // scale_vecs.
//         panic!("TODO")
//     });

//     *out.as_array().unwrap()
// }

pub fn run_relax_with_eigenvectors<P, Q>(settings: &Settings, input: P, outdir: Q) -> Result<(), Panic>
where P: AsRef<Path>, Q: AsRef<Path>,
{
    use ::std::io::prelude::*;
    use ::sp2_phonopy_io as p;
    use ::sp2_structure_io::poscar;
    use ::sp2_slice_math::{v, V, vdot};
    use ::std::fs::File;

    let relax = |structure: CoordStructure| -> Result<CoordStructure, Panic> {
        let sc_dims = tup3(settings.supercell_relax.dim_for_unitcell(structure.lattice()));
        let (supercell, sc_token) = supercell::diagonal(sc_dims, structure);

        // FIXME confusing for Lammps::new_carbon to take initial position
        let mut lmp = Lammps::new_carbon(supercell.clone())?;
        let relaxed_flat = ::sp2_minimize::acgsd(
            &settings.cg,
            &supercell.to_carts().flat(),
            &mut *lammps_flat_diff_fn(&mut lmp),
        )?.position;

        let supercell = supercell.with_coords(Coords::Carts(relaxed_flat.nest().to_vec()));
        Ok(multi_threshold_deconstruct(sc_token, 1e-10, 1e-3, supercell)?)
    };

    use ::sp2_structure::supercell::{SupercellToken, DeconstructionError};
    fn multi_threshold_deconstruct(
        sc_token: SupercellToken,
        warn: f64,
        fail: f64,
        supercell: CoordStructure,
    ) -> Result<CoordStructure, DeconstructionError>
    {
        match sc_token.deconstruct(warn, supercell.clone()) {
            Ok(x) => Ok(x),
            Err(e) => {
                warn!("Suspiciously broad deviations in supercell: {:?}", e);
                Ok(sc_token.deconstruct(fail, supercell)?)
            }
        }
    }

    let diagonalize = |structure: CoordStructure| -> Result<_, Panic> {
        let conf = collect![
            (format!("DISPLACEMENT_DISTANCE"), format!("{:e}", settings.displacement_distance)),
            (format!("DIM"), {
                let (a, b, c) = tup3(settings.supercell_phonopy.dim_for_unitcell(structure.lattice()));
                format!("{} {} {}", a, b, c)
            }),
            (format!("HDF5"), format!(".TRUE.")),
            (format!("DIAG"), format!(".FALSE.")), // maybe?
        ];

        let (superstructure, displacements, disp_token)
            = p::cmd::phonopy_displacements_carbon(&conf, structure)?;

        let mut lmp = Lammps::new_carbon(superstructure.clone())?;
        println!();
        let mut i = 0;
        let force_sets = p::force_sets::compute_from_grad(
            superstructure.clone(), // FIXME only a borrow is needed later, and only for natom (dumb)
            &displacements,
            |s| {
                i += 1;
                print!("\rdisp {} of {}", i, displacements.len());
                ::std::io::stdout().flush().unwrap();
                lmp.set_structure(s.clone())?;
                lmp.compute().map(|diff| diff.1)
            }
        )?;
        println!();

        let (eval, evec) = p::cmd::phonopy_gamma_eigensystem(&conf, force_sets, &disp_token)?;
        let V(eval) = THZ_TO_WAVENUMBER * v(eval);
        Ok((eval, evec))
    };

    let minimize_evec = |structure: CoordStructure, evec: &[[f64; 3]]| -> Result<CoordStructure, Panic> {
        let sc_dims = tup3(settings.supercell_relax.dim_for_unitcell(structure.lattice()));
        let (structure, sc_token) = supercell::diagonal(sc_dims, structure);
        let evec = sc_token.replicate(evec);
        let mut lmp = Lammps::new_carbon(structure.clone())?;

        let from_structure = structure;
        let direction = &evec[..];
        let from_pos = from_structure.to_carts();
        let pos_at_alpha = |alpha| {
            let V(pos) = v(from_pos.flat()) + alpha * v(direction.flat());
            pos
        };
        let alpha = ::sp2_minimize::exact_ls::<LmpError, _>(0.0, 1e-4, |alpha| {
            let gradient = lammps_flat_diff_fn(&mut lmp)(&pos_at_alpha(alpha))?.1;
            let slope = vdot(&gradient[..], direction.flat());
            Ok(::sp2_minimize::exact_ls::Slope(slope))
        })??.alpha;
        let pos = pos_at_alpha(alpha);
        let structure = from_structure.with_coords(Coords::Carts(pos.nest().to_vec()));

        Ok(multi_threshold_deconstruct(sc_token, 1e-10, 1e-3, structure)?)
    };

    let write_eigen_info = |f: &mut Write, einfos: &EigenInfo| -> Result<_, Panic>
    {
        writeln!(f, "{:27}  {:4} {:5} [{:4.2}, {:4.2}, {:4.2}]",
            "# Frequency (cm^-1)", "Acou", "Layer", "X", "Y", "Z")?;
        for item in einfos.iter() {
            let eval = item.frequency;
            let acou = item.acousticness;
            let layer = item.layer_acousticness;
            let (x, y, z) = tup3(item.polarization);
            writeln!(f, "{:27}  {:4.2}  {:4.2} [{:4.2}, {:4.2}, {:4.2}]",
                eval, acou, layer, x, y, z)?;
        }
        Ok(())
    };

    let mut original = poscar::load_carbon(File::open(input)?)?;
    original.scale_vecs(&settings.hack_scale); // HACK
    let original = original;

    ::std::fs::create_dir(&outdir)?;
    {
        // dumb/lazy solution to ensuring all output files go in the dir
        let cwd_guard = push_dir(outdir)?;

        let mut from_structure = original;
        // HACK to stop one iteration AFTER all non-acoustics are positive
        let mut iteration = 1;
        let (structure, evals, _evecs) = loop { // NOTE: we use break with value
            let structure = relax(from_structure)?;
            let (evals, evecs) = diagonalize(structure.clone())?;

            println!("============================");
            println!("Finished relaxation # {}", iteration + 1);

            let (layers, _nlayer) = ::sp2_structure::assign_layers(&structure, &[0, 0, 1], 0.25);
            let einfos = get_eigensystem_info(&evals, &evecs, &layers[..]);
            write_eigen_info(
                &mut File::create(format!("eigenvalues.{:02}", iteration))?,
                &einfos,
            )?;
            {
                let out = ::std::io::stdout();
                write_eigen_info(&mut out.lock(), &einfos)?;
            }

            let mut all_ok = true;
            let mut structure = structure;
            for (i, info, evec) in izip!(1.., &einfos, &evecs) {
                if info.frequency < 0.0 && info.acousticness < 0.95 {
                    println!("!! Optimizing along band {} ({})", i, info.frequency);
                    structure = minimize_evec(structure, &evec[..])?;

                    all_ok = false;
                }
            }

            if all_ok {
                break (structure, evals, evecs);
            }

            println!("============================");

            from_structure = structure;
            iteration += 1;
        }; // (structure, evals, evecs)

        {
            let mut f = File::create("eigenvalues.final")?;
            for &x in &evals {
                writeln!(f, "{:?}", x)?;
            }
        }

        poscar::dump_carbon(File::create("./final.vasp")?, "", &structure)?;

        cwd_guard.pop()?;
    }

    Ok(())
}

pub type EigenInfo = Vec<eigen_info::Item>;

pub fn get_eigensystem_info(
    evals: &[f64],
    evecs: &[Vec<[f64; 3]>],
    layers: &[::sp2_structure::Layer],
) -> EigenInfo
{
    use ::sp2_eigenvector_classify::{keyed_acoustic_basis, polarization};

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
}

#[derive(Serialize, Deserialize)]
#[derive(Debug, Clone, PartialEq)]
#[serde(rename_all="kebab-case")]
pub enum SupercellSpec {
    Target([f64; 3]),
    Dim([u32; 3]),
}
impl SupercellSpec {
    fn dim_for_unitcell(&self, prim: &Lattice) -> [u32; 3] {
        match *self {
            SupercellSpec::Dim(d) => d,
            SupercellSpec::Target(targets) => {
                let unit_lengths = prim.lengths();
                vec_from_fn(|k| {
                    (targets[k] / unit_lengths[k]).ceil().max(1.0) as u32
                })
            },
        }
    }
}

// HACK
fn tup3<T:Copy>(arr: [T; 3]) -> (T, T, T) { (arr[0], arr[1], arr[2]) }

use util::push_dir;
mod util {
    use ::std::path::{Path, PathBuf};
    use ::std::io::Result as IoResult;

    /// RAII type to temporarily enter a directory.
    ///
    /// The recommended usage is actually not to rely on the implicit destructor
    /// (which panics on failure), but to instead explicitly call `.pop()`.
    /// The advantage of doing so over just manually calling 'set_current_dir'
    /// is the unused variable lint can help remind you to call `pop`.
    ///
    /// Usage is highly discouraged in multithreaded contexts where
    /// another thread may need to access the filesystem.
    #[must_use]
    pub struct PushDir(Option<PathBuf>);
    pub fn push_dir<P: AsRef<Path>>(path: P) -> IoResult<PushDir> {
        let old = ::std::env::current_dir()?;
        ::std::env::set_current_dir(path)?;
        Ok(PushDir(Some(old)))
    }

    impl PushDir {
        /// Explicitly destroy the PushDir.
        ///
        /// This lets you handle the IO error, and has an advantage over
        /// manual calls to 'env::set_current_dir' in that the compiler will
        pub fn pop(mut self) -> IoResult<()> {
            ::std::env::set_current_dir(self.0.take().unwrap())
        }
    }

    impl Drop for PushDir {
        fn drop(&mut self) {
            if let Some(d) = self.0.take() {
                if let Err(e) = ::std::env::set_current_dir(d) {
                    // uh oh.
                    panic!("automatic popdir failed: {}", e);
                }
            }
        }
    }
}