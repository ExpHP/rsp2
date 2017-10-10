// HERE BE DRAGONS

extern crate sp2_lammps_wrap;
extern crate sp2_minimize;
extern crate sp2_structure;
extern crate sp2_structure_io;
extern crate sp2_phonopy_io;
extern crate sp2_array_utils;
extern crate sp2_slice_of_array;
extern crate sp2_slice_math;
extern crate sp2_tempdir;
#[macro_use] extern crate sp2_util_macros;

extern crate rand;
extern crate env_logger;
extern crate serde;
extern crate serde_json;
#[macro_use] extern crate serde_derive;
#[macro_use] extern crate log;

const THZ_TO_WAVENUMBER: f64 = 33.35641;

use ::rand::random;
use ::sp2_array_utils::vec_from_fn;
use ::sp2_slice_of_array::prelude::*;
use ::sp2_slice_math::{v,vnorm};
use ::sp2_structure::{supercell, Coords, CoordStructure};
use ::sp2_lammps_wrap::Lammps;
use ::std::io::Result as IoResult;
use ::std::path::{Path, PathBuf};

type LmpError = ::sp2_lammps_wrap::Error;

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct Settings {
    supercell_relax: (u32, u32, u32),
    supercell_phonopy: (u32, u32, u32),
    displacement_distance: f64, // 1e-3
    neg_frequency_threshold: f64, // 1e-3
    cg: ::sp2_minimize::acgsd::Settings,
}

fn array_sum(arrs: &[[f64; 3]]) -> [f64; 3] {
    let mut acc = [0.0, 0.0, 0.0];
    for arr in arrs {
        acc = vec_from_fn(|k| acc[k] + arr[k]);
    }
    acc
}

fn array_mean(arrs: &[[f64; 3]]) -> [f64; 3] {
    assert!(arrs.len() > 0);
    let out = array_sum(arrs);
    vec_from_fn(|k| out[k] / arrs.len() as f64)
}

fn remove_mean_shift(a: &mut [[f64; 3]], b: &[[f64; 3]]) {
    let shifts = v(a.flat()) - v(b.flat());
    let mean = array_mean(shifts.nest());
    for row in a {
        *row = vec_from_fn(|k| row[k] - mean[k]);
    }
}

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
    use ::sp2_array_utils::prelude::*;
    use ::sp2_structure_io::{xyz, poscar};
    use ::sp2_slice_math::{v, V, vdot};
    use ::std::fs::File;

    let relax = |structure| -> Result<CoordStructure, Panic> {
        let (supercell, sc_token) = supercell::diagonal(settings.supercell_phonopy, structure);

        // FIXME confusing for Lammps::new_carbon to take initial position
        let mut lmp = Lammps::new_carbon(supercell.clone())?;
        let relaxed_flat = ::sp2_minimize::acgsd(
            &settings.cg,
            &supercell.to_carts().flat(),
            &mut *lammps_flat_diff_fn(&mut lmp),
        )?.position;

        let supercell = supercell.with_coords(Coords::Carts(relaxed_flat.nest().to_vec()));
        match sc_token.deconstruct(1e-5, supercell.clone()) {
            Ok(x) => Ok(x),
            Err(e) => {
                warn!("Suspiciously broad deviations in supercell: {:?}", e);
                Ok(sc_token.deconstruct(1.0, supercell)?)
            }
        }
    };

    let diagonalize = |structure| -> Result<_, Panic> {
        let conf = collect![
            (format!("DISPLACEMENT_DISTANCE"), format!("{:e}", settings.displacement_distance)),
            (format!("DIM"), {
                let (a, b, c) = settings.supercell_phonopy;
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
                lmp.set_structure(s.clone());
                lmp.compute().map(|diff| diff.1)
            }
        )?;
        println!();

        let (eval, evec) = p::cmd::phonopy_gamma_eigensystem(&conf, force_sets, &disp_token)?;
        let V(eval) = THZ_TO_WAVENUMBER * v(eval);
        Ok((eval, evec))
    };

    let minimize_evec = |structure: CoordStructure, evec: &[[f64; 3]]| -> Result<CoordStructure, Panic> {

        let (structure, sc_token) = supercell::diagonal(settings.supercell_relax, structure);
        let evec = sc_token.replicate(evec);
        let mut lmp = Lammps::new_carbon(structure.clone())?;

        let from_structure = structure;
        let direction = &evec[..];
        let from_pos = from_structure.to_carts();
        let alpha = ::sp2_minimize::exact_ls::<LmpError, _>(0.0, 1e-4, |alpha| {
            let V(pos) = v(from_pos.flat()) + alpha * v(direction.flat());
            let (_, gradient) = lammps_flat_diff_fn(&mut lmp)(&pos[..])?;
            let slope = vdot(&gradient[..], direction.flat());
            Ok(::sp2_minimize::exact_ls::Slope(slope))
        })??.alpha;
        let V(pos) = v(from_pos.flat()) + alpha * v(direction.flat());
        let structure = from_structure.with_coords(Coords::Carts(pos.nest().to_vec()));

        Ok(sc_token.deconstruct(1e-5, structure)?)
    };

    let original = poscar::load_carbon(File::open(input)?)?;
    ::std::fs::create_dir(&outdir)?;
    {
        // dumb/lazy solution to ensuring all output files go in the dir
        let cwd_guard = push_dir(outdir)?;

        let mut iter_count = 0;
        let mut from_structure = original;
        let (structure, eval, evec) = loop {
            let structure = relax(from_structure)?;
            let (eval, evec) = diagonalize(structure.clone())?;

            println!("============================");
            println!("Finished relaxation # {}", iter_count + 1);
            println!("Eigenvalues: (cm-1)");
            for &x in &eval {
                println!(" - {:?}", x);
            }

            // FIXME: this is dumb and there could be cases where it never succeeds,
            //        which arguably we shouldn't care about anyways since at that
            //        point we're probably looking at an acoustic frequency anyways
            if eval[0] > -settings.neg_frequency_threshold.abs() {
                println!("============================");
                break (structure, eval, evec);
            }
            println!();
            println!("!! Unsatisfied with frequency:  {:e}", eval[0]);
            println!("!! Optimizing along this band!");
            println!();

            let structure = minimize_evec(structure, &evec[0])?;

            from_structure = structure;

            println!("============================");

            iter_count += 1;
        }; // (structure, eval, evec)

        {
            let mut f = File::create("eigenvalues.final")?;
            for &x in &eval {
                writeln!(f, "{:?}", x)?;
            }
        }

        poscar::dump_carbon(File::create("./final.vasp")?, "", &structure)?;

        cwd_guard.pop()?;
    }

    Ok(())
}

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
        ::std::env::set_current_dir(path);
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
            if let Err(e) = ::std::env::set_current_dir(&self.0.take().unwrap()) {
                // uh oh.
                panic!("automatic popdir failed: {}", e);
            }
        }
    }
}
