extern crate sp2_lammps_wrap;
extern crate sp2_minimize;
extern crate sp2_structure;
extern crate sp2_structure_io;
extern crate sp2_phonopy_io;
extern crate sp2_array_utils;
extern crate sp2_slice_of_array;
extern crate sp2_slice_math;
extern crate sp2_tempdir;

extern crate rand;
extern crate env_logger;
#[macro_use] extern crate serde_json;

const THZ_TO_WAVENUMBER: f64 = 33.35641;

macro_rules! collect {
    ($($e:tt)*) => { vec![$($e)*].into_iter().collect() };
}

use ::rand::random;
use ::serde_json::from_value;
use ::sp2_array_utils::vec_from_fn;
use ::sp2_slice_of_array::prelude::*;
use ::sp2_slice_math::{v,vnorm};

fn init_logger() {
    let _ = ::env_logger::init();
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

pub enum Panic {}
impl<E: ::std::fmt::Debug> From<E> for Panic {
    fn from(e: E) -> Panic { Err::<(),_>(e).unwrap(); unreachable!() }
}

fn main() { let _ = _main(); }
fn _main() -> Result<(), Panic> {
    use ::std::io::prelude::*;
    use ::sp2_phonopy_io as p;
    use ::sp2_array_utils::prelude::*;
    use ::sp2_structure_io::{xyz, poscar};
    use ::sp2_structure::{supercell, Coords, CoordStructure};
    use ::sp2_slice_math::{v, V};
    use ::sp2_lammps_wrap::Lammps;
    use ::std::fs::File;
    init_logger();

    let sc_size_relax = 4;
    let sc_size_phonopy = 4;

    let mut names = vec![
        String::from("aba-007-a"),
        String::from("shift-01"),
    ];
    for i in 0..7 {
        names.push(format!("shift-0{}", i));
    }

    let relax = |structure| -> Result<CoordStructure, Panic> {
        let (supercell, sc_token) = supercell::diagonal((sc_size_relax, sc_size_relax, 1), structure);

        // FIXME confusing for Lammps::new_carbon to take initial position
        let mut lmp = Lammps::new_carbon(&supercell.lattice().matrix(), &supercell.to_carts())?;
        println!("NUM ATOM {}", supercell.num_atoms());
        let relaxed_flat = ::sp2_minimize::acgsd(
            &from_value(json!({"stop-condition": {"grad-rms": 1e-8}}))?,
            &supercell.to_carts().flat(),
            move |pos: &[f64]| {
                let pos = pos.nest();
                let (value, grad) = lmp.compute(pos)?;
                Ok::<_, sp2_lammps_wrap::Error>((value, grad.flat().to_vec()))
            },
        )?.position;

        let supercell = supercell.with_coords(Coords::Carts(relaxed_flat.nest().to_vec()));
        Ok(sc_token.deconstruct(1e-10, supercell)?)
    };

    let diagonalize = |structure| -> Result<_, Panic> {
        let conf = collect![
            (format!("DIM"), format!("{0} {0} 1", sc_size_phonopy)),
            (format!("HDF5"), format!(".TRUE.")),
        ];

        let (superstructure, displacements, disp_token)
            = p::cmd::phonopy_displacements_carbon(&conf, structure)?;

        let mut lmp = Lammps::new_carbon(&superstructure.lattice().matrix(), &superstructure.to_carts())?;
        println!();
        let mut i = 0;
        let force_sets = p::force_sets::compute_from_grad(
            superstructure.clone(), // FIXME only a borrow is needed later, and only for natom (dumb)
            &displacements,
            |s| {
                i += 1;
                print!("\rdisp {} of {}", i, displacements.len());
                ::std::io::stdout().flush().unwrap();
                Ok::<_, sp2_lammps_wrap::Error>(lmp.compute(&s.to_carts())?.1)
            }
        )?;
        println!();

        let (eval, evec) = p::cmd::phonopy_gamma_eigensystem(&conf, force_sets, &disp_token)?;
        let V(eval) = THZ_TO_WAVENUMBER * v(eval);
        Ok((eval, evec))
    };

    for name in names {
        let name = &name;

        let original = poscar::load_carbon(File::open(format!("./examples/data/{}.vasp", name))?)?;

        loop {
            let relaxed = relax(original)?;
            let (eval, evec) = diagonalize(relaxed)?;
            for x in eval {
                eprintln!("{:?}", x);
            }
            break; // XXX
        }
        // poscar::dump_carbon(
        //     File::create(format!("./out-{}.vasp", name))?,
        //     name,
        //     &finished,
        // )?;
    }

    Ok(())
}
