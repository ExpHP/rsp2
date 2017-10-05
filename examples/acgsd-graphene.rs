extern crate sp2_lammps_wrap;
extern crate sp2_minimize;
extern crate sp2_structure;
extern crate sp2_structure_io;
extern crate sp2_array_utils;
extern crate sp2_slice_of_array;
extern crate sp2_slice_math;

extern crate rand;
extern crate env_logger;
#[macro_use] extern crate serde_json;

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
    use ::sp2_array_utils::prelude::*;
    use ::sp2_structure_io::{xyz, poscar};
    use ::sp2_structure::{supercell, Coords};
    use ::std::fs::File;
    init_logger();

    let mut names = vec![
        String::from("aba-007-a"),
        String::from("shift-01"),
    ];
    for i in 0..7 {
        names.push(format!("shift-0{}", i));
    }

    for name in names {
        let name = &name;

        let original = poscar::load_carbon(File::open(format!("./examples/data/{}.vasp", name))?)?;

        let relaxed = {
            let (supercell, sc_token) = supercell::diagonal((4,4,1), original);

            // FIXME confusing for Lammps::new_carbon to take initial position
            let mut lmp = sp2_lammps_wrap::Lammps::new_carbon(&supercell.lattice().matrix(), &supercell.to_carts())?;
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
            sc_token.deconstruct(1e-10, supercell)?
        };

        poscar::dump_carbon(
            File::create(format!("./out-{}.vasp", name))?,
            name,
            &relaxed,
        )?;
    }

    Ok(())
}
