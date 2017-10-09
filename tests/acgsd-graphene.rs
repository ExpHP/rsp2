extern crate sp2_lammps_wrap;
extern crate sp2_minimize;
extern crate sp2_array_utils;
extern crate sp2_slice_of_array;
extern crate sp2_structure;
extern crate sp2_structure_io;
extern crate sp2_slice_math;
#[macro_use] extern crate sp2_util_macros;

extern crate rand;
extern crate env_logger;
#[macro_use] extern crate serde_json;

use ::rand::random;
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

//#[test]
fn perturbed_graphene() {
    use ::sp2_structure::{CoordStructure, Lattice, Coords, supercell};
    init_logger();

    let a = 2.46;
    let r3 = 3f64.sqrt();
    let unit_structure = CoordStructure::new_coords(
        Lattice::new([
            [     a,      0.0,  0.0],
            [-0.5*a, 0.5*r3*a,  0.0],
            [   0.0,      0.0, 10.0],
        ]),
        Coords::Fracs(vec![
            [0.0, 0.0, 0.0],
            [2./3., 1./3., 0.0],
        ]),
    );
    let (superstructure, _) = supercell::diagonal((25,25,1), unit_structure);

    let correct = superstructure.to_carts();
    let input = {
        use rand::{Rng, SeedableRng, StdRng};

        let mut rng: StdRng = SeedableRng::from_seed(&[1, 2, 3, 4][..]);
        let mut coords = correct.clone();
        for x in coords.flat_mut() {
            *x += rng.gen::<f64>() * 1e-4;
        }
        coords
    };

    let mut lmp = sp2_lammps_wrap::Lammps::new_carbon(
        &superstructure.lattice().matrix(),
        &input,
    ).unwrap();

    let mut relaxed = ::sp2_minimize::acgsd(
        &from_json!({"stop-condition": {"grad-rms": 1e-5}}),
        input.flat(),
        move |pos: &[f64]| {
            let pos = pos.nest();
            let (value, grad) = lmp.compute(pos)?;
            Ok::<_, sp2_lammps_wrap::Error>((value, grad.flat().to_vec()))
        },
    ).unwrap().position;
    let mut input = input;
    remove_mean_shift(&mut input, &correct);
    remove_mean_shift(relaxed.nest_mut(), &correct);

    let distance_to_correct = vnorm(&(v(&relaxed) - v(correct.flat())));
    let distance_to_input = vnorm(&(v(&relaxed) - v(input.flat())));
    assert!(distance_to_correct < distance_to_input);
}

pub enum Panic {}
impl<E: ::std::fmt::Debug> From<E> for Panic {
    fn from(e: E) -> Panic { Err::<(),_>(e).unwrap(); unreachable!() }
}
