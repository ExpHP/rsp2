#![deny(unused_must_use)]
#![deny(unused_variables)]

extern crate rsp2_lammps_wrap;
extern crate rsp2_minimize;
extern crate rsp2_array_utils;
extern crate rsp2_structure;
extern crate rsp2_structure_io;
extern crate rsp2_slice_math;
#[macro_use] extern crate rsp2_util_macros;

extern crate rand;
extern crate env_logger;
extern crate slice_of_array;
#[macro_use] extern crate serde_json;

use ::rsp2_array_utils::vec_from_fn;
use ::slice_of_array::prelude::*;
use ::rsp2_slice_math::{v,vnorm};
use ::rsp2_lammps_wrap::{Lammps, Error as LmpError};

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

fn lammps_flat_diff_fn<'a>(lmp: &'a mut Lammps)
-> Box<FnMut(&[f64]) -> Result<(f64, Vec<f64>), LmpError> + 'a>
{
    Box::new(move |pos| {
        lmp.set_carts(pos.nest())?;
        lmp.compute().map(|(v,g)| (v, g.flat().to_vec()))
    })
}

// FIXME: slow
//#[test]
fn perturbed_graphene() {
    use ::rsp2_structure::{CoordStructure, Lattice, Coords, supercell};
    init_logger();

    let a = 2.46;
    let r3 = 3f64.sqrt();
    let unit_structure = CoordStructure::new_coords(
        Lattice::new(&[
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

    // don't multithread in tests
    let mut lmp = rsp2_lammps_wrap::Builder::new().threaded(false)
        .initialize_carbon(superstructure).unwrap();

    let mut relaxed = ::rsp2_minimize::acgsd(
        &from_json!({"stop-condition": {"grad-rms": 1e-5}}),
        input.flat(),
        &mut *lammps_flat_diff_fn(&mut lmp),
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
