/* ********************************************************************** **
**  This file is part of rsp2.                                            **
**                                                                        **
**  rsp2 is free software: you can redistribute it and/or modify it under **
**  the terms of the GNU General Public License as published by the Free  **
**  Software Foundation, either version 3 of the License, or (at your     **
**  option) any later version.                                            **
**                                                                        **
**      http://www.gnu.org/licenses/                                      **
**                                                                        **
** Do note that, while the whole of rsp2 is licensed under the GPL, many  **
** parts of it are licensed under more permissive terms.                  **
** ********************************************************************** */

//! Wyatt Gibbons' external potential. (2022)

use super::{DynCloneDetail, PotentialBuilder, DiffFn, DispFn, BondDiffFn};
use crate::{FailResult};
#[allow(unused)] // rustc bug
use crate::meta::{self, prelude::*};

use rsp2_structure::{Coords, Lattice};
use rsp2_tasks_config as cfg;
use rsp2_array_types::{V3};

use core::f64::consts::PI;

pub use gibbons_product::GibbonsProduct;
mod gibbons_product {
    use super::*;

    /// A `DiffFn` for a product of three cosines, used by Wyatt Gibbons.
    #[derive(Debug, Clone)]
    pub struct GibbonsProduct {
        cfg: cfg::PotentialGibbonsProduct,
    }

    impl GibbonsProduct {
        pub fn new(cfg: &cfg::PotentialGibbonsProduct) -> FailResult<Self> {
            if cfg.num_maxima <= 0 {
                bail!("num-maxima must be >= 1");
            }
            Ok(GibbonsProduct { cfg: cfg.clone() })
        }
    }

    // It serves as its own `PotentialBuilder`.
    impl<Meta: Clone + 'static> PotentialBuilder<Meta> for GibbonsProduct {
        fn initialize_diff_fn(&self, _: &Coords, _: Meta) -> FailResult<Box<dyn DiffFn<Meta>>>
        { Ok(Box::new(self.clone()) as Box<_>) }

        fn initialize_bond_diff_fn(&self, _: &Coords, _: Meta) -> FailResult<Option<Box<dyn BondDiffFn<Meta>>>>
        { Ok(None) }

        fn initialize_disp_fn(&self, coords: &Coords, meta: Meta) -> FailResult<Box<dyn DispFn>>
        { self._default_initialize_disp_fn(coords, meta) }
    }

    impl_dyn_clone_detail!{
        impl[Meta: Clone + 'static] DynCloneDetail<Meta> for GibbonsProduct { ... }
    }

    impl<M> DiffFn<M> for GibbonsProduct {
        fn compute(&mut self, coords: &Coords, meta: M) -> FailResult<(f64, Vec<V3>)> {
            (&*self).compute(coords, meta)
        }
    }

    impl<'a, M> DiffFn<M> for &'a GibbonsProduct {
        fn compute(&mut self, coords: &Coords, _: M) -> FailResult<(f64, Vec<V3>)> {
            let covectors = get_cosine_covectors(self.cfg.num_maxima, coords.lattice())?;

            let mut value = 0.0;
            let mut cart_grad = Vec::with_capacity(coords.len());
            for cart in coords.to_carts() {
                let args = V3::from_fn(|i| covectors[i].dot(&cart)).0;
                let cosines = V3::from_fn(|i| f64::cos(args[i])).0;

                value += self.cfg.amplitude * args.iter().map(|&arg| f64::cos(arg)).product::<f64>();

                cart_grad.push((0..3).map(|i| {
                    let cos_d_arg = -f64::sin(args[i]);
                    let arg_d_cart = covectors[i];
                    let term_d_cos = self.cfg.amplitude * cosines[(i + 1) % 3] * cosines[(i + 2) % 3];
                    term_d_cos * cos_d_arg * arg_d_cart
                }).sum());
            }
            Ok((value, cart_grad))
        }
    }
}

/// Get the three vectors that, when dotted with a position, give the argument for one of the
/// cosines in the potential.
fn get_cosine_covectors(num_maxima: u32, lattice: &Lattice) -> FailResult<[V3; 3]> {
    let [a, b] = find_in_plane_vectors(lattice.vectors())?;

    // Some of these restrictions can be overcome by converting into an equivalent primitive cell
    // but I am too lazy; just require a 120 degree angle for now.
    if f64::abs(a.norm() - b.norm()) > 1e-5 * f64::min(a.norm(), b.norm()) {
        bail!("non-equal vector lengths. {}", GIBBONS_CELL_REQUIREMENT_HINT);
    }
    if f64::to_degrees(V3::angle_to(a, b)) - 120.0 > 1e-3 {
        bail!("cell angle is not 120 degrees. {}", GIBBONS_CELL_REQUIREMENT_HINT)
    }

    let make_covector = |v: &V3| v.unit() * (2.0 * PI / v.norm()) * num_maxima as f64;
    Ok([
        make_covector(a),
        make_covector(b),
        make_covector(&(-a-b)),
    ])
}

fn find_in_plane_vectors(lattice_vectors: &[V3; 3]) -> FailResult<[&V3; 2]> {
    assert!(lattice_vectors.iter().all(|v| v.norm() != 0.0), "degenerate lattice!?");

    for normal_i in 0..3 {
        let normal = &lattice_vectors[normal_i];
        let a = &lattice_vectors[(normal_i + 1) % 3];
        let b = &lattice_vectors[(normal_i + 2) % 3];

        let similarity = |u, v| f64::abs(V3::dot(u, v)) / (u.norm() * v.norm());
        let is_normal = |u, v| similarity(u, v) < 1e-4;
        if is_normal(normal, a) && is_normal(normal, b) && !is_normal(a, b) {
            return Ok([a, b]);
        }
    }
    bail!("could not identify normal vector. {}", GIBBONS_CELL_REQUIREMENT_HINT);
}

const GIBBONS_CELL_REQUIREMENT_HINT: &'static str = "\
gibbons potentials require two cell vectors to be \
of equal length and at 120 degrees to each other, with the third perpendicular.\
";


#[cfg(test)]
#[deny(unused)]
mod tests {
    use super::*;
    use rand::{Rand, Rng};
    use rsp2_structure::{Lattice, CoordsKind};
    use rsp2_array_types::{Envee, M3, M33};
    use rsp2_soa_ops::Perm;
    use slice_of_array::prelude::*;
    use rsp2_minimize::numerical;
    use crate::util::uniform;

    #[derive(PartialEq)]
    enum PlaneAxes { Random, FirstTwo }

    fn random_accepted_lattice(plane_axes: PlaneAxes) -> Lattice {
        let mut lattice = Lattice::from(&[
            [ 1.0, 0.0, 0.0],
            [-0.5, f64::sqrt(3.0) * 0.5, 0.0],
            [ 0.0, 0.0, 1.0],
        ]);
        let a_scale = uniform(1.0, 50.0);
        let c_scale = uniform(1.0, 10.0);
        lattice = &M33::from_diag(V3([a_scale, a_scale, c_scale])) * &lattice;

        // permute the vectors
        if plane_axes == PlaneAxes::Random {
            let mut perm_matrix = M33::zero();
            for (r, c) in Perm::random(3).into_vec().into_iter().enumerate() {
                perm_matrix[r][c] = 1.0;
            }
            lattice = &perm_matrix * &lattice;
        }

        // rotate around x to spice things up
        let radians = uniform(0.0, 2.0 * PI);
        let (cos, sin) = (radians.cos(), radians.sin());
        let rot_matrix = M3([
            [1.0, 0.0, 0.0],
            [0.0, cos, -sin],
            [0.0, sin, cos],
        ].envee());
        lattice = &lattice * &rot_matrix.t();

        lattice
    }

    fn random_vec<T: Rand>(n: usize) -> Vec<T>
    { (0..n).map(|_| rand::random()).collect() }

    #[test]
    fn value() {
        crate::ui::logging::init_test_logger();

        for _ in 0..10 {
            let lattice = random_accepted_lattice(PlaneAxes::FirstTwo);

            let value_at_frac = |pot: &dyn PotentialBuilder<()>, frac: V3| {
                let coords = Coords::new(lattice.clone(), CoordsKind::Fracs(vec![frac]));
                let mut diff_fn = pot.initialize_diff_fn(&coords, ()).unwrap();
                diff_fn.compute(&coords, ()).unwrap().0
            };

            let cfg = from_json!{{
                "amplitude": uniform(0.1, 10.0),
                "num-maxima": 1,
            }};
            let pot = GibbonsProduct::new(&cfg).unwrap();

            let tol = 1e-10;
            assert_close!(rel=tol, value_at_frac(&pot, V3([0.0, 0.0, 0.0])), cfg.amplitude);
            assert_close!(rel=tol, value_at_frac(&pot, V3([1.0, 0.0, 0.0])), cfg.amplitude);
            assert_close!(rel=tol, value_at_frac(&pot, V3([5.0, 0.0, 0.0])), cfg.amplitude);
            assert_close!(rel=tol, value_at_frac(&pot, V3([0.0, 1.0, 0.0])), cfg.amplitude);
            assert_close!(rel=tol, value_at_frac(&pot, V3([-1.0, -1.0, 0.0])), cfg.amplitude);
            assert_close!(abs=tol, value_at_frac(&pot, V3([0.5, 0.0, 0.0])), 0.0);
            assert_close!(abs=tol, value_at_frac(&pot, V3([0.0, 0.5, 0.0])), 0.0);
            assert_close!(abs=tol, value_at_frac(&pot, V3([-0.5, -0.5, 0.0])), 0.0);

            let cfg = from_json!{{
                "amplitude": uniform(0.1, 10.0),
                "num-maxima": 2,
            }};
            let pot = GibbonsProduct::new(&cfg).unwrap();
            assert_close!(rel=tol, value_at_frac(&pot, V3([0.0, 0.0, 0.0])), cfg.amplitude);
            assert_close!(rel=tol, value_at_frac(&pot, V3([0.0, 0.5, 0.0])), cfg.amplitude);
            assert_close!(rel=tol, value_at_frac(&pot, V3([-0.5, -0.5, 0.0])), cfg.amplitude);
            assert_close!(abs=tol, value_at_frac(&pot, V3([0.0, 0.25, 0.0])), 0.0);
            assert_close!(abs=tol, value_at_frac(&pot, V3([-0.25, -0.25, 0.0])), 0.0);
        }
    }

    #[test]
    fn numerical_derivative() {
        crate::ui::logging::init_test_logger();

        let mut rng = rand::thread_rng();

        for _ in 0..10 {
            let lattice = random_accepted_lattice(PlaneAxes::Random);
            let fracs: Vec<V3> = random_vec::<[f64; 3]>(10).into_iter().map(|v| V3(v) * 4.0).collect();
            let coords = Coords::new(lattice.clone(), CoordsKind::Fracs(fracs));

            let cfg: cfg::PotentialGibbonsProduct = from_json!{{
                "amplitude": uniform(0.1, 10.0),
                "num-maxima": rng.gen_range(1, 4),
            }};
            let mut diff_fn = GibbonsProduct::new(&cfg).unwrap().initialize_diff_fn(&coords, ()).unwrap();

            let computed_grad = diff_fn.compute(&coords, ()).unwrap().1;
            let numerical_grad = &numerical::gradient(1e-7, None, coords.to_carts().flat(), |x| {
                let new_carts = CoordsKind::Carts(x.nest().to_vec());
                let new_coords = Coords::new(lattice.clone(), new_carts);
                diff_fn.compute(&new_coords, ()).unwrap().0
            })[..];
            let tol = 3e-6;
            assert_close!(rel=tol, abs=tol, computed_grad.flat(), numerical_grad);
        }
    }

    #[test]
    #[cfg(nope)]
    fn plot() {
        crate::ui::logging::init_test_logger();

        let lattice = Lattice::from(&[
            [ 1.0, 0.0, 0.0],
            [-0.5, f64::sqrt(3.0) * 0.5, 0.0],
            [ 0.0, 0.0, 1.0],
        ]);

        let value_at_frac = |pot: &dyn PotentialBuilder<()>, frac: V3| {
            let coords = Coords::new(lattice.clone(), CoordsKind::Fracs(vec![frac]));
            let mut diff_fn = pot.initialize_diff_fn(&coords, ()).unwrap();
            diff_fn.compute(&coords, ()).unwrap().0
        };

        let cfg = from_json!{{
            "amplitude": uniform(0.1, 10.0),
            "num-maxima": 1,
        }};
        let pot = GibbonsProduct::new(&cfg).unwrap();

        let xs = (0..100).map(|x| (x as f64 / 100.0) * 3.0).collect::<Vec<_>>();
        let ys = (0..100).map(|x| (x as f64 / 100.0) * 3.0).collect::<Vec<_>>();
        for &fx in &xs {
            for &fy in &ys {
                let frac = V3([fx, fy, 0.0]);
                let V3([x, y, _]) = frac * &lattice;
                println!("{x} {y} {}", value_at_frac(&pot, frac));
            }
        }
        panic!();
    }
}
