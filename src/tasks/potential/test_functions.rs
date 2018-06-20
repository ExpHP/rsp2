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

//! Dummy potentials for testing purposes

use super::{DynCloneDetail, PotentialBuilder, DiffFn, DispFn};
use ::FailResult;
use ::rsp2_structure::{Coords, CoordsKind};
use ::rsp2_array_types::{V3};

/// The test Potential `V = 0`.
#[derive(Debug, Clone)]
pub struct Zero;

impl<Meta: Clone + 'static> PotentialBuilder<Meta> for Zero {
    fn initialize_diff_fn<'a>(&self, _: &Coords, _: Meta) -> FailResult<Box<DiffFn<Meta>>>
    {
        struct Diff;
        impl<M> DiffFn<M> for Diff {
            fn compute(&mut self, coords: &Coords, _: M) -> FailResult<(f64, Vec<V3>)> {
                Ok((0.0, vec![V3([0.0; 3]); coords.num_atoms()]))
            }
        }
        Ok(Box::new(Diff) as Box<_>)
    }

    fn initialize_disp_fn(&self, coords: &Coords, meta: Meta) -> FailResult<Box<DispFn>>
    { self._default_initialize_disp_fn(coords, meta) }
}

impl_dyn_clone_detail!{
    impl[M: Clone + 'static] DynCloneDetail<M> for Zero { ... }
}

// ---------------

/// A test DiffFn that moves atoms to fixed positions.
#[derive(Debug, Clone)]
pub struct ConvergeTowards {
    target: Coords,
}

impl ConvergeTowards {
    pub fn new(coords: Coords) -> Self
    { ConvergeTowards { target: coords.clone() } }
}

/// ConvergeTowards can also serve as its own PotentialBuilder.
impl<Meta: Clone + 'static> PotentialBuilder<Meta> for ConvergeTowards {
    fn initialize_diff_fn(&self, _: &Coords, _: Meta) -> FailResult<Box<DiffFn<Meta>>>
    { Ok(Box::new(self.clone()) as Box<_>) }

    fn initialize_disp_fn(&self, coords: &Coords, meta: Meta) -> FailResult<Box<DispFn>>
    { self._default_initialize_disp_fn(coords, meta) }
}

impl_dyn_clone_detail!{
    impl[Meta: Clone + 'static] DynCloneDetail<Meta> for ConvergeTowards { ... }
}

impl<M> DiffFn<M> for ConvergeTowards {
    fn compute(&mut self, coords: &Coords, meta: M) -> FailResult<(f64, Vec<V3>)> {
        (&*self).compute(coords, meta)
    }
}

// ConvergeTowards does not get mutated
impl<'a, M> DiffFn<M> for &'a ConvergeTowards {
    fn compute(&mut self, input_coords: &Coords, _: M) -> FailResult<(f64, Vec<V3>)> {
        assert_eq!(input_coords.num_atoms(), self.target.num_atoms());
        assert_close!(abs=1e-8, input_coords.lattice(), self.target.lattice());

        // Each position in `structure` experiences a force generated only by the
        // corresponding position in `target`.

        // In fractional coords, the potential is:
        //
        //    Sum_a  Product_k { 1.0 - cos[(x[a,k] - xtarg[a,k]) 2 pi] }
        //    (atom)  (axis)
        //
        // This is a periodic function with a minimum at each image of the target point,
        // and derivatives that are continuous everywhere.
        use ::std::f64::consts::PI;

        let cur_fracs = input_coords.to_fracs();
        let target_fracs = self.target.to_fracs();
        let args_by_coord = {
            zip_eq!(&cur_fracs, target_fracs)
                .map(|(c, t)| V3::from_fn(|k| (c[k] - t[k]) * 2.0 * PI))
                .collect::<Vec<_>>()
        };
        let d_arg_d_coord = 2.0 * PI;

        let parts_by_coord = {
            args_by_coord.iter()
                .map(|&v| v.map(|arg| 2.0 - f64::cos(arg)))
                .collect::<Vec<_>>()
        };
        let derivs_by_coord = {
            args_by_coord.iter()
                .map(|&v| v.map(|arg| d_arg_d_coord * f64::sin(arg)))
                .collect::<Vec<_>>()
        };

        let value = {
            parts_by_coord.iter().map(|v| v.iter().product::<f64>()).sum()
        };
        let frac_grad = {
            zip_eq!(&parts_by_coord, &derivs_by_coord)
                .map(|(parts, derivs)| { // by atom
                    V3::from_fn(|k0| {
                        (0..3).map(|k| {
                            if k == k0 { derivs[k] }
                                else { parts[k] }
                        }).product()
                    })
                })
        };

        // Partial derivatives transform like reciprocal coords.
        let recip = input_coords.lattice().reciprocal();
        let cart_grad = frac_grad.map(|v| v * &recip).collect::<Vec<_>>();
        Ok((value, cart_grad))
    }
}

// ---------------

/// A test Potential that creates a chain along the first lattice vector.
#[derive(Debug, Clone)]
pub struct Chainify;

impl<Meta: Clone + 'static> PotentialBuilder<Meta> for Chainify {
    fn initialize_diff_fn(&self, initial_coords: &Coords, _: Meta) -> FailResult<Box<DiffFn<Meta>>>
    {
        let na = initial_coords.num_atoms();
        let target = Coords::new(
            initial_coords.lattice().clone(),
            CoordsKind::Fracs({
                (0..na)
                    .map(|i| V3([i as f64 / na as f64, 0.5, 0.5]))
                    .collect()
            }),
        );
        Ok(Box::new(ConvergeTowards::new(target)) as Box<_>)
    }

    fn initialize_disp_fn(&self, coords: &Coords, meta: Meta) -> FailResult<Box<DispFn>>
    { self._default_initialize_disp_fn(coords, meta) }
}

impl_dyn_clone_detail!{
    impl[Meta: Clone + 'static] DynCloneDetail<Meta> for Chainify { ... }
}

// ---------------

#[cfg(test)]
#[deny(unused)]
mod tests {
    use super::*;
    use ::FailOk;
    use rsp2_structure::{Lattice, CoordsKind};
    use ::rsp2_array_types::Envee;

    #[test]
    fn converge_towards() {
        ::ui::logging::init_test_logger();

        let lattice = Lattice::from(&[
            // chosen arbitrarily
            [ 2.0,  3.0, 4.0],
            [-1.0,  7.0, 8.0],
            [-3.0, -4.0, 7.0],
        ]);

        let target_coords = CoordsKind::Fracs(vec![
            [ 0.1, 0.7, 3.3],
            [ 1.2, 1.5, 4.3],
            [ 0.1, 1.2, 7.8],
            [-0.6, 0.1, 0.8],
        ].envee());
        let start_coords = CoordsKind::Fracs(vec![
            [ 1.2, 1.5, 4.3],
            [ 0.1, 0.7, 3.3],
            [-0.6, 0.1, 0.4],
            [ 0.1, 1.2, 7.8],
        ].envee());
        let expected_fracs = vec![
            [ 1.1, 1.7, 4.3],
            [ 0.2, 0.5, 3.3],
            [-0.9, 0.2, 0.8],
            [ 0.4, 1.1, 7.8],
        ];

        let target = Coords::new(lattice.clone(), target_coords.clone());
        let start = Coords::new(lattice.clone(), start_coords.clone());

        let diff_fn = ConvergeTowards::new(target);
        let cg_settings = &from_json!{{
                "stop-condition": {"grad-max": 1e-10},
                "alpha-guess-first": 0.1,
            }};

        let mut flat_diff_fn = diff_fn.initialize_flat_diff_fn(&start, ()).unwrap();
        let flat_diff_fn = &mut *flat_diff_fn;

        let data = ::rsp2_minimize::acgsd(
            cg_settings,
            start.to_carts().flat(),
            &mut *flat_diff_fn,
        ).unwrap();
        println!("DerpG: {:?}", flat_diff_fn(start.to_carts().flat()).unwrap().1);
        println!("NumrG: {:?}", ::rsp2_minimize::numerical::gradient(
            1e-7, None,
            start.to_carts().flat(),
            |p| FailOk(flat_diff_fn(p)?.0),
        ).unwrap());
        println!(" Grad: {:?}", data.gradient);
        println!("Numer: {:?}", ::rsp2_minimize::numerical::gradient(
            1e-7, None,
            &data.position,
            |p| FailOk(flat_diff_fn(p)?.0),
        ).unwrap());
        let final_carts = data.position.nest::<V3>().to_vec();
        let final_fracs = CoordsKind::Carts(final_carts).into_fracs(&lattice);
        assert_close!(final_fracs.unvee(), expected_fracs);
    }
}
