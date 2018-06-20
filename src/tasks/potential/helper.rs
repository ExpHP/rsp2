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

//! Combinators and other helper implementations of PotentialBuilder and friends.

use super::{DynCloneDetail, PotentialBuilder, DiffFn, DispFn};
use ::FailResult;
use ::rsp2_structure::{Coords, CoordsKind, Lattice};
use ::rsp2_array_types::{V3};
use ::std::collections::BTreeMap;

/// A sum of two PotentialBuilders or DiffFns.
#[derive(Debug, Clone)]
pub struct Sum<A, B>(pub A, pub B);

impl<M, A, B> PotentialBuilder<M> for Sum<A, B>
where
    M: Clone + 'static,
    A: Clone + PotentialBuilder<M>,
    B: Clone + PotentialBuilder<M>,
{
    fn threaded(&self, threaded: bool) -> Box<PotentialBuilder<M>>
    { Box::new(Sum(self.0.threaded(threaded), self.1.threaded(threaded))) }

    fn initialize_diff_fn(&self, coords: &Coords, meta: M) -> FailResult<Box<DiffFn<M>>>
    {
        let a_diff_fn = self.0.initialize_diff_fn(coords, meta.clone())?;
        let b_diff_fn = self.1.initialize_diff_fn(coords, meta.clone())?;
        Ok(Box::new(Sum(a_diff_fn, b_diff_fn)))
    }

    fn initialize_disp_fn(&self, _: &Coords, _: M) -> FailResult<Box<DispFn>>
    { unimplemented!("should use DispFns of sub potentials, rather than DefaultDispFn") }
}

impl_dyn_clone_detail!{
    impl[
        M: Clone + 'static,
        A: Clone + PotentialBuilder<M>,
        B: Clone + PotentialBuilder<M>,
    ] DynCloneDetail<M> for Sum<A, B> { ... }
}

impl<M, A, B> DiffFn<M> for Sum<A, B>
where
    M: Clone,
    A: DiffFn<M>,
    B: DiffFn<M>,
{
    fn compute(&mut self, coords: &Coords, meta: M) -> FailResult<(f64, Vec<V3>)> {
        let (a_value, a_grad) = self.0.compute(coords, meta.clone())?;
        let (b_value, b_grad) = self.1.compute(coords, meta.clone())?;
        let value = a_value + b_value;

        let mut grad = a_grad;
        for (out_vec, b_vec) in zip_eq!(&mut grad, b_grad) {
            *out_vec += b_vec;
        }
        Ok((value, grad))
    }
}

//--------------------------------

pub struct DefaultDispFn<Meta> {
    // this is carts instead of Coords for the same reason that `ensure_only_carts` exists;
    // see that function
    equilibrium_carts: Vec<V3>,
    lattice: Lattice,
    equilibrium_force: Vec<V3>,
    meta: Meta,
    diff_fn: Box<DiffFn<Meta>>,
}

impl<Meta> DefaultDispFn<Meta>
where Meta: Clone + 'static,
{
    pub fn initialize(equilibrium_coords: &Coords, meta: Meta, pot: &PotentialBuilder<Meta>) -> FailResult<Self>
    {Ok({
        let lattice = equilibrium_coords.lattice().clone();
        let equilibrium_carts = equilibrium_coords.to_carts();

        let equilibrium_coords = Coords::new(lattice.clone(), CoordsKind::Carts(equilibrium_carts.clone()));

        let mut diff_fn = pot.initialize_diff_fn(&equilibrium_coords, meta.clone())?;
        let equilibrium_force = diff_fn.compute_force(&equilibrium_coords, meta.clone())?;

        DefaultDispFn { lattice, equilibrium_carts, equilibrium_force, meta, diff_fn }
    })}
}

impl<Meta> DispFn for DefaultDispFn<Meta>
where Meta: Clone,
{
    fn compute_dense_force(&mut self, disp: (usize, V3)) -> FailResult<Vec<V3>>
    {Ok({
        let mut carts = self.equilibrium_carts.to_vec();
        carts[disp.0] += disp.1;

        let coords = CoordsKind::Carts(carts);
        let coords = Coords::new(self.lattice.clone(), coords);
        self.diff_fn.compute_force(&coords, self.meta.clone())?
    })}

    fn compute_sparse_force_delta(&mut self, disp: (usize, V3)) -> FailResult<BTreeMap<usize, V3>>
    {
        let orig_force = self.equilibrium_force.clone();
        sparse_force_from_dense_deterministic(self, &orig_force, disp)
    }
}

/// Implements sparse force sets in terms of dense force sets.
///
/// Assumes `compute_dense_force` produces values that only differ from the
/// original forces in a neighborhood of the displacement. This can be true if
/// the potential...
///
///  * is deterministic,
///  * implements a cutoff radius, and
///  * does not recklessly adjust coordinates
///
/// ...so that with the help of the "ensure_only_carts", even this
/// exact equality check should be effective at sparsifying the data.
///
/// Which is good, because it's tough to define an approximate scale for comparison
/// here, as the forces are the end-result of catastrophic cancellations.
pub fn sparse_force_from_dense_deterministic(
    disp_fn: &mut DispFn,
    original_force: &[V3],
    disp: (usize, V3),
) -> FailResult<BTreeMap<usize, V3>> {
    let displaced_force = disp_fn.compute_dense_force(disp)?;

    let diffs = {
        zip_eq!(original_force, displaced_force).enumerate()
            .map(|(atom, (old, new))| (atom, new - old))
            .filter(|&(_, v)| v != V3::zero())

        // this one is a closer approximation of phonopy, producing a dense matrix with
        // just the new forces (assuming the old ones are zero)
//                .map(|(atom, (_old, new))| (atom, new))
    };
    Ok(diffs.collect())
}
