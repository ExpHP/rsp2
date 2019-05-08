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

use super::{DynCloneDetail, PotentialBuilder, DiffFn, DispFn, BondDiffFn, BondGrad};
use crate::FailResult;
use rsp2_structure::{Coords, CoordsKind, Lattice};
use rsp2_array_types::{V3};
use std::collections::BTreeMap;

/// A sum of two PotentialBuilders or DiffFns.
#[derive(Debug, Clone)]
pub struct Sum<A, B>(pub A, pub B);

impl<M, A, B> PotentialBuilder<M> for Sum<A, B>
where
    M: Clone + 'static,
    A: Clone + PotentialBuilder<M>,
    B: Clone + PotentialBuilder<M>,
{
    fn parallel(&self, parallel: bool) -> Box<dyn PotentialBuilder<M>>
    { Box::new(Sum(self.0.parallel(parallel), self.1.parallel(parallel))) }

    fn allow_blocking(&self, allow: bool) -> Box<dyn PotentialBuilder<M>>
    { Box::new(Sum(self.0.allow_blocking(allow), self.1.allow_blocking(allow))) }

    fn initialize_diff_fn(&self, coords: &Coords, meta: M) -> FailResult<Box<dyn DiffFn<M>>>
    {
        let a_diff_fn = self.0.initialize_diff_fn(coords, meta.clone())?;
        let b_diff_fn = self.1.initialize_diff_fn(coords, meta.clone())?;
        Ok(Box::new(Sum(a_diff_fn, b_diff_fn)))
    }

    fn initialize_bond_diff_fn(&self, coords: &Coords, meta: M) -> FailResult<Option<Box<dyn BondDiffFn<M>>>>
    {
        let a_diff_fn = match self.0.initialize_bond_diff_fn(coords, meta.clone())? {
            Some(x) => x,
            None => return Ok(None),
        };
        let b_diff_fn = match self.1.initialize_bond_diff_fn(coords, meta.clone())? {
            Some(x) => x,
            None => return Ok(None),
        };
        Ok(Some(Box::new(Sum(a_diff_fn, b_diff_fn))))
    }

    fn initialize_disp_fn(&self, coords: &Coords, meta: M) -> FailResult<Box<dyn DispFn>>
    {
        let a_disp_fn = self.0.initialize_disp_fn(coords, meta.clone())?;
        let b_disp_fn = self.1.initialize_disp_fn(coords, meta.clone())?;
        Ok(Box::new(Sum(a_disp_fn, b_disp_fn)))
    }

    fn _eco_mode(&self, cont: &mut dyn FnMut())
    { (self.0)._eco_mode(&mut || (self.1)._eco_mode(cont)) }
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

impl<M, A, B> BondDiffFn<M> for Sum<A, B>
where
    M: Clone,
    A: BondDiffFn<M>,
    B: BondDiffFn<M>,
{
    fn compute(&mut self, coords: &Coords, meta: M) -> FailResult<(f64, Vec<BondGrad>)> {
        let (a_value, a_grad) = self.0.compute(coords, meta.clone())?;
        let (b_value, b_grad) = self.1.compute(coords, meta.clone())?;
        let value = a_value + b_value;
        let mut grad = a_grad;
        grad.extend(b_grad);
        Ok((value, grad))
    }
}

impl<A, B> DispFn for Sum<A, B>
where
    A: DispFn,
    B: DispFn,
{
    fn compute_dense_force(&mut self, disp: (usize, V3)) -> FailResult<Vec<V3>>
    {
        let a_force = self.0.compute_dense_force(disp)?;
        let b_force = self.1.compute_dense_force(disp)?;
        Ok(zip_eq!(a_force, b_force).map(|(a, b)| a + b).collect())
    }

    fn compute_sparse_force_delta(&mut self, disp: (usize, V3)) -> FailResult<BTreeMap<usize, V3>>
    {
        let mut out = self.0.compute_sparse_force_delta(disp)?;
        for (key, value) in self.1.compute_sparse_force_delta(disp)? {
            *out.entry(key).or_insert(V3::zero()) += value;
        }
        Ok(out)
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
    diff_fn: Box<dyn DiffFn<Meta>>,
}

impl<Meta> DefaultDispFn<Meta>
where Meta: Clone + 'static,
{
    pub fn initialize(equilibrium_coords: &Coords, meta: Meta, pot: &dyn PotentialBuilder<Meta>) -> FailResult<Self>
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
    disp_fn: &mut dyn DispFn,
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

//--------------------------------

/// Provides a default DiffFn that can be used by a potential that defines a BondDiffFn.
pub struct DiffFnFromBondDiffFn<Meta>(Box<dyn BondDiffFn<Meta>>);

impl<Meta> DiffFnFromBondDiffFn<Meta>
where Meta: Clone + 'static,
{
    pub fn new(f: Box<dyn BondDiffFn<Meta>>) -> Self {
        DiffFnFromBondDiffFn(f)
    }
}


impl<Meta> DiffFn<Meta> for DiffFnFromBondDiffFn<Meta>
where Meta: Clone,
{
    fn compute(&mut self, coords: &Coords, meta: Meta) -> FailResult<(f64, Vec<V3>)>
    {Ok({
        let (potential, bond_grad) = self.0.compute(coords, meta)?;
        let mut out_grad = vec![V3::zero(); coords.len()];
        for BondGrad { plus_site, minus_site, grad, cart_vector: _ } in bond_grad {
            out_grad[plus_site] += grad;
            out_grad[minus_site] -= grad;
        }
        (potential, out_grad)
    })}
}
