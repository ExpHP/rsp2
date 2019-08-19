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

use super::{DynCloneDetail, PotentialBuilder, DiffFn, DispFn, BondDiffFn, PairwiseDDiffFn, BondGrad};
use crate::FailResult;
use rsp2_structure::{Coords, CoordsKind, Lattice};
use rsp2_array_types::{V3, M33};
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

    fn initialize_pairwise_ddiff_fn(&self, coords: &Coords, meta: M) -> FailResult<Option<Box<dyn PairwiseDDiffFn<M>>>>
    {
        let a_ddiff_fn = match self.0.initialize_pairwise_ddiff_fn(coords, meta.clone())? {
            Some(x) => x,
            None => return Ok(None),
        };
        let b_ddiff_fn = match self.1.initialize_pairwise_ddiff_fn(coords, meta.clone())? {
            Some(x) => x,
            None => return Ok(None),
        };
        Ok(Some(Box::new(Sum(a_ddiff_fn, b_ddiff_fn))))
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

    fn check(&mut self, coords: &Coords, meta: M) -> FailResult<()> {
        self.0.check(coords, meta.clone())?;
        self.1.check(coords, meta.clone())?;
        Ok(())
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

    fn check(&mut self, coords: &Coords, meta: M) -> FailResult<()> {
        self.0.check(coords, meta.clone())?;
        self.1.check(coords, meta.clone())?;
        Ok(())
    }
}

impl<M, A, B> PairwiseDDiffFn<M> for Sum<A, B>
where
    M: Clone,
    A: PairwiseDDiffFn<M>,
    B: PairwiseDDiffFn<M>,
{
    fn compute(&mut self, coords: &Coords, meta: M) -> FailResult<(f64, Vec<(BondGrad, M33)>)> {
        let (a_value, a_grad) = self.0.compute(coords, meta.clone())?;
        let (b_value, b_grad) = self.1.compute(coords, meta.clone())?;
        let value = a_value + b_value;
        let mut grad = a_grad;
        grad.extend(b_grad);
        Ok((value, grad))
    }

    fn check(&mut self, coords: &Coords, meta: M) -> FailResult<()> {
        self.0.check(coords, meta.clone())?;
        self.1.check(coords, meta.clone())?;
        Ok(())
    }
}

impl<A, B> DispFn for Sum<A, B>
where
    A: DispFn,
    B: DispFn,
{
    fn compute_sparse_force_delta(&mut self, disp: (usize, V3)) -> FailResult<BTreeMap<usize, V3>>
    {
        let mut larger = self.0.compute_sparse_force_delta(disp)?;
        let mut smaller = self.1.compute_sparse_force_delta(disp)?;
        if larger.len() < smaller.len() {
            std::mem::swap(&mut larger, &mut smaller);
        }

        for (key, value) in smaller {
            *larger.entry(key).or_insert(V3::zero()) += value;
        }
        Ok(larger)
    }
}

//--------------------------------

/// A default implementation for `PotentialBuilder::initialize_disp_fn` in terms of a `DiffFn`.
///
/// It computes the dense forces at each displacement and subtracts the forces of the
/// original coordinates.  Differences are suppressed from the result only where they are
/// exactly zero (i.e. it is expected that the potential will produce identical forces
/// for atoms far away from the displaced atom).
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
    pub fn initialize(equilibrium_coords: &Coords, meta: Meta, mut diff_fn: Box<dyn DiffFn<Meta>>) -> FailResult<Self>
    {Ok({
        let lattice = equilibrium_coords.lattice().clone();
        let equilibrium_carts = equilibrium_coords.to_carts();

        let equilibrium_coords = equilibrium_coords.with_carts(equilibrium_carts.clone());
        let equilibrium_force = diff_fn.compute_force(&equilibrium_coords, meta.clone())?;

        DefaultDispFn { lattice, equilibrium_carts, equilibrium_force, meta, diff_fn }
    })}
}

impl<Meta> DispFn for DefaultDispFn<Meta>
where Meta: Clone,
{
    fn compute_sparse_force_delta(&mut self, disp: (usize, V3)) -> FailResult<BTreeMap<usize, V3>>
    {
        let final_force = {
            let mut carts = self.equilibrium_carts.to_vec();
            carts[disp.0] += disp.1;

            let coords = CoordsKind::Carts(carts);
            let coords = Coords::new(self.lattice.clone(), coords);
            self.diff_fn.compute_force(&coords, self.meta.clone())?
        };
        Ok(sparse_deltas_from_dense_deterministic(&self.equilibrium_force, &final_force))
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
pub fn sparse_deltas_from_dense_deterministic(
    original_force: &[V3],
    final_force: &[V3],
) -> BTreeMap<usize, V3> {
    zip_eq!(original_force, final_force).enumerate()
        .map(|(atom, (old, new))| (atom, new - old))
        .filter(|&(_, v)| v != V3::zero())
        .collect()
}

//--------------------------------

pub use disp_fn_helper::DispFnHelper;
pub mod disp_fn_helper {
    use super::*;

    /// A type that can help simplify `DispFn` implementations by remembering the equilibrium
    /// coords and gradient.
    ///
    /// With this type, you only need to implement the `disp_fn_helper::`[`Callback`] trait
    /// instead of `DispFn`.  The `Callback` trait is designed to be implemented by the same
    /// type that implements `DiffFn` or `BondDiffFn`.
    pub struct DispFnHelper<Meta, Other> {
        equilibrium_coords: Coords,
        equilibrium_grad: Option<Vec<V3>>,
        equilibrium_bond_grad: Option<Vec<BondGrad>>,
        meta: Meta,
        other: Other,
    }

    struct MyDispFn<Meta, Other> {
        builder: DispFnHelper<Meta, Other>,
        diff_fn: Box<dyn Callback<Meta, Other>>,
    }

    impl<Meta> DispFnHelper<Meta, ()> {
        pub fn new(equilibrium_coords: &Coords, meta: Meta) -> Self
        { DispFnHelper {
            equilibrium_coords: equilibrium_coords.clone(),
            equilibrium_grad: None,
            equilibrium_bond_grad: None,
            other: (),
            meta,
        }}
    }

    impl<Meta, Other> DispFnHelper<Meta, Other> {
        #[allow(unused)]
        pub fn with_grad(mut self, grad: Vec<V3>) -> Self
        { self.equilibrium_grad = Some(grad); self }

        #[allow(unused)]
        pub fn with_bond_grad(mut self, grad: Vec<BondGrad>) -> Self
        { self.equilibrium_bond_grad = Some(grad); self }

        pub fn with_other<NewOther>(self, other: NewOther) -> DispFnHelper<Meta, NewOther>
        { DispFnHelper {
            equilibrium_coords: self.equilibrium_coords,
            equilibrium_grad: self.equilibrium_grad,
            equilibrium_bond_grad: self.equilibrium_bond_grad,
            meta: self.meta,
            other,
        }}
    }

    impl<Meta, Other> DispFnHelper<Meta, Other>
    where
        Meta: Clone + 'static,
        Other: 'static,
    {
        pub fn build(self, diff_fn: impl Callback<Meta, Other> + 'static) -> Box<dyn DispFn>
        { Box::new(MyDispFn {
            builder: self,
            diff_fn: Box::new(diff_fn),
        })}
    }

    /// Data recorded about the equilibrium structure to make a `DispFn` easier to implement.
    pub struct Context<'a, Meta, Other> {
        pub equilibrium_coords: &'a Coords,
        pub meta: Meta,
        pub other: &'a Other,
        /// This will be `Some(_)` only if `with_grad` was called on the `DispFnHelper`.
        pub equilibrium_grad: Option<&'a [V3]>,
        /// This will be `Some(_)` only if `with_bond_grad` was called on the `DispFnHelper`.
        pub equilibrium_bond_grad: Option<&'a [BondGrad]>,
        _no_complete_destructure: (),
    }

    /// Alternative to the `DispFn` trait that's generally easier to implement for a type that
    /// already implements `DiffFn` or `BondDiffFn`.
    ///
    /// It will receive all of the context added to the `DispFnHelper` about the equilibrium
    /// structure.
    pub trait Callback<Meta, Other> {
        fn compute_sparse_grad_delta(
            &mut self,
            context: Context<'_, Meta, Other>,
            disp: (usize, V3),
        ) -> FailResult<BTreeMap<usize, V3>>;
    }

    impl<Meta, Other> DispFn for MyDispFn<Meta, Other>
    where Meta: Clone,
    {
        fn compute_sparse_force_delta(&mut self, disp: (usize, V3)) -> FailResult<BTreeMap<usize, V3>>
        {Ok({
            let context = Context {
                equilibrium_coords: &self.builder.equilibrium_coords,
                equilibrium_grad: self.builder.equilibrium_grad.as_ref().map(|x| &x[..]),
                equilibrium_bond_grad: self.builder.equilibrium_bond_grad.as_ref().map(|x| &x[..]),
                meta: self.builder.meta.clone(),
                other: &self.builder.other,
                _no_complete_destructure: (),
            };

            let grad = self.diff_fn.compute_sparse_grad_delta(context, disp)?;

            // grad to force
            grad.into_iter().map(|(i, grad)| (i, -grad)).collect()
        })}
    }
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

    fn check(&mut self, coords: &Coords, meta: Meta) -> FailResult<()>
    { self.0.check(coords, meta) }
}

//--------------------------------

pub fn sparse_grad_from_bond_grad(bond_grads: impl IntoIterator<Item=BondGrad>) -> BTreeMap<usize, V3> {
    let mut map = BTreeMap::new();
    for item in bond_grads {
        *map.entry(item.plus_site).or_insert_with(V3::zero) += item.grad;
        *map.entry(item.minus_site).or_insert_with(V3::zero) -= item.grad;
    }
    map
}
