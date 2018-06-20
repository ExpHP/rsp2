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

//! PotentialBuilder implementations for potentials implemented within rsp2.

use super::{DynCloneDetail, PotentialBuilder, DiffFn, DispFn, CommonMeta};
use ::FailResult;
use ::rsp2_structure::{Coords};
use ::rsp2_tasks_config as cfg;
use ::rsp2_array_types::{V3};
use ::math::bonds::{FracBonds, CartBond};

/// Rust implementation of Kolmogorov-Crespi Z.
///
/// NOTE: This has the limitation that the set of pairs within interaction range
///       must not change after the construction of the DiffFn.
#[derive(Debug, Clone)]
pub struct KolmogorovCrespiZ(pub(super) cfg::PotentialKolmogorovCrespiZNew);

impl PotentialBuilder<CommonMeta> for KolmogorovCrespiZ {
    fn initialize_diff_fn(&self, coords: &Coords, meta: CommonMeta) -> FailResult<Box<DiffFn<CommonMeta>>>
    {
        fn fn_body(me: &KolmogorovCrespiZ, coords: &Coords, _: CommonMeta) -> FailResult<Box<DiffFn<CommonMeta>>> {
            let cfg::PotentialKolmogorovCrespiZNew { cutoff_begin } = me.0;
            let mut params = ::math::crespi::Params::default();
            if let Some(cutoff_begin) = cutoff_begin {
                params.cutoff_begin = cutoff_begin;
            }
            let bonds = FracBonds::from_brute_force_very_dumb(&coords, params.cutoff_end() * 1.001)?;
            Ok(Box::new(Diff { params, bonds }))
        }

        struct Diff {
            params: ::math::crespi::Params,
            bonds: FracBonds,
        }

        impl DiffFn<CommonMeta> for Diff {
            fn compute(&mut self, coords: &Coords, _: CommonMeta) -> FailResult<(f64, Vec<V3>)> {
                let bonds = self.bonds.to_cart_bonds(coords);

                let mut value = 0.0;
                let mut grad = vec![V3::zero(); coords.num_atoms()];
                for CartBond { from: _, to, cart_vector } in &bonds {
                    let ::math::crespi::Output {
                        value: part_value,
                        grad_rij: part_grad, ..
                    } = self.params.crespi_z(cart_vector);

                    value += part_value;
                    grad[to] += part_grad;
                }
                trace!("KCZ: {}", value);
                Ok((value, grad))
            }
        }

        fn_body(self, coords, meta)
    }

    fn initialize_disp_fn(&self, coords: &Coords, meta: CommonMeta) -> FailResult<Box<DispFn>>
    { self._default_initialize_disp_fn(coords, meta) }
}

impl_dyn_clone_detail!{
    impl[] DynCloneDetail<CommonMeta> for KolmogorovCrespiZ { ... }
}
