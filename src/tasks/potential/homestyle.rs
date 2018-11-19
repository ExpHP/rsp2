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
use crate::FailResult;
use crate::meta::{self, prelude::*};
use rsp2_structure::{Coords, consts as elem, layer::Layers};
use rsp2_tasks_config as cfg;
use rsp2_array_types::{V3};
use rsp2_structure::bonds::{FracBonds, CartBond, FracBond, PeriodicGraph};
use rsp2_potentials::crespi as crespi_imp;
use rsp2_potentials::rebo::nonreactive as rebo_imp;

/// Rust implementation of Kolmogorov-Crespi Z.
///
/// **NOTE:** This has the limitation that the set of pairs within interaction range
/// must not change after the construction of the DiffFn. The elements also must not change.
#[derive(Debug, Clone)]
pub struct KolmogorovCrespiZ(pub(super) cfg::PotentialKolmogorovCrespiZNew);

// FIXME the whole layer deal is such a mess
impl PotentialBuilder<CommonMeta> for KolmogorovCrespiZ {
    fn initialize_diff_fn(&self, coords: &Coords, meta: CommonMeta) -> FailResult<Box<DiffFn<CommonMeta>>>
    {
        fn fn_body(me: &KolmogorovCrespiZ, coords: &Coords, meta: CommonMeta) -> FailResult<Box<DiffFn<CommonMeta>>> {
            let cfg::PotentialKolmogorovCrespiZNew { cutoff_begin } = me.0;
            let mut params = crespi_imp::Params::default();
            if let Some(cutoff_begin) = cutoff_begin {
                params.cutoff_begin = cutoff_begin;
            }

            let elements: meta::SiteElements = meta.pick();
            let layers = me.find_layers(coords, &meta).by_atom();

            // Collect VDW bonds
            // (with a much larger interaction radius than the bonds from CommonMeta)
            let bonds = FracBonds::from_brute_force(&coords, params.cutoff_end() * 1.001)?;
            let bonds = FracBonds::from_iter(coords.len(),
                (&bonds).into_iter()
                     // FIXME we should only enable interactions between adjacent layers
                     .filter(|&FracBond { from, to, image_diff: _ }| {
                        layers[from] != layers[to] && elements[from] != elements[to]
                     })
            );

            Ok(Box::new(Diff { params, bonds }))
        }

        struct Diff {
            params: crespi_imp::Params,
            bonds: FracBonds,
        }

        impl DiffFn<CommonMeta> for Diff {
            fn compute(&mut self, coords: &Coords, meta: CommonMeta) -> FailResult<(f64, Vec<V3>)> {
                let elements: meta::SiteElements = meta.pick();

                let bonds = self.bonds.to_cart_bonds(coords);

                let mut value = 0.0;
                let mut grad = vec![V3::zero(); coords.num_atoms()];
                for CartBond { from, to, cart_vector } in &bonds {
                    if (elements[from], elements[to]) != (elem::CARBON, elem::CARBON) {
                        continue;
                    }
                    let crespi_imp::Output {
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

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// FIXME: copy-pasta from KCZ in lammps.rs
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
impl KolmogorovCrespiZ {
    fn find_layers(&self, coords: &Coords, meta: &CommonMeta) -> Layers
    {
        use ::rsp2_structure::layer;

        let bonds: Option<meta::FracBonds> = meta.pick();
        let result = match bonds {
            Some(bonds) => {
                // enhance layer assignments using connected components to support layers that
                // overlap after buckling
                let ccs = bonds.to_periodic_graph().connected_components_by_site();
                layer::find_layers_with_labels(&ccs, &coords, V3([0, 0, 1]), 0.25)
            },

            // FIXME: this is supported just so that we can wait until after parameter
            //        optimization before generating the bond graph.
            //        There ought to be a better way...
            None => layer::find_layers(&coords, V3([0, 0, 1]), 0.25),
        };
        result.unwrap_or_else(|e| {
            panic!("Failure to determine layers when using kolmogorov/crespi/z: {}", e);
        })
    }
}

/// Rust implementation of REBO.
///
/// NOTE: This has the limitation that the set of pairs within interaction range
///       must not change after the construction of the DiffFn.
#[derive(Debug, Clone)]
pub struct Rebo {
    pub(super) cfg: cfg::PotentialReboNew,
    pub(super) parallel: bool,
}

impl PotentialBuilder<CommonMeta> for Rebo {
    fn parallel(&self, parallel: bool) -> Box<dyn PotentialBuilder<CommonMeta>> {
        let mut me = self.clone();
        me.parallel = parallel;
        Box::new(me)
    }

    fn initialize_diff_fn(&self, coords: &Coords, meta: CommonMeta) -> FailResult<Box<DiffFn<CommonMeta>>>
    {
        fn fn_body(me: &Rebo, coords: &Coords, meta: CommonMeta) -> FailResult<Box<DiffFn<CommonMeta>>> {
            let cfg::PotentialReboNew { params } = me.cfg;
            let params = match params {
                cfg::PotentialReboNewParams::Lammps => rebo_imp::Params::new_lammps(),
                cfg::PotentialReboNewParams::Brenner => rebo_imp::Params::new_brenner(),
            };

//            let bonds: meta::FracBonds = match meta.pick() {
//                Some(bonds) => bonds,
//                None => bail!("REBO requires a bond graph."),
//            };

            // FIXME: We can't currently use the bonds from meta because they might not have
            //        the right bond distances for our params.
            let elements: meta::SiteElements = meta.pick();
            let bonds = rebo_imp::compute_bond_graph(&params, coords, &elements)?;
            let parallel = me.parallel;
            Ok(Box::new(Diff { params, bonds, parallel }))
        }

        struct Diff {
            params: rebo_imp::Params,
            bonds: PeriodicGraph,
            parallel: bool,
        }

        impl DiffFn<CommonMeta> for Diff {
            fn compute(&mut self, coords: &Coords, meta: CommonMeta) -> FailResult<(f64, Vec<V3>)> {
                let elements: meta::SiteElements = meta.pick();
                rebo_imp::compute(&self.params, coords, &elements, &self.bonds, self.parallel)
            }
        }

        fn_body(self, coords, meta)
    }

    fn initialize_disp_fn(&self, coords: &Coords, meta: CommonMeta) -> FailResult<Box<DispFn>>
    { self._default_initialize_disp_fn(coords, meta) }
}

impl_dyn_clone_detail!{
    impl[] DynCloneDetail<CommonMeta> for Rebo { ... }
}

#[cfg(not_now)]
#[test]
fn test_rebo_diff() -> FailResult<()> {
    use ::rsp2_structure::{Lattice, CoordsKind, consts as elem};
    use ::rsp2_array_types::{Envee};
    use ::rsp2_minimize::numerical;
    use ::slice_of_array::prelude::*;
    use ::meta::{self, prelude::*};

    let mut coords = Coords::new(
        Lattice::from([
            [2.459270778739769, 0.0, 0.0],
            [-1.2296353893698847, 2.129790969173379, 0.0],
            [0.0, 0.0, 13.374096340130473],
        ]),
        CoordsKind::Carts(vec![
            [0.0, 0.0, 5.0],
            [1.2296353893698847, 0.7099303230577932, 5.0],
        ].envee()),
    );
    coords.carts_mut()[1][0] += 0.1;
    coords.carts_mut()[1][2] += 0.1;
//    let cfg_lmp: cfg::PotentialKind = from_json!{{
//        "rebo": {
//            "omp": false,
//        },
//    }};
    let cfg_rsp2: cfg::PotentialKind = from_json!{{
        "rebo-new": {
            "params": "lammps",
        },
    }};
    let elements: meta::SiteElements = vec![elem::CARBON; 2].into();
    let masses: meta::SiteMasses = vec![meta::Mass(12.0107); 2].into();
    let bonds: meta::FracBonds = ::std::rc::Rc::new(FracBonds::from_brute_force(&coords, 2.0)?);
    let meta = hlist![elements, masses, Some(bonds)];

//    let pot_lmp = PotentialBuilder::from_config_parts(None, None, &cfg::Threading::Serial, &cfg::LammpsUpdateStyle::Safe, &[true; 3], &cfg_lmp).allow_blocking(true);
    let pot_rsp2 = PotentialBuilder::from_config_parts(None, None, &cfg::Threading::Serial, &cfg::LammpsUpdateStyle::Safe, &[true; 3], &cfg_rsp2).allow_blocking(true);

    //let diff_lmp = pot_lmp.initialize_diff_fn(&coords, meta.sift())?.compute(&coords, meta.sift())?;
    let diff_rsp2 = pot_rsp2.initialize_diff_fn(&coords, meta.sift())?.compute(&coords, meta.sift())?;

    let num_grad = numerical::try_gradient(1e-4, None, &coords.to_carts().flat(), |carts| {
        let coords = coords.clone().with_carts(carts.nest().to_vec());
        pot_rsp2.initialize_diff_fn(&coords, meta.sift())?.compute(&coords, meta.sift())
            .map(|x| x.0)
    })?;

    assert_close!(diff_rsp2.1.flat(), &num_grad[..]);
    Ok(())
}

#[cfg(not_now)]
#[test]
fn test_rebo_value() -> FailResult<()> {
    use ::rsp2_structure::{Lattice, CoordsKind, consts as elem};
    use ::rsp2_array_types::{Envee, Unvee};
    use ::meta::{self, prelude::*};

    let mut coords = Coords::new(
        Lattice::from([
            [2.459270778739769, 0.0, 0.0],
            [-1.2296353893698847, 2.129790969173379, 0.0],
            [0.0, 0.0, 13.374096340130473],
        ]),
        CoordsKind::Carts(vec![
            [0.0, 0.0, 5.0],
            [1.2296353893698847, 0.7099303230577932, 5.0],
        ].envee()),
    );
    coords.carts_mut()[1][0] += 0.1;
    coords.carts_mut()[1][2] += 0.1;

    let cfg_lmp: cfg::PotentialKind = from_json!{{
        "rebo": {
            "omp": false,
        },
    }};
    let cfg_rsp2: cfg::PotentialKind = from_json!{{
        "rebo-new": {
            "params": "lammps",
        },
    }};

    let elements: meta::SiteElements = vec![elem::CARBON; 2].into();
    let masses: meta::SiteMasses = vec![meta::Mass(12.0107); 2].into();
    let bonds: meta::FracBonds = ::std::rc::Rc::new(FracBonds::from_brute_force(&coords, 2.0)?);
    println!("{:?}", bonds);
    let meta = hlist![elements, masses, Some(bonds)];

    let pot_lmp = PotentialBuilder::from_config_parts(None, None, &cfg::Threading::Serial, &cfg::LammpsUpdateStyle::Safe, &[true; 3], &cfg_lmp).allow_blocking(true);
    let pot_rsp2 = PotentialBuilder::from_config_parts(None, None, &cfg::Threading::Serial, &cfg::LammpsUpdateStyle::Safe, &[true; 3], &cfg_rsp2).allow_blocking(true);

    let diff_lmp = pot_lmp.initialize_diff_fn(&coords, meta.sift())?.compute(&coords, meta.sift())?;
    let diff_rsp2 = pot_rsp2.initialize_diff_fn(&coords, meta.sift())?.compute(&coords, meta.sift())?;

    println!("{:?}", diff_lmp);
    println!("{:?}", diff_rsp2);
    assert_close!(diff_lmp.0, diff_rsp2.0);
    assert_close!(diff_lmp.1.unvee(), diff_rsp2.1.unvee());
    Ok(())
}
