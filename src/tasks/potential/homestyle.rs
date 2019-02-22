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

use super::{DynCloneDetail, PotentialBuilder, DiffFn, DispFn, CommonMeta, BondDiffFn, BondGrad};
use super::helper::DiffFnFromBondDiffFn;
use crate::FailResult;
use crate::math::frac_bonds_with_skin::FracBondsWithSkin;
use crate::meta::{self, prelude::*};
use rayon_cond::CondIterator;
use rsp2_structure::{Coords, layer::Layers, Element, consts::{CARBON}};
use rsp2_tasks_config as cfg;
use rsp2_array_types::{V3};
use rsp2_potentials::crespi as crespi_imp;
use rsp2_potentials::rebo::nonreactive as rebo_imp;

/// Rust implementation of Kolmogorov-Crespi Z.
///
/// **NOTE:** This has the limitation that the set of pairs within interaction range
/// must not change after the construction of the DiffFn. The elements also must not change.
#[derive(Debug, Clone)]
pub struct KolmogorovCrespiZ {
    pub(super) cfg: cfg::PotentialKolmogorovCrespiZNew,
    pub(super) parallel: bool,
}

// FIXME the whole layer deal is such a mess
impl PotentialBuilder<CommonMeta> for KolmogorovCrespiZ {
    fn initialize_diff_fn(&self, coords: &Coords, meta: CommonMeta) -> FailResult<Box<dyn DiffFn<CommonMeta>>>
    { Ok(Box::new(DiffFnFromBondDiffFn::new(self.initialize_bond_diff_fn(coords, meta)?.unwrap()))) }

    fn parallel(&self, parallel: bool) -> Box<dyn PotentialBuilder<CommonMeta>> {
        let mut me = self.clone();
        me.parallel = parallel;
        Box::new(me)
    }

    fn initialize_bond_diff_fn(&self, coords: &Coords, meta: CommonMeta) -> FailResult<Option<Box<dyn BondDiffFn<CommonMeta>>>>
    {
        fn fn_body(me: &KolmogorovCrespiZ, coords: &Coords, meta: CommonMeta) -> FailResult<Option<Box<dyn BondDiffFn<CommonMeta>>>> {
            let cfg::PotentialKolmogorovCrespiZNew { cutoff_begin, cutoff_transition_dist, skin_depth, skin_check_frequency } = me.cfg;
            let parallel = me.parallel;

            let mut params = crespi_imp::Params::default();
            if let Some(cutoff_begin) = cutoff_begin {
                params.cutoff_begin = cutoff_begin;
            }
            // (note: these are both Option<f64> but the meaning of None is different;
            //        in the settings, it means the value wasn't provided;
            //        in the params, it means to use a sharp cutoff)
            if let Some(cutoff_transition_dist) = cutoff_transition_dist {
                if cutoff_transition_dist == 0.0 {
                    params.cutoff_transition_dist = None;
                } else {
                    params.cutoff_transition_dist = Some(cutoff_transition_dist);
                }
            } else {
                // use value from Params::default()
            }

            let layers = me.find_layers(coords, &meta).by_atom();

            let interaction_radius = params.cutoff_end() * (1.0 + 1e-7);
            let mut bonds = FracBondsWithSkin::new(
                Box::new(move |&(elem_a, layer_a): &BondMeta, &(elem_b, layer_b): &BondMeta| {
                    match (elem_a, elem_b) {
                        (CARBON, CARBON) => match i32::abs(layer_a as i32 - layer_b as i32) {
                            1 => Some(interaction_radius),
                            _ => None,
                        },
                        _ => None,
                    }
                }) as Box<dyn Fn(&_, &_) -> _>,
                skin_depth,
            );
            bonds.set_check_frequency(skin_check_frequency);

            Ok(Some(Box::new(Diff { params, bonds, layers, parallel })))
        }

        type BondMeta = (Element, usize);
        struct Diff {
            params: crespi_imp::Params,
            layers: Vec<usize>,
            bonds: FracBondsWithSkin<
                BondMeta,
                dyn Fn(&BondMeta, &BondMeta) -> Option<f64>,
            >,
            parallel: bool,
        }

        impl BondDiffFn<CommonMeta> for Diff {
            fn compute(&mut self, coords: &Coords, meta: CommonMeta) -> FailResult<(f64, Vec<BondGrad>)> {
                let elements: meta::SiteElements = meta.pick();

                let meta_for_bonds = zip_eq!(elements.iter().cloned(), self.layers.iter().cloned());
                let frac_bonds = self.bonds.compute(coords, meta_for_bonds)?;

                let cart_coords = coords.with_carts(coords.to_carts());

                let params = &self.params;

                let (part_values, bond_grads): (Vec<f64>, Vec<_>) = {
                    // HACK: collect to vec so that it implements IntoParallelIterator
                    let frac_bonds = frac_bonds.into_iter().filter(|bond| bond.is_canonical()).collect::<Vec<_>>();
                    // HACK: collect from Rc<[_]> to Vec to impl Send
                    let elements = elements.iter().cloned().collect::<Vec<_>>();

                    CondIterator::new(frac_bonds, self.parallel)
                        .map(|bond| {
                            debug_assert_eq!((elements[bond.from], elements[bond.to]), (CARBON, CARBON));
                            let cart_vector = bond.cart_vector_using_cache(&cart_coords).unwrap();
                            let (part_value, part_grad) = params.compute_z(cart_vector);

                            let bond_grad = BondGrad {
                                plus_site: bond.to,
                                minus_site: bond.from,
                                grad: part_grad,
                                cart_vector,
                            };
                            (part_value, bond_grad)
                        }).unzip()
                };
                let value = part_values.iter().sum();
                Ok((value, bond_grads))
            }
        }

        fn_body(self, coords, meta)
    }

    fn initialize_disp_fn(&self, coords: &Coords, meta: CommonMeta) -> FailResult<Box<dyn DispFn>>
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
        use rsp2_structure::layer;

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
    fn initialize_diff_fn(&self, coords: &Coords, meta: CommonMeta) -> FailResult<Box<dyn DiffFn<CommonMeta>>>
    { Ok(Box::new(DiffFnFromBondDiffFn::new(self.initialize_bond_diff_fn(coords, meta)?.unwrap()))) }

    fn parallel(&self, parallel: bool) -> Box<dyn PotentialBuilder<CommonMeta>> {
        let mut me = self.clone();
        me.parallel = parallel;
        Box::new(me)
    }

    fn initialize_bond_diff_fn(&self, coords: &Coords, meta: CommonMeta) -> FailResult<Option<Box<dyn BondDiffFn<CommonMeta>>>>
    {
        fn fn_body(me: &Rebo, coords: &Coords, meta: CommonMeta) -> FailResult<Option<Box<dyn BondDiffFn<CommonMeta>>>> {
            let cfg::PotentialReboNew { params } = me.cfg;
            let params = match params {
                cfg::PotentialReboNewParams::Lammps => rebo_imp::Params::new_lammps(),
                cfg::PotentialReboNewParams::Brenner => rebo_imp::Params::new_brenner(),
            };

            // NOTE: We can't (currently) use the bonds from meta because they might not have
            //       the right bond distances for our params.
            let elements: meta::SiteElements = meta.pick();
            let interactions = rebo_imp::find_all_interactions(&params, coords, &elements)?;
            let parallel = me.parallel;
            Ok(Some(Box::new(Diff { params, interactions, parallel })))
        }

        struct Diff {
            params: rebo_imp::Params,
            interactions: rebo_imp::Interactions,
            parallel: bool,
        }

        impl BondDiffFn<CommonMeta> for Diff {
            fn compute(&mut self, coords: &Coords, _: CommonMeta) -> FailResult<(f64, Vec<BondGrad>)> {
                let (value, grad) = rebo_imp::compute_by_bond(&self.params, &self.interactions, coords, self.parallel)?;
                let grad = {
                    grad.into_iter().map(|item| {
                        let rebo_imp::BondGrad { plus_site, minus_site, cart_vector, grad } = item;
                        super::BondGrad { plus_site, minus_site, cart_vector, grad }
                    }).collect()
                };
                Ok((value, grad))
            }
        }

        fn_body(self, coords, meta)
    }

    fn initialize_disp_fn(&self, coords: &Coords, meta: CommonMeta) -> FailResult<Box<dyn DispFn>>
    { self._default_initialize_disp_fn(coords, meta) }
}

impl_dyn_clone_detail!{
    impl[] DynCloneDetail<CommonMeta> for Rebo { ... }
}

#[cfg(not_now)]
#[test]
fn test_rebo_diff() -> FailResult<()> {
    use rsp2_structure::{Lattice, CoordsKind, bonds::FracBonds};
    use rsp2_array_types::{Envee};
    use rsp2_minimize::numerical;
    use slice_of_array::prelude::*;
    use meta::{self, prelude::*};

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
    let elements: meta::SiteElements = vec![CARBON; 2].into();
    let masses: meta::SiteMasses = vec![meta::Mass(12.0107); 2].into();
    let bonds: meta::FracBonds = std::rc::Rc::new(FracBonds::from_brute_force(&coords, 2.0)?);
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
    use rsp2_structure::{Lattice, CoordsKind, bonds::FracBonds};
    use rsp2_array_types::{Envee, Unvee};
    use meta::{self, prelude::*};

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

    let elements: meta::SiteElements = vec![CARBON; 2].into();
    let masses: meta::SiteMasses = vec![meta::Mass(12.0107); 2].into();
    let bonds: meta::FracBonds = std::rc::Rc::new(FracBonds::from_brute_force(&coords, 2.0)?);
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
