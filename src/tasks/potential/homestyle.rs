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

use super::{DynCloneDetail, PotentialBuilder, DiffFn, DispFn, CommonMeta, BondDiffFn, PairwiseDDiffFn, BondGrad};
use super::helper::{self, DiffFnFromBondDiffFn, disp_fn_helper::{self, DispFnHelper}};
use crate::FailResult;
use crate::math::frac_bonds_with_skin::FracBondsWithSkin;
#[allow(unused)] // rustc bug
use crate::meta::{self, prelude::*};

use rsp2_structure::{Coords, layer::Layers, Element, bonds::{FracBond, FracBonds}};
use rsp2_tasks_config as cfg;
use rsp2_array_types::{V3, M33};
use rsp2_potentials::crespi as crespi_imp;
use rsp2_potentials::rebo::nonreactive as rebo_imp;

use rayon_cond::CondIterator;
use std::collections::BTreeMap;

pub use kc::Builder as KolmogorovCrespi;
mod kc {
    use super::*;
    use rsp2_structure::bonds::PeriodicGraph;

    /// Rust implementation of Kolmogorov-Crespi, for layers along the Z axis.
    ///
    /// **NOTE:** This has the following limitations on how the structure must not change:
    ///
    /// * The set of pairs within interaction range must not change after the construction of
    ///   the DiffFn. (TODO: Is this still true? We have that FracBondsWithSkin thing. Was
    ///   there some other reason for this limitation?)
    /// * The intralayer bond graph must not change.
    /// * The elements must not change.
    #[derive(Debug, Clone)]
    pub struct Builder {
        pub(in crate::potential) cfg: cfg::PotentialKolmogorovCrespi,
        pub(in crate::potential) parallel: bool,
    }

    // FIXME the whole layer deal is such a mess
    impl PotentialBuilder<CommonMeta> for KolmogorovCrespi {
        fn initialize_diff_fn(&self, coords: &Coords, meta: CommonMeta) -> FailResult<Box<dyn DiffFn<CommonMeta>>>
        { Ok(Box::new(DiffFnFromBondDiffFn::new(self.initialize_bond_diff_fn(coords, meta)?.unwrap()))) }

        fn parallel(&self, parallel: bool) -> Box<dyn PotentialBuilder<CommonMeta>> {
            let mut me = self.clone();
            me.parallel = parallel;
            Box::new(me)
        }

        fn initialize_bond_diff_fn(&self, coords: &Coords, meta: CommonMeta) -> FailResult<Option<Box<dyn BondDiffFn<CommonMeta>>>>
        { Ok(Some(Box::new(self._initialize_bond_diff_fn(coords, meta)?) as Box<_>)) }

        fn initialize_pairwise_ddiff_fn(&self, coords: &Coords, meta: CommonMeta) -> FailResult<Option<Box<dyn PairwiseDDiffFn<CommonMeta>>>>
        { Ok(Some(Box::new(self._initialize_bond_diff_fn(coords, meta)?) as Box<_>)) }

        fn initialize_disp_fn(&self, coords: &Coords, meta: CommonMeta) -> FailResult<Box<dyn DispFn>>
        { Ok(Box::new(self._initialize_disp_fn(coords, meta)?) as Box<_>) }
    }

    impl Builder {
        fn _initialize_bond_diff_fn(&self, coords: &Coords, meta: CommonMeta) -> FailResult<Diff>
        {
            let cfg::PotentialKolmogorovCrespi {
                cutoff_begin, cutoff_transition_dist, skin_depth, skin_check_frequency,
                ref normals, ref params,
            } = self.cfg;
            let parallel = self.parallel;

            let mut params = match params {
                cfg::KolmogorovCrespiParams::Original => crespi_imp::Params::original(),
                cfg::KolmogorovCrespiParams::Ouyang => crespi_imp::Params::ouyang(),
            };
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

            let layers = self.find_layers(coords, &meta).by_atom();

            let interaction_radius = params.cutoff_end() * (1.0 + 1e-7);
            let mut bonds = FracBondsWithSkin::new(
                Box::new(move |&(elem_a, layer_a): &BondMeta, &(elem_b, layer_b): &BondMeta| {
                    match (elem_a, elem_b) {
                        (Element::CARBON, Element::CARBON) => {
                            match i32::abs(layer_a as i32 - layer_b as i32) {
                                1 => Some(interaction_radius),
                                _ => None,
                            }
                        },
                        _ => None,
                    }
                }) as Box<dyn Fn(&_, &_) -> _>,
                skin_depth,
            );
            bonds.set_check_frequency(skin_check_frequency);

            let get_bond_graph = || {
                let intralayer_bonds: Option<meta::FracBonds> = meta.pick();
                let intralayer_bonds = match intralayer_bonds {
                    Some(bonds) => bonds,
                    None => {
                        // FIXME: copypasta
                        let elements: meta::SiteElements = meta.pick();
                        std::rc::Rc::new(rsp2_structure::bonds::FracBonds::compute_with_meta(
                            coords,
                            elements.iter().cloned(),
                            |&a, &b| match (a, b) {
                                (Element::CARBON, Element::CARBON) => Some(2.01),
                                (Element::CARBON, Element::HYDROGEN) |
                                (Element::HYDROGEN, Element::CARBON) => Some(1.51),
                                (Element::HYDROGEN, Element::HYDROGEN) => Some(1.21),
                                _ => None,
                            },
                        ).expect("couldn't get bonds"))
                    },
                };
                intralayer_bonds.to_periodic_graph()
            };

            let normal_info = match normals {
                cfg::KolmogorovCrespiNormals::Z {} => NormalInfo::Z,
                cfg::KolmogorovCrespiNormals::LocalFast {} => NormalInfo::NotZ {
                    kind: NormalKind::LocalFast,
                    intralayer_graph: get_bond_graph(),
                },
                cfg::KolmogorovCrespiNormals::LocalTrue {} => NormalInfo::NotZ {
                    kind: NormalKind::Local,
                    intralayer_graph: get_bond_graph(),
                },
            };

            Ok(Diff { params, interaction_pairs: bonds, layers, parallel, normal_info })
        }
    }

    impl_dyn_clone_detail!{
        impl[] DynCloneDetail<CommonMeta> for KolmogorovCrespi { ... }
    }

    type BondMeta = (Element, usize);
    /// The object responsible for performing computations and maintaining the list of interactions.
    struct Diff {
        params: crespi_imp::Params,
        layers: Vec<usize>,
        interaction_pairs: FracBondsWithSkin<
            BondMeta,
            dyn Fn(&BondMeta, &BondMeta) -> Option<f64>,
        >,
        parallel: bool,
        normal_info: NormalInfo,
    }

    // FIXME we should probably just get rid of this type in favor of NormalKind
    //
    // Information needed to compute normals
    enum NormalInfo {
        Z,
        NotZ {
            kind: NormalKind,
            intralayer_graph: PeriodicGraph,
        },
    }

    impl BondDiffFn<CommonMeta> for Diff {
        fn compute(&mut self, coords: &Coords, meta: CommonMeta) -> FailResult<(f64, Vec<BondGrad>)> {
            let elements: meta::SiteElements = meta.pick();

            let meta_for_bonds = zip_eq!(elements.iter().cloned(), self.layers.iter().cloned());
            let frac_bonds = self.interaction_pairs.compute(coords, meta_for_bonds)?;

            compute_using_frac_bonds(
                self.parallel, &self.params,
                coords, meta, &self.normal_info,
                frac_bonds.into_iter().collect(),
            )
        }
    }

    impl PairwiseDDiffFn<CommonMeta> for Diff {
        fn compute(&mut self, coords: &Coords, meta: CommonMeta) -> FailResult<(f64, Vec<(BondGrad, M33)>)> {
            let elements: meta::SiteElements = meta.pick();

            let meta_for_bonds = zip_eq!(elements.iter().cloned(), self.layers.iter().cloned());
            let frac_bonds = self.interaction_pairs.compute(coords, meta_for_bonds)?;

            compute_with_hessian_using_frac_bonds(
                self.parallel, &self.params,
                coords, meta, &self.normal_info,
                frac_bonds.into_iter().collect(),
            )
        }
    }

    impl Diff {
        /// Get all FracBonds for a structure, using cached data if its still valid.
        fn compute_frac_bonds(&mut self, coords: &Coords, meta: CommonMeta) -> FailResult<&FracBonds> {
            let elements: meta::SiteElements = meta.pick();
            let meta_for_bonds = zip_eq!(elements.iter().cloned(), self.layers.iter().cloned());
            self.interaction_pairs.compute(coords, meta_for_bonds)
        }
    }

    /// Compute terms for all of the provided FracBonds, possibly in parallel.
    ///
    /// For z normals, one term is produced per interlayer pair interaction.  For other normals,
    /// additional terms are produced for intralayer interactions. (each intralayer bond will appear
    /// multiple times, with the endpoint atoms listed in arbitrary order)
    fn compute_using_frac_bonds(
        parallel: bool,
        params: &crespi_imp::Params,
        coords: &Coords,
        meta: CommonMeta,
        normal_info: &NormalInfo,
        interaction_pairs: Vec<FracBond>, // to be monomorphic
    ) -> FailResult<(f64, Vec<BondGrad>)> {
        let elements: meta::SiteElements = meta.pick();
        let lattice = coords.lattice();
        let carts = coords.to_carts();

        // HACK: collect to vec so that it implements IntoParallelIterator
        let interaction_pairs = interaction_pairs.into_iter().filter(|bond| bond.is_canonical()).collect::<Vec<_>>();
        // HACK: collect from Rc<[_]> to Vec to impl Send
        let elements = elements.iter().cloned().collect::<Vec<_>>();

        let (part_values, bond_grads) = match normal_info {
            NormalInfo::Z { } => {
                CondIterator::new(interaction_pairs, parallel)
                    .map(|bond| {
                        debug_assert!(bond.is_canonical());
                        debug_assert_eq!(elements[bond.from], Element::CARBON);
                        debug_assert_eq!(elements[bond.to], Element::CARBON);
                        let cart_vector = bond.cart_vector_using_carts(lattice, &carts);
                        let (part_value, part_grad) = params.compute_z(cart_vector);

                        let bond_grad = BondGrad {
                            plus_site: bond.to,
                            minus_site: bond.from,
                            grad: part_grad,
                            cart_vector,
                        };
                        (part_value, bond_grad)
                    }).unzip()
            },

            &NormalInfo::NotZ { kind, ref intralayer_graph } => {
                let normals = Normals::compute(intralayer_graph, coords, kind)?;

                let (part_values, grad_items): (Vec<_>, Vec<_>) = {
                    CondIterator::new(interaction_pairs, parallel)
                        .map(|bond| {
                            debug_assert!(bond.is_canonical());
                            debug_assert_eq!(elements[bond.from], Element::CARBON);
                            debug_assert_eq!(elements[bond.to], Element::CARBON);
                            let cart_vector = bond.cart_vector_using_carts(lattice, &carts);
                            let ni = normals.lookup(bond.from);
                            let nj = normals.lookup(bond.to);

                            let crespi_imp::Output {
                                value, grad_rij, grad_ni, grad_nj,
                            } = params.compute(cart_vector, ni, nj);

                            let bond_grad = BondGrad {
                                plus_site: bond.to,
                                minus_site: bond.from,
                                grad: grad_rij,
                                cart_vector,
                            };
                            (value, (bond_grad, grad_ni, grad_nj))
                        }).unzip()
                };

                let mut bond_grads = Vec::with_capacity(grad_items.len() + 2 * coords.len());
                let mut normal_grads = vec![V3::zero(); coords.len()];
                for (bond_grad, grad_ni, grad_nj) in grad_items {
                    bond_grads.push(bond_grad);
                    normal_grads[bond_grad.minus_site] += grad_ni;
                    normal_grads[bond_grad.plus_site] += grad_nj;
                }

                for (site, d_normal) in normal_grads.into_iter().enumerate() {
                    bond_grads.extend(normals.get_bond_grad_terms(site, d_normal));
                }

                (part_values, bond_grads)
            }, // NormalInfo::NotZ => { ... }
        }; // let (part_values, bond_grads) = match normal_info { ... }
        let value = part_values.iter().sum();
        Ok((value, bond_grads))
    }

    /// FIXME: code duplication
    fn compute_with_hessian_using_frac_bonds(
        parallel: bool,
        params: &crespi_imp::Params,
        coords: &Coords,
        meta: CommonMeta,
        normal_info: &NormalInfo,
        frac_bonds: Vec<FracBond>, // to be monomorphic
    ) -> FailResult<(f64, Vec<(BondGrad, M33)>)> {
        let (part_values, extras): (Vec<f64>, Vec<_>) = {
            let elements: meta::SiteElements = meta.pick();
            let lattice = coords.lattice();
            let carts = coords.to_carts();

            // HACK: collect to vec so that it implements IntoParallelIterator
            let frac_bonds = frac_bonds.into_iter().filter(|bond| bond.is_canonical()).collect::<Vec<_>>();
            // HACK: collect from Rc<[_]> to Vec to impl Send
            let elements = elements.iter().cloned().collect::<Vec<_>>();

            match normal_info {
                NormalInfo::Z => {
                    CondIterator::new(frac_bonds, parallel)
                        .map(|bond| {
                            debug_assert!(bond.is_canonical());
                            debug_assert_eq!(elements[bond.from], Element::CARBON);
                            debug_assert_eq!(elements[bond.to], Element::CARBON);
                            let cart_vector = bond.cart_vector_using_carts(lattice, &carts);
                            let (part_value, part_grad, part_hessian) = params.compute_z_with_hessian(cart_vector);

                            let bond_grad = BondGrad {
                                plus_site: bond.to,
                                minus_site: bond.from,
                                grad: part_grad,
                                cart_vector,
                            };
                            (part_value, (bond_grad, part_hessian))
                        }).unzip()
                },

                NormalInfo::NotZ { .. } => {
                    unimplemented!("analytic hessian with general normal vectors for Kolmogorov Crespi is not supported!");
                },
            }
        };
        let value = part_values.iter().sum();
        Ok((value, extras))
    }

    /// KCZ has an optimized `DispFn` made possible by its simple definition.
    ///
    /// This `DispFn` simply considers all of the `BondGrad` terms where either `plus_site` or
    /// `minus_site` is the displaced atom.
    ///
    /// On large structures, this reduces the total worst-case time complexity of computing all
    /// force sets by a factor of `num_atoms`.
    ///
    /// This is only valid because KCZ is a "pairwise" potentials, where every summand in the
    /// potential is a function of exactly one bond vector. This could not be done for REBO, which
    /// contains terms that depend on two vectors (the bond-angle terms) or more (the dihedral
    /// angle terms).
    struct ZDispFnOther {
        // For each site, all of the FracBonds in the equilibrium structure where
        // the site appears as either the `to` or the `from` site.
        //
        // It remains valid to use these on displaced structures so long as the displacement
        // distance does not exceed the neighbor list skin depth.
        canonical_bonds_by_site: Vec<Vec<FracBond>>,
    }

    impl Builder {
        fn _initialize_disp_fn(&self, coords: &Coords, meta: CommonMeta) -> FailResult<impl DispFn>
        {Ok({
            match self.cfg.normals {
                cfg::KolmogorovCrespiNormals::Z {} => Box::new(self.optimized_disp_fn_for_z(coords, meta)?),
                cfg::KolmogorovCrespiNormals::LocalFast {} |
                cfg::KolmogorovCrespiNormals::LocalTrue {} => self._default_initialize_disp_fn(coords, meta)?,
            }
        })}

        fn optimized_disp_fn_for_z(&self, coords: &Coords, meta: CommonMeta) -> FailResult<impl DispFn>
        {Ok({
            trace!("Using optimized DispFn for KCZ");

            let mut diff = self._initialize_bond_diff_fn(coords, meta.clone())?;

            let canonical_bonds_by_site = {
                let frac_bonds = diff.compute_frac_bonds(coords, meta.clone())?;

                let mut canonical_bonds_by_site = vec![vec![]; coords.len()];
                for bond in frac_bonds {
                    if !bond.is_canonical() {
                        continue;
                    }

                    canonical_bonds_by_site[bond.from].push(bond);
                    // TODO: Test removing this condition on a small unit cell.
                    if bond.from != bond.to {
                        canonical_bonds_by_site[bond.to].push(bond);
                    }
                }
                canonical_bonds_by_site
            };
            let other = ZDispFnOther { canonical_bonds_by_site };

            DispFnHelper::new(coords, meta)
                .with_other(other)
                .build(diff)
        })}
    }

    impl disp_fn_helper::Callback<CommonMeta, ZDispFnOther> for Diff {
        fn compute_sparse_grad_delta(
            &mut self,
            context: disp_fn_helper::Context<'_, CommonMeta, ZDispFnOther>,
            disp: (usize, V3),
        ) -> FailResult<BTreeMap<usize, V3>> {
            // FIXME: This needs better documentation; I am confused how it works.
            // Why is it not necessary to subtract the corresponding bond grads at equilibrium?

            let filtered_bonds = context.other.canonical_bonds_by_site[disp.0].to_vec();

            let mut coords = context.equilibrium_coords.clone();
            coords.carts_mut()[disp.0] += disp.1;

            let (_, bond_grad) = compute_using_frac_bonds(
                self.parallel, &self.params,
                &coords, context.meta, &self.normal_info,
                filtered_bonds,
            )?;

            Ok(helper::sparse_grad_from_bond_grad(bond_grad))
        }
    }

    use normals::{Normals, NormalKind};
    mod normals {
        use super::*;
        use stack::{ArrayVec, Vector as _};

        pub struct Normals {
            data: Vec<NormalData>,
        }

        #[derive(Debug, Copy, Clone)]
        pub enum NormalKind {
            /// Local normals, as originally implemented in rsp2.
            ///
            /// This definition comes from a slight misinterpretation of the paper.
            /// It defines the local using ONLY the three neighbors. (it is the normal
            /// of the plane that those three neighbors co-inhabit)
            LocalFast,

            /// Local normals, as ACTUALLY defined in Kolmogorov and Crespi, 2005.
            ///
            /// The exact definition comes from this sentence in the paper:
            /// "For example, one can average the three normalized cross
            /// products of the displacement vectors to the nearest neighbors."
            Local,
        }

        /// Data for the normal at a single site.
        struct NormalData {
            normal: V3,
            /// Data for every bond (WITHIN a layer) that played a role in computing `normal`,
            /// so that derivatives with respect to the bonds can be recovered.
            ///
            /// The bonds don't need to be in canonical order.
            items: ArrayVec<[NormalItem; 3]>,
        }

        /// Describes the derivative of a single normal with respect to a single bond.
        struct NormalItem {
            bond: FracBond, // the virtual bond...
            r: V3, // ...in cartesian form...
            normal_J_r: M33, // ...and derivative with respect to it
        }

        impl Normals {
            pub fn compute(
                intralayer_graph: &PeriodicGraph,
                coords: &Coords,
                normal_kind: NormalKind,
            ) -> FailResult<Self> {
                assert_eq!(coords.len(), intralayer_graph.node_count());
                let lattice = coords.lattice();
                let carts = coords.to_carts();

                let data = {
                    (0..coords.len()).map(|site| {
                        match normal_kind {
                            NormalKind::LocalFast => {
                                let [b01, b02, b03] = Self::expect_three_bonds(intralayer_graph, site)?;
                                let bond_12 = b01.flip().join(b02).expect("(BUG) bond mismatch?");
                                let bond_13 = b01.flip().join(b03).expect("(BUG) bond mismatch?");

                                let r12 = bond_12.cart_vector_using_carts(lattice, &carts);
                                let r13 = bond_13.cart_vector_using_carts(lattice, &carts);

                                let (normal, (normal_J_r12, normal_J_r13)) = crespi_imp::unit_cross(r12, r13);

                                let mut items = ArrayVec::new();
                                items.push(NormalItem { bond: bond_12, r: r12, normal_J_r: normal_J_r12 });
                                items.push(NormalItem { bond: bond_13, r: r13, normal_J_r: normal_J_r13 });

                                Ok(NormalData { normal, items })
                            },
                            NormalKind::Local => {
                                let [bond_01, bond_02, bond_03] = Self::expect_three_bonds(intralayer_graph, site)?;
                                let r01 = bond_01.cart_vector_using_carts(lattice, &carts);
                                let r02 = bond_02.cart_vector_using_carts(lattice, &carts);
                                let r03 = bond_03.cart_vector_using_carts(lattice, &carts);

                                let (n012, (n012_J_r01, n012_J_r02)) = crespi_imp::unit_cross(r01, r02);
                                let (n023, (n023_J_r02, n023_J_r03)) = crespi_imp::unit_cross(r02, r03);
                                let (n031, (n031_J_r03, n031_J_r01)) = crespi_imp::unit_cross(r03, r01);

                                let nsum = n012 + n023 + n031;
                                let nsum_J_r01 = n012_J_r01 + n031_J_r01;
                                let nsum_J_r02 = n023_J_r02 + n012_J_r02;
                                let nsum_J_r03 = n031_J_r03 + n023_J_r03;

                                let (normal, normal_J_nsum) = crespi_imp::unit(nsum);
                                let normal_J_r01 = normal_J_nsum * nsum_J_r01;
                                let normal_J_r02 = normal_J_nsum * nsum_J_r02;
                                let normal_J_r03 = normal_J_nsum * nsum_J_r03;

                                let mut items = ArrayVec::new();
                                items.push(NormalItem { bond: bond_01, r: r01, normal_J_r: normal_J_r01 });
                                items.push(NormalItem { bond: bond_02, r: r02, normal_J_r: normal_J_r02 });
                                items.push(NormalItem { bond: bond_03, r: r03, normal_J_r: normal_J_r03 });

                                Ok(NormalData { normal, items })
                            },
                        }
                    }).collect::<FailResult<_>>()?
                };

                Ok(Normals { data })
            }

            /// Get FracBonds pointing from `site` to its neighbors, but fail if there are
            /// not exactly three neighbors.
            fn expect_three_bonds(intralayer_graph: &PeriodicGraph, site: usize) -> FailResult<[FracBond; 3]> {
                let mut edges = intralayer_graph.frac_bonds_from(site);
                match (edges.next(), edges.next(), edges.next(), edges.next()) {
                    (Some(b01), Some(b02), Some(b03), None) => Ok([b01, b02, b03]),
                    _ => bail!{
                        "\
                            The chosen 'kc.normals' setting requires all sites to have three neighbors; \
                            a site was found with {}.\
                        ", intralayer_graph.frac_bonds_from(site).count(),
                    },
                }
            }

            pub fn lookup(&self, index: usize) -> V3 {
                self.data[index].normal
            }

            pub fn get_bond_grad_terms(
                &self,
                site: usize,
                value_d_normal: V3,
            ) -> ArrayVec<[BondGrad; 3]> {
                self.data[site].items.iter().map(|item| BondGrad {
                    minus_site: item.bond.from,
                    plus_site: item.bond.to,
                    grad: value_d_normal * item.normal_J_r,
                    cart_vector: item.r,
                }).collect()
            }
        }
    }

    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    // FIXME: copy-pasta from KCZ in lammps.rs
    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    impl KolmogorovCrespi {
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
}

pub use rebo::Rebo;
mod rebo {
    use super::*;

    /// Rust implementation of REBO.
    ///
    /// NOTE: This has the limitation that the set of pairs within interaction range
    ///       must not change after the construction of the DiffFn.
    #[derive(Debug, Clone)]
    pub struct Rebo {
        pub(in crate::potential) cfg: cfg::PotentialReboNonreactive,
        pub(in crate::potential) parallel: bool,
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
                let cfg::PotentialReboNonreactive { params } = me.cfg;
                let params = match params {
                    cfg::PotentialReboNewParams::Lammps => rebo_imp::Params::new_lammps(),
                    cfg::PotentialReboNewParams::LammpsFavata => rebo_imp::Params::new_favata(),
                    cfg::PotentialReboNewParams::Brenner => rebo_imp::Params::new_brenner(),
                    cfg::PotentialReboNewParams::Lindsay => rebo_imp::Params::new_lindsay(),
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

                fn check(&mut self, coords: &Coords, _: CommonMeta) -> FailResult<()> {
                    self.interactions.check_distances(coords, self.parallel)
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
        let bonds: meta::FracBonds = std::rc::Rc::new(FracBonds::compute(&coords, 2.0)?);
        let meta = hlist![elements, masses, Some(bonds)];

    //    let pot_lmp = <dyn PotentialBuilder>::from_config_parts(None, None, &cfg::Threading::Serial, &cfg::LammpsUpdateStyle::Safe, &[true; 3], &cfg_lmp).allow_blocking(true);
        let pot_rsp2 = <dyn PotentialBuilder>::from_config_parts(None, None, &cfg::Threading::Serial, &cfg::LammpsUpdateStyle::Safe, &[true; 3], &cfg_rsp2).allow_blocking(true);

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
        let bonds: meta::FracBonds = std::rc::Rc::new(FracBonds::compute(&coords, 2.0)?);
        println!("{:?}", bonds);
        let meta = hlist![elements, masses, Some(bonds)];

        let pot_lmp = <dyn PotentialBuilder>::from_config_parts(None, None, &cfg::Threading::Serial, &cfg::LammpsUpdateStyle::Safe, &[true; 3], &cfg_lmp).allow_blocking(true);
        let pot_rsp2 = <dyn PotentialBuilder>::from_config_parts(None, None, &cfg::Threading::Serial, &cfg::LammpsUpdateStyle::Safe, &[true; 3], &cfg_rsp2).allow_blocking(true);

        let diff_lmp = pot_lmp.initialize_diff_fn(&coords, meta.sift())?.compute(&coords, meta.sift())?;
        let diff_rsp2 = pot_rsp2.initialize_diff_fn(&coords, meta.sift())?.compute(&coords, meta.sift())?;

        println!("{:?}", diff_lmp);
        println!("{:?}", diff_rsp2);
        assert_close!(diff_lmp.0, diff_rsp2.0);
        assert_close!(diff_lmp.1.unvee(), diff_rsp2.1.unvee());
        Ok(())
    }
}
