/* ************************************************************************ **
** This file is part of rsp2, and is licensed under EITHER the MIT license  **
** or the Apache 2.0 license, at your option.                               **
**                                                                          **
**     http://www.apache.org/licenses/LICENSE-2.0                           **
**     http://opensource.org/licenses/MIT                                   **
**                                                                          **
** Be aware that not all of rsp2 is provided under this permissive license, **
** and that the project as a whole is licensed under the GPL 3.0.           **
** ************************************************************************ */

#![allow(non_snake_case)]
#![allow(non_shorthand_field_patterns)]
#![allow(unused_imports)] // FIXME REMOVE
#![allow(dead_code)] // FIXME REMOVE
#![allow(unreachable_code)] // FIXME REMOVE

//! Implementation of the second-generation REBO reactive potential.
//!
//! The second-generation REBO paper contains many errors; this implementation was written
//! with the help of the AIREBO paper, and by reviewing the implementation in LAMMPS.
//!
//! # Citations
//!
//! * **2nd gen REBO:** Donald W Brenner et al 2002 J. Phys.: Condens. Matter 14 783
//! * **AIREBO:** Steven J Stuart et al J. Chem. Phys. 112, 6472 (2000)
//! * **LAMMPS:** S. Plimpton, J Comp Phys, 117, 1-19 (1995)

use ::FailResult;
use ::math::bond_graph::PeriodicGraph;
use ::math::bonds::FracBond;
use ::meta;

use ::stack::{ArrayVec, Vector as StackVector};
#[cfg(test)]
use ::std::f64::{consts::PI};
use ::std::f64::NAN;
use ::std::borrow::Cow;
use ::rsp2_array_types::{V2, V3, M33, M3};
use ::rsp2_structure::Coords;
#[allow(unused)] // https://github.com/rust-lang/rust/issues/45268
use ::rsp2_newtype_indices::{Idx, IndexVec, Indexed, self as idx};
#[allow(unused)] // https://github.com/rust-lang/rust/issues/45268
use ::petgraph::prelude::EdgeRef;
use ::enum_map::EnumMap;
use ::rayon::prelude::*;
use ::slice_of_array::prelude::*;

#[cfg(test)]
use ::rsp2_minimize::numerical::{self, DerivativeKind::*};

//-------------------------------------------------------
// # A note on stylistic conventions
//
// Although removing the reactive parts of the potential has drastically simplified
// the computation of derivatives to the point where most of the code can be written
// with a fairly relaxed attitude, this file still rigidly adheres to a couple of rules:
//
// 1. The derivative of `foo` with respect to `bar` is systematically named
//    `foo_d_bar`. It does not matter if `bar` is a vector (in which case the
//    derivative is a gradient) or if `foo` is a vector, because in practice
//    this is obvious from context and it's hard to use them incorrectly.
//
//    An exception is made for the case where `foo` and `bar` are BOTH vectors;
//    search this file for the word "Jacobian".
//
//    Generally speaking, `foo_d_bar * bar_d_baz = foo_d_baz`, assuming the
//    multiplication operation is supported by those types.
//
// 2. When inputs and outputs of a function can be ambiguous, wrap in a module
//    to define input/output types with named members.
//
// 3. Fully destructure the output of any function that computes pieces of the potential,
//    so that the compiler complains about missing fields when one is added.
//
//-------------------------------------------------------

/// A stack-allocated vector of data whose indices correspond to the elements
/// returned by `interactions.bonds(site)` for some particular `site`, in order.
pub type SiteBondVec<T> = ArrayVec<[T; SITE_MAX_BONDS]>;

/// The most bonds an atom can have.
pub const SITE_MAX_BONDS: usize = 4;
pub const BOND_MAX_DIHEDRALS: usize = (SITE_MAX_BONDS - 1) * (SITE_MAX_BONDS - 1);

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, enum_map::Enum)]
pub enum AtomType { Carbon, Hydrogen }
impl AtomType {
    pub fn char(self) -> char {
        match self {
            AtomType::Carbon => 'C',
            AtomType::Hydrogen => 'H',
        }
    }

    pub fn from_element(elem: meta::Element) -> FailResult<Self> {
        use ::rsp2_structure::consts;
        match elem {
            consts::CARBON => Ok(AtomType::Carbon),
            consts::HYDROGEN => Ok(AtomType::Hydrogen),
            _ => bail!("REBO only supports Carbon and Hydrogen"),
        }
    }

    pub fn iter_all() -> impl Iterator<Item=AtomType> {
        ::std::iter::once(AtomType::Carbon).chain(::std::iter::once(AtomType::Hydrogen))
    }
}
type TypeMap<T> = EnumMap<AtomType, T>;

newtype_index!{SiteI}
newtype_index!{BondI}

pub use self::params::Params;
mod params {
    use super::*;

    // TODO: Use LAMMPS' parameters
    #[derive(Debug, Clone)]
    pub struct Params {
        pub by_type: TypeMap<TypeMap<TypeParams>>,
        pub G: Cow<'static, g_spline::SplineSet>,
        pub T: Cow<'static, t_spline::SplineSet>,
        pub F: Cow<'static, f_spline::SplineSet>,
        pub P: Cow<'static, p_spline::SplineSet>,
        pub use_airebo_lambda: bool,
    }

    #[derive(Debug, Copy, Clone)]
    pub struct TypeParams {
        pub B: [f64; 3], // eV
        pub beta: [f64; 3], // Å-1
        pub Q: f64, // Å
        pub A: f64, // eV
        pub alpha: f64, // Å-1
        pub Dmin: f64, // Å
        pub Dmax: f64, // Å
    }

    impl Params {
        /// Parameters published in the original REBO paper.
        pub fn new_brenner() -> Self {
            // Brenner Table 2
            let type_params_cc = TypeParams {
                B: [
                    12388.791_977_98, // eV
                    17.567_406_465_09, // eV
                    30.714_932_080_65, // eV
                ],
                beta: [
                    4.720_452_3127, // Å-1
                    1.433_213_2499, // Å-1
                    1.382_691_2506, // Å-1
                ],
                Q: 0.313_460_296_0833, // Å
                A: 10953.544_162_170, // eV
                alpha: 4.746_539_060_6595, // Å-1
                Dmin: 1.7, // Å
                Dmax: 2.0, // Å
            };

            // Brenner Table 6
            let type_params_hh = TypeParams {
                B: [29.632_593, 0.0, 0.0], // eV
                beta: [1.715_892_17, 0.0, 0.0], // Å-1
                Q: 0.370_471_487_045, // Å
                A: 32.817_355_747, // Å
                alpha: 3.536_298_648, // Å-1
                Dmin: 1.1, // Å
                Dmax: 1.7, // Å
            };

            // Brenner Table 7
            let type_params_ch = TypeParams {
                B: [32.355_186_6587, 0.0, 0.0], // eV
                beta: [1.434_458_059_25, 0.0, 0.0], // Å-1
                Q: 0.340_775_728, // Å
                A: 149.940_987_23, // eV
                alpha: 4.102_549_83, // Å-1
                Dmin: 1.3, // Å
                Dmax: 1.8, // Å
            };

            let by_type = enum_map! {
                AtomType::Carbon => enum_map! {
                    AtomType::Carbon => type_params_cc,
                    AtomType::Hydrogen => type_params_ch,
                },
                AtomType::Hydrogen => enum_map! {
                    AtomType::Carbon => type_params_ch,
                    AtomType::Hydrogen => type_params_hh,
                },
            };
            Params {
                use_airebo_lambda: false,
                G: Cow::Owned(g_spline::BRENNER_SPLINES),
                P: Cow::Borrowed(&p_spline::BRENNER_SPLINES),
                T: Cow::Borrowed(&t_spline::BRENNER_SPLINES),
                F: Cow::Borrowed(&f_spline::BRENNER_SPLINES),
                by_type,
            }
        }

        /// Parameters consistent with LAMMPS' `Airebo.CH`, for comparison purposes.
        ///
        /// The HH parameters have changed a fair bit compared to the parameters
        /// originally published in Brenner's paper.
        ///
        /// The G splines (functions of bond angle) are also significantly different,
        /// and quite possibly less precise.
        pub fn new_lammps() -> Self {
            // Brenner Table 2
            let type_params_cc = TypeParams {
                B: [
                    12388.791_977_983_75, // eV
                    17.567_406_465_089_68, // eV
                    30.714_932_080_651_62, // eV
                ],
                beta: [
                    4.720_452_312_717_397, // Å-1
                    1.433_213_249_951_261, // Å-1
                    1.382_691_250_599_169, // Å-1
                ],
                Q: 0.313_460_296_083_2605, // Å
                A: 10953.544_162_169_92, // eV
                alpha: 4.746_539_060_659_529, // Å-1
                Dmin: 1.7, // Å
                Dmax: 2.0, // Å
            };

            // Brenner Table 6
            let type_params_hh = TypeParams {
                B: [28.2297, 0.0, 0.0], // eV
                beta: [1.708, 1.0, 1.0], // Å-1
                Q: 0.370, // Å
                A: 31.6731, // Å
                alpha: 3.536, // Å-1
                Dmin: 1.1, // Å
                Dmax: 1.7, // Å
            };

            // Brenner Table 7
            let type_params_ch = TypeParams {
                B: [32.355_186_658_732_56, 0.0, 0.0], // eV
                beta: [1.434_458_059_249_837, 0.0, 0.0], // Å-1
                Q: 0.340_775_728_225_7080, // Å
                A: 149.940_987_228_812, // eV
                alpha: 4.102_549_828_548_784, // Å-1
                Dmin: 1.3, // Å
                Dmax: 1.8, // Å
            };

            let by_type = enum_map! {
                AtomType::Carbon => enum_map! {
                    AtomType::Carbon => type_params_cc,
                    AtomType::Hydrogen => type_params_ch,
                },
                AtomType::Hydrogen => enum_map! {
                    AtomType::Carbon => type_params_ch,
                    AtomType::Hydrogen => type_params_hh,
                },
            };
            Params {
                use_airebo_lambda: true,
                G: Cow::Owned(g_spline::LAMMPS_SPLINES),
                P: Cow::Borrowed(&p_spline::FAVATA_SPLINES),
                F: Cow::Borrowed(&f_spline::BRENNER_SPLINES),
                T: Cow::Borrowed(&t_spline::STUART_SPLINES),
                by_type,
            }
        }
    }
}

use self::interactions::Interactions;
mod interactions {
    use super::*;

    // collects all the terms we need to compute
    //
    // TODO: Use me!
    pub struct Interactions {
        /// CSR-style divider indices for bonds at each site.
        bond_div: IndexVec<SiteI, BondI>,

        site_type: IndexVec<SiteI, AtomType>,
        bond_cart_vector: IndexVec<BondI, V3>,
        bond_image_diff: IndexVec<BondI, V3<i32>>,
        bond_reverse_index: IndexVec<BondI, BondI>,
        bond_is_canonical: IndexVec<BondI, bool>,
        bond_source: IndexVec<BondI, SiteI>,
        bond_target: IndexVec<BondI, SiteI>,
    }

    impl Interactions {
        pub fn compute(
            coords: &Coords,
            types: &[AtomType],
            bond_graph: &PeriodicGraph,
        ) -> FailResult<Self> {
            let mut bond_div = IndexVec::<SiteI, _>::from_raw(vec![BondI(0)]);
            let mut bond_cart_vector = IndexVec::<BondI, _>::new();
            let mut bond_is_canonical = IndexVec::<BondI, _>::new();
            let mut bond_source = IndexVec::<BondI, SiteI>::new();
            let mut bond_target = IndexVec::<BondI, SiteI>::new();
            let mut bond_image_diff = IndexVec::<BondI, V3<i32>>::new();
            let site_type = IndexVec::<SiteI, _>::from_raw(types.to_vec());

            let cart_cache = coords.with_carts(coords.to_carts());

            // Make a pass to get all the bond divs right.
            for node in bond_graph.node_indices() {
                let site_i = SiteI(node.index());

                for frac_bond_ij in bond_graph.frac_bonds_from(site_i.index()) {
                    let site_j = SiteI(frac_bond_ij.to);
                    let cart_vector = frac_bond_ij.cart_vector_using_cache(&cart_cache).unwrap();
                    bond_source.push(site_i);
                    bond_target.push(site_j);
                    bond_is_canonical.push(frac_bond_ij.is_canonical());
                    bond_cart_vector.push(cart_vector);
                    bond_image_diff.push(frac_bond_ij.image_diff);
                } // for bond_ij

                let num_bonds = bond_target.len() - bond_div.raw.last().unwrap().index();
                if num_bonds > SITE_MAX_BONDS {
                    bail!("An atom has too many bonds! ({}, max: {})", num_bonds, SITE_MAX_BONDS);
                }
                bond_div.push(BondI(bond_target.len()));
            } // for node

            // Get the index of each bond's reverse.
            let mut bond_reverse_index = IndexVec::<BondI, _>::new();
            for node in bond_graph.node_indices() {
                let site_i = SiteI(node.index());

                for frac_bond_ij in bond_graph.frac_bonds_from(site_i.index()) {
                    let site_j = SiteI(frac_bond_ij.to);
                    let index_ji = {
                        bond_graph.frac_bonds_from(site_j.index())
                            .position(|bond| frac_bond_ij == bond.flip())
                    };
                    let index_ji = match index_ji {
                        Some(x) => x,
                        None => bail!("A bond has no counterpart in the reverse direction!"),
                    };
                    let bond_ji = BondI(bond_div[site_j].0 + index_ji);
                    bond_reverse_index.push(bond_ji);
                } // for bond_ij
            } // for node

            Ok(Interactions {
                bond_div, site_type, bond_cart_vector, bond_is_canonical,
                bond_target, bond_image_diff, bond_reverse_index, bond_source,
            })
        }
    }

    pub struct Site {
        pub index: SiteI,
        pub atom_type: AtomType,
    }

    pub struct Bond {
        pub index: BondI,
        pub is_canonical: bool,
        pub source: SiteI,
        pub target: SiteI,
        pub cart_vector: V3,
        pub image_diff: V3<i32>,
        pub reverse_index: BondI,
    }

    impl Interactions {
        pub fn num_sites(&self) -> usize { self.site_type.len() }
        pub fn num_bonds(&self) -> usize { self.bond_target.len() }

        #[inline(always)] // hopefully eliminate unused lookups to waste less cache
        pub fn site(&self, index: SiteI) -> Site {
            let atom_type = self.site_type[index];
            Site { index, atom_type }
        }

        #[inline(always)] // hopefully eliminate unused lookups to waste less cache
        pub fn bond(&self, index: BondI) -> Bond {
            let is_canonical = self.bond_is_canonical[index];
            let cart_vector = self.bond_cart_vector[index];
            let source = self.bond_source[index];
            let target = self.bond_target[index];
            let image_diff = self.bond_image_diff[index];
            let reverse_index = self.bond_reverse_index[index];
            Bond { index, is_canonical, cart_vector, target, reverse_index, image_diff, source }
        }

        pub fn sites(&self) -> impl ExactSizeIterator<Item=Site> + '_ {
            (0..self.site_type.len()).map(move |i| self.site(SiteI(i)))
        }

        pub fn site_range(&self) -> ::std::ops::Range<usize> {
            0..self.site_type.len()
        }

        pub fn frac_bond(&self, bond: BondI) -> FracBond {
            let bond = self.bond(bond);
            FracBond {
                from: bond.source.0,
                to: bond.target.0,
                image_diff: bond.image_diff,
            }
        }

        pub fn bonds(&self, site: SiteI) -> impl ExactSizeIterator<Item=Bond> + '_ {
            self.bond_range(site).map(move |i| self.bond(BondI(i)))
        }

        pub fn bond_range(&self, site: SiteI) -> ::std::ops::Range<usize> {
            self.bond_div[site].0..self.bond_div[site.next()].0
        }
    }
}

pub fn compute(
    params: &Params,
    coords: &Coords,
    elements: &[meta::Element],
    bonds: &PeriodicGraph,
) -> FailResult<(f64, Vec<V3>)> {
    let types = elements.iter().cloned().map(AtomType::from_element).collect::<FailResult<Vec<_>>>()?;
    let interactions = Interactions::compute(coords, &types, bonds)?;
    let (value, d_deltas) = compute_rebo_bonds(params, &interactions)?;

    let mut d_positions = IndexVec::from_elem_n(V3::zero(), interactions.num_sites());
    for __site in interactions.sites() {
        let site_i = __site.index;

        for __bond in interactions.bonds(site_i) {
            let bond_ij = __bond.index;
            let site_j = __bond.target;

            // delta_ij = (-pos_i) + pos_j
            d_positions[site_i] -= d_deltas[bond_ij];
            d_positions[site_j] += d_deltas[bond_ij];
        }
    }
    Ok((value, d_positions.raw))
}

fn compute_rebo_bonds(
    params: &Params,
    interactions: &Interactions,
) -> FailResult<(f64, IndexVec<BondI, V3>)> {
    use self::interactions::{Site, Bond};
    // Brenner:
    // Eq  1:  V = sum_{i < j} V^R(r_ij) - b_{ij} V^A(r_ij)
    // Eq  5:  V^R(r) = f(r) (1 + Q/r) A e^{-alpha r}
    // Eq  6:  V^A(r) = f(r) sum_{n in 1..=3} B_n e^{-beta_n r}
    // Eq  3:  b_{ij} = 0.5 * (b_{ij}^{sigma-pi} + b_{ji}^{sigma-pi}) + b_ij^pi
    // Eq  4:  b_{ij}^{pi} = PI_{ij}^{RC} + b_{ij}^{DH}
    // Eq 14:  PI_{ij}^{RC} = F spline
    //
    // r_ij are bond vectors
    // f_ij is bond weight (0 to 1)
    // Q, A, alpha, beta_n, B_n are parameters
    // b_{ij}^{pi}, b_{ij}^{sigma-pi}, b_{ij}^{DH} are complicated bond-order subexpressions

    //-------------------
    // NOTE:
    //
    // We will also define U_ij (and U^A_ij, and U^R_ij) to be the V terms without the
    // f_ij scale factor.
    //
    // Eq  5':  U^R(r) = (1 + Q/r) A e^{-alpha r}
    // Eq  6':  U^A(r) = sum_{n in 1..=3} B_n e^{-beta_n r}
    //
    // We also redefine the sums in the potential to be over all i,j pairs, not just i < j.
    //
    // Eq 1':     V = sum_{i != j} V_ij
    // Eq 2':  V_ij = 0.5 * V^R_ij - b_ij * V^A_ij
    // Eq 3':  b_ij = 0.5 * b_ij^{sigma-pi} + boole(i < j) * b_ij^{pi}

    // On large systems, our performance is expected to be bounded by cache misses.
    // For this reason, we should aim to make as few passes over the data as necessary,
    // leaving vectorization as only a secondary concern.
    struct FirstPassSiteData {
        // Brenner's N_i
        tcoord: u32,
        // Brenner's f_ij.
        //
        // We keep these around because they may be zero. (even for nonreactive REBO,
        // we use a bond search radius that is large enough to include the point of
        // zero weight so that we can detect if a fractional weight ever appears and
        // bail out)
        bond_weight: SiteBondVec<u32>,
        // Brenner's V^R(r_ij)
        bond_VR: SiteBondVec<f64>,
        bond_VR_d_delta: SiteBondVec<V3>,
        // Brenner's V^A(r_ij)
        bond_VA: SiteBondVec<f64>,
        bond_VA_d_delta: SiteBondVec<V3>,
    }

    let site_data = IndexVec::<SiteI, _>::from_raw({
        interactions.site_range().into_par_iter().map(SiteI::new).map(|site_i| {
            let __site = interactions.site(site_i);
            let type_i = __site.atom_type;

            let mut tcoord = 0;
            let mut bond_VR = SiteBondVec::new();
            let mut bond_VR_d_delta = SiteBondVec::new();
            let mut bond_VA = SiteBondVec::new();
            let mut bond_VA_d_delta = SiteBondVec::new();
            let mut bond_weight = SiteBondVec::new();

            for __bond in interactions.bonds(site_i) {
                let site_j = __bond.target;
                let delta_ij = __bond.cart_vector;
                let type_j = interactions.site(site_j).atom_type;
                let params_ij = params.by_type[type_i][type_j];

                let (length, length_d_delta) = norm(delta_ij);
                let (weight, weight_d_length) = switch((params_ij.Dmax, params_ij.Dmin), length);

                ensure!(
                    weight_d_length == 0.0 && (weight == 1.0 || weight == 0.0),
                    "detected reaction in non-reactive REBO potential"
                );
                tcoord += weight as u32;
                bond_weight.push(weight as u32);

                // these also depend only on bond length
                // (and also conveniently don't depend on anything else)

                let VR;
                let VR_d_length;
                {
                    // UR_ij = (1 + Q/r_ij) A e^{-alpha r_ij}
                    // write as a product of two subexpressions

                    let sub1 = 1.0 + params_ij.Q / length;
                    let sub1_d_length = - params_ij.Q / (length * length);

                    let sub2 = params_ij.A * f64::exp(-params_ij.alpha * length);
                    let sub2_d_length = -params_ij.alpha * sub2;

                    let UR = sub1 * sub2;
                    let UR_d_length = sub1_d_length * sub2 + sub1 * sub2_d_length;

                    // VR_ij = f_ij UR_ij
                    VR = weight * UR;
                    VR_d_length = weight_d_length * UR + weight * UR_d_length;
                }
                println!("rs-VR: {}", VR);
                bond_VR.push(VR);
                bond_VR_d_delta.push(VR_d_length * length_d_delta);

                let VA;
                let VA_d_length;
                {
                    // UA_ij = sum_{n in 1..=3} B_n e^{-beta_n r_ij}
                    let mut UA = 0.0;
                    let mut UA_d_length = 0.0;
                    for (&B, &beta) in zip_eq!(&params_ij.B, &params_ij.beta) {
                        let term = B * f64::exp(-beta * length);
                        let term_d_length = -beta * term;
                        UA += term;
                        UA_d_length += term_d_length;
                    }

                    // VA_ij = f_ij UA_ij
                    VA = weight * UA;
                    VA_d_length = weight_d_length * UA + weight * UA_d_length;
                }
                println!("rs-VA: {}", VA);
                bond_VA.push(VA);
                bond_VA_d_delta.push(VA_d_length * length_d_delta);
            } // for _ in interactions.bonds(site)

            Ok(FirstPassSiteData {
                tcoord,
                bond_weight,
                bond_VR, bond_VR_d_delta,
                bond_VA, bond_VA_d_delta,
                // NOTE: weight_d_length is now dropped from consideration,
                //       meaning the rest of the code only models a nonreactive
                //       potential
            })
        }).collect::<FailResult<_>>()?
    });

    let out = interactions.site_range().into_par_iter().map(SiteI::new).map(|site_i| {
        let __site = interactions.site(site_i);
        let FirstPassSiteData {
            tcoord: _,
            ref bond_weight,
            ref bond_VR, ref bond_VR_d_delta,
            ref bond_VA, ref bond_VA_d_delta,
        } = site_data[site_i];

        // Eq 2':  V_ij = 0.5 * V^R_ij - b_ij * V^A_ij
        let mut site_V = 0.0;

        // derivatives with respect to the deltas for site i
        let mut site_V_d_delta = sbvec_filled(V3::zero(), bond_weight.len());

        // derivatives with respect to the deltas for each site j.
        // parallel code can't accumulate these into any large sort of vector, so we build
        // a list of terms to be added in a short serial segment at the end.
        let mut site_V_d_other_deltas = SiteBondVec::<(SiteI, SiteBondVec<V3>)>::new();

        //-----------------------------------------------
        // The repulsive terms
        //-----------------------------------------------

        // Add in the repulsive terms, which each only depend on one bond delta.
        site_V += 0.5 * bond_VR.iter().sum::<f64>();
        axpy_mut(&mut site_V_d_delta, 0.5, &bond_VR_d_delta);

        //-----------------------------------------------
        // The sigma-pi terms
        //-----------------------------------------------

        // sigma-pi terms are present for all bonds, regardless of direction.
        //
        // This is a sum of `0.5 * V^A_ij * b_ij^{sigma-pi}` over all of the bonds at site i.
        let out = site_sigma_pi_term::Input {
            params,
            interactions,
            site: site_i,
            bond_weights: bond_weight,
            bond_VAs: bond_VA,
            bond_VAs_d_delta: bond_VA_d_delta,
        }.compute();
        let SiteSigmaPiTerm {
            value: Vsp_i,
            d_deltas: Vsp_i_d_deltas,
        } = out;

        site_V -= Vsp_i;
        axpy_mut(&mut site_V_d_delta, -1.0, &Vsp_i_d_deltas);

        //-----------------------------------------------
        // The pi terms
        //-----------------------------------------------

        // These are the only parts that depend on other sites' bond deltas.

        // Eq 3':  b_ij = boole(i < j) * b_ij^{pi}
        for (index_ij, __bond) in interactions.bonds(site_i).enumerate() {
            if !__bond.is_canonical {
                continue;
            }
            let bond_ij = __bond.index;
            let site_j = __bond.target;

            let ref weights_ik = bond_weight;
            let ref weights_jl = site_data[site_j].bond_weight;

            let ref tcoords_k: SiteBondVec<_> = interactions.bonds(site_i).map(|bond| site_data[bond.target].tcoord).collect();
            let ref tcoords_l: SiteBondVec<_> = interactions.bonds(site_j).map(|bond| site_data[bond.target].tcoord).collect();

            let out = bondorder_pi::Input {
                params, interactions, site_i, bond_ij,
                tcoords_k, tcoords_l, weights_ik, weights_jl,
            }.compute();
            let BondOrderPi {
                value: bpi,
                d_deltas_ik: mut bpi_d_deltas_ik,
                d_deltas_jl: mut bpi_d_deltas_jl,
            } = out;
            println!("rs-bpi: {}", bpi);

            let VA_ij = bond_VA[index_ij];
            let VA_ij_d_delta_ij = bond_VA_d_delta[index_ij];

            site_V += bpi * VA_ij;
            site_V_d_delta[index_ij] += bpi * VA_ij_d_delta_ij;

            axpy_mut(&mut site_V_d_delta, VA_ij, &bpi_d_deltas_ik);
            site_V_d_other_deltas.push((site_j, sbvec_scaled(VA_ij, bpi_d_deltas_jl)));
        }

        (site_V, site_V_d_delta, site_V_d_other_deltas)

    // well this is awkward
    }).fold(
        || (0.0, IndexVec::new(), vec![]),
        |
            (mut value, mut d_deltas, mut d_other_deltas),
            (site_V, site_V_d_delta, site_V_d_other_deltas),
        | {
            value += site_V;
            d_deltas.extend(site_V_d_delta);
            d_other_deltas.extend(site_V_d_other_deltas);
            (value, d_deltas, d_other_deltas)
        },
    ).reduce(
        || (0.0, IndexVec::new(), vec![]),
        |
            (mut value, mut d_deltas, d_other_deltas),
            (value_part, d_delta_part, d_other_deltas_part),
        | {
            value += value_part;
            d_deltas.extend(d_delta_part); // must be concatenated in order
            let d_other_deltas = concat_any_order(d_other_deltas, d_other_deltas_part);
            (value, d_deltas, d_other_deltas)
        },
    );
    let (value, mut d_deltas, d_other_deltas) = out;

    assert_eq!(d_deltas.len(), interactions.num_bonds());

    // absorb the other terms we couldn't take care of into d_deltas
    // and end this miserable function for good
    for (site_i, d_deltas_ij) in d_other_deltas {
        let _: SiteBondVec<V3> = d_deltas_ij;
        axpy_mut(&mut d_deltas.raw[interactions.bond_range(site_i)], 1.0, &d_deltas_ij);
    }

    assert_eq!(d_deltas.len(), interactions.num_bonds());
    Ok((value, d_deltas))
}

use self::site_sigma_pi_term::SiteSigmaPiTerm;
mod site_sigma_pi_term {
    //! Represents the sum of `0.5 * b_{ij}^{sigma-pi} * VR` over all `j` for a given `i`.
    //!
    //! This quantity is useful to consider in its own right because it encapsulates
    //! the need for the P spline values (only two of which are needed per site),
    //! and it only has derivatives with respect to the bond vectors of site `i`;
    //! these properties give it a fairly simple signature.

    use super::*;

    pub type Output = SiteSigmaPiTerm;
    pub struct Input<'a> {
        pub params: &'a Params,
        pub interactions: &'a Interactions,
        pub site: SiteI,
        pub bond_weights: &'a [u32],
        // The VA_ij terms for each bond at site i.
        pub bond_VAs: &'a SiteBondVec<f64>,
        pub bond_VAs_d_delta: &'a SiteBondVec<V3>,
    }
    pub struct SiteSigmaPiTerm {
        pub value: f64,
        /// Derivatives with respect to the bonds listed in order of `interactions.bonds(site_i)`.
        pub d_deltas: SiteBondVec<V3>,
    }

    impl<'a> Input<'a> {
        pub fn compute(self) -> Output { compute(self) }
    }

    // free function for smaller indent
    fn compute(input: Input<'_>) -> Output {
        // Eq 8:  b_{ij}^{sigma-pi} = sqrt(
        //                     1 + sum_{k /= i, j} f^c(r_{ik}) G(cos(t_{ijk}) e^{lambda_{ijk}
        //                       + P_{ij}(N_ij^C, N_ij^H)
        //        )
        let Input {
            params, interactions, bond_weights,
            bond_VAs, bond_VAs_d_delta,
            site: site_i,
        } = input;
        let type_i = interactions.site(site_i).atom_type;

        // Tally up data about the bonds
        let mut type_present = enum_map!{_ => false};
        let mut ccoord_i = 0;
        let mut hcoord_i = 0;
        let mut bond_target_types = SiteBondVec::<AtomType>::new();
        for (bond, &weight) in zip_eq!(interactions.bonds(site_i), bond_weights) {
            let target_type = interactions.site(bond.target).atom_type;
            match target_type {
                AtomType::Carbon => ccoord_i += weight,
                AtomType::Hydrogen => hcoord_i += weight,
            }
            bond_target_types.push(target_type);
            type_present[target_type] = true;
        }

        // Handle all terms
        let mut value = 0.0;
        let mut d_deltas: SiteBondVec<V3> = sbvec_filled(V3::zero(), bond_weights.len());

        for (index_ij, __bond) in interactions.bonds(site_i).enumerate() {
            let type_j = bond_target_types[index_ij];
            let delta_ij = __bond.cart_vector;
            let weight_ij = bond_weights[index_ij];

            // These are what Brenner's Ni REALLY are.
            let ccoord_ij = ccoord_i - boole(type_j == AtomType::Carbon) as u32 * weight_ij;
            let hcoord_ij = hcoord_i - boole(type_j == AtomType::Hydrogen) as u32 * weight_ij;

            let P_ij = p_spline::Input { params, type_i, type_j, ccoord_ij, hcoord_ij }.compute();

            // Gather all cosines between bond i->j and other bonds i->k.
            let mut coses_ijk = SiteBondVec::new();
            let mut coses_ijk_d_delta_ij = SiteBondVec::new();
            let mut coses_ijk_d_delta_ik = SiteBondVec::new();
            for (index_ik, __bond) in interactions.bonds(site_i).enumerate() {
                let delta_ik = __bond.cart_vector;
                if index_ij == index_ik {
                    // set up bombs in case of possible misuse
                    coses_ijk.push(NAN);
                    coses_ijk_d_delta_ij.push(V3::from_fn(|_| NAN));
                    coses_ijk_d_delta_ik.push(V3::from_fn(|_| NAN));
                } else {
                    let out = bond_cosine::Input { delta_ij, delta_ik }.compute();
                    let BondCosine {
                        value: cos,
                        d_delta_ij: cos_d_delta_ij,
                        d_delta_ik: cos_d_delta_ik,
                    } = out;
                    coses_ijk.push(cos);
                    coses_ijk_d_delta_ij.push(cos_d_delta_ij);
                    coses_ijk_d_delta_ik.push(cos_d_delta_ik);
                }
            }

            // We're finally ready to compute the bond order.

            let bsp_ij;
            let bsp_ij_d_deltas;
            {
                // Compute bsp as a function of many things...
                let out = bondorder_sigma_pi::Input {
                    params,
                    type_i, type_j, ccoord_ij, hcoord_ij, P_ij,
                    coses_ijk: &coses_ijk,
                    types_k: &bond_target_types,
                    weights_ik: bond_weights,
                    skip_index: index_ij, // used to exclude the ijj angle
                }.compute();
                let BondOrderSigmaPi {
                    value: tmp_value,
                    d_coses_ijk: bsp_ij_d_coses_ijk,
                } = out;

                // ...and now reformulate it as a function solely of the deltas.
                let mut tmp_d_deltas: SiteBondVec<V3> = sbvec_filled(V3::zero(), bond_weights.len());

                // Some derivatives also come from the ik bonds.
                for (index_ik, bsp_ij_d_cos_ijk) in bsp_ij_d_coses_ijk.into_iter().enumerate() {
                    // Mind the gap
                    if index_ij == index_ik {
                        continue;
                    }
                    let cos_ijk_d_delta_ij = coses_ijk_d_delta_ij[index_ik];
                    let cos_ijk_d_delta_ik = coses_ijk_d_delta_ik[index_ik];

                    tmp_d_deltas[index_ij] += bsp_ij_d_cos_ijk * cos_ijk_d_delta_ij;
                    tmp_d_deltas[index_ik] += bsp_ij_d_cos_ijk * cos_ijk_d_delta_ik;
                }

                bsp_ij = tmp_value;
                bsp_ij_d_deltas = tmp_d_deltas;
                println!("rs-bsp: {}", bsp_ij);
            }

            // True term to add to sum is 0.5 * VR_ij * bsp_ij
            let VA_ij = bond_VAs[index_ij];
            let VA_ij_d_delta_ij = bond_VAs_d_delta[index_ij];

            value += 0.5 * VA_ij * bsp_ij;
            d_deltas[index_ij] += 0.5 * VA_ij_d_delta_ij * bsp_ij;
            axpy_mut(&mut d_deltas, 0.5 * VA_ij, &bsp_ij_d_deltas);
        }
        Output { value, d_deltas }
    }
}

use self::bondorder_sigma_pi::BondOrderSigmaPi;
mod bondorder_sigma_pi {
    use super::*;

    pub type Output = BondOrderSigmaPi;
    pub struct Input<'a> {
        pub params: &'a Params,
        // bond from atom i to another atom j
        pub type_i: AtomType,
        pub type_j: AtomType,
        pub ccoord_ij: u32,
        pub hcoord_ij: u32,
        // precomputed spline that depends on the coordination at i and the atom type at j
        pub P_ij: f64,
        // cosines of this bond with every other bond at i, and their weights
        pub types_k: &'a [AtomType],
        pub weights_ik: &'a [u32],
        pub coses_ijk: &'a [f64], // cosine between i->j and i->k
        // one of the items in the arrays is the ij bond (we must ignore it)
        pub skip_index: usize,
    }
    pub struct BondOrderSigmaPi {
        pub value: f64,
        // values at skip_index are unspecified
        pub d_coses_ijk: SiteBondVec<f64>,
    }

    impl<'a> Input<'a> {
        pub fn compute(self) -> Output { compute(self) }
    }

    // free function for smaller indent
    fn compute<'a>(input: Input<'a>) -> Output {
        // Eq 8:  b_{ij}^{sigma-pi} = sqrt(
        //                     1 + sum_{k /= i, j} f^c(r_{ik}) G(cos(t_{ijk}) e^{lambda_{ijk}
        //                       + P_{ij}(N_i^C, N_i^H)
        //        )
        let Input {
            params, type_i, type_j, ccoord_ij, hcoord_ij, P_ij,
            types_k, weights_ik, coses_ijk, skip_index,
        } = input;
        let tcoord_ij = ccoord_ij + hcoord_ij;

        // properties of the stuff in the square root
        let mut inner_value = 0.0;
        let mut inner_d_coses_ijk = SiteBondVec::new();

        // 1 + P_{ij}(N_i^C, N_i^H)
        //   + sum_{k /= i, j} e^{\lambda_{ijk}} f^c(r_{ik}) G(cos(t_{ijk})
        inner_value += 1.0;
        inner_value += P_ij;

        let iter = zip_eq!(coses_ijk, types_k).enumerate();
        for (index_ik, (&cos_ijk, &type_k)) in iter {
            let weight_ik = weights_ik[index_ik] as f64;
            if weight_ik == 0.0 {
                inner_d_coses_ijk.push(0.0);
                continue;
            }
            if index_ik == skip_index {
                inner_d_coses_ijk.push(NAN);
                continue;
            }

            let exp_lambda = match params.use_airebo_lambda {
                true => airebo_exp_lambda(type_i, (type_j, type_k)),
                false => brenner_exp_lambda(type_i, (type_j, type_k)),
            };
            println!("rs-explambda: {}", exp_lambda);
            println!("rs-tcoord: {}", tcoord_ij);

            let GSpline {
                value: G,
                d_cos_ijk: G_d_cos_ijk,
            } = g_spline::Input { params, type_i, cos_ijk, tcoord_ij }.compute();
            println!("rs-g gc {} {}", G, G_d_cos_ijk);

            inner_value += exp_lambda * weight_ik * G;
            inner_d_coses_ijk.push(exp_lambda * weight_ik * G_d_cos_ijk);
        }

        // Now take the square root.
        //
        // (d/dx) sqrt(f(x))  =  (1/2) (df/dx) / sqrt(f(x))
        let value = f64::sqrt(inner_value);
        let prefactor = 0.5 / value;
        Output {
            value: value,
            d_coses_ijk: sbvec_scaled(prefactor, inner_d_coses_ijk),
        }
    }
}

// b_{ij}^{pi} in Brenner
use self::bondorder_pi::BondOrderPi;
mod bondorder_pi {
    use super::*;

    pub type Output = BondOrderPi;
    pub struct Input<'a> {
        pub params: &'a Params,
        pub interactions: &'a Interactions,
        pub site_i: SiteI,
        pub bond_ij: BondI,
        // info about all bonds ik connected to site i
        // and all bonds jl connected to site j
        pub tcoords_k: &'a [u32],
        pub tcoords_l: &'a [u32],
        pub weights_ik: &'a [u32],
        pub weights_jl: &'a [u32],
    }
    pub struct BondOrderPi {
        pub value: f64,
        // These come exclusively from the sine-squareds
        pub d_deltas_ik: SiteBondVec<V3>,
        pub d_deltas_jl: SiteBondVec<V3>,
    }

    impl<'a> Input<'a> {
        pub fn compute(self) -> Output { compute(self) }
    }

    // free function for smaller indent
    fn compute(input: Input<'_>) -> Output {
        let Input {
            params, interactions, site_i, bond_ij, weights_ik, weights_jl,
            tcoords_k, tcoords_l,
        } = input;

        let site_j = interactions.bond(bond_ij).target;
        let type_i = interactions.site(site_i).atom_type;
        let type_j = interactions.site(site_j).atom_type;
        let bond_ji = interactions.bond(bond_ij).reverse_index;

        let types_k: SiteBondVec<_> = interactions.bonds(site_i).map(|bond| interactions.site(bond.target).atom_type).collect();
        let types_l: SiteBondVec<_> = interactions.bonds(site_j).map(|bond| interactions.site(bond.target).atom_type).collect();

        let index_ij = bond_ij.0 - interactions.bond_range(site_i).start;
        let index_ji = bond_ji.0 - interactions.bond_range(site_j).start;

        let YCoord {
            value: ycoord_ij,
        } = ycoord::Input {
            skip_index: index_ij,
            weights_ik: weights_ik,
            tcoords_k: tcoords_k,
            types_k: &types_k,
        }.compute();

        let YCoord {
            value: ycoord_ji,
        } = ycoord::Input {
            skip_index: index_ji,
            weights_ik: weights_jl,
            tcoords_k: tcoords_l,
            types_k: &types_l,
        }.compute();

        // NConj = 1 + (square sum over bonds ik) + (square sum over bonds jl)
        let xcoord_ij = 1 + ycoord_ij + ycoord_ji;

        let weight_ij = weights_ik[index_ij];
        let weight_ji = weights_jl[index_ji];

        // (these are not flipped; tcoords_k and tcoords_l describe the tcoords of
        //  the *target* atoms, so tcoord_i is in tcoords_l and etc.)
        let tcoord_i = tcoords_l[index_ji];
        let tcoord_j = tcoords_k[index_ij];

        let tcoord_ij = tcoord_i - weight_ij;
        let tcoord_ji = tcoord_j - weight_ji;

        // Accumulators for the output value,
        // initially formulated in terms of a larger set of variables.
        let mut value = 0.0;
        let mut d_deltas_ik = sbvec_filled(V3::zero(), types_k.len());
        let mut d_deltas_jl = sbvec_filled(V3::zero(), types_l.len());

        // First term has a T_ij prefactor, which is zero for non-CC bonds.
        if (type_i, type_j) == (AtomType::Carbon, AtomType::Carbon) {
            // sum = ∑_{k,l} sin^2(Θ_{ijkl}) f_{ik} f_{jl}
            let mut sum = 0.0;
            let mut sum_d_deltas_ik = sbvec_filled(V3::zero(), weights_ik.len()); // w.r.t. bonds around site i
            let mut sum_d_deltas_jl = sbvec_filled(V3::zero(), weights_jl.len()); // w.r.t. bonds around site j
            // We must iterate over groups of four DIFFERENT atoms ijkl,
            // with bonds between ij, ik, and jl.
            for __bond in interactions.bonds(site_i) {
                let bond_ik = __bond.index;

                // Recall that because a site may have multiple images involved in the interaction,
                // simply comparing site indices is not good enough.
                //
                // Thankfully, verifying i != l and j != k is easily done because we can
                // compare bond indices.
                if bond_ik == bond_ij {
                    continue; // site k is site j
                }

                for __bond in interactions.bonds(site_j) {
                    let bond_jl = __bond.index;

                    if bond_jl == bond_ji {
                        continue; // site l is site i
                    }

                    // Use FracBonds here to correctly account for images
                    let frac_bond_ij = interactions.frac_bond(bond_ij);
                    let frac_bond_ik = interactions.frac_bond(bond_ik);
                    let frac_bond_jl = interactions.frac_bond(bond_jl);
                    let frac_bond_il = frac_bond_ij.join(frac_bond_jl).unwrap();
                    if frac_bond_il == frac_bond_ik {
                        continue; // site k is site l
                    }

                    let delta_ij = interactions.bond(bond_ij).cart_vector;
                    let delta_ik = interactions.bond(bond_ik).cart_vector;
                    let delta_jl = interactions.bond(bond_jl).cart_vector;
                    let DihedralSineSq {
                        value: sinsq,
                        d_delta_ij: sinsq_d_delta_ij,
                        d_delta_ik: sinsq_d_delta_ik,
                        d_delta_jl: sinsq_d_delta_jl,
                    } = dihedral_sine_sq::Input { delta_ij, delta_ik, delta_jl }.compute();

                    let index_ik = bond_ik.0 - interactions.bond_range(site_i).start;
                    let index_jl = bond_ji.0 - interactions.bond_range(site_j).start;

                    //-----------
                    // NOTE: for reasons I cannot determine, the (otherwise extremely sensible)
                    //       AIREBO paper also uses Heaviside step functions here:
                    //
                    //   sinsq_ijkl * weight_ik * weight_jl * H(sin_ijk - s_min)
                    //                                      * H(sin_jil - s_min)   (s_min = 0.1)
                    //
                    // These appear designed to cut the term off for very small angles (i.e.
                    // neighbors that are very closely packed).  But this destroys the C1
                    // continuity of the function and I couldn't find any justification for it.
                    //-----------

                    // term to add to sum is  sinsq * weight_ik * weight_jl
                    //
                    // independent variables are chosen for now as the deltas that define sinsq,
                    // and the weights that appear directly
                    let weight_ik = weights_ik[index_ik] as f64;
                    let weight_jl = weights_jl[index_jl] as f64;

                    sum += sinsq * weight_ik * weight_jl;

                    sum_d_deltas_ik[index_ij] += sinsq_d_delta_ij * weight_ik * weight_jl;
                    sum_d_deltas_ik[index_ik] += sinsq_d_delta_ik * weight_ik * weight_jl;
                    sum_d_deltas_jl[index_jl] += sinsq_d_delta_jl * weight_ik * weight_jl;
                } // for bond_jl
            } // for bond_ik

            let T = t_spline::Input { params, type_i, type_j, tcoord_ij, tcoord_ji, xcoord_ij }.compute();

            value += T * sum;
            axpy_mut(&mut d_deltas_ik, T, &sum_d_deltas_ik);
            axpy_mut(&mut d_deltas_jl, T, &sum_d_deltas_jl);
        }

        // Second term: Just F.
        let f_input = f_spline::Input { params, type_i, type_j, tcoord_ij, tcoord_ji, xcoord_ij };
        if !f_input.can_assume_zero() {
            warn!("untested codepath: 2dc43f8b-12a2-46ca-8654-82f447664c04");
            let F = f_input.compute();

            value += F;
        }

        Output { value, d_deltas_ik, d_deltas_jl }
    }
}

// One of the square sum terms that appear in the definition of N^{conj}.  (Brenner, eq. 15)
//
// We call this `ycoord` due to its close relation with N^{conj}, which we call `xcoord`.
// Basically:
//
//     xcoord_ij = 1 + ycoord_ij + ycoord_ji
//
// ...but `xcoord` is not a useful thing to wrap a function around because it has a ton of
// derivatives and amounts to little more than two computations of `ycoord`.
use self::ycoord::YCoord;
mod ycoord {
    use super::*;

    pub type Output = YCoord;
    pub struct Input<'a> {
        pub skip_index: usize,
        pub weights_ik: &'a [u32],
        pub tcoords_k: &'a [u32],
        pub types_k: &'a [AtomType],
    }

    pub struct YCoord {
        pub value: u32,
    }

    impl<'a> Input<'a> {
        pub fn compute(self) -> Output { compute(self) }
    }

    // free function for smaller indent
    fn compute(input: Input<'_>) -> Output {
        let Input { skip_index, weights_ik, tcoords_k, types_k } = input;

        // Compute the sum without the square
        let mut inner_value = 0;
        let iter = zip_eq!(tcoords_k, weights_ik, types_k).enumerate();
        for (index_ik, (&tcoord_k, &weight_ik, &type_k)) in iter {
            if index_ik == skip_index || type_k == AtomType::Hydrogen {
                continue;
            }
            let xik = tcoord_k - weight_ik;

            let (F, F_d_xik) = switch((3.0, 2.0), xik as f64);
            assert_eq!(F.fract(), 0.0);
            assert_eq!(F_d_xik, 0.0);
            inner_value += weight_ik * F as u32;
        }

        // Now square it
        let value = inner_value * inner_value;
        Output {
            value: value,
        }
    }
}

// some parameter `exp(\lambda_{ijk})`
//
fn brenner_exp_lambda(i: AtomType, jk: (AtomType, AtomType)) -> f64 {
    match (i, jk) {
        (AtomType::Carbon, _) |
        (AtomType::Hydrogen, (AtomType::Carbon, AtomType::Carbon)) => 1.0, // exp(0)

        (AtomType::Hydrogen, (AtomType::Hydrogen, AtomType::Hydrogen)) => f64::exp(4.0),

        (AtomType::Hydrogen, (AtomType::Carbon, AtomType::Hydrogen)) |
        (AtomType::Hydrogen, (AtomType::Hydrogen, AtomType::Carbon)) => {
            // FIXME: The brenner paper says they fit this, but I can't find the value anywhere.
            //
            // (lammps does something weird here involving the bond lengths and a parameter
            //  they call rho... did I miss something?)
            panic!{"\
                Bond-bond interactions of type HHC (an H and a C both bonded to an H) are \
                currently missing an interaction parameter\
            "}
        },
    }
}

fn airebo_exp_lambda(i: AtomType, jk: (AtomType, AtomType)) -> f64 {
    match (i, jk) {
        (AtomType::Carbon, _) |
        (AtomType::Hydrogen, (AtomType::Carbon, AtomType::Carbon)) => 1.0, // exp(0)

        (AtomType::Hydrogen, (AtomType::Hydrogen, AtomType::Hydrogen)) => f64::exp(4.0),

        (AtomType::Hydrogen, (AtomType::Carbon, AtomType::Hydrogen)) |
        (AtomType::Hydrogen, (AtomType::Hydrogen, AtomType::Carbon)) => {
            // FIXME: The brenner paper says they fit this, but I can't find the value anywhere.
            //
            // (lammps does something weird here involving the bond lengths and a parameter
            //  they call rho... did I miss something?)
            panic!{"\
                Bond-bond interactions of type HHC (an H and a C both bonded to an H) are \
                currently missing an interaction parameter\
            "}
        },
    }
}

// `1 - \cos(\Theta_{ijkl})^2` in Brenner (equation 19)
use self::dihedral_sine_sq::DihedralSineSq;
mod dihedral_sine_sq {
    //! Diff function for the `1 - cos(Θ_{ijkl})^2` factor
    //! describing interactions of 4 atoms.
    use super::*;

    pub type Output = DihedralSineSq;
    pub struct Input {
        pub delta_ij: V3,
        pub delta_ik: V3,
        pub delta_jl: V3,
    }

    pub struct DihedralSineSq {
        pub value: f64,
        pub d_delta_ij: V3,
        pub d_delta_ik: V3,
        pub d_delta_jl: V3,
    }

    impl Input {
        pub fn compute(self) -> Output { compute(self) }
    }

    // free function for smaller indent
    fn compute(input: Input) -> Output {
        let Input { delta_ij: a, delta_ik: b1, delta_jl: b2 } = input;

        // let the cosine be
        // cos = unit(rji ⨯ rik) ∙ unit(rij ⨯ rjl)
        //     =  − unit(a ⨯ b1) ∙ unit(a ⨯ b2)     (a ≡ rij, b1 ≡ rik, b2 ≡ rjl)
        //     =            − e1 ∙ e2               (ei ≡ unit(a ⨯ bi))
        //
        // Taking partial derivatives with respect to each element of a and bi:
        //
        //  ∂cos/∂ai = − (∂e1/∂ai) ∙ e2
        //             − (∂e2/∂ai) ∙ e1
        // ∂cos/∂b1i = − (∂e1/∂b1i) ∙ e2
        // ∂cos/∂b2i = − (∂e2/∂b2i) ∙ e1

        // NOTE: analytically, e1_J_b1 = e2_J_b2 so there is a bit of unnecessary computation here
        let (e1, (e1_J_a, e1_J_b1)) = unit_cross(a, b1);
        let (e2, (e2_J_a, e2_J_b2)) = unit_cross(a, b2);

        // computes one of the dot product terms like `(∂e1/∂ai) ∙ e2` seen above.
        // It does this over all `i`, producing a gradient.
        //
        // (Since the rows of J are gradients, the rows of J.T are partial derivatives.)
        let dot_pds = |ei_J_v: M33, ej| ei_J_v.t() * ej;

        let cos = - V3::dot(&e1, &e2);
        let cos_d_a = - dot_pds(e1_J_a, e2) - dot_pds(e2_J_a, e1);
        let cos_d_b1 = - dot_pds(e1_J_b1, e2);
        let cos_d_b2 = - dot_pds(e2_J_b2, e1);

        // Output is sin^2 = 1 - cos^2
        let value = 1.0 - cos * cos;
        let prefactor = - 2.0 * cos;
        let d_a = prefactor * cos_d_a;
        let d_b1 = prefactor * cos_d_b1;
        let d_b2 = prefactor * cos_d_b2;

        Output {
            value,
            d_delta_ij: d_a,
            d_delta_ik: d_b1,
            d_delta_jl: d_b2,
        }
    }

    #[test]
    fn value() {
        let value = |delta_ij, delta_ik, delta_jl| Input { delta_ij, delta_ik, delta_jl }.compute().value;

        // 3 coplanar vectors
        assert_close!(0.0, value(V3([2.11, 0.0, 0.0]), V3([1.11, 10.0, 0.0]), V3([-3.4, 1.0, 0.0])));

        // FIXME: Should have an example with the value from diamond.
        //        (but what is the correct value?)
    }

    #[test]
    fn derivatives() {
        for _ in 0..10 {
            let delta_ij = V3::from_fn(|_| uniform(-10.0, 10.0));
            let delta_ik = V3::from_fn(|_| uniform(-10.0, 10.0));
            let delta_jl = V3::from_fn(|_| uniform(-10.0, 10.0));

            let Output {
                value: _,
                d_delta_ij: output_d_delta_ij,
                d_delta_ik: output_d_delta_ik,
                d_delta_jl: output_d_delta_jl,
            } = Input { delta_ij, delta_ik, delta_jl }.compute();

            let numerical_d_delta_ij = num_grad_v3(1e-5, delta_ij, |delta_ij| Input { delta_ij, delta_ik, delta_jl }.compute().value);
            let numerical_d_delta_ik = num_grad_v3(1e-5, delta_ik, |delta_ik| Input { delta_ij, delta_ik, delta_jl }.compute().value);
            let numerical_d_delta_jl = num_grad_v3(1e-5, delta_jl, |delta_jl| Input { delta_ij, delta_ik, delta_jl }.compute().value);

            assert_close!(abs=1e-9, output_d_delta_ij.0, numerical_d_delta_ij.0);
            assert_close!(abs=1e-9, output_d_delta_ik.0, numerical_d_delta_ik.0);
            assert_close!(abs=1e-9, output_d_delta_jl.0, numerical_d_delta_jl.0);
        }
    }
}

use self::bond_cosine::BondCosine;
mod bond_cosine {
    //! Diff function for the cos(θ) between bonds.
    use super::*;

    pub type Output = BondCosine;
    pub struct Input {
        pub delta_ij: V3,
        pub delta_ik: V3,
    }

    pub struct BondCosine {
        pub value: f64,
        pub d_delta_ij: V3,
        pub d_delta_ik: V3,
    }

    impl Input {
        pub fn compute(self) -> Output { compute(self) }
    }

    // free function for smaller indent
    fn compute(input: Input) -> Output {
        let Input { delta_ij, delta_ik } = input;
        let unit_ij = delta_ij.unit();
        let unit_ik = delta_ik.unit();

        // worked out by hand
        let value = V3::dot(&unit_ij, &unit_ik);
        let d_delta_ij = (unit_ik - unit_ij * value) / delta_ij.norm();
        let d_delta_ik = (unit_ij - unit_ik * value) / delta_ik.norm();
        Output { value, d_delta_ij, d_delta_ik }
    }

    #[test]
    fn value() {
        let x1 = V3([1.0, 0.0, 0.0]);
        let x2 = V3([2.0, 0.0, 0.0]);
        let y1 = V3([0.0, 1.0, 0.0]);
        let y3 = V3([0.0, 3.0, 0.0]);
        let xy1 = V3([1.0, 1.0, 0.0]);
        let value = |delta_ij, delta_ik| Input { delta_ij, delta_ik }.compute().value;
        assert_close!(abs=1e-8, value(x1, y1), 0.0);
        assert_close!(abs=1e-8, value(x1, x1), 1.0);
        assert_close!(abs=1e-8, value(x1, x2), 1.0);
        assert_close!(abs=1e-8, value(x1, -x2), -1.0);
        assert_close!(abs=1e-8, value(x2, -y3), 0.0);
        assert_close!(abs=1e-8, value(x2, xy1), f64::sqrt(2.0).recip());
    }

    #[test]
    fn derivatives() {
        for _ in 0..10 {
            let delta_ij = V3::from_fn(|_| uniform(-10.0, 10.0));
            let delta_ik = V3::from_fn(|_| uniform(-10.0, 10.0));

            let Output {
                value: _,
                d_delta_ij: output_d_delta_ij,
                d_delta_ik: output_d_delta_ik,
            } = Input { delta_ij, delta_ik }.compute();

            let numerical_d_delta_ij = num_grad_v3(1e-4, delta_ij, |delta_ij| Input { delta_ij, delta_ik }.compute().value);
            let numerical_d_delta_ik = num_grad_v3(1e-4, delta_ik, |delta_ik| Input { delta_ij, delta_ik }.compute().value);

            assert_close!(abs=1e-8, output_d_delta_ij.0, numerical_d_delta_ij.0);
            assert_close!(abs=1e-8, output_d_delta_ik.0, numerical_d_delta_ik.0);
        }
    }
}

//-----------------------------------------------------------------------------
// Splines

use self::g_spline::GSpline;
mod g_spline {
    use super::*;

    pub type Output = GSpline;
    pub struct Input<'a> {
        pub params: &'a Params,
        pub type_i: AtomType,
        pub tcoord_ij: u32,
        pub cos_ijk: f64,
    }

    pub struct GSpline {
        pub value: f64,
        pub d_cos_ijk: f64,
    }

    impl<'a> Input<'a> {
        pub fn compute(self) -> Output { compute(self) }
    }

    // free function for smaller indent
    fn compute(input: Input<'_>) -> Output {
        let Input { params, type_i, cos_ijk, tcoord_ij } = input;

        // FIXME: Double-double check this with Stuart.  Why is the switching regime for
        //        a coordination number larger than diamond?  (remember N_ij is usually
        //        coordination number minus one)
        // Almost all cases can be referred to a single polynomial evaluation
        // with no local dependence on tcoord_ij.
        //
        // The sole exception is the regime 3.2 <= tcoord_ij <= 3.7 for carbon.
        macro_rules! use_single_poly {
            ($poly:expr) => {{
                let (value, d_cos_ijk) = $poly.evaluate(cos_ijk);
                Output { value, d_cos_ijk }
            }}
        }

        match type_i {
            AtomType::Carbon => {
                if (tcoord_ij as f64) < C_T_LOW_COORDINATION {
                    use_single_poly!(&params.G.carbon_low_coord)
                } else if (tcoord_ij as f64) > C_T_HIGH_COORDINATION {
                    warn!("untested codepath: 37236e5f-9810-4ee5-a8c3-0a5150d9bd26");
                    use_single_poly!(&params.G.carbon_high_coord)
                } else {
                    // The one case where use_single_poly! cannot be used.

                    // d(linterp(α, A, B)) = d(α A + (1 - α) B)
                    //                     = (A - B) dα + α dA + (1 - α) dB
                    //                     = ...let's not do this right now
                    unreachable!("impossible condition found for non-reactive REBO");
                }
            },
            AtomType::Hydrogen => {
                warn!("untested codepath: 4d03fe04-5312-468e-9e30-01beddec4793");
                use_single_poly!(&params.G.hydrogen)
            },
        }
    }

    // Spline coeffs were precomputed with:
    //
    // (FIXME: would be better to do this in Rust so they can be configured)
    /*
import numpy as np
from math import radians

# Construct a bunch of terms representing the boundary conditions.
# (the nth derivative of G at x0 equals some y0)

# produces a row in the matrix to be multiplied against
# the column vector [c0, c1, ..., c5] of polynomial coeffs
def matrix_row(term):
    x, order, _value = term
    coeffs = np.polyder([1]*6, order).tolist() + [0] * order
    powers = np.arange(6).tolist()[:6-order][::-1] + [0] * order
    return np.array(x) ** powers * coeffs

def solve_spline(terms):
    matrix = np.array(list(map(matrix_row, terms)))
    b = [[value] for (_, _, value) in terms]

    coeffs, = np.linalg.solve(matrix, b).T
    for (x, order, value) in terms:
        assert abs(np.polyval(np.polyder(coeffs, order), x) - value) < 1e-13
    return coeffs

# Data from Donald W Brenner et al 2002 J. Phys.: Condens. Matter 14 783
# Table 3 (C) and Table 6 (H)

# Terms for G(x) = y, G'(x) = yp, G''(x) = ypp
def terms_at(x, ys):
    y, yp, ypp = ys
    return [(x, 0, y), (x, 1, yp), (x, 2, ypp)]

cterms_1 = terms_at(  -1, (-0.00100, 0.10400, 0.00000)) # x = cos(pi)
cterms_2 = terms_at(-1/2, ( 0.05280, 0.17000, 0.37000)) # x = cos(2/3 pi)
cterms_3 = terms_at(-1/3, ( 0.09733, 0.40000, 1.98000)) # x = cos(0.6081 pi)
cterms_4_G = [
    (0.0, 0, 0.37545), # x = cos(pi/2)
    (0.5, 0, 2.0014),  # x = cos(pi/3)
    (1.0, 0, 8.0),     # x = cos(0)
]
cterms_4_gamma = [
    (0.0, 0, 0.271856), # x = cos(pi/2)
    (0.5, 0, 0.416335), # x = cos(pi/3)
    (1.0, 0, 1.0),      # x = cos(0)
]
hterms = [
    (np.cos(radians(  0)), 0, 19.991787),
    (np.cos(radians( 60)), 0, 19.704059),
    (np.cos(radians( 90)), 0, 19.065124),
    (np.cos(radians(120)), 0, 16.811574),
    (np.cos(radians(150)), 0, 12.164186),
    (np.cos(radians(180)), 0, 11.235870),
]
pieces = [
    ("C_COEFFS_1", "Segment 1: -1 to -1/2  (pi to 2pi/3)", cterms_1 + cterms_2),
    ("C_COEFFS_2", "Segment 2: -1/2 to -1/3  (2pi/3 to 109.47°)", cterms_2 + cterms_3),
    ("C_COEFFS_3_HIGH_COORDINATION", "Segment 3 (G): -1/3 to +1  (109.47° to 0°)", cterms_3 + cterms_4_G),
    ("C_COEFFS_3_LOW_COORDINATION", "Segment 3 (gamma): -1/3 to +1  (109.47° to 0°)", cterms_3 + cterms_4_gamma),
    ("H_COEFFS", "Full curve for hydrogen", hterms),
]

print("/*")
print(open(__file__).read(), end='')
print("*/")
print("// Coeffs listed from x**5 to x**0")
for (i, xval) in enumerate(["-1.0", "-0.5", "-1.0/3.0", "1.0"]):
    print(f"const C_X_{i}: f64 = {xval};")

for (name, heading, terms) in pieces:
    print()
    print(f"// {heading}")
    print(f"const {name}: &'static [f64] = &[")
    for x in solve_spline(terms):
        print(f"{x},")
    print(f"];")
*/
    // Switch interval for tcoord in third region
    const C_T_LOW_COORDINATION: f64 = 3.2;
    const C_T_HIGH_COORDINATION: f64 = 3.7;

    /// A piecewise polynomial, optimized for the use case of only having a few segments.
    ///
    /// Between each two elements of x_div, it uses a polynomial from `coeffs`.
    #[derive(Debug, Clone)]
    struct SmallSpline1d<Array: AsRef<[f64]> + 'static> {
        x_div: &'static [f64],
        /// Polynomials between each two points in `x_div`, with coefficients in
        /// descending order.
        coeffs: &'static [Array],
    }

    #[derive(Debug, Clone)]
    pub struct SplineSet {
        carbon_high_coord: SmallSpline1d<[f64; 6]>,
        carbon_low_coord: SmallSpline1d<[f64; 6]>,
        hydrogen: SmallSpline1d<[f64; 6]>,
    }

    impl SplineSet {
        #[cfg(test)]
        fn all_splines(&self) -> Vec<SmallSpline1d<[f64; 6]>> {
            vec![
                self.carbon_high_coord.clone(),
                self.carbon_low_coord.clone(),
                self.hydrogen.clone(),
            ]
        }
    }

    /// Splines produced by fitting the data in Brenner Table 3.
    pub const BRENNER_SPLINES: SplineSet = SplineSet {
        carbon_high_coord: SmallSpline1d {
            x_div: &[-1.0, -0.5, -1.0/3.0, 1.0],
            coeffs: &[[
                // Segment 1: -1 to -1/2  (pi to 2pi/3)
                -1.342399999999925, -4.927999999999722, -6.829999999999602,
                -4.3459999999997265, -1.0979999999999095, 0.002600000000011547,
            ], [
                // Segment 2: -1/2 to -1/3  (2pi/3 to 109.47°)
                35.3116800000094, 69.87600000001967, 55.94760000001625,
                23.43200000000662, 5.544400000001327, 0.6966900000001047,
            ], [
                // Segment 3 (G): -1/3 to +1  (109.47° to 0°)
                0.5064259725000047, 1.4271989062499966, 2.028821591249997,
                2.254920828750001, 1.4071827012500007, 0.37545,
            ]],
        },
        carbon_low_coord: SmallSpline1d {
            x_div: &[-1.0, -0.5, -1.0/3.0, 1.0],
            coeffs: &[[
                // Segment 1: -1 to -1/2  (pi to 2pi/3)
                -1.342399999999925, -4.927999999999722, -6.829999999999602,
                -4.3459999999997265, -1.0979999999999095, 0.002600000000011547,
            ], [
                // Segment 2: -1/2 to -1/3  (2pi/3 to 109.47°)
                35.3116800000094, 69.87600000001967, 55.94760000001625,
                23.43200000000662, 5.544400000001327, 0.6966900000001047,
            ], [
                // Segment 3 (G): -1/3 to +1  (109.47° to 0°)
                -0.03793074749999925, 1.2711119062499994, -0.5613989287500004,
                -0.4328552912499998, 0.4892170612500001, 0.271856,
            ]],
        },
        hydrogen: SmallSpline1d {
            x_div: &[-1.0, 1.0],
            coeffs: &[[
                -9.287290931116942, -0.29608733333332005, 13.589744997229507,
                -3.1552081666666805, 0.0755044338874331, 19.065124,
            ]],
        },
    };

    /// From CH.airebo.
    ///
    /// These appear to have been produced by fitting the data in the AIREBO paper. (Stuart 2000)
    ///
    /// My current understanding is that it is okay to use these for REBO, and that they are
    /// simply an improvement upon the curves provided in Brenner (2002) that goes hand-in-hand
    /// with the modifications to `lambda_ijk`.
    ///
    /// ...however, the coefficients here are rounded to dangerously low precision, which
    /// might introduce discontinuities at the switch points (most troublingly so at 120°)
    /// that could ruin optimization algorithms.
    ///
    /// TODO: Build our own splines without such insane rounding errors
    pub const LAMMPS_SPLINES: SplineSet = SplineSet {
        carbon_high_coord: SmallSpline1d {
            x_div: &[-1.0, -0.6666666667, -0.5, -0.3333333333, 1.0],
            coeffs: &[[
                0.3862485000, 1.5544035000, 2.5334145000,
                2.1363075000, 1.0627430000, 0.2816950000,
            ], [
                0.4025160000, 1.6019100000, 2.5885710000,
                2.1681365000, 1.0718770000, 0.2827390000,
            ], [
                34.7051520000, 68.6124000000, 54.9086400000,
                23.0108000000, 5.4601600000, 0.6900250000,
            ], [
                0.5063519355, 1.4269207324, 2.0288747461,
                2.2551320117, 1.4072691309, 0.3754514434,
            ]],
        },
        carbon_low_coord: SmallSpline1d {
            x_div: &[-1.0, -0.6666666667, -0.5, -0.3333333333, 1.0],
            coeffs: &[[
                0.3862485000, 1.5544035000, 2.5334145000,
                2.1363075000, 1.0627430000, 0.2816950000,
            ], [
                0.4025160000, 1.6019100000, 2.5885710000,
                2.1681365000, 1.0718770000, 0.2827390000,
            ], [
                34.7051520000, 68.6124000000, 54.9086400000,
                23.0108000000, 5.4601600000, 0.6900250000,
            ], [
                -0.0375008379, 1.2708702246, -0.5616817383,
                -0.4328177539, 0.4892740137, 0.2718560918,
            ]],
        },
        hydrogen: SmallSpline1d {
            x_div: &[-1.0, -0.8333333333, -0.5, 1.0],
            coeffs: &[[
                630.6336000042, 2721.4308000191, 4582.1544000348,
                3781.7719000316, 1549.6358000143, 270.4568000026,
            ], [
                -94.9946400000, -229.8471299999, -210.6432299999,
                -102.4683000000, -21.0823875000, 16.9534406250,
            ], [
                0.8376699753, -2.6535615062, 3.2913322346,
                -2.5664219198, 2.0177562840, 19.0650249321,
            ]],
        },
    };

    impl<Array: AsRef<[f64]> + 'static> SmallSpline1d<Array> {
        fn evaluate(&self, x: f64) -> (f64, f64) {
            // NOTE: This linear search will *almost always* stop at one of the first two
            //       elements.  Large cosine means small angles, which are rare.
            for (i, &div) in self.x_div.iter().skip(1).enumerate() {
                if x <= div {
                    return polyval_dec(self.coeffs[i].as_ref(), x);
                }
            }

            // tolerate fuzz
            let high = *self.x_div.last().unwrap();
            let width = high - self.x_div[0];
            assert!(x < high + width * 1e-8);

            polyval_dec(self.coeffs.last().unwrap().as_ref(), x)
        }
    }

    /// Evaluate a polynomial with coefficients listed in decreasing order
    pub(super) fn polyval_dec(coeffs: &[f64], x: f64) -> (f64, f64) {
        let poly_coeffs = coeffs.iter().cloned();
        let deriv_coeffs = polyder_dec(coeffs.iter().cloned());
        (_polyval_dec(poly_coeffs, x), _polyval_dec(deriv_coeffs, x))
    }

    pub(super) fn polyder_dec(
        coeffs: impl DoubleEndedIterator<Item=f64> + ExactSizeIterator + Clone,
    ) -> impl DoubleEndedIterator<Item=f64> + ExactSizeIterator + Clone
    { coeffs.rev().skip(1).enumerate().map(|(n, x)| (n + 1) as f64 * x).rev() }

    #[inline(always)]
    pub(super) fn _polyval_dec(coeffs: impl Iterator<Item=f64>, x: f64) -> f64 {
        coeffs.fold(0.0, |acc, c| acc * x + c)
    }

    #[test]
    fn common_cases() {
        let all_params = vec![
            // The precision of lammps' precomputed splines is a joke
            (1e-4, Params::new_lammps()),
            (1e-10, Params::new_brenner()),
        ];
        for (tol, ref params) in all_params {
            // graphite
            let GSpline { value, d_cos_ijk } = Input {
                params,
                type_i: AtomType::Carbon,
                cos_ijk: f64::cos(120.0 * PI / 180.0) + 1e-12,
                tcoord_ij: 3,
            }.compute();
            // Brenner Table 3
            assert_close!(rel=tol, value, 0.05280);
            assert_close!(rel=tol, d_cos_ijk, 0.17000);

            let GSpline { value, d_cos_ijk } = Input {
                params,
                type_i: AtomType::Carbon,
                cos_ijk: f64::cos(120.0 * PI / 180.0) - 1e-12,
                tcoord_ij: 3,
            }.compute();
            assert_close!(rel=tol, value, 0.05280);
            assert_close!(rel=tol, d_cos_ijk, 0.17000);

            // diamond
            let GSpline { value, d_cos_ijk } = Input {
                params,
                type_i: AtomType::Carbon,
                cos_ijk: -1.0/3.0 + 1e-12,
                tcoord_ij: 4,
            }.compute();
            assert_close!(rel=tol, value, 0.09733);
            assert_close!(rel=tol, d_cos_ijk, 0.40000);

            let GSpline { value, d_cos_ijk } = Input {
                params,
                type_i: AtomType::Carbon,
                cos_ijk: -1.0/3.0 - 1e-12,
                tcoord_ij: 4,
            }.compute();
            assert_close!(rel=tol, value, 0.09733);
            assert_close!(rel=tol, d_cos_ijk, 0.40000);
        }
    }

    #[test]
    fn numerical_derivatives() {
        let all_params = vec![
            (1e-7, Params::new_brenner()),
            (1e-4, Params::new_lammps()),
        ];
        for (tol, ref params) in all_params {
            for type_i in AtomType::iter_all() {
                let x_divs = match type_i {
                    AtomType::Carbon => params.G.carbon_low_coord.x_div,
                    AtomType::Hydrogen => params.G.hydrogen.x_div,
                };

                let mut coses = vec![];
                // points within a region
                coses.extend(x_divs[1..].iter().map(|x| x - 1e-4));
                // points straddling two regions
                coses.extend(x_divs[1..x_divs.len()-1].iter().cloned());

                let mut tcoords = vec![3, 4];
                for &cos_ijk in &coses {
                    for &tcoord_ij in &tcoords {
                        let input = Input { params, type_i, cos_ijk, tcoord_ij };
                        let GSpline { value: _, d_cos_ijk } = input.compute();
                        assert_close!(
                            rel=tol, abs=tol,
                            d_cos_ijk,
                            numerical::slope(
                                1e-7, None,
                                cos_ijk,
                                |cos_ijk| Input { params, type_i, cos_ijk, tcoord_ij }.compute().value,
                            ),
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn continuity() {
        let iter = vec![
            (1e-13, Params::new_brenner()),
            (1e-9, Params::new_lammps()), // these coeffs are rounded pretty badly
        ];
        for (tol, ref params) in iter {
            for spline in params.G.all_splines() {
                for i in 1..spline.coeffs.len() {
                    // Should be continuous up to 2nd derivative
                    let x = spline.x_div[i];
                    let coeffs_a = spline.coeffs[i-1].iter().cloned();
                    let coeffs_b = spline.coeffs[i].iter().cloned();
                    let coeffs_da = polyder_dec(coeffs_a.clone());
                    let coeffs_db = polyder_dec(coeffs_b.clone());
                    let coeffs_dda = polyder_dec(coeffs_da.clone());
                    let coeffs_ddb = polyder_dec(coeffs_db.clone());
                    assert_close!(rel=tol, _polyval_dec(coeffs_a, x), _polyval_dec(coeffs_b, x));
                    assert_close!(rel=tol, _polyval_dec(coeffs_da, x), _polyval_dec(coeffs_db, x));
                    assert_close!(rel=tol, _polyval_dec(coeffs_dda, x), _polyval_dec(coeffs_ddb, x));
                }
            }
        }
    }
}

mod p_spline {
    use super::*;

    use self::spline_grid::bicubic;

    type Output = f64;
    pub struct Input<'a> {
        pub params: &'a Params,
        pub type_i: AtomType,
        pub type_j: AtomType,
        pub ccoord_ij: u32,
        pub hcoord_ij: u32,
    }

    #[derive(Debug, Clone)]
    pub struct SplineSet {
        CC: BicubicGrid,
        CH: BicubicGrid, // P_CH only. (P_HC is zero)
    }

    impl<'a> Input<'a> {
        pub fn compute(&self) -> Output {
            let Input { ref params, type_i, type_j, ccoord_ij, hcoord_ij } = *self;

            let poly = match (type_i, type_j) {
                (AtomType::Hydrogen, _) => return 0.0,
                (AtomType::Carbon, AtomType::Carbon) => &params.P.CC,
                (AtomType::Carbon, AtomType::Hydrogen) => &params.P.CH,
            };
            // Ignore grad because total derivative of each variable is locally zero in
            // the non-reactive case.
            let (value, _grad) = poly.evaluate(V2([ccoord_ij, hcoord_ij]));

            value
        }
    }

    lazy_static!{
        /// The only fully correct choice for PCC in second-generation REBO.
        ///
        /// Brenner (2002), Table 8.
        pub static ref BRENNER_SPLINES: SplineSet = SplineSet {
            CC: brenner_CC_input().solve().unwrap(),
            CH: brenner_CH(),
        };

        /// Suitable for AIREBO. (with the torsion term enabled)
        ///
        /// * Stuart (2000) Table VIII
        /// * LAMMPS, `pair_style airebo`.
        /// * LAMMPS, `pair_style rebo` prior ot 05Oct2016.
        ///
        /// Modifies a few of the terms to counteract AIREBO's torsional
        /// forces in unsaturated systems. (e.g. graphene)
        ///
        /// The rounding of values is chosen to match Brenner where
        /// available, and Stuart otherwise.
        pub static ref STUART_SPLINES: SplineSet = SplineSet {
            CC: stuart_CC(),
            CH: brenner_CH(),
        };

        /// Used by LAMMPS 05Oct2016–current (09Nov2018) in `pair_style rebo`.
        ///
        /// In 2016, Favata et. al reported that `pair_style rebo` erroneously
        /// used a parameter from AIREBO. LAMMPS was updated accordingly.
        ///
        /// However, there are actually three parameters that change, and
        /// this update only corrected one of them. Hence, this spline is not
        /// fully correct for neither REBO nor AIREBO.
        pub static ref FAVATA_SPLINES: SplineSet = SplineSet {
            CC: favata_CC(),
            CH: brenner_CH(),
        };
    }

    // * Brenner Table 8
    fn brenner_CC_input() -> bicubic::Input {
        let mut input = bicubic::Input::default();

        // NOTE: In the paper, Table 8 has the columns for i and j flipped.
        input.value[1][1] = 0.003_026_697_473_481; // (CH3)HC=CH(CH3)
        input.value[0][2] = 0.007_860_700_254_745; // C2H4
        input.value[0][3] = 0.016_125_364_564_267; // C2H6
        input.value[2][1] = 0.003_179_530_830_731; // i-C4H10
        input.value[1][2] = 0.006_326_248_241_119; // c-c6H12
        input
    }

    // * Brenner (Table 8)
    // * Stuart (Table VIII)  (at much lower precision)
    // * LAMMPS REBO/AIREBO  (rounded only sightly differently)
    fn brenner_CH() -> BicubicGrid {
        let mut input =  bicubic::Input::default();
        input.value[0][1] = 0.209_336_732_825_0380;  // CH2
        input.value[0][2] = -0.064_449_615_432_525;  // CH3
        input.value[0][3] = -0.303_927_546_346_162;  // CH4
        input.value[1][0] = 0.01;                    // C2H2
        input.value[2][0] = -0.122_042_146_278_2555; // (CH3)HC=CH(CH3)
        input.value[1][1] = -0.125_123_400_628_7090; // C2H4
        input.value[1][2] = -0.298_905_245_783;      // C2H6
        input.value[3][0] = -0.307_584_705_066;      // i-C4H10
        input.value[2][1] = -0.300_529_172_406_7579; // c-C6H12
        input.solve().unwrap()
    }

    fn stuart_CC() -> BicubicGrid {
        let mut input = brenner_CC_input();

        // Terms modified to counteract AIREBO's torsion.
        input.value[1][1] = -0.010_960;
        input.value[0][2] = -0.000_500;
        input.value[2][0] = -0.027_603;
        input.solve().unwrap()
    }

    fn favata_CC() -> BicubicGrid {
        let mut input = brenner_CC_input();

        // Beginning from the AIREBO params, Favata fixes one of the terms to match REBO
        // while leaving the other two. (because we're starting from REBO, we change the other two)
        input.value[1][1] = -0.010_960;
        input.value[0][2] = -0.000_500;
        // [2][0] is the one that was fixed.

        input.solve().unwrap()
    }
}

mod f_spline {
    use super::*;

    use self::spline_grid::tricubic::{self, ArrayAssignExt};

    // FIXME:
    //
    // The fitting data for the F spline is a terror to behold.
    // Brenner's table also contains numerous suspicious-looking things (i.e. possible errors)
    // that I don't want to have to fret over right now.
    //
    // For that reason, we don't actually currently use the tricubic splines, and instead just
    // check for a few special cases.
    pub type Output = f64;
    pub struct Input<'a> {
        pub params: &'a Params,
        pub type_i: AtomType,
        pub type_j: AtomType,
        pub tcoord_ij: u32,
        pub tcoord_ji: u32,
        pub xcoord_ij: u32,
    }

    impl<'a> Input<'a> {
        fn flip(self) -> Self {
            Input {
                params: self.params,
                type_i: self.type_j,
                type_j: self.type_i,
                tcoord_ij: self.tcoord_ji,
                tcoord_ji: self.tcoord_ij,
                xcoord_ij: self.xcoord_ij,
            }
        }
    }

    #[derive(Debug, Clone)]
    pub struct SplineSet {
        CC: TricubicGrid,
        CH: TricubicGrid, // F_CH and F_HC.
        HH: TricubicGrid,
    }

    impl<'a> Input<'a> {
        pub fn compute(self) -> Output { compute(self) }
    }

    // Tables 4, 6, and 9
    fn compute(input: Input<'_>) -> Output {
        let Input { params, type_i, type_j, tcoord_ij, tcoord_ji, xcoord_ij } = input;

        // NOTE: We're bailing out because I currently don't trust this spline.
        (|| panic!(
            "Not yet implemented for Brenner F: {}{} bond, Nij = {}, Nji = {}, Nconj = {}",
            type_i.char(), type_j.char(), tcoord_ij, tcoord_ji, xcoord_ij,
        ))();

        let poly = match (type_i, type_j) {
            (AtomType::Hydrogen, AtomType::Carbon) => {
                // (written to break if the output changes to another type that needs
                //  to be flipped back)
                let value: f64 = input.flip().compute();
                return value;
            },
            (AtomType::Carbon, AtomType::Carbon) => &params.F.CC,
            (AtomType::Carbon, AtomType::Hydrogen) => &params.F.CH,
            (AtomType::Hydrogen, AtomType::Hydrogen) => &params.F.HH,
        };
        // Ignore grad because total derivative of each variable is locally zero in
        // the non-reactive case.
        let (value, _grad) = poly.evaluate(V3([tcoord_ij, tcoord_ji, xcoord_ij]));

        value
    }

    // check if the value and all derivatives can be assumed to be zero,
    // without needing to compute N^{conj}, because I currently have
    // very little confidence in the spline for this one and would rather
    // bail out than use it (until it is better tested).
    impl<'a> Input<'a> {
        pub fn can_assume_zero(&self) -> bool {
            let Input { params: _, type_i, type_j, tcoord_ij, tcoord_ji, xcoord_ij } = *self;

            let int_point = V3([tcoord_ij, tcoord_ji, xcoord_ij]);
            match (type_i, type_j) {
                (AtomType::Carbon, AtomType::Carbon) => match int_point.0 {
                    [2, 2, 9] => true, // graphene/graphite
                    [3, 3, _] => true, // diamond
                    _ => false,
                },

                // Tables 6 and 9
                _ => false,
            }
        }
    }

    lazy_static! {
        /// Brenner (2002), Tables 4, 6, and 9.
        ///
        /// This has NOT been thoroughly checked against Stuart and LAMMPS, although
        /// all of what I have checked so far against Stuart matches.  (LAMMPS takes
        /// some creative freedom with the values of the splines near some of the
        /// boundaries, and I haven't yet figured out precisely what it does)
        ///
        /// **Caution:** This very likely contains errors, and using it is currently
        /// inadvisable.
        pub static ref BRENNER_SPLINES: SplineSet = SplineSet {
            CC: brenner_CC(),
            HH: brenner_HH(),
            CH: brenner_CH(),
        };
    }

    // Brenner, Table 4
    fn brenner_CC() -> TricubicGrid {
        let mut input = tricubic::Input::default();

        // NOTE: LAMMPS flattens out some values at high coordinates rather than having them
        //       go to zero. This might be a good idea for an alternate parameterization.

        input.value.assign((1, 1, 1), 0.105_000); // Acetylene
        input.value.assign((1, 1, 2), -0.004_177_5); // H2C=C=CH
        input.value.assign((1, 1, 3..=9), -0.016_085_6); // C4
        input.value.assign((2, 2, 1), 0.094_449_57); // (CH3)2C=C(CH3)2
        input.value.assign((2, 2, 2), 0.022_000_00); // Benzene

        // !!!!!!!!!!!!
        // FIXME: Are these correct?
        //
        // These are the exact values written in the paper, but it describes them as
        // "Average from difference F(2, 2, 2) to difference F(2, 2, 9)".
        //
        // They do have a constant difference, but if we were really starting from
        // the value of F[2][2][2], then that difference should be around 0.00314, not 0.00662.
        // (notice how F[2][2][3] > F[2][2][2])
        //
        // NOTE: The corresponding derivative is written as dF[2, 2, 4..=8]/dk, which is
        //       consistent with the values. Perhaps simply the comment in the paper is wrong.
        //
        // NOTE: Stuart and LAMMPS both also use these values.
        // !!!!!!!!!!!!
        input.value.assign((2, 2, 3), 0.039_705_87);
        input.value.assign((2, 2, 4), 0.033_088_22);
        input.value.assign((2, 2, 5), 0.026_470_58);
        input.value.assign((2, 2, 6), 0.019_852_93);
        input.value.assign((2, 2, 7), 0.013_235_29);
        input.value.assign((2, 2, 8), 0.006_617_64);
        input.value.assign((2, 2, 9), 0.0);

        input.value.assign((0, 1, 1), 0.043_386_99); // C2H

        input.value.assign((0, 1, 2), 0.009_917_2158); // C3
        input.value.assign((0, 2, 1), 0.049_397_6637); // CCH2
        input.value.assign((0, 2, 2), -0.011_942_669); // CCH(CH2)
        input.value.assign((0, 3, 1..=9), -0.119_798_935); // H3CC

        input.value.assign((1, 2, 1), 0.009_649_5698); // H2CCH
        input.value.assign((1, 2, 2), 0.030); // H2C=C=CH2
        input.value.assign((1, 2, 3), -0.0200); // C6H5

        // "Average from F(1,2,3) to F(1,2,6)".
        // At least this time, the description checks out.
        input.value.assign((1, 2, 4), -0.023_377_8774);
        input.value.assign((1, 2, 5), -0.026_755_7548);

        input.value.assign((1, 2, 6..=9), -0.030_133_632); // Graphite vacancy
        input.value.assign((1, 3, 2..=9), -0.124_836_752); // H3C–CCH
        input.value.assign((2, 3, 1..=9), -0.044_709_383); // Diamond vacancy

        // --------------------------
        // Derivatives

        input.di.assign((2, 1, 1), -0.052_500);
        input.di.assign((2, 1, 5..=9), -0.054_376);
        input.di.assign((2, 3, 1), 0.000_00);

        // NOTE: another oddity. These two ranges are written separately
        //       in the paper even though they could be a single range 2..=9.
        //       Does one contain an error?
        input.di.assign((2, 3, 2..=6), 0.062_418);
        input.di.assign((2, 3, 7..=9), 0.062_418);

        // !!!!!!!!!!!!!!!!!!
        // NOTE
        //
        // This derivative is related to the seemingly problematic values
        // in F[2][2][3..=8]
        // !!!!!!!!!!!!!!!!!!
        input.dk.assign((2, 2, 4..=8), -0.006_618);

        input.dk.assign((1, 1, 2), -0.060_543);

        // !!!!!!!!!!!!!!!!!!
        // FIXME
        //
        // This is highly suspicious; it seems this is intended to be the slope
        // from the linear equation describing F[1][2][3..=6], but it is at least
        // an order of magnitude off!
        //
        // This apparent error is unfortunately replicated in Stuart (2000) as
        // well as LAMMPS.
        // !!!!!!!!!!!!!!!!!!
        input.dk.assign((1, 2, 4), -0.020_044);
        input.dk.assign((1, 2, 5), -0.020_044);

        // symmetrize
        let n = input.value.len();
        for upper in 0..n {
            for lower in 0..upper {
                for k in 0..input.value[0][0].len() {
                    assert_eq!(input.value[upper][lower][k], 0.0);
                    input.value[upper][lower][k] = input.value[lower][upper][k];
                    input.dk[upper][lower][k] = input.dk[lower][upper][k];
                }
            }
        }
        for i in 0..n {
            for j in 0..n {
                input.dj[i][j].copy_from_slice(&input.di[j][i]);
            }
        }

        // The values in Brenner (2002) are actually 2 * F.
        let input = input.scale(0.5);
        input.solve().unwrap()
    }

    // Brenner, Table 6
    // TODO: Check against Stuart and LAMMPS
    fn brenner_HH() -> TricubicGrid {
        let mut input = tricubic::Input::default();
        input.value.assign((1, 1, 1), 0.249_831_916);

        // The values in Brenner (2002) are actually 2 * F.
        let input = input.scale(0.5);
        input.solve().unwrap()
    }

    // Brenner, Table 9
    // TODO: Check against Stuart and LAMMPS
    fn brenner_CH() -> TricubicGrid {
        let mut input = tricubic::Input::default();

        input.value.assign((0, 2, 5..=9), -0.009_047_787_516_128_8110); // C6H6
        input.value.assign((1, 2, 1..=9), -0.25);  // Equations (23)–(25)
        input.value.assign((1, 3, 1..=9), -0.213); // Equations (23)–(25)
        input.value.assign((1, 1, 1..=9), -0.5);   // Equations (23)–(25)

        // The values in Brenner (2002) are actually 2 * F.
        let input = input.scale(0.5);
        input.solve().unwrap()
    }
}

mod t_spline {
    //! T spline
    //!
    //! * Brenner, Table 5
    //! * Stuart, Table X

    use super::*;

    pub type Output = f64;
    pub struct Input<'a> {
        pub params: &'a Params,
        pub type_i: AtomType,
        pub type_j: AtomType,
        pub tcoord_ij: u32,
        pub tcoord_ji: u32,
        pub xcoord_ij: u32,
    }

    #[derive(Debug, Clone)]
    pub struct SplineSet {
        CC: TricubicGrid,
    }

    impl<'a> Input<'a> {
        pub fn compute(self) -> Output { compute(self) }
    }

    // Tables 4, 6, and 9
    fn compute(input: Input<'_>) -> Output {
        let Input { params, type_i, type_j, tcoord_ij, tcoord_ji, xcoord_ij } = input;

        let poly = match (type_i, type_j) {
            (AtomType::Carbon, AtomType::Carbon) => &params.T.CC,
            (AtomType::Hydrogen, _) |
            (_, AtomType::Hydrogen) => return 0.0,
        };
        // Ignore grad because total derivative of each variable is locally zero in
        // the non-reactive case.
        let (value, _grad) = poly.evaluate(V3([tcoord_ij, tcoord_ji, xcoord_ij]));

        value
    }

    lazy_static!{
        /// The TCC spline found in:
        ///
        /// * The 2nd-gen REBO paper (Brenner, 2002)
        ///
        /// This differs from Stuart's in a manner which *seems* that it could
        /// possibly be a typo. (I have not yet looked further into this.)
        ///
        /// (namely, a value defined by Brenner at only `Tij(2,2,9)` is defined
        ///  by Stuart on `Tij(2,2,2..=9)` without any ceremony).
        pub static ref BRENNER_SPLINES: SplineSet = SplineSet {
            CC: brenner_CC(),
        };

        /// The TCC spline found in:
        ///
        /// * The AIREBO paper (Stuart, 2000)
        /// * The LAMMPS implementation of REBO and AIREBO
        ///
        /// It seems plausible that this is a "bugfix" of Brenner's table.
        pub static ref STUART_SPLINES: SplineSet = SplineSet {
            CC: stuart_CC(),
        };
    }

    /// Brenner, Table 5\
    fn brenner_CC() -> TricubicGrid {
        use self::spline_grid::tricubic::{self, ArrayAssignExt};

        let mut input = tricubic::Input::default();
        input.value.assign((2, 2, 1), -0.070_280_085); // Ethane
        input.value.assign((2, 2, 9), -0.008_096_75);  // "Solid state carbon." (Graphene/graphite)

        let input = input.scale(0.5); // The values in Brenner's table are doubled
        input.solve().unwrap()
    }

    /// Stuart, Table X
    fn stuart_CC() -> TricubicGrid {
        use self::spline_grid::tricubic::{self, ArrayAssignExt};

        let mut input = tricubic::Input::default();
        input.value.assign((2, 2, 1), -0.035_140);
        input.value.assign((2, 2, 2..=9), -0.004_048); // NOTE: different from Brenner

        input.solve().unwrap()
    }
}

//-----------------------------------------------------------------------------
// Vector differentials
//
// The convention used for the output derivative (Jacobian) is to define
// the Jacobian of f(x) as
//
//           [f1_d_x]   [∇x(f1)^T]   [∂f1/∂x1  ∂f1/∂x2  ∂f1/∂x3]
//   f_J_x = [f2_d_x] = [∇x(f2)^T] = [∂f2/∂x1  ∂f2/∂x2  ∂f2/∂x3]
//           [f3_d_x]   [∇x(f3)^T]   [∂f3/∂x1  ∂f3/∂x2  ∂f3/∂x3]
//
// i.e. each row is the gradient of an element of the output.
//
// It is named using `_J_` instead of `_d_` to remind that their multiplication
// might not be commutative. (all pairs of things named with `_d_` have either
// a commutative `Mul` impl or no `Mul` impl)
//
// This form is chosen because it composes naturally from left to right
// (the manner preferred in RSP2's largely row-based formalism).
// Observe the following identities:
//
// 1. Let f, g : ℝ³ → ℝ³, and consider f(g(x)).
//    The following is true:  f_J_x = f_J_g * g_J_x
//
// 2. Let f : ℝ³ → ℝ, g : ℝ³ → ℝ³, and consider f(g(x)).
//    The following is true:  f_d_x = f_d_g * g_J_x
//    (here, the `_d_`s are partial derivatives)
//
// 3. Let f : ℝ³ → ℝ³, g : ℝ → ℝ³, and consider f(g(x)).
//    The following is true:  f_d_x = f_J_g * g_d_x
//    (here, the `_d_`s are gradients)

/// Differential of `unit(a ⨯ b)`.
///
/// Format of the output derivative (a Jacobian) is declared above.
fn unit_cross(a: V3, b: V3) -> (V3, (M33, M33)) {
    let (cross, (cross_J_a, cross_J_b)) = cross(a, b);
    let (unit, unit_J_cross) = unit(cross);
    let unit_J_a = unit_J_cross * cross_J_a;
    let unit_J_b = unit_J_cross * cross_J_b;
    (unit, (unit_J_a, unit_J_b))
}

/// Differential of the function that produces a unit vector.
///
/// Format of the output derivative (a Jacobian) is declared above.
fn unit(vec: V3) -> (V3, M33) {
    // (expression for gradient optimized by hand on paper)
    let norm = vec.norm();
    let unit = vec / norm;
    let outer_product = M3(unit.map(|x| x * unit).0);
    let grad = norm.recip() * (M33::eye() - outer_product);
    (unit, grad)
}

/// Differential of the function that computes a vector's norm.
fn norm(vec: V3) -> (f64, V3) {
    let norm = vec.norm();
    (norm, vec / norm)
}

/// Differential of the cross-product.
///
/// Format of the output derivative (a Jacobian) is declared above.
fn cross(a: V3, b: V3) -> (V3, (M33, M33)) {
    let value = a.cross(&b);
    let J_a = M3([
        // partial derivatives of value
        V3([1.0, 0.0, 0.0]).cross(&b),
        V3([0.0, 1.0, 0.0]).cross(&b),
        V3([0.0, 0.0, 1.0]).cross(&b),
    ]).t(); // transpose so rows are now gradients
    let J_b = M3([
        a.cross(&V3([1.0, 0.0, 0.0])),
        a.cross(&V3([0.0, 1.0, 0.0])),
        a.cross(&V3([0.0, 0.0, 1.0])),
    ]).t();
    (value, (J_a, J_b))
}

//-----------------------------------------------------------------------------
// utils

/// Switches from 0 to 1 as x goes from `interval.0` to `interval.1`.
#[inline(always)] // elide direction check hopefully since intervals should be constant
fn switch(interval: (f64, f64), x: f64) -> (f64, f64) {
    match IntervalSide::classify(interval, x) {
        IntervalSide::Left => (0.0, 0.0),
        IntervalSide::Inside => switch_poly5(interval, x),
        IntervalSide::Right => (1.0, 0.0),
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum IntervalSide { Left, Inside, Right }
impl IntervalSide {
    /// Determine if a value is before the beginning or after the end of a directed interval
    /// (directed as in, `interval.1 < interval.0` is ok and flips the classifications of ±∞)
    #[inline(always)] // elide direction check hopefully since intervals should be constant
    fn classify(interval: (f64, f64), x: f64) -> Self {
        if interval.0 < interval.1 {
            // interval is (min, max)
            match x {
                x if x < interval.0 => IntervalSide::Left,
                x if interval.1 < x => IntervalSide::Right,
                _ => IntervalSide::Inside,
            }
        } else {
            // interval is (max, min)
            match x {
                x if interval.0 < x => IntervalSide::Left,
                x if x < interval.1 => IntervalSide::Right,
                _ => IntervalSide::Inside,
            }
        }
    }
}

#[test]
fn switch_direction() {
    assert_eq!(switch((1.5, 2.0), 1.0).0, 0.0);
    assert_eq!(switch((1.5, 2.0), 2.5).0, 1.0);
    assert_eq!(switch((2.0, 1.5), 1.0).0, 1.0);
    assert_eq!(switch((2.0, 1.5), 2.5).0, 0.0);
}

#[test]
fn switch_middle() {
    assert_close!(switch((1.5, 2.0), 1.75).0, 0.5);
    assert_close!(switch((2.0, 1.5), 1.75).0, 0.5);
}

// Solution to:  y[x0] = 0;  y'[x0] = 0;  y''[x0] = 0;
//               y[x1] = 1;  y'[x1] = 0;  y''[x1] = 0;
fn switch_poly5(interval: (f64, f64), x: f64) -> (f64, f64) {
    let (alpha, alpha_d_x) = linterp_from(interval, (0.0, 1.0), x);

    let alpha2 = alpha*alpha;
    let alpha3 = alpha2*alpha;
    let alpha4 = alpha2*alpha2;
    let alpha5 = alpha2*alpha3;
    let value = 10.0*alpha3 - 15.0*alpha4 + 6.0*alpha5;
    let d_alpha = 10.0*3.0*alpha2 - 15.0*4.0*alpha3 + 6.0*5.0*alpha4;
    let d_x = d_alpha * alpha_d_x;

    (value, d_x)
}

/// Maps `[0, 1]` to `[y1, y2]` by a linear transform.
#[inline(always)] // enable elision of ops when args are 0.0, 1.0 etc
fn linterp((y_lo, y_hi): (f64, f64), alpha: f64) -> (f64, f64) {
    let slope = y_hi - y_lo;
    (alpha * slope + y_lo, slope)
}

/// Maps `[x1, x1]` to `[y1, y2]` by a linear transform.
///
/// `x_hi` might not map *exactly* to `y_hi`.
#[inline(always)] // enable elision of ops when args are 0.0, 1.0 etc
fn linterp_from((x_lo, x_hi): (f64, f64), y_interval: (f64, f64), x: f64) -> (f64, f64) {
    let x_width = x_hi - x_lo;
    let alpha = (x - x_lo) / x_width;
    let alpha_d_x = x_width.recip();
    let (val, val_d_alpha) = linterp(y_interval, alpha);
    (val, val_d_alpha * alpha_d_x)
}

#[inline(always)]
fn boole(cond: bool) -> f64 {
    match cond { true => 1.0, false => 0.0 }
}

//-----------------------------------------------------------------------------

#[inline(always)] // elide large stack-to-stack copies
fn sbvec_scaled<T: ::std::ops::MulAssign<f64>>(f: f64, mut xs: SiteBondVec<T>) -> SiteBondVec<T>
{ scale_mut(f, &mut xs); xs }

#[inline(always)] // elide large stack-to-stack copies
fn sbvec_filled<T: Clone>(fill: T, len: usize) -> SiteBondVec<T>
{ ::std::iter::repeat(fill).take(len).collect() }

#[inline(always)] // elide large stack-to-stack copies
fn axpy_mut<T: Copy>(a: &mut [T], alpha: f64, b: &[T])
where f64: ::std::ops::Mul<T, Output=T>, T: ::std::ops::AddAssign<T>,
{
    for (a, b) in zip_eq!(a, b) {
        *a += alpha * *b;
    }
}

fn scaled<T: ::std::ops::MulAssign<f64>>(f: f64, mut xs: Vec<T>) -> Vec<T>
{ scale_mut(f, &mut xs); xs }


fn scale_mut<T: ::std::ops::MulAssign<f64>>(factor: f64, xs: &mut [T]) {
    for x in xs {
        *x *= factor;
    }
}

fn concat_any_order<T>(a: Vec<T>, b: Vec<T>) -> Vec<T> {
    let (mut long, short) = match a.len() > b.len() {
        true => (a, b),
        false => (b, a),
    };
    long.extend(short);
    long
}

fn cleared<T>(mut vec: Vec<T>) -> Vec<T> {
    vec.clear();
    vec
}

//-----------------------------------------------------------------------------

/// A tricubic or bicubic spline with constraints defined on an integer grid.
pub use self::spline_grid::{TricubicGrid, BicubicGrid};
pub mod spline_grid {
    use super::*;

    // Until we get const generics, it's too much trouble to be generic over lengths,
    // so we'll just use one fixed dimension.
    pub const MAX_I: usize = 4;
    pub const MAX_J: usize = 4;
    pub const MAX_K: usize = 9;

    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    enum EvalKind { Fast, Slow }

    pub use self::tricubic::TricubicGrid;
    pub mod tricubic {
        use super::*;
        use ::std::ops::RangeInclusive;

        /// A grid of "fencepost" values.
        pub type EndpointGrid<T> = nd![T; MAX_I+1; MAX_J+1; MAX_K+1];
        /// A grid of "fence segment" values.
        pub type Grid<T> = nd![T; MAX_I; MAX_J; MAX_K];
        pub type Input = _Input<EndpointGrid<f64>>;

        /// The values and derivatives that are fitted to produce a tricubic spline.
        ///
        /// NOTE: not all constraints are explicitly listed;
        /// We also place implicit constraints that `d^2/didj`, `d^2/didk`,
        /// `d^2/djdk`, and `d^3/didjdk` are zero at all integer points.
        ///
        /// (why these particular derivatives?  It turns out that these are the
        ///  ones that produce linearly independent equations. See Lekien.)
        ///
        /// # References
        ///
        /// F. Lekien and J. Marsden, Tricubic interpolation in three dimensions,
        /// Int. J. Numer. Meth. Engng 2005; 63:455–471
        #[derive(Debug, Clone, Default)]
        pub struct _Input<G> {
            pub value: G,
            pub di: G,
            pub dj: G,
            pub dk: G,
        }

        //------------------------------------

        // uniform interface for assigning a single element or a range
        pub trait ArrayAssignExt<I> {
            fn assign(&mut self, i: I, fill: f64);
        }

        impl ArrayAssignExt<(usize, usize, usize)> for EndpointGrid<f64> {
            fn assign(&mut self, (i, j, k): (usize, usize, usize), fill: f64) {
                self[i][j][k] = fill;
            }
        }

        impl ArrayAssignExt<(usize, usize, RangeInclusive<usize>)> for EndpointGrid<f64> {
            fn assign(&mut self, (i, j, k): (usize, usize, RangeInclusive<usize>), fill: f64) {
                for x in &mut self[i][j][k] {
                    *x = fill;
                }
            }
        }

        //------------------------------------

        // FIXME: Remove
        //        (leftover from version of code that actually did the splines)
        #[derive(Debug, Clone)]
        pub struct TricubicGrid {
            pub(super) fit_params: Box<Input>,
        }

        impl TricubicGrid {
            pub fn evaluate(&self, point: V3<u32>) -> (f64, V3) {
                // We assume the splines are flat with constant value outside the fitted regions.
                let point = clip_point(point);

                let V3([i, j, k]): V3<usize> = point.map(|x: u32| x as usize);

                let value = self.fit_params.value[i][j][k];
                let di = self.fit_params.di[i][j][k];
                let dj = self.fit_params.dj[i][j][k];
                let dk = self.fit_params.dk[i][j][k];
                (value, V3([di, dj, dk]))
            }
        }

        impl<A> _Input<A> {
            fn map_grids<B>(&self, mut func: impl FnMut(&A) -> B) -> _Input<B> {
                _Input {
                    value: func(&self.value),
                    di: func(&self.di),
                    dj: func(&self.dj),
                    dk: func(&self.dk),
                }
            }
        }

        impl Input {
            // HACK: This is just here temporarily to keep the interface resembling what
            //       it looked like back when this module actually solved the splines.
            pub fn solve(&self) -> FailResult<TricubicGrid> {
                use ::rsp2_array_utils::{try_arr_from_fn, arr_from_fn, map_arr};
                self.verify_clipping_is_valid()?;

                let fit_params = Box::new(self.clone());
                Ok(TricubicGrid { fit_params })
            }

            pub fn scale(mut self, factor: f64) -> Self {
                { // FIXME: block will be unnecessary once NLL lands
                    let Input { value, di, dj, dk } = &mut self;
                    for &mut &mut ref mut array in &mut[value, di, dj, dk] {
                        for plane in array {
                            for row in plane {
                                for x in row {
                                    *x *= factor;
                                }
                            }
                        }
                    }
                }
                self
            }

            #[cfg(test)]
            pub fn random(scale: f64) -> Self {
                Input {
                    value: ::rand::random(),
                    di: ::rand::random(),
                    dj: ::rand::random(),
                    dk: ::rand::random(),
                }.scale(scale).ensure_clipping_is_valid()
            }
        }

        impl Input {
            // To make clipping always valid, we envision that the spline is flat outside of
            // the fitted region.  For C1 continuity, this means the derivatives at these
            // boundaries must be zero.
            pub fn verify_clipping_is_valid(&self) -> FailResult<()> {
                let Input { value: _, di, dj, dk } = self;

                macro_rules! check {
                    ($iter:expr) => {
                        ensure!(
                            $iter.into_iter().all(|&x| x == 0.0),
                            "derivatives must be zero at the endpoints of the spline"
                        )
                    };
                }

                check!(di[0].flat());
                check!(di.last().unwrap().flat());
                check!(dj.iter().flat_map(|plane| &plane[0]));
                check!(dj.iter().flat_map(|plane| plane.last().unwrap()));
                check!(dk.iter().flat_map(|plane| plane.iter().map(|row| &row[0])));
                check!(dk.iter().flat_map(|plane| plane.iter().map(|row| row.last().unwrap())));
                Ok(())
            }

            // useful for tests
            pub(super) fn ensure_clipping_is_valid(mut self) -> Self {
                { // FIXME block is unnecessary once NLL lands
                    let Input { value: _, di, dj, dk } = &mut self;
                    fn zero<'a>(xs: impl IntoIterator<Item=&'a mut f64>) {
                        for x in xs { *x = 0.0; }
                    }

                    zero(di[0].flat_mut());
                    zero(di.last_mut().unwrap().flat_mut());
                    zero(dj.iter_mut().flat_map(|plane| &mut plane[0]));
                    zero(dj.iter_mut().flat_map(|plane| plane.last_mut().unwrap()));
                    zero(dk.iter_mut().flat_map(|plane| plane.iter_mut().map(|row| &mut row[0])));
                    zero(dk.iter_mut().flat_map(|plane| plane.iter_mut().map(|row| row.last_mut().unwrap())));
                }
                self
            }
        }

        pub fn clip_point(mut point: V3<u32>) -> V3<u32> {
            point[0] = point[0].min(MAX_I as u32);
            point[1] = point[1].min(MAX_J as u32);
            point[2] = point[2].min(MAX_K as u32);
            point
        }

        //------------------------------------
        // tests

        #[test]
        fn test_spline_fit_accuracy() -> FailResult<()> {
            let fit_params = Input::random(1.0);
            let spline = fit_params.solve()?;

            for i in 0..=MAX_I {
                for j in 0..=MAX_J {
                    for k in 0..=MAX_K {
                        let output = spline.evaluate(V3([i, j, k]).map(|x| x as u32));
                        let (value, V3([di, dj, dk])) = output;
                        assert_eq!(value, fit_params.value[i][j][k]);
                        assert_eq!(di, fit_params.di[i][j][k]);
                        assert_eq!(dj, fit_params.dj[i][j][k]);
                        assert_eq!(dk, fit_params.dk[i][j][k]);
                    }
                }
            }

            // points outside the boundaries assume a flat curve
            let base_point = V3([2, 2, 2]);
            let base_index = V3([2, 2, 2]);
            for axis in 0..3 {
                let mut input_point = base_point;
                let mut expected_index = base_index;
                input_point[axis] = [MAX_I, MAX_J, MAX_K][axis] as u32 + 3;
                expected_index[axis] = [MAX_I, MAX_J, MAX_K][axis];

                let (value, V3([di, dj, dk])) = spline.evaluate(input_point);

                let V3([i, j, k]) = expected_index;
                assert_eq!(value, fit_params.value[i][j][k]);
                assert_eq!(di, fit_params.di[i][j][k]);
                assert_eq!(dj, fit_params.dj[i][j][k]);
                assert_eq!(dk, fit_params.dk[i][j][k]);
            }
            Ok(())
        }
    } // mod tricubic

    //------------------------------------
    // bicubic

    pub use self::bicubic::BicubicGrid;
    pub mod bicubic {
        use super::*;

        /// A grid of "fencepost" values.
        pub type EndpointGrid<T> = nd![T; MAX_I+1; MAX_J+1];
        /// A grid of "fence segment" values.
        pub type Grid<T> = nd![T; MAX_I; MAX_J];

        /// Input for a bicubic spline.
        ///
        /// Not included is an implicit constraint that `d^2/didj = 0` at all integer points.
        #[derive(Default)]
        pub struct Input {
            pub value: EndpointGrid<f64>,
            pub di: EndpointGrid<f64>,
            pub dj: EndpointGrid<f64>,
        }

        #[derive(Debug, Clone)]
        pub struct BicubicGrid {
            // "Do the simplest thing that will work."
            tricubic: TricubicGrid,
        }

        impl BicubicGrid {
            pub fn evaluate(&self, V2([i, j]): V2<u32>) -> (f64, V2) {
                let (value, V3([di, dj, _dk])) = self.tricubic.evaluate(V3([i, j, 0]));
                (value, V2([di, dj]))
            }
        }

        impl Input {
            pub fn solve(&self) -> FailResult<BicubicGrid> {
                let tricubic = self.to_tricubic_input().solve()?;
                Ok(BicubicGrid { tricubic })
            }

            fn to_tricubic_input(&self) -> tricubic::Input {
                use ::rsp2_array_utils::{map_arr};
                let Input { value, di, dj } = *self;

                // make everything constant along the k axis
                let extend = |arr| map_arr(arr, |row| map_arr(row, |x| [x; MAX_K+1]));
                tricubic::Input {
                    value: extend(value),
                    di: extend(di),
                    dj: extend(dj),
                    dk: Default::default(),
                }
            }

            #[cfg(test)]
            fn from_tricubic_input(input: &tricubic::Input) -> Self {
                use ::rsp2_array_utils::{map_arr};

                let tricubic::Input { value, di, dj, dk } = *input;

                let unextend = |arr| map_arr(arr, |plane| map_arr(plane, |row: [_; MAX_K+1]| row[0]));

                assert_eq!(unextend(dk), unextend(<tricubic::EndpointGrid<f64>>::default()));
                Input {
                    value: unextend(value),
                    di: unextend(di),
                    dj: unextend(dj),
                }
            }

            #[cfg(test)]
            pub fn random(scale: f64) -> Self {
                Self::from_tricubic_input(&tricubic::Input::random(scale))
            }
        }

        //------------------------------------
        // tests

        #[test]
        fn test_spline_fit_accuracy() -> FailResult<()> {
            let fit_params = Input::random(1.0);
            let spline = fit_params.solve()?;

            for i in 0..=MAX_I {
                for j in 0..=MAX_J {
                    let output = spline.evaluate(V2([i, j]).map(|x| x as u32));
                    let (value, V2([di, dj])) = output;
                    assert_eq!(value, fit_params.value[i][j]);
                    assert_eq!(di, fit_params.di[i][j]);
                    assert_eq!(dj, fit_params.dj[i][j]);
                }
            }

            // points outside the boundaries assume a flat curve
            let base_point = V2([2, 2]);
            let base_index = V2([2, 2]);
            for axis in 0..2 {
                let mut input_point = base_point;
                let mut expected_index = base_index;
                input_point[axis] = [MAX_I, MAX_J][axis] as u32 + 3;
                expected_index[axis] = [MAX_I, MAX_J][axis];

                let (value, V2([di, dj])) = spline.evaluate(input_point);

                let V2([i, j]) = expected_index;
                assert_eq!(value, fit_params.value[i][j]);
                assert_eq!(di, fit_params.di[i][j]);
                assert_eq!(dj, fit_params.dj[i][j]);
            }
            Ok(())
        }
    }
}


//-----------------------------------------------------------------------------

#[cfg(test)]
#[deny(unused)]
mod derivative_tests {
    use super::*;

    #[test]
    fn d_linterp_from() {
        for _ in 0..10 {
            let xs = (uniform(-10.0, 10.0), uniform(-10.0, 10.0));
            let ys = (uniform(-10.0, 10.0), uniform(-10.0, 10.0));
            let x = uniform(-30.0, 30.0);
            let f = |x| linterp_from(xs, ys, x);
            let (_, diff) = f(x);

            // linear function so central difference is exact
            //
            // sometimes the diff is small, so also allow abs tolerance
            let num_diff = numerical::slope(1e-1, Some(CentralDifference), x, |x| f(x).0);
            assert_close!{rel=1e-11, abs=1e-11, diff, num_diff};
        }
    }

    // Numerical tests for brenner_G are in that module
}

//-----------------------------------------------------------------------------

#[cfg(test)]
fn uniform(a: f64, b: f64) -> f64 { ::rand::random::<f64>() * (b - a) + a }

#[cfg(test)]
fn num_grad_v3(
    interval: f64,
    point: V3,
    mut value_fn: impl FnMut(V3) -> f64,
) -> V3 {
    numerical::gradient(interval, None, &point.0, |v| value_fn(v.to_array())).to_array()
}
