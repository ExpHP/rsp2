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

// FIXME: Kill this whole file at some point.

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
//! # Caution
//!
//! The reactive bits of the potential (i.e. fractional weights) are implemented,
//! but not at all well-tested because RSP2 does not need them.
//! Considering that they constitute the bulk of the implementation effort, they almost
//! very certainly contain bugs that will need to be ironed out once they are needed.
//!
//! Use with extreme caution.
//!
//! # Citations
//!
//! * **2nd gen REBO:** Donald W Brenner et al 2002 J. Phys.: Condens. Matter 14 783
//! * **AIREBO:** Steven J Stuart et al J. Chem. Phys. 112, 6472 (2000)
//! * **LAMMPS:** S. Plimpton, J Comp Phys, 117, 1-19 (1995)

use super::splines::{self, TricubicGrid, BicubicGrid};

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
// This code needs to compute some extremely ugly-looking derivatives
// of an almost-as-ugly-looking potential. As usual, this leads to a
// *catastrophic explosion* of things that are temporarily needed as
// part of the computation, and it only gets worse when you try to
// factor stuff out.
//
// Like some other places in rsp2, this code makes heavy use of ad-hoc
// structs to simulate keyword arguments and to name the output variables.
// This makes it impossible to accidentally switch the order of similarly-
// typed arguments.
//
// (unfortunately, it also produces an insane amount of boilerplate around
//  each function. Them's the brakes.)
//
// ## Dealing with derivatives
//
// All code dealing with derivatives is written in a manner that maximizes
// the utility of the unused binding and unused assignment lints.
//
// This is done because the set of derivatives that something must account for
// can vary depending on how you define its independent variables. Sometimes it
// is convenient to treat two normally related things (like bond weight and the
// displacement vectors) as independent variables up until a higher point in the
// call stack.  We want to ensure that all necessary changes can be easily found
// when refactoring something to add or remove explicit dependence on some derivative.
//
// Systematic rules are also followed to make sure that the mathematical parts of
// the code are as boring as possible (i.e. the process of writing this code is
// extremely mechanical):
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
// 2. Fully destructure the output of any function that computes pieces of the potential,
//    so that the compiler complains about missing fields when one is added.
//
// 3. Please ensure that derivatives can be reasoned about locally as much as possible.
//
//    Generally speaking, one should be able to verify the correctness of a derivative
//    from nearby statements and comments, without knowing the definitions of expressions
//    that appear in the value. Even make trivial bindings for dumb and obvious
//    derivatives if a temporary is defined far away from where some of its derivatives
//    are used. E.g.
//
//    ```rust
//    let tcoord = ccoord_i + hcoord_i;
//    let tcoord_d_ccoord_i = 1.0;
//    let tcoord_d_hcoord_i = 1.0;
//    ```
//
// 4. Use the following pattern when building vectors over a loop that may contain
//    values and derivatives.
//
//    ```rust
//    let name;
//    let name_d_x1;
//    let name_d_x2;
//    {
//        let mut tmp_name = vec![];
//        let mut tmp_name_d_x1 = vec![];
//        let mut tmp_name_d_x2 = vec![];
//
//        for item in iter {
//            ...
//        }
//
//        name = tmp_name;
//        name_d_x1 = tmp_name_d_x1;
//        name_d_x2 = tmp_name_d_x2;
//    }
//    ```
//
//    This guarantees that you will get a warning *somewhere* if you forget to use
//    one of the derivatives later.
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
        pub G: Cow<'static, splines::G::SplineSet>,
        pub T: Cow<'static, splines::T::SplineSet>,
        pub F: Cow<'static, splines::F::SplineSet>,
        pub P: Cow<'static, splines::P::SplineSet>,
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
                G: Cow::Owned(splines::G::BRENNER),
                P: Cow::Borrowed(&splines::P::BRENNER),
                T: Cow::Borrowed(&splines::T::BRENNER),
                F: Cow::Borrowed(&splines::F::BRENNER),
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
                G: Cow::Owned(splines::G::LAMMPS),
                P: Cow::Borrowed(&splines::P::FAVATA),
                F: Cow::Borrowed(&splines::F::BRENNER),
                T: Cow::Borrowed(&splines::T::STUART),
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
    let (value, d_deltas) = compute_rebo_bonds(params, &interactions);

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
) -> (f64, IndexVec<BondI, V3>) {
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
        tcoord: f64,
        // Brenner's f_ij
        bond_weight: SiteBondVec<f64>,
        bond_weight_d_delta: SiteBondVec<V3>,
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

            let mut tcoord = 0.0;
            let mut bond_VR = SiteBondVec::new();
            let mut bond_VR_d_delta = SiteBondVec::new();
            let mut bond_VA = SiteBondVec::new();
            let mut bond_VA_d_delta = SiteBondVec::new();
            let mut bond_weight = SiteBondVec::new();
            let mut bond_weight_d_delta = SiteBondVec::new();

            for __bond in interactions.bonds(site_i) {
                let site_j = __bond.target;
                let delta_ij = __bond.cart_vector;
                let type_j = interactions.site(site_j).atom_type;
                let params_ij = params.by_type[type_i][type_j];

                let (length, length_d_delta) = norm(delta_ij);
                let (weight, weight_d_length) = switch((params_ij.Dmax, params_ij.Dmin), length);
                let weight_d_delta = weight_d_length * length_d_delta;

                tcoord += weight;
                bond_weight.push(weight);
                bond_weight_d_delta.push(weight_d_delta);

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

            FirstPassSiteData {
                tcoord,
                bond_weight, bond_weight_d_delta,
                bond_VR, bond_VR_d_delta,
                bond_VA, bond_VA_d_delta,
            }
        }).collect()
    });

    let out = interactions.site_range().into_par_iter().map(SiteI::new).map(|site_i| {
        let __site = interactions.site(site_i);
        let FirstPassSiteData {
            tcoord: _,
            ref bond_weight, ref bond_weight_d_delta,
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
        // some things also depend on tcoords of distant atoms.
        // This is too much of a pain to handle immediately.
        let mut site_V_d_other_tcoords = ArrayVec::<[(SiteI, SiteBondVec<f64>); SITE_MAX_BONDS * 2]>::new();

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
            d_weights: Vsp_i_d_weights,
        } = out;

        site_V -= Vsp_i;
        axpy_mut(&mut site_V_d_delta, -1.0, &Vsp_i_d_deltas);
        for (index_ij, _) in interactions.bonds(site_i).enumerate() {
            let Vsp_i_d_weight_ij = Vsp_i_d_weights[index_ij];
            let weight_ij_d_delta_ij = bond_weight_d_delta[index_ij];
            site_V_d_delta[index_ij] += -1.0 * Vsp_i_d_weight_ij * weight_ij_d_delta_ij;
        }

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
                d_tcoords_k: bpi_d_tcoords_k,
                d_tcoords_l: bpi_d_tcoords_l,
                d_weights_ik: bpi_d_weights_ik,
                d_weights_jl: bpi_d_weights_jl,
            } = out;

            // remove the explicit dependence of bpi on {weights_ik}, {weights_jl}
            { // FIXME Block unnecessary after NLL lands
                let iter = zip_eq!(&mut bpi_d_deltas_ik, bpi_d_weights_ik, bond_weight_d_delta);
                for (bpi_d_delta_ik, bpi_d_weight_ik, &weight_ik_d_delta_ik) in iter {
                    *bpi_d_delta_ik += bpi_d_weight_ik * weight_ik_d_delta_ik;
                }
            }

            let ref site_j_weights_d_delta = site_data[site_j].bond_weight_d_delta;
            { // FIXME Block unnecessary after NLL lands
                let iter = zip_eq!(&mut bpi_d_deltas_jl, bpi_d_weights_jl, site_j_weights_d_delta);
                for (bpi_d_delta_jl, bpi_d_weight_jl, &weight_ik_d_delta_jl) in iter {
                    *bpi_d_delta_jl += bpi_d_weight_jl * weight_ik_d_delta_jl;
                }
            }
            println!("rs-bpi: {}", bpi);

            let VA_ij = bond_VA[index_ij];
            let VA_ij_d_delta_ij = bond_VA_d_delta[index_ij];

            site_V += bpi * VA_ij;
            site_V_d_delta[index_ij] += bpi * VA_ij_d_delta_ij;

            axpy_mut(&mut site_V_d_delta, VA_ij, &bpi_d_deltas_ik);
            site_V_d_other_deltas.push((site_j, sbvec_scaled(VA_ij, bpi_d_deltas_jl)));
            site_V_d_other_tcoords.push((site_i, sbvec_scaled(VA_ij, bpi_d_tcoords_k)));
            site_V_d_other_tcoords.push((site_j, sbvec_scaled(VA_ij, bpi_d_tcoords_l)));
        }

        (site_V, site_V_d_delta, site_V_d_other_deltas, site_V_d_other_tcoords)

    // well this is awkward
    }).fold(
        || (0.0, IndexVec::new(), vec![], vec![]),
        |
            (mut value, mut d_deltas, mut d_other_deltas, mut d_other_tcoords),
            (site_V, site_V_d_delta, site_V_d_other_deltas, site_V_d_other_tcoords),
        | {
            value += site_V;
            d_deltas.extend(site_V_d_delta);
            d_other_deltas.extend(site_V_d_other_deltas);
            d_other_tcoords.extend(site_V_d_other_tcoords);
            (value, d_deltas, d_other_deltas, d_other_tcoords)
        },
    ).reduce(
        || (0.0, IndexVec::new(), vec![], vec![]),
        |
            (mut value, mut d_deltas, d_other_deltas, d_other_tcoords),
            (value_part, d_delta_part, d_other_deltas_part, d_other_tcoords_part),
        | {
            value += value_part;
            d_deltas.extend(d_delta_part); // must be concatenated in order
            let d_other_deltas = concat_any_order(d_other_deltas, d_other_deltas_part);
            let d_other_tcoords = concat_any_order(d_other_tcoords, d_other_tcoords_part);
            (value, d_deltas, d_other_deltas, d_other_tcoords)
        },
    );
    let (value, mut d_deltas, d_other_deltas, d_other_tcoords) = out;

    assert_eq!(d_deltas.len(), interactions.num_bonds());

    // absorb the other terms we couldn't take care of into d_deltas
    // and end this miserable function for good
    for (site_i, d_deltas_ij) in d_other_deltas {
        let _: SiteBondVec<V3> = d_deltas_ij;
        axpy_mut(&mut d_deltas.raw[interactions.bond_range(site_i)], 1.0, &d_deltas_ij);
    }

    for (site_i, d_tcoords_j) in d_other_tcoords {
        for (__bond, d_tcoord_j) in zip_eq!(interactions.bonds(site_i), d_tcoords_j) {
            let site_j = __bond.target;

            // tcoord_j = sum_l weight_jl
            for (index_jl, __bond) in interactions.bonds(site_j).enumerate() {
                let bond_jl = __bond.index;

                let tcoord_j_d_weight_jl = 1.0;
                let weight_jl_d_delta_jl = site_data[site_j].bond_weight_d_delta[index_jl];
                d_deltas[bond_jl] += d_tcoord_j * tcoord_j_d_weight_jl * weight_jl_d_delta_jl;
            }
        }
    }

    assert_eq!(d_deltas.len(), interactions.num_bonds());
    (value, d_deltas)
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
        pub bond_weights: &'a [f64],
        // The VA_ij terms for each bond at site i.
        pub bond_VAs: &'a SiteBondVec<f64>,
        pub bond_VAs_d_delta: &'a SiteBondVec<V3>,
    }
    pub struct SiteSigmaPiTerm {
        pub value: f64,
        /// Derivatives with respect to the bonds listed in order of `interactions.bonds(site_i)`.
        pub d_deltas: SiteBondVec<V3>,
        pub d_weights: SiteBondVec<f64>,
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
        let mut ccoord_i = 0.0;
        let mut hcoord_i = 0.0;
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
        let mut d_weights: SiteBondVec<f64> = sbvec_filled(0.0, bond_weights.len());

        for (index_ij, __bond) in interactions.bonds(site_i).enumerate() {
            let type_j = bond_target_types[index_ij];
            let delta_ij = __bond.cart_vector;
            let weight_ij = bond_weights[index_ij];

            // These are what Brenner's Ni REALLY are.
            let ccoord_ij = ccoord_i - boole(type_j == AtomType::Carbon) * weight_ij;
            let hcoord_ij = hcoord_i - boole(type_j == AtomType::Hydrogen) * weight_ij;

            // NOTE: When all weights are integers, each site will only ever have at most
            //       two unique values of P computed. (one for each target type).
            //
            //       However, for fractional weights, the Nij may not all be equal.
            //
            //       Given that integer weights already take a fast path in BrennerP,
            //       it's not worth optimizing this; we'll just compute it for every bond.
            let P_ij = p_spline::Input { params, type_i, type_j, ccoord_ij, hcoord_ij }.compute();

            // Gather all cosines between bond i->j and other bonds i->k.
            let coses_ijk;
            let coses_ijk_d_delta_ij;
            let coses_ijk_d_delta_ik;
            {
                let mut tmp_coses_ijk = SiteBondVec::new();
                let mut tmp_coses_ijk_d_delta_ij = SiteBondVec::new();
                let mut tmp_coses_ijk_d_delta_ik = SiteBondVec::new();

                for (index_ik, __bond) in interactions.bonds(site_i).enumerate() {
                    let delta_ik = __bond.cart_vector;
                    if index_ij == index_ik {
                        // set up bombs in case of possible misuse
                        tmp_coses_ijk.push(NAN);
                        tmp_coses_ijk_d_delta_ij.push(V3::from_fn(|_| NAN));
                        tmp_coses_ijk_d_delta_ik.push(V3::from_fn(|_| NAN));
                    } else {
                        let out = bond_cosine::Input { delta_ij, delta_ik }.compute();
                        let BondCosine {
                            value: cos,
                            d_delta_ij: cos_d_delta_ij,
                            d_delta_ik: cos_d_delta_ik,
                        } = out;
                        tmp_coses_ijk.push(cos);
                        tmp_coses_ijk_d_delta_ij.push(cos_d_delta_ij);
                        tmp_coses_ijk_d_delta_ik.push(cos_d_delta_ik);
                    }
                }
                coses_ijk = tmp_coses_ijk;
                coses_ijk_d_delta_ij = tmp_coses_ijk_d_delta_ij;
                coses_ijk_d_delta_ik = tmp_coses_ijk_d_delta_ik;
            }

            // We're finally ready to compute the bond order.

            let bsp_ij;
            let bsp_ij_d_deltas;
            let bsp_ij_d_weights;
            { // dedicated scope for accumulating bsp
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
                    d_ccoord_ij: bsp_ij_d_ccoord_ij,
                    d_hcoord_ij: bsp_ij_d_hcoord_ij,
                    d_coses_ijk: bsp_ij_d_coses_ijk,
                    d_weights_ik: bsp_ij_d_weights_ik,
                } = out;

                // ...and now reformulate it as a function solely of the deltas and weights.
                let mut tmp_d_deltas: SiteBondVec<V3> = sbvec_filled(V3::zero(), bond_weights.len());
                let mut tmp_d_weights: SiteBondVec<f64> = sbvec_filled(0.0, bond_weights.len());

                // Even though this term describes a single bond, its dependence on the tcoord_ij
                // produce derivatives with respect to all of the bond weights *except* that of ij.
                if (bsp_ij_d_ccoord_ij, bsp_ij_d_hcoord_ij) != (0.0, 0.0) {
                    for (index_ik, ty) in bond_target_types.iter().enumerate() {
                        if index_ik == index_ij {
                            // ccoord_ij_d_weight_ik = 0.0,  hcoord_ij_d_weight_ik = 0.0
                            continue;
                        } else {
                            match ty {
                                // ccoord_ij_d_weight_ik = 1.0,  hcoord_ij_d_weight_ik = 0.0
                                AtomType::Carbon => tmp_d_weights[index_ik] += bsp_ij_d_ccoord_ij,
                                // ccoord_ij_d_weight_ik = 0.0,  hcoord_ij_d_weight_ik = 1.0
                                AtomType::Hydrogen => tmp_d_weights[index_ik] += bsp_ij_d_hcoord_ij,
                            }
                        }
                    }
                }

                // Some derivatives also come from the ik bonds.
                let iter = zip_eq!(bsp_ij_d_weights_ik, bsp_ij_d_coses_ijk).enumerate();
                for (index_ik, (bsp_ij_d_weight_ik, bsp_ij_d_cos_ijk)) in iter {
                    // Mind the gap
                    if index_ij == index_ik {
                        continue;
                    }
                    let cos_ijk_d_delta_ij = coses_ijk_d_delta_ij[index_ik];
                    let cos_ijk_d_delta_ik = coses_ijk_d_delta_ik[index_ik];

                    tmp_d_weights[index_ik] += bsp_ij_d_weight_ik;
                    tmp_d_deltas[index_ij] += bsp_ij_d_cos_ijk * cos_ijk_d_delta_ij;
                    tmp_d_deltas[index_ik] += bsp_ij_d_cos_ijk * cos_ijk_d_delta_ik;
                }

                bsp_ij = tmp_value;
                bsp_ij_d_deltas = tmp_d_deltas;
                bsp_ij_d_weights = tmp_d_weights;
                println!("rs-bsp: {}", bsp_ij);
            } // dedicated scope for accumulating bsp

            // True term to add to sum is 0.5 * VR_ij * bsp_ij
            let VA_ij = bond_VAs[index_ij];
            let VA_ij_d_delta_ij = bond_VAs_d_delta[index_ij];

            value += 0.5 * VA_ij * bsp_ij;
            d_deltas[index_ij] += 0.5 * VA_ij_d_delta_ij * bsp_ij;
            axpy_mut(&mut d_deltas, 0.5 * VA_ij, &bsp_ij_d_deltas);
            axpy_mut(&mut d_weights, 0.5 * VA_ij, &bsp_ij_d_weights);
        }
        Output { value, d_weights, d_deltas }
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
        pub ccoord_ij: f64,
        pub hcoord_ij: f64,
        // precomputed spline that depends on the coordination at i and the atom type at j
        pub P_ij: PSpline,
        // cosines of this bond with every other bond at i, and their weights
        pub types_k: &'a [AtomType],
        pub weights_ik: &'a [f64],
        pub coses_ijk: &'a [f64], // cosine between i->j and i->k
        // one of the items in the arrays is the ij bond (we must ignore it)
        pub skip_index: usize,
    }
    pub struct BondOrderSigmaPi {
        pub value: f64,
        pub d_ccoord_ij: f64,
        pub d_hcoord_ij: f64,
        // values at skip_index are unspecified
        pub d_coses_ijk: SiteBondVec<f64>,
        pub d_weights_ik: SiteBondVec<f64>,
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
            params,
            type_i, type_j,
            ccoord_ij, hcoord_ij, P_ij,
            types_k, weights_ik, coses_ijk,
            skip_index,
        } = input;
        let tcoord_ij = ccoord_ij + hcoord_ij;
        let tcoord_ij_d_ccoord_ij = 1.0;
        let tcoord_ij_d_hcoord_ij = 1.0;

        // properties of the stuff in the square root
        let inner_value;
        let inner_d_ccoord_ij;
        let inner_d_hcoord_ij;
        let inner_d_coses_ijk;
        let inner_d_weights_ik;
        {
            let mut tmp_value = 0.0;
            let mut tmp_d_ccoord_ij = 0.0;
            let mut tmp_d_hcoord_ij = 0.0;
            let mut tmp_d_coses_ijk = SiteBondVec::new();
            let mut tmp_d_weights_ik = SiteBondVec::new();

            // 1 + P_{ij}(N_i^C, N_i^H)
            //   + sum_{k /= i, j} e^{\lambda_{ijk}} f^c(r_{ik}) G(cos(t_{ijk})
            tmp_value += 1.0;

            let PSpline {
                value: P,
                d_ccoord_ij: P_d_ccoord_ij,
                d_hcoord_ij: P_d_hcoord_ij,
            } = P_ij;
            tmp_value += P;
            tmp_d_ccoord_ij += P_d_ccoord_ij;
            tmp_d_hcoord_ij += P_d_hcoord_ij;

            let iter = zip_eq!(weights_ik, coses_ijk, types_k).enumerate();
            for (index_ik, (&weight_ik, &cos_ijk, &type_k)) in iter {
                if index_ik == skip_index {
                    tmp_d_coses_ijk.push(NAN);
                    tmp_d_weights_ik.push(NAN);
                } else {
                    let exp_lambda = match params.use_airebo_lambda {
                        true => airebo_exp_lambda(type_i, (type_j, type_k)),
                        false => brenner_exp_lambda(type_i, (type_j, type_k)),
                    };
                    println!("rs-explambda: {}", exp_lambda);
                    println!("rs-tcoord: {}", tcoord_ij);

                    let GSpline {
                        value: G,
                        d_cos_ijk: G_d_cos_ijk,
                        d_tcoord_ij: G_d_tcoord_ij,
                    } = g_spline::Input { params, type_i, cos_ijk, tcoord_ij }.compute();
                    println!("rs-g gc dN {} {} {}", G, G_d_cos_ijk, G_d_tcoord_ij);

                    let G_d_ccoord_ij = G_d_tcoord_ij * tcoord_ij_d_ccoord_ij;
                    let G_d_hcoord_ij = G_d_tcoord_ij * tcoord_ij_d_hcoord_ij;

                    tmp_value += exp_lambda * weight_ik * G;
                    tmp_d_ccoord_ij += exp_lambda * weight_ik * G_d_ccoord_ij;
                    tmp_d_hcoord_ij += exp_lambda * weight_ik * G_d_hcoord_ij;
                    tmp_d_coses_ijk.push(exp_lambda * weight_ik * G_d_cos_ijk);
                    tmp_d_weights_ik.push(exp_lambda * 1.0 * G);
                }
            }

            inner_value = tmp_value;
            inner_d_ccoord_ij = tmp_d_ccoord_ij;
            inner_d_hcoord_ij = tmp_d_hcoord_ij;
            inner_d_coses_ijk = tmp_d_coses_ijk;
            inner_d_weights_ik = tmp_d_weights_ik;
        }

        // Now take the square root.
        //
        // (d/dx) sqrt(f(x))  =  (1/2) (df/dx) / sqrt(f(x))
        let value = f64::sqrt(inner_value);
        let prefactor = 0.5 / value;
        Output {
            value: value,
            d_ccoord_ij: prefactor * inner_d_ccoord_ij,
            d_hcoord_ij: prefactor * inner_d_hcoord_ij,
            d_coses_ijk: sbvec_scaled(prefactor, inner_d_coses_ijk),
            d_weights_ik: sbvec_scaled(prefactor, inner_d_weights_ik),
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
        pub tcoords_k: &'a [f64],
        pub tcoords_l: &'a [f64],
        pub weights_ik: &'a [f64],
        pub weights_jl: &'a [f64],
    }
    pub struct BondOrderPi {
        pub value: f64,
        pub d_tcoords_k: SiteBondVec<f64>,
        pub d_tcoords_l: SiteBondVec<f64>,
        pub d_weights_ik: SiteBondVec<f64>,
        pub d_weights_jl: SiteBondVec<f64>,
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
            d_weights_ik: ycoord_ij_d_weights_ik,
            d_tcoords_k: ycoord_ij_d_tcoords_k,
        } = ycoord::Input {
            skip_index: index_ij,
            weights_ik: weights_ik,
            tcoords_k: tcoords_k,
            types_k: &types_k,
        }.compute();

        let YCoord {
            value: ycoord_ji,
            d_weights_ik: ycoord_ji_d_weights_jl,
            d_tcoords_k: ycoord_ji_d_tcoords_l,
        } = ycoord::Input {
            skip_index: index_ji,
            weights_ik: weights_jl,
            tcoords_k: tcoords_l,
            types_k: &types_l,
        }.compute();

        // NConj = 1 + (square sum over bonds ik) + (square sum over bonds jl)
        let xcoord_ij = 1.0 + ycoord_ij + ycoord_ji;
        let xcoord_ij_d_ycoord_ij = 1.0;
        let xcoord_ij_d_ycoord_ji = 1.0;

        let weight_ij = weights_ik[index_ij];
        let weight_ji = weights_jl[index_ji];

        // (these are not flipped; tcoords_k and tcoords_l describe the tcoords of
        //  the *target* atoms, so tcoord_i is in tcoords_l and etc.)
        let tcoord_i = tcoords_l[index_ji];
        let tcoord_j = tcoords_k[index_ij];

        let tcoord_ij = tcoord_i - weight_ij;
        let tcoord_ij_d_tcoord_i = 1.0;
        let tcoord_ij_d_weight_ij = -1.0;

        let tcoord_ji = tcoord_j - weight_ji;
        let tcoord_ji_d_tcoord_j = 1.0;
        let tcoord_ji_d_weight_ji = -1.0;

        // Accumulators for the output value,
        // initially formulated in terms of a larger set of variables.
        let mut tmp_value = 0.0;
        let mut tmp_d_weights_ik = sbvec_filled(0.0, types_k.len());
        let mut tmp_d_weights_jl = sbvec_filled(0.0, types_l.len());
        let mut tmp_d_deltas_ik = sbvec_filled(V3::zero(), types_k.len());
        let mut tmp_d_deltas_jl = sbvec_filled(V3::zero(), types_l.len());
        let mut tmp_d_tcoord_ij = 0.0;
        let mut tmp_d_tcoord_ji = 0.0;
        let mut tmp_d_xcoord_ij = 0.0;

        // First term has a T_ij prefactor, which is zero for non-CC bonds.
        if (type_i, type_j) == (AtomType::Carbon, AtomType::Carbon) {
            // sum = ∑_{k,l} sin^2(Θ_{ijkl}) f_{ik} f_{jl}
            let sum;
            let (sum_d_weights_ik, sum_d_deltas_ik); // w.r.t. bonds around site i
            let (sum_d_weights_jl, sum_d_deltas_jl); // w.r.t. bonds around site j
            { // scope temporaries
                let mut tmp_sum = 0.0;
                let mut tmp_sum_d_weights_ik = sbvec_filled(0.0, weights_ik.len());
                let mut tmp_sum_d_weights_jl = sbvec_filled(0.0, weights_jl.len());
                let mut tmp_sum_d_deltas_ik = sbvec_filled(V3::zero(), weights_ik.len());
                let mut tmp_sum_d_deltas_jl = sbvec_filled(V3::zero(), weights_jl.len());

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
                        let weight_ik = weights_ik[index_ik];
                        let weight_jl = weights_jl[index_jl];

                        tmp_sum += sinsq * weight_ik * weight_jl;

                        tmp_sum_d_deltas_ik[index_ij] += sinsq_d_delta_ij * weight_ik * weight_jl;
                        tmp_sum_d_deltas_ik[index_ik] += sinsq_d_delta_ik * weight_ik * weight_jl;
                        tmp_sum_d_deltas_jl[index_jl] += sinsq_d_delta_jl * weight_ik * weight_jl;

                        tmp_sum_d_weights_ik[index_ik] += sinsq * 1.0 * weight_jl;
                        tmp_sum_d_weights_jl[index_jl] += sinsq * weight_ik * 1.0;
                    } // for bond_jl
                } // for bond_ik
                sum = tmp_sum;
                sum_d_deltas_ik = tmp_sum_d_deltas_ik;
                sum_d_deltas_jl = tmp_sum_d_deltas_jl;
                sum_d_weights_ik = tmp_sum_d_weights_ik;
                sum_d_weights_jl = tmp_sum_d_weights_jl;
            } // scope temporaries

            let TSpline {
                value: T,
                d_tcoord_ij: T_d_tcoord_ij,
                d_tcoord_ji: T_d_tcoord_ji,
                d_xcoord_ij: T_d_xcoord_ij,
            } = t_spline::Input { params, type_i, type_j, tcoord_ij, tcoord_ji, xcoord_ij }.compute();

            tmp_value += T * sum;
            tmp_d_tcoord_ij += T_d_tcoord_ij * sum;
            tmp_d_tcoord_ji += T_d_tcoord_ji * sum;
            tmp_d_xcoord_ij += T_d_xcoord_ij * sum;
            axpy_mut(&mut tmp_d_weights_ik, T, &sum_d_weights_ik);
            axpy_mut(&mut tmp_d_weights_jl, T, &sum_d_weights_jl);
            axpy_mut(&mut tmp_d_deltas_ik, T, &sum_d_deltas_ik);
            axpy_mut(&mut tmp_d_deltas_jl, T, &sum_d_deltas_jl);
        }

        // Second term: Just F.
        let f_input = f_spline::Input { params, type_i, type_j, tcoord_ij, tcoord_ji, xcoord_ij };
        if !f_input.can_assume_zero() {
            warn!("untested codepath: 2dc43f8b-12a2-46ca-8654-82f447664c04");
            let FSpline {
                value: F,
                d_tcoord_ij: F_d_tcoord_ij,
                d_tcoord_ji: F_d_tcoord_ji,
                d_xcoord_ij: F_d_xcoord_ij,
            } = f_input.compute();

            tmp_value += F;
            tmp_d_tcoord_ij += F_d_tcoord_ij;
            tmp_d_tcoord_ji += F_d_tcoord_ji;
            tmp_d_xcoord_ij += F_d_xcoord_ij;
        }

        // Now reformulate the value in terms of a different set of variables.
        //
        // starting with: {weights_ik}, {weights_jl}, {deltas_ik}, {deltas_jl},
        //                tcoord_ij, tcoord_ji, xcoord_ij
        //
        //     change to: {weights_ik}, {weights_jl}, {deltas_ik}, {deltas_jl},
        //                ycoord_ij, ycoord_ji, tcoord_i,  tcoord_j,
        //
        let value = tmp_value;
        let d_deltas_ik = tmp_d_deltas_ik;
        let d_deltas_jl = tmp_d_deltas_jl;
        let mut d_weights_ik = tmp_d_weights_ik;
        let mut d_weights_jl = tmp_d_weights_jl;

        // xcoord_ij = xcoord_ij(ycoord_ij, ycoord_ji)
        let d_ycoord_ij = tmp_d_xcoord_ij * xcoord_ij_d_ycoord_ij;
        let d_ycoord_ji = tmp_d_xcoord_ij * xcoord_ij_d_ycoord_ji;

        // tcoord_ij = tcoord_ij(tcoord_i, weight_ij)
        let d_tcoord_i = tmp_d_tcoord_ij * tcoord_ij_d_tcoord_i;
        let d_tcoord_j = tmp_d_tcoord_ji * tcoord_ji_d_tcoord_j;
        d_weights_ik[index_ij] += tmp_d_tcoord_ij * tcoord_ij_d_weight_ij;
        d_weights_jl[index_ji] += tmp_d_tcoord_ji * tcoord_ji_d_weight_ji;

        //  and now from: {weights_ik}, {weights_jl}, {deltas_ik}, {deltas_jl},
        //                ycoord_ij, ycoord_ji, tcoord_i,  tcoord_j,
        //
        //            to: {weights_ik}, {weights_jl}, {deltas_ik}, {deltas_jl},
        //                {tcoords_k}, {tcoords_l}

        // ycoord_ij = ycoord_ij({tcoords_k}, {weights_ik})
        axpy_mut(&mut d_weights_ik, d_ycoord_ij, &ycoord_ij_d_weights_ik);
        axpy_mut(&mut d_weights_jl, d_ycoord_ji, &ycoord_ji_d_weights_jl);
        let mut d_tcoords_k = sbvec_scaled(d_ycoord_ij, ycoord_ij_d_tcoords_k);
        let mut d_tcoords_l = sbvec_scaled(d_ycoord_ji, ycoord_ji_d_tcoords_l);

        // reminder: because tcoords_k describes sites bonded to site i, it contains
        //           tcoord_j rather than tcoord_i
        d_tcoords_k[index_ij] += d_tcoord_j;
        d_tcoords_l[index_ji] += d_tcoord_i;

        Output {
            value, d_tcoords_k, d_tcoords_l, d_weights_ik,
            d_weights_jl, d_deltas_ik, d_deltas_jl,
        }
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
        pub weights_ik: &'a [f64],
        pub tcoords_k: &'a [f64],
        pub types_k: &'a [AtomType],
    }

    pub struct YCoord {
        pub value: f64,
        pub d_weights_ik: SiteBondVec<f64>,
        pub d_tcoords_k: SiteBondVec<f64>,
    }

    impl<'a> Input<'a> {
        pub fn compute(self) -> Output { compute(self) }
    }

    // free function for smaller indent
    fn compute(input: Input<'_>) -> Output {
        let Input { skip_index, weights_ik, tcoords_k, types_k } = input;

        // Compute the sum without the square
        let mut inner_value = 0.0;
        let mut inner_d_weights_ik = SiteBondVec::new();
        let mut inner_d_tcoords_k = SiteBondVec::new();
        let iter = zip_eq!(tcoords_k, weights_ik, types_k).enumerate();
        for (index_ik, (&tcoord_k, &weight_ik, &type_k)) in iter {
            if index_ik == skip_index || type_k == AtomType::Hydrogen {
                inner_d_weights_ik.push(0.0);
                inner_d_tcoords_k.push(0.0);
                continue;
            }
            let xik = tcoord_k - weight_ik;
            let xik_d_tcoord_k = 1.0;
            let xik_d_weight_ik = -1.0;

            let (F, F_d_xik) = switch((3.0, 2.0), xik);
            let mut inner_d_weight_ik = 0.0;
            let mut inner_d_tcoord_k = 0.0;
            inner_value += weight_ik * F;
            inner_d_tcoord_k += weight_ik * F_d_xik * xik_d_tcoord_k;
            inner_d_weight_ik += F;
            inner_d_weight_ik += weight_ik * F_d_xik * xik_d_weight_ik;
            inner_d_weights_ik.push(inner_d_weight_ik);
            inner_d_tcoords_k.push(inner_d_tcoord_k);
        }

        // Now square it
        let value = inner_value * inner_value;
        Output {
            value: value,
            d_tcoords_k: sbvec_scaled(2.0 * value, inner_d_tcoords_k),
            d_weights_ik: sbvec_scaled(2.0 * value, inner_d_weights_ik),
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
        pub tcoord_ij: f64,
        pub cos_ijk: f64,
    }

    pub struct GSpline {
        pub value: f64,
        pub d_tcoord_ij: f64,
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
                let d_tcoord_ij = 0.0;
                let (value, d_cos_ijk) = $poly.evaluate(cos_ijk);
                Output { value, d_cos_ijk, d_tcoord_ij }
            }}
        }

        match type_i {
            AtomType::Carbon => {
                let switch_interval = (params.G.low_coord, params.G.high_coord);
                let (alpha, alpha_d_tcoord_ij) = switch(switch_interval, tcoord_ij);

                if alpha == 0.0 && alpha_d_tcoord_ij == 0.0 {
                    use_single_poly!(&params.G.carbon_low_coord)
                } else if alpha == 1.0 && alpha_d_tcoord_ij == 0.0 {
                    warn!("untested codepath: 37236e5f-9810-4ee5-a8c3-0a5150d9bd26");
                    use_single_poly!(&params.G.carbon_high_coord)
                } else {
                    warn!("untested codepath: fd6eff7e-b4b2-4bbf-ad89-3227a6099d59");
                    // The one case where use_single_poly! cannot be used.

                    // d(linterp(α, A, B)) = d(α A + (1 - α) B)
                    //                     = (A - B) dα + α dA + (1 - α) dB
                    //                     = ...let's not do this right now
                    unimplemented!()
                }
            },
            AtomType::Hydrogen => {
                warn!("untested codepath: 4d03fe04-5312-468e-9e30-01beddec4793");
                use_single_poly!(&params.G.hydrogen)
            },
        }
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
            let GSpline { value, d_cos_ijk, d_tcoord_ij } = Input {
                params,
                type_i: AtomType::Carbon,
                cos_ijk: f64::cos(120.0 * PI / 180.0) + 1e-12,
                tcoord_ij: 3.0,
            }.compute();
            // Brenner Table 3
            assert_close!(rel=tol, value, 0.05280);
            assert_close!(rel=tol, d_cos_ijk, 0.17000);
            assert_eq!(d_tcoord_ij, 0.0);

            let GSpline { value, d_cos_ijk, d_tcoord_ij } = Input {
                params,
                type_i: AtomType::Carbon,
                cos_ijk: f64::cos(120.0 * PI / 180.0) - 1e-12,
                tcoord_ij: 3.0,
            }.compute();
            assert_close!(rel=tol, value, 0.05280);
            assert_close!(rel=tol, d_cos_ijk, 0.17000);
            assert_eq!(d_tcoord_ij, 0.0);

            // diamond
            let GSpline { value, d_cos_ijk, d_tcoord_ij } = Input {
                params,
                type_i: AtomType::Carbon,
                cos_ijk: -1.0/3.0 + 1e-12,
                tcoord_ij: 4.0,
            }.compute();
            assert_close!(rel=tol, value, 0.09733);
            assert_close!(rel=tol, d_cos_ijk, 0.40000);
            assert_eq!(d_tcoord_ij, 0.0);

            let GSpline { value, d_cos_ijk, d_tcoord_ij } = Input {
                params,
                type_i: AtomType::Carbon,
                cos_ijk: -1.0/3.0 - 1e-12,
                tcoord_ij: 4.0,
            }.compute();
            assert_close!(rel=tol, value, 0.09733);
            assert_close!(rel=tol, d_cos_ijk, 0.40000);
            assert_eq!(d_tcoord_ij, 0.0);
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

                let mut tcoords = vec![
                    3.0,
                    // FIXME: points around 3.2 and 3.7 once the interpolation is implemented
                ];
                for &cos_ijk in &coses {
                    for &tcoord_ij in &tcoords {
                        let input = Input { params, type_i, cos_ijk, tcoord_ij };
                        let GSpline { value: _, d_cos_ijk, d_tcoord_ij } = input.compute();
                        assert_close!(
                            rel=tol, abs=tol,
                            d_cos_ijk,
                            numerical::slope(
                                1e-7, None,
                                cos_ijk,
                                |cos_ijk| Input { params, type_i, cos_ijk, tcoord_ij }.compute().value,
                            ),
                        );
                        assert_close!(
                            rel=tol, abs=tol,
                            d_tcoord_ij,
                            numerical::slope(
                                1e-7, None,
                                tcoord_ij,
                                |tcoord_ij| Input { params, type_i, cos_ijk, tcoord_ij }.compute().value,
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
                for i in 1..spline.poly.len() {
                    // Should be continuous up to 2nd derivative
                    let x = spline.x_div[i];
                    let poly_a = spline.poly[i-1].clone();
                    let poly_b = spline.poly[i].clone();
                    let poly_da = poly_a.derivative();
                    let poly_db = poly_b.derivative();
                    let poly_dda = poly_da.derivative();
                    let poly_ddb = poly_db.derivative();
                    assert_close!(rel=tol, poly_a.evaluate(x).0, poly_b.evaluate(x).0);
                    assert_close!(rel=tol, poly_da.evaluate(x).0, poly_db.evaluate(x).0);
                    assert_close!(rel=tol, poly_dda.evaluate(x).0, poly_ddb.evaluate(x).0);
                }
            }
        }
    }
}

use self::p_spline::PSpline;
mod p_spline {
    use super::*;

    use self::splines::bicubic;

    pub type Output = PSpline;
    pub struct Input<'a> {
        pub params: &'a Params,
        pub type_i: AtomType,
        pub type_j: AtomType,
        pub ccoord_ij: f64,
        pub hcoord_ij: f64,
    }

    pub struct PSpline {
        pub value: f64,
        pub d_ccoord_ij: f64,
        pub d_hcoord_ij: f64,
    }

    impl Output {
        fn zero() -> Output {
            Output { value: 0.0, d_ccoord_ij: 0.0, d_hcoord_ij: 0.0 }
        }
    }

    impl<'a> Input<'a> {
        pub fn compute(&self) -> Output {
            let Input { ref params, type_i, type_j, ccoord_ij, hcoord_ij } = *self;

            let poly = match (type_i, type_j) {
                (AtomType::Hydrogen, _) => return Output::zero(),
                (AtomType::Carbon, AtomType::Carbon) => &params.P.CC,
                (AtomType::Carbon, AtomType::Hydrogen) => &params.P.CH,
            };
            let (value, grad) = poly.evaluate(V2([ccoord_ij, hcoord_ij]));
            let V2([d_ccoord_ij, d_hcoord_ij]) = grad;

            Output { value, d_ccoord_ij, d_hcoord_ij }
        }
    }
}

use self::f_spline::FSpline;
mod f_spline {
    use super::*;

    // FIXME:
    //
    // The fitting data for the F spline is a terror to behold.
    // Brenner's table also contains numerous suspicious-looking things (i.e. possible errors)
    // that I don't want to have to fret over right now.
    //
    // For that reason, we don't actually currently use the tricubic splines, and instead just
    // check for a few special cases.
    pub type Output = FSpline;
    pub struct Input<'a> {
        pub params: &'a Params,
        pub type_i: AtomType,
        pub type_j: AtomType,
        pub tcoord_ij: f64,
        pub tcoord_ji: f64,
        pub xcoord_ij: f64,
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

    pub struct FSpline {
        pub value: f64,
        pub d_tcoord_ij: f64,
        pub d_tcoord_ji: f64,
        pub d_xcoord_ij: f64,
    }

    impl FSpline {
        fn zero() -> Self {
            FSpline {
                value: 0.0,
                d_tcoord_ij: 0.0,
                d_tcoord_ji: 0.0,
                d_xcoord_ij: 0.0,
            }
        }

        fn flip(self) -> Self {
            FSpline {
                value: self.value,
                d_tcoord_ij: self.d_tcoord_ji,
                d_tcoord_ji: self.d_tcoord_ij,
                d_xcoord_ij: self.d_xcoord_ij,
            }
        }
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
            (AtomType::Hydrogen, AtomType::Carbon) => return input.flip().compute().flip(),
            (AtomType::Carbon, AtomType::Carbon) => &params.F.CC,
            (AtomType::Carbon, AtomType::Hydrogen) => &params.F.CH,
            (AtomType::Hydrogen, AtomType::Hydrogen) => &params.F.HH,
        };
        let (value, grad) = poly.evaluate(V3([tcoord_ij, tcoord_ji, xcoord_ij]));
        let V3([d_tcoord_ij, d_tcoord_ji, d_xcoord_ij]) = grad;

        Output { value, d_tcoord_ij, d_tcoord_ji, d_xcoord_ij }
    }

    // check if the value and all derivatives can be assumed to be zero,
    // without needing to compute N^{conj}, because I currently have
    // very little confidence in the spline for this one and would rather
    // bail out than use it (until it is better tested).
    impl<'a> Input<'a> {
        pub fn can_assume_zero(&self) -> bool {
            let Input { params: _, type_i, type_j, tcoord_ij, tcoord_ji, xcoord_ij } = *self;

            let frac_point = V3([tcoord_ij, tcoord_ji, xcoord_ij]);
            let int_point = frac_point.map(|x| x as i32);
            if frac_point != int_point.map(|x| x as f64) {
                return false;
            }

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
}

use self::t_spline::TSpline;
mod t_spline {
    //! T spline
    //!
    //! * Brenner, Table 5
    //! * Stuart, Table X

    use super::*;

    pub type Output = TSpline;
    pub struct Input<'a> {
        pub params: &'a Params,
        pub type_i: AtomType,
        pub type_j: AtomType,
        pub tcoord_ij: f64,
        pub tcoord_ji: f64,
        pub xcoord_ij: f64,
    }

    pub struct TSpline {
        pub value: f64,
        pub d_tcoord_ij: f64,
        pub d_tcoord_ji: f64,
        pub d_xcoord_ij: f64,
    }

    impl Output {
        fn zero() -> Output {
            Output {
                value: 0.0,
                d_tcoord_ij: 0.0,
                d_tcoord_ji: 0.0,
                d_xcoord_ij: 0.0,
            }
        }
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
            (_, AtomType::Hydrogen) => return Output::zero(),
        };
        let (value, grad) = poly.evaluate(V3([tcoord_ij, tcoord_ji, xcoord_ij]));
        let V3([d_tcoord_ij, d_tcoord_ji, d_xcoord_ij]) = grad;

        Output { value, d_tcoord_ij, d_tcoord_ji, d_xcoord_ij }
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
