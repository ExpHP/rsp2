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

//! Implementation of the REBO reactive potential. (currently without any of the reactive bits)
//!
//! # Citations
//!
//! * Donald W Brenner et al 2002 J. Phys.: Condens. Matter 14 783

use ::math::bond_graph::PeriodicGraph;
use ::meta;

use ::stack::{ArrayVec, Vector as StackVector};
#[cfg(test)]
use ::std::f64::{consts::PI};
use ::std::f64::NAN;
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

// FIXME remove
pub type FailResult<T> = Result<T, ::failure::Error>;

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
//newtype_index!{TripletI}
//newtype_index!{DihedralI}

pub use self::params::Params;
mod params {
    use super::*;

    // TODO: Use LAMMPS' parameters
    #[derive(Debug, Clone)]
    pub struct Params {
        pub by_type: TypeMap<TypeMap<TypeParams>>,
        pub G: brenner_G::SplineSet,
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
                G: brenner_G::BRENNER_SPLINES,
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
                G: brenner_G::LAMMPS_SPLINES,
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
        bond_is_canonical: IndexVec<BondI, bool>,
        bond_target: IndexVec<BondI, SiteI>,
    }

    impl Interactions {
        pub fn compute(
            coords: &Coords,
            types: &[AtomType],
            bond_graph: &PeriodicGraph,
        ) -> FailResult<Self> {
            let mut bond_div = IndexVec::<SiteI, _>::from_raw(vec![BondI(0)]);
//            let mut triplet_div = IndexVec::<BondI, _>::from_raw(vec![TripletI(0)]);
//            let mut dihedral_div = IndexVec::<TripletI, _>::from_raw(vec![DihedralI(0)]);
            let mut bond_cart_vector = IndexVec::<BondI, _>::new();
            let mut bond_is_canonical = IndexVec::<BondI, _>::new();
            let mut bond_target = IndexVec::<BondI, SiteI>::new();
//            let mut triplet_bond = IndexVec::<TripletI, BondI>::new();
//            let mut dihedral_bond = IndexVec::<DihedralI, BondI>::new();
            let site_type = IndexVec::<SiteI, _>::from_raw(types.to_vec());

            let cart_cache = coords.with_carts(coords.to_carts());

            // Make a pass to get all the bond divs right.
            for node in bond_graph.node_indices() {
                let site_i = SiteI(node.index());

                for frac_bond_ij in bond_graph.frac_bonds_from(site_i.index()) {
                    let site_j = SiteI(frac_bond_ij.to);
                    let cart_vector = frac_bond_ij.cart_vector_using_cache(&cart_cache).unwrap();
                    bond_target.push(site_j);
                    bond_is_canonical.push(frac_bond_ij.is_canonical());
                    bond_cart_vector.push(cart_vector);
                } // for bond_ij

                let num_bonds = bond_target.len() - bond_div.raw.last().unwrap().index();
                if num_bonds > SITE_MAX_BONDS {
                    bail!("An atom has too many bonds! ({}, max: {})", num_bonds, SITE_MAX_BONDS);
                }
                bond_div.push(BondI(bond_target.len()));
            } // for node

//            { // FIXME: block will be unnecessary after NLL lands
//                // now that all bond_divs are prepped, we can get the BondI indices
//                // of any arbitrary site
//                let bonds_enumerated_from = |site: SiteI| {
//                    let first_bond = bond_div[site];
//                    idx::iota(first_bond).zip(bond_graph.frac_bonds_from(site.index()))
//                };
//
//                // Deeper pass for triples and dihedrals.
//                // Apologies for the horrendously nested loop...
//                for node in bond_graph.node_indices() {
//                    let site_i = SiteI(node.index());
//                    let bonds_i_enumerated = || bonds_enumerated_from(site_i);
//                    for (bond_ij, frac_bond_ij) in bonds_i_enumerated() {
//                        let site_j = SiteI(frac_bond_ij.to);
//
//                        if frac_bond_ij.is_canonical() {
//                            // Iterate over the other bonds attached to site i for triplets
//                            for (bond_ik, frac_bond_ik) in bonds_i_enumerated() {
//                                if bond_ij == bond_ik {
//                                    continue;
//                                }
//                                let type_i = site_type[site_i];
//                                let type_j = site_type[site_j];
//
//                                // Get another bond attached to j for dihedrals
//                                if (type_i, type_j) == (AtomType::Carbon, AtomType::Carbon) {
//                                    let bonds_j_enumerated = || bonds_enumerated_from(site_j);
//                                    for (bond_jl, frac_bond_jl) in bonds_j_enumerated() {
//                                        // Figure out if k and l are the same image of the same site
//                                        let frac_bond_il = frac_bond_ij.join(frac_bond_jl).unwrap();
//                                        if frac_bond_ik == frac_bond_il {
//                                            // I think I can see my house from here!
//                                            continue;
//                                        }
//                                        dihedral_bond.push(bond_jl);
//                                    }
//                                }
//                                triplet_bond.push(bond_ik);
//                                dihedral_div.push(DihedralI(dihedral_bond.len()));
//                            } // for bond_ik
//                        } // if is_canonical
//                        triplet_div.push(TripletI(triplet_bond.len()));
//                    } // for bond_ij
//                } // for node
//            } // NLL workaround

            Ok(Interactions {
                bond_div, site_type, bond_cart_vector,
                bond_is_canonical, bond_target,
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
        pub target: SiteI,
        pub cart_vector: V3,
    }

//    pub struct Triplet {
//        pub index: TripletI,
//        pub bond_ik: BondI,
//    }
//
//    pub struct Dihedral {
//        pub bond_jl: BondI,
//    }

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
            let target = self.bond_target[index];
            Bond { index, is_canonical, cart_vector, target }
        }

        pub fn sites(&self) -> impl ExactSizeIterator<Item=Site> + '_ {
            (0..self.site_type.len()).map(move |i| self.site(SiteI(i)))
        }

        pub fn site_range(&self) -> ::std::ops::Range<usize> {
            0..self.site_type.len()
        }

        pub fn bonds(&self, site: SiteI) -> impl ExactSizeIterator<Item=Bond> + '_ {
            self.bond_range(site).map(move |i| self.bond(BondI(i)))
        }

        pub fn bond_range(&self, site: SiteI) -> ::std::ops::Range<usize> {
            self.bond_div[site].0..self.bond_div[site.next()].0
        }

//        pub fn triplets(&self, bond: BondI) -> impl ExactSizeIterator<Item=Triplet> + '_ {
//            idx::range(self.triplet_div[bond]..self.triplet_div[bond.next()]).map(move |index| {
//                let bond_ik = self.triplet_bond[index];
//                Triplet { index, bond_ik }
//            })
//        }
//
//        pub fn dihedrals(&self, triplet: TripletI) -> impl ExactSizeIterator<Item=Dihedral> + '_ {
//            idx::range(self.dihedral_div[triplet]..self.dihedral_div[triplet.next()])
//                .map(move |index| {
//                    let bond_jl = self.dihedral_bond[index];
//                    Dihedral { bond_jl }
//                })
//        }
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
        let type_i = __site.atom_type;
        let FirstPassSiteData {
            tcoord: tcoord_i,
            ref bond_weight, ref bond_weight_d_delta,
            ref bond_VR, ref bond_VR_d_delta,
            ref bond_VA, ref bond_VA_d_delta,
        } = site_data[site_i];

        // FIXME simplification: Assume the pi bondorder is zero.
        //
        //       This means that there's only a sigma-pi term for each bond,
        //       whose derivatives are entirely of data local to the originating site
        //       (i.e. we can fit them in a SiteBondVec).
        //
        //       This considerably simplifies the evaluation strategy for the rest of the
        //       function, because we can simply visit each site in order and concatenate
        //       the derivatives.

        // Eq 3':  b_ij = 0.5 * b_ij^{sigma-pi} + boole(i < j) * b_ij^{pi}
        for (index_ij, __bond) in interactions.bonds(site_i).enumerate() {
            let site_j = __bond.target;
            let tcoord_j = site_data[site_j].tcoord;
            let type_j = interactions.site(site_j).atom_type;
            let weight_ij = bond_weight[index_ij];

            let weight_ji = weight_ij;

            // This is what Ni almost always *actually* means in Brenner.
            // (he really should have called it Nij, with two indices; Stuart's AIREBO
            //  paper does it better)
            let tcoord_ij = tcoord_i - weight_ij;
            let tcoord_ji = tcoord_j - weight_ji;

            // boole(i < j) * b_ij^{pi}
            if __bond.is_canonical {
                if !brenner_T::can_assume_zero((type_i, type_j), (tcoord_ij, tcoord_ji)) {
                    panic!("brenner T spline may be nonzero; this is not yet implemented");
                }
                if !brenner_F::can_assume_zero((type_i, type_j), (tcoord_ij, tcoord_ji)) {
                    panic!("brenner F spline may be nonzero; this is not yet implemented");
                }
            }
        }

        // Eq 2':  V_ij = 0.5 * V^R_ij - b_ij * V^A_ij
        //
        // As stated, with the above assumption that b^pi = 0, each site's terms
        // can only depend on the derivatives of that site's bonds, so they fit
        // in a SiteBondVec.

        let mut site_V = 0.0;
        let mut site_V_d_delta = sbvec_filled(V3::zero(), bond_weight.len());

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

        // Add in the repulsive terms, which each only depend on one bond delta.
        // (written tersely here for vectorization...)
        site_V += 0.5 * bond_VR.iter().sum::<f64>();
        axpy_mut(&mut site_V_d_delta, 0.5, &bond_VR_d_delta);

        // Attractive terms are fully encompassed in Vsp
        site_V -= Vsp_i;
        axpy_mut(&mut site_V_d_delta, -1.0, &Vsp_i_d_deltas);
        for (index_ij, _) in interactions.bonds(site_i).enumerate() {
            let Vsp_i_d_weight_ij = Vsp_i_d_weights[index_ij];
            let weight_ij_d_delta_ij = bond_weight_d_delta[index_ij];
            site_V_d_delta[index_ij] += -1.0 * Vsp_i_d_weight_ij * weight_ij_d_delta_ij;
        }

        (site_V, site_V_d_delta)

    // well this is awkward
    }).fold(
        || (0.0, IndexVec::new()),
        |(mut value, mut d_deltas), (site_V, site_V_d_delta)| {
            // because of our b_ij^pi = 0 assumption,
            // we can just concatenate the derivatives from each site.
            value += site_V;
            d_deltas.extend(site_V_d_delta);
            (value, d_deltas)
        },
    ).reduce(
        // (despite all appearances, the second closure here is different from the one above
        //  due to the types of the arguments; above, site_V is BondSiteVec; here it is IndexVec)
        || (0.0, IndexVec::new()),
        |(mut value, mut d_deltas), (value_part, d_delta_part)| {
            value += value_part;
            d_deltas.extend(d_delta_part);
            (value, d_deltas)
        },
    );
    let (value, d_deltas) = out;
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
            let P_ij = brenner_P::Input { type_i, type_j, ccoord_ij, hcoord_ij }.compute();

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
            }

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
        pub P_ij: BrennerP,
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

            let BrennerP {
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

                    let BrennerG {
                        value: G,
                        d_cos_ijk: G_d_cos_ijk,
                        d_tcoord_ij: G_d_tcoord_ij,
                    } = brenner_G::Input { params, type_i, cos_ijk, tcoord_ij }.compute();
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

//// b_{ij}^{DH} in Brenner (equation 18)
//use self::bond_order_dihedral::BondOrderDihedral;
//mod bond_order_dihedral {
//    use super::*;
//
//    pub type Output = BondOrderDihedral;
//    pub struct Input<I: IntoIterator<Item=DihedralItem>> {
//        pub type_i: AtomType,
//        pub type_j: AtomType,
//        pub tcoord_ij: f64,
//        pub tcoord_ji: f64,
//        pub xcoord_ij: f64,
//        pub dihedrals_ijkl: I,
//    }
//    pub struct DihedralItem {
//        pub sinsq_ijkl: f64,
//        pub weight_ik: f64,
//        pub weight_jl: f64,
//    }
//    pub struct BondOrderDihedral {
//        pub value: f64,
//        pub d_tcoord_ij: f64,
//        pub d_tcoord_ji: f64,
//        pub d_xcoord_ij: f64,
//        pub d_sinsqs_ijkl: Vec<f64>,
//        pub d_weights_ik: Vec<f64>,
//        pub d_weights_jl: Vec<f64>,
//    }
//
//    impl<I: IntoIterator<Item=DihedralItem>> Input<I> {
//        pub fn compute(self, preallocated: Option<Output>) -> Output {
//            compute(self, preallocated.unwrap_or_default())
//        }
//    }
//
//    // free function for smaller indent
//    fn compute(
//        input: Input<impl IntoIterator<Item=DihedralItem>>,
//        preallocated: Output,
//    ) -> Output {
//        // value = T * sum
//
//        // T = a tricubic spline
//        let Input { type_i, type_j, tcoord_ij, tcoord_ji, xcoord_ij, dihedrals_ijkl } = input;
//        let BrennerT {
//            value: T,
//            d_tcoord_ij: T_d_tcoord_ij,
//            d_tcoord_ji: T_d_tcoord_ji,
//            d_xcoord_ij: T_d_xcoord_ij,
//        } = brenner_T::Input { type_i, type_j, tcoord_ij, tcoord_ji, xcoord_ij }.compute();
//
//        // sum = ∑_{k,l} ((1 - cos(Θ_{ijkl})^2) f_{ik} f_{jl})
//        let mut sum = 0.0;
//        let mut sum_d_sinsqs_ijkl = cleared(preallocated.d_sinsqs_ijkl);
//        let mut sum_d_weights_ik = cleared(preallocated.d_weights_ik);
//        let mut sum_d_weights_jl = cleared(preallocated.d_weights_jl);
//        for item in dihedrals_ijkl {
//            let DihedralItem { sinsq_ijkl, weight_ik, weight_jl } = item;
//            sum += sinsq_ijkl * weight_ik * weight_jl;
//            sum_d_sinsqs_ijkl.push(1.0 * weight_ik * weight_jl);
//            sum_d_weights_ik.push(sinsq_ijkl * 1.0 * weight_jl);
//            sum_d_weights_jl.push(sinsq_ijkl * weight_ik * 1.0);
//        }
//
//        // output is product
//        Output {
//            value: T * sum,
//            d_tcoord_ij: T_d_tcoord_ij * sum,
//            d_tcoord_ji: T_d_tcoord_ji * sum,
//            d_xcoord_ij: T_d_xcoord_ij * sum,
//            d_sinsqs_ijkl: scaled(T, sum_d_sinsqs_ijkl),
//            d_weights_ik: scaled(T, sum_d_weights_ik),
//            d_weights_jl: scaled(T, sum_d_weights_jl),
//        }
//    }
//}
//
//// The coordination value describing local conjugacy (N^{conj} in Brenner)
////
//// FIXME: Terrible signal to noise ratio. This is almost all boilerplate.
////        Probably better to just inline the computation wherever it ultimately gets used.
////
////        I keep wanting to just precompute all of the XCoordSumsqs, but the trouble is that
////        they depend on the coordinations of neighboring sites. This in turn means it depends
////        on the weights of many bonds that don't even involve site i.
////
////        Not only are these derivatives exhausting to keep track of, but dealing with them
////        poorly will also probably lead to poor memory access patterns.
//mod xcoord {
//    use super::*;
//
//    pub type Output = XCoord;
//    pub struct Input<'a> {
//        pub skip_index_k: usize, // index of ij bond in i bond arrays
//        pub skip_index_l: usize, // index of ji bond in j bond arrays
//        pub weights_ik: &'a [f64],
//        pub weights_jl: &'a [f64],
//        pub tcoords_k: &'a [f64],
//        pub tcoords_l: &'a [f64],
//    }
//
//    pub struct XCoord {
//        pub value: f64,
//        pub d_weights_ik: SiteBondVec<f64>,
//        pub d_weights_jl: SiteBondVec<f64>,
//        pub d_tcoords_k: SiteBondVec<f64>,
//        pub d_tcoords_l: SiteBondVec<f64>,
//    }
//
//    impl<'a> Input<'a> {
//        pub fn compute(self) -> Output {
//            let Input {
//                skip_index_k, skip_index_l,
//                weights_ik, weights_jl,
//                tcoords_k, tcoords_l,
//                types_k, types_l,
//            } = self;
//            assert!(skip_index_k < tcoords_k.len());
//            assert!(skip_index_l < tcoords_l.len());
//            assert_eq!(weights_ik.len(), tcoords_k.len());
//            assert_eq!(weights_jl.len(), tcoords_l.len());
//
//            let out = xcoord_sumsq::Input {
//                skip_index: skip_index_k,
//                weights_ik: weights_ik,
//                tcoords_k: tcoords_k,
//                types_k: types_k,
//            }.compute();
//            let XCoordSumsq {
//                value: sumsq_ik,
//                d_weights_ik: d_weights_ik,
//                d_tcoords_k: d_tcoords_k,
//            } = out;
//
//            let out = xcoord_sumsq::Input {
//                skip_index: skip_index_j,
//                weights_ik: weights_jl,
//                tcoords_k: tcoords_j,
//                types_k: types_l,
//            }.compute();
//            let XCoordSumsq {
//                value: sumsq_jl,
//                d_weights_ik: d_weights_jl,
//                d_tcoords_k: d_tcoords_l,
//            } = out;
//
//            // Hmmm. That was 60 lines of boilerplate just for a simple sum. :/
//            let value = 1.0 + sumsq_ik + sumsq_jl;
//            Output { value, d_weights_ik, d_weights_jl, d_tcoords_k, d_tcoords_l }
//        }
//    }
//}
//
//// One of the square sum terms that appear in the definition of N^{conj}.  (Brenner, eq. 15)
//use self::xcoord_sumsq::XCoordSumsq;
//mod xcoord_sumsq {
//    use super::*;
//
//    pub type Output = XCoordSumsq;
//    pub struct Input<'a> {
//        pub skip_index: usize,
//        pub weights_ik: &'a [f64],
//        pub tcoords_k: &'a [f64],
//        pub types_k: &'a [AtomType],
//    }
//
//    pub struct XCoordSumsq {
//        pub value: f64,
//        pub d_weights_ik: SiteBondVec<f64>,
//        pub d_tcoords_k: SiteBondVec<f64>,
//    }
//
//    impl<'a> Input<'a> {
//        pub fn compute(self) -> Output { compute(self) }
//    }
//
//    // free function for smaller indent
//    fn compute(input: Input<'_>) -> Output {
//        let Input { skip_index, weights_ik, tcoords_k, types_k } = input;
//
//        // Compute the sum without the square
//        let mut inner_value = 0.0;
//        let mut inner_d_weights_ik = SiteBondVec::new();
//        let mut inner_d_tcoords_k = SiteBondVec::new();
//        let iter = zip_eq!(&tcoords_k, &weights_ik, &types_k).enumerate();
//        for (index_ik, (&tcoord_k, &weight_ik, &type_k)) in iter {
//            if index_ik == skip_index || type_k == AtomType::Hydrogen {
//                inner_d_weights_ik.push(0.0);
//                inner_d_tcoords_k.push(0.0);
//                continue;
//            }
//            let xik = tcoord_k - weight_ik;
//            let xik_d_tcoord_k = 1.0;
//            let xik_d_weight_ik = -1.0;
//
//            let (F, F_d_xik) = switch((3.0, 2.0), xik);
//            let mut inner_d_weight_ik = 0.0;
//            let mut inner_d_tcoord_k = 0.0;
//            inner_value += weight_ik * F;
//            inner_d_tcoord_k += weight_ik * F_d_xik * xik_d_tcoord_k;
//            inner_d_weight_ik += F;
//            inner_d_weight_ik += weight_ik * F_d_xik * xik_d_weight_ik;
//            inner_d_weights_ik.push(inner_d_weight_ik);
//            inner_d_tcoords_k.push(inner_d_tcoord_k);
//        }
//
//        let scaled = |f, mut arr| { scale_mut(f, &mut arr); arr };
//
//        // Now square it
//        let value = inner_value * inner_value;
//        let prefactor = 2.0 * value;
//        Output {
//            value: value,
//            d_tcoords_k: scaled(prefactor, inner_d_tcoords_k),
//            d_weights_ik: scaled(prefactor, inner_d_weights_ik),
//        }
//    }
//}

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

            assert_close!(rel=1e-5, output_d_delta_ij.0, numerical_d_delta_ij.0);
            assert_close!(rel=1e-5, output_d_delta_ik.0, numerical_d_delta_ik.0);
            assert_close!(rel=1e-5, output_d_delta_jl.0, numerical_d_delta_jl.0);
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

            let numerical_d_delta_ij = num_grad_v3(1e-5, delta_ij, |delta_ij| Input { delta_ij, delta_ik }.compute().value);
            let numerical_d_delta_ik = num_grad_v3(1e-5, delta_ik, |delta_ik| Input { delta_ij, delta_ik }.compute().value);

            // FIXME these required tolerances are ridiculous;
            //       why are we still using 1st order central difference for
            //       numerical differentiation?
            assert_close!(rel=1e-5, output_d_delta_ij.0, numerical_d_delta_ij.0);
            assert_close!(rel=1e-5, output_d_delta_ik.0, numerical_d_delta_ik.0);
        }
    }
}

use self::brenner_P::BrennerP;
mod brenner_P {
    use super::*;

    pub type Output = BrennerP;
    pub struct Input {
        pub type_i: AtomType,
        pub type_j: AtomType,
        pub ccoord_ij: f64,
        pub hcoord_ij: f64,
    }

    #[derive(Debug, Copy, Clone)]
    pub struct BrennerP {
        pub value: f64,
        pub d_ccoord_ij: f64,
        pub d_hcoord_ij: f64,
    }

    impl BrennerP {
        fn with_zero_deriv(value: f64) -> Self {
            BrennerP { value, d_ccoord_ij: 0.0, d_hcoord_ij: 0.0 }
        }
        fn zero() -> Self { Self::with_zero_deriv(0.0) }
    }

    impl Input {
        pub fn compute(self) -> Output {
            compute(self)
        }
    }

    fn compute(input: Input) -> Output {
        let Input { type_i, type_j, ccoord_ij, hcoord_ij } = input;

        // NOTE:
        //
        // RSP2 does not need the spline because it does not do anything
        // that requires the reactive parts of the potential.
        //
        // Thus, this function is a cop-out.
        let int_ccoord = ccoord_ij as i64;
        let int_hcoord = hcoord_ij as i64;
        if int_ccoord as f64 != ccoord_ij || int_hcoord as f64 != hcoord_ij {
            panic!("Fractional coordination not yet implemented for Brenner P splines.");
        }

        match (type_i, type_j) {
            (AtomType::Hydrogen, _) => Output::zero(),

            // NOTE: In the paper, Table 8 has the columns for i and j flipped.
            //
            // NOTE: My interpretation of the table is that all integer points of this spline
            //       are saddle points, as "all derivatives not listed are zero" and none
            //       are listed.
            (AtomType::Carbon, AtomType::Carbon) => {
                // NOTE: comments show the values from AIREBO (Stuart, 2000),
                //       three of which are modified to counteract terms introduced from
                //       torsion in undercoordinated systems like graphite.
                match (int_ccoord, int_hcoord) {
                    (1, 1) => Output::with_zero_deriv(0.003_026_697_473_481), // -0.010_960 (!!!)
                    (0, 2) => Output::with_zero_deriv(0.007_860_700_254_745), // -0.000_500 (!!!)
                    (0, 3) => Output::with_zero_deriv(0.016_125_364_564_267), //  0.016_125
                    (2, 1) => Output::with_zero_deriv(0.003_179_530_830_731), //  0.003_180
                    (1, 2) => Output::with_zero_deriv(0.006_326_248_241_119), //  0.006_326
                    // (2, 0) =>                                              // -0.027_603 (!!!)
                    _ => Output::zero(),
                }
            },
            (AtomType::Carbon, AtomType::Hydrogen) => {
                match (int_ccoord, int_hcoord) {
                    (0, 1) => Output::with_zero_deriv(0.209_336_732_825_0380),  //  0.209_337
                    (0, 2) => Output::with_zero_deriv(-0.064_449_615_432_525),  // -0.064_450
                    (0, 3) => Output::with_zero_deriv(-0.303_927_546_346_162),  // -0.303.928
                    (1, 0) => Output::with_zero_deriv(0.01),                    //  0.010_000
                    (2, 0) => Output::with_zero_deriv(-0.122_042_146_278_2555), // -0.122_042
                    (1, 1) => Output::with_zero_deriv(-0.125_123_400_628_7090), // -0.125_123
                    (1, 2) => Output::with_zero_deriv(-0.298_905_245_783),      // -0.298_905
                    (3, 0) => Output::with_zero_deriv(-0.307_584_705_066),      // -0.307_585
                    (2, 1) => Output::with_zero_deriv(-0.300_529_172_406_7579), // -0.300_529
                    _ => Output::zero(),
                }
            },
        }
    }
}

use self::brenner_F::BrennerF;
mod brenner_F {
    use super::*;

    pub type Output = BrennerF;
    pub struct Input {
        pub type_i: AtomType,
        pub type_j: AtomType,
        pub tcoord_ij: f64,
        pub tcoord_ji: f64,
        pub xcoord_ij: f64,
    }

    pub struct BrennerF {
        pub value: f64,
        pub d_tcoord_ij: f64,
        pub d_tcoord_ji: f64,
        pub d_xcoord_ij: f64,
    }

    impl BrennerF {
        fn zero() -> Self {
            BrennerF {
                value: 0.0,
                d_tcoord_ij: 0.0,
                d_tcoord_ji: 0.0,
                d_xcoord_ij: 0.0,
            }
        }
    }

    impl Input {
        pub fn compute(self) -> Output { compute(self) }
    }

    // Tables 4, 6, and 9
    fn compute(input: Input) -> Output {
        let Input { type_i, type_j, tcoord_ij, tcoord_ji, xcoord_ij } = input;


        // NOTE:
        //
        // RSP2 does not need the spline because it does not do anything
        // that requires the reactive parts of the potential.
        //
        // Thus, this function is a complete cop-out.
        panic!(
            "Not yet implemented for Brenner F: {}{} bond, Nij = {}, Nji = {}, Nconj = {}",
            type_i.char(), type_j.char(), tcoord_ij, tcoord_ji, xcoord_ij,
        );
    }

    // check if the value and all derivatives can be assumed to be zero,
    // without needing to compute N^{conj}
    pub fn can_assume_zero(
        (type_i, type_j): (AtomType, AtomType),
        (tcoord_ij, tcoord_ji): (f64, f64),
    ) -> bool {
        let frac_point = V2([tcoord_ij, tcoord_ji]);
        let int_point = frac_point.map(|x| x as i32);
        if frac_point != int_point.map(|x| x as f64) {
            return false;
        }

        match (type_i, type_j) {
            // Table 4, a monstrous-looking terror containing a couple of
            // highly suspicious-looking entries.
            //
            // ...I don't want to deal with this right now.
            (AtomType::Carbon, AtomType::Carbon) => match int_point.0 {
                // graphene
                // The table has no entries for (3, 3, ..) so...
                [3, 3] => true,
                _ => false,
            },

            // Table 6?
            (AtomType::Hydrogen, AtomType::Hydrogen) => {
                false
            },

            // Table 9
            (AtomType::Carbon, AtomType::Hydrogen) |
            (AtomType::Hydrogen, AtomType::Carbon) => {
                false
            },
        }
    }

    // NOTE: Parameters for F_CC.
    //
    //       There's so much room for error, and almost none of these values will
    //       be tested by rsp2, whose use cases do not require the reactive parts
    //       of the potential.
    //

//    const N_COORDINATION: usize = 4;
//    const N_CONJ: usize = 10;
//
//    type ParamArray = nd![f64; N_COORDINATION; N_COORDINATION; N_CONJ];
//    struct FCarbonCarbon {
//        value: ParamArray,   // indices are N_ij^T, N_ji^T, N_{ij}^{conj}
//        d_di: ParamArray,    // Derivative with respect to N_ij^T
//        d_dj: ParamArray,    // Derivative with respect to N_ji^T
//        d_dconj: ParamArray, // Derivative with respect to N_{ij}^{conj}
//    }
//
//    // FIXME: Largely untested outside of values at `[3][3][..]`... which are all zero.
//    //        So yeah. Don't trust this.
//    //
//    // Brenner, Table 4
//    impl FCarbonCarbon {
//        fn get() -> Self {
//            let fill = |xs, fill| for p in &mut xs { *p = fill; };
//
//            // "All values and derivatives not listed are equal to zero."
//            let mut out = FCarbonCarbon {
//                value: Default::default(),
//                d_di: Default::default(),
//                d_dj: Default::default(),
//                d_dconj: Default::default(),
//            };
//
//            // TODO: flatten out above 3
//
//            out.value[1][1][1] = 0.105_000; // Acetylene
//            out.value[1][1][2] = -0.004_177_5; // H2C=C=CH
//            fill(&mut out.value[1][1][3..=9], -0.016_085_6); // C4
//            out.value[2][2][1] = 0.094_449_57; // (CH3)2C=C(CH3)2
//            out.value[2][2][2] = 0.022_000_00; // Benzene
//
//            // !!!!!!!!!!!!
//            // FIXME: Are these correct?
//            //
//            // These are the exact values written in the paper, but it describes them as
//            // "Average from difference F(2, 2, 2) to difference F(2, 2, 9)".
//            //
//            // They do have a constant difference, but if we were really starting from
//            // the value of F[2][2][2], then that difference should be around 0.00314, not 0.00662.
//            // (notice how F[2][2][3] > F[2][2][2])
//            //
//            // NOTE: For reference, LAMMPS does use the values exactly as written in the paper.
//            //       (although it's hard to tell at first since they also fix Brenner's mistake
//            //        of doubling the value of each parameter)
//            // !!!!!!!!!!!!
//            out.value[2][2][3] = 0.039_705_87;
//            out.value[2][2][4] = 0.033_088_22;
//            out.value[2][2][5] = 0.026_470_58;
//            out.value[2][2][6] = 0.019_852_93;
//            out.value[2][2][7] = 0.013_235_29;
//            out.value[2][2][8] = 0.006_617_64;
//            out.value[2][2][9] = 0.0;
//
//            out.value[0][1][1] = 0.043_386_99; // C2H
//
//            out.value[0][1][2] = 0.009_917_2158; // C3
//            out.value[0][2][1] = 0.049_397_6637; // CCH2
//            out.value[0][2][2] = -0.011_942_669; // CCH(CH2)
//            fill(&mut out.value[0][3][1..=9], -0.119_798_935); // H3CC
//
//            out.value[1][2][1] = 0.009_649_5698; // H2CCH
//            out.value[1][2][2] = 0.030; // H2C=C=CH2
//            out.value[1][2][3] = -0.0200; // C6H5
//
//            // "Average from F(1,2,3) to F(1,2,6)".
//            // At least this time, the description checks out.
//            out.value[1][2][4] = -0.023_377_8774;
//            out.value[1][2][5] = -0.026_755_7548;
//
//            fill(&mut out.value[1][2][6..=9], -0.030_133_632); // Graphite vacancy
//            fill(&mut out.value[1][3][2..=9], -0.124_836_752); // H3C–CCH
//            fill(&mut out.value[2][3][1..=9], -0.044_709_383); // Diamond vacancy
//
//            // --------------------------
//            // Derivatives
//
//            out.d_di[2][1][1] = -0.052_500;
//            fill(&mut out.d_di[2][1][5..=9], -0.054_376);
//            out.d_di[2][3][1] = 0.000_00;
//
//            // NOTE: another oddity. These two ranges are written separately
//            //       in the paper even though they could be a single range 2..=9.
//            //       Does one contain an error?
//            fill(&mut out.d_di[2][3][2..=6], 0.062_418);
//            fill(&mut out.d_di[2][3][7..=9], 0.062_418);
//
//            // !!!!!!!!!!!!!!!!!!
//            // FIXME
//            //
//            // This derivative is related to the seemingly problematic values
//            // in F[2][2][3..=8]
//            // !!!!!!!!!!!!!!!!!!
//            fill(&mut out.d_dconj[2][2][4..=8], -0.006_618);
//            out.d_dconj[1][1][2] = -0.060_543;
//            out.d_dconj[1][2][4] = -0.020_044;
//            out.d_dconj[1][2][5] = -0.020_044;
//
//            // symmetrize
//            for upper in 0..N_COORDINATION {
//                for lower in 0..upper {
//                    assert_eq!(out.value[upper][lower], 0.0);
//                    out.value[upper][lower].copy_from_slice(&out.value[lower][upper]);
//                    out.d_dconj[upper][lower].copy_from_slice(&out.d_dconj[lower][upper]);
//                }
//            }
//            for i in 0..N_COORDINATION {
//                for j in 0..N_COORDINATION {
//                    out.d_dj[i][j].copy_from_slice(&out.value[j][i]);
//                }
//            }
//
//            // HACK: The values in Brenner (2002) are actually 2 * F.
//            //
//            // (TODO: Fix the values above)
//            for i in 0..N_COORDINATION {
//                for j in 0..N_COORDINATION {
//                    for k in 0..N_CONJ {
//                        out.d_dj[i][j][k] /= 2.0;
//                    }
//                }
//            }
//        }
//    }
}

use self::brenner_T::BrennerT;
mod brenner_T {
    //! T spline (Brenner, Table 5)

    use super::*;

    pub type Output = BrennerT;
    pub struct Input {
        pub type_i: AtomType,
        pub type_j: AtomType,
        pub tcoord_ij: f64,
        pub tcoord_ji: f64,
        pub xcoord_ij: f64,
    }

    pub struct BrennerT {
        pub value: f64,
        pub d_tcoord_ij: f64,
        pub d_tcoord_ji: f64,
        pub d_xcoord_ij: f64,
    }

    impl Output {
        fn with_zero_deriv(value: f64) -> Self {
            Output {
                value,
                d_tcoord_ij: 0.0,
                d_tcoord_ji: 0.0,
                d_xcoord_ij: 0.0,
            }
        }
        fn zero() -> Self { Self::with_zero_deriv(0.0) }
    }

    impl Input {
        pub fn compute(self) -> Output { compute(self) }
    }

    // free function for smaller indent
    fn compute(input: Input) -> Output {
        let Input { type_i, type_j, tcoord_ij, tcoord_ji, xcoord_ij } = input;

        // NOTE:
        //
        // RSP2 does not need the spline because it does not do anything
        // that requires the reactive parts of the potential.
        //
        // Thus, this function is a complete cop-out.
        let frac_point = V3([tcoord_ij, tcoord_ji, xcoord_ij]);
        let int_point = frac_point.map(|x| {
            let out = x as i32;
            assert_eq!(
                out as f64, x,
                "Fractional coordination not yet implemented for Brenner T splines.",
            );
            out
        });

        macro_rules! not_supported {
            () => {{
                panic!(
                    "Not yet implemented for Brenner T: {}{} bond, Nij = {}, Nji = {}, Nconj = {}",
                    type_i.char(), type_j.char(), tcoord_ij, tcoord_ji, xcoord_ij,
                )
            }};
        }

        match (type_i, type_j) {
            // Brenner, Table 5
            //
            // NOTE: My interpretation of the table is that all integer points of this spline
            //       are saddle points, as "all derivatives not listed are zero" and none
            //       are listed.
            (AtomType::Carbon, AtomType::Carbon) => match int_point.0 {
                [2, 2, 1] => Output::with_zero_deriv(-0.070_280_085), // Ethane
                [2, 2, 9] => Output::with_zero_deriv(-0.008_096_75),  // Graphene/graphite
                _ => Output::zero(),
            },

            // NOTE: My understanding is that these are all zero since I can't find any
            //       mention of them, but I should check with other implementations first.
            (AtomType::Hydrogen, AtomType::Hydrogen) |
            (AtomType::Carbon, AtomType::Hydrogen) |
            (AtomType::Hydrogen, AtomType::Carbon) => {
                not_supported!()
            },
        }
    }

    // check if the value and all derivatives can be assumed to be zero,
    // without needing to compute N^{conj}
    pub fn can_assume_zero(
        (type_i, type_j): (AtomType, AtomType),
        (tcoord_ij, tcoord_ji): (f64, f64),
    ) -> bool {
        let frac_point = V2([tcoord_ij, tcoord_ji]);
        let int_point = frac_point.map(|x| x as i32);
        if frac_point != int_point.map(|x| x as f64) {
            return false;
        }

        match (type_i, type_j) {
            // Brenner, Table 5
            //
            // NOTE: My interpretation of the table is that all integer points of this spline
            //       are saddle points, as "all derivatives not listed are zero" and none
            //       are listed.
            (AtomType::Carbon, AtomType::Carbon) => match int_point.0 {
                [2, 2] => false,
                _ => true,
            },

            // NOTE: My understanding is that these are all zero since I can't find any
            //       mention of them, but I should check with other implementations first.
            (AtomType::Hydrogen, AtomType::Hydrogen) |
            (AtomType::Carbon, AtomType::Hydrogen) |
            (AtomType::Hydrogen, AtomType::Carbon) => false,
        }
    }
}

use self::brenner_G::BrennerG;
mod brenner_G {
    use super::*;

    pub type Output = BrennerG;
    pub struct Input<'a> {
        pub params: &'a Params,
        pub type_i: AtomType,
        pub tcoord_ij: f64,
        pub cos_ijk: f64,
    }

    pub struct BrennerG {
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
                let (alpha, alpha_d_tcoord_ij) = switch((3.2, 3.7), tcoord_ij);

                // easy way out for now
                //
                // condition on alpha_d_tcoord is necessary for cases that are just barely
                // in the switching regime that may happen to have zero alpha
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
    /// These are.... quite different from the function described by Brenner!
    ///
    /// (TODO: look further into this; is this what AIREBO does?)
    ///
    /// They also appear to be rounded to dangerously low precision, which
    /// might introduce discontinuities at the switch points (most troublingly so at 120°).
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
            let BrennerG { value, d_cos_ijk, d_tcoord_ij } = Input {
                params,
                type_i: AtomType::Carbon,
                cos_ijk: f64::cos(120.0 * PI / 180.0) + 1e-12,
                tcoord_ij: 3.0,
            }.compute();
            // Brenner Table 3
            assert_close!(rel=tol, value, 0.05280);
            assert_close!(rel=tol, d_cos_ijk, 0.17000);
            assert_eq!(d_tcoord_ij, 0.0);

            let BrennerG { value, d_cos_ijk, d_tcoord_ij } = Input {
                params,
                type_i: AtomType::Carbon,
                cos_ijk: f64::cos(120.0 * PI / 180.0) - 1e-12,
                tcoord_ij: 3.0,
            }.compute();
            assert_close!(rel=tol, value, 0.05280);
            assert_close!(rel=tol, d_cos_ijk, 0.17000);
            assert_eq!(d_tcoord_ij, 0.0);

            // diamond
            let BrennerG { value, d_cos_ijk, d_tcoord_ij } = Input {
                params,
                type_i: AtomType::Carbon,
                cos_ijk: -1.0/3.0 + 1e-12,
                tcoord_ij: 4.0,
            }.compute();
            assert_close!(rel=tol, value, 0.09733);
            assert_close!(rel=tol, d_cos_ijk, 0.40000);
            assert_eq!(d_tcoord_ij, 0.0);

            let BrennerG { value, d_cos_ijk, d_tcoord_ij } = Input {
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
                        let BrennerG { value: _, d_cos_ijk, d_tcoord_ij } = input.compute();
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
            println!("set");
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
            let num_diff = numerical::slope(1e-1, Some(CentralDifference), x, |x| f(x).0);
            assert_close!{rel=1e-12, diff, num_diff};
        }
    }

    #[test]
    fn d_switch() {
        unimplemented!()
    }

    // Numerical tests for brenner_G are in that module
}

//-----------------------------------------------------------------------------

/// A tricubic spline with constraints defined on an integer grid.
pub mod tricubic_grid {
    use super::*;

    // Until we get const generics, it's too much trouble to be generic over lengths,
    // so we'll just use one fixed dimension.
    pub const MAX_I: usize = 4;
    pub const MAX_J: usize = 4;
    pub const MAX_K: usize = 9;
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
    #[derive(Clone)]
    pub struct _Input<G> {
        pub value: G,
        pub di: G,
        pub dj: G,
        pub dk: G,
    }

    //------------------------------------

    pub struct TricubicGrid {
        fit_params: Box<Input>,
        polys: Box<Grid<(TriPoly3, V3<TriPoly3>)>>,
    }

    impl TricubicGrid {
        pub fn evaluate(&self, point: V3) -> (f64, V3) { self._evaluate(point).1 }

        fn _evaluate(&self, point: V3) -> (EvalKind, (f64, V3)) {
            // We assume the splines are flat with constant value outside the fitted regions.
            let point = clip_point(point);

            let indices = point.map(|x| x as usize);

            if point == indices.map(|x| x as f64) {
                // Fast path (integer point)

                let V3([i, j, k]) = indices;
                let value = self.fit_params.value[i][j][k];
                let di = self.fit_params.di[i][j][k];
                let dj = self.fit_params.dj[i][j][k];
                let dk = self.fit_params.dk[i][j][k];
                (EvalKind::Fast, (value, V3([di, dj, dk])))
            } else {
                // Slow path.
                //
                // It is only ever possible to take this path when a reaction is occurring.
                warn!("untested codepath: 70dfe923-e1af-45f1-8dc6-eb50ae4ce1cc");

                // Indices must now be constrained to the smaller range that is valid
                // for the polynomials. (i.e. the max index is no longer valid)
                //
                // (Yes, we must account for this even though we clipped the point; if the
                //  point is only out of bounds along one axis, the others may still be
                //  fractional and thus the slow path could still be taken)
                let V3([mut i, mut j, mut k]) = indices;
                i = i.min(MAX_I - 1);
                j = j.min(MAX_J - 1);
                k = k.min(MAX_K - 1);

                let frac_point = point - V3([i, j, k]).map(|x| x as f64);
                let (value_poly, diff_polys) = &self.polys[i][j][k];
                let value = value_poly.evaluate(point);
                let diff = V3::from_fn(|axis| diff_polys[axis].evaluate(frac_point));
                (EvalKind::Slow, (value, diff))
            }
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
        pub fn solve(&self) -> FailResult<TricubicGrid> {
            use ::rsp2_array_utils::{try_arr_from_fn, arr_from_fn, map_arr};
            self.verify_clipping_is_valid()?;

            let polys = Box::new({
                try_arr_from_fn(|i| {
                    try_arr_from_fn(|j| {
                        try_arr_from_fn(|k| -> FailResult<_> {
                            // Gather the 8 points describing this region.
                            // (ni,nj,nk = 0 or 1)
                            let poly_input: TriPoly3Input = self.map_grids(|grid| {
                                arr_from_fn(|ni| {
                                    arr_from_fn(|nj| {
                                        arr_from_fn(|nk| {
                                            grid[i + ni][j + nj][k + nk]
                                        })
                                    })
                                })
                            });
                            let value_poly = poly_input.solve()?;
                            let diff_polys = V3::from_fn(|axis| value_poly.axis_derivative(axis));
                            Ok((value_poly, diff_polys))
                        })
                    })
                })?
            });

            let fit_params = Box::new(self.clone());
            Ok(TricubicGrid { fit_params, polys })
        }
    }

    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    enum EvalKind { Fast, Slow }

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
        fn ensure_clipping_is_valid(mut self) -> Self {
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

    pub fn clip_point(point: V3) -> V3 {
        let mut point = point.map(|x| f64::max(x, 0.0));
        point[0] = point[0].min(MAX_I as f64);
        point[1] = point[1].min(MAX_J as f64);
        point[2] = point[2].min(MAX_K as f64);
        point
    }

    //------------------------------------

    /// A third-order polynomial in three variables.
    #[derive(Clone)]
    pub struct TriPoly3 {
        /// coeffs along each index are listed in order of increasing power
        coeff: Box<nd![f64; 4; 4; 4]>,
    }

    pub type TriPoly3Input = _Input<nd![f64; 2; 2; 2]>;
    impl TriPoly3Input {
        fn solve(&self) -> FailResult<TriPoly3> {
            let b_vec: nd![f64; 8; 2; 2; 2] = [
                self.value,
                self.di, self.dj, self.dk,
                Default::default(), // constraints on didj
                Default::default(), // constraints on didk
                Default::default(), // constraints on djdk
                Default::default(), // constraints on didjdk
            ];
            let b_vec: &[[f64; 1]] = b_vec.flat().flat().flat().nest();
            let b_vec: ::rsp2_linalg::CMatrix = b_vec.into();

            let coeff = ::rsp2_linalg::lapacke_linear_solve(ZERO_ONE_CMATRIX.clone(), b_vec)?;
            Ok(TriPoly3 {
                coeff: Box::new(coeff.c_order_data().nest().nest().to_array()),
            })
        }
    }

    impl TriPoly3 {
        pub fn zero() -> Self {
            TriPoly3 { coeff: Box::new(<nd![f64; 4; 4; 4]>::default()) }
        }

        pub fn evaluate(&self, point: V3) -> f64 {
            let V3([i, j, k]) = point;

            let powers = |x| [1.0, x, x*x, x*x*x];
            let i_pows = powers(i);
            let j_pows = powers(j);
            let k_pows = powers(k);

            let mut acc = 0.0;
            for (coeff_plane, &i_pow) in zip_eq!(&self.coeff[..], &i_pows) {
                for (coeff_row, &j_pow) in zip_eq!(coeff_plane, &j_pows) {
                    let row_sum = zip_eq!(coeff_row, &k_pows).map(|(&a, &b)| a * b).sum::<f64>();
                    acc += i_pow * j_pow * row_sum;
                }
            }
            acc
        }

        #[inline(always)]
        fn coeff(&self, (i, j, k): (usize, usize, usize)) -> f64 { self.coeff[i][j][k] }
        #[inline(always)]
        fn coeff_mut(&mut self, (i, j, k): (usize, usize, usize)) -> &mut f64 { &mut self.coeff[i][j][k] }

        pub fn axis_derivative(&self, axis: usize) -> Self {
            let mut out = Self::zero();
            for scan_idx_1 in 0..4 {
                for scan_idx_2 in 0..4 {
                    let get_pos = |i| match axis {
                        0 => (i, scan_idx_1, scan_idx_2),
                        1 => (scan_idx_1, i, scan_idx_2),
                        2 => (scan_idx_1, scan_idx_2, i),
                        _ => panic!("invalid axis: {}", axis),
                    };
                    for i in 1..4 {
                        *out.coeff_mut(get_pos(i-1)) = i as f64 * self.coeff(get_pos(i));
                    }
                }
            }
            out
        }
    }


    lazy_static! {
        // The matrix representing the system of equations that must be solved for
        // a piece of a tricubic spline with boundaries at zero and one.
        //
        // Indices are, from slowest to fastest:
        // - row (8x2x2x2 = 64), broken into two levels:
        //   - constraint kind (8: [value, di, dj, dk, didj, didk, djdk, didjdk])
        //   - constraint location (2x2x2: [i=0, i=1] x [j=0, j=1] x [k=0, k=1])
        // - col (4x4x4 = 64), where each axis is the power of one of the variables
        //   for the coefficient belonging to this column
        static ref ZERO_ONE_MATRIX: nd![f64; 8; 2; 2; 2; 4; 4; 4] = compute_zero_one_matrix();
        static ref ZERO_ONE_CMATRIX: ::rsp2_linalg::CMatrix = {
            ZERO_ONE_MATRIX
                .flat().flat().flat().flat()
                .flat().flat().nest::<[_; 64]>()
                .into()
        };
    }

    fn compute_zero_one_matrix() -> nd![f64; 8; 2; 2; 2; 4; 4; 4] {
        use ::rsp2_array_utils::{arr_from_fn, map_arr};

        // we build a system of equations from our constraints
        //
        // we end up with an equation of the form  M a = b,
        // where M is a square matrix whose elements are products of the end-point coords
        // raised to various powers.

        #[derive(Debug, Copy, Clone)]
        struct Monomial {
            coeff: f64,
            powers: [u32; 3],
        }
        impl Monomial {
            fn axis_derivative(mut self, axis: usize) -> Self {
                self.coeff *= self.powers[axis] as f64;
                if self.powers[axis] > 0 {
                    self.powers[axis] -= 1;
                }
                self
            }

            fn evaluate(&self, point: V3) -> f64 {
                let mut out = self.coeff;
                for i in 0..3 {
                    out *= point[i].powi(self.powers[i] as i32);
                }
                out
            }
        }

        // Polynomials here are represented as values to be multiplied against each coefficient.
        //
        // e.g. [1, x, x^2, x^3, y, y*x, y*x^2, y*x^3, ... ]
        let derive = |poly: &[Monomial], axis| -> Vec<Monomial> {
            poly.iter().map(|m| m.axis_derivative(axis)).collect()
        };

        let value_poly: nd![Monomial; 4; 4; 4] = {
            arr_from_fn(|i| {
                arr_from_fn(|j| {
                    arr_from_fn(|k| {
                        Monomial { coeff: 1.0, powers: [i as u32, j as u32, k as u32] }
                    })
                })
            })
        };
        let value_poly = value_poly.flat().flat().to_vec();
        let di_poly = derive(&value_poly, 0);
        let dj_poly = derive(&value_poly, 1);
        let dk_poly = derive(&value_poly, 2);
        let didj_poly = derive(&di_poly, 1);
        let didk_poly = derive(&di_poly, 2);
        let djdk_poly = derive(&dj_poly, 2);
        let didjdk_poly = derive(&didj_poly, 2);

        map_arr([
            value_poly, di_poly, dj_poly, dk_poly,
            didj_poly, didk_poly, djdk_poly, didjdk_poly,
        ], |poly| {
            // coords of each corner (0 or 1)
            arr_from_fn(|i| {
                arr_from_fn(|j| {
                    arr_from_fn(|k| {
                        // powers
                        let poly: &nd![_; 4; 4; 4] = poly.nest().nest().as_array();
                        arr_from_fn(|ei| {
                            arr_from_fn(|ej| {
                                arr_from_fn(|ek| {
                                    poly[ei][ej][ek].evaluate(V3([i, j, k]).map(|x| x as f64))
                                })
                            })
                        })
                    })
                })
            })
        })
    }

    //------------------------------------
    // tests

    #[test]
    fn test_spline_fast_path() -> FailResult<()> {
        let fit_params = Input {
            value: ::rand::random(),
            di: ::rand::random(),
            dj: ::rand::random(),
            dk: ::rand::random(),
        }.ensure_clipping_is_valid();

        let spline = fit_params.solve()?;

        // every valid integer point should be evaluated quickly
        for i in 0..=MAX_I {
            for j in 0..=MAX_J {
                for k in 0..=MAX_K {
                    let (kind, output) = spline._evaluate(V3([i, j, k]).map(|x| x as f64));
                    let (value, grad) = output;
                    assert_eq!(kind, EvalKind::Fast);
                    assert_eq!(value, fit_params.value[i][j][k]);
                    assert_eq!(grad[0], fit_params.di[i][j][k]);
                    assert_eq!(grad[1], fit_params.dj[i][j][k]);
                    assert_eq!(grad[2], fit_params.dk[i][j][k]);
                }
            }
        }

        // points outside the boundaries should also be evaluated quickly if the
        // remaining coords are integers
        let base_point = V3([2.0, 2.0, 2.0]);
        let base_index = V3([2, 2, 2]);
        for axis in 0..3 {
            for do_right_side in vec![false, true] {
                let mut input_point = base_point;
                let mut expected_index = base_index;
                match do_right_side {
                    false => {
                        input_point[axis] = -1.2;
                        expected_index[axis] = 0;
                    },
                    true => {
                        input_point[axis] = [MAX_I, MAX_J, MAX_K][axis] as f64 + 3.2;
                        expected_index[axis] = [MAX_I, MAX_J, MAX_K][axis];
                    }
                }

                let (kind, output) = spline._evaluate(input_point);
                let (value, grad) = output;

                let V3([i, j, k]) = expected_index;
                assert_eq!(kind, EvalKind::Fast);
                assert_eq!(value, fit_params.value[i][j][k]);
                assert_eq!(grad[0], fit_params.di[i][j][k]);
                assert_eq!(grad[1], fit_params.dj[i][j][k]);
                assert_eq!(grad[2], fit_params.dk[i][j][k]);
            }
        }
        Ok(())
    }

    #[test]
    fn test_spline_fit_accuracy() -> FailResult<()> {
        for _ in 0..3 {
            let fit_params = Input {
                value: ::rand::random(),
                di: ::rand::random(),
                dj: ::rand::random(),
                dk: ::rand::random(),
            }.ensure_clipping_is_valid();;

            let spline = fit_params.solve()?;

            // index of a polynomial
            for i in 0..MAX_I {
                for j in 0..MAX_J {
                    for k in 0..MAX_K {
                        // index of a corner of the polynomial
                        for ni in 0..2 {
                            for nj in 0..2 {
                                for nk in 0..2 {
                                    // index of the point of evaluation
                                    let V3([pi, pj, pk]) = V3([i + ni, j + nj, k + nk]);
                                    let frac_point = V3([ni, nj, nk]).map(|x| x as f64);

                                    let (value_poly, diff_polys) = &spline.polys[i][j][k];
                                    let V3([di_poly, dj_poly, dk_poly]) = diff_polys;
                                    assert_close!(rel=1e-8, abs=1e-8, value_poly.evaluate(frac_point), fit_params.value[pi][pj][pk]);
                                    assert_close!(rel=1e-8, abs=1e-8, di_poly.evaluate(frac_point), fit_params.di[pi][pj][pk]);
                                    assert_close!(rel=1e-8, abs=1e-8, dj_poly.evaluate(frac_point), fit_params.dj[pi][pj][pk]);
                                    assert_close!(rel=1e-8, abs=1e-8, dk_poly.evaluate(frac_point), fit_params.dk[pi][pj][pk]);
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn test_poly3_evaluate() {
        for _ in 0..1 {
            let point = V3::from_fn(|_| uniform(-1.0, 1.0));
            let poly = TriPoly3 {
                coeff: Box::new({
                    ::std::iter::repeat_with(|| uniform(-5.0, 5.0)).take(64).collect::<Vec<_>>()
                        .nest().nest().to_array()
                }),
            };

            let expected = {
                // brute force
                let mut acc = 0.0;
                for i in 0..4 {
                    for j in 0..4 {
                        for k in 0..4 {
                            acc += {
                                poly.coeff[i][j][k]
                                    * point[0].powi(i as i32)
                                    * point[1].powi(j as i32)
                                    * point[2].powi(k as i32)
                            };
                        }
                    }
                }
                acc
            };
            assert_close!(poly.evaluate(point), expected);
        }
    }

    #[test]
    fn test_poly3_numerical_deriv() -> () {
        for _ in 0..20 {
            let value_poly = TriPoly3 {
                coeff: Box::new(::rand::random()),
            };
            let grad_polys = V3::from_fn(|axis| value_poly.axis_derivative(axis));

            let point = V3::from_fn(|_| uniform(-6.0, 6.0));

            let computed_grad = grad_polys.map(|poly| poly.evaluate(point));
            let numerical_grad = num_grad_v3(1e-6, point, |p| value_poly.evaluate(p));

            // This can fail pretty bad if the polynomial produces lots of cancellation
            // in one of the derivatives.  We must accept either abs or rel tolerance.
            assert_close!(rel=1e-5, abs=1e-5, computed_grad.0, numerical_grad.0)
        }
    }
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
