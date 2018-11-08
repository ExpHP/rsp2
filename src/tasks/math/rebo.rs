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
#[cfg(test)]
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
            Params { by_type }
        }

        /// Parameters consistent with LAMMPS' `Airebo.CH`.
        ///
        /// The HH parameters have changed a fair bit compared to the parameters
        /// originally published in Brenner's paper.
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
            Params { by_type }
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

            d_positions[site_i] += d_deltas[bond_ij];
            d_positions[site_j] -= d_deltas[bond_ij];
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
    // Eq  1:  V = sum_{i < j} V^R(r_ij) + b_{ij} V^A(r_ij)
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
    // We will redefine V^R and V^A to pull out the common factor of f(r),
    // renaming them U^R and U^A:
    //
    // Eq  1':  V = sum_{i < j} f_ij [ U^R(r_ij) + b_ij U^A(r_ij) ]
    // Eq  5':  U^R(r) = (1 + Q/r) A e^{-alpha r}
    // Eq  6':  U^A(r) = sum_{n in 1..=3} B_n e^{-beta_n r}
    //
    // We also redefine the sums in the potential to be over all i,j pairs, not just i < j.
    //
    // Eq 1'':     V = sum_{i != j} f_ij U_ij
    // Eq 2'':  U_ij = 0.5 * U^R_ij + b_ij * U^A_ij
    // Eq 3'':  b_ij = 0.5 * b_ij^{sigma-pi} + boole(i < j) * b_ij^{pi}

    // On large systems, our performance is expected to be bounded by cache misses.
    // For this reason, we should aim to make as few passes over the data as necessary,
    // leaving vectorization as only a secondary concern.
    struct FirstPassSiteData {
        // Brenner's N_i
        tcoord: f64,
        // Brenner's f_ij
        bond_weight: SiteBondVec<f64>,
        bond_weight_d_delta: SiteBondVec<V3>,
        // Brenner's V^R_ij, without the embedded f_ij factor
        bond_UR: SiteBondVec<f64>,
        bond_UR_d_delta: SiteBondVec<V3>,
        // Brenner's V^A_ij, without the embedded f_ij factor
        bond_UA: SiteBondVec<f64>,
        bond_UA_d_delta: SiteBondVec<V3>,
    }

    let site_data = IndexVec::<SiteI, _>::from_raw({
        interactions.site_range().into_par_iter().map(SiteI::new).map(|site_i| {
            let __site = interactions.site(site_i);
            let type_i = __site.atom_type;

            let mut tcoord = 0.0;
            let mut bond_UR = SiteBondVec::new();
            let mut bond_UR_d_delta = SiteBondVec::new();
            let mut bond_UA = SiteBondVec::new();
            let mut bond_UA_d_delta = SiteBondVec::new();
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

                // UA = (1 + Q/r) A e^{-alpha r}
                // write as a product of two subexpressions
                let UA;
                let UA_d_delta;
                {
                    let sub1 = 1.0 + params_ij.Q / length;
                    let sub1_d_length = - params_ij.Q / (length * length);

                    let sub2 = params_ij.A * f64::exp(-params_ij.alpha * length);
                    let sub2_d_length = -params_ij.alpha * sub2;

                    let UA_d_length = sub1_d_length * sub2 + sub1 * sub2_d_length;

                    UA = sub1 * sub2;
                    UA_d_delta = UA_d_length * length_d_delta;
                }
                bond_UA.push(UA);
                bond_UA_d_delta.push(UA_d_delta);

                // UR = sum_{n in 1..=3} B_n e^{-beta_n r}
                let mut UR = 0.0;
                let mut UR_d_length = 0.0;
                for (&B, &beta) in zip_eq!(&params_ij.B, &params_ij.beta) {
                    let term = B * f64::exp(-beta * length);
                    let term_d_length = -beta * term;
                    UR += term;
                    UR_d_length += term_d_length;
                }
                let UR_d_delta = UR_d_length * length_d_delta;
                bond_UR.push(UR);
                bond_UR_d_delta.push(UR_d_delta);
            } // for _ in interactions.bonds(site)

            FirstPassSiteData {
                tcoord,
                bond_weight, bond_weight_d_delta,
                bond_UR, bond_UR_d_delta,
                bond_UA, bond_UA_d_delta,
            }
        }).collect()
    });

    let out = interactions.site_range().into_par_iter().map(SiteI::new).map(|site_i| {
        let __site = interactions.site(site_i);
        let type_i = __site.atom_type;
        let FirstPassSiteData {
            tcoord: tcoord_i,
            ref bond_weight, ref bond_weight_d_delta,
            ref bond_UR, ref bond_UR_d_delta,
            ref bond_UA, ref bond_UA_d_delta,
        } = site_data[site_i];

        // Eq 4'':  V_ij = f_ij U_ij
        // Eq 2'':  U_ij = 0.5 * U^R_ij + b_ij * U^A_ij
        //
        // site_V is a sum of V_ij over all j
        //
        // NOTE: Due to the simplification inside the bond loop, we can assume that
        //       we will only find derivatives with respect to this atom's bonds.
        let mut site_V = 0.0;
        let mut site_V_d_delta = sbvec_filled(V3::zero(), bond_weight.len());

        let iter = zip_eq![bond_UR, bond_UR_d_delta].enumerate();
        for (index_ij, (UR_ij, UR_ij_d_delta_ij)) in iter {
            // V_ij = f_ij * U_ij
            let VR_ij = UR_ij * bond_weight[index_ij];
            let VR_ij_d_UR_ij = bond_weight[index_ij];

            let VR_ij_d_delta_ij = VR_ij_d_UR_ij * UR_ij_d_delta_ij;

            site_V += 0.5 * VR_ij;
            site_V_d_delta[index_ij] += 0.5 * VR_ij_d_delta_ij;
        }

        // Eq 3'':  b_ij = 0.5 * b_ij^{sigma-pi} + boole(i < j) * b_ij^{pi}
        for (index_ij, __bond) in interactions.bonds(site_i).enumerate() {
            let site_j = __bond.target;
            let tcoord_j = site_data[site_j].tcoord;
            let type_j = interactions.site(site_j).atom_type;

            // sigma-pi terms are present for all bonds, regardless of direction
            let out = bond_order_sigma_pi_sitesum::Input {
                interactions,
                site: site_i,
                bond_weights: bond_weight,
            }.compute();
            let BondOrderSigmaPiSitesum {
                value: bsp_ij,
                d_deltas: bsp_ij_d_deltas,
                d_weights: bsp_ij_d_weights,
            } = out;

            if __bond.is_canonical {
                // canonical bonds also have DH and RC terms computed

                // FIXME simplification: Assume the pi bondorder is zero.
                //
                //       This means that there's only a sigma-pi term for each bond,
                //       whose derivatives are entirely of data local to the originating site
                //       (i.e. we can fit them in a SiteBondVec)
                if !brenner_T::can_assume_zero((type_i, type_j), (tcoord_i, tcoord_j)) {
                    panic!("brenner T spline may be nonzero; this is not yet implemented");
                }
                if !brenner_F::can_assume_zero((type_i, type_j), (tcoord_i, tcoord_j)) {
                    panic!("brenner F spline may be nonzero; this is not yet implemented");
                }
            }

            // Eq 3'':  b_ij = 0.5 * b_ij^{sigma-pi} + boole(i < j) * b_ij^{pi}

            // Again, by our simplification, the complete bondorder b_ij
            // simply comes from the sigma-pi bondorder ij.
            let b_ij = 0.5 * bsp_ij;
            let b_ij_d_weights = sbvec_scaled(0.5, bsp_ij_d_weights);
            let b_ij_d_deltas = sbvec_scaled(0.5, bsp_ij_d_deltas);

            // Eq 2'':  U_ij = 0.5 * U^R_ij + b_ij * U^A_ij
            //
            // The U^R part was already taken care of.
            let UA_ij = bond_UA[index_ij];
            let UA_ij_d_delta_ij = bond_UA_d_delta[index_ij];

            // V_ij = f_ij * U_ij
            let VA_ij = UA_ij * bond_weight[index_ij];
            let VA_ij_d_UA_ij = bond_weight[index_ij];

            let VA_ij_d_delta_ij = VA_ij_d_UA_ij * UA_ij_d_delta_ij;

            site_V += VA_ij * b_ij;
            site_V_d_delta[index_ij] += VA_ij_d_delta_ij * b_ij;
            for index_ik in 0..bond_weight.len() {
                let b_ij_d_weight_ik = b_ij_d_weights[index_ik];
                let b_ij_d_delta_ik = b_ij_d_deltas[index_ik];
                let weight_ik_d_delta_ik = bond_weight_d_delta[index_ik];

                site_V_d_delta[index_ik] += VA_ij * b_ij_d_weight_ik * weight_ik_d_delta_ik;
                site_V_d_delta[index_ik] += VA_ij * b_ij_d_delta_ik;
            }
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

use self::bond_order_sigma_pi_sitesum::BondOrderSigmaPiSitesum;
mod bond_order_sigma_pi_sitesum {
    //! Represents the sum of `b_{ij}^{sigma-pi}` over all `j` for a given `i`.
    //!
    //! This quantity is useful to consider in its own right because it encapsulates
    //! the need for the P spline values, and it only has derivatives with respect
    //! to the bond vectors of site `i`; these properties give it a fairly simple
    //! signature to make up for its absurdly long name.

    use super::*;

    pub type Output = BondOrderSigmaPiSitesum;
    pub struct Input<'a> {
        pub interactions: &'a Interactions,
        pub site: SiteI,
        pub bond_weights: &'a [f64],
    }
    pub struct BondOrderSigmaPiSitesum {
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
        //                       + P_{ij}(N_i^C, N_i^H)
        //        )
        let Input {
            interactions, bond_weights,
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

        // P only needs to be evaluated up to two times per atom
        // (depending on the types of its neighbors)
        let P_by_type = enum_map!{
            target_type => {
                let type_j = target_type;
                match type_present[target_type] {
                    true => Some(brenner_P::Input { type_i, type_j, ccoord_i, hcoord_i }.compute()),
                    false => None,
                }
            }
        };

        // Handle all terms
        let mut value = 0.0;
        let mut d_deltas: SiteBondVec<V3> = sbvec_filled(V3::zero(), bond_weights.len());
        let mut d_weights: SiteBondVec<f64> = sbvec_filled(0.0, bond_weights.len());

        for (index_ij, __bond) in interactions.bonds(site_i).enumerate() {
            let type_j = bond_target_types[index_ij];
            let delta_ij = __bond.cart_vector;

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

            // We're finally ready to compute this term.
            let out = bondorder_sigma_pi::Input {
                type_i, type_j, ccoord_i, hcoord_i,
                coses_ijk: &coses_ijk,
                types_k: &bond_target_types,
                weights_ik: bond_weights,
                skip_index: index_ij, // used to exclude the ijj angle
                P_ij: P_by_type[type_j].expect(""),
            }.compute();
            let BondOrderSigmaPi {
                value: term,
                d_ccoord_i: term_d_ccoord_i,
                d_hcoord_i: term_d_hcoord_i,
                d_coses_ijk: term_d_coses_ijk,
                d_weights_ik: term_d_weights_ik,
            } = out;

            // Add the term.
            value += term;

            // Simplify all derivatives of the term to be with respect to deltas and weights.

            // Even though this term describes a single bond, its dependence on the coordination
            // numbers produce derivatives with respect to all of the bond weights.
            if (term_d_ccoord_i, term_d_hcoord_i) != (0.0, 0.0) {
                for (index, ty) in bond_target_types.iter().enumerate() {
                    match ty {
                        // ccoord_i_d_weight = 1.0,  hcoord_i_d_weight = 0.0
                        AtomType::Carbon => d_weights[index] += term_d_ccoord_i,
                        // ccoord_i_d_weight = 0.0,  hcoord_i_d_weight = 1.0
                        AtomType::Hydrogen => d_weights[index] += term_d_hcoord_i,
                    }
                }
            }

            // Some derivatives also come from the ik bonds.
            let iter = zip_eq!(term_d_weights_ik, term_d_coses_ijk).enumerate();
            for (index_ik, (term_d_weight_ik, term_d_cos_ijk)) in iter {
                // Mind the gap
                if index_ij == index_ik {
                    continue;
                }
                let cos_ijk_d_delta_ij = coses_ijk_d_delta_ij[index_ik];
                let cos_ijk_d_delta_ik = coses_ijk_d_delta_ik[index_ik];

                d_weights[index_ik] += term_d_weight_ik;
                d_deltas[index_ij] += term_d_cos_ijk * cos_ijk_d_delta_ij;
                d_deltas[index_ik] += term_d_cos_ijk * cos_ijk_d_delta_ik;
            }
        }
        Output { value, d_weights, d_deltas }
    }
}

use self::bondorder_sigma_pi::BondOrderSigmaPi;
mod bondorder_sigma_pi {
    use super::*;

    pub type Output = BondOrderSigmaPi;
    pub struct Input<'a> {
        // bond from atom i to another atom j
        pub type_i: AtomType,
        pub type_j: AtomType,
        pub ccoord_i: f64,
        pub hcoord_i: f64,
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
        pub d_ccoord_i: f64,
        pub d_hcoord_i: f64,
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
            type_i, type_j,
            ccoord_i, hcoord_i, P_ij,
            types_k, weights_ik, coses_ijk,
            skip_index,
        } = input;
        let tcoord = ccoord_i + hcoord_i;
        let tcoord_d_ccoord_i = 1.0;
        let tcoord_d_hcoord_i = 1.0;

        // properties of the stuff in the square root
        let inner_value;
        let inner_d_ccoord_i;
        let inner_d_hcoord_i;
        let inner_d_coses_ijk;
        let inner_d_weights_ik;
        {
            let mut tmp_value = 0.0;
            let mut tmp_d_ccoord_i = 0.0;
            let mut tmp_d_hcoord_i = 0.0;
            let mut tmp_d_coses_ijk = SiteBondVec::new();
            let mut tmp_d_weights_ik = SiteBondVec::new();

            // 1 + P_{ij}(N_i^C, N_i^H)
            //   + sum_{k /= i, j} e^{\lambda_{ijk}} f^c(r_{ik}) G(cos(t_{ijk})
            tmp_value += 1.0;

            let BrennerP {
                value: P,
                d_ccoord_i: P_d_ccoord_i,
                d_hcoord_i: P_d_hcoord_i,
            } = P_ij;
            tmp_value += P;
            tmp_d_ccoord_i += P_d_ccoord_i;
            tmp_d_hcoord_i += P_d_hcoord_i;

            let iter = zip_eq!(weights_ik, coses_ijk, types_k).enumerate();
            for (index_ik, (&weight_ik, &cos_ijk, &type_k)) in iter {
                if index_ik == skip_index {
                    tmp_d_coses_ijk.push(NAN);
                    tmp_d_weights_ik.push(NAN);
                } else {
                    let exp_lambda = brenner_exp_lambda(type_i, (type_j, type_k));

                    let BrennerG {
                        value: G,
                        d_cos: G_d_cos,
                        d_tcoord: G_d_tcoord,
                    } = brenner_G::Input { type_i, cos: cos_ijk, tcoord }.compute();

                    let G_d_ccoord = G_d_tcoord * tcoord_d_ccoord_i;
                    let G_d_hcoord = G_d_tcoord * tcoord_d_hcoord_i;

                    tmp_value += exp_lambda * weight_ik * G;
                    tmp_d_ccoord_i += exp_lambda * weight_ik * G_d_ccoord;
                    tmp_d_hcoord_i += exp_lambda * weight_ik * G_d_hcoord;
                    tmp_d_coses_ijk.push(exp_lambda * weight_ik * G_d_cos);
                    tmp_d_weights_ik.push(exp_lambda * 1.0 * G);
                }
            }

            inner_value = tmp_value;
            inner_d_ccoord_i = tmp_d_ccoord_i;
            inner_d_hcoord_i = tmp_d_hcoord_i;
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
            d_ccoord_i: prefactor * inner_d_ccoord_i,
            d_hcoord_i: prefactor * inner_d_hcoord_i,
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
//        pub tcoord_i: f64,
//        pub tcoord_j: f64,
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
//        pub d_tcoord_i: f64,
//        pub d_tcoord_j: f64,
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
//        let Input { type_i, type_j, tcoord_i, tcoord_j, xcoord_ij, dihedrals_ijkl } = input;
//        let BrennerT {
//            value: T,
//            d_tcoord_i: T_d_tcoord_i,
//            d_tcoord_j: T_d_tcoord_j,
//            d_xcoord_ij: T_d_xcoord_ij,
//        } = brenner_T::Input { type_i, type_j, tcoord_i, tcoord_j, xcoord_ij }.compute();
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
//            d_tcoord_i: T_d_tcoord_i * sum,
//            d_tcoord_j: T_d_tcoord_j * sum,
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
        pub ccoord_i: f64,
        pub hcoord_i: f64,
    }

    #[derive(Debug, Copy, Clone)]
    pub struct BrennerP {
        pub value: f64,
        pub d_ccoord_i: f64,
        pub d_hcoord_i: f64,
    }

    impl BrennerP {
        fn with_zero_deriv(value: f64) -> Self {
            BrennerP { value, d_ccoord_i: 0.0, d_hcoord_i: 0.0 }
        }
        fn zero() -> Self { Self::with_zero_deriv(0.0) }
    }

    impl Input {
        pub fn compute(self) -> Output {
            compute(self)
        }
    }

    fn compute(input: Input) -> Output {
        let Input { type_i, type_j, ccoord_i, hcoord_i } = input;

        // NOTE:
        //
        // RSP2 does not need the spline because it does not do anything
        // that requires the reactive parts of the potential.
        //
        // Thus, this function is a cop-out.
        let int_ccoord = ccoord_i as i64;
        let int_hcoord = hcoord_i as i64;
        if int_ccoord as f64 != ccoord_i || int_hcoord as f64 != hcoord_i {
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
                match (int_ccoord, int_hcoord) {
                    (1, 1) => Output::with_zero_deriv(0.003_026_697_473_481),
                    (0, 2) => Output::with_zero_deriv(0.007_860_700_254_745),
                    (0, 3) => Output::with_zero_deriv(0.016_125_364_564_267),
                    (2, 1) => Output::with_zero_deriv(0.003_179_530_830_731),
                    (1, 2) => Output::with_zero_deriv(0.006_326_248_241_119),
                    _ => Output::zero(),
                }
            },
            (AtomType::Carbon, AtomType::Hydrogen) => {
                match (int_ccoord, int_hcoord) {
                    (0, 1) => Output::with_zero_deriv(0.209_336_732_825_0380),
                    (0, 2) => Output::with_zero_deriv(-0.064_449_615_432_525),
                    (0, 3) => Output::with_zero_deriv(-0.303_927_546_346_162),
                    (1, 0) => Output::with_zero_deriv(0.01),
                    (2, 0) => Output::with_zero_deriv(-0.122_042_146_278_2555),
                    (1, 1) => Output::with_zero_deriv(-0.125_123_400_628_7090),
                    (1, 2) => Output::with_zero_deriv(-0.298_905_245_783),
                    (3, 0) => Output::with_zero_deriv(-0.307_584_705_066),
                    (2, 1) => Output::with_zero_deriv(-0.300_529_172_406_7579),
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
        pub tcoord_i: f64,
        pub tcoord_j: f64,
        pub xcoord_ij: f64,
    }

    pub struct BrennerF {
        pub value: f64,
        pub d_ccoord_i: f64,
        pub d_hcoord_i: f64,
        pub d_xcoord_ij: f64,
    }

    impl BrennerF {
        fn zero() -> Self {
            BrennerF {
                value: 0.0,
                d_ccoord_i: 0.0,
                d_hcoord_i: 0.0,
                d_xcoord_ij: 0.0,
            }
        }
    }

    impl Input {
        pub fn compute(self) -> Output { compute(self) }
    }

    // Tables 4, 6, and 9
    fn compute(input: Input) -> Output {
        let Input { type_i, type_j, tcoord_i, tcoord_j, xcoord_ij } = input;


        // NOTE:
        //
        // RSP2 does not need the spline because it does not do anything
        // that requires the reactive parts of the potential.
        //
        // Thus, this function is a complete cop-out.
        panic!(
            "Not yet implemented for Brenner F: {}{} bond, Ni = {}, Nj = {}, Nconj = {}",
            type_i.char(), type_j.char(), tcoord_i, tcoord_j, xcoord_ij,
        );
    }

    // check if the value and all derivatives can be assumed to be zero,
    // without needing to compute N^{conj}
    pub fn can_assume_zero(
        (type_i, type_j): (AtomType, AtomType),
        (tcoord_i, tcoord_j): (f64, f64),
    ) -> bool {
        let frac_point = V2([tcoord_i, tcoord_j]);
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
//        value: ParamArray,   // indices are N_i^T, N_j^T, N_{ij}^{conj}
//        d_di: ParamArray,    // Derivative with respect to N_i^T
//        d_dj: ParamArray,    // Derivative with respect to N_j^T
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
//            //       (although it's hard to tell at first since they uniformly scale all parameters
//            //        by 1/2)
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
        pub tcoord_i: f64,
        pub tcoord_j: f64,
        pub xcoord_ij: f64,
    }

    pub struct BrennerT {
        pub value: f64,
        pub d_tcoord_i: f64,
        pub d_tcoord_j: f64,
        pub d_xcoord_ij: f64,
    }

    impl Output {
        fn with_zero_deriv(value: f64) -> Self {
            Output {
                value,
                d_tcoord_i: 0.0,
                d_tcoord_j: 0.0,
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
        let Input { type_i, type_j, tcoord_i, tcoord_j, xcoord_ij } = input;

        // NOTE:
        //
        // RSP2 does not need the spline because it does not do anything
        // that requires the reactive parts of the potential.
        //
        // Thus, this function is a complete cop-out.
        let frac_point = V3([tcoord_i, tcoord_j, xcoord_ij]);
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
                    "Not yet implemented for Brenner T: {}{} bond, Ni = {}, Nj = {}, Nconj = {}",
                    type_i.char(), type_j.char(), tcoord_i, tcoord_j, xcoord_ij,
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
                [2, 2, 9] => Output::with_zero_deriv(-0.008_096_75),  // Solid-state structure
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
        (tcoord_i, tcoord_j): (f64, f64),
    ) -> bool {
        let frac_point = V2([tcoord_i, tcoord_j]);
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
    pub struct Input {
        pub type_i: AtomType,
        pub cos: f64,
        pub tcoord: f64,
    }

    pub struct BrennerG {
        pub value: f64,
        pub d_cos: f64,
        pub d_tcoord: f64,
    }

    impl Input {
        pub fn compute(self) -> Output { compute(self) }
    }

    // free function for smaller indent
    fn compute(input: Input) -> Output {
        let Input { type_i, cos, tcoord } = input;

        // Almost all cases can be referred to a single polynomial evaluation
        // with no local dependence on tcoord.
        //
        // The sole exception is the regime 3.2 <= tcoord <= 3.7 for carbon.
        macro_rules! use_single_poly {
            ($poly:expr) => {{
                let d_tcoord = 0.0;
                let (value, d_cos) = polyval_dec(&$poly, cos);
                Output { value, d_cos, d_tcoord }
            }}
        }

        // NOTE: this matching and switching is done many times for the same `i`,
        //       and the coeffs from switch() could probably be precomputed.
        match type_i {
            AtomType::Carbon => {
                debug_assert!(cos > C_X_0 - 1e-9, "{} vs {}", cos, C_X_0);
                debug_assert!(cos < C_X_3 + 1e-9, "{} vs {}", cos, C_X_3);

                // prioritize the branch taken by graphene
                if cos < C_X_1 {
                    warn!("untested codepath: c17f90cf-a390-4f02-9233-78f2a7c9c424");
                    use_single_poly!(&C_COEFFS_1)

                    // branch point at 120 degrees
                } else if cos < C_X_2 {
                    warn!("untested codepath: e7367196-1df8-407b-8bad-357064cf6911");
                    use_single_poly!(&C_COEFFS_2)

                    // branch point at tetrahedral angle
                } else {
                    let (alpha, alpha_d_tcoord) = switch((3.2, 3.7), tcoord);

                    // easy way out for now
                    //
                    // condition on alpha_d_tcoord is necessary for cases that are just barely
                    // in the switching regime that may happen to have zero alpha
                    if alpha == 0.0 && alpha_d_tcoord == 0.0 {
                        warn!("untested codepath: cad37a46-e4ee-4baa-80bf-f1b689cebaa9");
                        use_single_poly!(C_COEFFS_3_LOW_COORDINATION)
                    } else if alpha == 1.0 && alpha_d_tcoord == 0.0 {
                        warn!("untested codepath: 37236e5f-9810-4ee5-a8c3-0a5150d9bd26");
                        use_single_poly!(&C_COEFFS_3_HIGH_COORDINATION)
                    } else {
                        warn!("untested codepath: fd6eff7e-b4b2-4bbf-ad89-3227a6099d59");
                        // The one case where use_single_poly! cannot be used.

                        // d(linterp(α, A, B)) = d(α A + (1 - α) B)
                        //                     = (A - B) dα + α dA + (1 - α) dB
                        //                     = ...let's not do this right now
                        unimplemented!()
                    }
                }
            },
            AtomType::Hydrogen => {
                warn!("untested codepath: 4d03fe04-5312-468e-9e30-01beddec4793");
                use_single_poly!(&H_COEFFS)
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
    // Coeffs listed from x**5 to x**0
    const C_X_0: f64 = -1.0;
    const C_X_1: f64 = -0.5;
    const C_X_2: f64 = -1.0/3.0;
    const C_X_3: f64 = 1.0;

    // Switch interval for tcoord in third region
    const C_T_LOW_COORDINATION: f64 = 3.2;
    const C_T_HIGH_COORDINATION: f64 = 3.7;

    // Segment 1: -1 to -1/2  (pi to 2pi/3)
    const C_COEFFS_1: &'static [f64] = &[
        -1.342399999999925, -4.927999999999722, -6.829999999999602,
        -4.3459999999997265, -1.0979999999999095, 0.002600000000011547,
    ];

    // Segment 2: -1/2 to -1/3  (2pi/3 to 109.47°)
    const C_COEFFS_2: &'static [f64] = &[
        35.3116800000094, 69.87600000001967, 55.94760000001625,
        23.43200000000662, 5.544400000001327, 0.6966900000001047,
    ];

    // Segment 3 (G): -1/3 to +1  (109.47° to 0°)
    const C_COEFFS_3_HIGH_COORDINATION: &'static [f64] = &[
        0.5064259725000047, 1.4271989062499966, 2.028821591249997,
        2.254920828750001, 1.4071827012500007, 0.37545,
    ];

    // Segment 3 (gamma): -1/3 to +1  (109.47° to 0°)
    const C_COEFFS_3_LOW_COORDINATION: &'static [f64] = &[
        -0.03793074749999925, 1.2711119062499994, -0.5613989287500004,
        -0.4328552912499998, 0.4892170612500001, 0.271856,
    ];

    // Full curve for hydrogen
    const H_COEFFS: &'static [f64] = &[
        -9.287290931116942, -0.29608733333332005, 13.589744997229507,
        -3.1552081666666805, 0.0755044338874331, 19.065124,
    ];

    /// Evaluate a polynomial with coefficients listed in decreasing order
    pub(super) fn polyval_dec(coeffs: &[f64], x: f64) -> (f64, f64) {
        let poly_coeffs = coeffs.iter().cloned();
        let deriv_coeffs = coeffs.iter().rev().skip(1).enumerate().map(|(n, x)| (n + 1) as f64 * x).rev();
        (_polyval_dec(poly_coeffs, x), _polyval_dec(deriv_coeffs, x))
    }

    pub(super) fn _polyval_dec(coeffs: impl Iterator<Item=f64>, x: f64) -> f64 {
        coeffs.fold(0.0, |acc, c| acc * x + c)
    }

    #[test]
    fn common_cases() {
        // graphite
        let BrennerG { value, d_cos, d_tcoord } = Input {
            type_i: AtomType::Carbon,
            cos: f64::cos(120.0 * PI / 180.0) + 1e-12,
            tcoord: 3.0,
        }.compute();
        // Brenner Table 3
        assert_close!(value, 0.05280);
        assert_close!(d_cos, 0.17000);
        assert_eq!(d_tcoord, 0.0);

        let BrennerG { value, d_cos, d_tcoord } = Input {
            type_i: AtomType::Carbon,
            cos: f64::cos(120.0 * PI / 180.0) - 1e-12,
            tcoord: 3.0,
        }.compute();
        assert_close!(value, 0.05280);
        assert_close!(d_cos, 0.17000);
        assert_eq!(d_tcoord, 0.0);

        // diamond
        let BrennerG { value, d_cos, d_tcoord } = Input {
            type_i: AtomType::Carbon,
            cos: -1.0/3.0 + 1e-12,
            tcoord: 4.0,
        }.compute();
        assert_close!(value, 0.09733);
        assert_close!(d_cos, 0.40000);
        assert_eq!(d_tcoord, 0.0);

        let BrennerG { value, d_cos, d_tcoord } = Input {
            type_i: AtomType::Carbon,
            cos: -1.0/3.0 - 1e-12,
            tcoord: 4.0,
        }.compute();
        assert_close!(value, 0.09733);
        assert_close!(d_cos, 0.40000);
        assert_eq!(d_tcoord, 0.0);
    }

    #[test]
    fn numerical_derivatives() {
        macro_rules! check_derivatives {
            ($input:expr) => {{
                let Input { type_i, cos, tcoord } = $input;
                let BrennerG { value: _, d_cos, d_tcoord } = $input.compute();
                assert_close!(
                    rel=1e-7,
                    d_cos,
                    numerical::slope(1e-7, None, cos, |cos| Input { type_i, cos, tcoord }.compute().value),
                );
                assert_close!(
                    rel=1e-7,
                    d_tcoord,
                    numerical::slope(1e-7, None, tcoord, |tcoord| Input { type_i, cos, tcoord }.compute().value),
                );
            }}
        }

        // 120 degrees is a branch point so try both sides as well as straddling it
        for type_i in AtomType::iter_all() {
            let coses = [
                // within a region
                C_X_0 + 1e-4, C_X_1 - 1e-4,
                C_X_1 + 1e-4, C_X_2 - 1e-4,
                C_X_2 + 1e-4, C_X_3 - 1e-4,
                // straddle two regions
                C_X_1,
                C_X_2,
            ];
            let tcoords = [
                C_T_LOW_COORDINATION - 1e-4,
                C_T_LOW_COORDINATION + 1e-4,
                C_T_HIGH_COORDINATION - 1e-4,
                C_T_HIGH_COORDINATION + 1e-4,
                C_T_HIGH_COORDINATION,
                C_T_LOW_COORDINATION,
            ];
            for &cos in &coses {
                for &tcoord in &tcoords {
                    check_derivatives!{ Input { type_i, cos, tcoord } }
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

//-----------------------------------------------------------------------------

#[inline(always)] // elide large stack-to-stack copies
fn sbvec_scaled<T: ::std::ops::MulAssign<f64>>(f: f64, mut xs: SiteBondVec<T>) -> SiteBondVec<T>
{ scale_mut(f, &mut xs); xs }

#[inline(always)] // elide large stack-to-stack copies
fn sbvec_filled<T: Clone>(fill: T, len: usize) -> SiteBondVec<T>
{ ::std::iter::repeat(fill).take(len).collect() }

fn scaled<T: ::std::ops::MulAssign<f64>>(f: f64, mut xs: Vec<T>) -> Vec<T>
{ scale_mut(f, &mut xs); xs }


fn scale_mut<T: ::std::ops::MulAssign<f64>>(factor: f64, xs: &mut [T]) {
    for x in xs { *x *= factor; }
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

//-----------------------------------------------------------------------------

// FIXME delete once integrated into main codebase
mod util {
    use super::*;

    // these f64 -> i32 conversions are written on a silly little type
    // simply to avoid having a function with a signature like 'fn f(x: f64, tol: f64)'
    // where the arguments could be swapped
    pub(crate) struct Tol(pub(crate) f64);
    #[allow(unused)]
    impl Tol {
        pub(crate) fn unfloat(&self, x: f64) -> Option<i32>
        {Some({
            let r = x.round();
            if (r - x).abs() > self.0 {
                return None;
            }
            r as i32
        })}

        pub(crate) fn unfloat_v3(&self, v: &V3) -> Option<V3<i32>>
        { v.opt_map(|x| self.unfloat(x)) }
    }
}

// FIXME remove
fn main() {}
