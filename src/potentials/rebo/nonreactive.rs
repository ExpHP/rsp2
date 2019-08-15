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

//! Implementation of the second-generation REBO potential, without the reactive bits
//!
//! This implementation was written with the help of the AIREBO paper,
//! and by reviewing the implementation in LAMMPS.
//!
//! # Reactive parts
//!
//! The code was originally written to support fractional weights, but this was dropped
//! for ease of maintenance, since `rsp2` does not currently actually require a reactive
//! potential.  If bond lengths in the transition radius are found, `rsp2` will bail out.
//!
//! Dropping support for fractional weights *significantly* simplifies the computation
//! of forces.
//!
//! The original, almost certainly VERY buggy reactive code still exists in the git history
//! (see `557b47c570517b8d604ce9767937fdf225084c14` and diff `reactive.rs` and `nonreactive.rs`),
//! and many parts of the current code are still written in a manner that will make it less
//! difficult to add them back by carrying them around and multiplying them in even though
//! they are all `1.0` or `0.0`.
//!
//! # Citations
//!
//! * **2nd gen REBO:** Donald W Brenner et al 2002 J. Phys.: Condens. Matter 14 783
//! * **AIREBO:** Steven J Stuart et al J. Chem. Phys. 112, 6472 (2000)
//! * **LAMMPS:** S. Plimpton, J Comp Phys, 117, 1-19 (1995)

use super::splines::{self, TricubicGrid, BicubicGrid};
use crate::FailResult;
#[cfg(test)] use crate::util::uniform;
use crate::util::{try_num_grad_v3, num_grad_v3, switch};

use rayon_cond::CondIterator;
use rsp2_structure::{Element, Coords};
use rsp2_structure::bonds::{FracBond, PeriodicGraph};
use rsp2_minimize::numerical;
#[cfg(test)]
use rsp2_minimize::numerical::DerivativeKind;
use rsp2_array_types::{V2, V3, M33, M3};
#[allow(unused)] // https://github.com/rust-lang/rust/issues/45268
use rsp2_newtype_indices::{Idx, IndexVec, Indexed, self as idx};

use stack::{ArrayVec, Vector as StackVector};
#[cfg(test)]
use std::f64::{consts::PI};
use std::f64::NAN;
use std::ops;
use std::borrow::Cow;
#[allow(unused)] // https://github.com/rust-lang/rust/issues/45268
use petgraph::prelude::EdgeRef;
use enum_map::EnumMap;
use rayon::prelude::*;
use slice_of_array::prelude::*;

//-------------------------------------------------------
// Debugging utils:

/// Emit tracing debug output.
///
/// Optimizations will constant-fold this so that no code is emitted if the requisite
/// env var is not defined during compilation.
///
/// Ideally, you also have a version of lammps that is patched to produce similar output
/// to the usage of `dbg!` in this file. See the `sorted-diff` and `filter-out` scripts
/// in the rebo test directory for how to compare these outputs.
macro_rules! dbg {
    ($flag:expr, $($t:tt)*) => {
        if option_env!("RSP2_REBO_TRACE") == Some("1".as_ref()) {
            if $flag == Debug::Auto {
                println!($($t)*);
            }
        }
    };
}

/// Used to prevent extraneous calls to a function from producing misleading debug output.
///
/// The default value of `Auto` is generally propagated from the root function all the way down.
///
/// Some `compute` functions in this module have a `compute_paranoid(tol)` alternative that
/// checks numerical derivatives on every call.  The calls made during numerical differentiation
/// use `Debug::Never` to suppress debug output.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum Debug { Auto, Never }

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
// 2. When inputs and outputs of a function can be ambiguous, simulate named parameters by
//    wrapping it in a module that defines input/output structs.
//
// 3. Fully destructure the output of any function that computes pieces of the potential,
//    so that the compiler complains about missing fields when one is added.
//
// 4. A function should not accept derivatives as arguments, because such a function
//    cannot be tested with numerical differentiation. It should instead return
//    derivatives with respect to its immediate inputs and let the caller handle them.
//
// 5. Naming: Use indices `ijkl` in naming as in the papers.  `i` and `j` are the source
//    and target of a bond.  `k` is another site bonded with site `i`.  `l` is another site
//    bonded with `j`.
//
//    Sometimes the indices are information overload. For this I can only offer my apologies
//    and suggest you take a break.  I experimented with factoring some parts out in ways that
//    allowed some indices to be removed, but in the long run I've found that keeping the indices
//    around makes the code easier to maintain regardless of how it is factored.
//
//    Use plurals to denote when something is an array over its last index (alphabetically).
//
//    E.g.
//    - `cos_ijk_d_delta_ik` is the gradient of `cos_ijk` with respect to `delta_ik`.
//    - `coses_ijk_d_delta_ik` would be a `SiteBondVec<V3>` where the `k`th element is the
//      gradient of `cos_ijk` with respect to `delta_ik`.
//    - `coses_ijk_d_delta_ij` would be a `SiteBondVec<V3>` where the `k`th element is the
//      gradient of `cos_ijk` with respect to `delta_ij`.
//
//    (FIXME: Some parts don't adhere to these naming rules yet)
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

    pub fn from_element(elem: rsp2_structure::Element) -> FailResult<Self> {
        use rsp2_structure::consts;
        match elem {
            consts::CARBON => Ok(AtomType::Carbon),
            consts::HYDROGEN => Ok(AtomType::Hydrogen),
            _ => bail!("REBO only supports Carbon and Hydrogen"),
        }
    }

    pub fn iter_all() -> impl Iterator<Item=AtomType> {
        std::iter::once(AtomType::Carbon).chain(::std::iter::once(AtomType::Hydrogen))
    }
}
type TypeMap<T> = EnumMap<AtomType, T>;

//---------------------------------------------------------------------------------

// NOTE: While it's unlikely that one will ever confuse a site index for a bond index,
//       wrapping BondI is extremely useful to prevent accidental confusion between
//       "site-bond" indices (indices into a SiteBondVec) and "bond" indices (unique
//       indices for all bonds)
pub(crate) use self::newtype_indices::*;
mod newtype_indices {
    // mod so they aren't public
    newtype_index!{SiteI}
    newtype_index!{BondI}
}

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
        /// The would-be smooth cutoff region with respect to bond length in reactive potential.
        ///
        /// We don't model the smooth cutoff, but this is useful to keep around for reference,
        /// and could be used to generate warnings.
        pub cutoff_region: (f64, f64), // Å
        /// This is the region that actually defines our weights.
        ///
        /// * Below the region: Weight is 1.
        /// * Above the region: Weight is 0.
        /// * Inside the region: The computation is aborted.
        pub forbidden_region: (f64, f64), // Å

        /// A LAMMPS/Airebo thing.
        ///
        /// This parameter is used by Stuart in the `exp(lambda)` for AIREBO.
        /// The CC value is never used. Also not used unless `use_airebo_lambda = true`.
        pub airebo_rho: f64, // Å

        /// A LAMMPS/Airebo thing.
        ///
        /// `r_max'` in the AIREBO paper, `rcmaxp` in LAMMPS.  In AIREBO, the weights that
        /// appear in the dihedral sum are defined using a slightly shorter cutoff interval for
        /// CH bonds than the weights that appear everywhere else.
        ///
        /// (It is not yet clear to me whether this change was made specifically to accomodate
        ///  AIREBO, or if it is an improvement that can be transferred to REBO   -ML)
        ///
        /// This has no impact on our nonreactive potential at all; it's existence
        /// is just to serve as a comment about something that's easy to miss.
        pub airebo_cutoff_max_2: f64, // Å
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
                cutoff_region: (1.7, 2.0), // Å
                forbidden_region: (1.8, 1.9), // Å

                airebo_rho: NAN, // unused for CC
                airebo_cutoff_max_2: 2.0, // Å (LAMMPS value)
            };

            // Brenner Table 6
            let type_params_hh = TypeParams {
                B: [29.632_593, 0.0, 0.0], // eV
                beta: [1.715_892_17, 0.0, 0.0], // Å-1
                Q: 0.370_471_487_045, // Å
                A: 32.817_355_747, // Å
                alpha: 3.536_298_648, // Å-1
                cutoff_region: (1.1, 1.7), // Å
                forbidden_region: (1.3, 1.5), // Å

                airebo_rho: 0.7415887, // Å (LAMMPS value)
                airebo_cutoff_max_2: 1.7, // Å (LAMMPS value)
            };

            // Brenner Table 7
            let type_params_ch = TypeParams {
                B: [32.355_186_6587, 0.0, 0.0], // eV
                beta: [1.434_458_059_25, 0.0, 0.0], // Å-1
                Q: 0.340_775_728, // Å
                A: 149.940_987_23, // eV
                alpha: 4.102_549_83, // Å-1
                cutoff_region: (1.3, 1.8), // Å
                forbidden_region: (1.5, 1.6), // Å

                airebo_rho: 1.09, // Å (LAMMPS value)
                airebo_cutoff_max_2: 1.6, // Å (LAMMPS value)
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
        /// The G splines (functions of bond angle) are also different, solved from
        /// the parameters in the AIREBO paper.  (as a result, these are not quite
        /// identical to LAMMPS's splines, which were formatted to poor precision)
        ///
        /// These include Falvo's fixes for hydrocarbons that are not yet available
        /// in lammps upstream.
        pub fn new_lammps() -> Self {
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
                cutoff_region: (1.7, 2.0), // Å
                forbidden_region: (1.8, 1.9), // Å

                airebo_rho: NAN, // unused for CC
                airebo_cutoff_max_2: 2.0, // Å
            };

            let type_params_hh = TypeParams {
                B: [28.2297, 0.0, 0.0], // eV
                beta: [1.708, 1.0, 1.0], // Å-1
                Q: 0.370, // Å
                A: 31.6731, // Å
                alpha: 3.536, // Å-1
                cutoff_region: (1.1, 1.7), // Å
                forbidden_region: (1.3, 1.5), // Å

                airebo_rho: 0.7415887, // Å
                airebo_cutoff_max_2: 1.7, // Å
            };

            let type_params_ch = TypeParams {
                B: [32.355_186_658_732_56, 0.0, 0.0], // eV
                beta: [1.434_458_059_249_837, 0.0, 0.0], // Å-1
                Q: 0.340_775_728_225_7080, // Å
                A: 149.940_987_228_812, // eV
                alpha: 4.102_549_828_548_784, // Å-1
                cutoff_region: (1.3, 1.8), // Å
                forbidden_region: (1.5, 1.6), // Å

                airebo_rho: 1.09, // Å
                airebo_cutoff_max_2: 1.6, // Å
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
                G: Cow::Owned(splines::G::STUART),
                P: Cow::Borrowed(&splines::P::BRENNER),
                F: Cow::Borrowed(&splines::F::LAMMPS),
                T: Cow::Borrowed(&splines::T::STUART),
                by_type,
            }
        }

        /// Same as `new_lammps`, but without Falvo's fixes.
        ///
        /// This reproduces bugs that exist in current versions of lammps
        /// when using rebo on hydrocarbons.
        pub fn new_favata() -> Self {
            let mut params = Self::new_lammps();
            params.P = Cow::Borrowed(&splines::P::FAVATA);
            params
        }
    }
}

//---------------------------------------------------------------------------------

pub use self::interactions::Interactions;
mod interactions {
    use super::*;

    // collects all the terms we need to compute
    #[derive(Debug, Clone)]
    pub struct Interactions {
        /// CSR-style divider indices for bonds at each site.
        bond_div: IndexVec<SiteI, BondI>,

        site_type: IndexVec<SiteI, AtomType>,
        bond_image_diff: IndexVec<BondI, V3<i32>>,
        bond_reverse_index: IndexVec<BondI, BondI>,
        bond_source: IndexVec<BondI, SiteI>,
        bond_target: IndexVec<BondI, SiteI>,
    }

    impl Interactions {
        /// Identify which pairs of atoms are close enough to interact and build a list of
        /// interactions.
        ///
        /// This will warn or generate errors when bond lengths fall into the reactive regime,
        /// as if the `check_distances` method was called.
        pub fn compute(
            params: &Params,
            coords: &Coords,
            types: &[AtomType],
            bond_graph: &PeriodicGraph,
        ) -> FailResult<Self> {
            let mut bond_div = IndexVec::<SiteI, _>::from_raw(vec![BondI(0)]);
            bond_div.raw.reserve(bond_graph.node_count());

            let mut bond_source = IndexVec::<BondI, SiteI>::with_capacity(bond_graph.edge_count());
            let mut bond_target = IndexVec::<BondI, SiteI>::with_capacity(bond_graph.edge_count());
            let mut bond_image_diff = IndexVec::<BondI, V3<i32>>::with_capacity(bond_graph.edge_count());
            let site_type = IndexVec::<SiteI, _>::from_raw(types.to_vec());

            let cart_cache = coords.with_carts(coords.to_carts());

            // Make a pass to get all the bond divs right.
            for node in bond_graph.node_indices() {
                let site_i = SiteI(node.index());

                for frac_bond_ij in bond_graph.frac_bonds_from(site_i.index()) {
                    let site_j = SiteI(frac_bond_ij.to);
                    let cart_vector = frac_bond_ij.cart_vector_using_cache(&cart_cache).unwrap();

                    if let IsInteracting(false) = check_distance(
                        params, cart_vector.norm(), (site_type[site_i], site_type[site_j]),
                    )? {
                        continue;
                    }

                    bond_source.push(site_i);
                    bond_target.push(site_j);
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

            // We can no longer use bond_graph.frac_bonds_from because not all bonds in the input
            // graph are represented in our vectors.
            let frac_bonds_from = |site_i: SiteI| {
                let range = bond_div[site_i].0..bond_div[site_i.next()].0;
                zip_eq!(
                    &bond_source.raw[range.clone()],
                    &bond_target.raw[range.clone()],
                    &bond_image_diff.raw[range.clone()],
                ).map(|(&SiteI(from), &SiteI(to), &image_diff)| FracBond { from, to, image_diff })
            };

            for node in bond_graph.node_indices() {
                let site_i = SiteI(node.index());

                for frac_bond_ij in frac_bonds_from(site_i) {
                    let site_j = SiteI(frac_bond_ij.to);
                    let sbindex_ji = {
                        frac_bonds_from(site_j)
                            .position(|bond| frac_bond_ij == bond.flip())
                    };
                    let sbindex_ji = match sbindex_ji {
                        Some(x) => x,
                        None => bail!("A bond has no counterpart in the reverse direction!"),
                    };
                    let bond_ji = BondI(bond_div[site_j].0 + sbindex_ji);
                    bond_reverse_index.push(bond_ji);

                    if (site_type[site_i], site_type[site_j]) == (AtomType::Hydrogen, AtomType::Hydrogen) {
                        println!("HH bond: {} {}", site_i, site_j);
                    }
                } // for bond_ij
            } // for node

            assert_eq!(bond_reverse_index.len(), bond_div.raw.last().unwrap().0);
            Ok(Interactions {
                bond_div, site_type,
                bond_target, bond_image_diff, bond_reverse_index, bond_source,
            })
        }
    }

    impl Interactions {
        pub fn num_sites(&self) -> usize { self.site_type.len() }
        pub fn num_bonds(&self) -> usize { self.bond_target.len() }

        #[inline(always)] pub(crate) fn site_type(&self, site: SiteI) -> AtomType { self.site_type[site] }
        #[inline(always)] pub(crate) fn bond_source(&self, bond: BondI) -> SiteI { self.bond_source[bond] }
        #[inline(always)] pub(crate) fn bond_target(&self, bond: BondI) -> SiteI { self.bond_target[bond] }
        #[inline(always)] pub(crate) fn bond_reverse_index(&self, bond: BondI) -> BondI { self.bond_reverse_index[bond] }
        #[inline(always)] pub(crate) fn bond_image_diff(&self, bond: BondI) -> V3<i32> { self.bond_image_diff[bond] }
        /// index of a bond into a SiteBondVec for its source Site
        #[inline(always)] pub(crate) fn bond_sbvec_index(&self, bond: BondI) -> usize {
            bond.0 - self.site_bond_range(self.bond_source[bond]).start
        }

        #[inline(always)]
        pub(crate) fn bond_targets(&self, site: SiteI) -> impl ExactSizeIterator<Item=SiteI> + '_ {
            self.bonds(site).map(move |x| self.bond_target(x))
        }

        // NOTE: we often can't use this due to rayon
        pub(crate) fn sites(&self) -> impl ExactSizeIterator<Item=SiteI> + '_ {
            (0..self.num_sites()).map(SiteI)
        }

        pub(crate) fn bonds(&self, site: SiteI) -> impl ExactSizeIterator<Item=BondI> + '_ {
            self.site_bond_range(site).map(BondI)
        }

        // Type-safe extraction of a "site-bond"-indexed slice of data from a vector of data
        // at all bonds.
        pub(crate) fn site_bond_slice<'b, T>(&self, site: SiteI, data: &'b Indexed<BondI, [T]>) -> &'b [T] {
            &data.raw[self.site_bond_range(site)]
        }

        pub(crate) fn site_bond_slice_mut<'b, T>(&self, site: SiteI, data: &'b mut Indexed<BondI, [T]>) -> &'b mut [T] {
            &mut data.raw[self.site_bond_range(site)]
        }

        pub(crate) fn site_bond_range(&self, site: SiteI) -> std::ops::Range<usize> {
            self.bond_div[site].0..self.bond_div[site.next()].0
        }

        pub(crate) fn bond_is_canonical(&self, bond: BondI) -> bool {
            FracBond {
                from: self.bond_source[bond].0,
                to: self.bond_target[bond].0,
                image_diff: self.bond_image_diff[bond],
            }.is_canonical()
        }
    }

    impl Interactions {
        /// Check that all bond lengths are outside of the reactive regime.
        ///
        /// NOTE: This can only really catch bonds that break.  It won't notice if any atoms have
        /// moved into bonding range after the construction of `Interactions`.
        ///
        /// Prints warnings or returns `Err` on really bad cases, as configured in [`Params`].
        pub fn check_distances(&self, params: &Params, coords: &Coords, use_rayon: bool) -> FailResult<()> {
            let bond_deltas = compute_bond_deltas(coords, &self, use_rayon);
            for (bond_ij, delta_ij) in bond_deltas.iter_enumerated() {
                let type_i = self.site_type(self.bond_source(bond_ij));
                let type_j = self.site_type(self.bond_target(bond_ij));
                check_distance(params, delta_ij.norm(), (type_i, type_j))?;
            }
            Ok(())
        }
    }
}

pub fn find_all_interactions(
    params: &Params,
    coords: &Coords,
    elements: &[Element],
) -> FailResult<Interactions> {
    let ref types = elements.iter().cloned().map(AtomType::from_element).collect::<FailResult<Vec<_>>>()?;
    let ref graph = {
        rsp2_structure::bonds::FracBonds::compute_with_meta(
            coords, types.iter().cloned(),
            // FIXME should return None for other elements
            |&a, &b| Some(params.by_type[a][b].cutoff_region.1),
        )?.to_periodic_graph()
    };

    Interactions::compute(params, coords, types, graph)
}

//---------------------------------------------------------------------------------

pub struct BondGrad {
    pub plus_site: usize,
    pub minus_site: usize,
    pub cart_vector: V3,
    pub grad: V3,
}

pub fn compute(
    params: &Params,
    interactions: &Interactions,
    coords: &Coords,
    use_rayon: bool,
) -> FailResult<(f64, Vec<V3>)> {
    let bond_deltas = compute_bond_deltas(coords, &interactions, use_rayon);
    let (value, d_deltas) = compute_rebo_bonds(params, &interactions, &bond_deltas, use_rayon)?;

    let mut d_positions = IndexVec::from_elem_n(V3::zero(), interactions.num_sites());
    for site_i in interactions.sites() {
        for bond_ij in interactions.bonds(site_i) {
            let site_j = interactions.bond_target(bond_ij);

            // delta_ij = (-pos_i) + pos_j
            d_positions[site_i] -= d_deltas[bond_ij];
            d_positions[site_j] += d_deltas[bond_ij];
        }
    }
    Ok((value, d_positions.raw))
}

pub fn compute_by_bond(
    params: &Params,
    interactions: &Interactions,
    coords: &Coords,
    use_rayon: bool,
) -> FailResult<(f64, Vec<BondGrad>)> {
    let bond_deltas = compute_bond_deltas(coords, &interactions, use_rayon);
    let (value, d_deltas) = compute_rebo_bonds(params, &interactions, &bond_deltas, use_rayon)?;

    let mut grad_items = Vec::with_capacity(interactions.num_bonds());
    for site_i in interactions.sites() {
        for bond_ij in interactions.bonds(site_i) {
            let site_j = interactions.bond_target(bond_ij);

            // delta_ij = (-pos_i) + pos_j
            grad_items.push(BondGrad {
                plus_site: site_j.0,
                minus_site: site_i.0,
                cart_vector: bond_deltas[bond_ij],
                grad: d_deltas[bond_ij],
            });
        }
    }
    Ok((value, grad_items))
}

fn compute_bond_deltas(
    coords: &Coords,
    interactions: &Interactions,
    use_rayon: bool,
) -> IndexVec<BondI, V3> {
    let site_carts = IndexVec::<SiteI, _>::from_raw(coords.to_carts());
    let vec = CondIterator::new(0..interactions.num_bonds(), use_rayon).map(|bond| {
        let bond = BondI(bond);
        let cart_from = site_carts[interactions.bond_source(bond)];
        let cart_to = site_carts[interactions.bond_target(bond)];
        let image_diff = interactions.bond_image_diff(bond);
        cart_to - cart_from + image_diff.map(|x| x as f64) * coords.lattice()
    }).collect();
    IndexVec::from_raw(vec)
}

fn compute_rebo_bonds(
    params: &Params,
    interactions: &Interactions,
    bond_deltas: &Indexed<BondI, [V3]>,
    use_rayon: bool,
) -> FailResult<(f64, IndexVec<BondI, V3>)> {
    _compute_rebo_bonds(params, interactions, bond_deltas, use_rayon, Debug::Auto)
}

fn _compute_rebo_bonds(
    params: &Params,
    interactions: &Interactions,
    bond_deltas: &Indexed<BondI, [V3]>,
    use_rayon: bool,
    dbg: Debug,
) -> FailResult<(f64, IndexVec<BondI, V3>)> {
    // Brenner:
    // Eq  1:  V = sum_{i < j} V^R(r_ij) + b_{ij} V^A(r_ij)
    // Eq  5:  V^R(r) = f(r) (1 + Q/r) A e^{-alpha r}
    // Eq  6:  V^A(r) = - f(r) sum_{n in 1..=3} B_n e^{-beta_n r}
    // Eq  3:  b_{ij} = 0.5 * (b_{ij}^{sigma-pi} + b_{ji}^{sigma-pi}) + b_ij^pi
    // Eq  4:  b_{ij}^{pi} = PI_{ij}^{RC} + b_{ij}^{DH}
    // Eq 14:  PI_{ij}^{RC} = F spline
    //
    // r_ij are bond vectors
    // f_ij is bond weight (0 to 1)
    // Q, A, alpha, beta_n, B_n are parameters
    // b_{ij}^{pi}, b_{ij}^{sigma-pi}, b_{ij}^{DH} are complicated bond-order subexpressions

    dbg!(dbg, "nsites: {}", interactions.num_sites());
    dbg!(dbg, "nbonds: {}", interactions.num_bonds());

    //-------------------
    // NOTE:
    //
    // We will also define U_ij (and U^A_ij, and U^R_ij) to be the V terms without the
    // f_ij scale factor.
    //
    // Eq  5':  U^R(r) = (1 + Q/r) A e^{-alpha r}
    // Eq  6':  U^A(r) = - sum_{n in 1..=3} B_n e^{-beta_n r}
    //
    // We also redefine the sums in the potential to be over all i,j pairs, not just i < j.
    //
    // Eq 1':     V = sum_{i != j} V_ij
    // Eq 2':  V_ij = 0.5 * V^R_ij + b_ij * V^A_ij
    // Eq 3':  b_ij = 0.5 * b_ij^{sigma-pi} + boole(i < j) * b_ij^{pi}

    // On large systems, our performance is expected to be bounded by cache misses.
    // For this reason, we should aim to make as few passes over the data as necessary,
    // leaving vectorization as only a secondary concern.
    //
    // TODO: Actual benchmarks >_>
    struct FirstPassSiteData {
        // Brenner's N_i
        tcoord: f64,
        // Brenner's f_ij.
        //
        // We keep these around because they may be zero. (even for nonreactive REBO,
        // we use a bond search radius that is large enough to include the point of
        // zero weight so that we can detect if a fractional weight ever appears)
        bond_weight: SiteBondVec<f64>,
        // Brenner's V^R(r_ij)
        bond_VR: SiteBondVec<f64>,
        bond_VR_d_delta: SiteBondVec<V3>,
        // Brenner's V^A(r_ij)
        bond_VA: SiteBondVec<f64>,
        bond_VA_d_delta: SiteBondVec<V3>,
    }

    let site_data = IndexVec::<SiteI, _>::from_raw({
        CondIterator::new(0..interactions.num_sites(), use_rayon).map(|site_i| {
            let site_i = SiteI(site_i);
            let type_i = interactions.site_type(site_i);

            let mut tcoord = 0.0;
            let mut bond_VR = SiteBondVec::new();
            let mut bond_VR_d_delta = SiteBondVec::new();
            let mut bond_VA = SiteBondVec::new();
            let mut bond_VA_d_delta = SiteBondVec::new();
            let mut bond_weight = SiteBondVec::new();

            for bond in interactions.bonds(site_i) {
                let site_j = interactions.bond_target(bond);
                let type_j = interactions.site_type(site_j);
                let delta = bond_deltas[bond];

                let (length, length_d_delta) = norm(delta);
                let EasyParts {
                    weight, VA, VA_d_length, VR, VR_d_length,
                } = easy_parts::Input {
                    params, type_i, type_j, length
                //}.compute_paranoid(dbg, 1e-9)?;
                }.compute(dbg)?;

                dbg!(dbg, "length: {:.9}", length);
                dbg!(dbg, "weight: {:.9}", weight);
                tcoord += weight;
                bond_weight.push(weight);

                dbg!(dbg, "VR: {:.9}", VR);
                bond_VR.push(VR);
                bond_VR_d_delta.push(VR_d_length * length_d_delta);

                dbg!(dbg, "VA: {:.9}", VA);
                bond_VA.push(VA);
                bond_VA_d_delta.push(VA_d_length * length_d_delta);
            } // for _ in interactions.bonds(site)

            Ok(FirstPassSiteData {
                tcoord,
                bond_weight,
                bond_VR, bond_VR_d_delta,
                bond_VA, bond_VA_d_delta,
            })
        }).collect::<FailResult<_>>()?
    });

    let out = CondIterator::new(0..interactions.num_sites(), use_rayon).map(|raw_site_i| {
        let site_i = SiteI(raw_site_i);
        let FirstPassSiteData {
            tcoord: _,
            ref bond_weight,
            ref bond_VR, ref bond_VR_d_delta,
            ref bond_VA, ref bond_VA_d_delta,
        } = site_data[site_i];

        // Eq 2':  V_ij = 0.5 * V^R_ij + b_ij * V^A_ij
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
        for _bond_VR_ij in bond_VR {
            dbg!(dbg, "Vterm(R): {:.9}", 0.5 * _bond_VR_ij);
        }
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
            bond_deltas: interactions.site_bond_slice(site_i, bond_deltas),
//        }.compute_paranoid(dbg, 1e-7);
        }.compute(dbg);
        let SiteSigmaPiTerm {
            value: Vsp_i,
            d_deltas: Vsp_i_d_delta,
            d_VAs: Vsp_i_d_VA,
        } = out;

        site_V += Vsp_i;
        axpy_mut(&mut site_V_d_delta, 1.0, &Vsp_i_d_delta);
        for sbindex_ij in 0..bond_VA.len() {
            let Vsp_i_d_VA_ij = Vsp_i_d_VA[sbindex_ij];
            let VA_ij_d_delta_ij = bond_VA_d_delta[sbindex_ij];
            site_V_d_delta[sbindex_ij] += Vsp_i_d_VA_ij * VA_ij_d_delta_ij;
        }

        //-----------------------------------------------
        // The pi terms
        //-----------------------------------------------

        // These are the only parts that depend on other sites' bond deltas.

        // Eq 3':  b_ij = boole(i < j) * b_ij^{pi}
        for (sbidx_ij, bond_ij) in interactions.bonds(site_i).enumerate() {
            if !interactions.bond_is_canonical(bond_ij) {
                continue;
            }
            let site_j = interactions.bond_target(bond_ij);

            let ref weights_ik = bond_weight;
            let ref weights_jl = site_data[site_j].bond_weight;

            // NOTE:
            // * In the REBO paper, these weights are the same as all the others.
            // * In the AIREBO paper, these weights use a slightly different (shorter) interval:
            let _the_interval_airebo_would_use = |i, k| (
                params.by_type[i][k].cutoff_region.0,
                params.by_type[i][k].airebo_cutoff_max_2,
            );
            // For our purposes, with this being non-reactive REBO, the difference is irrelevant;
            // both ranges produce the same weights.
            let alt_weights_ik = weights_ik;
            let alt_weights_jl = weights_jl;

            let gather_site_tcoords = |site| -> SiteBondVec<_> {
                interactions.bond_targets(site)
                    .map(|target| site_data[target].tcoord)
                    .collect()
            };

            let out = bondorder_pi::Input {
                params, interactions, bond_ij,
                weights_ik, weights_jl,
                alt_weights_ik, alt_weights_jl,
                tcoords_k: &gather_site_tcoords(site_i),
                tcoords_l: &gather_site_tcoords(site_j),
                deltas_ik: interactions.site_bond_slice(site_i, &bond_deltas),
                deltas_jl: interactions.site_bond_slice(site_j, &bond_deltas),
            }.compute(dbg);
//            }.compute_paranoid(dbg, 1e-9);
            let BondOrderPi {
                value: bpi,
                d_deltas_ik: bpi_d_deltas_ik,
                d_deltas_jl: bpi_d_deltas_jl,
            } = out;
            dbg!(dbg, "bpi: {:.9}", bpi);

            let VA_ij = bond_VA[sbidx_ij];
            let VA_ij_d_delta_ij = bond_VA_d_delta[sbidx_ij];

            dbg!(dbg, "Vterm(pi): {:.9}", bpi * VA_ij);
            site_V += bpi * VA_ij;
            site_V_d_delta[sbidx_ij] += bpi * VA_ij_d_delta_ij;

            axpy_mut(&mut site_V_d_delta, VA_ij, &bpi_d_deltas_ik);
            site_V_d_other_deltas.push((site_j, sbvec_scaled(VA_ij, bpi_d_deltas_jl)));
        }

        (site_V, site_V_d_delta, site_V_d_other_deltas)
    }).collect::<Vec<_>>();

    // put it all together in serial code
    let mut value = 0.0;
    let mut d_deltas = IndexVec::new();
    let mut d_other_deltas = vec![];
    for (site_V, site_V_d_delta, site_V_d_other_deltas) in out {
        value += site_V;
        d_deltas.extend(site_V_d_delta);
        d_other_deltas.extend(site_V_d_other_deltas);
    }
    assert_eq!(d_deltas.len(), interactions.num_bonds());

    // absorb the other terms we couldn't take care of into d_deltas
    // and end this miserable function for good
    for (site_i, d_deltas_ij) in d_other_deltas {
        let _: SiteBondVec<V3> = d_deltas_ij;
        axpy_mut(interactions.site_bond_slice_mut(site_i, &mut d_deltas), 1.0, &d_deltas_ij);
    }

    assert_eq!(d_deltas.len(), interactions.num_bonds());
    Ok((value, d_deltas))
}

use self::easy_parts::EasyParts;
mod easy_parts {
    use super::*;

    pub(super) type Output = EasyParts;

    #[derive(Debug, Clone)]
    pub(super) struct Input<'a> {
        pub params: &'a Params,
        pub type_i: AtomType,
        pub type_j: AtomType,
        pub length: f64,
    }

    pub(super) struct EasyParts {
        pub weight: f64,
        pub VA: f64,
        pub VA_d_length: f64,
        pub VR: f64,
        pub VR_d_length: f64,
    }

    impl<'a> Input<'a> {
        pub(super) fn compute(self, dbg: Debug) -> FailResult<Output> { compute(self, dbg) }
    }

    // free function for smaller indent
    fn compute(input: Input<'_>, dbg: Debug) -> FailResult<Output> {
        let _ = dbg;
        let Input { params, type_i, type_j, length } = input;
        let params_ij = params.by_type[type_i][type_j];

        // Bonds of zero weight were trimmed from the interactions, and we're doing nonreactive.
        let weight = 1.0;

        let VA;
        let VA_d_length;
        {
            // UA_ij = - sum_{n in 1..=3} B_n e^{-beta_n r_ij}
            let mut UA = 0.0;
            let mut UA_d_length = 0.0;
            for (&B, &beta) in zip_eq!(&params_ij.B, &params_ij.beta) {
                let term = -B * f64::exp(-beta * length);
                let term_d_length = -beta * term;
                UA += term;
                UA_d_length += term_d_length;
            }

            // VA_ij = f_ij UA_ij
            VA = weight * UA;
            VA_d_length = weight * UA_d_length;
        }

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
            VR_d_length = weight * UR_d_length;
        }

        Ok(Output {
            weight, VA, VA_d_length, VR, VR_d_length,
        })
    }


    impl<'a> Input<'a> {
        // For debugging; feel free to swap out the `compute` call with this.
        #[allow(unused)]
        pub(super) fn compute_paranoid(self, dbg: Debug, tol: f64) -> FailResult<Output> {
            let output = self.clone().compute(dbg)?;
            let Output { weight, VA, VA_d_length, VR, VR_d_length } = output;

            assert_close!(
                rel=tol, abs=tol,
                VA_d_length,
                numerical::try_slope(
                    1e-4, None, self.length,
                    |length| Input { length, ..self }.compute(Debug::Never).map(|x| x.VA),
                )?,
            );
            assert_close!(
                rel=tol, abs=tol,
                VR_d_length,
                numerical::try_slope(
                    1e-4, None, self.length,
                    |length| Input { length, ..self }.compute(Debug::Never).map(|x| x.VR),
                )?,
            );
            Ok(output)
        }
    }
}

use self::site_sigma_pi_term::SiteSigmaPiTerm;
mod site_sigma_pi_term {
    //! Represents the sum of `0.5 * b_{ij}^{sigma-pi} * VR` over all `j` for a given `i`.
    //!
    //! This quantity is useful to consider in its own right because it encapsulates
    //! the need for the P spline values, and it only has derivatives with respect
    //! to the bond vectors of site `i`; these properties give it a fairly simple signature.

    use super::*;

    pub(super) type Output = SiteSigmaPiTerm;

    #[derive(Debug, Clone)]
    pub(super) struct Input<'a> {
        pub params: &'a Params,
        pub interactions: &'a Interactions,
        pub site: SiteI,
        pub bond_deltas: &'a [V3],
        pub bond_weights: &'a [f64],
        // The VA_ij terms for each bond at site i.
        pub bond_VAs: &'a [f64],
    }
    pub(super) struct SiteSigmaPiTerm {
        pub value: f64,
        /// Derivatives with respect to the bonds listed in order of `interactions.bonds(site_i)`.
        pub d_deltas: SiteBondVec<V3>,
        pub d_VAs: SiteBondVec<f64>,
    }

    impl<'a> Input<'a> {
        pub(super) fn compute(self, dbg: Debug) -> Output { compute(self, dbg) }
    }

    // free function for smaller indent
    fn compute(input: Input<'_>, dbg: Debug) -> Output {
        // Eq 8:  b_{ij}^{sigma-pi} = sqrt(
        //                     1 + sum_{k /= i, j} f^c(r_{ik}) G(cos(t_{ijk}) e^{lambda_{ijk}
        //                       + P_{ij}(N_ij^C, N_ij^H)
        //        )
        let Input {
            params, interactions, bond_weights, bond_VAs, bond_deltas,
            site: site_i,
        } = input;
        let type_i = interactions.site_type(site_i);

        // Tally up data about the bonds
        let mut type_present = enum_map!{_ => false};
        let mut ccoord_i = 0.0;
        let mut hcoord_i = 0.0;
        let mut bond_target_types = SiteBondVec::new();
        // (recompute these for a simpler signature and less data management)
        let mut bond_lengths = SiteBondVec::new();
        let mut bond_lengths_d_delta = SiteBondVec::new();
        for (bond, &weight, &delta) in zip_eq!(interactions.bonds(site_i), bond_weights, bond_deltas) {
            let target_type = interactions.site_type(interactions.bond_target(bond));
            match target_type {
                AtomType::Carbon => ccoord_i += weight,
                AtomType::Hydrogen => hcoord_i += weight,
            }
            bond_target_types.push(target_type);
            type_present[target_type] = true;

            let (length, length_d_delta) = norm(delta);
            bond_lengths.push(length);
            bond_lengths_d_delta.push(length_d_delta);
        }

        // Handle all terms
        let mut value = 0.0;
        let mut d_deltas = sbvec_filled(V3::zero(), bond_weights.len());
        let mut d_VAs = sbvec_filled(0.0, bond_weights.len());

        for (sbindex_ij, _) in interactions.bonds(site_i).enumerate() {
            let type_j = bond_target_types[sbindex_ij];
            let weight_ij = bond_weights[sbindex_ij];

            // These are what Brenner's Ni REALLY are.
            let ccoord_ij = ccoord_i - boole(type_j == AtomType::Carbon) * weight_ij;
            let hcoord_ij = hcoord_i - boole(type_j == AtomType::Hydrogen) * weight_ij;

            let P_ij = p_spline::Input { params, type_i, type_j, ccoord_ij, hcoord_ij }.compute();
            dbg!(dbg, "P: {:.9}", P_ij);

            // Gather all cosines between bond i->j and other bonds i->k.
            let BondCosines {
                coses_ijk, coses_ijk_d_delta_ij, coses_ijk_d_delta_ik,
            } = bond_cosines::Input { sbindex_ij, deltas_ik: bond_deltas }.compute();

            // We're finally ready to compute the bond order.
            let bsp_ij;
            let bsp_ij_d_deltas;
            {
                // Compute bsp as a function of many things...
                let out = bondorder_sigma_pi::Input {
                    params, type_i, type_j, ccoord_ij, hcoord_ij, P_ij, sbindex_ij,
                    coses_ijk: &coses_ijk,
                    types_k: &bond_target_types,
                    weights_ik: bond_weights,
                    lengths_ik: &bond_lengths,
//                }.compute_paranoid(dbg, 1e-7); // generous tolerance for G
                }.compute(dbg);
                let BondOrderSigmaPi {
                    value: tmp_value,
                    d_lengths_ik: bsp_ij_d_lengths_ik,
                    d_coses_ijk: bsp_ij_d_coses_ijk,
                } = out;

                // ...and now reformulate away the explicit dependence on the cosines,
                // and lengths, knowing that they each are a function of the deltas.
                let mut tmp_d_deltas: SiteBondVec<V3> = sbvec_filled(V3::zero(), bond_weights.len());

                // Cosines at all indices except sbindex_ij
                for (sbindex_ik, bsp_ij_d_cos_ijk) in bsp_ij_d_coses_ijk.into_iter().enumerate() {
                    // Mind the gap
                    if sbindex_ij == sbindex_ik {
                        continue;
                    }

                    // cos_ijk = cos_ijk(delta_ij, delta_ik)
                    // These are both sbindex_ik because we are indexing the cosines.
                    let cos_ijk_d_delta_ij = coses_ijk_d_delta_ij[sbindex_ik];
                    let cos_ijk_d_delta_ik = coses_ijk_d_delta_ik[sbindex_ik];

                    // These are sbindex_ij and sbindex_ik because we are indexing the deltas.
                    tmp_d_deltas[sbindex_ij] += bsp_ij_d_cos_ijk * cos_ijk_d_delta_ij;
                    tmp_d_deltas[sbindex_ik] += bsp_ij_d_cos_ijk * cos_ijk_d_delta_ik;
                }

                // Lengths at all indices
                for (sbindex_ik, bsp_ij_d_length_ik) in bsp_ij_d_lengths_ik.into_iter().enumerate() {
                    let length_ik_d_delta_ik = bond_lengths_d_delta[sbindex_ik];
                    tmp_d_deltas[sbindex_ik] += bsp_ij_d_length_ik * length_ik_d_delta_ik;
                }

                bsp_ij = tmp_value;
                bsp_ij_d_deltas = tmp_d_deltas;
                dbg!(dbg, "bsp: {:.9}", bsp_ij);
            }

            // True term to add to sum is 0.5 * VA_ij * bsp_ij
            let VA_ij = bond_VAs[sbindex_ij];

            dbg!(dbg, "Vterm(sp): {:.9}", 0.5 * VA_ij * bsp_ij);
            value += 0.5 * VA_ij * bsp_ij;
            d_VAs[sbindex_ij] += 0.5 * bsp_ij;
            axpy_mut(&mut d_deltas, 0.5 * VA_ij, &bsp_ij_d_deltas);
        }
        Output { value, d_deltas, d_VAs }
    }


    impl<'a> Input<'a> {
        // For debugging; feel free to swap out the `compute` call with this.
        #[allow(unused)]
        pub fn compute_paranoid(self, dbg: Debug, tol: f64) -> Output {

            let output = self.clone().compute(dbg);
            let Output { value, ref d_deltas, ref d_VAs } = output;

            assert_close!(
                rel=tol, abs=tol,
                d_deltas.flat(),
                &numerical::gradient(1e-4, None, self.bond_deltas.flat(), |x| {
                    Input { bond_deltas: x.nest(), ..self }.compute(Debug::Never).value
                })[..],
            );

            assert_close!(
                rel=tol, abs=tol,
                &d_VAs[..],
                &numerical::gradient(1e-3, None, self.bond_VAs, |x| {
                    Input { bond_VAs: x, ..self }.compute(Debug::Never).value
                })[..],
            );
            output
        }
    }
}

use self::bondorder_sigma_pi::BondOrderSigmaPi;
mod bondorder_sigma_pi {
    use super::*;

    pub(super) type Output = BondOrderSigmaPi;

    #[derive(Debug, Clone)]
    pub(super) struct Input<'a> {
        pub params: &'a Params,
        // bond from atom i to another atom j
        pub type_i: AtomType,
        pub type_j: AtomType,
        pub sbindex_ij: usize, // SiteBondVec index
        pub ccoord_ij: f64,
        pub hcoord_ij: f64,
        // precomputed spline that depends on the coordination at i and the atom type at j
        pub P_ij: f64,
        // cosines of this bond with every other bond at i, and their weights
        pub types_k: &'a [AtomType],
        pub lengths_ik: &'a [f64],
        pub weights_ik: &'a [f64],
        pub coses_ijk: &'a [f64], // cosine between i->j and i->k
    }

    pub(super) struct BondOrderSigmaPi {
        pub value: f64,
        pub d_coses_ijk: SiteBondVec<f64>, // value at sbindex_ij is unspecified
        pub d_lengths_ik: SiteBondVec<f64>, // this one has values at all indices
    }

    impl<'a> Input<'a> {
        pub(super) fn compute(self, dbg: Debug) -> Output { compute(self, dbg) }
    }

    // free function for smaller indent
    fn compute<'a>(input: Input<'a>, dbg: Debug) -> Output {
        // Eq 8:  b_{ij}^{sigma-pi} = recip(sqrt(
        //                     1 + sum_{k /= i, j} f^c(r_{ik}) G(cos(t_{ijk}) e^{lambda_{ijk}
        //                       + P_{ij}(N_i^C, N_i^H)
        //        ))
        let Input {
            params, type_i, type_j, ccoord_ij, hcoord_ij, P_ij,
            types_k, lengths_ik, weights_ik, coses_ijk, sbindex_ij,
        } = input;
        let tcoord_ij = ccoord_ij + hcoord_ij;
        dbg!(dbg, "Nt: {:.9}", tcoord_ij);

        // properties of the stuff in the square root
        let mut inner_value = 0.0;
        let mut inner_d_coses_ijk = SiteBondVec::new();
        let mut inner_d_lengths_ik = sbvec_filled(0.0, lengths_ik.len());

        // 1 + P_{ij}(N_i^C, N_i^H)
        //   + sum_{k /= i, j} e^{\lambda_{ijk}} f^c(r_{ik}) G(cos(t_{ijk})
        inner_value += 1.0;
        inner_value += P_ij;

        let iter = zip_eq!(coses_ijk, types_k).enumerate();
        for (sbindex_ik, (&cos_ijk, &type_k)) in iter {
            let weight_ik = weights_ik[sbindex_ik];
            if weight_ik == 0.0 {
                inner_d_coses_ijk.push(0.0);
                continue;
            }
            if sbindex_ik == sbindex_ij {
                inner_d_coses_ijk.push(NAN);
                continue;
            }

            let length_ij = lengths_ik[sbindex_ij];
            let length_ik = lengths_ik[sbindex_ik];

            let ExpLambda {
                value: exp,
                d_length_ij: exp_d_length_ij,
                d_length_ik: exp_d_length_ik,
            } = exp_lambda::Input {
                params, type_i, type_j, type_k, length_ij, length_ik,
            }.compute(dbg);
            dbg!(dbg, "explambda: {:.9}", exp);

            let GSpline {
                value: G,
                d_cos_ijk: G_d_cos_ijk,
            } = g_spline::Input { params, type_i, cos_ijk, tcoord_ij }.compute();
            dbg!(dbg, "g: {:.9} {:.9}", G, G_d_cos_ijk);

//            dbg!("bspterm: {:.9}", exp_lambda * weight_ik * G);
            inner_value += exp * weight_ik * G;
            inner_d_coses_ijk.push(exp * weight_ik * G_d_cos_ijk);
            inner_d_lengths_ik[sbindex_ik] += exp_d_length_ik * weight_ik * G;
            inner_d_lengths_ik[sbindex_ij] += exp_d_length_ij * weight_ik * G;
        }

//        dbg!("bspsum: {:.9}", inner_value);

        // Now take the reciprocal square root.
        //
        // (d/dx) 1 / sqrt(f(x))  =  (-1/2) (df/dx) (1/sqrt(f(x)))^3
        let value = f64::sqrt(inner_value).recip();
        let prefactor = -0.5 * value * value * value;
        Output {
            value: value,
            d_coses_ijk: sbvec_scaled(prefactor, inner_d_coses_ijk),
            d_lengths_ik: sbvec_scaled(prefactor, inner_d_lengths_ik),
        }
    }

    impl<'a> Input<'a> {
        // For debugging; feel free to swap out the `compute` call with this.
        //
        // CAUTION: This should use a fairly generous tolerance because numerical derivatives of
        //          the G spline can have fairly big error.
        #[allow(unused)]
        pub(super) fn compute_paranoid(self, dbg: Debug, tol: f64) -> Output {
            let output = self.clone().compute(dbg);
            let Output { value, ref d_coses_ijk, ref d_lengths_ik } = output;

            let mut coses_ijk = self.coses_ijk.to_vec();
            for sbindex_ik in 0..self.coses_ijk.len() {
                if sbindex_ik == self.sbindex_ij {
                    continue;
                }

                assert_close!(
                    rel=tol, abs=tol,
                    d_coses_ijk[sbindex_ik],
                    numerical::slope(1e-4, None, self.coses_ijk[sbindex_ik], |cos_ijk| {
                        // FIXME so dumb
                        let old = self.coses_ijk[sbindex_ik];
                        coses_ijk[sbindex_ik] = cos_ijk;
                        let out = Input { coses_ijk: &coses_ijk, ..self }.compute(Debug::Never).value;
                        coses_ijk[sbindex_ik] = old;
                        out
                    }),
                );
            }

            assert_close!(
                rel=tol, abs=tol,
                &d_lengths_ik[..],
                &numerical::gradient(1e-4, None, self.lengths_ik, |lengths_ik| {
                    Input { lengths_ik, ..self }.compute(Debug::Never).value
                })[..],
            );
            output
        }
    }
}

// b_{ij}^{pi} in Brenner
use self::bondorder_pi::BondOrderPi;
mod bondorder_pi {
    use super::*;

    pub(super) type Output = BondOrderPi;

    #[derive(Debug, Clone)]
    pub(super) struct Input<'a> {
        pub params: &'a Params,
        pub interactions: &'a Interactions,
        pub bond_ij: BondI,
        // info about all bonds ik connected to site i
        // and all bonds jl connected to site j
        pub tcoords_k: &'a [f64], // tcoords of sites connected to i
        pub tcoords_l: &'a [f64], // tcoords of sites connected to j
        pub deltas_ik: &'a [V3],
        pub deltas_jl: &'a [V3],
        pub weights_ik: &'a [f64],
        pub weights_jl: &'a [f64],
        // weights that use an alternate interval in AIREBO (defined by `cutoff_max_2`)
        pub alt_weights_ik: &'a [f64],
        pub alt_weights_jl: &'a [f64],
    }
    pub(super) struct BondOrderPi {
        pub value: f64,
        pub d_deltas_ik: SiteBondVec<V3>,
        pub d_deltas_jl: SiteBondVec<V3>,
    }

    impl<'a> Input<'a> {
        pub(super) fn compute(self, dbg: Debug) -> Output { compute(self, dbg) }
    }

    // free function for smaller indent
    fn compute(input: Input<'_>, dbg: Debug) -> Output {
        let Input {
            params, interactions, bond_ij,
            tcoords_k, weights_ik, alt_weights_ik, deltas_ik,
            tcoords_l, weights_jl, alt_weights_jl, deltas_jl,
        } = input;

        // Gather all sorts of boring info about the bond from the bond index.
        let site_i = interactions.bond_source(bond_ij);
        let site_j = interactions.bond_target(bond_ij);
        let type_i = interactions.site_type(site_i);
        let type_j = interactions.site_type(site_j);
        let bond_ji = interactions.bond_reverse_index(bond_ij);

        let gather_target_tcoords = |site| -> SiteBondVec<_> {
            interactions.bond_targets(site)
                .map(|target| interactions.site_type(target))
                .collect()
        };
        let types_k: SiteBondVec<_> = gather_target_tcoords(site_i);
        let types_l: SiteBondVec<_> = gather_target_tcoords(site_j);

        let sbindex_ij = interactions.bond_sbvec_index(bond_ij);
        let sbindex_ji = interactions.bond_sbvec_index(bond_ji);

        //----------------
        // Both pieces of this term depend on NConj (which we call the "x coordination number").
        // It is an argument to the splines.
        //
        //     NConj = 1 + (square sum over bonds ik) + (square sum over bonds jl)
        //
        // or in our nomenclature
        //
        //     xcoord_ij = 1 + ycoord_ij + ycoord_ji
        //
        let xcoord_ij;
        {
            let ycoord_ij = ycoord::Input {
                skip_index: sbindex_ij,
                weights_ik: weights_ik,
                tcoords_k: tcoords_k,
                types_k: &types_k,
            }.compute(dbg);

            let ycoord_ji = ycoord::Input {
                skip_index: sbindex_ji,
                weights_ik: weights_jl,
                tcoords_k: tcoords_l,
                types_k: &types_l,
            }.compute(dbg);

            xcoord_ij = 1.0 + ycoord_ij + ycoord_ji;
        }

        //----------------
        // The splines also depend on N_{ij} and N_{ji}. (tcoord_ij, tcoord_ji)

        let (tcoord_ij, tcoord_ji);
        {
            let weight_ij = weights_ik[sbindex_ij];
            let weight_ji = weights_jl[sbindex_ji];

            // (these are not flipped; tcoords_k and tcoords_l describe the tcoords of
            //  the *target* atoms, so tcoord_i is in tcoords_l and etc.)
            let tcoord_i = tcoords_l[sbindex_ji];
            let tcoord_j = tcoords_k[sbindex_ij];

            tcoord_ij = tcoord_i - weight_ij;
            tcoord_ji = tcoord_j - weight_ji;
        }

        //----------------
        // Accumulators for the output value.
        // (we'll be adding two terms to this)
        let mut value = 0.0;
        let mut d_deltas_ik = sbvec_filled(V3::zero(), types_k.len());
        let mut d_deltas_jl = sbvec_filled(V3::zero(), types_l.len());

        // First term has a T_ij prefactor, which is frequently zero.
        // (it's only nonzero for CC bonds between two carbons of coordination number 3)
        let T = t_spline::Input { params, type_i, type_j, tcoord_ij, tcoord_ji, xcoord_ij }.compute();
        dbg!(dbg, "T: {:.9}", T);

        if T != 0.0 {

            // NOTE: The AIREBO paper introduces cutoffs here:
            //
            //        sinsq_ijkl * alt_weight_ik * alt_weight_jl
            //                   * H(|sin_ijk| - s_min)
            //                   * H(|sin_jil| - s_min)    (s_min = 0.1)
            //
            // where H is the Heaviside function. The absolute value bars (not present in the paper)
            // were added based on my personal interpretation of the intent.
            //
            // This is to avoid an ill-defined region of sinsq where a bond angle is 0 or 180
            // degrees. The latter is a condition that actually can occur in carbon chains, and the
            // former means that we can skip self-angles (e.g. where k = j) without an additional
            // test.
            //
            // Of course, that Heaviside is a pretty harsh cutoff, and we prefer to use something
            // with C1 continuity. LAMMPS does something similar (thmin and thmax), but only around
            // 180 degrees.
            // (NOTE: for some reason, LAMMPS also has a hard cutoff at abs(sin) < 1e-9.  This seems
            //        pointless; we just use a thmax that is smaller than 1, and apply the smooth
            //        cutoff around 0 degrees as well)
            const T_COS_0: f64 = 1.0 - 1e-11; // = -thmax
            const T_COS_1: f64 = 0.995;       // = -thmin

            // Because every ijk angle gets paired with every jil angle, we precompute all of these
            // cutoff factors right now.

            let BondCosines {
                coses_ijk,
                coses_ijk_d_delta_ij,
                coses_ijk_d_delta_ik,
            } = bond_cosines::Input {
                deltas_ik,
                sbindex_ij,
            }.compute();

            let BondCosines {
                coses_ijk: coses_jil,
                coses_ijk_d_delta_ij: coses_jil_d_delta_ji,
                coses_ijk_d_delta_ik: coses_jil_d_delta_jl,
            } = bond_cosines::Input {
                deltas_ik: deltas_jl,
                sbindex_ij: sbindex_ji,
            }.compute();

            let get_cutoffs = |site_i, sbindex_ij, coses_ijk: &[f64]| {
                let mut cutoffs_ijk = SiteBondVec::new();
                let mut cutoffs_ijk_d_cos_ijk = SiteBondVec::new();
                for (sbindex_ik, _) in interactions.bonds(site_i).enumerate() {
                    if sbindex_ik == sbindex_ij {
                        // A cutoff of zero ensures this is skipped later.
                        cutoffs_ijk.push(0.0);
                        cutoffs_ijk_d_cos_ijk.push(NAN); // bomb; should never be used
                    } else {
                        // (we could do a single switch on abs(cos_ijk) here but then we'd
                        //  also need to potentially fix the sign of the derivative)
                        let (pcut, pcut_d_cos, _) = switch::poly3((T_COS_0, T_COS_1), coses_ijk[sbindex_ik]);
                        let (mcut, mcut_d_cos, _) = switch::poly3((-T_COS_0, -T_COS_1), coses_ijk[sbindex_ik]);

                        let cut = pcut * mcut;
                        let cut_d_cos = pcut * mcut_d_cos + pcut_d_cos * mcut;
                        dbg!(dbg, "sincut: {:.9}", cut);

                        if cut != 1.0 {
                            // Basically the only test for this is the unphysical structure
                            // "carbon-chain-4-hydro" in the tests directory.
                            warn_once!(
                                "REBO: Found a bond angle near 0 or 180 degrees that has a nonzero \
                                T coefficient. This is a rare circumstance with a nontrivial \
                                implementation that may still have bugs."
                            );
                        }

                        cutoffs_ijk.push(cut);
                        cutoffs_ijk_d_cos_ijk.push(cut_d_cos);
                    }
                }
                (cutoffs_ijk, cutoffs_ijk_d_cos_ijk)
            };
            let (cutoffs_ijk, cutoffs_ijk_d_cos_ijk) = get_cutoffs(site_i, sbindex_ij, &coses_ijk);
            let (cutoffs_jil, cutoffs_jil_d_cos_jil) = get_cutoffs(site_j, sbindex_ji, &coses_jil);

            // Now for the actual sum:
            //
            //     sum = ∑_{k,l} sin^2(Θ_{ijkl}) f_{ik} f_{jl} cut_{ijk} cut_{jil}
            //
            // We must iterate over groups of four atoms ijkl, where k != i and l != j.
            // For now it is less work if we consider the cosines to be independent variables.
            let mut sum = 0.0;
            let mut sum_d_deltas_ik = sbvec_filled(V3::zero(), weights_ik.len()); // w.r.t. bonds around site i
            let mut sum_d_deltas_jl = sbvec_filled(V3::zero(), weights_jl.len()); // w.r.t. bonds around site j
            let mut sum_d_coses_ijk = sbvec_filled(0.0, weights_ik.len()); // w.r.t. bonds around site i
            let mut sum_d_coses_jil = sbvec_filled(0.0, weights_jl.len()); // w.r.t. bonds around site j

            for sbindex_ik in 0..types_k.len() {
                if cutoffs_ijk[sbindex_ik] == 0.0 {
                    continue; // region where sinsq_ijkl is ill-defined
                }

                for sbindex_jl in 0..types_l.len() {
                    if cutoffs_jil[sbindex_jl] == 0.0 {
                        continue; // region where sinsq_ijkl is ill-defined
                    }

                    let delta_ij = deltas_ik[sbindex_ij];
                    let delta_ik = deltas_ik[sbindex_ik];
                    let delta_jl = deltas_jl[sbindex_jl];
                    let DihedralSineSq {
                        value: sinsq,
                        d_delta_ij: sinsq_d_delta_ij,
                        d_delta_ik: sinsq_d_delta_ik,
                        d_delta_jl: sinsq_d_delta_jl,
                    } = dihedral_sine_sq::Input { delta_ij, delta_ik, delta_jl }.compute();
                    debug_assert_eq!(
                        sinsq, sinsq,
                        "sinsq_ijkl = NaN for vectors:\nij: {:?}\nik: {:?}\njl: {:?}",
                        delta_ij, delta_ik, delta_jl,
                    );

                    // Other terms in the product
                    let alt_weight_ik = alt_weights_ik[sbindex_ik];
                    let alt_weight_jl = alt_weights_jl[sbindex_jl];
                    let cutoff_ijk = cutoffs_ijk[sbindex_ik];
                    let cutoff_jil = cutoffs_jil[sbindex_jl];
                    let cutoff_ijk_d_cos_ijk = cutoffs_ijk_d_cos_ijk[sbindex_ik];
                    let cutoff_jil_d_cos_jil = cutoffs_jil_d_cos_jil[sbindex_jl];

                    // Postfactor of sinsq_ijkl
                    let post = alt_weight_ik * alt_weight_jl * cutoff_ijk * cutoff_jil;
                    let post_d_cos_ijk = alt_weight_ik * alt_weight_jl * cutoff_ijk_d_cos_ijk * cutoff_jil;
                    let post_d_cos_jil = alt_weight_ik * alt_weight_jl * cutoff_ijk * cutoff_jil_d_cos_jil;

                    sum += sinsq * post;

                    sum_d_deltas_ik[sbindex_ij] += sinsq_d_delta_ij * post;
                    sum_d_deltas_ik[sbindex_ik] += sinsq_d_delta_ik * post;
                    sum_d_deltas_jl[sbindex_jl] += sinsq_d_delta_jl * post;

                    sum_d_coses_ijk[sbindex_ik] += sinsq * post_d_cos_ijk;
                    sum_d_coses_jil[sbindex_jl] += sinsq * post_d_cos_jil;
                } // for bond_jl
            } // for bond_ik

            // Now add to the outer accumulators.
            // This is a convenient time to eliminate the cosines from our independent variables.
            value += T * sum;
            axpy_mut(&mut d_deltas_ik, T, &sum_d_deltas_ik);
            axpy_mut(&mut d_deltas_jl, T, &sum_d_deltas_jl);

            for (sbindex_ik, &sum_d_cos_ijk) in sum_d_coses_ijk.iter().enumerate() {
                if sbindex_ik != sbindex_ij {
                    d_deltas_ik[sbindex_ik] += T * sum_d_cos_ijk * coses_ijk_d_delta_ik[sbindex_ik];
                    d_deltas_ik[sbindex_ij] += T * sum_d_cos_ijk * coses_ijk_d_delta_ij[sbindex_ik];
                }
            }
            for (sbindex_jl, &sum_d_cos_jil) in sum_d_coses_jil.iter().enumerate() {
                if sbindex_jl != sbindex_ji {
                    d_deltas_jl[sbindex_jl] += T * sum_d_cos_jil * coses_jil_d_delta_jl[sbindex_jl];
                    d_deltas_jl[sbindex_ji] += T * sum_d_cos_jil * coses_jil_d_delta_ji[sbindex_jl];
                }
            }
        }

        // Second term: Just F.
        let F = f_spline::Input {
            params, type_i, type_j, tcoord_ij, tcoord_ji, xcoord_ij,
        }.compute();
        value += F;

        dbg!(dbg, "F: {:.9}", F);

        Output { value, d_deltas_ik, d_deltas_jl }
    }

    impl<'a> Input<'a> {
        // For debugging; feel free to swap out the `compute` call with this.
        #[allow(unused)]
        pub(super) fn compute_paranoid(self, dbg: Debug, tol: f64) -> Output {
            let output = self.clone().compute(dbg);
            let Output { value, ref d_deltas_ik, ref d_deltas_jl } = output;

            assert_close!(
                rel=tol, abs=tol,
                d_deltas_ik.flat(),
                &numerical::gradient(1e-4, None, self.deltas_ik.flat(), |x| {
                    Input { deltas_ik: x.nest(), ..self }.compute(Debug::Never).value
                })[..],
            );

            assert_close!(
                rel=tol, abs=tol,
                d_deltas_jl.flat(),
                &numerical::gradient(1e-4, None, self.deltas_jl.flat(), |x| {
                    Input { deltas_jl: x.nest(), ..self }.compute(Debug::Never).value
                })[..],
            );
            output
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
// derivatives (in the reactive form, at least) and amounts to little more than two computations
// of `ycoord`.
mod ycoord {
    use super::*;

    pub(super) type Output = f64;
    pub(super) struct Input<'a> {
        pub skip_index: usize,
        pub weights_ik: &'a [f64],
        pub tcoords_k: &'a [f64],
        pub types_k: &'a [AtomType],
    }

    impl<'a> Input<'a> {
        pub(super) fn compute(self, dbg: Debug) -> Output { compute(self, dbg) }
    }

    // free function for smaller indent
    fn compute(input: Input<'_>, dbg: Debug) -> Output {
        let _ = dbg;

        let Input { skip_index, weights_ik, tcoords_k, types_k } = input;

        // Compute the sum without the square
        let mut inner_value = 0.0;
        let iter = zip_eq!(tcoords_k, weights_ik, types_k).enumerate();
        for (sbindex_ik, (&tcoord_k, &weight_ik, &type_k)) in iter {
            if sbindex_ik == skip_index || type_k == AtomType::Hydrogen {
                continue;
            }
            let tcoord_ki = tcoord_k - weight_ik;

            let (alpha, alpha_d_tcoord_ki, _) = switch::poly5((3.0, 2.0), tcoord_ki);
            assert_eq!(alpha.fract(), 0.0);
            assert_eq!(alpha_d_tcoord_ki, 0.0);
            inner_value += weight_ik * alpha;
        }

        inner_value * inner_value
    }
}

use self::exp_lambda::ExpLambda;
mod exp_lambda {
    use super::*;

    pub(super) type Output = ExpLambda;
    pub(super) struct Input<'a> {
        pub params: &'a Params,
        pub type_i: AtomType,
        pub type_j: AtomType,
        pub type_k: AtomType,
        pub length_ij: f64,
        pub length_ik: f64,
    }

    pub(super) struct ExpLambda {
        pub value: f64,
        pub d_length_ij: f64,
        pub d_length_ik: f64,
    }

    impl Output {
        fn constant(value: f64) -> Self {
            Output { value, d_length_ij: 0.0, d_length_ik: 0.0 }
        }
    }

    impl<'a> Input<'a> {
        pub(super) fn compute(self, dbg: Debug) -> Output { compute(self, dbg) }
    }

    fn compute(input: Input<'_>, dbg: Debug) -> Output {
        let _ = dbg;

        let Input { params, type_i, type_j, type_k, length_ij, length_ik } = input;
        match params.use_airebo_lambda {
            // `exp(lambda_ijk)` as defined in the 2nd-gen REBO paper.
            false => {
                match (type_i, (type_j, type_k)) {
                    (AtomType::Carbon, _) |
                    (AtomType::Hydrogen, (AtomType::Carbon, AtomType::Carbon)) => Output::constant(1.0),

                    (AtomType::Hydrogen, (AtomType::Hydrogen, AtomType::Hydrogen)) => Output::constant(f64::exp(4.0)),

                    (AtomType::Hydrogen, (AtomType::Carbon, AtomType::Hydrogen)) |
                    (AtomType::Hydrogen, (AtomType::Hydrogen, AtomType::Carbon)) => {
                        // FIXME: The brenner paper says they fit this, but I can't find the value anywhere.
                        panic!{"\
                            Bond-bond interactions of type HHC (an H and a C both bonded to an H) are \
                            currently missing an interaction parameter\
                        "}
                    },
                }
            },

            // The highly suspicious definition of lambda in the AIREBO paper.
            //
            // LAMMPS faithfully implements exactly this for both REBO and AIREBO.
            //
            // You may notice that the definition is not dimensionally sound;
            // it computes the exponential of a quantity with units of length.
            // Presumably this quantity must be in Angstroms.
            //
            // TODO: Figure out what on Earth the deal with this is...
            true => {
                // delta(i, H) * 4 * [ kronecker(k, C) * rho_CH + kronecker(k, H) * rho_HH
                //                   - kronecker(j, C) * rho_CH - kronecker(j, H) * rho_HH
                //                   + r_ij - r_ik
                //                   ]
                match type_i {
                    AtomType::Carbon => Output::constant(1.0),
                    AtomType::Hydrogen => {
                        let mut arg = 0.0;
                        arg += params.by_type[type_k][AtomType::Hydrogen].airebo_rho;
                        arg -= params.by_type[type_j][AtomType::Hydrogen].airebo_rho;
                        arg += length_ij;
                        arg -= length_ik;

                        let value = f64::exp(4.0 * arg);
                        let d_length_ij = 4.0 * value;
                        let d_length_ik = -4.0 * value;
                        Output { value, d_length_ij, d_length_ik }
                    },
                }
            },
        }
    }
}

// `1 - \cos(\Theta_{ijkl})^2` in Brenner (equation 19)
use self::dihedral_sine_sq::DihedralSineSq;
mod dihedral_sine_sq {
    //! Diff function for the `1 - cos(Θ_{ijkl})^2` factor
    //! describing interactions of 4 atoms.
    use super::*;

    pub(super) type Output = DihedralSineSq;
    pub(super) struct Input {
        pub delta_ij: V3,
        pub delta_ik: V3,
        pub delta_jl: V3,
    }

    pub(super) struct DihedralSineSq {
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

    // shows there is no need to check l != k because it is well-defined in that case
    #[test]
    fn l_equal_k() {
        for _ in 0..10 {
            let a = V3::from_fn(|_| uniform(-10.0, 10.0));
            let b = V3::from_fn(|_| uniform(-10.0, 10.0));

            let tests = vec![
                (a, a + b, b),
                (a - b, a, b),
            ];
            for (delta_ij, delta_ik, delta_jl) in tests {
                let Output {
                    value, d_delta_ij, d_delta_ik, d_delta_jl,
                } = Input { delta_ij, delta_ik, delta_jl }.compute();
                assert_close!(abs=1e-9, 0.0, value);
                assert_close!(abs=1e-9, V3::zero().0, d_delta_ij.0);
                assert_close!(abs=1e-9, V3::zero().0, d_delta_ik.0);
                assert_close!(abs=1e-9, V3::zero().0, d_delta_jl.0);
            }
        }
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

use self::bond_cosine::{BondCosine};
mod bond_cosine {
    //! Diff function for the cos(θ) between bonds.
    use super::*;

    pub(super) type Output = BondCosine;
    pub(super) struct Input {
        pub delta_ij: V3,
        pub delta_ik: V3,
    }

    pub(super) struct BondCosine {
        pub value: f64,
        pub d_delta_ij: V3,
        pub d_delta_ik: V3,
    }

    impl Input {
        pub(super) fn compute(self) -> Output { compute(self) }
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

use self::bond_cosines::{BondCosines};
mod bond_cosines {
    //! For a given bond, get all cosines with other bonds at the originating site.
    //!
    //! This is mostly a trivial wrapper around `BondCosine`, but the output SiteBondVecs will
    //! contain NaNs in the location that correspond to the current bond (as a bond has no cosine
    //! with itself).
    use super::*;

    pub(super) type Output = BondCosines;
    pub(super) struct Input<'a> {
        pub deltas_ik: &'a [V3],
        pub sbindex_ij: usize,
    }

    pub(super) struct BondCosines {
        pub coses_ijk: SiteBondVec<f64>,
        pub coses_ijk_d_delta_ij: SiteBondVec<V3>,
        pub coses_ijk_d_delta_ik: SiteBondVec<V3>,
    }

    impl<'a> Input<'a> {
        pub(super) fn compute(self) -> Output { compute(self) }
    }

    // free function for smaller indent
    fn compute(input: Input) -> Output {
        let Input { deltas_ik, sbindex_ij } = input;

        let delta_ij = deltas_ik[sbindex_ij];

        let mut coses_ijk = SiteBondVec::new();
        let mut coses_ijk_d_delta_ij = SiteBondVec::new();
        let mut coses_ijk_d_delta_ik = SiteBondVec::new();
        for sbindex_ik in 0..deltas_ik.len() {
            let delta_ik = deltas_ik[sbindex_ik];
            if sbindex_ij == sbindex_ik {
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

        BondCosines { coses_ijk, coses_ijk_d_delta_ij, coses_ijk_d_delta_ik }
    }
}

//-----------------------------------------------------------------------------
// Splines

use self::g_spline::GSpline;
mod g_spline {
    use super::*;

    pub(super) type Output = GSpline;
    #[derive(Clone)]
    pub(super) struct Input<'a> {
        pub params: &'a Params,
        pub type_i: AtomType,
        pub tcoord_ij: f64,
        pub cos_ijk: f64,
    }

    pub(super) struct GSpline {
        pub value: f64,
        pub d_cos_ijk: f64,
    }

    impl<'a> Input<'a> {
        pub(super) fn compute(self) -> Output { compute(self) }
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
                let switch_interval = (params.G.low_coord, params.G.high_coord);
                let (alpha, alpha_d_tcoord_ij, _) = switch::poly5(switch_interval, tcoord_ij);

                if alpha == 0.0 && alpha_d_tcoord_ij == 0.0 {
                    use_single_poly!(&params.G.carbon_low_coord)
                } else if alpha == 1.0 && alpha_d_tcoord_ij == 0.0 {
                    warn!("untested codepath: 37236e5f-9810-4ee5-a8c3-0a5150d9bd26");
                    use_single_poly!(&params.G.carbon_high_coord)
                } else {
                    unreachable!("impossible condition found for non-reactive REBO");
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
        let brenner = Params::new_brenner();
        let lammps = Params::new_lammps();

        fn try_above_and_below(input: Input<'_>) -> impl Iterator<Item=GSpline> {
            let mut above = input.clone();
            let mut below = input.clone();
            above.cos_ijk += 1e-12;
            below.cos_ijk -= 1e-12;
            assert_ne!(above.cos_ijk, input.cos_ijk);
            assert_ne!(below.cos_ijk, input.cos_ijk);
            vec![above.compute(), below.compute()].into_iter()
        }

        // graphite
        let mut input = Input {
            params: &brenner,
            type_i: AtomType::Carbon,
            cos_ijk: f64::cos(120.0 * PI / 180.0),
            tcoord_ij: 2.0,
        };
        for GSpline { value, d_cos_ijk } in try_above_and_below(input.clone()) {
            // Brenner Table 3
            assert_close!(rel=1e-10, value, 0.05280);
            assert_close!(rel=1e-10, d_cos_ijk, 0.17000);
        }

        input.params = &lammps;
        for GSpline { value, d_cos_ijk } in try_above_and_below(input) {
            // Stuart Table VII
            assert_close!(rel=1e-10, value, 0.052_804);
            assert_close!(rel=1e-10, d_cos_ijk, 0.170_000);
        }

        // diamond
        let mut input = Input {
            params: &brenner,
            type_i: AtomType::Carbon,
            cos_ijk: -1.0/3.0,
            tcoord_ij: 3.0,
        };
        for GSpline { value, d_cos_ijk } in try_above_and_below(input.clone()) {
            // Brenner table 3
            assert_close!(rel=1e-10, value, 0.09733);
            assert_close!(rel=1e-10, d_cos_ijk, 0.40000);
        }

        input.params = &lammps;
        for GSpline { value, d_cos_ijk } in try_above_and_below(input) {
            // Stuart Table VII
            assert_close!(rel=1e-10, value, 0.097_321);
            assert_close!(rel=1e-10, d_cos_ijk, 0.400_000);
        }
    }

    #[test]
    fn numerical_derivatives() {
        let all_params = vec![
            Params::new_brenner(),
            Params::new_lammps(),
        ];
        for ref params in all_params {
            for type_i in AtomType::iter_all() {
                let x_divs = match type_i {
                    AtomType::Carbon => params.G.carbon_low_coord.x_div,
                    AtomType::Hydrogen => params.G.hydrogen.x_div,
                };

                let mut coses = vec![];

                let lo_xs = &x_divs[..x_divs.len()-1];
                let hi_xs = &x_divs[1..];
                let mid_xs = &x_divs[1..x_divs.len()-1];

                // points within a region
                // points straddling two regions (with a tiny shift to get the derivative
                coses.extend(zip_eq!(lo_xs, hi_xs).map(|(&a, &b)| 0.5 * (a + b)));
                //  from either of the two regions)
                coses.extend(mid_xs.iter().map(|&x| x - 1e-11));
                coses.extend(mid_xs.iter().map(|&x| x + 1e-11));

                let tcoords = vec![
                    3.0,
                    4.0,
                    // FIXME: points around 3.2 and 3.7 once the interpolation is implemented
                ];
                for &cos_ijk in &coses {
                    for &tcoord_ij in &tcoords {
                        println!("{} {}", cos_ijk, tcoord_ij);
                        let input = Input { params, type_i, cos_ijk, tcoord_ij };

                        // NOTE: Numerical differentation of this function near a region boundary
                        //       is particularly sensitive to step size, due to inherent errors
                        //       explained in the next unit test below.
                        let GSpline { value: _, d_cos_ijk } = input.compute();
                        let num_diff_with_step = |step| {
                            numerical::slope(
                                step, Some(DerivativeKind::Stencil(5)),
                                cos_ijk,
                                |cos_ijk| Input { params, type_i, cos_ijk, tcoord_ij }.compute().value,
                            )
                        };

                        // (these are the best tolerances we can manage based on step size)
                        assert_close!(rel=1e-7, abs=1e-7, d_cos_ijk, num_diff_with_step(1e-3));
                        assert_close!(rel=1e-9, abs=1e-9, d_cos_ijk, num_diff_with_step(1e-4));
                    }
                }
            }
        }
    }

    //----------------------------------------------------------------------------------------------
    //
    //           ERROR INHERENT TO NUMERICAL DIFFERENTIATION OF G(cos_ijk) NEAR A KNOT
    //
    //
    // Consider the method used to derive the finite-difference form of f'(x) using a 5-point
    // stencil.  Typically, one begins by using the Taylor expansion of f around x to approximate
    // f(x) at neighboring points:
    //
    //  f(x-2h) = f(x)  - 2h f'(x)  +   2  h² f"(x)  - (4/3) h³ f‴(x)  +  (2/3) h⁴ f""(x)  + O(h⁵)
    //  f(x-1h) = f(x)  -  h f'(x)  + (1/2)h² f"(x)  - (1/6) h³ f‴(x)  + (1/24) h⁴ f""(x)  + O(h⁵)
    //  f(x+1h) = f(x)  +  h f'(x)  + (1/2)h² f"(x)  + (1/6) h³ f‴(x)  + (1/24) h⁴ f""(x)  + O(h⁵)
    //  f(x+2h) = f(x)  + 2h f'(x)  +   2  h² f"(x)  + (4/3) h³ f‴(x)  +  (2/3) h⁴ f""(x)  + O(h⁵)
    //
    // And by taking linear combinations of equations, one can cause all h², h³, and h⁴ terms to
    // cancel out, to eventually obtain:
    //
    //     8[f(x+2h) - f(x-2h)] - [f(x+h) - f(x-h)]  =  12 h f'(x)  + O(h⁵)
    //
    // However, suppose that we use this to perform numerical differentiation at one of the knots
    // of G(cos_ijk).  Because G is not analytic at this point, points above and below x must use
    // different Taylor expansions!  Luckily, G does have C2 continuity since all of our fitting
    // parameters were chosen to match the value, first, and second derivatives at the knots. This
    // means that f‴(x) is the first derivative that will be different between the two Taylor
    // expansions.
    //
    // Here are the equations adjusted to take that into account: (the only change is that f‴(x)
    // and f""(x) have been replaced by subscripted quantities f‴₊(x), f‴₋(x), and etc.)
    //
    //  f(x-2h) = f(x) - 2h f'(x)  +   2  h² f"(x)  - (4/3) h³ f‴₋(x)  +  (2/3) h⁴ f""₋(x)  + O(h⁵)
    //  f(x-1h) = f(x) -  h f'(x)  + (1/2)h² f"(x)  - (1/6) h³ f‴₋(x)  + (1/24) h⁴ f""₋(x)  + O(h⁵)
    //  f(x+1h) = f(x) +  h f'(x)  + (1/2)h² f"(x)  + (1/6) h³ f‴₊(x)  + (1/24) h⁴ f""₊(x)  + O(h⁵)
    //  f(x+2h) = f(x) + 2h f'(x)  +   2  h² f"(x)  + (4/3) h³ f‴₊(x)  +  (2/3) h⁴ f""₊(x)  + O(h⁵)
    //
    // If you try to apply the same linear combination that solved the problem earlier, you'll find
    // that the third derivatives *do* still cancel out, but the fourth derivatives do not.
    //
    //     8[f(x+2h) - f(x-2h)] - [f(x+h) - f(x-h)]  = 12 h f'(x)
    //                                                - (1/3) h⁴ [ f""₊(x) - f""₋(x) ]
    //                                                + O(h⁵)
    //
    // or in other words,
    //
    //     f'(x) = (8[f(x+2h) - f(x-2h)] - [f(x+h) - f(x-h)])/12h  <--- naive 5-point stencil value
    //             + (1/36) h³ [ f""₊(x) - f""₋(x) ]               <--- correction
    //             + O(h⁴)
    //
    // For a step size of 1e-3, this correction factor can take on values of over 1e-8, which is
    // why numerical derivatives of G performed naively should use a small step size.
    //
    //----------------------------------------------------------------------

    // A test which demonstrates that this does indeed account for the terrible quality observed
    // in the numerical derivatives of G.
    #[test]
    fn numerical_derivative_inherent_error() {
        let all_params = vec![
            Params::new_brenner(),
            Params::new_lammps(),
        ];

        for ref params in all_params {
            let all_splines = vec![
                &params.G.carbon_low_coord,
                &params.G.carbon_high_coord,
                &params.G.hydrogen,
            ];

            for spline in all_splines {
                // Try each x value that is sitting right in-between two polynomial regions.
                for i in 1..spline.poly.len() {
                    let cos_ijk = spline.x_div[i];

                    // First and 4th derivatives
                    let left_d1 = spline.poly[i-1].evaluate(cos_ijk).1;
                    let right_d1 = spline.poly[i].evaluate(cos_ijk).1;
                    let left_d4 = spline.poly[i-1].nth_derivative(4).evaluate(cos_ijk).0;
                    let right_d4 = spline.poly[i].nth_derivative(4).evaluate(cos_ijk).0;

                    let step = 1e-3_f64;

                    // (derivatives are close enough that it shouldn't matter which is used as
                    //  the expected value)
                    assert_close!(rel=1e-12, abs=1e-12, left_d1, right_d1);
                    let expected_d_cos_ijk = left_d1;

                    let num_deriv = numerical::slope(
                        step, Some(DerivativeKind::Stencil(5)), cos_ijk, |x| spline.evaluate(x).0,
                    );
                    let correction = (1.0/36.0) * step.powi(3) * (right_d4 - left_d4);

                    // With the correction in tow, we can use much harsher tolerances here than
                    // were used in the numerical_derivative test.
                    assert_close!(
                        rel=1e-10, abs=1e-10,
                        expected_d_cos_ijk,
                        num_deriv + correction,
                    );
                }
            }
        }
    }

    #[test]
    fn continuity() {
        let iter = vec![
            Params::new_brenner(),
            Params::new_lammps(),
        ];
        for ref params in iter {
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
                    assert_close!(rel=1e-13, poly_a.evaluate(x).0, poly_b.evaluate(x).0);
                    assert_close!(rel=1e-13, poly_da.evaluate(x).0, poly_db.evaluate(x).0);
                    assert_close!(rel=1e-13, poly_dda.evaluate(x).0, poly_ddb.evaluate(x).0);
                }
            }
        }
    }
}

mod p_spline {
    use super::*;

    use self::splines::bicubic;

    type Output = f64;
    pub(super) struct Input<'a> {
        pub params: &'a Params,
        pub type_i: AtomType,
        pub type_j: AtomType,
        pub ccoord_ij: f64,
        pub hcoord_ij: f64,
    }

    impl<'a> Input<'a> {
        pub(super) fn compute(&self) -> Output {
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
}

mod f_spline {
    use super::*;

    pub(super) type Output = f64;
    pub(super) struct Input<'a> {
        pub params: &'a Params,
        pub type_i: AtomType,
        pub type_j: AtomType,
        pub tcoord_ij: f64,
        pub tcoord_ji: f64,
        pub xcoord_ij: f64,
    }

    impl<'a> Input<'a> {
        pub(super) fn compute(self) -> Output { compute(self) }
    }

    // Tables 4, 6, and 9
    fn compute(input: Input<'_>) -> Output {
        let Input { params, type_i, type_j, tcoord_ij, tcoord_ji, xcoord_ij } = input;

        let poly = match (type_i, type_j) {
            (AtomType::Carbon, AtomType::Carbon) => &params.F.CC,
            (AtomType::Hydrogen, AtomType::Hydrogen) => &params.F.HH,

            // NOTE: The fact that F_CH(j,i,k) = F_CH(i,j,k) (see note in `splines::F::brenner_CH`)
            //       implies that F_HC(i,j,k) = F_CH(i,j,k).
            (AtomType::Carbon, AtomType::Hydrogen) |
            (AtomType::Hydrogen, AtomType::Carbon) => &params.F.CH,
        };
        // Ignore grad because total derivative of each variable is locally zero in
        // the non-reactive case.
        let (value, _grad) = poly.evaluate(V3([tcoord_ij, tcoord_ji, xcoord_ij]));

        value
    }
}

mod t_spline {
    //! T spline
    //!
    //! * Brenner, Table 5
    //! * Stuart, Table X

    use super::*;

    pub(super) type Output = f64;
    pub(super) struct Input<'a> {
        pub params: &'a Params,
        pub type_i: AtomType,
        pub type_j: AtomType,
        pub tcoord_ij: f64,
        pub tcoord_ji: f64,
        pub xcoord_ij: f64,
    }

    impl<'a> Input<'a> {
        pub(super) fn compute(self) -> Output { compute(self) }
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

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct IsInteracting(bool);

fn check_distance(
    params: &Params,
    distance: f64,
    types: (AtomType, AtomType),
) -> FailResult<IsInteracting>
{
    let params_ij = params.by_type[types.0][types.1];
    match distance {
        r if r > params_ij.cutoff_region.1 => Ok(IsInteracting(false)),
        r if r > params_ij.forbidden_region.1 => {
            reactive_warnings::log_nonbonded(params_ij.cutoff_region, r);
            Ok(IsInteracting(false))
        },
        r if r >= params_ij.forbidden_region.0 => {
            bail!{"detected reaction in non-reactive REBO potential (r = {})", r};
            unreachable!();
        },
        r if r >= params_ij.cutoff_region.0 => {
            reactive_warnings::log_bonded(params_ij.cutoff_region, r);
            Ok(IsInteracting(true))
        },
        _ => Ok(IsInteracting(true)),
    }
}

mod reactive_warnings {
    use super::*;

    use std::sync::RwLock;

    struct Records {
        worst_nonbonded_alpha: Option<f64>,
        worst_bonded_alpha: Option<f64>,
    }

    lazy_static! {
        static ref RECORDS: RwLock<Records> = {
            RwLock::new(Records {
                worst_nonbonded_alpha: None,
                worst_bonded_alpha: None,
            })
        };
    }

    pub(super) fn log_nonbonded(interval: (f64, f64), value: f64) {
        let (alpha, _) = linterp_from(interval, (1.0, 0.0), value); // flipped so unbonded gets 0
        if alpha > RECORDS.read().unwrap().worst_nonbonded_alpha.unwrap_or(0.0) {
            let mut records = RECORDS.write().unwrap();

            print_explanation_once(&records);
            warn!(
                "rebo: New record length for nearby atoms: {} ({:.02}% into reactive regime)",
                value, alpha * 100.0,
            );
            records.worst_nonbonded_alpha = Some(alpha);
        }
    }

    pub(super) fn log_bonded(interval: (f64, f64), value: f64) {
        let (alpha, _) = linterp_from(interval, (0.0, 1.0), value);
        if alpha > RECORDS.read().unwrap().worst_bonded_alpha.unwrap_or(0.0) {
            let mut records = RECORDS.write().unwrap();

            print_explanation_once(&records);
            warn!(
                "rebo: New record length for distant atoms: {} ({:.02}% into reactive regime)",
                value, alpha * 100.0,
            );
            records.worst_bonded_alpha = Some(alpha);
        }
    }

    #[inline(always)]
    fn print_explanation_once(Records { worst_bonded_alpha, worst_nonbonded_alpha }: &Records) {
        match (worst_bonded_alpha, worst_nonbonded_alpha) {
            (None, None) => warn!{"\
                A pair of atoms were found in nonreactive REBO that would normally have a \
                fractional weight associated with their interaction. The potential will thus \
                produce slightly different results from the full, reactive implementation.\n\
            "},
            _ => {},
        }
    }
}

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
        let num_diff = numerical::slope(1e-1, Some(DerivativeKind::CentralDifference), x, |x| f(x).0);
        assert_close!{rel=1e-11, abs=1e-11, diff, num_diff};
    }
}

//-----------------------------------------------------------------------------

#[inline(always)] // elide large stack-to-stack copies
fn sbvec_scaled<T: ops::MulAssign<f64>>(f: f64, mut xs: SiteBondVec<T>) -> SiteBondVec<T>
{ scale_mut(f, &mut xs); xs }

#[inline(always)] // elide large stack-to-stack copies
fn sbvec_filled<T: Clone>(fill: T, len: usize) -> SiteBondVec<T>
{ std::iter::repeat(fill).take(len).collect() }

#[inline(always)] // elide large stack-to-stack copies
fn axpy_mut<T: Copy>(a: &mut [T], alpha: f64, b: &[T])
where
    f64: ops::Mul<T, Output=T>,
    T: ops::AddAssign<T>,
{
    for (a, b) in zip_eq!(a, b) {
        *a += alpha * *b;
    }
}

fn scaled<T: ops::MulAssign<f64>>(f: f64, mut xs: Vec<T>) -> Vec<T>
{ scale_mut(f, &mut xs); xs }

fn scale_mut<T: std::ops::MulAssign<f64>>(factor: f64, xs: &mut [T]) {
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
#[derive(Deserialize)]
struct ForceFile {
    value: f64,
    grad: Vec<V3>,
}

#[cfg(test)]
mod input_tests {
    use super::*;
    use std::{path::Path, fs::File, io};
    use rsp2_structure_io::Poscar;
    use rsp2_array_types::Unvee;

    const RESOURCE_DIR: &'static str = "tests/resources";
    const BIG_INPUT_1: &'static str = "tblg-2011-150-a";
    const BIG_INPUT_2: &'static str = "gyroid-1";

    #[test]
    fn all() -> FailResult<()> {
        let mut matches = vec![];
        for entry in Path::new(RESOURCE_DIR).join("rebo").read_dir()? {
            let entry: String = entry?.path().display().to_string();
            if let Some(base) = strip_suffix(".lmp.json.xz", &entry) {
                matches.push(Path::new(&base).file_name().unwrap().to_string_lossy().into_owned());
            }
        }
        assert!(!matches.is_empty(), "failed to locate test inputs!");

        for name in matches {
            if name != BIG_INPUT_1 && name != BIG_INPUT_2 {
                println!("Testing {}", name);
                single(&name)?;
            }
        }
        Ok(())
    }

    #[test]
    #[ignore] // time consuming; run with --ignored
    fn big_input_1() -> FailResult<()> {
        single(BIG_INPUT_1)
    }

    #[test]
    #[ignore] // time consuming; run with --ignored
    fn big_input_2() -> FailResult<()> {
        single(BIG_INPUT_2)
    }

    fn single(name: &str) -> FailResult<()> {
        use std::{path, fs::File};
        use rsp2_structure_io::Poscar;
        use rsp2_array_types::Unvee;

        // Set this to false to let tests capture stdout
        let use_rayon = false; // FIXME: revert to true

        // Note: The bug in favata affects the test structure "wing-1".
        let ref params = Params::new_favata();

        let in_path = Path::new(RESOURCE_DIR).join("structure").join(name).join("structure.vasp.xz");
        let out_path = Path::new(RESOURCE_DIR).join("rebo").join(name.to_string() + ".lmp.json.xz");

        let expected: ForceFile = serde_json::from_reader(open_xz(out_path)?)?;
        let Poscar { ref coords, ref elements, .. } = Poscar::from_reader(open_xz(in_path)?)?;
        let ref interactions = find_all_interactions(params, coords, elements)?;

        let (value, grad) = compute(params, interactions, coords, use_rayon)?;

        assert_close!(abs=1e-7, rel=1e-6, value, expected.value, "in file: {}", name);
        assert_close!(abs=1e-7, rel=1e-6, grad.unvee(), expected.grad.unvee(), "in file: {}", name);
        Ok(())
    }

    fn open_xz(path: impl AsRef<Path>) -> io::Result<impl io::Read> {
        File::open(path).map(::xz2::read::XzDecoder::new)
    }

    fn strip_suffix<'a>(suffix: &str, s: &str) -> Option<String> {
        if s.ends_with(suffix) {
            Some(s[..s.len()-suffix.len()].to_string())
        } else {
            None
        }
    }
}
