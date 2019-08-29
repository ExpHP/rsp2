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

use crate::FailResult;

use rayon_cond::CondIterator;
use rsp2_structure::Coords;
use rsp2_structure::bonds::{FracBond, PeriodicGraph};
use rsp2_array_types::V3;
#[allow(unused)] // https://github.com/rust-lang/rust/issues/45268
use rsp2_newtype_indices::{Idx, IndexVec, Indexed, self as idx};

// NOTE: While it's unlikely that one will ever confuse a site index for a bond index,
//       wrapping BondI is extremely useful to prevent accidental confusion between
//       "site-bond" indices (indices into a SiteBondVec) and "bond" indices (unique
//       indices for all bonds)
pub(crate) use newtype_indices::*;
pub(crate) mod newtype_indices {
    // mod so they're not pub
    newtype_index!{SiteI}
    newtype_index!{BondI}
}

/// Helper used by some potentials to manage the list of terms they need to compute.
#[derive(Debug, Clone)]
pub struct Interactions<P: Potential> {
    potential: P,

    /// CSR-style divider indices for bonds at each site.
    bond_div: IndexVec<SiteI, BondI>,

    site_type: IndexVec<SiteI, P::AtomType>,
    bond_image_diff: IndexVec<BondI, V3<i32>>,
    bond_reverse_index: IndexVec<BondI, BondI>,
    bond_source: IndexVec<BondI, SiteI>,
    bond_target: IndexVec<BondI, SiteI>,
}

impl<P: Potential> Interactions<P> {
    /// Identify which pairs of atoms are close enough to interact and build a list of
    /// interactions.
    ///
    /// This will warn or generate errors when bond lengths fall into the reactive regime,
    /// as if the `check_distances` method was called.
    pub fn compute(
        potential: P,
        coords: &Coords,
        types: &[P::AtomType],
        bond_graph: &PeriodicGraph,
    ) -> FailResult<Self> {
        let mut bond_div = IndexVec::<SiteI, _>::from_raw(vec![BondI(0)]);
        bond_div.raw.reserve(bond_graph.node_count());

        let mut bond_source = IndexVec::<BondI, SiteI>::with_capacity(bond_graph.edge_count());
        let mut bond_target = IndexVec::<BondI, SiteI>::with_capacity(bond_graph.edge_count());
        let mut bond_image_diff = IndexVec::<BondI, V3<i32>>::with_capacity(bond_graph.edge_count());
        let site_type = IndexVec::<SiteI, _>::from_raw(types.to_vec());

        let carts = coords.to_carts();

        // Make a pass to get all the bond divs right.
        for node in bond_graph.node_indices() {
            let site_i = SiteI(node.index());

            for frac_bond_ij in bond_graph.frac_bonds_from(site_i.index()) {
                let site_j = SiteI(frac_bond_ij.to);
                let cart_vector = frac_bond_ij.cart_vector_using_carts(coords.lattice(), &carts);

                if let IsInteracting(false) = potential.check_distance(
                    cart_vector.norm(), (site_type[site_i].clone(), site_type[site_j].clone()),
                )? {
                    continue;
                }

                bond_source.push(site_i);
                bond_target.push(site_j);
                bond_image_diff.push(frac_bond_ij.image_diff);
            } // for bond_ij

            let num_bonds = bond_target.len() - bond_div.raw.last().unwrap().index();
            if let Some(max_bonds) = P::max_bonds() {
                if num_bonds > max_bonds {
                    bail!("An atom has too many bonds! ({}, max: {})", num_bonds, max_bonds);
                }
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
            } // for bond_ij
        } // for node

        assert_eq!(bond_reverse_index.len(), bond_div.raw.last().unwrap().0);
        Ok(Interactions {
            potential, bond_div, site_type,
            bond_target, bond_image_diff, bond_reverse_index, bond_source,
        })
    }
}

impl<P: Potential> Interactions<P> {
    pub fn num_sites(&self) -> usize { self.site_type.len() }
    pub fn num_bonds(&self) -> usize { self.bond_target.len() }

    #[inline(always)] pub(crate) fn site_type(&self, site: SiteI) -> P::AtomType { self.site_type[site].clone() }
    #[inline(always)] pub(crate) fn bond_source(&self, bond: BondI) -> SiteI { self.bond_source[bond] }
    #[inline(always)] pub(crate) fn bond_target(&self, bond: BondI) -> SiteI { self.bond_target[bond] }
    #[inline(always)] pub(crate) fn bond_reverse_index(&self, bond: BondI) -> BondI { self.bond_reverse_index[bond] }
    #[inline(always)] pub(crate) fn bond_image_diff(&self, bond: BondI) -> V3<i32> { self.bond_image_diff[bond] }

    /// Retrieves a "site-bond" index, i.e. the index of a bond in the list of bonds for a single
    /// site.
    ///
    /// More specifically, given a slice of data acquired by calling `site_bond_slice` at the
    /// source site, this returns the index into that slice which corresponds to this bond.
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

    pub(crate) fn sites_maybe_parallel(&self, use_rayon: bool) -> CondIterator<
        impl rayon::iter::IndexedParallelIterator<Item=SiteI>,
        impl ExactSizeIterator<Item=SiteI>,
    > { CondIterator::new(0..self.num_sites(), use_rayon).map(SiteI) }

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

    /// Check that all bond lengths are still reasonable.
    ///
    /// NOTE: This can only really catch bonds that break.  It won't notice if any atoms have
    /// moved into bonding range after the construction of `Interactions`.
    ///
    /// Depending on the potential, may log warnings or return `Err` on really bad cases.
    pub fn check_distances(&self, coords: &Coords, use_rayon: bool) -> FailResult<()> {
        let bond_deltas = self.compute_bond_deltas(coords, use_rayon);
        for (bond_ij, delta_ij) in bond_deltas.iter_enumerated() {
            let type_i = self.site_type(self.bond_source(bond_ij));
            let type_j = self.site_type(self.bond_target(bond_ij));
            self.potential.check_distance(delta_ij.norm(), (type_i, type_j))?;
        }
        Ok(())
    }
}

/// The trait a potential needs to implement to use `Interactions`.
pub trait Potential: Sync {
    type AtomType: Clone + Sync;

    /// Determine whether a bond of this length should be included in the bond list.
    ///
    /// Some potentials may choose to fail for some bond lengths.  (e.g. potentials that can't
    /// handle reactions may fail on borderline values of length). Some implementations may also
    /// choose to log warnings.
    fn check_distance(&self, distance: f64, types: (Self::AtomType, Self::AtomType)) -> FailResult<IsInteracting>;

    fn max_bonds() -> Option<usize>;
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct IsInteracting(pub bool);

impl<P: Potential> Interactions<P> {
    /// Compute all bond delta vectors, defined as the position of the target site minus the
    /// position of the source site.
    pub fn compute_bond_deltas(&self, coords: &Coords, use_rayon: bool) -> IndexVec<BondI, V3> {
        let site_carts = IndexVec::<SiteI, _>::from_raw(coords.to_carts());
        let vec = CondIterator::new(0..self.num_bonds(), use_rayon).map(|bond| {
            let bond = BondI(bond);
            let cart_from = site_carts[self.bond_source(bond)];
            let cart_to = site_carts[self.bond_target(bond)];
            let image_diff = self.bond_image_diff(bond);
            cart_to - cart_from + image_diff.map(|x| x as f64) * coords.lattice()
        }).collect();
        IndexVec::from_raw(vec)
    }
}
