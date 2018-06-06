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

use ::util::VeclikeIterator;

use ::rsp2_newtype_indices::{Idx, Indexed, IndexVec};
use ::rsp2_soa_ops::{Perm};

use ::std::collections::BTreeMap;

/// Compute stars from the depermutations of a group.
///
/// (this could be *any* kind of group, even one with pure translations; all
///  that matters is that it can be represented by permutations)
pub fn compute_stars(deperms: &[Perm]) -> Stars
{ _compute_stars(Indexed::from_raw_ref(deperms)) }

fn _compute_stars<
    SiteI: Idx,
    OperI: Idx,
    StarI: Idx,
>(
    deperms: &Indexed<OperI, [Perm]>, // FIXME: Perm<SiteI, SiteI>
) -> Stars<SiteI, OperI, StarI> {
    let unassigned = StarI::new(::std::usize::MAX); // an impossible index
    let num_sites = deperms.raw.first().expect("empty groups do not exist!").len();

    let mut stars = IndexVec::<StarI, _>::new();
    let mut assignments = IndexVec::from_elem_n(unassigned, num_sites);

    let mut first_unassigned = SiteI::new(0);
    loop {
        // find an atom which is not yet in any star
        match (first_unassigned.index() ..num_sites).map(SiteI::new).find(|&x| assignments[x] == unassigned) {
            None => break, // all atoms are assigned
            Some(i) => {
                first_unassigned = i;
            },
        }

        // the atom we found shall represent a new star
        let representative = first_unassigned;

        let star_i = stars.push(StarInfo {
            representative,
            sites: Default::default(),
        });
        let star_sites = &mut stars[star_i].sites;

        // find which site this atom is displaced to (keeping coords fixed)
        // by each symmetry operator
        for (oper_i, deperm) in deperms.iter_enumerated() {
            let permuted = SiteI::new(deperm.permute_index(representative.index()));

            // Mark them off.
            // If one of them already belongs to another star, the input deperms were
            //  evidently not closed under inversion and composition.
            assert!(
                assignments[permuted] == star_i || assignments[permuted] == unassigned,
                "compute_stars: input deperms violate the group axioms!",
            );
            assignments[permuted] = star_i;

            // Record each operator that brings us here.
            let entry = {
                star_sites.entry(permuted)
                    .or_insert(StarSiteInfo { opers_from_rep: vec![] })
            };
            entry.opers_from_rep.push(oper_i);
        }
    }
    assert!(assignments.iter().all(|&i| i != unassigned));

    Stars { stars, assignments }
}

/// Collects information describing collections of sites which are equivalent under
/// symmetry.
#[derive(Debug, Clone)]
pub struct Stars<
    SiteI: Idx = usize,
    OperI: Idx = usize,
    StarI: Idx = usize,
> {
    assignments: IndexVec<SiteI, StarI>,
    stars: IndexVec<StarI, StarInfo<SiteI, OperI>>,
}

impl<SiteI: Idx, OperI: Idx, StarI: Idx> Stars<SiteI, OperI, StarI> {
    #[allow(unused)]
    pub fn assignments(&self) -> &Indexed<SiteI, [StarI]>
    { &self.assignments }
}

impl<SiteI: Idx, OperI: Idx, StarI: Idx> ::std::ops::Deref for Stars<SiteI, OperI, StarI> {
    type Target = Indexed<StarI, [StarInfo<SiteI, OperI>]>;

    fn deref(&self) -> &Self::Target { &self.stars }
}

#[derive(Debug, Clone)]
pub struct StarInfo<
    SiteI: Idx = usize,
    OperI: Idx = usize,
> {
    representative: SiteI,
    sites: BTreeMap<SiteI, StarSiteInfo<OperI>>,
}

#[derive(Debug, Clone)]
pub struct StarSiteInfo<
    OperI: Idx = usize,
> {
    opers_from_rep: Vec<OperI>,
}

impl<SiteI: Idx, OperI: Idx> StarInfo<SiteI, OperI> {
    pub fn representative(&self) -> SiteI
    { self.representative }

    #[allow(unused)]
    pub fn members<'a>(&'a self) -> impl VeclikeIterator<Item=SiteI> + 'a
    { self.sites.keys().cloned() }

    // (this API forces unnecessary searching of the map when we are already iterating over it,
    //  but I highly doubt this code is hot enough for it to matter)

    /// All operators that map the representative into a given site.
    pub fn opers_from_rep(&self, site: SiteI) -> &[OperI]
    { &self.sites[&site].opers_from_rep }
}
