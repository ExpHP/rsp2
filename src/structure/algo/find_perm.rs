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

use crate::{Lattice, Coords, CoordsKind};
use crate::{CartOp};
use super::group::GroupTree;

use rsp2_array_types::V3;
use rsp2_soa_ops::{Perm, Permute};

use failure::{Backtrace, Error};

/// Compute copermutations for all operators in a spacegroup.
///
/// Apologies for the invented terminology; see `conventions.md` about the difference
/// between copermutations and depermutations.
///
/// This method can be called on superstructures so long as pure translations
/// are not included in the list of operators.  Be aware that if the superlattice
/// breaks symmetries of the primitive structure, those symmetries might not have
/// a valid representation as a permutation (and the method will fail).
//
// (NOTE: currently, it actually fails even earlier, when trying to construct IntRot
//        (which fails if and only if the superlattice breaks the symmetry).
//        I don't know / have not yet proven whether there may exist symmetry-broken
//        supercells which DO have a valid permutation representation)
pub fn spacegroup_coperms(
    // Arbitrary superstructure (the thing we want to permute)
    coords: &Coords,

    // Spacegroup operators.
    //
    // * Must be closed under composition.
    // * Must not include pure translations. (this limitation is because the
    //   the method used to equate two operators only considers the rotations)
    ops: &[CartOp],

    tol: f64,
) -> Result<Vec<Perm>, Error>
{
    let dummy_meta = vec![(); coords.num_atoms()];
    spacegroup_coperms_with_meta(coords, &dummy_meta, ops, tol)
}

/// Compute depermutations for all operators in a spacegroup.
///
/// Apologies for the invented terminology; see `conventions.md` about the difference
/// between copermutations and depermutations.
///
/// This method can be called on superstructures so long as pure translations
/// are not included in the list of operators.  Be aware that if the superlattice
/// breaks symmetries of the primitive structure, those symmetries might not have
/// a valid representation as a permutation (and the method will fail).
//
// (NOTE: currently, it actually fails even earlier, when trying to construct IntRot
//        (which fails if and only if the superlattice breaks the symmetry).
//        I don't know / have not yet proven whether there may exist symmetry-broken
//        supercells which DO have a valid permutation representation)
pub fn spacegroup_deperms(
    // Arbitrary superstructure (the thing we want to permute)
    coords: &Coords,

    // Spacegroup operators.
    //
    // * Must be closed under composition.
    // * Must not include pure translations. (this limitation is because the
    //   the method used to equate two operators only considers the rotations)
    ops: &[CartOp],

    tol: f64,
) -> Result<Vec<Perm>, Error>
{ spacegroup_coperms(coords, ops, tol).map(invert_each) }

// NOTE: This version uses the metadata to group the atoms and potentially
//       elide even more comparisons. Is it effective? No idea! But it comes at
//       zero extra cost for `M = ()` and hasn't been hurting anyone, so I
//       figured I'll leave it in.
pub fn spacegroup_coperms_with_meta<M: Ord>(
    // Arbitrary superstructure (the thing we want to permute)
    coords: &Coords,
    // Metadata, which is assumed to obey the symmetry of the spacegroup.
    // (i.e. sites in a symmetry star should have the same metadata)
    //
    // (FIXME: This is not checked)
    metadata: &[M],

    // Spacegroup operators.
    cart_ops: &[CartOp],

    tol: f64,
) -> Result<Vec<Perm>, Error>
{Ok({
    let lattice = coords.lattice();
    let from_fracs = coords.to_fracs();

    // Integer-based spacegroup operators, because we need a form of
    // the operators that is hashable for the GroupTree composition trick.
    let int_ops: Vec<_> = {
        cart_ops.iter()
            .map(|c| c.int_rot(lattice))
            .collect::<Result<_, _>>()?
    };

    // Find relations between the group operators and
    // identify a small number of base cases ("generators").
    let tree = GroupTree::from_all_members(
        int_ops,
        |a, b| a.then(b),
    );

    tree.try_compute_homomorphism(
        // Generators: Do a (very expensive!) brute force search.
        |op_ind, _int_op| {
            let to_fracs = cart_ops[op_ind].transform_fracs(lattice, &from_fracs);
            brute_force_with_sort_trick(
                lattice,
                metadata, CoordsKind::Fracs(&from_fracs),
                metadata, CoordsKind::Fracs(&to_fracs[..]),
                tol,
            )
        },
        // Other operators: Quickly compose the results from other operators.
        |a, b| Ok({
            // Flip the order, because the permutations we seek
            //  actually come from the opposite group.
            //
            // i.e.  given P_a X = X R_a
            //         and P_b X = X R_b,
            //  one can easily show that  X R_a R_b = P_a P_b X
            b.clone().permuted_by(a)
        }),
    )?
})}

pub fn spacegroup_deperms_with_meta<M: Ord>(
    // Arbitrary superstructure (the thing we want to permute)
    coords: &Coords,
    // Metadata, which is assumed to obey the symmetry of the spacegroup.
    // (i.e. sites in a symmetry star should have the same metadata)
    //
    // (FIXME: This is not checked)
    metadata: &[M],

    // Spacegroup operators.
    cart_ops: &[CartOp],

    tol: f64,
) -> Result<Vec<Perm>, Error>
{ spacegroup_coperms_with_meta(coords, metadata, cart_ops, tol).map(invert_each) }

fn invert_each(perms: impl IntoIterator<Item=Perm>) -> Vec<Perm>
{ perms.into_iter().map(|p| p.inverted()).collect() }

pub(crate) fn brute_force_with_sort_trick<M: Ord>(
    lattice: &Lattice,
    from_meta: &[M],
    from: CoordsKind<impl AsRef<[V3]>>,
    to_meta: &[M],
    to: CoordsKind<impl AsRef<[V3]>>,
    tol: f64,
) -> Result<Perm, PositionMatchError>
{Ok({
    let (perm_from, sorted_from) = fracs_sorted_by_lattice_distance(lattice, from, from_meta);
    let (perm_to, sorted_to) = fracs_sorted_by_lattice_distance(lattice, to, to_meta);

    let perm_between = brute_force_near_identity(
        lattice, &sorted_from[..], &sorted_to[..], tol,
    )?;

    // Compose all of the permutations for the full permutation.
    //
    // Note that permutations are associative; that is,
    //     x.permute(p).permute(q) == x.permute(p.permute(q))
    perm_from
        .permuted_by(&perm_between)
        .permuted_by(&perm_to.inverted())
})}

// FIXME allows mismatched meta
pub(crate) fn set_difference_with_sort_trick<M: Ord>(
    lattice: &Lattice,
    a_meta: &[M],
    a: CoordsKind<impl AsRef<[V3]>>,
    b_meta: &[M],
    b: CoordsKind<impl AsRef<[V3]>>,
    tol: f64,
) -> Missing
{
    let (perm_a, sorted_a) = fracs_sorted_by_lattice_distance(lattice, a, a_meta);
    let (perm_b, sorted_b) = fracs_sorted_by_lattice_distance(lattice, b, b_meta);

    let missing = brute_force_set_difference(
        lattice, &sorted_a[..], &sorted_b[..], tol,
    );

    missing
        .permuting_a_by(perm_a.inverted())
        .permuting_b_by(perm_b.inverted())
}

// Both structures are sorted by some measure which is likely to produce a small
// maximum value of (sorted_rotated_index - sorted_original_index).
// This reduces an O(n^2) search down to ~O(n).
// (for O(n log n) work overall, including the sort)
//
// We choose to sort first by atom type, then by distance to the nearest
// bravais lattice point.
#[inline(always)]
fn fracs_sorted_by_lattice_distance<M: Ord>(
    lattice: &Lattice,
    coords: CoordsKind<impl AsRef<[V3]>>,
    meta: &[M],
) -> (Perm, Vec<V3>) {
    use ordered_float::NotNan;

    let mut fracs = coords.to_fracs(lattice);
    for v in &mut fracs {
        *v -= v.map(f64::round);
    }

    let data_to_sort = {
        crate::CoordsKind::Fracs(fracs.clone())
            // NOTE: It's possible that computing distances in fractional space can be
            //       even more effective than in cartesian space.  See this comment
            //       and the conversation leading up to it:
            //
            //       https://github.com/atztogo/spglib/pull/44#issuecomment-356516736
            //
            //       But for now, I'll leave it as is.
            .to_carts(lattice)
            .into_iter()
            .zip(meta)
            .map(|(x, m)| {
                (
                    m, // first by atom type
                    NotNan::new(x.norm()).unwrap(),
                )
            })
            .collect::<Vec<_>>()
    };
    let perm = Perm::argsort(&data_to_sort);
    (perm.clone(), fracs.permuted_by(&perm))
}

#[derive(Debug, Fail)]
pub enum PositionMatchError {
    #[fail(display = "positions are too dissimilar")]
    NoMatch(Backtrace),
    #[fail(display = "multiple positions mapped to the same index")]
    DuplicateMatch(Backtrace),
}

/// The type returned by set difference operations on coords, providing the
/// atoms missing from each structure via unambiguously-named fields.
#[derive(Debug, Clone)]
pub struct Missing {
    pub only_in_a: Vec<usize>,
    pub only_in_b: Vec<usize>,
}

impl Missing {
    fn permuting_a_by(mut self, perm: Perm) -> Self {
        self.only_in_a = self.only_in_a.into_iter().map(|i| perm.permute_index(i)).collect();
        self
    }
    fn permuting_b_by(mut self, perm: Perm) -> Self {
        self.only_in_b = self.only_in_b.into_iter().map(|i| perm.permute_index(i)).collect();
        self
    }
}

// Optimized for permutations near the identity.
// NOTE: Lattice must be reduced so that the voronoi cell fits
//       within the eight unit cells around the origin
fn brute_force_near_identity(
    lattice: &Lattice,
    from_fracs: &[V3],
    to_fracs: &[V3],
    tol: f64,
) -> Result<Perm, PositionMatchError>
{Ok({
    assert_eq!(from_fracs.len(), to_fracs.len());
    let n = from_fracs.len();

    const UNSET: usize = std::usize::MAX;
    assert!(n < UNSET);

    let mut perm = vec![UNSET; from_fracs.len()];

    // optimization: Rather than filling the out vector in order,
    // we find where each index belongs (e.g. we place the 0, then
    // we place the 1, etc.).
    // Then we can track the first unassigned index.
    //
    // This works best if the permutation is close to the identity.
    // (more specifically, if the max value of 'out[i] - i' is small)
    //
    // This optimization does create some data dependencies which block
    // opportunities for parallelization; but for reducing O(n^2)
    // computations down to O(n), it is worth it.
    let mut search_start = 0;

    'from: for from in 0..n {

        // Skip through things filled out of order.
        while perm[search_start] != UNSET {
            search_start += 1;
        }

        for to in search_start..n {
            if perm[to] != UNSET {
                continue;
            }

            if fracs_within(lattice, from_fracs[from], to_fracs[to], tol) {
                perm[to] = from;
                continue 'from;
            }
        }
        return Err(PositionMatchError::NoMatch(Backtrace::new()));
    }

    if perm.iter().any(|&x| x == UNSET) {
        return Err(PositionMatchError::DuplicateMatch(Backtrace::new()));
    }

    Perm::from_vec(perm).expect("(BUG) invalid perm without match error!?")
})}

// Optimized for permutations near the identity.
// NOTE: Lattice must be reduced so that the voronoi cell fits
//       within the eight unit cells around the origin
fn brute_force_set_difference(
    lattice: &Lattice,
    a_fracs: &[V3],
    b_fracs: &[V3],
    tol: f64,
) -> Missing
{
    let mut found_b = vec![false; b_fracs.len()];

    // Same optimization described in from_perm.
    //
    // FIXME: If even a single thing is missing from `to`, then we end up with
    //        quadratic complexity. I think this can be fixed by implementing some
    //        sort of jump list?  Meh.
    let mut search_start = 0;
    let mut only_in_a = vec![];

    'a: for a in 0..a_fracs.len() {
        // Skip through things filled out of order.
        while search_start < b_fracs.len() && found_b[search_start] {
            search_start += 1;
        }

        for b in search_start..b_fracs.len() {
            if found_b[b] {
                continue;
            }

            if fracs_within(lattice, a_fracs[a], b_fracs[b], tol) {
                found_b[b] = true;
                continue 'a;
            }
        }
        only_in_a.push(a);
    }

    let only_in_b = {
        found_b.iter().enumerate()
            .filter(|(_, &x)| !x)
            .map(|(b, _)| b)
            .collect()
    };
    Missing { only_in_a, only_in_b }
}

// Determine whether two fractional points have images that lie within a cartesian
// distance tol of each other, assuming that the voronoi cell fits within the eight
// unit cells around the origin.
#[inline(always)] // hopefully lift the `tol * tol` out of a loop
fn fracs_within(lattice: &Lattice, a: V3, b: V3, tol: f64) -> bool {
    let shortest_diff = (a - b).map(|x| x - x.round());
    let shortest_cart = shortest_diff * lattice;
    shortest_cart.sqnorm() < tol * tol
}

#[cfg(test)]
#[deny(unused)]
mod tests {
    use crate::Lattice;
    use super::*;

    use rand::{Rand, Rng};

    use rsp2_array_types::Envee;
    use slice_of_array::prelude::*;

    fn random_vec<T: Rand>(n: usize) -> Vec<T>
    { (0..n).map(|_| rand::random()).collect() }

    fn random_problem(n: usize) -> (Vec<V3>, Perm, Vec<V3>)
    {
        let original: Vec<[f64; 3]> = random_vec(n);
        let perm = Perm::random(n);
        let permuted = original.clone().permuted_by(&perm);
        (original.envee(), perm, permuted.envee())
    }

    #[test]
    fn brute_force_works() {
        for _ in 0..10 {
            let (original, perm, permuted) = random_problem(20);
            let lattice = Lattice::random_uniform(1.0);

            let output = super::brute_force_near_identity(
                &lattice, &original, &permuted, 1e-5,
            ).unwrap();

            assert_eq!(output, perm);
        }
    }

    #[test]
    fn sort_trick_works() {
        for _ in 0..10 {
            let (original, perm, permuted) = random_problem(20);
            let lattice = Lattice::random_uniform(1.0);

            let output = super::brute_force_with_sort_trick(
                &lattice,
                &[(); 20], CoordsKind::Fracs(&original),
                &[(); 20], CoordsKind::Fracs(&permuted),
                1e-5,
            ).unwrap();

            assert_eq!(output, perm);
        }
    }

    #[test]
    fn sort_trick_images() {
        for _ in 0..10 {
            let (mut original, perm, mut permuted) = random_problem(20);
            // FIXME: Figure out how to detect if a lattice is "well-behaved".
            //
            // HACK: Use a fixed lattice so that we can be sure it supports our limitations on
            //       the voronoi cell.
            let half_r3 = 0.5 * f64::sqrt(3.0);
            let lattice = Lattice::from(&[
                [ 1.0,     0.0, 0.0],
                [-0.5, half_r3, 0.0],
                [ 0.0,     0.0, 1.0],
            ]);

            for x in std::iter::empty().chain(original.flat_mut()).chain(permuted.flat_mut()) {
                *x += rand::thread_rng().gen_range::<i32>(-5, 5+1) as f64;
            }

            let output = super::brute_force_with_sort_trick(
                &lattice,
                &[(); 20], CoordsKind::Fracs(&original),
                &[(); 20], CoordsKind::Fracs(&permuted),
                1e-5,
            ).unwrap();

            assert_eq!(output, perm);
        }
    }

    // FIXME known failure
//    #[test]
//    fn meta_mismatch() {
//        let meta = &['A', 'B'];
//        let lattice = Lattice::random_uniform(1.0);
//        let original = &[V3([0.0; 3]), V3([0.5; 3])];
//        let permuted = &[V3([0.5; 3]), V3([0.0; 3])];
//        assert!(
//            super::brute_force_with_sort_trick(&lattice, meta, original, meta, permuted, 1e-5)
//                .is_err()
//        );
//    }

    // a nonempty set of max size r containing indices into a vec of length n, in sorted order
    fn random_index_subset(n: usize, r: usize) -> Vec<usize> {
        let mut mask = vec![false; n];
        for _ in 0..r {
            let u = rand::random::<usize>() % mask.len();
            mask[u] = true;
        }
        mask.iter().enumerate().filter(|(_, &x)| x).map(|(i, _)| i).collect()
    }

    fn remove_item<T: PartialEq>(vec: &mut Vec<T>, item: &T) -> Option<T> {
        let index = vec.iter().position(|x| x == item)?;
        Some(vec.remove(index))
    }

    #[test]
    fn missing() {
        for _ in 0..10 {
            let (full, _, permuted) = random_problem(20);
            let lattice = Lattice::random_uniform(1.0);

            let empty: Vec<usize> = vec![];

            let full_meta = vec![(); 20];

            // test with nothing missing
            let Missing {
                only_in_a, only_in_b,
            } = super::set_difference_with_sort_trick(
                &lattice,
                &full_meta, CoordsKind::Fracs(&full),
                &full_meta, CoordsKind::Fracs(&permuted),
                1e-5,
            );
            assert_eq!(&only_in_a, &empty);
            assert_eq!(&only_in_b, &empty);

            // now remove some things....
            let removed = random_index_subset(20, 3);
            assert!(removed.len() > 0);

            let mut partial = permuted.clone();
            for &i in &removed {
                remove_item(&mut partial, &full[i]);
            }
            let partial_meta = vec![(); partial.len()];

            // test things missing from b
            let Missing {
                mut only_in_a, only_in_b,
            } = super::set_difference_with_sort_trick(
                &lattice,
                &full_meta, CoordsKind::Fracs(&full),
                &partial_meta, CoordsKind::Fracs(&partial),
                1e-5,
            );
            only_in_a.sort();
            assert_eq!(&only_in_a, &removed);
            assert_eq!(&only_in_b, &empty);

            // test things missing from a
            let Missing {
                only_in_a, mut only_in_b,
            } = super::set_difference_with_sort_trick(
                &lattice,
                &partial_meta, CoordsKind::Fracs(&partial),
                &full_meta, CoordsKind::Fracs(&full),
                1e-5,
            );
            only_in_b.sort();
            assert_eq!(&only_in_a, &empty);
            assert_eq!(&only_in_b, &removed);
        }
    }
}
