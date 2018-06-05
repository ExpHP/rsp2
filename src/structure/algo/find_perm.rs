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

use ::{Lattice, Coords};
use ::{CartOp};
use super::group::GroupTree;

use ::rsp2_array_types::V3;
use ::rsp2_soa_ops::{Perm, Permute};

use ::failure::{Backtrace, Error};

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
            brute_force_with_sort_trick(lattice, metadata, &from_fracs, &to_fracs[..], tol)
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
    meta: &[M],
    from_fracs: &[V3],
    to_fracs: &[V3],
    tol: f64,
) -> Result<Perm, PositionMatchError>
{Ok({
    use ::ordered_float::NotNaN;
    use ::CoordsKind::Fracs;

    // Sort both sides by some measure which is likely to produce a small
    // maximum value of (sorted_rotated_index - sorted_original_index).
    // This reduces an O(n^2) search down to ~O(n).
    // (for O(n log n) work overall, including the sort)
    //
    // We choose to sort first by atom type, then by distance to the nearest
    // bravais lattice point.
    let sort_by_lattice_distance = |fracs: &[V3]| {
        let mut fracs = fracs.to_vec();
        for v in &mut fracs {
            *v -= v.map(f64::round);
        }

        let data_to_sort = {
            Fracs(fracs.clone())
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
                        NotNaN::new(x.norm()).unwrap(),
                    )
                })
                .collect::<Vec<_>>()
        };
        let perm = Perm::argsort(&data_to_sort);
        (perm.clone(), fracs.permuted_by(&perm))
    };

    let (perm_from, sorted_from) = sort_by_lattice_distance(&from_fracs);
    let (perm_to, sorted_to) = sort_by_lattice_distance(&to_fracs);

    let perm_between = brute_force_near_identity(
        lattice,
        &sorted_from[..],
        &sorted_to[..],
        tol,
    )?;

    // Compose all of the permutations for the full permutation.
    //
    // Note that permutations are associative; that is,
    //     x.permute(p).permute(q) == x.permute(p.permute(q))
    perm_from
        .permuted_by(&perm_between)
        .permuted_by(&perm_to.inverted())
})}

#[derive(Debug, Fail)]
pub enum PositionMatchError {
    #[fail(display = "positions are too dissimilar")]
    NoMatch(Backtrace),
    #[fail(display = "multiple positions mapped to the same index")]
    DuplicateMatch(Backtrace),
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

    const UNSET: usize = ::std::usize::MAX;
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

            let distance2 = {
                let diff = (from_fracs[from] - to_fracs[to]).map(|x| x - x.round());
                let cart = diff * lattice;
                cart.sqnorm()
            };
            if distance2 < tol * tol {
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

#[cfg(test)]
#[deny(unused)]
mod tests {
    use ::Lattice;
    use super::*;

    use ::rand::Rand;

    use ::rsp2_array_types::Envee;

    fn random_vec<T: Rand>(n: usize) -> Vec<T>
    { (0..n).map(|_| ::rand::random()).collect() }

    fn random_problem(n: usize) -> (Vec<V3>, Perm, Vec<V3>)
    {
        let original: Vec<[f64; 3]> = random_vec(n);
        let perm = Perm::random(n);
        let permuted = original.clone().permuted_by(&perm);
        (original.envee(), perm, permuted.envee())
    }

    #[test]
    fn brute_force_works() {
        let (original, perm, permuted) = random_problem(20);
        let lattice = Lattice::random_uniform(1.0);

        let output = super::brute_force_near_identity(
            &lattice, &original, &permuted, 1e-5,
        ).unwrap();

        assert_eq!(output, perm);
    }

    #[test]
    fn of_rotation_impl_works() {
        let (original, perm, permuted) = random_problem(20);
        let lattice = Lattice::random_uniform(1.0);

        let output = super::brute_force_with_sort_trick(
            &lattice, &[(); 20], &original, &permuted, 1e-5,
        ).unwrap();

        assert_eq!(output, perm);
    }
}
