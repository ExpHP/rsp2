#![allow(deprecated)]
#![allow(non_snake_case)]
use ::{Lattice, Coords, CoordsKind};
use ::{IntRot, FracOp, CartOp};
use super::group::GroupTree;

use ::rsp2_array_types::V3;
use ::rsp2_soa_ops::{Perm, Permute};

use ::failure::{Backtrace, Error};

/// Validate that structure is symmetric under the given operators.
///
/// Slow, and not even always correct. (the voronoi cell of the lattice
/// must be fully contained within one cell image in each direction)
#[deprecated]
pub fn frac__dumb_symmetry_test(
    structure: &Coords,
    ops: &[FracOp],
    tol: f64,
) -> Result<(), Error>
{Ok({
    let lattice = structure.lattice();
    let from_fracs = structure.to_fracs();
    let perms = frac__of_spacegroup_for_primitive(structure, ops, tol)?;

    for (op, perm) in izip!(ops, perms) {
        let transformed_fracs = op.transform_prim(&from_fracs);
        let transformed = Coords::new(lattice.clone(), CoordsKind::Fracs(transformed_fracs));
        let permuted = structure.clone().permuted_by(&perm);

        transformed.check_same_cell_and_order(&permuted, tol)?;
    }
})}

/// Compute permutations for all operators in a spacegroup.
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
#[deprecated]
pub fn of_spacegroup(
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
    of_spacegroup_with_meta(coords, &dummy_meta, ops, tol)
}

/// Compute permutations for all operators in a spacegroup.
pub fn frac__of_spacegroup_for_general(
    // NOTE: This really does require data from both the primitive and supercell.
    //       Justifications are in the `with_meta` variant.

    // Arbitrary superstructure (the thing we want to permute)
    coords: &Coords,
    // Spacegroup operators for the primitive structure.
    prim_ops: &[FracOp],
    // The lattice that the fractional operators operate on.
    prim_lattice: &Lattice,
    tol: f64,
) -> Result<Vec<Perm>, Error>
{
    let dummy_meta = vec![(); coords.num_atoms()];
    frac__of_spacegroup_for_general_with_meta(coords, &dummy_meta, prim_ops, prim_lattice, tol)
}

/// Compute permutations for all operators in a spacegroup.
pub fn frac__of_spacegroup_for_primitive(
    prim_coords: &Coords,
    ops: &[FracOp],
    tol: f64,
) -> Result<Vec<Perm>, Error>
{
    let dummy_meta = vec![(); prim_coords.num_atoms()];
    frac__of_spacegroup_for_primitive_with_meta(prim_coords, &dummy_meta, ops, tol)
}

// NOTE: These versions use the metadata to group the atoms and potentially
//       elide even more comparisons. Is it effective? No idea! But it comes at
//       zero extra cost for `M = ()` and hasn't been hurting anyone, so I
//       figured I'll leave it in.
#[deprecated]
pub fn frac__of_spacegroup_for_general_with_meta<M: Ord>(
    // NOTE: This really does require data from both the primitive and supercell.
    //       Justifications follow.

    // Arbitrary superstructure (the thing we want to permute)
    coords: &Coords,
    metadata: &[M],

    // Integer-based spacegroup operators for the primitive structure,
    // because we need a form of the operators that is hashable
    // for the GroupTree composition trick.
    prim_ops: &[FracOp],

    // The lattice that the fractional operators are in units of,
    // so that we can recover cartesian forms of the operators.
    // (which will be safe to apply to the superstructure)
    prim_lattice: &Lattice,

    tol: f64,
) -> Result<Vec<Perm>, Error>
{
    let cart_ops: Vec<_> = {
        prim_ops.iter()
            .map(|f| f.to_rot().to_cart_op_with_frac_trans(f.to_trans().frac(), prim_lattice))
            .collect()
    };
    of_spacegroup_with_meta(coords, metadata, &cart_ops, tol)
}

pub fn of_spacegroup_with_meta<M: Ord>(
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

#[deprecated]
pub fn frac__of_spacegroup_for_primitive_with_meta<M: Ord>(
    prim_structure: &Coords,
    metadata: &[M],
    ops: &[FracOp],
    tol: f64,
) -> Result<Vec<Perm>, Error>
{
    frac__of_spacegroup_for_general_with_meta(prim_structure, metadata, ops, prim_structure.lattice(), tol)
}

#[allow(unused)]
#[deprecated]
pub(crate) fn frac__of_rotation_with_meta<M: Ord>(
    structure: &Coords,
    meta: &[M],
    rotation: &IntRot,
    tol: f64,
) -> Result<Perm, PositionMatchError>
{Ok({
    let lattice = structure.lattice();
    let from_fracs = structure.to_fracs();
    let to_fracs = rotation.transform_fracs(&from_fracs);

    brute_force_with_sort_trick(lattice, meta, &from_fracs, &to_fracs, tol)?
})}

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
