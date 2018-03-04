use ::slice_of_array::prelude::*;
use ::{Lattice, Structure, CoordStructure};
use ::{FracRot, FracOp};
use super::group::GroupTree;

use ::rsp2_array_types::{V3, Unvee};

use ::Result;
use ::{Perm, Permute};

/// Validate that structure is symmetric under the given operators.
///
/// Slow, and not even always correct. (the voronoi cell of the lattice
/// must be fully contained within one cell image in each direction)
pub fn dumb_symmetry_test(
    structure: &CoordStructure,
    ops: &[FracOp],
    tol: f64,
) -> Result<()>
{Ok({
    let lattice = structure.lattice();
    let from_fracs = structure.to_fracs();
    let perms = of_spacegroup(structure, ops, tol)?;

    for (op, perm) in izip!(ops, perms) {
        dumb_validate_equivalent(
            lattice,
            &op.transform_prim(&from_fracs),
            &from_fracs.to_vec().permuted_by(&perm),
            tol,
        )
    }
})}

// Slow, and not even always correct
fn dumb_nearest_distance(
    lattice: &Lattice,
    frac_a: &V3,
    frac_b: &V3,
) -> f64
{
    use ::Coords;
    let diff = (frac_a - frac_b).map(|x| x - x.round());

    let mut diffs = vec![];
    for &a in &[-1., 0., 1.] {
        for &b in &[-1., 0., 1.] {
            for &c in &[-1., 0., 1.] {
                diffs.push(diff + V3([a, b, c]));
            }
        }
    }

    let carts = Coords::Fracs(diffs).to_carts(lattice);
    carts.into_iter().map(|v| v.norm())
        .min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()
}

// Slow, and not even always correct
fn dumb_validate_equivalent(
    lattice: &Lattice,
    frac_a: &[V3],
    frac_b: &[V3],
    tol: f64,
) {
    assert_eq!(frac_a.len(), frac_b.len());
    for (a, b) in izip!(frac_a, frac_b) {
        let d = dumb_nearest_distance(lattice, a, b);
        assert!(d < tol * (1.0 + 1e-7));
    }
}

// NOTE: Takes CoordStructure as a speedbump to prevent accidental use
//       with inappropriate metadata.
pub(crate) fn of_spacegroup(
    prim_structure: &CoordStructure,
    ops: &[FracOp],
    tol: f64,
) -> Result<Vec<Perm>>
{
    of_spacegroup_with_meta(prim_structure, ops, tol)
}

// NOTE: This version uses the metadata to group the atoms and potentially
//       elide even more comparisons. Is it effective? No idea! But adding it
//       came at zero extra cost for `M = ()`, so I figured it's worth trying out.
pub(crate) fn of_spacegroup_with_meta<M: Ord>(
    prim_structure: &Structure<M>,
    ops: &[FracOp],
    tol: f64,
) -> Result<Vec<Perm>>
{Ok({
    use ::errors::*;
    let lattice = prim_structure.lattice();
    let from_fracs = prim_structure.to_fracs();

    let tree = GroupTree::from_all_members(
        ops.to_vec(),
        |a, b| a.then(b),
    );

    tree.try_compute_homomorphism(
        |op| Ok::<_, Error>({
            let to_fracs = op.transform_prim(&from_fracs);
            let perm = of_rotation_impl(lattice, prim_structure.metadata(), &from_fracs, &to_fracs[..], tol)?;
            dumb_validate_equivalent(
                lattice,
                &to_fracs[..],
                &from_fracs.to_vec().permuted_by(&perm)[..],
                tol
            );
            perm
        }),
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

// NOTE: Takes CoordStructure as a speedbump to prevent accidental use
//       with inappropriate metadata.
#[allow(unused)]
pub(crate) fn of_rotation(
    structure: &CoordStructure,
    rotation: &FracRot,
    tol: f64,
) -> Result<Perm>
{ of_rotation_with_meta(structure, rotation, tol) }

// NOTE: This version uses the metadata to group the atoms and potentially
//       elide even more comparisons. Is it effective? No idea! But adding it
//       came at zero extra cost for `M = ()`, so I figured it's worth trying out.
#[allow(unused)]
pub(crate) fn of_rotation_with_meta<M: Ord>(
    structure: &Structure<M>,
    rotation: &FracRot,
    tol: f64,
) -> Result<Perm>
{Ok({
    let lattice = structure.lattice();
    let from_fracs = structure.to_fracs();
    let to_fracs = rotation.transform_prim(&from_fracs);
    let meta = structure.metadata();

    of_rotation_impl(lattice, meta, &from_fracs, &to_fracs, tol)?
})}

fn of_rotation_impl<M: Ord>(
    lattice: &Lattice,
    meta: &[M],
    from_fracs: &[V3],
    to_fracs: &[V3],
    tol: f64,
) -> Result<Perm>
{Ok({
    use ::ordered_float::NotNaN;
    use ::Coords::Fracs;

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

        let data_to_sort = Fracs(fracs.clone())
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
                .collect::<Vec<_>>();
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


// Optimized for permutations near the identity.
// NOTE: Lattice must be reduced so that the voronoi cell fits
//       within the eight unit cells around the origin
fn brute_force_near_identity(
    lattice: &Lattice,
    from_fracs: &[V3],
    to_fracs: &[V3],
    tol: f64,
) -> Result<Perm>
{Ok({

    assert_eq!(from_fracs.len(), to_fracs.len());
    let n = from_fracs.len();

    const UNSET: u32 = ::std::u32::MAX;
    assert!(n < UNSET as usize);

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

                let cart = ::rsp2_array_utils::dot(&diff.0, lattice.matrix());
                V3(cart).sqnorm()
            };
            if distance2 < tol * tol {
                perm[to] = from as u32;
                continue 'from;
            }
        }
        bail!("positions are too dissimilar");
    }

    ensure!(
        perm.iter().all(|&x| x != UNSET),
        "multiple positions mapped to the same index");

    Perm::from_vec(perm)?
})}

#[cfg(test)]
#[deny(unused)]
mod tests {
    use ::Lattice;
    use super::*;

    use ::rand::Rand;

    use ::rsp2_array_types::Envee;

    fn random_vec<T: Rand>(n: u32) -> Vec<T>
    { (0..n).map(|_| ::rand::random()).collect() }

    fn random_problem(n: u32) -> (Vec<V3>, Perm, Vec<V3>)
    {
        let original: Vec<[f64; 3]> = random_vec(n);
        let perm = Perm::random(n);
        let permuted = original.clone().permuted_by(&perm);
        (original.envee(), perm, permuted.envee())
    }

    #[test]
    fn brute_force_works() {
        let (original, perm, permuted) = random_problem(20);
        let lattice = Lattice::new(random_vec(3).as_array());

        let output = super::brute_force_near_identity(
            &lattice, &original, &permuted, 1e-5,
        ).unwrap();

        assert_eq!(output, perm);
    }

    #[test]
    fn of_rotation_impl_works() {
        let (original, perm, permuted) = random_problem(20);
        let lattice = Lattice::new(random_vec(3).as_array());

        let output = super::of_rotation_impl(
            &lattice, &[(); 20], &original, &permuted, 1e-5,
        ).unwrap();

        assert_eq!(output, perm);
    }
}
