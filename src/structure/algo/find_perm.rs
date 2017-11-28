use ::slice_of_array::prelude::*;
use ::{Lattice, CoordStructure};
use ::{FracRot, FracOp};
use super::group::GroupTree;

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
    frac_a: &[f64; 3],
    frac_b: &[f64; 3],
) -> f64
{
    use ::rsp2_array_utils::{arr_from_fn, dot};
    use ::Coords;
    let diff: [_; 3] = arr_from_fn(|k| frac_a[k] - frac_b[k]);
    let diff: [_; 3] = arr_from_fn(|k| diff[k] - diff[k].round());

    let mut diffs = vec![];
    for &a in &[-1., 0., 1.] {
        for &b in &[-1., 0., 1.] {
            for &c in &[-1., 0., 1.] {
                diffs.push([diff[0] + a, diff[1] + b, diff[2] + c]);
            }
        }
    }
    let carts = Coords::Fracs(diffs).to_carts(lattice);
    carts.into_iter().map(|v| dot(&v, &v).sqrt())
        .min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()
}

// Slow, and not even always correct
fn dumb_validate_equivalent(
    lattice: &Lattice,
    frac_a: &[[f64; 3]],
    frac_b: &[[f64; 3]],
    tol: f64,
)
{
    for i in 0..frac_a.len() {
        let d = dumb_nearest_distance(lattice, &frac_a[i], &frac_b[i]);
        assert!(d < tol * (1.0 + 1e-7));
    }
}

// NOTE: Takes CoordStructure to communicate that the algorithm only cares
//       about positions.  There is a small use-case for an <M: Eq> variant
//       which could possibly allow two identical positions to be distinguished
//       (maybe e.g. representing a defect as some superposition with a ghost)
//       but I wouldn't want it to be the default.
pub(crate) fn of_spacegroup(
    prim_structure: &CoordStructure,
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
            let perm = of_rotation_impl(lattice, &from_fracs, &to_fracs[..], tol)?;
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

// NOTE: Takes CoordStructure to communicate that the algorithm only cares
//       about positions.  There is a small use-case for an <M: Eq> variant
//       which could possibly allow two identical positions to be distinguished
//       (maybe e.g. representing a defect as some superposition with a ghost)
//       but I wouldn't want it to be the default.
#[allow(unused)]
pub(crate) fn of_rotation(
    structure: &CoordStructure,
    rotation: &FracRot,
    tol: f64,
) -> Result<Perm>
{Ok({
    let lattice = structure.lattice();
    let from_fracs = structure.to_fracs();
    let to_fracs = rotation.transform_prim(&from_fracs);

    of_rotation_impl(lattice, &from_fracs, &to_fracs, tol)?
})}

fn of_rotation_impl(
    lattice: &Lattice,
    from_fracs: &[[f64; 3]],
    to_fracs: &[[f64; 3]],
    tol: f64,
) -> Result<Perm>
{Ok({
    use ::rsp2_array_utils::dot;
    use ::ordered_float::NotNaN;
    use ::Coords::Fracs;

    // Sort both sides by some measure which is likely to produce a small
    // maximum value of (sorted_rotated_index - sorted_original_index).
    // The C code is optimized for this case, reducing an O(n^2)
    // search down to ~O(n). (for O(n log n) work overall, including the sort)
    //
    // We choose distance from the nearest bravais lattice point as our measure.
    let sort_by_lattice_distance = |fracs: &[[f64; 3]]| {
        let mut fracs = fracs.to_vec();
        for x in fracs.flat_mut() {
            *x -= x.round();
        }

        let distances = Fracs(fracs.clone())
                .to_carts(lattice)
                .into_iter()
                .map(|x| NotNaN::new(dot(&x, &x).sqrt()).unwrap())
                .collect::<Vec<_>>();
        let perm = Perm::argsort(&distances);
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
//       within the four unit cells around the origin
fn brute_force_near_identity(
    lattice: &Lattice,
    from_fracs: &[[f64; 3]],
    to_fracs: &[[f64; 3]],
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
                use ::rsp2_array_utils::{arr_from_fn, dot};
                let diff: [_; 3] = arr_from_fn(|k| from_fracs[from][k] - to_fracs[to][k]);
                let diff: [_; 3] = arr_from_fn(|k| diff[k] - diff[k].round());

                let cart = dot(&diff, lattice.matrix());
                dot(&cart, &cart)
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
    use super::{Perm, Permute};

    use ::slice_of_array::*;
    use ::rand::Rand;

    fn random_vec<T: Rand>(n: u32) -> Vec<T>
    { (0..n).map(|_| ::rand::random()).collect() }

    fn random_problem(n: u32) -> (Vec<[f64; 3]>, Perm, Vec<[f64; 3]>)
    {
        let original: Vec<[f64; 3]> = random_vec(n);
        let perm = Perm::random(n);
        let permuted = original.clone().permuted_by(&perm);
        (original, perm, permuted)
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
            &lattice, &original, &permuted, 1e-5,
        ).unwrap();

        assert_eq!(output, perm);
    }
}
