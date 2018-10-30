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

use ::failure::Error;
use ::{Coords, Lattice};
use ::rsp2_soa_ops::{Permute, Perm, Part, Partition};

use ::std::mem;
use ::itertools::Itertools;
use ::ordered_float::NotNan;

use ::rsp2_array_types::{V3, dot};

/// Return type of `find_layers`.
#[derive(Debug, PartialEq, Clone)]
pub enum Layers {
    /// Common case.  N layers, with N gaps between them.
    PerUnitCell(LayersPerUnitCell),

    /// The structure has no distinct layers.  Maybe you could say it
    /// has "one" layer and no gaps, but what makes this different from
    /// `PerUnitCell` is that there will continue to be one layer even
    /// if you took a supercell.
    ///
    /// Basically: It's an edge case that you may need to worry about.
    NoDistinctLayers { sorted_indices: Vec<usize> },

    /// The structure is empty.
    NoAtoms,
}

/// The common case for `Layers`, indicating at least one distinct layer per unit cell.
///
/// # Guarantees
///
/// Barring modification to the public data members:
///
/// * `groups.len() == gaps.len()`
/// * The last index of any group and the first index of the next are the two
///   atoms that define the layer sep. (even when considering the last group, whose
///   "next" group is the first group)
/// * The indices within each group are arranged to appear contiguous
///   when projected along the axis.
/// * The groups are sorted by the location of the *last* atom in the group,
///   mapped into the unit cell.
/// * This should be implied by some of the above points, but just to be clear:
///   If a layer happens to cross the periodic boundary, then it will be
///   the first layer listed in `groups`, and it will list the atoms whose
///   reduced coordinates are closer to 1 before those closer to zero.
/// * **Gaps are allowed to be zero or even negative.**  However, if constructed
///   using the variant of `find_layers` that does not accept labels, they will
///   all be strictly positive.
///
/// In hindsight it might have been better to put the last gap at index 0, so
/// that the gaps properly correspond to insertion indices. (i.e. the gap at
/// index 1 ought to lie between layers 0 and 1. Currently it lies between
/// layers 1 and 2)...
#[derive(Debug, PartialEq, Clone)]
pub struct LayersPerUnitCell {
    pub groups: Vec<Vec<usize>>,
    pub gaps: Vec<f64>,
}

impl Layers {
    /// Maps the common case to Some and the edge cases to None.
    pub fn per_unit_cell(self) -> Option<LayersPerUnitCell>
    { match self {
        Layers::PerUnitCell(layers) => Some(layers),
        _ => None,
    }}

    pub fn by_atom(&self) -> Vec<usize>
    { match self {
        Layers::PerUnitCell(layers) => layers.by_atom(),
        Layers::NoDistinctLayers { sorted_indices } => vec![0; sorted_indices.len()],
        Layers::NoAtoms => vec![],
    }}
}

impl LayersPerUnitCell {
    pub fn len(&self) -> usize { self.groups.len() }

    /// NOTE: If you modify the public members this may give nonsensical
    ///       results or panic.
    pub fn by_atom(&self) -> Vec<usize> {
        let n_atoms = self.groups.iter().map(|x| x.len()).sum();

        let mut out = vec![0xf00d; n_atoms];
        for (layer, group) in self.groups.iter().enumerate() {
            for &i in group {
                out[i] = layer;
            }
        }
        out
    }
}

// -------------------------------------------------------------

// FIXME this is wrong wrong wrong.
//       Only correct when the other two lattice vectors are
//         perpendicular to the chosen lattice vector.
//       May need to rethink the API.
//
// NOTE probably can fix this by treating 'normal' as a miller index.
//      (i.e. fractional in units of the *reciprocal* vectors)
//
/// Determine layers in a structure, numbered from zero.
/// Also returns the count.
///
/// This finds "layers" defined as groups of atoms isolated from
/// other groups by at least some minimum distance projected along
/// a normal vector.
///
/// Normal is in fractional coords, and is currently limited such
/// that it must be one of the lattice vectors.
///
/// `Layers` produced by this method will satisfy the property that
/// all gaps are `> 0` and `<=` the periodic length of the normal axis.
pub fn find_layers(structure: &Coords, normal: V3<i32>, cart_threshold: f64)
-> Result<Layers, Error>
{
    find_layers_impl::<()>(
        &structure.to_fracs(),
        structure.lattice(),
        normal,
        cart_threshold,
        None,
    )
}

/// Construct Layers from predetermined site layers (represented by any Ord type).
///
/// Basically, this returns the same sort of object returned by `find_layers`,
/// except that the groups assigned will correspond precisely to the provided indices
/// (not necessarily in the same order).
///
/// # Advantage over `find_layers`
///
/// This function can support layers that are "bent".  E.g. layers that curve up
/// and down, like this: (profile view)
///
/// ```text
///                ____        ____
///               /    \      /    \
/// Layer 2 \____/ ____ \____/ ____
///               /    \      /    \
/// Layer 1 \____/      \____/
/// ```
///
/// Of course, because it requires the layer indices, you need to obtain them separately
/// (e.g. via a bond graph).
///
/// # Why even bother?
///
/// Why even bother, given that you clearly already must have layer indices?  Well, the
/// reason is that Layers includes some information not easily obtained from the layer
/// indices alone, such as:
///
/// * The cartesian gap distances.  (some of which may now be negative, but you can still
///   readily identify vacuum separation)
/// * The fact that the layers are sorted.
/// * It distinguishes between the case of a layered material with one layer (with/without
///   vacuum), versus the case of an unlayered material (e.g. diamond).
///
pub fn find_layers_with_labels<L: Ord>(
    labels: &[L],
    structure: &Coords,
    normal: V3<i32>,
    threshold: f64,
) -> Result<Layers, Error>
{
    find_layers_impl(
        &structure.to_fracs(),
        structure.lattice(),
        normal,
        threshold,
        Some(labels)
    )
}

fn find_layers_impl<L: Ord>(
    fracs: &[V3<f64>],
    lattice: &Lattice,
    normal: V3<i32>,
    cart_threshold: f64,
    labels: Option<&[L]>,
) -> Result<Layers, Error>
{Ok({
    if fracs.len() == 0 {
        return Ok(Layers::NoAtoms);
    }

    // NOTE: the validity of the following algorithm is
    //       predicated on two things:
    //
    //  * The normal points precisely along a lattice vector.
    //    (This requirement can perhaps be eased through the use of
    //    unimodular transforms.)
    //
    //  * The normal must be orthogonal to the other lattice vectors.
    //
    // This ensures that there's no funny business where the projected
    // distance along the axis could suddenly change as a particle crosses
    // a periodic surface while traveling within a layer.
    //
    // Fail if these conditions are not met.
    let axis = require_simple_axis_normal(normal, lattice)?;

    // Transform into units used by the bulk of the algorithm:
    let periodic_length = lattice.norms()[axis];
    let frac_values = fracs.iter().map(|f| f[axis]).collect::<Vec<_>>();
    let frac_threshold = cart_threshold / periodic_length;

    // Perform the bulk of the algorithm
    let layers = assign_layers_impl_frac_1d(&frac_values, frac_threshold, labels)?;

    // Convert units back
    match layers {
        Layers::NoAtoms => unreachable!(),
        Layers::NoDistinctLayers { sorted_indices } => {
            Layers::NoDistinctLayers { sorted_indices }
        },
        Layers::PerUnitCell(layers) => {
            let LayersPerUnitCell { groups, gaps } = layers;
            let gaps = gaps.into_iter().map(|x| x * periodic_length).collect();
            Layers::PerUnitCell(LayersPerUnitCell { groups, gaps })
        },
    }
})}

// NOTE: All the warnings and fixmes about being "wrong wrong wrong"
//       do not apply to this algorithm, which solves a simpler problem.
//
// Given a sequence of positions `x`, each of which has periodic images
// with a period of 1, identify the layers that exist per unit cell.
fn assign_layers_impl_frac_1d<L: Ord>(
    positions: &[f64],
    threshold: f64,
    labels: Option<&[L]>,
) -> Result<Layers, Error>
{
    // dispatch to completely different algorithms; not much code can be reused
    match labels {
        Some(labels) => assign_layers_impl_frac_1d_with_labels(positions, threshold, labels),
        None => assign_layers_impl_frac_1d_no_labels(positions, threshold),
    }
}

fn assign_layers_impl_frac_1d_no_labels(
    positions: &[f64],
    threshold: f64,
) -> Result<Layers, Error>
{Ok({
    assert!(!positions.is_empty());

    let reduce = |x: f64| (x.fract() + 1.0).fract();
    assert_eq!(reduce(-1e-30), 0.0, "just making sure...");

    let sorted: Vec<(usize, f64)> = {
        let mut vec: Vec<_> = {
            positions.iter().cloned().map(reduce).enumerate().collect()
        };

        vec.sort_by_key(|&(_, x)| NotNan::new(x).unwrap());
        vec
    };

    // Split the positions into contiguous segments of values
    // where the difference between any two consecutive values
    // is at most `threshold`.
    //
    // Don't worry yet about the possibility that a segment
    // straddles the periodic boundary; we'll handle that later.

    // The first position begins a segment.
    let mut groups = vec![];
    let mut cur_group = vec![sorted[0].0];
    let mut layer_seps = vec![];

    // Find large separations after the first position.
    for (&(_, ax), &(bi, bx)) in sorted.iter().tuple_windows() {
        // Invariant
        assert!(!cur_group.is_empty());

        let sep = bx - ax;
        if sep > threshold {
            let done = mem::replace(&mut cur_group, vec![]);
            groups.push(done);
            layer_seps.push(sep);
        }
        cur_group.push(bi);
    }

    // At this point, the final group has not yet been committed,
    // because we have yet to consider the size of the separation
    // after the last value.
    // This is the separation that crosses the periodic boundary.
    assert!(!cur_group.is_empty());

    // Compute the sep and decide whether the final group is a new
    // segment, or if these values really belong to the first segment.
    {
        let sep = {
            let first = sorted.first().unwrap().1;
            let last_image = sorted.last().unwrap().1 - 1.0;
            first - last_image
        };
        if sep <= threshold {
            // Try to join with the first group...
            match groups.first_mut() {
                // Edge case: this IS the first group!
                None => return Ok(Layers::NoDistinctLayers { sorted_indices: cur_group }),

                Some(first_group) => {
                    // Put the new indices first so that they're in contiguous order.
                    cur_group.extend(first_group.drain(..));

                    assert_eq!(mem::replace(first_group, cur_group), Vec::<usize>::new());
                },
            }
        } else {
            // separate groups
            groups.push(cur_group);
            layer_seps.push(sep);
        }
    }

    // Sanity check, to guard against bugs of the `-1e-30 % 1.0 (== 1.0)` variety.
    // Notice that separations CAN be equal to 1, but not 0, which is interestingly
    //  reversed from most half-open domains.
    assert!(layer_seps.iter().all(|&x| 0.0 < x && x <= 1.0));
    println!("A: {:?}", groups.iter().map(|xs| xs.iter().map(|&i| positions[i]).collect_vec()).collect_vec());

    Layers::PerUnitCell(LayersPerUnitCell { groups, gaps: layer_seps })
})}

fn assign_layers_impl_frac_1d_with_labels<L: Ord>(
    positions: &[f64],
    threshold: f64,
    labels: &[L],
) -> Result<Layers, Error>
{Ok({
    use std::mem::replace;
    use std::iter::once;

    assert!(!positions.is_empty());

    let reduce = |x: f64| (x.fract() + 1.0).fract();
    assert_eq!(reduce(-1e-30), 0.0, "just making sure...");

    let positions = positions.iter().map(|&x| reduce(x)).collect_vec();

    let part = Part::from_ord_keys(labels);
    let indices = (0..positions.len()).collect_vec();
    let mut parted_indices = indices.into_unlabeled_partitions(&part).collect_vec();
    println!("{:?}", parted_indices);

    // Sort within each layer.
    // Track which ones cross the PBC boundary.
    let mut found_layer_without_gap = false;
    let mut parted_crosses_boundary = vec![];
    for part_indices in &mut parted_indices {
        println!("{:?}", part_indices.iter().map(|&i| positions[i]).collect::<Vec<_>>());
        let perm = argsort_floats(part_indices.iter().map(|&i| positions[i]));
        let new_indices = replace(part_indices, vec![]).permuted_by(&perm);
        *part_indices = new_indices;

        // Fix the arrangement within layers that cross the PBC boundary
        // (of which there may be multiple):
        //
        // If we consider the distance along the normal from each atom to the next
        // (including from the last atom to the next image of the first),
        // there should be at most one gap larger than the threshold.
        //
        // That gap should be moved to the end of the array, so that the atoms are
        // arranged contiguously.
        let mut splits = {
            part_indices.iter().map(|&i| positions[i])
                .chain(once(positions[part_indices[0]] + 1.0))
                // successive differences
                .tuple_windows().map(|(a, b)| b - a)
                // enumerate starting from 1, since the first difference
                // is the gap at insertion index 1 (i.e. between elements 0 and 1).
                .enumerate().map(|(i, d)| (i + 1, d))
                .filter(|&(_, d)| d > threshold)
                .map(|(i, _)| i)
                .collect_vec().into_iter() // unborrow part_indices
        };
        if let Some(split) = splits.next() {
            if let Some(_) = splits.next() {
                bail!("\
                    A layer was found with multiple large gaps; \
                    unable to determine where it starts!\
                ");
            }
            parted_crosses_boundary.push(split != part_indices.len());
            part_indices.rotate_left(split);
        } else {
            found_layer_without_gap = true;
        }
    }

    if found_layer_without_gap {
        ensure!{
            parted_indices.len() == 1, "\
                Found a segment with no gap when there is more than one layer. \
                This is not supported.\
            "
        }
        let sorted_indices = parted_indices.pop().unwrap();
        return Ok(Layers::NoDistinctLayers { sorted_indices });
    }

    // arrange the layers to increase in their terminal positions
    {
        let perm = argsort_floats({
            parted_indices.iter().map(|idxs| positions[*idxs.last().unwrap()])
        });
        parted_indices = parted_indices.permuted_by(&perm);
        parted_crosses_boundary = parted_crosses_boundary.permuted_by(&perm);
    }

    let gaps = {
        // this is surprisingly tricky compared to the no_labels case.
        (1..parted_indices.len())
            .map(|layer| {
                let first = positions[parted_indices[layer][0]];
                let last = positions[*parted_indices[layer - 1].last().unwrap()];
                // If this layer crosses the boundary, then we actually want an image of
                // its first atom.
                let first_image = first + if parted_crosses_boundary[layer] { -1.0 } else { 0.0 };
                first_image - last
            })
            .chain(once({
                // The 0th gap, which we skipped in the iterator.
                // Similar deal, but now we also need an image of the previous layer.
                let first = positions[parted_indices[0][0]];
                let last_image = positions[*parted_indices.last().unwrap().last().unwrap()] - 1.0;
                let first_image = first + if parted_crosses_boundary[0] { -1.0 } else { 0.0 };
                println!("B: {}, {}, {}", first, last_image, first - last_image);
                first_image - last_image
            }))
            .collect()
    };
    println!("B: {:?}", parted_indices.iter().map(|xs| xs.iter().map(|&i| positions[i]).collect_vec()).collect_vec());
    let groups = parted_indices;

    Layers::PerUnitCell(LayersPerUnitCell { groups, gaps })
})}

// -------------------------------------------------------------

fn argsort_floats(xs: impl IntoIterator<Item=f64>) -> Perm {
    Perm::argsort(&xs.into_iter().map(|x| NotNan::new(x).unwrap()).collect_vec())
}

// -------------------------------------------------------------

pub fn require_simple_axis_normal(normal: V3<i32>, lattice: &Lattice) -> Result<usize, Error> {
    let axis = {
        let mut sorted = normal;
        sorted.sort_unstable();
        ensure!(sorted == V3([0, 0, 1]),
            "unsupported layer normal: {:?}", normal);

        normal.iter().position(|&x| x == 1).unwrap()
    };

    let norms = lattice.norms();
    let vecs = lattice.vectors();
    for k in 0..3 {
        if k != axis {
            let cos = dot(&vecs[k], &vecs[axis]) / (norms[k] * norms[axis]);
            ensure!(cos.abs() < 1e-7,
                "due to a design flaw, assign_layers is currently limited to \
                lattices where the normal is perpendicular to the other two \
                lattice vectors.");
        }
    }
    Ok(axis)
}

// -------------------------------------------------------------

impl Permute for Layers {
    fn permuted_by(self, perm: &Perm) -> Self {
        match self {
            Layers::NoAtoms => Layers::NoAtoms,
            Layers::NoDistinctLayers { mut sorted_indices } => {
                for x in &mut sorted_indices {
                    *x = perm.permute_index(*x);
                }
                Layers::NoDistinctLayers { sorted_indices }
            },
            Layers::PerUnitCell(layers) => {
                Layers::PerUnitCell(layers.permuted_by(perm))
            },
        }
    }
}

impl Permute for LayersPerUnitCell {
    fn permuted_by(self, perm: &Perm) -> Self {
        let LayersPerUnitCell { mut groups, gaps } = self;
        for group in &mut groups {
            for x in group {
                *x = perm.permute_index(*x);
            }
        }
        LayersPerUnitCell { groups, gaps }
    }
}

// -------------------------------------------------------------

impl LayersPerUnitCell {
    pub fn get_part(&self) -> Part<usize> {
        let parted_indices = self.groups.iter().cloned().enumerate().collect();
        Part::new(parted_indices).unwrap()
    }

    /// Partition into structures where each layer's structure has contiguous
    /// cartesian coordinates along the normal axis. (be aware this means that images
    /// will be taken!)
    pub fn partition_into_contiguous_layers(
        &self,
        normal: V3<i32>,
        mut coords: Coords,
    ) -> Vec<Coords> {
        let axis = {
            require_simple_axis_normal(normal, coords.lattice())
                .expect("method has not been updated to support other normal vectors")
        };

        { // scope fracs_mut()
            let fracs = coords.fracs_mut();

            // reduce into first unit cell to make most layers contiguous
            for v in &mut fracs[..] {
                // (note: this might produce exactly 1.0)
                v[axis] -= f64::floor(v[axis]);
            }

            // Due to the type "invariants," (scare quotes due to public fields)
            // only the first layer is capable of crossing the periodic boundary,
            // and all misplaced images must be at the beginning.  Ideally, if we
            // looked at successive differences within the layer, we would see a
            // bunch of small positive values with a single large negative value.
            //
            // (however, we can't say this must be EXACTLY true, because `structure`
            //  might have gone through some small numerical changes in the time
            //  between the LayersPerUnitCell was computed and this method was called.
            //  We'll have to settle for "approximately" true.)
            let first_group = &self.groups[0];
            let diffs = {
                first_group.windows(2)
                    .map(|w| fracs[w[1]][axis] - fracs[w[0]][axis])
                    .collect::<Vec<_>>()
            };
            use ::std::f64::INFINITY;
            let min_diff = diffs.iter().cloned().fold(INFINITY, f64::min);
            let min_diff_pos = diffs.iter().position(|&x| x == min_diff).unwrap();

            // (min_diff is the index of the atom *before* the big negative difference
            //  meaning it is the last one we want to touch)
            for &atom in &first_group[..=min_diff_pos] {
                fracs[atom][axis] -= 1.0;
            }

            // sanity check.  All remaining diffs should be "pretty much nonnegative".
            let new_min = {
                first_group.windows(2)
                    .map(|w| fracs[w[1]][axis] - fracs[w[0]][axis])
                    .fold(INFINITY, f64::min)
            };
            assert!(
                -1e-1 < new_min,
                "(BUG) either a bug occurred in partition_into_contiguous_layers, \
                or it was given a bad structure. (new_diff = {})", new_min);
        } // scope fracs_mut()

        let part = self.get_part();
        let vec = coords.into_unlabeled_partitions(&part).collect::<Vec<_>>();
        vec
    }
}

// -------------------------------------------------------------

#[cfg(test)]
#[deny(unused)]
mod tests {
    use super::*;
    use ::rsp2_soa_ops::{Permute, Perm};
    use ::rsp2_array_types::Envee;

    fn shuffle<T: Clone>(xs: &[T]) -> (Vec<T>, Perm)
    {
        let xs = xs.to_vec();
        let perm = Perm::random(xs.len());
        (xs.permuted_by(&perm), perm)
    }

    fn scale(v: &[f64], fac: f64) -> Vec<f64> { v.iter().map(|&x| x * fac).collect() }

    #[test]
    fn layer_separation_eq_one() {
        // Perhaps surprisingly, fractional layer separations are
        // actually in the range `0.0 < sep <= 1.0`. And a value equal
        // to 1.0 is not unlikely for a single layer structure recently
        // read from an input file, so let's make sure it works.
        let fracs = vec![
            [0.00, 0.25, 0.5],
            [0.25, 0.00, 0.5],
        ].envee();
        let lattice = Lattice::eye();

        // deliberately test using exact equality; the periodic length
        // is 1.0 so there should be no rounding difficulties.
        assert_eq!(
            super::find_layers_impl::<()>(&fracs, &lattice, V3([0, 0, 1]), 0.15, None).unwrap(),
            Layers::PerUnitCell(LayersPerUnitCell {
                groups: vec![vec![0, 1]],
                gaps: vec![1.0], // <-- field of greatest interest
            }),
        );
    }

    #[test]
    fn find_layers_impl() {
        let fracs = vec![
            // we will be looking along y with frac_tol = 0.11
            [0.0, 0.1, 0.0],
            [0.0, 0.2, 0.0], // obviously same layer
            [0.8, 0.3, 0.4], // laterally displaced, but still same layer
            [0.0, 0.7, 0.0], // a second layer
            [0.0, 0.8, 0.0],
            // (first and last not close enough to meet)
        ].envee();

        // NOTE this does try a non-diagonal lattice and even
        //      goes along an awkwardly oriented vector
        //       but we restrict it to a form that the
        //       (currently broken) algorithm will work on
        //      (1st and 3rd vecs must be orthogonal to 2nd vec)
        let ylen = 10.0;                // periodic length
        let cart_tol    = 0.11 * ylen;  // produces 2 layers
        let smaller_tol = 0.09 * ylen;  // makes all atoms look separate

        const IR2: f64 = ::std::f64::consts::FRAC_1_SQRT_2;
        let lattice = Lattice::from(&[
            [ ylen * IR2, ylen *  IR2,  0.0],
            [ ylen * IR2, ylen * -IR2,  0.0], // (axis we're using)
            [        0.0,         0.0, 13.0],
        ]);

        fn check(
            (fracs, lattice, normal, cart_tol): (&[V3], &Lattice, V3<i32>, f64),
            expected_layers: Layers,
            expected_by_atom: Vec<usize>,
        ) {
            // exercise the wildly different codepaths between labeled and unlabeled
            let layers_1 = super::find_layers_impl(fracs, lattice, normal, cart_tol, None::<&[()]>).unwrap();
            let layers_2 = super::find_layers_impl(fracs, lattice, normal, cart_tol, Some(&expected_by_atom)).unwrap();

            // assert_eq, but be careful with floats
            for (layers, message) in vec![(layers_1, "no_labels"), (layers_2, "with_labels")] {
                match (&layers, &expected_layers) {
                    (Layers::PerUnitCell(actual), Layers::PerUnitCell(expected)) => {
                        // seps should be tested with approximate equality
                        let LayersPerUnitCell { groups: actual_groups, gaps: actual_gaps } = actual;
                        let LayersPerUnitCell { groups: expected_groups, gaps: expected_gaps } = expected;
                        assert_eq!(actual_groups, expected_groups, "error in {}", message);
                        assert_close!(abs=1e-13, actual_gaps, expected_gaps, "error in {}", message);
                    },
                    (actual, expected) => assert_eq!(actual, expected, "error in {}", message),
                }
                assert_eq!(layers.by_atom(), expected_by_atom, "error in {}", message);
            }
        }

        // --------------------------------------
        // test cases

        // FIXME: These tests are too stateful; attempting to reuse this much data
        //        between tests makes it difficult to reason about what each test
        //        case is actually testing.

        check(
            (&fracs, &lattice, V3([0, 1, 0]), cart_tol),
            Layers::PerUnitCell(LayersPerUnitCell {
                groups: vec![vec![0, 1, 2], vec![3, 4]],
                gaps: scale(&[0.4, 0.3], ylen),
            }),
            vec![0, 0, 0, 1, 1],
        );

        // put them out of order
        let (fracs, perm) = shuffle(&fracs);

        check(
            (&fracs, &lattice, V3([0, 1, 0]), cart_tol),
            Layers::PerUnitCell(LayersPerUnitCell {
                groups: vec![vec![0, 1, 2], vec![3, 4]],
                gaps: scale(&[0.4, 0.3], ylen),
            }).permuted_by(&perm),
            vec![0, 0, 0, 1, 1].permuted_by(&perm),
        );

        // try a smaller tolerance
        check(
            (&fracs, &lattice, V3([0, 1, 0]), smaller_tol),
            Layers::PerUnitCell(LayersPerUnitCell {
                groups: vec![vec![0], vec![1], vec![2], vec![3], vec![4]],
                gaps: scale(&[0.1, 0.1, 0.4, 0.1, 0.3], ylen),
            }).permuted_by(&perm),
            vec![0, 1, 2, 3, 4].permuted_by(&perm),
        );

        // try bridging across the periodic boundary to
        //   join the two layers.
        // also, try a position outside the unit cell.
        let (mut fracs, mut perm) = (fracs, perm);
        fracs.push(V3([0.0, 1.9, 0.0]));
        fracs.push(V3([0.0, 0.0, 0.0]));
        perm.append_mut(&Perm::eye(2));

        check(
            (&fracs, &lattice, V3([0, 1, 0]), cart_tol),
            Layers::PerUnitCell(LayersPerUnitCell {
                groups: vec![
                    vec![
                        3 /* y = 0.7 */, 4 /* y = 0.8 */, 5 /* y = 0.9 */,
                        6 /* y = 0.0 */, 0 /* y = 0.1 */, 1 /* y = 0.2 */,
                        2 /* y = 0.3 */,
                    ],
                ],
                gaps: scale(&[0.4], ylen),
            }).permuted_by(&perm),
            vec![0, 0, 0, 0, 0, 0, 0].permuted_by(&perm),
        );

        // try joining the end regions when there are distinct layers
        fracs.push(V3([0.0, 0.5, 0.0]));
        perm.append_mut(&Perm::eye(1));

        check(
            (&fracs, &lattice, V3([0, 1, 0]), cart_tol),
            Layers::PerUnitCell(LayersPerUnitCell {
                groups: vec![
                    vec![
                        3 /* y = 0.7 */, 4 /* y = 0.8 */, 5 /* y = 0.9 */,
                        6 /* y = 0.0 */, 0 /* y = 0.1 */, 1 /* y = 0.2 */,
                        2 /* y = 0.3 */,
                    ], vec![
                        7 /* y = 0.5 */,
                    ],
                ],
                gaps: scale(&[0.2, 0.2], ylen),
            }).permuted_by(&perm),
            vec![0, 0, 0, 0, 0, 0, 0, 1].permuted_by(&perm),
        );

        // try merging all layers into one continuous blob for
        // the NoDistinctLayers case
        fracs.push(V3([0.0, 0.4, 0.0]));
        fracs.push(V3([0.0, 0.6, 0.0]));
        perm.append_mut(&Perm::eye(2));

        check(
            (&fracs, &lattice, V3([0, 1, 0]), cart_tol),
            Layers::NoDistinctLayers {
                sorted_indices: vec![
                    6 /* y = 0.0 */, 0 /* y = 0.1 */, 1 /* y = 0.2 */,
                    2 /* y = 0.3 */, 8 /* y = 0.4 */, 7 /* y = 0.5 */,
                    9 /* y = 0.6 */, 3 /* y = 0.7 */, 4 /* y = 0.8 */,
                    5 /* y = 0.9 */,
                ],
            }.permuted_by(&perm),
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0].permuted_by(&perm),
        );
    }

    #[test]
    fn overlapping_layers() {
        const FRAC_TOL: f64 = 0.21;
        const LATTICE_MATRIX: [[f64; 3]; 3] = [
            // gobbledygook; anything invertible
            [ 4.0, 1.0, 2.0],
            [ 2.0, 3.0, 7.0],
            [ 0.0, 6.0, 8.0],
        ];

        let zlen = Lattice::from(&LATTICE_MATRIX).norms()[2];

        fn check(points: &[(char, f64)], expected_layers: Layers) {
            let normal = V3([0, 0, 1]);
            let lattice = Lattice::from(&LATTICE_MATRIX);
            let cart_tol = lattice.norms()[2] * FRAC_TOL;
            let (labels, frac_zs): (Vec<_>, Vec<_>) = points.into_iter().cloned().unzip();
            let fracs = frac_zs.into_iter().map(|z| V3([0.0, 0.0, z])).collect_vec();

            // try multiple permutations to reduce false negatives
            for _ in 0..8 {
                let perm = Perm::random(points.len());
                let fracs = fracs.to_vec().permuted_by(&perm);
                let expected_layers = expected_layers.clone().permuted_by(&perm);

                // Only check the algorithm that supports labels
                let layers = super::find_layers_impl(&fracs, &lattice, normal, cart_tol, Some(&labels)).unwrap();

                // assert_eq, but be careful with floats
                match (&layers, &expected_layers) {
                    (Layers::PerUnitCell(actual), Layers::PerUnitCell(expected)) => {
                        // seps should be tested with approximate equality
                        let LayersPerUnitCell { groups: actual_groups, gaps: actual_gaps } = actual;
                        let LayersPerUnitCell { groups: expected_groups, gaps: expected_gaps } = expected;
                        assert_eq!(actual_groups, expected_groups);
                        assert_close!(abs=1e-13, actual_gaps, expected_gaps);
                    },
                    (actual, expected) => assert_eq!(actual, expected),
                }
            }
        }

        // --------------------------------------
        // test cases

        // Make sure they're supported, for one...
        check(
            &[('A', 0.4), ('A', 0.6), ('B', 0.5), ('B', 0.7)],
            Layers::PerUnitCell(LayersPerUnitCell {
                groups: vec![vec![0, 1], vec![2, 3]],
                gaps: scale(&[-0.1, 0.7], zlen),
            }),
        );

        // Make sure Ord impl of labels doesn't matter
        check(
            &[('B', 0.4), ('B', 0.6), ('A', 0.5), ('A', 0.7)],
            Layers::PerUnitCell(LayersPerUnitCell {
                groups: vec![vec![0, 1], vec![2, 3]],
                gaps: scale(&[-0.1, 0.7], zlen),
            }),
        );

        // make them straddle the boundary in various funky ways to make life
        // miserable for the gap-computing code
        check(
            // A straddles
            &[('A', 0.9), ('A', 0.1), ('B', 0.01), ('B', 0.20)],
            Layers::PerUnitCell(LayersPerUnitCell {
                groups: vec![vec![0, 1], vec![2, 3]],
                gaps: scale(&[-0.09, 0.70], zlen),
            }),
        );

        check(
            // B straddles
            &[('A', 0.8), ('A', 0.99), ('B', 0.90), ('B', 0.10)],
            Layers::PerUnitCell(LayersPerUnitCell {
                groups: vec![vec![2, 3], vec![0, 1]],
                gaps: scale(&[0.70, -0.09], zlen),
            }),
        );

        check(
            // Both straddle
            &[('A', 0.85), ('A', 0.05), ('B', 0.95), ('B', 0.15)],
            Layers::PerUnitCell(LayersPerUnitCell {
                groups: vec![vec![0, 1], vec![2, 3]],
                gaps: scale(&[-0.1, 0.7], zlen),
            }),
        );

        // evidence that all gaps can be negative
        check(
            &[
                ('A', 0.5), ('A', 0.7), ('A', 0.9), ('A', 0.1),
                ('B', 0.0), ('B', 0.2), ('B', 0.4), ('B', 0.6),
            ],
            Layers::PerUnitCell(LayersPerUnitCell {
                groups: vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]],
                gaps: scale(&[-0.1, -0.1], zlen),
            }),
        );
    }
}
