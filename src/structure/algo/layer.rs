use ::Result;
use ::{Structure, Lattice};
use ::{Permute, Perm};

use ::std::mem;
use ::itertools::Itertools;
use ::ordered_float::NotNaN;

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
/// Guarantees (barring modification to the public data members):
///
/// * `groups.len() == gaps.len()`
/// * `gaps[i]` is the gap between `groups[i]` and `groups[(i + 1) % groups.len()]`.
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
    { match *self {
        Layers::PerUnitCell(ref layers) => layers.by_atom(),
        Layers::NoDistinctLayers { ref sorted_indices } => vec![0; sorted_indices.len()],
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
/// Determine layers in a structure, numbered from zero.
/// Also returns the count.
///
/// This finds "layers" defined as groups of atoms isolated from
/// other groups by at least some minimum distance projected along
/// a normal vector.
///
/// Normal is in fractional coords, and is currently limited such
/// that it must be one of the lattice vectors.
pub fn find_layers<M>(structure: &Structure<M>, normal: &[i32; 3], threshold: f64)
-> Result<Layers>
{
    find_layers_impl(
        &structure.to_fracs(),
        structure.lattice(),
        normal,
        threshold,
    )
}

// monomorphic for less codegen
fn find_layers_impl(fracs: &[[f64; 3]], lattice: &Lattice, normal: &[i32; 3], cart_threshold: f64)
-> Result<Layers>
{Ok({
    if fracs.len() == 0 {
        return Ok(Layers::NoAtoms);
    }

    let axis = {
        let mut sorted = *normal;
        sorted.sort_unstable();
        ensure!(sorted == [0, 0, 1],
            "unsupported layer normal: {:?}", normal);

        normal.iter().position(|&x| x == 1).unwrap()
    };

    // FIXME: On second thought I think this is incorrect.
    //        Our requirement should not be that the normal is a
    //        lattice vector; but rather, that two of the lattice
    //        vectors lie within the plane.
    //
    //        So let's make sure that is the case:
    { // Safety HACK!
        use ::rsp2_array_utils::dot;
        let lengths = lattice.lengths();
        let vecs = lattice.matrix();
        for k in 0..3 {
            if k != axis {
                let cos = dot(&vecs[k], &vecs[axis]) / (lengths[k] * lengths[axis]);
                ensure!(cos.abs() < 1e-7,
                    "For your safety, assign_layers is currently limited to \
                    lattices where the normal is perpendicular to the other two \
                    lattice vectors.");
            }
        }
    }

    // --(original (incorrect) text)--
    // NOTE: the validity of the following algorithm is
    //       predicated on the normal pointing precisely along
    //       a lattice vector.  This ensures that there's no
    //       funny business where the projected distance along the
    //       axis could suddenly change as a particle crosses a
    //       periodic surface while traveling within a layer.
    //
    //       Some other directions with integer coordinates
    //       could be handled in the future by a unimodular
    //       transform to make that direction become one of the
    //       lattice vectors....In theory.
    // --(end original text)--

    // Transform into units used by the bulk of the algorithm:
    let periodic_length = lattice.lengths()[axis];
    let frac_values = fracs.iter().map(|f| f[axis]).collect::<Vec<_>>();
    let frac_threshold = cart_threshold / periodic_length;

    // Perform the bulk of the algorithm
    let layers = assign_layers_impl_frac_1d(&frac_values, frac_threshold)?;

    // Convert units back
    match layers {
        Layers::NoAtoms => Layers::NoAtoms,
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
fn assign_layers_impl_frac_1d(
    positions: &[f64],
    threshold: f64,
) -> Result<Layers>
{Ok({
    let reduce = |x: f64| (x.fract() + 1.0).fract();
    assert_eq!(reduce(-1e-30), 0.0, "just making sure...");

    let sorted: Vec<(usize, f64)> = {
        let mut vec: Vec<_> = {
            positions.iter().cloned().map(reduce).enumerate().collect()
        };

        vec.sort_by_key(|&(_, x)| NotNaN::new(x).unwrap());
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

                    assert_eq!(mem::replace(first_group, cur_group), vec![]);
                },
            }
        } else {
            // separate groups
            groups.push(cur_group);
            layer_seps.push(sep);
        }
    }

    // Sanity check, to guard against bugs of the `-1e-30 % 1.0 (== 1.0)` variety.
    assert!(layer_seps.iter().all(|&x| 0.0 <= x && x < 1.0));

    Layers::PerUnitCell(LayersPerUnitCell { groups, gaps: layer_seps })
})}


// The indices contained within are mapped by `perm.inverted()`,
// so that the output of `by_atoms` transforms according to `perm`.
impl Permute for Layers {
    fn permuted_by(self, perm: &Perm) -> Self {
        match self {
            Layers::NoAtoms => Layers::NoAtoms,
            Layers::NoDistinctLayers { mut sorted_indices } => {
                let inv_perm = perm.inverted().into_vec();
                for x in &mut sorted_indices {
                    *x = inv_perm[*x] as usize;
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
        let inv_perm = perm.inverted().into_vec();

        let LayersPerUnitCell { mut groups, gaps } = self;
        for group in &mut groups {
            for x in group {
                *x = inv_perm[*x] as usize;
            }
        }
        LayersPerUnitCell { groups, gaps }
    }
}

#[cfg(test)]
#[deny(unused)]
mod tests {
    use super::*;
    use ::{Permute, Perm};

    #[test]
    fn assign_layers_impl() {
        let fracs = vec![
            // we will be looking along y with frac_tol = 0.11
            [0.0, 0.1, 0.0],
            [0.0, 0.2, 0.0], // obviously same layer
            [0.8, 0.3, 0.4], // laterally displaced, but still same layer
            [0.0, 0.7, 0.0], // a second layer
            [0.0, 0.8, 0.0],
            // (first and last not close enough to meet)
        ];

        // NOTE this does try a non-diagonal lattice and even
        //      goes along an awkwardly oriented vector
        //       but we restrict it to a form that the
        //       (currently broken) algorithm will work on
        //      (1st and 3rd vecs must be orthogonal to 2nd vec)
        let ylen = 4.0;                 // periodic length
        let cart_tol    = 0.11 * ylen;  // produces 2 layers
        let smaller_tol = 0.09 * ylen;  // makes all atoms look separate

        const IR2: f64 = ::std::f64::consts::FRAC_1_SQRT_2;
        let lattice = Lattice::new(&[
            [ ylen * IR2, ylen *  IR2,  0.0],
            [ ylen * IR2, ylen * -IR2,  0.0], // (axis we're using)
            [        0.0,         0.0, 13.0],
        ]);

        fn check(
            (fracs, lattice, normal, cart_tol): (&[[f64; 3]], &Lattice, &[i32; 3], f64),
            expected_layers: Layers,
            expected_by_atom: Vec<usize>,
        ) {
            let layers = super::find_layers_impl(fracs, lattice, normal, cart_tol).unwrap();
            let by_atom = layers.by_atom();

            // assert_eq, but be careful with floats
            match (layers, expected_layers) {
                (Layers::PerUnitCell(actual), Layers::PerUnitCell(expected)) => {
                    // seps should be tested with approximate equality
                    let LayersPerUnitCell { groups: actual_groups, gaps: actual_gaps } = actual;
                    let LayersPerUnitCell { groups: expected_groups, gaps: expected_gaps } = expected;
                    assert_eq!(actual_groups, expected_groups);
                    for (a, b) in ::util::zip_eq(actual_gaps, expected_gaps) {
                        assert_close!(abs=1e-13, a, b);
                    }
                },
                (actual, expected) => assert_eq!(actual, expected),
            }
            assert_eq!(by_atom, expected_by_atom);
        }

        let scale = |v: Vec<f64>, fac: f64| v.into_iter().map(|x| x * fac).collect();

        // --------------------------------------
        // test cases

        check(
            (&fracs, &lattice, &[0, 1, 0], cart_tol),
            Layers::PerUnitCell(LayersPerUnitCell {
                groups: vec![vec![0, 1, 2], vec![3, 4]],
                gaps: scale(vec![0.4, 0.3], ylen),
            }),
            vec![0, 0, 0, 1, 1],
        );

        // put them out of order
        let (fracs, perm) = ::oper::perm::shuffle(&fracs);

        check(
            (&fracs, &lattice, &[0, 1, 0], cart_tol),
            Layers::PerUnitCell(LayersPerUnitCell {
                groups: vec![vec![0, 1, 2], vec![3, 4]],
                gaps: scale(vec![0.4, 0.3], ylen),
            }).permuted_by(&perm),
            vec![0, 0, 0, 1, 1].permuted_by(&perm),
        );

        // try a smaller tolerance
        check(
            (&fracs, &lattice, &[0, 1, 0], smaller_tol),
            Layers::PerUnitCell(LayersPerUnitCell {
                groups: vec![vec![0], vec![1], vec![2], vec![3], vec![4]],
                gaps: scale(vec![0.1, 0.1, 0.4, 0.1, 0.3], ylen),
            }).permuted_by(&perm),
            vec![0, 1, 2, 3, 4].permuted_by(&perm),
        );

        // try bridging across the periodic boundary to
        //   join the two layers.
        // also, try a position outside the unit cell.
        let (mut fracs, mut perm) = (fracs, perm);
        fracs.push([0.0, 1.9, 0.0]);
        fracs.push([0.0, 0.0, 0.0]);
        perm.append_mut(&Perm::eye(2));

        check(
            (&fracs, &lattice, &[0, 1, 0], cart_tol),
            Layers::PerUnitCell(LayersPerUnitCell {
                groups: vec![
                    vec![
                        3 /* y = 0.7 */, 4 /* y = 0.8 */, 5 /* y = 0.9 */,
                        6 /* y = 0.0 */, 0 /* y = 0.1 */, 1 /* y = 0.2 */,
                        2 /* y = 0.3 */,
                    ],
                ],
                gaps: scale(vec![0.4], ylen),
            }).permuted_by(&perm),
            vec![0, 0, 0, 0, 0, 0, 0].permuted_by(&perm),
        );

        // try joining the end regions when there are distinct layers
        fracs.push([0.0, 0.5, 0.0]);
        perm.append_mut(&Perm::eye(1));

        check(
            (&fracs, &lattice, &[0, 1, 0], cart_tol),
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
                gaps: scale(vec![0.2, 0.2], ylen),
            }).permuted_by(&perm),
            vec![0, 0, 0, 0, 0, 0, 0, 1].permuted_by(&perm),
        );

        // try merging all layers into one continuous blob for
        // the NoDistinctLayers case
        fracs.push([0.0, 0.4, 0.0]);
        fracs.push([0.0, 0.6, 0.0]);
        perm.append_mut(&Perm::eye(2));

        check(
            (&fracs, &lattice, &[0, 1, 0], cart_tol),
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
}
