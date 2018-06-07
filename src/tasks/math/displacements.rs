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

use ::math::stars::Stars;

use ::rsp2_tasks_config as cfg;

use ::rsp2_newtype_indices::{Idx, Indexed, IndexVec};
use ::rsp2_array_types::{V3, M3};
use ::rsp2_structure::{Coords, Lattice, IntRot};
use ::rsp2_soa_ops::{Perm, Permute};

lazy_static! {
    /// Comparable to Phonopy with `DIAG = .False.`.
    static ref DIRECTIONS_AXIAL: Vec<V3<i32>> = vec![
        V3([1, 0, 0]),
        V3([0, 1, 0]),
        V3([0, 0, 1]),
    ];
    /// Comparable to Phonopy with `DIAG = .TRUE.`.
    static ref DIRECTIONS_DIAG_1: Vec<V3<i32>> = make_nice_directions_list(1);
    /// Experimental "cleverer" list.
    static ref DIRECTIONS_DIAG_2: Vec<V3<i32>> = make_nice_directions_list(2);
}

#[allow(unused)] // useful for debugging
struct FmtDisp(V3<i32>);
impl ::std::fmt::Display for FmtDisp {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        let sign = |x: i32| if x < 0 { "-" } else { " " };
        macro_rules! fmt { ($e:expr) => (format_args!("{}{}", sign($e), $e.abs())); }
        let [a, b, c] = *self.0;
        write!(f, "[{},{},{}]", fmt!(a), fmt!(b), fmt!(c))
    }
}

pub fn compute_displacements(
    cfg: &cfg::PhononDispFinderRsp2Directions,
    int_rots: impl IntoIterator<Item=IntRot>,
    stars: &Stars,
    coords: &Coords,
    amplitude: f64,
) -> Vec<(usize, V3)> {
    let int_rots = int_rots.into_iter().collect::<IndexVec<usize, _>>();

    let go = |choices: &[_]| {
        _compute_displacements::<usize, _, _, _>(
            choices, &int_rots[..], stars, coords.lattice(), amplitude,
        ).raw
    };

    match cfg {
        cfg::PhononDispFinderRsp2Directions::Axial => go(&DIRECTIONS_AXIAL),
        cfg::PhononDispFinderRsp2Directions::Diag  => go(&DIRECTIONS_DIAG_1),
        cfg::PhononDispFinderRsp2Directions::Diag2 => go(&DIRECTIONS_DIAG_2),
        cfg::PhononDispFinderRsp2Directions::Survey => {
            debug!("Surveying displacement implementations:");
            debug!("  axial: Produces {}", go(&DIRECTIONS_AXIAL).len());
            debug!("   diag: Produces {}", go(&DIRECTIONS_DIAG_1).len());
            debug!(" diag-2: Produces {}", go(&DIRECTIONS_DIAG_2).len());
            go(&DIRECTIONS_DIAG_2)
        },
    }
}

fn _compute_displacements<DispI: Idx, SiteI: Idx, OperI: Idx, StarI: Idx>(
    choices: &[V3<i32>], // possible directions in descending order of niceness
    int_rots: &Indexed<OperI, [IntRot]>,
    stars: &Stars<SiteI, OperI, StarI>,
    lattice: &Lattice,
    amplitude: f64,
) -> IndexVec<DispI, (SiteI, V3)> {
    // Our goal is to have data for every atom being displaced along three linearly independent
    // axes, in both + and - directions.

    let mut out = IndexVec::new();

    // In practice, we only ever need to displace star representatives.
    // (After computing their force constants, we can easily derive the rest by symmetry.)
    for star in &**stars {

        // Each possible displacement at the representatives may have others that can be produced
        // by applying a symmetry op which maps the representative into itself.
        let site_symmetry_opers = star.opers_from_rep(star.representative());
        let site_symmetry = || site_symmetry_opers.iter().map(|&oper| int_rots[oper]);

        // Sort the choices by decreasing number of unique displacements produced.
        let counts: Vec<_> = {
            choices.iter()
                .map(|&v| {
                    let count = independent_rotation_count(choices, site_symmetry(), v);
                    usize::checked_sub(6, count).unwrap() // descending sort order
                })
                .collect()
        };

        // `Perm::argsort` is a stable sort, so ties will be broken by niceness.
        let perm = Perm::argsort(&counts);
        let mut best_choices = choices.to_vec().permuted_by(&perm).into_iter();

        // `current_basis` tracks the axes that our choices account for, in order to exclude
        // from future consideration less-nice choices from the same "star" of directions.
        // (and anything else linearly dependent with our choices)
        //
        // The loop logic here is precisely as awkward as it needs to be, though it will never run
        // more than three times.  Each iteration, 1-3 vectors are added to `current_basis`.
        let mut current_basis = vec![];
        'three: while current_basis.len() < 3 {
            // Find best remaining choice that isn't redundant.
            let choice = {
                // Strictly speaking, choices from previous iterations of the `'three` loop could
                // render an arbitrary number of directions from each "star" of directions
                // redundant, requiring us to re-sort the choices to find the new next best choice.
                //
                // However, I can hardly imagine a scenario where that would actually lead to us
                // making a suboptimal selection. The only time it could even *theoretically* happen
                // is on the second iteration of the 'three loop.
                //
                // So we just greedily take the first linearly independent choice without
                // worrying about the linear dependence of its starmates.
                let mut choice = best_choices.next().expect("(BUG) ran out of choices!");
                while !is_lindep_with(&current_basis, choice) {
                    choice = best_choices.next().expect("(BUG) ran out of choices!");
                }
                choice
            };

            let (new_positives, has_negatives) = independent_rotations(choices, site_symmetry(), choice);

            // Add only the best disp from this "star" of displacements to the output.
            let cart = {
                let lattice_point_cart = choice.map(|x| x as f64) * lattice;
                amplitude * lattice_point_cart.unit()
            };
            out.push((star.representative(), cart));
            if !has_negatives.0 {
                out.push((star.representative(), -cart));
            }

            // Update `current_basis`.
            for positive in new_positives {
                if !is_lindep_with(&current_basis, positive) {
                    // (this edge case is a prerequisite for the theoretical "suboptimal selection"
                    // scenario described in the big comment above. But *even then*, it does not
                    // guarantee that our choices were suboptimal)
                    debug!("(I wonder if this edge case ever happens. Hello? Can anyone hear me?)");
                    continue;
                }
                current_basis.push(positive);
            }
        }
    }
    out
}

// When true, the site symmetry at a point produces negative versions of each vector.
struct HasNegatives(bool);

fn independent_rotation_count(
    positive_choices: &[V3<i32>],
    // operators that map a site into itself
    site_symmetry: impl IntoIterator<Item=IntRot>,
    v: V3<i32>,
) -> usize {
    match independent_rotations(positive_choices, site_symmetry, v) {
        (positives, HasNegatives(true))  => 2 * positives.len(),
        (positives, HasNegatives(false)) => positives.len(),
    }
}

fn independent_rotations(
    positive_choices: &[V3<i32>],
    // operators that map a site into itself
    site_symmetry: impl IntoIterator<Item=IntRot>,
    v: V3<i32>,
) -> (Vec<V3<i32>>, HasNegatives) {
    assert!(positive_choices.contains(&v));

    let mut out = vec![v];
    let mut has_negatives = false;

    for rot in site_symmetry {
        let rotated = rot * v;

        // as soon as we see `-v` we know that positive and negative versions will appear
        // for all vectors in the output. (provable from the group properties of site symmetry)
        if rotated == -v {
            has_negatives = true;
        }

        let rotated_is_choice = positive_choices.contains(&rotated);
        let negated_is_choice = positive_choices.contains(&-rotated);
        let positive = match (rotated_is_choice, negated_is_choice) {
            (true, true) => panic!("(BUG) linearly_independent_rotations: bad choices"),
            (false, false) => rotated,
            (true, false) => rotated,
            (false, true) => -rotated,
        };

        if is_lindep_with(&out, positive) {
            out.push(positive);
        }
        // (even if `out` has 3 elements, keep scanning in case we haven't seen a negative yet)
    }
    (out, HasNegatives(has_negatives))
}

//==================================================================================================

fn make_nice_directions_list(max_abs: i32) -> Vec<V3<i32>> {
    let mut directions = vec![];
    let mut ranks = vec![];
    for a in -max_abs ..= max_abs {
        for b in -max_abs ..= max_abs {
            for c in -max_abs ..= max_abs {
                // Ultimately only the direction of a point matters, so e.g. [2, 0, 2] is
                // effectively the same as [1, 0, 1].  Therefore, skip if gcd(a, b, c) != 1.
                assert!(max_abs == 1 || max_abs == 2, "simplifying assumption of gcd check");
                if (a & 1) + (b & 1) + (c & 1) == 0 {
                    continue
                }

                let direction = V3([a, b, c]);
                directions.push(direction);

                // Tuple whose lexical ordering places "nicer" directions first.
                ranks.push((
                    direction.iter().filter(|&&n| n.abs() == 2).count(), // prefer fewer 2s
                    direction.iter().filter(|&&n| n.abs() == 1).count(), // prefer fewer 1s
                    direction.iter().filter(|&&n| n < 0).count(), // prefer fewer minus signs
                    direction.map(|n| n < 0), // prefer minus signs near back, not front
                    [c.abs(), b.abs(), a.abs()], // prefer larger numbers near front, not back
                ))
            }
        }
    }
    // Sort by niceness.
    directions = directions.permuted_by(&Perm::argsort(&ranks));

    // For each pair of a vector and its negative, only keep the nicest one.
    for nice_i in 0.. {
        // (NOTE: not 0..directions.len() because the length of `directions` changes)
        if nice_i >= directions.len() { break; }

        for ugly_i in nice_i + 1..directions.len() {
            if directions[ugly_i] == -directions[nice_i] {
                directions.remove(ugly_i);
                break;
            }
        }
    }
    directions
}

fn is_lindep_with(vs: &[V3<i32>], v: V3<i32>) -> bool {
    match vs {
        &[]     => true,
        &[a]    => a != v && a != -v,
        &[a, b] => M3([a, b, v]).det() != 0,
        &_      => false,
    }
}
