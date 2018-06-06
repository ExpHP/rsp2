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

use ::rsp2_newtype_indices::{Idx, Indexed, IndexVec};
use ::rsp2_array_types::V3;
use ::rsp2_structure::{Coords, Lattice, IntRot};

/// Type with custom Ord impl
#[derive(Debug, Eq, PartialEq)]
struct AxialDisp(V3<i32>);

pub fn compute_displacements(
    int_rots: impl IntoIterator<Item=IntRot>,
    stars: &Stars,
    coords: &Coords,
    amplitude: f64,
) -> Vec<(usize, V3)> {
    let int_rots = int_rots.into_iter().collect::<IndexVec<usize, _>>();

    _compute_displacements::<usize, _, _, _>(
        &int_rots[..], stars, coords.lattice(), amplitude,
    ).raw
}

fn _compute_displacements<DispI: Idx, SiteI: Idx, OperI: Idx, StarI: Idx>(
    int_rots: &Indexed<OperI, [IntRot]>,
    stars: &Stars<SiteI, OperI, StarI>,
    lattice: &Lattice,
    amplitude: f64,
) -> IndexVec<DispI, (SiteI, V3)> {
    // Our goal is to have data for every atom being displaced in the direction of
    // each of the six lattice points around the origin.
    // (+ and - are both necessary to have enough data for approximating the second derivative)

    // In practice, we only ever need to displace star representatives.
    // (After computing their force constants, we can easily derive the rest by symmetry.)
    let mut out = vec![];
    for star in &**stars {
        // Use Vec as a set because it's easy to control the order of the output
        // (we just insert items into the vec in reverse order of preference),
        // and we only have six items anyways.
        let mut all_disps = Vec::with_capacity(6);
        for axis in (0..3).rev() {
            for &sign in &[-1, 1] {
                let mut disp = V3([0; 3]);
                disp[axis] = sign;
                all_disps.push(disp);
            }
        }

        while let Some(unique_disp) = all_disps.pop() {
            let cart_disp = {
                let lattice_point_cart = unique_disp.map(|x| x as f64) * lattice;
                amplitude * lattice_point_cart.unit()
            };
            // Only ever displace representatives (justification above).
            out.push((star.representative(), cart_disp));

            // Ignore any (atom, direction) pairs that can be obtained by applying symmetry ops.
            // (of course, since we only care about `atom = star.representative`, we just look at
            //  the ops that map the representative into itself)
            for &oper in star.opers_from_rep(star.representative()) {
                let derived_disp = unique_disp * int_rots[oper].matrix();

                if let Some(index) = all_disps.iter().position(|x| x == &derived_disp) {
                    all_disps.remove(index);
                }
            }
        }
    }
    Indexed::from_raw(out)
}
