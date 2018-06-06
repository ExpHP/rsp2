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

use ::FailResult;
use ::rsp2_structure::supercell;
use ::rsp2_structure::{Coords, Lattice};
use ::rsp2_array_utils::{arr_from_fn, try_map_arr};

use ::rsp2_array_types::{V3, M3, dot};

/// Bond data in a more widely-reusable form.
///
/// The following actions will invalidate the bonds:
///
/// * removal, addition, or reordering of sites
/// * mapping of coordinates into the unit cell
/// * unimodular transformations of the lattice
///
/// The following actions are okay:
///
/// * motion of atoms, even if they cross cell boundaries,
///   (so long as relative distances do not change enough that the
///    bonds WOULD change)
/// * cartesian transformations of the lattice (preserving frac coords)
///
#[derive(Serialize, Deserialize)]
#[derive(Debug, Clone, PartialEq)]
pub struct FracBonds {
    // used for sanity checks
    num_atoms: usize,

    // NOTE: Multiple bonds may have the same `(from, to)` pair, for interaction
    //       with multiple images!
    from: Vec<usize>,
    to: Vec<usize>,

    // Rather than the cartesian vectors (which change as the structure relaxes),
    // we keep differences in image index (as cell_to - cell_from).
    //
    // This also saves us from constantly having to worry about "nearest images",
    // as long as the positions don't get reduced.
    image_diff: Vec<V3<i32>>,
}

#[derive(Serialize, Deserialize)]
#[derive(Debug, Clone, PartialEq)]
pub struct CartBonds {
    num_atoms: usize, // used for sanity checks
    from: Vec<usize>,
    to: Vec<usize>,
    cart_vector: Vec<V3<f64>>,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FracBond {
    pub from: usize,
    pub to: usize,
    pub image_diff: V3<i32>,
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct CartBond {
    pub from: usize,
    pub to: usize,
    pub cart_vector: V3,
}

//=================================================================

pub trait VeclikeIterator: ExactSizeIterator + DoubleEndedIterator {}

impl<I> VeclikeIterator for I where I: ExactSizeIterator + DoubleEndedIterator {}

pub type FracIter<'a> = Box<VeclikeIterator<Item = FracBond> + 'a>;
impl<'a> IntoIterator for &'a FracBonds {
    type Item = FracBond;
    type IntoIter = FracIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        Box::new(izip!(
            self.from.iter().cloned(),
            self.to.iter().cloned(),
            self.image_diff.iter().cloned(),
        ).map(|(from, to, image_diff)| FracBond { from, to, image_diff }))
    }
}

pub type CartIter<'a> = Box<VeclikeIterator<Item = CartBond> + 'a>;
impl<'a> IntoIterator for &'a CartBonds {
    type Item = CartBond;
    type IntoIter = CartIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        Box::new(izip!(
            self.from.iter().cloned(),
            self.to.iter().cloned(),
            self.cart_vector.iter().cloned(),
        ).map(|(from, to, cart_vector)| CartBond { from, to, cart_vector }))
    }
}

//=================================================================

pub trait BondRange: Copy {
    fn minimum(self) -> f64;
    fn maximum(self) -> f64;
}

impl BondRange for f64 {
    fn minimum(self) -> f64 { 0.0 }
    fn maximum(self) -> f64 { self }
}

impl BondRange for (f64, f64) {
    fn minimum(self) -> f64 { self.0 }
    fn maximum(self) -> f64 { self.1 }
}

//=================================================================

impl FracBonds {
    pub fn len(&self) -> usize
    { self.from.len() }
}

impl CartBonds {
    pub fn len(&self) -> usize
    { self.from.len() }
}

impl FracBonds {
    pub fn from_brute_force_very_dumb(
        coords: &Coords,
        range: impl BondRange,
    ) -> FailResult<Self> {

        // Construct a supercell large enough to contain all atoms that interact with an atom
        // in the centermost unit cell.
        let sc_builder = sufficiently_large_centered_supercell(coords.lattice(), range.maximum())?;
        let (superstructure, sc) = sc_builder.build(coords);
        let centermost_cell = sc.lattice_point_from_cell(sc.center_cell());

        let mut from = vec![];
        let mut to = vec![];
        let mut image_diff = vec![];

        let carts = superstructure.to_carts();
        let cells = sc.atom_lattice_points();
        let sites = sc.atom_primitive_atoms();

        for (&cell_from, &site_from, &cart_from) in izip!(&cells, &sites, &carts) {
            if cell_from != centermost_cell {
                continue;
            }

            for (&cell_to, &site_to, &cart_to) in izip!(&cells, &sites, &carts) {
                let sqnorm = (cart_to - cart_from).sqnorm();
                let square = |x| x*x;
                if square(range.minimum()) <= sqnorm && sqnorm <= square(range.maximum()) {
                    if (site_from, cell_from) == (site_to, cell_to) {
                        continue;
                    }
                    from.push(site_from);
                    to.push(site_to);
                    image_diff.push(cell_to - cell_from);
                }
            }
        }
        assert_ne!(from.len(), 0, "(BUG) nothing in center cell?");
        let num_atoms = coords.num_atoms();
        Ok(FracBonds { num_atoms, from, to, image_diff })
    }

    pub fn to_cart_bonds(&self, coords: &Coords) -> CartBonds {
        let FracBonds { num_atoms, ref from, ref to, ref image_diff } = *self;
        let from = from.to_vec();
        let to = to.to_vec();

        // (NOTE: we'd also get ruined by reordering of coordinates or mapping into
        //        the unit cell; but those are too difficult to test)
        assert_eq!(num_atoms, coords.num_atoms(), "number of atoms has changed!");
        let lattice = coords.lattice();
        let carts = coords.to_carts();

        let cart_vector = {
            zip_eq!(&from, &to, image_diff)
                .map(|(&from, &to, image_diff)| {
                    let cart_image_diff = image_diff.map(|x| x as f64) * lattice;
                    carts[to] - carts[from] + cart_image_diff
                })
                .collect()
        };
        CartBonds { num_atoms, from, to, cart_vector }
    }
}

//=================================================================

// Construct a supercell large enough to contain all atoms that interact with an atom
// in the unit cell with coeffs [0, 0, 0]
//
// FIXME: even for how retardedly conservative this idea is for large unit cells,
//        it's still VERY HARD to actually implement this function correctly!
//        (Just as with all things PBC, general lattices continue to be a nightmare
//         when it comes to finding "sufficiently large search volumes")
//
//        We took the easy way out for nothing!
fn sufficiently_large_centered_supercell(
    lattice: &Lattice,
    mut interaction_range: f64,
) -> FailResult<supercell::Builder> {
    // Search for a slightly larger range to account for numerical fuzz.
    interaction_range *= 1.0 + 1e-4;

    let get_permuted_vectors = |vectors: &[_; 3], k| (
        vectors[(k + 0) % 3],
        vectors[(k + 1) % 3],
        vectors[(k + 2) % 3],
    );

    let get_scaled_vectors = |coeffs: [u32; 3]| {
        let mut vectors = lattice.vectors().clone();
        for k in 0..3 {
            vectors[k] *= coeffs[k] as f64;
        }
        vectors
    };

    // Bail out on skewed structures, because it's unclear how to handle them.
    //
    // Specifically, this implementation is limited to cells with the following property
    // (henceforth the 'small-skew' property):
    //
    // * Take any two of the lattice vectors; call them `a` and `b`.
    //   Draw a ray from the center of the unit cell which is perpendicular to both
    //   `a` and `b`. When this ray exits the unit cell, it will touch the face of the
    //   periodic boundary spanned by `a` and `b`. (i.e. the one it is normal to)
    //
    // The key advantage of such a cell is that the closest points on the parallelepiped
    // surface to the cell center must necessarily be among those six "normal ray"
    // intersection points. (FIXME: proof? Seems obviously true if you picture it though)
    //
    // (TODO: can it be shown that a Delaunay or Niggli cell meets this criteria?)
    let check_plane_distances = |vectors: &[V3; 3]| {

        let matrix = M3(*vectors);
        let origin = V3([0.0; 3]) * &matrix;
        let center = V3([0.5; 3]) * &matrix;

        try_map_arr([0, 1, 2], |k| {
            use self::geometry::PointNormalPlane as Plane;
            use self::geometry::ParametricLine as Line;

            let (v, p, q) = get_permuted_vectors(vectors, k);

            // Planes touching the origin. (the other three planes are equivalent under inversion)
            let good = Plane::from_span_and_point((p, q), origin);
            let bad1 = Plane::from_span_and_point((v, p), origin);
            let bad2 = Plane::from_span_and_point((v, q), origin);

            // Line touching the cell center with unit direction.
            let line = Line { start: center, vector: good.normal };

            // I have no strategy for how to adjust supercell size when this property fails,
            // so verify that the correct surface is touched first.
            let dist = |plane| line.t_at_plane_intersection(plane).abs();
            ensure!(
                // NOTE: previously this used a strict `<` comparison, with the note:
                //
                //      (numerical fuzz doesn't matter here; there are no ties because
                //       we cannot possibly touch the paralellepiped at a vertex)
                //
                // I'm not sure why I thought this. It doesn't need to pass through a vertex for
                // there to be a tie (it can pass through an edge), and in fact it is easy to
                // construct a 'small-skew' lattice where this occurs:
                //
                //                     [[1 1 0], [0 1 0], [0 0 1]]
                //
                // (which is a useful lattice for test cases since it is unimodular)
                //                                                                   - ML
                dist(&good) <= f64::min(dist(&bad1), dist(&bad2)) * (1.0 + 1e-4),
                "cell is too skewed for bond search"
            );

            Ok(dist(&good))
        })
    };
    check_plane_distances(lattice.vectors())?;

    // If we expand the cell large enough that all of these normal intersection points
    // are at a distance `>= interaction_range` from the center, then we have guaranteed
    // that the cell includes all points in a sphere of interaction radius around the
    // cell center.

    // The small-skew property makes it safe to do this analysis per-axis and take a
    // diagonal supercell.
    let mut coeffs = arr_from_fn(|k| {
        // Get the part of this vector orthogonal to both other vectors.
        let (v, p, q) = get_permuted_vectors(lattice.vectors(), k);
        let v = v.par(&V3::cross(&p, &q));

        // Make sure the "height" of the parallelepiped can fit the sphere diameter.
        f64::ceil((2.0 * interaction_range) / v.norm()) as u32
    });

    // ...just one thing.  By picking different multiples for each cell vector,
    // we may have destroyed the small-skew property, and thus the shortest distance
    // to the periodic boundary may be shorter than what we targeted.
    //
    // ...I think.  Better safe than sorry, anyways.
    let distances = match check_plane_distances(&get_scaled_vectors(coeffs)) {
        Ok(distances) => {
            trace!("bond graph: intermediate supercell: {:?}, r = {}", coeffs, interaction_range);
            distances
        }
        Err(_) => {
            // Pick a larger cell with uniform scaling.
            coeffs = [*coeffs.iter().max().unwrap(); 3];
            trace!("bond graph: taking uniform intermediate supercell: {:?}, r = {}", coeffs, interaction_range);

            check_plane_distances(&get_scaled_vectors(coeffs))
                .expect("(bug) uniform supercell does not satisfy the property!?")
        }
    };

    // The supercell for 'coeffs' should now accommodate a sphere around the center.
    assert!(distances.iter().all(|&dist| interaction_range <= dist * (1.0 + 1e-4)));

    // But hold on!
    //
    // What we *really* need is to be able to fit a sphere around ANY point in the
    // central cell; not just the center.  Calling the supercell we just chose `S1`,
    // we can pick an even larger supercell `S2` which is guaranteed to include a
    // region shaped like `S1` around any point in the original unit cell.
    //
    // `(2 * c + 1)` cells (where `c` was the old number) should be enough.
    // Lucky for us, we automatically get this by building a centered supercell.
    trace!("bond graph: true supercell: centered_diagonal({:?})", coeffs);

    Ok(supercell::centered_diagonal(coeffs))
}

mod geometry {
    use super::*;

    /// Locus of points `x` in 3D space satisfying `dot(normal, x - point) == 0`.
    pub struct PointNormalPlane {
        pub point: V3,
        pub normal: V3,
    }

    /// The curve `x(t) = start + t * vector`.
    pub struct ParametricLine {
        pub start: V3,
        pub vector: V3,
    }

    impl PointNormalPlane {
        /// NOTE: The direction of the normal is arbitrarily chosen
        pub fn from_span_and_point((a, b): (V3, V3), point: V3) -> Self {
            PointNormalPlane {
                normal: V3::cross(&a, &b).unit(),
                point,
            }
        }
    }

    impl ParametricLine {
        /// Solve for the value of `t` where this line intersects a plane.
        pub fn t_at_plane_intersection(&self, plane: &PointNormalPlane) -> f64 {
            let numer = dot(&plane.normal, &(&plane.point - &self.start));
            let denom = dot(&plane.normal, &self.vector);
            numer / denom
        }
    }
}

//=================================================================

#[cfg(test)]
#[deny(unused)]
mod tests {
    use super::*;
    use ::rsp2_structure::CoordsKind;
    use ::std::collections::BTreeSet;
    use ::rsp2_array_types::Envee;

    #[test]
    fn self_interactions() {
        //   . . . . .
        //   . # # # .   o : central atom
        //   . # o # .   # : in range
        //   . # # # .   . : too far
        //   . . . . .
        let coords = Coords::new(
            Lattice::orthorhombic(1.0, 1.0, 2.0),
            CoordsKind::Carts(vec![V3::zero()]),
        );
        let range = f64::sqrt(2.0) * 1.1;

        let bonds = FracBonds::from_brute_force_very_dumb(&coords, range).unwrap();
        let actual = bonds.into_iter().collect::<BTreeSet<_>>();
        assert_eq!{
            actual,
            vec![
                // should include other images of the atom, but not the atom itself
                FracBond { from: 0, to: 0, image_diff: V3([ 1,  0, 0]) },
                FracBond { from: 0, to: 0, image_diff: V3([-1,  0, 0]) },
                FracBond { from: 0, to: 0, image_diff: V3([ 0,  1, 0]) },
                FracBond { from: 0, to: 0, image_diff: V3([ 0, -1, 0]) },
                FracBond { from: 0, to: 0, image_diff: V3([ 1,  1, 0]) },
                FracBond { from: 0, to: 0, image_diff: V3([ 1, -1, 0]) },
                FracBond { from: 0, to: 0, image_diff: V3([-1,  1, 0]) },
                FracBond { from: 0, to: 0, image_diff: V3([-1, -1, 0]) },
            ].into_iter().collect::<BTreeSet<_>>(),
        }
    }

    #[test]
    fn non_orthogonal_lattice() {
        // same physical scenario as `self_interactions`, but an equivalent cell is chosen
        let coords = Coords::new(
            Lattice::from([
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 2.0],
            ]),
            CoordsKind::Carts(vec![V3::zero()]),
        );
        let range = f64::sqrt(2.0) * 1.1;

        let bonds = FracBonds::from_brute_force_very_dumb(&coords, range).unwrap();
        let actual = bonds.into_iter().collect::<BTreeSet<_>>();
        
        // For frac vector v and lattice L:
        //                         [1 1 0]
        // L transformed as  L ->  [0 1 0] L,   and the cart vector (v L) must remain fixed,
        //                         [0 0 1]
        //                                 [1 -1  0]
        // so v transforms as   v.T -> v.T [0  1  0]   (subtract first element from second)
        //                                 [0  0  1]
        assert_eq!{
            actual,
            vec![
                FracBond { from: 0, to: 0, image_diff: V3([ 1, -1,  0]) },
                FracBond { from: 0, to: 0, image_diff: V3([-1,  1,  0]) },
                FracBond { from: 0, to: 0, image_diff: V3([ 0,  1,  0]) },
                FracBond { from: 0, to: 0, image_diff: V3([ 0, -1,  0]) },
                FracBond { from: 0, to: 0, image_diff: V3([ 1,  0,  0]) },
                FracBond { from: 0, to: 0, image_diff: V3([ 1, -2,  0]) },
                FracBond { from: 0, to: 0, image_diff: V3([-1,  2,  0]) },
                FracBond { from: 0, to: 0, image_diff: V3([-1,  0,  0]) },
            ].into_iter().collect::<BTreeSet<_>>(),
        }
    }

    #[test]
    #[should_panic(expected = "too skewed")]
    fn too_skewed() {
        let lattice = Lattice::from([
            [1.0, 3.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]);
        let coords = CoordsKind::Carts(vec![V3::zero()]);
        let coords = Coords::new(lattice, coords);
        FracBonds::from_brute_force_very_dumb(&coords, 1.2).unwrap();
    }

    #[test]
    fn both_cutoffs() {
        // Test on a linear chain of atoms, with both cutoffs enabled.
        //
        // There are three atoms per cell, and they each only interact with
        // the atoms at a distance of 2 away.
        let coords = Coords::new(
            Lattice::orthorhombic(10.0, 10.0, 3.0),
            CoordsKind::Carts(vec![
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 2.0],
            ].envee()),
        );
        let range = (1.5, 2.5);

        let bonds = FracBonds::from_brute_force_very_dumb(&coords, range).unwrap();
        let actual = bonds.into_iter().collect::<BTreeSet<_>>();
        assert_eq!{
            actual,
            vec![
                FracBond { from: 0, to: 2, image_diff: V3([0, 0,  0]) },
                FracBond { from: 0, to: 1, image_diff: V3([0, 0, -1]) },
                FracBond { from: 1, to: 2, image_diff: V3([0, 0, -1]) },
                FracBond { from: 1, to: 0, image_diff: V3([0, 0,  1]) },
                FracBond { from: 2, to: 0, image_diff: V3([0, 0,  0]) },
                FracBond { from: 2, to: 1, image_diff: V3([0, 0,  1]) },
            ].into_iter().collect::<BTreeSet<_>>(),
        }
    }
}
