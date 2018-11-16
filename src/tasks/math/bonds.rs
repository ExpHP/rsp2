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
use ::rsp2_soa_ops::{Perm, Permute};
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
#[serde(rename_all = "kebab-case")]
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
#[serde(rename_all = "kebab-case")]
pub struct CartBonds {
    num_atoms: usize, // used for sanity checks
    from: Vec<usize>,
    to: Vec<usize>,
    cart_vector: Vec<V3<f64>>,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FracBond {
    /// Source atom of this edge.
    pub from: usize,

    /// Target atom of this edge.
    pub to: usize,

    /// Determines which ghost of `to` is interacting with which ghost of `from`.
    ///
    /// Chosen to be measured relative to the actual positions that were given to the bond
    /// search (as opposed to the reduced positions).  Specifically, it is chosen such that
    ///
    /// ```text
    /// (original_coords[to] - original_coords[from]) + image_diff * lattice
    /// ```
    ///
    /// was the cartesian vector representing the bond at the time of construction.
    pub image_diff: V3<i32>,
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct CartBond {
    pub from: usize,
    pub to: usize,
    pub cart_vector: V3,
}

//=================================================================

impl FracBond {
    /// A function that returns `true` for one of the bonds representing
    /// a pair of interacting ghosts, and `false` for the other bond.
    ///
    /// The definition is otherwise unspecified.
    ///
    /// # Panics
    ///
    /// Panics on self-interactions with `image_diff = [0, 0, 0]`.
    /// (These never show up in `FracBonds` normally, but can be created through other means,
    ///  such as directly constructing one or by using `join`)
    pub fn is_canonical(&self) -> bool {
        match self.from.cmp(&self.to) {
            ::std::cmp::Ordering::Less => true,
            ::std::cmp::Ordering::Greater => false,
            ::std::cmp::Ordering::Equal => {
                for &x in &self.image_diff.0 {
                    if x != 0 {
                        return x > 0;
                    }
                }
                panic!("self interactions with zero image diff cannot be canonicalized");
            },
        }
    }

    /// Get the bond in the reverse direction.
    ///
    /// This will invert the output of `is_canonical()`.
    #[inline]
    pub fn flip(self) -> FracBond {
        FracBond {
            from: self.to,
            to: self.from,
            image_diff: -self.image_diff,
        }
    }

    /// If this bond is from `A` to `B` and the other bond is from `B` to `C`, construct
    /// a bond from `A` to `C` that has the correct `image_diff`.
    ///
    /// This can be useful for identifying groups of mutually interacting atoms in a manner
    /// that properly accounts for images.
    #[inline]
    pub fn join(self, other: FracBond) -> Option<FracBond> {
        if self.to == other.from {
            Some(FracBond {
                from: self.from,
                to: other.to,
                image_diff: self.image_diff + other.image_diff,
            })
        } else { None }
    }

    // NOTE: when working with types like PeriodicGraph, FracBonds::to_cart_bonds is not
    //       available. There was no convenient place to put a method for getting all of the cart
    //       vectors from a format like that, so this is a method for getting a single vector.
    //
    //       I went for an unambiguous signature over ergonomics or efficiency; hopefully, the
    //       branch can be elided at use sites.
    //
    // FIXME: If I ever actually do add separate types for Fracs and Carts, this should simply
    //        be of type (&Carts) -> V3.
    //
    /// Get the Cartesian vector, if the `coords` have cached Cartesian coordinates.
    #[inline] // hoping for optimizations that elide the branch
    pub fn cart_vector_using_cache(&self, coords: &Coords) -> Option<V3> {
        let FracBond { from, to, image_diff } = *self;
        coords.as_carts_cached().map(|carts| {
            let cart_image_diff = image_diff.map(|x| x as f64) * coords.lattice();
            carts[to] - carts[from] + cart_image_diff
        })
    }
}

//=================================================================

impl FracBonds {
    pub fn from_iter(num_atoms: usize, iter: impl IntoIterator<Item=FracBond>) -> Self {
        let iter = iter.into_iter();
        let size = iter.size_hint().0;
        let mut from = Vec::with_capacity(size);
        let mut to = Vec::with_capacity(size);
        let mut image_diff = Vec::with_capacity(size);

        for FracBond { from: f, to: t, image_diff: i } in iter {
            assert!(f < num_atoms);
            assert!(t < num_atoms);
            from.push(f);
            to.push(t);
            image_diff.push(i);
        }
        FracBonds { from, to, image_diff, num_atoms }
    }
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
    /// Compute bonds using a really dumb brute force strategy.
    ///
    /// This function was originally even called `from_brute_force_very_dumb`, because when
    /// I wrote it I was almost 100% certain it would have to be replaced with something
    /// faster eventually.
    ///
    /// ...it has survived surprisingly well.
    pub fn from_brute_force(
        original_coords: &Coords,
        range: impl BondRange,
    ) -> FailResult<Self> {
        let fake_meta = vec![(); original_coords.len()];
        Self::from_brute_force_with_meta(
            original_coords,
            (range.minimum(), range.maximum()),
            &fake_meta,
            |(), ()| range.maximum(),
        )
    }

    /// Compute bonds, using different bond lengths for different types.
    ///
    /// `meta_range` must be symmetric, i.e. `meta_range(a, b) == meta_range(b, a)`;
    /// this is not validated.
    pub fn from_brute_force_with_meta<M>(
        original_coords: &Coords,
        // The largest possible range needed. This will affect the size of the
        // supercell used to search for bonds.
        full_range: impl BondRange,
        meta: &[M],
        // (note: returning impl BondRange is not yet supported by rust)
        mut meta_range: impl FnMut(&M, &M) -> f64,
    ) -> FailResult<Self> {
        // Construct a supercell large enough to contain all atoms that interact with an atom
        // in the centermost unit cell, assuming they're all reduced.
        let sc_builder = {
            sufficiently_large_centered_supercell(original_coords.lattice(), full_range.maximum())?
        };

        // ...like I said; they gotta be reduced.
        let (reduced_coords, original_latts) = decompose_coords(original_coords);
        let (superstructure, sc) = sc_builder.build(&reduced_coords);

        let num_atoms = original_coords.num_atoms();
        assert_eq!(num_atoms, sc.num_primitive_atoms());

        let mut from = vec![];
        let mut to = vec![];
        let mut image_diff = vec![];

        let sc_centermost_latt = sc.lattice_point_from_cell(sc.center_cell());
        let sc_carts = superstructure.to_carts();
        let sc_latts = sc.atom_lattice_points();
        let sc_sites = sc.atom_primitive_atoms();

        // The supercell is large enough that we can disregard its periodicity, and consider
        // interactions between its centermost cell and any other atom.
        let mut num_visited = 0;
        for (&latt_from, &site_from, &cart_from) in izip!(&sc_latts, &sc_sites, &sc_carts) {
            if latt_from != sc_centermost_latt {
                continue;
            }
            num_visited += 1;

            for (&latt_to, &site_to, &cart_to) in izip!(&sc_latts, &sc_sites, &sc_carts) {
                let range = (full_range.minimum(), meta_range(&meta[site_from], &meta[site_to]));

                // FIXME: To ensure the result is symmetric, we rather precariously rely on the
                //        assumption that `a - b == -(b - a)` for all floating point numbers where
                //        the result is not NaN.  However, I do not believe this is true in all
                //        possible rounding modes.
                let sqnorm = (cart_to - cart_from).sqnorm();
                let square = |x| x*x;
                if square(range.0) <= sqnorm && sqnorm <= square(range.1) {
                    // No self interactions!
                    if (site_from, latt_from) == (site_to, latt_to) {
                        continue;
                    }
                    from.push(site_from);
                    to.push(site_to);

                    // `latt_to - latt_from` would give us the image diff between the images in our
                    // supercell, but that's computed from the reduced positions. We actually want
                    // the image diffs for the original positions.
                    let adjusted_latt_to = latt_to - original_latts[site_to];
                    let adjusted_latt_from = latt_from - original_latts[site_from];
                    image_diff.push(adjusted_latt_to - adjusted_latt_from);
                }
            }
        }
        let num_atoms = original_coords.num_atoms();
        assert_eq!(num_visited, num_atoms, "(BUG) wrong # atoms in center cell?");

        let out = FracBonds { num_atoms, from, to, image_diff };
        if cfg!(debug_assertions) {
            out.sanity_check(original_coords, full_range, meta, meta_range);
        }
        Ok(out)
    }

    #[inline]
    pub fn to_cart_bonds(&self, coords: &Coords) -> CartBonds {
        let FracBonds { num_atoms, ref from, ref to, .. } = *self;
        let from = from.to_vec();
        let to = to.to_vec();

        // (NOTE: we'd also get ruined by reordering of coordinates or mapping into
        //        the unit cell; but those are too difficult to test)
        assert_eq!(num_atoms, coords.num_atoms(), "number of atoms has changed!");
        let coords = coords.with_carts(coords.to_carts());
        let cart_vector = {
            self.into_iter().map(|v| v.cart_vector_using_cache(&coords).unwrap()).collect()
        };
        CartBonds { num_atoms, from, to, cart_vector }
    }

    #[allow(unused)]
    pub(crate) fn sanity_check<M>(
        &self,
        coords: &Coords,
        full_range: impl BondRange,
        meta: &[M],
        mut meta_range: impl FnMut(&M, &M) -> f64,
    ) {
        let square = |x| x*x;
        let cart_bonds = self.to_cart_bonds(coords);
        for CartBond { cart_vector, from, to } in &cart_bonds {
            let sqnorm = cart_vector.sqnorm();
            let min = full_range.minimum();
            let max = meta_range(&meta[from], &meta[to]).maximum();
            assert!(
                square(min) * (1.0 - 1e-9) <= sqnorm && sqnorm <= square(max) * (1.0 + 1e-9),
                "(BUG) bad bond length: {} vs ({}, {})",
                cart_vector.norm(), min, max,
            );
        }
    }

    pub fn num_atoms_per_cell(&self) -> usize
    { self.num_atoms }
}

// Split coords into reduced coordinates, and their original lattice points.
fn decompose_coords(coords: &Coords) -> (Coords, Vec<V3<i32>>) {
    let mut coords = coords.clone();
    coords.ensure_fracs();

    let original_fracs = coords.to_fracs();
    coords.reduce_positions();
    let mapped_fracs = coords.to_fracs();

    let latts = {
        izip!(original_fracs, mapped_fracs)
            .map(|(orig, mapped)| (orig - mapped).map(|x| f64::round(x) as i32))
            .collect()
    };
    (coords, latts)
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

impl Permute for FracBonds {
    fn permuted_by(self, perm: &Perm) -> Self {
        assert_eq!(self.num_atoms, perm.len());
        FracBonds {
            num_atoms: self.num_atoms,
            from: self.from.into_iter().map(|x| perm.permute_index(x)).collect(),
            to: self.to.into_iter().map(|x| perm.permute_index(x)).collect(),
            image_diff: self.image_diff,
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

        let bonds = FracBonds::from_brute_force(&coords, range).unwrap();
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

        let bonds = FracBonds::from_brute_force(&coords, range).unwrap();
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
        FracBonds::from_brute_force(&coords, 1.2).unwrap();
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

        let bonds = FracBonds::from_brute_force(&coords, range).unwrap();
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

    #[test]
    fn weird_initial_images() {
        // A very short-range interaction with only one interacting pair,
        // but we start with really weird images; this will be reflected in the
        // image_diff vector.
        let coords = Coords::new(
            Lattice::orthorhombic(1.0, 1.0, 1.0),
            CoordsKind::Carts(vec![
                // make sure to include negative values in case of dumb negative modulus bugs
                V3([6.5, -3.5,  2.5]),  // the [6, -4,  2] image of [0.5, 0.5, 0.5]
                V3([8.5, -3.5, -7.4]),  // the [8, -4, -8] image of [0.5, 0.5, 0.6]
            ]),
        );
        let range = 0.1 * 1.1;

        let bonds = FracBonds::from_brute_force(&coords, range).unwrap();
        let actual = bonds.into_iter().collect::<BTreeSet<_>>();
        assert_eq!{
            actual,
            vec![
                // Two crucial things:
                // * one bond should be found
                // * `image_diff` should be `-(image_to - image_from)`
                FracBond { from: 0, to: 1, image_diff: V3([-2, 0,  10]) },
                FracBond { from: 1, to: 0, image_diff: V3([ 2, 0, -10]) },
            ].into_iter().collect::<BTreeSet<_>>(),
        }
    }

    #[test]
    fn regression_nanotube() {
        let coords = Coords::new(
            Lattice::orthorhombic(100.0, 100.0, 22.510051727295),
            CoordsKind::Carts(vec![
                V3([4.125935,  0.464881, 8.843235]),
                V3([4.125935, -0.464881, 9.915142]),
            ]),
        );
        let range = 1.01 * 2.0;

        let bonds = FracBonds::from_brute_force(&coords, range).unwrap();
        let actual = bonds.into_iter().collect::<BTreeSet<_>>();
        assert_eq!{
            actual,
            vec![
                FracBond { from: 0, to: 1, image_diff: V3([0, 0, 0]) },
                FracBond { from: 1, to: 0, image_diff: V3([0, 0, 0]) },
            ].into_iter().collect::<BTreeSet<_>>(),
        }
    }
}
