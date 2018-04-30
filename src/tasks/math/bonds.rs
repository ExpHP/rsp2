use ::FailResult;
use ::rsp2_structure::supercell;
use ::rsp2_structure::{CoordStructure, Structure, Lattice};
use ::rsp2_array_utils::{arr_from_fn, try_map_arr};

use ::rsp2_array_types::{V3, M3, dot};

#[derive(Serialize, Deserialize)]
#[derive(Debug, Clone, PartialEq)]
pub struct Bonds {
    from: Vec<usize>,
    to: Vec<usize>,
    // FIXME: Rather than the cartesian vectors (which change as the structure
    //        relaxes), we should keep the `[i32; 3]` image indices.
    //        Then a function could be provided that computes the V3s from a
    //        structure, assuming the bonds haven't changed.
    cart_vectors: Vec<V3>,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Bond<V = V3> {
    pub from: usize,
    pub to: usize,
    pub cart_vector: V,
}

//=================================================================

pub trait VeclikeIterator: ExactSizeIterator + DoubleEndedIterator {}

impl<I> VeclikeIterator for I where I: ExactSizeIterator + DoubleEndedIterator {}

pub type Iter<'a> = Box<VeclikeIterator<Item = Bond<&'a V3>> + 'a>;
impl<'a> IntoIterator for &'a Bonds {
    type Item = Bond<&'a V3>;
    type IntoIter = Iter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        Box::new(izip!(
            self.from.iter().cloned(),
            self.to.iter().cloned(),
            self.cart_vectors.iter(),
        ).map(|(from, to, cart_vector)| Bond { from, to, cart_vector }))
    }
}

//=================================================================

impl Bonds {
    pub fn len(&self) -> usize
    { self.from.len() }

    pub fn from_brute_force_very_dumb<M>(
        structure: &Structure<M>,
        range: f64,
    ) -> FailResult<Self> {
        Self::_from_brute_force_very_dumb(
            structure.map_metadata_to(|_| ()),
            range,
        )
    }

    // monomorphic
    fn _from_brute_force_very_dumb(
        structure: CoordStructure,
        range: f64,
    ) -> FailResult<Self> {

        // Construct a supercell large enough to contain all atoms that interact with an atom
        // in the centermost unit cell.
        let sc_builder = sufficiently_large_centered_supercell(structure.lattice(), range)?;
        let (superstructure, sc_info) = sc_builder.build(structure);
        let centermost_cell = V3([0, 0, 0]);

        let mut from = vec![];
        let mut to = vec![];
        let mut cart_vectors = vec![];

        let carts = superstructure.to_carts();
        let cells = sc_info.signed_cell_indices();
        let sites = sc_info.primitive_site_indices();

        for (cell_from, &site_from, &cart_from) in izip!(cells, &sites, &carts) {
            if cell_from != centermost_cell {
                continue;
            }

            for (&site_to, &cart_to) in izip!(&sites, &carts) {
                let vector = cart_to - cart_from;
                if vector.sqnorm() < range * range {
                    if site_from == site_to {
                        continue;
                    }
                    from.push(site_from);
                    to.push(site_to);
                    cart_vectors.push(vector);
                }
            }
        }
        assert_ne!(from.len(), 0, "(BUG) nothing in center cell?");
        Ok(Bonds { from, to, cart_vectors })
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
    //   `a` and `b`. When this ray exits the unit cell, the particular face of the
    //   periodic boundary that it will exit through will be one spanned by `a` and `b`.
    //   (that is, the ray will be normal to the surface that it passes through)
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
                // (numerical fuzz doesn't matter here; there are no ties because
                //  we cannot possibly touch the paralellepiped at a vertex)
                dist(&good) < f64::min(dist(&bad1), dist(&bad2)),
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
            trace!("bond graph: intermediate supercell: {:?}", coeffs);
            distances
        }
        Err(_) => {
            // Pick a larger cell with uniform scaling.
            coeffs = [*coeffs.iter().max().unwrap(); 3];
            trace!("bond graph: taking uniform intermediate supercell: {:?}", coeffs);

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
