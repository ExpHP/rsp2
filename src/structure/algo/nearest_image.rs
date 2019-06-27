use failure::Error;
use rsp2_array_types::V3;
use crate::Lattice;

/// A helper type for locating nearest images under periodic boundary conditions.
#[derive(Debug, Clone)]
pub struct NearestImageFinder {
    lattice: Lattice,
    lattice_vector_carts_around_origin: Vec<V3>,
}

impl NearestImageFinder {
    /// Construct a `NearestImageFinder` for a lattice.
    ///
    /// Requires that the lattice be of sufficiently small skew such that:
    ///
    /// * for any point in the unit cell...
    /// * ...the nearest image of any other point in the unit cell lies within the 27 cells
    ///   centered around the unit cell.
    ///
    /// (the precise criteria for this is not yet known; for now, lattices are simply rejected if
    ///  they "look troublesome")
    pub fn new(lattice: &Lattice) -> Result<Self, Error> {
        // NOTE: not sure if this precise condition is necessary/sufficient
        ensure!(!lattice.is_large_skew(1e-4), "cell is too skewed for image finding");

        let mut vecs = Vec::with_capacity(27);
        for &fa in &[-1.0, 0.0, 1.0] {
            for &fb in &[-1.0, 0.0, 1.0] {
                for &fc in &[-1.0, 0.0, 1.0] {
                    vecs.push(V3([fa, fb, fc]) * lattice);
                }
            }
        }

        Ok(NearestImageFinder {
            lattice: lattice.clone(),
            lattice_vector_carts_around_origin: vecs,
        })
    }

    /// Find the shortest possible vectors between periodic images.
    ///
    /// **Input:** A displacement vector between two arbitrary points.
    ///
    /// **Output:** `out` will contain the shortest images (up to a tolerance of `tol` in units of
    /// length) of that vector under the lattice.
    pub fn shortest_images_cart(&self, out: &mut Vec<V3>, cart: V3, tol: f64) {
        let frac = cart / &self.lattice;
        // Note: This doesn't need the second round of '% 1.0' because it's not floored modulus.
        //       (and even if it was, the algorithm is tolerant of coordinates exactly equal to 1)
        let frac = frac.map(|x| x % 1.0);
        let cart = frac * &self.lattice;
        self.shortest_images_cart_fast(out, cart, tol);
    }

    /// A faster version of `shortest_image_cart` provided that the input vector meets certain
    /// criteria.
    ///
    /// **Input:** A displacement vector from one point in the unit cell to another (where both
    /// points have been reduced into the cell prior to taking the difference).  That is to say,
    /// the vector must have fractional coordinates in the range `[-1, 1]`.
    ///
    /// **Output:** `out` will contain the shortest images (up to a tolerance of `tol` in units of
    /// length) of that vector under the lattice.
    #[inline(never)]
    pub fn shortest_images_cart_fast(&self, out: &mut Vec<V3>, cart: V3, tol: f64) {
        // norms instead of sqnorms for the sake of letting tol have units of length
        let mut norms = [0.0; 27];
        for i in 0..27 {
            norms[i] = (self.lattice_vector_carts_around_origin[i] + cart).norm();
        };
        let minimum = norms.iter().cloned().min_by(|a, b| f64::partial_cmp(a, b).unwrap()).unwrap();

        out.clear();
        for i in 0..27 {
            if norms[i] <= minimum + tol {
                out.push(self.lattice_vector_carts_around_origin[i] + cart);
            }
        }
        assert!(out.len() >= 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shortest_images() {
        let lattice = Lattice::diagonal(&[4.0, 8.0, 12.0]);
        // a reduced point
        let quarter_way = V3([1.0, -2.0, 3.0]);
        // image of quarter_way, valid input for _fast
        let quarter_way_nearby = V3([1.0, 6.0, -9.0]);
        // image of quarter_way, not valid input for _fast
        let quarter_way_far = V3([13.0, 30.0, 15.0]);
        // a point with many ties.
        let approx_center = V3([2.0 - 1e-7, -4.0 - 1e-6, 6.0]);
        let tol = 1e-5;

        let finder = NearestImageFinder::new(&lattice).unwrap();
        let mut out_buf = vec![];
        finder.shortest_images_cart_fast(&mut out_buf, quarter_way, tol);
        assert_eq!(out_buf, vec![quarter_way]);

        finder.shortest_images_cart_fast(&mut out_buf, quarter_way_nearby, tol);
        assert_eq!(out_buf, vec![quarter_way]);

        finder.shortest_images_cart(&mut out_buf, quarter_way_far, tol);
        assert_eq!(out_buf, vec![quarter_way]);

        finder.shortest_images_cart(&mut out_buf, approx_center, tol);
        assert_eq!(out_buf.len(), 8);
        for v in out_buf {
            assert_close!(abs=2e-5, v.norm(), approx_center.norm());
        }
    }
}
