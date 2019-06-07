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

use std::ops::{Mul, Div};
use std::sync::Arc;
#[cfg(feature = "serde")]
use serde::{Serialize, Serializer, Deserialize, Deserializer};

use rsp2_array_types::{V3, M33, M3, mat, inv};
use rsp2_assert_close::{CheckClose, Tolerances, CheckCloseError};

/// Defines a vector basis for periodic boundary conditions in three dimensions.
///
/// This type currently has two conflated roles:
///
/// * A linear transformation between "fractional" data to "cartesian" data.
///   See [`CoordsKind`] for definitions.
/// * A 3x3 matrix with a precomputed inverse.
///   (you might see this usage referred to as "abuse"...)
///
/// [`CoordsKind`]: enum.CoordsKind.html
#[derive(Debug, Clone)]
pub struct Lattice {
    matrix: Arc<M33>,
    inverse: Arc<M33>,
}

// Manual impl that doesn't compare the inverse.
impl PartialEq<Lattice> for Lattice {
    fn eq(&self, other: &Lattice) -> bool {
        // deconstruct to get errors when new fields are added
        let Lattice { matrix, inverse: _ } = self;
        matrix == &other.matrix
    }
}

impl Lattice {
    /// Create a lattice from a matrix where the rows are lattice vectors.
    #[inline]
    pub fn new(matrix: &M33) -> Self {
        let inverse = Arc::new(inv(matrix));
        let matrix = Arc::new(*matrix);
        Self { matrix, inverse }
    }

    #[inline(always)]
    pub fn from_vectors(vectors: &[V3; 3]) -> Self {
        Self::new(&M3(*vectors))
    }

    // NOTE: There's no inverse or transpose because they're basically
    //       never the right answer. (at least, I've yet to see any problem
    //       that is solved by them)
    //
    //       In virtually all circumstances (barring a couple of spots that basically
    //       amount to abuse), a Lattice in RSP2 is something that you multiply against
    //       "fractional" data (which may be real-space fractional, or reciprocal
    //       fractional, or even partial derivatives of something with respect to one
    //       of these bases) to produce "cartesian" data.
    //
    //       In these circumstances, the inverse and transpose are never useful.
    //       (the inverse *matrix* is useful, but not as a Lattice)
    //            - ML
    //
    /// Get the reciprocal lattice.
    ///
    /// This is defined as the inverse transpose. **There is no 2 PI factor.**
    /// It is the lattice that transforms between reciprocal space fractional
    /// and reciprocal space Euclidean coordinates.
    ///
    /// Notice that partial derivatives with respect to a coordinate system
    /// also transform like the reciprocal of that coordinate system.
    #[inline] // for dead-code elimination of matrices we don't use
    pub fn reciprocal(&self) -> Self {
        // FIXME: *strictly speaking* this violates the invariant that 'inverse'
        //        is uniquely determined by 'matrix', which our PartialEq
        //        relies on.
        Self {
            matrix: Arc::new(self.inverse.t()),
            inverse: Arc::new(self.matrix.t()),
        }
    }

    /// Matrix where lattice vectors are rows.
    #[inline]
    pub fn matrix(&self) -> &M33
    { &self.matrix }

    /// Get the (precomputed) inverse of the matrix where lattice vectors are rows.
    #[inline]
    pub fn inverse_matrix(&self) -> &M33
    { &self.inverse }

    #[inline]
    pub fn vectors(&self) -> &[V3; 3]
    { &self.matrix().0 }

    pub fn norms(&self) -> [f64; 3]
    { V3(*self.vectors()).map(|v| v.norm()).0 }

    pub fn sqnorms(&self) -> [f64; 3]
    { V3(*self.vectors()).map(|v| v.sqnorm()).0 }

    /// Get the (positive) volume of the lattice cell.
    pub fn volume(&self) -> f64
    { self.matrix().det().abs() }

    /// Apply a cartesian transformation to the lattice.
    pub fn transformed_by(&self, m: &M33) -> Lattice
    { self * &m.t() }

    /// Take an integer linear combination of the lattice vectors.
    pub fn linear_combination(&self, coeffs: &M33<i32>) -> Lattice
    { &coeffs.map(|x| x as f64) * self }

    /// Get the normal to the family of planes with a given Miller index.
    ///
    /// Miller indices with `gcd > 1` are permitted and are interpreted according to
    /// the documentation in the [`miller`] module.  They should produce the same value
    /// as a primitive vector, numerical errors not withstanding.
    ///
    /// # Panics
    ///
    /// Panics on a Miller index of `[0, 0, 0]`.
    pub fn plane_normal(&self, miller: V3<i32>) -> V3
    { self.miller_to_recip_cart(miller).unit() }

    /// Get the interplanar spacing of the family of planes specified by a Miller index.
    ///
    /// Miller indices with `gcd > 1` are permitted and are interpreted according to
    /// the documentation in the [`miller`] module.  They should produce a spacing that
    /// is `1 / gcd` times the typical spacing.
    ///
    /// # Panics
    ///
    /// Panics on a Miller index of `[0, 0, 0]`.
    pub fn plane_spacing(&self, miller: V3<i32>) -> f64
    { self.miller_to_recip_cart(miller).norm().recip() }

    #[inline(always)]
    fn miller_to_recip_cart(&self, miller: V3<i32>) -> V3 {
        assert_ne!(miller, V3::zero());
        // rowvec-mat multiplication by A.T^-1,
        // or equivalently mat-colvec multiplication by A^-1
        &*self.inverse * miller.map(|x| x as f64)
    }

    /// Test if two Lattices represent the same lattice,
    /// in the mathematical sense. This is to say that they each
    /// generate the same infinite set of displacement vectors.
    ///
    /// The row-based matrices A and B both represent the same
    /// Bravais lattice if and only if A B^-1 is unimodular.
    ///
    /// tol is an absolute tolerance used to test integerness
    /// of a float.
    // FIXME: should probably find a source to cite for the
    //        unimodular fact, or maybe put up some condensed
    //        form of my own notes on the matter somewhere. - ML
    pub fn is_equivalent_to(&self, tol: f64, other: &Lattice) -> bool
    {
        warn!("Untested code path: ee755063-5bc7-4a44-b049-ea0b923a73c8");
        let m = self.matrix() * other.inverse_matrix();
        match crate::util::Tol(tol).unfloat_m33(&m) {
            Ok(m) => m.det().abs() == 1,
            Err(_) => false,
        }
    }

    /// Rotate into a form where the first vector points entirely along x, and the
    /// second vector is in the xy-plane.
    ///
    /// This is the form required by LAMMPS and `V_Sim`.  This form makes the lattice
    /// matrix lower triangular in RSP2's convention.
    ///
    /// If the input cell is left-handed, the output will also be left-handed; however
    /// it is not specified which of the 3 diagonal elements will be negative.
    pub fn rotate_to_lower_triangular(&self) -> Lattice {
        // https://lammps.sandia.gov/doc/Howto_triclinic.html
        let &[a, b, c] = self.vectors();

        // The formulas are tuned for a right-handed matrix.
        //
        // For a left-handed matrix, we simply negate the first vector, solve the problem for the
        // resulting right-handed basis, then negate the first vector again to get the answer for
        // our matrix.
        let sign = self.matrix().det().signum();
        let a = a * sign;

        let a_unit = a.unit();
        let m00 = a.norm();
        let m10 = b.dot(&a_unit);
        let m20 = c.dot(&a_unit);
        let m11 = f64::sqrt(b.sqnorm() - m10*m10);
        let m21 = (b.dot(&c) - m10*m20) / m11;
        let m22 = f64::sqrt(c.sqnorm() - m20*m20 - m21*m21);

        Lattice::from([
            [sign * m00, 0.0, 0.0],
            [m10, m11, 0.0],
            [m20, m21, m22],
        ])
    }

    /// Determine if the lattice is highly skewed.
    ///
    /// More specifically, this returns true if there exist `i != k` such that:
    ///
    /// ```text
    /// abs(b_i dot b_k) > b_i dot b_i
    /// ```
    ///
    /// where `b_i` are the reciprocal basis vectors.
    /// Adding some `fuzz` will cause more cells to be classified as not skewed.
    pub(crate) fn is_large_skew(&self, fuzz: f64) -> bool {
        let recip = self.reciprocal();
        let recip_metric = recip.matrix() * &recip.matrix().t(); // matrix of dot products
        for i in 0..3 {
            for k in 0..3 {
                if i != k {
                    if f64::abs(recip_metric[i][k]) > recip_metric[i][i] * (1.0 + fuzz) {
                        return true;
                    }
                }
            }
        }
        false
    }
}

/// Helper constructors
impl Lattice {
    /// The identity lattice.
    #[inline]
    pub fn eye() -> Self { Self::cubic(1.0) }

    #[inline]
    pub fn diagonal(&diag: &[f64; 3]) -> Self
    { Self::new(&M33::from_diag(V3(diag))) }

    // NOTE: Currently there are only helpers for bravais lattices whose
    //       matrix representations are DEAD OBVIOUS.
    //
    //       A helper for e.g. `hexagonal()` could spell trouble if one uses
    //       it together with fractional coords (without considering whether
    //       the lattice follows the same conventions)

    /// A cubic lattice ((a, a, a), (90, 90, 90))
    #[inline]
    pub fn cubic(a: f64) -> Self
    { Self::orthorhombic(a, a, a) }

    /// An orthorhombic lattice ((a, b, c), (90, 90, 90))
    #[inline]
    pub fn orthorhombic(a: f64, b: f64, c: f64) -> Self
    { Self::diagonal(&[a, b, c]) }

    // who needs quickcheck
    /// Generate a random lattice.
    ///
    /// Elements are pulled from a uniform distribution of `[-x, x]`
    /// and in all honesty I doubt this is representative of typical lattices.
    /// (symmetry in particular is poorly represented)
    #[cfg(test)]
    pub fn random_uniform(max: f64) -> Self {
        use slice_of_array::prelude::*;

        Lattice::from(
            (0..9)
            .map(|_| (::rand::random::<f64>() - 0.5) * 2.0 * max)
            .collect::<Vec<_>>()
            .nest().as_array::<[_;3]>()
        )
    }
}

/// Defaults to the identity matrix.
impl Default for Lattice {
    #[inline]
    fn default() -> Lattice { Lattice::eye() }
}

impl<'a> From<&'a [[f64; 3]; 3]> for Lattice {
    #[inline(always)]
    fn from(m: &'a [[f64; 3]; 3]) -> Self
    { Lattice::new(&mat::from_array(*m)) }
}

impl From<[[f64; 3]; 3]> for Lattice {
    #[inline(always)]
    fn from(m: [[f64; 3]; 3]) -> Self
    { Lattice::from(&m) }
}

// FIXME so far I have managed to accidentally flip the order of virtually
//       every multiplication with a lattice that I have written, despite my
//       love of row-based matrices. Maybe the '*' operator is just a bad idea.
//
// FIXME given my declaration that lattices are always for conversion between
//       fractional and cartesian data, this impl shouldn't exist.
impl<'a, 'b> Mul<&'b Lattice> for &'a Lattice {
    type Output = Lattice;

    #[inline(always)]
    fn mul(self, other: &'b Lattice) -> Lattice {
        self * other.matrix()
    }
}

impl<'a, 'b> Mul<&'b M33> for &'a Lattice {
    type Output = Lattice;

    fn mul(self, other: &'b M33) -> Lattice {
        // Let the inverse be computed from scratch,
        // for sustained accuracy after many products
        Lattice::new(&(self.matrix() * other))
    }
}

impl<'a, 'b> Mul<&'b Lattice> for &'a M33 {
    type Output = Lattice;

    fn mul(self, other: &'b Lattice) -> Lattice {
        Lattice::new(&(self * other.matrix()))
    }
}

impl<'a, 'b> Mul<&'b Lattice> for &'a V3 {
    type Output = V3;

    fn mul(self, other: &'b Lattice) -> V3 {
        self * other.matrix()
    }
}

impl<'b> Mul<&'b Lattice> for V3 {
    type Output = V3;

    fn mul(self, other: &'b Lattice) -> V3 {
        self * other.matrix()
    }
}

impl<'a, 'b> Div<&'b Lattice> for &'a V3 {
    type Output = V3;

    fn div(self, other: &'b Lattice) -> V3 {
        self * other.inverse_matrix()
    }
}

impl<'b> Div<&'b Lattice> for V3 {
    type Output = V3;

    fn div(self, other: &'b Lattice) -> V3 {
        self * other.inverse_matrix()
    }
}

impl CheckClose for Lattice {
    type Scalar = f64;

    fn check_close(&self, other: &Lattice, tol: Tolerances) -> Result<(), CheckCloseError> {
        use rsp2_array_types::Unvee;
        self.matrix().unvee().check_close(&other.matrix().unvee(), tol)
    }
}

#[cfg(feature = "serde")]
impl Serialize for Lattice {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.matrix().serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de> Deserialize<'de> for Lattice {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = M33::<f64>::deserialize(deserializer)?;
        Ok(Lattice::new(&raw))
    }
}

#[cfg(test)]
#[deny(unused)]
mod tests {
    use super::*;
    use rsp2_array_types::Unvee;

    #[test]
    fn get_inverse() {
        // matrix whose inverse should be able to be computed exactly
        // by any reasonable matrix inversion algorithm working on f64s
        let matrix = mat::from_array([
            [2.0, 2.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 2.0],
        ]);
        let exact_inverse = mat::from_array([
            [0.5, -0.25, 0.0],
            [0.0,  0.25, 0.0],
            [0.0,   0.0, 0.5],
        ]);

        let lattice = Lattice::new(&matrix);
        assert_eq!(&matrix, lattice.matrix());
        assert_eq!(&exact_inverse, lattice.inverse_matrix());

        let recip = lattice.reciprocal();
        assert_eq!(&exact_inverse.t(), recip.matrix());
        assert_eq!(&matrix.t(), recip.inverse_matrix());

        assert_ne!(&Lattice::eye(), &lattice);
    }

    #[test]
    fn multiplication_order()  {
        // matrices that don't commute
        let a = Lattice::from(&[
            [2.0, 2.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 2.0],
        ]);
        let b = Lattice::from(&[
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]);

        assert_eq!(&a * &b, Lattice::from(&[
            [2.0, 2.0, 0.0],
            [4.0, 0.0, 0.0],
            [0.0, 0.0, 2.0],
        ]));
        assert_eq!(&b * &a,  Lattice::from(&[
            [0.0, 4.0, 0.0],
            [2.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
        ]));
    }

    #[test]
    fn planes() {
        use rand::Rng;

        // check the normals of some arbitrary planes
        for _ in 0..20 {
            let lattice = Lattice::random_uniform(20.0);
            let miller = crate::miller::random_nonzero(10);
            let gcd = crate::miller::gcd(miller);

            let recip = miller.map(|x| x as f64) * &lattice.reciprocal();
            let normal = lattice.plane_normal(miller);
            let plane_translation_vector = normal * lattice.plane_spacing(miller);
            assert_close!(
                lattice.plane_normal(miller).0,
                lattice.plane_normal(miller.map(|x| x / gcd)).0,
            );
            assert_close!(1.0, plane_translation_vector.dot(&recip));
        }

        // Test specifically along the (001) planes for more properties we can easily verify.
        for _ in 0..20 {
            let lattice = Lattice::random_uniform(20.0);

            let n = rand::thread_rng().gen_range(1, 10+1);
            let miller = V3([0, 0, n]);

            let [a, b, c] = lattice.vectors();
            let expected_normal = a.cross(b).unit(); // up to sign...
            let expected_spacing = expected_normal.dot(c).abs() / n as f64;

            assert!(lattice.plane_normal(miller).dot(c) > 0.0);
            assert_close!(lattice.plane_normal(miller).dot(&expected_normal).abs(), 1.0);
            assert_close!(lattice.plane_spacing(miller), expected_spacing);
        }
    }

    #[test]
    fn rotation_to_lower_triangular()  {
        for _ in 0..30 {
            let lattice = Lattice::random_uniform(10.0);
            let rotated = lattice.rotate_to_lower_triangular();

            // Rotated must be lower triangular
            assert_close!(abs=1e-11, 0.0, rotated.matrix()[0][1]);
            assert_close!(abs=1e-11, 0.0, rotated.matrix()[0][2]);
            assert_close!(abs=1e-11, 0.0, rotated.matrix()[1][2]);

            // The lattices must be equal up to rotation.
            //
            // A sufficient condition for a 3x3 matrix R to be a rotation matrix
            // is that it is orthogonal with determinant 1.
            let rotation = lattice.inverse_matrix() * rotated.matrix();

            // FIXME: why must these tolerances be so large?
            //        Does our implementation amplify rounding errors?
            assert_close!(rel=1e-5, abs=1e-5, M33::eye().unvee(), (rotation * rotation.t()).unvee());
            assert_close!(rel=1e-5, 1.0, rotation.det());
        }
    }
}
