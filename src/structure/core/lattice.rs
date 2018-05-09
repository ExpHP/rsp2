use ::rsp2_array_utils::{map_arr};
use ::std::ops::{Mul, Div};
use ::std::sync::Arc;

use ::rsp2_array_types::{V3, M33, M3, mat, inv};
use ::rsp2_assert_close::{CheckClose, Tolerances, CheckCloseError};

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
        let Lattice { ref matrix, inverse: _ } = *self;
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
    { map_arr(*self.vectors(), |v| v.norm()) }

    pub fn sqnorms(&self) -> [f64; 3]
    { map_arr(*self.vectors(), |v| v.sqnorm()) }

    /// Get the (positive) volume of the lattice cell.
    pub fn volume(&self) -> f64
    { self.matrix().det().abs() }

    /// Apply a cartesian transformation to the lattice.
    pub fn transformed_by(&self, m: &M33) -> Lattice
    { self * &m.t() }

    /// Take an integer linear combination of the lattice vectors.
    pub fn linear_combination(&self, coeffs: &M33<i32>) -> Lattice
    { &coeffs.map(|x| x as f64) * self }

    /// Test if two Lattices represent the same Bravais lattice,
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
        match ::util::Tol(tol).unfloat_m33(&m) {
            Ok(m) => m.det().abs() == 1,
            Err(_) => false,
        }
    }
}

/// Helper constructors
impl Lattice {
    /// The identity lattice.
    #[inline]
    pub fn eye() -> Self { Self::cubic(1.0) }

    #[inline]
    pub fn diagonal(&[x, y, z]: &[f64; 3]) -> Self { Self::orthorhombic(x, y, z) }

    // NOTE: Currently there are only helpers for bravais lattices whose
    //       matrix representations are DEAD OBVIOUS.
    //
    //       A helper for e.g. `hexagonal()` could spell trouble if one uses
    //       it together with fractional coords (without considering whether
    //       the lattice follows the same conventions)

    /// A cubic lattice ((a, a, a), (90, 90, 90))
    #[inline]
    pub fn cubic(a: f64) -> Self { Self::orthorhombic(a, a, a) }

    /// An orthorhombic lattice ((a, b, c), (90, 90, 90))
    #[inline]
    pub fn orthorhombic(a: f64, b: f64, c: f64) -> Self
    { Self::from(&[[a, 0., 0.], [0., b, 0.], [0., 0., c]]) }

    // who needs quickcheck
    /// Generate a random lattice.
    ///
    /// NOTE:
    /// Elements are pulled from a uniform distribution of [-x, x]
    /// and in all honesty I doubt this is representative of typical
    #[cfg(test)]
    pub fn random_uniform(max: f64) -> Self {
        use ::slice_of_array::prelude::*;

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
        // Let the inverse be computed from scratch,
        // for sustained accuracy after many products
        self * other.matrix()
    }
}

impl<'a, 'b> Mul<&'b M33> for &'a Lattice {
    type Output = Lattice;

    fn mul(self, other: &'b M33) -> Lattice {
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
        use ::rsp2_array_types::Unvee;
        self.matrix().unvee().check_close(&other.matrix().unvee(), tol)
    }
}

#[cfg(tests)]
#[deny(unused)]
mod tests {
    use super::*;

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
        assert_eq!(matrix, lattice.matrix());
        assert_eq!(exact_inverse, lattice.inverse_matrix());

        let inverted = lattice.inverted();
        assert_eq!(exact_inverse, inverted.matrix());
        assert_eq!(matrix, inverted.inverse_matrix());

        assert_eq!(Lattice::eye(), &lattice * &inverted);
        assert_eq!(Lattice::eye(), &inverted * &lattice);

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
}
