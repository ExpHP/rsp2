use ::rsp2_array_utils::{map_arr};
use ::std::ops::Mul;
use ::std::sync::Arc;

use ::rsp2_array_types::{V3, M33, M3, mat, inv};

/// A 3x3 matrix with a precomputed inverse.
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

    /// Invert the lattice.
    #[inline]
    pub fn inverted(&self) -> Self {
        // FIXME: *strictly speaking* this violates the invariant that 'inverse'
        //        is uniquely determined by 'matrix', which our PartialEq
        //        relies on.
        Self {
            matrix: self.inverse.clone(),
            inverse: self.matrix.clone(),
        }
    }

    /// Matrix where lattice vectors are rows.
    #[inline]
    pub fn matrix(&self) -> &M33
    { &self.matrix }

    /// Inverse of the matrix where lattice vectors are rows.
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
    pub fn diagonal(vec: &[f64; 3]) -> Self { Self::orthorhombic(vec[0], vec[1], vec[2]) }

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
