use ::rsp2_array_utils::{MatrixInverseExt, MatrixDeterminantExt};
use ::rsp2_array_utils::{dot, vec_from_fn};
use ::std::ops::Mul;
use ::std::rc::Rc;

/// A 3x3 matrix with a precomputed inverse.
#[derive(Debug, Clone)]
pub struct Lattice {
    matrix: Rc<[[f64; 3]; 3]>,
    inverse: Rc<[[f64; 3]; 3]>,
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
    pub fn new(matrix: &[[f64; 3]; 3]) -> Self {
        let inverse = Rc::new(matrix.inverse());
        let matrix = Rc::new(*matrix);
        Self { matrix, inverse }
    }

    /// Invert the lattice.
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
    pub fn matrix(&self) -> &[[f64; 3]; 3] { &self.matrix }
    /// Inverse of the matrix where lattice vectors are rows.
    pub fn inverse_matrix(&self) -> &[[f64; 3]; 3] { &self.inverse }

    pub fn lengths(&self) -> [f64; 3]
    {
        let quadrances = self.quadrances();
        vec_from_fn(|k| quadrances[k].sqrt())
    }

    pub fn quadrances(&self) -> [f64; 3]
    { vec_from_fn(|k| dot(&self.matrix[k], &self.matrix[k])) }

    /// Get the (positive) volume of the lattice cell.
    pub fn volume(&self) -> f64
    { self.matrix.determinant().abs() }
}

/// Helper constructors
impl Lattice {
    /// The identity lattice.
    pub fn eye() -> Self { Self::cubic(1.0) }

    pub fn diagonal(vec: &[f64; 3]) -> Self { Self::orthorhombic(vec[0], vec[1], vec[2]) }

    // NOTE: Currently there are only helpers for bravais lattices whose
    //       matrix representations are DEAD OBVIOUS.
    //
    //       A helper for e.g. `hexagonal()` could spell trouble if one uses
    //       it together with fractional coords (without considering whether
    //       the lattice follows the same conventions)

    /// A cubic lattice ((a, a, a), (90, 90, 90))
    pub fn cubic(a: f64) -> Self { Self::orthorhombic(a, a, a) }

    /// An orthorhombic lattice ((a, b, c), (90, 90, 90))
    pub fn orthorhombic(a: f64, b: f64, c: f64) -> Self { Self::new(&[[a, 0., 0.], [0., b, 0.], [0., 0., c]]) }

    // who needs quickcheck
    /// Generate a random lattice.
    ///
    /// NOTE:
    /// Elements are pulled from a uniform distribution of [-x, x]
    /// and in all honesty I doubt this is representative of typical
    #[cfg(test)]
    pub fn random_uniform(max: f64) -> Self {
        use ::slice_of_array::prelude::*;

        Lattice::new(
            (0..9)
            .map(|_| (::rand::random::<f64>() - 0.5) * 2.0 * max)
            .collect::<Vec<_>>()
            .nest().as_array()
        )
    }
}

/// Defaults to the identity matrix.
impl Default for Lattice {
    fn default() -> Lattice { Lattice::eye() }
}

// FIXME so far I have managed to accidentally flip the order of virtually
//       every multiplication with a lattice that I have written, despite my
//       love of row-based matrices. Maybe the '*' operator is just a bad idea.
impl<'a, 'b> Mul<&'b Lattice> for &'a Lattice {
    type Output = Lattice;
    fn mul(self, other: &'b Lattice) -> Lattice {
        // Let the inverse be computed from scratch,
        // for sustained accuracy after many products
        self * other.matrix()
    }
}

impl<'a, 'b> Mul<&'b [[f64; 3]; 3]> for &'a Lattice {
    type Output = Lattice;
    fn mul(self, other: &'b [[f64; 3]; 3]) -> Lattice {
        Lattice::new(&dot(self.matrix(), other))
    }
}

impl<'a, 'b> Mul<&'b Lattice> for &'a [[f64; 3]; 3] {
    type Output = Lattice;
    fn mul(self, other: &'b Lattice) -> Lattice {
        Lattice::new(&dot(self, other.matrix()))
    }
}

/// A Lattice that can be sent across thread boundaries.
#[derive(Debug, Clone)]
pub struct Sent {
    matrix: Box<[[f64; 3]; 3]>,
    inverse: Box<[[f64; 3]; 3]>,
}

impl Lattice {
    pub fn send(self) -> Sent {
        let Lattice { matrix, inverse } = self;
        let matrix = Box::new((&*matrix).clone());
        let inverse = Box::new((&*inverse).clone());
        Sent { matrix, inverse }
    }
}

impl Sent {
    pub fn recv(self) -> Lattice {
        let Sent { matrix, inverse } = self;
        let matrix = matrix.into();
        let inverse = inverse.into();
        Lattice { matrix, inverse }
    }
}

#[cfg(tests)]
#[deny(unused)]
mod tests {

    #[test]
    fn get_inverse() {
        // matrix whose inverse should be able to be computed exactly
        // by any reasonable matrix inversion algorithm working on f64s
        let matrix = [
            [2.0, 2.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 2.0],
        ];
        let exact_inverse = [
            [0.5, -0.25, 0.0],
            [0.0,  0.25, 0.0],
            [0.0,   0.0, 0.5],
        ];

        let lattice = Lattice::new(matrix);
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
        let a = Lattice::new([
            [2.0, 2.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 2.0],
        ]);
        let b = Lattice::new([
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]);

        assert_eq!(&a * &b, Lattice::new([
            [2.0, 2.0, 0.0],
            [4.0, 0.0, 0.0],
            [0.0, 0.0, 2.0],
        ]));
        assert_eq!(&b * &a,  Lattice::new([
            [0.0, 4.0, 0.0],
            [2.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
        ]));
    }
}
