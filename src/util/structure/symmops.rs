use ::rsp2_array_utils::{dot, mat_from_fn, MatrixDeterminantExt};

use ::std::ops::Mul;

pub struct FracRot{
    /// This is the transpose of what one would
    /// typically think of as the "rotation matrix"
    t: [[i16; 3]; 3],
}

impl FracRot {
    pub fn new(mat: &[[i16; 3]; 3]) -> FracRot {
        assert_eq!(mat.determinant().abs(), 1);
        FracRot { t: mat_from_fn(|r, c| mat[c][r]) }
    }
}

impl<'a, 'b> Mul<&'b FracRot> for &'a FracRot {
    type Output = FracRot;

    fn mul(self, other: &'b FracRot) -> FracRot
    {
        // reverse order due to working with transpose
        FracRot { t: dot(&other.t, &self.t) }
    }
}

impl FracRot {
    pub fn transform(&self, fracs: &[[f64; 3]]) -> Vec<[f64; 3]>
    {
        let t = mat_from_fn(|c, r| self.t[c][r].into());
        ::util::dot_n3_33T(fracs, &t)
    }
}
