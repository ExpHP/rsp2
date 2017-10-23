use ::rsp2_array_utils::{dot, vec_from_fn, try_vec_from_fn, mat_from_fn, MatrixDeterminantExt};
use ::errors::*;
use ::std::ops::Mul;
use ::std::rc::Rc;

// NOTE: This API is in flux. Some (possibly incorrect) notes:
//
//  * Correct usage of spacegroup operators requires knowing the
//    primitive cell.  A supercell does not have its own spacegroup
//    independent of the primitive cell; that would allow a supercell
//    to have different physics!

/// A point group operation on a primitive cell.
#[derive(Debug, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct FracRot {
    /// This is the transpose of what one would
    /// typically think of as the "rotation matrix"
    ///
    /// Invariants:
    ///  - `abs(det(t)) == 1`
    t: [[i32; 3]; 3],
}

/// The translation part of a spacegroup operation on a primitive cell.
///
/// This always has coordinates that are multiples of `1/12`.
#[derive(Debug, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct FracTrans (
    /// This is the vector times 12.
    ///
    /// Invariants:
    ///  - elements are reduced into the range `0 <= x < 12`.
    [i32; 3],
);

/// A spacegroup operation on a primitive cell.
#[derive(Debug, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct FracOp {
    /// This is the transpose of what one would typically
    /// think of as the affine transformation matrix.
    ///
    /// The translation elements in the final row are numerators over 12.
    ///
    /// Invariants:
    ///  - translation elements are reduced into the range `0 <= x < 12`.
    ///  - final element is 1
    t: Rc<[[i32; 4]; 4]>,
}

impl Default for FracOp {
    fn default() -> Self
    { Self::eye() }
}

impl Default for FracTrans {
    fn default() -> Self
    { Self::eye() }
}

impl Default for FracRot {
    fn default() -> Self
    { Self::eye() }
}

impl From<FracTrans> for FracOp {
    fn from(v: FracTrans) -> Self
    { Self::new(&Default::default(), &v) }
}

impl From<FracRot> for FracOp {
    fn from(r: FracRot) -> Self
    { Self::new(&r, &Default::default()) }
}

impl FracRot {
    pub fn eye() -> Self
    { Self { t: [[1, 0, 0], [0, 1, 0], [0, 0, 1]] } }

    pub fn new(mat: &[[i32; 3]; 3]) -> FracRot
    {
        assert_eq!(mat.determinant().abs(), 1);
        FracRot { t: mat_from_fn(|r, c| mat[c][r]) }
    }

    // transposed float matrix
    fn float_t(&self) -> [[f64; 3]; 3]
    { mat_from_fn(|r, c| self.t[r][c].into()) }
}

impl FracTrans {
    pub fn eye() -> Self
    { FracTrans([0, 0, 0]) }

    pub fn from_floats(xs: &[f64; 3]) -> Result<FracTrans>
    {Ok({
        FracTrans(try_vec_from_fn(|k| round_checked(xs[k] * 12.0, 1e-4))?)
    })}

    fn float(&self) -> [f64; 3]
    { vec_from_fn(|k| f64::from(self.0[k]) / 12f64) }
}

const FRAC_OP_EYE: [[i32; 4]; 4] = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
];

impl FracOp {
    pub fn eye() -> Self
    { Self { t: FRAC_OP_EYE.into() } }

    pub fn new(rot: &FracRot, trans: &FracTrans) -> Self
    {
        let mut out = FRAC_OP_EYE;
        out[0][..3].copy_from_slice(&rot.t[0]);
        out[1][..3].copy_from_slice(&rot.t[1]);
        out[2][..3].copy_from_slice(&rot.t[2]);
        out[3][..3].copy_from_slice(&trans.0);
        FracOp { t: out.into() }
    }

    pub fn to_rot(&self) -> FracRot
    {
        let mut out = FracRot::eye();
        out.t[0].copy_from_slice(&self.t[0][..3]);
        out.t[1].copy_from_slice(&self.t[1][..3]);
        out.t[2].copy_from_slice(&self.t[2][..3]);
        out
    }

    pub fn to_trans(&self) -> FracTrans
    {
        let mut out = FracTrans::eye();
        out.0.copy_from_slice(&self.t[3][..3]);
        out
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

impl<'a, 'b> Mul<&'b FracOp> for &'a FracOp {
    type Output = FracOp;

    fn mul(self, other: &'b FracOp) -> FracOp
    {
        // reverse order due to working with transpose
        let mut t = dot(&*other.t, &*self.t);

        // reduce the translation for a unique representation
        for x in &mut t[3][..3] {
            *x %= 12;
            *x += 12;
            *x %= 12;
        }
        debug_assert!(t[3][..3].iter().all(|&x| 0 <= x && x < 12));
        debug_assert_eq!(t[3][3], 1);

        FracOp { t: t.into() }
    }
}

impl FracRot {
    pub fn transform_prim(&self, fracs: &[[f64; 3]]) -> Vec<[f64; 3]>
    { ::util::dot_n3_33(fracs, &self.float_t()) }
}

impl FracTrans {
    pub fn transform_prim_mut(&self, fracs: &mut [[f64; 3]])
    { ::util::translate_mut_n3_3(fracs, &self.float()) }
}

impl FracOp {
    pub fn transform_prim(&self, fracs: &[[f64; 3]]) -> Vec<[f64; 3]>
    {
        let mut out = self.to_rot().transform_prim(fracs);
        self.to_trans().transform_prim_mut(&mut out);
        out
    }
}

fn round_checked(x: f64, tol: f64) -> Result<i32>
{Ok({
    let r = x.round();
    ensure!((r - x).abs() < tol, ErrorKind::IntPrecisionError(x));
    r as i32
})}

#[cfg(test)]
#[deny(unused)]
mod tests {
    use super::*;

    #[test]
    fn rot_transform()
    {
        let r = [
            [0, -1, 0],
            [1,  0, 0],
            [0,  0, 1],
        ];
        assert_eq!(
            FracRot::new(&r).transform_prim(&[[1.0, 5.0, 7.0]]),
            vec![[-5.0, 1.0, 7.0]]
        );
    }

    #[test]
    fn two_transform()
    {
        let xy = FracRot::new(&[
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ]);
        let zx = FracRot::new(&[
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
        ]);
        let zxxy = FracRot::new(&[
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
        ]);
        assert_eq!(&zx * &xy, zxxy);
        assert_eq!(
            zx.transform_prim(&xy.transform_prim(&[[1., 2., 3.]])),
            zxxy.transform_prim(&[[1., 2., 3.]]));

        let t = FracTrans::eye();
        let zx = FracOp::new(&zx, &t);
        let xy = FracOp::new(&xy, &t);
        let zxxy = FracOp::new(&zxxy, &t);
        assert_eq!(&zx * &xy, zxxy);
        assert_eq!(
            zx.transform_prim(&xy.transform_prim(&[[1., 2., 3.]])),
            zxxy.transform_prim(&[[1., 2., 3.]]));

    }

    #[test]
    fn symmop_mul()
    {
        // FIXME should really test with things that don't commute
        // (this is just a regression test)
        let op = FracOp::new(
            &FracRot::new(&[
                [ 0,  1, 0],
                [-1,  1, 0],
                [ 0,  0, 1],
            ]),
            &FracTrans::from_floats(&[1./3., 2./3., 0.0]).unwrap(),
        );
        let square = FracOp::new(
            &FracRot::new(&[
                [-1, 1, 0],
                [-1, 0, 0],
                [ 0, 0, 1],
            ]),
            &FracTrans::from_floats(&[0., 0., 0.]).unwrap(),
        );

        assert_eq!(&op * &op, square);
    }
}
