use ::errors::*;
use ::std::rc::Rc;

use ::rsp2_array_types::{V3, M33, M44, M4, V4, mat};

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
    t: M33<i32>,
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
    V3<i32>,
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
    t: Rc<M44<i32>>,
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
    { Self { t: mat::from_array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) } }

    /// Construct from a matrix.
    ///
    /// The input should be a matrix `R` such that `X R^T ~ X`,
    /// where the rows of `X` are fractional positions.
    pub fn new(mat: &M33<i32>) -> FracRot
    {
        assert_eq!(mat.det().abs(), 1);
        FracRot { t: mat.t() }
    }

    // transposed float matrix
    pub(crate) fn float_t(&self) -> M33
    { self.t.map(Into::into) }
}

impl FracRot {
    /// Flipped group operator.
    ///
    /// `a.then(b) == b.of(a)`.  The flipped order is more aligned
    /// with this library's generally row-centric design.
    pub fn then(&self, other: &FracRot) -> FracRot
    {
        // (since these are transposes, this is the natural order of application)
        FracRot { t: &self.t * &other.t }
    }

    /// Conventional group operator.
    pub fn of(&self, other: &FracRot) -> FracRot
    { other.then(self) }
}

impl FracTrans {
    pub fn eye() -> Self
    { FracTrans(V3([0, 0, 0])) }

    pub fn from_floats(xs: &V3) -> Result<FracTrans>
    { xs.try_map(|x| ::util::Tol(1e-4).unfloat(x * 12.0)).map(FracTrans) }

    fn float(&self) -> V3
    { self.0.map(|x| f64::from(x) / 12f64) }
}

const FRAC_OP_EYE: M44<i32> = M4([
    V4([1, 0, 0, 0]),
    V4([0, 1, 0, 0]),
    V4([0, 0, 1, 0]),
    V4([0, 0, 0, 1]),
]);

impl FracOp {
    pub fn eye() -> Self
    { Self { t: FRAC_OP_EYE.into() } }

    pub fn new(rot: &FracRot, trans: &FracTrans) -> Self
    {
        let mut out = FRAC_OP_EYE;
        out[0][..3].copy_from_slice(&*rot.t[0]);
        out[1][..3].copy_from_slice(&*rot.t[1]);
        out[2][..3].copy_from_slice(&*rot.t[2]);
        out[3][..3].copy_from_slice(&*trans.0);
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

impl FracOp {
    /// Flipped group operator.
    ///
    /// `a.then(b) == b.of(a)`.  The flipped order is more aligned
    /// with this library's generally row-centric design.
    pub fn then(&self, other: &FracOp) -> FracOp
    {
        // this is the natural order of application for transposes
        let mut t = &*self.t * &*other.t;

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

    /// Conventional group operator.
    pub fn of(&self, other: &FracOp) -> FracOp
    { other.then(self) }
}

impl FracRot {
    pub fn transform_prim(&self, fracs: &[V3]) -> Vec<V3>
    { fracs.iter().map(|v| v * &self.float_t()).collect() }
}

impl<'a> From<&'a [[i32; 3]; 3]> for FracRot {
    fn from(m: &'a [[i32; 3]; 3]) -> Self
    { FracRot::new(&mat::from_array(*m))}
}

impl FracTrans {
    pub fn transform_prim_mut(&self, fracs: &mut [V3])
    { ::util::translate_mut_n3_3(fracs, &self.float()) }
}

impl FracOp {
    pub fn transform_prim(&self, fracs: &[V3]) -> Vec<V3>
    {
        let mut out = self.to_rot().transform_prim(fracs);
        self.to_trans().transform_prim_mut(&mut out);
        out
    }
}

#[cfg(test)]
#[deny(unused)]
mod tests {
    use super::*;
    use ::rsp2_array_types::Envee;

    #[test]
    fn rot_transform()
    {
        let r = [
            [0, -1, 0],
            [1,  0, 0],
            [0,  0, 1],
        ];
        assert_eq!(
            FracRot::from(&r).transform_prim([[1.0, 5.0, 7.0]].envee_ref()),
            vec![[-5.0, 1.0, 7.0]].envee(),
        );
    }

    #[test]
    fn two_transform()
    {
        // two operations that don't commute
        let xy = FracRot::from(&[
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ]);
        let zx = FracRot::from(&[
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
        ]);
        let xyzx = FracRot::from(&[
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
        ]);
        // a primitive structure that is sensitive to any permutations of the axes
        let prim = vec![[1., 2., 3.]].envee();
        assert_eq!(xy.then(&zx), xyzx);
        assert_eq!(zx.of(&xy), xyzx);
        assert_eq!(
            zx.transform_prim(&xy.transform_prim(&prim)),
            xyzx.transform_prim(&prim),
        );

        let t = FracTrans::eye();
        let xy = FracOp::new(&xy, &t);
        let zx = FracOp::new(&zx, &t);
        let xyzx = FracOp::new(&xyzx, &t);
        assert_eq!(xy.then(&zx), xyzx);
        assert_eq!(zx.of(&xy), xyzx);
        assert_eq!(
            zx.transform_prim(&xy.transform_prim(&prim)),
            xyzx.transform_prim(&prim),
        );
    }

    #[test]
    fn symmop_mul()
    {
        let op = FracOp::new(
            &FracRot::from(&[
                [ 0,  1, 0],
                [-1,  1, 0],
                [ 0,  0, 1],
            ]),
            &FracTrans::from_floats(&V3([1./3., 2./3., 0.0])).unwrap(),
        );
        let square = FracOp::new(
            &FracRot::from(&[
                [-1, 1, 0],
                [-1, 0, 0],
                [ 0, 0, 1],
            ]),
            &FracTrans::from_floats(&V3([0., 0., 0.])).unwrap(),
        );

        assert_eq!(op.then(&op), square);
    }
}
