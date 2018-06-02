use ::std::rc::Rc;
use ::Lattice;

use ::rsp2_array_types::{V3, M33, M44, M4, V4, mat};


// NOTE: This API is in flux. Some (possibly incorrect) notes:
//
//  * There exists an integer encoding of translations for primitive structures,
//    where, after recentering the rotations around a certain point, all of the
//    translations will have fractional coords that are multiples of 1/12.
//
//    rsp2 used to use these integer translations (in some funky 4x4 integer affine
//    matrix), but this was removed once I discovered the bit about recentering.
//
//  * I tried my best to clean it up, but you may still find language in some places
//    around the code base claiming a primitive cell is required for working with
//    space group operators. I believe that, in most cases, this is now
//    *almost always false,* since most code is now working with CartOps.
//

/// A point group operation in units of **a particular lattice matrix**.
///
/// This uses an integral representation that can provide `Eq + Hash + Ord` impls,
/// making it a great choice for analyzing the mathematical structure of the point
/// group and space group.
///
/// Its representation is sensitive to the choice of cell.
/// (i.e. it is not transferrable between identical structures expressed in terms
///  of different unit cells, or between primitive cells and supercells)
///
/// In this form, it is not capable of being applied to coordinate data.
#[derive(Debug, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct IntRot {
    /// The transpose of the rotation matrix in fractional coords.
    /// (equivalently (given rsp2's other conventions), one might say this
    ///  is the rotation matrix in a vector-major layout (as opposed to
    ///  coordinate-major))
    ///
    /// Invariants:
    ///  - `abs(det(t)) == 1`
    t: M33<i32>,
}

// FIXME: **Where the hell** did I ever get the notion that translations
//        are always multiples of `1/12`

/// The translation part of a spacegroup operation **on a primitive cell**.
///
/// This always has coordinates that are multiples of `1/12`.
///
/// (FIXME: CITATION NEEDED for the fact that at least one primitive cell
///  satisfies this property for every periodic structure.)
///
/// (However, so long as at least one primitive cell of a given structure
///  satisfies this property, it can easily be shown that they ALL do.
///  When transforming from any one primitive cell to another, the
///  translations get multiplied with a unimodular matrix)
///
/// Before applying it to supercells, you should convert it into cartesian.
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

impl Default for IntRot {
    fn default() -> Self
    { Self::eye() }
}

impl From<FracTrans> for FracOp {
    fn from(v: FracTrans) -> Self
    { Self::new(&Default::default(), &v) }
}

impl From<IntRot> for FracOp {
    fn from(r: IntRot) -> Self
    { Self::new(&r, &Default::default()) }
}

impl IntRot {
    pub fn eye() -> Self
    { Self { t: mat::from_array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) } }

    /// Construct from a matrix.
    ///
    /// The input should be a matrix `R` such that `X R^T ~ X`,
    /// where the rows of `X` are fractional positions.
    pub fn new(mat: &M33<i32>) -> IntRot
    { Self::opt_new(mat).expect("matrix not unimodular") }

    fn opt_new(mat: &M33<i32>) -> Option<IntRot> {
        match mat.det().abs() {
            1 => Some(IntRot { t: mat.t() }),
            _ => None,
        }
    }

    fn from_frac_t(float_t: &M33<f64>) -> Result<IntRot, ::IntPrecisionError>
    { float_t.try_map(|x| ::util::Tol(1e-4).unfloat(x)).map(|t| IntRot { t }) }

    pub fn from_cart(lattice: &Lattice, mat: &M33) -> Result<IntRot, ::IntPrecisionError>
    { Self::from_cart_t(lattice, &mat.t()) }

    fn from_cart_t(lattice: &Lattice, &cart_t: &M33) -> Result<IntRot, ::IntPrecisionError>
    { Self::from_frac_t(&frac_t_from_cart_t(lattice, cart_t)) }

    pub fn matrix(&self) -> M33<i32>
    { self.t.t() }

    /// Get the transpose of the cartesian rotation matrix, assuming that
    /// the the operator is expressed in units of the given lattice.
    pub fn cart_t(&self, lattice: &Lattice) -> M33
    { cart_t_from_frac_t(lattice, self.frac_t()) }

    /// Recover the cartesian rotation matrix, assuming that
    /// the the operator is expressed in units of the given lattice.
    pub fn cart(&self, lattice: &Lattice) -> M33
    { self.cart_t(lattice).t() }

    // transposed float matrix
    pub(crate) fn frac_t(&self) -> M33
    { self.t.map(Into::into) }
}

fn frac_t_from_cart_t(lattice: &Lattice, cart_t: M33) -> M33
{ lattice.matrix() * cart_t * lattice.inverse_matrix() }

fn cart_t_from_frac_t(lattice: &Lattice, frac_t: M33) -> M33
{ lattice.inverse_matrix() * frac_t * lattice.matrix() }

impl IntRot {
    /// Flipped group operator.
    ///
    /// `a.then(b) == b.of(a)`.  The flipped order is more aligned
    /// with this library's generally row-centric design.
    pub fn then(&self, other: &IntRot) -> IntRot
    {
        // (since these are transposes, this is the natural order of application)
        IntRot { t: self.t * other.t }
    }

    /// Conventional group operator.
    pub fn of(&self, other: &IntRot) -> IntRot
    { other.then(self) }
}

impl FracTrans {
    pub fn eye() -> Self
    { FracTrans(V3([0, 0, 0])) }

    fn opt_from_raw(raw: V3<i32>) -> Option<Self>
    {
        match raw.iter().all(|&x| 0 <= x && x < 12) {
            true => Some(FracTrans(raw)),
            false => None,
        }
    }

    pub fn from_floats(xs: V3) -> Result<FracTrans, ::IntPrecisionError>
    { xs.try_map(|x| ::util::Tol(1e-4).unfloat(x * 12.0)).map(FracTrans).map(Self::reduce) }

    fn reduce(self) -> FracTrans
    { FracTrans(self.0.map(|x| ::util::mod_euc(x, 12))) }

    pub fn from_cart(prim_lattice: &Lattice, cart: V3) -> Result<FracTrans, ::IntPrecisionError>
    { FracTrans::from_floats(cart / prim_lattice) }

    /// Recover the cartesian translation vector.
    ///
    /// This conversion requires the same primitive lattice that was used to compute this
    /// symmetry operator.
    pub fn cart(&self, prim_lattice: &Lattice) -> V3
    { self.float() * prim_lattice }

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

    pub fn new(rot: &IntRot, trans: &FracTrans) -> Self
    {
        let mut out = FRAC_OP_EYE;
        out[0][..3].copy_from_slice(&*rot.t[0]);
        out[1][..3].copy_from_slice(&*rot.t[1]);
        out[2][..3].copy_from_slice(&*rot.t[2]);
        out[3][..3].copy_from_slice(&*trans.0);
        FracOp { t: out.into() }
    }

    pub fn to_rot(&self) -> IntRot
    {
        let mut out = IntRot::eye();
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
            *x = ::util::mod_euc(*x, 12);
        }
        debug_assert!(t[3][..3].iter().all(|&x| 0 <= x && x < 12));
        debug_assert_eq!(t[3][3], 1);

        FracOp { t: t.into() }
    }

    /// Conventional group operator.
    pub fn of(&self, other: &FracOp) -> FracOp
    { other.then(self) }
}

impl IntRot {
    pub fn transform_fracs(&self, fracs: &[V3]) -> Vec<V3>
    { fracs.iter().map(|v| v * self.frac_t()).collect() }
}

impl<'a> From<&'a [[i32; 3]; 3]> for IntRot {
    fn from(m: &'a [[i32; 3]; 3]) -> Self
    { IntRot::new(&mat::from_array(*m)) }
}

impl FracTrans {
    pub fn transform_prim_mut(&self, fracs: &mut [V3])
    { ::util::translate_mut_n3_3(fracs, &self.float()) }
}

impl FracOp {
    pub fn transform_prim(&self, fracs: &[V3]) -> Vec<V3>
    {
        let mut out = self.to_rot().transform_fracs(fracs);
        self.to_trans().transform_prim_mut(&mut out);
        out
    }
}

mod serde_impls {
    use super::*;
    use ::serde::{Serialize, Serializer, Deserialize, Deserializer};
    use ::serde::de::Error;

    #[derive(Serialize, Deserialize)]
    pub struct RawFracOp {
        rot: IntRot,
        trans: FracTrans,
    }

    impl Serialize for IntRot {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let raw: M33<i32> = self.matrix();
            raw.serialize(serializer)
        }
    }

    impl Serialize for FracTrans {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let raw: V3<i32> = self.0;
            raw.serialize(serializer)
        }
    }

    impl Serialize for FracOp {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            RawFracOp {
                rot: self.to_rot(),
                trans: self.to_trans(),
            }.serialize(serializer)
        }
    }

    impl<'de> Deserialize<'de> for IntRot {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            let raw = M33::<i32>::deserialize(deserializer)?;
            IntRot::opt_new(&raw)
                .ok_or_else(|| D::Error::custom("matrix not unimodular"))
        }
    }

    impl<'de> Deserialize<'de> for FracTrans {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            let raw = V3::<i32>::deserialize(deserializer)?;
            FracTrans::opt_from_raw(raw)
                .ok_or_else(|| D::Error::custom("frac translation out of range"))
        }
    }

    impl<'de> Deserialize<'de> for FracOp {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            let RawFracOp { rot, trans } = RawFracOp::deserialize(deserializer)?;
            Ok(FracOp::new(&rot, &trans))
        }
    }

    #[cfg(test)]
    fn check_round_trip<T>(x: &T)
    where T: ::std::fmt::Debug + PartialEq + Serialize + for<'de> Deserialize<'de>,
    {
        let ser = ::serde_json::to_string(&x).unwrap();
        let de: T = ::serde_json::from_str(&ser).unwrap();
        assert_eq!(x, &de);
    }

    #[test]
    fn round_trip() {
        let r = IntRot::from(&[
            [-1, 1, 0],
            [-1, 0, 0],
            [ 0, 0, 1],
        ]);
        let t = FracTrans::from_floats(V3([0.0, 1.0/3.0, 1.0/6.0])).unwrap();
        check_round_trip(&r);
        check_round_trip(&t);
        check_round_trip(&FracOp::new(&r, &t));
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn bad_trans() {
        let _: FracTrans = ::serde_json::from_str("[0, -4, 0]").unwrap();
    }

    #[test]
    #[should_panic(expected = "unimodular")]
    fn bad_rot() {
        let _: IntRot = ::serde_json::from_str("[[-1, 1, 0], [-2, 0, 0], [0, 0, 1]]").unwrap();
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn bad_trans_in_op() {
        let _: FracOp = ::serde_json::from_str(r#"{
            "rot": [[-1, 1, 0], [-1, 0, 0], [ 0, 0, 1]],
            "trans": [0, -4, 0]
        }"#).unwrap();
    }

    #[test]
    #[should_panic(expected = "unimodular")]
    fn bad_rot_in_op() {
        let _: FracOp = ::serde_json::from_str(r#"{
            "rot": [[-1, 1, 0], [-2, 0, 0], [ 0, 0, 1]],
            "trans": [0, 3, 6]
        }"#).unwrap();
    }
}

#[cfg(test)]
#[deny(unused)]
mod tests {
    use super::*;
    use ::rsp2_array_types::{mat, Unvee, Envee};

    #[test]
    fn rot_transform()
    {
        let r = [
            [0, -1, 0],
            [1,  0, 0],
            [0,  0, 1],
        ];
        assert_eq!(
            IntRot::from(&r).transform_fracs([[1.0, 5.0, 7.0]].envee_ref()),
            vec![[-5.0, 1.0, 7.0]].envee(),
        );
    }

    #[test]
    fn two_transform()
    {
        // two operations that don't commute
        let xy = IntRot::from(&[
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ]);
        let zx = IntRot::from(&[
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
        ]);
        let xyzx = IntRot::from(&[
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
        ]);
        // a primitive structure that is sensitive to any permutations of the axes
        let prim = vec![[1., 2., 3.]].envee();
        assert_eq!(xy.then(&zx), xyzx);
        assert_eq!(zx.of(&xy), xyzx);
        assert_eq!(
            zx.transform_fracs(&xy.transform_fracs(&prim)),
            xyzx.transform_fracs(&prim),
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
            &IntRot::from(&[
                [ 0,  1, 0],
                [-1,  1, 0],
                [ 0,  0, 1],
            ]),
            &FracTrans::from_floats(V3([1./3., 2./3., 0.0])).unwrap(),
        );
        let square = FracOp::new(
            &IntRot::from(&[
                [-1, 1, 0],
                [-1, 0, 0],
                [ 0, 0, 1],
            ]),
            &FracTrans::from_floats(V3([0., 0., 0.])).unwrap(),
        );

        assert_eq!(op.then(&op), square);
    }

    #[test]
    fn symmop_to_cart_from_cart()
    {
        // graphene lattice
        let half_r3 = 0.5 * f64::sqrt(3.0);
        let lattice = Lattice::new(&(2.4 * &mat::from_array([
            [ 1.0,     0.0, 0.0],
            [-0.5, half_r3, 0.0],
            [ 0.0,     0.0, 1.0],
        ])));
        // threefold rotation
        let op = IntRot::from(&[
            [-1, 1, 0],
            [-1, 0, 0],
            [ 0, 0, 1],
        ]);
        assert_close!(op.cart(&lattice).unvee(), [
            [    -0.5, half_r3, 0.0],
            [-half_r3,    -0.5, 0.0],
            [     0.0,     0.0, 1.0],
        ]);
        assert_close!(op.cart_t(&lattice).unvee(), [
            [   -0.5, -half_r3, 0.0],
            [half_r3,     -0.5, 0.0],
            [    0.0,      0.0, 1.0],
        ]);

        // one with nonzero translation
        let rot = IntRot::from(&[
            [ 0, 1,  0],
            [-1, 1,  0],
            [ 0, 0, -1],
        ]);
        let trans = FracTrans::from_floats(V3([-1./3., 1./3., 0.0])).unwrap();
        assert_eq!(rot, IntRot::from_cart(&lattice, &rot.cart(&lattice)).unwrap());
        assert_eq!(trans, FracTrans::from_cart(&lattice, trans.cart(&lattice)).unwrap());
    }
}
