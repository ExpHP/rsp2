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

use ::{Coords, CoordsKind, Lattice};
use ::std::fmt;

use ::rsp2_array_types::{V3, M33, mat};

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
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Ord, PartialOrd)]
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

/// A space group operation in cartesian units.
///
/// The cartesian representation allows CartOps to be transferable between
/// primitive cells and supercells. More specifically, it is valid for any
/// (non-strict) supercell for which the point group of the superlattice contains
/// the operator. This necessarily includes all possible choices of equivalent
/// primitive cells (since they all share the same lattice).
///
/// (**Note:** if you apply it to a supercell which violates this rule, you may
///  witness multiple atoms being mapped into the same position--even though
///  *physically speaking* the operator ought to be a symmetry.)
///
/// Be aware that almost any change to the coordinate data will invalidate CartOps.
/// Obviously, the motion of individual atoms can break symmetry, but there is more
/// to it than that:
///
/// * uniform translations of the structure require a similarity transform
///   to recenter the rotations
/// * uniform rotations of the structure also require a similarity transform.
///   (picture a rotation which permutes the XYZ axes of a 2D structure,
///    and the issue here should be self-evident!)
/// * uniform scaling by a constant factor requires rescaling the translations
///
/// The issues with uniform translations and rotations would remain true even if we
/// used a fractional representation.
#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub struct CartOp {
    /// The transpose of the rotation matrix.
    rot_t: M33,
    trans: V3,
}

impl Default for IntRot {
    fn default() -> Self
    { Self::eye() }
}

impl Default for CartOp {
    fn default() -> Self
    { Self::eye() }
}

#[derive(Debug, Fail)]
pub struct LatticeSymmetryError {
    lattice: Lattice,
    cart_rot: M33,
    #[cause]
    cause: ::IntPrecisionError,
}

impl fmt::Display for LatticeSymmetryError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f, "{:?} is not in the point group of the lattice {:?}",
            self.cart_rot, self.lattice.matrix(),
        )
    }
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

    /// Obtain an integer representation of a point group operation, in units
    /// of a particular lattice.
    ///
    /// # Errors
    ///
    /// Returns an error if the rotation part of `self` is not a symmetry
    /// of the lattice. (using an approximate numerical test with a *generous*
    /// tolerance)
    pub fn from_cart(lattice: &Lattice, mat: &M33) -> Result<IntRot, LatticeSymmetryError>
    { Self::from_cart_t(lattice, &mat.t()) }

    fn from_cart_t(lattice: &Lattice, cart_t: &M33) -> Result<IntRot, LatticeSymmetryError>
    {
        Self::from_frac_t(&frac_t_from_cart_t(lattice, *cart_t))
            .map_err(|cause| {
                LatticeSymmetryError {
                    lattice: lattice.clone(),
                    cart_rot: cart_t.t(),
                    cause,
                }
            })
    }

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

    /// Recover a cartesian operation with zero translation, assuming that
    /// the the operator is expressed in units of the given lattice.
    ///
    /// To construct one with a cartesian translation, follow up with `.then_translate()`.
    /// To construct one with a fractional translation, use `.to_cart_op_with_frac_trans()`.
    ///
    /// This conversion requires the same primitive lattice that was used to compute this
    /// symmetry operator.
    pub fn to_cart_op(&self, lattice: &Lattice) -> CartOp
    { CartOp {
        rot_t: self.cart_t(lattice),
        trans: V3::zero(),
    }}

    /// This helper is provided because otherwise all possible methods of doing this
    /// require naming the lattice twice.
    ///
    /// Enjoy the long name, sucker.
    pub fn to_cart_op_with_frac_trans(&self, trans: V3, lattice: &Lattice) -> CartOp
    { self.to_cart_op(lattice).then_translate(trans * lattice) }
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

impl ::std::ops::Mul<V3<i32>> for IntRot {
    type Output = V3<i32>;

    fn mul(self, v: V3<i32>) -> Self::Output
    { v * self.t }
}

impl CartOp {
    pub fn eye() -> Self
    { Self {
        rot_t: M33::eye(),
        trans: V3::zero(),
    }}

    pub fn new(rot: &M33, trans: V3) -> Self
    { CartOp { rot_t: rot.t(), trans } }

    /// Obtain an integer representation of the associated point group operation,
    /// in units of a particular lattice.
    ///
    /// # Errors
    ///
    /// Returns an error if the rotation part of `self` is not a symmetry of the
    /// lattice. (using an approximate numerical test with a *generous* tolerance)
    pub fn int_rot(&self, lattice: &Lattice) -> Result<IntRot, LatticeSymmetryError>
    { IntRot::from_cart_t(lattice, &self.rot_t) }

    pub fn cart_rot_t(&self) -> M33
    { self.rot_t }

    pub fn cart_rot(&self) -> M33
    { self.rot_t.t() }

    pub fn cart_trans(&self) -> V3
    { self.trans }

    fn frac_rot_t(&self, lattice: &Lattice) -> M33
    { frac_t_from_cart_t(lattice, self.rot_t) }

    /// Compose with a translation.
    pub fn then_translate(&self, cart: V3) -> Self
    {
        let mut out = *self;
        out.trans += cart;
        out
    }
}

impl CartOp {
    /// Flipped group operator.
    ///
    /// `a.then(b) == b.of(a)`.  The flipped order is more aligned
    /// with this library's generally row-centric design.
    pub fn then(&self, other: &CartOp) -> CartOp
    { CartOp {
        rot_t: self.rot_t * other.rot_t,
        trans: self.trans * other.rot_t + other.trans,
    }}

    /// Conventional group operator.
    pub fn of(&self, other: &CartOp) -> CartOp
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

impl From<[[i32; 3]; 3]> for IntRot {
    fn from(m: [[i32; 3]; 3]) -> Self
    { IntRot::from(&m) }
}

impl CartOp {
    pub fn transform_carts(&self, carts: &[V3]) -> Vec<V3>
    { carts.iter().map(|v| v * self.rot_t + self.trans).collect() }

    pub fn transform_fracs(&self, lattice: &Lattice, fracs: &[V3]) -> Vec<V3>
    {
        let rot_t = self.frac_rot_t(lattice);
        let trans = self.trans / lattice;
        fracs.iter().map(|v| v * rot_t + trans).collect()
    }

    pub fn transform(&self, coords: &Coords) -> Coords
    {
        let Coords { lattice, coords } = coords;
        Coords::new(lattice.clone(), match coords {
            CoordsKind::Fracs(vs) => CoordsKind::Fracs(self.transform_fracs(lattice, vs)),
            CoordsKind::Carts(vs) => CoordsKind::Carts(self.transform_carts(vs)),
        })
    }
}

#[cfg(feature = "serde")]
mod serde_impls {
    use super::*;
    use ::serde::{Serialize, Serializer, Deserialize, Deserializer};
    use ::serde::de::Error;

    #[derive(Serialize, Deserialize)]
    pub struct RawCartOp {
        rot: M33,
        trans: V3,
    }

    impl Serialize for IntRot {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let raw: M33<i32> = self.matrix();
            raw.serialize(serializer)
        }
    }

    impl Serialize for CartOp {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            RawCartOp {
                rot: self.cart_rot(),
                trans: self.cart_trans(),
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

    impl<'de> Deserialize<'de> for CartOp {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            let RawCartOp { rot, trans } = RawCartOp::deserialize(deserializer)?;
            Ok(CartOp::new(&rot, trans))
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
        let rot = IntRot::from([
            [-1, 1, 0],
            [-1, 0, 0],
            [ 0, 0, 1],
        ]);
        check_round_trip(&rot);

        let lattice = super::tests::graphene_lattice();
        let op = rot.to_cart_op(&lattice).then_translate(V3([0.6, 7.7777, -2.1]));
        check_round_trip(&op);
    }

    #[test]
    #[should_panic(expected = "unimodular")]
    fn bad_rot() {
        let _: IntRot = ::serde_json::from_str("[[-1, 1, 0], [-2, 0, 0], [0, 0, 1]]").unwrap();
    }
}

#[cfg(test)]
#[deny(unused)]
mod tests {
    use super::*;
    use ::rsp2_array_types::{mat, Unvee, Envee};
    use CoordsKind;

    pub(super) fn graphene_lattice() -> Lattice {
        let half_r3 = 0.5 * f64::sqrt(3.0);
        Lattice::new(&(2.4 * &mat::from_array([
            [ 1.0,     0.0, 0.0],
            [-0.5, half_r3, 0.0],
            [ 0.0,     0.0, 1.0],
        ])))
    }

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

    fn check_cart_ops_close(a: CartOp, b: CartOp, lattice: &Lattice) {
        assert_close!(rel=1e-14, a.cart_rot().unvee(), b.cart_rot().unvee());

        let diff = (a.cart_trans() - b.cart_trans()) / lattice;
        let diff = diff.map(|x| x - x.round());
        assert!((diff * lattice).sqnorm() < 1e-14, "{}", (diff * lattice).sqnorm());
    }

    #[test]
    fn composition()
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
        let fracs = vec![[1., 2., 3.]].envee();
        assert_eq!(xy.then(&zx), xyzx);
        assert_eq!(zx.of(&xy), xyzx);
        assert_eq!(
            zx.transform_fracs(&xy.transform_fracs(&fracs)),
            xyzx.transform_fracs(&fracs),
        );

        let lattice = graphene_lattice();
        let carts = CoordsKind::Fracs(fracs.clone()).to_carts(&lattice);

        let xy = xy.to_cart_op_with_frac_trans(V3([0.01, 0.02, 0.03]) , &lattice);
        let zx = zx.to_cart_op_with_frac_trans(V3([0.10, 0.20, 0.30]), &lattice);
        let xyzx = xyzx.to_cart_op_with_frac_trans(V3([0.13, 0.22, 0.31]), &lattice);

        check_cart_ops_close(xy.then(&zx), xyzx, &lattice);
        check_cart_ops_close(zx.of(&xy), xyzx, &lattice);
        assert_close!(
            rel=1e-14,
            zx.transform_fracs(&lattice, &xy.transform_fracs(&lattice, &fracs)).unvee(),
            xyzx.transform_fracs(&lattice, &fracs).unvee(),
        );
        assert_close!(
            rel=1e-14,
            zx.transform_carts(&xy.transform_carts(&carts)).unvee(),
            xyzx.transform_carts(&carts).unvee(),
        );
    }

    #[test]
    fn symmop_mul()
    {
        let lattice = Lattice::eye();
        let op_rot = IntRot::from(&[
            [ 0,  1, 0],
            [-1,  1, 0],
            [ 0,  0, 1],
        ]);
        let op = op_rot.to_cart_op(&lattice).then_translate(V3([1./3., 2./3., 0.0]));

        let square_rot = IntRot::from(&[
            [-1, 1, 0],
            [-1, 0, 0],
            [ 0, 0, 1],
        ]);
        let square = square_rot.to_cart_op(&lattice);

        assert_eq!(op_rot.then(&op_rot), square_rot);
        check_cart_ops_close(op.then(&op), square, &Lattice::eye());
    }

    #[test]
    fn int_rot_to_cart_from_cart()
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

        // CartOp
        let rot = IntRot::from(&[
            [ 0, 1,  0],
            [-1, 1,  0],
            [ 0, 0, -1],
        ]);
        let op = rot.to_cart_op_with_frac_trans(V3([-1./3., 1./3., 0.0]), &lattice);
        assert_eq!(rot, IntRot::from_cart(&lattice, &op.cart_rot()).unwrap());
    }
}
