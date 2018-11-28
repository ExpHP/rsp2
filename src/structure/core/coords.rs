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

use crate::Lattice;
use rsp2_soa_ops::{Perm, Permute};
use rsp2_soa_ops::{Part, Partition, Unlabeled};

use rsp2_array_types::{V3, M33};

/// Wrapper type for coordinates used as input to some APIs.
///
/// This allows a function to support either cartesian coordinates,
/// or fractional coordinates with respect to some lattice.
///
/// Direct usage of this type is uncommon, as `Coords` (which includes the
/// lattice) is usually the preferable choice.  An example use case for
/// `CoordsKind` might be in a function signature that takes two sets of
/// coordinates sharing a single lattice.
//
// NOTE: The type parameter here was originally introduced as a step towards
//       an HList-based Structure type. While those plans have been scrapped,
//       the type parameter itself is still ever-so-slightly useful because it
//       lets you do `CoordsKind::Carts(&v[..]).to_fracs(lattice)` without
//       having to allocate an intermediate vec for the carts.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum CoordsKind<V = Vec<V3>> {
    /// Data that is expressed in a Euclidean basis.
    ///
    /// This includes:
    ///
    /// * Cartesian coordinates, in units of length.
    /// * Reciprocal cartesian coordinates, in units of inverse length
    ///   **without the 2 PI factor**.
    /// * Partial derivatives of a function with respect to cartesian coordinates.
    ///   (These conveniently transform like reciprocal cartesian coords.)
    ///
    /// ...or basically, anything represented in a form where "distance" is a
    /// meaningful concept, and where normalization can be meaningfully performed.
    #[cfg_attr(feature = "serde", serde(rename = "carts"))]
    Carts(V),

    /// Data that is expressed in a fractional basis.
    ///
    /// This includes:
    ///
    /// * Fractional coordinates, as coefficients of lattice vectors.
    /// * Reciprocal fractional coordinates, as coefficients of a reciprocal lattice.
    /// * Partial derivatives of a function with respect to fractional coordinates.
    ///   (These conveniently transform like reciprocal fractional coords.)
    ///
    /// Generally speaking, these are representations where integers have a
    /// special meaning, and where distance is poorly defined.
    #[cfg_attr(feature = "serde", serde(rename = "fracs"))]
    Fracs(V),
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum Tag { Cart, Frac }

#[allow(unused)]
impl<V> CoordsKind<V>
where V: AsRef<[V3]>,
{
    #[inline(always)]
    pub fn len(&self) -> usize
    { self.as_slice().1.len() }

    pub(crate) fn as_slice(&self) -> (Tag, &[V3])
    { match self {
        CoordsKind::Carts(c) => (Tag::Cart, c.as_ref()),
        CoordsKind::Fracs(c) => (Tag::Frac, c.as_ref()),
    }}
}

impl<V> CoordsKind<V>
where V: AsRef<[V3]>,
{
    pub fn as_ref(&self) -> CoordsKind<&[V3]>
    { match self {
        &CoordsKind::Carts(ref c) => CoordsKind::Carts(c.as_ref()),
        &CoordsKind::Fracs(ref c) => CoordsKind::Fracs(c.as_ref()),
    }}
}

#[allow(unused)]
impl CoordsKind<Vec<V3>>
{
    pub(crate) fn as_mut_vec(&mut self) -> (Tag, &mut Vec<V3>)
    { match self {
        CoordsKind::Carts(c) => (Tag::Cart, c),
        CoordsKind::Fracs(c) => (Tag::Frac, c),
    }}

    pub(crate) fn into_vec(self) -> (Tag, Vec<V3>)
    { match self {
        CoordsKind::Carts(c) => (Tag::Cart, c),
        CoordsKind::Fracs(c) => (Tag::Frac, c),
    }}

    pub(crate) fn from_vec(tag: Tag, c: Vec<V3>) -> Self
    { match tag {
        Tag::Cart => CoordsKind::Carts(c),
        Tag::Frac => CoordsKind::Fracs(c),
    }}
}

// projections
#[allow(unused)]
impl<V> CoordsKind<V>
where V: AsRef<[V3]>,
{
    pub(crate) fn as_carts_opt(&self) -> Option<&[V3]>
    { match self {
        CoordsKind::Carts(x) => Some(x.as_ref()),
        CoordsKind::Fracs(_) => None,
    }}

    pub(crate) fn as_fracs_opt(&self) -> Option<&[V3]>
    { match self {
        CoordsKind::Carts(_) => None,
        CoordsKind::Fracs(x) => Some(x.as_ref()),
    }}
}

// conversions
impl CoordsKind<Vec<V3>>
{
    pub fn into_carts(self, lattice: &Lattice) -> Vec<V3>
    { match self {
        CoordsKind::Carts(c) => c,
        CoordsKind::Fracs(c) => dot_n3_33(&c, lattice.matrix()),
    }}

    pub fn into_fracs(self, lattice: &Lattice) -> Vec<V3>
    { match self {
        CoordsKind::Carts(c) => dot_n3_33(&c, lattice.inverse_matrix()),
        CoordsKind::Fracs(c) => c,
    }}

    #[allow(unused)]
    pub(crate) fn into_tag(self, tag: Tag, lattice: &Lattice) -> Vec<V3>
    { match tag {
        Tag::Cart => self.into_carts(lattice),
        Tag::Frac => self.into_fracs(lattice),
    }}
}

impl<V> CoordsKind<V>
where V: AsRef<[V3]>,
{
    pub fn to_carts(&self, lattice: &Lattice) -> Vec<V3>
    { match self {
        CoordsKind::Carts(c) => c.as_ref().to_vec(),
        CoordsKind::Fracs(c) => dot_n3_33(c.as_ref(), lattice.matrix()),
    }}

    pub fn to_fracs(&self, lattice: &Lattice) -> Vec<V3>
    { match self {
        CoordsKind::Carts(c) => dot_n3_33(c.as_ref(), lattice.inverse_matrix()),
        CoordsKind::Fracs(c) => c.as_ref().to_vec(),
    }}

    #[allow(unused)]
    pub(crate) fn to_tag(&self, tag: Tag, lattice: &Lattice) -> Vec<V3>
    { match tag {
        Tag::Cart => self.to_carts(lattice),
        Tag::Frac => self.to_fracs(lattice),
    }}
}

fn dot_n3_33(c: &[V3], m: &M33) -> Vec<V3>
{ c.iter().map(|v| v * m).collect() }

impl<V> Permute for CoordsKind<V>
where V: Permute,
{
    fn permuted_by(self, perm: &Perm) -> CoordsKind<V>
    { match self {
        CoordsKind::Carts(c) => CoordsKind::Carts(c.permuted_by(perm)),
        CoordsKind::Fracs(c) => CoordsKind::Fracs(c.permuted_by(perm)),
    }}
}

impl<'iter> Partition<'iter> for CoordsKind<Vec<V3>>
{
    fn into_unlabeled_partitions<L>(self, part: &'iter Part<L>) -> Unlabeled<'iter, Self>
    {
        let (tag, coords) = self.into_vec();
        Box::new(coords.into_unlabeled_partitions(part).map(move |c| Self::from_vec(tag, c)))
    }
}

#[cfg(test)]
#[deny(unused)]
mod tests {
    use crate::Lattice;
    use crate::CoordsKind::{Fracs, Carts};

    use rsp2_array_types::Envee;

    // make sure the library correctly chooses whether to use the
    // regular matrix, the inverse matrix, or no matrix
    #[test]
    fn div_vs_mul() {

        let x = |mag| vec![[mag, 0.0, 0.0]].envee();
        let lattice = Lattice::cubic(2.0);

        assert_eq!(x(1.0), Fracs(x(1.0)).to_fracs(&lattice));
        assert_eq!(x(1.0), Fracs(x(1.0)).into_fracs(&lattice));
        assert_eq!(x(2.0), Fracs(x(1.0)).to_carts(&lattice));
        assert_eq!(x(2.0), Fracs(x(1.0)).into_carts(&lattice));

        assert_eq!(x(0.5), Carts(x(1.0)).to_fracs(&lattice));
        assert_eq!(x(0.5), Carts(x(1.0)).into_fracs(&lattice));
        assert_eq!(x(1.0), Carts(x(1.0)).to_carts(&lattice));
        assert_eq!(x(1.0), Carts(x(1.0)).into_carts(&lattice));
    }

    // make sure matrix multiplication is done in the correct order
    #[test]
    fn multiplication_order() {

        // a matrix not equal to its transpose
        let lattice = Lattice::from(&[
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ]);

        // what happens to [1,0,0] when we interpret it in one coord system
        //  and then convert to the other system
        let input = vec![[1.0, 0.0, 0.0]].envee();
        let frac_to_cart = vec![[0.0, 1.0, 0.0]].envee();
        let cart_to_frac = vec![[0.0, 0.0, 1.0]].envee();

        assert_eq!(&frac_to_cart, &Fracs(input.clone()).to_carts(&lattice));
        assert_eq!(&frac_to_cart, &Fracs(input.clone()).into_carts(&lattice));
        assert_eq!(&cart_to_frac, &Carts(input.clone()).to_fracs(&lattice));
        assert_eq!(&cart_to_frac, &Carts(input.clone()).into_fracs(&lattice));
    }
}
