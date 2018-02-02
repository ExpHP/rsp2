use ::Lattice;
use ::oper::{Perm, Permute};
use ::oper::{Part, Partition};
use ::oper::part::Unlabeled;

/// Wrapper type for coordinates used as input to some APIs.
///
/// This allows a function to support either cartesian coordinates,
/// or fractional coordinates with respect to some lattice.
#[derive(Debug, Clone, PartialEq)]
pub enum Coords {
    Carts(Vec<[f64; 3]>),
    Fracs(Vec<[f64; 3]>),
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum Tag { Cart, Frac }

impl Coords {
    pub fn len(&self) -> usize
    { self.as_slice().1.len() }

    pub(crate) fn as_slice(&self) -> (Tag, &[[f64; 3]])
    { match *self {
        Coords::Carts(ref c) => (Tag::Cart, c),
        Coords::Fracs(ref c) => (Tag::Frac, c),
    }}

    pub(crate) fn as_mut_vec(&mut self) -> (Tag, &mut Vec<[f64; 3]>)
    { match *self {
        Coords::Carts(ref mut c) => (Tag::Cart, c),
        Coords::Fracs(ref mut c) => (Tag::Frac, c),
    }}

    pub(crate) fn into_vec(self) -> (Tag, Vec<[f64; 3]>)
    { match self {
        Coords::Carts(c) => (Tag::Cart, c),
        Coords::Fracs(c) => (Tag::Frac, c),
    }}

    pub(crate) fn from_vec(tag: Tag, c: Vec<[f64; 3]>) -> Self
    { match tag {
        Tag::Cart => Coords::Carts(c),
        Tag::Frac => Coords::Fracs(c),
    }}
}

// projections
impl Coords {
    pub(crate) fn as_carts_opt(&self) -> Option<&[[f64; 3]]>
    { match *self {
        Coords::Carts(ref x) => Some(x),
        Coords::Fracs(_) => None,
    }}

    pub(crate) fn as_fracs_opt(&self) -> Option<&[[f64; 3]]>
    { match *self {
        Coords::Carts(_) => None,
        Coords::Fracs(ref x) => Some(x),
    }}
}

// conversions
impl Coords {
    pub fn into_carts(self, lattice: &Lattice) -> Vec<[f64; 3]>
    { match self {
        Coords::Carts(c) => c,
        Coords::Fracs(c) => ::util::dot_n3_33(&c, &lattice.matrix()),
    }}

    pub fn into_fracs(self, lattice: &Lattice) -> Vec<[f64; 3]>
    { match self {
        Coords::Carts(c) => ::util::dot_n3_33(&c, &lattice.inverse_matrix()),
        Coords::Fracs(c) => c,
    }}

    pub fn to_carts(&self, lattice: &Lattice) -> Vec<[f64; 3]>
    { match *self {
        Coords::Carts(ref c) => c.clone(),
        Coords::Fracs(ref c) => ::util::dot_n3_33(c, &lattice.matrix()),
    }}

    pub fn to_fracs(&self, lattice: &Lattice) -> Vec<[f64; 3]>
    { match *self {
        Coords::Carts(ref c) => ::util::dot_n3_33(c, &lattice.inverse_matrix()),
        Coords::Fracs(ref c) => c.clone(),
    }}

    pub(crate) fn into_tag(self, tag: Tag, lattice: &Lattice) -> Vec<[f64; 3]>
    { match tag {
        Tag::Cart => self.into_carts(lattice),
        Tag::Frac => self.into_fracs(lattice),
    }}

    #[allow(unused)]
    pub(crate) fn to_tag(&self, tag: Tag, lattice: &Lattice) -> Vec<[f64; 3]>
    { match tag {
        Tag::Cart => self.to_carts(lattice),
        Tag::Frac => self.to_fracs(lattice),
    }}
}

impl Permute for Coords {
    fn permuted_by(self, perm: &Perm) -> Coords
    { match self {
        Coords::Carts(c) => Coords::Carts(c.permuted_by(perm)),
        Coords::Fracs(c) => Coords::Fracs(c.permuted_by(perm)),
    }}
}

impl Partition for Coords {
    fn into_unlabeled_partitions<L>(self, part: &Part<L>) -> Unlabeled<Self>
    {
        let (tag, coords) = self.into_vec();
        Box::new(coords.into_unlabeled_partitions(part).map(move |c| Self::from_vec(tag, c)))
    }
}

#[cfg(test)]
#[deny(unused)]
mod tests {

    // make sure the library correctly chooses whether to use the
    // regular matrix, the inverse matrix, or no matrix
    #[test]
    fn div_vs_mul() {
        use ::Lattice;
        use ::Coords::{Fracs, Carts};

        let x = |mag| vec![[mag, 0.0, 0.0]];
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
        use ::Lattice;
        use ::Coords::{Fracs, Carts};

        // a matrix not equal to its transpose
        let lattice = Lattice::new(&[
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ]);

        // what happens to [1,0,0] when we interpret it in one coord system
        //  and then convert to the other system
        let input = vec![[1.0, 0.0, 0.0]];
        let frac_to_cart = vec![[0.0, 1.0, 0.0]];
        let cart_to_frac = vec![[0.0, 0.0, 1.0]];

        assert_eq!(&frac_to_cart, &Fracs(input.clone()).to_carts(&lattice));
        assert_eq!(&frac_to_cart, &Fracs(input.clone()).into_carts(&lattice));
        assert_eq!(&cart_to_frac, &Carts(input.clone()).to_fracs(&lattice));
        assert_eq!(&cart_to_frac, &Carts(input.clone()).into_fracs(&lattice));
    }
}
