use ::Lattice;

/// Wrapper type for coordinates used as input to some APIs.
///
/// This allows a function to support either cartesian coordinates,
/// or fractional coordinates with respect to some lattice.
#[derive(Debug, Clone, PartialEq)]
pub enum Coords {
    Carts(Vec<[f64; 3]>),
    Fracs(Vec<[f64; 3]>),
}

impl Coords {
    pub fn len(&self) -> usize
    { self.as_slice().len() }

    pub fn as_slice(&self) -> &[[f64; 3]]
    { match *self {
        Coords::Carts(ref c) => c,
        Coords::Fracs(ref c) => c,
    }}
}

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
