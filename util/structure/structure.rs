use ::{Lattice, Coords};

/// Pairs [`Coords`] together with their [`Lattice`] and metadata.
///
/// [`Coords`]: ../struct.Coords.html
/// [`Lattice`]: ../struct.Lattice.html
#[derive(Debug,Clone,PartialEq)]
pub struct Structure<M = ()> {
    pub(crate) lattice: Lattice,
    pub(crate) coords: Coords,
    pub(crate) meta: Vec<M>,
}

/// Type of a Structure with no metadata at all (not even atom types).
pub type CoordStructure = Structure<()>;

impl CoordStructure {
    /// Create a structure with no metadata; just coordinates.
    pub fn new_coords(lattice: Lattice, coords: Coords) -> Self {
        let meta = vec![(); coords.len()];
        Self::new(lattice, coords, meta)
    }
}

impl<M> Structure<M> {
    pub fn new(lattice: Lattice, coords: Coords, meta: Vec<M>) -> Self {
        assert_eq!(coords.len(), meta.len());
        Self { lattice, coords, meta }
    }

    pub fn num_atoms(&self) -> usize { self.coords.len() }
    pub fn lattice(&self) -> &Lattice { &self.lattice }
    pub fn metadata(&self) -> &[M] { &self.meta }
}

// to/as/mut for coords
impl<M> Structure<M> {
    // NOTE: We can produce `Vec<_>` and `&mut [_]`,
    //       but not `&[_]` because the data might not be present.
    //       (`&mut [_]` works because can insert the data)
    pub fn to_carts(&self) -> Vec<[f64; 3]> { self.coords.to_carts(&self.lattice) }
    pub fn to_fracs(&self) -> Vec<[f64; 3]> { self.coords.to_fracs(&self.lattice) }

    pub fn carts_mut(&mut self) -> &mut [[f64; 3]] {
        self.ensure_carts();
        match self.coords {
            Coords::Fracs(_) => unreachable!(),
            Coords::Carts(ref mut c) => c,
        }
    }

    pub fn fracs_mut(&mut self) -> &mut [[f64; 3]] {
        self.ensure_fracs();
        match self.coords {
            Coords::Fracs(ref mut c) => c,
            Coords::Carts(_) => unreachable!(),
        }
    }

    /// Replace the coordinates in the structure.
    ///
    /// # Panics
    /// Panics if the length does not match.
    pub fn set_coords(&mut self, coords: Coords) {
        assert_eq!(self.coords.len(), coords.len());
        self.coords = coords;
    }

    /// # Panics
    /// Panics if the length does not match.
    pub fn with_coords(mut self, coords: Coords) -> Self { self.set_coords(coords); self }

    /// Ensures that the cartesian coordinates are cached if they aren't already.
    pub fn ensure_carts(&mut self) {
        let dummy = Coords::Carts(vec![]);
        let coords = ::std::mem::replace(&mut self.coords, dummy);
        self.coords = Coords::Carts(coords.into_carts(&self.lattice));
    }

    /// Ensures that the fractional coordinates are cached if they aren't already.
    pub fn ensure_fracs(&mut self) {
        let dummy = Coords::Carts(vec![]);
        let coords = ::std::mem::replace(&mut self.coords, dummy);
        self.coords = Coords::Fracs(coords.into_fracs(&self.lattice));
    }
}