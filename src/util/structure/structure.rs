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



    // FIXME bad idea for stable interface, but good enough for now
    pub fn metadata(&self) -> &[M] { &self.meta }
    pub fn map_metadata<M2, F>(self, mut f: F) -> Structure<M2>
    where F: FnMut(M) -> M2
    {
        let Structure { lattice, coords, meta } = self;
        let meta = meta.into_iter().map(f).collect();
        Structure { lattice, coords, meta }
    }
}

/// Functions for rescaling the structure.
///
/// These functions preserve fractional position while changing the lattice.
impl<M> Structure<M> {
    pub fn set_lattice(&mut self, lattice: &Lattice) {
        self.ensure_only_fracs();
        self.lattice = lattice.clone();
    }

    /// Individually multiply each lattice vector by a scale factor
    pub fn scale_vecs(&mut self, scale: &[f64; 3]) {
        let lattice = &Lattice::diagonal(scale) * &self.lattice;
        self.set_lattice(&lattice);
    }
}

// messing around with coords
impl<M> Structure<M> {
    // NOTE: We can produce `Vec<_>` and `&mut [_]`,
    //       but not `&[_]` because the data might not be present.
    //       (`&mut [_]` works because can insert the data)
    pub fn to_carts(&self) -> Vec<[f64; 3]> { self.coords.to_carts(&self.lattice) }
    pub fn to_fracs(&self) -> Vec<[f64; 3]> { self.coords.to_fracs(&self.lattice) }

    pub fn carts_mut(&mut self) -> &mut [[f64; 3]] {
        self.ensure_only_carts(); // 'only' because user modifications will invalidate fracs
        match self.coords {
            Coords::Fracs(_) => unreachable!(),
            Coords::Carts(ref mut c) => c,
        }
    }

    pub fn fracs_mut(&mut self) -> &mut [[f64; 3]] {
        self.ensure_only_fracs(); // 'only' because user modifications will invalidate carts
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
    pub fn set_carts(&mut self, carts: Vec<[f64; 3]>) { self.set_coords(Coords::Carts(carts)); }
    pub fn set_fracs(&mut self, fracs: Vec<[f64; 3]>) { self.set_coords(Coords::Fracs(fracs)); }


    /// # Panics
    /// Panics if the length does not match.
    pub fn with_coords(mut self, coords: Coords) -> Self { self.set_coords(coords); self }
    pub fn with_carts(mut self, carts: Vec<[f64; 3]>) -> Self { self.set_carts(carts); self }
    pub fn with_fracs(mut self, fracs: Vec<[f64; 3]>) -> Self { self.set_fracs(fracs); self }

    /// Ensures that the cartesian coordinates are cached if they aren't already.
    pub fn ensure_carts(&mut self) { self.ensure_only_carts(); }
    /// Ensures that the fractional coordinates are cached if they aren't already.
    pub fn ensure_fracs(&mut self) { self.ensure_only_fracs(); }

    /// Ensure that carts are available, and that fracs are NOT available.
    ///
    /// Currently equivalent to 'ensure_carts,' but that method may eventually
    /// retain the fracs.
    fn ensure_only_carts(&mut self) {
        let dummy = Coords::Carts(vec![]);
        let coords = ::std::mem::replace(&mut self.coords, dummy);
        self.coords = Coords::Carts(coords.into_carts(&self.lattice));
    }

    /// Ensure that fracs are available, and that carts are NOT available.
    ///
    /// Currently equivalent to 'ensure_fracs,' but that method may eventually
    /// retain the carts.
    fn ensure_only_fracs(&mut self) {
        let dummy = Coords::Carts(vec![]);
        let coords = ::std::mem::replace(&mut self.coords, dummy);
        self.coords = Coords::Fracs(coords.into_fracs(&self.lattice));
    }
}
