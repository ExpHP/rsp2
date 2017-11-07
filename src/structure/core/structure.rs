use ::{Lattice, Coords, Element, SentLattice};
use ::errors::*;
use ::oper::{Perm, Permute};

/// Pairs [`Coords`] together with their [`Lattice`], and metadata.
///
/// Currently the metadata story is pretty weak and hardly resembles
/// any sort of final design.  You should let the `M` type contain all
/// of the information you need, and then when certain functions request
/// a certain `M` type, use `map_metadata_to` to extract that information.
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

/// A Structure whose only metadata is atomic numbers.
pub type ElementStructure = Structure<Element>;

impl CoordStructure {
    /// Create a structure with no metadata; just coordinates.
    pub fn new_coords(lattice: Lattice, coords: Coords) -> Self {
        let meta = vec![(); coords.len()];
        Self::new(lattice, coords, meta)
    }
}

impl<M> Structure<M> {
    pub fn new<Ms>(lattice: Lattice, coords: Coords, meta: Ms) -> Self
    where Ms: IntoIterator<Item=M>,
    {
        let meta: Vec<_> = meta.into_iter().collect();
        assert_eq!(coords.len(), meta.len());
        Self { lattice, coords, meta }
    }

    pub fn num_atoms(&self) -> usize { self.coords.len() }
    pub fn lattice(&self) -> &Lattice { &self.lattice }

    // FIXME bad idea for stable interface, but good enough for now
    pub fn metadata(&self) -> &[M] { &self.meta }
    pub fn map_metadata_into<M2, F>(self, f: F) -> Structure<M2>
    where F: FnMut(M) -> M2
    {
        let Structure { lattice, coords, meta } = self;
        let meta = meta.into_iter().map(f).collect();
        Structure { lattice, coords, meta }
    }
    // This variant can be useful when using the by-value variant
    // would require the user to clone() first, uneccessarily
    // cloning the entire metadata.
    pub fn map_metadata_to<M2, F>(&self, f: F) -> Structure<M2>
    where F: FnMut(&M) -> M2
    {
        let lattice = self.lattice.clone();
        let coords = self.coords.clone();
        let meta = self.meta.iter().map(f).collect();
        Structure { lattice, coords, meta }
    }

    /// Move all data out by value.
    ///
    /// This operation is not guaranteed to be zero-cost.
    pub fn into_parts(self) -> Parts<M>
    { Parts {
        lattice: self.lattice,
        coords: self.coords,
        metadata: self.meta,
    }}

    pub fn extend<Ms>(&mut self, coords: Coords, meta: Ms)
    where Ms: IntoIterator<Item=M>,
    {
        warn!("Untested code path: 0f3b98e1-203b-4632-af21-db8a6dcb479d");
        let meta = meta.into_iter().collect::<Vec<_>>();
        assert_eq!(meta.len(), coords.len());

        let (my_tag, my_coords) = self.coords.as_mut_vec();
        my_coords.extend(coords.into_tag(my_tag, &self.lattice));
        self.meta.extend(meta);
    }
}

/// Data moved by value out of a structure.
#[derive(Debug, Clone)]
pub struct Parts<M> {
    pub lattice: Lattice,
    pub coords: Coords,
    pub metadata: Vec<M>,
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

impl<M> Permute for Structure<M> {
    fn permuted_by(self, perm: &Perm) -> Self
    {
        let lattice = self.lattice;
        let coords = self.coords.permuted_by(perm);
        let meta = self.meta.permuted_by(perm);
        Structure { lattice, coords, meta }
    }
}

impl<M> Structure<M> {
    pub fn translate_frac(&mut self, v: &[f64; 3])
    { ::util::translate_mut_n3_3(self.fracs_mut(), v); }

    pub fn translate_cart(&mut self, v: &[f64; 3])
    { ::util::translate_mut_n3_3(self.carts_mut(), v); }

    /// Applies a cartesian transformation matrix.
    ///
    /// This will keep fractional positions fixed
    /// by rotating the lattice instead.
    pub fn transform(&mut self, m: &[[f64; 3]; 3])
    {
        self.ensure_only_fracs();
        self.lattice = self.lattice.transformed_by(m);
    }

    /// Take a linear combination of the lattice vectors to produce
    /// an identical structure with a different choice of primitive cell.
    ///
    /// Each row of the input are integer coefficients of a lattice vector
    /// to be used in the output.  The absolute value of the matrix
    /// determinant must be 1.
    ///
    /// Cartesian coordinates are preserved.
    ///
    /// # Panics
    ///
    /// Panics if `abs(det(m)) != 1`.
    pub fn apply_unimodular(&mut self, m: &[[i32; 3]; 3])
    {
        warn!("Untested code path: 1e58907e-ae0b-4af8-8653-f003d88c262d");
        use ::rsp2_array_utils::det;

        // Cartesian - not fractional - coords are preserved under unimodular transforms.
        self.ensure_only_carts();

        assert_eq!(det(m).abs(), 1, "Matrix is not unimodular: {:?}", m);
        self.lattice = self.lattice.linear_combination(&m);
    }

    /// Produce an identical structure (up to precision loss) represented
    /// in terms of a different unit cell.
    ///
    /// The new cell is required to be equivalent to the original cell.
    /// Otherwise, it fails with `NonEquivalentLattice.`
    ///
    /// Numerically speaking, the input cell will not be used exactly;
    /// instead, the new lattice is recomputed as a linear combination of
    /// the original lattice vectors. The expectation is that this
    /// causes less of a disturbance to the positions of sites which
    /// are distant to the origin, in the case where one wishes to reduce
    /// the sites into the new unit cell afterwards.
    pub fn use_equivalent_cell(&mut self, tol: f64, target_lattice: &Lattice) -> Result<()>
    {Ok({
        warn!("Untested code path: 1650857f-42df-47e4-8ff0-cdd9dcb85020");
        let unimodular = &self.lattice * target_lattice.inverse_matrix();
        let unimodular = match ::util::Tol(tol).unfloat_33(unimodular.matrix()) {
            Ok(m) => m,
            Err(_) => bail!(ErrorKind::NonEquivalentLattice(*unimodular.matrix()))
        };
        self.apply_unimodular(&unimodular);
    })}

    /// Reduces all fractional coordinates into [0.0, 1.0).
    pub fn reduce_positions(&mut self)
    {
        self.reduce_positions_fast(); // -> [0.0, 1.0]
        self.reduce_positions_fast(); // -> [0.0, 1.0)
    }

    /// Reduces all fractional coordinates into [0.0, 1.0].
    ///
    /// Yes, that is a doubly inclusive range.
    /// If a coordinate is initially very small and negative,
    /// (say, -1e-20), it will map to 1.0.
    fn reduce_positions_fast(&mut self)
    {
        use ::slice_of_array::prelude::*;
        for x in self.fracs_mut().flat_mut() {
            *x -= x.floor();
        }
    }
}

/// A Structure rendered into a form sendable across threads.
#[derive(Debug, Clone)]
pub struct Sent<M> {
    pub(crate) lattice: SentLattice,
    pub(crate) coords: Coords,
    pub(crate) meta: Vec<M>,
}

impl<M: Send> Structure<M> {
    pub fn send(self) -> Sent<M> {
        let Structure { lattice, coords, meta } = self;
        let lattice = lattice.send();
        Sent { lattice, coords, meta }
    }
}

impl<M: Send> Sent<M> {
    pub fn recv(self) -> Structure<M> {
        let Sent { lattice, coords, meta } = self;
        let lattice = lattice.recv();
        Structure { lattice, coords, meta }
    }
}
