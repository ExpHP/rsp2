use ::{Lattice, CoordsKind, Element};
use ::oper::{Perm, Permute};
use ::oper::{Part, Partition};
use ::oper::part::Unlabeled;
use ::rsp2_array_types::{M33, V3, Unvee};

/// Pairs [`CoordsKind`] together with their [`Lattice`], and metadata.
///
/// Currently the metadata story is pretty weak and hardly resembles
/// any sort of final design.  You should let the `M` type contain all
/// of the information you need, and then when certain functions request
/// a certain `M` type, use `map_metadata_to` to extract that information.
///
/// [`CoordsKind`]: ../struct.CoordsKind.html
/// [`Lattice`]: ../struct.Lattice.html
#[derive(Debug,Clone,PartialEq)]
pub struct Structure<M = ()> {
    pub(crate) lattice: Lattice,
    pub(crate) coords: CoordsKind,
    pub(crate) meta: Vec<M>,
}

/// Type of a Structure with no metadata at all (not even atom types).
pub type CoordStructure = Structure<()>;

/// A Structure whose only metadata is atomic numbers.
pub type ElementStructure = Structure<Element>;

impl CoordStructure {
    /// Create a structure with no metadata; just coordinates.
    pub fn new_coords(lattice: Lattice, coords: CoordsKind) -> Self {
        let meta = vec![(); coords.len()];
        Self::new(lattice, coords, meta)
    }
}

impl<M> Structure<M> {
    pub fn new<Ms>(lattice: Lattice, coords: CoordsKind, meta: Ms) -> Self
    where Ms: IntoIterator<Item=M>,
    {
        let meta: Vec<_> = meta.into_iter().collect();
        assert_eq!(coords.len(), meta.len());
        Self { lattice, coords, meta }
    }

    pub fn num_atoms(&self) -> usize { self.coords.len() }
    pub fn lattice(&self) -> &Lattice { &self.lattice }
}

impl<M> Structure<M> {
    // FIXME bad idea for stable interface, but good enough for now
    pub fn metadata(&self) -> &[M] { &self.meta }

    pub fn try_map_metadata_into<M2, E, F>(self, f: F) -> Result<Structure<M2>, E>
    where F: FnMut(M) -> Result<M2, E>,
    {Ok({
        let Structure { lattice, coords, meta } = self;
        let meta = meta.into_iter().map(f).collect::<Result<_, E>>()?;
        Structure { lattice, coords, meta }
    })}

    pub fn map_metadata_into<M2, F>(self, f: F) -> Structure<M2>
    where F: FnMut(M) -> M2,
    {
        let Structure { lattice, coords, meta } = self;
        let meta = meta.into_iter().map(f).collect();
        Structure { lattice, coords, meta }
    }

    // This variant can be useful when using the by-value variant
    // would require the user to clone() first, needlessly
    // cloning the entire metadata.
    pub fn map_metadata_to<M2, F>(&self, f: F) -> Structure<M2>
    where F: FnMut(&M) -> M2,
    {
        let lattice = self.lattice.clone();
        let coords = self.coords.clone();
        let meta = self.meta.iter().map(f).collect();
        Structure { lattice, coords, meta }
    }

    /// Store new metadata in-place.
    pub fn set_metadata<Ms>(&mut self, meta: Ms)
    where Ms: IntoIterator<Item=M>,
    {
        let old = self.meta.len();
        self.meta.clear();
        self.meta.extend(meta.into_iter());
        assert_eq!(self.meta.len(), old);
    }

    /// Use the given vector as the metadata.
    /// No cost other than the destruction of the old metadata.
    pub fn with_metadata<M2>(self, meta: Vec<M2>) -> Structure<M2>
    {
        assert_eq!(meta.len(), self.meta.len());
        let Structure { lattice, coords, .. } = self;
        Structure { lattice, coords, meta }
    }
}

impl<M> Structure<M> {
    /// Move all data out by value.
    ///
    /// This operation is not guaranteed to be zero-cost.
    pub fn into_parts(self) -> Parts<M>
    { Parts {
        lattice: self.lattice,
        coords: self.coords,
        metadata: self.meta,
    }}

    pub fn extend<Ms>(&mut self, coords: CoordsKind, meta: Ms)
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
    pub coords: CoordsKind,
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
    pub fn to_carts(&self) -> Vec<V3> { self.coords.to_carts(&self.lattice) }
    pub fn to_fracs(&self) -> Vec<V3> { self.coords.to_fracs(&self.lattice) }

    // `ensure_carts` should be called before this to guarantee that the value is `Some(_)`.
    //
    // Yes, having to unwrap the option sucks, but it's unavoidable; this is simply the
    // only way you will ever be able to borrow positions from a borrowed &Structure.
    pub fn as_carts_cached(&self) -> Option<&[V3]> { self.coords.as_carts_opt() }
    pub fn as_fracs_cached(&self) -> Option<&[V3]> { self.coords.as_fracs_opt() }

    pub fn carts_mut(&mut self) -> &mut [V3] {
        self.ensure_only_carts(); // 'only' because user modifications will invalidate fracs
        match self.coords {
            CoordsKind::Fracs(_) => unreachable!(),
            CoordsKind::Carts(ref mut c) => c,
        }
    }

    pub fn fracs_mut(&mut self) -> &mut [V3] {
        self.ensure_only_fracs(); // 'only' because user modifications will invalidate carts
        match self.coords {
            CoordsKind::Fracs(ref mut c) => c,
            CoordsKind::Carts(_) => unreachable!(),
        }
    }

    /// Replace the coordinates in the structure.
    ///
    /// # Panics
    /// Panics if the length does not match.
    pub fn set_coords(&mut self, coords: CoordsKind) {
        assert_eq!(self.coords.len(), coords.len());
        self.coords = coords;
    }
    pub fn set_carts(&mut self, carts: Vec<V3>) { self.set_coords(CoordsKind::Carts(carts)); }
    pub fn set_fracs(&mut self, fracs: Vec<V3>) { self.set_coords(CoordsKind::Fracs(fracs)); }


    /// # Panics
    /// Panics if the length does not match.
    pub fn with_coords(mut self, coords: CoordsKind) -> Self { self.set_coords(coords); self }
    pub fn with_carts(mut self, carts: Vec<V3>) -> Self { self.set_carts(carts); self }
    pub fn with_fracs(mut self, fracs: Vec<V3>) -> Self { self.set_fracs(fracs); self }

    /// Ensures that the cartesian coordinates are cached if they aren't already.
    pub fn ensure_carts(&mut self) { self.ensure_only_carts(); }
    /// Ensures that the fractional coordinates are cached if they aren't already.
    pub fn ensure_fracs(&mut self) { self.ensure_only_fracs(); }

    /// Ensure that carts are available, and that fracs are NOT available.
    ///
    /// Currently equivalent to 'ensure_carts,' but that method may eventually
    /// retain the fracs.
    fn ensure_only_carts(&mut self) {
        let dummy = CoordsKind::Carts(vec![]);
        let coords = ::std::mem::replace(&mut self.coords, dummy);
        self.coords = CoordsKind::Carts(coords.into_carts(&self.lattice));
    }

    /// Ensure that fracs are available, and that carts are NOT available.
    ///
    /// Currently equivalent to 'ensure_fracs,' but that method may eventually
    /// retain the carts.
    fn ensure_only_fracs(&mut self) {
        let dummy = CoordsKind::Carts(vec![]);
        let coords = ::std::mem::replace(&mut self.coords, dummy);
        self.coords = CoordsKind::Fracs(coords.into_fracs(&self.lattice));
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

impl<'iter, M: 'iter> Partition<'iter> for Structure<M> {
    fn into_unlabeled_partitions<L>(self, part: &'iter Part<L>) -> Unlabeled<'iter, Self>
    {
        let Structure { lattice, coords, meta } = self;

        Box::new({
            (coords, meta).into_unlabeled_partitions(part)
                .map(move |(coords, meta)| {
                    let lattice = lattice.clone();
                    Structure { lattice, coords, meta }
                })
        })
    }
}

#[derive(Debug, Fail)]
#[fail(display = "The new lattice is not equivalent to the original. (A B^-1 = {:?})", a_binv)]
pub struct NonEquivalentLattice {
    a_binv: [[f64; 3]; 3],
    backtrace: ::failure::Backtrace,
}

impl<M> Structure<M> {
    pub fn translate_frac(&mut self, v: &V3)
    { ::util::translate_mut_n3_3(self.fracs_mut(), v); }

    pub fn translate_cart(&mut self, v: &V3)
    { ::util::translate_mut_n3_3(self.carts_mut(), v); }

    /// Applies a cartesian transformation matrix.
    ///
    /// This will keep fractional positions fixed
    /// by rotating the lattice instead.
    pub fn transform(&mut self, m: &M33)
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
    pub fn apply_unimodular(&mut self, m: &M33<i32>)
    {
        warn!("Untested code path: 1e58907e-ae0b-4af8-8653-f003d88c262d");

        // Cartesian - not fractional - coords are preserved under unimodular transforms.
        self.ensure_only_carts();

        assert_eq!(m.det().abs(), 1, "Matrix is not unimodular: {:?}", m);
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
    pub fn use_equivalent_cell(&mut self, tol: f64, target_lattice: &Lattice) -> Result<(), NonEquivalentLattice>
    {Ok({
        warn!("Untested code path: 1650857f-42df-47e4-8ff0-cdd9dcb85020");
        let unimodular = &self.lattice * target_lattice.inverse_matrix();
        let unimodular = match ::util::Tol(tol).unfloat_m33(unimodular.matrix()) {
            Ok(m) => m,
            Err(_) => {
                throw!(NonEquivalentLattice {
                    backtrace: ::failure::Backtrace::new(),
                    a_binv: unimodular.matrix().into_array(),
                })
            },
        };
        self.apply_unimodular(&unimodular);
    })}

    /// Reduces all fractional coordinates into [0.0, 1.0).
    pub fn reduce_positions(&mut self)
    {
        self.reduce_positions_fast(); // -> [0.0, 1.0]
        self.reduce_positions_fast(); // -> [0.0, 1.0)
    }

    /// Get the permutation that, when applied to this structure, makes the
    /// coords match another.
    pub fn perm_to_match_coords<M2>(
        &self,
        other: &Structure<M2>,
        tol: f64,
    ) -> Result<Perm, ::algo::find_perm::PositionMatchError>
    {
        // (include abs; arbitrary lattices may have near-zero cancellations)
        // (NOTE: maybe this test on the lattice should use a larger or smaller
        //        tolerance than the perm search. Haven't thought it through)
        assert_eq!(self.num_atoms(), other.num_atoms());
        assert_close!(rel=tol, abs=tol,
            self.lattice().matrix().unvee(),
            other.lattice().matrix().unvee(),
        );
        let meta = vec![(); self.coords.len()];
        ::algo::find_perm::brute_force_with_sort_trick(
            self.lattice(),
            &meta,
            &self.to_fracs(),
            &other.to_fracs(),
            tol,
        )
    }

    /// Reduces all fractional coordinates into [0.0, 1.0].
    ///
    /// Yes, that is a doubly inclusive range.
    /// If a coordinate is initially very small and negative,
    /// (say, -1e-20), it will map to 1.0.
    fn reduce_positions_fast(&mut self)
    {
        use ::slice_of_array::prelude::*;
        for x in self.fracs_mut().unvee().flat_mut() {
            *x -= x.floor();
        }
    }
}

#[cfg(test)]
mod compiletest {
    use super::*;

    fn assert_send<S: Send>() {}
    fn assert_sync<S: Sync>() {}

    #[test]
    fn structure_is_send_and_sync() {
        assert_send::<CoordStructure>();
        assert_sync::<CoordStructure>();
    }
}
