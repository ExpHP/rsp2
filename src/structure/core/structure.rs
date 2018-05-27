use ::{Lattice, CoordsKind, Element};
use ::rsp2_soa_ops::{Perm, Permute};
use ::rsp2_soa_ops::{Part, Partition, Unlabeled};
use ::rsp2_array_types::{M33, V3, Unvee};

/// Pairs [`CoordsKind`] together with their [`Lattice`].
///
/// In contrast to `CoordsKind`, this type is capable of providing both
/// fractional and cartesian representations, as it contains all of the
/// necessary information to convert between them.
///
/// [`CoordsKind`]: ../struct.CoordsKind.html
/// [`Lattice`]: ../struct.Lattice.html
#[derive(Debug,Clone,PartialEq)]
pub struct Coords<V = Vec<V3>> {
    pub(crate) lattice: Lattice,
    pub(crate) coords: CoordsKind<V>,
}

/// [`Coords`] with metadata.
///
/// Currently the metadata story is pretty weak and hardly resembles
/// any sort of final design.  You should let the `M` type contain all
/// of the information you need, and then when certain functions request
/// a certain `M` type, use `map_metadata_to` to extract that information.
/// (FIXME: That was the original intent, at least, but anyone looking at
///  `rsp2_tasks` can clearly see that that is not how it is actually used!)
///
/// All methods of `Coords` are also available on `Structure`.
///
/// [`Coords`]: ../struct.Coords.html
#[derive(Debug,Clone,PartialEq)]
pub struct Structure<M> {
    pub(crate) coords: Coords,
    pub(crate) meta: Vec<M>,
}

/// A Structure whose only metadata is atomic numbers.
pub type ElementStructure = Structure<Element>;

//--------------------------------------------------------------------------------------------------

// NOTE: Not 100% certain about this being a good idea.
//
//       The point of this is to allow methods that take `&Coords` to also
//       accept `&Structure<M>` (through deref coercions). This seems like a
//       potentially bad idea, but I couldn't think of any legitimate scenario
//       in which it would be a potential footgun.
//
//       Historically, rsp2 used `Structure<()>` where it now uses `Coords`,
//       so this conversion works well conceptually in many parts of the code.
//
//       I don't want a method like `as_coords` because it would add to the
//       confusion over the difference between `Coords` and `CoordsKind`, both of
//       which are referred to as simply `coords` in method names.
//       Maybe I should rename one of them to `Positions` or `Sites` or something...
impl<M> ::std::ops::Deref for Structure<M> {
    type Target = Coords;

    #[inline(always)]
    fn deref(&self) -> &Coords { &self.coords }
}

// no DerefMut; I fear that the coordinates of a structure might get accidentally
// permuted independently of the metadata if we did that.

//--------------------------------------------------------------------------------------------------

/// # Construction
impl Coords {
    /// Create a structure with no metadata; just coordinates.
    #[inline]
    pub fn new(lattice: Lattice, coords: CoordsKind) -> Self {
        Self { lattice, coords }
    }
}

/// # Construction
impl<M> Structure<M> {
    pub fn new(lattice: Lattice, coords: CoordsKind, meta: impl IntoIterator<Item=M>) -> Self
    {
        let meta: Vec<_> = meta.into_iter().collect();
        assert_eq!(coords.len(), meta.len());
        let coords = Coords { lattice, coords };
        Self { coords, meta }
    }

    pub fn from_coords(coords: Coords, meta: impl IntoIterator<Item=M>) -> Self
    {
        let meta: Vec<_> = meta.into_iter().collect();
        assert_eq!(coords.num_atoms(), meta.len());
        Self { coords, meta }
    }
}

// compatibility helpers while Structure is being eliminated from the codebase
impl<M> Structure<M> {
    pub fn into_parts(self) -> (Coords, Vec<M>)
    { (self.coords, self.meta) }

    pub fn borrow_coords(&self) -> &Coords
    { &self.coords }
}

//---------------------------------------

/// # Simple data access
impl Coords {
    #[inline] pub fn num_atoms(&self) -> usize { self.coords.len() }
    #[inline] pub fn lattice(&self) -> &Lattice { &self.lattice }
}

// Structure gets these through Deref

//---------------------------------------

/// # Chainable metadata modification
impl Coords {
    /// Add the given vector as metadata. Zero-cost.
    pub fn with_metadata<M>(self, meta: Vec<M>) -> Structure<M>
    { Structure::from_coords(self, meta) }

    /// Add a constant value as metadata.
    pub fn with_uniform_metadata<M: Clone>(self, meta: M) -> Structure<M>
    {
        let meta = vec![meta; self.coords.len()];
        Structure::from_coords(self, meta)
    }
}

/// # Chainable metadata modification
impl<M> Structure<M> {
    /// Drop the metadata.
    #[inline(always)]
    pub fn without_metadata(self) -> Coords { self.coords }

    /// Use the given vector as the metadata.
    /// No cost other than the destruction of the old metadata.
    pub fn with_metadata<M2>(self, meta: Vec<M2>) -> Structure<M2>
    {
        assert_eq!(meta.len(), self.meta.len());
        let Structure { coords, .. } = self;
        Structure { coords, meta }
    }
}

//---------------------------------------

/// # Functions for rescaling the structure.
///
/// These functions preserve fractional position while changing the lattice.
impl Coords {
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

// note: don't use macros, for Go To Definition's sake
/// # Functions for rescaling the structure.
///
/// These functions preserve fractional position while changing the lattice.
impl<M> Structure<M> {
    pub fn set_lattice(&mut self, lattice: &Lattice) { self.coords.set_lattice(lattice) }
    pub fn scale_vecs(&mut self, scale: &[f64; 3]) { self.coords.scale_vecs(scale) }
}

//---------------------------------------

/// # Reading coordinates
impl Coords {
    // NOTE: We can produce `Vec<_>` and `&mut [_]`,
    //       but not `&[_]` because the data might not be present.
    //       (`&mut [_]` works because can insert the data)
    pub fn to_carts(&self) -> Vec<V3> { self.coords.to_carts(&self.lattice) }
    pub fn to_fracs(&self) -> Vec<V3> { self.coords.to_fracs(&self.lattice) }

    // `ensure_carts` should be called before this to guarantee that the value is `Some(_)`.
    //
    // Yes, having to unwrap the option sucks, but it's unavoidable; this is simply the
    // only way you will ever be able to borrow positions from a borrowed &Coords.
    #[inline] pub fn as_carts_cached(&self) -> Option<&[V3]> { self.coords.as_carts_opt() }
    #[inline] pub fn as_fracs_cached(&self) -> Option<&[V3]> { self.coords.as_fracs_opt() }
}

// Structure gets these through Deref

//---------------------------------------

/// # Modifying coordinates
impl Coords {
    pub fn carts_mut(&mut self) -> &mut [V3] {
        self.ensure_only_carts(); // 'only' because user modifications will invalidate fracs
        match &mut self.coords {
            CoordsKind::Fracs(_) => unreachable!(),
            CoordsKind::Carts(c) => c,
        }
    }

    pub fn fracs_mut(&mut self) -> &mut [V3] {
        self.ensure_only_fracs(); // 'only' because user modifications will invalidate carts
        match &mut self.coords {
            CoordsKind::Fracs(c) => c,
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
}

// note: don't use macros, for Go To Definition's sake
/// # Modifying coordinates
impl<M> Structure<M> {
    #[inline(always)] pub fn carts_mut(&mut self) -> &mut [V3] { self.coords.carts_mut() }
    #[inline(always)] pub fn fracs_mut(&mut self) -> &mut [V3] { self.coords.fracs_mut() }
    #[inline(always)] pub fn set_coords(&mut self, coords: CoordsKind) { self.coords.set_coords(coords) }
    #[inline(always)] pub fn set_carts(&mut self, carts: Vec<V3>) { self.coords.set_carts(carts) }
    #[inline(always)] pub fn set_fracs(&mut self, fracs: Vec<V3>) { self.coords.set_fracs(fracs) }
}

//---------------------------------------

/// # Populating caches
impl Coords {
    /// Ensures that the cartesian coordinates are cached if they aren't already.
    ///
    /// It is unspecified whether fractional coordinates remain cached after this call.
    pub fn ensure_carts(&mut self) { self.ensure_only_carts(); }

    /// Ensures that the fractional coordinates are cached if they aren't already.
    ///
    /// It is unspecified whether cartesian coordinates remain cached after this call.
    pub fn ensure_fracs(&mut self) { self.ensure_only_fracs(); }

    /// Ensure that carts are available, and that fracs are NOT available.
    ///
    /// Currently equivalent to `ensure_carts`, but that method may eventually
    /// retain the fracs.
    fn ensure_only_carts(&mut self) {
        let dummy = CoordsKind::Carts(vec![]);
        let coords = ::std::mem::replace(&mut self.coords, dummy);
        self.coords = CoordsKind::Carts(coords.into_carts(&self.lattice));
    }

    /// Ensure that fracs are available, and that carts are NOT available.
    ///
    /// Currently equivalent to `ensure_fracs`, but that method may eventually
    /// retain the carts.
    fn ensure_only_fracs(&mut self) {
        let dummy = CoordsKind::Carts(vec![]);
        let coords = ::std::mem::replace(&mut self.coords, dummy);
        self.coords = CoordsKind::Fracs(coords.into_fracs(&self.lattice));
    }
}

// note: don't use macros, for Go To Definition's sake
/// # Populating caches
impl<M> Structure<M> {
    #[inline(always)] pub fn ensure_carts(&mut self) { self.coords.ensure_carts() }
    #[inline(always)] pub fn ensure_fracs(&mut self) { self.coords.ensure_fracs() }
}

//---------------------------------------

/// # Chainable modification
impl Coords {
    /// # Panics
    /// Panics if the length does not match.
    pub fn with_coords(mut self, coords: CoordsKind) -> Self { self.set_coords(coords); self }
    pub fn with_carts(mut self, carts: Vec<V3>) -> Self { self.set_carts(carts); self }
    pub fn with_fracs(mut self, fracs: Vec<V3>) -> Self { self.set_fracs(fracs); self }
}

/// # Chainable modification
impl<M> Structure<M> {
    /// # Panics
    /// Panics if the length does not match.
    pub fn with_coords(mut self, coords: CoordsKind) -> Self { self.set_coords(coords); self }
    pub fn with_carts(mut self, carts: Vec<V3>) -> Self { self.set_carts(carts); self }
    pub fn with_fracs(mut self, fracs: Vec<V3>) -> Self { self.set_fracs(fracs); self }
}

//---------------------------------------

/// # Spatial operations
impl Coords {
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
}

// note: don't use macros, for Go To Definition's sake
/// # Spatial operations
impl<M> Structure<M> {
    #[inline(always)] pub fn translate_frac(&mut self, v: &V3) { self.coords.translate_frac(v) }
    #[inline(always)] pub fn translate_cart(&mut self, v: &V3) { self.coords.translate_cart(v) }
    #[inline(always)] pub fn transform(&mut self, m: &M33) { self.coords.transform(m) }
}

//---------------------------------------

#[derive(Debug, Fail)]
#[fail(display = "The new lattice is not equivalent to the original. (A B^-1 = {:?})", a_binv)]
pub struct NonEquivalentLattice {
    a_binv: [[f64; 3]; 3],
    backtrace: ::failure::Backtrace,
}

/// # Transformations between equivalent cells
impl Coords {
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
    {
        Ok({
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
        })
    }

    /// Reduces all fractional coordinates into `[0.0, 1.0)`.
    pub fn reduce_positions(&mut self)
    {
        use ::slice_of_array::prelude::*;
        for x in self.fracs_mut().unvee().flat_mut() {
            *x -= x.floor(); // into `[0.0, 1.0]`
            *x -= x.trunc(); // into `[0.0, 1.0)`
        }
    }
}

// note: don't use macros, for Go To Definition's sake
/// # Transformations between equivalent cells
impl<M> Structure<M> {
    #[inline(always)] pub fn apply_unimodular(&mut self, m: &M33<i32>) { self.coords.apply_unimodular(m) }
    #[inline(always)] pub fn use_equivalent_cell(&mut self, tol: f64, target_lattice: &Lattice) -> Result<(), NonEquivalentLattice> { self.coords.use_equivalent_cell(tol, target_lattice) }
    #[inline(always)] pub fn reduce_positions(&mut self) { self.coords.reduce_positions() }
}

//---------------------------------------

/// # Structure matching
impl Coords {
    /// Get the permutation that, when applied to this structure, makes the
    /// coords match another.
    ///
    /// The existing implementation is pretty strict.  I mean, structure matching
    /// is *kind of a tricky problem.*
    pub fn perm_to_match_coords(
        &self,
        other: &Coords,
        tol: f64,
    ) -> Result<Perm, ::algo::find_perm::PositionMatchError>
    {
        // FIXME: incompatible lattices should return Error and not panic
        // NOTE: maybe this test on the lattice should use a larger or smaller
        //        tolerance than the perm search. Haven't thought it through
        // (include abs; arbitrary lattices may have near-zero cancellations)
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
}

//--------------------------------------------------------------------------------------------------
// methods on Structure only

/// # Working with metadata
impl<M> Structure<M> {
    // FIXME bad idea for stable interface, but good enough for now
    pub fn metadata(&self) -> &[M] { &self.meta }

    pub fn try_map_metadata_into<M2, E>(
        self,
        f: impl FnMut(M) -> Result<M2, E>,
    ) -> Result<Structure<M2>, E> {
        Ok({
            let Structure { coords, meta } = self;
            let meta = meta.into_iter().map(f).collect::<Result<_, E>>()?;
            Structure { coords, meta }
        })
    }

    pub fn map_metadata_into<M2>(
        self,
        f: impl FnMut(M) -> M2,
    ) -> Structure<M2> {
        let Structure { coords, meta } = self;
        let meta = meta.into_iter().map(f).collect();
        Structure { coords, meta }
    }

    // This variant can be useful when using the by-value variant
    // would require the user to clone() first, needlessly
    // cloning the old metadata.
    pub fn map_metadata_to<M2>(
        &self,
        f: impl FnMut(&M) -> M2,
    ) -> Structure<M2> {
        let coords = self.coords.clone();
        let meta = self.meta.iter().map(f).collect();
        Structure { coords, meta }
    }

    /// Store new metadata in-place.
    pub fn set_metadata(&mut self, meta: impl IntoIterator<Item=M>)
    {
        let old = self.meta.len();
        self.meta.clear();
        self.meta.extend(meta.into_iter());
        assert_eq!(self.meta.len(), old);
    }

    #[inline(always)] pub fn as_coords(&self) -> &Coords { self }
}

//--------------------------------------------------------------------------------------------------
// trait impls

impl<M> Permute for Structure<M> {
    fn permuted_by(self, perm: &Perm) -> Self
    {
        let coords = self.coords.permuted_by(perm);
        let meta = self.meta.permuted_by(perm);
        Structure { coords, meta }
    }
}

impl Permute for Coords {
    fn permuted_by(self, perm: &Perm) -> Self
    {
        let lattice = self.lattice;
        let coords = self.coords.permuted_by(perm);
        Coords { lattice, coords }
    }
}

impl<'iter, M: 'iter> Partition<'iter> for Structure<M> {
    fn into_unlabeled_partitions<L>(self, part: &'iter Part<L>) -> Unlabeled<'iter, Self>
    {
        let Structure { coords, meta } = self;
        Box::new({
            (coords, meta).into_unlabeled_partitions(part)
                .map(|(coords, meta)| Structure { coords, meta })
        })
    }
}

impl<'iter> Partition<'iter> for Coords {
    fn into_unlabeled_partitions<L>(self, part: &'iter Part<L>) -> Unlabeled<'iter, Self>
    {
        let Coords { lattice, coords } = self;
        Box::new({
            coords.into_unlabeled_partitions(part)
                .map(move |coords| {
                    let lattice = lattice.clone();
                    Coords { lattice, coords }
                })
        })
    }
}

//--------------------------------------------------------------------------------------------------

#[cfg(test)]
mod compiletest {
    use super::*;
    use ::rsp2_array_types::Envee;

    fn assert_send_sync<S: Send + Sync>() {}

    #[test]
    fn structure_is_send_and_sync() {
        assert_send_sync::<Coords>();
        assert_send_sync::<Structure<()>>();
    }

    #[test]
    fn typos_are_bad_mmmkay() {
        // every time we have two functions that do cart vs frac, there
        // is a chance for them to get flipped in a nasty typo that the
        // type system cannot catch.
        // head these concerns off pre-emptively.

        let coords = CoordsKind::Fracs(vec![[0.5, 0.0, 0.0]].envee());
        let lattice = Lattice::diagonal(&[2.0, 1.0, 1.0]);
        let coords = Coords::new(lattice, coords);

        // Structure has all of those dumb trivially-forwarded methods,
        // creating even more room for potential typos.
        // We must test them just as thoroughly.
        let structure = coords.clone().with_uniform_metadata(());

        const ORIG_FRACS: [V3; 1] = [V3([0.5, 0.0, 0.0])];
        const ORIG_CARTS: [V3; 1] = [V3([1.0, 0.0, 0.0])];

        // Make them mutable so we can test functions like `ensure_carts`.
        // (we won't be changing any of the actual coordinates until near
        //  the very end of the test)
        let mut coords = coords;
        let mut structure = structure;

        assert_eq!(coords.to_carts(), ORIG_CARTS.to_vec());
        assert_eq!(coords.to_fracs(), ORIG_FRACS.to_vec());
        assert_eq!(structure.to_carts(), ORIG_CARTS.to_vec());
        assert_eq!(structure.to_fracs(), ORIG_FRACS.to_vec());

        // `*_mut` is the perfect opportunity to indirectly test `ensure_only_*`,
        // because there is simply no way that these methods can preserve the
        // other coordinate system's data and still be correct.
        assert_eq!(coords.carts_mut(), &mut ORIG_CARTS);
        assert_eq!(coords.as_carts_cached(), Some(&ORIG_CARTS[..]));
        assert_eq!(coords.as_fracs_cached(), None);
        assert_eq!(coords.fracs_mut(), &mut ORIG_FRACS);
        assert_eq!(coords.as_fracs_cached(), Some(&ORIG_FRACS[..]));
        assert_eq!(coords.as_carts_cached(), None);

        assert_eq!(structure.carts_mut(), &mut ORIG_CARTS);
        assert_eq!(structure.as_carts_cached(), Some(&ORIG_CARTS[..]));
        assert_eq!(structure.as_fracs_cached(), None);
        assert_eq!(structure.fracs_mut(), &mut ORIG_FRACS);
        assert_eq!(structure.as_fracs_cached(), Some(&ORIG_FRACS[..]));
        assert_eq!(structure.as_carts_cached(), None);

        coords.ensure_carts();
        assert_eq!(coords.as_carts_cached(), Some(&ORIG_CARTS[..]));
        coords.ensure_fracs();
        assert_eq!(coords.as_fracs_cached(), Some(&ORIG_FRACS[..]));

        structure.ensure_carts();
        assert_eq!(structure.as_carts_cached(), Some(&ORIG_CARTS[..]));
        structure.ensure_fracs();
        assert_eq!(structure.as_fracs_cached(), Some(&ORIG_FRACS[..]));

        // these last few will temporarily change the coordinates.
        // We'll put them back each time in the second step.
        coords.set_carts(vec![[2.0, 0.0, 0.0]].envee());
        assert_eq!(coords.to_carts(), vec![[2.0, 0.0, 0.0]].envee());
        assert_eq!(coords.to_fracs(), vec![[1.0, 0.0, 0.0]].envee());
        coords.set_fracs(vec![[0.5, 0.0, 0.0]].envee());
        assert_eq!(coords.to_carts(), vec![[1.0, 0.0, 0.0]].envee());
        assert_eq!(coords.to_fracs(), vec![[0.5, 0.0, 0.0]].envee());

        structure.set_carts(vec![[2.0, 0.0, 0.0]].envee());
        assert_eq!(structure.to_carts(), vec![[2.0, 0.0, 0.0]].envee());
        assert_eq!(structure.to_fracs(), vec![[1.0, 0.0, 0.0]].envee());
        structure.set_fracs(vec![[0.5, 0.0, 0.0]].envee());
        assert_eq!(structure.to_carts(), vec![[1.0, 0.0, 0.0]].envee());
        assert_eq!(structure.to_fracs(), vec![[0.5, 0.0, 0.0]].envee());

        let coords = coords.with_carts(vec![[2.0, 0.0, 0.0]].envee());
        assert_eq!(coords.to_carts(), vec![[2.0, 0.0, 0.0]].envee());
        assert_eq!(coords.to_fracs(), vec![[1.0, 0.0, 0.0]].envee());
        let coords = coords.with_fracs(vec![[0.5, 0.0, 0.0]].envee());
        assert_eq!(coords.to_carts(), vec![[1.0, 0.0, 0.0]].envee());
        assert_eq!(coords.to_fracs(), vec![[0.5, 0.0, 0.0]].envee());

        let structure = structure.with_carts(vec![[2.0, 0.0, 0.0]].envee());
        assert_eq!(structure.to_carts(), vec![[2.0, 0.0, 0.0]].envee());
        assert_eq!(structure.to_fracs(), vec![[1.0, 0.0, 0.0]].envee());
        let structure = structure.with_fracs(vec![[0.5, 0.0, 0.0]].envee());
        assert_eq!(structure.to_carts(), vec![[1.0, 0.0, 0.0]].envee());
        assert_eq!(structure.to_fracs(), vec![[0.5, 0.0, 0.0]].envee());

        let mut coords = coords;
        coords.translate_cart(&V3([1.0, 0.0, 0.0]));
        assert_eq!(coords.to_carts(), vec![[2.0, 0.0, 0.0]].envee());
        assert_eq!(coords.to_fracs(), vec![[1.0, 0.0, 0.0]].envee());
        coords.translate_frac(&V3([-0.5, 0.0, 0.0]));
        assert_eq!(coords.to_carts(), vec![[1.0, 0.0, 0.0]].envee());
        assert_eq!(coords.to_fracs(), vec![[0.5, 0.0, 0.0]].envee());
        let _ = coords;

        let mut structure = structure;
        structure.translate_cart(&V3([1.0, 0.0, 0.0]));
        assert_eq!(structure.to_carts(), vec![[2.0, 0.0, 0.0]].envee());
        assert_eq!(structure.to_fracs(), vec![[1.0, 0.0, 0.0]].envee());
        structure.translate_frac(&V3([-0.5, 0.0, 0.0]));
        assert_eq!(structure.to_carts(), vec![[1.0, 0.0, 0.0]].envee());
        assert_eq!(structure.to_fracs(), vec![[0.5, 0.0, 0.0]].envee());
        let _ = structure;
    }
}
