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

use crate::{Lattice, CoordsKind, Missing};
use rsp2_soa_ops::{Perm, Permute};
use rsp2_soa_ops::{Part, Partition, Unlabeled};
use rsp2_array_types::{M33, V3, Unvee};
pub use failure::Error as Error;

/// Pairs [`CoordsKind`] together with their [`Lattice`].
///
/// In contrast to `CoordsKind`, this type is capable of providing both
/// fractional and cartesian representations, as it contains all of the
/// necessary information to convert between them.
///
/// # Site metadata
///
/// **tl;dr:** You don't need it, and `Coords` (along with every API it touches)
/// is better off for not having it.
///
/// rsp2 used to have a `Structure<M>` type that additionally held a `Vec<M>`
/// for site metadata. In the end, this was finally removed because it was found
/// to cause more problems than it solves.  Many kinds of site data couldn't be
/// shoehorned into the metadata because it led to unnatural representations
/// (such as `Vec<Option<L>>` for something that is `Option<Vec<L>>` in spirit),
/// and for this reason most code just used `Structure<Element>` or `Structure<()>`
/// (which eventually led to the introduction of `Coords`).
///
/// A `Structure<M>` type that holds an `M` (for e.g. `M = (Vec<Element>, Vec<Mass>)`)
/// is *possible,* but also an *obscene* design effort r.e. ownership issues, if you
/// want something suitable for use in low-level computational code. (which `Coords`
/// aims to be).
///
/// A `Structure` that just holds `Vec<Element>` solves nothing yet would demand
/// support from all `Coord`-based APIs.
///
/// While rsp2 has no standard solution to this problem, it is recommended
/// that low-level code (computations, file IO) simply take slices of the
/// data it needs and produce `Vec`s.  The highest-level application code
/// in rsp2 uses HLists of `Rc<[T]>` with just a small number of utility
/// extension traits to meet almost all of its needs.
///
/// (or at least, such was true when I wrote this)
///
/// [`CoordsKind`]: ../struct.CoordsKind.html
/// [`Lattice`]: ../struct.Lattice.html
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Coords {
    pub(crate) lattice: Lattice,
    #[cfg_attr(feature = "serde", serde(flatten))]
    pub(crate) coords: CoordsKind,
}

//--------------------------------------------------------------------------------------------------

/// # Construction
impl Coords {
    /// Create coordinates.
    #[inline]
    pub fn new(lattice: Lattice, coords: CoordsKind) -> Self {
        Self { lattice, coords }
    }
}

//---------------------------------------

/// # Simple data access
impl Coords {
    /// Alias for `num_atoms`.
    #[inline] pub fn len(&self) -> usize { self.coords.len() }
    #[inline] pub fn num_atoms(&self) -> usize { self.coords.len() }
    #[inline] pub fn lattice(&self) -> &Lattice { &self.lattice }
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

//---------------------------------------

/// # Modifying coordinates
impl Coords {
    /// Mutably borrow the cartesian coordinates.
    ///
    /// This forcibly clears any cached fractional data.
    #[inline] // hopefully elide branches in as_xxx_cached()
    pub fn carts_mut(&mut self) -> &mut [V3] {
        self.ensure_only_carts(); // 'only' because user modifications will invalidate fracs
        match &mut self.coords {
            CoordsKind::Fracs(_) => unreachable!(),
            CoordsKind::Carts(c) => c,
        }
    }

    /// Mutably borrow the fractional coordinates.
    ///
    /// This forcibly clears any cached cartesian data.
    #[inline] // hopefully elide branches in as_xxx_cached()
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
    /// Panics if the length does not match. (this restriction is a holdover from earlier
    /// designs, but is now fairly arbitrary, and may be lifted in the future.)
    pub fn set_coords(&mut self, coords: CoordsKind) {
        assert_eq!(self.coords.len(), coords.len());
        self.coords = coords;
    }
    #[inline] // hopefully elide branches in as_xxx_cached()
    pub fn set_carts(&mut self, carts: Vec<V3>) { self.set_coords(CoordsKind::Carts(carts)); }
    #[inline] // hopefully elide branches in as_xxx_cached()
    pub fn set_fracs(&mut self, fracs: Vec<V3>) { self.set_coords(CoordsKind::Fracs(fracs)); }
}

//---------------------------------------

/// # Populating caches
impl Coords {
    /// Ensures that the cartesian coordinates are cached if they aren't already.
    ///
    /// It is unspecified whether fractional coordinates remain cached after this call.
    #[inline] // hopefully elide branches in as_xxx_cached()
    pub fn ensure_carts(&mut self) { self.ensure_only_carts(); }

    /// Ensures that the fractional coordinates are cached if they aren't already.
    ///
    /// It is unspecified whether cartesian coordinates remain cached after this call.
    #[inline] // hopefully elide branches in as_xxx_cached()
    pub fn ensure_fracs(&mut self) { self.ensure_only_fracs(); }

    /// Ensure that carts are available, and that fracs are **not** available.
    ///
    /// Currently equivalent to `ensure_carts`, but that method may eventually
    /// retain the carts.
    #[inline] // hopefully elide branches in as_xxx_cached()
    fn ensure_only_carts(&mut self) {
        let dummy = CoordsKind::Carts(vec![]);
        let coords = std::mem::replace(&mut self.coords, dummy);
        self.coords = CoordsKind::Carts(coords.into_carts(&self.lattice));
    }

    /// Ensure that fracs are available, and that carts are NOT available.
    ///
    /// Currently equivalent to `ensure_fracs`, but that method may eventually
    /// retain the carts.
    #[inline] // hopefully elide branches in as_xxx_cached()
    fn ensure_only_fracs(&mut self) {
        let dummy = CoordsKind::Carts(vec![]);
        let coords = std::mem::replace(&mut self.coords, dummy);
        self.coords = CoordsKind::Fracs(coords.into_fracs(&self.lattice));
    }
}

//---------------------------------------

/// # Chainable modification
impl Coords {
    /// Construct a new `Coords` with the same lattice.  Length need not match.
    #[inline] // hopefully elide branches in as_xxx_cached()
    pub fn with_coords(&self, coords: CoordsKind) -> Self { Coords::new(self.lattice.clone(), coords) }
    #[inline] // hopefully elide branches in as_xxx_cached()
    pub fn with_carts(&self, carts: Vec<V3>) -> Self { self.with_coords(CoordsKind::Carts(carts)) }
    #[inline] // hopefully elide branches in as_xxx_cached()
    pub fn with_fracs(&self, fracs: Vec<V3>) -> Self { self.with_coords(CoordsKind::Fracs(fracs)) }
}

//---------------------------------------

/// # Spatial operations
impl Coords {
    pub fn translate_frac(&mut self, v: &V3)
    { crate::util::translate_mut_n3_3(self.fracs_mut(), v); }

    pub fn translate_cart(&mut self, v: &V3)
    { crate::util::translate_mut_n3_3(self.carts_mut(), v); }

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

//---------------------------------------

#[derive(Debug, Fail)]
#[fail(display = "The new lattice is not equivalent to the original. (A B^-1 = {:?})", a_binv)]
pub struct NonEquivalentLattice {
    a_binv: [[f64; 3]; 3],
    backtrace: failure::Backtrace,
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
            let unimodular = match crate::util::Tol(tol).unfloat_m33(unimodular.matrix()) {
                Ok(m) => m,
                Err(_) => {
                    throw!(NonEquivalentLattice {
                        backtrace: failure::Backtrace::new(),
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
        use slice_of_array::prelude::*;
        for x in self.fracs_mut().unvee().flat_mut() {
            *x -= x.floor(); // into `[0.0, 1.0]`
            *x -= x.trunc(); // into `[0.0, 1.0)`
        }
    }
}

//---------------------------------------

/// # Structure matching
impl Coords {
    /// Get the permutation that, when applied to this structure, makes it
    /// match the target (when reduced under the lattice).
    ///
    /// The existing implementation is pretty strict.  I mean, structure matching
    /// is *kind of a tricky problem.*
    pub fn perm_to_match(
        &self,
        target: &Coords,
        tol: f64,
    ) -> Result<Perm, crate::algo::find_perm::PositionMatchError>
    {
        // FIXME: incompatible lattices should return Error and not panic
        // NOTE: maybe this test on the lattice should use a larger or smaller
        //        tolerance than the perm search. Haven't thought it through
        // (include abs; arbitrary lattices may have near-zero cancellations)
        assert_eq!(self.num_atoms(), target.num_atoms());
        assert_close!(rel=tol, abs=tol,
            self.lattice().matrix().unvee(),
            target.lattice().matrix().unvee(),
        );
        let fake_meta = vec![(); self.coords.len()];
        crate::algo::find_perm::brute_force_with_sort_trick(
            self.lattice(),
            &fake_meta, self.coords.as_ref(),
            &fake_meta, target.coords.as_ref(),
            tol,
        )
    }

    /// Get indices of sites that are only in one structure, but not another.
    ///
    /// The structures must have matching lattices.
    ///
    /// **FIXME:** The current implementation has a known flaw that it will become significantly
    /// slower (jumping up from from `O(n log n)` to `O(n^2)` complexity) if the second structure
    /// is missing anything.  I have an inkling how to fix it but I'm too lazy to benchmark it.
    pub fn set_difference_indices(
        &self,
        other: &Coords,
        tol: f64,
    ) -> Missing
    {
        // FIXME: incompatible lattices should return Error and not panic
        // NOTE: maybe this test on the lattice should use a larger or smaller
        //        tolerance than the perm search. Haven't thought it through
        // (include abs; arbitrary lattices may have near-zero cancellations)
        assert_close!(rel=tol, abs=tol,
            self.lattice().matrix().unvee(),
            other.lattice().matrix().unvee(),
        );
        let fake_meta = vec![(); self.coords.len()];
        crate::algo::find_perm::set_difference_with_sort_trick(
            self.lattice(),
            &fake_meta, self.coords.as_ref(),
            &fake_meta, other.coords.as_ref(),
            tol,
        )
    }
}

//--------------------------------------------------------------------------------------------------
// approximate equality checking
//
// There's a lot of knobs to turn here, and eventually it may deserve a builder-type API.
// For now, we focus on the most common use cases.

impl Coords {
    /// An approximate equality check which:
    ///
    /// * Panics on mismatched length.
    /// * Does not permit uniform translations or rotations.
    /// * Does not permit equivalent cells. (lattice matrices must be approximately equal)
    /// * Does not permit reordering of sites.
    /// * Permits different periodic images of the same site.
    ///
    /// May erroneously fail on heavily skewed unit cells where a significant
    /// portion of the voronoi cell lies outside of the 27 unit cells around
    /// the origin.
    pub fn check_same_cell_and_order(
        &self,
        other: &Coords,
        cart_tol: f64,
    ) -> Result<(), Error> {
        check_close!(rel=cart_tol, abs=cart_tol, self.lattice(), other.lattice())?;
        dumb_validate_equivalent(self.lattice(), &self.to_fracs(), &other.to_fracs(), cart_tol)
    }
}

// Slow, and not even always correct
fn dumb_validate_equivalent(
    lattice: &Lattice,
    frac_a: &[V3],
    frac_b: &[V3],
    tol: f64,
) -> Result<(), Error> {
    assert_eq!(frac_a.len(), frac_b.len());
    for (a, b) in izip!(frac_a, frac_b) {
        check_close!(abs=tol, 0.0, dumb_nearest_distance(lattice, a, b))?;
    }
    Ok(())
}

// Slow, and not even always correct
fn dumb_nearest_distance(
    lattice: &Lattice,
    frac_a: &V3,
    frac_b: &V3,
) -> f64 {
    use crate::CoordsKind;
    let diff = (frac_a - frac_b).map(|x| x - x.round());

    let mut diffs = Vec::with_capacity(27);
    for &a in &[-1., 0., 1.] {
        for &b in &[-1., 0., 1.] {
            for &c in &[-1., 0., 1.] {
                diffs.push(diff + V3([a, b, c]));
            }
        }
    }

    CoordsKind::Fracs(diffs)
        .to_carts(lattice).into_iter()
        .map(|v| v.norm())
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap()
}

//--------------------------------------------------------------------------------------------------
// trait impls

impl Permute for Coords {
    fn permuted_by(self, perm: &Perm) -> Self
    {
        let lattice = self.lattice;
        let coords = self.coords.permuted_by(perm);
        Coords { lattice, coords }
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
    use rsp2_array_types::Envee;

    fn assert_send_sync<S: Send + Sync>() {}

    #[test]
    fn structure_is_send_and_sync() {
        assert_send_sync::<Coords>();
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

        const ORIG_FRACS: [V3; 1] = [V3([0.5, 0.0, 0.0])];
        const ORIG_CARTS: [V3; 1] = [V3([1.0, 0.0, 0.0])];

        // Make them mutable so we can test functions like `ensure_carts`.
        // (we won't be changing any of the actual coordinates until near
        //  the very end of the test)
        let mut coords = coords;

        assert_eq!(coords.to_carts(), ORIG_CARTS.to_vec());
        assert_eq!(coords.to_fracs(), ORIG_FRACS.to_vec());

        // `*_mut` is the perfect opportunity to indirectly test `ensure_only_*`,
        // because there is simply no way that these methods can preserve the
        // other coordinate system's data and still be correct.
        assert_eq!(coords.carts_mut(), &mut ORIG_CARTS);
        assert_eq!(coords.as_carts_cached(), Some(&ORIG_CARTS[..]));
        assert_eq!(coords.as_fracs_cached(), None);
        assert_eq!(coords.fracs_mut(), &mut ORIG_FRACS);
        assert_eq!(coords.as_fracs_cached(), Some(&ORIG_FRACS[..]));
        assert_eq!(coords.as_carts_cached(), None);

        coords.ensure_carts();
        assert_eq!(coords.as_carts_cached(), Some(&ORIG_CARTS[..]));
        coords.ensure_fracs();
        assert_eq!(coords.as_fracs_cached(), Some(&ORIG_FRACS[..]));

        // these last few will temporarily change the coordinates.
        // We'll put them back each time in the second step.
        coords.set_carts(vec![[2.0, 0.0, 0.0]].envee());
        assert_eq!(coords.to_carts(), vec![[2.0, 0.0, 0.0]].envee());
        assert_eq!(coords.to_fracs(), vec![[1.0, 0.0, 0.0]].envee());
        coords.set_fracs(vec![[0.5, 0.0, 0.0]].envee());
        assert_eq!(coords.to_carts(), vec![[1.0, 0.0, 0.0]].envee());
        assert_eq!(coords.to_fracs(), vec![[0.5, 0.0, 0.0]].envee());

        let coords = coords.with_carts(vec![[2.0, 0.0, 0.0]].envee());
        assert_eq!(coords.to_carts(), vec![[2.0, 0.0, 0.0]].envee());
        assert_eq!(coords.to_fracs(), vec![[1.0, 0.0, 0.0]].envee());
        let coords = coords.with_fracs(vec![[0.5, 0.0, 0.0]].envee());
        assert_eq!(coords.to_carts(), vec![[1.0, 0.0, 0.0]].envee());
        assert_eq!(coords.to_fracs(), vec![[0.5, 0.0, 0.0]].envee());

        let mut coords = coords;
        coords.translate_cart(&V3([1.0, 0.0, 0.0]));
        assert_eq!(coords.to_carts(), vec![[2.0, 0.0, 0.0]].envee());
        assert_eq!(coords.to_fracs(), vec![[1.0, 0.0, 0.0]].envee());
        coords.translate_frac(&V3([-0.5, 0.0, 0.0]));
        assert_eq!(coords.to_carts(), vec![[1.0, 0.0, 0.0]].envee());
        assert_eq!(coords.to_fracs(), vec![[0.5, 0.0, 0.0]].envee());
        let _ = coords;
    }

    #[test]
    fn serde() {
        let de: Coords = serde_json::from_str(r#"{
            "lattice": [[2.4192432809928756, 0.0, 0.0],
                        [-1.2096216404964375, 2.095126139274645, 0.0],
                        [0, 0, 10]],
            "carts": [[0, 0, 5],
                      [1.2096216404964375, 0.6983753797582153, 5]]
        }"#).unwrap();
        assert_eq!(de.lattice().matrix().unvee(), [
            [2.4192432809928756, 0.0, 0.0],
            [-1.2096216404964375, 2.095126139274645, 0.0],
            [0.0, 0.0, 10.0],
        ]);
        assert_eq!(de.to_carts().unvee(), vec![
            [0.0, 0.0, 5.0],
            [1.2096216404964375, 0.6983753797582153, 5.0],
        ]);
    }
}
