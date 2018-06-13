/* ********************************************************************** **
**  This file is part of rsp2.                                            **
**                                                                        **
**  rsp2 is free software: you can redistribute it and/or modify it under **
**  the terms of the GNU General Public License as published by the Free  **
**  Software Foundation, either version 3 of the License, or (at your     **
**  option) any later version.                                            **
**                                                                        **
**      http://www.gnu.org/licenses/                                      **
**                                                                        **
** Do note that, while the whole of rsp2 is licensed under the GPL, many  **
** parts of it are licensed under more permissive terms.                  **
** ********************************************************************** */

// The purpose of this module is to wrap `rsp2_lammps_wrap` with code specific to
// the potentials we care about using, and to support our metadata scheme.
//
// This is where we decide e.g. atom type assignments and `pair_coeff` commands.
// (which are decisions that `rsp2_lammps_wrap` has largely chosen to defer)

use ::FailResult;
use ::hlist_aliases::*;
use ::meta::{Mass, Element};
#[allow(unused)] // rustc bug
use ::meta::prelude::*;
#[allow(unused)] // rustc bug
use ::rsp2_soa_ops::{Part, Partition};
use ::rsp2_structure::{Coords, consts, CoordsKind, Lattice};
use ::rsp2_structure::layer::Layers;
use ::rsp2_tasks_config as cfg;
#[allow(unused)] // rustc bug
use ::rsp2_array_types::{V3, Unvee};
#[allow(unused)] // rustc bug
use ::slice_of_array::prelude::*;
use ::std::rc::Rc;
use ::std::collections::BTreeMap;
use ::cmd::trial::TrialDir;

const DEFAULT_KC_Z_CUTOFF: f64 = 14.0; // (Angstrom?)
const DEFAULT_KC_Z_MAX_LAYER_SEP: f64 = 4.5; // Angstrom

const DEFAULT_AIREBO_LJ_SIGMA:    f64 = 3.0; // (cutoff, x3.4 A)
const DEFAULT_AIREBO_LJ_ENABLED:      bool = true;
const DEFAULT_AIREBO_TORSION_ENABLED: bool = false;

/// Metadata type shared by all potentials usable in the main code.
///
/// (all potentials usable in the main code must use a single metadata
///  type by necessity, due to the use of dynamic polymorphism)
pub type CommonMeta = HList2<
    Rc<[Element]>,
    Rc<[Mass]>,
>;

/// Trait alias for a function producing flat potential and gradient,
/// for compatibility with `rsp2_minimize`.
pub trait FlatDiffFn: FnMut(&[f64]) -> FailResult<(f64, Vec<f64>)> {}

impl<F> FlatDiffFn for F where F: FnMut(&[f64]) -> FailResult<(f64, Vec<f64>)> {}

// Type aliases for the trait object types, to work around #23856
pub type DynFlatDiffFn<'a> = FlatDiffFn<Output=FailResult<(f64, Vec<f64>)>> + 'a;

/// This is what gets passed around by very high level code to represent a
/// potential function. Basically:
///
/// * Configuration is read to produce one of these.
///   A trait is used instead of an enum to increase the flexibility of the
///   implementation, and to localize the impact that newly added potentials
///   have on the rest of the codebase.
///
/// * This thing is sendable across threads and should be relatively cheap
///   to clone (basically a bundle of config data). Most code passes
///   it around as a trait object because there is little to be gained from
///   static dispatch in such high level code.
///
/// * When it is time to compute, you build the `DiffFn`, which will have a
///   method for computing potential and gradient. Preferably, you should
///   try to use the same `DiffFn` for numerous computations (subject to
///   some limitations that are... not currently very well-specified.
///   See the `DiffFn` trait)
///
/// The default Metadata type here is the one used by all high-level code.
/// It is a generic type parameter because some implementation may benefit
/// from using their own type (using an adapter to expose a PotentialBuilder
/// implementation with the default metadata type).
pub trait PotentialBuilder<Meta = CommonMeta>
    : Send + Sync
    // 'static just makes the signatures of the trait easier.
    //
    // Supporting PotentialBuilders with borrowed data is as cumbersome as it is possible,
    // infecting function signatures all over the module (see commit 6c36bddbb08d21),
    // and there is little to be gained because PotentialBuilders are rarely created.
    + 'static
    // ...sigh.  Use impl_dyn_clone_detail! to satisfy this.
    + DynCloneDetail<Meta>
{
    // NOTE: when adding methods like "threaded", make sure to override the
    //       default implementations in generic impls!!!
    //       (e.g. Box<PotentialBuilder<M>>, Sum<A, B>, ...)
    /// Sometimes called as a last-minute hint to control threading
    /// within the potential based on the current circumstances.
    /// Use `true` to recommend the creation of threads, and `false` to discourage it.
    ///
    /// The default implementation just ignores the call.
    #[cfg_attr(feature = "nightly", must_use = "this is not an in-place mutation!")]
    fn threaded(&self, _threaded: bool) -> Box<PotentialBuilder<Meta>>
    { self.box_clone() }

    /// Create the DiffFn.  This does potentially expensive initialization, maybe calling out
    /// to external C APIs and etc.
    ///
    /// **NOTE:** This takes a structure for historic (read: dumb) reasons.  Hopefully it can
    /// be removed soon. For now, just make sure to give it something with the same chemical
    /// composition, number of atoms, and lattice type as the structures you'll be computing.
    fn initialize_diff_fn(&self, coords: &Coords, meta: Meta) -> FailResult<Box<DiffFn<Meta>>>;

    /// Convenience method to get a function suitable for `rsp2_minimize`.
    ///
    /// The structure given to this is used to supply the lattice and metadata.
    /// Also, some other data may be precomputed from it.
    ///
    /// Because Boxes don't implement `Fn` traits for technical reasons,
    /// you will likely need to write `&mut *flat_diff_fn` in order to get
    /// a `&mut DynFlatDiffFn`.
    fn initialize_flat_diff_fn(&self, init_coords: &Coords, meta: Meta) -> FailResult<Box<DynFlatDiffFn<'static>>>
    where Meta: Clone + 'static
    {
        let mut diff_fn = self.initialize_diff_fn(init_coords, meta.clone())?;
        let mut coords = init_coords.clone();
        Ok(Box::new(move |pos: &[f64]| Ok({
            coords.set_carts(pos.nest().to_vec());

            let (value, grad) = diff_fn.compute(&coords, meta.clone())?;
            (value, grad.unvee().flat().to_vec())
        })))
    }

    /// Create a DispFn, a non-threadsafe object that can compute many displacements very quickly.
    fn initialize_disp_fn(&self, equilibrium_coords: &Coords, meta: Meta) -> FailResult<Box<DispFn>>
    where Meta: Clone + 'static,
    ;

    /// Convenience adapter for one-off computations.
    ///
    /// The output type is a DiffFn instance that will initialize the true `DiffFn`
    /// from scratch each time it is called. (though presumably, you only intend
    /// to call it once!)
    ///
    /// Usage: `pot.one_off().compute(&structure)`
    ///
    /// This is provided because otherwise, you end up needing to write stuff
    /// like `pot.initialize_diff_fn(&coords, meta.sift()).compute(&coords, meta.sift())`,
    /// which is both confusing and awkward.
    fn one_off(&self) -> OneOff<'_, Meta>
    where Meta: Clone,
    { OneOff(self._as_ref_dyn()) }
}

//-------------------------------------

/// Dumb implementation detail of PotentialBuilder.
///
/// It makes it possible to clone or borrow the PotentialBuilder trait object.
///
/// It needs to be implemented manually for each PotentialBuilder.
/// There's a macro for this.
pub trait DynCloneDetail<Meta> {
    /// "Clone" the trait object.
    fn box_clone(&self) -> Box<PotentialBuilder<Meta>>;

    /// "Borrow" the trait object.
    fn _as_ref_dyn(&self) -> &PotentialBuilder<Meta>;
}

#[macro_export]
macro_rules! impl_dyn_clone_detail {
    (impl[$($bnd:tt)*] DynCloneDetail<$Meta:ty> for $Type:ty { ... }) => {
        impl<$($bnd)*> DynCloneDetail<$Meta> for $Type {
            fn box_clone(&self) -> Box<PotentialBuilder<$Meta>> {
                Box::new(<$Type as Clone>::clone(self))
            }
            fn _as_ref_dyn(&self) -> &PotentialBuilder<$Meta> { self }
        }
    };
}

impl<M> Clone for Box<PotentialBuilder<M>>
where M: 'static, // FIXME why is this necessary? PotentialBuilder doesn't borrow from M...
{
    fn clone(&self) -> Self { self.box_clone() }
}

// necessary for combinators like Sum to be possible
impl<Meta> PotentialBuilder<Meta> for Box<PotentialBuilder<Meta>>
where Meta: Clone + 'static,
{
    fn threaded(&self, threaded: bool) -> Box<PotentialBuilder<Meta>>
    { (**self).threaded(threaded) }

    fn initialize_diff_fn(&self, coords: &Coords, meta: Meta) -> FailResult<Box<DiffFn<Meta>>>
    { (**self).initialize_diff_fn(coords, meta) }

    fn initialize_flat_diff_fn(&self, coords: &Coords, meta: Meta) -> FailResult<Box<DynFlatDiffFn<'static>>>
    { (**self).initialize_flat_diff_fn(coords, meta) }

    fn one_off(&self) -> OneOff<'_, Meta>
    { (**self).one_off() }

    fn initialize_disp_fn(&self, equilibrium_coords: &Coords, meta: Meta) -> FailResult<Box<DispFn>>
    { (**self).initialize_disp_fn(equilibrium_coords, meta) }
}

impl_dyn_clone_detail!{
    impl[Meta: Clone + 'static] DynCloneDetail<Meta> for Box<PotentialBuilder<Meta>> { ... }
}

//-------------------------------------

/// This is `FnMut(&Coords, Meta) -> FailResult<(f64, Vec<V3>)>` with convenience methods.
///
/// A `DiffFn` may contain pre-computed or cached data that is only valid for
/// certain structures.  Most code that handles `DiffFns` must handle them
/// opaquely, so generally speaking, *all code that uses a `DiffFn`* is subject
/// to the *union of the limitations across all of the implementations.*
///
/// (...not a huge deal since all uses and all implementations are local to this crate)
pub trait DiffFn<Meta> {
    /// Compute the value and gradient.
    fn compute(&mut self, coords: &Coords, meta: Meta) -> FailResult<(f64, Vec<V3>)>;

    /// Convenience method to compute the potential.
    fn compute_value(&mut self, coords: &Coords, meta: Meta) -> FailResult<f64>
    { Ok(self.compute(coords, meta)?.0) }

    /// Convenience method to compute the gradient.
    fn compute_grad(&mut self, coords: &Coords, meta: Meta) -> FailResult<Vec<V3>>
    { Ok(self.compute(coords, meta)?.1) }

    /// Convenience method to compute the force.
    fn compute_force(&mut self, coords: &Coords, meta: Meta) -> FailResult<Vec<V3>>
    {
        let mut force = self.compute_grad(coords, meta)?;
        for v in &mut force { *v = -*v; }
        Ok(force)
    }
}

// necessary for combinators like sum
impl<'d, Meta> DiffFn<Meta> for Box<DiffFn<Meta> + 'd> {
    /// Compute the value and gradient.
    fn compute(&mut self, coords: &Coords, meta: Meta) -> FailResult<(f64, Vec<V3>)>
    { (**self).compute(coords, meta) }

    /// Convenience method to compute the potential.
    fn compute_value(&mut self, coords: &Coords, meta: Meta) -> FailResult<f64>
    { (**self).compute_value(coords, meta) }

    /// Convenience method to compute the gradient.
    fn compute_grad(&mut self, coords: &Coords, meta: Meta) -> FailResult<Vec<V3>>
    { (**self).compute_grad(coords, meta) }

    /// Convenience method to compute the force.
    fn compute_force(&mut self, coords: &Coords, meta: Meta) -> FailResult<Vec<V3>>
    { (**self).compute_force(coords, meta) }
}

// FIXME: this is no longer used, but another comment in this file refers to its
//        doc comment for explanation.  The relevant details should be moved there.
//
/// Ensure that carts are available on a Coords, and that fracs are **not** available.
///
/// **TL;DR:**  Its usage in `compute_force_set` makes it feasible to use exact equality
/// when converting forces into a sparse format.
///
/// ---
///
/// The use case is extremely niche.  So niche that I've chosen to just simulate it with
/// this free function helper rather than exposing the (currently private) method on Coords.
///
/// Basically, the purpose is to guarantee that an ensuing call to `carts_mut()` will not
/// change the values returned by `to_fracs()` (which matters to us in the body of
/// `compute_force_set` because the potential might use `to_fracs()`). That is to say:
///
/// ```rust,ignore
/// fn bad(mut coords: Coords) {
///     coords.ensure_carts();
///     let v1 = coords.to_fracs();
///     let _ = coords.carts_mut();
///     let v2 = coords.to_fracs();
///
///     // this is not guaranteed to succeed! (it does at the time of writing, but may not in the
///     // future). Basically, v2 must have been recomputed from cartesian data, but v1 might
///     // have been cached; thus they may differ on the order of machine epsilon if v1
///     // was the source of the cartesian data.
///     assert_eq!(v1, v2);
/// }
///
/// fn good(mut coords: Coords) {
///     ensure_only_carts(&mut coords);  // <--- the only change
///     let v1 = coords.to_fracs();
///     let _ = coords.carts_mut();
///     let v2 = coords.to_fracs();
///
///     // this IS guaranteed to succeed! (for now, and forever)
///     assert_eq!(v1, v2);
/// }
/// ```
#[allow(unused)]
pub fn ensure_only_carts(coords: &mut Coords) {
    // The signature of `carts_mut()` guarantees that it drops all fractional data;
    // it could not be correct otherwise.
    let _ = coords.carts_mut();
}

// default impl for initialize_disp_fn, which cannot be inlined because it needs `Self: Sized`
pub fn default_initialize_disp_fn<Meta>(
    pot: &PotentialBuilder<Meta>,
    equilibrium_coords: &Coords,
    meta: Meta,
) -> FailResult<Box<DispFn>>
where Meta: Clone + 'static,
{ Ok(Box::new(helper::DefaultDispFn::initialize(equilibrium_coords, meta, pot)?)) }

//-------------------------------------

/// This is `FnMut((usize, V3)) -> FailResult<(f64, Vec<V3>)>` with convenience methods.
///
/// A `DispFn` usually has been initialized with the equilibrium structure, and may contain
/// pre-computed equilibrium forces. It may behave differently from a DiffFn in order to take
/// advantage of reliable properties of the structures produced by displacements.
pub trait DispFn {
    /// Compute the change in force caused by the displacement.
    fn compute_dense_force_delta(&mut self, disp: (usize, V3)) -> FailResult<Vec<V3>>;

    /// Compute the change in force caused by the displacement, in a sparse representation.
    ///
    /// This is not required to be identical to `compute_dense_force_delta`; the two methods
    /// may differ in assumptions (e.g. one method might treat the equilibrium structure as
    /// having zero force).
    fn compute_sparse_force_delta(&mut self, disp: (usize, V3)) -> FailResult<BTreeMap<usize, V3>>;
}

/// Implements sparse force sets in terms of dense force sets.
///
/// Assumes `compute_dense_force_delta` produces values that only differ from the
/// original forces in a neighborhood of the displacement. This can be true if
/// the potential...
///
///  * is deterministic,
///  * implements a cutoff radius, and
///  * does not recklessly adjust coordinates
///
/// ...so that with the help of the "ensure_only_carts", even this
/// exact equality check should be effective at sparsifying the data.
///
/// Which is good, because it's tough to define an approximate scale for comparison
/// here, as the forces are the end-result of catastrophic cancellations.
fn sparse_force_from_dense_deterministic(
    disp_fn: &mut DispFn,
    original_force: &[V3],
    disp: (usize, V3),
) -> FailResult<BTreeMap<usize, V3>> {
    let displaced_force = disp_fn.compute_dense_force_delta(disp)?;

    let diffs = {
        zip_eq!(original_force, displaced_force).enumerate()
            .map(|(atom, (old, new))| (atom, new - old))
            .filter(|&(_, v)| v != V3::zero())

        // this one is a closer approximation of phonopy, producing a dense matrix with
        // just the new forces (assuming the old ones are zero)
//                .map(|(atom, (_old, new))| (atom, new))
    };
    Ok(diffs.collect())
}

//-------------------------------------

/// See `PotentialBuilder::one_off` for more information.
pub struct OneOff<'a, M: 'a>(&'a PotentialBuilder<M>);
impl<'a, M: Clone + 'static> DiffFn<M> for OneOff<'a, M> {
    fn compute(&mut self, coords: &Coords, meta: M) -> FailResult<(f64, Vec<V3>)> {
        self.0.initialize_diff_fn(coords, meta.clone())?.compute(coords, meta)
    }
}

//-------------------------------------

/// High-level logic
impl PotentialBuilder {
    pub(crate) fn from_root_config(
        trial_dir: &TrialDir,
        cfg: &cfg::Settings,
    ) -> Box<PotentialBuilder> {
        Self::from_config_parts(trial_dir, &cfg.threading, &cfg.lammps_update_style, &cfg.potential)
    }

    pub(crate) fn from_config_parts(
        trial_dir: &TrialDir,
        threading: &cfg::Threading,
        update_style: &cfg::LammpsUpdateStyle,
        config: &cfg::PotentialKind,
    ) -> Box<PotentialBuilder> {
        match config {
            cfg::PotentialKind::Rebo => {
                let lammps_pot = self::lammps::Airebo::Rebo;
                let pot = self::lammps::Builder::new(trial_dir, threading, update_style, lammps_pot);
                Box::new(pot)
            }
            cfg::PotentialKind::Airebo(cfg) => {
                let lammps_pot = self::lammps::Airebo::from(cfg);
                let pot = self::lammps::Builder::new(trial_dir, threading, update_style, lammps_pot);
                Box::new(pot)
            },
            cfg::PotentialKind::KolmogorovCrespiZ(cfg) => {
                let lammps_pot = self::lammps::KolmogorovCrespiZ::from(cfg);
                let pot = self::lammps::Builder::new(trial_dir, threading, update_style, lammps_pot);
                Box::new(pot)
            },
            cfg::PotentialKind::KolmogorovCrespiZNew(cfg) => {
                let rebo = PotentialBuilder::from_config_parts(trial_dir, threading, update_style, &cfg::PotentialKind::Rebo);
                let kc_z = self::homestyle::KolmogorovCrespiZ(cfg.clone());
                let pot = self::helper::Sum(rebo, kc_z);
                Box::new(pot)
            },
            cfg::PotentialKind::TestZero => Box::new(self::test::Zero),
            cfg::PotentialKind::TestChainify => Box::new(self::test::Chainify),
        }
    }
}

//-------------------------------------

mod helper {
    use super::*;

    /// A sum of two PotentialBuilders or DiffFns.
    #[derive(Debug, Clone)]
    pub struct Sum<A, B>(pub A, pub B);

    impl<M, A, B> PotentialBuilder<M> for Sum<A, B>
    where
        M: Clone + 'static,
        A: Clone + PotentialBuilder<M>,
        B: Clone + PotentialBuilder<M>,
    {
        fn threaded(&self, threaded: bool) -> Box<PotentialBuilder<M>>
        { Box::new(Sum(self.0.threaded(threaded), self.1.threaded(threaded))) }

        fn initialize_diff_fn(&self, coords: &Coords, meta: M) -> FailResult<Box<DiffFn<M>>>
        {
            let a_diff_fn = self.0.initialize_diff_fn(coords, meta.clone())?;
            let b_diff_fn = self.1.initialize_diff_fn(coords, meta.clone())?;
            Ok(Box::new(Sum(a_diff_fn, b_diff_fn)))
        }

        fn initialize_disp_fn(&self, _: &Coords, _: M) -> FailResult<Box<DispFn>>
        { unimplemented!() }
    }

    impl_dyn_clone_detail!{
        impl[
            M: Clone + 'static,
            A: Clone + PotentialBuilder<M>,
            B: Clone + PotentialBuilder<M>,
        ] DynCloneDetail<M> for Sum<A, B> { ... }
    }

    impl<M, A, B> DiffFn<M> for Sum<A, B>
    where
        M: Clone,
        A: DiffFn<M>,
        B: DiffFn<M>,
    {
        fn compute(&mut self, coords: &Coords, meta: M) -> FailResult<(f64, Vec<V3>)> {
            let (a_value, a_grad) = self.0.compute(coords, meta.clone())?;
            let (b_value, b_grad) = self.1.compute(coords, meta.clone())?;
            let value = a_value + b_value;

            let mut grad = a_grad;
            for (out_vec, b_vec) in zip_eq!(&mut grad, b_grad) {
                *out_vec += b_vec;
            }
            Ok((value, grad))
        }
    }

    //--------------------------------

    pub struct DefaultDispFn<Meta> {
        // this is carts instead of Coords for the same reason that `ensure_only_carts` exists;
        // see that function
        equilibrium_carts: Vec<V3>,
        lattice: Lattice,
        equilibrium_force: Vec<V3>,
        meta: Meta,
        diff_fn: Box<DiffFn<Meta>>,
    }

    impl<Meta> DefaultDispFn<Meta>
    where Meta: Clone + 'static,
    {
        pub fn initialize(equilibrium_coords: &Coords, meta: Meta, pot: &PotentialBuilder<Meta>) -> FailResult<Self>
        {Ok({
            let lattice = equilibrium_coords.lattice().clone();
            let equilibrium_carts = equilibrium_coords.to_carts();

            let equilibrium_coords = Coords::new(lattice.clone(), CoordsKind::Carts(equilibrium_carts.clone()));

            let mut diff_fn = pot.initialize_diff_fn(&equilibrium_coords, meta.clone())?;
            let equilibrium_force = diff_fn.compute_force(&equilibrium_coords, meta.clone())?;

            DefaultDispFn { lattice, equilibrium_carts, equilibrium_force, meta, diff_fn }
        })}
    }

    impl<Meta> DispFn for DefaultDispFn<Meta>
    where Meta: Clone,
    {
        fn compute_dense_force_delta(&mut self, disp: (usize, V3)) -> FailResult<Vec<V3>>
        {Ok({
            let mut carts = self.equilibrium_carts.to_vec();
            carts[disp.0] += disp.1;

            let coords = CoordsKind::Carts(carts);
            let coords = Coords::new(self.lattice.clone(), coords);
            self.diff_fn.compute_force(&coords, self.meta.clone())?
        })}

        fn compute_sparse_force_delta(&mut self, disp: (usize, V3)) -> FailResult<BTreeMap<usize, V3>>
        {
            let orig_force = self.equilibrium_force.clone();
            sparse_force_from_dense_deterministic(self, &orig_force, disp)
        }
    }
}

//-------------------------------------

/// PotentialBuilder implementations for potentials implemented within rsp2.
mod homestyle {
    use super::*;
    use ::math::bonds::{FracBonds, CartBond};

    /// Rust implementation of Kolmogorov-Crespi Z.
    ///
    /// NOTE: This has the limitation that the set of pairs within interaction range
    ///       must not change after the construction of the DiffFn.
    #[derive(Debug, Clone)]
    pub struct KolmogorovCrespiZ(pub(super) cfg::PotentialKolmogorovCrespiZNew);

    impl PotentialBuilder<CommonMeta> for KolmogorovCrespiZ {
        fn initialize_diff_fn(&self, coords: &Coords, meta: CommonMeta) -> FailResult<Box<DiffFn<CommonMeta>>>
        {
            fn fn_body(me: &KolmogorovCrespiZ, coords: &Coords, _: CommonMeta) -> FailResult<Box<DiffFn<CommonMeta>>> {
                let cfg::PotentialKolmogorovCrespiZNew { cutoff_begin } = me.0;
                let mut params = ::math::crespi::Params::default();
                if let Some(cutoff_begin) = cutoff_begin {
                    params.cutoff_begin = cutoff_begin;
                }
                let bonds = FracBonds::from_brute_force_very_dumb(&coords, params.cutoff_end() * 1.001)?;
                Ok(Box::new(Diff { params, bonds }))
            }

            struct Diff {
                params: ::math::crespi::Params,
                bonds: FracBonds,
            }

            impl DiffFn<CommonMeta> for Diff {
                fn compute(&mut self, coords: &Coords, _: CommonMeta) -> FailResult<(f64, Vec<V3>)> {
                    let bonds = self.bonds.to_cart_bonds(coords);

                    let mut value = 0.0;
                    let mut grad = vec![V3::zero(); coords.num_atoms()];
                    for CartBond { from: _, to, cart_vector } in &bonds {
                        let ::math::crespi::Output {
                            value: part_value,
                            grad_rij: part_grad, ..
                        } = self.params.crespi_z(cart_vector);

                        value += part_value;
                        grad[to] += part_grad;
                    }
                    trace!("KCZ: {}", value);
                    Ok((value, grad))
                }
            }

            fn_body(self, coords, meta)
        }

        fn initialize_disp_fn(&self, coords: &Coords, meta: CommonMeta) -> FailResult<Box<DispFn>>
        { default_initialize_disp_fn(self, coords, meta) }
    }

    impl_dyn_clone_detail!{
        impl[] DynCloneDetail<CommonMeta> for KolmogorovCrespiZ { ... }
    }
}

//-------------------------------------

/// All usage of the public API presented by `rsp2_lammps_wrap` is encapsulated here.
mod lammps {
    use super::*;

    use ::rsp2_lammps_wrap::{InitInfo, AtomType, PairCommand};
    use ::rsp2_lammps_wrap::Builder as InnerBuilder;
    use ::rsp2_lammps_wrap::Potential as LammpsPotential;
    use ::rsp2_lammps_wrap::UpdateStyle;

    /// A bundle of everything we need to initialize a Lammps API object.
    ///
    /// It is nothing more than a bundle of configuration, and can be freely
    /// sent across threads.
    #[derive(Debug, Clone)]
    pub(crate) struct Builder<P> {
        inner: InnerBuilder,
        pub potential: P,
    }

    fn assert_send_sync<S: Send + Sync>() {}

    #[allow(unused)] // compile-time test
    fn assert_lammps_builder_send_sync() {
        assert_send_sync::<Builder<()>>();
    }

    impl<P: Clone> Builder<P>
    {
        pub(crate) fn new(
            trial_dir: &TrialDir,
            threading: &cfg::Threading,
            update_style: &cfg::LammpsUpdateStyle,
            potential: P,
        ) -> Self {
            let mut inner = InnerBuilder::new();
            inner.append_log(trial_dir.join("lammps.log"));
            inner.threaded(*threading == cfg::Threading::Lammps);
            inner.update_style(match *update_style {
                cfg::LammpsUpdateStyle::Safe => UpdateStyle::safe(),
                cfg::LammpsUpdateStyle::Run{ n, pre, post } => {
                    warn_once!("lammps-update-style: run' is only for debugging purposes");
                    UpdateStyle { n, pre, post }
                },
                cfg::LammpsUpdateStyle::Fast => {
                    warn_once!("'lammps-update-style: fast' is experimental");
                    UpdateStyle::fast()
                },
            });
            // XXX
//            inner.data_trace_dir(Some(trial_dir.join("lammps-data-trace")));

            Builder { inner, potential }
        }

        pub(crate) fn threaded(&self, threaded: bool) -> Self {
            let mut me = self.clone();
            me.inner.threaded(threaded);
            me
        }
    }

    impl<M: Clone + 'static, P: LammpsPotential<Meta=M> + Clone + Send + Sync + 'static> Builder<P>
    {
        /// Initialize Lammps to make a DiffFn.
        ///
        /// This keeps the Lammps instance between calls to save time.
        ///
        /// Some data may be pre-allocated or precomputed based on the input structure,
        /// so the resulting DiffFn may not support arbitrary structures as input.
        pub(crate) fn lammps_diff_fn(&self, coords: &Coords, meta: M) -> FailResult<Box<DiffFn<M>>>
        {
            // a DiffFn 'lambda' whose type will be erased
            struct MyDiffFn<Mm: Clone>(::rsp2_lammps_wrap::Lammps<Box<LammpsPotential<Meta=Mm>>>);
            impl<Mm: Clone> DiffFn<Mm> for MyDiffFn<Mm> {
                fn compute(&mut self, coords: &Coords, meta: Mm) -> FailResult<(f64, Vec<V3>)> {
                    let lmp = &mut self.0;

                    lmp.set_structure(coords.clone(), meta)?;
                    let value = lmp.compute_value()?;
                    let grad = lmp.compute_grad()?;
                    Ok((value, grad))
                }
            }

            let lammps_pot = Box::new(self.potential.clone()) as Box<LammpsPotential<Meta=P::Meta>>;
            let lmp = self.inner.build(lammps_pot, coords.clone(), meta)?;
            Ok(Box::new(MyDiffFn::<M>(lmp)) as Box<_>)
        }
    }

    impl<M: Clone + 'static, P: Clone + LammpsPotential<Meta=M> + Send + Sync + 'static> PotentialBuilder<M> for Builder<P>
    {
        fn threaded(&self, threaded: bool) -> Box<PotentialBuilder<M>>
        { Box::new(<Builder<_>>::threaded(self, threaded)) }

        fn initialize_diff_fn(&self, coords: &Coords, meta: M) -> FailResult<Box<DiffFn<M>>>
        { self.lammps_diff_fn(coords, meta) }

        fn initialize_disp_fn(&self, coords: &Coords, meta: P::Meta) -> FailResult<Box<DispFn>>
        {
            // FIXME optimize
            default_initialize_disp_fn(self, coords, meta)
        }
    }

    impl_dyn_clone_detail!{
        impl[M: Clone + 'static, P: Clone + LammpsPotential<Meta=M> + Send + Sync + 'static]
        DynCloneDetail<M> for Builder<P> { ... }
    }

    pub use self::airebo::Airebo;
    mod airebo {
        use super::*;

        /// Uses `pair_style airebo` or `pair_style rebo`.
        #[derive(Debug, Clone)]
        pub enum Airebo {
            Airebo {
                lj_sigma: f64,
                lj_enabled: bool,
                torsion_enabled: bool,
            },
            /// Uses `pair_style rebo`.
            ///
            /// This is NOT equivalent to `pair_style airebo 0 0 0`, because it
            /// additionally sets one of the C-C interaction parameters to zero.
            Rebo,
        }

        impl<'a> From<&'a cfg::PotentialAirebo> for Airebo {
            fn from(cfg: &'a cfg::PotentialAirebo) -> Self {
                let cfg::PotentialAirebo { lj_sigma, lj_enabled, torsion_enabled } = *cfg;
                Airebo::Airebo {
                    lj_sigma: lj_sigma.unwrap_or(DEFAULT_AIREBO_LJ_SIGMA),
                    lj_enabled: lj_enabled.unwrap_or(DEFAULT_AIREBO_LJ_ENABLED),
                    torsion_enabled: torsion_enabled.unwrap_or(DEFAULT_AIREBO_TORSION_ENABLED),
                }
            }
        }

        impl LammpsPotential for Airebo {
            type Meta = CommonMeta;

            fn atom_types(&self, _: &Coords, meta: &CommonMeta) -> Vec<AtomType>
            {
                let elements: Rc<[Element]> = meta.pick();
                elements.iter().map(|elem| match elem.symbol() {
                    "H" => AtomType::new(1),
                    "C" => AtomType::new(2),
                    sym => panic!("Unexpected element in Airebo: {}", sym),
                }).collect()
            }

            fn init_info(&self, _: &Coords, meta: &CommonMeta) -> InitInfo {
                let elements: Rc<[Element]> = meta.pick();
                let masses: Rc<[Mass]> = meta.pick();

                let only_unique_mass = |target_elem| {
                    let iter = {
                        zip_eq!(&elements[..], &masses[..])
                            .filter(|&(&elem, _)| elem == target_elem)
                            .map(|(_, &mass)| mass)
                    };

                    match only_unique_value(iter) {
                        OnlyUniqueResult::Ok(Mass(mass)) => mass,
                        OnlyUniqueResult::NoValues => {
                            // It is unlikely that this value will ever come into play (atoms of
                            // this element would need to be introduced afterwards), and if their
                            // mass does not match the value supplied here it will fail.
                            //
                            // For now I think I'd rather have it universally fail in such cases,
                            // forcing us to update this code and add a way to read the values from
                            // config here if this situation arises.
                            f64::from_bits(0xdeadbeef) // absurd value
                        },
                        OnlyUniqueResult::Conflict(_, _) => {
                            panic!("different masses for same element not yet supported by Airebo");
                        },
                    }
                };

                let masses = vec![
                    only_unique_mass(consts::HYDROGEN),
                    only_unique_mass(consts::CARBON),
                ];
                let pair_commands = match *self {
                    Airebo::Airebo { lj_sigma, lj_enabled, torsion_enabled } => vec![
                        PairCommand::pair_style("airebo/omp")
                            .arg(lj_sigma)
                            .arg(boole(lj_enabled))
                            .arg(boole(torsion_enabled))
                            ,
                        PairCommand::pair_coeff(.., ..).args(&["CH.airebo", "H", "C"]),
                    ],
                    Airebo::Rebo => vec![
                        PairCommand::pair_style("rebo/omp"),
                        PairCommand::pair_coeff(.., ..).args(&["CH.airebo", "H", "C"]),
                    ],
                };
                InitInfo { masses, pair_commands }
            }
        }

        fn boole(b: bool) -> u32 { b as _ }
    }

    pub use self::kc_z::KolmogorovCrespiZ;
    mod kc_z {
        use super::*;

        /// Uses `pair_style kolmogorov/crespi/z`.
        #[derive(Debug, Clone, Default)]
        pub struct KolmogorovCrespiZ {
            cutoff: f64,
            // max distance between interacting layers.
            // used to identify vacuum separation.
            max_layer_sep: f64,
        }

        #[derive(Debug, Clone, Copy)]
        enum GapKind { Interacting, Vacuum }

        impl<'a> From<&'a cfg::PotentialKolmogorovCrespiZ> for KolmogorovCrespiZ {
            fn from(cfg: &'a cfg::PotentialKolmogorovCrespiZ) -> Self {
                let cfg::PotentialKolmogorovCrespiZ { cutoff, max_layer_sep } = *cfg;
                KolmogorovCrespiZ {
                    cutoff: cutoff.unwrap_or(DEFAULT_KC_Z_CUTOFF),
                    max_layer_sep: max_layer_sep.unwrap_or(DEFAULT_KC_Z_MAX_LAYER_SEP),
                }
            }
        }

        impl KolmogorovCrespiZ {
            // NOTE: This ends up getting called stupidly often, but I don't think
            //       it is expensive enough to be a real cause for concern.
            fn find_layers(&self, structure: &Coords) -> Layers
            {
                ::rsp2_structure::layer::find_layers(&structure, V3([0, 0, 1]), 0.25)
                    .unwrap_or_else(|e| {
                        panic!("Failure to determine layers when using kolmogorov/crespi/z: {}", e);
                    })
            }

            fn classify_gap(&self, gap: f64) -> GapKind
            {
                if gap <= self.max_layer_sep { GapKind::Interacting }
                else { GapKind::Vacuum }
            }
        }

        impl LammpsPotential for KolmogorovCrespiZ {
            type Meta = CommonMeta;

            // FIXME: We should use layers from metadata, now that they can be
            //        carried around more naturally, but:
            //
            //        * We need a way to verify that the layers were taken
            //          specifically along the Z axis.
            //        * We need a way to identify the vacuum separation gap.
            //
            fn atom_types(&self, coords: &Coords, _: &CommonMeta) -> Vec<AtomType>
            {
                self.find_layers(coords)
                    .by_atom().into_iter()
                    .map(|x| AtomType::from_index(x))
                    .collect()
            }

            fn init_info(&self, coords: &Coords, meta: &CommonMeta) -> InitInfo
            {
                let elements: Rc<[Element]> = meta.pick();
                let masses: Rc<[Mass]> = meta.pick();

                let layers = match self.find_layers(coords).per_unit_cell() {
                    None => panic!("kolmogorov/crespi/z is only supported for layered materials"),
                    Some(layers) => layers,
                };

                {
                    // stupid limitation of current design.  To fix it we really just need
                    // to add an extra atom type.
                    assert!(
                        elements.iter().cloned().all(|x| x == consts::CARBON),
                        "KCZ does not yet support the inclusion of non-carbon elements"
                    );
                }

                let masses: Vec<f64> = {
                    let layer_part = Part::from_ord_keys(layers.by_atom());
                    let m = masses.to_vec()
                        .into_unlabeled_partitions(&layer_part)
                        .map(|layer_masses| match only_unique_value(layer_masses) {
                            OnlyUniqueResult::Ok(Mass(x)) => x,
                            OnlyUniqueResult::Conflict(_, _) => {
                                panic!("KCZ does not support multiple masses within a layer");
                            }
                            OnlyUniqueResult::NoValues => unreachable!(),
                        })
                        .collect();
                    m
                };

                let interacting_pairs: Vec<_> = {
                    let gaps: Vec<_> = layers.gaps.iter().map(|&x| self.classify_gap(x)).collect();

                    determine_pair_coeff_pairs(&gaps)
                        .unwrap_or_else(|e| panic!("{}", e))
                };

                // Need to specify kc/z once for each interacting pair of layers. e.g.:
                //
                //     pair_style hybrid/overlay rebo kolmogorov/crespi/z 20 &
                //                 kolmogorov/crespi/z 20 kolmogorov/crespi/z 20
                let mut pair_commands = vec![];
                pair_commands.push({
                    let mut cmd = PairCommand::pair_style("hybrid/overlay").arg("rebo");
                    for _ in &interacting_pairs {
                        cmd = cmd.arg("kolmogorov/crespi/z").arg(self.cutoff);
                    }
                    cmd
                });

                pair_commands.push({
                    PairCommand::pair_coeff(.., ..)
                        .args(&["rebo", "CH.airebo"])
                        .args(&vec!["C"; layers.len()])
                });

                // If there is a single kcz term, then the pair_coeff command is:
                //
                //       pair_coeff 1 2 kolmogorov/crespi/z CC.KC C C
                //
                // If there are more than one, then the pair_coeff commands need integer labels.
                // Also, you still need elements for all atom types; these can be set to NULL.
                //
                //                                          v--this
                //       pair_coeff 1 2 kolmogorov/crespi/z 1 CC.KC C C NULL
                //       pair_coeff 1 3 kolmogorov/crespi/z 2 CC.KC C NULL C
                //       pair_coeff 2 3 kolmogorov/crespi/z 3 CC.KC NULL C C
                //
                // Although I think I prefer not using NULL, because it appears to me that this
                // internally sets the type to '-1' and this is NEVER CHECKED, IF YOU USE IT
                // THEN IT JUST SEGFAULTS, WTF, WHY ON EARTH WOULD YOU ASSIGN A SPECIAL VALUE
                // FOR SOME CASE AND THEN NEVER ACTUALLY CHECK FOR IT!? WHY!? WHY!? WHYYYY!?!
                //                                       - ML
                for (cmd_n, &(i, j)) in interacting_pairs.iter().enumerate() {
                    let cmd = PairCommand::pair_coeff(i, j);
                    let cmd = match interacting_pairs.len() {
                        1 => cmd.arg("kolmogorov/crespi/z"),
                        _ => cmd.arg("kolmogorov/crespi/z").arg(cmd_n + 1),
                    };
                    let cmd = cmd.arg("CC.KC");
                    let cmd = cmd.args(vec!["C"; layers.len()]); // ...let's not use NULL.
                    pair_commands.push(cmd);
                };

                InitInfo { masses, pair_commands }
            }
        }

        // the 'pair_coeff' in the name is meant to emphasize that this
        // takes care of issues like ensuring I < J.
        fn determine_pair_coeff_pairs(gaps: &[GapKind]) -> FailResult<Vec<(AtomType, AtomType)>>
        {Ok({
            let pairs: Vec<(AtomType, AtomType)> = match gaps.len() {
                0 => {
                    // NOTE: This code path is probably unreachable due to these
                    //       cases already being handled earlier.
                    warn_once!(
                        "kolmogorov/crespi/z was used on a structure with no distinct layers. \
                        It will have no effect!"
                    );
                    vec![]
                },
                1 => match gaps[0] {
                    // A single layer, vacuum-separated from its images.
                    GapKind::Vacuum => vec![],
                    // A single layer, close enough to its images to interact with them.
                    GapKind::Interacting => {
                        bail!("kolmogorov/crespi/z cannot be used (or rather, has not yet \
                            been tested) on a layer that interacts with images of itself. \
                            Please take a supercell.")
                    },
                },
                _ => {
                    gaps.iter()
                        .enumerate()
                        .filter_map(|(i, &gap)| match gap {
                            GapKind::Interacting => {
                                let this = AtomType::from_index(i);
                                let next = AtomType::from_index((i + 1) % gaps.len());
                                Some((this, next))
                            },
                            GapKind::Vacuum => None,
                        })
                        .collect()
                },
            };

            // Canonicalize the entries, because:
            // - 'pair_coeff I J' is ignored by lammps if I > J
            let mut pairs: Vec<_> = pairs.into_iter()
                .map(|(a, b)| (a.min(b), a.max(b)))
                .collect();

            // - there might be duplicates (e.g. for two layers)
            pairs.sort();
            pairs.dedup();
            pairs
        })}

        #[test]
        fn test_determine_pair_coeff_pairs() {
            use self::GapKind::Vacuum as V;
            use self::GapKind::Interacting as I;
            let f_ok = |gaps| determine_pair_coeff_pairs(gaps).unwrap();
            let f_err = |gaps| determine_pair_coeff_pairs(gaps).unwrap_err();
            let pair = |i, j| (AtomType::new(i), AtomType::new(j));

            assert_eq!(f_ok(&[]), vec![]);
            assert_eq!(f_ok(&[V]), vec![]);
            let _ = f_err(&[I]);
            assert_eq!(f_ok(&[V, V]), vec![]);
            assert_eq!(f_ok(&[I, V]), vec![pair(1, 2)]);
            assert_eq!(f_ok(&[V, I]), vec![pair(1, 2)]);
            assert_eq!(f_ok(&[I, I]), vec![pair(1, 2)]);
            assert_eq!(f_ok(&[I, I, I]), vec![pair(1, 2), pair(1, 3), pair(2, 3)]);
            assert_eq!(f_ok(&[I, I, V]), vec![pair(1, 2), pair(2, 3)]);
            assert_eq!(f_ok(&[I, I, I, I]), vec![pair(1, 2), pair(1, 4), pair(2, 3), pair(3, 4)]);
        }
    }

    // util for compressing atom type properties
    enum OnlyUniqueResult<T> {
        Ok(T),
        Conflict(T, T),
        NoValues,
    }
    fn only_unique_value<T: PartialEq>(iter: impl IntoIterator<Item=T>) -> OnlyUniqueResult<T> {
        let mut iter = iter.into_iter();
        if let Some(first) = iter.next() {
            for x in iter {
                if x != first {
                    return OnlyUniqueResult::Conflict(first, x);
                }
            }
            OnlyUniqueResult::Ok(first)
        } else {
            OnlyUniqueResult::NoValues
        }
    }
}

//-------------------------------------

/// Dummy potentials for testing purposes
pub mod test {
    use super::*;
    use ::rsp2_structure::{Coords, CoordsKind};

    /// The test Potential `V = 0`.
    #[derive(Debug, Clone)]
    pub struct Zero;

    impl<Meta: Clone + 'static> PotentialBuilder<Meta> for Zero {
        fn initialize_diff_fn<'a>(&self, _: &Coords, _: Meta) -> FailResult<Box<DiffFn<Meta>>>
        {
            struct Diff;
            impl<M> DiffFn<M> for Diff {
                fn compute(&mut self, coords: &Coords, _: M) -> FailResult<(f64, Vec<V3>)> {
                    Ok((0.0, vec![V3([0.0; 3]); coords.num_atoms()]))
                }
            }
            Ok(Box::new(Diff) as Box<_>)
        }

        fn initialize_disp_fn(&self, coords: &Coords, meta: Meta) -> FailResult<Box<DispFn>>
        { default_initialize_disp_fn(self, coords, meta) }
    }

    impl_dyn_clone_detail!{
        impl[M: Clone + 'static] DynCloneDetail<M> for Zero { ... }
    }

    // ---------------

    /// A test DiffFn that moves atoms to fixed positions.
    #[derive(Debug, Clone)]
    pub struct ConvergeTowards {
        target: Coords,
    }

    impl ConvergeTowards {
        pub fn new(coords: Coords) -> Self
        { ConvergeTowards { target: coords.clone() } }
    }

    /// ConvergeTowards can also serve as its own PotentialBuilder.
    impl<Meta: Clone + 'static> PotentialBuilder<Meta> for ConvergeTowards {
        fn initialize_diff_fn(&self, _: &Coords, _: Meta) -> FailResult<Box<DiffFn<Meta>>>
        { Ok(Box::new(self.clone()) as Box<_>) }

        fn initialize_disp_fn(&self, coords: &Coords, meta: Meta) -> FailResult<Box<DispFn>>
        { default_initialize_disp_fn(self, coords, meta) }
    }

    impl_dyn_clone_detail!{
        impl[Meta: Clone + 'static] DynCloneDetail<Meta> for ConvergeTowards { ... }
    }

    impl<M> DiffFn<M> for ConvergeTowards {
        fn compute(&mut self, coords: &Coords, meta: M) -> FailResult<(f64, Vec<V3>)> {
            (&*self).compute(coords, meta)
        }
    }

    // ConvergeTowards does not get mutated
    impl<'a, M> DiffFn<M> for &'a ConvergeTowards {
        fn compute(&mut self, input_coords: &Coords, _: M) -> FailResult<(f64, Vec<V3>)> {
            assert_eq!(input_coords.num_atoms(), self.target.num_atoms());
            assert_close!(abs=1e-8, input_coords.lattice(), self.target.lattice());

            // Each position in `structure` experiences a force generated only by the
            // corresponding position in `target`.

            // In fractional coords, the potential is:
            //
            //    Sum_a  Product_k { 1.0 - cos[(x[a,k] - xtarg[a,k]) 2 pi] }
            //    (atom)  (axis)
            //
            // This is a periodic function with a minimum at each image of the target point,
            // and derivatives that are continuous everywhere.
            use ::std::f64::consts::PI;

            let cur_fracs = input_coords.to_fracs();
            let target_fracs = self.target.to_fracs();
            let args_by_coord = {
                zip_eq!(&cur_fracs, target_fracs)
                    .map(|(c, t)| V3::from_fn(|k| (c[k] - t[k]) * 2.0 * PI))
                    .collect::<Vec<_>>()
            };
            let d_arg_d_coord = 2.0 * PI;

            let parts_by_coord = {
                args_by_coord.iter()
                    .map(|&v| v.map(|arg| 2.0 - f64::cos(arg)))
                    .collect::<Vec<_>>()
            };
            let derivs_by_coord = {
                args_by_coord.iter()
                    .map(|&v| v.map(|arg| d_arg_d_coord * f64::sin(arg)))
                    .collect::<Vec<_>>()
            };

            let value = {
                parts_by_coord.iter().map(|v| v.iter().product::<f64>()).sum()
            };
            let frac_grad = {
                zip_eq!(&parts_by_coord, &derivs_by_coord)
                    .map(|(parts, derivs)| { // by atom
                        V3::from_fn(|k0| {
                            (0..3).map(|k| {
                                if k == k0 { derivs[k] }
                                else { parts[k] }
                            }).product()
                        })
                    })
            };

            // Partial derivatives transform like reciprocal coords.
            let recip = input_coords.lattice().reciprocal();
            let cart_grad = frac_grad.map(|v| v * &recip).collect::<Vec<_>>();
            Ok((value, cart_grad))
        }
    }

    // ---------------

    /// A test Potential that creates a chain along the first lattice vector.
    #[derive(Debug, Clone)]
    pub struct Chainify;

    impl<Meta: Clone + 'static> PotentialBuilder<Meta> for Chainify {
        fn initialize_diff_fn(&self, initial_coords: &Coords, _: Meta) -> FailResult<Box<DiffFn<Meta>>>
        {
            let na = initial_coords.num_atoms();
            let target = Coords::new(
                initial_coords.lattice().clone(),
                CoordsKind::Fracs({
                    (0..na)
                        .map(|i| V3([i as f64 / na as f64, 0.5, 0.5]))
                        .collect()
                }),
            );
            Ok(Box::new(ConvergeTowards::new(target)) as Box<_>)
        }

        fn initialize_disp_fn(&self, coords: &Coords, meta: Meta) -> FailResult<Box<DispFn>>
        { default_initialize_disp_fn(self, coords, meta) }
    }

    impl_dyn_clone_detail!{
        impl[Meta: Clone + 'static] DynCloneDetail<Meta> for Chainify { ... }
    }

    // ---------------

    #[cfg(test)]
    #[deny(unused)]
    mod tests {
        use super::*;
        use ::FailOk;
        use rsp2_structure::{Lattice, CoordsKind};
        use ::rsp2_array_types::Envee;

        #[test]
        fn converge_towards() {
            ::ui::logging::init_test_logger();

            let lattice = Lattice::from(&[
                // chosen arbitrarily
                [ 2.0,  3.0, 4.0],
                [-1.0,  7.0, 8.0],
                [-3.0, -4.0, 7.0],
            ]);

            let target_coords = CoordsKind::Fracs(vec![
                [ 0.1, 0.7, 3.3],
                [ 1.2, 1.5, 4.3],
                [ 0.1, 1.2, 7.8],
                [-0.6, 0.1, 0.8],
            ].envee());
            let start_coords = CoordsKind::Fracs(vec![
                [ 1.2, 1.5, 4.3],
                [ 0.1, 0.7, 3.3],
                [-0.6, 0.1, 0.4],
                [ 0.1, 1.2, 7.8],
            ].envee());
            let expected_fracs = vec![
                [ 1.1, 1.7, 4.3],
                [ 0.2, 0.5, 3.3],
                [-0.9, 0.2, 0.8],
                [ 0.4, 1.1, 7.8],
            ];

            let target = Coords::new(lattice.clone(), target_coords.clone());
            let start = Coords::new(lattice.clone(), start_coords.clone());

            let diff_fn = ConvergeTowards::new(target);
            let cg_settings = &from_json!{{
                "stop-condition": {"grad-max": 1e-10},
                "alpha-guess-first": 0.1,
            }};

            let mut flat_diff_fn = diff_fn.initialize_flat_diff_fn(&start, ()).unwrap();
            let flat_diff_fn = &mut *flat_diff_fn;

            let data = ::rsp2_minimize::acgsd(
                cg_settings,
                start.to_carts().flat(),
                &mut *flat_diff_fn,
            ).unwrap();
            println!("DerpG: {:?}", flat_diff_fn(start.to_carts().flat()).unwrap().1);
            println!("NumrG: {:?}", ::rsp2_minimize::numerical::gradient(
                1e-7, None,
                start.to_carts().flat(),
                |p| FailOk(flat_diff_fn(p)?.0),
            ).unwrap());
            println!(" Grad: {:?}", data.gradient);
            println!("Numer: {:?}", ::rsp2_minimize::numerical::gradient(
                1e-7, None,
                &data.position,
                |p| FailOk(flat_diff_fn(p)?.0),
            ).unwrap());
            let final_carts = data.position.nest::<V3>().to_vec();
            let final_fracs = CoordsKind::Carts(final_carts).into_fracs(&lattice);
            assert_close!(final_fracs.unvee(), expected_fracs);
        }
    }
}
