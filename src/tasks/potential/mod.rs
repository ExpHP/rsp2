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

use ::FailResult;
use ::hlist_aliases::*;
use ::meta;
use ::rsp2_structure::{Coords};
use ::rsp2_tasks_config as cfg;
use ::rsp2_array_types::{V3, Unvee};
use ::slice_of_array::prelude::*;
use ::std::collections::BTreeMap;
use ::cmd::trial::TrialDir;
use ::rsp2_lammps_wrap::LammpsOnDemand;

/// Metadata type shared by all potentials usable in the main code.
///
/// (all potentials usable in the main code must use a single metadata
///  type by necessity, due to the use of dynamic polymorphism)
pub type CommonMeta = HList3<
    meta::SiteElements,
    meta::SiteMasses,
    Option<meta::FracBonds>,
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
    // NOTE: when adding methods like "parallel", make sure to override the
    //       default implementations in generic impls!!!
    //       (e.g. Box<PotentialBuilder<M>>, Sum<A, B>, ...)
    /// Sometimes called as a last-minute hint to control parallelism
    /// within the potential based on the current circumstances.
    ///
    /// `.parallel(true)` is an invitation to enable maximum parallelism.
    /// `.parallel(false)` forbids all forms of parallelism.
    ///
    /// This method is not reversible.  If `pot` has parallelism enabled, there is no guarantee that
    /// `pot.parallel(false).parallel(true)` will have the same settings.
    ///
    /// The default implementation just ignores the call.
    #[must_use = "this is not an in-place mutation!"]
    fn parallel(&self, _parallel: bool) -> Box<dyn PotentialBuilder<Meta>>
    { self.box_clone() }

    /// May be used to provide a hint that blocking indefinitely for a resource
    /// is acceptible. Intended for use in unit tests, which cargo tries to run
    /// in parallel.
    #[must_use = "this is not an in-place mutation!"]
    fn allow_blocking(&self, _allow: bool) -> Box<dyn PotentialBuilder<Meta>>
    { self.box_clone() }

    /// Create the DiffFn.  This does potentially expensive initialization, maybe calling out
    /// to external C APIs and etc.
    ///
    /// The structure given to this is used to supply the lattice and metadata.
    /// Also, some other data may be precomputed from it.
    fn initialize_diff_fn(&self, coords: &Coords, meta: Meta) -> FailResult<Box<dyn DiffFn<Meta>>>;

    /// Create a DiffFn that computes individual forces per pair interaction.
    ///
    /// Not all potentials support this; the default implementation simply returns `Ok(None)`.
    /// An implementation is required if one wants to optimize lattice params during relaxation.
    ///
    /// The structure given to this is used to supply the lattice and metadata, and possibly
    /// even the set of bonds (e.g. in the case of nonreactive REBO).
    fn initialize_bond_diff_fn(&self, _init_coords: &Coords, _meta: Meta) -> FailResult<Option<Box<dyn BondDiffFn<Meta>>>>
    where Meta: Clone + 'static
    { Ok(None) }

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
    fn initialize_disp_fn(&self, equilibrium_coords: &Coords, meta: Meta) -> FailResult<Box<dyn DispFn>>
    where Meta: Clone + 'static,
    ;

    /// default impl for initialize_disp_fn, which cannot be inlined because it needs `Self: Sized`.
    ///
    /// **This method exists solely for the convenience of implementors of the trait.** It should not
    /// be overridden, and the only place it should ever be used is in the definition of
    /// `initialize_disp_fn` in a trait impl.
    fn _default_initialize_disp_fn(&self, coords: &Coords, meta: Meta) -> FailResult<Box<dyn DispFn>>
    where
        Meta: Clone + 'static,
        Self: Sized,
    { Ok(Box::new(helper::DefaultDispFn::initialize(coords, meta, self)?)) }

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

    /// An object-safe method used to implement `eco_mode`.
    ///
    /// The default implementation simply calls `cont`.
    fn _eco_mode(&self, cont: &mut dyn FnMut())
    { cont() }
}

// On the trait object rather than the trait to dodge Self: Sized issues.
impl<Meta> dyn PotentialBuilder<Meta>
where
    Meta: Clone + 'static
{
    /// Some implementations of PotentialBuilder might result in loads of 100% CPU on many
    /// processors merely by existing.
    ///
    /// This method tells them to hold off for the duration of a closure.
    ///
    /// The default implementation simply calls the closure.
    ///
    /// # Caution
    ///
    /// **You must not call other methods of `PotentialBuilder` from inside the closure!**
    /// (or else it might deadlock, or cause MPI to abort, or somesuch).  Ideally, this method
    /// would take `&mut self` to signify this limitation, but that would require a massive design
    /// overhaul for `PotentialBuilder`, soooo...... *just be careful.*
    pub fn eco_mode<B>(&self, cont: impl FnOnce() -> B) -> B {
        let mut cont = Some(cont);
        let mut out = None;
        { // NLL please save us ;_;
            let mut cont_as_fn_mut = || {
                let cont = cont.take().expect("(BUG!) _eco_mode called continuation twice!");
                out = Some(cont());
            };
            self._eco_mode(&mut cont_as_fn_mut);
        }

        out.expect("(BUG!) _eco_mode failed to call continuation!")
    }
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
    fn box_clone(&self) -> Box<dyn PotentialBuilder<Meta>>;

    /// "Borrow" the trait object.
    fn _as_ref_dyn(&self) -> &dyn PotentialBuilder<Meta>;
}

#[macro_export]
macro_rules! impl_dyn_clone_detail {
    (impl[$($bnd:tt)*] DynCloneDetail<$Meta:ty> for $Type:ty { ... }) => {
        impl<$($bnd)*> DynCloneDetail<$Meta> for $Type {
            fn box_clone(&self) -> Box<dyn PotentialBuilder<$Meta>> {
                Box::new(<$Type as Clone>::clone(self))
            }
            fn _as_ref_dyn(&self) -> &dyn PotentialBuilder<$Meta> { self }
        }
    };
}

impl<M> Clone for Box<dyn PotentialBuilder<M>>
where M: 'static, // FIXME why is this necessary? PotentialBuilder doesn't borrow from M...
{
    fn clone(&self) -> Self { self.box_clone() }
}

// necessary for combinators like Sum to be possible
impl<Meta> PotentialBuilder<Meta> for Box<dyn PotentialBuilder<Meta>>
where Meta: Clone + 'static,
{
    fn parallel(&self, parallel: bool) -> Box<dyn PotentialBuilder<Meta>>
    { (**self).parallel(parallel) }

    fn allow_blocking(&self, allow: bool) -> Box<dyn PotentialBuilder<Meta>>
    { (**self).allow_blocking(allow) }

    fn initialize_diff_fn(&self, coords: &Coords, meta: Meta) -> FailResult<Box<dyn DiffFn<Meta>>>
    { (**self).initialize_diff_fn(coords, meta) }

    fn initialize_flat_diff_fn(&self, coords: &Coords, meta: Meta) -> FailResult<Box<DynFlatDiffFn<'static>>>
    { (**self).initialize_flat_diff_fn(coords, meta) }

    fn one_off(&self) -> OneOff<'_, Meta>
    { (**self).one_off() }

    fn initialize_disp_fn(&self, equilibrium_coords: &Coords, meta: Meta) -> FailResult<Box<dyn DispFn>>
    { (**self).initialize_disp_fn(equilibrium_coords, meta) }

    fn _eco_mode(&self, cont: &mut dyn FnMut())
    { (**self)._eco_mode(cont) }
}

impl_dyn_clone_detail!{
    impl[Meta: Clone + 'static] DynCloneDetail<Meta> for Box<dyn PotentialBuilder<Meta>> { ... }
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
impl<'d, Meta> DiffFn<Meta> for Box<dyn DiffFn<Meta> + 'd> {
    fn compute(&mut self, coords: &Coords, meta: Meta) -> FailResult<(f64, Vec<V3>)>
    { (**self).compute(coords, meta) }

    fn compute_value(&mut self, coords: &Coords, meta: Meta) -> FailResult<f64>
    { (**self).compute_value(coords, meta) }

    fn compute_grad(&mut self, coords: &Coords, meta: Meta) -> FailResult<Vec<V3>>
    { (**self).compute_grad(coords, meta) }

    fn compute_force(&mut self, coords: &Coords, meta: Meta) -> FailResult<Vec<V3>>
    { (**self).compute_force(coords, meta) }
}

//-------------------------------------

/// Represents a single pair interaction in the potential.
pub struct BondGrad {
    /// The bond vector `carts[plus_site] - carts[minus_site] + unspecified_lattice_point`.
    pub cart_vector: V3,
    /// The gradient of a summand in the potential with respect to that vector.
    pub grad: V3,
    pub minus_site: usize,
    pub plus_site: usize,
}

pub trait BondDiffFn<Meta> {
    fn compute(&mut self, coords: &Coords, meta: Meta) -> FailResult<(f64, Vec<BondGrad>)>;
}

// necessary for combinators like sum
impl<'d, Meta> BondDiffFn<Meta> for Box<dyn BondDiffFn<Meta> + 'd> {
    fn compute(&mut self, coords: &Coords, meta: Meta) -> FailResult<(f64, Vec<BondGrad>)>
    { (**self).compute(coords, meta) }
}

//-------------------------------------

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
/// ```text
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

/// This is `FnMut((usize, V3)) -> FailResult<(f64, Vec<V3>)>` with convenience methods.
///
/// A `DispFn` usually has been initialized with the equilibrium structure, and may contain
/// pre-computed equilibrium forces. It may behave differently from a DiffFn in order to take
/// advantage of reliable properties of the structures produced by displacements.
pub trait DispFn {
    /// Compute the change in force caused by the displacement.
    fn compute_dense_force(&mut self, disp: (usize, V3)) -> FailResult<Vec<V3>>;

    /// Compute the change in force caused by the displacement, in a sparse representation.
    ///
    /// This is not required to be identical to `compute_dense_force_delta`; the two methods
    /// may differ in assumptions (e.g. one method might treat the equilibrium structure as
    /// having zero force).
    fn compute_sparse_force_delta(&mut self, disp: (usize, V3)) -> FailResult<BTreeMap<usize, V3>>;
}

//-------------------------------------

/// See `PotentialBuilder::one_off` for more information.
pub struct OneOff<'a, M: 'a>(&'a dyn PotentialBuilder<M>);
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
        on_demand: Option<LammpsOnDemand>,
        cfg: &cfg::Settings,
    ) -> Box<PotentialBuilder> {
        Self::from_config_parts(
            Some(trial_dir),
            on_demand,
            &cfg.threading,
            &cfg.lammps_update_style,
            &cfg.lammps_processor_axis_mask,
            &cfg.potential,
        )
    }

    pub(crate) fn from_config_parts(
        trial_dir: Option<&TrialDir>,
        on_demand: Option<LammpsOnDemand>,
        threading: &cfg::Threading,
        update_style: &cfg::LammpsUpdateStyle,
        axis_mask: &[bool; 3], // HACK; lammps should have its own section
        config: &cfg::PotentialKind,
    ) -> Box<PotentialBuilder> {
        match config {
            cfg::PotentialKind::Rebo(cfg) => {
                let lammps_pot = self::lammps::Airebo::from(cfg);
                let pot = self::lammps::Builder::new(trial_dir, on_demand, threading, update_style, axis_mask, lammps_pot);
                Box::new(pot)
            }
            cfg::PotentialKind::Airebo(cfg) => {
                let lammps_pot = self::lammps::Airebo::from(cfg);
                let pot = self::lammps::Builder::new(trial_dir, on_demand, threading, update_style, axis_mask, lammps_pot);
                Box::new(pot)
            },
            cfg::PotentialKind::KolmogorovCrespiZ(cfg) => {
                let lammps_pot = self::lammps::KolmogorovCrespiZ::from(cfg);
                let pot = self::lammps::Builder::new(trial_dir, on_demand, threading, update_style, axis_mask, lammps_pot);
                Box::new(pot)
            },
            cfg::PotentialKind::KolmogorovCrespiZNew(cfg) => {
                let rebo_cfg = cfg::PotentialKind::Rebo(from_json!{ {} });
                let rebo = PotentialBuilder::from_config_parts(trial_dir, on_demand, threading, update_style, axis_mask, &rebo_cfg);
                let kc_z = self::homestyle::KolmogorovCrespiZ(cfg.clone());
                let pot = self::helper::Sum(rebo, kc_z);
                Box::new(pot)
            },
            cfg::PotentialKind::ReboNew(cfg) => {
                let cfg = cfg.clone();
                let parallel = threading == &cfg::Threading::Rayon;
                Box::new(self::homestyle::Rebo { cfg, parallel })
            },
            cfg::PotentialKind::TestZero => Box::new(self::test_functions::Zero),
            cfg::PotentialKind::TestChainify => Box::new(self::test_functions::Chainify),
        }
    }
}

//-------------------------------------
// interop with other crates in rsp2

pub struct Rsp2MinimizeDiffFnShim {
    pub ndim: usize,
    pub diff_fn: Box<DynFlatDiffFn<'static>>
}

impl<'a> ::rsp2_minimize::test::n_dee::OnceDifferentiable for Rsp2MinimizeDiffFnShim {
    fn ndim(&self) -> usize
    { self.ndim }

    fn diff(&mut self, pos: &[f64]) -> (f64, Vec<f64>)
    { (*self.diff_fn)(pos).unwrap() }
}

//-------------------------------------
// Implementations

// (these are down here because they depend on the macro defined above)

pub mod test_functions;

mod helper;

mod homestyle;

mod lammps;

