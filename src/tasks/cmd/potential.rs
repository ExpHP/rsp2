// The purpose of this module is to wrap `rsp2_lammps_wrap` with code specific to
// the potentials we care about using, and to support ElementStructure.
//
// This is where we decide e.g. atom type assignments and `pair_coeff` commands.
// (which are decisions that `rsp2_lammps_wrap` has largely chosen to defer)

use ::FailResult;
use ::rsp2_structure::{Structure, consts};
use ::rsp2_structure::{Element};
use ::rsp2_structure::layer::Layers;
use ::rsp2_tasks_config as cfg;
#[allow(unused)] // rustc bug
use ::rsp2_array_types::{V3, Unvee};
#[allow(unused)] // rustc bug
use ::slice_of_array::prelude::*;
use ::std::collections::BTreeMap;
use rsp2_structure::Coords;

const DEFAULT_KC_Z_CUTOFF: f64 = 14.0; // (Angstrom?)
const DEFAULT_KC_Z_MAX_LAYER_SEP: f64 = 4.5; // Angstrom

const DEFAULT_AIREBO_LJ_SIGMA:    f64 = 3.0; // (cutoff, x3.4 A)
const DEFAULT_AIREBO_LJ_ENABLED:      bool = true;
const DEFAULT_AIREBO_TORSION_ENABLED: bool = false;

/// Trait alias for a function producing flat potential and gradient,
/// for compatibility with `rsp2_minimize`.
pub trait FlatDiffFn: FnMut(&[f64]) -> FailResult<(f64, Vec<f64>)> {}

impl<F> FlatDiffFn for F where F: FnMut(&[f64]) -> FailResult<(f64, Vec<f64>)> {}

// Type aliases for the trait object types, to work around #23856
pub type DynFlatDiffFn<'a> = FlatDiffFn<Output=FailResult<(f64, Vec<f64>)>> + 'a;

/// This is what gets passed around by very high level code to represent a
/// potential function in very high-level code. Basically:
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
pub trait PotentialBuilder<Meta = Element>
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
    fn initialize_diff_fn(&self, structure: Structure<Meta>) -> FailResult<Box<DiffFn<Meta>>>;

    /// Convenience method to get a function suitable for `rsp2_minimize`.
    ///
    /// The structure given to this is used to supply the lattice and metadata.
    /// Also, some other data may be precomputed from it.
    ///
    /// Because Boxes don't implement `Fn` traits for technical reasons,
    /// you will likely need to write `&mut *flat_diff_fn` in order to get
    /// a `&mut DynFlatDiffFn`.
    fn initialize_flat_diff_fn(&self, structure: Structure<Meta>) -> FailResult<Box<DynFlatDiffFn<'static>>>
    where Meta: Clone + 'static
    {
        let mut diff_fn = self.initialize_diff_fn(structure.clone())?;
        let mut structure = structure;
        Ok(Box::new(move |pos: &[f64]| Ok({
            structure.set_carts(pos.nest().to_vec());

            let (value, grad) = diff_fn.compute(&structure)?;
            (value, grad.unvee().flat().to_vec())
        })))
    }

    /// Convenience adapter for one-off computations.
    ///
    /// The output type is a DiffFn instance that will initialize the true `DiffFn`
    /// from scratch each time it is called. (though presumably, you only intend
    /// to call it once!)
    ///
    /// Usage: `pot.one_off().compute(&structure)`
    ///
    /// This is provided because otherwise, you end up needing to write stuff
    /// like `pot.initialize_diff_fn(structure.clone()).compute(&structure)`, which
    /// is both confusing and awkward.
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

    fn initialize_diff_fn(&self, structure: Structure<Meta>) -> FailResult<Box<DiffFn<Meta>>>
    { (**self).initialize_diff_fn(structure) }

    fn initialize_flat_diff_fn(&self, structure: Structure<Meta>) -> FailResult<Box<DynFlatDiffFn<'static>>>
    where Meta: Clone + 'static
    { (**self).initialize_flat_diff_fn(structure) }

    fn one_off(&self) -> OneOff<'_, Meta>
    where Meta: Clone,
    { (**self).one_off() }
}

impl_dyn_clone_detail!{
    impl[Meta: Clone + 'static] DynCloneDetail<Meta> for Box<PotentialBuilder<Meta>> { ... }
}

//-------------------------------------

/// This is `FnMut(ElementStructure) -> FailResult<(f64, Vec<V3>)>` with convenience methods.
///
/// A `DiffFn` may contain pre-computed or cached data that is only valid for
/// certain structures.  Most code that handles `DiffFns` must handle them
/// opaquely, so generally speaking, *all code that uses a `DiffFn`* is subject
/// to *union of the limitations across all of the implementations.*
///
/// (...not a huge deal since all uses and all implementations are local to this crate)
pub trait DiffFn<Meta> {
    /// Compute the value and gradient.
    fn compute(&mut self, structure: &Structure<Meta>) -> FailResult<(f64, Vec<V3>)>;

    /// Convenience method to compute the potential.
    fn compute_value(&mut self, structure: &Structure<Meta>) -> FailResult<f64>
    { Ok(self.compute(structure)?.0) }

    /// Convenience method to compute the gradient.
    fn compute_grad(&mut self, structure: &Structure<Meta>) -> FailResult<Vec<V3>>
    { Ok(self.compute(structure)?.1) }

    /// Convenience method to compute the force.
    fn compute_force(&mut self, structure: &Structure<Meta>) -> FailResult<Vec<V3>>
    {
        let mut force = self.compute_grad(structure)?;
        for v in &mut force { *v = -*v; }
        Ok(force)
    }

    /// Compute the change in force when displacing an atom, in a sparse representation.
    ///
    /// Some potentials may assume that `original` has zero force.
    ///
    /// The default implementation works from the dense representation of the forces.
    fn compute_force_set(&mut self, original: &Structure<Meta>, displacement: (usize, V3)) -> FailResult<BTreeMap<usize, V3>>
    where Meta: Clone,
    {
        let mut structure = original.clone();
        // FIXME maybe ensure_only_carts() should be exposed.
        let _ = structure.carts_mut(); // prevent numerical differences from change in representation

        // FIXME this is going to get recomputed frequently unless we change the signature.
        let original_force = self.compute_force(&structure)?;
        structure.carts_mut()[displacement.0] += displacement.1;
        let displaced_force = self.compute_force(&structure)?;

        let diffs = {
            zip_eq!(original_force, displaced_force).enumerate()
                // assuming that the potential is deterministic and implements a cutoff radius,
                // this might actually succeed at filtering out a lot of zero terms.
                .filter(|(_, (old, new))| old != new)
                .map(|(atom, (old, new))| (atom, new - old))

                // this one is a closer approximation of phonopy
//                .map(|(atom, (_old, new))| (atom, new))
        };
        Ok(diffs.collect())
//        Ok(::math::dynmat::ForceSets::from_displacement(
//            structure.num_atoms(),
//            displacement,
//            diffs,
//        ))
    }
}

// necessary for combinators like sum
impl<'d, Meta> DiffFn<Meta> for Box<DiffFn<Meta> + 'd> {
    /// Compute the value and gradient.
    fn compute(&mut self, structure: &Structure<Meta>) -> FailResult<(f64, Vec<V3>)>
    { (**self).compute(structure) }

    /// Convenience method to compute the potential.
    fn compute_value(&mut self, structure: &Structure<Meta>) -> FailResult<f64>
    { (**self).compute_value(structure) }

    /// Convenience method to compute the gradient.
    fn compute_grad(&mut self, structure: &Structure<Meta>) -> FailResult<Vec<V3>>
    { (**self).compute_grad(structure) }

    /// Convenience method to compute the force.
    fn compute_force(&mut self, structure: &Structure<Meta>) -> FailResult<Vec<V3>>
    { (**self).compute_force(structure) }
}

//-------------------------------------

/// See `PotentialBuilder::one_off` for more information.
pub struct OneOff<'a, M: 'a>(&'a PotentialBuilder<M>);
impl<'a, M: Clone + 'static> DiffFn<M> for OneOff<'a, M> {
    fn compute(&mut self, structure: &Structure<M>) -> FailResult<(f64, Vec<V3>)> {
        self.0.initialize_diff_fn(structure.clone())?.compute(structure)
    }
}

//-------------------------------------

/// High-level logic
impl PotentialBuilder {
    pub(crate) fn from_config(
        threading: &cfg::Threading,
        config: &cfg::PotentialKind,
    ) -> Box<PotentialBuilder> {
        match config {
            cfg::PotentialKind::Rebo => {
                let lammps_pot = self::lammps::Airebo::Rebo;
                let pot = self::lammps::Builder::new(threading, lammps_pot);
                Box::new(pot)
            }
            cfg::PotentialKind::Airebo(cfg) => {
                let lammps_pot = self::lammps::Airebo::from(cfg);
                let pot = self::lammps::Builder::new(threading, lammps_pot);
                Box::new(pot)
            },
            cfg::PotentialKind::KolmogorovCrespiZ(cfg) => {
                let lammps_pot = self::lammps::KolmogorovCrespiZ::from(cfg);
                let pot = self::lammps::Builder::new(threading, lammps_pot);
                Box::new(pot)
            },
            cfg::PotentialKind::KolmogorovCrespiZNew(cfg) => {
                let rebo = PotentialBuilder::from_config(threading, &cfg::PotentialKind::Rebo);
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

        fn initialize_diff_fn(&self, structure: Structure<M>) -> FailResult<Box<DiffFn<M>>>
        {
            let a_diff_fn = self.0.initialize_diff_fn(structure.clone())?;
            let b_diff_fn = self.1.initialize_diff_fn(structure.clone())?;
            Ok(Box::new(Sum(a_diff_fn, b_diff_fn)))
        }
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
        A: DiffFn<M>,
        B: DiffFn<M>,
    {
        fn compute(&mut self, structure: &Structure<M>) -> FailResult<(f64, Vec<V3>)> {
            let (a_value, a_grad) = self.0.compute(structure)?;
            let (b_value, b_grad) = self.1.compute(structure)?;
            let value = a_value + b_value;

            let mut grad = a_grad;
            for (out_vec, b_vec) in zip_eq!(&mut grad, b_grad) {
                *out_vec += b_vec;
            }
            Ok((value, grad))
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

    impl PotentialBuilder<Element> for KolmogorovCrespiZ {
        fn initialize_diff_fn(&self, structure: Structure<Element>) -> FailResult<Box<DiffFn<Element>>>
        {
            struct Diff {
                params: ::math::crespi::Params,
                bonds: FracBonds,
            }
            impl DiffFn<Element> for Diff {
                fn compute(&mut self, structure: &Structure<Element>) -> FailResult<(f64, Vec<V3>)> {
                    let bonds = self.bonds.to_cart_bonds(structure);

                    let mut value = 0.0;
                    let mut grad = vec![V3::zero(); structure.num_atoms()];
                    for CartBond { from: _, to, cart_vector } in &bonds {
                        let ::math::crespi::Output {
                            value: part_value,
                            grad_rij: part_grad, ..
                        } = self.params.crespi_z(*cart_vector);

                        value += part_value;
                        grad[to] += part_grad;
                    }
                    trace!("KCZ: {}", value);
                    Ok((value, grad))
                }
            }

            let cfg::PotentialKolmogorovCrespiZNew { cutoff_begin } = self.0;
            let mut params = ::math::crespi::Params::default();
            if let Some(cutoff_begin) = cutoff_begin {
                params.cutoff_begin = cutoff_begin;
            }
            let bonds = FracBonds::from_brute_force_very_dumb(&structure, params.cutoff_end() * 1.001)?;
            Ok(Box::new(Diff { params, bonds }))
        }
    }

    impl_dyn_clone_detail!{
        impl[] DynCloneDetail<Element> for KolmogorovCrespiZ { ... }
    }
}

//-------------------------------------

/// All usage of the public API presented by `rsp2_lammps_wrap` is encapsulated here.
mod lammps {
    use super::*;

    use ::rsp2_lammps_wrap::{InitInfo, AtomType, PairCommand};
    use ::rsp2_lammps_wrap::Builder as InnerBuilder;
    use ::rsp2_lammps_wrap::Potential as LammpsPotential;

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
            threading: &cfg::Threading,
            potential: P,
        ) -> Self {
            let mut inner = InnerBuilder::new();
            inner.append_log("lammps.log");
            inner.threaded(*threading == cfg::Threading::Lammps);

            Builder { inner, potential }
        }

        pub(crate) fn threaded(&self, threaded: bool) -> Self {
            let mut me = self.clone();
            me.inner.threaded(threaded);
            me
        }
    }

    impl<M: Clone + 'static, P: LammpsPotential<Meta=Vec<M>> + Clone + Send + Sync + 'static> Builder<P>
    {
        /// Initialize Lammps to make a DiffFn.
        ///
        /// This keeps the Lammps instance between calls to save time.
        ///
        /// Some data may be pre-allocated or precomputed based on the input structure,
        /// so the resulting DiffFn may not support arbitrary structures as input.
        pub(crate) fn diff_fn(&self, structure: Structure<M>) -> FailResult<Box<DiffFn<M>>>
        {
            // a DiffFn 'lambda' whose type will be erased
            struct MyDiffFn<Mm: Clone>(::rsp2_lammps_wrap::Lammps<Box<LammpsPotential<Meta=Vec<Mm>>>>);
            impl<Mm: Clone> DiffFn<Mm> for MyDiffFn<Mm> {
                fn compute(&mut self, structure: &Structure<Mm>) -> FailResult<(f64, Vec<V3>)> {
                    let lmp = &mut self.0;

                    let (coords, meta) = structure.clone().into_parts();
                    lmp.set_structure(coords, meta)?;
                    let value = lmp.compute_value()?;
                    let grad = lmp.compute_grad()?;
                    Ok((value, grad))
                }
            }

            let lammps_pot = Box::new(self.potential.clone()) as Box<LammpsPotential<Meta=P::Meta>>;
            let (coords, meta) = structure.into_parts();
            let lmp = self.inner.build(lammps_pot, coords, meta)?;
            Ok(Box::new(MyDiffFn::<M>(lmp)) as Box<_>)
        }
    }

    impl<M: Clone + 'static, P: Clone + LammpsPotential<Meta=Vec<M>> + Send + Sync + 'static> PotentialBuilder<M> for Builder<P>
    {
        fn threaded(&self, threaded: bool) -> Box<PotentialBuilder<M>>
        { Box::new(<Builder<_>>::threaded(self, threaded)) }

        fn initialize_diff_fn(&self, structure: Structure<M>) -> FailResult<Box<DiffFn<M>>>
        { self.diff_fn(structure) }
    }

    impl_dyn_clone_detail!{
        impl[M: Clone + 'static, P: Clone + LammpsPotential<Meta=Vec<M>> + Send + Sync + 'static]
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

        #[derive(Debug, Clone)]
        pub struct Rebo;

        impl<'a> From<&'a cfg::PotentialAirebo> for Airebo {
            fn from(cfg: &'a cfg::PotentialAirebo) -> Self {
                let cfg::PotentialAirebo {
                    lj_sigma, lj_enabled, torsion_enabled,
                } = *cfg;

                Airebo::Airebo {
                    lj_sigma: lj_sigma.unwrap_or(DEFAULT_AIREBO_LJ_SIGMA),
                    lj_enabled: lj_enabled.unwrap_or(DEFAULT_AIREBO_LJ_ENABLED),
                    torsion_enabled: torsion_enabled.unwrap_or(DEFAULT_AIREBO_TORSION_ENABLED),
                }
            }
        }

        impl LammpsPotential for Airebo {
            type Meta = Vec<Element>;

            fn atom_types(&self, _: &Coords, elements: &Vec<Element>) -> Vec<AtomType>
            { elements.iter().map(|elem| match elem.symbol() {
                "H" => AtomType::new(1),
                "C" => AtomType::new(2),
                sym => panic!("Unexpected element in Airebo: {}", sym),
            }).collect() }

            fn init_info(&self, _: &Coords, _: &Vec<Element>) -> InitInfo
            {
                InitInfo {
                    masses: vec![
                        ::common::element_mass(consts::HYDROGEN).unwrap(),
                        ::common::element_mass(consts::CARBON).unwrap(),
                    ],
                    pair_commands: match *self {
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
                    },
                }
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
            type Meta = Vec<Element>;

            fn atom_types(&self, coords: &Coords, _: &Vec<Element>) -> Vec<AtomType>
            {
                self.find_layers(coords)
                    .by_atom().into_iter()
                    .map(|x| AtomType::from_index(x))
                    .collect()
            }

            fn init_info(&self, coords: &Coords, _: &Vec<Element>) -> InitInfo
            {
                let layers = match self.find_layers(coords).per_unit_cell() {
                    None => panic!("kolmogorov/crespi/z is only supported for layered materials"),
                    Some(layers) => layers,
                };

                let masses = vec![::common::element_mass(consts::CARBON).unwrap(); layers.len()];

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
                //
                // TODO: Should maybe factor out some Hybrid: Potential that takes
                //       care of this nonsense, and put it in rsp2_lammps_wrap sometime.
                //       Then again, it seems like a dangerous abstraction, considering
                //       that the syntax for 'pair_style hybrid' obviously has massive
                //       room for ambiguity (which of course, the Lammps documentation
                //       does not acknowledge, because physicists are blind I guess).
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
}

//-------------------------------------

/// Dummy potentials for testing purposes
pub mod test {
    use super::*;
    use ::rsp2_structure::{Coords, CoordsKind};

    /// The test Potential `V = 0`.
    #[derive(Debug, Clone)]
    pub struct Zero;

    impl<Meta: Clone> PotentialBuilder<Meta> for Zero {
        fn initialize_diff_fn<'a>(&self, _: Structure<Meta>) -> FailResult<Box<DiffFn<Meta>>>
        {
            struct Diff;
            impl<M> DiffFn<M> for Diff {
                fn compute(&mut self, structure: &Structure<M>) -> FailResult<(f64, Vec<V3>)> {
                    Ok((0.0, vec![V3([0.0; 3]); structure.num_atoms()]))
                }
            }
            Ok(Box::new(Diff) as Box<_>)
        }
    }

    impl_dyn_clone_detail!{
        impl[M: Clone] DynCloneDetail<M> for Zero { ... }
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
    impl<Meta: Clone> PotentialBuilder<Meta> for ConvergeTowards {
        fn initialize_diff_fn(&self, _: Structure<Meta>) -> FailResult<Box<DiffFn<Meta>>>
        { Ok(Box::new(self.clone()) as Box<_>) }
    }

    impl_dyn_clone_detail!{
        impl[Meta: Clone] DynCloneDetail<Meta> for ConvergeTowards { ... }
    }

    impl<M> DiffFn<M> for ConvergeTowards {
        fn compute(&mut self, structure: &Structure<M>) -> FailResult<(f64, Vec<V3>)> {
            (&*self).compute(structure)
        }
    }

    // ConvergeTowards does not get mutated
    impl<'a, M> DiffFn<M> for &'a ConvergeTowards {
        fn compute(&mut self, structure: &Structure<M>) -> FailResult<(f64, Vec<V3>)> {
            assert_eq!(structure.num_atoms(), self.target.num_atoms());
            assert_close!(abs=1e-8, structure.lattice(), self.target.lattice());

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

            let cur_fracs = structure.to_fracs();
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
            let recip = structure.lattice().reciprocal();
            let cart_grad = frac_grad.map(|v| v * &recip).collect::<Vec<_>>();
            Ok((value, cart_grad))
        }
    }

    // ---------------

    /// A test Potential that creates a chain along the first lattice vector.
    #[derive(Debug, Clone)]
    pub struct Chainify;

    impl<Meta: Clone> PotentialBuilder<Meta> for Chainify {
        fn initialize_diff_fn(&self, structure: Structure<Meta>) -> FailResult<Box<DiffFn<Meta>>>
        {
            let na = structure.num_atoms();
            let fracs = (0..na).map(|i| {
                V3([i as f64 / na as f64, 0.5, 0.5])
            }).collect();
            let coords = CoordsKind::Fracs(fracs);
            let target = Coords::new(structure.lattice().clone(), coords);
            Ok(Box::new(ConvergeTowards::new(target)) as Box<_>)
        }
    }

    impl_dyn_clone_detail!{
        impl[Meta: Clone] DynCloneDetail<Meta> for Chainify { ... }
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

            let meta = vec![(); start_coords.len()];
            let target = Coords::new(lattice.clone(), target_coords.clone());
            let start = Structure::new(lattice.clone(), start_coords.clone(), meta.clone());

            let diff_fn = ConvergeTowards::new(target);
            let cg_settings = &from_json!{{
                "stop-condition": {"grad-max": 1e-10},
                "alpha-guess-first": 0.1,
            }};

            let mut flat_diff_fn = diff_fn.initialize_flat_diff_fn(start.clone()).unwrap();
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
