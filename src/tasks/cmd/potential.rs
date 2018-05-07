// The purpose of this module is to wrap `rsp2_lammps_wrap` with code specific to
// the potentials we care about using, and to support ElementStructure.
//
// This is where we decide e.g. atom type assignments and `pair_coeff` commands.
// (which are decisions that `rsp2_lammps_wrap` has largely chosen to defer)

use ::FailResult;
use ::rsp2_structure::{Structure, ElementStructure, consts};
use ::rsp2_structure::{Layers, Element};
use ::rsp2_tasks_config as cfg;
#[allow(unused)] // rustc bug
use ::rsp2_array_types::{V3, Unvee};
#[allow(unused)] // rustc bug
use ::slice_of_array::prelude::*;

const DEFAULT_KC_Z_CUTOFF: f64 = 14.0; // (Angstrom?)
const DEFAULT_KC_Z_MAX_LAYER_SEP: f64 = 4.5; // Angstrom

const DEFAULT_AIREBO_LJ_SIGMA:    f64 = 3.0; // (cutoff, x3.4 A)
const DEFAULT_AIREBO_LJ_ENABLED:      bool = true;
const DEFAULT_AIREBO_TORSION_ENABLED: bool = false;

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
{
    /// Sometimes called as a last-minute hint to control threading
    /// within the potential based on the current circumstances.
    ///
    /// Implementations that do not care may simply call `box_clone()`.
    #[cfg_attr(feature = "nightly", must_use = "this is not an in-place mutation!")]
    fn threaded(&self, _threaded: bool) -> Box<PotentialBuilder<Meta>>;

    /// "Clone" the trait object.
    fn box_clone(&self) -> Box<PotentialBuilder<Meta>>;

    /// dumb dumb dumb stupid implementation detail.
    ///
    /// A default implementation cannot be provided. Just return `self`.
    fn _as_ref_dyn(&self) -> &PotentialBuilder<Meta>;

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
    fn one_off<'r>(&'r self) -> OneOff<'r, Meta>
    where Meta: Clone,
    { OneOff(self._as_ref_dyn()) }
}

/// A simple implementation of `box_clone` that most implementors of `PotentialBuilder`
/// can use as long as they implement `Clone`.
pub fn simple_box_clone<P, Meta>(pot: &P) -> Box<PotentialBuilder<Meta>>
where
    P: PotentialBuilder<Meta> + Clone,
{ Box::new(pot.clone()) }

impl<M> Clone for Box<PotentialBuilder<M>>
where M: 'static, // FIXME why is this necessary? PotentialBuilder doesn't borrow from M...
{
    fn clone(&self) -> Self { self.box_clone() }
}

/// Trait alias for a function producing flat potential and gradient,
/// for compatibility with `rsp2_minimize`.
pub trait FlatDiffFn: FnMut(&[f64]) -> FailResult<(f64, Vec<f64>)> {}

impl<F> FlatDiffFn for F where F: FnMut(&[f64]) -> FailResult<(f64, Vec<f64>)> {}

// Type aliases for the trait object types, to work around #23856
pub type DynFlatDiffFn<'a> = FlatDiffFn<Output=FailResult<(f64, Vec<f64>)>> + 'a;

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
}

/// See `PotentialBuilder::one_off` for more information.
pub struct OneOff<'a, M: 'a>(&'a PotentialBuilder<M>);
impl<'a, M: Clone + 'static> DiffFn<M> for OneOff<'a, M> {
    fn compute(&mut self, structure: &Structure<M>) -> FailResult<(f64, Vec<V3>)> {
        self.0.initialize_diff_fn(structure.clone())?.compute(structure)
    }
}

impl PotentialBuilder {
    pub(crate) fn from_config(
        threading: &cfg::Threading,
        config: &cfg::PotentialKind,
    ) -> Box<PotentialBuilder> {
        match *config {
            cfg::PotentialKind::Airebo(ref cfg) => {
                let lammps_pot = self::lammps::Airebo::from(cfg);
                let pot = self::lammps::Builder::new(threading, lammps_pot);
                Box::new(pot)
            },
            cfg::PotentialKind::KolmogorovCrespiZ(ref cfg) => {
                let lammps_pot = self::lammps::KolmogorovCrespiZ::from(cfg);
                let pot = self::lammps::Builder::new(threading, lammps_pot);
                Box::new(pot)
            },
            cfg::PotentialKind::TestZero => Box::new(self::test::Zero),
            cfg::PotentialKind::TestChainify => Box::new(self::test::Chainify),
        }
    }
}

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

    impl<P: LammpsPotential + Clone + Send + Sync + 'static> Builder<P>
    where P::Meta: Clone,
    {
        /// Initialize Lammps to make a DiffFn.
        ///
        /// This keeps the Lammps instance between calls to save time.
        ///
        /// Some data may be pre-allocated or precomputed based on the input structure,
        /// so the resulting DiffFn may not support arbitrary structures as input.
        pub(crate) fn diff_fn(&self, structure: Structure<P::Meta>) -> FailResult<Box<DiffFn<P::Meta>>>
        {
            // a DiffFn 'lambda' whose type will be erased
            struct MyDiffFn<M: Clone>(::rsp2_lammps_wrap::Lammps<Box<LammpsPotential<Meta=M>>>);
            impl<M: Clone> DiffFn<M> for MyDiffFn<M> {
                fn compute(&mut self, structure: &Structure<M>) -> FailResult<(f64, Vec<V3>)> {
                    let lmp = &mut self.0;

                    lmp.set_structure(structure.clone())?;
                    let value = lmp.compute_value()?;
                    let grad = lmp.compute_grad()?;
                    Ok((value, grad))
                }
            }

            let lammps_pot = Box::new(self.potential.clone()) as Box<LammpsPotential<Meta=P::Meta>>;
            let lmp = self.inner.build(lammps_pot, structure)?;
            Ok(Box::new(MyDiffFn::<P::Meta>(lmp)) as Box<_>)
        }
    }

    impl<P: Clone + LammpsPotential + Send + Sync + 'static> PotentialBuilder<P::Meta> for Builder<P>
    where P::Meta: Clone,
    {
        fn threaded(&self, threaded: bool) -> Box<PotentialBuilder<P::Meta>>
        { Box::new(<Builder<_>>::threaded(self, threaded)) }

        fn box_clone(&self) -> Box<PotentialBuilder<P::Meta>>
        { simple_box_clone(self) }

        fn _as_ref_dyn(&self) -> &PotentialBuilder<P::Meta>
        { self }

        fn initialize_diff_fn(&self, structure: Structure<P::Meta>) -> FailResult<Box<DiffFn<P::Meta>>>
        { self.diff_fn(structure) }
    }

    pub use self::airebo::Airebo;
    mod airebo {
        use super::*;

        /// Uses `pair_style airebo`.
        #[derive(Debug, Clone)]
        pub struct Airebo {
            lj_sigma: f64,
            lj_enabled: bool,
            torsion_enabled: bool,
        }

        impl<'a> From<&'a cfg::PotentialAirebo> for Airebo {
            fn from(cfg: &'a cfg::PotentialAirebo) -> Self {
                let cfg::PotentialAirebo {
                    lj_sigma, lj_enabled, torsion_enabled,
                } = *cfg;

                Airebo {
                    lj_sigma: lj_sigma.unwrap_or(DEFAULT_AIREBO_LJ_SIGMA),
                    lj_enabled: lj_enabled.unwrap_or(DEFAULT_AIREBO_LJ_ENABLED),
                    torsion_enabled: torsion_enabled.unwrap_or(DEFAULT_AIREBO_TORSION_ENABLED),
                }
            }
        }

        impl LammpsPotential for Airebo {
            type Meta = Element;

            fn atom_types(&self, structure: &ElementStructure) -> Vec<AtomType>
            { structure.metadata().iter().map(|elem| match elem.symbol() {
                "H" => AtomType::new(1),
                "C" => AtomType::new(2),
                sym => panic!("Unexpected element in Airebo: {}", sym),
            }).collect() }

            fn init_info(&self, _: &ElementStructure) -> InitInfo
            {
                InitInfo {
                    masses: vec![::common::element_mass(consts::HYDROGEN),
                                 ::common::element_mass(consts::CARBON)],
                    pair_commands: vec![
                        PairCommand::pair_style("airebo/omp")
                            .arg(self.lj_sigma)
                            .arg(boole(self.lj_enabled))
                            .arg(boole(self.torsion_enabled))
                            ,
                        PairCommand::pair_coeff(.., ..).args(&["CH.airebo", "H", "C"]),
                    ],
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
            //
            //       If we *really* wanted to, we could store precomputed layers in
            //       the potential, but IMO it's just cleaner if we don't need to.
            fn find_layers<M>(&self, structure: &Structure<M>) -> Layers
            {
                ::rsp2_structure::find_layers(&structure, &V3([0, 0, 1]), 0.25)
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
            type Meta = Element;

            fn atom_types(&self, structure: &ElementStructure) -> Vec<AtomType>
            {
                self.find_layers(structure)
                    .by_atom().into_iter()
                    .map(|x| AtomType::from_index(x))
                    .collect()
            }

            fn init_info(&self, structure: &ElementStructure) -> InitInfo
            {
                let layers = match self.find_layers(structure).per_unit_cell() {
                    None => panic!("kolmogorov/crespi/z is only supported for layered materials"),
                    Some(layers) => layers,
                };

                let masses = vec![::common::element_mass(consts::CARBON); layers.len()];

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

// Dummy potentials for testing purposes
pub mod test {
    use super::*;
    use ::rsp2_structure::{Coords, CoordsKind};

    /// The test Potential `V = 0`.
    #[derive(Debug, Clone)]
    pub struct Zero;

    impl<Meta: Clone> PotentialBuilder<Meta> for Zero {
        fn threaded(&self, _threaded: bool) -> Box<PotentialBuilder<Meta>>
        { self.box_clone() }

        fn box_clone<'a>(&self) -> Box<PotentialBuilder<Meta>>
        { simple_box_clone(self) }

        fn _as_ref_dyn<'a>(&self) -> &PotentialBuilder<Meta>
        { self }

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

    /// A test Potential that creates a chain along the Z axis.
    #[derive(Debug, Clone)]
    pub struct Chainify;

    /// A test DiffFn that moves atoms to fixed positions.
    #[derive(Debug, Clone)]
    pub struct ConvergeTowards {
        target: Coords,
    }

    impl<Meta: Clone> PotentialBuilder<Meta> for ConvergeTowards {
        fn threaded(&self, _threaded: bool) -> Box<PotentialBuilder<Meta>>
        { self.box_clone() }

        fn box_clone(&self) -> Box<PotentialBuilder<Meta>>
        { simple_box_clone(self) }

        fn _as_ref_dyn(&self) -> &PotentialBuilder<Meta>
        { self }

        fn initialize_diff_fn(&self, _: Structure<Meta>) -> FailResult<Box<DiffFn<Meta>>>
        { Ok(Box::new(self.clone()) as Box<_>) }
    }

    impl<Meta: Clone> PotentialBuilder<Meta> for Chainify {
        fn threaded(&self, _threaded: bool) -> Box<PotentialBuilder<Meta>>
        { self.box_clone() }

        fn box_clone(&self) -> Box<PotentialBuilder<Meta>>
        { simple_box_clone(self) }

        fn _as_ref_dyn(&self) -> & (PotentialBuilder<Meta>)
        { self }

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

    impl ConvergeTowards {
        pub fn new(coords: Coords) -> Self
        { ConvergeTowards { target: coords.clone() } }
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
                ::util::zip_eq(&cur_fracs, target_fracs)
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
                ::util::zip_eq(&parts_by_coord, &derivs_by_coord)
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
            //let expected_carts = CoordsKind::Fracs(expected_fracs.clone().envee()).to_carts(&lattice);

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
            let final_fracs = final_carts.iter().map(|v| v / &lattice).collect::<Vec<_>>();
            assert_close!(final_fracs.unvee(), expected_fracs);
        }
    }
}
