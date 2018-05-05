// The purpose of this module is to wrap `rsp2_lammps_wrap` with code specific to
// the potentials we care about using, and to support ElementStructure.
//
// This is where we decide e.g. atom type assignments and `pair_coeff` commands.
// (which are decisions that `rsp2_lammps_wrap` has largely chosen to defer)

use ::FailResult;
use ::rsp2_structure::{Layers, Element, Structure, ElementStructure, consts};
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

//pub type Lammps = ::rsp2_lammps_wrap::Lammps<DynPotential>;

/// Trait alias for a function producing flat potential and gradient,
/// for compatibility with `rsp2_minimize`.
pub trait FlatDiffFn: FnMut(&[f64]) -> FailResult<(f64, Vec<f64>)> {}

impl<F> FlatDiffFn for F where F: FnMut(&[f64]) -> FailResult<(f64, Vec<f64>)> {}

// Type aliases for the trait object types, to work around #23856
pub type DynFlatDiffFn<'a> = FlatDiffFn<Output=FailResult<(f64, Vec<f64>)>> + 'a;

pub(crate) use self::lammps::Builder as PotentialBuilder;

/// This is `FnMut(ElementStructure) -> FailResult<(f64, Vec<V3>)>` with convenience methods.
pub trait DiffFn {
    /// Compute the value and gradient.
    fn compute(&mut self, structure: &ElementStructure) -> FailResult<(f64, Vec<V3>)>;

    /// Convenience method to compute the potential.
    fn compute_value(&mut self, structure: &ElementStructure) -> FailResult<f64>
    { Ok(self.compute(structure)?.0) }

    /// Convenience method to compute the gradient.
    fn compute_grad(&mut self, structure: &ElementStructure) -> FailResult<Vec<V3>>
    { Ok(self.compute(structure)?.1) }

    /// Convenience method to compute the force.
    fn compute_force(&mut self, structure: &ElementStructure) -> FailResult<Vec<V3>>
    {
        let mut force = self.compute_grad(structure)?;
        for v in &mut force { *v = -*v; }
        Ok(force)
    }
}

/// All usage of the public API presented by `rsp2_lammps_wrap` is encapsulated here.
mod lammps {
    use super::*;

    use ::rsp2_lammps_wrap::{InitInfo, AtomType, PairCommand};
    use ::rsp2_lammps_wrap::Builder as InnerBuilder;
    use ::rsp2_lammps_wrap::Potential as LammpsPotential;

    pub type DynLammpsPotential = Box<LammpsPotential<Meta=Element>>;

    /// A bundle of everything we need to initialize a Lammps API object.
    ///
    /// It is nothing more than a bundle of configuration, and can be freely
    /// sent across threads.
    #[derive(Debug, Clone)]
    pub(crate) struct Builder {
        inner: InnerBuilder,
        potential: cfg::PotentialKind,
    }

    fn assert_send_sync<S: Send + Sync>() {}

    #[allow(unused)] // compile-time test
    fn assert_lammps_builder_send_sync() {
        assert_send_sync::<Builder>();
    }

    impl Builder {
        pub(crate) fn new(
            threading: &cfg::Threading,
            potential: &cfg::PotentialKind,
        ) -> Builder
        {
            let mut inner = InnerBuilder::new();
            inner.append_log("lammps.log");
            inner.threaded(*threading == cfg::Threading::Lammps);

            let potential = potential.clone();

            Builder { inner, potential }
        }

        pub(crate) fn threaded(&self, threaded: bool) -> Self
        { let mut me = self.clone(); me.inner.threaded(threaded); me }

        /// Initialize Lammps to make a DiffFn.
        ///
        /// This keeps the Lammps instance between calls to save time.
        ///
        /// Some data may be pre-allocated or precomputed based on the input structure,
        /// so the resulting DiffFn may not support arbitrary structures as input.
        pub(crate) fn diff_fn(&self, structure: ElementStructure) -> FailResult<Box<DiffFn>>
        {
            // a DiffFn 'lambda' whose type will be erased
            struct MyDiffFn(::rsp2_lammps_wrap::Lammps<DynLammpsPotential>);
            impl DiffFn for MyDiffFn {
                fn compute(&mut self, structure: &ElementStructure) -> FailResult<(f64, Vec<V3>)> {
                    self.0.set_structure(structure.clone())?;
                    Ok(self.0.compute()?)
                }
            }
            let lmp = self.build_lammps(structure)?;
            Ok(Box::new(MyDiffFn(lmp)) as Box<_>)
        }


        // FIXME: Not sure how to accomodate other potentials yet here.
        //        I tried moving it to the DiffFn trait, but there is a bit of
        //        nuance in that a flat_diff_fn produced by Builder should be 'static,
        //        while one produced by a DiffFn should borrow `from &mut self`.
        //
        /// Convenience method to get a function suitable for `rsp2_minimize`.
        ///
        /// The structure given to this is used to supply the lattice and element data.
        /// Also, some other data may be precomputed from it.
        ///
        /// Because Boxes don't implement `Fn` traits for technical reasons,
        /// you will likely need to write `&mut *pot.flat_diff_fn()` in order to get
        /// a `&mut DynFlatDiffFn`.
        pub(crate) fn flat_diff_fn(&self, structure: ElementStructure) -> FailResult<Box<DynFlatDiffFn<'static>>>
        {Ok({
            let mut lmp = self.build_lammps(structure)?;
            Box::new(move |pos: &[f64]| Ok({
                lmp.set_carts(pos.nest())?;
                lmp.compute().map(|(v, g)| (v, g.unvee().flat().to_vec()))?
            }))
        })}

        fn build_lammps(
            &self,
            structure: ElementStructure,
        ) -> FailResult<::rsp2_lammps_wrap::Lammps<DynLammpsPotential>>
        {
            let potential: DynLammpsPotential = match self.potential {
                cfg::PotentialKind::Airebo(ref cfg) => {
                    Box::new(Airebo::from(cfg))
                },
                cfg::PotentialKind::KolmogorovCrespiZ(ref cfg) => {
                    Box::new(KolmogorovCrespiZ::from(cfg))
                },
            };
            self.inner.build(potential, structure)
                .map_err(Into::into)
        }

        pub(crate) fn one_off(&self) -> OneOff { OneOff(self) }
    }

    /// One-off computations for convenience.  Lammps will be initialized from
    /// stratch, and dropped at the end.
    ///
    /// Usage: `pot.one_off().compute(&structure)`
    ///
    /// These are provided because otherwise, you end up needing to write stuff
    /// like `pot.diff_fn(structure.clone()).compute(&structure)`, which is both
    /// confusing and awkward.
    pub struct OneOff<'a>(&'a Builder);
    impl<'a> DiffFn for OneOff<'a> {
        fn compute(&mut self, structure: &ElementStructure) -> FailResult<(f64, Vec<V3>)>
        { Ok(self.0.build_lammps(structure.clone())?.compute()?) }
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