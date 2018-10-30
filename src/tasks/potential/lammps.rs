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

//! All usage of the public API presented by `rsp2_lammps_wrap` is encapsulated here.

// The purpose of this module is to wrap `rsp2_lammps_wrap` with code specific to
// the potentials we care about using, and to support our metadata scheme.
//
// This is where we decide e.g. atom type assignments and `pair_coeff` commands.
// (which are decisions that `rsp2_lammps_wrap` has largely chosen to defer)

use super::{DynCloneDetail, PotentialBuilder, DiffFn, DispFn, CommonMeta, helper};
use ::FailResult;
#[allow(unused)] // rustc bug
use ::meta::{self, prelude::*};
#[allow(unused)] // rustc bug
use ::rsp2_soa_ops::{Part, Partition};
use ::rsp2_structure::{Coords, consts};
use ::rsp2_structure::layer::Layers;
use ::rsp2_tasks_config as cfg;
use ::rsp2_array_types::{V3};
use ::std::collections::BTreeMap;
use ::cmd::trial::TrialDir;

use ::rsp2_lammps_wrap::{InitInfo, AtomType, PairStyle, PairCoeff};
use ::rsp2_lammps_wrap::Builder as InnerBuilder;
use ::rsp2_lammps_wrap::Potential as LammpsPotential;
use ::rsp2_lammps_wrap::UpdateStyle;
use ::rsp2_lammps_wrap::LammpsOnDemand;
use ::rsp2_lammps_wrap::INSTANCE_LOCK;

const DEFAULT_KC_Z_CUTOFF: f64 = 14.0; // (Angstrom?)
const DEFAULT_KC_Z_MAX_LAYER_SEP: f64 = 4.5; // Angstrom

const DEFAULT_AIREBO_LJ_SIGMA:    f64 = 3.0; // (cutoff, x3.4 A)
const DEFAULT_AIREBO_LJ_ENABLED:      bool = true;
const DEFAULT_AIREBO_TORSION_ENABLED: bool = false;
const DEFAULT_AIREBO_OMP: bool = true;
const DEFAULT_REBO_OMP: bool = true;

/// A bundle of everything we need to initialize a Lammps API object.
///
/// It is nothing more than a bundle of configuration, and can be freely
/// sent across threads.
#[derive(Debug, Clone)]
pub(crate) struct Builder<P> {
    inner: InnerBuilder,
    pub potential: P,
    processor_axis_mask: [bool; 3],
}

fn assert_send_sync<S: Send + Sync>() {}

#[allow(unused)] // compile-time test
fn assert_lammps_builder_send_sync() {
    assert_send_sync::<Builder<()>>();
}

impl<P: Clone> Builder<P>
{
    pub(crate) fn new(
        trial_dir: Option<&TrialDir>,
        on_demand: Option<LammpsOnDemand>,
        threading: &cfg::Threading,
        update_style: &cfg::LammpsUpdateStyle,
        processor_axis_mask: &[bool; 3],
        potential: P,
    ) -> Self {
        let mut inner = InnerBuilder::new();
        if let Some(trial_dir) = trial_dir {
            inner.append_log(trial_dir.join("lammps.log"));
            inner.debug_dir(Some(trial_dir.as_path()));
        }
        inner.update_style(match *update_style {
            cfg::LammpsUpdateStyle::Safe => UpdateStyle::safe(),
            cfg::LammpsUpdateStyle::Run{ n, pre, post, sync_positions_every } => {
                warn_once!("lammps-update-style: run' is only for debugging purposes");
                UpdateStyle { n, pre, post, sync_positions_every }
            },
            cfg::LammpsUpdateStyle::Fast { sync_positions_every } => {
                warn_once!("'lammps-update-style: fast' is experimental");
                UpdateStyle::fast(sync_positions_every)
            },
        });
        if let Some(on_demand) = on_demand {
            inner.on_demand(on_demand);
        }
        if log_enabled!(target: "rsp2_tasks::special::lammps_data_trace", ::log::Level::Trace) {
            inner.data_trace_dir(Some({
                trial_dir.map(|t| t.as_path().to_owned())
                    .unwrap_or(::std::env::current_dir().unwrap())
                    .join("lammps-data-trace")
            }));
        }

        #[cfg(feature = "mpi")] {
            if *threading != cfg::Threading::Lammps && ::env::num_mpi_processes() != 1 {
                // We can't fool lammps into thinking it has fewer processes than it actually has.
                // At least, not without adding support for custom communicators to lammps-wrap.
                //
                // (but why bother? clearly, the user intended to use MPI!)
                panic!("Must use threading = \"lammps\" when using multiple MPI processes.");
            }
        }

        let processor_axis_mask = *processor_axis_mask;

        Builder { inner, potential, processor_axis_mask }
            .parallel(*threading == cfg::Threading::Lammps)
    }

    pub(crate) fn parallel(&self, parallel: bool) -> Self {
        use ::rsp2_array_utils::map_arr;

        let processors = match parallel {
            true => map_arr(self.processor_axis_mask, |flag| if flag { None } else { Some(1) }),
            false => [Some(1); 3],
        };

        let mut me = self.clone();
        me.inner.processors(processors);
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

        // (panic on lock already acquired; blocking could easily deadlock)
        let lock = INSTANCE_LOCK.try_lock().unwrap();

        let lammps_pot = Box::new(self.potential.clone()) as Box<LammpsPotential<Meta=P::Meta>>;
        let lmp = self.inner.build(lock, lammps_pot, coords.clone(), meta)?;
        Ok(Box::new(MyDiffFn::<M>(lmp)) as Box<_>)
    }

    pub(crate) fn lammps_disp_fn(&self, coords: &Coords, meta: M) -> FailResult<Box<DispFn>>
    {
        struct MyDispFn<Q: LammpsPotential>(::rsp2_lammps_wrap::DispFn<Q>);
        impl<Mm: Clone + 'static, Q: LammpsPotential<Meta=Mm>> DispFn for MyDispFn<Q> {
            fn compute_dense_force(&mut self, disp: (usize, V3)) -> FailResult<Vec<V3>>
            { self.0.compute_force_at_disp(disp) }

            fn compute_sparse_force_delta(&mut self, disp: (usize, V3)) -> FailResult<BTreeMap<usize, V3>>
            {
                let orig_force = self.0.equilibrium_force();
                helper::sparse_force_from_dense_deterministic(self, &orig_force, disp)
            }
        }

        // (panic on lock already acquired; blocking could easily deadlock)
        let lock = INSTANCE_LOCK.try_lock().unwrap();

        let lmp_disp_fn = self.inner.build_disp_fn(lock, self.potential.clone(), coords.clone(), meta)?;
        Ok(Box::new(MyDispFn(lmp_disp_fn)) as Box<_>)
    }
}

impl<M: Clone + 'static, P: Clone + LammpsPotential<Meta=M> + Send + Sync + 'static> PotentialBuilder<M> for Builder<P>
{
    fn parallel(&self, parallel: bool) -> Box<PotentialBuilder<M>>
    { Box::new(<Builder<_>>::parallel(self, parallel)) }

    fn initialize_diff_fn(&self, coords: &Coords, meta: M) -> FailResult<Box<DiffFn<M>>>
    { self.lammps_diff_fn(coords, meta) }

    fn initialize_disp_fn(&self, coords: &Coords, meta: M) -> FailResult<Box<DispFn>>
    { self.lammps_disp_fn(coords, meta) }

    fn _eco_mode(&self, cont: &mut dyn FnMut())
    { self.inner.eco_mode(cont) }
}

impl_dyn_clone_detail!{
    impl[M: Clone + 'static, P: Clone + LammpsPotential<Meta=M> + Send + Sync + 'static]
    DynCloneDetail<M> for Builder<P> { ... }
}

pub use self::overlay::Overlay;
mod overlay {
    use super::*;

    /// Helper type for composing the commands for a `pair_style hybrid/overlay` potential.
    ///
    /// It takes in pair commands for a bunch of potentials, and spits out what actually needs
    /// to be written to use them in a hybrid/overlay potential.
    #[derive(Debug, Clone)]
    pub struct Overlay {
        items: Vec<Item>,

        // Disambiguation indices to be used in `pair_coeff` commands.
        // (hybrid/overlay requires the pair_coeff commands to have additional index fields
        //  if and only if there are multiple styles with the same name).
        //
        // invariant: Correctly describes `items` at all times.
        indices: Vec<Option<u32>>,
    }

    #[derive(Debug, Clone)]
    pub struct Item {
        pub pair_style: PairStyle,
        pub pair_coeffs: Vec<PairCoeff>,
    }

    impl Item {
        fn is_same_style(&self, other: &Item) -> bool {
            self.pair_style.name() == other.pair_style.name()
        }
    }

    impl Overlay {
        /// Create a hybrid/overlay that implicitly begins with "pair_coeff * * none".
        pub fn new() -> Self {
            let mut me = Overlay {
                items: vec![],
                indices: vec![],
            };
            me.set_none(.., ..);
            me
        }

        /// Append a new potential.
        ///
        /// This cannot be used to insert `pair_coeff i j none` commands.
        /// For that, see `set_none`.
        pub fn push(&mut self, new_item: Item) -> &mut Self
        {
            { // scope &. NLL, come save us!!
                let name = new_item.pair_style.name();
                assert_ne!(name, "none", "attempted to push() pair_style none");
                assert_ne!(name, "hybrid", "attempted to nest hybrid potentials");
                assert_ne!(name, "hybrid/overlay", "attempted to nest hybrid potentials");
            }

            { // scope &mut. NLL, please visit soon!!
                // If there are any duplicates at all of a style, its first occurrence should have
                // index 1.
                let mut iter = zip_eq!(&mut self.items, &mut self.indices);
                if let Some((_, index)) = iter.find(|(x, _)| x.is_same_style(&new_item)) {
                    debug_assert!(*index == None || *index == Some(1));
                    *index = Some(1);
                };
            }

            let new_index = {
                zip_eq!(&self.items, &self.indices)
                    .rfind(|(x, _)| x.is_same_style(&new_item))
                    .map(|(_, index)| index.expect("") + 1)
            };
            self.indices.push(new_index);
            self.items.push(new_item);
            self
        }

        /// Inserts a `pair_coeff i j none` command to erase all previously added
        /// interactions between the given atom types.
        pub fn set_none<I, J>(&mut self, i: I, j: J) -> &mut Self
        where ::rsp2_lammps_wrap::AtomTypeRange: From<I> + From<J>
        {
            self.indices.push(None); // "pair_coeff i j none" never uses disambiguation indices
            self.items.push(Item {
                pair_style: PairStyle::named("none"),
                pair_coeffs: vec![PairCoeff::new(i, j)],
            });
            self
        }

        pub fn init_info(&self) -> (PairStyle, Vec<PairCoeff>) {
            // `pair_style hybrid/overlay` throws errors without at least one entry.
            if self.items.is_empty() {
                // (this is correct because we always start with `pair_coeff * * none`.
                // Were that not the case, then we might want to still fail (with a better message),
                // for consistency with how Lammps fails on pairs missing coefficients)
                return (PairStyle::named("none"), vec![]);
            }

            let pair_style = {
                let mut pair_style = PairStyle::named("hybrid/overlay");
                for item in &self.items {
                    if item.pair_style.name() != "none" {
                        pair_style = pair_style.arg(item.pair_style.name());
                    }
                    pair_style = pair_style.args(&item.pair_style.1);
                }
                pair_style
            };

            let pair_coeffs = {
                zip_eq!(&self.items, &self.indices)
                    .flat_map(|(item, index)| {
                        item.pair_coeffs.iter().cloned().map(move |PairCoeff(i, j, old_args)| {
                            let mut coeff = PairCoeff::new(i, j).arg(item.pair_style.name());
                            if let Some(index) = index {
                                coeff = coeff.arg(index);
                            }
                            coeff.args(old_args)
                        })
                    })
                    .collect()
            };

            (pair_style, pair_coeffs)
        }
    }

    #[test]
    fn smoke_test() {
        let ty = |n| AtomType::new(n);

        let (pair_style, pair_coeffs) = {
            Overlay::new()
                // include a potential with at least 3 repeats
                .push(Item {
                    pair_style: PairStyle::named("a"),
                    pair_coeffs: vec![PairCoeff::new(ty(1), ty(2)).arg("1.0")],
                })
                // include a potential with no repeats
                .push(Item {
                    pair_style: PairStyle::named("b").arg("3"),
                    pair_coeffs: vec![PairCoeff::new(ty(1), ty(3))],
                })
                // include a `pair_coeff none`
                .set_none(ty(1), ty(1))
                .push(Item {
                    pair_style: PairStyle::named("a"),
                    pair_coeffs: vec![PairCoeff::new(ty(2), ty(3)).arg("1.1")],
                })
                .push(Item {
                    pair_style: PairStyle::named("a"),
                    pair_coeffs: vec![PairCoeff::new(ty(3), ty(3)).arg("1.2")],
                })
                .init_info()
        };

        assert_eq!(
            pair_style,
            PairStyle::named("hybrid/overlay").args(vec!["a", "b", "3", "a", "a"]),
        );
        assert_eq!(
            pair_coeffs,
            vec![
                PairCoeff::new(.., ..).arg("none"),
                PairCoeff::new(ty(1), ty(2)).arg("a").arg("1").arg("1.0"),
                PairCoeff::new(ty(1), ty(3)).arg("b"),
                PairCoeff::new(ty(1), ty(1)).arg("none"),
                PairCoeff::new(ty(2), ty(3)).arg("a").arg("2").arg("1.1"),
                PairCoeff::new(ty(3), ty(3)).arg("a").arg("3").arg("1.2"),
            ],
        );
    }
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
            omp: bool,
        },
        /// Uses `pair_style rebo`.
        ///
        /// This is NOT equivalent to `pair_style airebo 0 0 0`, because it
        /// additionally sets one of the C-C interaction parameters to zero.
        Rebo {
            omp: bool,
        },
    }

    impl<'a> From<&'a cfg::PotentialAirebo> for Airebo {
        fn from(cfg: &'a cfg::PotentialAirebo) -> Self {
            let cfg::PotentialAirebo { lj_sigma, lj_enabled, torsion_enabled, omp } = *cfg;
            Airebo::Airebo {
                lj_sigma: lj_sigma.unwrap_or(DEFAULT_AIREBO_LJ_SIGMA),
                lj_enabled: lj_enabled.unwrap_or(DEFAULT_AIREBO_LJ_ENABLED),
                torsion_enabled: torsion_enabled.unwrap_or(DEFAULT_AIREBO_TORSION_ENABLED),
                omp: omp.unwrap_or(DEFAULT_AIREBO_OMP),
            }
        }
    }

    impl<'a> From<&'a cfg::PotentialRebo> for Airebo {
        fn from(cfg: &'a cfg::PotentialRebo) -> Self {
            let cfg::PotentialRebo { omp } = *cfg;
            Airebo::Rebo {
                omp: omp.unwrap_or(DEFAULT_REBO_OMP),
            }
        }
    }

    impl LammpsPotential for Airebo {
        type Meta = CommonMeta;

        fn atom_types(&self, _: &Coords, meta: &CommonMeta) -> Vec<AtomType>
        {
            let elements: meta::SiteElements = meta.pick();
            elements.iter().map(|elem| match elem.symbol() {
                "H" => AtomType::new(1),
                "C" => AtomType::new(2),
                sym => panic!("Unexpected element in Airebo: {}", sym),
            }).collect()
        }

        fn init_info(&self, _: &Coords, meta: &CommonMeta) -> InitInfo {
            let elements: meta::SiteElements = meta.pick();
            let masses: meta::SiteMasses = meta.pick();

            let only_unique_mass = |target_elem| {
                let iter = {
                    zip_eq!(&elements[..], &masses[..])
                        .filter(|&(&elem, _)| elem == target_elem)
                        .map(|(_, &mass)| mass)
                };

                match only_unique_value(iter) {
                    OnlyUniqueResult::Ok(meta::Mass(mass)) => mass,
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
            let pair_style = match *self {
                Airebo::Airebo { lj_sigma, lj_enabled, torsion_enabled, omp } => {
                    let style = match omp {
                        true => "airebo/omp",
                        false => "airebo"
                    };
                    PairStyle::named(style)
                        .arg(lj_sigma)
                        .arg(boole(lj_enabled))
                        .arg(boole(torsion_enabled))
                },
                Airebo::Rebo { omp } => {
                    let style = match omp {
                        true => "rebo/omp",
                        false => "rebo"
                    };
                    PairStyle::named(style)
                },
            };
            let pair_coeffs = vec![
                PairCoeff::new(.., ..).args(&["CH.airebo", "H", "C"]),
            ];
            InitInfo { masses, pair_style, pair_coeffs }
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
        rebo: bool,
        cutoff: f64,
        cutoff_interval: Option<f64>,
        // max distance between interacting layers.
        // used to identify vacuum separation.
        max_layer_sep: f64,
    }

    #[derive(Debug, Clone, Copy)]
    enum GapKind { Interacting, Vacuum }

    impl<'a> From<&'a cfg::PotentialKolmogorovCrespiZ> for KolmogorovCrespiZ {
        fn from(cfg: &'a cfg::PotentialKolmogorovCrespiZ) -> Self {
            let cfg::PotentialKolmogorovCrespiZ {
                rebo, cutoff, max_layer_sep, cutoff_interval,
            } = *cfg;
            KolmogorovCrespiZ {
                rebo,
                cutoff: cutoff.unwrap_or(DEFAULT_KC_Z_CUTOFF),
                cutoff_interval,
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
            let elements: meta::SiteElements = meta.pick();
            let masses: meta::SiteMasses = meta.pick();

            let layers = match self.find_layers(coords).per_unit_cell() {
                None => panic!("kolmogorov/crespi/z is only supported for layered materials"),
                Some(layers) => layers,
            };

            // stupid limitation of current design.  To fix it we really just need
            // to add an extra atom type.
            assert!(
                elements.iter().cloned().all(|x| x == consts::CARBON),
                "KCZ does not yet support the inclusion of non-carbon elements"
            );

            let masses: Vec<f64> = {
                let layer_part = Part::from_ord_keys(layers.by_atom());
                let m = masses.to_vec()
                    .into_unlabeled_partitions(&layer_part)
                    .map(|layer_masses| match only_unique_value(layer_masses) {
                        OnlyUniqueResult::Ok(meta::Mass(x)) => x,
                        OnlyUniqueResult::Conflict(_, _) => {
                            panic!("KCZ does not support multiple masses within a layer");
                        }
                        OnlyUniqueResult::NoValues => unreachable!(),
                    })
                    .collect();
                m // without this binding it thinks the borrow of `layer_part` outlives the block
            };

            let interacting_pairs: Vec<_> = {
                let gaps: Vec<_> = layers.gaps.iter().map(|&x| self.classify_gap(x)).collect();

                determine_pair_coeff_pairs(&gaps)
                    .unwrap_or_else(|e| panic!("{}", e))
            };

            let mut overlay = Overlay::new();

            if self.rebo {
                overlay.push(overlay::Item {
                    pair_style: PairStyle::named("rebo"),
                    pair_coeffs: vec![{
                        PairCoeff::new(.., ..)
                            .arg("CH.airebo")
                            .args(&vec!["C"; layers.len()])
                    }],
                });
            }

            for &(i, j) in &interacting_pairs {
                overlay.push(overlay::Item {
                    pair_style: {
                        let mut cmd = PairStyle::named("kolmogorov/crespi/z").arg(self.cutoff);
                        if let Some(cutoff_interval) = self.cutoff_interval {
                            cmd = cmd.arg(cutoff_interval);
                        }
                        cmd
                    },
                    pair_coeffs: vec![{
                        PairCoeff::new(i, j)
                            .arg("CC.KC")
                            .args(vec!["C"; layers.len()])
                    }],
                });
            }

            let (pair_style, pair_coeffs) = overlay.init_info();

            InitInfo { masses, pair_style, pair_coeffs }
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
