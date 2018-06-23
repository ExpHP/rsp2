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
use ::meta::{Mass, Element};
#[allow(unused)] // rustc bug
use ::meta::prelude::*;
#[allow(unused)] // rustc bug
use ::rsp2_soa_ops::{Part, Partition};
use ::rsp2_structure::{Coords, consts};
use ::rsp2_structure::layer::Layers;
use ::rsp2_tasks_config as cfg;
use ::rsp2_array_types::{V3};
use ::std::rc::Rc;
use ::std::collections::BTreeMap;
use ::cmd::trial::TrialDir;

use ::rsp2_lammps_wrap::{InitInfo, AtomType, PairCommand};
use ::rsp2_lammps_wrap::Builder as InnerBuilder;
use ::rsp2_lammps_wrap::Potential as LammpsPotential;
use ::rsp2_lammps_wrap::UpdateStyle;

const DEFAULT_KC_Z_CUTOFF: f64 = 14.0; // (Angstrom?)
const DEFAULT_KC_Z_MAX_LAYER_SEP: f64 = 4.5; // Angstrom

const DEFAULT_AIREBO_LJ_SIGMA:    f64 = 3.0; // (cutoff, x3.4 A)
const DEFAULT_AIREBO_LJ_ENABLED:      bool = true;
const DEFAULT_AIREBO_TORSION_ENABLED: bool = false;

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
        trial_dir: Option<&TrialDir>,
        threading: &cfg::Threading,
        update_style: &cfg::LammpsUpdateStyle,
        potential: P,
    ) -> Self {
        let mut inner = InnerBuilder::new();
        if let Some(trial_dir) = trial_dir {
            inner.append_log(trial_dir.join("lammps.log"));
        }
        inner.threaded(*threading == cfg::Threading::Lammps);
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
        // XXX
//        inner.data_trace_dir(Some(trial_dir.join("lammps-data-trace")));

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

        let lmp_disp_fn = self.inner.build_disp_fn(self.potential.clone(), coords.clone(), meta)?;
        Ok(Box::new(MyDispFn(lmp_disp_fn)) as Box<_>)
    }
}

impl<M: Clone + 'static, P: Clone + LammpsPotential<Meta=M> + Send + Sync + 'static> PotentialBuilder<M> for Builder<P>
{
    fn threaded(&self, threaded: bool) -> Box<PotentialBuilder<M>>
    { Box::new(<Builder<_>>::threaded(self, threaded)) }

    fn initialize_diff_fn(&self, coords: &Coords, meta: M) -> FailResult<Box<DiffFn<M>>>
    { self.lammps_diff_fn(coords, meta) }

    fn initialize_disp_fn(&self, coords: &Coords, meta: M) -> FailResult<Box<DispFn>>
    { self.lammps_disp_fn(coords, meta) }
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
        rebo: bool,
        cutoff: f64,
        // max distance between interacting layers.
        // used to identify vacuum separation.
        max_layer_sep: f64,
    }

    #[derive(Debug, Clone, Copy)]
    enum GapKind { Interacting, Vacuum }

    impl<'a> From<&'a cfg::PotentialKolmogorovCrespiZ> for KolmogorovCrespiZ {
        fn from(cfg: &'a cfg::PotentialKolmogorovCrespiZ) -> Self {
            let cfg::PotentialKolmogorovCrespiZ { rebo, cutoff, max_layer_sep } = *cfg;
            KolmogorovCrespiZ {
                rebo,
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
            pair_commands.push((|| { // iife
                let mut cmd = PairCommand::pair_style("hybrid/overlay");
                let mut has_coeffs = false;
                if self.rebo {
                    cmd = cmd.arg("rebo");
                    has_coeffs = true;
                }
                for _ in &interacting_pairs {
                    cmd = cmd.arg("kolmogorov/crespi/z").arg(self.cutoff);
                    has_coeffs = true;
                }

                // hybrid/overlay without at least one potential is invalid.
                // (hey, I don't make the rules)
                // FIXME: This would be cleaner with a type dedicated to handle hybrid/overlay
                if !has_coeffs {
                    // NOTE: returns to iife inside `push(...)`
                    return PairCommand::pair_style("none");
                }

                cmd
            })());

            if self.rebo {
                pair_commands.push({
                    PairCommand::pair_coeff(.., ..)
                        .args(&["rebo", "CH.airebo"])
                        .args(&vec!["C"; layers.len()])
                });
            }

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
