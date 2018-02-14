// The purpose of this module is to wrap `rsp2_lammps_wrap` with code specific to
// the potentials we care about using.
//
// This is where we decide e.g. atom type assignments and `pair_coeff` commands.
// (which are decisions that `rsp2_lammps_wrap` has largely chosen to defer)

use ::Result;
use ::rsp2_lammps_wrap::{InitInfo, Potential, AtomType, PairCommand};
use ::rsp2_lammps_wrap::Builder as InnerBuilder;
use ::rsp2_structure::{Structure, Element, ElementStructure, Layers};
use ::rsp2_tasks_config as cfg;

const REBO_MASS_HYDROGEN: f64 =  1.00;
const REBO_MASS_CARBON:   f64 = 12.01;

const DEFAULT_KC_Z_CUTOFF: f64 = 20.0; // (Angstrom?)
const DEFAULT_KC_Z_MAX_LAYER_SEP: f64 = 4.5; // Angstrom

const DEFAULT_AIREBO_LJ_STRENGTH: f64 = 1.0;
const DEFAULT_AIREBO_LJ_SIGMA:    f64 = 3.0; // (cutoff, x3.4 A)
const DEFAULT_AIREBO_LJ_ENABLED:      bool = true;
const DEFAULT_AIREBO_TORSION_ENABLED: bool = false;

pub type DynPotential = Box<Potential<Meta=Element>>;
pub type Lammps = ::rsp2_lammps_wrap::Lammps<DynPotential>;

// A bundle of everything we need to initialize a Lammps API object.
//
// It is nothing more than a bundle of configuration, and can be freely
// sent across threads.
#[derive(Debug, Clone)]
pub(crate) struct LammpsBuilder {
    pub(crate) builder: InnerBuilder,
    pub(crate) potential: cfg::PotentialKind,
}

impl LammpsBuilder {
    // laziest route to easily adapt code that used to receive
    // an InnerBuilder directly (that's what LammpsBuilder USED to be)
    pub(crate) fn with_modified_inner<F>(&self, mut f: F) -> Self
    where F: FnMut(&mut InnerBuilder) -> &mut InnerBuilder,
    {
        let mut out = self.clone();
        let _ = f(&mut out.builder);
        out
    }
}

fn assert_send_sync<S: Send + Sync>() {}

#[allow(unused)]
fn assert_lammps_builder_send_sync() {
    assert_send_sync::<LammpsBuilder>();
}

impl LammpsBuilder {
    pub(crate) fn new(
        threading: &cfg::Threading,
        potential: &cfg::PotentialKind,
    ) -> LammpsBuilder
    {
        let mut builder = InnerBuilder::new();
        builder.append_log("lammps.log");
        builder.threaded(*threading == cfg::Threading::Lammps);

        let potential = potential.clone();

        LammpsBuilder { builder, potential }
    }

    pub(crate) fn build(&self, structure: ElementStructure) -> Result<Lammps>
    {Ok({
        let potential: DynPotential = match self.potential {
            cfg::PotentialKind::Airebo(ref cfg) => {
                Box::new(Airebo::from(cfg))
            },
            cfg::PotentialKind::KolmogorovCrespiZ(ref cfg) => {
                Box::new(KolmogorovCrespiZ::from(cfg))
            },
        };
        self.builder.build(potential, structure)?
    })}
}

pub use self::airebo::Airebo;
mod airebo {
    use super::*;

    /// Uses `pair_style airebo`.
    #[derive(Debug, Clone)]
    pub struct Airebo {
        lj_sigma: f64,
        lj_strength: f64,
        lj_enabled: bool,
        torsion_enabled: bool,
    }

    impl<'a> From<&'a cfg::PotentialAirebo> for Airebo {
        fn from(cfg: &'a cfg::PotentialAirebo) -> Self {
            let cfg::PotentialAirebo {
                lj_sigma, lj_strength, lj_enabled, torsion_enabled,
            } = *cfg;

            Airebo {
                lj_sigma: lj_sigma.unwrap_or(DEFAULT_AIREBO_LJ_SIGMA),
                lj_strength: lj_strength.unwrap_or(DEFAULT_AIREBO_LJ_STRENGTH),
                lj_enabled: lj_enabled.unwrap_or(DEFAULT_AIREBO_LJ_ENABLED),
                torsion_enabled: torsion_enabled.unwrap_or(DEFAULT_AIREBO_TORSION_ENABLED),
            }
        }
    }

    impl Potential for Airebo {
        type Meta = Element;

        fn atom_types(&self, structure: &ElementStructure) -> Vec<AtomType>
        { structure.metadata().iter().map(|elem| match elem.symbol() {
            "H" => AtomType::new(1),
            "C" => AtomType::new(2),
            sym => panic!("Unexpected element in Airebo: {}", sym),
        }).collect() }

        fn init_info(&self, _: &Structure<Self::Meta>) -> InitInfo
        {
            InitInfo {
                masses: vec![REBO_MASS_HYDROGEN, REBO_MASS_CARBON],
                pair_commands: vec![
                    PairCommand::pair_style("airebo/omp")
                        .arg(self.lj_sigma)
                        .arg(boole(self.lj_enabled))
                        .arg(boole(self.torsion_enabled))
                        ,
                    PairCommand::pair_coeff(.., ..).args(&["CH.airebo", "H", "C"]),
                    // colin's lj scaling HACK
                    PairCommand::pair_coeff(.., ..).arg("lj/scale").arg(self.lj_strength),
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
            ::rsp2_structure::find_layers(&structure, &[0, 0, 1], 0.25)
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

    impl Potential for KolmogorovCrespiZ {
        type Meta = Element;

        fn atom_types(&self, structure: &ElementStructure) -> Vec<AtomType>
        {
            self.find_layers(structure)
                .by_atom().into_iter()
                .map(|x| AtomType::from_index(x))
                .collect()
        }

        fn init_info(&self, structure: &Structure<Self::Meta>) -> InitInfo
        {
            let layers = match self.find_layers(structure).per_unit_cell() {
                None => panic!("kolmogorov/crespi/z is only supported for layered materials"),
                Some(layers) => layers,
            };

            let masses = vec![REBO_MASS_CARBON; layers.len()];

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
    fn determine_pair_coeff_pairs(gaps: &[GapKind]) -> Result<Vec<(AtomType, AtomType)>>
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