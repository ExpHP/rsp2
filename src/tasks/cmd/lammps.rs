// The purpose of this module is to wrap `rsp2_lammps_wrap` with code specific to
// the potentials we care about using.
//
// This is where we decide e.g. atom type assignments and `pair_coeff` commands.
// (which are decisions that `rsp2_lammps_wrap` has largely chosen to defer)

use ::Result;
use ::rsp2_lammps_wrap::{InitInfo, Potential, AtomType, PairCommand};
use ::rsp2_lammps_wrap::Builder as InnerBuilder;
use ::rsp2_structure::{Structure, Element, ElementStructure, Layers};
use ::config as cfg;

const REBO_MASS_HYDROGEN: f64 =  1.00;
const REBO_MASS_CARBON:   f64 = 12.01;

const DEFAULT_KOLMOGOROV_CRESPI_Z_CUTOFF: f64 = 20.0; // (Angstrom?)

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

/// Uses `pair_style kolmogorov/crespi/z`.
#[derive(Debug, Clone, Default)]
pub struct KolmogorovCrespiZ {
    cutoff: f64,
}

impl<'a> From<&'a cfg::PotentialKolmogorovCrespiZ> for KolmogorovCrespiZ {
    fn from(cfg: &'a cfg::PotentialKolmogorovCrespiZ) -> Self {
        let cfg::PotentialKolmogorovCrespiZ { cutoff } = *cfg;
        KolmogorovCrespiZ {
            cutoff: cutoff.unwrap_or(DEFAULT_KOLMOGOROV_CRESPI_Z_CUTOFF),
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
}

impl Potential for KolmogorovCrespiZ {
    type Meta = Element;

    // TODO FIXME I am aware that, as written, this is incorrect for
    //            bulk graphite with 3+ layers per unit cell, because
    //            it will fail to enable interactions between the
    //            1st and 3rd layers.
    //            For multilayer graphite it is also sensitive to the
    //            centering of the structure along the Z axis.
    //            (z=0 must lie within the vacuum gap)
    //
    //            The only fix is to just do it properly, and enable/disable
    //            interactions between each pair of layers based on their separation.
    //            (that separation is a pain in the ass to compute, but is naturally
    //             found as a part of the algorithm in rsp2_structure; it should be
    //             added to the output in some way)
    //
    //            Unfortunately, for now we can only silently handle these cases incorrectly.
    //            (the code to detect the problems would be equally as complicated as the
    // TODO FIXME  code to do it correctly)

    fn atom_types(&self, structure: &ElementStructure) -> Vec<AtomType>
    {
        self.find_layers(structure)
            .by_atom().into_iter()
            .map(|x| AtomType::new((x + 1) as _)).collect()
    }

    fn init_info(&self, structure: &Structure<Self::Meta>) -> InitInfo
    {
        // TODO: Judging from the documentation, if I want more than 2 layers,
        //       the commands will need to look something like this:
        //
        //           # Need to specify the potential once for each time it is used.
        //           pair_style hybrid/overlay rebo kolmogorov/crespi/z 20 &
        //                             kolmogorov/crespi/z 20 kolmogorov/crespi/z 20
        //
        //           # No I,J pairs are allowed to be missing.
        //           # (but the docs don't specify what happens if they ARE missing...)
        //           pair_coeff 1 3 none
        //           pair_coeff 1 4 none
        //           pair_coeff 2 4 none
        //
        //           # (I'm hoping I can get away with this instead, but certain language
        //           #  leads me to worry that the implementation of 'none' in 'hybrid'
        //           #  styles is such a needlessly bizarre hack that this wouldn't
        //           #  actually work)
        //           pair_coeff * * none
        //
        //           # If (and only if?) there is more than one copy of the potential,
        //           # you need another numeric argument after its name in pair_coeff
        //           # to disambiguate it.             (v-- here)
        //           pair_coeff * * rebo                  CH.airebo  C C
        //           pair_coeff 1 2 kolmogorov/crespi/z 1 CC.KC      C C
        //           pair_coeff 2 3 kolmogorov/crespi/z 2 CC.KC      C C
        //           pair_coeff 3 4 kolmogorov/crespi/z 3 CC.KC      C C
        //
        //       Or something else that is similarly horrible.
        //       I have too many questions right now that can only be answered by
        // TODO: actually trying them out on Lammps, so for now we just bail out.
        let layers = match self.find_layers(structure).per_unit_cell() {
            None => panic!("kolmogorov/crespi/z is only supported for layered materials"),
            Some(layers) => layers,
        };
        assert_eq!(
            layers.len(), 2,
            "Sorry, kolmogorov/crespi/z is not yet supported for arbitrary # layers.",
        );

        let masses = vec![REBO_MASS_CARBON; layers.len() as usize];

        let mut pair_commands = vec![
            PairCommand::pair_style("hybrid/overlay")
                .arg("rebo")
                .arg("kolmogorov/crespi/z").arg(self.cutoff),
            PairCommand::pair_coeff(.., ..)
                .args(&["rebo", "CH.airebo", "C", "C"]),
        ];
        pair_commands.extend((0..layers.len() - 1).map(|i| {
            let first = AtomType::new((i + 1) as _);
            let second = AtomType::new((i + 2) as _);
            PairCommand::pair_coeff(first, second)
                .args(&["kolmogorov/crespi/z", "CC.KC", "C", "C"])
        }));

        InitInfo { masses, pair_commands }
    }
}
