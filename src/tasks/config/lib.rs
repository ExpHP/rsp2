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

#![allow(non_snake_case)]

//! Crate where serde_yaml code for the 'tasks' crate is monomorphized,
//! because this is a huge compile time sink.
//!
//! The functions here also make use of serde_ignored to catch typos in the config.

// NOTE: Please make sure to use the YamlRead trait!
//       DO NOT USE serde_yaml::from_{reader,value,etc.} OUTSIDE THIS CRATE
//       or else you defeat the entire reason for its existence.

// (NOTE: I can't enforce this through the type system without completely destroying
//        the ergonomics of these types. Just Ctrl+Shift+F the workspace for "serde_yaml"
//        if compile times seem suspiciously off...)

#[macro_use]
extern crate serde_derive;
extern crate serde_yaml;

extern crate serde;
extern crate serde_ignored;

extern crate failure;

extern crate rsp2_minimize;

#[macro_use]
extern crate log;

use ::std::io::Read;
use ::std::collections::HashMap;
use ::failure::Error;
pub use ::rsp2_minimize::acgsd::Settings as Acgsd;

/// Provides an alternative to serde_yaml::from_reader where all of the
/// expensive codegen has already been performed in this crate.
pub trait YamlRead: for <'de> ::serde::Deserialize<'de> {
    fn from_reader(mut r: impl Read) -> Result<Self, Error>
    { YamlRead::from_dyn_reader(&mut r) }

    fn from_dyn_reader(r: &mut Read) -> Result<Self, Error> {
        // serde_ignored needs a Deserializer.
        // unlike serde_json, serde_yaml doesn't seem to expose a Deserializer that is
        // directly constructable from a Read... but it does impl Deserialize for Value.
        //
        // However, on top of that, deserializing a Value through serde_ignored makes
        // one lose all of the detail from the error messages. So...
        //
        // First, parse to a form that we can read from multiple times.
        let mut s = String::new();
        r.read_to_string(&mut s)?;

        // try deserializing from Value, printing warnings on unused keys.
        // (if value_from_dyn_reader fails, that error should be fine)
        let value = value_from_str(&s)?;

        match Self::__serde_ignored__from_value(value) {
            Ok(out) => Ok(out),
            Err(_) => {
                // That error message was surely garbage. Let's re-parse again
                // from the string, without serde_ignored:
                Self::__serde_yaml__from_str(&s)?;
                unreachable!();
            }
        }
    }

    // trait-provided function definitions seem to be lazily monomorphized, so we
    // must put the meat of what we need monomorphized directly into the impls
    fn __serde_ignored__from_value(value: ::serde_yaml::Value) -> Result<Self, Error>;
    fn __serde_yaml__from_str(s: &str) -> Result<Self, Error>;
}

macro_rules! derive_yaml_read {
    ($Type:ty) => {
        impl YamlRead for $Type {
            fn __serde_ignored__from_value(value: ::serde_yaml::Value) -> Result<$Type, Error> {
                ::serde_ignored::deserialize(
                    value,
                    |path| warn!("Unused config item (possible typo?): {}", path),
                ).map_err(Into::into)
            }

            fn __serde_yaml__from_str(s: &str) -> Result<$Type, Error> {
                ::serde_yaml::from_str(s)
                    .map_err(Into::into)
            }
        }
    };
}

derive_yaml_read!{::serde_yaml::Value}

// (this also exists solely for codegen reasons)
fn value_from_str(r: &str) -> Result<::serde_yaml::Value, Error>
{ ::serde_yaml::from_str(r).map_err(Into::into) }

#[derive(Serialize, Deserialize)]
#[derive(Debug, Clone, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub struct Settings {
    #[serde(default)]
    pub threading: Threading,

    pub potential: PotentialKind,

    // (FIXME: weird name)
    pub scale_ranges: ScaleRanges,

    #[serde(default)]
    pub acoustic_search: AcousticSearch,

    pub cg: Acgsd,

    pub phonons: Phonons,

    pub ev_chase: EigenvectorChase,

    /// `None` disables layer search.
    /// (layer_search is also ignored if layers.yaml is provided)
    #[serde(default)]
    pub layer_search: Option<LayerSearch>,

    /// `None` disables bond graph.
    #[serde(default)]
    pub bond_radius: Option<f64>,

    // FIXME move
    pub layer_gamma_threshold: f64,

    #[serde(default)]
    pub masses: Option<Masses>,

    #[serde(default)]
    pub ev_loop: EvLoop,
}
derive_yaml_read!{Settings}

#[derive(Serialize, Deserialize)]
#[derive(Debug, Clone, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub struct ScaleRanges {
    pub scalables: Vec<Scalable>,
    /// How many times to repeat the process of relaxing all parameters.
    ///
    /// This may yield better results if one of the parameters relaxed
    /// earlier in the sequence impacts one of the ones relaxed earlier.
    #[serde(default="_scale_ranges__repeat_count")]
    pub repeat_count: u32,

    /// Warn if the optimized value of a parameter falls within this amount of
    /// the edge of the search window (relative to the search window size),
    /// which likely indicates that the search window was not big enough.
    #[serde(default="_scale_ranges__warn_threshold")]
    pub warn_threshold: Option<f64>,

    /// Panic on violations of `warn_threshold`.
    #[serde(default="_scale_ranges__fail")]
    pub fail: bool,
}
fn _scale_ranges__repeat_count() -> u32 { 1 }
fn _scale_ranges__warn_threshold() -> Option<f64> { Some(0.01) }
fn _scale_ranges__fail() -> bool { false }

#[derive(Debug, Clone, PartialEq)]
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Scalable {
    /// Uniformly scale one or more lattice vectors.
    #[serde(rename = "parameter")]
    #[serde(rename_all = "kebab-case")]
    Param {
        axis_mask: [MaskBit; 3],
        #[serde(flatten)]
        range: ScalableRange,
    },

    /// Optimize a single value shared by all layer separations.
    ///
    /// Under certain conditions, the optimum separation IS identical for
    /// all layers (e.g. generated structures where all pairs of layers
    /// look similar, and where the potential only affects adjacent layers).
    ///
    /// There are also conditions where the separation obtained from this method
    /// is "good enough" that CG can be trusted to take care of the rest.
    #[serde(rename_all = "kebab-case")]
    UniformLayerSep {
        #[serde(flatten)]
        range: ScalableRange,
    },

    /// Optimize each layer separation individually. Can be costly.
    #[serde(rename_all = "kebab-case")]
    LayerSeps {
        #[serde(flatten)]
        range: ScalableRange,
    },
}

// a bool that serializes as an integer
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct MaskBit(pub bool);

impl serde::Serialize for MaskBit {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where S: serde::Serializer,
    { (self.0 as i32).serialize(serializer) }
}

impl<'de> serde::Deserialize<'de> for MaskBit {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where D: serde::Deserializer<'de>,
    {
        use serde::de::Unexpected;
        use serde::de::Error;
        match serde::Deserialize::deserialize(deserializer)? {
            0i64 => Ok(MaskBit(false)),
            1i64 => Ok(MaskBit(true)),
            n => Err(Error::invalid_value(Unexpected::Signed(n), &"a mask bit equal to 0 or 1")),
        }
    }
}

#[derive(Serialize, Deserialize)]
#[derive(Debug, Clone, PartialEq)]
#[serde(untagged)]
pub enum ScalableRange {
    // NOTE: This enum gets `serde(flatten)`ed into its container. Beware field-name clashes.
    #[serde(rename_all = "kebab-case")]
    Search {
        range: (f64, f64),
        /// A "reasonable value" that might be used while another
        ///  parameter is optimized.
        #[serde(default)]
        guess: Option<f64>,
    },
    #[serde(rename_all = "kebab-case")]
    Exact {
        value: f64,
    },
}

#[derive(Serialize, Deserialize)]
#[derive(Debug, Clone, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub struct LayerSearch {
    /// Axis along which to search for layers, expressed as the integer coordinates
    /// of a lattice point found in that direction from the origin.
    ///
    /// (...`rsp2` technically only currently supports `[1, 0, 0]`, `[0, 1, 0]`,
    /// and `[0, 0, 1]`, but implementing support for arbitrary integer vectors
    /// is *possible* if somebody needs it...)
    pub normal: [i32; 3],

    /// The cutoff distance that decides whether two atoms belong to the same layer;
    /// if and only if the shortest distance between them (projected onto the normal)
    /// exceeds this value, they belong to separate layers.
    pub threshold: f64,

    /// Expected number of layers, for a sanity check.
    /// (rsp2 will fail if this is provided and does not match the count found)
    #[serde(default)]
    pub count: Option<u32>,
}
derive_yaml_read!{LayerSearch}

#[derive(Serialize, Deserialize)]
#[derive(Debug, Clone, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub struct EnergyPlotSettings {
    #[serde(default)]
    pub threading: Threading,
    pub xlim: [f64; 2],
    pub ylim: [f64; 2],
    pub dim: [usize; 2],
    pub ev_indices: EnergyPlotEvIndices,
    /// Defines scale of xlim/ylim.
    pub normalization: NormalizationMode,
    //pub phonons: Phonons,

    pub potential: PotentialKind,
}
derive_yaml_read!{EnergyPlotSettings}

#[derive(Serialize, Deserialize)]
#[derive(Debug, Clone, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum EnergyPlotEvIndices {
    Shear,
    These(usize, usize),
}

#[derive(Serialize, Deserialize)]
#[derive(Debug, Clone, PartialEq)]
pub enum PotentialKind {
    #[serde(rename = "rebo")] Rebo,
    #[serde(rename = "airebo")] Airebo(PotentialAirebo),
    #[serde(rename = "kc-z")] KolmogorovCrespiZ(PotentialKolmogorovCrespiZ),
    #[serde(rename = "kc-z-new")] KolmogorovCrespiZNew(PotentialKolmogorovCrespiZNew),
    /// V = 0
    #[serde(rename = "test-func-zero")] TestZero,
    /// Arranges atoms into a chain along the first lattice vector.
    #[serde(rename = "test-func-chainify")] TestChainify,
}

#[derive(Serialize, Deserialize)]
#[derive(Debug, Clone, PartialEq, Default)]
#[serde(rename_all = "kebab-case")]
pub struct PotentialAirebo {
    /// Cutoff radius (x3.4A)
    pub lj_sigma: Option<f64>,
    // (I'm too lazy to make an ADT for this)
    pub lj_enabled: Option<bool>,
    pub torsion_enabled: Option<bool>,
}

#[derive(Serialize, Deserialize)]
#[derive(Debug, Clone, PartialEq, Default)]
#[serde(rename_all = "kebab-case")]
pub struct PotentialKolmogorovCrespiZ {
    // NOTE: defaults are not here because they are defined in rsp2_tasks,
    //       which depends on this crate
    /// Cutoff radius (Angstrom?)
    pub cutoff: Option<f64>,
    /// Separations larger than this are regarded as vacuum and do not interact. (Angstrom)
    pub max_layer_sep: Option<f64>,
}

#[derive(Serialize, Deserialize)]
#[derive(Debug, Clone, PartialEq, Default)]
#[serde(rename_all = "kebab-case")]
pub struct PotentialKolmogorovCrespiZNew {
    // NOTE: defaults are not here because they are defined in rsp2_tasks,
    //       which depends on this crate
    /// Cutoff radius (Angstrom?)
    #[serde(rename = "cutoff")]
    pub cutoff_begin: Option<f64>,
}

#[derive(Serialize, Deserialize)]
#[derive(Debug, Clone, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum EigenvectorChase {
    OneByOne,
    #[serde(rename = "cg")]
    Acgsd(Acgsd),
}

#[derive(Serialize, Deserialize)]
#[derive(Debug, Clone, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub struct Phonons {
    pub symmetry_tolerance: f64,
    pub displacement_distance: f64,

    #[serde(default = "_phonons__eigensolver")]
    pub eigensolver: PhononEigenSolver,

    /// Supercell used for force constants.
    ///
    /// Ideally, this should be large enough for the following to be true:
    ///
    /// * Given an atom with index `p` in the primitive cell...
    /// * ... and an atom with index `s` in the supercell...
    /// * ... `p` must interact with at most one image of `s` under the superlattice.
    ///
    /// The primary role of the supercell is to help ensure that multiple, distinct force
    /// terms are computed when a primitive atom `p` interacts with multiple images of a
    /// primitive atom `q` under the primitive lattice. (each image will have a different
    /// phase factor in the dynamical matrix at nonzero `Q` points, and therefore must be
    /// individually accounted for in the force constants)
    ///
    /// Strictly speaking, no supercell should be required for computing the dynamical
    /// matrix at Gamma, even for small primitive cells. (If one is required to get the
    /// right eigensolutions at gamma, it might indicate a bug in rsp2's potentials)
    pub supercell: SupercellSpec,
}
fn _phonons__eigensolver() -> PhononEigenSolver { PhononEigenSolver::Phonopy { save_bands: false } }

#[derive(Serialize, Deserialize)]
#[derive(Debug, Clone, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum PhononEigenSolver {
    #[serde(rename_all = "kebab-case")]
    Phonopy {
        /// Save the directory from the last phonopy computation,
        /// which may contain incomprehensibly large files.
        #[serde(default = "_phonon_eigen_solver__phonopy__save_bands")]
        save_bands: bool,
    },
    #[serde(rename_all = "kebab-case")]
    Sparse {
        /// Solve for up to this many solutions when looking for imaginary modes.
        ///
        /// This will be clipped to the greatest number that the sparse solver is capable
        /// of solving for, which is `rank - 2` (where `rank = 3 * num_sites`).
        max_count: usize,
    },
}
fn _phonon_eigen_solver__phonopy__save_bands() -> bool { false }

#[derive(Serialize, Deserialize)]
#[derive(Debug, Clone, PartialEq)]
#[serde(rename_all="kebab-case")]
pub enum SupercellSpec {
    Target([f64; 3]),
    Dim([u32; 3]),
}

#[derive(Serialize, Deserialize)]
#[derive(Debug, Clone, PartialEq)]
#[serde(rename_all="kebab-case")]
pub enum Threading {
    Lammps,
    Rayon,
    Serial,
}

#[derive(Serialize, Deserialize)]
#[derive(Debug, Clone, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub struct AcousticSearch {
    /// Known number of non-translational acoustic modes.
    #[serde(default)]
    pub expected_non_translations: Option<usize>,

    /// Displacement to use for checking changes in force along the mode.
    #[serde(default = "_acoustic_search__displacement_distance")]
    pub displacement_distance: f64,

    /// `-1 <= threshold < 1`.  How anti-parallel the changes in force
    /// have to be at small displacements along the mode for it to be classified
    /// as rotational.
    #[serde(default = "_acoustic_search__rotational_fdot_threshold")]
    pub rotational_fdot_threshold: f64,

    /// `-1 <= threshold < 1`.  How, uh, "pro-parallel" the changes in force
    /// have to be at small displacements along the mode for it to be classified
    /// as imaginary.
    #[serde(default = "_acoustic_search__imaginary_fdot_threshold")]
    pub imaginary_fdot_threshold: f64,
}
fn _acoustic_search__displacement_distance() -> f64 { 1e-5 }
fn _acoustic_search__imaginary_fdot_threshold() -> f64 { 0.80 }
fn _acoustic_search__rotational_fdot_threshold() -> f64 { 0.80 }

#[derive(Serialize, Deserialize)]
#[derive(Debug, Clone, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum NormalizationMode {
    /// Normalize the 2-norm of the 3N-component vector.
    CoordNorm,

    // These are anticipated but YAGNI for now.
    //    /// Normalize rms of the 3N-component vector to 1.
    //    CoordRms,
    //    /// Normalize mean of the 3N-component vector to 1.
    //    CoordMean,
    //    /// Normalize max value of the 3N-component vector to 1.
    //    CoordMax,

    /// Normalize rms atomic displacement distance to 1.
    AtomRms,
    /// Normalize mean atomic displacement distance to 1.
    AtomMean,
    /// Normalize max atomic displacement distance to 1.
    AtomMax,
}

#[derive(Serialize, Deserialize)]
#[derive(Debug, Clone, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub struct EvLoop {
    // Relaxation stops after all EVs are positive this many times
    #[serde(default = "_ev_loop__min_positive_iter")]
    pub min_positive_iter: u32,
    #[serde(default = "_ev_loop__max_iter")]
    pub max_iter: u32,
    #[serde(default = "_ev_loop__fail")]
    pub fail: bool,
}
fn _ev_loop__min_positive_iter() -> u32 { 3 }
fn _ev_loop__max_iter() -> u32 { 15 }
fn _ev_loop__fail() -> bool { true }

#[derive(Serialize, Deserialize)]
#[derive(Debug, Clone, PartialEq)]
pub struct Masses(pub HashMap<String, f64>);

// --------------------------------------------------------

impl Default for Threading {
    fn default() -> Self { Threading::Lammps }
}

impl Default for EvLoop {
    fn default() -> Self { from_empty_mapping().unwrap() }
}

impl Default for AcousticSearch {
    fn default() -> Self { from_empty_mapping().unwrap() }
}

#[test]
fn test_defaults()
{
    // NOTE: This simply checks that `from_empty_mapping` can succeed
    //       for each type that uses it.
    //       (it will fail if one of the fields does not have a default
    //        value and is not an Option type)
    let _ = Threading::default();
    let _ = EvLoop::default();
    let _ = AcousticSearch::default();
}

fn from_empty_mapping<T: for<'de> ::serde::Deserialize<'de>>() -> ::serde_yaml::Result<T> {
    use ::serde_yaml::{from_value, Value, Mapping};
    from_value(Value::Mapping(Mapping::new()))
}

// --------------------------------------------------------

mod defaults {
    // a reminder to myself:
    //
    // the serde default functions used to all be collected under here so that
    // they could be namespaced, like `self::defaults::ev_loop::max_iter`.
    // Reading the code, however, required jumping back and forth and it was
    // enormously frustrating and easy to lose your focus.
    //
    // **Keeping them next to their relevant structs is the superior choice.**
}
