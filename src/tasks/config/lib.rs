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

use serde; // FIXME: cargo fix artifact ???
use rsp2_minimize; // FIXME: cargo fix artifact ???

#[macro_use]
extern crate log;

use std::io::Read;
use std::collections::HashMap;
use failure::Error;

/// Provides an alternative to serde_yaml::from_reader where all of the
/// expensive codegen has already been performed in this crate.
pub trait YamlRead: for <'de> serde::Deserialize<'de> {
    fn from_reader(mut r: impl Read) -> Result<Self, Error>
    { YamlRead::from_dyn_reader(&mut r) }

    fn from_dyn_reader(r: &mut dyn Read) -> Result<Self, Error> {
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
    fn __serde_ignored__from_value(value: serde_yaml::Value) -> Result<Self, Error>;
    fn __serde_yaml__from_str(s: &str) -> Result<Self, Error>;
}

macro_rules! derive_yaml_read {
    ($Type:ty) => {
        impl YamlRead for $Type {
            fn __serde_ignored__from_value(value: serde_yaml::Value) -> Result<$Type, Error> {
                serde_ignored::deserialize(
                    value,
                    |path| warn!("Unused config item (possible typo?): {}", path),
                ).map_err(Into::into)
            }

            fn __serde_yaml__from_str(s: &str) -> Result<$Type, Error> {
                serde_yaml::from_str(s)
                    .map_err(Into::into)
            }
        }
    };
}

derive_yaml_read!{::serde_yaml::Value}

/// Alias used for `Option<T>` to indicate that this field has a default which is implemented
/// outside of this module. (e.g. in the implementation of `Default` or `new` for a builder
/// somewhere)
pub type OrDefault<T> = Option<T>;
/// Alias used for `Option<T>` to indicate that omitting this field has special meaning.
pub type Nullable<T> = Option<T>;

// (this also exists solely for codegen reasons)
fn value_from_str(r: &str) -> Result<::serde_yaml::Value, Error>
{ serde_yaml::from_str(r).map_err(Into::into) }

#[derive(Serialize, Deserialize)]
#[derive(Debug, Clone, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub struct Settings {
    #[serde(default)]
    pub threading: Threading,

    pub potential: Potential,

    // (FIXME: weird name)
    #[serde(default)]
    pub scale_ranges: ScaleRanges,

    #[serde(default)]
    pub parameters: Nullable<Parameters>,

    #[serde(default)]
    pub acoustic_search: AcousticSearch,

    pub cg: Cg,

    pub phonons: Phonons,

    pub ev_chase: EigenvectorChase,

    /// `None` disables layer search.
    /// (layer_search is also ignored if layers.yaml is provided)
    #[serde(default)]
    pub layer_search: Nullable<LayerSearch>,

    /// `None` disables bond graph.
    #[serde(default)]
    pub bond_radius: Nullable<f64>,

    // FIXME move
    pub layer_gamma_threshold: f64,

    #[serde(default)]
    pub masses: Nullable<Masses>,

    #[serde(default)]
    pub ev_loop: EvLoop,

    // FIXME there should be a general lammps section
    #[serde(default)]
    pub lammps_update_style: LammpsUpdateStyle,
    #[serde(default = "_settings__lammps_processor_axis_mask")]
    pub lammps_processor_axis_mask: [bool; 3],
}
derive_yaml_read!{Settings}

fn _settings__update_large_neighbor_lists() -> bool { true }
fn _settings__lammps_processor_axis_mask() -> [bool; 3] { [true; 3] }

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
    pub warn_threshold: Nullable<f64>,

    /// Panic on violations of `warn_threshold`.
    #[serde(default="_scale_ranges__fail")]
    pub fail: bool,
}
fn _scale_ranges__repeat_count() -> u32 { 1 }
fn _scale_ranges__warn_threshold() -> Nullable<f64> { Some(0.01) }
fn _scale_ranges__fail() -> bool { false }

// Require "scalables" if "scale-ranges" is provided, but allow it to be defaulted to
// an empty list otherwise.
impl Default for ScaleRanges {
    fn default() -> Self {
        ScaleRanges {
            scalables: vec![],
            repeat_count: _scale_ranges__repeat_count(),
            warn_threshold: _scale_ranges__warn_threshold(),
            fail: _scale_ranges__fail(),
        }
    }
}

pub type Parameters = [Parameter; 3];
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Parameter {
    Param(char),
    One,
    NotPeriodic,
}

impl<'de> serde::Deserialize<'de> for Parameter {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where D: serde::Deserializer<'de>,
    {
        use serde::de::Unexpected;
        use serde::de::Error;

        struct Visitor;

        impl<'a> serde::de::Visitor<'a> for Visitor {
            type Value = Parameter;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "1, a single character, or null")
            }

            fn visit_i64<E>(self, value: i64) -> Result<Parameter, E>
            where E: Error,
            { match value {
                1 => Ok(Parameter::One),
                n => Err(Error::invalid_value(Unexpected::Signed(n), &self)),
            }}

            fn visit_str<E>(self, s: &str) -> Result<Parameter, E>
            where E: Error,
            { match s.len() {
                1 => Ok(Parameter::Param(s.chars().next().unwrap())),
                _ => Err(Error::invalid_value(Unexpected::Str(s), &self)),
            }}

            fn visit_unit<E>(self) -> Result<Parameter, E>
            where E: Error,
            { Ok(Parameter::NotPeriodic) }
        }

        deserializer.deserialize_any(Visitor)
    }
}

impl serde::Serialize for Parameter {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where S: serde::Serializer,
    { match *self {
        Parameter::One => serializer.serialize_i32(1),
        Parameter::NotPeriodic => serializer.serialize_none(),
        Parameter::Param(c) => serializer.serialize_char(c),
    }}
}

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
        guess: OrDefault<f64>,
    },
    #[serde(rename_all = "kebab-case")]
    Exact {
        value: f64,
    },
}

#[derive(Serialize, Deserialize)]
#[derive(Debug, Clone, PartialEq)]
#[serde(rename_all="kebab-case")]
pub struct Cg {
    #[serde()]
    pub stop_condition: CgStopCondition,
    #[serde(default)]
    pub flavor: CgFlavor,
    #[serde(default)]
    pub on_ls_failure: CgOnLsFailure,
    #[serde(default = "_cg__alpha_guess_first")]
    pub alpha_guess_first: f64,
    #[serde(default)]
    pub alpha_guess_max: f64,
}
// Been using these values for a while on structures of arbitrary size.
fn _cg__alpha_guess_first() -> f64 { 0.01 }
fn _cg__alpha_guess_max() -> f64 { 0.1 }

pub type CgStopCondition = rsp2_minimize::cg::StopCondition;

/// Behavior when a linesearch along the steepest descent direction fails.
/// (this is phenomenally rare for the Hager linesearch method, and when it
///  does occur it may very well be due to exceptionally good convergence,
///  rather than any sort of actual failure)
#[derive(Serialize, Deserialize)]
#[derive(Debug, Clone, PartialEq)]
#[serde(rename_all="kebab-case")]
pub enum CgOnLsFailure {
    /// Treat a second linesearch failure as a successful stop condition.
    Succeed,
    /// Succeed, but log a warning.
    Warn,
    /// Complain loudly and exit with a nonzero exit code.
    Fail,
}
impl Default for CgOnLsFailure {
    fn default() -> Self { CgOnLsFailure::Succeed }
}

#[derive(Serialize, Deserialize)]
#[derive(Debug, Clone, PartialEq)]
#[serde(rename_all="kebab-case")]
pub enum CgFlavor {
    #[serde(rename_all="kebab-case")]
    Acgsd {
        #[serde(rename="iteration-limit")] // for compatibility
        ls_iteration_limit: OrDefault<u32>,
    },
    #[serde(rename_all="kebab-case")]
    Hager {},
}
impl Default for CgFlavor {
    fn default() -> Self { CgFlavor::Hager {} }
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
    pub count: Nullable<u32>,
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

    pub potential: Potential,

    // FIXME there should be a general lammps section
    #[serde(default)]
    pub lammps_update_style: LammpsUpdateStyle,
    pub lammps_processor_axis_mask: [bool; 3],
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
#[serde(untagged)]
pub enum Potential {
    Single(PotentialKind),
    Sum(Vec<PotentialKind>),
}
derive_yaml_read!{Potential}

#[derive(Serialize, Deserialize)]
#[derive(Debug, Clone, PartialEq)]
pub enum PotentialKind {
    #[serde(rename = "rebo")] Rebo(PotentialRebo),
    #[serde(rename = "airebo")] Airebo(PotentialAirebo),
    #[serde(rename = "kc-z")] KolmogorovCrespiZ(PotentialKolmogorovCrespiZ),
    #[serde(rename = "kc-z-new")] KolmogorovCrespiZNew(PotentialKolmogorovCrespiZNew),
    #[serde(rename = "rebo-new")] ReboNew(PotentialReboNew),
    /// V = 0
    #[serde(rename = "test-func-zero")] TestZero,
    /// Arranges atoms into a chain along the first lattice vector.
    #[serde(rename = "test-func-chainify")] TestChainify,
}

#[derive(Serialize, Deserialize)]
#[derive(Debug, Clone, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub struct PotentialAirebo {
    /// Cutoff radius (x3.4A)
    pub lj_sigma: OrDefault<f64>,
    // (I'm too lazy to make an ADT for this)
    pub lj_enabled: OrDefault<bool>,
    pub torsion_enabled: OrDefault<bool>,
    pub omp: OrDefault<bool>,
}

#[derive(Serialize, Deserialize)]
#[derive(Debug, Clone, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub struct PotentialRebo {
    pub omp: OrDefault<bool>,
}

#[derive(Serialize, Deserialize)]
#[derive(Debug, Clone, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub struct PotentialKolmogorovCrespiZ {
    // NOTE: some defaults are not here because they are defined in rsp2_tasks,
    //       which depends on this crate
    #[serde(default = "_potential_kolmogorov_crespi_z__rebo")]
    pub rebo: bool,
    /// Cutoff radius (Angstrom?)
    pub cutoff: OrDefault<f64>,
    /// Separations larger than this are regarded as vacuum and do not interact. (Angstrom)
    pub max_layer_sep: OrDefault<f64>,

    /// Enable a smooth cutoff starting at `r = cutoff - cutoff_interval` and ending at
    /// `r = cutoff`.
    ///
    /// NOTE: This requires a patched lammps.
    pub cutoff_interval: Nullable<f64>,
}
fn _potential_kolmogorov_crespi_z__rebo() -> bool { true }

#[derive(Serialize, Deserialize)]
#[derive(Debug, Clone, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub struct PotentialKolmogorovCrespiZNew {
    // NOTE: defaults are not here because they are defined in rsp2_tasks,
    //       which depends on this crate
    /// Cutoff radius (Angstrom?)
    #[serde(rename = "cutoff")]
    pub cutoff_begin: OrDefault<f64>,

    /// Thickness of the "smooth cutoff" shell.
    ///
    /// NOTE: If a value of 0.0 is used, the value is offset to maintain C0 continuity.
    /// (This makes it effectively identical to LAMMPS)
    #[serde(rename = "cutoff-length")]
    pub cutoff_transition_dist: OrDefault<f64>,

    /// Skin depth for neighbor searches.  Adjusting this may wildly improve (or hurt!)
    /// performance depending on the application.
    #[serde(default = "_potential_kolmogorov_crespi_z_new__skin_depth")]
    pub skin_depth: f64,

    // FIXME: hack
    /// Perform a skin check every `n` computations (`0` = never) rather than every computation.
    ///
    /// Even though it may give slightly incorrect results, this is provided because in some cases,
    /// states speculatively observed by an algorithm (such as conjugate gradient with built-in
    /// param optimization) may have a large tendency to briefly violate the skin check.  For such
    /// purposes it is expected that the (stronger) forces from REBO will quickly discourage CG from
    /// actually selecting states with drastically modified neighbor lists, and that the initial
    /// bond list should remain sufficient for all states actually visited by CG.
    ///
    /// Because various parts of the code may call the potential any arbitrary number of times,
    /// the frequency here does not necessarily correspond to anything meaningful.
    #[serde(default = "_potential_kolmogorov_crespi_z_new__skin_check_frequency")]
    pub skin_check_frequency: u64,
}
fn _potential_kolmogorov_crespi_z_new__skin_depth() -> f64 { 1.0 }
fn _potential_kolmogorov_crespi_z_new__skin_check_frequency() -> u64 { 1 }

#[derive(Serialize, Deserialize)]
#[derive(Debug, Clone, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub struct PotentialReboNew {
    /// "brenner" or "lammps"
    pub params: PotentialReboNewParams,
}

#[derive(Serialize, Deserialize)]
#[derive(Debug, Clone, Copy, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum PotentialReboNewParams {
    Brenner,
    Lammps,
    LammpsFavata,
}

#[derive(Serialize, Deserialize)]
#[derive(Debug, Clone, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum EigenvectorChase {
    OneByOne,
    Cg(Cg),
}

#[derive(Serialize, Deserialize)]
#[derive(Debug, Clone, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub struct Phonons {
    pub symmetry_tolerance: f64,
    pub displacement_distance: f64,

    #[serde(default = "_phonons__disp_finder")]
    pub disp_finder: PhononDispFinder,

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
fn _phonons__eigensolver() -> PhononEigenSolver { PhononEigenSolver::Phonopy { save_bands: _phonon_eigen_solver__phonopy__save_bands() } }
fn _phonons__disp_finder() -> PhononDispFinder { PhononDispFinder::Phonopy { diag: _phonon_disp_finder__phonopy__diag() } }

#[derive(Serialize, Deserialize)]
#[derive(Debug, Clone, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum PhononEigenSolver {
    /// Diagonalize the dynamical matrix by having phonopy compute `band.hdf5`,
    /// computing all eigensolutions.
    ///
    /// **Cannot be used with `disp-finder: rsp2`.**
    #[serde(rename_all = "kebab-case")]
    Phonopy {
        /// Save the directory from the last phonopy computation,
        /// which may contain incomprehensibly large files.
        #[serde(default = "_phonon_eigen_solver__phonopy__save_bands")]
        save_bands: bool,
    },

    /// Diagonalize the dynamical matrix using ARPACK through Scipy or Numpy
    /// (dense). Aliased to 'sparse' for backwards compatibility.
    #[serde(rename_all = "kebab-case")]
    #[serde(alias = "sparse")]
    Rsp2 {
        #[serde(default = "_phonon_eigen_solver__rsp2__shift_invert_attempts")]
        shift_invert_attempts: u32,
        #[serde(default = "_phonon_eigen_solver__rsp2__dense")]
        dense: bool,
    },
}
fn _phonon_eigen_solver__phonopy__save_bands() -> bool { false }
fn _phonon_eigen_solver__rsp2__shift_invert_attempts() -> u32 { 4 }
fn _phonon_eigen_solver__rsp2__dense() -> bool { false }

#[derive(Serialize, Deserialize)]
#[derive(Debug, Clone, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum PhononDispFinder {
    /// Use `phonopy --disp` to compute the displacements.
    ///
    /// Phonopy is battle-tested and fairly clever at choosing displacements, but has
    /// difficulty handling larger structures (>800 atoms).
    #[serde(rename_all = "kebab-case")]
    Phonopy {
        /// Phonopy's DIAG option. This makes it select fewer displacements.
        #[serde(default = "_phonon_disp_finder__phonopy__diag")]
        diag: bool,
    },

    /// Use built-in methods to compute the displacements.
    ///
    /// This isn't very clever, and is maybe comparable to phonopy's `DIAG = .FALSE.`
    ///
    /// **Cannot be used with `eigensolver: phonopy`.**
    Rsp2 {
        #[serde(default = "_phonon_disp_finder__rsp2__directions")]
        directions: PhononDispFinderRsp2Directions,
    }
}
fn _phonon_disp_finder__phonopy__diag() -> bool { true }
fn _phonon_disp_finder__rsp2__directions() -> PhononDispFinderRsp2Directions { PhononDispFinderRsp2Directions::Diag }

#[derive(Serialize, Deserialize)]
#[derive(Debug, Clone, PartialEq)]
#[serde(rename_all="kebab-case")]
pub enum PhononDispFinderRsp2Directions {
    /// Comparable to phonopy with `DIAG = .FALSE.`
    Axial,
    /// Comparable to phonopy with `DIAG = .TRUE.`
    Diag,
    /// (Experimental) Diagonal displacements with fractional coords up to 2.
    #[serde(rename = "diag-2")]
    Diag2,
    /// (Debug) Try all three of them and report how many they find, in an attempt
    /// to answer the question "is diag-2 worthless?"
    Survey,
}

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
#[serde(rename_all="kebab-case")]
pub enum LammpsUpdateStyle {
    /// Use `run 0` to notify LAMMPS of updates.
    Safe,
    /// (Experimental) Use `run 1 pre no post no` to notify LAMMPS of updates.
    ///
    /// To make this a bit safer, the delay on neighbor update checks is removed.
    /// (However, if an atom is not at the same image where LAMMPS would prefer to find it,
    /// then the optimization is defeated, and neighbor lists will end up being built
    /// every step...)
    #[serde(rename_all="kebab-case")]
    Fast {
        sync_positions_every: u32,
    },
    /// (Debug) Use a custom `run _ pre _ post _` to notify LAMMPS of updates.
    #[serde(rename_all="kebab-case")]
    Run {
        n: u32, pre: bool, post: bool, sync_positions_every: u32,
    },
}
impl Default for LammpsUpdateStyle {
    fn default() -> Self { LammpsUpdateStyle::Safe }
}

#[derive(Serialize, Deserialize)]
#[derive(Debug, Clone, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub struct AcousticSearch {
    /// Known number of non-translational acoustic modes.
    #[serde(default)]
    pub expected_non_translations: Nullable<usize>,

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

fn from_empty_mapping<T: for<'de> serde::Deserialize<'de>>() -> serde_yaml::Result<T> {
    use serde_yaml::{from_value, Value, Mapping};
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
