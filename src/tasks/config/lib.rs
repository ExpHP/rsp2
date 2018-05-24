#![allow(non_snake_case)]

// Crate where serde_yaml code for the 'tasks' crate is monomorphized,
// because this is a huge compile time sink.
//
// The functions here also make use of serde_ignored to catch typos in the config.

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

extern crate rsp2_minimize;

#[macro_use]
extern crate log;

use ::std::io::Read;
pub use ::rsp2_minimize::acgsd::Settings as Acgsd;

/// Provides an alternative to serde_yaml::from_reader where all of the
/// expensive codegen has already been performed in this crate.
pub trait YamlRead: for <'de> ::serde::Deserialize<'de> {
    fn from_reader(mut r: impl Read) -> Result<Self, ::serde_yaml::Error>
    { YamlRead::from_dyn_reader(&mut r) }

    fn from_dyn_reader(r: &mut Read) -> Result<Self, ::serde_yaml::Error> {
        // serde_ignored needs a Deserializer.
        // unlike serde_json, serde_yaml doesn't seem to expose a Deserializer that is
        // directly constructable from a Read... but it does impl Deserialize for Value.
        Self::from_value(value_from_dyn_reader(r)?)
    }

    fn from_value(value: ::serde_yaml::Value) -> Result<Self, ::serde_yaml::Error>;
}

macro_rules! derive_yaml_read {
    ($Type:ty) => {
        impl YamlRead for $Type {
            // NOTE: Moving this body into a default fn definition on the trait
            //       appears to make codegen lazy for some reason (compilation
            //       of this crate becomes suspiciously quick).
            //       Hence we generate these identical bodies in a macro.
            fn from_value(value: ::serde_yaml::Value) -> Result<$Type, ::serde_yaml::Error> {
                ::serde_ignored::deserialize(
                    value,
                    |path| warn!("Unused config item (possible typo?): {}", path),
                )
            }
        }
    };
}

derive_yaml_read!{::serde_yaml::Value}

// (this also exists solely for codegen reasons)
fn value_from_dyn_reader(r: &mut Read) -> Result<::serde_yaml::Value, ::serde_yaml::Error>
{ ::serde_yaml::from_reader(r) }

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

    /// `None` disables bond graph.
    #[serde(default)]
    pub bond_radius: Option<f64>,

    // FIXME move
    pub layer_gamma_threshold: f64,

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
//#[serde(rename_all = "kebab-case")]
pub enum Scalable {
    /// Uniformly scale one or more lattice vectors.
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
    UniformLayerSep {
        #[serde(flatten)]
        range: ScalableRange,
    },

    /// Optimize each layer separation individually. Can be costly.
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
    pub supercell: SupercellSpec,
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
