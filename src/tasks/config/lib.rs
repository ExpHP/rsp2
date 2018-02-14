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
    fn from_reader<R: Read>(mut r: R) -> Result<Self, ::serde_yaml::Error>
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
    pub potential: Potential,
    pub scale_ranges: ScaleRanges,
    // Number of layers, when known in advance
    pub layers: Option<u32>,
    pub cg: Acgsd,
    pub phonons: Phonons,
    pub ev_chase: EigenvectorChase,
    pub layer_gamma_threshold: f64,
    #[serde(default)]
    pub ev_loop: EvLoop,
    #[serde(default)]
    pub tweaks: Tweaks,
}
derive_yaml_read!{Settings}

#[derive(Serialize, Deserialize)]
#[derive(Debug, Clone, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub struct ScaleRanges {
    pub parameter: ScaleRange,
    pub layer_sep: ScaleRange,
    #[serde(default)]
    pub layer_sep_style: ScaleRangesLayerSepStyle,
    /// How many times to repeat the process of relaxing all parameters.
    ///
    /// This may yield better results if one of the parameters relaxed
    /// earlier in the sequence impacts one of the ones relaxed earlier.
    #[serde(default="self::defaults::scale_ranges::repeat_count")]
    pub repeat_count: u32,
    #[serde(default="self::defaults::scale_ranges::warn")]
    pub warn: Option<f64>,
}

#[derive(Serialize, Deserialize)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum ScaleRangesLayerSepStyle {
    /// Optimize each layer separation individually. Can be costly.
    Individual,
    /// Optimize a single value shared by all layer separations.
    ///
    /// Under certain conditions, the optimum separation IS identical for
    /// all layers (e.g. generated structures where all pairs of layers
    /// look similar, and where the potential only affects adjacent layers).
    ///
    /// There are also conditions where the separation obtained from this method
    /// is "good enough" that CG can be trusted to take care of the rest.
    Uniform,
}

#[derive(Serialize, Deserialize)]
#[derive(Debug, Clone, PartialEq)]
#[serde(untagged)]
pub enum ScaleRange {
    #[serde(rename_all = "kebab-case")]
    Range {
        range: (f64, f64),
        /// A "reasonable value" that might be used while another
        ///  parameter is optimized.
        guess: Option<f64>,
    },
    Exact(f64),
}

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

    // FIXME confusingly this is placed under ["potential"]["kind"]
    //       in one type of config, but just ["potential"] here
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
#[serde(rename_all = "kebab-case")]
pub struct Potential {
    // supercell for lammps.
    // Purpose is to help eliminate boundary effects or something?
    // I forget.  Might not be necessary
    pub supercell: SupercellSpec,
    pub kind: PotentialKind,
}

#[derive(Serialize, Deserialize)]
#[derive(Debug, Clone, PartialEq)]
pub enum PotentialKind {
    #[serde(rename = "airebo")] Airebo(PotentialAirebo),
    #[serde(rename = "kc-z")] KolmogorovCrespiZ(PotentialKolmogorovCrespiZ),
}

#[derive(Serialize, Deserialize)]
#[derive(Debug, Clone, PartialEq, Default)]
#[serde(rename_all = "kebab-case")]
pub struct PotentialAirebo {
    /// Cutoff radius (x3.4A)
    pub lj_sigma: Option<f64>,
    /// Colin's scale hack
    pub lj_strength: Option<f64>,
    // (I'm too lazy to make an ADT for this)
    pub lj_enabled: Option<bool>,
    pub torsion_enabled: Option<bool>,
}

#[derive(Serialize, Deserialize)]
#[derive(Debug, Clone, PartialEq, Default)]
#[serde(rename_all = "kebab-case")]
pub struct PotentialKolmogorovCrespiZ {
    /// Cutoff radius (Angstrom?)
    pub cutoff: Option<f64>,
    /// Separations larger than this are regarded as vacuum and do not interact. (Angstrom)
    pub max_layer_sep: Option<f64>
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
pub struct Tweaks {
    #[serde(default = "self::defaults::tweaks::sparse_sets")]
    pub sparse_sets: bool,
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
    #[serde(default = "self::defaults::ev_loop::min_positive_iter")]
    pub min_positive_iter: u32,
    #[serde(default = "self::defaults::ev_loop::max_iter")]
    pub max_iter: u32,
    #[serde(default = "self::defaults::ev_loop::fail")]
    pub fail: bool,
}

// --------------------------------------------------------

impl Default for Threading {
    fn default() -> Self { Threading::Lammps }
}

impl Default for ScaleRangesLayerSepStyle {
    fn default() -> Self { ScaleRangesLayerSepStyle::Individual }
}

impl Default for EvLoop {
    fn default() -> Self { from_empty_mapping().unwrap() }
}
#[test] fn test_ev_loop_default() { let _ = EvLoop::default(); }

impl Default for Tweaks {
    fn default() -> Self { from_empty_mapping().unwrap() }
}
#[test] fn test_tweaks_default() { let _ = Tweaks::default(); }

fn from_empty_mapping<T: for<'de> ::serde::Deserialize<'de>>() -> ::serde_yaml::Result<T> {
    use ::serde_yaml::{from_value, Value, Mapping};
    from_value(Value::Mapping(Mapping::new()))
}

mod defaults {
    pub(crate) mod tweaks {
        pub(crate) fn sparse_sets() -> bool { false }
    }

    pub(crate) mod scale_ranges {
        pub(crate) fn repeat_count() -> u32 { 1 }
        pub(crate) fn warn() -> Option<f64> { Some(0.01) }
    }

    pub(crate) mod ev_loop {
        pub(crate) fn min_positive_iter() -> u32 { 3 }
        pub(crate) fn max_iter() -> u32 { 15 }
        pub(crate) fn fail() -> bool { true }
    }
}
